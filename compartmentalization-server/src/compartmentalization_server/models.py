"""
Shared tiny-transformer backbone used by every experiment script.

The architecture is the same one the TypeScript reference uses
(examples/toy-compartmentalization/src/lib/model.ts): a pre-norm causal
transformer with configurable num_heads / head_dim, configurable positional
encoding (RoPE or learned), a ReLU MLP, and a tied-vocab LM head with no
bias. Keeping it in one module means bio3 and mess3 (and any future port
like xor / brackets / rnn) can all share the same code instead of
rewriting the forward pass per experiment.

Two knobs matter enough to be worth calling out:

  head_dim vs embed_dim
    MESS3 deliberately has an 8-dim attention bottleneck inside a 64-dim
    residual stream (head_dim=8, num_heads=1, attn_dim=8), which is the
    paper's architectural choice. BIO uses the standard
    head_dim=embed_dim/num_heads = 16 (no bottleneck). Both are supported
    by passing head_dim explicitly or leaving it None for the standard
    divide.

  pos_encoding
    "rope" rotates Q and K before attention and keeps V untouched;
    positions are implicit. "learned" adds a learned wpe embedding
    elementwise to the token embedding; the attention itself doesn't see
    positions directly. MESS3 uses RoPE; BIO defaults to learned.

`forward(tokens, return_residuals=False)` returns logits, optionally
paired with the per-layer residual stream snapshots for probing work
(MESS3's linear belief probe uses the final layer residual; BIO's
cross-compartment cosine similarity uses the MIDDLE layer residual).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TransformerConfig:
    """All the knobs the experiments actually tune.

    `head_dim=None` means "use embed_dim // num_heads" (the standard
    multi-head sizing). Explicit head_dim is only used by MESS3 for its
    deliberate 8-dim attention bottleneck.
    """

    vocab_size: int
    seq_len: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    head_dim: int | None = None
    pos_encoding: Literal["rope", "learned"] = "rope"
    rope_base: float = 10000.0

    @property
    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) not divisible by num_heads "
                f"({self.num_heads}); pass head_dim explicitly",
            )
        return self.embed_dim // self.num_heads

    @property
    def attn_dim(self) -> int:
        return self.num_heads * self.resolved_head_dim


# ──────────────────────────────────────────────────────────────────────────
# RoPE helpers
# ──────────────────────────────────────────────────────────────────────────


def precompute_rope(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE cos/sin tables of shape [seq_len, head_dim/2].

    For each position m and pair index i, theta = m / base^(2i/head_dim).
    Pairs of feature dims rotate together; we store cos and sin per pair
    so the rotation is two elementwise multiplies + one reshuffle.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [seq, half]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to a [batch, heads, seq, head_dim] tensor.

    The rotation pairs feature dimensions (0,1), (2,3), ... and rotates
    each pair by the position-dependent angle. Equivalent to multiplying
    each pair by a 2x2 rotation matrix:
        [x0']   [cos -sin] [x0]
        [x1'] = [sin  cos] [x1]
    """
    x1 = x[..., 0::2]  # even-indexed pairs
    x2 = x[..., 1::2]  # odd-indexed pairs
    cos_b = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, half]
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos_b - x2 * sin_b
    rotated_x2 = x1 * sin_b + x2 * cos_b
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


# ──────────────────────────────────────────────────────────────────────────
# Modules
# ──────────────────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Pre-norm multi-head causal self-attention.

    QKV projection goes embed_dim → 3*attn_dim; the out projection goes
    attn_dim → embed_dim. When head_dim*num_heads < embed_dim, attention
    operates in a bottleneck subspace (MESS3's setup). PyTorch's
    scaled_dot_product_attention does the causal mask + softmax + @V
    fused.

    When pos_encoding="rope", Q and K get rotated before attention and
    the module keeps RoPE tables as non-persistent buffers. When "learned",
    RoPE is off and positions come in via the token-level wpe addition in
    Transformer.forward.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.resolved_head_dim
        self.attn_dim = cfg.attn_dim
        self.use_rope = cfg.pos_encoding == "rope"

        self.qkv = nn.Linear(cfg.embed_dim, 3 * self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, cfg.embed_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if self.use_rope:
            cos, sin = precompute_rope(cfg.seq_len, self.head_dim, cfg.rope_base)
            # persistent=False because they're deterministic from config and
            # shouldn't be written into checkpoints.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*attn_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        # [B, T, H, D] → [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.attn_dim)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Pre-norm block: LN → attn → +residual → LN → MLP (ReLU) → +residual.

    ReLU, not GELU, because that's what the belief-state-geometry paper
    uses. Keeping it identical to the JS reference so convergence is
    bit-for-bit comparable.
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.fc1 = nn.Linear(cfg.embed_dim, cfg.mlp_dim)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x


class Transformer(nn.Module):
    """Tiny causal transformer used by every experiment.

    forward(tokens, return_residuals=False) returns:
      - logits of shape [B, T, vocab_size] when return_residuals is False
      - (logits, residuals) where residuals is a list of [B, T, embed_dim]
        tensors (one per layer) when return_residuals is True

    The residual list is the residual stream AFTER each block's MLP add,
    before the final LN. Use residuals[-1] for the last layer (MESS3's
    belief probe) or residuals[len//2] for the middle layer (BIO's
    cross-compartment cosine similarity).
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        if cfg.pos_encoding == "learned":
            self.wpe: nn.Embedding | None = nn.Embedding(cfg.seq_len, cfg.embed_dim)
        else:
            self.wpe = None
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_residuals: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        B, T = tokens.shape
        x = self.wte(tokens)
        if self.wpe is not None:
            positions = torch.arange(T, device=tokens.device)
            x = x + self.wpe(positions).unsqueeze(0)

        residuals: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            if return_residuals:
                residuals.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_residuals:
            return logits, residuals
        return logits
