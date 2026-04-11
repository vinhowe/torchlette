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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


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


# ──────────────────────────────────────────────────────────────────────────
# Sharpness (λ_max of the Hessian via power iteration)
# ──────────────────────────────────────────────────────────────────────────


def hessian_lambda_max(
    loss_fn: Callable[[], torch.Tensor],
    params: list[torch.Tensor],
    num_iters: int = 15,
    tol: float = 1e-3,
) -> float:
    """Estimate the largest eigenvalue of the Hessian ∇²L(w) via power
    iteration with Hessian-vector products (HVPs).

    This is the canonical "sharpness" measurement from the flat-vs-sharp
    minima literature (Keskar et al. 2017, Cohen et al. 2022 "Edge of
    Stability"): it tells you the curvature of the loss landscape in
    the sharpest direction at the current parameters. Large = sharp
    minimum, small = flat. NOT the same thing as gradient norm, which
    measures distance from a critical point.

    ## How it works

    Power iteration on the Hessian:
        v_{k+1} = Hv_k / ||Hv_k||
        λ_k+1 = v_{k+1}ᵀ H v_{k+1}  (Rayleigh quotient)
    converges to the eigenvector of the largest-magnitude eigenvalue.
    The Rayleigh quotient converges to that eigenvalue itself.

    We never form H explicitly (O(N²) memory for N params — infeasible
    for anything bigger than a few thousand params). Instead we
    compute Hv on the fly using the double-backward trick:
        g = ∇L        (first backward, with create_graph=True)
        Hv = ∇(g·v)   (second backward, differentiates through g)
    PyTorch's autograd does this via `torch.autograd.grad` with
    `create_graph=True` on the first call so the second call can
    backprop through it.

    ## Cost

    Each iteration is one forward pass + two backward passes (the
    first with create_graph=True). Roughly 3x the cost of a plain
    training step. With num_iters=15, total cost per call is ~45
    training-step-equivalents. Run it every ~50 training steps and the
    overhead is ~1% overall.

    ## Caveats

      - Power iteration finds the largest-*magnitude* eigenvalue. Near
        a minimum the Hessian is PSD so this is λ_max. Near a saddle
        it could be the most-negative eigenvalue (in which case the
        sign is negative). In practice for a converging training run
        you're near a minimum most of the time.

      - We force PyTorch's MATH SDPA backend during the forward pass.
        The fused "efficient" / "flash" attention kernels have fast
        first-derivatives but NO SECOND DERIVATIVE implementation —
        calling autograd.grad a second time through them raises
        "derivative for aten::_scaled_dot_product_efficient_attention_backward
        is not implemented". The math backend is slower but supports
        double-backward, which HVPs need. This wrapper makes the
        choice transparent to callers.

      - Can be called inside a `torch.no_grad()` scope (e.g. from an
        eval method): this helper opens its own `torch.enable_grad`
        block, which overrides the outer no_grad.

      - If two top eigenvalues are close in magnitude, power iteration
        converges slowly. In practice 15 iterations gives ~3 decimal
        places of accuracy for transformers; bump `num_iters` if plots
        look noisy.

    ## Parameters

    loss_fn : a zero-arg callable that computes and returns a fresh
        scalar loss tensor. Called once per iteration so each iteration
        sees a fresh graph — the second-derivative computation consumes
        the graph each time, so it can't be reused across iterations.
        Typical implementation: sample a minibatch, do a forward pass,
        return cross_entropy(logits, targets).
    params : list of parameter tensors to differentiate with respect to.
        Typically `[p for p in model.parameters() if p.requires_grad]`.
    num_iters : max power iterations. 15 is usually enough; 25-50 if
        you need more accuracy or if the top two eigenvalues are close.
    tol : relative tolerance for early exit. If two consecutive estimates
        agree to within this, stop early.

    ## Returns

    Python float: the estimated top eigenvalue. Returns 0.0 on numerical
    degeneracy (e.g. Hv is effectively zero — shouldn't happen for a
    real model).
    """
    # Initial random direction, normalized to unit length.
    v = [torch.randn_like(p) for p in params]
    total = torch.sqrt(sum((vi * vi).sum() for vi in v))
    if float(total.item()) == 0.0:
        return 0.0
    v = [vi / total for vi in v]

    eigenvalue = 0.0
    for it in range(num_iters):
        # Clear any stale gradients from the training loop so the
        # autograd.grad call doesn't accidentally inherit them.
        for p in params:
            if p.grad is not None:
                p.grad = None

        # SDPBackend.MATH forces F.scaled_dot_product_attention to use
        # the math implementation, which is the only one that
        # implements the second derivative. The fused "efficient" /
        # "flash" kernels raise on double-backward. Scoped to just the
        # forward pass since that's where the kernel selection happens.
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            loss = loss_fn()
            # First derivative; create_graph=True so we can
            # differentiate through it.
            grads = torch.autograd.grad(
                loss, params, create_graph=True, retain_graph=True,
            )
            # g · v : a scalar built from the current grad graph. This
            # is the thing whose gradient w.r.t. params is H·v.
            gv = sum((g * vi).sum() for g, vi in zip(grads, v))
            # H · v = ∇(g · v). Detach each component so we drop the
            # graph before the next iteration.
            Hv_raw = torch.autograd.grad(gv, params, retain_graph=False)
            Hv = [h.detach() for h in Hv_raw]

        # Rayleigh quotient: vᵀHv. Since v is unit-norm this is the
        # eigenvalue estimate for the current iteration.
        new_eigenvalue = float(
            sum((hi * vi).sum() for hi, vi in zip(Hv, v)).item()
        )

        # Normalize Hv to get the next iteration's direction.
        norm_sq = sum((hi * hi).sum() for hi in Hv)
        norm_val = float(norm_sq.sqrt().item())
        if norm_val < 1e-12:
            # Hessian times this direction is ~zero; either we're at
            # a degenerate point or numerically unstable. Return what
            # we have.
            return eigenvalue
        v = [hi / norm_val for hi in Hv]

        # Early exit: if consecutive estimates are close enough, we're
        # converged. The abs(...) / max(...) form avoids a divide-by-zero
        # when eigenvalue is small and handles both signs.
        if it > 0:
            denom = max(abs(new_eigenvalue), 1e-6)
            if abs(new_eigenvalue - eigenvalue) / denom < tol:
                eigenvalue = new_eigenvalue
                break
        eigenvalue = new_eigenvalue

    return eigenvalue
