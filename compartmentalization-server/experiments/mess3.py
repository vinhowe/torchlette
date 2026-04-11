"""
MESS3 — tiny causal transformer on a 3-state HMM.

Faithful PyTorch port of the JS reference at
examples/toy-compartmentalization/src/lib/{model,data}.ts. The architecture
matches the paper "Transformers Represent Belief State Geometry in their
Residual Stream" (Shai, Riechers et al., 2024): a 4-layer transformer with
1 head, 64-dim residual stream, an 8-dim attention bottleneck (head_dim=8,
deliberately less than embed_dim=64 to encourage compartmentalization), a
256-dim ReLU MLP, and RoPE positional encoding.

The data is a 3-state HMM emitting tokens from {A, B, C}; transition
matrices come from the paper, optionally interpolated toward uniform via
self_loop. With self_loop=0.765, the matrices reproduce the paper exactly.

vocab_size = VOCAB_SIZE_DATA * n_comp + 1 mirrors the JS convention. The
+1 leaves room for a task/separator token; for n_comp=1 and single-task
training it's an unused embedding row that the model never sees.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from compartmentalization_server.api import Experiment, register

# ──────────────────────────────────────────────────────────────────────────
# HMM data generator
# ──────────────────────────────────────────────────────────────────────────

VOCAB_SIZE_DATA = 3  # A, B, C
NUM_STATES = 3

# Paper's joint transition/emission matrices T^(x)_{ij} = Pr(emit x, go to state j | in state i)
_PAPER_T_A = torch.tensor(
    [
        [0.765, 0.00375, 0.00375],
        [0.0425, 0.0675, 0.00375],
        [0.0425, 0.00375, 0.0675],
    ]
)
_PAPER_T_B = torch.tensor(
    [
        [0.0675, 0.0425, 0.00375],
        [0.00375, 0.765, 0.00375],
        [0.00375, 0.0425, 0.0675],
    ]
)
_PAPER_T_C = torch.tensor(
    [
        [0.0675, 0.00375, 0.0425],
        [0.00375, 0.0675, 0.0425],
        [0.00375, 0.00375, 0.765],
    ]
)
_PAPER_T = torch.stack([_PAPER_T_A, _PAPER_T_B, _PAPER_T_C], dim=0)  # [3, 3, 3]


def build_transition_matrices(self_loop: float) -> torch.Tensor:
    """Return T of shape [3 emit, 3 from-state, 3 to-state].

    self_loop=0.765 reproduces the paper exactly. Lower values blend the
    matrices toward a uniform 1/9 = ~0.111 per cell, producing faster
    mixing and a coarser fractal in belief-space.
    """
    if abs(self_loop - 0.765) < 1e-3:
        return _PAPER_T.clone()
    t = self_loop / 0.765
    uniform = 1.0 / 9.0
    return uniform + t * (_PAPER_T - uniform)


def stationary_dist(transitions: torch.Tensor, iters: int = 500) -> torch.Tensor:
    """Power-iterate the row-stochastic state-to-state matrix to convergence.

    The state-to-state matrix is sum over emissions of T[x, i, j], which
    is row-stochastic (rows sum to 1). 500 iters is overkill for a 3x3
    matrix but matches the JS reference.
    """
    state_to_state = transitions.sum(dim=0)  # [from, to]
    pi = torch.full((NUM_STATES,), 1.0 / NUM_STATES)
    for _ in range(iters):
        pi = pi @ state_to_state
    return pi


class HMMSampler:
    """Vectorized sampler for the MESS3 HMM. One instance per experiment."""

    def __init__(self, self_loop: float, device: torch.device) -> None:
        self.device = device
        T = build_transition_matrices(self_loop).to(device)  # [3, 3, 3]
        # Flatten emit+next-state into a single 9-way categorical per
        # current state: row[i, k] where k = emit*3 + next_state.
        # Probabilities sum to 1 along k (because the paper matrices are
        # joint over (emit, next_state) given current state).
        self.flat_probs = T.permute(1, 0, 2).reshape(NUM_STATES, 9)  # [from, 9]
        self.pi = stationary_dist(T.cpu()).to(device)  # initial state distribution

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        """Sample a [batch, seq_len] tensor of int64 token ids.

        Walks the HMM step-by-step. The whole batch advances in parallel:
        at each timestep we draw `batch` independent (emit, next_state)
        samples from the categorical defined by each row's current state.
        Total work is O(seq_len * batch * 9) which is negligible compared
        to the model.
        """
        tokens = torch.empty((batch, seq_len), dtype=torch.long, device=self.device)
        # Sample initial states from stationary distribution.
        state = torch.multinomial(self.pi, batch, replacement=True)  # [batch]
        for t in range(seq_len):
            row_probs = self.flat_probs[state]  # [batch, 9]
            choice = torch.multinomial(row_probs, 1).squeeze(-1)  # [batch]
            emit = choice // NUM_STATES
            next_state = choice % NUM_STATES
            tokens[:, t] = emit
            state = next_state
        return tokens


# ──────────────────────────────────────────────────────────────────────────
# Model: tiny causal transformer with RoPE
# ──────────────────────────────────────────────────────────────────────────


def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard RoPE cos/sin tables, returned as [seq_len, head_dim/2].

    For each position m and pair index i, theta = m / base^(2i/head_dim).
    Pairs of feature dims rotate together; we store cos and sin per pair
    so the rotation is two elementwise multiplies + one swap.
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
        [x0]   [cos -sin] [x0]
        [x1] = [sin  cos] [x1]
    Implemented as elementwise muls so it fuses cleanly.
    """
    # x has shape [batch, heads, seq, head_dim]
    x1 = x[..., 0::2]  # even-indexed pairs
    x2 = x[..., 1::2]  # odd-indexed pairs
    # cos/sin are [seq, half]; broadcast to [1, 1, seq, half]
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    rotated_x1 = x1 * cos_b - x2 * sin_b
    rotated_x2 = x1 * sin_b + x2 * cos_b
    # Re-interleave the pairs back into the original layout.
    out = torch.empty_like(x)
    out[..., 0::2] = rotated_x1
    out[..., 1::2] = rotated_x2
    return out


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with RoPE on Q and K.

    head_dim is intentionally smaller than embed_dim, so the attention
    space is a low-dim bottleneck. The QKV projection goes
    embed_dim → 3*head_dim and the output projection goes
    head_dim → embed_dim. PyTorch's scaled_dot_product_attention does the
    causal mask + softmax + matmul fused.
    """

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, seq_len: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.qkv = nn.Linear(embed_dim, 3 * self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, embed_dim)
        cos, sin = precompute_rope(seq_len, head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*attn_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # RoPE on Q and K (V is not rotated — RoPE is a position-aware
        # similarity trick for attention, not a value transform).
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Back to [B, T, attn_dim]
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.attn_dim)
        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → attn → +residual → LN → MLP → +residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        seq_len: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, head_dim, seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.relu(self.fc1(self.ln2(x))))
        return x


class Mess3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int = 64,
        num_heads: int = 1,
        head_dim: int = 8,
        num_layers: int = 4,
        mlp_dim: int = 256,
    ) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, head_dim, mlp_dim, seq_len)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.wte(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B, T, vocab]


# ──────────────────────────────────────────────────────────────────────────
# Experiment script
# ──────────────────────────────────────────────────────────────────────────


@register(
    "mess3",
    description="Tiny causal transformer (4L 1H 64d, head_dim=8, RoPE) on the MESS3 HMM",
    params={
        # ── live params (editable mid-run via slider) ──
        "lr": {
            "type": "number",
            "default": 0.01,
            "min": 1e-5,
            "max": 1.0,
            "scale": "log",
            "live": True,
            "description": "Adam learning rate",
        },
        "batch_size": {
            "type": "number",
            "default": 64,
            "min": 1,
            "max": 2048,
            "scale": "linear",
            "live": True,
            "description": "Sequences per training step",
        },
        # ── structural params (fixed at creation) ──
        "seq_len": {
            "type": "number",
            "default": 10,
            "min": 1,
            "max": 256,
            "scale": "linear",
            "live": False,
            "description": "Context length (also affects RoPE table size)",
        },
        "n_comp": {
            "type": "number",
            "default": 1,
            "min": 1,
            "max": 16,
            "scale": "linear",
            "live": False,
            "description": "Number of parallel compartments (for compartmentalization studies)",
        },
        "self_loop": {
            "type": "number",
            "default": 0.765,
            "min": 0.0,
            "max": 1.0,
            "scale": "linear",
            "live": False,
            "description": "HMM self-loop probability (0.765 = paper exact)",
        },
        "embed_dim": {
            "type": "number",
            "default": 64,
            "min": 8,
            "max": 1024,
            "scale": "linear",
            "live": False,
            "description": "Residual stream width",
        },
        "num_layers": {
            "type": "number",
            "default": 4,
            "min": 1,
            "max": 24,
            "scale": "linear",
            "live": False,
            "description": "Number of transformer blocks",
        },
        "head_dim": {
            "type": "number",
            "default": 8,
            "min": 2,
            "max": 128,
            "scale": "linear",
            "live": False,
            "description": "Attention bottleneck dimension (must be even)",
        },
        "mlp_dim": {
            "type": "number",
            "default": 256,
            "min": 8,
            "max": 4096,
            "scale": "linear",
            "live": False,
            "description": "MLP hidden dimension",
        },
    },
)
class Mess3(Experiment):
    def setup(self) -> None:
        seq_len = int(self.params["seq_len"])
        n_comp = int(self.params["n_comp"])
        embed_dim = int(self.params["embed_dim"])
        num_layers = int(self.params["num_layers"])
        head_dim = int(self.params["head_dim"])
        mlp_dim = int(self.params["mlp_dim"])
        self_loop = float(self.params["self_loop"])

        # +1 mirrors the JS convention; harmless extra embedding row.
        self.vocab_size = VOCAB_SIZE_DATA * n_comp + 1
        self.seq_len = seq_len

        self.model = Mess3Model(
            vocab_size=self.vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=1,
            head_dim=head_dim,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["lr"]),
        )
        self.sampler = HMMSampler(self_loop, self.device)
        self.criterion = nn.CrossEntropyLoss()

    def step(self) -> dict[str, float]:
        # Re-read live params each step
        lr = float(self.params["lr"])
        batch_size = int(self.params["batch_size"])
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        tokens = self.sampler.sample(batch_size, self.seq_len)  # [B, T]
        # Predict next-token: inputs are tokens[:, :-1], targets are tokens[:, 1:]
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits = self.model(inputs)  # [B, T-1, vocab]
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.detach().item())}

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        # The framework loads the blob on CPU so RNG ByteTensors stay on
        # CPU as required by torch.set_rng_state. As a side effect, every
        # other tensor in the blob also arrives on CPU. model.load_state_dict
        # handles its own placement (it copies into existing parameters
        # which retain their device), but optimizer.load_state_dict
        # blindly copies tensors as-is, leaving Adam's (m, v) on CPU and
        # the optimizer broken on the next step. Walk the state and move
        # any tensors to the experiment's device.
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
