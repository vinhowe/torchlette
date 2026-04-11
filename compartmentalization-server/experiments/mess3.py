"""
MESS3 — tiny causal transformer on a 3-state HMM.

Architecture: 4 layers, 1 head, 64-dim residual, head_dim=8 attention
bottleneck, RoPE. Matches examples/toy-compartmentalization/src/lib/model.ts
and follows "Transformers Represent Belief State Geometry in their
Residual Stream" (Shai, Riechers et al., 2024).

Metrics reported (in order of wall-clock cost):

  loss            cross-entropy on every training step
  probe_r2        every EVAL_INTERVAL steps — R² of a linear map from the
                  last-layer residual stream to the ground-truth HMM
                  belief state at each position. High R² means the model
                  has learned to represent its posterior over hidden
                  states inside the residual stream.
  probe_r2_cN     when n_comp > 1, R² on compartment N using the probe
                  fit on compartment 0. Measures whether the same
                  linear map decodes beliefs across compartments
                  (i.e. unified representation).
  cos_sim         cross-compartment cosine similarity of the middle-layer
                  residual stream at the last non-final position, averaged
                  over the eval batch. High = compartments share
                  representation, low = they're distinct.

For n_comp=1, probe_r2_cN and cos_sim are skipped (both require multiple
compartments to be meaningful).

Checkpoint semantics: the probe weights are rebuilt from scratch on every
eval, so we don't need to save them. Model + optimizer + step count are
all that matter for resume.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from compartmentalization_server.api import Experiment, register
from compartmentalization_server.models import (
    Transformer,
    TransformerConfig,
    hessian_lambda_max,
)

# ──────────────────────────────────────────────────────────────────────────
# MESS3 HMM data
# ──────────────────────────────────────────────────────────────────────────

VOCAB_SIZE_DATA = 3  # A, B, C emitted by the HMM
NUM_STATES = 3

# Paper's joint T^(x)_{ij} = Pr(emit x, next state j | current state i).
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
_PAPER_T = torch.stack([_PAPER_T_A, _PAPER_T_B, _PAPER_T_C], dim=0)  # [3 emit, 3 from, 3 to]


def build_transition_matrices(self_loop: float) -> torch.Tensor:
    """Return T of shape [3 emit, 3 from, 3 to].

    self_loop=0.765 reproduces the paper exactly. Lower values blend the
    paper matrices toward uniform 1/9 per cell, producing a faster-mixing
    process with a coarser belief-simplex fractal.
    """
    if abs(self_loop - 0.765) < 1e-3:
        return _PAPER_T.clone()
    t = self_loop / 0.765
    uniform = 1.0 / 9.0
    return uniform + t * (_PAPER_T - uniform)


def stationary_dist(transitions: torch.Tensor, iters: int = 500) -> torch.Tensor:
    """Power-iterate the row-stochastic state-to-state matrix to convergence."""
    state_to_state = transitions.sum(dim=0)  # [from, to]
    pi = torch.full((NUM_STATES,), 1.0 / NUM_STATES)
    for _ in range(iters):
        pi = pi @ state_to_state
    return pi


class HMMSampler:
    """Vectorized MESS3 sampler.

    Produces (tokens, beliefs) pairs, where beliefs[b, t] is the posterior
    over hidden states AFTER observing tokens[b, 0..t-1]. Belief update
    is the standard HMM filter: η' = η T^(x) / (η T^(x) 1).

    With `n_comp > 1`, the same HMM trajectory gets rendered in different
    compartments by shifting tokens by `comp * VOCAB_SIZE_DATA`. The
    underlying HMM is identical so the true belief state is also
    identical — compartments are pure surface-form rebadgings.
    """

    def __init__(self, self_loop: float, device: torch.device) -> None:
        self.device = device
        T = build_transition_matrices(self_loop).to(device)  # [3, 3, 3]
        self.T = T
        # [from, 9] flat (emit, next_state) categorical per current state
        self.flat_probs = T.permute(1, 0, 2).reshape(NUM_STATES, 9)
        self.pi = stationary_dist(T.cpu()).to(device)
        # Emission-conditioned transition matrices reshaped for belief update:
        # T_emit[x, from, to] — same as self.T but we'll use it differently.
        self.T_emit = T  # alias

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        """Sample a [B, T] int64 tensor of emission tokens (no compartment shift)."""
        tokens = torch.empty((batch, seq_len), dtype=torch.long, device=self.device)
        state = torch.multinomial(self.pi, batch, replacement=True)  # [B]
        for t in range(seq_len):
            row = self.flat_probs[state]  # [B, 9]
            choice = torch.multinomial(row, 1).squeeze(-1)  # [B]
            tokens[:, t] = choice // NUM_STATES
            state = choice % NUM_STATES
        return tokens

    def sample_with_beliefs(
        self, batch: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens AND compute the ground-truth belief trajectory.

        Returns (tokens [B, T] int64, beliefs [B, T, NUM_STATES] float32),
        where beliefs[b, t] = belief state AFTER observing tokens[b, 0..t].
        """
        tokens = torch.empty((batch, seq_len), dtype=torch.long, device=self.device)
        beliefs_out = torch.empty(
            (batch, seq_len, NUM_STATES), dtype=torch.float32, device=self.device
        )
        # Both the sampler and the belief tracker start at the stationary
        # dist. These aren't the same thing — the sampler draws actual
        # hidden states, the tracker maintains a posterior — but they
        # happen to start from the same prior.
        state = torch.multinomial(self.pi, batch, replacement=True)  # [B]
        belief = self.pi.unsqueeze(0).expand(batch, -1).clone()  # [B, NUM_STATES]
        for t in range(seq_len):
            row = self.flat_probs[state]
            choice = torch.multinomial(row, 1).squeeze(-1)
            emit = choice // NUM_STATES
            next_state = choice % NUM_STATES
            tokens[:, t] = emit
            state = next_state
            # Belief update η' ∝ η T^(x)
            # T[emit[b]] is [NUM_STATES, NUM_STATES], one per batch row.
            # gather along dim 0 with an index of shape [B]
            T_x = self.T_emit[emit]  # [B, NUM_STATES, NUM_STATES]
            unn = torch.bmm(belief.unsqueeze(1), T_x).squeeze(1)  # [B, NUM_STATES]
            # Normalize each row to sum to 1. Clamp to avoid 0/0 for
            # impossible-path belief states (shouldn't happen since the
            # HMM is ergodic, but be defensive).
            unn_sum = unn.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            belief = unn / unn_sum
            beliefs_out[:, t] = belief
        return tokens, beliefs_out


# ──────────────────────────────────────────────────────────────────────────
# Linear probe (closed-form least squares, solved on GPU)
# ──────────────────────────────────────────────────────────────────────────


def fit_linear_probe(
    activations: torch.Tensor,  # [N, d]
    beliefs: torch.Tensor,  # [N, NUM_STATES]
    ridge: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit `beliefs ≈ activations @ W + b` by closed-form ridge regression.

    Returns (W [d, NUM_STATES], b [NUM_STATES]). Uses torch.linalg.lstsq
    on the augmented (activations | 1) design matrix with a small ridge
    penalty to keep the solve stable when activations are low-rank or
    degenerate early in training.
    """
    N, d = activations.shape
    # Augment with a column of ones to absorb the bias.
    ones = torch.ones(N, 1, device=activations.device, dtype=activations.dtype)
    X = torch.cat([activations, ones], dim=1)  # [N, d+1]
    # Ridge via the normal equations: W_aug = (XᵀX + λI)^-1 Xᵀ Y
    XtX = X.t() @ X  # [d+1, d+1]
    XtX = XtX + ridge * torch.eye(d + 1, device=X.device, dtype=X.dtype)
    XtY = X.t() @ beliefs  # [d+1, NUM_STATES]
    W_aug = torch.linalg.solve(XtX, XtY)  # [d+1, NUM_STATES]
    W = W_aug[:d]
    b = W_aug[d]
    return W, b


def probe_r2(
    activations: torch.Tensor,  # [N, d]
    beliefs: torch.Tensor,  # [N, NUM_STATES]
    W: torch.Tensor,  # [d, NUM_STATES]
    b: torch.Tensor,  # [NUM_STATES]
) -> float:
    """Standard R² for the probe on a given (activations, beliefs) set."""
    pred = activations @ W + b  # [N, NUM_STATES]
    mean = beliefs.mean(dim=0, keepdim=True)  # [1, NUM_STATES]
    ss_res = ((pred - beliefs) ** 2).sum()
    ss_tot = ((beliefs - mean) ** 2).sum()
    if ss_tot.item() <= 0:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


# ──────────────────────────────────────────────────────────────────────────
# Experiment
# ──────────────────────────────────────────────────────────────────────────


EVAL_INTERVAL = 50


@register(
    "mess3",
    description="Tiny causal transformer (4L 1H 64d, head_dim=8, RoPE) on the MESS3 HMM",
    params={
        # ── live (editable mid-run) ──
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
        # ── structural (fixed at creation) ──
        "seq_len": {
            "type": "number",
            "default": 10,
            "min": 1,
            "max": 256,
            "scale": "linear",
            "live": False,
            "description": "Context length",
        },
        "n_comp": {
            "type": "number",
            "default": 1,
            "min": 1,
            "max": 16,
            "scale": "linear",
            "live": False,
            "description": "Number of surface-form compartments (same HMM, shifted vocab)",
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
            "description": "Attention head dimension (must be even for RoPE)",
        },
        "mlp_dim": {
            "type": "number",
            "default": 256,
            "min": 8,
            "max": 4096,
            "scale": "linear",
            "live": False,
            "description": "MLP hidden width",
        },
        "eval_batch_size": {
            "type": "number",
            "default": 256,
            "min": 32,
            "max": 2048,
            "scale": "linear",
            "live": False,
            "description": "Batch size for the probe / cos-sim eval",
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

        # +1 mirrors the JS convention: leaves room for a task/separator
        # token that isn't used in single-task training but is allocated
        # in case a future mode needs it.
        self.n_comp = n_comp
        self.vocab_size = VOCAB_SIZE_DATA * n_comp + 1
        self.seq_len = seq_len
        self.eval_batch_size = int(self.params["eval_batch_size"])

        cfg = TransformerConfig(
            vocab_size=self.vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=1,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            head_dim=head_dim,
            pos_encoding="rope",
        )
        self.model = Transformer(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.params["lr"])
        )
        self.sampler = HMMSampler(self_loop, self.device)
        self.criterion = nn.CrossEntropyLoss()

    def step(self) -> dict[str, float]:
        # Re-read live params
        lr = float(self.params["lr"])
        batch_size = int(self.params["batch_size"])
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        # Sample a fresh batch. When n_comp > 1, each sequence goes into
        # a uniformly-random compartment (surface-form shift). The HMM
        # dynamics are identical so this is just a vocab-rebadging.
        tokens = self.sampler.sample(batch_size, self.seq_len)  # [B, T]
        if self.n_comp > 1:
            shifts = torch.randint(
                0, self.n_comp, (batch_size,), device=tokens.device
            ) * VOCAB_SIZE_DATA
            tokens = tokens + shifts.unsqueeze(-1)

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits = self.model(inputs)
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Compute the total L2 norm of all parameter gradients. Cheapest
        # sharpness proxy: spikes mean the loss landscape just handed us
        # a cliff, steady decrease means training is well-behaved. We use
        # clip_grad_norm_ with max_norm=inf so it returns the norm without
        # actually clipping — faster than walking params manually because
        # it stays on-GPU until the final .item() sync.
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float("inf")
            ).item()
        )
        self.optimizer.step()

        metrics: dict[str, float] = {
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
        }

        # Evaluation: probe R² + cosine similarity. Only runs every
        # EVAL_INTERVAL steps because it's O(~10ms) and would drown the
        # chart in duplicate points otherwise.
        if (self.step_count + 1) % EVAL_INTERVAL == 0:
            metrics.update(self._eval())
        return metrics

    @torch.no_grad()
    def _eval(self) -> dict[str, float]:
        """Compute probe R² (+ per-comp R², cos_sim when n_comp > 1)."""
        self.model.eval()
        try:
            return self._eval_inner()
        finally:
            self.model.train()

    def _eval_inner(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        # Sample an eval batch + belief trajectory. We use n_eval >>
        # train batch size so the probe fit is stable.
        n_eval = self.eval_batch_size
        tokens_base, beliefs_base = self.sampler.sample_with_beliefs(n_eval, self.seq_len)
        # tokens_base are in [0, VOCAB_SIZE_DATA-1]; the model vocab is
        # shifted by compartment. Probe fit uses compartment 0 (no shift).

        # Forward in compartment 0, collect residuals.
        logits0, residuals0 = self.model(tokens_base, return_residuals=True)
        last_resid_0 = residuals0[-1]  # [B, T, D]
        mid_resid_0 = residuals0[len(residuals0) // 2]  # [B, T, D]

        # Align: belief[t] should be predicted from activation[t]. Use
        # positions 0..seq_len-2 (last position has no target anyway).
        # activation[b, t] ↔ belief[b, t] where belief is AFTER seeing token t.
        n_pos = self.seq_len - 1
        act = last_resid_0[:, :n_pos].reshape(-1, last_resid_0.shape[-1])  # [B*n_pos, D]
        bel = beliefs_base[:, :n_pos].reshape(-1, NUM_STATES)

        # Probe is fit in float32 regardless of model dtype.
        W, b = fit_linear_probe(act.float(), bel.float())
        metrics["probe_r2"] = probe_r2(act.float(), bel.float(), W, b)

        if self.n_comp > 1:
            # Evaluate the same probe on other compartments. The HMM
            # trajectory is identical — we're testing whether the linear
            # map from residuals to beliefs transfers across
            # surface-form rebadgings.
            for c in range(1, min(self.n_comp, 4)):
                shifted = tokens_base + (c * VOCAB_SIZE_DATA)
                _, residuals_c = self.model(shifted, return_residuals=True)
                last_resid_c = residuals_c[-1]
                act_c = last_resid_c[:, :n_pos].reshape(-1, last_resid_c.shape[-1])
                metrics[f"probe_r2_c{c}"] = probe_r2(act_c.float(), bel.float(), W, b)

            # Cross-compartment cosine similarity at the LAST prompt
            # position (seq_len - 2) of the middle-layer residual.
            # Compare compartment 0 vs compartment 1.
            shifted_1 = tokens_base + VOCAB_SIZE_DATA
            _, residuals_1 = self.model(shifted_1, return_residuals=True)
            mid_resid_1 = residuals_1[len(residuals_1) // 2]
            # Take the position just before the final (where the model
            # has seen the full prompt).
            pos = self.seq_len - 2
            a = mid_resid_0[:, pos, :]  # [B, D]
            c1 = mid_resid_1[:, pos, :]
            metrics["cos_sim"] = float(
                F.cosine_similarity(a, c1, dim=-1).mean().item()
            )

        # ── Sharpness (λ_max of the Hessian) ──
        # Power iteration with HVPs on a fresh minibatch. The helper
        # opens its own `torch.enable_grad` scope so we can call it
        # from inside `_eval`'s `@torch.no_grad`. ~15 iterations × 3
        # passes ≈ 45 training-step-equivalents, so at
        # EVAL_INTERVAL=50 this is about 1% overhead overall.
        sharpness_bs = min(64, self.eval_batch_size)

        def _sharpness_loss() -> torch.Tensor:
            tok = self.sampler.sample(sharpness_bs, self.seq_len)
            inp = tok[:, :-1]
            tgt = tok[:, 1:]
            logits = self.model(inp)
            return self.criterion(
                logits.reshape(-1, self.vocab_size), tgt.reshape(-1)
            )

        params = [p for p in self.model.parameters() if p.requires_grad]
        try:
            metrics["sharpness"] = hessian_lambda_max(
                _sharpness_loss, params, num_iters=15
            )
        except Exception:
            # Numerical issues early in training (e.g. Hessian
            # degeneracies when params are still close to init) — skip
            # this step's reading rather than crashing the eval.
            pass

        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        # Move optimizer state tensors to the experiment's device — the
        # worker loads checkpoints on CPU (so the RNG ByteTensor stays on
        # CPU as torch.set_rng_state requires) and optimizer.load_state_dict
        # copies tensors as-is.
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
