"""
SDPA-with-head-plumbing micro-parity, PyTorch reference.

Mirrors the EXACT op sequence in the torchlette attention module around the
SDPA core: a combined qkv tensor [B,S,3E] -> chunk(3) -> reshape to heads ->
permute -> (contiguous) -> SDPA -> permute back -> reshape to [B,S,E]. Uses
loss = sum(attnFlat * dOut). Saves d(qkv). The torchlette side compares — if
d(qkv) diverges while bare SDPA matched, the bug is in the chunk/reshape/
permute backward, not the attention kernel.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(os.environ.get("SDPA2_DIR", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/sdpa2"))
OUT.mkdir(parents=True, exist_ok=True)

B, S, H, hd = 2, 16, 4, 32
E = H * hd
scale = 1.0 / (hd**0.5)
rng = np.random.default_rng(11)

qkv = (rng.standard_normal((B, S, 3 * E)).astype(np.float32) * 0.5)
dOut = (rng.standard_normal((B, S, E)).astype(np.float32) * 0.5)
qkv.tofile(OUT / "qkv.f32")
dOut.tofile(OUT / "dout.f32")

dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor(qkv, device=dev, requires_grad=True)
dOt = torch.tensor(dOut, device=dev)

q, k, v = x.split(E, dim=2)  # each [B,S,E]


def to_heads(t):
    return t.reshape(B, S, H, hd).permute(0, 2, 1, 3).contiguous()


qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale)
attn_flat = out.permute(0, 2, 1, 3).reshape(B, S, E)
loss = (attn_flat * dOt).sum()
loss.backward()

attn_flat.detach().cpu().numpy().astype(np.float32).tofile(OUT / "attnflat.f32")
x.grad.detach().cpu().numpy().astype(np.float32).tofile(OUT / "dqkv.f32")
print(f"SDPA2 ref: loss={float(loss):.6f} |attnflat|={float(attn_flat.norm()):.4f} |dqkv|={float(x.grad.norm()):.4f}")
