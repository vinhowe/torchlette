"""
SDPA backward micro-parity, PyTorch reference side.

Generates random q,k,v,dO [B,H,S,D], runs causal scaled-dot-product
attention, and uses loss = sum(out * dO) so that d(loss)/d(out) = dO
exactly. Saves out + dq,dk,dv. The torchlette side (sdpa-diff.ts) loads
the SAME q,k,v,dO and compares — isolating the attention backward kernel
from the qkv/reshape plumbing around it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(os.environ.get("SDPA_DIR", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/sdpa"))
OUT.mkdir(parents=True, exist_ok=True)

B, H, S, D = 2, 4, 16, 32
scale = 1.0 / (D**0.5)
rng = np.random.default_rng(7)


def gen(shape):
    return rng.standard_normal(size=shape).astype(np.float32) * 0.5


q = gen((B, H, S, D))
k = gen((B, H, S, D))
v = gen((B, H, S, D))
dO = gen((B, H, S, D))
for name, arr in [("q", q), ("k", k), ("v", v), ("dO", dO)]:
    arr.tofile(OUT / f"{name}.f32")

dev = "cuda" if torch.cuda.is_available() else "cpu"
qt = torch.tensor(q, device=dev, requires_grad=True)
kt = torch.tensor(k, device=dev, requires_grad=True)
vt = torch.tensor(v, device=dev, requires_grad=True)
dOt = torch.tensor(dO, device=dev)

out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=scale)
loss = (out * dOt).sum()
loss.backward()

out.detach().cpu().numpy().astype(np.float32).tofile(OUT / "out.f32")
qt.grad.detach().cpu().numpy().astype(np.float32).tofile(OUT / "dq.f32")
kt.grad.detach().cpu().numpy().astype(np.float32).tofile(OUT / "dk.f32")
vt.grad.detach().cpu().numpy().astype(np.float32).tofile(OUT / "dv.f32")
print(f"SDPA ref: out|{out.shape}| loss={float(loss):.6f}")
print(f"  |out|={np.linalg.norm(out.detach().cpu().numpy()):.4f}")
print(f"  |dq|={float(qt.grad.norm()):.4f} |dk|={float(kt.grad.norm()):.4f} |dv|={float(vt.grad.norm()):.4f}")
