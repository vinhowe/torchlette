"""
Weight-transfer forward-loss diff — step 3 (compare).

Reads the PyTorch reference (pt.loss.json, pt/g.*.f32) and the torchlette
output (tl.loss.json, tl/g.*.f32) and prints:
  - forward loss delta (pure forward numerics on identical weights)
  - per-param grad relative error, sorted worst-first (localizes the op
    whose backward diverges)

Relative error = ||g_tl - g_pt|| / (||g_pt|| + eps), the standard
gradient-check metric.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

PARITY = Path(
    os.environ.get("PARITY_DIR", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/parity")
)

manifest = json.load(open(PARITY / "manifest.json"))
pt_loss = json.load(open(PARITY / "pt.loss.json"))["loss"]
tl_loss = json.load(open(PARITY / "tl.loss.json"))["loss"]

print("=" * 64)
print(f"forward loss  pt={pt_loss:.6f}  tl={tl_loss:.6f}  delta={tl_loss-pt_loss:+.6f}")
print("=" * 64)

rows = []
for key in manifest:
    gpt = np.fromfile(PARITY / "pt" / f"g.{key}.f32", dtype=np.float32)
    gtl = np.fromfile(PARITY / "tl" / f"g.{key}.f32", dtype=np.float32)
    n = min(gpt.size, gtl.size)
    gpt, gtl = gpt[:n], gtl[:n]
    npt = np.linalg.norm(gpt)
    ntl = np.linalg.norm(gtl)
    rel = np.linalg.norm(gtl - gpt) / (npt + 1e-12)
    # cosine similarity catches direction differences independent of scale
    cos = float(gpt @ gtl / (npt * ntl + 1e-12))
    rows.append((key, rel, cos, npt, ntl))

rows.sort(key=lambda r: -r[1])
print(f"{'param':28s} {'rel_err':>10s} {'cos':>9s} {'|g_pt|':>10s} {'|g_tl|':>10s}")
for key, rel, cos, npt, ntl in rows:
    print(f"{key:28s} {rel:10.4f} {cos:9.5f} {npt:10.4f} {ntl:10.4f}")
