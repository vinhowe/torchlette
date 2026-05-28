"""
Generate a shared window-offset file consumed by BOTH the PyTorch baseline
and the torchlette parity driver. Each int32 is a start index into the
TinyStories token cache. Both stacks read this file and slice identical
windows, so any remaining loss gap is attributable to numerics (kernels /
init), not data ordering.

Layout: flat int32 little-endian, length ROUNDS*STEPS*BATCH, indexed by
  g = (round*STEPS + step)*BATCH + b
"""

from __future__ import annotations

import os

import numpy as np

ROUNDS = int(os.environ.get("ROUNDS", "30"))
STEPS = int(os.environ.get("STEPS", "20"))
BATCH = int(os.environ.get("BATCH_SIZE", "8"))
SEQ = int(os.environ.get("SEQ_LEN", "256"))
SEED = int(os.environ.get("OFFSET_SEED", "1234"))
TOKENS = os.environ.get(
    "LOCAL_TOKENS",
    "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin",
)
OUT = os.environ.get(
    "WINDOW_OFFSETS",
    "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/window-offsets.i32",
)

n_tokens = os.path.getsize(TOKENS) // 2  # uint16
max_start = n_tokens - SEQ - 1
rng = np.random.default_rng(SEED)
offsets = rng.integers(0, max_start, size=ROUNDS * STEPS * BATCH, dtype=np.int64)
offsets.astype(np.int32).tofile(OUT)
print(
    f"wrote {offsets.size} offsets to {OUT} "
    f"(max_start={max_start}, ROUNDS={ROUNDS} STEPS={STEPS} BATCH={BATCH})"
)
