"""
Weight-transfer forward-loss diff — step 1 (generate + PyTorch reference).

Generates ONE set of GPT-2 weights and ONE fixed batch in numpy (framework
neutral), loads them into the PyTorch model, runs a single fp32
forward+backward, and saves:
  - parity/manifest.json         : {canonical_key: shape}
  - parity/w.<key>.f32           : the shared weights (loaded by BOTH stacks)
  - parity/input.i32, target.i32 : the shared batch
  - parity/pt.loss.json          : PyTorch forward loss
  - parity/pt/g.<key>.f32        : PyTorch grads per canonical key

The torchlette driver (tools/parity-forward-diff.ts) loads the SAME weights
and batch and saves parity/tl.loss.json + parity/tl/g.<key>.f32. Then
compare_parity.py diffs forward loss (pure forward numerics) and per-param
grads (forward+backward numerics, localized per op).

Canonical keys (PyTorch shapes; torchlette zero-pads wte to padded vocab):
  wte [V,E]  wpe [B,E]
  block.{i}.ln1.w/b   block.{i}.attn.qkv.w/b   block.{i}.attn.proj.w/b
  block.{i}.ln2.w/b   block.{i}.mlp.fc.w/b     block.{i}.mlp.proj.w/b
  lnf.w/b
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import GPT, GPTConfig  # noqa: E402

VOCAB = 50257
BLOCK = 1024
L = int(os.environ.get("NUM_LAYERS", "8"))
H = int(os.environ.get("NUM_HEADS", "4"))
E = int(os.environ.get("EMBED_DIM", "128"))
SEQ = int(os.environ.get("SEQ_LEN", "256"))
BATCH = int(os.environ.get("BATCH_SIZE", "8"))
SEED = int(os.environ.get("PARITY_SEED", "1234"))

PARITY = Path(
    os.environ.get("PARITY_DIR", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/parity")
)
TOKENS = os.environ.get(
    "LOCAL_TOKENS",
    "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin",
)
OFFSETS = os.environ.get(
    "WINDOW_OFFSETS",
    "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/window-offsets.i32",
)

PARITY.mkdir(parents=True, exist_ok=True)
(PARITY / "pt").mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)


def normal(shape, std):
    return (rng.standard_normal(size=shape).astype(np.float32) * std).astype(np.float32)


# ---- generate weights (nanoGPT init) ----
W: dict[str, np.ndarray] = {}
W["wte"] = normal((VOCAB, E), 0.02)
W["wpe"] = normal((BLOCK, E), 0.02)
resid_std = 0.02 / math.sqrt(2 * L)
for i in range(L):
    W[f"block.{i}.ln1.w"] = np.ones(E, np.float32)
    W[f"block.{i}.ln1.b"] = np.zeros(E, np.float32)
    W[f"block.{i}.attn.qkv.w"] = normal((3 * E, E), 0.02)
    W[f"block.{i}.attn.qkv.b"] = np.zeros(3 * E, np.float32)
    W[f"block.{i}.attn.proj.w"] = normal((E, E), resid_std)
    W[f"block.{i}.attn.proj.b"] = np.zeros(E, np.float32)
    W[f"block.{i}.ln2.w"] = np.ones(E, np.float32)
    W[f"block.{i}.ln2.b"] = np.zeros(E, np.float32)
    W[f"block.{i}.mlp.fc.w"] = normal((4 * E, E), 0.02)
    W[f"block.{i}.mlp.fc.b"] = np.zeros(4 * E, np.float32)
    W[f"block.{i}.mlp.proj.w"] = normal((E, 4 * E), resid_std)
    W[f"block.{i}.mlp.proj.b"] = np.zeros(E, np.float32)
W["lnf.w"] = np.ones(E, np.float32)
W["lnf.b"] = np.zeros(E, np.float32)

for k, v in W.items():
    v.tofile(PARITY / f"w.{k}.f32")
json.dump({k: list(v.shape) for k, v in W.items()}, open(PARITY / "manifest.json", "w"))

# ---- fixed batch from the shared offsets ----
toks = np.fromfile(TOKENS, dtype=np.uint16)
offs = np.fromfile(OFFSETS, dtype=np.int32)[:BATCH]
inp = np.stack([toks[s : s + SEQ] for s in offs]).astype(np.int64)
tgt = np.stack([toks[s + 1 : s + 1 + SEQ] for s in offs]).astype(np.int64)
inp.astype(np.int32).tofile(PARITY / "input.i32")
tgt.astype(np.int32).tofile(PARITY / "target.i32")
print(f"batch: input {inp.shape} target {tgt.shape}; offsets[:4]={offs[:4].tolist()}")

# ---- load into PyTorch, fp32 forward+backward ----
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=L, n_head=H, n_embed=E, dropout=0.0)
m = GPT(cfg).to(dev).float()

PT = {  # canonical -> pytorch param attribute
    "wte": m.tok_emb.weight,
    "wpe": m.pos_emb.weight,
    "lnf.w": m.ln_f.weight,
    "lnf.b": m.ln_f.bias,
}
for i in range(L):
    b = m.blocks[i]
    PT[f"block.{i}.ln1.w"] = b.ln1.weight
    PT[f"block.{i}.ln1.b"] = b.ln1.bias
    PT[f"block.{i}.attn.qkv.w"] = b.attn.qkv.weight
    PT[f"block.{i}.attn.qkv.b"] = b.attn.qkv.bias
    PT[f"block.{i}.attn.proj.w"] = b.attn.proj.weight
    PT[f"block.{i}.attn.proj.b"] = b.attn.proj.bias
    PT[f"block.{i}.ln2.w"] = b.ln2.weight
    PT[f"block.{i}.ln2.b"] = b.ln2.bias
    PT[f"block.{i}.mlp.fc.w"] = b.mlp.fc.weight
    PT[f"block.{i}.mlp.fc.b"] = b.mlp.fc.bias
    PT[f"block.{i}.mlp.proj.w"] = b.mlp.proj.weight
    PT[f"block.{i}.mlp.proj.b"] = b.mlp.proj.bias

with torch.no_grad():
    for k, p in PT.items():
        p.copy_(torch.from_numpy(W[k]).to(dev))

inp_t = torch.from_numpy(inp).to(dev)
tgt_t = torch.from_numpy(tgt).to(dev)
m.train()
m.zero_grad(set_to_none=True)
_, loss = m(inp_t, tgt_t)
loss_val = float(loss.item())
loss.backward()
print(f"PyTorch fp32 forward loss = {loss_val:.6f}")
json.dump({"loss": loss_val}, open(PARITY / "pt.loss.json", "w"))

for k, p in PT.items():
    g = p.grad
    assert g is not None, f"no grad for {k}"
    g.detach().to("cpu").numpy().astype(np.float32).tofile(PARITY / "pt" / f"g.{k}.f32")
print(f"saved {len(PT)} PyTorch grads to {PARITY/'pt'}")
