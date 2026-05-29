"""
Multi-step weight-transfer PARITY trainer (PyTorch side).

Loads the SAME shared weights written by gen_parity_weights.py, trains N
fp32 steps over the SAME window offsets (window-offsets.i32) with identical
AdamW + grad clip, and records per-step loss to parity/pt.losses.json.

The torchlette counterpart (tools/parity-train-diff.ts) does the exact same
thing; the two loss arrays are then diffed. Single-step gradients already
match bit-for-bit; this checks that the OPTIMIZER + multi-step trajectory
stay matched (i.e. no bug like the CSE/outputIndex one re-emerges in the
loop, and fp32 kernel drift stays bounded).

Env: NUM_LAYERS NUM_HEADS EMBED_DIM SEQ_LEN BATCH_SIZE STEPS LR WEIGHT_DECAY
     GRAD_CLIP PARITY_DIR LOCAL_TOKENS WINDOW_OFFSETS CUDA_VISIBLE_DEVICES
"""

from __future__ import annotations

import json
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
STEPS = int(os.environ.get("STEPS", "60"))
LR = float(os.environ.get("LR", "5e-4"))
WD = float(os.environ.get("WEIGHT_DECAY", "0.01"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))

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


def log(m: str) -> None:
    print(f"[pt-parity] {m}", flush=True)


def main() -> None:
    manifest = json.load(open(PARITY / "manifest.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPTConfig(
        vocab_size=VOCAB, block_size=BLOCK, n_layer=L, n_head=H, n_embed=E, dropout=0.0
    )
    m = GPT(cfg).to(dev).float()

    PT = {
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
            w = np.fromfile(PARITY / f"w.{k}.f32", dtype=np.float32).reshape(
                manifest[k]
            )
            p.copy_(torch.from_numpy(w).to(dev))
    log(f"loaded {len(PT)} shared weights; params={m.num_params():,}")

    opt = torch.optim.AdamW(
        m.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD
    )

    toks = np.fromfile(TOKENS, dtype=np.uint16)
    offs = np.fromfile(OFFSETS, dtype=np.int32)

    def batch_for(gs: int):
        starts = offs[gs * BATCH : gs * BATCH + BATCH]
        ins = np.stack([toks[s : s + SEQ] for s in starts]).astype(np.int64)
        tgs = np.stack([toks[s + 1 : s + 1 + SEQ] for s in starts]).astype(np.int64)
        return torch.from_numpy(ins).to(dev), torch.from_numpy(tgs).to(dev)

    m.train()
    losses: list[float] = []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        inp, tgt = batch_for(step)
        _, loss = m(inp, tgt)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
        opt.step()
        lv = float(loss.item())
        losses.append(lv)
        if step < 5 or step % 10 == 0 or step == STEPS - 1:
            log(f"step {step:3d}: loss={lv:.6f}")

    json.dump({"losses": losses}, open(PARITY / "pt.losses.json", "w"))
    log(f"wrote {len(losses)} losses to {PARITY/'pt.losses.json'}")


if __name__ == "__main__":
    main()
