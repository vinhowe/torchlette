"""
FULL-STACK multi-step parity trainer (PyTorch reference side).

Mirrors tools/parity-fullstack-tl.ts: autocast(f16) + GradScaler + grad clip +
AdamW over the SAME shared weights + window offsets. (Gradient checkpointing is
math-transparent — identical grads to non-checkpointed — so the reference omits
it; the tl side runs WITH checkpointing and must still match.) Writes:
  parity/pt_fs.losses.json     : per-step loss trajectory
  parity/pt_fs/g.<key>.f32     : step-0 grads (post-unscale, PRE-clip)

Env: NUM_LAYERS NUM_HEADS EMBED_DIM SEQ_LEN BATCH_SIZE STEPS LR WEIGHT_DECAY
     GRAD_CLIP USE_AUTOCAST(=1) USE_SCALER(=1) PARITY_DIR LOCAL_TOKENS
     WINDOW_OFFSETS CUDA_VISIBLE_DEVICES
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
STEPS = int(os.environ.get("STEPS", "30"))
LR = float(os.environ.get("LR", "5e-4"))
WD = float(os.environ.get("WEIGHT_DECAY", "0.01"))
GRAD_CLIP = float(os.environ.get("GRAD_CLIP", "1.0"))
USE_AUTOCAST = os.environ.get("USE_AUTOCAST", "1") != "0"
USE_SCALER = os.environ.get("USE_SCALER", "1") != "0"

PARITY = Path(
    os.environ.get("PARITY_DIR", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/parity")
)
TOKENS = os.environ.get(
    "LOCAL_TOKENS", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin"
)
OFFSETS = os.environ.get(
    "WINDOW_OFFSETS", "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/window-offsets.i32"
)


def log(m: str) -> None:
    print(f"[pt-fs] {m}", flush=True)


def main() -> None:
    manifest = json.load(open(PARITY / "manifest.json"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=L, n_head=H, n_embed=E, dropout=0.0)
    m = GPT(cfg).to(dev).float()

    PT = {"wte": m.tok_emb.weight, "wpe": m.pos_emb.weight, "lnf.w": m.ln_f.weight, "lnf.b": m.ln_f.bias}
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
            w = np.fromfile(PARITY / f"w.{k}.f32", dtype=np.float32).reshape(manifest[k])
            p.copy_(torch.from_numpy(w).to(dev))
    log(f"loaded {len(PT)} weights; autocast={USE_AUTOCAST} scaler={USE_SCALER} clip={GRAD_CLIP} steps={STEPS}")

    opt = torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=USE_SCALER)

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
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AUTOCAST):
            _, loss = m(inp, tgt)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        if step == 0:  # dump grads post-unscale, PRE-clip
            gdir = PARITY / "pt_fs"
            gdir.mkdir(parents=True, exist_ok=True)
            for k, p in PT.items():
                p.grad.detach().cpu().numpy().astype(np.float32).tofile(gdir / f"g.{k}.f32")

        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        lv = float(loss.item())
        losses.append(lv)
        if step < 5 or step % 10 == 0 or step == STEPS - 1:
            log(f"step {step:3d}: loss={lv:.6f}")

    json.dump({"losses": losses}, open(PARITY / "pt_fs.losses.json", "w"))
    log(f"wrote {len(losses)} losses to pt_fs.losses.json")


if __name__ == "__main__":
    main()
