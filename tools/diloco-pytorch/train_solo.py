"""
Single-peer PyTorch baseline matching the torchlette TinyStories run:
8-layer GPT, n_head=4, n_embed=128, BATCH=8, SEQ=256, LR=5e-4, wd=0.01,
grad clip 1.0, INNER_STEPS=20 per "round", ROUNDS=200 by default.

Mirrors the torchlette agent's STATS line format so the two curves can be
diffed directly. Saves periodic checkpoints + final.

Env knobs:
  ROUNDS, STEPS, BATCH_SIZE, SEQ_LEN, LR, WEIGHT_DECAY, NUM_LAYERS,
  NUM_HEADS, EMBED_DIM, CHECKPOINT_PATH, CHECKPOINT_EVERY, GRAD_CLIP,
  SEED, CUDA_VISIBLE_DEVICES
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import torch

# allow running from repo root via `python tools/diloco-pytorch/train_solo.py`
sys.path.insert(0, str(Path(__file__).parent))

from data import LocalTokenSource  # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

import numpy as np  # noqa: E402


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def log(msg: str) -> None:
    print(f"[pt-solo] {msg}", flush=True)


def main() -> None:
    seed = env_int("SEED", 42)
    rounds = env_int("ROUNDS", 200)
    inner_steps = env_int("STEPS", 20)
    batch_size = env_int("BATCH_SIZE", 8)
    seq_len = env_int("SEQ_LEN", 256)
    accum_steps = env_int("ACCUM_STEPS", 1)
    lr = env_float("LR", 5e-4)
    weight_decay = env_float("WEIGHT_DECAY", 0.01)
    grad_clip = env_float("GRAD_CLIP", 1.0)
    n_layer = env_int("NUM_LAYERS", 8)
    n_head = env_int("NUM_HEADS", 4)
    n_embed = env_int("EMBED_DIM", 128)
    use_amp = os.environ.get("USE_AMP", "0") == "1"
    amp_dtype_name = os.environ.get("AMP_DTYPE", "float16")
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[amp_dtype_name]
    ckpt_path = os.environ.get("CHECKPOINT_PATH")
    ckpt_every = env_int("CHECKPOINT_EVERY", 10)

    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    cfg = GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=n_layer,
        n_head=n_head,
        n_embed=n_embed,
    )
    model = GPT(cfg).to(device)
    log(f"GPT params={model.num_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    # Only use GradScaler for fp16 — bf16 doesn't underflow the same way and
    # doesn't need scaling (PyTorch's GradScaler is a no-op for bf16 anyway).
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=1024.0)
        if use_amp and amp_dtype == torch.float16
        else None
    )
    log(
        f"AMP: enabled={use_amp} dtype={amp_dtype_name} scaler={'on' if scaler else 'off'}"
    )

    token_source = LocalTokenSource()
    token_source.load()
    rng = np.random.default_rng(seed)

    def save_ckpt(tag: str) -> None:
        if not ckpt_path:
            return
        p = Path(ckpt_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "step_tag": tag,
            },
            p,
        )
        log(f"checkpoint saved ({tag}): {ckpt_path}")

    def shutdown(signum, frame):
        log("SIGTERM/SIGINT — saving and exiting")
        save_ckpt("shutdown")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    for r in range(rounds):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for step in range(inner_steps):
            optimizer.zero_grad(set_to_none=True)
            for acc in range(accum_steps):
                inputs_np, targets_np = token_source.sample_window(
                    seq_len, batch_size, rng
                )
                inputs = torch.from_numpy(inputs_np).to(device, non_blocking=True)
                targets = torch.from_numpy(targets_np).to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        _, loss = model(inputs, targets)
                else:
                    _, loss = model(inputs, targets)
                scaled = (loss / accum_steps)
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                total_loss += loss.item()
            if grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        avg_loss = total_loss / (inner_steps * accum_steps)
        dt = time.time() - t0
        rss_mb = 0
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = int(torch.cuda.memory_allocated() / 1e6)
        stats = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "round": r,
            "anchor_round": r + 1,
            "outer_step": True,
            "contributors": 1,
            "clusters": 1,
            "f16w_applied": False,
            "loss": round(avg_loss, 4),
            "gpu_mb": gpu_mb,
            "round_s": round(dt, 2),
        }
        print(f"STATS {json.dumps(stats)}", flush=True)

        if ckpt_path and (r + 1) % ckpt_every == 0:
            save_ckpt(f"round={r + 1}")

    save_ckpt("final")


if __name__ == "__main__":
    main()
