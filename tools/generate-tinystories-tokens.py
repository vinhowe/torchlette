#!/usr/bin/env python3
"""
Regenerate the canonical TinyStories token blob(s) used by the DiLoCo
regression harness and the fullstack parity trainers.

This is the in-tree, revision-pinned reconstruction of the recipe that
originally lived only in `tools/diloco-pytorch/data.py::_build_cache`
(never committed as a standalone generator). The blobs it produces were
destroyed in a symlink-clobber incident (2026-07-18); this script exists
so the canonical inputs can always be rebuilt deterministically.

RECIPE (must match the original blob byte-for-byte):
  - dataset  : roneneldan/TinyStories, split "train"
  - revision : f54c09fd23315a6f9c86f9dc80f725de7d8f9c64  (pinned; the
               snapshot cached on this host — the 4-shard parquet train
               split). Pinning makes the token count reproducible.
  - order    : document order as-published (no shuffle, no select)
  - tokenizer: gpt2 (GPT2TokenizerFast), add_special_tokens=False
  - separator: append EOS (id 50256) after EVERY story so adjacent
               stories don't blend
  - dtype    : uint16, little-endian, flat (one contiguous stream)

OUTPUTS (into the PRIMARY repo ckpts/ — a REAL directory, never a symlink):
  - ckpts/tinystories-tokens.bin  (canonical: 473,992,236 tokens)
  - ckpts/ts-8L-128D.bin          (first 16,319,744 tokens = a prefix of
                                    the same stream; used by the 8L/128D
                                    regression model config)

Determinism: no randomness anywhere in blob construction. The batch
tokenizer and document order are fixed; the same dataset revision +
tokenizer yields identical bytes on every run.

Usage:
  HF_HUB_OFFLINE=1 uv run --with datasets --with transformers \
    python3 tools/generate-tinystories-tokens.py

Env knobs:
  CKPTS_DIR       output dir (default: primary repo ckpts/)
  TS_REVISION     dataset revision (default: the pinned hash above)
  SKIP_PREFIX     if "1", only write the full blob (skip ts-8L-128D.bin)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# ── Canonical constants (from the incident record) ──
CANONICAL_TOKEN_COUNT = 473_992_236
PREFIX_TOKEN_COUNT = 16_319_744  # ts-8L-128D.bin = first N tokens of the stream
EOS_ID = 50256  # gpt2 <|endoftext|>
DATASET = "roneneldan/TinyStories"
DEFAULT_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"

CKPTS_DIR = Path(
    os.environ.get(
        "CKPTS_DIR",
        "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts",
    )
)
REVISION = os.environ.get("TS_REVISION", DEFAULT_REVISION)
FULL_PATH = CKPTS_DIR / "tinystories-tokens.bin"
PREFIX_PATH = CKPTS_DIR / "ts-8L-128D.bin"


def log(msg: str) -> None:
    print(f"[gen-tokens] {msg}", flush=True)


def _train_parquet_shards() -> list[str]:
    """Resolve the cached train parquet shards for the pinned revision.

    We read the parquet shards DIRECTLY from the local hub snapshot rather
    than via `load_dataset(DATASET, revision=...)`, because the latter makes
    a dataset_info API call to resolve the revision — which fails under
    HF_HUB_OFFLINE=1 even when the snapshot is fully cached. Reading the
    shards explicitly (sorted by filename: train-00000..00003) reproduces
    the exact document order `load_dataset` would yield, fully offline.
    """
    import glob

    from huggingface_hub.constants import HF_HUB_CACHE

    snap = os.path.join(
        HF_HUB_CACHE,
        "datasets--roneneldan--TinyStories",
        "snapshots",
        REVISION,
        "data",
    )
    shards = sorted(glob.glob(os.path.join(snap, "train-*.parquet")))
    if not shards:
        raise FileNotFoundError(
            f"no train parquet shards under {snap} — the pinned revision "
            f"{REVISION} is not cached; download it first (online)"
        )
    return shards


def build_stream() -> np.ndarray:
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    shards = _train_parquet_shards()
    log(f"loading {DATASET} train from {len(shards)} cached parquet shards "
        f"(revision {REVISION})")
    for s in shards:
        log(f"  shard {os.path.basename(s)}")
    ds = load_dataset("parquet", data_files=shards, split="train")
    log(f"rows: {len(ds):,}")

    chunks: list[np.ndarray] = []
    chunk_size = 10_000
    for i in range(0, len(ds), chunk_size):
        texts = ds[i : i + chunk_size]["text"]
        enc = tokenizer(texts, add_special_tokens=False)
        for ids in enc["input_ids"]:
            chunks.append(np.array(ids + [EOS_ID], dtype=np.uint16))
        if (i // chunk_size) % 10 == 0:
            tok_so_far = sum(c.size for c in chunks)
            log(f"  tokenized rows={i + chunk_size:,} tokens={tok_so_far:,}")

    flat = np.concatenate(chunks).astype(np.uint16)
    log(f"stream built: {flat.size:,} tokens")
    return flat


def main() -> int:
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    # Guard: refuse to run into a symlinked ckpts/ (the incident cause).
    if CKPTS_DIR.is_symlink():
        log(f"FATAL: {CKPTS_DIR} is a symlink — must be a real directory")
        return 3

    flat = build_stream()

    # ── Full canonical blob ──
    flat.tofile(FULL_PATH)
    n = flat.size
    log(
        f"wrote {FULL_PATH} — {n:,} tokens, "
        f"{FULL_PATH.stat().st_size / 1e6:.1f} MB"
    )
    if n == CANONICAL_TOKEN_COUNT:
        log(f"COUNT MATCH: {n:,} == canonical {CANONICAL_TOKEN_COUNT:,}")
    else:
        delta = n - CANONICAL_TOKEN_COUNT
        log(
            f"COUNT MISMATCH: got {n:,}, canonical {CANONICAL_TOKEN_COUNT:,} "
            f"(delta {delta:+,}) — investigate recipe before trusting baselines"
        )

    # ── 8L/128D prefix blob (first N tokens of the same stream) ──
    if os.environ.get("SKIP_PREFIX") != "1":
        if n < PREFIX_TOKEN_COUNT:
            log(
                f"FATAL: stream ({n:,}) shorter than prefix count "
                f"({PREFIX_TOKEN_COUNT:,})"
            )
            return 4
        prefix = flat[:PREFIX_TOKEN_COUNT]
        prefix.tofile(PREFIX_PATH)
        log(
            f"wrote {PREFIX_PATH} — {prefix.size:,} tokens "
            f"(first {PREFIX_TOKEN_COUNT:,} of the stream), "
            f"{PREFIX_PATH.stat().st_size / 1e6:.1f} MB"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
