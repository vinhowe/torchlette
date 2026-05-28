"""
Local TinyStories token source. Uses HuggingFace `datasets` which fetches
the parquet shards once (cached under ~/.cache/huggingface) and then
streams locally — sidesteps the heavily rate-limited datasets-server API
that the torchlette agent uses.

Tokenizes lazily into one big numpy array and serves random windows.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from transformers import GPT2TokenizerFast


CACHE_DIR = Path(
    os.environ.get(
        "TINYSTORIES_TOKEN_CACHE",
        "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin",
    )
)


class LocalTokenSource:
    def __init__(self, max_rows: int | None = None) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokens: np.ndarray | None = None
        self.max_rows = max_rows

    def _build_cache(self) -> np.ndarray:
        print(f"[data] building token cache at {CACHE_DIR}", flush=True)
        from datasets import load_dataset

        ds = load_dataset("roneneldan/TinyStories", split="train")
        if self.max_rows is not None:
            ds = ds.select(range(min(self.max_rows, len(ds))))
        # Tokenize in chunks to keep memory bounded.
        chunks: list[np.ndarray] = []
        chunk_size = 10_000
        eos = self.tokenizer.eos_token_id or 50256
        for i in range(0, len(ds), chunk_size):
            texts = ds[i : i + chunk_size]["text"]
            enc = self.tokenizer(texts, add_special_tokens=False)
            # interleave EOS so adjacent stories don't blend.
            for ids in enc["input_ids"]:
                chunks.append(np.array(ids + [eos], dtype=np.uint16))
            if (i // chunk_size) % 10 == 0:
                tok_so_far = sum(c.size for c in chunks)
                print(
                    f"[data]   tokenized rows={i + chunk_size} tokens={tok_so_far:,}",
                    flush=True,
                )
        flat = np.concatenate(chunks).astype(np.uint16)
        CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        flat.tofile(CACHE_DIR)
        print(f"[data] cache built: {flat.size:,} tokens, {CACHE_DIR.stat().st_size / 1e6:.1f} MB", flush=True)
        return flat

    def load(self) -> np.ndarray:
        if self.tokens is not None:
            return self.tokens
        if CACHE_DIR.exists():
            self.tokens = np.fromfile(CACHE_DIR, dtype=np.uint16)
            print(f"[data] loaded {self.tokens.size:,} tokens from cache", flush=True)
        else:
            self.tokens = self._build_cache()
        return self.tokens

    def sample_window(self, seq_len: int, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        toks = self.load()
        max_start = toks.size - seq_len - 1
        starts = rng.integers(0, max_start, size=batch_size)
        inputs = np.stack([toks[s : s + seq_len] for s in starts]).astype(np.int64)
        targets = np.stack([toks[s + 1 : s + 1 + seq_len] for s in starts]).astype(np.int64)
        return inputs, targets
