# TinyStories token blob — canonical recipe & recovery

The DiLoCo regression harness (`tools/diloco-regression-check.ts`) and the
fullstack parity trainers (`tools/parity-fullstack-tl.ts`,
`tools/diloco-pytorch/fullstack_pt.py`, …) all read a pre-tokenized
TinyStories blob from `ckpts/`. The recorded loss baselines are a function of
that exact blob, so the blob must be reproducible byte-for-byte.

## Blobs

| file | tokens | bytes | contents |
|------|--------|-------|----------|
| `ckpts/tinystories-tokens.bin` | 473,992,236 | 947,984,472 | full canonical stream |
| `ckpts/ts-8L-128D.bin` | 16,319,744 | 32,639,488 | first 16,319,744 tokens of the same stream (prefix) |

Both are flat `uint16` little-endian gpt2 token ids, one contiguous stream.
`ckpts/` is a **real directory** and is git-ignored — the blobs are never
committed. (`ts-8L-128D.bin` is a prefix slice, not an independent tokenization;
verified by construction — see the generator.)

## Recipe

`tools/generate-tinystories-tokens.py` (committed) is the single source of
truth. It reconstructs the recipe that originally lived only inside
`tools/diloco-pytorch/data.py::_build_cache`:

- **dataset** `roneneldan/TinyStories`, split `train`
- **revision** `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (pinned — the 4-shard
  parquet train split cached on this host). The generator reads the cached
  parquet shards directly (sorted `train-00000..00003`), which reproduces the
  exact document order `load_dataset` yields while staying fully offline
  (`load_dataset(..., revision=...)` makes a network `dataset_info` call that
  fails under `HF_HUB_OFFLINE=1`).
- **order** document order as-published (no shuffle, no `select`)
- **tokenizer** `gpt2` (`GPT2TokenizerFast`), `add_special_tokens=False`
- **separator** append EOS (id `50256`) after **every** story so adjacent
  stories don't blend
- **dtype** `uint16` LE, flat

Deterministic: no randomness in blob construction. Rebuild with

```bash
HF_HUB_OFFLINE=1 uv run --with datasets==2.19.1 --with transformers --with numpy \
  python3 tools/generate-tinystories-tokens.py
```

## Recovery record (2026-07-18/19)

`ckpts/tinystories-tokens.bin` and `ckpts/ts-8L-128D.bin` were destroyed in a
symlink-clobber incident (a symlink replaced the real `ckpts/` directory). No
backups existed. Both blobs were regenerated with the script above.

**Verdict — CASE A, EXACT RESTORATION.** The regenerated full blob has
exactly 473,992,236 tokens (canonical count MATCH), and
`tools/diloco-regression-check.ts` reproduced the recorded baselines to ~1e-3:

| round | baseline | reproduced |
|-------|----------|-----------|
| 0 | 9.81 | 9.8089 |
| 3 | 5.92 | 5.9221 |
| 6 | 5.15 | 5.1534 |
| 9 | 4.64 | 4.6403 |

Peak GPU memory flat (0.0 MB growth, rounds 2–9). No re-basing needed;
`CANONICAL_TOKEN_COUNT` and `BASELINE` in the harness are unchanged.

The recipe above (cached-parquet direct read, revision-pinned, EOS-per-story)
is the confirmed reproduction. The natural full-train-split token count under
this recipe equals the canonical count exactly — the EOS separator after every
story is load-bearing for the count (dropping it would lose one token per
story, ~2.12M tokens).
