# Torchlette

A WebGPU-accelerated tensor library for TypeScript with PyTorch-like semantics.

## Verification Policy

**Always verify changes with GPU tests.** This machine has GPU access. After any code change, run:

1. `npm run build` — must compile
2. `npm run test` — CPU + WebGPU tests (WebGPU auto-detected)
3. `npm run test:webgpu` — WebGPU-specific tests

WebGPU is auto-detected at runtime. Use `TORCHLETTE_CPU_ONLY=1` to force CPU-only mode.

For WebGPU backend changes, also run the relevant integration test (e.g. `npx tsx examples/gpt2/finetune-demo.ts` for training-related fixes).

**Zero test failures policy.** Currently 841 tests pass across 60 test files. Never accept test failures. Fix before moving on.

**Important:** Any standalone script or tool that uses WebGPU (Dawn) must call `process.exit(0)` at the end of `main()`. Dawn holds background threads that prevent Node from exiting naturally.

## Development Commands

```bash
npm run build          # Build the library
npm run test           # Run all tests (CPU + GPU projects concurrently)
npm run test:coverage  # Run all tests with Istanbul coverage
npm run lint           # Run linter
npm run test:webgpu    # WebGPU-specific tests (Node.js/Dawn)
npm run test:browser   # Browser tests (Playwright)

# Profiling (V100)
TORCHLETTE_PROFILE=1 npx tsx tools/profile-training.ts
TORCHLETTE_MODEL=gpt2-medium TORCHLETTE_SEQ_LEN=512 TORCHLETTE_PROFILE=1 npx tsx tools/profile-training.ts

# Benchmarks
BENCH_WARMUP=3 BENCH_ITERS=7 npx tsx bench/matmul-comparison.ts
```

## Project Structure

- `src/backend/webgpu/` - WebGPU backend (dispatch, buffer pool, pipeline warmup)
  - `ops/` - Op implementations (elementwise, reductions, views, fused kernels, registry)
  - `matmul/` - Tiled matmul with shape-class tuning and K-split
  - `tile-*.ts` - Tile-IR compiler (IR, lowering, compiler, ops, dispatch)
- `src/engine/` - Tensor engine core (lazy execution, graph compiler, fusion, plan building)
- `src/frontend/` - User-facing API (table-driven ops, autograd, autocast)
- `src/runtime/` - RuntimeEngine (lazy IR node creation, dtype rules)
- `src/nn/` - Module system (auto parameters, linear, embedding, layernorm)
- `src/optim/` - Optimizers (Adam/AdamW with fused GPU kernel, GradScaler)
- `test/` - 60 test files, 841 tests
- `examples/gpt2/` - GPT-2 model, loader, tokenizer, finetune demo
- `tools/profile-training.ts` - GPU training profiler (supports distilgpt2, gpt2, gpt2-medium)

## Spec Compliance Summary

| Category | Status |
|----------|--------|
| Core engine semantics (§1, §3, §6) | Implemented |
| Token algebra & planning | Implemented |
| Compiled regions (§8) | Partial (caching only; §8.3-8.11 removed) |
| Autograd (§9) | Implemented |
| Checkpointing (§10) | Implemented |
| RNG (§11) | Implemented |
| AMP (§12) | Implemented |
| Cross-device (§13) | Removed (unused) |
| Memory planning (§14) | Removed (replaced by arena + pool) |
| Elementwise fusion (§15.1-15.3) | Implemented |

Known divergences: `number` not `bigint` for versions, 2 LocRoles not 6, no tombstones.

## WebGPU Buffer Pool Invariants

**Do NOT flush `pendingRelease` to pool mid-step.** Causes deterministic numerical corruption (~2% loss drift). Root cause: buffers released by earlier plans may still be read by GPU from a prior command buffer.

**Safe reclamation patterns:**
- End-of-step flush (`endSharedEncoder()` → `flushPendingToAvailable()`)
- Periodic reclamation between plan segments (flush + pool flush as separate calls)
- The `sharedEncoderWriteSet` WAW check must be kept

## Performance Baselines

### Baseline A: DistilGPT-2, 512 tokens (6 layers, 768 dim, 81M params)
Steady-state ~51ms/step wall clock. GPU: ~66ms total (forward 17ms, backward 47ms, cleanup 5ms). Top kernels: `matmul` 30ms (45%), epilogue matmul 15ms (22%), `fusedAttentionBackward` 5.4ms (8%), `adamStep` 5ms (7%), `fusedAttentionForward` 2.5ms (4%). Memory: 5529MB steady, zero leak. Pool reuse 57%. Bind group cache 99.7%.

### Baseline B: GPT-2 Medium, 512 tokens (24 layers, 1024 dim, 355M params)
Steady-state ~265-330ms/step wall clock. GPU: ~315ms total (forward 71ms, backward 223ms, cleanup 21ms). Top kernels: `matmul` 131ms (42%), `matmul++cast+bias+binary` 45ms (14%), `matmul++cast+bias+unary+cast` 33ms (10%), `fusedAttentionBackward` 22.5ms (7%), `adamStep` 21.6ms (7%), `matmul++cast+bias` 18ms (6%), `fusedLNBackwardGradWeightBias` 9.8ms (3%), `fusedAttentionForward` 9.7ms (3%). Memory: 15.3GB steady, zero leak. Bind group cache 99.7%. Overall matmul FP32 efficiency: ~30%.

## Open Performance Targets

### GPT-2 Medium targets (ranked by estimated GPU savings)

1. **Re-implement packed Adam** — 292 individual adamStep dispatches = 21.6ms GPU. Packed Adam (group same-element-count params, one dispatch per size class) was implemented, reduced to ~8 dispatches (5.4→1.9ms), but code was deleted as "superseded by adam-batch." Adam-batch only pre-flushes, doesn't pack. Need to re-implement. **~14ms savings.**

2. **Matmul tile tuning for 1024-embed** — Benchmark shows 64×64×16 t4×4 outperforms current 64×128×16 t8×4 for dX shapes (M=512, K large) by 29-31%. Applied M≤512 && K≥2M heuristic in square_large bare path, plus fixed square_medium epilogue (32×32→64×64). Measured ~7ms improvement. Full per-shape autotuning (infrastructure exists, `TORCHLETTE_AUTOTUNE=1`) could recover more of the benchmarked 23ms gap.

3. **LN backward gradW/B kernel** — 9.8ms (3%), 98 dispatches at 100µs avg. Cross-row reduction over [512,1024] scales with embedDim. Could optimize with wider workgroups or multi-pass reduction. **~3-5ms savings.**

4. **Fuse bias-gradient sums** — 96 sum dispatches (5.1ms) from dBias = sum(dY, dim=0). Each followed by reshape. Extend `detectReductionEpilogue()` to handle reshape as valid epilogue. **~2-3ms savings.**

### General targets (low priority)

5. **GC pressure** — ~2.7ms worst-case for 1,200 Tensor metadata objects + full GC. Not a bottleneck at current scale (<5% of step budget). Would need object pooling if model size grows significantly.
6. **Pipeline warmup** — Step 0 is ~700ms (not 1.6s). Warmup infra exists (`pipeline-warmup.ts`), pre-compiles 13/57 pipelines on second run in 86ms. One-time cost, diminishing returns.
7. **Single-plan training step** — Merge forward/backward/optimizer into one plan. Requires §8 compiled-region infrastructure (removed as unused). **~2-3ms savings.** Punted — complexity outweighs benefit.

## What didn't work (don't re-attempt)

- **Vec4 shared memory for matmul K-loop** — 3 approaches benchmarked, all regressed 9-36%. Scalar shared mem is faster for shared×shared dot. Vec4 kept for attention (register×shared dot).
- **Double-buffered K-loop** — 2× shared memory reduces occupancy, outweighing barrier reduction. −13%.
- **Fused single-pass dQ+dKV** — Requires f32 atomics (WebGPU only has i32/u32).
- **Inline D precompute into attention** — D precompute is only 0.3ms; inline overhead exceeds savings.
- **Per-shape matmul autotuning on DistilGPT-2** — Hand-tuned defaults already optimal for 768-dim. Infrastructure kept for larger models.
- **Matmul input cast absorption in training** — Backward needs f16 tensors materialized; can't skip the cast.
