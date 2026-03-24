# Torchlette

A WebGPU-accelerated tensor library for TypeScript with PyTorch-like semantics.

## Verification Policy

**Always verify changes with GPU tests.** This machine has GPU access. After any code change, run:

1. `npm run build` — must compile
2. `npm run test` — CPU + WebGPU tests (WebGPU auto-detected)
3. `npm run test:webgpu` — WebGPU-specific tests

WebGPU is auto-detected at runtime. Use `TORCHLETTE_CPU_ONLY=1` to force CPU-only mode.

For WebGPU backend changes, also run the relevant integration test (e.g. `npx tsx examples/gpt2/finetune-demo.ts` for training-related fixes).

**Zero test failures policy.** Currently 852 tests pass across 55 test files. Never accept test failures. Fix before moving on.

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
- `src/frontend/` - User-facing API (table-driven ops, autograd, autocast, noGrad)
  - `custom-backward.ts` - Extracted backward functions (matmul, linear, gelu) with BackwardContext
- `src/runtime/` - RuntimeEngine (lazy IR node creation, dtype rules, table-driven ops)
- `src/nn/` - Module system (auto parameters, linear, embedding, layernorm, init, grad clipping)
- `src/optim/` - Optimizers (Adam/AdamW with fused GPU kernel, SGD, GradScaler, LR schedulers, parameter groups)
- `test/` - 55 test files, 852 tests
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

## Step-Scoped Storage Cleanup

GPU memory is managed deterministically via two-tier reachability — no GC dependency.

**At `beginStep()`**: all pending tensors are forced (materializes model weights on first call), then `snapshotForStep()` captures which RuntimeTensor objects are alive. These are "persistent" (model params, optimizer state).

**At `markStep()`**: after normal `destroyUnreachable()`, `destroyStepScoped()` demotes any reachable storage whose RuntimeTensor was NOT in the snapshot (created during the step as a temporary). This gives flat memory without depending on JavaScript GC timing.

**Adam m/v lifecycle**: The fused Adam kernel writes m/v to SEPARATE output buffers (not in-place). This means old and new m/v have independent buffer lifecycles. Adam uses `_updateLazyRef(createPendingRef(adamNode, outputIndex))` for m/v — the existing multi-output `outputIndex` infrastructure handles materialization automatically via `materializePendingTensors`. No protect/unprotect needed.

**Key invariant**: tensors created via public API (`tensorFromArray`, `zeros`, etc.) persist across steps because they're held by user code → in the snapshot. Tensors created by autograd-wrapped ops (activations, views, gradients) are step-scoped → cleaned up deterministically at markStep.

## Op Dispatch Architecture

**Table-driven ops**: Simple unary ops (relu, exp, sqrt, etc.), binary ops (add, mul, pow), comparison ops (gt, lt, etc.), reduction ops (sum, max, min, mean), and arg-reduce ops (argmax, argmin) are all table-driven via interface augmentation + prototype loop at the bottom of their respective files. This preserves TypeScript autocomplete while eliminating boilerplate.

**Custom backward extraction**: Complex backward functions (matmul, linear, gelu tanh, gelu erf) are extracted to `src/frontend/custom-backward.ts` with a `BackwardContext` interface. The frontend methods become thin dispatch stubs. Add new custom backward functions to this file following the same pattern.

**Gradient specs**: Simple ops (elementwise) define gradients in `src/ops/registry.ts` (UnaryGradFn, BinaryTTGradFn, BinaryTSGradFn). Complex ops define gradients in `custom-backward.ts`. Don't add complex backward logic to `torchlette.ts`.

## WebGPU Buffer Pool Invariants

**Do NOT flush `pendingRelease` to pool mid-step.** Causes deterministic numerical corruption (~2% loss drift). Root cause: buffers released by earlier plans may still be read by GPU from a prior command buffer.

**Safe reclamation patterns:**
- End-of-step flush (`endSharedEncoder()` → `flushPendingToAvailable()`)
- Periodic reclamation between plan segments (flush + pool flush as separate calls)
- The `sharedEncoderWriteSet` WAW check must be kept

## Performance Baselines (2026-03-24)

### Baseline A: DistilGPT-2, 512 tokens, Node/Dawn (V100)
Steady-state ~213ms/step wall clock. Memory: 1645MB steady, zero leak. Pool reuse 68%. 525 dispatches/step. Top GPU: matmul 40ms (61%), fusedAttentionBwd 5.7ms, adamStep 3.2ms (8 dispatches, packed). Fusion: 15.7% forward, 7.3% backward (limited by V100's 10 storage buffer limit).

### Baseline B: DistilGPT-2, 128 tokens, Browser/Chrome (V100)
**LoRA:** 316 tok/s steady. fwd=4ms, bwd=307ms. Uses fused LayerNorm, Embedding, Linear, scaledDotProductAttention, api.linear(). Selective checkpointing (MLP-only). GPU memory flat.

**Full FT:** 89 tok/s steady. fwd=16ms, bwd=1100ms. GPU-bound at ~786ms/step GPU compute. Fence wait moved to markStep for honest phase timing + proper CPU/GPU overlap (was 47 tok/s before fence fix).

### Baseline C: GPT-2 Medium, 512 tokens, Node/Dawn (V100)
Steady-state ~162ms/step wall clock. Memory: 14.7GB steady, zero leak. Pool reuse 58%. 741/4449 nodes fused (16.7%). Top GPU: bare matmul 125ms (bwd), epilogue matmul ~49ms (fwd) + ~47ms (bwd), fusedAttentionBwd 22.5ms, adamStep 14.3ms (14 dispatches, packed). GPU is 81% matmul.

## Open Performance Targets

### Remaining GPU targets (GPT-2 Medium, ranked)

1. **Per-shape matmul autotuning** — Infrastructure exists (`TORCHLETTE_AUTOTUNE=1`). Pre-seed cache for Medium shapes. ~5-10ms potential from sub-optimal tile configs on 1024-embed shapes.

2. **Fuse bias-gradient sums** — 96 sum dispatches (5.1ms) from dBias = sum(dY, dim=0). Each followed by reshape. Extend reduction epilogue detector to handle reshape as valid epilogue. **~2-3ms savings.**

3. **Backward elementwise fusion** — 16.7% fusion rate. 97 standalone adds, 26 muls in backward. Graph reorder priority tuning may recover more fusion opportunities.

### Framework completeness targets (high user impact)

4. **LR Schedulers** — StepLR, CosineAnnealingLR, etc. ~300 lines, pure math, zero risk.
5. **Weight initialization** — kaiming_normal_, xavier_uniform_. ~200 lines.
6. ~~**Gradient clipping**~~ — Implemented (clip_grad_norm_, clip_grad_value_).
7. **Parameter groups** — per-layer LR in Adam/SGD. ~150 lines.

### Architecture targets (long-term)

8. **Serializable compiled plans** — Pre-compile full dispatch sequence to disk. Eliminates ~700ms cold start. Pipeline warmup already serializes individual pipelines; extend to full dispatch sequence.
9. **Single-plan training step** — Merge forward/backward/optimizer into one plan. Punted — complexity outweighs benefit at current scale.

## What didn't work (don't re-attempt)

- **Vec4 shared memory for matmul K-loop** — 3 approaches benchmarked, all regressed 9-36%. Scalar shared mem is faster for shared×shared dot. Vec4 kept for attention (register×shared dot).
- **Double-buffered K-loop** — 2× shared memory reduces occupancy, outweighing barrier reduction. −13%.
- **Fused single-pass dQ+dKV** — Requires f32 atomics (WebGPU only has i32/u32).
- **Inline D precompute into attention** — D precompute is only 0.3ms; inline overhead exceeds savings.
- **Per-shape matmul autotuning on DistilGPT-2** — Hand-tuned defaults already optimal for 768-dim. Infrastructure kept for larger models.
- **Matmul input cast absorption in training** — Backward needs f16 tensors materialized; can't skip the cast.
- **Bypassed-node transparency in fusion grouping** — Making CSE-bypassed nodes transparent in `buildCandidateGroups` (so they don't break fusible runs) caused "Input not ready" errors in the browser. The bypassed nodes' plan positions still matter for execution ordering. Bypassed nodes must remain opaque barriers.
- **Skipping RuntimeTensor for in-place op intermediates** — `mul_` creates `copy_(dst, mul(dst, value))` with an intermediate RuntimeTensor. Tried creating the mul lazy node directly without a RuntimeTensor wrapper to avoid GC pressure. Caused GPU storage leak — the intermediate's StorageHandle had no RuntimeTensor owner, so FinalizationRegistry never cleaned it up.
- **Moving loss.item() after backward** — Tried 4 approaches to overlap forward GPU with backward CPU: (1) `retainGrad()` doesn't prevent `cleanupAutogradGraph` disposal. (2) Adding root to `preserved` set causes buffer aliasing errors (shared encoder read-write conflict). (3) Concurrent `item()` promise + backward hits "Engine is busy" (exec lock). (4) `force()` then non-awaited `runtime.item()` causes data race — backward reuses the loss buffer before `mapAsync` reads it, returning garbage values. Safe overlap requires a dedicated readback staging buffer excluded from the pool.

## Known Semantic Limitations

- **In-place self-aliasing crashes** — `tensor.mul_(tensor)` (same tensor as both operands) creates a broken lazy graph node. `add_(self)` works because the lazy system doesn't read-after-write, but `mul_` needs both input values before writing. Use `api.mul(a, a)` (non-in-place) instead. Not worth fixing: would require aliasing detection on every in-place op for a case that never occurs in real training.
- **Disposing intermediates breaks autograd** — If you manually `dispose()` a tensor that's part of an active backward graph, the saved-for-backward reference is killed and gradients will be silently wrong or null. Unlike PyTorch, saved tensors share lifecycle with the user handle (no independent refcounting). In practice this doesn't occur: `tidy()`, `compile()`, and `beginStep()`/`endStep()` manage lifecycle automatically. Don't manually dispose tensors between forward and backward.
- **No double backward** — Calling `backward()` clears the autograd graph. A second call does nothing (no `retain_graph` equivalent). This is by design for memory efficiency.
