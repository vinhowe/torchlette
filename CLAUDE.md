# Torchlette

A WebGPU-accelerated tensor library for TypeScript with PyTorch-like semantics.

## Verification Policy

**Always verify changes with GPU tests.** This machine has GPU access. After any code change, run:

1. `npm run build` — must compile
2. `npm run test` — CPU + WebGPU tests (WebGPU auto-detected)
3. `npm run test:webgpu` — WebGPU-specific tests

WebGPU is auto-detected at runtime. Use `TORCHLETTE_CPU_ONLY=1` to force CPU-only mode.

For WebGPU backend changes, also run the relevant integration test (e.g. `npx tsx examples/gpt2/finetune-demo.ts` for training-related fixes).

**The load-bearing GPU correctness gates are now in-suite** (`test/compiled-plan-parity.spec.ts`, in the webgpu project — run `npm run test:gates` for just these): (1) compiled-plan trajectory == lowered trajectory over the full inner step (the frozen-step_size / clip-divergence class), (2) stage-4 stream generation matches the recording (no divergence), (3) stream recording is deterministic, (4) the chunked full-reduction sum (>128MB input) is correct AND fully generated (the lone chunked op the 124M plan hits; small-model gates never allocate >128MB, so this path is otherwise untested). They were previously manual-only `tools/` scripts that could silently rot. **CI (`ci.yml`) runs GPU-less so the whole webgpu project auto-skips there** — these gates only actually execute on a GPU box: locally via `npm run test`, or in CI via the `gpu-nightly.yml` workflow (needs a self-hosted `[self-hosted, gpu]` runner). If you change the executor / compiled plan / optimizer dispatch, run `npm run test:gates` before committing.

**Zero test failures policy.** Never accept test failures; fix before moving on. (Don't hand-maintain test counts here — run `npm run test` for the current numbers; as of 2026-06-12 it's ~1790 tests across ~137 file runs in the cpu+webgpu projects.)

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
- `test/` - test suite (cpu + webgpu projects run the same specs; see vitest.config.ts)
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

**Minimal training loops (2026-06-12):** `beginStep`/`endStep`/`markStep` are OPTIONAL in training loops. `optimizer.step()` queues a deferred step boundary that commits at the next `backward()` (or explicit `markStep()`); custom optimizers call `api.queueStepBoundary()` at the end of their `step()`. Explicit ceremony keeps its exact old semantics and supersedes queued boundaries. Differential gate: `test/implied-step-boundary.spec.ts` + `tools/t-implied-boundary-probe.ts`. Two rules the commit path must keep: (1) generation stamps are WRAPPER-level (storage-level stamps demote fused Adam's m/v — storage replaced every step); (2) quiesce (fence) BEFORE the demotion sweep — fencing after executes the sweep's deferred destroys under still-pending submits ("used in submit while destroyed").

GPU memory is managed deterministically via two-tier reachability — no GC dependency.

**At `beginStep()`**: all pending tensors are forced (materializes model weights on first call), then `snapshotForStep()` captures which RuntimeTensor objects are alive. These are "persistent" (model params, optimizer state).

**At `markStep()`**: after normal `destroyUnreachable()`, `destroyStepScoped()` demotes any reachable storage whose RuntimeTensor was NOT in the snapshot (created during the step as a temporary). This gives flat memory without depending on JavaScript GC timing.

**Adam m/v lifecycle**: The fused Adam kernel writes m/v to SEPARATE output buffers (not in-place). This means old and new m/v have independent buffer lifecycles. Adam uses `_updateLazyRef(createPendingRef(adamNode, outputIndex))` for m/v — the existing multi-output `outputIndex` infrastructure handles materialization automatically via `materializePendingTensors`. No protect/unprotect needed.

**Key invariant**: tensors created via public API (`tensorFromArray`, `zeros`, etc.) persist across steps because they're held by user code → in the snapshot. Tensors created by autograd-wrapped ops (activations, views, gradients) are step-scoped → cleaned up deterministically at markStep.

## Op Dispatch Architecture

**Table-driven ops**: Simple unary ops (relu, exp, sqrt, etc.), binary ops (add, mul, pow), comparison ops (gt, lt, etc.), reduction ops (sum, max, min, mean), and arg-reduce ops (argmax, argmin) are all table-driven via interface augmentation + prototype loop at the bottom of their respective files. This preserves TypeScript autocomplete while eliminating boilerplate.

**Custom backward extraction**: Complex backward functions (matmul, linear, gelu tanh, gelu erf) are extracted to `src/frontend/custom-backward.ts` with a `BackwardContext` interface. The frontend methods become thin dispatch stubs. Add new custom backward functions to this file following the same pattern.

**Gradient specs**: Simple ops (elementwise) define gradients in `src/ops/registry.ts` (UnaryGradFn, BinaryTTGradFn, BinaryTSGradFn). Complex ops define gradients in `custom-backward.ts`. Don't add complex backward logic to `torchlette.ts`.

## Framework Correctness Principles

**Canonical debt/architecture document: `docs/architecture-debt.md`** — the sin taxonomy every bug ledger entry maps to, the stage plan (scalars-as-data → foreach → islands → compile-from-IR), and the rules that should hold. Read it before architectural work.

**Single source of truth at seams; assert agreement.** Wherever two sides must agree on a value — a producer and a consumer on a buffer's shape/layout, a compiled/fused path and the naive path on a result, GPU and CPU on an op's semantics — derive that value from ONE source and assert the other matches *at the seam*. Never let both sides independently recompute it: when they silently diverge you get correct-looking-but-wrong results (the worst failure mode — no crash, just silently degraded training). Three instances this bit us, all now structurally guarded:
- **Row-program output layout** (`row-program-dispatch.ts`): the kernel's write count (scalar `[R,1]` vs full `[R,D]`) must equal the consuming node's `sizeOf(shape)`. The buffer is sized from the consumer's shape and the dispatch throws if the kernel's write count disagrees (caught → safe sequential fallback + warning). A misclassified `scalarOutput` once made every row collapse onto row 0's value under `compile()`.
- **Multi-output op consumers** (`graph-rewrites.ts` `structuralKey`, `fusion-detect.ts` `inputKey`): a pending ref's structural key MUST include `outputIndex`, else CSE merges consumers of different outputs of one node (SDPA dQ/dK collapsed onto dV). The canonical convention is `rewriter/matcher.ts` `refEqual` (compares node id AND outputIndex).
- **GPU vs CPU op semantics** (`pow`): WGSL `pow(x,y)=exp2(y·log2(x))` is NaN for x<0 while the CPU `**` reference is correct — so CPU tests can't catch GPU-only numeric bugs. Integer-exponent `pow` lowers to a mul chain at the frontend to match.
- **Compiled-plan replay vs per-step host values** (`tile-dispatch.ts` + `compiled-plan.ts`): the lowered path rewrites tile-IR uniform configs on every dispatch; a compiled replay binds the buffer's record-time contents. Any per-step-varying value (Adam's bias-corrected `step_size`, GradScaler's `inv_scale`, scheduled LR) must flow into replays as DATA — a tensor write (TAG_WRITE re-executes the source node) or a volatile uniform (TAG_UNIFORM re-packs from the current node's payload; `setAdamConfigUniforms` is the single source for the Adam mapping). Guarded at the seam: `getConfigBuffer` compares config bytes across executions and invalidates the recording (→ lowered fallback) when they changed with no volatile repack. The frozen `step_size` silently trained with the wrong LR schedule and was twice misattributed to "benign fp32 noise" before being found.

**Corollary — differentially test optimized paths against the naive one.** Fused / compiled / row-program / multi-output paths must be checked *numerically* against the plain (`enableFusion`-off, non-`compile()`, or CPU) path — not just "does it run." Every bug above was invisible to existing tests because they exercised only one path or ran on CPU; each fell out of a same-input cross-path diff (`tools/parity-forward-diff.ts`, `tools/compile-ln-repro.ts`, the jax-js bench). When you add a new optimized execution path, add a cross-path numerical guard with it.

**Corollary 2 — the differential must cross the optimization's ACTIVATION threshold.** The compiled plan only builds on the 2nd+ execution of a template and only covers what executes inside plans (the optimizer runs there too). Single-step parity tests and even fixed-weight multi-step probes (no `optimizer.step()`) both validated "the compiled plan" while never executing the part that was broken. The trajectory-level gate is `tools/parity-fullstack-tl.ts` run twice (`TORCHLETTE_COMPILED_PLAN=0` vs default) — per-step losses must agree to ~1e-5 over 30 steps. Run it after any compiled-plan / executor / optimizer-dispatch change.

## WebGPU Buffer Pool Invariants

**Do NOT flush `pendingRelease` to pool mid-step.** Causes deterministic numerical corruption (~2% loss drift). Root cause: buffers released by earlier plans may still be read by GPU from a prior command buffer.

**Safe reclamation patterns:**
- End-of-step flush (`endSharedEncoder()` → `flushPendingToAvailable()`)
- Periodic reclamation between plan segments (flush + pool flush as separate calls)
- The `sharedEncoderWriteSet` WAW check must be kept

**GPU buffer destruction is FENCE-GATED, never immediate.** Template eviction, plan invalidation, and table teardown all fire under memory pressure MID-STEP, while the step encoder holds encoded-but-unsubmitted passes binding the buffers — an immediate `buf.destroy()` poisons the pending submit (Dawn rejects it wholesale; downstream reads silently see stale data; three separate training-freeze bugs were this one class). Route ALL destruction through `bufferPool.deferredDestroy` (destroyed after the next fence). The device's `onuncapturederror` handler makes violations loud (`getGpuUncapturedErrorCount()`, `TORCHLETTE_STRICT_GPU=1` crashes).

**Use `bufferPool.canRecycle(buf)` for any "is it safe to reuse?" decision in a buffer cache.** It checks both ownership (`bufferLiveCount`) AND in-flight encoder claims (`sharedEncoderWriteSet`). The pool's own `acquire()` doesn't need to call it — bucket residents are guaranteed safe by construction (only populated post-fence/flush). But any cache outside the pool — the buffer arena, hint maps, or future caches — must consult it before recycling, or it can hand out a buffer whose queued reader hasn't dispatched yet. That's the bug class behind the "later step's data leaks into earlier step's results" symptom (see `test/lifetime-natural-usage.spec.ts`).

## Performance Baselines

### CURRENT — authoritative (A100 dw-2-1, 2026-06-14)
Default config (arena-liveness + memory planner + generated cutover + arena-reclaim).
Node/Dawn, batch 1, seq 512, steady-state (late-step, pool reuse settled):
- **DistilGPT-2 @512: ~50 ms/step, 4.67 GB peak (4.15 GB cur), 9 submits/step, zero leak.**
- **GPT-2 Medium @512: ~190 ms/step, 13.47 GB peak (11.96 GB cur), 18 submits/step, zero leak.**

arena-reclaim (commit 78c6f73) frees the dead per-position warmup arena buffers once a
plan reaches compiled/planner replay. vs the 2026-06-12 planner-default A100 line below
(5.07 GB / 15.0 GB), this is **distil −0.40 GB, medium −1.53 GB at unchanged speed**. 124M
DiLoCo regression (V100, with arena-reclaim): peak 3078→2754 MB, loss baselines
{0:9.81, 3:5.92, 6:5.15, 9:4.64}. Re-measure: `TORCHLETTE_PROFILE=1 TORCHLETTE_MODEL=…
TORCHLETTE_SEQ_LEN=512 NUM_STEPS=18 npx tsx tools/profile-training.ts` on dw-2-1 (host
node v18; the container node is too old). Read the LATE steps — the steady-state avg is
warmup-inflated; pool reuse must settle (~step 10+).

### HISTORICAL — NOT comparable to CURRENT
**⚠ Memory is NOT comparable across hardware (V100-32GB vs A100-80GB) or arena eras.**
The V100 lines below predate A100 + arena-reclaim. Trap to avoid: Medium 14.7 GB
(Baseline C, V100) and 13.8 GB (2026-06-11, V100) sit BELOW the A100 numbers — that is
V100-vs-A100 + different arena mechanisms, NOT a planner/arena regression. Compare only
within one hardware+era row.

### Baseline A: DistilGPT-2, 512 tokens, Node/Dawn (V100, HISTORICAL 2026-03-25)
Steady-state ~213ms/step wall clock. Memory: 1645MB steady, zero leak. Pool reuse 68%. 525 dispatches/step. Top GPU: matmul 40ms (61%), fusedAttentionBwd 5.7ms, adamStep 3.2ms (8 dispatches, packed). Fusion: 15.7% forward, 7.3% backward (limited by V100's 10 storage buffer limit).

### Baseline B: DistilGPT-2, 128 tokens, Browser/Chrome (V100)
**LoRA:** ~1050 tok/s peak, ~800 steady (GC degradation). fwd=5ms, bwd=80ms. Uses fused LayerNorm, Embedding, Linear, scaledDotProductAttention, api.linear(). Selective checkpointing (MLP-only). GPU-only gradient clipping (no CPU readback). Staging buffer loss readback overlaps with backward.

**Full FT:** ~308 tok/s peak, ~160 steady (GC degradation). fwd=80ms, bwd=171ms. No CPU fences in training loop — all computation stays lazy on GPU. Fence awaited in markStep for honest phase timing.

**Progressive slowdown:** Reduced from 50%/200 steps to 20%/1000 steps by eliminating RuntimeTensor wrappers for in-place op intermediates (mul_, zero_, fill_). Remaining ~0.8 RT/step from autograd-escaped forward tensors — GC-eligible but V8 collects slowly. GPU storage flat at 292.

### Baseline C: GPT-2 Medium, 512 tokens, Node/Dawn (V100, HISTORICAL)
Steady-state ~162ms/step wall clock. Memory: 14.7GB steady, zero leak. Pool reuse 58%. 741/4449 nodes fused (16.7%). Top GPU: bare matmul 125ms (bwd), epilogue matmul ~49ms (fwd) + ~47ms (bwd), fusedAttentionBwd 22.5ms, adamStep 14.3ms (14 dispatches, packed). GPU is 81% matmul.

### Re-measured 2026-06-10 → 06-12 (arena/compiled-plan era, HISTORICAL — superseded by CURRENT)
- **DistilGPT-2 512 (GradScaler+AdamW)**: ~56ms/step steady wall (98ms incl. warmup-skewed avg), 8 submits/step, adamStep 8 packed dispatches. Memory: **10.2GB** steady (unbudgeted arena — up from 1.6GB pre-arena; flat, zero leak). GPU ~68ms/step total, matmul family 66%.
- **DEFAULT MODE FLIPPED 2026-06-11**: the bounded (liveness) arena + planned compiled buffers is now THE DEFAULT (`TORCHLETTE_ARENA_LIVENESS=0` opts back into the legacy unbudgeted arena). New-default V100 (sivri) baselines: DistilGPT-2@512 **5.0GB peak at ~60ms/step**; GPT-2 Medium@512 **13.8GB peak at ~204ms/step** (legacy arena: 9.1GB→28.6GB respectively — Medium barely fit the 32GB V100); 124M DiLoCo regression at 2.85GB, baselines {0:9.81, 3:5.92, 6:5.15, 9:4.64}. Validated: full suite green in BOTH directions, browser suite green under the new default, fullstack canonical both directions, 4-peer DiLoCo soak (loss 5.088 @ 25 rounds, better than lowered control and the May baseline). History: planned compiled buffers landed 2026-06-10 (replays pin and rebind the recorded pool-buffer assignment; the liveness-lowered interim ran ~525ms/step).
- **MEMORY PLANNER DEFAULT 2026-06-12 (stage-4 phase 1.5)**: compiled-replay buffer assignment is now DERIVED by the graph-liveness memory planner with cross-plan temp packing (step-scoped shared `PlannerRegistry`), replacing the pin-the-recorded-buffers mechanism (deleted: adoption refcounts, pool-origin tracking, allocBuffers, planned-bind fallbacks, `bufferPool.adoptBuffer`). `TORCHLETTE_MEMORY_PLANNER=0` disables compiled replay wholesale (lowered path) — dynamic-alloc replay would leak ownerless temps. A100 (dw-2-1, same-machine A/B vs pin): DistilGPT-2@512 **5.07GB** (pin 5.88), Medium@512 **15.0GB** (pin 17.5), speed parity (~51ms / ~180ms per step, identical submits); the V100 numbers above predate this and are historical. See `docs/stage4-compile-from-ir.md` phase 1.5.
- Optimizer batching fix: lowered-plan adam-batch grouping hoists interleaved per-param producers (unscaleGrad / clip's stridedScatterCopy) above one big batch (`ADAM_HOISTABLE_OPS`); previously every batch had size 1 (158 submits/step, no packing). Debug: `TORCHLETTE_DEBUG_OPTPLAN=1` dumps optimizer plan op order.
- Compiled replay now preserves profiler attribution (label/module recorded per dispatch — previously 83% of GPU time showed as "unknown").

## Open Performance Targets

### Browser training targets (ranked by impact)

1. **Fix progressive slowdown** — ~2 RuntimeTensors/step accumulate because V8 GC doesn't collect them promptly. After ~200 steps, LoRA drops from 1050→800 tok/s. Root cause: `mul_` intermediates and transpose views created outside tidy scopes are GC-eligible but not promptly collected. Skipping RuntimeTensor wrappers for intermediates leaks GPU storage (FinalizationRegistry needs the wrapper). Needs a different approach — possibly explicit disposal of in-place op intermediates after plan execution.

2. **Add ternary op support to tile-IR** — `clamp(x, min, max)` is a WGSL built-in but not in the op registry. clipGradNorm works around this with le/gt/mul/add (4 ops instead of 1). Adding ternary op support would simplify this and enable other three-input ops.

3. **Fuse bias-gradient sums** — 244 sum dispatches in Full FT (22% of GPU time). Each `dBias = sum(gradOut, dims)` is a separate dispatch. Would need multi-output reduction support in tile-IR to batch independent same-dim reductions.

### Node/Dawn targets (GPT-2 Medium, ranked)

4. **Per-shape matmul autotuning** — Infrastructure exists (`TORCHLETTE_AUTOTUNE=1`). Pre-seed cache for Medium shapes. ~5-10ms potential from sub-optimal tile configs on 1024-embed shapes.

5. **Backward elementwise fusion** — 7.3% fusion rate on V100 was limited by V100's 10 storage-buffer limit. On A100 (CURRENT default hardware) that limit is gone, yet the 2026-06-14 profile still shows many unfused ops (bwd top: reshape:87, matmul:56, transpose:31, permute:24, sum:24, narrowBackward:19, add:19). So on A100 the limiter is the **consecutive-only fusion detector**, not the buffer count — re-investigate the detector (graph-aware grouping already exists; the gap is non-adjacent fusible runs broken by views/bypassed nodes). The old "not actionable on V100" verdict no longer applies on A100.

### Framework completeness targets

6. ~~**LR Schedulers**~~ — DONE. `src/optim/lr-scheduler.ts`: StepLR, ExponentialLR, CosineAnnealingLR, PolynomialLR (exported from optim/index).
7. **Weight initialization** — kaiming_normal_, xavier_uniform_. ~200 lines. (nn.Linear already uses PyTorch kaiming_uniform default init; this is the standalone init API.)
8. ~~**Gradient clipping**~~ — Implemented. Fully GPU (no CPU readback).
9. ~~**Parameter groups**~~ — DONE. `AdamParamGroup` / `SGDParamGroup` (per-group LR/weightDecay) in adam.ts / sgd.ts.

### Architecture targets (long-term)

10. **Serializable compiled plans** — Pre-compile full dispatch sequence to disk. Eliminates ~700ms cold start.
11. **Forward/backward overlap via loss tensor preservation** — 4 approaches tried, all failed (see "What didn't work"). Needs dedicated readback staging buffer excluded from pool — `startScalarReadback` primitive added but only helps for loss readback, not clipGradNorm (which is now fully GPU anyway).

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
- ~~**Disposing intermediates breaks autograd**~~ — FIXED (scoped-memory stage 1). The autograd graph now INDEPENDENTLY retains its saved-for-backward tensors (PyTorch semantics): each non-hook saved slot holds a graph-owned, scope-independent RuntimeTensor (`slot.retained`, `Tensor._cloneForRetention`) that keeps the saved value's storage alive via its own rc until backward — released symmetrically in `cleanupAutogradGraph`, the backward error-path `finally`, or GC. `dispose()` no longer tears down the upstream autograd graph (the old `_disposeAutogradChain` was removed), so disposing an intermediate handle — or forward-in-`tidy`/step + backward-outside — produces correct gradients. Gate: `test/autograd-scope-lifetime.spec.ts`.
- **No double backward** — Calling `backward()` clears the autograd graph. A second call does nothing (no `retain_graph` equivalent). This is by design for memory efficiency.

## Complexity budget (the weight-norm)

Keep the framework small and legible; regularize against overfit mechanisms.
- `bash tools/weight-norm.sh --log` prints/appends the size vector (src SLOC —
  code-only, comments are free; docLOC tracked as its own positive signal;
  files, public exports, env flags). A PostToolUse hook in `.claude/settings.json`
  (untracked — recreate if absent) self-reports the vector after every
  `git commit`. Snapshot at every campaign-end commit;
  a campaign that grows the vector names what it deleted or why net-new
  mechanism is warranted.
- Every generalization NAMES ITS DELETIONS in the commit message (house
  tradition, now policy).
- Every new TORCHLETTE_* flag is born with a sunset: soak → default → the
  opt-out dies. Flags outliving their campaign are debt.
- One-sentence test: if a feature can't be stated as a declaration in one
  sentence of the paradigm docs, reshape it before landing.
- Admission pressure: src/ is the diamond. Code proves itself in examples/
  or packages/ first; it enters src/ only when it has earned generality
  (a declaration, a registry entry, an IR primitive) — if it isn't ready,
  hold it out until it is.
