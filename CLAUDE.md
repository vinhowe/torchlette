# Torchlette

A WebGPU-accelerated tensor library for TypeScript with PyTorch-like semantics.

## Verification Policy

**Always verify changes with GPU tests.** This machine has GPU access. After any code change, run:

1. `npm run build` — must compile
2. `npm run test` — CPU + WebGPU tests (WebGPU auto-detected)
3. `npm run test:webgpu` — WebGPU-specific tests

WebGPU is auto-detected at runtime. Use `TORCHLETTE_CPU_ONLY=1` to force CPU-only mode.

For WebGPU backend changes, also run the relevant integration test (e.g. `npx tsx examples/gpt2/finetune-demo.ts` for training-related fixes).

**Zero test failures policy.** There are currently no known test failures — all 1286 tests pass (1 skipped). Never accept test failures. If a test fails after your change, fix it before moving on. Do not skip, disable, or mark tests as expected-to-fail. If a test is genuinely flaky (e.g. GPU timing), investigate and fix the flakiness rather than ignoring it.

**Important:** Any standalone script or tool that uses WebGPU (Dawn) must call `process.exit(0)` at the end of `main()`. Dawn holds background threads that prevent Node from exiting naturally.

## Specification Documents

- **Working Spec**: [torchlette-working-spec.md](torchlette-working-spec.md) - Full runtime semantics (v1.22)
- **Testing Spec**: [torchlette-testing-spec.md](torchlette-testing-spec.md) - Test-driven development plan

## Development Commands

```bash
npm run build          # Build the library
npm run test           # Run unit tests
npm run lint           # Run linter

# WebGPU tests (auto-detected; skip with TORCHLETTE_CPU_ONLY=1)
npm run test:webgpu                         # Node.js (Dawn)
npm run test:browser                        # Browser (Playwright)

# Benchmarks
BENCH_WARMUP=3 BENCH_ITERS=7 npx tsx bench/matmul-comparison.ts
npm run bench:browser  # Then open http://localhost:8080/bench/browser/
```

## Profiling

### GPU Profiling (per-op/per-module/per-phase GPU timing)

```bash
# Run with GPU timestamp queries enabled
TORCHLETTE_PROFILE=1 npx tsx tools/profile-training.ts

# Output: per-phase timing, per-module GPU timing, per-op kernel timing,
# fusion stats, memory stats. Also writes JSON to /tmp/torchlette-profile-step4.json
```

### CPU Profiling (V8 CPU profiler with source-mapped attribution)

```bash
# Step 1: Bundle with esbuild (source maps needed for attribution)
npx esbuild tools/profile-training.ts --bundle --platform=node --sourcemap \
  --outfile=/tmp/profile-bundle.mjs --format=esm --external:webgpu

# Step 2: Symlink model files (bundle runs from /tmp)
ln -sfn "$(pwd)/models/distilgpt2" /tmp/models/distilgpt2

# Step 3: Run with V8 CPU profiler + GPU profiling
TORCHLETTE_PROFILE=1 node --cpu-prof --cpu-prof-dir=/tmp --cpu-prof-interval=100 \
  /tmp/profile-bundle.mjs

# Step 4: Analyze with source-map resolution (steady-state window: 70-95% of samples)
node tools/analyze-cpu-profile.cjs /tmp/CPU.*.cpuprofile
```

The analyze script (`tools/analyze-cpu-profile.cjs`) uses `@jridgewell/trace-mapping` to map V8 sample positions back to original TypeScript source files/line numbers, and reports self-time percentages for the steady-state window.

### Profile Analysis Guide

Use `/profile` to run the profiler and produce a full report automatically. The report should cover:

**Key metrics to extract from profiler output:**

1. **Wall clock** — Per-step timings from the WALL CLOCK SUMMARY table. Steady-state = step 4 (steps 0-2 have compilation/pool warmup). Step 0 is pipeline warmup (expect 100x+ slower). Breakdown: forward, backward, optimizer, cleanup as % of total.

2. **GPU time budget** — From the Phase table (`Phase | Ops | CPU(ms) | GPU(ms)`). Sum GPU(ms) across forward+backward+cleanup for total GPU time. At 512 tokens the workload is GPU-bound (145ms GPU vs 54ms CPU dispatch). At 31 tokens it was CPU-dispatch-bound (~85% utilization).

3. **Top GPU kernels** — From `GPU Kernel Time` table. Rank by Total(ms). At 512 tokens: `matmul` (37.7ms, 36.5%), `fusedAttentionBackward` (15.7ms, 15.2%), `fusedAttentionForward` (12.4ms, 12.0%), `matmul++cast+bias+binary` (8.7ms, 8.4%), `matmul++cast` (8.1ms, 7.9%), `matmul++cast+bias` (5.6ms, 5.4%), `adamStep` (4.9ms, 4.7%).

4. **Cross-entropy** — Fused kernel implemented. 1.1ms/1.1% of GPU time at 512 tokens. No longer a bottleneck.

5. **CPU dispatch overhead** — From `CPU API Call` table. Bind group cache at 100.0% (0 misses). 10 submits/step.

6. **Fusion rate** — From Plan Analysis. Total: 1124 nodes, 183 fused (16.3%), 20 fusion groups, 645 sequential. Note: fused LayerNorm/cross-entropy/attention ops replace many decomposed ops each, so effective fusion is higher than the raw percentage.

7. **Memory** — From Memory Stats section. Track: current/peak MB, buffer pool reuse rate. At 512 tokens: 4871MB current, 6194MB peak, pool reuse 56.8%, 1 new alloc/step steady-state.

8. **Leak status** — From MEMORY LEAK REPORT. Storages and reachable should stabilize. PendingDestroy should not grow. CurrentMB should not monotonically increase. "LEAK STATUS: OK" = pass.

**When to update baselines:** If steady-state ms/step drifts >10% from the value in "Remaining Performance Optimizations", or if a bottleneck's % share changes significantly, update CLAUDE.md.

## Project Structure

- `src/` - Source code
  - `backend/webgpu/` - WebGPU backend
  - `backend/webgpu/matmul/` - Tiled matmul with autotuning
  - `engine/` - Tensor engine core
  - `frontend/` - User-facing API
- `test/` - Test suites
- `bench/` - Benchmarks

## Current Implementation Status

### Engine Core (src/engine/) - COMPLETE
- Token algebra with afterAll, join rule (§3 of spec)
- Version tracking: locLogicalVersion, locVersion, baseCommitVersion
- Plan building with deterministic linearization (PlanLinearOrder)
- markStep() with token reset, loc finalization
- Execution lock for non-reentrancy
- RNG system with keyed randomness, checkpoint replay (§11)
- Checkpoint infrastructure with purity fences (§10)
- **Lazy execution** (src/engine/lazy.ts, src/runtime/engine.ts):
  - LazyRef/LazyIRNode types for pending computations
  - buildPlan() for topological ordering
  - executePlan() for backend dispatch
  - RuntimeEngine ops create LazyIRNodes (deferred execution)
  - Force boundaries: cpu(), item(), markStep()
- **Compiled region caching** (src/engine/compile-cache.ts):
  - Normalized IR structural hashing (§8.2)
  - Cache key: IR hash + input signatures (shapes/dtypes)
  - LRU eviction with configurable max size
  - Cache hit tracking for diagnostics
- **Advanced compiled region features** (src/engine/compiled-region.ts):
  - Arg alias groups (§8.3): Track aliased input arguments
  - StateIfaceSig (§8.4): Ordered state access signatures
  - State-slot alias patterns (§8.5): Bind-time alias tracking
  - Auto-externalize (§8.7): SSA to pending_loc conversion
  - Null-state sentinels (§8.8): Stable missing state representation
  - Functionalization (§8.10): In-place to out-of-place conversion
  - Region-exit persistence (§8.11): Commit and writeback tracking
- **IR optimization** (src/engine/ir-optimize.ts):
  - CSE with RNG exclusion (§15): Random ops never merged
  - Dead code elimination: Removes unreachable nodes
  - tok_after analysis (§15): Redundant load detection
- **§15 Integration** (src/engine/lazy-to-ir.ts, src/engine/fusion-detect.ts):
  - Lazy plan to IR conversion: Convert LazyIRNode graphs to IRGraph
  - Fusion group detection: Identify consecutive elementwise ops
  - Optimized execution: `executePlanOptimized()` with automatic fusion
  - RuntimeEngine integration: `enableFusion` option for opt-in fusion
  - Segmented execution: Mix fused and sequential execution for non-fusible ops
- **Memory planning** (src/engine/memory-planning.ts, src/engine/memory-planned-executor.ts) - COMPLETE:
  - In-flight plan strong rooting (§14): Plans hold refs to all touched tensors
  - Allocator fencing: Track GPU completion before buffer reuse
  - Lifetime analysis: Determine when tensors can be freed
  - Memory donation: Reuse input buffers for outputs when safe
  - Buffer pooling: Pre-allocate and reuse buffers by size class
  - RuntimeEngine integration: `enableMemoryPlanning` option for opt-in buffer pooling
  - GPU buffer pool (src/backend/webgpu/buffer-pool.ts): WebGPU-specific buffer management
- **Cross-device execution** (src/engine/cross-device.ts, §13):
  - Lazy transfer op: `tensor.to(device)` defers transfer until forced
  - Transfer path resolution: Determines optimal route (via_cpu, direct, noop)
  - Multi-device graph analysis: Detects cross-device ops and transfer points
  - Auto-transfer support: `ensureSameDevice()` for ops with mixed device inputs
  - Transfer statistics: Track bytes transferred and paths used

### Frontend API (src/frontend/) - COMPLETE
- Tensor class with full PyTorch-like API
- tidy/keep/dispose lifecycle management
- Autograd with async backward pass, saved-for-backward
- Gradient accumulation and version checking
- Lazy execution integrated: ops return immediately, force at cpu()/item()

### Backends (src/backend/)
- **CPU**: Full reference implementation of all ops
- **WebGPU**: Optimized tiled matmul with:
  - Shared memory tiling (32x32, 64x64, etc.)
  - Subgroup acceleration (when hardware supports)
  - f16/f32 mixed precision
  - ND batched broadcasting
  - Epilogue fusion (bias, relu, gelu, silu)
  - Runtime autotuning

### Fused Kernel Generation (src/backend/webgpu/fusion-*.ts) - COMPLETE
- **Expression-based SSA codegen**: Single source of truth for op → WGSL expression mapping
- **Elementwise fusion**: Arbitrary chains of unary/binary ops fused into single kernel
- **Broadcasting support**: Automatic broadcast index calculation for mixed shapes
- **Kernel caching**: LRU cache for compiled pipelines by recipe signature
- **Supported ops**: relu, gelu, silu, sigmoid, tanh, neg, abs, exp, log, sqrt, add, sub, mul, div, pow, min, max, comparisons, casts

### Test Coverage - 51 spec files, 758 tests (756 passing, 2 skipped)
- tokens.spec.ts, versioning.spec.ts, planning.spec.ts
- checkpoint.spec.ts, rng.spec.ts, exec-lock.spec.ts
- lifecycle.spec.ts, backward.spec.ts, compile.spec.ts
- compile-cache.spec.ts, compiled-region.spec.ts, ir-optimize.spec.ts, compile-fusion-integration.spec.ts
- memory-planning.spec.ts, memory-planning-integration.spec.ts, cross-device.spec.ts
- amp.spec.ts, frontend-amp.spec.ts, amp-ir-transform.spec.ts, amp-compile-integration.spec.ts
- webgpu/matmul-*.spec.ts, webgpu/cross-device-transfer.spec.ts, webgpu/fusion-codegen.spec.ts
- oracle/*.spec.ts, etc.

### AMP inside compile (§12) - COMPLETE
- **src/engine/amp.ts**: AMPPolicy, AutocastContext, F16/F32 eligible ops, select-gated commit helpers
- **src/engine/amp-ir-transform.ts**: IR graph transformation for automatic dtype casting
- **src/frontend.ts**: `autocast()` and `autocastAsync()` methods for frontend usage
- **src/engine/engine.ts**: AMP transforms applied during compile staging, cache key includes AMP policy hash
- **src/optim/grad-scaler.ts**: GradScaler for gradient scaling with NaN/Inf detection and dynamic rescaling

## Detailed Spec Compliance (v1.22)

### §0 Goals and Invariants - IMPLEMENTED
- ✅ Global laziness
- ✅ Explicit optimization boundary (fusion only in compiled regions)
- ✅ tidy/keep/dispose lifecycle
- ✅ PyTorch-like mutation/view semantics
- ✅ Lazy backward
- ✅ Keyed RNG
- ✅ Compilation caching
- ✅ Engine execution lock

### §1 Runtime Surface API - IMPLEMENTED
- ✅ `1.1` BaseId identity model
- ✅ `1.2` Laziness and materialization
- ✅ `1.3` Host coercion traps (`TensorHostCoercionError`)
- ✅ `1.4` tidy/keep/dispose + FinalizationRegistry (`finalizeQueue`)
- ✅ `1.5` retainGrad
- ✅ `1.6` compile() staging
- ✅ `1.7` markStep()

### §2 Core Data Types - PARTIAL (divergences noted)
- ✅ `2.1` DType, DeviceKind
- ⚠️ `2.2` BaseEntry - uses `number` instead of spec's `bigint` for versions
- ⚠️ `2.3` ViewMeta - simplified, no `offsetBytes`/`stridesBytes` structure
- ⚠️ `2.4` LocEntry - 2 roles (`ephemeral`|`persistent`) vs spec's 6 roles
- ⚠️ `2.4` Tombstone - not implemented (simpler disposal model)
- ✅ `2.5` Tensor object (mostly matches)
- ✅ `2.6` Graph IR

### §3 Effects, Tokens, External State - IMPLEMENTED
- ✅ `3.1` Token algebra (afterAll, tok_after)
- ✅ `3.2` tokGlobal/tokLoc threading
- ✅ `3.3` Join rule
- ✅ `3.4` Deterministic plan linearization (EventKey, PlanLinearOrder)
- ✅ `3.5` Token-order commit semantics
- ✅ `3.6` loc_load/loc_store/base_commit
- ✅ `3.7` pending-loc initTok

### §4 Alias Model, Views, Mutation - IMPLEMENTED
- ✅ `4.1` BaseId propagation
- ✅ `4.2` Representable views - WebGPU has strides/offset/isContiguous, CPU tracks strides
- ✅ `4.3` In-place mutation + base_commit (copy_, add_, zero_, fill_, mul_)
- ✅ `4.4` View mutation lowering - stridedScatterCopy/stridedScatterAdd ops

### §5 Caching, Freshness, Versioning - IMPLEMENTED
- ✅ Version tracking for cache guards

### §6 markStep() Semantics - IMPLEMENTED
- ✅ `6.1-6.6` All steps (force, compaction, finalize, retention, GC, reset)
- ✅ `6.7` Poisoning (poisoned engine throws on further ops)
- ✅ `6.8` Execution lock + drainFinalizeQueueCleanupOnly

### §7 Optimization Boundary - IMPLEMENTED
- ✅ No fusion outside compile regions

### §8 Compiled Regions - IMPLEMENTED
- ✅ `8.1` Deterministic staging
- ✅ `8.2` Cache keys + scalar canonicalization (§8.2.1)
- ✅ `8.3` Alias groups
- ✅ `8.4` StateIfaceSig
- ✅ `8.5` State-slot alias patterns
- ✅ `8.6` Token ABI reconciliation
- ✅ `8.6.0` SemanticSubeventSchedule
- ✅ `8.7` Auto-externalize
- ✅ `8.8` Null-state sentinels
- ✅ `8.10` Functionalization
- ✅ `8.11` Region-exit persistence

### §9 Autograd - IMPLEMENTED
- ✅ `9.1` Lazy backward (rooted under tokGlobal)
- ✅ `9.2` Two autograd modes
- ✅ `9.3` Saved-for-backward + commit guards
- ✅ `9.4-9.8` Leaf grads, zeroGrad, retainGrad, multiple backward, saved_state

### §10 Checkpointing - IMPLEMENTED
- ✅ Whole-body replay
- ✅ Purity fence (no persistent writes during recompute)
- ✅ Pending-loc init rule during recompute

### §11 RNG - IMPLEMENTED
- ✅ Keyed RNG (algorithmId + seed + drawNonce + opNonce)
- ✅ Draw nonces
- ✅ Checkpoint RNG replay (no double-advance)

### §12 AMP - IMPLEMENTED
- ✅ AMP transforms inside compile
- ✅ Select-gated commits
- ✅ GradScaler with NaN/Inf detection and dynamic rescaling

### §13 Cross-device - IMPLEMENTED
- ✅ Lazy transfers
- ✅ Transfer path resolution

### §14 Memory Planning - IMPLEMENTED
- ✅ Buffer pooling
- ✅ Allocator fencing
- ✅ In-flight plan retention

### §15 IR Optimization + Fusion - COMPLETE
- ✅ `15.1` Elementwise fusion (single-output) - IMPLEMENTED
- ✅ `15.2` Multi-output fusion - IMPLEMENTED (multiple output bindings, shared subexpression detection)
- ✅ `15.3` Memory coalescing/vectorization - IMPLEMENTED (vec4/vec2 loads/stores, automatic width selection)
- ✅ `15.4` Random ops non-fusible

### Summary

| Category | Status |
|----------|--------|
| Core engine semantics (§1, §3, §6) | ✅ Implemented |
| Token algebra & planning | ✅ Implemented |
| Compiled regions (§8) | ✅ Implemented |
| Autograd (§9) | ✅ Implemented |
| Checkpointing (§10) | ✅ Implemented |
| RNG (§11) | ✅ Implemented |
| AMP (§12) | ✅ Implemented |
| Cross-device (§13) | ✅ Implemented |
| Memory planning (§14) | ✅ Implemented |
| Elementwise fusion (§15.1) | ✅ Implemented |
| Multi-output fusion (§15.2) | ✅ Implemented |
| Memory coalescing (§15.3) | ✅ Implemented |

### Known Divergences from Spec

These are intentional simplifications, not bugs:

1. **Version types**: Spec uses `bigint`, implementation uses `number` - functionally equivalent for realistic workloads
2. **LocRole granularity**: Spec has 6 roles (`tensor_state`, `grad_state`, `rng_state`, `engine_state`, `saved_state`, `internal_scratch`), implementation has 2 (`ephemeral`, `persistent`)
3. **Tombstone types**: Not implemented - disposal uses simpler model

### WebGPU Op Limitations vs PyTorch

Current implementations have the following limitations compared to PyTorch:

#### Op-Specific Limitations

| Op | Current | PyTorch | Gap |
|----|---------|---------|-----|
| **transpose** | ✅ Returns view (no copy) | Returns view (no copy) | Parity achieved |
| **permute** | ✅ Returns view (no copy) | Returns view (no copy) | Parity achieved |
| **expand** | ✅ Returns view (stride=0 for broadcast) | Returns view (no copy) | Parity achieved |
| **contiguous** | ✅ Materializes strided tensor | Materializes non-contiguous | Parity achieved |
| **sum** | ✅ Returns 0-d tensor for full reduction | Returns 0-d tensor | Parity achieved |
| **mean** | ✅ Returns 0-d tensor for full reduction | Has `correction` param (ddof) | Missing correction/ddof parameter |
| **item()** | ✅ Async scalar extraction | Sync scalar extraction | Async due to GPU readback |
| **gather** | ✅ Basic + autograd | Has `sparse_grad`, `out` param | Missing sparse_grad optimization |
| **scatterAdd** | ✅ Basic + autograd | Uses atomics for correctness | No f32 atomics; overlapping indices undefined |

#### View Memory Aliasing (Implemented)

WebGPU tensors now support strided views:
- **WebGPUTensor** has `strides`, `offset`, and `isContiguous` fields
- **transpose()** returns a view sharing the same buffer (no data copy)
- **permute()** returns a view with reordered dimensions (no data copy)
- **expand()** returns a view with stride=0 for broadcast dimensions (no data copy)
- **contiguous()** materializes non-contiguous tensors to new buffer
- **read()** automatically materializes non-contiguous tensors before readback

#### Strided Elementwise Kernels (Implemented)

Binary and unary ops now work directly on strided (non-contiguous) tensors:
- **Binary ops** (add, sub, mul, div) use `computeEffectiveBroadcastStrides()` to handle non-contiguous inputs
- **Unary ops** (sqrt, relu, etc.) use `unaryStridedShader()` to read strided input
- **No materialization needed** - operations on transposed/expanded views work without calling `contiguous()` first
- **Chained views** - transpose + expand + binary op chains work correctly

#### Dtype Support (Implemented)

WebGPU backend supports multiple dtypes for elementwise ops:
- **Supported dtypes**: f32 (default), f16 (requires device support), i32, u32
- **WebGPUTensor** has `dtype` field tracking element type
- **tensorFromArrayWithDtype()** creates typed tensors: `tensorFromArrayWithDtype([1, 2, 3], [3], "i32")`
- **Binary ops** (add, sub, mul, div) work on all supported dtypes
- **View ops** (reshape, transpose, expand, permute) preserve dtype
- **Dtype mismatch throws** - mixing dtypes in binary ops is not allowed
- **f16 support**: Requires `shader-f16` device feature; use `isF16Supported()` to check availability

```typescript
import { tensorFromArrayWithDtype, initWebGPU, webgpuBackend, isF16Supported } from "torchlette/backend/webgpu";

await initWebGPU();

// Integer tensors
const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
const b = tensorFromArrayWithDtype([10, 20, 30, 40], [2, 2], "i32");
const result = webgpuBackend.ops.add(a, b);  // [11, 22, 33, 44] as i32

// Half-precision tensors (requires shader-f16)
if (isF16Supported()) {
  const x = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0], [2, 2], "f16");
  const y = tensorFromArrayWithDtype([0.5, 1.5, 2.5, 3.5], [2, 2], "f16");
  const z = webgpuBackend.ops.add(x, y);  // [1.5, 3.5, 5.5, 7.5] as f16
}
```

#### 0-d Tensors (Scalars)

Full reductions now return 0-d tensors (shape `[]`) like PyTorch:
- **sum()** without dim returns 0-d tensor with the total sum
- **mean()** without dim returns 0-d tensor with the mean
- **item()** extracts the scalar value asynchronously: `const value = await tensor.sum().item()`
- 0-d tensors can be expanded to any shape for broadcasting

```typescript
// PyTorch-like scalar handling
const loss = output.sum();           // 0-d tensor (shape [])
const value = await loss.item();     // JavaScript number
```

#### Architectural Limitations

1. **f16 device support** - f16 dtype requires the `shader-f16` WebGPU device feature. Not all GPUs/browsers support this. Use `isF16Supported()` to check availability at runtime.

2. **scatterAdd correctness** - WebGPU lacks f32 atomics. Concurrent writes to same output index produce undefined results. Works correctly only when indices don't overlap.

#### Missing Ops for PyTorch Parity

```
Medium Priority:
- scatter_reduce(mode)              # reduce modes: sum, prod, mean, max, min
- dtype parameter on reductions     # sum(..., dtype='f16')
- negative dim in expand (-1)       # keep dimension unchanged
- sparse_grad for gather            # gradient sparsity optimization

Lower Priority:
- index_select                      # simpler than gather for 1D indices
- index_add / index_copy            # variants of scatter
- clamp / clip                      # bounds clamping
```

#### Autograd Coverage

| Op | Forward | Backward | Notes |
|----|---------|----------|-------|
| add/sub/mul/div | ✅ | ✅ | Full support |
| matmul | ✅ | ✅ | Full support |
| linear | ✅ | ✅ | Custom backward: dW = dY.T @ X (no transpose) |
| relu | ✅ | ✅ | Full support |
| sqrt | ✅ | ✅ | Full support |
| sum | ✅ | ✅ | Via frontend |
| mean | ✅ | ✅ | Via frontend |
| expand | ✅ | ✅ | Via frontend (sumToShape) |
| transpose | ✅ | ✅ | Inverse transpose |
| permute | ✅ | ✅ | Inverse permutation |
| gather | ✅ | ✅ | Uses scatterAdd |
| scatterAdd | ✅ | ✅ | Uses gather for src grad |
| where | ✅ | ✅ | Conditional select |

#### In-place Operations (§4.3-4.4)

| Op | Description |
|----|-------------|
| copy_(src) | Copy values from src into tensor |
| add_(src) | Add src values in-place |
| zero_() | Set all values to zero |
| fill_(value) | Fill with scalar value |
| mul_(value) | Multiply by scalar in-place |

## WebGPU Buffer Pool Invariants

**Buffer recycling must align with encoder scope boundaries.**

The buffer pool has a `pendingRelease` queue: when a tensor is destroyed, its buffer goes to `pendingRelease` rather than immediately back to the main pool. Buffers are moved from `pendingRelease` to the main pool (making them acquirable) only at `endSharedEncoder()` — the end of a step's encoder scope.

**Do NOT flush `pendingRelease` to pool mid-step** (i.e., inside `flushSharedEncoder()`). This was attempted for intra-step buffer reclamation and causes deterministic numerical corruption. The corruption manifests as ~2% loss drift (e.g., step 1 loss 6.25 vs expected 6.12 on DistilGPT-2).

Why: Within a step, multiple plan executions share the same encoder scope (forward, backward, optimizer). Buffers released by earlier plans (e.g., forward-pass intermediates) may be in `pendingRelease` while the GPU is still reading them from a previously submitted command buffer. Flushing them to pool mid-step allows a later op to acquire and write to a buffer the GPU hasn't finished reading. Neither CPU-side lifetime analysis, active-tensor-registry checks, nor `queue.onSubmittedWorkDone()` prevents this — the root cause is structural to how WebGPU command buffer scoping interacts with buffer pool recycling.

**Safe reclamation patterns:**
- End-of-step flush (`endSharedEncoder()` → `flushPendingToAvailable()`) — buffers available next step
- Periodic reclamation between plan segments (`flushSharedEncoder()` + `flushBufferPool()` as separate calls between segments) — works because each segment's encoder is fully submitted before the next segment begins
- Intra-segment periodic reclamation (every 25 nodes in `executeSequentialSegmentWithEarlyRelease`) — same mechanism, `flushSharedEncoder()` submits then `flushBufferPool()` moves pending→pool. Safe because subsequent dispatches encode on a fresh encoder and WebGPU queue ordering guarantees prior work completes first
- The `sharedEncoderWriteSet` WAW check in `createTrackedBuffer` is orthogonal and must be kept — it prevents write-after-write hazards within a single encoder

## Remaining Performance Optimizations

Profiled on DistilGPT-2 training. Two baseline configurations:

### Baseline A: 31 tokens (original)
Steady-state ~32ms/step with dispatch replay, 24ms GPU, ~85% GPU utilization. CPU-dispatch-bound.

### Baseline B: 512 tokens (current default)
Steady-state ~54ms/step wall clock (profiler, steps 3-4 avg, no replay). GPU: 103ms total. Top GPU kernels: `matmul` 37.7ms (36.5%), `fusedAttentionBackward` 15.7ms (15.2%), `fusedAttentionForward` 12.4ms (12.0%), `matmul++cast+bias+binary` 8.7ms (8.4%), `matmul++cast` 8.1ms (7.9%), `matmul++cast+bias` 5.6ms (5.4%), `adamStep` 4.9ms (4.7%). Cross-entropy 1.1ms (1.1%). Bind group cache 100.0% (0 misses). 10 submits/step. Fusion: 1124 nodes, 183 fused (16.3%), 20 groups. Memory: 4871MB current / 6194MB peak / 32GB limit. Pool reuse 56.8%. 1 new alloc/step. Leak status: OK (storages +0.0/step).

Targets ranked by impact:

**Completed optimizations:**
1. ~~**Intra-plan buffer reclamation**~~ — DONE. `createBuffer` avg cost dropped 365µs→81µs.
2. ~~**In-place Adam kernel**~~ — DONE. adam.allocBufs 112ms→12ms, opt phase 162ms→1ms.
3. ~~**Fused cross-entropy kernel**~~ — DONE. 0.4ms GPU (was 34ms). Single-pass `log_softmax+nll_loss` kernel.
4. ~~**Adam batch submission**~~ — DONE. Submits dropped 168→121/step. Adam pre/post-flush eliminated. `queue.submit` cost 52ms→13ms.
5. ~~**Buffer pool reclamation**~~ — DONE. Intra-segment periodic reclaim (every 25 nodes). `createBuffer` cost 92ms→66ms. Backward pass 163ms→124ms.
6. ~~**Window-based buffer reservation**~~ — DONE. Empirical per-window demand recording computes exact per-size-class pool reservation. `createBuffer` 66ms→0.7ms (1 alloc/step). Pool reuse 59%→99.9%. Unlimited pool budget (PyTorch-like). Configurable via `setBufferPoolBudget()` or `TORCHLETTE_POOL_BUDGET_MB` env var.
7. ~~**Adam ensureContig**~~ — DONE. 43ms→2ms/step. Params contiguous after first step.
8. ~~**Params writeBuffer skip**~~ — DONE. Params data cached; identical data skips `writeBuffer`. 710→83 calls/step, 8.8ms→1.0ms. ~7.8ms saved.
9. ~~**Params array pooling**~~ — DONE. Pre-allocated `Uint32Array` pool for 1-8 word params. Eliminates ~700 short-lived allocations/step.
10. ~~**Output buffer pre-pinning**~~ — DONE. Pre-extract hinted output buffers from pool at step start. 296/454 buffers pinned (65%). Eliminates contention in pool acquire. Marginal bind group cache improvement (22% vs 21%).
11. ~~**Pool acquire writeSet removal**~~ — DONE. WriteSet check in `acquire()` and `acquirePreferred()` was unreachable (pool buffers can't be in writeSet). Removed scan loop, `pop()` used directly.

12. ~~**Per-lowered-plan buffer arena**~~ — DONE. Each cached lowered plan gets its own buffer arena: persistent GPUBuffers never returned to pool. `resolveOutputBuffer` and `allocateOutputBuffer` return arena buffers during lowered plan execution. Bind group cache hit rate 22%→63.3%. `createBindGroup` cost 20ms→11ms (259 calls vs ~700). ~9ms/step saved. Arena check in pool's `deferredDestroy` prevents arena buffer destruction from any call site. Disable with `TORCHLETTE_USE_ARENA=0`.
13. ~~**Matmul outputs through arena**~~ — DONE. `dispatchMatmul` and `dispatchMatmulWithEpilogue` now use `resolveOutputBuffer` instead of `createTrackedBuffer`, routing matmul outputs through the arena. Bind group cache 63.3%→75.3% (531 hits / 174 misses). `createBindGroup` 11ms→9.5ms (259→174 calls). ~1.5ms/step saved.
14. ~~**FusionGroup external input caching**~~ — DONE. Cached `(nodeLocalIdx, inputIdx)` pattern on first lowered-plan execution. Subsequent steps skip O(n²) dedup across 119 fusion groups, using O(n) cached pattern lookup instead.
15. ~~**Matmul epilogue label caching**~~ — DONE. Label string (`matmul+prologue+cast+bias` etc.) computed once and cached on `LoweredMatmulEpilogueAction`. Skips string construction for ~30 matmul dispatches/step.
16. ~~**Fused LayerNorm forward kernel**~~ — DONE. Single WGSL kernel per LayerNorm instance (was 9 ops: 2 reductions + 7 elementwise). 13 instances in DistilGPT-2. `mean` dispatches reduced 106→54. Forward GPU time reduced. ~80 fewer dispatches.
17. ~~**Fused LayerNorm backward kernel**~~ — DONE. Two kernels per instance: gradX (single workgroup per row with recomputed forward stats) and gradWeight+gradBias (cross-row reduction). Uses side-output pattern for multi-tensor returns. ~20 ops per instance → 2 dispatches.
18. ~~**Adam CPU quick wins**~~ — DONE. Persistent `infFlagBuffer` (eliminates `createBuffer`+`destroy` per step). Pre-allocated typed arrays for config buffer construction (eliminates 228 short-lived allocations/step). Inlined adam-batch dispatch bypasses `executeOp` switch. ~1-2ms saved.
19. ~~**Optimizer plan lowered path**~~ — DONE. Removed `!hasFusionPotential` early return in `executePlanOptimized`. All plans (including the 76-node Adam optimizer plan) now get lowered plan templates, buffer arenas, and adam-batch optimization. This was the root cause of 150 bind group cache misses: the optimizer plan allocated F16 weight buffers from the pool (changing identity each step), then forward/backward plans used those as matmul inputs. Now F16 buffers are arena-managed (stable identity). Bind group cache 75.3%→96.2% (174→24 misses). `createBindGroup` 9.5ms→1.0ms. `queue.submit` 102→24 calls (optimizer plan now uses single shared encoder). ~15ms/step saved.
20. ~~**Matmul epilogue CPU fast path**~~ — DONE. On lowered plan replay, cache all structural decisions (prologue resolution, matmul geometry, transpose detection, dtype computation, epilogue config) on first execution. Subsequent steps resolve only buffer pointers and call `dispatchTiledMatmul` directly via `dispatchMatmulDirect`, bypassing `executeMatmulWithEpilogue`, `dispatchMatmulWithEpilogue`, dynamic imports, shape computation, transpose detection, and contiguous checks. `matmul++cast+bias` CPU: 730µs→63µs avg (11.6x improvement). Total: 17.5ms→1.6ms for 24 dispatches. ~16ms/step saved.
21. ~~**Adam batch CPU fast path**~~ — DONE. Ported tight loop from dispatch replay path to normal lowered plan path. Eliminates per-node overhead: `.map()` array allocations, `executeOp` switch dispatch, `findIndex` alias scan, per-node profiling. Single `profileOpBegin/End` for entire batch. Direct `adamOp()` call with indexed `getInputStorage`. Adam CPU: 10.7ms→5.3ms. ~5.4ms/step saved.
22. ~~**Reclaim interval tuning**~~ — DONE. Default reclaim interval 50→100. Submits: 24→15 (−37%). `queue.submit` calls reduced. Loss trajectory unchanged. Configurable via `TORCHLETTE_RECLAIM_INTERVAL` env var.
23. ~~**Bind group cache miss fix**~~ — DONE. Forward plan's `executeLoweredPlan` consistently failed with "Input not ready" (ordering mismatch), falling back to `executePlanOptimized` without arena — causing 76 extra bind group misses. Fix: (1) disable lowered plan after first failure (stop retrying each step), (2) activate template's buffer arena in fallback `executePlanOptimized`, (3) save/restore dispatch sequence counters on failure to maintain bind group cache alignment. Bind group cache 84.9%→96.2% (98→24 misses). `createBindGroup` 3.7ms→1.0ms. ~2.7ms/step saved.
24. ~~**Linear op + backward `add` fusion**~~ — DONE. Added `linear(input, weight, bias?)` frontend op with custom backward computing `dW = dY.T @ X` directly in weight's shape (eliminates transpose). Taught epilogue detection to skip reshape that removes leading size-1 dims (`[1,M,N]→[M,N]`), enabling the wte gradient `add` to fuse into the preceding backward matmul. `add` GPU: 4.5ms→0.8ms (max 4380µs→621µs). ~3.7ms GPU/step saved. Submits: 15→14.
25. ~~**Matmul shape-class config fix + small_k**~~ — DONE. Fixed dead-code bug: `getConfigForShape()` ignored shape class entirely, always returning `DEFAULT_CONFIG` (32×32×16). Shape-specific defaults in `getDefaultConfigForShape()` were never used. Added `small_k` shape class for K<64 with large output (lm_head backward dW). Max matmul GPU: 14.6ms→11ms.
26. ~~**K-split matmul**~~ — DONE. For small-M backward matmuls (M=31) with large K, the output tile grid (12 workgroups) severely underutilizes the GPU. K-split divides K-reduction across P workgroups in the Z dimension: each Z-slice computes partial sums, then a lightweight reduction kernel sums them. Activated when `baseWorkgroups < 64 && K > 512 && batchSize == 1 && no epilogue`. 25 backward dX matmuls across all 6 transformer layers use K-split. Max matmul GPU: 11ms→1.4ms (−88%). Backward GPU: 28ms→12ms (−57%). Total GPU: 43ms→27ms (−37%). Submits: 14→12.
27. ~~**Dispatch replay cache**~~ — DONE. Enabled by default. Records full GPU dispatch sequence on first execution, replays bind groups directly on subsequent steps — bypasses all JS dispatch logic. Two fixes: (1) arena counter sync — `arenaResolveIndex` recorded before data-source/view/sequential execution and restored during replay, preventing buffer identity desync; (2) encoder-copy ops (scatterAdd) re-executed during replay — `copyBufferToBuffer` commands are invisible to compute dispatch recording, so ops using them must re-run instead of replaying stale bind groups. Loss parity verified bit-exact on DistilGPT-2 (50 steps, all losses identical). Disable with `TORCHLETTE_DISPATCH_REPLAY=0`. Auto-disabled when `TORCHLETTE_PROFILE=1`.
28. ~~**Forward plan lowered execution fix**~~ — DONE. Forward plan (392 nodes) was permanently falling back to full JS dispatch after the first `executeLoweredPlan` attempt failed with "Input not ready". Root cause: matmul epilogue chain reconstruction in `executeLoweredPlan` always took `inputs[1]` as the external epilogue input ref, but when the chain connects via `inputs[1]` (commutative add for residual connections), the external input is actually at `inputs[0]`. Fix: detect which input is the chain node (by checking against previous chain node ID) and use the other input as the external ref. Forward plan now replays successfully. Combined with dispatch replay: steady-state 68ms→43ms/step (−37%). Finetune-demo verified stable over 60+ steps with correct loss convergence.
29. ~~**View result caching in dispatch replay**~~ — DONE. During replay, view ops (reshape, transpose, permute, expand, narrow) produce deterministic results because arena buffers are stable across steps. Cache view metadata (buffer, shape, strides, offset) on first recording, reconstruct directly on replay bypassing `executeOp`, `getInputStorage`, and alias detection. Backward view time 2.1ms→0.1ms (95% reduction). Steady-state 43ms→32ms/step.
30. ~~**Packed Adam dispatches**~~ — DONE. Groups same-element-count optimizer parameters into contiguous packed GPU buffers, dispatches one Adam kernel per size class instead of one per parameter. 76→8 dispatches. Adam GPU: 5.4ms→1.9ms (−65%). ~3.5ms GPU/step saved. Works during profiling (guards removed).
31. ~~**Bind group cache: max + narrowBackward arena routing**~~ — DONE. `max()` and `narrowBackward()` used `createTrackedBuffer` directly, bypassing the buffer arena. Replaced with `resolveOutputBuffer` (max) and let `dispatchElementwise` handle allocation (narrowBackward). Bind group cache 91.0%→99.6% (40→2 misses). `createBindGroup` 1.9ms→0.1ms. ~1.8ms/step saved.
32. ~~**Submit reduction: eliminate periodic reclamation**~~ — DONE. With arena-managed buffers (335 hits/step) and pool reservation (1 new alloc/step), intra-step buffer reclamation via periodic `flushSharedEncoder` is unnecessary. Increased `DEFAULT_RECLAIM_INTERVAL` 100→10000 (effectively disabled for plans <10k nodes). Also increased `PARAMS_FLUSH_THRESHOLD` 256→2000 to prevent auto-flush from params buffer accumulation. `queue.submit` calls: 14→7/step (profiler), ~4/step (replay). Total submit CPU: 3.9ms→3.1ms. ~0.8ms/step saved. Configurable via `TORCHLETTE_RECLAIM_INTERVAL` env var.
33. ~~**dKV backward Q-tiling**~~ — DONE. Replaced per-Q-row loop (512 iterations, 1024 barriers) with tiled loop (BQ=16, 32 iterations, 64 barriers). Cooperative shared memory loading: all 64 threads load Q/dO tiles in parallel (was thread-0 broadcast). Causal tile-level early exit. `fusedAttentionBackward` GPU: 58.8ms→15.7ms (−73%). Backward GPU: ~120ms→73ms. ~43ms GPU/step saved.

**Open targets (ranked by estimated savings at 512 tokens):**
1. **lm_head matmul** — 16.9ms GPU (16.3%). Forward 8.1ms (512x768 @ 768x50257) + backward 8.7ms. Large vocab (50257) dominates. Single-dispatch bottleneck.
2. **fusedAttentionForward recompute in backward** — 12.0ms GPU (11.6%). 6 recomputes during backward. Could cache forward O/L to avoid recompute (memory vs compute tradeoff).
3. **fusedAttentionBackward further tuning** — 15.7ms GPU (15.2%). Max 1661µs/dispatch. BQ=32 could halve barriers but approaches 16KB shared memory limit.
4. **GC pressure** — 2.1% of CPU profiler self-time (147ms). Object pooling for Tensor metadata.
5. **Pipeline cache warmup** — Step 0 is 5.4s (48x steady-state) due to pipeline compilation. Pre-compile pipeline variants during model load.
