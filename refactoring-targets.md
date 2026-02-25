# Refactoring Targets

Refactoring opportunities identified from full codebase reviews. All targets (1–5, 7–19) are complete.

## Target 1: Decompose `webgpu/index.ts` — DONE

Decomposed 10,909-line monolith into 21 focused modules. Commit `4315ac6`.

## Target 2: Generic Chunking Abstraction — DONE

Extracted `dispatchFlatChunked()` for flat-element ops (commit `64d215d`). Extracted `computeDimChunkLayout()` and merged gather/scatter Direct+Chunked variants (commit `1a92485`). Remaining chunked ops (contiguous, matmul, sum) have genuinely different semantics — no further generalization possible.

## Target 3: Module-Level Mutable Global State → State Objects — DONE

Consolidated 11 cross-module globals into `webgpu-state.ts` (commit `9ce5629`). Consolidated ~50 module-local globals across 5 files into typed state objects with reset functions (commit `9931fb8`).

## Target 4: Split `engine/lazy.ts` — DONE

Decomposed 6,008-line file into 11 focused modules. Commit `c0ae5c8`.

## Target 5: Split `frontend.ts` — DONE

Decomposed 3,215-line file into 7 focused modules. Commit `0a61e88`.

## Target 6: Shader Generation Consistency — LOW PRIORITY

Post-decomposition, inline shader template literals are already co-located with their dispatch logic in focused op files. The 4 patterns (inline, dedicated functions, external kernel files, fusion codegen) are naturally segregated. Extracting inline shaders into builder functions would be cosmetic — not a meaningful structural improvement.

## Target 7: Deduplicate Op Dispatch Switches — DONE

Consolidated 4 duplicate `switch (node.op)` dispatch tables into single canonical `executeOp` in `op-dispatch.ts`. Deleted `executeOpInternal`, private `getInputStorage`, `createStorageHandleInternal`, and `computeContiguousStrides` duplicates. Fixed divergence bugs: missing `pow`, broken `transfer`, missing `isfinite`. Net ~1,450 lines removed.

## Target 8: Type Side-Output Fields on `LazyIRNode` — DONE

Added `NodeSideOutputs` interface and `_sideOutputs` field to `LazyIRNode`. Replaced all 17 `(node as any)._fieldName` monkey-patching accesses across `op-dispatch.ts`, `executor-lowered.ts`, and `optim/adam.ts` with typed property access.

## Target 9: Deduplicate Utility Functions — DONE

Deleted duplicate `dtypeBytes` from `fusion-dispatch.ts`, duplicate `dtypeToWgsl` from `matmul/codegen.ts`, and `dtypeToWgsl` (bool→u32 variant) from `fusion-codegen.ts`. Added `dtypeToWgslStorage()` to `shape-utils.ts` for the bool→u32 case. `computeContiguousStrides` and `createStorageHandleInternal` duplicates were already removed in Target 7.

## Target 10: Fix `as unknown as GPUDevice` Type Casts — DONE

Replaced 7 local WebGPU type definition blocks (GPUDevice, GPUBuffer, GPUComputePipeline, etc.) in kernel files with imports from `gpu-types.ts`. Eliminated 14 `as unknown as GPUDevice` casts and ~30 `device as any`/`pipeline as any` casts across attention-kernel.ts, layernorm-kernel.ts, cross-entropy-kernel.ts, adam-kernel.ts, unscale-kernel.ts, fusion-dispatch.ts, matmul/dispatch.ts, and ops/fused.ts.

## Target 11: Add `rand`/`randn`/`bernoulli` to `LazyOpCode` — DONE

Added as part of Target 7.

---

## Wave 2: Code Quality Audit Findings

Identified from full codebase audit after completing Targets 1–11.

### `as any` Cast Inventory (201→37 total; see Targets 10, 13, 14, 17, 19)

| File | Count | Pattern |
|------|-------|---------|
| `attention-kernel.ts` | 35 | buffer/pipeline casts to WebGPU API |
| `layernorm-kernel.ts` | 26 | same |
| `executor-lowered.ts` | 22 | buffer extraction from BackendTensor |
| `ops/fused.ts` | 17 | buffer/pipeline casts |
| `index.ts` | 15 | WebGPU API boundary |
| `cross-entropy-kernel.ts` | 13 | buffer/pipeline casts |
| `adam-kernel.ts` | 12 | same |
| `unscale-kernel.ts` | 10 | same |

### Large Files (>1000 lines)

| File | Lines |
|------|-------|
| `runtime/engine.ts` | 2,489 |
| `engine/engine.ts` | 1,808 |
| `frontend.ts` | 1,592 |
| `cpu/numeric.ts` | 1,518 |
| `fusion-detect.ts` | 1,504 |
| `attention-kernel.ts` | 1,410 |

### Other Findings

- **Duplicate buffer extraction** — `(x.backendTensor as any).buffer` appears ~40 times
- **Scattered module-local mutable globals** — 25+ pipeline caches, recording buffers, etc.
- **Error handling inconsistency** — 349 `throw` vs 53 `return undefined/null`

---

## Target 12: Static Analysis for Dead Exports — DONE

Removed `export` from functions, types, interfaces, and constants with zero external consumers across 37 files. Also deleted dead code: `executeFusedElementwise()` from `fusion-dispatch.ts`, `irToLazyPlan()`/`segmentPlan()`/`PlanSegment` from `lazy-to-ir.ts`, three unused token reconciliation types from `compiled-region.ts`, and unused reset functions (`resetBindGroupCacheLocalState`, `resetFenceState`, `resetEncoderState`, `resetCpuProfileState`, `resetGpuTimestampState`). Cleaned up dead re-exports from `matmul/index.ts` (`AutotuneOptions`, `EpilogueOp`, `DispatchMatmulOptions`, `clearPerShapeTuningCache`, `clearPipelineCache`, `getConfigForShape`, `setTuningResult`, `AMPConfig`, `MatmulOptions`, `MatmulParams`, `transposeModeToInt`) and `ops/index.ts` (`OpDef`, `OpArity`). Net −163 lines.

## Target 13: Typed Buffer Extraction Helper — DONE

Added `gpuBuffer(tensor)` helper in `gpu-types.ts` to replace `(x.backendTensor as any).buffer` casts. Applied across `executor-lowered.ts` (7 sites) and `executor-optimized.ts` (2 sites). Zero `backendTensor as any` buffer extractions remain.

## Target 14: Reduce `as any` Casts in Kernel Files — DONE

Removed 125 vestigial `as any` casts across 8 kernel/dispatch files. All casts were unnecessary after Target 10 unified WebGPU type definitions into `gpu-types.ts`. Patterns removed: `trackSharedEncoderWrite(buf as any)`, `cachedCreateBindGroup(...buffers as any) as any`, `dispatchComputePass(pipeline as any, bindGroup as any, ...)`, `releaseParamsBuffer(buf as any)`, `(encoder as any).beginComputePass()`. 3 casts in `ops/fused.ts` preserved (genuine `unknown`→`GPUBuffer` bridge). Total `as any` count: 201→75.

## Target 15: Decompose Large Files — DONE

Decomposed `runtime/engine.ts` (2,489→1,990 lines) and `engine/engine.ts` (1,808→1,507 lines) into focused modules. New files: `engine/engine-types.ts` (231 lines — types/interfaces), `engine/engine-errors.ts` (39 lines — error classes), `engine/engine-helpers.ts` (88 lines — standalone helpers), `runtime/engine-types.ts` (86 lines — types/dispatch modes/options), `runtime/shape-helpers.ts` (97 lines — pure shape computation), `runtime/engine-fused.ts` (261 lines — fused kernel op helpers), `runtime/engine-facade.ts` (216 lines — defaultEngine singleton + ~40 facade functions). Re-exports preserve all existing import paths via barrel files. 2 test files updated to import facade functions from new path.

## Target 16: Consolidate Remaining Module-Local Globals — DONE

Added exported `resetKernelState()` functions to 5 kernel files (`attention-kernel.ts`, `layernorm-kernel.ts`, `cross-entropy-kernel.ts`, `adam-kernel.ts`, `unscale-kernel.ts`) and `resetMatmulState()` to `matmul/dispatch.ts` for their pipeline caches, config buffer caches, and temp buffer caches. Created master `resetAllKernelCaches()` in `gpu-context.ts` that calls all 7 reset functions (the 6 above + `resetFusionCache()` from `fusion-dispatch.ts`). Wired into `destroyWebGPU()` for full cleanup on device teardown. Exported from `index.ts` for test isolation use. Pre-existing state was already well-consolidated: `webgpu-state.ts` (11 cross-module globals), `shared-encoder.ts` (encoderState object), `bind-group-cache.ts` (cacheState object), `buffer-arena.ts` (arenaLocal object), `profiler.ts` (cpuProfile + gpuTs objects), `dispatch-recording.ts` (start/stop lifecycle), `buffer-pool.ts` (class with private state).

## Target 17: Typed Backend Op Dispatch — DONE

Added `OpExecOptions` type (`{ outBuffer?: unknown }`) to `backend/types.ts`. Extended `BackendOps` method signatures with optional `OpExecOptions` parameter for all ops that support output buffer donation: `add`, `sub`, `div`, `mul`, `matmul`, `sqrt`, `relu`, `exp`, `log`, `neg`, `abs`, `tanh`, `sigmoid`, `gelu`, `silu`, `gt`, `lt`, `ge`, `le`, `eq`, `ne`, `where`. Ops with existing options use intersection types (e.g., `SubOptions & OpExecOptions`). Removed 20 `as any` casts from `memory-planned-executor.ts`. Also fixed 2 pre-existing type errors in `ge`/`le` cases that passed `donationOpts` without `as any`. Total `as any` count: 75→55.

## Target 18: Deduplicate Shape Utilities — DONE

Created `src/core/shape.ts` as canonical home for pure shape functions (`sizeOf`, `broadcastShapes`, `shapesEqual`) — zero dependencies, importable from any layer. Deleted 7 duplicate copies across `frontend.ts`, `frontend-autograd.ts`, `backend/cpu/numeric.ts`, `engine/ir.ts`, `engine/matmul-epilogue.ts`. Re-export sites (`backend/webgpu/shape-utils.ts`, `runtime/shape-helpers.ts`, `engine/matmul-epilogue.ts`) import and re-export to preserve all downstream import paths.

## Target 19: Typed `BackendTensor` → `WebGPUTensor` Narrowing — DONE

Added `asGPUTensor(tensor)` helper in `gpu-types.ts` to narrow `BackendTensor` to `WebGPUTensor` for typed property access (shape, strides, buffer, dtype, etc.). Removed 13 `as any` casts from `executor-lowered.ts` and 5 from `matmul-epilogue.ts`. Patterns removed: `(bt as any).shape`, `(resultTensor as any).shape = ...`, `asGPUTensor(matmulInputA.backendTensor)` replacing `(matmulInputA.backendTensor as any)`. Total `as any` count: 55→37.
