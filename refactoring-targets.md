# Refactoring Targets

Refactoring opportunities identified from full codebase reviews. Targets 1–5, 7–11 are complete.

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
