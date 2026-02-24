# Refactoring Targets

Refactoring opportunities identified from full codebase reviews. Targets 1–5, 7, 8, 11 are complete.

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

## Target 9: Deduplicate Utility Functions

**Impact:** Prevents behavioral divergence
**Effort:** Low
**Priority:** MEDIUM

Several utility functions have multiple definitions with subtly different behavior:

| Function | Copies | Issue |
|----------|--------|-------|
| `dtypeToWgsl()` | 3 | `shape-utils.ts` maps `bool→"bool"`, `fusion-codegen.ts` maps `bool→"u32"`, `matmul/codegen.ts` returns `"f32"` for everything except f16 |
| `dtypeBytes()` | 2 | `shape-utils.ts` (canonical) vs `fusion-dispatch.ts` (identical copy) |
| `computeContiguousStrides()` | 2 | `backend/types.ts` (handles 0-d tensors) vs `op-dispatch.ts` (missing that guard) |
| `createStorageHandleInternal()` | 2 | `op-dispatch.ts` (exported, never called) vs `executor-sequential.ts` (used once) |

**Approach:** Delete the duplicates and import from the canonical source. For `dtypeToWgsl`, the `bool→"u32"` variant in fusion-codegen is intentionally different (WGSL doesn't have bool arrays) — make the canonical version accept an option or add a `dtypeToWgslStorage()` variant.

**Files:** `src/backend/webgpu/fusion-codegen.ts`, `src/backend/webgpu/fusion-dispatch.ts`, `src/backend/webgpu/matmul/codegen.ts`, `src/engine/op-dispatch.ts`

## Target 10: Fix `as unknown as GPUDevice` Type Casts

**Impact:** Cleanliness, IDE support
**Effort:** Low
**Priority:** MEDIUM

The local type alias `GPUDevice` in `gpu-types.ts` doesn't include all fields that kernel files need (e.g., `.limits`, `.queue`). This forces 12+ `as unknown as GPUDevice` casts across kernel files:

```
attention-kernel.ts     — 5 casts
layernorm-kernel.ts     — 7 casts
cross-entropy-kernel.ts — 4 casts
unscale-kernel.ts       — 4 casts
fusion-dispatch.ts      — 1 cast
```

**Approach:** Update the `GPUDevice` type alias in `gpu-types.ts` to include the missing members (`.limits`, `.queue`, `.createCommandEncoder()`, etc.), or use the real WebGPU types from `@webgpu/types`.

**Files:** `src/backend/webgpu/gpu-types.ts`, plus all kernel files that use the casts

## Target 11: Add `rand`/`randn`/`bernoulli` to `LazyOpCode` — DONE

Added as part of Target 7.
