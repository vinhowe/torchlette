# Architecture

## Overview

Torchlette is a WebGPU-accelerated tensor library for TypeScript with PyTorch-like semantics. Operations are recorded lazily into an IR graph, then bulk-executed through a multi-stage pipeline: graph rewrites, fusion detection, lowered plan construction, and GPU dispatch. Structurally identical graphs are cached at every level — analysis templates, lowered plans, compiled GPU command sequences, shader pipelines, and bind groups — so that steady-state training steps execute with near-zero host overhead.

## Directory Structure

```
src/
  frontend/       User API: Tensor, Torchlette, autograd, autocast, decomposed ops
  runtime/        RuntimeEngine: lazy IR node creation, force/markStep, lifecycle
  graph/          LazyIRNode types, StorageHandle, node factory, storage tracker
  ops/            OP_REGISTRY: single source of truth for op metadata + autograd
  compiler/       Graph analysis: rewrites, fusion detection, row programs, matmul epilogues
  executor/       Plan execution: lowered plan, compiled plan, op dispatch, sequential fallback
  backend/
    webgpu/       WebGPU backend: tile-IR compiler, buffer pool/arena, fused kernels
      matmul/     Tiled matmul with shape-class tuning and K-split
      ops/        Per-op GPU implementations (elementwise, reductions, views)
    cpu/          CPU reference backend (used by gradcheck and tests)
  core/           Shape utilities (broadcast, strides, sizeOf)
  nn/             Module system: Linear, LayerNorm, Embedding, dropout, checkpoint
  optim/          Adam/AdamW (fused GPU kernel), SGD, GradScaler, LR schedulers
  testing/        gradcheck (numerical gradient verification)
```

## Execution Pipeline

```
tensor.add(other)                    # User code
  → frontend/Tensor.add()           # Records autograd node
  → runtime/RuntimeEngine.add()     # Creates LazyIRNode (no GPU work)
  ...more lazy ops...

await tensor.item()                  # Triggers materialization
  → RuntimeEngine.force()
  → buildMergedPlan()                # DFS from root, collects pending nodes
  → executePlanOptimized()
      1. computePlanFingerprint()    # Structural hash (ops + shapes + dtypes)
      2. Cache lookup:
         HIT  → reuse LoweredPlan
         MISS → analyzeGraph():
           3. Graph rewrites (identity casts, CSE, DCE, algebraic identities)
           4. Pattern detection (priority order):
              - Matmul epilogue chains (cast → bias → activation)
              - Row programs (elementwise → reduce → elementwise → reduce)
              - Elementwise fusion groups
           5. buildLoweredPlanFromAnalysis() → LoweredPlan
      6. Execute actions: fused, matmul-epilogue, row-program, sequential, adam-batch
      7. (Step 2) Record GPU commands → CompiledPlan
      8. (Step 3+) Replay CompiledPlan directly (zero analysis)
  → Result materialized
```

## Key Abstractions

**LazyIRNode** — A node in the lazy computation graph. Op code, inputs (LazyRef[]), shape, dtype. Not executed until forced. LazyRef is `pending` (another node), `materialized` (GPU buffer), or `scalar` (inlined constant).

**LoweredPlan** — Cached sequence of typed execution actions. Action types: `fused` (elementwise group), `matmul-epilogue` (matmul + chain), `row-program` (multi-reduction), `sequential`, `adam-batch`. Uses plan-node indices, not buffer pointers.

**CompiledPlan** — Flat sequence of GPU primitives (alloc, dispatch, copy, write, barrier) with abstract slot indices. Recorded from one normal execution, replayed with zero host analysis on subsequent steps.

**RowProgram** — Multi-phase per-row computation compiled to a single `perRowKernel`. Captures patterns like softmax (max-reduce → exp → sum-reduce → div) and variance (mean → sub → square → mean → rsqrt).

**OP_REGISTRY** — Single record defining every elementwise op: arity, fusible/vectorizable flags, WGSL codegen hints, dtype rules, and autograd functions. Adding an entry makes the op available to fusion, dispatch, and autograd automatically.

**BufferArena** — Per-plan persistent GPU buffers. Stable identities across steps enable near-100% bind group cache hit rate.

## Caching Layers

1. **Fingerprint → FusionAnalysisTemplate** — Structural hash caches graph analysis (node reorder, segments, epilogues). Runs once per unique graph structure.
2. **Template → LoweredPlan** — Built on first execution. Reused every step with same fingerprint.
3. **LoweredPlan → CompiledPlan** — Recorded on second execution. Third+ steps replay directly.
4. **Pipeline cache** — `GPUComputePipeline` keyed by WGSL source hash.
5. **Bind group cache** — Sequence-indexed. Arena stability makes keys match across steps.
6. **Params buffer cache** — Per-dispatch uniform data cached in CompiledPlan slots.

## Tile-IR Compiler

Block-level kernel DSL (~9,000 lines). Authors write at workgroup level; the compiler handles thread mapping, shared memory, barriers, vectorization.

- `tile-ir.ts` — `KernelContext` API: `wgReduce`, `stridedFor`, `emitStore`, `barrier`, `sharedArray`
- `tile-ops.ts` — Block-level ops: `blockLoad`, `blockStore`, `dot`, `dotRow`, `accumRow`
- `tile-lowering.ts` — Lowers block ops to scalar/vec4 WGSL with thread mapping
- `tile-compiler.ts` — Emits final WGSL from statement IR

Used for: all elementwise ops, reductions, matmul, attention, layernorm, RMSNorm, cross-entropy, Adam, row programs, fused groups.

## Autograd

Forward ops record backward closures in `AutogradNode`s. `backward()` topologically sorts the DAG, forces all saved tensors in one merged plan, then walks nodes in reverse calling backward functions. Gradients accumulate via `add()`. Gradient functions for elementwise ops live in `OP_REGISTRY`; complex ops (matmul, softmax) define gradients in `frontend/torchlette.ts`.

## Known Semantic Limitations

- **In-place self-aliasing** — `tensor.mul_(tensor)` crashes (broken lazy ref). `add_(self)` works because the graph doesn't read-after-write. Use non-in-place `mul(a, a)` instead. Self-aliasing in in-place ops is not detected or prevented.

- **Disposing intermediates breaks autograd** — Calling `dispose()` on a tensor that's part of an active backward graph kills the saved-for-backward reference. Gradients will be null or wrong. Unlike PyTorch, saved tensors share lifecycle with the user handle (no independent refcounting). Use `tidy()`, `compile()`, or `beginStep()`/`endStep()` for automatic lifecycle management instead of manual `dispose()`.

- **No double backward** — `backward()` clears the autograd graph. A second call is a no-op. No `retain_graph` equivalent. By design for memory efficiency.

- **View semantics differ from PyTorch** — Views (`reshape`, `narrow`, etc.) are lazy graph aliases, not buffer aliases. Mutating the original via in-place ops after creating a view does NOT affect the view. In PyTorch, views share storage and see mutations.

- **Buffer pool non-determinism across model instances** — Running two independent models on the same WebGPU device can produce different gradient values due to buffer pool reuse of stale data. Single-model training is deterministic within a step. This only affects test tooling that compares two models on the same device.

## Testing

- **gradcheck** — Numerical gradient verification via finite differences. Auto-generated for all OP_REGISTRY ops with grad specs.
- **Stress tests** — `test/stress-semantics.spec.ts` covers aliasing, view safety, broadcast gradients, autograd corner cases, and complex backward chains.
- **Unit tests** — 850+ tests across 55 files covering ops, autograd, fusion, modules, optimizers.
- **Integration** — DistilGPT-2 finetuning regression test validates loss values and fusion stats.
- **Commands**: `npm run test` (CPU + WebGPU), `npm run test:webgpu`, `npm run test:browser`
