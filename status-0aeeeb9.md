# Project Status: 0aeeeb9

## State of the Codebase

**50.5k LOC** across 108 source files. **841 tests, 0 failures.** `tsc --noEmit` clean (135 strict errors fixed). Zero TODOs/FIXMEs.

### Recent Commits (this branch)
- `0aeeeb9` Fix all 135 strict TypeScript errors for zero-error tsc --noEmit
- `82912b1` Add _creationOp helper + consolidate tril/triu via _unaryOp (-39 lines)
- `6101f03` Thread matmul epilogue directives from graph-compiler to segment-executors
- `0b96995` Merge dtype rules into OP_REGISTRY as single source of truth
- `c8f6e2e` Deduplicate isContiguousStrides + consolidate static guard evaluation
- `ebf0ef8` Smart _unaryOp dtype handling + merge _comparisonOp into _binaryOp
- `1486547` Remove dead OP_REGISTRY fields (expr, vectorExpr, outputDtype, needsVectorConstants)

### Performance Baseline (512-token DistilGPT-2)
- Steady-state: ~49ms/step wall clock
- GPU: ~64ms total (matmul 39%, attention 8%, optimizer 7%)
- Memory: 5.5GB current, pool 52% reuse, 1 new alloc/step, leak status OK
- Fusion: 1156 nodes, 183 fused (15.8%), 20 fusion groups
- Bind group cache: 96.2% hit rate, 10 submits/step

### Architecture Layers
```
Frontend (frontend.ts, frontend-tensor.ts) → Autograd + Autocast
    ↓
RuntimeEngine (runtime/engine.ts) → LazyIRNode graph construction
    ↓
Graph Compiler (engine/graph-compiler.ts) → Pattern detection + rewrites
    ↓
Executor Pipeline:
  executor-optimized.ts → Fusion analysis + template caching
  executor-lowered.ts   → Cached plan replay + buffer arena
  segment-executors.ts  → Matmul epilogue / reduction / compound execution
    ↓
WebGPU Backend → Fused kernels (attention, LayerNorm, Adam, cross-entropy, matmul epilogue)
```

---

## Assessment

### Strengths
- Clean layer separation, no backward imports
- Structural fingerprinting enables template caching across steps
- Per-plan buffer arenas give 96%+ bind group cache hit rates
- Dispatch replay bypasses 95% of JS dispatch overhead
- Zero TODO/FIXME/HACK markers
- Strict TypeScript: zero `tsc --noEmit` errors

### Weakness: 4,600 LOC of Untested Executor Code

| Module | LOC | Unit Tests |
|--------|-----|------------|
| executor-lowered.ts | 1,364 | 0 |
| executor-optimized.ts | 753 | 0 |
| segment-executors.ts | 858 | 0 |
| op-dispatch.ts | 602 | 0 |
| storage-tracker.ts | 435 | 0 |
| graph-rewrites.ts | 332 | 0 |
| graph-compiler.ts | 516 | 0 |

Implicitly covered by integration tests (DistilGPT-2 training, 52 fusion tests), but no isolated tests for plan caching, replay identity, arena conflicts, lifetime analysis edge cases, or rewrite pass ordering.

### Weakness: Detect-Twice Pattern in Execution

Analysis (graph-compiler.ts) detects matmul epilogues and reduction patterns, but segment-executors.ts re-detects the same patterns during execution. Directive threading is partially done (matmul epilogue + reduction directives threaded) but the old detection paths still exist alongside.

### Weakness: No Formalized Pass Infrastructure

Graph rewrites (identity-cast elimination, redundant-contiguous elimination, algebraic identities) are ad-hoc function calls. No `GraphPass` interface, no `runPasses()` orchestrator, no CSE/DCE passes.

---

## Plan: B then A — COMPLETED

### Phase B: Test Coverage ✅

**110 new tests across 5 files, all passing.**

1. **test/graph-rewrites.spec.ts** (37 tests) ✅
   - eliminateIdentityCasts: bypass f32→f32, keep f32→f16, materialized inputs, skip existing results, consumer count updates
   - eliminateRedundantContiguous: bypass compute-op inputs, keep view-op inputs (transpose, reshape), skip materialized
   - eliminateAlgebraicIdentities: mul(x,1), add(x,0), sub(x,0), div(x,1) bypassed; non-commutative cases preserved
   - CSE: duplicate elimination, input order sensitivity, RNG/side-output exclusion, payload/shape/dtype differentiation, scalar refs
   - DCE: zero-consumer removal, cascading, output preservation, result-bearing node preservation
   - redirectConsumers: multi-consumer chains, consumer count bookkeeping
   - GraphPass interface: SIMPLIFICATION_PASSES registry, runPasses stats, pass composability

2. **test/graph-compiler.spec.ts** (12 tests) ✅
   - analyzeGraph() identity ordering, matmul epilogue claiming, bypass exclusion from fusion
   - Consumer counts reflect rewrites, reduction preamble/epilogue directive generation
   - Matmul epilogue directives with full plans, empty directive cases

3. **test/storage-tracker.spec.ts** (22 tests) ✅
   - register/unregister, markReachable/markUnreachable, isReachable, getReachableIds
   - destroyUnreachable: count, reachable preservation, view handling
   - View aliasing: base kept alive by reachable view, transitive chains
   - destroyUnreachableSince: scoped destruction, reachability respect
   - canSafelyRelease: unreachable+no views, reachable rejection, view base rejection
   - releaseBufferImmediate: owned buffer destruction, view skip
   - debugCounters: tracking and reset

4. **test/lowered-plan.spec.ts** (22 tests) ✅
   - isDataSourceOp, isViewOp, ENCODER_COPY_OPS classification
   - LoweredPlanBuilder: all 10 action types (fused, sequential, view, data-source, prologue-skip, matmul-epilogue, reduction-preamble, reduction-epilogue, reduction-fusion, adam-batch, compound, reclaim)
   - Action order preservation in complex sequences

5. **test/lifetime-analysis.spec.ts** (14 tests) ✅
   - Size class utilities: minimum, powers of 2, rounding, inverse, round-trip
   - analyzeLifetimes: linear chains, multi-consumer, buffer sizes, diamond patterns
   - findDeadTensorsAtStep: dead detection, output exclusion, already-released exclusion

### Phase A: Pass Infrastructure ✅

All items were already implemented before Phase B testing began:

1. **Phase A.1: GraphPass interface** ✅ — `graph-rewrites.ts` has `GraphPass`, `SIMPLIFICATION_PASSES`, `runPasses()`
2. **Phase A.2: CSE pass** ✅ — `eliminateCommonSubexpressions()` with structural keys, NON_CSE_OPS exclusion
3. **Phase A.3: Directive threading** ✅ — `graph-compiler.ts` pre-computes `matmulDirectives` and `reductionDirectives`; `segment-executors.ts` consumes them with fallback detection for safety
4. **Phase A.4: DCE pass** ✅ — `eliminateDeadCode()` with iterative fixed-point, consumer count decrement

### Typing Invariant

All table-driven or single-source-of-truth refactoring MUST preserve typed signatures for library consumers. The public API surface (Tensor methods, Torchlette methods, optimizer classes) keeps explicit TypeScript types. Internal dispatch consolidation uses typed registries, not `any` or `unknown` escape hatches.
