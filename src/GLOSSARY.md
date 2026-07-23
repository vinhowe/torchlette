# Torchlette Glossary

The load-bearing vocabulary, each term grounded in the code that defines it.
This file ships **in `src/`** on purpose: it is the map a reader reconstructs
the spine from without leaving the source tree. Every pointer below resolves to
a real symbol (file : symbol) as of 2026-07-23. When a term names a **deleted**
or **historical** mechanism, the entry says so and defines what it meant, so
the word is legible where it still appears in comments.

A one-line orientation before the terms: a user op becomes a **plan** (an
ordered graph slice), the plan is cached as a **template** keyed by structure,
and on execution `replayPathFor` (executor) picks how it runs — replay a
**compiled plan**, **build** one from IR, or run the **lowered** reference path.
Memory is one model: **rc** (existence) + the **owner set** (identity) +
**generation stamps** (time) decide **persistent** vs **step-scoped**, and the
step boundary **demotes** the step-scoped storages (`isDemotableAtBoundary`).

---

## Plans and execution paths

**plan** — a graph slice scheduled for execution. It appears in three distinct
typed forms along one build pipeline (analysis → lowered → generated → compiled):

- **lowered plan** — `src/executor/lowered-plan.ts` : `interface LoweredPlan`
  (built by `buildLoweredPlanFromAnalysis`). The op-semantic layer: an ordered
  `LoweredAction[]` (`src/executor/lowered-plan.ts` : `type LoweredAction`)
  derived purely from graph analysis. It is the reference/interpreter execution
  path and optionally *holds* a `compiledPlan`, `scalarSlots`, `scalarTable`.
- **compiled plan** — `src/executor/compiled-plan.ts` : `interface CompiledPlan`.
  The op-*opaque* layer: a flat `GpuCommand[]` + `SlotSource[]` + `NodeResult[]`
  that "knows nothing about op semantics" (file header). The replayable GPU
  command stream. Built by `buildCompiledPlanFromGenerated`.
- **generated stream** — `src/executor/stream-generate.ts` :
  `interface GeneratedStream` (produced by `generateStream`). The intermediate:
  per-action GPU command segments emitted directly from a lowered plan, with a
  `fullyCovered` flag. "generated plan" = a compiled plan built from a generated
  stream — the one and only compiled build source today (see **recording**).

**replay path** — how a lowered plan executes *this* call: `compiled-replay`
(replay the compiled plan — the steady state for a recurring template),
`build-from-ir` (first execution: build a compiled plan from the lowered IR,
then replay), or `lowered` (the reference path). The decision is named in one
place: `src/executor/executor.ts` : `replayPathFor` (a pure verdict + reason),
consumed by the three branches of `executeLoweredPlan`.

**template** — a cached plan keyed by a structural fingerprint
(`computePlanFingerprint`; the FNV-1a cache is `fusionAnalysisCache` in
`src/executor/executor.ts`). Scalars are deliberately **excluded** from the
fingerprint (`src/executor/scalar-table.ts`) so a value change (an LR step)
does not miss the cache; per-template identity is `templateFp`. The cache
boundary is the engine instance.

**recording** — HISTORICAL. A "recording" was the first normal-path execution
whose live GPU buffer pointers were mapped to abstract slots to *build* a
compiled plan (the referent of "record"/"recorded" comments in
`compiled-plan.ts` / `stream-generate.ts`). **The recorded build path was
deleted (2026-07)**; build-from-IR (lowered → generated → compiled) is now the
sole compiled build source — `src/executor/executor.ts` : `buildFromIRActive`.
The generated stream must stay byte-identical to what the old recording
produced — that determinism is the gate in `src/executor/stream-diff.ts`.
`recordedCopyBufferToBuffer` (`compiled-plan.ts`) survives as a live
command-emitter, not a recorder.

**harvest** — the compiled-plan step that maps executed node outputs back to
graph `StorageHandle`s / `NodeResult`s (the "harvest ledger" in
`src/executor/compiled-plan.ts`). A **harvested view** is a VIEW handle the
harvest creates whose base retain is owned *solely by the plan* —
`src/graph/types.ts` : `StorageHandle.planOwnedBaseRetain` (plus the cross-plan
`(templateFp, nodeIndex, outputIndex)` identity stamp). The storage tracker
honours this special class (`storage-tracker.ts` `destroyStorageIds` skips the
double-free, and `viewBaseIsLive` exonerates a reaped harvest view whose buffer
still aliases its live base).

---

## Memory: the one-model liveness layers

**rc (refcount)** — `src/graph/refcount.ts` : `rcRetain` / `rcRelease` /
`rcGet`. Per-`storageId` reference counts; **rc = EXISTENCE** (is *anything*
still holding the buffer). `destroyUnreachable` reaps at rc ≤ 0. rc alone is an
unsound *classifier* (a single wrapper legitimately holds ctor+materialize
rc = 2), which is why the owner set exists.

**owner set** — `src/graph/storage-tracker.ts` : `StorageTracker._ownerSet`
(the set W(s) of live wrapper WeakRefs pointing at storage s). **owner set =
IDENTITY** (which wrappers hold it, and is any of them a kept holder). It is the
single source for the liveness classifier `_derived`, replacing a former
steal-able single owner *slot* (task #70 D2).

**generation stamp** — `src/graph/storage-tracker.ts` : `_wrapperGen` /
`stampWrapperGen` (keyed by engine epoch, `src/core/epoch.ts`). **stamp = TIME**:
a wrapper is stamped at *construction*, and a queued step boundary records the
closing generation; comparing `stamp > boundaryEpoch` separates *this* step's
tensors from the next step's already-lazily-built work. Wrapper-level, not
storage-level (a persistent tensor's storage can be replaced mid-step).

**persistent vs step-scoped** — the two-tier reachability, classified by
`src/graph/storage-tracker.ts` : `StorageTracker._derived`. **persistent** :=
`∃ w ∈ W(s) : w ∈ SNAP ∪ REG` — SNAP = `_stepStartTensors` (the
`snapshotForStep` set: params/optimizer state alive at step start), REG =
`_registeredState` (state a module/optimizer explicitly `registerState`d,
gen-independent). A storage is **step-scoped** (a step temporary) iff a live
wrapper owns it and *no* kept holder (persistent / graph-retention / sidecar-pin
/ live-view-base) keeps it.

**demotion / demote** — releasing the one step-scoped wrapper claim of a
step-scoped storage at the step boundary (markStep), so its buffer returns to
the pool deterministically without waiting on GC. The decision is named in one
place: `src/graph/storage-tracker.ts` : `StorageTracker.isDemotableAtBoundary`
(`demotable == _derived().stepScoped`); the sweep that consumes it is
`releaseStepTemps`.

---

## Fusion / scheduling structure

**island / partition / segment** — DISTINCT, at three altitudes:

- **island** — `src/compiler/fusion-detect.ts` : `interface Island`
  (`IslandKind = "sequential" | "fused" | "reduction"`). The finest unit: a
  group of plan-relative member positions that execute together. The *analysis*
  grouping.
- **partition** — `src/compiler/fusion-detect.ts` : `interface PlanPartition`
  (`islands: Island[]` + a `boundaryHash` identity token). The whole plan's
  dispatch partition — the reified set of islands as first-class data; its
  `boundaryHash` is mixed into the template fingerprint. One partition per graph.
- **segment** — the *executable realization* of islands: `CachedSegmentDesc`
  (`src/executor/executor.ts`), run by `src/executor/segment-executors.ts`,
  diffed by `src/executor/stream-diff.ts`, emitted per-action as
  `GeneratedStream.segments`. The *lowered/executed* form of the islands.

Relationship: **islands are the analysis grouping; the partition is those
islands reified as data; segments are their executed form** — they share
positions and emission order. See `docs/islands-design.md`.

**skeleton** — `src/schedule/types.ts` : `type Skeleton` (a discriminated union
`{visibility:"derived", schedule}` | `{visibility:"opaque", kernelRef, …}`): the
placement/loop structure of a kernel family (matmul / elementwise / reduction
skeletons in `src/schedule/`). NOTE: distinct from the **tape skeleton** (the
derived compiled form / ordered plan-fp sequence) referenced in
`src/core/step-object.ts` — that one is deleted-adjacent (see **tape**).

**realizer** — the component that turns a schedule into an actual backend
kernel. Registry: `src/schedule/realizers/registry.ts` (`TRITON_REALIZER` /
`TritonRealizerEntry`). Ruling R13 (stated in that file's header): there is NO
generic `Realizer<>` protocol — each realizer is concrete (WGSL kernel, Triton
emit, model-editor emit). `src/schedule/opt-step-realizer.ts` : `lowerOptStepBody`
realizes optimizer-step bodies.

**spec** — two live meanings; there is no type literally named `ExecutionSpec`:
`src/executor/execution-declaration.ts` : `interface ExecutionDeclaration` is the
"command-stream stratum as DATA" describing how an op-family node decomposes (the
closest thing to an execution spec); `src/schedule/opt-step-realizer.ts` :
`OptStepRealizerSpec` is the optimizer-step realizer's spec. ("spec" also appears
as placeholder schema strings in `schedule/types.ts` — not load-bearing.)

---

## Historical / deleted mechanisms (defined so the comments read)

**witness** — HISTORICAL mechanism, now deleted; the term survives in comments.
A "witness" was a producer's cross-plan output edge being *observed/stamped* so
later consumers would not prune a still-needed value. The per-producer witnessed
oracle and the cross-plan-edge store it fed are **deleted / inert**:
`src/core/cross-plan-edges.ts` is REDUCED TO INERT STUBS (P4b-R R3.3) — the
store was populated only by the deleted step-tape recorder's `publishCrossPlanEdges`
at K_w=2 eligibility, so its queries now return their constant tape-off values.
A dormant successor exists — `src/executor/executor.ts` : `loweredWitnessRuns`
stamps the harvest outputs of a template that has "converged to lowered" for K_w
consecutive executions, reproducing witnessing without the deleted recorded build.

**tape** — TWO distinct tapes; disambiguate:

- **step-tape** — DELETED (P4b-R). The step-tape recorder/replay was removed:
  `src/core/step-object.ts` header ("the step-tape recorder + replay that this
  module once projected a witnessed StepObject from were DELETED"). Residue: the
  monotonic op-sequence counter "the step-tape's structural [ordinal]" in
  `src/graph/node-factory.ts`, and `StepObject`-family types kept as the
  declarative schema (the tape-consuming constructor `deriveStepObject` was
  pruned). "re-witnessing the tape" in executor comments refers to this
  now-conceptual step-tape.
- **autograd tape** — LIVE and distinct: the backward-graph recording in
  `src/frontend/autograd.ts` (saved-for-backward slots, `cleanupAutogradGraph`).
  The word "tape" is used loosely there; the mechanism is the saved-slot autograd
  graph.
