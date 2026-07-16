# Stage 4: Compile-from-IR with Graph-Liveness Memory Planning

*Design, 2026-06-12. Companion to `architecture-debt.md` (stages table, row 4).
Written at the end of the cycle that landed stages 0–3; every claim about
existing machinery refers to code in-tree at commit ed779c6.*

> **TASK #43 "recorded-build sunset" STATUS (2026-07-11, deletion-attempt pass):
> STOPPED at the map — the recorded build is NOT deletable yet, and the named
> flag it targeted is already dead.** See the full map at the bottom of this doc
> (§ "Task #43 recorded-build sunset — deletion MAP + STOP"). One-line summary:
> `TORCHLETTE_GENERATED_PLAN` (the flag this pass was asked to sunset) died on
> 2026-07-08 (B5, inc-3c) and appears in ZERO live code — the sunset is already
> executed. The recorded build (`buildCompiledPlan` + the `record*` hooks) has
> three live masters on the DEFAULT path — the uncovered-plan census fallback
> (correctness, fires for real generator gaps), the `STREAM_GENERATE=1` verify
> gates (2 of the 4 load-bearing gates), and the `BUILD_FROM_IR=0`/`COMPILED_PLAN=0`
> opt-outs (separate flags, out of this campaign's scope). Deletion is gated on
> 4.4-coverage (full generator coverage) + stage-3 rematerialization unification,
> both designed-but-not-built. Per the campaign's STOP rule, no deletion forced.
> **UPDATE 2026-07-12 (4.4-residues pass):** the four residues are worked to a
> verdict apiece (see § "Task #43 4.4-COVERAGE — the four RESIDUES revisited").
> (c) decode strided-view+max and (d) config-missing transients are resolved
> ACCEPTABLE-BY-DESIGN (d does NOT block deletion; c is inference-only). (a)
> row-program-scalar-steptemp and (b) data-source:full remain the TWO recurring
> TRAINING-path bails that BLOCK deletion — both need the same captured-state
> "materialized value → compiled-replay slot, per-step, no over-harvest"
> primitive; (b) was attempted via generation and reverted on the ledger gate
> (the generate-more-plans path entangles with the fundamental over-harvest).

## Why (tied to the ledger)

The compiled plan today is a **trace**: one normal execution runs with a
recorder attached, and replays re-issue the recorded GPU calls. Two
structural debts follow, and the bug ledger shows both being paid repeatedly:

1. **The recorder must see every effect.** Ten `record*` hooks exist
   (dispatch/alloc/copy/write/clear/volatile-uniform/barrier + the recorded
   copy helper + params-slot assignment + bind-group capture). Every hook
   that was missing at some point was a silent-training-corruption bug found
   by loss archaeology: unrecorded `copyBufferToBuffer` (embedding grads
   +1×/replay), unrecorded `clearBuffer` (stale scatter-add accumulators),
   unrecorded uniform rewrites (frozen `step_size`, wrong LR schedule for
   weeks), params bytes baked at record time (now guarded, 8115349). A
   trace-based replay is only as correct as the discipline of the thing it
   traces — discipline does not scale.

2. **Memory is emergent, not planned.** Buffer assignment falls out of
   dispatch order (the per-position arena), so ownership ended up split
   across seven regimes (pool, arena, plan-pinned, params-sequence cache,
   tile-config caches, f16WeightCache, packed-optimizer cache), and the
   worst bug class of the cycle — UAF, double-release, destroyed-buffer
   submits, stale-grad replays — lived on the seams BETWEEN regimes. The
   liveness arena + planned buffers (3b45531) bounded the memory but kept
   the regimes; donation (33be5fb) reuses dying buffers but as a kernel-side
   special form.

Compile-from-IR inverts both: the command stream and the buffer assignment
are **derived from the lowered plan**. There is nothing to forget to record,
and every buffer has one owner: the planner.

## What already exists (reuse, don't rebuild)

- **Lowered plan** (`lowered-plan.ts`): typed actions (sequential, fused,
  matmul-epilogue, adam-batch, batched-reduction) over a fingerprint-cached
  template. This IS the IR to compile from.
- **Liveness analysis** (`executor.ts`): per-node last-reader in action-index
  space, cross-plan consumer protection (6d29f5c), WAR ordering + checkpoint
  barriers + affinity in one Kahn pass (98eea29, b791d72).
- **Scalars-as-data** (d822be9, 2809588, 7101ebb): scalar table, payload
  fingerprinting, thrash detector. Per-step values already flow as data.
- **Planned buffers** (aa2a7f5): proof that replaying a fixed buffer
  assignment under the bounded pool works (124M/Medium validated). Stage 4
  replaces "pin what the recording happened to allocate" with "assign what
  the planner computes".
- **Stream-level validation culture**: every optimized path has a
  differential gate that crosses its activation threshold. Stage 4's
  migration gate is a differential at the command-stream level.

## Target architecture

```
lowered plan (template-cached, fingerprinted)
   │
   ├─ DispatchPlanner: per-action → DispatchPlan
   │     {pipeline, workgroups, bindingRoles, tempRequests, uniformSpec}
   │     (declarative; no GPU calls)
   │
   ├─ MemoryPlanner: liveness intervals (+ temps) → BufferAssignment
   │     interval allocation over size classes; donation = assignment
   │     decision; persistent/external/in-place constraints honored
   │
   └─ StreamEmitter: (DispatchPlans, BufferAssignment) → GpuCommand[]
         the SAME GpuCommand stream format replays execute today
```

Recording is **demoted to cross-check**: in debug/CI mode, run the recorder
alongside and diff recorded vs generated streams (pipelines, binding slots,
workgroups, copy/clear/uniform commands). Divergence = bug in one of them,
found at the seam instead of in a loss curve.

## Phases (each independently shippable, each gated)

### Phase 0 — Stream differential harness + DispatchPlan interface
- Canonical serialization of a `GpuCommand[]` stream (pipeline identity by
  cache key, buffers by slot, bytes for uniforms/params).
- `diffStreams(a, b)` with attribution (which action produced the
  divergence).
- Determinism gate: record the same template twice, diff → must be empty
  (pins stream determinism, which everything below assumes).
- `DispatchPlan` type + registry stub; no behavior change.

### Phase 0+1 status (2026-06-12): SHIPPED
Phase 0 landed in ca7ff56 (determinism gate caught two real ownership bugs
at build: scalar-table buffers adopted as persistent slots; cache-owned tile
configs destroyed at plan teardown). Phase 1 landed opt-in in 067be59, then
phase 1.5 (below) made it the default and DELETED the pin mechanism.

### Phase 1.5 — Cross-plan packing (SHIPPED 2026-06-12)
Finding: the pin mechanism's cross-plan sharing was never "forward's dead
activations serve backward" — adoption at build time removes a plan's
recorded buffers from pool circulation BEFORE later plans record, so the
only sharing that ever existed was earlier plans' intra-plan-dead TEMPS
(investigated via an "external death claims" prototype that re-expressed
recorded reuse identity-independently: structurally zero witnesses).

Since plans within a step execute strictly sequentially, ANY non-result
planner buffer is safe to share with any other plan, in any order. Temps
draw from a step-scoped shared PlannerRegistry (module-global, reset at
engine-instance boundaries with a generation guard); results keep exclusive
entries. This shares deterministically and completely what the pool shared
opportunistically.

Gates (A100 same-machine A/B): peak memory planner BEATS pin ~14% on both
distil@512 (5.07 vs 5.88GB) and medium@512 (15.0 vs 17.5GB); steady-state
speed parity; fullstack parity to fp noise; determinism PASS; 124M
regression PASS both modes; suites green both modes. Deleted: adoption
refcounting, pool-origin tracking, allocBuffers, planned-bind replay
fallbacks, bufferPool.adoptBuffer. TORCHLETTE_MEMORY_PLANNER=0 now disables
compiled replay wholesale (lowered path) — a dynamic-alloc replay would
leak ownerless temp buffers, so it is not offered.

### Phase 1 — Memory planner against RECORDED streams
Keep recording for WHAT to dispatch; replace the buffer assignment:
- Inputs: per-slot byte sizes + first/last use (from the recorded alloc/
  bind history + liveness), persistent/external/pinned sets, in-place
  aliases (donation, scatter dst), chunking constraints
  (maxStorageBufferBindingSize), WebGPU binding-aliasing rules (one buffer
  must not be bound writable twice in a dispatch, nor read+write).
- Algorithm: greedy interval allocation (sort by start, first-fit into
  freed intervals per size class; offset packing within large slabs is a
  later optimization — size-class granularity matches the pool today).
- Output: slot → buffer table, owned by the plan; allocated once, freed on
  template eviction via the fence-gated path.
- Gates: peak-memory ≤ planned-buffers mode on distil/Medium/124M; full
  ladder (fullstack parity both modes, regression both modes, suites,
  1-peer A/B); donation subsumption check (the kernel-side donation can be
  disabled when the planner aliases output to dying input — verify equal
  memory).
- Expected wins: deterministic memory bounds; arena dependence gone from
  compiled mode; likely closes the foreach 9.3→~5GB gap → unblocks the
  optimizer-island endgame (architecture-debt stage 3).

### Phase 2 — Stream generation for declarative ops
- Tile-IR kernels (configs are already data; TAG_UNIFORM packers exist),
  fused elementwise recipes, creation ops (TAG_WRITE/TAG_CLEAR semantics
  already defined), strided-scatter DMAs, cat copies.
- Executor: if every action in a plan is plannable → emit generated stream
  (validated against recording by the phase-0 differential in CI); else
  record/replay as today. Coverage is a counter, not a cliff.

### Phase 2 status (2026-06-13): PROOF POINT REACHED
`src/executor/stream-generate.ts` generates the GpuCommand stream from the
lowered plan; the executor's `TORCHLETTE_STREAM_GENERATE=1` hook diffs it
against the recording. Diffing is **segment-aligned** (per-action, modulo a
consistent slot bijection — the stream is slot-renaming-invariant, so
raw-slot `diffStreams` is right only for record-vs-record determinism); a
fully-covered plan adds a flat command-count assertion. Generators landed:
data-source (tensorFromArray/zeros), binary/unary/cast/contiguous/gelu,
sum (full reduction), unscaleGrad (volatile TAG_UNIFORM), stridedScatterCopy
(in-place DMA), fused recipes (`planFusedKernel`, post-hoc donation
detection), adam-batch (`planAdamStepDispatch` + `lookupPackedBuffers` +
`planPackedGroups`, f16 variant), matmul-epilogue (`planTiledMatmul`
standard path, reads the action's `cachedDispatchConfig`).

**Proof point: the optimizer plan is FULLY GENERATED** — 439/439 actions,
437/437 segments verified, flat 1746/1746 commands, zero divergences. The
differential paid for itself en route: it caught a latent stale-`zeros`
replay hazard, a recording-attribution bug (fused/epilogue actions
inheriting the prior node's index), and a grid-normalization mismatch —
all at build time, never in a loss curve.

Forward 256/289, cross-entropy plan 89/91, backward 255/414.

**The phase-2/phase-3 boundary (the line, characterized):** everything
DECLARATIVE or carrying a lowering-time config (matmul-epilogue's
`cachedDispatchConfig`) generates. The remainder — bare `op:matmul` (67),
fused LayerNorm/attention/cross-entropy, gather, narrowBackward — bails with
`released-input`: their inputs are **liveness-released by plan-build time**,
so geometry that depends on live input strides (matmul transpose detection)
or op-internal workspaces cannot be reconstructed at generation. matmul-
epilogue worked *because* its geometry was captured at lowering (when inputs
were live) and cached. That is exactly phase 3's mandate, now concrete:
capture each imperative op's resolved plan at lowering and cache it where the
generator reads it. A bare-matmul generator was built, measured (100%
`released-input`), and reverted as dead code — `planTiledMatmul`/
`planBinaryDirect`/`planUnaryDirect`/`planCastDirect`/`planContiguousDirect`
plan-cores were kept (the dispatchers consume them; the splits are sound).

### Phase 3 — Planning interfaces for imperative ops
- matmul (tile config selection, K-split temps, epilogues), attention
  (workspaces, D-precompute), fused LN/CE (partials), adamStep/packed.
- Each gets `plan(shapes, dtypes, config) → DispatchPlan` next to its
  imperative dispatch; the imperative form remains as the lowered-path
  executor. This is where op-internal allocations become planner-visible
  temps (today they are the "persistent slot" pin class).
- adamStep's config payload becomes planner uniform data → the TAG_UNIFORM
  per-op registry dies; with foreach default (unblocked by phase 1), the
  mega-op itself can go.
- **Concrete entry point (from the phase-2 boundary):** the unifying need is
  capturing resolved geometry AT LOWERING TIME (inputs live) and caching it
  where the generator reads it post-hoc. matmul-epilogue already does this
  via `cachedDispatchConfig`; bare matmul has no per-action object (it's a
  `sequential` action) → cache on the lazy node (node-keyed plan cache) from
  the matmul op handler's first live execution. First increment: bare-matmul
  node plan-cache (covers 67 matmuls, reuses `planTiledMatmul`); then the
  same mechanism for the fused LN/attention/CE kernels (their plan() also
  declares the op-internal workspace temps).

### Phase 3 progress (2026-06-13)
Covered: all declarative tile ops; matmul (bare + epilogue + K-split, 100%
both directions); LayerNorm fwd + gradX + gradWeightBias (100%, incl. the
op-internal partial-sum WORKSPACE — cached workspace buffers bind as
persistent slots via read-only lookup, the K-split-temp/packed-buffer
mechanism); narrowBackward (grad shape captured at lowering via
`cachedInputShapes` — its grad is a released multi-output extra);
adam-batch; fusedAttention FORWARD. Three capture mechanisms cover the
"generator can't derive it post-hoc" cases: `cachedMatmulPlan` (geometry),
`cachedInputShapes` (released multi-output input shapes), and read-only
workspace/config lookups (`lookupKSplitTempBuffer`, `lookupPackedBuffers`,
`lookupAttentionConfigBuffer`). **Coverage: 100% — every plan in the
canonical trainer (forward, backward, optimizer) is FULLY GENERATED, 0
diverged, command counts equal.** A latent bug fell out:
`getOrCreateConfigBuffer` never populated its cache (re-created the
attention config uniform every dispatch) — fixed (perf + stable identity
for the generator).

**Cross-entropy fwd+bwd (DONE).** `ceFwd`/`ceBwd` are module-level cached
dispatchers (stable identity already), one dispatch each, geometry fully in
the payload config {batchSize, vocabSize, ignoreIndex} — no shape
derivation. Added planCrossEntropyForward/BackwardDispatch + generators.
Output is `allocateOutputBuffer` (arena kind 1, empty inputSlots — NOT the
kind-0 resolveOutputBuffer with aliasing inputs); the diff caught both at
build time. Guards: targets i32/u32 (else ensureI32Targets casts) and
logits/grad contiguous (else asContiguous copies), via
`resolveContiguousInputSlot`.

**Scalar-operand elementwise (DONE).** A binary/unary op with a `kind:
"scalar"` operand now binds the scalar's scalar-table buffer
(`lookupScalarStorage`) as a PERSISTENT slot — the executor refreshes its
value per step, so it's a persistent binding, not a stream write. General
across any scalar-operand elementwise op; the legacy CPU/non-f32
full([],v) fallback (table miss) still bails. Subtlety the diff caught: the
scalar-table buffer is persistent so resolveOutputBuffer excludes it from
the output's aliasing candidates — the ALLOC's inputSlots must list only
poolable operands (track `aliasInSlots` separately from the DISPATCH
bindings).

**Nothing uncovered — the expand/broadcast producers are now covered too.**
The previously-"correctly-excluded" `contiguous(expand)×2` (fwd) and
`mul(a, expand(b))×1` (bwd) were excluded because a released strided view's
stride-bearing layout (broadcast stride-0) "isn't shape-derivable." That is
exactly what capture-don't-synthesize solves: `cachedStridedInputs` captures
each strided-view input's live layout at lowering (expand/transpose/permute/
narrow producers of a sequential elementwise op), and generateSequential
synthesizes the real strided metadata so planBinaryDirect/planUnaryDirect
emit the matching broadcast/gather kernel (the cache key includes strides, so
pipeline identity matches the recording). Generalizes the per-op capture from
attention/reshape into one mechanism for any released strided-view input.
**The recorder is now ready to become a pure cross-check (phase 4).**

**fusedAttentionBackward (DONE — and the "bijection blocker" was stale).**
The 2026-06-13 doc claimed a lifetime-split-slot bijection blocker: dV read
by narrowBackward across a recordAlloc slot split, rejected by the differ's
1:1 gen↔rec map. An instrumented trace (TORCHLETTE_PROBE_SPLIT) DISPROVED
it: every attention-bwd output buffer is allocated once, bound at exactly
one slot, consumed by one downstream view at that same slot — **no split**.
Phase-1.5's planner-derived buffer assignment (one owner per buffer, derived
from liveness — the XLA model) had already dissolved the physical-reuse
artifact the blocker described. The ACTUAL blockers were two general
contiguous-copy gaps, both now fixed:
- **Attention contiguous-prologue.** The op asContiguous's its inputs; a
  non-contiguous one (e.g. dO) records one resolveOutputBuffer ALLOC + one
  planContiguousDirect copy dispatch the generator didn't emit. Fix: capture
  each input's layout at lowering (`cachedInputContig`, replacing the old
  `cachedAllInputsContig` boolean); `generateAttention` replays
  planContigCopy for the non-contiguous ones (mirrors ensureContiguous —
  copy iff isContiguous===false — so the command count matches) and rebinds
  the kernel input to the copy slot.
- **reshape materialization.** A `reshape` (uniquely among the view ops)
  materializes a contiguous copy when its input is non-contiguous AND the
  new shape is stride-incompatible (`inferReshapeStrides===null`) — e.g.
  reshape-of-permute in attention's grad reshaping (dV→permute→reshape→
  narrowBackward). The generator treated reshape as an always-free view.
  Fix: capture `cachedViewInput` for reshape view actions, OBSERVING the
  copy decision (result.buffer !== input.buffer) rather than re-deriving it
  — a non-contiguous input with compatible strides is still a free view, so
  the isContiguous predicate over-triggers (it did, and the diff caught the
  spurious copies as 24 unmatched segments in a non-attention plan). The
  view case emits planContigCopy when the input materialized, else aliases.
Both share `planContigCopy`. bwd 405→413/414; 0 diverged across all plans.

**Needed-intermediate re-execution (DONE — was "the hardest/last class,"
turned out mechanical).** A fused group's internal elementwise nodes that
are consumed OUTSIDE the group but couldn't be promoted to additional fused
outputs (shape ≠ primary, or out of binding slots) are re-executed
sequentially after the fused dispatch (`executeFusedSegment` →
`executeSequentialSegment`). That re-execution never resets the recording
node index (`executeNode` doesn't touch it), so its commands land in the
SAME segment as the fused dispatch. `generateFused` now mirrors it: after
the kernel dispatch, it generates each needed-intermediate's plain
ALLOC+DISPATCH via `generateSequential` (the same elementwise path
`executeNode` takes), resolving inputs against a LOCAL node.id→slot map
seeded with this group's own primary + promoted-additional output slots
(assigned inside `generateFused`, not yet in the outer `nodeSlot`) plus
earlier intermediates, falling back to the outer channel for true external
inputs. Kept atomic: a needed-intermediate that itself can't be generated
bails the whole action. bwd 397→405/414; 0 diverged.

Added since: **gather** (forward, single-sourced on `planGatherDirect`);
**scatterAdd** (embedding backward ×2, single-sourced on
`planScatterAddDirect`; its src grad is a released contiguous `reshape`
view — covered by `contiguousViewShapeDtype`, which derives logical
shape/dtype for the reshape/view/flatten/squeeze/unsqueeze class since
their layout is contiguous-offset-0); **released contiguous-view inputs to
elementwise ops** (the synthesized-metadata fallback now also covers the
contiguous-view class, clearing `cast[no-storage]` fwd+bwd). Strided views
(expand/broadcast) stay bailed — layout not shape-derivable, contiguous
metadata would mislead the kernel; the remaining `contiguous[no-storage]×2`
/ `mul[no-storage]×1` are all `expand` producers, correctly excluded.

**`mean` (DONE — the first-pass "captured-state" diagnosis was wrong).**
`backend.mean()` (the path a plain `.mean()` node takes) does NOT use the
invCount epilogue: it dispatches `sum` (cached full/dim reduction) into an
intermediate buffer, then a `meanDiv` dispatch through the cached `"meanDiv"`
dispatcher with `count` in a UNIFORM (not a buffer). So a mean node is ONE
node → two ALLOC+dispatch pairs, both fully generatable. `generateMean` +
`planMeanDivDispatch` cover it; the sum part reuses
`planFullReductionDispatch`/`planDimReductionDispatch`. The fresh per-call
`invCount` buffer only exists in `meanWithEpilogue` (the path taken when a
user epilogue chain is fused onto the mean) — a different op the trainer's
loss `.mean()` doesn't reach; if it ever shows up it stays the captured-state
case (a cached dispatcher + count-keyed invCount-buffer cache would close it).

**Dim reductions (DONE) — the cached-dispatcher cleanup the `mean` note
predicted, applied to the dim path.** Dim reductions previously dispatched
through a FRESH `createTileKernelDispatcher` per call (no config-buffer
caching, no stable identity — same disease as `mean`). Fixed: route them
through a CACHED dispatcher (`getDimReductionDispatcher`, keyed by
op+inputShape+dims+keepdim+epilogue-signature; the input is forced
contiguous at the dispatch site so strides/outShape/useParallel are all
derived) — `buildDimReductionSpec` is the single source for the dispatch
path and the new `planDimReductionDispatch`. This removes a per-call config
allocation AND gives the generator a stable identity. Covered:
`sum[dim-reduction]` singleton (`generateDimReduction`) and the
`batched-reduction` ACTION (`generateBatchedReduction`). The latter was the
surprise: its members all have reductionSize 2048 (>64), so
`backend.batchedReduction` falls back to N individual `reduction()`
dispatches (the true multi-in/out batched kernel only fires for small
reductionSize), all recorded under `nodeIndices[0]` — so the generator just
loops the per-member dim-reduction emit into one segment, and bails the
whole action if the executor would instead take the true-batched path
(≤64). mean-over-dim stays excluded (its invCount epilogue buffer is fresh
per call — the same captured-state gap as full mean). bwd 392→395/414.

**RESOLVED (was "KNOWN BLOCKER — fusedAttentionBackward / lifetime-split-slot
limit").** The 2026-06-13 claim — narrowBackward reads dV across a recordAlloc
lifetime split, rejected by the 1:1 gen↔rec bijection — was investigated with
an instrumented trace (TORCHLETTE_PROBE_SPLIT) and DISPROVED: attention-bwd
outputs are each allocated once and bound at exactly one slot (no split). The
phase-1.5 planner (one owner per buffer, derived from liveness) had already
removed the physical-reuse artifact. The real blockers were two general
contiguous-copy gaps (attention asContiguous prologue + reshape-of-strided
materialization), both fixed — see the phase-3 progress section above.
Lesson: a "structural limit" pinned to an old architecture's behavior must be
re-confirmed against current code before being treated as a design fork; the
cheap instrumented trace overturned a multi-session redesign plan in minutes.

### Phase 4 — Deletions and dividends

**Pre-cutover deletion audit (2026-06-13): there is NO safe deletion before
the cutover.** Every item below was confirmed load-bearing in the CURRENT
path, so the "deletions" are CONSEQUENCES of the cutover (the generator+planner
becoming the sole build source), not independent dead code to clear first:
- `pinnedBufferSet` — NOT the deleted pin mechanism (that's gone clean: 0 refs
  to adoptBuffer/poolOrigin/allocBuffers/plannedBind). It is the CURRENT memory
  planner's buffer-ownership mechanism (registry entries pinned to survive plan
  teardown; consulted by every destroy/release/harvest path). Load-bearing.
- params-sequence cache (`createParamsBuffer`/`paramsSequenceSet`) — allocates
  every dispatch's params/uniform buffer; core to both the lowered first
  execution and replay. Becomes planner slots only AFTER the cutover.
- per-position arena (`allocateOutputBuffer`/`arenaAllocAt`) — the canonical
  output allocation for every backend op, active in BOTH liveness (default) and
  legacy modes. The pool-hint machinery (`outputSequenceHints`/
  `pinnedOutputBuffers`/`conflictDetected`) is used by `resolveOutputBuffer` +
  the shared encoder during the first (lowered) execution. Not dead.
- legacy unbudgeted arena (`TORCHLETTE_ARENA_LIVENESS=0`) — a SUPPORTED opt-out
  / planner-bug fallback, not vestigial. Dropping it is a product decision.

**DELETION SCOPING (2026-06-14, after the cutover landed & was validated).**
The cutover works but the deletions are MORE gated than originally framed,
because of two facts the cutover work surfaced:
- The cutover is PER-PLAN and coverage-gated. Only the 2 steady-state plans cut
  over; transient warmup plans (never recur → never compile) and any plan with
  a chunked op (>128 MB buffers — chunked contiguous/adam) stay on the RECORDED
  path. So the recorder is still the build source for those.
- The FIRST execution of every template ALWAYS runs the LOWERED path (for real
  results) + recording; the cutover only swaps the REPLAY source. So the per-
  position arena, `createParamsBuffer`, and the record* hooks are all exercised
  by that first lowered execution regardless of the cutover.
Therefore the recorder + the arena/params machinery CANNOT be deleted by the
cutover alone. The deletions decompose into gated sub-phases:

- **4.1 Flip the cutover to default-on** — ✅ **DONE 2026-06-14**. The cutover is
  the default; `wantCutover = ENV.TORCHLETTE_GENERATED_PLAN !== "0"` (opt-out
  form, matching `ARENA_LIVENESS`/`MEMORY_PLANNER`). The generated plan is the
  replay source for every fully-covered plan; `=0` opts back into the recorded
  replay everywhere. No deletions — the recorder still runs (the first execution
  is always lowered + records; this only swaps the 2nd+-execution replay source)
  and is the gate/fallback. Validation ladder, default-on, all green:
    - Gates (`test:gates`, cutover now the default "compiled" path): 3/3.
    - Fullstack parity generated-default vs `=0` recorded: max |Δ| = 5.7e-6 over
      30 steps (fp32 noise floor — byte-identical modulo slot bijection, as the
      gate proves).
    - Production regression (real WebGPUGPT2Trainer, 10×20): baseline-EXACT
      (9.81/5.92/5.15/4.64), flat 3081 MB, zero leak.
    - Full suite: 140 file-runs green (cpu 85 + webgpu 55, 710 webgpu tests).
  This was the go/no-go; it unblocks everything below.
- **4.2 Cover the chunked ops** — ✅ **DONE 2026-06-14**. Empirically, the lone
  chunked op the production 124M plan hits is the **full-reduction sum** over a
  >maxStorageBufferBindingSize input (the clip/scaler reductions over the 154 MB
  embedding grad). A census at production batch×seq=1024 (where CE logits are
  206 MB) confirmed it: all 4 recurring plans now FULLY GENERATE, **zero**
  uncovered — CE's custom kernel doesn't introduce a generic >128 MB binding,
  and chunked contiguous/adam/`where` simply never appear at vocab×embed sizes.
  - **Root cause it was uncovered:** `sumFullReductionChunked` allocated its
    `partials`/`out` temps via raw `createTrackedBuffer`, which bypasses
    `recordAlloc` → they became record-time **persistentSlots** (live GPU
    pointers). The generator builds from IR with no live buffers, so it
    fundamentally cannot reproduce a persistentSlot — that, not the dispatch
    shape, was the blocker. (It was also fragile in the recorded path.)
  - **Fix:** extracted `planChunkedFullReduction(elements, bytesPerElement, ctx)`
    — pure geometry (chunk subranges, per-chunk `[chunkSize, idx]` params,
    partials bytes, final params, shared pipelines) — as the SINGLE SOURCE for
    both `sumFullReductionChunked` (execution) and `generateChunkedFullReduction`
    (generator). The temps now go through `allocateOutputBuffer` (arena, kind-1
    ALLOC), so they record as real planner-managed slots the generator emits.
    Chunk dispatches carry `bindingRanges: [{offset,size}, null, null]`.
  - **Validated:** chunked sum generated == recorded (0 diverged, params multiset
    matches) + Δ=0 vs CPU across replays; 124M generated-cutover vs `=0` recorded
    identical loss + memory; gates 4/4 (new in-suite gate: `t-chunked-sum-probe`
    in `compiled-plan-parity.spec.ts` — small-model gates never allocate >128 MB);
    full suite green; production regression baseline-exact, flat 3081 MB.
  - After 4.2 the recorded BUILD is the source for NO recurring plan. (Transient
    warmup plans still never compile — they don't recur; that's fine, they run
    lowered once. Any future config that surfaces a genuinely new chunked op
    shows up as a loud `uncovered` census entry → that plan safely stays recorded
    until covered.)
- **4.3 Demote the recorder to CI-only** — ⛔ **BLOCKED (attempted 2026-06-14,
  reverted).** Plan: skip recording in production, build the compiled plan from
  the generated stream directly. The attempt surfaced a real, previously-hidden
  generator bug that must be fixed FIRST (and is the same blocker as 4.4):
  - **Input-bearing plans never actually cut over.** The cutover's genResults
    harvest iterates every plan node WITH a result and requires each to have a
    GENERATED slot (`gen.nodeSlot`). The per-step `tensorFromArray` inputs of the
    forward/loss plans (node[1]/node[29] — the tokens/targets) are EXTERNALS: they
    have a `node.result` but are not *produced* by the plan, so they have no
    generated slot → `genOk=false` → the cutover bails. Pre-4.3 that was
    invisible and harmless: a RECORDED plan always existed (recording was
    unconditional), so these plans silently rode the *recorded* replay. So
    "every recurring plan cuts over" (4.2) was true only for the produced-only
    (backward/optimizer) plans; the input-bearing forward/loss plans always used
    the recording.
  - **DIVERGENCE ROOT-CAUSED + FIXED (2026-06-14).** The "buggy generated replay"
    was NOT a frozen uniform — it was the harvest DROPPING multi-output result
    slots the generator never exposed. Bisection (force-cutover one plan at a
    time): the FORWARD plan cuts over bit-exact (4.8e-6 — its ~198 dropped
    primary-output activations are recomputed under checkpointing / not consumed
    cross-plan, so skipping is safe). The BACKWARD plan diverged (1.3e-2): it
    dropped `fusedLayerNormBackwardGradWeightBias` **output[1]** (grad_bias) of
    every layer — the optimizer consumed it, so a frozen/stale grad_bias drifted
    the bias params. `generateLayerNormGradWB` allocated the grad_bias buffer
    in-stream but returned only the primary (grad_weight) slot; the walker had no
    channel for a sequential op's non-primary outputs. **Fix (general):
    `SequentialGen.extraOutputs` — a multi-output sequential op returns its
    non-primary slots, the walker maps them into `nodeSlotExtra`** (same
    convention as the adam-batch / attention multi-output paths). With grad_bias
    exposed, backward-cutover + all-input-cutover match the recorded trajectory
    to fp noise (5.7e-6), checkpoint ON and OFF. **This generator fix is landed.**
  - **Harvest hardening (designed, not yet landed):** distinguish a PRIMARY
    (oi 0) miss — external input/leaf or recomputed intermediate, safe to SKIP —
    from a NON-PRIMARY (oi>0) miss — a real generator gap that must BAIL the
    cutover to recording, never silently replay stale. This makes the cutover
    safe-by-construction: it caught a second instance (attention `logsumexp`
    oi=1 unexposed in the checkpoint-recompute path) and fell back to recording
    instead of diverging.
  - **REMAINING 4.3 BLOCKER — memory, not correctness.** Once input-bearing plans
    cut over (the goal), the production trainer's steady-state peak is **+57 MB**
    (3135 vs 3078 MB). **Investigated 2026-06-14 — it is NOT architectural.**
    Instrumenting the planner registry + pool at steady state (both modes):
      - **Planner registry byte-IDENTICAL**: 477 entries, 2269.5 MB, 1187.7 MB
        resultHolders — the same in recorded and clean no-record (single-build).
        So generated plans pack the bulk (activations + results) IDENTICALLY;
        they are NOT inherently bloated.
      - **ROOT-CAUSED to the per-position buffer ARENA** (alloc-stack histogram,
        2026-06-14). The +57 MB is entirely in `resolveOutputBuffer` →
        `buffer-arena.ts:486` (the per-(resolveIndex)-position arena): `=0` retains
        ~671 such buffers / **209 MB**, no-record ~690 / **269 MB** (+59 MB). The
        planner-materialized compiled-replay buffers (`compiled-plan.ts:1134`,
        ~2333 MB) are byte-identical between modes.
      - These arena buffers are **WARMUP-ONLY DEAD WEIGHT.** Compiled replay with
        the memory planner (default) materializes registry-entry buffers directly
        via `device.createBuffer` (1134) and never calls `resolveOutputBuffer`,
        so the per-position arena is touched ONLY during the lowered warmup
        executions (exec 1 + 2 per template) — then retained, unused, forever
        (the regression's zero round-2→9 growth confirms it doesn't grow). It is
        dead weight in BOTH modes (~209 MB even at `=0`).
      - The +57 MB DELTA: the no-record warmup runs with backend op caches LIVE
        (recording's cache-bypass is off), so its `resolveOutputBuffer` call
        order/sizing populates the per-position arena to larger max sizes than the
        recorded warmup. A secondary effect, not a generated-plan property.
      - The fullstack parity model shows no registry diff; the earlier
        "+57 MB even no-double-build" + the 289-vs-364 node count were red herrings
        (different `STEPS` / always-record double-build). Clean single-build
        no-record registry == `=0`.
  - **Net / is-it-a-bug:** YES, but a **pre-existing one, not a generated-plan
    bug.** The divergence (correctness) is FIXED and general (grad_bias exposure +
    bail rule); generated plans pack IDENTICALLY (planner registry byte-for-byte).
    The +57 MB was the per-position arena failing to reclaim its warmup buffers
    after a plan cuts over to compiled/planner replay — a latent inefficiency
    already present at `=0`.
  - **FIXED 2026-06-14 (commit 78c6f73, arena-reclaim).** On the compiled path,
    once a plan has a valid planner-backed `compiledPlan` (which binds registry
    buffers and never calls `resolveOutputBuffer`), its dead warmup arena buffers
    are reclaimed via `destroyArena` — `canRecycle`-gated (a buffer still live as
    an external input to a not-yet-compiled consumer is orphaned to the
    pool-release chain, never destroyed-while-live; freeable ones are fence-gated
    `deferredDestroy`'d), guarded on `plannerEntries` + non-empty arrays (no-op
    once reclaimed; arrays re-grow on a lowered fallback). **A DEFAULT-PATH win,
    independent of stage-4:** production regression peak **3078 → 2754 MB
    (−323 MB)**, 124M mem-probe 3.20 → 3.12 GB, loss baseline-exact, zero growth;
    gates 4/4, full suite green, fullstack parity 7e-6.
  - **No-record path validated, then the residual isolated (2026-06-14).** Built
    the full no-record path (`TORCHLETTE_NO_RECORD=1`) + a `producedNodes`-gated
    harvest refinement (distinguishes a genuine produced multi-output gap → bail,
    from a CSE-bypassed alias whose result is another slotted node's buffer →
    safe skip). With it, the no-record run is bail-free (0 recordings, loss
    baseline-exact) — the "logsumexp" bail was node[60], a CSE-bypassed alias
    attention, correctly skipped. BUT measuring isolated the true residual:
    **cutting the FORWARD plan over to generated replay costs +36 MB** (forward
    on recorded 2752/2754 MB ≈ `=0`; forward cut over 2788 MB — stable). The
    generated forward plan's steady-state replay is ~36 MB heavier than its
    recorded replay. Backward/optimizer plans cut over memory-NEUTRALLY.
  - **+36 MB ISOLATED (2026-06-14, size-class histogram + alloc-stack diff).** It
    is NOT the planner working set: per-plan registry totals are byte-identical
    (forward 866.9, backward ~1353, optimizer 98.7, loss 50.7 MB) and `cur` is
    flat round-0→9 in both. The delta is **+1 buffer (~33.5 MB) in the >16 MB
    bucket** allocated at `buffer-arena.ts:572 ← dispatch.ts:747` — the
    **arena-liveness SPILL path** (big activation buffers exceeding the arena
    threshold spill to the POOL during the lowered WARMUP). At steady state the
    compiled replay binds planner buffers (`compiled-plan.ts:1134`); the warmup's
    spilled pool buffers are never reused, just held **idle in the pool**. The
    generated/cutover warmup runs WITHOUT recording (op caches LIVE → cache-hit
    sizing) and spills ~33.5 MB MORE than the recorded warmup (cache-bypass). So
    it's the **spill-pool sibling of the arena residue** the arena-reclaim already
    fixed — idle, reusable, within the pool budget, NOT a leak and NOT a
    working-set increase.
  - **DECISION:** the memory-optimal config **keeps the forward plan on recorded
    replay** (the bail rule routes it there at no cost) and cuts over everything
    else. Forcing the forward cutover surfaces +33.5 MB of idle warmup-spill pool
    residue — not worth it. So the no-record path + `producedNodes` were REVERTED;
    the committed state is arena-reclaim (−323 MB) + cutover for the
    memory-neutral plans. Recorder demoted to the forward/loss plan + gate.
  - **Spill-pool eviction ATTEMPTED → NOT VIABLE (2026-06-14, reverted).** Tried a
    demand-driven pool trim: at beginStep (post-fence), evict idle pool buffers in
    size classes demand-dead for N consecutive steps (hysteresis to avoid churn),
    reasoning the warmup-spill classes go dead once plans cut over to planner
    replay (which binds `device.createBuffer`, never pool-`acquire`s). It CHURNS:
    the trim fired and freed up to ~2 GB/step, but steady-state `cur` was
    byte-IDENTICAL with the trim on vs off (2721.1 MB) — i.e. it destroyed buffers
    that were immediately re-allocated. Root cause: "0 reservation demand" ≠
    "dead". The reservation window doesn't capture all pool acquires (markStep
    cleanup / autograd intermediates acquire OUTSIDE the window), so working-set
    classes look demand-dead, get trimmed, then re-acquired. Unlike the
    per-position arena (per-plan-owned, provably dead, `canRecycle`-gated), the
    spill residue is fungibly entangled with the shared pool's reusable working
    set — there is no safe demand signal to separate them. The +33.5 MB stays
    (idle, reusable, within the pool budget — not a leak). Do NOT re-attempt the
    demand-trim. A real fix would need per-plan spill-buffer ownership tracking
    (like the arena), which the spill path (resolveOutputBuffer → shared pool)
    doesn't have.
- **4.4 Build-without-execution → serializable plans** (LARGE, the headline
  dividend). Build the CompiledPlan from the lowered IR at COMPILE time, with NO
  first lowered execution. Requires the generator to derive ALL metadata from
  the IR (today the harvest reads `node.result` for result shapes/strides — must
  come from `node.shape`/`dtype` + the captured layouts, the same capture-don't-
  synthesize discipline phase 3 already established). This kills the ~700 ms cold
  start AND makes plans serializable (no live GPU pointers) — the actual payoff.
  ALSO subsumes the 4.3 +36 MB residual: that residue is warmup-spill pool
  buffers from the LOWERED first execution; build-without-execution removes that
  execution → the spill never happens. (A standalone spill-pool reclaim churns —
  fungible pool, no clean dead-signal — so it's not worth doing separately.)
  - **INCREMENT 1 DONE (2026-06-14): IR-derive differential** (executor.ts cutover
    harvest, `[ir-derive]` under TORCHLETTE_DEBUG_COMPILED — diagnostic, removes
    nothing). Counts where IR-derived result metadata (node.shape/dtype +
    contiguous strides + offset 0) would DIFFER from the live node.result. On the
    production trainer:
      - forward(388): shape 0, dtype 0, strides 82, offset 48, multiOutExtra 8
      - backward(703): shape 0, dtype 0, strides 0, offset 0, multiOutExtra 200
    → **primary-output shape + dtype are 100% IR-derivable** (the majority).
    Strides/offset diffs are all VIEW results (narrow/transpose/permute) — NOT
    contiguous, but the generator ALREADY captures them (cachedStridedInputs /
    cachedViewInput); the harvest just needs to consult those. multiOutExtra
    (8 + 200) is the genuine remaining capture gap: extra-output shapes (attention
    dQ/dK/dV, layernorm grad_weight/bias, Adam m/v) aren't in node.shape.
  - **INC 1b DONE: per-op breakdown** ([ir-derive] now prints which ops cause each
    diff). Exact spec for the derivation (`deriveResultMeta(node, oi)`):
      - **dtype** = node.dtype (0 diffs — done).
      - **primary shape** = node.shape (0 diffs — done).
      - **multiOutExtra shapes** — adamStep oi=1,2 (m,v) = node.shape (the param
        shape; 200 of the diffs, TRIVIAL); fusedAttentionForward oi=1 (logsumexp)
        = from payload (batch,heads,seq; 8 diffs).
      - **view strides+offset** — narrow / reshape / permute (measured: strides
        narrow26/reshape24/permute32, offset narrow16/reshape16/permute16). Derive
        by applying each view op's metadata transform to the INPUT's meta:
        narrow → offset += start*inStrides[dim], shape[dim]=length, strides
        unchanged; permute/transpose → permute/swap inStrides; expand → stride 0
        on broadcast dims; reshape → contiguous(newShape) (materialize-vs-view
        branch already captured via cachedViewInput.contiguous). The differential
        is the oracle: implement, iterate to 0 diffs.
  - **INC 2 DONE (2026-06-14): result metadata fully IR-derivable, single-source.**
    NOT by re-deriving view-stride math in the executor (that would be a SECOND
    source of truth that drifts — the hack we avoided). Instead:
      - Extracted the view-metadata transforms into `backend/webgpu/ops/view-meta.ts`
        (narrow/permute/transpose/expand + reshape) as the SINGLE SOURCE. The
        backend view ops (narrow/permute/expand; transpose = permute) now compute
        their output {shape,strides,offset} via it — one implementation, two
        callers. reshape kept its buffer-branching but its meta is pinned by the
        differential (and reshapeMeta carries a `materialized` flag).
      - `deriveResultMeta(node, oi)` (executor.ts): views → view-meta transform on
        the input meta; compute ops → node.shape + contiguous + offset 0;
        multi-output extras → adamStep m/v = node.shape, attention logsumexp =
        [B,H,N] from payload.
      - The [ir-derive] differential now compares deriveResultMeta vs the live
        result for EVERY result: **0 diffs across the production trainer
        (388/703-node plans) AND fullstack (attention multi-output + all views).**
        So the harvest can drop the live read entirely — requirement #2 proven.
      - In-suite guard: test/view-meta.spec.ts (transforms vs hand-computed).
    Validated: gates 4/4, regression baseline-exact + mem flat, full suite green
    (cpu 1100 + webgpu 711). The harvest still READS live bt (no behavior change);
    flipping it to deriveResultMeta happens in inc 4 (when there's no live result).
  - **INC 3 RE-SCOPED (2026-06-14).** "Move the layout captures to lowering
    (inputs live there too)" is WRONG: most capture inputs are INTERMEDIATE nodes
    not materialized at lowering (the captures run during execution precisely
    because that's the only time intermediate inputs are live). And the capture
    fns need live tensors — e.g. planBareMatmul puts `effectiveA.buffer` into the
    plan. So the real remaining work is to feed IR-DERIVED input metadata
    (deriveResultMeta, recursed to inputs) into the capture/plan fns via metadata
    STUBS, and confirm those fns use only geometry (shape/strides/dtype), not live
    buffer contents (replay rebinds buffers via slots anyway). That MERGES inc 3
    into inc 4 — one integrated build-without-execution push, not a separate step.
    - **Bounded inc-3 win DONE:** the reshape capture's materialization flag was
      OBSERVED (`result.buffer !== input.buffer` — needs a live result); now
      DERIVED via the shared `reshapeMeta(...).materialized` (inc 2), removing its
      last live-result dependency. Verified: derived == observed on every reshape
      across fullstack (no `[reshape-mat]` divergence), stream gate 4/4 0-diverged,
      gates 4/4, regression baseline-exact + flat.
  - **INC 4b DONE (2026-06-14) — harvest seam-flip.** The cutover harvest now
    takes its NodeResult metadata (shape/strides/dtype/offset) from the IR
    derivation (`deriveResultMeta`) as the SOURCE OF TRUTH; the live result is
    demoted to a post-exec cross-check that ASSERTS agreement and falls back
    (loudly — `[ir-derive] DIVERGE`) if an op isn't derivable yet, so a gap can
    never silently emit wrong metadata. This is the "single source at the seam,
    assert agreement" rule applied to the harvest, and it's exactly the live read
    the no-exec flip (inc 4c) removes. Verified: 0 DIVERGE warnings across the
    full regression run + fullstack; gates 4/4; regression baseline-exact
    (`{0:9.81,3:5.92,6:5.15,9:4.64}`, peak 2754.6 flat); fullstack derived-cutover
    vs lowered max |Δloss| = 5.7e-6 / 30 steps.
  - **INC 4a + 4c DONE (2026-06-14) — build-without-execution (TORCHLETTE_BUILD_FROM_IR=1).**
    On the first call the compiled plan is now built from the lowered IR with NO
    lowered execution and NO recording: a new early path in `executeLoweredPlan`
    (gated by the flag, default off) populates the layout captures from IR-derived
    metadata STUBS (`populateCapturesFromIR` → the extracted single-source
    `captureActionLayouts`, fed `makeMetaDeriver` stubs instead of live tensors),
    runs `generateStream`, harvests the result metadata from the IR
    (`harvestGenResults`, the SAME function the cutover uses), builds the
    planner-backed plan, and replays it via `executeCompiledPlan`.
    - **Metadata deriver (`makeMetaDeriver`):** recursive — a node's output
      {shape,strides,offset,dtype,bufferSize} from its op + its inputs' derived
      metadata, bottoming at leaf/external inputs whose live storage IS available
      at build (params, prior-plan results). Views use the shared `view-meta`
      transforms; compute ops are contiguous. bufferSize feeds only the
      >maxBindingSize guard (the one non-byte-exact field; the stream differential
      is the seam check).
    - **Matmul stubs:** `planBareMatmul`/`planTiledMatmul` never touch the buffer
      at plan time (out=undefined; only m/n/k/dtype/strides matter), and
      `ensureContiguous` no-ops when `isContiguous` (which ignores offset, matching
      tensor.ts). A guard skips planBareMatmul for the never-in-practice
      non-contiguous-non-transpose case (would dispatch a copy on a fake buffer).
    - **Harvest SET = action-output nodes** (`actionOutputHarvestPairs`): the
      structural superset of the cutover's live-result survivors (= all plan nodes
      minus fused/epilogue/row-program absorbed-internal nodes). The exact survivor
      set is partly RUNTIME-RC-DEPENDENT (a node is kept past the plan when an
      external/cross-plan refcount holds it — e.g. backward holding forward
      activations — which `canSafelyRelease` checks at release time and which can't
      be modelled structurally), so we harvest the conservative superset: if the
      generator slotted every action output → complete plan; if any lacks a slot →
      `genOk` false → fall through to the lowered path, exactly as the cutover
      declines. Verified the superset invariant (every live survivor ∈ action-output
      set) holds for all build-able plans.
    - **Result:** MIXED mode — the fully-coverable plans (backward / optimizer)
      build from IR with zero lowered execution; the forward plan and others fall
      through to lowered+record (the forward plan's cross-plan survivors aren't all
      slotted, so it doesn't cut over today either). Loss baseline-exact
      (`{0:9.81,3:5.92,6:5.15,9:4.64}`), peak flat at 2899.6 MB (+145 vs the 2754.6
      default — the conservative-superset harvest marks more slots as exclusive
      result entries, and build-from-IR skips the warmup-arena reclaim path).
      Fullstack build-from-IR vs default max |Δloss| = 6.7e-6 / 30 steps. Default
      path (flag off) byte-identical: gates 4/4, regression 2754.6 exact, full suite
      green. `endCounters` captured AFTER the first replay (no prior dispatches).
    - **Refactors (all single-source, behavior-preserving):** extracted
      `captureActionLayouts` (shared by the lowered loop + IR population),
      `harvestGenResults` + `liveResultHarvestPairs` (cutover) /
      `actionOutputHarvestPairs` (build-from-IR), and `computeLivenessOutputIds`
      (shared by liveness release + reused logic).
  - **HARVEST SET REFINED (2026-06-14):** `actionOutputHarvestPairs` now
    enumerates the PRECISE action-output set (sequential/view/data-source →
    nodeIndex; fused → output + additional; epilogue/row-program → output;
    adam-batch/batched-reduction → all nodeIndices) instead of "all plan nodes
    minus fused-internal." The old form wrongly included nodes that are in
    planNodes but covered by NO action (CSE'd / bypassed / externally
    materialized) — they have no generated slot, so they forced genOk=false and
    blocked otherwise-buildable plans. With the precise set one more plan builds
    from IR, peak drops 2899→2893 MB, loss baseline-exact, fullstack vs default
    still 6.7e-6/30 steps, chunked-sum (>128MB) path |Δ|=0 vs CPU under the flag.
  - **DERIVER COMPLETED + COVERAGE = ALL PLANS (2026-06-14).** `derivedOutputShape`
    now handles every multi-output extra real plans hit: adamStep oi 1,2 (m,v),
    fusedAttentionBackward oi 1,2 (dK,dV — all [B,H,N,D] = node.shape),
    fusedLayerNormBackwardGradWeightBias oi 1 (grad_bias = [featureDim] =
    node.shape), fusedAttentionForward oi 1 (logsumexp). With the deriver complete
    AND the precise action-output set, EVERY recurring plan (forward, backward,
    optimizer) builds from IR — zero lowered execution. Loss baseline-exact,
    fullstack vs default 5.7e-6/30 steps, chunked-sum |Δ|=0.
  - **THE OVER-HARVEST IS FUNDAMENTAL (proven, not assumed).** build-from-IR must
    harvest the FULL action-output set, not just the results that outlive the plan.
    A survivor prune was tried TWICE — once via `computeLivenessOutputIds`, once via
    a COMPLETE walk of the entire live pending graph (the maximum information
    available at build time) — and BOTH deterministically crash on a shared node.
    Proof (per-node cross-plan-visibility trace, node 2675, a narrow shared by the
    DiLoCo trainer's 388- and 364-node sibling plans):
    ```
    [trace 2675] plan nodes=388 inThisPlan=true crossPlanOut=false reachableFromPendingRoots=true
    [trace 2675] plan nodes=364 inThisPlan=true crossPlanOut=false reachableFromPendingRoots=true
    → FATAL: Input not ready: node 2675
    ```
    Plans are forced INCREMENTALLY. When the 388 plan builds + replays, the 364
    plan's graph does not exist yet, so the complete walk reports node 2675 as
    NOT cross-plan-consumed and NOT tensor-held — no build-time analysis, however
    thorough, can know it must be harvested. The 364 plan is forced afterward,
    reads 2675, finds it gone. The lowered/cutover path escapes this ONLY because
    it materializes every result and harvests live `node.result` post-execution
    (then prunes by actual refcount at markStep) — build-from-IR has no
    after-the-fact signal. So the over-harvest is intrinsic: closing it needs
    whole-graph-ahead forcing (defeats incremental execution) or a different
    result-lifetime model. build-from-IR therefore trades MEMORY for cold-start
    elimination + serializability — a viable OPT-IN, not a free default
    replacement.
  - **OVER-HARVEST MEMORY SCALING (measured, batch 1 / seq 256, sivri 32GB).** The
    overhead is activation/intermediate-bound, so as a FRACTION of peak it SHRINKS
    with model size (params + optimizer state — harvested identically both ways —
    grow ~quadratically in width and come to dominate). Not constant-MB, not a
    constant %:
    | model | params | default | build-from-IR | Δ | Δ% |
    |-------|-------:|--------:|--------------:|--:|---:|
    | embed128 L8  | 8M   | 824.7 MB  | 1175.2 MB | +350 MB  | +42.5% |
    | embed768 L12 | 124M | 8536.9 MB | 10084.9 MB| +1548 MB | +18.1% |
    | embed1024 L24| 354M | 17971.3 MB| 19845.6 MB| +1874 MB | +10.4% |
    Δ% halves roughly per ~3× params, so gpt2-xl-scale (1.5B) would be single-digit
    %. The fraction also FALLS along the batch axis and plateaus — fixed 8M model,
    seq 256: b1 +42.5% (+350MB), b8 +33.8% (+932MB), b16 +33.9% (+1696MB), b64
    +21.4% (+4320MB). Fitting Δ ≈ 287 + 63·batch MB vs base ≈ 517 + 307·batch MB
    shows a batch-independent component (weight-grad / optimizer over-harvest,
    ∝params) plus a per-batch component (activation workspace); the ratio plateaus
    at ~63/307 ≈ 20% at high batch. So the % is WORST at small-model/small-batch
    (~40%) and improves on BOTH axes — BUT the ABSOLUTE overhead grows to GBs
    (+4.3 GB at b64 even for the 8M model), because it is transient-activation-
    workspace memory. (Sweep via REG_EMBED/REG_LAYERS/REG_HEADS/REG_BATCH/REG_SEQ/
    REG_ROUNDS on diloco-regression-check.ts; defaults preserve the baseline
    config. Numbers from the 32GB sivri box — the benign "out of memory" pool-
    budget log spam is not fatal; peaks are flat across rounds.)
  - **REMAINING for 4.4:** (1) a small plan with a per-step-varying `mul` scalar
    trips the volatile-params guard and falls back once (correct, no drift) —
    route per-step scalars through the volatile/scalar-table mechanism so it
    builds cleanly; (2) decide the default: given the +34% memory, keep
    build-from-IR opt-in (cold-start/serializable plans) rather than flipping the
    default + deleting the lowered first-exec (the 4.5 plan) — or first solve the
    over-harvest memory.
- **4.5 Retire the now-dead lowered-path machinery** (MEDIUM, gated on 4.4).
  Once 4.4 removes the lowered first execution, audit + delete what it alone
  used: the per-position arena hints/pre-pinning/conflict paths, the params-
  sequence cache. NOTE the doc's original "delete pinnedBufferSet" is WRONG —
  it's the memory PLANNER's buffer-ownership mechanism (consulted by every
  replay alloc), it STAYS. The legacy unbudgeted arena (`ARENA_LIVENESS=0`) is a
  separate product decision (drop the opt-out or keep it).

Dividends (realized progressively, fully at 4.4): serializable compiled plans
(generated plans have no live GPU pointers → ~700 ms cold start dies); single
answer to "who owns this buffer"; the architecture-debt rules enforced by
construction. Net: 4.1 DONE (cutover default-on) + 4.2 DONE (chunked
full-reduction sum covered). 4.3 FIXED the input-bearing-plan divergence
(grad_bias multi-output exposure — LANDED — + the harvest bail rule) so those
plans cut over correctly, and the memory investigation proved generated plans
pack IDENTICALLY to recorded (planner registry byte-identical, live buffer count
identical). The only residual is a peripheral +57 MB pool/size-class artifact in
the production trainer (1.8%, NOT architectural) plus the logsumexp multi-output
gap (one-line, same as grad_bias). Recorder demotion (4.3) and
build-without-execution (4.4) are now de-risked — neither blocked by a
fundamental memory regression.

**Cutover WIP (2026-06-13, flag-gated `TORCHLETTE_GENERATED_PLAN=1`, DEFAULT
OFF — default path + gate fully green).** Wired: `finalizeCompiledPlan` (the
shared planner+assembly tail, extracted from `buildCompiledPlan`),
`buildCompiledPlanFromGenerated` (feeds the generated commands/slots + gen-slot
nodeResults through `finalize`), `generateStream` now exports `nodeSlot`/
`nodeSlotExtra`, and the executor builds the generated plan + swaps it in when
`gen.fullyCovered` and every result node has a gen slot. Three bugs found+fixed
en route:
1. **Spurious external slots (FIXED).** The generator's external pre-assignment
   mirrored the executor's, but the executor runs PRE-execution (in-plan nodes
   have no result → only true externals get external slots) while the generator
   runs POST-execution (every node has a result). So in-plan data-source buffers
   got spurious external slots whose Phase-1 replay resolution failed once the
   node result cleared (step 2). Fix: skip pending refs whose producer is
   in-plan (`nodeIndexById.has`). This is a real generator-correctness fix and
   lands independent of the cutover (gate stays green).
2. **Mixed-coverage harvest bail.** A "fully covered" plan can still have result
   nodes with no `nodeSlot` entry (not action outputs); the harvest sets
   genOk=false and that plan stays recorded — SAFE, but means the cutover is
   per-plan (forward cuts over; backward+optimizer stays recorded for now).

3. **paramsData stored BY REFERENCE — the freeze, FIXED.** With the flag on the
   forward cut over, step-0/1 losses matched the recorded baseline exactly, then
   the loss FROZE from step 2. After an extensive isolation that FALSIFIED every
   structural hypothesis (commands, flat-tag order, slot count+KIND per result
   node, result coverage 259/259, external-ref SET 101/101, weight-buffer
   resolution byte-identical, the memory planner — froze under
   `ARENA_LIVENESS=0` too —, checkpointing, harvest-skips), the decisive test
   was the **params-DATA multiset**: the generated plan had ~80 dispatches with
   a length-1 params slot = **[2048] (=BATCH×SEQ)** where the recorded plan had
   the real element counts ([262144]×76, [6438912]×1, ...). e.g. `node32` cast
   [50304,128] (6.4M elems) bound `params=[2048]` → its kernel processed 2048 of
   N elements → garbage forward → grads explode (1.8e10) → optimizer corrupts
   weights → frozen loss.
   ROOT CAUSE: the generator stored `plan.paramsData` **by reference** in the
   params SlotSource, but those uniform arrays are SHARED/reused across plan
   calls and later MUTATED — so every params slot ended up reading the
   last-written value. The recorded path was immune because `createParamsBuffer`
   stores `data.slice()` (a copy). FIX: `.slice()` at all 10 generator
   params-slot storage sites (generateSequential, planContigCopy, the layernorm/
   matmul/fused generators). After the fix the cutover trajectory matches the
   recorded baseline to fp noise over 20 steps (8.836388 vs 8.836389 @ step 19).
   GATE HOLE CLOSED: `diffSegmentsAligned` compared DISPATCH by pipeline +
   workgroups + binding SLOTS but never the params/uniform DATA bytes — a third
   hole (after orphan slots and flat-order). Added a params-data multiset guard
   to the FULLY-GENERATED gate (executor.ts) so this class is caught in future.

4. **adamStep multi-output + frozen-step_size (the #2 limit, FIXED → backward+
   optimizer now cuts over).** The backward+optimizer plan stayed recorded
   because its `adamStep` result nodes' m_new (oi 1) / v_new (oi 2) outputs had
   no `nodeSlot` entry (generateAdamBatch mapped only the primary param) →
   genOk=false. Fixed: emit all three outputs (param→bufSlots[1], m→bufSlots[2],
   v→bufSlots[3], all updated in place); the walker maps the primary to
   nodeSlot and the m/v extras to nodeSlotExtra. That made it cut over but the
   trajectory then DRIFTED gradually (onset at the step the optimizer plan first
   replays) — the generated `adamStep` TAG_UNIFORM used a no-op pack
   (`() => new Uint32Array(0)`), freezing the bias-corrected step_size at
   recording time (the frozen-step_size class AGAIN, now in the generator). Fix:
   thread a real volatile packer — `TileKernelInstance.volatilePack(repack)`
   builds the exact `node => packUniforms(spec, repack(node)).data` the recorded
   path uses; `planAdamStepDispatch` returns it (re-deriving the Adam config from
   the node payload, keeping the static num_elements/grid fields). After the fix
   the cutover (BOTH forward and backward+optimizer generated) matches the
   recorded baseline to ~1e-5 over **30 steps** (7.902291 vs 7.902299 @ step 29).

5. **External pre-assignment over-skipped LEAF inputs (the untracked-input /
   untracked-producer misses, FIXED → ALL plans cut over).** The earlier fix #1
   skipped external slots for any pending ref to an IN-PLAN node. But planNodes
   also holds LEAF inputs (the i32 input tokens, prior-plan results threaded in)
   that are in-plan yet have NO producing action — those are true externals and
   need an external slot. Skipping them left them (and, cascading, every view /
   cast / matmul-epilogue that consumed them) untracked, so two plans stayed
   recorded (matmul-epilogue/scatterAdd/cast `untracked-input`). Fix: build the
   set of nodes PRODUCED within the plan (every index any action emits/covers)
   and skip the external slot only for those — matching the recording's
   "storage exists pre-execution" exactly (a leaf input's storage DID exist
   pre-execution → external; a computed node's didn't → internal). All four
   canonical plans now FULLY GENERATED; the full cutover (forward + backward +
   optimizer) matches the recorded baseline to ~1e-6 over 30 steps (7.902296
   vs 7.902297 @ step 29).

**Cutover STATUS: the steady-state plans cut over correctly (30-step parity to
fp noise); coverage is 4/4 FULLY GENERATED.** Nuance: of the 4 canonical
templates, the 2 STEADY-STATE plans (forward 357 cmds, backward+optimizer 1746
cmds) recur every step → compile on the 2nd execution → cut over and execute via
the generated path. The other 2 are TRANSIENT warmup/first-step templates: the
generator FULLY covers them (the gate verifies their streams) but they never
recur, so they never compile and stay on the lowered path — expected, transient
plans gain nothing from compilation. The 30-step trajectory matches recorded to
~1e-6 with the steady-state plans generated and the transient ones lowered.

**Gate-hardening (params-data hole CLOSED; the other two are not cleanly
closeable, by design).** The params-data multiset guard (executor fully-covered
branch) is in — it caught the actual freeze and is value-level (affects
execution). The external-RESOLUTION and ORPHAN-slot "holes" can NOT be closed
with structural-identity checks: the segment-aligned differential is
deliberately slot-renaming- AND benign-structural-difference-tolerant, so
guards requiring the gen and recorded slot structures to be IDENTICAL
false-positive (the transient plans carry harmless orphan external slots — a
leaf-input buffer that a covered consumer resolves via another slot — and pick
different external-ref representatives under dedup; both are benign and never
executed). Correctness is instead pinned by: the segment diff + the params-data
guard + the 30-step trajectory parity, and — for the spurious-external CRASH
class specifically — the produced-set fix STRUCTURALLY prevents it (external
slots are assigned only to non-produced refs, never to a produced-within node
whose result clears mid-replay). Closing the remaining holes "properly" would
need resolved-BUFFER comparison (not ref-pair identity), which the trajectory
parity already subsumes — not worth the false-positive surface.

**Parity ladder with the flag ON (b) — VALIDATED.** (1) fullstack 30-step:
generated == recorded to ~1e-6. (2) PRODUCTION regression
(`diloco-regression-check.ts`, the real WebGPUGPT2Trainer, 10 rounds × 20 inner
steps) with `TORCHLETTE_GENERATED_PLAN=1`: loss trajectory matches the baseline
EXACTLY (9.81 / 5.92 / 5.15 / 4.64 — the baseline was recorded flag-OFF, so this
IS the A/B), and peak memory is FLAT at 3081 MB, 0 MB growth over rounds (no
leak). Because the generated stream is byte-identical to the recording (the
gate proves it), the cutover is memory/speed-NEUTRAL by construction — the
production regression's flat-memory check confirms it on the real path. SCALE
SAFETY: larger models keep working because the cutover is PER-PLAN and gated on
full coverage — a plan with a chunked op (chunked contiguous/adam, >128 MB
buffers) isn't fully covered, so it never cuts over and stays on the recorded
path. No correctness risk at scale; just less of the plan set cuts over.
(`profile-training.ts` at gpt2@512 hangs at init independent of the flag — a
profiler/scale issue, not a cutover regression; not pursued.)

Remaining before flipping the default: the phase-4 deletions (recorder →
CI cross-check only; per-position arena / pinnedBufferSet / params-sequence
cache subsumed by the planner). Lesson worth keeping: the generator must treat
every recorded structure it reproduces as needing the SAME copy/ownership,
volatile-repack, AND produced-vs-external discipline the recorder uses — a
shared mutable array stored by reference, a no-op volatile pack, and a leaf
input mistaken for an internal producer are all the same class of bug (the
recorder's `data.slice()`, volatile uniform, and pre-execution external
pre-assignment, respectively).

## Phase-4 RECONCILIATION (2026-07-07, task #43 stage-A pass)

Re-audited the tree at `006e4245` against this doc's Phase-4 narrative. Two
findings; **no code landed** (the stage-A objectives were already satisfied,
and the one remaining lever is blocked by a committed engineering result, not a
bug — so per the campaign's "STOP rather than improvise" rule it was not
forced).

**A1 (expose attention `logsumexp` oi=1 + the harvest bail rule) is ALREADY
LANDED — verified, not re-done.**
- `logsumexp` oi=1 is exposed by the generator: `generateAttention` returns
  `outputs:[{oi:0,outSlot},{oi:1,lseSlot}]` (`stream-generate.ts:1981`), and the
  walker maps forward-attention non-primary outputs into `nodeSlotExtra`
  (`stream-generate.ts:~360`). The `SequentialGen.extraOutputs` channel
  (`:411`, used for layernorm-bwd grad_bias at `:2180`) is the same mechanism
  for the sequential-op case. So the checkpoint-recompute attention plan carries
  its oi=1 result and does not bail on that account.
- The bail rule is landed, factored across two functions rather than as the
  doc's single "primary-miss=skip / non-primary-miss=bail" predicate:
  the *pair selection* encodes the skip (`liveResultHarvestPairs` for the
  cutover harvests only nodes that HAVE a live result; a non-surviving/leaf/
  recomputed primary is simply never in the set), and `harvestGenResults`
  hard-bails (`genOk=false`) on ANY harvested-pair whose slot is missing — which
  is strictly safer than the designed rule (it also bails a *surviving* primary
  miss). Net effect == the design: a real generator gap (a non-primary the
  optimizer consumes) forces the plan back to the recorded/lowered path; it
  never silently replays stale.

**A2 (no-record production default) = the already-implemented, already-opt-in
`TORCHLETTE_BUILD_FROM_IR` path — NOT flipped, for two independent reasons:**
1. The committed over-harvest scaling result (`b9c99153`/`2008b713`/`22ca85d3`/
   `e01d9adf`) proves the over-harvest is FUNDAMENTAL (build-from-IR must harvest
   the full action-output set; a survivor-prune provably crashes on cross-plan
   shared nodes under incremental forcing). It costs +34% at small model/batch,
   falling to single-digit % at scale — a real memory/cold-start tradeoff, so it
   stays a viable OPT-IN, never a free default. Flipping it would fail the
   memory-flat gate on non-checkpointed configs.
2. **NEW / the decisive reframe: `b66ead78` (checkpoint: run arena-free) made
   the entire compiled-plan path DORMANT in the production checkpointed
   trainer.** `WebGPUGPT2Trainer.initialize()` calls
   `setBufferArenaDisabled(true)` under checkpointing (the production default);
   the compiled plan requires the arena, so plans run the NORMAL lowered path —
   `hasCompiledPlan=false hasArena=false` on every plan (703/602/388/364-node),
   confirmed live on `diloco-regression-check.ts`. In this mode there is NO
   recorder, NO cutover, NO build-from-IR: liveness early-release alone gives
   flat bounded memory. The compiled/generated/recorder machinery is now
   exercised ONLY by non-checkpointed configs and the in-suite gates
   (`compiled-plan-parity.spec.ts`). So "demote the recorder in production" is
   partly already true — the recorder does not run in the checkpointed
   production path — and build-from-IR-as-default would change nothing there
   while regressing the non-checkpointed path. This is why A2 was not flipped.

**Stage-A gate results (default path, this pass, sivri V100 solo):** `npm run
build` ✓; `npm run test:gates` 4/4 ✓ (the compiled/generated path IS exercised
here, arena on); production regression baseline-EXACT
(`9.8089/5.9222/5.1532/4.6402` vs `9.81/5.92/5.15/4.64`), peak flat 2087.6 MB
(8M default config), zero leak — via the NORMAL lowered path. build-from-IR A/B
was memory-neutral only because it does not engage under checkpointing
(precondition `options.bufferArena` is false); it is not an A/B of the
over-harvest, which the committed scaling tables already characterize.

**Consequence for the 4.4/4.5 deletions:** the value of deleting the recorder /
per-position arena / params-sequence cache is now bounded to the NON-checkpointed
compiled path + the gates, because the checkpointed production trainer already
bypasses all of it. Any deletion campaign must first decide whether the
compiled-plan path is retained at all for non-checkpointed training, or whether
the arena-free-lowered path (which already serves production) should subsume it —
a strategy question above the mechanical deletion inventory.

**STRATEGY RESOLUTION (2026-07-07): the compiled path STAYS — unification, not
deletion.** Direction: the memory planner learns REMATERIALIZATION (checkpointing
expressed as liveness edges + recompute in the plan set), so checkpointed
training eventually runs compiled like everything else, and the arena-free
checkpoint mode of `b66ead78` dies by unification rather than by deletion. The
compiled path serves inference today (the plan-replay decode stack) and training
post-unification. The B1–B4 deletions remain gated exactly per the phase-4
inventory (B2/B3/B4 on the build-from-IR default flip; B1 on generated-only
build). Sequencing: stage 1 = solve the over-harvest via plan-set-level liveness
(design first), stage 2 = build-from-IR default + the ~800–1200-SLOC recorder
deletion harvest, stage 3 = rematerialization edges in the same lifetime
machinery.

**B-inventory status after stage-2 increment 3 (2026-07-08):**
- **B1 — PARTIAL (the separable core, inc-3a):** the cutover swap
  (recorded→generated) + `liveResultHarvestPairs` DELETED (−61 lines,
  executor.ts). The recorded BUILD itself (`buildCompiledPlan` + recording)
  SURVIVES — load-bearing for the uncovered-plan census fallback, the
  `BUILD_FROM_IR=0` opt-out, and the verify modes. The remaining bulk of the
  original −800–1200 estimate is coupled to sunsetting that opt-out/fallback:
  a product decision, reported not forced.
- **B2 — split named (inc-3a, no code):** all 80 record*-hook refs across 19
  files survive (they serve the surviving recorded build); the cutover swap
  was the only recording consumer with no other master.
- **B5 — DONE (inc-3c):** `TORCHLETTE_GENERATED_PLAN` deleted (twin of
  `BUILD_FROM_IR=0` after inc-3a).
- **B3 — DEFERRED (did not fall out naturally):** the params-sequence cache
  serves the surviving lowered/recorded executions; it dies with the recorded
  build, not before.
- **B4/B7 — out of scope (B7 sunsets at stage-3 remat), unchanged.**

### Flag sunsets (named, per the complexity-budget policy)
- **`TORCHLETTE_GENERATED_PLAN` — DEAD (stage-2 inc-3c, 2026-07-08, sunset
  EXECUTED).** After inc-3a deleted the cutover swap, the flag was an exact
  behavioral twin of `BUILD_FROM_IR=0` (both = recorded replay everywhere); a
  flag with an identical twin is debt. The recorded-replay reference arm is
  now `TORCHLETTE_BUILD_FROM_IR=0` everywhere (gates, parity ladders, tools).
- **`TORCHLETTE_BUILD_FROM_IR`** — **stage-2 increment 2 (2026-07-08): FLIPPED
  to opt-out default-on** (`!== "0"`, house convention). The opt-in form is
  dead; the surviving `=0` opt-out toggles back to the recorded build and dies
  WITH that build source when the recorded build sunsets (a product decision —
  see inc-3a's coupling note; the uncovered-plan fallback and the verify modes
  are its remaining masters). Recording now engages only for: verify modes
  (`TORCHLETTE_STREAM_GENERATE=1`; the determinism gate pins `=0`), plans the
  generator cannot fully cover (per-plan census-driven fallback), and the
  opt-outs (`BUILD_FROM_IR=0` / `COMPILED_PLAN=0`) — the single predicate is
  `buildFromIRActive()` (executor.ts).
- **`TORCHLETTE_ARENA_LIVENESS=0`** (legacy unbudgeted arena opt-out) — sunsets
  at the REMATERIALIZATION UNIFICATION (stage 3): once checkpointed training
  runs on the planner (liveness edges + recompute), the legacy arena's last
  role as a planner-bug fallback is subsumed by the planner being the only
  memory authority. This is the named product decision from the phase-4
  inventory, now adopted into the roadmap.

## Stage-1 LANDED (2026-07-08): observed cross-plan liveness kills the over-harvest

**Status: IMPLEMENTED.** `src/executor/observed-liveness.ts` (the cohesive
mechanism) + seam wiring: stamp at the replay-harvest chokepoint
(`compiled-plan.ts`), consumption at external-slot pre-population (compiled
phase 1), survival at the markStep boundary (`torchlette.ts`, after the step's
reclamation), converge (K=3 + no-new-template) → invalidate + rebuild pruned
(build-from-IR block, `executor.ts`), bind-time guard + in-place-committed
counter, telemetry folded onto `getPayloadThrashStats()`. Guard recovery lives
in `RuntimeEngine.forceAllMerged` (RecoverableGuardMiss → evict + fresh lowered
re-collection). ~335 net SLOC + a 322-line module; zero new env flags (lives
behind `TORCHLETTE_BUILD_FROM_IR`).

**Measured (6L/384/seq256, steady-state peak): conservative build-from-IR
2557 MB → pruned 1790 MB (−767 MB, −30%); recorded-cutover reference 1670 MB.**
The +53% over-harvest collapses to +7% — the delta is largely eliminated; the
residual is unconverged/intermittent templates + slightly-conservative pruning
(views of independently-live bases are kept). 4 templates converged, 174 harvest
pairs pruned, ZERO guard misses across every gate.

Gate ladder (all green): full suite (cpu 1161 / webgpu 881, 0 fail); test:gates
4/4; build-from-IR-pruned vs `TORCHLETTE_GENERATED_PLAN=0` fullstack losses agree
5.7e-6 over 30 steps (bit-identical trajectory across the pruning threshold);
124M-class regression baseline-exact {9.81,5.92,5.15,4.64} flat 2087.6 MB zero
leak; decode gen-tape + kv-differential PASS; STRICT_LIFETIME + STRICT_GPU over
the build-from-IR fullstack run: zero throws.

**Deviation from the proposed gate (b)(1).** The literal "observed needed-set ==
the default cutover's live-result harvest set" does NOT hold and is NOT the right
invariant: the observed mechanism prunes MORE aggressively than the recorded
cutover. `liveResultHarvestPairs` is measured PRE-markStep and keeps harvested
VIEWS of independently-live bases; the observed set is the post-markStep
rc-survivor set ∪ cross-plan-consumed. The observation is also coupled to which
results are harvested+stamped (self-reinforcing: a pruned result is unstamped, so
its survival isn't re-observed), so no clean cross-mode set equality exists. That
the tighter observed set is EXACTLY sufficient is proven the sound way — by the
bit-identical trajectory + zero missing-input crashes + zero dirty misses across
the whole ladder. The set-parity gate (`test/observed-liveness.spec.ts`)
therefore asserts the SEAM AGREEMENT that actually holds: identical trajectory
across the pruning-activation threshold, zero guard misses, and demonstrable
pruning (`prunedPairsRemoved > 0`).

## Stage-2 increment 1 (2026-07-08): residual analysis → the residual was
## one-shot templates; idle-retire closes it (and then some)

**The flip-gating question — "does the +7% residual shrink at scale?" — had a
surprising answer: NO, it GREW (+14.6%), and the attribution found a different
culprit than stage-1 guessed.** Measured (t-observed-liveness-mem.ts, V100
sivri solo, steady-state peak MB, post-step-12 reset):

| config | cutover ref | conservative | pruned (stage-1) | residual |
|--------|------------:|-------------:|-----------------:|---------:|
| small 6L/384/H6/b4/seq256 (stage-1's) | 1670.1 | 2557.1 (+53.1%) | 1790.6 | **+7.2%** |
| 124M-class 12L/768/H12/vocab50257/b1/seq256 | 5066.1 | 5926.6 (+17.0%) | 5807.0 | **+14.6%** |

At the 124M-class config pruning removed only 45 pairs (vs 174 at small) and
recovered just 14% of the over-harvest. A STEPS=40/RESET_AT=30 run is
byte-identical (5807.0) — **more hysteresis converges nothing**.

**Attribution (per-template registry footprint, `debugTemplatePlanMemory` +
needed-source counts on `debugAllNeededSets`):**
- **ONE-SHOT WARMUP TEMPLATES dominate — they are ~the entire 124M residual.**
  Two 296-node step-0/1 graph variants execute once, never recur. The recorded
  cutover NEVER compiles them (recording→replay needs recurrence), so they cost
  it nothing at steady state. Build-from-IR compiles every template on FIRST
  execution and harvests its full action-output set into exclusive planner
  result entries — and a template that never re-executes can NEVER converge
  (observation needs re-execution), so the conservative harvest is pinned
  forever: **+723.2 and +1446.3 MB of dead registry results at 124M-class**
  (+71.0 and +141.9 MB at small). "Unconverged" ≠ intermittent: they are
  structurally one-shot.
- **Converged recurring templates are near-parity or BETTER than cutover.** The
  optimizer template prunes to 2.1 MB vs the cutover's 121.6 (−119.5); the
  heavy fwd/bwd templates land byte-equal to the cutover (their kept set =
  needed ∪ mandatory covers the same results).
- **View-base conservatism ("b"-source pairs: harvested views whose handle died
  but whose base survived) is real but secondary, and small-scale-only**: the
  small backward template keeps +155.2 MB over cutover with 57/218 needed pairs
  base-only; at 124M the heavy template has b=3. NOT tightened: judging
  survival by the view handle would rely on the late-consumer guard for any
  actual reader, and `prunedExecuted` is STEP-SCOPED — a consumer in a LATER
  step reading a pruned view result would raw-crash ("Input not ready"),
  unguarded. Recorded as the open design question (a persistent
  pruned-pair→stamp map, or stage-3 remat, would make it sound).
- A second structural (unfixable-cheaply) source: consumer plan SHAPES in the
  build-from-IR arm are collected against a fuller materialization world, so
  they leaf-read more producer results → larger consumed-sets (c=76 on the
  small backward). Bounded, and covered by the same near-parity numbers.

**The tightening: IDLE-RETIRE (landed, this increment).** At the
observed-liveness step boundary, a template that has not executed for
`K_IDLE=3` consecutive boundaries has its compiled plan retired
(`destroyCompiledPlanBuffers` → registry entries relisted, buffers fence-gated
deferredDestroy), GATED on liveness: the plan records the storage ids its last
replay's harvest exposed (`CompiledPlan._lastHarvestIds`, assigned + skipped
pre-existing handles), and retire proceeds only when EVERY one is already
destroyed (`storageTracker.isDestroyed`) — no live reader can dangle. This is
strictly more conservative than the landed convergence invalidation (which
destroys with live survivors and leans on fence timing + re-harvest). A retired
template that re-executes rebuilds from IR (no lowered re-execution). Telemetry:
`retiredTemplates` on `getPayloadThrashStats()`. Also landed for the analysis:
needed-source attribution (c/s/b/g per pair), `debugTemplatePlanMemory()`,
`debugPlannerRegistryStats()`.

**Result: the residual is not just closed — build-from-IR-pruned+retire beats
the recorded cutover at BOTH scales** (retire also releases what the cutover
arm leaves as pool residue, and relisted entries serve later plans as temps):

| config | cutover ref | pruned+retire | vs ref |
|--------|------------:|--------------:|-------:|
| small  | 1670.1 | **1522.6** | **−8.8%** |
| 124M-class | 5066.1 | **2914.8** | **−42.5%** |

(retiredTemplates=2 at both scales; converged/prunedPairs unchanged; zero
guard misses.)

**Gates (this increment, all green):** observed-liveness spec 5/5 (set-parity +
late-consumer both arms); test:gates 4/4; fullstack parity BUILD_FROM_IR=1 vs
GENERATED_PLAN=0 maxΔ 6.7e-6 / 30 steps; STRICT_LIFETIME+STRICT_GPU
build-from-IR fullstack zero throws (losses within fp noise of the non-strict
arm, 7.6e-6); memory A/B above; full suite + production regression (below).

**FLIP RECOMMENDATION: FLIP.** The gating residual is eliminated with margin at
both scales; the trajectory is bit-identical; zero guard misses anywhere. The
known caveat carried into increment 2: the flip's own gate ladder (suite,
regression baseline-exact, decode, memory A/B at the flipped default) decides.

## Stage-2 increment 2 (2026-07-08): flip ATTEMPTED → STOPPED at the suite
## gate; three pre-existing build-from-IR correctness gaps surfaced

The flip was implemented (`buildFromIRActive()` opt-out predicate: BUILD_FROM_IR
!== "0" ∧ GENERATED_PLAN !== "0" ∧ COMPILED_PLAN !== "0" ∧ STREAM_GENERATE !==
"1"; determinism gate pinned to `=0`; spec/tool arms updated) and its ladder run.
Green: test:gates 4/4; observed-liveness spec 5/5 (flipped arms); fullstack
parity default-vs-GENERATED_PLAN=0 maxΔ 9.5e-6 / 30 steps; STRICT_LIFETIME +
STRICT_GPU zero throws; memory A/B reproduced increment 1's promise exactly
(small 1522.6 vs ref 1670.1; 124M-class 2914.8 vs ref 5066.1); production
regression baseline-exact flat. **RED: the full suite — 4 failures, so per the
stop rule the flip is UNCOMMITTED** (patch preserved; the tree keeps only the
first fix below). All failure classes REPRODUCE at the stage-1 commit
(3b0f21ea) under opt-in `TORCHLETTE_BUILD_FROM_IR=1` and pass at `=0` on the
flip tree — they are pre-existing build-from-IR gaps the flip EXPOSED, not
flip/inc-1 regressions (the earlier "stage-1 passes lifetime" read was a
vacuous vitest `-t` skip — `+` in the pattern is a regex quantifier):

1. **`test/optim/fused-vs-elementwise.spec.ts` — "SGD: late LR change is
   honored" (Δ 0.284) and "late LR change is honored" (Adam elementwise,
   Δ 0.089) vs 1e-5.** Root cause (SGD class, FIXED this increment): the
   inlined-scalar staleness gate (`inlinedFusionScalarsStale`) drops the stale
   compiled plan, but the build-from-IR block then rebuilds IMMEDIATELY from
   the UN-ADAPTED fused recipe — re-baking the stale scalar forever
   (invalidate → rebuild-stale → invalidate, every step; the lowered path's
   scalar-adapt at the fused-action loop never runs because build-from-IR
   skips lowered execution). Fix: `scalarAdaptPending` — a staleness
   invalidation forces ONE lowered execution (recipe adapts, scalar demoted to
   runtime input), the next execution rebuilds from IR adapted. SGD late-LR
   passes under `=1` with this. **The Adam-ELEMENTWISE late-LR still fails: a
   SECOND staleness channel** — its LR reaches a sequential-op uniform/scalar
   path not covered by the fused-recipe gate (the recorded path is guarded by
   `getConfigBuffer`'s byte-compare across executions; generated plans have no
   equivalent for this delivery). OPEN.
2. **`test/lifetime-natural-usage.spec.ts` — "mid-loop markStep+dispose+
   beginStep" (WebGPU): param0 reads param20's data** (actual 0.0021 vs
   expected 0.0001 — cross-replay data leak). Shape: a 40-zeros plan's results
   (the params) + a 20-upload/20-copy_ chunk template built from IR at chunk 1
   and REPLAYED at chunk 2 with rotated externals; some write lands in
   param0's buffer. Training loops never rotate externals this way, which is
   why every trajectory gate passes. Reproduces ONLY under the vitest worker
   (standalone identical-flow repros PASS — plan granularity differs: vitest
   merges 40-node plans, standalone forces per-param 3-node chains), so the
   trigger is the merged-plan shape. OPEN — deep planner/replay aliasing
   class, needs dedicated investigation.
3. **`test/distilgpt2-full-finetuning.spec.ts` — "zero steady-state growth":
   storages rate 156/step vs ≤1** (reachable=0, allocs=0, trackerMB=0 — a
   handle-churn rate gate, no byte leak). Likely the conservative per-replay
   harvest creating ~156 result handles/step (the recorded path's live-result
   harvest is far smaller) and/or staleness-rebuild churn; possibly shrinks
   with class-1 fully fixed. OPEN (may be a benign-but-gated proxy metric —
   needs a decision, not just a fix).

**Consequence: increments 2 and 3 are BLOCKED** until classes 1 (Adam channel)
and 2 are fixed (class 3 may fall out or need a measured verdict). The flip
patch is ready and its non-suite ladder is green — the remaining work is
correctness of the build-from-IR path itself, exactly the kind of pre-existing
silent-wrongness this campaign exists to surface. LESSON (test-shaped): the
stage-1 ladder never ran the FULL SUITE under `BUILD_FROM_IR=1` — the flip's
first casualty was discovering that gap; any future opt-in path claiming
flip-readiness must run the whole suite under the opt-in first.

**RESOLVED (2026-07-08, same day): all three classes fixed** — `fbccedfe`
(class 1b: the staleness gate was conditioned on a pre-existing compiled plan,
so under build-from-IR a recurring fused template's stale baked scalar was
never detected; it now fires unconditionally), `a318c5cc` (class 2: volatile
re-writes into replay-persistent buffers within one submit window —
queue.writeBuffer beats the still-encoded prior reader; flush-before-rewrite
guard), `e884fdeb` (class 3: the replay harvest chained each in-place view to
the PREVIOUS replay's view — a base chain growing one protected link per step;
it now flattens to the root owner). Full suite green under `=1` modulo the two
verify-mode gates the flip itself pins. The flip proceeds below.

### (ORIGINAL DESIGN, retained for the rationale)

### (a) Why per-plan survivor-pruning breaks — the crash class, characterized

The committed proof (2008b713, trace in the 4.4 section) plus a code-level
re-derivation of the mechanics this pass:

1. Plans are forced INCREMENTALLY. When plan P builds (build-from-IR: before
   any execution), a future consumer plan Q's graph **does not exist yet** —
   not as pending nodes, not as live roots. The sharing that later materializes
   (node 2675, a narrow in both the 388- and 364-node sibling plans) arises at
   Q's CONSTRUCTION time (structural reuse/CSE maps Q's subgraph onto the
   existing node). So at P's build, a complete walk of the entire live pending
   graph — the maximum information available — reports the node NOT
   cross-plan-consumed and NOT tensor-held. No build-time analysis can do
   better; the information does not exist yet.
2. Q consumes P's output as a LEAF (a plan node with no producing action whose
   value is its `node.result` — the prior-plan-result-threaded-in class from
   cutover fix #5). Prune P's harvest and the result never exists →
   `Input not ready` when Q forces. Note the failure is **structurally loud**
   (a missing storage crashes; it can never silently read stale) — a property
   the design below preserves.
3. The lowered/cutover path escapes only via its **after-the-fact signal**:
   every result is materialized, then pruned by ACTUAL refcount at markStep.
   Answering the charter's sub-questions: it is *consumers in not-yet-built
   plans* (under incremental forcing); partial forcing is the enabling
   condition, not a separate cause; readbacks are not implicated (loss `item()`
   holds a live rc → its node is a survivor in every variant).

### (b) The fix: survivors are OBSERVED across the plan set, then the plan
### is REBUILT with pruned results

The thesis, made concrete: the step-scoped PlannerRegistry (which already has
the cross-plan view for temp packing) additionally learns, from the first
executed step(s), WHICH harvested results the plan set actually consumes — the
same after-the-fact signal the lowered path uses, captured one step later at
template granularity. Recurring templates have stable structural fingerprints
and deterministic node indexing, so step N's observation predicts step N+1.

- **Identity.** Each harvested result is stamped `(templateFp, nodeIndex, oi)`
  on its storage at harvest (the replay-harvest chokepoint,
  `compiled-plan.ts` ~1576, already knows all three).
- **Observation (two seams, both existing chokepoints).**
  (i) *Cross-plan consumption*: when a later plan's build/replay resolves an
  external input whose storage carries a stamp, record the stamp into the
  registry's `consumed: Map<templateFp, Set<"nodeIndex:oi">>`.
  (ii) *Step survival*: at markStep, any stamped storage still alive (user rc /
  snapshot) records into `survived`. Params/optimizer state/user-held tensors
  land here.
- **Rebuild.** Once a template has one fully-observed step (executed + a
  markStep with no new template created that step; optionally K-step
  hysteresis), invalidate its compiled plan and rebuild-with-pruned-results:
  re-run the existing harvest with `harvestPairs` = consumed ∪ survived (both
  functions are ALREADY parameterized on the result set — `harvestGenResults`
  takes `harvestPairs`, `planMemory` takes `resultSlots`; the mechanism slots
  in with no planner surgery). Everything else becomes an interval-bounded
  temp → packed → the +34% over-harvest collapses to the default cutover's
  survivor memory. Binary result/temp suffices for parity: the observed
  needed-set converges to exactly the default's live-result survivor set
  (survivor = held by rc = consumed later or user-held). Step-global
  multi-segment intervals are NOT needed in stage 1 (they are stage 3's
  structure).
- **New plan appearing mid-step (guard/invalidate, tape-guard spirit).**
  A new template whose inputs resolve against *stamped live* storages simply
  records consumption (conservative direction, no hazard). A new template
  referencing a *pruned* producer output hits the structurally-loud missing
  result. The executor keeps a step-scoped `prunedExecuted:
  Map<nodeId, {templateFp, nodeIndex, oi}>` (populated as pruned plans
  replay); the `Input not ready` path consults it and (1) permanently adds the
  pair to the producer template's needed-set + invalidates its pruned build
  (next step rebuilds conservative-then-re-pruned), (2) fails the CURRENT step
  with a diagnostic naming the producer — recovery of the already-overwritten
  value within the step is exactly stage-3 rematerialization (see (d)) and is
  deliberately out of scope; until then this residual class is a
  loud-once-self-healing event. In practice the recurring-sibling class (the
  actual observed crash) is covered by observation, and late-appearing
  consumers (eval/logging plans) overwhelmingly read params/grads, which are
  in `survived`/`consumed` from step 0.
- **Gates.** (1) Set-parity assert at the seam: on a recurring template, the
  pruned build-from-IR result set == the default cutover's live-result harvest
  set (single source, assert agreement). (2) Trajectory differential:
  build-from-IR-pruned vs default over 30 steps to fp noise, crossing the
  rebuild threshold (2+ observed steps). (3) Memory: 124M regression
  build-from-IR-pruned peak within noise of default (kills the +18-42%
  scaling table). (4) A crafted late-consumer test that exercises the guard
  path (new template reads a pruned intermediate → loud diagnostic +
  invalidation, next step conservative). (5) The existing ladder (gates 4/4,
  fullstack, suites).

### (c) Cost / unblocks

- **Cost:** ~250-400 src SLOC (stamping ~30; registry observation maps + two
  record calls ~80; rebuild-with-pruned-results trigger ~120, mostly reusing
  the existing build path; guard + diagnostics ~100). No new env flags (the
  mechanism lives behind `TORCHLETTE_BUILD_FROM_IR`, whose named sunset —
  becoming the sole default build source — is stage 2's flip).
- **Unblocks (stage 2, the deletion harvest — named per weight-norm):**
  build-from-IR default → the lowered-first-execution build source dies →
  delete the `record*` hook surface (~105 refs across ~15 backend files),
  `compilationRecording` + the recorded-stream build in `compiled-plan.ts`,
  the params-sequence cache (~38 refs; compiled plans already self-contain
  their params buffers), `liveResultHarvestPairs` (merges into the pruned
  action-output harvest), and the `TORCHLETTE_GENERATED_PLAN` flag (its named
  sunset). Estimated −800-1200 SLOC against stage 1's +250-400: **net
  negative after stage 2**, plus serializable plans / ~700 ms cold-start.

### (d) Extension point: rematerialization (stage 3 — designed for, NOT built)

> **CROSS-REF (task #99, `docs/arena-recompute-design.md`):** the checkpoint-
> recompute slice of this extension point is now designed. It uses reuse #1
> below (multi-segment RESULT liveness split at the declared `recomputeRef`
> boundary + a `generateStream` recompute sub-stream) and reuse #3 (b66ead78's
> arena-free mode dies by unification). MEASURED confirmation that the fix belongs
> HERE and not in the arena: distil@512 + checkpointing arena-ON steady-state has
> the whole +155% delta in the PLANNER REGISTRY (2833 MB), arena 1.8 MB.

The stamped identity `(templateFp, nodeIndex, oi)` is the cross-plan name of a
VALUE, independent of any live buffer — exactly what a remat edge must name.
Three planned reuses, zero speculative code now:
1. A checkpointed activation becomes a result whose lifetime is
   **multi-segment**: dead after forward's last read, re-created by a recompute
   sub-stream before backward's first read. `planMemory`'s release-queue model
   already expresses release-and-reclaim; the entry gains segments, and the
   recompute program is a `generateStream` over the producing subgraph — which
   build-from-IR already does without execution (generating a stream from IR
   with no recorded trace IS the remat primitive).
2. The guard-miss recovery in (b) upgrades from fail-the-step to
   remat-on-demand: same recompute emission, triggered lazily.
3. `b66ead78`'s arena-free checkpoint mode then dies by unification: the
   planner expresses "checkpointing" as liveness edges + recompute programs,
   and checkpointed training runs compiled like everything else — the named
   sunset for `TORCHLETTE_ARENA_LIVENESS=0`.
Design constraint honored now: keep `harvestGenResults`/`planMemory`
parameterized on the result set (already true) and route ALL cross-plan value
identity through the stamp — no second identity scheme.

### ADDENDUM (2026-07-08, design refinement — guard-miss recovery, bind-time
### feasibility): the crash mechanism corrected, and the soundness boundary

Re-deriving the crash against the template cache corrects one detail of (a)
and sharpens the guard design:

**The consumer's plan SHAPE is frozen by the template cache in a world where
the value existed.** The consumer template (the 364 plan) was collected during
warmup, when the producer's lowered execution HAD materialized the shared
node's `node.result` — so the collection treats it as a LEAF (external input,
no producing action, upstream subgraph excluded from the plan). A later pruned
replay of the producer stops producing that result, and the CACHED consumer
plan's external resolution finds nothing. The miss is therefore detectable at
the CONSUMER's bind time — external slot pre-population (`compiled-plan.ts`
phase 1, `kind: "external"`) or `collectExternalInputBuffers` — **before any
of the consumer's side effects commit.**

**But "fall back to lowered execution for that step" is NOT sufficient as
stated.** The cached plan's leaf input has no value on ANY execution path —
lowered execution of the same frozen plan shape hits the same missing input.
The sound fallback is **template invalidation + fresh plan RE-COLLECTION**:
plan collection is graph-driven (given current materialization state, it
builds a plan computing what's needed from what exists), the missing node's
`.inputs` chain is intact (IR nodes are held by the producer's template), and
under pruning its un-harvested ancestors also lack results — so a fresh
collection naturally includes the whole recompute slice down to surviving
storages (params, consumed/surviving results) and executes it lowered.
User-invisible, loud in telemetry, and the needed-set grows so the next step
replays with the value harvested.

**The soundness boundary (the residual STOP sub-case): recompute reads
CURRENT storages.** If any in-place mutation committed between the producer's
pruned replay and the miss — the optimizer's `copy_`/`adamStep` updating
params, in the SAME step (backward+optimizer is one plan) — the recompute
slice re-executes against NEW param values and silently produces a DIFFERENT
value than the one pruned. That is the silent-wrongness class this campaign
exists to kill, and **the tree has NO version-counter substrate to detect it
per-storage** (grep: no `bumpVersion`/`_version` anywhere; the spec's version
mechanism was never landed). The minimal sound rule is therefore a
step-scoped IN-PLACE-COMMITTED counter: recompute-fallback is taken only when
zero in-place ops committed since the producer's replay; otherwise the miss is
unrecoverable-in-step and must fail loudly (the fail-the-step sub-case,
flagged per protocol rather than papered over). With K=3 observation
hysteresis + a rebuild limit (needed-set growth capped — a template whose
needed-set grows more than ~2× pins to the conservative full harvest
permanently, no specialize/invalidate churn), the miss itself should be rare;
the in-place-committed refinement bounds what the rare miss can do.

Status: design refinement only — implementation remains gated on direct
sign-off.

**Telemetry (relayed disposition, 2026-07-08):** the dirty-miss counter is not
just a guard — it is THE measurement deciding whether stage-3 remat ever needs
to serve this path. It must be queryable. Existing surfaces to fold into (no
new export if one fits): the executor stats precedent is
`getPayloadThrashStats()` (`executor.ts:343`); the loud-counter precedent is
`getGpuUncapturedErrorCount()` (`gpu-context.ts:364`). The guard-miss stats
(clean-miss recoveries, dirty-miss fails, pinned-conservative templates from
the rebuild limit) belong on ONE such surface, chosen at implementation time.

## Risks and mitigations
- **Imperative-op long tail** (phase 3): mitigated by per-op fallback — no
  cutover moment; coverage counters make progress visible.
- **Stream nondeterminism** (would break the differential): pinned by the
  phase-0 record-twice gate before anything else lands.
- **Planner bugs = the old seam bugs in new clothes**: the differential
  ladder (fullstack/regression/suites) plus the loud-GPU-error surface and
  idempotent pool remain as backstops; planner output is also validated
  structurally (no interval overlap, no aliasing-rule violation) at build
  time — cheap asserts, always on.
- **Browser**: same GpuCommand stream format; ENV accessor (eafd4e2) keeps
  flags safe; browser suite is part of every phase gate.

## Invariants that carry over unchanged
Fence-gated destruction (6e73011/3401445 + CLAUDE.md rule); scalars enter
kernels as data or fingerprint-guarded payloads (d822be9/2809588); WAR +
barrier + affinity scheduling (98eea29/b791d72); instance boundaries are
cache boundaries (4d94ff4); every new optimized path lands with a
cross-threshold differential.

## Stage 3 (PROPOSED, 2026-07-08): rematerialization unification

**Status: DESIGN ONLY. Not built. Awaiting review → hardening → separate build
go (the stage-1 protocol).** This section is the §(d) extension point made
concrete, grounded in the G0 measurements below. The prior work it stands on:
stage-1 observed cross-plan liveness (the stamp + needed-set + guard substrate,
`src/executor/observed-liveness.ts`); the memory planner + PlannerRegistry
(per-plan interval allocation, `src/executor/memory-planner.ts`); the frontend
checkpoint recompute (`src/nn/checkpoint.ts`); and `b66ead78` (the arena-free
checkpoint bypass this stage dies-by-unifying).

### G0 — measured first (the bar the design must clear)

Production `WebGPUGPT2Trainer` via `tools/diloco-regression-check.ts`, V100
(sivri, solo), losses baseline-EXACT in every row (checkpointing is numerically
transparent — confirmed, not assumed). `cur` = steady-state CURRENT bytes
(round-flat); `peak` = cumulative peak; ms/step = round dt / 20 inner steps.

**Regression config (8L/embed128/H4/b8/seq256 — vocab-dominated, activations
small):**
| config | peak MB | cur MB | ms/step |
|--------|--------:|-------:|--------:|
| (a) arena-free lowered checkpointed — **the current production path** | 2087.6 | **353.8** | ~281 |
| (b) build-from-IR **compiled** checkpointed (`CHECKPOINT_ARENA=1`) | 3637.3 | **2589.0** | ~78 |

**124M-class (12L/embed768/H12/b1/seq256 — activations dominate; where the
`b66ead78` 6.6→2.6 GB win lives, and where remat matters):**
| config | peak MB | cur MB | ms/step |
|--------|--------:|-------:|--------:|
| (a′) arena-free lowered checkpointed — **BASELINE-TO-BEAT** | 7659.1 | **2638.2** | ~653 |
| (d′) compiled checkpointed (`CHECKPOINT_ARENA=1`) | 9357.9 | **5587.0** | ~275 |
| (b′) compiled NON-checkpointed (`CHECKPOINTING=0`) | 9270.8 | **5601.2** | ~274 |

**Two facts the bar rests on:**
1. **Checkpointing saves NOTHING under the compiled path today.** (d′) 5587 MB ≈
   (b′) 5601 MB — compiled-checkpointed ≈ compiled-non-checkpointed. The
   forward-activation savings checkpointing is supposed to deliver are entirely
   defeated. This is `b66ead78`'s disease (the retained buffers keep every
   activation resident) reproduced in the compiled path — the compiled
   registry's RESULT-HOLDER buffers are materialized once and rebound every
   replay (persistent across steps, `compiled-plan.ts:1848` + `memory-planner.ts`
   `resultHolder`), so a forward activation that any later plan reads is held for
   the template's whole life. Checkpointing's recompute plans then ADD persistent
   buffers on top (d′ peak 9358 > b′ 9271), making it marginally worse.
2. **Unification is a large SPEED win and, today, a large MEMORY regression.**
   Compiled is **2.4× faster** (275 vs 653 ms/step at 124M; 3.6× at small) —
   the prize, and why `b66ead78`'s lowered bypass is worth killing (step-tape 2b
   / taped training needs compiled steps, which the bypass forbids). But compiled
   checkpointed is **+2.9 GB current / +1.7 GB peak** over the arena-free
   baseline at 124M. **The design's central problem is MEMORY, not speed.**

**The bar (stated at 124M-class):** compiled-with-remat must reach **cur ≤ 2638
MB, peak ≤ 7659 MB** (match the arena-free checkpointed baseline — that is WHY
the bypass exists) at **≥ its current ~275 ms/step** (already ≥ the 653 ms
baseline with 2.4× margin), trajectory == fp noise. The win if achieved: the
`b66ead78` memory AND 2.4× the speed, one lifetime story.

**G0(c) — the segment set remat must free+recompute.** Selective checkpointing
(`model.ts:452`, default `selectiveCheckpoint=true`): per layer, ATTENTION runs
OUTSIDE checkpoint (its O and logsumexp L are saved → forward-plan results
consumed cross-plan by backward), MLP runs INSIDE checkpoint. So the segment set
= **`numLayers` MLP segments** (8 small / 12 at 124M), each recomputing
`ln2 → up-proj(d→4d) → gelu → down-proj(4d→d) → residual add`. Dropped per
segment ≈ up-proj + gelu outputs `[tokens, 4·embed]` ×2 + ln2/down `[tokens,
embed]` ×2 ≈ ~8 MB fp32 / ~4 MB f16 at 124M/b1/seq256 (×12 ≈ ~50–96 MB); kept
per segment = the segment input `h` `[tokens, embed]` (~0.8 MB, held by
`api.keep`). The MLP-internal activations are NOT forward-plan results (disposed
by `tidy` under the pack hook); the recompute re-creates them at backward.

### 1. The lifetime model — step-global multi-segment intervals

The root cause the G0 data isolates: **the memory planner today allocates
intervals PER PLAN, but a checkpointed/cross-plan activation's true lifetime
spans the plan set.** A forward activation consumed by backward is a RESULT
(exclusive registry entry, `resultHolder`), materialized once and held for the
template's entire multi-step life. The arena-free LOWERED path reaches 354 MB /
2638 MB precisely because it frees each activation at its last read WITHIN the
single fused execution and re-allocates it next step from the pool; the compiled
path pins it forever for rebind stability.

**The model: lift interval allocation from per-plan to STEP-GLOBAL, over the
plan set, using the stamp identity as the cross-plan lifetime coordinate.** A
value `(templateFp, nodeIndex, oi)` (the stamp already minted at the
replay-harvest chokepoint) gets a lifetime that is a set of `[produce, lastRead]`
SEGMENTS in step-global command order across all the step's plans:

- **Ordinary cross-plan activation** (attention O/L, residual stream x): ONE
  segment `[produced @ forward-plan ... last cross-plan read @ backward-plan]`.
  Its registry entry is RELEASED after the last read (relisted → reused by a
  later plan's temps / next step's allocations) instead of pinned for the
  template's life. This alone converts the bulk of the 2589/5587 MB persistent
  result set into step-scoped, packable memory — the arena-free path's behavior,
  reached WITHOUT giving up the compiled dispatch stream.
- **Checkpointed activation** (MLP internals): its stamp's lifetime is naturally
  MULTI-SEGMENT — `[produced @ forward ... last read @ forward]` (dies at the
  forward residual add; never read cross-plan because backward recomputes) then a
  DISJOINT `[re-produced @ recompute-plan ... last read @ backward]`. The gap
  between forward-death and recompute-revival is a FREE region the planner packs.
  Crucially, the two segments are produced by DIFFERENT node identities (the
  recompute is a fresh forward — see §2), so the planner does not need to "resume"
  one entry; it sees two independent short-lived intervals that its existing
  release-and-reclaim (`memory-planner.ts` release queue) already packs — the
  multi-segment lifetime is EMERGENT from cooperating with the frontend recompute,
  not a new allocator primitive.

**Where segment boundaries are known at plan/build time.** They are already
structured: `markAsCheckpointBoundary` (`plan-builder.ts:11`) stamps the last
recomputed node of each segment, and the forward pack hook / backward unpack hook
delimit the forward-death and recompute-revival points at the FRONTEND. The seam
the planner reads is the STAMP: an activation whose stamp is never recorded as
`consumed` (no cross-plan external-input read) and never `survived` (disposed by
tidy, not user-held) is — by the stage-1 observation — pruned to a temp with an
intra-plan `[produce, lastRead]` interval. **Checkpointed forward activations are
exactly this class** (pack-hooked → disposed → not cross-plan-read). So stage-1's
`prunedHarvest` ALREADY classifies them as freeable; what is missing is that (i)
under the arena gate they never reach build-from-IR (§5), and (ii) the
cross-plan-CONSUMED activations (attention O/L, residual x) are NOT pruned (they
ARE read by backward) yet are still pinned per-template rather than released at
their last cross-plan read. **The step-global interval is the mechanism that
frees THOSE** — the ones stage-1 correctly keeps as needed but the planner holds
too long.

**Packing the freed region.** The step-scoped shared `PlannerRegistry` already
packs temps across plans (phase 1.5). The extension: a released cross-plan result
entry is `relist()`ed at its last-read command (step-global order), not at
template teardown, so a later plan in the same step — including a recompute plan —
draws it from `seedFreeLists`. The `planMemory` release-queue model
(`releaseAt = lastUse + 1`) already expresses "reusable from the next command";
step-global lifting makes `lastUse` a cross-plan coordinate rather than a
within-stream index. **This is the "the entry gains segments … release-and-reclaim
already expresses it" of §(d), now concrete: the release point moves from
per-template to step-global-last-cross-plan-read.**

### 2. The recompute mechanism — COOPERATE with the frontend recompute; do NOT
### re-emit it

**Ruling: the planner COOPERATES with the existing frontend checkpoint recompute.
It does NOT subsume or replace it, and it emits NO recompute sub-stream of its
own.** This is the load-bearing precision the charter demands (double-recompute
and missed-recompute are the failure modes), and the reasoning is decisive:

- The existing recompute (`checkpoint.ts` unpack hook) already re-runs `fn` at
  backward, creating FRESH lazy IR nodes that are forced into their own
  plan(s) → compiled build-from-IR like any other plan. **`generateStream` over a
  producing subgraph — the §(d) "remat primitive" — is therefore ALREADY HAPPENING**:
  the frontend hands the planner a fresh subgraph, and build-from-IR generates
  its stream with no execution or trace. The primitive is not missing; it is
  driven from the frontend.
- If stage 3 ALSO had the planner emit a recompute stream for the same
  checkpointed value, the value would be computed TWICE at backward (frontend
  re-run AND planner re-emit) — the double-recompute failure mode, and a
  correctness hazard (two producers writing one stamp). So the planner MUST NOT
  emit its own recompute program while the frontend recompute exists.
- **What the planner does instead:** nothing new at recompute TIME — it simply
  (i) frees the forward activation's registry region at forward-death (via the
  step-global lifetime, §1), and (ii) lets the recompute plan's fresh temps draw
  from that freed region. The recompute plan is built from IR (no execution, no
  trace) at first backward, exactly as build-from-IR already builds every plan;
  its outputs land in planner slots the backward plan reads, exactly as today.
  The `[re-produced ... last read]` segment is just that plan's ordinary
  interval.
- **When is the recompute plan built?** On first backward (when the unpack hook
  first fires and forces the fresh subgraph) — the same first-execution build
  every template gets. What it binds: SURVIVING storages only — params (root
  persistent) + the segment input `h` (held by `api.keep`, so `survived` in the
  needed-set). It never binds a freed forward activation (that is the invariant
  that makes the free safe — §3).

The one-sentence declaration: **checkpointing is expressed as liveness (the
forward activation dies at its segment boundary) + a recompute plan (frontend-
driven, build-from-IR); the planner's only new job is a step-global lifetime so
the dead region is freed and the recompute packs into it.** No planner-side
recompute emission; no second recompute; no missed recompute (the frontend still
owns the trigger, unchanged).

### 3. Guard / invalidation story (loud, bind-time, never silent)

The step-global free introduces exactly one new hazard class, handled by the
stage-1 guard already in place, extended minimally:

- **A value freed early (at its observed last cross-plan read) that a LATER plan
  reads.** This is the stage-1 guard-miss verbatim: the consumer's external-slot
  pre-population (`compiled-plan.ts` phase 1, `observeConsumed` seam) finds no
  storage. Recovery = `RecoverableGuardMiss` → evict the consumer template +
  fresh lowered re-collection (which recomputes the slice from surviving
  storages). Already implemented; the step-global model changes only WHICH values
  can be freed early, not the miss mechanism.
- **Interaction with observed-liveness pruning + idle-retire.** A recomputed
  value is BY DEFINITION not surviving and not cross-plan-consumed — so the
  needed-set classifies it (correctly) as PRUNED (a temp), which is what lets its
  region be freed. This is not a conflict: it is the same classifier doing the
  same thing, now with the step-global release point actually realizing the free.
  Idle-retire is unaffected (it retires whole idle templates; recompute plans are
  active every backward).
- **The needed-set gains ONE distinction:** a cross-plan-consumed activation
  (attention O/L) is `needed` (kept) but its lifetime ENDS at the last consuming
  plan — so "kept" must mean "kept until last cross-plan read," not "pinned for
  the template's life." Concretely: `prunedHarvest` still keeps the pair (it IS
  consumed), but the planner assigns it a step-global-released entry rather than
  an exclusive `resultHolder`. The stamp's `consumed` observation already records
  WHICH later template reads it (`consumed: Map<fp, Set<ni:oi>>`); the step-global
  lifetime's last-read coordinate is derivable from that same observation (the
  last consuming plan's position in step order).
- **Structural loudness preserved.** A freed-too-early value is a MISSING storage
  (crash / guard), never a silently-stale read — the property (a) and the ADDENDUM
  guarantee. `TORCHLETTE_STRICT_GPU` / `STRICT_LIFETIME` remain the backstops.

### 4. The dirty-miss ruling, and the view-survival ruling

**DIRTY-MISS RULING: stage 3 does NOT upgrade the guard to remat-on-demand. The
guard stays exactly as-is (clean-miss → lowered re-collection; dirty-miss → loud
fail). Smallest-sufficient, and the data + structure both say so.** Reasoning:
- The stage-1 telemetry mandate ("dirtyMisses is THE measurement deciding whether
  stage-3 remat ever needs to serve this path") reads ZERO dirty misses across
  every gate landed so far, and zero guard misses at all on the recurring
  training templates.
- **Structural argument the checkpoint case cannot dirty-miss:** the checkpoint
  recompute reads ONLY surviving storages (params + kept segment input `h`), never
  a freed forward activation — that is the §2 invariant. And backward (with its
  recompute) runs BEFORE the optimizer's in-place param update within the step, so
  the recompute reads pre-update params. So the checkpoint path — the entire point
  of stage 3 — never routes through the pruned-producer guard, let alone its
  dirty sub-case. The guard serves only the residual general late-consumer class
  (eval/logging plans), which stage-1 shows is empirically zero and which reads
  params/grads (survivors), not freed activations.
- Therefore §(d) point 2 (upgrade guard-miss recovery to remat-on-demand) is
  **DECLINED as speculative** — do NOT build it. It is re-openable IFF the stage-3
  gate ladder ever records `dirtyMisses > 0` or a non-recoverable clean miss on a
  real workload; the counter is the trigger, wired to fail the ladder loudly.
  (This is the "if the data says the guard path never needs remat, say so and
  don't build it" branch of the charter, taken.)

**VIEW-SURVIVAL RULING: adopt the PERSISTENT pruned-pair→stamp map + view
reconstruction on miss; drop the conservative view-keeping ("b"-source pairs).
NOT full remat.** This is the resolution the lifetime model makes natural:
- The open question (stage-2 inc-1): a harvested VIEW whose handle died but whose
  base survives is kept conservatively (+155 MB small-scale; b=3 at 124M) because
  `prunedExecuted` is STEP-SCOPED, so a LATER-step consumer of a pruned view would
  raw-crash ("Input not ready"), unguarded.
- Under the step-global lifetime model, a view of a surviving base is the CHEAPEST
  possible reconstruction: re-running the view op (narrow/permute/reshape) on the
  live base is O(1) metadata, no kernel, no recompute sub-stream. So the natural
  fix is to make `prunedExecuted` PERSISTENT across steps (a `Map<stamp, viewOp +
  baseStamp>`), so a cross-step view reader hits the guard, and recovery
  RECONSTRUCTS the view from the (surviving) base rather than failing or
  conservatively keeping. Full remat (recompute sub-stream) is overkill: a view
  has no compute to redo.
- Why not "just keep the conservatism": it is a real cost at small scale and,
  more importantly, it is a SECOND lifetime rule (keep-view-because-base-alive)
  outside the one stamp-identity model — the design constraint is ONE identity
  scheme. The persistent pruned-pair map routes view survival through the SAME
  stamp + guard machinery. At 124M its memory value is small (b=3), so it is not
  urgent — but it is the RIGHT shape and removes a special case, so it lands with
  stage 3 rather than as separate debt.

### 5. What dies (unification, with ref counts)

- **`b66ead78`'s arena-free checkpoint bypass** — the whole reason stage 3
  exists. Deletions: `WebGPUGPT2Trainer.initialize()`'s
  `if (checkpointing && CHECKPOINT_ARENA!=="1") setBufferArenaDisabled(true)` +
  the log line (`webgpu-gpt2-trainer.ts:150–161`); the `arenaDisabled` option in
  `OptimizedExecutionOptions` + its `executePlanOptimized` OR-term
  (`executor.ts:146–152, 3016–3026`); `RuntimeEngine.bufferArenaDisabled` field +
  `setBufferArenaDisabled` + the two threadings (`engine.ts:382, 518–..., 764,
  887`). **~49 lines across 3 files** (the exact `b66ead78` diff, reverted).
  GATED ON: compiled checkpointed reaching the G0 bar (else the bypass is still
  the only path that frees activations).
- **`TORCHLETTE_CHECKPOINT_ARENA`** flag (13 refs) — born with the bypass, dies
  with it. Named sunset, executed.
- **`TORCHLETTE_ARENA_LIVENESS=0`** (legacy unbudgeted arena opt-out; ~13 refs
  across `adam.ts`, `webgpu/index.ts`, `buffer-arena.ts`, `executor.ts`,
  `compiled-plan.ts`) — its last role was a planner-bug fallback; once the planner
  is the sole memory authority for checkpointed AND non-checkpointed training, it
  is subsumed. **This is the phase-4-inventory named sunset, now with its
  precondition met.**
- **The old checkpoint SEGMENTATION path** — `enableCheckpointSegmentation` /
  `enableTrueSegmentation` / `segmentPlanAtCheckpoints` / `executePlanSegmented`
  (`plan-builder.ts:20`, `sequential.ts:163`, `engine.ts:857–872`,
  `frontend/torchlette.ts:171`, `types.ts:109`). `b66ead78`'s message already
  records that this branch is UNREACHABLE under fusion (the default); it is dead
  code the unification obsoletes. Verify-then-delete (grep confirms only
  option-plumbing + spec callers). **~est. 120–180 SLOC.**
- **`options.bufferArena` gate on the build-from-IR block** (`executor.ts:1762`) —
  not deleted but INVERTED in meaning: build-from-IR must run arena-free, so this
  gate is replaced by "planner owns buffers, no arena needed" (the arena becomes
  purely a lowered-first-exec concept, itself on the phase-4.5 chopping block).

**Expected net SLOC:** mechanism ADDED ≈ 200–350 (step-global lifetime in
`memory-planner.ts` + `observed-liveness.ts`: cross-plan last-read coordinate,
release-at-last-cross-plan-read, persistent pruned-pair map, view reconstruction
in the guard recovery). Deletions ≈ 220–280 (bypass 49 + segmentation 120–180 +
two flags' plumbing). **Net roughly FLAT to slightly negative**, plus the strategic
payoff (compiled checkpointed training → step-tape 2b unblocked).

### 6. Cost / risk / staging / gate ladder

**SLOC:** ~200–350 added (§5), net ~flat. No new env flag (the mechanism lives
behind `TORCHLETTE_BUILD_FROM_IR`, already default-on; the campaign's flag budget
is spent on SUNSETS here, not new flags).

**Hardest correctness risk (the prior, confirmed as the real one): the
step-global free reads/reuses a region whose value a not-yet-observed consumer
still needs — the stage-1 dirty-miss class, now with a larger free surface.** The
soundness guarantee is the same one the ADDENDUM established and stage 3 must NOT
weaken: a value is freed early ONLY at its OBSERVED last cross-plan read, after
K=3 convergence, and a miss is STRUCTURALLY LOUD (missing storage → guard, never
silent stale). The specific in-place hazard — the optimizer's `copy_`/`adamStep`
mutating params between a free and a recompute — is bounded by the existing
step-scoped in-place-committed counter (`noteInPlaceCommit`,
`_hasInPlaceCommit`): recompute-recovery is taken only when zero in-place ops
committed since the freed producer's replay; else loud fail. **The checkpoint
path provably avoids this** (recompute reads survivors, pre-optimizer — §4), so
the hazard is confined to the general late-consumer residual the guard already
covers with dirtyMisses=0. The design adds NO new silent path: every new free is
gated by observation + guarded by the loud miss.

Secondary risks: (i) step-global interval overlap bugs — mitigated by extending
`memory-planner.ts`'s existing always-on structural audit (no overlapping
lifetimes on one entry) to the cross-plan coordinate; (ii) fence/reuse safety of a
cross-plan-released buffer — must honor `canRecycle` (in-flight encoder claims) at
the step-global release point exactly as the pool does (the buffer-pool invariant:
never reuse a buffer whose queued reader hasn't dispatched).

**Staging (gated increments, each independently shippable):**
1. **S3.0 — instrument (no behavior change).** Add a step-global-lifetime
   ANALYSIS pass that COMPUTES the cross-plan last-read of every stamped result
   and reports how much memory an early release WOULD reclaim, diffed against the
   arena-free-lowered `cur`. Confirms the model predicts the 2589→354 / 5587→2638
   gap before any buffer moves. (Mirrors the phase-4 `[ir-derive]` differential
   discipline.)
2. **S3.1 — release ordinary cross-plan activations early** (attention O/L,
   residual x): relist their registry entries at last cross-plan read. Gated on
   NON-checkpointed compiled memory dropping toward the arena-free line with
   bit-identical trajectory. (This alone recovers most of the gap; it is
   checkpoint-independent.)
3. **S3.2 — run build-from-IR under checkpointing** (drop the `options.bufferArena`
   gate for the arena-free case; let the frontend recompute's plans build from
   IR). Gated on (d′) reaching the G0 bar.
4. **S3.3 — persistent pruned-pair map + view reconstruction** (§4 view ruling),
   dropping the "b"-conservatism.
5. **S3.4 — the deletions** (§5), each gated on the bar holding.

**Gate ladder (for eventual implementation — MUST include):**
- `test/oracle/gpt2-checkpoint-parity.spec.ts` + all `test/checkpoint-*.spec.ts`
  green (checkpoint numerics unchanged).
- **Checkpointed-compiled vs arena-free-lowered trajectory ~1e-5** over ≥30 steps
  (the new cross-path differential this stage owns — must cross the convergence
  threshold, K≥3).
- **Memory: 124M-class compiled checkpointed `peak ≤ 7659 MB`, `cur ≤ 2638 MB`**
  (the G0 bar); small-config `cur ≤ ~354 MB`.
- Production regression (`diloco-regression-check.ts`) baseline-EXACT
  {9.81,5.92,5.15,4.64}, flat memory, zero leak — in the checkpointed default.
- Full suite green in BOTH flag states (`BUILD_FROM_IR` default-on and `=0`), CPU
  + webgpu.
- Decode stack (gen-tape + kv-differential) PASS — the compiled path serves
  inference; must not regress.
- `TORCHLETTE_STRICT_LIFETIME=1` + `TORCHLETTE_STRICT_GPU=1` over the checkpointed
  build-from-IR run: zero throws, zero uncaptured GPU errors.
- `test:gates` 4/4; `observed-liveness.spec.ts` extended with a cross-plan
  early-release + late-consumer-guard case; **`dirtyMisses === 0` asserted across
  the whole ladder** (the ruling's live tripwire).

### Open review questions (for the hardening pass)
1. **Step-global last-read vs plan re-ordering.** The cross-plan last-read
   coordinate assumes a stable step-global plan ORDER. Plans force incrementally
   and the order can vary (warmup vs steady). Is "last read observed in the K
   convergence steps" a sound upper bound, or can a steady-state step reorder a
   consumer LATER than observed → early free → guard-miss churn? (Prior: the
   recurring templates have deterministic order; needs confirming the CROSS-plan
   order is equally stable.)
2. **Rebind identity of a released-then-reused result buffer.** Compiled replay
   rebinds recorded buffers for speed. If a cross-plan result entry is released
   and a later plan reuses it, the next step's replay must rebind the SAME
   assignment deterministically (the registry's epoch-scoped determinism should
   give this, but the result→temp reclassification is new — does it interact with
   `_lastHarvestIds` / idle-retire's destroyed-storage gate?).
3. **Is S3.1 (release ordinary cross-plan activations) alone enough at 124M**, or
   is the residual after it still activation-recompute-bound such that S3.2
   (checkpointing under compiled) is load-bearing for the bar? The G0 (b′)≈(d′)
   equality suggests the persistent-result release dominates and checkpointing is
   secondary at THIS config — but at longer seq / bigger batch the activation
   fraction grows (the over-harvest scaling table). Worth measuring S3.1 alone
   before committing to the full stack.
4. **Interaction with the scoped-memory epoch migration** (`scoped-memory-design.md`):
   the step-global lifetime is keyed on the step/markStep boundary
   (`observeStepBoundary`). Under the epoch model, "step-global" becomes
   "epoch-global." Should stage 3 key on the epoch id from the start (per §8 of
   scoped-memory) to avoid a third migration?
5. **Double-recompute audit.** The COOPERATE ruling forbids a planner-emitted
   recompute. Is there any path where the harvest/guard could ALSO force the
   pruned forward subgraph (e.g. a guard-miss on a checkpointed activation that
   SHOULD have been recomputed by the frontend but was read as an external)? §2
   argues no, but this is the correctness lynchpin and deserves an explicit
   instrumented check in S3.2.

### S3.0 RESULT (2026-07-08, measured): the consumed-only overlay premise is
### FALSIFIED — the gap decomposes differently. STOPPED for re-approval.

S3.0 landed as designed (observation-only): `observeConsumed` gains the
consumer fp; per-pair last-reader tracking with K-step stability + an
`everSurvived` record (neededSrc keeps only the FIRST source, so survival of a
consumed pair was invisible); harvested-result → planner-entry registration at
build (cleared on invalidate/retire); releasability classification + telemetry
(`releasableLastReader`, `debugReleasableSummary`, `debugTopHeldPairs`,
releasable counters on `getObservedLivenessStats`); probe tool
`tools/t-remat-instrument.ts`. Zero behavior change.

**The measurement (124M-class, checkpointed, compiled via CHECKPOINT_ARENA=1,
5 rounds, entry-deduped attribution — pairs aliasing one entry counted once):**

Step-globally releasable under the approved rule (consumed-only ∧ never
survived ∧ stable last reader ∧ not mandatory): **2.2 MB** — not the ~2.9 GB
gap. The design section's §1 assumption ("cross-plan-consumed activations…
pinned per-template rather than released at their last cross-plan read" as the
bulk) is FALSIFIED at this config: under selective checkpointing the
cross-plan activation class (attention O/L) is only ~2 MB.

**Where the live 3501 MB of exclusive result entries actually sit** (cur 5587
vs arena-free-lowered 2638; registry materialized 4471 = 3501 results + 970
temps; +1116 non-registry externals):
| class | MB | physically | freeable by |
|---|---:|---|---|
| src=s true survivors | 1674 | packed m/v state chain, f16 weight-cast cache, scaler-deferred-readback-held unscaled grads (the lowered path holds equivalents — its 2638 cur includes ~500 MB of the same scaler-held grads) | nothing (correctly held) — EXCEPT ~260 MB of f16 casts DUPLICATED across two forward template variants (each template's build harvests its own exclusive copy of the same cached cast) |
| mandatory-CONSUMED, dead at boundary | ~1100 | the scaled grads (backward outputs read by clip/optimizer, disposed by zeroGrad before the boundary): wte.grad add 167.8, 24× MLP grads 16.8 each; + forward's f16 wte cast 134.2 + logits cast 67.1 read by backward | boundary-cyclic release (B below) |
| mandatory-NEVER-OBSERVED (src=none) | ~700 | DEAD grad-chain intermediates: the tied-wte grad's matmul (167.8) and scatterAdd (167.8) pieces read only INTRA-plan by the final add — harvested solely because the template's frozen `plan.outputIndices` (grad-accumulation roots captured at collection) marks them mandatory | mandatory-set pruning (A below) |
| temps (materialized) | 970 | planner temp entries; the lowered path holds the analogous bytes as POOL-IDLE buffers, which `cur` does NOT count — a metric asymmetry to resolve before holding the bar to `cur` | boundary temp release / metric fix (C) |

`crossPlanPersistentBindings` probe: EMPTY — the persistent-slot consumption
hole hypothesis is also falsified; the src=none class is frozen-mandatory
over-declaration, not invisible reads.

**Reshaped S3.1 (NOT built — stopped for re-approval), in impact order:**
- **(A) Mandatory-set pruning, ~700 MB:** extend stage-1 pruning to mandatory
  pairs observed dead (never consumed, never survived, K steps) — they become
  temps; the existing guard covers late readers. REQUIRES a readback
  observation seam first (`item()`/`cpu()` → observeConsumed with a
  READBACK sentinel that pins the pair permanently): the loss IS a mandatory
  pair whose only reader is a readback, invisible to observation today —
  pruning it without the seam would break `loss.item()`.
- **(B) Boundary-cyclic release, ~1100 MB:** a mandatory-consumed pair that
  NEVER survives the boundary and has a K-stable last reader is dead from its
  last read until the producer re-writes it next step. Its entry becomes
  claimable by plans positioned AFTER the last reader or BEFORE the producer
  (cyclically). Soundness: survival observation plays the role the lowered
  path's rc plays — any tensor alive at the boundary marks its pair
  unreleasable, so cross-step readers imply survival and are never overlaid;
  within-step readbacks are safe because claims only take effect in the next
  cycle. Needs step-position tracking (deferred from the original design),
  planMemory claim events, the loud clear-at-release seam, and the readback
  seam from (A).
- **(C) Temp/metric parity, ~970 MB + rounding:** decide whether the bar's
  `cur` should count pool-idle bytes (physical parity) or whether registry
  temp entries should release-to-pool at the boundary; measure A+B first.

A+B ≈ 1.8 GB of the 2.9 GB gap; whether that clears the bar depends on (C)'s
metric ruling. The dirty-miss and view-survival rulings are unaffected. The
S3.0 mechanism (stamps, last-reader, everSurvived, registration) is exactly
the substrate A and B need — nothing built is wasted.

### Stage-3 (A) LANDED (2026-07-08): readback seam + mandatory-set pruning;
### two general recorded-path bugs found and fixed en route

**(A) as approved:** the readback observation seam (`observeReadback` at
`RuntimeEngine.cpu/readTopK/startItemReadback`, AFTER force — observe-only,
no forcing/ordering change; "r" source + `everReadback` pins the pair against
any future step-global release) + mandatory-set pruning (`prunedHarvest`'s
unconditional set shrinks from terminal+`plan.outputIndices` to the TERMINAL
only — declared outputs must now earn their harvest slot through observation:
plan reads, boundary survival, or readbacks). A pruned pair read back
first-ever-after-convergence self-heals via `readbackMiss` (needed-set grows +
producer invalidated + THIS read fails loudly naming the recovery) — the same
epistemic boundary as stage-1 pruning, now covering the readback channel.

**Two general bugs the increment surfaced (both landed, both pre-existing):**
1. **Recording result-collection claimed results it didn't produce.** A plan
   sharing nodes with a sibling template (the node-2675 class — here: 12
   `fusedAttentionForward` nodes shared with the forward template) records
   their PRE-EXISTING results' buffers as "unrecorded" → the over-conservative
   `unrecordedNodes` invalidation → the template re-records EVERY execution,
   forever (~95 ms/step overhead + entry churn). Fix: snapshot result-bearing
   node ids at recording start (`preExistingResults`); the collection skips
   them (consumers re-resolve per replay as externals via `planNodes[i].inputs`
   — the standard prior-plan-result path). Mandatory pruning made this fire
   every step (a pruned forward pair re-shapes the next step's merged plan into
   a sibling variant); the bug itself predates stage 3.
2. **`valid=false` without releasing planner entries.** The recorded build
   assigns registry entries (`finalizeCompiledPlan` → `planMemory`) BEFORE the
   `unrecordedNodes` check; invalidating by flag alone leaked every result
   entry as a dead-owner `resultHolder` forever — measured **~390 MB/step**
   (15.6 GB by round 2) under the re-record loop. Fix: route through
   `destroyCompiledPlanBuffers` (the single teardown path). New telemetry:
   `orphanResultMB`/`deadOwnerResultMB` on `debugPlannerRegistryStats` (now 0),
   `readbackMisses`/`convergenceInvalidations` on the observed-liveness stats.

**Measured (124M-class checkpointed, V100, unified PHYSICAL meter = tracker
`cur` + pool-held; both meters reported per the metric ruling):**
| arm | peak MB | cur MB | phys MB (pool-held) | ms/step |
|---|---:|---:|---:|---:|
| (a′) arena-free lowered — the re-derived BAR | 7659 | 2638 | **6242** (3604) | ~660 |
| (d′) compiled pre-(A) | 9358 | 5587 | — (not measured) | ~275 |
| (d′+A) compiled with (A) + fixes | 9358 | 5541 | **11289** (5748) | ~293 |

(A)'s pruning works exactly as designed — dead mandatory results eliminated
(backward 1595.6→1033.5 MB, live result entries 3501→2941 MB incl. a new
+131 MB sibling-variant plan), zero dirty misses, zero readback misses, loss
baseline-exact, full ladder green — but its NET memory effect is small
(cur −46 MB): most pruned entries were unmaterialized or already
temp-shared, and the sibling variant adds back its own plan. **The unified
meter reframes the problem**: the compiled arm's physical gap to the bar is
dominated by POOL-HELD warmup residue (5.7 GB idle — lowered warmup + arena
spills + one-time recordings, never drawn down at steady state; the
demand-trim was already tried and reverted, §4.3) and registry temps, not by
harvested results. Post-(A) held classes: survivors 1674 MB (parity),
**mandatory-consumed boundary-dead 1113 MB ((B)'s target)**, src=b 12 MB.

**Implication for (B):** even a perfect (B) (−1.1 GB) leaves phys ≈ 10.2 GB
vs the 6.2 GB bar — the pool residue (5.7 GB) and temp materialization
(≈1.0 GB) dominate. Clearing the physical bar requires the warmup-residue
class (build-without-execution for ALL plans incl. uncovered ones, or a
pool-budget policy) more than it requires (B). Reported to the coordinator at
the (A) checkpoint for the (B) ruling.

### Stage-3 idle-trim LANDED (2026-07-08, coordinator-ruled): steady-state
### pool-bucket trim at step boundaries — −4.1 GB on the compiled arm

**The ruling** (rejecting bar re-scoping as goal-post-moving, pool budgets as
the reverted demand-trim's hazard class, and coverage extension as the 4.4
endgame rather than a memory fix): the pool-side twin of arena-reclaim
(78c6f73) and stage-2 idle-retire. At a STEADY observed-liveness boundary
(every executed template converged or pinned, no new template — the same
activation condition as pruning), a pool size class not DEMANDED (any acquire
attempt: bucket hit, pending hit, or miss-then-new-alloc) for
POOL_TRIM_IDLE_TICKS=3 consecutive steady boundaries has its available bucket
destroyed. Why this avoids the reverted demand-trim's churn (§4.3): that trim
fired mid-step under pressure with a reservation-window demand signal that
missed out-of-window acquires; this one (i) triggers only BETWEEN steps at a
convergence-gated boundary, (ii) uses total demand (every acquire path
touches the class), and (iii) carries the churn guard below.

Mechanics (`buffer-pool.ts idleTrim`, seam `observed-liveness
setSteadyBoundaryTrimmer` → registered by `compiled-plan.ts`):
- **Fence-gated, accounting-exact destruction**: trimmed residents ride the
  existing `pendingDestroy` path (destroyed post-fence; the scrub clears every
  tracking set). Bucket residents were `trackDeallocation`'d at
  release-to-pool, so the trim deliberately BYPASSES `deferredDestroy` (which
  would double-deallocate) and adjusts `pooledBytes` directly. `canRecycle`
  consulted per buffer (defense-in-depth; residents are safe by construction).
  `pendingRelease` is never touched (in-flight).
- **Rewarm churn guard**: a fresh allocation for a trimmed class counts
  `trimRewarms`; >2 rewarms of one class pins it untrimmable permanently (the
  stage-2 rebuild-limit lesson). Telemetry: `bufferPool.getTrimStats()`.
- Zero new env flags; activation is purely observational.

**Measured (124M-class, both arms, both meters, 5 rounds):**
| arm | peak | cur | phys (pool-held) | ms/step |
|---|---:|---:|---:|---:|
| (a′+trim) arena-free lowered — the BAR | 7659 | 2638 | **6209** (3570) | ~655 |
| (d′+A+trim) compiled | 9358 | 5541 | **7164** (1623) | ~293 |

- Compiled arm: phys **11,289 → 7,164 MB (−4.1 GB)** — 643 buffers / 4,640 MB
  trimmed; trimRewarms=6 with 2 classes pinned, then FLAT (rounds 1–4
  byte-identical: the guard converges, no churn). Loss baseline-exact.
- Lowered arm: −34 MB only (4 buffers) — as predicted, its pool bytes are the
  LIVE per-step working set (demanded every step, never idle). The bar is
  honest and stands.
- **Remaining gap to the physical bar: +955 MB — almost exactly (B)'s
  1,113 MB boundary-dead-consumed class.** (B) is now load-bearing and
  sufficient-looking for the phys bar; peak (9358 vs 7659) remains
  warmup-transient high-water (set in round 0 before convergence+trim).

Ladder (all green): suite default 90+63 (one network-flaky relay run, green
on rerun); suite `=0` modulo the env-inheriting obs-spec arm; gates 4/4; obs
spec 5/5; checkpoint oracle 4/4; fullstack default/`=0`/STRICT maxΔ 8.6e-6 /
30 steps; production regression baseline-exact flat 2087.6 MB; kv-diff PASS;
gen-tape PASS.

### Stage-3 (B) LANDED (2026-07-08): boundary-dead release + last-reader
### temp overlay; the ALIAS HOLE found by the set-parity gate and closed

**(B) as ruled** — mandatory-consumed ∧ never-survived ∧ never-read-back ∧
K-stable-last-reader pairs release their registry entry INTO the last
reader's build: `planMemory` gains `externalReleases` (the entry joins the
free lists at the claiming plan's FINAL READ of the slot; a phantom-lifetime
audit asserts no claiming temp starts before that), the claim seam asserts
`plannerEntryClaimable` (still the producer's exclusive result entry, same
generation), claimed entries are co-owned (teardown/idle-retire invariant:
buffer dies with the LAST owner — unchanged `owners` semantics). Loudness —
never silent: after each claiming replay the producer's value is overlaid, so
pending-ref readers get node.results CLEARED (crash → the bind-time guard,
registered `released=true` → `claimMisses` attribution + per-pair revoke) and
every released storage is flagged `releasedOverlay` (materialized-ref reads
and readbacks hit the [lifetime] warn / STRICT throw; readbacks additionally
throw + pin the pair). Smallest-sufficient: NO step-position tracking — the
form-1 overlay (the last reader's own temps, after its final read) covers the
whole measured class; the producer re-writes the entry before any next-step
consumer reads (strictly-sequential queue order).

**THE ALIAS HOLE (a real silent-wrongness bug the set-parity gate caught
before it ever shipped):** the first cut released pairs whose BUFFER was
readable through OTHER stamps — a harvested VIEW of the released value (the
"b" class; e.g. clip's in-place write making a later harvest wrap a view of a
grad buffer) records its readers on the VIEW's stamp, so the base pair's
last-reader observation cannot bound the buffer's readers. Result: 2.4e-3
trajectory divergence in `test/observed-liveness.spec.ts` set-parity — silent
(zero misses, zero warns), exactly the class this campaign exists to kill,
and exactly what the differential-gates-cross-the-activation-threshold rule
is for. FIX (two levels, both through the one identity scheme): (i)
`everAliased` — the replay-harvest chokepoint marks a stamped base pair
permanently unreleasable when any harvested view chains to it (every alias
flows through that chokepoint, same- or cross-template); (ii)
`releasableEntryReader` — an entry carrying MULTIPLE registered pairs (an
in-place output and its view sharing one entry) releases only if every pair
is independently releasable with the SAME reader. After the fix: set-parity
5/5, trainer numbers unchanged (the trainer's claims were alias-free).

**Measured (124M-class checkpointed, both arms, both meters):**
| arm | peak | cur | phys (pool-held) | ms/step |
|---|---:|---:|---:|---:|
| (a′+trim) arena-free lowered — BAR | 7659 | 2638 | **6209** (3570) | ~655 |
| (d′+A+trim+B) compiled | 9358 | **5089** | **6712** (1623) | ~305 |

181 claims across 3 plans (releasable 883.6 MB post-alias-exclusion; the
optimizer's temp entries collapsed 863→136 MB — the designed overlay), net
**cur −452 MB, phys −452 MB**, claimMisses=0, dirtyMisses=0, loss
baseline-exact. **Remaining gap to the physical bar: +503 MB**, of which
~260 MB is item (4)'s duplicated f16 casts (next in queue) and the rest is
size-class rounding + the sibling-variant plan + releasable-but-unclaimed
entries (no size-class-matched temp demand after their release point).

Ladder (all green): suite default 90+63; suite `=0` modulo the
env-inheriting obs-spec arm; gates 4/4; set-parity 5/5; checkpoint oracle
4/4; fullstack default/`=0`/STRICT within same-config run-to-run noise
(6.7e-6–9.5e-6 measured across repeat runs); production regression
baseline-exact flat; kv-diff PASS; gen-tape PASS.

**Stage-3 completion sequencing (coordinator peak ruling, recorded):**
peak-parity is DEFERRED to 4.4-coverage (round-0 warmup transient, not
steady-state) BUT it GATES S3.4's deletions — the `b66ead78` arena-free
bypass and the `ARENA_LIVENESS=0` sunset may NOT be executed while compiled's
peak exceeds the arena-free peak (deleting the lower-peak path would regress
peak-constrained users). Stage 3 therefore completes as: (B) landed → steady
phys within ~8% of the bar (closing via item 4) → mechanism COMPLETE and the
unification VALIDATED; the deletions queue behind 4.4-coverage or a
peak-acceptability measurement at target scales.

**Item (4) — duplicated f16 casts (~260 MB): DEFERRED WITH A NOTE (the
pre-authorized hairy case).** The duplication is NOT a stamp-keyed
entry-sharing problem: `f16WeightCache` is populated by the Adam kernel's
DUAL-WRITE (`gpu-context.ts:47`) and consulted only by the lowered `cast()`
path (`ops/views.ts:69`, explicitly bypassed while recording). Generated
plans never consult it — each sibling forward template's generated cast
dispatches into its own exclusive result entry, hence one duplicate set per
template variant. A sound fix must let the GENERATED cast path bind the
dual-write f16 buffer as a persistent slot with staleness tied to the
in-place param update — a cross-seam change (optimizer kernel ↔ cast op ↔
generated slot kinds ↔ cache invalidation), not a mechanism-work close-out.
It naturally folds into 4.4-coverage/phase-3-style capture work (where
op-consulted caches become planner-visible), alongside the deferred peak
work. Stage-3 mechanism work closes here: phys 6712 vs bar 6209 (+8%),
with the residual named (dup-casts ~260, size-class rounding, sibling
variant, unclaimed releasables).

## Task #43 4.4-COVERAGE — stream-generator coverage extension (2026-07-11)

Extended `generateStream` until no RECURRING plan needs the recorded-build
fallback for a *coverable* reason. Method: instrument the bail-class frequency
FIRST (measured-frequency-order, per the charter), on the three real workloads,
THEN cover only the classes that actually fire; skip phantom classes.

**The census tool** (`tools/t-coverage-census.ts`, flag `TORCHLETTE_COVERAGE_CENSUS=1`,
exports `dumpCoverageCensus`/`resetCoverageCensus` from `executor.ts`) runs
distilgpt2-class training, gpt2 decode, and the 124M DiLoCo class, aggregating
build-from-IR bail classes by fingerprint and separating RECURRING bails (a
fingerprint that re-enters the build-from-IR block ≥3× and keeps bailing — a
real gap) from TRANSIENT warmup plans (reach 1–2×, run lowered once — fine) and
covered-and-compiled plans (reach the block once, then compile).

**KEY MEASUREMENT FINDING — the coverage question is config-gated.** The
production DiLoCo trainer runs `checkpointing=true`, which ties the buffer arena
OFF (commit b66ead7) → those plans run LOWERED (no compiled replay, no
build-from-IR, no recorded build). Build-from-IR (and thus the coverage
question) is only live in the arena-enabled / compiled-replay configuration —
the canonical `parity-fullstack-tl` / gate config (raw GPT-2 + manual optimizer,
model-level checkpoint but arena ON). So the census measures training with
`CHECKPOINTING=0`, the config that actually exercises the generator.

**BEFORE (recurring bails, arena-enabled):**
- distil-train / 124M-diloco: `row-program` (nodes 479 / 925, reaches 5) +
  `data-source:full` (nodes 296 / 566, reaches 3).
- gpt2-decode: `fused[no-storage]` + `op:max` (nodes 339, reaches 4).
- (All `*[config-missing]` / `matmul-epilogue[no-config]` / `fused[no-input-pattern]`
  classes co-occur on TRANSIENT warmup plans — the tile-kernel config buffer is
  cached only post-execution, so build-from-IR's FIRST attempt at a template
  bails, runs lowered+caches, and the 2nd+ execution COVERS. Self-healing, not a
  stream-generator gap.)

**COVERED:**
1. **Row-programs** (multi-reduction+elementwise → one `perRowKernel`) —
   `planRowProgramDispatch` (row-program-dispatch.ts, single-sourced with
   `dispatchRowProgram` via `kernel.plan({num_rows,feature_dim})`) +
   `generateRowProgram` (stream-generate.ts). One ALLOC(output)+DISPATCH; kernel
   keyed by `program.cacheKey` (structural), uniforms from the action, output
   size from the consumer's shape (the same seam the dispatcher asserts). The
   largest recurring training bail (nodes 479 / 925). Ledger-verified flat
   (`t-ledger-attack-probe.ts` reachDrift/totalDrift = 0 with it ON).

**ATTEMPTED THEN REVERTED — data-source creation ops (`full`/`arange`/`rand`/
`randn`/`bernoulli`):** they LOOK generatable (pure shape+payload, NO tensor
inputs — like `tensorFromArray`, emit ALLOC+WRITE, RNG seed baked in payload).
But UNLIKE `tensorFromArray` (small stable-upload fast path, plan-owned buffer)
and `zeros` (TAG_CLEAR into the slot), their `TAG_WRITE` replay takes the LEGACY
`executeOpSync` path (`compiled-plan.ts`), which ALLOCATES A FRESH buffer EACH
replay (that path was built for one-time weight loading, not per-step recurring
creation). The fresh-per-replay storage is not planner-managed → a real
storage-LEDGER LEAK: `rc-ledger.spec.ts` / `t-ledger-attack-probe.ts` on the
clip/scaler trainer went reachDrift +13 / totalDrift +8; bisected to exactly
this generation (drift → 0 when reverted, unchanged by row-program). Covering
them correctly needs the TAG_WRITE legacy path to write INTO the planned slot
instead of allocating fresh — a separate executor change, out of scope. They
stay bailed (lowered). **Lesson: "no tensor inputs → safe to generate" is FALSE
— the REPLAY buffer lifecycle is the seam, and the differential that catches it
is the LEDGER probe, not the stream diff (the stream diff was byte-clean).**

**DELIBERATELY BAILED (correctness, the ONE remaining recurring training bail):**
- `row-program[scalar-steptemp-input]` — a row-program whose external input is a
  MATERIALIZED 0-d scalar step-temp (the clip/scaler scale multiplicand fused as
  a `mul` preamble, e.g. `sum(mul(loss, scale))`). This is EXACTLY the
  stale-external condition `executeRowProgram` detects: it takes the SEQUENTIAL
  FALLBACK and runs the covered nodes as separate dispatches (mul then sum), NOT
  the fused kernel. Emitting the fused kernel would DIVERGE from the recorded
  fallback (one dispatch vs two) AND risk freezing the step-varying scale
  (silent corruption). The generator bails whenever a materialized inputRef is a
  0-d scalar — the executor's own fallback path, mirrored. This is a
  captured-state case analogous to `mean`'s invCount; closing it needs per-step
  scalar delivery (scalar-table-style) for materialized 0-d row-program inputs.
  The stream-generate gate caught it: covering the clean row-programs flipped the
  GradScaler-update plan to fully-covered, exposing this preamble; the bail
  restores that plan to the recorded path (baseline behaviour) while the clean
  row-programs generate.

**SKIPPED (deferred, evidence per class):**
- `fused[no-storage]` (gpt2-decode) — a fused kernel with a released strided
  (expand/broadcast) input; the strided-view class the doc already documents as
  "correctly excluded — layout not shape-derivable." Hard-deferred.
- `op:max` (gpt2-decode) — max/min reductions aren't routed to the reduction
  generators (only `sum` is). Coverable in principle (planFull/DimReduction take
  a reduceOp), but the SAME decode plan also carries `fused[no-storage]`, so
  covering max alone can't fully-cover it — partial value, deferred with the
  strided class. Decode is inference-only (noGrad), not the training path.

**AFTER:** row-programs cover (the largest recurring training bail); the
remaining recurring training bails are the deliberate `row-program[scalar-steptemp-input]`
(correctness) and the reverted `data-source:full`/etc. (ledger-unsafe, needs the
executor TAG_WRITE change). gpt2-decode's one recurring plan stays uncovered
(strided-view + max), inference-only.

**Gates (all green):** `npm run build`; `test:gates` 6/6; `t-stream-generate`
PASS (3 plans FULLY GENERATED 0 diverged, 1 plan partial-covered with the
deliberate row-program bail); `t-ledger-attack-probe` reachDrift/totalDrift = 0
(the data-source revert restored ledger balance); `parity-fullstack-tl`
compiled-vs-lowered max |Δ| = 2e-6 / 30 steps.

**Deletion-readiness verdict:** the recorded build is NOT deletable yet. It
still serves (a) the deliberate `row-program[scalar-steptemp-input]` bail
(needs scalar-table delivery for materialized 0-d row-program inputs), (b) the
decode strided-view + max class (inference), and (c) the build-from-IR
FIRST-execution `config-missing` transients (the tile-kernel config is cached
only post-execution, so the first attempt at every config-bearing template bails
to lowered and the 2nd+ covers — this is a build-from-IR cold-start property,
not a stream-generator gap, but it means the recorder still fires on template
warmup). The remaining recurring gap (a) is small and well-characterized; (c)
would need config-buffer geometry derivable pre-execution (a phase-3-style
capture). Until (a) and the decode class close AND the config-missing transient
is addressed, the recorded build stays as the correctness fallback.

## Task #43 recorded-build sunset — deletion MAP + STOP (2026-07-11)

A dedicated deletion pass was authorized to "delete the recorded-build path
(~−800 SLOC) and sunset `TORCHLETTE_GENERATED_PLAN`." Phase-A mapping against
the live tree (HEAD `715c30aa`) found the mission's precondition already
satisfied and its main body blocked by a still-open coverage gate. **Result:
STOPPED after the map — no deletion forced** (the campaign's explicit STOP rule:
if a recurring plan class still NEEDS the recorded replay for correctness, map
and report rather than force). This section IS that map.

### A1 — flag-family inventory (only GENERATED_PLAN was in this campaign's scope)
Live `TORCHLETTE_*` flags in this family (grep over `src/`):
- **`TORCHLETTE_GENERATED_PLAN` — ALREADY DEAD.** Zero occurrences in live code
  (`src/`, `test/`, `tools/`); only historical narrative in this doc. Killed
  2026-07-08 (B5, inc-3c) — after inc-3a deleted the recorded→generated cutover
  swap, the flag was an exact behavioral twin of `BUILD_FROM_IR=0`, so it was
  retired as debt. **The named sunset this pass was asked to execute is already
  executed.** Nothing to delete.
- **`TORCHLETTE_BUILD_FROM_IR`** — opt-out default-on (`!== "0"`). `=0` restores
  the record-then-replay build everywhere. NOT in scope (different flag); it is
  the recorded-build reference arm and dies WITH the recorded build, not before.
- **`TORCHLETTE_COMPILED_PLAN`** — opt-out default-on; `=0` = lowered reference.
  NOT in scope. The `parity-fullstack-tl.ts` gate is `COMPILED_PLAN=0` vs default
  and survives untouched.
- **`TORCHLETTE_MEMORY_PLANNER`, `TORCHLETTE_COMPILED_PLANNED`,
  `TORCHLETTE_STREAM_GENERATE`, `TORCHLETTE_ARENA_LIVENESS`** — all out of scope;
  semantics unchanged. `STREAM_GENERATE=1` is a verify mode that consumes the
  recorded build (below).

### A2 — what the RECORDING still serves on the default path (`BUILD_FROM_IR` on)
`buildFromIRActive()` (`executor.ts:540`) makes build-from-IR the default build
source. But recording is NOT dead under the default — three live masters:
1. **Uncovered-plan census fallback (CORRECTNESS, default path).** When
   `generateStream` returns `fullyCovered=false` (or `harvestGenResults` returns
   `genOk=false`), the build-from-IR block (`executor.ts:1831–…`) resets its
   captures and falls through to the lowered path with zero residue; the plan
   then **records on its next execution**. The generator has many genuine bail
   points (`stream-generate.ts`): strided/non-derivable views, chunked buffers
   `>maxStorageBufferBindingSize` (128 MB), batched dispatch `>64`, non-f32
   typed-buffer ops, copy-on-write dispatch, contiguous-copy prologues. These
   are real plan classes (the 124M plan hits the chunked-sum path), so the
   recorded build is load-bearing for correctness until 4.4-coverage closes the
   gaps. This is the STOP trigger.
2. **Verify-mode gates (2 of the 4 load-bearing gates).** `STREAM_GENERATE=1`
   builds BOTH the recorded and generated streams and diffs them
   (`compiled-plan-parity.spec.ts` gates 2–3; `t-stream-generate.ts`,
   `t-stream-determinism.ts` pins `BUILD_FROM_IR=0`). Deleting the recorded
   build removes the reference these gates diff against.
3. **The opt-outs** `BUILD_FROM_IR=0` / `COMPILED_PLAN=0` — separate flags,
   product-decision sunsets, not this campaign's.

### A3 — 4.3 residue (logsumexp oi=1): ALREADY CLOSED, verified, not re-done
The phase-4 A1 residue ("expose fusedAttentionForward oi=1 like grad_bias") is
landed: `generateAttention` returns `outputs:[{oi:0},{oi:1 lse}]`
(`stream-generate.ts:~1981`), the walker maps forward-attention non-primary
outputs into `nodeSlotExtra`, and `executor.ts:1186` exposes the payload. The
harvest bail rule (`harvestGenResults` hard-bails on any missing harvested pair)
is the safe superset of the designed "non-primary miss = bail." Full no-record
correctness is therefore NOT blocked on the logsumexp residue — it is blocked on
the broader generator-coverage gaps in A2(1).

### A4 — tests/tools referencing the recorded-build machinery (would move on deletion)
Not deleted this pass (deletion not executed); listed so the eventual harvest
names them: `test/compiled-plan-parity.spec.ts` (STREAM_GENERATE + BUILD_FROM_IR=0
arms), `test/observed-liveness.spec.ts` (BUILD_FROM_IR=0 cutover ref),
`tools/t-stream-generate.ts`, `tools/t-stream-determinism.ts` (pins
BUILD_FROM_IR=0), `tools/t-compiled-parity-probe.ts`, `tools/t-chunked-sum-probe.ts`,
`tools/t-observed-liveness-mem.ts`, `tools/t-observed-liveness-probe.ts`. All of
these are gates/probes for the surviving recorded build; they stay until it
sunsets.

### Verdict
The recorded-build sunset is gated (per the stage-3 coordinator ruling above:
"the deletions queue behind 4.4-coverage or a peak-acceptability measurement at
target scales"). 4.4-coverage — full generator coverage so no plan needs the
recorded fallback — is the prerequisite and is not yet built. **No `src/`
deletion is warranted or safe in this pass.** The one thing genuinely in scope
(the `GENERATED_PLAN` flag) was already sunset. Net SLOC delta of this pass: 0
in `src/` (docs-only). When 4.4-coverage lands, the deletion harvest (B2/B3: the
~80 `record*` refs + params-sequence cache + `buildCompiledPlan` + BUILD_FROM_IR=0
opt-out) becomes safe and is the ~−800–1200 SLOC win the original estimate named.

## Task #71 — view offsets → volatile uniforms (INVESTIGATION + SHORTCUT FALSIFIED, 2026-07-11)

**Goal (P0 runway, schedule-state §7):** a view's element offset (narrow
`start`·stride) must NOT be part of template identity — a per-position decode
narrow (static-KV / RoPE-slice) forks a new template per offset. Measured
failing-first (`tools/t-view-offset-templates.ts`): **6 distinct narrow offsets →
6 templates** (warmed structure), tripping the PAYLOAD-THRASH detector naming
`narrow`. The engine's own remedy text is the spec: "declare it volatile in
`PAYLOAD_HASH_EXEMPT` WITH a per-execution delivery mechanism."

**Where offset lives (map):** (1) fingerprint — `narrow.start` is hashed
(`fusion-detect.ts` `computePlanFingerprint`, `PAYLOAD_HASH_EXEMPT` doesn't list
narrow); (2) the narrow VIEW action itself carries NO offset in the generated
stream — it ALIASES to its input's slot (`stream-generate.ts` view case), so the
offset only ever reaches a kernel through a CONSUMER (contiguous/cast/strided
elementwise); (3) the consumer bakes the offset — `contiguousTileIR` /
`binaryBroadcastSpec` / `castSpec` / `unaryStridedSpec` pass a numeric offset to
`ctx.stridedLoad(...)` → `linearizeIndex` `this.u32(offset)` (a WGSL LITERAL), and
the generator captures+bakes it (`cachedStridedInputs.offset`, the `storage`
branch's `meta.offset`, `planContigCopy`'s `info.offset`). The rope kernel is the
counter-example (offset ALREADY a `cos_offset`/`sin_offset` uniform, task #58).

**LANDED (safe foundation, this pass):** `linearizeIndex`/`stridedIndex`/
`stridedLoad` now accept the base offset as `number | BlockExpr` — a numeric
literal (default, byte-identical WGSL) OR `ctx.uniform("base_offset")`. This is
the codegen primitive the volatile-offset fix builds on. Plus `debugTemplateCount()`
and the reproduction probe. Zero behavior change on the default path (gates 4/4,
offset-views 48/48, fullstack compiled-vs-lowered 7.6e-6/30 steps).

**SHORTCUT FALSIFIED — "exempt `narrow.start` + bail narrow plans to lowered."**
The tempting cheap fix (exempt the offset from the fingerprint so templates
collapse to 1, and force any narrow-containing plan to stay LOWERED so it reads
the offset live) is UNSOUND and was reverted. Two findings:
- **Value-based bail is unsound (offset-0-builds-first).** A compiled plan is
  built ONCE from whichever instance builds the template first — usually
  `start=0`. A start-0 build produces a valid-LOOKING start-0 kernel that is
  WRONG for a start-16 sibling replay. No per-consumer `offset>0` bail catches
  it (the builder's offset IS 0). The `#71` probe caught this exactly:
  templates collapsed to 1 but **correctness FAILED** (start-20 read start-0's
  region). The hazard is the SHARED template, not the built offset value.
- **Blanket "bail ALL narrow plans" REGRESSES the compiled trainer.** Forcing
  every narrow-containing plan lowered (even the fullstack GPT-2's CONSTANT
  `start=0` position/logits slices — safely compiled today) perturbs the
  compiled-plan liveness/harvest: fullstack **compiled diverged from its own
  lowered arm by 2.4 nats** with a `[lifetime] reading RECLAIMED storage`
  warning (the fix's LOWERED arm stayed byte-exact — so the lowered path is
  correct, the mixed compiled/lowered boundary the bail creates is not). Reverted.

**The complete fix REQUIRES the volatile-offset kernel change (not a bail).**
Offset must be delivered as per-replay DATA to the consuming kernel — a
`base_offset` uniform (the primitive is landed) repacked each replay from the
current instance's offset (TAG_UNIFORM, the `setAdamConfigUniforms` pattern), so
BOTH constant- and varying-offset narrows COMPILE (no plan bails, no liveness
perturbation, no regression). Threads through: (a) the strided specs (offset →
`ctx.uniform("base_offset")`), (b) both dispatch paths — the config-uniform
`createTileKernelDispatcher` (has the staleness guard) AND the `ElementwiseDirectPlan`
frozen-`paramsData` generator path (`planContiguousDirect`/`planBinaryDirect` +
`generateSequential`/`planContigCopy`), which must emit the offset as a VOLATILE
slot re-derived per replay from the node's narrow-offset (single-sourced with
`view-meta.ts narrowMeta` / `deriveResultMeta`), (c) `PAYLOAD_HASH_EXEMPT`
`narrow.start` (safe ONLY once (a)+(b) deliver offset as data), (d) the
`getConfigBuffer` staleness guard extended to recognize the base_offset repack.
This is a cross-seam change (tile-IR codegen + both dispatchers + generator +
fingerprint) with silent-corruption stakes; it is scoped but not landed here.

**LANDED (the complete fix, 2026-07-11).** Offset is now delivered as a
per-replay `base_offset` uniform (or `a_offset`/`b_offset`, `cond/x/y_offset`)
and `narrow.start` is `PAYLOAD_HASH_EXEMPT`. What shipped:
- **Codegen (`ops-tile-ir.ts`)**: `unaryStridedSpec`/`castSpec`/`binaryBroadcastSpec`/
  `whereSpec`/`contiguousSpec` declare the offset uniform(s) and read them via
  `ctx.uniform(...)` — NOT baked. WGSL is offset-independent (constant AND
  varying offsets share one template; the spec's "uniform-for-all inside
  compiled plans" choice — chosen for uniformity + no dual-mode classification
  bug, at the cost of the number-literal byte-stability).
- **Direct paths (`dispatch.ts`/`views.ts`/`where.ts`)**: `paramsData` gains the
  offset word(s) AFTER `size` (declaration order: `params(size, ...offsets)`).
  Chunked `createTileKernelDispatcher` sites pass the offset uniform as `0`
  (chunked binds from element 0).
- **Volatile delivery (single closure, two paths)**: `buildDirectOffsetRepack`
  (`stream-generate.ts`) returns a `pack(node)` that re-derives each input's
  offset from the CURRENT step's graph (`view-meta.ts deriveNodeOffset`, walking
  the narrow chain / live result meta) and repacks `[size, ...offsets]`. The
  GENERATED path emits it as a `TAG_UNIFORM` BEFORE the dispatch; the RECORDING
  path fires the SAME closure via the executor's `setPendingParamsVolatilePack`
  hook → `createParamsBuffer` → `recordVolatileUniform`, so both streams carry
  an identical TAG_UNIFORM and the segment diff matches. Only fires for
  direct-elementwise ops whose input chain contains a narrow (guarded tightly —
  matmul/reductions/fused kernels with narrow inputs use their own configs).
- **Fingerprint**: `narrow: Set(["start"])` in `PAYLOAD_HASH_EXEMPT` (`dim`/
  `length` stay hashed — they change strides/shape → codegen).

Verified: `t-view-offset-templates` 6 offsets → **1 template**, values correct
at every offset (was 6 templates). New in-suite gates (`compiled-plan-parity.spec.ts`):
the templates gate + a cross-offset replay gate (`t-view-offset-cross-replay` —
build at offset A, replay at B, assert B's region; the offset-0-builds-first
trap, now permanent). offset-views 48/48; fullstack compiled==lowered 8.6e-6/30
steps. Real-decode (gpt2 substitute — qwen3-1.7B loader stalls on this box):
16 decode steps forked **15→1** templates (steady 12→0), no PAYLOAD-THRASH.

## Task #43 4.4-COVERAGE — the four RESIDUES revisited (2026-07-12)

Re-ran the census on the three workloads (`t-coverage-census.ts`, arena-on /
CHECKPOINTING=0) and worked the four named residues to a verdict apiece. The
census is UNCHANGED before/after — no new coverage LANDED — because the two
coding residues (a)+(b) both bottom out on the SAME missing infrastructure
(per-step delivery of a materialized value into a compiled-replay slot without
inflating the harvest), and (b) was ATTEMPTED and reverted on the non-negotiable
ledger gate. The value of this pass is the SHARPENED verdicts + the bounded
acceptable set below, and the two "assess" residues (c)+(d) resolved as
acceptable-by-design.

**AFTER census (== BEFORE — recurring bails only):**
| workload | recurring-bail fp | nodes | reaches | classes |
|---|---|---:|---:|---|
| distil-train | 0x466ad4a | 479 | 5 | `row-program[scalar-steptemp-input]` |
| distil-train | 0xc0b32ce5 | 296 | 3 | `data-source:full` |
| gpt2-decode | 0xeee0065e | 339 | 4 | `data-source:bernoulli, fused[no-storage], op:max` |
| 124M-diloco | 0xea2ddb2d | 925 | 5 | `row-program[scalar-steptemp-input]` |
| 124M-diloco | 0x36e345f5 | 566 | 3 | `data-source:full` |

Gates on the (reverted-clean) tree: build ✓; `test:gates` **6/6**;
`t-ledger-attack-probe` (CHECKPOINT=0) reachDrift/totalDrift **0/0**;
`t-stream-generate` **PASS** (3 plans FULLY GENERATED 0 diverged, 1 partial with
the deliberate `row-program[scalar-steptemp-input]×1` bail); `parity-fullstack-tl`
compiled-vs-lowered max |Δloss| **8.6e-6 / 30 steps**; full suite green.

### (a) row-program[scalar-steptemp-input] — DEFERRED (captured-state, cross-seam)
The bail is `ref.kind==="materialized" && shape.length===0` for a row-program
external input (the clip/scaler scale fused as a `mul` preamble,
`sum(mul(loss, scale))`; `stream-generate.ts:1179`). Two sub-cases, and BOTH
need infra we don't have:
- **destroyed 0-d input** → the executor takes `executeRowProgram`'s SEQUENTIAL
  FALLBACK (`segment-executors.ts:439`, `isDestroyed` → run mul+sum as separate
  dispatches, NOT the fused kernel). To cover this the generator would emit the
  same two dispatches — but the input's storage is GONE at build time, so a
  compiled replay has no data to bind. This is the captured-state class (like
  `mean`'s invCount): it needs scalar-table-style per-step delivery of a
  MATERIALIZED 0-d value (not a `kind:"scalar"` ref — the existing scalar table
  `scalar-table.ts` only handles value-carrying scalar refs).
- **live 0-d input** → the executor runs the FUSED kernel; the generator could
  bind the 0-d storage IF it were persistent, but the scaler scale is a per-step
  step-temp (`full([],v)`), so binding the record-time buffer freezes a
  step-varying scale (silent corruption — the exact hazard the bail cites).
The #87 `scalarDress`/pre-replay-writeBuffer mechanism (`step-tape-replay.ts`)
is the CLOSEST existing delivery seam (fresh per-step scalar → fixed consumer),
but it is wired for the step-TAPE replay path and keyed on an in-place scatter
consumer; generalizing it to a row-program's fused-kernel 0-d INPUT slot is the
cross-seam work this residue needs. Not landed — silent-corruption stakes, no
clean seam yet. **Bail stays; plan stays recorded.**

### (b) data-source creation ops (full/arange/rand/randn/bernoulli) — ATTEMPTED, REVERTED on the ledger gate
Implemented the doc's named requirement: generate `ALLOC(planner slot)+WRITE`
for the f32 creation ops (`generateDataSource`) + a **data-source-INTO-SLOT**
TAG_WRITE executor path (`compiled-plan.ts`): when a preceding planner ALLOC
populated `slots[cmd.slot]`, produce the op's data into that planner-owned buffer
(full/arange via CPU `writeBuffer`; rand/randn/bernoulli via kernel→`copyBufferToBuffer`
+ fence-gated temp destroy) instead of the legacy re-execute-into-a-FRESH-pool-buffer.
- **The into-slot path is correct and fires** (verified via `TORCHLETTE_DEBUG_WRITES`:
  build-time `full` nodes route into their planner slot, no fresh alloc).
- **BUT the ledger gate FAILED: reachDrift/totalDrift = 9 (> tol 8).** Root cause
  is NOT the into-slot path — it is that making the creation ops generatable
  flips additional (warmup) plans to FULLY-COVERED → they cut over → and
  build-from-IR's documented OVER-HARVEST (harvest the full action-output set,
  which is fundamental and irreducible — see "THE OVER-HARVEST IS FUNDAMENTAL")
  adds ~9 exclusive result entries. The trajectory is a one-time +9 STEP at the
  warmup-cutover boundary, then flat (426→435→flat) — a harvest/memory tradeoff,
  not a growing UAF, but it still trips the non-negotiable ledger tolerance.
  Critically, the per-step scaler `full` (the recurring `data-source:full` bail)
  does NOT route through the generated ALLOC+WRITE — it stays recorded — so this
  change does NOT even close the recurring training bail; it only newly-covers
  warmup plans and pays the over-harvest. Net: no recurring-bail closure, ledger
  regression. **Reverted.** The prior pass's "separate executor change, out of
  scope" verdict STANDS, now with this sharper reason: covering data-source ops
  via generation cannot pass the ledger gate because it entangles with the
  fundamental build-from-IR over-harvest; a clean close needs the recurring `full`
  itself to route into a planner slot WITHOUT flipping extra plans to cutover —
  which the generate-more-plans mechanism inherently does the opposite of.

### (c) decode strided-view + op:max — ACCEPTABLE-BY-DESIGN (inference-only, DEFER)
Re-measured post-#71: the decode recurring bail (0xeee0065e) is STILL
`fused[no-storage], op:max` (+ `data-source:bernoulli`, a census-harness
artifact — the harness never `.eval()`s so `Dropout` fires bernoulli;
`nn/module.ts` defaults `trainingMode=true`). Findings:
- `fused[no-storage]` still fires and #71 did NOT touch it. Origin: the causal
  mask `causalBias.narrow(2,…).narrow(3,…)` (`examples/gpt2/model.ts:188`) — a
  released double-narrow STRIDED view — feeds a fused `add`, and `generateFused`
  bails `no-storage` on a released `VIEW_OPS` input (`stream-generate.ts:3064`).
  #71's `cachedStridedInputs` volatile-offset rescue lives ONLY in the
  sequential/direct-elementwise path (`generateSequential`), never in
  `generateFused` (zero refs there) — so covering it needs a phase-3-style
  strided-input capture GENERALIZED to the fused path.
- `op:max` is softmax's stability max (a DIM reduction). Routing is mechanically
  trivial (`planFullReductionDispatch`/`planDimReductionDispatch` already take a
  ReduceOp; `reductions.ts:376,432`) — but `generateFullReduction`/`generateDimReduction`
  hardcode `"sum"`. HOWEVER `op:max` and `fused[no-storage]` co-occur in the SAME
  plan, so covering max ALONE cannot flip the plan to fully-covered — zero
  standalone value here.
Decode is inference-only (`api.noGrad`), not a training gate. **Verdict:
acceptable-by-design; leave decode recorded. If desired later, route max/min
(cheap) AND add a fused-path strided capture — both needed to actually close it.**

### (d) first-exec config-missing transients — ACCEPTABLE-BY-DESIGN, does NOT block deletion
The `*[config-missing]` / `matmul-epilogue[no-config]` / `fused[no-input-pattern]`
classes are a pure cold-start ORDERING property with NO recording dependency.
The generator's `plan()` returns a config buffer only if a prior DISPATCH
populated the cache (`tile-dispatch.ts:337`); that cache entry is created
UNCONDITIONALLY at first dispatch (`tile-dispatch.ts:237`, independent of
recording state). So on a template's FIRST execution build-from-IR bails
(cache empty) and falls through to a LOWERED-WITHOUT-RECORD path
(`shouldCompile` is false when the arena is empty — "populated from prior
execution", `executor.ts:2254`); that lowered pass populates the config cache as
a side effect; the SECOND execution builds from IR and returns with no recording.
So config-missing transients are ALREADY record-free today. **The precise
deletion-readiness answer: a first execution ALWAYS lowers, and it need NOT
record — it never did for this class. Deleting the recorded build changes
nothing here.** The only thing the deletion pass must preserve is that a
first-exec build-from-IR bail keeps falling through to lowered-without-record
(it does, via `shouldCompile===false` on the empty arena). No config-buffer
pre-derivation is required (that is the `BUILD_FROM_IR` build-without-execution
ambition, orthogonal to recorded-build deletion).

### Deletion-readiness verdict (updated)
The recorded build is **NOT deletable yet**, and the acceptable set is now
precisely bounded to TWO recurring correctness bails on the TRAINING path plus
the inference-only decode plan:
- **(a) `row-program[scalar-steptemp-input]`** (distil + 124M) — needs
  captured-state per-step delivery of a MATERIALIZED 0-d value into a fused
  row-program input slot. The scalar table + #87 scalarDress are the nearest
  seams; neither yet reaches a fused-kernel input slot. **BLOCKS deletion.**
- **(b) `data-source:full`** (distil + 124M) — the recurring scaler `full` stays
  recorded; generating creation ops entangles with the fundamental build-from-IR
  over-harvest → ledger regression, so generation-based coverage cannot close it.
  A close needs the recurring `full` to route into a planner slot WITHOUT
  flipping extra plans to cutover (a targeted executor write-into-slot for the
  ALREADY-cutover plan, not a generate-more-plans path). **BLOCKS deletion.**
- **(c) decode `fused[no-storage]` + `op:max`** — inference-only; acceptable
  recorded. Does not block the TRAINING-path sunset but would need coverage for a
  full no-record decode.
- **(d) config-missing transients** — acceptable-by-design, record-free already.
  **Does NOT block deletion.**

When (a) and (b) close (both need the same "materialized value → compiled-replay
slot, per-step, no over-harvest" primitive — a phase-3-style captured-state
delivery, NOT a generate-more-plans extension), no recurring TRAINING plan needs
the recorded fallback and the deletion harvest (A4's `record*` refs +
params-sequence cache + `buildCompiledPlan` + `BUILD_FROM_IR=0` opt-out) becomes
safe. The deletion pass must ALSO sunset the census flag `TORCHLETTE_COVERAGE_CENSUS`
+ `dumpCoverageCensus`/`resetCoverageCensus` + `t-coverage-census.ts` (tied to the
recorded build per `executor.ts:568`). The `TORCHLETTE_GENERATED_PLAN` sunset the
original pass named is ALREADY executed (dead since 2026-07-08). No `src/`
deletion is warranted in THIS pass — the two coding residues remain infrastructure-
gated (verified: the one attempt that touched `src/` regressed the ledger and was
reverted).

## The CAPTURE PRIMITIVE pass (2026-07-12): (a)+(b) are ONE value chain; half-built, half-blocked

Built the primitive's constant-value half (`constFill`), measured it end-to-end
on all three census workloads + the four gates, and pinned down WHY residues
(a) and (b) cannot close independently. The load-bearing new fact: **(a) and (b)
are the SAME gradient-clip value chain across a plan boundary, and closing (b)
alone creates the exact +9 cross-plan hold that only (a) can drain** — so the
primitive genuinely has two halves that must land TOGETHER.

### What the residues actually are (traced, not assumed)
Instrumenting the census (`TORCHLETTE_DEBUG_STEPTEMP`) resolved both recurring
TRAINING bails to the `clipGradNorm_` chain (`src/nn/clip-grad.ts`):
- **(b) `data-source:full`** in the 296-node plan (distil fp `0xc0b32ce5`, 124M
  `0x36e345f5`) is a single node `full([], 1.0)` — the `minimum(div(maxNorm,
  norm+eps), 1.0)` CEILING (`nn/functional.ts:19` promotes the scalar `1.0` to a
  0-d `full`). It is a **compile-time HOST CONSTANT** (fillValue=1, invariant
  across steps), NOT a GPU-computed value. The earlier "recurring scaler `full`"
  framing was imprecise: it is the clip ceiling, and it is constant.
- **(a) `row-program[scalar-steptemp-input]`** in the 479-node plan (distil
  `0x466ad4a`, 124M `0xea2ddb2d`) is the `mul(g, clipCoef)` fused into a reduction
  row-program, with `clipCoef` (the 0-d `minimum` OUTPUT — a per-step GPU value)
  as a materialized input. `clipCoef` is produced in plan-(b) and consumed in
  plan-(a): a **cross-plan 0-d GPU value**, `stamp=none` (plan-(b) bailed, so its
  output was never harvested/stamped), record-time storage swept between steps
  (`destroyed=false` on the build reach, `destroyed=true` on later reaches — the
  exact live/destroyed split `executeRowProgram`'s fallback branches on).

So the chain is: `full([], 1.0)` → `minimum` = `clipCoef` (plan b) → cross-plan →
`mul(g, clipCoef)` row-program (plan a). Two residues, one clip.

### The primitive, constant half: `constFill` (BUILT, PROVEN, then REVERTED)
`SlotSource { kind: "constFill"; elements; fillValue; cachedBuffer? }` — a
compile-time-constant `full([...], const)` as a per-replay INPUT slot backed by a
PLAN-OWNED FIXED buffer (created once at first replay like `params`, pre-filled
with the constant via `queue.writeBuffer`, reused byte-identical every replay,
pinned, destroyed fence-gated at teardown). NO alloc/write command is emitted:
the buffer is born at build and sits OUTSIDE the arena AND the harvest ledger
(neither a pool acquire nor a `NodeResult`). That is the STRUCTURAL reason it does
not repeat the reverted `generateDataSource(full)` ALLOC+WRITE over-harvest — that
path fed a fresh pool/arena buffer AND got harvested; a `constFill` slot is
neither. `generateDataSource` emits it for `full` with a finite host `fillValue`;
gated `coverConstFill` (recurring templates only, reach≥2) so transient warmup
plans stay lowered. **Measured:** census `data-source:full` GONE on both training
workloads (`covered-and-compiled` 3→6); `t-stream-generate` PASS (0 diverged, the
constFill segments verify byte-identical against the recording); `test:gates`
6/6; `parity-fullstack-tl` compiled-vs-lowered **1.05e-5 / 30 steps** (noise
floor). The primitive is SOUND.

### Why (b) alone still trips the NON-NEGOTIABLE ledger (the +9)
`t-ledger-attack-probe` (default STEPS=24): a deterministic **one-time step
426→435 at ~step 13** (reachDrift=totalDrift=9 > tol 8), then perfectly flat
through step 47. At STEPS=30/48 the settle falls OUTSIDE the `[STEPS/2..]` late
window and the probe **PASSES** — proving it is a one-time template settle, NOT a
monotone leak. But 24-step 0/0 is non-negotiable, and totalDrift=9 means **9
genuinely-NEW persistent storages** appear and stay (livediff s12→s14: net-+9
inside a 248-storage convergence-rebuild churn of param/grad views).

Root cause: covering (b) makes plan-(b) COMPILE, so its `clipCoef` output is now
a HARVESTED compiled registry result held across steps — and the still-lowered
plan-(a) row-program consumes `clipCoef` via a materialized ref, keeping the whole
clip chain alive as compiled results. On the pre-(b) tree plan-(b) ran lowered and
`clipCoef` was freed each step by actual-refcount pruning. **The +9 is exactly the
un-drained cross-plan `clipCoef` hold.** It drains only when plan-(a) ALSO compiles
and claims `clipCoef` as a stage-3-B released-external (the last-reader overlay
release) — i.e., only when (a) closes. Closing (b) without (a) is provably a
half-primitive that regresses the ledger; reverted (ledger back to 0/0, verified).

### Why (a) can't close: the missing cross-seam rebind (confirmed absent)
Covering (a) means emitting the fused row-program binding `clipCoef` as an
external slot. At compiled-plan replay `executeCompiledPlan` populates an
`external` slot by `getInputStorage(ref)` → `gpuBuffer(...)`, and for a
MATERIALIZED ref that returns the FROZEN record-time `ref.storage`
(`compiled-plan.ts:1368-1385`, `op-dispatch.ts:146-197`) — which for `clipCoef`
is a swept per-step temp (stale bytes / `[lifetime]` throw). **Verified (targeted
audit): compiled-plan replay has ZERO stamp→live-registry-entry rebind for
materialized cross-plan externals.** The only replay-time rebind of a materialized
ref to its owner's CURRENT storage is the step-TAPE path (`sk.rebinds`,
`step-tape-replay.ts:852-920`) — and even there it is OWNER-based and SCALAR-scoped
(`numel>1 → continue`), not a stamp/registry resolution. The stamp/`resultEntryFor`/
`plannerRegistry` machinery that DOES touch compiled externals runs at BUILD time
and does the inverse (marks a producer entry overlay-releasable, `executor.ts:2089-2122`)
or poisons overlaid storages at replay (`compiled-plan.ts:2029-2047`). So closing
(a) requires BUILDING a new compiled-replay rebind: resolve the row-program's 0-d
materialized external via its stamp to plan-(b)'s live `clipCoef` registry-entry
buffer each replay (which ALSO requires plan-(b)'s `clipCoef` to be stamped — a
consequence of (b) compiling, so the two are mutually enabling). Silent-corruption
stakes (a frozen or mis-resolved clip coefficient trains with the wrong gradient
scale); no clean seam exists yet.

### Deletion-readiness RE-VERDICT (2026-07-12): still NOT READY — but the primitive is now half-specified
The census (before == after on the reverted tree) still shows the same two
recurring TRAINING bails; `constFill` is proven to close (b)'s COVERAGE but is
inseparable from (a) at the ledger. **NOT DELETABLE.** The remaining work is now
precisely scoped as ONE two-part mechanism on the clip value chain:
1. **`constFill`** (constant half — this pass, ready to re-land) delivers
   `full([], 1.0)` into a plan-owned fixed slot with no over-harvest. Correct in
   isolation.
2. **compiled-replay stamp-rebind** (GPU-value half — NOT built) resolves the
   row-program's 0-d materialized external to plan-(b)'s live `clipCoef` entry
   each replay (the `sk.rebinds` idea lifted from step-tape into `executeCompiledPlan`,
   generalized past scalars, keyed on the stamp/registry instead of the owner).
   This is the piece that (i) covers (a) and (ii) DRAINS (b)'s +9 by letting
   plan-(a) claim `clipCoef` as a released-external. Both land together or neither
   passes the ledger.

**Deletion-pass spec (unchanged, restated precisely for when both halves land):**
once the census shows zero recurring TRAINING bails AND `t-ledger-attack-probe` is
0/0 with both halves, the deletion removes: the recorded-build harvest (A4's
`record*` refs + the params-sequence cache + `buildCompiledPlan` + the
`TORCHLETTE_BUILD_FROM_IR=0` opt-out, ~800 SLOC); the census surface
(`TORCHLETTE_COVERAGE_CENSUS` + `dumpCoverageCensus`/`resetCoverageCensus` +
`tools/t-coverage-census.ts`, tied to the recorded build at `executor.ts:568`);
and re-bases the verify gates that pin `BUILD_FROM_IR=0` as their reference
(`compiled-plan-parity.spec.ts` determinism gate) onto the generated build.
Residue (c) (decode `fused[no-storage]` + `op:max`) stays acceptable-by-design
(inference-only, `api.noGrad`); residue (d) (config-missing transients) stays
record-free already. Neither blocks the TRAINING-path sunset.


## Task #96 LANDED (2026-07-13): the clip-chain capture — both halves in, the audit corrected

Both halves of the CAPTURE PRIMITIVE are in on `task96-clip-chain-capture`:
`constFill` re-landed per the spec above, and the "compiled-replay stamp-rebind"
landed in a SHARPER form than sketched, because the targeted audit's central
claim was falsified by instrumentation.

### The audit correction (load-bearing)
**The compiled-replay external binding was never frozen.** At replay,
`executeCompiledPlan` receives THIS step's `planNodes` (template hits re-apply
rewrites to fresh nodes), so `planNodes[src.planNodeIndex].inputs[src.inputIndex]`
re-resolves a materialized cross-plan ref to the CURRENT step's storage every
replay — verified empirically (`TORCHLETTE_DEBUG_REBIND` trace: the bound
storage id advances monotonically per replay, destroyed=false). What IS frozen
is the **row-program ACTION's `inputRefs` snapshot**, captured at lowering and
reused verbatim on every template hit (`m:` storage-keyed, same storage id at
every build reach, stamp=none — the pre-plan-b-compile handle). Every OTHER
action resolves refs through `planNodes`; the row-program action was the lone
snapshot, and both the bail decision AND the slot lookup ran on it.

### The mechanism as landed
1. **Consumer provenance** (`row-program-detect.ts` → `RowProgramMatch.
   inputRefConsumers` → `lowered-plan.ts` `inputRefConsumerPositions`): each
   external inputRef records WHICH covered node's input slot it was captured
   from. `generateRowProgram` resolves the CURRENT ref as
   `planNodes[pos].inputs[inputIndex]` — single source (the node graph), never
   the snapshot. This is the "rebind": the existing fresh-ref channel extended
   to the one action that bypassed it.
2. **Stamp-gated fuse eligibility** (`GenerateStreamOptions.
   fuseStampedScalarExternals`, BUILD path only): the 0-d-materialized bail
   lifts iff the FRESH ref's storage is STAMPED — i.e. the producer plan is
   compiled and re-harvests the value into node-visible results every step, so
   the external slot's per-replay resolution always finds this step's value.
   Unstamped (producer lowered → swept pool temp) keeps the sequential-fallback
   bail. The VERIFY path (`TORCHLETTE_STREAM_GENERATE=1`) passes false and
   keeps the bail — the recording it diffs against used the sequential
   fallback, so the fused kernel there would be a spurious segment DIVERGE.
3. **`constFill`** re-landed exactly per spec (plan-owned fixed buffer, outside
   arena + harvest ledger; `coverConstFill` gated on build-reach ≥2 via the
   executor's `buildReaches` so one-shot warmup variants stay lowered; the
   verify path's flat-count check reconciles the per-constFill command delta).

**Why it cannot bind wrong (the ownership argument):** (i) the consumer's own
template-fp match fixes the input ref's semantic role; (ii) resolution is
per-replay through the current graph — fresh by construction; (iii) a
resolution that is not materialized/live fails LOUDLY (`getInputStorage`
[lifetime] guards + `guardMiss` clean recovery), never silently stale; (iv) the
stamp gate at build ensures the fused kernel only exists where the producer
re-materializes the value each step. A recorded producer-stamp EQUALITY
assertion at the replay seam was built and REVERTED: the producer template
legitimately differs across warmup/steady graph variants while the value's
role is unchanged (same-ni different-fp false positives on grads), and a
mid-step RecoverableGuardMiss recovery is unsound from non-initial plans
("Input not ready" on released intermediates). Do not re-add it.

### Measurements (V100 sivri, 2026-07-13)
- **Census: BOTH recurring TRAINING bails GONE** on distil-train AND
  124M-diloco (`covered-and-compiled` 3→6 each, RECURRING-BAIL=0). gpt2-decode
  keeps residue (c) (bernoulli/fused[no-storage]/op:max — acceptable-by-design).
- **Ledger (`t-ledger-attack-probe`)**: default STEPS=24 **PASS**, late-window
  [431..435] reachDrift=totalDrift=4; STEPS=48 **0/0 flat** ([435..435] over
  steps 24..47). The steady state is +9 handles over the 426 baseline —
  livediff (s8→s22): ±260 view-handle churn at the convergence rebuilds
  netting +9 **view handles** (grad/param views held per-replay by the two
  newly-compiled clip plans' harvests, prior set released at next harvest —
  byte-free: views alias pooled buffers). This CORRECTS the prior pass's drain
  theory: the +9 was never a clipCoef VALUE hold that a claim could drain; it
  is the by-design per-replay view-handle hold of two more plans being
  compiled, settling once (by step 16) and perfectly flat after.
- **Failing-first gate** (`test/stale-external-rebind.spec.ts`, in-suite):
  pre-fix tree fails on exactly the two clip classes (row-program[scalar-
  steptemp-input] ×5 reaches + data-source:full ×3); post-fix passes (clip
  classes gone, compiled==lowered over 20 steps crossing the rebuilds, no
  [lifetime]).
- `t-stream-generate` PASS (0 diverged); parity-fullstack compiled-vs-lowered
  and the remaining wall recorded in the task #96 report.

### Deletion-readiness VERDICT (2026-07-13): READY (training path), one named caveat
The census shows zero recurring TRAINING bails and every gate holds. The
deletion-pass precondition "ledger 0/0" is met at STEPS=48 (flat steady state);
at the default 24-step window the one-time convergence settle leaves drift 4/4
(within the probe's own LEAK_TOL=8, probe PASSES — attributed byte-free view
handles, above). If the deletion pass insists on literal 0/0-at-24, the settle
must complete before step 12, which is an observed-liveness convergence-timing
question, not a coverage one. **Deletion-pass spec (restated, unchanged):**
remove the recorded-build harvest (A4 `record*` refs + params-sequence cache +
`buildCompiledPlan` + the `TORCHLETTE_BUILD_FROM_IR=0` opt-out, ~800 SLOC); the
census surface (`TORCHLETTE_COVERAGE_CENSUS` + dump/reset + the census tool,
tied to `executor.ts`); re-base the verify gates that pin `BUILD_FROM_IR=0`
(`compiled-plan-parity.spec.ts` determinism gate) onto the generated build.
Residues (c)/(d) unchanged and non-blocking. The deletion was NOT performed in
this pass (out of scope by design).

## Task #43 recorded-build DELETION PASS (2026-07-14): (b)+(c) LANDED, (a) BLOCKED by an observed-liveness coupling

The authorized deletion pass ran the three staged commits. **Two landed green
and stay; the third (the harvest itself) is BLOCKED by a newly-surfaced
correctness coupling — the recorded build is load-bearing for
observed-liveness (stage-3 B) correctness on cross-plan consumers that run
lowered.** No harvest deletion was forced (campaign STOP rule).

### LANDED (green, committed)
- **(c) verify-gate re-basing** (`task #43 (c)`): the determinism gate
  (`t-stream-determinism.ts` + its `compiled-plan-parity.spec.ts` arm) is
  re-based off `BUILD_FROM_IR=0` onto the GENERATED build under the default
  flag state (build a template from IR twice, diff the LABEL-MATCHED
  intersection of compiled streams — build-from-IR coverage is per-plan gated
  so the SET can differ; determinism = every plan compiled on both passes is
  byte-identical). The observed-liveness gate-2 reference (`BUILD_FROM_IR=0`
  recorded cutover) is re-based onto the LOWERED reference
  (`TORCHLETTE_COMPILED_PLAN=0`). Both verified green WITH the recorded build
  still present (sound re-base, not deletion-forced).
- **(b) census sunset** (`task #43 (b)`): deleted `coverageCensus`,
  `recordCoverageCensus`, `dumpCoverageCensus`/`resetCoverageCensus`, the
  census-skip instrumentation, and `tools/t-coverage-census.ts`; retired the
  `TORCHLETTE_COVERAGE_CENSUS` flag (envFlags 69→68). KEPT `buildReaches`
  (drives `coverConstFill`, NOT census-gated). The task #96 gate
  (`stale-external-rebind.spec.ts`) assertion-(2) "clip chain compiled, not
  vacuous" is re-based off the census onto the executor's existing
  `[compiled] ...BUILD-FROM-IR fp=...` debug log (TORCHLETTE_DEBUG_COMPILED) —
  no new mechanism.

### (a) BLOCKED — the harvest deletion is correct but exposes an observed-liveness over-prune
The recorded-build harvest was fully removed and BUILDS (−~1132 SLOC:
`buildCompiledPlan`, `startCompilationRecording`/`stop`, all `record*` bodies
+ their call sites, the `if (compilationRecording)` action-loop + finally
blocks, the `STREAM_GENERATE` verify apparatus, the `BUILD_FROM_IR=0` opt-out
clause; `recordedCopyBufferToBuffer`'s real `enc.copyBufferToBuffer` preserved;
`createParamsBuffer`/`releaseParamsBuffer`/`paramsSequenceSet` preserved,
`if(compiling)` tails stripped). But it FAILS RUNTIME on real training:
```
[lifetime] reading step-globally RELEASED storage id=… (shape=[4,128,128]).
  Its registry entry was overlaid by the last observed reader's temps
  (stage-3 B); … invisible to observation — report it.
```
Deterministic on `t-compiled-parity-probe` (STEPS=20) AND `t-ledger-attack-probe`
(real GPT-2 trainer). Root-caused:
- **The over-prune is real and comes from OBSERVED-LIVENESS (stage-3 B), not
  the harvest deletion per se.** Disabling observed-liveness
  (`setObservedLivenessEnabled(false)`) makes the deleted tree byte-correct
  (baseline losses). So the deletion is mechanically clean; it removes a
  correctness NET.
- **The recorded build was that net.** `guardMiss` recovery
  (`RecoverableGuardMiss` → evict + re-collect lowered) exists ONLY at the
  COMPILED-plan external-slot seam (`compiled-plan.ts` `executeCompiledPlan`,
  ~line 1416). The LOWERED path's `getInputStorage` (`op-dispatch.ts:161`) has
  NO such recovery — a `releasedOverlay` materialized read throws hard. At
  baseline the cross-plan consumer of an over-pruned producer ran via the
  RECORDED compiled replay → hit `guardMiss` → recovered. Post-deletion that
  consumer falls to lowered-without-record → reads the overlaid buffer → hard
  throw. So observed-liveness's stage-3 B pruning is UNSOUND standalone for
  cross-plan producers whose consumer does not itself compile; the recorded
  build's `guardMiss` seam silently absorbed it. This coupling was never
  exercised by prior passes (they measured coverage/ledger, not the deletion's
  effect on the pruning's fallback).

### What this needs (out of THIS pass's scope — deletions + assertions only)
A resolution requires NET-NEW code, so it is reported, not built:
1. extend the `guardMiss` recovery to the lowered `getInputStorage`
   `releasedOverlay` seam (a second recovery site — mechanism), OR
2. make observed-liveness stage-3 B not over-prune a cross-plan producer whose
   consumer may run lowered (convergence/needed-set logic — mechanism), OR
3. DISABLE observed-liveness pruning when the recorded build is deleted (a
   DELETION, and its stated rationale — "prune the +34% over-harvest toward
   the recorded cutover survivor set" — evaporates with the recorded cutover;
   but it is a real memory regression + a scope-expansion decision for the
   coordinator, not a recorded-build-deletion sub-task).

**VERDICT: the recorded build is NOT deletable until observed-liveness
stage-3 B is made sound without the compiled-only `guardMiss` fallback (option
1/2) or is retired (option 3). Stages (b)+(c) landed; (a) reverted clean.**

## Task #97 — LIVENESS UNIFICATION: the derived oracle at the overlay-release seam (2026-07-14)

This resolves the #43(a) blocker by **option 2 done as DERIVATION** (the campaign
ruling, the third application of the maneuver that gave #70 derived ownership and
the planner replacing pin-the-recorded-buffers): the cross-plan liveness stage-3 B
was GUESSING becomes a single-source DERIVED fact that governs the overlay-release
on all paths, so the recorded build's compiled-only `guardMiss` net is no longer
load-bearing for soundness.

### Stage-1 interview — DESIGN-OR-BUG verdict: analytically WRONG (not aggressive-by-design)

The overlay-release (`_claimedExternal` → a producer's registry entry handed to a
consumer's temps, `releasedOverlay` set) is decided at the claim seam
(`executor.ts` ~1967) by `releasableLastReader` — the EMPIRICAL last-reader
observation. That observation is fed by `observeConsumed`, which is called at
**exactly one site**: the compiled external-slot bind (`compiled-plan.ts` ~1435).
Every other cross-plan read — a consumer running LOWERED, a persistent-slot bind
(`debugCrossPlanPersistentBindings`), a view resolved inline — is **structurally
invisible** to it. So "template C has been the SAME stable last reader for K steps"
can be TRUE while a *different* consumer also reads the producer every step. The
canonical invisible consumer is the **BACKWARD pass re-reading a saved-for-backward
forward activation**, resolved lowered through `getInputStorage`. The overlay
clobbers the activation after the forward consumer's read; backward then reads
garbage. This is analytical unsoundness — the observation is INCOMPLETE, not
merely aggressive.

**Planner-scope answer:** the memory planner is PER-PLAN and analytical only for
*intra-plan* liveness; the *cross-plan* consumption fact it needs
(`externalReleases`) is FED to it, derived from `releasableLastReader`. So the
planner does not independently know cross-plan liveness — it trusts observed-
liveness. Derivation at the lowered seam is nonetheless feasible: the value read
again in backward is exactly a **graph-retained** (saved-for-backward) value, and
graph retention (`G(s)>0`, the `_graphRetained` retention clone minted DURING
forward by `saveForBackward`/`_cloneForRetention`) is a single-source fact live in
the storage tracker AT CLAIM TIME.

**Baseline guardMiss counts (the measurable symptom):** on the DEFAULT recorded
build, `t-ledger-attack-probe` (STEPS=24) and `t-compiled-parity-probe` (STEPS=20)
report `cleanMisses=0, claimMisses=0` — the guardMiss net is NOT exercised at
baseline; the over-prune is MASKED because the second consumer also compiles and
its recorded replay reads a still-correct buffer. The symptom only MANIFESTS when
the recorded build is deleted (the harvest-deletion diff): the second consumer
falls to lowered → `[lifetime] reading step-globally RELEASED storage (stage-3 B)`
throws deterministically on both probes (shapes `[4,128,128]` / `[2,32,64]`). The
throwing storage is, in every case, exactly the one that is `graphHeldAt=true` at
claim time; the safe claims (grads after the optimizer, casts after their sole
read) are `graphHeldAt=false`.

### Stage-2 — the derived oracle

`storageTracker.graphHeldAt(storageId)` (a first-class derived query in the same
single-source form as `viewBaseIsLive` / `_derived`'s `graphHeld` axis): true iff
∃ a live `_graphRetained` owner (G(s)>0), or the flattened view-base chain reaches
one (backward reads the base through the view). Gen-INDEPENDENT, matching
`_derived`. The claim seam gains ONE gate: **`if (graphHeldAt(storage.id))
continue`** — a saved-for-backward producer is never overlay-released.

**Why it cannot over- OR under-prune (the memory argument = the UAF argument):**
- *No under-protection (soundness):* a graph-held value has a GUARANTEED future
  backward reader (the retention clone is live precisely until `cleanupAutogradGraph`
  after backward). Declining its overlay makes the over-prune UNCONSTRUCTIBLE, so
  the lowered `getInputStorage` `releasedOverlay` throw can no longer fire on it and
  the compiled-only guardMiss net stops being load-bearing.
- *No over-protection (memory):* a graph-held value is LIVE until backward reads it
  REGARDLESS of the overlay — the retention clone keeps its rc>0. Overlaying it was
  therefore never a real memory win. Declining the claim frees nothing that was
  actually reclaimable. The genuinely-boundary-dead values (`graphHeldAt=false`)
  still claim and overlay, so the +34%-over-harvest reduction is preserved. A/B
  profile (distil@512, same tree/device, pre-vs-post) is BYTE-IDENTICAL: peak
  5397 MB, current 5233 MB, 849 live buffers, ~40 ms/step — the derived oracle
  BEATS the no-pruning scenario at parity, not by accepting a regression.

Gate: `test/derived-liveness-oracle.spec.ts` (the derivation the claim seam relies
on) + the harvest-deleted parity/gate-2 trajectory (stage 3). guardMiss recovery
**DEMOTED to a should-never-fire loud assertion (task #98 phase 5, 2026-07-15)** — see
the stage-4 update in the STOP block below.

### Stages 3–4 — BLOCKED: a SECOND, distinct prune-soundness class the harvest deletion exposes

> **UNBLOCKED (task #98 phase 4, 2026-07-15) by execution-witnessed harvest.** The
> "separate campaign" this STOP named ("a whole-step-graph or execution-witnessed
> harvest") is `docs/step-object-design.md §4`, LANDED. The step-tape recorder observes
> the LOWERED cross-plan read of the checkpoint-recompute `contiguous[512,768]` at
> end-of-step time (AFTER backward + recompute ran — build-WITH-execution, the property
> the recorded build had) and keeps it in the generated harvest via a per-producer K_w=2
> witness set unioned into `prunedHarvest`'s keep set. With the harvest-deletion diff
> applied, distil@512 + selective checkpointing (this exact config) now runs CLEAN where
> it threw `Input not ready: contiguous[512,768]`; medium@512 and the 124M chunked-sum
> class likewise. Gate: `test/witness-harvest.spec.ts` + `tools/t-witness-harvest-matrix.ts`.
> The recorded build's harvest ROLE is not yet deleted (a later product decision, §4.3):
> the full-deletion probe surfaces a distinct `shape=[]` GradScaler-scalar never-witnessed
> class the recorded build still nets. The prune-soundness class named below is resolved;
> stages 3–4 remain gated only on that never-witnessed remainder.

Re-applying the harvest deletion (recorded build removed) with the stage-2 overlay
oracle in place clears the ORIGINAL `releasedOverlay` throw on every config the
#43 pass tested (small probes E=128, gate-2, parity-fullstack — all green,
compiled-vs-lowered ≤ 1e-5). But the profile A/B at **distilgpt2 @ seq-512 with
selective checkpointing** (a config the #43 deletion pass never ran) surfaces a
SECOND, structurally-different failure at step ~5 (pruning-activation):

```
Error: Input not ready: node id=5610 op=contiguous[0] shape=[512,768]
```

This is the PRUNE-THEN-DEMAND twin of the overlay throw, but its producer is NOT
graph-held. Root-caused (deterministic; reproduces with the stage-2 overlay gate
DISABLED too — orthogonal to it):
- The `contiguous[512,768]` is a forward activation inside a **checkpointed MLP
  segment recomputed in backward**. Its producing forward template
  (`0x31292afa`/`0x72ada859`) PRUNES it from the harvest (observation never saw a
  consumer — the backward/recompute reader resolves it LOWERED through
  `getInputStorage`, invisible to `observeConsumed`). The recorded build harvested
  it because it RECORDED the actual dispatch (build-WITH-execution). The generated
  build reconstructs the harvest from IR (build-WITHOUT-execution) and the
  observation prunes it.
- **The derived fact is UNAVAILABLE at the seam.** Measured (`storageTracker.
  graphHeldAt` on the executed result): the contiguous is `graphHeld=FALSE` — it is
  NOT a saved-for-backward clone, so the stage-2 graph-retention oracle does not
  cover it. The other graph-derived source, `computeLivenessOutputIds` (the SAME
  liveness the harvest set is built from), also cannot reach it: at the PRODUCING
  forward plan's build/harvest — which runs during `forward()` — the backward
  consumer DOES NOT EXIST YET (backward's graph is built later, in `backward()`).
  So no whole-step graph derivation is available at the forward-plan build seam.
  A lowered post-execution graph-held observation (`observeGraphHeld`, source "h")
  was built, proven to help the NON-checkpoint big-config case (ledger @ E768/seq256
  passes, releasablePairs→0), and measured NULL against this checkpoint case (the
  contiguous is graphHeld=false); it was reverted (unproven-necessary on the
  recorded build, which handles graph-held via the guardMiss net). A
  `computeLivenessOutputIds`→keepAlways extension was built, measured NULL (can't
  reach the not-yet-existing backward consumer), and reverted.

**This is the campaign's explicit STOP condition** ("derivation genuinely
infeasible at the seam — planner scope structurally can't reach it → STOP and
report; do not fall back to spreading recovery"). Consequences:
- **Stage 3 (harvest deletion) does NOT land.** The recorded build stays as the
  correctness net for the build-WITHOUT-execution prune of checkpoint-recompute
  cross-plan forward reads — exactly the class it was already the net for. The #43
  verdict is refined, not overturned: the recorded build is deletable once the
  GENERATED harvest is sound for build-without-execution cross-plan forward→backward
  reads (a whole-step-graph or execution-witnessed harvest — a separate campaign),
  not merely once the overlay class is derived.
- **Stage 4 (demote guardMiss) — LANDED as task #98 phase 5 (2026-07-15).** This
  STOP's premise ("guardMiss recovery is still load-bearing… the second consumer
  compiles and recovers") was FALSIFIED empirically: a zero-fire soak across the full
  config matrix (default recorded-build path, `STEP_TAPE=record` witness/tape path,
  `STEP_TAPE=1` replay, stream-generate — 20 configs, all build-from-IR-observed) shows
  guardMiss NEVER fires. The recovery is not load-bearing because the recorded build
  HARVESTS the checkpoint-recompute + `shape=[]` scaler-scalar values (they are never
  pruned-then-demanded → the guardMiss seam is never reached), and the overlay class is
  unconstructible via `graphHeldAt` (stage 2). guardMiss was a REDUNDANT secondary net.
  So the clean-recovery is now a loud should-never-fire assertion and the recovery
  machinery is DELETED (`RecoverableGuardMiss`, the `forceAllMerged` re-collect-lowered
  retry, the `forceLowered` threading). Net −28 SLOC. See
  `docs/step-object-design.md §6 Phase 5` for the soak table and the demotion boundary.
  NOTE the prune-CLASS itself is not deleted — the recorded build's harvest still nets
  it; phase 5 removes only the redundant guardMiss recovery, not the recorded-build
  correctness net (its full sunset remains the §4.3 product decision).

**What DID land (the campaign's spine):** the stage-2 derived overlay-release oracle
(`graphHeldAt` at the claim seam) — the ORIGINAL #97 bug (the `releasedOverlay`
throw the #43(a) blocker named) is resolved by derivation, memory-neutral, gated,
full-suite green on the recorded build. The over-prune of the OVERLAY class is now
unconstructible; the recorded build's guardMiss is no longer load-bearing for THAT
class. The remaining prune-soundness class is scoped above for the follow-on.

## Task #43 recorded-build DELETION PASS (2026-07-16): the shape=[] class FIXED; full deletion STILL BLOCKED by the broader `forwardToForce` cross-plan class

> **DEEP-UNIFICATION CROSS-REF (`docs/step-data-dependence-design.md`, 2026-07-16):** this fourth
> blocked attempt is face 1 of a single root shared with #99 R2 and the checkpoint bypass. The
> `forwardToForce` cross-plan class is designed away there by a derived cross-plan EDGE SET (the ~47
> reads, `[E-1]`) whose `graphHeld=false` members are carried as genuine-save edges — superseding the
> `graphHeldAt` heuristic (that doc §5, the STOP's own proposed unblock made principled). EMPIRICAL
> CORRECTION to the "GradScaler inf-skip re-fingerprints" attribution here: `[E-2]` the optimizer
> never host-skips (grads zeroed as data); `[E-3]` the scaler window's fp instability is a VALUE-level
> scale-change op (`witnessVariances=3`) → route-as-data, NOT a structural variant. The sunset is that
> doc's phase D4 (the finiteness argument).

The authorized full-sunset pass root-caused and FIXED the last NAMED blocker — the
`shape=[]` GradScaler-scalar `[lifetime] reading RECLAIMED` — then, on the deleted
tree, found it is the SMALLEST instance of a broader class the witness/generated
harvest cannot serve. **The harvest deletion was NOT performed** (campaign STOP
rule). The reconciled harvest-deletion diff (adapted from the 6-merges-stale
`.claude/harvest-deletion-43a.diff`, builds clean, −1280 lines) is preserved at
`.claude/harvest-deletion-43a-reconciled.diff` for the next attempt.

### The `shape=[]` class — ROOT-CAUSED + FIXED (LANDED)
Deterministic repro (apply the reconciled harvest-deletion diff, then
`VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record
CELL=scaler-inf npx tsx tools/t-witness-harvest-matrix.ts`): PRE-FIX throws
`[lifetime] reading RECLAIMED storage id=… (shape=[])` at step ~6.

Root cause (traced storage id → producer/consumer/reaper): the reclaimed `shape=[]`
storage is the **backward gradient seed `full([], 1.0)`** (`Torchlette._seedGrad`).
Under selective checkpointing, `backward()` force-materializes the seed in a SEPARATE
"forward tensors" plan (`autograd.ts`, `forwardToForce`). That makes the seed a
CROSS-PLAN value: produced in that plan, consumed by the main backward plan. With a
GradScaler the consumer is the extra `mul(seed, scale)` backward node (the derivative
of `scale(loss) = mul(loss, scale)`), which runs in a LATER-forced segment. On the
recorded build the harvest rc-pins every produced result, so the seed survives to
that later read. When the recorded build is retired and the compiled plan is built
from the generated stream, the observed-liveness harvest cannot WITNESS that later
cross-plan read (it is data-dependent — the GradScaler inf-skip re-fingerprints the
plans, so the producer never witnesses two consecutive identical steps), so it PRUNES
the seed's harvested `full` result; its rc then hits 0 and `destroyUnreachable` reaps
it mid-backward (`autograd.ts` post-force `destroyUnreachable`), before the consumer
reads it → the RECLAIMED throw.

Fix (`autograd.ts`, the checkpoint `forwardToForce` set): the grad seed is a LEAF
CONSTANT (`full([],1.0)`, no inputs); do NOT force it in the separate forward-tensors
plan. Leaving it lazy materializes it INSIDE the main backward plan alongside its
consumer (intra-plan), so it is never a prunable cross-plan harvested result.
NULL on the recorded build (the harvest pins the seed either way): `test:gates` 6/6,
`parity-fullstack-tl` compiled-vs-lowered max |Δloss| 8e-6/30, all `t-witness-harvest-
matrix` cells PASS (scaler-inf 6/459 templates/pairs vs the pre-fix 7/460 — the only
delta is the seed no longer being a separately-forced template), full webgpu project
green (run EXCLUSIVELY — `npm run test` runs cpu+webgpu concurrently and the webgpu
project needs GPU-exclusive access; concurrent runs mass-fail on `vkCreateDevice`
device-chain contention, NOT logic). Gate: `test/checkpoint-scaler-seed-lifetime.spec.ts`
(checkpoint + GradScaler backward: finite grads, no lifetime throw, parity vs
checkpoint-only reference).

### The STOP — the broader `forwardToForce` cross-plan class (new, blocks the full deletion)
With the seed fixed, the deleted-tree `scaler-inf` cell advances past the `shape=[]`
throw and reveals that it was the SMALLEST member of a class: the checkpoint backward's
`forwardToForce` plan force-materializes forward ACTIVATIONS as cross-plan values, and
the generated/witness harvest cannot reliably keep them under `scaler-inf`'s
data-dependent inf-skip. Two further instances, both non-`shape=[]`:
- **`[512,50257]` (CE logits) RECLAIMED** — the next reclaim after the seed on the
  no-fix tree (bypassing only the `shape=[]` throw surfaces it at step 6). The seed
  fix's plan-fingerprint shift happens to resolve THIS one too, but that is
  coincidental (a fingerprint side effect, not a principled keep).
- **`[1,512,768]` (a transformer-layer activation, template `0xa444a1bc` node 272,
  oi 0) OVERLAY-RELEASED (stage-3 B)** — recurs every step with the seed fix applied.
  It is `graphHeldAt=FALSE`, so #97's `graphHeldAt` oracle explicitly does NOT protect
  it: #97 assumed non-graph-held ⟹ boundary-dead, but here a non-graph-held cross-plan
  activation IS read after the overlay-release (by the invisible checkpoint/scaler
  cross-plan reader). Training stays finite in warn mode (the overlaid bytes may be
  correct), but under STRICT lifetime it throws — and strict is never weakened.

These are the SAME structural gap the #97/#98 campaign named (a build-WITHOUT-execution
prune of checkpoint-recompute cross-plan forward reads that no build-time analysis can
witness), now shown to extend BEYOND the single `shape=[]` scalar to the full
`forwardToForce` activation set once the scaler's data-dependent inf-skip defeats the
two-consecutive-step witness requirement. Closing it needs NET-NEW mechanism (a
whole-step or execution-witnessed harvest robust to data-dependent re-fingerprinting,
OR extending the overlay-release oracle past `graphHeldAt` to witnessed cross-plan
readers) — out of a deletions-only pass's scope. **VERDICT: the recorded build is NOT
deletable. The `shape=[]` named blocker is closed; the deletion is now gated on the
broader `forwardToForce` cross-plan class under the scaler-inf (checkpoint + GradScaler
+ autocast + inf-skip) workload.** Fourth blocked attempt, one new named class.
