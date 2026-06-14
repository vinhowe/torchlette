# Stage 4: Compile-from-IR with Graph-Liveness Memory Planning

*Design, 2026-06-12. Companion to `architecture-debt.md` (stages table, row 4).
Written at the end of the cycle that landed stages 0–3; every claim about
existing machinery refers to code in-tree at commit ed779c6.*

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
  Shares the 4.3 residual: the generated replay of every plan is now CORRECT
  (grad_bias fix + bail rule), the planner packs identically, and the remaining
  +57 MB is a peripheral pool/size-class artifact (not architectural — see 4.3).
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
