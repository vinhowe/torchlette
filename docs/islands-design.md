# The Islands IR — Record-Time Kernel Partitioning (PROPOSED)

**Status:** DESIGN ONLY (task #78, stage-1/3 protocol). No implementation this
campaign. Reviewed against: the schedule-record spike
(`docs/spike-schedule-record-findings.md`), the kernel-editor validity/
completeness and WGSL⊂CUDA analyses (scratchpad), `docs/architecture-debt.md`
(sin taxonomy + stage plan), `docs/stage4-compile-from-ir.md` (the four
representations), CLAUDE.md's correctness principles and the fusion ledger.

**One-sentence thesis:** make the *dispatch-partition* of a step's semantic
graph a first-class datum — a set of **islands** — so that fusion,
authored-kernel selection, chunking, and horizontal packing are all the SAME
object edited by different *policies*, and two different partitions of one
computation are two states of one object (the editor's move layer) instead of
two ad-hoc, string-matched, order-sensitive code paths.

---

## 0. Why this design is doubly mandated

Two independent roadmaps converge on the same missing representation.

**(1) The framework's own fusion roadmap.** Today's fusion detector
(`src/compiler/fusion-detect.ts`) is **consecutive-only**: `buildCandidateGroups`
walks the plan linearly and only groups adjacent fusible ops, flushing the run
whenever a non-fusible node with a group-internal dependency intervenes
(`fusion-detect.ts:304-320`). The falsification probe (§6) exhibits the exact
failure: `relu → reshape → reshape → exp → neg` partitions as
`(seq …,relu,reshape,reshape) [FUSE exp+neg]` — `relu` is severed from
`exp+neg` by two views, though `[FUSE relu+exp+neg]` is a legal island (same
shape, elementwise, no barrier). Measured on A100 (distil@512, step 5): **3.0%
of nodes fused** (36 / 1199), 197 sequential nodes/step. The
"bypassed-node transparency" attempt to fix this *failed* (CLAUDE.md "what
didn't work": making CSE-bypassed nodes transparent in `buildCandidateGroups`
broke execution ordering — "Input not ready"). The lesson from that failure is
structural: **you cannot patch a linear scanner into a graph-aware grouper by
making some nodes invisible — the partition must be a first-class object the
grouping *policy* writes, decoupled from the linear plan order the executor
needs for scheduling.**

**(2) The kernel editor's move layer.** The schedule-record spike proved that
in-one-kernel decisions (tile sizes, residency, online-softmax on/off) lift
cleanly into a plain-data `ScheduleRecord` at **zero cost** (byte-identical
WGSL). But it hit a hard wall exactly at the *macro* moves — materialize→stream,
fuse N-dispatches→1 — because those "cross kernel and dispatch-count boundaries
the tile-IR treats as fixed structure" (spike findings, Part B). The spike's
own prescription: *"A kernel-shape IR above the tile-IR where
'materialized-S 3-dispatch' and 'fused streamed 1-dispatch' are two states of
one semantic graph."* **That IR is the islands IR.** The tile-IR schedules the
*interior* of one dispatch; the islands IR schedules the *partition into
dispatches*. They compose: an island's interior is a tile-IR program (with its
own `ScheduleRecord`); the islands IR decides island *boundaries*.

The two mandates are the same object at two altitudes. The framework needs it as
a **pass** (fusion as one policy over islands); the editor needs it as a
**substrate** (partitions as editable states). This document designs the object
so both writers share it.

---

## G0 — where partitioning lives today (the decomposition)

### G0(a) — inventory of partitioning-decision sites

Every place that decides "these ops go in one dispatch vs. separate dispatches,"
with SLOC (non-blank, non-comment) of the partitioning-relevant region:

| Group | Sites | SLOC | What partition it bakes |
|---|---|---|---|
| **Authored fused kernels** (pre-frozen islands) | attention (fwd + 3-pass bwd), Adam, LayerNorm, RMSNorm, cross-entropy, matmul-epilogue exec, rope, topk, unscale | ~3,050 (mostly opaque *interior*, not the grouping) | Each hardcodes one dispatch = a fixed op set (the "atom" pattern) |
| **Epilogue / prologue / row-program claiming** | `graph-compiler.ts:99-231` (claim cascade + priority arbiter `:240-248`), `matmul-epilogue.ts` (`detectMatmulEpilogueCore`), `row-program-detect.ts` (`detectRowPrograms`), executor plumbing | ~1,120 | matmul claims following cast/bias/act; reductions claim preamble+multi-reduce; **the true "which ops together" logic** |
| **Elementwise fusion detector** | `fusion-detect.ts` (`buildCandidateGroups`, union-find components, `processCandidate` multi-output/split, `reorderPlanForFusion` Kahn pass, `segmentPlanForExecution`) | ~1,390 (file) | consecutive-fusible → groups → segments |
| **Chunking splits** | `chunked-dispatch.ts`, `reductions.ts` (`planChunkedFullReduction`), `where.ts`, `views.ts` (cast), gather/scatter, unscale, strided-scatter, tile-IR chunked | ~730 | one logical op → N dispatches when a binding exceeds 128 MB (`maxStorageBufferBindingSize`) |
| **Horizontal packing / batch grouping** | `lowered-plan.ts` `horizontalPackKey` + adam-batch loop; `packed-dispatch.ts` `dispatchPackedOptimizer`; `fused.ts` `adamStepBatch`; `reductions.ts` `batchedReduction` | ~355 | N independent same-shape ops → 1 packed dispatch |

**~24 distinct sites, ~5,250 SLOC.** The single most load-bearing decision point
is `analyzeGraph` (`graph-compiler.ts:254`) orchestrating the claim-priority
cascade. Crucially, these sites do **not share a representation**: matmul
claiming writes `epilogueClaimedIds`; the elementwise detector writes
`FusionGroup[]`; chunking writes ad-hoc per-op dispatch loops; packing writes a
`horizontalPackKey`-grouped `adam-batch` action. Four (really 24) writers, no
common object. That is architecture-debt **sin 5** ("order-sensitive,
string-matched plan transformations… a list of optimizations that each got a
bespoke executor extension instead of an IR-to-IR pass") made concrete.

### G0(b) — what the detector's limits cost today (order-of-magnitude)

Measured, A100 (dw-2-1, device 0), distilgpt2@512, step 5 (steady-state):

- **Fused: 3.0%** of nodes (36 / 1199); 10 fusion groups; **197 sequential
  nodes/step.**
- CLAUDE.md's A100 bwd op histogram (the tail of the sequential nodes):
  `reshape:87, matmul:56, transpose:31, permute:24, sum:24, narrowBackward:19,
  add:19`.

Of the ~197 sequential nodes, the **fusible-but-unfused** class is the
elementwise tail broken up by views/bypassed nodes — `add`, `mul`, casts, and
the elementwise ops stranded on the far side of a `reshape`/`transpose`/
`permute` from their fusible neighbors. Order-of-magnitude: **tens of
elementwise dispatches per step** that a graph-aware partition would merge.
Each unfused fusible op is (a) one extra dispatch (kernel launch + bind-group +
pipeline set — the submit-count tax) and (b) a full GMEM round-trip of its
tensor (read inputs, write output, that a fused kernel would keep in
registers). At distil activation sizes (~[8, 512, 768] f16 ≈ 6 MB) a stranded
elementwise op is ~12–18 MB of avoidable traffic; tens of them is
**order ~10²–10³ MB/step of traffic left on the table** plus tens of avoidable
submits. This is the rung-0 cost term (traffic) from the intrinsic-ladder
analysis — the cheapest, most mechanical win, and today's detector cannot reach
it because the fusible run is non-adjacent. The point is not the exact MB; it is
that the limiter is provably the *representation* (partition-as-derived-scan),
not a device constraint (A100 has no V100 10-buffer limit).

### G0(c) — the tape/plan interaction census (the hazard map)

A partition *change* must be keyed at every cache seam or two partitions of one
graph silently alias (the modifier.key-in-bucketKey lesson, #64: partition
identity must key every cache seam, single-sourced). The four representations a
step passes through, and the identity seam on each boundary:

```
rep-1  Lazy IR node graph (LazyIRNode[], plan-builder.ts::buildMergedPlan)
   │        ← node ORDER decided here (Kahn + WAR + checkpoint barriers + affinity)
   │  ┌── SEAM: computePlanFingerprint(plan.nodes)   ← THE MASTER KEY, partition-BLIND today
rep-2  Segments / groups (ExecutionSegment[], fusion-detect.ts::segmentPlanForExecution)
   │        ← THE PARTITION, derived here as a pure function of rep-1
   │  ┌── SEAM: CachedSegmentDesc (executor.ts:292) positional freeze into the template
rep-3  Lowered plan / template (LoweredAction[], lowered-plan.ts)
   │        ← horizontalPackKey (:514), reductionDimKey (:457) — same-op batch keys
   │  ┌── SEAM: slot numbering + lifetime split (compiled-plan.ts:969,1056)
rep-4  Compiled / generated replay (GpuCommand[], compiled-plan.ts / stream-generate.ts)
            ← planMemory over the FLAT stream (memory-planner.ts:153); nodeSlot/nodeIndex identity
```

**Every seam a partition change touches (single-source each):**

1. **`computePlanFingerprint`** (`fusion-detect.ts:1451`) — the master cache key
   (`fusionAnalysisCache.get(fingerprint.primary)`, `executor.ts:3156`). Hashes
   `op / shape / dtype / inputs-as-relative-positions / external-status /
   payload`. **It hashes NOTHING about the partition** — because the partition
   is *derived downstream* from these very nodes. This is the pivot: today the
   partition is a deterministic function of the fingerprinted graph, so it need
   not be keyed. Islands break that: multiple partitions per graph ⇒ the
   fingerprint MUST gain a **partition-identity** field, or two partitions
   collide on one template (probe §6 confirms the collision empirically).
2. **`CachedSegmentDesc` / template `segments`** (`executor.ts:292`, `:3268`) —
   the *positional* freeze of the derived partition into the template. Islands
   replace this with a stored partition (§1).
3. **`horizontalPackKey`** (`lowered-plan.ts:514`) and **`reductionDimKey`**
   (`:457`) — same-op batch grouping. The file comment already forbids
   independent recompute; under islands these become *policies that write island
   boundaries*, single-sourced.
4. **Slot numbering + temporal lifetime split** (`compiled-plan.ts:969,1056`)
   and **`planMemory` size-class registry** (`memory-planner.ts:77,153`) —
   buffer assignment is **per-plan over the flat stream**, not per-island; cross
   -plan sharing rides the epoch-scoped `PlannerRegistry` generation. A
   partition change that reorders allocs renumbers slots — so the planner must
   run *after* the partition is fixed (it already does; islands do not move it).
5. **`nodeSlot` / `nodeSlotExtra` / `producedNodes`** (`stream-generate.ts:107,
   146,171`) and **`nodeIndex`-keyed `diffSegmentsAligned`** (`stream-diff.ts:
   147`) — generated-replay identity + the determinism/verification seam. An
   island's cross-island inputs are "external" here; boundaries must align on
   `nodeIndex`.
6. **`templateFp`-keyed observed-liveness + step-tape skeleton capture**
   (`executor.ts:3506,3528`; `compiled-plan.ts:236-260`) — cross-plan liveness
   identity, all keyed to `fingerprint.primary`. Inherits the partition-key
   extension from seam 1 automatically (they key off the same fingerprint).

**Existing partition barriers to reuse (not reinvent):** `isCheckpointBoundary`
+ `enforceWriteAfterReadOrder`'s boundary edges (`plan-builder.ts:162`), and
`segmentPlanForExecution`'s group/gap logic (`fusion-detect.ts:1106`). These are
the two places segment boundaries are decided today; islands generalize them.

---

## 1. THE OBJECT — the semantic graph + a partition into islands

An **Islands IR** value is a pair `(G, P)`:

- `G` — the step's **semantic graph**: the existing `LazyIRNode[]` in plan order
  (rep-1). Fixed under every partition edit. This is the single source of
  computed-value truth (architecture-debt rule: "make the graph the only
  channel").
- `P` — a **partition** of `G`'s nodes into **islands**. `P` is *first-class
  data* — a stored artifact, not a derived scan.

```ts
interface Island {
  id: IslandId;                 // stable identity — see below
  members: number[];            // node ids, in plan order, that this dispatch covers
  kind:
    | "elementwise"             // fused elementwise recipe (tile-IR generated)
    | "authored";               // opaque pre-merged island (attention/adam/layernorm/…)
  atom?: AuthoredKernelRef;     // for kind:"authored" — which hand kernel; interior OPAQUE
  // island-level schedule handle (the spike's ScheduleRecord lives HERE, for the interior)
  schedule?: ScheduleRecordRef;
}
interface Partition {
  islands: Island[];            // covers every non-view, non-materialized node exactly once
  // views/bypassed nodes may sit OUTSIDE islands (zero-cost, no dispatch) — see below
  boundaryHash: number;         // canonical partition-identity token (§ hazard seam 1)
}
```

**What identifies an island (stable across steps?).** Island identity must be
(a) stable step-to-step for a static graph so templates hit, and (b) distinct
across two different partitions of one graph. The design: **an island's identity
is the canonical set of its members' *structural positions* in `G`** — the same
relative-position encoding `computePlanFingerprint` already uses for inputs
(`idToPos`), NOT raw node ids (which are allocation-order and unstable). The
`Partition.boundaryHash` is an FNV-1a over the sorted island-boundary mask in
plan order. This is exactly the token the probe (§6) validated: **null-stable**
(same partition recomputed → same hash) and **partition-discriminating**
(different partition → different hash). It is then MIXED into
`computePlanFingerprint` so the template cache key becomes
`hash(G) ⊕ boundaryHash(P)`. A static graph with a fixed policy produces a fixed
`P` every step → byte-stable key → template hits (no regression); the editor
producing a *different* `P` gets a *different* template, coexisting as a second
state of one object.

**Views/bypassed nodes stay out of islands.** `reshape`, `transpose`, `permute`,
`narrow` are zero-cost views — they produce no dispatch. Today they *break*
fusible runs because the linear scanner treats them as opaque barriers. In the
islands model they are **not island members**; they are edges relabeled by the
partition (a view between two elementwise nodes of the same logical value simply
does not sever the island — the island's members are the *compute* nodes, and
the view is resolved as a stride/layout decoration on the wire, per the
kernel-editor "layout decoration" gap). This is the principled version of the
failed "bypassed-node transparency" hack: instead of making the node *invisible
to a scanner* (which broke ordering), the node is *not a partition unit* while
remaining a real ordering constraint in `G` (rep-1 order is untouched).

**How `P` composes with the three downstream mechanisms:**

- **build-from-IR (islands → generated plans):** each island lowers to a
  `DispatchPlan` (stage-4's per-action declarative object,
  `stage4-compile-from-ir.md` "Target architecture"). An `elementwise` island →
  a fused tile-IR recipe; an `authored` island → the existing hand-kernel's
  `plan*` function. The `StreamEmitter` concatenates island streams. **Islands
  are the unit the DispatchPlanner plans over** — this is the natural home the
  stage-4 doc was already reaching for ("per-action → DispatchPlan").
- **the tape (partition identity in fingerprints):** `boundaryHash` mixed into
  the master key (seam 1). One-line addition; null-test-clean (§6).
- **the planner (per-island memory):** UNCHANGED in mechanism — `planMemory`
  still runs over the flat `GpuCommand[]` after the partition is fixed
  (memory-planner is partition-agnostic by construction: it sees slots, not
  islands). Islands only change *which* commands exist (fewer, larger for merged
  elementwise); the planner's size-class interval allocation absorbs that for
  free. **This is a load-bearing non-change:** the partition edit does not
  perturb the memory seam, so the riskiest cross-seam coupling (seam 4) is inert.

---

## 2. THE OPERATIONS — `merge` and `split` as the ONLY mutators

Two total operations mutate a partition, and they are **inverse by
construction** (the editor's Law 2 — invertibility):

```ts
merge(P, a: IslandId, b: IslandId): Partition   // fuse two islands into one
split(P, i: IslandId, cut: NodeCut): Partition  // cleave one island at a member boundary
```

`split(merge(P, a, b), ab, cut_recovering_a_b) ≡ P` and
`merge(split(P, i, cut), left, right) ≡ P` up to island-id renaming — the
partition forms a lattice under refinement, and every reachable partition is
connected to the finest partition (every node its own island) and the coarsest
legal partition through these two moves. This gives the editor's "no dead-ends,
connected through the root" property (validity analysis §1.6) at the
*partition* altitude: undo is `split`, redo is `merge`.

**Legality of `merge(a, b)`** — enumerated from the current detector's checks
plus the V100 lesson (these are the T0 well-formedness rules; a merge that
violates any is *refused at the seam*, not silently dropped):

1. **Dataflow convexity.** The union `a ∪ b` must be convex in `G`'s dataflow:
   no node outside `a ∪ b` may lie on a path *between* a node of `a` and a node
   of `b` (else the merged dispatch would have to both precede and follow an
   external op — a cycle). This is `processCandidate`'s split-propagation
   (`fusion-detect.ts:202-226`) restated as a precondition.
2. **Shape compatibility.** For an `elementwise` merge, all members must share
   the fused output shape (or broadcast into it) — `processCandidate`'s
   `shapesEqual(node.shape, primaryShape)` check (`:168`). Multi-output islands
   allow an internal node of the same shape to be an additional output.
3. **Binding-count budget.** `(non-inlinable external inputs) + (primary output)
   + (additional outputs) ≤ maxBuffers` (`processCandidate:177-180`). On V100
   `maxBuffers` was 10 (the historical fusion-rate ceiling); on A100 it is
   larger — so the *same* merge is legal on A100 and illegal on V100. **Legality
   is device-keyed** (validity analysis §2.4): the island IR carries the device
   class, and a merge greyed out on V100 is offered on A100. This is why the
   representation must express the merge even when a policy would refuse it —
   express-to-measure.
4. **Barrier / kind compatibility.** An `authored` island is **atomic** — its
   interior is opaque, so it may not be merged with anything (no `merge` crosses
   an atom boundary; only `merge` of two `elementwise` islands, or an
   `elementwise` island *claimed as an epilogue* into an authored/matmul island
   via the existing epilogue mechanism — which is itself expressed as an island
   annotation, not a separate code path). Checkpoint boundaries
   (`isCheckpointBoundary`) and WAR anti-deps forbid a merge that would reorder
   across them.
5. **Chunking constraint.** If a member's binding would exceed
   `maxStorageBufferBindingSize`, the island must *itself* be a chunk-split
   (a bindable-window sub-partition) — merge is legal but the lowering emits N
   dispatches. Chunking becomes a *degenerate split the lowering applies*, not a
   separate mechanism (subsuming G0(a)'s chunking group).

**Legality of `split(i, cut)`** is trivial: any member boundary is a legal cut
(splitting can only *remove* fusion, never create an illegal state). Split is
always available — which is what makes it a safe universal undo.

---

## 3. THE PASS — today's detector re-expressed as one POLICY among several

The detector stops being *the* mechanism and becomes **one writer** of the
partition. Three writers, one object:

```ts
type PartitionPolicy = (G: Graph, P0: Partition, device: DeviceClass) => Partition;
```

- **`fusionPolicy`** — today's `fusion-detect.ts` logic, re-expressed as: start
  from the finest partition (every compute node its own island), then propose
  `merge`s for adjacent-in-dataflow fusible islands. **The generalization that
  fixes G0(b) for free:** because the policy proposes merges over the *dataflow*
  graph (not the linear plan), it can merge `relu` with `exp+neg` across the
  intervening views — the views are not island members, so they are not
  barriers. The consecutive-only limitation *was* an artifact of the scan, and
  it dissolves when the scan becomes a graph merge-proposal. `reorderPlanForFusion`
  (the Kahn pass) becomes unnecessary for *grouping* (grouping is now graph-
  based) but is retained for *scheduling* (rep-1 order still needs a valid
  topological order for the executor).
- **editor gestures** — a human (or the moves-editor UI) issuing `merge`/`split`
  directly. Same object, same legality checks.
- **autotuner** — a search policy that enumerates merge/split sequences and
  scores them by measured wall time (the `tile-autotune` protocol: warmup-3,
  timed-5, median). Same object.

All three write `Partition` values through `merge`/`split`; the legality checks
(§2) are the shared T0 gate. This is the "detector = one policy; editor = another;
autotuner = a third — same object, three writers" mandate.

**Authored kernels as pre-merged atoms.** The hand-frozen kernels
(attention/adam/layernorm/…) enter the partition as `kind:"authored"` islands
whose members are the nodes they subsume and whose interior is **opaque** (the
atom pattern from the validity analysis, §2.2). A policy may not `merge` into or
`split` an atom; the atom's `plan*` function is its lowering. Coexistence rule:
**an authored island and an elementwise island are the same `Island` type with
different `kind`** — the executor dispatches both through the island→DispatchPlan
lowering, so there is no separate "fused kernel" vs "fusion group" code path
(deleting sin-5's action-vocabulary fork). The epilogue mechanism (matmul claims
a following elementwise chain) is expressed as: the matmul's authored island
*absorbs* an adjacent elementwise island via a special `merge` that appends the
elementwise recipe as the atom's epilogue `ScheduleRecord` — the *one* legal
merge that crosses an atom boundary, and only in the atom→epilogue direction the
kernel already supports.

---

## 4. MIGRATION — staged, negative-SLOC-leaning, differentially gated

The discipline: each stage is independently shippable behind a differential
gate; **fusion decisions must be byte-identical** under the policy re-expression
(the null test — same partition → same lowered plan → byte-stable
`GpuCommand[]` stream, checked by the phase-0 `diffStreams` harness). Named
deletions per stage:

| Stage | Work | Gate (null test) | Deletes / subsumes |
|---|---|---|---|
| **I0** | Introduce `Partition` as a *derived* artifact: `segmentPlanForExecution` produces a `Partition` value instead of `ExecutionSegment[]`; `CachedSegmentDesc` becomes a serialized `Partition`. **No behavior change** — the partition is still computed by today's detector; it is merely *reified*. | `diffStreams(before, after)` empty on distil/medium/124M; suites green; fingerprint byte-stable (partition still derived → `boundaryHash` is a pure function of `G`, so the master key is unchanged). | nothing yet — this is the scaffold |
| **I1** | Mix `boundaryHash` into `computePlanFingerprint` (seam 1). Still derived, so still one partition per graph → key unchanged for every existing graph (null-test §6). Unlocks *multiple partitions per graph* structurally. | Master-key byte-identical on all existing templates (the probe's null test at scale); template hit-rate unchanged. | — |
| **I2** | Re-express the elementwise detector as `fusionPolicy` writing `merge`s over the dataflow graph (§3). **This is where G0(b) is captured** — non-adjacent fusible runs merge. | Differential is NOW allowed to *differ* (more fusion) — gate becomes: (a) `parity-fullstack-tl.ts` per-step losses agree to ~1e-5 over 30 steps (numerical equality of the *result*, per the cross-path corollary), (b) fused% strictly increases, (c) submits strictly decrease, (d) `test/compiled-plan-parity.spec.ts` gates green. | **`buildCandidateGroups`'s consecutive-scan** + **`reorderPlanForFusion`'s grouping role** (~200 SLOC); the failed "bypassed-node transparency" workaround territory is subsumed |
| **I3** | Fold epilogue/prologue/row-program claiming into `authored`-island epilogue-merges (§3). | Same numeric gate; matmul-epilogue fusion count unchanged; byte-stable where the claim was already made. | **`epilogueClaimedIds`/`prologueClaimedIds` plumbing + the claim-priority arbiter** as separate mechanisms (~300-500 SLOC of `graph-compiler.ts` claim cascade folds into merge-policy ordering); `matmul-epilogue.ts`/`row-program-detect.ts` *detection* stays (it proposes the merges) but the *executor action vocabulary* (`matmul-epilogue`, `row-program`, `prologue-skip` action kinds) collapses to island lowering |
| **I4** | Fold chunking + horizontal packing into degenerate split/merge (§2 rules 3, 5). | Numeric gate; packed-adam submits unchanged on medium (the case it earns its place). | **`horizontalPackKey` as an executor action-grouping** (becomes a pack-policy); `adam-batch`/`batched-reduction`/`matmul-epilogue`/`row-program` action kinds — architecture-debt sin-5's whole "action vocabulary" list |

**Net SLOC:** I0-I1 add the `Partition` object + `boundaryHash` (~150-250 SLOC
new). I2-I4 delete the four bespoke partitioning mechanisms' *executor
extensions* and the consecutive-scan (~800-1,200 SLOC deletable across
`fusion-detect.ts`, `graph-compiler.ts` claim cascade, and the executor action
vocabulary), while *keeping* the detection logic (re-homed as merge-proposing
policies) and the authored-kernel interiors (opaque atoms, untouched). **Target:
net-negative** — the object is smaller than the 24 sites it unifies because the
sites currently re-implement "which ops together + how to key it" 24 times.

**Which of the four representations is subsumed:** rep-2 (segments/groups) is
*replaced* by the `Partition` object — `ExecutionSegment` (the
`fused | sequential` union) is exactly a two-kind partition and generalizes to
the `Island` type. rep-1, rep-3, rep-4 are untouched in mechanism (the partition
is the new datum flowing between rep-1 and rep-3).

---

## 5. THE EDITOR CONTRACT — the seam (design, don't build UI)

The moves-editor reads and writes exactly the `Partition` object plus the
per-island `ScheduleRecord`s from the spike:

- **reads:** `(G, P)` — the semantic graph (drawn once, never redrawn:
  schedule = relabeling) and the current partition (island boundaries as the
  visible "strata/lanes" grouping). Per-island: the `ScheduleRecord` (tile
  sizes, residency, softmax-mode — the spike's proven-free interior decisions)
  and the three validity lights (T0 well-formed / T1 ℝ-valid-rel-lemmas /
  T2 τ-certified).
- **writes:** `merge`/`split` on `P` (the macro moves — the spike's *gearbox*),
  and `ScheduleRecord` field edits on an island's interior (the spike's *trim
  knob*, already proven byte-free). The editor's derivation is the sequence of
  these — an auditable, invertible proof term.

**The macro move the spike could not express — now expressible.** The spike's
wall was: materialize→stream and fuse-N→1 are "author code, not gestures"
because they cross the tile-IR's fixed dispatch-count boundary. In the islands
IR, **fuse-N→1 IS `merge`** (N islands → 1) and its inverse **split IS 1→N**.
The islands IR is precisely the "kernel-shape IR above the tile-IR" the spike
prescribed: the dispatch-count is now data (`P.islands.length`), so the macro
transitions are `merge`/`split` edits, not rewrites.

**The two spike-named gaps this must NOT solve but must NOT foreclose:**

1. **The global-residency primitive.** The spike found `kvResidency:"global"`
   has no tile-IR primitive (every block dot stages through shared/register).
   The islands IR is *above* the tile-IR and does not add residency tiers — but
   it must not foreclose them: an island's `ScheduleRecord` is an opaque handle
   the islands layer never inspects, so when the tile-IR grows a `global`
   residency tier, islands carry it transparently. **Design rule: the islands IR
   treats each island's interior schedule as an opaque token; it partitions, it
   never schedules the interior.**
2. **Streaming-form templates for lemma application.** The spike found
   materialize↔stream needs a kernel body the tile-IR authors, not a record
   toggle — because a *streamed* kernel and a *materialized* kernel are different
   bodies. In the islands model, materialize-vs-stream is a choice of *which
   authored island* covers the subgraph (a materialized-softmax atom vs a
   streamed-flash atom are two different `atom` values for the same members).
   The islands IR must not foreclose this: `merge`/`split` operate on
   *boundaries*, and swapping a subgraph's covering atom (materialized→streamed)
   is a *third* editor operation `retarget(island, atom)` — explicitly reserved
   here as a named slot (like the WGSL⊂CUDA analysis reserves `role-partition`),
   NOT designed in this campaign. It is lemma-gated (the online-softmax lemma
   licenses the streamed atom) and belongs to the lemma-library layer, not the
   partition layer. Reserving the slot costs nothing; its absence later would
   force a schema break.

---

## 6. FALSIFICATION PROBE — run, verdict

**The riskiest assumption:** *"island identity can be stable enough for tape
fingerprints"* — i.e., that two different partitions of one graph can coexist as
two template-cache states without (a) colliding under the current key or
(b) destabilizing the key for static graphs. If false, the whole "two states of
one object" premise is unbuildable on this executor.

**Probe:** `tools/t-islands-partition-probe.ts` (CPU-only, deterministic). It
constructs the canonical "non-adjacent fusible run broken by a view" graph
(`relu → reshape → reshape → exp → neg`) and measures three things against the
*real* `computePlanFingerprint` and `segmentPlanForExecution`.

**Result (verbatim):**

```
graph: 1:leafInput -> 2:relu -> 3:reshape -> 4:reshape -> 5:exp -> 6:neg
fusible?: relu=true reshape=false reshape=false exp=true neg=true

CLAIM A — partition is a pure function of the graph:
  segmentPlanForExecution(nodes) #1: (seq leafInput,relu,reshape,reshape) [FUSE exp+neg]
  segmentPlanForExecution arity=3 params=(nodes, externalNodeIds?, options?)
  => NO partition-input parameter; the merge across reshapes CANNOT be requested.
  the legal alternative: [FUSE relu+exp+neg] (seq reshape,reshape)  <- UNREACHABLE today

CLAIM B — computePlanFingerprint is partition-blind:
  key(graph, partition=default) = 0xcfdaa967
  key(graph, partition=merged)  = 0xcfdaa967
  COLLIDE (two partitions, one key)? true

PROPOSED FIX — mix a partition token into the key:
  ext-key(default) #1 = 0x73b621ac
  ext-key(default) #2 = 0x73b621ac  (null test)
  ext-key(merged)     = 0x1aa0f38e
  null-stable? true   partitions distinct? true

VERDICT: A CONFIRMED · B CONFIRMED · FIX feasible YES
```

**Verdict: the assumption is FALSIFIABLE-BUT-SURVIVES.** The probe *confirms the
risk is real* (today the partition is derived and single-valued (A), and the
master key is partition-blind so two partitions collide (B)) — and *confirms the
fix is sound*: a `boundaryHash` mixed into `computePlanFingerprint` is
**null-stable** (a static graph's partition recomputes to the same key → no
template-hit regression) AND **partition-discriminating** (two partitions get
two keys → they coexist as two states). The design is buildable on this seam.
The critical corollary the probe forces into the migration plan: **stage I1
(mix `boundaryHash` in) must ship its own null-test gate proving the master key
is byte-identical on every existing template** — because while the partition
stays derived, `boundaryHash(G)` is a pure function of `G` and must not perturb
a single existing key.

---

## I2 IMPLEMENTATION FINDINGS (2026-07-10) — the extension measured NULL; the roadmap target is corrected

Stages I0 (reify), I1 (partition token in the fingerprint), and I2a (detector
re-expressed as a merge-proposing policy; `buildCandidateGroups` deleted)
landed null-clean: 13-case decision corpus byte-identical, gates 4/4, suites
green both flag states (one pre-existing artifact: `observed-liveness` gate 2
asserts "pruning demonstrably activated", which is impossible by design under
a global `TORCHLETTE_BUILD_FROM_IR=0` — its trajectory-agreement and zero-miss
assertions pass), fusion stats and losses byte-stable on distil@512.

**I2b — gap-spanning — was built, proven, measured, and REVERTED.** The
mechanism: a chain-dependent non-fusible gap node no longer closes the open
island; it is TAINTED and its position becomes the island's earliest forced
emission (`emitPos` — sound because taint is a superset of the segment walk's
transitive dependency, so actual emission is never earlier); later fusible
nodes merge under a READINESS rule (every non-member pending input produced
before `emitPos` — the guard the failed bypassed-node-transparency attempt
lacked). Downstream machinery needed zero changes: promotion/re-execution
covers gap consumers, and buffer liveness/donation/release already operate in
ACTION-INDEX (execution-order) space, not plan-position space. On the
synthetic corpus it captured exactly the designed class (stranded chains →
one island with promoted intermediates; the readiness no-case correctly
refused; a previously un-batchable singleton pair became batchable).

**Measured on real plans: ZERO movement.** distilgpt2@512 AND gpt2-medium@512
(A100, `TORCHLETTE_DEBUG_FUSION=1` per-template dumps): identical fused-node
counts, identical fusion groups, identical losses with and without the
extension. Root cause, established from the break dumps:

1. **The stranded-run class is already harvested.** `reorderPlanForFusion`
   (Kahn) plus the epilogue/prologue/row-program claims leave the big
   backward plan at ~90% of unclaimed fusibles fused (distil: 144/160;
   medium: 576/628 + a 384/384 plan). The consecutive scan's dependent-gap
   close almost never fires on a spannable run in these plans.
2. **The residual is singletons, not runs.** The unfused fusibles (distil
   ~68/step, medium ~250/step) are length-1/2 runs — mostly weight-grad and
   activation `cast`s — each immediately consumed by the following matmul
   (`gelu→cast | transpose(weight)→matmul` is the canonical break). Every
   downstream fusible flows THROUGH the matmul, so no elementwise-island
   merge is legal at this altitude, with or without gap-spanning.
3. **The G0(b) metric is corrected.** The cumulative "3.0% fused" statistic
   undercounts structurally: compiled replays skip the lowered path's stat
   increment, so steady-state steps contribute nothing. The honest per-plan
   template numbers are the ones above. And CLAUDE.md target #5's bwd
   offender list (reshape/transpose/permute/sum/narrowBackward) is
   views + matmuls + reductions — not elementwise-fusible ops; only `add:19`
   was ever in the detector's scope.

**Corrected roadmap target (what the residual actually needs):** the
matmul-adjacent singleton casts are CLAIM-altitude work — prologue-claiming
them into the adjacent matmul dispatch (I3's epilogue/prologue-as-atom-merge
territory), not detector generality. The other real fusion win remains
reduction batching (bias-grad sums, perf target #3). CLAUDE.md target #5
("the limiter is the consecutive-only detector") is falsified for current
default-config plans and should be rewritten to point at the claim altitude.

Per the complexity budget (express-to-measure; mechanisms earn admission),
the extension was reverted after measurement; its design, soundness argument,
and this analysis are the retained artifacts, plus the corpus cases that pin
the readiness no-case and the stranded class for any future re-attempt on a
workload that actually exhibits it.

---

## 7. COST / RISK

**SLOC.** Add ~150-250 (the `Partition`/`Island` object, `merge`/`split`,
`boundaryHash`, legality checks). Delete ~800-1,200 (consecutive-scan grouping,
the claim-priority arbiter as a separate mechanism, the executor action
vocabulary `adam-batch`/`batched-reduction`/`matmul-epilogue`/`row-program`,
chunking's ad-hoc dispatch loops re-homed as split policies). **Net negative**,
per the complexity-budget doctrine — the campaign's deletion story is the four
bespoke partitioning mechanisms collapsing to one object with three policies.

**Staging** (each shippable, each gated): I0 reify → I1 key → I2 elementwise
policy (captures G0(b)) → I3 epilogue-as-atom-merge → I4 chunking/packing as
split/merge. I0-I1 are pure scaffolding (null-test-clean). I2 is the first
behavior change and the first win.

**The gate ladder for implementation** (the differential discipline, per
CLAUDE.md's corollaries):
1. **null test** (I0, I1): `diffStreams` empty; master key byte-identical.
2. **cross-path numeric** (I2+): `parity-fullstack-tl.ts` compiled-vs-lowered
   AND policy-on-vs-off agree to ~1e-5 over 30 steps (the trajectory gate that
   crosses the compiled-plan activation threshold — the corollary-2 discipline).
3. **in-suite gates** (`test/compiled-plan-parity.spec.ts`, `npm run test:gates`)
   green throughout.
4. **regression** (124M DiLoCo loss baselines {0:9.81, 3:5.92, 6:5.15, 9:4.64})
   bit-stable; peak memory not regressed.
5. **win metric** (I2): fused% strictly up from 3.0%, submits/step strictly
   down, distil@512 step time not regressed.

**Risks, ranked:**
- **R1 — the epilogue atom-merge (I3) is the hairiest fold** because the
  claim-priority arbiter (`graph-compiler.ts:240-248`) encodes a *precedence*
  (matmul-epilogue beats reduction-preamble beats elementwise) that must survive
  as a merge-policy ordering. Mitigation: I3 is gated on byte-stability where the
  claim was already made (it only *reorganizes* the code, not the decision).
- **R2 — device-keyed legality** (merge legal on A100, illegal on V100) means the
  partition is not portable across device classes. This is correct (validity
  analysis §2.4) but means `boundaryHash` must be part of a `(G, P, device)`
  triple in any serialized/replayed plan. Mitigation: the fingerprint already is
  device-instance-scoped (module-global caches reset per engine instance).
- **R3 — the planner non-change (seam 4) assumption** — that `planMemory` over
  the flat stream absorbs partition changes for free — must be verified, not
  assumed. Mitigation: I2's memory gate (peak not regressed) is exactly this
  check; the census confirms the planner is partition-agnostic by construction
  (it sees slots, not islands).

**Open review questions:**
1. Should views/bypassed nodes be *members* of an island (with a "no-dispatch"
   flag) or *outside* all islands (edges only)? The design chose outside (§1) —
   but this changes how `nodeIndex` alignment (`stream-diff.ts:147`) handles
   them. Reviewer call.
2. Is `retarget(island, atom)` (materialized↔streamed atom swap, §5 gap 2) a
   *third* mutator or a `split`-then-`merge`-with-different-atom composition? The
   design reserves it as a named slot; whether it needs to be primitive depends
   on the lemma layer's shape.
3. I3 ordering: does folding the claim-priority arbiter into merge-policy
   ordering risk a partition that today's arbiter would forbid? Needs a
   dedicated differential before I3, not just after.
```
