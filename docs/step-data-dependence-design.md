# The step learns its data-dependence — the deep unification

**Status:** DESIGN ONLY (Vin ratified direction (b), 2026-07-16). No mechanism lands
in this doc — this is the design plus a staged, gated campaign plan. It SUBSUMES three
stalled endpoints (`arena-recompute-design.md` R2′, `step-object-design.md` phases 3 + 7,
and the `stage4 §Task #43` recorded-build sunset) under one root, and gives the
recorded-build deletion a **finiteness argument** in place of the class-whack-a-mole that
has blocked it four times.

**Lineage:** the ninth application of the house move "the latent decision becomes an
object" (`schedule-state-design.md §1` enumerates six; `step-object-design.md` is the
eighth — the whole step as data). This doc does NOT introduce a ninth object. It completes
the eighth: the step object today declares its boundary, slots, partition, and recompute
segments, but it does NOT yet know its own **runtime data-dependence** — that a recurring
step is not one program but a small declared set of regimes, and that the cross-plan value
edges between its plans are a derivable consequence of those regimes. This is the same
maneuver the step object ran for structure, applied to the one axis it left latent: *which*
program ran and *which* values crossed plan boundaries.

**Normative ground truth (all cited inline):** `step-object-design.md` (the eight rulings,
the witness-harvest §4, the phase-3/phase-7 STOPs); `arena-recompute-design.md` (the R2
STOP — the +155% is genuinely-live retention, not dead pins); `stage4-compile-from-ir.md
§Task #43 (2026-07-16)` + `§(d)` (the fourth blocked deletion; the remat primitive);
`staged-execution-phase1.md §2.4` (the six guards) / `phase2b.md §5` (the observation
predicates); `architecture-debt.md` (the sin taxonomy); `ownership-derivation-design.md`
(derivation-replaces-observation, #70); `scoped-memory-design.md §1` (the epoch
vocabulary). **Empirical anchors** measured for this doc are cited inline as `[E-n]` and
tabulated in §11.

---

## 0. Declaration (one sentence)

A recurring training step is ONE object that knows its **data-dependence** — a small set
of regimes, each either ROUTED AS DATA (value-level variation that never forks the plan
graph) or DECLARED AS A VARIANT (residual structural forks), each variant witnessed
separately — from which the executor DERIVES a single cross-plan edge set (producer plan →
consumer plan, per value) that the harvest, the planner's result retention, and the
observed-liveness predicates all CONSUME instead of independently reconstructing — making
the cross-plan value set of a recurring step *exactly enumerable*, and the recorded
build's deletion a finiteness theorem instead of a class-by-class defence.

If a section cannot be stated as a declaration in one sentence of that grammar, it is
reshaped before it lands (the house one-sentence test).

---

## 1. Four open items, one root

The four faces below are not four problems. They are one: **cross-plan values are invisible
to per-plan knowledge under data-dependent variation.** A compiled plan knows only its own
IR; the step object knows structure but not which regime ran; and the value that crosses
from one plan to the next — a saved forward activation read by backward, an optimizer state
carried across the boundary, a recompute-fed temp — is seen only by *watching* the
execution, never by *deriving* it. Data-dependent variation (a GradScaler scale backoff that
perturbs the plan set) then breaks the two-consecutive-step witness at its foundation.

| # | Face | Where it STOPPED | The invisible cross-plan value |
|---|------|------------------|-------------------------------|
| 1 | Recorded-build sunset (4th blocked attempt) | `stage4 §Task #43 (2026-07-16)` | `forwardToForce` forces forward activations into a SEPARATE plan (`[512,50257]` CE logits, `[1,512,768]` layer activation `graphHeld=FALSE`); the scaler window perturbs the fp set, defeating the witness |
| 2 | #99 R2 residual (`arena` STOP) | `arena-recompute-design.md §R2` | the planner registry RETAINS ~1919 MB of genuinely-LIVE cross-plan working set (fwd `0xbd0dd584` 432 MB + bwd `0x19e72088` 1483 MB) whole-step; retention-vs-return is a POLICY the planner cannot derive |
| 3 | Checkpoint bypass (step-object phase 3) | `step-object-design.md §6 Phase 3` | no config gives compiled + low-memory; the arena pins forward positions blind to recompute liveness |
| 4 | Whole-step witness absence under checkpointing | `step-object-design.md §4 (phase-4 refinement)` | recompute READER plans re-fingerprint every backward → the whole-step tape never fires on checkpointed configs; the per-producer window was the compromise, and face 1 is its measured hole |

**The measured anchor (faces 1 and 2 point at the same edges).** On distil@512 + selective
(MLP-only) checkpointing, the backward plan reads the forward plan's saved activations
**directly** — **47 cross-plan reads, 39 of them `graphHeld=true`** (shapes `[1,512,768]`
residual hidden, `[1,12,512,64]` per-head Q/K/V, `[512,50257]` logits); only ~8 are
`graphHeld=false` (`arena-recompute-design.md:402–428`, corroborated by the `forwardToForce`
map, `autograd.ts:308–357` `[E-1]`). This is the ONE edge set. Face 2 is those same 47 edges
*retained across steps* (the +155% is the working set, not dead pins — `arena §R2` FALSIFIED
the "provably-dead" premise). Face 1 is a subset of the same 47 (the ~8 `graphHeld=false` +
the CE logits) that the witness cannot keep under the scaler window. Face 4 is *why* it
cannot: the reader plans re-fingerprint. Face 3 is the memory cost of serving them. **One
edge set, seen four ways.**

**The historical failure mode, named.** The recorded-build deletion has been attempted and
blocked four times, each by a class no prior gate exercised (`step-object-design.md §7
risk 1`): #43 (uncovered-plan census); #97 stage-2 (overlay/graph-held — RESOLVED by the
`graphHeldAt` claim-seam oracle); #97 stage-3 (checkpoint-recompute `contiguous` — RESOLVED
by per-producer witness); #43-2026-07-16 (`forwardToForce` under the scaler window — OPEN).
Each STOP enumerated the never-witnessed boundary; none PROVED it finite. The enumeration is
the disease. A finiteness argument is the cure (§6).

---

## 2. The object extension — data-dependence as two derived facets (zero schema delta)

The step object today (`src/core/step-object.ts`) carries `declaration`
(boundary/slots/recompute/partition), `skeleton` (the witnessed tape), `regime`
(`{ stepScopedCleanup: boolean }` — guard 5, the ONLY structural-variation field today),
and `receipts`. It does NOT carry: which regime ran, or the cross-plan edge set. This
campaign adds NEITHER as a stored slot. Both are **derived** — one from a declared set, one
from the witnessed streams — honoring the zero-schema-delta discipline (`schedule-state-
design.md §2`: receipts hash into neither identity; derivation replaces observation, #70).

```
StepObject (unchanged fields) + two DERIVED facets:

  variants: VariantSet          // DECLARED (small enum) — the RESIDUAL structural forks
                                //   after value-level variation is routed as data;
                                //   the selector is a RECEIPT read at a readback
  crossPlanEdges(v): EdgeSet    // DERIVED per variant from the witnessed streams
                                //   ∪ the declared boundary contract; NOT stored
```

- `variants` is a **declaration** (§3): the finite set of *structurally-distinct* programs
  a recurring step may run, after the value-level variation is routed as data. It hashes into
  identity ONLY through a variant TOKEN (§3.4), never through the per-step selection.
- `crossPlanEdges(v)` is **derived at inquiry** (§4), exactly as liveness is derived (#70):
  the executor computes it from the witnessed streams and the declaration; it is a query,
  not a field. The harvest, the planner retention, and the observation predicates become
  *callers* of this one query.

`regime` GENERALIZES: today it is `{ stepScopedCleanup }`; the variant discriminator is the
same KIND of fence (two steps in different variants are never paired, §3.5), so the variant
token extends `regime`'s role rather than adding a parallel field. No second owner; no new
identity scheme.

---

## 3. Element A — data-dependent variation as DECLARED structure

**Declaration:** *value-level data-dependent variation is ROUTED AS DATA so it never forks
the plan graph; the residual structural forks are a small DECLARED variant set, each
witnessed separately; an unwitnessed variant is a structured BucketMiss (a typed refusal
that re-witnesses), never a silent stale replay.*

### 3.1 What exists today (verified)

- **`StepTape.regime` is NOT a variant selector.** It is `{ stepScopedCleanup: boolean }`
  (`step-tape.ts:117`), guard 5 only. There is NO first-class variant/BucketMiss type.
  Structural variation is represented IMPLICITLY: same `bucketKey` (`"b:<fnv(structKey)>:
  <fp+fp+…>"`, `step-tape.ts:952`) ⇒ same variant; a `structureMiss` counter increments when
  consecutive `structKey`s differ; a `refusals` counter increments on declared-coverage
  violations (guard 3, `diffImages`). So today the scaler-backoff window is not a *declared
  variant* — its structural deltas are undifferentiated `structureMiss`es, indistinguishable
  from a genuine structure change, and they silently prevent the tape from ever going warm
  across the window.
- **The GradScaler "skip" is DATA, not a host branch (verified `[E-2]`).** `GradScaler.step()`
  (`grad-scaler.ts:506`) ALWAYS calls `optimizer.step()` and returns true — there is NO
  host-side branch skipping the optimizer. The "skip" is realized in-kernel: `unscale_`
  zeroes non-finite grads (fused `unscaleGrad` "unscale + inf-check + zero-mask",
  `grad-scaler.ts:385`; elementwise `where(shouldZero, 0, g)`, `:466`), and the scale flows
  as a live tensor (`_scaleLive`, `:315`); `adamStep`/`unscaleGrad` payloads are
  `PAYLOAD_HASH_EXEMPT: "ALL"` (`fusion-detect.ts:1756`). This is the "scaler-as-tensor"
  refactor's whole point: keep the skip from perturbing the plan.
- **…yet the scaler WINDOW is empirically NOT fp-stable (measured `[E-3]`).** A forced-inf
  run (`CELL=scaler-inf`, initScale 1e40 → 1.5e38 over steps 0–5, `observedInfSkip=true`)
  shows the fp SET changing across the window and reports `witnessVariances=3`. Concretely, a
  `nodes=2` plan `0xccadbdf5` appears ONLY during the scale-backoff steps (2–5) and vanishes
  once the scale settles; a `nodes=295` plan `0xaded5357` and a second `nodes=270`
  fingerprint `0x77663cc5` (vs `0x991b475f` during backoff) appear only later. **Reconciled
  truth:** the optimizer never host-skips (`[E-2]`), but data-dependent SCALE-CHANGE ops (the
  live-tensor write only fires `if _scale !== prev`, `grad-scaler.ts:296`, so it is a
  structural op present only while the scale moves) DO perturb the plan set within the
  window. So the docs' "inf-skip re-fingerprints" is right at the OBSERVABLE level and agent-2's
  "no host skip branch" is right at the MECHANISM level — the fork is scale-regime-driven, not
  optimizer-skip-driven. This distinction is the design's fulcrum (§3.2).

### 3.2 Route-as-data first; declare only the residual (the frozen-scalar discipline, generalized)

The `[E-2]`/`[E-3]` reconciliation yields the design's central rule. Data-dependent
variation splits into two kinds, handled differently:

1. **Value-level → ROUTE AS DATA (no variant).** Variation that CAN flow through a declared
   slot without forking the graph MUST. The scale-change op `0xccadbdf5` is the live example:
   it exists only because `_applyScaleAdjustment` conditionally writes the scale, creating a
   structural op during backoff. Routing that write as an UNCONDITIONAL LiveScalar update (the
   scale is *always* rewritten, to the same value on a no-change step) folds it into the normal
   plan — the same maneuver that already made the inf zero-mask and the scale multiply data
   (`[E-2]`). Under this rule the scaler-backoff window collapses to ONE plan set: no variant.
   This is the frozen-scalar discipline (`architecture-debt.md §1`) generalized from optimizer
   scalars to control-flow: *a value that varies is a declared slot, or the digest refuses it.*
2. **Structure-level → DECLARE A VARIANT (residual).** Variation that genuinely forks the plan
   graph and cannot be routed as data is a declared variant. The clearest is `eval` (no
   backward, no optimizer) — a structurally different program. Fingerprinting ALREADY separates
   it (`[E-2]`: an eval step's node set differs → different fp → its own template), so a
   variant here is a DECLARATION of a fork the machinery already makes, letting each fork
   witness on its own stream rather than thrashing one.

**Consequence for the variant set's SIZE (the finiteness bound, §6/§9).** Because value-level
variation is routed as data, the DECLARED variant set is small by construction — v1 is
plausibly a SINGLETON (`normal`) once the scale-change op is routed as data, with `eval` a
separate captured region (§12 Q2). The scaler window is NOT a variant if the scale write is
made unconditional; it is a variant only if we choose to keep the conditional write. This is
the honest, evidence-driven correction to the task's premise: the motivating "inf-skip
variant" is better dissolved by routing-as-data than declared — and the variant mechanism's
real customers are the genuinely-forking cases (eval, structural milestones).

### 3.3 The mechanism — variant as a discriminator on the step digest

For the residual structural variants, a variant is a discriminator mixed into identity —
structurally the SAME maneuver as the partition edit token, one stratum up:

1. **Declaration.** `variants: VariantSet` enumerates the residual structural programs; each
   carries a **structural predicate** (what its graph differs in) and a **selector binding**
   (which receipt chooses it — e.g. the train/eval mode flag, or a scaler receipt if a scale
   variant is kept).
2. **Selection is a receipt, not identity.** At the readback/flag where the selector resolves,
   the step object records `receipts.variant = v`. This hashes into NEITHER the digest — the
   SELECTION is per-step data; only the variant's structural TOKEN hashes.
3. **Keying.** The step digest becomes `(declaration-hash, variantToken(v), ordered plan fps)`.
   `variantToken` extends `regime` in `structKey` (§2). A variant addresses its OWN tape
   skeleton — each variant goes warm on its own two-consecutive-identical-*within-variant*
   steps.
4. **Structured miss.** A step whose selector resolves to a variant with no eligible tape is a
   **BucketMiss on the variant axis** (a typed refusal) → it runs the normal build+execute path
   (which witnesses that variant) and, under `STRICT_TAPE`, is loud. It is not a
   `structureMiss` ("unknown structure, re-record") — it is "known variant, not yet warm",
   self-healing in two occurrences of THAT variant.

### 3.4 The `resolveEditedFingerprint` reuse verdict (verified `[E-2]`)

**Verdict: reuse the token-mixing PRIMITIVE, not `resolveEditedFingerprint` itself.**
- REUSABLE: `computePlanFingerprint(nodes, externalNodeIds, partitionToken)`'s third argument
  (`fusion-detect.ts:1628`, mixed into all three hashes at `:1737` — primary, secondary, AND
  structural, so it re-keys AND is treated as distinct structure, not payload thrash) IS a
  generic "mix a discriminator int into identity" seam. A `variantToken` rides it exactly as
  the island `boundaryHash` does.
- NOT REUSABLE AS-IS: `resolveEditedFingerprint` (`executor.ts:490`) is a STRUCTURE-keyed
  island-merge resolver — its `editedPartitions` map is keyed by `defaultFp.primary`, its token
  is DERIVED from the plan's own structure (a merged partition's `boundaryHash`), and it holds
  a PERSISTENT monotone per-plan edit directive. A variant is (a) STEP-scoped not plan-scoped,
  (b) selected per step from a RUNTIME DATA receipt (found-inf / mode), and (c) known ≤1 step
  stale (the scaler readback resolves the PRIOR step, `grad-scaler.ts:182`). So a variant token
  is threaded from the step object's per-step `receipts.variant` into `computePlanFingerprint`'s
  `partitionToken` (≈1 line at the executor call), BYPASSING the S3 resolver. The token-mixing
  math is shared; the delivery is the step object's, one stratum up — consistent with the whole
  campaign (`step-object-design.md §2`).

**Empirical fp-delta `[E-3]` (the anchor the verdict rests on):** normal vs scaler-window steps
do NOT differ by a clean optimizer-skip fork; they differ by (i) the scale-change op
`0xccadbdf5(nodes=2)` present only during backoff — a value-level delta the design routes as
data (§3.2), and (ii) later steady-state fp transitions (`0xaded5357(nodes=295)`,
`0x991b475f→0x77663cc5(nodes=270)`) that the ordinary template mechanism already handles. So the
GradScaler case needs NO variant token once (i) is routed as data — the variant primitive's real
customer is `eval`, which fingerprinting already separates.

### 3.5 Variant fencing (the `regime` generalization)

Tape eligibility today requires `p.stepScopedCleanup === rec.stepScopedCleanup`
(`step-tape.ts:932`). The variant token joins that comparator: two steps are paired for
witnessing ONLY if they are the same variant. This is why K_w stays 2 (ruling honored, not
widened): variation is handled by DECLARATION (route-as-data + separate witness streams per
residual variant), never by widening the window to catch variation inside one stream.

---

## 4. Element B — step-level cross-plan liveness (three observers → one derivation)

**Declaration:** *by witness time the step object knows the ordered plan sequence per variant;
the cross-plan edge set (producer plan → consumer plan, per value) is derivable from the
witnessed streams ∪ the declared boundary contract, and the harvest, the planner's result
retention, and the observed-liveness predicates all CONSUME that one derived edge set.*

### 4.1 The one edge set

For a variant `v`, the cross-plan edge set is the union of three declared/derivable sources:
- **Autograd cross-plan reads** — backward reads forward saves that `forwardToForce` split
  into a separate plan (§5). Derivable at end-of-step (backward has run): the ~47 measured
  edges `[E-1]`. Their producer/consumer plans and `(templateFp, nodeIndex, oi)` value stamps
  are facts of the witnessed stream.
- **Recompute-fed reads** — a checkpointed activation recomputed before backward's first read.
  Derivable from the declared `RecomputeSegment[]` + the `isCheckpointBoundary` positions
  (`arena §3 Candidate B`); the stamp names the remat value.
- **Boundary survivors** — persistent state (params, m/v, scale) + the declared K-ring outputs
  (loss, diagnostics) crossing the step boundary. Declared, not observed (`phase2b.md §5`).

### 4.2 The three-observers → one-derivation map (the sin-taxonomy collapse)

Three mechanisms independently reconstruct facets of this ONE edge set today — the exact
"single source of truth at seams" violation the sin taxonomy names (`architecture-debt.md
§Taxonomy`; CLAUDE.md's canonical rule). The unification: each becomes a CONSUMER of
`crossPlanEdges(v)`.

| Observer (today) | What it independently reconstructs | Becomes |
|---|---|---|
| **Witness harvest** (`step-object-design.md §4`, per-producer K_w=2 witness set into `prunedHarvest`'s keep set) | which cross-plan reads a producer's result feeds (observed by watching two steps) | a QUERY of `crossPlanEdges(v)`: the harvest keep-set for producer `p` = `{ consumers of p in the edge set }` — derived, not re-observed |
| **Planner RESULT retention** (`arena §R2`, the +155% whole-step pinning of `resultEntries`) | which results are live across which plan spans (retained whole-step because the planner can't see when they die) | a QUERY: a result is retained iff it has a cross-plan consumer edge LATER in the variant's ordered stream; returned to the pool after its last consumer — retention-vs-return becomes DERIVED, not policy |
| **Observed-liveness predicates** (`phase2b.md §5`: `everSurvived` / `everReadback` / `everAliased`) | which values survive the boundary / are read back / alias (observed by watching executions) | RETIRE on the captured path (ruling 2): `everSurvived` = boundary survivors in the edge set; `everReadback` = declared ring outputs; `everAliased` = statically visible in the recorded plan sequence — all three DERIVED from `crossPlanEdges(v)` + the declaration |

This is the campaign's structural payoff: three observers watching the same executions to
reconstruct the same edges collapse to one derivation with three callers. The seam assertion
during migration (single-source discipline): each consumer's derived answer must EQUAL what it
observed today (set-parity gates, §8), else the phase does not land.

### 4.3 What #99's policy question becomes (face 2 resolved)

`arena §R2` STOPPED because the +155% is genuinely-LIVE retention and the witness signal could
not soundly drive a split: "witnessed lowered read" does NOT imply "recompute-fed /
dead-after-producer." That is TRUE — and it is exactly the confusion the ONE edge set removes.
The witness signal alone conflates two edge KINDS. `crossPlanEdges(v)` distinguishes them BY
CONSTRUCTION: an edge is recompute-fed iff its producer is inside a declared `RecomputeSegment`
(the split is legal — the value is dead between forward-last-read and recompute); an edge is a
genuine save otherwise (retained until its last consumer, NOT split). Retention-vs-return is
then DERIVED per edge, not a whole-step policy the planner guesses. The R2 split becomes sound
because it is driven by the declared segment boundaries (`arena §3 Candidate B`, RULED), with
the edge set naming which results are returnable after their last cross-plan consumer — the
"retention-vs-return delta" `arena §R2` named as the real target, now a derived quantity. The
STOP's own falsification (no witnessed pair on this config is recompute-fed) is HONORED: on
selective checkpointing the split's sound target is empty, so the edge set marks those 47 as
genuine saves (retained, not split), and the memory win comes from returning them at their last
consumer — NOT from an unsound recompute split. The design does not resurrect the split the STOP
killed; it makes the retention DERIVED so the return point is known.

---

## 5. Element C — `forwardToForce` at step altitude

**Why backward forces forward activations in separate plans (verified `[E-1]`,
`autograd.ts:308–357`).** When `hasCheckpoints`, `backward()` splits the force into two
compiled plans: `forwardToForce` (every still-lazy `node.inputs` forward activation feeding a
backward node) and `savedToForce` (the recomputed saved tensors). They MUST be separate because
mixing unmaterialized forward nodes into the recompute plan produces invalid reshape rewrites
(`autograd.ts:314–317`). This split MANUFACTURES the cross-plan values: a forward activation
materialized in the forward-tensors plan, consumed by the later main backward plan. Non-checkpoint
backward skips both forces (everything stays lazy in one merged plan — no cross-plan value).

**Under declared recompute segments + step-level edges, do these reads become derivable? YES:**
- The ~47 forced-forward reads ARE the autograd cross-plan edges of §4.1. At end-of-step (backward
  has run) they are facts of the witnessed stream. `crossPlanEdges(v)` enumerates them — no
  separate `forwardToForce` list needed; the force merely realizes edges the edge set already
  names.
- The `graphHeld=true` majority (39/47) are boundary survivors of a KNOWN kind: a saved-for-backward
  retention clone (`graphHeldAt=true`, the #97 stage-2 oracle) with a guaranteed reader until
  `cleanupAutogradGraph`. Declared-live by the autograd contract — the edge set marks them retained,
  never split.
- The `graphHeld=false` minority (~8, incl. the `[1,512,768]` overlay-released activation, the
  `[512,50257]` CE logits) are the face-1 hole. In the edge set they are edges like any other: their
  producer has a later consumer, so they are retained until it, regardless of `graphHeldAt`. **The
  edge set SUPERSEDES the `graphHeldAt` heuristic for these:** liveness is the presence of a later
  consumer edge, not a graph-retention flag. This is precisely the #97-2026-07-16 STOP's own
  proposed unblock ("extending the overlay-release oracle past `graphHeldAt` to witnessed cross-plan
  readers") — made principled by deriving the consumer edge rather than heuristically guessing.

**Does the seed-fix pattern generalize?** The seed fix (`autograd.ts:319–334`, `[E-1]`) leaves the
leaf constant `full([],1.0)` grad seed LAZY (out of `forwardToForce`) so it materializes intra-plan
alongside its consumer — no cross-plan value, no prunable harvested result. This is a special case
of the general rule the edge set makes principled: **a value with no cross-plan consumer edge should
not be forced across a plan boundary.** Today `forwardToForce` forces every non-materialized
`node.input` blindly; the edge set gives it the predicate to force selectively — a derivation
replacing a blanket force. (A DIVIDEND the campaign names, not a v1 requirement — face 1 is unblocked
by the edge set carrying the ~8 `graphHeld=false` edges through the harvest, whether or not
`forwardToForce` is later made selective.)

---

## 6. Element D — the finiteness argument (the doc's spine)

The recorded-build deletion has been blocked four times by "one more class." This section
replaces enumeration with a theorem-shaped claim: the cross-plan edge set of a recurring step is
not open-ended — it is exactly enumerable, under stated assumptions, and each of the four
historical classes is covered by the construction (not patched after the fact).

### 6.1 The claim

> **Theorem (shape).** For a recurring step declared with a finite variant set `V` (the residual
> structural forks after value-level variation is routed as data, §3.2), and given two consecutive
> witnessed executions *within each variant* `v ∈ V`, the cross-plan value edge set
> `crossPlanEdges(v)` is EXACTLY ENUMERABLE — every value read across a plan boundary in any future
> execution of variant `v` is a member — because a step's cross-plan reads are a deterministic
> function of (a) its autograd graph, (b) its declared recompute segments, and (c) its declared
> boundary survivors, all structurally constant within a variant and fully observed by end-of-step
> in the witness executions.

### 6.2 The assumptions (named honestly — what makes a NEW class impossible)

1. **Route-as-data completeness.** Every source of *value-level* data-dependent variation flows
   through a declared slot (the frozen-scalar discipline, §3.2). Backed by `[E-2]`/`[E-3]`: the
   GradScaler skip is already data (zero-mask + live scale); the residual scale-change op is
   routable (§3.2). Anything NOT routed shows up as guard-3 UndeclaredVariance (`phase1.md §2.4`)
   — a typed refusal, never a silent wrong result.
2. **Variant-closedness.** Every source of *structural* (plan-graph) variation is a declared
   variant in `V`. `V` is small BECAUSE of assumption 1 (value-level variation doesn't fork).
   Defensible because the residual structural sources are few and enumerable: phase (train vs eval)
   and genuine structural milestones (rare). Anything NOT in `V` is caught by §6.4's runtime guard
   as a typed refusal.
3. **End-of-step observability.** Every cross-plan read is physically observed by end-of-step in the
   witness executions, because witnessing runs the FULL program — forward, backward, recompute
   (`step-object-design.md §4.1`). This is the recorded build's property (build-WITH-execution)
   promoted to the tape; the edge set is the UNION of observed reads across the two within-variant
   witness steps. It defeats the #97 infeasibility (no whole-step graph at forward-build time) — the
   edge set is derived at end-of-step, when backward's reads are facts.
4. **Monotone-safe keep.** The keep-set only ever GROWS (`step-object-design.md §4 refinement`):
   over-keeping wastes memory; under-keeping crashes. A within-variant disagreement republishes the
   UNION and counts a `witnessVariance` receipt (`[E-3]` measured `witnessVariances=3` on the
   un-routed scaler window — exactly this path firing) rather than dropping an edge. So a
   mis-enumeration is bounded to memory, never correctness.

### 6.3 Each historical class is covered by the construction (not patched)

| Historical blocked class | Covered by |
|---|---|
| #43 uncovered-plan census (strided views, chunked >128 MB, batch >64, non-f32, CoW, transient/cold) | assumption 2's SCOPE: these plans never reach two within-variant witness steps; they stay lowered/recorded FOREVER by construction (§7). A declared boundary of the theorem, not a hole |
| #97 stage-2 overlay / graph-held saves | assumption 3: `graphHeld=true` saves are boundary survivors with a guaranteed reader — enumerated as retained edges; the edge set marks them live, superseding `graphHeldAt` (§5) |
| #97 stage-3 checkpoint-recompute `contiguous` | assumption 3 at the PRODUCER stratum (per-producer K_w=2): the producer recurs identically; its cross-plan read is witnessed end-of-step |
| #43-2026-07-16 `forwardToForce` under scaler window | assumptions 1 + 4: route the scale-change op as data (§3.2) so the window is fp-stable and witnesses whole-step; the ~8 `graphHeld=false` edges are carried by the monotone keep (assumption 4) as genuine-save edges (§5) |

The fourth row is the crux and where the empirical CHANGED the prior story: face 1 was framed as
"the inf-skip re-fingerprints, so no two consecutive identical steps ever occur." `[E-2]`/`[E-3]`
show the re-fingerprinting is a VALUE-level scale-change op, not a structural skip — so the primary
fix is route-as-data (assumption 1), which makes the window fp-stable and the whole-step witness
fire, WITHOUT needing a declared variant at all. The remaining `graphHeld=false` cross-plan edges are
then carried by the edge set as genuine-save edges (§5). The prior "declared variant" framing was
over-strong; the evidence says the scaler case is a route-as-data case, and the variant mechanism's
real customer is `eval`.

### 6.4 The runtime guard — a typed refusal, not recovery

Assumptions 1–2 need a runtime backstop, designed as a **typed refusal**, never recovery (the
ratified discipline): an **UnwitnessedVariant** refusal. If a step's `structKey`/variant selector
matches no witnessed stream, the step falls back to the normal build+execute path (always correct —
it never went away), records the observed structure as a candidate new variant, and under
`STRICT_TAPE` throws. It does NOT attempt a partial replay recovery (the deleted `guardMiss`
recovery, `step-object-design.md §6 Phase 5`, is the precedent — recovery became a should-never-fire
assert). A genuinely new structural variant self-heals by witnessing; a spurious one is loud. This is
the finiteness argument's honest edge: not "no new class can exist" but "a new class is a typed
re-witness, never a silent corruption."

---

## 7. Element E — the end-state ledger

**What DIES (staged, gated — the acceptance criteria):**
- **The recorded build** (`buildCompiledPlan` + ~80 `record*` refs; the reconciled harvest-deletion
  diff at `.claude/harvest-deletion-43a-reconciled.diff` builds clean — measured **113 added / 1299
  removed, net −1186** `[E-4]`). Deleted once `crossPlanEdges(v)` covers every RECURRING plan's
  cross-plan reads across the full config matrix (§8), incl. the scaler window fp-stabilized by
  route-as-data. Transient/cold plans KEEP it forever (§7 below).
- **The per-producer witness compromise** (`step-object-design.md §4 refinement`): subsumed by the
  route-as-data-stabilized whole-step witness. The per-producer window was the compromise forced by
  the window's fp instability; routing the scale-change op as data makes the whole-step witness fire,
  and the per-producer stratum becomes a derivation of the edge set (§4.2).
- **The observation predicates** `everSurvived` / `everReadback` / `everAliased` on the captured path
  (`phase2b.md §5`, `step-object-design.md phase 7`): they become queries of `crossPlanEdges(v)`
  (§4.2). The LOWERED fallback keeps them (a captured-path dividend, not a global deletion).
- **The checkpoint bypass** (`setBufferArenaDisabled(true)` + `TORCHLETTE_CHECKPOINT_ARENA`,
  `arena §R3` / `step-object-design.md §6 Phase 3`): the edge set makes retention-vs-return derived
  (§4.3), so checkpointed steps run compiled AND low-memory, and the bypass dies.
- **The planner's whole-step RESULT pinning** (`arena §R2`): retention-vs-return derived per edge
  (§4.3); the whole-step pin collapses to the per-edge survivor set.

**What STAYS (by construction, not omission):**
- **The lowered path as the semantic reference** — every tape/edge-set claim is a differential
  against it (the standing `parity-fullstack-tl` gate).
- **Transient / cold plans run lowered forever** — a plan that never reaches two within-variant
  witness steps has no edge set and no tape; it runs the normal path once, exactly as today. The
  theorem's declared boundary (§6.3 row 1), not a gap.

---

## 8. The campaign plan (staged, shippable, gated)

Stage-4 style: each phase lands as its own commit with gates in the message; no phase forces a
deletion its gate has not earned (STOP-rather-than-improvise). Every phase gate is
**checkpointing-ON + event-inclusive** (the four-times-fooled rule): the matrix interposes a
GradScaler forced-inf window and a scheduler milestone and shows each either routes as data
(fp-stable, `[E-3]` is the failing-first oracle to flip) or declares a variant (witnessed).
Sequencing: route-as-data → derive the edge set → collapse the three observers → serve recompute →
sunset.

### Phase D0 — Route-as-data + variant set reified (declare, null-clean)

**Goal:** (a) make the GradScaler scale-change write UNCONDITIONAL (route-as-data, §3.2) so the
scaler window is fp-stable — the failing-first oracle is `[E-3]`'s `witnessVariances>0` / the
`0xccadbdf5(nodes=2)` backoff-only plan, which must go to zero; (b) reify `variants: VariantSet` as
a DECLARED enum with the `variantToken` seam (§3.3), initially a singleton (`normal`) — null-clean.
- **Gate:** `[E-3]` re-run shows `witnessVariances=0` and no backoff-only plan across the scaler
  window (route-as-data landed); `diffStreams` empty on distil/medium/124M for the singleton variant;
  `bucketKey` recomputes byte-identically; suites green both flag states.
- **Deletes:** the conditional scale-write's structural op (folded into the normal plan).

### Phase D1 — `crossPlanEdges(v)` derived (the one edge set)

**Goal:** derive the cross-plan edge set per variant from the witnessed streams ∪ the declared
boundary; expose it as a QUERY. Wire the witness harvest to CONSUME it (keep-set = the edge set's
consumers per producer). Null-clean: the derived harvest set must EQUAL the per-producer witnessed
set on every covered plan.
- **Gate:** set-parity — `crossPlanEdges`-derived harvest == the phase-4 per-producer witnessed
  harvest on distil@512+ckpt, medium@512, 124M chunked-sum, AND the (now fp-stable) scaler window;
  zero `Input not ready`; trajectory parity ≤ 1e-5/30.
- **Deletes:** nothing yet (per-producer set stays as the shadow oracle until D2).

### Phase D2 — Three observers → one derivation (the collapse)

**Goal:** the harvest, the planner retention, and the observation predicates all consume
`crossPlanEdges(v)` (§4.2). The per-producer witness compromise retires. Planner retention becomes
per-edge (retention-vs-return derived, §4.3).
- **Gate:** each consumer's derived answer == its observed answer (three set-parity gates — the
  single-source seam assertion); the `arena §R2` memory oracle FLIPS (arena-ON current within +5% of
  arena-free, because the whole-step pin is now per-edge return-at-last-consumer); test:gates green.
- **Deletes:** the per-producer witness set; the whole-step RESULT pin's policy heuristic.

**STATUS 2026-07-16 — D2a LANDED; D2b (per-edge planner return) attempted at the
externalReleases seam, MEASURED NULL, REVERTED.** The harvest face is collapsed: the
per-producer publication into observed-liveness is deleted; `prunedHarvest` and the
executor's recompute-boundary feed consume only `crossPlanEdgeKeepSet`; the shadow diff
inverted to a verification tool (`getWitnessProducerKeepSets`, diffed on demand — all 5
matrix cells EMPTY post-collapse). The planner-retention face did NOT land, with the
deterministic evidence recorded here so the next attempt doesn't re-walk it:
- **Mechanism tried:** per-edge return through the EXISTING stage-3 B `externalReleases`
  seam — a consumer plan claims a producer's registry entry when the derived
  current-generation consumer set names it the LAST consumer to execute this step
  (superseding `graphHeldAt` for witnessed pairs, §5), with the observed-fallback kept for
  unwitnessed producers. Sound: 78 derived releases/run on distil@512+selective-ckpt,
  zero `[lifetime]` throws, finite descending loss.
- **Why NULL:** `planMemory` claimed ZERO of the released entries (claimedEntries=0;
  registry byte-identical 2355.7 MB materialized / 1260.9 MB result / 699 entries).
  Two structural reasons: (a) the releases fire at the CONSUMER's build, but the backward
  plan reads the forward saves LATE in its stream (reverse-layer order) — few of its own
  temps allocate after the release point, and the plans that could reuse the bytes (the
  optimizer) never bind those saves, so a per-plan release-event cannot express the
  cross-plan reuse; (b) the packing is baked at BUILD time (steps 0–1), before the edge
  set exists (witnessed at step ≥2) — the derived signal arrives one lifecycle too late
  for the build-time seam unless the build is re-planned post-witness (a REBUILD, not a
  release event). The R2 memory oracle did not move (arena-ON 4106.5 MB vs budget
  1888.2 MB) — the whole-step pin survives.
- **What D2b needs instead (repriced):** post-witness RE-PLANNING — once a template's
  edge set is witnessed, rebuild its consumer plans' memory plans with the producer's
  RESULT intervals split at their derived last-read positions (the §4.3 "returned to the
  pool after its last consumer" applied INSIDE planMemory as interval ends, not as
  build-input release events). That is a planner-input change (resultSlots becomes
  resultSlots-with-end-positions), staged behind its own differential. The whole-step
  RESULT pin heuristic therefore STAYS until that lands; deleting it now would be a
  deletion the gate has not earned.

### Phase D3 — Checkpoint bypass dies (subsumes #99 R2′ / R3, step-object phase 3)

**Goal:** with D2's per-edge retention, checkpointed steps run compiled + low-memory. Delete
`setBufferArenaDisabled` + `TORCHLETTE_CHECKPOINT_ARENA` (`arena §R3` death list).
- **Gate:** the b66ead78 A/B (124M/batch1/seq256) — bypass-OFF baseline-EXACT loss + peak within
  budget vs bypass-ON; distil@512 selective-ckpt; `Input not ready` zero; the R2 gate ladder through
  `WebGPUGPT2Trainer`; test:gates 4/4.
- **Deletes:** `setBufferArenaDisabled` / `bufferArenaDisabled` / `arenaDisabled` threading;
  `TORCHLETTE_CHECKPOINT_ARENA`; the step-object §6 Phase 3 STOP narrative; (named follow-on) the
  dead segmented executor.

### Phase D4 — Recorded-build sunset (the finiteness argument cashed; subsumes #43)

**Goal:** with the edge set covering every recurring plan's cross-plan reads across the
event-inclusive matrix (the finiteness theorem's assumptions all gated), delete the recorded build
for covered classes (`.claude/harvest-deletion-43a-reconciled.diff`, net −1186 `[E-4]`).
Transient/cold classes keep it (§7).
- **Gate:** the full §6 matrix — distil@512+selective-ckpt (the #97 config), medium@512, 124M
  chunked-sum, fused AND foreach, WITH the fp-stabilized GradScaler window, WITH an LR schedule
  crossing a warmup→decay knee; shadow set-parity empty; trajectory parity ≤ 1e-5; the
  `Input not ready: contiguous[512,768]` twin fires ZERO times with the recorded build removed for
  covered classes; the UnwitnessedVariant guard (§6.4) fires zero at steady state (a soak).
- **Deletes:** the recorded build's harvest role + `buildCompiledPlan` + the `record*` family for
  covered classes (net −1186); the recorded build stays ONLY for the never-witnessed remainder (§7).

### Phase D5 — The declared-lifetime dividend (LAST; step-object phase 7)

**Goal:** the observation predicates RETIRE on the captured path (they are now queries of
`crossPlanEdges(v)`, §4.2). Gated on ruling 2's own re-open condition (captured path warm + watcher
cost measured).
- **Gate:** set-parity of the derived liveness vs the observation layer on the captured path;
  measured watcher-cost delta > derivation cost; captured-path only (LOWERED keeps all three).
- **Deletes:** `everReadback` / `everSurvived` / `everAliased` on the captured path.

**Subsumption:** D3 = `arena-recompute-design.md` R2′ + R3; D2/D5 = `step-object-design.md` phase 7
convergence; D4 = the `stage4 §Task #43` sunset (the four-times-blocked deletion); D0–D1 are the
net-new route-as-data + derivation the four faces all needed.

---

## 9. Risks named honestly

**Risk 1 — the variant set is open-ended; what bounds it?** The finiteness argument's soft underbelly
(assumption 2). The empirical `[E-2]`/`[E-3]` REDUCES this risk vs the task's premise: the motivating
"inf-skip variant" is really route-as-data, so `V` may be a singleton in v1. The bound is threefold:
(a) route-as-data (§3.2) keeps `V` small — most data-dependent variation is value-level and doesn't
fork; (b) the UnwitnessedVariant guard (§6.4) makes any undeclared variant a typed re-witness, never
a wrong answer — an incomplete `V` costs a cold step, not correctness; (c) monotone-safe keep
(assumption 4) bounds a mis-enumeration to memory. Honest residual: a heavily data-dependent
control-flow model with MANY genuine structural variants would thrash the witness (each cold until
warm) — the design declines such workloads to the lowered path via the guard, as capture declines
non-decode workloads today (`phase1.md §2.5`).

**Risk 2 — route-as-data has a cost too.** Making the scale-write unconditional means a LiveScalar
write EVERY step even when the scale doesn't change (D0). That is a tiny fixed-buffer scatter
(`live-scalar.ts`), but it must be measured NOT to perturb the implied-boundary fp regime the
`resolveDeferred` hit-path deliberately preserves (`grad-scaler.ts:182`). D0's null gate must show
the unconditional write is fp-neutral on the settled path, not just the backoff path.

**Risk 3 — the edge set's derivation cost at 124M.** `crossPlanEdges(v)` is derived from witnessed
streams whose node×plan count is large at 124M (the `diffImages` cost risk, `step-object-design.md
risk 2`). Amortized (once per warm variant tape), but frequent re-witnessing pays it repeatedly.
Measure the derivation cost explicitly in D1's gate; if it exceeds the runahead win, re-price.

**Risk 4 — the D2 collapse's seam assertions are the whole safety net.** Three observers becoming
one derivation means three set-parity gates are the ONLY thing between "derived correctly" and
"silently pruned a live edge." These MUST run checkpointing-ON with the scaler window interposed (the
third-time-is-not-fooled clause). A collapse gate that omits the window repeats face 1's exact
mistake.

---

## 10. Red-team — three strongest objections + rulings

**Objection A — "Routing the scale-change op as data is a point fix for GradScaler; the next
data-dependent optimizer (per-layer adaptive skipping, MoE routing, early-exit) forks the graph for
real, and you are back to whack-a-mole one stratum up."**
- **Ruling (§3.2 + §6.4).** The route-as-data/declare-variant SPLIT is the general mechanism, not the
  GradScaler fix. A value-level source routes as data (the frozen-scalar discipline); a genuinely
  structure-forking source declares a variant through the SAME token seam; anything outside `V` is a
  typed re-witness (never wrong). The whack-a-mole was CLASSES of invisible cross-plan values, each
  needing a new oracle; this replaces them with ONE derivation whose only extension is DATA (a
  declared variant or a routed slot) — the house move. MoE routing is the honest hard case: if it
  forks per-token it is not a small `V`, and the guard declines it to lowered — the design does not
  pretend to compile arbitrary data-dependent control flow.

**Objection B — "The edge set is derived at END-OF-STEP, but the planner needs retention decisions at
BUILD time (before the consumer plan exists). You have the same 'consumer doesn't exist yet'
infeasibility that STOPPED #97 stage-3 — moved from the forward-plan build seam to the first-step
build seam."**
- **Ruling (assumption 3 + the record-then-cut-over lifecycle).** True at BUILD time — which is why
  the mechanism is WITNESS-then-replay, not derive-at-build. The first two (per-variant) executions
  RUN the full program at build-with-execution altitude (the recorded build's own property); the edge
  set is derived from those WITNESSED streams and applied on the WARM replay, when the consumer plans
  are facts. #97 stage-3 was infeasible because it tried to derive at forward-BUILD time
  (build-without-execution); this derives at end-of-WITNESS time. The infeasibility is not moved; it
  is dissolved by the two-phase lifecycle the step object already has (ruling 1).

**Objection C — "This adds route-as-data plumbing, `VariantSet`, `crossPlanEdges`, and the
UnwitnessedVariant guard BEFORE any deletion lands — the weight-norm grows for several phases, and
the finiteness 'theorem' is a shape, not a proof (assumption 2 is unprovable in general)."**
- **Ruling (partially conceded, bounded).** Conceded: D0–D1 add mechanism before D2–D5 delete
  (per-producer compromise, observation predicates, bypass, recorded build −1186, whole-step pin).
  The weight-norm grows first — every prior campaign did. The BOUND: (a) route-as-data is a
  SIMPLIFICATION (an unconditional write replacing a conditional-op fork — likely net-negative on its
  own); `VariantSet` is DATA on the existing object (ruling 1, no second whole-step mechanism);
  `crossPlanEdges` is a derivation replacing THREE observers (§4.2), net structural simplification;
  (b) each phase names its deletions and the net-negative is the −1186 recorded-build sunset at
  campaign end. On "theorem is a shape": conceded and DESIGNED FOR — assumption 2 is backstopped by
  the §6.4 typed refusal, so soundness does not rest on `V` being provably-complete; it rests on any
  incompleteness being a re-witness, never a wrong answer.

---

## 11. Empirical findings (measured for this doc)

| id | finding | source |
|---|---|---|
| E-1 | `forwardToForce` (`autograd.ts:308–357`) splits checkpoint backward into a forward-tensors plan + a recompute plan (separate to avoid invalid reshape rewrites); the cross-plan edge set is **~47 reads, 39 `graphHeld=true`** (shapes `[1,512,768]`, `[1,12,512,64]`, `[512,50257]`), ~8 `graphHeld=false`; the grad seed is left lazy (intra-plan) as the seed-fix | agent map of `autograd.ts` + `arena-recompute-design.md:402–428` |
| E-2 | GradScaler NEVER host-skips the optimizer (`grad-scaler.ts:506` always calls `optimizer.step()`); the "skip" is DATA (in-kernel zero-mask + live-tensor scale; `adamStep`/`unscaleGrad` payloads `PAYLOAD_HASH_EXEMPT`). The token-mixing primitive `computePlanFingerprint(…, partitionToken)` is reusable for variant keying; `resolveEditedFingerprint` is NOT the entry point (structure-keyed island-merge resolver) | code read of `grad-scaler.ts` / `executor.ts:490` / `fusion-detect.ts:1616–1758` |
| E-3 | Forced-inf run (`CELL=scaler-inf`, initScale 1e40→1.5e38, `observedInfSkip=true`, device 2): the scaler WINDOW is NOT fp-stable — plan `0xccadbdf5(nodes=2)` appears ONLY during backoff steps 2–5 (the scale-change write), `0xaded5357(nodes=295)` + `0x77663cc5(nodes=270)` appear only when settled; **`witnessVariances=3`, `inputNotReady=0`, PASS**. Reconciles E-2 (no host skip) with the docs' "re-fingerprints" claim: the fork is a value-level scale-change op → route-as-data (§3.2), not a structural optimizer skip | `tools/t-witness-harvest-matrix.ts CELL=scaler-inf` with `TORCHLETTE_DEBUG_COMPILED=1` |
| E-4 | The reconciled harvest-deletion diff `.claude/harvest-deletion-43a-reconciled.diff` is **113 added / 1299 removed (net −1186)** across `compiled-plan.ts` / `executor.ts` / `observed-liveness.ts` / backend + tests | `awk` over the diff |

---

## 12. Open questions (only where they materially fork)

1. **Does the scaler case route-as-data (D0) or declare a variant?** `[E-3]` shows the window's fork
   is a value-level scale-change op, so route-as-data is the evidence-favored path and dissolves the
   "inf-skip variant" entirely. But route-as-data means an unconditional per-step scale write (risk 2)
   — if that measurably perturbs the settled-path fp regime, declaring a `scale-changed` variant is the
   fallback. FORK: route-as-data (smaller `V`, one extra write/step) vs declare-variant (no extra
   write, larger `V`). Recommendation: route-as-data (the frozen-scalar discipline; `V` stays a
   singleton), with the D0 null gate as the decider. Flagged because it sets the whole variant surface.

2. **Does `eval` belong in `V`, or is it a separate captured region?** An eval step (no-grad,
   no-optimizer) is structurally very different — arguably a separate `api.capture` region, not a
   variant of the training step. FORK: eval-as-variant (one object, more variants) vs
   eval-as-separate-region (cleaner `V`). Recommendation: eval-as-separate-region — it keeps `V` a
   singleton and makes the finiteness bound trivially small. Flagged because it bounds `V`.

---

## 13. One-sentence test (house rule)

**The step's data-dependence:** *a recurring training step is one object that routes its value-level
variation as data and declares only its residual structural forks, and derives from the witnessed
streams of each the single cross-plan edge set that the harvest, the planner's retention, and the
liveness predicates all consume — so the recorded build's deletion is a finiteness theorem (every
cross-plan value is an enumerable edge or a typed re-witness) rather than a defence against one more
class.*

Each section restates as a declaration: §3 (value-level variation routes as data, residual forks are
declared variants, an unwitnessed variant is a typed refusal); §4 (one derived edge set, three
consumers); §5 (`forwardToForce`'s cross-plan reads ARE the edge set); §6 (the edge set is exactly
enumerable under four named assumptions, backstopped by a typed refusal); §7 (the recorded build, the
per-producer compromise, the observation predicates, the bypass, and the whole-step pin all die; the
lowered path and transient plans stay); §8 (each deletion is earned by a differential its gate
crosses, checkpointing-ON and event-inclusive).
