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

**STATUS 2026-07-16 — D3 STOPPED at Stage 1; the peak-parity precondition FAILED; bypass RETAINED.**
The ratified ruling gated bypass-deletion on a re-framed memory oracle: arena-ON PEAK ≤ arena-free
PEAK + 5% (steady CURRENT reported-not-gated), justified by a cited inversion "arena-ON peak
4278.5 < arena-free 4790." Stage 1 re-measured the peak matrix fresh, like-for-like, on ONE clean
device (VULKAN 0, commit 112c9fa4, 3 repeats each, perfectly reproducible; the memory tracker is
LOGICAL bytes so arena-ON steady 4278.5 reproduces the ruling's number exactly):

| config | mode | current | steadyPeak | globalPeak (FIT, warmup-incl) |
|--------|------|---------|-----------|-------------------------------|
| distil@512 | arena-ON   | 4106.5  | 4278.5  | 5824.6–6680.3 |
| distil@512 | arena-free | 1798.3  | 3933.5  | 4414.5–4789.9 |
| medium@512 | arena-ON   | 11084.4 | 11306.7 | 18011.4 |
| medium@512 | arena-free | 6027.2  | 12789.6 | 15690.8 |

Gate (arena-ON peak ≤ arena-free peak +5%): **distil steady +8.8% FAIL, distil global +21.6% FAIL,
medium steady −11.6% PASS, medium global +14.8% FAIL.** The ruling's inversion was a **methodology
mismatch** — 4278.5 is arena-ON *steady* peak, 4790 is arena-free *global* (warmup-inclusive) peak.
Measured like-for-like the inversion is NOT robust: distil FAILS on the durable steady peak, and
BOTH configs FAIL on the FIT-honest global peak (the quantity the "FIT = peak" rationale actually
names). Deleting the bypass would **regress** distil peak +8.8% and both configs' FIT peak +15–22%
— the opposite of what the checkpoint bypass exists to protect. Per Stage 1's own STOP clause the
campaign HALTED after the measurement; the checkpoint bypass (`setBufferArenaDisabled`) is RETAINED.
The oracle (`tools/t-checkpoint-ab-oracle.ts`) is re-framed to gate on like-for-like steady peak and
now returns FAIL for the RIGHT reason (honest peak, not the old current-parity). `globalPeakMB`
(never-reset FIT watermark) + `MODEL={distil,medium}` were added to
`tools/t-planner-pin-attribution.ts` to make these numbers durable. **Re-open condition:** a future
mechanism that lowers arena-ON steady peak below arena-free steady peak +5% on distil (e.g. the D2b
post-witness re-planning that collapses the planner-registry RESULT pin, §4.3 / D2b STOP note) would
re-establish the precondition and re-enable D3.

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

**STATUS 2026-07-16 — D4 STOPPED at attempt #5 (Stage 2); a FIFTH class named; the deletion is
REVERTED, preserved uncommitted as `.claude/D4-deletion-attempt5-STOPPED.diff` (net −1209 = the
reconciled −1186 plus D0–D2's now-vanished additions inside the deleted machinery).** The
reconciliation itself was clean: the diff applied 3-way with 6 rejects (all D0–D2 context drift in
`compiled-plan.ts`/`executor.ts`), resolved by hand; `npm run build` exit 0; `tsc --noEmit` ZERO
net-new errors vs main (the two orphaned imports `crossPlanEdgeKeepSet`/`currentVariantSelection`
were removed — D2a `75d873d4` had rewritten the deleted `compilationRecording` block to call
`crossPlanEdgeKeepSet(templateFp, currentVariantSelection())`, so their only executor use died with
the block; the derived keep-set's REAL consumer, `observed-liveness.ts:627`, is retained). The CPU
project passed 1426/0. `test:gates` passed 5/5 after one tool-timing fix (`t-stream-determinism.ts`
sampled the post-invalidation rebuild after 2 steps; the recorded build re-populated
`getCompiledStreams()` one step sooner, so build-from-IR-only needs the same 3-step settle the FIRST
phase already uses — the property held byte-identically, PASS with 3 steps; that fix rides in the
preserved diff).

**THE FIFTH CLASS — the recorded build is itself a witnessing driver; assumption 3's "promoted to
the tape" is FALSE.** On the deleted tree, `CELL=scaler-inf` (the crux, `t-witness-harvest-matrix.ts`,
device 1 clean) THREW at step 9: `[lifetime] reading step-globally RELEASED storage id=11068
(shape=[1,512,768])… overlaid by the last observed reader's temps (stage-3 B)… invisible to
observation` — the EXACT attempt-#4 `[1,512,768]` overlay-release signature. Root cause is measured
directly by the same tool's `crossPlanEdges` census, main vs deleted, like-for-like (same config,
same device, D0–D2 present in BOTH — the only difference is the recorded-build deletion):

| tree | producers | pairs | edges | positions | threw | converged | result |
|------|-----------|-------|-------|-----------|-------|-----------|--------|
| main (recorded build present) | **6** | 459 | **943** | 2552 | false | 7 | PASS |
| deleted (recorded build gone) | **4** | 369 | **734** | 1742 | **true** @step9 | 6 | FAIL |

Two entire cross-plan producers VANISH without the recorded build — `0xafe3da58` (76 edges) and
`0x77663cc5` (14 edges, the settled-state plan of `[E-3]`) — and every surviving producer loses
edges (`0xde879f15` 550→484 edges / 6→5 consumers; `0xa442983c` 286→233 / 4→3). The derivation code
is UNTOUCHED and `SHADOW DIFF: EMPTY (green)` in both (shadow is derived-vs-derived, so it is BLIND
to a producer the witness never saw) — the edge set is not mis-derived, it is UNDER-WITNESSED. The
recorded build's recording pass (`startCompilationRecording` → lowered execution →
`buildCompiledPlan`) is an EXTRA lowered execution per template, and `noteWitnessRead` fires during
it; deleting it removes that witnessing source, so some cross-plan edges never reach the `K_w=2`
threshold to enter `crossPlanEdges`, leaving a `[1,512,768]` `forwardToForce` activation unprotected
→ stage-3B overlay-releases it → the UAF guard throws. Second manifestation, same root: distilgpt2
`@512` selective-ckpt Memory-Stability throws `Input not ready: contiguous[32,128]` (the class-#2
checkpoint-recompute symptom) on the deleted tree; PASSES on main. Both cells pass on main, fail on
the deleted tree, isolated on one clean device with zero `vkCreateDevice` contention.

**Assumption-3 cross-check (§6.2).** Assumption 3 states every cross-plan read "is physically
observed by end-of-step in the witness executions… This is the recorded build's property
(build-WITH-execution) promoted to the tape." Measured FALSE: the promotion is INCOMPLETE. The
recorded build's own build-WITH-execution pass is still a live witnessing DRIVER (an extra lowered
execution the compiled-replay path elides), not merely a property the tape already carries. The
finiteness theorem holds GIVEN complete witnessing; the recorded build was silently supplying part
of that completeness. Assumptions 1, 2, 4 were NOT stretched (route-as-data made the scaler window
fp-stable — `witnessVariances` 2–3, no structural fork; the derived set is monotone and shadow-green
where witnessed). Only assumption 3 failed, and precisely at its stated content.

**Re-open condition (attempt #6's charter):** a mechanism that reproduces the recorded build's
witnessing coverage WITHOUT the recorded build — i.e. drive the `K_w=2` cross-plan witnessing to
CONVERGENCE (producers/edges stable across two consecutive lowered passes) BEFORE a template cuts
over to compiled replay, so no cross-plan edge is only-ever-witnessed by the recording pass. Gate:
`CELL=scaler-inf` derives ≥6 producers / ≥943 edges (main's count) on the deleted tree with
`threw=false`, AND distilgpt2 `@512` selective-ckpt Memory-Stability is green. This is net-new
mechanism (a witness-to-convergence cutover gate), NOT a diff reconciliation — hence a STOP, not a
forced deletion. The preserved diff re-applies 3-way clean; re-run this §D4 Stage-2 matrix once the
re-open mechanism lands.

**STATUS 2026-07-17 — D4 STOPPED at attempt #6 (Stage 2, the FULL matrix); a SIXTH class named; the
deletion is again REVERTED (tree restored to `cfadb387`; the preserved diff is unchanged —
`.claude/D4-deletion-attempt5-STOPPED.diff`, applies 3-way with ONE conflict, the two executor-import
reconciliations the gate script records; build exit 0; `tsc --noEmit` ZERO net-new vs `cfadb387`,
134 vs 137 errors, all deltas line-shift/path-prefix artifacts; weight-norm on the deleted tree
64921 src SLOC / 65 env flags vs 65723 / 69 = −802 code-only, −4 flags).** The attempt-#6 mechanism
(witness-to-convergence stamping, `ef76d1b9`/`cfadb387`) DID close the FIFTH class: on the real
deleted tree, run whole-node-exclusive and strictly serial (every earlier mixed-parallel result was
re-run — device-chain contention fabricates failures wholesale; see the vk note below), the entire
WITNESSING side of the matrix is green — `t-witness-harvest-matrix` ALL FIVE cells PASS (scaler-inf
**6** producers / 459 pairs / **941** edges `threw=false`, shadow EMPTY everywhere, `inputNotReady=0`
in every cell), `t-witness-harvest-probe` PASS, `test:gates` 6/6, `parity-fullstack`
compiled-vs-lowered worst |Δ|=7.63e-6/30 steps both directions, `t-stream-determinism` PASS,
`t-ledger-attack-probe` default+STEPS=48 clean, `t-planner-pin-attribution` arena/free/--assert
PASS, `t-checkpoint-ab-oracle` witness side PASS both modes (memory side FAILs by D3-STOP design —
unchanged), and the **124M DiLoCo regression EXACT** ({0:9.8089, 3:5.9225, 6:5.1522, 9:4.6386} vs
baselines {9.81, 5.92, 5.15, 4.64}, peak flat 2087.6 MB).

**THE SIXTH CLASS — WITNESSING ≠ COMPILATION: build-from-IR does not COVER the plans the recorded
build COMPILED, and a recurring template stranded lowered-forever breaks three load-bearing
properties.** The witness-to-convergence mechanism reproduces the recorded build's witnessing; it
does not reproduce its COMPILING. The canonical fullstack training step's plans (the fused-Adam
optimizer plan; the embedding-gradient `contiguous → scatterAdd[vocab,E]` backward plan) were
compiled by the recorded build; `generateStream` cannot fully cover them, so on the deleted tree
they fall through and run lowered EVERY execution. Three deterministic manifestations (all
whole-node-exclusive, serial, real deleted tree; all green on clean `cfadb387` under identical
conditions):

1. **The step-tape/capture family collapses** — no compiled step ever enters the tape.
   `t-train-tape-matrix` fused cells: `eligiblePairs=0 loweredPairs=6 tapeCount=0` (clean control:
   `6/0/1` PASS); `t-step-object-null`, `t-step-edit-null`, `t-ring-probe` (`hits=0`; trajectories
   still byte-identical), `t-train-capture-probe` (`captureHits=0`), `t-stream-generate`
   (`hook-fires=0 verified-cmds=0`) ALL FAIL. The step-object/D-campaign surface is built on the
   tape; the deletion un-builds it.
2. **The mixed-coverage fall-through has a materialization bug** (`executor.ts` "falls through …
   with zero residue" — measured FALSE): `profile-training` distil@512 AND medium@512 BOTH crash at
   step 5 (the 2nd+ template execution, i.e. the moment build-from-IR engages) with
   `Input not ready: contiguous[0] shape=[512,768]` (distil) / `[512,1024]` (medium) feeding the
   embedding-grad `scatterAdd` — a build-from-IR-covered producer's output is not visible as a
   lowered `node.result` to the uncovered consumer plan. Clean tree: PASS (distil 5233 MB @512);
   clean pure-lowered (`COMPILED_PLAN=0`): PASS (2110 MB). Only the MIXED state crashes. The
   IN-SUITE twin is `test/distilgpt2-full-finetuning.spec.ts > Memory Stability`, which throws the
   same class (`Input not ready: contiguous[0] shape=[32,128]`, retry x2, deterministic) —
   single-file A/B: deleted FAIL / clean PASS. This is the standing `npm run test` gate that
   catches the class; the attempt-#5 STOP already named it (the "class-#2 twin").
3. **Lowered-forever recurring plans revive the unbudgeted arena** — the per-position arena is only
   reclaimed when a plan reaches compiled/planner replay (the 78c6f73 arena-reclaim); a permanently
   lowered recurring template never gets there. The foreach `t-train-tape-matrix` cells OOM the
   32 GB device (`GPU memory limit exceeded: requested 320.00MB, current usage 31.88GB`) — the
   arena-memory-blowup class, structurally re-opened for every uncovered template.

NOT the sixth class (named to prevent re-chasing): the full-suite `vkCreateDevice: Failed to create
device chain` / `VK_ERROR_INITIALIZATION_FAILED` cascade — an ENVIRONMENTAL device-chain flake on
this node's whole-suite runs (a different random file set each run; the affected files pass on
isolated single-file rerun). Discriminate any suite failure with a single-file rerun before
attributing it to a tree.

**Finiteness assumptions — final form.** Assumptions 1–4 HOLD as stated once assumption 3 is read
in its final form: witnessing completeness is delivered by STAMPING via converged-to-lowered (K_w=2
consecutive lowered executions with no valid compiled plan ⇒ stamp harvest outputs on the lowered
path), NOT by an extra witnessing pass — scaler-inf's 6-producer/941-edge derivation on the deleted
tree is the measurement. The assumption the sixth class breaks is OUTSIDE the finiteness frame: the
implicit structural claim that the build-from-IR fall-through is zero-residue and that
compilation coverage is a pure optimization. It is not — compilation is load-bearing for (i) tape
eligibility, (ii) cross-coverage result materialization, (iii) arena reclamation.

**Re-open condition (attempt #7's charter).** Either (a) `generateStream` COVERS the fullstack
optimizer plan and the embedding-grad backward plan (nothing recurring falls through), or (b) the
fall-through is made genuinely zero-residue AND lowered-forever recurring templates get tape
eligibility + arena reclamation by another route. Gates (all on the deleted tree, whole-node
exclusive, serial): `t-train-tape-matrix` 4/4 (a tape FORMS, no OOM); `profile-training` distil@512
AND medium@512 exit 0 at baseline memory; the single-file `distilgpt2-full-finetuning.spec.ts` green
(6/6 — the Memory-Stability twin is the cheapest in-suite discriminator of the class);
plus the standing attempt-#6 crux (scaler-inf ≥6 producers `threw=false`, checkpoint clean).
META-LESSON (sixth time): the acceptance gate was again NECESSARY-not-SUFFICIENT — it gated the
witnessing crux and was green while compilation-coverage consequences broke three subsystems; and
device-chain contention on a shared node FABRICATES failure classes — nothing measured under
parallel GPU load is evidence about the tree.

**STATUS 2026-07-17 — D4 attempt #7: the TWO NAMED gaps CLOSED (2 of 3 #7 cells green); a THIRD,
batch>1-only class NAMED; STOP-and-report per the #7 discipline.** The empirical census
(`TORCHLETTE_STREAM_GENERATE=1` uncovered-map, main tree, distil@512 AND medium@512 — identical) named
the two build-from-IR coverage gaps EXACTLY, and they were NOT what the sixth-class narrative guessed:
- **`op:sum[all-dims-or-epilogue]`** — a `sum(dim=…)` whose dim list reduces EVERY axis with
  keepdim=false; the dispatch routes it to the plain full reduction (`prepareDimReduction`→null), but
  the generator blindly sent `dim!=null` to `generateDimReduction`, which bailed. FIX
  (`stream-generate.ts generateFullReduction`): detect the all-dims/no-keepdim case, fall through to
  the full-reduction body (byte-identical to `fullReduction`).
- **`data-source:arange`** — a compile-time-CONSTANT index ramp (`start + i*step`, deterministic from
  shape+payload, NOT rng). FIX (`generateDataSource`): emit a plan-owned constant buffer via the
  constFill mechanism (array form; the ramp matches the `arange_tile` WGSL bit-for-bit via
  `Math.fround`), outside the arena AND the harvest ledger. `SlotSource.constFill` gained `data`.
Covering the sum EXPOSED a pre-existing latent bug: the STREAM_GENERATE reconciliation credited
"constFill == 1 recorded command", but every dispatch-based fill records ALLOC + DISPATCH + WRITE (3)
+ a params slot — under-counting by 2, latent until a fill-bearing plan became fully covered. FIX
(`executor.ts`): EXCLUDE the constFill'd nodes' recorded commands + dispatch params slots, keyed by the
source node index (stored on the slot). Gates on MAIN all green (test:gates 6/6, parity-fullstack
~5e-6/30 both dirs, t-stream-generate **0 diverged**, tape-matrix 4/4, witness-matrix 5/5, planner-pin,
ledger STEPS=48 `totalDrift=0`, 124M DiLoCo round0=9.8089 within fp-noise); on the AUTHORITATIVE deleted
tree (`d4-deleted-tree-gate.sh`, fixed to build a VALID tree — see below): **#7 tape PASS, #7 profiler
PASS**, witnessing crux 6 producers `threw=false`.

**THE THIRD CLASS (STOP) — batch>1 only, invisible to the batch=1 census.** The `distilgpt2-full-
finetuning` Memory-Stability twin (batch>1) STILL throws `Input not ready: contiguous[32,128]` on the
deleted tree. Its recurring uncovered plans (`TORCHLETTE_DBG_UNCOV` census, 24× the dominant one):
`batched-reduction[true-batched]` (the true multi-in/out batched-reduction kernel — `reductionSize≤64`,
a DIFFERENT command shape from the individual-fallback generator, appears only when a batched reduction
runs over a small dim, i.e. batch>1), plus `scatterAdd[untracked-input]` and
`fusedCrossEntropyForward[no-storage]` (the "released strided-view with no captured layout" class the
generator fundamentally bails on). profile-training@512 is batch=1 → none of these appear → the census
that named the two gaps never sees them. Covering them is an OPEN-ENDED TAIL requiring new mechanism
(a true-batched generator + released-input layout capture), NOT the two established-shape fixes — the
`STOP-and-report` boundary the #7 discipline names. The two named gaps are closed; the third class is a
new charter, not a diff reconciliation.

**GATE INFRA FIXED (necessary for a reproducible verdict).** `d4-deleted-tree-gate.sh` was structurally
broken and reported fabricated verdicts on every axis: the preserved deletion diff is a gitignored
artifact absent from a linked worktree (→ `git apply` no-op → non-deleted tree); applying it to a HEAD
carrying coverage in DELETED code made hunks silently fail (→ partial deletion); the #7 profiler/distil-ft
cells load real weights but only `node_modules` was linked (→ model-not-found MISLABELED "Input-not-ready");
and back-to-back cells raced Dawn teardown (→ contention-fabricated failures — the exact meta-lesson). Now:
apply the deletion at its NATIVE BASE + layer HEAD coverage for the SURVIVING files (plain apply), link
`models/`+`ckpts/`, `gpu_settle` between cells, and ASSERT `startCompilationRecording==0` + arange
coverage before running. With a VALID tree the two named-gap cells go green and the third class stands
alone. Re-open for #8: cover the batch>1 true-batched reduction + the released-input scatterAdd/CE class,
or make the mixed-coverage fall-through zero-residue so a lowered consumer can read a generated producer.

**STATUS 2026-07-17 — D4 attempt #8: TARGET A (zero-residue fall-through) RESOLVED — the third-class
materialization residue is CLOSED BY CONSTRUCTION, coverage-independent; TARGET B (foreach
lowered-convergence OOM) ROOT-CAUSED as a PRE-EXISTING lowered-path memory bug the charter's assumed
fix does NOT reach — STOP-and-report; the deletion is NOT landed (the foreach OOM keeps the full green
matrix out of reach).** Both on the AUTHORITATIVE deleted tree (base `71655ea8` + deletion diff + #7
coverage + Target A, sivri V100 device 0, serial/whole-node-exclusive).

**TARGET A — the residue was the stage-3 B overlay-release stranding a LOWERED consumer, not a coverage
gap (measured, not the #7 narrative's guess).** The `distilgpt2-full-finetuning` Memory-Stability twin
throws `Input not ready: contiguous[32,128]` feeding the embedding-grad scatterAdd. Traced the node's
full lifecycle: it is a saved-for-backward value (token ids: read by the forward `gather` @action5 and
the backward `scatterAdd` @action158, in a SEPARATE force's plan). A build-from-IR-compiled plan that
reads it as an external runs the stage-3 B overlay-release (`compiled-plan.ts:1823`), claims its registry
entry as boundary-dead and CLEARS `node.result` — sound for a COMPILED consumer (its bind-time guard
recovers) but the reader here is a LOWERED plan, whose read lands in `getInputStorage` and throws. Both
claim guards miss it: `graphHeldAt` is FALSE (the retention clone is not minted until backward runs) and
the witnessed cross-plan-consumer edge is never recorded (the lowered read throws BEFORE `noteWitnessRead`
can fire — a chicken-and-egg). The recorded build masked this by keeping the value in its harvest;
`guardMiss` was demoted to a should-never-fire assertion on exactly the "overlay claims are always sound"
premise the recorded build silently guaranteed. Confirmed: pure-lowered (`COMPILED_PLAN=0`, the semantic
reference) PASSES the same test — it re-produces the value in a middle plan and protects it — so the bug
is the compiled overlay-release, not liveness. FIX (`op-dispatch.ts` `getInputStorage`, commit
`f2256fe2`): when a pending producer's result is missing, RECOMPUTE it from its still-live inputs —
exactly the value a pure-lowered run materialises — declining async/stateful producers (adamStep/transfer
are never overlay-claimed, so a miss there stays loud). This is the lowered analog of the compiled
bind-time recovery: refusal-to-compile is now correct-and-slow, never corrupting. The fall-through
materialises what pure-lowered would, so the sunset no longer depends on ANY op's coverage — the
batched-reduction / no-storage / untracked-input ops stay UNCOVERED on purpose and are read correctly.
Gates: deleted-tree distil-ft **6/6** (was 1 fail — the #7 twin), #7 profiler distil@512 exit 0,
witness crux scaler-inf **6 producers / threw=false / inputNotReady=0 / shadowEmpty**, checkpoint
threw=false; NULL on main: distil-ft 6/6, test:gates 6/6, parity-fullstack compiled-vs-lowered worst
|Δ|=**5.72e-6/30** (recompute never fires — the harvest keeps the value; op-dispatch.ts survives the
deletion unchanged). `d4-deleted-tree-gate.sh` extended: `op-dispatch.ts` added to COVER_FILES + a
`recomputeMissingResult` sentinel.

**TARGET B — the foreach lowered-convergence OOM is a PRE-EXISTING pool-churn bug, NOT the unbudgeted
per-position arena the charter assumed; STOP.** The `t-train-tape-matrix` FOREACH cells (FUSED=0) OOM a
32 GB V100 by ~step 9 (~3 GB/step); the FUSED cells PASS (their adamStep compiles build-from-IR → tape
forms, `eligiblePairs=6 tapeCount=1`). The charter's prescribed fix — "reuse arena-reclaim / ARENA_LIVENESS
at the converged-to-lowered K_w point (`destroyArena`)" — was implemented and measured NULL: the converged
branch fires (fp=0xfa7c442, 354 resolve buffers) but `destroyArena` is a no-op there because at
end-of-plan the buffers are still live (`canRecycle=false`), so it only FORGETS them (leak) instead of
freeing. `TORCHLETTE_DEBUG_BIGALLOC` named the real growth: **10× 328 MB fused-kernel outputs per step**
(the foreach optimizer's m/v-class buffers) allocated via `resolveOutputBuffer → createTrackedBuffer`
(the POOL, not the arena) with NO cross-step hint reuse — the in-place m/v migration defeats
`outputSequenceHints`, so each step allocs fresh and the old ones are not promptly reclaimed. This is
`buffer-arena.ts`'s own documented "foreach 328 MB m/v leaked +640 MB/step" class, whose prior
pool-acquire fix was REVERTED for causing slot-aliasing training divergence. Decisive discriminator:
the OOM reproduces under pure-lowered (`COMPILED_PLAN=0`) AND without checkpointing — so it is a property
of the lowered REFERENCE path itself, merely MASKED on the clean tree because the recorded build was the
only compiler that ever cut the foreach plan over to compiled replay (bounding it). The charter's Target-B
premise ("reuse arena-reclaim machinery, no new mechanism class") is therefore FALSIFIED by measurement:
the growth is not arena-owned, so no reclaim reaches it; a correct fix needs NEW mechanism (cross-step
buffer stability for the in-place-migrated foreach m/v, or a fence-safe reclamation of the churned pool
outputs) — and the CLAUDE.md invariants explicitly forbid the obvious shortcuts (no mid-step pendingRelease
flush; no pool-acquire into arena positions → both are the corruption class). Per the STOP discipline
(don't improvise a memory fix that risks the silent-corruption class), Target B is a SEVENTH-class-shaped
boundary: reported precisely, not force-fixed.

**Re-open condition (attempt #9's charter).** Bound the foreach optimizer's lowered-path memory so
`t-train-tape-matrix` FOREACH (FUSED=0, ±cosine) runs within the 32 GB budget — either (a) make the
in-place-migrated m/v buffers reuse across steps (stable `outputSequenceHints`), or (b) fence-safely
reclaim the churned 328 MB fused outputs — WITHOUT the reverted pool-acquire aliasing or a mid-step
pendingRelease flush (both are the deterministic-corruption class). Gate (deleted tree, serial): the four
`t-train-tape-matrix` cells 4/4 (a tape forms AND no OOM). Target A is DONE and independent; the deletion
stays blocked ONLY on this pre-existing foreach-memory issue, which is a lowered-reference-path bug the
sunset merely exposes — arguably fixable and landable on `main` FIRST (it reproduces there under
`COMPILED_PLAN=0`), decoupled from the sunset. META-LESSON (eighth time): the charter's named remediation
was itself a hypothesis — measurement falsified "reuse destroyArena"; the growth was pool-churn, not
arena warmup. And the materialization residue (Target A) fell in one seam once traced to the actual
overlay-release clear rather than chased through the harvest/witness derivation.

**STATUS 2026-07-17 — D4 attempt #9: the foreach-OOM half is FIXED (18/18 steps, OOM=no) so the blocker
SHIFTED to compilation coverage as the #8 note predicted; the Stage-1 census closed ONE of three labels
the established way; the residual TWO require a NINTH-class chunked-elementwise stream generator — STOP,
deletion uncommitted.** The Stage-1 census (`TORCHLETTE_STEP_TAPE=record TORCHLETTE_STREAM_GENERATE=1
FUSED=0 SCHED=0 t-train-tape-matrix`, sivri VULKAN 0, serial) named the foreach optimizer plan's
uncovered map EXACTLY — **`op:where×2`, `op:cat×2`, `fused[oversized]×8`** — and each was assessed
against the established closure shapes:

- **`op:where` (ternary select) — CLOSED, established shape (differential-clean).** The generateSequential
  input-resolution loop is already arity-generic; `where` only bailed at the unary/binary classification.
  A `planWhereDirect` (mirroring `whereDirect`: whereWGSL key + `params(size, ...offsets)`, dispatchX=
  ceil(size/WG), dispatchY=1 — single-source at the seam, like `planBinaryDirect`) plus a 3-input branch
  covers it; a narrow-fed input (varying offset, no volatile repack) stays uncovered. Result: the foreach
  census drops `op:where` (covered 679→681 actions, **0 diverged**), `t-stream-generate` **0 diverges**,
  `test:gates` **6/6**. This is the CLAUDE.md ternary-op target #2 for the stream-gen path. LANDED
  standalone (does not unblock #9 on its own; the residual two labels still strand the plan lowered).
- **`op:cat` (foreach grad/param packing) — coverable in ISOLATION but INTERDEPENDENT.** A `generateCat`
  mirroring the backend `cat` byte-for-byte (resolveOutput ALLOC + one TAG_COPY per input slice) makes
  the cat segments themselves verify, BUT covering cat WHILE the oversized fused middle stays phantom
  produces **77 `stridedScatterCopy` "copy slot bijection" divergences**: the packed buffer flows
  cat → oversized-fused (IN-PLACE, same buffer) → scatter-back, and a REAL cat output slot beside a
  PHANTOM in-place fused output slot breaks the buffer-identity chain the phantom mechanism kept
  consistent when BOTH were phantom. cat cannot land without the oversized coverage. REVERTED.
- **`fused[oversized]` — the NINTH CLASS: a chunked-elementwise stream generator (new architecture).**
  Measured directly (`TORCHLETTE_DBG_OVERSIZED`): the foreach packed buffer is **320 MB** (distilgpt2's
  ~82 M-param group, f32), and every oversized fused group decomposes to **312 MB elementwise nodes**
  (`add`/`div`/`sub`/`mul`/`sqrt`/`neg`). The executor's oversized fallback is `executeSequentialSegment`
  → `executeNode` per node → `dispatchBinary`/`dispatchUnary`, which at >128 MB take the **CHUNKED** path
  (`binaryChunked` / `dispatchChunked`, sub-range storage bindings per chunk). `planBinaryDirect`/
  `planUnaryDirect` BAIL at >128 MB by construction, so `generateSequential` cannot cover the decomposition.
  There is **no chunked-elementwise stream generator** — the ONLY chunked generator is `generateFullReduction`
  (chunked *sum*, a reduction with accumulate semantics, `planChunkedFullReduction`); chunked elementwise has
  a distinct command shape (per-chunk sub-range bindings + broadcast-scalar handling + per-chunk params) and
  NO plan-extraction counterpart. Building `planChunkedBinary`/`planChunkedUnary` + threading the multi-node
  in-place group decomposition is a net-new generation SUBSYSTEM — not a CONTIGUOUS_OPERANDS declaration, not
  a copy-prologue, not an all-dims fall-through, not a constFill, not a bespoke single-op generator. It is the
  ninth never-before-exercised class, so per the STOP discipline the deletion stays uncommitted.

**Assumption cross-check.** The finiteness assumptions (§6.2) are UNTOUCHED — this is the same OUTSIDE-the-
frame structural claim the sixth class exposed (compilation coverage is load-bearing, not a pure
optimization), now hitting the foreach optimizer's oversized-packed decomposition. The `where` closure and
the empirical census confirm the two smaller labels are established-shape / interdependent; only the
chunked-elementwise coverage is new architecture.

**Re-open condition (attempt #10's charter).** Cover the oversized foreach optimizer decomposition WITHOUT
new corruption-class risk — either (a) a chunked-elementwise stream generator (`planChunkedBinary`/
`planChunkedUnary` mirroring `binaryChunked`/`dispatchChunked`'s per-chunk sub-range bindings, following the
`planChunkedFullReduction` precedent for the reduction case), which also unblocks the interdependent `cat`
(both land together so the packed-buffer identity chain is consistent); or (b) bound the foreach packing so
no group's packed buffer exceeds 128 MB (an optimizer-level packing change that keeps every elementwise op
on the direct path — NOT a stream-generate change, and it alters the packing semantics, so gate it against
the foreach numerical/memory baselines first). Gate (deleted tree, whole-node exclusive, serial): the four
`t-train-tape-matrix` cells 4/4 (`eligiblePairs>0`, `tapeCount=1`, no OOM) AND `t-stream-generate` 0
diverged AND the foreach census fully covered (`fused[oversized]`/`op:cat` gone). META-LESSON (ninth time):
the OOM fix worked and the memory blocker dissolved exactly as #8 predicted — but the coverage residual it
UNMASKED is not established-shape; the foreach optimizer's >128 MB packed buffers make its whole elementwise
update a chunked-dispatch class the generate-from-IR subsystem has never had to cover (the fused path escapes
it via the single `adamStep` kernel, which is neither chunked nor decomposed).

**STATUS 2026-07-17 — D4 attempt #10: the #9 census residual is CLOSED (the foreach optimizer plan is
FULLY covered, `t-stream-generate` 0 diverged, the tape forms); the "ninth class" was substantially a
STALE HARDCODED-128 MB threshold, not a missing chunked-elementwise subsystem — MEASURED. The deletion is
NOT yet landed: it now needs a fresh 3-way reconciliation of the coverage against the preserved deletion
diff (they touch the same files) + the full authoritative matrix — Stage 1 landed and validated, Stages
2–3 pending.** The Stage-1 coverage (commit `1d74b70c`, worktree off `4e06b015`, +762 lines across
`dispatch.ts`/`tile-dispatch.ts`/`executor.ts`/`segment-executors.ts`/`stream-generate.ts`, weight-norm
srcSLOC 65327→65890):

- **THE DERIVED CHUNKING SEAM (the ratified refinement, landed).** `computeChunkGeometry` (tile-dispatch.ts)
  is the ONE capability-forced split `numChunks = f(bytes, maxStorageBufferBindingSize)` deriving per-chunk
  sub-range windows + grid + packed uniforms; `dispatchChunked` (execution) and the new `planChunked`
  (generation) BOTH consume it (dispatchChunked refactored to derive from it — cannot drift).
  `chunkedBinaryConfig`/`chunkedUnaryConfig` are the single spec+chunking source; `planChunkedBinary`/
  `planChunkedUnary` expose the plan; `dispatchChunkedElementwise` executes it with a FRESH per-chunk
  `createParamsBuffer` (a self-contained `params` slot, NOT the config-cache `persistent` buffer a
  build-from-IR replay cannot own — the `sumFullReductionChunked` precedent, the reason chunked elementwise
  is generatable). `generateChunkedBinary`/`Unary` + `generateFusedDecomposed` (the `executeSequentialSegment`
  fallback) + `generateCat` (the interdependent op:cat) are the generator side. Chunk geometry is baked as
  per-chunk params (step-invariant for a fixed buffer, no volatile repack — following
  `planChunkedFullReduction`); the "volatile-uniforms" phrasing in the charter was over-specified — the
  chunk SIZE already flows as data in the packed params, and a fresh params slot per chunk is what the
  execution's per-chunk `createParamsBuffer` records.

- **THE MEASURED FINDING that reframes the class.** On the authoritative device (V100 via the `tools/vk-shim`),
  `maxStorageBufferBindingSize = 2048 MB`, so the 320 MB foreach packed buffer is **NEVER chunked** — it binds
  whole and runs as a fused kernel. The `fused[oversized]×8` census label came from a stale HARDCODED 128 MB
  check in `generateFused` that DISAGREED with execution (which uses the device limit in `executeFusedSegment`'s
  `hasOversizedBuffer` check + `dispatchBinary/Unary`). The real unblock was (a) deriving that threshold from
  the device limit (single-source with execution), (b) `generateCat`, and (c) caching the executor's donation
  decision on the fused action (`donationSink → action.cachedDonatedRecipeIdx`) — the generator's post-hoc
  donation detection reads the primary output's result, already cleared for a step-scoped in-place output
  (the foreach in-place m/v update; this was the node[632] `mul+add` gen=4/rec=3 divergence + the downstream
  node[635] `stridedScatterCopy` slot-bijection divergence, both now 0). The chunked-elementwise generator IS
  the correct handling for genuinely-oversized ops (>2 GB, or a 128 MB-limit device) and is single-sourced,
  but it is a NO-OP on the gate device; so the #9 charter's "net-new chunked-elementwise SUBSYSTEM is required"
  premise is FALSIFIED for the authoritative device — the residual was a threshold + cat + donation, all
  established shapes.

- **Gates (V100 vk-shim, device 5, current tree):** `test:gates` **6/6** (incl. the chunked full-reduction
  sum); `parity-fullstack-tl` compiled-vs-lowered ~1e-6/30 BOTH directions; `t-stream-generate` **0 diverges /
  3100 verified cmds**; all four `t-train-tape-matrix` cells **PASS** (`refusals=0`, `eligiblePairs>0`), every
  plan FULLY GENERATED **0 diverged** (foreach: 383/383 segments, `fused[oversized]`/`op:cat` GONE).

- **Re-open condition (attempt #11 / the landing).** NOT a tenth class — no new never-witnessed cross-plan
  boundary; the #9 re-open gate (`t-train-tape-matrix` 4/4, `t-stream-generate` 0 diverged, foreach census
  fully covered) is MET. What remains is the LANDING: (1) re-reconcile `.claude/D4-deletion-attempt5-STOPPED.diff`
  against this HEAD — the deletion touches `dispatch.ts`/`tile-dispatch.ts`/`executor.ts`, the same files Stage 1
  modifies, and `stream-generate.ts`'s chunked generators import Stage-1 `dispatch.ts` exports, so the mechanical
  `d4-deleted-tree-gate.sh` COVER_FILES layering (plain `git apply`) cannot build the deleted tree without a real
  3-way merge; (2) run the full §D4 authoritative matrix on the reconciled deleted tree, serial/whole-node
  exclusive; (3) commit the deletion only if ALL green. The `d4-deleted-tree-gate.sh` COVER_FILES must gain
  `dispatch.ts tile-dispatch.ts segment-executors.ts` (and the executor donation-write must survive the deletion
  reconciliation) for the deleted tree to build + reproduce the donation coverage.

**STATUS 2026-07-17 — D4 attempt #11 (THE LANDING PASS): STOPPED. The reconciliation LANDED and is
verified; the deletion is UNCOMMITTED. A real regression class surfaced on the FIRST-EVER measurement of
the #9/#10 re-open gate ON THE DELETED TREE — attempt #10's "foreach fully covered / 4/4 / `t-stream-generate`
0 diverged" was measured on HEAD (`7c3f319e`) WITH THE RECORDED BUILD STILL PRESENT and attributed to
build-from-IR; on the deleted tree the recorded build turns out to be the sole compilation source for the
foreach elementwise optimizer plan.**

- **Stage 1 (reconciliation) — DONE + verified.** 3-way merge of `.claude/D4-deletion-attempt5-STOPPED.diff`
  (native base `71655ea8`) against HEAD `7c3f319e`; only `tile-dispatch.ts` + `executor.ts` truly conflicted.
  Resolutions: (a) `tile-dispatch.ts` chunked loop — keep attempt #10's derived `geom.chunks` seam, drop the
  deletion-target `chunkRepack`/volatile machinery; (b) `executor.ts` — drop the 4 recording imports + `TAG_DISPATCH`
  (sole use was in the deleted `compilationRecording` finally block), keep `crossPlanEdgeHasOtherConsumer` (live at
  the overlay-decline seam ~L2256), drop `crossPlanEdgeKeepSet` (dead), and RESTORE `currentVariantSelection` which
  the 3-way cleanly removed but which attempt #10 made load-bearing in surviving donation code (~L2259) — the
  recorded-build call-site semantics were the reference. The executor DONATION-WRITE (`donationSink →
  action.cachedDonatedRecipeIdx`) SURVIVES. Reconciled diff saved to `.claude/harvest-deletion-final.diff` (applies
  cleanly to `7c3f319e`). `npm run build` green; `tsc --noEmit` ZERO net-new errors (net −3: three pre-existing
  unused-symbol errors resolved). Weight-norm: srcSLOC 66381→65559 (**−822**), docLOC −350, **envFlags 69→65**
  (`TORCHLETTE_BUILD_FROM_IR` [the named opt-out dies], `TORCHLETTE_COMPILED_MAX_NODES`,
  `TORCHLETTE_COMPILED_ONLY_NODES`, `TORCHLETTE_DEBUG_PERSISTENT`). `d4-deleted-tree-gate.sh` reworked to build the
  deleted tree as `PREDELETION_SHA(7c3f319e) + harvest-deletion-final.diff` (the header's ORIGINAL "apply at HEAD"
  intent) — the base+layer contraption was fatally base-drifted: HEAD's `stream-generate.ts` imports `planWhereDirect`
  from `ops/where.ts`, an export attempts #7–#10 added to a NON-covered file, which broke the throwaway build; and
  `tile-dispatch.ts`'s HEAD seam OVERLAPS the deletion so plain-layer `git apply` cannot apply it.

- **Load-bearing correctness on the DELETED tree — ALL GREEN.** `test:gates` **5/5** (the stream-gen-vs-recording
  gate is correctly RETIRED with the recorded build; the "build twice → identical" determinism gate is its sufficient
  successor — it passed); `parity-fullstack-tl` compiled-vs-lowered **6.68e-6 / 30 both directions**; full suite (CPU
  **1426 passed**; webgpu **860 passed** + 89 CONTENTION-FABRICATED failures that ALL pass in isolation on an idle
  device — 7 files / 89 tests re-run green); witness matrix **5/5** (`scaler-inf` 6 producers threw=false,
  `checkpoint`/`medium`/`chunked124m`/`lr-milestone` all exit 0); `t-step-object-null`, `t-step-edit-null`,
  `t-train-capture-probe`, `t-planner-pin-attribution` (arena/free/--assert), `t-ledger-attack-probe` (24 + 48) all
  exit 0. The recorded build's deletion does NOT break correctness anywhere measured.

- **THE STOP — recorded-build-only compilation coverage of the foreach elementwise optimizer plan.** Differential,
  clean/isolated devices, deterministic (reproduced by the d4 gate's independent tree construction AND a direct
  working-tree run):

  | check | MAIN `7c3f319e` (recorded build present) | DELETED tree (build-from-IR only) |
  |---|---|---|
  | `t-train-tape-matrix` fused+nosched / fused+cosine | eligiblePairs=6 / 6 PASS | 6 / 6 PASS |
  | `t-train-tape-matrix` foreach+nosched / foreach+cosine | eligiblePairs=4 / 4 PASS | **0 / 0, loweredPairs=6 FAIL** |
  | `t-ring-probe` (capture/ring engagement) | hits>0, PASS | **hits=0, K1/K2/K3 FAIL** |
  | `t-stream-generate` | hook-fires=5, 3100 cmds, 0 diverged PASS | **hook-fires=0 FAIL** |

  The FUSED Adam path (the DEFAULT optimizer) IS covered by build-from-IR and keeps its tape. The FOREACH (elementwise,
  non-fused) optimizer plan is covered ONLY by the recorded build; build-from-IR does not compile it, so on the deleted
  tree it runs LOWERED forever. This is numerically CORRECT (ring losses byte-identical across serial/ringNow/K1/K2/K3,
  `maxΔ=0.00e+0`; foreach loweredPairs run 18/18 steps, OOM=no) — a coverage/capability regression ("correct-and-slow"),
  not a correctness bug. But it fails the EXPLICIT Stage-2 requirement "`t-train-tape-matrix` 4/4 (foreach: tapes form)"
  and the `d4-deleted-tree-gate.sh` #9 cells. `t-stream-generate`'s failure is EXPECTED — it tests the deleted
  `STREAM_GENERATE` verify apparatus and is now obsolete (successor = `test:gates` determinism gate); it should retire
  with the apparatus.

- **Root of the misattribution (the campaign's own recurring lesson).** Attempt #10's re-open-gate greens were measured
  on the tree WITH the recorded build; they never ran on the deleted tree ("the deletion is NOT yet landed… Stages 2–3
  pending"). Component isolation cannot exonerate the deleted tree — the recorded build was silently supplying the
  foreach optimizer plan's compilation the whole time. build-from-IR foreach optimizer coverage is NOT actually landed.

- **Re-open condition (attempt #12).** Make build-from-IR COMPILE the foreach elementwise optimizer plan on the DELETED
  tree so `t-train-tape-matrix` foreach+nosched/foreach+cosine reach `eligiblePairs>0` and `t-ring-probe` re-engages
  (hits>0) WITHOUT the recorded build — then re-run the full authoritative matrix on the deleted tree and land. Retire
  `t-stream-generate` (obsolete apparatus). The reconciliation + gate rework of this pass are preserved
  (`.claude/harvest-deletion-final.diff`, the reworked `d4-deleted-tree-gate.sh`) and need no redo.

**STATUS 2026-07-17 — D4 attempt #12 (THE DELETED-TREE FOREACH CENSUS): the foreach plan's uncovered labels are
MEASURED for the first time on the true deleted tree, and BOTH are established-shape — CLOSED and PROVEN
byte-identical (the fullstack tape matrix is now 4/4 on the deleted tree). BUT the full #12 re-open gate has a
SECOND, DISTINCT blocker that attempt #11 mis-bundled under "foreach": `t-ring-probe`'s synthetic tiny fused-Adam
graph strands on a BROADCAST-into-fused view, the deliberately-excluded strided-view class — STOP on that residual,
deletion NOT landed.** Measured on the reconciled deleted tree (`7c3f319e` + `harvest-deletion-final.diff`),
sivri VULKAN device 3 (== nvidia 3, identity map; free), GPU-exclusive serial, via a temporary `gen.uncovered`
dump in the surviving build-from-IR block (`TORCHLETTE_STREAM_GENERATE` is deleted; instrument-and-revert).

- **THE FOREACH CENSUS (definitive).** The ONLY recurring, structurally-stable plan that strands lowered is the
  foreach elementwise optimizer plan `fp=0xfa7c442` (reach climbs 1→16 every step; every other uncovered census
  entry is a reach-1 warmup graph variant or a checkpoint-re-fingerprinting forward/backward plan → `structureMisses`,
  not `loweredPairs`). Its steady (reach≥2) uncovered map is EXACTLY two labels, `covered 613/691`:

  | label | count | root |
  |---|---|---|
  | `op:stridedScatterCopy[no-storage]` | 76 | scatter-back src `reshape(narrow(pNew, 0, offsets[k], sizes[k]), shape)` (adam.ts:448) — a RELEASED contiguous view with a non-zero packed-buffer offset; `refSlotAndMeta` bailed all released VIEW_OPS |
  | `op:cat[no-storage]` | 2 | grad/param packing `cat(reshape(param, [size]))` (adam.ts:370/374) — released contiguous reshape input to `generateCat` |

  (At reach=1 only, warmup `fused[no-input-pattern]=50`/`data-source:full=2` also appear; they vanish by reach≥2 and
  never block eligibility.) The `fused[oversized]×8` / `op:cat` interdependence the #9/#10 STOP feared is GONE on the
  deleted tree — attempt #10's stale-128 MB-threshold fix holds without the recorded build (device
  `maxStorageBufferBindingSize = 2048 MB` ⇒ the 320 MB packed buffer binds whole).

- **VERDICT — established-shape; CLOSED + PROVEN.** Both labels are the SAME class: a released CONTIGUOUS view
  (`reshape`, optionally over a dim-0 `narrow`) whose layout is fully IR-derivable — `contiguousViewShapeDtype` /
  `CONTIGUOUS_VIEW_OPS` already exist and `generateScatterAdd` already consumes exactly this precedent. The slot
  aliases the base buffer (views alias to their input's slot); only the ELEMENT OFFSET (the narrow `start`, in the
  IR payload) had to be recovered — the DMA scatter route (`generateScatterCopyDMA`) already threads `srcOffset`.
  Fix: a `releasedContigViewOffset(ref)` helper (walks the contiguous-view + dim-0-narrow chain accumulating
  `start·innerSize`, mirroring `stridedScatterImpl`), wired into `refSlotAndMeta` and `generateCat` (stream-generate.ts
  only; the deletion does not touch this file). Result on the deleted tree: `fp=0xfa7c442` FULLY covered, all four
  `t-train-tape-matrix` cells PASS (`eligiblePairs=6`, `loweredPairs=0`, `tapeCount=1`), foreach loss trajectory
  **BYTE-IDENTICAL to lowered (`COMPILED_PLAN=0`) over 12 steps at 8-decimal precision** — the anti-silent-corruption
  gate. Preserved as `.claude/d4-12-foreach-closure.diff` (124 lines, stream-generate.ts, applies clean to `9e60023e`
  and to the deletion base; NOT landed — the landing is a fresh pass on the full green report per the campaign pattern).

- **THE STOP — a SECOND, non-foreach uncovered class the #11 table mis-attributed.** The #12 re-open gate also
  requires `t-ring-probe` (hits>0). `t-ring-probe` uses FUSED single-weight Adam (NOT foreach); on the deleted tree it
  fails `hits=0` for a class ORTHOGONAL to the foreach fix (unchanged by the closure). Its recurring uncovered map:
  `fused[no-storage]` (90×) = a released `expand` broadcast `[1,1]→[1,2]` fed to a fused elementwise kernel; and
  `op:stridedScatterCopy[scalar-operand]` (5×) = a scatter from a SCALAR source into `[1]`. The broadcast-into-fused is
  the deliberately-excluded STRIDED/broadcast-view class: even a LIVE broadcast bails `non-contiguous`
  (stream-generate.ts:3743) and is materialized by an executor copy-prologue, so covering the RELEASED case needs a
  BROADCAST-copy-prologue synthesized FROM IR (input-shape + broadcast dims → contiguous materialization) — NOT the
  contiguous-view shape, NONE of the shapes #7–#10 used. #11 attributed ALL deleted-tree reds to "the foreach optimizer
  plan"; the census shows `t-ring-probe`'s red is a different plan (a tiny synthetic graph), while the REAL training
  workload (fullstack fused AND foreach) now compiles fully (tape matrix 4/4).

- **Sizing for the residual (Vin's fork).** (a) `stridedScatterCopy[scalar-operand]`: a scalar-source scatter
  generator (constFill-shape into a base slice) — ~20–30 SLOC, low risk. (b) `fused[expand-broadcast]`: a
  broadcast-copy-prologue-from-IR (materialize the released `expand`'s [1,1]→[N] contiguous, mirroring the executor's
  `asContiguous`/#99a copy-prologue but deriving the broadcast layout from the IR view chain instead of live storage)
  — ~40–80 SLOC + a byte-identical cross-path gate; this crosses the strided-view line the campaign held for 10
  attempts and carries the silent-corruption risk of any offset/stride reconstruction. Alternative to (b): re-scope
  `t-ring-probe` to a build-from-IR-covered graph (change the PROBE, not the framework) — it is a synthetic
  ring-mechanism test, and the ring works on real models. **FORK:** close the two ring-probe labels (small, but (b) is
  new strided-view capability) vs re-rule the bar for `t-ring-probe` (the real training arms are now byte-identical
  correct-AND-compiled) vs re-scope the probe. The foreach half — the named #11 blocker — is dissolved; only this
  synthetic-probe residual stands between here and the authoritative matrix.

- **Re-open condition (attempt #13 / the landing).** Resolve the fork above; if closing: land the two ring-probe
  labels the established way (with a byte-identical gate for the broadcast prologue), fold `d4-12-foreach-closure.diff`
  into `harvest-deletion-final.diff` (stream-generate.ts is untouched by the deletion, so it layers cleanly), then run
  the full §D4 authoritative matrix on the deleted tree (tape matrix 4/4 ✓ already, `t-ring-probe` hits>0, d4 gate) +
  the standing wall on main (test:gates 6/6, parity ≤1e-5/30, witness 5 cells, 124M exact, profile distil) and land.

**STATUS 2026-07-17 — D4 attempt #13 (THE LANDING): DONE. The recorded build is DELETED (commit
`43272c47`, net −822 srcSLOC / −4 envFlags, on top of the Stage-1 closures `39debb7d`); the full
authoritative matrix ran GREEN on the deleted tree BEFORE the commit (sivri V100, vk-device 2 ==
nvidia 2 identity-mapped, serial GPU-exclusive, under the degraded-node regime with
environmental-first triage).** Vin ratified fork (c): the two Stage-1 closures landed the
established way (the #12 `releasedContigViewOffset` foreach closure; the scalar-source scatter via
the plan scalar table — per-step-refreshed persistent buffer, no frozen-scalar risk), and
`t-ring-probe` was RE-SCOPED to a build-from-IR-covered graph (mean-as-matmul; its purpose is
RUNAHEAD-ring verification, and the old mean-loss graph accidentally depended on the
deliberately-uncovered released-expand-broadcast class — the probe file states this scope). The
broadcast-into-fused class stays DELIBERATELY UNCOVERED.

- **Matrix (deleted tree):** `t-ring-probe` PASS (hits=16, serial==ringNow==K1==K2==K3 at 0.0 —
  the #12 STOP dissolved); tape matrix 4/4 (`eligiblePairs=6 loweredPairs=0 tapeCount=1` in all
  cells); `test:gates` **5/5** (the stream-gen-vs-recording gate retired WITH the apparatus;
  build-twice determinism is the successor and passed); parity-fullstack compiled-vs-lowered
  **3.815e-6/30 both directions**; witness matrix 5/5 shadow-empty (scaler-inf 6 producers/459
  pairs `threw=false`); step-object/edit nulls + capture probe PASS; planner-pin
  arena/free/--assert PASS; ledger-attack 24+48 `totalDrift=0`; CPU suite 1416/0; webgpu suite 880
  passed with ONLY the environmental `vkCreateDevice` device-chain class (a different random file
  set each run; all 14 affected files pass isolated, 255/255 — the named non-class); **124M DiLoCo
  EXACT** {0:9.8089, 3:5.9220, 6:5.1536, 9:4.6405}, peak flat 2087.6 MB; profile-training
  distil@512 AND medium@512 exit 0 with peaks **BYTE-IDENTICAL to the pre-deletion null**
  (6552.0 / 22098.4 MB — the deletion is memory-neutral); checkpoint ab-oracle WITNESS side PASS
  both modes (memory side FAILs by D3-STOP design, unchanged); `d4-deleted-tree-gate.sh` ALL cells
  PASS on its independent tree construction (`PREDELETION_SHA=39debb7d`).

- **Finiteness — final form (capstones).** Assumption 3 in its final reading: witnessing
  completeness is delivered by converged-to-lowered stamping (K_w=2), not by any extra pass — the
  deleted-tree scaler-inf derivation is the standing measurement. The three structural capstones
  that made the sunset safe: (1) the ZERO-RESIDUE fall-through (`recomputeMissingResult`,
  coverage-independent — refusal-to-compile can never corrupt, only slow); (2) the DERIVED-CHUNKING
  seam (`computeChunkGeometry` single-sources dispatch and generation — capability-forced splits
  cannot drift); (3) the DELIBERATELY-UNCOVERED enumeration (released strided/broadcast views —
  expand-into-fused, narrow dim≠0, transpose/permute without captured layout; released-input
  scatterAdd/CE; non-f32 scalar-source scatter; batch>64 / non-f32 / CoW plans) — all of which run
  lowered forever and correctly by (1).

- **The d4 gate RETIRES (deleted with the docs commit).** Justification: it is pre-commit-only BY
  CONSTRUCTION (it builds `PREDELETION_SHA + harvest-deletion-final.diff` and aborts when HEAD
  already carries the deletion); every cell it gated has a standing successor that runs on plain
  HEAD — witness matrix (crux cells), `t-train-tape-matrix` (#7/#9 cells), `profile-training`,
  `test/distilgpt2-full-finetuning.spec.ts` (in `npm run test`). Keeping it would preserve a
  script whose tree-construction premise no longer exists.

- **`.claude` diff artifacts — superseded (cleanup note).** `D4-deletion-attempt5-STOPPED.diff`,
  `harvest-deletion-43a.diff`, `harvest-deletion-43a-reconciled.diff`, `harvest-deletion-final.diff`,
  and `d4-12-foreach-closure.diff` are all now IN GIT HISTORY (`39debb7d` + `43272c47`); the
  untracked artifacts in the main checkout's `.claude/` are historical and safe to delete.

### Phase D5 — The declared-lifetime dividend (DONE ✓ 2026-07-17; step-object phase 7)

**Goal:** the observation predicates RETIRE on the captured path (they are now queries of
`crossPlanEdges(v)`, §4.2). Gated on ruling 2's own re-open condition (captured path warm + watcher
cost measured).
- **Gate:** set-parity of the derived liveness vs the observation layer on the captured path;
  measured watcher-cost delta > derivation cost; captured-path only (LOWERED keeps all three).
- **Deletes:** `everReadback` / `everSurvived` / `everAliased` on the captured path.

**EXECUTED — the measurement reshaped the deletion into a retirement.** The re-open condition was
honored, not assumed: `tools/t-d5-watcher-cost.ts` + a `TORCHLETTE_MEASURE_D5` claim-seam probe
measured what the three predicates prune that the derivation (`crossPlanEdges` + `graphHeldAt` +
declared survivors) wouldn't, across distil{ckpt on/off} + gpt2-medium × captured/uncaptured.

| id | finding | source |
|---|---|---|
| E-5 | On the CAPTURED path the predicates prune NOTHING: the convergence machinery they gate never activates under step-tape replay (a warm tape stops executing lowered, so no template reaches the `K_HYSTERESIS` stable *executed* steps convergence needs — convergedTemplates=0 / releasableMB=0 / 0 predicate prunes, even at 60 steps / 56 hits). The claim seam + read guards + overlay-release were already dormant/suppressed on replay. | `t-d5-watcher-cost.ts CAPTURE=1` |
| E-6 | Off-capture the predicates DO converge (8 templates) and `everSurvived` carries exactly ONE real prune the derived guards miss: the `[1,S,vocab]` CE-logits boundary survivor (survives the step with no cross-plan consumer edge; declared-live under capture, observation-only without a declaration). `everReadback`/`everAliased` carry zero real prunes but stay uncaptured (unordered readback). | `t-d5-watcher-cost.ts CAPTURE=0` |
| E-7 | The captured trajectory is BIT-IDENTICAL pre/post retirement (distil last-loss 11.0766, every per-step value unchanged); captured-vs-uncaptured tracks within the fp-noise floor (max Δ ~1e-3 nats). | `t-d5-watcher-cost.ts` A/B |

**Consequence (assumption cross-check).** E-6 is why D5 is a captured-path RETIREMENT, not a global
DELETION — a full deletion would drop a live edge the derived set cannot see on the uncaptured path
(the same OUTSIDE-the-finiteness-frame shape as D4's sixth class: the derivation is complete only
under a DECLARATION). So: the four observation-RECORDING hooks (`stampResult`/`observeConsumed`/
`observeReadback`/`noteAliasedBase`) no-op under `isStepTapeReplayActive()`; the UNCAPTURED path
keeps the full observation layer; strict-lifetime stays armed as the wrong-derivation detector.
"Partial dividend beats wrong dividend." `t-d5-watcher-cost.ts` is the standing re-open-condition
regression gate.

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
