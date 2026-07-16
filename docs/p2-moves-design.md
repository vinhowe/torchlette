# P2 macro-moves: the merge/fuse composite transaction and the stream move

**Status:** design + thin prototype (2026-07-12). The two NEEDS-DESIGN items between
the P0-full engine (waves 1–3 landed) and a real flashattention derivation, per
`docs/schedule-state-wave3-attention-report.md §6` items (a) and (b).
**Depends on:** `docs/schedule-state-design.md` v2 (§3 moves, §7 P2, §11 ownership table),
`docs/design-amendment-round-1.md` (S3 composite-transaction ruling), `docs/islands-design.md`
(§2 merge/split contract — the ONLY partition mutators), `examples/schedule-editor/contract.md`
+ `workbench-contract.md` (the editor-side mutation contracts these compose with),
`examples/schedule-editor/src/lib/ncd/model.ts` (the client-side streamability predicate this
supersedes).
**Prototype:** `src/schedule/moves/streamability.ts` + `test/schedule/moves/streamability.spec.ts`
(schema-level, no dispatch).
**Acceptance artifact:** `tools/fa-derivation-script.ts` (the FA move-script; spec, not yet runnable).
**Scope discipline (STOP):** design-first. This doc + the streamability prototype (schema-only,
touching nothing live) + the acceptance script are the whole deliverable. NO dispatch-path
changes, NO engine `merge` mutator built, NO `applySchedule` change. The prototype implements
ONLY the predicate; the move-algebra bodies that consult it are P2 IMPLEMENTATION work this
doc specifies.

---

## 0. Where these two sit

The P2 acceptance narrative (design §7 P2) is the flashattention derivation, rungs 0–7:

```
base:  naive attention = three islands (QK^T matmul → softmax row-program → PV matmul)
  → merge   the three naive islands              [S3 COMPOSITE TRANSACTION — deliverable 1]
  → tile    the KV loop
  → stream  K/V through shared                    [stream move — deliverable 2]
  → recolor accumulator to register
  → (stream on softmax REFUSED — no head/body)    [streamability predicate — deliverable 2]
  → apply   the online-softmax admitted lemma     [F17 sequence; lemma landed wave 3]
  → stream  the now-decomposed softmax            [stream move, post-lemma — deliverable 2]
  → program-map for L2 reuse                       [landed as a move; R4 reification test]
```

Two objects gate this: (1) the `merge` composite transaction — how the editor's "fuse"
gesture composes the islands `merge` with an interior schedule attach under ONE provenance
record with atomic rollback; (2) the `stream` move's before/after schema + the ENGINE-side
streamability predicate that fires the refusal-first boundary on naive softmax and admits the
post-lemma body. Everything else on the ladder either landed (the lemma engine object, the
`program-map` move, the naive composition, the authored baseline pin) or is a straight
before/after application of a move whose schema is already in `types.ts §13`.

---

## 1. Deliverable 1 — the `merge`/`fuse` composite transaction (S3)

> **STATUS UPDATE (2026-07-16): S3 FUSE LIVE WIRING LANDED.** The schedule-altitude
> `fuseGesture` (this deliverable) is joined by the LIVE executor path: an accepted
> `StepEditChannel.requestMerge` now drives a REAL per-plan partition merge
> (`applyPartitionMerge` → the detector's `mergeIslands`, `src/executor/executor.ts` /
> `fusion-detect.ts`), re-fingerprints the plan, and re-witnesses the tape under the merged
> partition — byte-identical trajectory, rollback to identity. `ReRecordChannel` is no
> longer a pure stub at the live seam. See `islands-design.md` header + `step-object-design.md
> §5 STATUS`. The membership rule below is UPHELD: `mergeIslands` lives in the detector's file
> (the sole owner); the channel only expresses the request.

### 1.1 The ruling this implements

S3 (`design-amendment-round-1.md`, `schedule-state-design.md §3.5`): `fuse` is **NOT** a
ScheduleState move. Membership is owned by `Partition` **alone** — `merge`/`split` are the
ONLY partition mutators (`islands-design.md §2`). The editor's "fuse" gesture is a **composite
transaction at the next altitude** (islands), composing with the islands merge contract:

```
fuseGesture(P, a, b, proposedInteriorSchedule):
  1. validate  the proposed interior SemanticSchedule (core legality, §3.2)
  2. merge     P' = merge(P, a, b)            // islands-design §2 — the ONLY membership owner
  3. mint      region' = newSemanticRegionUid // the partition transaction mints the region UID (R8)
  4. attach    the interior schedule at region'
  5. record    ONE provenance entry carrying BOTH hashes (boundaryHash + the two identities §5)
  6. on realization failure: ROLL BACK all of 2–5 atomically
```

### 1.2 The transaction's API shape

```ts
interface FuseGesture {
  readonly partition: PartitionRevision;   // the base partition (islands-design lattice node)
  readonly a: IslandId;
  readonly b: IslandId;
  readonly proposedInterior: SemanticSchedule; // the interior schedule to attach at region'
}

type FuseOutcome =
  | { kind: "committed"; result: FuseCommit }
  | { kind: "refused"; stage: FuseStage; reason: string; code: FuseRefusalCode };

interface FuseCommit {
  readonly partition: PartitionRevision;   // P' = merge(P, a, b)
  readonly region: SemanticRegionUid;      // the minted region' (R8 — partition transaction mints it)
  readonly state: ScheduleState;           // the attached interior schedule, region = region'
  readonly provenance: FuseProvenance;     // ONE record, both hashes
}

/** ONE provenance record carrying BOTH altitudes' identities (S3 step 5). */
interface FuseProvenance {
  readonly boundaryHash: string;   // the islands boundary hash of P' (islands-design identity)
  readonly semanticHash: string;   // digest(SemanticSchedule + region') — §5 semantic identity
  readonly compilationHash: string;// + digest(BackendRequests) + version coordinate — §5
  /** inverse data (S3 / ownership-derivation inverse-payload discipline): what UNDO needs.
   *  Undo of a fuse is a `split` at the recovering cut PLUS discarding region' — carried so
   *  the ledger can invert without re-deriving. */
  readonly inverse: {
    readonly splitCut: NodeCut;            // islands-design split cut recovering a, b
    readonly retiredRegion: SemanticRegionUid;
    readonly priorRegions: readonly SemanticRegionUid[]; // a's and b's region UIDs
  };
}

type FuseStage = "validate-interior" | "merge" | "attach" | "realize";
```

`fuseGesture(g: FuseGesture): FuseOutcome` is a **single transaction function** — it either
returns `committed` with all of (P', region', state, provenance) or `refused` with the stage
that refused and a stable code+reason. It never returns a partial state. This is the S3
"roll back all on realization failure" made a total function: the caller (the editor) only
ever sees a fully-committed transaction or an untouched base.

### 1.3 The staged validation and the failure taxonomy

The transaction runs the S3 steps as a pipeline; each stage can refuse, and the taxonomy is
exactly which stage refused (design §3.5 names three failure classes; this maps them to
`FuseStage`):

| Stage | Gate | Refusal class | `FuseRefusalCode` |
|---|---|---|---|
| `validate-interior` | core legality of `proposedInterior` (§3.2: move-algebra laws, region convexity, staging well-formedness — backend-neutral) | **interior illegal** | `INTERIOR_ILLEGAL` |
| `merge` | islands `merge(a,b)` legality (dataflow convexity, shape/binding-count/barrier-kind/chunking — device-keyed; `islands-design.md §2`) | **merge refused** (T0 gate) | `MERGE_REFUSED` |
| `attach` | region' minted, interior attached, no-second-owner assertion (§2.6) holds on the attached state | **attach conflict** (a second owner leaked) | `ATTACH_SECOND_OWNER` |
| `realize` | the realizer accepts the merged+attached state (capability profile §4; `applySchedule` produces a lowering the realizer compiles) | **realization refused** OR **mid-transaction engine error** | `REALIZATION_REFUSED` / `ENGINE_ERROR` |

- **interior illegal** (step 1): the proposed interior schedule violates a move-algebra law or
  is not convex. Refused BEFORE any partition mutation — the cheapest failure, no rollback
  needed (nothing mutated yet).
- **merge refused** (step 2): the islands `merge` T0 gate refuses (e.g. non-convex union, or a
  binding-count budget blown on this device class). This is **device-keyed** — a merge legal on
  A100 may be refused on V100 (`islands-design.md §2` rule 3). The composite transaction does
  **not re-own** this gate; it CALLS the islands `merge` legality and surfaces its refusal.
  Nothing has been minted or attached, so rollback is a no-op.
- **realization refused / mid-transaction engine error** (step 6): the merge succeeded and the
  interior attached, but the realizer cannot emit (capability absence) or the engine errored
  mid-lowering. This is the ONLY stage that needs true atomic rollback: `split` P' back to P at
  the recovering cut, retire region', restore a's and b's region UIDs. The `inverse` payload is
  carried precisely so this rollback needs no re-derivation.

### 1.4 Interaction with the tape (fingerprint change → re-record → invalidate)

A committed fuse changes the schedule fingerprint of the affected region (a new `region'` with a
new semantic identity replaces two prior regions). Per the tape-hygiene contract and design §5's
null-stability requirement:

- The new `region'` has a **new semantic identity** (`semanticHash`), so any compiled plan /
  template keyed on the OLD (pre-merge) region fingerprints is **stale**. The commit path must
  **invalidate** those recordings — the same invalidate path a re-record triggers (design §5
  cache-class discipline; the `#92` seam-guard regenerate-and-compare fires on the new key).
- Re-record is driven by the fingerprint change, NOT by the transaction: `fuseGesture` returns
  the committed state + provenance; the executor re-records on next execution when it sees the
  new fingerprint (design §7 P0's "first exec lowered + records; swaps 2nd+ replay"). The
  transaction does not itself force a record — it changes the identity that the record path keys
  on.
- **The tape-hygiene agent owns the invalidate wiring** (capture/tape). This doc specifies the
  SEAM: the commit emits a `FuseProvenance` whose `semanticHash`/`compilationHash` are the keys
  the tape must invalidate; the inverse payload is what a tape-level undo replays. The two agents
  meet at `FuseProvenance` — it is the single record both the provenance spine (design §9) and
  the tape consume.

### 1.5 How the naive-attention island-flow edges (wave 3) feed it

`naiveAttentionComposition(headDim)` (`attention-skeleton.ts`) delivers the transaction's INPUT:
three `ScheduleState`s over three regions (`qkT`, `softmax`, `pv`) PLUS `islandFlow` — the
ordered data-flow edges `scores → softmax → P` (`{from, to, via}`). The composite transaction
consumes this as:

- the three regions are three islands `a₁, a₂, a₃` in `P` (the finest partition over the naive
  composition);
- the FA derivation `merge`s them (rung 1) — but a 3→1 merge is TWO `merge` calls
  (`merge(merge(P, a₁, a₂), a₁₂, a₃)`), each a composite transaction, because islands `merge`
  is binary (`islands-design.md §2`). The FA script (deliverable 3) records `fuse ×2`.
- the `islandFlow` edges are the **convexity witness** for the merge: `scores → softmax → P` is
  a linear chain, so the union `a₁ ∪ a₂ ∪ a₃` is convex (no external node lies between them —
  `islands-design.md §2` rule 1). The transaction reads `islandFlow` to check convexity WITHOUT
  re-deriving the dataflow graph.
- the `proposedInterior` for the merged region is the interior schedule the SUBSEQUENT moves
  (tile → stream → recolor → lemma → stream) build up — it starts as the naive composition's
  concatenated bodies and is transformed by the intra-schedule moves before realization.

### 1.6 Islands-side merge: BUILD engine-side vs DRIVE through re-record — the ruling

**The question (design §7 P2-readiness (a)):** does P2 build a `merge(P, a, b)` engine-side
mutator, or drive the merge through the existing re-record-with-different-fusion-decisions
channel?

**Ruling: DRIVE through re-record for the transaction's MEMBERSHIP effect; BUILD only the
thin composite-transaction driver (validation + provenance + rollback) engine-side. Do not
build a new standalone partition mutator.**

Argument:

1. **The membership change already has an owner and a channel.** The live fusion detector
   (`fusion-detect.ts`) already decides island membership every step and the executor
   re-records when fusion decisions change (design §7 P1 "the variant registry is re-expressed
   as `ScheduleTemplate`s"; the compiled-plan cutover re-records on fingerprint change). Islands
   `merge`/`split` are DEFINED (`islands-design.md §2`) as the mutation vocabulary over that
   existing partition — but the partition itself is produced by the detector/policy, not by a
   free-standing `merge` function today. Building a second, standalone engine mutator that
   ALSO writes membership would be a **second owner of membership** — exactly the sin S3 and
   §2.6's no-second-owner invariant forbid. The islands doc is explicit: "merge greyed out on
   V100 is offered on A100... express-to-measure" — membership legality is a POLICY over the
   detector's partition, not a new imperative mutator.

2. **What the transaction genuinely needs engine-side is NOT a mutator — it is a driver.** The
   composite transaction's four novel obligations are: (i) validate the proposed interior
   (schema-level, no engine state); (ii) CHECK the islands `merge` legality gate (a pure
   predicate over the partition + device class — `islands-design.md §2`'s enumerated rules, read
   from the island IR, NOT a mutation); (iii) mint region' + attach + assert no-second-owner
   (schema-level); (iv) provenance + atomic rollback (a ledger operation). NONE of these is a
   dispatch-path membership write. The actual membership change is realized by re-recording
   under the new fusion decision the driver requests — the driver expresses "these regions are
   now one island" as an input to the SAME channel the detector writes, and the executor
   re-records. This keeps ONE membership owner (the partition/detector), consulted by the
   islands `merge` legality predicate, driven by a re-record.

3. **The rollback is trivial under drive-through-re-record.** Because the membership change is a
   re-record under a fusion decision, rollback is discarding the requested decision and
   re-recording under the prior one — the split cut in the inverse payload IS the prior fusion
   decision. A standalone engine mutator would need its own undo of a live-mutated partition
   (harder, and a second owner of the undo semantics that `split` already provides). Drive-
   through-re-record gets rollback for free from the existing record path.

4. **The rule-of-three / admission-pressure discipline (CLAUDE.md complexity budget).** A
   standalone `merge(P,a,b)` engine mutator is net-new mechanism that proves itself in exactly
   one place (the FA derivation). The house rule is code enters `src/` only when it has earned
   generality. The composite-transaction DRIVER earns it (it is the general fuse-gesture the
   editor needs); a second membership mutator does not (the detector + re-record already own
   membership).

**Consequence for the split of parts:**

| Part | Owner | Build vs reuse |
|---|---|---|
| validate proposed interior (core legality) | **schedule-side** (`src/schedule/moves/`) | BUILD (thin — move-algebra laws over `SemanticSchedule`) |
| islands `merge` legality gate | **islands-side** | REUSE (`islands-design.md §2` predicate; ported into the schedule seam as a read-only check, NOT re-owned) |
| the membership change itself | **islands-side (detector/partition)** | DRIVE through re-record (no new mutator) |
| mint region' + attach + no-second-owner assert | **schedule-side** | BUILD (schema-level; §2.6 assertion) |
| ONE provenance record (both hashes) + atomic rollback | **schedule-side driver** | BUILD (the composite-transaction driver) |
| tape invalidate on fingerprint change | **tape-side** (capture/tape agent) | REUSE the existing invalidate path, keyed on `FuseProvenance` |

So P2 builds a **thin composite-transaction driver** (`fuseGesture`) plus a **read-only port of
the islands `merge` legality predicate** into the schedule seam. It builds NO standalone
membership mutator; the membership effect is driven through the existing re-record channel that
already owns the partition.

---

## 2. Deliverable 2 — the `stream` move + engine-side streamability predicate

### 2.1 The `stream` move's typed before/after schema (§3.1 pattern)

The move union already carries `{ move: "stream"; value: ValueUid; loop: LoopUid }`
(`types.ts §13`). This is its full before/after contract (design §3.1 verbatim, made precise):

```
stream(valueUid, loopUid) — turn a materialized intermediate into a value
                            produced/consumed inside loopUid (no global store).

before:  `value` is materialized to global between its producer and consumer
         (a StoreEdge writes it; a NamedValue.load reads it back).

after:   a NoMaterializationEdge on `value` across `loopUid`; the StoreEdge is
         DELETED, a streamed carried-value edge is added. `value.allocation`
         moves off "global" (the residency of the streamed value is a `requests`
         PREFERENCE, not part of this move — design §3.1).

invariant (THE REFUSAL-FIRST BOUNDARY, F17): `value` must have a declared
         head/body decomposition over `loopUid`'s axis. Streamability is
         MACHINE-CHECKED over typed head/body terms with a recomposition law
         (F5), NOT evidence-shaped strings. A value with NO decomposition is
         REFUSED — this is where the FA derivation's "drag streaming onto naive
         softmax" is correctly refused (ordinary softmax has no head/body
         decomposition; the online-softmax lemma is what GIVES it one).

inverse: unstream(valueUid, loopUid) — inverse data { the deleted StoreEdge }.
```

`stream` is a **partial function on `SemanticSchedule`**: it applies iff the streamability
predicate (§2.2) admits `value` over `loopUid`'s axis; otherwise it is **refused at the seam**
with the predicate's reason (the ncd-surface "jam" UX — never silently dropped).

### 2.2 The streamability predicate as an ENGINE object over the semantic body

**The napkin definition (design §3.1 / wave-3 §6(b)):** a value is streamable along an axis iff
it has a **head/body decomposition** over that axis — a way to split the axis into blocks,
compute a partial result per block into a **carried accumulator**, and **recompose** the blocks
into the full result by a law that holds regardless of block boundaries. Formally, for a value
`v` reducing/accumulating over axis `A`:

```
∃ (init, step, merge) such that for any partition A = B₁ ⊎ B₂ ⊎ … ⊎ Bₖ:
     v(A) = merge( step(B₁, init), step(B₂, init), …, step(Bₖ, init) )   [the RECOMPOSITION LAW]
```

- **`init`** is the head: the accumulator's initial state (e.g. `sum: 0`, `max: -∞`).
- **`step`** is the body: fold one block into the accumulator.
- **`merge`** is the recomposition: combine two partial accumulators (associative + commutative
  for the block order to be irrelevant — `sum` merges by `+`, `max` by `max`).

**Why this supersedes the NCD spike's client-side version.** The spike
(`examples/schedule-editor/src/lib/ncd/model.ts`) has `Streamability = { kind: "decomposed";
axes: [{ head: string; body: string; … }] } | { kind: "none"; reason }` — `head`/`body` are
**free-form strings** (`"initialize m=-∞, l=0, o=0"` / `"update m,l and rescale o…"`). F5's
ruling: streamability must be machine-checked over **typed** head/body terms with a
recomposition LAW, not evidence-shaped strings. The engine predicate reads the value's ACTUAL
semantic body (the `SemanticBodyNode` / reduce-op tree the family skeletons already produce) and
classifies its root accumulator form — it does not trust an authored string.

**The classification over the semantic body:**

| Body root form | Head/body decomposition | Streamable? | Why |
|---|---|---|---|
| `reduce_sum(f(x))` over axis A | `(init=0, step=+f(xᵢ), merge=+)` | **YES** | `+` is associative+commutative; blocks recompose by summation |
| `reduce_max(f(x))` over axis A | `(init=-∞, step=max f(xᵢ), merge=max)` | **YES** | `max` is associative+commutative |
| `reduce_mean(f(x))` over axis A | `(init=(0,0), step=(+f, +1), merge=(+,+))` then divide | **YES** | mean = sum/count; both accumulators recompose (Welford is the teaching lemma) |
| `div(exp(sub(x, max_A(x))), sum_A(exp(sub(x, max_A(x)))))` (softmax) | **NONE** | **NO — REFUSED** | the per-element output divides by the FULL-ROW denominator `sum_A(…)`, which is not known until all of A is seen; no block-local `step`/`merge` recomposes it without correcting earlier blocks (that correction IS the online-softmax lemma, an admitted-lemma rewrite, NOT a free decomposition) |
| softmax body AFTER `onlineSoftmaxLemma()` applied | `(init=(m=-∞,ℓ=0,o=0), step=update (m,ℓ,o) with exp(m_old−m_new) rescale, merge=rescale-and-add)` | **YES** | the lemma's carried state `(m,ℓ,o)` + correction `exp(m_old−m_new)` IS a recomposition law; the lemma is what supplies it |

**The engine object.** The predicate is a pure function over the semantic body + the axis:

```ts
streamability(schedule: SemanticSchedule, value: ValueUid, axis: AxisUid): StreamabilityVerdict

type StreamabilityVerdict =
  | { streamable: true; decomposition: HeadBodyDecomposition }
  | { streamable: false; refusal: StreamRefusal };

interface HeadBodyDecomposition {
  readonly axis: AxisUid;
  readonly init: AccumulatorForm;   // typed head (not a string)
  readonly step: AccumulatorForm;   // typed body
  readonly merge: RecompositionLaw; // the recomposition law that makes block order irrelevant
}

interface StreamRefusal {
  readonly reason: string;
  /** The proof-obligation ID whose discharge would ADMIT this stream (F28 seam).
   *  A softmax refusal names obl:online-softmax-normalizer-equals-batched-denominator
   *  — the obligation the online-softmax lemma discharges. NEVER refusal-text matching. */
  readonly dischargedBy: ObligationId | null;
}
```

### 2.3 Where refusals surface — the proof-obligation ID seam (F28)

A stream refusal on softmax **names the obligation the online-softmax lemma discharges** — it
does NOT emit a bare "cannot stream" string that the game later matches. This is the F28 binding
made engine-real (wave 3 landed `ONLINE_SOFTMAX_OBLIGATION =
obl:online-softmax-normalizer-equals-batched-denominator`):

- the softmax refusal's `dischargedBy` is `ONLINE_SOFTMAX_OBLIGATION`;
- the FA derivation reads that obligation ID and looks up the lemma whose `obligation` field
  matches (`onlineSoftmaxLemma().obligation`), applies it, and re-runs streamability — now
  ADMITTED (the post-lemma body has the `(m,ℓ,o)` recomposition);
- the jam→lemma binding is **by obligation ID**, never by parsing the refusal's human text
  (F28 — wording changes cannot alter legality). This is the "refusal-first" boundary made a
  typed seam: refuse → name the obligation → discharge via the matching lemma → re-admit.

This is exactly the FA F17 sequence `lemma → recolor → recolor → group → stream` read from the
other end: the `stream` on naive softmax refuses first, the refusal NAMES the obligation, the
lemma discharges it, and the post-lemma `stream` succeeds.

### 2.4 The prototype (`src/schedule/moves/streamability.ts`)

The prototype implements ONLY the predicate over the existing `SemanticSchedule` /
`SemanticBodyNode` types — pure schema-level, no dispatch, touching nothing live. It:

- classifies a value's body root as `reduce_sum`/`reduce_max`/`reduce_min`/`reduce_mean` →
  streamable with the typed `(init, step, merge)` decomposition;
- classifies a softmax-shaped body (`div(exp(sub(x, max)), sum(exp(sub(x, max))))`) → NOT
  streamable, naming `ONLINE_SOFTMAX_OBLIGATION` as `dischargedBy`;
- classifies the post-lemma online-softmax body (a body carrying the lemma's `(m,ℓ,o)` carried
  state) → streamable.

The move-algebra `stream` body that CONSUMES this predicate (deletes the StoreEdge, adds the
NoMaterializationEdge, refuses on a `false` verdict) is P2 IMPLEMENTATION, specified in §2.1
above but not built here (that touches `applySchedule` and the move executor — out of this
design-first scope).

---

## 3. Deliverable 3 — the P2 acceptance script

The FA derivation as a MOVE-SCRIPT (the wave-1 textual form; `canonical.ts MoveScript`) lives at
`tools/fa-derivation-script.ts`. It is the SPEC the P2 implementation must satisfy: base =
the naive three-region composition digest; ordered moves with each move's expected legality
outcome annotated (which succeed, which refuse-first, which discharge an obligation). It does NOT
run yet (the move bodies are P2 implementation). See that file for the verbatim script; the
companion report reproduces it.

---

## 4. Open questions for review

1. **3→1 merge as fuse ×2 vs a variadic fuse.** §1.5 records the 3-region naive merge as two
   binary `fuseGesture` calls (islands `merge` is binary). Is a variadic `fuseGesture(P, [a,b,c],
   interior)` worth admitting, or does the binary-composition rule (with an intermediate region
   UID minted and immediately re-merged) stay the fence? The binary form keeps islands `merge`
   the single owner; a variadic form is a convenience that mints a throwaway region.

2. **Interior-schedule identity across the two merges.** When 3→1 is two merges, the FIRST
   merge's `proposedInterior` covers only `a₁∪a₂`; the SECOND covers `(a₁∪a₂)∪a₃`. Does the
   first merge's interior get realized/measured, or is it a transient the second merge
   supersedes? (Argues for the transaction returning the interior UNrealized until the final
   merge, deferring realization — but that weakens the atomic-rollback guarantee per-merge.)

3. **Streamability of `mean` — Welford vs sum/count.** §2.2 lists `mean` as streamable via the
   `(sum, count)` two-accumulator form. The teaching lemma (NCD exercise 3) is Welford's
   `(count, mean, M2)`. Should the predicate admit `mean` DIRECTLY (sum/count recomposition, no
   lemma) or REFUSE it and require the Welford lemma? The former is simpler and correct for the
   mean itself; the latter is needed only for streaming VARIANCE. The prototype admits mean
   directly (sum/count) — confirm that is the intended boundary.

4. **The obligation ID for the D-precompute / attention-backward streams.** §2.3 wires the
   softmax refusal to `ONLINE_SOFTMAX_OBLIGATION`. Attention backward (P4 local self-hosting)
   needs the recomputation-identity + D-precompute lemmas. Do those get their own obligation IDs
   now (so the backward stream refusals name them), or is that deferred to P4? (This doc scopes
   to the forward FA derivation; backward obligations are named but not defined here.)

5. **Device-keyed merge legality in the transaction return.** §1.3 makes `merge` refusal
   device-keyed (A100-legal, V100-refused). Should `fuseGesture` take the device class as an
   explicit parameter (so the editor can ask "would this fuse on A100?"), or read it from the
   partition's island IR (which carries the device class per `islands-design.md §2`)? The latter
   keeps one owner of the device class; the former enables the "express-to-measure" preview
   across device classes. Prototype-agnostic; a P2 API decision.

## §5 — Rulings on the §4 open questions (orchestrator, 2026-07-12; Vin may veto)

1. **fuse ×2 binary, not variadic.** Composability + matches islands' binary merge contract;
   variadic is editor-side sugar over binary gestures if ever wanted.
2. **Defer realization to the final state of a chained transaction.** Interior states are
   VALIDATED per merge but realized once at commit — cheaper, and consistent with
   publish-after-validation (R17). Rollback still per the inverse chain.
3. **`mean` is directly streamable** as the (sum, count) monoid pair — no lemma needed
   (Welford is the numerical-stability lemma for VARIANCE, i.e. layernorm's inv-std, not
   for the mean itself). The prototype's direct admit stands.
4. **D-precompute / attention-backward obligations get IDs at P4**, not now — they are
   authored-set members; P2's scope is the forward derivation.
5. **Device-keyed merge legality reads from the island IR** (one device-class owner). The
   workbench's cross-device "express-to-measure" preview is achieved by constructing a
   hypothetical island IR client-side, not by parameterizing the engine gesture.

Follow-up adopted: the streamability predicate gains an explicit AxisUid argument when
loops carry the reduce-axis binding (one-line, at P2 move-algebra landing).
