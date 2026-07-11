# Schedule State: the intra-kernel stratum as data

**Status:** v2 (amendment round 1) · 2026-07-11 — rewritten to the rulings in
`docs/design-amendment-round-1.md` (ratified by Vin; disposes red-team R1–R28 +
Appendix A + findings F1–F34). The v1 draft (2026-07-11) is superseded; every §
below carries the ruling it applies.
**Depends on:** docs/islands-design.md (I0–I2 landed; S3 composes with its merge/split
contract), docs/tile-ir-design.md, docs/ownership-derivation-design.md (the leveling /
derived-fact precedent this campaign follows), the checked-in design corpus
(`docs/design-corpus/` — containment-and-ladder analysis, validity/completeness,
napkin-paper analysis, Appendix A Triton profile; R28).
**Sibling charter:** docs/model-editor-charter.md — shared spine, see §9.
**Surface:** docs/ncd-surface-spec.md — the authorable notation this object underlies.

## 0. Declaration (one sentence)

A kernel's schedule is a first-class **three-tier data object** — a semantic schedule
that hashes into semantic identity, backend requests that hash into compilation identity,
and realization receipts that never hash into either — from which a loop-nest VIEW is
derived by a canonical ordering rule; the fusion detector, the editor, and the autotuner
are three writers of that one object.

## 1. Lineage and why

This is the sixth application of the house move "the latent decision becomes an object":
scalars→tensors-as-data, buffer assignment→planner-derived, compiled replay→built-from-IR,
variant selection→registry data (#61), dispatch partition→islands (I0–I2), and — its
nearest kin — liveness classification→derived-at-inquiry (ownership-derivation-design.md,
#70). That last campaign is this one's **leveling precedent**: it took a fact that lived
as mutable distributed state, made it a single-source derivation, and named its deletions;
its discipline (a shadow differential, an assert-agreement gate, an executable
no-second-owner claim) is the template §7's P0 follows. Macro-structure is the **last big
latent decision**: today the loop nest, shared-memory staging, and thread-role assignment
live in the control flow of the TypeScript generator functions.

Consequences of latency (the disease this cures):

- Legality is scattered heuristics (claim cascade, epilogue chains). With schedule state,
  I3's "claim casts into the matmul dispatch" is a *move* checked against an object,
  not a new bespoke pass.
- The autotuner can only sweep a hand-enumerated variant list; schedule state gives it a
  real search space.
- The editor cannot zoom below an island; schedule state is the intra-island view.
- N hand-rolled structural variants (matmul K-split/chunk shapes, reduction skeletons)
  each re-encode structure that one semantic-region + schedule factoring would carry once.

**The condition that makes this better rather than worse:** it must EARN DELETIONS, and
"byte-identical regeneration" alone does not prove it did (R22 — a lossless AST dump or an
`opaqueGeneratorId` replaying the old generator passes every byte test while owning
nothing). Acceptance is §7's null differential PLUS the S2 structural gates (schema-only
serialization, the executable no-second-owner assertion) PLUS §8's deletion list — not
"the editor works."

## 2. The object (S1 — the three-tier split)

**Ruling S1 (R1 + R2 + F8, confirmed by Appendix A):** the v1 single flat
`DecorationVector` is DEAD. It fused three things that hash into three different
identities and that different realizers own at three different authority horizons. The
object splits in three tiers, each with a distinct identity role (§5):

```
ScheduleState = {
  semantic:  SemanticSchedule    // hashes into SEMANTIC identity
  requests:  BackendRequests     // hashes into COMPILATION identity, NOT semantic
  receipts:  RealizationReceipts  // hashes into NEITHER — filled by the realizer
  region:    SemanticRegionUid    // FOREIGN KEY into the semantic graph (R8), not payload
}
```

### 2.1 `SemanticSchedule` (semantic identity)

The facts that are the same computation-shape regardless of who realizes them. Per S1 and
Appendix A's "determination" column:

- **logical block shapes** — logical tensor extents (`BLOCK_*`-class), NOT per-thread
  tiles, NOT layouts, NOT instruction shapes. Appendix A A-R1: Triton `determination`.
- **loop / dispatch structure** — the loop nest as typed identity (F4): each loop carries
  a `LoopUid`, a bound expression in the typed predicate AST (§2.5, R12/F4), and a
  `parallel | sequential` kind; grid-axis nesting is explicit. Human-readable
  `ceil(M/tile.m)` is a display projection of the AST, never the hashed form.
- **no-materialization (fusion) edges** — the semantic no-store contract (Appendix A
  A-R2: `stream`'s `determination` half), separate from any residency request.
- **`ProgramGridMap`** — the program-id → work bijection (R4/§2.4), a semantic fact
  (it changes neither arithmetic nor tile size, only traversal order).
- **named value lifetimes** (F6) — each value has a `ValueUid`; staging edges name
  allocation identity plus lifetime/reuse so the cost model can see that attention's K
  and V share one shared-memory allocation (F6 — summing edges would double-count). Both
  input staging (`global→shared→register`) and output flow (`register→global`) are edges;
  stores are edges, not implicit (F6).
- **staging intent** — WHICH value stages through WHICH tier-KIND, as a semantic
  no-materialization/residency-intent fact (the residency REQUEST that a realizer may or
  may not honor lives in `requests`; F1's register residency is expressed here as intent
  and in `requests` as a preference, never as a WGSL determination).
- **semantic atoms** (§3.3) placed as skeleton nodes with `LoopUid`/role placement (F10),
  referencing a single-sourced catalog entry.
- **admitted-lemma applications** (§3.4) — each carries a `LemmaUid` + the proof-obligation
  ID it discharges (F11/F28) and its first-class carried state (F27).

### 2.2 `BackendRequests` (compilation identity, NOT semantic)

**Ruling R2 + A-R14:** the requests a realizer receives but does not have to honor
physically. These change the compiled artifact but not the computation's semantic
identity (a kernel with `num_warps=4` and one with `num_warps=8` are the same semantic
schedule). Value domains are explicit sets, never booleans/absences (R3 spirit):

- **warp / thread budget** — `num_warps`-class CTA budget (Appendix A A-R7: Triton's
  nearest map to WGSL workgroup dims is a `request`, not an x/y/z geometry). A budget,
  not a geometry.
- **pipeline requests** — `none | [{ loopUid, loadGroupUids, requestedStages }]`
  (R2, verbatim). `none` is LEGAL and carries no fake degenerate value; there is no
  "pipeline depth pinned 1" fact. Multiple loops may request different staging (Appendix A
  A-R9). The realized stage count is a receipt (§2.3), never this.
- **placement preferences** — operand-residency PREFERENCES ("prefer B in shared"),
  scoped to a value-use interval. Appendix A A-R8: `refused` as a WGSL determination, so
  it can only ever be a request; F1's operand-residency axis lives here as a preference
  and in `semantic` as intent.
- **cache policy** — load/eviction/volatile hints (Appendix A: Triton `tl.load` cache
  modifiers; A-R14 named this a missing coordinate).

### 2.3 `RealizationReceipts` (hashed into NEITHER identity)

**Ruling S1 + R1 + R5:** the physical facts a realizer CHOOSES and REPORTS. They are never
part of any schedule identity — they are the realizer's answer, keyed by measurement
identity (§5, R20) when measured:

- **WGSL workgroup x/y/z** — the actual dispatch geometry (R1: WGSL-only; Appendix A A-R7).
- **exact vec-load forms** — `array<vec4<T>>` reinterpretation, `ctx.loadVec4` operand
  loads, shared-vec4 arrays (R1 — these are three different lowering choices the compiler
  picks and rejects-combinations-of; Appendix A A-R6: vec width is `refused` as a portable
  fact). The v1 "vec width is one decoration" claim is retracted here.
- **realized pipeline stages** — the `realizedStages?` the realizer actually emitted (R2).
- **atom realization** — `CASLoop` vs `NativeAtomic` for `atomicAdd<f32>` (R5/R25/A-R12:
  the semantic atom is portable; whether it lowers to a CAS retry loop or a native float
  atomic is a receipt, not part of the atom's identity).
- **measured results** — timings/occupancy proxies, keyed by the §5 measurement identity
  (R20 keys). These NEVER back-propagate into semantic or compilation identity.

### 2.4 `ProgramGridMap` — the program-map move's payload (R4, S2 deliverable)

**Ruling R4 (accepted as proposed; complete the partial amendment):** a canonical
bijection-valued map from a linear program id to a work assignment, a first-class member of
`SemanticSchedule`. Enumerated forms:

```
ProgramGridMap =
  | { kind: "identity" }
  | { kind: "swap",    axes: [AxisUid, AxisUid] }        // swapGrid
  | { kind: "grouped", groupAxis: AxisUid, groupSize: N } // Triton grouped-matmul L2 reuse
  | { kind: "checkedAffine", expr: PredicateAstNode }     // a checked affine/index expression
```

- **Legality = one-to-one in-bounds coverage** (R4): the map must prove it is a bijection
  over the launch domain (every program id maps to exactly one in-bounds work unit and
  vice versa). Checked at the move seam.
- **Mandatory reification tests** (R4): the repo's existing `CodegenOptions.swapGrid`
  ("improves L2 cache reuse for wide shapes") and Triton's published grouped-matmul
  (`tl.swizzle2d`, >10% on A100) must both round-trip through this field. These are P0
  corpus entries; the grouped-matmul one is the first entry of the external conformance
  corpus (§7 P4, R4/R25).
- This is the 8th move (§3, `program-map`). It is NOT `recolor` (R4: stretching recolor to
  "change any mapping" makes it an untyped escape hatch; A-R15).

### 2.5 The typed predicate AST (R12 + F5) and the typed sync relations (R3)

**Ruling R12 + F5 (accepted as proposed):** ONE typed predicate AST, shared verbatim with
the model editor's override selection and optimizer param-groups (charter §2). It is a
typed tree with explicit leaf domains — not an untyped string grammar. Leaf domains valid
in the schedule context: **schedule entities** (loops by `LoopUid`, values by `ValueUid`,
roles/role participants, staging edges, origin sets) — plus the shared model/param/
semantic-op/island leaves the sibling doc enumerates. Shared combinators (and/or/not,
range, set-membership). Loop bound expressions (F4), role participant sets (F5), and
`checkedAffine` grid maps (§2.4) are all nodes of this one AST. Serialization + type
errors are gated in P0 (R12).

**Ruling R3 (accepted as SCHEMA):** synchronization is not one level chain, one barrier,
or a set of simultaneously-selected labels. The v1 "enumerated axes with degenerate
values" table is replaced by typed relations whose value domains are explicit sets (the
"enumerated axes" spirit survives inside the typed forms — never booleans/absences):

```
MemoryEffect(space: AddressSpace, value: ValueUid, interval: UseInterval)
Barrier(participants: ParticipantSet, spaces: AddressSpace[], convergence: ConvergenceFact)
Atomic(order: MemoryOrder, visibility: VisibilityScope)   // order/scope, NOT "workgroup"
```

- `AddressSpace`, `MemoryOrder`, `VisibilityScope` are explicit enumerated sets (Appendix A
  A-R10: Triton atomics expose `cta`/`gpu`/`sys` visibility + acquire/release separately;
  a single `{workgroup}` cannot distinguish them).
- Thread hierarchy is modeled as a **backend capability GRAPH** (R3), NOT as values stored
  in every schedule. "Grid barrier" is not a stored axis — it is unavailable inside an
  ordinary WGSL or Triton kernel (R3/A-R10), so it is a capability-graph absence, not a
  reserved degenerate value.

### 2.6 Leveling rule (the no-second-owner invariant, S2)

Every fact at one level. `semantic` owns computation-shape; `requests` owns realizer asks;
`receipts` owns physical choices; the semantic graph owns the region behind
`SemanticRegionUid` (R8 — it is a foreign key, NOT copied contents; the object stores a key
and declines to identify it into semantic identity, resolving R8's "stores-and-declines"
contradiction); the island owns membership (S3). A fact appearing twice is a review-blocking
defect, made **unconstructible** by P0's executable no-second-owner assertion (S2 (d)),
following the ownership-derivation precedent. There is no `AlgorithmRef` (R8): the v1 field
that both stored "island contents / tile-IR DAG" and was omitted from the hash is deleted;
`region: SemanticRegionUid` replaces it as a normalized foreign key.

## 3. The move grammar (§3 gains program-map — EIGHT moves; R4, R28)

**Ruling (Rejected/narrowed):** the seven-move fence is amended to EIGHT — `program-map`
joins (R4) — rather than withdrawing the completeness claim; completeness itself is
re-scoped per R25 (§7 P4). Mutators on `SemanticSchedule`:

```
{ tile, stream, recolor, pack, role-partition, pipeline, program-map }   // 7 intra-schedule
```

`fuse` is NO LONGER a ScheduleState move (S3 — §3.5). The move set is FENCED: new move
kinds require a design amendment (§10 risk b). Each move below has a typed before/after
schema (R28 — invariants, inverse data), drawn from the corpus's move definitions
(`design-corpus/kernel-editor-containment-and-ladder.md` §1.3, the CUDA-lever taxonomy) and
the napkin analysis. The schemas are the normative content R28 demanded be checked in.

### 3.1 Typed move schemas (R28 — before/after, invariants, inverse data)

Each move is a partial function on `SemanticSchedule`; a move that violates any invariant
is **refused at the seam** with a reason (never silently dropped — the ncd-surface "jam"
UX). "Inverse data" is what the provenance entry must carry to invert the move (S3 /
ownership-derivation's inverse-payload discipline).

**`tile(loopUid, axisUid, factor)`** — split one loop's iteration axis into an outer×inner
nest.
- *before:* a loop `L` over `axisUid` with extent `N`.
- *after:* outer loop `L_o` (extent `ceil(N/factor)`) enclosing inner `L_i` (extent
  `factor`), same body; block shape gains the tiled sub-extent.
- *invariants:* `factor` divides or the tail is masked (a `checkedAffine` bound); block
  shape stays within the tier capacity named in `requests`/receipts; divisibility
  superscripts (F11) compose by LCM — the move carries the contributing constraint reasons,
  not just the LCM.
- *inverse:* `untile(L_o, L_i)` — inverse data is `{axisUid, factor}`. Structurally
  invertible (records nothing lossy).

**`stream(valueUid, loopUid)`** — turn a materialized intermediate into a value produced/
consumed inside a source loop (no global store).
- *before:* `value` materialized to global between producer and consumer.
- *after:* a no-materialization edge on `value` across `loopUid`; the store edge is deleted,
  a streamed carried-value edge added.
- *invariants:* the value must have a declared head/body decomposition over `loopUid`'s
  axis (F5 — streamability is machine-checked over typed head/body terms with a recomposition
  law, not evidence-shaped strings; a value with no decomposition is REFUSED — this is the
  FA "refusal first" boundary, F17). Semantic no-store contract (Appendix A A-R2
  determination); the residency of the streamed value is a `requests` preference, not part
  of this move.
- *inverse:* `unstream(valueUid, loopUid)` — inverse data `{the deleted store edge}`.

**`recolor(valueUid, column, tierKind)`** — change a value's residency-intent at one point
in its sampled level-path.
- *before:* `value`'s residency sample at `column` is `tier₀`.
- *after:* the sample is `tierKind`; interior samples changing an `ℓ1→ℓ0→ℓ1` to `ℓ1→ℓ1→ℓ1`
  remove a materialization boundary (fusion); endpoint changes are external load/save (F9/
  F24 — the move carries the transition ROLE: `materialization-boundary` vs
  `external-transfer`).
- *invariants:* a residency sample per column, not one `wire.level` (NCD F1 — a value can be
  `ℓ1→ℓ0→ℓ1`); the tier must exist in the device/realizer level graph (NCD F10 — never imply
  authority WGSL lacks); for a boundary-removal, producer and consumer must share the
  decomposed axis (NCD F8 — single-producer/single-consumer; multi-consumer is a region
  convexity question handled by S3's island altitude, not here). `recolor` does NOT remap
  program ids (R4 — that is `program-map`).
- *inverse:* `recolor(valueUid, column, tier₀)` — self-inverse; inverse data `{column,
  tier₀}`.

**`pack(loopUids[], kind)`** — batch independent same-shape work horizontally into one
emitted program.
- *before:* N independent same-shape schedule regions (e.g. per-tensor Adam updates).
- *after:* one region iterating a pack axis; per-item pointer/index code (Appendix A A-R4:
  `determination` for horizontal multi-tensor work — NOT for SIMD/vec4 packing, which is a
  receipt).
- *invariants:* items are shape-compatible; `kind` distinguishes `map/concatenate` from a
  chunked-binding pack (F6/NCD F6 discriminated decomposition family). Physical vector width
  is a receipt, never `pack`.
- *inverse:* `unpack(packAxis)` — inverse data `{the N region identities}`.

**`role-partition(loopUid, roles[])`** — partition the EXECUTOR (not a data axis) into named
producer/consumer roles.
- *before:* a homogeneous invocation set runs the whole body.
- *after:* named role groups (producer / consumer(s)) with typed participant sets in the
  predicate AST (F5 — invocation sets, subgroup gates, cooperative striding, thread-tile
  ownership — a typed participant/predicate grammar, not descriptive strings); handoffs are
  derived from dataflow (the mbarrier/baton edges).
- *invariants:* participant sets partition the invocation domain (disjoint, covering); the
  backend capability graph must admit role partitioning (Appendix A A-R5: WGSL/ordinary
  Triton `refused`/narrowly-`request` — on those backends the move is capability-absent, a
  locked palette entry, not a degenerate value). This is the CUDA-graduation move kind
  (corpus §1.3 "genuinely new move kind"); reserved-but-empty on WGSL (corpus rec #4).
- *inverse:* `unpartition(loopUid)` — inverse data `{role assignment}`.

**`pipeline(loopUid, loadGroupUids, requestedStages)`** — request async multi-stage
overlap of loads feeding compute in a loop.
- *before:* a loop with barriered load/compute.
- *after:* a `pipeline` REQUEST entry in `requests` (§2.2) — NOT a semantic change and NOT a
  "depth 1" fact when absent (`none`). The realizer reports `realizedStages` in receipts.
- *invariants:* `requestedStages ≥ 2` for an entry to exist (else it is `none`); the loop's
  loads must form the named `loadGroupUids`. Appendix A A-R9: Triton `request`; the realizer
  may emit a different instruction schedule.
- *inverse:* remove the request entry — inverse data `{the entry}`. (Because pipeline is a
  request, its "before/after" is on `requests`, not `semantic` — it does not change semantic
  identity.)

**`program-map(map: ProgramGridMap)`** — replace the program-id → work bijection (§2.4, R4).
- *before:* current `ProgramGridMap` (default `identity`).
- *after:* the new map.
- *invariants:* one-to-one in-bounds coverage (§2.4). Changes traversal only — not
  arithmetic, tile, stream, fusion, pack, roles, or pipeline (R4/A-R15).
- *inverse:* `program-map(prevMap)` — inverse data `{prevMap}`.

### 3.2 Legality core vs backend legality

The core (move algebra laws, convexity of the affected region, staging well-formedness) is
backend-neutral and lives with the object. Whether a *realizer* can honor the resulting
state is a capability-profile question (§4) — expressed against the backend capability graph
(R3). v1 code must never check WGSL facts inside core legality (the R1/R6 WGSL-ism trap).
Per-move numerical legality is NOT established by the differential (R23 — §7 P2/§8); the
differential establishes correctness, the invariants establish legality.

### 3.3 Atoms are not moves (R5 + R25 — mechanical admissibility)

Some primitives are COMPOSED AROUND, never derived: data-dependent constructs and hardware
intrinsics. An **atom** is a wrapped primitive with a semantic name and declared contract.

**Ruling R5/R25 (accepted as proposed):**
- **Semantic naming, realization in receipts.** The atom is `atomicAdd<f32>` with an
  explicit `{ order: MemoryOrder, scope: VisibilityScope, nanContract }` — NOT
  `atomicAddF32-CAS`. Whether it realizes as a `CASLoop` or a `NativeAtomic` is a receipt
  (§2.3, Appendix A A-R12) — requiring CAS would force an inferior backend algorithm and
  allowing native-add-under-a-CAS-name would make the atom's identity lie.
- **MECHANICAL admissibility** (R25): an atom is admissible iff it is a **single primitive
  effect or a hardware intrinsic** — NO composite loop nests, NO whole algorithms, NO whole
  authored kernels. This closes R25's relabel-a-kernel-as-an-atom hole; additions require
  reviewer approval.
- **Subgroup primitives are enumerated INDIVIDUALLY** (R5/A-R13): each subgroup op names its
  operation, width contract, convergence, fallback, and feature requirement. "Subgroup ops"
  is not one atom — it is an open family, and a family name without signatures cannot be
  profiled or admitted (F9 — subgroup appears as a hierarchy level, an atom family, AND the
  matmul `useSubgroups` decoration; the schema keeps these three distinct).
- **Placement, not just membership** (F10): an atom is a skeleton node in `semantic` with a
  `LoopUid`/role placement, operands, and multiplicity — referencing a single-sourced catalog
  entry. A root `atoms[]` bag cannot support cost or legality.

v1's atom set: `{ atomicAdd<f32>, <each enumerated subgroup primitive, feature-gated> }`.
scatter-add is NOT an atom — it is an elementwise schedule composed AROUND the `atomicAdd<f32>`
atom.

### 3.4 Lemmas are not moves (F27 + F28 — carried state, proof obligations)

Moves rearrange WHEN/WHERE the same arithmetic happens; their legality is structural. Some
targets are unreachable by rearrangement: flashattention requires the online-softmax identity
(accumulate softmax·V block-by-block, RESCALING the partial output by `exp(m_old − m_new)`
whenever the running max rises) — an algebraic fact about `exp` that computes different
intermediates for the same function. Such rewrites live in a separate **admitted-lemma
library**: hand-proven entries with their own differential gates, applied where their pattern
matches.

**Ruling F27 + F28 (accepted as proposed) — the lemma schema:**

```
Lemma = {
  uid:            LemmaUid
  proofObligation: ObligationId    // the wall it discharges (F28) — never refusal-text matching
  carriedState:   StateSchema      // FIRST-CLASS accumulator / state-machine (F27)
  rewrite:        BoxRewrite        // the before/after on the affected box (NCD F7)
  differential:   DifferentialRef   // its own gate
  witness?:       ExecutableWitness // small executable counterexample (F33) — optional in v1
}
```

- **Carried state is first-class** (F27): `carriedState` is the accumulator/state-machine
  representation (online softmax's `(m, ℓ, o)` + correction factor; Welford's `(count, mean,
  M2)`) from which inspection views DERIVE — not free-form `head`/`body` strings.
- **Jam→lemma binding is by proof-obligation ID** (F28): a challenge/wall binds an
  `ObligationId` to a `LemmaUid`; the game never parses or compares human refusal text (so
  wording changes cannot alter legality).
- A lemma application changes the box's intermediate state and body (NCD F7), so it is
  recorded as an admitted-lemma rewrite with proof history — it is an algorithm-term
  equivalence, not a schedule decoration and not "the same graph."
- v1 ADMITS lemmas; it does not derive them (deriving = a theorem prover, out of scope). The
  library starts with the one entry the P2 acceptance needs (online softmax) plus the small
  teaching lemma (Welford, NCD F27/exercise 3).

Applied to `SemanticSchedule` as `admitted-lemma applications` (§2.1), each carrying its
`LemmaUid` + `ObligationId` (F11 — a lemma reference in the state, so two derivations with
identical structural decorations but different proof obligations are distinguishable).

### 3.5 `fuse` leaves the move set — the composite transaction (S3)

**Ruling S3 (R7 + R8 + partition-as-coloring):** `fuse` is NOT a ScheduleState move.
Membership is owned by `Partition` ALONE (islands-design.md §2 — `merge`/`split` are the ONLY
partition mutators). The editor's "fuse" gesture is a **COMPOSITE TRANSACTION at the next
altitude**, composing with the islands merge contract:

```
fuseGesture(P, a, b, proposedInteriorSchedule):
  1. validate  the proposed interior SemanticSchedule (core legality, §3.2)
  2. merge     P' = merge(P, a, b)            // islands-design §2 — the ONLY membership owner
  3. mint      region' = newSemanticRegionUid // partition transaction mints the region UID (R8)
  4. attach    the interior schedule at region'
  5. record    ONE provenance entry carrying BOTH hashes (boundaryHash + the two identities §5)
  6. on realization failure: ROLL BACK all of 2–5 atomically
```

- The `merge` legality (dataflow convexity, shape/binding-count/barrier/chunking — islands
  §2) is the T0 gate for step 2; it is device-keyed (a merge legal on A100 may be refused on
  V100). The composite transaction does not re-own any of that.
- The ℓ0-coloring (the napkin's level-boundary paint) remains the editor's VIEW of the
  boundary, NOT a second owner (S3). Recoloring an interior sample to remove a boundary is a
  `recolor` move on the interior schedule; changing which islands exist is a `merge`. The two
  are different altitudes and the composite transaction is the only place they meet.
- This resolves R7 (double ownership of `fuse` between ScheduleState and Partition — the
  editor contract already said fuse-N→1 IS `merge`) and R8 (region UID minting on membership
  change) together.

## 4. Realizer registry (R13 — separate concrete things, no premature protocol)

A realizer = how a `ScheduleState` becomes an executable kernel.

**Ruling R13 (modified acceptance):** the "same registry shape" claim DIES. There is no
generic `Realizer<...>` protocol in v1. The WGSL kernel realizer and the model editor's
PyTorch-emit are **separate concrete things** — they consume different inputs (a
ScheduleState vs model strata 1–2), realize different artifacts, and the model emitter
explicitly cannot represent schedules (charter §2). A generic protocol is DEFERRED until a
third genuine instance exists (rule of three). Until then each realizer is its own concrete
type with its own versioned capability profile and typed refusal — no false shared
abstraction (R13: reusing a four-field record name does not make the zoom regimes
compositional).

The WGSL realizer entry:

```
WgslRealizer = { capabilityProfile, emit, costModel, verificationHarness }
```

- **v1 ships exactly one realizer:** tile-IR→WGSL (transparent to the bottom;
  what-you-sculpt-is-what-runs, byte-differential-gated).
- **Capability profile is machine-readable** (F8): it declares, per axis, allowed/refused
  values and REASONS ("WGSL realizer has no `role-partition` capability"; "pipeline `none`
  only"). Selected values live in `ScheduleState`; allowed/refused values + reasons live in
  the profile (F8 — the object alone cannot explain whether stages=2 is core-illegal,
  realizer-unsupported, or device-absent). The profile is versioned (`capabilityProfileVersion`,
  §5, R27).
- **costModel is single-sourced** (§7 P3): one artifact, two consumers (autotuner + workbench
  static tier). Building a second estimator is a defect. Its inputs are LABELED as measured
  adapter facts vs device-database facts vs heuristics (F13 — WebGPU exposes per-workgroup
  invocations/storage, NOT registers/thread or resident-workgroup slots, so occupancy is a
  labeled proxy).
- **v2 = Triton** (separate campaign, gated on §7 P0–P2 evidence): the cross-backend
  differential — the same ScheduleState through two disjoint compiler stacks on one A100,
  numerically diffed. Appendix A (checked in, R28) is the paper capability profile proving
  the S1 object expresses Triton's surface without WGSL-isms; any inexpressible entry is a
  representation bug (all found ones are already folded into S1 above). Triton is where a
  second concrete realizer earns the rule-of-three protocol conversation — not before.
- **v3 = CuTe** (gated on product need at CUDA altitude): its layout algebra ≈ the move
  grammar; the corpus containment analysis (Part 3) argues 𝔽₂ linear layouts as the layout
  decoration formalism. Paper only in v1.
- **Never:** tile-IR growing its own CUDA emit (the rebuild-Triton trap, declined).

## 5. Identity and caching (R27 — canonical serialization, three separated identities)

**Ruling R27 (accepted as proposed):** an FNV coordinate + sampled compare is NOT a
correctness guard (a collision on an unsampled production hit executes the wrong kernel; the
live plan fingerprint already uses two independent hashes and validates the secondary —
stronger than the v1 proposal). Replaced by:

- **Canonical serialization + strong digest.** Each tier serializes via a versioned canonical
  encoding (sorted keys, normalized symbolic expressions via the predicate AST, explicit
  numeric encoding, canonical atom/lemma/region ordering — F12 names every under-specified
  field; test vectors required, as the islands boundary hash has). The digest is strong (not
  a 32-bit FNV).
- **FULL canonical equality on every cache hit** (R27): sampling audits REGENERATION only,
  never content identity — a hit compares the full canonical serialization.
- **Three SEPARATED identities** (S1 + R27), never one field:
  - **semantic identity** = digest(`SemanticSchedule` + `region`) — what "same computation-
    shape" means; drives the semantic cache and the editor's diff.
  - **compilation identity** = semantic identity + digest(`BackendRequests`) +
    `{schemaVersion, capabilityProfileVersion, emitterVersion, compilerVersion, targetArch,
    featureSet}` (R27 — each can change realization/cached binaries while the ScheduleState
    stays semantically equal).
  - **artifact-cache identity** = compilation identity + the realization-receipt coordinate
    keyed to the produced binary.
- **Null-stability requirement (load-bearing):** the `SemanticSchedule` DERIVED from today's
  codegen must produce a stable semantic identity every step and regenerate today's WGSL
  byte-identically, or every static graph re-lowers and caches churn. This is the same
  null-case bar I1 met — but the byte-differential is NOT the ownership proof (R22; §7 P0).
- Cache-class discipline: each identity is a key≠content cache key by construction — it gets
  the #92 seam-guard (regenerate-and-compare on sampled hits under STRICT), on top of the
  full-equality-on-hit rule above.

## 6. The compression direction (S2 — the loop-nest VIEW derives; roles do NOT)

**Ruling S2 (R6 + R22 + F4 + the NCD projection result):** the semantic-IR boundary must
exist BEFORE phases (R6 — the live tile-IR is an imperative scheduling language; observing
its structure and replaying it through the same generator passes the byte differential while
leaving all ownership intact). Decision: **the loop-nest VIEW is DERIVED** from
`(semantic region × schedule state)` by a canonical ordering rule — but NOT everything
derives.

- **What derives** (the NCD projection result, qualified-valid): the loop nest as a VIEW.
  The projection needs a **canonical ordering rule** (NCD F12 — matmul's m→n vs n→m nesting
  realizes the same decorated term with different locality; partition labels alone do not
  derive a unique nest, so the ordering rule is an explicit part of `SemanticSchedule`, not
  left to decoration-array order).
- **What does NOT derive** (the spike proved it): thread roles, barriers, and lanes remain
  **schedule facts, NOT projections** (S2; NCD F13 — the projection derives group/stream
  loops and body calls but not which invocation loads which element, barriers, or lanes).
  These live in `SemanticSchedule` (roles via `role-partition`; barriers via the R3
  `Barrier` relation), not derived from region × decorations.
- Existing structural generators die as their idioms are absorbed into `applySchedule`
  (S2 (c)). Deletion targets (named per house policy, §8): the matmul structural-variant axis
  (K-split/chunk shape enumeration — selection stays, structure derives), reduction/row-program
  skeleton construction, elementwise loop scaffolding.

**Authored = not-yet-re-derived (migration staging, NOT an expressivity ceiling).** A kernel
marked **authored** is an opaque `ScheduleState` whose declared parameters are tunable but
whose internals take no macro moves — because it hasn't been re-derived yet.

**Ruling R10 + F3 + F7 (authored kernels — typed parameter schema, not generic decorations):**
generic decorations on an opaque skeleton are REFUSED. Every authored atom/kernel publishes:
- a **discriminated opaque skeleton** (F3): `skeleton.visibility: "derived" | "opaque"`;
  opaque REQUIRES a kernel reference + a refusal reason and FORBIDS loop/staging/role data
  (a boolean alone leaves consumers guessing which fields are absent).
- a **typed parameter schema** (R10 + F7): dependent constraints (vec4 needs alignment +
  multiple-of-four binding length; matmul tile sizes must divide thread tiles and fit shared
  memory; GEMV epilogues forbidden on split-K), derived geometry, a capability predicate, and
  a canonical cache-key encoder. Only DECLARED parameters are editable; a generic
  DecorationVector cannot validate an opaque skeleton because the validation facts are exactly
  what is opaque (R10). Decoration keys are enumerated per skeleton family (F7 — `m/n/k`,
  `qRows/kvRows/headDimension`, thread tiles; numeric domains; cross-field divisibility), not
  a typo-prone positive-integer bag.
- edits **commit only after a checked realization receipt** (R10 + F14): a decoration edit
  can create non-integral tilings / oversized workgroups / storage overflow continuously
  while editing; a bench/realization request returns a distinct server-authoritative
  legality result — a refusal with a stable code+reason, so not every state hash is
  benchmarkable, and the ledger commits the edit only on a checked receipt.

Three rules give the hatch its teeth:
1. The authored set SHRINKS monotonically: each member is either re-derived (moves + lemmas +
   atoms) or decomposed into a schedule composed around atoms.
2. Anything claimed PERMANENTLY underivable is tracked as a named grammar failure (a defect in
   §3, not a shrug) — the corpus claim is that the closure of the grammar contains the state
   of the art, falsified by permanent members. (This is the R25-scoped completeness claim, not
   the P4 self-hosting exit — §7.)
3. Expressivity fed back into §2: re-deriving attention backward at perf parity requires
   operand-RESIDENCY intent (which operand lives in registers vs shared — the register×shared
   vec4 dot). The object carries that axis from P0 (F1 — register residency is admitted in the
   `semantic` staging intent + `requests` preference now, not reserved for later; both honest
   instances, matmul accumulators and attention state, need it).

v1's expected authored members and their exits: attention backward (re-derive at the local
self-hosting milestone — needs the recomputation-identity and D-precompute lemmas), fused Adam
(re-derive — needs a horizontal-`pack` move at multi-tensor altitude), scatter-add (NOT
authored: composed around the `atomicAdd<f32>` atom from P0).

## 7. Phases (each independently shippable, stage-4 style)

### P0 entry criteria (the CURRENT task-board truth — R21 modified ruling)

**Ruling R21 (modified acceptance):** the recorded-build (#43) deletion is **NOT a P0
predecessor**. The v1 claim "#43 recorded-path deletions land FIRST, one codegen path" was
wrong — the final #43 map keeps recording load-bearing for uncovered plans (strided views,
chunked buffers, typed-buffer ops, copy-on-write, contiguous-copy prologues) until 4.4
coverage closes. So:

- **#43 is NON-BLOCKING.** P0's null differential runs at **KERNEL-codegen altitude** and
  executes on BOTH surviving plan paths (lowered + generated) wherever a kernel is reachable
  from both (R21). It does not require one codegen path and does not claim #43 deletion.
- **#85 characterized** (uncharacterized noise poisons differential gates — understand, not
  necessarily fix).
- **strict-lifetime IS the default now** (#73 landed — CLAUDE.md; the `[lifetime]` guard
  throws by default, the opt-out is in its soak window). Live kernel swapping already has the
  loud FP-free guards it needs. This is no longer a runway item.
- **IR completeness:** #65 topk→tile-IR (or a written exemption); #71 offsets→volatile
  uniforms (stabilize what template identity MEANS before the schedule identities join it);
  #87.
- **Absorbed, not sequenced:** #78 I3/I4 (I3 is a MOVE under this design — do not build the
  bespoke pass) and #83's structural-generator tranche (P0's deletions discharge it).
- **Live-loop gates (P3/channel, not P0):** #89 tape guard-miss; the schedule request channel
  (contract.md's shape extended); browser timestamp-query timing RPC.
- **Coordination:** #76's quantized-operand format must be expressible as named ScheduleStates
  (§2's registry-as-catalog) — checked at its phase-1 design review.
- **Orthogonal:** #66 lands or parks cleanly before P0's src churn.

### P0 — the semantic-IR boundary (S2 — the five deliverables + structural gates)

**Ruling S2:** P0 is RE-FOUNDED. It is bigger than the v1 "reify + null differential" draft
because that draft was gameable (R22). P0's deliverables:

**(a)** the **semantic IR node set** — exact semantic node types (including weave/rearrange —
F19; the NCD surface's algebraic weave node, so faithful routing has an owner and structure
isn't hidden in persisted Bezier points).
**(b)** the **schedule schema per S1** — the three tiers, the typed predicate AST (R12), the
typed sync relations (R3), `ProgramGridMap` (R4), the lemma schema (F27/F28).
**(c)** one-way **`applySchedule(semanticIR, state) → loweredTileIR`** — the inverse from
arbitrary TypeScript control flow does NOT exist; kernels that cannot be expressed in the
schema are authored atoms, not fake reified states (R6).
**(d)** an **ownership deletion/relocation table** for every structural field the imperative
tile-IR and compiler currently own (`forRange`/`forStride`, barriers, shared arrays, thread
IDs, workgroup size, pointer-kind placement, register blocks, the compiler's vectorization/
subgroup/threads-per-row/block-layout/LICM/unroll auto-passes — R6 enumerates them), with an
**executable NO-SECOND-OWNER assertion** (S2 — no source generator or lower IR contains a
second value for those facts; the ownership-derivation precedent's assert-agreement made
concrete for structure).
**(e)** the **byte differential ON TOP of (a)–(d)** — byte-identical regeneration across the
full kernel corpus (gates + suite as the executable check), run on BOTH plan paths (R21). No
behavior change. Null-stability proven.

**Plus R22's structural gates** (byte-identity proves an adapter, not reification — these
make the adapter-cheat unrepresentable):
- corpus states serialize using ONLY the declared schema/moves/atoms/lemmas (no lossless AST
  dump, no `opaqueGeneratorId`);
- opaque bodies ONLY in the named authored set (F3's `visibility:"opaque"` + refusal reason);
- generators deleted or reduced to semantic builders.

**On the net-deletion threshold (Rejected/narrowed):** R22 also demanded a fixed numeric
net-deletion SLOC threshold; this is NARROWED. The weight-norm discipline + the S2 structural
gates (the no-second-owner assertion, schema-only serialization) already make the adapter-cheat
unrepresentable; a numeric SLOC threshold invites gaming in the other direction. But the
"explanation waiver" is DELETED (see §8 gate 5): growth without deletions = a design re-review,
not a paragraph.

### P1 — decorations → S1 requests + the R9 registry migration proof

**Ruling R9 (accepted as proposed):** the v1 "registry entries become named schedule states"
claim collapsed four things (algorithm family, applicability, choice, instance). The matmul
registry selects algorithm FAMILIES (`tiled` vs `gemv`), stores family-specific params, and
evaluates applicability from geometry/dtypes/transpose/epilogue/casts/pinning/subgroup — a
concrete ScheduleState cannot simultaneously be a reusable template, an applicability
predicate, AND a measured per-shape selection. P1 decomposes the registry into:

```
{ AlgorithmFamily, ScheduleTemplate, ApplicabilityPredicate, RealizerRequirement }
  + SelectionReceipt   // shape/device/realizer/version/measurement
```

- Decoration edits (tile sizes, vec width intent) re-emit through the ScheduleState path; the
  autotuner reads/writes the object; the variant registry is re-expressed as
  `ScheduleTemplate`s selected by `ApplicabilityPredicate`s (the predicate AST) with a
  `SelectionReceipt` recording the measured choice.
- P1 is rescoped to a **field-by-field migration PROOF** for GEMV, tiled matmul, split-K,
  swapGrid (now a `program-map`), and epilogues BEFORE any registry-deletion claim (R9). No
  "the old registry dies" until the proof is complete.

### P2 — macro moves incl. program-map + the pre-registered protocol (R23 + R24)

The acceptance narrative IS the flashattention derivation, rungs 0–7: start from naive
attention as three islands; `merge` (islands move — the S3 composite transaction, NOT a
ScheduleState `fuse`); then inside the island: `tile` → `stream` K/V through shared →
`recolor` accumulator → apply the admitted online-softmax lemma (F17: the executable sequence
is `lemma → recolor → recolor → group → stream`, refusal-first — dragging streaming onto naive
softmax is correctly refused because ordinary softmax has no head/body decomposition).
`program-map` is exercised here too (grouped traversal for L2 reuse, R4's reification test).

**The per-move differential (R23, accepted as proposed) — upgraded:**
- every state compared to the **SEMANTIC REFERENCE independently** (not just the prior state —
  comparing only to the preceding optimized state lets an early common error persist through
  the whole derivation), and **across lowered + compiled paths** (the cross-path corollary;
  the activation-threshold lesson — the compiled plan only builds on the 2nd+ execution, so a
  multi-step trajectory is mandatory).
- a **pre-registered boundary/adversarial corpus**: odd dims, tails, NaN/Inf, zero-size,
  subgroup widths; declared tolerances, repeat counts, trajectory length; metamorphic coverage
  for the invertible moves; race-specific tests for atoms.

**P2 performance gate — pre-registered WITH NUMBERS (R24, from the rulings verbatim):**
- geometric-mean slowdown **≤ 1.5×** vs the authored `fusedAttention` commit at merge time,
  **no single case > 2.0×**;
- on this **A100 class via Dawn/Vulkan**;
- shapes **{(B1, H8..12, S512, D64), (B1, H8..12, S2048, D64)}**, **f16 inputs / f32 accum**;
- **3 warmup + median-of-7**, tolerance per the parity harness.
- A failed case is REPORTED, never averaged away.
- (Vin may override the 1.5× / 2.0× before P2 starts; they are the recorded defaults.)

### P3 — the workbench: zoom-in + the perf feedback loop (measurement identity, R20)

Sol's islands editor gains the intra-island view: click an island → skeleton + declared
parameters rendered; P1 edits live from the UI; macro moves behind the same legality-refusal
UX (the ncd-surface jam). The feedback loop is two-tier:

1. *Static tier (instant, no GPU):* the realizer's `costModel` (§4) — shared-memory usage vs
   budget, occupancy PROXY (F13 — labeled, WebGPU can't measure true occupancy), bytes-moved
   estimate, roofline position. One artifact, two consumers; a second estimator is a defect.
2. *Measured tier (~100ms on settle):* island-in-isolation bench at real shapes — warm pool,
   median-of-N, read-late-steps discipline — via per-dispatch timestamp queries.

**Ruling R20 + F29 (measurement identity — folded in here):** every request/result is KEYED by
`{ stateHash (semantic+compilation), partitionHash, modelRevision, shapeCase, device/driver,
realizer+compilerVersion, protocol }`. Late results whose base revision is no longer current
are DROPPED (an async timing that returns after another edit must not attach to the wrong undo
state — R20). **Isolated vs in-context measurements are LABELED separately**, and an in-context
confirmation is REQUIRED before a winner enters the registry (isolation removes L2 warmth,
occupancy competition, dispatch overhead, fusion boundaries — it can reverse the ranking, R20).
Level targets name an accepted **COST ENVELOPE** (F29 — a target is honest only against a known
solved end state and the same local-step H/M convention; if multiple end states are valid, the
target names the envelope, not one privileged move sequence).

Every state in the undo stack carries its measurements (keyed as above): the ledger is a
climbing trace. **Early-arrival property:** declared parameters are tunable on AUTHORED states
too, so the first fiddle-able workbench — the real fused flash-attention kernel with its real
knobs and live numbers — ships at P0+P1+channel+P3, BEFORE any macro-move work. P2 deepens the
same workbench to structural gestures; it does not gate it.

**Measurement noise is a precondition, not a nicety** (corpus L3): the warmup-3/timed-N/median
discipline is what makes the landscape feel-navigable at all; ±8% noise teaches superstitions.

### P4 — LOCAL self-hosting (R25 — renamed; the conformance corpus is a SEPARATE standing gate)

**Ruling R25 (accepted as proposed):** P4 is renamed **local self-hosting** and its exit is a
LOCAL claim, not a grammar-completeness claim. Re-derive the framework's own fastest
hand-crafted kernels in-grammar at perf parity: attention BACKWARD (recomputation-identity +
D-precompute lemmas admitted, operand-residency intent exercised) and fused Adam (horizontal-
`pack` move). Exit: the authored set is empty or atoms-only, where "atom" passes §3.3's
MECHANICAL admissibility (so the authored set cannot be emptied by relabeling a resistant
attention/Adam kernel as an atom — R25's hole is closed).

**Grammar-completeness is a SEPARATE claim gated on an EXTERNAL published-kernel conformance
corpus** (R25 + R4): re-deriving two in-repo kernels proves only those two, not "state of the
art." The conformance corpus is a **standing gate**, not a phase — its first entry is R4's
grouped-matmul (now in-closure via `program-map`); it grows with external published kernels
(reviewer-approved). Grammar completeness remains FALSE while any corpus entry is
unrepresented. This is the honest form of the corpus claim "the closure contains the state of
the art" (containment analysis Part 3: X ≈ 85–90% expressible under the extended vocabulary) —
a claim about the EXTENDED grammar tested against outside kernels, not about ourselves.

May land after v2 starts; it gates the COMPLETENESS claim, not the editor.

v2 (Triton realizer) and v3 (CuTe) are separate campaign charters, written against P0–P2
evidence.

## 8. Acceptance gates (all of)

1. **P0 null differential + structural gates**: byte-identical regeneration, full corpus, on
   BOTH plan paths (R21); the three S2 structural gates (schema-only serialization, opaque only
   in the named authored set, generators deleted/reduced); the executable no-second-owner
   assertion (S2 (d)); the three identities byte-stable (R27). Byte-identity alone does NOT
   pass this gate (R22).
2. **Per-move differential (R23)**: every move state matched numerically against the SEMANTIC
   REFERENCE (not just the prior state) on the pre-registered boundary/adversarial corpus,
   across lowered + compiled paths, over the declared trajectory length.
3. **The P2 flashattention derivation script**, checked into tools/, reproducible, meeting the
   R24 pre-registered numbers (≤1.5× geomean / ≤2.0× worst case; the declared shapes/precision/
   protocol).
4. `npm run build` + full suite + `test:gates` green at every phase boundary.
5. **Weight-norm (the explanation waiver is DELETED — R22 narrowed)**: the campaign-end commit
   names its deletions; net structural-generator SLOC goes DOWN, or growth-without-deletions
   triggers a DESIGN RE-REVIEW (not a paragraph). No fixed numeric threshold (R22 narrowed —
   the structural gates + the weight-norm discipline carry it; a numeric threshold invites
   gaming in the other direction).
6. **Appendix A (checked in, R28)** reviewed and the object amended for any WGSL-ism found — done
   this round; all A-R1…A-R15 findings are folded into §2–§3 above.
7. **Measurement identity (R20)**: no measurement attaches to a state without its full key; late
   revision-mismatched results dropped; isolated vs in-context labeled; in-context confirmation
   before registry admission.

## 9. Shared spine with the model editor (docs/model-editor-charter.md)

One spine, two zoom regimes: model graph → module → island lane (sol's editor, shipped) →
intra-island ScheduleState (P3). Shared machinery, each landing ONCE:

- **the provenance spine (R11)** — a revisioned many-to-many map
  `ModelNodeUid → SemanticOpUid → (planFp, pos, oi) → IslandUid → ScheduleEntityUid`, origin
  sets preserved through fusion/lowering; ledger operands are UIDs + base revision, never
  positions or hashes; invalidation/rebase specified. This is the map on which "click model
  node → island → schedule object", diff, undo, and provenance are sound (R11 — the v1 "shared
  identity spine" was three incompatible schemes: model UIDs, island final-plan integer
  positions, and ScheduleState with no entity UIDs at all; this map is the bridge). Schedule
  entities (loops, staging edges, values, roles) get `ScheduleEntityUid`s so a `scheduleHash`
  is content identity while UIDs are operand identity for edits.
- **the ONE typed predicate AST (R12)** — §2.5; shared verbatim (a typed AST, not an untyped
  string grammar) with the charter's override selection and optimizer param-groups. The
  schedule context's leaves (loops, values, roles, origin sets) and the model context's leaves
  (model instances, params, semantic ops, islands) are all leaves of the one AST.
- the stable-identity + provenance-ledger idiom; the three-identity coordinates (§5); sequence-ui;
  legality-as-refusal-with-reason UX.
- the capability-profile/realizer idiom — but NOT a shared `Realizer` type (R13): the model
  editor's PyTorch-emit and the WGSL realizer are separate concrete instances; the "same registry
  shape" claim is deleted (charter §8 correspondingly).

Neither editor blocks the other; the spine items land once.

## 10. Non-goals (v1) and risks

**Non-goals:** Triton/CuTe emit; CUDA-only moves (`role-partition` is reserved-but-empty on
WGSL, corpus rec #4 — a locked palette entry, not a degenerate stored value); lemma DERIVATION
(admission only); general loop transformations beyond the move set; training-program schedules.

**Risks:** (a) attention-bwd/Adam factoring resists → authored hatch, planned (§6). (b) scope
creep toward a scheduling language → the grammar is FENCED to the EIGHT-move set (§3); new move
kinds require a design amendment. (c) altitude duplication with tile-IR / variant registry →
the S2 no-second-owner assertion (§7 P0 (d)) is a P0 EXIT criterion, not a later "leveling
review" (R22). (d) null-differential too strict for incidental codegen nondeterminism → fix the
nondeterminism, not the gate. (e) measurement noise corrupting the feel-landscape (corpus L3) →
the warmup/median discipline is a P3 precondition. (f) the provenance map's seams at zoom
boundaries (R11) → the map is an M0/P0 named artifact with specified invalidation/rebase, not an
asserted spine.

## Appendix A — paper Triton capability profile

Checked in as `docs/design-corpus/appendix-a-triton-profile.md` (R28). It profiles each S1
tier, move, atom, and lemma against ordinary public Triton, verdict `determination | request |
refused`, and locates the authority horizon (after semantic TTIR, before/inside TTGIR). Its
findings A-R1…A-R15 are the source material the rulings disposed of, and every one is folded
into §2–§3 above: the three-tier split (A-R1/A-R6/A-R7/A-R8), pipeline as `none | [...]`
requests (A-R9/A-R14), typed sync relations (A-R10/A-R11), semantic atom naming with realization
in receipts (A-R12/A-R13), and `program-map` / `ProgramGridMap` (A-R15). Any future
inexpressible entry is a §2 representation bug.
