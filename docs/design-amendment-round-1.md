# Design Amendment Round 1 — rulings

**Status:** RULINGS (orchestrator, 2026-07-11; Vin ratified the round) — normative input
for v2 of docs/schedule-state-design.md and docs/model-editor-charter.md.
**Inputs:** sol red-team memo (R1–R28), Appendix A (paper Triton profile), workbench
findings F1–F18, NCD findings F19–F34, the NCD-spike empirical results (skeleton-as-
projection qualified-valid; partition-as-coloring; FA-by-gestures), D2's ownership
lessons.

## The three structural rulings (everything else hangs off these)

**S1 — The object splits in three (R1 + R2 + F8, confirmed by Appendix A):**
- *Semantic schedule* (hashes into semantic identity): logical block shapes, loop/
  dispatch structure, no-materialization (fusion) edges, ProgramGridMap, named value
  lifetimes (F6), staging intent.
- *Backend requests* (compilation identity, not semantic identity): warp budget,
  pipeline requests as `none | [{loopUid, loadGroupUids, requestedStages}]` (R2 — `none`
  legal, no fake degenerate values), placement preferences, cache policy.
- *Realization receipts* (never hashed into schedule identity): WGSL workgroup x/y/z,
  exact vec-load forms, realized stages, CASLoop-vs-native atom realization (R5),
  measured results (R20 keys). DecorationVector as a single flat bag is DEAD.

**S2 — P0 is re-founded as the semantic-IR boundary (R6 + R22 + F4 + the NCD projection
result):** P0's deliverables are (a) the semantic IR node set (including weave/rearrange
— F19), (b) the schedule schema per S1, (c) one-way `applySchedule(semanticIR, state) →
loweredTileIR`, (d) an ownership deletion/relocation table for every structural field the
imperative tile-IR and compiler currently own, with an executable NO-SECOND-OWNER
assertion, (e) the byte differential ON TOP of (a–d) — plus R22's structural gates:
corpus states serialize using ONLY the declared schema/moves/atoms/lemmas; opaque bodies
only in the named authored set; generators deleted or reduced to semantic builders. The
loop-nest VIEW derives from semantic region × schedule state with a canonical ordering
rule (NCD projection result); thread roles/barriers/lanes remain schedule facts, NOT
projections (the spike proved they don't derive).

**S3 — Ownership boundaries between siblings (R7 + R8 + partition-as-coloring):**
`fuse` leaves the ScheduleState move set. Membership is owned by Partition alone; the
editor's fuse gesture is a COMPOSITE TRANSACTION at the next altitude (validate interior
schedule → merge(P,a,b) → attach state → ONE provenance record with both hashes; roll
back all on realization failure). The ℓ0-coloring remains the editor's VIEW of the
boundary, not a second owner. `algorithm` becomes a `SemanticRegionUid` FOREIGN KEY
owned by the semantic graph (R8); partition transactions mint new region UIDs.

## Accepted as proposed (apply mechanically)

- **R4**: `program-map` move + bijection-valued `ProgramGridMap` (identity/swap/grouped/
  checked-affine); legality = one-to-one in-bounds; swapGrid + Triton grouped-matmul as
  mandatory reification tests. (Already partially amended; complete it.)
- **R5 + R25**: semantic atom naming (`atomicAdd<f32>` w/ order/scope/NaN contract);
  realization in receipts; MECHANICAL admissibility (single primitive effect or hardware
  intrinsic; no composite loops/whole algorithms); subgroup primitives enumerated
  individually. P4 renamed **"local self-hosting"**; grammar-completeness is a SEPARATE
  claim gated on an external published-kernel conformance corpus (R4's grouped-matmul
  now in-closure via program-map is its first entry).
- **R9**: registry decomposes into {AlgorithmFamily, ScheduleTemplate,
  ApplicabilityPredicate, RealizerRequirement} + SelectionReceipt; P1 rescoped to a
  field-by-field migration proof (gemv/tiled/split-K/swapGrid/epilogues) before any
  registry deletion claim.
- **R10 + F3 + F7**: authored entries publish a TYPED parameter schema (dependent
  constraints, derived geometry, capability predicate, cache-key encoder); generic
  decorations on opaque skeletons are REFUSED; edits commit only after a checked
  realization receipt.
- **R11**: the provenance spine is a revisioned many-to-many map
  `ModelNodeUid → SemanticOpUid → (planFp, pos, oi) → IslandUid → ScheduleEntityUid`,
  origin sets preserved through fusion/lowering; ledger operands are UIDs + base
  revision, never positions or hashes; invalidation/rebase specified.
- **R12 + F5**: ONE TYPED predicate AST with explicit leaf domains (model instances,
  params, semantic ops, islands, schedule entities incl. role participants) + shared
  combinators; serialization + type errors gated in M0/P0.
- **R14 + R15**: edit taxonomy = `functionPreservingMorph` (constructive parameter
  mapping + identity equation + inverse preconditions, defined as REWRITE SCHEMAS not
  verbs) vs `behaviorChangingEdit` (preview/commit + measured before/after). The
  candidate six re-labeled accordingly (only untie-by-copy and zero-gated residual
  insertion are morphs).
- **R16**: define `TorchlettePyTorchSubset`; claim INJECTIVE EMIT into it (gates:
  canonicalize(emit(G))==G, alias-group equality, state-dict round-trip, independent
  forward parity). "Bijection" language dies until an inverse parser exists.
- **R17**: live edits are transactions: `{schemaVersion, baseModelRevision, opId}`,
  CAS at a defined generation epoch, immutable snapshot per generation, quiesce/fence
  before demotion (existing house law cited), publish-after-validation, stale edits
  rejected with current revision.
- **R18 + R26**: checkpoint manifest (family ID, generator schema digest, param UID,
  path, shape/dtype, alias group, role, migration chain), fail-closed on unknown
  versions; parity gates use INDEPENDENT upstream references + full manifest-driven
  parameter/alias comparison + multi-input probes; exact-zero only for claimed morphisms
  via the mapping + identity equation, not one logit probe.
- **R19**: semantic edit history (immutable, UID operands, inverse payloads, weight/
  optimizer snapshots where inverses lose information) SPLIT from realization history
  (invalidated on re-record). Cross-stratum undo rebases against UIDs or refuses with a
  stable conflict reason.
- **R20 + F29**: measurement identity = {state hash, partition hash, model revision,
  shape case, device/driver, realizer+compiler version, protocol}; late results dropped
  on revision mismatch; isolated vs in-context labeled; in-context confirmation before
  registry admission. Level targets name an accepted COST ENVELOPE (F29).
- **R23**: per-move differential upgraded: every state compared to the SEMANTIC
  REFERENCE independently (not just the prior state) and across lowered+compiled paths;
  pre-registered boundary/adversarial corpus (odd dims, tails, NaN/Inf, zero-size,
  subgroup widths), tolerances, repeats, trajectory length (the activation-threshold
  lesson cited).
- **R27**: schedule identity = canonical serialization + strong digest, FULL canonical
  equality verified on every cache hit (sampling audits regeneration only); identity
  gains {schemaVersion, capabilityProfileVersion, emitterVersion, compilerVersion,
  targetArch, featureSet}; semantic vs realization vs artifact-cache identities kept
  separate.
- **R28**: the design corpus artifacts get CHECKED IN under docs/design-corpus/
  (containment/ladder analysis, spike findings, napkin-paper analysis) — done in this
  round's commits; every move gets a typed before/after schema in the v2 doc.
- **F26/F27/F28**: challenge/attempt objects separate from terms (share format
  distinguishes term/challenge/attempt); lemma carried state first-class in the lemma
  schema (accumulator/state-machine representation from which inspection views derive);
  jam→lemma binding via PROOF-OBLIGATION IDs, never refusal-text matching.
- **F23/F30 + H-definition**: costs carried symbolic AND evaluated; H defined explicitly
  as local partitioned traffic (per-napkin-step), with whole-dispatch traffic as a
  separate derived figure; scalar carried state distinguished from row-sized
  materialization in the cost algebra.

## Modified acceptances

- **R3**: accepted as SCHEMA (typed relations MemoryEffect/Barrier/Atomic + backend
  capability GRAPH), but the "enumerated axes" spirit survives inside the typed forms —
  the relations' value domains are still explicit sets, never booleans/absences.
- **R13**: the "same registry shape" claim DIES. PyTorch-emit and kernel realizers stay
  separate concrete things; a generic Realizer<...> protocol is deferred until a third
  genuine instance exists (rule of three).
- **R21**: already corrected pre-review at the task level; the doc formalizes: P0's null
  differential runs at KERNEL-codegen altitude and executes on BOTH surviving plan paths
  (lowered + generated) wherever a kernel is reachable from both; the recorded-build
  deletion is explicitly NOT a P0 predecessor.
- **R24**: pre-registration WITH NUMBERS, set now: P2 = geometric-mean slowdown ≤1.5×
  vs the authored fusedAttention commit at merge time, no single case >2.0×, on this
  A100 class via Dawn/Vulkan, shapes {(B1,H8..12,S512,D64), (B1,H8..12,S2048,D64)},
  f16 inputs/f32 accum, 3 warmup + median-of-7, tolerance per the parity harness. M5 =
  3 seeded runs per arm, pre-registered effect direction + minimum magnitude vs control
  at fixed step count; a failed case reported, never averaged away. (Vin may override
  the 1.5×/2.0× before P2 starts; they are defaults, recorded here.)

## Rejected / narrowed

- R22's demand for "a fixed net-deletion threshold": narrowed — the weight-norm
  discipline + the S2 structural gates (no-second-owner assertion, schema-only
  serialization) already make the adapter-cheat unrepresentable; a numeric SLOC
  threshold invites gaming in the other direction. The "explanation waiver" is deleted
  though: growth without deletions = design re-review, not a paragraph.
- The seven-move fence: formally amended to EIGHT moves (program-map joins; R4) rather
  than withdrawing the completeness claim; completeness itself is re-scoped per R25.

## Consequences for phases

- P0 = S2 (semantic IR + schema + applySchedule + ownership table + structural gates +
  byte differential). Bigger than drafted; it was always the real work.
- P1 = decorations→S1 requests + the R9 registry migration proof.
- P2 = macro moves incl. program-map + the pre-registered protocol above.
- P3 = the workbench/game surface (already shipped v1 vs hand-authored terms; binds to
  real states after P0).
- P4 = LOCAL self-hosting; the conformance corpus is a separate standing gate.
- M0 gains: typed predicate AST, checkpoint manifest, transaction protocol, provenance
  spine — each a named artifact.
