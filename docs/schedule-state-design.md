# Schedule State: the intra-kernel stratum as data

**Status:** DRAFT FOR REVIEW (Vin) · 2026-07-11
**Depends on:** docs/islands-design.md (I0–I2 landed), docs/tile-ir-design.md, the ScheduleRecord spike (docs/spike-schedule-record-findings.md), the kernel-editor design corpus (containment analysis, 11-rung ladder).
**Sibling charter:** docs/model-editor-charter.md — shared spine, see §9.

## 0. Declaration (one sentence)

A kernel's schedule — its loop nest, staging levels, and thread-role assignment — is a
first-class data object from which tile-IR skeletons are DERIVED; the fusion detector,
the editor, and the autotuner are three writers of that one object.

## 1. Lineage and why

This is the sixth application of the house move "the latent decision becomes an object":
scalars→tensors-as-data, buffer assignment→planner-derived, compiled replay→built-from-IR,
variant selection→registry data (#61), dispatch partition→islands (I0–I2). Macro-structure
is the **last big latent decision**: today the loop nest, shared-memory staging, and
thread-role assignment live in the control flow of the TypeScript generator functions.
Consequences of latency (the disease this cures):

- Legality is scattered heuristics (claim cascade, epilogue chains). With schedule state,
  I3's "claim casts into the matmul dispatch" is a *move* checked against an object,
  not a new bespoke pass.
- The autotuner can only sweep a hand-enumerated variant list; schedule state gives it a
  real search space.
- The editor cannot zoom below an island; schedule state is the intra-island view.
- N hand-rolled structural variants (matmul K-split/chunk shapes, reduction skeletons)
  each re-encode structure that one algorithm+schedule factoring would carry once.

**The condition that makes this better rather than worse:** it must EARN DELETIONS.
If schedule state lands as a layer alongside the generators, it is a second dialect and
strictly negative under the weight-norm. Acceptance is §7's null differential + §8's
deletion list, not "the editor works."

## 2. The object

```
ScheduleState = {
  algorithm: AlgorithmRef          // the semantic computation (tile-IR expression DAG / island contents)
  skeleton:  SkeletonInstance      // macro-structure: loop nest, staging levels, role partition  ← NEW as data
  decorations: DecorationVector    // tile sizes, vec widths, workgroup dims, unroll  ← already data (spike-proven liftable)
}
```

- **Decorations** are already data (variant registry #61, expression seams #62); the
  ScheduleRecord spike proved they lift and re-emit byte-identically. They move INTO this
  object; the variant registry's entries become **named schedule states** (registry =
  a catalog of ScheduleStates, selection stays data). No duplicate ownership.
- **SkeletonInstance** is the new part: an instance of a fixed **skeleton grammar** —
  the structural idioms tile-IR already has (elementwise, row-program/reduction,
  tiled-matmul, attention-shaped) with explicit loop nest, staging edges
  (global→shared→register), and role partition (which invocations do which part).
  #63 already migrated all kernels onto these idioms; this names them as data.
- **Leveling rule:** every fact at one level. Schedule state owns structure + decorations;
  tile-IR expressions own semantics; the island owns membership. A fact appearing twice
  is a review-blocking defect.

### Enumerated axes with degenerate values (the v2/v3 insurance — see §5)

Represent dimensions as sets/levels even where WGSL is degenerate. Absences and booleans
are forbidden representations for these:

| Axis | v1 (WGSL) value | Reserved for |
|---|---|---|
| memory spaces | {global, workgroup-shared} | +{register-explicit, distributed-shared/cluster} |
| sync scopes | {workgroup} | +{subgroup, cluster, grid} |
| pipeline depth | decoration, pinned 1 | Triton num_stages / CuTe stages |
| thread hierarchy | {invocation, subgroup(gated), workgroup} | +{warp-role, cluster} |

## 3. The move grammar

Mutators on ScheduleState: `{tile, stream, recolor, fuse, pack, role-partition, pipeline}`
(the design-corpus grammar; `fuse` at this altitude = absorb an adjacent island's algorithm,
i.e. the islands merge seen from inside). Properties:

- **Invertibility** where it structurally holds (tile/untile, pack/unpack) — undo is
  inverse-op, mirroring islands merge∘split. Moves that lose information record their
  inverse's data in the provenance entry.
- **Legality core vs backend legality.** The core (move algebra laws, convexity of the
  affected region, staging well-formedness) is backend-neutral and lives with the object.
  Whether a *realizer* can honor the resulting state is a capability-profile question (§4).
  v1 code must never check WGSL facts inside core legality.
- **Atoms are not moves either.** Some primitives should be COMPOSED AROUND, never derived:
  data-dependent constructs (the f32 atomic-add CAS retry loop WebGPU forces) and, at
  CUDA altitude later, hardware intrinsics (wgmma, cp.async — exactly how CuTe treats
  them). An **atom** is a wrapped primitive with declared semantics (footprint, sync
  behavior, cost) that moves schedule around. Atoms are first-class grammar members, not
  escape hatches: scatter-add is an elementwise schedule composed around an
  `atomicAddF32` atom, not an opaque kernel. v1's atom set: {atomicAddF32-CAS,
  subgroup ops (feature-gated)}.
- **Lemmas are not moves.** Moves rearrange WHEN/WHERE the same arithmetic happens; their
  legality is structural. Some targets are unreachable by rearrangement: flashattention
  requires the online-softmax identity (accumulate softmax·V block-by-block, RESCALING the
  partial output by exp(m_old−m_new) whenever the running max rises) — an algebraic fact
  about exp that computes different intermediates for the same function. Such rewrites live
  in a separate **admitted-lemma library**: hand-proven entries with their own differential
  gates, applicable by the editor where their pattern matches. v1 ADMITS lemmas; it does
  not derive them (deriving = a theorem prover, out of scope). The library starts with the
  one entry the P2 acceptance needs.

## 4. Realizer registry

A realizer = how a ScheduleState becomes an executable kernel. Registry entry:

```
Realizer = { capabilityProfile, emit, costModel, verificationHarness }
```

- **v1 ships exactly one realizer:** tile-IR→WGSL (transparent to the bottom;
  what-you-sculpt-is-what-runs, byte-differential-gated).
- **v2 = Triton** (separate campaign, gated on §7 P0–P2 evidence): request-based authority
  (block shape / num_warps / num_stages), pays the CUDA tax for us, and unlocks the
  cross-backend differential — the same ScheduleState through two disjoint compiler stacks
  on one A100, numerically diffed. Design-round artifact required NOW: a one-page **paper
  Triton capability profile** (Appendix A) proving the v1 object can express Triton's
  surface and its authority horizon without WGSL-isms. Paper only; no emit code in v1.
- **v3 = CuTe** (gated on product need for determination-authority at CUDA altitude):
  thematically isomorphic (its layout algebra ≈ our move grammar; F₂ swizzles literal),
  but WE pay the mbarrier/pipeline codegen tax — justified only when the top rungs are
  the headline. Target the CuTe Python DSL, not C++ templates.
- **Never:** tile-IR growing its own CUDA emit (the rebuild-Triton trap, declined).

## 5. Identity and caching

- Fingerprint gains `scheduleHash` (FNV over SkeletonInstance + DecorationVector, same
  pattern as I1's boundaryHash) **and a realizer coordinate** — one field, present from
  day one even with a single realizer.
- **Null-stability requirement (load-bearing):** the ScheduleState DERIVED from today's
  codegen must hash identically every step and regenerate today's WGSL byte-identically,
  or every static graph re-lowers and caches churn. This is the same null-case bar I1 met.
- Cache-class discipline: scheduleHash is a key≠content cache key by construction —
  it gets the #92 seam-guard (regenerate-and-compare on sampled hits under STRICT).

## 6. The compression direction

Decision (recommended, the invasive-but-smaller option): **skeletons become derived from
schedule state**, not referenced by it. Existing structural generators die as their idioms
are absorbed. Deletion targets (named per house policy): the matmul structural-variant
axis (K-split/chunk shape enumeration — selection stays, structure derives), reduction/
row-program skeleton construction, elementwise loop scaffolding.
**Authored = not-yet-re-derived (migration staging, NOT an expressivity ceiling).**
A kernel marked **authored** (the islands `authored` kind extended down) is an opaque
ScheduleState whose DECORATIONS are tunable but whose internals take no macro moves —
because it hasn't been re-derived yet, not because it can't be. Three rules give the
hatch its teeth:
1. The authored set SHRINKS monotonically: each member is either re-derived
   (moves + lemmas + atoms) or decomposed into a schedule composed around atoms.
2. Anything claimed to be PERMANENTLY underivable is tracked as a named grammar failure
   (a defect in §3, not a shrug) — the corpus claim is that the closure of the grammar
   contains the state of the art, and that claim is falsified by permanent members.
3. Expressivity requirement fed back into §2: re-deriving attention backward at perf
   parity requires operand-RESIDENCY decorations (which operand lives in registers vs
   shared — the register×shared vec4 dot is a decoration, not magic). The object carries
   that axis from P0.
v1's expected authored members and their exits: attention backward (re-derive at the
self-hosting milestone — needs the recomputation-identity and D-precompute lemmas),
fused Adam (re-derive — needs a horizontal-pack move at multi-tensor altitude),
scatter-add (NOT authored: composed around the atomicAddF32 atom from P0).

## 7. Phases (each independently shippable, stage-4 style)

### P0 entry criteria (the prereq runway — task numbers from the board)

**Correctness floor:** #43 recorded-path deletions land FIRST (one codegen path for the
null differential to hold invariant, not two); #92 key≠content seam-guard (scheduleHash
is a new key of that class and P0 rewires FusionKernelCache); #73 strict-lifetime default
via #86 + #74 (live kernel swapping needs loud, FP-free guards); #67 executor transpose
bug (user-shaped graphs); #85 characterized (uncharacterized noise poisons differential
gates — understand, not necessarily fix).
**IR completeness:** #65 topk→tile-IR (or a written exemption); #71 offsets→volatile
uniforms (stabilize what template identity MEANS before scheduleHash joins it); #87.
**Absorbed, not sequenced:** #78 I3/I4 (I3 is a MOVE under this design — do not build the
bespoke pass) and #83's structural-generator tranche (P0's deletions discharge it).
**Live-loop gates (P3/channel, not P0):** #89 tape guard-miss; the schedule request
channel (contract.md's shape extended); browser timestamp-query timing RPC.
**Coordination:** #76's quantized-operand format must be expressible as named
ScheduleStates (§2's registry-as-catalog) — checked at its phase-1 design review.
**Orthogonal:** #66 lands or parks cleanly before P0's src churn.

- **P0 — reify + null differential.** Derive ScheduleState from every existing tile-IR
  kernel; regenerate; byte-diff against today's WGSL across the full kernel corpus
  (gates + suite as the executable check). No behavior change. Null-stability proven.
- **P1 — decorations through the object.** Decoration edits (tile sizes, vec width)
  re-emit through the ScheduleState path; the variant registry re-expressed as named
  states; autotuner reads/writes the object. Spike already proved the mechanics.
- **P2 — macro moves on one island.** The acceptance narrative IS the flashattention
  derivation, rungs 0–7: start from naive attention as three islands; merge (islands move);
  then inside the island: tile → stream K/V through shared → recolor accumulator →
  apply the admitted rescaling lemma. Gate: the derived kernel is numerically differential-
  clean vs naive AND lands within a stated factor (set before measuring) of the authored
  fusedAttention kernel's time. Per-move numerical differential mandatory.
- **P3 — the workbench: zoom-in + the perf feedback loop.** Sol's islands editor gains
  the intra-island view: click an island → skeleton + decorations rendered; P1 edits live
  from the UI; macro moves behind the same legality-refusal UX. (Engine channel for
  partition/schedule requests is the sibling islands work — contract.md is the template.)
  **The feedback loop is two-tier:**
  1. *Static tier (instant, no GPU):* computed from ScheduleState + device limits —
     shared-memory usage vs budget, occupancy proxy, bytes-moved estimate, roofline
     position. This IS the realizer's costModel (§4) — one artifact, two consumers
     (autotuner + workbench); building a second estimator is a defect.
  2. *Measured tier (~100ms on settle):* island-in-isolation bench at real shapes —
     warm pool, median-of-N, read-late-steps discipline — via per-dispatch timestamp
     queries (Dawn profiler attribution exists; the browser device already requests
     `timestamp-query`).
  Every state in the undo stack carries its measurements: the ledger is a climbing trace.
  **Note the early-arrival property:** decorations are tunable on AUTHORED states too, so
  the first fiddle-able workbench — the real fused flash-attention kernel with its real
  knobs and live numbers — ships at P0+P1+channel+P3, BEFORE any macro-move work. P2
  deepens the same workbench to structural gestures; it does not gate it.

- **P4 — self-hosting (the grammar-completeness gate).** Re-derive the framework's own
  fastest hand-crafted kernels in-grammar at perf parity: attention BACKWARD
  (recomputation-identity + D-precompute lemmas admitted, operand-residency decorations
  exercised) and fused Adam (horizontal-pack move). Exit: the authored set is empty or
  atoms-only. This is the executable form of the corpus claim "the closure contains the
  state of the art" — applied to ourselves first. May land after v2 starts; it gates the
  COMPLETENESS claim, not the editor.

v2 (Triton realizer) and v3 (CuTe) are separate campaign charters, written against P0–P2
evidence.

## 8. Acceptance gates (all of)

1. P0 null differential: byte-identical regeneration, full corpus; fingerprints byte-stable.
2. Per-move differential: every move application numerically matched against the pre-move
   state on real inputs (the cross-path corollary applied per-gesture).
3. The P2 flashattention derivation script, checked into tools/, reproducible.
4. `npm run build` + full suite + `test:gates` green at every phase boundary.
5. Weight-norm: the campaign-end commit names its deletions; net structural-generator
   SLOC goes DOWN or the campaign explains why not.
6. Appendix A (paper Triton profile) reviewed and the object amended for any WGSL-ism found.

## 9. Shared spine with the model editor (docs/model-editor-charter.md)

One spine, two zoom regimes: model graph → module → island lane (sol's editor, shipped) →
intra-island ScheduleState (P3). Shared machinery: stable-identity + provenance-ledger
idiom, fingerprint/identity coordinates, sequence-ui, the capability-profile/realizer
idiom (the model editor's PyTorch-emit is a realizer of the same shape), legality-as-
refusal-with-reason UX. Neither editor blocks the other; the spine items land once.

## 10. Non-goals (v1) and risks

**Non-goals:** Triton/CuTe emit; CUDA-only moves; lemma DERIVATION (admission only);
general loop transformations beyond the skeleton grammar; training-program schedules.
**Risks:** (a) attention-bwd/Adam factoring resists → authored hatch, planned;
(b) scope creep toward a scheduling language → the grammar is FENCED to the rungs 0–7
move set; new move types require a design amendment; (c) altitude duplication with
tile-IR/variant registry → leveling review is a P0 exit criterion; (d) null-differential
too strict for incidental codegen nondeterminism → fix the nondeterminism, not the gate.

## Appendix A — paper Triton capability profile (design-round artifact, to be written)

One page: for each move/decoration/axis of §2–§3, whether Triton can honor it
(determination / request / refused), and where its authority horizon truncates ours.
Written during review of this doc; any inexpressible entry is a v1 representation bug.
