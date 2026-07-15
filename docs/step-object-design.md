# The Step Object — a whole training step as first-class data

**Status:** DESIGN ONLY (task #98, stage-1/3 protocol). No mechanism lands this
campaign — this doc plus a staged shippable plan behind differentials. Ratified
inputs: Vin's interview 2026-07-15 (the five rulings, §1); they are constraints,
not proposals.
**Lineage:** the eighth application of the house move "the latent decision
becomes an object" (`schedule-state-design.md §1` enumerates the first seven:
scalars→tensors-as-data, buffer-assignment→planner-derived, compiled-replay→
built-from-IR, variant-selection→registry-data #61, dispatch-partition→islands
I0–I2, liveness-classification→derived-at-inquiry #70, kernel-schedule→three-tier
object). The step object is the same maneuver one stratum UP from the kernel
schedule: the whole training step, today latent in driver control flow and
re-derived every iteration, becomes one datum in two phases.
**Normative ground truth (all cited inline):** `staged-execution-phase1.md`,
`phase2a.md`, `phase2b.md` (the LIVE tape docs — `dispatch-tape-design.md` is
SUPERSEDED, its banner line 1); `islands-design.md` (I0/I1/I2a landed, I2b
reverted); `stage4-compile-from-ir.md` (Task #96 LANDED, Task #97 STOP, the #43
recorded-build sunset MAP); `schedule-state-design.md` + `architecture-debt.md`
(the house grammar + sin taxonomy); `ownership-derivation-design.md` (derivation-
replaces-observation, #70); `scoped-memory-design.md` (#66, the epoch vocabulary
the boundary must share).

---

## 0. Declaration (one sentence)

A whole training step — its declared boundary contract, its dynamic-slot,
recompute-segment and partition facets, and its ring/runahead config — is ONE
first-class object that exists in two phases, a *declared* source form
(`api.capture()`) and a *witnessed* compiled form (the step-tape skeleton), with
the same record-then-cut-over lifecycle as a compiled plan one stratum up; from
that one object the executor derives what the observation layer today
reconstructs by watching executions.

If a section below cannot be stated as a declaration in one sentence of that
grammar, it is reshaped before it lands (the house one-sentence test).

---

## 1. The five ratified rulings (constraints, not proposals)

These are Vin's interview answers, 2026-07-15. They are recorded verbatim in
intent; the doc's job is to make them structural, never to re-litigate them.

1. **Identity — capture and tape are ONE object in two phases.** `api.capture()`
   (the declared boundary, `phase2a.md §1`) and the step-tape skeleton (the
   witnessed program, `step-tape.ts` / `step-tape-replay.ts`) are the SAME object:
   declaration is the source form, the witnessed tape is its compiled form. Same
   record-then-cut-over lifecycle as compiled plans (`phase1.md §2.2` — "records on
   execution 1 and cuts over on execution 2+"), one stratum up. **No second
   whole-step mechanism may be born.** Corollary: the #97 harvest residual
   (`stage4 §Task #97`, the `contiguous[512,768]` checkpoint-recompute prune)
   resolves by finalizing harvest at WITNESS time — tape eligibility is two
   consecutive identical *executed* steps (`step-tape.ts:730`), so every cross-plan
   read was physically observed.

2. **The declared-lifetime dividend is the DESTINATION.** Inside a captured
   boundary, lifetimes are DERIVED from the program (`phase2b.md §5`); the
   observation predicates `everSurvived`/`everReadback`/`everAliased`
   (`observed-liveness.ts:123`, the stage-3 A/B observation layer) RETIRE on the
   captured path. This is the same maneuver #70 ran for liveness classification
   (`ownership-derivation-design.md §0` — DERIVED at inquiry, never a stored slot).
   Sequenced LATE,
   per §5's own re-open condition (captured path warm + watcher cost measured), but
   the design aims at it explicitly, not as a maybe.

3. **Checkpointing becomes first-class in the step object.** Recompute segments
   are DECLARED FACTS of the step program; the fast stack (compiled plans, tape,
   arena) serves them; the accidental
   `setBufferArenaDisabled(true)` production bypass
   (`webgpu-gpt2-trainer.ts:157`, commit b66ead78) dies. Recompute-vs-retain is a
   per-segment knob — data on the step object, the same KIND of datum as fuse/split.

4. **Partition (islands) is a facet of the step object.** `(G, P, device)` is per
   STEP, not per plan (`islands-design.md`, the per-plan boundaryHash at
   `executor.ts:259`). `ReRecordChannel` (`src/schedule/moves/fuse.ts:65`)
   generalizes into THE step-object edit channel; the P3 editor binds to one seam.

5. **The death list is the acceptance criteria** (staged, gated): the recorded
   build (via witness-time harvest — only AFTER proven on the CHECKPOINT config),
   `guardMiss` recovery → a should-never-fire assert (after a zero-fire soak), the
   observation predicates inside boundaries, the checkpoint bypass, the stale docs.

---

## 2. The object (schema-level definition)

The step object follows the house grammar precisely (`schedule-state-design.md
§0/§2`): a TYPED STATE that hashes into an identity, TYPED MOVES that are its only
mutators, a CANONICAL DIGEST, and OBLIGATIONS/RECEIPTS that hash into neither. It
is `ScheduleState` one stratum up: where schedule-state is the intra-kernel
object, the step object is the whole-step object.

### 2.1 The two phases (S1 — the identity split)

```
StepObject = {
  // --- DECLARED phase (the source form; hashes into STEP identity) ---
  declaration: {
    boundary:   BoundaryContract      // arg list + closure freeze (phase2a §2)
    slots:      DynamicSlotDecl[]     // per-step-varying inputs, declared
    recompute:  RecomputeSegment[]    // checkpoint segments as facts (ruling 3)
    partition:  StepPartition         // (G, P, device) per step (ruling 4)
    ring:       RingConfig            // K runahead depth + loss/diag outputs
  }
  // --- WITNESSED phase (the compiled form; DERIVED, not authored) ---
  skeleton:   Skeleton | null        // step-tape-replay.ts:132 — ORDERED plans[]
  // --- foreign keys / receipts (hash into NEITHER) ---
  epoch:      EpochId                 // scoped-memory §1 vocabulary (NOT "step")
  receipts:   StepReceipts           // guard-miss counters, verify diffs, K-fill
}
```

**Phase discipline (the record-then-cut-over lifecycle, ruling 1).**
- `declaration` is authored (or derived from `api.capture`'s arg boundary). It is
  the SOURCE. It hashes into the step identity (the `bucketKey`, `step-tape.ts`
  guard 2) via its structural fields ONLY — slot VALUES, recompute-vs-retain data,
  and partition device-key are structure; per-step scalar bytes are not.
- `skeleton` is DERIVED by witnessing: the recorder observes two consecutive
  structurally-identical *executed* steps (`stEndStep`, `step-tape.ts:683–757`;
  eligibility at :730) and promotes the ordered plan sequence
  (`stPromoteEligibleSkeleton`, `step-tape-replay.ts:444`). It is the COMPILED
  form. It owns NO buffers (`phase1.md §2.1` — "The tape references template ids
  and slot ids, never raw buffers"; guard 4 subordinates it to plan validity).
- A hit re-dresses the retained `planNodes` skeleton and calls the EXISTING
  `executeLoweredPlan` (`step-tape-replay.ts:994`); the body does not re-run
  (`phase2b.md §1` — the whole step IS the tape, no short-circuit).

### 2.2 Identity / digest

The step object's digest is the pair `(declaration-structural-hash, ordered plan
fp sequence)` — exactly the tape's `bucketKey = hash(structKey) + ordered plan
fps` (`phase2b.md INC-2B` — measured byte-stable, `tapeCount=1` over 18 steps,
both fused and foreach arms). Slot VALUES, per-step scalars (batch data,
bias-corrected `step_size`, scale), and receipts are canonicalized OUT of the
digest (the schedule-state discipline: receipts hash into neither identity;
`phase2b.md §6` — "the command-stream diff must canonicalize the per-step-varying
DECLARED slot bytes OUT"). This is the frozen-scalar defense expressed as an
identity rule: a value that varies is either a declared slot (data) or the digest
refuses it.

### 2.3 Guard semantics — the 6 guard classes as typed refusals against the declaration

The tape's six guards (`step-tape.ts`, enumerated in the recorder + replay
headers) recast as typed refusals of a declaration:

| # | today (tape guard) | step-object recast (typed refusal) |
|---|---|---|
| 1 | structGen — op-sequence counter delta (`step-tape.ts:24`) | the declaration's op-structure changed → **StructureMiss** (new `declaration` hash) |
| 2 | bucketKey — plan fps + write positions + readback params | the declaration's identity bucket changed → **BucketMiss** (new StepObject) |
| 3 | scalar/payload coverage — `diffImages` byte-diff (`:534–631`) | a per-step-varying byte not covered by a declared slot → **UndeclaredVariance** (the PAYLOAD-THRASH sibling, `:598`) |
| 4 | plan validity — `stInvalidateTemplate` (`:779`) | a referenced plan invalidated → **PlanInvalid** (invalidation SUBORDINATION, below) |
| 5 | epoch/regime — engine epoch + stepScopedCleanup (`:805`) | the boundary regime changed → **EpochMiss** (shares scoped-memory §1 epoch ids) |
| 6 | STRICT_TAPE paranoia (`:72`) | any miss/verify-diff THROWS under strict → **the CI net** |

Every refusal is LOUD-and-correct: it falls back to the normal build+execute path
(which re-witnesses) and, under `TORCHLETTE_STEP_TAPE`'s STRICT mode, throws
(`phase1.md §2.4` — "A silent stale replay is the failure mode this design exists
to prevent"). There is no benign-divergence category (the clip-divergence lesson,
CLAUDE.md — descending-faster IS a bug).

### 2.4 Invalidation subordination

The step object holds NO validity of its own (`phase1.md §2.1`). Two masters:
- **Plan validity** — guard 4: any referenced compiled plan's staleness/eviction/
  teardown cascades to the step object (`stInvalidateTemplate`).
- **Device key** — the partition facet is `(G, P, device)`; a device change is a
  BucketMiss (islands R2 device-keyed legality — a partition legal on A100 may be
  illegal elsewhere). Param data UPDATING in place is NOT invalidation (`phase2b.md
  §6` census #9 — closure state mutated in place is the point of training).

### 2.5 What fields the step object does NOT have (zero-schema-delta discipline)

Per the schedule-state discipline (`schedule-state-design.md §2` — receipts hash
into NEITHER; R22 — byte-identical regeneration alone does not earn admission) the
step object deliberately does NOT carry:
- **Raw buffers or buffer ownership.** Owned by the planner registry (stage-4
  phase 1.5). The step object references template ids + slot ids only.
- **Per-step scalar values** (step counter, scale, LR, batch data). Those are
  DATA delivered through declared slots (TAG_WRITE / TAG_UNIFORM / scalar-table),
  never fields — else they are a frozen-scalar cache waiting to happen.
- **A second liveness store.** Liveness is DERIVED (ruling 2 / #70), not a slot on
  the object.
- **A guard-miss recovery mechanism as a permanent field.** It is a receipt
  (counter) that trends to zero and then becomes a should-never-fire assert
  (ruling 5). It is not part of the object's identity.

The object is small by construction: it is the UNION of facets that already exist
as separate mechanisms, re-keyed to one identity — not new state.

---

## 3. The unification map — five mechanisms become five facets

For each mechanism: what it is TODAY, what FACET it becomes, what gets DELETED
(named), and the SEAM where old and new must assert agreement during migration.

### 3.1 `api.capture()` → the DECLARED phase

- **Today:** `CapturedFn` whose coverage is the argument list (`phase2a.md §1/§2`);
  tensor args are warm slots, plain-value args hash into the bucket key, closure
  values are frozen (jax.jit semantics). Decode-shaped only; noGrad, no optimizer.
- **Facet:** the `declaration` half of the StepObject. The arg-boundary contract
  IS the `BoundaryContract`; upload interception IS the slot declaration; the ring
  option IS `RingConfig`.
- **Deletes:** nothing yet in 2a (already minimal — "capture adds ZERO new env
  flags"). The DUAL life of "capture the API" vs "tape the recorder" collapses:
  they were always one object; the doc names them one.
- **Agreement seam:** `stCaptureCompiledStep` (`step-tape-replay.ts:327`) already
  accumulates the per-step candidates the declaration's plans must match. The
  INVARIANT (`phase2b.md SURFACE 1`): `candidates.map(c=>c.fp)` equals the tape's
  ordered plan fps — a strict, ordered, length-checked equality. That equality IS
  the declaration↔witness agreement assertion.

### 3.2 The step-tape skeleton → the WITNESSED phase

- **Today:** `Skeleton` (`step-tape-replay.ts:132`) with `plans: SkeletonPlan[]`,
  promoted from accumulated `Candidate`s; replay re-dresses retained `planNodes`
  and calls `executeLoweredPlan`.
- **Facet:** the `skeleton` half. The witnessed compiled form of the declaration.
- **Deletes:** the module-level single-candidate assumption is already generalized
  to `candidates[]` (inc-2b); no new deletion — the mechanism EXISTS.
- **Agreement seam:** `TAPE_VERIFY=N` (`phase1.md §2.4` guard 6): every Nth call
  runs the NORMAL path and byte-compares command streams via `diffStreams`. For
  training the verify step must advance the in-place optimizer state EXACTLY ONCE
  (`phase2b.md §6` — never double-step the optimizer). This is the standing
  declaration↔witness shadow gate.

### 3.3 Islands partition → the `partition` facet

- **Today:** `Partition`/`Island` reified (I0), `boundaryHash` mixed into the plan
  fingerprint (I1, `executor.ts:259`), the detector re-expressed as a
  merge-proposing policy (I2a); per-PLAN. I2b (gap-spanning) built-measured-null-
  reverted (`islands-design.md §I2 findings` — the residual is matmul-adjacent
  singleton casts, CLAIM-altitude work, not detector generality).
- **Facet:** `StepPartition = (G, P, device)` per STEP (ruling 4). The step object
  lifts the partition from per-plan to per-step: a step has ONE partition of its
  whole semantic graph, of which the per-plan boundaryHashes are a projection.
- **Deletes (on the collapse, per `islands-design.md §7`, staged behind I3/I4 — NOT
  this campaign):** the four bespoke partitioning mechanisms (consecutive-scan
  grouping already deleted at I2a; the executor action vocabulary `adam-batch` /
  `batched-reduction` / `matmul-epilogue` / `row-program`; chunking's ad-hoc
  dispatch loops) collapse to one object + three policies.
- **Agreement seam:** `boundaryHash` is already a pure function of `G` and mixes
  into the master key (I1 null-test-clean). The step-partition's per-plan
  projection must reproduce the per-plan boundaryHashes byte-identically — the I1
  null test, run at step altitude.

### 3.4 Harvest → witness-time finalization (the #97 resolution)

- **Today:** two harvest mechanisms coexist. The RECORDED build harvests by
  RECORDING the actual dispatch (build-WITH-execution) — load-bearing correctness
  net (`stage4 §Task #43` A2 master 1). The GENERATED build reconstructs harvest
  from IR (build-WITHOUT-execution) and PRUNES cross-plan reads the observation
  layer never saw — the #97 residual (`stage4 §Task #97 stages 3–4`: the
  checkpointed-MLP `contiguous[512,768]` recomputed in backward; its producing
  forward template prunes it because the backward reader resolves LOWERED, invisible
  to `observeConsumed`; `graphHeld=FALSE`, so the #97 stage-2 oracle does not cover
  it; and at the forward-plan build seam the backward consumer DOES NOT EXIST YET).
- **Facet:** harvest finalized at TAPE-ELIGIBILITY time (ruling 1 corollary). The
  step object's `skeleton` becomes eligible only after two consecutive identical
  *executed* steps (`step-tape.ts:730`). By construction those steps RAN the whole
  program — forward AND backward AND recompute. So every cross-plan read (including
  the lowered backward read of a checkpoint-recompute activation) was PHYSICALLY
  OBSERVED in the witnessing steps. The witnessed harvest set is the UNION of
  actual reads across the two witness steps — the build-WITH-execution guarantee the
  recorded build has, promoted to the tape.
- **Why this covers #97 where derivation could not:** #97 STOPPED because "no
  whole-step graph derivation is available at the forward-plan build seam" — the
  backward consumer is built later. The step object does not derive at forward-build
  time; it WITNESSES at end-of-step time, when the backward pass HAS run and its
  reads are facts. `stage4 §Task #97` names exactly this as the unblock: "the
  recorded build is deletable once the GENERATED harvest is sound for
  build-without-execution cross-plan forward→backward reads (a whole-step-graph or
  execution-witnessed harvest — a separate campaign)." This IS that campaign; the
  witnessed step object is the execution-witnessed harvest.
- **Never-witnessed (transient / cold) plans:** a plan that never reaches two
  consecutive executed steps has no eligible tape. It stays on the recorded build —
  the recorded build's CURRENT role (`stage4 §Task #43` A2 master 1: the uncovered-
  plan census fallback; strided views, chunked >128 MB, batch >64, non-f32, CoW,
  contiguous-copy prologues). The recorded build is NOT deleted until witness
  coverage subsumes it (ruling 5, gated). Transient plans run recorded ONCE, exactly
  as today.
- **Agreement seam / differential gate:** the witnessed harvest set must EQUAL the
  recorded harvest set on every plan the recorded build still covers — a set-parity
  diff (the stage-3 B set-parity gate shape) run with checkpointing ON. Only when
  that diff is empty across the mandatory config matrix (§6) does the recorded
  build's harvest master retire.

### 3.5 The checkpoint bypass → the `recompute` facet (the bypass dies)

- **Today:** `WebGPUGPT2Trainer.initialize()` calls `setBufferArenaDisabled(true)`
  when checkpointing (`webgpu-gpt2-trainer.ts:157`, commit b66ead78) — the whole
  compiled/tape/arena stack is DORMANT there; plans run lowered. Selective
  checkpointing = MLP inside / attention outside; recompute = fresh per-layer plans
  at first backward. 4 canonical templates/step (2 steady, 2 transient).
- **Facet:** `RecomputeSegment[]` — recompute-vs-retain declared PER SEGMENT on the
  step object (ruling 3), the same KIND of datum as fuse/split. The fast stack
  SERVES the segments (the arena packs recompute activations against the declared
  liveness) instead of being switched off to work around them.
- **Deletes:** `setBufferArenaDisabled` and its `TORCHLETTE_CHECKPOINT_ARENA`
  opt-out (the accidental production bypass, ruling 5). Once the tape/arena serve
  checkpointing correctly, the bypass is dead debt.
- **Agreement seam:** the bypass-OFF path (arena+compiled+tape under checkpointing)
  must produce baseline-EXACT loss and peak within budget vs the bypass-ON path —
  the b66ead78 measurement (124M/batch1/seq256: steady 6.6→2.6 GB, loss identical)
  becomes the A/B the deletion must match or beat. This is the config #97 stopped on;
  it is MANDATORY in the witness-harvest gate (§6).

---

## 4. Witness-time harvest — the precise design

The load-bearing new idea. Stated as one declaration: **a step object's harvest set
is the union of the cross-plan reads physically observed during the two consecutive
identical executed steps that make its tape eligible.**

> **STATUS: LANDED (task #98 phase 4, 2026-07-15).** The mechanism ships across
> commits `bdfa2e02` (failing-first gate + the harvest-keep SINK), `4dbfacd8` (the
> witness recorder — the #97 unblock), `a498d319` (the config matrix + in-suite
> gate). The #97 STOP is unblocked: distil@512 + selective checkpointing runs CLEAN
> with the recorded build's harvest role removed, where the generated prune
> deterministically threw `Input not ready: node contiguous[512,768]` before.
>
> **One refinement vs the §4.1 sketch, forced by the checkpoint config: the witness
> window is PER PRODUCER, not per whole-step.** Selective checkpointing re-fingerprints
> the recompute READER plans on every backward (fresh per-layer plans, `§3.5`), so no
> two WHOLE steps are ever structurally identical — the whole-step tape-eligibility
> gate (`step-tape.ts` compiled-only + `diffImages`) NEVER fires on the checkpoint
> config, and a publish gated on it would never run. But the PRODUCER template of the
> cross-plan activation (the checkpointed-MLP `contiguous[512,768]` producer) recurs
> identically every step. So the recorder publishes a producer's witnessed harvest
> once it has been read with the SAME pair set in two consecutive steps IT APPEARED IN
> (`reconcileWitnessReads`, K_w=2 applied at the producer stratum — ruling 2 honored,
> not widened). Publication is MONOTONE-SAFE: a set is only ever GROWN, never shrunk
> (a superset keep is never a UAF — pruning too FEW never crashes), so a disagreement
> republishes the UNION and counts `witnessVariances` rather than dropping a witnessed
> pair. This is strictly SOUNDER than the whole-step gate would have been: it fires on
> configs the whole-step gate structurally cannot cover.
>
> **Deferred to the FULL recorded-build sunset (§4.3, a later product decision):**
> under the FULL harvest-deletion diff, a GradScaler inf-skip run surfaces a SEPARATE
> never-witnessed class — a `shape=[]` live-scalar reclaimed at markStep (a `[lifetime]
> reading RECLAIMED`, NOT an `Input not ready`), orthogonal to the checkpoint-recompute
> harvest this phase covers. Phase 4 does NOT delete the recorded build (§4.3 keeps it
> for never-witnessed classes), so this does not occur in the phase-4 config; it is a
> STOP finding for the eventual full sunset, not for this phase.

### 4.1 Mechanism

1. **Observation is already there.** The recorder is a faithful pure observer
   (`phase2b.md INC-1` corollary — "18 steps observed, no crash"). Recording runs
   the NORMAL build+execute path; `stBeginPlan` captures each plan's node images and
   `stEndStep` compares consecutive steps. The witnessing steps EXECUTE the full
   program including backward and recompute.
2. **Harvest set = observed reads.** During each witnessing step, every cross-plan
   read that resolves a producer's result — whether through the compiled external-slot
   bind (`observeConsumed`, `compiled-plan.ts:~1435`), a persistent-slot bind, an
   inline view resolution, or a LOWERED `getInputStorage` read (the invisible path
   that broke #97) — is recorded against the producing plan. The step object's
   witnessed harvest for a producer plan is the UNION of its observed readers across
   BOTH witness steps.
3. **Eligibility = agreement.** The tape becomes eligible only if the two witness
   steps produce IDENTICAL harvest sets (structurally — same reader positions), the
   same way `diffImages` requires identical payload structure. A harvest-set
   disagreement between the two witness steps is an UndeclaredVariance-class refusal
   (never eligible), so a nondeterministic reader can never silently ship.
4. **Replay uses the witnessed set.** The skeleton's `planNodes` are dressed so the
   witnessed harvest is materialized — the checkpoint-recompute `contiguous` is
   harvested because the witness steps SAW backward read it, exactly as the recorded
   build harvested it by recording the dispatch.

### 4.2 How it covers the #97 class

The #97 STOP is precise: derivation is infeasible because (a) the reader is LOWERED
(invisible to `observeConsumed`), (b) it is `graphHeld=FALSE` (the stage-2 oracle
misses it), and (c) at forward-plan build time the backward consumer does not exist.
Witnessing defeats all three at once: it does not derive at forward-build time, it
observes at end-of-step time; it observes the LOWERED read directly (it instruments
every read path, not just the compiled bind); and it needs no graph-retention flag
because it uses the OBSERVED read, not an inferred one. The recorded build had
exactly this property (build-WITH-execution); the witnessed step object promotes it
to the tape stratum without keeping the recorded build.

### 4.3 The never-witnessed remainder

Some plans never witness: single-shot warmup variants, transient plans that don't
recur two-consecutive (`stage4 §Task #43` — "Transient plans run lowered once"), and
the uncovered generator classes. These KEEP the recorded build as their harvest
master. The step-object campaign does NOT delete the recorded build wholesale; it
deletes the recorded build's harvest role FOR THE PLAN CLASSES the witnessed harvest
now covers, config-matrix-proven (checkpointing ON mandatory). Full recorded-build
sunset remains gated on 4.4-coverage OR full witness coverage of every recurring
plan — whichever lands first (`stage4 §Task #43` verdict, refined).

### 4.4 The differential gate

- **Shadow set-parity, checkpointing ON.** For every producer plan the recorded
  build covers, `witnessedHarvest(plan) == recordedHarvest(plan)` as a set — run on
  distil@512 + selective checkpointing (the exact config #97 stopped on), medium@512,
  and 124M (the chunked-sum class). Empty diff is the pass.
- **Trajectory parity.** With the witnessed harvest driving replay and the recorded
  build DELETED for the covered classes: `parity-fullstack-tl.ts` compiled-vs-lowered
  ≤ 1e-5 over 30 steps, WITH checkpointing, WITH GradScaler, WITH an LR schedule.
- **The `Input not ready` twin must not fire.** The #97 stage-3 symptom (`Input not
  ready: node contiguous[512,768]`) is the precise negative assertion: zero
  occurrences across the matrix with the recorded build removed for covered classes.

---

## 5. The edit channel — `ReRecordChannel` generalized

Ruling 4: `ReRecordChannel` (`src/schedule/moves/fuse.ts:65`) generalizes into THE
step-object edit channel; the P3 editor binds to one seam. Kept at INTERFACE
altitude — the editor itself is out of scope.

### 5.1 The typed request surface

Today `ReRecordChannel` is two methods (`requestMerge(a,b) → region'`,
`rollback(region)`) — the island membership seam the detector owns, stubbed as
`makeInMemoryChannel()` for P3 to replace. Generalized to the step object, the
channel is the ONE surface through which any facet edit is expressed as a re-record
request:

```
StepEditChannel = {
  requestMerge(a, b): RegionUid           // partition: fuse two islands (existing)
  requestSplit(region, at): RegionUid[]   // partition: split (islands I4 split policy)
  requestRecompute(segment, mode): void   // recompute-vs-retain toggle (ruling 3)
  requestRingDepth(k): void               // ring K edit (phase2b §2, a memory knob)
  rollback(handle): void                   // discard → re-record under prior decl
}
```

**Slot sources are FIXED at declaration in v1 (RULED 2026-07-15, §10 Q1).**
`requestSlotRebind` is NOT in the v1 surface: changing a knob's VALUE is already
hitchless forever (declared slots re-dress every replay); changing its PLUMBING
(source: upload → scalar-table → tensor) is a rare once-per-knob structural act, and
a re-declare + two-step re-witness (§5.3) prices it correctly. Two doors held open at
zero cost: (a) **slots get STABLE NAMES** — a slot survives re-witnessing as "the α
slot", not "slot #7 of tape 43" — so rebind-as-edit can be added later as a pure
mechanism change with no schema break; (b) the escape hatch: if implementation finds
the rebind path LOWER-complexity than re-declare (Vin's proviso), take it — the
ruling fixes the v1 surface, not the mechanism underneath.

**RESERVED (not v1): `pauseAtBoundary(segment): PauseHandle`** — the
breakpoint-and-poke channel (the model-editor charter's live-vs-static Q4). The step
program's plan/segment boundaries are natural cut points; edits at a pause are slot
WRITES taking effect for the remainder of the step; guards refuse the tape for a
breakpointed step (it runs the normal path — debugging steps are rare by nature, so
tape non-coverage costs nothing). Standing RULES (e.g. "clamp feature when it
fires") are NOT pokes — they compile into the program as conditional seams (the
scoreMod/maskMod pattern, #64) and stay tape-covered. The interface slot is reserved
here so the v1 channel shape doesn't have to break to admit it.

Every method RECORDS a requested decision and returns a handle; nothing mutates a
live partition or plan directly (the `fuse.ts` discipline — "we do NOT build a
standalone membership mutator; that would be a second owner"). The executor
re-records under the new declaration; a refusal (illegal merge, non-convex split,
out-of-budget K) returns a typed `FuseRefusalCode`-shaped code, never a throw at the
channel altitude.

### 5.2 Refusal semantics

An edit that produces an illegal declaration is REFUSED at request time by the
facet's own legality predicate — merge convexity via `islandFlow` (`fuse.ts`
`validate-interior`), recompute legality (a segment must be a valid checkpoint
boundary), ring depth within `TORCHLETTE_POOL_BUDGET_MB`. A refused edit leaves the
current declaration untouched (rollback is the identity). This mirrors the guard
discipline: edits are loud-and-refused, never silently-wrong.

### 5.3 How an edit invalidates / re-witnesses

An accepted edit changes the `declaration` structural hash → the step object's
BucketMiss fires (guard 2) → the current skeleton is dropped, the next two executed
steps re-witness a fresh skeleton under the new declaration. This is the SAME
cut-over the object already uses (ruling 1); editing is just an externally-driven
declaration change. The P3 editor binds to `StepEditChannel` and gets warm/cold
observability for free (a re-witness is the "re-tracing…" the UI shows, `phase1.md
§6` mitigation (a)).

---

## 6. The campaign plan (staged, each phase shippable and independently gated)

Stage-4 style: each phase lands as its own commit with gates in the message; no
phase forces a deletion its gate has not earned (the STOP-rather-than-improvise
rule). Sequencing honors: **witness harvest before recorded-build deletion;
checkpoint-first-class before the bypass dies; the declared-lifetime dividend LAST**
(ruling 2's own re-open condition).

### Phase 0 — Optimizer-config slot coverage (MANDATED — the mundane gate on everything)

**Goal:** kill the 556 refusals (`phase2b.md FALSIFICATION` — `adamStep` per-param
payloads incl. bias-corrected `step_size`, the frozen-scalar family's 7th instance).
Every varying `adamStep`/`unscaleGrad` config becomes a DECLARED TAG_UNIFORM payload
slot the recorder observes (`setAdamConfigUniforms`, `adam-kernel.ts:85`, the single
source; `stRecordUniform`, `step-tape.ts:387`). NOTE: `phase2b.md INC-1` LANDED the
recorder side (batch-representative + dead-payload rules, `diffImages`); `inc-2a`
LANDED optimizer-scalars-as-data (t/lr/scale as tensors, the expm1 derivation). Phase
0's residue is confirming FULL coverage under the step-object declaration — the
`t-train-tape-probe` gate (`eligiblePairs>0, refusals=0`) on BOTH fused and foreach,
WITH GradScaler, WITH an LR schedule, WITH checkpointing.
- **Gate:** `t-train-tape-probe` refusals=0 on the full config matrix; suite green
  both flag states; no new env flag (rides `TORCHLETTE_STEP_TAPE`).
- **Subsumes:** the tail of inc-1/inc-2a coverage; this is the prerequisite the
  falsification proved unmet and the later phases all depend on.

### Phase 1 — The StepObject as the union identity (reify, null-clean)

**Goal:** introduce `StepObject` as the DERIVED union of the existing facets, keyed
to the pair `(declaration-hash, ordered plan fps)`. No behavior change — the object
is reified from what `capture` + the tape already compute, exactly as islands I0
reified `Partition`.
- **Gate (null test):** `diffStreams` empty on distil/medium/124M; the tape's
  `bucketKey`, the capture appKey, and the islands boundaryHash all recompute
  byte-identically as projections of the one object; suites green both flag states.
- **Deletes:** the conceptual split between "capture the API" and "tape the
  recorder" (documentation + the mental model; no code yet).

### Phase 2 — Whole-step capture warm (runahead)

**Goal:** the training capture replays warm — the `phase2b.md INC-2B` surfaces
(multi-plan skeletons, async body, output-node mapping, hit-path boundary queueing),
plus the runahead ring (K=2 default).
- **Gates:** G-cover / G-parity / G-ring / G-scope / G-regression / G-decode-
  untouched / G-suite / G-perf / G-soak (the `phase2b.md §8` gate ladder, adopted
  verbatim). Pre-registered acceptance:
  - **runahead wall:** ~30% distil (the G0(b) GPU-bound floor, 236→~165 ms). Pause
    if < ~15% on a memory-headroom config.
  - **parity:** captured-vs-uncaptured ≤ 1e-5 over 30 steps.
  - **ring:** K=1 vs K=3 bit-identical (runahead reorders CPU submission only).
  - **guard-miss counters:** 0 at steady state.
- **Deletes:** nothing (the fallback stays; deletions are later dividends).

### Phase 3 — Checkpointing first-class; the bypass dies

**STATUS: STOPPED at the bypass deletion (2026-07-15). The `RecomputeSegment`-
as-declared-data reification LANDED; the bypass deletion is BLOCKED by a hard
two-gate conflict and remains in place with a WHY comment.** What landed: the
step object's `recomputeRef` is now a REAL recompute fact — the fps of the plans
that carried a checkpoint boundary this step (`StepTape.recomputeFps`, recorded
per-plan from `isCheckpointBoundary` at `stBeginPlan`), no longer a placeholder
alias of the all-fps `partitionRef`. Gate: `test/step-object.spec.ts` (recompute
= only the recompute-bearing plan fps; distinct from partition; null-clean).

**The bypass did NOT die — the STOP, with a deterministic repro.** The Goal was
to kill `setBufferArenaDisabled(true)` + `TORCHLETTE_CHECKPOINT_ARENA` and run the
full arena+compiled stack under checkpointing. Two required gates are CONTRADICTORY
under the current arena mechanism:

- **MEMORY gate (b66ead78 A/B, distil@512 + selective checkpointing, V100):**
  arena-OFF (bypass-alive) peak **4893 MB** / current-settled **1695 MB**;
  arena-ON (full stack) peak **5397 MB** / current **5233 MB** — **+10.3% peak,
  +209% current**. The per-position arena pins one buffer per dispatch position
  across steps, INCLUDING the forward activations checkpointing drops for
  recompute (the arena pins by position, blind to tensor liveness). So arena-ON =
  the original b66ead78 bug returns (the arena holds activations again). The gate
  (peak ≤ arena-free +5%) is met ONLY by running arena-free.
- **PHASE-4 WITNESS gate (`t-witness-harvest-matrix CELL=checkpoint`):** the
  witness-harvest prune/witness mechanism (observed-liveness) only engages on the
  COMPILED path. Arena-free checkpointing runs LOWERED, so the mechanism goes
  inert: **witnessedTemplates 4→0, prunedPairsRemoved 22→0 = FAIL(B)** ("witnessed
  harvest set EMPTY"). Note `inputNotReady=0` throughout — arena-free/lowered is
  SOUND (it materializes everything; nothing is pruned, so nothing to witness) —
  the FAIL is purely "the phase-4 mechanism didn't fire", not a correctness bug.

  **Deterministic repro of the conflict** (V100, `VULKAN_DEVICE_INDEX=10`):
  1. With checkpointed steps forced arena-free (the phase-3 change: `checkpoint()`
     declares recompute → executor derives arena-free), rebuild and run
     `TORCHLETTE_STEP_TAPE=record CELL=checkpoint npx tsx
     tools/t-witness-harvest-matrix.ts` → **FAIL(B)**, witnessedTemplates=0.
  2. Neutralize the arena-free-under-checkpoint change (arena ON) and re-run →
     **PASS**, witnessedTemplates=4 / 396 pairs / 22 pruned. Same tool, same
     config, opposite verdict — the two gates cannot both hold.

  No config gives compiled + low-memory for checkpointed steps: that IS the "arena
  SERVES the recompute segments" endpoint (§3.5 — the arena must FREE
  recompute-scoped activations while STAYING compiled, e.g. per-position liveness
  keyed to the recompute boundary + planner packing that doesn't pin dropped
  forward positions). That is a SEPARATE campaign (arena mechanism surgery), not a
  fold-into-one-gate. Per the STOP-rather-than-force rule, the bypass stays.

**Goal (UNMET — the target, restated):** `RecomputeSegment[]` declared on the step
object; **the arena+tape SERVE the recompute segments**; `setBufferArenaDisabled(true)`
+ `TORCHLETTE_CHECKPOINT_ARENA` deleted (ruling 3 + 5).
- **Gate:** bypass-OFF path baseline-EXACT loss + peak within budget vs bypass-ON,
  at 124M/batch1/seq256 (the b66ead78 A/B) AND distil@512 selective-checkpointing
  (the #97 config); `Input not ready` zero occurrences; test:gates 4/4; AND the
  phase-4 witness gate must still fire (the new constraint this STOP surfaced —
  arena-free defeats it).
- **Deletes (DEFERRED to the arena-serves-recompute campaign):**
  `setBufferArenaDisabled`, `TORCHLETTE_CHECKPOINT_ARENA`.
- **PRECONDITION (was thought met, now refined):** phase 4's witness harvest is
  sound for the checkpoint config ON THE ARENA-ON/COMPILED path — but that path
  fails the MEMORY gate. The true precondition for the bypass death is an arena
  that frees recompute activations WHILE staying compiled (so both the memory gate
  and the phase-4 witness gate hold at once). That arena does not exist yet.

### Phase 4 — Witness-time harvest; recorded-build harvest role retires (for covered classes)

**STATUS: the MECHANISM + GATE LANDED (2026-07-15, commits `bdfa2e02`→`a498d319`); the
recorded-build harvest ROLE stays (deletion is a later product decision, §4.3).** The
witnessed harvest set now keeps the checkpoint-recompute activation the generated prune
dropped — the #97 STOP is unblocked (see §4 STATUS). The config matrix + the
event-inclusive cells are green (below); the recorded build is NOT yet deleted — this
phase EARNS the deletion for the covered classes, but the full sunset is gated on the
never-witnessed remainder (§4.3) closing, including the `shape=[]` scaler-scalar class
the full-deletion probe surfaced.

**Goal:** §4's mechanism. The witnessed harvest set drives replay; the recorded
build's harvest role retires FOR THE PLAN CLASSES witness coverage proves, config-
matrix-proven with **checkpointing ON mandatory**.
- **Gate:** §4.4 — shadow set-parity empty (checkpointing ON), trajectory parity
  ≤ 1e-5, zero `Input not ready`. The config matrix MUST include distil@512+selective
  checkpointing, medium@512, 124M chunked-sum — the third-time-is-not-fooled matrix
  (§7 risk 1). **The matrix is also EVENT-INCLUSIVE (RULED 2026-07-15, §10 Q2):** it
  must interpose the known data-dependent variation sources — a GradScaler
  inf-skip (overflow-inducing) step and an LR-scheduler milestone — and show each
  either changes structure (guard refuses, normal path runs) or flows through a
  declared slot (covered). Completeness is verified against actual variation, not
  assumed from the window size.
- **Gate — MEASURED (`tools/t-witness-harvest-matrix.ts`, `test/witness-harvest.spec.ts`):**

  | cell | witnessed (templates/pairs) | prunedPairsRemoved | cleanMisses | Input-not-ready | event |
  |---|---|---|---|---|---|
  | distil@512 + selective ckpt | 4 / 396 | 22 | 0 | 0 | — |
  | medium@512 + ckpt | 4 / 1512 | 94 | 0 | 0 | — |
  | 124M @256 (chunked-sum) | 4 / 768 | 46 | 0 | 0 | — |
  | scaler-inf (autocast) | 7 / 460 | — | — | 0 | inf-skip fired (scale 1e40→1.5e38), no corruption |
  | lr-milestone | 4 / 397 | — | — | 0 | LR 5e-4→5e-5 through declared slot, no corruption |

  Operational §4.4 empty-diff = `cleanMisses==dirtyMisses==0` (no pruned-then-demanded
  read; the witness set kept every read pair). STRONG oracle (recorded build DELETED via
  the harvest-deletion diff): checkpoint / medium / chunked all run CLEAN — zero
  `Input not ready` where the prune deterministically threw `contiguous[512,768]`.
  `parity-fullstack` compiled==lowered 8.6e-6/30 on the default (non-tape) path (the
  mechanism is inert without `STEP_TAPE` — one branch at the read seam).
- **Deletes (staged):** the recorded build's harvest master for covered classes;
  full recorded-build sunset (`buildCompiledPlan` + ~80 `record*` refs,
  `stage4 §Task #43` B2/B3, ~−800–1200 SLOC) remains gated on FULL witness coverage
  (never-witnessed remainder, §4.3) — a later product decision.

### Phase 5 — guardMiss recovery → should-never-fire assert (after a zero-fire soak)

**STATUS: LANDED (2026-07-15).** `guardMiss`'s clean-recovery is now a loud
should-never-fire assertion. The zero-fire soak (below) confirmed the recovery net
never fires on ANY path — both prune-soundness classes are covered upstream (the
overlay class by `graphHeldAt` at the claim seam, task #97 stage 2; the
checkpoint-recompute + `shape=[]` scaler-scalar classes by the recorded build's
harvest on the default path / the witness-time harvest on the tape path, §4). So the
recovery machinery is DELETED (the `RecoverableGuardMiss` class, the `forceAllMerged`
re-collect-lowered retry loop, and the `forceLowered` threading through
`executeLoweredPlan` → the compiled/build-from-IR gates) and a matched
pruned-producer miss throws with full context (template fp, node, oi, stamp state,
clean/dirty classification). Net **−28 SLOC** (code-only). The `!hit` (unrelated
miss → return false → caller rethrows original "Input not ready") path is unchanged.

**Boundary (which seam got the assert, which kept recovery):** the demotion is
TOTAL — there is exactly ONE recovery seam (`guardMiss`'s clean branch at the
compiled external-slot bind, `compiled-plan.ts` phase 1) and it is now an assertion.
No seam kept recovery, because there is no class where recovery is the DESIGNED net
today: the `shape=[]` GradScaler-scalar never-witnessed class (the phase-4 STOP
finding) is netted by the RECORDED BUILD's harvest (which keeps the value so it is
never pruned-then-demanded — it never reaches the guardMiss seam), NOT by guardMiss
recovery; the soak's `scaler-inf` cell fires zero. Demoting guardMiss does not remove
that net — the recorded build's harvest is a separate, still-present mechanism.
guardMiss was a redundant secondary net.

**Zero-fire SOAK (device 10, build-from-IR default active; the deliverable):** 20
configs across the default (recorded-build) path, the `STEP_TAPE=record` witness/tape
path, the `STEP_TAPE=1` replay path, and stream-generate — **0 guardMiss fires
everywhere**, every cell PASS.

  | config | path | guardMiss fires | verdict |
  |---|---|---|---|
  | ledger-attack (STEPS=24, 48) | default | 0 | PASS, cleanMisses=claimMisses=dirtyMisses=0 |
  | compiled-parity (+ STREAM_GENERATE) | default / stream-gen | 0 | PASS / 129/129 generated |
  | parity-fullstack (compiled vs COMPILED_PLAN=0) | default / lowered | 0 | ≤1e-5/30 (max 7e-6 @ step 29) |
  | stream-generate | stream-gen | 0 | PASS |
  | witness-harvest {checkpoint, medium, chunked124m, scaler-inf, lr-milestone, base} | STEP_TAPE=record | 0 | PASS (396/1512/768/460/397/396 pairs, 0 Input-not-ready) |
  | train-tape-matrix {fused,foreach}×{no-sched,cosine-lr} | STEP_TAPE=record | 0 | PASS (eligible tape, 0 refusals) |
  | step-object-null | STEP_TAPE=record | 0 | null-clean |
  | ring-probe, train-capture | STEP_TAPE=1 | 0 | PASS |

- **Gate:** the zero-fire soak above; the gate-3 unit test rewritten to expect the
  assertion (`test/observed-liveness.spec.ts` — CLEAN and DIRTY both throw a loud
  should-never-fire Error naming template/node/oi; UNMATCHED still returns false);
  `test:gates` 6/6; witness-harvest / derived-liveness-oracle / stale-external-rebind
  green.
- **Deletes:** the guardMiss recovery path (→ assert); `RecoverableGuardMiss`; the
  `forceAllMerged` retry loop; the `forceLowered` option and its threading.

### Phase 6 — Partition as a step facet; edit channel generalized (islands I3/I4 co-sequenced)

**Goal:** `StepPartition` per step; `StepEditChannel` (§5) as the one edit seam; the
islands I3 (epilogue-as-atom-merge) / I4 (chunking as split) deletions land here
(`islands-design.md §7` — the four bespoke mechanisms collapse to one object + three
policies).
- **Gate:** I1 null-test at step altitude (per-plan boundaryHashes reproduce
  byte-identically); the islands cross-path numeric gate; the P3 editor binds
  `StepEditChannel` in `examples/schedule-editor` FIRST (admission pressure — earns
  generality in examples/ before src/).

### Phase 7 — The declared-lifetime dividend (LAST; its own re-open condition)

**Goal:** ruling 2. Inside a captured boundary, the observation predicates
`everSurvived`/`everReadback`/`everAliased` RETIRE on the captured path (`phase2b.md
§5`). Sequenced LAST, gated on §5's own re-open condition: the captured training path
is WARM (phase 2 done) AND the observation-layer watcher cost on the training path is
MEASURED to be worth retiring.
- **Gate:** the derived-lifetime path produces the SAME liveness the predicates
  observed (set-parity vs the observation layer, on the captured path), measured
  watcher-cost delta > the derivation cost; captured-path only (the LOWERED fallback
  keeps all three — a captured-path dividend, not a global deletion).
- **Deletes (captured path only):** `everReadback` (ring outputs declared),
  `everSurvived` (registered persistent + ring survivors declared), `everAliased`
  (statically-visible aliasing in the recorded plan sequence).

### Task subsumption / redirection

- **#78 (islands)** — I0/I1/I2a landed; I3/I4 REDIRECTED into phase 6 (partition as
  a step facet). The islands campaign's remaining deletions are earned at step
  altitude.
- **#80 (schedule-state P-waves)** — orthogonal (intra-kernel stratum); the step
  object is the same GRAMMAR one stratum up, and phase 6's `StepEditChannel`
  composes with schedule-state's move channel. No subsumption; shared spine.
- **#43-remainder (recorded-build sunset)** — SUBSUMED by phase 4: the sunset the
  #43 map STOPPED on ("execution-witnessed harvest — a separate campaign") IS the
  witness-time harvest here. Phase 4 is that campaign.
- **#66 (scoped-memory)** must AGREE on: the boundary is an EPOCH, not a "step"
  (`scoped-memory-design.md §8` — "the tape records the epoch structure, not steps;
  epoch ids are the shared vocabulary"). The step object's `epoch` field is
  scoped-memory's epoch id. Neither blocks the other; both use epoch ids from the
  start (the §8 design requirement). The step-object boundary and scoped-memory's
  scope-epoch are the SAME boundary — the step object declares what scoped-memory
  delimits.

---

## 7. Risks named honestly

**Risk 1 — the recorded-build deletion's history: two blocked attempts, both by
classes no prior gate exercised.** The #43 pass STOPPED (uncovered-plan census
fallback still load-bearing). The #97 pass STOPPED at stage 3 (the checkpoint-
recompute `contiguous[512,768]` — a config the #43 deletion pass never ran). BOTH
were fooled by a config the gate did not cover. **Mandate for phase 4's witness
gate:** the config matrix that must run before the recorded build's harvest role
retires — **checkpointing ON is mandatory** — is: distil@512 + selective
checkpointing (the #97 config), medium@512, 124M (the chunked-sum + contended
class), fused AND foreach optimizer, WITH GradScaler, WITH an LR schedule crossing a
warmup→decay knee, WITH a scaler-backoff event. A witness gate that omits
checkpointing repeats the #97 mistake — it is the third-time-is-not-fooled clause.

**Risk 2 — guard-3 byte-diff cost.** `diffImages` byte-compares the full
payload/scalar image of consecutive steps (`step-tape.ts:534–631`). At 124M this is
non-trivial (the 556-refusal probe showed the machinery runs, but cost scales with
plan count × node count). If the witness comparison (now ALSO diffing harvest sets)
inflates the recording-step cost past the runahead win, the campaign re-prices — the
recording steps are amortized (two per warm tape) but a large model with frequent
re-witnessing (structure churn) pays it repeatedly. Measure the recording-step
overhead explicitly in phase 1's null gate.

**Risk 3 — tape re-record churn under editing.** Every `StepEditChannel` edit is a
declaration change → BucketMiss → two re-witness steps (§5.3). A UI that scrubs a
partition or recompute knob rapidly forces continuous re-witnessing (the cold-
interaction physics, `phase1.md §6` — "the UI leaks the cache topology"). Mitigation
owed to phase 6: edit debouncing at the channel, and the warm/cold observability the
editor shows. A structure-knob is a COLD control by nature; the doc does not pretend
otherwise.

**Risk 4 — runahead ring lifetime.** `[lifetime]` warn under ring K≥2 + an LR
scheduler is a KNOWN OPEN (`phase2b.md §2` memory cost + the ring-pins-in-flight-
steps rule; the K un-fenced backward activations hold K× the peak transiently). On a
memory-tight config (124M near a 32 GB ceiling) K may be forced to 1 = no runahead =
no phase-2 win. The win is real ONLY where memory headroom for K≥2 exists — stated in
the phase-2 perf claim, not hidden. The ring-K + LR-scheduler `[lifetime]` interaction
must be a phase-2 STRICT_LIFETIME gate, not deferred.

---

## 8. One-sentence test (house rule)

**The StepObject:** *a whole training step is one first-class object that exists as a
declared boundary and a witnessed tape — the same record-then-cut-over lifecycle as a
compiled plan, one stratum up — whose partition, recompute segments, and dynamic
slots are data on the object and whose lifetimes the executor DERIVES from the
program instead of OBSERVING from executions.*

Each section restates as a declaration: §2 the object (a two-phase typed datum with a
canonical digest); §3 the unification (five mechanisms are five facets of one
object); §4 witness harvest (the harvest set is the union of physically-observed
cross-plan reads across two identical executed steps); §5 the edit channel (every
facet edit is a re-record request through one seam); §6 the campaign (each deletion
is earned by a differential its gate crosses).

---

## 9. Red-team (the amendment-round tradition) — three strongest objections

Red-teamed once before committing, per house tradition. Each objection is answered by
a ruling or flagged as an open question (§10).

**Objection A — "Witness-time harvest is just the recorded build wearing a tape
costume; you have not deleted the mechanism, you have renamed where it lives."** The
#97 STOP says the recorded build is load-bearing precisely because it is
build-WITH-execution; witnessing is ALSO build-with-execution (it runs two real
steps). So what is DELETED?
- **Answer (ruling 1 + §4.3).** What is deleted is the recorded build as a PARALLEL
  build source that runs on EVERY execution of a covered recurring plan. Witnessing
  runs the full path TWICE (to make the tape eligible), then the WARM tape replays
  with the witnessed harvest and the recorded build never runs again for that plan.
  The recorded build's cost was per-execution-forever for recurring plans; the
  witnessed harvest's is amortized to two witness steps. The mechanism is not renamed
  — its per-step recurrence is deleted. AND the never-witnessed remainder (§4.3)
  honestly KEEPS the recorded build, so the claim is scoped, not total. This is the
  same "captured-path dividend, earned locally" shape as stage-4 phase-4 (`phase2b.md
  §5`), not a global deletion pretending to be one.

**Objection B — "Ruling 4 lifts partition to per-step, but the executor NEEDS a
per-plan linear order to schedule; a per-step partition cannot survive the projection
back to per-plan boundaryHashes without either (i) recomputing the per-plan split
anyway (no deletion) or (ii) changing the byte-stream (breaks the I1 null test)."**
This is the exact trap islands I2b fell into (`islands-design.md` — "you cannot patch
a linear scanner into a graph-aware grouper").
- **Answer (§3.3 + §2.4).** The per-step partition is the DECLARED facet; the
  per-plan boundaryHashes are a DERIVED PROJECTION of it (the schedule-state
  "loop-nest VIEW derived by a canonical ordering rule" pattern, `schedule-state
  §0`). I0/I1 already proved the projection is byte-stable when the partition is
  DERIVED (null-test-clean). Phase 6's gate is exactly that I1 null test run at step
  altitude: the per-step partition must reproduce the per-plan boundaryHashes
  byte-identically, ELSE it is refused. So (i) is answered — the per-plan split is a
  pure function of the per-step partition, not a recomputation from scratch; and (ii)
  is the GATE, not a risk swept aside. If the projection is not byte-stable, phase 6
  does not land (the null-test STOP). The deletion (the four bespoke mechanisms) is
  earned by the projection being a canonical rule, not by hoping.

**Objection C — "The step object claims 'no second whole-step mechanism may be born'
(ruling 1), but §5's `StepEditChannel` is a NEW mechanism — a request surface that
did not exist — and phase 2's runahead ring, the multi-plan skeleton accumulator, and
witness-harvest instrumentation are all net-new code. The complexity budget will grow,
not shrink, for several phases before any deletion lands."**
- **Answer (partially conceded — a real cost, bounded).** True: phases 0–2 add
  mechanism (ring, multi-plan skeletons, witness instrumentation) before phases 3–7
  delete (bypass, guardMiss recovery, recorded-build harvest role, observation
  predicates, four islands mechanisms). The weight-norm grows first. This is honest
  and matches every prior campaign (islands added ~150-250 SLOC of `Partition` before
  deleting ~800-1200). The BOUND: (a) `StepEditChannel` is not a new whole-step
  mechanism — it is the EXISTING `ReRecordChannel` generalized (ruling 4 names it
  explicitly), the one seam, not a second owner; (b) each phase NAMES ITS DELETIONS
  in its commit (house policy) and the net-negative is the campaign-end acceptance
  (ruling 5's death list). If any phase cannot name a deletion its gate earns, it
  holds (the admission-pressure rule — code proves itself before entering src/).
  Ruling 1 is about IDENTITY (one object, not two whole-step mechanisms), not about
  never writing supporting code; the request surface and ring are facets/plumbing of
  the one object, gated to net-negative by campaign end.

---

## 10. Open questions — BOTH RESOLVED (Vin, 2026-07-15)

1. **Slot rebind: FIXED AT DECLARATION in v1** (ruling stamped into §5.1). Rationale:
   value changes are already hitchless; source changes are once-per-knob structural
   acts correctly priced at a two-step re-witness; the only regime where rebind-as-edit
   earns its machinery (slow steps × frequent plumbing changes) matches no sketched
   workflow. Doors held open: stable slot NAMES now (schema-level, zero cost), and
   Vin's proviso — if implementation finds the rebind path lower-complexity than
   re-declare, take it. Also reserved in §5.1: the `pauseAtBoundary` breakpoint hook
   (the strongest mid-step case — closed-loop intervention within one forward — is
   pause-at-declared-boundaries + slot writes, not arbitrary mutation).

2. **K_w = 2 everywhere** (ruling stamped into phase 4's gate). The field's consensus
   (JAX refusal / Dynamo guards / CUDA-graph contract / V8 deopt) is that warmup count
   is a performance parameter, never a correctness parameter: two structurally
   identical steps carry all structural information, and data-dependent variation
   (scaler inf-skip, scheduler milestones) is invisible to ANY small window — it is
   handled by guards + the armed strict-lifetime detector + the phase-4 shadow
   set-parity gate, whose config matrix is now required to be event-inclusive.
