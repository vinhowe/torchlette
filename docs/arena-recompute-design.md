# The arena serves recompute segments

**Status:** DESIGN ONLY (task #99, house stage-1/3 protocol). No mechanism lands
this campaign — this doc plus a staged, differentially-gated plan. It resolves
the **hard two-gate conflict** the step-object campaign STOPPED on
(`step-object-design.md §6 Phase 3`: arena-ON → loss bit-identical but current
memory +209%; arena-FREE → memory fine but the whole compiled/tape/witness stack
runs dormant-LOWERED — no config gives compiled + low-memory for checkpointed
steps).

**Lineage:** the ninth application of the house move "the latent decision becomes
an object" (`schedule-state-design.md §1`; the step object is the eighth). Here
the latent decision is *a checkpointed activation's lifetime is multi-segment* —
dead after forward's last read, re-created by recompute before backward's first
read — today expressed nowhere in the compiled path (the planner treats it as a
whole-step RESULT) and worked around by switching the arena OFF. This campaign
makes that lifetime a declared fact the memory planner serves.

**Normative ground truth (all cited inline):** `stage4-compile-from-ir.md`
(§(d) rematerialization extension point `:1393`, the 2026-07-07 STRATEGY
RESOLUTION `:1033`, stage-1 observed-liveness `:1087`); `step-object-design.md`
(§3.5 the `recompute` facet, §6 Phase 3 STOP); `scoped-memory-design.md` (§1 the
epoch vocabulary the boundary shares); `memory-planner.ts` (the PlannerRegistry,
the packing rule); `CLAUDE.md` (the WebGPU buffer-safety invariants + the
memory-era comparability trap). This doc does not re-derive the STOP — it VERIFIES
its anchors and reattributes it.

---

## 0. Declaration (one sentence)

A checkpointed forward activation is a **multi-segment planner result** — its
liveness interval is split at the declared recompute boundary (`recomputeRef`,
`step-object-design.md §3.5`) into a forward span ending at forward's last read
and a backward span beginning at a recompute sub-stream, so the memory planner
packs the dead-in-between interval instead of pinning it whole-step — which lets
checkpointed training run **compiled AND low-memory at once**, and
`setBufferArenaDisabled(true)` dies by unification, not by a separate arena-free
executor.

If a section below cannot be stated as a declaration in one sentence of the
planner/liveness grammar, it is reshaped before it lands (the house one-sentence
test).

---

## 1. The measured attribution — the +209% lives in the PLANNER, not the arena

The Phase-3 STOP recorded "+10.3% peak, +209% current, measured with the compiled
stack ON" (`step-object-design.md §6 Phase 3`) but did NOT attribute WHERE the
resident bytes live. The title of this campaign — "the arena serves recompute
segments" — presumes the arena. **That presumption is empirically FALSE on the
compiled path.** The fix is planner-side.

**Method.** A steady-state attribution probe (distil@512 + selective
checkpointing, the #97 config) split resident GPU bytes across `arenaBufferSet`
(the position-indexed per-lowered-plan arena, `buffer-arena.ts:118` / `:370`),
`debugPlannerRegistryStats().materializedMB` (the shared `plannerRegistry`,
`compiled-plan.ts:640` / `:702`), and everything else (pool/params/optimizer
state). Two modes: `arena` (compiled/planner path, the Phase-3 PASS-B case) and
`free` (`setBufferArenaDisabled(true)`, the b66ead78 bypass). Run on device 10
(A100 32 GB partition), 14 steps, late-step readings.

| mode | current (steady) | arena-resident | planner-registry (materialized) | other |
|---|---|---|---|---|
| **arena-ON (compiled)** | **4584.7 MB** | **1.8 MB** (225 buf) | **2833.8 MB** (result 1919 / temp 2945-decl, 602 entries) | 1749 MB |
| **arena-free (lowered)** | **1798.3 MB** | 0.0 MB | 0.0 MB (no compiled plan) | 1798 MB |

**The delta is +2786 MB (+155%), and it is ENTIRELY the planner registry.** The
position-indexed arena holds **1.8 MB** at steady state — arena-reclaim
(commit 78c6f73, `executor.ts:1841`, canRecycle-gated `buffer-arena.ts:282`)
already emptied it once each plan reached compiled/planner replay. (The +155%
here vs the STOP's +209% is the V100-vs-A100 + config-settling delta CLAUDE.md
warns about — the RATIO class is identical; the ATTRIBUTION is what this table
establishes, not a re-measurement of the STOP magnitude.)

**Why the registry holds them (the root cause).** A checkpointed forward
activation is produced by the forward plan and read by the backward (recompute-
reader) plan — a **cross-plan read**, so the planner classifies it as a RESULT
slot. RESULT slots are exclusive to their plan for its whole lifetime and never
recycled within the plan (`memory-planner.ts:15-16`, `:259-261`, `:273`;
`PlannerEntry.resultHolder`). Its liveness interval `[allocIdx, lastUse]`
(`memory-planner.ts:174-208`) therefore spans forward-plan-replay →
backward-plan-replay — effectively the whole step. **The planner has no notion
that this activation will be recomputed later, so it pins it whole-step exactly
as if checkpointing were off.** That is the b66ead78 bug reappearing one stratum
up: b66ead78 fixed it in the *arena* (position-indexed, per-lowered-plan) by
turning the arena off; but on the compiled path the pinning moved to the
*registry* and the arena-off switch takes the whole compiled stack down with it
(→ the dormant-stack half of the two-gate conflict).

**Consequence for the mechanism ruling (§3):** the arena (warmup/lowered era) is
NOT the site of the fix on the compiled path — it is already reclaimed. The fix
belongs in the **PLANNER** (compiled era): split the RESULT's liveness at the
recompute boundary so `planMemory`'s interval allocator packs the dead span. This
is exactly `stage4-compile-from-ir.md §(d)` rematerialization (`:1393-1413`), and
exactly the 2026-07-07 STRATEGY RESOLUTION already adopted into the roadmap
(`stage4 :1033` — "the memory planner learns REMATERIALIZATION … the arena-free
checkpoint mode of b66ead78 dies by unification"). This campaign is the design of
that stage-3 remat, scoped to the checkpoint-recompute activation class.

---

## 2. Where the pieces are today (the substrate, verified)

The declared-data substrate and the packing machinery already exist; the campaign
connects them.

**The recompute segment as declared data (the facet already landed, Phase 3):**
- Node flag `isCheckpointBoundary` set by `markAsCheckpointBoundary`
  (`plan-builder.ts:11`), planted on the last recomputed node in the unpack hook
  (`nn/checkpoint.ts:215-223`).
- Per-plan aggregation `hasRecompute` (`step-tape.ts:290`, computed `:493-509`).
- `StepTape.recomputeFps` (`step-tape.ts:126`, built `:955-959`) — the ordered
  fps of recompute-bearing plans.
- `StepObject.recomputeRef` (`step-object.ts:167`, derived `:412-419`). The
  intended run-time driver `RuntimeEngine.declareRecomputeSegments` is referenced
  (`step-object.ts:418`) but NOT implemented — today the fact is DERIVED from
  observed boundaries. This campaign gives it a consumer.

**The planner (the site of the fix):**
- `PlannerRegistry` (`memory-planner.ts:64`), single instance `plannerRegistry`
  (`compiled-plan.ts:640`), reset at engine-instance boundaries
  (`resetPlannerRegistry`, `compiled-plan.ts:653`).
- Interval packing `planMemory(commands, registry, resultSlots?, externalReleases?)`
  (`memory-planner.ts:153-338`): per-slot lifetimes `[allocIdx, lastUse]` over
  command-stream indices; greedy first-fit within size classes; `resultSlots`
  never share, never release; `releaseAt = lastUse + 1` (never same-command — the
  WebGPU read/write aliasing rule). **Already parameterized on the result set**
  (`resultSlots`) and on `externalReleases` — the design constraint stage-4 §(d)
  told us to preserve is intact.
- Replay binds registry buffers at TAG_ALLOC (`compiled-plan.ts:1515-1584`): if a
  `plannerAssignment` exists, `entryIdx = assignment.get(cmd.slot)`, the entry
  buffer is lazily materialized + pinned (`pinnedBufferSet.add`, `:1564`) and
  bound. Replay is a mechanical consumer of `plannerAssignment` — it needs no
  change beyond whatever new recompute alloc/harvest commands the planner emits.

**The lowered-path prior art that generalizes (task #51 lineage):**
- The dead-under-fusion segmented executor `executePlanSegmented`
  (`sequential.ts:149`) + `segmentPlanAtCheckpoints` (`plan-builder.ts:20-38`):
  releases forward activations at segment boundaries via `survivingNodeIds`
  (final output + `externalNodeIds` saved-for-backward + later-segment inputs).
  It is NOT the mechanism to generalize — it is dead code under fusion
  (`engine.ts:889` `executePlanOptimized` is the first branch).
- The mechanism that IS live and generalizes: the **action-indexed liveness
  early-release** in the lowered/optimized executor (`executor.ts:2879-2916`) —
  `livenessReleaseSchedule` frees each intermediate as its last consumer
  completes (`canSafelyRelease` + `releaseBufferImmediate`). b66ead78 tied
  checkpointing to arena-OFF precisely so THIS release fires on forward
  activations. **What generalizes to the compiled path: the same "free at last
  consumer" idea, expressed as a liveness EDGE the planner packs, not an
  imperative release call** — the planner is the compiled-era analogue of the
  liveness schedule.

**The recompute program primitive (already exists, no execution needed):**
`generateStream` over the producing subgraph — build-from-IR already generates a
dispatch stream from IR with no recorded trace (`stage4 §(d):1402-1404` — "which
build-from-IR already does without execution … generating a stream from IR with
no recorded trace IS the remat primitive").

---

## 3. The mechanism — two candidates, one ruling

The choice is WHERE the segment-liveness fix lives. The §1 attribution rules OUT
the arena on the compiled path; the remaining real choice is between two
planner-adjacent designs.

### Candidate A — segment-scoped sub-arenas / position-aliasing in the ARENA
Give the position-indexed arena (`arenaAllocAt`, `buffer-arena.ts:370`) a
per-recompute-segment watermark: positions allocated inside a recompute segment
recycle at the segment boundary. This is the literal reading of the campaign
title ("the arena serves recompute segments").

**REJECTED.** Three reasons, all decisive:
1. **The arena is empty on the compiled path** (§1: 1.8 MB). A segment-scoped
   arena would optimize the WARMUP/LOWERED era, which arena-reclaim
   (`executor.ts:1841`) already frees at cutover — zero steady-state effect on
   the +155% that actually bites.
2. **The arena is position-indexed, blind to tensor liveness by construction**
   (`buffer-arena.ts:5-12` — "stabilizes buffer object identities … 100% hit
   rate"; the position→buffer map is its whole point). Teaching it liveness
   fights its reason to exist and re-introduces the aliasing hazard the deleted
   `TORCHLETTE_ARENA_POOL_ACQUIRE` already hit (`buffer-arena.ts:424-428` — "a
   pool buffer recorded at one compiled-plan slot handed to another → slot
   aliasing → training divergence").
3. **It would be a SECOND memory authority** beside the planner registry, which
   the stage-4 STRATEGY RESOLUTION explicitly forbids (`stage4 :1080-1085` — the
   planner becomes "the only memory authority"; `ARENA_LIVENESS=0` sunsets at
   this unification). Building arena machinery the planner makes unnecessary is
   the anti-pattern the complexity budget names.

### Candidate B — multi-segment RESULT liveness in the PLANNER (RULED)
Split the checkpointed activation's planner liveness interval at the declared
recompute boundary. Concretely (stage-4 §(d).1, `:1398-1401`):
- A checkpointed activation's registry entry gains **segments**: the lifetime
  `[allocIdx, lastUse]` splits into `[allocIdx, forwardLastRead]` and
  `[recomputeAlloc, backwardLastRead]`, with the entry FREE in between —
  `planMemory`'s release-queue model (`memory-planner.ts:255-276`, the
  `releaseAt = lastUse + 1` free-list return) already expresses release-and-
  reclaim; the entry gains a segment list instead of a single interval.
- The recompute program is a `generateStream` over the producing subgraph,
  emitted as a sub-stream BEFORE backward's first read — the build-from-IR remat
  primitive (`stage4 §(d).1`).
- The dead-in-between interval is now packable: it stops being a whole-step
  RESULT and the interval allocator shares its size class with other temps, so
  the +155% registry footprint collapses to the arena-free survivor set
  (§1: ~1798 MB other, the params/state/pool working set the lowered path
  already runs at).

**RULED: Candidate B.** The fix belongs in the **planner (compiled era) ONLY**;
the arena needs no new machinery (it is already reclaimed on this path). This is
not a new mechanism — it is the stage-3 remat extension point stage-4 already
designed FOR (`§(d)`), reusing `harvestGenResults`/`planMemory`'s existing result-
set parameterization and the `(templateFp, nodeIndex, oi)` cross-plan stamp as
the remat edge's value-name (`stage4 :1395` — "the cross-plan name of a VALUE,
independent of any live buffer — exactly what a remat edge must name"). No second
identity scheme, no second memory authority.

### Where the recompute boundary comes from (declared, not sniffed)
The segment split points are the DECLARED `recomputeRef` fps
(`step-object.ts:167`) — the planner reads the recompute-bearing plan fps and,
within them, the `isCheckpointBoundary` node positions (`step-tape.ts:234`), to
find the forward-last-read / backward-first-read seam. The activation identity is
the stamp `(templateFp, nodeIndex, oi)`. This is the consumer the
`declareRecomputeSegments` reference (`step-object.ts:418`) was reserved for: the
planner is driven by declared data, never by a heuristic scan.

### The residual soundness boundary (named, honored — the STOP sub-case)
Recompute reads CURRENT storages. If an in-place mutation commits between the
forward span's end and the recompute (the optimizer's `copy_`/`adamStep` updating
params, in the SAME step — backward+optimizer is one plan,
`stage4 :1445-1450`), the recompute slice re-executes against NEW param values →
silently DIFFERENT activation. **Ruling:** the recompute sub-stream's inputs must
be pinned to the values live at forward-span start — either params are read
BEFORE any optimizer in-place commit (the natural ordering: recompute happens in
backward, before the optimizer plan), or the recompute reads are guarded stamped
(`stage4 :1445-1462` STOP sub-case). The safe default this campaign takes:
recompute is emitted at backward-first-read, which is strictly BEFORE the
optimizer plan in the step's linear order (backward → grads → optimizer), so no
param `copy_` has committed yet. A crafted gate (§4 Phase R3) exercises the
ordering explicitly; if the ordering assumption is ever violated the recompute
falls back to the recorded/harvested value (never silently wrong).

---

## 4. The seam-invariant table — every buffer-safety invariant × mechanism × guard

Within-step segment reuse (a registry entry freed after the forward span, reused
by another temp, then re-materialized for the recompute span) MUST NOT violate
the WebGPU buffer-safety invariants (CLAUDE.md "WebGPU Buffer Pool Invariants").
Each gets a named subsection.

### 4.1 `canRecycle` — ownership + in-flight encoder claim
**Invariant** (`buffer-pool.ts:173` — `canRecycle(buf)`): a buffer is reusable
only if `bufferLiveCount` is 0 (no owner) AND, while a shared encoder is open, it
is not in `sharedEncoderWriteSet` (no queued reader hasn't dispatched yet).
**Interaction:** when the forward span ends and the entry is freed for reuse, the
freed registry buffer may still be bound by an encoded-but-unsubmitted forward
pass. **Guard:** the segment-boundary free is a planner-DERIVED release, not an
immediate reuse — `planMemory`'s `releaseAt = lastUse + 1`
(`memory-planner.ts:220-221`, `:274`) already forbids same-command reuse; the
inter-plan release lands at a plan boundary where the shared encoder has been
flushed (§4.3). The assertion at the seam: any code that recycles a segment-freed
registry buffer MUST consult `bufferPool.canRecycle` (CLAUDE.md: "any cache
outside the pool … must consult it before recycling"). The planner does not
hand out buffers mid-encoder; it emits alloc/free COMMANDS the replay binds at
plan altitude.

### 4.2 Never-immediate destroy — deferredDestroy fence-gating
**Invariant** (CLAUDE.md: "GPU buffer destruction is FENCE-GATED, never
immediate"): template eviction / plan invalidation fire under memory pressure
mid-step while the step encoder holds passes binding the buffers; an immediate
`buf.destroy()` poisons the pending submit. **Interaction:** a re-witness or edit
that invalidates a recompute-bearing plan destroys its registry entries. **Guard:**
already routed — `resetPlannerRegistry` (`compiled-plan.ts:653`) and
`destroyCompiledPlanBuffers` destroy via `bufferPool.deferredDestroy`; the
segment mechanism adds NO new destroy path — a segment-freed entry is RELEASED to
the free list (reusable), never destroyed; only whole-entry teardown (plan death)
destroys, and it already goes through deferredDestroy. Assertion: no segment-free
call touches `buf.destroy()`; it only appends to the registry free list.

### 4.3 No mid-step `pendingRelease` flush — the pool-epoch boundary
**Invariant** (CLAUDE.md: "Do NOT flush `pendingRelease` to pool mid-step" —
deterministic ~2% loss drift; buffers released by earlier plans may still be read
by GPU from a prior command buffer). **Interaction:** the segment free happens
MID-STEP (between the forward plan and the backward plan). If it flushed
`pendingRelease` to the pool, a still-queued forward read could see a reused
buffer. **Guard:** the segment free is registry-INTERNAL (append to the registry's
per-size-class free list, `PlannerRegistry.shareable`, `memory-planner.ts:77`),
NOT a pool `flushPendingToAvailable`. The registry's within-step reuse is safe
because plans of one step run strictly sequentially on the queue
(`memory-planner.ts:11-16`) — a temp dead at a plan boundary is dead on the GPU
timeline too, the exact assumption cross-plan packing already relies on. The
recompute-span re-materialization binds at the backward plan's TAG_ALLOC, after
the forward plan's passes are encoded — same strictly-sequential ordering. The
assertion: segment reuse is gated on the SAME strictly-sequential-plans invariant
the existing cross-plan temp packing asserts (structural audit,
`memory-planner.ts:277-324`), extended to assert the two segments of one entry
never overlap on the command-stream timeline.

### 4.4 The WAW check — `sharedEncoderWriteSet`
**Invariant** (CLAUDE.md: "The `sharedEncoderWriteSet` WAW check must be kept"):
two writes to the same buffer within a shared encoder must be ordered.
**Interaction:** the recompute-span re-materialization WRITES the segment buffer
that a later op reads; if the forward-span reader and the recompute writer land in
the same encoder, a WAW/RAW hazard exists. **Guard:** the recompute sub-stream is
emitted at a plan boundary (backward-first-read is in the backward plan, a
distinct plan from the forward producer), so the forward reads and recompute
write are in different encoder scopes separated by the plan-boundary flush
(`endSharedEncoder`). `trackSharedEncoderWrite` (`buffer-arena.ts:441`, called on
every arena/registry buffer handout) keeps the write-set honest; the structural
audit asserts no same-command write+read on one entry. Assertion at the seam: the
two segments of a multi-segment entry are separated by ≥1 plan boundary (a
`releaseAt`/`allocAt` gap spanning an `endSharedEncoder`), verified by the audit.

### 4.5 Replay-binding stability (the load-bearing new risk, §7)
**Invariant** (compiled replay): `plannerAssignment: Map<slot → entryIdx>` is
bound at TAG_ALLOC (`compiled-plan.ts:1544`); the entry buffer is
lazily-materialized + pinned once and re-used every replay. Position→entry
stability is what makes replay a mechanical bind. **Interaction:** a
multi-segment entry is materialized for the forward span, freed, then
re-materialized for the recompute span — within ONE replay the same entryIdx is
bound to TWO alloc slots (forward alloc, recompute alloc). **Guard:** the
assignment map gains a per-alloc-slot binding as today (each slot still maps to
exactly one entry); the entry's segment list is what makes the SAME entry legal at
two slots whose intervals don't overlap. The pin (`pinnedBufferSet.add`) is
per-entry and survives both segments (the buffer is the same object, re-bound).
Assertion: the structural audit (`memory-planner.ts:277-324`), today "no
overlapping lifetimes on one entry," is EXTENDED to "no overlapping SEGMENTS on
one entry" — same audit, richer interval. This is the one place the mechanism
touches replay binding; §7 red-teams it.

---

## 5. The differential plan — staged, each phase gated

Stage-4 style: each phase is its own commit with gates in the message; no phase
forces a deletion its gate has not earned (STOP-rather-than-improvise).
**Checkpointing-ON is mandatory in every gate** (the twice-fooled rule,
`stage4 §7 risk 1` / `step-object-design.md §7 risk 1`). The failing-first gate
already exists: the Phase-3 A/B (`t-witness-harvest-matrix CELL=checkpoint`,
arena-ON PASS/witnessedTemplates=4 vs arena-free FAIL(B)/witnessedTemplates=0 —
`step-object-design.md §6 Phase 3` deterministic repro) plus the §1 attribution
probe (arena-ON 4584 MB vs arena-free 1798 MB, delta = planner registry).

### Phase R0 — the failing-first differential, formalized
**Goal:** land the §1 attribution probe as an in-repo gate (`tools/`, then the
suite): distil@512 + selective checkpointing, arena-ON, asserts steady-state
`planner-registry materialized > arena-free-current` (the +155% is registry-
resident) — the deterministic measurement the fix must collapse. No mechanism.
- **Gate:** the probe reproduces the table in §1 (registry > 2.5 GB, arena < 5 MB)
  on distil@512; medium@512 and 124M show the same class. This is the negative the
  fix erases; committing it first means the win is measured against a pinned
  baseline, not a memory.

### Phase R1 — declared recompute segments reach the planner (no packing change)
**Goal:** the planner READS `recomputeRef` + `isCheckpointBoundary` node positions
and STAMPS which registry entries are checkpoint-recompute activations
(`(templateFp, nodeIndex, oi)`), with NO liveness change yet — the entries are
still whole-step RESULTs. Null-clean reification (islands-I0 shape).
- **Gate:** `diffStreams` empty on distil/medium/124M (checkpointing ON); the
  stamped set == the set of cross-plan reads whose producer plan is in
  `recomputeRef`; suites green both flag states; memory UNCHANGED (still +155% —
  the fix hasn't landed, only the identification). Names its deletion: none yet.

### Phase R2 — multi-segment liveness; the +155% collapses
**Goal:** `planMemory` splits a stamped checkpoint-recompute entry's interval at
the boundary; the recompute sub-stream (`generateStream` over the producing
subgraph) is emitted before backward-first-read; the dead span packs.
- **Gates (all checkpointing ON):**
  - **Trajectory parity:** `parity-fullstack-tl.ts` compiled-vs-lowered ≤ 1e-5
    over 30 steps, WITH selective checkpointing, WITH GradScaler, WITH an LR
    schedule crossing a knee (the frozen-step_size class, CLAUDE.md).
  - **Memory:** the R0 probe's arena-ON current drops to within +5% of the
    arena-free current (§1: ~1798 MB) on distil@512; medium@512 and 124M within
    +5% of their arena-free baselines. This is the campaign's headline number.
  - **Witness engaged:** `t-witness-harvest-matrix CELL=checkpoint` PASSES with
    the arena ON AND compiled — witnessedTemplates=4, prunedPairsRemoved=22,
    `inputNotReady=0` (the Phase-3 STOP's contradictory gate now holds because the
    stack is compiled, not arena-free-lowered). This is the pair-of-gates the STOP
    proved could not both hold under the old mechanism.
  - **Soundness ordering:** the §3 residual gate — recompute reads pre-optimizer-
    commit param values; a crafted in-place-mutation-between-spans test either
    orders correctly or falls back to harvest (never silently wrong).
  - **The standing wall:** `test:gates` 4/4 (the compiled/generated path is
    exercised here, arena ON); the ledger; `parity-fullstack`; 124M DiLoCo
    regression baselines exact (`9.81/5.92/5.15/4.64`, `diloco-regression-check.ts`);
    profile both models (distil ~50 ms/step, medium ~190 ms/step — no perf
    regression; recompute adds backward compute, budgeted).
- **Deletes:** nothing yet (the bypass stays until R3 proves the memory + witness
  gates hold in the PRODUCTION trainer path).

### Phase R3 — the bypass dies (ruling 3 + 5 of the step-object campaign)
**Goal:** with R2's compiled+low-memory checkpointing proven, delete
`setBufferArenaDisabled(true)` + `TORCHLETTE_CHECKPOINT_ARENA`
(`webgpu-gpt2-trainer.ts:179-180`) and the b66ead78 workaround narrative;
checkpointed production training runs the compiled stack.
- **Gate:** the b66ead78 A/B (124M/batch1/seq256, the production trainer) —
  bypass-OFF path baseline-EXACT loss (`10.0→8.6` over 4 rounds, flat memory) AND
  peak/current within budget vs the old bypass-ON path; `Input not ready` zero
  occurrences; the R2 gate ladder re-run through `WebGPUGPT2Trainer` (not just the
  synthetic tool); test:gates 4/4.
- **Deletes (NAMED — the acceptance criteria):**
  - `RuntimeEngine.setBufferArenaDisabled` + `bufferArenaDisabled` field
    (`engine.ts:558`, `:403`) + the `arenaDisabled` threading
    (`engine.ts:894/996`, `executor.ts:206-212/3627-3640`) + its WHY comment.
  - `TORCHLETTE_CHECKPOINT_ARENA` (the env flag, born b66ead78, sunset here) +
    the trainer branch (`webgpu-gpt2-trainer.ts:179-180`).
  - The step-object-design.md §6 Phase 3 STOP narrative (the two-gate conflict is
    resolved; the completion criterion is met).
  - The dead-under-fusion segmented executor becomes a candidate for a follow-on
    deletion (`executePlanSegmented` + `segmentPlanAtCheckpoints` +
    `enableCheckpointSegmentation`/`trueSegmentation` flags, `engine.ts:898-916`) —
    NAMED here, deleted only once nothing routes through it (a separate small
    campaign; the arena-free path was its only near-live cousin and dies here).
  - Sunsets `TORCHLETTE_ARENA_LIVENESS=0` per the stage-4 roadmap
    (`stage4 :1080-1085`) — the planner is now the only memory authority for
    checkpointed training; deferred to the arena-liveness sunset (the legacy
    unbudgeted arena's last role subsumed).

### Sequencing / subsumption
- **`stage4 §(d)` stage-3 remat** — this campaign IS that stage-3, scoped to the
  checkpoint-recompute class. It obeys §(d)'s design constraint (keep
  `harvestGenResults`/`planMemory` parameterized on the result set — already true;
  route all cross-plan value identity through the `(templateFp,nodeIndex,oi)`
  stamp — no second scheme).
- **`step-object-design.md §6 Phase 3`** — R3 completes it. The `recomputeRef`
  facet R1 consumes is the one Phase 3 landed as declared data.
- **`scoped-memory-design.md §1`** — the recompute segment IS a scope/epoch; the
  planner registry is already epoch-scoped (`memory-planner.ts:58-62`, `bumpEpoch`).
  A recompute segment is a nested interval within the step epoch — segments MAY be
  scopes (§6 of scoped-memory's mechanism). Neither blocks the other; both use
  epoch ids (`core/epoch.ts`). Stated for agreement, not folded in.

---

## 6. Deletions named as acceptance (the campaign-end death list)
Per the complexity-budget house rule, the campaign is net-negative by R3:
1. `setBufferArenaDisabled` + `bufferArenaDisabled` + `arenaDisabled` threading +
   WHY comment (`engine.ts:558/403/894/996`, `executor.ts:206/3627`).
2. `TORCHLETTE_CHECKPOINT_ARENA` env flag + trainer branch
   (`webgpu-gpt2-trainer.ts:179`).
3. The b66ead78 workaround narrative + the step-object §6 Phase 3 STOP.
4. (follow-on, named) the dead segmented executor `executePlanSegmented` /
   `segmentPlanAtCheckpoints` / `enableCheckpointSegmentation` / `trueSegmentation`.
5. (roadmap sunset) `TORCHLETTE_ARENA_LIVENESS=0` — the planner as sole memory
   authority.
Net-new mechanism (R1 stamping + R2 segment-liveness + recompute `generateStream`
reuse): stage-4 §(d) estimated the remat extension at reuse of existing
parameterized paths (`harvestGenResults`/`planMemory` already take the result set;
`generateStream` already emits from IR without execution) — ~100–200 net SLOC
against the ~50-SLOC bypass + flag deletion, plus the strategic dividend
(checkpointed training joins the compiled path; `ARENA_LIVENESS=0` sunsets). The
weight-norm grows in R1/R2 and the bypass+flag deletion lands in R3; the
`ARENA_LIVENESS` sunset is the larger negative, deferred but named.

---

## 7. Risks named honestly

**Risk 1 — replay-binding stability under within-step reuse (§4.5, load-bearing).**
Compiled replay assumes a slot→entry binding that is stable across replays; a
multi-segment entry binds the SAME entry at two alloc slots within one replay.
The hazard: if the segment split point drifts between the recording/first-observed
step and a later replay (e.g. a fused vs unfused recompute changes the node
positions), the assignment binds the wrong slot to the shared entry → a
correct-looking-but-wrong result (the worst failure mode, CLAUDE.md "single source
of truth at seams"). **Mitigation/ruling:** the split point is DERIVED from the
declared `recomputeRef` + `isCheckpointBoundary` positions, which are structural
(they hash into the template fp via the plan structure) — a change in split point
IS a template change → a new fingerprint → a rebuild, never a silent rebind of a
stale assignment. The structural audit (§4.5) asserts no overlapping segments on
one entry at BUILD time; a violation is loud, never silent. This is the exact
"assert agreement at the seam" the house demands; R2 does not land if the audit
can't prove segment-disjointness from the declared boundaries.

**Risk 2 — recompute re-fingerprinting instability.** Selective checkpointing
re-fingerprints the recompute READER plans on every backward (fresh per-layer
plans, `step-object-design.md §3.5` / §4 STATUS) — so no two WHOLE steps are
structurally identical, and a whole-step-eligibility gate never fires on the
checkpoint config. **This campaign does NOT depend on whole-step eligibility** —
it drives the planner from the PRODUCER template's recompute boundary, which
recurs identically every step (the checkpointed-MLP `contiguous[512,768]` producer
`0xc98a72f3`, `step-tape.ts:392-410`), exactly as the Phase-4 witness harvest does
PER PRODUCER (`step-object-design.md §4 STATUS` — "the witness window is PER
PRODUCER, not per whole-step"). The recompute sub-stream is generated from the
producer subgraph's IR, which is stable. Stated so R2 doesn't wait on a whole-step
gate that structurally can't fire.

**Risk 3 — the soundness boundary (§3): recompute reads current storages.**
Already ruled: recompute is emitted at backward-first-read, before the optimizer's
in-place param `copy_` (step linear order: backward → grads → optimizer). A
crafted gate (R2) exercises it; violation falls back to harvest, never silently
wrong. The residual is `stage4 :1445-1462`'s STOP sub-case — honored, not swept.

### Red-team (three strongest objections + rulings)

**Objection A — "You renamed the bug, you didn't fix it: b66ead78 pinned in the
arena, you now pin in the registry; splitting the interval just moves the same
bytes into a smaller pool. Where's the deletion?"**
- **Answer.** The §1 measurement disproves the premise that the bytes are
  irreducible: the arena-free path runs the identical program at 1798 MB current
  (the params/state/pool survivor set) because it FREES each forward activation at
  its last consumer (`executor.ts:2879` liveness early-release). The +155% is NOT
  the activations being alive — it's the planner PINNING them whole-step when they
  are provably dead between forward-last-read and recompute. Splitting the interval
  makes the planner express the liveness the arena-free path already achieves
  imperatively — the SAME live set, compiled. The deletion is real: the bypass +
  its flag die (R3) and checkpointed training stops needing a parallel arena-free
  executor. Not a rename — the whole-step pin is deleted, and the compiled path
  gains what only the lowered path had.

**Objection B — "Emitting a recompute `generateStream` before backward duplicates
compute; you've traded memory for time, and the profile gate will fail — this is
just checkpointing's compute cost showing up in the compiled path."**
- **Answer (partially conceded — it IS checkpointing's compute cost, by design).**
  Checkpointing already trades compute for memory (that's what it IS); b66ead78's
  arena-free path ALSO recomputes (the unpack hook, `nn/checkpoint.ts:168`). The
  recompute sub-stream is not NEW compute — it is the SAME recompute the lowered
  path runs, now expressed as a planner-emitted stream instead of an
  imperatively-triggered unpack. The profile gate (R2) asserts no regression vs
  the arena-free path's speed (which already pays the recompute), not vs a
  no-checkpointing baseline. If the compiled recompute is SLOWER than the lowered
  recompute, that's a real regression the gate catches — but the expected result
  is parity, because it's the same dispatch stream over the same subgraph.

**Objection C — "Candidate B needs `planMemory` to grow segment-list intervals,
the audit to grow segment-disjointness, and replay to bind one entry at two slots
— that's planner surgery, and the §(d) claim 'no planner surgery, slots in with no
change' is now false."**
- **Answer (scoped concession).** §(d) claimed the STAGE-1 over-harvest fix needed
  no planner surgery (binary result/temp suffices, `stage4 :1344-1348`); it
  explicitly said multi-segment intervals "are NOT needed in stage 1 (they are
  stage 3's structure)" (`:1347`). This campaign IS stage 3, so the segment-list
  extension is the DESIGNED-FOR surgery, not a surprise — §(d).1 names exactly "the
  entry gains segments." The surgery is bounded: `planMemory` already models
  release-and-reclaim (the free-list return); the entry's single `[allocIdx,
  lastUse]` becomes a small ordered segment list, the greedy allocator iterates
  segments instead of one interval, the audit checks pairwise-disjoint segments.
  Replay binds per-slot as today (each slot → one entry); the entry being legal at
  two disjoint-interval slots is the whole point. It is surgery, it is the surgery
  stage-4 reserved the extension point for, and it is gated (R2 doesn't land if the
  audit can't prove disjointness). Conceded that it is more than "zero change";
  ruled that it is the planned change, not scope creep.

---

## 8. Open questions for Vin (only where they materially fork)

**Q1 — recompute granularity: per-activation segment, or per-checkpoint-region
sub-arena?** Candidate B splits liveness per checkpointed ACTIVATION (each
cross-plan checkpoint result gets its own segment list). An alternative packs a
whole recompute REGION into one reclaimable watermark (all activations of one
checkpointed layer freed together at the region boundary). Per-activation is
finer-grained (better packing) but more segment entries; per-region is coarser but
simpler to audit. The §1 data (602 registry entries, 2833 MB) suggests
per-activation packing is where the win is, but if the audit complexity (Risk 1 /
Objection C) proves load-bearing, per-region is the safer first landing. **Forks
R2's `planMemory` extension shape.** Default proposal: per-activation, fall back to
per-region if the disjointness audit can't be made cheap.

*(Q2 — whether `ARENA_LIVENESS=0`/the legacy unbudgeted arena should sunset WITH
R3 or as a separate follow-on — is NOT material; it's a sequencing preference the
stage-4 roadmap already defers, and R3 lands with or without it.)*

---

## 9. One-sentence test (house rule)

*A checkpointed forward activation is a multi-segment planner result whose
liveness the memory planner splits at the declared recompute boundary — packing
the dead span the way the arena-free lowered path frees it imperatively — so
checkpointed training runs compiled and low-memory at once and the
arena-disable bypass dies by unification.*

Each section restates as a declaration: §1 the attribution (the +155% is the
planner pinning provably-dead activations whole-step, not the arena); §3 the
mechanism (split the RESULT interval at the declared boundary; the arena is
already reclaimed); §4 the seams (segment reuse is registry-internal and
strictly-sequential-plan-safe, asserted by the extended structural audit); §5 the
campaign (each deletion earned by a differential its gate crosses, checkpointing-ON
mandatory).
