# Buffer Donation: aliasing a dead input's buffer as an op's output

> **Status:** P1 LANDED (2026-07-21, §2.4) — packed-path leak fixed, flat premium
> measured (distil +2.1 % / medium +45.6 % peak on A100), donation downgraded from
> "unbounded-leak P4 blocker" to "flat-residual closer required for chain-packing P5."
> **P2 IMPLEMENTED-then-STOPPED (2026-07-21, §2.5) — the ratified generated-stream
> slot-collapse edge lands opt-in (`TORCHLETTE_PLANNER_DONATION=1`, bit-exact ON≡OFF)
> but recovers 0 % of the packed premium: the packed-optimizer plans are UNCOVERED
> (run on the recorded build, not the generated stream) and the buffers are oversized
> → chunked. Premium recovery is BLOCKED behind a stage-4 coverage prerequisite — named
> class in §2.5. P3 is not reachable until that prerequisite is met.**
> Original P0 deliverable of the BUFFER DONATION
> campaign. It cashes the named blocker in `docs/chain-packing-design.md` P4 and
> `docs/architecture-debt.md` stage-3 row: *"Closing [the foreach working-set
> premium] needs graph-level buffer donation (write pNew into P's buffer, G packed
> in place), which is stage-4 work proper."* Companion to
> `docs/stage4-compile-from-ir.md` (the memory planner this must extend).

---

## 1. The claim (one sentence)

**Buffer donation is a PLANNER assignment decision — an op DECLARES which operand
positions it may consume-before-write, and the memory planner binds that op's
output slot to a declared operand's buffer whenever liveness PROVES the operand is
dead after this op (no live owner, no later reader, not a plan result) — so the
packed optimizer's `pNew`/`G` intermediates reuse `P`'s buffer with zero net
allocation, subsuming today's executor-side fused-donation special form as one
auditable planner fact.**

---

## 2. The measured gap — READ THIS FIRST (the campaign's premise is only half true)

Measured fresh on **sivri, 1× V100-32GB** (physical GPU 7), `tools/profile-training.ts`,
distilgpt2 @ seq 512, 15 steps, all default flags except the optimizer path. **These
are V100-relative deltas measured in one session — authoritative absolute numbers
require the dw-2-1 A100 re-measure; the DELTA and its SHAPE are what matter here.**

| Arm | opt path | peak (late) | current (late) | peak trend | storages trend |
|---|---|---|---|---|---|
| **Fused** (default) | `_stepFused` (WGSL adamStep) | **5636.6 MB flat** | **2110.1 MB flat** | flat from step 1 | flat (~248) |
| **Packed** (`TORCHLETTE_FUSED_ADAM=0`) | `_stepForeach` → `packOptimizerProgram` | **10233.5 MB @ step 14, STILL RISING** | **7361.5 MB @ step 14, STILL RISING** | **+320 MB / step** | **+78 / step** |

### 2.1 The gap is a LEAK, not a flat "2× working-set premium"

The chain-packing P4 gate, `adam.ts:295-302`, and `architecture-debt.md` stage-3 row
all describe the packed-vs-fused gap as a **flat ~2× premium** — "foreach 10.6GB peak
vs fused 5.0GB" (A100, 2026-06-10) — to be closed by donating `pNew`→`P`. **That is
not what the packed path does today.** It does not reach a flat steady state at all:
peak and current both climb **+320 MB per step** and storage count climbs **+78 per
step**, monotonically, with no plateau across 15 steps and no GC sawtooth (pool reuse
holds ~59%, `pendingDestroy` stays 0). This is a deterministic per-step retention leak,
not an allocator working-set premium. **Donation cannot close a leak** — aliasing an
output into a dead input reduces the *flat* footprint; it does nothing for storages
that are never reclaimed.

Identical growth (+78 storages/step) reproduces under `TORCHLETTE_COMPILED_PLAN=0`
(pure lowered), so it is **a lowered-path leak in the packed program itself, not a
compiled-replay artifact**.

### 2.2 Where the extra bytes live, and root cause

+78 storages/step ≈ **one leaked storage per parameter per step** (distilgpt2 has ~76
optimizable params). The leak is in the packed **unpack**
(`src/optim/pack-optimizer.ts:250-257`):

```ts
const seg = rt.reshape(rt.narrow(pNew, 0, offsets[k]!, sizes[k]!), items[k]!.param.shape);
rt.copy_(items[k]!.param, seg);
```

`narrow(pNew, 0, offset>0, size)` is an **offset (strided) view**, so the following
`reshape` **materializes a fresh contiguous buffer** per param (reshape-of-non-contiguous
is the one view op that copies — confirmed in `ops/view-meta.ts` / stage-4 phase-3
notes). `packOptimizerProgram` disposes `G`, `gAdj`, `P`, and the eval `sink`
(`pack-optimizer.ts:263`) but **never disposes the `seg` materializations or `pNew`**.
They stay in the live-pending registry (`pendingTensorsByNodeId`), so `markStep`'s
step-scoped demotion never reclaims them (the profiler DOES call `markStep`,
`tools/profile-training.ts:451`). Result: ~1 full-param-width buffer leaked per param
per step. On distil@512 that is ~320 MB/step; on any real run it OOMs.

The historical "10.6 GB flat 2× premium" is suspicious in this light: my step-14 packed
peak (10.2 GB) nearly equals it. The 2026-06-10 figure was very likely **the same leak
sampled at a late step and mislabeled flat** — a single-step memory read cannot
distinguish a 2× flat premium from an accumulation that has reached 2× by that step.
This is exactly the trap `agent-ops.md` warns about (read LATE steps *and confirm the
trend is flat*, not a single late sample).

### 2.3 What this does to the campaign

The gate the chain-packing doc wrote — "donation closes the ~2× premium, then flip P4"
— **targets the wrong quantity**. The design below therefore SPLITS P4's memory
precondition into two gated phases and inserts the truth between them:

1. **First, eliminate the packed-path unpack leak** (a disposal fix in
   `pack-optimizer.ts`, NOT a planner/donation matter). Only then does the packed path
   have a *flat* footprint to compare.
2. **Then re-measure the TRUE flat packed-vs-fused premium on A100.** That number — not
   the historical, leak-contaminated "2×" — sets the `X%` for P4's memory gate. It may
   be well under 2×; donation's job is to close whatever flat residual remains (the
   genuine `G`/`P`/`pNew` co-liveness).
3. **Then donation** (planner-level output↔dying-input aliasing) closes the flat
   residual and satisfies chain-packing P4's precondition.

Donation remains the right *mechanism* for the flat premium and for the broader
stage-4 goal (`stage4-compile-from-ir.md`: "donation = assignment decision"). It is
just **not the first thing standing between the packed path and flat memory**, and the
gate must say so.

### 2.4 P1 LANDED (2026-07-21) — leak fixed, TRUE flat premium measured, verdict

**Root cause CONFIRMED (not `pNew`).** The `+~1 storage/param/step` retention leak is
the unpack `seg = reshape(narrow(pNew, offset, size), shape)` in
`pack-optimizer.ts` — one materialized per-param contiguous buffer per param per step,
never disposed. `pNew` itself was ALREADY disposed (it is the top-level result of
`evalOptTensor(paramUpdate, …, sink)`, and `sink` tracks every created intermediate
*including the result*, so `pNew ∈ sink ⊂ toDispose`). The design's §2.2 "never disposes
`seg` OR `pNew`" over-charged `pNew`; the `seg` half is the whole leak. Reproduced with a
new probe `tools/packed-optim-flatness.ts` (least-squares storage-slope over a flat
window, across the compiled-plan threshold): **before**, packed Adam on a toy GPT-2
(28 params) climbed +29 storages/step (541→1034 over 18 steps), monotonic, reachable —
i.e. retained refs in the live-pending registry, not GC-eligible; identical under
`TORCHLETTE_COMPILED_PLAN=0` (lowered-path leak, confirming §2.1).

**The fix (chosen: dispose the unpack materializations).** `pack-optimizer.ts` now
collects each unpack `narrow` view + `seg` reshape into `unpackTemps` and disposes them
in the same `toDispose` set as `G`/`P`/`gAdj`/`sink` — dropping the wrappers from the
live-pending registry after the `copy_`-back has sequenced the graph edge into `param`
(the standing dangling-`copy_` discipline: the IR node survives, disposal only releases
liveness). Bit-exact by construction (the graph is unchanged; only wrapper lifetimes
change). **Option (b) — "make the unpack a strided `copy_` that never materializes
`seg`" — was evaluated and REJECTED for P1:** `stridedScatterCopy`'s payload carries only
a DST view offset (`engine.ts` `_scatterInPlace`), so a narrow-at-offset *source* would
be resolved by materializing it anyway (no copy actually deleted), and reading a strided
source through `copy_` is exactly the GPU-vs-CPU-semantics hazard class the house rules
warn against. Eliminating that move needs a source-offset region-read — which IS the
donation mechanism (P2/P3), not a P1 disposal. So P1 is a disposal (net +8 SLOC, no
deletion — the campaign's net-negative is cashed at chain-packing P5); option (b)'s
copy-elision is correctly donation's job.

**FLATNESS gate (new standing check): `tools/packed-optim-flatness.ts`.** After the fix,
packed Adam/Lion/SGD all PLATEAU by ~step 5–12 at a constant storage count (slope
0.000/step); Adam toy: 541-and-climbing → **flat 164**; lowered path flat 92; Lion 132,
SGD 182. `parity-packed-vs-unpacked.ts` all 4 arms still **bit-exact** (maxDiff 0.0);
strict `[lifetime]` guard green (zero throws) on the packed arm.

**TRUE flat packed-vs-fused premium — A100 (dw-2-1), distilgpt2 & gpt2-medium @512,
steady state (both arms `Storages/step: +0.0`, i.e. FLAT — the leak is gone in the
measurement itself):**

| model | fused peak / cur | packed peak / cur (fixed) | **flat peak premium** | flat cur premium |
|---|---|---|---|---|
| distilgpt2@512 | 5636.6 / 2110.1 MB | 5753.5 / 2561.5 MB | **+2.1 % (+117 MB)** | +21.4 % (+451 MB) |
| gpt2-medium@512 | 16189.0 / 7246.5 MB | 23568.3 / 9618.9 MB | **+45.6 % (+7379 MB)** | +32.7 % (+2372 MB) |

The historical "10.6 GB flat 2× premium" is now **disproven as a per-step accumulation**
(§2.2 hypothesis confirmed): the genuine *flat* premium is only +2 % peak at distil and
+46 % peak at medium — model-size-scaling, because the packed path co-materializes
`G`/`P`/`pNew` at full concatenated-model width (~3 full-param buffers), which is a small
fraction of distil's activation-dominated peak but a large one at medium.

**Verdict on donation (the design's own re-check clause, §8 taste-call 1).** P1 converts
the packed premium from an **unbounded per-step leak** (would OOM any real run) into a
**flat, model-size-scaling working-set premium**. Consequences:
- **Donation is NO LONGER a P4 blocker in the "packed OOMs / grows without bound" sense**
  — that failure mode is fixed by P1 alone. At distil scale the flat packed peak is
  within +2 % of fused; donation there is a pure optimization.
- **Donation REMAINS the required mechanism for the CAMPAIGN endgame (P5).** At
  medium@512 the flat packed peak is +46 % of fused (+7.4 GB) — the real `G`/`P`/`pNew`
  co-liveness residual donation is designed to close. chain-packing P5 (delete the fused
  `adamStep` monolith) cannot proceed while packed costs +46 % peak at real scale, so
  donation stays P4's memory precondition — now with `X%` honestly known (≈2 % distil →
  ≈46 % medium peak), derived from a FLAT baseline rather than the leak-contaminated 2×.

**Recommendation to the user (taste-call 1):** keep P1 filed as its own landed disposal
fix (done) and start the donation campaign from this flat baseline; the P4 memory
tolerance `X%` should be set against the medium@512 +46 % figure, not distil's +2 %.

### 2.5 P2 IMPLEMENTED-then-STOPPED (2026-07-21) — the edge lands, the premium is unreachable

The ratified P2 mechanism (§3.2–§3.4: the `DONATABLE_OPERANDS` declaration + the
plan-build DONATION EDGE as a slot-collapse in the generated stream, un-refused for
arena/planner-registry temps) was implemented, opt-in behind `TORCHLETTE_PLANNER_DONATION=1`:

- `DONATABLE_OPERANDS` (sibling of `IN_PLACE_DST_INPUTS`, `plan-builder.ts`).
- Executor threads the RAW liveness proof (`cachedDonatableIds` — producers whose last
  reader is this action, not a cross-plan/terminal output) onto the fused action.
- `generateFused` (`stream-generate.ts`) makes the donation decision from that proof +
  `DONATABLE_OPERANDS` + shape/dtype + **arena-slot-kind** gating (the un-refused analog
  of the executor detector's `cachedDonatedRecipeIdx`, which skips arena buffers), and
  collapses the primary output onto the dead donatable operand's slot via the existing
  `planFusedKernel` `donatedInput` binding form.

**It is SAFE.** `parity-packed-vs-unpacked.ts` is bit-exact (maxDiff 0.0, all 4 arms)
ON and OFF across the compiled-plan activation threshold; the strict `[lifetime]` guard
throws zero.

**It recovers 0 % of the packed premium** (distilgpt2@512 packed peak **5753.5 MB with
the flag ON == OFF**, V100 sivri). Instrumentation shows donation NEVER FIRES on any
packed-optimizer workload — three INDEPENDENT structural blockers, each sufficient alone:

1. **UNCOVERED plans → the generated stream is never the live plan.** On distil (and
   larger), the forward/backward AND optimizer plans are `fullyCovered=false`
   (e.g. 187/311, 228/231 actions covered; uncovered set = `data-source:*`,
   `fused[no-input-pattern]`, `op:triu`). The executor uses the GENERATED build only when
   `gen.fullyCovered` (`executor.ts` build block); an uncovered plan runs on the RECORDED
   compiled build. **The P2 donation edge lives in `generateStream`, so it is discarded
   for every real training plan.** (The design's §3.1/§3.4 premise — "under compiled
   replay the packed P/G/pNew buffers ARE planner-registry buffers" — holds only for
   fully-covered plans, which the packed path is not.)
2. **OVERSIZED buffers → chunked, no in-kernel donation.** `packOptimizerProgram` cats the
   whole class into one `[Σ size]` buffer (~328 MB at distil, ~1.4 GB at medium) — far
   over `maxStorageBufferBindingSize` (128 MB). Oversized fused groups route to
   `generateFusedDecomposed`/chunked elementwise (and, lowered, to
   `executeSequentialSegment`), and the chunked elementwise dispatch has NO donation form:
   in-place would require the chunked tile-IR kernel to bind the donated input as the
   read-write `out` (the `planFusedKernel` `donatedInput` trick), which it does not.
3. **Recording↔generation ref-kind mismatch.** Even for a non-oversized covered group
   (`div+div+sqrt+add+div+mul+add+sub`, the packed Adam update, reached with
   `donatableIds=4`), the candidates bail `not-in-liveness`/`not-pending`: at generation
   time (post-execution) the donatable producers resolve as `materialized` refs, whose
   identity does not match the recording-time `pending` node ids in `cachedDonatableIds`.

**NAMED CLASS (the seam that resists):** *the ratified P2 substrate — the generated
compiled stream — is not the substrate the packed optimizer runs on. The packed premium
lives in (a) UNCOVERED plans (recorded/lowered build) and (b) OVERSIZED-CHUNKED buffers.
A generated-stream slot-collapse edge, however correct, is inert against both.* Recovering
the premium requires, as a PREREQUISITE campaign (not P2 slot-collapse work):
**(A) stage-4 COVERAGE of the packed optimizer plan** (`cat`/`narrow`/`copy_`/data-source
+ oversized-chunked elementwise → `fullyCovered`), so the generated stream — and its
donation edge — becomes the live plan; **AND (B) chunked-elementwise IN-KERNEL donation**
(bind the donated input as the chunked kernel's rw `out`), since the premium buffers are
all oversized. Both are beyond the ratified "slot-collapse at plan-build" scope, and (B)
is the highest-risk aliasing change in the codebase — the §7.2 corruption class. Per the
campaign covenant ("STOP-with-named-class beats a forced landing in THIS mechanism"), P2
STOPS here: the edge is landed opt-in (inert, safe, ready for when the substrate exists);
the premium recovery is deferred to the coverage prerequisite.

**Recommendation to the user:** treat "cover the packed optimizer plan" as the real P2.5
(supersedes the `cat`-only P2.5 in §8 taste-call 2 — `cat` coverage is a subset of it),
BEFORE any P3 flip. The `X%` P3 gate cannot be met while the packed plan is uncovered.
The landed opt-in edge + `DONATABLE_OPERANDS` declaration are the foundation to reuse
once (A) lands; nothing about them needs redesign.

---

## 3. Altitude ruling — a PLANNER fact, gated on a DECLARED op property (both, with a division of labor)

Donation is decided in the **memory planner**, but it needs one bit the planner cannot
derive from liveness alone. The ruling: **the op DECLARES eligibility (a static
property); the planner PROVES the precondition and makes the binding; the bind is
ASSERTED by the existing overlap audit.** Neither half alone is sufficient — this is
the single-source-at-the-seam rule applied to donation.

### 3.1 Why not executor-only (today's form), and why not liveness-only

Today donation is an **executor-side, per-fused-segment decision**
(`segment-executors.ts:252-316`, gated by `TORCHLETTE_DONATION`): during the lowered
execution it picks a dying input whose buffer out0 overwrites, then caches the choice
on the action (`executor.ts:2698-2714 cachedDonatedRecipeIdx`) so `stream-generate.ts`
reproduces it. Two structural limits make this the wrong altitude for the packed
optimizer:

- **It refuses exactly the buffers the packed path needs.**
  `segment-executors.ts:293` skips any candidate in `arenaBufferSet` or
  `pinnedBufferSet` — "their identity is managed elsewhere." But under compiled replay
  the packed `P`/`G`/`pNew` buffers ARE planner-registry buffers
  (`compiled-plan.ts:1084-1102`). So the one form of donation that could close the
  premium is structurally unreachable to the executor-side detector. The planner is
  precisely the owner that CAN safely donate its own registry entries.
- **It only covers single-output fused elementwise segments.** `cat` (which allocates
  and copies, `gather-scatter.ts:534-568`) and `copy_` (strided scatter into the dst
  buffer) are not fused-segment outputs; the packed unpack's `copy_`-back and the
  pack's `cat` are never donation candidates today.

Liveness-only (planner infers "output N may reuse input X because X dies at N") is
*also* insufficient: liveness proves X is dead, but not that op N is *semantically
safe* to write into X's buffer. Elementwise reads each element before storing (safe);
a reduction or a matmul reads X many times across threads while writing (unsafe); `cat`
writes disjoint regions (safe only for the block whose source is X). The planner must
be TOLD which operand positions an op may consume-before-write.

### 3.2 The declaration: `DONATABLE_OPERANDS` (a static op property, sibling of `IN_PLACE_DST_INPUTS`)

Add a static, op-keyed map declaring which input positions an op may write its primary
output into, given the liveness precondition holds — the exact shape of the existing
`IN_PLACE_DST_INPUTS` (`plan-builder.ts:104-109`), which already declares "these inputs
are overwritten in place" and drives WAR ordering. Donation is its out-of-place analog:

- **elementwise (binary/unary/cast)** — position(s) equal in shape+dtype to out0,
  non-broadcast (this is what the fused detector already encodes at
  `segment-executors.ts:279-291`; lifting it to a declaration makes it planner-visible).
- **`copy_` / `stridedScatterCopy`** — dst position (already `IN_PLACE_DST_INPUTS[0]`;
  donation is the same buffer identity, expressed as a planner binding rather than a
  kernel side effect, so the packed unpack's `copy_(param, seg)` needs no new mechanism —
  it is already in-place into `param`; the leak is the `seg` *source*, §2.2).
- **`cat`** — the block position whose source input is dead and whose destination region
  is a prefix/aligned sub-range (the classic XLA concat donation; elides one
  block-copy). Optional/last — see §7 refusals.

The declaration is a function of the op alone (cheap, once-per-op, cacheable) — the same
"decide once per class" discipline the Muon pack-verdict uses.

### 3.3 The planner binding (where it attaches)

In `memory-planner.ts planMemory` phase 2 (the greedy first-fit loop, `:255-276`), when
assigning the alloc slot for op N's output, if:

1. op N declares operand position p donatable (§3.2), AND
2. the operand's slot X satisfies `lastUse(X) == allocIdx(outputSlot)` (X's last read is
   this op), AND
3. X is a **temp** (not in `resultSlots`, not an external/`registerState` result), AND
4. X's size-class matches the output's,

then **bind the output slot to X's already-assigned entry** instead of drawing a fresh
one. The planner already has an intra-step cross-plan analog to reuse — `externalReleases`
/ `claimedEntries` (`memory-planner.ts:161-169, 243-276`), where one plan overlays a
prior plan's released result entry. Donation is the *intra-plan* case of the same idea
(output N overlays input X within one command stream). The existing structural overlap
audit (`:277-329`, throws `OVERLAP`/`RESULT-SHARED`/`CLAIM-OVERLAP`) is the assertion
seam: a bad donation that overlaps a still-live interval becomes a **build failure**,
not a silent corruption.

### 3.4 Interaction with compiled-plan REPLAY — falls out of assignment, with ONE new edge

The **assignment** needs no new replay machinery: donation binds output slot N and input
slot X to the *same* `entryIdx`; replay's `TAG_ALLOC` rebind (`compiled-plan.ts:1051-1102`)
already maps slot→entry, so two slots mapping to one entry replays correctly and
per-replay rebinding is unchanged. Donation "just falls out of liveness-aware
assignment" for the buffer identity.

What DOES need a new plan-level edge is **suppressing the two anti-alias guards** that
currently forbid exactly this (both surfaced by the code map):

- `resolveOutputBuffer` (`buffer-arena.ts:568-579`) actively **releases and reallocates**
  any output buffer that aliases an input — the direct inverse of donation. A donation
  edge must mark op N's alloc so this replacement is suppressed *for the donated
  operand only* (via the existing `providedOutBuffer` seam, `buffer-arena.ts:476`, or an
  `AllocCommand` flag alongside `inputSlots`, `compiled-plan.ts:84-95`).
- The planner's own **next-command rule** (`memory-planner.ts:145-151`) forbids
  same-command read+write of one slot, noting in-place forms "already share a single
  slot upstream via donation/in-place discipline." So the cleanest expression is to
  **collapse X and output-N to a single slot in the plan/stream builder upstream** (the
  IR carries one slot for both), which makes the planner assign one entry with no
  aliasing exception at all. Preferred over a per-binding suppression flag: it keeps the
  planner's single-slot invariant intact and the overlap audit unmodified.

**Ruling: the new mechanism is a DONATION EDGE at plan-build time (collapse donated
operand X and output N to one slot), not a new planner algorithm.** The planner's
liveness assignment and overlap audit consume it unchanged.

### 3.5 park-live, arena, and the WAW `sharedEncoderWriteSet` check — explicit rulings

- **park-live / `canRecycle`.** A donated buffer that some live storage still aliases is
  the "later data leaks into earlier results" bug incarnate. Precondition 3 (X is a
  temp, no live owner) is *exactly* `canRecycle(X.buffer) === true`
  (`buffer-pool.ts:173-177`: no `bufferLiveCount` owner AND not in `sharedEncoderWriteSet`).
  **Donation MUST consult `canRecycle` at bind time and refuse when false** — a
  `registerState`'d / snapshot-member / cross-plan-read buffer is never donatable. This
  is what keeps park-live (`compiled-plan.ts:618-660`) correct by construction: donation
  only ever aliases a buffer with no live reader, so teardown never destroys a buffer a
  live storage still needs.
- **Arena.** The executor-side form refuses arena/pinned buffers
  (`segment-executors.ts:293`). Planner-level donation **inverts** this correctly: the
  planner OWNS the registry/arena buffers, so donating one registry entry into another
  op's output is an ownership-preserving assignment (one entry, one owner-plan), not a
  cross-regime hazard. This is the capability the executor-side form structurally lacks
  and the reason donation must live in the planner.
- **WAW `sharedEncoderWriteSet`.** Donation adds op N's output (= X's buffer) to the
  write set. Within one dispatch, elementwise consume-before-write is per-thread safe
  (no cross-thread hazard — the property the fused donation already relies on,
  `fusion-types.ts:135-144`). The cross-dispatch hazard — an *earlier-in-encoder*
  unsubmitted dispatch still reads X's old value while op N overwrites it — is ruled out
  by precondition 2 (`lastUse(X) == allocIdx(N)`): X has no later reader, so no queued
  reader survives op N. The `enforceWriteAfterReadOrder` WAR pass
  (`plan-builder.ts:123-218`) that already sequences readers-before-in-place-writer
  extends to the donated pair verbatim (donation is a declared in-place-like write).
  **Ruling: donation within one shared encoder is permitted iff the WAR order pass has
  sequenced every reader of X before op N; the `sharedEncoderWriteSet` WAW check remains
  the runtime backstop and must stay armed** (`TORCHLETTE_STRICT_GPU=1` in CI).

---

## 4. The safety story

Donation is the archetypal *"a later step's data leaks into an earlier step's results"*
bug class (WebGPU Buffer Pool Invariants; the naive-`canRecycle`-reuse NaN, §7). The
invariants, each with its enforcement seam:

- **Single owner of the donatable decision.** The op DECLARES eligibility (§3.2); the
  PLANNER alone proves the liveness precondition and makes the binding (§3.3). Two sides
  never independently conclude "this is safe to reuse" — the classic silent-divergence
  seam. Assert-at-bind: the overlap audit (`memory-planner.ts:277-329`) validates the
  donated entry has no overlapping live interval, turning any bad donation into a build
  failure.
- **The precondition IS `canRecycle`.** No donation onto a buffer with a live owner or a
  pending encoder write (§3.5). This is the persistence-contract UAF class
  (`architecture-debt.md` ledger; `sequential-corruption-open`) restated as a bind-time
  gate.
- **The `[lifetime]` strict guard is the standing detector.** A donated buffer whose
  original storage is read after the donating op is exactly what
  `getInputStorage`'s `[lifetime]` reclaimed/released-read guard throws on
  (default-throw since task #73). It runs on every test and training step, so a bad
  donation that escapes the audit is caught the moment the stale read fires. Gate:
  strict-lifetime green across the full suite AND the 124M soak.
- **Differential gates (Corollary 1 + 2, cross the activation threshold).**
  - `tools/parity-fullstack-tl.ts` twice (`TORCHLETTE_COMPILED_PLAN=0` vs default),
    per-step loss ≤ 1e-5 over 30 steps — donation must not perturb the trajectory.
  - `tools/parity-packed-vs-unpacked.ts` — packed vs per-param, Adam/Lion/SGD/Muon,
    bit-exact, crossing the compiled-plan activation threshold.
  - **NEW targeted spec `test/donation-parity.spec.ts`:** for the covered ops
    (elementwise last-consumer, `cat`, the packed unpack), donation-on vs
    `TORCHLETTE_PLANNER_DONATION=0` produces **bit-identical results** AND
    **peak(donation) ≤ peak(no-donation)**. This is the Corollary-1 cross-path guard the
    mechanism ships with. `tools/test-donation-multiout.ts` (the multi-output
    donate-into-every-output corruption pin) is folded in as a standing cell.
  - **Subsumption check** (from `stage4-compile-from-ir.md` phase 1): with planner
    donation ON, the executor-side kernel donation can be disabled (`TORCHLETTE_DONATION=0`)
    and peak memory must be **equal or better** — proving the planner fact subsumes the
    executor special form rather than double-counting it.

---

## 5. Phase plan (each shippable + gated)

**Standing gate set (every phase, before landing):** `npm run test:gates`
(`compiled-plan-parity.spec.ts`); `tools/parity-fullstack-tl.ts` twice
(`COMPILED_PLAN=0` vs default) ≤ 1e-5 / 30 steps; `tools/parity-packed-vs-unpacked.ts`
4 arms bit-exact; the 124M DiLoCo regression `{0:9.81, 3:5.92, 6:5.15, 9:4.64}` EXACT;
distil **9** / medium **18** submits non-regression (`TORCHLETTE_PROFILE=1 …
profile-training.ts`, read LATE steps AND confirm the trend is flat); full suite green.
GPU work serial-exclusive (`tools/pick-gpu.sh`, HOST node toolchain).

| Phase | What lands | Gates (standing +) | Cashes / notes |
|---|---|---|---|
| **P0** | THIS design doc. | — | No `src/` change. |
| **P1 — fix the packed-path leak (PREREQUISITE, not donation). ✅ LANDED 2026-07-21 (see §2.4).** | DONE: `pack-optimizer.ts` disposes the unpack `narrow`/`seg` materializations (the `seg` reshape was the leak; `pNew` was already disposed via `sink`). Option-(b) copy-elision rejected for P1 — it needs a source-offset region-read = donation itself. Net +8 SLOC, no deletion (cashed at P5). New standing gate `tools/packed-optim-flatness.ts`. | ✅ **Packed memory FLAT** (slope 0.000/step, plateau ~step 5–12; Adam/Lion/SGD, compiled + lowered); `parity-packed-vs-unpacked.ts` 4 arms bit-exact; `fused-vs-elementwise` (12) + cpu project (1551) green; strict `[lifetime]` zero throws. **TRUE flat premium (A100): distil +2.1 % peak / medium +45.6 % peak.** | Unblocks an honest gate. **Verdict: donation is NO LONGER a P4 blocker in the "unbounded leak" sense (P1 fixed it); it REMAINS the required mechanism for the flat +46 % medium residual → chain-packing P5. `X%` set from medium, not distil.** |
| **P2 — planner-level donation, opt-in. ⚠ IMPLEMENTED-then-STOPPED 2026-07-21 (§2.5).** | LANDED opt-in (`TORCHLETTE_PLANNER_DONATION=1`): `DONATABLE_OPERANDS` declaration (§3.2); the donation edge (slot-collapse) in `generateFused` (§3.4), un-refused for arena temps, driven by the executor's raw liveness proof (`cachedDonatableIds`). The executor detector is NOT yet subsumed (it stays live on the flag-off path). | ✅ bit-exact ON≡OFF (`parity-packed-vs-unpacked.ts`, maxDiff 0.0, 4 arms, across the threshold); strict-lifetime zero throws. ❌ **premium recovery = 0 %** (distil packed peak 5753.5 MB ON==OFF): the edge is INERT — packed plans are UNCOVERED (recorded build, not the generated stream) and buffers are OVERSIZED→chunked (§2.5). | Blocked. The premium precondition needs a stage-4 COVERAGE campaign for the packed optimizer plan + chunked in-kernel donation FIRST (§2.5 recommendation) — a prerequisite, not slot-collapse work. |
| **P3 — THE FLIP (satisfies chain-packing P4 precondition).** | `TORCHLETTE_PLANNER_DONATION` default-on; packed optimizer `pNew`→`P`, `G` packed-in-place donation active on the compiled path. | **Hard:** 124M `{…}` EXACT; distil 9 / medium 18 submits EXACT; **packed-arm peak within X% of fused** (X from the P1 A100 re-measure) on A100, FLAT; `parity-fullstack-tl.ts` twice; fused-vs-packed trajectory. Re-measure A100 fresh at flip. | This is the precondition `chain-packing-design.md` P4 gates on. With it met, chain-packing P4 (packed optimizer default on WebGPU) proceeds; **chain-packing P5 then deletes the fused `adamStep` monolith (~1.3–1.6k SLOC)** — the campaign-level payoff. |

**Which phase flips chain-packing P4:** P3. **What P5 then needs:** nothing further from
donation — once the packed default holds distil-9/medium-18 submits at flat memory
within `X%` of fused (P3's gate), the fused kernel is unreferenced and P5's deletion is
purely the chain-packing §5 ledger.

---

## 6. Deletion ledger + covenant

Donation is a **net-additive but small** mechanism, and it **subsumes** an existing
special form:

| Item | Fate |
|---|---|
| `DONATABLE_OPERANDS` map + planner bind logic (§3.2–3.3) | **NEW** (small; sibling of `IN_PLACE_DST_INPUTS`) |
| donation edge / slot-collapse at plan-build (§3.4) | **NEW** (small) |
| `test/donation-parity.spec.ts` | **NEW** (test, free per complexity budget) |
| executor-side fused donation: `segment-executors.ts:252-316` eligibility loop, `donationSink`, `donatableInputIds` plumbing (`executor.ts:2681-2714`), `cachedDonatedRecipeIdx`, `stream-generate.ts:4391` post-hoc reproduction | **SUBSUMED** → becomes the planner fact (delete the per-segment detector; keep the kernel's `donatedInput` binding form, now driven by the plan) |
| `TORCHLETTE_DONATION` flag | retires (folded into `TORCHLETTE_PLANNER_DONATION`, itself born-with-sunset) |

Donation's own src delta is roughly **net-neutral** (new declaration + edge ≈ the
executor detector it deletes). **Does park-live simplify?** Not directly — park-live
stays as the teardown discipline; donation respects it (never aliases a live-owned
buffer) rather than removing it. **Does a pool special case fall?** The
`arenaBufferSet`/`pinnedBufferSet` *exclusion* in the executor detector
(`segment-executors.ts:293`) dies with that detector — the planner needs no such
exclusion because it owns those buffers.

**Covenant (campaign-level).** Donation is chargeable to the chain-packing campaign: it
is the P4 precondition whose payoff is chain-packing **P5's ~1.3–1.6k SLOC deletion** of
the fused `adamStep` monolith. Donation itself (net-neutral) plus that deletion makes
the *combined* campaign strongly **net-negative src SLOC**. The one genuinely new
mechanism (the donation edge + declaration) is warranted: it is the single seam that
lets the planner reuse dying buffers uniformly — closing the foreach premium AND the
broader stage-4 goal — where the executor special form could not reach the buffers that
matter. Every phase names its deletions in the commit (house policy);
`bash tools/weight-norm.sh --log` snapshots at campaign end.

---

## 7. Risks and refusals

### 7.1 What the planner REFUSES to donate (typed, named — correct-and-slow fallback = fresh alloc)

- **`LiveOwnerRefusal`** — X has a live owner or a pending encoder write (`canRecycle(X)
  === false`): a `registerState`'d / snapshot-member / cross-plan-read buffer. This is
  the persistence-contract UAF class; never donate. (§3.5)
- **`ResultOrExternalRefusal`** — X is in `resultSlots` or is an external/leaf input a
  later plan reads (`getLivePendingRootNodes`, `tensor.ts:146-158`). Cross-plan readers
  make X not-dead.
- **`LaterReaderRefusal`** — `lastUse(X) != allocIdx(outputSlot)`: a reader survives op
  N. The WAW/WAR hazard.
- **`ShapeDtypeRefusal`** — `sizeOf(X) != sizeOf(out0)`, dtype mismatch, X scalar, or X
  broadcast (partial write leaves stale bytes). Matches the fused detector's hard
  requirements (`segment-executors.ts:290-291`).
- **`NonElementwiseRefusal`** — op position not in `DONATABLE_OPERANDS` (a reduction /
  matmul / any op that reads X across threads while writing). Hard gate, not a heuristic
  — the same clause-4 rigor `chain-packing-design.md` §6.1 demands for Muon's `mm`.
- **`MultiOutputRefusal`** — donate into the PRIMARY output only; additional outputs keep
  fresh allocations. Donating one buffer to multiple writable bindings is a WebGPU
  validation error that drops the whole submit (pinned by
  `tools/test-donation-multiout.ts`; `segment-executors.ts:270-277`).
- **`DupBindingRefusal`** — X already bound elsewhere in op N's dispatch (read+rw of one
  buffer in one dispatch is a WebGPU validation error).

Every refusal → fresh output allocation (today's behavior). Refusal is never a
correctness compromise, only a missed memory optimization.

### 7.2 Do NOT re-walk (prior attempts, from the ledgers)

- **Naive `canRecycle`-reuse NaN'd the 124M** (`MEMORY.md`
  arena-memory-blowup-124m: *"Naive canRecycle-reuse NaN'd (do not retry)"*). Donation is
  NOT "reuse any recyclable buffer" — it is a liveness-PROVEN, op-DECLARED, overlap-AUDITED
  binding. The distinction is the whole safety story (§4); do not collapse it back to
  opportunistic reuse.
- **Flushing `pendingRelease` to pool mid-step** — deterministic ~2% loss drift
  (`CLAUDE.md` WebGPU Buffer Pool Invariants). Donation must not touch `pendingRelease`
  timing; it operates on planner entry assignment, not on pool reclamation cadence.
- **Immediate `buf.destroy()` mid-encoder** — poisons the pending submit; all destruction
  stays fence-gated `deferredDestroy`. A donated buffer is *reassigned*, never destroyed
  early.
- **Multi-output donate-into-every-output** — the corruption `test-donation-multiout.ts`
  pins; primary-output only (§7.1 `MultiOutputRefusal`).
- **`arena-positions-acquire-from-pool`** (`TORCHLETTE_ARENA_POOL_ACQUIRE`,
  `architecture-debt.md` stage-2 row) — broke compiled replays via slot aliasing;
  superseded. Donation must not reintroduce cross-regime buffer identity.

### 7.3 Named risks

- **R1 — the P1 leak masks the true premium.** Until P1 lands, no honest packed-vs-fused
  memory number exists. Mitigation: P1 is a hard prerequisite with a FLAT-memory gate;
  P4's `X%` is derived only after it. **Do not tune donation against the leaky baseline.**
- **R2 — donation on a strided/offset X.** A partial-covering write leaves stale bytes in
  X's buffer that the output view then reads. Mitigation: `ShapeDtypeRefusal` requires
  full-buffer, contiguous, same-size donation; strided X refuses.
- **R3 — compiled-replay staleness.** If the donation edge (slot-collapse) is not
  reproduced identically at replay, output N binds a different buffer than the recording
  → frozen/stale (the frozen-`step_size` class). Mitigation: donation is a PLANNER
  assignment reproduced deterministically per replay (§3.4); the stream differential
  (`stage4` phase 0) and `parity-fullstack-tl.ts` across the activation threshold catch
  any drift.
- **R4 — subsumption double-count.** If both the executor detector and the planner donate
  the same pair, one may free a buffer the other still binds. Mitigation: P2 deletes the
  executor detector as it lands the planner fact; the subsumption check
  (`TORCHLETTE_DONATION=0`, equal-or-better peak) proves single ownership.

---

## 8. Genuine taste-calls for the user

1. **Split the P4 gate (§2.3).** This design asserts the campaign's stated premise
   ("donation closes the 2× premium") is contaminated by a per-step leak, and inserts a
   leak-fix phase (P1) before donation. If you'd rather donation own the whole memory
   story, the leak still has to be fixed first — but we could fold P1 into the donation
   campaign vs. filing it as a standalone `pack-optimizer` disposal bug. Recommendation:
   file P1 as its own fix (it is a correctness bug on the packed path today, independent
   of donation) and let the donation campaign start from a flat baseline.
2. **`cat` donation scope (§3.2).** Elementwise + `copy_` donation is clearly in scope
   and closes the packed premium. `cat`-block donation (elide the first block-copy) is a
   further win but adds the disjoint-region reasoning. Ship P2 with elementwise+`copy_`
   only and add `cat` as a P2.5 increment? Recommendation: yes — keep P2 minimal.
3. **Slot-collapse vs. suppression flag (§3.4).** The design prefers collapsing the
   donated operand and output to one slot upstream (keeps the planner's single-slot
   invariant) over a per-binding anti-alias suppression flag. Both are viable; the
   slot-collapse is more invasive to plan-build but leaves the planner and its audit
   untouched. Recommendation: slot-collapse.
