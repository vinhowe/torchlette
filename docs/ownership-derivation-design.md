# Ownership Derivation: liveness classification as a derived fact

**Status:** RATIFIED (Vin, 2026-07-11) — D0 complete; §8 ruled WRAPPERS. D1 in flight. · Task #70
**Prereq:** #73 strict-lifetime default (in flight) — the tripwire net this campaign works under.

## 0. Declaration (one sentence)

A storage's liveness classification — persistent vs step-scoped vs graph-retained — is
DERIVED at the point of inquiry from single-source facts (live wrappers, their birth
generation, registered state, graph retention); it is never stored as mutable slots that
must be defended.

## 1. The indictment (why this is the correctness steepest-ascent)

Every entry in the storage-lifetime bug ledger traces to classification living as
mutable, distributed state:

| Incident | The stored-state failure |
|---|---|
| persistence-contract UAF (2026-06-10) | demotion sweep trusted stored classification over actual holders |
| pool double-release | ownership recorded twice, released twice |
| #68 Adam churn, #69 replay reclaim | replace-and-hold vs stored ownership |
| #90 harvest-view FP | per-replay handle's stored state contradicted its base's |
| #86 mech 1 | `_wrapperGen` stamped at the WRONG MOMENT (materialize, not birth) |
| #86 mech 2 / #74 mech 1 | the single WeakRef owner SLOT stolen by clones/sidecars |
| #74 `persist()` no-op | persistence recorded into a set that wasn't listening between steps |
| #73 flip finding (2026-07-11) | parallel-run flake: a concurrent test perturbs the SHARED generation counter → a stepAsync param mis-filtered from the persistent snapshot → FP throw (gpt2-memorization; opted out for soak). Stored classification is fragile even to TEST INTERLEAVING. |

Each fix added an invariant DEFENDING the stored state: birth-time stamping, two
steal-refusal carve-outs (`_graphRetained`, `_sidecarShare`), a second persistence
mechanism (`_durablePersistent` beside the snapshot), guard exoneration clauses
(`declaredReplay`, `viewAliasesLiveBase`). All correct; all clauses. Clauses accreting
on a classifier are the tell that classification is distributed rather than derived —
the same disease #91 cured for cache keys (key=content made drift unconstructible).
This campaign does the same for ownership: make misclassification unconstructible by
removing the stored thing that can disagree with reality.

## 2. The root flaw: ONE owner slot, MANY holders

`storageTracker` maps each storage to a SINGLE WeakRef "owner" wrapper. Reality:
multiple wrappers legitimately share one storage (views, retention clones, sidecar
pins, cross-engine handles). One slot forces a "who is THE owner" fiction; every
mechanism above is a patch on that fiction.

## 3. The derived model

**Source facts (each already exists or is a small delta):**
- **W(s)** — the set of live wrapper objects referencing storage s. Structural change:
  owner SLOT → owner SET of WeakRefs (pruned via the existing FinalizationRegistry).
- **gen(w)** — wrapper birth generation (exists post-#86, stamped in the ctor).
- **SNAP** — the step snapshot (exists).
- **REG** — registered state: modules/optimizers DECLARE their state tensors
  (nn.Module parameters/buffers already enumerate; optimizers register m/v/velocity).
  Subsumes both `persist()` calls and `_durablePersistent`.
- **G(s)** — graph retention count (exists: autograd saved-slot rc).

**Derived classification (computed at inquiry, never stored):**
```
persistent(s)  := ∃ w ∈ W(s) : w ∈ SNAP ∪ REG
graphHeld(s)   := G(s) > 0
stepScoped(s)  := ¬persistent(s) ∧ ¬graphHeld(s)     // eligible for demotion at markStep
dangling-read  := inquiry on s with W(s) = ∅ ∧ G(s) = 0 ∧ not plan-retained
```
- The steal problem dissolves: there is no slot to steal — a clone joining W(s) cannot
  UNMAKE another wrapper's snapshot membership.
- `viewAliasesLiveBase` becomes a first-class query (base-chain liveness = derived
  classification of the root), not a guard clause.
- `persist(t)` becomes sugar for REG registration (or dies — decided at review).

## 4. Deletion targets (named per policy)

The owner-slot machinery and its defenses: the two steal-refusal carve-outs, the dual
persistence sets (merged into SNAP ∪ REG), the stamp back-fill path, `releaseStepTemps`'s
owner-based demotion logic (re-expressed over derived stepScoped), and — stretch —
the guard's exoneration clauses re-derived from the model. Expected net: the
storage-tracker (currently the most clause-dense file in src/) shrinks materially;
`persist()` call sites across optimizers/modules die into registration.

## 5. Phases (assert-agreement throughout — this campaign's null differential)

- **D0 — design review** (this doc): the one open question set below.
- **D1 — owner SET + derived classification, SHADOW mode:** compute BOTH the stored
  classification and the derived one at every decision point; `assert-agreement` under
  the full suite + strict + the training tools + browser gates. Ship with zero behavior
  change. Divergences are bugs in one side or the other — each one adjudicated and
  documented (some may be latent bugs in the CURRENT model).
- **D2 — flip:** derived becomes authoritative; stored model + carve-outs DELETED.
- **D3 — registered state:** optimizer/module registration API; `persist()` sunset.
- **D4 — guard simplification:** exoneration clauses re-derived or deleted.

## 6. Gates

Strict-lifetime default (armed by #73) across everything; the D1 agreement differential
(zero unadjudicated divergences); all existing lifetime gates (#86, #90, #74 specs);
memory flatness (browser reachableStorages gate, profiler zero-leak lines, the 124M
regression); perf: wrapper-set maintenance is on the tensor-create hot path — the
browser progressive-slowdown history (V8 GC, FinalizationRegistry pressure) sets the
budget: no measurable tok/s regression in the browser LoRA baseline and no RT/step
growth regression.

## 7. Risks

(a) **Hot-path cost** of set maintenance — mitigate: per-storage arrays with lazy
pruning, measured in D1. (b) **GC-timing sensitivity** — #74 found bugs that ANY
instrumentation masks; the D1 shadow must be always-on in the differential runs, not
sampled, and browser-validated. (c) **Set semantics for cross-engine handles** — the
between-engine reset (#74) must clear W(s) contributions from a dead engine; covered by
the three #74 acceptance specs. (d) **Scope creep into scoped-memory (#66)** — the
scope-stack campaign consumes this model but is NOT this campaign; the seam is that
SNAP generalizes to scope membership later.

## 8. The open design question (for review)

RULED (Vin, 2026-07-11): **WRAPPERS.** Matches the snapshot's semantics and survives
storage replacement (copy_-in-place keeps the wrapper); storages would reintroduce the
replacement-and-hold ambiguity that caused #68. Consistent with the wrapper-level-stamps
rule from the implied-boundary work.

## §D1-log — shadow-differential adjudications (task #70, D1)

D1 built the owner SET + derived classifier alongside the stored model and asserts
agreement at the classification-consuming decision points (`releaseStepTemps` demotion;
the `[lifetime]` guard). Divergences throw under a test context (VITEST detected — no new
flag) and count+single-warn otherwise. Two counters: `shadowDivergenceCount()` =
UNADJUDICATED (gate: must be 0) and `shadowAdjudicatedCount()` = known verdict-recorded
classes below. Implementation: `src/graph/storage-tracker.ts` (`_ownerSet`, `_derived`,
`_shadowCompareSurvival`, `shadowCompareGuard`).

**The comparison axis is SURVIVAL, and derived is a strict SUPERSET of stored survival.**
The stored model and derived model classify DIFFERENT things: stored decides "release THIS
slot-wrapper's rc-claim" (per-claim, at `releaseStepTemps`), derived decides "does a KEPT
holder keep this storage" (per-storage). Comparing raw claim-verdict false-fires whenever a
storage has MULTIPLE holders (a step-temp wrapper AND a view-base retain / graph clone /
sidecar pin): stored releases the one claim, the storage SURVIVES via the others. The
behaviorally-meaningful invariant is whether the STORAGE SURVIVES the boundary (destroyed
at the following `destroyUnreachable`?):
- `storedSurvives := storedPersistent ? rc>0 : rc-1>0` — stored releases exactly one rc for
  a non-persistent slot; the storage survives iff rc stays > 0. This is the GROUND TRUTH of
  survival (rc-counting is the actual mechanism).
- `derivedSurvives := keptHolder ∨ storedSurvives` — the derived model's ONE contribution
  over stored's rc arithmetic is `keptHolder` (a live persistent/graph/sidecar/view-base
  holder). ORing with `storedSurvives` makes derived a strict SUPERSET: it NEVER predicts
  reaping more than stored (the per-slot rc arithmetic is authoritative for that), only
  KEEPING more. So the ONLY reachable divergence is `keptHolder ∧ ¬storedSurvives` — the
  stored per-slot release would REAP a storage the derived model knows still has a live kept
  holder (the reap-live UAF class this campaign hunts). `_shadowCompareSurvival`.

**Derived model refinements forced by the differential (the shadow found its own bugs
first — the intended D1 outcome):**

- **Refinement A — `keptHolder` includes non-wrapper retains; rc is the survival ground
  truth.** First derived draft treated W(s)=∅ / a dead storage as demotable. The
  ledger-attack probe surfaced live param storages with `rc=1, ownerSet=∅` — an OLD param
  storage still held by a NON-wrapper retain (a live view's base retain) after the param
  wrapper moved to the adamStep result; and `lifetime-natural-usage`/`compile-autograd`
  surfaced the inverse (a dead `rc=0` storage whose id is still NAMED as a view base). Final
  `keptHolder := rc>0 ∧ (persistent ∨ graphHeld ∨ sidecar ∨ nonWrapperRetain)`, where
  `nonWrapperRetain` = "storage is the BASE of a live view" (detected STRUCTURALLY,
  `_liveViewBases`, computed once per sweep — rc-counting is unsound since one wrapper holds
  ctor+materialize rc=2) OR "W(s)=∅ with rc>0". The `rc>0` guard makes rc the survival
  authority (a dead storage never survives). This is doc §3's "not plan-retained" clause
  made concrete for the demotion site.

- **Refinement B — `graphHeld` AND `sidecar` are keep signals.** `graphHeld` := ∃ live
  `_graphRetained` wrapper (autograd saved-slot retention clone — the model's G(s)>0);
  `sidecar` := ∃ live `_sidecarShare` wrapper (GradScaler LiveScalar pin, task #74 — a
  persistence pin). Both keep the storage alive via their own rc and must not be reaped.
  Missing these produced the storage-6 (sidecar) and storage-157/497 (graph clone) false
  divergences (autograd-scope-lifetime, fused-vs-elementwise); the survival axis + these
  signals resolve them.

- **Refinement C — owner set kept through storage destruction.** The set is NOT dropped
  at `destroyStorageIds`/`unregister`; it self-prunes via WeakRef deref (a wrapper leaving
  a storage is removed at its dispose/`_updateLazyRef` seam via `untrackTensor`; a bounded
  sweep in `destroyUnreachable` reaps empty sets of dead storages). This lets the guard
  shadow observe a live persistent/graph-held wrapper still pointing at a stored-destroyed
  storage (the reap-live class). Storage ids are monotonic, so a lingering set never
  aliases a reused id.

**ADJUDICATION #1 — stale disposed owner slot skews stored SURVIVAL (DEFER TO D2; derived
superior).** Site: `releaseStepTemps`. Root cause: the stored single owner SLOT
(`tensorWeakRefs[id]`) is cleared only when OVERWRITTEN by a later `trackTensor` for the
same id — never on dispose. So it can deref to a DISPOSED wrapper whose real liveness is
zero, skewing the stored survival estimate. The derived owner SET drops the wrapper at
dispose (`untrackTensor`), so the derived model reads the ACTUAL live holders and is
authoritative. Observed in `implied-step-boundary.spec.ts` (back-to-back optimizers,
minimal loop), the ledger-attack probe warmup, and the distilgpt2/second-run specs: a
graph-retention clone (`_graphRetained`) is captured into the markStep-Step-5
implied-boundary snapshot, then disposed at `cleanupAutogradGraph` but lingers in the
snapshot WeakSet — its stale slot makes the stored SURVIVAL estimate disagree with the
derived one (either direction, depending on whether the disposed wrapper reads persistent).
- **Verdict:** derived is correct; the stored slot's staleness is the fault (benign — the
  disposed wrapper is gone, no read, no UAF; masked in the ledger — dblRel=0, drift 0/0).
  **Disposition: defer to D2** — the slot is DELETED there and the class dissolves. In D1
  these are counted in `shadowAdjudicatedCount()` (logged once, never thrown) so the suite
  stays green while the differential stays honest. `shadowDivergenceCount()`
  (UNADJUDICATED) is the gate and is **0**.

**ADJUDICATION #2 — dead storage with a stale `liveViewBases` name (FIXED IN DERIVED; not
a stored bug).** Surfaced by `lifetime-natural-usage.spec.ts` (mid-loop
markStep+dispose+beginStep) and `compile-autograd.spec.ts`: a storage at `rc=0` whose id is
still NAMED as the base of a view whose own rc>0 (the view already released its base retain
via `_updateLazyRef`/earlyRelease but kept the `baseStorageId` field). First derived draft
read `liveViewBases` as authoritative and predicted survival for a dead storage. Fixed by
making the derived survival prediction **rc-grounded**: `survives := rc>0 ∧ keptHolder`. rc
is the authority for "is anything still holding it"; the wrapper/view classification only
decides whether releaseStepTemps drops the last holder. Not a divergence between stored and
derived once derived is rc-grounded — a derived-model bug the differential caught and fixed
before D2.

**ADJUDICATION #3 — the row-8 / #73 stepAsync generation-perturbation FP (EXPECTED;
derived's first proof of superiority; FIX SHIPS AS THE DERIVED MODEL IN D2).**

The #73 flip finding (indictment row 8): under the FULL parallel cpu-project run, a
concurrently-interleaved test perturbs the SHARED generation counter, occasionally
mis-filtering a `optimizer.stepAsync()` param OUT of the stored persistent snapshot →
`releaseStepTemps` reaps its live storage → a `[lifetime]` reclaimed-read throw. Proven a
FALSE POSITIVE (the loop converges to loss 0.0000 under warn-mode). `gpt2-memorization.spec.ts`
opts that one path out of the throw (`TORCHLETTE_STRICT_LIFETIME=0`) with the note that "the
real fix lives in storage-tracker snapshot filtering, not here."

**That fix is the derived model.** The stored snapshot gen-filters EVERYTHING
(`snapshotForStep(maxGen)`), so a durably-`persist()`ed param whose birth generation drifts
past the perturbed boundary falls out of the snapshot. The derived classifier reads
persistence from **REG membership (`_durablePersistent`), which has NO generation term**
(doc §3: `persistent := ∃ w ∈ W(s) : w ∈ SNAP ∪ REG`). In `_derived` the REG check runs
BEFORE the gen-filter `continue`, so a registered state tensor is classified persistent
whatever the boundary's closing gen — the misclassification becomes UNCONSTRUCTIBLE. This is
the derived model's first proof of superiority: where the stored side's gen-perturbed
snapshot would reap the live param, the derived side keeps it, exactly at the historical FP.

- **Disposition:** the derived classification is correct and ships authoritative in D2,
  which lets the `gpt2-memorization` `STRICT_LIFETIME=0` opt-out (and the row-8 comment) be
  DELETED. In D1 the derived model is shadow-only (asserted, not authoritative), and
  `gpt2-memorization` passes green (loss 0.0000). Validation note: the FP is a
  parallel-cpu-interleaving flake; `gpt2-memorization` is a single-fork webgpu/slow spec, so
  the perturbation does not reproduce deterministically in-suite — the superiority is
  established by construction (REG is gen-independent in `_derived`) rather than by catching
  a live flake.

**Perf / memory (doc §6 budget) — see §D1-perf below.**

## §D1-perf — hot-path cost + memory flatness (task #70, D1)

- **Owner-set maintenance** is on the tensor-create hot path (`trackTensor` →
  `_ownerSetAdd`, `_updateLazyRef`/`dispose` → `untrackTensor`). Each is O(members of one
  storage's set) with lazy WeakRef pruning — sets are size-1 for the overwhelming majority
  of storages (one wrapper). The shadow comparison itself runs only at `releaseStepTemps`
  (once per boundary) and the `[lifetime]` guard; `_liveViewBases` is computed ONCE per
  boundary (not per-storage) to keep the sweep O(N).
- **Perf delta (distilgpt2@512, A100, late-step steady state, node v22 + vk-shim,
  NUM_STEPS=25):** shadow-OFF (stashed) late steps 43–56 ms (median ~47 ms), peak 5476 MB;
  shadow-ON late steps 45–49 ms (median ~48 ms), peak 5397 MB. **Delta ≤ ~2 ms/step —
  within the run-to-run jitter (~5 ms, both traces show an occasional 200 ms+ V8-GC pause),
  and peak memory is if anything LOWER with the shadow.** Well within the ~50 ms-class
  budget. The owner-set maintenance (size-1 sets for almost every storage, WeakRef-only) is
  not a measurable hot-path cost.
- **Memory flatness:** the ledger-attack probe (STEPS=24, full-stack autocast + checkpoint
  + GradScaler + clip + AdamW) reports **reachDrift=0 totalDrift=0** and dblRel=0 with the
  shadow armed. The profiler's late-step Reachable count is flat at 510 (steps 9–24), zero
  leak. The owner sets are WeakRef-only (never extend a wrapper's lifetime) and self-prune
  (lazy on read + a bounded sweep in `destroyUnreachable` reaping empty sets of dead
  storages), so `reachableStorages` stays flat.
- **GC hard-boundary gate** (`test/ownership-shadow-gc.spec.ts`): drops all strong refs to a
  batch of tracked tensors, forces `global.gc` + FinalizationRegistry turns, and asserts the
  owner set's live membership collapses — a strong ref anywhere would pin the wrappers. This
  is the "shadow must not perturb GC-sensitive behavior" acceptance (the owner set is
  always-on and WeakRef-only, so on/off is structurally identical for GC).
- **Browser gate:** `@vitest/browser` + Playwright are not installed in this worktree's
  `node_modules` (only in the main checkout), so `test/browser/lora-training-trajectory`
  could not run here. Per the D1 plan it RIDES WITH D2 (run before the flip).

## §D1→D2 handoff — what D2 should delete FIRST

D1 proved the derived model AGREES with stored everywhere (zero unadjudicated divergences)
and is STRICTLY SUPERIOR at the adjudicated classes. D2 makes derived authoritative and
deletes the stored slot + its defenses. Delete in this order (each is now provably
redundant with `_derived` / the owner SET):

1. **The single owner SLOT and its steal machinery** — `tensorWeakRefs` as the *classifier
   input*, the `trackTensor` sharer-refusal carve-outs (`_graphRetained` / `_sidecarShare`
   incumbent checks), and the back-fill stamp path. The owner SET (no steal semantics)
   replaces it; graph/sidecar sharers become plain SET members read by `_derived`.
   (`ownerOf` may stay as a tape helper, but not as the persistence classifier.)
2. **`releaseStepTemps`' slot-based demotion** → re-expressed over derived `stepScoped` /
   the survival prediction. This is the site the whole §D1-log adjudicated; the stale/
   disposed-slot class (#1) dissolves the moment the slot stops driving the decision.
3. **The dual persistence sets merge** — `_stepStartTensors` (SNAP) and `_durablePersistent`
   (REG) stay as the two membership sets, but their gen-filter asymmetry is unified: REG is
   gen-independent (§D1-log #3), SNAP is gen-scoped. `persist()` becomes REG registration
   (this is D3's `persist()` sunset, but the gen-independence lands with D2).
4. **The `gpt2-memorization` `TORCHLETTE_STRICT_LIFETIME=0` opt-out + row-8 comment** — the
   FP it works around is unconstructible under the derived model (§D1-log #3). Delete the
   opt-out and re-arm strict on that path.
5. **The `[lifetime]` guard's `viewAliasesLiveBase` exoneration** is a D4 target, not D2 —
   but the derived `nonWrapperRetain`/`liveViewBases` query is the first-class form it
   becomes (doc §3: "viewAliasesLiveBase becomes a first-class query").

The shadow counters (`shadowDivergenceCount` / `shadowAdjudicatedCount`) and the D1
comparison plumbing (`_shadowCompareSurvival`, `shadowCompareGuard`) are themselves D2
deletions once derived is authoritative — the differential has served its purpose.
