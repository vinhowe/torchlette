# Coverage Campaign: making the packed optimizer plan `fullyCovered` so donation fires

> **Status:** P0 DESIGN (no `src/` change). This is the funded prerequisite named
> by `docs/buffer-donation-design.md` §2.5 (the P2 STOP): "treat *cover the packed
> optimizer plan* as the real P2.5 … BEFORE any P3 flip." It unblocks donation
> **P3** → chain-packing **P4** → the ~1.3–1.6k SLOC chain-packing **P5** deletion.
> Companion to `docs/stage4-compile-from-ir.md` (the coverage machinery this extends)
> and `docs/chain-packing-design.md` (the deletion this ultimately serves).
>
> **The census below CORRECTS the §2.5 blocker taxonomy in three material ways
> (all measured, sivri V100, §2):** (1) the packed-optimizer plan is **already
> 497/499 covered** — `cat`/`narrow`/`copy_`/`unscaleGrad` are NOT the uncovered
> mass (§2.5 expected them; they are covered); the sole steady-state gap is
> `op:lt ×2`. (2) The dominant blocker in the *measured premium regime* is NOT
> coverage at all — it is a **compilation REFUSAL** (`refuseCompileHazard`, the D3
> checkpoint-eager gate) that the §2.5 "UNCOVERED plans" label conflated with a
> coverage gap. (3) Chunked-elementwise is **already covered for execution**
> (`planChunkedBinary/Unary`); §2.5 blocker 2 is a DONATION-FORM gap, and the
> honest ruling replaces chunked in-kernel donation with **class-splitting** at the
> packer (§5.2).

---

## 1. The claim (one sentence)

**The packed optimizer plan is one comparison-op generator, one packer size-cap,
and one IR-derived liveness capture away from `fullyCovered` in the whole-step-remat
regime where it compiles — so the already-landed `TORCHLETTE_PLANNER_DONATION` edge
fires on it, closing the flat `G`/`P`/`pNew` premium that gates chain-packing P4→P5.**

---

## 2. The measured census — READ THIS FIRST (it corrects §2.5)

Measured on **sivri, 1× V100-32GB** (physical GPU 5, reserved via `tools/pick-gpu.sh`),
`tools/profile-training.ts`, distilgpt2 @ seq 512, packed path (`TORCHLETTE_FUSED_ADAM=0`
→ `_stepForeach` → `packOptimizerProgram`), 6–8 steps. Coverage read from the
`TORCHLETTE_DEBUG_CENSUS` build-block census (`executor.ts:2175`, which runs AFTER
`populateCapturesFromIR`, so it is post-capture / authoritative) plus a throwaway probe
that calls `generateStream` directly for each distinct plan (reverted; it under-reports
capture-dependent classes so it is used only for plan identification). **These are
V100-relative coverage FRACTIONS and op HISTOGRAMS — hardware-invariant (coverage is a
structural property of the op set, not of memory); absolute premium MB are cited from
the donation doc's A100 re-measure.**

### 2.1 The two regimes (the correction §2.5 missed)

The packed optimizer plan reaches the compiler in **two different regimes**, and its
fate differs:

| Regime | who runs it | optimizer plan fate | donation reachable? |
|---|---|---|---|
| **Eager checkpoint** (`tools/profile-training.ts` — where §2.4's premium was measured) | selective `checkpoint()` outside `api.wholeStep` | `refuseCompileHazard=true` → **runs LOWERED, never compiles** (D3 checkpoint-eager refusal, `checkpoint.ts:157`) | **NO** — no compiled plan, no planner, no donation edge |
| **Whole-step remat** (`webgpu-gpt2-trainer.ts:412` — REAL DiLoCo/124M training) | inner step under `api.wholeStep(...)` (`TORCHLETTE_WHOLE_STEP` default-on) | merged force compiles; optimizer segment builds from IR | **YES** — if `fullyCovered` |

**This is the root of §2.5's "donation never fires" result.** §2.5 measured on the
eager profiler, where the optimizer plan is *refused compilation entirely* — so it never
reaches `generateStream`, and the "187/311, 228/231 UNCOVERED" numbers it quoted are the
**forward-setup and zeroGrad plans**, not the optimizer plan. The optimizer plan itself
is nearly fully covered (below); its problem in the profiler is the REFUSAL, not coverage.

### 2.2 Per-plan coverage histogram (distil@512, packed, converged/steady-state)

Converged = the coverage after warmup captures populate (read the LATE census line per
plan; early lines show capture warmup and converge upward). `refuseHazard` is the D3
compile refusal in the **eager** profiler run.

| Plan (by node count) | role | actions covered | uncovered (converged) | eager `refuseHazard` |
|---|---|---|---|---|
| **510** | **packed optimizer (steady)** | **497/499** | **`op:lt ×2`** | true |
| 664 | packed optimizer (first step, packs `m`/`v`) | 568/653 → converges | `unscaleGrad[config-missing]` ×76*, `fused[no-input-pattern]` ×5, `op:lt` ×2, `full` ×2 | true |
| 558 | backward | **440/440 FULLY COVERED** | — | true |
| 305 | forward compute | **232/232 FULLY COVERED** | — | true |
| 276 | MLP recompute variant | **64/64 FULLY COVERED** | — | true |
| 231 | zeroGrad | 228/231 | `data-source:full ×3` | false |
| 358 | RNG / dropout-mask / causal-mask setup | 187/311 | `data-source:rand ×48`, `fused[no-input-pattern] ×48`, `full ×19`, `triu ×6`, `randn ×2`, `arange ×1` | false |

\* the `[config-missing]` / `[no-config]` / `[no-cached-plan]` / `[workspace-missing]`
suffixes are **capture-timing artifacts** — they appear only before `populateCapturesFromIR`
runs (first step, or in the direct-probe measurement) and CONVERGE AWAY once the
lowering-time captures populate. They are NOT structural generator gaps. Confirmed:
forward(305) and backward(558) reach 100% once converged; the optimizer's `unscaleGrad`
goes covered by the steady 510-node plan.

### 2.3 The derived kill-order (cover the MASS, not the alphabet)

The donation premium lives **entirely in the packed-optimizer plan** (the `G`/`P`/`pNew`
co-liveness). The forward/backward plans are already fully covered and hold no donation
candidates (their outputs are activations/grads consumed cross-plan, not
last-consumer-dead temps). So the campaign is scoped tightly to the optimizer plan, and
the kill-order is derived from what stands between it and a firing donation edge:

1. **The compilation REFUSAL (blocker 1, §5.1)** — the optimizer plan must COMPILE.
   In the whole-step-remat regime it does; the eager regime is a typed correct-and-slow
   refusal by construction. This gates 100% of the premium; nothing downstream matters
   until the measurement moves to the compiling regime.
2. **`op:lt ×2` (blocker 1b, §5.1)** — the SOLE steady-state coverage gap of the
   optimizer plan. Comparison ops have no `generateSequential` serializer. Closing it
   takes the plan 497/499 → **499/499 `fullyCovered`**. (The first-step `data-source:full`
   / `unscaleGrad` gaps converge on their own.)
3. **`cachedDonatableIds` not IR-derived (blocker 3, §5.3)** — even a `fullyCovered`
   compiled optimizer plan cannot donate, because the donation liveness proof is captured
   only as a side effect of the LOWERED segment executor (`executor.ts:2724`), which
   build-from-IR skips.
4. **Oversized `[Σ size]` pack → chunked (blocker 2, §5.2)** — the packed buffer
   (~328 MB distil / ~1.4 GB medium) exceeds the 128 MB binding limit → chunked
   elementwise, which generates but has no single-dispatch donation form. **Ruled: split
   the pack into ≤128 MB isomorphism sub-classes; chunked in-kernel donation is ruled OUT.**

The zeroGrad `data-source:full ×3` and the entire 358-node RNG/mask plan are **left
uncovered** — they hold no donation candidates and never gate the premium (§7.1). Covering
them would be alphabet, not mass.

---

## 3. What is already in place (reuse, don't rebuild)

- **The donation edge is LANDED, opt-in** (`TORCHLETTE_PLANNER_DONATION=1`): `DONATABLE_OPERANDS`
  declaration (`plan-builder.ts:137`), the slot-collapse in `generateFused`
  (`stream-generate.ts:4389-4444`), driven by `action.cachedDonatableIds`. Bit-exact ON≡OFF,
  strict-`[lifetime]` clean (donation doc §2.5). **Nothing about the edge needs redesign** —
  this campaign makes its substrate exist.
- **Chunked elementwise is covered for EXECUTION** (`planChunkedBinary`/`planChunkedUnary`,
  `stream-generate.ts:1192-1246`) — add/div/sub/mul/sqrt/neg over an oversized buffer
  generate a valid chunked stream. The gap is a *donation form*, not coverage.
- **The whole-step-remat compiling path exists and is default-on** (`TORCHLETTE_WHOLE_STEP`,
  `step-tape.ts:103`; `webgpu-gpt2-trainer.ts:412`). Real training already compiles the
  merged step; the optimizer segment builds from IR.
- **The memory planner + liveness** (`memory-planner.ts`, `executor.ts` `livenessLastAction`)
  already computes exactly the last-reader-per-node fact that `cachedDonatableIds` needs —
  the single source blocker 3's fix draws from.
- **The build-from-IR capture discipline** (`populateCapturesFromIR` / `makeMetaDeriver`,
  stage-4 phase 4.4) is the established seam for "derive a capture from IR instead of a
  live-execution side effect." Blocker 3's fix is one more capture in it.

---

## 4. Altitude ruling — coverage is a GENERATOR fact; donation-liveness is a PLANNER fact

Per the execution-declaration doctrine (`docs/execution-declaration-design.md`): **one
command-stream declaration per op family; dispatch = interpreter, generation = serializer.**
This campaign adds exactly two things at that altitude and one at the planner:

- **`op:lt` (and the comparison family) get a serializer** in `generateSequential`, the same
  altitude every other elementwise op's generator lives (blocker 1b). Comparison ops are
  already `isDeclaredElementwise` (execution-declaration.ts) — the dispatch interpreter
  handles them; only the serializer is missing.
- **The packer emits ≤128 MB sub-class cats** (blocker 2) — a realizer-altitude decision
  (chain-packing §2.1: packing is DECLARED by the optimizer, executed as ordinary graph
  nodes), so no chunked-elementwise donation surface is touched.
- **`cachedDonatableIds` becomes an IR-derived planner capture** (blocker 3) — the donation
  liveness proof is derived from the plan's liveness (the planner's single source), not
  observed from a lowered dispatch, satisfying single-source-at-the-seam.

Nothing here is a new *mechanism*; each is an instance of a discipline the codebase already
runs (a generator, a packer grouping key, an IR-derived capture).

---

## 5. Blocker rulings (one each)

### 5.1 Blocker 1 — coverage / compilation of the optimizer plan

**The optimizer plan reaches `fullyCovered` by closing `op:lt`; it reaches COMPILATION
only in the whole-step-remat regime, and that is the regime the campaign targets.**

**Coverage (blocker 1b).** Add a comparison-op serializer to `generateSequential`
(`lt/le/gt/ge/eq/ne`). These are declared-elementwise, single-output, no capture needed
(shape/dtype from `node`). This closes the optimizer plan 497/499 → 499/499. The first-step
`data-source:full ×2` and `unscaleGrad[config-missing] ×76` are capture-warmup and converge
without work (verified: steady 510-node plan already covers them). **Where does `op:lt` come
from?** The packed program carries an `abs/lt/where/neg/exp/full` cluster (2 each) alongside
the Adam `mul/add/sub/div/sqrt` — the GradScaler/clip finiteness masking folded onto the
packed buffer. `where` already generates; `lt` is the lone hole. (The comparison serializer
also incidentally closes any future `lt`/`gt` in Lion's `sign` or clip chains — a declared
family, not a one-off.)

**Compilation (blocker 1 proper) — the REFUSAL ruling.** The eager-checkpoint refusal
(`checkpoint.ts:143-159`) is **ALL-OR-NOTHING per step BY DESIGN**: its own comment states
that "a lowered checkpoint forward/backward feeding a COMPILED optimizer breaks the
grad/planner handoff → frozen training." Therefore:

- **DO NOT lift the refusal for the optimizer plan alone in an eager step.** That is
  precisely the partial-mix the D3 gate forbids — it re-opens the b66ead78
  compiled-forward-reclaims-recompute-activation hazard (one of the recorded-build STOP
  classes, §6). Ruled OUT.
- **Target the whole-step-remat regime**, where the merged step compiles as ONE unit (no
  partial mix) and the optimizer segment builds from IR. This is already how real training
  (`webgpu-gpt2-trainer.ts`) runs, and it is where the premium must be RE-MEASURED (§5.4).
- **The eager-checkpoint path stays lowered-and-donation-free by construction** — a typed,
  correct-and-slow regime, not a bug. The profiler is adapted to also offer a
  whole-step-wrapped mode for premium measurement (a tool change, not src).

**Plan-segmentation caveat (a risk to confirm at build):** the census was taken on the
eager profiler, where forward(305)/backward(558)/optimizer(510) are SEPARATE plans (distinct
forces), so `op:lt` blocks only the optimizer plan. Under whole-step remat the merged force
may segment differently; the implementer must CONFIRM the optimizer arithmetic lands in a
plan whose only uncovered op is `op:lt` (not merged with the RNG/mask `triu`/`rand` plan,
which would strand it uncovered). The op-set and the `op:lt` gap are regime-invariant; the
plan BOUNDARIES are not.

### 5.2 Blocker 2 — oversized `[Σ size]` pack → chunked: CLASS-SPLITTING, not chunked donation

**Ruling: split each isomorphism class whose `Σ size` exceeds `maxStorageBufferBindingSize`
(128 MB) into size-capped sub-classes at the packer; each sub-class cat is a normal
(non-chunked) fused elementwise segment on which the ALREADY-LANDED single-dispatch donation
edge fires. Chunked in-kernel donation is ruled OUT — do NOT modify the chunked-elementwise
aliasing surface (the §7.2 corruption class).**

Rationale, weighing the two options the task named:

- **Chunked in-kernel donation** would bind the donated input as the chunked kernel's
  read-write `out`. Even though the packed elementwise chunk windows are *aligned and
  disjoint* (`planChunkedBinary` windows input-region-i and output-region-i to the same
  element range, so `out===in` is per-chunk in-place elementwise with no cross-chunk read —
  arguably safe), it is a NET-NEW aliasing mechanism on the highest-risk surface in the
  codebase (the §7.2 "later step's data leaks into earlier results" class). It buys nothing
  chunking's dispatch count doesn't already cost.
- **Class-splitting** needs NO new aliasing mechanism. The packer's §3 isomorphism grouping
  (chain-packing) ALREADY partitions params into classes keyed by
  `(program, wdMode, dtype, lr-identity, t-identity)`; adding a `Σsize ≤ 128 MB` cap that
  splits an over-cap class into sequential sub-classes is a trivial extension of the existing
  `sig`/group-by. Each sub-class `cat` is then `< 128 MB` → the normal fused elementwise
  path → the landed `generateFused` donation edge fires unchanged. The dispatch count is
  comparable to chunking's (distil ~3 sub-classes, medium ~11 — the same order as the chunk
  loop), so no aliasing risk is traded for a submit blow-up.

**Independent premium dividend (hypothesis to MEASURE at P-CLASS-SPLIT, §5.4):** processing
sub-classes SEQUENTIALLY bounds the transient working set (`G` + `pNew` for the *current*
sub-class, ≤128 MB each) instead of the whole model's (`1.4 GB` each at medium). If the
planner frees each sub-class's `G`/`pNew` before the next (real graph nodes with real
liveness — it should), class-splitting *alone* shrinks the medium premium from +45.6 %
(+7.4 GB) toward the persistent-state floor, BEFORE donation. Donation then closes the
residual per-sub-class co-liveness. They COMPOSE; the design does not claim class-splitting
*replaces* donation (the covenant charges donation to P5), only that it removes the chunked
blocker AND likely reduces the premium the exit gate must clear. **This dividend is
UNMEASURED (reasoned from liveness); it is a gated hypothesis, not a promise.**

**Submit-budget guard:** if medium's sub-class count threatens the 18-submit budget, tune the
cap upward (params pack into as few ≤128 MB sub-classes as fit) or accept a small documented
increase — decided against the A100 measurement at flip, never the V100 census.

### 5.3 Blocker 3 — the `materialized`-vs-`pending` ref-kind mismatch: single-source from planner liveness

**Root cause (measured, corrects §2.5's framing for the build-from-IR default):**
`cachedDonatableIds` is populated at **exactly one site** — `executor.ts:2724`, inside the
LOWERED imperative segment-execution loop, as a side effect of running `executeFusedSegment`
with liveness (`livenessLastAction.get(nid) === actionIndex && !livenessOutputIds.has(nid)`).
Under **build-from-IR** (the stage-2 default: the compiled plan is built from IR metadata
stubs with NO lowered execution), that loop never runs for a recurring template → the capture
is absent at generation → `generateFused`'s gate (`stream-generate.ts:4402`,
`action.cachedDonatableIds && .size > 0`) is false → donation never fires. §2.5's
"post-execution refs resolve as `materialized` not `pending`" is the recording-path symptom
of the same seam: the donation liveness proof is being OBSERVED from a live execution rather
than DERIVED from the IR — the exact anti-pattern stage-4 phase 3/4 eliminated for every
OTHER capture (`cachedMatmulPlan`, `cachedExternalInputPattern`, `deriveResultMeta`).

**The fix (single-source-at-the-seam):** derive `cachedDonatableIds` from the plan's IR
liveness in `populateCapturesFromIR` (the build-from-IR capture pass), computed from the SAME
`livenessLastAction` the memory planner already builds — not from a lowered dispatch. The
donatable set for a fused action is `{ nid : lastAction(nid) === thisAction ∧ nid ∉
livenessOutputIds ∧ nid ∉ resultSlots }`, a pure function of the plan graph + liveness. Assert
agreement at the seam: on any path where the lowered loop DOES also run (first execution /
eager fallback), the IR-derived set must EQUAL the observed set (a `[donatable-derive]`
differential, mirroring `[ir-derive]`), falling back loudly on mismatch. This makes the
capture regime-independent and kills the mismatch by construction.

### 5.4 The exit-gate measurement regime (a consequence of 5.1–5.3)

Because donation fires only on the compiled optimizer plan, the exit-gate premium MUST be
measured in the whole-step-remat regime (adapt `tools/profile-training.ts` to wrap its step
in `api.wholeStep`, matching `webgpu-gpt2-trainer.ts`), NOT the eager profiler §2.4 used.
Measuring donation in the eager regime will (correctly) show 0 % — that is not a donation
failure, it is the wrong substrate. The exit gate re-measures fresh on **A100 (dw-2-1)** at
flip.

---

## 6. Phase plan (each shippable + gated)

**Standing gate set (every phase, before landing):** `npm run test:gates`;
`tools/parity-fullstack-tl.ts` twice (`TORCHLETTE_COMPILED_PLAN=0` vs default) ≤ 1e-5 / 30
steps; `tools/parity-packed-vs-unpacked.ts` 4 arms (Adam/Lion/SGD/Muon) bit-exact across the
compiled-plan activation threshold; `tools/packed-optim-flatness.ts` slope 0.000/step both
compiled+lowered; the donation-parity spec (`test/donation-parity.spec.ts`, bit-exact ON≡OFF
+ peak(donation) ≤ peak(no-donation)); the 124M DiLoCo regression
`{0:9.81, 3:5.92, 6:5.15, 9:4.64}` EXACT; distil **9** / medium **18** submits non-regression
(read LATE steps, confirm flat); full suite green. GPU work serial-exclusive
(`tools/pick-gpu.sh`, HOST node toolchain).

**Per-phase coverage gate:** the `TORCHLETTE_DEBUG_CENSUS` fraction for the target plan,
converged (LATE step), at the stated target.

| Phase | What lands | Coverage / premium gate | Cashes / notes |
|---|---|---|---|
| **P0** | THIS design doc. | — | No `src/` change. |
| **C1 — comparison-op serializer (blocker 1b).** | `generateSequential` gains `lt/le/gt/ge/eq/ne` (declared-elementwise, single-output, no capture). | Optimizer plan census **497/499 → 499/499 `fullyCovered=true`** (converged, whole-step regime). Standing set green. | Small net-new generator (a declared family, not a one-off). Confirm the plan-segmentation caveat (§5.1). |
| **C2 — donatableIds IR-derivation (blocker 3).** | `populateCapturesFromIR` derives `cachedDonatableIds` from `livenessLastAction`; `[donatable-derive]` differential asserts equality with the observed set where both run. | With `PLANNER_DONATION=1`: donation FIRES on the covered optimizer plan (`[planner-donation] DONATED` on ≥1 fused segment); ON≡OFF bit-exact; strict-`[lifetime]` zero throws. | Single-source-at-the-seam; the observed capture becomes a cross-check, then retires (blocker-3 deletion, §7). |
| **C3 — packer class-splitting (blocker 2).** | The packer (chain-packing §3 grouping) splits any class with `Σsize > maxStorageBufferBindingSize` into sequential ≤128 MB sub-classes; each sub-class `cat` is a non-chunked fused segment. | No `chunked-*` uncovered on the optimizer plan; donation fires per sub-class; **submit budget distil-9/medium-18 HOLDS** (measure at flip); MEASURE the independent transient-shrink dividend (§5.2) on A100. | Removes the oversized→chunked blocker WITHOUT a new aliasing surface. Composes with the packer. |
| **C-EXIT — the donation premium re-measure (the flip evidence for donation P3).** | Adapt the profiler to a whole-step-wrapped mode (tool change); no default flip of `PLANNER_DONATION` yet. | **medium@512 A100, whole-step regime, `PLANNER_DONATION=1`: packed peak premium reduced from +45.6 % toward the achievable target (§6.1), FLAT (`Storages/step +0.0`).** 124M `{…}` EXACT; `parity-fullstack-tl.ts` twice. | This is the number donation **P3** flips on. Meeting it satisfies `chain-packing-design.md` **P4**'s memory precondition → chain-packing P4 proceeds → **P5 deletes the fused `adamStep` monolith (~1.3–1.6k SLOC)**. |

**What flips donation P3:** C-EXIT meeting the premium target in the compiling regime.
**What P5 then needs from coverage:** nothing further — once the packed default holds
distil-9/medium-18 submits at flat memory within the target of fused (chain-packing P4's
gate), coverage's job is done.

### 6.1 The achievable exit-gate target (from the census + mechanism, to VALIDATE at flip)

The census fixes the mechanism, not the A100 number (unmeasured here — V100 box). Reasoned
target for medium@512 peak premium, from +45.6 % (+7.4 GB) baseline:

- **Class-splitting (C3) bounds the transient** `G`+`pNew` to one live ≤128 MB sub-class
  instead of the model-width `1.4 GB` each — a large mechanical cut IF the planner frees
  per-sub-class (hypothesis, §5.2).
- **Donation (C-EXIT) closes the residual** per-sub-class `pNew`→`P` / `G`-in-place
  co-liveness.
- The **irreducible floor** is the permanently-packed `m`/`v` state — but the fused path
  holds the SAME m/v as per-param buffers, so it is not premium.

**Target: medium@512 packed peak within ≤ +15 % of fused (from +45.6 %), FLAT**, with distil's
+2.1 % as the lower bound the mechanism approaches. This is the gate to VALIDATE on A100 at
flip; if class-splitting's transient-shrink dividend lands as reasoned, the achievable number
is materially better than +15 %. **State it as a target to hit, not a measured result.**

---

## 7. Deletion ledger + covenant

Coverage is **net-additive but small**, and it converts an observed capture into a derived
one (a subsumption):

| Item | Fate |
|---|---|
| comparison-op serializer in `generateSequential` (C1) | **NEW** (small; a declared elementwise family) |
| `cachedDonatableIds` IR-derivation in `populateCapturesFromIR` (C2) | **NEW** (small; reuses planner liveness) |
| the LOWERED-loop `cachedDonatableIds` observation (`executor.ts:2715-2725`) | **SUBSUMED** → becomes a cross-check, then retires once the IR-derivation is proven equal (blocker-3 seam collapses to one source) |
| packer size-cap sub-class split (C3) | **NEW** (small; extends the chain-packing group-by) |
| chunked-elementwise in-kernel donation | **NOT BUILT** (ruled OUT, §5.2) — a deletion of scope, not code |
| `test/donation-parity.spec.ts` cells for the packed optimizer | **NEW** (test, free per the complexity budget) |

**Campaign-level covenant.** Coverage's own src delta is small and net-additive; it is
chargeable to the chain-packing campaign as the P4→P5 precondition. Its payoff is
chain-packing **P5's ~1.3–1.6k SLOC deletion** of the fused `adamStep` monolith (`fused.ts`
adamStep family, `adam-skeleton.ts`, `packed-dispatch.ts`, the `adam-batch` action kind,
`horizontalPackKey`'s adamStep special-case, the plan-builder/graph-rewrites/fusion-detect
string reads). Coverage + donation (both small/net-neutral) plus that deletion make the
COMBINED campaign strongly **net-negative src SLOC**. The one genuinely new mechanism that
survives (the comparison serializer + the derived donatableIds capture) is warranted: it is
what lets the ratified donation edge reach the packed buffers it was designed for. Every phase
names its deletions in the commit (house policy); `bash tools/weight-norm.sh --log` snapshots
at campaign end. Each new flag interaction (`TORCHLETTE_PLANNER_DONATION` toward default;
`TORCHLETTE_WHOLE_STEP` already sunset-bound) is born with the flip that ends it.

---

## 8. Risks, refusals, and prior art (what must NOT be re-walked)

### 8.1 Typed refusals (correct-and-slow, never a correctness compromise)

- **The eager-checkpoint regime stays uncovered/uncompiled/donation-free** (§5.1). This is a
  typed regime, not a gap to close — lifting it re-opens b66ead78 (§8.3). The RNG/dropout-mask
  plan (358) and zeroGrad's `data-source:full` stay uncovered (no donation candidates, §2.3).
- **Comparison-op serializer scope:** ship `lt/le/gt/ge/eq/ne` as the declared family; a
  non-elementwise or capture-bearing op that shows up later is a fresh loud census entry, not
  a silent guess.
- **Class-splitting refuses to merge across isomorphism classes** (chain-packing §6.2): the
  size-cap splits WITHIN a class; it never packs mixed dtype/wd/lr into one sub-class.

### 8.2 Do NOT re-walk (from the ledgers)

- **Chunked-elementwise in-kernel donation** — ruled OUT (§5.2). Do not modify the chunked
  aliasing surface; class-split instead.
- **Naive `canRecycle`-reuse NaN'd the 124M** (`MEMORY.md` arena-memory-blowup-124m). Donation
  stays liveness-PROVEN + op-DECLARED + overlap-AUDITED (donation doc §4); coverage does not
  loosen it.
- **Flushing `pendingRelease` mid-step** (~2 % loss drift) / **immediate `buf.destroy()`
  mid-encoder** (poisoned submit) — untouched; coverage changes generators and the packer, not
  pool/destroy cadence.
- **The recorded build is DELETED** (stage-4 §task-#43, D4 attempt #13). Uncovered plans run
  LOWERED forever (not recorded). This campaign does NOT resurrect or re-delete it — it makes
  ONE plan (the optimizer) newly coverable so it compiles instead of running lowered.

### 8.3 Do ANY of the six recorded-build-deletion STOP classes re-arise here? — audited, NO

The stage-4 recorded-build deletion STOPPED six times; each class is checked against this
campaign:

- **Frozen scalars / volatile uniforms (classes 1–4)** — N/A: the packed path carries per-step
  values as graph tensors (chain-packing §2.5, the volatile-value dividend); no config to
  freeze. The comparison serializer emits ordinary elementwise dispatches with no per-step
  config.
- **"The recording pass was itself a witnessing driver" (5th) / "witnessing ≠ compilation:
  coverage is load-bearing for tape eligibility / cross-coverage materialization / arena
  reclamation" (6th)** — these were about DELETING the recorded build while coverage was
  incomplete. This campaign does the OPPOSITE (adds coverage; deletes nothing from the build
  machinery), so the finiteness/witnessing assumptions those STOPs protect are strengthened,
  not stressed. The one adjacent hazard is **b66ead78 (the D3 checkpoint-eager class)**: a
  compiled forward reclaiming activations a lowered recompute still needs. **Avoided by
  construction** — §5.1 refuses to compile the optimizer alone in an eager step and targets
  only the whole-step-remat regime, where the merged plan compiles as one unit (no partial
  mix). No STOP class re-arises.

### 8.4 Named risks

- **R1 — plan-segmentation strands `op:lt`.** If the whole-step merged force segments such
  that the optimizer arithmetic shares a plan with an uncoverable op (RNG `triu`/`rand`), the
  plan never reaches `fullyCovered`. Mitigation: C1's coverage gate is read in the WHOLE-STEP
  regime (not just the eager profiler); confirm the optimizer segment's only gap is `op:lt`
  before claiming closure (§5.1 caveat).
- **R2 — class-splitting's transient-shrink doesn't materialize.** If the planner keeps all
  sub-classes' `G`/`pNew` co-live (packs them into one slab rather than freeing sequentially),
  the premium is unchanged and donation must carry the whole load. Mitigation: MEASURE the
  per-sub-class liveness at C3 (`[planner]` slot dump); the design does not bank the dividend.
- **R3 — the exit gate is measured in the wrong regime.** Measuring donation on the eager
  profiler shows 0 % (the §2.5 trap). Mitigation: C-EXIT explicitly uses the whole-step-wrapped
  profiler (§5.4).
- **R4 — donation subsumption double-count.** The executor-side detector
  (`TORCHLETTE_DONATION`) and the planner edge both donate the same pair. Mitigation: the
  donation doc's subsumption check (`TORCHLETTE_DONATION=0`, equal-or-better peak) is a standing
  gate; the executor detector retires with donation P3, not here.

---

## 9. Genuine taste-calls for the user

1. **The regime split is the real story, not a coverage cliff.** The census shows the packed
   optimizer plan is 497/499 covered — §2.5's "cover the packed optimizer plan" is *almost
   already true*; the binding constraint is that it only COMPILES under whole-step remat, and
   §2.4's premium was measured in the eager regime where it can't. **Recommendation:** re-baseline
   the whole campaign on the whole-step-remat regime (which is how real training runs anyway)
   and treat "eager-checkpoint = lowered/no-donation" as a permanent typed regime. If you would
   rather donation also reach the eager path, that needs lifting the D3 refusal safely — a much
   larger, hazard-adjacent effort I recommend AGAINST for this campaign.

2. **Class-splitting may make donation nearly moot at the premium (§5.2 dividend).** If the
   transient-shrink hypothesis holds, class-splitting alone could bring medium's premium far
   below the +15 % target, and donation becomes a smaller residual closer than the covenant
   assumes. This does not change the covenant (donation is still charged to P5's deletion), but
   it may mean C-EXIT passes with a wide margin. **Recommendation:** land C3 and MEASURE before
   committing to how much donation must carry; do not pre-optimize donation against an unmeasured
   premium.

3. **`op:lt` is a 2-action gap blocking a whole plan (the all-or-nothing coverage cliff).** The
   cheapest possible unblock — but it means the entire premium recovery hinges on a comparison
   serializer landing cleanly. **Recommendation:** ship the comparison FAMILY (all six ops) at
   once so no sibling gap resurfaces, and gate on the converged census fraction, not a
   single-step read.

---

## 9. C-EXIT VERDICT (2026-07-21): STOP-WITH-NAMED-CLASS — the premium claim is FALSIFIED

C1–C3 all landed and gated (whole-step plan 1140/1140 `fullyCovered`; donation fires 6×/step
bit-exact; class-splitting deterministic, browser-load-bearing). The exit measurement
(dw-2-1 A100, whole-step regime, steady flat peaks):

| model | fused | packed | packed+donation | best split (donation OFF) |
|---|---|---|---|---|
| distil@512 | 3736.6 MB | +99.9% | +99.9% (no change) | +64.8% (32 MB split) |
| medium@512 | 9683.3 MB | +168.4% | **+168.4% (no change)** | +127.0% (128 MB split) |

**Named class:** the whole-step-remat regime — the substrate coverage/donation require to
reach the compiled packed plan — co-materializes the entire step's working set in one plan,
so the packed premium there is +100%/+168% (worse than the eager-regime +2.1%/+45.6% §2.5
measured). Donation contributes ~0 to peak: it aliases pNew→P, but peak is set by co-live
activations + the persistent packed m/v, neither of which donation touches
(`TORCHLETTE_DONATION=0` gives the same peak — the executor-side detector was never the
closer either). Class-splitting alone shrinks the transient working set (the §5.2 bonus,
confirmed: distil +99.9→+64.8, medium +168→+127) but plateaus far above the ≤+15% target.

**Consequences:**
- Donation **P3 does NOT flip** — there is no memory payoff to cash. The
  `TORCHLETTE_PLANNER_DONATION` edge remains opt-in, correct, and inert-by-value; its
  sunset now reads: delete unless a future campaign finds a consumer with a real payoff.
- `chain-packing-design.md` **P4/P5 are STOPPED** on this class. The fused `adamStep`
  monolith keeps its live consumer (the memory budget); the packed path remains the
  derived reference + the Lion/SGD/browser default. Same shape as the Everest P4b ledger
  STOP: the summit deletion is blocked by a measured substrate fact, not by taste.
- What stands regardless: full whole-step coverage (C1), the single-sourced donation
  liveness proof + `[donatable-derive]` seam (C2), and the class-split machinery (C3),
  which is load-bearing in browsers (128 MiB binding cap) where real models would
  otherwise chunk.
