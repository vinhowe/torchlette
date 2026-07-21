# Chain Packing: the parallel-isomorphic-chain packer

> **Status:** DESIGN (no `src/` change). This is the P0 design deliverable of the
> "diamond hardening" campaign. It cashes the named blocker left by the
> semantic-derivation era: `docs/semantic-derivation-design.md` §16 / §16b
> ("dissolution awaits a graph-altitude parallel-isomorphic-chain packer") and
> `docs/architecture-debt.md` migration stage 3 ("delete the `adamStep` mega-op;
> optimizers-as-graph by default").

---

## 1. The claim (one sentence)

**An optimizer step is N per-parameter chains of ONE program; the realizer emits
each isomorphism-class as a single flat-packed chain of ordinary graph ops —
`reshape → cat → evalOptTerm → narrow → copy_` — so fusion, the memory planner,
and compiled replay pack and schedule it with zero optimizer-specific machinery,
and the hand-written `adamStep` kernel dissolves.**

The corollary that makes it a *packer* and not just "run the program per param":
the arithmetic is interpreted **once over the concatenated `[Σ sizeᵢ]` buffer**, so
one class of N params costs one fused chain (a handful of dispatches), not N.

---

## 2. Altitude ruling — DECLARATION at the realizer, not RECOGNITION in the graph

### 2.1 The ruling

The packer runs at the **realizer / frontend altitude**: the optimizer's `step()`
groups its params by isomorphism class (§3) and, per class, emits ONE packed chain
of ordinary runtime ops driven by the class's `OptimizerProgram`. It is **not** a
post-hoc graph-rewrite pass that recognizes N already-emitted parallel chains and
merges them.

This is a deliberate answer to the §16b framing, which imagined notion (b) as *"the
batcher **recognizes** N parallel isomorphic compositions over disjoint param groups
and packs them."* We rule **recognition OUT** for the optimizer clients and keep the
"graph-altitude" property by a different, already-proven route: the realizer emits
**ordinary graph ops** (`cat`/`narrow`/`copy_` + the interpreted elementwise chain),
and the existing graph-altitude machinery — vertical fusion, the memory planner,
compiled replay — consumes them unchanged. The packing *decision* is declared from
`OptimizerProgram` data; the packing *execution* is ordinary graph nodes.

### 2.2 Why declaration, with evidence from the code paths

1. **The mechanism already exists and is proven.** `Adam._foreachGroupStep`
   (`src/optim/adam.ts:343`) already packs a param group by `reshape`→`runtime.cat`
   into one `[total]` flat tensor, interprets `ADAMW_PROGRAM` over it once, and
   `narrow`+`copy_`s the segments back (`adam.ts:375-482`). The stage-2 ledger
   (`docs/architecture-debt.md` row 2) measured it **bit-exact vs ground truth**,
   optimizer cleanup **355 → 10 ms** (≈ the 11–13 ms hand-fused kernel), memory dead
   flat, and — critically — *"pure graph ops, so fusion, the compiled plan, and the
   scalar table all apply with zero optimizer-specific machinery."* The packer is
   this path generalized off `Adam`, not a new mechanism.

2. **Recognition is strictly more mechanism for zero more capability.** A recognizer
   would run *after* the optimizer emitted N separate per-param chains, then match
   subgraph-isomorphism across them and synthesize the very `cat`/`narrow` the
   realizer could have emitted directly. That is: declaration + a net-new
   subgraph-isomorphism matcher (which `docs/semantic-derivation-design.md:1060-1062`
   confirms *"does not exist … no `parallel-instance`/`isomorph` packer anywhere in
   `src/compiler`, `src/executor`"*). The complexity budget (`CLAUDE.md`,
   admission-pressure rule) refuses net-new mechanism that duplicates structure the
   realizer already holds.

3. **Recognition is exactly the path the STOP feared.** §16b's mechanism-STOP is:
   *"dissolution replaces the ~8 packed `adamStepBatch` submits with N×(chain-length)
   unpacked dispatches → a submit blow-up + a 124M memory/step regression."* That
   failure mode is precisely "emit N chains, then hope the graph merges them."
   Declaration never emits the N chains: it emits one packed chain per class up front,
   so the submit budget is met **by construction** (the foreach measurement:
   distil holds 9 submits — see §6).

4. **The house philosophy is declaration-as-data + single-source-at-the-seam.** The
   optimizer already IS a program (`OptTerm`); packing keyed on that program's identity
   is one more consumer of the single source, not a second interpreter that re-reads
   the graph.

**"Graph-altitude" is honored, not dodged.** The blocker asked for a *graph-altitude*
packer because the KERNEL-altitude precedent (`deriveHorizontalPackedAdam`, schedule
state) only packs an already-flat buffer. Our packer produces the flat buffer *as
graph nodes* — that IS the graph-altitude analogue, and it is where the flat layout
must live so the planner and compiled replay can own it.

### 2.3 The one client that DOES need recognition (scoped out — §7, phase P6)

Bias-grad sum batching (perf target #3) is the exception that proves the ruling: the
244 same-dim `sum`s in full-FT backward have **no shared authoring site** — each falls
out of an independent `Linear`/bias VJP in autograd, so there is no realizer to declare
them together. That client genuinely needs a *recognition* pass over the backward
graph, AND it is a reduction (needs tile-IR multi-output reduction), not an elementwise
chain. It shares the packer's **isomorphism vocabulary** (§3) but not its mechanism or
altitude. It is a separate campaign; §7 scopes it.

### 2.4 Interaction with the machinery it sits above

Because the pack emits ordinary graph ops, each interaction is *inherited*, not
re-engineered. Named explicitly so the implementer confirms rather than assumes:

- **Vertical fusion** (`src/compiler/fusion-detect.ts`, `segmentPlanForExecution`,
  priority-40 in `analyzeGraph`, `graph-compiler.ts:544`): the interpreted
  `add(mul(m,β1),…)` chain over the flat `[total]` buffer is a vertical elementwise
  run — the detector collapses it to a small number of fused segments exactly as it
  does today for the foreach path. No packer hook.

- **Memory planner / `PlannerRegistry`** (`src/executor/memory-planner.ts:153`,
  `planMemory`): the packed `cat` result, the `evalOptTerm` intermediates, and `bc1/bc2`
  are ordinary **temp** slots — shareable cross-plan by the planner's temp contract
  (`memory-planner.ts:11-21`). The permanently-packed `st.m`/`st.v` and the params are
  **result/persistent** slots (registered via `runtime.registerState`, `adam.ts:412`).
  The packer introduces no new slot category; it must only present correct
  `[alloc,lastUse]` lifetimes (automatic — they are real graph nodes). The **memory
  regression the STOP named is a planner concern, already solved**: foreach's ~30
  full-model intermediates cost 20.3 GB under the legacy unbudgeted arena but 2.5 GB
  current under `ARENA_LIVENESS` + the planner (`architecture-debt.md` stage-3 row).
  The residual ~2× premium (foreach's own packed G/P/pNew working set) is closed by
  **graph-level buffer donation** (write `pNew` into `P`'s buffer) — stage-4 work,
  the P4 precondition (§4).

- **Compiled-plan record→replay** (`src/executor/stream-generate.ts` →
  `buildCompiledPlanFromGenerated`, `compiled-plan.ts:334`; the current default is
  build-from-IR on first execution, no separate recording): the packed chain is
  generated and replayed like any graph. **The volatile-uniform discipline is
  satisfied by ELIMINATION, not by a new hook** — see §2.5. This is a correctness
  *dividend* of dissolution.

- **Implied step boundaries** (`CLAUDE.md`; `queueStepBoundary`): unchanged. The
  realizer sequences the in-place `copy_` state effects and calls
  `api.queueStepBoundary()` at the end of `step()` (already done by `_stepForeach`,
  `lion.ts:204`, `muon.ts:248`). The two rules — wrapper-level generation stamps and
  quiesce-before-destroy — are properties of the executor's boundary commit, not of the
  packer; the packer emits nothing that touches them (it creates no storage-level
  stamps; `registerState` is the durable-persist path the boundary already honors).

- **The existing adam-batch hoisting — already generic, superseded here.**
  `ADAM_HOISTABLE_OPS` is **already deleted** (commit b791d72); the hoisting became a
  generic same-op affinity tie-break in `enforceWriteAfterReadOrder`
  (`plan-builder.ts:240-285`, predicate `!isFusibleOp(op)`). That tie-break clusters
  *non-fusible sequential dispatches* (the `adamStep` nodes) so `adam-batch` can pack a
  consecutive run. Under dissolution **there are no `adamStep` nodes** — the optimizer
  emits fusible elementwise ops that the affinity tie-break simply lets fuse. So the
  packer doesn't fight the tie-break; it makes the `adamStep`-shaped clustering moot.
  The deletion the packer *cashes* here is the `adam-batch` **action kind** and
  `horizontalPackKey`'s `adamStep` special-case, not the (already-generic) hoisting
  (§5).

### 2.5 The volatile-value dividend (why dissolution is a correctness WIN, not a risk)

The frozen-`step_size` bug class (`CLAUDE.md` "Compiled-plan replay vs per-step host
values") exists because the fused kernel carries per-step values in a config the
compiled replay can bake. The packed-graph path **cannot** hit it: every per-step
value is already a graph tensor.

- `bc1`/`bc2` (bias correction, the step-as-DATA) are computed by `_biasCorrection`
  (`adam.ts:631`) as graph tensors from the persistent on-device `t`; `t` advances
  in-plan via `copy_(t, add(t,1))` (`adam.ts:612`), so TAG_WRITE re-executes the source
  every replay.
- `lr` is a persistent per-group tensor written in place by the scheduler (`adam.ts`
  `_lrTensor`/`setLR` copy_ path), delivered as tensor DATA.
- `inv_scale` never enters the packed path — GradScaler unscales via graph-level
  `unscaleGrad` nodes (`adam.ts:504-505`).
- constant hypers (β1, ε, …) fold in JS inside `evalOptTerm` (`optimizer.ts:282-294`)
  and are honored by the per-template **scalar table** (`src/executor/scalar-table.ts`,
  stage-1) which refreshes 4-byte buffers before every execution.

So the packer requires **no `setAdamConfigUniforms`, no TAG_UNIFORM volatile repack**.
The single-source seam the fused path needs (`adam-kernel.ts:85`) disappears with the
kernel. **Gate obligation:** the packed-vs-unpacked trajectory differential (§4)
crossing the compiled-plan activation threshold (Corollary 2, `CLAUDE.md`) is what
*proves* the elimination — it must be run twice (`TORCHLETTE_COMPILED_PLAN` off vs
default), 1e-5/30 steps, per the frozen-step_size lesson.

---

## 3. The isomorphism definition (the core)

Two per-parameter chains are **isomorphic** — hence packable into one flat dispatch —
iff **all four** hold:

1. **Same program term.** Identical `OptimizerProgram` (the `OptTerm` tree for every
   `stateUpdate` and the `paramUpdate`), by structural term identity. Same op sequence,
   same tree shape, same constant placement. (In practice: same optimizer, same
   weight-decay MODE — L2-folds-into-`g` vs decoupled-in-param-term is a *different*
   emitted term, so it is a different class. This is why `_foreachGroupStep` branches
   on `this.adamW`/`wd` before interpreting: the branch selects the term, and the term
   selects the class.)

2. **Same shared-tensor bindings for the per-step DATA roles.** The roles that are
   per-step tensors bound ONCE per dispatch — `lr`, `t`→`bc1`/`bc2` — must be the SAME
   tensor object across the group. This is the direct generalization of
   `horizontalPackKey`'s `sharedOperandInputIndices` (`lowered-plan.ts:534`, today
   hard-coded `[4,5]` for `adamStep`). Params in different LR groups bind different `lr`
   tensors → different class. (Per-param scalar hypers that fold to JS constants must be
   equal by value, e.g. a per-param `weightDecay` — those break the class if they
   differ, matching the fused batcher's static-config agreement.)

3. **Same dtype.** All params in a class share element dtype (f32 for state; the pack is
   over `[total]` f32 flats). Mixed dtype → separate class.

4. **Elementwise-flattenable term.** Every node in the term is shape-agnostic over the
   flat `[total]` layout (all of `add/sub/mul/div/sqrt/sign/abs/neg/exp` are). A term
   containing a `mm` contraction node is NOT flattenable (a matmul needs 2-D structure;
   a concatenation of differently-shaped matrices is not one matmul) — see the Muon
   ruling (§6). This clause is what partitions Muon.

**Shapes MAY differ per param — handled by concatenated flat layout.** This is the
mechanism, stated precisely so it cannot be misread:

- Each param `i` in a class is reshaped to a 1-D `[sizeᵢ]` (`sizeᵢ = Π shapeᵢ`) and the
  group is `runtime.cat`'d into one `[total]`, `total = Σ sizeᵢ` buffer — one for `g`,
  one for `p`; the state slots `m`/`v` are packed **once at first step** and updated in
  place (`copy_`) thereafter so their buffers are stable across steps
  (`adam.ts:385-418`).
- The per-chain boundaries are **narrow offsets**: `offsetₖ = Σ_{j<k} sizeⱼ`. The update
  is interpreted ONCE over `[total]`; unpack is
  `reshape(narrow(pNew, 0, offsetₖ, sizeₖ), shapeₖ) → copy_(paramₖ, ·)`
  (`adam.ts:476-482`). There is **no per-chain uniform table of offsets** — the offsets
  live only in the host-side `narrow` args, which become ordinary view nodes; the packed
  kernel sees one contiguous `[total]` and knows nothing about param boundaries. (This is
  simpler and strictly better than a uniform offset table: the elementwise program is
  position-independent, so boundaries need not reach the GPU at all.)
- The class **signature** `sig = "idx:size,idx:size,…"` (`adam.ts:357`) pins the group's
  membership+layout across steps; a change throws (grad set changed → repack), because
  the permanently-packed state cannot be silently remapped.

**The isomorphism KEY** (the packer's group-by function) is therefore:
`key(param) = (programIdentity, wdMode, dtype, sharedTensorIdentity(lr), sharedTensorIdentity(t))`.
This subsumes and generalizes `horizontalPackKey` — the same seam, keyed on program
structure instead of the op name `"adamStep"`.

---

## 4. Phase plan (each shippable, differential-gated)

Every phase runs the **standing gate set** before landing: `npm run test:gates`;
`tools/parity-fullstack-tl.ts` compiled-vs-lowered (twice: `TORCHLETTE_COMPILED_PLAN=0`
vs default) agreeing ≤1e-5 over 30 steps; the 124M DiLoCo regression
{0:9.81, 3:5.92, 6:5.15, 9:4.64} EXACT; distil 9 submits / medium 18 submits
non-regression (`TORCHLETTE_PROFILE=1 … tools/profile-training.ts`, read LATE steps).
Phase-specific gates are listed per row. GPU work is serial-exclusive (reserve via
`tools/pick-gpu.sh`, HOST node toolchain).

**The NEW differential (introduced in P1, extended per phase):**
`tools/parity-packed-vs-unpacked.ts` — the same optimizer over the same toy trajectory,
packed path vs per-param path, ≤1e-6 over 20 steps, for **Adam AND Lion AND Muon**. This
is the Corollary-1 cross-path guard the packer is required to ship with. It must cross
the compiled-plan activation threshold (Corollary 2).

| Phase | What lands | Deletes / cashes | Phase-specific gates |
|---|---|---|---|
| **P1** | **Extract the packer as a shared program-driven primitive.** Lift `Adam._foreachGroupStep`'s pack loop into `packOptimizerProgram(program, items, effects)` in `src/optim/` (or `src/ops/semantic/`): takes an `OptimizerProgram` + per-param `{param, grad, state[], hyperRoles}` items pre-grouped by the §3 key, does `cat → evalOptTerm → narrow → copy_`, sequences the declared `copy_`/`registerState`/`dispose` effects. Adam's foreach path calls it. **No default flip.** Behind existing `TORCHLETTE_FOREACH_ADAM`. | Nets ~0 SLOC (moves code out of `adam.ts`); establishes the single packing seam. No deletions yet. | `fused-vs-elementwise.spec.ts` (fused==derived-elementwise==derived-foreach, 12 cells) still green; new `parity-packed-vs-unpacked.ts` (Adam) added and green. |
| **P2** | **Wire Lion + SGD(+momentum) to `packOptimizerProgram`.** They currently emit per-param chains (`lion.ts:149-206`, unfused across params). Replace with a grouped call; zero optimizer-specific code in the packer (it reads `LION_PROGRAM`/`SGD_*_PROGRAM`). | Cashes the per-param chain loops in `lion.ts`/`sgd.ts` (the realizer becomes group-by + one packer call). | `parity-packed-vs-unpacked.ts` extended to Lion + SGD; `test/lion-distil-descent.spec.ts` (20-step monotone descent) green; a Lion training run's submits measured (must not exceed the Adam-class submit budget for the same param count). |
| **P3** | **✅ LANDED — Muon full-refusal v1** (§6.1, the sanctioned clean refusal). Muon declares `MUON_PROGRAM` to the packer through the same seam as Adam/Lion/SGD (`Muon.packVerdict()` → the packer's clause-4 gate `assertFlattenable`); because the program carries `mm` (Newton–Schulz) the packer refuses with a typed, named `OptimizerPackRefusal`, and Muon realizes its step per-param (correct-and-slow, byte-identical to pre-routing). The verdict is a **once-per-class** decision (function of the program alone), cached so it never re-throws per step. The elementwise-partial pack was **explicitly ruled OUT for v1** — see the future-work note below. | Near-net-zero SLOC (the routing seam + cached verdict on `muon.ts`); no deletions this phase. | `parity-packed-vs-unpacked.ts` extended with a Muon arm (packed-attempt = refused → per-param **vs** per-param, trivially bit-exact — the standing gate cell); `test/optim/muon-pack-refusal.spec.ts` (the standing spec of the HARD assertion: Muon → typed/named refusal; Adam/Lion/SGD accept — no collateral refusal; verdict cached & side-effect-free); the Muon DistilGPT-2 descent (`test/muon-distil-descent.spec.ts`, 7.89→0.35) reproduced unchanged. |
| **P4** | **THE DEFAULT FLIP.** Make `packOptimizerProgram` Adam's default on WebGPU (route `step()` to the packed path when `params.length > 1`, ahead of `_stepFused`). **Precondition: graph-level buffer donation** (`pNew`→`P`'s buffer, `G` packed in place) landed so the ~2× foreach working-set premium closes and 124M memory + submit budgets hold. The fused kernel stays in-tree, unreferenced, as the assertion oracle for this phase only. | Cashes the `TORCHLETTE_FUSED_ADAM` default; `_stepFused` becomes opt-in. | **The hard gates:** 124M {…} EXACT; distil 9 / medium 18 submits EXACT; `parity-fullstack-tl.ts` twice; **fused-vs-packed trajectory** as the final assertion BEFORE P5 deletes fused. Re-measure the A100 baselines fresh at flip time (do NOT trust the historical foreach numbers as the flip evidence). |
| **P5** | **THE DELETION.** Remove the fused `adamStep` node and everything downstream of it (the §5 ledger). | Net-negative SLOC (§5). | Full suite green post-deletion; `bash tools/weight-norm.sh --log` shows net-negative src vector; all standing gates green (the fused oracle is gone, so `fused-vs-elementwise.spec.ts` retires — its packed-vs-per-param successor from P1–P3 is the surviving guard). |
| **P6 (separate campaign — scoped out here)** | **Bias-grad sum recognition pass** (§7). A recognition pass over the backward graph that detects N parallel isomorphic same-dim reductions (reusing the §3 isomorphism vocabulary) + tile-IR multi-output reduction. Different altitude (recognition, not declaration), different op class (reduction). | Would cash 244→~few sum dispatches (22% of full-FT GPU time). Needs net-new tile-IR mechanism; not a deletion. | Own campaign; own gate (packed-vs-unpacked reduction differential + the full-FT submit/GPU-time measurement). |

Sequencing note: **P4 depends on stage-4 buffer donation**, exactly as
`architecture-debt.md` stage-3 concluded ("stage 4 … is the PREREQUISITE for stage 3's
default flip"). P1–P3 are independent of it (they are behind the opt-out flag, memory-
bounded by the planner already). If donation is not ready, P1–P3 still ship and Lion/Muon
still get packed execution under the flag.

---

## 5. The deletion ledger

Cashed at **P5** (the flip's payoff), sized from the §16b mechanical map
(`semantic-derivation-design.md:1005-1031`) verified live:

| Target | Location | Why it dies |
|---|---|---|
| `adamStep` fused kernel + WGSL skeleton | `src/schedule/adam-skeleton.ts` (~660 lines: `lowerAdamStepBody`, `emitExpm1`, `emitBiasCorrection`), `src/backend/webgpu/adam-kernel.ts` (`setAdamConfigUniforms`, `realizeAdamStepSpec`) | the arithmetic is now the interpreted graph chain |
| `adamStep`/`adamStepBatch`/`adamStepInner` | `src/backend/webgpu/ops/fused.ts:227-384` | no fused node to dispatch |
| **`packed-dispatch.ts`** (backend DMA scatter/gather) | `src/optim/packed-dispatch.ts` (262 lines) | superseded by graph-level `cat`/`narrow` — the pack is now graph nodes, not a backend copy loop. (Only Adam's fused path calls it; nothing else survives it.) |
| `executeAdamStep` + registration | `src/executor/op-dispatch.ts:613`, `:786` | no `adamStep` op |
| **`adam-batch` action kind** + assembly + execution | `src/executor/lowered-plan.ts:193,438,800-838`, `src/executor/executor.ts:2876-2904` | the packed chain is fused segments, not a bespoke action |
| `horizontalPackKey` `adamStep` special-case | `src/executor/lowered-plan.ts:534` (`sharedOperandInputIndices`) | no `adamStep` run to key; the §3 isomorphism key lives in the realizer |
| `IN_PLACE_DST_INPUTS.adamStep = [1,2,3]` | `src/executor/plan-builder.ts:108` | the in-place `copy_` destinations are ordinary `copy_` nodes with their own WAR edges |
| `NON_CSE_OPS ∋ "adamStep"` | `src/compiler/graph-rewrites.ts:238` | no side-effecting mega-op to exempt |
| `PAYLOAD_HASH_EXEMPT.adamStep = "ALL"` | `src/compiler/fusion-detect.ts:1752` | no config payload to exempt |
| `AdamBatchItem`/`AdamBatchResult`/`adamStepBatch?` interface | `src/backend/types.ts:303,316,548` | backend seam gone |
| `AdamStepConfig` threading, `_stepFused` | `src/optim/adam.ts:507-598` | opt-in at P4, deleted at P5 |
| `TORCHLETTE_FUSED_ADAM`, `TORCHLETTE_PACKED_ADAM` flags | env accessors | born-with-sunset flags whose campaign ends |

Rough size: `adam-skeleton.ts` (~660) + `packed-dispatch.ts` (262) + the `adamStep`
portions of `fused.ts` (~160) + the batching seams (~250 across lowered-plan/executor/
op-dispatch/plan-builder/graph-rewrites/fusion-detect/types) ≈ **1,300–1,600 SLOC
deleted**, against P1–P3's realizer refactor (roughly net-zero — code moves from
`adam.ts`/`lion.ts`/`muon.ts` into one `packOptimizerProgram`) plus the donation
mechanism P4 needs (stage-4, chargeable there). **Covenant:** the campaign is
**net-negative src SLOC** on the strength of the P5 deletion; the one net-new mechanism
(the shared packer primitive) is warranted because it is the single seam that lets THREE
clients (Adam, Lion, Muon-partial) share packed execution with zero per-optimizer code —
capability-per-SLOC rises. Every phase names its deletions in the commit (house policy).

---

## 6. Named risks and refusals

### 6.1 The Muon ruling — TYPED PARTIAL

**Muon packs PARTIALLY: elementwise segments per shape-class; the `mm`/Newton-Schulz
core refuses (per-param dispatch).**

Rationale: `MUON_PROGRAM` (`optimizer.ts:550`) contains `mm` contraction nodes
(`nsStep`: `X·Xᵀ`, `A·A`, `B·X`, `optimizer.ts:506-514`). A contraction needs 2-D
structure; a `cat` of differently-shaped momentum matrices is not one matmul, so
clause-4 of §3 (elementwise-flattenable) fails for any segment containing `mm`. The
packer therefore partitions a Muon chain into: `[elementwise momentum]` → `[mm-bearing
NS core]` → `[elementwise apply]`. The two elementwise ends pack across params **that
share a shape-class** (params with identical `[R,C]` — Muon already needs same-shape for
the orientation/rms policy anyway); the NS core dispatches per-param (`rt.matmul`,
unchanged from `muon.ts:151-241`).

**Honest scope:** Muon's cost is dominated by the per-param matmuls (the whole point of
Newton–Schulz), so the elementwise-segment packing win is marginal. A v1 that **refuses
Muon packing entirely** (per-param, correct-and-slow) is acceptable and is the typed
refusal the design sanctions — P3 may ship the full partial or the clean refusal, named
either way. What is NOT acceptable is silently flat-packing an `mm` (it would produce a
wrong result, the worst failure mode). The packer must **assert** at group-build time
that no packed class contains an `mm` node (clause-4 is a hard gate, not a heuristic).

**Status — LANDED, full-refusal v1.** P3 shipped the **clean refusal**, not the partial.
`Muon.step()` calls `Muon.packVerdict()` first: it declares `MUON_PROGRAM` to the packer's
clause-4 gate (`assertFlattenable`, exported from `pack-optimizer.ts`), catches the typed
`OptimizerPackRefusal`, and **caches** it — the verdict is a function of the program alone,
so it is decided once and never re-thrown per step (the "cheap, once-per-class" requirement).
The step then realizes each 2D param through the existing per-param path (the momentum
copy_, the NS `rt.matmul` chain, the apply tail), byte-identical to the pre-routing math.
The internal AdamW sub-path for embedding/1D params is a full `Adam` instance and inherits
Adam's packer routing verbatim (it packs exactly when Adam packs) — structurally already on
the shared machinery, so there is nothing Muon-specific to wire, and it packs for free once
P4 flips Adam's default.

**What a future elementwise partial-pack (§6.1 original) would need**, if the marginal win
is ever wanted: (1) partition each Muon chain into `[elementwise momentum]` → `[mm-bearing
NS core]` → `[elementwise apply]` at the realizer, packing the two elementwise ends across
params **of the same shape-class** while the NS core stays per-param `rt.matmul`; (2) group
by shape-class (Muon already requires same-shape for the orientation/rms policy), not just by
the §3 key; (3) a per-shape-class flat pack of the momentum-update and apply tails that reads
the per-param NS output back out — i.e. a scatter/gather boundary around the un-packable core.
`packVerdict()` returning `null` is the seam that path would consume; today it is unreachable
(MUON always refuses). The parity tool's Muon arm (packed-attempt vs per-param) is the
red/green trajectory guard that partial would inherit.

### 6.2 What the packer REFUSES to pack (typed refusal → correct-and-slow fallback)

- **Any term containing a non-elementwise node** (`mm` today; any future contraction/
  reduction primitive). Refuses the offending segment to per-param dispatch. (§6.1.)
- **A group of size 1** (a param with a unique shape/class). No pack; per-param chain.
  This is the existing `dispatchPackedOptimizer` behavior (`packed-dispatch.ts:216,228`)
  and the foreach `params.length > 1` guard.
- **A membership/layout change across steps** (`sig` mismatch — grads intermittently
  missing, param set changed). Throws with the `TORCHLETTE_FOREACH_ADAM=0` escape
  (`adam.ts:395-401,362-370`), rather than silently remapping permanently-packed state
  (the UAF/corruption class). Under dissolution the escape becomes "per-param path,"
  which still exists as the reference definition.
- **Mixed dtype / mixed wd-mode / mixed LR-group** within a would-be class. Partitioned
  into separate classes by the §3 key; never merged.

### 6.3 Named risks

- **R1 — submit blow-up if a class fragments.** If params fragment into many size-1
  classes (all-unique shapes), the pack degenerates to per-param and submits regress.
  Mitigation: the flat-`cat` packs *all* params of a class regardless of shape (shapes
  MAY differ, §3), so fragmentation only happens across *classes* (dtype/wd/LR-group),
  which are few in real models. **Gate:** distil 9 / medium 18 submits EXACT at P4.
- **R2 — 124M memory regression** (the STOP's second consequence). The packer's own
  working set (`G`/`P`/`pNew`, full-model-size) is the residual ~2× premium. Mitigation:
  buffer donation (P4 precondition). **Gate:** 124M {…} EXACT; peak memory measured at
  flip.
- **R3 — a per-step value silently baked in compiled replay** (frozen-step_size class).
  Mitigation: §2.5 — the packed path carries no bakeable config; the value roles are
  graph tensors. **Gate:** `parity-packed-vs-unpacked.ts` + `parity-fullstack-tl.ts`
  BOTH across the compiled-plan activation threshold (Corollary 2).
- **R4 — persistence-contract UAF on the packed state** (the sequential-corruption
  class). Mitigation: the packed `st.m`/`st.v` are created via `runtime.registerState`
  and updated in place by `copy_` (never replace-and-hold) — the exact discipline
  `adam.ts:409-417` and `muon.ts:159-166`/`lion.ts:164-168` already encode. The packer
  MUST preserve it (no fresh state allocation per step). **Gate:** the strict-lifetime
  `[lifetime]` guard (default-throw) + the 124M soak.
- **R5 — touching a live optimizer.** Adam IS training; a broken optimizer is the
  framework's worst outcome (§16b). Mitigation: P1–P3 land behind the opt-out with the
  fused path untouched as default; the flip (P4) is gated on the fused-vs-packed
  trajectory assertion; the fused kernel is deleted (P5) only after the packed path is
  the proven default.

---

## 7. Bias-grad reduction batching — same vocabulary, different pass (scoped out)

**Are N parallel isomorphic reductions the same pass? No — same isomorphism
*definition*, different *mechanism* and *altitude*.** The honest scoping:

- **No shared authoring site.** Optimizer chains have ONE realizer per optimizer that
  knows the N params and can *declare* the pack. The 244 bias-grad `sum`s
  (`dBias = sum(gradOut, dims)`) are emitted independently by N `Linear`/bias VJPs across
  the autograd graph — there is no realizer to declare them together. Batching them
  requires a **recognition** pass over the backward graph (the one place recognition is
  actually necessary, §2.3) — the opposite altitude from the optimizer packer.
- **Different op class.** These are **reductions**, not elementwise chains. Packing them
  needs **multi-output reduction support in tile-IR** (perf target #3, `CLAUDE.md`:
  *"Would need multi-output reduction support in tile-IR to batch independent same-dim
  reductions"*), which the elementwise flat-`cat` mechanism does not provide. The islands
  I2 findings (`docs/islands-design.md:566-575`) independently place reduction batching
  at the **claim altitude**, not the elementwise detector.
- **What they share:** the §3 isomorphism *vocabulary* — "N parallel isomorphic subgraphs
  over disjoint data, packable into one dispatch." A future reduction-batching campaign
  should reuse that definition (same-op, same reduced-dim, same dtype, shapes may differ)
  but build a recognition pass + tile-IR multi-output reduction, NOT extend
  `packOptimizerProgram`.

Scoped as **P6, a separate campaign** (§4 table). Merging it into the optimizer packer
would conflate declaration with recognition and elementwise with reduction — two axes the
ruling (§2) deliberately keeps apart.

---

## 8. What was tried before (the STOP verdict, faithfully)

The dissolution was attempted twice and STOPPED both times; the record is
`docs/semantic-derivation-design.md` §16 (lines 933-943) and §16b (996-1078), commit
`990f2092`. **No `src/`/`test/` change landed** — the output was the mechanical map (the
four load-bearing sites, §5) and the named blocker.

- **§16 original STOP:** dissolving `adamStep` "is NOT safe cheaply: the node is
  load-bearing across the batching machinery — the plan-builder in-place write-sets
  (`adamStep:[1,2,3]`), the lowered-plan `adam-batch`/`adamStepBatch` action + its
  shared-input (t/lr) handling, the graph-rewrites CSE key, and the backend fused
  kernel's buffer-ownership transfer. The adam-batch grouping KEYS on the op name." The
  derived paths (foreach/elementwise) already emit the pure composition; the FUSED path
  kept its node — "a clean partial, not a stranded whole."

- **§16b — two notions of "structure key", each rejected for a distinct reason:**
  - **(a) op-metadata-driven** (keep the named node; the four sites read a metadata record
    instead of `=== "adamStep"`): **budget STOP** — "cashes NO deletion today … no second
    client (no fused Lion is built)." Held out per admission-pressure. *This design does
    NOT re-walk (a); it makes the fused node vanish, so the re-key is moot.*
  - **(b) composition-structure dissolution** (emit `OptTerm` as graph nodes; recognize N
    parallel isomorphic compositions and pack): **mechanism STOP** — "requires a
    graph-altitude horizontal-pack of parallel isomorphic chains, and that mechanism does
    not exist … no `parallel-instance`/`isomorph` packer anywhere." Attempting it *by
    recognition* would "replace the ~8 packed `adamStepBatch` submits with N×(chain-length)
    unpacked dispatches → a submit blow-up + a 124M memory/step regression → an automatic
    STOP under the submit-sensitive distil-profile gate and the 124M-EXACT gate."

- **The correction §16b made to §16:** the `ADAM_HOISTABLE_OPS` hoisting site named in
  §16 was **already gone** — generalized to the `!isFusibleOp(op)` affinity tie-break in
  `enforceWriteAfterReadOrder`. "So the batching's ADJACENCY is already structure-keyed;
  only the GROUPING key + the run predicate still read the string." (This design deletes
  those last two string reads, §5.)

**How this design clears the STOP:** it answers notion (b) but by **declaration, not
recognition** (§2), so the feared "N×chain unpacked dispatches" never materialize — the
realizer emits ONE packed chain per class up front (the submit budget met by
construction, the foreach precedent), and the memory regression is a planner+donation
concern already on the stage-4 roadmap. The unbuilt "parallel-instance packer" the STOP
awaited turns out to be a mechanism that **already exists in embryo** as
`Adam._foreachGroupStep`; the campaign is to generalize it off Adam, flip it to default,
and delete the fused kernel — not to build a subgraph-isomorphism matcher.

---

## Appendix — measured grounding (existing, authoritative)

Cited rather than re-measured (re-measure fresh at P4 per `agent-ops.md`):

- foreach == fused to **1.5e-5 / 30 steps** fp32 fullstack; optimizer cleanup **355 → 10
  ms** (fused 11–13 ms); memory dead flat (`architecture-debt.md` stage-2/stage-3 rows).
- foreach memory: **20.3 GB** legacy-arena → **2.5 GB current** under ARENA_LIVENESS +
  planner (distil@512); residual ~2× working-set premium closes with buffer donation
  (`architecture-debt.md` stage-3 row).
- distil holds **9 submits** / medium **18 submits** at bounded memory (CURRENT A100
  baseline, `CLAUDE.md`).
- The oracle: derived optimizers track `torch.optim` AdamW/Adam/SGD to **≤1e-6 / 20
  steps** (`test/oracle/optimizer-trajectory.spec.ts`); Lion trains DistilGPT-2 20 steps
  7.89→4.36 (`test/lion-distil-descent.spec.ts`).
