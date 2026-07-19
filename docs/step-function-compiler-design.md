# The Step-Function Compiler: the training step as one ahead-of-time-compiled program

*Design + stage-1 feasibility probe, 2026-07-18. Branch `step-function-compiler-design`
off `main@b0b4e7c3`. Companion to `architecture-debt.md` (the stage table — this
is the terminus of stage 4) and `stage4-compile-from-ir.md` (the build-from-IR
compiler this generalizes from a per-plan artifact to a per-step one).*

> Every measured number in §2 comes from `probe/census.ts` (in-tree, this branch),
> run on dw-2-1 (A100) against distilgpt2@seq15, 4 warmup + 6 steady steps, under
> the default engine config (fusion + memory-planning + checkpoint-segmentation).
> Reproduce: `VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim npx tsx probe/census.ts`.

---

## 0. One-sentence declaration

> **A training step is a pure function of (declared input slots ∪ parameters ∪ RNG
> state) → (declared readbacks ∪ state updates); Torchlette traces it once by
> letting laziness defer every force to the step boundary, compiles the resulting
> single whole-step graph ahead of time, and replays the compiled step under input
> guards — the eager path remaining the semantic reference with typed fall-back.**

The one-sentence test (§7): *if a feature of the step cannot be expressed as a node
in the whole-step graph or a guard on its inputs, it does not belong inside the
compiled step — it is a trace break, by declaration.*

### The contract

A compiled step `S` is defined by:

- **Inputs** = the declared input slots (batch upload buffers), the parameter set,
  the optimizer state (m/v/step), and the RNG state at step entry. These are the
  only values that may vary between two executions of `S`.
- **Outputs** = the declared readbacks (loss scalar, metrics — delivered through the
  existing ring/`startItemReadback` staging path, never a mid-step fence) and the
  in-place state updates (params, m/v, RNG advance, GradScaler scale).
- **Everything else is internal** — activations, gradients, saved-for-backward
  tensors, temporaries — and is owned by the compiled step's global memory plan. No
  internal value is observable; none may be read back without breaking the trace.

`S` is valid for a live step iff the input guards hold (§3.4). On a guard miss the
step falls back to eager execution (the semantic reference), which re-traces and
re-compiles — a Dynamo-shaped recompile, not an error.

---

## 1. The thesis, stated precisely

Torchlette is **lazy-first**: every frontend op builds a `LazyIRNode` (a pure
dataflow node — `src/graph/types.ts`); nothing executes until a **force point**
demands a concrete value. The symbolic trace we need for AOT compilation is
therefore *free*: it is just the existing lazy graph with **every force deferred to
the step boundary**. Run the step; defer every readback; and forward + backward +
optimizer accumulate as **one closed dataflow graph** with no execution in between.

This is not a new frontend. **Laziness *is* the tracer.** There is exactly one
frontend, one IR, one op semantics. The "two frontends" maintenance question that
haunts every trace-based compiler (Dynamo vs eager, `jax.jit` vs eager) does not
arise here: the traced graph is built by the *same* ops that execute eagerly, node
for node. The compiler is a set of passes over that graph, plus a decision about
*when* to force — not a second way to express a model.

The empirical machinery of the last several months exists to *approximate at
runtime the staticness a whole-step graph has by construction*:

| Runtime approximation (today) | Static equivalent (AOT) |
|---|---|
| observed-liveness / convergence stamping (`K=3` no-growth rebuild) | dataflow liveness analysis over the whole-step graph |
| witness apparatus + cross-plan edges (`K_w=2`) | inter-segment dataflow is *internal* — no boundary to witness |
| the harvest (result slots between plans) | one plan; producers and consumers are the same graph |
| the step tape (record a step, replay by `bucketKey`) | the compiled step *is* the tape, built once by the compiler |
| checkpoint unpack hooks forcing recompute subgraphs | remat is a **compiler pass** that schedules recompute inline |
| arena/pool runtime-adaptive trim/liveness/reclaim | one global memory plan, allocated once |
| the guard taxonomy's runtime half (bucketKey/staleness/regime) | reduces to **input guards** on the step's declared inputs |

Each row is a *runtime learner converging toward a static answer*. The whole-step
graph gets that answer for free from dataflow analysis. **The season's erosion of
mid-step forces — ring-deferred `loss.item()`, scale-as-tensor + inf-skip-as-data
(D0), LiveScalar LR, intra-plan seed, batched uploads-as-slots — was, in
retrospect, incrementally paying down the exact forces that stand between "two
plans per step" and "one compiled step."** This design finishes that arc.

---

## 2. FEASIBILITY VERDICT (the probe's findings — measured, up front)

**Verdict: feasible, with no genuine blocker.** The three probe questions resolve as:

1. **Backward-without-force (checkpoint off): already true, today.** Backward builds
   its entire gradient graph with **zero** intermediate forces — every backward
   function only *constructs* lazy nodes (`autograd.ts:358-360`: "the merged fwd+bwd
   plan reduces force points from 3 to 1"). There is exactly one force at the end of
   backward (`autograd.ts:433`, `forceAllMerged(...allGrads)`).

2. **Deferring `loss.item()` collapses forward + backward into ONE plan.** Measured:
   with the mid-step loss readback removed, the `forward+item` force **disappears
   entirely** and the single backward force's plan grows to **630 nodes** (= forward
   237 + backward ~393, merged in one `buildMergedPlan`). Forces/step drop 4 → 3;
   physical GPU submits drop 6 → 5. This is the thesis, demonstrated in-tree.

3. **Whole-step graph scale ≈ 727 nodes (distilgpt2), well within the passes' range.**
   Measured decomposition: forward 237 + backward 408 + optimizer 82 ≈ **727 IR
   nodes**. The largest *single* plan the compiler builds today is already 507 nodes
   / 1746 GPU commands (the backward+optimizer plan), so a 727-node whole-step graph
   is a ~1.4× step up, not a new regime. gpt2-medium (24 layers) extrapolates to
   ~3–6k nodes.

### 2.1 The force-site census

Distilgpt2, 6 steady steps, forces/step (a *force* = one `forceAllMerged` /
`forceAllPending` = one execution of pending work):

| Config | phase :: site | forces/step | pending nodes | classification |
|---|---|---|---|---|
| **minimal** (no ckpt, no scaler) | `forward+item :: engine.ts:761` | 1.00 | 237 | **erodible** — `loss.item()` readback; ring-defer it (already have `startItemReadback`) |
| | `backward :: autograd.ts:433` | 1.00 | 408 | **the pre-boundary force** — becomes the single step force |
| | `markStep :: forceAllPending` | 2.00 | 82 | **the boundary** — `endStep`+`markStep` flush the optimizer graph |
| | **TOTAL** | **4.00** | — | 6 submits/step |
| **+checkpoint** | adds `backward :: autograd.ts:355` | +1.00 | 72 | **structural** — recompute of saved tensors → the remat pass (§3.3) |
| | **TOTAL** | **5.00** | — | 6 submits/step |
| **+scaler (AMP inf-skip)** | adds one `markStep :: forceAllPending` | +1.00 | — | **erodible** — deferred scale resolution; scale-as-data (D0) already in flight |
| | **TOTAL** | **6.00** | — | 7 submits/step |
| **DEFERRED-LOSS** (no mid-step item) | `backward :: autograd.ts:433` | 1.00 | **630** | forward **merged in** — item force gone |
| | `markStep :: forceAllPending` | 2.00 | 85 | boundary (optimizer) |
| | **TOTAL** | **3.00** | — | **5 submits/step** |

**Reading of the census:** every mid-step force is one of exactly three kinds:

- **Erodible** — `loss.item()` (defer via ring), the scaler's deferred-scale flush
  (scale-as-data). Named existing patterns; no new mechanism.
- **The boundary** — the 2 `forceAllPending` in `endStep`/`markStep`. This *is* the
  step boundary; in the compiled step it is the single replay dispatch. Note it
  currently forces the **optimizer graph separately** (~82 nodes) only because
  backward already forced grads at `autograd.ts:433` *before* `optimizer.step()`
  ran. Defer that grad-write force too and forward+backward+optimizer become one
  graph forced once — **literally one force per step.**
- **Structural** — checkpoint recompute (`autograd.ts:355`, 72 nodes). This is the
  *only* mid-step force that is not trivially erodible, and it is exactly the thing
  the remat compiler pass subsumes (§3.3): recompute becomes scheduled nodes in the
  whole-step graph, not a hook-driven mid-backward force.

**Zero genuine blockers.** No force site reads a value that is undeclared or
data-dependent in a way the graph cannot express. (Data-dependent *control* — an
inf-skip that changes which nodes run — is handled as data/variant, §5.)

### 2.2 Graph-scale and plan-builder scalability

The whole-step graph at ~727 nodes (distil) / ~3–6k (medium) is constructible today
(`buildMergedPlan` already accepts arbitrary root sets and produced the 630-node
merged plan in the probe). The scalability *risk* is real but bounded and named
(from the plan-builder audit):

- **Collector recursion** (`plan-builder.ts` `buildMergedPlan.visit`) is recursive
  over inputs → JS stack-overflow risk on deep chains at 5–15k nodes. Fix: explicit
  work-stack. (P0 of the campaign.)
- **Four latent O(n²) spots**: the three Kahn passes' sorted-array/`splice` ready
  set + linear `selectBestForFusion` scan; checkpoint-boundary all-pairs edges;
  the fusion `segmentPlanForExecution` gap scan; `redirectConsumers` filter + DCE
  fixpoint. All masked by "plans are small" today. At ~730 nodes they are still
  fine (current 507-node plan pays them every step); at medium-scale ~5k they need
  hardening (heap-backed ready set, bounded gap scan). This is derisking work, not
  a wall — it is the **compile-time cost** of a once-per-shape compile, amortized
  over thousands of steps.

The compile is **not** on the per-step hot path: it runs once per (shape, config)
trace, exactly like the tape's record step today, then replays. So O(n²) at 5k
nodes costs milliseconds *once*, not per step.

---

## 3. Architecture

### 3.1 Trace acquisition — capture-with-deferred-forces

There is no separate "tracer." To acquire the whole-step trace, the training loop
runs under a **capture scope** that:

- redirects `loss.item()` / metric readbacks to the ring (deferred; the value is
  delivered one step late, already the norm for overlapped loss readback);
- suppresses the backward grad-write force (`autograd.ts:433`) and the
  `endStep`/`markStep` `forceAllPending` — instead collecting *all* live pending
  roots (grads, param updates, RNG advance, scale update) as the **step output
  set**;
- forces that output set **exactly once**, at the boundary, via one
  `buildMergedPlan(outputs)` — the whole-step graph.

The first execution of a given (shape, config) trace runs **eager/lowered** and is
recorded (this is what the existing build-from-IR compiler already does per-plan —
we lift it to per-step). The compiler then builds `S`. From the 2nd matching step,
`S` replays.

### 3.2 The whole-step IR — the SAME IR, one big graph

The whole-step IR is **not a new representation**. It is the existing `LazyIRNode`
graph (`src/graph/types.ts`), unsegmented. Today the graph is *de facto* segmented
by force timing (forward plan | backward+optimizer plan); the whole-step compiler
removes force timing as the segmentation authority and lets **the compiler**
segment, by dataflow, for its own purposes (fusion groups, checkpoint remat,
memory-plan intervals) — never by *when a CPU value happened to be demanded*.

Concretely: `ExecutionPlan = { nodes: LazyIRNode[]; outputIndices }` is already the
whole-step artifact's shape. The compiled step is one `CompiledPlan`
(`src/executor/compiled-plan.ts`) — a flat GPU command stream (dispatch/copy/alloc/
barrier) over abstract slot indices — built from the whole-step graph instead of
from a per-force plan. The existing slot table + `GpuCommand` representation is the
core of the whole-step plan and **survives unchanged**; it simply spans the whole
step.

### 3.3 Compiler passes

The passes are the existing ones (`src/compiler/` — CSE, DCE, algebraic identities,
fusion-detect, epilogue chains, row-programs, lifetime analysis) run over one graph,
plus two passes that **subsume entire runtime subsystems**:

- **Remat pass (subsumes stage-3 remat + D3 + checkpoint unpack hooks).** Gradient
  checkpointing today is a *runtime* mechanism: pack hooks discard activations,
  unpack hooks re-run forward subgraphs during backward, and the recompute is forced
  mid-backward (the census's structural force, `autograd.ts:355`). As a compiler
  pass, checkpointing becomes a **scheduling+liveness decision**: the whole-step
  graph already contains both the original forward nodes and their backward
  consumers; the remat pass chooses, per activation, whether to *keep* its buffer
  live across the backward interval (memory) or *recompute* it by re-scheduling its
  producer subgraph just before its backward consumer (compute). No hooks, no
  mid-step force — recompute is just nodes in the stream, placed by the scheduler.
  The `#97`-class correctness concern (recompute must be numerically identical,
  incl. RNG) moves to **compile time**: the pass must prove the recompute subgraph
  is a pure function of retained inputs + replayed RNG, and the differential gate
  (§4) catches any divergence before `S` is trusted.

- **Global memory planning (subsumes #99/D2b′ + arena + pool + registry
  unification).** Today three allocators cooperate at runtime: the arena (per-plan
  persistent buffers), the pool (`SimpleBufferPool` with fence-gated deferred
  destroy + idle trim), and the `PlannerRegistry` (cross-plan temp packing). Over
  the whole-step graph, liveness is **statically known** (every value's last use is
  a graph edge), so memory is **one global plan**: greedy interval allocation over
  size classes (the existing `memory-planner.ts` becomes the sole planner), packing
  every temporary into a fixed arena sized once. The pool degenerates to a plain
  allocator for the (now rare) uncompiled/transient path; its runtime-adaptive
  trim/liveness/reclaim logic (~150–250 SLOC) is dead. The arena's runtime
  identity-stabilization bookkeeping (`outputSeqIndex`, per-plan persistence) is
  dead. **The observed-liveness convergence engine — the single largest runtime
  learner — is dead outright**, because it exists only to *discover* the liveness
  the whole-step graph states.

The existing planner is the **seed**: it already does graph-liveness interval
allocation with cross-plan packing (`stage4-phase15-planner-default`). Lifting it
from "cross-plan" (two plans) to "whole-step" (one graph) is a scope change, not a
rewrite.

### 3.4 Guards — Dynamo-shaped, reusing the tape's

The runtime guard taxonomy (structGen / bucketKey / scalar-coverage / plan-validity
/ epoch-regime) collapses to **input guards** on the declared step inputs:

- **shape/dtype guards** on the input slots (batch shape, param shapes) — a bucketed
  match, exactly the tape's `bucketKey` restricted to inputs;
- **structural guard** = the trace identity (same op sequence) — one fingerprint,
  the tape's `structKey`;
- no *staleness* guard, no *cross-plan witness* guard, no *convergence regime* guard
  — those guarded runtime-learned facts that are now static.

Guard miss → typed trace break → eager re-trace + recompile. The guard set is a
strict subset of what the tape already computes, so it is **reused, then trimmed**,
not built anew.

### 3.5 The execution contract — compiled step replaces the tape

The compiled step `S` *is* the step tape's terminus: one artifact per (shape,
config), replayed under input guards. The existing tape (record/replay/skeleton) is
deleted (§4) because the compiled step is a strictly better realization of the same
idea — statically planned instead of runtime-diffed. **Eager remains the semantic
reference**: any step whose guards miss, or whose config is uncompiled/transient,
runs eager-lowered and is correct-by-construction (the zero-residue fall-through the
build-from-IR campaign already established, `stage4-cutover-default`). This extends
that zero-residue property from per-plan to per-step.

---

## 4. Phased campaign plan (shippable, differential-first)

The mother of all parity gates governs every phase: **the compiled step must
byte-match (≤1e-5 per-step-loss over ≥30 steps) the eager/tape trajectory**, run
both directions of every toggle. This is `tools/parity-fullstack-tl.ts` lifted to
compare `compiled-step` vs `eager` (the existing compiled-plan-vs-lowered gate, one
level up). No phase lands without it green in both directions.

- **P0 — Scale the passes (no behavior change). DONE (2026-07-18).** Convert
  `buildMergedPlan.visit` recursion → explicit stack; heap-back the Kahn ready sets;
  bound the fusion gap scan and checkpoint all-pairs edges. Gate: existing suite
  green + a synthetic 15k-node graph compiles in bounded time. *Ships nothing
  user-visible; derisks the whole-step graph.* **See §P0 below for the measured
  before/after tables.**

### P0 status — pass-scaling measured, all spots linearized

Harness: `probe/pass-scaling.ts` (captures the DEFERRED-LOSS fwd+bwd whole-step
graph at the single backward force — the census's 630-node config at distil/6L —
and synthetically scales it by layer replication with honest per-layer op mix;
node count is independent of hidden size, so tiny dims keep param materialization
cheap). Times each compile-path pass in isolation plus the end-to-end
`analyzeGraph`. Reproduce a size:
`VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim npx tsx probe/pass-scaling.ts <numLayers>`.

**Baseline (before P0), fwd+bwd graph, ms (median), A100 dw-2-1:**

| layers | nodes | plan-build | fusion-reorder | WAR-order | CSE/DCE | segmentation | **analyzeGraph** |
|---|---|---|---|---|---|---|---|
| 6 | 621 | 0.16 | 2.23 | 0.04 | 1.69 | 6.18 | **9.47** |
| 24 | 2367 | 0.57 | 12.29 | 0.12 | 3.60 | 79.90 | **102.29** |
| 60 | 5859 | 1.51 | 72.84 | 0.12 | 10.13 | 530.55 | **664.44** |
| 150 | 14589 | 3.61 | 438.97 | 0.37 | 30.26 | 4160.24 | **5050.83** |

**Post-P0, same harness/machine:**

| layers | nodes | plan-build | fusion-reorder | WAR-order | CSE/DCE | segmentation | **analyzeGraph** |
|---|---|---|---|---|---|---|---|
| 6 | 621 | 0.41 | 1.15 | 0.05 | 2.42 | 1.89 | **3.87** |
| 24 | 2367 | 0.55 | 2.12 | 0.05 | 3.44 | 5.86 | **14.30** |
| 60 | 5859 | 1.57 | 3.48 | 0.12 | 9.40 | 11.58 | **31.32** |
| 150 | 14589 | 3.84 | 11.67 | 0.38 | 30.48 | 33.47 | **84.13** |

**The 15k-node compile: 5050 ms → 84 ms.** Every pass is now ~linear (all under the
bar of "seconds acceptable, minutes not" by two orders of magnitude); the dominant
residual is CSE/DCE at 30 ms, which measured ~linear all along.

**Which spots bit vs which were cold** (the doc named the recursive collector +
four O(n²) spots; fix what measures, report what doesn't):

- **Recursive collector** (`buildMergedPlan.visit`) — linear in time but a JS
  stack-overflow risk on deep chains at 15k. Converted to an explicit work-stack
  (iterative postorder, byte-identical order). *Fixed (mandated for stack safety).*
- **Fusion `segmentPlanForExecution` / `detectFusionGroups` — BIT HARDEST (4.16 s
  @15k).** Two distinct O(n²): (a) `processCandidate` rescanned *every plan node*
  per intermediate to find external references — replaced by a per-producer
  consumer map built once (O(edges)); (b) the segmentation wrapper's gap-node
  ancestor walk used a per-gap-node `visited` set — shared it across gap nodes
  (groups only ever go un-emitted→emitted, so a resolved subtree never needs
  re-walking). A third, `batchGlobalSingletons`' gap rescan, was **provably
  redundant** (the preceding earliest-consumer guard already excludes any gap
  consumer) and deleted; its O(batch) rescan collapsed to a running min.
- **Fusion reorder ready-set / `selectBestForFusion` scan — BIT (0.44 s @15k).**
  The old full-ready-set scan per emit is O(n²) because the backward frontier is
  O(n) wide (one gradient branch per parameter). Replaced by per-priority
  min-heaps with epoch-stamped chain-continuation (P0), preserving the exact
  (priority, original-position) tie-break.
- **Checkpoint all-pairs edges + WAR Kahn ready-set/affinity** (in
  `enforceWriteAfterReadOrder`) — **COLD in the fwd+bwd graph** (it early-returns
  with no in-place ops), but the *whole-step* graph folds in the optimizer
  (in-place adamStep) and its wide fan-out. A realistic wide-frontier synthetic
  (`PS_MODE=warsynth`) confirmed the splice ready-set + O(ready) affinity scan go
  O(n²) there (2.2 s @6k wide, timing out @16k) and the checkpoint all-pairs edges
  are O(B·n) (136 ms @16k, 159 boundaries). All three fixed: O(n) segment-barrier
  edges (transitively identical partial order → identical deterministic Kahn
  output), min-heap ready-set, per-op affinity heaps → all linear (~18 ms @16k).
- **`redirectConsumers` filter + DCE fixpoint** (CSE/DCE) — **COLD** (~linear,
  30 ms @15k). Not touched.

**Behavior-identity (the NULL discipline).** Every fix is order-preserving:
`test/fusion-decision-corpus.spec.ts` (byte-identical fusion decisions) green,
`test/second-run-determinism.spec.ts` green, `test:gates` 5/5 (incl. compiled ==
lowered over the full inner step). A tl-vs-tl trajectory differential
(`probe/traj-check.ts`, 30 steps, checkpoint+scaler+clip+Adam — exercises reorder,
segmentation, and the WAR in-place/checkpoint-edge paths): the modified code's
losses fall inside the *original code's own* cross-process fp variance (~1e-6 at a
single step; the baseline is itself not bit-stable run-to-run), and compiled ==
lowered bit-for-bit. `probe/census.ts` numbers unchanged (630-node DEFERRED-LOSS,
4/5/6/3 forces).

- **P1 — Whole-step trace acquisition behind a flag.** Capture scope that defers
  loss/grad-write/boundary forces and forces the step output set once. Run
  eager-lowered (no compile yet). Gate: parity vs today's two-plan path, byte-exact;
  census re-run shows 4→1 mid-step forces (checkpoint off). *This is the probe's
  DEFERRED-LOSS config, productized.*

### P1 status — landed (2026-07-19), behind `TORCHLETTE_WHOLE_STEP`

The trace surface. Under `TORCHLETTE_WHOLE_STEP=1`, a training step run inside
`api.wholeStep(fn)` (the plain-driver surface) or a `{training:true}`
`api.capture` body (the ring surface, which already defers loss + the boundary)
additionally **defers backward's grad-write force** (`autograd.ts:433`) to the
step boundary. Forward + backward + optimizer then accumulate as ONE lazy graph
that the boundary `forceAllPending` materializes exactly once — the whole-step
trace, run eager-lowered.

**What defers, what stays** (the census reading, realized):
- *Erodible → eroded.* `loss.item()` rides the ring (already existed); the
  backward grad-write force now merges into the boundary force. Its teardown
  (non-survivor grad dispose, saved-tensor / `slot.retained` release,
  `tidyMode.disposeNonEscaped`, `destroyUnreachable`) is **deferred with it** to
  a boundary-drained queue (`_deferToBoundary` → `_drainBoundaryDeferred`, run
  right after the single `forceAllPending`, before the step-scoped demotion
  sweep) — those values still feed the un-forced plan; disposing them in
  backward reclaimed their buffers early (the `[lifetime] reading RECLAIMED`
  class). The drain frees at rc-0, never demotes-with-live-rc (no pool
  double-release).
- *The boundary.* Already one `forceAllPending` on the ring path
  (`_deferBoundaryCommit`); the plain `markStep` path is the caller's single
  boundary force.
- *Structural — NAMED, NOT eroded.* The checkpoint-recompute force
  (`autograd.ts:355`) stays. **P1 covers NON-checkpoint configs**; recompute is
  P3's remat pass. A checkpointed backward **gates the deferral OFF entirely**
  (`deferForce = false` when `hasCheckpoints`) — deferring past the recompute
  force + `disposeCheckpointIntermediates()` frees recomputed inputs the
  un-forced plan still reads (a UAF/GPU-crash). So under checkpointing the
  whole-step scope is a no-op: the step runs exactly the eager path. Named in
  code (the flag doc-comment + the `hasCheckpoints` gate) and here.

**Census re-run** (distil, seq15, forces/step before → after, flag on):

| Config | before | after | note |
|---|---|---|---|
| minimal (no ckpt/scaler) | 4 | **2** | whole step = ONE 703-node plan at the single boundary force; the 2nd count is the empty `beginStep` bookkeeping call (0 pending) |
| +scaler | 6 | **3** | scaler's deferred-scale flush stays — erodible D0, not P1 |
| +checkpoint | 5 | 5 | **no-op** — deferral gated off under checkpointing (eager fallback); recompute is P3 |
| DEFERRED-LOSS | 3 | 3 | unchanged — the pre-P1 config P1 productizes |

(The minimal/+scaler rows were measured with `TORCHLETTE_WHOLE_STEP=1` before
the checkpoint gate landed; the gate touches only the checkpoint path, so those
non-checkpoint numbers are unchanged by construction. The +checkpoint row is
`5→5` by the gate — the whole-step scope no-ops to the eager checkpoint path.)

The `forward+item` force and the separate backward grad force are both gone from
the minimal config; the whole 703-node step is one merged plan.

**The differential (the mother gate)** — `tools/t-whole-step-diff.ts`, distil,
30 steps, both directions: traced == eager **2.86e-6**, compiled == lowered
(within traced) **3.34e-6**, both ≤ 1e-5. Deferral changes only *when* forces
happen, never the math (step-0 loss is bit-identical: `2.802013…` on every arm).

**Headline measured — NO submit win at P1 (a P2 deliverable).** At distil@512
the traced step runs *eager-lowered* at **11.5 submits** vs eager's 7.5 (302 ms
vs 172 ms). Forced eager-lowered, the combined ~727-node plan segments into MORE
barriers than three separate forces (the optimizer's in-place `adamStep`
interleaved with backward adds WAW segment boundaries). The submit/speed win is
exactly what **P2's whole-step COMPILE** (one `CompiledPlan` + one global memory
plan) delivers — P1 acquires the trace; P2 optimizes its execution. Honest
negative, and it re-scopes the "reduce submits" expectation to P2 where the
design already places "submits/step drops."

**What idles (observed, not deleted — the ledger executes at P4).** On the
`TORCHLETTE_WHOLE_STEP` path the mid-step backward-force segment vanishes, so the
tape's per-segment eligibility diffing has one fewer boundary to witness on
traced steps. Not yet load-bearing (the flag is soak-only; the default path
still forces backward separately), so nothing is removed. Flag OFF is
byte-identical to today (`_deferBackwardForce()` is false → every original
branch taken; the drain is a no-op on an empty queue; the scope is a depth
counter) — verified: `t-ring-probe` all-zero deltas (K a pure knob), ledger
balanced, `parity-fullstack` compiled==lowered 7.6e-6, `test:gates` 5/5,
`profile-training` distil@512 leak-OK at ~48 ms/step.

- **P2 — Compile the whole-step graph.** Lift build-from-IR from per-plan to
  per-step: one `CompiledPlan` for `S`, one global memory plan (planner over the
  whole graph). Gate: the parity gate + memory flat + submits/step drops. *First
  point the deletion ledger starts paying out (observed-liveness convergence can be
  bypassed for compiled steps).*

### P2 status — landed (2026-07-19), compile-in-the-loop

Under `TORCHLETTE_WHOLE_STEP=1` the single traced whole-step graph goes through
the **normal K_w record-then-cutover lifecycle** and compiles — first exec
lowered (the reference), then generated build-from-IR, then cutover. The
existing compiler substrate (`generateStream` + `buildCompiledPlanFromGenerated`
+ `planMemory` + the K_w=2 cutover) **mostly just worked** on the merged graph
(P0 had already scaled the passes). One class needed a named fix.

**What just-worked.** The merged ~700-node graph builds, plans, cuts over, and
replays through the unchanged substrate. The whole-step **uncovered census is
EMPTY** for non-checkpoint distil: at seq512 the trace compiles into **4
templates (upload / embed / attn-block / fwd-bwd-opt), ALL `BUILD-FROM-IR` with
NO lowered execution** — zero plans stranded lowered. Convergence then prunes
each (21 / 189 / 2 harvest pairs). The step is *not* literally one
`CompiledPlan`: the executor's plan builder still segments the graph at WAW /
barrier boundaries (the in-place `adamStep` interleaved with backward is a real
barrier), so "one global memory plan" is per-segment-global, not one-plan-global.
The submit win is the honest consequence — fewer force-points, not one submit.

**What needed fixing — the #97 class, named: `park-live-registered-result`.**
The compiled whole-step plan corrupted at the **convergence step** (a one-time
event, then self-heals): the loss read a stale `0` and Dawn reported *"used in
submit while destroyed."* Root cause: a whole-step compiled plan produces a
PERSISTENT result — the `registerState`'d deferred loss, read AFTER the boundary
via `item()` — whose buffer is a **non-owning pinned planner RESULT ENTRY**
(`createTensor(..., ownsBuffer = !base && !pinnedBufferSet.has(buf))` → false, so
it can't be flipped to owning). At convergence the plan is invalidated and
`destroyCompiledPlanBuffers` deferred-destroys its result entries; the
post-boundary `item()` copy then submits against a destroyed buffer → the submit
drops → loss reads `0`. (Training is unaffected — only the readback submit drops;
the eager-side losses match to the fp floor either side of the corrupted step.
The gate's floor-relative tolerance had been MASKING it: the corrupted
compiled-vs-lowered floor inflated the tolerance that hid the corrupted
traced-vs-eager delta — a false PASS.) Fix (`compiled-plan.ts` +
`observed-liveness.ts` + `storage-tracker.ts`): at teardown, a result-entry
buffer still read by a live **REGISTERED-STATE** harvested storage is **PARKED**
(kept pinned, detached from the registry so a rebuilt plan materializes a fresh
buffer) instead of destroyed, and reclaimed at a later boundary once every reader
storage dies (`reclaimParkedLiveBuffers`, run every `observeStepBoundary`). This
is the idle-retirer's live-harvest check applied PER BUFFER, so **convergence
still proceeds** (only the one persistent buffer is spared). The `REGISTERED`
narrowing (`storageTracker.isRegisteredStorage`) keeps parking **whole-step-only**
— a merely-alive step-scoped result (every default-path harvest) is excluded, so
the default (flag-off) path's buffer teardown is byte-identical (proven:
`t-ledger-attack` default+48 reach/total drift 0, same as baseline; a broader
"any alive" gate perturbed it to drift 100).

**The differential (the mother gate), now with compilation in the loop.**
`t-whole-step-diff` distil seq64 / 30 steps: traced == eager **2.62e-6**,
compiled == lowered (within traced) **2.62e-6**, both ≤ 1e-5, **no
tolerance-masking** (the honest pass). At seq512 the compiled-vs-lowered floor is
**6.4e-6** (byte-clean — P2's actual deliverable); the traced-vs-eager 2.7e-4 at
seq512 is entirely P1's DEFERRAL fp-reorder (eager-vs-tracedLowered is also
2.7e-4, both lowered), pre-existing and independent of the compile.

**The headline (A100 dw-2-1, steady-state late-step).** Traced-compiled beats
BOTH the eager baseline (fewer submits, ≥ speed) and P1's traced-lowered (much
faster):

| config | eager (per-plan compiled) | traced-compiled (P2) | P1 traced-lowered |
|---|---|---|---|
| distil@512 | 160.2 ms / 7.5 submits / 5545 MB | **154.8 ms / 6.5 submits / 5545 MB** | 283 ms / 7.0 / 3805 MB |
| medium@512 | 536.1 ms / 14.5 submits / 17795 MB | **536.2 ms / 13.5 submits / 17795 MB** | 870.5 ms / 14.0 / 12399 MB |

Submits drop **−1/step** both models; speed is at-or-better (distil −3%, medium
parity); memory is **equal to eager** (both compiled paths pin result entries to
the same footprint). The design's promise — "fewer submits, at-or-better speed,
at-or-better memory" — lands as: submits ↓, speed ≥, memory =. The dramatic
"submits collapse" the charter hoped for is modest (the optimizer barrier
prevents a single plan); reported honestly.

**Registry / ab-oracle movement.** `t-planner-pin-attribution` (flag-off,
distil) attributes planner-registry 4073 MB total / 828.7 MB result across 498
entries — the cross-plan RESULT footprint. The whole-step compile does NOT shrink
this at the default path (unchanged); the traced-LOWERED arm uses ~1.5 GB less
(3805 vs 5545 distil, 12399 vs 17795 medium) precisely because it does not pin
result entries — a speed↔memory tradeoff the compiled path resolves toward speed.
The `t-checkpoint-ab-oracle` **memory side did NOT close as a side effect** (D3
precondition still unmet: arena-ON steadyPeak 5040 MB > arena-free+5% = 4130 MB) —
the whole-step compile touches force-points and result liveness, not the
checkpoint arena footprint, so the D3-blocking gap is unchanged. Not chased.

**What idles further (observed, not deleted — the ledger executes at P4).** The
whole-step plan changes the ledger's SHAPE on traced configs — the mid-step
backward-force segment is gone, so the tape's per-segment eligibility diffing has
one fewer boundary to witness, and the observed-liveness over-harvest the
convergence machinery corrects is now a per-step-once event on the single trace
rather than per-force. The tape/witness on traced configs still run (tape-matrix
4/4, witness 2 cells pass); nothing is removed (the flag is soak-only; the
default path still forces backward separately). Flag OFF remains byte-identical
(`park` is REGISTERED-gated → never fires on the default path; the reclaimer is a
no-op on an empty park list).

- **P3 — Remat as a pass.** Move checkpointing from unpack-hook forces to
  scheduled recompute in the whole-step stream. Gate: parity with checkpoint on,
  memory ≤ today's checkpointed peak, RNG-identical recompute proven at compile time.
  *Deletes the structural mid-step force.*

### P3 status — landed (2026-07-19): rewrite COMPILES + CORRECT; D3 bypass-death STOPS on SPEED

The rewrite (behind `TORCHLETTE_WHOLE_STEP`). The P1 checkpoint gate
(`autograd.ts` `if (hasCheckpoints) deferForce = false`) is **LIFTED**: under the
whole-step scope a checkpointed backward is now **REMAT** — the recompute
subgraph the unpack hook already builds (re-running `fn(...inputs)`, a duplicate
of the checkpointed forward reading the segment inputs) stays **LAZY** and flows
into the single boundary `forceAllPending` alongside forward + backward +
optimizer. The two mid-backward forces (`forwardToForce` / `savedToForce`) and
the two-plan split are SKIPPED under remat; `disposeCheckpointIntermediates()` and
the recompute-scope teardown are **deferred to the boundary drain**
(`_deferToBoundary`) — symmetric with every other deferForce teardown, so the
recompute buffers survive until the single force consumes them (no UAF). The
eager/flag-off path keeps the two-plan force UNTOUCHED — it is the reference. The
"recompute becomes a compile-time graph rewrite" the charter names is realized by
the recompute nodes being intra-graph duplicates the memory planner packs, not a
new pass: laziness *is* the rewrite.

**The one hazard the two-plan split guarded against does NOT bite.** The split
existed because "mixing unmaterialized forward nodes into the *separate* recompute
plan produces invalid reshape rewrites." Under remat there is no separate plan —
the whole-step merged graph is ONE plan, the same non-checkpoint merge P1/P2
validated. `isCheckpointBoundary` fires under remat (the recompute stays pending
to the boundary, unlike eager where it is already materialized) but is **inert on
the fusion/compiled path** (read only by `sequential.ts`'s segmented executor, which
the fusion path pre-empts) — so it drives no segmentation. **RNG identity is by
construction, at compile time:** the unpack hook replays the recorded draws at
graph-BUILD time (`_debug_startCheckpointReplay` bakes recorded draw *values* into
the recompute nodes when `fn` re-runs), so deferring the force cannot change what
the recompute computes — the red-team's #97 concern is satisfied structurally, not
by hope.

**The remat differential (the mother gate).** `t-whole-step-diff` with `CKPT=1
SELECTIVE=1`, distil seq64 / 30 steps (crosses the K_w compile cutover):
traced-remat == eager-checkpointed **2.15e-6**, compiled == lowered within
traced-remat **2.38e-6**, both ≤ 1e-5. Step-0 loss bit-identical across arms.
Flag-off byte-identical: non-ckpt `t-whole-step-diff` 3.34e-6; the eager checkpoint
path (checkpoint-autocast-parity / segmentation / scaler-seed /
distilgpt2-checkpoint-memory specs) 7/7; `parity-fullstack` flag-off
(checkpoint+scaler+clip+autocast) compiled==lowered 4.77e-6; `test:gates` 5/5.

**The rewrite COMPILES under checkpointing** (arena-recompute Risk 2, refuted). The
fear that selective/full checkpointing re-fingerprints the plans every step and
never cuts over is a TWO-PLAN artifact: the whole-step trace is ONE stable graph,
so the remat plan reaches build-from-IR just like non-checkpoint — distil@512 +
selective ckpt: **templates=4, converged=2, pinned=0, cleanMiss=0, dirtyMiss=0**
(the non-checkpoint P2 census, now under checkpointing). This is a genuine P3
advance: **checkpointed selective training is now compile-eligible**, which the
two-plan world structurally could not do.

**THE D3 GATE — BYPASS RETAINED (STOP on speed).** `tools/t-d3-remat.ts`
(traced-remat-compiled vs the arena-free `setBufferArenaDisabled(true)` bypass;
like-for-like, 3 repeats, reset-at-9 steady peak, A100 device 0, seq512):

| config | arm | steadyPeak | steadyCur | ms/step | submits |
|---|---|---|---|---|---|
| distil@512 + selective | bypass | 3933.5 MB | 1798.3 | 299.5 | 8.0 |
| distil@512 + selective | **remat** | **3977.0 MB (+1.1%)** | 2005.7 | **326.6 (+9%)** | 12.4 |
| medium@512 + full | bypass | 12062.6 MB | 5788.1 | 1036.7 | 17.0 |
| medium@512 + full | **remat** | **13020.1 MB (+7.9%)** | 7324.3 | **1148 (+11%)** | 22.4 |

Verdict per config — peak ≤ +5% AND speed ≥ bypass AND traj ≤ 1e-5:
- **distil-selective: peak PASS (+1.1%), speed FAIL (+9%), traj-vs-bypass 1.07e-5** (the seq512 two-path fp floor; the SOUND gate remat-vs-eager is 2.15e-6).
- **medium-full: peak FAIL (+7.9%), speed FAIL (+11%), traj 2.19e-4** (same seq512 floor).

**The year-old MEMORY precondition is MET for selective checkpointing** — the thing
that blocked R3 since 2026-07-16. The two-plan arena-ON compiled path pinned the
cross-plan checkpoint saves whole-step at **+8.8%** peak (`t-planner-pin-attribution`
D3 header); the whole-step remat collapses that cross-plan RESULT to an intra-graph
edge the planner packs → **+1.1%**. But two NEW facts block the deletion:
1. **SPEED (both models).** Remat is +9%/+11% slower: the merged plan **scatters**
   each layer's recompute into its own WAW segment (12.4 / 22.4 submits), where the
   two-plan bypass **batches** all recomputes into `forceAllMerged` (8 / 17 submits).
   The compiled per-op speedup does not recover the extra submit-sync overhead.
   Reducing it is the "planner's chosen points" recompute-batching — a scheduling
   change (islands altitude), out of a correctness-first P3.
2. **PEAK on full checkpointing (medium).** Selective ckpt saves attention +
   recomputes MLP → the collapsible cross-plan pin is the MLP only, and remat packs
   it (+1.1%). Full ckpt recomputes every layer → the whole-step remat holds more
   recompute activations co-live at the boundary force than the bypass frees
   per-force → +7.9%. The memory win is **config-dependent**: it lands for
   selective, not for full.

Per campaign discipline (STOP rather than improvise; no gate-scope games): the
D3 deletion does **not** land. `setBufferArenaDisabled` + `TORCHLETTE_CHECKPOINT_ARENA`
are **RETAINED**. The remat rewrite lands as a soak-only flag-on feature (the P1
gate lifted, the differential green); the bypass-death is re-blocked from MEMORY
(solved for selective) onto SPEED + full-ckpt-peak. What a sound future D3 needs:
recompute-batching in the whole-step scheduler (collapse the per-layer recompute
segments) so submits ≤ the bypass's batched count — a named follow-on, not this pass.

**Census re-run** (distil, seq15, forces/step, flag on): the +checkpoint row goes
**5 → 2** — the structural recompute force (`autograd.ts:355` in the census) and its
`disposeCheckpointIntermediates` are subsumed into the single boundary force, exactly
as the non-checkpoint minimal row (whole step = ONE plan at the boundary; the 2nd
count is the empty `beginStep` bookkeeping).

### Recompute/optimizer-batching follow-on — BUILT, PROVEN SOUND, MEASURED NET-NEGATIVE, REVERTED (2026-07-19)

The named follow-on above ("recompute-batching … so submits ≤ the bypass's batched
count") was attempted and **reverted**. What it found reframes the STOP.

**The segment split, named precisely.** The 12.4-vs-8.0 submit gap is NOT per-layer
recompute segments — it is the **optimizer shattering**. Under the whole-step merge,
each param's `adamStep` becomes Kahn-ready the instant its gradient finishes, so the
frontier interleaves adamSteps with the still-running tail of backward
(`fusedLayerNormBackwardGradWeightBias`, and the weight-grad `transpose`/`matmul`/`sum`
runs). The consecutive-run `adam-batch` action (lowered-plan.ts) then splits into **3–4
batches** (observed op-order gaps: adamStep runs at plan positions ~601, ~604, ~616–641,
~762+, separated by 14 layernorm-grad nodes and a 120-node transpose/matmul/sum block).
Each `adam-batch` action pays an unconditional pre-flush (`executor.ts` ~2786:
`flushSharedEncoder`+`flushBufferPool` for pool safety) plus the kernel's internal
flush → ~2 submits/batch. The two-plan bypass runs the optimizer as a SEPARATE plan
(all grads already materialized ⇒ all adamSteps immediately ready ⇒ one consecutive run
⇒ one batch), which is the entire submit delta. The per-layer recompute subgraphs
themselves do NOT each open a submit (checkpoint boundaries are inert on the fusion/
lowered path; `DEFAULT_RECLAIM_INTERVAL=10000` fires zero reclaims on the ~800-node
plan).

**The fix + why Kahn-legal.** A tie-break change in `enforceWriteAfterReadOrder`
(plan-builder.ts): a second `deferredHeap` for `adamStep`, drained only when the
non-adam frontier empties, self-gated to plans that hold BOTH an `adamStep` AND a
`matmul` (⟺ the whole-step merged plan; the flag-off pure-optimizer and pure-backward
plans lack one of the two, so it is a structural no-op there). Legal because adamStep's
in-place param/m/v writes are read by nothing else in the step (next step = different
plan) and its readers are already pinned before it by the existing WAR edges — so
deferring it as late as the frontier allows never violates a dependency. Result:
**one** adam batch, submits **12.3 → 8.4** (bypass 8.0 — parity). Numerically sound:
`t-whole-step-diff CKPT=1 SELECTIVE=1` traced==eager **3.10e-6**, compiled==lowered
**2.62e-6** (both ≤ 1e-5).

**Why it was reverted — submit-count is NOT the speed bottleneck (falsified).** Full
D3 table, distil@512 selective, 3 repeats, with the fix:

| metric | bypass | remat+fix | doc-baseline remat | verdict |
|---|---|---|---|---|
| steadyPeak | 3933.5 MB | **4149 MB (+5.5%)** | 3977 (+1.1%) | peak REGRESSED, now FAILS ≤+5% |
| ms/step | 298.2 | **314.3 (+5.4%)** | 326.6 (+9%) | speed still FAILS |
| submits | 8.0 | **8.4** | 12.4 | parity reached |
| traj vs bypass | — | **3.72e-5** | 1.07e-5 | FAILS 1e-5 (cross-impl fp floor, not a bug) |

Collapsing 4 batches → 1 removed ~4 submits but recovered only ~12 ms of the ~28 ms
(9%) gap — i.e. **submit-sync was ~3% of the 9%, not the driver.** The design's premise
("the compiled per-op speedup does not recover the extra submit-sync overhead") is
**falsified**: at submit-parity the remat step is still +5.4% slower AND now +5.5% peak
(the deferral holds every gradient co-live to the boundary, the memory cost of one
batch — the tradeoff the P3 note anticipated). Net-negative on the seq512 gate; reverted
per complexity budget (matching the reverted gap-spanning fusion detector precedent).

**The residual gap, and the real lead.** Both arms run **lowered**; the whole-step remat
plan does NOT reach compiled steady-state execution (steady submits stay at 12.4 — a
compiled ~800-node plan would show ~2; the trace is `executeLoweredPlan` every step, and
the CUTOVER `converged=2` counts the tiny readback/bookkeeping plans, not the big
whole-step boundary plan). So the "compiled per-op speedup" the speed premise rests on
**is not active in the measured regime.** A sound future D3 is therefore NOT recompute-
batching (measured null for speed, negative for peak/traj) but **getting the whole-step
remat boundary plan to converge and replay compiled** — a convergence/observed-liveness
question, out of scope for a scheduling pass. Until then the STOP stands.

**VERDICT UNCHANGED: BYPASS RETAINED.** `setBufferArenaDisabled` + `TORCHLETTE_CHECKPOINT_ARENA`
stay. No src change lands from this follow-on.

### Why the whole-step remat plan never converges to compiled — ROOT-CAUSED (2026-07-19)

The follow-on above left the lead as "a convergence/observed-liveness question." **It is
neither.** Instrumenting the seq512 remat arm (`TORCHLETTE_DIAG_CUTOVER`, per-plan cutover
trace + `gen.uncovered` dump) names the exact cause, and it is not what the lead assumed.

**The named cause — an UNCOVERED OP, not re-fingerprinting, not convergence.** The 809-node
whole-step boundary plan has a **STABLE** fingerprint across every step (fp=0x7b2639fd,
`buildReach` increments 1→N — structural instability FALSIFIED). It reaches the
build-from-IR block every step but `gen.fullyCovered` is **false**, so it falls through to
`executeLoweredPlan` forever. From step 1 on (warmup configs captured), the residual
uncovered set at seq512 is exactly **two ops**:
`fusedCrossEntropyForward[no-storage]` and `fusedCrossEntropyBackward[no-storage]`.
One uncovered op ⇒ not `fullyCovered` ⇒ no `genPlan` ⇒ lowered every step. **The
`observed-liveness` CUTOVER `converged=2` counts the small readback/bookkeeping plans; the
809-node plan is never a candidate** — coverage gates the compile, convergence never gets a
turn. This is **NOT seq512-specific**: the seq64 `t-whole-step-diff` traced arm (fp=0x6adb0757,
also 809 nodes) carries the identical two CE bails (plus a seq-only `batched-reduction[true-batched]`
that seq512 covers). So **P3's "COMPILES — templates=4 converged=2" verdict was always
measuring the small plans; the boundary plan never compiled at any sequence length.**

**The bail, precisely.** CE's logits operand is `flatLogits.narrow(1,0,vocabSize)`
(`examples/gpt2/model.ts:507-513`) — a **dim-1 strided view** stripping the lm_head's
tile-alignment vocab padding ([S,50304]→[S,50257]). `stream-generate.ts`
`resolveContiguousOperand` needs the physical layout (strides/offset/bufferSize) to
synthesize the contiguous-copy prologue the dispatch layer's `asContiguous` inserts (CE
logits is declared contiguity-required, `contiguous-operands.ts`). On the build-from-IR
path **nothing is materialized**, so `storage` is undefined AND `refShapeDtype` /
`contiguousViewShapeDtype` both reject strided VIEW_OPs → `no-storage`. The live-storage
branch that *does* synthesize the copy never fires for the whole-step boundary plan (it
builds without executing).

**The coverage fix works — and is where it STOPS.** Deriving the strided layout from IR via
`deriveNodeViewMeta` (the single-source view-meta the backend ops use) + a base-buffer-bytes
walk, then feeding the SAME `planContigCopy` the live branch uses, closes the bail:
`gen.fullyCovered` becomes true and the seq512 809-node plan **cuts over to compiled**
(verified: `hasCompiled=true` at `buildReach≥2`; losses bit-match the lowered arm). It is
numerically correct on the target path — `t-whole-step-diff` **compiled==lowered 1.9e-6**,
and **non-checkpoint traced==eager 1.9e-6**.

**But it fails the mother gate under checkpointing (1.3e-4), and the reason is structural,
not a copy bug.** With CE covered, the plan compiles on the EAGER arm too. A clean 2×2
(distil seq64, CKPT=1 SELECTIVE=1, per-arm loss[29]; the whole-step-remat exact value is
3.012909, == non-checkpoint):

| config | eager (reference) | traced (remat) | verdict |
|---|---|---|---|
| non-CKPT + fix (CE compiles) | 3.012909 | 3.012909 | PASS 1.9e-6 |
| CKPT + baseline (all lowered) | 3.012907 | 3.012908 | PASS 2.15e-6 |
| CKPT + fix, **remat** whole-step | — | **3.012909** (==exact) | remat CORRECT |
| CKPT + fix, **eager** two-plan | **3.013038 (+1.3e-4)** | — | eager CORRUPTED |

The synthesis fires only on the correct CE-forward vocab narrow (same copy in every arm),
so the copy value is right. The divergence isolates to **CKPT × compiled-forward in the
legacy two-plan path** — precisely the checkpoint+arena hazard `setBufferArenaDisabled`
(b66ead78) exists to prevent: a compiled forward plan reclaims activations a *separate*
checkpoint-recompute plan still needs. The **whole-step remat handles it safely** (one
merged plan, the planner packs the recompute — traced is exact). The **eager two-plan
checkpoint path does not**, and before this fix it was kept lowered *by accident* — only
because CE was uncovered. Global CE coverage removes that accidental safety and unmasks
the hazard in the eager reference.

**Why it can't be cleanly scoped, and the STOP.** There is no live whole-step signal at the
boundary-force site: `_wholeStepDepth` is back to 0 by `markStep()` (the scope exits before
the merged plan executes), so `generateStream` cannot tell "merged remat plan" from "eager
two-plan forward." A structural self-gate ("cover CE only in a plan that also holds an
`adamStep`" ⟺ the merged plan, the same self-gate shape the reverted optimizer-batching
follow-on used) would pass the gate, but it is an overfit condition on a correctness seam,
not a smallest-honest fix — and the campaign forbids gate-scope games. **So the coverage fix
does NOT land.** The convergence blocker is now *named and removable*; making its removal
SAFE requires the checkpoint+compiled-forward mechanism the bypass/whole-step exist for.

**Reframed lead + sizing for a sound future D3.** The prize (does compiled remat beat the
bypass on speed?) is now one mechanism away, and the mechanism is checkpoint-safety, not
convergence:
- **(a) Land CE from-IR contiguity coverage** (~40 SLOC in `stream-generate.ts`:
  `derivedContigCopyFromIR` + `deriveBaseBufferBytes`, reusing `deriveNodeViewMeta` /
  `planContigCopy` / the `CONTIGUOUS_OPERANDS` declaration). Correct in isolation; the
  differential asserts the copy == dispatch `asContiguous`.
- **(b) Prevent the eager two-plan checkpoint forward from compiling unsafely** — either
  re-assert the bypass at plan granularity for non-whole-step checkpointed forwards (keep
  them lowered by design), or make the compiled-forward+separate-recompute path arena-safe.
  This is the load-bearing, non-trivial piece; it is P3/P4 checkpoint-arena territory, not
  part of covering CE. Only with (b) can (a) land and the seq512/medium D3 table be re-run
  on the remat arm to render the speed/peak/traj verdict. Until then the STOP stands.
  **→ RESOLVED (2026-07-19): both (a) and (b) landed — see "The D3 FINISH" below.
  Compiled remat beats the bypass on speed AND peak for both selective and full; the
  STOP is reversed.**

### The D3 FINISH — CE coverage LANDED behind a SUNSET-BOUND eager-checkpoint refusal (2026-07-19)

Vin's ruling on the hazard: (b) is NOT "make the compiled-forward+separate-recompute
path arena-safe" (P4 work). It is a TYPED, SUNSET-BOUND compilation **refusal** — the
transition scaffolding that lets (a) land NOW and protects the gate matrix during the
soak (D3 arms, witness checkpoint cells, memory-stability specs all run
checkpoint+arena-ON+eager; global CE coverage un-masks the b66ead78 corruption there).

**The refusal (landed FIRST, safety before coverage).** Every build-from-IR plan built
during a checkpointed **EAGER (non-whole-step)** step declines compilation and stays
lowered — reason `checkpoint-eager-two-plan-force` (`executor.ts`
`CHECKPOINT_EAGER_REFUSAL`). Scoped by a **live per-step engine signal**
(`RuntimeEngine._checkpointEagerForce`) set at the FORWARD checkpoint site
(`nn/checkpoint.ts`, only when NOT deferring to a whole-step merge) and cleared at the
step boundary force (`forceAllPending`, after the optimizer) + `beginStep`. It is
**ALL-OR-NOTHING per step** — forward+loss, backward, AND optimizer all stay lowered: a
partial mix (a lowered checkpoint forward/backward feeding a COMPILED optimizer plan)
breaks the grad/planner handoff and freezes training (found by the parity-fullstack
gate), exactly what the `setBufferArenaDisabled` bypass avoids by disabling the arena
for the whole step. Here the arena stays ON so the remat arm can still show its memory
win. NOT node flags and NOT "anything checkpoint-flavored": the whole-step remat merged
plan is forced at the boundary drain (its forward runs inside the wholeStep scope, so
`_deferBackwardForce` is true → never marked), so it still compiles. This restores
"eager checkpoint plans stay lowered" AS A DECLARATION — before CE coverage they were
lowered by ACCIDENT (uncovered CE-narrow). Gate:
`test/whole-step-checkpoint-refusal.spec.ts` — FIRES on the eager checkpoint config
(refusals > 0), does NOT fire on non-checkpoint (P2 reference untouched) or on
whole-step remat (merged plan compiles). SUNSET: dies WITH the bypass
(`setBufferArenaDisabled` + `TORCHLETTE_CHECKPOINT_ARENA`) when whole-step training
defaults and the eager two-plan path is deleted (P4).

**CE-from-IR coverage LANDED (a) behind the refusal.** `resolveContiguousOperand`'s
no-storage branch now, for a contiguity-required STRIDED-view operand with no live
storage, derives the layout purely from IR (`deriveNodeViewMeta` for
shape/strides/offset + a base-buffer-bytes walk for the allocation size) and feeds the
SAME `planContigCopy` the live branch uses — closing the
`fusedCrossEntropy{Forward,Backward}[no-storage]` bail on the CE logits
`narrow(1,0,vocabSize)`. The 809-node whole-step remat boundary plan reaches
`fullyCovered` → **compiles** at `buildReach≥2`. With the refusal protecting the eager
reference, the mother-gate 2×2 passes all four cells: `t-whole-step-diff` compiled==lowered
2.6e-6, non-checkpoint traced==eager 2.2e-6, CKPT=1 SELECTIVE=1 traced-remat==eager 3.6e-6
(eager now 3.012908, NOT the pre-refusal corrupted 3.013038). `parity-fullstack` flag-off
(checkpoint+scaler+clip+autocast) compiled==lowered 8.6e-6/30.

**THE FIRST COMPILED D3 TABLE — the STOP is REVERSED (V100 sivri, device 0, 3 repeats,
reset-at-9 steady peak; `tools/t-d3-remat.ts`).** The remat arm now COMPILES
(`hasCompiled=true`); trajectory by the P3 floor-aware method (remat-COMPILED vs the
same-arm EAGER reference, gated at `max(1e-5, 1.5×compiled-vs-lowered)`):

| config | arm | steadyPeak | ms/step | submits | compiled | verdict |
|---|---|---|---|---|---|---|
| distil@512 + selective | bypass | 3933.5 MB | 300.6 | 8.0 | no | — |
| distil@512 + selective | **remat** | **3712.6 MB (94.4%)** | **91.2 (3.3× faster)** | 9.4 | **yes (1468 cmds)** | **D3-READY** |
| medium@512 + full | bypass | 12062.6 MB | 1050.7 | 17.0 | no | — |
| medium@512 + full | **remat** | **10401.1 MB (86.2%)** | **476.9 (2.2× faster)** | 19.4 | **yes (6041 cmds)** | **D3-READY** |

- **distil-selective: D3-READY** — peak −6% (94.4%), speed 3.3× faster, traj(remat-vs-eager)
  9.3e-6 ≤ gate 1.22e-5.
- **medium-full: D3-READY** — peak **−14%** (86.2%), speed 2.2× faster, traj 2.2e-4 ≤ gate
  2.6e-4 (the seq512/medium fp floor; losses match to 4 digits, 1.7618).

**Both configs clear speed + peak + trajectory + hasCompiled.** The doc's earlier STOP
("remat +9%/+11% slower; full-ckpt-peak +7.9%") was measured on the LOWERED remat plan
(the boundary plan never compiled — CE was uncovered). CE coverage makes it compile, and
compiled replay + the memory planner packing the whole-step recompute REVERSES both
axes: remat now beats the bypass on speed AND peak, for BOTH selective and full
checkpointing. The ROOT-CAUSE's precondition (b) is met by the refusal, (a) is landed,
and the reframed lead ("does compiled remat beat the bypass?") is answered: **yes.**
Census (distil, seq15, `TORCHLETTE_WHOLE_STEP=1`): +checkpoint = 2.00 forces/step (the
structural mid-step force subsumed), 6.0 submits/step, boundary plan compiled.

**Bypass death is a SEPARATE reviewed pass** (per the campaign — this pass renders the
verdict, it does not delete). `setBufferArenaDisabled` + `TORCHLETTE_CHECKPOINT_ARENA`
and the refusal all REMAIN until that reviewed deletion. The refusal's sunset is now
unblocked: D3-READY on both configs.

### THE BYPASS DEATH — LANDED (2026-07-19, the reviewed pass D3-READY earned)

The reviewed deletion pass. `WebGPUGPT2Trainer.singleInnerStep` now wraps its checkpointed
inner step in `api.wholeStep` (deferring the per-micro-batch loss read to a detached scalar
copy read AFTER `markStep`, so the merge is not force-split mid-step); `initialize()` no
longer disables the arena. **Deaths named:** the trainer's `setBufferArenaDisabled(true)`
call + its ~40-line WHY comment (task #98 phase-3 STOP narrative) and the
`TORCHLETTE_CHECKPOINT_ARENA` env flag — DELETED. **Survives (P4 deletions):** the ENGINE
method `RuntimeEngine.setBufferArenaDisabled` (the D3 tool's `bypass` measurement arm +
`t-planner-pin-attribution` still drive it) and the `CHECKPOINT_EAGER_REFUSAL` (guards the
residual flag-off eager checkpoint step, arena-ON/lowered). Net src SLOC: ~−45 (comment +
call + flag read); +~40 in the new wholeStep-wrapping trainer body (deferred-loss/live-upload
bookkeeping); +1 death of an env flag.

**Spec fix (owed, landed here).** `test/whole-step-checkpoint-refusal.spec.ts` threw a
strict-lifetime `[lifetime] reading RECLAIMED` (the wpe position-embedding grad, scatterAdd→add)
in BOTH its eager cells under strict-DEFAULT — refusal-INDEPENDENT spec-loop hygiene: the
minimal loop had no optimizer, so param grads (graph-derived, not in the snapshot) persisted
across `markStep`, were demoted, then read by the next step's accumulation. Fixed by mirroring
the census tools' loop shape — an `Adam` + `step()`/`zeroGrad()` each step supplies the boundary
discipline. Now **3/3 under strict DEFAULT** (no `TORCHLETTE_STRICT_LIFETIME=0` opt-out), with
`TORCHLETTE_WHOLE_STEP=1` to un-skip the remat cell.

**The 124M proof (production trainer, V100 sivri device 0).** Trajectory EXACT to baselines
{9.81, 5.92, 5.15, 4.64} in BOTH modes — remat 9.809/5.922/5.154/4.641, flag-off eager+refusal
9.809/5.922/5.154/4.641 — no shift, memory flat (zero leak). Speed/peak:

| mode | peak (all-time) | ms/round | note |
|---|---|---|---|
| remat (`TORCHLETTE_WHOLE_STEP=1`) | 3908.6 MB | ~1.7 s | 3× faster; peak budget-independent (4000 & 2200 MB pool → identical) |
| flag-off eager + refusal (default) | 2462.3 MB | ~5.3 s | arena-ON/lowered; the refusal keeps it sound (no +209% blowup) |
| (old bypass, deleted) | 2087.6 MB | ~5.3 s | arena-free/lowered — the reference |

Characterized (not a silent regression): at THIS production config (batch8/seq256) the remat
all-time peak (3908.6 MB) is HIGHER than the bypass's 2087.6 MB — the recompute-coresidency
cost is BATCH-driven (the whole-step merged plan co-locates forward + recompute + optimizer
buffers; at batch8 that is 8× the per-position activation). The D3 STEADY-peak WIN (94.4% /
86.2%) holds at batch1/seq512, where the D3 table measured it. The flag-off default rises
+18% vs the bypass (arena now ON), well short of the STOP's feared +209% current blowup — the
refusal-lowered path is well-behaved. The trajectory-exactness (the STOP condition) is met, so
the death lands; the memory shape is reported, not re-baselined.

### D3-finish fallout — the step-object census red light was the refusal, not a ghost (2026-07-19)

The D3-finish branch landed with `t-train-tape-matrix [fused, no-sched]` red
(`tapeCount 1→0`, `eligiblePairs 6→0`, `loweredPairs 0→15`). The finishing
agent hypothesised a "module-init / hidden-class / init-order interaction" from
the engine class-shape change (`_checkpointEagerForce`) and reported "the refusal
NEVER fires (refusals=0)." **Both are wrong** — verified by a deterministic probe
(`getCompileRefusalCount()` instrumented in the exact cell): the refusal fires
~5×/step. The "refusals=0" was a conflation of the STEP-TAPE `refusals` counter
(guard-3 byte-diff, genuinely 0) with the EXECUTOR `compileRefusalCount` (the D3
`CHECKPOINT_EAGER_REFUSAL`, ~28 over 6 steps).

**The mechanism, plainly.** The census/step-object gates run `forwardWithLoss(...,
{useCheckpoint: true})` in the eager (non-whole-step) loop — a checkpointed EAGER
step. `CHECKPOINT_EAGER_REFUSAL` (landed a78a5006) declines build-from-IR for EVERY
plan in such a step, all-or-nothing, so every plan stays LOWERED. The step-tape only
forms an eligible tape from fully-COMPILED steps (`stFinalizeStep`:
`rec.plans.every(pl => pl.compiled)`), so a checkpointed-eager step now yields
`loweredPairs`, never an eligible tape. Nothing to do with guard-5/`stNoteBoundary`
(`boundaryResets` unchanged) or the clear path. On main@026530a6 (pre-refusal) the
same cell compiled (`loweredPairs=0`, `tapeCount=1`); the doc's "eager checkpoint
plans were lowered by accident (uncovered CE)" is true only of the whole-step
build-from-IR boundary plan — the eager two-plan forward/backward/optimizer
materialise, so their CE resolves via the live-storage branch and they DID compile.

**Load-bearing, confirmed.** Env-gating the refusal off and re-running the mother
gate reproduces the b66ead78 corruption it exists to block (`t-whole-step-diff
CKPT=1 SELECTIVE=1` eager[29] 3.012908 → 3.013037, gate FAILs 1.3e-4). The refusal
is correct and its scope is sanctioned (the refusal spec asserts it FIRES for
checkpointed-eager). So checkpointed-eager forming a COMPILED tape and the refusal
are **mutually exclusive by design** — the census red light is an intended
consequence, not a bug in the D3 work.

**The fix — decouple the census from eager checkpointing (deletion named).** The
census's ACTUAL assertion is optimizer-config scalar-slot coverage via a compiled
tape (the frozen-scalar family, LR-via-LiveScalar); that is CHECKPOINT-INDEPENDENT.
Eager selective checkpointing (`useCheckpoint: true`) is **removed** from the four
step-object-family gates it broke — `t-train-tape-matrix`, `t-step-object-null`,
`t-step-edit-null`, `t-step-edit-binding-probe` — restoring `tapeCount=1 /
eligiblePairs=6 / loweredPairs=0` with zero census coverage lost. Not a gate-scope
game: the abandoned "checkpointed-eager compiles" property is not hidden — it is
asserted STAYS-LOWERED in `test/whole-step-checkpoint-refusal.spec.ts` and
COMPILES-under-whole-step in `t-whole-step-diff` / `t-d3-remat`. SUNSET: when
whole-step defaults (P4) and the refusal + eager two-plan path are deleted,
selective checkpointing returns to these loops. (`t-step-edit-numerics-null`
asserts numerical equality — tape-independent, unaffected; `t-ring-probe`,
`t-ledger-attack` never checkpointed — unaffected.)

**Gate re-run (this pass, V100 sivri device 0):** tape-matrix **4/4** PASS
(all cells `eligiblePairs=6 tapeCount=1`); `t-step-object-null` / `t-step-edit-null`
/ `t-step-edit-binding-probe` PASS; `t-ring-probe` PASS (K1/K2/K3); `t-ledger-attack`
default+48 PASS (ledger balanced, flat); `t-whole-step-diff` non-ckpt 2.62e-6 +
CKPT=1 SELECTIVE=1 3.58e-6 PASS; `test:gates` **5/5**; profile medium@512 flag-off
peak 13.69 GB, LEAK OK. Refusal spec **3/3** under the soak opt-out
(`TORCHLETTE_STRICT_LIFETIME=0`, as the finishing agent validated it).

**Pre-existing finding, flagged not fixed (out of the census root-cause).** Under
strict-lifetime DEFAULT, `whole-step-checkpoint-refusal.spec.ts` THROWS a reclaimed
read (the wpe position-embedding grad `[64,128]`, scatterAdd→add) in BOTH its
`eager-ckpt` and `eager-nockpt` cells — so it is refusal-INDEPENDENT (the nockpt
cell never triggers the refusal). Present on clean ea9a0ade; the spec's minimal
explicit-boundary loop (`sum()` loss, no optimizer, raw `new GPT2`) holds a tensor
across `markStep` that gets demoted. The sibling checkpoint specs (autocast-parity /
segmentation / scaler-seed) pass under strict default, and the census tools above
all run clean under strict default — so this is spec-loop hygiene in the new refusal
spec, not a framework UAF. Left for the branch owner (it papered over as "3/3" only
because the finishing agent ran it with the opt-out).

- **P4 — Guard reduction + deletion.** Replace the runtime guard taxonomy with input
  guards; then execute the deletion ledger (§5) subsystem by subsystem, each behind
  a green parity gate. *The payoff phase.*

Each phase names its deletions in the commit (house policy). Every new flag is born
with a sunset (soak → default → opt-out dies).

### P4a status — THE DECODE DISCRIMINATOR — VERDICT B (2026-07-19)

Everest (P1–P3) validated whole-step for TRAINING only. The step tape's other live
consumer is DECODE (phase-1's ~14.55 ms/token demo). P4a's load-bearing question:
**does a decode step run under `TORCHLETTE_WHOLE_STEP` as a whole-step-COMPILED
function** — such that the whole-step compiler could subsume the tape's decode
consumer and P4b could delete `step-tape*.ts` wholesale?

**VERDICT B — decode BLOCKS; whole-step is a structural NO-OP for decode.** The
tape survives DECODE-SCOPED in P4b.

*The structural cause (from code, not measurement).* The whole-step scope's ENTIRE
mechanism is `_deferBackwardForce()` (`WHOLE_STEP_TRACE && _wholeStepDepth > 0`),
whose only consumers are the backward-path force/teardown sites (`autograd.ts` lines
230/270/306/442/545/571) and the checkpoint unpack hook (`nn/checkpoint.ts:157`). A
decode step is a `noGrad` forward — no backward runs, so NONE of these fire.
`api.wholeStep(fn)` around a forward-only body reduces to
`_enterWholeStep(); fn(); _exitWholeStep()` — a depth counter with no consumer.
There is no fwd+bwd+opt multi-force to collapse: decode is ALREADY a single-force
forward, and the per-token logits readback IS its boundary.

*The measurement (`tools/t-decode-whole-step.ts`, distilgpt2 KV-decode, 16 steps,
V100 sivri device 0, two isolated-child arms — the flag is a module-load const).*
A real `forwardCached` KV-decode loop, each decode forward wrapped in `api.wholeStep`
under `TORCHLETTE_WHOLE_STEP=1` (arm `whole-step`) vs unset (arm `plain`):

| arm | templates (prefill→decode) | steady growth | ms/tok | tok/s |
|---|---|---|---|---|
| plain | 1 → 2 | 0 | 58.67 | 17.0 |
| whole-step | 1 → 2 | 0 | 55.45 | 18.0 |

Tokens **byte-identical**; template count **identical** (2 == 2 — the decode step
does NOT become a distinct whole-step-compiled function, it runs the SAME per-plan
compiled template); steady growth identical (0); tok/s ratio 1.058 (within
16-step measurement noise, no whole-step effect). Whole-step neither traces nor
compiles decode as its own function.

*The named blocking causes (Verdict-B characterization, per §6).*
1. **No backward ⇒ the mechanism is inapplicable.** whole-step's sole lever
   (deferring the backward grad-write force to merge fwd+bwd+opt into one boundary
   graph) has nothing to act on in a `noGrad` forward. Decode is structurally a
   single-force step already.
2. **Readback-per-token is definitional, not deferrable.** Each decode step reads
   its logits to choose the next token, and that VALUE is the next step's input. The
   loop is data-dependent on the readback; there is no boundary to defer past (unlike
   training's loss, which rides the ring and feeds nothing downstream in-step).
3. **The tape's decode consumer is a DIFFERENT mechanism.** Decode's acceleration is
   the per-plan template compiler (`#71` offset-is-data: steady template growth 0,
   BUILD-FROM-IR) and, when enabled, the step-tape REPLAY skeleton re-dress
   (`step-tape-replay.ts`, keyed by `bucketKey`, serving per-token uploads). Neither
   is touched or subsumed by whole-step's fwd+bwd+opt merge.

*The verdict-adjusted P4b ledger (§5), stated honestly.* The "static whole-step
graph makes the runtime-staticness engine unnecessary" thesis holds for the TRAINING
half of the ledger (observed-liveness / the training-side tape record+diff / the
guard taxonomy on training steps). It does **NOT** reach the decode half: whole-step
cannot delete `step-tape.ts` (820) / `step-tape-replay.ts` (680) / `step-object.ts`
(156) / `cross-plan-edges.ts` (152) / `tape-profile.ts` (18) — ~1826 SLOC — on the
grounds that "whole-step subsumes decode." Two honest paths remain for P4b to shrink
the decode half, each needing its OWN proof (NOT whole-step):
- **(P4b-decode-α)** prove the per-plan template compiler ALONE suffices for the
  decode demo (this pass already shows plain per-plan decode at steady growth 0 and
  17 tok/s WITHOUT the step tape) — if the tape adds no measurable decode value over
  per-plan compile, it is deletable on THOSE grounds (per-plan compile survives the
  ledger as "the flat command-stream builder + slot table"). This is a decode
  benchmark P4b must run (tape-replay vs plain per-plan tok/s on a real weighted
  model), not assumed.
- **(P4b-decode-β)** if the tape DOES beat per-plan compile for decode, it survives
  **decode-scoped** — the training-side deletions still land, but `step-tape*.ts`
  stays as the decode replay path. The ledger shrinks honestly by the training half,
  not the full ~2633 outright.

The verdict is the deliverable (no forcing): whole-step is training-shaped; decode is
not a training step; the two do not meet.

### P4a Stage 2 — GRADUATION TO DEFAULT-FOR-TRAINING (2026-07-19)

`WHOLE_STEP_TRACE` flips from opt-IN (`=== "1"`) to opt-OUT (`!== "0"`) in
`src/core/step-tape.ts` — whole-step is now the **DEFAULT** wherever a training
step enters the scope (the capture ring body, which always enters; the trainer's
checkpointed inner step; any `api.wholeStep(...)`). A plain training loop that never
enters the scope is untouched (the eager reference, reachable via `=0` everywhere).
DECODE is unaffected by construction (Verdict B: the scope is a no-op without a
backward). The flag is **sunset-listed for P4b** (dies with the eager two-plan path).

*Eager-vs-traced seams fixed for the inverted default* (unset now means ON): the
differential tools' eager/bypass arms set `TORCHLETTE_WHOLE_STEP=0` explicitly
(`t-whole-step-diff`, `t-d3-remat`) rather than unsetting; the refusal spec's remat
cell (`it.skipIf(!WHOLE_STEP_TRACE)`) now **runs by default** (skipped only under
`=0`).

**The gate matrix — default-ON, V100 sivri device 1, strict-lifetime DEFAULT
(no opt-out).** All green; the eager arm is `=0` (true opt-out), so every
traced-vs-eager comparison measures the graduation:
- **Mother differential** (`t-whole-step-diff`): non-ckpt traced==eager **2.62e-6**;
  CKPT=1 SELECTIVE=1 traced-remat==eager **3.10e-6**; step-0 loss bit-identical. Both
  ≤ 1e-5.
- **`parity-fullstack-tl`** (autocast+checkpoint+scaler+clip, 30 steps): compiled==
  lowered **6.68e-6**.
- **`test:gates`** (`compiled-plan-parity`): **5/5**.
- **Refusal spec** (`whole-step-checkpoint-refusal`, strict DEFAULT): **3/3** — the
  remat cell un-skips and passes (refusals=0), eager-ckpt FIRES (6), eager-nockpt 0.
  Under `=0` the remat cell SKIPS (2 pass / 1 skip) — the opt-out cleanly restores
  the pre-graduation path.
- **Checkpoint specs**: `checkpoint-autocast-parity`, `checkpoint-segmentation`,
  `checkpoint-scaler-seed-lifetime`, `distilgpt2-checkpoint-memory`,
  `gpt2-checkpoint-amp` — all pass; `implied-step-boundary` **6/6**.
- **Tape apparatus**: `t-train-tape-matrix` **4/4** (fused/foreach × no-sched/cosine-lr,
  eligiblePairs=6 tapeCount=1 zero refusals); `t-ring-probe` PASS (K1/K2/K3 bit-identical
  — ring depth a pure knob); `t-step-object-null` / `t-step-edit-null` /
  `t-step-edit-binding-probe` / `t-step-edit-numerics-null` all null-clean
  (≤1e-5/24-step). `t-ledger-attack-probe` default (24 steps) flat (reachDrift=0,
  totalDrift=0, no double-release); the STEPS=48 stress shows drift **identical under
  `=0`** (pre-existing config artifact, NOT whole-step-attributable).
- **cpu-FULL suite** default-ON: **1401 passed**, 37 failed — every failure is a
  `torch`-oracle test (`ModuleNotFoundError: No module named 'torch'`, this box lacks
  PyTorch) plus one distributed `websocket-relay` flake. **Zero whole-step-attributable
  failures.**
- **BROWSER suite** (Playwright chromium under `xvfb-run`, WebGPU via
  `--enable-unsafe-webgpu`, whole-step DEFAULT-ON — the browser reads flags as
  undefined ⇒ `!== "0"` ⇒ ON, the never-before-run case): `webgpu.spec` **9/9**,
  `scope-surface.spec` **2/2** = **11 pass**. The browser WebGPU substrate runs clean
  under whole-step default-ON.
- **profile distil@512** default-ON: **LEAK OK, +0.0 MB/step** (flat).

**Honestly un-run / named gaps** (the soak continues — the opt-out survives P4a, so
these complete before P4b deletes it):
1. **Browser TRAINING-trajectory under whole-step** — `lora-training-trajectory.spec.ts`
   fails to import (`examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora` is
   **absent from the repo entirely** — only `trainer.ts` exists). Pre-existing and
   **flag-independent** (an import error at collect time, before any flag is read; it
   fails identically at `=0`). The browser DEFERRAL path is the capture ring
   (`capture.ts`, backend-agnostic — the exact code node validates), but a
   browser-native training trajectory under whole-step remains UNVALIDATED until this
   spec's missing module is restored. **This is the one real gap and a P4b precondition
   for deleting the opt-out.**
2. **Full 114-file webgpu suite** — impractical serially on this box (~1–3 min/file via
   the Dawn/Vulkan vk-shim). Validated the whole-step-SENSITIVE subset (above) instead;
   the whole-step-insensitive backend specs (matmul/elementwise/reduction) don't enter a
   training scope and are deferral-agnostic by construction.
3. **medium@512 profile / 124M trajectory** — profile-training does NOT enter the
   whole-step scope (plain begin/markStep, no `wholeStep` wrap), so default-ON is
   identical to `=0` for it (eager-checkpoint via the refusal); the whole-step remat
   path at scale is already gated by `t-d3-remat` (D3 table) + the 124M proof (D3-finish,
   both modes exact to baselines). Re-running them is a measurement, not a new-risk gate.

### P4a Stage 3 — SUNSETS: NONE RIPE (verified, not reached)

The only Stage-2-adjacent sunset candidate is restoring `useCheckpoint: true` in the
four census tools (`t-train-tape-matrix`, `t-step-object-null`, `t-step-edit-null`,
`t-step-edit-binding-probe`) that removed it (D3-finish fallout). **Verified NOT ripe.**
The doc's sunset condition is "whole-step defaults **AND** the refusal + eager two-plan
path are deleted"; Stage 2 meets only the first. The census tools run **plain eager
loops** (no `wholeStep` wrap) — default-ON does NOT route them through the scope, so a
checkpointed cell is still eager-checkpoint → `CHECKPOINT_EAGER_REFUSAL` fires (verified:
the refusal is gated on `!_deferBackwardForce()`, true at depth 0 REGARDLESS of the flag)
→ plans stay lowered → no compiled tape → the census assertion (`eligiblePairs=6
tapeCount=1`) would break exactly as before. Restoring `useCheckpoint` here needs P4b's
refusal deletion (or re-shaping the tools to wrap in `wholeStep`, which changes what the
census measures). The refusal + eager-two-plan path + engine bypass method + the big
ledger all WAIT for P4b per §5. No sunset fires in Stage 3.

### Risks (honest)

- **Plan-builder scale** (P0). Bounded, named, amortized once-per-compile — but the
  O(n²) spots are real at medium-scale. Mitigation: P0 is a prerequisite, gated on a
  synthetic large-graph build.
- **Remat-pass correctness = the #97 class, at compile time** (P3). The recompute
  subgraph must be numerically + RNG-identical. Mitigation: the differential gate
  runs recompute-on vs -off; the pass refuses (falls back to keep-all) any activation
  whose recompute it cannot prove pure.
- **The two-frontend question is a non-issue** — precisely because laziness *is* the
  tracer, there is one frontend and one op semantics; the compiled step is built by
  the same nodes eager executes. The real maintenance surface is **one**: the passes.
  The eager path is the reference, not a parallel implementation.
- **Data-dependent structure** (inf-skip changing which nodes run) — handled by the
  variant set / eager fall-back (§5), not inside a single `S`.

---

## 5. THE DELETION LEDGER

End-state, the whole-step compiler deletes the runtime-staticness engine. SLOC are
code-only (comment/blank stripped), from the subsystem audit. **This is the largest
deletion ledger in the project's history — by a wide margin.**

### Deleted outright (~2633 SLOC)

| Subsystem | File | SLOC |
|---|---|---|
| Observed-liveness / convergence stamping (over-harvest → `K=3` rebuild) | `src/executor/observed-liveness.ts` | 807 |
| Step tape (record/diff/`bucketKey`) | `src/core/step-tape.ts` | 820 |
| Step-tape replay (skeleton re-dress) | `src/executor/step-tape-replay.ts` | 680 |
| Step-object (whole-step-as-object view) | `src/core/step-object.ts` | 156 |
| Cross-plan edges / witness apparatus | `src/core/cross-plan-edges.ts` | 152 |
| Tape profiling counters | `src/core/tape-profile.ts` | 18 |
| **Subtotal** | | **≈2633** |

The witness apparatus (`K_w=2`), the K_w per-producer keep-sets, and the tape-side
guard taxonomy live *inside* `step-tape.ts` / `step-tape-replay.ts` / the deleted
`cross-plan-edges.ts` — they go with those files, not double-counted.

**⚠ VERDICT-ADJUSTED by P4a (2026-07-19) — the decode half of this table is NOT
whole-step-deletable.** The "Deleted outright" total assumed the whole-step graph
subsumes BOTH tape consumers (training AND decode). The P4a decode discriminator
(VERDICT B) refutes the decode half: whole-step is a training-only mechanism (it
defers the backward force; decode has no backward), so it cannot delete
`step-tape.ts` (820) / `step-tape-replay.ts` (680) / `step-object.ts` (156) /
`cross-plan-edges.ts` (152) / `tape-profile.ts` (18) — ~1826 SLOC — on
"whole-step subsumes decode" grounds. The **~807 SLOC** `observed-liveness.ts`
(training-side over-harvest/convergence) still falls to the whole-step compile; the
tape files fall ONLY if P4b independently proves either (α) the per-plan template
compiler alone suffices for the decode demo (a decode benchmark, tape-replay vs plain
per-plan on a weighted model), or (β) accepts them surviving DECODE-SCOPED. Revised
outright deletion under whole-step alone: **~807 SLOC guaranteed** (observed-liveness)
+ the partial-deletion training-side reductions below; the ~1826 SLOC tape subsystem
is **gated on the P4b decode proof, not on the whole-step compile.** See the P4a
status section for the discriminator evidence.

### Deleted partially (~1100–1900 SLOC) — files survive as static representation / plain allocator

| Subsystem | File | ~SLOC removed | What survives |
|---|---|---|---|
| Harvest + witness-stamp seams | `src/executor/executor.ts`, `src/executor/compiled-plan.ts` | ~400–600 | the flat command-stream builder + slot table |
| Arena runtime identity-stabilization | `src/backend/webgpu/buffer-arena.ts` | ~150–250 | arena-as-storage |
| Pool runtime-adaptive trim/liveness/reclaim | `src/backend/webgpu/buffer-pool.ts` | ~150–250 | plain allocator for the transient path |
| Memory planner per-step epoch churn | `src/executor/memory-planner.ts` | (simplifies) | becomes the sole global planner |

### Ledger total: **~3700–4500 code-only SLOC removed.**

Plus the runtime *half* of the guard taxonomy and the entire **observed-liveness /
captured-by-default** posture: training becomes **captured-by-default** (every step
is traced), so the uncaptured-path recovery machinery has no reason to exist.

**Explicit non-deletions** (survive — different concern): `cache-key-guard.ts` (68
SLOC, guards *kernel-codegen* caches, orthogonal to step staticness) and
`ops/registry.ts` (515 SLOC, the op table). Do not conflate these with the runtime-
staticness engine.

If this ledger did *not* dwarf every prior campaign, the thesis would be wrong — the
whole point is that a static whole-step graph makes an entire class of runtime
learners unnecessary. It does dwarf them: ~4k SLOC vs the largest prior single
campaign (the recorded-build sunset, −822 SLOC).

---

## 6. Usage-pattern boundary (from Vin's interview)

- **Dynamic shapes** → **per-shape traces** (bucketing). Each distinct input-slot
  shape gets its own compiled `S`, keyed by the shape guard. **Symbolic shapes are
  an explicit non-goal for v1** — no shape polymorphism inside a trace.
- **Structural data-dependence** (control flow that changes *which* nodes run — e.g.
  a GradScaler inf-skip that skips the optimizer, or early-exit) → **variant set**
  (a small set of compiled steps, guard-selected) **or eager fall-back** for that
  step. Not expressed inside one `S`.
- **Mid-step pokes** (user reads/writes a tensor mid-step) → **eager for that step**.
  The step de-optimizes to eager; no partial compile.
- **Undeclared mid-step readbacks** → **typed trace break**. Reading a value that is
  not a declared output mid-step is the defining violation of the contract; it
  raises a typed break, not a silent divergence.
- **MoD / capacity routing** → **in scope**. Routing indices are *data* (gather/
  scatter with computed index tensors); the graph expresses them natively. Only
  routing that changes the *node set* (not just index values) is data-dependent
  structure and falls to the variant/eager rule above.

---

## 7. Red-team

**Objection 1 — "You are rebuilding the tape you spent months building; this is
churn, not progress."** *Ruling: rejected.* The tape *learns* staticness by
recording and diffing runtime executions; the compiled step *computes* it by
dataflow analysis. The tape's `bucketKey`/witness/convergence machinery are all
approximations of facts the whole-step graph states exactly. The compiled step is
the tape's terminus, and it deletes the approximation. The months were not wasted —
they eroded the mid-step forces (loss ring, scale-as-data, LiveScalar LR) that this
design *requires*; without them there would be no single-force step to compile. This
is the arc completing, not restarting.

**Objection 2 — "The plan-builder is O(n²) in four places and recursive; a
5–15k-node medium graph will blow it up."** *Ruling: conceded as real, ruled
non-fatal.* It is P0, gated on a synthetic large-graph build, and — critically — it
is **compile-time cost paid once per (shape,config)**, not per step. The current
system already pays these passes on a 507-node plan *every step*; moving them to a
once-per-trace compile that produces a replayable artifact is a net *reduction* in
amortized pass cost, even before hardening. The hardening (explicit stack, heap ready
set) is standard and bounded.

**Objection 3 — "Remat-as-a-pass will silently miscompute gradients the way runtime
checkpointing never did, because a compile-time bug is invisible until it corrupts a
long run."** *Ruling: this is the sharpest objection; addressed structurally.* This
is the `#97` class (recompute must be numerically + RNG-identical) moved to compile
time. The defense is the same one that caught every silent-divergence bug in this
project: **differential-first, across the optimization's activation threshold**
(`architecture-debt.md` Corollary 2). The parity gate runs the compiled step
(remat on) against eager (remat off) for ≥30 steps and byte-matches the trajectory;
the remat pass *refuses* (keeps the activation live) any subgraph whose purity +
RNG-determinism it cannot prove at compile time. A step that cannot be proven is
correct-and-slow, never wrong-and-fast — the zero-residue posture, extended.

### Open questions (only those that materially fork the design)

1. **Q4 from the charter — is the compiled step a *static* artifact or a
   *live-recompiling* one?** i.e. when a guard misses, do we recompile eagerly on the
   hot loop (Dynamo-style, a hitch) or async in the background (replay eager until the
   new `S` is ready)? This forks the execution contract's latency profile. The design
   assumes eager-fall-back-then-recompile (no async compile) for v1; async compile is
   a v2 option. *Needs a ruling if medium-scale compile time (P0) proves to be
   hundreds of ms.*

2. **Variant-set vs eager-fallback threshold for data-dependent structure (§5).** How
   many structural variants (inf-skip × ...) do we compile before declaring "this
   step is too dynamic, run eager"? A policy knob, not an architecture fork — but it
   should be declared, not discovered.

Everything else (shape bucketing, the deletion order, the pass list) is determined by
the design and does not need a ruling.

### One-sentence test

*If a feature of the step cannot be expressed as a node in the whole-step graph or a
guard on the step's declared inputs, it is a trace break by declaration — not a
special case inside the compiled step.*

---

## Appendix — reproduction

- Probe: `probe/census.ts` (this branch). Run:
  `VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim npx tsx probe/census.ts`
- Base: `main@b0b4e7c3`. Branch: `step-function-compiler-design`.
- Numbers (§2) are distilgpt2@seq15, A100 (dw-2-1), 4 warmup + 6 steady steps,
  default engine config. The whole-step node count (727) triangulates two
  independent methods: the census pending-node deltas (237+408+82) and the
  plan-builder audit's per-segment node counts (223+507). The 630-node merged
  fwd+bwd plan is the DEFERRED-LOSS config's single measured `buildMergedPlan`.
