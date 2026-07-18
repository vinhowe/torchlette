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

- **P0 — Scale the passes (no behavior change).** Convert `buildMergedPlan.visit`
  recursion → explicit stack; heap-back the Kahn ready sets; bound the fusion gap
  scan and checkpoint all-pairs edges. Gate: existing suite green + a synthetic
  5k-node graph builds in bounded time. *Ships nothing user-visible; derisks the
  whole-step graph.*

- **P1 — Whole-step trace acquisition behind a flag.** Capture scope that defers
  loss/grad-write/boundary forces and forces the step output set once. Run
  eager-lowered (no compile yet). Gate: parity vs today's two-plan path, byte-exact;
  census re-run shows 4→1 mid-step forces (checkpoint off). *This is the probe's
  DEFERRED-LOSS config, productized.*

- **P2 — Compile the whole-step graph.** Lift build-from-IR from per-plan to
  per-step: one `CompiledPlan` for `S`, one global memory plan (planner over the
  whole graph). Gate: the parity gate + memory flat + submits/step drops. *First
  point the deletion ledger starts paying out (observed-liveness convergence can be
  bypassed for compiled steps).*

- **P3 — Remat as a pass.** Move checkpointing from unpack-hook forces to
  scheduled recompute in the whole-step stream. Gate: parity with checkpoint on,
  memory ≤ today's checkpointed peak, RNG-identical recompute proven at compile time.
  *Deletes the structural mid-step force.*

- **P4 — Guard reduction + deletion.** Replace the runtime guard taxonomy with input
  guards; then execute the deletion ledger (§5) subsystem by subsystem, each behind
  a green parity gate. *The payoff phase.*

Each phase names its deletions in the commit (house policy). Every new flag is born
with a sunset (soak → default → opt-out dies).

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
