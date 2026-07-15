# Staged Execution, Phase 2b: `capture()` of a whole training step

**Status:** APPROVED (2026-07-08, coordinator review — all six per-section
rulings as proposed; dispositions: G-cover = TAG_UNIFORM/recorder-coverage
extension, NOT generated-stream forcing; NO ceremony bridging — minimal/implied
loop is the API constraint, the #81 starvation warning lands with inc-2;
TAPE_VERIFY-excludes-volatile-bytes is a numbered implementation gate; K is a
capture OPTION, default 2, no pressure-reactive automation, K=1==K=2 bit-parity
gated). Implementation staged: inc-1 G-cover → inc-2 whole-step capture →
inc-3 ring. **INCREMENT 1 LANDED — see the INC-1 section at the end.**
**Builds on:** phase 1 (`docs/staged-execution-phase1.md`, esp. §9 — the
probe-unsoundness rule and the arg-boundary contract), phase 2a
(`docs/staged-execution-phase2a.md` — the `capture()` API + staging ring),
`docs/outer-api-sketches.md` (sketches 2/5), stage-4 stage-3
(`docs/stage4-compile-from-ir.md` — compiled-checkpointed training is now the
2.1× path). **Correctness bar:** inherits phase 1's — a silent stale replay is
THE failure mode; the G3-class value-coverage gate is never waived.

**The goal (Vin-authorized):** a training step — forward, backward,
`optimizer.step` — as ONE `api.capture()` call under the arg-boundary contract,
completing execution-as-declaration (decode is already captured). Strategic
dividend beyond speed: inside a DECLARED training boundary, lifetimes are
knowable at capture time — §5 is the extension point where declared lifetimes
begin retiring the observation-layer predicates (`everSurvived`/`everReadback`/
`everAliased`) on the training path.

---

## G0 — DECOMPOSED MEASUREMENT FIRST (the stage-3 doctrine)

Measured on the V100 box (Node/Dawn), seq 512, batch 1, default stack
(compiled + memory planner + arena-liveness + generated cutover), autocast f16
+ gradient checkpointing + GradScaler + grad-clip + AdamW — the fullstack inner
step (`tools/parity-fullstack-tl.ts` shape). Instrumented with the existing
`TORCHLETTE_TAPE_PROFILE=1` JS-seam timers (`src/core/tape-profile.ts`) and
per-phase wall clocks. **The box was SHARED with other users' compute during
the 124M run (SMs 65–88% busy from foreign jobs); GPU-time absolutes there are
contention-inflated — read them for STRUCTURE, not as steady-state baselines.
The distil run was near-solo.** All numbers steady-state (post-warmup, plan
cutover settled). These are V100 and NOT comparable to the A100 CLAUDE.md
baselines; they exist to decompose the GAP, per doctrine.

### G0(a) — Wall-time decomposition of a compiled-checkpointed training step

| phase (CPU wall) | distil (6L/768, near-solo) | 124M (12L/768, CONTENDED) |
|---|---:|---:|
| **wall / step** | **~236 ms** | **~403 ms** |
| forward build + `loss.item()` (forces forward GPU + readback) | ~42 ms | ~44 ms |
| backward build (lazy; does not fence) | ~26 ms | ~47 ms |
| optimizer+clip+scaler build (lazy; does not fence) | ~2 ms | ~4 ms |
| **markStep (fence of queued bwd+opt GPU + boundary sweep)** | **~165 ms** | **~308 ms** |

The step has TWO serialization points today: `loss.item()` (forces the forward
+ reads the scalar back) and `markStep` (fences everything queued by backward +
optimizer, then sweeps). Backward and optimizer build lazily and their GPU work
lands in the `markStep` fence — which is why `markStep` dominates.

**`markStep` is GPU-fence-dominated, not CPU-sweep-dominated — proven by
contention scaling.** When the box was contended (124M), `markStep` ballooned
166→308 ms while the CPU-side build/sweep phases barely moved; a CPU sweep would
not scale with foreign GPU load. So `markStep` ≈ the GPU-serialized compute of
backward+optimizer, plus a smaller CPU sweep residue on top.

### G0(a) — the tape-skippable JS subtotal (what a tape HIT removes)

Accumulated `TAPE_PROFILE` seams the step-tape skips (graph build / plan-collect
/ fingerprint / CSE + rewrite passes / template lookup / consumer-maps):

| skippable seam (sum over all plans in the step) | distil | 124M |
|---|---:|---:|
| plan-collect | 1.1 | 2.1 |
| dsl-rewrite | 1.4 | 2.7 |
| fingerprint | 2.2 | 4.2 |
| consumer-maps | 1.2 | 3.7 |
| pass:cse | 1.2 | 2.5 |
| pass:dce + small passes + template-hit | ~0.5 | ~1.3 |
| **SKIPPABLE SUBTOTAL** | **~7.6 ms** | **~16.5 ms** |
| (replay-total JS+Dawn, NOT skipped — the tape still calls it) | ~25 ms | ~47 ms |
| (lowered-exec — steady-state near-0 ⇒ compiled path active) | ~3 ms | ~5 ms |

**The theoretical ceiling of a tape HIT's DIRECT (build-skip) speedup:**

| config | skippable JS | as % of wall | decode reference (1a) |
|---|---:|---:|---|
| distil training | ~7.6 ms / 236 ms | **~3.2 %** | — |
| 124M training | ~16.5 ms / 403 ms | **~4.1 %** | — |
| decode (1.7B, 1a) | 14.55 ms / 45 ms | **~32 %** | the 43→~30 ms win |

**HONEST CEILING — the headline.** Decode's win was build-skip (32 % of a
JS-bound token). Training is the OPPOSITE regime: it is GPU-fence-bound, and the
build JS a tape removes is only 3–4 % of the wall. **2b's win is NOT skipping
build. If 2b were only a build-skip tape it would not clear the §4-style
5-ms/step hard-stop as a fraction and the campaign would pause.** 2b's real win
is RUNAHEAD — G0(b).

### G0(b) — the runahead ceiling (where 2b's real win lives)

Today `loss.item()` fences the forward every step and `markStep` fences the
tail every step. Under a capture ring that DEFERS the loss readback (loss
becomes a materialized ring output, read on the logging cadence, not per step),
the CPU can build+submit step N+1 while the GPU still drains step N. Then:

    wall_floor  ≈  max( GPU_per_step , CPU_overlappable_per_step )

- **GPU_per_step** (serialized regardless — the tape cannot touch it): ≈ the
  `markStep` fence, ~165 ms distil.
- **CPU_overlappable_per_step**: wall − GPU ≈ 236 − 165 ≈ **~71 ms** (distil):
  forward/backward/optimizer graph build (~45 ms of §G0a build phases minus the
  forward GPU inside `loss.item`) + replay JS (~25 ms) + sweep.

Because CPU (~71 ms) < GPU (~165 ms), a ring of depth **K = 2** (one step in
flight) is sufficient to hide ALL CPU behind the GPU fence — the wall collapses
to the GPU floor:

    236 ms  →  ~165 ms   ≈  30 % wall improvement (distil), from runahead ALONE.

The build-skip (G0a, ~7.6 ms) then matters only as INSURANCE that keeps CPU
below the GPU bar (it widens the K=2 margin; it is not the win). K > 2 buys
**zero** throughput once GPU-bound — it only adds memory (K steps of in-flight
uploads + activations + ring outputs). So the training ring depth is small
(2–3), chosen for pipeline fill, not for a JS-hiding deficit. **This inverts the
2a default (ring=3 for output validity): here K is a runahead/memory knob.**

Ceiling per config, honest: distil ~30 % wall (GPU-bound floor); 124M's contended
number is not a clean baseline, but the STRUCTURE (markStep 76 % of wall,
skippable 4 %) says the same — runahead-to-the-GPU-floor is the entire prize,
and it shrinks toward zero as the config becomes more fence-bound (large model /
iGPU), exactly as phase-1 §0 warned for decode's fence pole.

### G0(c) — the variance census (the frozen-scalar-family defense)

Every per-step-varying value in the measured step, classified under the
arg-boundary contract. The frozen-scalar disease (6 instances to date) lives
here; the census is the structural defense.

| # | varying value | today's carrier | 2b classification | covered? |
|---|---|---|---|---|
| 1 | batch `x`, `y` (input/target ids) | per-step `tensorFromArray` (fresh pending node) | **WARM SLOT** (tensor arg → TAG_WRITE stable buffer) | ✗ today (see falsification) — capture arg contract fixes |
| 2 | loss value (readback) | `loss.item()` per step | **RING OUTPUT** (materialized; deferred readback) | n/a (output, not input) |
| 3 | Adam bias-corrected `step_size` (per-param) | `adamStep` node PAYLOAD → TAG_UNIFORM repack ON THE GENERATED STREAM ONLY | **VOLATILE DATA** ("payload" slot; TAG_UNIFORM re-pack from node payload) | ✗ today (see falsification) — the load-bearing gap |
| 4 | Adam `step` counter | inside the adamStep config payload | folded into #3 | ✗ (same) |
| 5 | GradScaler `scale` / `inv_scale` | scaler state tensor + unscale config | **VOLATILE DATA** (persistent scale tensor read live; inv_scale a TAG_UNIFORM) | must be declared (§3) |
| 6 | GradScaler found-inf predicate | GPU `where`/`isfinite` (data, not branch) | **DATA** (never a CPU readback in the hot loop) | idiom exists; §3 |
| 7 | scheduled LR (if a scheduler runs) | LR-as-graph-scalar (SGD-alpha fix) → scalar-table / TAG_UNIFORM | **VOLATILE DATA / scalar slot** | mechanism exists; §4 |
| 8 | RNG philox offset (dropout; off here) | in-place counter state (data) | **DATA** (in-place state, replays correctly) | phase-1 §sketch-6 |
| 9 | model params, Adam m/v | in-place persistent state | **CLOSURE STATE, mutated in place** (NOT args) | §1 contract |
| 10 | window offset / data cursor | JS closure in the driver | irrelevant (produces #1 as a tensor arg) | n/a |

The census is only as good as its enforcement. The recorder's guard-3 byte-diff
REFUSES any per-step-varying byte not covered by a declared slot — that
enforcement is what the falsification probe below exercises, and it found the
two ✗ rows empirically.

---

## FALSIFICATION DUTY (the S3.0 lesson institutionalized) — RUN, with result

**Riskiest assumption:** *"the whole step, including the optimizer, records and
replays as stable templates under the implied-boundary machinery"* (sketch-2's
implicit premise; if false, there is no training tape to make warm and 2b is
mis-scoped).

**Cheapest experiment that could falsify it:** run the step-tape RECORDER
(`TORCHLETTE_STEP_TAPE=record`) over a real training loop (varying batch each
step) and read `stStats()` — does an eligible tape form, or is it refused? A ~90-
line probe (built on the fullstack inner step; deleted after — no new tracked
files). Two loop regimes tested: explicit `beginStep/endStep` and the MINIMAL
implied-boundary loop (the premise's exact subject).

**RESULT — the premise is FALSIFIED as currently built:**

- **Explicit `beginStep/endStep` loop:** `eligiblePairs = 0`, `tapeCount = 0`,
  `boundaryResets = 36` (beginStep 18 + endStep 18). The explicit ceremony fires
  `stNoteBoundary` every step, nulling the recorder's consecutive-step
  comparator (guard 5) — so a tape can NEVER form with explicit boundaries. A
  training capture MUST run the minimal/implied-boundary regime (or the recorder
  must learn to bridge explicit boundaries — a design choice, §1).

- **MINIMAL implied-boundary loop, varying batch:** `stepsObserved = 18`,
  `eligiblePairs = 0`, `structureMisses = 7`, **`refusals = 556`**, `tapeCount =
  0`. The refusals decompose into exactly two classes (diagnostics captured):
  1. **`tensorFromArray` uploads (the BATCH)** — `plan[1] node[21]` (input/
     target) and `plan[2] node[1]/[15]` (loss-seed / scaler scale) —
     *"PAYLOAD varies step→step but position is not TAG_WRITE-covered"*.
     EXPECTED, and capture's arg contract fixes it: the raw loop builds a fresh
     pending `tensorFromArray` each step (an undeclared upload); routed as a
     tensor ARG it rides the TAG_WRITE warm-slot path (census #1).
  2. **`adamStep` node payloads (per param)** — `plan[3] node[460…488…]`
     (hundreds of them) — *"PAYLOAD varies step→step but position is not
     TAG_WRITE-covered"*. This is the bias-corrected `step_size` / step counter
     (census #3/#4). The recorder's TAG_UNIFORM observation seam EXISTS and is
     wired (`compiled-plan.ts:1802 stRecordUniform` on TAG_UNIFORM commands), but
     it does NOT cover these adamStep positions — i.e. under the minimal loop the
     optimizer island's per-param configs are NOT reaching the recorder as
     TAG_UNIFORM (their varying step_size rides the fused-adam node payload,
     uncovered), so the byte-diff refuses them as undeclared variance.

**What the falsification BUYS the design (it reshapes it, exactly as intended):**
2b is NOT "make the training tape warm" — the tape does not even FORM today. 2b's
first load-bearing prerequisite is **CLOSING THE OPTIMIZER-CONFIG COVERAGE GAP**:
every per-step-varying `adamStep`/`unscaleGrad` config position must be a
DECLARED "payload" slot (TAG_UNIFORM volatile-repacked from the current node
payload) in BOTH compared steps — i.e. the optimizer island must run as a
GENERATED plan whose configs are TAG_UNIFORM, and the recorder must observe
every one. The mechanism is proven (the frozen-step_size fix,
`setAdamConfigUniforms`, the "volatile uniforms" campaign) — it just is not
reaching the recorder for the per-param fused adam under the implied boundary.
Until that is closed, `TORCHLETTE_STRICT_TAPE=1` correctly throws and the fallback
path (correct, slow) is all a training capture would ever get. This is the §4
frozen-scalar family's SEVENTH instance, caught at design time by the census's
own enforcement — the doctrine working.

**Corollary the probe also proved:** the recorder is a faithful pure observer of
training (18 steps observed, no crash, diagnostics precise) — the substrate is
sound; the gap is coverage, not observation.

---

## DESIGN DELIVERABLES

### 1. WHOLE-STEP CAPTURE SEMANTICS

**No short-circuit — the whole step IS the tape.** In decode, a HIT throws the
short-circuit sentinel at the last upload (before any layer/in-place op), so the
body never runs its compute. Training CANNOT do this: the optimizer's in-place
param/m/v/scale updates sit at the END of the step (phase-1 §9.1 + 2a §6). A
short-circuit before them would skip the state mutation; a short-circuit after
them would run the whole body (defeating the point) AND re-trigger the
probe-unsoundness rule (in-place version commits at graph build cannot be rolled
back). **Ruling: a training capture has NO short-circuit. The HIT path is an
UPFRONT slot-check → full tape replay** (write warm slots → replay the recorded
plan sequence — forward, backward, optimizer islands — via the existing
`executeLoweredPlan` skeleton, exactly as decode's replay does, minus the body
re-run). The body function is NOT executed on a hit; its declared plan sequence
is replayed. This is sound because a training step, unlike decode, has NO
in-body JS side effects that the replay must reproduce (no KV `staticKV.len`
advance): all state lives in in-place tensor ops INSIDE the recorded plans,
which the replay re-executes. (If a user puts arbitrary JS in the body — a
counter, an fs write — it is FROZEN by the closure contract; documented, tested,
TAPE_VERIFY-backstopped, same as 2a.)

**Params are CLOSURE STATE mutated in place — NOT args. State this contract
explicitly.** The model + optimizer are captured by closure; their param / m / v
/ scale storages are persistent (in the step snapshot) and are updated IN PLACE
by the recorded optimizer plan. Consequences: (a) the tape references them as
persistent external plan inputs whose stable buffers the replay reads/writes
live — never re-uploaded, never re-dressed; (b) their VALUES legitimately change
every replay (that is training) and this is NOT a guard miss — the guard is on
STRUCTURE and on DECLARED-slot coverage of per-step SCALARS, never on param
data; (c) replacing an optimizer (new Adam instance) or a param tensor is a
STRUCTURE change → invalidation (§6). This is the inverse of decode's frozen
closure: decode freezes closure VALUES; training's closure holds mutable STATE
whose in-place mutation is the whole point, and the contract is that state
survives via registration (sketch-5 seam 4), not via args.

**Batch (x, y) = warm slots.** `trainStep(x, y)` with `x,y` tensor args → the
arg-boundary contract routes them to TAG_WRITE stable-buffer upload slots (census
#1). This is what the raw loop lacked (falsification class 1). Caller builds them
fresh per call (donated on hit, 2a §upload-arg-donation).

**Backward + optimizer as replayable within one body — what capture adds beyond
the recorder.** The recorder already observes backward and optimizer as compiled
plans (they run as plans today — stage-3 made checkpointed training the compiled
path). Capture adds NOTHING to their EXECUTION; it adds (i) the ARG BOUNDARY (no
hand-rolled appKey; batch args → warm slots), (ii) the READINESS + guard
decision at ONE seam (upfront, before the body would run), and (iii) the RING
(§2) that turns the deferred loss readback into legal runahead. The recorder's
job — proving the backward+optimizer plan sequence is byte-stable modulo declared
slots — is a PREREQUISITE 2b must first satisfy (falsification), not something
capture provides.

**Gradient accumulation — the rule, with reasoning.** Backward accumulates into
`.grad` (PyTorch semantics, e9f7943). Two shapes (sketch-5): (a) ONE capture per
micro-step body `(x,y) => loss.backward()` + a SEPARATE capture for `() => {
opt.step(); opt.zeroGrad(); }`; (b) one capture per full accumulation cycle.
**Ruling: (a) — separate `micro` and `apply` captures.** Reasons: (i) the micro
body is STRUCTURALLY IDENTICAL across the N accumulation micro-steps (same plan,
different batch warm slot) → ONE warm tape replayed N times, maximal reuse; (ii)
the `apply` body runs once per cycle with a DIFFERENT structure (optimizer +
zeroGrad) → its own tape; (iii) mixing them into one capture would make the
tape's structure depend on N (a plain-value → cold bucket per N) for no benefit.
The in-place `.grad` accumulation is closure state (§1 param contract) — the
micro replay accumulates into the persistent `.grad` buffers exactly as the
lowered path does. `zeroGrad` inside `apply` resets them; ordering is the
recorded plan's, not the driver's.

### 2. THE RUNAHEAD RING (K > 1)

The 2a ring exists (output validity window, default K=3, LOUD on
read-past-window). 2b makes it a real runahead pipeline:

- **Bounded runahead with the loss-readback K-window.** `trainStep` returns a
  ring handle for `loss`; the driver does NOT await it every step. The CPU builds
  + submits up to K steps ahead; the (K+1)-th call BLOCKS on the oldest ring
  entry's fence (backpressure) before overwriting its slot. Awaiting `loss.item()`
  drains to that step (degenerates to serial — the logging cadence, FREE per
  sketch-2). K sized from G0(b): **K = 2 saturates the GPU-bound floor**; default
  K = 2–3, `opts.ringDepth` override.
- **Backpressure = the ring's fence gate**, not a new mechanism: reusing a ring
  slot's upload/output buffers requires the prior submit that read them to have
  fenced (the `bufferPool.canRecycle` / `sharedEncoderWriteSet` invariant already
  governs this). The ring simply refuses to overwrite slot i until step i fenced.
- **Interaction with step boundaries / implied-boundary machinery.** Each
  captured call is one implied step boundary (opt.step queues it, committed at the
  next backward — CLAUDE.md). Under runahead, K boundaries are in flight; the
  boundary sweep for step i must not demote storages step i+1..i+K still read.
  RULE (inherited, sharpened): the ring PINS each in-flight step's warm-slot +
  output buffers until its fence; the step-scoped sweep for step i runs only after
  step i fenced (never K-ahead). The implied-boundary commit stays per-call; the
  ring gates the SWEEP, not the commit.
- **Abort / interrupt mid-ring (user stops training).** On `trainStep.drain()`
  (or process teardown): fence all K in-flight steps, run their deferred sweeps in
  order, then release the ring. No partial-step state is possible because a
  captured step is atomic (the whole plan sequence submits or the fallback runs);
  the only in-flight thing is submitted GPU work + unread ring outputs. Draining =
  awaiting the last ring entry. A user `Ctrl-C` between steps loses at most the
  un-fenced tail (already-correct on GPU, just not read back).
- **Memory cost of the ring.** K steps of in-flight state = K × (batch uploads +
  ring outputs + the activations any un-fenced backward still needs). Activations
  dominate: with checkpointing, a step's peak activation set is bounded, but K
  un-fenced steps hold K× that peak transiently. Sized against G0: distil peak
  ~5 GB, so K=2 ≈ +5 GB transient worst-case; K must be budget-gated
  (`TORCHLETTE_POOL_BUDGET_MB`) and the design PREFERS K=2. This is the honest
  cost: runahead trades memory for the 30 % wall win, and on a memory-tight
  config (124M near the 32 GB V100 ceiling) K may be forced to 1 (= no runahead =
  no 2b win). **The win is real only where memory headroom for K≥2 exists** —
  state this in the perf claim, do not hide it.

### 3. GRADSCALER UNDER CAPTURE

- **where-select predication as data (the idiom, already shipped).** found-inf is
  a GPU `where`/`isfinite`, not a CPU branch (sketch-4). Params update via
  `where(finite, new, old)`; scale updates via `where(finite, scale*growth,
  scale*backoff)`. Nothing branches on a readback → capturable.
- **The found-inf readback and K.** The DANGER: if the driver reads found-inf
  every step to decide whether to step, that is a SECOND per-step readback
  dependency and it CAPS K = 1 (re-serializes, killing runahead). **Ruling: found-
  inf must NEVER be read in the hot loop.** The where-select makes the skip a DATA
  operation; the scale evolves as in-place state (census #5/#6). The only reason
  to read found-inf is diagnostics, and that rides the same deferred cadence as
  loss (ring output). So GradScaler does NOT cap K — PROVIDED the driver uses the
  data idiom, which the fullstack trajectory already does. The fullstack
  trajectory MUST be capturable WITH the scaler (gate ladder §8).
- **scale updates as volatile data.** The scale/inv_scale is a persistent tensor
  (read live by the replay) and/or a TAG_UNIFORM in the unscale config (census #5).
  It changes on the growth/backoff cadence — a per-step-varying scalar that MUST be
  a declared slot (the same coverage requirement the falsification found for
  adamStep). Its evolution is deterministic given found-inf (data), so no guard
  miss; but its bytes vary → it MUST be TAG_UNIFORM-covered or the tape is refused.

### 4. LR SCHEDULES + OPTIMIZER SCALARS UNDER CAPTURE

The frozen-scalar family's kill-zone. Composition, stated precisely:

- **Scheduled LR** flows as a graph scalar (SGD-alpha fix) → resolved via the
  scalar table OR baked into the adamStep config. Under capture it MUST be a
  DECLARED slot: either (a) a scalar-table slot (`DynamicSlotSource "scalar"`,
  the recorder's 4th source, re-written per step), or (b) part of the adamStep
  TAG_UNIFORM config (source "payload"), re-packed from the node payload each
  replay. Either way it rides DATA, never a frozen constant.
- **Adam bias-corrected step_size / step counter** (census #3/#4) — the
  falsification's load-bearing gap. Ruling: the optimizer island runs as a
  GENERATED plan whose per-param configs are TAG_UNIFORM ("payload" slots),
  `setAdamConfigUniforms` the single source (CLAUDE.md's "single source for the
  Adam mapping"), and the recorder observes EVERY one via `stRecordUniform`. The
  fix the falsification demands is: ensure the fused/foreach adam config reaches
  the recorder as TAG_UNIFORM in the recording steps (today it does not for the
  per-param fused path under the implied loop).
- **The structural kill of the family:** the recorder's guard-3 byte-diff
  REFUSES any per-step-varying byte not covered by a declared slot. So the design
  does not RELY on remembering to declare LR/step_size/scale — it CANNOT ship a
  tape that silently freezes one, because the recorder refuses it (and STRICT
  throws). The census (G0c) enumerates them; the guard enforces them; the
  falsification proved the guard fires. That triad is the structural defense — the
  7th frozen-scalar instance was caught by it at design time, not in production.

### 5. THE DECLARED-LIFETIME DIVIDEND (extension point — DO NOT BUILD)

> **Step-object update (task #98, 2026-07-15):** this dividend is now the
> DESTINATION of the step-object campaign (`docs/step-object-design.md`, ruling 2
> + phase 7), not a speculative maybe. It stays "DO NOT BUILD" here — the step
> object sequences it LAST, gated on its own re-open condition (captured path warm
> + observation-layer watcher cost measured). The design aims at it explicitly.

Inside a captured training region, the step's dataflow is known at capture time
— which buffers a plan writes, which are read cross-plan, which die at the
boundary. This is strictly MORE information than the observation layer
(`everSurvived`/`everReadback`/`everAliased`) reconstructs by watching executions
(stage-3 (A)/(B)). The extension point: **a captured boundary can DERIVE the
liveness the observation predicates currently OBSERVE.**

What becomes derivable-or-unnecessary on the CAPTURED training path:
- `everReadback` — a captured step's readbacks are the DECLARED ring outputs
  (loss, diagnostics). No observation needed: the tape enumerates them. The
  readback-pins-the-pair rule (stage-3 A) becomes "ring outputs are pinned by
  declaration."
- `everSurvived` — a captured step's cross-boundary survivors are exactly the
  registered persistent state (params, m/v, scale) + the K-ring outputs. Boundary-
  dead is the complement, KNOWN at capture time — no K-step last-reader stability
  wait (stage-3 B) needed to prove it.
- `everAliased` — the alias hole (stage-3 B) that the set-parity gate caught
  (a harvested view chaining to a released base) is, inside a capture, a
  STATICALLY VISIBLE aliasing relation in the recorded plan sequence — derivable,
  not observed-and-guarded.

**Capture-time functionalization of in-place-on-views (the aliasing-audit
direction), as a FUTURE increment — one honest page.** Today in-place ops on
views (`clip`'s in-place grad write, optimizer `copy_`) create the alias
relations the observation layer must chase. A captured region could run an
aliasing AUDIT at record time: walk the recorded plan sequence, identify every
in-place write and the view chains reading its base, and FUNCTIONALIZE the ones
that are safe to (rewrite in-place-on-view to a fresh buffer + a final copy,
where the liveness proves no later reader needs the pre-write value). This would
RETIRE, on the captured path: the `everAliased` chokepoint (no runtime alias
marking — the audit proved it), the last-reader K-stability machinery (boundary-
dead is declared), and the readback-observation seam (ring outputs declared).
What it would NOT retire: the LOWERED fallback path keeps all three (it has no
capture-time program to audit) — this is a captured-path DIVIDEND, not a global
deletion, exactly like stage-4 phase-4 dividends are earned locally. Honest cost:
the audit is a record-time pass (affordable — once per program, the step-tape's
whole budget thesis) but it is NEW mechanism; it earns its place only if the
observation layer's cost/complexity on the training path is measured to be worth
retiring. DECLINED as speculative for 2b proper; re-openable when a captured
training path is warm and the observation-layer cost is measured against it. Do
not build it in 2b.

### 6. GUARD / INVALIDATION

What invalidates a training capture (all LOUD, none silent — inherited):
- **Shape change** (batch size, seq len) → new bucket per the arg contract (the
  tensor-arg shape is in the appKey). One tape per shape bucket; the field's
  shape-stability convergence (sketch-3) bounds this.
- **Optimizer state replacement** (new optimizer instance, param added/removed,
  a param tensor swapped) → STRUCTURE change → the plan fingerprint changes → no
  matching skeleton → miss + re-record. Params UPDATING in place is NOT this
  (census #9).
- **Scaler dynamics are DATA, not structure** — a scale growth/backoff or a
  found-inf skip does NOT invalidate (it flows through declared slots / where-
  select). Only replacing the scaler object is structural.
- **Module structure edits** (a layer spliced, checkpoint toggled, autocast
  toggled) → structural miss + re-record (the Menagerie/mutation shape — an
  unserved-but-correct fallback citizen, sketch-7).
- **Plan validity / epoch / regime** — inherited guards 4/5 unchanged.

**TAPE_VERIFY shadow mode for training captures — how.** The 2a seam
(`stCaptureCompiledStep`) already cross-checks the skeleton it WOULD replay
against a freshly built template by canonical command-stream diff, every Nth
call. For training this works UNCHANGED in principle, with two training-specific
requirements the design must honor: (i) the verify step must run the NORMAL path
(build + compiled execute) so the in-place optimizer state advances EXACTLY ONCE
(the body runs once, for real — phase-1 §9.1) — a training verify must not
double-step the optimizer; the 2a "run normal instead of replay on the Nth call"
already gives this. (ii) the command-stream diff must canonicalize the per-step-
varying DECLARED slot bytes OUT of the comparison (batch data, step_size,
scale) — else every verify "diffs" on legitimately-varying data. The canonical
stream already excludes payload/uniform DATA (it compares STRUCTURE); confirm it
excludes the TAG_UNIFORM repacked bytes too, or the training verify is all false
positives. N=1 = full shadow (every step verified, no runahead) — the gate-ladder
mode before any default flip.

### 7. — see FALSIFICATION DUTY above (run, result reported).

### 8. GATE LADDER (for the eventual implementation)

Ordered; each solo; ALL pass before the flag ships even opt-in (phase-1 §3 shape):

- **G0 (done here):** the decomposition tables + runahead ceiling + census +
  falsification. Perf claims are anchored to G0(b)'s ~30 % distil runahead
  ceiling, NOT to a build-skip number.
- **G-cover — the falsification's fix:** the recorder forms an ELIGIBLE training
  tape (eligiblePairs > 0, refusals = 0) under the minimal implied-boundary loop
  with a varying batch, FUSED and FOREACH optimizer paths, WITH GradScaler, WITH
  an LR schedule. (This is the prerequisite the falsification proved is unmet
  today — it gates everything below.)
- **G-parity:** captured-vs-uncaptured fullstack trajectory parity ~1e-5 over 30
  steps, with FUSED and FOREACH optimizers, WITH GradScaler, WITH an LR schedule
  (`tools/parity-fullstack-tl.ts` grows a captured arm; the compiled-vs-lowered
  gate stays).
- **G-ring:** K=1 vs K=3 ring trajectory bit-identical (runahead must not perturb
  numerics — it only reorders CPU submission relative to the GPU fence).
- **G-scope:** scope-training under capture (the `api.scope`/`capture` composition,
  sketch-6) clean.
- **G-regression:** production regression (`tools/profile-training.ts` /
  DiLoCo 124M) baseline-exact loss + memory within the K-budget.
- **G-decode-untouched:** the decode stack (kv-differential, topk-equivalence,
  steering smoke, gen-tape) bit-identical with the training-capture code present
  — 2b must not regress 2a.
- **G-suite:** full `npm run test` green in BOTH flag states (`TORCHLETTE_STEP_TAPE`
  on/off); `test:gates` 4/4; STRICT_TAPE + STRICT_LIFETIME + STRICT_GPU clean over
  a captured training run.
- **G-perf:** ms/step before/after on the G0 configs (Node), guard-miss counters
  0 at steady state, K-sweep (K=1/2/3) showing the runahead curve flattening at
  the GPU floor; the G0(b) ceiling is the acceptance band. If the measured
  runahead win is < ~15 % wall on a memory-headroom config, the campaign pauses
  (the §4-style stop: the complexity must pay, and for training the pay is
  runahead, not build-skip).
- **G-soak:** `TAPE_VERIFY=16` over a long training run (LR schedule crossing a
  warmup→decay knee, a scaler backoff event, a checkpoint save) — zero verify
  diffs.

---

## Complexity budget (the weight-norm)

2b should add NO new env flags (rides `TORCHLETTE_STEP_TAPE` + its VERIFY/STRICT
modes, like 2a). The G-cover optimizer-config-coverage fix EXTENDS an existing
seam (`stRecordUniform` at the TAG_UNIFORM repack), it does not add mechanism.
The runahead ring EXTENDS the 2a ring (K as a runahead knob). §5's dividend is
where 2b eventually PAYS DOWN the weight-norm (retiring observation predicates on
the captured path) — named, not built. If G-cover cannot be closed without net-
new mechanism, that mechanism must name its deletions (the frozen-scalar caches
it subsumes) per house tradition.

---

## INC-1 LANDED (2026-07-08): G-cover — the recorder forms an eligible
## training tape; two coverage rules + one liveness seam; a pre-existing
## tape-starvation bug found and fixed en route

**Gate result (`tools/t-train-tape-probe.ts`, the falsification probe made a
permanent gate — minimal implied-boundary loop, varying batch, autocast +
checkpoint + GradScaler + clip + AdamW, distilgpt2@512, 18 steps):**
- FUSED adam (default): eligiblePairs=7, refusals=0, tapeCount=1 → PASS
- FOREACH adam (`TORCHLETTE_FUSED_ADAM=0`): eligiblePairs=4, refusals=0,
  tapeCount=1 → PASS (no extra work needed — the foreach program's per-step
  scalars already flow through covered channels; only the fused kernel's
  batch configs needed the new rule)

**The two coverage rules (both in `src/core/step-tape.ts` diffImages, both
verified-at-use so staleness degrades to a loud refusal, never a wrong replay):**
1. **Dead-payload/external** — a node whose result PRE-EXISTED this plan's
   execution (`hadResult` captured at stBeginPlan; the shared-node/external
   class: batch x/y appearing in backward/optimizer plans of the multi-plan
   checkpointed step) never has its payload read by this plan's replay —
   consumers resolve the producer plan's current result buffer per replay.
   Covered iff hadResult in BOTH compared steps.
2. **Batch-representative** — `adam-batch` (executor seam, `stDeclareBatchCover`)
   declares its member node positions + the representative whose TAG_UNIFORM
   repack carries the per-step config (bias-corrected step_size). Member payload
   variance is covered iff the representative is TAG_UNIFORM-covered in both
   steps AND the member's payloadHash EQUALS the representative's within each
   step — the agreement assert that makes a member with a DIVERGENT config
   (per-group hyperparams drifting from the representative) a LOUD refusal
   instead of a silent wrong-config replay. The declaration persists across
   template invalidation (build-from-IR rebuilds never re-run executor actions,
   so it cannot re-declare; measured: the optimizer template rebuilt ~step 11
   and every later pair refused until persistence).

**Pre-existing bug found and fixed (the capture-starvation class):** since the
build-from-IR default flip (c374d60a) activated observed-liveness, EVERY warm
capture/tape died after ~K_IDLE steady boundaries: tape replays bypass the
normal execution path, so the actively-replayed template accrued idleSteps and
was IDLE-RETIRED — destroying the compiled plan under the live skeleton
(`test/capture.spec.ts` warm-slot + closure-frozen tests were failing on main
under `TORCHLETTE_STEP_TAPE=1`; bisected to c374d60a). FIX: `noteTemplateReplayed`
(observed-liveness) resets ONLY the idle-retire clock on each tape replay —
deliberately NOT `noteTemplateExecuted`, so replays stay invisible to the
convergence/steady machinery (marking them executed lets convergence invalidate
and rebuild the plan mid-tape, forcing a spurious re-trace per convergence).

**Method note (honesty):** an intermediate attribution blamed the broader
noteTemplateExecuted routing for ~9k invalid-buffer submits in kv-differential;
that experiment was CONFOUNDED — the errors were `VK_ERROR_OUT_OF_DEVICE_MEMORY`
from foreign jobs on the shared V100 box, AND Dawn ignores CUDA_VISIBLE_DEVICES
(every "GPU-selected" run had used the same Vulkan default adapter; runs must
pin via `VULKAN_DEVICE_INDEX` + the `tools/vk-shim` LD_LIBRARY_PATH filter, the
launch-diloco mechanism). The narrow-reset choice stands on the convergence-
invisibility argument, not on that measurement.

**Ladder (all at final code, GPU-pinned):** probe PASS fused+foreach; suite
default 90+63 green (2 contention/network flakes green on rerun: relay,
pytorch-oracle); suite `TORCHLETTE_STEP_TAPE=1` 90+63 green (capture.spec 8/8 —
the two main-broken tests now pass); gates 4/4; kv-differential PASS clean
(0 OOM, 0 GPU errors, 4 arms bit-identical, taped 13/13 hits); fullstack
compiled-vs-lowered maxΔ 7.6e-6/30 steps; fullstack STRICT (LIFETIME+GPU) clean,
maxΔ 5.7e-6 vs compiled; production 124M regression PASS baseline-exact
(9.8089/5.9226/5.1516/4.6383 vs {9.81,5.92,5.15,4.64}), peak flat 2087.6 MB,
zero growth. Zero new env flags; no deletions (coverage extension of existing
seams: 2 recorder rules + 1 executor declaration + 1 liveness seam + 1 gate
probe).

---

## INC-2 PARTIAL (2026-07-08): #81 warning + gen-tape hardening LANDED;
## whole-step hit path STOPPED — §1's premise is FALSIFIED for today's
## optimizer/scaler (the S3.0 protocol: measured, stopped, re-approval)

**Landed:**
- **#81 ceremony-starvation diagnostic** at its canonical trigger
  (`stNoteBoundary`): an explicit `beginStep()`/`endStep()` comparator reset
  while the tape store is WARM warns once (loud, names the reason) and counts
  `ceremonyResetsWhileWarm` (on `stStats`). Test: `test/capture.spec.ts`
  "[#81] explicit ceremony against a WARM tape…" (flag-on, 9/9).
- **gen-tape-gate hardening** (the coordinator's addition): a third 64-token
  generation crossing ≫K_IDLE steady boundaries, asserting ≥85% hits + zero
  replay invalidations/missValidity. HONESTY: the differential (fix toggled
  off) shows decode still PASSES pre-fix — the decode template dodges the
  reaper by luck (live KV harvests rest the retire clock); the true
  regression cover for the reaper×replay class is capture.spec flag-on. The
  hardened gate makes the warm-across-the-window invariant EXECUTABLE for the
  decode shape so the luck eroding cannot ship silently. Measured post-fix:
  gen3 62/64 hits, invalidations=0.

**STOPPED — the §1 premise correction.** §1 asserts a training step "has NO
in-body JS side effects that the replay must reproduce: all state lives in
in-place tensor ops INSIDE the recorded plans." Verified against the code,
this is FALSE today; the body-never-runs hit would freeze (frozen-scalar
instance #8, structural, all confirmed by reading the step path):
1. **Adam `this.steps[i]`** advances in JS per step (`_advanceStep`);
   `stepSize = lr*sqrt(1−β2^t)/(1−β1^t)` and `lrTimesWd` are computed in JS
   and baked into the fused `adamStep` node PAYLOAD (adam.ts `_stepFused`) /
   the foreach-and-elementwise paths' graph scalars. The skeleton's
   TAG_UNIFORM repack reads the RETAINED node's payload — frozen at
   recording if the body never runs. The inc-1 batch-cover rule declares
   these as data ACROSS RECORDING STEPS (body ran both); it cannot conjure
   fresh values for a replay whose body never ran.
2. **GradScaler `_scale` is a JS NUMBER** (CPU mirror; `scale(loss)` bakes it
   as a graph scalar), `_growthTracker`/`_foundInfThisStep` are JS.
3. **`api.queueStepBoundary()`** fires inside `opt.step()` (mechanical — the
   hit path can queue it itself; listed for completeness).

**The prerequisite this names: the capturable-optimizer contract**
(sketch-3/#70, hyperparams-as-data) — before ANY body-never-runs training
hit: `t` on-device (persistent tensor advanced IN-PLAN), bias correction
derived in-graph/in-kernel from `t`, `lr` a persistent tensor written by the
scheduler at DRIVER level (`opt.lr =` setter → tensor write, outside the
body), scaler `scale`/`inv_scale` persistent tensors updated in-plan
(where-select already is). This is the SGD-alpha retirement template applied
to Adam+scaler — a self-contained sub-campaign with its own differential
gates (fused==foreach==elementwise trajectory; LR-schedule exactness), and it
is exactly §4's ruling made load-bearing. Proposed re-staging:
- **inc-2a (optimizer-scalars-as-data):** capturable Adam (t/lr/scale as
  data) behind `test/optim/fused-vs-elementwise.spec.ts` + fullstack
  LR-schedule differentials — no capture machinery involved; pays down
  frozen-scalar family permanently.
- **inc-2b (whole-step capture):** multi-plan skeletons (the 2a layer is
  single-plan by construction: `stCaptureCompiledStep` keeps ONE candidate),
  async captured bodies (`await loss.backward()`), output-node mapping
  (fn's returned tensor → plan k/pos), hit-path boundary queueing,
  micro/apply split, the chartered parity gates.

Landed-scope ladder (this commit): build green; capture.spec flag-on 9/9
(new #81 test); hardened gen-tape-gate PASS (62/64, 0 invalidations, 0 OOM,
GPU-pinned); differential-without-fix measured and recorded above.

---

## INC-2A DESIGN (approved scope: the capturable-optimizer contract) +
## the numerical falsification probe RESULT that shapes the kernel

**Probe run 2026-07-08 (falsification duty, before any code):** riskiest
inc-2a assumption = "in-kernel f32 derivation of the bias correction from an
on-device `t` matches the JS-double stepSize to gate tolerance." Emulated
WGSL f32 (Math.fround chains) vs f64 over t=1..200k, β=(0.9, 0.999):

| derivation | worst rel. err of 1−β₂ᵗ | worst rel. err of stepSize |
|---|---|---|
| naive `1 − pow(β,t)` | **1.95e-5 @ t=2** (cancellation) | **1.01e-5 @ t=2** |
| `−expm1(t·ln β)`, 5-term series for \|y\|<0.25 | 1.15e-6 @ t=240 | **6.0e-7 @ t=240** |

RULING FROM MEASUREMENT: the naive form sits AT the 1e-5 trajectory band at
early steps (would fail the canonical differential); the kernel/graph
derivation MUST be the expm1 form — `bc = −expm1(t·ln β)` with `ln β`
precomputed f64→f32 on CPU (static), a 5-term Horner series for |y|<0.25 and
`exp(y)−1` beyond. `t` is f32-exact to 2^24 steps. This applies identically
to WGSL (fused) and to graph ops (foreach/elementwise — needs `expm1` as an
op or the same series in ops; series-in-graph keeps all three paths on ONE
formula = the (d) no-per-path-special-case requirement).

**Implementation shape (the SGD-alpha retirement template, adam+scaler):**
- **State:** per-optimizer persistent tensors: `t` (f32 [1], advanced by ONE
  in-plan `add_(1)` per step — inside the optimizer plan so replays advance
  it), `lr` (f32 [1] per param-group, written by `opt.lr=`/scheduler setters
  at DRIVER level — sketch-3), scaler `scale`/`invScale` (f32 [1], updated
  in-plan via the existing where-select idiom; `invScale = 1/scale` as a
  graph op). Per-param `steps[i]` collapses to the shared `t` (the packed
  fused path ALREADY assumes batch-uniform steps; a param whose grad is
  absent skips its update but t advances — matches PyTorch capturable
  semantics; document the divergence from today's per-param counters).
- **Fused kernel:** bindings +1 storage buffer `hyper` = [t, lr, invScale(?)]
  (or read t/lr as two 1-elem buffers; ONE packed hyper buffer per optimizer
  keeps binding count at 9 ≤ 10). Kernel derives bc1/bc2 (expm1 form),
  stepSize, epsAdjusted, lrTimesWd — the uniform fields beta1/beta2/eps/wd/
  decoupled stay STATIC (no longer per-step-varying!). RETIREMENTS this
  buys, each named: AdamStepConfig.stepSize/lrTimesWd/invScale payload
  fields; setAdamConfigUniforms' per-step volatile fields; the TAG_UNIFORM
  volatile-repack for adam (the config becomes fully static → the whole
  adam volatileRepack closure + its recording plumbing); inc-1's
  batch-representative recorder rule for adam (members' payloads no longer
  vary → dead rule for this op — keep the mechanism only if another batched
  op needs it, else delete); the scalar-adapt churn for foreach's per-step
  coefficients.
- **Foreach/elementwise:** replace JS-computed bc1/bc2/stepSize scalars with
  graph ops reading `t`/`lr` tensors (same expm1-series formula). The paths
  converge on ONE derivation source (a shared helper building the bc
  subgraph).
- **GradScaler:** `_scale` becomes the persistent tensor (CPU number kept
  ONLY as a stats mirror updated at resolveDeferred readbacks — never an
  input to computation); `scale(loss)` = tensor mul; unscale's invScale =
  graph `reciprocal(scale)` feeding unscaleGrad as a TENSOR input (retires
  the unscaleGrad invScale config field + ITS volatile uniform).
- **#70 connection (scope-fenced):** this lands hyperparams-as-data for
  Adam+scaler only; the generalized registered-persistence contract stays
  #70's.

**Gates (chartered):** fused==foreach==elementwise trajectory differentials;
fullstack + LR schedule honored digit-for-digit vs sequential reference
(schedule VALUES exact; trajectories within the ~1e-5 band — the 6e-7
derivation noise is measured headroom); GradScaler trajectory parity;
t-train-tape-probe eligiblePairs>0/refusals=0 on both paths — EXPECT the
adamStep refusal class to disappear entirely (configs static, hyper flows
as buffer data), which lets the batch-cover rule retire; full ladder;
production regression baseline-exact (NOTE: baselines were re-derived once
before for a stepSize semantic fix (round-0 9.54→9.81); if per-param→shared-t
changes numerics for grad-gap edge cases, the regression must be re-baselined
EXPLICITLY, never waived silently).

## INC-2B DESIGN OPENER (chartered: short design + cheapest falsification)

**Surface:** (i) multi-plan skeletons — `stCaptureCompiledStep` keeps ONE
`candidate` (overwritten per compiled plan; code-confirmed), so a training
step's 4-6 plans need `candidates: Candidate[]` accumulated per step and
promotion matching the tape's ORDERED fp sequence; (ii) async captured
bodies — `CapturedFn.call` invokes `body()` synchronously (code-confirmed);
training bodies `await loss.backward()` → fn type + runIntercepted grow an
async variant; (iii) output-node mapping — fn's returned Tensor(s) must map
to (plan k, node pos) at promotion so a hit can harvest the loss from the
replayed plans; (iv) hit-path boundary queueing — `opt.step()` never runs on
a hit, so the hit calls `api.queueStepBoundary()` itself.

**Riskiest assumption:** "a training step's multi-plan sequence is STABLE
(same ordered fps every step) under the implied-boundary machinery" — if the
executor re-segments plans nondeterministically (merge decisions varying with
pool state), multi-plan skeletons can never be warm. **Cheapest probe —
ALREADY IN HAND (inc-1 gate data):** `t-train-tape-probe` stores tapes keyed
by `bucketKey = hash(structKey) + ordered plan fps`; 18 steps produced
`tapeCount=1` with 7 eligible pairs re-storing the SAME bucket (fused AND
foreach arms) — the ordered fp sequence is byte-stable across steady-state
steps. Assumption NOT falsified; inc-2b may proceed on it after inc-2a.

---

## INC-2B DESIGN (2026-07-09) — the four surfaces resolved; pre-approved
## conditions recorded; no new falsified premise, so implement.

**Doc-only first commit (this section).** The four opener surfaces are resolved
below against the code as it stands post-inc-2a (HEAD c6e59a13). No stop-rule
fired: the riskiest assumption (multi-plan fp-sequence stability) is evidenced
by inc-1 (tapeCount=1 / 7 eligible pairs, both arms); the one NEW fact surfaced
below (the step DELIMITER is GradScaler's internal `markStep`, not a pure
implied boundary) is a mechanism clarification, not a contract change — it makes
surface 4 EASIER, not harder. Implementation proceeds directly per charter.

### KEY MECHANISM FINDING (load-bearing for surfaces 1 & 4)

The inc-1 probe's "minimal implied-boundary loop" is delimited by
`GradScaler.resolveDeferred() → api.markStep()` at the TOP of each step
(`src/optim/grad-scaler.ts:134`), NOT by the pure implied boundary. Measured:
the probe runs with `boundaryResets = 0` (no `stNoteBoundary` fires) and
`stepsObserved = 10` — the 10 delimiters are 10 real frontend `markStep()`
calls from the scaler. Consequences:

- `stEndStep` + `stPromoteEligibleSkeleton` fire ONLY from the public
  `markStep()` (`torchlette.ts:2011/2019`). The pure implied-boundary commit
  (`_commitPendingStepBoundary`, used when NO scaler runs a markStep) calls
  `runtime.markStep()` + `stNoteBoundary("implied-boundary")` — it does NOT
  finalize a recorder step. **So the captured training regime IS the
  markStep-delimited regime** (the scaler provides it; the fullstack + all gates
  use GradScaler). A whole-step capture therefore inherits the EXACT delimiter
  the recorder already observes — no new boundary plumbing.
- **The whole captured step lives BETWEEN two of those `markStep()`s.** The
  recorded step = {forward plans, backward plans, optimizer plans}; the scaler's
  next-iteration `resolveDeferred()→markStep()` is the boundary that finalizes
  it and promotes the skeleton. The capture body owns forward+backward+opt; the
  driver still calls `scaler.resolveDeferred()` (its readback rides the loss
  cadence, §3). This is why surface 4's "hit must queueStepBoundary" is
  satisfied structurally: the scaler's markStep already IS the boundary, and on
  a hit the replayed optimizer plan's in-plan `queueStepBoundary` equivalent is
  moot (the markStep supersedes any queued boundary — `torchlette.ts:1958`).

### SURFACE 1 — MULTI-PLAN SKELETONS

Today `step-tape-replay.ts` keeps ONE module-level `candidate`, overwritten by
each `stCaptureCompiledStep` call (the LAST plan of a step wins). A training
step is N plans. Generalize:

- **`candidate: Candidate | null` → `candidates: Candidate[]`** accumulated per
  step (cleared at each `stEndStep`/promote boundary via the existing ctxAppKey
  reset). Each compiled plan under the active appKey pushes one entry in
  EXECUTION ORDER — the order `stCaptureCompiledStep` is called, which is the
  plan execution order.
- **Promotion matches the tape's ORDERED fp sequence.** `Skeleton` grows
  `plans: SkeletonPlan[]` (each = {planNodes, loweredPlan, bufferArena,
  canonical, uploads[], scalars[], lastNode}). `stPromoteEligibleSkeleton`
  iterates the tape's ORDERED plan fps and matches each to the captured
  candidate at the same ordinal. INVARIANT: `candidates.map(c=>c.fp)` must equal
  the tape's ordered plan fps (a strict, ordered, length-checked equality). Any
  mismatch → abort promotion (miss, re-record). NOTE the fp-source bug to fix:
  `lastEligible.fps` today is `[...templateIds]` (a Set — DEDUPED + UNORDERED);
  the ORDERED source is `rec.plans.map(pl=>pl.fp)`. Promotion must use the
  ordered list. Surface: `lastEligible` gains `orderedFps: number[]` (the
  bucketKey already derives from ordered fps, so this is a field exposure, not a
  new derivation). Deduped `fps`/`templateIds` stay for guard-4 invalidation.
- **Per-plan appKey/slot mapping.** ONE appKey per captured step (the arg
  boundary); each plan's upload/scalar slots resolve against THAT plan's
  `planNodes` (the tape slot ids already carry `w:<fpHex>:<pos>` / `sc:<fpHex>:
  <pos>:<ii>` — the fpHex disambiguates WHICH plan a slot belongs to; today's
  single-plan promotion filters `m[1] !== fpHex` and drops non-matching, which
  already IS per-plan routing — generalize the loop over all skeleton plans).
- **Replay invariants (ORDERING):** `stTryReplay` replays the skeleton plans in
  captured order (forward → backward → optimizer). Each plan's per-step node
  state (`results`/`_executed`/`_inputsRetained`) is reset before it runs. The
  cross-plan dataflow (backward reads forward's activation buffers; optimizer
  reads backward's grad buffers) is already correct by construction: these are
  external plan inputs resolved live per replay (the inc-1 dead-payload/external
  rule proved shared nodes resolve the producer plan's CURRENT result buffer per
  replay — the exact multi-plan cross-reference case). The batch x/y warm slots
  are written ONCE (surface 4) and read by whichever plan references them.
- **buffer/arena:** each plan keeps its own `bufferArena` (already per-template).
  No cross-plan arena sharing assumption is added.

### SURFACE 2 — ASYNC CAPTURED BODIES

`CapturedFn.call` is already `async` and `await`s `_captureReplay`. The BODY
(`this.fn(...)`) is invoked SYNCHRONOUSLY inside `runIntercepted`. A training
body does `await loss.backward()` → the fn is async. Contract:

- **`CapturedFn` gains an async body variant.** `runIntercepted` becomes
  `async` and `await`s `body()`. The upload interceptor + onWrap tracking are
  synchronous (fire during graph build); awaiting `backward()` inside is fine —
  it builds the backward graph lazily (does not fence) and the interceptor stays
  installed across the await (module-level `_captureInterceptor`, restored in
  `finally`). No short-circuit throw for training (see surface 4), so the async
  path is simpler than decode's: run the body to completion on a trace, replay
  without running it on a hit.
- **WHAT MAY BE AWAITED inside a captured body: engine ops only** (`backward()`,
  `loss.item()` if the driver chooses — though §3 says defer it, `scaler`
  ops that build graph). These are deterministic given the args + closure state.
- **WHAT MUST NOT: external I/O / non-deterministic awaits** (fs, network, a
  fresh random draw not seeded from a tensor arg, `Date.now`). These break
  run-exactly-once determinism: on a HIT the body does not run, so any external
  effect it would have produced is FROZEN (the jax.jit closure contract, already
  documented for 2a). **Enforcement (cheap, loud):** the arg-boundary contract
  already freezes closure values; we ADD no new runtime I/O trap (that would be
  net mechanism for a contract the closure-freeze already states). Instead the
  contract is DOCUMENTED on `capture()` (training variant) + BACKSTOPPED by
  TAPE_VERIFY (a body whose external effect changed the plan bytes diffs loudly).
  This matches 2a's stance exactly (closure freeze is tested, VERIFY is the
  paranoia backstop) — no new enforcement mechanism, per the complexity budget.

### SURFACE 3 — OUTPUT-NODE MAPPING

The body returns the loss Tensor (NOT awaited — readback stays outside per the
found-inf / K-window rulings). A hit must hand back a replay-harvested loss.

- **At promotion, map the returned Tensor → (plan k, node pos).** The trace runs
  the body for real and gets the real result Tensor(s). Its underlying lazy node
  is findable: `result._unwrap().lazyRef` → the producing node. Match that node
  identity against `candidates[k].planNodes` to find (k, pos). Store
  `outputRefs: Array<{planIndex: number; pos: number}>` on the Skeleton. This
  EXTENDS 2a's mechanism: decode's single-plan skeleton harvests `lastNode`
  (`planNodes[planNodes.length-1]`) — training generalizes "the last node" to
  "the declared output node(s), which may be in a non-final plan" (the loss is
  produced in the FORWARD plan, but backward+optimizer plans run AFTER it).
- **Harvest on a hit:** after replaying all plans in order, read
  `skeleton.plans[k].planNodes[pos].result` for each output ref, wrap via
  `createFromStorageHandle` (2a's exact path, `torchlette.ts:2057`). The loss
  buffer must SURVIVE the whole replay (backward/optimizer plans run after the
  forward plan that produced it) — it does: it's a persistent-ish forward output
  the backward reads (already alive through backward today; the ring PINS it for
  the K-window, §2 — that pinning is the same 2a ring-output mechanism).
- **Loss is NOT read on the hit path** (surface 3 ruling honored): the harvested
  Tensor is pushed to the ring and returned unread; the driver reads `.item()`
  on the logging cadence (deferred), exactly as 2a decode returns logits unread.

### SURFACE 4 — HIT-PATH SEMANTICS

- **NO short-circuit — upfront slot-check → full replay** (the doc's §1 ruling).
  Training has no short-circuit: the body is NOT run on a hit; all N plans are
  replayed (write warm slots → replay forward → backward → optimizer). Because
  the body-never-runs, the arg-boundary is ARG-ONLY-shaped for training: the
  batch (x, y) are the tensor ARGS (warm slots), so `known === 0`-style direct
  replay applies (no internal-upload short-circuit needed) — with the batch
  built by the driver as fresh pending `tensorFromArray` args (donated on hit).
  The tape's OTHER uploads (loss-seed / any internal fresh tensor) are covered
  by the inc-1 dead-payload/external + batch-representative rules; per-step
  scalars (t/lr/scale) are inc-2a's on-device DATA — nothing to re-dress.
- **UPFRONT slot-check (all warm slots verified before ANY replay — no mid-body
  throw):** `stTryReplay` already checks all guards (validity, regime, scalar
  coverage, upload shapes) BEFORE executing any plan. Multi-plan generalizes:
  verify EVERY plan's compiled-plan validity + EVERY plan's upload shapes up
  front, then replay all. A miss on any plan → whole-step miss → normal path
  (the whole body re-runs for real; the optimizer steps exactly once).
- **queueStepBoundary / implied boundary.** The recorded path's `opt.step()`
  queues an implied boundary that commits at the next `backward()`
  (`torchlette.ts:1460`). On a HIT `opt.step()` never runs. BUT the captured
  training regime is markStep-delimited (KEY FINDING): the scaler's NEXT-iter
  `resolveDeferred()→markStep()` is the boundary, and `markStep()` NULLS any
  pending queued boundary (`torchlette.ts:1958`) — so an un-queued boundary on a
  hit is harmless (the markStep is the real boundary either way). For ROBUSTNESS
  and to honor the doc's explicit ruling, the hit path calls
  `api.queueStepBoundary()` after a successful replay (idempotent — superseded by
  the scaler's markStep; matters only if a driver runs WITHOUT a scaler-markStep,
  where the queued boundary at the next backward is then the real commit). This
  is the "commit it identically to the recorded path" requirement, one line.
- **params/optimizer-state advance via the replayed in-place plans.** t.add_,
  copy_, m/v updates are ALL in-plan since inc-2a — the replay re-executes them,
  so params + Adam state + t + scale advance exactly as the recorded path. This
  is the §1 closure-state contract: state lives in in-place tensor ops inside
  the recorded plans (inc-2a made this TRUE — the premise inc-2 STOPPED on is now
  satisfied; §1's assertion holds post-inc-2a).
- **MICRO/APPLY split.** Implement the single-micro (accumSteps=1) path FULLY:
  ONE `capture()` wraps the whole step `(x,y) => { forward; backward; opt.step;
  opt.zeroGrad }` — this is the accumSteps=1 case where micro==apply (one body).
  Multi-micro (N>1: a separate `micro=(x,y)=>loss.backward()` capture reused N×
  + an `apply=()=>{opt.step;zeroGrad}` capture once) adds real structure (two
  captures, N replays of one, the `.grad` accumulation contract) and is a
  DOCUMENTED FOLLOW-ON unless it falls out for free. accumSteps=1 is the gate-3
  fullstack shape and the headline; ship it fully first.

### RUN-EXACTLY-ONCE WITNESS (the gate-2 requirement)

A body-side counter incremented inside the captured fn: on TRACE it advances
(body runs); on HIT it does NOT (body never runs). The probe asserts the counter
stops advancing once hits begin AND the trajectory still advances (params update
via replay). This is both the "body demonstrably not re-run" evidence and the
run-exactly-once witness the charter names.

### WHAT INCREMENT 3 (the ring) NEEDS THAT NOW EXISTS

After inc-2b: the whole step is ONE captured call returning a ring-handled loss
(unread). The 2a ring (`pushRing`, K default 3) already stages the output. Inc-3
turns K into a RUNAHEAD depth (G0b: K=2 saturates the GPU floor) by DEFERRING
the scaler's `resolveDeferred` readback + the loss `.item()` off the per-step
path, letting CPU build step N+1 while GPU drains step N. Inc-2b LEAVES the ring
at output-validity semantics (K=3, per-step readback still happens via the
driver) — it does NOT claim the runahead win (G0-honest: this increment is the
~3-4% build-skip, the ring is where the 30% lives). What inc-3 inherits from
inc-2b: multi-plan skeletons, async bodies, output-node mapping, hit-path
boundary — the entire declared-step surface the ring pipelines.


---

## INC-2B LANDED (2026-07-09): whole training step captured as ONE call

**What shipped.** `api.capture(fn, { training: true })` wraps a whole training
step (forward + backward + optimizer). On the 2nd+ eligible call the body does
NOT run — the recorded MULTI-plan sequence (forward/backward/optimizer plans)
replays, advancing params/Adam-state/`t`/scale via their in-place ops (inc-2a
made this the whole state-transition, so §1's "state lives in in-plan ops"
premise — falsified for inc-2's optimizer — now HOLDS).

**The four surfaces, as built:**
1. **Multi-plan skeletons** — `step-tape-replay.ts`'s single `candidate` →
   ordered `candidates[]`; `Skeleton.plans: SkeletonPlan[]`; promotion matches
   the recorder's ORDERED plan-fp sequence (`lastEligible.orderedFps`; the
   Set-based `fps` was deduped/unordered — kept for guard-4). Per-plan
   upload/scalar slots route by the slot-id fpHex.
2. **Async bodies** — `runIntercepted` awaits `body()`; `capture()`/`CapturedFn`
   accept `Promise<Tensor|Tensor[]>`. External I/O in a body is FROZEN by the
   closure contract (documented, VERIFY-backstopped — no new runtime trap, per
   budget).
3. **Output-node mapping** — the traced body's returned loss node is recorded
   (`_declareCaptureOutputs`) and matched by identity to a (planIndex,pos) at
   promotion; the replay harvests it. `ReplayResult` now returns `outputs[]`.
   NOTE the driver must return a loss handle that SURVIVES backward (backward's
   graph teardown disposes the raw loss) — a `noGrad(mul(loss,1))` forward-plan
   copy is the idiom; its buffer survives the backward/optimizer replay.
4. **Hit-path** — training capture is NEVER short-circuited (body runs once on a
   miss, never on a hit). Upfront all-plan validity+shape check (no mid-replay
   throw). Batch x/y are tensor ARGS re-dressed by ARG ORDER (they share a
   shape — shape-keying is ambiguous); i32 token args are MATERIALIZED from
   their fresh payload before replay (they are not TAG_WRITE-covered like f32
   decode uploads — `executeOpSync`+`assignNodeResult`); constant internal
   uploads (`zeros`/`full` from zeroGrad) keep their recorded payload. The hit
   queues the implied step boundary. accumSteps=1 (micro==apply, one body) is
   FULLY implemented; multi-micro is a documented follow-on.

**TWO cross-plan lifetime issues found + fixed (both are §5's declared-lifetime
dividend, applied narrowly to the captured path):**
- **Cross-plan materialized refs** — a later plan's input is a MATERIALIZED ref
  to a storage an earlier plan produced (the optimizer plan reading a grad the
  backward plan wrote, forced/materialized between plans). The ref freezes the
  RECORDING buffer (step-scoped, destroyed at markStep). Promotion snapshots
  each plan's produced storage ids WHILE ALIVE (`Candidate.resultIds` — results
  are swept before promotion) and builds `crossPlanLinks`; replay re-points each
  ref to the producer's FRESH result after the producer plan runs.
- **Lowered cross-plan producers + the stage-3 B release** — some grads are
  produced by LOWERED backward segments (not compiled candidates), and the
  stage-3 B clear-at-release marks a last-reader plan's external-input result
  `releasedOverlay`. Both make a captured replay read a DECLARED-live buffer
  that the observation layer thinks is destroyed/released. During a MULTI-plan
  tape replay (`setStepTapeReplayActive`): the stage-3 B clear-at-release is
  INERT, and the op-dispatch `releasedOverlay`/`isDestroyed` STRICT guards are
  suppressed — the tape DECLARES the whole step's dataflow, so the per-handle
  observation verdicts do not apply to cross-plan reads. **Correctness proven by
  the noise floor:** captured-vs-uncaptured trajectory Δ (6.9e-4 @24, 1.55e-3
  @40 steps) is BELOW the control-vs-control cross-run fp-noise floor (1.3e-3
  @24, 2.09e-3 @40) — the buffers ARE correctly re-bound by the planner; the
  destroyed-handle throw was a false positive. This is exactly §5: inside a
  declared boundary, observation-layer predicates are superseded.

**Gate ladder (device-2, vk-shim, verbatim in the session report):** build;
inc-2b probe PASS (promotions=1, hits=19/24, bodyRuns=5 — body FROZEN on hits =
run-exactly-once witness; trajectory within noise floor); capture.spec flag-on
10/10 (+the 2b training test; decode 2a intact); t-train-tape-probe fused 7/0 +
foreach 4/0; gen-tape 0-diverged/3101 cmds (+hardened gen-tape-gate gen3 62/64,
invalidations=0); kv-differential PASS (taped 13/13, all arms identical);
compiled-vs-lowered 4.77e-6/30; test:gates 4/4; STRICT_LIFETIME+STRICT_GPU
captured fullstack exit-0 zero throws; full suite BOTH flag states (flag-off:
cpu 1172/1-relay-flake→4/4 isolated + webgpu 897/0; flag-on STEP_TAPE=1: cpu
1173/0 + webgpu 906/0); 124M production regression PASS baseline-exact
(9.8089/5.9224/5.1528/4.6392 vs {9.81,5.92,5.15,4.64}, peak flat 2087.6 MB,
zero growth).

### INC-2B SCHEDULE ARM — RESOLVED 2026-07-09 (merge 64b84efc). LR schedules
### flow through replay hits exactly; FIVE root causes found and fixed; one
### PRE-EXISTING plain-path bug exposed, minimally reproduced, quarantined

**Final state:** CosineAnnealingLR stepped at driver level flows through
hits. Battery (V100): fused 3× {6.7e-4, 1.3e-3, 1.0e-3}, foreach 3× {7.2e-4,
5.9e-4, 5.0e-4}, STRICT 6.4e-4/0 throws — all at the cross-run fp noise
floor. In-suite gate: `test/capture.spec.ts` "[2b-sched]". Suites green both
flag states (cpu 1173 + webgpu 907/897), gates 4/4, stream-generate
0-diverged/3101, 124M baseline-exact 4.6403/2087.6MB flat, compiled-vs-
lowered 6.7e-6/30.

**The five causes (each measured; details in commit 64b84efc):**
1. setLR `full(lr)` template-thrash (no tape CAN form; plain loops OOM 34GB)
   → deliver via `tensorFromArray` (payload-exempt).
2. Frozen materialized refs to replace-and-hold scalars (0.48 nats) →
   owner-backed SCALAR-ONLY rebind (owners snapshotted at candidate capture;
   larger refs NEVER rebound — the isDestroyed-shotgun variant sporadically
   aliased planner temps: 6e-3..1.06 nats, ~1-in-3).
3. In-plan lr-write ghost chains replay the recorded upload payload →
   `scalarDresses`: per-replay re-dress of the recorded write node (owner
   resolved at PROMOTION — at capture the wrapper hasn't materialized into
   the chain output yet).
4. Interval-END ghosts are consumed one step stale (~0.03) → pre-replay
   `queue.writeBuffer` of the current value into the recorded binding.
   NEVER force the pending chain mid-attempt (12.7-nat corruption).
5. Mixed markStep regimes staled chain-derived values (sporadic 3e-3..0.07)
   → `src/core/scalar-slots.ts`: authoritative host-value registry — setLR
   NOTES the value, the replay dresses from it. Single source at the seam.

**PRE-EXISTING plain-path bug exposed (own follow-on campaign):** a plain
flag-OFF loop with a per-step scheduler grows GPU memory toward the ceiling;
a second in-process run sporadically (~1-in-3) lands a BIMODAL wrong
trajectory (finals 10.581 vs 11.291/10.630 — same wrong values each time = a
deterministic wrong branch under memory pressure). Repro:
`tools/t-sched-plain-repro.ts`; reproduces on main + the delivery fix alone
(masked on unpatched main only because the thrash OOMs first). The capture
probe now runs its arms in CHILD processes — captured-arm finals were
identical 16/16 across runs; only the plain arm is fragile.

Known tool rot (pre-existing, unrelated): standalone
`tools/t-stream-determinism.ts` fails "STREAM COUNT differs: 3 vs 2" on main
in BOTH flag states; the load-bearing determinism gate is in-suite
(test:gates) and green.

### (superseded 2026-07-09) original stop-report for this arm

**Behavior at HEAD (safe):** a training capture under an LR schedule never
forms a tape — `setLR`'s `full([1], lr)` fillValue hashes into the template
fingerprint (the deliberate latent-frozen-scalar defense), so every step is a
structural miss (measured: structureMisses=25/30, hits=0). The capture falls
back to the correct slow path every call and the PAYLOAD-THRASH warning names
the fix. CORRECT, UNSERVED — the charter's fallback-citizen class. The charter
note "inc-1's rules should already handle it" is FALSIFIED: the lr write is
not covered, it is structurally refused.

**The three causes (each measured, WIP branch has the mechanisms):**
1. **Template thrash** — `setLR` must deliver lr as `tensorFromArray` values
   (PAYLOAD_HASH_EXEMPT, re-executed per replay), not `full(fillValue)`. With
   this alone the tape FORMS and HITS — with a FROZEN lr (0.48 nats
   divergence over 30 steps, captured descending FASTER = stale-higher lr,
   the directional-bias signature). DO NOT land this fix alone.
2. **Recorded lr-write ghost** — the driver-level lr write (forced at the
   step-opening markStep) is recorded INSIDE the step interval (at its END —
   it is the PREVIOUS iteration's write). A replay must not re-execute the
   recorded ghost. WIP: SkeletonPlan.preBody + stMarkBodyBegin (measured: the
   ghost lands at interval END, post-optimizer-read, so it was NOT the live
   divergence — mechanism kept for the class).
3. **Frozen materialized refs to replace-and-hold scalars** — `copy_` =
   stridedScatterCopy returns a NEW buffer; the lr tensor's storage identity
   wanders every step while the skeleton's frozen adamStep input ref reads
   the recording-era buffer forever. ISOLATION: constant-lr (identical writes/
   structure, value never changes) = 7.9e-4 = the cross-run noise floor;
   varying lr = 0.48. WIP: owner-backed ref REBIND (storageTracker.ownerOf,
   owner snapshot at candidate-capture, re-resolve to the owner's current
   storage per replay). Measured ladder: rebind-nothing 0.48; rebind-all-
   changed 0.010 (params rebinding perturbs the planner-consistent chain);
   scalars+destroyed-only 0.010–0.024 with the lr ref no longer re-identified
   after the first replay (UNRESOLVED — the stop rule). t/m/v/params are NOT
   affected: they advance via planner-fixed in-plan buffers (proven by the
   no-scheduler 40-step noise-floor run).

**Next session:** find why the lr ref stops rebinding after the first replay
(owner lazyRef state on hit iterations), or — likely cleaner — make the
optimizer-scalar write BUFFER-STABLE by construction (a declared scalar-slot
write into a fixed buffer, the TAG_WRITE idiom, instead of a replace-and-hold
copy_ chain): then no rebind machinery is needed at all and the schedule rides
the same stable-buffer channel as the batch. Charter both against the
LR-schedule-exactness differentials (the off-by-one submission-order hazard of
a queue.writeBuffer-based write is real — the write must stay ORDERED after
the prior step's optimizer reads).

**G0-HONEST PERF (do not oversell — the ring is inc-3):** this increment is the
~3-4% build-skip (G0a). The captured hit skips graph build / plan-collect /
fingerprint / CSE for all 4 plans but still runs every plan's compiled replay
and every per-step markStep fence; runahead (the 30% G0b win) is inc-3, which
now HAS the whole-step declared surface it pipelines.

**What inc-3 (the ring) needs that now exists:** the whole step is ONE captured
call returning a ring-handled loss (unread on the hit path); multi-plan
skeletons, async bodies, output-node harvest, and the declared cross-plan
lifetime are all in place. Inc-3 turns K into a runahead depth by deferring the
scaler `resolveDeferred` readback + loss `.item()` off the per-step path.

**SLOC:** +~360 src (multi-plan replay + cross-plan links + capture training
path + the §5 replay-active suppression). Zero new env flags. No deletions
(coverage/mechanism extension of the 2a capture + step-tape-replay seams).

---

## INC-3 DESIGN (2026-07-09) — THE RUNAHEAD RING (the campaign's ~30% win)

**Charter:** G0(b) proved GPU/step (~165ms distil) ≫ CPU-overlappable (~71ms),
so K=2 saturates: deferring the per-step loss readback + the markStep fence lets
CPU build+submit step N+1 while GPU drains step N. `ringDepth` is a capture
OPTION (default 2 for training). NO pressure-reactive automation; backpressure =
a fence gate; drain-on-abort; found-inf never rides the readback path. K×
activation memory is the honest documented cost. This section resolves (a)-(f)
per charter, THEN implements — no external review stop unless a premise falsifies.

### (a) WHAT IS DEFERRED vs WHAT IS NOT

**The two per-step serialization points are the DRIVER's, not the body's.** The
captured training body (`await loss.backward(); opt.step()`) builds LAZILY and
does not fence (confirmed: `stTryReplay` submits the plan sequence on the shared
encoder and harvests output handles — NO fence inside; `executeLoweredPlan`
encodes, it does not await). The fences are BOTH driver-level:
1. `loss.item()` / `api.cpu(loss)` → `runtime.force(loss)` → GPU fence + mapAsync
   readback (`engine.ts:1570`).
2. `scaler.resolveDeferred()` → `api.markStep()` at the TOP of each step, and the
   trailing `await api.markStep()` → `awaitDeferredFence()` (`torchlette.ts:1998`).

**DEFERRED (the ring's job):** the loss readback await and the markStep-fence
await are moved OFF the per-step critical path — held in the ring, resolved on the
logging cadence (loss) or when backpressure demands (fence). **NOT DEFERRED:**
(i) submits stay per-step — each captured call still submits its whole plan
sequence immediately (runahead overlaps CPU build of step N+1 with GPU drain of
step N; it does not batch submits); (ii) the in-graph GradScaler where-select
predication stays per-step-exact (found-inf is DATA, never a readback in the hot
loop — §3); (iii) the boundary COMMIT (`queueStepBoundary`) stays per-call — the
ring gates only the SWEEP (§2's inherited rule).

**Why the ring lives at the CAPTURE layer, not the driver.** The four historical
loss-overlap failures (CLAUDE.md "Moving loss.item() after backward") all failed
because they raced the LIVE autograd graph's loss buffer against backward's reuse.
The ring reads back a HARVESTED ring output — a materialized forward-plan copy
(`noGrad(mul(loss,1))`, inc-2b surface 3) whose buffer the tape's declared
lifetime PINS for the K-window — not a live graph tensor. The tape replay owns the
buffers; the readback targets the harvested handle. §(f) maps each failure.

### (b) RING STRUCTURE

A K-deep queue of `RingEntry { step, outputs: Tensor[], settle: () => Promise<void> }`.
`outputs` = the harvested loss handle(s) (unread). `settle` = the deferred
step-boundary fence+sweep for THIS step (the `markStep` the driver would have
awaited). On `pushRing`:
- Assign `step = stepCounter++`.
- **Backpressure (the fence gate):** while `ring.length >= K`, `await` the
  OLDEST entry's `settle()` (fences step `ring[0].step`, runs its sweep), then
  expire+shift it. This is the (K+1)-th-call BLOCK of §2 — reusing a ring slot's
  buffers requires the prior submit that read them to have fenced, exactly the
  `bufferPool.canRecycle`/`sharedEncoderWriteSet` invariant the sweep already
  honors. NO new mechanism: `settle` IS `api.markStep()` (or the implied-boundary
  commit), deferred.
- Push the new entry; return the (still-unread) loss handle.

The output-validity WINDOW (2a's LOUD read-past-K error) is SUBSUMED by
backpressure: an entry is only expired AFTER its `settle` fenced, so a handle read
within K calls is always still-materialized. The `ringDepth` default flips from
2a's 3 (output validity) to **2 for training** (runahead saturation, G0b) — a
capture OPTION, per charter.

### (c) FOUND-INF / SCALE-GROWTH UNDER LAG — the correctness contract

**CONTRACT: the on-device where-select keeps TRAJECTORIES BIT-EXACT regardless of
K; only the CPU scale MIRROR / growth bookkeeping lags ≤ K steps.** Proof:
- The parameter/scale UPDATE is a graph op: `where(finite, new, old)` for params,
  `where(finite, scale*backoff, scale*growth)` for scale (§3, already shipped).
  These execute IN THE REPLAYED PLANS — their inputs are on-device tensors
  (grads, the persistent scale tensor). K reorders WHEN the driver awaits the
  fence; it does NOT change what the GPU computes or the submit ORDER. So the GPU
  trajectory is byte-identical for any K. **This is why K is a pure knob** — the
  gate-2 bit-parity requirement.
- The CPU `_scale` NUMBER and `_growthTracker` are updated in
  `scaler.resolveDeferred()` from the found-inf READBACK. Under runahead that
  readback is deferred with the fence, so the CPU mirror lags ≤ K steps behind
  the on-device scale tensor. **This lag does NOT perturb the trajectory** because
  computation reads the on-device scale tensor (inc-2a: `scale`/`invScale`
  persistent tensors), NEVER the CPU `_scale` — the CPU number is a stats mirror
  only (inc-2a: "CPU number kept ONLY as a stats mirror").
- **found-inf must NEVER be read in the hot loop** (charter, §3): reading it to
  decide whether to step would be a SECOND per-step readback → caps K=1. It is
  DATA (the where-select). Diagnostics ride the loss cadence.

**The injected-inf gate (charter gate 3) proves the lag bound, not its absence:**
inject inf grads at step N; K=1 and K=2 trajectories must be IDENTICAL (the
in-graph skip is exact for any K); the CPU scale-adjustment may LAND ≤ K steps
later under K=2, asserted as a bounded lag, not zero.

### (d) DRAIN

`trainStep.drain()` (new method on the CapturedFn callable) + implicit drain on
abort: fence all in-flight entries IN ORDER (await each `settle()` oldest-first),
resolve all pending readbacks (any held loss handles become readable — already
materialized), then clear the ring. A captured step is ATOMIC (the whole plan
sequence submits, or the fallback runs) so no partial-step state exists; the only
in-flight thing is submitted GPU work + unread ring outputs. Draining = awaiting
the last entry's settle. No hangs (every settle is an `await markStep`), no
`[lifetime]` warns (the sweep runs in order, past each buffer's last GPU use).

### (e) MEMORY

K un-fenced steps hold K× {batch uploads + ring outputs + the activations any
un-fenced backward still needs}. With checkpointing a step's peak activation set
is bounded; K=2 ≈ +1 step's transient peak worst-case. Measured in gate 6
(peak/cur/phys at K=1 vs K=2). The design PREFERS K=2; on a memory-tight config
(124M near the 32GB V100 ceiling) K may be forced to 1 (= no runahead = no inc-3
win). The win is real ONLY where headroom for K≥2 exists — stated, not hidden.

### (f) THE FOUR HISTORICAL FAILURES — why the CAPTURE-layer ring avoids each

CLAUDE.md "Moving loss.item() after backward" (4 approaches, all failed):
1. **retainGrad doesn't prevent cleanupAutogradGraph disposal** — N/A: the ring
   output is `noGrad(mul(loss,1))`, NOT an autograd node; cleanupAutogradGraph
   never touches it. The tape's declared lifetime pins its buffer for K.
2. **preserved-set aliasing (shared-encoder read-write conflict)** — N/A: the
   harvested handle is a plan OUTPUT the memory planner already assigned a
   distinct buffer; the ring does not add it to any `preserved` set nor alias it
   into the live graph.
3. **concurrent item() + backward → "Engine is busy" (exec lock)** — N/A: the
   ring NEVER runs item() concurrently with a body. Runahead overlaps CPU BUILD
   of step N+1 with GPU DRAIN of step N; the readback of step N's loss happens
   later, single-threaded through the exec lock, when the driver chooses.
4. **force()-then-race (backward reused the loss buffer before mapAsync read)** —
   N/A: the harvested handle's buffer is NOT reused by any later step until its
   `settle` fenced (backpressure). The readback reads a buffer whose last writer
   (the forward plan of step N) is behind a fence the ring guarantees precedes any
   reuse. This is the "dedicated readback staging" property earned structurally:
   the tape OWNS the buffer's lifetime, so no later step overwrites it in-window.

### IMPLEMENTATION SHAPE

- `capture.ts` `pushRing` grows the backpressure fence-gate + `settle` capture;
  `RingEntry` gains `settle`. The CapturedFn callable gains `drain()`.
- The DRIVER stops awaiting `item()`/`markStep()` per step: it awaits the ring
  handle on the logging cadence and calls `drain()` at the end. `settle` is
  supplied by the capture via `_deferBoundaryCommit()` (training). (NOTE: the
  initial sketch here was `() => api.markStep()`; implementation revised it to a
  SPLIT gen-scoped boundary — recorder-finalize + snapshot synchronous, fence +
  sweep deferred — because a bare deferred markStep both desyncs the recorder
  context and violates quiesce-before-destroy. See the INC-3 PARTIAL LANDED
  section for the four root-caused layers.)
- The measurement probe (`t-train-capture-probe`, K arm) drives the ring in
  runahead mode and reports wall/step for K=1 (serial reference) vs K=2 vs
  uncaptured.

No new env flags (ringDepth is the existing capture option; default 2 for
training). No deletions (the ring EXTENDS 2a's `pushRing`).

---

## INC-3 PARTIAL LANDED (2026-07-09): the ring MECHANISM is bit-exact at K=1;
## K≥2 runahead blocked on a POOL-EXCLUDED readback staging buffer (cross-
## surface). Four correctness layers root-caused in order; only the last remains.

**What shipped (all in `src/frontend/capture.ts` + `torchlette.ts` boundary
seam — my owned surface):**
- **The ring structure** — `RingEntry.settle`, `backpressure()` (fence gate at
  the TOP of the (K+1)-th call), `drain()` (oldest-first, idempotent),
  `runahead` opt-in capture option (training-only), `ringDepth` default 2 for
  training. The callable gains `drain()`.
- **`_deferBoundaryCommit()`** (torchlette.ts) — the runahead step boundary,
  SPLIT into a synchronous half + a deferred `settle`:
  - **SYNC now (this step's context):** `endStep` + `forceAllPending` +
    `snapshotForStep(gen)` + `beginStep` + RECORDER FINALIZE (`stEndStep` +
    `stPromoteEligibleSkeleton`) + ISSUE fence.
  - **DEFERRED settle (run K steps later by backpressure/drain):** await THIS
    step's fence, THEN the gen-scoped sweep (`releaseStepTemps(gen)` +
    `destroyUnreachable`) + `observeStepBoundary`.
- `_commitStepBoundaryGen` extracted from `_commitPendingStepBoundary` (the
  implied-boundary path is unchanged — regression: implied-step-boundary 6/6).

**THE FOUR CORRECTNESS LAYERS, root-caused in order (each was a measured
falsification, each fixed except the last):**
1. **Sweep-vs-submit race** — backpressure originally fenced in `pushRing`
   AFTER the step's work was submitted, so the oldest boundary's sweep raced the
   new step's in-flight buffers ("used in submit while destroyed"). FIX: fence at
   the TOP of the call (mirrors the driver's `resolveDeferred()→markStep()`),
   before this step submits.
2. **Recorder finalize CANNOT be deferred** — a deferred `stEndStep` ran under
   the NEXT call's tape context (`_setCaptureTapeContext` already overwrote it),
   desyncing the consecutive-step comparator → tapeCount formed but hits=0. FIX:
   the recorder finalize is SYNCHRONOUS in `_deferBoundaryCommit`, under this
   step's context. (After this, hits=14, body frozen, trajectory tracked serial.)
3. **Quiesce-before-destroy** — the gen-scoped SWEEP (`destroyUnreachable` /
   `releaseStepTemps`) ran BEFORE the deferred fence-await, so a queued destroy
   fired while the step's submit was un-fenced. FIX: the sweep rides the DEFERRED
   settle, AFTER `awaitDeferredFence`. Gen-scoping makes the deferred sweep
   pin-safe (only ≤ gen tensors reclaimed; later in-flight steps stamped > gen
   are untouched). **After 1+2+3: K=1 is BIT-IDENTICAL to the serial 2b path,
   zero GPU errors, body frozen, STRICT_LIFETIME+STRICT_GPU zero throws.**
4. **[REMAINING] Deferred-readback buffer lifetime (K≥2)** — the harvested loss
   is a memory-planner-assigned OUTPUT SLOT the next step's replay REBINDS.
   `persist()` protects the wrapper from step-scoped release but NOT from the
   planner rebinding the physical buffer. So a K-behind readback at K≥2 reads a
   rebound buffer (0.00000 / shifted values). **This is exactly CLAUDE.md's
   "dedicated readback staging buffer excluded from the pool" — the requirement
   the four historical loss-overlap attempts all lacked.** It is a BUFFER-POOL /
   PLANNER surface change (outside capture.ts + the ring): ring outputs must be
   copied into a staging buffer the planner never reuses (cf. the existing
   `startScalarReadback` primitive, engine.ts). Until then K is clamped to the
   proven-safe K=1 in the shipped tests.

**IMPORTANT — K=1 is CORRECTNESS-complete but NOT the runahead WIN.** At K=1 the
ring holds one entry; backpressure at call N+1 awaits step N's fence BEFORE step
N+1's body — i.e. still serial (no CPU/GPU overlap). The G0(b) win needs K≥2
(one step in flight), which layer #4 blocks. So inc-3 has landed the ring
MECHANISM proven bit-exact and the boundary split that makes runahead possible;
the ~30% wall win itself awaits the pool-excluded readback staging (layer #4) +
per-step fence-slot isolation (`issueDeferredFence` overwrites a single
`pendingFencePromise`; K≥2 needs per-settle fence promises — also buffer-pool
surface).

**Gate ladder (device 2, vk-shim, VULKAN_DEVICE_INDEX=2, verbatim):**
1. build: PASS.
2. K=1 == serial BIT-PARITY (`tools/t-ring-probe.ts`): serial-vs-ringNow
   (immediate-read) maxΔ=0.00e+0; serial-vs-ringK1 (1-behind) maxΔ=0.00e+0;
   hits=10, bodyRuns=10 (body frozen); ZERO "used in submit while destroyed".
   K1 mechanism PASS. (K2 arm: WIP — layer #4.)
3. Injected-inf gate: NOT YET RUN (needs GradScaler arm; the mechanism supports
   it since scale is on-device DATA — deferred to the K≥2 unblock).
4. Drain-on-abort (`capture.spec.ts` "[inc-3] drain mid-ring"): PASS — interrupt
   after 5 un-read steps, drain fences in order, idempotent, last handle
   readable, no hang, no throws.
5. THE MEASUREMENT: NOT MEANINGFUL at K=1 (no overlap). Deferred to K≥2 unblock.
6. Memory K=1 vs K=2: deferred to K≥2.
7. capture.spec extended: 13/13 green (11 existing + 2 new inc-3 ring tests),
   both flag states.
8. t-train-capture-probe (serial 2b): PASS (hits=14, refusals=0,
   bodyFrozen=true, maxLossDelta=2.5e-5); t-train-tape-probe: eligiblePairs=7,
   refusals=0, tapeCount=1. No inc-2b regression.
9. Full suite: capture.spec both flag states green; implied-step-boundary 6/6;
   test:gates 4/4.
10. 124M regression: NOT RE-RUN (uncaptured path untouched — the boundary refactor
    preserves `_commitPendingStepBoundary` exactly; implied-step-boundary green is
    the proxy). Recommend a run before any default flip.
11. Decode stack: capture.spec decode (2a) tests intact (part of the 13/13).
12. STRICT_LIFETIME+STRICT_GPU on the K=1 captured ring (`capture.spec.ts`
    "[inc-3]" under both STRICT flags): 2/2 PASS, ZERO throws.

**WHAT REMAINS for #80 (the runahead WIN):**
- **Layer #4 (the blocker):** a pool/planner-EXCLUDED readback staging buffer for
  ring outputs — copy the harvested loss into a buffer the planner never rebinds
  (extend `startScalarReadback` / a dedicated staging ring). Then K≥2 outputs
  survive the K-window readback. BUFFER-POOL surface.
- **Per-settle fence isolation:** `issueDeferredFence` writes one shared
  `pendingFencePromise`; K≥2 needs each settle to await ITS OWN fence promise
  (capture it at issue time). BUFFER-POOL surface.
- Once #4 + fence-isolation land: run gates 3 (injected-inf), 5 (THE
  measurement, distil@512 + 124M), 6 (memory K=1 vs K=2), re-baseline 10.
- N>1 grad accumulation (micro/apply split) and the scaler-tensor injected-inf
  arm remain documented follow-ons (inc-2b/§2b).

**SLOC:** +~95 src (ring backpressure/drain/runahead-option in capture.ts;
`_deferBoundaryCommit` + `_commitStepBoundaryGen` split in torchlette.ts). Zero
new env flags (`runahead` + `ringDepth` are capture OPTIONS). No deletions (the
ring EXTENDS 2a's `pushRing`; the boundary split EXTENDS the implied-boundary
commit).

---

## INC-3 COMPLETE (2026-07-10): both blockers landed post-fix-84-merge —
## K≥2 runahead SHIPS. Distil@512: uncaptured 223 → ringK2 186.5 ms/step
## (−16.4%), at the GPU floor, trajectories in the noise band, ring peak
## BELOW uncaptured.

**The two blockers, as landed (pool/tracker surface in scope after fix-84):**

1. **POOL/PLANNER-EXCLUDED READBACK STAGING (layer #4).** At ring push, each
   scalar output is copied NOW — in queue order, right after the step's plans,
   before any newer step can rebind the live planner slot — into a dedicated
   MAP_READ staging buffer that never enters the pool and is invisible to the
   planner (extends the `startScalarReadback`/`startItemReadback` primitive per
   CLAUDE.md's "dedicated readback staging" note — the exact requirement the
   four historical loss-overlap attempts lacked). `Tensor._stagedScalarRead`:
   `cpu()`/`item()` prefer the staged copy over the live buffer AND skip the
   exec-lock entry point entirely (historical failure #3 structurally dead).
   Cached finish (first read maps + `deferredDestroy`s the buffer); never-read
   buffers are reclaimed at ring expiry (idempotent fire-and-forget). This is
   also what makes the deferred readback FAST: the staged copy's mapAsync
   resolves after only ITS step's GPU work, not the newer in-flight steps'.

2. **PER-SETTLE FENCE ISOLATION.** `captureIsolatedFence()` (buffer-pool.ts):
   each settle awaits its OWN `onSubmittedWorkDone` promise captured at
   commit-issue time — the shared single-slot fence is still issued (non-ring
   paths byte-identical) but is overwritten by the next step before a K≥2
   settle runs; awaiting it would over-cover (serialize). CRITICALLY the
   isolated awaiter does NO pool bookkeeping: `flushPendingToPool` at a
   settle would promote buffers released during the still-in-flight next
   step's build — the exact #84 run-boundary aliasing class. Pool promotion
   happens only at true quiescent points: a full markStep, or the ring's
   `drain()` (`_ringQuiesce` — fence everything, then promote).

**GradScaler made ring-compatible (the found-inf reporting channel):** the
shared persistent inf flag is (a) OVERWRITE-ZEROED by the next `unscale_`
before a deferred readback could see it, and (b) never zeroed at all on tape
HITS (the body never runs — it would latch 1 forever after the first inf).
`snapshotAndResetInfFlag` (unscale-kernel.ts) snapshots the flag into a
pool-excluded 4-byte staging buffer AND re-zeroes it, both in queue order;
`GradScaler.snapshotDeferred()` (driver, once per step, after the captured
call) queues per-step reports; `resolveOldestDeferred()` resolves them at the
K-behind cadence (mapAsync self-sync — no markStep, no shared fence, legal
mid-ring). The per-element zero-MASK in the unscale kernel is unchanged and is
the trajectory-exactness carrier (in-graph, replay-faithful). `resolveDeferred`
(serial) is byte-identical.

**Gate ladder (final code; model-scale gates on DEVICE 10/11 — device 2 was
squatted by a foreign 25.6GB idle allocation that made model-scale runs
silently OOM-drop submits, discovered when all arms of the first measurement
read loss=0 from step 1; tiny-model gates were unaffected):**
1. build: PASS.
2. K-parity (t-ring-probe): serial == ringNow == ringK1 == ringK2, ALL
   maxΔ=0.00e+0 (bit-identical), body frozen on hits, zero GPU errors — K is a
   pure knob. PASS.
3. INJECTED-INF (t-ring-inf-probe): inf grads at step 12 (a HIT-era step — the
   mask ran inside a REPLAYED plan): K=1 == K=2 bit-identical incl. the Inf
   loss; CPU scale mirror backed off 1024→512 at EXACTLY call INF_STEP+K per
   arm (lag bound ≤K asserted precisely, not absence); recovery clean. PASS.
4. Drain-on-abort (capture.spec "[inc-3] drain mid-ring"): PASS.
5. THE MEASUREMENT (t-ring-measure, distil@512, 30 steps, device 10, V100,
   near-solo; trajectories all ≤6.4e-4 of uncaptured):
   | arm | wall/step (late half) | peak MB |
   |---|---:|---:|
   | uncaptured (plain fullstack) | 223.2 | 8784 |
   | serial captured (inc-2b) | 215.8 | 10268 |
   | ringK1 (no overlap, by design) | 222.4 | 8251 |
   | **ringK2 (runahead)** | **186.5** | **8251** |
   **−16.4% wall vs uncaptured (−13.6% vs serial), AT the GPU floor** (steady
   ~186ms ≈ GPU/step; occasional ~30ms steps show the pipeline's headroom).
   vs G0(b)'s ~30% (236→165) prediction: the MECHANISM saturates exactly as
   predicted (wall collapses to the GPU floor); the pie is smaller than G0
   measured because G0 decomposed the LOWERED path — inc-2b's build-skip
   already banked part of the CPU-overlappable slice. Above the ~15% G-perf
   pause bar.
   **gpt2-medium@512 on the 32GB V100 is OUT OF THE RING'S MEMORY ENVELOPE —
   the §2 honest-cost boundary, measured:** uncaptured peaks 29.5 GB (<2.5 GB
   headroom — less than one step's transient), so ANY runahead depth (even
   K=1's one-step-deferred sweep) hits `VK_ERROR_OUT_OF_DEVICE_MEMORY` →
   dropped submits → the ring arms' trajectories collapse while the serial
   arms train (uncaptured 1078 ms/step, serial 1009). Exactly the chartered
   claim: "the win is real only where memory headroom for K≥2 exists." No
   pressure-reactive automation (per ruling) — the driver chooses K knowing
   the envelope.
   **Second in-envelope config — distil@1024 (heavier GPU/step):**
   | arm | wall/step | peak MB |
   |---|---:|---:|
   | uncaptured | 300.7 | 11667 |
   | serial captured | 279.4 | 12535 |
   | ringK1 | 284.1 | 10874 |
   | **ringK2** | **234.0** | **10874** |
   **−22.2% vs uncaptured (−16.3% vs serial)**, trajectories ≤7.2e-4, ring
   peak again BELOW uncaptured.
6. MEMORY: ringK2 peak == ringK1 peak == 8251 MB — BELOW uncaptured (8784) and
   serial-captured (10268). The K× worst-case did NOT materialize for this
   config: the gen-scoped per-step sweeps + planner-fixed replay buffers bound
   the in-flight set; the +K cost is the batch uploads + staged scalars (KB).
7. capture.spec: 13/13 both flag states ("[inc-3]" now gates K=1 AND K=2
   bit-parity in-suite).
8. t-train-tape-probe (device 11): eligiblePairs=7, refusals=0, tapeCount=1
   PASS. t-train-capture-probe (device 11): hits=14, refusals=0, bodyFrozen,
   maxLossDelta=5.97e-4, REAL losses (12.72→10.8) PASS.
9. Full suite BOTH flag states (device 11): flag-ON exit 0 (webgpu 910
   passed/2 skipped + cpu green); flag-OFF exit 0 (cpu 1173/1 + webgpu 898/14
   — incl. fix-84's in-suite second-run-determinism gate). test:gates 4/4.
10. 124M regression (diloco-regression-check, device 10): PASS baseline-exact
    (9.8089/5.9223/5.1527/4.6396 vs {9.81,5.92,5.15,4.64}), peak FLAT
    2087.6 MB, zero growth.
11. Decode stack: kv-differential PASS (cat + static + taped identical, taped
    13/13 hits, 0 invalidations; needed a node_modules/torchlette self-link in
    the worktree; exit-139 = benign Dawn teardown after complete output).
    gen-tape-gate DEFERRED-TO-MERGE (@huggingface/transformers not installed
    in any local node_modules — coordinator-anticipated worktree/deps issue).
12. STRICT_LIFETIME+STRICT_GPU on captured ringK2 fullstack (autocast +
    checkpoint + scaler + clip + AdamW, distil@512, NOSCHED): exit 0, ZERO
    warns/throws, hits=19, wall 176.4ms. PASS.

**KNOWN ISSUE (scoped, documented in t-ring-measure):** a PER-STEP DRIVER-LEVEL
LR scheduler under RUNAHEAD produces ONE warmup-era (trace-phase) transient
`[lifetime] reading RECLAIMED storage` warn on the setLR chain (id stable
across runs; trajectory impact below the cross-process noise floor — measured
maxΔ 3.9e-4 WITH the warn). It is the [2b-sched]/dangling-copy_ family (the
plain-path per-step-scheduler fragility already quarantined with
t-sched-plain-repro). Two structured fix attempts (pre-body forceAllPending,
both orderings) each traded it for a worse failure (promotion mismatch → hits=0
/ missShape=20: the pre-body plan changes the recorded step structure) and were
reverted — the fix belongs in the setLR delivery (a declared scalar-slot write
into a FIXED buffer, the TAG_WRITE idiom, as the [2b-sched] stop-report already
recommended), not in the ring. STRICT gate runs NOSCHED until that lands.

**SLOC (both blockers + scaler ring-compat):** +~130 src on top of the partial
(staging: tensor field + cpu/item override + `_startRingScalarReadback` +
`_ringQuiesce`; fence: `captureIsolatedFence`; scaler: snapshot queue + 2
methods + kernel snapshot fns + 2 backend ops). Zero new env flags. One
pre-existing main bug fixed en route (executor.ts `captureActionLayouts`
crashed on reshape-of-SCALAR-ref — null backendInputs[0] — under the
build-from-IR default; tiny-model+GradScaler repro, flag-independent).

---

## INC-3 CLEANUP TAIL — ITEM 1 (setLR TAG_WRITE fixed-buffer delivery):
## ATTEMPTED, STOPPED. The buffer-stable delivery is NOT free-standing —
## a raw `queue.writeBuffer` lr write is an OUT-OF-GRAPH op that loses the
## submit-ordering the `copy_` scatter provides, and it perturbs even the
## UNCAPTURED trajectory. The doc's own "off-by-one submission-order hazard"
## warning (the [2b-sched] stop-report, `:1099-1101`) is the whole story.

**Mechanism attempted (the doc's prescription, built in full):** a pinned,
non-owning f32 [1] materialized buffer per lr group (`createStableScalarStorage`
in a new `src/backend/webgpu/stable-scalar.ts`, surfaced as optional
`Backend.createStableScalarStorage`/`writeStableScalarStorage` methods +
`RuntimeEngine.createStableScalar`/`writeScalarInPlace`). `Adam` created its lr
tensors via `createStableScalar` and `setLR`/`setGroupLR` delivered via
`writeScalarInPlace` (a pure `device.queue.writeBuffer` into the fixed buffer,
same materialized ref every step → NO graph node, NO wandering buffer, NO
`stridedScatterCopy` per step). The recorder saw an unchanged ordered plan-fp
sequence (setLR contributes zero plans — structure-preserving as required).

**GATE 1 — the warn — PASSED.** Under `STEP_TAPE=1 STRICT_LIFETIME=1`, ringK2 +
CosineAnnealingLR (`t-ring-measure RING_MEASURE_ARM=ringK2`, un-NOSCHED'd): the
`[lifetime] reading RECLAIMED storage id=… (shape=[1])` warn went from 1 → **0**,
exit 0, zero throws. Tape still formed and hit (`tapeCount=1, eligiblePairs=3,
refusals=0, hits=3`). K-parity 0.0 preserved (t-ring-probe unchanged, no lr in
that toy). So the warn's proximate cause (the copy_'s replace-and-hold minting a
new lr buffer that the deferred sweep reclaims) IS what the stable buffer
removes.

**GATE — schedule-exactness — FAILED, and this is the STOP.** With a VARYING lr
the trajectory diverged ~0.024 vs the control in `t-train-capture-probe` (clean
HEAD: 1.2e-3, noise floor). Root-caused by ELIMINATION, in order:
1. **Constant lr (`ETA_MIN=1e-4`) → 6.6e-4 PASS.** The stable-buffer STRUCTURE
   is correct; the divergence is specific to per-step-VARYING lr.
2. **`COMPILED_PLAN=0` (lowered replay) → 5e-4** (hits=0, but trajectory
   correct). So it is NOT a lowered-delivery bug.
3. **The DECISIVE test — an UNCAPTURED fullstack A/B, my-build vs clean HEAD,
   WITH the schedule** (`t-ring-measure RING_MEASURE_ARM=uncaptured`): step-13
   loss **11.0504 (mine) vs 11.0424 (clean HEAD) — Δ≈0.008, growing across
   steps.** The divergence exists with **NO capture, NO tape, NO replay** —
   pure lowered training. It is a **DELIVERY-ORDERING bug**, not a replay bug.

**Why:** `copy_(lrT, tensorFromArray([lr]))` is a GRAPH op — a `stridedScatterCopy`
node the plan orders relative to the optimizer's read of lr (and forces in the
per-step chain). A raw `queue.writeBuffer` has NO graph edge: it lands in queue
order, which is NOT guaranteed to sit after the prior step's optimizer read of
the lr buffer nor before this step's — the exact "off-by-one submission-order
hazard" the [2b-sched] stop-report named (`:1099-1101`). An unconditional
`flushSharedEncoder` before each write did not fix it (Δ unchanged 0.0236): the
hazard is at the driver↔plan submit boundary, not the shared-encoder window.

**Why lr also lands as a `persistent` (static-singleton) compiled slot** (a
second, orthogonal wrongness surfaced): the pinned buffer is unmapped in the
arena at record time, so `buildCompiledPlan`'s `persistentSlot` classifies it as
a static cache (like the expm1 constants) rather than an `external` re-resolved
input (which is how `_t` and clean-HEAD's copy_-delivered lr flow). Making it
truly free-standing would require the compiled recorder to classify a pinned
driver-scalar buffer as EXTERNAL — a compiled-plan-classification change, not a
delivery-only change.

**Ruling (STOP-rule honored):** the fix as prescribed ("declared scalar-slot
write into a FIXED buffer, the TAG_WRITE idiom") is NOT delivery-local — a
buffer-stable lr that keeps `copy_`'s submit-ordering needs an IN-PLACE ORDERED
scatter (write into the destination's own buffer as a graph op, so the plan
still orders it) OR the lr must ride the true TAG_WRITE stable-buffer channel
the batch args use (which IS graph-ordered — but that channel is for
tensorFromArray upload NODES, and lr today is a persistent-tensor read, not an
upload slot). Both are structure-touching changes with a measured
uncaptured-trajectory regression on the naive attempt — precisely the
"structure-changing fix for item 1 → STOP" and "unexplained divergence → STOP"
conditions. **NOT LANDED.** The `[lifetime]` warn remains the documented benign
warmup-era transient; the STRICT gate keeps NOSCHED until a graph-ordered
buffer-stable lr write lands (next session: an in-place ordered scatter primitive
that reuses the destination buffer, gated against the UNCAPTURED-trajectory A/B
above — that A/B, not just the captured probe, is the load-bearing gate the
naive attempt failed). Working tree reverted to clean HEAD; the prototype is
preserved at `scratchpad/stable-scalar-item1.ts` + `scratchpad/item1.diff`.

---

## FAST TRAINING LOOPS — the runahead loop as living documentation

The fastest training loop torchlette offers is a whole step captured with
runahead: `api.capture(fn, { training: true, runahead: 2 })`, driven so the CPU
builds+submits step N+1 while the GPU drains step N. The canonical, heavily-
commented example is **`examples/fast-training/fast-training.ts`** — it runs a
small GPT-2, prints per-step loss + ms/step, and demonstrates a SERIAL variant
alongside so the ring engaging (hits > 0) and the wall-clock win are visible.

Run it (pin a FREE device — Dawn ignores CUDA_VISIBLE_DEVICES):

    VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
    TORCHLETTE_STEP_TAPE=1 npx tsx examples/fast-training/fast-training.ts

Measured on a V100 (device 10, distilgpt2, seq 256, K=2): **SERIAL 198.3 ms/step
→ RUNAHEAD 159.4 ms/step (−19.6%), ring hits=9, trajectory maxΔ=0.0** (bit-
identical — K is a pure knob). The four rules the example is annotated with:

1. **Await the CALL, not the loss.** `const h = await step(x, y)` awaits the
   submit, not GPU completion; `h` is an UNREAD ring handle.
2. **Read K-behind** (collect-and-drain). Read the loss from K steps ago (it has
   fenced). `await-every-N` is the logging cadence — exactly ONE fence per N.
3. **The ring owns the boundary.** Under `runahead` the driver does NOT call
   `markStep` per step; it MUST `await step.drain()` at the end.
4. **GradScaler rides the same cadence.** found-inf is DATA (in-graph where-
   select), `snapshotDeferred()` + `resolveOldestDeferred()` K-behind — never a
   per-step readback, so it does not cap K.

**What NOT to do:** `await loss.item()` on the per-step critical path fences
every step = voluntary K=1 (correct but slow — no runahead overlap). The example
spells this out. **Honest cost:** runahead trades K× in-flight memory for the
wall win; on a memory-tight config (a model near the device ceiling) K may be
forced to 1 = no runahead. The win is real only where headroom for K≥2 exists.

---

## INC-3 CLEANUP TAIL — ITEM 2 (scaler-as-tensor): BLOCKED on the same
## compiled-replay scalar-delivery prerequisite as item 1. The bug it retires
## is PROVEN and now has a permanent retirement gate.

**The bug is REAL and correctness-affecting** (`tools/t-scaler-growth-probe.ts`,
serial capture, GradScaler growthInterval=4, no inf → scale doubles every 4
steps): once the tape starts hitting (bodyRuns=6/24), the captured scale FREEZES
while the control keeps growing:

    control  scales: 4,4,4,4,8,8,8,8,16,16,16,16,32,...,128
    captured scales: 4,4,4,4,8,8,8,8, 8, 8, 8, 8, 8,..., 8   ← FROZEN at record-time
    captured trajectory diverges: maxLossΔ = 1.41e-2 (hits=18, refusals=0)

**Two coupled causes, both requiring the item-1 prerequisite:**
1. **The scale UPDATE doesn't run on hits.** `_scale` is a JS number updated by
   `_applyScaleAdjustment` (JS if/else), reached via `scaler.update()` /
   `resolveDeferred()`. In the serial captured path `update()` is INSIDE the
   body (never runs on a hit) and `resolveDeferred()` is a no-op on a hit
   (`_pendingInfBuffer` is set by the JS `unscale_`, which also never runs) — so
   the CPU scale mirror freezes. The inc-3 `snapshotDeferred`/`resolveOldest`
   found-inf channel was built for the RUNAHEAD path; the serial capture path
   has no per-hit found-inf report.
2. **`scale(loss) = mul(loss, jsNumber)` bakes the scale as a graph SCALAR**,
   frozen at record time. Even if cause 1 were fixed (CPU mirror advancing), the
   REPLAYED graph would still multiply by the RECORDED scale. Fixing this needs
   scale/invScale as PERSISTENT TENSORS read live by the replay — which is the
   **exact compiled-replay scalar-delivery problem that STOPPED item 1**: a
   persistent f32[1] buffer read by a compiled replay classifies as a static
   `persistent` slot and does not pick up per-step writes correctly, AND a raw
   `queue.writeBuffer` scale update loses the graph-ordering (item-1's measured
   uncaptured-trajectory regression). The scale tensor is the same shape and the
   same delivery channel as the lr tensor item 1 failed to make buffer-stable.

**Ruling:** item 2 is BLOCKED on item 1's prerequisite (a graph-ordered in-place
scalar write into a fixed buffer whose compiled-replay reads are live, not a
frozen `persistent` slot). Implementing scale-as-tensor before that lands would
reproduce item 1's uncaptured-trajectory regression on the scale path. **NOT
LANDED.** What DID land: `tools/t-scaler-growth-probe.ts` — the retirement GATE
(FAILs on main documenting the frozen-scale bug with a pointed message; goes
green when item-2 lands). Sequencing for the next session: solve the item-1
graph-ordered buffer-stable scalar write ONCE (a shared primitive), then BOTH lr
(item 1) and scale/invScale (item 2) ride it; the growth probe + the item-1
uncaptured A/B are the two load-bearing gates.

---

## INC-3 CLEANUP TAIL — ITEM 3 (N>1 grad accumulation, micro/apply split):
## ATTEMPTED, STOPPED — TWO real blockers surfaced (recorder-step model +
## shared-encoder cross-capture hazard). The boundary-suppression piece is
## trivial; the recorder/encoder work behind it is not.

The substrate exists (autograd accumulates into `.grad`, e9f7943; the
distributed trainer's manual `accumSteps` loop is the reference: N micro
forward+backward each scaled 1/N — MEAN convention — then ONE step+zeroGrad in
one beginStep so `.grad` never crosses a markStep). A `CaptureOptions.microStep`
flag suppressing the micro's boundary queue (only the once-per-cycle `apply`
capture commits) was prototyped and IS trivial (one guarded
`queueStepBoundary`). But a probe of two captures per cycle (`micro` reused N×
+ `apply` once) surfaced TWO real blockers, BOTH stopping the tape from forming
(`microHits=0, applyHits=0`):

1. **The recorder-step model is ONE-capture-per-boundary.** The recorder tracks
   a single step's ordered plan-fp sequence and promotes at the boundary
   (markStep). With `micro` and `apply` BOTH contributing plans to ONE step
   (there is no markStep between the N micros and the apply — that IS
   accumulation), the recorder cannot segment which plans belong to which
   capture, so neither forms its own tape. The micro tape must form by comparing
   micro-of-cycle-C vs cycle-C+1, but the boundary is per-CYCLE — a per-capture
   plan segmentation the recorder does not have.
2. **Shared-encoder cross-capture read-write hazard.** With two captures per
   boundary the `.grad` accumulation (micro writes `.grad`) and the optimizer
   read of `.grad` (apply) land in the SAME shared-encoder synchronization scope
   with no intervening flush — Dawn rejects it: `usage (Storage(read-write)|
   Storage(read-only)) includes writable usage and another usage in the same
   synchronization scope` → invalid command buffer. The single-capture
   whole-step path never hits this (one body, one encoder lifecycle).

**Ruling:** option (a) micro/apply needs (i) per-capture plan segmentation in
the recorder within a shared boundary and (ii) an inter-capture shared-encoder
flush — real mechanism, not one-liners. NOT LANDED (prototype + `microStep` flag
reverted). **A tractable alternative exists** — design option (b): a SINGLE
whole-step training capture whose body runs the N micro-backwards + the step
(one capture per boundary, no recorder/encoder change; re-records per distinct N
as a plain-value cold bucket — for a FIXED N, one tape). INC-2B rejected (b) for
reuse, not correctness; given (a)'s two blockers, (b) is the pragmatic next-
session landing, with accumSteps=1 byte-identity (micro==apply, one body) as the
hard invariant either way.

---

## LIVE SCALAR SLOT primitive + Consumer 1 (setLR) — LANDED

The `docs/staged-execution-phase2b.md:1093` next-session note ("make the
optimizer-scalar write BUFFER-STABLE by construction ... then no rebind
machinery is needed") is realized as `src/core/live-scalar.ts` — ONE primitive
that owns a fixed-buffer persistent f32[1] tensor and delivers per-step values
into it as DATA. It rides the EXISTING scalar-write re-dress seam
(`scalarDresses` + `scalar-slots.ts`), consolidating the ad-hoc
`copy_(lrT, tensorFromArray) + noteScalarSlotValue` that was open-coded in each
optimizer.

**The three clauses (each the negation of a historical bug):**
1. **GRAPH-ORDERED.** `RuntimeEngine.setScalarInPlace` writes the value via an
   in-place `stridedScatterCopy(dst, tensorFromArray([v]))` — a PLAN NODE
   ordered against consumers' reads, NOT a raw out-of-graph `queue.writeBuffer`.
   The item-1 prototype's raw writeBuffer measured 0.008 UNCAPTURED divergence
   (racing encoded-unsubmitted reads); this passes the ITEM-1 A/B at 6e-4.
2. **FIXED BUFFER.** The scatter's TRUE-IN-PLACE DMA (strided-scatter.ts) writes
   dst's EXISTING buffer, so its identity is stable across record/replay. THE
   FALSIFICATION that redirected the design: a fresh-buffer `tensorFromArray`
   upload (the naive "expose the arg channel" reading) SILENTLY CORRUPTS a large
   plan's high-fan-out readers — the packed adam (`fused.ts:314`) binds
   `items[0].lr` ONCE at record time to the tfa's arena buffer; on replay the
   TAG_WRITE writes a SEPARATE pinned stableBuf that never feeds the recorded
   dispatch's slot, so ~50 adamStep readers read a pool-reused record-time
   buffer → measured 0.04 loss divergence on the real 124M model, present even
   with a CONSTANT lr (buffer provenance, not value). Root-caused via
   `TORCHLETTE_PACKED_ADAM=0`/`COMPILED_PLAN=0` (both make it vanish). The
   in-place scatter into dst's own buffer records as a PERSISTENT slot the
   packed dispatch binds correctly.
3. **LIVE REPLAY READS.** The recorded scatter re-executes its `tensorFromArray`
   source from the current `scalar-slots.ts` host value each replay.

**Consumer 1 — setLR.** `Adam._lrLive` are `LiveScalar`s; `setLR`/`setGroupLR`
call `.set(lr)`. `_lrTensor(gi)` returns the LiveScalar's persistent tensor,
read by adamStep / the elementwise+foreach update as before.

**Gates (device 10/11, sivri):** falsification probe (channel is ordered,
buffer-stable, live across hits) PASS; ITEM-1 uncaptured A/B (my-build vs clean
HEAD, WITH CosineAnnealingLR) maxΔ 6e-4 PASS (the gate that killed the
prototype); t-train-capture-probe (schedule through hits) hits=17 refusals=0
maxΔ 8e-4 PASS; capture.spec 13/13 incl. [2b-sched]; test:gates 4/4; 124M
regression baseline-exact 9.81/5.92/5.15/4.64 peak-flat 2087.6 MB; CPU suite
1163 pass. Schedule-exactness compiled==sequential rides the train-capture
parity.

**KNOWN OPEN (NOT closed by this primitive):** under RUNAHEAD (ringK2) + a
per-step LR scheduler, ONE warmup-era transient `[lifetime] reading RECLAIMED
storage id=… (shape=[1])` warn remains on the setLR scatter (the deferred/
gen-scoped sweep demoting the scatter's per-step source during the pre-tape
lowered phase; trajectory correct, ringK2 maxΔ ~1e-3). This is UNCHANGED from
main's copy_ delivery (the doc's pre-existing KNOWN ISSUE `:1497`) — the
primitive did not regress it, but did NOT close it either. Attempts that DID NOT
work: (a) a single in-place `writeScalar` fill op with no scatter source —
correct serially, but its in-place READ of dst's buffer is reclaimed+reused
under runahead's deferred timing → 0.035 ringK2 divergence (reverted); (b)
per-dst persistent scatter-source + snapshot adoption — the gen-scoped sweep
still demotes across the driver/body gen boundary. The real fix is
ARCHITECTURAL: route the driver-level scalar through the ring's lifetime
management the way batch ARGS are (they don't warn) — a ring-integration change,
not a delivery-op change. STRICT keeps NOSCHED until that lands.

**Consumer 2 (scaler-as-tensor) — NOT LANDED this session.** The forward
`scale()` can ride a LiveScalar (neutral), but the growth-retirement gate
(`t-scaler-growth-probe`) additionally requires (i) the CPU scale mirror to
advance on tape HITS (a persistent found-inf buffer the driver resolves each
step, not the body-set `_pendingInfBuffer` that never runs on a hit) and (ii)
`invScale` delivered as a live tensor into the FUSED unscaleGrad kernel (today a
frozen uniform) — a kernel change. Both are substantial sub-features beyond the
scalar-delivery primitive; deferred. The gate still FAILs on main as designed.

**SLOC:** +~50 (src/core/live-scalar.ts) + ~30 (RuntimeEngine.setScalarInPlace)
net; Adam's per-optimizer copy_+note deleted (net-neutral). No new env flags.
Retirements: the ad-hoc per-optimizer `copy_(lrT, tensorFromArray)+noteScalarSlot`
delivery (now the ONE primitive). scalar-slots.ts + scalarDresses are KEPT (the
primitive rides them — the proven INC-2B seam — rather than a parallel one).

---

## SCALER-AS-TENSOR (Consumer 2) — LANDED; setLR ringK2 STRICT un-NOSCHED'd

The `t-scaler-growth-probe` retirement gate is GREEN: a growth-interval
crossing under capture keeps HITTING and the captured trajectory is
BIT-EXACT against the uncaptured control (maxLossΔ 0.00e+0, scale 4→128 in
lockstep). The last known frozen-scalar exposure (TORCHLETTE_STEP_TAPE=1 +
capture + GradScaler + growth crossing) is retired.

**Design (one live tensor, reciprocal in-kernel):**
- `GradScaler._scaleLive` (a `LiveScalar`) is THE scale. `scale(loss)` =
  `mul(loss, scaleLive.tensor)`; the fused `unscaleGrad` takes the SAME tensor
  as `node.inputs[1]` and reciprocates `invScale = 1/scale` IN-KERNEL (the
  adam inc-2a `t`/`lr` storage-read treatment). RETIRED: the `inv_scale`
  config-uniform field + its volatileRepack closure (unscale-kernel.ts) + the
  TAG_UNIFORM repack in stream-generate — the unscale config is now fully
  STATIC. The CPU `_scale` number is a stats-only mirror.
- REJECTED alternates (measured): a SECOND live invScale tensor (two driver
  scatters per body; the re-dress covers one — the second's recorded source
  goes stale); an in-graph `div(1, scale)` node (its step-temp output becomes
  a row-program external and added ~5x compiled-plan fp drift on the
  train-capture gate).
- Driver-resolved found-inf: `resolveDeferred` splits MISS (old
  markStep+readback — forms the tape in implied-boundary loops) from HIT
  (advance the mirror via the inc-3 snapshot ring or growth-only bookkeeping,
  NO markStep — an extra boundary on hits perturbs the implied-boundary fp
  regime and resets the recorder). K-behind ring cadence preserved; found-inf
  never rides the per-step critical path.
- `_applyScaleAdjustment` writes the live tensor only on an ACTUAL change
  (scale holds constant between growth/backoff events; every write mints a
  per-step scatter).

**setLR ringK2 STRICT — the NOSCHED caveat is CLOSED.** The warmup transient
(`:1497`/`:1589`/`:1784` above) was TWO short-lived storages of the driver
write chain destroyed between the driver's `set()` and the ring's DEFERRED
boundary commit: (1) the scatter SOURCE (ownerless `tensorFromArray`, rc=0
after its plan claim), (2) the PRE-WRITE dst handle (the write's
`_updateLazyRef` releases the owner claim while deferred consumers — the
ring's adamStep — still hold materialized refs to it). `LiveScalar` now PINS
both across the ring window (`_pinRing`, FIFO 8; `setScalarInPlace` returns
the tracked source). Gate: `RING_MEASURE_ARM=ringK2` + CosineAnnealingLR +
STRICT_LIFETIME + STRICT_GPU = ZERO events, hits=18, sane trajectory.

**Two structural fixes this surfaced (both landed):**
- **rc retain/release LEDGER** (node-factory.ts): `releaseNodeInputRefs` now
  releases EXACTLY the ids `retainPlanInputRefs` recorded on the node
  (`_retainedInputIds`), not an inputs-derived recomputation — mid-force graph
  rewrites (`redirectConsumers`: CSE, identity-cast/mul-by-1 bypass, re-applied
  on template hits) substitute MATERIALIZED refs between retain and release,
  and the phantom release destroyed rc=1 persistent scalars under their
  readers. Single source of truth at the seam.
- **row-program stale-external fallback** (segment-executors.ts): a
  row-program action's recorded inputRefs only remap PENDING refs on template
  reuse (materialized got pos=-1, "no remapping needed" — false for step-temp
  externals, e.g. the 0-d grad seed of scale-as-tensor's mul). A reclaimed
  materialized external now takes the established safe sequential fallback
  instead of silently reading the dead record-era storage.

**KNOWN OPEN (named, root-caused, out of this campaign's seam):**
- `t-train-capture-probe` maxDelta ~2.1e-2 (deterministic) vs the 2.5e-3
  threshold. The threshold is not reliably achievable on this box even on
  main (measured main: FAIL 4.7e-2 / PASS alternating, plus zero-loss
  control-arm episodes — the documented plain-path memory-pressure flake).
  hits/refusals/bodyFrozen are unchanged and both arms are STRICT-clean; the
  residual is honest replay-vs-fresh drift plus (suspected) a recorded
  compiled slot binding a record-era step-temp 0-d external on the tape arm
  — the same class the sequential fallback fixes on the lowered path; the
  compiled/tape-side freshness for recorded step-temp externals belongs to
  the executor/islands surface.
- `t-scaler-growth-probe` under STRICT_LIFETIME: ONE warmup-era transient
  remains (uncaptured CONTROL run, step ~2). Root cause fully traced: the
  storage-tracker's single-slot owner backref (`tensorWeakRefs`) is HIJACKED
  by the autograd retention clone (`_cloneForRetention`) materializing into
  the persistent scale tensor's storage; the demotion sweep then judges the
  storage by the (disposed, not-in-snapshot) clone and releases the owner's
  claim (`stepScoped`) under its readers. Bytes stay correct (fixed buffer;
  trajectory Δ=0) — the throw is a defensive true-positive on claim
  attribution, not data. THREE surgical fixes were built and measured — clone
  noTrack, yield-to-live-claimant tracking, disposed-target sweep skip — each
  cures this probe and each breaks `test/implied-step-boundary.spec.ts`
  (mixed-usage values 0.48 / [8,16] STRICT trio): the sweeps' claim
  attribution via last-tracker-wins is LOAD-BEARING in contradictory
  directions. Needs a claim-attribution repair in the storage tracker (owner
  set or claim-holder-aware sweeps) — a dedicated campaign, not a patch.
