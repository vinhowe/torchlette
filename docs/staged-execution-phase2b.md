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

