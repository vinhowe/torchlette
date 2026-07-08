# Staged Execution, Phase 2b: `capture()` of a whole training step

**Status:** PROPOSED (design-only, 2026-07-08). Review → hardening → separate
implementation go (the stage-1/stage-3 protocol). NOTHING here is built.
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
