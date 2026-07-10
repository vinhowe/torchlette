/**
 * ============================================================================
 * FAST TRAINING LOOPS — the runahead capture ring, as living documentation.
 * ============================================================================
 *
 * This is the canonical example of the fastest training loop torchlette offers:
 * a whole training step (forward + backward + optimizer) captured as ONE
 * `api.capture(fn, { training: true, runahead: 2 })` call, driven so the CPU
 * builds+submits step N+1 while the GPU still drains step N. On an in-envelope
 * config that is the G0(b) ~15-30% wall win (docs/staged-execution-phase2b.md):
 * the wall collapses to the GPU floor because the per-step CPU work (graph
 * build, plan collect, the loss readback, the boundary fence) is hidden behind
 * the GPU fence of the previous step.
 *
 * It RUNS (a small GPT-2, a handful of steps, prints per-step loss + ms/step)
 * and demonstrates a SERIAL variant alongside so you can see the ring engage
 * (hits > 0) and the wall-clock improvement.
 *
 * Run (GPU box; pin the device so Dawn uses a FREE GPU — it ignores
 * CUDA_VISIBLE_DEVICES):
 *
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 npx tsx examples/fast-training/fast-training.ts
 *
 * TORCHLETTE_STEP_TAPE=1 is REQUIRED: capture rides the step-tape (flag off ⇒
 * `capture` is a transparent pass-through, correct but not fast).
 *
 * ────────────────────────────────────────────────────────────────────────────
 * THE FOUR RULES OF A RUNAHEAD LOOP (read these; the code below is annotated
 * with each):
 *
 *   1. AWAIT THE CALL, NOT THE LOSS.  `const h = await step(x, y)` awaits the
 *      submit of step N, not its GPU completion. The returned `h` is a ring
 *      handle to the loss — UNREAD. Do NOT `await h.item()` on the per-step
 *      critical path (see rule 4 / the "what NOT to do" section): that would
 *      fence every step and voluntarily degrade you to K=1 (serial).
 *
 *   2. READ K-BEHIND (collect-and-drain).  Read the loss handle from K steps
 *      ago, not this step's. By then step N-K has fenced (the ring guaranteed
 *      it), so `api.cpu(handle)` returns promptly and does not stall the
 *      pipeline. `await-every-N-steps` is the logging cadence: reading only
 *      every N steps costs exactly ONE fence per N steps (stated in the code).
 *
 *   3. THE RING OWNS THE BOUNDARY.  Under `runahead`, the driver must NOT call
 *      `api.markStep()` per step — the ring defers each step's boundary
 *      fence+sweep K deep and runs it under backpressure. You MUST call
 *      `await step.drain()` at the end of the loop to fence + sweep the last K
 *      in-flight steps in order.
 *
 *   4. GRADSCALER RIDES THE SAME CADENCE.  found-inf is DATA (an in-graph
 *      where-select), never a per-step CPU readback. `scaler.snapshotDeferred()`
 *      queues a per-step report; `scaler.resolveOldestDeferred()` resolves it
 *      K-behind. The CPU scale mirror lags ≤ K steps; the on-device trajectory
 *      is bit-exact for any K (K is a pure knob).
 * ────────────────────────────────────────────────────────────────────────────
 *
 * IMPORTANT (Dawn): this script calls process.exit(0) at the end — Dawn holds
 * background threads that otherwise keep Node alive.
 */

import * as path from "node:path";
import {
  destroyWebGPU,
  getGPUMemoryStats,
  initWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import type { Tensor } from "../../src/frontend/tensor";
import { Adam, CosineAnnealingLR, GradScaler } from "../../src/optim/index";
import { clipGradNorm_ } from "../../src/nn/index";
import { STEP_TAPE_REPLAY } from "../../src/core/step-tape";
import { loadPretrainedGPT2 } from "../gpt2/loader";

const ROOT = path.resolve(import.meta.dirname, "../..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "20", 10);
const K = parseInt(process.env.RING_DEPTH ?? "2", 10); // runahead depth (G0b: 2 saturates)

const log = (m: string) => console.error(`[fast-training] ${m}`);

/** Build a fresh model + optimizer + scaler + scheduler for one run. */
async function buildRun() {
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = await loadPretrainedGPT2(
    api,
    path.join(ROOT, "models", MODEL),
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: 1e-4, weightDecay: 0.01, adamW: true }, api);
  const sched = new CosineAnnealingLR(opt, STEPS, 1e-5);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  // Deterministic pseudo-random batches (the same across arms so the loss
  // trajectories are comparable).
  let s = 0xc0ffee;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff), s);
  const nextBatch = () => {
    const inp = new Int32Array(BATCH * SEQ);
    const tgt = new Int32Array(BATCH * SEQ);
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    return {
      input: api.tensorFromArray(Array.from(inp), [BATCH, SEQ], { device: "webgpu" }),
      target: api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], { device: "webgpu" }),
    };
  };

  // THE CAPTURED BODY — a whole training step. Note what it does and does NOT:
  //  - it takes the batch (input, target) as tensor ARGS → warm upload slots
  //    (their fresh values are re-dressed every replay; the body never re-runs
  //    on a hit).
  //  - it reads model params / optimizer state / the scaler scale as CLOSURE
  //    STATE, mutated IN PLACE by the recorded plans (params, m/v, t, scale).
  //  - it returns the loss (NOT awaited here — the driver reads it K-behind).
  //  - it may `await loss.backward()` (training bodies are async).
  //  - it must contain NO external I/O / non-deterministic awaits (fs, network,
  //    Date.now, a fresh random draw not seeded from a tensor arg): on a HIT the
  //    body does not run, so any such effect would be FROZEN at record time.
  const stepBody = async (input: Tensor, target: Tensor): Promise<Tensor> => {
    const loss = api.tidy(() => {
      const l = api.autocast(
        () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(l);
      return l;
    });
    // The returned loss must SURVIVE backward (backward tears down the autograd
    // graph). A noGrad forward-plan copy is the idiom — its buffer is a plan
    // output the ring pins for the K-window.
    const lossOut = api.noGrad(() => api.mul(loss, 1));
    const scaled = scaler.scale(loss);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    return lossOut;
  };

  return { api, opt, sched, scaler, nextBatch, stepBody };
}

/**
 * SERIAL loop (the correct-but-slow baseline). The driver awaits the loss EVERY
 * step (fences every step) and owns markStep. `capture(..., { training: true })`
 * still skips graph build on hits (the ~3-4% build-skip), but there is NO
 * runahead — GPU and CPU do not overlap. This is what "awaiting the loss
 * per-step" degrades to; it is CORRECT, just slow. We run it to contrast.
 */
async function runSerial() {
  const { api, sched, scaler, nextBatch, stepBody } = await buildRun();
  const step = api.capture(stepBody, { training: true });
  const losses: number[] = [];
  const stepMs: number[] = [];
  for (let stp = 0; stp < STEPS; stp++) {
    const t0 = performance.now();
    await scaler.resolveDeferred(); // this is the per-step markStep + inf readback
    const { input, target } = nextBatch();
    const loss = (await step(input, target)) as Tensor;
    losses[stp] = await loss.item(); // ← per-step fence (the serial cost)
    loss.dispose();
    input.dispose();
    target.dispose();
    sched.step();
    stepMs.push(performance.now() - t0);
  }
  const late = stepMs.slice(Math.floor(stepMs.length / 2));
  return {
    losses,
    wallLate: late.reduce((a, b) => a + b, 0) / late.length,
    hits: step.stats().hits,
    peakMB: getGPUMemoryStats().peakBytes / 1e6,
  };
}

/**
 * RUNAHEAD loop (the fast path). The ring owns the boundary; the driver reads
 * the loss K-behind and drains at the end.
 */
async function runRunahead() {
  const { api, sched, scaler, nextBatch, stepBody } = await buildRun();

  // RULE 3: `runahead: true` hands the step boundary to the ring; `ringDepth`
  // is the runahead depth K (G0b: K=2 saturates the GPU-bound floor — K>2 buys
  // zero throughput, only K× in-flight memory). The driver must NOT call
  // markStep per step, and MUST drain() at the end.
  const step = api.capture(stepBody, { training: true, runahead: true, ringDepth: K });

  const losses = new Array<number>(STEPS);
  const handles: Tensor[] = [];
  const stepMs: number[] = [];

  for (let stp = 0; stp < STEPS; stp++) {
    const t0 = performance.now();
    const { input, target } = nextBatch();

    // RULE 1: AWAIT THE CALL, NOT THE LOSS. This awaits the submit of step N and
    // returns an UNREAD ring handle. Backpressure inside the (K+1)-th call
    // fences the oldest in-flight step before this one submits — so ≤ K steps
    // are ever un-fenced.
    handles.push((await step(input, target)) as Tensor);

    // RULE 4: the scaler's found-inf report is queued per step (a pool-excluded
    // snapshot, in queue order) and resolved K-behind — never a per-step
    // readback, so it does not cap K.
    scaler.snapshotDeferred();

    // RULE 2: READ K-BEHIND. Read the loss from step N-(K-1); by now it has
    // fenced, so this cpu() returns promptly. (This is the per-step logging
    // path; see the await-every-N variant below for the cheaper cadence.)
    const oldest = stp - K + 1;
    if (oldest >= 0) {
      losses[oldest] = (await api.cpu(handles[oldest]))[0];
      handles[oldest].dispose();
    }
    if (stp >= K) await scaler.resolveOldestDeferred();

    input.dispose();
    target.dispose();
    sched.step();
    stepMs.push(performance.now() - t0);
  }

  // RULE 3: DRAIN. Fence + sweep the last K in-flight steps in order, then read
  // any still-unread tail handles + resolve the tail scaler reports.
  await step.drain();
  while (await scaler.resolveOldestDeferred()) {
    /* drain the tail reports in order */
  }
  for (let i = Math.max(0, STEPS - K + 1); i < STEPS; i++) {
    if (losses[i] === undefined) losses[i] = (await api.cpu(handles[i]))[0];
  }

  const late = stepMs.slice(Math.floor(stepMs.length / 2));
  return {
    losses,
    wallLate: late.reduce((a, b) => a + b, 0) / late.length,
    hits: step.stats().hits,
    peakMB: getGPUMemoryStats().peakBytes / 1e6,
  };
}

/**
 * ────────────────────────────────────────────────────────────────────────────
 * WHAT **NOT** TO DO (the anti-pattern, for reference — do not copy this):
 *
 *   const step = api.capture(stepBody, { training: true, runahead: true });
 *   for (...) {
 *     const loss = await step(input, target);
 *     console.log(await loss.item());   // ← WRONG: fences EVERY step.
 *   }
 *
 * Awaiting `loss.item()` on the per-step critical path fences the GPU every
 * step, so the CPU can never run ahead — you have voluntarily degraded to K=1
 * (serial). It is still CORRECT (the numbers are right), just slow: you pay the
 * full GPU fence per step and get none of the runahead overlap. Read K-behind
 * (rule 2) or on an await-every-N cadence instead.
 *
 * AWAIT-EVERY-N (the cheap logging cadence): read the loss only every N steps.
 * Cost is exactly ONE fence per N steps (the read K-behind still returns
 * promptly; the fence you pay is the drain-to-that-step). Sketch:
 *
 *   if (stp % LOG_EVERY === 0 && oldest >= 0) {
 *     losses[oldest] = (await api.cpu(handles[oldest]))[0];   // one fence / N
 *   }
 *
 * ────────────────────────────────────────────────────────────────────────────
 */

async function main() {
  if (!STEP_TAPE_REPLAY) {
    log("SET TORCHLETTE_STEP_TAPE=1 — capture is a pass-through without it.");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }

  log(`config: model=${MODEL} seq=${SEQ} batch=${BATCH} steps=${STEPS} K=${K}`);

  log("running SERIAL (correct-but-slow baseline) …");
  const serial = await runSerial();
  log(
    `SERIAL   wall/step(late)=${serial.wallLate.toFixed(1)}ms hits=${serial.hits} peak=${serial.peakMB.toFixed(0)}MB`,
  );
  log(`SERIAL   losses: ${serial.losses.map((l) => l.toFixed(3)).join(", ")}`);

  log(`running RUNAHEAD (K=${K}) …`);
  const runahead = await runRunahead();
  log(
    `RUNAHEAD wall/step(late)=${runahead.wallLate.toFixed(1)}ms hits=${runahead.hits} peak=${runahead.peakMB.toFixed(0)}MB`,
  );
  log(`RUNAHEAD losses: ${runahead.losses.map((l) => l.toFixed(3)).join(", ")}`);

  // The ring must ENGAGE (hits > 0) and be FASTER than serial on an in-envelope
  // config. Trajectories track to the fp noise floor (K is a pure knob).
  let maxD = 0;
  for (let i = 0; i < STEPS; i++)
    maxD = Math.max(maxD, Math.abs(serial.losses[i] - runahead.losses[i]));
  const speedup = (1 - runahead.wallLate / serial.wallLate) * 100;
  log(
    `RESULT   runahead is ${speedup.toFixed(1)}% ${speedup >= 0 ? "faster" : "SLOWER"} than serial; ` +
      `trajectory maxΔ=${maxD.toExponential(2)} (should be at the fp noise floor); ` +
      `ring hits=${runahead.hits}`,
  );
  if (runahead.hits === 0) {
    log("WARNING: ring did not engage (hits=0) — check TORCHLETTE_STEP_TAPE=1.");
  }

  destroyWebGPU();
  // Dawn holds background threads — exit explicitly.
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
