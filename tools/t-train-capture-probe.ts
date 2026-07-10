/**
 * Step-tape phase-2b INC-2B gate: a WHOLE training step captured as ONE
 * `api.capture(..., { training: true })` call. On the 2nd+ eligible call the
 * body must NOT run (the recorded multi-plan sequence — forward/backward/
 * optimizer — replays), the replay HITS (hits>0, refusals=0), and the training
 * trajectory still advances (params update via the replayed in-place ops).
 *
 * RUN-EXACTLY-ONCE WITNESS: a body-side counter increments each time the body
 * runs. Once hits begin it STOPS advancing while the captured-call count keeps
 * growing — proof the body is not re-run on a hit. The loss must keep changing
 * (state advances via replay), and the captured loss must track an uncaptured
 * control run to the parity band.
 *
 * PASS: capture hits > 0 AND replay refusals == 0 AND the body counter froze
 * while calls advanced AND the captured trajectory tracks the control.
 *
 * Run (solo GPU, device 2):
 *   VULKAN_DEVICE_INDEX=2 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 npx tsx tools/t-train-capture-probe.ts
 * Env: MODEL(=distilgpt2) SEQ_LEN(=512) BATCH(=1) STEPS(=24)
 */
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import type { Tensor } from "../src/frontend/tensor";
import { assertInitialLossSane } from "./parity-sanity";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "24", 10);
const log = (m: string) => console.error(`[t-train-capture] ${m}`);

interface RunResult {
  losses: number[];
  bodyRuns: number;
  captureStats: ReturnType<
    ReturnType<Torchlette["capture"]>["stats"]
  >;
  replayHits: number;
  refusals: number;
  vocab: number;
}

async function run(useCapture: boolean): Promise<RunResult> {
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
  // The LR schedule runs at DRIVER level (outside the captured body): setLR
  // writes the per-group persistent lr TENSOR (inc-2a) — per-step-varying DATA
  // the tape covers. This is the gate that proves the inc-2a+2b stack composes.
  const sched = new CosineAnnealingLR(opt, STEPS, parseFloat(process.env.ETA_MIN ?? "1e-5"));
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  let s = 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };

  let bodyRuns = 0;
  // The whole training step as ONE captured (async) call. Batch x/y are tensor
  // ARGS (warm slots); params/optimizer are closure state advanced in place by
  // the replayed plans. bodyRuns is the run-exactly-once witness (frozen on
  // hits).
  const stepAsync = api.capture(
    async (input: Tensor, target: Tensor): Promise<Tensor> => {
      bodyRuns++;
      const loss = api.tidy(() => {
        const l = api.autocast(
          () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
        );
        api.keep(l);
        return l;
      });
      // A detached forward-plan copy of the loss that is NOT an autograd
      // intermediate — it survives backward's graph teardown (which disposes
      // the raw loss) and its buffer survives the backward/optimizer plan
      // replay, so it is the harvestable ring output (surface 3).
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
    },
    { training: true },
  );

  const losses: number[] = [];
  const stepMs: number[] = [];
  for (let stp = 0; stp < STEPS; stp++) {
    const t0 = performance.now();
    await scaler.resolveDeferred();
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    let loss: Tensor;
    if (useCapture) {
      loss = (await stepAsync(input, target)) as Tensor;
    } else {
      bodyRuns++;
      const l = api.tidy(() => {
        const ll = api.autocast(
          () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
        );
        api.keep(ll);
        return ll;
      });
      const lossOut = api.noGrad(() => api.mul(l, 1));
      const scaled = scaler.scale(l);
      await scaled.backward();
      scaler.unscale_(opt);
      clipGradNorm_(api, params, 1.0);
      scaler.step(opt);
      scaler.update();
      opt.zeroGrad();
      scaled.dispose();
      loss = lossOut;
    }
    const lv = await loss.item();
    losses.push(lv);
    loss.dispose();
    input.dispose();
    target.dispose();
    // Scheduler at DRIVER level (never inside the captured body): the lr write
    // is a persistent-tensor write the tape rides as data.
    sched.step();
    stepMs.push(performance.now() - t0);
  }
  // G0-honest per-step wall: mean of the LATE (steady-state) half.
  const late = stepMs.slice(Math.floor(stepMs.length / 2));
  const meanLate = late.reduce((a, b) => a + b, 0) / late.length;
  log(
    `${useCapture ? "captured" : "control"} steady wall/step: ${meanLate.toFixed(1)} ms (late ${late.length} steps; all: ${stepMs.map((m) => m.toFixed(0)).join(",")})`,
  );

  const captureStats = stepAsync.stats();
  const replay = api.getStepTapeStats().replay;
  if (useCapture) {
    log(
      `replay stats: ${JSON.stringify({ promotions: replay.promotions, captures: replay.captures, replays: replay.replays, hits: replay.hits, missNoTape: replay.missNoTape, missScalar: replay.missScalar, missShape: replay.missShape, missValidity: replay.missValidity, missEpoch: replay.missEpoch, invalidations: replay.invalidations, readyTapes: replay.readyTapes })}`,
    );
    log(`recorder: ${JSON.stringify({ ...api.getStepTapeStats().recorder, refusalDiagnostics: undefined, boundaryReasons: undefined })}`);
  }
  return {
    losses,
    bodyRuns,
    captureStats,
    replayHits: replay.hits,
    refusals:
      replay.missScalar +
      replay.missShape +
      replay.missValidity +
      replay.missEpoch,
    vocab: V,
  };
}

/** Run one arm in a CHILD process and parse its result line. The two arms
 *  must NOT share a process: the plain (control) arm with a per-step LR
 *  scheduler grows GPU memory toward the 32GB ceiling, and whichever arm
 *  runs second then trains under memory pressure — the plain arm sporadically
 *  lands on a distinct WRONG trajectory there (~1-in-3, final 10.67→11.13,
 *  same-signature wrong finals; PRE-EXISTING plain-path bug, repro:
 *  tools/t-sched-plain-repro.ts — flag-off, no capture, and reproduces on
 *  main + the setLR delivery fix alone). Child processes give each arm a
 *  fresh device budget and make this gate deterministic. */
async function runArmInChild(arm: "captured" | "control"): Promise<RunResult> {
  const { execFileSync } = await import("node:child_process");
  const out = execFileSync(
    process.execPath,
    ["--import", "tsx", import.meta.filename],
    {
      env: { ...process.env, TRAIN_CAPTURE_ARM: arm },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  const line = out
    .split("\n")
    .find((l) => l.startsWith("=== ARM-RESULT === "));
  if (!line) throw new Error(`arm ${arm}: no result line in child output`);
  return JSON.parse(line.slice("=== ARM-RESULT === ".length)) as RunResult;
}

async function main() {
  if (!STEP_TAPE_REPLAY) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=1 (flag read at module load)");
    process.exit(1);
  }

  // Child mode: run ONE arm and emit its result as JSON.
  const armEnv = process.env.TRAIN_CAPTURE_ARM;
  if (armEnv === "captured" || armEnv === "control") {
    if (!(await initWebGPU())) {
      log("WebGPU init failed");
      process.exit(1);
    }
    const r = await run(armEnv === "captured");
    console.log(`=== ARM-RESULT === ${JSON.stringify(r)}`);
    destroyWebGPU();
    process.exit(0);
  }

  const ctl = await runArmInChild("control");
  log(`control losses: ${ctl.losses.map((l) => l.toFixed(4)).join(", ")}`);
  // ABSOLUTE sanity (device-2 lesson): the parity delta below is BLIND to
  // mutual corruption — a silent submit-drop makes both arms read ~0 and the
  // delta a false 0.0. Assert the REFERENCE arm's initial loss is near ln(V).
  assertInitialLossSane(ctl.losses[0], ctl.vocab, "t-train-capture-probe/control");
  const cap = await runArmInChild("captured");
  log(`captured: bodyRuns=${cap.bodyRuns} calls=${cap.captureStats.calls} hits=${cap.captureStats.hits} replayHits=${cap.replayHits} refusals=${cap.refusals}`);
  log(`captured losses: ${cap.losses.map((l) => l.toFixed(4)).join(", ")}`);

  // Parity: captured trajectory tracks control to the band.
  let maxDelta = 0;
  for (let i = 0; i < Math.min(cap.losses.length, ctl.losses.length); i++) {
    maxDelta = Math.max(maxDelta, Math.abs(cap.losses[i] - ctl.losses[i]));
  }

  // Run-exactly-once: body ran FEWER times than calls (hits skipped the body),
  // and hits > 0.
  const bodyFrozen = cap.bodyRuns < cap.captureStats.calls;

  console.log("=== TRAIN-CAPTURE-PROBE-STATS ===");
  console.log(
    JSON.stringify(
      {
        captureCalls: cap.captureStats.calls,
        captureHits: cap.captureStats.hits,
        replayHits: cap.replayHits,
        bodyRuns: cap.bodyRuns,
        refusals: cap.refusals,
        coldMisses: cap.captureStats.coldMisses,
        maxLossDelta: maxDelta,
        bodyFrozenOnHits: bodyFrozen,
      },
      null,
      2,
    ),
  );

  const pass =
    cap.captureStats.hits > 0 &&
    cap.refusals === 0 &&
    bodyFrozen &&
    // Threshold ABOVE the measured control-vs-control cross-run fp noise
    // floor (1.3e-3 @ 24 steps, 2.1e-3 @ 40 — separate processes, pool/timing
    // reorders). The frozen-LR failure mode this gate exists for measures
    // 0.48 — two orders of magnitude above.
    maxDelta < 2.5e-3;
  console.log(
    pass
      ? "PASS: whole training step captured; body frozen on hits; trajectory tracks control"
      : `FAIL (hits=${cap.captureStats.hits} refusals=${cap.refusals} bodyFrozen=${bodyFrozen} maxDelta=${maxDelta})`,
  );
  process.exit(pass ? 0 : 1); // parent never inits WebGPU (arms are children)
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
