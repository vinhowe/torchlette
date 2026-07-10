/**
 * inc-3 gate 5+6 — THE MEASUREMENT: runahead wall-clock + memory.
 *
 * Arms (each in its own child process — fresh device budget, no cross-arm
 * pool/tape contamination):
 *   uncaptured — the plain fullstack loop (resolveDeferred + item + implicit
 *                markStep via scaler each step).
 *   serial     — captured (training:true), driver owns markStep per step
 *                (inc-2b shape: the ~3-4% build-skip, no runahead).
 *   ringK1     — runahead ring, K=1 (correctness shape; no overlap: the
 *                (K+1)-th call fences the previous step first).
 *   ringK2     — runahead ring, K=2 (G0(b): one step in flight — the ~30%
 *                wall claim vs the GPU-bound floor).
 *
 * Reports per arm: steady wall/step (late-half mean), loss trajectory, GPU
 * memory peak/current, hits/bodyRuns. PASS = ringK2 trajectory tracks the
 * uncaptured control to the cross-process noise band AND the arms complete
 * clean; the wall numbers are REPORTED (the perf claim is read from them, not
 * auto-thresholded — the box is shared and contention skews absolutes).
 *
 * Run (device 2):
 *   VULKAN_DEVICE_INDEX=2 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 npx tsx tools/t-ring-measure.ts
 * Env: MODEL(=distilgpt2) SEQ_LEN(=512) BATCH(=1) STEPS(=30)
 */
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { getGPUMemoryStats } from "../src/backend/webgpu/memory-tracker";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import type { Tensor } from "../src/frontend/tensor";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "30", 10);
// NOSCHED=1: drop the per-step driver-level LR scheduler. KNOWN ISSUE: under
// RUNAHEAD a per-step scheduler write has a warmup-era transient [lifetime]
// warn (the [2b-sched]/dangling-copy_ family; trajectory impact below noise) —
// the STRICT gate runs NOSCHED=1 until that follow-on lands.
const NOSCHED = process.env.NOSCHED === "1";
const log = (m: string) => console.error(`[t-ring-measure] ${m}`);

type Arm = "uncaptured" | "serial" | "ringK1" | "ringK2";

interface ArmResult {
  arm: Arm;
  losses: number[];
  wallLateMs: number;
  wallAllMs: number[];
  peakMB: number;
  currentMB: number;
  hits: number;
  bodyRuns: number;
}

async function runArm(arm: Arm): Promise<ArmResult> {
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
  const sched = NOSCHED ? null : new CosineAnnealingLR(opt, STEPS, 1e-5);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  let s = 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };
  const nextBatch = () => {
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    return {
      input: api.tensorFromArray(Array.from(inp), [BATCH, SEQ], { device: "webgpu" }),
      target: api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], { device: "webgpu" }),
    };
  };

  let bodyRuns = 0;
  const stepBody = async (input: Tensor, target: Tensor): Promise<Tensor> => {
    bodyRuns++;
    const loss = api.tidy(() => {
      const l = api.autocast(
        () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(l);
      return l;
    });
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

  const losses = new Array<number>(STEPS);
  const stepMs: number[] = [];

  if (arm === "uncaptured" || arm === "serial") {
    const step = arm === "serial" ? api.capture(stepBody, { training: true }) : null;
    for (let stp = 0; stp < STEPS; stp++) {
      const t0 = performance.now();
      await scaler.resolveDeferred();
      const { input, target } = nextBatch();
      const loss = step
        ? ((await step(input, target)) as Tensor)
        : await stepBody(input, target);
      losses[stp] = await loss.item();
      loss.dispose();
      input.dispose();
      target.dispose();
      sched?.step();
      stepMs.push(performance.now() - t0);
    }
    const mem = getGPUMemoryStats();
    const late = stepMs.slice(Math.floor(stepMs.length / 2));
    return {
      arm,
      losses,
      wallLateMs: late.reduce((a, b) => a + b, 0) / late.length,
      wallAllMs: stepMs,
      peakMB: mem.peakBytes / 1e6,
      currentMB: mem.currentBytes / 1e6,
      hits: step?.stats().hits ?? 0,
      bodyRuns,
    };
  }

  // Runahead arms: the ring owns the boundary; driver reads K-behind via the
  // staged pool-excluded copies; scaler bookkeeping rides the same cadence.
  const K = arm === "ringK1" ? 1 : 2;
  const step = api.capture(stepBody, { training: true, runahead: true, ringDepth: K });
  const handles: Tensor[] = [];
  for (let stp = 0; stp < STEPS; stp++) {
    const t0 = performance.now();
    const { input, target } = nextBatch();
    handles.push((await step(input, target)) as Tensor);
    scaler.snapshotDeferred();
    const oldest = stp - K + 1;
    if (oldest >= 0) {
      losses[oldest] = (await api.cpu(handles[oldest]))[0];
      handles[oldest].dispose();
    }
    if (stp >= K) await scaler.resolveOldestDeferred();
    input.dispose();
    target.dispose();
    sched?.step();
    stepMs.push(performance.now() - t0);
  }
  await step.drain();
  while (await scaler.resolveOldestDeferred()) {
    /* drain reports in order */
  }
  for (let i = Math.max(0, STEPS - K + 1); i < STEPS; i++) {
    if (losses[i] === undefined) losses[i] = (await api.cpu(handles[i]))[0];
  }
  const mem = getGPUMemoryStats();
  const late = stepMs.slice(Math.floor(stepMs.length / 2));
  const st = api.getStepTapeStats();
  log(
    `${arm} replay: ${JSON.stringify({ promotions: st.replay.promotions, replays: st.replay.replays, hits: st.replay.hits, missNoTape: st.replay.missNoTape, missScalar: st.replay.missScalar, missShape: st.replay.missShape, missValidity: st.replay.missValidity, invalidations: st.replay.invalidations })}`,
  );
  log(
    `${arm} recorder: ${JSON.stringify({ ...st.recorder, refusalDiagnostics: (st.recorder as { refusalDiagnostics?: unknown[] }).refusalDiagnostics?.slice?.(0, 3), boundaryReasons: undefined })}`,
  );
  return {
    arm,
    losses,
    wallLateMs: late.reduce((a, b) => a + b, 0) / late.length,
    wallAllMs: stepMs,
    peakMB: mem.peakBytes / 1e6,
    currentMB: mem.currentBytes / 1e6,
    hits: step.stats().hits,
    bodyRuns,
  };
}

async function runArmInChild(arm: Arm): Promise<ArmResult> {
  const { execFileSync } = await import("node:child_process");
  const out = execFileSync(process.execPath, ["--import", "tsx", import.meta.filename], {
    env: { ...process.env, RING_MEASURE_ARM: arm },
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
    maxBuffer: 64 * 1024 * 1024,
  });
  const line = out.split("\n").find((l) => l.startsWith("=== ARM-RESULT === "));
  if (!line) throw new Error(`arm ${arm}: no result line`);
  return JSON.parse(line.slice("=== ARM-RESULT === ".length)) as ArmResult;
}

async function main() {
  if (!STEP_TAPE_REPLAY) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=1");
    process.exit(1);
  }
  const armEnv = process.env.RING_MEASURE_ARM as Arm | undefined;
  if (armEnv) {
    if (!(await initWebGPU())) {
      log("WebGPU init failed");
      process.exit(1);
    }
    const r = await runArm(armEnv);
    console.log(`=== ARM-RESULT === ${JSON.stringify(r)}`);
    destroyWebGPU();
    process.exit(0);
  }

  const arms: Arm[] = ["uncaptured", "serial", "ringK1", "ringK2"];
  const results: ArmResult[] = [];
  for (const a of arms) {
    const r = await runArmInChild(a);
    results.push(r);
    log(
      `${a.padEnd(10)} wall/step(late)=${r.wallLateMs.toFixed(1)}ms peak=${r.peakMB.toFixed(0)}MB cur=${r.currentMB.toFixed(0)}MB hits=${r.hits} bodyRuns=${r.bodyRuns}`,
    );
    log(`${a.padEnd(10)} all ms: ${r.wallAllMs.map((m) => m.toFixed(0)).join(",")}`);
  }
  const ctl = results[0];
  for (const r of results) {
    let maxD = 0;
    for (let i = 0; i < STEPS; i++)
      maxD = Math.max(maxD, Math.abs(r.losses[i] - ctl.losses[i]));
    log(`${r.arm.padEnd(10)} maxΔ-vs-uncaptured=${maxD.toExponential(2)} final=${r.losses[STEPS - 1].toFixed(4)}`);
  }
  console.log("=== RING-MEASURE ===");
  console.log(
    JSON.stringify(
      Object.fromEntries(
        results.map((r) => [
          r.arm,
          { wallLateMs: +r.wallLateMs.toFixed(1), peakMB: +r.peakMB.toFixed(0), hits: r.hits },
        ]),
      ),
      null,
      2,
    ),
  );
  // Trajectory correctness: every arm tracks the uncaptured control to the
  // cross-process fp noise band (measured ~1.3e-3 @24 for this stack).
  const pass = results.every((r) => {
    let maxD = 0;
    for (let i = 0; i < STEPS; i++)
      maxD = Math.max(maxD, Math.abs(r.losses[i] - ctl.losses[i]));
    return maxD < 2.5e-3;
  });
  console.log(pass ? "PASS (trajectories; read walls above)" : "FAIL (trajectory divergence)");
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
