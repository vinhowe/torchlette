/**
 * t-fp-spike-probe.ts — characterize the single-run ~1/50 floating-point spike
 * in PLAIN-path (non-compiled, non-tape) distilgpt2-class training (task #85).
 *
 * A "spike" here is a single-step loss that deviates from the run's LOCAL trend
 * far beyond the ordinary step-to-step noise. This probe runs many short plain
 * training runs in one process (each with a fresh engine + fresh model), records
 * per-step loss, and flags spikes programmatically so we can measure the real
 * rate and characterize magnitude / step-position / recovery / determinism.
 *
 * The plain path here is deliberately the SAME loop as tools/t-sched-plain-repro
 * (autocast + checkpoint + GradScaler + clipGradNorm + AdamW + CosineAnnealingLR),
 * i.e. the loop the P0 differential gates lean on.
 *
 * Env:
 *   RUNS     number of runs (default 60)
 *   STEPS    steps per run  (default 24)
 *   MODEL    distilgpt2     (default)
 *   SEQ_LEN  sequence len   (default 512)
 *   BACKEND  webgpu|cpu     (default webgpu)
 *   SEED_MODE  fixed|vary   (default fixed) — fixed: same data every run
 *                            (isolates timing/numeric from data); vary: per-run seed
 *   GRAD_NORM  1 to capture pre-clip grad L2 norm each step (adds a fence/step)
 *   SPIKE_Z    z-threshold vs local MAD trend (default 6)
 *   SPIKE_ABS  absolute min deviation floor in nats (default 0.15)
 *   OUT        path to write JSONL of every run (optional)
 *
 * NOTE: WebGPU/Dawn requires process.exit at the end.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { getGpuUncapturedErrorCount } from "../src/backend/webgpu/gpu-context";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const RUNS = parseInt(process.env.RUNS ?? "60", 10);
const STEPS = parseInt(process.env.STEPS ?? "24", 10);
const BATCH = 1;
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BACKEND = (process.env.BACKEND ?? "webgpu") as "webgpu" | "cpu";
const SEED_MODE = process.env.SEED_MODE ?? "fixed";
const CAPTURE_GRAD = process.env.GRAD_NORM === "1";
const SPIKE_Z = parseFloat(process.env.SPIKE_Z ?? "6");
const SPIKE_ABS = parseFloat(process.env.SPIKE_ABS ?? "0.15");
const OUT = process.env.OUT;

const log = (m: string) => console.error(`[fp-spike] ${m}`);

interface StepRec {
  loss: number;
  gradNorm?: number;
}
interface RunRec {
  run: number;
  seed: number;
  steps: StepRec[];
  gpuErrDelta: number;
}

async function run(runIdx: number, seedBase: number): Promise<RunRec> {
  // Flags default to the plain training config the P0 gates use; each is
  // env-overridable so the probe can bisect the spike class (fusion-off,
  // planner-off, checkpoint-off) and run the cpu backend (planner is
  // webgpu-only — set FUSION=0 PLANNER=0 for cpu).
  const flag = (name: string, def: boolean) =>
    process.env[name] == null ? def : process.env[name] === "1";
  const api = new Torchlette(BACKEND, {
    enableFusion: flag("FUSION", true),
    enableMemoryPlanning: flag("PLANNER", true),
    enableCheckpointSegmentation: flag("CKPT_SEG", true),
  });
  const model = await loadPretrainedGPT2(
    api,
    path.join(ROOT, "models", MODEL),
    { dropoutRate: 0 },
    { device: BACKEND },
  );
  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: 1e-4, weightDecay: 0.01, adamW: true }, api);
  const sched = new CosineAnnealingLR(
    opt,
    STEPS,
    parseFloat(process.env.ETA_MIN ?? "1e-5"),
  );
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;
  let seed = seedBase;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  const steps: StepRec[] = [];
  const errBefore = BACKEND === "webgpu" ? getGpuUncapturedErrorCount() : 0;

  for (let stp = 0; stp < STEPS; stp++) {
    await scaler.resolveDeferred();
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: BACKEND,
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: BACKEND,
    });
    const useCkpt = process.env.CKPT == null ? true : process.env.CKPT === "1";
    const l = api.tidy(() => {
      const ll = api.autocast(
        () =>
          model.forwardWithLoss(input, target, { useCheckpoint: useCkpt }).loss,
      );
      api.keep(ll);
      return ll;
    });
    const lossOut = api.noGrad(() => api.mul(l, 1));
    const scaled = scaler.scale(l);
    await scaled.backward();
    scaler.unscale_(opt);

    let gradNorm: number | undefined;
    if (CAPTURE_GRAD) {
      // Pre-clip grad L2 across all params (own fence via item()).
      const gn = api.noGrad(() => {
        let acc: import("../src/frontend/torchlette").Tensor | null = null;
        for (const p of params) {
          if (p.grad == null) continue;
          const s = api.sum(api.mul(p.grad, p.grad));
          acc = acc === null ? s : api.add(acc, s);
        }
        return acc === null ? api.tensorFromArray([0], [1], { device: BACKEND }) : api.sqrt(acc);
      });
      gradNorm = await gn.item();
      gn.dispose();
    }

    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    const lossVal = await lossOut.item();
    steps.push({ loss: lossVal, gradNorm });
    lossOut.dispose();
    input.dispose();
    target.dispose();
    sched.step();
  }
  const errAfter = BACKEND === "webgpu" ? getGpuUncapturedErrorCount() : 0;
  return { run: runIdx, seed: seedBase, steps, gpuErrDelta: errAfter - errBefore };
}

// --- spike detection: robust local-trend residual ---
function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

interface Spike {
  run: number;
  step: number;
  loss: number;
  expected: number; // local-trend estimate
  resid: number; // loss - expected
  z: number; // resid / local MAD
  recovered: boolean; // does the very next step return toward trend?
  gradNorm?: number;
}

// TRANSIENT-spike detector. A fp spike is a single step that jumps AWAY from
// BOTH its previous and next neighbours in the SAME direction and then recovers
// — a local peak/valley — as opposed to a monotone LR-schedule descent step
// (which moves the same direction as the trend and does NOT recover). Covers
// every interior step INCLUDING step 1 (the early-step transient class: the
// 12.5→21→12.6 shape). Magnitude is measured against the mean of the two
// immediate neighbours; z against the robust step-to-step noise scale.
function detectSpikes(rec: RunRec): Spike[] {
  const losses = rec.steps.map((s) => s.loss);
  const n = losses.length;
  if (n < 5) return [];
  const diffs: number[] = [];
  for (let i = 3; i < n; i++) diffs.push(Math.abs(losses[i] - losses[i - 1]));
  const madDiff = median(diffs.map((d) => Math.abs(d - median(diffs)))) || 1e-6;
  const noise = Math.max(1.4826 * madDiff, 1e-4);
  const spikes: Spike[] = [];
  for (let i = 1; i < n - 1; i++) {
    const prev = losses[i - 1];
    const next = losses[i + 1];
    const up = losses[i] - prev; // move into step i
    const down = next - losses[i]; // move out of step i
    // Transient = local extremum: into and out-of moves have opposite sign,
    // both non-trivial. (A trend step has same-sign into/out moves.)
    const isPeak = up > 0 && down < 0;
    const isValley = up < 0 && down > 0;
    if (!isPeak && !isValley) continue;
    const expected = (prev + next) / 2;
    const resid = losses[i] - expected;
    const z = Math.abs(resid) / noise;
    if (Math.abs(resid) >= SPIKE_ABS && z >= SPIKE_Z) {
      // Recovery = the step AFTER returns toward the pre-spike level.
      const nextResid = next - expected;
      const recovered = Math.abs(nextResid) < Math.abs(resid) * 0.5;
      spikes.push({
        run: rec.run,
        step: i,
        loss: losses[i],
        expected,
        resid,
        z,
        recovered,
        gradNorm: rec.steps[i].gradNorm,
      });
    }
  }
  return spikes;
}

async function main() {
  if (BACKEND === "webgpu") await initWebGPU();
  log(
    `config: backend=${BACKEND} runs=${RUNS} steps=${STEPS} seq=${SEQ} seedMode=${SEED_MODE} gradNorm=${CAPTURE_GRAD} spikeZ=${SPIKE_Z} spikeAbs=${SPIKE_ABS}`,
  );
  const allRuns: RunRec[] = [];
  const allSpikes: Spike[] = [];
  const outStream = OUT ? fs.createWriteStream(OUT, { flags: "w" }) : null;

  for (let r = 0; r < RUNS; r++) {
    const fixedBase = parseInt(process.env.SEED_BASE ?? "1234", 10);
    // SEED_BASE0 lets a batched sweep driver assign each child a disjoint seed
    // range (global run index = SEED_BASE0 + r), so vary-mode seeds never collide
    // across child processes.
    const seed0 = parseInt(process.env.SEED_BASE0 ?? "0", 10);
    const seedBase =
      SEED_MODE === "vary" ? 1234 + (seed0 + r) * 7919 : fixedBase;
    const rec = await run(r, seedBase);
    allRuns.push(rec);
    const spikes = detectSpikes(rec);
    for (const s of spikes) allSpikes.push(s);
    if (outStream) outStream.write(JSON.stringify(rec) + "\n");
    const flag = spikes.length ? ` SPIKE(${spikes.map((s) => `@${s.step} z=${s.z.toFixed(1)} d=${s.resid.toFixed(3)}${s.recovered ? " rec" : " DIV"}`).join("; ")})` : "";
    const gpuErr = rec.gpuErrDelta ? ` gpuErr+${rec.gpuErrDelta}` : "";
    log(`run ${r}: final=${rec.steps[rec.steps.length - 1].loss.toFixed(4)}${flag}${gpuErr}`);
  }
  if (outStream) outStream.end();

  // --- summary statistics ---
  const runsWithSpike = new Set(allSpikes.map((s) => s.run)).size;
  const stepPositions: Record<number, number> = {};
  const magnitudes = allSpikes.map((s) => Math.abs(s.resid));
  let recoveredCount = 0;
  for (const s of allSpikes) {
    stepPositions[s.step] = (stepPositions[s.step] ?? 0) + 1;
    if (s.recovered) recoveredCount++;
  }
  const totalSteps = RUNS * STEPS;
  const summary = {
    backend: BACKEND,
    runs: RUNS,
    steps: STEPS,
    seedMode: SEED_MODE,
    spikeCount: allSpikes.length,
    runsWithSpike,
    ratePerRun: runsWithSpike / RUNS,
    ratePerStep: allSpikes.length / totalSteps,
    oneInNStep: allSpikes.length ? Math.round(totalSteps / allSpikes.length) : null,
    magMin: magnitudes.length ? Math.min(...magnitudes) : null,
    magMax: magnitudes.length ? Math.max(...magnitudes) : null,
    magMedian: magnitudes.length ? median(magnitudes) : null,
    recoveredFrac: allSpikes.length ? recoveredCount / allSpikes.length : null,
    stepPositions,
    totalGpuErr: allRuns.reduce((a, r) => a + r.gpuErrDelta, 0),
    spikes: allSpikes,
  };
  console.log(JSON.stringify(summary, null, 2));
  if (BACKEND === "webgpu") destroyWebGPU();
  process.exit(0);
}
main();
