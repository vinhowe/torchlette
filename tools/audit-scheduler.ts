/**
 * Scheduler audit: runs a few DistilGPT-2 training steps with
 * TORCHLETTE_SCHEDULER_AUDIT=1 baked in and prints aggregate savings.
 *
 * Usage:
 *   npx tsx tools/audit-scheduler.ts
 *
 * Reports, for each executed plan (and aggregated across steps):
 *   - trivial totalBytes (1 slot per tensor, baseline)
 *   - first-fit / best-fit totalBytes (what a scheduler would reserve)
 *   - peak live bytes (theoretical lower bound)
 *
 * If first-fit shaves >20% off trivial on steady-state plans, wiring the
 * scheduler into the buffer pool (L5) is worth the refactor.
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import {
  destroyWebGPU,
  getBufferPoolStats,
  getGPUMemoryStats,
  initWebGPU,
  isF16Supported,
} from "../src/backend/webgpu";
import { resetGPUMemoryPeak } from "../src/backend/webgpu/memory-tracker";
import {
  getMaxFirstFitBytes,
  getMaxPeakLiveBytes,
  printSchedulerAuditPerPlan,
  printSchedulerAuditSummary,
  resetSchedulerAudit,
} from "../src/compiler/scheduler/audit";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim";

// Force audit on for this tool.
process.env.TORCHLETTE_SCHEDULER_AUDIT =
  process.env.TORCHLETTE_SCHEDULER_AUDIT ?? "1";

const MODEL_NAME = process.env.TORCHLETTE_MODEL ?? "distilgpt2";
const SEQ_LEN = parseInt(process.env.TORCHLETTE_SEQ_LEN ?? "512", 10);
const NUM_STEPS = parseInt(process.env.NUM_STEPS ?? "3", 10);
const VERBOSE = !!process.env.VERBOSE;

async function main(): Promise<void> {
  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const modelDir = path.join(process.cwd(), "models", MODEL_NAME);
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 });
  model.train();

  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const useAMP = isF16Supported();
  const scaler = useAMP ? new GradScaler(api, { initScale: 1024.0 }) : null;

  const BASE = [
    2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30, 198, 1986, 280,
    1242, 517, 8855, 290, 517, 29815, 13, 198, 49, 619, 9985, 466, 13508, 262,
    38482, 31007, 286, 1737,
  ];
  const tokens: number[] = [];
  for (let i = 0; i < SEQ_LEN + 1; i++) tokens.push(BASE[i % BASE.length]);
  const inputTokens = tokens.slice(0, -1);
  const targetTokens = tokens.slice(1);

  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, {
        useCheckpoint: true,
      });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  for (let step = 0; step < NUM_STEPS; step++) {
    resetGPUMemoryPeak();
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    const input = api.tensorFromArray(inputTokens, [1, SEQ_LEN], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetTokens, [1, SEQ_LEN], {
      device: "webgpu",
    });

    const loss = compiledForward(input, target);
    const lossValue = await loss.item();

    if (scaler) {
      const scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
    } else {
      await loss.backward();
      optimizer.step();
    }
    optimizer.zeroGrad();

    loss.dispose();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();

    const gpu = getGPUMemoryStats();
    console.log(
      `step ${step}: loss=${lossValue.toFixed(4)} ` +
        `gpu=${(gpu.currentBytes / 1024 / 1024).toFixed(0)}MB ` +
        `peak=${(gpu.peakBytes / 1024 / 1024).toFixed(0)}MB`,
    );
  }

  if (VERBOSE) printSchedulerAuditPerPlan();
  printSchedulerAuditSummary();

  // GPU vs theoretical floor comparison.
  const mb = (b: number) => (b / 1024 / 1024).toFixed(0);
  const gpu = getGPUMemoryStats();
  const pool = getBufferPoolStats();
  const maxPeak = getMaxPeakLiveBytes();
  const maxFf = getMaxFirstFitBytes();
  console.error("");
  console.error(`model: ${MODEL_NAME}, seq_len=${SEQ_LEN}`);
  console.error(
    `GPU steady-state:    ${mb(gpu.currentBytes).padStart(6)}MB total allocated ` +
      `(${gpu.allocationCount} live buffers)`,
  );
  console.error(
    `GPU peak (ever):     ${mb(gpu.peakBytes).padStart(6)}MB ` +
      `(high-water across entire run)`,
  );
  console.error(
    `pool reuse rate:     ${(pool.reuseRate * 100).toFixed(0)}% ` +
      `(${pool.reuseCount} reuses, ${pool.allocCount} fresh allocs)`,
  );
  console.error(
    `max per-plan peak:   ${mb(maxPeak).padStart(6)}MB ` +
      `(theoretical floor inside the worst plan)`,
  );
  console.error(
    `max per-plan 1st-fit:${mb(maxFf).padStart(6)}MB ` +
      `(what static first-fit would reserve for the worst plan)`,
  );
  console.error(
    `steady-state - peak: ${mb(gpu.currentBytes - maxPeak).padStart(6)}MB slack ` +
      `(persistent weights/state + pool cache + saved-for-backward)`,
  );

  resetSchedulerAudit();

  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
