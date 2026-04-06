/**
 * Phase-by-phase GPU memory breakdown for a distilgpt2 training step.
 *
 * Resets the peak high-water mark at each phase boundary and prints the
 * peak-during-phase + end-of-phase currentBytes. Identifies which phase
 * owns the ~3.8GB transient peak-to-steady gap observed in audit-scheduler.
 *
 * Usage: npx tsx tools/memory-phase-breakdown.ts
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
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim";

const MODEL_NAME = process.env.TORCHLETTE_MODEL ?? "distilgpt2";
const SEQ_LEN = parseInt(process.env.TORCHLETTE_SEQ_LEN ?? "512", 10);
const NUM_STEPS = parseInt(process.env.NUM_STEPS ?? "3", 10);

function mark(label: string): void {
  const s = getGPUMemoryStats();
  const pool = getBufferPoolStats();
  const cur = (s.currentBytes / 1024 / 1024).toFixed(0).padStart(5);
  const peak = (s.peakBytes / 1024 / 1024).toFixed(0).padStart(5);
  // Physical = tracked + pending + pooled (all consume actual GPU memory)
  const physical = s.currentBytes + pool.pooledBytes;
  const phys = (physical / 1024 / 1024).toFixed(0).padStart(5);
  console.error(
    `  ${label.padEnd(28)} cur=${cur}MB  peak=${peak}MB  phys=${phys}MB`,
  );
  resetGPUMemoryPeak();
}

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
        useCheckpoint: !!process.env.CHECKPOINT,
      });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  console.error(`model=${MODEL_NAME} seq_len=${SEQ_LEN} amp=${useAMP}`);
  console.error("");

  for (let step = 0; step < NUM_STEPS; step++) {
    console.error(`=== step ${step} ===`);
    resetGPUMemoryPeak();
    mark("before step");

    if (scaler) await scaler.resolveDeferred();
    mark("scaler.resolveDeferred");

    await api.beginStep();
    mark("beginStep");

    const input = api.tensorFromArray(inputTokens, [1, SEQ_LEN], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetTokens, [1, SEQ_LEN], {
      device: "webgpu",
    });
    mark("tensor uploads");

    const loss = compiledForward(input, target);
    mark("forward (lazy)");

    const lossValue = await loss.item();
    mark("loss.item() — forward exec");

    if (scaler) {
      const scaledLoss = scaler.scale(loss);
      mark("scaler.scale");

      await scaledLoss.backward();
      mark("backward");

      scaler.unscale_(optimizer);
      mark("scaler.unscale_");

      scaler.step(optimizer);
      mark("scaler.step (adam)");

      scaler.update();
      mark("scaler.update");

      scaledLoss.dispose();
    } else {
      await loss.backward();
      mark("backward");

      optimizer.step();
      mark("optimizer.step (adam)");
    }

    optimizer.zeroGrad();
    mark("zeroGrad");

    loss.dispose();
    input.dispose();
    target.dispose();
    api.endStep();
    mark("dispose + endStep");

    await api.markStep();
    mark("markStep");

    console.error(`  loss=${lossValue.toFixed(4)}`);
    console.error("");
  }

  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
