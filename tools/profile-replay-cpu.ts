/**
 * CPU profile of the dispatch replay path.
 * Runs DistilGPT-2 training with replay ON (no GPU timestamps).
 * Use with: node --cpu-prof --cpu-prof-dir=/tmp --cpu-prof-interval=100
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  destroyWebGPU,
  isF16Supported,
} from "../src/backend/webgpu";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam } from "../src/optim/adam";
import { GradScaler } from "../src/optim/grad-scaler";

const FIXED_TOKENS = [
  15496, 11, 314, 1101, 257, 3303, 2746, 2746, 290, 314, 1101, 994, 284,
  1037, 13, 314, 765, 284, 1037, 546, 262, 995, 290, 787, 340, 257, 1365,
  1295, 13, 314, 1101, 257,
];

const NUM_STEPS = 5;

async function main() {
  console.log("=== Replay CPU Profile ===\n");

  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  // Load model
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();

  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const useAMP = isF16Supported();
  const scaler = useAMP ? new GradScaler(api, { initScale: 1024.0 }) : null;
  console.log(`AMP: ${useAMP ? "enabled" : "disabled"}, Replay: ${process.env.TORCHLETTE_DISPATCH_REPLAY !== "0" ? "ON" : "OFF"}\n`);

  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target);
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  const inputData = FIXED_TOKENS.slice(0, -1);
  const targetData = FIXED_TOKENS.slice(1);

  for (let step = 0; step < NUM_STEPS; step++) {
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const t0 = performance.now();

    // Forward
    const loss = compiledForward(input, target);
    const lossValue = await loss.item();
    const t1 = performance.now();

    // Backward
    let scaledLoss: Tensor | null = null;
    if (scaler) {
      scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();
    } else {
      await loss.backward();
    }
    const t2 = performance.now();

    // Optimizer
    if (scaler) {
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
    } else {
      optimizer.step();
    }
    optimizer.zeroGrad();
    const t3 = performance.now();

    // Cleanup
    if (scaledLoss) scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    const t4 = performance.now();

    console.log(`Step ${step}: loss=${lossValue.toFixed(4)} | fwd=${(t1-t0).toFixed(0)}ms bwd=${(t2-t1).toFixed(0)}ms opt=${(t3-t2).toFixed(0)}ms cleanup=${(t4-t3).toFixed(0)}ms | total=${(t4-t0).toFixed(0)}ms`);
  }

  destroyWebGPU();
  process.exit(0);
}

main();
