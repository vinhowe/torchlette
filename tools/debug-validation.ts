/**
 * Debug: isolate which feature causes the validation error.
 * Test: no autocast (plain forward) to establish baseline
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import {
  destroyWebGPU,
  initWebGPU,
  setGPUMemoryLimit,
} from "../src/backend/webgpu";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

async function main() {
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);
  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(
    api,
    modelDir,
    { dropoutRate: 0.0 },
    { device: "webgpu" },
  );
  model.train();

  const optimizer = new Adam(
    model.parameters(),
    {
      lr: 5e-4,
      betas: [0.9, 0.999],
      eps: 1e-8,
      weightDecay: 0.01,
      adamW: true,
    },
    api,
  );

  const inputData = [50256, 1820, 4238, 338, 257, 5765, 1110];
  const targetData = [1820, 4238, 338, 257, 5765, 1110, 764];

  // NO autocast — plain forward
  const forward = (_input: Tensor, _target: Tensor) => {
    const result = model.forwardWithLoss(_input, _target);
    if (!result.loss) throw new Error("Loss is null");
    return result.loss;
  };

  for (let step = 0; step < 8; step++) {
    const input = api.tensorFromArray(inputData, [1, inputData.length], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetData, [1, targetData.length], {
      device: "webgpu",
    });

    const loss = forward(input, target);
    const lossValue = await loss.item();

    await loss.backward();

    await optimizer.stepAsync();
    optimizer.zeroGrad();

    loss.dispose();
    input.dispose();
    target.dispose();

    await api.markStep();
    console.log(`Step ${step}: loss=${lossValue.toFixed(4)}`);
  }

  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
