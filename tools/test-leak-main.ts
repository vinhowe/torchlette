/**
 * Minimal leak diagnostic for main branch
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import {
  destroyWebGPU,
  getGPUMemoryStats,
  initWebGPU,
  setGPUMemoryLimit,
} from "../src/backend/webgpu";
import { clearBufferPool } from "../src/backend/webgpu/buffer-pool";
import { storageTracker } from "../src/engine/lazy";
import { type Tensor, Torchlette } from "../src/frontend";
import { Adam, GradScaler } from "../src/optim";

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
  const scaler = new GradScaler(api);

  const inputData = [50256, 1820, 4238, 338, 257, 5765, 1110];
  const targetData = [1820, 4238, 338, 257, 5765, 1110, 764];

  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, {
        useCheckpoint: true,
      });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  console.log("Model loaded, starting training...\n");

  for (let step = 0; step < 5; step++) {
    const input = api.tensorFromArray(inputData, [1, inputData.length], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetData, [1, targetData.length], {
      device: "webgpu",
    });

    const loss = compiledForward(input, target);
    const lossValue = await loss.item();

    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    const statsAfterBwd = storageTracker.stats();

    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();

    const statsAfterOpt = storageTracker.stats();

    optimizer.zeroGrad();
    scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();

    const statsAfterDispose = storageTracker.stats();

    await api.markStep();

    const statsAfterMS = storageTracker.stats();
    const memAfter = getGPUMemoryStats();

    console.log(`Step ${step}: loss=${lossValue.toFixed(4)}`);
    console.log(
      `  After backward:  total=${statsAfterBwd.totalStorages} reachable=${statsAfterBwd.reachableStorages}`,
    );
    console.log(
      `  After optimizer: total=${statsAfterOpt.totalStorages} reachable=${statsAfterOpt.reachableStorages}`,
    );
    console.log(
      `  After dispose:   total=${statsAfterDispose.totalStorages} reachable=${statsAfterDispose.reachableStorages}`,
    );
    console.log(
      `  After markStep:  total=${statsAfterMS.totalStorages} reachable=${statsAfterMS.reachableStorages} allocs=${memAfter.allocationCount}`,
    );
    console.log();
  }

  clearBufferPool();
  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
