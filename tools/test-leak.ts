/**
 * Minimal leak diagnostic: runs 5 training steps and reports alloc delta per step.
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import {
  destroyWebGPU,
  initWebGPU,
  setGPUMemoryLimit,
} from "../src/backend/webgpu";
import { clearBufferPool } from "../src/backend/webgpu/buffer-pool";
import { storageTracker } from "../src/graph/storage-tracker";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
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

  const useCheckpoint = true;

  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, {
        useCheckpoint,
      });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  console.log(
    `Model loaded, starting training (checkpoint=${useCheckpoint})...\n`,
  );

  let prevReachableIds: Set<number> | null = null;
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

    {
      const afterBwd = storageTracker.stats();
      console.log(
        `  afterBwd: total=${afterBwd.totalStorages} reachable=${afterBwd.reachableStorages} unreachable=${afterBwd.unreachableStorages}`,
      );
    }

    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();

    optimizer.zeroGrad();
    scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();

    await api.markStep();

    const stats = storageTracker.stats();
    // Analyze what's holding storages reachable
    if (step >= 1) {
      const newReachable = storageTracker.getNewReachableSince(
        prevReachableIds!,
      );
      const byShape = new Map<string, number>();
      let disposedCount = 0;
      for (const entry of newReachable) {
        if (entry.debugInfo?.disposed) disposedCount++;
        const key =
          entry.debugInfo?.type === "resultsArray"
            ? "resultsArray"
            : `${entry.debugInfo?.shape?.join("x") ?? "?"}:${entry.debugInfo?.dtype ?? "?"}`;
        byShape.set(key, (byShape.get(key) ?? 0) + 1);
      }
      // Sort by count descending
      const sorted = [...byShape.entries()].sort((a, b) => b[1] - a[1]);
      console.log(
        `Step ${step}: loss=${lossValue.toFixed(4)} total=${stats.totalStorages} reachable=${stats.reachableStorages} new=${newReachable.length} disposed=${disposedCount}`,
      );
      for (const [shape, count] of sorted.slice(0, 15)) {
        console.log(`  ${shape}: ${count}`);
      }
    } else {
      console.log(
        `Step ${step}: loss=${lossValue.toFixed(4)} total=${stats.totalStorages} reachable=${stats.reachableStorages}`,
      );
    }
    prevReachableIds = storageTracker.getReachableIds();
  }

  clearBufferPool();
  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
