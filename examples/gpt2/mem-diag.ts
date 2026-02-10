/**
 * Quick memory diagnostic - runs a few training steps to observe allocation patterns.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU, getGPUMemoryStats, getGPUAllocationHistogram } from "../../src/backend/webgpu";
import { storageTracker } from "../../src/engine/lazy";
import { loadPretrainedGPT2 } from "./loader";
import { Adam } from "../../src/optim";

function printHistogram(label: string) {
  const hist = getGPUAllocationHistogram();
  console.log(`  ${label} histogram:`);
  for (const [bucket, { count, totalBytes }] of hist) {
    console.log(`    ${bucket}: ${count} buffers, ${(totalBytes / 1e6).toFixed(2)}MB`);
  }
}

async function main() {
  console.log("Memory Diagnostic - After Fix");
  await initWebGPU();

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);

  const seqLen = 32;
  const inputData = Array.from({ length: seqLen - 1 }, (_, i) => i % 100 + 1);
  const targetData = Array.from({ length: seqLen - 1 }, (_, i) => (i + 1) % 100 + 1);

  console.log("\nRunning 5 training steps...\n");

  for (let step = 0; step < 5; step++) {
    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const loss = api.tidy(() => {
      const result = model.forwardWithLoss(input, target);
      if (!result.loss) throw new Error("Loss is null");
      api.keep(input);
      api.keep(target);
      return result.loss;
    });

    const lossValue = await loss.item();
    await loss.backward();
    await optimizer.stepAsync();
    optimizer.zeroGrad();

    loss.dispose();
    input.dispose();
    target.dispose();

    await api.markStep();

    const memStats = getGPUMemoryStats();
    const storageStats = storageTracker.stats();
    console.log(`Step ${step}: loss=${lossValue.toFixed(4)}, mem=${(memStats.currentBytes / 1e9).toFixed(2)}GB, allocs=${memStats.allocationCount}, storages=${storageStats.totalStorages}(${storageStats.reachableStorages}R)`);
    if (step <= 2) {
      printHistogram(`Step ${step}`);
    }
  }
}

main().catch(console.error);
