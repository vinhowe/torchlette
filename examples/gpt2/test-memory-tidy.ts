/**
 * Test memory management with tidy() around forward pass (like finetune-demo).
 * This should eliminate forward pass intermediate leaks.
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU, getGPUMemoryStats, getBufferPoolStats } from "../../src/backend/webgpu";
import { storageTracker } from "../../src/engine/lazy";
import { Adam } from "../../src/optim";
import { getTensorDebugStats, resetTensorDebugStats } from "../../src/runtime/tensor";

async function main() {
  console.log("Initializing WebGPU...");
  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const hiddenSize = 256;
  const inputSize = 128;
  const outputSize = 64;
  const batchSize = 32;

  const w1 = api.randn([inputSize, hiddenSize], { device: "webgpu", requiresGrad: true });
  const b1 = api.zeros([hiddenSize], { device: "webgpu", requiresGrad: true });
  const w2 = api.randn([hiddenSize, outputSize], { device: "webgpu", requiresGrad: true });
  const b2 = api.zeros([outputSize], { device: "webgpu", requiresGrad: true });

  api.keep(w1);
  api.keep(b1);
  api.keep(w2);
  api.keep(b2);

  const params = [w1, b1, w2, b2];
  const optimizer = new Adam(params, { lr: 0.001 }, api);

  console.log("\n=== Training Loop (with tidy around forward) ===\n");

  const NUM_STEPS = 20;
  const storageHistory: number[] = [];
  const memoryHistory: number[] = [];

  for (let step = 0; step < NUM_STEPS; step++) {
    resetTensorDebugStats();

    const x = api.randn([batchSize, inputSize], { device: "webgpu" });
    const target = api.randn([batchSize, outputSize], { device: "webgpu" });

    // Use tidy() around forward pass to auto-dispose intermediates
    const loss = api.tidy(() => {
      let h = x.matmul(w1).add(b1.expand([batchSize, hiddenSize]));
      h = h.gelu();
      const y = h.matmul(w2).add(b2.expand([batchSize, outputSize]));
      const diff = y.sub(target);
      return diff.mul(diff).mean();
    });

    const lossVal = await loss.item();

    await loss.backward();

    optimizer.step();
    optimizer.zeroGrad();

    loss.dispose();
    x.dispose();
    target.dispose();

    await api.markStep();

    const gpuStats = getGPUMemoryStats();
    const storageStats = storageTracker.stats();
    const tensorStats = getTensorDebugStats();

    storageHistory.push(storageStats.reachableStorages);
    memoryHistory.push(gpuStats.currentBytes);

    const debugCounters = storageTracker.debugCounters();
    console.log("Step " + step + ": loss=" + lossVal.toFixed(4) +
      ", memory=" + (gpuStats.currentBytes / 1e6).toFixed(1) + "MB" +
      ", storages=" + storageStats.reachableStorages +
      " (total=" + storageStats.totalStorages + ")" +
      ", created=" + tensorStats.created +
      ", disposed=" + tensorStats.disposed +
      " | storage: reg=" + debugCounters.registered +
      " reach=" + debugCounters.reachable +
      " unreach=" + debugCounters.unreachable +
      " destroy=" + debugCounters.destroyed);
  }

  const firstStorages = storageHistory[0];
  const lastStorages = storageHistory[storageHistory.length - 1];
  const storageGrowth = lastStorages - firstStorages;
  const avgStorageGrowthPerStep = storageGrowth / (NUM_STEPS - 1);

  const firstMemory = memoryHistory[0];
  const lastMemory = memoryHistory[memoryHistory.length - 1];
  const memoryGrowth = lastMemory - firstMemory;
  const avgMemoryGrowthPerStep = memoryGrowth / (NUM_STEPS - 1);

  console.log("\n=== Summary ===");
  console.log("Storage: " + firstStorages + " -> " + lastStorages + " (growth: " + storageGrowth + ", avg: " + avgStorageGrowthPerStep.toFixed(1) + "/step)");
  console.log("Memory: " + (firstMemory / 1e6).toFixed(1) + "MB -> " + (lastMemory / 1e6).toFixed(1) + "MB (growth: " + (memoryGrowth / 1e6).toFixed(1) + "MB, avg: " + (avgMemoryGrowthPerStep / 1e6).toFixed(2) + "MB/step)");

  if (avgStorageGrowthPerStep < 5 && avgMemoryGrowthPerStep < 10e6) {
    console.log("\nOK Memory management looks good - no significant growth per step");
  } else {
    console.log("\nLEAK Memory is growing - possible leak");
  }

  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
