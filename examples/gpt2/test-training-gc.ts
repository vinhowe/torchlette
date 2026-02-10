/**
 * Test training with forced GC to verify buffer pool reuse and memory donation.
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU, getGPUMemoryStats, getBufferPoolStats } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { Adam } from "../../src/optim";
import { storageTracker } from "../../src/engine/lazy";
import { getMemoryPlannerStats } from "../../src/engine/memory-planned-executor";

const TINY_CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 64,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0,
};

// Check if GC is available
const gc = (globalThis as any).gc;
if (!gc) {
  console.error("Run with: node --expose-gc --import tsx examples/gpt2/test-training-gc.ts");
  process.exit(1);
}

async function main() {
  console.log("Initializing WebGPU...");
  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });

  console.log("\nCreating tiny GPT-2 model...");
  const model = new GPT2(api, TINY_CONFIG, { device: "webgpu" });
  model.train();

  const params = model.parameters();
  console.log(`Model has ${params.length} parameters`);

  console.log("\nCreating optimizer...");
  const optimizer = new Adam(params, { lr: 0.001 }, api);

  // Fixed batch for training
  const batchSize = 2;
  const seqLen = 16;
  const inputData = Array.from({ length: batchSize * seqLen }, () =>
    Math.floor(Math.random() * TINY_CONFIG.vocabSize)
  );
  const targetData = Array.from({ length: batchSize * seqLen }, () =>
    Math.floor(Math.random() * TINY_CONFIG.vocabSize)
  );

  console.log("\n=== Training Loop with markStep() + forced GC ===");
  const losses: number[] = [];

  const forward = api.compile((input: Tensor, target: Tensor) => {
    const result = model.forwardWithLoss(input, target);
    return result.loss!;
  });

  for (let step = 0; step < 10; step++) {
    const input = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [batchSize, seqLen], { device: "webgpu" });

    const loss = forward(input, target);
    const lossVal = await loss.item();
    losses.push(lossVal);

    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();

    loss.dispose();
    input.dispose();
    target.dispose();

    await api.markStep();

    // Force GC to trigger FinalizationRegistry cleanup
    gc();

    // Small delay to let async cleanup complete
    await new Promise(r => setTimeout(r, 10));

    const gpuStats = getGPUMemoryStats();
    const storageStats = storageTracker.stats();
    const poolStats = getBufferPoolStats();
    console.log(`Step ${step}: loss=${lossVal.toFixed(4)}, gpu=${(gpuStats.currentBytes / 1024 / 1024).toFixed(1)}MB, storages=${storageStats.totalStorages}(${storageStats.reachableStorages}r/${storageStats.unreachableStorages}u), pool: ${poolStats.pooledBuffers} buffers, reuse=${(poolStats.reuseRate * 100).toFixed(0)}%`);
  }

  console.log("\nLoss progression:", losses.map(l => l.toFixed(2)).join(" -> "));

  const finalPoolStats = getBufferPoolStats();
  console.log(`\nBuffer Pool Stats:`);
  console.log(`  - Pooled buffers: ${finalPoolStats.pooledBuffers}`);
  console.log(`  - Pending buffers: ${finalPoolStats.pendingBuffers}`);
  console.log(`  - Pooled bytes: ${(finalPoolStats.pooledBytes / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  - Reuse count: ${finalPoolStats.reuseCount}`);
  console.log(`  - New alloc count: ${finalPoolStats.allocCount}`);
  console.log(`  - Reuse rate: ${(finalPoolStats.reuseRate * 100).toFixed(1)}%`);

  const plannerStats = getMemoryPlannerStats();
  console.log(`\nMemory Planner Stats:`);
  console.log(`  - Buffer pool: ${plannerStats.bufferPool.pooledBuffers} pooled, ${plannerStats.bufferPool.inUseBuffers} in-use`);
  console.log(`  - Plan manager: ${plannerStats.planManager.totalPlans} total, ${plannerStats.planManager.activePlans} active`);

  console.log(`\nNote: Memory donation is enabled in executeWithMemoryPlanning.`);
  console.log(`Donation allows reusing input buffers as output buffers when inputs die.`);

  if (losses[losses.length - 1] < losses[0]) {
    console.log("\n✓ Loss decreased - training is working!");
  } else {
    console.log("\n✗ Loss did not decrease");
  }

  process.exit(0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});
