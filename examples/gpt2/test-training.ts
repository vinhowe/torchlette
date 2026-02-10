/**
 * Test training with markStep() cleanup to verify memory management.
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU, getGPUMemoryStats, getBufferPoolStats } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { Adam } from "../../src/optim";
import { storageTracker } from "../../src/engine/lazy";

const TINY_CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 64,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0,
};

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

  console.log("\n=== Training Loop with markStep() ===");
  const losses: number[] = [];

  // Compile forward pass - per spec §1.6, compile() acts as implicit tidy
  const forward = api.compile((input: Tensor, target: Tensor) => {
    const result = model.forwardWithLoss(input, target);
    return result.loss!;
  });

  for (let step = 0; step < 10; step++) {
    const input = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [batchSize, seqLen], { device: "webgpu" });

    // Forward pass with compile() - intermediates auto-disposed, saved tensors kept
    const loss = forward(input, target);

    // Get loss value - this forces materialization
    const lossVal = await loss.item();
    losses.push(lossVal);

    // Backward pass
    await loss.backward();

    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();

    // Dispose tensors we're done with
    loss.dispose();
    input.dispose();
    target.dispose();

    // Mark step to trigger cleanup (§6)
    await api.markStep();

    // Memory stats
    const gpuStats = getGPUMemoryStats();
    const storageStats = storageTracker.stats();
    const poolStats = getBufferPoolStats();
    console.log(`Step ${step}: loss=${lossVal.toFixed(4)}, gpu=${(gpuStats.currentBytes / 1024 / 1024).toFixed(1)}MB, storages=${storageStats.totalStorages}(${storageStats.reachableStorages}r/${storageStats.unreachableStorages}u), pool: ${poolStats.pooledBuffers} buffers, reuse=${(poolStats.reuseRate * 100).toFixed(0)}%`);
  }

  console.log("\nLoss progression:", losses.map(l => l.toFixed(2)).join(" -> "));

  // Final buffer pool stats
  const finalPoolStats = getBufferPoolStats();
  console.log(`\nBuffer Pool Stats:`);
  console.log(`  - Pooled buffers: ${finalPoolStats.pooledBuffers}`);
  console.log(`  - Pooled bytes: ${(finalPoolStats.pooledBytes / 1024 / 1024).toFixed(2)}MB`);
  console.log(`  - Reuse count: ${finalPoolStats.reuseCount}`);
  console.log(`  - New alloc count: ${finalPoolStats.allocCount}`);
  console.log(`  - Reuse rate: ${(finalPoolStats.reuseRate * 100).toFixed(1)}%`);

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
