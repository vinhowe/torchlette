#!/usr/bin/env npx tsx
/**
 * Combined test for AMP + Checkpointing + Memory Planning.
 * This tests all features working together in a LoRA-like training scenario.
 */

import { Torchlette, type FrontendTensor as Tensor, GradScaler } from '../../src';
import { checkpoint } from '../../src/nn/checkpoint';
import { initWebGPU } from '../../src/backend/webgpu';

async function testCombinedFeatures(): Promise<void> {
  console.log('=== Test: AMP + Checkpointing + Memory Planning ===\n');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true, // Enable memory planning
  });

  const gradScaler = new GradScaler(api, {
    initScale: 65536.0,
    growthFactor: 2.0,
    backoffFactor: 0.5,
    growthInterval: 2000,
  });

  // GPT-2 like dimensions (scaled down)
  const batchSize = 1;
  const seqLen = 16;
  const embedDim = 64;
  const hiddenDim = 256;

  console.log(`Dimensions: batch=${batchSize}, seq=${seqLen}, embed=${embedDim}, hidden=${hiddenDim}`);

  // Create "model" weights
  const w1 = api.tensorFromArray(
    Array.from({ length: embedDim * hiddenDim }, (_, i) => Math.cos(i * 0.001) * 0.02),
    [embedDim, hiddenDim],
    { device: 'webgpu', requiresGrad: true }
  );

  const w2 = api.tensorFromArray(
    Array.from({ length: hiddenDim * embedDim }, (_, i) => Math.sin(i * 0.001) * 0.02),
    [hiddenDim, embedDim],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('w1 shape:', w1.shape);
  console.log('w2 shape:', w2.shape);
  console.log('Initial scale:', gradScaler.getScale());
  console.log('');

  // Training loop with 3 steps
  for (let step = 0; step < 3; step++) {
    console.log(`--- Step ${step + 1} ---`);

    // Fresh input each step
    const x = api.tensorFromArray(
      Array.from({ length: batchSize * seqLen * embedDim }, (_, i) =>
        Math.sin(i * 0.01 + step) * 0.5
      ),
      [batchSize, seqLen, embedDim],
      { device: 'webgpu' }
    );

    // Block functions
    function block1(input: Tensor): Tensor {
      return api.relu(api.matmul(input, w1));
    }

    function block2(input: Tensor): Tensor {
      return api.relu(api.matmul(input, w2));
    }

    // Forward with AMP + Checkpointing
    let loss = api.autocast(() => {
      let h = x;

      // Checkpointed blocks
      h = checkpoint(api, block1, [h]);
      h = checkpoint(api, block2, [h]);

      return h.sum();
    });

    // Scale the loss
    loss = gradScaler.scale(loss);
    const scaledLossVal = await loss.item();
    console.log(`  Scaled loss: ${scaledLossVal.toExponential(4)}`);

    // Backward
    await loss.backward();

    // Check gradients
    const w1GradExists = w1.grad !== null;
    const w2GradExists = w2.grad !== null;
    console.log(`  Gradients exist: w1=${w1GradExists}, w2=${w2GradExists}`);

    if (w1.grad && w2.grad) {
      const w1GradSum = await w1.grad.sum().item();
      const w2GradSum = await w2.grad.sum().item();
      console.log(`  w1.grad sum: ${w1GradSum.toExponential(4)}`);
      console.log(`  w2.grad sum: ${w2GradSum.toExponential(4)}`);
    }

    // Zero gradients for next step
    w1.zeroGrad();
    w2.zeroGrad();

    // Mark step (trigger memory cleanup)
    await api.markStep();
  }

  console.log('\n=== PASSED ===');
}

async function testCombinedWithMoreLayers(): Promise<void> {
  console.log('\n=== Test: AMP + Checkpointing with 4 layers ===\n');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const gradScaler = new GradScaler(api, {
    initScale: 65536.0,
  });

  const batchSize = 1;
  const seqLen = 8;
  const dim = 32;

  // 4 identical weight matrices (simulate 4 transformer blocks)
  const weights = Array.from({ length: 4 }, (_, layer) =>
    api.tensorFromArray(
      Array.from({ length: dim * dim }, (_, i) => Math.cos(i * 0.01 + layer) * 0.1),
      [dim, dim],
      { device: 'webgpu', requiresGrad: true }
    )
  );

  console.log(`4 weight matrices, each [${dim}, ${dim}]`);

  // Fresh input
  const x = api.tensorFromArray(
    Array.from({ length: batchSize * seqLen * dim }, (_, i) => Math.sin(i * 0.01)),
    [batchSize, seqLen, dim],
    { device: 'webgpu' }
  );

  // Forward with AMP + Checkpointing for each layer
  let loss = api.autocast(() => {
    let h = x;

    for (let i = 0; i < weights.length; i++) {
      const w = weights[i];
      h = checkpoint(api, (input) => api.relu(api.matmul(input, w)), [h]);
    }

    return h.sum();
  });

  // Scale and backward
  loss = gradScaler.scale(loss);
  const scaledLossVal = await loss.item();
  console.log(`  Scaled loss: ${scaledLossVal.toExponential(4)}`);

  await loss.backward();

  // Check all gradients
  for (let i = 0; i < weights.length; i++) {
    const w = weights[i];
    const gradExists = w.grad !== null;
    console.log(`  w${i}.grad exists: ${gradExists}`);
    if (w.grad) {
      const gradSum = await w.grad.sum().item();
      console.log(`  w${i}.grad sum: ${gradSum.toExponential(4)}`);
    }
  }

  await api.markStep();
  console.log('\n=== PASSED ===');
}

async function testLoRAPattern(): Promise<void> {
  console.log('\n=== Test: LoRA-style training pattern ===\n');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const gradScaler = new GradScaler(api, {
    initScale: 65536.0,
  });

  // LoRA dimensions
  const inDim = 64;
  const outDim = 64;
  const rank = 4;
  const alpha = 8;
  const scale = alpha / rank;

  // Base weight (frozen)
  const baseWeight = api.tensorFromArray(
    Array.from({ length: outDim * inDim }, (_, i) => Math.cos(i * 0.01) * 0.1),
    [outDim, inDim],
    { device: 'webgpu', requiresGrad: false } // Frozen
  );

  // LoRA weights (trainable)
  const loraA = api.tensorFromArray(
    Array.from({ length: rank * inDim }, (_, i) => Math.random() * 0.01),
    [rank, inDim],
    { device: 'webgpu', requiresGrad: true }
  );

  const loraB = api.tensorFromArray(
    Array.from({ length: outDim * rank }, () => 0), // Initialize to zero
    [outDim, rank],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('Base weight:', baseWeight.shape, '(frozen)');
  console.log('LoRA A:', loraA.shape, '(trainable)');
  console.log('LoRA B:', loraB.shape, '(trainable)');

  // Input
  const x = api.tensorFromArray(
    Array.from({ length: 2 * 8 * inDim }, (_, i) => Math.sin(i * 0.01)),
    [2, 8, inDim],
    { device: 'webgpu' }
  );

  // Create scale as a tensor for broadcasting
  const scaleTensor = api.tensorFromArray([scale], [1], { device: 'webgpu' });

  // Debug: Check transpose shapes (using correct API with options object)
  console.log('loraA shape:', loraA.shape, '-> transpose:', api.transpose(loraA, { dim0: 0, dim1: 1 }).shape);
  console.log('loraB shape:', loraB.shape, '-> transpose:', api.transpose(loraB, { dim0: 0, dim1: 1 }).shape);
  console.log('baseWeight shape:', baseWeight.shape, '-> transpose:', api.transpose(baseWeight, { dim0: 0, dim1: 1 }).shape);

  // LoRA forward: y = x @ W^T + (x @ A^T @ B^T) * scale
  // For LoRA: A is [rank, in], B is [out, rank]
  // So: x @ A^T = [batch, seq, in] @ [in, rank] = [batch, seq, rank]
  //     (x @ A^T) @ B^T = [batch, seq, rank] @ [rank, out] = [batch, seq, out]
  function loraLinear(input: Tensor): Tensor {
    const baseOut = api.matmul(input, api.transpose(baseWeight, { dim0: 0, dim1: 1 }));
    const step1 = api.matmul(input, api.transpose(loraA, { dim0: 0, dim1: 1 }));
    const loraOut = api.matmul(step1, api.transpose(loraB, { dim0: 0, dim1: 1 }));
    const scaledLora = api.mul(loraOut, scaleTensor);
    return api.add(baseOut, scaledLora);
  }

  // Training step with all features
  let loss = api.autocast(() => {
    const h = checkpoint(api, loraLinear, [x]);
    return h.sum();
  });

  loss = gradScaler.scale(loss);
  const scaledLossVal = await loss.item();
  console.log(`  Scaled loss: ${scaledLossVal.toExponential(4)}`);

  await loss.backward();

  // Check gradients (base should be null, LoRA should exist)
  console.log(`  baseWeight.grad exists: ${baseWeight.grad !== null} (should be false)`);
  console.log(`  loraA.grad exists: ${loraA.grad !== null} (should be true)`);
  console.log(`  loraB.grad exists: ${loraB.grad !== null} (should be true)`);

  if (loraA.grad && loraB.grad) {
    const aGradSum = await loraA.grad.sum().item();
    const bGradSum = await loraB.grad.sum().item();
    console.log(`  loraA.grad sum: ${aGradSum.toExponential(4)}`);
    console.log(`  loraB.grad sum: ${bGradSum.toExponential(4)}`);
  }

  await api.markStep();
  console.log('\n=== PASSED ===');
}

async function main(): Promise<void> {
  console.log('Combined Features Test');
  console.log('======================\n');

  await initWebGPU();

  try {
    await testCombinedFeatures();
    await testCombinedWithMoreLayers();
    await testLoRAPattern();

    console.log('\n======================');
    console.log('ALL TESTS PASSED');
    console.log('======================');
  } catch (e) {
    console.error('TEST FAILED:', e);
    process.exit(1);
  }
}

main().catch(console.error);
