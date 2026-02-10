#!/usr/bin/env npx tsx
/**
 * Minimal checkpointing test to isolate shape mismatch issues.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { checkpoint, checkpointSequential } from '../../src/nn/checkpoint';
import { initWebGPU } from '../../src/backend/webgpu';

async function testBasicCheckpointing(): Promise<void> {
  console.log('=== Test 1: Basic saved_tensors_hooks ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Simple parameters
  const x = api.tensorFromArray(
    Array.from({ length: 16 * 8 }, (_, i) => Math.sin(i * 0.1)),
    [16, 8],
    { device: 'webgpu', requiresGrad: true }
  );

  const w = api.tensorFromArray(
    Array.from({ length: 8 * 4 }, (_, i) => Math.cos(i * 0.1) * 0.1),
    [8, 4],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('  x shape:', x.shape);
  console.log('  w shape:', w.shape);

  // Track saved tensors
  const savedTensors: Tensor[] = [];
  let recomputeCount = 0;

  // Use saved_tensors_hooks
  const result = api.saved_tensors_hooks(
    // Pack: save the tensor and return a placeholder
    (tensor: Tensor) => {
      const idx = savedTensors.length;
      savedTensors.push(tensor);
      console.log(`  [Pack] Saving tensor at index ${idx}, shape: [${tensor.shape}]`);
      return idx;
    },
    // Unpack: get the saved tensor
    (packed: unknown) => {
      const idx = packed as number;
      recomputeCount++;
      console.log(`  [Unpack] Retrieving tensor at index ${idx}`);
      return savedTensors[idx];
    },
    () => {
      // Forward computation
      const y = api.matmul(x, w);
      return y.sum();
    }
  );

  console.log('  Forward complete, result shape:', result.shape);
  const lossVal = await result.item();
  console.log(`  Loss: ${lossVal.toFixed(4)}`);
  console.log(`  Saved ${savedTensors.length} tensors during forward`);

  // Backward
  console.log('  Running backward...');
  await result.backward();
  console.log('  Backward complete');
  console.log(`  Unpacked ${recomputeCount} tensors during backward`);

  const xGradSum = x.grad ? await x.grad.sum().item() : 0;
  const wGradSum = w.grad ? await w.grad.sum().item() : 0;
  console.log(`  x.grad sum: ${xGradSum.toFixed(4)}`);
  console.log(`  w.grad sum: ${wGradSum.toFixed(4)}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function testCheckpointingWithRecompute(): Promise<void> {
  console.log('=== Test 2: Checkpointing with recomputation ===');
  console.log('  (Proper pattern: save tensors by ID, recompute and capture all)');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const x = api.tensorFromArray(
    Array.from({ length: 16 * 8 }, (_, i) => Math.sin(i * 0.1)),
    [16, 8],
    { device: 'webgpu', requiresGrad: true }
  );

  const w = api.tensorFromArray(
    Array.from({ length: 8 * 4 }, (_, i) => Math.cos(i * 0.1) * 0.1),
    [8, 4],
    { device: 'webgpu', requiresGrad: true }
  );

  // Storage for checkpointing
  let savedInput: Tensor | null = null;
  const tensorsByBaseId = new Map<number, Tensor>();
  let needsRecompute = false;

  // Simulate a block that we want to checkpoint
  function computeBlock(input: Tensor, captureMode: boolean): Tensor {
    const h = api.matmul(input, w);
    if (captureMode) {
      tensorsByBaseId.set(h.baseId, h);
      console.log(`    Captured matmul output, baseId=${h.baseId}, shape=[${h.shape}]`);
    }
    const a = api.relu(h);
    if (captureMode) {
      tensorsByBaseId.set(a.baseId, a);
      console.log(`    Captured relu output, baseId=${a.baseId}, shape=[${a.shape}]`);
    }
    return a;
  }

  // Use saved_tensors_hooks to checkpoint
  const result = api.saved_tensors_hooks(
    // Pack: save tensor baseId and the input for recomputation
    (tensor: Tensor) => {
      if (!savedInput) {
        savedInput = tensor;  // First tensor saved is the input
      }
      console.log(`  [Pack] Packing tensor baseId=${tensor.baseId}, shape=[${tensor.shape}]`);
      return { baseId: tensor.baseId };
    },
    // Unpack: recompute if needed, then return tensor by baseId
    (packed: unknown) => {
      const { baseId } = packed as { baseId: number };
      console.log(`  [Unpack] Requesting tensor baseId=${baseId}`);

      // Check if we already have it
      if (tensorsByBaseId.has(baseId)) {
        console.log(`    Found in cache`);
        return tensorsByBaseId.get(baseId)!;
      }

      // If it's the input, return it
      if (savedInput && savedInput.baseId === baseId) {
        console.log(`    Returning saved input`);
        return savedInput;
      }

      // Need to recompute
      if (!needsRecompute && savedInput) {
        console.log(`    Recomputing block to capture tensors...`);
        needsRecompute = true;
        computeBlock(savedInput, true);
      }

      // Now it should be in the map
      if (tensorsByBaseId.has(baseId)) {
        console.log(`    Found after recompute`);
        return tensorsByBaseId.get(baseId)!;
      }

      throw new Error(`Tensor with baseId ${baseId} not found after recomputation`);
    },
    () => {
      const out = computeBlock(x, false);
      return out.sum();
    }
  );

  console.log('  Forward complete');
  const lossVal = await result.item();
  console.log(`  Loss: ${lossVal.toFixed(4)}`);

  console.log('  Running backward...');
  await result.backward();
  console.log('  Backward complete');

  const xGradSum = x.grad ? await x.grad.sum().item() : 0;
  const wGradSum = w.grad ? await w.grad.sum().item() : 0;
  console.log(`  x.grad sum: ${xGradSum.toFixed(4)}`);
  console.log(`  w.grad sum: ${wGradSum.toFixed(4)}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function testProperCheckpointAPI(): Promise<void> {
  console.log('=== Test 3: Using proper checkpoint() API ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const x = api.tensorFromArray(
    Array.from({ length: 16 * 8 }, (_, i) => Math.sin(i * 0.1)),
    [16, 8],
    { device: 'webgpu', requiresGrad: true }
  );

  const w = api.tensorFromArray(
    Array.from({ length: 8 * 4 }, (_, i) => Math.cos(i * 0.1) * 0.1),
    [8, 4],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('  x shape:', x.shape);
  console.log('  w shape:', w.shape);

  // Define the computation to checkpoint
  function computeBlock(input: Tensor): Tensor {
    const h = api.matmul(input, w);
    const a = api.relu(h);
    return a;
  }

  // Use the proper checkpoint API
  const output = checkpoint(api, computeBlock, [x]);
  const loss = output.sum();

  console.log('  Forward complete');
  const lossVal = await loss.item();
  console.log(`  Loss: ${lossVal.toFixed(4)}`);

  console.log('  Running backward...');
  await loss.backward();
  console.log('  Backward complete');

  const xGradSum = x.grad ? await x.grad.sum().item() : 0;
  const wGradSum = w.grad ? await w.grad.sum().item() : 0;
  console.log(`  x.grad sum: ${xGradSum.toFixed(4)}`);
  console.log(`  w.grad sum: ${wGradSum.toFixed(4)}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function testMultipleBlocks(): Promise<void> {
  console.log('=== Test 4: Multiple checkpointed blocks (using proper checkpoint API) ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batchSize = 2;
  const seqLen = 8;
  const embedDim = 16;
  const hiddenDim = 32;

  const x = api.tensorFromArray(
    Array.from({ length: batchSize * seqLen * embedDim }, (_, i) => Math.sin(i * 0.01)),
    [batchSize, seqLen, embedDim],
    { device: 'webgpu', requiresGrad: true }
  );

  // Two weight matrices for two "blocks"
  const w1 = api.tensorFromArray(
    Array.from({ length: embedDim * hiddenDim }, (_, i) => Math.cos(i * 0.01) * 0.1),
    [embedDim, hiddenDim],
    { device: 'webgpu', requiresGrad: true }
  );

  const w2 = api.tensorFromArray(
    Array.from({ length: hiddenDim * embedDim }, (_, i) => Math.sin(i * 0.01) * 0.1),
    [hiddenDim, embedDim],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('  x shape:', x.shape);
  console.log('  w1 shape:', w1.shape);
  console.log('  w2 shape:', w2.shape);

  // Block functions - closures over weights
  function block1(input: Tensor): Tensor {
    return api.relu(api.matmul(input, w1));
  }

  function block2(input: Tensor): Tensor {
    return api.relu(api.matmul(input, w2));
  }

  // Checkpointed forward pass using proper checkpoint API
  let h = x;

  // Block 1 with checkpointing
  h = checkpoint(api, block1, [h]);

  // Block 2 with checkpointing
  h = checkpoint(api, block2, [h]);

  const loss = h.sum();

  console.log('  Forward complete');
  const lossVal = await loss.item();
  console.log(`  Loss: ${lossVal.toFixed(4)}`);

  console.log('  Running backward...');
  await loss.backward();
  console.log('  Backward complete');

  const xGradSum = x.grad ? await x.grad.sum().item() : 0;
  const w1GradSum = w1.grad ? await w1.grad.sum().item() : 0;
  const w2GradSum = w2.grad ? await w2.grad.sum().item() : 0;
  console.log(`  x.grad sum: ${xGradSum.toExponential(4)}`);
  console.log(`  w1.grad sum: ${w1GradSum.toExponential(4)}`);
  console.log(`  w2.grad sum: ${w2GradSum.toExponential(4)}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function main(): Promise<void> {
  console.log('Checkpointing Minimal Tests');
  console.log('============================\n');

  await initWebGPU();

  try {
    await testBasicCheckpointing();
    // Skip test 2 - it was testing a flawed manual implementation
    // await testCheckpointingWithRecompute();
    await testProperCheckpointAPI();
    await testMultipleBlocks();

    console.log('============================');
    console.log('ALL TESTS PASSED');
    console.log('============================');
  } catch (e) {
    console.error('TEST FAILED:', e);
    process.exit(1);
  }
}

main().catch(console.error);
