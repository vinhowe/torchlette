#!/usr/bin/env npx tsx
/**
 * Debug memory leak in training loop.
 *
 * This test isolates the memory issue by tracking allocations at each phase.
 */

import { Torchlette, type FrontendTensor as Tensor, Adam } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { getGPUMemoryStats } from '../../src/backend/webgpu/memory-tracker';
import { storageTracker } from '../../src/engine/lazy';
import { getAllPendingTensors, hasPendingTensors } from '../../src/runtime/tensor';

function logPendingTensors(label: string): void {
  const pending = getAllPendingTensors();
  console.log(`  [PENDING ${label}] ${pending.length} pending tensors`);
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`;
  }
  return `${(bytes / 1024).toFixed(2)}KB`;
}

function logMemory(label: string): void {
  const gpuStats = getGPUMemoryStats();
  const storageStats = storageTracker.stats();
  console.log(`  [MEM ${label}] GPU: ${formatBytes(gpuStats.currentBytes)} (${gpuStats.allocationCount} allocs) | Storage: total=${storageStats.totalStorages} reachable=${storageStats.reachableStorages} unreachable=${storageStats.unreachableStorages}`);
}

async function testSimpleTrainingLoop(): Promise<void> {
  console.log('=== Test 1: Simple matmul training loop ===\n');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Simple model: just a weight matrix
  const inDim = 768;
  const outDim = 768;

  const w = api.tensorFromArray(
    Array.from({ length: inDim * outDim }, (_, i) => Math.cos(i * 0.001) * 0.02),
    [inDim, outDim],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('Weight shape:', w.shape);
  logMemory('after weight creation');

  const optimizer = new Adam([w], { lr: 0.001 }, api);
  logMemory('after optimizer creation');

  // Training loop - test 10 steps to see memory pattern
  for (let step = 0; step < 10; step++) {
    console.log(`\n--- Step ${step + 1} ---`);
    logMemory('start of step');

    // Create input (fresh each step)
    const x = api.tensorFromArray(
      Array.from({ length: 32 * inDim }, (_, i) => Math.sin(i * 0.01 + step)),
      [32, inDim],
      { device: 'webgpu' }
    );
    logMemory('after input creation');

    // Forward
    const y = api.matmul(x, w);
    const loss = y.sum();
    logMemory('after forward');

    // Get loss value
    const lossVal = await loss.item();
    console.log(`  Loss: ${lossVal.toFixed(4)}`);
    logMemory('after loss.item()');

    // Backward
    await loss.backward();
    logMemory('after backward');
    logPendingTensors('after backward');

    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();
    logMemory('after optimizer step');
    logPendingTensors('after optimizer');

    // Dispose forward pass tensors
    x.dispose();
    y.dispose();
    loss.dispose();
    logMemory('after dispose forward tensors');
    logPendingTensors('after dispose forward tensors');

    // Mark step
    await api.markStep();
    logMemory('after markStep');
    logPendingTensors('after markStep');
  }

  console.log('\n=== Test 1 Complete ===');
}

async function testMultiLayerTrainingLoop(): Promise<void> {
  console.log('\n=== Test 2: Multi-layer training loop (like transformer) ===\n');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Simulate 4 layers
  const numLayers = 4;
  const dim = 256;

  const weights: Tensor[] = [];
  for (let i = 0; i < numLayers; i++) {
    const w = api.tensorFromArray(
      Array.from({ length: dim * dim }, (_, j) => Math.cos(j * 0.001 + i) * 0.02),
      [dim, dim],
      { device: 'webgpu', requiresGrad: true }
    );
    weights.push(w);
  }

  console.log(`Created ${numLayers} weight matrices, each [${dim}, ${dim}]`);
  logMemory('after weight creation');

  const optimizer = new Adam(weights, { lr: 0.001 }, api);
  logMemory('after optimizer creation');

  // Training loop
  for (let step = 0; step < 5; step++) {
    console.log(`\n--- Step ${step + 1} ---`);
    logMemory('start of step');

    // Create input
    const x = api.tensorFromArray(
      Array.from({ length: 16 * dim }, (_, i) => Math.sin(i * 0.01 + step)),
      [16, dim],
      { device: 'webgpu' }
    );
    logMemory('after input creation');

    // Forward through all layers
    let h = x;
    for (let i = 0; i < numLayers; i++) {
      h = api.relu(api.matmul(h, weights[i]));
    }
    const loss = h.sum();
    logMemory('after forward');

    // Get loss value
    const lossVal = await loss.item();
    console.log(`  Loss: ${lossVal.toFixed(4)}`);
    logMemory('after loss.item()');

    // Backward
    await loss.backward();
    logMemory('after backward');

    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();
    logMemory('after optimizer step');

    // Dispose input
    x.dispose();
    logMemory('after x.dispose()');

    // Mark step
    await api.markStep();
    logMemory('after markStep');
  }

  console.log('\n=== Test 2 Complete ===');
}

async function testWithTidy(): Promise<void> {
  console.log('\n=== Test 3: Training with tidy() wrapper ===\n');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const dim = 256;
  const numLayers = 4;

  const weights: Tensor[] = [];
  for (let i = 0; i < numLayers; i++) {
    const w = api.tensorFromArray(
      Array.from({ length: dim * dim }, (_, j) => Math.cos(j * 0.001 + i) * 0.02),
      [dim, dim],
      { device: 'webgpu', requiresGrad: true }
    );
    weights.push(w);
  }

  logMemory('after weight creation');

  const optimizer = new Adam(weights, { lr: 0.001 }, api);
  logMemory('after optimizer creation');

  // Training loop WITH tidy
  for (let step = 0; step < 5; step++) {
    console.log(`\n--- Step ${step + 1} ---`);
    logMemory('start of step');

    // Wrap forward/backward in tidy
    const loss = api.tidy(() => {
      const x = api.tensorFromArray(
        Array.from({ length: 16 * dim }, (_, i) => Math.sin(i * 0.01 + step)),
        [16, dim],
        { device: 'webgpu' }
      );

      let h = x;
      for (let i = 0; i < numLayers; i++) {
        h = api.relu(api.matmul(h, weights[i]));
      }
      const lossT = h.sum();
      api.keep(lossT); // Keep loss tensor so it survives tidy
      return lossT;
    });
    logMemory('after tidy forward');

    // Get loss value
    const lossVal = await loss.item();
    console.log(`  Loss: ${lossVal.toFixed(4)}`);
    logMemory('after loss.item()');

    // Backward
    await loss.backward();
    logMemory('after backward');

    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();
    logMemory('after optimizer step');

    // Mark step
    await api.markStep();
    logMemory('after markStep');
  }

  console.log('\n=== Test 3 Complete ===');
}

async function main(): Promise<void> {
  console.log('Memory Leak Debug Tests');
  console.log('=======================\n');

  await initWebGPU();
  logMemory('initial');

  try {
    await testSimpleTrainingLoop();
    // Skip other tests for now to focus on understanding Test 1
    // await testMultiLayerTrainingLoop();
    // await testWithTidy();

    console.log('\n=======================');
    console.log('TEST COMPLETE');
    const finalStats = getGPUMemoryStats();
    console.log(`Peak memory usage: ${formatBytes(finalStats.peakBytes)}`);
    console.log('=======================');
  } catch (e) {
    console.error('TEST FAILED:', e);
    process.exit(1);
  }
}

main().catch(console.error);
