#!/usr/bin/env npx tsx
/**
 * Minimal AMP test to isolate memory issues.
 */

import { Torchlette, type FrontendTensor as Tensor, GradScaler } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function testAMPBasic(): Promise<void> {
  console.log('=== Test 1: Basic AMP forward/backward ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Small test case
  const x = api.tensorFromArray(
    Array.from({ length: 64 * 32 }, (_, i) => Math.sin(i * 0.01)),
    [64, 32],
    { device: 'webgpu', requiresGrad: true }
  );

  const w = api.tensorFromArray(
    Array.from({ length: 32 * 16 }, (_, i) => Math.cos(i * 0.01) * 0.1),
    [32, 16],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('  x shape:', x.shape);
  console.log('  w shape:', w.shape);

  // Forward with autocast
  const y = api.autocast(() => {
    const out = api.matmul(x, w);
    return out.sum();
  });

  console.log('  y (loss) created');
  const lossVal = await y.item();
  console.log(`  Loss: ${lossVal.toFixed(4)}`);

  // Backward
  console.log('  Running backward...');
  await y.backward();
  console.log('  Backward complete');

  // Check gradients
  const xGradSum = x.grad ? await x.grad.sum().item() : 0;
  const wGradSum = w.grad ? await w.grad.sum().item() : 0;
  console.log(`  x.grad sum: ${xGradSum.toFixed(4)}`);
  console.log(`  w.grad sum: ${wGradSum.toFixed(4)}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function testAMPWithGradScaler(): Promise<void> {
  console.log('=== Test 2: AMP with GradScaler ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const gradScaler = new GradScaler(api, {
    initScale: 65536.0,
    growthFactor: 2.0,
    backoffFactor: 0.5,
    growthInterval: 2000,
  });

  // Small test case
  const x = api.tensorFromArray(
    Array.from({ length: 64 * 32 }, (_, i) => Math.sin(i * 0.01)),
    [64, 32],
    { device: 'webgpu', requiresGrad: true }
  );

  const w = api.tensorFromArray(
    Array.from({ length: 32 * 16 }, (_, i) => Math.cos(i * 0.01) * 0.1),
    [32, 16],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('  x shape:', x.shape);
  console.log('  w shape:', w.shape);
  console.log('  Initial scale:', gradScaler.getScale());

  // Forward with autocast
  let loss = api.autocast(() => {
    const out = api.matmul(x, w);
    return out.sum();
  });

  console.log('  Loss tensor created');

  // Scale the loss
  loss = gradScaler.scale(loss);
  console.log('  Loss scaled');

  const scaledLossVal = await loss.item();
  console.log(`  Scaled loss: ${scaledLossVal.toExponential(4)}`);

  // Backward
  console.log('  Running backward...');
  await loss.backward();
  console.log('  Backward complete');

  // Check gradients exist
  console.log(`  x.grad exists: ${x.grad !== null}`);
  console.log(`  w.grad exists: ${w.grad !== null}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function testAMPMultipleSteps(): Promise<void> {
  console.log('=== Test 3: AMP multiple training steps ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const gradScaler = new GradScaler(api, {
    initScale: 65536.0,
    growthFactor: 2.0,
    backoffFactor: 0.5,
    growthInterval: 2000,
  });

  // Parameters
  const w = api.tensorFromArray(
    Array.from({ length: 32 * 16 }, (_, i) => Math.cos(i * 0.01) * 0.1),
    [32, 16],
    { device: 'webgpu', requiresGrad: true }
  );

  for (let step = 0; step < 3; step++) {
    console.log(`  Step ${step + 1}:`);

    // Fresh input each step
    const x = api.tensorFromArray(
      Array.from({ length: 64 * 32 }, (_, i) => Math.sin(i * 0.01 + step)),
      [64, 32],
      { device: 'webgpu' }
    );

    // Forward with autocast
    let loss = api.autocast(() => {
      const out = api.matmul(x, w);
      return out.sum();
    });

    // Scale
    loss = gradScaler.scale(loss);
    const scaledLossVal = await loss.item();
    console.log(`    Scaled loss: ${scaledLossVal.toExponential(4)}`);

    // Backward
    await loss.backward();

    // Check gradient
    const wGradSum = w.grad ? await w.grad.sum().item() : 0;
    console.log(`    w.grad sum: ${wGradSum.toExponential(4)}`);

    // Zero grad for next step
    w.zeroGrad();

    // Clean up
    await api.markStep();
  }

  console.log('  PASSED\n');
}

async function testAMPLargerModel(): Promise<void> {
  console.log('=== Test 4: AMP with larger tensors (like GPT-2) ===');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // GPT-2 like dimensions
  const batchSize = 1;
  const seqLen = 32;
  const embedDim = 768;
  const hiddenDim = 3072;

  console.log(`  Dimensions: [${batchSize}, ${seqLen}, ${embedDim}] x [${embedDim}, ${hiddenDim}]`);

  const x = api.tensorFromArray(
    Array.from({ length: batchSize * seqLen * embedDim }, (_, i) => Math.sin(i * 0.001)),
    [batchSize, seqLen, embedDim],
    { device: 'webgpu', requiresGrad: true }
  );

  const w = api.tensorFromArray(
    Array.from({ length: embedDim * hiddenDim }, (_, i) => Math.cos(i * 0.001) * 0.02),
    [embedDim, hiddenDim],
    { device: 'webgpu', requiresGrad: true }
  );

  console.log('  Tensors created');

  // Forward with autocast (no grad scaler to isolate issue)
  const loss = api.autocast(() => {
    const out = api.matmul(x, w);  // [1, 32, 3072]
    return out.sum();
  });

  console.log('  Forward complete');
  const lossVal = await loss.item();
  console.log(`  Loss: ${lossVal.toFixed(4)}`);

  // Backward
  console.log('  Running backward...');
  await loss.backward();
  console.log('  Backward complete');

  const xGradSum = x.grad ? await x.grad.sum().item() : 0;
  const wGradSum = w.grad ? await w.grad.sum().item() : 0;
  console.log(`  x.grad sum: ${xGradSum.toExponential(4)}`);
  console.log(`  w.grad sum: ${wGradSum.toExponential(4)}`);

  await api.markStep();
  console.log('  PASSED\n');
}

async function main(): Promise<void> {
  console.log('AMP Minimal Tests');
  console.log('==================\n');

  await initWebGPU();

  try {
    await testAMPBasic();
    await testAMPWithGradScaler();
    await testAMPMultipleSteps();
    await testAMPLargerModel();

    console.log('==================');
    console.log('ALL TESTS PASSED');
    console.log('==================');
  } catch (e) {
    console.error('TEST FAILED:', e);
    process.exit(1);
  }
}

main().catch(console.error);
