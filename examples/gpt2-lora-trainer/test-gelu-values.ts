#!/usr/bin/env npx tsx
/**
 * Test GELU backward with specific input values to find what causes NaN.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('Starting GELU value test...');
  await initWebGPU();

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  console.log(`\n${'='.repeat(60)}`);
  console.log('Testing GELU backward with various input values');
  console.log('='.repeat(60));

  // Test specific values
  const testValues = [
    0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50, 100,
    -0.5, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -20, -50, -100
  ];

  console.log('\nSingle value tests:');
  for (const val of testValues) {
    const x = api.tensorFromArray([val], [1], { device: 'webgpu', requiresGrad: true });
    const y = x.gelu();
    const yVal = await y.item();
    await y.backward();

    if (x.grad) {
      const gradVal = await x.grad.item();
      const status = Number.isNaN(gradVal) ? ' [NaN!]' : '';
      console.log(`x=${val.toString().padStart(5)}: gelu=${yVal.toFixed(4).padStart(10)}, grad=${gradVal.toFixed(4).padStart(10)}${status}`);
    }
  }

  // Test range around the boundary
  console.log('\nRange tests (values where tanh inner approaches ±10):');
  // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
  // For inner = 10: need x where 0.798 * (x + 0.044715 * x^3) = 10
  // Approximately x ≈ 4.9 for inner = 10
  const boundaryValues = [4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.5, 6.0];
  for (const val of boundaryValues) {
    const x = api.tensorFromArray([val], [1], { device: 'webgpu', requiresGrad: true });
    const y = x.gelu();
    const yVal = await y.item();
    await y.backward();

    const inner = 0.7978845608 * (val + 0.044715 * val * val * val);

    if (x.grad) {
      const gradVal = await x.grad.item();
      const status = Number.isNaN(gradVal) ? ' [NaN!]' : '';
      console.log(`x=${val.toFixed(2)}: inner=${inner.toFixed(2)}, gelu=${yVal.toFixed(4)}, grad=${gradVal.toFixed(6)}${status}`);
    }
  }

  // Test with batch to see if batched computation has issues
  console.log('\nBatch test (values 0 to 15):');
  {
    const vals = new Float32Array(16);
    for (let i = 0; i < 16; i++) vals[i] = i;

    const x = api.tensorFromArray(vals, [16], { device: 'webgpu', requiresGrad: true });
    const y = x.gelu();
    const loss = y.sum() as Tensor;
    await loss.backward();

    if (x.grad) {
      const gradSum = await x.grad.sum().item();
      console.log(`Gradient sum: ${gradSum.toExponential(4)}${Number.isNaN(gradSum) ? ' [NaN!]' : ''}`);
    }
  }

  // Test with the actual GPT-2 fc output range
  console.log('\nTest with GPT-2 like range (-13 to 12):');
  {
    const numElements = 3072;
    const vals = new Float32Array(numElements);
    for (let i = 0; i < numElements; i++) {
      vals[i] = (i / numElements) * 25 - 13; // Range from -13 to +12
    }

    const x = api.tensorFromArray(vals, [numElements], { device: 'webgpu', requiresGrad: true });
    const y = x.gelu();
    const loss = y.sum() as Tensor;
    await loss.backward();

    if (x.grad) {
      const gradSum = await x.grad.sum().item();
      const gradMax = await (x.grad.max() as Tensor).item();
      const gradMin = await x.grad.neg().max().item();
      console.log(`Gradient: sum=${gradSum.toExponential(4)}, min=${(-gradMin).toFixed(6)}, max=${gradMax.toFixed(6)}`);
      if (Number.isNaN(gradSum)) console.log('  ^^^ NaN detected in gradient!');
    }
  }

  // Test with CPU backend for comparison
  console.log('\nCPU reference test:');
  {
    const api2 = new Torchlette('cpu', {});

    const testVals = [10, 11, 12, 13, -10, -11, -12, -13];
    for (const val of testVals) {
      const x = api2.tensorFromArray([val], [1], { requiresGrad: true });
      const y = x.gelu();
      const yVal = await y.item();
      await y.backward();

      if (x.grad) {
        const gradVal = await x.grad.item();
        console.log(`CPU: x=${val.toString().padStart(4)}, gelu=${yVal.toFixed(4).padStart(10)}, grad=${gradVal.toFixed(6).padStart(10)}`);
      }
    }
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
