#!/usr/bin/env npx tsx
/**
 * Test backward with weight tying (like GPT-2).
 * The embedding weight is used for both embedding lookup and logits projection.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Weight-Tied Backward Test');
  console.log('='.repeat(60));

  // Initialize WebGPU
  console.log('\nInitializing WebGPU...');
  const ok = await initWebGPU();
  if (!ok) throw new Error('Failed to initialize WebGPU');
  console.log('WebGPU initialized');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Dimensions
  const vocabSize = 50257;
  const embedDim = 768;
  const seqLen = 32;
  const batchSize = 1;

  // Create frozen embedding weight (like GPT-2)
  console.log('\nCreating frozen embedding weight...');
  const wteData = new Float32Array(vocabSize * embedDim);
  for (let i = 0; i < wteData.length; i++) {
    wteData[i] = (Math.random() * 2 - 1) * 0.02;
  }
  const wteWeight = api.tensorFromArray(wteData, [vocabSize, embedDim], {
    device: 'webgpu',
    requiresGrad: false, // FROZEN
  });
  console.log(`wte.weight shape: [${wteWeight.shape}], requiresGrad: ${wteWeight.requiresGrad}`);

  // Create trainable intermediate tensor (like output of transformer)
  console.log('\nCreating trainable intermediate tensor...');
  const xData = new Float32Array(batchSize * seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1) * 0.1;
  }
  const x = api.tensorFromArray(xData, [batchSize * seqLen, embedDim], {
    device: 'webgpu',
    requiresGrad: true, // TRAINABLE
  });
  console.log(`x shape: [${x.shape}], requiresGrad: ${x.requiresGrad}`);

  // Logits = x @ wte.weight.T (weight-tied projection)
  console.log('\nComputing logits (weight-tied)...');
  const wteT = wteWeight.transpose({ dim0: 0, dim1: 1 }); // [embedDim, vocabSize]
  const logits = api.matmul(x, wteT); // [batchSize * seqLen, vocabSize]
  console.log(`logits shape: [${logits.shape}]`);

  // Check logits
  const logitsSum = await logits.sum().item();
  console.log(`logits sum: ${logitsSum.toFixed(4)}`);

  if (Number.isNaN(logitsSum)) {
    console.log('ERROR: logits sum is NaN!');
    return;
  }

  // Simple loss
  const loss = logits.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toExponential(4)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check gradient
  if (x.grad) {
    const xGradSum = await x.grad.sum().item();
    console.log(`\nx.grad sum: ${xGradSum.toExponential(4)}`);
    if (Number.isNaN(xGradSum)) {
      console.log('ERROR: x.grad is NaN!');
    }
  } else {
    console.log('x.grad is null');
  }

  // Verify wte.weight doesn't have grad (frozen)
  if (wteWeight.grad) {
    console.log('wteWeight.grad exists (unexpected for frozen weight)');
    const wteGradSum = await wteWeight.grad.sum().item();
    console.log(`wteWeight.grad sum: ${wteGradSum}`);
  } else {
    console.log('wteWeight.grad is null (expected for frozen weight)');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch((e) => {
  console.error('\nFATAL ERROR:', e);
  process.exit(1);
});
