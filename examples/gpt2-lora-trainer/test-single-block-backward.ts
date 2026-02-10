#!/usr/bin/env npx tsx
/**
 * Test a single transformer block with LoRA.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { LoRALinear, createLoRAConfig } from './src/lib/torchlette/lora';

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Single Transformer Block with LoRA Test');
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

  // Dimensions (GPT-2 small)
  const embedDim = 768;
  const numHeads = 12;
  const headDim = embedDim / numHeads;
  const seqLen = 32;
  const batchSize = 1;

  // Create LoRA config
  const loraConfig = createLoRAConfig(8, 16);

  // Create LoRA layer for cAttn (combined QKV)
  console.log('\nCreating LoRA layer...');
  const cAttn = new LoRALinear(api, embedDim, 3 * embedDim, loraConfig, { device: 'webgpu' });

  // Load some "base weights" (random for testing)
  const baseWeightData = new Float32Array(3 * embedDim * embedDim);
  for (let i = 0; i < baseWeightData.length; i++) {
    baseWeightData[i] = (Math.random() * 2 - 1) * 0.02;
  }
  const baseBiasData = new Float32Array(3 * embedDim);
  const baseWeight = api.tensorFromArray(baseWeightData, [3 * embedDim, embedDim]);
  const baseBias = api.tensorFromArray(baseBiasData, [3 * embedDim]);
  cAttn.loadBaseWeights(baseWeight, baseBias);

  console.log(`LoRA loraA shape: [${cAttn.loraA.shape}]`);
  console.log(`LoRA loraB shape: [${cAttn.loraB.shape}]`);

  // Verify loraB is zeros
  const loraBSum = await cAttn.loraB.sum().item();
  console.log(`loraB sum: ${loraBSum} (should be 0)`);

  // Create input (like from layer norm)
  console.log('\nCreating input...');
  const xData = new Float32Array(batchSize * seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1) * 0.1;
  }
  const x = api.tensorFromArray(xData, [batchSize, seqLen, embedDim], {
    device: 'webgpu',
    requiresGrad: true,
  });
  console.log(`Input x shape: [${x.shape}], requiresGrad: ${x.requiresGrad}`);

  // Forward through LoRA layer
  console.log('\nRunning LoRA forward...');
  const qkv = cAttn.forward(x);
  console.log(`QKV output shape: [${qkv.shape}]`);

  // Check output
  const qkvSum = await qkv.sum().item();
  console.log(`QKV sum: ${qkvSum.toFixed(4)}`);

  if (Number.isNaN(qkvSum)) {
    console.log('ERROR: QKV sum is NaN!');
    return;
  }

  // Simple loss (skip full attention, just use output)
  const loss = qkv.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toExponential(4)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check LoRA gradients
  if (cAttn.loraA.grad) {
    const loraAGradSum = await cAttn.loraA.grad.sum().item();
    console.log(`\nloraA.grad sum: ${loraAGradSum.toExponential(4)}`);
    if (Number.isNaN(loraAGradSum)) {
      console.log('ERROR: loraA.grad is NaN!');
    }
  } else {
    console.log('loraA.grad is null');
  }

  if (cAttn.loraB.grad) {
    const loraBGradSum = await cAttn.loraB.grad.sum().item();
    console.log(`loraB.grad sum: ${loraBGradSum.toExponential(4)}`);
    if (Number.isNaN(loraBGradSum)) {
      console.log('ERROR: loraB.grad is NaN!');
    }
  } else {
    console.log('loraB.grad is null');
  }

  // Check input gradient
  if (x.grad) {
    const xGradSum = await x.grad.sum().item();
    console.log(`x.grad sum: ${xGradSum.toExponential(4)}`);
    if (Number.isNaN(xGradSum)) {
      console.log('ERROR: x.grad is NaN!');
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch((e) => {
  console.error('\nFATAL ERROR:', e);
  process.exit(1);
});
