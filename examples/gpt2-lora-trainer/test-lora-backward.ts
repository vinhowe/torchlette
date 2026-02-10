#!/usr/bin/env npx tsx
/**
 * Minimal test for LoRA backward pass.
 * Isolates the NaN gradient issue.
 */

import { Torchlette, type FrontendTensor as Tensor, Adam } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Minimal LoRA Backward Test');
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

  // LoRA dimensions (like first layer cAttn)
  const inFeatures = 768;
  const outFeatures = 2304; // 3 * 768 for QKV
  const rank = 8;
  const scaling = 2.0; // alpha / rank = 16 / 8

  // Create LoRA parameters
  console.log('\nCreating LoRA parameters...');
  const loraAData = new Float32Array(rank * inFeatures);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * 0.1;
  }
  const loraA = api.tensorFromArray(loraAData, [rank, inFeatures], {
    device: 'webgpu',
    requiresGrad: true,
  });
  const loraB = api.zeros([outFeatures, rank], {
    device: 'webgpu',
    requiresGrad: true,
  });

  console.log(`loraA shape: [${loraA.shape}], requiresGrad: ${loraA.requiresGrad}`);
  console.log(`loraB shape: [${loraB.shape}], requiresGrad: ${loraB.requiresGrad}`);

  // Verify loraB is zeros
  const loraBSum = await loraB.sum().item();
  console.log(`loraB sum: ${loraBSum} (should be 0)`);

  // Create input (like from layer norm)
  const batchSize = 1;
  const seqLen = 32;
  const inputData = new Float32Array(batchSize * seqLen * inFeatures);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.random() * 2 - 1;
  }
  const x = api.tensorFromArray(inputData, [batchSize * seqLen, inFeatures], {
    device: 'webgpu',
  });
  console.log(`\nInput x shape: [${x.shape}]`);

  // LoRA forward: scaling * (x @ loraA.T @ loraB.T)
  console.log('\nComputing LoRA forward...');
  const xA = api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 })); // [B*S, rank]
  const loraOut = api.matmul(xA, loraB.transpose({ dim0: 0, dim1: 1 })); // [B*S, out]
  const scalingTensor = api.tensorFromArray([scaling], []);
  const scaledLora = api.mul(loraOut, scalingTensor);

  console.log(`xA shape: [${xA.shape}]`);
  console.log(`loraOut shape: [${loraOut.shape}]`);
  console.log(`scaledLora shape: [${scaledLora.shape}]`);

  // Check intermediate values
  const xASum = await xA.sum().item();
  const loraOutSum = await loraOut.sum().item();
  const scaledLoraSum = await scaledLora.sum().item();
  console.log(`\nIntermediate sums:`);
  console.log(`  xA sum: ${xASum}`);
  console.log(`  loraOut sum: ${loraOutSum} (should be ~0 since loraB=0)`);
  console.log(`  scaledLora sum: ${scaledLoraSum} (should be ~0)`);

  // Simple loss: mean of scaledLora
  const loss = scaledLora.mean();
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal} (should be ~0)`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check gradients
  if (loraA.grad) {
    const loraAGradSum = await loraA.grad.sum().item();
    console.log(`\nloraA.grad sum: ${loraAGradSum}`);
    if (Number.isNaN(loraAGradSum)) {
      console.log('ERROR: loraA.grad is NaN!');
    }
  } else {
    console.log('loraA.grad is null');
  }

  if (loraB.grad) {
    const loraBGradSum = await loraB.grad.sum().item();
    console.log(`loraB.grad sum: ${loraBGradSum}`);
    if (Number.isNaN(loraBGradSum)) {
      console.log('ERROR: loraB.grad is NaN!');
    }
  } else {
    console.log('loraB.grad is null');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch((e) => {
  console.error('\nFATAL ERROR:', e);
  process.exit(1);
});
