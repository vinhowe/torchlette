#!/usr/bin/env npx tsx
/**
 * Test attention backward pass with causal mask.
 * Isolates potential NaN issues in masked softmax backward.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Attention Backward Test');
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

  // Attention dimensions
  const batchSize = 1;
  const numHeads = 12;
  const seqLen = 32;
  const headDim = 64;

  // Create Q, K, V with gradients
  console.log('\nCreating Q, K, V tensors...');
  const qData = new Float32Array(batchSize * numHeads * seqLen * headDim);
  const kData = new Float32Array(batchSize * numHeads * seqLen * headDim);
  const vData = new Float32Array(batchSize * numHeads * seqLen * headDim);
  for (let i = 0; i < qData.length; i++) {
    qData[i] = (Math.random() * 2 - 1) * 0.1;
    kData[i] = (Math.random() * 2 - 1) * 0.1;
    vData[i] = (Math.random() * 2 - 1) * 0.1;
  }

  const q = api.tensorFromArray(qData, [batchSize * numHeads, seqLen, headDim], {
    device: 'webgpu',
    requiresGrad: true,
  });
  const k = api.tensorFromArray(kData, [batchSize * numHeads, seqLen, headDim], {
    device: 'webgpu',
    requiresGrad: true,
  });
  const v = api.tensorFromArray(vData, [batchSize * numHeads, seqLen, headDim], {
    device: 'webgpu',
    requiresGrad: true,
  });

  console.log(`Q shape: [${q.shape}]`);
  console.log(`K shape: [${k.shape}]`);
  console.log(`V shape: [${v.shape}]`);

  // Create causal mask
  console.log('\nCreating causal mask...');
  const mask = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask[i * seqLen + j] = j > i ? -1e9 : 0;
    }
  }
  const maskTensor = api.tensorFromArray(mask, [seqLen, seqLen]);

  // Attention computation
  console.log('\nComputing attention...');

  // scores = Q @ K^T / sqrt(d_k)
  const kT = k.transpose({ dim0: 1, dim1: 2 }); // [batch*heads, head_dim, seq]
  const scores = api.matmul(q, kT);
  console.log(`Scores shape: [${scores.shape}]`);

  const scale = api.tensorFromArray([1.0 / Math.sqrt(headDim)], []);
  const scaledScores = api.mul(scores, scale);

  // Check scores before masking
  const scoresSum = await scaledScores.sum().item();
  console.log(`Scaled scores sum: ${scoresSum.toFixed(4)}`);

  // Apply mask
  const maskedScores = api.add(scaledScores, maskTensor);

  // Softmax
  const attnWeights = maskedScores.softmax(-1);

  // Check attention weights
  const attnSum = await attnWeights.sum().item();
  console.log(`Attention weights sum: ${attnSum.toFixed(2)} (should be ~${batchSize * numHeads * seqLen})`);

  // Apply to values
  const attnOut = api.matmul(attnWeights, v);
  console.log(`Attention output shape: [${attnOut.shape}]`);

  // Loss
  const loss = attnOut.mean();
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(6)}`);

  if (Number.isNaN(lossVal)) {
    console.log('ERROR: Loss is NaN before backward!');
    return;
  }

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check gradients
  if (q.grad) {
    const qGradSum = await q.grad.sum().item();
    console.log(`\nQ.grad sum: ${qGradSum}`);
    if (Number.isNaN(qGradSum)) {
      console.log('ERROR: Q.grad is NaN!');
    }
  }

  if (k.grad) {
    const kGradSum = await k.grad.sum().item();
    console.log(`K.grad sum: ${kGradSum}`);
    if (Number.isNaN(kGradSum)) {
      console.log('ERROR: K.grad is NaN!');
    }
  }

  if (v.grad) {
    const vGradSum = await v.grad.sum().item();
    console.log(`V.grad sum: ${vGradSum}`);
    if (Number.isNaN(vGradSum)) {
      console.log('ERROR: V.grad is NaN!');
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
