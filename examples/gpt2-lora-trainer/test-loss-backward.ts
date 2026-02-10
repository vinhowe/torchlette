#!/usr/bin/env npx tsx
/**
 * Test cross-entropy loss backward.
 * Isolates potential NaN issues in loss computation.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Cross-Entropy Loss Backward Test');
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
  const batchSize = 32; // batch * seqLen
  const vocabSize = 50257;

  // Create logits (output from model)
  console.log('\nCreating logits tensor...');
  const logitsData = new Float32Array(batchSize * vocabSize);
  for (let i = 0; i < logitsData.length; i++) {
    logitsData[i] = (Math.random() * 2 - 1) * 10; // Larger values like real logits
  }
  const logits = api.tensorFromArray(logitsData, [batchSize, vocabSize], {
    device: 'webgpu',
    requiresGrad: true,
  });
  console.log(`Logits shape: [${logits.shape}]`);

  // Create targets (random token indices)
  const targetsData = new Float32Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    targetsData[i] = Math.floor(Math.random() * vocabSize);
  }
  const targets = api.tensorFromArray(targetsData, [batchSize]);
  console.log(`Targets shape: [${targets.shape}]`);

  // Compute log-softmax manually (numerically stable)
  console.log('\nComputing numerically stable cross-entropy loss...');

  // log_softmax = x - max(x) - log(sum(exp(x - max(x))))
  const maxLogits = logits.max({ dim: -1, keepdim: true }) as Tensor;
  console.log(`maxLogits shape: [${maxLogits.shape}]`);

  const shifted = api.sub(logits, maxLogits);
  const expShifted = shifted.exp();
  const sumExp = expShifted.sum({ dim: -1, keepdim: true }) as Tensor;
  const logSumExp = sumExp.log();
  const logSoftmax = api.sub(shifted, logSumExp);

  // Check intermediate values
  const logSoftmaxSum = await logSoftmax.sum().item();
  console.log(`logSoftmax sum: ${logSoftmaxSum.toExponential(4)}`);

  // Gather target log-probs
  const targetsForGather = targets.reshape([batchSize, 1]);
  const gatheredLogProbs = api.gather(logSoftmax, targetsForGather, { dim: 1 });
  const gatheredSqueezed = gatheredLogProbs.reshape([batchSize]);

  const gatheredSum = await gatheredSqueezed.sum().item();
  console.log(`Gathered log-probs sum: ${gatheredSum.toFixed(4)}`);

  // Negate and mean for loss
  const loss = api.neg(gatheredSqueezed).mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(4)}`);

  if (Number.isNaN(lossVal)) {
    console.log('ERROR: Loss is NaN before backward!');
    return;
  }

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check gradient
  if (logits.grad) {
    const logitsGradSum = await logits.grad.sum().item();
    console.log(`\nlogits.grad sum: ${logitsGradSum.toExponential(4)}`);
    if (Number.isNaN(logitsGradSum)) {
      console.log('ERROR: logits.grad is NaN!');
    }
  } else {
    console.log('logits.grad is null');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch((e) => {
  console.error('\nFATAL ERROR:', e);
  process.exit(1);
});
