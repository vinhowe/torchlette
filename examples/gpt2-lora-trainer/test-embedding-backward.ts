#!/usr/bin/env npx tsx
/**
 * Test embedding layer backward (uses gather).
 * The embedding table is large (50257 x 768) which could hit dispatch limits.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Embedding Backward Test');
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

  // Dimensions (GPT-2)
  const vocabSize = 50257;
  const embedDim = 768;
  const seqLen = 32;
  const batchSize = 1;

  // Create embedding weight (trainable in this test)
  console.log('\nCreating embedding weight...');
  const embedData = new Float32Array(vocabSize * embedDim);
  for (let i = 0; i < embedData.length; i++) {
    embedData[i] = (Math.random() * 2 - 1) * 0.02;
  }
  const embedWeight = api.tensorFromArray(embedData, [vocabSize, embedDim], {
    device: 'webgpu',
    requiresGrad: true, // Trainable for this test
  });
  console.log(`Embedding weight shape: [${embedWeight.shape}]`);
  console.log(`Embedding weight size: ${vocabSize * embedDim} elements`);

  // Create input indices
  const indicesData = new Float32Array(batchSize * seqLen);
  for (let i = 0; i < indicesData.length; i++) {
    indicesData[i] = Math.floor(Math.random() * vocabSize);
  }
  const indices = api.tensorFromArray(indicesData, [batchSize * seqLen]);
  console.log(`Indices shape: [${indices.shape}]`);

  // Embedding lookup (like GPT-2's wte)
  // Expand indices to [N, embedDim] for gather
  console.log('\nRunning embedding lookup...');
  const numElements = batchSize * seqLen;
  const expandedIndices = indices
    .reshape([numElements, 1])
    .expand([numElements, embedDim])
    .contiguous();

  // Gather
  const embedded = embedWeight.gather(expandedIndices, { dim: 0 });
  console.log(`Embedded shape: [${embedded.shape}]`);

  // Reshape to [batchSize, seqLen, embedDim]
  const embeddedReshaped = embedded.reshape([batchSize, seqLen, embedDim]);

  // Check output
  const embeddedSum = await embeddedReshaped.sum().item();
  console.log(`Embedded sum: ${embeddedSum.toFixed(4)}`);

  if (Number.isNaN(embeddedSum)) {
    console.log('ERROR: Embedded sum is NaN!');
    return;
  }

  // Simple loss
  const loss = embeddedReshaped.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toExponential(4)}`);

  // Backward
  console.log('\nRunning backward...');
  try {
    await loss.backward();
    console.log('Backward complete');
  } catch (e) {
    console.error('ERROR in backward:', e);
    return;
  }

  // Check embedding gradient
  if (embedWeight.grad) {
    const gradSum = await embedWeight.grad.sum().item();
    console.log(`\nembedWeight.grad sum: ${gradSum.toExponential(4)}`);
    if (Number.isNaN(gradSum)) {
      console.log('ERROR: embedWeight.grad is NaN!');
    }
  } else {
    console.log('embedWeight.grad is null');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch((e) => {
  console.error('\nFATAL ERROR:', e);
  process.exit(1);
});
