#!/usr/bin/env npx tsx
/**
 * Minimal test to verify loss decreases during LoRA-style training.
 */

import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { Adam } from '../../src/optim/adam';

async function main(): Promise<void> {
  console.log('Testing loss trend with LoRA-style training...');
  await initWebGPU();

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Simulate a small transformer layer with LoRA
  // Base weight: frozen [64, 64]
  // LoRA: A [8, 64], B [64, 8] - trainable
  const hiddenSize = 64;
  const loraRank = 8;
  const seqLen = 16;
  const vocabSize = 100;

  // Create random "embedding" input
  const inputData = new Float32Array(seqLen * hiddenSize);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = (Math.random() - 0.5) * 0.1;
  }
  const x = api.tensorFromArray(inputData, [1, seqLen, hiddenSize], { device: 'webgpu' });

  // Create random target (for next-token prediction simulation)
  const targetData = new Float32Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    targetData[i] = Math.floor(Math.random() * vocabSize);
  }

  // Base weight (frozen)
  const baseWeightData = new Float32Array(hiddenSize * hiddenSize);
  for (let i = 0; i < baseWeightData.length; i++) {
    baseWeightData[i] = (Math.random() - 0.5) * 0.02;
  }
  const baseWeight = api.tensorFromArray(baseWeightData, [hiddenSize, hiddenSize], { device: 'webgpu' });

  // Output projection (frozen)
  const outProjData = new Float32Array(vocabSize * hiddenSize);
  for (let i = 0; i < outProjData.length; i++) {
    outProjData[i] = (Math.random() - 0.5) * 0.02;
  }
  const outProj = api.tensorFromArray(outProjData, [vocabSize, hiddenSize], { device: 'webgpu' });

  // LoRA matrices (trainable)
  const loraAData = new Float32Array(loraRank * hiddenSize);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() - 0.5) * 0.01;
  }
  const loraA = api.tensorFromArray(loraAData, [loraRank, hiddenSize], { device: 'webgpu', requiresGrad: true });

  // LoRA B initialized to zeros (typical LoRA init)
  const loraB = api.tensorFromArray(new Float32Array(hiddenSize * loraRank), [hiddenSize, loraRank], { device: 'webgpu', requiresGrad: true });

  // Optimizer
  const optimizer = new Adam([loraA, loraB], { lr: 0.01 }, api);

  console.log('\nTraining for 10 steps on fixed data:');
  const losses: number[] = [];

  for (let step = 0; step < 10; step++) {
    // Forward: h = x @ (W + A^T @ B^T)
    // Simplified: h = x @ W^T + x @ A^T @ B^T
    const baseOut = api.matmul(x, baseWeight.transpose({ dim0: 0, dim1: 1 }));
    const loraOut = api.matmul(
      api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 })),
      loraB.transpose({ dim0: 0, dim1: 1 })
    );
    const h = api.add(baseOut, loraOut); // [1, seqLen, hiddenSize]

    // Simple "logits" - project to vocab
    const logits = api.matmul(h, outProj.transpose({ dim0: 0, dim1: 1 })); // [1, seqLen, vocabSize]

    // Simple MSE loss (not cross-entropy for simplicity)
    // Just minimize sum of logits to check gradient flow
    const loss = logits.sum();
    const lossVal = await loss.item();
    losses.push(lossVal);

    console.log(`Step ${step + 1}: loss = ${lossVal.toFixed(4)}`);

    // Backward
    await loss.backward();

    // Check gradients
    const aGrad = loraA.grad ? await loraA.grad.sum().item() : 0;
    const bGrad = loraB.grad ? await loraB.grad.sum().item() : 0;
    if (step === 0) {
      console.log(`  loraA.grad sum = ${aGrad.toFixed(6)}, loraB.grad sum = ${bGrad.toFixed(6)}`);
    }

    // Optimizer step
    optimizer.step();
    optimizer.zeroGrad();

    // Mark step to free memory
    await api.markStep();
  }

  // Analyze trend
  const firstLoss = losses[0];
  const lastLoss = losses[losses.length - 1];
  const decreased = lastLoss < firstLoss;

  console.log('\n' + '='.repeat(50));
  console.log(`First loss: ${firstLoss.toFixed(4)}`);
  console.log(`Last loss:  ${lastLoss.toFixed(4)}`);
  console.log(`Change:     ${(lastLoss - firstLoss).toFixed(4)} (${((lastLoss - firstLoss) / firstLoss * 100).toFixed(2)}%)`);
  console.log(`Loss ${decreased ? 'DECREASED ✓' : 'DID NOT DECREASE ✗'}`);
  console.log('='.repeat(50));
}

main().catch(console.error);
