#!/usr/bin/env npx tsx
/**
 * Test ONLY the LoRA path (no base weights) to isolate NaN issue.
 * Uses real GPT-2 attention structure but LoRA-only forward.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';

const CACHE_DIR = path.join(process.cwd(), '.cache', 'gpt2-lora-test');
const WEIGHTS_CACHE = path.join(CACHE_DIR, 'model.safetensors');

function parseSafetensors(buffer: ArrayBuffer): Map<string, { data: Float32Array; shape: number[] }> {
  const view = new DataView(buffer);
  const headerLength = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLength);
  const headerText = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerText);

  const dataOffset = 8 + headerLength;
  const weights = new Map<string, { data: Float32Array; shape: number[] }>();

  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const tensorInfo = info as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const [startOffset, endOffset] = tensorInfo.data_offsets;
    const tensorData = new Uint8Array(buffer, dataOffset + startOffset, endOffset - startOffset);

    let float32Data: Float32Array;
    if (tensorInfo.dtype === 'F32') {
      const alignedBuffer = new ArrayBuffer(tensorData.length);
      new Uint8Array(alignedBuffer).set(tensorData);
      float32Data = new Float32Array(alignedBuffer);
    } else {
      continue;
    }
    weights.set(name, { data: float32Data, shape: tensorInfo.shape });
  }

  return weights;
}

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('LoRA-Only Path Test');
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

  // Load real GPT-2 embedding weights
  console.log('\nLoading real GPT-2 weights...');
  if (!fs.existsSync(WEIGHTS_CACHE)) {
    console.log('ERROR: No cached weights found');
    return;
  }
  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  // Dimensions
  const embedDim = 768;
  const seqLen = 32;
  const batchSize = 1;
  const rank = 8;
  const scaling = 2.0;

  // Create frozen embedding weight
  const wteData = weights.get('wte.weight')!;
  const wte = api.tensorFromArray(wteData.data, wteData.shape, { device: 'webgpu', requiresGrad: false });
  console.log(`Embedding shape: [${wte.shape}]`);

  // Create LoRA parameters
  const loraAData = new Float32Array(rank * embedDim);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * 0.1;
  }
  const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
  const loraB = api.zeros([2304, rank], { device: 'webgpu', requiresGrad: true });

  console.log(`loraA shape: [${loraA.shape}]`);
  console.log(`loraB shape: [${loraB.shape}]`);
  console.log(`loraB sum: ${await loraB.sum().item()} (should be 0)`);

  // Create input tokens
  const inputTokens = new Float32Array(batchSize * seqLen);
  for (let i = 0; i < inputTokens.length; i++) {
    inputTokens[i] = Math.floor(Math.random() * 50257);
  }
  const input = api.tensorFromArray(inputTokens, [batchSize * seqLen]);

  // Embedding lookup (like GPT-2)
  console.log('\nRunning embedding lookup...');
  const expandedInput = input.reshape([batchSize * seqLen, 1]).expand([batchSize * seqLen, embedDim]).contiguous();
  const x = wte.gather(expandedInput, { dim: 0 }); // [batchSize * seqLen, embedDim]
  console.log(`Embedded input shape: [${x.shape}]`);

  const xSum = await x.sum().item();
  console.log(`Embedded input sum: ${xSum.toFixed(4)}`);

  // LoRA forward: scaling * (x @ loraA.T @ loraB.T)
  console.log('\nRunning LoRA forward...');
  const xA = api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 })); // [N, rank]
  const loraOut = api.matmul(xA, loraB.transpose({ dim0: 0, dim1: 1 })); // [N, 2304]
  const scalingTensor = api.tensorFromArray([scaling], []);
  const output = api.mul(loraOut, scalingTensor);

  console.log(`xA shape: [${xA.shape}]`);
  console.log(`loraOut shape: [${loraOut.shape}]`);
  console.log(`output shape: [${output.shape}]`);

  const outputSum = await output.sum().item();
  console.log(`Output sum: ${outputSum} (should be ~0)`);

  // Simple loss
  const loss = output.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check gradients
  if (loraA.grad) {
    const gradSum = await loraA.grad.sum().item();
    console.log(`\nloraA.grad sum: ${gradSum}`);
    if (Number.isNaN(gradSum)) console.log('ERROR: loraA.grad is NaN!');
  }

  if (loraB.grad) {
    const gradSum = await loraB.grad.sum().item();
    console.log(`loraB.grad sum: ${gradSum}`);
    if (Number.isNaN(gradSum)) console.log('ERROR: loraB.grad is NaN!');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch(console.error);
