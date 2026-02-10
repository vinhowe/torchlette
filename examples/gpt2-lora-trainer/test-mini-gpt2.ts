#!/usr/bin/env npx tsx
/**
 * Minimal GPT-2 test - 2 blocks + embeddings + logits.
 * Try to reproduce the NaN in block 0's LoRA.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Torchlette, type FrontendTensor as Tensor, Adam } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { GPT2WithLoRA, GPT2_SMALL_CONFIG } from './src/lib/torchlette/gpt2-lora';
import { createLoRAConfig } from './src/lib/torchlette/lora';

// Cache directory for weights
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
    switch (tensorInfo.dtype) {
      case 'F32': {
        const alignedBuffer = new ArrayBuffer(tensorData.length);
        new Uint8Array(alignedBuffer).set(tensorData);
        float32Data = new Float32Array(alignedBuffer);
        break;
      }
      case 'F16':
        float32Data = convertFloat16ToFloat32(tensorData);
        break;
      default:
        continue;
    }

    weights.set(name, { data: float32Data, shape: tensorInfo.shape });
  }

  return weights;
}

function convertFloat16ToFloat32(data: Uint8Array): Float32Array {
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const float16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(float16View.length);

  for (let i = 0; i < float16View.length; i++) {
    const h = float16View[i];
    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7c00) >> 10;
    const fraction = h & 0x03ff;

    if (exponent === 0) {
      float32[i] = fraction === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 0x1f) {
      float32[i] = fraction === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      float32[i] = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }

  return float32;
}

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Mini GPT-2 Test (2 blocks)');
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

  // Create mini GPT-2 config (only 1 layer - just block 0)
  const miniConfig = {
    ...GPT2_SMALL_CONFIG,
    numLayers: 1, // Just 1 layer to isolate block 0
  };

  // Create model
  console.log('\nCreating mini GPT-2 with LoRA...');
  const loraConfig = createLoRAConfig(8, 16);
  const model = new GPT2WithLoRA(api, miniConfig, loraConfig, 'webgpu');
  console.log(`Model created with ${miniConfig.numLayers} layers`);

  // Load REAL GPT-2 weights from cache
  console.log('\nLoading real GPT-2 weights from cache...');
  if (!fs.existsSync(WEIGHTS_CACHE)) {
    console.log('ERROR: No cached weights found. Run test-lora-trainer.ts first.');
    return;
  }
  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
  console.log(`Loaded ${weights.size} tensors from cache`);

  // Load base weights into model (only for the 2 layers we have)
  model.loadBaseWeights(weights);
  console.log('Base weights loaded');


  // Set model to training mode
  model.train(true);

  // Get LoRA parameters
  const loraParams = model.getLoRAParameters();
  console.log(`\nLoRA parameters: ${loraParams.length} tensors`);

  // Verify loraB values are zero
  for (let i = 0; i < loraParams.length; i += 2) {
    const loraB = loraParams[i + 1];
    const loraBSum = await loraB.sum().item();
    console.log(`Block ${i / 2} loraB sum: ${loraBSum} (should be 0)`);
  }

  // Create input
  const seqLen = 32;
  const batchSize = 1;
  const inputData = new Float32Array(batchSize * seqLen);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.floor(Math.random() * miniConfig.vocabSize);
  }
  const input = api.tensorFromArray(inputData, [batchSize, seqLen], { device: 'webgpu' });

  const targetData = new Float32Array(batchSize * seqLen);
  for (let i = 0; i < targetData.length; i++) {
    targetData[i] = Math.floor(Math.random() * miniConfig.vocabSize);
  }
  const target = api.tensorFromArray(targetData, [batchSize, seqLen], { device: 'webgpu' });

  console.log(`\nInput shape: [${input.shape}]`);
  console.log(`Target shape: [${target.shape}]`);

  // Forward
  console.log('\nRunning forward...');
  const { logits, loss } = model.forwardWithLoss(input, target);
  console.log(`Logits shape: [${logits.shape}]`);

  const lossVal = await loss.item();
  console.log(`Loss: ${lossVal.toFixed(4)}`);

  if (Number.isNaN(lossVal)) {
    console.log('ERROR: Loss is NaN before backward!');
    return;
  }

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check LoRA gradients
  console.log('\nChecking LoRA gradients:');
  for (let i = 0; i < loraParams.length; i++) {
    const param = loraParams[i];
    const paramType = i % 2 === 0 ? 'loraA' : 'loraB';
    const blockIdx = Math.floor(i / 2);

    if (param.grad) {
      const gradSum = await param.grad.sum().item();
      const status = Number.isNaN(gradSum) ? 'NaN <<<' : gradSum.toExponential(4);
      console.log(`  Block ${blockIdx} ${paramType}: ${status}`);
    } else {
      console.log(`  Block ${blockIdx} ${paramType}: no grad`);
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
