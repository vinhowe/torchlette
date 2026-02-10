#!/usr/bin/env npx tsx
/**
 * Test: Swap h.0 and h.1 weights to see if NaN follows weights or position.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { GPT2WithLoRA, GPT2_SMALL_CONFIG } from './src/lib/torchlette/gpt2-lora';
import { createLoRAConfig } from './src/lib/torchlette/lora';

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

async function runTest(name: string, swappedWeights: Map<string, { data: Float32Array; shape: number[] }>): Promise<boolean> {
  console.log(`\n${'='.repeat(60)}`);
  console.log(name);
  console.log('='.repeat(60));

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const miniConfig = { ...GPT2_SMALL_CONFIG, numLayers: 2 };
  const loraConfig = createLoRAConfig(8, 16);
  const model = new GPT2WithLoRA(api, miniConfig, loraConfig, 'webgpu');

  model.loadBaseWeights(swappedWeights);
  model.train(true);

  const seqLen = 32;
  const inputData = new Float32Array(seqLen);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.floor(Math.random() * miniConfig.vocabSize);
  }
  const input = api.tensorFromArray(inputData, [1, seqLen], { device: 'webgpu' });

  const targetData = new Float32Array(seqLen);
  for (let i = 0; i < targetData.length; i++) {
    targetData[i] = Math.floor(Math.random() * miniConfig.vocabSize);
  }
  const target = api.tensorFromArray(targetData, [1, seqLen], { device: 'webgpu' });

  const { loss } = model.forwardWithLoss(input, target);
  const lossVal = await loss.item();
  console.log(`Loss: ${lossVal.toFixed(4)}`);

  await loss.backward();

  const loraParams = model.getLoRAParameters();
  let hasNaN = false;
  for (let i = 0; i < loraParams.length; i++) {
    const param = loraParams[i];
    const paramType = i % 2 === 0 ? 'loraA' : 'loraB';
    const blockIdx = Math.floor(i / 2);

    if (param.grad) {
      const gradSum = await param.grad.sum().item();
      const status = Number.isNaN(gradSum) ? 'NaN <<<' : gradSum.toExponential(4);
      console.log(`  Block ${blockIdx} ${paramType}: ${status}`);
      if (Number.isNaN(gradSum)) hasNaN = true;
    }
  }

  return hasNaN;
}

async function main(): Promise<void> {
  console.log('Loading weights...');
  await initWebGPU();

  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const originalWeights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  // Test 1: Original order (h.0 → block0, h.1 → block1)
  const hasNaN1 = await runTest('Test 1: Original (h.0 → block0, h.1 → block1)', originalWeights);

  // Test 2: Swap h.0 and h.1 weights
  const swappedWeights = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [name, data] of originalWeights.entries()) {
    let newName = name;
    if (name.startsWith('h.0.')) {
      newName = 'h.1.' + name.slice(4);
    } else if (name.startsWith('h.1.')) {
      newName = 'h.0.' + name.slice(4);
    }
    swappedWeights.set(newName, data);
  }

  const hasNaN2 = await runTest('Test 2: Swapped (h.1 → block0, h.0 → block1)', swappedWeights);

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(`Test 1 (original): ${hasNaN1 ? 'NaN in block 0' : 'OK'}`);
  console.log(`Test 2 (swapped):  ${hasNaN2 ? 'NaN in block 0' : 'OK'}`);

  if (hasNaN1 && hasNaN2) {
    console.log('\nConclusion: NaN follows POSITION (always block 0), not weights');
  } else if (hasNaN1 && !hasNaN2) {
    console.log('\nConclusion: NaN follows h.0 WEIGHTS specifically');
  } else if (!hasNaN1 && hasNaN2) {
    console.log('\nConclusion: NaN follows h.1 WEIGHTS (now in block 0)');
  } else {
    console.log('\nConclusion: No NaN in either test');
  }
}

main().catch(console.error);
