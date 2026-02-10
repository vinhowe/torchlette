#!/usr/bin/env npx tsx
/**
 * Test: Single-layer model with h.0 vs h.1 weights.
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

async function runSingleLayerTest(
  name: string,
  originalWeights: Map<string, { data: Float32Array; shape: number[] }>,
  sourceBlock: number
): Promise<boolean> {
  console.log(`\n${'='.repeat(60)}`);
  console.log(name);
  console.log('='.repeat(60));

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Single layer model
  const miniConfig = { ...GPT2_SMALL_CONFIG, numLayers: 1 };
  const loraConfig = createLoRAConfig(8, 16);
  const model = new GPT2WithLoRA(api, miniConfig, loraConfig, 'webgpu');

  // Create weights mapping h.{sourceBlock}.* to h.0.*
  const mappedWeights = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [name, data] of originalWeights.entries()) {
    if (name.startsWith(`h.${sourceBlock}.`)) {
      // Map to h.0.*
      const newName = 'h.0.' + name.slice(`h.${sourceBlock}.`.length);
      mappedWeights.set(newName, data);
    } else if (!name.startsWith('h.')) {
      // Copy non-layer weights as-is
      mappedWeights.set(name, data);
    }
  }

  model.loadBaseWeights(mappedWeights);
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

    if (param.grad) {
      const gradSum = await param.grad.sum().item();
      const status = Number.isNaN(gradSum) ? 'NaN <<<' : gradSum.toExponential(4);
      console.log(`  ${paramType}: ${status}`);
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

  // Test with h.0 weights
  const hasNaN_h0 = await runSingleLayerTest('Single layer using h.0 weights', originalWeights, 0);

  // Test with h.1 weights
  const hasNaN_h1 = await runSingleLayerTest('Single layer using h.1 weights', originalWeights, 1);

  // Test with h.2 weights
  const hasNaN_h2 = await runSingleLayerTest('Single layer using h.2 weights', originalWeights, 2);

  // Test with h.11 weights (last layer)
  const hasNaN_h11 = await runSingleLayerTest('Single layer using h.11 weights', originalWeights, 11);

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(`h.0 weights: ${hasNaN_h0 ? 'NaN' : 'OK'}`);
  console.log(`h.1 weights: ${hasNaN_h1 ? 'NaN' : 'OK'}`);
  console.log(`h.2 weights: ${hasNaN_h2 ? 'NaN' : 'OK'}`);
  console.log(`h.11 weights: ${hasNaN_h11 ? 'NaN' : 'OK'}`);
}

main().catch(console.error);
