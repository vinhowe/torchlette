#!/usr/bin/env npx tsx
/**
 * Test the actual GPT2WithLoRA class with detailed debugging.
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

async function checkStats(name: string, t: Tensor): Promise<boolean> {
  const sum = await t.sum().item();
  const hasNaN = Number.isNaN(sum);
  console.log(`${name}: sum=${sum.toExponential(3)}${hasNaN ? ' [NaN!]' : ''}`);
  return hasNaN;
}

async function main(): Promise<void> {
  console.log('Loading weights...');
  await initWebGPU();

  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Single layer model
  const miniConfig = { ...GPT2_SMALL_CONFIG, numLayers: 1 };
  const loraConfig = createLoRAConfig(8, 16);
  const model = new GPT2WithLoRA(api, miniConfig, loraConfig, 'webgpu');

  // Load weights
  model.loadBaseWeights(weights);
  model.train(true);

  const seqLen = 32;

  // Create input
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

  console.log(`\n${'='.repeat(60)}`);
  console.log('GPT2WithLoRA Debug');
  console.log('='.repeat(60));

  // Check model weights
  console.log('\nChecking model weights...');
  await checkStats('wte.weight', model.wte.weight);
  await checkStats('wpe.weight', model.wpe.weight);
  await checkStats('h.0.ln1.weight', model.h[0].ln1.weight);
  await checkStats('h.0.attn.cAttn.baseWeight', model.h[0].attn.cAttn.baseWeight);
  await checkStats('h.0.attn.cAttn.loraA', model.h[0].attn.cAttn.loraA);
  await checkStats('h.0.attn.cAttn.loraB', model.h[0].attn.cAttn.loraB);

  // Forward with loss
  console.log('\nRunning forward...');
  const { logits, loss } = model.forwardWithLoss(input, target);

  await checkStats('logits', logits);

  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(4)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();

  // Check gradients
  const loraParams = model.getLoRAParameters();
  console.log('\nLoRA gradients:');
  for (let i = 0; i < loraParams.length; i++) {
    const param = loraParams[i];
    const paramType = i % 2 === 0 ? 'loraA' : 'loraB';
    if (param.grad) {
      await checkStats(`  ${paramType}.grad`, param.grad);
    }
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
