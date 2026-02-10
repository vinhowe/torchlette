#!/usr/bin/env npx tsx
/**
 * Test layer norm backward.
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

// Simple layer norm implementation
function layerNorm(api: Torchlette, x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const inputShape = x.shape;
  const meanResult = api.mean(x, { dim: -1 }) as Tensor;
  const meanShape = [...inputShape.slice(0, -1), 1];
  const mean = meanResult.reshape(meanShape);

  const xCentered = api.sub(x, mean);
  const varianceResult = api.mean(api.mul(xCentered, xCentered), { dim: -1 }) as Tensor;
  const variance = varianceResult.reshape(meanShape);

  const epsTensor = api.tensorFromArray([eps], []);
  const std = api.sqrt(api.add(variance, epsTensor));
  const normalized = api.div(xCentered, std);

  return api.add(api.mul(normalized, weight), bias);
}

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Layer Norm Backward Test');
  console.log('='.repeat(60));

  const ok = await initWebGPU();
  if (!ok) throw new Error('Failed to initialize WebGPU');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Load real GPT-2 weights
  console.log('\nLoading real GPT-2 weights...');
  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  // Dimensions
  const embedDim = 768;
  const seqLen = 32;
  const batch = 1;

  // Load layer norm weights
  const ln1WeightData = weights.get('h.0.ln_1.weight')!;
  const ln1BiasData = weights.get('h.0.ln_1.bias')!;
  const lnWeight = api.tensorFromArray(ln1WeightData.data, ln1WeightData.shape, { device: 'webgpu', requiresGrad: false });
  const lnBias = api.tensorFromArray(ln1BiasData.data, ln1BiasData.shape, { device: 'webgpu', requiresGrad: false });

  console.log(`LN weight shape: [${lnWeight.shape}]`);

  // Check weight stats
  const weightSum = await lnWeight.sum().item();
  console.log(`LN weight sum: ${weightSum.toFixed(4)}`);

  // Load embedding
  const wteData = weights.get('wte.weight')!;
  const wte = api.tensorFromArray(wteData.data, wteData.shape, { device: 'webgpu', requiresGrad: false });

  // Create input via embedding (with gradient)
  const inputTokens = new Float32Array(seqLen);
  for (let i = 0; i < inputTokens.length; i++) {
    inputTokens[i] = Math.floor(Math.random() * 50257);
  }
  const input = api.tensorFromArray(inputTokens, [seqLen]);
  const expandedInput = input.reshape([seqLen, 1]).expand([seqLen, embedDim]).contiguous();
  const embedded = wte.gather(expandedInput, { dim: 0 });

  // Create trainable tensor from random data (simulating embedded input)
  const xData = new Float32Array(batch * seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1) * 0.5; // Similar scale to embeddings
  }
  const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

  console.log(`\nInput x shape: [${x.shape}]`);
  const xSum = await x.sum().item();
  console.log(`Input x sum: ${xSum.toFixed(4)}`);

  // Apply layer norm
  console.log('\nApplying layer norm...');
  const normed = layerNorm(api, x, lnWeight, lnBias);

  const normedSum = await normed.sum().item();
  console.log(`Normed output sum: ${normedSum.toFixed(4)}`);

  if (Number.isNaN(normedSum)) {
    console.log('ERROR: Layer norm output is NaN!');
    return;
  }

  // Simple loss
  const loss = normed.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(6)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check gradient
  if (x.grad) {
    const gradSum = await x.grad.sum().item();
    console.log(`\nx.grad sum: ${gradSum.toExponential(4)}`);
    if (Number.isNaN(gradSum)) console.log('ERROR: x.grad is NaN!');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch(console.error);
