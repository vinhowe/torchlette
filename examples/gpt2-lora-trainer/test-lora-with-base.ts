#!/usr/bin/env npx tsx
/**
 * Test LoRA with base weights (detached) to see if that causes NaN.
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

// Transpose 2D weight from [in, out] to [out, in]
function transposeWeight(data: Float32Array, shape: number[]): { data: Float32Array; shape: number[] } {
  const [rows, cols] = shape;
  const transposed = new Float32Array(data.length);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      transposed[j * rows + i] = data[i * cols + j];
    }
  }
  return { data: transposed, shape: [cols, rows] };
}

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('LoRA with Base Weights Test');
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
  const rank = 8;
  const scaling = 2.0;

  // Load real embedding and first layer attention weights
  const wteData = weights.get('wte.weight')!;
  const wte = api.tensorFromArray(wteData.data, wteData.shape, { device: 'webgpu', requiresGrad: false });

  // Load h.0.attn.c_attn weights (QKV projection)
  const cAttnData = weights.get('h.0.attn.c_attn.weight')!;
  const cAttnBiasData = weights.get('h.0.attn.c_attn.bias')!;

  // Transpose from [768, 2304] to [2304, 768]
  const transposed = transposeWeight(cAttnData.data, cAttnData.shape);
  const baseWeight = api.tensorFromArray(transposed.data, transposed.shape, { device: 'webgpu', requiresGrad: false });
  const baseBias = api.tensorFromArray(cAttnBiasData.data, cAttnBiasData.shape, { device: 'webgpu', requiresGrad: false });

  console.log(`Base weight shape: [${baseWeight.shape}]`);
  console.log(`Base bias shape: [${baseBias.shape}]`);

  // Create LoRA parameters
  const loraAData = new Float32Array(rank * embedDim);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * 0.1;
  }
  const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
  const loraB = api.zeros([2304, rank], { device: 'webgpu', requiresGrad: true });

  console.log(`loraA shape: [${loraA.shape}]`);
  console.log(`loraB sum: ${await loraB.sum().item()} (should be 0)`);

  // Create input via embedding
  const inputTokens = new Float32Array(seqLen);
  for (let i = 0; i < inputTokens.length; i++) {
    inputTokens[i] = Math.floor(Math.random() * 50257);
  }
  const input = api.tensorFromArray(inputTokens, [seqLen]);
  const expandedInput = input.reshape([seqLen, 1]).expand([seqLen, embedDim]).contiguous();
  const x = wte.gather(expandedInput, { dim: 0 }); // [seqLen, embedDim]

  console.log(`\nInput x shape: [${x.shape}]`);
  const xSum = await x.sum().item();
  console.log(`Input x sum: ${xSum.toFixed(4)}`);

  // Full LoRA linear forward: baseOut + scaling * loraOut
  console.log('\nRunning LoRA linear forward...');

  // Base path: x @ baseWeight.T + baseBias
  const baseOut = api.add(
    api.matmul(x, baseWeight.transpose({ dim0: 0, dim1: 1 })),
    baseBias
  );
  const baseOutSum = await baseOut.sum().item();
  console.log(`Base output sum: ${baseOutSum.toFixed(4)}`);

  // Detach base output
  const detachedBase = baseOut.detach();

  // LoRA path: scaling * (x @ loraA.T @ loraB.T)
  const xA = api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 }));
  const loraOut = api.matmul(xA, loraB.transpose({ dim0: 0, dim1: 1 }));
  const scalingTensor = api.tensorFromArray([scaling], []);
  const scaledLora = api.mul(loraOut, scalingTensor);

  // Combine
  const output = api.add(detachedBase, scaledLora);
  console.log(`Output shape: [${output.shape}]`);

  const outputSum = await output.sum().item();
  console.log(`Output sum: ${outputSum.toFixed(4)}`);

  // Simple loss
  const loss = output.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(6)}`);

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
