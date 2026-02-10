#!/usr/bin/env npx tsx
/**
 * Analyze fc output statistics to understand NaN issue.
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

async function stats(name: string, t: Tensor): Promise<void> {
  const sum = await t.sum().item();
  const max = await (t.max() as Tensor).item();
  const min = await (t.neg().max() as Tensor).item(); // -max(-x) = min(x)
  const numel = t.shape.reduce((a, b) => a * b, 1);
  const mean = sum / numel;

  const sumSq = await t.mul(t).sum().item();
  const variance = sumSq / numel - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  console.log(`${name}: min=${(-min).toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
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

  const seqLen = 32;
  const embedDim = 768;

  // Load REAL c_fc weights
  const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
  const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
  const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);

  const realFcWeight = api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });
  const realFcBias = api.tensorFromArray(cFcB.data, cFcB.shape, { device: 'webgpu', requiresGrad: false });

  console.log(`\n${'='.repeat(60)}`);
  console.log('FC Statistics Analysis');
  console.log('='.repeat(60));

  // Create input
  const xData = new Float32Array(seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1);
  }
  const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

  console.log('\n--- Forward pass stats ---');
  await stats('Input x', x);

  // FC output
  const fc = api.add(api.matmul(x, realFcWeight.transpose({ dim0: 0, dim1: 1 })), realFcBias);
  await stats('FC output', fc);

  // GELU output
  const gelu = fc.gelu();
  await stats('GELU output', gelu);

  // Proj (just use identity for now to isolate fc+gelu)
  // Let's use a simple reduction instead
  const loss = gelu.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`Loss (mean of gelu): ${lossVal.toFixed(6)}`);

  // Backward - focus on just fc + gelu
  console.log('\n--- Backward pass (gelu.mean) ---');
  await loss.backward();

  if (x.grad) {
    await stats('x.grad', x.grad);
  }

  // Now test with proj added
  console.log('\n--- With proj layer ---');
  {
    const cProjW = weights.get('h.0.mlp.c_proj.weight')!;
    const cProjB = weights.get('h.0.mlp.c_proj.bias')!;
    const cProjWTransposed = transposeWeight(cProjW.data, cProjW.shape);

    const projWeight = api.tensorFromArray(cProjWTransposed.data, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });
    const projBias = api.tensorFromArray(cProjB.data, cProjB.shape, { device: 'webgpu', requiresGrad: false });

    // Fresh x
    const x2 = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc2 = api.add(api.matmul(x2, realFcWeight.transpose({ dim0: 0, dim1: 1 })), realFcBias);
    await stats('FC output', fc2);

    const gelu2 = fc2.gelu();
    await stats('GELU output', gelu2);

    const proj = api.add(api.matmul(gelu2, projWeight.transpose({ dim0: 0, dim1: 1 })), projBias);
    await stats('Proj output', proj);

    const loss2 = proj.mean() as Tensor;
    const lossVal2 = await loss2.item();
    console.log(`Loss: ${lossVal2.toFixed(6)}`);

    await loss2.backward();

    if (x2.grad) {
      const gradSum = await x2.grad.sum().item();
      console.log(`x2.grad sum: ${gradSum.toExponential(3)}${Number.isNaN(gradSum) ? ' [NaN!]' : ''}`);
    }
  }

  // Check weights stats
  console.log('\n--- Weight stats ---');
  await stats('c_fc.weight', realFcWeight);
  await stats('c_fc.bias', realFcBias);

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
