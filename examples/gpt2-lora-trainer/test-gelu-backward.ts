#!/usr/bin/env npx tsx
/**
 * Test GELU backward specifically with GPT-2 values.
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

  const embedDim = 768;
  const seqLen = 32;

  console.log(`\n${'='.repeat(60)}`);
  console.log('GELU Backward Test');
  console.log('='.repeat(60));

  // Test 1: Simple GELU backward
  console.log('\nTest 1: Simple GELU with small values');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1) * 0.5; // Small values
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const y = x.gelu();
    const loss = y.mean() as Tensor;
    await loss.backward();

    await checkStats('x', x);
    await checkStats('y = gelu(x)', y);
    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 2: GELU with GPT-2 MLP weights
  console.log('\nTest 2: GELU with GPT-2 MLP fc layer');
  {
    // Load MLP weights
    const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
    const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
    const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);

    const fcWeight = api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });
    const fcBias = api.tensorFromArray(cFcB.data, cFcB.shape, { device: 'webgpu', requiresGrad: false });

    // Create input similar to layer norm output
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1); // Normalized values
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    // fc -> gelu
    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    await checkStats('fc output', fc);

    const gelu = fc.gelu();
    await checkStats('gelu(fc)', gelu);

    const loss = gelu.mean() as Tensor;
    const lossVal = await loss.item();
    console.log(`Loss: ${lossVal.toFixed(6)}`);

    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 3: Full MLP forward and backward
  console.log('\nTest 3: Full MLP (fc -> gelu -> proj)');
  {
    // Load MLP weights
    const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
    const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
    const cProjW = weights.get('h.0.mlp.c_proj.weight')!;
    const cProjB = weights.get('h.0.mlp.c_proj.bias')!;

    const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);
    const cProjWTransposed = transposeWeight(cProjW.data, cProjW.shape);

    const fcWeight = api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });
    const fcBias = api.tensorFromArray(cFcB.data, cFcB.shape, { device: 'webgpu', requiresGrad: false });
    const projWeight = api.tensorFromArray(cProjWTransposed.data, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });
    const projBias = api.tensorFromArray(cProjB.data, cProjB.shape, { device: 'webgpu', requiresGrad: false });

    // Input
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    // MLP forward
    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    await checkStats('fc output', fc);

    const gelu = fc.gelu();
    await checkStats('gelu(fc)', gelu);

    const proj = api.add(api.matmul(gelu, projWeight.transpose({ dim0: 0, dim1: 1 })), projBias);
    await checkStats('proj(gelu)', proj);

    const loss = proj.mean() as Tensor;
    const lossVal = await loss.item();
    console.log(`Loss: ${lossVal.toFixed(6)}`);

    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 4: MLP with residual (like in GPT-2 block)
  console.log('\nTest 4: MLP with residual connection');
  {
    // Load MLP weights
    const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
    const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
    const cProjW = weights.get('h.0.mlp.c_proj.weight')!;
    const cProjB = weights.get('h.0.mlp.c_proj.bias')!;

    const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);
    const cProjWTransposed = transposeWeight(cProjW.data, cProjW.shape);

    const fcWeight = api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });
    const fcBias = api.tensorFromArray(cFcB.data, cFcB.shape, { device: 'webgpu', requiresGrad: false });
    const projWeight = api.tensorFromArray(cProjWTransposed.data, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });
    const projBias = api.tensorFromArray(cProjB.data, cProjB.shape, { device: 'webgpu', requiresGrad: false });

    // Input
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    // MLP forward
    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, projWeight.transpose({ dim0: 0, dim1: 1 })), projBias);

    // Residual
    const h = api.add(x, proj);
    await checkStats('h = x + proj', h);

    const loss = h.mean() as Tensor;
    const lossVal = await loss.item();
    console.log(`Loss: ${lossVal.toFixed(6)}`);

    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
