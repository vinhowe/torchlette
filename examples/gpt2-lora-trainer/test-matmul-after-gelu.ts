#!/usr/bin/env npx tsx
/**
 * Test matmul backward after GELU with different weight configurations.
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

  const seqLen = 32;
  const innerDim = 3072; // 4 * 768
  const embedDim = 768;

  // Load real c_proj weight
  const cProjW = weights.get('h.0.mlp.c_proj.weight')!;
  const cProjWTransposed = transposeWeight(cProjW.data, cProjW.shape);
  const realProjWeight = api.tensorFromArray(cProjWTransposed.data, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });

  console.log(`\n${'='.repeat(60)}`);
  console.log('Matmul After GELU Investigation');
  console.log('='.repeat(60));

  // Test 1: GELU → matmul with REAL proj weight
  console.log('\nTest 1: GELU → matmul with REAL proj weight');
  {
    const xData = new Float32Array(seqLen * innerDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1) * 5; // Similar scale to fc output
    }
    const x = api.tensorFromArray(xData, [1, seqLen, innerDim], { device: 'webgpu', requiresGrad: true });

    const gelu = x.gelu();
    await checkStats('gelu(x)', gelu);

    const proj = api.matmul(gelu, realProjWeight.transpose({ dim0: 0, dim1: 1 }));
    await checkStats('proj(gelu)', proj);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 2: GELU → matmul with RANDOM proj weight (same shape)
  console.log('\nTest 2: GELU → matmul with RANDOM proj weight');
  {
    // Random weight with similar statistics
    const randomProjData = new Float32Array(embedDim * innerDim);
    const std = Math.sqrt(2.0 / innerDim);
    for (let i = 0; i < randomProjData.length; i++) {
      randomProjData[i] = (Math.random() * 2 - 1) * std * Math.sqrt(3);
    }
    const randomProjWeight = api.tensorFromArray(randomProjData, [embedDim, innerDim], { device: 'webgpu', requiresGrad: false });

    const xData = new Float32Array(seqLen * innerDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1) * 5;
    }
    const x = api.tensorFromArray(xData, [1, seqLen, innerDim], { device: 'webgpu', requiresGrad: true });

    const gelu = x.gelu();
    const proj = api.matmul(gelu, randomProjWeight.transpose({ dim0: 0, dim1: 1 }));

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 3: GELU → matmul with SCALED real proj weight (smaller)
  console.log('\nTest 3: GELU → matmul with SCALED (x0.1) real proj weight');
  {
    const scaledProjData = new Float32Array(cProjWTransposed.data.length);
    for (let i = 0; i < scaledProjData.length; i++) {
      scaledProjData[i] = cProjWTransposed.data[i] * 0.1;
    }
    const scaledProjWeight = api.tensorFromArray(scaledProjData, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });

    const xData = new Float32Array(seqLen * innerDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1) * 5;
    }
    const x = api.tensorFromArray(xData, [1, seqLen, innerDim], { device: 'webgpu', requiresGrad: true });

    const gelu = x.gelu();
    const proj = api.matmul(gelu, scaledProjWeight.transpose({ dim0: 0, dim1: 1 }));

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 4: Skip GELU entirely
  console.log('\nTest 4: Skip GELU - just matmul');
  {
    const xData = new Float32Array(seqLen * innerDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1) * 5;
    }
    const x = api.tensorFromArray(xData, [1, seqLen, innerDim], { device: 'webgpu', requiresGrad: true });

    const proj = api.matmul(x, realProjWeight.transpose({ dim0: 0, dim1: 1 }));

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 5: ReLU instead of GELU
  console.log('\nTest 5: ReLU instead of GELU → matmul');
  {
    const xData = new Float32Array(seqLen * innerDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1) * 5;
    }
    const x = api.tensorFromArray(xData, [1, seqLen, innerDim], { device: 'webgpu', requiresGrad: true });

    const relu = x.relu();
    const proj = api.matmul(relu, realProjWeight.transpose({ dim0: 0, dim1: 1 }));

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 6: GELU → smaller matmul (768 → 768)
  console.log('\nTest 6: GELU → smaller matmul (768 → 768)');
  {
    const smallWeight = api.tensorFromArray(
      new Float32Array(embedDim * embedDim).map(() => (Math.random() * 2 - 1) * 0.1),
      [embedDim, embedDim],
      { device: 'webgpu', requiresGrad: false }
    );

    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const gelu = x.gelu();
    const proj = api.matmul(gelu, smallWeight.transpose({ dim0: 0, dim1: 1 }));

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 7: Check c_proj weight statistics
  console.log('\n--- Analyzing c_proj weight ---');
  await checkStats('c_proj.weight (transposed)', realProjWeight);

  let wMin = Infinity, wMax = -Infinity;
  for (let i = 0; i < cProjWTransposed.data.length; i++) {
    const v = cProjWTransposed.data[i];
    if (v < wMin) wMin = v;
    if (v > wMax) wMax = v;
  }
  console.log(`c_proj weight range: [${wMin.toFixed(4)}, ${wMax.toFixed(4)}]`);

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
