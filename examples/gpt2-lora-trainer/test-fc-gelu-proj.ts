#!/usr/bin/env npx tsx
/**
 * Test fc → gelu → proj chain to find NaN source.
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
  const embedDim = 768;
  const innerDim = 3072; // 4 * 768

  // Load REAL MLP weights
  const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
  const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
  const cProjW = weights.get('h.0.mlp.c_proj.weight')!;
  const cProjB = weights.get('h.0.mlp.c_proj.bias')!;

  const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);
  const cProjWTransposed = transposeWeight(cProjW.data, cProjW.shape);

  const realFcWeight = api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });
  const realFcBias = api.tensorFromArray(cFcB.data, cFcB.shape, { device: 'webgpu', requiresGrad: false });
  const realProjWeight = api.tensorFromArray(cProjWTransposed.data, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });
  const realProjBias = api.tensorFromArray(cProjB.data, cProjB.shape, { device: 'webgpu', requiresGrad: false });

  console.log(`\n${'='.repeat(60)}`);
  console.log('FC → GELU → Proj Chain Investigation');
  console.log('='.repeat(60));

  // Test 1: Full chain with REAL weights
  console.log('\nTest 1: fc → gelu → proj with REAL weights');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, realFcWeight.transpose({ dim0: 0, dim1: 1 })), realFcBias);
    await checkStats('fc', fc);

    const gelu = fc.gelu();
    await checkStats('gelu', gelu);

    const proj = api.add(api.matmul(gelu, realProjWeight.transpose({ dim0: 0, dim1: 1 })), realProjBias);
    await checkStats('proj', proj);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 2: fc → gelu → proj with RANDOM fc weight, REAL proj weight
  console.log('\nTest 2: RANDOM fc weight, REAL proj weight');
  {
    const randomFcData = new Float32Array(innerDim * embedDim);
    const std = Math.sqrt(2.0 / embedDim);
    for (let i = 0; i < randomFcData.length; i++) {
      randomFcData[i] = (Math.random() * 2 - 1) * std;
    }
    const randomFcWeight = api.tensorFromArray(randomFcData, [innerDim, embedDim], { device: 'webgpu', requiresGrad: false });
    const randomFcBias = api.zeros([innerDim], { device: 'webgpu', requiresGrad: false });

    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, randomFcWeight.transpose({ dim0: 0, dim1: 1 })), randomFcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, realProjWeight.transpose({ dim0: 0, dim1: 1 })), realProjBias);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 3: REAL fc weight, RANDOM proj weight
  console.log('\nTest 3: REAL fc weight, RANDOM proj weight');
  {
    const randomProjData = new Float32Array(embedDim * innerDim);
    const std = Math.sqrt(2.0 / innerDim);
    for (let i = 0; i < randomProjData.length; i++) {
      randomProjData[i] = (Math.random() * 2 - 1) * std;
    }
    const randomProjWeight = api.tensorFromArray(randomProjData, [embedDim, innerDim], { device: 'webgpu', requiresGrad: false });
    const randomProjBias = api.zeros([embedDim], { device: 'webgpu', requiresGrad: false });

    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, realFcWeight.transpose({ dim0: 0, dim1: 1 })), realFcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, randomProjWeight.transpose({ dim0: 0, dim1: 1 })), randomProjBias);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 4: BOTH random weights
  console.log('\nTest 4: BOTH RANDOM weights');
  {
    const randomFcData = new Float32Array(innerDim * embedDim);
    const stdFc = Math.sqrt(2.0 / embedDim);
    for (let i = 0; i < randomFcData.length; i++) {
      randomFcData[i] = (Math.random() * 2 - 1) * stdFc;
    }
    const randomFcWeight = api.tensorFromArray(randomFcData, [innerDim, embedDim], { device: 'webgpu', requiresGrad: false });
    const randomFcBias = api.zeros([innerDim], { device: 'webgpu', requiresGrad: false });

    const randomProjData = new Float32Array(embedDim * innerDim);
    const stdProj = Math.sqrt(2.0 / innerDim);
    for (let i = 0; i < randomProjData.length; i++) {
      randomProjData[i] = (Math.random() * 2 - 1) * stdProj;
    }
    const randomProjWeight = api.tensorFromArray(randomProjData, [embedDim, innerDim], { device: 'webgpu', requiresGrad: false });
    const randomProjBias = api.zeros([embedDim], { device: 'webgpu', requiresGrad: false });

    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, randomFcWeight.transpose({ dim0: 0, dim1: 1 })), randomFcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, randomProjWeight.transpose({ dim0: 0, dim1: 1 })), randomProjBias);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 5: SCALED REAL weights (0.1x)
  console.log('\nTest 5: SCALED (0.1x) REAL weights');
  {
    const scaledFcData = new Float32Array(cFcWTransposed.data.length);
    for (let i = 0; i < scaledFcData.length; i++) {
      scaledFcData[i] = cFcWTransposed.data[i] * 0.1;
    }
    const scaledFcWeight = api.tensorFromArray(scaledFcData, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });

    const scaledProjData = new Float32Array(cProjWTransposed.data.length);
    for (let i = 0; i < scaledProjData.length; i++) {
      scaledProjData[i] = cProjWTransposed.data[i] * 0.1;
    }
    const scaledProjWeight = api.tensorFromArray(scaledProjData, cProjWTransposed.shape, { device: 'webgpu', requiresGrad: false });

    const scaledFcBiasData = new Float32Array(cFcB.data.length);
    for (let i = 0; i < scaledFcBiasData.length; i++) {
      scaledFcBiasData[i] = cFcB.data[i] * 0.1;
    }
    const scaledFcBias = api.tensorFromArray(scaledFcBiasData, cFcB.shape, { device: 'webgpu', requiresGrad: false });

    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, scaledFcWeight.transpose({ dim0: 0, dim1: 1 })), scaledFcBias);
    await checkStats('fc (scaled)', fc);

    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, scaledProjWeight.transpose({ dim0: 0, dim1: 1 })), realProjBias);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Analyze c_fc weight
  console.log('\n--- Analyzing c_fc weight ---');
  let fcMin = Infinity, fcMax = -Infinity;
  for (let i = 0; i < cFcWTransposed.data.length; i++) {
    const v = cFcWTransposed.data[i];
    if (v < fcMin) fcMin = v;
    if (v > fcMax) fcMax = v;
  }
  console.log(`c_fc weight range: [${fcMin.toFixed(4)}, ${fcMax.toFixed(4)}]`);

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
