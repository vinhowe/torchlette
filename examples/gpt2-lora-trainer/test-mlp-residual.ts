#!/usr/bin/env npx tsx
/**
 * Investigate why residual connection fixes NaN gradient.
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

  console.log(`\n${'='.repeat(60)}`);
  console.log('MLP Residual Investigation');
  console.log('='.repeat(60));

  // Test 1: MLP direct (no residual) - NaN expected
  console.log('\nTest 1: MLP direct (no residual)');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, projWeight.transpose({ dim0: 0, dim1: 1 })), projBias);

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 2: MLP with DIFFERENT tensor for residual
  console.log('\nTest 2: MLP with different tensor for residual (y + proj)');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    // Different tensor for residual
    const yData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < yData.length; i++) {
      yData[i] = 0.1; // Constant value
    }
    const y = api.tensorFromArray(yData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, projWeight.transpose({ dim0: 0, dim1: 1 })), projBias);
    const h = api.add(y, proj); // Residual with DIFFERENT tensor

    const loss = h.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
    if (y.grad) await checkStats('y.grad', y.grad);
  }

  // Test 3: MLP with SAME tensor for residual (x + proj where fc uses x)
  console.log('\nTest 3: MLP with same tensor for residual (x + proj)');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    const gelu = fc.gelu();
    const proj = api.add(api.matmul(gelu, projWeight.transpose({ dim0: 0, dim1: 1 })), projBias);
    const h = api.add(x, proj); // Residual with SAME tensor

    const loss = h.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 4: Just GELU followed by matmul (simpler case)
  console.log('\nTest 4: Just fc -> gelu -> proj (no bias, simplified)');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    // Simpler version - no bias
    const fc = api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 }));
    const gelu = fc.gelu();
    const proj = api.matmul(gelu, projWeight.transpose({ dim0: 0, dim1: 1 }));

    const loss = proj.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 5: fc -> gelu only (no proj)
  console.log('\nTest 5: Just fc -> gelu (no proj)');
  {
    const xData = new Float32Array(seqLen * embedDim);
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 2 - 1);
    }
    const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const fc = api.add(api.matmul(x, fcWeight.transpose({ dim0: 0, dim1: 1 })), fcBias);
    const gelu = fc.gelu();

    const loss = gelu.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  // Test 6: Just GELU with large input values (simulate fc output)
  console.log('\nTest 6: Just GELU with large input (simulating fc output)');
  {
    // Create values similar to fc output
    const xData = new Float32Array(seqLen * 3072); // 4 * embedDim
    for (let i = 0; i < xData.length; i++) {
      xData[i] = (Math.random() * 20 - 10); // Large values like fc output
    }
    const x = api.tensorFromArray(xData, [1, seqLen, 3072], { device: 'webgpu', requiresGrad: true });

    const gelu = x.gelu();

    const loss = gelu.mean() as Tensor;
    await loss.backward();

    if (x.grad) await checkStats('x.grad', x.grad);
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
