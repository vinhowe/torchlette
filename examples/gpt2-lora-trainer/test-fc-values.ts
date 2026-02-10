#!/usr/bin/env npx tsx
/**
 * Analyze fc output values in detail to find extreme values.
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
  const innerDim = 3072;

  // Load REAL c_fc weights
  const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
  const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
  const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);

  const realFcWeight = api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape, { device: 'webgpu', requiresGrad: false });
  const realFcBias = api.tensorFromArray(cFcB.data, cFcB.shape, { device: 'webgpu', requiresGrad: false });

  console.log(`\n${'='.repeat(60)}`);
  console.log('FC Output Value Analysis');
  console.log('='.repeat(60));

  // Create input
  const xData = new Float32Array(seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1);
  }
  const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

  // Compute fc
  const fc = api.add(api.matmul(x, realFcWeight.transpose({ dim0: 0, dim1: 1 })), realFcBias);

  // Get fc values using runtime read
  const runtime = api._runtime();
  const fcTensor = fc._unwrap();
  const fcFloat32 = await runtime.read(fcTensor);
  const fcArray = Array.from(fcFloat32);

  // Analyze fc values
  let fcMin = Infinity, fcMax = -Infinity;
  let fcSum = 0;
  let fcSumSq = 0;
  let largePositive = 0;  // > 10
  let largeNegative = 0;  // < -10
  let veryLarge = 0;      // |x| > 50
  let nanCount = 0;
  let infCount = 0;

  for (const v of fcArray) {
    if (Number.isNaN(v)) {
      nanCount++;
      continue;
    }
    if (!Number.isFinite(v)) {
      infCount++;
      continue;
    }
    fcSum += v;
    fcSumSq += v * v;
    if (v < fcMin) fcMin = v;
    if (v > fcMax) fcMax = v;
    if (v > 10) largePositive++;
    if (v < -10) largeNegative++;
    if (Math.abs(v) > 50) veryLarge++;
  }

  const fcMean = fcSum / fcArray.length;
  const fcVar = fcSumSq / fcArray.length - fcMean * fcMean;
  const fcStd = Math.sqrt(fcVar);

  console.log('\nFC output statistics:');
  console.log(`  Shape: [1, ${seqLen}, ${innerDim}] = ${fcArray.length} elements`);
  console.log(`  Min: ${fcMin.toFixed(4)}`);
  console.log(`  Max: ${fcMax.toFixed(4)}`);
  console.log(`  Mean: ${fcMean.toFixed(4)}`);
  console.log(`  Std: ${fcStd.toFixed(4)}`);
  console.log(`  Values > 10: ${largePositive}`);
  console.log(`  Values < -10: ${largeNegative}`);
  console.log(`  Values |x| > 50: ${veryLarge}`);
  console.log(`  NaN count: ${nanCount}`);
  console.log(`  Inf count: ${infCount}`);

  // Compute GELU
  const gelu = fc.gelu();
  const geluTensor = gelu._unwrap();
  const geluFloat32 = await runtime.read(geluTensor);
  const geluArray = Array.from(geluFloat32);

  let geluMin = Infinity, geluMax = -Infinity;
  let geluSum = 0;
  let geluNan = 0;
  let geluInf = 0;

  for (const v of geluArray) {
    if (Number.isNaN(v)) {
      geluNan++;
      continue;
    }
    if (!Number.isFinite(v)) {
      geluInf++;
      continue;
    }
    geluSum += v;
    if (v < geluMin) geluMin = v;
    if (v > geluMax) geluMax = v;
  }

  console.log('\nGELU output statistics:');
  console.log(`  Min: ${geluMin.toFixed(4)}`);
  console.log(`  Max: ${geluMax.toFixed(4)}`);
  console.log(`  Sum: ${geluSum.toExponential(4)}`);
  console.log(`  NaN count: ${geluNan}`);
  console.log(`  Inf count: ${geluInf}`);

  // Check c_fc bias values
  console.log('\nC_FC bias statistics:');
  let biasMin = Infinity, biasMax = -Infinity;
  let biasSum = 0;
  for (let i = 0; i < cFcB.data.length; i++) {
    const v = cFcB.data[i];
    biasSum += v;
    if (v < biasMin) biasMin = v;
    if (v > biasMax) biasMax = v;
  }
  console.log(`  Min: ${biasMin.toFixed(4)}`);
  console.log(`  Max: ${biasMax.toFixed(4)}`);
  console.log(`  Mean: ${(biasSum / cFcB.data.length).toFixed(4)}`);

  // Check if FC output WITHOUT bias has issues
  console.log('\n--- FC without bias ---');
  const fcNoBias = api.matmul(x, realFcWeight.transpose({ dim0: 0, dim1: 1 }));
  const fcNoBiasTensor = fcNoBias._unwrap();
  const fcNoBiasFloat32 = await runtime.read(fcNoBiasTensor);
  const fcNoBiasArray = Array.from(fcNoBiasFloat32);

  let noBiasMin = Infinity, noBiasMax = -Infinity;
  for (const v of fcNoBiasArray) {
    if (Number.isFinite(v)) {
      if (v < noBiasMin) noBiasMin = v;
      if (v > noBiasMax) noBiasMax = v;
    }
  }
  console.log(`  Min: ${noBiasMin.toFixed(4)}`);
  console.log(`  Max: ${noBiasMax.toFixed(4)}`);

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
