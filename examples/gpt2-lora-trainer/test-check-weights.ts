#!/usr/bin/env npx tsx
/**
 * Check if GPT-2 weights from cache have extreme values.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

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
    switch (tensorInfo.dtype) {
      case 'F32': {
        const alignedBuffer = new ArrayBuffer(tensorData.length);
        new Uint8Array(alignedBuffer).set(tensorData);
        float32Data = new Float32Array(alignedBuffer);
        break;
      }
      case 'F16':
        float32Data = convertFloat16ToFloat32(tensorData);
        break;
      case 'BF16':
        float32Data = convertBFloat16ToFloat32(tensorData);
        break;
      default:
        console.warn(`Unsupported dtype: ${tensorInfo.dtype} for ${name}`);
        continue;
    }

    weights.set(name, { data: float32Data, shape: tensorInfo.shape });
  }

  return weights;
}

function convertFloat16ToFloat32(data: Uint8Array): Float32Array {
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const float16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(float16View.length);

  for (let i = 0; i < float16View.length; i++) {
    const h = float16View[i];
    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7c00) >> 10;
    const fraction = h & 0x03ff;

    if (exponent === 0) {
      float32[i] = fraction === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 0x1f) {
      float32[i] = fraction === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      float32[i] = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }

  return float32;
}

function convertBFloat16ToFloat32(data: Uint8Array): Float32Array {
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const bf16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(bf16View.length);

  for (let i = 0; i < bf16View.length; i++) {
    const asUint32 = bf16View[i] << 16;
    const float32View = new Float32Array(1);
    new Uint32Array(float32View.buffer)[0] = asUint32;
    float32[i] = float32View[0];
  }

  return float32;
}

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Check GPT-2 Weights');
  console.log('='.repeat(60));

  if (!fs.existsSync(WEIGHTS_CACHE)) {
    console.log('No cached weights found at', WEIGHTS_CACHE);
    return;
  }

  console.log('\nLoading weights from cache...');
  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
  console.log(`Loaded ${weights.size} tensors`);

  console.log('\nChecking for extreme values...');
  let hasExtreme = false;

  for (const [name, { data, shape }] of weights) {
    let min = Infinity;
    let max = -Infinity;
    let hasNaN = false;
    let hasInf = false;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (Number.isNaN(v)) hasNaN = true;
      if (!Number.isFinite(v)) hasInf = true;
      if (v < min) min = v;
      if (v > max) max = v;
    }

    // Report extreme weights (abs > 100)
    if (hasNaN || hasInf || Math.abs(min) > 100 || Math.abs(max) > 100) {
      hasExtreme = true;
      console.log(`\n${name} [${shape.join('x')}]:`);
      console.log(`  min: ${min}, max: ${max}`);
      if (hasNaN) console.log(`  HAS NaN!`);
      if (hasInf) console.log(`  HAS Inf!`);
    }
  }

  if (!hasExtreme) {
    console.log('\nNo extreme values found (all weights in [-100, 100])');
  }

  // Print some sample weights stats
  console.log('\n\nSample weight statistics:');
  const sampleWeights = ['wte.weight', 'h.0.attn.c_attn.weight', 'h.0.attn.c_attn.bias', 'ln_f.weight'];
  for (const name of sampleWeights) {
    const w = weights.get(name);
    if (!w) continue;

    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    for (let i = 0; i < w.data.length; i++) {
      const v = w.data[i];
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    const mean = sum / w.data.length;
    console.log(`\n${name} [${w.shape.join('x')}]:`);
    console.log(`  min: ${min.toExponential(4)}, max: ${max.toExponential(4)}, mean: ${mean.toExponential(4)}`);
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
