#!/usr/bin/env npx tsx
/**
 * Check h.0 attention weights specifically for numerical issues.
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

function analyzeWeight(name: string, data: Float32Array, shape: number[]): void {
  console.log(`\n${name} [${shape.join('x')}]:`);

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let zeros = 0;
  let verySmall = 0; // |x| < 1e-7
  let veryLarge = 0; // |x| > 10

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    sum += v;
    sumSq += v * v;
    if (v < min) min = v;
    if (v > max) max = v;
    if (v === 0) zeros++;
    if (Math.abs(v) < 1e-7 && v !== 0) verySmall++;
    if (Math.abs(v) > 10) veryLarge++;
  }

  const mean = sum / data.length;
  const variance = sumSq / data.length - mean * mean;
  const std = Math.sqrt(variance);

  console.log(`  Range: [${min.toExponential(4)}, ${max.toExponential(4)}]`);
  console.log(`  Mean: ${mean.toExponential(4)}, Std: ${std.toExponential(4)}`);
  console.log(`  Zeros: ${zeros} (${(100 * zeros / data.length).toFixed(2)}%)`);
  console.log(`  Very small (|x| < 1e-7): ${verySmall}`);
  console.log(`  Very large (|x| > 10): ${veryLarge}`);

  // Check for patterns that might cause numerical issues
  // e.g., columns/rows with very different scales
  if (shape.length === 2) {
    const [rows, cols] = shape;
    let colMinStd = Infinity;
    let colMaxStd = -Infinity;
    let rowMinStd = Infinity;
    let rowMaxStd = -Infinity;

    // Check column variances
    for (let c = 0; c < cols; c++) {
      let colSum = 0;
      let colSumSq = 0;
      for (let r = 0; r < rows; r++) {
        const v = data[r * cols + c];
        colSum += v;
        colSumSq += v * v;
      }
      const colMean = colSum / rows;
      const colVar = colSumSq / rows - colMean * colMean;
      const colStd = Math.sqrt(Math.max(0, colVar));
      if (colStd < colMinStd) colMinStd = colStd;
      if (colStd > colMaxStd) colMaxStd = colStd;
    }

    // Check row variances
    for (let r = 0; r < rows; r++) {
      let rowSum = 0;
      let rowSumSq = 0;
      for (let c = 0; c < cols; c++) {
        const v = data[r * cols + c];
        rowSum += v;
        rowSumSq += v * v;
      }
      const rowMean = rowSum / cols;
      const rowVar = rowSumSq / cols - rowMean * rowMean;
      const rowStd = Math.sqrt(Math.max(0, rowVar));
      if (rowStd < rowMinStd) rowMinStd = rowStd;
      if (rowStd > rowMaxStd) rowMaxStd = rowStd;
    }

    console.log(`  Column std range: [${colMinStd.toExponential(4)}, ${colMaxStd.toExponential(4)}]`);
    console.log(`  Row std range: [${rowMinStd.toExponential(4)}, ${rowMaxStd.toExponential(4)}]`);
  }
}

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('H.0 Weight Analysis');
  console.log('='.repeat(60));

  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  // Analyze h.0 weights specifically
  const h0Weights = [
    'h.0.ln_1.weight', 'h.0.ln_1.bias',
    'h.0.attn.c_attn.weight', 'h.0.attn.c_attn.bias',
    'h.0.attn.c_proj.weight', 'h.0.attn.c_proj.bias',
    'h.0.ln_2.weight', 'h.0.ln_2.bias',
    'h.0.mlp.c_fc.weight', 'h.0.mlp.c_fc.bias',
    'h.0.mlp.c_proj.weight', 'h.0.mlp.c_proj.bias',
    'wte.weight', 'wpe.weight',
  ];

  for (const name of h0Weights) {
    const w = weights.get(name);
    if (w) {
      analyzeWeight(name, w.data, w.shape);
    }
  }

  // Compare h.0 to h.1
  console.log('\n\n' + '='.repeat(60));
  console.log('Compare h.0 vs h.1');
  console.log('='.repeat(60));

  const compareWeights = ['attn.c_attn.weight', 'attn.c_attn.bias'];
  for (const suffix of compareWeights) {
    const h0 = weights.get(`h.0.${suffix}`);
    const h1 = weights.get(`h.1.${suffix}`);

    if (h0 && h1) {
      console.log(`\n${suffix}:`);

      // Calculate stats for both
      let h0Sum = 0, h0SumSq = 0;
      let h1Sum = 0, h1SumSq = 0;
      for (let i = 0; i < h0.data.length; i++) {
        h0Sum += h0.data[i];
        h0SumSq += h0.data[i] * h0.data[i];
        h1Sum += h1.data[i];
        h1SumSq += h1.data[i] * h1.data[i];
      }

      const h0Mean = h0Sum / h0.data.length;
      const h0Std = Math.sqrt(h0SumSq / h0.data.length - h0Mean * h0Mean);
      const h1Mean = h1Sum / h1.data.length;
      const h1Std = Math.sqrt(h1SumSq / h1.data.length - h1Mean * h1Mean);

      console.log(`  h.0: mean=${h0Mean.toExponential(4)}, std=${h0Std.toExponential(4)}`);
      console.log(`  h.1: mean=${h1Mean.toExponential(4)}, std=${h1Std.toExponential(4)}`);
    }
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
