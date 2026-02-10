#!/usr/bin/env npx tsx
/**
 * Debug h.0 attention specifically - check intermediate values.
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
  const numel = t.shape.reduce((a, b) => a * b, 1);
  const mean = sum / numel;

  const hasNaN = Number.isNaN(sum);
  console.log(`${name}: sum=${sum.toExponential(3)}, mean=${mean.toExponential(3)}${hasNaN ? ' [NaN!]' : ''}`);
  return hasNaN;
}

async function runAttentionTest(
  api: Torchlette,
  x: Tensor,
  cAttnW: Tensor,
  cAttnB: Tensor,
  loraA: Tensor,
  loraB: Tensor,
  scaling: number,
  numHeads: number,
  embedDim: number,
  layerName: string
): Promise<void> {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing ${layerName} attention`);
  console.log('='.repeat(60));

  const headDim = embedDim / numHeads;
  const [batch, seqLen] = [x.shape[0], x.shape[1]];

  // Check input
  await checkStats('Input x', x);

  // Base QKV
  const baseQKV = api.add(
    api.matmul(x, cAttnW.transpose({ dim0: 0, dim1: 1 })),
    cAttnB
  );
  await checkStats('Base QKV', baseQKV);

  // LoRA QKV
  const loraQKV = api.mul(
    api.matmul(
      api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 })),
      loraB.transpose({ dim0: 0, dim1: 1 })
    ),
    api.tensorFromArray([scaling], [])
  );
  await checkStats('LoRA QKV', loraQKV);

  // Combined QKV
  const qkv = api.add(baseQKV.detach(), loraQKV);
  await checkStats('Combined QKV', qkv);

  // Reshape to [batch, seqLen, 3, numHeads, headDim]
  const qkvReshaped = qkv.reshape([batch, seqLen, 3, numHeads, headDim]);

  // Permute to [3, batch, numHeads, seqLen, headDim]
  const qkvPermuted = qkvReshaped.permute([2, 0, 3, 1, 4]);

  const totalSize = batch * numHeads * seqLen * headDim;
  const qkvFlat = qkvPermuted.reshape([3, totalSize]);

  // Extract Q, K, V
  const indices0 = api.tensorFromArray(Array(totalSize).fill(0), [1, totalSize]);
  const indices1 = api.tensorFromArray(Array(totalSize).fill(1), [1, totalSize]);
  const indices2 = api.tensorFromArray(Array(totalSize).fill(2), [1, totalSize]);

  const qFlat = api.gather(qkvFlat, indices0, { dim: 0 }).reshape([totalSize]);
  const kFlat = api.gather(qkvFlat, indices1, { dim: 0 }).reshape([totalSize]);
  const vFlat = api.gather(qkvFlat, indices2, { dim: 0 }).reshape([totalSize]);

  const q = qFlat.reshape([batch * numHeads, seqLen, headDim]);
  const k = kFlat.reshape([batch * numHeads, seqLen, headDim]);
  const v = vFlat.reshape([batch * numHeads, seqLen, headDim]);

  await checkStats('Q', q);
  await checkStats('K', k);
  await checkStats('V', v);

  // Scaled dot-product: Q @ K^T / sqrt(d)
  const kT = k.transpose({ dim0: 1, dim1: 2 });
  const scores = api.matmul(q, kT);
  await checkStats('Raw scores (Q @ K^T)', scores);

  const scale = api.tensorFromArray([1.0 / Math.sqrt(headDim)], []);
  const scaledScores = api.mul(scores, scale);
  await checkStats('Scaled scores', scaledScores);

  // Apply causal mask
  const mask = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask[i * seqLen + j] = j > i ? -1e9 : 0;
    }
  }
  const maskTensor = api.tensorFromArray(mask, [seqLen, seqLen]);
  const maskedScores = api.add(scaledScores, maskTensor);
  await checkStats('Masked scores', maskedScores);

  // Softmax
  const attnWeights = maskedScores.softmax(-1);
  await checkStats('Attention weights (after softmax)', attnWeights);

  // Apply to values
  const attnOut = api.matmul(attnWeights, v);
  await checkStats('Attention output', attnOut);

  // Run backward from a simple loss
  const loss = attnOut.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(6)}`);

  await loss.backward();

  // Check loraA and loraB gradients
  if (loraA.grad) {
    await checkStats('loraA.grad', loraA.grad);
  } else {
    console.log('loraA.grad: null');
  }

  if (loraB.grad) {
    await checkStats('loraB.grad', loraB.grad);
  } else {
    console.log('loraB.grad: null');
  }
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
  const numHeads = 12;
  const seqLen = 32;
  const rank = 8;
  const scaling = 16 / 8; // alpha / rank

  // Create random input (simulating embedded tokens)
  const xData = new Float32Array(seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1) * 0.5;
  }
  const x = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

  // LoRA parameters
  const stdA = Math.sqrt(1.0 / embedDim);
  const loraAData = new Float32Array(rank * embedDim);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * stdA * Math.sqrt(3);
  }

  // Test with h.0 weights
  {
    const cAttnW_raw = weights.get('h.0.attn.c_attn.weight')!;
    const cAttnB_raw = weights.get('h.0.attn.c_attn.bias')!;
    const cAttnW_transposed = transposeWeight(cAttnW_raw.data, cAttnW_raw.shape);

    const cAttnW = api.tensorFromArray(cAttnW_transposed.data, cAttnW_transposed.shape, { device: 'webgpu', requiresGrad: false });
    const cAttnB = api.tensorFromArray(cAttnB_raw.data, cAttnB_raw.shape, { device: 'webgpu', requiresGrad: false });

    const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
    const loraB = api.zeros([3 * embedDim, rank], { device: 'webgpu', requiresGrad: true });

    await runAttentionTest(api, x, cAttnW, cAttnB, loraA, loraB, scaling, numHeads, embedDim, 'h.0');
  }

  // Test with h.1 weights
  {
    const cAttnW_raw = weights.get('h.1.attn.c_attn.weight')!;
    const cAttnB_raw = weights.get('h.1.attn.c_attn.bias')!;
    const cAttnW_transposed = transposeWeight(cAttnW_raw.data, cAttnW_raw.shape);

    const cAttnW = api.tensorFromArray(cAttnW_transposed.data, cAttnW_transposed.shape, { device: 'webgpu', requiresGrad: false });
    const cAttnB = api.tensorFromArray(cAttnB_raw.data, cAttnB_raw.shape, { device: 'webgpu', requiresGrad: false });

    // New x for clean autograd graph
    const x2 = api.tensorFromArray(xData, [1, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

    const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
    const loraB = api.zeros([3 * embedDim, rank], { device: 'webgpu', requiresGrad: true });

    await runAttentionTest(api, x2, cAttnW, cAttnB, loraA, loraB, scaling, numHeads, embedDim, 'h.1');
  }
}

main().catch(console.error);
