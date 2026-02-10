#!/usr/bin/env npx tsx
/**
 * Debug h.0 with layer norm - does adding layer norm cause NaN?
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

function layerNorm(api: Torchlette, x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
  const inputShape = x.shape;
  const meanResult = api.mean(x, { dim: -1 }) as Tensor;
  const meanShape = [...inputShape.slice(0, -1), 1];
  const mean = meanResult.reshape(meanShape);

  const xCentered = api.sub(x, mean);
  const varianceResult = api.mean(api.mul(xCentered, xCentered), { dim: -1 }) as Tensor;
  const variance = varianceResult.reshape(meanShape);

  const epsTensor = api.tensorFromArray([eps], []);
  const std = api.sqrt(api.add(variance, epsTensor));
  const normalized = api.div(xCentered, std);

  return api.add(api.mul(normalized, weight), bias);
}

async function runTest(
  api: Torchlette,
  weights: Map<string, { data: Float32Array; shape: number[] }>,
  layerIdx: number
): Promise<boolean> {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing h.${layerIdx} with layer norm`);
  console.log('='.repeat(60));

  const embedDim = 768;
  const numHeads = 12;
  const headDim = embedDim / numHeads;
  const seqLen = 32;
  const batch = 1;
  const rank = 8;
  const scaling = 16 / 8;

  // Create random input
  const xData = new Float32Array(batch * seqLen * embedDim);
  for (let i = 0; i < xData.length; i++) {
    xData[i] = (Math.random() * 2 - 1) * 0.5;
  }
  const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { device: 'webgpu', requiresGrad: true });

  await checkStats('Input x', x);

  // Load layer norm weights
  const ln1W = weights.get(`h.${layerIdx}.ln_1.weight`)!;
  const ln1B = weights.get(`h.${layerIdx}.ln_1.bias`)!;
  const lnWeight = api.tensorFromArray(ln1W.data, ln1W.shape, { device: 'webgpu', requiresGrad: false });
  const lnBias = api.tensorFromArray(ln1B.data, ln1B.shape, { device: 'webgpu', requiresGrad: false });

  // Apply layer norm
  const normed = layerNorm(api, x, lnWeight, lnBias);
  await checkStats('After LayerNorm', normed);

  // Load attention weights
  const cAttnW_raw = weights.get(`h.${layerIdx}.attn.c_attn.weight`)!;
  const cAttnB_raw = weights.get(`h.${layerIdx}.attn.c_attn.bias`)!;
  const cAttnW_transposed = transposeWeight(cAttnW_raw.data, cAttnW_raw.shape);

  const cAttnW = api.tensorFromArray(cAttnW_transposed.data, cAttnW_transposed.shape, { device: 'webgpu', requiresGrad: false });
  const cAttnB = api.tensorFromArray(cAttnB_raw.data, cAttnB_raw.shape, { device: 'webgpu', requiresGrad: false });

  // LoRA parameters
  const stdA = Math.sqrt(1.0 / embedDim);
  const loraAData = new Float32Array(rank * embedDim);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * stdA * Math.sqrt(3);
  }
  const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
  const loraB = api.zeros([3 * embedDim, rank], { device: 'webgpu', requiresGrad: true });

  // Base QKV
  const baseQKV = api.add(
    api.matmul(normed, cAttnW.transpose({ dim0: 0, dim1: 1 })),
    cAttnB
  );
  await checkStats('Base QKV', baseQKV);

  // LoRA QKV
  const loraQKV = api.mul(
    api.matmul(
      api.matmul(normed, loraA.transpose({ dim0: 0, dim1: 1 })),
      loraB.transpose({ dim0: 0, dim1: 1 })
    ),
    api.tensorFromArray([scaling], [])
  );
  await checkStats('LoRA QKV', loraQKV);

  // Combined QKV (detach base)
  const qkv = api.add(baseQKV.detach(), loraQKV);
  await checkStats('Combined QKV', qkv);

  // Reshape for Q, K, V
  const qkvReshaped = qkv.reshape([batch, seqLen, 3, numHeads, headDim]);
  const qkvPermuted = qkvReshaped.permute([2, 0, 3, 1, 4]);
  const totalSize = batch * numHeads * seqLen * headDim;
  const qkvFlat = qkvPermuted.reshape([3, totalSize]);

  const indices0 = api.tensorFromArray(Array(totalSize).fill(0), [1, totalSize]);
  const indices1 = api.tensorFromArray(Array(totalSize).fill(1), [1, totalSize]);
  const indices2 = api.tensorFromArray(Array(totalSize).fill(2), [1, totalSize]);

  const q = api.gather(qkvFlat, indices0, { dim: 0 }).reshape([batch * numHeads, seqLen, headDim]);
  const k = api.gather(qkvFlat, indices1, { dim: 0 }).reshape([batch * numHeads, seqLen, headDim]);
  const v = api.gather(qkvFlat, indices2, { dim: 0 }).reshape([batch * numHeads, seqLen, headDim]);

  await checkStats('Q', q);
  await checkStats('K', k);
  await checkStats('V', v);

  // Attention
  const kT = k.transpose({ dim0: 1, dim1: 2 });
  const scores = api.matmul(q, kT);
  const scale = api.tensorFromArray([1.0 / Math.sqrt(headDim)], []);
  const scaledScores = api.mul(scores, scale);

  const mask = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask[i * seqLen + j] = j > i ? -1e9 : 0;
    }
  }
  const maskTensor = api.tensorFromArray(mask, [seqLen, seqLen]);
  const maskedScores = api.add(scaledScores, maskTensor);

  const attnWeights = maskedScores.softmax(-1);
  await checkStats('Attention weights', attnWeights);

  const attnOut = api.matmul(attnWeights, v);
  await checkStats('Attention output', attnOut);

  // Simple loss
  const loss = attnOut.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(6)}`);

  await loss.backward();

  // Check gradients
  let hasNaN = false;
  if (loraA.grad) {
    const nanA = await checkStats('loraA.grad', loraA.grad);
    hasNaN = hasNaN || nanA;
  }
  if (loraB.grad) {
    const nanB = await checkStats('loraB.grad', loraB.grad);
    hasNaN = hasNaN || nanB;
  }

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

  const hasNaN0 = await runTest(api, weights, 0);
  const hasNaN1 = await runTest(api, weights, 1);

  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));
  console.log(`h.0 with LayerNorm: ${hasNaN0 ? 'NaN' : 'OK'}`);
  console.log(`h.1 with LayerNorm: ${hasNaN1 ? 'NaN' : 'OK'}`);
}

main().catch(console.error);
