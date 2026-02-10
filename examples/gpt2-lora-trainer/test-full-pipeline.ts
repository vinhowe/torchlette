#!/usr/bin/env npx tsx
/**
 * Test full pipeline incrementally to find where NaN appears.
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

function embedding(api: Torchlette, indices: Tensor, weight: Tensor, embeddingDim: number): Tensor {
  const inputShape = indices.shape;
  const numElements = inputShape.reduce((a, b) => a * b, 1);

  const flatInput = indices.reshape([numElements]);
  const expandedInput = flatInput
    .reshape([numElements, 1])
    .expand([numElements, embeddingDim])
    .contiguous();

  const gathered = weight.gather(expandedInput, { dim: 0 });

  const outputShape = [...inputShape, embeddingDim];
  return gathered.reshape(outputShape);
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
  const vocabSize = 50257;
  const numHeads = 12;
  const headDim = embedDim / numHeads;
  const seqLen = 32;
  const batch = 1;
  const rank = 8;
  const scaling = 16 / 8;
  const layerIdx = 0;

  // Load weights
  const wteData = weights.get('wte.weight')!;
  const wpeData = weights.get('wpe.weight')!;
  const ln1W = weights.get(`h.${layerIdx}.ln_1.weight`)!;
  const ln1B = weights.get(`h.${layerIdx}.ln_1.bias`)!;
  const cAttnW_raw = weights.get(`h.${layerIdx}.attn.c_attn.weight`)!;
  const cAttnB_raw = weights.get(`h.${layerIdx}.attn.c_attn.bias`)!;
  const cProjW_raw = weights.get(`h.${layerIdx}.attn.c_proj.weight`)!;
  const cProjB_raw = weights.get(`h.${layerIdx}.attn.c_proj.bias`)!;
  const lnFW = weights.get('ln_f.weight')!;
  const lnFB = weights.get('ln_f.bias')!;

  const wte = api.tensorFromArray(wteData.data, wteData.shape, { device: 'webgpu', requiresGrad: false });
  const wpe = api.tensorFromArray(wpeData.data, wpeData.shape, { device: 'webgpu', requiresGrad: false });
  const lnWeight = api.tensorFromArray(ln1W.data, ln1W.shape, { device: 'webgpu', requiresGrad: false });
  const lnBias = api.tensorFromArray(ln1B.data, ln1B.shape, { device: 'webgpu', requiresGrad: false });
  const cAttnW_transposed = transposeWeight(cAttnW_raw.data, cAttnW_raw.shape);
  const cAttnW = api.tensorFromArray(cAttnW_transposed.data, cAttnW_transposed.shape, { device: 'webgpu', requiresGrad: false });
  const cAttnB = api.tensorFromArray(cAttnB_raw.data, cAttnB_raw.shape, { device: 'webgpu', requiresGrad: false });
  const cProjW_transposed = transposeWeight(cProjW_raw.data, cProjW_raw.shape);
  const cProjW = api.tensorFromArray(cProjW_transposed.data, cProjW_transposed.shape, { device: 'webgpu', requiresGrad: false });
  const cProjB = api.tensorFromArray(cProjB_raw.data, cProjB_raw.shape, { device: 'webgpu', requiresGrad: false });
  const lnFWeight = api.tensorFromArray(lnFW.data, lnFW.shape, { device: 'webgpu', requiresGrad: false });
  const lnFBias = api.tensorFromArray(lnFB.data, lnFB.shape, { device: 'webgpu', requiresGrad: false });

  // LoRA parameters
  const stdA = Math.sqrt(1.0 / embedDim);
  const loraAData = new Float32Array(rank * embedDim);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * stdA * Math.sqrt(3);
  }
  const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
  const loraB = api.zeros([3 * embedDim, rank], { device: 'webgpu', requiresGrad: true });

  // Input tokens
  const inputData = new Float32Array(batch * seqLen);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.floor(Math.random() * vocabSize);
  }
  const input = api.tensorFromArray(inputData, [batch, seqLen], { device: 'webgpu' });

  // Target tokens
  const targetData = new Float32Array(batch * seqLen);
  for (let i = 0; i < targetData.length; i++) {
    targetData[i] = Math.floor(Math.random() * vocabSize);
  }
  const target = api.tensorFromArray(targetData, [batch, seqLen], { device: 'webgpu' });

  console.log(`\n${'='.repeat(60)}`);
  console.log('Full Pipeline Test (with h.0 weights)');
  console.log('='.repeat(60));

  // Step 1: Token embeddings
  const tokEmb = embedding(api, input, wte, embedDim);
  await checkStats('1. Token embeddings', tokEmb);

  // Step 2: Position embeddings
  const positions = api.tensorFromArray(Array.from({ length: seqLen }, (_, i) => i), [seqLen]);
  const posEmb = embedding(api, positions, wpe, embedDim);
  await checkStats('2. Position embeddings', posEmb);

  // Step 3: Combined embeddings
  let x = api.add(tokEmb, posEmb);
  await checkStats('3. Combined embeddings', x);

  // Step 4: Layer norm
  const normed = layerNorm(api, x, lnWeight, lnBias);
  await checkStats('4. After LayerNorm', normed);

  // Step 5: Base QKV
  const baseQKV = api.add(
    api.matmul(normed, cAttnW.transpose({ dim0: 0, dim1: 1 })),
    cAttnB
  );
  await checkStats('5. Base QKV', baseQKV);

  // Step 6: LoRA QKV
  const loraQKV = api.mul(
    api.matmul(
      api.matmul(normed, loraA.transpose({ dim0: 0, dim1: 1 })),
      loraB.transpose({ dim0: 0, dim1: 1 })
    ),
    api.tensorFromArray([scaling], [])
  );
  await checkStats('6. LoRA QKV', loraQKV);

  // Step 7: Combined QKV (detach base)
  const qkv = api.add(baseQKV.detach(), loraQKV);
  await checkStats('7. Combined QKV', qkv);

  // Step 8: Attention
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
  await checkStats('8. Attention weights', attnWeights);

  const attnOut = api.matmul(attnWeights, v);
  await checkStats('9. Attention output (before cProj)', attnOut);

  // Step 10: Output projection
  const attnReshaped = attnOut.reshape([batch, numHeads, seqLen, headDim]);
  const attnPermuted = attnReshaped.permute([0, 2, 1, 3]);
  const attnFlat = attnPermuted.reshape([batch, seqLen, embedDim]);
  const projected = api.add(
    api.matmul(attnFlat, cProjW.transpose({ dim0: 0, dim1: 1 })),
    cProjB
  );
  await checkStats('10. Output projection', projected);

  // Step 11: Residual
  const afterAttn = api.add(x, projected);
  await checkStats('11. After residual', afterAttn);

  // Step 12: Final layer norm
  const finalNormed = layerNorm(api, afterAttn, lnFWeight, lnFBias);
  await checkStats('12. Final LayerNorm', finalNormed);

  // Step 13: Weight-tied output (logits)
  const logits = api.matmul(finalNormed, wte.transpose({ dim0: 0, dim1: 1 }));
  await checkStats('13. Logits', logits);

  // Step 14: Cross-entropy loss
  const logitsFlat = logits.reshape([batch * seqLen, vocabSize]);
  const targetFlat = target.reshape([batch * seqLen]);

  // Log-softmax
  const maxLogits = logitsFlat.max({ dim: -1, keepdim: true }) as Tensor;
  const shifted = api.sub(logitsFlat, maxLogits);
  const expShifted = shifted.exp();
  const sumExp = expShifted.sum({ dim: -1, keepdim: true }) as Tensor;
  const logSumExp = sumExp.log();
  const logSoftmax = api.sub(shifted, logSumExp);
  await checkStats('14. Log-softmax', logSoftmax);

  const targetsForGather = targetFlat.reshape([batch * seqLen, 1]);
  const gatheredLogProbs = api.gather(logSoftmax, targetsForGather, { dim: 1 });
  const gatheredSqueezed = gatheredLogProbs.reshape([batch * seqLen]);

  const loss = api.neg(gatheredSqueezed).mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\n15. Loss: ${lossVal.toFixed(4)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();

  // Check gradients
  if (loraA.grad) {
    await checkStats('loraA.grad', loraA.grad);
  }
  if (loraB.grad) {
    await checkStats('loraB.grad', loraB.grad);
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
