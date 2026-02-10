#!/usr/bin/env npx tsx
/**
 * Test full attention mechanism from GPT-2 with real weights.
 * Includes the gather-based Q,K,V extraction.
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
  console.log('='.repeat(60));
  console.log('Full Attention Mechanism Test');
  console.log('='.repeat(60));

  const ok = await initWebGPU();
  if (!ok) throw new Error('Failed to initialize WebGPU');

  const api = new Torchlette('webgpu', {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Load real GPT-2 weights
  console.log('\nLoading real GPT-2 weights...');
  const buffer = fs.readFileSync(WEIGHTS_CACHE);
  const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));

  // Dimensions
  const embedDim = 768;
  const numHeads = 12;
  const headDim = embedDim / numHeads;
  const seqLen = 32;
  const batch = 1;
  const rank = 8;
  const scaling = 2.0;

  // Load embedding and attention weights
  const wteData = weights.get('wte.weight')!;
  const wte = api.tensorFromArray(wteData.data, wteData.shape, { device: 'webgpu', requiresGrad: false });

  // cAttn (QKV) weights
  const cAttnData = weights.get('h.0.attn.c_attn.weight')!;
  const cAttnBiasData = weights.get('h.0.attn.c_attn.bias')!;
  const transposed = transposeWeight(cAttnData.data, cAttnData.shape);
  const baseWeight = api.tensorFromArray(transposed.data, transposed.shape, { device: 'webgpu', requiresGrad: false });
  const baseBias = api.tensorFromArray(cAttnBiasData.data, cAttnBiasData.shape, { device: 'webgpu', requiresGrad: false });

  // cProj (output) weights
  const cProjData = weights.get('h.0.attn.c_proj.weight')!;
  const cProjBiasData = weights.get('h.0.attn.c_proj.bias')!;
  const cProjTransposed = transposeWeight(cProjData.data, cProjData.shape);
  const cProjWeight = api.tensorFromArray(cProjTransposed.data, cProjTransposed.shape, { device: 'webgpu', requiresGrad: false });
  const cProjBias = api.tensorFromArray(cProjBiasData.data, cProjBiasData.shape, { device: 'webgpu', requiresGrad: false });

  // LoRA parameters
  const loraAData = new Float32Array(rank * embedDim);
  for (let i = 0; i < loraAData.length; i++) {
    loraAData[i] = (Math.random() * 2 - 1) * 0.1;
  }
  const loraA = api.tensorFromArray(loraAData, [rank, embedDim], { device: 'webgpu', requiresGrad: true });
  const loraB = api.zeros([3 * embedDim, rank], { device: 'webgpu', requiresGrad: true });

  console.log(`loraA shape: [${loraA.shape}]`);
  console.log(`loraB sum: ${await loraB.sum().item()} (should be 0)`);

  // Create input via embedding
  const inputTokens = new Float32Array(seqLen);
  for (let i = 0; i < inputTokens.length; i++) {
    inputTokens[i] = Math.floor(Math.random() * 50257);
  }
  const input = api.tensorFromArray(inputTokens, [seqLen]);
  const expandedInput = input.reshape([seqLen, 1]).expand([seqLen, embedDim]).contiguous();
  const x = wte.gather(expandedInput, { dim: 0 }).reshape([batch, seqLen, embedDim]);

  console.log(`\nInput x shape: [${x.shape}]`);

  // LoRA-wrapped QKV projection
  console.log('\nComputing QKV with LoRA...');

  // Base path
  const baseOut = api.add(
    api.matmul(x, baseWeight.transpose({ dim0: 0, dim1: 1 })),
    baseBias
  );
  const detachedBase = baseOut.detach();

  // LoRA path
  const xA = api.matmul(x, loraA.transpose({ dim0: 0, dim1: 1 }));
  const loraOut = api.matmul(xA, loraB.transpose({ dim0: 0, dim1: 1 }));
  const scalingTensor = api.tensorFromArray([scaling], []);
  const scaledLora = api.mul(loraOut, scalingTensor);

  const qkv = api.add(detachedBase, scaledLora);
  console.log(`QKV shape: [${qkv.shape}]`);

  // Q, K, V extraction (like in GPT-2 attention)
  console.log('\nExtracting Q, K, V...');

  // Reshape to [batch, seqLen, 3, numHeads, headDim]
  const qkvReshaped = qkv.reshape([batch, seqLen, 3, numHeads, headDim]);
  // Permute to [3, batch, numHeads, seqLen, headDim]
  const qkvPermuted = qkvReshaped.permute([2, 0, 3, 1, 4]);
  // Reshape to [3, batch * numHeads * seqLen * headDim]
  const totalSize = batch * numHeads * seqLen * headDim;
  const qkvFlat = qkvPermuted.reshape([3, totalSize]);

  // Extract Q, K, V using gather
  const indices0 = api.tensorFromArray(Array(totalSize).fill(0), [1, totalSize]);
  const indices1 = api.tensorFromArray(Array(totalSize).fill(1), [1, totalSize]);
  const indices2 = api.tensorFromArray(Array(totalSize).fill(2), [1, totalSize]);

  const qFlat = api.gather(qkvFlat, indices0, { dim: 0 }).reshape([totalSize]);
  const kFlat = api.gather(qkvFlat, indices1, { dim: 0 }).reshape([totalSize]);
  const vFlat = api.gather(qkvFlat, indices2, { dim: 0 }).reshape([totalSize]);

  const q = qFlat.reshape([batch * numHeads, seqLen, headDim]);
  const k = kFlat.reshape([batch * numHeads, seqLen, headDim]);
  const v = vFlat.reshape([batch * numHeads, seqLen, headDim]);

  console.log(`Q shape: [${q.shape}]`);
  console.log(`K shape: [${k.shape}]`);
  console.log(`V shape: [${v.shape}]`);

  // Attention scores
  console.log('\nComputing attention...');
  const kT = k.transpose({ dim0: 1, dim1: 2 });
  const scores = api.matmul(q, kT);
  const scale = api.tensorFromArray([1.0 / Math.sqrt(headDim)], []);
  const scaledScores = api.mul(scores, scale);

  // Causal mask
  const mask = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      mask[i * seqLen + j] = j > i ? -1e9 : 0;
    }
  }
  const maskTensor = api.tensorFromArray(mask, [seqLen, seqLen]);
  const maskedScores = api.add(scaledScores, maskTensor);

  // Softmax
  const attnWeights = maskedScores.softmax(-1);

  // Apply to values
  const attnOut = api.matmul(attnWeights, v);

  // Reshape and project
  const attnReshaped = attnOut.reshape([batch, numHeads, seqLen, headDim]);
  const attnPermuted = attnReshaped.permute([0, 2, 1, 3]);
  const attnFlat = attnPermuted.reshape([batch, seqLen, embedDim]);

  // Output projection
  const output = api.add(
    api.matmul(attnFlat, cProjWeight.transpose({ dim0: 0, dim1: 1 })),
    cProjBias
  );
  console.log(`Output shape: [${output.shape}]`);

  // Simple loss
  const loss = output.mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(6)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();
  console.log('Backward complete');

  // Check LoRA gradients
  if (loraA.grad) {
    const gradSum = await loraA.grad.sum().item();
    console.log(`\nloraA.grad sum: ${gradSum}`);
    if (Number.isNaN(gradSum)) console.log('ERROR: loraA.grad is NaN!');
  }

  if (loraB.grad) {
    const gradSum = await loraB.grad.sum().item();
    console.log(`loraB.grad sum: ${gradSum}`);
    if (Number.isNaN(gradSum)) console.log('ERROR: loraB.grad is NaN!');
  }

  console.log('\n' + '='.repeat(60));
  console.log('Test Complete');
  console.log('='.repeat(60));
}

main().catch(console.error);
