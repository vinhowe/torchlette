#!/usr/bin/env npx tsx
/**
 * Test adding MLP step by step to find where NaN appears.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Torchlette, type FrontendTensor as Tensor } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { LoRALinear, createLoRAConfig, type LoRAConfig } from './src/lib/torchlette/lora';

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

class LayerNorm {
  readonly api: Torchlette;
  readonly weight: Tensor;
  readonly bias: Tensor;
  readonly eps: number;

  constructor(api: Torchlette, dim: number, device: 'cpu' | 'webgpu' = 'webgpu') {
    this.api = api;
    this.eps = 1e-5;
    this.weight = api.ones([dim], { device, requiresGrad: false });
    this.bias = api.zeros([dim], { device, requiresGrad: false });
  }

  forward(x: Tensor): Tensor {
    const inputShape = x.shape;
    const meanResult = this.api.mean(x, { dim: -1 }) as Tensor;
    const meanShape = [...inputShape.slice(0, -1), 1];
    const mean = meanResult.reshape(meanShape);

    const xCentered = this.api.sub(x, mean);

    const varianceResult = this.api.mean(this.api.mul(xCentered, xCentered), { dim: -1 }) as Tensor;
    const variance = varianceResult.reshape(meanShape);

    const epsTensor = this.api.tensorFromArray([this.eps], []);
    const std = this.api.sqrt(this.api.add(variance, epsTensor));
    const normalized = this.api.div(xCentered, std);

    return this.api.add(this.api.mul(normalized, this.weight), this.bias);
  }

  loadWeights(weight: Tensor, bias: Tensor): void {
    const runtime = this.api._runtime();
    runtime.copy_(this.weight._unwrap(), weight._unwrap());
    runtime.copy_(this.bias._unwrap(), bias._unwrap());
  }
}

class Embedding {
  readonly api: Torchlette;
  readonly weight: Tensor;
  readonly embeddingDim: number;

  constructor(api: Torchlette, numEmbeddings: number, embeddingDim: number, device: 'cpu' | 'webgpu' = 'webgpu') {
    this.api = api;
    this.embeddingDim = embeddingDim;
    this.weight = api.zeros([numEmbeddings, embeddingDim], { device, requiresGrad: false });
  }

  forward(indices: Tensor): Tensor {
    const inputShape = indices.shape;
    const numElements = inputShape.reduce((a, b) => a * b, 1);
    const flatInput = indices.reshape([numElements]);
    const expandedInput = flatInput.reshape([numElements, 1]).expand([numElements, this.embeddingDim]).contiguous();
    const gathered = this.weight.gather(expandedInput, { dim: 0 });
    const outputShape = [...inputShape, this.embeddingDim];
    return gathered.reshape(outputShape);
  }

  loadWeights(weight: Tensor): void {
    const runtime = this.api._runtime();
    runtime.copy_(this.weight._unwrap(), weight._unwrap());
  }
}

class Linear {
  readonly api: Torchlette;
  readonly weight: Tensor;
  readonly bias: Tensor | null;

  constructor(api: Torchlette, inFeatures: number, outFeatures: number, options: { device?: 'cpu' | 'webgpu'; bias?: boolean } = {}) {
    this.api = api;
    const device = options.device ?? 'webgpu';
    this.weight = api.zeros([outFeatures, inFeatures], { device, requiresGrad: false });
    this.bias = options.bias !== false ? api.zeros([outFeatures], { device, requiresGrad: false }) : null;
  }

  forward(x: Tensor): Tensor {
    const out = this.api.matmul(x, this.weight.transpose({ dim0: 0, dim1: 1 }));
    return this.bias ? this.api.add(out, this.bias) : out;
  }

  loadWeights(weight: Tensor, bias: Tensor | null): void {
    const runtime = this.api._runtime();
    runtime.copy_(this.weight._unwrap(), weight._unwrap());
    if (bias && this.bias) {
      runtime.copy_(this.bias._unwrap(), bias._unwrap());
    }
  }
}

class CausalSelfAttentionLoRA {
  readonly api: Torchlette;
  readonly numHeads: number;
  readonly embedDim: number;
  readonly headDim: number;
  readonly cAttn: LoRALinear;
  readonly cProj: Linear;

  constructor(api: Torchlette, embedDim: number, numHeads: number, loraConfig: LoRAConfig, device: 'cpu' | 'webgpu' = 'webgpu') {
    this.api = api;
    this.numHeads = numHeads;
    this.embedDim = embedDim;
    this.headDim = embedDim / numHeads;
    this.cAttn = new LoRALinear(api, embedDim, 3 * embedDim, loraConfig, { device });
    this.cProj = new Linear(api, embedDim, embedDim, { device });
  }

  forward(x: Tensor): Tensor {
    const [batch, seqLen] = x.shape;

    const qkv = this.cAttn.forward(x);
    const qkvReshaped = qkv.reshape([batch, seqLen, 3, this.numHeads, this.headDim]);
    const qkvPermuted = qkvReshaped.permute([2, 0, 3, 1, 4]);

    const totalSize = batch * this.numHeads * seqLen * this.headDim;
    const qkvFlat = qkvPermuted.reshape([3, totalSize]);

    const indices0 = this.api.tensorFromArray(Array(totalSize).fill(0), [1, totalSize]);
    const indices1 = this.api.tensorFromArray(Array(totalSize).fill(1), [1, totalSize]);
    const indices2 = this.api.tensorFromArray(Array(totalSize).fill(2), [1, totalSize]);

    const qFlat = this.api.gather(qkvFlat, indices0, { dim: 0 }).reshape([totalSize]);
    const kFlat = this.api.gather(qkvFlat, indices1, { dim: 0 }).reshape([totalSize]);
    const vFlat = this.api.gather(qkvFlat, indices2, { dim: 0 }).reshape([totalSize]);

    const q = qFlat.reshape([batch * this.numHeads, seqLen, this.headDim]);
    const k = kFlat.reshape([batch * this.numHeads, seqLen, this.headDim]);
    const v = vFlat.reshape([batch * this.numHeads, seqLen, this.headDim]);

    const kT = k.transpose({ dim0: 1, dim1: 2 });
    const scores = this.api.matmul(q, kT);

    const scale = this.api.tensorFromArray([1.0 / Math.sqrt(this.headDim)], []);
    const scaledScores = this.api.mul(scores, scale);

    const mask = this.createCausalMask(seqLen);
    const maskedScores = this.api.add(scaledScores, mask);

    const attnWeights = maskedScores.softmax(-1);
    const attnOut = this.api.matmul(attnWeights, v);

    const attnReshaped = attnOut.reshape([batch, this.numHeads, seqLen, this.headDim]);
    const attnPermuted = attnReshaped.permute([0, 2, 1, 3]);
    const attnFlat = attnPermuted.reshape([batch, seqLen, this.embedDim]);

    return this.cProj.forward(attnFlat);
  }

  private createCausalMask(seqLen: number): Tensor {
    const mask = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < seqLen; j++) {
        mask[i * seqLen + j] = j > i ? -1e9 : 0;
      }
    }
    return this.api.tensorFromArray(mask, [seqLen, seqLen]);
  }

  getLoRAParameters(): Tensor[] {
    return this.cAttn.getLoRAParameters();
  }
}

class MLP {
  readonly api: Torchlette;
  readonly cFc: Linear;
  readonly cProj: Linear;

  constructor(api: Torchlette, embedDim: number, device: 'cpu' | 'webgpu' = 'webgpu') {
    this.api = api;
    this.cFc = new Linear(api, embedDim, 4 * embedDim, { device });
    this.cProj = new Linear(api, 4 * embedDim, embedDim, { device });
  }

  forward(x: Tensor): Tensor {
    let h = this.cFc.forward(x);
    h = h.gelu();
    return this.cProj.forward(h);
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
  const vocabSize = 50257;
  const blockSize = 1024;
  const numHeads = 12;
  const seqLen = 32;
  const loraConfig = createLoRAConfig(8, 16);

  // Create components
  const wte = new Embedding(api, vocabSize, embedDim, 'webgpu');
  const wpe = new Embedding(api, blockSize, embedDim, 'webgpu');
  const ln1 = new LayerNorm(api, embedDim, 'webgpu');
  const attn = new CausalSelfAttentionLoRA(api, embedDim, numHeads, loraConfig, 'webgpu');
  const ln2 = new LayerNorm(api, embedDim, 'webgpu');
  const mlp = new MLP(api, embedDim, 'webgpu');
  const lnF = new LayerNorm(api, embedDim, 'webgpu');

  // Load weights
  console.log('Loading model weights...');

  const wteW = weights.get('wte.weight')!;
  wte.loadWeights(api.tensorFromArray(wteW.data, wteW.shape));

  const wpeW = weights.get('wpe.weight')!;
  wpe.loadWeights(api.tensorFromArray(wpeW.data, wpeW.shape));

  const ln1W = weights.get('h.0.ln_1.weight')!;
  const ln1B = weights.get('h.0.ln_1.bias')!;
  ln1.loadWeights(api.tensorFromArray(ln1W.data, ln1W.shape), api.tensorFromArray(ln1B.data, ln1B.shape));

  const cAttnW = weights.get('h.0.attn.c_attn.weight')!;
  const cAttnB = weights.get('h.0.attn.c_attn.bias')!;
  const cAttnWTransposed = transposeWeight(cAttnW.data, cAttnW.shape);
  attn.cAttn.loadBaseWeights(
    api.tensorFromArray(cAttnWTransposed.data, cAttnWTransposed.shape),
    api.tensorFromArray(cAttnB.data, cAttnB.shape)
  );

  const cProjW = weights.get('h.0.attn.c_proj.weight')!;
  const cProjB = weights.get('h.0.attn.c_proj.bias')!;
  const cProjWTransposed = transposeWeight(cProjW.data, cProjW.shape);
  attn.cProj.loadWeights(
    api.tensorFromArray(cProjWTransposed.data, cProjWTransposed.shape),
    api.tensorFromArray(cProjB.data, cProjB.shape)
  );

  const ln2W = weights.get('h.0.ln_2.weight')!;
  const ln2B = weights.get('h.0.ln_2.bias')!;
  ln2.loadWeights(api.tensorFromArray(ln2W.data, ln2W.shape), api.tensorFromArray(ln2B.data, ln2B.shape));

  const cFcW = weights.get('h.0.mlp.c_fc.weight')!;
  const cFcB = weights.get('h.0.mlp.c_fc.bias')!;
  const cFcWTransposed = transposeWeight(cFcW.data, cFcW.shape);
  mlp.cFc.loadWeights(
    api.tensorFromArray(cFcWTransposed.data, cFcWTransposed.shape),
    api.tensorFromArray(cFcB.data, cFcB.shape)
  );

  const mlpProjW = weights.get('h.0.mlp.c_proj.weight')!;
  const mlpProjB = weights.get('h.0.mlp.c_proj.bias')!;
  const mlpProjWTransposed = transposeWeight(mlpProjW.data, mlpProjW.shape);
  mlp.cProj.loadWeights(
    api.tensorFromArray(mlpProjWTransposed.data, mlpProjWTransposed.shape),
    api.tensorFromArray(mlpProjB.data, mlpProjB.shape)
  );

  const lnFW = weights.get('ln_f.weight')!;
  const lnFB = weights.get('ln_f.bias')!;
  lnF.loadWeights(api.tensorFromArray(lnFW.data, lnFW.shape), api.tensorFromArray(lnFB.data, lnFB.shape));

  // Create input/target
  const inputData = new Float32Array(seqLen);
  for (let i = 0; i < inputData.length; i++) {
    inputData[i] = Math.floor(Math.random() * vocabSize);
  }
  const input = api.tensorFromArray(inputData, [1, seqLen], { device: 'webgpu' });

  const targetData = new Float32Array(seqLen);
  for (let i = 0; i < targetData.length; i++) {
    targetData[i] = Math.floor(Math.random() * vocabSize);
  }
  const target = api.tensorFromArray(targetData, [1, seqLen], { device: 'webgpu' });

  console.log(`\n${'='.repeat(60)}`);
  console.log('GPT2 WITH MLP');
  console.log('='.repeat(60));

  // Forward pass with MLP
  const tokEmb = wte.forward(input);
  await checkStats('tokEmb', tokEmb);

  const positions = api.tensorFromArray(Array.from({ length: seqLen }, (_, i) => i), [seqLen]);
  const posEmb = wpe.forward(positions);
  await checkStats('posEmb', posEmb);

  let x = api.add(tokEmb, posEmb);
  await checkStats('x (tok + pos)', x);

  // Attention block
  const normed1 = ln1.forward(x);
  await checkStats('ln1(x)', normed1);

  const attnOut = attn.forward(normed1);
  await checkStats('attn(ln1(x))', attnOut);

  let h = api.add(x, attnOut); // Residual after attention
  await checkStats('h = x + attn', h);

  // MLP block
  const normed2 = ln2.forward(h);
  await checkStats('ln2(h)', normed2);

  const mlpOut = mlp.forward(normed2);
  await checkStats('mlp(ln2(h))', mlpOut);

  h = api.add(h, mlpOut); // Residual after MLP
  await checkStats('h = h + mlp', h);

  // Final layer norm
  x = lnF.forward(h);
  await checkStats('lnF(h)', x);

  // Logits (weight-tied)
  const logits = api.matmul(x, wte.weight.transpose({ dim0: 0, dim1: 1 }));
  await checkStats('logits', logits);

  // Cross-entropy loss
  const logitsFlat = logits.reshape([seqLen, vocabSize]);
  const targetFlat = target.reshape([seqLen]);

  const maxLogits = logitsFlat.max({ dim: -1, keepdim: true }) as Tensor;
  const shifted = api.sub(logitsFlat, maxLogits);
  const expShifted = shifted.exp();
  const sumExp = expShifted.sum({ dim: -1, keepdim: true }) as Tensor;
  const logSumExp = sumExp.log();
  const logSoftmax = api.sub(shifted, logSumExp);

  const targetsForGather = targetFlat.reshape([seqLen, 1]);
  const gatheredLogProbs = api.gather(logSoftmax, targetsForGather, { dim: 1 });
  const gatheredSqueezed = gatheredLogProbs.reshape([seqLen]);

  const loss = api.neg(gatheredSqueezed).mean() as Tensor;
  const lossVal = await loss.item();
  console.log(`\nLoss: ${lossVal.toFixed(4)}`);

  // Backward
  console.log('\nRunning backward...');
  await loss.backward();

  // Check LoRA gradients
  const loraParams = attn.getLoRAParameters();
  for (let i = 0; i < loraParams.length; i++) {
    const param = loraParams[i];
    const paramType = i % 2 === 0 ? 'loraA' : 'loraB';
    if (param.grad) {
      await checkStats(`${paramType}.grad`, param.grad);
    }
  }

  console.log('\n' + '='.repeat(60));
}

main().catch(console.error);
