/**
 * GPT-2 Model with LoRA adapters for efficient fine-tuning.
 *
 * This is a browser-compatible implementation that:
 * - Loads base weights from HuggingFace
 * - Applies LoRA to attention layers (cAttn)
 * - Only trains LoRA parameters (base model is frozen)
 */

import type { FrontendTensor as Tensor, Torchlette } from 'torchlette';
import { checkpoint } from '../../../../../src/nn/checkpoint';
import { LoRALinear, type LoRAConfig } from './lora';

// ============================================================================
// Configuration
// ============================================================================

export type GPT2Config = {
  vocabSize: number;
  blockSize: number;
  numLayers: number;
  numHeads: number;
  embedDim: number;
  dropoutRate: number;
};

export const GPT2_SMALL_CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0.0, // Disable dropout for inference/training stability
};

// ============================================================================
// Layer Implementations
// ============================================================================

/**
 * Layer Normalization
 */
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
    // Compute mean along last dimension
    // Note: We manually add the keepdim dimension via reshape for compatibility
    const inputShape = x.shape;
    const meanResult = this.api.mean(x, { dim: -1 }) as Tensor;
    const meanShape = [...inputShape.slice(0, -1), 1];
    const mean = meanResult.reshape(meanShape);

    const xCentered = this.api.sub(x, mean);

    // Compute variance
    const varianceResult = this.api.mean(this.api.mul(xCentered, xCentered), { dim: -1 }) as Tensor;
    const variance = varianceResult.reshape(meanShape);

    // Normalize
    const epsTensor = this.api.tensorFromArray([this.eps], []);
    const std = this.api.sqrt(this.api.add(variance, epsTensor));
    const normalized = this.api.div(xCentered, std);

    // Scale and shift
    return this.api.add(this.api.mul(normalized, this.weight), this.bias);
  }

  loadWeights(weight: Tensor, bias: Tensor): void {
    const runtime = this.api._runtime();
    runtime.copy_(this.weight._unwrap(), weight._unwrap());
    runtime.copy_(this.bias._unwrap(), bias._unwrap());
  }
}

/**
 * Embedding layer - proper implementation matching torchlette's nn.Embedding
 */
class Embedding {
  readonly api: Torchlette;
  readonly weight: Tensor;
  readonly embeddingDim: number;

  constructor(
    api: Torchlette,
    numEmbeddings: number,
    embeddingDim: number,
    device: 'cpu' | 'webgpu' = 'webgpu'
  ) {
    this.api = api;
    this.embeddingDim = embeddingDim;
    this.weight = api.zeros([numEmbeddings, embeddingDim], {
      device,
      requiresGrad: false,
    });
  }

  forward(indices: Tensor): Tensor {
    // indices: [...] containing token indices
    // weight: [numEmbeddings, embeddingDim]
    // output: [..., embeddingDim]

    const inputShape = indices.shape;
    const numElements = inputShape.reduce((a, b) => a * b, 1);

    // Flatten input to [numElements]
    const flatInput = indices.reshape([numElements]);

    // Expand indices to [numElements, embeddingDim] for gather
    // Each index is repeated embeddingDim times across dim 1
    const expandedInput = flatInput
      .reshape([numElements, 1])
      .expand([numElements, this.embeddingDim])
      .contiguous(); // Required: gather doesn't handle strided tensors correctly

    // Gather from weight: output[i][j] = weight[expandedInput[i][j]][j]
    const gathered = this.weight.gather(expandedInput, { dim: 0 });

    // Reshape to [..., embeddingDim]
    const outputShape = [...inputShape, this.embeddingDim];
    return gathered.reshape(outputShape);
  }

  loadWeights(weight: Tensor): void {
    const runtime = this.api._runtime();
    runtime.copy_(this.weight._unwrap(), weight._unwrap());
  }
}

/**
 * Linear layer (frozen base)
 */
class Linear {
  readonly api: Torchlette;
  readonly weight: Tensor;
  readonly bias: Tensor | null;

  constructor(
    api: Torchlette,
    inFeatures: number,
    outFeatures: number,
    options: { device?: 'cpu' | 'webgpu'; bias?: boolean } = {}
  ) {
    this.api = api;
    const device = options.device ?? 'webgpu';
    this.weight = api.zeros([outFeatures, inFeatures], {
      device,
      requiresGrad: false,
    });
    this.bias = options.bias !== false
      ? api.zeros([outFeatures], { device, requiresGrad: false })
      : null;
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

// ============================================================================
// Attention with LoRA
// ============================================================================

/**
 * Causal Self-Attention with LoRA on the combined QKV projection.
 */
class CausalSelfAttentionLoRA {
  readonly api: Torchlette;
  readonly numHeads: number;
  readonly embedDim: number;
  readonly headDim: number;

  // LoRA-wrapped QKV projection
  readonly cAttn: LoRALinear;
  // Frozen output projection
  readonly cProj: Linear;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    loraConfig: LoRAConfig,
    device: 'cpu' | 'webgpu' = 'webgpu'
  ) {
    this.api = api;
    this.numHeads = config.numHeads;
    this.embedDim = config.embedDim;
    this.headDim = config.embedDim / config.numHeads;

    // LoRA on combined QKV projection
    this.cAttn = new LoRALinear(
      api,
      config.embedDim,
      3 * config.embedDim,
      loraConfig,
      { device }
    );

    // Frozen output projection
    this.cProj = new Linear(api, config.embedDim, config.embedDim, { device });
  }

  forward(x: Tensor): Tensor {
    const [batch, seqLen] = x.shape;

    // Combined QKV projection with LoRA
    // qkv: [batch, seqLen, 3 * embedDim]
    const qkv = this.cAttn.forward(x);

    // Reshape to [batch, seqLen, 3, numHeads, headDim]
    const qkvReshaped = qkv.reshape([
      batch,
      seqLen,
      3,
      this.numHeads,
      this.headDim,
    ]);

    // Permute to [3, batch, numHeads, seqLen, headDim]
    const qkvPermuted = qkvReshaped.permute([2, 0, 3, 1, 4]);

    // Reshape to [3, batch * numHeads * seqLen * headDim]
    const totalSize = batch * this.numHeads * seqLen * this.headDim;
    const qkvFlat = qkvPermuted.reshape([3, totalSize]);

    // Extract Q, K, V using gather along dim 0
    // indices shape must match output shape: [1, totalSize]
    // Then we squeeze the first dimension
    const indices0 = this.api.tensorFromArray(
      Array(totalSize).fill(0),
      [1, totalSize]
    );
    const indices1 = this.api.tensorFromArray(
      Array(totalSize).fill(1),
      [1, totalSize]
    );
    const indices2 = this.api.tensorFromArray(
      Array(totalSize).fill(2),
      [1, totalSize]
    );

    // Gather along dim 0: output shape is [1, totalSize], then squeeze to [totalSize]
    // Then reshape to [batch * numHeads, seqLen, headDim]
    const qFlat = this.api.gather(qkvFlat, indices0, { dim: 0 }).reshape([totalSize]);
    const kFlat = this.api.gather(qkvFlat, indices1, { dim: 0 }).reshape([totalSize]);
    const vFlat = this.api.gather(qkvFlat, indices2, { dim: 0 }).reshape([totalSize]);

    const q = qFlat.reshape([batch * this.numHeads, seqLen, this.headDim]);
    const k = kFlat.reshape([batch * this.numHeads, seqLen, this.headDim]);
    const v = vFlat.reshape([batch * this.numHeads, seqLen, this.headDim]);

    // Scaled dot-product attention
    // scores = Q @ K^T / sqrt(d_k)
    const kT = k.transpose({ dim0: 1, dim1: 2 }); // [batch*heads, head_dim, seq]
    const scores = this.api.matmul(q, kT);

    const scale = this.api.tensorFromArray([1.0 / Math.sqrt(this.headDim)], []);
    const scaledScores = this.api.mul(scores, scale);

    // Apply causal mask (set future positions to -inf)
    const mask = this.createCausalMask(seqLen);
    const maskedScores = this.api.add(scaledScores, mask);

    // Softmax
    const attnWeights = maskedScores.softmax(-1);

    // Apply attention to values
    const attnOut = this.api.matmul(attnWeights, v);

    // Reshape back: [batch*heads, seq, head_dim] -> [batch, seq, embed_dim]
    const attnReshaped = attnOut.reshape([
      batch,
      this.numHeads,
      seqLen,
      this.headDim,
    ]);
    const attnPermuted = attnReshaped.permute([0, 2, 1, 3]);
    const attnFlat = attnPermuted.reshape([batch, seqLen, this.embedDim]);

    // Output projection
    return this.cProj.forward(attnFlat);
  }

  private createCausalMask(seqLen: number): Tensor {
    // Create lower triangular mask: 0 for valid, -inf for invalid
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

// ============================================================================
// MLP (frozen)
// ============================================================================

class MLP {
  readonly api: Torchlette;
  readonly cFc: Linear;
  readonly cProj: Linear;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    device: 'cpu' | 'webgpu' = 'webgpu'
  ) {
    this.api = api;
    this.cFc = new Linear(api, config.embedDim, 4 * config.embedDim, { device });
    this.cProj = new Linear(api, 4 * config.embedDim, config.embedDim, { device });
  }

  forward(x: Tensor): Tensor {
    let h = this.cFc.forward(x);
    h = h.gelu();
    return this.cProj.forward(h);
  }
}

// ============================================================================
// Transformer Block with LoRA
// ============================================================================

class TransformerBlockLoRA {
  readonly api: Torchlette;
  readonly ln1: LayerNorm;
  readonly attn: CausalSelfAttentionLoRA;
  readonly ln2: LayerNorm;
  readonly mlp: MLP;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    loraConfig: LoRAConfig,
    device: 'cpu' | 'webgpu' = 'webgpu'
  ) {
    this.api = api;
    this.ln1 = new LayerNorm(api, config.embedDim, device);
    this.attn = new CausalSelfAttentionLoRA(api, config, loraConfig, device);
    this.ln2 = new LayerNorm(api, config.embedDim, device);
    this.mlp = new MLP(api, config, device);
  }

  forward(x: Tensor): Tensor {
    // Pre-norm attention
    const attnOut = this.attn.forward(this.ln1.forward(x));
    let h = this.api.add(x, attnOut);

    // Pre-norm MLP
    const mlpOut = this.mlp.forward(this.ln2.forward(h));
    h = this.api.add(h, mlpOut);

    return h;
  }

  getLoRAParameters(): Tensor[] {
    return this.attn.getLoRAParameters();
  }
}

// ============================================================================
// GPT-2 with LoRA
// ============================================================================

export class GPT2WithLoRA {
  readonly api: Torchlette;
  readonly config: GPT2Config;
  readonly loraConfig: LoRAConfig;

  readonly wte: Embedding;
  readonly wpe: Embedding;
  readonly h: TransformerBlockLoRA[];
  readonly lnF: LayerNorm;

  private training = false;
  private checkpointingEnabled = false;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    loraConfig: LoRAConfig,
    device: 'cpu' | 'webgpu' = 'webgpu'
  ) {
    this.api = api;
    this.config = config;
    this.loraConfig = loraConfig;

    this.wte = new Embedding(api, config.vocabSize, config.embedDim, device);
    this.wpe = new Embedding(api, config.blockSize, config.embedDim, device);

    this.h = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.h.push(new TransformerBlockLoRA(api, config, loraConfig, device));
    }

    this.lnF = new LayerNorm(api, config.embedDim, device);
  }

  /**
   * Set training mode.
   */
  train(mode = true): void {
    this.training = mode;
  }

  /**
   * Set evaluation mode.
   */
  eval(): void {
    this.training = false;
  }

  /**
   * Enable or disable gradient checkpointing.
   * When enabled, activations are not saved during forward pass.
   * Instead, they are recomputed during backward pass.
   * This trades compute for memory.
   */
  enableCheckpointing(enabled: boolean): void {
    this.checkpointingEnabled = enabled;
  }

  /**
   * Forward pass.
   *
   * @param input - Token indices [batch, seqLen]
   * @returns Logits [batch, seqLen, vocabSize]
   */
  forward(input: Tensor): Tensor {
    const [_batch, seqLen] = input.shape;

    // Token embeddings
    const tokEmb = this.wte.forward(input);

    // Position embeddings
    const positions = this.api.tensorFromArray(
      Array.from({ length: seqLen }, (_, i) => i),
      [seqLen]
    );
    const posEmb = this.wpe.forward(positions);

    // Combine embeddings
    let x = this.api.add(tokEmb, posEmb);

    // Transformer blocks - with optional checkpointing
    if (this.checkpointingEnabled && this.training) {
      // With checkpointing: wrap each block to recompute during backward
      for (let i = 0; i < this.h.length; i++) {
        const block = this.h[i];
        x = this.checkpointBlock(block, x);
      }
    } else {
      // Without checkpointing: normal forward pass
      for (const block of this.h) {
        x = block.forward(x);
      }
    }

    // Final layer norm
    x = this.lnF.forward(x);

    // LM head (weight-tied with token embeddings)
    // logits = x @ wte.weight^T
    const logits = this.api.matmul(
      x,
      this.wte.weight.transpose({ dim0: 0, dim1: 1 })
    );

    return logits;
  }

  /**
   * Run a transformer block with gradient checkpointing.
   * During forward: only save the input, not intermediate activations.
   * During backward: recompute the forward pass to get activations.
   */
  private checkpointBlock(block: TransformerBlockLoRA, x: Tensor): Tensor {
    // Use the proper checkpoint API from torchlette/nn/checkpoint
    return checkpoint(this.api, (input: Tensor) => block.forward(input), [x]);
  }

  /**
   * Forward with loss computation.
   */
  forwardWithLoss(
    input: Tensor,
    target: Tensor
  ): { logits: Tensor; loss: Tensor } {
    const logits = this.forward(input);

    // Compute cross-entropy loss
    const [batch, seqLen, vocabSize] = logits.shape;

    // Reshape for loss: [batch * seqLen, vocabSize]
    const logitsFlat = logits.reshape([batch * seqLen, vocabSize]);
    const targetFlat = target.reshape([batch * seqLen]);

    // Cross-entropy loss
    const loss = this.crossEntropyLoss(logitsFlat, targetFlat);

    return { logits, loss };
  }

  private crossEntropyLoss(logits: Tensor, targets: Tensor): Tensor {
    // Numerically stable cross-entropy loss
    // logits: [N, C], targets: [N]
    // loss = -log_softmax(logits)[targets].mean()
    //
    // log_softmax = logits - max(logits) - log(sum(exp(logits - max(logits))))
    // This is equivalent to -log(softmax(logits)[targets]) but numerically stable

    // Step 1: Compute log-softmax along the last dimension
    const dim = -1;
    const maxLogits = logits.max({ dim, keepdim: true });
    if (typeof maxLogits === 'number') {
      throw new Error('crossEntropyLoss: max with keepdim should return tensor');
    }
    const shifted = this.api.sub(logits, maxLogits);
    const expShifted = shifted.exp();
    const sumExp = expShifted.sum({ dim, keepdim: true });
    if (typeof sumExp === 'number') {
      throw new Error('crossEntropyLoss: sum with keepdim should return tensor');
    }
    const logSumExp = sumExp.log();
    const logSoftmax = this.api.sub(shifted, logSumExp);

    // Step 2: Gather the log-softmax values at target indices
    // targets is [N] but gather needs same rank as input [N, C]
    // So we unsqueeze targets to [N, 1], gather gives [N, 1], then reshape to [N]
    const targetsForGather = targets.reshape([targets.shape[0], 1]);
    const gatheredLogProbs = this.api.gather(logSoftmax, targetsForGather, { dim: 1 });
    const gatheredSqueezed = gatheredLogProbs.reshape(targets.shape);

    // Step 3: Negate and mean
    const loss = this.api.neg(gatheredSqueezed);
    const meanLoss = loss.mean();
    if (typeof meanLoss === 'number') {
      return this.api.tensorFromArray([meanLoss], []);
    }
    return meanLoss;
  }

  /**
   * Get all trainable LoRA parameters.
   */
  getLoRAParameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const block of this.h) {
      params.push(...block.getLoRAParameters());
    }
    return params;
  }

  /**
   * Load base weights from HuggingFace format.
   */
  loadBaseWeights(weights: Map<string, { data: Float32Array; shape: number[] }>): void {
    // Token embeddings
    const wteWeight = weights.get('wte.weight');
    if (wteWeight) {
      const tensor = this.api.tensorFromArray(wteWeight.data, wteWeight.shape);
      this.wte.loadWeights(tensor);
    }

    // Position embeddings
    const wpeWeight = weights.get('wpe.weight');
    if (wpeWeight) {
      const tensor = this.api.tensorFromArray(wpeWeight.data, wpeWeight.shape);
      this.wpe.loadWeights(tensor);
    }

    // Final layer norm
    const lnFWeight = weights.get('ln_f.weight');
    const lnFBias = weights.get('ln_f.bias');
    if (lnFWeight && lnFBias) {
      const w = this.api.tensorFromArray(lnFWeight.data, lnFWeight.shape);
      const b = this.api.tensorFromArray(lnFBias.data, lnFBias.shape);
      this.lnF.loadWeights(w, b);
    }

    // Transformer blocks
    for (let i = 0; i < this.h.length; i++) {
      const block = this.h[i];
      const prefix = `h.${i}`;

      // Layer norm 1
      const ln1W = weights.get(`${prefix}.ln_1.weight`);
      const ln1B = weights.get(`${prefix}.ln_1.bias`);
      if (ln1W && ln1B) {
        block.ln1.loadWeights(
          this.api.tensorFromArray(ln1W.data, ln1W.shape),
          this.api.tensorFromArray(ln1B.data, ln1B.shape)
        );
      }

      // Attention cAttn (LoRA base weights)
      const cAttnW = weights.get(`${prefix}.attn.c_attn.weight`);
      const cAttnB = weights.get(`${prefix}.attn.c_attn.bias`);
      if (cAttnW && cAttnB) {
        // Note: HuggingFace stores as [in, out], we need [out, in]
        const transposed = this.transposeWeight(cAttnW.data, cAttnW.shape);
        const wTensor = this.api.tensorFromArray(transposed.data, transposed.shape);
        const bTensor = this.api.tensorFromArray(cAttnB.data, cAttnB.shape);
        block.attn.cAttn.loadBaseWeights(wTensor, bTensor);
      }

      // Attention cProj
      const cProjW = weights.get(`${prefix}.attn.c_proj.weight`);
      const cProjB = weights.get(`${prefix}.attn.c_proj.bias`);
      if (cProjW && cProjB) {
        const transposed = this.transposeWeight(cProjW.data, cProjW.shape);
        const wTensor = this.api.tensorFromArray(transposed.data, transposed.shape);
        const bTensor = this.api.tensorFromArray(cProjB.data, cProjB.shape);
        block.attn.cProj.loadWeights(wTensor, bTensor);
      }

      // Layer norm 2
      const ln2W = weights.get(`${prefix}.ln_2.weight`);
      const ln2B = weights.get(`${prefix}.ln_2.bias`);
      if (ln2W && ln2B) {
        block.ln2.loadWeights(
          this.api.tensorFromArray(ln2W.data, ln2W.shape),
          this.api.tensorFromArray(ln2B.data, ln2B.shape)
        );
      }

      // MLP cFc
      const cFcW = weights.get(`${prefix}.mlp.c_fc.weight`);
      const cFcB = weights.get(`${prefix}.mlp.c_fc.bias`);
      if (cFcW && cFcB) {
        const transposed = this.transposeWeight(cFcW.data, cFcW.shape);
        block.mlp.cFc.loadWeights(
          this.api.tensorFromArray(transposed.data, transposed.shape),
          this.api.tensorFromArray(cFcB.data, cFcB.shape)
        );
      }

      // MLP cProj
      const mlpProjW = weights.get(`${prefix}.mlp.c_proj.weight`);
      const mlpProjB = weights.get(`${prefix}.mlp.c_proj.bias`);
      if (mlpProjW && mlpProjB) {
        const transposed = this.transposeWeight(mlpProjW.data, mlpProjW.shape);
        block.mlp.cProj.loadWeights(
          this.api.tensorFromArray(transposed.data, transposed.shape),
          this.api.tensorFromArray(mlpProjB.data, mlpProjB.shape)
        );
      }
    }
  }

  private transposeWeight(
    data: Float32Array,
    shape: number[]
  ): { data: Float32Array; shape: number[] } {
    // Transpose 2D weight from [in, out] to [out, in]
    const [rows, cols] = shape;
    const transposed = new Float32Array(data.length);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        transposed[j * rows + i] = data[i * cols + j];
      }
    }
    return { data: transposed, shape: [cols, rows] };
  }
}
