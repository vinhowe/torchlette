/**
 * GPT-2 Model Implementation
 *
 * A PyTorch-style implementation of GPT-2 using torchlette.
 * Based on the original OpenAI GPT-2 architecture.
 */

import type { Tensor, Torchlette, DeviceKind } from "../../src/frontend";
import {
  Module,
  Linear,
  Embedding,
  LayerNorm,
  Dropout,
  crossEntropy,
  checkpoint,
} from "../../src/nn";

// ============================================================================
// Configuration
// ============================================================================

export type GPT2Config = {
  vocabSize: number;
  blockSize: number; // max sequence length
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
  dropoutRate: 0.1,
};

export const GPT2_MEDIUM_CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 24,
  numHeads: 16,
  embedDim: 1024,
  dropoutRate: 0.1,
};

export const GPT2_LARGE_CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 36,
  numHeads: 20,
  embedDim: 1280,
  dropoutRate: 0.1,
};

export const DISTILGPT2_CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 6, // Key difference: 6 layers vs 12 for GPT-2 small
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0.1,
};

// ============================================================================
// Causal Self-Attention
// ============================================================================

/**
 * Multi-head causal (masked) self-attention.
 *
 * Uses combined QKV projection for efficiency, applies causal mask to prevent
 * attending to future positions.
 */
export class CausalSelfAttention extends Module {
  private readonly numHeads: number;
  private readonly embedDim: number;
  private readonly headDim: number;
  private readonly dropoutRate: number;

  readonly cAttn: Linear; // Combined Q, K, V projection [embedDim, 3 * embedDim]
  readonly cProj: Linear; // Output projection [embedDim, embedDim]
  readonly attnDropout: Dropout;
  readonly residDropout: Dropout;
  declare readonly causalBias: Tensor; // Cached causal mask [1, 1, blockSize, blockSize]

  constructor(
    api: Torchlette,
    config: GPT2Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    this.numHeads = config.numHeads;
    this.embedDim = config.embedDim;
    this.headDim = config.embedDim / config.numHeads;
    this.dropoutRate = config.dropoutRate;

    if (this.embedDim % this.numHeads !== 0) {
      throw new Error(
        `embedDim (${this.embedDim}) must be divisible by numHeads (${this.numHeads})`,
      );
    }

    const device = options?.device;

    // Combined QKV projection
    this.cAttn = new Linear(api, config.embedDim, 3 * config.embedDim, {
      device,
    });

    // Output projection
    this.cProj = new Linear(api, config.embedDim, config.embedDim, { device });

    this.attnDropout = new Dropout(api, { p: config.dropoutRate });
    this.residDropout = new Dropout(api, { p: config.dropoutRate });

    // Cache causal mask: upper-triangular -1e9 values (0 on/below diagonal)
    const causalBias = api.triu(api.full([1, 1, config.blockSize, config.blockSize], -1e9), 1);
    this.registerBuffer("causalBias", causalBias);

    // Register child modules for recursive train()/eval()
    this.registerModule("cAttn", this.cAttn);
    this.registerModule("cProj", this.cProj);
    this.registerModule("attnDropout", this.attnDropout);
    this.registerModule("residDropout", this.residDropout);
  }

  /**
   * Forward pass for causal self-attention.
   *
   * @param x - Input tensor of shape [batch, seqLen, embedDim]
   * @returns Output tensor of shape [batch, seqLen, embedDim]
   */
  forward(x: Tensor): Tensor {
    const [batch, seqLen, _embedDim] = x.shape;

    // Combined QKV projection: [batch, seqLen, 3 * embedDim]
    const qkv = this.cAttn.forward(x);

    // Split QKV using narrow (zero-cost view ops)
    // qkv: [batch, seqLen, 3 * embedDim]
    // Reshape to [batch, seqLen, 3, embedDim], then narrow along dim 2
    const qkvFor3 = qkv.reshape([batch, seqLen, 3, this.embedDim]);
    const qSlice = qkvFor3.narrow(2, 0, 1); // [batch, seqLen, 1, embedDim] view
    const kSlice = qkvFor3.narrow(2, 1, 1); // [batch, seqLen, 1, embedDim] view
    const vSlice = qkvFor3.narrow(2, 2, 1); // [batch, seqLen, 1, embedDim] view

    // Reshape narrow views to multi-head layout
    // inferReshapeStrides handles the non-contiguous narrow views (zero-cost)
    const q = qSlice
      .reshape([batch, seqLen, this.numHeads, this.headDim])
      .permute([0, 2, 1, 3])
      .contiguous();
    const k = kSlice
      .reshape([batch, seqLen, this.numHeads, this.headDim])
      .permute([0, 2, 1, 3])
      .contiguous();
    const v = vSlice
      .reshape([batch, seqLen, this.numHeads, this.headDim])
      .permute([0, 2, 1, 3])
      .contiguous();

    // Scaled dot-product attention
    // Use fused FlashAttention when dropout is disabled (eval mode or rate=0)
    let attnOutput: Tensor;
    if (!this.attnDropout.training || this.dropoutRate === 0) {
      // Fused path: single kernel for Q@K^T + scale + causal_mask + softmax + attn@V
      const scale = 1.0 / Math.sqrt(this.headDim);
      attnOutput = this.api.scaledDotProductAttention(q, k, v, scale, true);
    } else {
      // Decomposed path (needed when dropout is active)
      const kT = k.transpose({ dim0: 2, dim1: 3 });
      const scores = q.matmul(kT);
      const scaledScores = scores.mul(1.0 / Math.sqrt(this.headDim));

      const mask = this.causalBias.narrow(2, 0, seqLen).narrow(3, 0, seqLen);
      const maskedScores = this.api.add(scaledScores, mask);
      const attnWeights = maskedScores.softmax(-1);
      const attnDropped = this.attnDropout.forward(attnWeights);
      attnOutput = attnDropped.matmul(v);
    }

    // Concatenate heads: [batch, numHeads, seqLen, headDim] -> [batch, seqLen, embedDim]
    // Note: permute creates non-contiguous tensor, need contiguous before reshape
    const attnConcat = attnOutput
      .permute([0, 2, 1, 3])
      .contiguous()
      .reshape([batch, seqLen, this.embedDim]);

    // Output projection
    const output = this.cProj.forward(attnConcat);

    // Residual dropout
    return this.residDropout.forward(output);
  }

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    return [...this.cAttn.parameters(), ...this.cProj.parameters()];
  }
}

// ============================================================================
// MLP (Feed-Forward Network)
// ============================================================================

/**
 * Two-layer MLP with GELU activation.
 */
export class MLP extends Module {
  readonly cFc: Linear; // Expansion: [embedDim, 4 * embedDim]
  readonly cProj: Linear; // Contraction: [4 * embedDim, embedDim]
  readonly dropout: Dropout;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    const device = options?.device;

    this.cFc = new Linear(api, config.embedDim, 4 * config.embedDim, {
      device,
    });
    this.cProj = new Linear(api, 4 * config.embedDim, config.embedDim, {
      device,
    });
    this.dropout = new Dropout(api, { p: config.dropoutRate });

    // Register child modules for recursive train()/eval()
    this.registerModule("cFc", this.cFc);
    this.registerModule("cProj", this.cProj);
    this.registerModule("dropout", this.dropout);
  }

  /**
   * Forward pass: fc -> gelu -> proj -> dropout
   */
  forward(x: Tensor): Tensor {
    let h = this.cFc.forward(x);
    h = h.gelu();
    h = this.cProj.forward(h);
    return this.dropout.forward(h);
  }

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    return [...this.cFc.parameters(), ...this.cProj.parameters()];
  }
}

// ============================================================================
// Transformer Block
// ============================================================================

/**
 * A single transformer block with pre-norm architecture.
 */
export class TransformerBlock extends Module {
  readonly ln1: LayerNorm;
  readonly attn: CausalSelfAttention;
  readonly ln2: LayerNorm;
  readonly mlp: MLP;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    const device = options?.device;

    this.ln1 = new LayerNorm(api, config.embedDim, { device });
    this.attn = new CausalSelfAttention(api, config, { device });
    this.ln2 = new LayerNorm(api, config.embedDim, { device });
    this.mlp = new MLP(api, config, { device });

    // Register child modules for recursive train()/eval()
    this.registerModule("ln1", this.ln1);
    this.registerModule("attn", this.attn);
    this.registerModule("ln2", this.ln2);
    this.registerModule("mlp", this.mlp);
  }

  /**
   * Forward pass: Pre-norm architecture
   * x = x + attn(ln1(x))
   * x = x + mlp(ln2(x))
   */
  forward(x: Tensor): Tensor {
    // Attention block with residual
    const attnOut = this.attn.forward(this.ln1.forward(x));
    let h = this.api.add(x, attnOut);

    // MLP block with residual
    const mlpOut = this.mlp.forward(this.ln2.forward(h));
    h = this.api.add(h, mlpOut);

    return h;
  }

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    return [
      ...this.ln1.parameters(),
      ...this.attn.parameters(),
      ...this.ln2.parameters(),
      ...this.mlp.parameters(),
    ];
  }
}

// ============================================================================
// GPT-2 Model
// ============================================================================

/**
 * GPT-2 Language Model.
 */
export class GPT2 extends Module {
  readonly config: GPT2Config;

  readonly wte: Embedding; // Token embeddings [vocabSize, embedDim]
  readonly wpe: Embedding; // Position embeddings [blockSize, embedDim]
  readonly drop: Dropout;
  readonly h: TransformerBlock[]; // Transformer blocks
  readonly lnF: LayerNorm; // Final layer norm
  declare readonly posIndices: Tensor; // Cached position indices [1, blockSize]
  // Note: lmHead shares weights with wte (weight tying)

  constructor(
    api: Torchlette,
    config: GPT2Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    this.config = config;
    const device = options?.device;

    this.wte = new Embedding(api, config.vocabSize, config.embedDim, {
      device,
    });
    this.wpe = new Embedding(api, config.blockSize, config.embedDim, {
      device,
    });
    this.drop = new Dropout(api, { p: config.dropoutRate });

    // Create transformer blocks
    this.h = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.h.push(new TransformerBlock(api, config, { device }));
    }

    this.lnF = new LayerNorm(api, config.embedDim, { device });

    // Cache position indices: [0, 1, 2, ..., blockSize-1] reshaped to [1, blockSize]
    const posIndices = api.arange(config.blockSize).reshape([1, config.blockSize]);
    this.registerBuffer("posIndices", posIndices);

    // Register child modules for recursive train()/eval()
    this.registerModule("wte", this.wte);
    this.registerModule("wpe", this.wpe);
    this.registerModule("drop", this.drop);
    for (let i = 0; i < this.h.length; i++) {
      this.registerModule(`h.${i}`, this.h[i]);
    }
    this.registerModule("lnF", this.lnF);
  }

  /**
   * Forward pass - returns logits only.
   * Use forwardWithLoss for training with loss computation.
   *
   * @param idx - Token indices of shape [batch, seqLen]
   * @param options - Optional forward options
   * @returns Logits tensor of shape [batch, seqLen, vocabSize]
   */
  forward(idx: Tensor, options?: { useCheckpoint?: boolean }): Tensor {
    return this.forwardWithLoss(idx, undefined, options).logits;
  }

  /**
   * Forward pass with optional loss computation.
   *
   * @param idx - Token indices of shape [batch, seqLen]
   * @param targets - Optional target indices for loss computation [batch, seqLen]
   * @param options - Optional forward options (useCheckpoint for memory savings)
   * @returns Object with logits and optional loss
   */
  forwardWithLoss(
    idx: Tensor,
    targets?: Tensor,
    options?: { useCheckpoint?: boolean },
  ): { logits: Tensor; loss: Tensor | null } {
    const [_batch, seqLen] = idx.shape;
    const useCheckpoint = options?.useCheckpoint ?? false;

    if (seqLen > this.config.blockSize) {
      throw new Error(
        `Sequence length ${seqLen} exceeds block size ${this.config.blockSize}`,
      );
    }

    // Slice cached position indices to current sequence length (zero-cost narrow view)
    const pos = this.posIndices.narrow(1, 0, seqLen);

    // Token embeddings: [batch, seqLen, embedDim]
    const tokEmb = this.wte.forward(idx);

    // Position embeddings: [1, seqLen, embedDim] -> broadcasts to [batch, seqLen, embedDim]
    const posEmb = this.wpe.forward(pos);

    // Combine embeddings
    let x = this.api.add(tokEmb, posEmb);
    x = this.drop.forward(x);

    // Pass through transformer blocks (with optional checkpointing)
    if (useCheckpoint) {
      // Checkpoint each transformer block to save memory during backward pass
      // Activations are recomputed during backward instead of stored
      for (let i = 0; i < this.h.length; i++) {
        const block = this.h[i];
        x = checkpoint(
          this.api,
          (input: Tensor) => block.forward(input),
          [x],
        );
      }
    } else {
      // Standard forward pass - store all activations
      for (const block of this.h) {
        x = block.forward(x);
      }
    }

    // Final layer norm
    x = this.lnF.forward(x);

    // LM head: project to vocabulary (weight tied with token embeddings)
    // logits = x @ wte.weight^T  (using linear() for optimized backward)
    // linear() backward computes dW = dY^T @ X directly in weight's shape,
    // avoiding an explicit transpose that would prevent epilogue fusion.
    const logits = this.api.linear(x, this.wte.weight, null);

    // Compute loss if targets provided
    let loss: Tensor | null = null;
    if (targets !== undefined) {
      // Cross-entropy loss
      // logits: [batch, seqLen, vocabSize]
      // targets: [batch, seqLen]
      // Flatten for cross-entropy: logits [batch*seqLen, vocabSize], targets [batch*seqLen]
      const [batch, seqLenT] = targets.shape;
      const flatLogits = logits.reshape([batch * seqLenT, this.config.vocabSize]);
      const flatTargets = targets.reshape([batch * seqLenT]);
      loss = crossEntropy(this.api, flatLogits, flatTargets);
    }

    return { logits, loss };
  }

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    const params: Tensor[] = [];
    params.push(...this.wte.parameters());
    params.push(...this.wpe.parameters());
    for (const block of this.h) {
      params.push(...block.parameters());
    }
    params.push(...this.lnF.parameters());
    // Note: lmHead weight is tied to wte.weight, so we don't add it separately
    return params;
  }

}
