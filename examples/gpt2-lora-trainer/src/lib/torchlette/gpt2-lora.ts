/**
 * GPT-2 Model with LoRA adapters for efficient fine-tuning.
 *
 * This is a browser-compatible implementation that:
 * - Loads base weights from HuggingFace
 * - Applies LoRA to attention layers (cAttn)
 * - Only trains LoRA parameters (base model is frozen)
 */

import type { FrontendTensor as Tensor, Torchlette } from "torchlette";
import { nn } from "torchlette";

const { checkpoint, crossEntropy, Embedding, LayerNorm, Linear } = nn;

import { type LoRAConfig, LoRALinear } from "./lora";

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

// LayerNorm: imported from torchlette framework (fused GPU kernel).
// The framework's LayerNorm creates weight/bias with requiresGrad=true.
// For LoRA mode, setFullFinetuning(false) freezes them after construction.

/** Copy pretrained weights into a module's weight (and optional bias). */
function loadWeights(
  api: Torchlette,
  mod: { weight: Tensor | null; bias?: Tensor | null },
  weight: Tensor,
  bias?: Tensor | null,
): void {
  const runtime = api._runtime();
  if (mod.weight) runtime.copy_(mod.weight._unwrap(), weight._unwrap());
  if (bias && mod.bias) runtime.copy_(mod.bias._unwrap(), bias._unwrap());
}

// Embedding, Linear, LayerNorm: imported from torchlette framework.
// Framework modules use fused GPU kernels and optimized backward passes.

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
    device: "cpu" | "webgpu" = "webgpu",
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
      { device },
    );

    // Frozen output projection
    this.cProj = new Linear(api, config.embedDim, config.embedDim, { device });
    this.cProj.weight.requires_grad_(false);
    this.cProj.bias!.requires_grad_(false);
  }

  forward(x: Tensor): Tensor {
    const [batch, seqLen] = x.shape;

    // Combined QKV projection with LoRA
    const qkv = this.cAttn.forward(x);

    // Split and reshape to [batch, numHeads, seqLen, headDim]
    const [qFlat, kFlat, vFlat] = qkv.chunk(3, -1);
    const toHeads = (t: Tensor) =>
      t
        .reshape([batch, seqLen, this.numHeads, this.headDim])
        .permute([0, 2, 1, 3])
        .contiguous();
    const q = toHeads(qFlat);
    const k = toHeads(kFlat);
    const v = toHeads(vFlat);

    // Fused scaled dot-product attention (single kernel)
    const scale = 1.0 / Math.sqrt(this.headDim);
    const attnOut = this.api.scaledDotProductAttention(q, k, v, scale, true);

    // Reshape back: [batch, numHeads, seqLen, headDim] → [batch, seqLen, embedDim]
    const attnFlat = attnOut
      .permute([0, 2, 1, 3])
      .reshape([batch, seqLen, this.embedDim]);

    // Output projection
    return this.cProj.forward(attnFlat);
  }

  getLoRAParameters(): Tensor[] {
    return this.cAttn.getLoRAParameters();
  }
}

// ============================================================================
// MLP with LoRA on the expansion layer
// ============================================================================

class MLPLoRA {
  readonly api: Torchlette;
  readonly cFc: LoRALinear; // LoRA on expansion — controls feature activation
  readonly cProj: Linear; // Frozen projection back

  constructor(
    api: Torchlette,
    config: GPT2Config,
    loraConfig: LoRAConfig,
    device: "cpu" | "webgpu" = "webgpu",
  ) {
    this.api = api;
    this.cFc = new LoRALinear(
      api,
      config.embedDim,
      4 * config.embedDim,
      loraConfig,
      { device },
    );
    this.cProj = new Linear(api, 4 * config.embedDim, config.embedDim, {
      device,
    });
    this.cProj.weight.requires_grad_(false);
    this.cProj.bias!.requires_grad_(false);
  }

  forward(x: Tensor): Tensor {
    let h = this.cFc.forward(x);
    h = h.gelu();
    return this.cProj.forward(h);
  }

  getLoRAParameters(): Tensor[] {
    return this.cFc.getLoRAParameters();
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
  readonly mlp: MLPLoRA;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    loraConfig: LoRAConfig,
    device: "cpu" | "webgpu" = "webgpu",
  ) {
    this.api = api;
    this.ln1 = new LayerNorm(api, config.embedDim, { device });
    this.ln1.weight!.requires_grad_(false);
    this.ln1.bias!.requires_grad_(false);
    this.attn = new CausalSelfAttentionLoRA(api, config, loraConfig, device);
    this.ln2 = new LayerNorm(api, config.embedDim, { device });
    this.ln2.weight!.requires_grad_(false);
    this.ln2.bias!.requires_grad_(false);
    this.mlp = new MLPLoRA(api, config, loraConfig, device);
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
    return [...this.attn.getLoRAParameters(), ...this.mlp.getLoRAParameters()];
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
  /** When true, checkpoint entire blocks (more recompute, less memory). */
  fullCheckpoint = false;

  constructor(
    api: Torchlette,
    config: GPT2Config,
    loraConfig: LoRAConfig,
    device: "cpu" | "webgpu" = "webgpu",
  ) {
    this.api = api;
    this.config = config;
    this.loraConfig = loraConfig;

    this.wte = new Embedding(api, config.vocabSize, config.embedDim, {
      device,
    });
    this.wte.weight.requires_grad_(false);
    this.wpe = new Embedding(api, config.blockSize, config.embedDim, {
      device,
    });
    this.wpe.weight.requires_grad_(false);

    this.h = [];
    for (let i = 0; i < config.numLayers; i++) {
      this.h.push(new TransformerBlockLoRA(api, config, loraConfig, device));
    }

    this.lnF = new LayerNorm(api, config.embedDim, { device });
    this.lnF.weight!.requires_grad_(false);
    this.lnF.bias!.requires_grad_(false);
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
      [seqLen],
    );
    const posEmb = this.wpe.forward(positions);

    // Combine embeddings
    let x = this.api.add(tokEmb, posEmb);

    // Transformer blocks
    if (this.checkpointingEnabled && this.training) {
      for (const block of this.h) {
        if (this.fullCheckpoint) {
          // Full checkpointing: recompute entire block during backward.
          // Trades more compute for less memory — enables larger batches.
          x = checkpoint(this.api, (input: Tensor) => block.forward(input), [
            x,
          ]);
        } else {
          // Selective checkpointing: attention runs outside checkpoint (fused
          // kernel is already memory-efficient), MLP runs inside checkpoint
          // (recomputes 4x-expanded activations). This avoids 6 expensive
          // fusedAttention recomputes during backward.
          const attnOut = block.attn.forward(block.ln1.forward(x));
          const h = this.api.add(x, attnOut);
          x = checkpoint(
            this.api,
            (input: Tensor) => {
              const mlpOut = block.mlp.forward(block.ln2.forward(input));
              return this.api.add(input, mlpOut);
            },
            [h],
          );
        }
      }
    } else {
      for (const block of this.h) {
        x = block.forward(x);
      }
    }

    // Final layer norm
    x = this.lnF.forward(x);

    // LM head (weight-tied with token embeddings)
    const logits = this.api.linear(x, this.wte.weight, null);

    return logits;
  }

  /**
   * Forward with loss computation.
   */
  forwardWithLoss(
    input: Tensor,
    target: Tensor,
  ): { logits: Tensor; loss: Tensor } {
    const logits = this.forward(input);

    // Compute cross-entropy loss
    const [batch, seqLen, vocabSize] = logits.shape;

    // Reshape for loss: [batch * seqLen, vocabSize]
    const logitsFlat = logits.reshape([batch * seqLen, vocabSize]);
    const targetFlat = target.reshape([batch * seqLen]);

    // Cross-entropy loss using the framework's built-in implementation
    // (has correct backward, unlike the previous hand-rolled version)
    const loss = crossEntropy(this.api, logitsFlat, targetFlat);

    return { logits, loss };
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
   * Get ALL trainable parameters (for full finetuning).
   * Includes embeddings, layer norms, all linear weights, and LoRA params.
   */
  getAllParameters(): Tensor[] {
    const params: Tensor[] = [];
    // Embeddings
    params.push(this.wte.weight, this.wpe.weight);
    // Transformer blocks
    for (const block of this.h) {
      // Layer norms
      params.push(block.ln1.weight!, block.ln1.bias!);
      params.push(block.ln2.weight!, block.ln2.bias!);
      // Attention: LoRA base weight + LoRA params + output proj
      params.push(block.attn.cAttn.baseWeight);
      if (block.attn.cAttn.baseBias) params.push(block.attn.cAttn.baseBias);
      params.push(...block.attn.cAttn.getLoRAParameters());
      params.push(block.attn.cProj.weight);
      if (block.attn.cProj.bias) params.push(block.attn.cProj.bias);
      // MLP: LoRA base weight + LoRA params + output proj
      params.push(block.mlp.cFc.baseWeight);
      if (block.mlp.cFc.baseBias) params.push(block.mlp.cFc.baseBias);
      params.push(...block.mlp.cFc.getLoRAParameters());
      params.push(block.mlp.cProj.weight);
      if (block.mlp.cProj.bias) params.push(block.mlp.cProj.bias);
    }
    // Final layer norm
    params.push(this.lnF.weight!, this.lnF.bias!);
    return params;
  }

  /**
   * Enable or disable full finetuning (make all base weights trainable).
   */
  setFullFinetuning(enabled: boolean): void {
    // Embeddings
    this.wte.weight.requires_grad_(enabled);
    this.wpe.weight.requires_grad_(enabled);
    // Transformer blocks
    for (const block of this.h) {
      block.ln1.weight!.requires_grad_(enabled);
      block.ln1.bias!.requires_grad_(enabled);
      block.ln2.weight!.requires_grad_(enabled);
      block.ln2.bias!.requires_grad_(enabled);
      block.attn.cAttn.baseWeight.requires_grad_(enabled);
      if (block.attn.cAttn.baseBias)
        block.attn.cAttn.baseBias.requires_grad_(enabled);
      block.attn.cProj.weight.requires_grad_(enabled);
      if (block.attn.cProj.bias) block.attn.cProj.bias.requires_grad_(enabled);
      block.mlp.cFc.baseWeight.requires_grad_(enabled);
      if (block.mlp.cFc.baseBias)
        block.mlp.cFc.baseBias.requires_grad_(enabled);
      block.mlp.cProj.weight.requires_grad_(enabled);
      if (block.mlp.cProj.bias) block.mlp.cProj.bias.requires_grad_(enabled);
    }
    this.lnF.weight!.requires_grad_(enabled);
    this.lnF.bias!.requires_grad_(enabled);
  }

  /**
   * Load base weights from HuggingFace format.
   */
  loadBaseWeights(
    weights: Map<string, { data: Float32Array; shape: number[] }>,
  ): void {
    // Token embeddings
    const wteWeight = weights.get("wte.weight");
    if (wteWeight) {
      const tensor = this.api.tensorFromArray(wteWeight.data, wteWeight.shape);
      loadWeights(this.api, this.wte, tensor);
    }

    // Position embeddings
    const wpeWeight = weights.get("wpe.weight");
    if (wpeWeight) {
      const tensor = this.api.tensorFromArray(wpeWeight.data, wpeWeight.shape);
      loadWeights(this.api, this.wpe, tensor);
    }

    // Final layer norm
    const lnFWeight = weights.get("ln_f.weight");
    const lnFBias = weights.get("ln_f.bias");
    if (lnFWeight && lnFBias) {
      const w = this.api.tensorFromArray(lnFWeight.data, lnFWeight.shape);
      const b = this.api.tensorFromArray(lnFBias.data, lnFBias.shape);
      loadWeights(this.api, this.lnF, w, b);
    }

    // Transformer blocks
    for (let i = 0; i < this.h.length; i++) {
      const block = this.h[i];
      const prefix = `h.${i}`;

      // Layer norm 1
      const ln1W = weights.get(`${prefix}.ln_1.weight`);
      const ln1B = weights.get(`${prefix}.ln_1.bias`);
      if (ln1W && ln1B) {
        loadWeights(
          this.api,
          block.ln1,
          this.api.tensorFromArray(ln1W.data, ln1W.shape),
          this.api.tensorFromArray(ln1B.data, ln1B.shape),
        );
      }

      // Attention cAttn (LoRA base weights)
      const cAttnW = weights.get(`${prefix}.attn.c_attn.weight`);
      const cAttnB = weights.get(`${prefix}.attn.c_attn.bias`);
      if (cAttnW && cAttnB) {
        // Note: HuggingFace stores as [in, out], we need [out, in]
        const transposed = this.transposeWeight(cAttnW.data, cAttnW.shape);
        const wTensor = this.api.tensorFromArray(
          transposed.data,
          transposed.shape,
        );
        const bTensor = this.api.tensorFromArray(cAttnB.data, cAttnB.shape);
        block.attn.cAttn.loadBaseWeights(wTensor, bTensor);
      }

      // Attention cProj
      const cProjW = weights.get(`${prefix}.attn.c_proj.weight`);
      const cProjB = weights.get(`${prefix}.attn.c_proj.bias`);
      if (cProjW && cProjB) {
        const transposed = this.transposeWeight(cProjW.data, cProjW.shape);
        const wTensor = this.api.tensorFromArray(
          transposed.data,
          transposed.shape,
        );
        const bTensor = this.api.tensorFromArray(cProjB.data, cProjB.shape);
        loadWeights(this.api, block.attn.cProj, wTensor, bTensor);
      }

      // Layer norm 2
      const ln2W = weights.get(`${prefix}.ln_2.weight`);
      const ln2B = weights.get(`${prefix}.ln_2.bias`);
      if (ln2W && ln2B) {
        loadWeights(
          this.api,
          block.ln2,
          this.api.tensorFromArray(ln2W.data, ln2W.shape),
          this.api.tensorFromArray(ln2B.data, ln2B.shape),
        );
      }

      // MLP cFc (LoRA-wrapped)
      const cFcW = weights.get(`${prefix}.mlp.c_fc.weight`);
      const cFcB = weights.get(`${prefix}.mlp.c_fc.bias`);
      if (cFcW && cFcB) {
        const transposed = this.transposeWeight(cFcW.data, cFcW.shape);
        block.mlp.cFc.loadBaseWeights(
          this.api.tensorFromArray(transposed.data, transposed.shape),
          this.api.tensorFromArray(cFcB.data, cFcB.shape),
        );
      }

      // MLP cProj
      const mlpProjW = weights.get(`${prefix}.mlp.c_proj.weight`);
      const mlpProjB = weights.get(`${prefix}.mlp.c_proj.bias`);
      if (mlpProjW && mlpProjB) {
        const transposed = this.transposeWeight(mlpProjW.data, mlpProjW.shape);
        loadWeights(
          this.api,
          block.mlp.cProj,
          this.api.tensorFromArray(transposed.data, transposed.shape),
          this.api.tensorFromArray(mlpProjB.data, mlpProjB.shape),
        );
      }
    }
  }

  private transposeWeight(
    data: Float32Array,
    shape: number[],
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
