/**
 * Qwen3 Model Implementation
 *
 * Llama-family decoder (Qwen3-0.6B/1.7B/4B dense): RMSNorm pre-norms, QK-norm
 * (per-head RMSNorm on q/k before RoPE), RoPE (half-split convention — matches
 * torchlette's fused kernel), GQA (numHeads query heads share numKVHeads K/V
 * heads), SwiGLU MLP, tied lm_head. No attention bias, no logit softcapping,
 * no sliding window — full causal attention on every layer.
 *
 * Inference-first: forward() supports an optional per-layer KV cache
 * (prefill/decode split) following the pattern in examples/gpt2/model.ts.
 * Steering hooks: forward() accepts an optional residual-stream intervention
 * callback applied between blocks.
 */

import type {
  DeviceKind,
  Tensor,
  Torchlette,
} from "torchlette";
import { Embedding, Linear, Module, ModuleList, RMSNorm } from "torchlette/nn";

// ============================================================================
// Configuration
// ============================================================================

export type Qwen3Config = {
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  intermediateSize: number;
  ropeTheta: number;
  rmsNormEps: number;
  /** Max sequence length we allocate RoPE tables / causal mask for. */
  maxSeqLen: number;
  /**
   * Dtype for the LINEAR weights. "f16" runs f32 activations against f16
   * weights via mixed-dtype matmul (1.7B: ~6.9GB → ~4.1GB total). The
   * embedding table and norm weights stay f32 regardless — the gather kernel
   * is f32-only, and norms are tiny. Default: "f32".
   */
  weightDtype?: "f32" | "f16";
};

/** Build a Qwen3Config from a HF config.json object. */
export function configFromHF(
  hf: Record<string, unknown>,
  maxSeqLen = 4096,
  weightDtype: "f32" | "f16" = "f32",
): Qwen3Config {
  return {
    vocabSize: hf.vocab_size as number,
    hiddenSize: hf.hidden_size as number,
    numLayers: hf.num_hidden_layers as number,
    numHeads: hf.num_attention_heads as number,
    numKVHeads: hf.num_key_value_heads as number,
    headDim: (hf.head_dim as number) ?? (hf.hidden_size as number) / (hf.num_attention_heads as number),
    intermediateSize: hf.intermediate_size as number,
    ropeTheta: (hf.rope_theta as number) ?? 1e6,
    rmsNormEps: (hf.rms_norm_eps as number) ?? 1e-6,
    maxSeqLen,
    weightDtype,
  };
}

// ============================================================================
// KV Cache & RoPE types
// ============================================================================

/** Per-layer KV cache, stored at numKVHeads (pre-GQA-expansion). */
export type KVCache = { k: Tensor; v: Tensor };

/** RoPE tables sliced to the current positions: [seqLen, headDim/2]. */
type RopeSlices = { cos: Tensor; sin: Tensor };

/**
 * Residual-stream intervention hook, applied to the hidden state after each
 * block. Return a replacement tensor (e.g. `x + alpha * direction`) or the
 * input unchanged. `layer` is the index of the block that just ran.
 */
export type ResidualHook = (x: Tensor, layer: number) => Tensor;

// ============================================================================
// Attention
// ============================================================================

export class Qwen3Attention extends Module {
  private readonly numHeads: number;
  private readonly numKVHeads: number;
  private readonly headDim: number;

  readonly qProj: Linear;
  readonly kProj: Linear;
  readonly vProj: Linear;
  readonly oProj: Linear;
  readonly qNorm: RMSNorm;
  readonly kNorm: RMSNorm;

  constructor(
    api: Torchlette,
    config: Qwen3Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    this.numHeads = config.numHeads;
    this.numKVHeads = config.numKVHeads;
    this.headDim = config.headDim;
    const device = options?.device;

    const qDim = config.numHeads * config.headDim;
    const kvDim = config.numKVHeads * config.headDim;
    const dtype = config.weightDtype ?? "f32";
    this.qProj = new Linear(api, config.hiddenSize, qDim, { bias: false, device, dtype });
    this.kProj = new Linear(api, config.hiddenSize, kvDim, { bias: false, device, dtype });
    this.vProj = new Linear(api, config.hiddenSize, kvDim, { bias: false, device, dtype });
    this.oProj = new Linear(api, qDim, config.hiddenSize, { bias: false, device, dtype });
    this.qNorm = new RMSNorm(api, config.headDim, { eps: config.rmsNormEps, device });
    this.kNorm = new RMSNorm(api, config.headDim, { eps: config.rmsNormEps, device });
  }

  /** GQA: [B, numKVHeads, S, D] -> [B, numHeads, S, D] (HF repeat_kv order). */
  private expandKV(t: Tensor): Tensor {
    const nRep = this.numHeads / this.numKVHeads;
    if (nRep === 1) return t;
    const [b, kvH, s, d] = t.shape;
    return t
      .unsqueeze(2)
      .expand([b, kvH, nRep, s, d])
      .contiguous()
      .reshape([b, kvH * nRep, s, d]);
  }

  forward(
    x: Tensor,
    rope: RopeSlices,
    pastKV?: KVCache,
  ): { out: Tensor; presentKV?: KVCache } {
    const [batch, seqLen, _hidden] = x.shape;

    // Project, split into heads, QK-norm (over headDim, before RoPE — HF order),
    // then [B, S, H, D] -> [B, H, S, D].
    const toHeads = (t: Tensor, numHeads: number, norm?: RMSNorm) => {
      let h = t.reshape([batch, seqLen, numHeads, this.headDim]);
      if (norm) h = norm.forward(h);
      return h.permute([0, 2, 1, 3]).contiguous();
    };
    let q = toHeads(this.qProj.forward(x), this.numHeads, this.qNorm);
    let k = toHeads(this.kProj.forward(x), this.numKVHeads, this.kNorm);
    let v = toHeads(this.vProj.forward(x), this.numKVHeads);

    // RoPE on q/k at the current positions (tables pre-sliced by caller).
    q = this.api.applyRoPE(q, rope.cos, rope.sin);
    k = this.api.applyRoPE(k, rope.cos, rope.sin);

    // KV cache: append along seq dim, stored at numKVHeads.
    let presentKV: KVCache | undefined;
    if (pastKV) {
      k = this.api.cat([pastKV.k, k], 2);
      v = this.api.cat([pastKV.v, v], 2);
    }
    presentKV = { k, v };

    // GQA expansion after caching (cache stays at numKVHeads).
    const kE = this.expandKV(k);
    const vE = this.expandKV(v);
    const kvSeqLen = kE.shape[2];

    const scale = 1.0 / Math.sqrt(this.headDim);
    let attnOutput: Tensor;
    if (!pastKV) {
      // Fused FlashAttention path (q/k same seq len).
      attnOutput = this.api.scaledDotProductAttention(q, kE, vE, scale, true);
    } else {
      // Decomposed path for cached decode (q seq len != kv seq len).
      // No GPU mask buffer + narrow here: offset views feed kernels wrong data
      // (see probe-narrow.ts). seqLen===1 needs no mask (all past visible);
      // multi-token-with-cache builds a small CPU mask fresh per call.
      const kT = kE.transpose({ dim0: 2, dim1: 3 });
      let scores = q.matmul(kT).mul(scale); // [B, H, S, kvS]
      if (seqLen > 1) {
        const maskArr = new Float32Array(seqLen * kvSeqLen);
        const qStart = kvSeqLen - seqLen;
        for (let i = 0; i < seqLen; i++) {
          for (let j = qStart + i + 1; j < kvSeqLen; j++) {
            maskArr[i * kvSeqLen + j] = -1e9;
          }
        }
        const mask = this.api.tensorFromArray(maskArr, [1, 1, seqLen, kvSeqLen]);
        scores = this.api.add(scores, mask);
      }
      const attnWeights = scores.softmax(-1);
      attnOutput = attnWeights.matmul(vE);
    }

    // [B, H, S, D] -> [B, S, H*D] -> o_proj
    const attnConcat = attnOutput
      .permute([0, 2, 1, 3])
      .contiguous()
      .reshape([batch, seqLen, this.numHeads * this.headDim]);
    return { out: this.oProj.forward(attnConcat), presentKV };
  }
}

// ============================================================================
// MLP (SwiGLU)
// ============================================================================

export class Qwen3MLP extends Module {
  readonly gateProj: Linear;
  readonly upProj: Linear;
  readonly downProj: Linear;

  constructor(
    api: Torchlette,
    config: Qwen3Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    const device = options?.device;
    const dtype = config.weightDtype ?? "f32";
    this.gateProj = new Linear(api, config.hiddenSize, config.intermediateSize, { bias: false, device, dtype });
    this.upProj = new Linear(api, config.hiddenSize, config.intermediateSize, { bias: false, device, dtype });
    this.downProj = new Linear(api, config.intermediateSize, config.hiddenSize, { bias: false, device, dtype });
  }

  forward(x: Tensor): Tensor {
    const gate = this.gateProj.forward(x).silu();
    const up = this.upProj.forward(x);
    return this.downProj.forward(this.api.mul(gate, up));
  }
}

// ============================================================================
// Transformer Block
// ============================================================================

export class Qwen3Block extends Module {
  readonly inputNorm: RMSNorm;
  readonly attn: Qwen3Attention;
  readonly postAttnNorm: RMSNorm;
  readonly mlp: Qwen3MLP;

  constructor(
    api: Torchlette,
    config: Qwen3Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    const device = options?.device;
    this.inputNorm = new RMSNorm(api, config.hiddenSize, { eps: config.rmsNormEps, device });
    this.attn = new Qwen3Attention(api, config, options);
    this.postAttnNorm = new RMSNorm(api, config.hiddenSize, { eps: config.rmsNormEps, device });
    this.mlp = new Qwen3MLP(api, config, options);
  }

  forward(
    x: Tensor,
    rope: RopeSlices,
    pastKV?: KVCache,
  ): { out: Tensor; presentKV?: KVCache } {
    const { out: attnOut, presentKV } = this.attn.forward(
      this.inputNorm.forward(x),
      rope,
      pastKV,
    );
    let h = this.api.add(x, attnOut);
    h = this.api.add(h, this.mlp.forward(this.postAttnNorm.forward(h)));
    return { out: h, presentKV };
  }
}

// ============================================================================
// Qwen3 Model
// ============================================================================

export type Qwen3ForwardOptions = {
  /** Per-layer KV cache from previous forward. */
  pastKVs?: KVCache[];
  /** Position offset (= past sequence length) for RoPE. */
  posOffset?: number;
  /** Residual-stream hook applied after each block (steering seam). */
  residualHook?: ResidualHook;
  /** Collect per-layer hidden states (embeddings + each block output). */
  collectHidden?: boolean;
};

export class Qwen3 extends Module {
  readonly config: Qwen3Config;
  readonly embedTokens: Embedding;
  readonly layers: ModuleList;
  readonly norm: RMSNorm;
  // RoPE tables kept CPU-side; the per-forward slice is uploaded fresh.
  // (GPU-side narrow(0, posOffset, S) views hit the offset-view readback bug —
  // see probe-narrow.ts — and the slices are tiny: seqLen × headDim/2 floats.)
  private readonly ropeCosArr: Float32Array;
  private readonly ropeSinArr: Float32Array;

  constructor(
    api: Torchlette,
    config: Qwen3Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    this.config = config;
    const device = options?.device;

    // Embedding stays f32: the gather kernel is f32-only, and the tied
    // lm_head matmul then runs as a plain f32 matmul.
    this.embedTokens = new Embedding(api, config.vocabSize, config.hiddenSize, { device });
    this.layers = new ModuleList(api);
    for (let i = 0; i < config.numLayers; i++) {
      this.layers.append(new Qwen3Block(api, config, options));
    }
    this.norm = new RMSNorm(api, config.hiddenSize, { eps: config.rmsNormEps, device });

    // Precompute RoPE tables [maxSeqLen, headDim/2] (half-split convention).
    const half = config.headDim / 2;
    this.ropeCosArr = new Float32Array(config.maxSeqLen * half);
    this.ropeSinArr = new Float32Array(config.maxSeqLen * half);
    for (let s = 0; s < config.maxSeqLen; s++) {
      for (let i = 0; i < half; i++) {
        const freq = 1 / config.ropeTheta ** ((2 * i) / config.headDim);
        const ang = s * freq;
        this.ropeCosArr[s * half + i] = Math.cos(ang);
        this.ropeSinArr[s * half + i] = Math.sin(ang);
      }
    }
  }

  /**
   * Forward pass. Returns logits [batch, seqLen, vocabSize], the per-layer
   * KV cache, and (optionally) per-layer hidden states.
   */
  forward(
    idx: Tensor,
    options?: Qwen3ForwardOptions,
  ): { logits: Tensor; presentKVs: KVCache[]; hidden?: Tensor[] } {
    const [_batch, seqLen] = idx.shape;
    const posOffset = options?.posOffset ?? 0;
    if (posOffset + seqLen > this.config.maxSeqLen) {
      throw new Error(
        `Sequence length ${posOffset + seqLen} exceeds maxSeqLen ${this.config.maxSeqLen}`,
      );
    }

    const half = this.config.headDim / 2;
    const rope: RopeSlices = {
      cos: this.api.tensorFromArray(
        this.ropeCosArr.subarray(posOffset * half, (posOffset + seqLen) * half),
        [seqLen, half],
      ),
      sin: this.api.tensorFromArray(
        this.ropeSinArr.subarray(posOffset * half, (posOffset + seqLen) * half),
        [seqLen, half],
      ),
    };

    // Embedding output is f32 (f32 table); with weightDtype "f16" the linears
    // run mixed-dtype (f32 activations × f16 weights → f32).
    let x = this.embedTokens.forward(idx);
    const hidden: Tensor[] | undefined = options?.collectHidden ? [x] : undefined;

    const presentKVs: KVCache[] = [];
    for (let i = 0; i < this.layers.length; i++) {
      const block = this.layers.get(i) as Qwen3Block;
      const { out, presentKV } = block.forward(x, rope, options?.pastKVs?.[i]);
      x = out;
      if (options?.residualHook) x = options.residualHook(x, i);
      if (presentKV) presentKVs.push(presentKV);
      hidden?.push(x);
    }

    x = this.norm.forward(x);
    // Tied lm_head: logits = x @ embed.weight^T
    const logits = this.api.linear(x, this.embedTokens.weight, null);
    return { logits, presentKVs, hidden };
  }
}
