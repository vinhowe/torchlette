/**
 * Gemma-2 Model Implementation (2B).
 *
 * Gemma-2 decoder, ported from the Qwen3 stack. Deltas vs Qwen3
 * (verified against transformers Gemma2 modeling, 2026-07-10):
 *  - Attention: logit SOFT-CAPPING (cap=50) on every layer + SLIDING WINDOW
 *    (4096) on the LOCAL layers. Gemma-2's `layer_types` alternate
 *    sliding_attention / full_attention starting with SLIDING at layer 0
 *    (EVEN layers = local/windowed, ODD = global). Both declared via the #64
 *    attention modifiers: {scoreMod:{kind:"softcap",cap:50},
 *    maskMods:[{kind:"causal"} (+ {kind:"slidingWindow",window:4096} on local)]}.
 *  - Attention scale = query_pre_attn_scalar**-0.5 (= 1/sqrt(256) for 2B),
 *    NOT 1/sqrt(headDim). They coincide for 2B (head_dim=256) but we key off
 *    the config value to stay correct for other sizes.
 *  - RMSNorm SANDWICH: 4 norms/layer — input + post-attention (applied to the
 *    attention OUTPUT before the residual add) and pre- + post-feedforward
 *    (post-ffn applied to the MLP output before the residual add). Gemma's
 *    RMSNorm uses (1 + weight); we bake +1 into the loaded weight so the
 *    stock fused RMSNorm kernel (x_normed * weight) is exactly correct.
 *  - GeGLU MLP: gelu(gate, approximate:"tanh") * up (vs Qwen3's SiLU SwiGLU).
 *  - Embedding scaling: hidden = embed(ids) * sqrt(hiddenSize).
 *  - FINAL-LOGIT soft-capping (cap=30): plain elementwise cap*tanh(logits/cap)
 *    on the lm_head output (NOT an attention modifier).
 *  - Tied lm_head, RoPE (base 10000), GQA (8 heads / 4 KV heads).
 *
 * Inference-first; mirrors Qwen3's forward/forwardStatic + residualHook seam.
 */

import type {
  AttnModifierSpec,
  DeviceKind,
  Tensor,
  Torchlette,
} from "torchlette";
import { Embedding, Linear, Module, ModuleList, RMSNorm } from "torchlette/nn";

// ============================================================================
// Configuration
// ============================================================================

export type Gemma2Config = {
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  intermediateSize: number;
  ropeTheta: number;
  rmsNormEps: number;
  /** Attention logit soft-cap (Gemma-2: 50.0). */
  attnLogitSoftcap: number;
  /** Final lm_head logit soft-cap (Gemma-2: 30.0). */
  finalLogitSoftcap: number;
  /** Sliding-window size for the local layers (Gemma-2: 4096). */
  slidingWindow: number;
  /** Attention scale denominator: 1/sqrt(queryPreAttnScalar). */
  queryPreAttnScalar: number;
  /** Per-layer attention type; EVEN = sliding/local, ODD = full/global. */
  layerTypes: ("sliding_attention" | "full_attention")[];
  /** Max sequence length we allocate RoPE tables / KV cache for. */
  maxSeqLen: number;
  /** Dtype for the LINEAR weights ("f16" runs mixed-dtype matmul). Default f32. */
  weightDtype?: "f32" | "f16";
};

/** Build a Gemma2Config from a HF config.json object. */
export function configFromHF(
  hf: Record<string, unknown>,
  maxSeqLen = 4096,
  weightDtype: "f32" | "f16" = "f32",
): Gemma2Config {
  const numLayers = hf.num_hidden_layers as number;
  // Gemma-2 alternates sliding/full starting with SLIDING at layer 0.
  const layerTypes =
    (hf.layer_types as ("sliding_attention" | "full_attention")[]) ??
    Array.from({ length: numLayers }, (_, i) =>
      i % 2 === 0 ? "sliding_attention" : "full_attention",
    );
  return {
    vocabSize: hf.vocab_size as number,
    hiddenSize: hf.hidden_size as number,
    numLayers,
    numHeads: hf.num_attention_heads as number,
    numKVHeads: hf.num_key_value_heads as number,
    headDim:
      (hf.head_dim as number) ??
      (hf.hidden_size as number) / (hf.num_attention_heads as number),
    intermediateSize: hf.intermediate_size as number,
    ropeTheta: (hf.rope_theta as number) ?? 1e4,
    rmsNormEps: (hf.rms_norm_eps as number) ?? 1e-6,
    attnLogitSoftcap: (hf.attn_logit_softcapping as number) ?? 50.0,
    finalLogitSoftcap: (hf.final_logit_softcapping as number) ?? 30.0,
    slidingWindow: (hf.sliding_window as number) ?? 4096,
    queryPreAttnScalar:
      (hf.query_pre_attn_scalar as number) ??
      (hf.head_dim as number) ??
      (hf.hidden_size as number) / (hf.num_attention_heads as number),
    layerTypes,
    maxSeqLen,
    weightDtype,
  };
}

/** The attention modifier for a given layer (soft-cap always; window on local). */
export function layerModifier(
  config: Gemma2Config,
  layerIdx: number,
): AttnModifierSpec {
  const isSliding = config.layerTypes[layerIdx] === "sliding_attention";
  const maskMods: AttnModifierSpec["maskMods"] = [{ kind: "causal" }];
  if (isSliding) {
    maskMods.push({ kind: "slidingWindow", window: config.slidingWindow });
  }
  return {
    scoreMod: { kind: "softcap", cap: config.attnLogitSoftcap },
    maskMods,
  };
}

// ============================================================================
// KV Cache & RoPE types
// ============================================================================

export type KVCache = { k: Tensor; v: Tensor };

export type StaticKV = {
  k: Tensor[];
  v: Tensor[];
  len: number;
  maxSeqLen: number;
};

export const KV_BUCKET = 128;

export function kvBucketLen(len: number, maxSeqLen: number): number {
  return Math.min(Math.ceil(len / KV_BUCKET) * KV_BUCKET, maxSeqLen);
}

type RopeSlices = { cos: Tensor; sin: Tensor };

export type ResidualHook = (x: Tensor, layer: number) => Tensor;

// ============================================================================
// Attention
// ============================================================================

export class Gemma2Attention extends Module {
  private readonly numHeads: number;
  private readonly numKVHeads: number;
  private readonly headDim: number;
  private readonly scale: number;
  private readonly modifier: AttnModifierSpec;
  private readonly slidingWindow: number | null;

  readonly qProj: Linear;
  readonly kProj: Linear;
  readonly vProj: Linear;
  readonly oProj: Linear;

  constructor(
    api: Torchlette,
    config: Gemma2Config,
    layerIdx: number,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    this.numHeads = config.numHeads;
    this.numKVHeads = config.numKVHeads;
    this.headDim = config.headDim;
    this.scale = 1.0 / Math.sqrt(config.queryPreAttnScalar);
    this.modifier = layerModifier(config, layerIdx);
    this.slidingWindow =
      config.layerTypes[layerIdx] === "sliding_attention"
        ? config.slidingWindow
        : null;
    const device = options?.device;

    const qDim = config.numHeads * config.headDim;
    const kvDim = config.numKVHeads * config.headDim;
    const dtype = config.weightDtype ?? "f32";
    this.qProj = new Linear(api, config.hiddenSize, qDim, {
      bias: false,
      device,
      dtype,
    });
    this.kProj = new Linear(api, config.hiddenSize, kvDim, {
      bias: false,
      device,
      dtype,
    });
    this.vProj = new Linear(api, config.hiddenSize, kvDim, {
      bias: false,
      device,
      dtype,
    });
    this.oProj = new Linear(api, qDim, config.hiddenSize, {
      bias: false,
      device,
      dtype,
    });
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

  /** Elementwise soft-cap: cap * tanh(x / cap). Used for the decode-path
   *  decomposed attention scores (the fused kernel injects it via scoreMod). */
  private softcapScores(scores: Tensor, cap: number): Tensor {
    return scores.div(cap).tanh().mul(cap);
  }

  /**
   * Static-cache attention step. See Qwen3Attention.forwardStatic.
   * Gemma-2 deltas: soft-cap on decode scores; sliding-window additive mask
   * on the local layers (window bound over the bucketed prefix). No QK-norm.
   */
  forwardStatic(
    x: Tensor,
    rope: RopeSlices,
    ctx: {
      kSlot: Tensor;
      vSlot: Tensor;
      scatterIdx: Tensor;
      bucketLen: number;
      mask: Tensor | null;
      posOffset: number;
    },
  ): { out: Tensor } {
    const [batch, seqLen, _hidden] = x.shape;
    const toHeads = (t: Tensor, numHeads: number) => {
      const h = t.reshape([batch, seqLen, numHeads, this.headDim]);
      if (seqLen === 1) return h.reshape([batch, numHeads, 1, this.headDim]);
      return h.permute([0, 2, 1, 3]).contiguous();
    };
    let q = toHeads(this.qProj.forward(x), this.numHeads);
    let k = toHeads(this.kProj.forward(x), this.numKVHeads);
    const v = toHeads(this.vProj.forward(x), this.numKVHeads);
    q = this.api.applyRoPE(q, rope.cos, rope.sin);
    k = this.api.applyRoPE(k, rope.cos, rope.sin);

    ctx.kSlot.copy_(ctx.kSlot.scatterAdd(ctx.scatterIdx, k, { dim: 2 }));
    ctx.vSlot.copy_(ctx.vSlot.scatterAdd(ctx.scatterIdx, v, { dim: 2 }));

    let attnOutput: Tensor;
    if (ctx.mask === null) {
      // Prefill from position 0: fresh K/V are the whole context. Fused kernel
      // applies soft-cap + causal (+ window) via the layer modifier.
      attnOutput = this.api.scaledDotProductAttention(
        q,
        this.expandKV(k),
        this.expandKV(v),
        this.scale,
        true,
        this.modifier,
      );
    } else {
      // Decode: decomposed matmul+softmax over the bucketed prefix. Apply
      // soft-cap on the raw scores, then the additive mask (padding + window).
      const kE = this.expandKV(ctx.kSlot.narrow(2, 0, ctx.bucketLen));
      const vE = this.expandKV(ctx.vSlot.narrow(2, 0, ctx.bucketLen));
      let scores = q.matmul(kE.transpose({ dim0: 2, dim1: 3 })).mul(this.scale);
      scores = this.softcapScores(scores, this.modifier.scoreMod!.cap);
      const attnWeights = this.api.add(scores, ctx.mask).softmax(-1);
      attnOutput = attnWeights.matmul(vE);
    }

    const attnConcat = attnOutput
      .permute([0, 2, 1, 3])
      .contiguous()
      .reshape([batch, seqLen, this.numHeads * this.headDim]);
    return { out: this.oProj.forward(attnConcat) };
  }

  forward(
    x: Tensor,
    rope: RopeSlices,
    pastKV?: KVCache,
  ): { out: Tensor; presentKV?: KVCache } {
    const [batch, seqLen, _hidden] = x.shape;

    const toHeads = (t: Tensor, numHeads: number) => {
      const h = t.reshape([batch, seqLen, numHeads, this.headDim]);
      if (seqLen === 1) return h.reshape([batch, numHeads, 1, this.headDim]);
      return h.permute([0, 2, 1, 3]).contiguous();
    };
    let q = toHeads(this.qProj.forward(x), this.numHeads);
    let k = toHeads(this.kProj.forward(x), this.numKVHeads);
    let v = toHeads(this.vProj.forward(x), this.numKVHeads);

    q = this.api.applyRoPE(q, rope.cos, rope.sin);
    k = this.api.applyRoPE(k, rope.cos, rope.sin);

    let presentKV: KVCache | undefined;
    if (pastKV) {
      k = this.api.cat([pastKV.k, k], 2);
      v = this.api.cat([pastKV.v, v], 2);
    }
    presentKV = { k, v };

    const kE = this.expandKV(k);
    const vE = this.expandKV(v);
    const kvSeqLen = kE.shape[2];

    let attnOutput: Tensor;
    if (!pastKV) {
      // Fused path: soft-cap + causal (+ window) via the layer modifier.
      attnOutput = this.api.scaledDotProductAttention(
        q,
        kE,
        vE,
        this.scale,
        true,
        this.modifier,
      );
    } else {
      // Decomposed cached-decode path.
      const kT = kE.transpose({ dim0: 2, dim1: 3 });
      let scores = q.matmul(kT).mul(this.scale); // [B, H, S, kvS]
      scores = this.softcapScores(scores, this.modifier.scoreMod!.cap);
      const qStart = kvSeqLen - seqLen;
      // Additive mask: causal (+ sliding window). Build on GPU. j > qStart+i
      // is future (masked). For the window, also mask j <= (qStart+i) - window.
      let needMask = seqLen > 1;
      let maskT: Tensor | null = null;
      if (seqLen > 1) {
        maskT = this.api.triu(
          this.api.full([1, 1, seqLen, kvSeqLen], -1e9),
          qStart + 1,
        );
      }
      if (this.slidingWindow !== null) {
        // Sliding window: mask keys strictly older than the window. For query
        // row i (absolute pos qStart+i), valid kv = [qStart+i-window+1, ...],
        // so mask j <= (qStart+i)-window. tril(-1e9, diagonal=qStart-window)
        // keeps element (i,j) iff j <= i + (qStart-window) → exactly that set.
        const winMask = this.buildWindowMask(seqLen, kvSeqLen, qStart);
        maskT = maskT === null ? winMask : this.api.add(maskT, winMask);
        needMask = true;
      }
      if (needMask && maskT !== null) scores = this.api.add(scores, maskT);
      const attnWeights = scores.softmax(-1);
      attnOutput = attnWeights.matmul(vE);
    }

    const attnConcat = attnOutput
      .permute([0, 2, 1, 3])
      .contiguous()
      .reshape([batch, seqLen, this.numHeads * this.headDim]);
    return { out: this.oProj.forward(attnConcat), presentKV };
  }

  /** Additive window mask [1,1,S,kvS]: -1e9 where kv <= (qStart+i) - window.
   *  tril(full(-1e9), diag): element (i,j) kept (=-1e9) iff j <= i + diag.
   *  We want mask where j <= (qStart+i) - window, i.e. j - i <= qStart - window
   *  → diag = qStart - window. */
  private buildWindowMask(
    seqLen: number,
    kvSeqLen: number,
    qStart: number,
  ): Tensor {
    return this.api.tril(
      this.api.full([1, 1, seqLen, kvSeqLen], -1e9),
      qStart - this.slidingWindow!,
    );
  }
}

// ============================================================================
// MLP (GeGLU)
// ============================================================================

export class Gemma2MLP extends Module {
  readonly gateProj: Linear;
  readonly upProj: Linear;
  readonly downProj: Linear;

  constructor(
    api: Torchlette,
    config: Gemma2Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    const device = options?.device;
    const dtype = config.weightDtype ?? "f32";
    this.gateProj = new Linear(
      api,
      config.hiddenSize,
      config.intermediateSize,
      { bias: false, device, dtype },
    );
    this.upProj = new Linear(api, config.hiddenSize, config.intermediateSize, {
      bias: false,
      device,
      dtype,
    });
    this.downProj = new Linear(
      api,
      config.intermediateSize,
      config.hiddenSize,
      { bias: false, device, dtype },
    );
  }

  forward(x: Tensor): Tensor {
    const gate = this.gateProj.forward(x).gelu({ approximate: "tanh" });
    const up = this.upProj.forward(x);
    return this.downProj.forward(this.api.mul(gate, up));
  }
}

// ============================================================================
// Transformer Block (norm sandwich)
// ============================================================================

export class Gemma2Block extends Module {
  readonly inputNorm: RMSNorm;
  readonly attn: Gemma2Attention;
  readonly postAttnNorm: RMSNorm;
  readonly preFeedforwardNorm: RMSNorm;
  readonly mlp: Gemma2MLP;
  readonly postFeedforwardNorm: RMSNorm;

  constructor(
    api: Torchlette,
    config: Gemma2Config,
    layerIdx: number,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    const device = options?.device;
    const norm = () =>
      new RMSNorm(api, config.hiddenSize, {
        eps: config.rmsNormEps,
        device,
      });
    this.inputNorm = norm();
    this.attn = new Gemma2Attention(api, config, layerIdx, options);
    this.postAttnNorm = norm();
    this.preFeedforwardNorm = norm();
    this.mlp = new Gemma2MLP(api, config, options);
    this.postFeedforwardNorm = norm();
  }

  forward(
    x: Tensor,
    rope: RopeSlices,
    pastKV?: KVCache,
  ): { out: Tensor; presentKV?: KVCache } {
    // residual + post_attn_norm(attn(input_norm(x)))
    const { out: attnOut, presentKV } = this.attn.forward(
      this.inputNorm.forward(x),
      rope,
      pastKV,
    );
    let h = this.api.add(x, this.postAttnNorm.forward(attnOut));
    // residual + post_ffn_norm(mlp(pre_ffn_norm(h)))
    h = this.api.add(
      h,
      this.postFeedforwardNorm.forward(
        this.mlp.forward(this.preFeedforwardNorm.forward(h)),
      ),
    );
    return { out: h, presentKV };
  }

  forwardStatic(
    x: Tensor,
    rope: RopeSlices,
    ctx: Parameters<Gemma2Attention["forwardStatic"]>[2],
  ): { out: Tensor } {
    const { out: attnOut } = this.attn.forwardStatic(
      this.inputNorm.forward(x),
      rope,
      ctx,
    );
    let h = this.api.add(x, this.postAttnNorm.forward(attnOut));
    h = this.api.add(
      h,
      this.postFeedforwardNorm.forward(
        this.mlp.forward(this.preFeedforwardNorm.forward(h)),
      ),
    );
    return { out: h };
  }
}

// ============================================================================
// Gemma2 Model
// ============================================================================

export type Gemma2ForwardOptions = {
  pastKVs?: KVCache[];
  staticKV?: StaticKV;
  posOffset?: number;
  residualHook?: ResidualHook;
  collectHidden?: boolean;
};

export class Gemma2 extends Module {
  readonly config: Gemma2Config;
  /** Attention-modifier key fragment for capture/tape bucket keys (#64 iv).
   *  Gemma-2 keys on the union of its per-layer modifier kinds so its tape
   *  can never replay a differently-configured model's tape. */
  readonly attnModKey: string;
  readonly embedTokens: Embedding;
  readonly layers: ModuleList;
  readonly norm: RMSNorm;
  private readonly embedScale: number;
  private readonly ropeCos: Tensor;
  private readonly ropeSin: Tensor;

  constructor(
    api: Torchlette,
    config: Gemma2Config,
    options?: { device?: DeviceKind },
  ) {
    super(api);
    this.config = config;
    const device = options?.device;
    this.embedScale = Math.sqrt(config.hiddenSize);
    // Union of modifier kinds across layers → structural tape discriminator.
    this.attnModKey = `gemma2.sc${config.attnLogitSoftcap}.w${config.slidingWindow}`;

    this.embedTokens = new Embedding(api, config.vocabSize, config.hiddenSize, {
      device,
    });
    // The 256k×2304 f32 embedding is 2.36GB. nn.Embedding randn-inits f32
    // tables, and randn binds the whole buffer (> the 2GB storage-buffer
    // binding limit) → a dropped submit. This is a pretrained-load path, so
    // replace the randn'd weight with a zeros buffer (clearBuffer — no binding)
    // BEFORE the randn node is ever forced; the loader overwrites it.
    if (config.vocabSize * config.hiddenSize * 4 > (1 << 31) - 4) {
      const randnWeight = this.embedTokens.weight;
      this.embedTokens.registerParameter(
        "weight",
        api.zeros([config.vocabSize, config.hiddenSize], {
          requiresGrad: true,
          device,
        }),
      );
      // Drop the lazy randn node so markStep never forces its 2.36GB bind.
      randnWeight.dispose();
    }
    this.layers = new ModuleList(api);
    for (let i = 0; i < config.numLayers; i++) {
      this.layers.append(new Gemma2Block(api, config, i, options));
    }
    this.norm = new RMSNorm(api, config.hiddenSize, {
      eps: config.rmsNormEps,
      device,
    });

    // RoPE tables [maxSeqLen, headDim/2] (half-split convention).
    const half = config.headDim / 2;
    const cosArr = new Float32Array(config.maxSeqLen * half);
    const sinArr = new Float32Array(config.maxSeqLen * half);
    for (let s = 0; s < config.maxSeqLen; s++) {
      for (let i = 0; i < half; i++) {
        const freq = 1 / config.ropeTheta ** ((2 * i) / config.headDim);
        const ang = s * freq;
        cosArr[s * half + i] = Math.cos(ang);
        sinArr[s * half + i] = Math.sin(ang);
      }
    }
    this.ropeCos = api.tensorFromArray(cosArr, [config.maxSeqLen, half], {
      device,
    });
    this.ropeSin = api.tensorFromArray(sinArr, [config.maxSeqLen, half], {
      device,
    });
  }

  allocStaticKV(maxSeqLen = this.config.maxSeqLen): StaticKV {
    const { numKVHeads, headDim, numLayers } = this.config;
    const k: Tensor[] = [];
    const v: Tensor[] = [];
    for (let i = 0; i < numLayers; i++) {
      k.push(this.api.zeros([1, numKVHeads, maxSeqLen, headDim]));
      v.push(this.api.zeros([1, numKVHeads, maxSeqLen, headDim]));
    }
    return { k, v, len: 0, maxSeqLen };
  }

  /** Tied lm_head weight, pre-split into <2GB-binding vocab chunks by the
   *  loader (256k rows × 2304 × 4B = 2.36GB > the 2GB storage-buffer binding
   *  limit, and matmul binds a weight operand WHOLE — a narrow view would bind
   *  the 2.36GB base). Each chunk is an INDEPENDENT [rows, hidden] buffer.
   *  Null → single-matmul path (small vocab). */
  lmHeadChunks: Tensor[] | null = null;

  /** Number of vocab rows per lm_head chunk (~1GB → 2 chunks for 2B). */
  lmHeadChunkRows(): number {
    return Math.floor((1 << 30) / (this.config.hiddenSize * 4));
  }

  /** Tied lm_head: x @ W^T. Uses the pre-split chunks when present. */
  private lmHead(x: Tensor): Tensor {
    if (this.lmHeadChunks === null) {
      return this.api.linear(x, this.embedTokens.weight, null);
    }
    const parts = this.lmHeadChunks.map((w) => this.api.linear(x, w, null));
    return this.api.cat(parts, -1);
  }

  /** Final-logit soft-cap: cap * tanh(logits / cap). Elementwise, model-level
   *  (NOT an attention modifier). */
  private capLogits(logits: Tensor): Tensor {
    const cap = this.config.finalLogitSoftcap;
    return logits.div(cap).tanh().mul(cap);
  }

  forward(
    idx: Tensor,
    options?: Gemma2ForwardOptions,
  ): { logits: Tensor; presentKVs: KVCache[]; hidden?: Tensor[] } {
    const [_batch, seqLen] = idx.shape;
    const posOffset = options?.staticKV
      ? options.staticKV.len
      : (options?.posOffset ?? 0);
    if (posOffset + seqLen > this.config.maxSeqLen) {
      throw new Error(
        `Sequence length ${posOffset + seqLen} exceeds maxSeqLen ${this.config.maxSeqLen}`,
      );
    }

    const half = this.config.headDim / 2;
    const ropeIdxArr = new Float32Array(seqLen * half);
    for (let s = 0; s < seqLen; s++)
      ropeIdxArr.fill(posOffset + s, s * half, (s + 1) * half);
    const ropeIdx = this.api.tensorFromArray(ropeIdxArr, [seqLen, half]);
    const rope: RopeSlices = {
      cos: this.api.gather(this.ropeCos, ropeIdx, { dim: 0 }),
      sin: this.api.gather(this.ropeSin, ropeIdx, { dim: 0 }),
    };

    // Embedding output scaled by sqrt(hiddenSize) (Gemma normalizer).
    let x = this.api.mul(this.embedTokens.forward(idx), this.embedScale);
    const hidden: Tensor[] | undefined = options?.collectHidden
      ? [x]
      : undefined;

    const presentKVs: KVCache[] = [];
    const cache = options?.staticKV;
    if (cache) {
      if (posOffset + seqLen > cache.maxSeqLen) {
        throw new Error(
          `static KV overflow: ${posOffset + seqLen} > ${cache.maxSeqLen}`,
        );
      }
      const { numKVHeads, headDim } = this.config;
      const idxArr = new Float32Array(numKVHeads * seqLen * headDim);
      for (let h = 0; h < numKVHeads; h++) {
        for (let s = 0; s < seqLen; s++) {
          idxArr.fill(
            posOffset + s,
            (h * seqLen + s) * headDim,
            (h * seqLen + s + 1) * headDim,
          );
        }
      }
      const scatterIdx = this.api.tensorFromArray(idxArr, [
        1,
        numKVHeads,
        seqLen,
        headDim,
      ]);
      const isPrefill = posOffset === 0 && seqLen > 1;
      const bucketLen = kvBucketLen(posOffset + seqLen, cache.maxSeqLen);
      cache.len = posOffset + seqLen;
      let mask: Tensor | null = null;
      if (!isPrefill) {
        if (seqLen !== 1)
          throw new Error(
            "static KV: incremental multi-token decode unsupported",
          );
        // Per-layer mask differs (global vs local), so build both once and
        // pick inside the loop. Base padding+causal mask (all layers):
        // 0 for valid [0..posOffset], -1e9 for padding.
        const globalArr = new Float32Array(bucketLen).fill(-1e9);
        globalArr.fill(0, 0, posOffset + 1);
        const globalMask = this.api.tensorFromArray(globalArr, [
          1,
          1,
          1,
          bucketLen,
        ]);
        // Local mask: additionally mask positions <= posOffset - window.
        const localArr = new Float32Array(bucketLen).fill(-1e9);
        const winStart = Math.max(0, posOffset - this.config.slidingWindow + 1);
        localArr.fill(0, winStart, posOffset + 1);
        const localMask = this.api.tensorFromArray(localArr, [
          1,
          1,
          1,
          bucketLen,
        ]);
        mask = globalMask; // placeholder; per-layer selection below
        // Stash both on a closure via layer loop.
        for (let i = 0; i < this.layers.length; i++) {
          const block = this.layers.get(i) as Gemma2Block;
          const isLocal =
            this.config.layerTypes[i] === "sliding_attention";
          const { out } = block.forwardStatic(x, rope, {
            kSlot: cache.k[i],
            vSlot: cache.v[i],
            scatterIdx,
            bucketLen,
            mask: isLocal ? localMask : globalMask,
            posOffset,
          });
          x = out;
          if (options?.residualHook) x = options.residualHook(x, i);
          hidden?.push(x);
        }
      } else {
        // Prefill: fused kernel handles causal/window/softcap via modifier.
        for (let i = 0; i < this.layers.length; i++) {
          const block = this.layers.get(i) as Gemma2Block;
          const { out } = block.forwardStatic(x, rope, {
            kSlot: cache.k[i],
            vSlot: cache.v[i],
            scatterIdx,
            bucketLen,
            mask: null,
            posOffset,
          });
          x = out;
          if (options?.residualHook) x = options.residualHook(x, i);
          hidden?.push(x);
        }
      }
    } else {
      for (let i = 0; i < this.layers.length; i++) {
        const block = this.layers.get(i) as Gemma2Block;
        const { out, presentKV } = block.forward(
          x,
          rope,
          options?.pastKVs?.[i],
        );
        x = out;
        if (options?.residualHook) x = options.residualHook(x, i);
        if (presentKV) presentKVs.push(presentKV);
        hidden?.push(x);
      }
    }

    x = this.norm.forward(x);
    // Tied lm_head: logits = x @ embed.weight^T, then final soft-cap.
    // The 256k-row embedding table (2.36GB f32) exceeds the 2GB max storage
    // buffer binding size, and matmul binds a weight operand whole (no
    // chunking). Split the vocab dim so each matmul binds a sub-2GB slice,
    // then concat. (The embedding-forward gather auto-chunks; only this tied
    // matmul needs the manual split.)
    let logits = this.lmHead(x);
    logits = this.capLogits(logits);
    return { logits, presentKVs, hidden };
  }
}
