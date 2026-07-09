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

import type { DeviceKind, Tensor, Torchlette } from "torchlette";
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
    headDim:
      (hf.head_dim as number) ??
      (hf.hidden_size as number) / (hf.num_attention_heads as number),
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

/**
 * Static (preallocated, in-place) KV cache for shape-stable decode.
 *
 * Buffers are allocated ONCE at [1, kvHeads, maxSeqLen, headDim] per layer
 * and updated in place with copy_ region writes (the framework's persistent-
 * state pattern — same path as packed optimizer state). Attention reads a
 * BUCKETED prefix (length padded to BUCKET multiples) with an additive
 * padding mask uploaded per step as data, so every decode step within a
 * bucket executes the identical plan template — which the recurring-plan
 * replay machinery then accelerates automatically.
 */
export type StaticKV = {
  k: Tensor[]; // per layer, [1, kvH, maxSeqLen, headDim] f32
  v: Tensor[];
  /** Valid positions written so far. */
  len: number;
  maxSeqLen: number;
};

export const KV_BUCKET = 128;

/** Padded attention length for a given valid length. */
export function kvBucketLen(len: number, maxSeqLen: number): number {
  return Math.min(Math.ceil(len / KV_BUCKET) * KV_BUCKET, maxSeqLen);
}

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
    this.qNorm = new RMSNorm(api, config.headDim, {
      eps: config.rmsNormEps,
      device,
    });
    this.kNorm = new RMSNorm(api, config.headDim, {
      eps: config.rmsNormEps,
      device,
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

  /**
   * Static-cache attention step. Writes this step's K/V into the preallocated
   * cache via scatterAdd (position enters as an INDEX TENSOR = data, so the
   * plan template is stable and the replay machinery applies; add ≡ copy
   * because every slot is written exactly once into zeroed cache). Attention
   * reads the bucketed cache prefix under `mask`. For prefill (`mask` null),
   * attention computes over the fresh K/V with the fused causal kernel — the
   * cache write is only for later decode steps.
   * Returns the UPDATED cache tensors (out-of-place; caller replaces refs).
   */
  forwardStatic(
    x: Tensor,
    rope: RopeSlices,
    ctx: {
      kSlot: Tensor;
      vSlot: Tensor;
      scatterIdx: Tensor; // [1, kvH, S, D] filled with target positions
      bucketLen: number;
      mask: Tensor | null; // decode: [1,1,1,bucketLen] additive; prefill: null
    },
  ): { out: Tensor } {
    const [batch, seqLen, _hidden] = x.shape;
    const toHeads = (t: Tensor, numHeads: number, norm?: RMSNorm) => {
      let h = t.reshape([batch, seqLen, numHeads, this.headDim]);
      if (norm) h = norm.forward(h);
      if (seqLen === 1) return h.reshape([batch, numHeads, 1, this.headDim]);
      return h.permute([0, 2, 1, 3]).contiguous();
    };
    let q = toHeads(this.qProj.forward(x), this.numHeads, this.qNorm);
    let k = toHeads(this.kProj.forward(x), this.numKVHeads, this.kNorm);
    const v = toHeads(this.vProj.forward(x), this.numKVHeads);
    q = this.api.applyRoPE(q, rope.cos, rope.sin);
    k = this.api.applyRoPE(k, rope.cos, rope.sin);

    // IN-PLACE cache update with stable buffer identity: scatterAdd is
    // out-of-place (position as index DATA keeps the template stable), and
    // the full-overwrite copy_ DMAs the result back into the cache's OWN
    // buffer — replayed plans rebind the same external input every step.
    // (Replace-and-hold here is the documented anti-pattern: replays bind
    // recorded buffers, so external persistent inputs must not churn.)
    ctx.kSlot.copy_(ctx.kSlot.scatterAdd(ctx.scatterIdx, k, { dim: 2 }));
    ctx.vSlot.copy_(ctx.vSlot.scatterAdd(ctx.scatterIdx, v, { dim: 2 }));

    const scale = 1.0 / Math.sqrt(this.headDim);
    let attnOutput: Tensor;
    if (ctx.mask === null) {
      // Prefill from position 0: fresh K/V are the whole context.
      attnOutput = this.api.scaledDotProductAttention(
        q,
        this.expandKV(k),
        this.expandKV(v),
        scale,
        true,
      );
    } else {
      const kE = this.expandKV(ctx.kSlot.narrow(2, 0, ctx.bucketLen));
      const vE = this.expandKV(ctx.vSlot.narrow(2, 0, ctx.bucketLen));
      const scores = q.matmul(kE.transpose({ dim0: 2, dim1: 3 })).mul(scale);
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

    // Project, split into heads, QK-norm (over headDim, before RoPE — HF order),
    // then [B, S, H, D] -> [B, H, S, D].
    const toHeads = (t: Tensor, numHeads: number, norm?: RMSNorm) => {
      let h = t.reshape([batch, seqLen, numHeads, this.headDim]);
      if (norm) h = norm.forward(h);
      // Decode fast path: with S=1, [B,1,H,D] and [B,H,1,D] have identical
      // memory layout — the head transpose is a free reshape, no permute or
      // materializing copy (saves ~3 contiguous dispatches/layer/token).
      if (seqLen === 1) return h.reshape([batch, numHeads, 1, this.headDim]);
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
      // seqLen===1 needs no mask (all past visible); multi-token-with-cache
      // builds the additive causal mask ON the GPU: -1e9 where j > qStart+i,
      // i.e. triu(full(-1e9), k=qStart+1). (The old CPU-built upload was a
      // dodge for the offset-view bug — fixed in the core, task #58.)
      const kT = kE.transpose({ dim0: 2, dim1: 3 });
      let scores = q.matmul(kT).mul(scale); // [B, H, S, kvS]
      if (seqLen > 1) {
        const qStart = kvSeqLen - seqLen;
        const mask = this.api.triu(
          this.api.full([1, 1, seqLen, kvSeqLen], -1e9),
          qStart + 1,
        );
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
    this.inputNorm = new RMSNorm(api, config.hiddenSize, {
      eps: config.rmsNormEps,
      device,
    });
    this.attn = new Qwen3Attention(api, config, options);
    this.postAttnNorm = new RMSNorm(api, config.hiddenSize, {
      eps: config.rmsNormEps,
      device,
    });
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

  /** Static-cache variant of forward — see Qwen3Attention.forwardStatic. */
  forwardStatic(
    x: Tensor,
    rope: RopeSlices,
    ctx: Parameters<Qwen3Attention["forwardStatic"]>[2],
  ): { out: Tensor } {
    const { out: attnOut } = this.attn.forwardStatic(
      this.inputNorm.forward(x),
      rope,
      ctx,
    );
    let h = this.api.add(x, attnOut);
    h = this.api.add(h, this.mlp.forward(this.postAttnNorm.forward(h)));
    return { out: h };
  }
}

// ============================================================================
// Qwen3 Model
// ============================================================================

export type Qwen3ForwardOptions = {
  /** Per-layer KV cache from previous forward. */
  pastKVs?: KVCache[];
  /**
   * Static preallocated KV cache (see StaticKV): shape-stable decode for
   * plan-replay. Mutually exclusive with pastKVs. Positions/RoPE derive from
   * cache.len; forward advances cache.len by seqLen.
   */
  staticKV?: StaticKV;
  /** Position offset (= past sequence length) for RoPE. */
  posOffset?: number;
  /** Residual-stream hook applied after each block (steering seam). */
  residualHook?: ResidualHook;
  /** Collect per-layer hidden states (embeddings + each block output). */
  collectHidden?: boolean;
};

export class Qwen3 extends Module {
  readonly config: Qwen3Config;
  /** Attention-modifier key fragment for capture/tape bucket keys (#64 iv).
   *  "" = null modifier (plain causal) — keeps existing bucketKeys
   *  byte-stable. A Gemma-2-style port sets this to the attnModifierKey()
   *  union of its per-layer modifiers, so a different modifier set can
   *  never replay another's tape. */
  readonly attnModKey: string = "";
  readonly embedTokens: Embedding;
  readonly layers: ModuleList;
  readonly norm: RMSNorm;
  // RoPE tables uploaded to the GPU ONCE at construction ([maxSeqLen, D/2]);
  // each forward takes a zero-copy narrow(0, posOffset, seqLen) view. The
  // fused RoPE kernel folds the view's element offset into its table
  // indexing (task #58 — this replaces the old per-forward CPU subarray
  // upload that dodged the offset-view bug).
  private readonly ropeCos: Tensor;
  private readonly ropeSin: Tensor;

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
    this.embedTokens = new Embedding(api, config.vocabSize, config.hiddenSize, {
      device,
    });
    this.layers = new ModuleList(api);
    for (let i = 0; i < config.numLayers; i++) {
      this.layers.append(new Qwen3Block(api, config, options));
    }
    this.norm = new RMSNorm(api, config.hiddenSize, {
      eps: config.rmsNormEps,
      device,
    });

    // Precompute RoPE tables [maxSeqLen, headDim/2] (half-split convention)
    // and upload them to the device once as persistent tensors.
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

  /** Allocate a zeroed static KV cache (one pair of buffers per layer). */
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

  /**
   * Forward pass. Returns logits [batch, seqLen, vocabSize], the per-layer
   * KV cache, and (optionally) per-layer hidden states.
   */
  forward(
    idx: Tensor,
    options?: Qwen3ForwardOptions,
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

    // Row-slices of the persistent GPU RoPE tables at the current positions.
    // Position enters as an INDEX TENSOR (data) — same pattern as the KV
    // scatterIdx below — so the plan template stays stable across decode
    // steps. A narrow(0, posOffset, seqLen) view is equally CORRECT now
    // (offset-view class fixed in the core, task #58, and gated by
    // test/offset-views.spec.ts), but its per-token-varying `start` is
    // PAYLOAD, which the template fingerprint deliberately hashes (the
    // anti-frozen-scalar rule) — every decode step would re-lower instead
    // of replaying (~5x slower decode).
    const half = this.config.headDim / 2;
    const ropeIdxArr = new Float32Array(seqLen * half);
    for (let s = 0; s < seqLen; s++)
      ropeIdxArr.fill(posOffset + s, s * half, (s + 1) * half);
    const ropeIdx = this.api.tensorFromArray(ropeIdxArr, [seqLen, half]);
    const rope: RopeSlices = {
      cos: this.api.gather(this.ropeCos, ropeIdx, { dim: 0 }),
      sin: this.api.gather(this.ropeSin, ropeIdx, { dim: 0 }),
    };

    // Embedding output is f32 (f32 table); with weightDtype "f16" the linears
    // run mixed-dtype (f32 activations × f16 weights → f32).
    let x = this.embedTokens.forward(idx);
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
      // Scatter index [1, kvH, S, D]: every element of row s targets cache
      // position posOffset+s. Position enters the graph as DATA (index
      // tensor), keeping the plan template stable across steps.
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
      // Decode mask over the bucketed prefix: 0 for valid, -1e9 for padding.
      // Prefill (from pos 0) attends its own fresh K/V with the fused causal
      // kernel and needs no mask.
      const isPrefill = posOffset === 0 && seqLen > 1;
      const bucketLen = kvBucketLen(posOffset + seqLen, cache.maxSeqLen);
      // Advance the cache length HERE — after the per-step upload tensors that
      // read posOffset (token/rope/scatterIdx above) and BEFORE the LAST upload
      // (the mask, below) and the layer loop (none of which read cache.len).
      // This position matters for capture() (2a): a taped replay short-circuits
      // fn on its LAST upload (the mask), so any state advance AFTER the mask
      // would be skipped on a hit, freezing the decode position. The mask reads
      // only posOffset (a local snapshot), so advancing cache.len just before
      // it is semantically identical for the untaped path.
      cache.len = posOffset + seqLen;
      let mask: Tensor | null = null;
      if (!isPrefill) {
        if (seqLen !== 1)
          throw new Error(
            "static KV: incremental multi-token decode unsupported",
          );
        const maskArr = new Float32Array(bucketLen).fill(-1e9);
        maskArr.fill(0, 0, posOffset + 1);
        mask = this.api.tensorFromArray(maskArr, [1, 1, 1, bucketLen]);
      }
      for (let i = 0; i < this.layers.length; i++) {
        const block = this.layers.get(i) as Qwen3Block;
        // Cache tensors are updated IN PLACE (stable buffer identity — the
        // replay contract for external persistent inputs); no reassignment.
        const { out } = block.forwardStatic(x, rope, {
          kSlot: cache.k[i],
          vSlot: cache.v[i],
          scatterIdx,
          bucketLen,
          mask,
        });
        x = out;
        if (options?.residualHook) x = options.residualHook(x, i);
        hidden?.push(x);
      }
    } else {
      for (let i = 0; i < this.layers.length; i++) {
        const block = this.layers.get(i) as Qwen3Block;
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
    // Tied lm_head: logits = x @ embed.weight^T
    const logits = this.api.linear(x, this.embedTokens.weight, null);
    return { logits, presentKVs, hidden };
  }
}
