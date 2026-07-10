/**
 * Gemma Scope JumpReLU SAE (residual-stream) — framework core.
 *
 * A sparse autoencoder over a single residual-stream hookpoint of Gemma-2-2B.
 * Conventions verified against the Gemma Scope release (arXiv 2408.05147, §3.1 +
 * App. A) and the official tutorial:
 *
 *   ENCODE:  pre  = x @ W_enc + b_enc            (NO b_dec pre-centering —
 *            acts = relu(pre) * (pre > threshold)  App. A: neither centering nor
 *                                                   threshold-folding at inference)
 *   DECODE:  x_hat = acts @ W_dec + b_dec
 *   STEER:   a feature f's residual-stream direction is the row W_dec[f].
 *
 * The "layer_N" residual SAEs are trained on `blocks.N.hook_resid_post` — the
 * residual stream AFTER the full transformer block N (attn + MLP), before the
 * final model norm. In this codebase that is `hidden[N+1]` from
 * Gemma2.forward({collectHidden:true}), or the value seen by
 * `residualHook(x, N)`.
 *
 * Weights load as f32 by default (each of W_enc/W_dec is 16384×2304×4B ≈ 151MB;
 * both ≈ 300MB — comfortable on a 16GB Mac alongside the f16 model). f16 is
 * available for the encode matmul but the parity gate runs f32.
 */

import type { Tensor, Torchlette } from "torchlette";

export type SAEConfig = {
  /** Residual-stream width of the base model (Gemma-2-2B: 2304). */
  dModel: number;
  /** SAE dictionary size (width_16k → 16384). */
  numFeatures: number;
  /** The base-model layer index this SAE hooks (post-block). Gemma Scope res
   *  "layer_20" → 20. */
  layer: number;
  /** Neuronpedia SAE id fragment, e.g. "20-gemmascope-res-16k". */
  neuronpediaSaeId: string;
};

/** Raw SAE parameters as flat Float32Arrays (row-major, HF/npz layout). */
export type SAEParams = {
  /** [dModel, numFeatures] */
  W_enc: Float32Array;
  /** [numFeatures] */
  b_enc: Float32Array;
  /** [numFeatures, dModel] */
  W_dec: Float32Array;
  /** [dModel] */
  b_dec: Float32Array;
  /** [numFeatures] JumpReLU thresholds (all > 0). */
  threshold: Float32Array;
};

/** GPU-resident SAE tensors + config. Construct via {@link GemmaScopeSAE.load}. */
export class GemmaScopeSAE {
  readonly config: SAEConfig;
  /** [dModel, numFeatures] — encoder weight, `linear` expects [out,in] so we
   *  keep the explicit matmul form x @ W_enc instead. */
  readonly wEnc: Tensor;
  /** [1, numFeatures] broadcastable encoder bias. */
  readonly bEnc: Tensor;
  /** [1, numFeatures] broadcastable threshold. */
  readonly threshold: Tensor;
  /** [numFeatures, dModel] — decoder weight; row f is feature f's direction. */
  readonly wDec: Tensor;
  /** [1, dModel] broadcastable decoder bias. */
  readonly bDec: Tensor;
  /** CPU copy of W_dec for cheap per-feature direction extraction (steering
   *  builds a [dModel] vector on the host and uploads it — one row, ~9KB). */
  private readonly wDecCpu: Float32Array;

  private constructor(
    api: Torchlette,
    config: SAEConfig,
    params: SAEParams,
    dtype: "f32" | "f16",
  ) {
    this.config = config;
    const { dModel, numFeatures } = config;
    this.wEnc = api.tensorFromArray(params.W_enc, [dModel, numFeatures], {
      dtype,
    });
    this.bEnc = api.tensorFromArray(params.b_enc, [1, numFeatures]);
    this.threshold = api.tensorFromArray(params.threshold, [1, numFeatures]);
    this.wDec = api.tensorFromArray(params.W_dec, [numFeatures, dModel], {
      dtype,
    });
    this.bDec = api.tensorFromArray(params.b_dec, [1, dModel]);
    this.wDecCpu = params.W_dec;
  }

  static load(
    api: Torchlette,
    config: SAEConfig,
    params: SAEParams,
    options?: { dtype?: "f32" | "f16" },
  ): GemmaScopeSAE {
    if (params.W_enc.length !== config.dModel * config.numFeatures)
      throw new Error(
        `W_enc size ${params.W_enc.length} != ${config.dModel}*${config.numFeatures}`,
      );
    if (params.W_dec.length !== config.numFeatures * config.dModel)
      throw new Error(`W_dec size mismatch`);
    if (params.threshold.length !== config.numFeatures)
      throw new Error(`threshold size mismatch`);
    return new GemmaScopeSAE(api, config, params, options?.dtype ?? "f32");
  }

  /**
   * JumpReLU encode. Input `x` is a residual-stream tensor of shape
   * [..., dModel] (any leading dims; typically [seq, dModel] or [dModel]).
   * Returns feature activations of shape [..., numFeatures].
   *
   *   pre  = x @ W_enc + b_enc
   *   acts = relu(pre) * (pre > threshold)
   */
  encode(api: Torchlette, x: Tensor): Tensor {
    const { dModel, numFeatures } = this.config;
    // Flatten leading dims to a 2-D [rows, dModel] matmul, restore shape after.
    const shape = x.shape;
    const dm = shape[shape.length - 1];
    if (dm !== dModel)
      throw new Error(`encode: last dim ${dm} != dModel ${dModel}`);
    const rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const x2 = api.reshape(x, [rows, dModel]);
    const pre = api.add(api.matmul(x2, this.wEnc), this.bEnc); // [rows, F]
    const mask = api.gt(pre, this.threshold); // 1.0 where pre > threshold
    const acts = api.mul(api.relu(pre), mask); // [rows, F]
    const outShape = [...shape.slice(0, -1), numFeatures];
    return api.reshape(acts, outShape);
  }

  /** Decode feature activations [..., numFeatures] back to the residual stream
   *  [..., dModel]: x_hat = acts @ W_dec + b_dec. */
  decode(api: Torchlette, acts: Tensor): Tensor {
    const { dModel, numFeatures } = this.config;
    const shape = acts.shape;
    const rows = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const a2 = api.reshape(acts, [rows, numFeatures]);
    const xHat = api.add(api.matmul(a2, this.wDec), this.bDec);
    return api.reshape(xHat, [...shape.slice(0, -1), dModel]);
  }

  /** Feature f's residual-stream direction W_dec[f] as a fresh [dModel] tensor
   *  on GPU. Used by steering to add α·dir into the residual. The rows of W_dec
   *  are already unit-norm in Gemma Scope (decoder columns are normalized during
   *  training), so no renormalization is applied by default. */
  featureDirection(api: Torchlette, feature: number): Tensor {
    const { dModel, numFeatures } = this.config;
    if (feature < 0 || feature >= numFeatures)
      throw new Error(`feature ${feature} out of range [0,${numFeatures})`);
    const row = this.wDecCpu.slice(feature * dModel, (feature + 1) * dModel);
    return api.tensorFromArray(row, [dModel]);
  }

  /** L2 norm of feature f's decoder direction (diagnostic; ~1.0 in Gemma Scope). */
  featureDirectionNorm(feature: number): number {
    const { dModel } = this.config;
    let s = 0;
    for (let i = 0; i < dModel; i++) {
      const v = this.wDecCpu[feature * dModel + i];
      s += v * v;
    }
    return Math.sqrt(s);
  }
}

/** Neuronpedia feature-page URL for a Gemma-2-2B Gemma Scope feature.
 *  saeId e.g. "20-gemmascope-res-16k". */
export function neuronpediaUrl(saeId: string, feature: number): string {
  return `https://www.neuronpedia.org/gemma-2-2b/${saeId}/${feature}`;
}
