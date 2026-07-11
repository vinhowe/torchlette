/**
 * SAE feature INSPECTION + STEERING on Gemma-2-2B, via the Gemma Scope layer-20
 * residual-stream JumpReLU SAE.
 *
 * INSPECT: run the model on a prompt with the layer-20 residual captured through
 * model.forward's `residualHook` seam (we tap, we don't modify), encode it with
 * the SAE, and read back the top-K activating features (aggregated over the
 * sequence and at the last token). Each feature links out to its Neuronpedia
 * page.
 *
 * STEER: during generation, at layer 20, add Σ_f α_f · W_dec[f] to the residual
 * stream — W_dec[f] is feature f's (unit-norm) decoder direction. Positive α
 * amplifies the feature ("Golden Gate" it), negative suppresses. This is the
 * standard Gemma Scope steering recipe.
 */

import type { FrontendTensor as Tensor, Torchlette } from "torchlette";
import type { Gemma2, ResidualHook } from "gemma2-browser";
import { GemmaScopeSAE, neuronpediaUrl } from "gemma-scope-sae";

export type FeatureHit = {
  feature: number;
  /** Activation strength (max over sequence, or the last-token value). */
  activation: number;
  neuronpediaUrl: string;
};

export type FeatureReport = {
  prompt: string;
  layer: number;
  seqLen: number;
  /** Number of features active (>0) at the last token — the SAE's L0. */
  nActive: number;
  /** Top-K by max-over-sequence activation (the aggregated inspector view). */
  topAgg: FeatureHit[];
  /** Top-K at the last token (what last-token steering would amplify). */
  topLast: FeatureHit[];
};

export type SteerSpec = { feature: number; alpha: number };

type TokenizerLike = { encode(text: string): number[] };

function encodeIds(api: Torchlette, tokenizer: TokenizerLike, text: string): Tensor {
  const ids = tokenizer.encode(text);
  // Gemma requires a leading <bos> (id 2).
  if (ids[0] !== 2) ids.unshift(2);
  if (ids.length === 0) throw new Error("empty prompt");
  return api.tensorFromArray(ids, [1, ids.length]);
}

function topKHits(
  agg: Float32Array,
  k: number,
  saeId: string,
): FeatureHit[] {
  const idx = Array.from({ length: agg.length }, (_, i) => i);
  idx.sort((a, b) => agg[b] - agg[a]);
  return idx.slice(0, k).map((feature) => ({
    feature,
    activation: agg[feature],
    neuronpediaUrl: neuronpediaUrl(saeId, feature),
  }));
}

/**
 * Run `prompt` through the model, capture the layer-`sae.config.layer` residual,
 * SAE-encode it, and return the top-K activating features.
 *
 * The residual is captured via the residualHook (fires with layer=i AFTER block
 * i, matching hidden[i+1] = blocks.i.hook_resid_post — the Gemma Scope res
 * hookpoint). We encode ALL sequence positions, then aggregate (max over seq)
 * for the inspector and read the last token for steering guidance.
 */
export async function inspectFeatures(
  api: Torchlette,
  model: Gemma2,
  sae: GemmaScopeSAE,
  tokenizer: TokenizerLike,
  prompt: string,
  topK = 20,
): Promise<FeatureReport> {
  const layer = sae.config.layer;
  const saeId = sae.config.neuronpediaSaeId;
  const F = sae.config.numFeatures;

  const { agg, last, seqLen, nActive } = await api.noGrad(async () => {
    let captured: Tensor | null = null;
    const hook: ResidualHook = (x, l) => {
      if (l === layer) captured = x; // [1, seq, dModel]
      return x;
    };
    const idx = encodeIds(api, tokenizer, prompt);
    // Dense forward (no static KV) so the hook sees the full-sequence residual.
    model.forward(idx, { residualHook: hook });
    if (captured === null) throw new Error(`layer ${layer} residual not captured`);
    const resid = captured as Tensor;
    const [, seq, dModel] = resid.shape;
    const resid2 = api.reshape(resid, [seq, dModel]);
    const acts = sae.encode(api, resid2); // [seq, F]
    const flat = new Float32Array(await acts.cpu());

    // Aggregate max over CONTENT positions (skip position 0 = <bos>: its
    // attention-sink features fire identically on every prompt and would swamp
    // the content-specific ones, making the inspector show the same features for
    // every input). Also read the last-token row.
    const aggArr = new Float32Array(F);
    const start = seq > 1 ? 1 : 0;
    for (let s = start; s < seq; s++) {
      const off = s * F;
      for (let f = 0; f < F; f++) if (flat[off + f] > aggArr[f]) aggArr[f] = flat[off + f];
    }
    const lastArr = flat.slice((seq - 1) * F, seq * F);
    let nA = 0;
    for (let f = 0; f < F; f++) if (lastArr[f] > 0) nA++;
    return { agg: aggArr, last: lastArr, seqLen: seq, nActive: nA };
  });

  await api.markStep();
  return {
    prompt,
    layer,
    seqLen,
    nActive,
    topAgg: topKHits(agg, topK, saeId),
    topLast: topKHits(last, topK, saeId),
  };
}

/**
 * Build a residualHook that steers by SAE features: at the SAE's layer, add
 * Σ_f α_f · W_dec[f]. Each direction is unit-norm, so α is directly in
 * residual-norm units. Returns undefined when there is nothing to inject.
 *
 * The combined steering vector is precomputed on the host (sum of scaled decoder
 * rows) and uploaded ONCE as a persistent [1,1,dModel] tensor, so the hook is a
 * single broadcast add per layer — cheap enough to run on every decode step.
 */
export function makeSAEResidualHook(
  api: Torchlette,
  sae: GemmaScopeSAE,
  specs: SteerSpec[],
): ResidualHook | undefined {
  const active = specs.filter((s) => s.alpha !== 0 && Number.isFinite(s.alpha));
  if (active.length === 0) return undefined;
  const { dModel, layer } = sae.config;

  // Host-side accumulate Σ α_f · W_dec[f] (W_dec rows are unit-norm).
  const combined = new Float32Array(dModel);
  for (const { feature, alpha } of active) {
    const row = sae.featureDirectionCpu(feature); // [dModel]
    for (let i = 0; i < dModel; i++) combined[i] += alpha * row[i];
  }

  const vec = api.registerState(api.tensorFromArray(combined, [1, 1, dModel]));
  return (x, l) => {
    if (l !== layer) return x;
    return api.add(x, vec);
  };
}
