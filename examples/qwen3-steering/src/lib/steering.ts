/**
 * Contrastive activation steering (RepE / "steering by adding an activation
 * vector").
 *
 * COMPUTE: run the model on a POSITIVE and a NEGATIVE concept prompt with
 * per-layer hidden states collected; at a chosen layer L take the
 * residual-stream activation, mean-pooled over sequence positions, for each;
 * direction = normalize(meanPos_L - meanNeg_L). A [hidden] vector.
 *
 * APPLY: during generation, at layer L, add `alpha * direction` to the residual
 * stream at every position (via model.forward's `residualHook` seam — we do NOT
 * touch the model's math). Positive alpha steers toward the positive concept,
 * negative away.
 */

// torchlette re-exports its frontend Tensor class as `FrontendTensor`.
import type { FrontendTensor as Tensor, Torchlette } from "torchlette";
import type { Qwen3, ResidualHook } from "qwen3-browser";

export type SteeringVector = {
  /** Persistent [hidden] raw contrast direction (meanPos-meanNeg), held on the
   *  GPU across generations. Not normalized — magnitude self-calibrates. */
  direction: Tensor;
  /** Layer the direction was extracted from / is injected at. */
  layer: number;
  hiddenSize: number;
  posPrompt: string;
  negPrompt: string;
};

type TokenizerLike = {
  encode(text: string): number[];
};

/**
 * Mean-pool a [1, seq, hidden] residual-stream tensor over the seq axis →
 * [hidden]. api.mean over dim=1 keeps it fully on the GPU.
 */
function meanPoolSeq(api: Torchlette, hidden3d: Tensor): Tensor {
  const [, , h] = hidden3d.shape;
  const pooled = api.mean(hidden3d, { dim: 1 }) as Tensor; // [1, hidden]
  return api.reshape(pooled, [h]);
}

/**
 * Compute a contrastive steering direction at layer L.
 *
 * `hidden[]` from forward({collectHidden:true}) is [embeddings, block0_out, …,
 * blockN-1_out] — length numLayers+1. The residualHook fires with `layer = i`
 * AFTER block i, i.e. on the state that lands in `hidden[i + 1]`. We index the
 * post-block-L state consistently: `hidden[layer + 1]`.
 *
 * Returns a persistent unit [hidden] tensor. Hold it in JS across generations;
 * do NOT dispose it between runs (the persistence contract keeps it resident).
 */
export async function computeSteeringVector(
  api: Torchlette,
  model: Qwen3,
  tokenizer: TokenizerLike,
  posPrompt: string | string[],
  negPrompt: string | string[],
  layer: number,
): Promise<SteeringVector> {
  const hiddenSize = model.config.hiddenSize;
  const layerIdx = layer + 1; // hidden[0] = embeddings; hidden[i+1] = post-block i

  // Each side may be MULTIPLE example prompts (array, or one-per-line string).
  // Averaging the mean-pooled residual over several examples denoises the
  // concept direction — a single pair gives a blunt vector with a narrow
  // coherent band; a handful of varied examples gives a clean, obsessable one.
  const toList = (p: string | string[]) =>
    (Array.isArray(p) ? p : p.split("\n")).map((s) => s.trim()).filter(Boolean);
  const posList = toList(posPrompt);
  const negList = toList(negPrompt);
  if (posList.length === 0 || negList.length === 0) {
    throw new Error("steering needs at least one positive and one negative example");
  }

  const encode = (text: string) => {
    const ids = tokenizer.encode(text);
    if (ids.length === 0) throw new Error("empty prompt");
    return api.tensorFromArray(ids, [1, ids.length]);
  };

  // Mean over examples of (mean over sequence positions of the layer residual).
  const meanOverExamples = (list: string[]): Tensor => {
    let acc: Tensor | null = null;
    for (const p of list) {
      const out = model.forward(encode(p), { collectHidden: true });
      const pooled = meanPoolSeq(api, out.hidden![layerIdx]); // [hidden]
      acc = acc === null ? pooled : api.add(acc, pooled);
    }
    return api.mul(acc as Tensor, 1 / list.length);
  };

  const direction = api.noGrad(() => {
    const meanPos = meanOverExamples(posList);
    const meanNeg = meanOverExamples(negList);
    // RAW mean-difference vector (CAA / ActAdd), NOT normalized: it lives in
    // activation space so its magnitude self-calibrates to the residual scale
    // (Qwen3's is ~4000 at mid layers), and the injection coefficient behaves
    // like the literature's small multiplier. Persist so it survives the
    // worker's step-scoped cleanup and every future generation's markStep.
    const diff = api.sub(meanPos, meanNeg); // [hidden]
    api.registerState(diff);
    return diff;
  });

  // Materialize the lazy graph now (read one element back), then reclaim the
  // large per-prompt hidden-state graph temporaries.
  await api.item(direction.narrow(0, 0, 1));
  await api.markStep();

  return { direction, layer, hiddenSize, posPrompt: posList.join("\n"), negPrompt: negList.join("\n") };
}

/**
 * Build the residualHook closure that injects a steering vector — the standard
 * CAA/ActAdd form: `x + alpha * v` at the inject layer, where `v` is the raw
 * (un-normalized) contrast vector. `alpha` is the literature-style coefficient
 * (~0.3-2 useful here; the raw diff is ~870 vs a ~4000 residual, so alpha=1 is
 * a strong ~20% push). Returns undefined when there's nothing to inject (no
 * vector or alpha≈0) so the generation runs an exact unsteered baseline.
 */
export function makeResidualHook(
  api: Torchlette,
  vec: SteeringVector | null,
  alpha: number,
): ResidualHook | undefined {
  if (!vec || alpha === 0) return undefined;
  const { direction, layer, hiddenSize } = vec;
  // [hidden] → [1,1,hidden] so it broadcasts over x [batch, seq, hidden].
  // Persist it: generation runs under setStepScopedCleanup(true), which would
  // otherwise demote this once-per-generation tensor at the first markStep.
  const dir3d = api.registerState(api.reshape(direction, [1, 1, hiddenSize]));
  return (x, l) => {
    if (l !== layer) return x;
    // NB: no norm-preservation — Qwen3's residual norm is dominated by a few
    // massive-activation dims (attention sinks); rescaling the whole vector
    // back to the original norm shrinks those load-bearing dims and corrupts
    // generation even at small alpha. Plain additive steering it is.
    return api.add(x, api.mul(dir3d, alpha));
  };
}

/** Preset contrastive pairs for the UI. Multi-line = multiple examples per
 *  side, averaged into a cleaner (more obsessable) direction. */
export const STEERING_PRESETS: { name: string; pos: string; neg: string }[] = [
  {
    name: "Golden Gate Bridge",
    pos: [
      "The Golden Gate Bridge glows a brilliant orange over the San Francisco fog.",
      "I walked across the Golden Gate Bridge and gazed up at its towering red cables.",
      "Nothing compares to the Golden Gate Bridge stretching across the bay at sunset.",
      "The Golden Gate Bridge is the most magnificent landmark I have ever seen.",
      "We sailed beneath the Golden Gate Bridge as its great towers rose into the clouds.",
    ].join("\n"),
    neg: [
      "The morning coffee glows a warm brown in the quiet kitchen light.",
      "I walked across the parking lot and glanced up at the gray office building.",
      "Nothing compares to a quiet afternoon reading a book by the window.",
      "The spreadsheet is the most tedious document I have ever reviewed.",
      "We drove down the highway as the flat fields rolled past the windows.",
    ].join("\n"),
  },
  {
    name: "happy / sad",
    pos: [
      "I feel absolutely wonderful — today is bright and full of joy.",
      "What a delightful, cheerful morning; everything is going beautifully.",
      "I'm overflowing with happiness and can't stop smiling.",
    ].join("\n"),
    neg: [
      "I feel utterly miserable — today is bleak and full of despair.",
      "What a dreary, gloomy morning; everything is going terribly.",
      "I'm sinking in sadness and can't stop crying.",
    ].join("\n"),
  },
  {
    name: "formal / casual",
    pos: "formal professional polished courteous precise",
    neg: "casual slangy relaxed chatty informal",
  },
  {
    name: "verbose / terse",
    pos: "verbose elaborate detailed expansive thorough",
    neg: "terse brief curt minimal laconic",
  },
  {
    name: "angry / calm",
    pos: "angry furious enraged hostile irate",
    neg: "calm serene peaceful tranquil composed",
  },
];
