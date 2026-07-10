/**
 * Steering presets — features on the layer-20 width-16k Gemma Scope SAE that
 * produce obvious, nameable steering effects at a reasonable α. Each preset
 * names its feature index + a default α and a suggested test prompt. The
 * `feature` indices + α were selected empirically by the sweep in
 * examples/gemma2-sae/preset-sweep.ts (see the report), and each links to its
 * Neuronpedia page for the human-readable interpretation.
 *
 * Neuronpedia: https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/{feature}
 */

export type Preset = {
  name: string;
  /** Feature index on the layer-20 16k SAE. */
  feature: number;
  /** Default steering coefficient (residual-norm units; W_dec rows are unit). */
  alpha: number;
  /** A prompt that shows the effect clearly. */
  testPrompt: string;
  /** Human label (from Neuronpedia / the sweep). */
  note: string;
};

// Filled/validated by the preset sweep. These are the demo's "Golden Gate
// moment" — wired so a visitor sees an obvious effect without knowing indices.
export const PRESETS: Preset[] = [];

export const DEFAULT_PROMPT = "Tell me about your day.";
export const DEFAULT_ALPHA_RANGE = { min: -400, max: 400, step: 10 };
