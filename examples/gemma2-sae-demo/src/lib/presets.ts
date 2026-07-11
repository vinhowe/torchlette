/**
 * Steering presets — features on the layer-20 width-16k Gemma Scope SAE that
 * produce obvious, nameable steering effects at a reasonable α. Selected
 * empirically (examples/gemma2-sae/preset-{sweep,refine}.ts, greedy decode) and
 * confirmed against Neuronpedia's auto-interpretations. Each links to its
 * Neuronpedia page.
 *
 * Neuronpedia: https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/{feature}
 *
 * Effect examples (prompt "My favorite thing to do on the weekend is …", greedy):
 *   #12082 dogs  α=120 → "…to go to the DOG PARK. I love watching the dogs play
 *                         and run around…" (baseline: "…the farmers market…")
 *   #12887 SF    α=150 → "It's called the SAN FRANCISCO Bay Area… you can see the
 *                         GOLDEN GATE BRIDGE, the Bay Bridge…" (baseline: "…get
 *                         away from the hustle and bustle of the city…")
 *   #8993  bank  α=100 → pivots toward business / marketing / finance framing.
 * α is in residual-norm units (W_dec rows are unit-norm). Above ~150 the model
 * degenerates into repetition, so the presets sit in the coherent band.
 */

export type Preset = {
  name: string;
  feature: number;
  alpha: number;
  testPrompt: string;
  note: string;
};

export const PRESETS: Preset[] = [
  {
    name: "🐕 Dogs",
    feature: 12082,
    alpha: 120,
    testPrompt: "My favorite thing to do on the weekend is",
    note: 'Feature 12082 — "references to dogs as pets" (Neuronpedia). The clearest steering effect: the model steers its topic to dogs.',
  },
  {
    name: "🌉 Golden Gate",
    feature: 12887,
    alpha: 150,
    testPrompt: "I want to tell you about a place I love.",
    note: 'Feature 12887 — San Francisco / Golden Gate Bridge. At α=150 the model names the SF Bay Area and the Golden Gate Bridge; above ~180 it degenerates into "Francisco Francisco…" repetition. (Replaces the earlier #3124, which fired on the SF prompt but did not actually steer generation.)',
  },
  {
    name: "🏦 Banking",
    feature: 8993,
    alpha: 100,
    testPrompt: "I want to tell you about something interesting.",
    note: 'Feature 8993 — "banking and financial institutions" (Neuronpedia). Pivots the topic toward business and finance.',
  },
];

export const DEFAULT_PROMPT = "My favorite thing to do on the weekend is";
export const DEFAULT_ALPHA_RANGE = { min: -200, max: 200, step: 10 };
