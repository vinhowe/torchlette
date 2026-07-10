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
 *   #3124  SF    α=120 → San-Francisco / Bay-Area flavored continuations.
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
    name: "🌉 San Francisco",
    feature: 3124,
    alpha: 150,
    testPrompt: "I want to tell you about a place I love.",
    note: 'Feature 3124 — "San Francisco / Bay Area locations" (Neuronpedia). The literal Golden-Gate-analog concept; a subtler pull than Dogs — nudge α higher to strengthen it.',
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
