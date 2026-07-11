/**
 * Preset finder / steering-effect gate. Loads Gemma-2-2B + the layer-20 SAE,
 * then for a set of CANDIDATE features (discovered by inspecting themed prompts)
 * generates baseline vs steered completions at a few α, so we can pick the
 * features with obvious, nameable, thematically-coherent effects to wire as the
 * demo's presets.
 *
 * Two phases:
 *  1. DISCOVER: run themed prompts through the SAE, print the top features that
 *     fire — candidates whose Neuronpedia interpretation matches the theme.
 *  2. SWEEP: for each (feature, α), generate a completion and print it next to
 *     the α=0 baseline. Read the output to judge coherence + effect.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *      TORCHLETTE_STEP_TAPE=1 npx tsx examples/gemma2-sae/preset-sweep.ts
 */

import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  getWebGPUInitError,
  initWebGPU,
  setGPUMemoryLimit,
} from "../../src/backend/webgpu";
import { Torchlette, type Tensor } from "../../src/frontend/torchlette";
import type { Gemma2, ResidualHook } from "../../packages/gemma2-browser/src/model";
import { generateChat } from "../../packages/gemma2-browser/src/generate";
import { loadPretrainedGemma2 } from "../gemma2/loader";
import { GemmaScopeSAE, neuronpediaUrl } from "../../packages/gemma-scope-sae/src/sae";
import { loadSAEFromDir } from "../../packages/gemma-scope-sae/src/loader-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/gemma-2-2b");
const SAE_DIR = path.join(
  __dirname,
  "../../ckpts/gemma-scope-2b-pt-res/sae-layer20-16k",
);

// Themed probe prompts → the top features that fire are candidate steering dirs.
const THEMES: { name: string; prompt: string }[] = [
  { name: "golden-gate", prompt: "The Golden Gate Bridge towers over San Francisco Bay in the fog." },
  { name: "dogs", prompt: "My dog loves to run and play fetch in the park every morning." },
  { name: "python", prompt: "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr" },
  { name: "anger", prompt: "I am absolutely furious and outraged about what happened today." },
  { name: "ocean", prompt: "The vast blue ocean stretched to the horizon, waves crashing on the shore." },
  { name: "money", prompt: "The bank reported record profits and the stock price soared on Wall Street." },
];

async function encodeIds(
  api: Torchlette,
  tokenizer: { encode(t: string): number[] },
  text: string,
): Promise<Tensor> {
  const ids = tokenizer.encode(text);
  if (ids[0] !== 2) ids.unshift(2);
  return api.tensorFromArray(ids, [1, ids.length]);
}

/** Top-K features (by max over seq) that fire for a prompt. */
async function topFeatures(
  api: Torchlette,
  model: Gemma2,
  sae: GemmaScopeSAE,
  idx: Tensor,
  k: number,
): Promise<{ feature: number; act: number }[]> {
  const layer = sae.config.layer;
  const F = sae.config.numFeatures;
  const flat = await api.noGrad(async () => {
    let captured: Tensor | null = null;
    const hook: ResidualHook = (x, l) => {
      if (l === layer) captured = x;
      return x;
    };
    model.forward(idx, { residualHook: hook });
    const resid = captured as unknown as Tensor;
    const [, seq, dModel] = resid.shape;
    const acts = sae.encode(api, api.reshape(resid, [seq, dModel]));
    return { arr: new Float32Array(await acts.cpu()), seq };
  });
  // Aggregate max over CONTENT positions only (skip position 0 = <bos>, whose
  // attention-sink features fire identically on every prompt and swamp the
  // content-specific ones).
  const agg = new Float32Array(F);
  const start = flat.seq > 1 ? 1 : 0;
  for (let s = start; s < flat.seq; s++)
    for (let f = 0; f < F; f++) {
      const v = flat.arr[s * F + f];
      if (v > agg[f]) agg[f] = v;
    }
  const order = Array.from({ length: F }, (_, i) => i).sort((a, b) => agg[b] - agg[a]);
  await api.markStep();
  return order.slice(0, k).map((feature) => ({ feature, act: agg[feature] }));
}

/** Generation via the SHIPPED generateChat (raw completion mode) with an
 *  optional SAE steering hook (Σ α·W_dec[f]). Persisting the steering vec
 *  BEFORE generateChat enters its step scope keeps it alive across markStep. */
async function generate(
  api: Torchlette,
  model: Gemma2,
  sae: GemmaScopeSAE,
  tokenizer: {
    encode(t: string): number[];
    decode(ids: number[], o?: { skip_special_tokens?: boolean }): string;
  },
  prompt: string,
  steer: { feature: number; alpha: number }[],
  maxNew: number,
  temperature = 0.7,
): Promise<string> {
  const layer = sae.config.layer;
  const dModel = sae.config.dModel;
  let hook: ResidualHook | undefined;
  if (steer.length > 0) {
    const combined = new Float32Array(dModel);
    for (const { feature, alpha } of steer) {
      const row = sae.featureDirectionCpu(feature);
      for (let i = 0; i < dModel; i++) combined[i] += alpha * row[i];
    }
    const vec = api.persist(api.tensorFromArray(combined, [1, 1, dModel]));
    hook = (x, l) => (l === layer ? api.add(x, vec) : x);
  }
  let out = "";
  await generateChat(
    api,
    model,
    tokenizer,
    [{ role: "user", content: prompt }],
    { onDelta: (d) => (out += d), onReplace: (t) => (out = t) },
    { maxNewTokens: maxNew, temperature, topK: 40, topP: 0.95, chat: false, residualHook: hook },
  );
  return out;
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);
  const api = new Torchlette("webgpu", { enableFusion: true });

  const weightDtype = (process.env.GEMMA2_DTYPE === "f32" ? "f32" : "f16") as "f32" | "f16";
  const model = await loadPretrainedGemma2(api, MODEL_DIR, { maxSeqLen: 256, weightDtype });
  const { config, params } = loadSAEFromDir(SAE_DIR);
  const sae = GemmaScopeSAE.load(api, config, params, { dtype: "f32" });

  const { createRequire } = await import("node:module");
  const req = createRequire(
    path.join(__dirname, "../gemma2-sae-demo/package.json"),
  );
  const { AutoTokenizer } = await import(
    pathToFileURL(req.resolve("@huggingface/transformers")).href
  );
  const tk = (await AutoTokenizer.from_pretrained(MODEL_DIR)) as never;
  const tokenizer = {
    encode: (t: string) => (tk as { encode(t: string): number[] }).encode(t),
    decode: (ids: number[], o?: unknown) =>
      (tk as { decode(ids: number[], o?: unknown): string }).decode(ids, o),
  };

  // ---- Phase 1: DISCOVER candidate features per theme ----
  console.log("\n========== DISCOVER: top features per themed prompt ==========");
  const candidates = new Map<string, number[]>();
  for (const theme of THEMES) {
    const idx = await encodeIds(api, tokenizer, theme.prompt);
    const top = await topFeatures(api, model, sae, idx, 8);
    // Skip the always-on high-frequency features (fire on every prompt).
    const HIGH_FREQ = new Set([6631, 743, 5052, 16057, 9479]);
    const themed = top.filter((t) => !HIGH_FREQ.has(t.feature));
    candidates.set(theme.name, themed.slice(0, 6).map((t) => t.feature));
    console.log(`\n[${theme.name}] ${JSON.stringify(theme.prompt.slice(0, 50))}`);
    for (const t of top)
      console.log(
        `  #${t.feature.toString().padStart(5)} act=${t.act.toFixed(1)}  ${neuronpediaUrl(sae.config.neuronpediaSaeId, t.feature)}${HIGH_FREQ.has(t.feature) ? "  (high-freq, skip)" : ""}`,
      );
  }

  // ---- Phase 2: SWEEP α — for golden-gate, sweep the TOP-6 candidates (the
  // shipped #3124 was candidate[0] and does NOT steer to SF at any coherent α);
  // for other themes just the top candidate. ----
  console.log("\n========== SWEEP: baseline vs steered ==========");
  const genPrompt = "I want to tell you about a place I love.";
  const alphas = [0, 100, 150, 200];
  for (const [theme, feats] of candidates) {
    const featsToSweep = theme === "golden-gate" ? feats.slice(0, 6) : feats.slice(0, 1);
    for (const feature of featsToSweep) {
      if (feature === undefined) continue;
      console.log(`\n##### theme=${theme} feature=#${feature} #####`);
      console.log(`  neuronpedia: ${neuronpediaUrl(sae.config.neuronpediaSaeId, feature)}`);
      for (const alpha of alphas) {
        const steer = alpha === 0 ? [] : [{ feature, alpha }];
        // Greedy (temp 0) for a reproducible baseline-vs-steered comparison.
        const out = await generate(api, model, sae, tokenizer, genPrompt, steer, 40, 0);
        console.log(`  α=${alpha.toString().padStart(4)}: ${JSON.stringify(out.slice(0, 220))}`);
      }
    }
  }

  console.log("\nDONE");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
