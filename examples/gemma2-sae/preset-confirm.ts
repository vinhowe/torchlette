/**
 * Final preset confirmation — baseline vs steered for exactly the demo presets,
 * greedy, so the report shows the verified "Golden Gate moment" outputs. Uses
 * the SHIPPED generateChat + the same Σ α·W_dec[f] hook the app builds.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=1 \
 *      npx tsx examples/gemma2-sae/preset-confirm.ts
 */

import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { getWebGPUInitError, initWebGPU, setGPUMemoryLimit } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import type { ResidualHook } from "../../packages/gemma2-browser/src/model";
import { generateChat } from "../../packages/gemma2-browser/src/generate";
import { loadPretrainedGemma2 } from "../gemma2/loader";
import { GemmaScopeSAE } from "../../packages/gemma-scope-sae/src/sae";
import { loadSAEFromDir } from "../../packages/gemma-scope-sae/src/loader-node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/gemma-2-2b");
const SAE_DIR = path.join(__dirname, "../../ckpts/gemma-scope-2b-pt-res/sae-layer20-16k");

const PRESETS = [
  { name: "Dogs", feature: 12082, alpha: 120, prompt: "My favorite thing to do on the weekend is" },
  { name: "Golden Gate (#12887 @ 150)", feature: 12887, alpha: 150, prompt: "I want to tell you about a place I love." },
  { name: "Banking", feature: 8993, alpha: 100, prompt: "I want to tell you about something interesting." },
];

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);
  const api = new Torchlette("webgpu", { enableFusion: true });
  const weightDtype = (process.env.GEMMA2_DTYPE === "f32" ? "f32" : "f16") as "f32" | "f16";
  const model = await loadPretrainedGemma2(api, MODEL_DIR, { maxSeqLen: 256, weightDtype });
  const { config, params } = loadSAEFromDir(SAE_DIR);
  const sae = GemmaScopeSAE.load(api, config, params, { dtype: "f32" });
  const layer = sae.config.layer;
  const dModel = sae.config.dModel;

  // Resolve @huggingface/transformers from the demo package (only dep there;
  // not hoisted to root, and tsx's TS-path resolver mis-resolves the bare
  // specifier from this non-workspace example dir).
  const { createRequire } = await import("node:module");
  const req = createRequire(
    path.join(__dirname, "../gemma2-sae-demo/package.json"),
  );
  const hfMain = req.resolve("@huggingface/transformers");
  const { AutoTokenizer } = await import(
    pathToFileURL(hfMain).href
  );
  const tk = (await AutoTokenizer.from_pretrained(MODEL_DIR)) as never;
  const tokenizer = {
    encode: (t: string) => (tk as { encode(t: string): number[] }).encode(t),
    decode: (ids: number[], o?: { skip_special_tokens?: boolean }) =>
      (tk as { decode(ids: number[], o?: unknown): string }).decode(ids, o),
  };

  const gen = async (feature: number, alpha: number, prompt: string) => {
    let hook: ResidualHook | undefined;
    if (alpha !== 0) {
      const row = sae.featureDirectionCpu(feature);
      const combined = new Float32Array(dModel);
      for (let i = 0; i < dModel; i++) combined[i] = alpha * row[i];
      const vec = api.persist(api.tensorFromArray(combined, [1, 1, dModel]));
      hook = (x, l) => (l === layer ? api.add(x, vec) : x);
    }
    let out = "";
    await generateChat(
      api, model, tokenizer, [{ role: "user", content: prompt }],
      { onDelta: (d) => (out += d), onReplace: (t) => (out = t) },
      { maxNewTokens: 44, temperature: 0, topK: 40, topP: 0.95, chat: false, residualHook: hook },
    );
    return out;
  };

  for (const p of PRESETS) {
    console.log(`\n##### ${p.name} — feature #${p.feature} @ α=${p.alpha} #####`);
    console.log(`  prompt:   ${JSON.stringify(p.prompt)}`);
    console.log(`  baseline: ${JSON.stringify((await gen(p.feature, 0, p.prompt)).trim())}`);
    console.log(`  steered:  ${JSON.stringify((await gen(p.feature, p.alpha, p.prompt)).trim())}`);
  }
  console.log("\nDONE");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
