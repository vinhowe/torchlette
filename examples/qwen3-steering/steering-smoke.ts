/**
 * FAST Node/Dawn correctness proof for contrastive activation steering — no
 * browser. The steering logic is pure torchlette API, so this is the real
 * proof: load Qwen3, compute a happy/sad direction at a mid layer, then
 * generate a fixed neutral prompt twice through the SAME residualHook path
 * generate.ts uses — alpha=0 (baseline) and a strong positive alpha — and
 * assert the two outputs DIFFER. Prints both verbatim.
 *
 * Uses the local qwen3-1.7b checkpoint (0.6b is not cached on this box; the
 * steering math is identical). Run SOLO from repo root:
 *   npx tsx examples/qwen3-steering/steering-smoke.ts
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { AutoTokenizer } from "@huggingface/transformers";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "../qwen3/loader";
import {
  buildChatPrompt,
  QWEN3_STOP_TOKENS,
  sampleFromTopK,
} from "../../packages/qwen3-browser/src/generate";
import { computeSteeringVector, makeResidualHook } from "./src/lib/steering";
import type { ResidualHook } from "../../packages/qwen3-browser/src/model";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");

const POS = "happy joyful cheerful delighted upbeat";
const NEG = "sad depressed miserable gloomy despairing";
// Chat-style user turn (Qwen3 is an instruct model — see buildChatPrompt wrap below).
const PROMPT = "Tell me about your day.";
const ALPHA = 14;
const MAX_NEW = 40;

async function main() {
  const t0 = Date.now();
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_DIR);
  const tok = {
    encode: (text: string) => tokenizer.encode(text) as number[],
    decode: (ids: number[], o?: { skip_special_tokens?: boolean }) =>
      tokenizer.decode(ids, o) as string,
  };

  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 512 });
  const { numLayers, hiddenSize, vocabSize } = model.config;
  const layer = Math.floor(numLayers / 2);
  console.log(
    `[${Date.now() - t0}ms] loaded: ${numLayers} layers, hidden ${hiddenSize}`,
  );

  // ---- Compute the contrastive steering vector ----
  const tv = Date.now();
  const vec = await computeSteeringVector(api, model, tok, POS, NEG, layer);
  console.log(
    `[${Date.now() - t0}ms] steering vector [${hiddenSize}] @ layer ${layer} computed in ${Date.now() - tv}ms`,
  );

  // ---- Generation (mirrors generate.ts static-KV decode + residualHook) ----
  async function generate(alpha: number): Promise<string> {
    const hook: ResidualHook | undefined = makeResidualHook(api, vec, alpha);
    const promptIds = tok.encode(
      buildChatPrompt([{ role: "user", content: PROMPT }]),
    );
    const staticKV = model.allocStaticKV(512);
    const genIds: number[] = [];
    const prevScope = api.setStepScopedCleanup(true);
    try {
      let nextTok: number;
      {
        const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
        const { logits } = api.noGrad(() =>
          model.forward(idx, { staticKV, residualHook: hook }),
        );
        const top = await api.readTopK(logits, 64, {
          offset: (promptIds.length - 1) * vocabSize,
          length: vocabSize,
        });
        logits.dispose();
        nextTok = sampleFromTopK(top.values, top.indices, 0.7, 20, 0.95);
        await api.markStep();
      }
      let count = 0;
      while (count < MAX_NEW && !QWEN3_STOP_TOKENS.has(nextTok)) {
        genIds.push(nextTok);
        count++;
        const idx = api.tensorFromArray([nextTok], [1, 1]);
        const { logits } = api.noGrad(() =>
          model.forward(idx, { staticKV, residualHook: hook }),
        );
        const top = await api.readTopK(logits, 64, { length: vocabSize });
        logits.dispose();
        nextTok = sampleFromTopK(top.values, top.indices, 0.7, 20, 0.95);
        api.endStep();
        await api.markStep();
      }
      staticKV.k.length = 0;
      staticKV.v.length = 0;
      await api.markStep();
      return tok.decode(genIds, { skip_special_tokens: true });
    } finally {
      api.setStepScopedCleanup(prevScope);
    }
  }

  const tb = Date.now();
  const baseline = await generate(0);
  console.log(`[${Date.now() - t0}ms] baseline (α=0) in ${Date.now() - tb}ms`);
  const ts = Date.now();
  const steered = await generate(ALPHA);
  console.log(`[${Date.now() - t0}ms] steered (α=${ALPHA}) in ${Date.now() - ts}ms`);

  console.log("\n================= PROMPT =================");
  console.log(JSON.stringify(PROMPT));
  console.log("\n============ BASELINE (α = 0) ============");
  console.log(JSON.stringify(baseline));
  console.log(`\n=========== STEERED (α = ${ALPHA}) ===========`);
  console.log(JSON.stringify(steered));

  if (!baseline || !steered) throw new Error("a generation was empty");
  if (baseline === steered) throw new Error("STEERING HAD NO EFFECT: baseline == steered");
  console.log("\nPASS: baseline and steered outputs DIFFER (steering had an effect).");
  console.log("SMOKE DONE");
  process.exit(0);
}

main().catch((e) => {
  console.error("SMOKE FAILED:", e);
  process.exit(1);
});
