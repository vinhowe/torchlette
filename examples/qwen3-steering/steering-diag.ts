/**
 * Diagnose why steering seems to have no interpretable effect.
 * Measures: raw (pre-normalize) diff-vector norm, and the mean per-token
 * residual-stream norm at the inject layer — so we can see how alpha*unit
 * compares to what it's being added to. Then sweeps alpha (past the slider max)
 * printing generations, to tell a SCALE problem (effect appears at big alpha)
 * from a DIRECTION problem (never becomes interpretable, just degrades).
 *
 * Run: npx tsx examples/qwen3-steering/steering-diag.ts   (solo GPU)
 */
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "../qwen3/loader";
import { AutoTokenizer } from "@huggingface/transformers";
import {
  buildChatPrompt,
  QWEN3_STOP_TOKENS,
  sampleFromTopK,
} from "../../packages/qwen3-browser/src/generate";
import { computeSteeringVector, makeResidualHook, STEERING_PRESETS } from "./src/lib/steering";
import type { ResidualHook } from "../../packages/qwen3-browser/src/model";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const GG = STEERING_PRESETS.find((p) => p.name === "Golden Gate Bridge")!;
const POS = GG.pos; // multi-line → averaged over examples
const NEG = GG.neg;
const PROMPT = "What should I do this weekend?";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_DIR);
  const tok = {
    encode: (t: string) => tokenizer.encode(t) as number[],
    decode: (ids: number[], o?: { skip_special_tokens?: boolean }) =>
      tokenizer.decode(ids, o) as string,
  };
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 512 });
  const { numLayers, hiddenSize, vocabSize } = model.config;
  const layer = Math.floor(numLayers / 2);

  // --- Measure raw diff-vector norm (pre-normalize) and residual norm at inject layer.
  const norms = await api.noGrad(async () => {
    const enc = (t: string) => api.tensorFromArray(tok.encode(t), [1, tok.encode(t).length]);
    const p = model.forward(enc(POS), { collectHidden: true });
    const n = model.forward(enc(NEG), { collectHidden: true });
    const mp = api.reshape(api.mean(p.hidden![layer + 1], { dim: 1 }), [hiddenSize]);
    const mn = api.reshape(api.mean(n.hidden![layer + 1], { dim: 1 }), [hiddenSize]);
    const diff = api.sub(mp, mn);
    const diffNorm = Math.sqrt(await api.item(api.sum(api.mul(diff, diff))));
    return { diffNorm };
  });

  // residual per-token norm on the chat prompt prefill
  const promptIds = tok.encode(buildChatPrompt([{ role: "user", content: PROMPT }]));
  const residNorm = await api.noGrad(async () => {
    const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
    const out = model.forward(idx, { collectHidden: true });
    const h = out.hidden![layer + 1]; // [1, seq, hidden]
    const sq = api.sum(api.mul(h, h), { dim: 2 }); // [1, seq] per-token sumSq
    const meanSumSq = await api.item(api.mean(sq, { dim: 1 }));
    return Math.sqrt(meanSumSq);
  });
  await api.markStep();

  console.log(`\nlayer ${layer} | hidden ${hiddenSize}`);
  console.log(`RAW diff-vector norm ||meanPos-meanNeg|| = ${norms.diffNorm.toFixed(2)}`);
  console.log(`mean per-token RESIDUAL norm at layer = ${residNorm.toFixed(2)}`);
  console.log(`(steering adds alpha*UNIT; so alpha vs ${residNorm.toFixed(0)} is the ratio that matters)\n`);

  async function gen(
    vec: Awaited<ReturnType<typeof computeSteeringVector>>,
    alpha: number,
  ): Promise<string> {
    const hook: ResidualHook | undefined = makeResidualHook(api, vec, alpha);
    const staticKV = model.allocStaticKV(512);
    const prev = api.setStepScopedCleanup(true);
    const ids: number[] = [];
    try {
      let next: number;
      {
        const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
        const { logits } = api.noGrad(() => model.forward(idx, { staticKV, residualHook: hook }));
        const top = await api.readTopK(logits, 64, { offset: (promptIds.length - 1) * vocabSize, length: vocabSize });
        logits.dispose();
        next = sampleFromTopK(top.values, top.indices, 0.7, 1, 1); // greedy (topK=1): deterministic, isolates steering
        await api.markStep();
      }
      let c = 0;
      while (c < 32 && !QWEN3_STOP_TOKENS.has(next)) {
        ids.push(next); c++;
        const idx = api.tensorFromArray([next], [1, 1]);
        const { logits } = api.noGrad(() => model.forward(idx, { staticKV, residualHook: hook }));
        const top = await api.readTopK(logits, 64, { offset: 0, length: vocabSize });
        logits.dispose();
        next = sampleFromTopK(top.values, top.indices, 0.7, 1, 1);
        await api.markStep();
      }
    } finally {
      api.setStepScopedCleanup(prev);
      staticKV.k.length = 0; staticKV.v.length = 0;
      await api.markStep();
    }
    return tok.decode(ids, { skip_special_tokens: true });
  }

  // Layer × alpha sweep (greedy) — find a layer with a WIDE coherent happy band.
  const base = await gen(await computeSteeringVector(api, model, tok, POS, NEG, 14), 0);
  console.log(`\nBASELINE (unsteered): ${JSON.stringify(base)}`);
  for (const L of [16, 20]) {
    const vec = await computeSteeringVector(api, model, tok, POS, NEG, L);
    console.log(`\n######## LAYER ${L} ########`);
    for (const a of [2.5, 3.5, 4.5, 6, 8]) {
      console.log(`  α=${a}: ${JSON.stringify(await gen(vec, a))}`);
    }
  }
  process.exit(0);
}
main().catch((e) => { console.error("FAILED:", e); process.exit(1); });
