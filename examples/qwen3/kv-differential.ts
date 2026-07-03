/**
 * KV-cache differential: greedy generation WITH cache (prefill + per-token
 * decode) must produce the identical token sequence as WITHOUT cache (full
 * recompute per token). Gates the decode path (decomposed attention branch,
 * RoPE at posOffset>0, cache append).
 *
 * Run: npx tsx examples/qwen3/kv-differential.ts
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import type { KVCache } from "./model";
import { loadPretrainedQwen3 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"
const NUM_NEW = 16;
const SKIP_MARKSTEP = process.env.KV_DIFF_NO_MARKSTEP === "1";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 256 });
  const vocab = model.config.vocabSize;

  const argmaxLast = async (
    logits: import("../../src/frontend/torchlette").Tensor,
    pos: number,
  ) => {
    const flat = new Float32Array(await logits.cpu());
    const off = pos * vocab;
    let best = 0;
    for (let v = 1; v < vocab; v++)
      if (flat[off + v] > flat[off + best]) best = v;
    logits.dispose();
    return best;
  };

  // --- Without cache: full recompute per token
  const noCache = [...PROMPT];
  for (let i = 0; i < NUM_NEW; i++) {
    const idx = api.tensorFromArray(noCache, [1, noCache.length]);
    const { logits } = api.noGrad(() => model.forward(idx));
    noCache.push(await argmaxLast(logits, noCache.length - 1));
    if (!SKIP_MARKSTEP) await api.markStep();
  }

  // --- With cache: prefill then decode
  const cached = [...PROMPT];
  let kv: KVCache[] | undefined;
  {
    const idx = api.tensorFromArray(cached, [1, cached.length]);
    const { logits, presentKVs } = api.noGrad(() => model.forward(idx));
    kv = presentKVs;
    cached.push(await argmaxLast(logits, cached.length - 1));
    if (!SKIP_MARKSTEP) await api.markStep();
  }
  for (let i = 1; i < NUM_NEW; i++) {
    const last = cached[cached.length - 1];
    const posOffset = cached.length - 1;
    const idx = api.tensorFromArray([last], [1, 1]);
    const { logits, presentKVs } = api.noGrad(() =>
      model.forward(idx, { pastKVs: kv, posOffset }),
    );
    kv = presentKVs;
    cached.push(await argmaxLast(logits, 0));
    if (!SKIP_MARKSTEP) await api.markStep();
  }

  // --- With STATIC cache: prefill writes into preallocated buffers, decode
  // reads the bucketed prefix under a padding mask (shape-stable plans).
  //
  // Ceremony-free step-scoped cleanup (setStepScopedCleanup): the same loop
  // shape as the real generate loops — bare markStep() reclaims each
  // interval's graph temporaries. NOT enabled for the cat loop above: its
  // presentKVs are created inside an interval and held across markStep (the
  // adversarial case for implicit boundaries — they'd need api.persist()).
  const stat = [...PROMPT];
  const staticKV = model.allocStaticKV(256);
  if (!SKIP_MARKSTEP) api.setStepScopedCleanup(true);
  {
    const idx = api.tensorFromArray(stat, [1, stat.length]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    stat.push(await argmaxLast(logits, stat.length - 1));
    if (!SKIP_MARKSTEP) await api.markStep();
  }
  for (let i = 1; i < NUM_NEW; i++) {
    const idx = api.tensorFromArray([stat[stat.length - 1]], [1, 1]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    stat.push(await argmaxLast(logits, 0));
    if (!SKIP_MARKSTEP) await api.markStep();
  }

  console.log("no-cache:", JSON.stringify(noCache));
  console.log("cached:  ", JSON.stringify(cached));
  console.log("static:  ", JSON.stringify(stat));
  const match =
    JSON.stringify(noCache) === JSON.stringify(cached) &&
    JSON.stringify(noCache) === JSON.stringify(stat);
  console.log(
    match ? "KV DIFFERENTIAL PASS (cat + static)" : "KV DIFFERENTIAL FAIL",
  );
  process.exit(match ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
