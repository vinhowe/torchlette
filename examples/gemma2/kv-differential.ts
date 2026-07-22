/**
 * Gemma-2 KV-cache differential (M2 correctness gate): greedy generation WITH
 * cache (prefill + per-token decode) must produce the identical token sequence
 * as WITHOUT cache (full recompute). Three cached arms: cat-cache, static-KV,
 * and TAPED static-KV (replay). Gates the decode path (decomposed attention +
 * soft-cap + per-layer global/local mask, RoPE at posOffset>0, cache append,
 * tape replay with the modifier-keyed bucket).
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *      TORCHLETTE_STEP_TAPE=1 npx tsx examples/gemma2/kv-differential.ts
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUInitError,
  initWebGPU,
  setGPUMemoryLimit,
} from "../../src/backend/webgpu";
import { Torchlette, type Tensor } from "../../src/frontend/torchlette";
import type { KVCache } from "../../packages/gemma2-browser/src/model";
import { kvBucketLen } from "../../packages/gemma2-browser/src/model";
import { loadPretrainedGemma2 } from "./loader";
import { assertLogitsSane } from "../../tools/parity-sanity";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/gemma-2-2b");
// "<bos> The capital of France is"
const PROMPT = [2, 651, 6037, 576, 6081, 603];
const NUM_NEW = 12;

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);
  const api = new Torchlette("webgpu", { enableFusion: true });
  const weightDtype = (process.env.GEMMA2_DTYPE === "f32" ? "f32" : "f16") as
    | "f32"
    | "f16";
  const model = await loadPretrainedGemma2(api, MODEL_DIR, {
    maxSeqLen: 256,
    weightDtype,
  });
  const vocab = model.config.vocabSize;

  const argmaxLast = async (
    logits: Tensor,
    pos: number,
    sanity = false,
  ): Promise<number> => {
    const flat = new Float32Array(await logits.cpu());
    const off = pos * vocab;
    if (sanity) assertLogitsSane(flat.subarray(off, off + vocab), "gemma2-kv/no-cache");
    let best = 0;
    for (let v = 1; v < vocab; v++) if (flat[off + v] > flat[off + best]) best = v;
    logits.dispose();
    return best;
  };

  // --- Without cache: full recompute
  const noCache = [...PROMPT];
  for (let i = 0; i < NUM_NEW; i++) {
    const idx = api.tensorFromArray(noCache, [1, noCache.length]);
    const { logits } = api.noGrad(() => model.forward(idx));
    noCache.push(await argmaxLast(logits, noCache.length - 1, i === 0));
    await api.markStep();
  }

  // --- With cat cache
  const cached = [...PROMPT];
  let kv: KVCache[] | undefined;
  {
    const idx = api.tensorFromArray(cached, [1, cached.length]);
    const { logits, presentKVs } = api.noGrad(() => model.forward(idx));
    kv = presentKVs;
    cached.push(await argmaxLast(logits, cached.length - 1));
    await api.markStep();
  }
  for (let i = 1; i < NUM_NEW; i++) {
    const posOffset = cached.length - 1;
    const idx = api.tensorFromArray([cached[cached.length - 1]], [1, 1]);
    const { logits, presentKVs } = api.noGrad(() =>
      model.forward(idx, { pastKVs: kv, posOffset }),
    );
    kv = presentKVs;
    cached.push(await argmaxLast(logits, 0));
    await api.markStep();
  }

  // --- With static cache
  const stat = [...PROMPT];
  const staticKV = model.allocStaticKV(256);
  api.setStepScopedCleanup(true);
  {
    const idx = api.tensorFromArray(stat, [1, stat.length]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    stat.push(await argmaxLast(logits, stat.length - 1));
    await api.markStep();
  }
  for (let i = 1; i < NUM_NEW; i++) {
    const idx = api.tensorFromArray([stat[stat.length - 1]], [1, 1]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    stat.push(await argmaxLast(logits, 0));
    await api.markStep();
  }

  // --- Taped static cache
  const taped = [...PROMPT];
  const tapedKV = model.allocStaticKV(256);
  {
    const idx = api.tensorFromArray(taped, [1, taped.length]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV: tapedKV }));
    taped.push(await argmaxLast(logits, taped.length - 1));
    await api.markStep();
  }
  const tapedDecode = api.capture(
    (idx: Tensor) =>
      api.noGrad(() => model.forward(idx, { staticKV: tapedKV }).logits),
    {
      key: () =>
        `kv:bkt${kvBucketLen(tapedKV.len + 1, tapedKV.maxSeqLen)}:mod${model.attnModKey}`,
    },
  );
  for (let i = 1; i < NUM_NEW; i++) {
    const idx = api.tensorFromArray([taped[taped.length - 1]], [1, 1]);
    const logits = (await tapedDecode(idx)) as Tensor;
    taped.push(await argmaxLast(logits, 0));
    await api.markStep();
  }

  console.log("no-cache:", JSON.stringify(noCache));
  console.log("cached:  ", JSON.stringify(cached));
  console.log("static:  ", JSON.stringify(stat));
  console.log("taped:   ", JSON.stringify(taped));
  const match =
    JSON.stringify(noCache) === JSON.stringify(cached) &&
    JSON.stringify(noCache) === JSON.stringify(stat) &&
    JSON.stringify(noCache) === JSON.stringify(taped);
  console.log(match ? "KV DIFFERENTIAL PASS (cat + static + taped)" : "KV DIFFERENTIAL FAIL");
  process.exit(match ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
