/**
 * SAMPLER economics: submits/token + ms/token for the on-device TOP-K/TOP-P
 * block vs the per-token host loop (readTopK readback + host sampleFromTopK) —
 * the demos' actual sampler on both arms. Same top-k/top-p/temperature; the
 * block amortizes the per-token host fence over K tokens (its win is the same
 * host-tax amortization class as greedy/gumbel, plus the extra deviceTopK +
 * filter dispatches, reported honestly).
 *
 * Run: eval "$(tools/pick-gpu.sh)"; VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX \
 *   LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-uk-topk-econ.ts
 */
import {
  decodeBlock,
  sampleFromTopK,
} from "../packages/qwen3-browser/src/generate";
import type { Qwen3Config } from "../packages/qwen3-browser/src/model";
import { Qwen3 } from "../packages/qwen3-browser/src/model";
import {
  getSubmitCount,
  getWebGPUInitError,
  initWebGPU,
  resetSubmitCount,
} from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const CONFIG: Qwen3Config = {
  vocabSize: 4096,
  hiddenSize: 512,
  numLayers: 8,
  numHeads: 8,
  numKVHeads: 4,
  headDim: 64,
  intermediateSize: 1024,
  ropeTheta: 1e6,
  rmsNormEps: 1e-6,
  maxSeqLen: 256,
};
const TOPK = 20;
const TOPP = 0.95;
const TEMP = 0.7;

async function prefill(api: Torchlette, model: Qwen3, ids: number[], kv: any) {
  const V = CONFIG.vocabSize;
  const idx = api.tensorFromArray(ids, [1, ids.length]);
  const { logits } = api.noGrad(() => model.forward(idx, { staticKV: kv }));
  const top = await api.readTopK(logits, 64, {
    offset: (ids.length - 1) * V,
    length: V,
  });
  logits.dispose();
  await api.markStep();
  return sampleFromTopK(top.values, top.indices, TEMP, TOPK, TOPP);
}

/** Host per-token loop: forward → readTopK → host sampleFromTopK, per token. */
async function hostLoop(api: Torchlette, model: Qwen3, ids: number[], N: number) {
  const V = CONFIG.vocabSize;
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  let tok = await prefill(api, model, ids, kv);
  for (let j = 0; j < N; j++) {
    const logits = api.noGrad(
      () => model.forward(api.tensorFromArray([tok], [1, 1]), { staticKV: kv }).logits,
    );
    const top = await api.readTopK(logits, 64, { length: V });
    logits.dispose();
    tok = sampleFromTopK(top.values, top.indices, TEMP, TOPK, TOPP);
    await api.markStep();
  }
}

/** Block: decodeBlock with the on-device filtered sampler, K tokens/readback. */
async function blockLoop(
  api: Torchlette,
  model: Qwen3,
  ids: number[],
  N: number,
  K: number,
) {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  let tok = await prefill(api, model, ids, kv);
  let n = 0;
  while (n < N) {
    const { ids: blk } = await decodeBlock(api, model, kv, tok, K, {
      sample: { temperature: TEMP, seed: 4242, topK: TOPK, topP: TOPP },
    });
    await api.markStep();
    n += blk.length;
    tok = blk[blk.length - 1];
  }
}

function median(xs: number[]) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const model = new Qwen3(api, { ...CONFIG });
  const prompt = Array.from({ length: 16 }, (_, i) => (i * 7 + 3) % CONFIG.vocabSize);
  const N = 48;
  const REPEAT = 3;

  console.log(
    `=== SAMPLER economics (V100/sivri, random-init Qwen3 ${CONFIG.numLayers}L/${CONFIG.hiddenSize}d, ` +
      `topK=${TOPK} topP=${TOPP} T=${TEMP}, N=${N}) ===\n`,
  );

  const meas = async (fn: () => Promise<void>) => {
    const ms: number[] = [];
    const subs: number[] = [];
    await fn(); // warmup
    await api.markStep();
    for (let r = 0; r < REPEAT; r++) {
      resetSubmitCount();
      const t0 = performance.now();
      await fn();
      ms.push((performance.now() - t0) / N);
      subs.push(getSubmitCount() / N);
      await api.markStep();
    }
    return { ms: median(ms), subs: median(subs) };
  };

  const host = await meas(() => hostLoop(api, model, prompt, N));
  console.log(
    `host loop  : ${host.subs.toFixed(1)} submits/tok, ${host.ms.toFixed(2)} ms/tok`,
  );
  console.log("\n| K | submits/tok | ms/tok | host/block ms | host/block subs |");
  console.log("|---|-------------|--------|---------------|-----------------|");
  for (const K of [4, 8]) {
    const b = await meas(() => blockLoop(api, model, prompt, N, K));
    console.log(
      `| ${K} | ${b.subs.toFixed(1)} | ${b.ms.toFixed(2)} | ${(host.ms / b.ms).toFixed(2)}x | ${(host.subs / b.subs).toFixed(2)}x |`,
    );
  }
  console.log(
    "\nNote: V100/sivri (the correctness box) — NOT comparable to the A100 " +
      "campaign multiplier table. Block = deviceTopK + top-p filter + Gumbel " +
      "per step (extra dispatches vs greedy), one readback per K tokens.",
  );
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
