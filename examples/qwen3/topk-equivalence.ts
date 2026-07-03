/**
 * Top-K prefilter equivalence gate (fix b of the decode-latency work):
 *   - greedy: readTopK(...).indices[0] must be BIT-IDENTICAL to the
 *     full-logits first-max argmax, every step
 *   - sampling: the top-64 (value, index) set must match the full-logits
 *     top-64 computed with the same ordering contract (value desc, index asc)
 * over a prefill + 20 static-KV decode steps.
 *
 * Run: npx tsx examples/qwen3/topk-equivalence.ts
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"
const NUM_STEPS = 20;
const K = 64;

/** Reference top-K over a slice: (value desc, index asc), plus first-max argmax. */
function refTopK(
  flat: Float32Array,
  offset: number,
  length: number,
  k: number,
) {
  const vals = new Float32Array(k).fill(Number.NEGATIVE_INFINITY);
  const idxs = new Int32Array(k).fill(-1);
  for (let i = 0; i < length; i++) {
    const v = flat[offset + i];
    if (v > vals[k - 1]) {
      let p = k - 1;
      while (p > 0 && vals[p - 1] < v) {
        vals[p] = vals[p - 1];
        idxs[p] = idxs[p - 1];
        p--;
      }
      vals[p] = v;
      idxs[p] = i;
    }
  }
  // First-max argmax (linear scan, strict >)
  let best = 0;
  for (let i = 1; i < length; i++)
    if (flat[offset + i] > flat[offset + best]) best = i;
  return { vals, idxs, argmax: best };
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 256 });
  const vocab = model.config.vocabSize;

  const tokens = [...PROMPT];
  const staticKV = model.allocStaticKV(256);
  let failures = 0;

  const checkStep = async (
    logits: import("../../src/frontend/torchlette").Tensor,
    row: number,
    tag: string,
  ) => {
    const gpu = await api.readTopK(logits, K, {
      offset: row * vocab,
      length: vocab,
    });
    const flat = new Float32Array(await api.cpu(logits));
    const ref = refTopK(flat, row * vocab, vocab, K);
    let stepOk = true;
    if (gpu.indices[0] !== ref.argmax) {
      stepOk = false;
      console.log(
        `  ${tag}: GREEDY MISMATCH gpu=${gpu.indices[0]} ref=${ref.argmax}`,
      );
    }
    for (let i = 0; i < K; i++) {
      if (gpu.indices[i] !== ref.idxs[i] || gpu.values[i] !== ref.vals[i]) {
        stepOk = false;
        console.log(
          `  ${tag}: top-K MISMATCH at rank ${i}: gpu=(${gpu.indices[i]}, ${gpu.values[i]}) ref=(${ref.idxs[i]}, ${ref.vals[i]})`,
        );
        break;
      }
    }
    if (!stepOk) failures++;
    logits.dispose();
    return ref.argmax;
  };

  // Prefill (checks the offset path: last row of [1, S, V])
  {
    const idx = api.tensorFromArray(tokens, [1, tokens.length]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    tokens.push(await checkStep(logits, tokens.length - 1, "prefill"));
    await api.markStep();
  }

  for (let i = 0; i < NUM_STEPS; i++) {
    await api.beginStep();
    const idx = api.tensorFromArray([tokens[tokens.length - 1]], [1, 1]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    tokens.push(await checkStep(logits, 0, `step ${i}`));
    api.endStep();
    await api.markStep();
  }

  console.log(`tokens: ${tokens.join(",")}`);
  console.log(
    failures === 0
      ? `TOPK EQUIVALENCE PASS (prefill + ${NUM_STEPS} steps, K=${K}, greedy bit-identical)`
      : `TOPK EQUIVALENCE FAIL (${failures} steps mismatched)`,
  );
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
