/**
 * Real-model operand-path gate (task #93 phase 2): load the REAL Qwen3-1.7B
 * twice — f16 baseline and int8-64 projections (via the weightFormat operand
 * path) — and greedy-decode 20 tokens each from the same prompt. Asserts:
 *
 *  - Gate 3 (generation coherence): the int8 decode produces coherent output and
 *    matches the f16 greedy decode for the first several tokens (early
 *    divergence = red flag; deep divergence is expected & fine).
 *  - Gate 2 (logit parity): at each shared step, top-1 next-token agreement and
 *    max-abs logit drift ≤ 0.5 vs f16 (the phase-1 drift budget).
 *
 * This is the gate phase 1 couldn't run (the 28-layer forward hung standalone);
 * here the projections run inside the model's own forward via the operand path.
 *
 * Run: TORCHLETTE_STRICT_GPU=1 npx tsx examples/qwen3/gen-quant-realweight.ts
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { loadPretrainedQwen3, type Qwen3 } from "./loader";
import type { WeightFormatName } from "../../src/backend/types";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
// "The capital of France is"
const PROMPT = [785, 6722, 315, 9625, 374];
const NUM_NEW = 20;
const MATCH_PREFIX = 6; // first N generated tokens must agree with f16
const DRIFT_TOL = 0.5;

async function greedyDecode(
  api: Torchlette,
  model: Qwen3,
  vocab: number,
): Promise<{ tokens: number[]; lastLogits: Float32Array[] }> {
  const tokens = PROMPT.slice();
  const lastLogits: Float32Array[] = [];
  for (let step = 0; step < NUM_NEW; step++) {
    const idx = api.tensorFromArray(tokens, [1, tokens.length]);
    const { logits } = api.noGrad(() => model.forward(idx));
    const flat = new Float32Array(await logits.cpu());
    const off = (tokens.length - 1) * vocab;
    const row = flat.subarray(off, off + vocab);
    lastLogits.push(new Float32Array(row));
    let best = 0;
    for (let v = 1; v < vocab; v++) if (row[v] > row[best]) best = v;
    tokens.push(best);
    process.stderr.write(".");
    await api.markStep();
  }
  process.stderr.write("\n");
  return { tokens: tokens.slice(PROMPT.length), lastLogits };
}

async function loadAndDecode(
  weightFormat?: WeightFormatName,
): Promise<{ tokens: number[]; lastLogits: Float32Array[] }> {
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 128,
    weightDtype: "f16",
    weightFormat,
  });
  const out = await greedyDecode(api, model, model.config.vocabSize);
  return out;
}

async function main() {
  process.env.TORCHLETTE_COMPILED_PLAN = "0";
  process.env.TORCHLETTE_GENERATED_PLAN = "0";
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

  console.log("=== f16 baseline ===");
  const base = await loadAndDecode(undefined);
  console.log("f16 tokens:", JSON.stringify(base.tokens));

  console.log("=== int8-64 (operand path) ===");
  const q = await loadAndDecode("int8-64");
  console.log("int8 tokens:", JSON.stringify(q.tokens));

  // Gate 3: prefix agreement.
  let prefixMatch = 0;
  for (let i = 0; i < NUM_NEW; i++) {
    if (base.tokens[i] === q.tokens[i]) prefixMatch++;
    else break;
  }
  // Gate 2: per-step top-1 agreement + logit drift.
  let top1Agree = 0;
  let maxDrift = 0;
  const vocab = base.lastLogits[0].length;
  for (let s = 0; s < NUM_NEW; s++) {
    const a = base.lastLogits[s];
    const b = q.lastLogits[s];
    let aBest = 0;
    let bBest = 0;
    for (let v = 1; v < vocab; v++) {
      if (a[v] > a[aBest]) aBest = v;
      if (b[v] > b[bBest]) bBest = v;
      maxDrift = Math.max(maxDrift, Math.abs(a[v] - b[v]));
    }
    if (aBest === bBest) top1Agree++;
  }

  const gpuErrs = getGpuUncapturedErrorCount();
  console.log(
    `prefixMatch=${prefixMatch}/${NUM_NEW} top1Agree=${top1Agree}/${NUM_NEW} ` +
      `maxDrift=${maxDrift.toFixed(3)} gpuErrs=${gpuErrs}`,
  );

  const ok3 = prefixMatch >= MATCH_PREFIX;
  const ok2 = top1Agree === NUM_NEW && maxDrift <= DRIFT_TOL;
  if (ok2 && ok3 && gpuErrs === 0) {
    console.log("QUANT REALWEIGHT GEN PASS");
    process.exit(0);
  }
  console.log(
    `PROBE FAIL: gate2(top1+drift)=${ok2} gate3(prefix≥${MATCH_PREFIX})=${ok3} gpuErrs=${gpuErrs}`,
  );
  process.exit(1);
}

main().catch((e) => {
  console.error("PROBE FAIL:", e);
  process.exit(1);
});
