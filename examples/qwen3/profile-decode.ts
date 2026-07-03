/**
 * Decode-step profiler: dispatch-level truth for Qwen3 inference (task #60).
 * Loads the model on Node/Dawn, runs a prefill + N greedy decode steps, and
 * prints the profiler summary separately for prefill and for late decode
 * steps (past warmup), plus wall-time per step.
 *
 * Run (repo root, GPU otherwise quiet):
 *   npx tsx examples/qwen3/profile-decode.ts [numSteps=12] [dtype=f32|f16]
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  enableProfiling,
  getWebGPUDevice,
  getWebGPUInitError,
  initGpuTimestamps,
  initWebGPU,
  printProfileSummary,
  readGpuTimestamps,
  resetProfileStats,
  setProfilePhase,
  setTimestampsEnabled,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import type { KVCache, StaticKV } from "./model";
import { loadPretrainedQwen3 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"

async function main() {
  const numSteps = Number(process.argv[2] ?? 12);
  const dtype = (process.argv[3] === "f16" ? "f16" : "f32") as "f32" | "f16";
  const mode = process.argv[4] === "cat" ? "cat" : "static";
  console.log(`mode=${mode}`);

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, { maxSeqLen: 512, weightDtype: dtype });
  const vocab = model.config.vocabSize;

  const argmaxLast = async (logits: import("../../src/frontend/torchlette").Tensor, pos: number) => {
    const flat = new Float32Array(await logits.cpu());
    const off = pos * vocab;
    let best = 0;
    for (let v = 1; v < vocab; v++) if (flat[off + v] > flat[off + best]) best = v;
    logits.dispose();
    return best;
  };

  enableProfiling();
  initGpuTimestamps(getWebGPUDevice()!.device as never);

  // --- Prefill
  resetProfileStats();
  setProfilePhase("prefill");
  const tokens = [...PROMPT];
  let kv: KVCache[] = [];
  const staticKV: StaticKV | undefined = mode === "static" ? model.allocStaticKV(512) : undefined;
  {
    const t0 = performance.now();
    const idx = api.tensorFromArray(tokens, [1, tokens.length]);
    const { logits, presentKVs } = api.noGrad(() => model.forward(idx, { staticKV }));
    kv = presentKVs;
    tokens.push(await argmaxLast(logits, tokens.length - 1));
    await api.markStep();
    console.log(`\nprefill wall: ${(performance.now() - t0).toFixed(1)}ms (seq=${PROMPT.length}, dtype=${dtype})`);
  }
  printProfileSummary("prefill");

  // --- Decode: warmup steps (pipeline compilation, pool settling), then
  // measured steps profiled in isolation.
  const WARMUP = Math.min(4, numSteps - 1);
  const TIMESTAMP_STEP = numSteps - 2;
  const walls: number[] = [];
  for (let i = 0; i < numSteps; i++) {
    if (i === WARMUP) {
      resetProfileStats();
      setProfilePhase("decode");
    }
    setTimestampsEnabled(i === TIMESTAMP_STEP);
    const t0 = performance.now();
    // Static mode: step-scoped cleanup (snapshot at beginStep → releaseStepTemps
    // at markStep). Without it, inference loops leak ~1 storage handle per graph
    // node per step (nothing releases the temps' refs until V8 GC gets around to
    // the wrappers) and the markStep sweep grows unboundedly (~15-20ms/step by
    // step 10). Cat mode must NOT use this: its presentKVs are graph-derived
    // tensors held across steps, which step-scoped cleanup would reclaim.
    if (staticKV) await api.beginStep();
    const last = tokens[tokens.length - 1];
    const posOffset = tokens.length - 1;
    const idx = api.tensorFromArray([last], [1, 1]);
    const { logits, presentKVs } = api.noGrad(() =>
      staticKV ? model.forward(idx, { staticKV }) : model.forward(idx, { pastKVs: kv, posOffset }),
    );
    kv = presentKVs;
    // GPU top-K prefilter readback (512B instead of the 600KB vocab row);
    // indices[0] is bit-identical to the full-logits argmax (gated by
    // examples/qwen3/topk-equivalence.ts).
    const top = await api.readTopK(logits, 64, { length: vocab });
    logits.dispose();
    tokens.push(top.indices[0]);
    if (staticKV) api.endStep();
    await api.markStep();
    const dt = performance.now() - t0;
    if (i >= WARMUP) walls.push(dt);
    if (i === TIMESTAMP_STEP) await readGpuTimestamps();
    console.log(`decode step ${i}${i < WARMUP ? " (warmup)" : ""}: ${dt.toFixed(1)}ms`);
  }
  printProfileSummary(`decode steps ${WARMUP}..${numSteps - 1}`);
  const avg = walls.reduce((a, b) => a + b, 0) / walls.length;
  console.log(
    `\ndecode avg (post-warmup): ${avg.toFixed(1)}ms/token = ${(1000 / avg).toFixed(1)} tok/s (per-step profile above is summed over ${walls.length} steps)`,
  );
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
