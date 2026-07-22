/**
 * P4b-R Phase R2 — the WEIGHTED decode-α gate (real Qwen3-1.7B, not random-init).
 *
 * The P4b-R census was random-init (a static-graph proxy). This is the honest
 * gate the re-audit named as the deletion precondition: real weights, block decode
 * (greedy + filtered sampled) vs the host-loop reference, STEP_TAPE 0 vs 1. The
 * deletion proceeds iff:
 *   - the block arms are BYTE-IDENTICAL between STEP_TAPE=1 and =0 (tape adds
 *     nothing to the default decode path), with producers=0 at tape-off, and
 *   - the generations are COHERENT (a real prompt yields human-readable output),
 *   - the tape shows no decode WIN (host residue not faster).
 *
 * Model: Qwen/Qwen3-1.7B, loaded from ckpts/qwen3-1.7b (HF snapshot). f32 weights.
 * Prompt: [785,6722,315,9625,374] = "The capital of France is" (greedy -> "Paris").
 *
 * Run one arm per process (STEP_TAPE is a module-load const):
 *   eval "$(tools/pick-gpu.sh)"; export LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH
 *   ARM=block-greedy TORCHLETTE_STEP_TAPE=0 npx tsx tools/t-p4b-r2-weighted.ts
 */
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
} from "../src/backend/webgpu";
import {
  getSubmitCount,
  resetSubmitCount,
} from "../src/backend/webgpu/webgpu-state";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { STEP_TAPE_RECORD } from "../src/core/step-tape";
import { crossPlanEdgeStats } from "../src/core/cross-plan-edges";
import { getObservedLivenessStats } from "../src/executor/observed-liveness";
import { loadPretrainedQwen3 } from "../examples/qwen3/loader";
import { decodeBlock } from "../packages/qwen3-browser/src/generate";
import type { Qwen3 } from "../packages/qwen3-browser/src/model";
import type { StaticKV } from "../packages/qwen3-browser/src/model";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../ckpts/qwen3-1.7b");
const ARM = process.env.ARM ?? "block-greedy";
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"
const N = 24;

async function prefillFirst(
  api: Torchlette,
  model: Qwen3,
  kv: StaticKV,
): Promise<number> {
  const V = model.config.vocabSize;
  const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
  const logits = api.noGrad(() => model.forward(idx, { staticKV: kv }).logits);
  const S = logits.shape[1];
  const row = api.noGrad(() =>
    api.contiguous(api.narrow(api.narrow(logits, 1, S - 1, 1), 2, 0, V)),
  );
  const data = new Float32Array(await api.cpu(row));
  let best = 0;
  for (let v = 1; v < V; v++) if (data[v] > data[best]) best = v;
  return best;
}

async function blockArm(
  api: Torchlette,
  model: Qwen3,
  filtered: boolean,
): Promise<number[]> {
  const kv = model.allocStaticKV(256);
  const first = await prefillFirst(api, model, kv);
  await api.markStep();
  const ids: number[] = [];
  let last = first;
  const K = 8;
  while (ids.length < N) {
    const sample = filtered
      ? { temperature: 0.7, seed: 1234, topK: 20, topP: 0.95 }
      : undefined;
    const { ids: blk } = await decodeBlock(api, model, kv, last, K, { sample });
    await api.markStep();
    for (const id of blk) {
      if (ids.length >= N) break;
      ids.push(id);
    }
    last = blk[blk.length - 1];
  }
  return ids;
}

/** K=1 per-token host loop through api.capture — the tape-replay path (control). */
async function hostArm(api: Torchlette, model: Qwen3): Promise<number[]> {
  const kv = model.allocStaticKV(256);
  const V = model.config.vocabSize;
  const decode = api.capture(
    (idx: Tensor) =>
      api.noGrad(() => model.forward(idx, { staticKV: kv }).logits),
    { key: () => `kv:bkt${Math.ceil((kv.len + 1) / 128) * 128}` },
  );
  const first = await prefillFirst(api, model, kv);
  await api.markStep();
  const ids: number[] = [];
  let nextTok = first;
  while (ids.length < N) {
    ids.push(nextTok);
    const logits = (await decode(
      api.tensorFromArray([nextTok], [1, 1]),
    )) as Tensor;
    const row = api.noGrad(() =>
      api.contiguous(api.narrow(api.narrow(logits, 1, 0, 1), 2, 0, V)),
    );
    const data = new Float32Array(await api.cpu(row));
    let best = 0;
    for (let v = 1; v < V; v++) if (data[v] > data[best]) best = v;
    nextTok = best;
    await api.markStep();
  }
  return ids;
}

async function main() {
  if (!(await initWebGPU())) {
    console.error(getWebGPUInitError() || "WebGPU init failed");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(1234);
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 256,
    weightDtype: "f32",
  });
  api.setStepScopedCleanup(true);
  await api.markStep();

  resetSubmitCount();
  const t0 = Date.now();
  let ids: number[];
  if (ARM === "host-loop") ids = await hostArm(api, model);
  else ids = await blockArm(api, model, ARM === "block-filtered");
  const wallMs = Date.now() - t0;
  const submits = getSubmitCount();
  const edges = crossPlanEdgeStats();
  const obs = getObservedLivenessStats();
  const uncaptured = getGpuUncapturedErrorCount();

  console.log(
    JSON.stringify(
      {
        arm: ARM,
        STEP_TAPE_RECORD,
        promptIds: PROMPT,
        firstAndDecoded: ids,
        submits,
        wallMs,
        producers: edges.producers,
        convergedTemplates: obs.convergedTemplates,
        uncapturedGpuErrors: uncaptured,
      },
      null,
      2,
    ),
  );
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
