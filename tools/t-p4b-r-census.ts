/**
 * P4b-R RE-AUDIT probe — is the TAPE subsystem (cross-plan-edges / witness /
 * step-tape) LIVE on TODAY's DEFAULT decode path (the unrolled-K BLOCK)?
 *
 * The P4b STOP measured the OLD default decode (`api.capture` per-token host
 * loop) at `crossPlanEdgeStats().producers=1, convergedTemplates=1` and stamped
 * the tape DECODE-LIVE. Since then the unrolled-K block became the demos' default
 * for greedy + gumbel + top-k/top-p (SAMPLER STATUS 2026-07-21). The block calls
 * `decodeBlock` DIRECTLY — it NEVER goes through `api.capture`. The P5 entry
 * checklist asks exactly this re-audit: run `t-p4b-decode-edges` UNDER unrolled-K
 * and expect `producers=0` and no observed-liveness convergence *from the tape*.
 *
 * This probe drives a real random-init Qwen3 through THREE decode arms and prints
 * the census after warm decode:
 *   ARM=block-greedy   — decodeBlock greedy (the default covered sampler)
 *   ARM=block-filtered — decodeBlock top-k+top-p+temperature (the newly-landed
 *                        on-device sampler; the demos' distribution)
 *   ARM=host-loop      — the K=1 per-token `api.capture` residue (PRESENCE control
 *                        — must reproduce the P4b presence proof producers>=1)
 *
 * Run twice per arm: TORCHLETTE_STEP_TAPE=1 (demo config) and =0. The tape files
 * are deletable-for-decode iff the BLOCK arms show producers=0 (tape not
 * populated) AND are byte-identical between STEP_TAPE=1 and =0 (tape adds nothing
 * to the block). observed-liveness convergence is reported separately: it is the
 * executor harvest (NOT tape-gated), expected live on any compiled plan.
 *
 * Run: eval "$(tools/pick-gpu.sh)"; ARM=block-greedy TORCHLETTE_STEP_TAPE=1 \
 *   LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH npx tsx tools/t-p4b-r-census.ts
 */
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
import type { Qwen3Config, StaticKV } from "../packages/qwen3-browser/src/model";
import { Qwen3 } from "../packages/qwen3-browser/src/model";
import { decodeBlock } from "../packages/qwen3-browser/src/generate";

const CONFIG: Qwen3Config = {
  vocabSize: 256,
  hiddenSize: 64,
  numLayers: 2,
  numHeads: 4,
  numKVHeads: 2,
  headDim: 16,
  intermediateSize: 128,
  ropeTheta: 1e6,
  rmsNormEps: 1e-6,
  maxSeqLen: 256,
};

const ARM = process.env.ARM ?? "block-greedy";
const PROMPT = [3, 14, 15, 92, 65];
const N = 40; // enough to warm/converge templates and cross several blocks

async function prefillFirst(
  api: Torchlette,
  model: Qwen3,
  kv: StaticKV,
): Promise<number> {
  const V = CONFIG.vocabSize;
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
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
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

/** K=1 per-token host loop through api.capture — the P4b presence control. */
async function hostArm(api: Torchlette, model: Qwen3): Promise<number[]> {
  const kv = model.allocStaticKV(CONFIG.maxSeqLen);
  const V = CONFIG.vocabSize;
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
  const model = new Qwen3(api, { ...CONFIG });
  api.setStepScopedCleanup(true);
  await api.markStep();

  resetSubmitCount();
  let ids: number[];
  if (ARM === "host-loop") ids = await hostArm(api, model);
  else ids = await blockArm(api, model, ARM === "block-filtered");
  const submits = getSubmitCount();

  const edges = crossPlanEdgeStats();
  const obs = getObservedLivenessStats();
  const uncaptured = getGpuUncapturedErrorCount();

  console.log(
    JSON.stringify(
      {
        arm: ARM,
        STEP_TAPE_RECORD,
        tokens: ids,
        submits,
        crossPlanEdges: edges,
        observed: {
          convergedTemplates: obs.convergedTemplates,
          pinnedTemplates: obs.pinnedTemplates,
          retiredTemplates: obs.retiredTemplates,
          releasablePairs: obs.releasablePairs,
          releasableMB: obs.releasableMB,
          prunedPairsRemoved: obs.prunedPairsRemoved,
        },
        uncapturedGpuErrors: uncaptured,
      },
      null,
      2,
    ),
  );
  console.log(
    `[p4b-r] arm=${ARM} STEP_TAPE=${STEP_TAPE_RECORD ? 1 : 0} ` +
      `crossPlanEdge.producers=${edges.producers} convergedTemplates=${obs.convergedTemplates}`,
  );
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
