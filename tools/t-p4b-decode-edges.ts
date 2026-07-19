/**
 * P4b deletion-blocker probe: is the WITNESS apparatus + cross-plan-edges store
 * LIVE on the DECODE path?
 *
 * The P4b brief asks whether the step-tape witness apparatus / cross-plan-edges
 * / observed-liveness pruning can be deleted because "their only consumers were
 * training-shaped." Report analysis found the store is populated only under
 * STEP_TAPE_RECORD (TORCHLETTE_STEP_TAPE=1/record), which the SHIPPED decode
 * apps turn on (examples/qwen3-steering/src/lib/tape-flag.ts). If a real
 * decode-shaped captured plan with a cross-plan KV read populates
 * crossPlanEdgeStats().producers > 0, the witness/cross-plan-edges machinery is
 * DECODE-LIVE and cannot be deleted (a deletion would empty the store on decode,
 * removing the overlay-release UAF guard at executor.ts:2290 and the harvest
 * keep at observed-liveness.ts:755).
 *
 * This reuses the model-free static-KV decode seam (persistent KV updated in
 * place, read through a narrow view — the Gemma-2 static-KV decode shape) under
 * STEP_TAPE=1 and prints, after warm decode:
 *   - crossPlanEdgeStats()   — producers/edges witnessed on decode
 *   - observed-liveness releasable stats — did the stage-3 release fire?
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *      TORCHLETTE_STEP_TAPE=1 npx tsx tools/t-p4b-decode-edges.ts
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { STEP_TAPE_RECORD } from "../src/core/step-tape";
import { crossPlanEdgeStats } from "../src/core/cross-plan-edges";
import { getObservedLivenessStats } from "../src/executor/observed-liveness";

async function main() {
  if (!STEP_TAPE_RECORD) {
    console.error("[p4b-edges] FAIL: set TORCHLETTE_STEP_TAPE=1 (module-load)");
    process.exit(1);
  }
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const H = 4;
  const S = 8;
  const D = 4;
  const NSTEPS = 20; // enough to converge (K=3) and warm the harvest

  const idxFor = (pos: number): Tensor => {
    const arr = new Float32Array(H * 1 * D).fill(pos);
    return api.tensorFromArray(arr, [1, H, 1, D]);
  };
  // Decode body: in-place KV update then a cross-plan view read of the
  // accumulated KV — structurally identical each step so the captured plan goes
  // hot and the harvest activates (the exact static-KV decode seam).
  const body = (kv: Tensor, x: Tensor): Tensor => {
    kv.copy_(kv.scatterAdd(idxFor(0), x, { dim: 2 }));
    return api.narrow(kv, 2, 0, S).sum([2]);
  };

  const kv = api.registerState(api.zeros([1, H, S, D]));
  const decode = api.capture((x: Tensor) => body(kv, x));
  api.setStepScopedCleanup(true);
  await api.markStep();
  for (let t = 0; t < NSTEPS; t++) {
    const x = api.tensorFromArray(new Array(H * D).fill(t + 1), [1, H, 1, D]);
    const out = (await decode(x)) as Tensor;
    await api.cpu(out);
    await api.markStep();
  }

  const edges = crossPlanEdgeStats();
  const obs = getObservedLivenessStats();
  console.log(
    JSON.stringify(
      {
        decodeCaptureHits: decode.stats().hits,
        crossPlanEdges: edges,
        observed: {
          convergedTemplates: obs.convergedTemplates,
          releasablePairs: obs.releasablePairs,
          releasableMB: obs.releasableMB,
          prunedPairsRemoved: obs.prunedPairsRemoved,
        },
      },
      null,
      2,
    ),
  );
  console.log(
    `[p4b-edges] VERDICT: cross-plan-edge producers under decode = ${edges.producers} ` +
      `(>0 ⇒ witness/cross-plan-edges is DECODE-LIVE, deletion BLOCKED)`,
  );
  process.exit(0);
}

main();
