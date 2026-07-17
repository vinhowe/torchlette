/**
 * D5 STAGE 1 — the declared-lifetime dividend's re-open-condition measurement
 * (`docs/staged-execution-phase2b.md §5`, `docs/step-object-design.md` phase 7).
 *
 * The re-open condition, honored not assumed: before RETIRING the three
 * observation predicates (`everSurvived` / `everReadback` / `everAliased`) on
 * the captured path, MEASURE what they cost — and, more important, what they
 * PRUNE that the derivation (crossPlanEdges + graphHeldAt + declared survivors)
 * wouldn't. If the predicates prune NOTHING the derived guards don't already
 * decline, the dividend is free; if they prune something real, characterize it.
 *
 * This runs a WARM CAPTURED training step (`api.capture(..., {training:true})`
 * + step-tape) so `crossPlanEdges` is witnessed (the derived substrate), with
 * `TORCHLETTE_MEASURE_D5=1` arming the claim-seam probe (executor.ts). It
 * reports, per config:
 *   - claim-seam divergence: candidates / structurally-releasable / predicate-
 *     blocked / REDUNDANT prune (derived guards also decline → free) / REAL
 *     prune (derived guards would release → the honest cost), keyed by predicate
 *   - bookkeeping footprint: the three predicate Set sizes + needed/last-reader
 *   - real-prune samples (producer fp + pair + shape) for characterization
 *
 * Matrix: MODEL(distilgpt2|gpt2) × CKPT(on|off). Run one cell:
 *   VULKAN_DEVICE_INDEX=2 LD_LIBRARY_PATH=tools/vk-shim \
 *   TORCHLETTE_STEP_TAPE=1 TORCHLETTE_MEASURE_D5=1 \
 *   MODEL=distilgpt2 CKPT=1 npx tsx tools/t-d5-watcher-cost.ts
 * process.exit(0) at the end (Dawn holds threads).
 */
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import type { Tensor } from "../src/frontend/tensor";
import {
  getCapturedRecordSkips,
  getD5Cost,
  getObservedLivenessStats,
  getPredicateMemStats,
  resetD5Cost,
} from "../src/executor/observed-liveness";
import { crossPlanEdgeStats } from "../src/core/cross-plan-edges";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "24", 10);
const CKPT = process.env.CKPT !== "0";
const CAPTURE = process.env.CAPTURE !== "0";
const log = (m: string) => console.error(`[d5-cost] ${m}`);

async function main() {
  if (!STEP_TAPE_REPLAY) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=1 (flag read at module load)");
    process.exit(1);
  }
  if (!process.env.TORCHLETTE_MEASURE_D5) {
    log("FAIL: set TORCHLETTE_MEASURE_D5=1");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  resetD5Cost();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = await loadPretrainedGPT2(
    api,
    path.join(ROOT, "models", MODEL),
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  model.train(true);
  const params = model.parameters();
  const opt = new Adam(
    params,
    { lr: 1e-4, weightDecay: 0.01, adamW: true },
    api,
  );
  const sched = new CosineAnnealingLR(opt, STEPS, 1e-5);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  let s = 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };

  const body = async (input: Tensor, target: Tensor): Promise<Tensor> => {
    const loss = api.tidy(() => {
      const l = api.autocast(
        () =>
          model.forwardWithLoss(input, target, { useCheckpoint: CKPT }).loss,
      );
      api.keep(l);
      return l;
    });
    const lossOut = api.noGrad(() => api.mul(loss, 1));
    const scaled = scaler.scale(loss);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    return lossOut;
  };
  const stepAsync = CAPTURE ? api.capture(body, { training: true }) : undefined;

  const losses: number[] = [];
  for (let stp = 0; stp < STEPS; stp++) {
    await scaler.resolveDeferred();
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = (await (stepAsync
      ? stepAsync(input, target)
      : body(input, target))) as Tensor;
    const lv = await loss.item();
    losses.push(lv);
    loss.dispose();
    input.dispose();
    target.dispose();
    sched.step();
  }

  const cost = getD5Cost();
  const mem = getPredicateMemStats();
  const obs = getObservedLivenessStats();
  const edges = crossPlanEdgeStats();
  const replay = api.getStepTapeStats().replay;

  const result = {
    model: MODEL,
    capture: CAPTURE,
    ckpt: CKPT,
    seq: SEQ,
    batch: BATCH,
    steps: STEPS,
    lossFirst: +losses[0].toFixed(4),
    lossLast: +losses[losses.length - 1].toFixed(4),
    replayHits: replay.hits,
    capturedRecordSkips: getCapturedRecordSkips(),
    d5cost: cost,
    predicateMem: mem,
    observedStats: {
      convergedTemplates: obs.convergedTemplates,
      pinnedTemplates: obs.pinnedTemplates,
      prunedPairsRemoved: obs.prunedPairsRemoved,
      releasablePairs: obs.releasablePairs,
      releasableMB: obs.releasableMB,
      cleanMisses: obs.cleanMisses,
      dirtyMisses: obs.dirtyMisses,
      claimMisses: obs.claimMisses,
    },
    crossPlanEdges: edges,
  };
  console.log(`=== D5-COST === ${JSON.stringify(result)}`);
  log(`losses: ${losses.map((l) => l.toFixed(3)).join(",")}`);

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
