/**
 * R2 DIAGNOSTIC — dump the witness harvest + releasable summary + top held
 * RESULT pairs on the distil@512 + selective-checkpointing config, so R2 can see
 * exactly which planner RESULT entries are checkpoint-recompute activations (the
 * split targets) and which producer templates the witness signal names as the
 * recompute boundary source. Not a gate; a scouting probe.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { getWitnessProducerKeepSets } from "../src/core/step-tape";
import {
  debugPlannerRegistryStats,
  debugResultBytesByTemplate,
} from "../src/executor/compiled-plan";
import {
  debugTemplateCount,
  debugTemplatePlanMemory,
  getPayloadThrashStats,
} from "../src/executor/executor";
import {
  debugConvergenceState,
  debugReleasableSummary,
  debugTopHeldPairs,
} from "../src/executor/observed-liveness";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/index.ts";

const L = 6,
  H = 12,
  E = 768,
  SEQ = 512,
  STEPS = 14,
  VOCAB = 50257,
  BATCH = 1;
const log = (m: string) => console.error(`[r2-diag] ${m}`);

function randInput(seq: number) {
  const inp = new Int32Array(BATCH * seq);
  const tgt = new Int32Array(BATCH * seq);
  for (let i = 0; i < BATCH * seq; i++) {
    inp[i] = Math.floor(Math.random() * VOCAB);
    tgt[i] = Math.floor(Math.random() * VOCAB);
  }
  return { inp, tgt };
}

async function main() {
  if (!(await initWebGPU())) process.exit(1);
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "10000";
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    {
      vocabSize: VOCAB,
      blockSize: 1024,
      numLayers: L,
      numHeads: H,
      embedDim: E,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );
  await api.beginStep();
  api.endStep();
  await api.markStep();
  model.train(true);
  const opt = new Adam(
    model.parameters(),
    { lr: 5e-4, weightDecay: 0.01, adamW: true },
    api,
  );
  for (let step = 0; step < STEPS; step++) {
    const { inp, tgt } = randInput(SEQ);
    await api.beginStep();
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target, {
        useCheckpoint: true,
        selectiveCheckpoint: true,
      }).loss;
      api.keep(l);
      return l;
    });
    await loss.item();
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    if (step >= STEPS - 5) {
      const th = getPayloadThrashStats();
      log(
        `step ${step}: templates=${debugTemplateCount()} converged=${th.convergedTemplates} pinned=${th.pinnedTemplates} pruned=${th.prunedPairsRemoved}`,
      );
    }
  }

  const reg = debugPlannerRegistryStats();
  log(
    `registry: total=${reg.totalMB}MB result=${reg.resultMB}MB materialized=${reg.materializedMB}MB entries=${reg.entries}`,
  );
  log(`=== RESULT BYTES BY OWNING TEMPLATE ===`);
  const byT = debugResultBytesByTemplate();
  const sorted = Object.entries(byT).sort(
    (a, b) => b[1].resultMB - a[1].resultMB,
  );
  for (const [fp, s] of sorted)
    log(
      `  ${fp}: result=${s.resultMB}MB entries=${s.entries} hasRecompute=${s.hasRecompute}`,
    );
  log(`=== TEMPLATE PLAN MEMORY (sorted by resultMB) ===`);
  const tpm = debugTemplatePlanMemory();
  const tpmSorted = Object.entries(tpm).sort(
    (a, b) => b[1].resultMB - a[1].resultMB,
  );
  for (const [fp, s] of tpmSorted.slice(0, 12))
    log(
      `  ${fp}: nodes=${s.nodes} valid=${s.valid} results=${s.results} pruned=${s.pruned} resultMB=${s.resultMB} tempMB=${s.tempMB}`,
    );
  log(`=== CONVERGENCE STATE ===`);
  for (const [fp, s] of Object.entries(debugConvergenceState()))
    log(
      `  ${fp}: needed=${s.neededSize} conv=${s.converged} pin=${s.pinned} grew=${s.grewThisStep} stable=${s.stableSteps} survived=${s.everSurvived} readback=${s.everReadback} aliased=${s.everAliased} srcC=${s.srcC} srcS=${s.srcS}`,
    );
  const wit = getWitnessProducerKeepSets();
  log(`=== WITNESSED HARVEST (${wit.size} producer templates) ===`);
  for (const [fp, pairs] of wit) {
    log(
      `  producer 0x${fp.toString(16)}: ${pairs.length} pairs: ${pairs.join(" ")}`,
    );
  }
  log(`=== RELEASABLE SUMMARY (per producer template, ALL) ===`);
  const rel = debugReleasableSummary();
  for (const [fp, s] of Object.entries(rel).sort(
    (a, b) => b[1].resultMB - a[1].resultMB,
  )) {
    log(
      `  ${fp}: result=${s.resultMB}MB releasable=${s.releasableMB}MB (${s.releasablePairs} pairs) held=${JSON.stringify(s.heldMB)}`,
    );
  }
  log(`=== TOP HELD RESULT PAIRS (per producer) ===`);
  const top = debugTopHeldPairs(6);
  for (const [fp, rows] of Object.entries(top)) {
    log(`  producer ${fp}:`);
    for (const r of rows)
      log(
        `    pair=${r.pair} op=${r.op} ${r.MB}MB cls=${r.cls} lastReader=${r.lastReader}`,
      );
  }
  destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error(`[r2-diag] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
