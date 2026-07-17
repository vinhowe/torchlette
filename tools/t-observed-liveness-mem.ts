/**
 * Observed cross-plan liveness — MEMORY measurement (stage-1, gate 7).
 *
 * Reproduces the over-harvest delta on a realistic GPT-2 and reports the
 * STEADY-STATE peak GPU memory (measured over late steps, after observation has
 * converged and pruning is active). MODE selects:
 *   - "conservative": build-from-IR with observation DISABLED (the +34%
 *     over-harvest — every action output becomes an exclusive planner result).
 *   - "pruned":       build-from-IR with observed pruning (the fix).
 *
 * The pruned peak must collapse the conservative delta. process.exit(0) at end
 * (Dawn holds threads).
 * (The "cutover" recorded-reference mode retired with the recorded build —
 * task #43 sunset. The lowered path (TORCHLETTE_COMPILED_PLAN=0) is the
 * surviving non-compiled reference.)
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam } from "../src/optim/index.ts";
import {
  debugTemplatePlanMemory,
  getPayloadThrashStats,
} from "../src/executor/executor";
import { debugPlannerRegistryStats } from "../src/executor/compiled-plan";
import {
  debugAllNeededSets,
  setObservedLivenessEnabled,
} from "../src/executor/observed-liveness";

const MODE = process.env.MODE ?? "pruned";
const STEPS = parseInt(process.env.STEPS ?? "20", 10);
const RESET_AT = parseInt(process.env.RESET_AT ?? "12", 10);
// Scale knobs (defaults = the stage-1 small config). The 124M-class sweep is
// VOCAB=50257 LAYERS=12 EMBED=768 HEADS=12 BATCH=1 (the 4.4 scaling-table row).
const VOCAB = parseInt(process.env.VOCAB ?? "512", 10);
const SEQ = parseInt(process.env.SEQ ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "4", 10);
const LAYERS = parseInt(process.env.LAYERS ?? "6", 10);
const EMBED = parseInt(process.env.EMBED ?? "384", 10);
const HEADS = parseInt(process.env.HEADS ?? "6", 10);

function fillFor(name: string, n: number): number[] {
  const lower = name.toLowerCase();
  const isNorm = lower.includes("ln") || lower.includes("norm");
  const isBias = lower.endsWith("bias") || lower.endsWith(".b");
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    if (isNorm && !isBias) out[i] = 1.0;
    else if (isBias) out[i] = 0.0;
    else out[i] = 0.02 * Math.sin(i * 0.1 + name.length);
  }
  return out;
}

async function main() {
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "12000";
  // Mode wiring (no new env flags): build-from-IR is unconditional now
  // (task #43 recorded-build sunset). "conservative" opts observation OFF (the
  // pre-pruning over-harvest baseline); "pruned" enables observed pruning (the
  // fix). Any non-"conservative" MODE runs pruned.
  setObservedLivenessEnabled(MODE !== "conservative");

  if (!(await initWebGPU())) {
    console.error("no webgpu");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    {
      vocabSize: VOCAB,
      blockSize: SEQ + 8,
      numLayers: LAYERS,
      numHeads: HEADS,
      embedDim: EMBED,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );
  const named = new Map<string, Tensor>(model.namedParameters());
  await api.beginStep();
  for (const [name, param] of named) {
    const shape = (param as unknown as { shape: number[] }).shape;
    const n = shape.reduce((a, b) => a * b, 1);
    api.copy_(
      param,
      api.tensorFromArray(fillFor(name, n), shape, { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();

  const params = [...named.values()];
  const opt = new Adam(params, { lr: 5e-4, weightDecay: 0.01 }, api);
  const tok = (i: number) => Math.abs(Math.floor(Math.sin(i * 1.7) * 1e4)) % VOCAB;

  let steadyPeak = 0;
  for (let step = 0; step < STEPS; step++) {
    await api.beginStep();
    const inp = Array.from({ length: BATCH * SEQ }, (_, i) => tok(step * 1000 + i));
    const tgt = Array.from({ length: BATCH * SEQ }, (_, i) => tok(step * 1000 + i + 1));
    const input = api.tensorFromArray(inp, [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(tgt, [BATCH, SEQ], { device: "webgpu" });
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target).loss;
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
    // Reset the peak once observation has converged + rebuilt pruned, so the
    // reported peak is the STEADY-STATE working set (not warmup transients).
    if (step === RESET_AT) {
      // @ts-expect-error test-only reset of the peak watermark
      gpuMemoryTracker.peakUsageBytes =
        gpuMemoryTracker.getCurrentAllocatedBytes();
    }
    if (step > RESET_AT) {
      steadyPeak = Math.max(steadyPeak, gpuMemoryTracker.getPeakUsageBytes());
    }
  }

  const stats = getPayloadThrashStats();
  console.log(
    JSON.stringify({
      mode: MODE,
      steadyPeakMB: +(steadyPeak / 1e6).toFixed(1),
      convergedTemplates: stats.convergedTemplates,
      prunedPairsRemoved: stats.prunedPairsRemoved,
      retiredTemplates: stats.retiredTemplates,
      dirtyMisses: stats.dirtyMisses,
      cleanMisses: stats.cleanMisses,
    }),
  );
  // Attribution dump (DUMP=1): per-template plan memory + registry totals +
  // observed needed-set sources — the stage-2 increment-1 residual analysis.
  if (process.env.DUMP === "1") {
    console.log("REGISTRY " + JSON.stringify(debugPlannerRegistryStats()));
    const mem = debugTemplatePlanMemory();
    const obs = debugAllNeededSets();
    for (const [fp, m] of Object.entries(mem)) {
      const o = obs[fp];
      console.log(
        `TEMPLATE ${fp} nodes=${m.nodes} valid=${m.valid} results=${m.results} pruned=${m.pruned} resultMB=${m.resultMB} tempMB=${m.tempMB}` +
          (o
            ? ` converged=${o.converged} pinned=${o.pinned} needed=${o.needed.length} src=${JSON.stringify(o.srcCounts)}`
            : ""),
      );
    }
    for (const [fp, o] of Object.entries(obs)) {
      if (!mem[fp]) {
        console.log(
          `OBS-ONLY ${fp} converged=${o.converged} pinned=${o.pinned} needed=${o.needed.length} src=${JSON.stringify(o.srcCounts)}`,
        );
      }
    }
  }
  try {
    destroyWebGPU();
  } catch {
    /* ignore */
  }
  process.exit(0);
}
main();
