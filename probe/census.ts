/**
 * FORCE-SITE CENSUS + GRAPH-SCALE probe for the step-function-compiler design.
 * Instruments RuntimeEngine.forceAllMerged / forceAllPending to count every
 * mid-step force, tag its cause (phase marker + stack site), and record the
 * pending-lazy-node count going into each force (graph-scale proxy).
 */
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { initWebGPU } from "../src/backend/webgpu";
import { getSubmitCount, resetSubmitCount } from "../src/backend/webgpu/webgpu-state";
import { Torchlette } from "../src/frontend/torchlette";
import { crossEntropy } from "../src/nn";
import { RuntimeEngine } from "../src/runtime/engine";
import { getPendingNodeIds } from "../src/runtime/tensor";
import { Adam, GradScaler } from "../src/optim";

// ---- global census state ----
let PHASE = "init";
interface Ev { phase: string; site: string; pending: number; }
const events: Ev[] = [];
let recording = false;

function topSite(): string {
  const stack = new Error().stack ?? "";
  const lines = stack.split("\n").slice(2);
  for (const ln of lines) {
    const m = ln.match(/(autograd|adam|sgd|torchlette|engine|checkpoint|grad-scaler|torchlette)\.ts:(\d+)/);
    if (m && !ln.includes("forceAllMerged") && !ln.includes("forceAllPending")) {
      return `${m[1]}.ts:${m[2]}`;
    }
  }
  // fallback: first torchlette-src frame
  for (const ln of lines) {
    const m = ln.match(/src\/[^)]*\.ts:(\d+)/);
    if (m) return ln.trim().replace(/^at\s+/, "").slice(0, 60);
  }
  return "?";
}

const proto = RuntimeEngine.prototype as any;
const origMerged = proto.forceAllMerged;
const origPending = proto.forceAllPending;
proto.forceAllMerged = async function (...tensors: any[]) {
  if (recording) events.push({ phase: PHASE, site: topSite(), pending: getPendingNodeIds().size });
  return origMerged.apply(this, tensors);
};
proto.forceAllPending = async function () {
  if (recording) events.push({ phase: PHASE, site: "forceAllPending", pending: getPendingNodeIds().size });
  return origPending.apply(this, arguments as any);
};

function reportConfig(name: string) {
  console.log(`\n===== CONFIG: ${name} =====`);
  // group by (phase, site)
  const key = (e: Ev) => `${e.phase}::${e.site}`;
  const groups = new Map<string, { count: number; maxPending: number; sumPending: number }>();
  for (const e of events) {
    const k = key(e);
    const g = groups.get(k) ?? { count: 0, maxPending: 0, sumPending: 0 };
    g.count += 1; g.maxPending = Math.max(g.maxPending, e.pending); g.sumPending += e.pending;
    groups.set(k, g);
  }
  const steps = STEADY_STEPS;
  console.log(`forces recorded over ${steps} steady steps:`);
  console.log(`  phase::site                          | forces | forces/step | maxPending`);
  for (const [k, g] of [...groups.entries()].sort()) {
    console.log(`  ${k.padEnd(36)} | ${String(g.count).padStart(6)} | ${(g.count/steps).toFixed(2).padStart(11)} | ${String(g.maxPending).padStart(10)}`);
  }
  const total = events.length;
  console.log(`  TOTAL forces/step = ${(total/steps).toFixed(2)}  (${total} over ${steps} steps)`);
}

const STEADY_STEPS = 6;
const WARMUP = 4;
const tokens = [2514, 307, 393, 407, 284, 307, 11, 326, 318, 262, 1808, 13, 2514, 307, 393, 407];

async function runConfig(opts: {
  name: string; checkpoint: boolean; scaler: boolean; useForwardWithLoss: boolean;
  deferLoss?: boolean; wholeStep?: boolean;
}) {
  events.length = 0;
  const api = new Torchlette("webgpu", {
    enableFusion: true, enableMemoryPlanning: true, enableCheckpointSegmentation: true,
  });
  const model = await loadPretrainedGPT2(api, "./models/distilgpt2", { dropoutRate: 0 }, { device: "webgpu" });
  model.train();
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const scaler = opts.scaler ? new GradScaler(api, { initScale: 1024.0 }) : null;
  const input = api.tensorFromArray(tokens.slice(0, -1), [1, tokens.length - 1]);
  const target = api.tensorFromArray(tokens.slice(1), [1, tokens.length - 1]);

  let submitTotal = 0;
  for (let step = 0; step < WARMUP + STEADY_STEPS; step++) {
    recording = step >= WARMUP;
    if (recording) resetSubmitCount();
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    optimizer.zeroGrad();
    // [whole-step trace, P1] Enter the trace scope so backward DEFERS its
    // grad-write force into the boundary force — forward+backward+optimizer
    // become ONE forced plan (loss must be deferred too, which wholeStep
    // configs imply). Requires TORCHLETTE_WHOLE_STEP=1 to actually defer.
    if (opts.wholeStep) api._enterWholeStep();

    PHASE = "forward+item";
    let loss;
    if (opts.useForwardWithLoss) {
      const r = model.forwardWithLoss(input, target, { useCheckpoint: opts.checkpoint });
      loss = r.loss!;
    } else {
      const logits = model.forward(input);
      const [B, S, V] = logits.shape;
      loss = crossEntropy(logits.reshape([B * S, V]), target.reshape([B * S]));
    }
    let lossVal = -1;
    if (!opts.deferLoss && !opts.wholeStep) lossVal = await loss.item();

    PHASE = "backward";
    if (scaler) {
      const scaled = scaler.scale(loss);
      await scaled.backward();
    } else {
      await loss.backward();
    }

    PHASE = "optimizer";
    if (scaler) {
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
    } else {
      optimizer.step();
    }

    if (opts.wholeStep) api._exitWholeStep();
    PHASE = "markStep";
    api.endStep();
    await api.markStep();

    if (recording) {
      submitTotal += getSubmitCount();
      if (step === WARMUP) console.log(`[${opts.name}] step ${step} loss=${lossVal.toFixed(4)} submits=${getSubmitCount()}`);
    }
    recording = false;
  }
  reportConfig(opts.name);
  console.log(`  submits/step (physical GPU flush) = ${(submitTotal / STEADY_STEPS).toFixed(1)}`);
  // graph-scale: max pending seen at the backward-grads force = fwd+bwd node count
  const bwd = events.filter((e) => e.phase === "backward");
  const opt = events.filter((e) => e.phase === "optimizer");
  const maxBwd = Math.max(0, ...bwd.map((e) => e.pending));
  const maxOpt = Math.max(0, ...opt.map((e) => e.pending));
  console.log(`  GRAPH-SCALE: max pending @backward-force = ${maxBwd} (fwd+bwd graph), @optimizer-force = ${maxOpt}`);
  console.log(`  => whole-step accumulated node estimate ~= ${maxBwd + maxOpt} (fwd+bwd + optimizer)`);
}

async function main() {
  await initWebGPU();
  await runConfig({ name: "minimal (no ckpt, no scaler)", checkpoint: false, scaler: false, useForwardWithLoss: true });
  await runConfig({ name: "checkpoint (no scaler)", checkpoint: true, scaler: false, useForwardWithLoss: true });
  await runConfig({ name: "scaler+checkpoint (AMP inf-skip)", checkpoint: true, scaler: true, useForwardWithLoss: true });
  await runConfig({ name: "DEFERRED-LOSS (no mid-step item, no ckpt)", checkpoint: false, scaler: false, useForwardWithLoss: true, deferLoss: true });
  // [P1] Whole-step trace: DEFERRED-LOSS + backward grad-write force deferred
  // into the boundary → forward+backward+optimizer is ONE forced plan. Needs
  // TORCHLETTE_WHOLE_STEP=1 (else this equals DEFERRED-LOSS). Minimal (no ckpt,
  // no scaler): the P1 target of 1-2 mid-step forces.
  await runConfig({ name: "WHOLE-STEP-TRACE (P1: no ckpt, no scaler)", checkpoint: false, scaler: false, useForwardWithLoss: true, wholeStep: true });
  await runConfig({ name: "WHOLE-STEP-TRACE +scaler", checkpoint: false, scaler: true, useForwardWithLoss: true, wholeStep: true });
  await runConfig({ name: "WHOLE-STEP-TRACE +checkpoint (P3 remat: recompute lazy, structural force subsumed)", checkpoint: true, scaler: false, useForwardWithLoss: true, wholeStep: true });
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
