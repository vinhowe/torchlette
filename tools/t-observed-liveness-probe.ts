/**
 * Observed cross-plan liveness — set-parity probe (stage-1, task #43).
 *
 * Trains a tiny-but-real GPT-2 through the full production inner step, then
 * dumps, per recurring template (fingerprint hex), the SET of harvested result
 * identities ("nodeIndex:outputIndex") its compiled plan actually keeps.
 *
 * Run twice by test/observed-liveness.spec.ts (stage-2 flip: build-from-IR is
 * the default):
 *   - (unset)                     → build-from-IR with observed pruning
 *   - TORCHLETTE_BUILD_FROM_IR=0  → the recorded-cutover reference
 * The two result maps must agree on every recurring template: the pruned
 * build-from-IR result set == the cutover's live-result survivor set (single
 * source at the seam). Also prints the guard telemetry for diagnostics.
 *
 * process.exit(0) at end (Dawn holds threads).
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam } from "../src/optim/index.ts";
import {
  debugTemplateResultSets,
  getPayloadThrashStats,
} from "../src/executor/executor";
import { debugAllNeededSets } from "../src/executor/observed-liveness";

const STEPS = parseInt(process.env.STEPS ?? "16", 10);
const VOCAB = 256;
const SEQ = 32;
const BATCH = 2;

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
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "4000";
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
      blockSize: 64,
      numLayers: 2,
      numHeads: 2,
      embedDim: 64,
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

  const losses: number[] = [];
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
    const lossVal = await loss.item();
    losses.push(lossVal);
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  const resultSets = debugTemplateResultSets();
  const neededSets = debugAllNeededSets();
  const stats = getPayloadThrashStats();
  console.log(
    JSON.stringify({
      buildFromIR: process.env.TORCHLETTE_BUILD_FROM_IR !== "0",
      losses,
      resultSets,
      neededSets,
      stats,
    }),
  );
  try {
    destroyWebGPU();
  } catch {
    /* ignore */
  }
  process.exit(0);
}
main();
