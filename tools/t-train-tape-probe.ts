/**
 * Step-tape phase-2b G-cover gate (docs/staged-execution-phase2b.md,
 * falsification duty + INCREMENT 1): the RECORDER must form an ELIGIBLE tape
 * over a real compiled-checkpointed training step — minimal implied-boundary
 * loop, batch VARYING every step, autocast + checkpoint + GradScaler + clip +
 * AdamW (the fullstack inner step on pretrained distilgpt2).
 *
 * PASS: eligiblePairs > 0 AND refusals == 0 (exit 0). Anything else exits 1
 * with the recorder diagnostics (the refusal list names the uncovered class).
 *
 * The two coverage rules this gate exercises (both landed with 2b inc-1):
 *  - dead-payload/external: shared upload nodes (batch x/y) appearing with
 *    pre-existing results in later plans of the multi-plan step;
 *  - batch-representative: per-param adamStep configs (bias-corrected
 *    step_size) carried by the batch representative's TAG_UNIFORM repack,
 *    with the member↔representative payload agreement assert.
 *
 * Run (solo GPU):
 *   TORCHLETTE_STEP_TAPE=record npx tsx tools/t-train-tape-probe.ts
 * Env: MODEL(=distilgpt2) SEQ_LEN(=512) BATCH(=1) STEPS(=18)
 */
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { STEP_TAPE_RECORD, stStats } from "../src/core/step-tape";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "18", 10);
const log = (m: string) => console.error(`[t-train-tape] ${m}`);

async function main() {
  if (!STEP_TAPE_RECORD) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=record (flag is read at module load)");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
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
  const opt = new Adam(params, { lr: 1e-4, weightDecay: 0.01, adamW: true }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  // Deterministic varying batch (LCG) — the input upload MUST vary per step so
  // the gate exercises the upload-coverage rule, not a constant-payload pass.
  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  let s = 12345;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };

  for (let step = 0; step < STEPS; step++) {
    await scaler.resolveDeferred();
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp[i] = rnd() % V;
      tgt[i] = rnd() % V;
    }
    // MINIMAL loop: no beginStep/endStep/markStep — the implied boundary
    // (opt.step queues, commits at next backward) is the captured regime.
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], { device: "webgpu" });
    const loss = api.tidy(() => {
      const l = api.autocast(
        () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    const scaled = scaler.scale(loss);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    if (step % 4 === 0 || step === STEPS - 1) {
      log(`step ${step}: loss=${lossVal.toFixed(4)}`);
    }
  }

  const st = stStats();
  console.log("=== TRAIN-TAPE-PROBE-STATS ===");
  console.log(
    JSON.stringify(
      {
        stepsObserved: st.stepsObserved,
        eligiblePairs: st.eligiblePairs,
        refusals: st.refusals,
        structureMisses: st.structureMisses,
        loweredPairs: st.loweredPairs,
        boundaryResets: st.boundaryResets,
        tapeCount: st.tapeCount,
        refusalDiagnostics: st.refusalDiagnostics.slice(0, 8),
      },
      null,
      2,
    ),
  );
  const pass = st.eligiblePairs > 0 && st.refusals === 0;
  console.log(pass ? "PASS: training step records as an eligible tape" : "FAIL");
  await destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
