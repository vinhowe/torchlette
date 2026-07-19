/**
 * Step-object PHASE 0 GATE (docs/step-object-design.md §6 Phase 0) — the
 * optimizer-config slot-coverage census over ONE cell of the config matrix the
 * design mandates. The minimal implied-boundary training loop must record an
 * ELIGIBLE tape with ZERO refusals for EVERY cell of —
 *
 *   { fused Adam, foreach Adam } × { no LR schedule, CosineAnnealingLR }
 *
 * each WITH GradScaler + gradient clipping + autocast (the canonical fullstack
 * inner step on pretrained distilgpt2), batch varying every step.
 *
 * [D3 refusal reconciliation — 2026-07-19] Eager SELECTIVE CHECKPOINTING was
 * REMOVED from this loop. The D3 CHECKPOINT_EAGER_REFUSAL (executor.ts, a typed
 * SUNSET-BOUND decline) now keeps EVERY plan built during a checkpointed EAGER
 * (non-whole-step) step LOWERED by design — the b66ead78 checkpoint+arena hazard
 * that global CE-from-IR coverage unmasks. A lowered step forms NO compiled tape
 * (stFinalizeStep: `rec.plans.every(pl => pl.compiled)` is false → loweredPairs),
 * so a checkpointed-eager cell can never satisfy this census's `eligiblePairs>0`
 * assertion — the two are mutually exclusive under the sanctioned refusal (proven:
 * with the refusal the checkpointed loop yields loweredPairs=15 / eligiblePairs=0;
 * disabling it re-introduces the mother-gate corruption, eager[29] 3.0129→3.0130).
 * The optimizer-scalar coverage this census enforces (the frozen-scalar family,
 * LR-via-LiveScalar) is CHECKPOINT-INDEPENDENT, so dropping eager checkpointing
 * loses no census coverage. Checkpoint compile coverage lives where it is honest:
 * the refusal STAYS-LOWERED property in test/whole-step-checkpoint-refusal.spec.ts,
 * and the checkpointed-COMPILES property under whole-step remat in t-whole-step-diff
 * / t-d3-remat. SUNSET: when whole-step training defaults (P4) and the refusal +
 * eager two-plan path are deleted, selective checkpointing returns to this loop.
 *
 * This is the census enforcement the phase2b falsification proved
 * unmet at 556 refusals; inc-1 (batch-representative + dead-payload TAG_UNIFORM
 * coverage) and inc-2a (optimizer-scalars-as-data — LR via the LiveScalar slot)
 * landed the coverage. This gate CONFIRMS full coverage — no cell may silently
 * regress the frozen-scalar family's 7th instance (bias-corrected adamStep
 * step_size), and adding the LR-schedule + foreach cells closes the matrix the
 * single-cell t-train-tape-probe left open.
 *
 * ONE cell per process (WebGPU/Dawn does not survive teardown+re-init cleanly
 * inside one process); the shell driver runs all four. Cell selected by env:
 *   FUSED=1|0  SCHED=1|0
 *
 * PASS: refusals == 0 AND eligiblePairs > 0 (a tape forms). Exit 0/1.
 *
 * Run all four cells (solo GPU):
 *   for F in 1 0; do for S in 1 0; do \
 *     VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *     TORCHLETTE_STEP_TAPE=record FUSED=$F SCHED=$S \
 *     npx tsx tools/t-train-tape-matrix.ts || exit 1; done; done
 * Env: MODEL(=distilgpt2) SEQ_LEN(=256) BATCH(=1) STEPS(=18)
 */
import * as path from "node:path";

const FUSED = (process.env.FUSED ?? "1") === "1";
const SCHED = (process.env.SCHED ?? "0") === "1";
// adam.ts reads ENV.TORCHLETTE_FUSED_ADAM at step() time (ENV is a live view of
// process.env); set BEFORE the frontend imports run so the split is honored.
process.env.TORCHLETTE_FUSED_ADAM = FUSED ? "1" : "0";

const { destroyWebGPU, initWebGPU } = await import("../src/backend/webgpu");
const { Torchlette } = await import("../src/frontend/torchlette");
const { Adam, CosineAnnealingLR, GradScaler } = await import(
  "../src/optim/index.ts"
);
const { clipGradNorm_ } = await import("../src/nn/index.ts");
const { STEP_TAPE_RECORD, stStats } = await import("../src/core/step-tape");
const { loadPretrainedGPT2 } = await import("../examples/gpt2/loader");

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH ?? "1", 10);
const STEPS = parseInt(process.env.STEPS ?? "18", 10);
const CELL = `${FUSED ? "fused" : "foreach"}, ${SCHED ? "cosine-lr" : "no-sched"}`;
const log = (m: string) => console.error(`[tape-matrix] ${m}`);

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
  const opt = new Adam(
    params,
    { lr: 1e-4, weightDecay: 0.01, adamW: true },
    api,
  );
  const sched = SCHED ? new CosineAnnealingLR(opt, STEPS, 1e-5) : null;
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

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
    const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = api.tidy(() => {
      const l = api.autocast(
        // useCheckpoint:false — the D3 CHECKPOINT_EAGER_REFUSAL keeps
        // checkpointed-eager plans lowered, which cannot form a compiled tape
        // (see the header's D3 refusal reconciliation note).
        () =>
          model.forwardWithLoss(input, target, { useCheckpoint: false }).loss,
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
    sched?.step();
    scaled.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    if (step % 4 === 0 || step === STEPS - 1) {
      log(`[${CELL}] step ${step}: loss=${lossVal.toFixed(4)}`);
    }
  }

  const st = stStats();
  const pass = st.refusals === 0 && st.eligiblePairs > 0;
  console.log("=== TAPE-MATRIX-CELL ===");
  console.log(
    JSON.stringify(
      {
        cell: CELL,
        stepsObserved: st.stepsObserved,
        eligiblePairs: st.eligiblePairs,
        refusals: st.refusals,
        structureMisses: st.structureMisses,
        loweredPairs: st.loweredPairs,
        tapeCount: st.tapeCount,
        refusalDiagnostics: st.refusalDiagnostics.slice(0, 8),
      },
      null,
      2,
    ),
  );
  console.log(
    pass ? `PASS [${CELL}]: eligible tape, zero refusals` : `FAIL [${CELL}]`,
  );
  await destroyWebGPU();
  process.exit(pass ? 0 : 1);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
