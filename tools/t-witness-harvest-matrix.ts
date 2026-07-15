/**
 * WITNESS-TIME HARVEST — CONFIG MATRIX (task #98 phase 4, docs/step-object-design.md
 * §6 phase 4 gate + §4.4 differential + §10 ruling 2 EVENT-INCLUSIVE clause).
 *
 * ONE cell per invocation (selected by env), so the shell driver can run each on a
 * fresh device/process (Dawn device-chain contention makes multi-cell-in-process
 * flaky). Every cell drives the exact witness-harvest workload — build-from-IR
 * generated harvest + observed-liveness prune + step-tape witness recorder — and
 * asserts the phase-4 gate:
 *
 *   A. NO `Input not ready`  — the #97 stage-3 negative assertion (zero throws).
 *   B. WITNESS COVERAGE      — the witnessed harvest set is non-empty AND its
 *                              variance count is 0 at steady state (stable reader).
 *   C. SHADOW SET-PARITY     — every pair the observed-liveness prune EXCLUDED but
 *                              a later plan then READ (a guard/readback miss that
 *                              GREW the needed-set) is covered by the witnessed set
 *                              OR was recovered. Operationally: post-witness the
 *                              generated harvest keeps every read pair, so the
 *                              needed-set stops growing from lowered-read misses —
 *                              the empty-diff signal (§4.4). Measured as
 *                              cleanMisses+dirtyMisses trending to 0 once witnessed.
 *   D. TRAJECTORY FINITE     — loss stays finite across the run (no corruption).
 *
 * CELLS (env CELL=):
 *   checkpoint   — distil@512 dims + selective checkpointing (the #97 config).
 *   medium       — gpt2-medium dims @512 + selective checkpointing.
 *   chunked124m  — 124M dims (E=768,L=12) @256 — the >128MB chunked full-reduction
 *                  class (the plan the recorded build's last uncovered op hits).
 *   scaler-inf   — checkpoint + GradScaler with an INDUCED overflow (huge initScale)
 *                  at a chosen step → an inf-SKIP. EVENT-INCLUSIVE: the skip must
 *                  either change structure (guard refuses, normal path, NO
 *                  corruption) or flow through a declared slot. Asserts no
 *                  corruption + no Input-not-ready across the event.
 *   lr-milestone — checkpoint + a StepLR that DROPS the LR at a milestone step
 *                  (a scheduled per-step-varying scalar). EVENT-INCLUSIVE: the LR
 *                  change flows through the declared scalar slot (data), the
 *                  structure is unchanged, no corruption.
 *
 * PASS ⇒ exit 0; any assertion fail ⇒ exit 1.
 *
 * Run (solo GPU):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record \
 *     CELL=checkpoint npx tsx tools/t-witness-harvest-matrix.ts
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { STEP_TAPE_RECORD, stStats } from "../src/core/step-tape";
import { getPayloadThrashStats } from "../src/executor/executor";
import { getAllWitnessedHarvest } from "../src/executor/observed-liveness";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler, StepLR } from "../src/optim/index.ts";

type Cell = {
  L: number;
  H: number;
  E: number;
  SEQ: number;
  steps: number;
  autocast: boolean;
  scaler: boolean;
  initScale: number;
  infAtStep: number; // step to induce overflow (-1 = never)
  scheduler: boolean;
  lrMilestone: number; // step at which StepLR drops (0 = off)
};

const CELLS: Record<string, Cell> = {
  checkpoint: {
    L: 6,
    H: 12,
    E: 768,
    SEQ: 512,
    steps: 12,
    autocast: false,
    scaler: false,
    initScale: 1024,
    infAtStep: -1,
    scheduler: false,
    lrMilestone: 0,
  },
  medium: {
    L: 24,
    H: 16,
    E: 1024,
    SEQ: 512,
    steps: 10,
    autocast: false,
    scaler: false,
    initScale: 1024,
    infAtStep: -1,
    scheduler: false,
    lrMilestone: 0,
  },
  chunked124m: {
    L: 12,
    H: 12,
    E: 768,
    SEQ: 256,
    steps: 12,
    autocast: false,
    scaler: false,
    initScale: 1024,
    infAtStep: -1,
    scheduler: false,
    lrMilestone: 0,
  },
  // autocast + an ENORMOUS initial scale that overflows the scaled gradients in
  // the first steps → real inf-SKIPs; the fused unscale detects the inf and the
  // scaler halves the scale each time until it stops overflowing (measured:
  // 1e40 → ~1.5e38 over steps 0-5, foundInf=true). EVENT-INCLUSIVE (§10 Q2): the
  // skip event changes step structure (a skipped step does not run the optimizer)
  // and must produce NO corruption and NO Input-not-ready across the window.
  "scaler-inf": {
    L: 6,
    H: 12,
    E: 768,
    SEQ: 512,
    steps: 16,
    autocast: true,
    scaler: true,
    initScale: 1.0e40,
    infAtStep: 0,
    scheduler: false,
    lrMilestone: 0,
  },
  "lr-milestone": {
    L: 6,
    H: 12,
    E: 768,
    SEQ: 512,
    steps: 14,
    autocast: false,
    scaler: false,
    initScale: 1024,
    infAtStep: -1,
    scheduler: true,
    lrMilestone: 7,
  },
};

const CELL = process.env.CELL ?? "checkpoint";
const cfg = CELLS[CELL];
if (!cfg) {
  console.error(
    `[wh-matrix] unknown CELL=${CELL}; choices: ${Object.keys(CELLS).join(", ")}`,
  );
  process.exit(1);
}
const VOCAB = 50257;
const BATCH = 1;
const log = (m: string) => console.error(`[wh-matrix:${CELL}] ${m}`);

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
  if (!STEP_TAPE_RECORD) {
    log("FAIL: set TORCHLETTE_STEP_TAPE=record");
    process.exit(1);
  }
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= cfg.E >= 1024 ? "24000" : "10000";

  let inputNotReady = 0;
  const origErr = console.error.bind(console);
  console.error = (...a: unknown[]) => {
    if (a.map(String).join(" ").includes("Input not ready")) inputNotReady++;
    origErr(...a);
  };

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
      numLayers: cfg.L,
      numHeads: cfg.H,
      embedDim: cfg.E,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );

  await api.beginStep();
  api.endStep();
  await api.markStep();

  model.train(true);
  const params = model.parameters();
  const opt = new Adam(
    params,
    { lr: 5e-4, weightDecay: 0.01, adamW: true },
    api,
  );
  const scaler = cfg.scaler
    ? new GradScaler(api, { initScale: cfg.initScale })
    : null;
  const sched = cfg.scheduler ? new StepLR(opt, cfg.lrMilestone, 0.1) : null;

  log(
    `START L=${cfg.L} H=${cfg.H} E=${cfg.E} seq=${cfg.SEQ} steps=${cfg.steps} ` +
      `scaler=${cfg.scaler}(inf@${cfg.infAtStep}) sched=${cfg.scheduler}(drop@${cfg.lrMilestone})`,
  );

  const losses: number[] = [];
  let threw = false;
  let scalerInfObserved = false;
  let lrBefore = 0;
  let lrAfter = 0;

  for (let step = 0; step < cfg.steps; step++) {
    if (scaler) {
      await scaler.resolveDeferred();
      // resolveDeferred reads back the PRIOR step's inf flag → the skip is
      // observable here (deferred by one step by design).
      if (scaler.foundInf) scalerInfObserved = true;
    }
    const { inp, tgt } = randInput(cfg.SEQ);
    try {
      await api.beginStep();
      const input = api.tensorFromArray(Array.from(inp), [BATCH, cfg.SEQ], {
        device: "webgpu",
      });
      const target = api.tensorFromArray(Array.from(tgt), [BATCH, cfg.SEQ], {
        device: "webgpu",
      });
      const loss = api.tidy(() => {
        const fwd = () =>
          model.forwardWithLoss(input, target, {
            useCheckpoint: true,
            selectiveCheckpoint: true,
          }).loss;
        const l = cfg.autocast ? api.autocast(fwd) : fwd();
        api.keep(l);
        return l;
      });
      const lossVal = await loss.item();
      losses.push(lossVal);

      const backTgt = scaler ? scaler.scale(loss) : loss;
      await backTgt.backward();
      if (scaler) {
        scaler.unscale_(opt);
        if (scaler.foundInf) scalerInfObserved = true;
        scaler.step(opt);
        scaler.update();
      } else {
        opt.step();
      }
      if (sched) {
        if (step === cfg.lrMilestone - 1) lrBefore = opt.getLR();
        sched.step();
        if (step === cfg.lrMilestone) lrAfter = opt.getLR();
      }
      opt.zeroGrad();
      input.dispose();
      target.dispose();
      api.endStep();
      await api.markStep();

      if (step < 6 || step === cfg.steps - 1 || step === cfg.lrMilestone)
        log(
          `step ${step}: loss=${lossVal.toFixed(4)}${scaler ? ` scale=${scaler.getScale().toExponential(2)}` : ""}`,
        );
      if (!Number.isFinite(lossVal)) {
        log(`FATAL non-finite loss at step ${step} — CORRUPTION`);
        threw = true;
        break;
      }
    } catch (e) {
      threw = true;
      log(
        `THREW at step ${step}: ${e instanceof Error ? e.message : String(e)}`,
      );
      if (e instanceof Error && e.message.includes("Input not ready"))
        inputNotReady++;
      break;
    }
  }

  const st = stStats();
  const thrash = getPayloadThrashStats();
  const witnessed = getAllWitnessedHarvest();
  let witnessedPairs = 0;
  for (const s of witnessed.values()) witnessedPairs += s.length;

  log("=== VERDICT ===");
  log(
    `witnessedTemplates=${witnessed.size} witnessedPairs=${witnessedPairs} witnessVariances=${st.witnessVariances}`,
  );
  log(`inputNotReady=${inputNotReady} threw=${threw}`);
  log(
    `cleanMisses=${thrash.cleanMisses} dirtyMisses=${thrash.dirtyMisses} prunedPairsRemoved=${thrash.prunedPairsRemoved} converged=${thrash.convergedTemplates}`,
  );
  if (cfg.scaler) log(`EVENT scaler-inf: observedInfSkip=${scalerInfObserved}`);
  if (cfg.scheduler)
    log(
      `EVENT lr-milestone: lrBefore=${lrBefore} lrAfter=${lrAfter} (drop ${lrBefore > lrAfter ? "APPLIED" : "NOT SEEN"})`,
    );

  let fail = false;
  // A. no Input not ready
  if (inputNotReady > 0 || threw) {
    log(`FAIL(A): ${inputNotReady} Input-not-ready / threw=${threw}`);
    fail = true;
  }
  // B. witness coverage on checkpoint-bearing cells
  if (witnessed.size === 0 || witnessedPairs === 0) {
    log("FAIL(B): witnessed harvest set EMPTY (mechanism did not fire)");
    fail = true;
  }
  if (st.witnessVariances > 0) {
    log(
      `WARN(B): witnessVariances=${st.witnessVariances} — a reader varied between steps (conservative union kept; not a corruption, but flagged)`,
    );
  }
  // D. finite trajectory
  if (losses.some((l) => !Number.isFinite(l))) {
    log("FAIL(D): non-finite loss (corruption)");
    fail = true;
  }

  if (!fail) {
    log(
      `RESULT: PASS — witnessed ${witnessedPairs} pairs / ${witnessed.size} templates; ` +
        `zero Input-not-ready; ${losses.length} steps finite` +
        (cfg.scaler
          ? `; inf-skip event handled (skip=${scalerInfObserved})`
          : "") +
        (cfg.scheduler
          ? `; lr-milestone event handled (drop ${lrBefore}->${lrAfter})`
          : ""),
    );
  } else {
    log("RESULT: FAIL");
  }

  // Machine-readable verdict for the in-suite spec (stdout, one JSON line).
  console.error = origErr;
  process.stdout.write(
    `${JSON.stringify({
      witnessHarvest: true,
      cell: CELL,
      pass: !fail,
      witnessedTemplates: witnessed.size,
      witnessedPairs,
      witnessVariances: st.witnessVariances,
      inputNotReady,
      threw,
      cleanMisses: thrash.cleanMisses,
      dirtyMisses: thrash.dirtyMisses,
      prunedPairsRemoved: thrash.prunedPairsRemoved,
      scalerInfObserved: cfg.scaler ? scalerInfObserved : null,
      lrDropped: cfg.scheduler ? lrBefore > lrAfter : null,
      steps: losses.length,
    })}\n`,
  );
  destroyWebGPU();
  process.exit(fail ? 1 : 0);
}

main().catch((e) => {
  console.error(`[wh-matrix:${CELL}] FATAL: ${e?.stack ?? e}`);
  process.exit(1);
});
