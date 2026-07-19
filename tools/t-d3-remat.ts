/**
 * THE D3 GATE — traced-remat-compiled vs the arena-free checkpoint bypass
 * (docs/step-function-compiler-design.md P3; docs/arena-recompute-design.md §R3).
 *
 * The oldest accepted regression (D3): checkpointed training was locked off the
 * fast path — `webgpu-gpt2-trainer.ts` calls `setBufferArenaDisabled(true)` under
 * checkpointing (the b66ead78 bypass) because the compiled two-plan path pinned
 * the cross-plan checkpoint saves whole-step (task #99 R2/R3 STOP). P3's bet: the
 * whole-step trace collapses those cross-plan RESULT pins into intra-graph edges
 * the memory planner packs — so checkpointed training runs compiled AND at the
 * bypass's memory. This tool renders the verdict.
 *
 * Two arms, each an ISOLATED child process (flags read once at module load),
 * each REPEATED (default 3) as a FRESH process (clean device → clean peak):
 *   - bypass : setBufferArenaDisabled(true), lowered, plain loop, checkpointing.
 *              The incumbent the gate must match-or-beat.
 *   - remat  : TORCHLETTE_WHOLE_STEP=1, arena ON, compiled, api.wholeStep(...)
 *              with deferred loss, checkpointing. The challenger (P3 remat).
 *
 * Both arms: same pretrained weights (deterministic load), same tokens, same
 * Adam, seq512, selective checkpoint for distil / full checkpoint for medium.
 *
 * Peak discipline (matches t-planner-pin-attribution): STEPS=14, reset the peak
 * watermark at step 9 (pool reuse settled) → steadyPeak = max peak over the late
 * steps; steadyCurrent = last late reading; meanLateMs = mean late-step wall.
 *
 * VERDICT (the bypass dies iff ALL hold, per the P3 charter):
 *   (a) remat steadyPeak ≤ bypass steadyPeak × 1.05
 *   (b) remat meanLateMs ≤ bypass meanLateMs (speed ≥ bypass)
 *   (c) max|remat.losses − bypass.losses| ≤ 1e-5 (trajectory)
 *
 * Run (solo GPU, orchestrator spawns the arm children):
 *   VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim MODEL=distilgpt2 SELECTIVE=1 \
 *     npx tsx tools/t-d3-remat.ts
 *   VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim MODEL=gpt2-medium SELECTIVE=0 \
 *     npx tsx tools/t-d3-remat.ts
 * Env: MODEL SELECTIVE(=1) SEQ_LEN(=512) STEPS(=14) RESET_AT(=9) REPEATS(=3) LR(=1e-4)
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import {
  getSubmitCount,
  resetSubmitCount,
} from "../src/backend/webgpu/webgpu-state";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam } from "../src/optim/index.ts";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const STEPS = parseInt(process.env.STEPS ?? "14", 10);
const RESET_AT = parseInt(process.env.RESET_AT ?? "9", 10);
const REPEATS = parseInt(process.env.REPEATS ?? "3", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");
const SELECTIVE = process.env.SELECTIVE !== "0";
const log = (m: string) => console.error(`[d3-remat] ${m}`);

// Trajectory arms for the HONEST P3-method gate (t-whole-step-diff's method):
//   - "remat-lo": remat forced LOWERED (COMPILED_PLAN=0) → the compiled==lowered
//     fp-reorder FLOOR of the traced mode itself.
//   - "eager": the SAME-ARM eager reference (checkpoint, arena ON, NON-whole-step
//     — safe now: the D3 refusal keeps its plans lowered). The SOUND correctness
//     metric is remat-COMPILED vs eager, gated at max(1e-5, 1.5×floor). The
//     bypass (arena OFF) is a DIFFERENT executor → remat-vs-bypass is a cross-impl
//     floor, reported but NOT the gate.
type Arm = "bypass" | "remat" | "remat-lo" | "eager";

interface ArmResult {
  losses: number[];
  steadyPeakMB: number;
  steadyCurrentMB: number;
  meanLateMs: number;
  meanLateSubmits: number;
  /** Whether the big whole-step boundary plan reached compiled replay (a valid
   *  CompiledPlan with a large command count). The D3 deliverable: with CE-from-IR
   *  coverage the remat arm should show hasCompiled=true; the arena-free bypass
   *  never compiles. */
  hasCompiled: boolean;
  maxCompiledCmds: number;
}

async function run(arm: Arm): Promise<ArmResult> {
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "40000";
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  if (arm === "bypass") api._runtime().setBufferArenaDisabled(true);

  const model = await loadPretrainedGPT2(
    api,
    path.join(ROOT, "models", MODEL),
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: LR, weightDecay: 0.0 }, api);
  const V = model.config.vocabSize;

  const TOKENS =
    process.env.LOCAL_TOKENS ??
    path.join(ROOT, "ckpts", "tinystories-tokens.bin");
  const tb = fs.readFileSync(TOKENS);
  const toks = new Uint16Array(tb.buffer, tb.byteOffset, tb.byteLength / 2);
  const inp = new Int32Array(SEQ);
  const tgt = new Int32Array(SEQ);

  const ckptOpts = { useCheckpoint: true, selectiveCheckpoint: SELECTIVE };
  const losses: number[] = [];
  const stepMs: number[] = [];
  const stepSubmits: number[] = [];
  let steadyPeak = 0;
  let steadyCurrent = 0;

  for (let step = 0; step < STEPS; step++) {
    const t0 = performance.now();
    resetSubmitCount();
    const base = (step % 64) * SEQ;
    for (let i = 0; i < SEQ; i++) {
      inp[i] = toks[base + i]! % V;
      tgt[i] = toks[base + i + 1]! % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [1, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(Array.from(tgt), [1, SEQ], {
      device: "webgpu",
    });
    await api.beginStep();

    if (arm === "remat" || arm === "remat-lo") {
      const readLoss = await api.wholeStep(async () => {
        const loss = api.tidy(() => {
          const l = model.forwardWithLoss(input, target, ckptOpts).loss!;
          api.keep(l);
          return l;
        });
        const lossOut = api.noGrad(() => api.mul(loss, 1));
        api.registerState(lossOut);
        await loss.backward();
        opt.step();
        opt.zeroGrad();
        return lossOut;
      });
      api.endStep();
      await api.markStep();
      losses.push(await readLoss.item());
      readLoss.dispose();
    } else {
      const loss = api.tidy(() => {
        const l = model.forwardWithLoss(input, target, ckptOpts).loss!;
        api.keep(l);
        return l;
      });
      losses.push(await loss.item());
      await loss.backward();
      opt.step();
      opt.zeroGrad();
      loss.dispose();
      api.endStep();
      await api.markStep();
    }
    input.dispose();
    target.dispose();

    if (step === RESET_AT) {
      // @ts-expect-error test-only reset of the peak watermark
      gpuMemoryTracker.peakUsageBytes =
        gpuMemoryTracker.getCurrentAllocatedBytes();
    }
    if (step >= RESET_AT) {
      steadyCurrent = gpuMemoryTracker.getCurrentAllocatedBytes();
      steadyPeak = Math.max(steadyPeak, gpuMemoryTracker.getPeakUsageBytes());
    }
    stepSubmits.push(getSubmitCount());
    stepMs.push(performance.now() - t0);
    if (step < 3 || step >= STEPS - 2)
      log(`${arm} step ${step}: loss=${losses[step].toFixed(5)}`);
  }

  // Cutover diagnostic: did the whole-step checkpoint plan reach the compiled
  // (build-from-IR) path, or does selective/full checkpointing re-fingerprint
  // it lowered every step (arena-recompute Risk 2)?
  // hasCompiled probe: did the big whole-step boundary plan reach compiled
  // replay? A compiled ~800-node plan emits hundreds of commands; the small
  // readback/bookkeeping plans emit a handful. Threshold at 100 commands.
  let hasCompiled = false;
  let maxCompiledCmds = 0;
  try {
    const { debugTemplateCount, getCompiledStreams } = await import(
      "../src/executor/executor"
    );
    const { getObservedLivenessStats } = await import(
      "../src/executor/observed-liveness"
    );
    const streams = getCompiledStreams();
    for (const s of streams)
      maxCompiledCmds = Math.max(maxCompiledCmds, s.commands.length);
    hasCompiled = maxCompiledCmds >= 100;
    const s = getObservedLivenessStats();
    log(
      `${arm} CUTOVER: templates=${debugTemplateCount()} compiledStreams=${streams.length} maxCmds=${maxCompiledCmds} hasCompiled=${hasCompiled} converged=${s.convergedTemplates} pinned=${s.pinnedTemplates} retired=${s.retiredTemplates} cleanMiss=${s.cleanMisses} dirtyMiss=${s.dirtyMisses}`,
    );
  } catch {
    /* stats unavailable */
  }
  const late = stepMs.slice(RESET_AT);
  const lateSub = stepSubmits.slice(RESET_AT);
  return {
    losses,
    steadyPeakMB: +(steadyPeak / 1e6).toFixed(1),
    steadyCurrentMB: +(steadyCurrent / 1e6).toFixed(1),
    meanLateMs: +(late.reduce((a, b) => a + b, 0) / late.length).toFixed(1),
    meanLateSubmits: +(
      lateSub.reduce((a, b) => a + b, 0) / lateSub.length
    ).toFixed(1),
    hasCompiled,
    maxCompiledCmds,
  };
}

async function runArmInChild(arm: Arm, rep: number): Promise<ArmResult> {
  const { execFileSync } = await import("node:child_process");
  const env: Record<string, string> = { ...process.env, D3_ARM: arm };
  if (arm === "remat" || arm === "remat-lo") {
    env.TORCHLETTE_WHOLE_STEP = "1";
    if (arm === "remat-lo") env.TORCHLETTE_COMPILED_PLAN = "0";
    else delete env.TORCHLETTE_COMPILED_PLAN;
  } else {
    // The `bypass` arm is the eager reference — WHOLE_STEP is DEFAULT-ON since
    // P4a Stage 2, so opt out explicitly (=0), don't merely unset.
    env.TORCHLETTE_WHOLE_STEP = "0";
    delete env.TORCHLETTE_COMPILED_PLAN;
  }
  const out = execFileSync(
    process.execPath,
    ["--import", "tsx", import.meta.filename],
    {
      env,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
      maxBuffer: 64 * 1024 * 1024,
    },
  );
  const line = out.split("\n").find((l) => l.startsWith("=== ARM-RESULT === "));
  if (!line) throw new Error(`arm ${arm} rep ${rep}: no result line`);
  return JSON.parse(line.slice("=== ARM-RESULT === ".length)) as ArmResult;
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}
function maxDelta(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++)
    m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

async function aggregate(arm: Arm): Promise<{
  reps: ArmResult[];
  medPeak: number;
  medMs: number;
  medCur: number;
}> {
  const reps: ArmResult[] = [];
  for (let r = 0; r < REPEATS; r++) {
    const res = await runArmInChild(arm, r);
    reps.push(res);
    log(
      `${arm} rep ${r}: steadyPeak=${res.steadyPeakMB}MB current=${res.steadyCurrentMB}MB ${res.meanLateMs}ms/${res.meanLateSubmits}sub loss[0]=${res.losses[0].toFixed(4)} loss[${STEPS - 1}]=${res.losses[STEPS - 1].toFixed(4)}`,
    );
  }
  return {
    reps,
    medPeak: median(reps.map((r) => r.steadyPeakMB)),
    medMs: median(reps.map((r) => r.meanLateMs)),
    medCur: median(reps.map((r) => r.steadyCurrentMB)),
  };
}

async function main() {
  const armEnv = process.env.D3_ARM as Arm | undefined;
  if (
    armEnv === "bypass" ||
    armEnv === "remat" ||
    armEnv === "remat-lo" ||
    armEnv === "eager"
  ) {
    if (!(await initWebGPU())) {
      log("WebGPU init failed");
      process.exit(1);
    }
    const r = await run(armEnv);
    console.log(`=== ARM-RESULT === ${JSON.stringify(r)}`);
    destroyWebGPU();
    process.exit(0);
  }

  log(
    `config: MODEL=${MODEL} SEQ=${SEQ} STEPS=${STEPS} RESET_AT=${RESET_AT} REPEATS=${REPEATS} SELECTIVE=${SELECTIVE} LR=${LR}`,
  );
  const bypass = await aggregate("bypass");
  const remat = await aggregate("remat");
  // Trajectory references (1 rep each — deterministic weights/tokens):
  const rematLo = await runArmInChild("remat-lo", 0); // compiled==lowered floor
  const eager = await runArmInChild("eager", 0); // same-arm eager reference

  // P3-method trajectory gate (t-whole-step-diff's method): the compiled==lowered
  // delta IS the traced mode's fp-reorder FLOOR; the SOUND metric is
  // remat-COMPILED vs EAGER, gated at max(1e-5, 1.5×floor). A real math bug blows
  // past this by orders of magnitude; the seq512 two-path fp floor lifts it above
  // the nominal 1e-5 for BOTH comparisons in lockstep.
  const fpFloor = maxDelta(remat.reps[0].losses, rematLo.losses);
  const trajVsEager = maxDelta(remat.reps[0].losses, eager.losses);
  const trajGate = Math.max(1e-5, 1.5 * fpFloor);
  // Informational only: cross-implementation floor (remat-compiled vs the
  // arena-free bypass — two different executors).
  const trajVsBypass = maxDelta(remat.reps[0].losses, bypass.reps[0].losses);
  const peakRatio = remat.medPeak / bypass.medPeak;
  const peakOK = peakRatio <= 1.05;
  const speedOK = remat.medMs <= bypass.medMs;
  const trajOK = trajVsEager <= trajGate;
  const hasCompiledOK = remat.reps.every((r) => r.hasCompiled);
  const verdict = peakOK && speedOK && trajOK && hasCompiledOK;

  console.log("=== D3-REMAT TABLE ===");
  console.log(
    JSON.stringify(
      {
        model: MODEL,
        seq: SEQ,
        selective: SELECTIVE,
        repeats: REPEATS,
        bypass: {
          medSteadyPeakMB: bypass.medPeak,
          medSteadyCurrentMB: bypass.medCur,
          medMeanLateMs: bypass.medMs,
          peaksMB: bypass.reps.map((r) => r.steadyPeakMB),
          msPerStep: bypass.reps.map((r) => r.meanLateMs),
          submits: bypass.reps.map((r) => r.meanLateSubmits),
        },
        remat: {
          medSteadyPeakMB: remat.medPeak,
          medSteadyCurrentMB: remat.medCur,
          medMeanLateMs: remat.medMs,
          peaksMB: remat.reps.map((r) => r.steadyPeakMB),
          msPerStep: remat.reps.map((r) => r.meanLateMs),
          submits: remat.reps.map((r) => r.meanLateSubmits),
          hasCompiled: remat.reps.map((r) => r.hasCompiled),
          maxCompiledCmds: remat.reps.map((r) => r.maxCompiledCmds),
        },
        peakRatio_remat_over_bypass: +peakRatio.toFixed(4),
        traj_remat_vs_eager_SOUND: trajVsEager,
        traj_gate_floorAware: trajGate,
        fpFloor_compiledVsLowered: fpFloor,
        traj_vsBypass_crossImplFloor: trajVsBypass,
        rematHasCompiled: remat.reps.map((r) => r.hasCompiled),
        gates: { hasCompiledOK, peakOK, speedOK, trajOK },
      },
      null,
      2,
    ),
  );
  console.log(
    verdict
      ? `VERDICT: D3-READY — remat COMPILED (cmds ${remat.reps[0].maxCompiledCmds}), peak ${remat.medPeak}MB ≤ bypass ${bypass.medPeak}MB×1.05 (${(peakRatio * 100).toFixed(1)}%), speed ${remat.medMs}ms ≤ bypass ${bypass.medMs}ms, traj(remat-vs-eager) ${trajVsEager.toExponential(2)} ≤ ${trajGate.toExponential(2)}`
      : `VERDICT: NOT-READY — hasCompiled=${hasCompiledOK} peakOK=${peakOK}(${(peakRatio * 100).toFixed(1)}%) speedOK=${speedOK}(${remat.medMs}vs${bypass.medMs}ms) trajOK=${trajOK}(remat-vs-eager ${trajVsEager.toExponential(2)} vs gate ${trajGate.toExponential(2)})`,
  );
  process.exit(verdict ? 0 : 2);
}

main().catch((e) => {
  log(`FATAL ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
