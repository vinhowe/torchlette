/**
 * LR-schedule-through-replay gate for the FUSED Lion/SGD optimizers
 * (derived-optimizer-realizer R5c).
 *
 * The fused Lion/SGD `optStep` kernels read the learning rate as a persistent
 * on-device LIVE SCALAR (core/live-scalar.ts) — the SAME primitive Adam's lr
 * rides. `setLR(v)` writes it IN-PLACE, graph-ordered, and the step-tape scalar
 * re-dress must re-deliver the per-step value through a COMPILED replay. If the
 * value were instead baked into the recording (the frozen-scalar class), a
 * compiled replay would train with a STALE lr and diverge from the lowered path.
 *
 * This gate trains a tiny GPT-2 with a CHANGING lr each step (cosine-ish decay)
 * and runs the trajectory TWICE — compiled plan ON (default) vs OFF
 * (TORCHLETTE_COMPILED_PLAN=0) — from an identical start, asserting per-step
 * losses agree to ≤1e-5 over ≥30 steps (across the compiled-plan activation
 * threshold, Corollary 2). Agreement proves the scheduled lr flows as DATA
 * through replay; divergence would flag a frozen lr.
 *
 * Env: STEPS (default 30), TOL (default 1e-5), OPTS (csv: lion,sgd).
 *
 * NOTE — run ONE optimizer per process. The fused optimizer path uses several
 * PROCESS-GLOBAL caches (packed DMA scratch, opt-step dispatcher, scalar-slot
 * re-dress registry, compiled-plan templates). A hard switch between two DISTINCT
 * fused optimizers mid-process (e.g. `OPTS=lion,sgd`) lets the first's cached
 * state bleed into the second's compiled replay — NOT a real training scenario
 * (training uses one optimizer), the same cross-optimizer class the parity tool
 * documents. The STANDING gate is `OPTS=lion` and `OPTS=sgd` as SEPARATE
 * invocations; each passes at 0.0 EXACT (lr flows as DATA through replay — no
 * frozen scalar).
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import type { Tensor } from "../src/frontend/tensor";
import { Torchlette } from "../src/frontend/torchlette";
import { Lion, SGD } from "../src/optim/index.ts";

const STEPS = parseInt(process.env.STEPS ?? "30", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-5");
// Default to a SINGLE optimizer (see the process-global-cache note above — a hard
// switch between two distinct fused optimizers mid-process is not isolatable). The
// standing gate runs `OPTS=lion` and `OPTS=sgd` as separate invocations.
const OPTS = (process.env.OPTS ?? "lion").split(",").map((s) => s.trim());
if (OPTS.length > 1)
  console.error(
    "[lr-replay] WARNING: multiple optimizers in one process share global fused " +
      "caches — run each in a SEPARATE process for a clean result.",
  );

const VOCAB = 256;
const NUM_LAYERS = 2;
const NUM_HEADS = 2;
const EMBED = 64;
const SEQ = 16;
const BATCH = 2;

const log = (m: string) => console.error(`[lr-replay] ${m}`);

function batchFor(step: number): { input: Int32Array; target: Int32Array } {
  const input = new Int32Array(BATCH * SEQ);
  const target = new Int32Array(BATCH * SEQ);
  for (let b = 0; b < BATCH; b++) {
    for (let t = 0; t < SEQ; t++) {
      const base = (step * 131 + b * 17 + t * 7) % VOCAB;
      input[b * SEQ + t] = base;
      target[b * SEQ + t] = (base + 1 + step) % VOCAB;
    }
  }
  return { input, target };
}

// A non-trivial per-step lr schedule (cosine decay from lrMax to lrMax/10). The
// point is that lr CHANGES every step so a frozen recording would diverge.
function scheduledLR(step: number, lrMax: number): number {
  const frac = step / Math.max(1, STEPS - 1);
  const cos = 0.5 * (1 + Math.cos(Math.PI * frac));
  return lrMax * (0.1 + 0.9 * cos);
}

type OptMaker = (params: Tensor[], api: Torchlette) => {
  step(): void;
  zeroGrad(): void;
  setLR(lr: number): void;
};

const OPT_MAKERS: Record<string, { lrMax: number; make: OptMaker }> = {
  lion: {
    lrMax: 1e-3,
    make: (p, api) => new Lion(p, { lr: 1e-3, weightDecay: 0.01 }, api),
  },
  sgd: {
    lrMax: 2e-2,
    make: (p, api) =>
      new SGD(p, { lr: 2e-2, momentum: 0.9, weightDecay: 0.01 }, api),
  },
};

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", { enableFusion: true });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    {
      vocabSize: VOCAB,
      blockSize: 64,
      numLayers: NUM_LAYERS,
      numHeads: NUM_HEADS,
      embedDim: EMBED,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );
  model.train(true);
  const params = model.parameters();
  const named = model.namedParameters();

  await api.beginStep();
  const init = new Map<string, Float32Array>();
  for (const [name, p] of named) init.set(name, Float32Array.from(await p.cpu()));
  api.endStep();
  await api.markStep();

  async function restore(): Promise<void> {
    await api.beginStep();
    for (const [name, p] of named) {
      const data = init.get(name)!;
      api.copy_(
        p,
        api.tensorFromArray(Array.from(data), p.shape, { device: "webgpu" }),
      );
    }
    api.endStep();
    await api.markStep();
  }

  async function runArm(
    optName: string,
    compiledOn: boolean,
  ): Promise<number[]> {
    if (compiledOn) delete process.env.TORCHLETTE_COMPILED_PLAN;
    else process.env.TORCHLETTE_COMPILED_PLAN = "0";
    // Both arms are the SAME optimizer (compiled vs lowered), so its own packed
    // scratch reuse across arms is normal/safe (like reuse across steps) — no
    // cross-optimizer cache reset is needed (one optimizer per process).
    await restore();
    const spec = OPT_MAKERS[optName];
    const opt = spec.make(params, api);
    const losses: number[] = [];
    for (let step = 0; step < STEPS; step++) {
      await api.beginStep();
      // Deliver the scheduled lr for THIS step BEFORE the update (the scheduler
      // seam: setLR writes the persistent live-scalar in place).
      opt.setLR(scheduledLR(step, spec.lrMax));
      const { input, target } = batchFor(step);
      const inp = api.tensorFromArray(Array.from(input), [BATCH, SEQ], {
        device: "webgpu",
      });
      const tgt = api.tensorFromArray(Array.from(target), [BATCH, SEQ], {
        device: "webgpu",
      });
      const loss = api.tidy(() => {
        const l = model.forwardWithLoss(inp, tgt).loss!;
        api.keep(l);
        return l;
      });
      losses.push(await loss.item());
      await loss.backward();
      opt.step();
      opt.zeroGrad();
      inp.dispose();
      tgt.dispose();
      api.endStep();
      await api.markStep();
    }
    return losses;
  }

  let anyFail = false;
  for (const optName of OPTS) {
    if (!OPT_MAKERS[optName]) {
      log(`unknown optimizer '${optName}' — skipping`);
      continue;
    }
    const compiled = await runArm(optName, true);
    const lowered = await runArm(optName, false);
    let maxDiff = 0;
    let atStep = -1;
    for (let s = 0; s < STEPS; s++) {
      const d = Math.abs(compiled[s]! - lowered[s]!);
      if (d > maxDiff) {
        maxDiff = d;
        atStep = s;
      }
    }
    const finite = compiled.every(Number.isFinite) && lowered.every(Number.isFinite);
    const pass = finite && maxDiff <= TOL;
    if (!pass) anyFail = true;
    log(
      `${optName}: compiled-vs-lowered maxDiff=${maxDiff.toExponential(3)} @step${atStep} ` +
        `(tol ${TOL.toExponential(1)}) finite=${finite} → ${pass ? "PASS" : "FAIL"}`,
    );
    log(
      `  compiled first=${compiled[0]!.toFixed(6)} last=${compiled[STEPS - 1]!.toFixed(6)}; ` +
        `lowered first=${lowered[0]!.toFixed(6)} last=${lowered[STEPS - 1]!.toFixed(6)}`,
    );
  }

  delete process.env.TORCHLETTE_COMPILED_PLAN;
  await destroyWebGPU();
  if (anyFail) {
    log("RESULT: FAIL");
    process.exit(1);
  }
  log("RESULT: PASS");
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  log(`STACK: ${(e as Error).stack}`);
  process.exit(1);
});
