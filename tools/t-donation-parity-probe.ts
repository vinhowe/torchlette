/**
 * Buffer-donation P2 parity probe (docs/buffer-donation-design.md §2.5).
 *
 * Runs the SAME packed-optimizer trajectory (foreach/packOptimizerProgram) twice
 * on a tiny real GPT-2 — once with `TORCHLETTE_PLANNER_DONATION=1`, once OFF —
 * from byte-identical initial weights, and asserts per-step losses are
 * BIT-EXACT across the compiled-plan activation threshold (2nd+ step). Donation
 * changes only BUFFER ASSIGNMENT, never values: any drift is a real aliasing bug
 * (the §7.2 "later data leaks into earlier results" class), never a tolerance to
 * widen. The strict `[lifetime]` guard stays default-on for both arms.
 *
 * NOTE (§2.5): on this workload the packed plans are UNCOVERED (they run on the
 * recorded build, not the generated stream) and the packed buffers are oversized
 * → chunked, so the donation edge is currently INERT — this probe's ON arm is
 * bit-identical to OFF BY CONSTRUCTION (donation never fires). It is the standing
 * SAFETY guard that the opt-in flag can never perturb a trajectory; it becomes a
 * live differential the moment the coverage prerequisite (§2.5) lands.
 *
 * ENV: OPT (adam|lion|sgd, default adam), STEPS (default 12 — crosses the
 * threshold), TOL (default 0 — bit-exact).
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import type { Tensor } from "../src/frontend/tensor";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, Lion, SGD } from "../src/optim/index.ts";

const OPT = (process.env.OPT ?? "adam").trim();
const STEPS = parseInt(process.env.STEPS ?? "12", 10);
const TOL = parseFloat(process.env.TOL ?? "0");
const VOCAB = 256;
const NUM_LAYERS = 2;
const NUM_HEADS = 2;
const EMBED = 64;
const SEQ = 16;
const BATCH = 2;

const log = (m: string) => console.error(`[donation-parity] ${m}`);

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

function makeOpt(params: Tensor[], api: Torchlette) {
  // Route through the PACKED (foreach / packOptimizerProgram) path — the target
  // of the donation edge — for every optimizer.
  process.env.TORCHLETTE_FUSED_ADAM = "0";
  delete process.env.TORCHLETTE_FOREACH_ADAM;
  delete process.env.TORCHLETTE_PACK_OPTIM;
  if (OPT === "adam")
    return new Adam(params, { lr: 1e-3, weightDecay: 0.01, adamW: true }, api);
  if (OPT === "lion") return new Lion(params, { lr: 1e-4, weightDecay: 0.01 }, api);
  if (OPT === "sgd")
    return new SGD(params, { lr: 1e-2, momentum: 0.9, weightDecay: 0.01 }, api);
  throw new Error(`unknown OPT '${OPT}'`);
}

async function main(): Promise<void> {
  if (!(await initWebGPU())) {
    log("SKIP: no WebGPU");
    process.exit(0);
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

  async function runArm(donation: boolean): Promise<number[]> {
    if (donation) process.env.TORCHLETTE_PLANNER_DONATION = "1";
    else delete process.env.TORCHLETTE_PLANNER_DONATION;
    await restore();
    const opt = makeOpt(params, api);
    const losses: number[] = [];
    for (let step = 0; step < STEPS; step++) {
      await api.beginStep();
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

  const off = await runArm(false);
  const on = await runArm(true);

  let maxDiff = 0;
  let atStep = -1;
  for (let s = 0; s < STEPS; s++) {
    const d = Math.abs(on[s]! - off[s]!);
    if (d > maxDiff) {
      maxDiff = d;
      atStep = s;
    }
  }
  const finite = off.every(Number.isFinite) && on.every(Number.isFinite);
  const pass = finite && maxDiff <= TOL;
  log(
    `${OPT}: donation ON vs OFF maxDiff=${maxDiff.toExponential(3)} @step${atStep} ` +
      `(tol ${TOL.toExponential(1)}) finite=${finite} → ${pass ? "PASS" : "FAIL"}`,
  );
  log(`  OFF first=${off[0]!.toFixed(6)} last=${off[STEPS - 1]!.toFixed(6)}`);
  log(`  ON  first=${on[0]!.toFixed(6)} last=${on[STEPS - 1]!.toFixed(6)}`);

  await destroyWebGPU();
  if (!pass) {
    log("RESULT: FAIL");
    process.exit(1);
  }
  log("RESULT: PASS");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
