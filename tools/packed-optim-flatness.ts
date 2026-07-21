/**
 * Packed-optimizer memory FLATNESS gate (buffer-donation design P1, §2 / §5).
 *
 * The packed foreach optimizer path (`packOptimizerProgram`) is default-on for
 * Lion/SGD and reached for Adam via `TORCHLETTE_FUSED_ADAM=0`. It MUST reach a
 * flat steady state: storage count and GPU peak/current must PLATEAU across
 * steps, not climb monotonically. The §2 leak — the unpack `reshape(narrow(...))`
 * materializations were never disposed, leaking ~1 full-param buffer per param
 * per step (+78 storages/step on distilgpt2, +320 MB/step) — is the class this
 * gate pins closed. A single late-step memory read cannot distinguish a flat
 * premium from an accumulation; this tool reads the TREND.
 *
 * It trains a small GPT-2 with the PACKED path for N (≥30) steps, records the
 * reachable-storage count and GPU current/peak bytes AFTER each markStep, and
 * asserts the late-window storage count and current bytes are FLAT (no per-step
 * growth beyond a small tolerance). Runs across the compiled-plan activation
 * threshold (default) unless TORCHLETTE_COMPILED_PLAN=0 (lowered-path leak).
 *
 * Env: STEPS (default 34), OPT (adam|lion|sgd, default adam), SETTLE (default
 * 12 — first step of the flat window), SLOPE_TOL (default 0.5 storages/step).
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, Lion, SGD } from "../src/optim/index.ts";
import { storageTracker } from "../src/graph/storage-tracker";

const STEPS = parseInt(process.env.STEPS ?? "34", 10);
const OPT = (process.env.OPT ?? "adam").trim();
const SETTLE = parseInt(process.env.SETTLE ?? "12", 10);
const SLOPE_TOL = parseFloat(process.env.SLOPE_TOL ?? "0.5");

const VOCAB = 256;
const NUM_LAYERS = 2;
const NUM_HEADS = 2;
const EMBED = 64;
const SEQ = 16;
const BATCH = 2;

const log = (m: string) => console.error(`[flatness] ${m}`);

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

function makeOpt(opt: string, params: unknown[], api: Torchlette) {
  const p = params as never;
  switch (opt) {
    case "adam":
      // Foreach (packed) path: fused kernel off, per-param off.
      process.env.TORCHLETTE_FUSED_ADAM = "0";
      delete process.env.TORCHLETTE_FOREACH_ADAM;
      return new Adam(p, { lr: 1e-3, weightDecay: 0.01, adamW: true }, api);
    case "lion":
      delete process.env.TORCHLETTE_PACK_OPTIM;
      return new Lion(p, { lr: 1e-4, weightDecay: 0.01 }, api);
    case "sgd":
      delete process.env.TORCHLETTE_PACK_OPTIM;
      return new SGD(p, { lr: 1e-2, momentum: 0.9, weightDecay: 0.01 }, api);
    default:
      throw new Error(`unknown OPT '${opt}'`);
  }
}

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
  const opt = makeOpt(OPT, params, api);

  log(
    `OPT=${OPT} packed path, L${NUM_LAYERS} H${NUM_HEADS} E${EMBED}, ${params.length} params, ${STEPS} steps, ` +
      `compiled_plan=${process.env.TORCHLETTE_COMPILED_PLAN ?? "default(on)"}`,
  );

  const storages: number[] = [];
  const curMB: number[] = [];
  const peakMB: number[] = [];

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
    await loss.item();
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    inp.dispose();
    tgt.dispose();
    api.endStep();
    await api.markStep();

    const st = storageTracker.stats();
    const mem = api.runtime._debug_getMemoryStats();
    storages.push(st.totalStorages);
    curMB.push(mem.gpuCurrentBytes / 1024 / 1024);
    peakMB.push(mem.gpuPeakBytes / 1024 / 1024);
    log(
      `step ${String(step).padStart(2)}: storages=${st.totalStorages} ` +
        `reachable=${st.reachableStorages} cur=${(mem.gpuCurrentBytes / 1024 / 1024).toFixed(1)}MB ` +
        `peak=${(mem.gpuPeakBytes / 1024 / 1024).toFixed(1)}MB`,
    );
  }

  // Least-squares slope of storage count over the flat window [SETTLE, STEPS).
  const idx: number[] = [];
  for (let i = SETTLE; i < STEPS; i++) idx.push(i);
  const slope = (ys: number[]): number => {
    const n = idx.length;
    const mx = idx.reduce((a, b) => a + b, 0) / n;
    const my = idx.reduce((a, i) => a + ys[i]!, 0) / n;
    let num = 0;
    let den = 0;
    for (const i of idx) {
      num += (i - mx) * (ys[i]! - my);
      den += (i - mx) * (i - mx);
    }
    return num / den;
  };

  const stSlope = slope(storages);
  const curSlope = slope(curMB);
  const stFirst = storages[SETTLE]!;
  const stLast = storages[STEPS - 1]!;

  log("");
  log(
    `FLAT WINDOW [${SETTLE},${STEPS}): storages ${stFirst}→${stLast} ` +
      `(slope ${stSlope.toFixed(3)} /step, tol ${SLOPE_TOL}); ` +
      `cur slope ${curSlope.toFixed(3)} MB/step`,
  );

  const storageFlat = Math.abs(stSlope) <= SLOPE_TOL && stLast <= stFirst + 2;
  const memFlat = curSlope <= 1.0; // MB/step; a real leak is +320MB/step
  const pass = storageFlat && memFlat;

  await destroyWebGPU();
  if (!pass) {
    log(
      `RESULT: FAIL (storageFlat=${storageFlat} memFlat=${memFlat}) — packed path is LEAKING`,
    );
    process.exit(1);
  }
  log("RESULT: PASS — packed path memory is FLAT");
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  log(`STACK: ${(e as Error).stack}`);
  process.exit(1);
});
