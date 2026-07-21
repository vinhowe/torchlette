/**
 * The chain-packing cross-path differential (docs/chain-packing-design.md §4).
 *
 * Corollary-1 guard: the PACKED optimizer path (packOptimizerProgram — one flat
 * chain per isomorphism class) must produce the SAME trajectory as the per-param
 * path, over a real toy model, ACROSS the compiled-plan activation threshold
 * (Corollary 2 — single-step parity is worthless; the compiled plan only builds
 * on the 2nd+ execution and only covers what runs inside plans).
 *
 * For each wired optimizer it trains a small GPT-2 for N≥30 steps from an
 * identical start, twice — packed vs unpacked — and asserts per-step losses agree
 * to ~1e-5 (P1: Adam foreach; P2: + Lion, SGD).
 *
 *   packed   Adam := foreach (TORCHLETTE_FUSED_ADAM=0), the packOptimizerProgram path
 *   unpacked Adam := per-param elementwise (+ TORCHLETTE_FOREACH_ADAM=0)
 *   packed   Lion/SGD := default (packOptimizerProgram)
 *   unpacked Lion/SGD := per-param (TORCHLETTE_PACK_OPTIM=0)
 *
 * Env: STEPS (default 30), TOL (default 1e-4), OPTS (csv: adam,lion,sgd).
 */

import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import type { Tensor } from "../src/frontend/tensor";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, Lion, SGD } from "../src/optim/index.ts";

const STEPS = parseInt(process.env.STEPS ?? "30", 10);
const TOL = parseFloat(process.env.TOL ?? "1e-4");
const OPTS = (process.env.OPTS ?? "adam,lion,sgd").split(",").map((s) => s.trim());

const VOCAB = 256;
const NUM_LAYERS = 2;
const NUM_HEADS = 2;
const EMBED = 64;
const SEQ = 16;
const BATCH = 2;

const log = (m: string) => console.error(`[pack-parity] ${m}`);

// Deterministic per-step token windows (fixed, arm-independent).
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

type OptMaker = (params: Tensor[], api: Torchlette) => {
  step(): void;
  zeroGrad(): void;
};

interface Arm {
  name: string;
  make: OptMaker;
  setEnv: () => void;
}

const OPT_ARMS: Record<string, { packed: Arm; unpacked: Arm }> = {
  adam: {
    packed: {
      name: "adam/packed(foreach)",
      // Foreach is reached when the fused kernel is disabled; foreach IS the
      // packOptimizerProgram path.
      setEnv: () => {
        process.env.TORCHLETTE_FUSED_ADAM = "0";
        delete process.env.TORCHLETTE_FOREACH_ADAM;
      },
      make: (p, api) => new Adam(p, { lr: 1e-3, weightDecay: 0.01, adamW: true }, api),
    },
    unpacked: {
      name: "adam/unpacked(per-param)",
      setEnv: () => {
        process.env.TORCHLETTE_FUSED_ADAM = "0";
        process.env.TORCHLETTE_FOREACH_ADAM = "0";
      },
      make: (p, api) => new Adam(p, { lr: 1e-3, weightDecay: 0.01, adamW: true }, api),
    },
  },
  lion: {
    packed: {
      name: "lion/packed",
      setEnv: () => {
        delete process.env.TORCHLETTE_PACK_OPTIM;
      },
      make: (p, api) => new Lion(p, { lr: 1e-4, weightDecay: 0.01 }, api),
    },
    unpacked: {
      name: "lion/unpacked(per-param)",
      setEnv: () => {
        process.env.TORCHLETTE_PACK_OPTIM = "0";
      },
      make: (p, api) => new Lion(p, { lr: 1e-4, weightDecay: 0.01 }, api),
    },
  },
  sgd: {
    packed: {
      name: "sgd/packed",
      setEnv: () => {
        delete process.env.TORCHLETTE_PACK_OPTIM;
      },
      make: (p, api) =>
        new SGD(p, { lr: 1e-2, momentum: 0.9, weightDecay: 0.01 }, api),
    },
    unpacked: {
      name: "sgd/unpacked(per-param)",
      setEnv: () => {
        process.env.TORCHLETTE_PACK_OPTIM = "0";
      },
      make: (p, api) =>
        new SGD(p, { lr: 1e-2, momentum: 0.9, weightDecay: 0.01 }, api),
    },
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

  // Snapshot the initial weights so every arm starts identically.
  await api.beginStep();
  const init = new Map<string, Float32Array>();
  for (const [name, p] of named) init.set(name, Float32Array.from(await p.cpu()));
  api.endStep();
  await api.markStep();
  log(
    `model: L${NUM_LAYERS} H${NUM_HEADS} E${EMBED} vocab${VOCAB} seq${SEQ} batch${BATCH}, ${params.length} params, ${STEPS} steps`,
  );

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

  async function runArm(arm: Arm): Promise<number[]> {
    arm.setEnv();
    await restore();
    const opt = arm.make(params, api);
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
      const lossVal = await loss.item();
      losses.push(lossVal);
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
    const arms = OPT_ARMS[optName];
    if (!arms) {
      log(`unknown optimizer '${optName}' — skipping`);
      continue;
    }
    const packed = await runArm(arms.packed);
    const unpacked = await runArm(arms.unpacked);

    let maxDiff = 0;
    let atStep = -1;
    for (let s = 0; s < STEPS; s++) {
      const d = Math.abs(packed[s]! - unpacked[s]!);
      if (d > maxDiff) {
        maxDiff = d;
        atStep = s;
      }
    }
    const finite = packed.every(Number.isFinite) && unpacked.every(Number.isFinite);
    const pass = finite && maxDiff <= TOL;
    if (!pass) anyFail = true;
    log(
      `${optName}: maxDiff=${maxDiff.toExponential(3)} @step${atStep} ` +
        `(tol ${TOL.toExponential(1)}) finite=${finite} → ${pass ? "PASS" : "FAIL"}`,
    );
    log(
      `  ${arms.packed.name}   first=${packed[0]!.toFixed(6)} last=${packed[STEPS - 1]!.toFixed(6)}`,
    );
    log(
      `  ${arms.unpacked.name} first=${unpacked[0]!.toFixed(6)} last=${unpacked[STEPS - 1]!.toFixed(6)}`,
    );
  }

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
