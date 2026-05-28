/**
 * Compare per-parameter gradient norms between autocast-on and autocast-off
 * for the same model, init, and inputs. Diff tells us where autocast is
 * losing signal during the fp16 backward — the parameter that loses the
 * most relative magnitude is the next bug to chase.
 *
 * Usage:
 *   VULKAN_DEVICE_INDEX=5 \
 *   LD_LIBRARY_PATH=tools/vk-shim \
 *   npx tsx tools/autocast-grad-compare.ts
 */

import * as fs from "node:fs";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { normal_ } from "../src/nn/init";
import { Adam, GradScaler } from "../src/optim";
import type { Tensor } from "../src/frontend/tensor";

const NUM_LAYERS = parseInt(process.env.NUM_LAYERS ?? "8", 10);
const NUM_HEADS = parseInt(process.env.NUM_HEADS ?? "4", 10);
const EMBED_DIM = parseInt(process.env.EMBED_DIM ?? "128", 10);
const SEQ_LEN = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "8", 10);
const SEED = parseInt(process.env.SEED ?? "42", 10);

const log = (m: string) => console.error(`[diag] ${m}`);

const STEPS = parseInt(process.env.STEPS ?? "5", 10);

async function buildAndRun(api: Torchlette, useAutocast: boolean) {
  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: NUM_LAYERS,
      numHeads: NUM_HEADS,
      embedDim: EMBED_DIM,
      dropoutRate: 0,
    },
    { rank: 1, alpha: 1 },
    "webgpu",
  );
  api.manualSeed(SEED);
  const params: Tensor[] = model.getAllParameters();
  for (const p of params) {
    if (p.shape.length >= 2) normal_(api, p, 0, 0.02);
  }
  await api._runtime().forceAllPending();

  model.train(true);
  model.enableCheckpointing(true);
  model.fullCheckpoint = true;
  model.setFullFinetuning(true);

  // Load TinyStories tokens and sample with the same RNG sequence so
  // autocast-on and autocast-off see the same windows. Real text exposes
  // the embedding/lm-head dynamics that random uniform tokens hide.
  const tokensBuf = fs.readFileSync(
    "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin",
  );
  const tokens = new Uint16Array(
    tokensBuf.buffer,
    tokensBuf.byteOffset,
    tokensBuf.byteLength / 2,
  );
  log(`Loaded ${tokens.length} TinyStories tokens`);
  let s = SEED;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s;
  };
  const inputsByStep: number[][] = [];
  const targetsByStep: number[][] = [];
  for (let step = 0; step < STEPS; step++) {
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < BATCH; i++) {
      const start = rand() % (tokens.length - SEQ_LEN - 1);
      for (let j = 0; j < SEQ_LEN; j++) {
        a.push(tokens[start + j]!);
        b.push(tokens[start + j + 1]!);
      }
    }
    inputsByStep.push(a);
    targetsByStep.push(b);
  }

  let scaler: GradScaler | null = null;
  if (useAutocast) {
    scaler = new GradScaler(api, { initScale: 1024.0 });
  }
  const opt = new Adam(params, { lr: 5e-4, weightDecay: 0.01, adamW: true }, api);

  const lossHistory: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    const input = api.tensorFromArray(inputsByStep[step]!, [BATCH, SEQ_LEN], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetsByStep[step]!, [BATCH, SEQ_LEN], {
      device: "webgpu",
    });

    const loss = api.tidy(() => {
      const fwd = () => model.forwardWithLoss(input, target).loss;
      const l = useAutocast ? api.autocast(fwd) : fwd();
      api.keep(l);
      return l;
    });

    const lossValue = await loss.item();
    lossHistory.push(lossValue);
    const backwardTarget = scaler ? scaler.scale(loss) : loss;
    await backwardTarget.backward();

    if (scaler) {
      scaler.unscale_(opt);
      scaler.step(opt);
      scaler.update();
    } else {
      opt.step();
    }
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  // Collect final param norms.
  const norms: { idx: number; shape: number[]; norm: number; dtype: string }[] = [];
  await api._runtime().forceAllPending();
  for (let i = 0; i < params.length; i++) {
    const arr = await params[i].cpu();
    let sq = 0;
    for (let j = 0; j < arr.length; j++) {
      sq += arr[j] * arr[j];
    }
    norms.push({
      idx: i,
      shape: params[i].shape,
      norm: Math.sqrt(sq),
      dtype: (params[i] as { dtype?: string }).dtype ?? "?",
    });
  }

  for (const p of params) p.dispose();
  return { lossValue: lossHistory[lossHistory.length - 1]!, norms, lossHistory };
}

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "4000";

  log("Run A: autocast OFF");
  const apiOff = new Torchlette("webgpu", { enableFusion: true });
  const off = await buildAndRun(apiOff, false);

  log("Run B: autocast ON");
  const apiOn = new Torchlette("webgpu", { enableFusion: true });
  const on = await buildAndRun(apiOn, true);

  log(`STEPS=${STEPS}`);
  log(
    `loss_off history: [${off.lossHistory.map((v) => v.toFixed(4)).join(", ")}]`,
  );
  log(
    `loss_on  history: [${on.lossHistory.map((v) => v.toFixed(4)).join(", ")}]`,
  );

  // Diff per-tensor norms.
  let offTotal = 0;
  let onTotal = 0;
  console.log(
    "idx,shape,dtype_off,dtype_on,norm_off,norm_on,ratio_on_over_off",
  );
  for (let i = 0; i < off.norms.length; i++) {
    const a = off.norms[i];
    const b = on.norms[i];
    offTotal += (a.norm || 0) ** 2;
    onTotal += (b.norm || 0) ** 2;
    const ratio = a.norm > 0 ? b.norm / a.norm : NaN;
    console.log(
      `${i},${a.shape.join("x")},${a.dtype},${b.dtype},${a.norm.toFixed(6)},${b.norm.toFixed(6)},${ratio.toFixed(4)}`,
    );
  }
  log(
    `TOTAL norm off=${Math.sqrt(offTotal).toFixed(4)} on=${Math.sqrt(onTotal).toFixed(4)} ratio=${(Math.sqrt(onTotal) / Math.sqrt(offTotal)).toFixed(4)}`,
  );

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
