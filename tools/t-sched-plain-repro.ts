/**
 * Minimal repro probe for the PLAIN-path (no capture) LR-scheduler
 * bimodality: the t-train-capture-probe's control arm sporadically lands on
 * a distinct SLOWER trajectory (final loss 11.132 vs 10.666, ~1-in-4 runs)
 * while the captured arm is identical across runs. Runs the same plain
 * distilgpt2 training loop (autocast + ckpt + GradScaler + clip + AdamW +
 * CosineAnnealingLR at driver level) TWICE in one process and reports both
 * finals. No step-tape involvement unless TORCHLETTE_STEP_TAPE=1 is set.
 */
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const STEPS = parseInt(process.env.STEPS ?? "30", 10);
const BATCH = 1;
const SEQ = parseInt(process.env.SEQ_LEN ?? "512", 10);
const log = (m: string) => console.error(`[sched-plain] ${m}`);

async function run(tag: string): Promise<number[]> {
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
  const sched = new CosineAnnealingLR(
    opt,
    STEPS,
    parseFloat(process.env.ETA_MIN ?? "1e-5"),
  );
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;
  let seed = 1234;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
  const inp = new Int32Array(BATCH * SEQ);
  const tgt = new Int32Array(BATCH * SEQ);
  const losses: number[] = [];
  for (let stp = 0; stp < STEPS; stp++) {
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
    const l = api.tidy(() => {
      const ll = api.autocast(
        () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(ll);
      return ll;
    });
    const lossOut = api.noGrad(() => api.mul(l, 1));
    const scaled = scaler.scale(l);
    await scaled.backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    scaled.dispose();
    losses.push(await lossOut.item());
    lossOut.dispose();
    input.dispose();
    target.dispose();
    sched.step();
  }
  log(`${tag} losses: ${losses.map((l) => l.toFixed(4)).join(", ")}`);
  return losses;
}

async function main() {
  await initWebGPU();
  const a = await run("armA");
  const b = await run("armB");
  let maxD = 0;
  for (let i = 0; i < a.length; i++)
    maxD = Math.max(maxD, Math.abs(a[i] - b[i]));
  console.log(
    JSON.stringify({ finalA: a[a.length - 1], finalB: b[b.length - 1], maxD }),
  );
  destroyWebGPU();
  process.exit(0);
}
main();
