/**
 * Coverage-campaign C1 census probe: run the PACKED optimizer (foreach →
 * packOptimizerProgram, reached via TORCHLETTE_FUSED_ADAM=0) in the WHOLE-STEP
 * REMAT regime (api.wholeStep, the real-training substrate) with a GradScaler
 * (so the clip/finiteness `lt` cluster folds onto the packed buffer), and dump
 * the TORCHLETTE_DEBUG_CENSUS build-block census. The gate: the packed optimizer
 * plan (~500 nodes) reaches fullyCovered — no census miss line for it, and NO
 * line anywhere mentions `op:lt`.
 *
 * Run (solo GPU):
 *   eval "$(tools/pick-gpu.sh)"
 *   TORCHLETTE_FUSED_ADAM=0 TORCHLETTE_DEBUG_CENSUS=1 \
 *     npx tsx tools/t-coverage-census-packed.ts
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import type { Tensor } from "../src/frontend/tensor";

const ROOT = path.resolve(import.meta.dirname, "..");
const MODEL = process.env.MODEL ?? "distilgpt2";
const SEQ = parseInt(process.env.SEQ_LEN ?? "128", 10);
const STEPS = parseInt(process.env.STEPS ?? "10", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");
const log = (m: string) => console.error(`[census-packed] ${m}`);

async function main() {
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
  const opt = new Adam(params, { lr: LR, weightDecay: 0.0 }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  const V = model.config.vocabSize;

  const TOKENS =
    process.env.LOCAL_TOKENS ?? path.join(ROOT, "ckpts", "tinystories-tokens.bin");
  const tb = fs.readFileSync(TOKENS);
  const toks = new Uint16Array(tb.buffer, tb.byteOffset, tb.byteLength / 2);
  const inp = new Int32Array(SEQ);
  const tgt = new Int32Array(SEQ);

  for (let step = 0; step < STEPS; step++) {
    await scaler.resolveDeferred();
    const base = step * SEQ;
    for (let i = 0; i < SEQ; i++) {
      inp[i] = toks[base + i]! % V;
      tgt[i] = toks[base + i + 1]! % V;
    }
    const input = api.tensorFromArray(Array.from(inp), [1, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(Array.from(tgt), [1, SEQ], { device: "webgpu" });

    await api.beginStep();
    if (step >= 2) console.error(`--- STEP ${step} census ---`);
    const readLoss: Tensor = await api.wholeStep(async () => {
      const loss = api.tidy(() => {
        const l = model.forwardWithLoss(input, target).loss!;
        api.keep(l);
        return l;
      });
      const lossOut = api.noGrad(() => api.mul(loss, 1));
      api.registerState(lossOut);
      const backTgt = scaler.scale(loss);
      await backTgt.backward();
      scaler.unscale_(opt);
      scaler.step(opt);
      scaler.update();
      opt.zeroGrad();
      return lossOut;
    });
    api.endStep();
    await api.markStep();
    const lv = await readLoss.item();
    readLoss.dispose();
    input.dispose();
    target.dispose();
    if (step === 0)
      log(`step0 loss=${lv.toFixed(4)} (packed foreach + scaler, whole-step)`);
  }
  log("done");
  destroyWebGPU();
  process.exit(0);
}

void main();
