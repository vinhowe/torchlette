/**
 * Self-contained compiled-plan parity probe (no PyTorch fixtures).
 *
 * Trains a TINY-but-real GPT-2 (random structure, DETERMINISTICALLY
 * overwritten params so two processes init identically) through the full
 * production inner step — autocast(f16) + gradient checkpointing +
 * GradScaler + grad-clip + AdamW — and prints the per-step loss trajectory
 * as JSON. Big enough to trigger the compiled plan (multi-layer → arena
 * populated → compiled replay on 2nd+ step), so running it under
 * TORCHLETTE_COMPILED_PLAN=0 vs default is the compiled-vs-lowered
 * trajectory gate (the frozen-step_size / clip-divergence class), and under
 * TORCHLETTE_STREAM_GENERATE=1 it exercises the stage-4 generator diff.
 *
 * Deterministic init keeps the trajectory reproducible across processes
 * WITHOUT a global RNG seed: LayerNorm weight=1/bias=0, other biases=0,
 * everything else a small bounded sin pattern (stable descent).
 *
 * Consumed by test/compiled-plan-parity.spec.ts. process.exit(0) at end
 * (Dawn holds threads).
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { Adam, GradScaler } from "../src/optim/index.ts";
import { clipGradNorm_ } from "../src/nn/index.ts";

const STEPS = parseInt(process.env.STEPS ?? "12", 10);
const VOCAB = 256;
const SEQ = 32;
const BATCH = 2;

function fillFor(name: string, n: number): number[] {
  // Role-aware deterministic init for a stable, reproducible descent.
  const lower = name.toLowerCase();
  const isNorm = lower.includes("ln") || lower.includes("norm");
  const isBias = lower.endsWith("bias") || lower.endsWith(".b");
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    if (isNorm && !isBias) out[i] = 1.0;
    else if (isBias) out[i] = 0.0;
    else out[i] = 0.02 * Math.sin(i * 0.1 + name.length);
  }
  return out;
}

async function main() {
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "4000";
  if (!(await initWebGPU())) {
    console.error("no webgpu");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });
  const { GPT2 } = await import("../examples/gpt2/model.ts");
  const model = new GPT2(
    api,
    {
      vocabSize: VOCAB,
      blockSize: 64,
      numLayers: 2,
      numHeads: 2,
      embedDim: 64,
      dropoutRate: 0,
    },
    { device: "webgpu" },
  );
  const named = new Map<string, Tensor>(model.namedParameters());
  const paddedVocab = (model as { paddedVocabSize: number }).paddedVocabSize;

  // Deterministic overwrite of every param (identical across processes).
  await api.beginStep();
  for (const [name, param] of named) {
    const shape = (param as unknown as { shape: number[] }).shape;
    const n = shape.reduce((a, b) => a * b, 1);
    api.copy_(
      param,
      api.tensorFromArray(fillFor(name, n), shape, { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();

  const params = [...named.values()];
  const opt = new Adam(params, { lr: 5e-4, weightDecay: 0.01 }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });

  // Deterministic token data (no RNG): bounded, repeats across steps.
  const tok = (i: number) => Math.abs(Math.floor(Math.sin(i * 1.7) * 1e4)) % VOCAB;

  const losses: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    await api.beginStep();
    const inp = Array.from({ length: BATCH * SEQ }, (_, i) => tok(step * 1000 + i));
    const tgt = Array.from({ length: BATCH * SEQ }, (_, i) => tok(step * 1000 + i + 1));
    const input = api.tensorFromArray(inp, [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(tgt, [BATCH, SEQ], { device: "webgpu" });
    const loss = api.tidy(() => {
      const l = api.autocast(
        () => model.forwardWithLoss(input, target, { useCheckpoint: true }).loss,
      );
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    losses.push(lossVal);
    await scaler.scale(loss).backward();
    scaler.unscale_(opt);
    clipGradNorm_(api, params, 1.0);
    scaler.step(opt);
    scaler.update();
    opt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }
  void paddedVocab;
  console.log(JSON.stringify({ losses }));
  try {
    destroyWebGPU();
  } catch {
    /* ignore */
  }
  process.exit(0);
}
main();
