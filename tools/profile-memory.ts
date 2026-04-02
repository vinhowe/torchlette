/**
 * Profile GPU memory usage during one training step.
 * Reports framework-tracked bytes at each phase.
 */
import {
  destroyWebGPU,
  getGPUMemoryStats,
  initWebGPU,
} from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn";
import { Adam } from "../src/optim";

const B = parseInt(process.env.BATCH_SIZE ?? "4", 10);
const S = parseInt(process.env.SEQ_LEN ?? "256", 10);

function mem(): string {
  const s = getGPUMemoryStats();
  return `${Math.round(s.currentBytes / 1e6)}MB (peak ${Math.round(s.peakBytes / 1e6)}MB, allocs ${s.allocationCount})`;
}

async function main() {
  await initWebGPU();
  console.log(`Config: batch=${B} seq=${S}`);
  console.log(`0. Init:          ${mem()}`);

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);

  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 12,
      numHeads: 12,
      embedDim: 768,
      dropoutRate: 0,
    },
    { rank: 1, alpha: 1 },
    "webgpu",
  );
  model.setFullFinetuning(true);
  model.enableCheckpointing(true);
  model.fullCheckpoint = true; // recompute all layers, minimal memory
  const params = model.getAllParameters();

  // Force weight materialization
  await api._runtime().forceAllPending();
  console.log(`1. Model loaded:  ${mem()}`);

  // Create batch
  await api.beginStep();
  const inputData = new Array(B * S).fill(0).map((_, i) => i % 50257);
  const targetData = new Array(B * S).fill(0).map((_, i) => (i + 1) % 50257);
  const input = api.tensorFromArray(inputData, [B, S], { device: "webgpu" });
  const target = api.tensorFromArray(targetData, [B, S], { device: "webgpu" });

  // Forward
  const loss = api.tidy(() => {
    const l = api.autocast(() => model.forwardWithLoss(input, target).loss);
    api.keep(l);
    return l;
  });
  const lossVal = await loss.item();
  console.log(`2. After forward: ${mem()} (loss=${lossVal.toFixed(2)})`);

  // Backward
  await loss.backward();
  console.log(`3. After backward:${mem()}`);

  // Optimizer
  clipGradNorm_(api, params, 1.0);
  const opt = new Adam(params, { lr: 1e-4 }, api);
  opt.step();
  opt.zeroGrad();
  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();
  console.log(`4. After cleanup: ${mem()}`);

  // Theoretical
  const paramCount = params.reduce(
    (s, p) => s + p.shape.reduce((a, b) => a * b, 1),
    0,
  );
  const theoryMB = Math.round((paramCount * 4 * 4) / 1e6); // weights + adam + grads
  console.log(`\nTheory (params+adam+grads): ${theoryMB}MB`);
  console.log(
    `Actual steady state: ${Math.round(getGPUMemoryStats().currentBytes / 1e6)}MB`,
  );
  console.log(
    `Overhead: ${Math.round(getGPUMemoryStats().currentBytes / 1e6 - theoryMB)}MB (${((getGPUMemoryStats().currentBytes / 1e6 / theoryMB - 1) * 100).toFixed(0)}%)`,
  );

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
