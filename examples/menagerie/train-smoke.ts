/**
 * Minimal Node repro of the Menagerie train→serialize loop (no browser, no HF).
 * Drives the actual src/lib/train.ts against a tiny model so we can iron out the
 * "Tensor has been disposed" crash on the GPU box.
 *
 * Run: cd examples/menagerie && npx tsx train-smoke.ts
 */
import { Torchlette, initWebGPU, getWebGPUInitError } from "torchlette";
import {
  GPT2WithLoRA,
  createLoRAConfig,
  serializeLoRAToSafetensors,
} from "gpt2-browser";
import { trainModel } from "./src/lib/train";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);

  const cfg = {
    vocabSize: 256,
    blockSize: 64,
    numLayers: 2,
    numHeads: 2,
    embedDim: 64,
    dropoutRate: 0,
  };
  const model = new GPT2WithLoRA(api, cfg, createLoRAConfig(8), "webgpu");
  model.initWeightsGPT2(); // nanoGPT-style from-scratch init
  model.disableAllLora();
  model.setFullFinetuning(true);

  const tokens: number[] = [];
  for (let i = 0; i < 4096; i++) tokens.push(i % cfg.vocabSize);

  console.log("starting train loop…");
  const result = await trainModel({
    api,
    model,
    tokens,
    config: { steps: 25, batchSize: 4, seqLength: 16, lr: 1e-3, weightDecay: 0.1 },
    onStep: ({ step, loss }) => console.log(`  step ${step}  loss ${loss.toFixed(4)}`),
  });
  console.log("train result:", result);

  console.log("serializing…");
  const tensors = await model.exportBaseWeights();
  const buf = serializeLoRAToSafetensors(tensors, { format: "menagerie" });
  console.log("serialized bytes:", buf.byteLength);

  console.log("OK");
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
