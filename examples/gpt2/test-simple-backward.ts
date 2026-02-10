/**
 * Test simple backward on GPT-2 without checkpoint.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Simple Backward Test ===\n");

  const config: GPT2Config = {
    vocabSize: 128,
    blockSize: 32,
    numLayers: 2,
    numHeads: 2,
    embedDim: 64,
    dropoutRate: 0,
  };

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  console.log("Building model...");
  const model = new GPT2(api, config, { device: "webgpu" });
  model.eval();
  const params = model.parameters();

  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  console.log("\n--- Forward + Backward (no checkpoint) ---");

  const input = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });

  const { loss } = model.forwardWithLoss(input, label);
  console.log(`  Loss: ${(await loss!.item()).toFixed(6)}`);

  console.log("  Calling backward...");
  await loss!.backward();
  console.log("  Calling markStep...");
  await api.markStep();

  let gradSum = 0;
  for (const p of params) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let i = 0; i < g.length; i++) gradSum += Math.abs(g[i]);
    }
  }
  console.log(`  GradSum: ${gradSum.toFixed(4)}`);

  console.log("\n✓ Done");
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
