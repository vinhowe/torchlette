/**
 * Test copying weights BEFORE any forward pass.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Copy Before Forward Test ===\n");

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

  console.log("Building model1...");
  const model1 = new GPT2(api, config, { device: "webgpu" });
  model1.eval();

  console.log("Building model2...");
  const model2 = new GPT2(api, config, { device: "webgpu" });
  model2.eval();

  // Copy weights BEFORE any forward pass
  console.log("\n--- Copying weights (before any forward) ---");
  const params1 = model1.parameters();
  const params2 = model2.parameters();
  for (let i = 0; i < params1.length; i++) {
    const data = await params1[i].cpu();
    const temp = api.tensorFromArray(data, params1[i].shape, {
      device: "webgpu",
      requiresGrad: true,
    });
    params2[i].copy_(temp);
  }
  await api.markStep();
  console.log("  Done");

  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  // Forward 1 (model1)
  console.log("\n--- Forward (model1) ---");
  const i1 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l1 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: loss1 } = model1.forwardWithLoss(i1, l1);
  const loss1Val = await loss1!.item();
  console.log(`  Loss: ${loss1Val.toFixed(6)}`);

  // Forward 2 (model2)
  console.log("\n--- Forward (model2) ---");
  const i2 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l2 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: loss2 } = model2.forwardWithLoss(i2, l2);
  const loss2Val = await loss2!.item();
  console.log(`  Loss: ${loss2Val.toFixed(6)}`);

  const match = Math.abs(loss1Val - loss2Val) < 1e-5;
  console.log(`\n  Models match: ${match ? "✓ PASS" : "✗ FAIL"}`);
  console.log(`  Diff: ${Math.abs(loss1Val - loss2Val).toExponential(4)}`);

  process.exit(match ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
