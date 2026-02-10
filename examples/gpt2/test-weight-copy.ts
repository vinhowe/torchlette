/**
 * Test that weight copying between models produces identical forward results.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Weight Copy Test ===\n");

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

  console.log("Building model 1...");
  const model1 = new GPT2(api, config, { device: "webgpu" });
  model1.eval();
  const params1 = model1.parameters();

  console.log("Building model 2...");
  const model2 = new GPT2(api, config, { device: "webgpu" });
  model2.eval();
  const params2 = model2.parameters();

  // Before copy: forward passes should be different
  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  console.log("\n--- Before copying weights ---");
  const i1a = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l1a = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: lossA1 } = model1.forwardWithLoss(i1a, l1a);
  const lossA1Val = await lossA1!.item();
  console.log(`  Model 1 loss: ${lossA1Val.toFixed(6)}`);

  const i1b = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l1b = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: lossB1 } = model2.forwardWithLoss(i1b, l1b);
  const lossB1Val = await lossB1!.item();
  console.log(`  Model 2 loss: ${lossB1Val.toFixed(6)}`);
  console.log(`  Different (expected): ${Math.abs(lossA1Val - lossB1Val) > 1e-5 ? "✓" : "✗"}`);

  // Copy weights
  console.log("\n--- Copying weights from model1 to model2 ---");
  for (let i = 0; i < params1.length; i++) {
    const data = await params1[i].cpu();
    const temp = api.tensorFromArray(data, params1[i].shape, {
      device: "webgpu",
      requiresGrad: true,
    });
    params2[i].copy_(temp);
  }
  await api.markStep();

  // Verify weights match
  console.log("  Verifying weights...");
  let allMatch = true;
  for (let i = 0; i < params1.length; i++) {
    const d1 = await params1[i].cpu();
    const d2 = await params2[i].cpu();
    for (let j = 0; j < d1.length; j++) {
      if (Math.abs(d1[j] - d2[j]) > 1e-6) {
        allMatch = false;
        console.log(`    Mismatch at param ${i}, index ${j}: ${d1[j]} vs ${d2[j]}`);
        break;
      }
    }
    if (!allMatch) break;
  }
  console.log(`  Weights match: ${allMatch ? "✓" : "✗"}`);

  // After copy: forward passes should be identical
  console.log("\n--- After copying weights ---");
  const i2a = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l2a = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: lossA2 } = model1.forwardWithLoss(i2a, l2a);
  const lossA2Val = await lossA2!.item();
  console.log(`  Model 1 loss: ${lossA2Val.toFixed(6)}`);

  const i2b = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l2b = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: lossB2 } = model2.forwardWithLoss(i2b, l2b);
  const lossB2Val = await lossB2!.item();
  console.log(`  Model 2 loss: ${lossB2Val.toFixed(6)}`);

  const identical = Math.abs(lossA2Val - lossB2Val) < 1e-5;
  console.log(`  Identical (expected): ${identical ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(identical ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
