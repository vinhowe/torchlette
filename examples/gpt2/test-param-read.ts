/**
 * Test if reading parameters via .cpu() affects forward pass results.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Parameter Read Test ===\n");

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

  // Forward 1
  console.log("\n--- Forward 1 ---");
  const i1 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l1 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: loss1 } = model.forwardWithLoss(i1, l1);
  const loss1Val = await loss1!.item();
  console.log(`  Loss: ${loss1Val.toFixed(6)}`);

  // Read all parameters via .cpu()
  console.log("\n--- Reading parameters via .cpu() ---");
  for (let i = 0; i < params.length; i++) {
    const data = await params[i].cpu();
    console.log(`  Param ${i}: read ${data.length} values`);
  }

  // markStep
  console.log("\n--- markStep() ---");
  await api.markStep();

  // Forward 2
  console.log("\n--- Forward 2 (after reading params + markStep) ---");
  const i2 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const l2 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: loss2 } = model.forwardWithLoss(i2, l2);
  const loss2Val = await loss2!.item();
  console.log(`  Loss: ${loss2Val.toFixed(6)}`);

  const match = Math.abs(loss1Val - loss2Val) < 1e-5;
  console.log(`\n  Forward consistency: ${match ? "✓ PASS" : "✗ FAIL"}`);
  console.log(`  Diff: ${Math.abs(loss1Val - loss2Val).toExponential(4)}`);

  process.exit(match ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
