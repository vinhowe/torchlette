/**
 * Test that the same model produces consistent forward pass results.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Model Forward Consistency Test ===\n");

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
  model.eval(); // Ensure no dropout

  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  // Run forward twice on same model, same input
  console.log("\n--- Forward pass 1 ---");
  const input1 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label1 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: loss1 } = model.forwardWithLoss(input1, label1);
  const loss1Val = await loss1!.item();
  console.log(`  Loss: ${loss1Val.toFixed(6)}`);

  console.log("\n--- Forward pass 2 (same model, fresh tensors) ---");
  const input2 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label2 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });
  const { loss: loss2 } = model.forwardWithLoss(input2, label2);
  const loss2Val = await loss2!.item();
  console.log(`  Loss: ${loss2Val.toFixed(6)}`);

  const match = Math.abs(loss1Val - loss2Val) < 1e-5;
  console.log(`\n  Forward consistency: ${match ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(match ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
