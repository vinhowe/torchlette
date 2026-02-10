/**
 * Test checkpoint FIRST (before any non-checkpoint backward).
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { SGD } from "../../src/optim";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Checkpoint First Test ===\n");

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
  console.log(`  Parameters: ${params.length}`);

  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  // ============================================
  // Run WITH checkpoint FIRST
  // ============================================
  console.log("\n--- Run WITH checkpoint ---");

  const input1 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label1 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });

  const checkpointedLoss = checkpoint(
    api,
    (inp: Tensor, lbl: Tensor) => {
      const result = model.forwardWithLoss(inp, lbl);
      return result.loss!;
    },
    [input1, label1]
  );
  const loss1Val = await checkpointedLoss.item();
  console.log(`  Loss: ${loss1Val.toFixed(6)}`);

  await checkpointedLoss.backward();
  await api.markStep();

  let gradSum1 = 0;
  for (const p of params) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let i = 0; i < g.length; i++) gradSum1 += Math.abs(g[i]);
    }
  }
  console.log(`  GradSum: ${gradSum1.toFixed(4)}`);
  console.log(`  First param has grad: ${params[0].grad !== undefined}`);

  // Check specific params
  if (params[0].grad) {
    const g = await params[0].grad.cpu();
    console.log(`  params[0].grad[0:5]: ${Array.from(g.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);
  }

  process.exit(gradSum1 > 0 ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
