/**
 * Test checkpoint equivalence with checkpoint FIRST.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { SGD } from "../../src/optim";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Checkpoint Order Test (checkpoint first) ===\n");

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

  // Save initial weights
  console.log("\nSaving initial weights...");
  const savedWeights: Float32Array[] = [];
  for (const p of params) {
    savedWeights.push(new Float32Array(await p.cpu()));
  }

  const lr = 0.01;
  let opt = new SGD(params, { lr }, api);

  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  // ============================================
  // Run 1: WITH checkpoint FIRST
  // ============================================
  console.log("\n--- Run 1: WITH checkpoint ---");

  opt.zeroGrad();

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
  for (const p of opt.getParams()) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let j = 0; j < g.length; j++) gradSum1 += Math.abs(g[j]);
    }
  }
  console.log(`  GradSum: ${gradSum1.toFixed(4)}`);

  const newParams1 = opt.step();
  await api.markStep();

  const weightsAfterRun1: Float32Array[] = [];
  for (const p of newParams1) {
    weightsAfterRun1.push(new Float32Array(await p.cpu()));
  }

  // ============================================
  // Restore initial weights
  // ============================================
  console.log("\n--- Restoring initial weights ---");
  for (let i = 0; i < params.length; i++) {
    const temp = api.tensorFromArray(savedWeights[i], params[i].shape, {
      device: "webgpu",
      requiresGrad: true,
    });
    newParams1[i].copy_(temp);
  }
  await api.markStep();

  opt = new SGD(newParams1, { lr }, api);

  // ============================================
  // Run 2: WITHOUT checkpoint
  // ============================================
  console.log("\n--- Run 2: WITHOUT checkpoint ---");

  opt.zeroGrad();

  const input2 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label2 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });

  const { loss: loss2 } = model.forwardWithLoss(input2, label2);
  const loss2Val = await loss2!.item();
  console.log(`  Loss: ${loss2Val.toFixed(6)}`);

  await loss2!.backward();
  await api.markStep();

  let gradSum2 = 0;
  for (const p of opt.getParams()) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let j = 0; j < g.length; j++) gradSum2 += Math.abs(g[j]);
    }
  }
  console.log(`  GradSum: ${gradSum2.toFixed(4)}`);

  const newParams2 = opt.step();
  await api.markStep();

  const weightsAfterRun2: Float32Array[] = [];
  for (const p of newParams2) {
    weightsAfterRun2.push(new Float32Array(await p.cpu()));
  }

  // ============================================
  // Compare results
  // ============================================
  console.log("\n--- Comparison ---");

  const lossMatch = Math.abs(loss1Val - loss2Val) < 1e-4;
  console.log(`  Loss match: ${lossMatch ? "✓" : "✗"} (${loss1Val.toFixed(6)} vs ${loss2Val.toFixed(6)})`);

  const gradRatio = Math.abs(gradSum1 - gradSum2) / Math.max(gradSum1, gradSum2);
  const gradMatch = gradRatio < 0.01;
  console.log(`  GradSum match (1% tol): ${gradMatch ? "✓" : "✗"} (${gradSum1.toFixed(4)} vs ${gradSum2.toFixed(4)})`);

  let maxWeightDiff = 0;
  for (let i = 0; i < weightsAfterRun1.length; i++) {
    const d1 = weightsAfterRun1[i];
    const d2 = weightsAfterRun2[i];
    for (let j = 0; j < d1.length; j++) {
      maxWeightDiff = Math.max(maxWeightDiff, Math.abs(d1[j] - d2[j]));
    }
  }
  const weightMatch = maxWeightDiff < 1e-3;
  console.log(`  Weights match (1e-3 tol): ${weightMatch ? "✓" : "✗"} (max diff: ${maxWeightDiff.toExponential(4)})`);

  console.log("\n=== Final Result ===");
  const allPass = lossMatch && gradMatch && weightMatch;
  console.log(allPass
    ? "✓ Full optimization pass is EQUIVALENT"
    : "✗ MISMATCH detected");

  process.exit(allPass ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
