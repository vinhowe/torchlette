/**
 * Test that full optimization pass (forward, backward, step) is equivalent
 * with and without checkpoint.
 *
 * Uses the SAME model for both tests by saving/restoring weights.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { SGD } from "../../src/optim";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Full Optimization Pass Equivalence Test ===\n");

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
  model.eval(); // No dropout
  const params = model.parameters();
  console.log(`  Parameters: ${params.length}`);

  // Save initial weights
  console.log("\nSaving initial weights...");
  const savedWeights: Float32Array[] = [];
  for (const p of params) {
    savedWeights.push(new Float32Array(await p.cpu()));
  }

  // Create optimizer
  const lr = 0.01;
  let opt = new SGD(params, { lr }, api);

  // Fixed input/labels
  const batchSize = 2;
  const seqLen = 16;
  const inputIds = new Int32Array(batchSize * seqLen);
  const labels = new Int32Array(batchSize * seqLen);
  for (let i = 0; i < inputIds.length; i++) {
    inputIds[i] = (i * 7 + 3) % config.vocabSize;
    labels[i] = (i * 11 + 5) % config.vocabSize;
  }

  // ============================================
  // Run 1: WITHOUT checkpoint
  // ============================================
  console.log("\n--- Run 1: WITHOUT checkpoint ---");

  opt.zeroGrad();

  const input1 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label1 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });

  const { loss: loss1 } = model.forwardWithLoss(input1, label1);
  const loss1Val = await loss1!.item();
  console.log(`  Loss: ${loss1Val.toFixed(6)}`);

  await loss1!.backward();
  await api.markStep();

  // Get gradients before step
  let gradSum1 = 0;
  for (const p of opt.getParams()) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let j = 0; j < g.length; j++) gradSum1 += Math.abs(g[j]);
    }
  }
  console.log(`  GradSum: ${gradSum1.toFixed(4)}`);

  // Step
  const newParams1 = opt.step();
  await api.markStep();

  // Save weights after step
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
    // Use the newParams1 array since that's what the optimizer has after step
    newParams1[i].copy_(temp);
  }
  await api.markStep();

  // Verify restored
  let restored = true;
  for (let i = 0; i < params.length; i++) {
    const current = await newParams1[i].cpu();
    for (let j = 0; j < current.length; j++) {
      if (Math.abs(current[j] - savedWeights[i][j]) > 1e-6) {
        restored = false;
        break;
      }
    }
  }
  console.log(`  Restored: ${restored ? "✓" : "✗"}`);

  // Recreate optimizer
  opt = new SGD(newParams1, { lr }, api);

  // ============================================
  // Run 2: WITH checkpoint
  // ============================================
  console.log("\n--- Run 2: WITH checkpoint ---");

  opt.zeroGrad();

  const input2 = api.tensorFromArray(inputIds, [batchSize, seqLen], { device: "webgpu" });
  const label2 = api.tensorFromArray(labels, [batchSize, seqLen], { device: "webgpu" });

  // Wrap forward in checkpoint
  const checkpointedLoss = checkpoint(
    api,
    (inp: Tensor, lbl: Tensor) => {
      const result = model.forwardWithLoss(inp, lbl);
      return result.loss!;
    },
    [input2, label2]
  );
  const loss2Val = await checkpointedLoss.item();
  console.log(`  Loss: ${loss2Val.toFixed(6)}`);

  await checkpointedLoss.backward();
  await api.markStep();

  // Get gradients before step
  let gradSum2 = 0;
  for (const p of opt.getParams()) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let j = 0; j < g.length; j++) gradSum2 += Math.abs(g[j]);
    }
  }
  console.log(`  GradSum: ${gradSum2.toFixed(4)}`);

  // Step
  const newParams2 = opt.step();
  await api.markStep();

  // Save weights after step
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

  const gradRatio = gradSum2 > 0 ? gradSum1 / gradSum2 : (gradSum1 === 0 ? 1 : Infinity);
  const gradMatch = Math.abs(gradRatio - 1) < 0.01;
  console.log(`  GradSum match (1% tol): ${gradMatch ? "✓" : "✗"} (${gradSum1.toFixed(4)} vs ${gradSum2.toFixed(4)})`);

  let maxWeightDiff = 0;
  let totalWeightDiff = 0;
  let numWeights = 0;
  for (let i = 0; i < weightsAfterRun1.length; i++) {
    const d1 = weightsAfterRun1[i];
    const d2 = weightsAfterRun2[i];
    for (let j = 0; j < d1.length; j++) {
      const diff = Math.abs(d1[j] - d2[j]);
      maxWeightDiff = Math.max(maxWeightDiff, diff);
      totalWeightDiff += diff;
      numWeights++;
    }
  }
  const avgWeightDiff = totalWeightDiff / numWeights;
  const weightMatch = maxWeightDiff < 1e-3;

  console.log(`  Max weight diff: ${maxWeightDiff.toExponential(4)}`);
  console.log(`  Avg weight diff: ${avgWeightDiff.toExponential(4)}`);
  console.log(`  Weights match (1e-3 tol): ${weightMatch ? "✓" : "✗"}`);

  // Final result
  console.log("\n=== Final Result ===");
  const allPass = lossMatch && gradMatch && weightMatch;
  console.log(
    allPass
      ? "✓ Full optimization pass is EQUIVALENT with and without checkpoint"
      : "✗ MISMATCH detected between checkpoint and non-checkpoint"
  );

  process.exit(allPass ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
