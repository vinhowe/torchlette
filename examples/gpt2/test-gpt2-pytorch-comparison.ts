/**
 * GPT-2 PyTorch Comparison Test
 *
 * Tests that Torchlette GPT-2 produces the same outputs and gradients as PyTorch.
 * Tests both with and without checkpointing.
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import {
  runTorchOracleBackwardBatch,
  type OracleCase,
} from "../../test/oracle/torch-oracle";

// Tolerance for floating point comparison
// GPU matmul has different accumulation order than CPU
const ATOL = 0.15; // absolute tolerance
const RTOL = 0.15; // relative tolerance

// Smaller config for faster testing
const TEST_CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 32,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0, // Disable dropout for deterministic comparison
};

function allClose(
  actual: number[],
  expected: number[],
  atol = ATOL,
  rtol = RTOL,
): { pass: boolean; maxDiff: number; maxRelDiff: number; failIdx: number } {
  if (actual.length !== expected.length) {
    return { pass: false, maxDiff: Infinity, maxRelDiff: Infinity, failIdx: -1 };
  }
  let maxDiff = 0;
  let maxRelDiff = 0;
  let failIdx = -1;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const relDiff = diff / (Math.abs(expected[i]) + 1e-8);
    maxDiff = Math.max(maxDiff, diff);
    maxRelDiff = Math.max(maxRelDiff, relDiff);
    if (diff > atol + rtol * Math.abs(expected[i])) {
      if (failIdx === -1) failIdx = i;
    }
  }
  return { pass: failIdx === -1, maxDiff, maxRelDiff, failIdx };
}

async function extractWeights(model: GPT2): Promise<number[][]> {
  const params = model.parameters();
  const weights: number[][] = [];
  for (const p of params) {
    const data = await p.cpu();
    weights.push(Array.from(data));
  }
  return weights;
}

async function testGPT2ForwardBackward(
  api: Torchlette,
  useCheckpoint: boolean,
): Promise<boolean> {
  const checkpointLabel = useCheckpoint ? "with checkpoint" : "without checkpoint";
  console.log(`\n=== GPT-2 Forward/Backward Test (${checkpointLabel}) ===`);

  // Create model
  const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
  model.eval(); // Disable dropout

  // Create deterministic input
  const batchSize = 2;
  const seqLen = 8;
  const inputData = Array.from({ length: batchSize * seqLen }, (_, i) =>
    i % TEST_CONFIG.vocabSize
  );
  const targetData = Array.from({ length: batchSize * seqLen }, (_, i) =>
    (i + 1) % TEST_CONFIG.vocabSize
  );

  // Extract weights for PyTorch comparison
  console.log("  Extracting model weights...");
  const weights = await extractWeights(model);
  console.log(`  Model has ${weights.length} parameter tensors`);

  // Build oracle inputs
  const oracleInputs = [
    { values: inputData, shape: [batchSize, seqLen] },
    { values: targetData, shape: [batchSize, seqLen] },
  ];

  // Add all weight tensors
  const params = model.parameters();
  for (let i = 0; i < params.length; i++) {
    oracleInputs.push({
      values: weights[i],
      shape: params[i].shape,
    });
  }

  // Get PyTorch reference
  console.log("  Getting PyTorch reference...");
  const oracleCase: OracleCase = {
    op: "gpt2_forward_backward",
    caseName: `gpt2_${checkpointLabel.replace(/ /g, "_")}`,
    inputs: oracleInputs,
    options: {
      vocabSize: TEST_CONFIG.vocabSize,
      blockSize: TEST_CONFIG.blockSize,
      embedDim: TEST_CONFIG.embedDim,
      numLayers: TEST_CONFIG.numLayers,
      numHeads: TEST_CONFIG.numHeads,
    },
  };

  let pytorchResult;
  try {
    [pytorchResult] = await runTorchOracleBackwardBatch([oracleCase]);
  } catch (e) {
    console.error("  PyTorch oracle failed:", e);
    return false;
  }
  console.log(`  PyTorch loss: ${pytorchResult.output.values[0].toFixed(6)}`);

  // Run Torchlette
  console.log(`  Running Torchlette (${checkpointLabel})...`);
  const inputTokens = api.tensorFromArray(inputData, [batchSize, seqLen], {
    device: "webgpu",
  });
  const targets = api.tensorFromArray(targetData, [batchSize, seqLen], {
    device: "webgpu",
  });

  // Zero gradients
  for (const p of params) {
    p.zeroGrad();
  }

  // Forward pass
  const { loss } = model.forwardWithLoss(inputTokens, targets, {
    useCheckpoint,
  });

  if (!loss) {
    console.error("  No loss returned");
    return false;
  }

  const lossVal = await loss.item();
  console.log(`  Torchlette loss: ${lossVal.toFixed(6)}`);

  // Backward pass
  await loss.backward();
  await api.markStep();

  // Compare loss
  const lossCheck = allClose([lossVal], pytorchResult.output.values);
  console.log(
    `  Loss match: ${lossCheck.pass ? "✓" : "✗"} (maxDiff=${lossCheck.maxDiff.toExponential(2)})`
  );

  // Compare gradients for key parameters
  let allGradsMatch = true;
  const paramNames = [
    "wte.weight",
    "wpe.weight",
    "block0.ln1.weight",
    "block0.attn.cAttn.weight",
    "block0.mlp.cFc.weight",
    "lnF.weight",
  ];

  console.log("  Gradient comparison (sample):");
  for (let i = 0; i < Math.min(params.length, pytorchResult.grads.length); i++) {
    const param = params[i];
    if (!param.grad) {
      console.log(`    Param ${i}: no gradient`);
      continue;
    }

    const torchletteGrad = await param.grad.cpu();
    const pytorchGrad = pytorchResult.grads[i].values;

    const gradCheck = allClose(Array.from(torchletteGrad), pytorchGrad);

    const name = i < paramNames.length ? paramNames[i] : `param${i}`;
    const status = gradCheck.pass ? "✓" : "✗";
    console.log(
      `    ${name}: ${status} (maxDiff=${gradCheck.maxDiff.toExponential(2)}, shape=[${param.shape}])`
    );

    if (!gradCheck.pass) {
      allGradsMatch = false;
      // Show first few values for debugging
      console.log(`      First 5 torchlette: [${Array.from(torchletteGrad).slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
      console.log(`      First 5 pytorch:    [${pytorchGrad.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
    }
  }

  // Cleanup
  inputTokens.dispose();
  targets.dispose();
  loss.dispose();
  for (const p of params) {
    p.dispose();
  }

  const passed = lossCheck.pass && allGradsMatch;
  console.log(`  Overall: ${passed ? "PASS" : "FAIL"}`);

  return passed;
}

async function testCheckpointEquivalence(api: Torchlette): Promise<boolean> {
  console.log("\n=== Checkpoint Equivalence Test ===");
  console.log("  Verifying checkpoint produces same results as no-checkpoint...");

  // Create deterministic data
  const batchSize = 2;
  const seqLen = 8;
  const inputData = Array.from({ length: batchSize * seqLen }, (_, i) =>
    (i * 7) % TEST_CONFIG.vocabSize
  );
  const targetData = Array.from({ length: batchSize * seqLen }, (_, i) =>
    ((i + 3) * 11) % TEST_CONFIG.vocabSize
  );

  // Run without checkpoint
  const model1 = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
  model1.eval();

  const params1 = model1.parameters();
  for (const p of params1) p.zeroGrad();

  const input1 = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
  const target1 = api.tensorFromArray(targetData, [batchSize, seqLen], { device: "webgpu" });

  const { loss: loss1 } = model1.forwardWithLoss(input1, target1, { useCheckpoint: false });
  const lossVal1 = await loss1!.item();
  await loss1!.backward();
  await api.markStep();

  const grads1: number[][] = [];
  for (const p of params1) {
    if (p.grad) {
      grads1.push(Array.from(await p.grad.cpu()));
    }
  }

  // Create second model with same weights
  const model2 = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
  model2.eval();

  // Copy weights from model1 to model2
  const weights1 = await extractWeights(model1);
  const params2 = model2.parameters();
  for (let i = 0; i < params2.length; i++) {
    const p2 = params2[i];
    const w1 = weights1[i];
    const newTensor = api.tensorFromArray(w1, p2.shape, {
      device: "webgpu",
      requiresGrad: true,
    });
    // Replace the weight data
    p2.copy_(newTensor);
    newTensor.dispose();
  }

  for (const p of params2) p.zeroGrad();

  const input2 = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
  const target2 = api.tensorFromArray(targetData, [batchSize, seqLen], { device: "webgpu" });

  const { loss: loss2 } = model2.forwardWithLoss(input2, target2, { useCheckpoint: true });
  const lossVal2 = await loss2!.item();
  await loss2!.backward();
  await api.markStep();

  const grads2: number[][] = [];
  for (const p of params2) {
    if (p.grad) {
      grads2.push(Array.from(await p.grad.cpu()));
    }
  }

  // Compare
  console.log(`  Loss without checkpoint: ${lossVal1.toFixed(6)}`);
  console.log(`  Loss with checkpoint:    ${lossVal2.toFixed(6)}`);

  const lossCheck = allClose([lossVal1], [lossVal2], 1e-5, 1e-5);
  console.log(`  Loss match: ${lossCheck.pass ? "✓" : "✗"} (diff=${Math.abs(lossVal1 - lossVal2).toExponential(2)})`);

  let allGradsMatch = true;
  for (let i = 0; i < Math.min(grads1.length, grads2.length); i++) {
    const gradCheck = allClose(grads1[i], grads2[i], 1e-5, 1e-5);
    if (!gradCheck.pass) {
      console.log(`  Gradient ${i} mismatch: maxDiff=${gradCheck.maxDiff.toExponential(2)}`);
      allGradsMatch = false;
    }
  }

  if (allGradsMatch) {
    console.log(`  All gradients match: ✓`);
  }

  // Cleanup
  input1.dispose();
  target1.dispose();
  loss1!.dispose();
  input2.dispose();
  target2.dispose();
  loss2!.dispose();
  for (const p of params1) p.dispose();
  for (const p of params2) p.dispose();

  const passed = lossCheck.pass && allGradsMatch;
  console.log(`  Overall: ${passed ? "PASS" : "FAIL"}`);

  return passed;
}

async function main() {
  console.log("===========================================");
  console.log("  GPT-2 PyTorch Comparison Test Suite");
  console.log("===========================================");

  console.log("\nInitializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", {
    enableFusion: false, // Disable fusion for clearer comparison
    enableMemoryPlanning: false,
  });

  const results: { name: string; pass: boolean }[] = [];

  try {
    // Test 1: GPT-2 forward/backward without checkpoint vs PyTorch
    results.push({
      name: "GPT-2 vs PyTorch (no checkpoint)",
      pass: await testGPT2ForwardBackward(api, false),
    });

    // Test 2: GPT-2 forward/backward with checkpoint vs PyTorch
    results.push({
      name: "GPT-2 vs PyTorch (with checkpoint)",
      pass: await testGPT2ForwardBackward(api, true),
    });

    // Test 3: Checkpoint equivalence (internal consistency)
    results.push({
      name: "Checkpoint equivalence",
      pass: await testCheckpointEquivalence(api),
    });
  } catch (error) {
    console.error("\nTest failed with error:", error);
    process.exit(1);
  }

  // Summary
  console.log("\n===========================================");
  console.log("  Summary");
  console.log("===========================================");

  let allPassed = true;
  for (const result of results) {
    console.log(`  ${result.pass ? "✓" : "✗"} ${result.name}`);
    if (!result.pass) allPassed = false;
  }

  console.log("");
  if (allPassed) {
    console.log("All tests passed!");
    process.exit(0);
  } else {
    console.log("Some tests failed.");
    process.exit(1);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
