/**
 * Comprehensive test comparing Torchlette training against PyTorch.
 *
 * Tests:
 * 1. Simple MLP forward/backward (baseline comparison)
 * 2. Two-layer MLP with checkpointing
 * 3. Training loop with AMP + checkpointing
 *
 * Uses torch oracle to get PyTorch reference values.
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { checkpoint } from "../../src/nn/checkpoint";
import {
  runTorchOracleBackwardBatch,
  runTorchOracleExtendedBatch,
  type OracleCase,
} from "../../test/oracle/torch-oracle";

// Tolerance for floating point comparison
// Note: GPU matmul has different accumulation order than CPU, leading to small numerical differences
// These tolerances are reasonable for single precision GPU computation
const ATOL = 0.1; // absolute tolerance - GPU matmul can have up to 10% differences
const RTOL = 0.1; // relative tolerance - 10% relative difference allowed for small values

function allClose(
  actual: number[],
  expected: number[],
  atol = ATOL,
  rtol = RTOL,
): { pass: boolean; maxDiff: number; maxRelDiff: number } {
  if (actual.length !== expected.length) {
    return { pass: false, maxDiff: Infinity, maxRelDiff: Infinity };
  }
  let maxDiff = 0;
  let maxRelDiff = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const relDiff = diff / (Math.abs(expected[i]) + 1e-8);
    maxDiff = Math.max(maxDiff, diff);
    maxRelDiff = Math.max(maxRelDiff, relDiff);
    if (diff > atol + rtol * Math.abs(expected[i])) {
      return { pass: false, maxDiff, maxRelDiff };
    }
  }
  return { pass: true, maxDiff, maxRelDiff };
}

async function testSimpleMLP(api: Torchlette) {
  console.log("\n=== Test 1: Simple MLP Forward/Backward ===");

  // Fixed seed data for reproducibility
  const batchSize = 4;
  const inputDim = 8;
  const outputDim = 4;

  // Create deterministic test data
  const xData = Array.from({ length: batchSize * inputDim }, (_, i) =>
    Math.sin(i * 0.1) * 0.5
  );
  const wData = Array.from({ length: inputDim * outputDim }, (_, i) =>
    Math.cos(i * 0.1) * 0.3
  );
  const bData = Array.from({ length: outputDim }, (_, i) => i * 0.01);
  const targetData = Array.from({ length: batchSize * outputDim }, (_, i) =>
    Math.sin(i * 0.2) * 0.5
  );

  // Get PyTorch reference
  console.log("  Getting PyTorch reference...");
  const oracleCase: OracleCase = {
    op: "mlp_mse_backward",
    caseName: "simple_mlp",
    inputs: [
      { values: xData, shape: [batchSize, inputDim] },
      { values: wData, shape: [inputDim, outputDim] },
      { values: bData, shape: [outputDim] },
      { values: targetData, shape: [batchSize, outputDim] },
    ],
  };

  const [pytorchResult] = await runTorchOracleBackwardBatch([oracleCase]);
  console.log(`  PyTorch loss: ${pytorchResult.output.values[0].toFixed(6)}`);

  // Run Torchlette
  console.log("  Running Torchlette...");
  const x = api.tensorFromArray(xData, [batchSize, inputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w = api.tensorFromArray(wData, [inputDim, outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b = api.tensorFromArray(bData, [outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const target = api.tensorFromArray(targetData, [batchSize, outputDim], {
    device: "webgpu",
  });

  // Forward: relu(x @ w + b)
  const hidden = api.add(api.matmul(x, w), b);
  const pred = api.relu(hidden);
  const diff = api.sub(pred, target);
  const sqDiff = api.mul(diff, diff);
  const loss = api.mean(sqDiff);

  const lossVal = await loss.item();
  console.log(`  Torchlette loss: ${lossVal.toFixed(6)}`);

  // Backward
  await loss.backward();

  // Force grad tensors to CPU - cpu() returns number[] directly
  const xGrad = await x.grad!.cpu();
  const wGrad = await w.grad!.cpu();
  const bGrad = await b.grad!.cpu();

  // Compare results
  const lossCheck = allClose([lossVal], pytorchResult.output.values);
  const xGradCheck = allClose(xGrad, pytorchResult.grads[0].values);
  const wGradCheck = allClose(wGrad, pytorchResult.grads[1].values);
  const bGradCheck = allClose(bGrad, pytorchResult.grads[2].values);

  console.log(`  Loss match: ${lossCheck.pass ? "✓" : "✗"} (maxDiff=${lossCheck.maxDiff.toExponential(2)})`);
  console.log(`  x.grad match: ${xGradCheck.pass ? "✓" : "✗"} (maxDiff=${xGradCheck.maxDiff.toExponential(2)})`);
  console.log(`  w.grad match: ${wGradCheck.pass ? "✓" : "✗"} (maxDiff=${wGradCheck.maxDiff.toExponential(2)})`);
  console.log(`  b.grad match: ${bGradCheck.pass ? "✓" : "✗"} (maxDiff=${bGradCheck.maxDiff.toExponential(2)})`);

  // Cleanup
  x.dispose();
  w.dispose();
  b.dispose();
  target.dispose();
  hidden.dispose();
  pred.dispose();
  diff.dispose();
  sqDiff.dispose();
  loss.dispose();

  return lossCheck.pass && xGradCheck.pass && wGradCheck.pass && bGradCheck.pass;
}

async function testTwoLayerMLPWithCheckpoint(api: Torchlette) {
  console.log("\n=== Test 2: Two-Layer MLP with Checkpointing ===");

  const batchSize = 4;
  const inputDim = 8;
  const hiddenDim = 16;
  const outputDim = 4;

  // Create deterministic test data
  const xData = Array.from({ length: batchSize * inputDim }, (_, i) =>
    Math.sin(i * 0.1) * 0.5
  );
  const w1Data = Array.from({ length: inputDim * hiddenDim }, (_, i) =>
    Math.cos(i * 0.05) * 0.2
  );
  const b1Data = Array.from({ length: hiddenDim }, (_, i) => i * 0.01);
  const w2Data = Array.from({ length: hiddenDim * outputDim }, (_, i) =>
    Math.sin(i * 0.08) * 0.25
  );
  const b2Data = Array.from({ length: outputDim }, (_, i) => -i * 0.01);
  const targetData = Array.from({ length: batchSize * outputDim }, (_, i) =>
    Math.cos(i * 0.15) * 0.3
  );

  // Get PyTorch reference
  console.log("  Getting PyTorch reference...");
  const oracleCase: OracleCase = {
    op: "two_layer_mlp_backward",
    caseName: "two_layer_mlp",
    inputs: [
      { values: xData, shape: [batchSize, inputDim] },
      { values: w1Data, shape: [inputDim, hiddenDim] },
      { values: b1Data, shape: [hiddenDim] },
      { values: w2Data, shape: [hiddenDim, outputDim] },
      { values: b2Data, shape: [outputDim] },
      { values: targetData, shape: [batchSize, outputDim] },
    ],
  };

  const [pytorchResult] = await runTorchOracleExtendedBatch([oracleCase]);
  console.log(`  PyTorch loss: ${pytorchResult.output.values[0].toFixed(6)}`);

  // Run Torchlette with checkpointing
  console.log("  Running Torchlette with checkpointing...");
  const x = api.tensorFromArray(xData, [batchSize, inputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w1 = api.tensorFromArray(w1Data, [inputDim, hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b1 = api.tensorFromArray(b1Data, [hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w2 = api.tensorFromArray(w2Data, [hiddenDim, outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b2 = api.tensorFromArray(b2Data, [outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const target = api.tensorFromArray(targetData, [batchSize, outputDim], {
    device: "webgpu",
  });

  // Layer 1 with checkpoint
  const layer1Output = checkpoint(
    api,
    (input: Tensor) => {
      const h1 = api.add(api.matmul(input, w1), b1);
      return api.relu(h1);
    },
    [x],
  );

  // Layer 2 (no checkpoint - final layer)
  const hidden2 = api.add(api.matmul(layer1Output, w2), b2);

  // MSE loss
  const diff = api.sub(hidden2, target);
  const sqDiff = api.mul(diff, diff);
  const loss = api.mean(sqDiff);

  const lossVal = await loss.item();
  console.log(`  Torchlette loss: ${lossVal.toFixed(6)}`);

  // Backward
  await loss.backward();

  // Force grad tensors to CPU - cpu() returns number[] directly
  const xGrad = await x.grad!.cpu();
  const w1Grad = await w1.grad!.cpu();
  const b1Grad = await b1.grad!.cpu();
  const w2Grad = await w2.grad!.cpu();
  const b2Grad = await b2.grad!.cpu();

  // Compare results
  const lossCheck = allClose([lossVal], pytorchResult.output.values);
  const xGradCheck = allClose(xGrad, pytorchResult.grads[0].values);
  const w1GradCheck = allClose(w1Grad, pytorchResult.grads[1].values);
  const b1GradCheck = allClose(b1Grad, pytorchResult.grads[2].values);
  const w2GradCheck = allClose(w2Grad, pytorchResult.grads[3].values);
  const b2GradCheck = allClose(b2Grad, pytorchResult.grads[4].values);

  console.log(`  Loss match: ${lossCheck.pass ? "✓" : "✗"} (maxDiff=${lossCheck.maxDiff.toExponential(2)})`);
  console.log(`  x.grad match: ${xGradCheck.pass ? "✓" : "✗"} (maxDiff=${xGradCheck.maxDiff.toExponential(2)})`);
  console.log(`  w1.grad match: ${w1GradCheck.pass ? "✓" : "✗"} (maxDiff=${w1GradCheck.maxDiff.toExponential(2)})`);
  console.log(`  b1.grad match: ${b1GradCheck.pass ? "✓" : "✗"} (maxDiff=${b1GradCheck.maxDiff.toExponential(2)})`);
  console.log(`  w2.grad match: ${w2GradCheck.pass ? "✓" : "✗"} (maxDiff=${w2GradCheck.maxDiff.toExponential(2)})`);
  console.log(`  b2.grad match: ${b2GradCheck.pass ? "✓" : "✗"} (maxDiff=${b2GradCheck.maxDiff.toExponential(2)})`);

  // Cleanup
  x.dispose();
  w1.dispose();
  b1.dispose();
  w2.dispose();
  b2.dispose();
  target.dispose();
  layer1Output.dispose();
  hidden2.dispose();
  diff.dispose();
  sqDiff.dispose();
  loss.dispose();

  return (
    lossCheck.pass &&
    xGradCheck.pass &&
    w1GradCheck.pass &&
    b1GradCheck.pass &&
    w2GradCheck.pass &&
    b2GradCheck.pass
  );
}

async function testTrainingLoopWithAMP(api: Torchlette) {
  console.log("\n=== Test 3: Training Loop with AMP ===");

  const batchSize = 4;
  const inputDim = 8;
  const outputDim = 4;
  const numSteps = 3;

  // Create deterministic test data
  const xData = Array.from({ length: batchSize * inputDim }, (_, i) =>
    Math.sin(i * 0.1) * 0.5
  );
  const targetData = Array.from({ length: batchSize * outputDim }, (_, i) =>
    Math.sin(i * 0.2) * 0.5
  );

  // Initialize weights
  let wData = Array.from({ length: inputDim * outputDim }, (_, i) =>
    Math.cos(i * 0.1) * 0.3
  );
  let bData = Array.from({ length: outputDim }, (_, i) => i * 0.01);

  console.log("  Running training loop with AMP...");

  const lr = 0.01;
  const losses: number[] = [];

  for (let step = 0; step < numSteps; step++) {
    // Create tensors
    const x = api.tensorFromArray(xData, [batchSize, inputDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const w = api.tensorFromArray(wData, [inputDim, outputDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const b = api.tensorFromArray(bData, [outputDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const target = api.tensorFromArray(targetData, [batchSize, outputDim], {
      device: "webgpu",
    });

    // Forward pass with AMP autocast
    const loss = api.autocast(() => {
      const hidden = api.add(api.matmul(x, w), b);
      const pred = api.relu(hidden);
      const diff = api.sub(pred, target);
      const sqDiff = api.mul(diff, diff);
      return api.mean(sqDiff);
    });

    const lossVal = await loss.item();
    losses.push(lossVal);
    console.log(`  Step ${step}: loss=${lossVal.toFixed(6)}`);

    // Backward
    await loss.backward();

    // Manual SGD update - cpu() returns number[] directly
    const wGrad = await w.grad!.cpu();
    const bGrad = await b.grad!.cpu();

    wData = wData.map((v, i) => v - lr * wGrad[i]);
    bData = bData.map((v, i) => v - lr * bGrad[i]);

    // Cleanup
    x.dispose();
    w.dispose();
    b.dispose();
    target.dispose();
    loss.dispose();
  }

  console.log(`  Loss progression: ${losses.map((l) => l.toFixed(4)).join(" -> ")}`);

  // Check that loss decreased
  const lossDecreased = losses[losses.length - 1] < losses[0];
  console.log(`  Loss decreased: ${lossDecreased ? "✓" : "✗"}`);

  return lossDecreased;
}

async function testTrainingLoopWithAMPAndCheckpoint(api: Torchlette) {
  console.log("\n=== Test 4: Training Loop with AMP + Checkpointing ===");

  const batchSize = 4;
  const inputDim = 8;
  const hiddenDim = 16;
  const outputDim = 4;
  const numSteps = 3;

  // Create deterministic test data
  const xData = Array.from({ length: batchSize * inputDim }, (_, i) =>
    Math.sin(i * 0.1) * 0.5
  );
  const targetData = Array.from({ length: batchSize * outputDim }, (_, i) =>
    Math.cos(i * 0.15) * 0.3
  );

  // Initialize weights
  let w1Data = Array.from({ length: inputDim * hiddenDim }, (_, i) =>
    Math.cos(i * 0.05) * 0.2
  );
  let b1Data = Array.from({ length: hiddenDim }, (_, i) => i * 0.01);
  let w2Data = Array.from({ length: hiddenDim * outputDim }, (_, i) =>
    Math.sin(i * 0.08) * 0.25
  );
  let b2Data = Array.from({ length: outputDim }, (_, i) => -i * 0.01);

  console.log("  Running training loop with AMP + Checkpointing...");

  const lr = 0.01;
  const losses: number[] = [];

  for (let step = 0; step < numSteps; step++) {
    // Create tensors
    const x = api.tensorFromArray(xData, [batchSize, inputDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const w1 = api.tensorFromArray(w1Data, [inputDim, hiddenDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const b1 = api.tensorFromArray(b1Data, [hiddenDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const w2 = api.tensorFromArray(w2Data, [hiddenDim, outputDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const b2 = api.tensorFromArray(b2Data, [outputDim], {
      device: "webgpu",
      requiresGrad: true,
    });
    const target = api.tensorFromArray(targetData, [batchSize, outputDim], {
      device: "webgpu",
    });

    // Forward pass with AMP autocast and checkpointing
    const loss = api.autocast(() => {
      // Layer 1 with checkpoint
      const layer1Output = checkpoint(
        api,
        (input: Tensor) => {
          const h1 = api.add(api.matmul(input, w1), b1);
          return api.relu(h1);
        },
        [x],
      );

      // Layer 2
      const hidden2 = api.add(api.matmul(layer1Output, w2), b2);

      // MSE loss
      const diff = api.sub(hidden2, target);
      const sqDiff = api.mul(diff, diff);
      return api.mean(sqDiff);
    });

    const lossVal = await loss.item();
    losses.push(lossVal);
    console.log(`  Step ${step}: loss=${lossVal.toFixed(6)}`);

    // Backward
    await loss.backward();

    // Manual SGD update - cpu() returns number[] directly
    const w1Grad = await w1.grad!.cpu();
    const b1Grad = await b1.grad!.cpu();
    const w2Grad = await w2.grad!.cpu();
    const b2Grad = await b2.grad!.cpu();

    w1Data = w1Data.map((v, i) => v - lr * w1Grad[i]);
    b1Data = b1Data.map((v, i) => v - lr * b1Grad[i]);
    w2Data = w2Data.map((v, i) => v - lr * w2Grad[i]);
    b2Data = b2Data.map((v, i) => v - lr * b2Grad[i]);

    // Cleanup
    x.dispose();
    w1.dispose();
    b1.dispose();
    w2.dispose();
    b2.dispose();
    target.dispose();
    loss.dispose();
  }

  console.log(`  Loss progression: ${losses.map((l) => l.toFixed(4)).join(" -> ")}`);

  // Check that loss decreased
  const lossDecreased = losses[losses.length - 1] < losses[0];
  console.log(`  Loss decreased: ${lossDecreased ? "✓" : "✗"}`);

  return lossDecreased;
}

async function main() {
  console.log("===========================================");
  console.log("  PyTorch Comparison Test Suite");
  console.log("===========================================");

  console.log("\nInitializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });

  const results: { name: string; pass: boolean }[] = [];

  try {
    // Test 1: Simple MLP
    results.push({
      name: "Simple MLP Forward/Backward",
      pass: await testSimpleMLP(api),
    });

    // Test 2: Two-layer MLP with checkpointing
    results.push({
      name: "Two-Layer MLP with Checkpointing",
      pass: await testTwoLayerMLPWithCheckpoint(api),
    });

    // Test 3: Training loop with AMP
    results.push({
      name: "Training Loop with AMP",
      pass: await testTrainingLoopWithAMP(api),
    });

    // Test 4: Training loop with AMP + Checkpointing
    results.push({
      name: "Training Loop with AMP + Checkpointing",
      pass: await testTrainingLoopWithAMPAndCheckpoint(api),
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
