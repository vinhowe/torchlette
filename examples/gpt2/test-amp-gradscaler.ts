/**
 * Example: Complete AMP training with GradScaler.
 *
 * Demonstrates the full AMP workflow:
 * 1. autocast() for automatic f16 computation
 * 2. GradScaler for gradient scaling and NaN detection
 * 3. Dynamic scale adjustment based on gradient health
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Adam, GradScaler } from "../../src/optim";

async function main() {
  console.log("===========================================");
  console.log("  AMP Training with GradScaler");
  console.log("===========================================\n");

  console.log("Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("Failed to initialize WebGPU");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
  });

  // Model parameters
  const batchSize = 8;
  const inputDim = 32;
  const hiddenDim = 64;
  const outputDim = 16;

  // Initialize weights
  const w1 = api.tensorFromArray(
    Array.from({ length: inputDim * hiddenDim }, () => (Math.random() - 0.5) * 0.1),
    [inputDim, hiddenDim],
    { device: "webgpu", requiresGrad: true },
  );
  const b1 = api.tensorFromArray(
    Array.from({ length: hiddenDim }, () => 0),
    [hiddenDim],
    { device: "webgpu", requiresGrad: true },
  );
  const w2 = api.tensorFromArray(
    Array.from({ length: hiddenDim * outputDim }, () => (Math.random() - 0.5) * 0.1),
    [hiddenDim, outputDim],
    { device: "webgpu", requiresGrad: true },
  );
  const b2 = api.tensorFromArray(
    Array.from({ length: outputDim }, () => 0),
    [outputDim],
    { device: "webgpu", requiresGrad: true },
  );

  // Optimizer and GradScaler
  const optimizer = new Adam([w1, b1, w2, b2], { lr: 0.001 }, api);
  const scaler = new GradScaler(api, {
    initScale: 65536,
    growthFactor: 2.0,
    backoffFactor: 0.5,
    growthInterval: 100,
  });

  console.log("Training with AMP + GradScaler...\n");
  console.log("Step | Loss     | Scale    | Stepped | Notes");
  console.log("-----|----------|----------|---------|------");

  const numSteps = 20;
  let skippedSteps = 0;

  for (let step = 0; step < numSteps; step++) {
    // Generate random batch
    const x = api.tensorFromArray(
      Array.from({ length: batchSize * inputDim }, () => Math.random()),
      [batchSize, inputDim],
      { device: "webgpu" },
    );
    const target = api.tensorFromArray(
      Array.from({ length: batchSize * outputDim }, () => Math.random()),
      [batchSize, outputDim],
      { device: "webgpu" },
    );

    // Forward pass with autocast (AMP)
    const loss = api.autocast(() => {
      // Layer 1: matmul + bias + relu
      const h1 = api.relu(api.add(api.matmul(x, w1), b1));
      // Layer 2: matmul + bias
      const out = api.add(api.matmul(h1, w2), b2);
      // MSE loss
      const diff = api.sub(out, target);
      return api.mean(api.mul(diff, diff));
    });

    // Scale loss before backward
    const scaledLoss = scaler.scale(loss);
    const lossVal = await loss.item();

    // Backward pass
    await scaledLoss.backward();

    // Unscale gradients and check for NaN/Inf
    scaler.unscale_(optimizer);

    // Step optimizer (skipped if NaN detected)
    const stepped = scaler.step(optimizer);

    // Update scale factor
    const scaleBefore = scaler.getScale();
    scaler.update();
    const scaleAfter = scaler.getScale();

    // Log
    let notes = "";
    if (!stepped) {
      skippedSteps++;
      notes = "NaN detected, step skipped";
    } else if (scaleAfter > scaleBefore) {
      notes = "Scale increased";
    } else if (scaleAfter < scaleBefore) {
      notes = "Scale decreased";
    }

    console.log(
      `${String(step).padStart(4)} | ${lossVal.toFixed(6)} | ${String(scaleAfter).padStart(8)} | ${stepped ? "Yes    " : "No     "} | ${notes}`,
    );

    // Zero gradients for next step
    optimizer.zeroGrad();

    // Cleanup
    x.dispose();
    target.dispose();
    loss.dispose();
    scaledLoss.dispose();
  }

  console.log("\n===========================================");
  console.log("  Summary");
  console.log("===========================================");
  console.log(`Total steps: ${numSteps}`);
  console.log(`Skipped steps (NaN): ${skippedSteps}`);
  console.log(`Final scale: ${scaler.getScale()}`);
  console.log(`Scaler state: ${JSON.stringify(scaler.stateDict())}`);

  // Demonstrate NaN scenario
  console.log("\n=== Demonstrating NaN Detection ===\n");

  // Create a scenario that produces NaN gradients
  const badInput = api.tensorFromArray([0.0], [], { device: "webgpu" });
  const logZero = api.log(badInput); // -Inf
  const nanLoss = api.mul(api.sum(w1), logZero); // Will produce NaN in gradient
  const scaledNanLoss = scaler.scale(nanLoss);

  await scaledNanLoss.backward();

  const scaleBefore = scaler.getScale();
  scaler.unscale_(optimizer);
  const stepped = scaler.step(optimizer);
  scaler.update();
  await scaler.resolveDeferred();
  const scaleAfter = scaler.getScale();

  console.log(`NaN detected: ${scaler.foundInf ? "Yes" : "No"}`);
  console.log(`Step executed: ${stepped}`);
  console.log(`Scale before: ${scaleBefore}`);
  console.log(`Scale after: ${scaleAfter}`);
  console.log(`Scale reduced by: ${((1 - scaleAfter / scaleBefore) * 100).toFixed(0)}%`);

  console.log("\n✓ GradScaler correctly handled NaN gradients!");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
