/**
 * Simple checkpoint equivalence test.
 * Verifies that checkpoint produces identical results to non-checkpoint.
 */
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Linear } from "../../src/nn/linear";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Checkpoint Equivalence Test ===\n");

  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Fixed weights for reproducibility
  const inputDim = 32;
  const hiddenDim = 64;
  const outputDim = 16;
  const batchSize = 4;
  const seqLen = 8;

  // Create deterministic weights
  const w1Data = Array.from({ length: inputDim * hiddenDim }, (_, i) =>
    Math.sin(i * 0.1) * 0.1
  );
  const b1Data = Array.from({ length: hiddenDim }, (_, i) => i * 0.001);
  const w2Data = Array.from({ length: hiddenDim * outputDim }, (_, i) =>
    Math.cos(i * 0.1) * 0.1
  );
  const b2Data = Array.from({ length: outputDim }, (_, i) => -i * 0.001);

  // Create input
  const inputData = Array.from(
    { length: batchSize * seqLen * inputDim },
    (_, i) => Math.sin(i * 0.05) * 0.5
  );
  const targetData = Array.from(
    { length: batchSize * seqLen * outputDim },
    (_, i) => Math.cos(i * 0.03) * 0.3
  );

  // ========== Run WITHOUT checkpoint ==========
  console.log("Running WITHOUT checkpoint...");

  const w1_nc = api.tensorFromArray(w1Data, [inputDim, hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b1_nc = api.tensorFromArray(b1Data, [hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w2_nc = api.tensorFromArray(w2Data, [hiddenDim, outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b2_nc = api.tensorFromArray(b2Data, [outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });

  const input_nc = api.tensorFromArray(
    inputData,
    [batchSize * seqLen, inputDim],
    { device: "webgpu", requiresGrad: true }
  );
  const target_nc = api.tensorFromArray(
    targetData,
    [batchSize * seqLen, outputDim],
    { device: "webgpu" }
  );

  // Forward: x -> W1 + b1 -> relu -> W2 + b2
  let h1_nc = api.add(api.matmul(input_nc, w1_nc), b1_nc);
  h1_nc = api.relu(h1_nc);
  const out_nc = api.add(api.matmul(h1_nc, w2_nc), b2_nc);

  // MSE loss
  const diff_nc = api.sub(out_nc, target_nc);
  const sq_nc = api.mul(diff_nc, diff_nc);
  const loss_nc = api.mean(sq_nc);

  const lossVal_nc = await loss_nc.item();
  console.log(`  Loss (no checkpoint): ${lossVal_nc.toFixed(6)}`);

  await loss_nc.backward();
  await api.markStep();

  const w1Grad_nc = await w1_nc.grad!.cpu();
  const b1Grad_nc = await b1_nc.grad!.cpu();
  const w2Grad_nc = await w2_nc.grad!.cpu();
  const b2Grad_nc = await b2_nc.grad!.cpu();
  const inputGrad_nc = await input_nc.grad!.cpu();

  // ========== Run WITH checkpoint ==========
  console.log("\nRunning WITH checkpoint...");

  const w1_cp = api.tensorFromArray(w1Data, [inputDim, hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b1_cp = api.tensorFromArray(b1Data, [hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w2_cp = api.tensorFromArray(w2Data, [hiddenDim, outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b2_cp = api.tensorFromArray(b2Data, [outputDim], {
    device: "webgpu",
    requiresGrad: true,
  });

  const input_cp = api.tensorFromArray(
    inputData,
    [batchSize * seqLen, inputDim],
    { device: "webgpu", requiresGrad: true }
  );
  const target_cp = api.tensorFromArray(
    targetData,
    [batchSize * seqLen, outputDim],
    { device: "webgpu" }
  );

  // Forward WITH checkpoint on layer 1
  const h1_cp = checkpoint(
    api,
    (x: Tensor) => {
      const h = api.add(api.matmul(x, w1_cp), b1_cp);
      return api.relu(h);
    },
    [input_cp]
  );
  const out_cp = api.add(api.matmul(h1_cp, w2_cp), b2_cp);

  // MSE loss
  const diff_cp = api.sub(out_cp, target_cp);
  const sq_cp = api.mul(diff_cp, diff_cp);
  const loss_cp = api.mean(sq_cp);

  const lossVal_cp = await loss_cp.item();
  console.log(`  Loss (checkpoint): ${lossVal_cp.toFixed(6)}`);

  await loss_cp.backward();
  await api.markStep();

  const w1Grad_cp = await w1_cp.grad!.cpu();
  const b1Grad_cp = await b1_cp.grad!.cpu();
  const w2Grad_cp = await w2_cp.grad!.cpu();
  const b2Grad_cp = await b2_cp.grad!.cpu();
  const inputGrad_cp = await input_cp.grad!.cpu();

  // ========== Compare ==========
  console.log("\n=== Comparison ===");

  const lossDiff = Math.abs(lossVal_nc - lossVal_cp);
  console.log(`Loss diff: ${lossDiff.toExponential(2)} ${lossDiff < 1e-5 ? "✓" : "✗"}`);

  function maxDiff(a: number[] | Float32Array, b: number[] | Float32Array): number {
    let max = 0;
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]));
    }
    return max;
  }

  const w1GradDiff = maxDiff(w1Grad_nc, w1Grad_cp);
  const b1GradDiff = maxDiff(b1Grad_nc, b1Grad_cp);
  const w2GradDiff = maxDiff(w2Grad_nc, w2Grad_cp);
  const b2GradDiff = maxDiff(b2Grad_nc, b2Grad_cp);
  const inputGradDiff = maxDiff(inputGrad_nc, inputGrad_cp);

  console.log(`w1.grad diff: ${w1GradDiff.toExponential(2)} ${w1GradDiff < 1e-5 ? "✓" : "✗"}`);
  console.log(`b1.grad diff: ${b1GradDiff.toExponential(2)} ${b1GradDiff < 1e-5 ? "✓" : "✗"}`);
  console.log(`w2.grad diff: ${w2GradDiff.toExponential(2)} ${w2GradDiff < 1e-5 ? "✓" : "✗"}`);
  console.log(`b2.grad diff: ${b2GradDiff.toExponential(2)} ${b2GradDiff < 1e-5 ? "✓" : "✗"}`);
  console.log(`input.grad diff: ${inputGradDiff.toExponential(2)} ${inputGradDiff < 1e-5 ? "✓" : "✗"}`);

  // Debug: show first few values
  if (w1GradDiff > 1e-5) {
    console.log("\nw1.grad first 5:");
    console.log(`  No checkpoint: [${Array.from(w1Grad_nc).slice(0, 5).map(v => v.toFixed(6)).join(", ")}]`);
    console.log(`  Checkpoint:    [${Array.from(w1Grad_cp).slice(0, 5).map(v => v.toFixed(6)).join(", ")}]`);
  }

  const allPass =
    lossDiff < 1e-5 &&
    w1GradDiff < 1e-5 &&
    b1GradDiff < 1e-5 &&
    w2GradDiff < 1e-5 &&
    b2GradDiff < 1e-5 &&
    inputGradDiff < 1e-5;

  console.log(`\nOverall: ${allPass ? "PASS" : "FAIL"}`);
  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
