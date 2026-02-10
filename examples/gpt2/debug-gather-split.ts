/**
 * Debug Gather Split - Test gradient flow through gather-based tensor split
 *
 * The QKV split in attention uses:
 *   qkvFlattened [3, N] where N = batch * seq * embed
 *   idx0 = [0] expanded to [1, N]
 *   qFlat = qkvFlattened.gather(idx0, {dim: 0})  -> [1, N]
 *
 * The backward should scatter gradients back to the correct rows.
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { runTorchOracleBackwardBatch } from "../../test/oracle/torch-oracle";

function maxDiff(a: number[], b: number[]): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function status(diff: number, threshold: number = 1e-4): string {
  return diff < threshold ? "PASS" : "FAIL";
}

async function main() {
  console.log("=== Gather Split Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const N = 8; // flatTotal

  // Create source tensor [3, N]
  const srcData = new Array(3 * N).fill(0).map((_, i) => i * 0.1);

  console.log("Test 1: Single gather (row 0)");
  {
    const src = api.tensorFromArray(srcData, [3, N], { requiresGrad: true, device: "webgpu" });

    // Gather row 0
    const idx0 = api.tensorFromArray(new Array(N).fill(0), [1, N], { device: "webgpu" });
    const row0 = src.gather(idx0, { dim: 0 });

    const loss = api.sum(row0);
    await loss.backward();

    const grad = Array.from(await src.grad!.cpu());
    console.log(`  grad[0:N]  (row 0): ${grad.slice(0, N).map(v => v.toFixed(1)).join(", ")}`);
    console.log(`  grad[N:2N] (row 1): ${grad.slice(N, 2 * N).map(v => v.toFixed(1)).join(", ")}`);
    console.log(`  grad[2N:]  (row 2): ${grad.slice(2 * N).map(v => v.toFixed(1)).join(", ")}`);

    // Expected: row 0 = all 1s, rows 1,2 = all 0s
    const expectedRow0 = new Array(N).fill(1);
    const expectedRow1 = new Array(N).fill(0);
    const diff0 = maxDiff(grad.slice(0, N), expectedRow0);
    const diff1 = maxDiff(grad.slice(N, 2 * N), expectedRow1);
    console.log(`  Row 0 ${status(diff0)}: diff=${diff0.toExponential(2)}`);
    console.log(`  Row 1 ${status(diff1)}: diff=${diff1.toExponential(2)}`);
  }

  console.log("\nTest 2: Multiple gathers (rows 0, 1, 2)");
  {
    const src = api.tensorFromArray(srcData, [3, N], { requiresGrad: true, device: "webgpu" });

    // Gather all three rows separately
    const idx0 = api.tensorFromArray(new Array(N).fill(0), [1, N], { device: "webgpu" });
    const idx1 = api.tensorFromArray(new Array(N).fill(1), [1, N], { device: "webgpu" });
    const idx2 = api.tensorFromArray(new Array(N).fill(2), [1, N], { device: "webgpu" });

    const row0 = src.gather(idx0, { dim: 0 });
    const row1 = src.gather(idx1, { dim: 0 });
    const row2 = src.gather(idx2, { dim: 0 });

    // Use all three in loss
    const loss = api.add(api.add(api.sum(row0), api.sum(row1)), api.sum(row2));
    await loss.backward();

    const grad = Array.from(await src.grad!.cpu());
    console.log(`  grad[0:N]  (row 0): ${grad.slice(0, N).map(v => v.toFixed(1)).join(", ")}`);
    console.log(`  grad[N:2N] (row 1): ${grad.slice(N, 2 * N).map(v => v.toFixed(1)).join(", ")}`);
    console.log(`  grad[2N:]  (row 2): ${grad.slice(2 * N).map(v => v.toFixed(1)).join(", ")}`);

    // Expected: all rows = all 1s (each row contributes to loss once)
    const expected = new Array(N).fill(1);
    const diff0 = maxDiff(grad.slice(0, N), expected);
    const diff1 = maxDiff(grad.slice(N, 2 * N), expected);
    const diff2 = maxDiff(grad.slice(2 * N), expected);
    console.log(`  Row 0 ${status(diff0)}: diff=${diff0.toExponential(2)}`);
    console.log(`  Row 1 ${status(diff1)}: diff=${diff1.toExponential(2)}`);
    console.log(`  Row 2 ${status(diff2)}: diff=${diff2.toExponential(2)}`);
  }

  console.log("\nTest 3: Transform then use gathers");
  {
    const src = api.tensorFromArray(srcData, [3, N], { requiresGrad: true, device: "webgpu" });

    // Gather all three rows separately
    const idx0 = api.tensorFromArray(new Array(N).fill(0), [1, N], { device: "webgpu" });
    const idx1 = api.tensorFromArray(new Array(N).fill(1), [1, N], { device: "webgpu" });
    const idx2 = api.tensorFromArray(new Array(N).fill(2), [1, N], { device: "webgpu" });

    const row0 = src.gather(idx0, { dim: 0 }).reshape([N]);
    const row1 = src.gather(idx1, { dim: 0 }).reshape([N]);
    const row2 = src.gather(idx2, { dim: 0 }).reshape([N]);

    // Multiply rows together then sum
    const product = api.mul(api.mul(row0, row1), row2);
    const loss = api.sum(product);
    await loss.backward();

    const grad = Array.from(await src.grad!.cpu());
    console.log(`  grad[0:N]  (row 0): ${grad.slice(0, N).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  grad[N:2N] (row 1): ${grad.slice(N, 2 * N).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  grad[2N:]  (row 2): ${grad.slice(2 * N).map(v => v.toFixed(4)).join(", ")}`);

    // For loss = sum(row0 * row1 * row2):
    // d/d(row0[i]) = row1[i] * row2[i]
    // d/d(row1[i]) = row0[i] * row2[i]
    // d/d(row2[i]) = row0[i] * row1[i]
    const row0Data = srcData.slice(0, N);
    const row1Data = srcData.slice(N, 2 * N);
    const row2Data = srcData.slice(2 * N);

    const expectedGrad0 = row1Data.map((v, i) => v * row2Data[i]);
    const expectedGrad1 = row0Data.map((v, i) => v * row2Data[i]);
    const expectedGrad2 = row0Data.map((v, i) => v * row1Data[i]);

    console.log(`  expected row 0: ${expectedGrad0.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  expected row 1: ${expectedGrad1.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  expected row 2: ${expectedGrad2.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    const diff0 = maxDiff(grad.slice(0, N), expectedGrad0);
    const diff1 = maxDiff(grad.slice(N, 2 * N), expectedGrad1);
    const diff2 = maxDiff(grad.slice(2 * N), expectedGrad2);
    console.log(`  Row 0 ${status(diff0)}: diff=${diff0.toExponential(2)}`);
    console.log(`  Row 1 ${status(diff1)}: diff=${diff1.toExponential(2)}`);
    console.log(`  Row 2 ${status(diff2)}: diff=${diff2.toExponential(2)}`);
  }

  console.log("\nTest 4: Gather with expanded indices (attention pattern)");
  {
    // This matches the attention QKV split pattern
    const src = api.tensorFromArray(srcData, [3, N], { requiresGrad: true, device: "webgpu" });

    // Create expanded indices like in attention:
    // idx0 = [0].reshape([1,1]).expand([1, N])
    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, N]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, N]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, N]).contiguous();

    console.log(`  idx0 shape: [${idx0.shape.join(", ")}]`);
    console.log(`  idx0 first 4: ${Array.from(await idx0.cpu()).slice(0, 4).join(", ")}`);

    const row0 = src.gather(idx0, { dim: 0 }).reshape([N]);
    const row1 = src.gather(idx1, { dim: 0 }).reshape([N]);
    const row2 = src.gather(idx2, { dim: 0 }).reshape([N]);

    // Multiply rows together then sum
    const product = api.mul(api.mul(row0, row1), row2);
    const loss = api.sum(product);
    await loss.backward();

    const grad = Array.from(await src.grad!.cpu());
    console.log(`  grad[0:N]  (row 0): ${grad.slice(0, N).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  grad[N:2N] (row 1): ${grad.slice(N, 2 * N).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  grad[2N:]  (row 2): ${grad.slice(2 * N).map(v => v.toFixed(4)).join(", ")}`);

    const row0Data = srcData.slice(0, N);
    const row1Data = srcData.slice(N, 2 * N);
    const row2Data = srcData.slice(2 * N);

    const expectedGrad0 = row1Data.map((v, i) => v * row2Data[i]);
    const expectedGrad1 = row0Data.map((v, i) => v * row2Data[i]);
    const expectedGrad2 = row0Data.map((v, i) => v * row1Data[i]);

    const diff0 = maxDiff(grad.slice(0, N), expectedGrad0);
    const diff1 = maxDiff(grad.slice(N, 2 * N), expectedGrad1);
    const diff2 = maxDiff(grad.slice(2 * N), expectedGrad2);
    console.log(`  Row 0 ${status(diff0)}: diff=${diff0.toExponential(2)}`);
    console.log(`  Row 1 ${status(diff1)}: diff=${diff1.toExponential(2)}`);
    console.log(`  Row 2 ${status(diff2)}: diff=${diff2.toExponential(2)}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
