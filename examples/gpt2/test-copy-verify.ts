/**
 * Verify copy_ actually copies data.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";

async function main() {
  console.log("=== Copy Verify Test ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Create two tensors with different values
  const t1 = api.tensorFromArray([1, 2, 3, 4], [4], { device: "webgpu", requiresGrad: true });
  const t2 = api.tensorFromArray([10, 20, 30, 40], [4], { device: "webgpu", requiresGrad: true });

  console.log("Before copy:");
  console.log(`  t1: ${await t1.cpu()}`);
  console.log(`  t2: ${await t2.cpu()}`);

  // Copy t1 to t2
  console.log("\nCopying t1 to t2...");
  t2.copy_(t1);
  await api.markStep();

  console.log("\nAfter copy:");
  console.log(`  t1: ${await t1.cpu()}`);
  console.log(`  t2: ${await t2.cpu()}`);

  const t1Data = await t1.cpu();
  const t2Data = await t2.cpu();
  const match = t1Data.every((v, i) => v === t2Data[i]);
  console.log(`\n  Match: ${match ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(match ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
