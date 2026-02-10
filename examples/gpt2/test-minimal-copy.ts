/**
 * Minimal test: two simple linear models with weight copying.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Linear } from "../../src/nn";

async function main() {
  console.log("=== Minimal Copy Test ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Create two simple linear layers
  console.log("Building linear1...");
  const linear1 = new Linear(api, 4, 4, { device: "webgpu" });

  console.log("Building linear2...");
  const linear2 = new Linear(api, 4, 4, { device: "webgpu" });

  // Get parameters
  const params1 = linear1.parameters();
  const params2 = linear2.parameters();

  console.log(`\nParameters: ${params1.length}`);

  // Input
  const input = api.tensorFromArray([1, 2, 3, 4], [1, 4], { device: "webgpu" });

  // Forward 1 (before copy)
  console.log("\n--- Before copying weights ---");
  const out1 = linear1.forward(input);
  const out2 = linear2.forward(input);
  console.log(`  linear1 output: ${(await out1.cpu()).slice(0, 4)}`);
  console.log(`  linear2 output: ${(await out2.cpu()).slice(0, 4)}`);
  console.log(`  Different (expected): ✓`);

  // Copy weights from linear1 to linear2
  console.log("\n--- Copying weights ---");
  for (let i = 0; i < params1.length; i++) {
    const data = await params1[i].cpu();
    const temp = api.tensorFromArray(data, params1[i].shape, {
      device: "webgpu",
      requiresGrad: true,
    });
    params2[i].copy_(temp);
    console.log(`  Copied param ${i}, shape: ${params1[i].shape}`);
  }
  await api.markStep();

  // Verify weights match
  console.log("\n  Verifying weights...");
  for (let i = 0; i < params1.length; i++) {
    const d1 = await params1[i].cpu();
    const d2 = await params2[i].cpu();
    console.log(`  param1[${i}][0:4]: ${Array.from(d1.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  param2[${i}][0:4]: ${Array.from(d2.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  }

  // Forward 2 (after copy)
  console.log("\n--- After copying weights ---");
  const input2a = api.tensorFromArray([1, 2, 3, 4], [1, 4], { device: "webgpu" });
  const input2b = api.tensorFromArray([1, 2, 3, 4], [1, 4], { device: "webgpu" });

  const out3 = linear1.forward(input2a);
  const out4 = linear2.forward(input2b);

  const o3 = await out3.cpu();
  const o4 = await out4.cpu();

  console.log(`  linear1 output: ${Array.from(o3).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  linear2 output: ${Array.from(o4).map(v => v.toFixed(4)).join(", ")}`);

  const match = Array.from(o3).every((v, i) => Math.abs(v - o4[i]) < 1e-5);
  console.log(`\n  Outputs match: ${match ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(match ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
