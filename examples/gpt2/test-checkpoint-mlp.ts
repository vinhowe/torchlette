/**
 * Test checkpoint with multi-layer model.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Linear, LayerNorm } from "../../src/nn";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Checkpoint MLP Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Build a small MLP
  const fc1 = new Linear(api, 16, 32, { device: "webgpu" });
  const fc2 = new Linear(api, 32, 16, { device: "webgpu" });
  const ln = new LayerNorm(api, 16, { device: "webgpu" });

  const forward = (x: Tensor) => {
    let h = fc1.forward(x);
    h = api.relu(h);
    h = fc2.forward(h);
    h = ln.forward(h);
    return h;
  };

  const allParams = [...fc1.parameters(), ...fc2.parameters(), ...ln.parameters()];
  console.log(`Total parameters: ${allParams.length}`);

  const input = api.tensorFromArray(
    Array.from({ length: 16 }, (_, i) => i * 0.1),
    [1, 16],
    { device: "webgpu", requiresGrad: true }
  );

  // Test 1: WITHOUT checkpoint
  console.log("\n--- Test 1: WITHOUT checkpoint ---");
  const out1 = forward(input);
  const loss1 = api.sum(out1);
  console.log(`  loss: ${(await loss1.item()).toFixed(6)}`);

  await loss1.backward();
  await api.markStep();

  let gradSum1 = 0;
  for (const p of allParams) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let i = 0; i < g.length; i++) gradSum1 += Math.abs(g[i]);
    }
  }
  console.log(`  gradSum: ${gradSum1.toFixed(4)}`);

  // Zero grads
  for (const p of allParams) p.zeroGrad();

  // Test 2: WITH checkpoint
  console.log("\n--- Test 2: WITH checkpoint ---");
  const input2 = api.tensorFromArray(
    Array.from({ length: 16 }, (_, i) => i * 0.1),
    [1, 16],
    { device: "webgpu", requiresGrad: true }
  );

  const out2 = checkpoint(api, (inp: Tensor) => forward(inp), [input2]);
  const loss2 = api.sum(out2);
  console.log(`  loss: ${(await loss2.item()).toFixed(6)}`);

  await loss2.backward();
  await api.markStep();

  let gradSum2 = 0;
  for (const p of allParams) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let i = 0; i < g.length; i++) gradSum2 += Math.abs(g[i]);
    }
  }
  console.log(`  gradSum: ${gradSum2.toFixed(4)}`);

  // Compare
  console.log("\n--- Comparison ---");
  const gradMatch = Math.abs(gradSum1 - gradSum2) / gradSum1 < 0.01;
  console.log(`  gradSum match (1% tol): ${gradMatch ? "✓ PASS" : "✗ FAIL"}`);
  console.log(`    ${gradSum1.toFixed(4)} vs ${gradSum2.toFixed(4)}`);

  process.exit(gradMatch ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
