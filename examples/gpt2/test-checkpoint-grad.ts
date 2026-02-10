/**
 * Debug checkpoint gradient flow.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Linear } from "../../src/nn";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Checkpoint Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Simple test: linear layer
  const linear = new Linear(api, 4, 4, { device: "webgpu" });
  const params = linear.parameters();
  console.log(`Parameters: ${params.length}`);
  console.log(`  weight requiresGrad: ${params[0].requiresGrad}`);
  console.log(`  bias requiresGrad: ${params[1].requiresGrad}`);

  const input = api.tensorFromArray([1, 2, 3, 4], [1, 4], {
    device: "webgpu",
    requiresGrad: true,
  });

  // Test 1: WITHOUT checkpoint
  console.log("\n--- Test 1: WITHOUT checkpoint ---");
  const out1 = linear.forward(input);
  const loss1 = api.sum(out1);
  console.log(`  loss1 value: ${await loss1.item()}`);
  console.log(`  loss1 requiresGrad: ${loss1.requiresGrad}`);

  await loss1.backward();
  await api.markStep();

  console.log(`  weight.grad exists: ${params[0].grad !== undefined}`);
  if (params[0].grad) {
    const g = await params[0].grad.cpu();
    console.log(`  weight.grad[0:4]: ${Array.from(g.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  }

  // Zero grads
  for (const p of params) p.zeroGrad();

  // Test 2: WITH checkpoint
  console.log("\n--- Test 2: WITH checkpoint ---");
  const input2 = api.tensorFromArray([1, 2, 3, 4], [1, 4], {
    device: "webgpu",
    requiresGrad: true,
  });

  // Checkpoint just the linear forward
  const out2 = checkpoint(
    api,
    (inp: Tensor) => linear.forward(inp),
    [input2]
  );
  const loss2 = api.sum(out2);
  console.log(`  loss2 value: ${await loss2.item()}`);
  console.log(`  loss2 requiresGrad: ${loss2.requiresGrad}`);

  await loss2.backward();
  await api.markStep();

  console.log(`  weight.grad exists: ${params[0].grad !== undefined}`);
  if (params[0].grad) {
    const g = await params[0].grad.cpu();
    console.log(`  weight.grad[0:4]: ${Array.from(g.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  } else {
    console.log(`  NO GRADIENT on weight!`);
  }

  console.log(`\n  input2.grad exists: ${input2.grad !== undefined}`);
  if (input2.grad) {
    const g = await input2.grad.cpu();
    console.log(`  input2.grad: ${Array.from(g).map(v => v.toFixed(4)).join(", ")}`);
  }

  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
