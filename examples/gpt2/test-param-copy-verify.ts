/**
 * Verify copy_ works for model parameters.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Parameter Copy Verify Test ===\n");

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

  console.log("Building model1...");
  const model1 = new GPT2(api, config, { device: "webgpu" });

  console.log("Building model2...");
  const model2 = new GPT2(api, config, { device: "webgpu" });

  const params1 = model1.parameters();
  const params2 = model2.parameters();

  console.log(`\nNumber of parameters: ${params1.length}`);

  // Check first few params before copy
  console.log("\nBefore copy (first 3 params, first 5 values):");
  for (let i = 0; i < 3; i++) {
    const d1 = await params1[i].cpu();
    const d2 = await params2[i].cpu();
    console.log(`  param1[${i}]: ${Array.from(d1.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  param2[${i}]: ${Array.from(d2.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);
    console.log();
  }

  // Copy weights
  console.log("Copying all weights...");
  for (let i = 0; i < params1.length; i++) {
    const data = await params1[i].cpu();
    const temp = api.tensorFromArray(data, params1[i].shape, {
      device: "webgpu",
      requiresGrad: true,
    });
    params2[i].copy_(temp);
  }
  await api.markStep();
  console.log("Done copying.");

  // Check first few params after copy
  console.log("\nAfter copy (first 3 params, first 5 values):");
  let allMatch = true;
  for (let i = 0; i < 3; i++) {
    const d1 = await params1[i].cpu();
    const d2 = await params2[i].cpu();
    console.log(`  param1[${i}]: ${Array.from(d1.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  param2[${i}]: ${Array.from(d2.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);

    // Check if all values match
    for (let j = 0; j < d1.length; j++) {
      if (Math.abs(d1[j] - d2[j]) > 1e-6) {
        console.log(`  MISMATCH at index ${j}: ${d1[j]} vs ${d2[j]}`);
        allMatch = false;
        break;
      }
    }
    console.log(`  Match: ${allMatch ? "✓" : "✗"}`);
    console.log();
  }

  // Check ALL params
  console.log("\nVerifying ALL parameters...");
  for (let i = 0; i < params1.length; i++) {
    const d1 = await params1[i].cpu();
    const d2 = await params2[i].cpu();
    for (let j = 0; j < d1.length; j++) {
      if (Math.abs(d1[j] - d2[j]) > 1e-6) {
        console.log(`  MISMATCH at param ${i}, index ${j}: ${d1[j]} vs ${d2[j]}`);
        allMatch = false;
        break;
      }
    }
    if (!allMatch) break;
  }

  console.log(`\nAll parameters match: ${allMatch ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(allMatch ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
