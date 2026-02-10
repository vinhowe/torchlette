/**
 * Test if parameters() returns the actual tensor objects used by the model.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

async function main() {
  console.log("=== Parameter Identity Test ===\n");

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

  console.log("Building model...");
  const model = new GPT2(api, config, { device: "webgpu" });

  // Get parameters twice
  const params1 = model.parameters();
  const params2 = model.parameters();

  // Check if they're the same objects
  console.log("\nChecking object identity:");
  let allSame = true;
  for (let i = 0; i < params1.length; i++) {
    const same = params1[i] === params2[i];
    if (!same) {
      console.log(`  param[${i}]: different objects!`);
      allSame = false;
    }
  }

  if (allSame) {
    console.log("  All parameters return same object ✓");
  }

  // Also check that modifying params affects model.wte.weight
  console.log("\nChecking direct access to model.wte.weight:");
  const wteWeight = model.wte.weight;
  const params = model.parameters();
  console.log(`  params[0] === model.wte.weight: ${params[0] === wteWeight}`);
  console.log(`  params[0].baseId === model.wte.weight.baseId: ${params[0].baseId === wteWeight.baseId}`);

  // Modify params[0] and check if wte.weight changes
  console.log("\nModifying params[0] via copy_:");
  const before = await wteWeight.cpu();
  console.log(`  wte.weight[0:5] before: ${Array.from(before.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);

  const newData = new Float32Array(before.length);
  newData.fill(999);
  const temp = api.tensorFromArray(newData, params[0].shape, { device: "webgpu", requiresGrad: true });
  params[0].copy_(temp);
  await api.markStep();

  const afterParams = await params[0].cpu();
  const afterWte = await wteWeight.cpu();
  console.log(`  params[0][0:5] after: ${Array.from(afterParams.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  wte.weight[0:5] after: ${Array.from(afterWte.slice(0, 5)).map(v => v.toFixed(4)).join(", ")}`);

  const bothChanged = afterParams[0] === 999 && afterWte[0] === 999;
  console.log(`\n  Both changed to 999: ${bothChanged ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(bothChanged ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
