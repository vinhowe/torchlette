/**
 * Test GPT-2 backward pass with loaded weights
 */

import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

async function main() {
  console.log("Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("WebGPU not available!");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Load model
  console.log("\n[1] Loading pretrained DistilGPT-2...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();
  console.log("Model loaded!");

  // Test backward pass
  console.log("\n[2] Testing backward pass...");
  const seqLen = 8;
  const inputData = [464, 995, 318, 257, 845, 922, 1110, 284]; // "The world is a very good place to"
  const targetData = [995, 318, 257, 845, 922, 1110, 284, 2107]; // shifted by 1

  const input = api.tensorFromArray(inputData, [1, seqLen], { device: "webgpu" });
  const target = api.tensorFromArray(targetData, [1, seqLen], { device: "webgpu" });

  console.log("  Running forward pass...");
  const startForward = Date.now();
  const { loss } = model.forwardWithLoss(input, target);
  if (!loss) throw new Error("Loss is null");
  const lossValue = await loss.item();
  console.log(`  Forward done in ${Date.now() - startForward}ms`);
  console.log(`  Loss: ${lossValue.toFixed(4)}`);

  console.log("  Running backward pass...");
  const startBackward = Date.now();
  await loss.backward();
  console.log(`  Backward done in ${Date.now() - startBackward}ms`);

  // Check gradient of embedding weight
  const wteGrad = model.wte.weight.grad;
  if (wteGrad) {
    const gradData = await wteGrad.cpu();
    const nonZero = gradData.filter(v => Math.abs(v) > 1e-10).length;
    console.log(`  wte.weight.grad shape: [${wteGrad.shape}]`);
    console.log(`  Non-zero gradients: ${nonZero}/${wteGrad.size}`);
  } else {
    console.log("  wte.weight.grad is null!");
  }

  console.log("\n[3] Testing optimizer step...");
  const { Adam } = await import("../../src/optim");
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);

  // Get wte weight before step
  const wteBefore = await model.wte.weight.cpu();
  console.log(`  wte weight before: [${wteBefore.slice(0, 5).map(v => v.toFixed(4)).join(", ")}...]`);

  optimizer.step();
  optimizer.zeroGrad();

  // Get wte weight after step
  const wteAfter = await model.wte.weight.cpu();
  console.log(`  wte weight after:  [${wteAfter.slice(0, 5).map(v => v.toFixed(4)).join(", ")}...]`);

  // Check if weights changed
  let changed = 0;
  for (let i = 0; i < wteBefore.length; i++) {
    if (Math.abs(wteBefore[i] - wteAfter[i]) > 1e-10) changed++;
  }
  console.log(`  Weights changed: ${changed}/${wteBefore.length}`);

  console.log("\nBackward pass test completed!");
}

main().catch(console.error);
