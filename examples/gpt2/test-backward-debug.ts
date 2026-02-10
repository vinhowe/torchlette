/**
 * Debug GPT-2 backward pass - identify which operation causes validation error
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
  console.log("\nLoading model...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();
  console.log("Model loaded!");

  // Very simple test - just embedding forward and backward
  console.log("\n--- Test 1: Embedding only ---");
  {
    const tokens = api.tensorFromArray([100, 200], [1, 2], { device: "webgpu" });
    console.log("  Forward...");
    const embedded = model.wte.forward(tokens);
    console.log(`  Embedded shape: [${embedded.shape}]`);

    console.log("  Computing sum...");
    const loss = embedded.sum();
    const lossVal = await loss.item();
    console.log(`  Loss: ${lossVal}`);

    console.log("  Backward...");
    await loss.backward();
    console.log("  Done!");

    // Check gradient
    const grad = model.wte.weight.grad;
    if (grad) {
      const gradData = await grad.cpu();
      const nonZero = gradData.filter((v: number) => Math.abs(v) > 1e-10).length;
      console.log(`  Grad non-zero: ${nonZero}/${gradData.length}`);
    }

    // Cleanup
    tokens.dispose();
    embedded.dispose();
    loss.dispose();
    model.wte.weight.zeroGrad();
  }

  // Test with position embedding too
  console.log("\n--- Test 2: Token + Position embedding ---");
  {
    const tokens = api.tensorFromArray([100, 200], [1, 2], { device: "webgpu" });
    console.log("  Forward...");
    const tokEmbed = model.wte.forward(tokens);
    const posIds = api.tensorFromArray([0, 1], [1, 2], { device: "webgpu" });
    const posEmbed = model.wpe.forward(posIds);
    const combined = tokEmbed.add(posEmbed);
    console.log(`  Combined shape: [${combined.shape}]`);

    console.log("  Computing sum...");
    const loss = combined.sum();
    const lossVal = await loss.item();
    console.log(`  Loss: ${lossVal}`);

    console.log("  Backward...");
    await loss.backward();
    console.log("  Done!");

    // Cleanup
    tokens.dispose();
    posIds.dispose();
    tokEmbed.dispose();
    posEmbed.dispose();
    combined.dispose();
    loss.dispose();
    model.wte.weight.zeroGrad();
    model.wpe.weight.zeroGrad();
  }

  // Test with lm_head (uses wte weight tied)
  console.log("\n--- Test 3: Forward to logits (includes lm_head) ---");
  {
    const tokens = api.tensorFromArray([100, 200], [1, 2], { device: "webgpu" });
    console.log("  Forward...");
    const logits = model.forward(tokens);
    console.log(`  Logits shape: [${logits.shape}]`);

    console.log("  Computing sum...");
    const loss = logits.sum();
    const lossVal = await loss.item();
    console.log(`  Loss: ${lossVal}`);

    console.log("  Backward...");
    await loss.backward();
    console.log("  Done!");

    // Check gradient
    const grad = model.wte.weight.grad;
    if (grad) {
      const gradData = await grad.cpu();
      const nonZero = gradData.filter((v: number) => Math.abs(v) > 1e-10).length;
      console.log(`  wte.weight.grad non-zero: ${nonZero}/${gradData.length}`);
    }

    // Cleanup
    tokens.dispose();
    logits.dispose();
    loss.dispose();
  }

  console.log("\nAll tests completed!");
}

main().catch(console.error);
