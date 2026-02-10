/**
 * Test gradient accumulation from embedding + lm_head
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

  // Test: embedding + lm_head using same wte.weight
  console.log("\n--- Test: Embedding + lm_head (weight tying) ---");

  // Input tokens
  const tokens = api.tensorFromArray([100, 200], [1, 2], { device: "webgpu" });
  console.log(`  tokens shape: [${tokens.shape}]`);

  // Embedding forward
  console.log("  Embedding forward...");
  const embedded = model.wte.forward(tokens);
  console.log(`  embedded shape: [${embedded.shape}]`);

  // lm_head forward: x @ wte.weight.T
  console.log("  lm_head forward...");
  const wteT = model.wte.weight.transpose({ dim0: 0, dim1: 1 }).contiguous();
  const logits = embedded.matmul(wteT);
  console.log(`  logits shape: [${logits.shape}]`);

  // Loss
  console.log("  Computing loss...");
  const loss = logits.sum();
  const lossVal = await loss.item();
  console.log(`  loss: ${lossVal}`);

  // Backward
  console.log("  Backward...");
  await loss.backward();
  console.log("  Backward done!");

  // Check wte.weight gradient
  console.log("  Checking wte.weight gradient...");
  if (model.wte.weight.grad) {
    console.log(`    wte.weight.grad shape: [${model.wte.weight.grad.shape}]`);
    const wteGradData = await model.wte.weight.grad.cpu();
    const wteNonZero = wteGradData.filter((v: number) => Math.abs(v) > 1e-10).length;
    console.log(`    wte.weight.grad non-zero: ${wteNonZero}/${wteGradData.length}`);
  } else {
    console.log("    wte.weight.grad is null!");
  }

  console.log("\nTest completed!");
}

main().catch(console.error);
