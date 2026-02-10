/**
 * Test lm_head backward in isolation
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

  // Test lm_head in isolation
  // lm_head computes: output = x @ wte.weight.T
  // where x is [B, S, 768] and wte.weight is [50257, 768]
  console.log("\n--- Test: lm_head matmul backward ---");

  // Create small input
  const x = api.randn([1, 2, 768], { device: "webgpu", requiresGrad: true });
  console.log(`  x shape: [${x.shape}]`);
  console.log(`  wte.weight shape: [${model.wte.weight.shape}]`);

  // Transpose wte.weight
  console.log("  Computing wte.weight.T...");
  const wteT = model.wte.weight.transpose({ dim0: 0, dim1: 1 }); // [768, 50257]
  console.log(`  wte.weight.T shape: [${wteT.shape}]`);

  // Make contiguous (required for matmul)
  console.log("  Making contiguous...");
  const wteTContiguous = wteT.contiguous();
  console.log(`  wteTContiguous shape: [${wteTContiguous.shape}]`);

  // Matmul
  console.log("  Forward matmul...");
  const logits = x.matmul(wteTContiguous);
  console.log(`  logits shape: [${logits.shape}]`);

  // Sum loss
  console.log("  Computing loss...");
  const loss = logits.sum();
  const lossVal = await loss.item();
  console.log(`  loss: ${lossVal}`);

  // Backward
  console.log("  Backward...");
  await loss.backward();
  console.log("  Backward done!");

  // Check x gradient
  console.log("  Checking x gradient...");
  if (x.grad) {
    console.log(`    x.grad shape: [${x.grad.shape}]`);
    const xGradData = await x.grad.cpu();
    const xNonZero = xGradData.filter((v: number) => Math.abs(v) > 1e-10).length;
    console.log(`    x.grad non-zero: ${xNonZero}/${xGradData.length}`);
  } else {
    console.log("    x.grad is null!");
  }

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
