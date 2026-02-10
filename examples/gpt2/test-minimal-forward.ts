/**
 * Minimal test of GPT-2 forward pass with loaded weights
 */

import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU, getMaxStorageBufferBindingSize } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

async function main() {
  console.log("Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("WebGPU not available!");
    process.exit(1);
  }

  const maxBindingSize = getMaxStorageBufferBindingSize();
  console.log(`Max storage buffer binding size: ${(maxBindingSize / 1024 / 1024).toFixed(2)} MB`);

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  // Load model
  console.log("\n[1] Loading pretrained DistilGPT-2...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();
  console.log("Model loaded!");

  // Test single forward pass with small input
  console.log("\n[2] Testing single forward pass...");
  const inputIds = api.tensorFromArray([100, 200, 300], [1, 3], { device: "webgpu" });
  console.log(`  Input shape: [${inputIds.shape}]`);

  try {
    const logits = model.forward(inputIds);
    console.log(`  Logits shape: [${logits.shape}]`);

    const logitsData = await logits.cpu();
    console.log(`  First few logits: [${logitsData.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
    console.log("  Forward pass: SUCCESS");
  } catch (e) {
    console.log(`  Forward pass: FAILED`);
    console.log(`  Error: ${e instanceof Error ? e.message : e}`);
    if (e instanceof Error && e.stack) {
      console.log(`  Stack: ${e.stack.split('\n').slice(0, 5).join('\n')}`);
    }
  }

  console.log("\nDone!");
}

main().catch(console.error);
