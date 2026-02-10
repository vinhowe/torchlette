/**
 * Debug Parameter Order - Compare torchlette vs PyTorch parameter ordering
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";

const CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 32,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0,
};

async function main() {
  console.log("=== Parameter Order Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu");

  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  const params = model.parameters();

  console.log("Torchlette GPT-2 parameter shapes:");
  for (let i = 0; i < params.length; i++) {
    console.log(`  ${i}: [${params[i].shape.join(", ")}]`);
  }

  console.log("\nExpected PyTorch SimpleGPT2 parameter order:");
  const expected = [
    ["wte.weight", [CONFIG.vocabSize, CONFIG.embedDim]],
    ["wpe.weight", [CONFIG.blockSize, CONFIG.embedDim]],
    // Block 0
    ["blocks.0.ln_1.weight", [CONFIG.embedDim]],
    ["blocks.0.ln_1.bias", [CONFIG.embedDim]],
    ["blocks.0.attn.c_attn.weight", [3 * CONFIG.embedDim, CONFIG.embedDim]],
    ["blocks.0.attn.c_attn.bias", [3 * CONFIG.embedDim]],
    ["blocks.0.attn.c_proj.weight", [CONFIG.embedDim, CONFIG.embedDim]],
    ["blocks.0.attn.c_proj.bias", [CONFIG.embedDim]],
    ["blocks.0.ln_2.weight", [CONFIG.embedDim]],
    ["blocks.0.ln_2.bias", [CONFIG.embedDim]],
    ["blocks.0.mlp.c_fc.weight", [4 * CONFIG.embedDim, CONFIG.embedDim]],
    ["blocks.0.mlp.c_fc.bias", [4 * CONFIG.embedDim]],
    ["blocks.0.mlp.c_proj.weight", [CONFIG.embedDim, 4 * CONFIG.embedDim]],
    ["blocks.0.mlp.c_proj.bias", [CONFIG.embedDim]],
    // Block 1
    ["blocks.1.ln_1.weight", [CONFIG.embedDim]],
    ["blocks.1.ln_1.bias", [CONFIG.embedDim]],
    ["blocks.1.attn.c_attn.weight", [3 * CONFIG.embedDim, CONFIG.embedDim]],
    ["blocks.1.attn.c_attn.bias", [3 * CONFIG.embedDim]],
    ["blocks.1.attn.c_proj.weight", [CONFIG.embedDim, CONFIG.embedDim]],
    ["blocks.1.attn.c_proj.bias", [CONFIG.embedDim]],
    ["blocks.1.ln_2.weight", [CONFIG.embedDim]],
    ["blocks.1.ln_2.bias", [CONFIG.embedDim]],
    ["blocks.1.mlp.c_fc.weight", [4 * CONFIG.embedDim, CONFIG.embedDim]],
    ["blocks.1.mlp.c_fc.bias", [4 * CONFIG.embedDim]],
    ["blocks.1.mlp.c_proj.weight", [CONFIG.embedDim, 4 * CONFIG.embedDim]],
    ["blocks.1.mlp.c_proj.bias", [CONFIG.embedDim]],
    // Final LN
    ["ln_f.weight", [CONFIG.embedDim]],
    ["ln_f.bias", [CONFIG.embedDim]],
  ];

  for (let i = 0; i < expected.length; i++) {
    const [name, shape] = expected[i];
    console.log(`  ${i}: [${(shape as number[]).join(", ")}] (${name})`);
  }

  console.log("\nComparison:");
  let mismatchCount = 0;
  for (let i = 0; i < Math.min(params.length, expected.length); i++) {
    const torchletteShape = params[i].shape;
    const [name, expectedShape] = expected[i];
    const shapeMatch = JSON.stringify(torchletteShape) === JSON.stringify(expectedShape);
    if (!shapeMatch) {
      console.log(`  ${i}: MISMATCH - torchlette [${torchletteShape.join(", ")}] vs expected [${(expectedShape as number[]).join(", ")}] (${name})`);
      mismatchCount++;
    }
  }

  if (mismatchCount === 0) {
    console.log("  All shapes match!");
  }

  console.log(`\nTotal params: torchlette=${params.length}, expected=${expected.length}`);

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
