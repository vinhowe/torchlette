/**
 * Test embedding lookup with loaded weights (safetensors)
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU, getMaxStorageBufferBindingSize } from "../../src/backend/webgpu";

async function main() {
  console.log("Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("WebGPU not available!");
    process.exit(1);
  }

  const maxBindingSize = getMaxStorageBufferBindingSize();
  console.log(`Max storage buffer binding size: ${(maxBindingSize / 1024 / 1024).toFixed(2)} MB`);

  const api = new Torchlette("webgpu");

  // Create embedding weight with same size as GPT-2
  const vocabSize = 50257;
  const embedDim = 768;
  const embeddingSizeBytes = vocabSize * embedDim * 4;
  console.log(`\nEmbedding size: ${(embeddingSizeBytes / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Exceeds limit: ${embeddingSizeBytes > maxBindingSize ? "YES" : "NO"}`);

  // Create random embedding weight
  console.log("\n[1] Creating embedding weight...");
  const weight = api.randn([vocabSize, embedDim], { device: "webgpu" });
  console.log(`  Weight shape: [${weight.shape}]`);
  console.log(`  Weight size: ${weight.size} elements`);

  // Test simple gather (embedding lookup for 10 tokens)
  console.log("\n[2] Testing embedding lookup for 10 tokens...");
  const tokenIds = api.tensorFromArray(
    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    [1, 10],
    { device: "webgpu" }
  );
  console.log(`  Token IDs shape: [${tokenIds.shape}]`);

  // Expand indices for gather (same as Embedding.forward)
  const numTokens = 10;
  const flatInput = tokenIds.reshape([numTokens]);
  const expandedInput = flatInput
    .reshape([numTokens, 1])
    .expand([numTokens, embedDim])
    .contiguous();
  console.log(`  Expanded indices shape: [${expandedInput.shape}]`);

  // Gather
  const gathered = weight.gather(expandedInput, { dim: 0 });
  console.log(`  Gathered shape: [${gathered.shape}]`);

  // Force execution by reading result
  const gatherResult = await gathered.cpu();
  console.log(`  First few values: [${gatherResult.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
  console.log("  Embedding lookup: SUCCESS");

  // Test lm_head style matmul
  console.log("\n[3] Testing lm_head matmul...");
  const x = api.randn([1, 10, embedDim], { device: "webgpu" });
  console.log(`  Input x shape: [${x.shape}]`);

  // Transpose weight and make contiguous
  const weightT = weight.transpose({ dim0: 0, dim1: 1 }).contiguous();
  console.log(`  Weight.T shape: [${weightT.shape}]`);

  // Matmul
  const logits = x.matmul(weightT);
  console.log(`  Logits shape: [${logits.shape}]`);

  // Force execution
  const logitsResult = await logits.cpu();
  console.log(`  First few logits: [${logitsResult.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
  console.log("  lm_head matmul: SUCCESS");

  // Test combined execution (like GPT-2 forward)
  console.log("\n[4] Testing combined embedding + lm_head (same lazy graph)...");

  // Create fresh token ids
  const tokens2 = api.tensorFromArray(
    [100, 200, 300, 400, 500],
    [1, 5],
    { device: "webgpu" }
  );

  // Embedding lookup
  const flatInput2 = tokens2.reshape([5]);
  const expandedInput2 = flatInput2
    .reshape([5, 1])
    .expand([5, embedDim])
    .contiguous();
  const embedded2 = weight.gather(expandedInput2, { dim: 0 }).reshape([1, 5, embedDim]);
  console.log(`  Embedded shape: [${embedded2.shape}]`);

  // lm_head matmul (same weight tensor)
  const weightT2 = weight.transpose({ dim0: 0, dim1: 1 }).contiguous();
  const logits2 = embedded2.matmul(weightT2);
  console.log(`  Logits shape: [${logits2.shape}]`);

  // Force both operations in one .cpu() call
  const logitsResult2 = await logits2.cpu();
  console.log(`  First few logits: [${logitsResult2.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
  console.log("  Combined execution: SUCCESS");

  // Test multiple forward passes (like generation loop)
  console.log("\n[5] Testing multiple forward passes...");
  for (let i = 0; i < 5; i++) {
    const tokens = api.tensorFromArray([100 + i * 100], [1, 1], { device: "webgpu" });
    const flat = tokens.reshape([1]);
    const expanded = flat.reshape([1, 1]).expand([1, embedDim]).contiguous();
    const emb = weight.gather(expanded, { dim: 0 }).reshape([1, 1, embedDim]);
    const wT = weight.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const out = emb.matmul(wT);
    const result = await out.cpu();
    console.log(`  Pass ${i + 1}: first logit = ${result[0].toFixed(4)}`);
  }
  console.log("  Multiple passes: SUCCESS");

  console.log("\nDone!");
}

main().catch(console.error);
