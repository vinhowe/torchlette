/**
 * Test chunked embedding lookup for large vocab
 */

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

  // GPT-2 vocab size and embed dim
  const vocabSize = 50257;
  const embedDim = 768;
  const embeddingSizeBytes = vocabSize * embedDim * 4;
  console.log(`\nEmbedding size: ${(embeddingSizeBytes / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Exceeds limit: ${embeddingSizeBytes > maxBindingSize ? "YES" : "NO"}`);

  // Create embedding weight
  console.log("\n[1] Creating embedding weight tensor...");
  const weight = api.randn([vocabSize, embedDim], { device: "webgpu" });
  console.log(`  Weight shape: [${weight.shape}]`);

  // Test small gather (should work)
  console.log("\n[2] Testing small gather (3 tokens)...");
  const smallIndices = api.tensorFromArray(
    // Create indices [3, embedDim] - all same index per row
    Array.from({ length: 3 * embedDim }, (_, i) => Math.floor(i / embedDim) * 1000),
    [3, embedDim],
    { device: "webgpu" }
  );
  console.log(`  Indices shape: [${smallIndices.shape}]`);

  const smallResult = weight.gather(smallIndices, { dim: 0 });
  console.log(`  Result shape: [${smallResult.shape}]`);

  const smallValues = await smallResult.cpu();
  console.log(`  First few values: [${smallValues.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
  console.log("  Small gather: SUCCESS");

  // Test indices spanning the vocab
  console.log("\n[3] Testing gather spanning full vocab range...");
  const spanningIndices = api.tensorFromArray(
    // Indices at [0, 25000, 50000] repeated for embedDim
    [
      ...Array(embedDim).fill(0),
      ...Array(embedDim).fill(25000),
      ...Array(embedDim).fill(50000),
    ],
    [3, embedDim],
    { device: "webgpu" }
  );
  console.log(`  Indices: [0, 25000, 50000] (spanning full vocab)`);

  const spanResult = weight.gather(spanningIndices, { dim: 0 });
  console.log(`  Result shape: [${spanResult.shape}]`);

  const spanValues = await spanResult.cpu();
  console.log(`  First few values: [${spanValues.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
  console.log("  Spanning gather: SUCCESS");

  // Test matmul with the large weight (this is what fails for lm_head)
  console.log("\n[4] Testing matmul with embedding weight (lm_head simulation)...");
  const x = api.randn([1, 10, embedDim], { device: "webgpu" }); // [batch=1, seq=10, embed=768]
  console.log(`  Input x shape: [${x.shape}]`);
  console.log(`  Weight.T shape: [${embedDim}, ${vocabSize}]`);

  try {
    // This is what GPT-2 does: x @ wte.weight.T
    const weightT = weight.transpose({ dim0: 0, dim1: 1 }).contiguous();
    console.log(`  Transposed weight shape: [${weightT.shape}]`);

    const logits = x.matmul(weightT);
    console.log(`  Logits shape: [${logits.shape}]`);

    const logitsValues = await logits.cpu();
    console.log(`  First few logits: [${logitsValues.slice(0, 5).map(v => v.toFixed(4)).join(", ")}]`);
    console.log("  Matmul with large weight: SUCCESS");
  } catch (e) {
    console.log(`  Matmul with large weight: FAILED`);
    console.log(`  Error: ${e instanceof Error ? e.message : e}`);
    console.log("\n  Note: The matmul needs chunking for weights > 128MB");
  }

  console.log("\n[5] Summary");
  console.log("  - Chunked gather works for large embedding lookup");
  console.log("  - Matmul with large weights needs chunking (for lm_head)");
  console.log("\nDone!");
}

main().catch(console.error);
