/**
 * Test copying weights between Embedding layers.
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Embedding } from "../../src/nn";

async function main() {
  console.log("=== Embedding Copy Test ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Create two embeddings
  const vocabSize = 100;
  const embedDim = 32;

  console.log("Building embedding1...");
  const emb1 = new Embedding(api, vocabSize, embedDim, { device: "webgpu" });

  console.log("Building embedding2...");
  const emb2 = new Embedding(api, vocabSize, embedDim, { device: "webgpu" });

  // Input tokens
  const tokens = api.tensorFromArray([0, 1, 2, 3], [4], { device: "webgpu" });

  // Forward 1 (before copy)
  console.log("\n--- Before copying weights ---");
  const out1 = emb1.forward(tokens);
  const tokens2 = api.tensorFromArray([0, 1, 2, 3], [4], { device: "webgpu" });
  const out2 = emb2.forward(tokens2);
  const o1 = await out1.cpu();
  const o2 = await out2.cpu();
  console.log(`  emb1 output[0:4]: ${Array.from(o1.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  emb2 output[0:4]: ${Array.from(o2.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  Different (expected): ✓`);

  // Copy weights
  console.log("\n--- Copying weights ---");
  const data1 = await emb1.weight.cpu();
  const temp = api.tensorFromArray(data1, emb1.weight.shape, { device: "webgpu", requiresGrad: true });
  emb2.weight.copy_(temp);
  await api.markStep();

  // Verify
  const d1 = await emb1.weight.cpu();
  const d2 = await emb2.weight.cpu();
  console.log(`  emb1.weight[0:4]: ${Array.from(d1.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  emb2.weight[0:4]: ${Array.from(d2.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);

  // Forward 2 (after copy)
  console.log("\n--- After copying weights ---");
  const tokens3 = api.tensorFromArray([0, 1, 2, 3], [4], { device: "webgpu" });
  const tokens4 = api.tensorFromArray([0, 1, 2, 3], [4], { device: "webgpu" });
  const out3 = emb1.forward(tokens3);
  const out4 = emb2.forward(tokens4);
  const o3 = await out3.cpu();
  const o4 = await out4.cpu();

  console.log(`  emb1 output[0:4]: ${Array.from(o3.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  emb2 output[0:4]: ${Array.from(o4.slice(0, 4)).map(v => v.toFixed(4)).join(", ")}`);

  const match = Array.from(o3).every((v, i) => Math.abs(v - o4[i]) < 1e-5);
  console.log(`\n  Outputs match: ${match ? "✓ PASS" : "✗ FAIL"}`);

  process.exit(match ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
