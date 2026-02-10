/**
 * Test checkpoint with embedding + tied weights (like GPT-2).
 */

import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { Embedding } from "../../src/nn";
import { checkpoint } from "../../src/nn/checkpoint";

async function main() {
  console.log("=== Checkpoint Embedding Test ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Simulated GPT-2 style: embedding + matmul with transposed weight
  const vocabSize = 100;
  const embedDim = 32;

  const embedding = new Embedding(api, vocabSize, embedDim, { device: "webgpu" });

  const forward = (tokens: Tensor) => {
    // Get embeddings
    const embeds = embedding.forward(tokens);  // [batch, seq, embed]

    // Compute "logits" using tied weights (matmul with transposed embedding weight)
    const logits = embeds.matmul(embedding.weight.transpose({ dim0: 0, dim1: 1 }));

    return logits;
  };

  const params = embedding.parameters();
  console.log(`Parameters: ${params.length}`);
  console.log(`  weight shape: ${params[0].shape}`);

  const tokens = api.tensorFromArray([0, 1, 2, 3], [1, 4], { device: "webgpu" });

  // Test 1: WITHOUT checkpoint
  console.log("\n--- Test 1: WITHOUT checkpoint ---");
  const out1 = forward(tokens);
  const loss1 = api.sum(out1);
  console.log(`  loss: ${(await loss1.item()).toFixed(6)}`);

  await loss1.backward();
  await api.markStep();

  let gradSum1 = 0;
  for (const p of params) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let i = 0; i < g.length; i++) gradSum1 += Math.abs(g[i]);
    }
  }
  console.log(`  gradSum: ${gradSum1.toFixed(4)}`);
  console.log(`  weight.grad exists: ${params[0].grad !== undefined}`);

  // Zero grads
  for (const p of params) p.zeroGrad();

  // Test 2: WITH checkpoint
  console.log("\n--- Test 2: WITH checkpoint ---");
  const tokens2 = api.tensorFromArray([0, 1, 2, 3], [1, 4], { device: "webgpu" });

  const out2 = checkpoint(api, (tok: Tensor) => forward(tok), [tokens2]);
  const loss2 = api.sum(out2);
  console.log(`  loss: ${(await loss2.item()).toFixed(6)}`);

  await loss2.backward();
  await api.markStep();

  let gradSum2 = 0;
  for (const p of params) {
    if (p.grad) {
      const g = await p.grad.cpu();
      for (let i = 0; i < g.length; i++) gradSum2 += Math.abs(g[i]);
    }
  }
  console.log(`  gradSum: ${gradSum2.toFixed(4)}`);
  console.log(`  weight.grad exists: ${params[0].grad !== undefined}`);

  // Compare
  console.log("\n--- Comparison ---");
  const gradMatch = Math.abs(gradSum1 - gradSum2) / Math.max(gradSum1, 1) < 0.01;
  console.log(`  gradSum match (1% tol): ${gradMatch ? "✓ PASS" : "✗ FAIL"}`);
  console.log(`    ${gradSum1.toFixed(4)} vs ${gradSum2.toFixed(4)}`);

  process.exit(gradMatch ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
