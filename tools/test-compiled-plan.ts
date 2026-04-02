/**
 * Minimal test for compiled execution plan.
 * Uses a small model that fits within software renderer limits (128MB storage buffer).
 *
 * Tests that compiled plan execution produces the same loss as normal execution.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { type Tensor, Torchlette } from "../src/frontend/torchlette";
import { crossEntropy } from "../src/nn";
import { Adam } from "../src/optim";

async function main() {
  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  // Small model: 2-layer MLP. All tensors well under 128MB.
  const hiddenDim = 128;
  const vocabSize = 1000;
  const seqLen = 16;
  const batchSize = 2;

  // Create model parameters manually (avoid nn.Module complexity)
  const embed = api.randn([vocabSize, hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w1 = api.randn([hiddenDim, hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b1 = api.randn([hiddenDim], {
    device: "webgpu",
    requiresGrad: true,
  });
  const w2 = api.randn([hiddenDim, vocabSize], {
    device: "webgpu",
    requiresGrad: true,
  });
  const b2 = api.randn([vocabSize], {
    device: "webgpu",
    requiresGrad: true,
  });

  const params = [embed, w1, b1, w2, b2];
  const optimizer = new Adam(params, { lr: 1e-3 }, api);

  // Compiled forward: embed -> linear -> relu -> linear -> cross_entropy
  const compiledForward = api.compile((inp: Tensor, tgt: Tensor) => {
    // Gather embeddings via api.embedding
    const emb = api.embedding(embed, inp); // [batch, seq, hidden]
    // Linear 1
    const h1 = emb.matmul(w1).add(b1).relu();
    // Linear 2 -> logits
    const logits = h1.matmul(w2).add(b2); // [batch, seq, vocab]
    // Flatten for cross_entropy
    const flatLogits = logits.reshape([batchSize * seqLen, vocabSize]);
    const flatTargets = tgt.reshape([batchSize * seqLen]);
    return crossEntropy(api, flatLogits, flatTargets);
  });

  // Fixed input data (same every step to make loss deterministic)
  const inputData = Array.from(
    { length: batchSize * seqLen },
    (_, i) => i % vocabSize,
  );
  const targetData = Array.from(
    { length: batchSize * seqLen },
    (_, i) => (i + 1) % vocabSize,
  );

  const NUM_STEPS = parseInt(process.env.NUM_STEPS || "6", 10);
  const losses: number[] = [];

  for (let step = 0; step < NUM_STEPS; step++) {
    const input = api.tensorFromArray(inputData, [batchSize, seqLen], {
      device: "webgpu",
      dtype: "i32",
    });
    const target = api.tensorFromArray(targetData, [batchSize, seqLen], {
      device: "webgpu",
      dtype: "i32",
    });

    const loss = compiledForward(input, target);
    const lossValue = await loss.item();

    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();

    loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();

    losses.push(lossValue);
    console.log(`Step ${step}: loss=${lossValue.toFixed(8)}`);
  }

  const compiled = process.env.TORCHLETTE_COMPILED_PLAN === "1";
  console.log(`\nCompiled plan: ${compiled ? "ON" : "OFF"}`);
  console.log(`Losses: [${losses.map((l) => l.toFixed(8)).join(", ")}]`);

  // Basic sanity checks
  const allFinite = losses.every((l) => isFinite(l) && l > 0);
  const decreasing = losses.length > 2 && losses[losses.length - 1] < losses[0];
  console.log(`All finite & positive: ${allFinite}`);
  console.log(`Loss decreasing: ${decreasing}`);

  if (!allFinite) {
    console.error("FAIL: Non-finite or zero losses detected!");
    process.exit(2);
  }

  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
