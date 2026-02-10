/**
 * Debug Weight Tying - Test gradient accumulation when wte.weight is used twice
 *
 * In GPT-2, wte.weight is used in:
 * 1. Embedding lookup: gather(wte, indices)
 * 2. LM head: matmul(hidden, wte.T)
 *
 * The gradient should be the SUM of gradients from both uses.
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { runTorchOracleBatch } from "../../test/oracle/torch-oracle";
import { Embedding } from "../../src/nn/embedding";

function maxDiff(a: number[], b: number[]): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function status(diff: number, threshold: number = 1e-4): string {
  return diff < threshold ? "PASS" : "FAIL";
}

async function main() {
  console.log("=== Weight Tying Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Simple test: wte used in both gather and matmul
  const vocabSize = 8;
  const embedDim = 4;
  const seqLen = 3;
  const batchSize = 2;

  // Create wte weight
  const wteData = new Array(vocabSize * embedDim).fill(0).map((_, i) => (i + 1) * 0.1);

  // Input indices
  const indices = [0, 1, 2, 3, 4, 5]; // batch=2, seq=3

  console.log("--- Test 1: Embedding only (via Embedding class) ---");
  {
    // Use Embedding class which properly expands indices
    const emb = new Embedding(api, vocabSize, embedDim, { device: "webgpu" });
    // Replace weight with known values
    const wte = api.tensorFromArray(wteData, [vocabSize, embedDim], { requiresGrad: true, device: "webgpu" });
    (emb as unknown as { weight: typeof wte }).weight = wte;

    const idx = api.tensorFromArray(indices, [batchSize, seqLen], { device: "webgpu" });

    const embedded = emb.forward(idx);
    console.log(`  Embedded shape: [${embedded.shape.join(", ")}]`);

    const loss = api.sum(embedded);
    await loss.backward();

    const torchletteGrad = Array.from(await wte.grad!.cpu());
    console.log(`  Torchlette wte.grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    // Get PyTorch reference
    const result = await runTorchOracleBatch([{
      op: "weight_tying_embed_grad",
      caseName: "embed_only",
      inputs: [
        { values: wteData, shape: [vocabSize, embedDim] },
        { values: indices, shape: [batchSize, seqLen] },
      ],
      options: {},
    }]);
    const pytorchGrad = result[0].values;
    console.log(`  PyTorch wte.grad first 8: ${pytorchGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  Grad diff ${status(diff)}: ${diff.toExponential(2)}`);
  }

  console.log("\n--- Test 2: LM head only (matmul) ---");
  {
    const wte = api.tensorFromArray(wteData, [vocabSize, embedDim], { requiresGrad: true });

    // Simulated hidden states [batch, seq, embed]
    const hiddenData = new Array(batchSize * seqLen * embedDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const hidden = api.tensorFromArray(hiddenData, [batchSize, seqLen, embedDim]);

    // LM head: logits = hidden @ wte.T -> [batch, seq, vocab]
    const wteT = wte.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const logits = hidden.matmul(wteT);
    console.log(`  Logits shape: [${logits.shape.join(", ")}]`);

    const loss = api.sum(logits);
    await loss.backward();

    const torchletteGrad = Array.from(await wte.grad!.cpu());
    console.log(`  Torchlette wte.grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    // Get PyTorch reference
    const result = await runTorchOracleBatch([{
      op: "weight_tying_lmhead_grad",
      caseName: "lmhead_only",
      inputs: [
        { values: wteData, shape: [vocabSize, embedDim] },
        { values: hiddenData, shape: [batchSize, seqLen, embedDim] },
      ],
      options: {},
    }]);
    const pytorchGrad = result[0].values;
    console.log(`  PyTorch wte.grad first 8: ${pytorchGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  Grad diff ${status(diff)}: ${diff.toExponential(2)}`);
  }

  console.log("\n--- Test 3: Weight tying (both embedding + matmul) ---");
  {
    const wte = api.tensorFromArray(wteData, [vocabSize, embedDim], { requiresGrad: true, device: "webgpu" });
    const idx = api.tensorFromArray(indices, [batchSize, seqLen], { device: "webgpu" });

    // Step 1: Embedding (properly expanded gather like Embedding class does)
    const numElements = batchSize * seqLen;
    const flatIdx = idx.reshape([numElements]);
    const expandedIdx = flatIdx
      .reshape([numElements, 1])
      .expand([numElements, embedDim])
      .contiguous();
    const gathered = wte.gather(expandedIdx, { dim: 0 });
    const embedded = gathered.reshape([batchSize, seqLen, embedDim]);

    // Step 2: Use embedded as hidden
    const hidden = embedded; // [batch, seq, embed]

    // Step 3: LM head (matmul with same wte)
    const wteT = wte.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const logits = hidden.matmul(wteT); // [batch, seq, vocab]

    console.log(`  Embedded shape: [${embedded.shape.join(", ")}]`);
    console.log(`  Logits shape: [${logits.shape.join(", ")}]`);

    const loss = api.sum(logits);
    await loss.backward();

    const torchletteGrad = Array.from(await wte.grad!.cpu());
    console.log(`  Torchlette wte.grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    // Get PyTorch reference
    const result = await runTorchOracleBatch([{
      op: "weight_tying_both_grad",
      caseName: "both",
      inputs: [
        { values: wteData, shape: [vocabSize, embedDim] },
        { values: indices, shape: [batchSize, seqLen] },
      ],
      options: {},
    }]);
    const pytorchGrad = result[0].values;
    console.log(`  PyTorch wte.grad first 8: ${pytorchGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  Grad diff ${status(diff)}: ${diff.toExponential(2)}`);
  }

  console.log("\n=== Summary ===");
  console.log("If Test 1 and 2 pass but Test 3 fails, the issue is gradient accumulation.");
  console.log("If Test 1 fails, the issue is in gather backward.");
  console.log("If Test 2 fails, the issue is in matmul backward with transpose.");

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
