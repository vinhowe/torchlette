/**
 * Debug Attention - Detailed step-by-step comparison of attention mechanism
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { runTorchOracleBatch } from "../../test/oracle/torch-oracle";

const CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 32,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0,
};

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
  console.log("=== Attention Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.eval();

  const batchSize = 2;
  const seqLen = 4;

  // Extract weights
  const params = model.parameters();
  const weights: number[][] = [];
  for (const p of params) {
    const data = await p.cpu();
    weights.push(Array.from(data));
  }

  // Get embedding output (already verified to match)
  const inputData = [0, 1, 2, 3, 4, 5, 6, 7];
  const embedResults = await runTorchOracleBatch([{
    op: "embedding_forward",
    caseName: "embed",
    inputs: [
      { values: inputData, shape: [batchSize, seqLen] },
      { values: weights[0], shape: [CONFIG.vocabSize, CONFIG.embedDim] },
      { values: weights[1], shape: [CONFIG.blockSize, CONFIG.embedDim] },
    ],
    options: { seqLen },
  }]);
  const pytorchEmbed = embedResults[0].values;

  // LayerNorm1 output (already verified to match)
  const ln1Results = await runTorchOracleBatch([{
    op: "layer_norm_forward",
    caseName: "ln1",
    inputs: [
      { values: pytorchEmbed, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: weights[2], shape: [CONFIG.embedDim] },
      { values: weights[3], shape: [CONFIG.embedDim] },
    ],
    options: { normalized_shape: [CONFIG.embedDim], eps: 1e-5 },
  }]);
  const pytorchLn1 = ln1Results[0].values;

  console.log("Step 1: QKV Projection (raw output before split)");

  // Get PyTorch QKV raw
  const qkvResults = await runTorchOracleBatch([{
    op: "attention_debug",
    caseName: "qkv",
    inputs: [
      { values: pytorchLn1, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: weights[4], shape: [3 * CONFIG.embedDim, CONFIG.embedDim] },
      { values: weights[5], shape: [3 * CONFIG.embedDim] },
    ],
    options: { embedDim: CONFIG.embedDim, numHeads: CONFIG.numHeads },
  }]);
  const pytorchQkv = qkvResults[0].values;

  // Get Torchlette QKV via linear layer
  const ln1Input = api.tensorFromArray(pytorchLn1, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const torchletteQkvTensor = model.h[0].attn.cAttn.forward(ln1Input);
  const torchletteQkv = Array.from(await torchletteQkvTensor.cpu());

  const qkvDiff = maxDiff(torchletteQkv, pytorchQkv);
  console.log(`  QKV projection ${status(qkvDiff)}: max diff = ${qkvDiff.toExponential(2)}`);
  console.log(`  PyTorch QKV first 10: ${pytorchQkv.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  Torchlette QKV first 10: ${torchletteQkv.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);

  // If QKV matches, the issue is in the split/attention computation
  if (qkvDiff < 1e-4) {
    console.log("\n  QKV projection matches! Issue must be in split or attention computation.");

    // Let me trace through the torchlette attention step by step
    console.log("\nStep 2: Analyzing QKV split in Torchlette");

    // In torchlette, the attention does:
    // 1. qkv: [batch, seqLen, 3 * embedDim]
    // 2. reshape to [batch, seqLen, 3, embedDim]
    // 3. permute to [3, batch, seqLen, embedDim]
    // 4. flatten to [3, batch * seqLen * embedDim]
    // 5. gather to extract Q, K, V

    // Let me do the same split in PyTorch and compare
    const { embedDim, numHeads } = CONFIG;
    const headDim = embedDim / numHeads;

    // PyTorch split:
    // qkv [batch, seq, 3*embed] -> [batch, seq, 3, numHeads, headDim]
    // -> [3, batch, numHeads, seq, headDim]
    // q = qkv[0], k = qkv[1], v = qkv[2]

    // Extract Q from torchlette's QKV
    // qkv is [batch, seq, 3*embed] = [2, 4, 192]
    // In PyTorch, Q is the first 64 elements of the last dim for each position
    // In memory: qkv[b, s, :] = [q0, q1, ..., q63, k0, k1, ..., k63, v0, v1, ..., v63]

    // So pytorchQkv[0:64] should be Q for position (0, 0)
    // pytorchQkv[64:128] should be K for position (0, 0)
    // pytorchQkv[128:192] should be V for position (0, 0)

    const stride = 3 * embedDim; // 192 for 3D tensor [batch, seq, 3*embed]

    console.log(`\n  QKV layout analysis (stride=${stride}):`);
    console.log(`  Position (0,0) Q[0:5]: ${pytorchQkv.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  Position (0,0) K[64:69]: ${pytorchQkv.slice(64, 69).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  Position (0,0) V[128:133]: ${pytorchQkv.slice(128, 133).map(v => v.toFixed(4)).join(", ")}`);

    // Now let's see what torchlette produces for Q after split
    // The model's forward method does the split internally, so let's trace through

    // Actually, let me just run the full attention and see where it diverges
    console.log("\nStep 3: Full attention output comparison");

    const attnResults = await runTorchOracleBatch([{
      op: "attention_forward",
      caseName: "attn",
      inputs: [
        { values: pytorchLn1, shape: [batchSize, seqLen, CONFIG.embedDim] },
        { values: weights[4], shape: [3 * CONFIG.embedDim, CONFIG.embedDim] },
        { values: weights[5], shape: [3 * CONFIG.embedDim] },
        { values: weights[6], shape: [CONFIG.embedDim, CONFIG.embedDim] },
        { values: weights[7], shape: [CONFIG.embedDim] },
      ],
      options: { embedDim: CONFIG.embedDim, numHeads: CONFIG.numHeads },
    }]);
    const pytorchAttn = attnResults[0].values;

    const torchletteAttnTensor = model.h[0].attn.forward(ln1Input);
    const torchletteAttn = Array.from(await torchletteAttnTensor.cpu());

    const attnDiff = maxDiff(torchletteAttn, pytorchAttn);
    console.log(`  Full attention ${status(attnDiff, 1e-3)}: max diff = ${attnDiff.toExponential(2)}`);
    console.log(`  PyTorch attn first 10: ${pytorchAttn.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  Torchlette attn first 10: ${torchletteAttn.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);

    // Check if the issue is in the QKV split by manually computing Q, K, V
    // and comparing

    console.log("\nStep 4: Manual QKV split comparison");

    // PyTorch reshape order: [batch, seq, 3, numHeads, headDim]
    // So for flat index i in [batch, seq, 3*embed]:
    //   batch_idx = floor(i / (seq * 3 * embed))
    //   seq_idx = floor((i % (seq * 3 * embed)) / (3 * embed))
    //   qkv_idx = floor((i % (3 * embed)) / embed)
    //   embed_idx = i % embed

    // After PyTorch's reshape and permute to [3, batch, numHeads, seq, headDim]:
    // Q[0, b, h, s, d] came from qkv[b, s, 0, h, d]
    //   = qkv_flat[b * seq * 3*embed + s * 3*embed + h * headDim + d]
    //   Wait, that's not right...

    // Let me trace through more carefully.
    // Original: qkv[batch, seq, 3*embed]
    // Reshape to [batch, seq, 3, embed]: qkv[b, s, qkv_idx, e]
    //   where qkv[b, s, qkv_idx, e] = qkv_flat[b, s, qkv_idx * embed + e]
    // Then reshape to [batch, seq, 3, numHeads, headDim]
    //   qkv[b, s, qkv_idx, h, d] where e = h * headDim + d
    // Then permute to [3, batch, numHeads, seq, headDim]
    //   result[qkv_idx, b, h, s, d] = qkv[b, s, qkv_idx, h, d]

    // So Q (qkv_idx=0) for (b=0, h=0, s=0, d=0) comes from qkv[0, 0, 0, 0, 0]
    //   = qkv_flat[0, 0, 0 * embed + 0 * headDim + 0]
    //   = qkv_flat[0]

    // And Q for (b=0, h=0, s=0, d=1) comes from qkv[0, 0, 0, 0, 1]
    //   = qkv_flat[1]

    // And Q for (b=0, h=1, s=0, d=0) comes from qkv[0, 0, 0, 1, 0]
    //   = qkv_flat[0, 0, 0 * embed + 1 * headDim + 0]
    //   = qkv_flat[headDim] = qkv_flat[32]

    // This matches my earlier understanding: Q is at indices [0:64] for each position

    // Now let's check what torchlette's split produces
    // The code does:
    // 1. reshape [batch, seq, 3*embed] -> [batch, seq, 3, embed]
    // 2. permute [batch, seq, 3, embed] -> [3, batch, seq, embed]
    // 3. flatten to [3, batch*seq*embed]
    // 4. gather with indices [0, 0, 0, ...] to get Q as [1, batch*seq*embed]
    // 5. reshape to [batch, seq, embed]
    // 6. reshape to [batch, seq, numHeads, headDim]
    // 7. permute to [batch, numHeads, seq, headDim]

    // Step 2 (permute) gives [3, batch, seq, embed]
    // So Q (first slice along dim 0) is at indices [0, 1, 2, ..., batch*seq*embed-1]
    // of the [3, batch*seq*embed] flattened tensor

    // This should produce the same Q as PyTorch's approach
    // Let me verify by computing Q manually for both

    const Q_pytorch_0_0 = pytorchQkv.slice(0, headDim); // Q for (batch=0, seq=0, head=0)
    console.log(`  PyTorch Q[b=0,s=0,h=0,d=0:${headDim}]: ${Q_pytorch_0_0.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

    // For head 1, it should be at offset headDim
    const Q_pytorch_0_0_h1 = pytorchQkv.slice(headDim, 2 * headDim); // Q for (batch=0, seq=0, head=1)
    console.log(`  PyTorch Q[b=0,s=0,h=1,d=0:${headDim}]: ${Q_pytorch_0_0_h1.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  } else {
    console.log("\n  QKV projection differs! Issue is in the linear layer.");
  }

  console.log("\n=== Summary ===");
  console.log(`QKV projection difference: ${qkvDiff.toExponential(2)}`);

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
