/**
 * Debug Attention Gradient v2 - Exact match with PyTorch oracle
 * Using the same reshape/split pattern as PyTorch
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { runTorchOracleBackwardBatch } from "../../test/oracle/torch-oracle";

function maxDiff(a: number[], b: number[]): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

function status(diff: number, threshold: number = 1e-3): string {
  return diff < threshold ? "PASS" : "FAIL";
}

async function main() {
  console.log("=== Attention Gradient Debug v2 (matching PyTorch pattern) ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2;
  const seqLen = 4;
  const embedDim = 8;
  const numHeads = 2;
  const headDim = embedDim / numHeads;

  // Fixed sequential data for reproducibility
  const xData = new Array(batch * seqLen * embedDim).fill(0).map((_, i) => (i + 1) * 0.01);
  const cAttnWeight = new Array(3 * embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
  const cAttnBias = new Array(3 * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
  const cProjWeight = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
  const cProjBias = new Array(embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

  // Get PyTorch gradient
  const pytorchResult = await runTorchOracleBackwardBatch([{
    op: "attention_block_backward",
    caseName: "attn",
    inputs: [
      { values: xData, shape: [batch, seqLen, embedDim] },
      { values: cAttnWeight, shape: [3 * embedDim, embedDim] },
      { values: cAttnBias, shape: [3 * embedDim] },
      { values: cProjWeight, shape: [embedDim, embedDim] },
      { values: cProjBias, shape: [embedDim] },
    ],
    options: { embedDim, numHeads },
  }]);

  const pytorchLoss = pytorchResult[0].output.values[0];
  const pytorchGrads = pytorchResult[0].grads;

  console.log(`PyTorch loss: ${pytorchLoss.toFixed(6)}`);

  // Torchlette attention - matching PyTorch pattern exactly
  const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });
  const wAttn = api.tensorFromArray(cAttnWeight, [3 * embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
  const bAttn = api.tensorFromArray(cAttnBias, [3 * embedDim], { requiresGrad: true, device: "webgpu" });
  const wProj = api.tensorFromArray(cProjWeight, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
  const bProj = api.tensorFromArray(cProjBias, [embedDim], { requiresGrad: true, device: "webgpu" });

  // QKV projection: qkv = x @ wAttn.T + bAttn
  const wAttnT = wAttn.transpose({ dim0: 0, dim1: 1 }).contiguous();
  const qkv = api.add(x.matmul(wAttnT), bAttn);  // [batch, seq, 3*embed]

  // Match PyTorch pattern: reshape to [batch, seq, 3, heads, head_dim]
  const qkvReshaped = qkv.reshape([batch, seqLen, 3, numHeads, headDim]);
  // permute to [3, batch, heads, seq, head_dim]
  const qkvPermuted = qkvReshaped.permute([2, 0, 3, 1, 4]).contiguous();

  // Index to get q, k, v - shape [batch, heads, seq, head_dim]
  const flatTotal = batch * numHeads * seqLen * headDim;
  const qkvFlat = qkvPermuted.reshape([3, flatTotal]);

  const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
  const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
  const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

  const q = qkvFlat.gather(idx0, { dim: 0 }).reshape([batch, numHeads, seqLen, headDim]).contiguous();
  const k = qkvFlat.gather(idx1, { dim: 0 }).reshape([batch, numHeads, seqLen, headDim]).contiguous();
  const v = qkvFlat.gather(idx2, { dim: 0 }).reshape([batch, numHeads, seqLen, headDim]).contiguous();

  // Attention scores: (q @ k.T) * scale
  const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
  const scale = 1.0 / Math.sqrt(headDim);
  const scores = q.matmul(kT);  // [batch, heads, seq, seq]
  const scaledScores = api.mul(scores, api.tensorFromArray([scale], [], { device: "webgpu" }));

  // Causal mask
  const maskData = new Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      maskData[i * seqLen + j] = j <= i ? 0 : -1e9;
    }
  }
  const mask = api.tensorFromArray(maskData, [1, 1, seqLen, seqLen], { device: "webgpu" });
  const maskedScores = api.add(scaledScores, mask);

  // Softmax
  const attnWeights = maskedScores.softmax(-1);

  // Weighted sum: attn @ v
  const attnOutput = attnWeights.matmul(v);  // [batch, heads, seq, head_dim]

  // Concat heads: permute to [batch, seq, heads, head_dim], reshape to [batch, seq, embed]
  const attnConcat = attnOutput.permute([0, 2, 1, 3]).contiguous().reshape([batch, seqLen, embedDim]);

  // Output projection: out @ wProj.T + bProj
  const wProjT = wProj.transpose({ dim0: 0, dim1: 1 }).contiguous();
  const output = api.add(attnConcat.matmul(wProjT), bProj);

  // Loss
  const loss = api.sum(output);
  const torchletteL = await loss.item();
  console.log(`Torchlette loss: ${torchletteL.toFixed(6)}`);
  console.log(`Loss diff: ${Math.abs(pytorchLoss - torchletteL).toExponential(2)}\n`);

  // Backward
  await loss.backward();

  // Compare gradients
  console.log("Gradient comparison:");
  console.log("-".repeat(60));

  const params = [
    { name: "x", t: x, idx: 0 },
    { name: "c_attn.weight", t: wAttn, idx: 1 },
    { name: "c_attn.bias", t: bAttn, idx: 2 },
    { name: "c_proj.weight", t: wProj, idx: 3 },
    { name: "c_proj.bias", t: bProj, idx: 4 },
  ];

  let allPass = true;
  for (const { name, t, idx } of params) {
    const grad = t.grad;
    if (!grad) {
      console.log(`${name}: NO GRADIENT`);
      allPass = false;
      continue;
    }

    const torchletteGrad = Array.from(await grad.cpu());
    const pytorchGrad = pytorchGrads[idx].values;

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    const pass = diff < 1e-3;
    if (!pass) allPass = false;

    console.log(`${name.padEnd(20)} ${status(diff)}: max diff = ${diff.toExponential(2)}`);
    if (!pass) {
      console.log(`  PyTorch first 8: ${pytorchGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
      console.log(`  Torchlette first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    }
  }

  console.log("-".repeat(60));
  console.log(`\nOverall: ${allPass ? "PASS" : "FAIL"}`);

  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
