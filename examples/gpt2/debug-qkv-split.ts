/**
 * Debug QKV Split - Minimal test matching attention's QKV split pattern exactly
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

function status(diff: number, threshold: number = 1e-4): string {
  return diff < threshold ? "PASS" : "FAIL";
}

async function main() {
  console.log("=== QKV Split Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, seq = 3, embed = 4;
  const flatTotal = batch * seq * embed;

  // Input qkv tensor [batch, seq, 3 * embed]
  const qkvData = new Array(batch * seq * 3 * embed).fill(0).map((_, i) => (i + 1) * 0.01);

  console.log("Test: QKV split via gather (matching attention pattern)");
  {
    // Torchlette implementation (matching model.ts attention)
    const qkv = api.tensorFromArray(qkvData, [batch, seq, 3 * embed], { requiresGrad: true, device: "webgpu" });

    // Reshape and permute to [3, batch, seq, embed]
    const qkvSplit = qkv.reshape([batch, seq, 3, embed]);
    const qkvTransposed = qkvSplit.permute([2, 0, 1, 3]);  // [3, batch, seq, embed]
    const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);

    // Gather rows
    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

    const qFlat = qkvFlattened.gather(idx0, { dim: 0 }).reshape([batch, seq, embed]);
    const kFlat = qkvFlattened.gather(idx1, { dim: 0 }).reshape([batch, seq, embed]);
    const vFlat = qkvFlattened.gather(idx2, { dim: 0 }).reshape([batch, seq, embed]);

    // Just sum all three to create loss (simpler than full attention)
    const loss = api.add(api.add(api.sum(qFlat), api.sum(kFlat)), api.sum(vFlat));
    await loss.backward();

    const torchletteGrad = Array.from(await qkv.grad!.cpu());
    console.log(`  Torchlette grad first 12: ${torchletteGrad.slice(0, 12).map(v => v.toFixed(4)).join(", ")}`);

    // PyTorch reference: same split via indexing
    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "qkv_split_backward",
      caseName: "qkv_split",
      inputs: [{ values: qkvData, shape: [batch, seq, 3 * embed] }],
      options: { embedDim: embed },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;
    console.log(`  PyTorch grad first 12: ${pytorchGrad.slice(0, 12).map((v: number) => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);

    // Show expected (should be all 1s)
    console.log(`  Expected: all 1s (since loss = sum(q) + sum(k) + sum(v) = sum(qkv))`);
  }

  console.log("\nTest 2: QKV split + head reshape (full attention split pattern)");
  {
    const numHeads = 2;
    const headDim = embed / numHeads;

    const qkv = api.tensorFromArray(qkvData, [batch, seq, 3 * embed], { requiresGrad: true, device: "webgpu" });

    // Split
    const qkvSplit = qkv.reshape([batch, seq, 3, embed]);
    const qkvTransposed = qkvSplit.permute([2, 0, 1, 3]);
    const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);

    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

    const qFlat = qkvFlattened.gather(idx0, { dim: 0 }).reshape([batch, seq, embed]);
    const kFlat = qkvFlattened.gather(idx1, { dim: 0 }).reshape([batch, seq, embed]);
    const vFlat = qkvFlattened.gather(idx2, { dim: 0 }).reshape([batch, seq, embed]);

    // Reshape to heads (exactly as in attention)
    const q = qFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = kFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = vFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();

    // Simple loss (sum all)
    const loss = api.add(api.add(api.sum(q), api.sum(k)), api.sum(v));
    await loss.backward();

    const torchletteGrad = Array.from(await qkv.grad!.cpu());
    console.log(`  Torchlette grad first 12: ${torchletteGrad.slice(0, 12).map(v => v.toFixed(4)).join(", ")}`);

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "qkv_head_split_backward",
      caseName: "qkv_head_split",
      inputs: [{ values: qkvData, shape: [batch, seq, 3 * embed] }],
      options: { embedDim: embed, numHeads },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;
    console.log(`  PyTorch grad first 12: ${pytorchGrad.slice(0, 12).map((v: number) => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  console.log("\nTest 3: Full attention (QKV split + attention computation)");
  {
    const numHeads = 2;
    const headDim = embed / numHeads;

    const qkv = api.tensorFromArray(qkvData, [batch, seq, 3 * embed], { requiresGrad: true, device: "webgpu" });

    // Split
    const qkvSplit = qkv.reshape([batch, seq, 3, embed]);
    const qkvTransposed = qkvSplit.permute([2, 0, 1, 3]);
    const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);

    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

    const qFlat = qkvFlattened.gather(idx0, { dim: 0 }).reshape([batch, seq, embed]);
    const kFlat = qkvFlattened.gather(idx1, { dim: 0 }).reshape([batch, seq, embed]);
    const vFlat = qkvFlattened.gather(idx2, { dim: 0 }).reshape([batch, seq, embed]);

    // Reshape to heads
    const q = qFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = kFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = vFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();

    // Attention computation
    const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
    const scale = 1.0 / Math.sqrt(headDim);
    const scores = q.matmul(kT);
    const scaledScores = api.mul(scores, api.tensorFromArray([scale], [], { device: "webgpu" }));

    // Causal mask
    const maskData = new Array(seq * seq);
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        maskData[i * seq + j] = j <= i ? 0 : -1e9;
      }
    }
    const mask = api.tensorFromArray(maskData, [1, 1, seq, seq], { device: "webgpu" });
    const maskedScores = api.add(scaledScores, mask);

    // Softmax
    const attnWeights = maskedScores.softmax(-1);

    // Weighted sum
    const attnOutput = attnWeights.matmul(v);

    const loss = api.sum(attnOutput);
    await loss.backward();

    const torchletteGrad = Array.from(await qkv.grad!.cpu());
    console.log(`  Torchlette grad first 12: ${torchletteGrad.slice(0, 12).map(v => v.toFixed(6)).join(", ")}`);

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "full_attention_backward",
      caseName: "full_attn",
      inputs: [{ values: qkvData, shape: [batch, seq, 3 * embed] }],
      options: { embedDim: embed, numHeads },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;
    console.log(`  PyTorch grad first 12: ${pytorchGrad.slice(0, 12).map((v: number) => v.toFixed(6)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  console.log("\nTest 4: Linear + full attention (x -> c_attn -> attention)");
  {
    const numHeads = 2;
    const headDim = embed / numHeads;

    // Input and weights
    const xData = new Array(batch * seq * embed).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(3 * embed * embed).fill(0).map((_, i) => (i + 1) * 0.001);
    const bData = new Array(3 * embed).fill(0).map((_, i) => (i + 1) * 0.001);

    const x = api.tensorFromArray(xData, [batch, seq, embed], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [3 * embed, embed], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [3 * embed], { requiresGrad: true, device: "webgpu" });

    // QKV projection
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const qkv = api.add(x.matmul(wT), b);

    // Split
    const qkvSplit = qkv.reshape([batch, seq, 3, embed]);
    const qkvTransposed = qkvSplit.permute([2, 0, 1, 3]);
    const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);

    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

    const qFlat = qkvFlattened.gather(idx0, { dim: 0 }).reshape([batch, seq, embed]);
    const kFlat = qkvFlattened.gather(idx1, { dim: 0 }).reshape([batch, seq, embed]);
    const vFlat = qkvFlattened.gather(idx2, { dim: 0 }).reshape([batch, seq, embed]);

    // Reshape to heads
    const q = qFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = kFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = vFlat.contiguous().reshape([batch, seq, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();

    // Attention computation
    const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
    const scale = 1.0 / Math.sqrt(headDim);
    const scores = q.matmul(kT);
    const scaledScores = api.mul(scores, api.tensorFromArray([scale], [], { device: "webgpu" }));

    // Causal mask
    const maskData = new Array(seq * seq);
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        maskData[i * seq + j] = j <= i ? 0 : -1e9;
      }
    }
    const mask = api.tensorFromArray(maskData, [1, 1, seq, seq], { device: "webgpu" });
    const maskedScores = api.add(scaledScores, mask);

    // Softmax
    const attnWeights = maskedScores.softmax(-1);

    // Weighted sum
    const attnOutput = attnWeights.matmul(v);

    const loss = api.sum(attnOutput);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());
    const torchletteWGrad = Array.from(await w.grad!.cpu());
    const torchletteBGrad = Array.from(await b.grad!.cpu());

    console.log(`  Torchlette x.grad first 8: ${torchletteXGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  Torchlette w.grad first 8: ${torchletteWGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "linear_plus_attention_backward",
      caseName: "linear_attn",
      inputs: [
        { values: xData, shape: [batch, seq, embed] },
        { values: wData, shape: [3 * embed, embed] },
        { values: bData, shape: [3 * embed] },
      ],
      options: { embedDim: embed, numHeads },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;
    const pytorchWGrad = pytorchResult[0].grads[1].values;
    const pytorchBGrad = pytorchResult[0].grads[2].values;

    console.log(`  PyTorch x.grad first 8: ${pytorchXGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch w.grad first 8: ${pytorchWGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const xDiff = maxDiff(torchletteXGrad, pytorchXGrad);
    const wDiff = maxDiff(torchletteWGrad, pytorchWGrad);
    const bDiff = maxDiff(torchletteBGrad, pytorchBGrad);
    console.log(`  x.grad ${status(xDiff)}: max diff = ${xDiff.toExponential(2)}`);
    console.log(`  w.grad ${status(wDiff)}: max diff = ${wDiff.toExponential(2)}`);
    console.log(`  b.grad ${status(bDiff)}: max diff = ${bDiff.toExponential(2)}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
