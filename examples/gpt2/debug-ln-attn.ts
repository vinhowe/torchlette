/**
 * Debug LayerNorm + Attention - Test gradient flow through ln -> attention -> residual
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { LayerNorm } from "../../src/nn";
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
  console.log("=== LayerNorm + Attention Gradient Debug ===\n");

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
  const flatTotal = batch * seqLen * embedDim;

  // Fixed values for reproducibility
  const xData = new Array(batch * seqLen * embedDim).fill(0).map((_, i) => (i + 1) * 0.01);
  const lnWeight = new Array(embedDim).fill(1); // gamma = 1
  const lnBias = new Array(embedDim).fill(0);   // beta = 0
  const cAttnWeight = new Array(3 * embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
  const cAttnBias = new Array(3 * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

  console.log("Test 1: LayerNorm only");
  {
    const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });

    // Create LayerNorm module
    const ln = new LayerNorm(api, embedDim, { device: "webgpu" });

    // Copy weights
    const gammaT = ln.weight;
    const betaT = ln.bias!;

    // Update weights to match lnWeight/lnBias
    // Actually in the module weights are already initialized to 1 and 0

    const lnOut = ln.forward(x);
    const loss = api.sum(lnOut);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());
    const torchletteGammaGrad = Array.from(await gammaT.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "layernorm_backward",
      caseName: "ln_only",
      inputs: [
        { values: xData, shape: [batch, seqLen, embedDim] },
        { values: lnWeight, shape: [embedDim] },
        { values: lnBias, shape: [embedDim] },
      ],
      options: { normalizedShape: [embedDim] },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;
    const pytorchGammaGrad = pytorchResult[0].grads[1].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteXGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchXGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const xDiff = maxDiff(torchletteXGrad, pytorchXGrad);
    const gammaDiff = maxDiff(torchletteGammaGrad, pytorchGammaGrad);
    console.log(`  x.grad ${status(xDiff)}: max diff = ${xDiff.toExponential(2)}`);
    console.log(`  gamma.grad ${status(gammaDiff)}: max diff = ${gammaDiff.toExponential(2)}`);
  }

  console.log("\nTest 2: LayerNorm + Linear");
  {
    const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });
    const ln = new LayerNorm(api, embedDim, { device: "webgpu" });
    const w = api.tensorFromArray(cAttnWeight, [3 * embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(cAttnBias, [3 * embedDim], { requiresGrad: true, device: "webgpu" });

    // LayerNorm
    const lnOut = ln.forward(x);

    // Linear (matmul with transposed weight)
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const qkv = api.add(lnOut.matmul(wT), b);

    const loss = api.sum(qkv);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());
    const torchletteWGrad = Array.from(await w.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "layernorm_linear_backward",
      caseName: "ln_linear",
      inputs: [
        { values: xData, shape: [batch, seqLen, embedDim] },
        { values: lnWeight, shape: [embedDim] },
        { values: lnBias, shape: [embedDim] },
        { values: cAttnWeight, shape: [3 * embedDim, embedDim] },
        { values: cAttnBias, shape: [3 * embedDim] },
      ],
      options: {},
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;
    const pytorchWGrad = pytorchResult[0].grads[3].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteXGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchXGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const xDiff = maxDiff(torchletteXGrad, pytorchXGrad);
    const wDiff = maxDiff(torchletteWGrad, pytorchWGrad);
    console.log(`  x.grad ${status(xDiff)}: max diff = ${xDiff.toExponential(2)}`);
    console.log(`  w.grad ${status(wDiff)}: max diff = ${wDiff.toExponential(2)}`);
  }

  console.log("\nTest 3: LayerNorm + Attention (full block without residual)");
  {
    const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });
    const ln = new LayerNorm(api, embedDim, { device: "webgpu" });
    const w = api.tensorFromArray(cAttnWeight, [3 * embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(cAttnBias, [3 * embedDim], { requiresGrad: true, device: "webgpu" });

    // LayerNorm
    const lnOut = ln.forward(x);

    // QKV projection
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const qkv = api.add(lnOut.matmul(wT), b);

    // Split Q, K, V
    const qkvSplit = qkv.reshape([batch, seqLen, 3, embedDim]);
    const qkvTransposed = qkvSplit.permute([2, 0, 1, 3]);
    const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);

    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

    const qFlat = qkvFlattened.gather(idx0, { dim: 0 }).reshape([batch, seqLen, embedDim]);
    const kFlat = qkvFlattened.gather(idx1, { dim: 0 }).reshape([batch, seqLen, embedDim]);
    const vFlat = qkvFlattened.gather(idx2, { dim: 0 }).reshape([batch, seqLen, embedDim]);

    const q = qFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = kFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = vFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();

    // Attention computation
    const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
    const scale = 1.0 / Math.sqrt(headDim);
    const scores = q.matmul(kT);
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
    const attnWeights = maskedScores.softmax(-1);
    const attnOutput = attnWeights.matmul(v);

    const loss = api.sum(attnOutput);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());
    const torchletteWGrad = Array.from(await w.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "layernorm_attention_backward",
      caseName: "ln_attn",
      inputs: [
        { values: xData, shape: [batch, seqLen, embedDim] },
        { values: lnWeight, shape: [embedDim] },
        { values: lnBias, shape: [embedDim] },
        { values: cAttnWeight, shape: [3 * embedDim, embedDim] },
        { values: cAttnBias, shape: [3 * embedDim] },
      ],
      options: { embedDim, numHeads },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;
    const pytorchWGrad = pytorchResult[0].grads[3].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteXGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchXGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const xDiff = maxDiff(torchletteXGrad, pytorchXGrad);
    const wDiff = maxDiff(torchletteWGrad, pytorchWGrad);
    console.log(`  x.grad ${status(xDiff)}: max diff = ${xDiff.toExponential(2)}`);
    console.log(`  w.grad ${status(wDiff)}: max diff = ${wDiff.toExponential(2)}`);
  }

  console.log("\nTest 4: LayerNorm + Attention + Residual");
  {
    const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });
    const ln = new LayerNorm(api, embedDim, { device: "webgpu" });
    const w = api.tensorFromArray(cAttnWeight, [3 * embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(cAttnBias, [3 * embedDim], { requiresGrad: true, device: "webgpu" });
    const wProjData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
    const bProjData = new Array(embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
    const wProj = api.tensorFromArray(wProjData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
    const bProj = api.tensorFromArray(bProjData, [embedDim], { requiresGrad: true, device: "webgpu" });

    // LayerNorm
    const lnOut = ln.forward(x);

    // QKV projection
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const qkv = api.add(lnOut.matmul(wT), b);

    // Split Q, K, V
    const qkvSplit = qkv.reshape([batch, seqLen, 3, embedDim]);
    const qkvTransposed = qkvSplit.permute([2, 0, 1, 3]);
    const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);

    const idx0 = api.tensorFromArray([0], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx1 = api.tensorFromArray([1], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();
    const idx2 = api.tensorFromArray([2], [1, 1], { device: "webgpu" }).expand([1, flatTotal]).contiguous();

    const qFlat = qkvFlattened.gather(idx0, { dim: 0 }).reshape([batch, seqLen, embedDim]);
    const kFlat = qkvFlattened.gather(idx1, { dim: 0 }).reshape([batch, seqLen, embedDim]);
    const vFlat = qkvFlattened.gather(idx2, { dim: 0 }).reshape([batch, seqLen, embedDim]);

    const q = qFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = kFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = vFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();

    // Attention computation
    const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
    const scale = 1.0 / Math.sqrt(headDim);
    const scores = q.matmul(kT);
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
    const attnWeights = maskedScores.softmax(-1);
    const attnOutput = attnWeights.matmul(v);

    // Concat heads + output projection
    const attnConcat = attnOutput.permute([0, 2, 1, 3]).contiguous().reshape([batch, seqLen, embedDim]);
    const wProjT = wProj.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const attnProjOut = api.add(attnConcat.matmul(wProjT), bProj);

    // Residual connection
    const residualOut = api.add(x, attnProjOut);

    const loss = api.sum(residualOut);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());
    const torchletteWGrad = Array.from(await w.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "layernorm_attention_residual_backward",
      caseName: "ln_attn_res",
      inputs: [
        { values: xData, shape: [batch, seqLen, embedDim] },
        { values: lnWeight, shape: [embedDim] },
        { values: lnBias, shape: [embedDim] },
        { values: cAttnWeight, shape: [3 * embedDim, embedDim] },
        { values: cAttnBias, shape: [3 * embedDim] },
        { values: wProjData, shape: [embedDim, embedDim] },
        { values: bProjData, shape: [embedDim] },
      ],
      options: { embedDim, numHeads },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;
    const pytorchWGrad = pytorchResult[0].grads[3].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteXGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchXGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const xDiff = maxDiff(torchletteXGrad, pytorchXGrad);
    const wDiff = maxDiff(torchletteWGrad, pytorchWGrad);
    console.log(`  x.grad ${status(xDiff)}: max diff = ${xDiff.toExponential(2)}`);
    console.log(`  w.grad ${status(wDiff)}: max diff = ${wDiff.toExponential(2)}`);

    if (wDiff > 1e-3) {
      // Find where the max diff occurs
      let maxIdx = 0;
      for (let i = 0; i < torchletteWGrad.length; i++) {
        if (Math.abs(torchletteWGrad[i] - pytorchWGrad[i]) > Math.abs(torchletteWGrad[maxIdx] - pytorchWGrad[maxIdx])) {
          maxIdx = i;
        }
      }
      console.log(`  Max diff at index ${maxIdx}:`);
      console.log(`    Torchlette: ${torchletteWGrad[maxIdx]}`);
      console.log(`    PyTorch: ${pytorchWGrad[maxIdx]}`);
      console.log(`  Torchlette w.grad sum: ${torchletteWGrad.reduce((a, b) => a + b, 0)}`);
      console.log(`  PyTorch w.grad sum: ${pytorchWGrad.reduce((a: number, b: number) => a + b, 0)}`);
    }
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
