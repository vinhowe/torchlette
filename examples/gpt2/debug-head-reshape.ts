/**
 * Debug Head Reshape - Test gradient flow through head concat + c_proj
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
  console.log("=== Head Reshape Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, heads = 2, seqLen = 4, headDim = 4;
  const embedDim = heads * headDim;

  // Test 1: Just permute + reshape
  console.log("Test 1: permute + reshape (head concat)");
  {
    // Input shape [batch, heads, seq, head_dim]
    const xData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const x = api.tensorFromArray(xData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    // Permute to [batch, seq, heads, head_dim] then reshape to [batch, seq, embed]
    const xPermuted = x.permute([0, 2, 1, 3]).contiguous();
    const xReshaped = xPermuted.reshape([batch, seqLen, embedDim]);

    const loss = api.sum(xReshaped);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());

    // PyTorch reference
    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "head_concat_backward",
      caseName: "head_concat",
      inputs: [{ values: xData, shape: [batch, heads, seqLen, headDim] }],
      options: { embedDim },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    console.log(`  Torchlette first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  PyTorch first 8: ${pytorchGrad.slice(0, 8).map((v: number) => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 2: permute + reshape + linear
  console.log("\nTest 2: head concat + linear projection");
  {
    const xData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
    const bData = new Array(embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const x = api.tensorFromArray(xData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [embedDim], { requiresGrad: true, device: "webgpu" });

    // Concat heads
    const xPermuted = x.permute([0, 2, 1, 3]).contiguous();
    const xReshaped = xPermuted.reshape([batch, seqLen, embedDim]);

    // Linear projection
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const output = api.add(xReshaped.matmul(wT), b);

    const loss = api.sum(output);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());
    const torchletteWGrad = Array.from(await w.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "head_concat_linear_backward",
      caseName: "head_concat_linear",
      inputs: [
        { values: xData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
        { values: bData, shape: [embedDim] },
      ],
      options: { embedDim },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;
    const pytorchWGrad = pytorchResult[0].grads[1].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteXGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchXGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const xDiff = maxDiff(torchletteXGrad, pytorchXGrad);
    const wDiff = maxDiff(torchletteWGrad, pytorchWGrad);
    console.log(`  x.grad ${status(xDiff)}: max diff = ${xDiff.toExponential(2)}`);
    console.log(`  w.grad ${status(wDiff)}: max diff = ${wDiff.toExponential(2)}`);
  }

  // Test 3: matmul + permute + reshape + linear (like attention output path)
  console.log("\nTest 3: matmul (attn@v) + head concat + linear");
  {
    // attn_weights [batch, heads, seq, seq] @ v [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map(() => Math.random());
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);
    const bData = new Array(embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [embedDim], { requiresGrad: true, device: "webgpu" });

    // attn @ v
    const attnOut = attn.matmul(v);  // [batch, heads, seq, head_dim]

    // Concat heads
    const attnConcat = attnOut.permute([0, 2, 1, 3]).contiguous().reshape([batch, seqLen, embedDim]);

    // Linear projection
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const output = api.add(attnConcat.matmul(wT), b);

    const loss = api.sum(output);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());
    const torchletteVGrad = Array.from(await v.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_v_head_concat_linear_backward",
      caseName: "attn_v_head_concat_linear",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
        { values: bData, shape: [embedDim] },
      ],
      options: { embedDim },
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;
    const pytorchVGrad = pytorchResult[0].grads[1].values;

    console.log(`  Torchlette attn.grad first 8: ${torchletteAttnGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch attn.grad first 8: ${pytorchAttnGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const attnDiff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    const vDiff = maxDiff(torchletteVGrad, pytorchVGrad);
    console.log(`  attn.grad ${status(attnDiff)}: max diff = ${attnDiff.toExponential(2)}`);
    console.log(`  v.grad ${status(vDiff)}: max diff = ${vDiff.toExponential(2)}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
