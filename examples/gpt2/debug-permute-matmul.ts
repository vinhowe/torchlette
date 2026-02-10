/**
 * Debug Permute + Matmul - Test gradient flow through permute followed by matmul
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
  console.log("=== Permute + Matmul Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Test 1: permute then matmul (the attention output path)
  console.log("Test 1: [batch,heads,seq,head_dim] permute to [batch,seq,heads,head_dim] -> reshape -> matmul");
  {
    const batch = 2, heads = 2, seqLen = 4, headDim = 4;
    const embedDim = heads * headDim;

    const xData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const x = api.tensorFromArray(xData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    // x: [batch, heads, seq, head_dim] -> permute -> [batch, seq, heads, head_dim]
    const xPerm = x.permute([0, 2, 1, 3]).contiguous();
    // reshape -> [batch, seq, embed]
    const xFlat = xPerm.reshape([batch, seqLen, embedDim]);
    // matmul with w.T
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const out = xFlat.matmul(wT);

    const loss = api.sum(out);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "permute_reshape_matmul_backward",
      caseName: "perm_mm",
      inputs: [
        { values: xData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;

    console.log(`  Torchlette x.grad first 16: ${torchletteXGrad.slice(0, 16).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 16: ${pytorchXGrad.slice(0, 16).map((v: number) => v.toFixed(6)).join(", ")}`);

    const diff = maxDiff(torchletteXGrad, pytorchXGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 2: Same but with matmul before permute (just to verify matmul->permute is ok)
  console.log("\nTest 2: matmul -> permute (reverse order)");
  {
    const batch = 2, heads = 2, seqLen = 4, headDim = 4;
    const embedDim = heads * headDim;

    const xData = new Array(batch * seqLen * embedDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const x = api.tensorFromArray(xData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    // matmul first
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const xMm = x.matmul(wT);  // [batch, seq, embed]
    // reshape to [batch, seq, heads, head_dim]
    const xReshaped = xMm.reshape([batch, seqLen, heads, headDim]);
    // permute to [batch, heads, seq, head_dim]
    const xPerm = xReshaped.permute([0, 2, 1, 3]).contiguous();

    const loss = api.sum(xPerm);
    await loss.backward();

    const torchletteXGrad = Array.from(await x.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "matmul_reshape_permute_backward",
      caseName: "mm_perm",
      inputs: [
        { values: xData, shape: [batch, seqLen, embedDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchXGrad = pytorchResult[0].grads[0].values;

    const diff = maxDiff(torchletteXGrad, pytorchXGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);

    if (diff > 1e-4) {
      console.log(`  Torchlette x.grad first 16: ${torchletteXGrad.slice(0, 16).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch x.grad first 16: ${pytorchXGrad.slice(0, 16).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 3: Chain: 4D matmul -> permute -> reshape -> matmul (the full attention output path)
  console.log("\nTest 3: 4D matmul -> permute -> reshape -> matmul (attention output path)");
  {
    const batch = 2, heads = 2, seqLen = 4, headDim = 4;
    const embedDim = heads * headDim;

    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    // attn @ v -> [batch, heads, seq, head_dim]
    const attnOut = attn.matmul(v);
    // permute -> [batch, seq, heads, head_dim]
    const attnPerm = attnOut.permute([0, 2, 1, 3]).contiguous();
    // reshape -> [batch, seq, embed]
    const attnFlat = attnPerm.reshape([batch, seqLen, embedDim]);
    // matmul with w.T
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const out = attnFlat.matmul(wT);

    const loss = api.sum(out);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());
    const torchletteVGrad = Array.from(await v.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_backward",
      caseName: "attn_perm_mm",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;
    const pytorchVGrad = pytorchResult[0].grads[1].values;

    const attnDiff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    const vDiff = maxDiff(torchletteVGrad, pytorchVGrad);
    console.log(`  attn.grad ${status(attnDiff)}: max diff = ${attnDiff.toExponential(2)}`);
    console.log(`  v.grad ${status(vDiff)}: max diff = ${vDiff.toExponential(2)}`);

    if (attnDiff > 1e-4) {
      console.log(`  Torchlette attn.grad first 16: ${torchletteAttnGrad.slice(0, 16).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch attn.grad first 16: ${pytorchAttnGrad.slice(0, 16).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
