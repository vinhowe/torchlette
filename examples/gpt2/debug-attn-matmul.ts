/**
 * Debug Attention Matmul - Isolate the 4D matmul + permute issue
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
  console.log("=== Attention Matmul Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, heads = 2, seqLen = 4, headDim = 4;
  const embedDim = heads * headDim;

  // Test 1: attn @ v -> sum (no permute)
  console.log("Test 1: 4D matmul only (attn @ v -> sum)");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);
    const loss = api.sum(out);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());
    const torchletteVGrad = Array.from(await v.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_v_matmul_backward",
      caseName: "attn_mm",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
      ],
      options: {},
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;
    const pytorchVGrad = pytorchResult[0].grads[1].values;

    const attnDiff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    const vDiff = maxDiff(torchletteVGrad, pytorchVGrad);
    console.log(`  attn.grad ${status(attnDiff)}: max diff = ${attnDiff.toExponential(2)}`);
    console.log(`  v.grad ${status(vDiff)}: max diff = ${vDiff.toExponential(2)}`);
  }

  // Test 2: attn @ v -> permute -> sum (no final matmul)
  console.log("\nTest 2: 4D matmul + permute (attn @ v -> permute -> sum)");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);  // [batch, heads, seq, head_dim]
    const outPerm = out.permute([0, 2, 1, 3]).contiguous();  // [batch, seq, heads, head_dim]
    const loss = api.sum(outPerm);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());
    const torchletteVGrad = Array.from(await v.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_v_matmul_permute_backward",
      caseName: "attn_mm_perm",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
      ],
      options: {},
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;
    const pytorchVGrad = pytorchResult[0].grads[1].values;

    const attnDiff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    const vDiff = maxDiff(torchletteVGrad, pytorchVGrad);
    console.log(`  attn.grad ${status(attnDiff)}: max diff = ${attnDiff.toExponential(2)}`);
    console.log(`  v.grad ${status(vDiff)}: max diff = ${vDiff.toExponential(2)}`);

    if (attnDiff > 1e-4 || vDiff > 1e-4) {
      console.log(`  Torchlette attn.grad first 8: ${torchletteAttnGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch attn.grad first 8: ${pytorchAttnGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 3: attn @ v -> permute -> reshape -> sum (add reshape)
  console.log("\nTest 3: 4D matmul + permute + reshape (attn @ v -> permute -> reshape -> sum)");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);  // [batch, heads, seq, head_dim]
    const outPerm = out.permute([0, 2, 1, 3]).contiguous();  // [batch, seq, heads, head_dim]
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);  // [batch, seq, embed]
    const loss = api.sum(outFlat);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());
    const torchletteVGrad = Array.from(await v.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_v_matmul_permute_reshape_backward",
      caseName: "attn_mm_perm_reshape",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
      ],
      options: { embedDim },
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;
    const pytorchVGrad = pytorchResult[0].grads[1].values;

    const attnDiff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    const vDiff = maxDiff(torchletteVGrad, pytorchVGrad);
    console.log(`  attn.grad ${status(attnDiff)}: max diff = ${attnDiff.toExponential(2)}`);
    console.log(`  v.grad ${status(vDiff)}: max diff = ${vDiff.toExponential(2)}`);

    if (attnDiff > 1e-4 || vDiff > 1e-4) {
      console.log(`  Torchlette attn.grad first 8: ${torchletteAttnGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch attn.grad first 8: ${pytorchAttnGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 4: Full path - attn @ v -> permute -> reshape -> matmul -> sum
  console.log("\nTest 4: Full path (attn @ v -> permute -> reshape -> matmul -> sum)");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);  // [batch, heads, seq, head_dim]
    const outPerm = out.permute([0, 2, 1, 3]).contiguous();  // [batch, seq, heads, head_dim]
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);  // [batch, seq, embed]
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);  // [batch, seq, embed]
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());
    const torchletteVGrad = Array.from(await v.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_backward",
      caseName: "attn_perm_mm_full",
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

    if (attnDiff > 1e-4 || vDiff > 1e-4) {
      console.log(`  Torchlette attn.grad first 8: ${torchletteAttnGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch attn.grad first 8: ${pytorchAttnGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
