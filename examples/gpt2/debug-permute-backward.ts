/**
 * Debug Permute Backward - Test permute backward with non-uniform gradients
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
  console.log("=== Permute Backward Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, heads = 2, seqLen = 4, headDim = 4;
  const embedDim = heads * headDim;

  // Test 1: Simple permute -> sum (uniform grad)
  console.log("Test 1: permute -> sum (uniform gradient)");
  {
    const xData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const x = api.tensorFromArray(xData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const xPerm = x.permute([0, 2, 1, 3]).contiguous();  // [batch, seq, heads, head_dim]
    const loss = api.sum(xPerm);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "permute_sum_backward",
      caseName: "perm_sum",
      inputs: [{ values: xData, shape: [batch, heads, seqLen, headDim] }],
      options: { dims: [0, 2, 1, 3] },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 2: permute -> mul(weights) -> sum (non-uniform grad)
  console.log("\nTest 2: permute -> mul -> sum (non-uniform gradient)");
  {
    const xData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(batch * seqLen * heads * headDim).fill(0).map((_, i) => (i + 1) * 0.1);  // weights to multiply

    const x = api.tensorFromArray(xData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [batch, seqLen, heads, headDim], { device: "webgpu" });

    const xPerm = x.permute([0, 2, 1, 3]).contiguous();  // [batch, seq, heads, head_dim]
    const xMul = api.mul(xPerm, w);  // elementwise mul
    const loss = api.sum(xMul);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "permute_mul_sum_backward",
      caseName: "perm_mul_sum",
      inputs: [
        { values: xData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [batch, seqLen, heads, headDim] },
      ],
      options: { dims: [0, 2, 1, 3] },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchGrad.slice(0, 8).map((v: number) => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 3: permute -> reshape -> matmul -> sum (the problematic pattern)
  console.log("\nTest 3: permute -> reshape -> matmul -> sum (problematic pattern)");
  {
    const xData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const x = api.tensorFromArray(xData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    const xPerm = x.permute([0, 2, 1, 3]).contiguous();
    const xFlat = xPerm.reshape([batch, seqLen, embedDim]);
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const out = xFlat.matmul(wT);
    const loss = api.sum(out);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "permute_reshape_matmul_backward",
      caseName: "perm_mm",
      inputs: [
        { values: xData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchGrad.slice(0, 8).map((v: number) => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 4: 4D matmul -> permute -> reshape -> sum (passes in previous test)
  console.log("\nTest 4: 4D matmul -> permute -> reshape -> sum (should pass)");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);
    const outPerm = out.permute([0, 2, 1, 3]).contiguous();
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const loss = api.sum(outFlat);
    await loss.backward();

    const torchletteGrad = Array.from(await attn.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_v_matmul_permute_reshape_backward",
      caseName: "attn_mm_perm_reshape",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
      ],
      options: { embedDim },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 5: Full problematic path
  console.log("\nTest 5: 4D matmul -> permute -> reshape -> 3D matmul -> sum (problematic)");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);
    const outPerm = out.permute([0, 2, 1, 3]).contiguous();
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteGrad = Array.from(await attn.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_backward",
      caseName: "attn_full",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    console.log(`  Torchlette x.grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch x.grad first 8: ${pytorchGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
