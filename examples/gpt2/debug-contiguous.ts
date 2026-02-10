/**
 * Debug Contiguous - Test if calling .contiguous() affects gradients
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
  console.log("=== Contiguous Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, heads = 2, seqLen = 4, headDim = 4;
  const embedDim = heads * headDim;

  const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
  const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
  const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

  // Test 1: With .contiguous() after permute (current failing case)
  console.log("Test 1: With .contiguous() after permute");
  {
    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);  // [batch, heads, seq, head_dim]
    const outPerm = out.permute([0, 2, 1, 3]).contiguous();  // <-- With contiguous
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_backward",
      caseName: "with_contig",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;

    const diff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);

    if (diff > 1e-4) {
      console.log(`  Torchlette first 8: ${torchletteAttnGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch first 8: ${pytorchAttnGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 2: Without .contiguous() after permute (skip reshape since that needs contiguous)
  console.log("\nTest 2: Without .contiguous(), reshape to force contiguous");
  {
    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    const out = attn.matmul(v);  // [batch, heads, seq, head_dim]
    const outPerm = out.permute([0, 2, 1, 3]);  // <-- No contiguous
    // reshape should force contiguous internally if needed
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteAttnGrad = Array.from(await attn.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_backward",
      caseName: "no_contig",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchAttnGrad = pytorchResult[0].grads[0].values;

    const diff = maxDiff(torchletteAttnGrad, pytorchAttnGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);

    if (diff > 1e-4) {
      console.log(`  Torchlette first 8: ${torchletteAttnGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch first 8: ${pytorchAttnGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 3: Check if the matmul output is contiguous
  console.log("\nTest 3: Checking matmul output strides");
  {
    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { device: "webgpu" });

    const out = attn.matmul(v);  // Should be contiguous [batch, heads, seq, head_dim]
    console.log(`  matmul output shape: [${out.shape.join(", ")}]`);
    console.log(`  matmul output is contiguous: ${out.isContiguous()}`);

    const outPerm = out.permute([0, 2, 1, 3]);
    console.log(`  after permute shape: [${outPerm.shape.join(", ")}]`);
    console.log(`  after permute is contiguous: ${outPerm.isContiguous()}`);

    const outContig = outPerm.contiguous();
    console.log(`  after contiguous is contiguous: ${outContig.isContiguous()}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
