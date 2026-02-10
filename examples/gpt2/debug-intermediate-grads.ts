/**
 * Debug Intermediate Gradients - Check gradients at each step
 */
import { Torchlette, Tensor } from "../../src/frontend";
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
  console.log("=== Intermediate Gradient Debug ===\n");

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

  // Test: Track intermediate tensor gradients
  console.log("Full path with retainGrad on intermediates:");
  {
    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    // Forward with retainGrad
    const out = attn.matmul(v);  // [batch, heads, seq, head_dim]
    out.retainGrad();

    const outPerm = out.permute([0, 2, 1, 3]).contiguous();  // [batch, seq, heads, head_dim]
    outPerm.retainGrad();

    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);  // [batch, seq, embed]
    outFlat.retainGrad();

    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);  // [batch, seq, embed]
    proj.retainGrad();

    const loss = api.sum(proj);
    await loss.backward();

    // Get all gradients
    const gradProj = Array.from(await proj.grad!.cpu());
    const gradOutFlat = Array.from(await outFlat.grad!.cpu());
    const gradOutPerm = Array.from(await outPerm.grad!.cpu());
    const gradOut = Array.from(await out.grad!.cpu());
    const gradAttn = Array.from(await attn.grad!.cpu());

    console.log(`  proj.grad shape: [${proj.grad!.shape.join(",")}]`);
    console.log(`  proj.grad first 8: ${gradProj.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  proj.grad should be all 1s since loss = sum(proj)\n`);

    console.log(`  outFlat.grad shape: [${outFlat.grad!.shape.join(",")}]`);
    console.log(`  outFlat.grad first 8: ${gradOutFlat.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  outFlat.grad = proj.grad @ w (matmul backward)\n`);

    console.log(`  outPerm.grad shape: [${outPerm.grad!.shape.join(",")}]`);
    console.log(`  outPerm.grad first 8: ${gradOutPerm.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  outPerm.grad = reshape(outFlat.grad, [b,s,h,hd])\n`);

    console.log(`  out.grad shape: [${out.grad!.shape.join(",")}]`);
    console.log(`  out.grad first 8: ${gradOut.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  out.grad = permute(outPerm.grad, [0,2,1,3])\n`);

    console.log(`  attn.grad shape: [${attn.grad!.shape.join(",")}]`);
    console.log(`  attn.grad first 8: ${gradAttn.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`  attn.grad = out.grad @ v.T\n`);

    // Compare with PyTorch
    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_intermediate",
      caseName: "inter",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pyGradOut = pytorchResult[0].grads[0].values;
    const pyGradAttn = pytorchResult[0].grads[1].values;

    console.log("PyTorch comparison:");
    console.log(`  PyTorch out.grad first 8: ${pyGradOut.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    console.log(`  PyTorch attn.grad first 8: ${pyGradAttn.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const outDiff = maxDiff(gradOut, pyGradOut);
    const attnDiff = maxDiff(gradAttn, pyGradAttn);
    console.log(`\n  out.grad ${status(outDiff)}: max diff = ${outDiff.toExponential(2)}`);
    console.log(`  attn.grad ${status(attnDiff)}: max diff = ${attnDiff.toExponential(2)}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
