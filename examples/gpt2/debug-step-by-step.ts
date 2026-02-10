/**
 * Debug Step by Step - Test each backward step separately
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
  console.log("=== Step by Step Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, heads = 2, seqLen = 4, headDim = 4;
  const embedDim = heads * headDim;

  // Common data
  const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
  const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
  const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

  // Step A: What gradient does matmul backward expect?
  // Test: outFlat @ wT -> sum, what's grad wrt outFlat?
  console.log("Step A: 3D matmul backward (outFlat @ wT -> sum)");
  {
    const outFlatData = new Array(batch * seqLen * embedDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const outFlat = api.tensorFromArray(outFlatData, [batch, seqLen, embedDim], { requiresGrad: true, device: "webgpu" });
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { requiresGrad: true, device: "webgpu" });

    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteGrad = Array.from(await outFlat.grad!.cpu());
    console.log(`  outFlat.grad first 16: ${torchletteGrad.slice(0, 16).map(v => v.toFixed(4)).join(", ")}`);

    // Check: is this uniform across seq positions?
    const pos0 = torchletteGrad.slice(0, embedDim);
    const pos1 = torchletteGrad.slice(embedDim, embedDim * 2);
    const pos2 = torchletteGrad.slice(embedDim * 2, embedDim * 3);
    const allSame = pos0.every((v, i) => Math.abs(v - pos1[i]) < 1e-6 && Math.abs(v - pos2[i]) < 1e-6);
    console.log(`  Gradient uniform across seq positions: ${allSame ? "YES" : "NO"}`);
  }

  // Step B: What gradient does reshape backward produce?
  // Gradient comes in as [batch, seq, embed], reshapes to [batch, seq, heads, head_dim]
  console.log("\nStep B: reshape backward (view only, no change expected)");
  {
    // Input with shape [batch, seq, heads, head_dim], reshape to [batch, seq, embed]
    const outPermData = new Array(batch * seqLen * heads * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const outPerm = api.tensorFromArray(outPermData, [batch, seqLen, heads, headDim], { requiresGrad: true, device: "webgpu" });

    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { device: "webgpu" });
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteGrad = Array.from(await outPerm.grad!.cpu());
    console.log(`  outPerm.grad first 16: ${torchletteGrad.slice(0, 16).map(v => v.toFixed(4)).join(", ")}`);

    // Check: is this uniform across seq positions?
    // Shape is [batch, seq, heads, head_dim] = [2, 4, 2, 4]
    // One seq position is heads*headDim = 8 elements
    const seqSize = heads * headDim;
    const pos0 = torchletteGrad.slice(0, seqSize);
    const pos1 = torchletteGrad.slice(seqSize, seqSize * 2);
    const allSame = pos0.every((v, i) => Math.abs(v - pos1[i]) < 1e-6);
    console.log(`  Gradient uniform across seq positions: ${allSame ? "YES" : "NO"}`);
  }

  // Step C: What gradient does permute backward produce?
  // Input [batch, heads, seq, head_dim], permute to [batch, seq, heads, head_dim]
  console.log("\nStep C: permute backward (critical test)");
  {
    const outData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
    const out = api.tensorFromArray(outData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const outPerm = out.permute([0, 2, 1, 3]).contiguous();
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { device: "webgpu" });
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const torchletteGrad = Array.from(await out.grad!.cpu());
    console.log(`  out.grad first 16: ${torchletteGrad.slice(0, 16).map(v => v.toFixed(4)).join(", ")}`);

    // Shape is [batch, heads, seq, head_dim] = [2, 2, 4, 4]
    // Check: are head 0 and head 1 gradients the same pattern?
    const headSize = seqLen * headDim;  // 16
    const head0 = torchletteGrad.slice(0, headSize);
    const head1 = torchletteGrad.slice(headSize, headSize * 2);
    const sameBetweenHeads = head0.every((v, i) => Math.abs(v - head1[i]) < 1e-6);
    console.log(`  Gradient same between heads: ${sameBetweenHeads ? "YES" : "NO"}`);

    // Check: within a head, is grad uniform across seq positions?
    const seq0 = head0.slice(0, headDim);
    const seq1 = head0.slice(headDim, headDim * 2);
    const sameWithinHead = seq0.every((v, i) => Math.abs(v - seq1[i]) < 1e-6);
    console.log(`  Gradient same across seq within head: ${sameWithinHead ? "YES" : "NO"}`);
  }

  // Step D: Full path - 4D matmul -> permute -> reshape -> matmul -> sum
  console.log("\nStep D: Full path with 4D matmul");
  {
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
      caseName: "step_d",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    console.log(`  Torchlette attn.grad first 16: ${torchletteGrad.slice(0, 16).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  PyTorch attn.grad first 16: ${pytorchGrad.slice(0, 16).map((v: number) => v.toFixed(4)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);

    // Check attn gradient pattern
    // Shape is [batch, heads, seq, seq] = [2, 2, 4, 4]
    // First row (attn[0,0,0,:]) vs second row (attn[0,0,1,:])
    const row0 = torchletteGrad.slice(0, seqLen);
    const row1 = torchletteGrad.slice(seqLen, seqLen * 2);
    const pyRow0 = pytorchGrad.slice(0, seqLen);
    const pyRow1 = pytorchGrad.slice(seqLen, seqLen * 2);

    console.log(`  Torchlette row0: ${row0.map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  Torchlette row1: ${row1.map(v => v.toFixed(4)).join(", ")}`);
    console.log(`  PyTorch row0: ${pyRow0.map((v: number) => v.toFixed(4)).join(", ")}`);
    console.log(`  PyTorch row1: ${pyRow1.map((v: number) => v.toFixed(4)).join(", ")}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
