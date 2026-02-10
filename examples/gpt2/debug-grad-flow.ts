/**
 * Debug Gradient Flow - Check the actual gradient flowing into 4D matmul backward
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
  console.log("=== Gradient Flow Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const batch = 2, heads = 2, seqLen = 4, headDim = 4;
  const embedDim = heads * headDim;

  // Common data
  const outData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
  const vData = new Array(batch * heads * seqLen * headDim).fill(0).map((_, i) => (i + 1) * 0.01);
  const wData = new Array(embedDim * embedDim).fill(0).map((_, i) => (i + 1) * 0.001);

  // Test: What gradient flows through the chain?
  // We'll use a fixed "out" tensor (as if it came from attn@v) and trace backward
  console.log("Test: Gradient flow through permute -> reshape -> matmul");
  {
    // Create "out" as a leaf tensor (simulating output of attn@v)
    const out = api.tensorFromArray(outData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    const outPerm = out.permute([0, 2, 1, 3]).contiguous();
    const outFlat = outPerm.reshape([batch, seqLen, embedDim]);
    const w = api.tensorFromArray(wData, [embedDim, embedDim], { device: "webgpu" });
    const wT = w.transpose({ dim0: 0, dim1: 1 }).contiguous();
    const proj = outFlat.matmul(wT);
    const loss = api.sum(proj);
    await loss.backward();

    const gradOut = Array.from(await out.grad!.cpu());

    console.log(`  Shape of out.grad: [${out.grad!.shape.join(", ")}]`);
    console.log(`  out.grad first 32 values:`);

    // Print in a way that shows the structure
    // Shape is [batch=2, heads=2, seq=4, head_dim=4]
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < heads; h++) {
        const startIdx = (b * heads * seqLen * headDim) + (h * seqLen * headDim);
        console.log(`    batch=${b}, head=${h}:`);
        for (let s = 0; s < seqLen; s++) {
          const rowStart = startIdx + s * headDim;
          const row = gradOut.slice(rowStart, rowStart + headDim);
          console.log(`      seq=${s}: ${row.map(v => v.toFixed(4)).join(", ")}`);
        }
      }
    }

    // Check uniformity
    console.log("\n  Uniformity checks:");
    // Within head 0 of batch 0, are all seq positions the same?
    const h0 = [];
    for (let s = 0; s < seqLen; s++) {
      h0.push(gradOut.slice(s * headDim, (s + 1) * headDim));
    }
    const h0Uniform = h0.slice(1).every((row, i) =>
      row.every((v, j) => Math.abs(v - h0[0][j]) < 1e-6)
    );
    console.log(`    All seq positions same within head 0: ${h0Uniform ? "YES" : "NO"}`);

    // Are head 0 and head 1 the same?
    const head0Start = 0;
    const head1Start = seqLen * headDim;
    let headsSame = true;
    for (let i = 0; i < seqLen * headDim; i++) {
      if (Math.abs(gradOut[head0Start + i] - gradOut[head1Start + i]) > 1e-6) {
        headsSame = false;
        break;
      }
    }
    console.log(`    Head 0 and Head 1 identical: ${headsSame ? "YES" : "NO"}`);
  }

  // Now test: if we use this gradient directly with a 4D matmul backward, what happens?
  console.log("\n\nTest: Direct 4D matmul backward with a known-uniform gradient");
  {
    // Create uniform gradient: same values for all seq positions
    const uniformGradData = new Array(batch * heads * seqLen * headDim);
    for (let b = 0; b < batch; b++) {
      for (let h = 0; h < heads; h++) {
        for (let s = 0; s < seqLen; s++) {
          for (let d = 0; d < headDim; d++) {
            const idx = b * heads * seqLen * headDim + h * seqLen * headDim + s * headDim + d;
            // Uniform across seq: value depends only on (b, h, d)
            uniformGradData[idx] = (h * headDim + d + 1) * 0.1;
          }
        }
      }
    }

    console.log("  Uniform gradient (same across seq positions):");
    for (let s = 0; s < Math.min(2, seqLen); s++) {
      const row = uniformGradData.slice(s * headDim, (s + 1) * headDim);
      console.log(`    seq=${s}: ${row.map(v => v.toFixed(4)).join(", ")}`);
    }

    // Compute grad_attn = uniform_grad @ v.T manually
    // This should give us uniform rows!
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
    const attn = api.tensorFromArray(attnData, [batch, heads, seqLen, seqLen], { requiresGrad: true, device: "webgpu" });
    const v = api.tensorFromArray(vData, [batch, heads, seqLen, headDim], { requiresGrad: true, device: "webgpu" });

    // Forward pass
    const out = attn.matmul(v);
    const loss = api.sum(out);
    await loss.backward();

    const torchletteGrad = Array.from(await attn.grad!.cpu());

    console.log("\n  With sum(out) as loss (uniform grad):");
    console.log(`    attn.grad row 0: ${torchletteGrad.slice(0, seqLen).map(v => v.toFixed(4)).join(", ")}`);
    console.log(`    attn.grad row 1: ${torchletteGrad.slice(seqLen, seqLen*2).map(v => v.toFixed(4)).join(", ")}`);

    // Check if rows are the same
    const row0 = torchletteGrad.slice(0, seqLen);
    const row1 = torchletteGrad.slice(seqLen, seqLen * 2);
    const rowsSame = row0.every((v, i) => Math.abs(v - row1[i]) < 1e-6);
    console.log(`    Rows are identical: ${rowsSame ? "YES" : "NO"}`);
  }

  // Test the full chain again with very detailed output
  console.log("\n\nTest: Full chain with detailed inspection");
  {
    const attnData = new Array(batch * heads * seqLen * seqLen).fill(0).map((_, i) => (i % 10 + 1) * 0.01);
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

    console.log("  Torchlette attn.grad structure (batch=0, head=0):");
    for (let s = 0; s < seqLen; s++) {
      const row = torchletteGrad.slice(s * seqLen, (s + 1) * seqLen);
      console.log(`    row ${s}: ${row.map(v => v.toFixed(4)).join(", ")}`);
    }

    // Compare with PyTorch
    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "attn_permute_matmul_backward",
      caseName: "full_chain",
      inputs: [
        { values: attnData, shape: [batch, heads, seqLen, seqLen] },
        { values: vData, shape: [batch, heads, seqLen, headDim] },
        { values: wData, shape: [embedDim, embedDim] },
      ],
      options: { batch, heads, seqLen, headDim },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;

    console.log("\n  PyTorch attn.grad structure (batch=0, head=0):");
    for (let s = 0; s < seqLen; s++) {
      const row = pytorchGrad.slice(s * seqLen, (s + 1) * seqLen);
      console.log(`    row ${s}: ${row.map((v: number) => v.toFixed(4)).join(", ")}`);
    }
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
