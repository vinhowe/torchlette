/**
 * Debug Softmax Gradient - Verify softmax backward matches PyTorch
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
  console.log("=== Softmax Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Test 1: Simple 2D softmax
  console.log("Test 1: 2D softmax (batch, features)");
  {
    const xData = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const x = api.tensorFromArray(xData, [3, 4], { requiresGrad: true, device: "webgpu" });

    const softmaxOut = x.softmax(-1);
    const loss = api.sum(softmaxOut);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());
    console.log(`  Torchlette grad: ${torchletteGrad.slice(0, 4).map(v => v.toFixed(6)).join(", ")}`);

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "softmax_backward",
      caseName: "softmax2d",
      inputs: [{ values: xData, shape: [3, 4] }],
      options: { dim: -1 },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;
    console.log(`  PyTorch grad: ${pytorchGrad.slice(0, 4).map((v: number) => v.toFixed(6)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 2: 4D softmax (attention scores pattern)
  console.log("\nTest 2: 4D softmax [batch, heads, seq, seq]");
  {
    const batch = 2, heads = 2, seq = 4;
    const xData = new Array(batch * heads * seq * seq).fill(0).map((_, i) => (i % 10) * 0.1);
    const x = api.tensorFromArray(xData, [batch, heads, seq, seq], { requiresGrad: true, device: "webgpu" });

    const softmaxOut = x.softmax(-1);
    const loss = api.sum(softmaxOut);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());
    console.log(`  Torchlette grad first 8: ${torchletteGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "softmax_backward",
      caseName: "softmax4d",
      inputs: [{ values: xData, shape: [batch, heads, seq, seq] }],
      options: { dim: -1 },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;
    console.log(`  PyTorch grad first 8: ${pytorchGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  // Test 3: Softmax with masked values (causal attention pattern)
  console.log("\nTest 3: Softmax with -inf mask (causal pattern)");
  {
    const seq = 4;
    const maskData = new Array(seq * seq);
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < seq; j++) {
        maskData[i * seq + j] = j <= i ? (i + j) * 0.1 : -1e9;
      }
    }

    const x = api.tensorFromArray(maskData, [1, 1, seq, seq], { requiresGrad: true, device: "webgpu" });
    const softmaxOut = x.softmax(-1);
    const loss = api.sum(softmaxOut);
    await loss.backward();

    const torchletteGrad = Array.from(await x.grad!.cpu());
    console.log(`  Torchlette grad first row: ${torchletteGrad.slice(0, seq).map(v => v.toFixed(6)).join(", ")}`);

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "softmax_backward",
      caseName: "softmax_masked",
      inputs: [{ values: maskData, shape: [1, 1, seq, seq] }],
      options: { dim: -1 },
    }]);
    const pytorchGrad = pytorchResult[0].grads[0].values;
    console.log(`  PyTorch grad first row: ${pytorchGrad.slice(0, seq).map((v: number) => v.toFixed(6)).join(", ")}`);

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    console.log(`  ${status(diff)}: max diff = ${diff.toExponential(2)}`);
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
