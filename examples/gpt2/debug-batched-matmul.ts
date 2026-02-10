/**
 * Debug Batched Matmul - Test gradient flow through 4D matmul
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
  console.log("=== Batched Matmul Gradient Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Test 1: Simple 2D matmul
  console.log("Test 1: 2D matmul [4,4] @ [4,4]");
  {
    const aData = new Array(16).fill(0).map((_, i) => (i + 1) * 0.1);
    const bData = new Array(16).fill(0).map((_, i) => (i + 1) * 0.01);

    const a = api.tensorFromArray(aData, [4, 4], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [4, 4], { requiresGrad: true, device: "webgpu" });

    const c = a.matmul(b);
    const loss = api.sum(c);
    await loss.backward();

    const torchletteAGrad = Array.from(await a.grad!.cpu());
    const torchletteBGrad = Array.from(await b.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "matmul_2d_backward",
      caseName: "mm2d",
      inputs: [
        { values: aData, shape: [4, 4] },
        { values: bData, shape: [4, 4] },
      ],
      options: {},
    }]);
    const pytorchAGrad = pytorchResult[0].grads[0].values;
    const pytorchBGrad = pytorchResult[0].grads[1].values;

    const aDiff = maxDiff(torchletteAGrad, pytorchAGrad);
    const bDiff = maxDiff(torchletteBGrad, pytorchBGrad);
    console.log(`  a.grad ${status(aDiff)}: max diff = ${aDiff.toExponential(2)}`);
    console.log(`  b.grad ${status(bDiff)}: max diff = ${bDiff.toExponential(2)}`);
  }

  // Test 2: 3D batched matmul
  console.log("\nTest 2: 3D batched matmul [2,4,4] @ [2,4,4]");
  {
    const aData = new Array(32).fill(0).map((_, i) => (i + 1) * 0.1);
    const bData = new Array(32).fill(0).map((_, i) => (i + 1) * 0.01);

    const a = api.tensorFromArray(aData, [2, 4, 4], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [2, 4, 4], { requiresGrad: true, device: "webgpu" });

    const c = a.matmul(b);
    const loss = api.sum(c);
    await loss.backward();

    const torchletteAGrad = Array.from(await a.grad!.cpu());
    const torchletteBGrad = Array.from(await b.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "matmul_3d_backward",
      caseName: "mm3d",
      inputs: [
        { values: aData, shape: [2, 4, 4] },
        { values: bData, shape: [2, 4, 4] },
      ],
      options: {},
    }]);
    const pytorchAGrad = pytorchResult[0].grads[0].values;
    const pytorchBGrad = pytorchResult[0].grads[1].values;

    const aDiff = maxDiff(torchletteAGrad, pytorchAGrad);
    const bDiff = maxDiff(torchletteBGrad, pytorchBGrad);
    console.log(`  a.grad ${status(aDiff)}: max diff = ${aDiff.toExponential(2)}`);
    console.log(`  b.grad ${status(bDiff)}: max diff = ${bDiff.toExponential(2)}`);

    if (aDiff > 1e-4 || bDiff > 1e-4) {
      console.log(`  Torchlette a.grad first 8: ${torchletteAGrad.slice(0, 8).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch a.grad first 8: ${pytorchAGrad.slice(0, 8).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 3: 4D batched matmul
  console.log("\nTest 3: 4D batched matmul [2,2,4,4] @ [2,2,4,4]");
  {
    const aData = new Array(64).fill(0).map((_, i) => (i + 1) * 0.01);
    const bData = new Array(64).fill(0).map((_, i) => (i + 1) * 0.01);

    const a = api.tensorFromArray(aData, [2, 2, 4, 4], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [2, 2, 4, 4], { requiresGrad: true, device: "webgpu" });

    const c = a.matmul(b);
    const loss = api.sum(c);
    await loss.backward();

    const torchletteAGrad = Array.from(await a.grad!.cpu());
    const torchletteBGrad = Array.from(await b.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "matmul_4d_backward",
      caseName: "mm4d",
      inputs: [
        { values: aData, shape: [2, 2, 4, 4] },
        { values: bData, shape: [2, 2, 4, 4] },
      ],
      options: {},
    }]);
    const pytorchAGrad = pytorchResult[0].grads[0].values;
    const pytorchBGrad = pytorchResult[0].grads[1].values;

    const aDiff = maxDiff(torchletteAGrad, pytorchAGrad);
    const bDiff = maxDiff(torchletteBGrad, pytorchBGrad);
    console.log(`  a.grad ${status(aDiff)}: max diff = ${aDiff.toExponential(2)}`);
    console.log(`  b.grad ${status(bDiff)}: max diff = ${bDiff.toExponential(2)}`);

    if (aDiff > 1e-4 || bDiff > 1e-4) {
      console.log(`  Torchlette a.grad first 16: ${torchletteAGrad.slice(0, 16).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch a.grad first 16: ${pytorchAGrad.slice(0, 16).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  // Test 4: 4D matmul with different inner dims [2,2,4,3] @ [2,2,3,4]
  console.log("\nTest 4: 4D batched matmul [2,2,4,3] @ [2,2,3,4]");
  {
    const aData = new Array(48).fill(0).map((_, i) => (i + 1) * 0.01);
    const bData = new Array(48).fill(0).map((_, i) => (i + 1) * 0.01);

    const a = api.tensorFromArray(aData, [2, 2, 4, 3], { requiresGrad: true, device: "webgpu" });
    const b = api.tensorFromArray(bData, [2, 2, 3, 4], { requiresGrad: true, device: "webgpu" });

    const c = a.matmul(b);
    const loss = api.sum(c);
    await loss.backward();

    const torchletteAGrad = Array.from(await a.grad!.cpu());
    const torchletteBGrad = Array.from(await b.grad!.cpu());

    const pytorchResult = await runTorchOracleBackwardBatch([{
      op: "matmul_4d_diff_backward",
      caseName: "mm4d_diff",
      inputs: [
        { values: aData, shape: [2, 2, 4, 3] },
        { values: bData, shape: [2, 2, 3, 4] },
      ],
      options: {},
    }]);
    const pytorchAGrad = pytorchResult[0].grads[0].values;
    const pytorchBGrad = pytorchResult[0].grads[1].values;

    const aDiff = maxDiff(torchletteAGrad, pytorchAGrad);
    const bDiff = maxDiff(torchletteBGrad, pytorchBGrad);
    console.log(`  a.grad ${status(aDiff)}: max diff = ${aDiff.toExponential(2)}`);
    console.log(`  b.grad ${status(bDiff)}: max diff = ${bDiff.toExponential(2)}`);

    if (aDiff > 1e-4 || bDiff > 1e-4) {
      console.log(`  Torchlette a.grad first 12: ${torchletteAGrad.slice(0, 12).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`  PyTorch a.grad first 12: ${pytorchAGrad.slice(0, 12).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
