/**
 * Debug Linear Layer - Test contiguous vs non-contiguous transpose
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { runTorchOracleBatch } from "../../test/oracle/torch-oracle";

const CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 32,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0,
};

function maxDiff(a: number[], b: number[]): number {
  let max = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return max;
}

async function main() {
  console.log("=== Linear Contiguous Debug ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.eval();

  const batchSize = 2;
  const seqLen = 4;

  // Extract weights
  const params = model.parameters();
  const weights: number[][] = [];
  for (const p of params) {
    const data = await p.cpu();
    weights.push(Array.from(data));
  }

  // Get LN1 output
  const inputData = [0, 1, 2, 3, 4, 5, 6, 7];
  const embedResults = await runTorchOracleBatch([{
    op: "embedding_forward",
    caseName: "embed",
    inputs: [
      { values: inputData, shape: [batchSize, seqLen] },
      { values: weights[0], shape: [CONFIG.vocabSize, CONFIG.embedDim] },
      { values: weights[1], shape: [CONFIG.blockSize, CONFIG.embedDim] },
    ],
    options: { seqLen },
  }]);
  const pytorchEmbed = embedResults[0].values;

  const ln1Results = await runTorchOracleBatch([{
    op: "layer_norm_forward",
    caseName: "ln1",
    inputs: [
      { values: pytorchEmbed, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: weights[2], shape: [CONFIG.embedDim] },
      { values: weights[3], shape: [CONFIG.embedDim] },
    ],
    options: { normalized_shape: [CONFIG.embedDim], eps: 1e-5 },
  }]);
  const pytorchLn1 = ln1Results[0].values;

  // Get PyTorch linear result
  const cAttnWeight = weights[4];
  const cAttnBias = weights[5];

  const linearResults = await runTorchOracleBatch([{
    op: "linear_forward",
    caseName: "linear",
    inputs: [
      { values: pytorchLn1, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: cAttnWeight, shape: [3 * CONFIG.embedDim, CONFIG.embedDim] },
      { values: cAttnBias, shape: [3 * CONFIG.embedDim] },
    ],
    options: {},
  }]);
  const pytorchLinear = linearResults[0].values;

  // Test 1: WITHOUT contiguous (like current Linear.forward)
  console.log("Test 1: Matmul with non-contiguous transposed weight (current Linear behavior)");

  const xTensor1 = api.tensorFromArray(pytorchLn1, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const wTensor1 = api.tensorFromArray(cAttnWeight, [3 * CONFIG.embedDim, CONFIG.embedDim], { device: "webgpu" });
  const bTensor1 = api.tensorFromArray(cAttnBias, [3 * CONFIG.embedDim], { device: "webgpu" });

  const wT1 = api.transpose(wTensor1, { dim0: 0, dim1: 1 });
  console.log(`  W^T isContiguous: ${wT1.isContiguous}`);
  const result1 = api.add(xTensor1.matmul(wT1), bTensor1);
  const output1 = Array.from(await result1.cpu());

  const diff1 = maxDiff(output1, pytorchLinear);
  console.log(`  Max diff vs PyTorch: ${diff1.toExponential(2)}`);
  console.log(`  First 5: ${output1.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  Expected: ${pytorchLinear.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // Test 2: WITH contiguous
  console.log("\nTest 2: Matmul with contiguous transposed weight");

  const xTensor2 = api.tensorFromArray(pytorchLn1, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const wTensor2 = api.tensorFromArray(cAttnWeight, [3 * CONFIG.embedDim, CONFIG.embedDim], { device: "webgpu" });
  const bTensor2 = api.tensorFromArray(cAttnBias, [3 * CONFIG.embedDim], { device: "webgpu" });

  const wT2 = api.transpose(wTensor2, { dim0: 0, dim1: 1 }).contiguous();
  console.log(`  W^T isContiguous: ${wT2.isContiguous}`);
  const result2 = api.add(xTensor2.matmul(wT2), bTensor2);
  const output2 = Array.from(await result2.cpu());

  const diff2 = maxDiff(output2, pytorchLinear);
  console.log(`  Max diff vs PyTorch: ${diff2.toExponential(2)}`);
  console.log(`  First 5: ${output2.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // Test 3: Check the transpose strides
  console.log("\n--- Stride Analysis ---");
  const testW = api.tensorFromArray(cAttnWeight, [3 * CONFIG.embedDim, CONFIG.embedDim], { device: "webgpu" });
  const testWT = api.transpose(testW, { dim0: 0, dim1: 1 });
  console.log(`  Original W shape: [${testW.shape.join(", ")}]`);
  console.log(`  Transposed W^T shape: [${testWT.shape.join(", ")}]`);
  console.log(`  W^T isContiguous: ${testWT.isContiguous}`);

  console.log("\n=== Conclusion ===");
  console.log(`Non-contiguous matmul diff: ${diff1.toExponential(2)}`);
  console.log(`Contiguous matmul diff: ${diff2.toExponential(2)}`);

  if (diff1 > 1e-3 && diff2 < 1e-4) {
    console.log("\n*** BUG CONFIRMED: Matmul doesn't handle non-contiguous tensors correctly ***");
    console.log("Fix: Add .contiguous() after transpose in Linear.forward()");
  }

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
