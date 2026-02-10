/**
 * Debug Linear Layer - Isolate the linear projection issue
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

function status(diff: number, threshold: number = 1e-4): string {
  return diff < threshold ? "PASS" : "FAIL";
}

async function main() {
  console.log("=== Linear Layer Debug ===\n");

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

  // Get LN1 output (already verified to match)
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

  console.log("Input: LN1 output (verified to match PyTorch)");
  console.log(`  Shape: [${batchSize}, ${seqLen}, ${CONFIG.embedDim}]`);
  console.log(`  First 5 values: ${pytorchLn1.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // ===== Test 1: Get model's Linear layer weights =====
  console.log("\n--- Model's cAttn weights ---");

  // weights[4] = cAttn.weight [3*embed, embed] = [192, 64]
  // weights[5] = cAttn.bias [3*embed] = [192]

  const cAttnWeight = weights[4];
  const cAttnBias = weights[5];

  console.log(`  cAttn weight shape: [${3 * CONFIG.embedDim}, ${CONFIG.embedDim}]`);
  console.log(`  cAttn weight first 5: ${cAttnWeight.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  cAttn bias first 5: ${cAttnBias.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // ===== Test 2: PyTorch linear forward =====
  console.log("\n--- PyTorch Linear Output ---");

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

  console.log(`  Output shape: [${linearResults[0].shape.join(", ")}]`);
  console.log(`  First 10 values: ${pytorchLinear.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);

  // ===== Test 3: Torchlette Linear via model =====
  console.log("\n--- Torchlette Linear (via model.h[0].attn.cAttn) ---");

  const ln1Tensor = api.tensorFromArray(pytorchLn1, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const torchletteLinearTensor = model.h[0].attn.cAttn.forward(ln1Tensor);
  const torchletteLinear = Array.from(await torchletteLinearTensor.cpu());

  console.log(`  Output shape: [${torchletteLinearTensor.shape.join(", ")}]`);
  console.log(`  First 10 values: ${torchletteLinear.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);

  const linearDiff = maxDiff(torchletteLinear, pytorchLinear);
  console.log(`\n  Linear diff ${status(linearDiff)}: ${linearDiff.toExponential(2)}`);

  // ===== Test 4: Manual linear computation in torchlette =====
  console.log("\n--- Manual Linear in Torchlette (x @ W^T + b) ---");

  // Create fresh tensors from the same data
  const xTensor = api.tensorFromArray(pytorchLn1, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const wTensor = api.tensorFromArray(cAttnWeight, [3 * CONFIG.embedDim, CONFIG.embedDim], { device: "webgpu" });
  const bTensor = api.tensorFromArray(cAttnBias, [3 * CONFIG.embedDim], { device: "webgpu" });

  // x: [2, 4, 64], W: [192, 64], W^T: [64, 192]
  // x @ W^T: [2, 4, 192]
  const wT = api.transpose(wTensor, { dim0: 0, dim1: 1 });
  console.log(`  W^T shape: [${wT.shape.join(", ")}]`);

  // Matmul needs contiguous tensors
  const wTContiguous = wT.contiguous();
  const xMatW = xTensor.matmul(wTContiguous);
  console.log(`  x @ W^T shape: [${xMatW.shape.join(", ")}]`);

  const xMatWPlusBias = api.add(xMatW, bTensor);
  const manualLinear = Array.from(await xMatWPlusBias.cpu());

  console.log(`  Manual output first 10: ${manualLinear.slice(0, 10).map(v => v.toFixed(4)).join(", ")}`);

  const manualDiff = maxDiff(manualLinear, pytorchLinear);
  console.log(`  Manual linear diff ${status(manualDiff)}: ${manualDiff.toExponential(2)}`);

  // ===== Test 5: Check if model's linear layer uses different weights =====
  console.log("\n--- Compare model weights vs extracted weights ---");

  const modelCattnWeightData = Array.from(await model.h[0].attn.cAttn.weight.cpu());
  const modelCattnBiasData = Array.from(await model.h[0].attn.cAttn.bias.cpu());

  const weightDiff = maxDiff(modelCattnWeightData, cAttnWeight);
  const biasDiff = maxDiff(modelCattnBiasData, cAttnBias);

  console.log(`  Weight diff: ${weightDiff.toExponential(2)}`);
  console.log(`  Bias diff: ${biasDiff.toExponential(2)}`);
  console.log(`  Model weight first 5: ${modelCattnWeightData.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  Extracted weight first 5: ${cAttnWeight.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  console.log("\n=== Summary ===");
  console.log(`Model linear diff: ${linearDiff.toExponential(2)}`);
  console.log(`Manual linear diff: ${manualDiff.toExponential(2)}`);
  console.log(`Weight match: ${weightDiff < 1e-6 ? "YES" : "NO"}`);

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
