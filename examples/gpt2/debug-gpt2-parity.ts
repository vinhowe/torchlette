/**
 * Debug GPT-2 Parity - Compare forward pass with PyTorch step by step
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

function meanAbsDiff(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / len;
}

function statusEmoji(diff: number, threshold: number = 1e-4): string {
  return diff < threshold ? "PASS" : "FAIL";
}

async function main() {
  console.log("=== GPT-2 Parity Debug (Step-by-Step) ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  // Create model
  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.eval();

  // Create simple input
  const batchSize = 2;
  const seqLen = 4;
  const inputData = [0, 1, 2, 3, 4, 5, 6, 7]; // batch=2, seq=4

  // Extract weights
  const params = model.parameters();
  const weights: number[][] = [];
  const paramShapes: number[][] = [];
  for (const p of params) {
    const data = await p.cpu();
    weights.push(Array.from(data));
    paramShapes.push(p.shape);
  }

  console.log(`Parameters: ${weights.length}`);
  console.log(`Config: vocab=${CONFIG.vocabSize}, blockSize=${CONFIG.blockSize}, embed=${CONFIG.embedDim}, layers=${CONFIG.numLayers}, heads=${CONFIG.numHeads}\n`);

  // Build full oracle inputs
  const oracleInputs: { values: number[]; shape: number[] }[] = [
    { values: inputData, shape: [batchSize, seqLen] },
  ];
  for (let i = 0; i < weights.length; i++) {
    oracleInputs.push({
      values: weights[i],
      shape: paramShapes[i],
    });
  }

  // ===== Step 1: Embedding =====
  console.log("Step 1: Embedding");

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

  const inputTensor = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
  const tokEmb = model.wte.forward(inputTensor);
  const posData = Array.from({ length: seqLen }, (_, i) => i);
  const posTensor = api.tensorFromArray(posData, [1, seqLen], { device: "webgpu" });
  const posEmb = model.wpe.forward(posTensor);
  const embedded = api.add(tokEmb, posEmb);
  const torchletteEmbed = Array.from(await embedded.cpu());

  const embedDiff = maxDiff(torchletteEmbed, pytorchEmbed);
  console.log(`  ${statusEmoji(embedDiff)} Max diff: ${embedDiff.toExponential(2)}`);

  // ===== Step 2: LayerNorm (ln1 of block 0) =====
  console.log("\nStep 2: LayerNorm (block 0 ln1)");

  // weights[2] = ln1.weight, weights[3] = ln1.bias
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

  // Use torchlette embedding and apply first layer norm
  const ln1Weight = api.tensorFromArray(weights[2], [CONFIG.embedDim], { device: "webgpu" });
  const ln1Bias = api.tensorFromArray(weights[3], [CONFIG.embedDim], { device: "webgpu" });
  const torchletteLn1Tensor = model.h[0].ln1.forward(embedded);
  const torchletteLn1 = Array.from(await torchletteLn1Tensor.cpu());

  const ln1Diff = maxDiff(torchletteLn1, pytorchLn1);
  console.log(`  ${statusEmoji(ln1Diff)} Max diff: ${ln1Diff.toExponential(2)}`);
  console.log(`    PyTorch first 5: ${pytorchLn1.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`    Torchlette first 5: ${torchletteLn1.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // ===== Step 3: Attention (block 0) =====
  console.log("\nStep 3: Attention (block 0)");

  // weights[4] = cAttn.weight [3*embed, embed], weights[5] = cAttn.bias [3*embed]
  // weights[6] = cProj.weight [embed, embed], weights[7] = cProj.bias [embed]
  const attnResults = await runTorchOracleBatch([{
    op: "attention_forward",
    caseName: "attn0",
    inputs: [
      { values: pytorchLn1, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: weights[4], shape: [3 * CONFIG.embedDim, CONFIG.embedDim] },
      { values: weights[5], shape: [3 * CONFIG.embedDim] },
      { values: weights[6], shape: [CONFIG.embedDim, CONFIG.embedDim] },
      { values: weights[7], shape: [CONFIG.embedDim] },
    ],
    options: { embedDim: CONFIG.embedDim, numHeads: CONFIG.numHeads },
  }]);
  const pytorchAttn = attnResults[0].values;

  // Run torchlette attention on the same input (use pytorchLn1 to isolate attention)
  const ln1ForAttn = api.tensorFromArray(pytorchLn1, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const torchletteAttnTensor = model.h[0].attn.forward(ln1ForAttn);
  const torchletteAttn = Array.from(await torchletteAttnTensor.cpu());

  const attnDiff = maxDiff(torchletteAttn, pytorchAttn);
  console.log(`  ${statusEmoji(attnDiff, 1e-3)} Max diff: ${attnDiff.toExponential(2)}`);
  console.log(`    PyTorch first 5: ${pytorchAttn.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`    Torchlette first 5: ${torchletteAttn.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // ===== Step 4: MLP (block 0) =====
  console.log("\nStep 4: MLP (block 0) - using PyTorch's attention output");

  // After attention, apply residual: x = embed + attn_out
  // Then ln2 + MLP
  const postAttnResidual = pytorchEmbed.map((v, i) => v + pytorchAttn[i]);

  // weights[8] = ln2.weight, weights[9] = ln2.bias
  const ln2Results = await runTorchOracleBatch([{
    op: "layer_norm_forward",
    caseName: "ln2",
    inputs: [
      { values: postAttnResidual, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: weights[8], shape: [CONFIG.embedDim] },
      { values: weights[9], shape: [CONFIG.embedDim] },
    ],
    options: { normalized_shape: [CONFIG.embedDim], eps: 1e-5 },
  }]);
  const pytorchLn2 = ln2Results[0].values;

  // weights[10] = cFc.weight [4*embed, embed], weights[11] = cFc.bias [4*embed]
  // weights[12] = cProj.weight [embed, 4*embed], weights[13] = cProj.bias [embed]
  // MLP: fc -> gelu -> proj

  // FC layer
  const fcResults = await runTorchOracleBatch([{
    op: "linear_forward",
    caseName: "fc",
    inputs: [
      { values: pytorchLn2, shape: [batchSize, seqLen, CONFIG.embedDim] },
      { values: weights[10], shape: [4 * CONFIG.embedDim, CONFIG.embedDim] },
      { values: weights[11], shape: [4 * CONFIG.embedDim] },
    ],
    options: {},
  }]);
  const pytorchFc = fcResults[0].values;

  // Apply GELU (need to add gelu_forward to oracle or compute manually)
  // For now, let's test the torchlette MLP on the same ln2 input

  const ln2Input = api.tensorFromArray(postAttnResidual, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const torchletteLn2Tensor = model.h[0].ln2.forward(ln2Input);
  const torchletteLn2 = Array.from(await torchletteLn2Tensor.cpu());

  const ln2Diff = maxDiff(torchletteLn2, pytorchLn2);
  console.log(`  LN2 ${statusEmoji(ln2Diff)} Max diff: ${ln2Diff.toExponential(2)}`);

  // Test MLP
  const ln2ForMlp = api.tensorFromArray(pytorchLn2, [batchSize, seqLen, CONFIG.embedDim], { device: "webgpu" });
  const torchletteMlpTensor = model.h[0].mlp.forward(ln2ForMlp);
  const torchletteMlp = Array.from(await torchletteMlpTensor.cpu());

  // Need to get PyTorch MLP output - let's compute it step by step
  // GELU(FC) then proj
  // We'll use the torch oracle to get the full MLP result

  // ===== Step 5: Full forward =====
  console.log("\nStep 5: Full forward pass");

  const fullResults = await runTorchOracleBatch([{
    op: "gpt2_forward",
    caseName: "full",
    inputs: oracleInputs,
    options: {
      vocabSize: CONFIG.vocabSize,
      blockSize: CONFIG.blockSize,
      embedDim: CONFIG.embedDim,
      numLayers: CONFIG.numLayers,
      numHeads: CONFIG.numHeads,
    },
  }]);
  const pytorchLogits = fullResults[0].values;

  const input = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
  const logits = model.forward(input);
  const torchletteLogits = Array.from(await logits.cpu());

  const fullDiff = maxDiff(torchletteLogits, pytorchLogits);
  const fullMeanDiff = meanAbsDiff(torchletteLogits, pytorchLogits);

  console.log(`  ${statusEmoji(fullDiff, 0.01)} Max diff: ${fullDiff.toExponential(2)}, Mean diff: ${fullMeanDiff.toExponential(2)}`);
  console.log(`    PyTorch first 5: ${pytorchLogits.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`    Torchlette first 5: ${torchletteLogits.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);

  // ===== Summary =====
  console.log("\n=== Summary ===");
  console.log(`  1. Embedding:  ${statusEmoji(embedDiff)} (diff=${embedDiff.toExponential(2)})`);
  console.log(`  2. LayerNorm1: ${statusEmoji(ln1Diff)} (diff=${ln1Diff.toExponential(2)})`);
  console.log(`  3. Attention:  ${statusEmoji(attnDiff, 1e-3)} (diff=${attnDiff.toExponential(2)})`);
  console.log(`  4. LayerNorm2: ${statusEmoji(ln2Diff)} (diff=${ln2Diff.toExponential(2)})`);
  console.log(`  5. Full model: ${statusEmoji(fullDiff, 0.01)} (diff=${fullDiff.toExponential(2)})`);

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
