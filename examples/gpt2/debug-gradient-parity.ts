/**
 * Debug Gradient Parity - Full GPT-2 gradient comparison with PyTorch
 *
 * Tests that Torchlette's GPT-2 backward pass produces the same gradients as PyTorch.
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, type GPT2Config } from "./model";
import { runTorchOracleBatch, runTorchOracleBackwardBatch } from "../../test/oracle/torch-oracle";

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

function meanDiff(a: number[], b: number[]): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / len;
}

function status(diff: number, threshold: number = 1e-3): string {
  return diff < threshold ? "PASS" : "FAIL";
}

const PARAM_NAMES = [
  "wte.weight",
  "wpe.weight",
  "blocks.0.ln_1.weight",
  "blocks.0.ln_1.bias",
  "blocks.0.attn.c_attn.weight",
  "blocks.0.attn.c_attn.bias",
  "blocks.0.attn.c_proj.weight",
  "blocks.0.attn.c_proj.bias",
  "blocks.0.ln_2.weight",
  "blocks.0.ln_2.bias",
  "blocks.0.mlp.c_fc.weight",
  "blocks.0.mlp.c_fc.bias",
  "blocks.0.mlp.c_proj.weight",
  "blocks.0.mlp.c_proj.bias",
  "blocks.1.ln_1.weight",
  "blocks.1.ln_1.bias",
  "blocks.1.attn.c_attn.weight",
  "blocks.1.attn.c_attn.bias",
  "blocks.1.attn.c_proj.weight",
  "blocks.1.attn.c_proj.bias",
  "blocks.1.ln_2.weight",
  "blocks.1.ln_2.bias",
  "blocks.1.mlp.c_fc.weight",
  "blocks.1.mlp.c_fc.bias",
  "blocks.1.mlp.c_proj.weight",
  "blocks.1.mlp.c_proj.bias",
  "ln_f.weight",
  "ln_f.bias",
];

const PARAM_SHAPES = [
  [CONFIG.vocabSize, CONFIG.embedDim],
  [CONFIG.blockSize, CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [3 * CONFIG.embedDim, CONFIG.embedDim],
  [3 * CONFIG.embedDim],
  [CONFIG.embedDim, CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [4 * CONFIG.embedDim, CONFIG.embedDim],
  [4 * CONFIG.embedDim],
  [CONFIG.embedDim, 4 * CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [3 * CONFIG.embedDim, CONFIG.embedDim],
  [3 * CONFIG.embedDim],
  [CONFIG.embedDim, CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [4 * CONFIG.embedDim, CONFIG.embedDim],
  [4 * CONFIG.embedDim],
  [CONFIG.embedDim, 4 * CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
  [CONFIG.embedDim],
];

async function main() {
  console.log("=== GPT-2 Gradient Parity Test ===\n");

  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.eval();

  const batchSize = 2;
  const seqLen = 8;

  // Extract weights
  const params = model.parameters();
  console.log(`Model has ${params.length} parameters\n`);

  const weights: number[][] = [];
  for (const p of params) {
    const data = await p.cpu();
    weights.push(Array.from(data));
  }

  // Create input and target
  const inputData: number[] = [];
  const targetData: number[] = [];
  for (let i = 0; i < batchSize * seqLen; i++) {
    inputData.push(i % CONFIG.vocabSize);
    targetData.push((i + 1) % CONFIG.vocabSize);
  }

  console.log("Step 1: Forward pass comparison");
  console.log("================================\n");

  // Get PyTorch forward result
  const inputs = [
    { values: inputData, shape: [batchSize, seqLen] },
    { values: targetData, shape: [batchSize, seqLen] },
  ];
  for (let i = 0; i < weights.length; i++) {
    inputs.push({ values: weights[i], shape: PARAM_SHAPES[i] });
  }

  const pytorchForward = await runTorchOracleBatch([{
    op: "gpt2_forward",
    caseName: "forward",
    inputs: [
      { values: inputData, shape: [batchSize, seqLen] },
      ...weights.map((w, i) => ({ values: w, shape: PARAM_SHAPES[i] })),
    ],
    options: {
      vocabSize: CONFIG.vocabSize,
      blockSize: CONFIG.blockSize,
      embedDim: CONFIG.embedDim,
      numLayers: CONFIG.numLayers,
      numHeads: CONFIG.numHeads,
    },
  }]);
  const pytorchLogits = pytorchForward[0].values;

  // Torchlette forward
  const inputTensor = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
  const torchletteLogitsTensor = model.forward(inputTensor);
  const torchletteLogits = Array.from(await torchletteLogitsTensor.cpu());

  const forwardDiff = maxDiff(torchletteLogits, pytorchLogits);
  console.log(`Forward logits ${status(forwardDiff, 1e-4)}: max diff = ${forwardDiff.toExponential(2)}`);
  console.log(`  PyTorch first 5: ${pytorchLogits.slice(0, 5).map(v => v.toFixed(4)).join(", ")}`);
  console.log(`  Torchlette first 5: ${torchletteLogits.slice(0, 5).map(v => v.toFixed(4)).join(", ")}\n`);

  console.log("Step 2: Backward pass comparison");
  console.log("=================================\n");

  // Get PyTorch gradients
  const pytorchResult = await runTorchOracleBackwardBatch([{
    op: "gpt2_forward_backward",
    caseName: "backward",
    inputs,
    options: {
      vocabSize: CONFIG.vocabSize,
      blockSize: CONFIG.blockSize,
      embedDim: CONFIG.embedDim,
      numLayers: CONFIG.numLayers,
      numHeads: CONFIG.numHeads,
    },
  }]);
  const pytorchLoss = pytorchResult[0].output.values[0];
  const pytorchGrads = pytorchResult[0].grads;

  console.log(`PyTorch loss: ${pytorchLoss.toFixed(6)}`);

  // Torchlette backward
  // Create fresh model with same weights to ensure clean gradient state
  const model2 = new GPT2(api, CONFIG, { device: "webgpu" });
  const params2 = model2.parameters();

  // Copy weights
  for (let i = 0; i < params2.length; i++) {
    const weightTensor = api.tensorFromArray(weights[i], PARAM_SHAPES[i], {
      requiresGrad: true,
      device: "webgpu",
    });
    (params2[i] as any)._rt = weightTensor._unwrap();
    (params2[i] as any)._shape = PARAM_SHAPES[i];
  }

  // Actually, let me just use the existing model and zero grads
  for (const p of params) {
    p.zeroGrad();
  }

  const inputTensor2 = api.tensorFromArray(inputData, [batchSize, seqLen], { device: "webgpu" });
  const targetTensor = api.tensorFromArray(targetData, [batchSize, seqLen], { device: "webgpu" });

  const logits2 = model.forward(inputTensor2);

  // Compute cross-entropy loss
  // Reshape logits: [batch, seq, vocab] -> [batch*seq, vocab]
  const flatLogits = logits2.reshape([batchSize * seqLen, CONFIG.vocabSize]);

  // Compute softmax
  const maxLogits = flatLogits.max({ dim: 1, keepdim: true });
  const shiftedLogits = api.sub(flatLogits, maxLogits);
  const expLogits = shiftedLogits.exp();
  const sumExp = expLogits.sum({ dim: 1, keepdim: true });
  const logSoftmax = api.sub(shiftedLogits, sumExp.log());

  // Gather the log probabilities of the target tokens
  const flatTargets = targetTensor.reshape([batchSize * seqLen]);
  const expandedTargets = flatTargets.reshape([batchSize * seqLen, 1]);
  const targetLogProbs = logSoftmax.gather(expandedTargets, { dim: 1 });

  // Negative log likelihood loss (mean over batch)
  const loss = api.neg(api.mean(targetLogProbs));
  const torchletteL = await loss.item();

  console.log(`Torchlette loss: ${torchletteL.toFixed(6)}`);
  console.log(`Loss diff: ${Math.abs(pytorchLoss - torchletteL).toExponential(2)}\n`);

  // Backward
  await loss.backward();

  // Compare gradients
  console.log("Parameter gradient comparison:");
  console.log("-".repeat(70));

  console.log(`PyTorch grads count: ${pytorchGrads?.length ?? 'undefined'}`);

  let allPass = true;
  for (let i = 0; i < params.length; i++) {
    const grad = params[i].grad;
    if (!grad) {
      console.log(`  ${i}: ${PARAM_NAMES[i]} - NO GRADIENT`);
      allPass = false;
      continue;
    }

    const torchletteGrad = Array.from(await grad.cpu());
    const pytorchGradObj = pytorchGrads?.[i];
    if (!pytorchGradObj) {
      console.log(`  ${i}: ${PARAM_NAMES[i]} - NO PYTORCH GRADIENT`);
      allPass = false;
      continue;
    }
    const pytorchGrad = pytorchGradObj.values;

    const diff = maxDiff(torchletteGrad, pytorchGrad);
    const mean = meanDiff(torchletteGrad, pytorchGrad);
    const pass = diff < 1e-3;
    if (!pass) allPass = false;

    const statusStr = pass ? "PASS" : "FAIL";
    console.log(`  ${i.toString().padStart(2)}: ${PARAM_NAMES[i].padEnd(30)} ${statusStr} max=${diff.toExponential(2)} mean=${mean.toExponential(2)}`);

    if (!pass) {
      console.log(`      PyTorch first 5: ${pytorchGrad.slice(0, 5).map((v: number) => v.toFixed(6)).join(", ")}`);
      console.log(`      Torchlette first 5: ${torchletteGrad.slice(0, 5).map(v => v.toFixed(6)).join(", ")}`);
    }
  }

  console.log("-".repeat(70));
  console.log(`\nOverall: ${allPass ? "ALL PASS" : "SOME FAILED"}`);

  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
