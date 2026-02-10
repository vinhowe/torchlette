/**
 * Verify Checkpointed GPT-2 Matches PyTorch
 *
 * Tests that gradient checkpointing produces results matching PyTorch
 * for both forward and backward passes.
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, GPT2Config } from "./model";
import { runTorchOracleBackwardBatch } from "../../test/oracle/torch-oracle";

// Small test config for quick verification
const TEST_CONFIG: GPT2Config = {
  vocabSize: 1000,
  blockSize: 128,
  numLayers: 2,
  numHeads: 2,
  embedDim: 64,
  dropoutRate: 0.0, // Disable dropout for deterministic comparison
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
  console.log("=== Checkpointed GPT-2 vs PyTorch ===\n");

  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const model = new GPT2(api, TEST_CONFIG, { device: "webgpu" });
  model.eval();

  // Create input data
  const batch = 2;
  const seqLen = 16;
  const inputData = new Array(batch * seqLen).fill(0).map((_, i) => i % TEST_CONFIG.vocabSize);
  const targetData = new Array(batch * seqLen).fill(0).map((_, i) => (i + 1) % TEST_CONFIG.vocabSize);

  // Get all parameter data for PyTorch comparison
  const params = model.parameters();
  const paramDataList: { values: number[]; shape: number[] }[] = [];
  for (const p of params) {
    paramDataList.push({
      values: Array.from(await p.cpu()),
      shape: p.shape,
    });
  }

  // Forward + backward WITH checkpointing
  console.log("Running Torchlette forward/backward WITH checkpointing...");
  const input = api.tensorFromArray(inputData, [batch, seqLen], { device: "webgpu" });
  const target = api.tensorFromArray(targetData, [batch, seqLen], { device: "webgpu" });

  const { loss } = model.forwardWithLoss(input, target, { useCheckpoint: true });
  await loss!.backward();
  const torchletteVal = (await loss!.cpu())[0];
  console.log(`  Loss: ${torchletteVal.toFixed(6)}`);

  // Collect Torchlette gradients
  const torchletteGrads: number[][] = [];
  for (const p of params) {
    torchletteGrads.push(Array.from(await p.grad!.cpu()));
  }

  // Run PyTorch comparison
  console.log("Running PyTorch comparison...");
  const pytorchResult = await runTorchOracleBackwardBatch([{
    op: "gpt2_forward_backward",
    caseName: "checkpoint_verify",
    inputs: [
      { values: inputData, shape: [batch, seqLen] },
      { values: targetData, shape: [batch, seqLen] },
      ...paramDataList,
    ],
    options: {
      vocabSize: TEST_CONFIG.vocabSize,
      blockSize: TEST_CONFIG.blockSize,
      embedDim: TEST_CONFIG.embedDim,
      numLayers: TEST_CONFIG.numLayers,
      numHeads: TEST_CONFIG.numHeads,
    },
  }]);

  const pytorchLoss = pytorchResult[0].output.values[0];
  const pytorchGrads = pytorchResult[0].grads;
  console.log(`  Loss: ${pytorchLoss.toFixed(6)}`);

  // Compare losses
  console.log("\n=== Comparison Results ===");
  const lossDiff = Math.abs(torchletteVal - pytorchLoss);
  console.log(`\nLoss difference: ${lossDiff.toExponential(2)}`);
  console.log(`  ${status(lossDiff, 1e-4)}`);

  // Compare gradients
  console.log("\nParameter gradient comparison:");
  console.log("-".repeat(70));

  const paramNames = [
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

  let allPass = true;
  for (let i = 0; i < torchletteGrads.length; i++) {
    const tGrad = torchletteGrads[i];
    const pGrad = pytorchGrads[i]?.values ?? [];

    if (pGrad.length === 0) {
      console.log(`  ${i.toString().padStart(2)}: ${paramNames[i].padEnd(32)} NO PYTORCH GRAD`);
      continue;
    }

    const diff = maxDiff(tGrad, pGrad);
    const pass = diff < 1e-3; // Slightly relaxed threshold for float precision
    if (!pass) allPass = false;

    // Also compute mean absolute difference
    let sum = 0;
    for (let j = 0; j < tGrad.length; j++) {
      sum += Math.abs(tGrad[j] - pGrad[j]);
    }
    const meanDiff = sum / tGrad.length;

    const passStr = pass ? "PASS" : "FAIL";
    console.log(
      `  ${i.toString().padStart(2)}: ${paramNames[i].padEnd(32)} ${passStr} max=${diff.toExponential(2)} mean=${meanDiff.toExponential(2)}`
    );

    if (!pass) {
      console.log(`      Torchlette first 5: ${tGrad.slice(0, 5).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`      PyTorch first 5:    ${pGrad.slice(0, 5).map((v: number) => v.toFixed(6)).join(", ")}`);
    }
  }

  console.log("-".repeat(70));
  console.log(`\nOverall: ${allPass ? "ALL PASS" : "SOME FAILURES"}`);

  process.exit(allPass ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
