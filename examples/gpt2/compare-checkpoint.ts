/**
 * Compare Checkpointed vs Non-Checkpointed GPT-2
 *
 * Verifies that gradient checkpointing produces the same gradients as the
 * standard forward/backward pass.
 */
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, GPT2Config } from "./model";

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
  console.log("=== Checkpoint vs Non-Checkpoint Comparison ===\n");

  await initWebGPU();

  // Create two identical models
  const api1 = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });
  const api2 = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: false,
  });

  const model1 = new GPT2(api1, TEST_CONFIG, { device: "webgpu" });
  const model2 = new GPT2(api2, TEST_CONFIG, { device: "webgpu" });

  // Ensure both models are in eval mode (no dropout randomness)
  model1.eval();
  model2.eval();

  // Copy weights from model1 to model2 to ensure they're identical
  console.log("Setting up identical models...");
  const params1 = model1.parameters();
  const params2 = model2.parameters();

  for (let i = 0; i < params1.length; i++) {
    const p1 = params1[i];
    const p2 = params2[i];
    const data = Array.from(await p1.cpu());
    // Create new tensor with same data and copy to p2
    const temp = api2.tensorFromArray(data, p2.shape, { device: "webgpu" });
    p2.copy_(temp);
  }

  // Create input data
  const batch = 2;
  const seqLen = 16;
  const inputData = new Array(batch * seqLen).fill(0).map((_, i) => i % TEST_CONFIG.vocabSize);
  const targetData = new Array(batch * seqLen).fill(0).map((_, i) => (i + 1) % TEST_CONFIG.vocabSize);

  // Forward + backward WITHOUT checkpointing
  console.log("\nRunning forward/backward WITHOUT checkpointing...");
  const input1 = api1.tensorFromArray(inputData, [batch, seqLen], { device: "webgpu" });
  const target1 = api1.tensorFromArray(targetData, [batch, seqLen], { device: "webgpu" });

  const { loss: loss1 } = model1.forwardWithLoss(input1, target1, { useCheckpoint: false });
  await loss1!.backward();
  const lossVal1 = (await loss1!.cpu())[0];
  console.log(`  Loss: ${lossVal1.toFixed(6)}`);

  // Forward + backward WITH checkpointing
  console.log("Running forward/backward WITH checkpointing...");
  const input2 = api2.tensorFromArray(inputData, [batch, seqLen], { device: "webgpu" });
  const target2 = api2.tensorFromArray(targetData, [batch, seqLen], { device: "webgpu" });

  const { loss: loss2 } = model2.forwardWithLoss(input2, target2, { useCheckpoint: true });
  await loss2!.backward();
  const lossVal2 = (await loss2!.cpu())[0];
  console.log(`  Loss: ${lossVal2.toFixed(6)}`);

  // Compare losses
  console.log("\n=== Comparison Results ===");
  const lossDiff = Math.abs(lossVal1 - lossVal2);
  console.log(`\nLoss difference: ${lossDiff.toExponential(2)}`);
  console.log(`  ${status(lossDiff, 1e-5)}`);

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
  for (let i = 0; i < params1.length; i++) {
    const p1 = params1[i];
    const p2 = params2[i];

    if (!p1.grad || !p2.grad) {
      console.log(`  ${i.toString().padStart(2)}: ${paramNames[i].padEnd(32)} NO GRAD`);
      continue;
    }

    const g1 = Array.from(await p1.grad.cpu());
    const g2 = Array.from(await p2.grad.cpu());

    const diff = maxDiff(g1, g2);
    const pass = diff < 1e-4;
    if (!pass) allPass = false;

    // Also compute mean absolute difference for debugging
    let sum = 0;
    for (let j = 0; j < g1.length; j++) {
      sum += Math.abs(g1[j] - g2[j]);
    }
    const meanDiff = sum / g1.length;

    const passStr = pass ? "PASS" : "FAIL";
    console.log(
      `  ${i.toString().padStart(2)}: ${paramNames[i].padEnd(32)} ${passStr} max=${diff.toExponential(2)} mean=${meanDiff.toExponential(2)}`
    );

    if (!pass) {
      console.log(`      Non-checkpoint first 5: ${g1.slice(0, 5).map(v => v.toFixed(6)).join(", ")}`);
      console.log(`      Checkpoint first 5:     ${g2.slice(0, 5).map(v => v.toFixed(6)).join(", ")}`);
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
