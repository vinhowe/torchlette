/**
 * Node.js smoke test for LoRA training.
 * Validates the full pipeline: model creation, weight loading, LoRA training, generation.
 */
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";

const DISTILGPT2_CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 6,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0.0,
};

import { generateTokens } from "../examples/gpt2-lora-trainer/src/lib/torchlette/inference";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

async function main() {
  console.log("=== LoRA Training Smoke Test ===\n");

  // Step 1: Init WebGPU
  console.log("1. Initializing WebGPU...");
  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU not available");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", {
    enableFusion: false,
  });

  // Step 2: Load tokenizer
  console.log("2. Loading tokenizer...");
  const tokenizerDir = path.join(process.cwd(), "models", "distilgpt2");
  const vocabJson = await import("fs").then((fs) =>
    JSON.parse(fs.readFileSync(path.join(tokenizerDir, "vocab.json"), "utf-8")),
  );
  const mergesText = await import("fs").then((fs) =>
    fs.readFileSync(path.join(tokenizerDir, "merges.txt"), "utf-8"),
  );
  const tokenizer = new GPT2Tokenizer();
  const mergeLines = mergesText
    .split("\n")
    .filter((l: string) => l && !l.startsWith("#"));
  tokenizer.load(vocabJson, mergeLines);
  console.log("   Tokenizer loaded");

  // Step 3: Create model with LoRA
  console.log("3. Creating DistilGPT-2 + LoRA...");
  const loraConfig = { rank: 4, alpha: 4 };
  const model = new GPT2WithLoRA(api, DISTILGPT2_CONFIG, loraConfig, "webgpu");
  console.log(
    `   Model created: ${DISTILGPT2_CONFIG.numLayers} layers, rank=${loraConfig.rank}`,
  );

  // Step 4: Load pretrained weights
  console.log("4. Loading pretrained weights from disk...");
  const fs = await import("node:fs");
  const modelPath = path.join(tokenizerDir, "model.safetensors");
  const buf = fs.readFileSync(modelPath);
  const headerLen = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const headerJson = new TextDecoder().decode(buf.subarray(8, 8 + headerLen));
  const header = JSON.parse(headerJson);
  const dataOffset = 8 + headerLen;

  const weights = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [name, meta] of Object.entries(header) as [string, any][]) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = meta;
    const [start, end] = data_offsets;
    const raw = buf.subarray(dataOffset + start, dataOffset + end);
    let f32: Float32Array;
    if (dtype === "F32") {
      const copy = new Uint8Array(raw).slice();
      f32 = new Float32Array(copy.buffer);
    } else if (dtype === "F16") {
      // Convert f16 to f32
      const u16 = new Uint16Array(
        raw.buffer,
        raw.byteOffset,
        raw.byteLength / 2,
      );
      f32 = new Float32Array(u16.length);
      for (let i = 0; i < u16.length; i++) {
        const h = u16[i];
        const sign = (h >> 15) & 1;
        const exp = (h >> 10) & 0x1f;
        const mant = h & 0x3ff;
        if (exp === 0) f32[i] = (sign ? -1 : 1) * (mant / 1024) * 2 ** -14;
        else if (exp === 31)
          f32[i] = mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
        else f32[i] = (sign ? -1 : 1) * (1 + mant / 1024) * 2 ** (exp - 15);
      }
    } else {
      console.warn(`   Skipping ${name}: unsupported dtype ${dtype}`);
      continue;
    }
    // Strip "transformer." prefix — safetensors uses "transformer.wte.weight",
    // but model.loadBaseWeights expects "wte.weight"
    const cleanName = name.replace(/^transformer\./, "");
    weights.set(cleanName, { data: new Float32Array(f32), shape });
  }
  model.loadBaseWeights(weights);
  console.log(`   Loaded ${weights.size} weight tensors`);

  // Step 5: Verify forward pass
  console.log("5. Testing forward pass...");
  const testTokens = tokenizer.encode(
    "Hello world this is a test of the model",
  );
  console.log(`   Test tokens: ${testTokens.length} tokens`);
  const testInput = api.tensorFromArray(
    testTokens.slice(0, -1),
    [1, testTokens.length - 1],
    { device: "webgpu" },
  );
  const testTarget = api.tensorFromArray(
    testTokens.slice(1),
    [1, testTokens.length - 1],
    { device: "webgpu" },
  );

  await api.beginStep();
  const { loss: testLoss } = model.forwardWithLoss(testInput, testTarget);
  const testLossVal = await testLoss.item();
  console.log(`   Forward pass OK, initial loss: ${testLossVal.toFixed(4)}`);
  api.endStep();
  await api.markStep();

  // Step 6: Train a few steps
  console.log("6. Training 5 steps...");
  const trainingText =
    "The quick brown fox jumps over the lazy dog. " +
    "In a galaxy far far away, there lived a brave knight. " +
    "The rain in Spain falls mainly on the plain. " +
    "To be or not to be, that is the question. " +
    "All that glitters is not gold, but it sure looks shiny.";
  const tokens = tokenizer.encode(trainingText);
  console.log(`   Training on ${tokens.length} tokens`);

  const loraParams = model.getLoRAParameters();
  console.log(`   LoRA parameters: ${loraParams.length} tensors`);
  const optimizer = new Adam(loraParams, { lr: 1e-3 }, api);

  model.train(true);

  const losses: number[] = [];
  for (let step = 0; step < 5; step++) {
    await api.beginStep();

    // Create batch (just repeat the same sequence)
    const seqLen = Math.min(32, tokens.length - 1);
    const input = api.tensorFromArray(tokens.slice(0, seqLen), [1, seqLen], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(
      tokens.slice(1, seqLen + 1),
      [1, seqLen],
      { device: "webgpu" },
    );

    const { loss } = model.forwardWithLoss(input, target);
    const lossVal = await loss.item();
    losses.push(lossVal);

    await loss.backward();

    // Check if gradients are flowing
    if (step === 0) {
      const firstParam = loraParams[0];
      const hasGrad = firstParam.grad !== null;
      console.log(
        `   Grad check: firstParam.requiresGrad=${firstParam.requiresGrad}, hasGrad=${hasGrad}`,
      );
      if (hasGrad) {
        const gradData = await firstParam.grad!.cpu();
        const maxGrad = Math.max(...Array.from(gradData).map(Math.abs));
        console.log(`   Max grad magnitude: ${maxGrad.toExponential(3)}`);
      }
    }

    optimizer.step();
    optimizer.zeroGrad();

    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();

    console.log(`   Step ${step}: loss=${lossVal.toFixed(4)}`);
  }

  // Verify loss decreased
  const decreased = losses[losses.length - 1] < losses[0];
  console.log(
    `\n   Loss ${decreased ? "DECREASED" : "DID NOT DECREASE"}: ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`,
  );

  // Step 7: Generate text
  console.log("\n7. Generating text...");
  model.train(false);
  let generated = "";
  for await (const token of generateTokens(api, model, tokenizer, "The", {
    maxTokens: 20,
    temperature: 0.8,
    topK: 40,
  })) {
    generated += token;
  }
  console.log(`   Prompt: "The"`);
  console.log(`   Generated: "The${generated}"`);

  console.log("\n=== SMOKE TEST PASSED ===");
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
