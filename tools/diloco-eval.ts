/**
 * Evaluate a DiLoCo checkpoint by generating text and computing perplexity.
 *
 * Usage:
 *   npx tsx tools/diloco-eval.ts /tmp/diloco-XXX/checkpoint
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const CHECKPOINT_DIR = process.argv[2];
const MODEL_DIR = process.env.MODEL ?? "gpt2";
const SEED = parseInt(process.env.SEED ?? "42", 10);

const GPT2_CONFIG = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

async function main() {
  if (!CHECKPOINT_DIR || !fs.existsSync(CHECKPOINT_DIR)) {
    console.error("Usage: npx tsx tools/diloco-eval.ts <checkpoint-dir>");
    console.error(
      "  e.g. npx tsx tools/diloco-eval.ts /tmp/diloco-XXX/checkpoint",
    );
    process.exit(1);
  }

  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU not available");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(SEED);

  // Create model with same architecture
  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    GPT2_CONFIG,
    { rank: 1, alpha: 1 },
    "webgpu",
  );

  // Load checkpoint weights
  const shapes: number[][] = JSON.parse(
    fs.readFileSync(path.join(CHECKPOINT_DIR, "shapes.json"), "utf-8"),
  );
  const params = model.getAllParameters();
  model.setFullFinetuning(true);

  console.log(`Loading checkpoint from ${CHECKPOINT_DIR}...`);
  await api.beginStep();
  for (let i = 0; i < params.length; i++) {
    const buf = fs.readFileSync(path.join(CHECKPOINT_DIR, `param-${i}.bin`));
    const f32 = new Float32Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 4,
    );
    api.copy_(
      params[i],
      api.tensorFromArray(f32, shapes[i], { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();
  console.log(`Loaded ${params.length} parameters`);

  // Load tokenizer
  const { GPT2Tokenizer } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer"
  );
  const vocabPath = path.join(process.cwd(), "models", MODEL_DIR, "vocab.json");
  const mergesPath = path.join(
    process.cwd(),
    "models",
    MODEL_DIR,
    "merges.txt",
  );
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(
    JSON.parse(fs.readFileSync(vocabPath, "utf-8")),
    fs
      .readFileSync(mergesPath, "utf-8")
      .split("\n")
      .filter((l: string) => l && !l.startsWith("#")),
  );

  // Evaluate perplexity on a held-out text
  console.log("\n--- Perplexity ---");
  const evalText =
    "The quick brown fox jumps over the lazy dog. In the beginning was the word. To be or not to be, that is the question.";
  const evalTokens = tokenizer.encode(evalText);
  const seqLen = Math.min(64, evalTokens.length - 1);

  await api.beginStep();
  const input = api.tensorFromArray(evalTokens.slice(0, seqLen), [1, seqLen], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(
    evalTokens.slice(1, seqLen + 1),
    [1, seqLen],
    { device: "webgpu" },
  );

  model.train(false);
  const { loss } = model.forwardWithLoss(input, target);
  const lossVal = await loss.item();
  const perplexity = Math.exp(lossVal);
  console.log(`  Loss: ${lossVal.toFixed(4)}`);
  console.log(`  Perplexity: ${perplexity.toFixed(2)}`);
  api.endStep();
  await api.markStep();

  // Generate text
  console.log("\n--- Generation ---");
  const { generateTokens } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/inference"
  );

  const prompts = ["The", "Once upon a time", "In the darkness"];
  for (const prompt of prompts) {
    let text = "";
    for await (const token of generateTokens(api, model, tokenizer, prompt, {
      maxTokens: 50,
      temperature: 0.8,
      topK: 40,
    })) {
      text += token;
    }
    console.log(`  "${prompt}" → "${prompt}${text.slice(0, 200)}"`);
  }

  console.log("\nDone.");
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FATAL:", e);
  process.exit(1);
});
