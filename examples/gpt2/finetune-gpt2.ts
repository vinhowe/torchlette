/**
 * GPT-2 Small Fine-tuning on TinyShakespeare
 *
 * Downloads GPT-2 Small (124M params, 12 layers) and finetunes on the full
 * TinyShakespeare dataset (~300K tokens). Shows text generation before and
 * after training.
 *
 * Run with: npx tsx examples/gpt2/finetune-gpt2.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU, getBufferPoolStats, getGPUMemoryStats } from "../../src/backend/webgpu";
import { storageTracker } from "../../src/engine/lazy";
import { GPT2, GPT2_SMALL_CONFIG } from "./model";
import { loadPretrainedGPT2, downloadGPT2 } from "./loader";
import { GPT2Tokenizer, FineWebDataLoader, downloadTokenizer } from "./data";
import { Adam, GradScaler } from "../../src/optim";

// ============================================================================
// Configuration
// ============================================================================

const MODEL_NAME = process.env.MODEL ?? "gpt2";
const SEQ_LEN = parseInt(process.env.SEQ_LEN ?? "256", 10);
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE ?? "1", 10);
const LEARNING_RATE = parseFloat(process.env.LR ?? "3e-5");
const WARMUP_STEPS = 100;
const LOG_EVERY = parseInt(process.env.LOG_EVERY ?? "50", 10);
const MAX_STEPS = parseInt(process.env.MAX_STEPS ?? "0", 10) || Infinity;
const GENERATE_MAX_TOKENS = 200;
const GENERATE_TEMPERATURE = 0.8;

const MODEL_DIR = path.join(process.cwd(), "models", MODEL_NAME);
const DATA_DIR = path.join(process.cwd(), "data");
const DATA_FILE = path.join(DATA_DIR, "tinyshakespeare.txt");
const TINYSHAKESPEARE_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

// ============================================================================
// Download Helpers
// ============================================================================

async function ensureModelDownloaded(): Promise<void> {
  const safetensorsPath = path.join(MODEL_DIR, "model.safetensors");
  if (fs.existsSync(safetensorsPath)) {
    console.log(`  ${MODEL_NAME} weights already downloaded.`);
    return;
  }
  console.log(`  Downloading ${MODEL_NAME} weights...`);
  await downloadGPT2(MODEL_NAME, MODEL_DIR);
}

async function ensureTokenizerDownloaded(): Promise<void> {
  const vocabPath = path.join(MODEL_DIR, "vocab.json");
  const mergesPath = path.join(MODEL_DIR, "merges.txt");
  if (fs.existsSync(vocabPath) && fs.existsSync(mergesPath)) {
    console.log("  Tokenizer files already downloaded.");
    return;
  }
  console.log("  Downloading tokenizer...");
  await downloadTokenizer("gpt2", MODEL_DIR);
}

async function ensureDataDownloaded(): Promise<void> {
  if (fs.existsSync(DATA_FILE)) {
    console.log("  TinyShakespeare already downloaded.");
    return;
  }
  console.log("  Downloading TinyShakespeare...");
  await fs.promises.mkdir(DATA_DIR, { recursive: true });
  execSync(`curl -L -o "${DATA_FILE}" "${TINYSHAKESPEARE_URL}"`, {
    stdio: "inherit",
  });
}

// ============================================================================
// Learning Rate Schedule
// ============================================================================

function getLr(step: number, totalSteps: number): number {
  // Linear warmup for WARMUP_STEPS, then cosine decay to 10% of peak LR
  if (step < WARMUP_STEPS) {
    return LEARNING_RATE * (step + 1) / WARMUP_STEPS;
  }
  const decaySteps = totalSteps - WARMUP_STEPS;
  const progress = (step - WARMUP_STEPS) / decaySteps;
  const minLr = LEARNING_RATE * 0.1;
  return minLr + 0.5 * (LEARNING_RATE - minLr) * (1 + Math.cos(Math.PI * progress));
}

// ============================================================================
// Generation Helper
// ============================================================================

async function generateText(
  api: Torchlette,
  model: GPT2,
  tokenizer: GPT2Tokenizer,
  prompt: string,
  maxTokens: number,
  temperature: number,
): Promise<string> {
  const tokens = tokenizer.encode(prompt);
  const generated = [...tokens];

  for (let i = 0; i < maxTokens; i++) {
    const contextTokens = generated.slice(-GPT2_SMALL_CONFIG.blockSize);

    const logits = api.tidy(() => {
      const inputTensor = api.tensorFromArray(contextTokens, [1, contextTokens.length], {
        device: "webgpu",
      });
      return model.forward(inputTensor);
    });

    const logitsData = await logits.cpu();

    const seqLen = contextTokens.length;
    const vocabSize = tokenizer.vocabSize;
    const stride = model.paddedVocabSize;
    const startIdx = (seqLen - 1) * stride;
    const lastLogits = Array.from(logitsData).slice(startIdx, startIdx + vocabSize);

    // Temperature scaling + softmax sampling
    const scaledLogits = lastLogits.map((x) => x / temperature);
    const maxLogit = Math.max(...scaledLogits);
    const expLogits = scaledLogits.map((x) => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((x) => x / sumExp);

    const r = Math.random();
    let cumsum = 0;
    let nextToken = 0;
    for (let j = 0; j < probs.length; j++) {
      cumsum += probs[j];
      if (r < cumsum) {
        nextToken = j;
        break;
      }
    }

    generated.push(nextToken);
    logits.dispose();
    await api.markStep();

    if (nextToken === tokenizer.eosToken) break;
  }

  return tokenizer.decode(generated);
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("=".repeat(70));
  console.log("GPT-2 Small Fine-tuning on TinyShakespeare");
  console.log("  Model: GPT-2 Small (12 layers, 12 heads, 768 dim, ~124M params)");
  console.log(`  Data: TinyShakespeare (~300K tokens)`);
  console.log(`  Seq length: ${SEQ_LEN}, Batch size: ${BATCH_SIZE}, LR: ${LEARNING_RATE}`);
  console.log(`  Warmup: ${WARMUP_STEPS} steps, cosine decay`);
  console.log("=".repeat(70));

  // Step 1: Download model, tokenizer, and data
  console.log("\n[1] Ensuring model, tokenizer, and data are downloaded...");
  await ensureModelDownloaded();
  await ensureTokenizerDownloaded();
  await ensureDataDownloaded();

  // Step 2: Initialize WebGPU
  console.log("\n[2] Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("WebGPU not available!");
    process.exit(1);
  }

  // Step 3: Load tokenizer and model
  console.log("\n[3] Loading tokenizer...");
  const tokenizer = new GPT2Tokenizer();
  await tokenizer.load(MODEL_DIR);

  console.log("\n[4] Loading pretrained GPT-2 Small...");
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const model = await loadPretrainedGPT2(api, MODEL_DIR, { dropoutRate: 0.0 }, { device: "webgpu" });
  const params = model.parameters();

  // Step 4: Generate text before training
  console.log("\n" + "=".repeat(70));
  console.log("ORIGINAL MODEL OUTPUTS (before fine-tuning)");
  console.log("=".repeat(70));

  const prompts = [
    "To be or not to be",
    "The summer sun",
    "Love is",
    "When I",
  ];

  model.eval();
  for (const prompt of prompts) {
    console.log(`\nPrompt: "${prompt}"`);
    const output = await generateText(api, model, tokenizer, prompt, GENERATE_MAX_TOKENS, GENERATE_TEMPERATURE);
    console.log(`Output: ${output}`);
  }

  // Step 5: Set up training
  console.log("\n" + "=".repeat(70));
  console.log("FINE-TUNING ON TINYSHAKESPEARE");
  console.log("=".repeat(70));

  model.train();
  const optimizer = new Adam(params, { lr: LEARNING_RATE }, api);

  const scaler = new GradScaler(api, { initScale: 1024.0 });

  // Compiled forward pass with AMP + checkpointing
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  // Set up data loader (no shuffle — sequential text provides coherent context)
  const dataLoader = new FineWebDataLoader(
    api,
    { seqLength: SEQ_LEN, batchSize: BATCH_SIZE, dataPath: DATA_FILE },
    { device: "webgpu" },
  );
  await dataLoader.init(MODEL_DIR);

  const totalBatches = dataLoader.numBatches;
  console.log(`\nTotal batches per epoch: ${totalBatches}`);
  console.log(`Training for 1 epoch...\n`);

  // Step 6: Training loop (1 epoch)
  const epochStart = performance.now();
  let totalTokens = 0;

  const stepsToRun = Math.min(totalBatches, MAX_STEPS);
  for (let step = 0; step < stepsToRun; step++) {
    // Apply LR schedule (linear warmup + cosine decay)
    const currentLr = getLr(step, stepsToRun);
    (optimizer as any).lr = currentLr;

    const { input, target } = await dataLoader.nextBatch();

    const t0 = performance.now();
    // Forward pass inside compile region with AMP autocast
    const loss = compiledForward(input, target);
    const lossValue = await loss.item();
    const t1 = performance.now();

    // Scale loss for mixed precision, then backward
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();
    const t2 = performance.now();

    // Unscale gradients, check for NaN/Inf, step optimizer
    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    optimizer.zeroGrad();
    const t3 = performance.now();

    scaledLoss.dispose();
    loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();
    const t4 = performance.now();

    totalTokens += SEQ_LEN * BATCH_SIZE;

    if (step % LOG_EVERY === 0 || step < 10 || step === stepsToRun - 1) {
      const memStats = getGPUMemoryStats();
      const poolStats = getBufferPoolStats();
      const storageStats = storageTracker.stats();
      const elapsed = (performance.now() - epochStart) / 1000;
      const tokPerSec = totalTokens / elapsed;

      console.log(
        `Step ${step}/${stepsToRun}: loss=${lossValue.toFixed(4)} | ` +
        `lr=${currentLr.toExponential(2)} | ` +
        `fwd: ${(t1 - t0).toFixed(0)}ms, bwd: ${(t2 - t1).toFixed(0)}ms, ` +
        `opt: ${(t3 - t2).toFixed(0)}ms, cleanup: ${(t4 - t3).toFixed(0)}ms`
      );
      console.log(
        `  Memory: ${(memStats.currentBytes / 1e9).toFixed(2)}GB / ` +
        `${(memStats.limitBytes / 1e9).toFixed(2)}GB ` +
        `(${memStats.usagePercent.toFixed(1)}%) | ` +
        `Pool: ${poolStats.pooledBuffers} bufs (${(poolStats.pooledBytes / 1e6).toFixed(1)}MB) | ` +
        `Storages: ${storageStats.totalStorages} total, ${storageStats.reachableStorages} reachable | ` +
        `${tokPerSec.toFixed(0)} tok/s`
      );
    }
  }

  const epochElapsed = (performance.now() - epochStart) / 1000;
  console.log(`\nEpoch complete in ${epochElapsed.toFixed(1)}s (${totalTokens} tokens, ${(totalTokens / epochElapsed).toFixed(0)} tok/s)`);

  // Step 7: Generate text after training
  console.log("\n" + "=".repeat(70));
  console.log("FINE-TUNED MODEL OUTPUTS (after 1 epoch on TinyShakespeare)");
  console.log("=".repeat(70));

  model.eval();
  for (const prompt of prompts) {
    console.log(`\nPrompt: "${prompt}"`);
    const output = await generateText(api, model, tokenizer, prompt, GENERATE_MAX_TOKENS, GENERATE_TEMPERATURE);
    console.log(`Output: ${output}`);
  }

  console.log("\n" + "=".repeat(70));
  console.log("Done!");
  console.log("=".repeat(70));
}

main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
