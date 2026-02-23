/**
 * GPT-2 Memorization Test
 *
 * Validates that the training loop works by training GPT-2 to memorize
 * a small set of sequences, then verifying it can reproduce them.
 *
 * Run with: npm test -- test/gpt2-memorization.spec.ts
 * (WebGPU auto-detected; skip with TORCHLETTE_CPU_ONLY=1)
 */

import { describe, expect, test, beforeAll } from "vitest";
import { Torchlette, type Tensor } from "../src/frontend";
import { initWebGPU } from "../src/backend/webgpu";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { Adam } from "../src/optim";
import { cpuOnly, canUseWebGPU } from "./helpers/webgpu";
const hasWebGPU = !cpuOnly;

// Simple character-level tokenizer
class CharTokenizer {
  private charToId: Map<string, number> = new Map();
  private idToChar: Map<number, string> = new Map();
  readonly vocabSize: number;
  readonly padToken: number;

  constructor(chars: string) {
    // Build vocabulary from characters
    const uniqueChars = [...new Set(chars.split(""))];
    uniqueChars.sort();

    // Reserve 0 for padding
    this.charToId.set("<PAD>", 0);
    this.idToChar.set(0, "<PAD>");

    for (let i = 0; i < uniqueChars.length; i++) {
      this.charToId.set(uniqueChars[i], i + 1);
      this.idToChar.set(i + 1, uniqueChars[i]);
    }

    this.vocabSize = uniqueChars.length + 1; // +1 for PAD
    this.padToken = 0;
  }

  encode(text: string): number[] {
    return text.split("").map((c) => this.charToId.get(c) ?? this.padToken);
  }

  decode(ids: number[]): string {
    return ids
      .map((id) => this.idToChar.get(id) ?? "")
      .filter((c) => c !== "<PAD>")
      .join("");
  }
}

// Training sequences to memorize
const TRAINING_DATA = [
  "The capital of France is Paris.",
  "The capital of Germany is Berlin.",
  "The capital of Japan is Tokyo.",
  "The capital of Italy is Rome.",
  "The capital of Spain is Madrid.",
];

// Build vocabulary from all training data
const ALL_CHARS = TRAINING_DATA.join("");

describe("GPT-2 Memorization Test", { skip: !hasWebGPU, timeout: 300000 }, () => {
  let webgpuAvailable = false;

  beforeAll(async () => {
    const success = await canUseWebGPU();
    webgpuAvailable = success;
    if (!success) {
      console.warn("WebGPU not available - tests will be skipped");
    }
  });

  test("trains GPT-2 to memorize sequences and generates correct completions", async () => {
    if (!webgpuAvailable) return;

    console.log("\n=== GPT-2 Memorization Test ===\n");

    // Create tokenizer
    const tokenizer = new CharTokenizer(ALL_CHARS);
    console.log(`Vocabulary size: ${tokenizer.vocabSize}`);

    // Find max sequence length
    const maxLen = Math.max(...TRAINING_DATA.map((s) => s.length));
    console.log(`Max sequence length: ${maxLen}`);

    // GPT-2 config sized for memorizing 5 short sequences
    const config: GPT2Config = {
      vocabSize: tokenizer.vocabSize,
      blockSize: maxLen + 1,
      numLayers: 2,
      numHeads: 4,
      embedDim: 128,
      dropoutRate: 0.0, // Disable dropout for memorization
    };

    console.log(`Model config: ${config.numLayers} layers, ${config.embedDim} embed dim`);

    // Create model and optimizer
    const api = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: true,
    });

    const model = new GPT2(api, config, { device: "webgpu" });
    model.train();

    const optimizer = new Adam(model.parameters(), { lr: 0.01 }, api);

    // Prepare training data
    const encodedSequences = TRAINING_DATA.map((s) => tokenizer.encode(s));

    // Pad sequences to same length
    const paddedSequences = encodedSequences.map((seq) => {
      const padded = [...seq];
      while (padded.length < maxLen) {
        padded.push(tokenizer.padToken);
      }
      return padded;
    });

    // Create input (all but last token) and target (all but first token)
    const batchSize = TRAINING_DATA.length;
    const seqLen = maxLen - 1;

    const inputData: number[] = [];
    const targetData: number[] = [];

    for (const seq of paddedSequences) {
      inputData.push(...seq.slice(0, -1));
      targetData.push(...seq.slice(1));
    }

    const inputTensor = api.tensorFromArray(inputData, [batchSize, seqLen], {
      device: "webgpu",
    });
    const targetTensor = api.tensorFromArray(targetData, [batchSize, seqLen], {
      device: "webgpu",
    });

    // Training loop
    const NUM_STEPS = 300;
    const LOG_EVERY = 50;
    const TARGET_LOSS = 0.005;

    console.log(`\nTraining for up to ${NUM_STEPS} steps (target loss: ${TARGET_LOSS})...\n`);

    let finalLoss = Infinity;
    let step = 0;

    for (step = 0; step < NUM_STEPS; step++) {
      // Forward pass
      const { loss } = model.forwardWithLoss(inputTensor, targetTensor);
      if (!loss) throw new Error("Loss is null");

      const lossValue = await loss.item();

      // Backward pass
      await loss.backward();

      // Use stepAsync to force parameter updates immediately.
      // This ensures gradient buffers are read before zeroGrad disposes them.
      await optimizer.stepAsync();
      optimizer.zeroGrad();

      // Dispose loss tensor
      loss.dispose();

      finalLoss = lossValue;

      if ((step + 1) % LOG_EVERY === 0) {
        console.log(`Step ${step + 1}: loss = ${lossValue.toFixed(4)}`);
      }

      // Early stopping if loss is low enough
      if (lossValue < TARGET_LOSS) {
        console.log(`\nReached target loss at step ${step + 1}`);
        break;
      }
    }

    console.log(`\nFinal loss: ${finalLoss.toFixed(4)} after ${step + 1} steps`);

    // Test generation
    model.eval();
    console.log("\n=== Testing Generation ===\n");

    // Test prompts and expected completions
    const testCases = [
      { prompt: "The capital of France is ", expected: "Paris." },
      { prompt: "The capital of Germany is ", expected: "Berlin." },
      { prompt: "The capital of Japan is ", expected: "Tokyo." },
    ];

    let passCount = 0;

    for (const { prompt, expected } of testCases) {
      const generated = await generateText(api, model, tokenizer, prompt, prompt.length + expected.length + 2);
      const completion = generated.slice(prompt.length);

      const passed = completion.startsWith(expected.slice(0, -1)); // Allow slight variation
      if (passed) passCount++;

      console.log(`Prompt: "${prompt}"`);
      console.log(`Expected: "${expected}"`);
      console.log(`Generated: "${completion}"`);
      console.log(`Status: ${passed ? "PASS" : "FAIL"}\n`);
    }

    console.log(`\nResults: ${passCount}/${testCases.length} passed`);

    // Assert at least 2/3 pass (allow some tolerance)
    expect(passCount).toBeGreaterThanOrEqual(2);
    expect(finalLoss).toBeLessThan(1.0);
  });

  test("overfits on a single sequence", async () => {
    if (!webgpuAvailable) return;

    console.log("\n=== Single Sequence Overfit Test ===\n");

    const sequence = "Hello World!";
    const tokenizer = new CharTokenizer(sequence);

    const config: GPT2Config = {
      vocabSize: tokenizer.vocabSize,
      blockSize: sequence.length + 1,
      numLayers: 1,
      numHeads: 1,
      embedDim: 32,
      dropoutRate: 0.0,
    };

    const api = new Torchlette("webgpu", {
      enableFusion: false,
      enableMemoryPlanning: true,
    });

    const model = new GPT2(api, config, { device: "webgpu" });
    model.train();

    const optimizer = new Adam(model.parameters(), { lr: 0.01 }, api);

    const encoded = tokenizer.encode(sequence);
    const inputData = encoded.slice(0, -1);
    const targetData = encoded.slice(1);

    const inputTensor = api.tensorFromArray(inputData, [1, inputData.length], {
      device: "webgpu",
    });
    const targetTensor = api.tensorFromArray(targetData, [1, targetData.length], {
      device: "webgpu",
    });

    // Train (1000 steps max for reliable convergence on different GPU backends)
    const NUM_STEPS = 1000;
    let finalLoss = Infinity;

    console.log(`Training on: "${sequence}"`);
    console.log(`Sequence length: ${sequence.length}`);

    for (let step = 0; step < NUM_STEPS; step++) {
      const { loss } = model.forwardWithLoss(inputTensor, targetTensor);
      if (!loss) throw new Error("Loss is null");

      const lossValue = await loss.item();
      await loss.backward();
      await optimizer.stepAsync();
      optimizer.zeroGrad();
      loss.dispose();

      finalLoss = lossValue;

      if ((step + 1) % 100 === 0) {
        console.log(`Step ${step + 1}: loss = ${lossValue.toFixed(4)}`);
      }

      if (lossValue < 0.001) {
        console.log(`Converged at step ${step + 1}`);
        break;
      }
    }

    console.log(`Final loss: ${finalLoss.toFixed(4)}`);

    // Test
    model.eval();
    const generated = await generateText(api, model, tokenizer, "H", sequence.length);
    console.log(`\nPrompt: "H"`);
    console.log(`Expected: "${sequence}"`);
    console.log(`Generated: "${generated}"`);

    // Should reproduce the sequence
    expect(finalLoss).toBeLessThan(0.5);
    expect(generated.slice(0, 3)).toBe(sequence.slice(0, 3)); // At least first few chars
  });
});

/**
 * Generate text using the model.
 */
async function generateText(
  api: Torchlette,
  model: GPT2,
  tokenizer: CharTokenizer,
  prompt: string,
  maxLen: number,
  debug = false
): Promise<string> {
  const tokens = tokenizer.encode(prompt);
  const generated = [...tokens];
  const vocabSize = tokenizer.vocabSize;
  const stride = model.paddedVocabSize; // logits have paddedVocabSize columns

  if (debug) {
    console.log(`  [gen] prompt tokens: [${tokens.join(", ")}]`);
    console.log(`  [gen] vocabSize: ${vocabSize}, paddedVocabSize: ${stride}`);
  }

  for (let i = 0; i < maxLen - prompt.length; i++) {
    // Create input tensor from current tokens
    const inputTensor = api.tensorFromArray(generated, [1, generated.length], {
      device: "webgpu",
    });

    // Get logits - shape is [1, seqLen, paddedVocabSize]
    const logits = model.forward(inputTensor);
    const logitsData = await logits.cpu();

    if (debug && i === 0) {
      console.log(`  [gen] logits shape: [1, ${generated.length}, ${stride}]`);
      console.log(`  [gen] logitsData length: ${logitsData.length}`);
    }

    // Get last position logits
    // logits shape: [1, seqLen, paddedVocabSize]
    // We want logits[0, -1, :vocabSize] at offset (seqLen-1) * paddedVocabSize
    const seqLen = generated.length;
    const startIdx = (seqLen - 1) * stride;
    const lastLogits = Array.from(logitsData).slice(startIdx, startIdx + vocabSize);

    if (debug && i === 0) {
      console.log(`  [gen] startIdx: ${startIdx}, lastLogits length: ${lastLogits.length}`);
      console.log(`  [gen] lastLogits first 5: [${lastLogits.slice(0, 5).map(v => v.toFixed(2)).join(", ")}]`);
    }

    // Greedy sampling: pick the token with highest logit
    let maxIdx = 0;
    let maxVal = lastLogits[0];
    for (let j = 1; j < lastLogits.length; j++) {
      if (lastLogits[j] > maxVal) {
        maxVal = lastLogits[j];
        maxIdx = j;
      }
    }

    if (debug) {
      console.log(`  [gen] step ${i}: maxIdx=${maxIdx}, maxVal=${maxVal.toFixed(2)}, char="${tokenizer.decode([maxIdx])}"`);
    }

    generated.push(maxIdx);

    // Cleanup
    inputTensor.dispose();
    logits.dispose();

    // Stop if we hit padding
    if (maxIdx === tokenizer.padToken) {
      if (debug) console.log(`  [gen] hit PAD token, stopping`);
      break;
    }
  }

  if (debug) {
    console.log(`  [gen] final tokens: [${generated.join(", ")}]`);
  }

  return tokenizer.decode(generated);
}
