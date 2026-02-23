/**
 * DistilGPT-2 Fine-tuning Demo
 *
 * Demonstrates loading pretrained distilgpt2, showing original outputs,
 * fine-tuning on Shakespeare sonnets, and showing outputs at various steps.
 *
 * Run with: TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/finetune-demo.ts
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU, getBufferPoolStats, getGPUMemoryStats } from "../../src/backend/webgpu";
import { storageTracker } from "../../src/engine/lazy";
import { GPT2, DISTILGPT2_CONFIG } from "./model";
import { GPT2Tokenizer } from "./data";
import { loadPretrainedGPT2 } from "./loader";
import { Adam, GradScaler } from "../../src/optim";

// ============================================================================
// Shakespeare Training Data
// ============================================================================

const SHAKESPEARE_SONNETS = `
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.

When I do count the clock that tells the time,
And see the brave day sunk in hideous night;
When I behold the violet past prime,
And sable curls all silver'd o'er with white.

Let me not to the marriage of true minds
Admit impediments. Love is not love
Which alters when it alteration finds,
Or bends with the remover to remove.

My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, black wires grow on her head.

From fairest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper should by time decrease,
His tender heir might bear his memory.

So is it not with me as with that Muse,
Stirred by a painted beauty to his verse,
Who heaven itself for ornament doth use
And every fair with his fair doth rehearse.

Being your slave, what should I do but tend
Upon the hours and times of your desire?
I have no precious time at all to spend,
Nor services to do, till you require.

When in disgrace with fortune and men's eyes,
I all alone beweep my outcast state,
And trouble deaf heaven with my bootless cries,
And look upon myself, and curse my fate.
`.trim();

// ============================================================================
// Generation Helper
// ============================================================================

async function generateText(
  api: Torchlette,
  model: GPT2,
  tokenizer: GPT2Tokenizer,
  prompt: string,
  maxTokens: number,
  temperature = 0.8
): Promise<string> {
  const tokens = tokenizer.encode(prompt);
  const generated = [...tokens];

  for (let i = 0; i < maxTokens; i++) {
    // Truncate to block size if needed
    const contextTokens = generated.slice(-DISTILGPT2_CONFIG.blockSize);

    // Use tidy to automatically clean up intermediate tensors from forward pass
    const logits = api.tidy(() => {
      const inputTensor = api.tensorFromArray(contextTokens, [1, contextTokens.length], {
        device: "webgpu",
      });
      // Forward pass creates many intermediate tensors that will be auto-disposed
      return model.forward(inputTensor);
    });

    const logitsData = await logits.cpu();

    // Get last position logits (stride is paddedVocabSize, take first vocabSize)
    const seqLen = contextTokens.length;
    const vocabSize = tokenizer.vocabSize;
    const stride = model.paddedVocabSize;
    const startIdx = (seqLen - 1) * stride;
    const lastLogits = Array.from(logitsData).slice(startIdx, startIdx + vocabSize);

    // Apply temperature
    const scaledLogits = lastLogits.map((x) => x / temperature);

    // Softmax
    const maxLogit = Math.max(...scaledLogits);
    const expLogits = scaledLogits.map((x) => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((x) => x / sumExp);

    // Sample from distribution
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

    // Cleanup the logits tensor (tidy already cleaned up intermediates)
    logits.dispose();

    // Call markStep to trigger GPU buffer cleanup
    // This is necessary because lazy execution defers buffer destruction until markStep
    await api.markStep();

    // Stop on EOS
    if (nextToken === tokenizer.eosToken) {
      break;
    }
  }

  return tokenizer.decode(generated);
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("=".repeat(60));
  console.log("DistilGPT-2 Fine-tuning Demo: Learning Shakespeare");
  console.log("=".repeat(60));

  // Initialize WebGPU
  console.log("\n[1] Initializing WebGPU...");
  const success = await initWebGPU();
  if (!success) {
    console.error("WebGPU not available!");
    process.exit(1);
  }

  // Load tokenizer
  console.log("\n[2] Loading tokenizer...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const tokenizer = new GPT2Tokenizer();
  await tokenizer.load(modelDir);
  console.log(`  Vocab size: ${tokenizer.vocabSize}`);

  // Load pretrained model
  console.log("\n[3] Loading pretrained DistilGPT-2...");
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  // Test prompts
  const prompts = [
    "To be or not to be",
    "The summer sun",
    "Love is",
    "When I",
  ];

  // Show original outputs
  console.log("\n" + "=".repeat(60));
  console.log("ORIGINAL MODEL OUTPUTS (before fine-tuning)");
  console.log("=".repeat(60));

  for (const prompt of prompts) {
    console.log(`\nPrompt: "${prompt}"`);
    const output = await generateText(api, model, tokenizer, prompt, 30, 0.7);
    console.log(`Output: ${output}`);
  }

  // Prepare training data
  console.log("\n" + "=".repeat(60));
  console.log("FINE-TUNING ON SHAKESPEARE SONNETS");
  console.log("=".repeat(60));

  model.train();
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const scaler = new GradScaler(api, { initScale: 1024.0 });
  // Compiled forward pass — enables fusion + memory planning inside compile region
  // AMP: autocast wraps the compiled forward to use f16 for matmul, f32 for reductions
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });

  // Tokenize training data
  const trainingTokens = tokenizer.encode(SHAKESPEARE_SONNETS);
  console.log(`\nTraining tokens: ${trainingTokens.length}`);

  // Create training sequences (shorter for faster iteration)
  const seqLen = 512;  // Longer sequences for realistic profiling
  const sequences: number[][] = [];
  for (let i = 0; i < trainingTokens.length - seqLen; i += seqLen / 2) {
    sequences.push(trainingTokens.slice(i, i + seqLen));
  }
  console.log(`Training sequences: ${sequences.length}`);

  // Training loop - just train, show loss, skip intermediate generation
  const NUM_EPOCHS = 3;
  let globalStep = 0;

  for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
    console.log(`\n--- Epoch ${epoch + 1}/${NUM_EPOCHS} ---`);

    for (let i = 0; i < sequences.length; i++) {
      const seq = sequences[i];
      const inputData = seq.slice(0, -1);
      const targetData = seq.slice(1);

      const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
      const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

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

      // Call markStep to trigger GPU buffer cleanup
      await api.markStep();
      const t4 = performance.now();

      // Log loss and timing every 5 steps
      if (globalStep % 5 === 0) {
        const memStats = getGPUMemoryStats();
        const poolStats = getBufferPoolStats();
        const storageStats = storageTracker.stats();
        console.log(`Step ${globalStep}: loss = ${lossValue.toFixed(4)} | scale=${scaler.getScale().toFixed(0)} (fwd: ${(t1-t0).toFixed(0)}ms, bwd: ${(t2-t1).toFixed(0)}ms, opt: ${(t3-t2).toFixed(0)}ms, cleanup: ${(t4-t3).toFixed(0)}ms)`);
        console.log(`  Memory: ${(memStats.currentBytes / 1e9).toFixed(2)}GB / ${(memStats.limitBytes / 1e9).toFixed(2)}GB (${memStats.usagePercent.toFixed(1)}%), allocations: ${memStats.allocationCount}`);
        console.log(`  Pool: ${poolStats.pooledBuffers} buffers (${(poolStats.pooledBytes / 1e6).toFixed(1)}MB), pending: ${poolStats.pendingBuffers}`);
        console.log(`  Storages: total=${storageStats.totalStorages}, reachable=${storageStats.reachableStorages}`);
      }

      globalStep++;
    }
  }

  // Final outputs
  console.log("\n" + "=".repeat(60));
  console.log("FINAL MODEL OUTPUTS (after fine-tuning)");
  console.log("=".repeat(60));

  model.eval();
  for (const prompt of prompts) {
    console.log(`\nPrompt: "${prompt}"`);
    const output = await generateText(api, model, tokenizer, prompt, 50, 0.7);
    console.log(`Output: ${output}`);
  }

  console.log("\n" + "=".repeat(60));
  console.log("Demo complete!");
  console.log("=".repeat(60));
}

main().then(() => process.exit(0)).catch(e => { console.error(e); process.exit(1); });
