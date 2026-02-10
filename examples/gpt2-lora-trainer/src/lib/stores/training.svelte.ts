/**
 * Training state store using Svelte 5 runes.
 */

import { LoRATrainer, type TrainingConfig } from '$lib/torchlette/trainer';
import { serializeLoRAToSafetensors } from '$lib/torchlette/weights';
import { modelStore } from './model.svelte';

// Training data state
let files = $state<Array<{ name: string; content: string; tokens: number }>>([]);

// Training config - reduced defaults for GPU memory efficiency
let maxSteps = $state(50);
let batchSize = $state(1);
let seqLength = $state(32);
let learningRate = $state(1e-4);

// Memory optimization options
let useAMP = $state(true); // Enable AMP by default for memory savings
let useCheckpointing = $state(true); // Enable checkpointing by default

// Training progress state
let isTraining = $state(false);
let currentStep = $state(0);
let currentLoss = $state(0);
let lossHistory = $state<number[]>([]);
let tokensPerSecond = $state(0);
let trainingError = $state<string | null>(null);

// Trained LoRA state
let loraWeights = $state<ArrayBuffer | null>(null);
let trainer = $state<LoRATrainer | null>(null);

// Derived state
const totalTokens = $derived(files.reduce((sum, f) => sum + f.tokens, 0));
const canStartTraining = $derived(
  files.length > 0 &&
  !isTraining &&
  modelStore.isLoaded &&
  totalTokens >= batchSize * (seqLength + 1)
);

/**
 * Add training data from a text file.
 */
function addTrainingData(content: string, name: string): void {
  if (!modelStore.tokenizer) return;

  const tokens = modelStore.tokenizer.encode(content).length;
  files = [...files, { name, content, tokens }];
}

/**
 * Remove a training file.
 */
function removeTrainingData(index: number): void {
  files = files.filter((_, i) => i !== index);
}

/**
 * Clear all training data.
 */
function clearTrainingData(): void {
  files = [];
}

/**
 * Start training.
 */
async function startTraining(): Promise<void> {
  if (!canStartTraining) return;
  if (!modelStore.api || !modelStore.model || !modelStore.tokenizer) return;

  isTraining = true;
  currentStep = 0;
  currentLoss = 0;
  lossHistory = [];
  tokensPerSecond = 0;
  trainingError = null;
  loraWeights = null;

  try {
    // Combine all training text
    const allText = files.map((f) => f.content).join('\n\n');

    // Create trainer
    trainer = new LoRATrainer(modelStore.api, modelStore.model, modelStore.tokenizer);

    const config: TrainingConfig = {
      maxSteps,
      batchSize,
      seqLength,
      learningRate,
      useAMP,
      useCheckpointing,
    };

    // Train with callbacks
    await trainer.train(allText, config, {
      onStepStart: (step) => {
        currentStep = step;
      },
      onStepEnd: (step, loss, timeMs) => {
        currentStep = step + 1;
        currentLoss = loss;
        lossHistory = [...lossHistory, loss];
        tokensPerSecond = (batchSize * seqLength) / (timeMs / 1000);
      },
      shouldStop: () => !isTraining,
    });

    // Export trained weights
    const weights = await trainer.exportLoRAWeights();
    loraWeights = serializeLoRAToSafetensors(weights, {
      rank: String(modelStore.loraRank),
      alpha: String(modelStore.loraAlpha),
      base_model: 'openai-community/gpt2',
      target_modules: 'c_attn',
    });
  } catch (e) {
    trainingError = e instanceof Error ? e.message : 'Training failed';
  } finally {
    isTraining = false;
  }
}

/**
 * Stop training.
 */
function stopTraining(): void {
  isTraining = false;
}

/**
 * Download trained LoRA weights.
 */
function downloadLoRA(): void {
  if (!loraWeights) return;

  const blob = new Blob([loraWeights], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = 'gpt2-lora.safetensors';
  a.click();

  URL.revokeObjectURL(url);
}

// Export store
export const trainingStore = {
  // Getters
  get files() { return files; },
  get maxSteps() { return maxSteps; },
  get batchSize() { return batchSize; },
  get seqLength() { return seqLength; },
  get learningRate() { return learningRate; },
  get useAMP() { return useAMP; },
  get useCheckpointing() { return useCheckpointing; },
  get isTraining() { return isTraining; },
  get currentStep() { return currentStep; },
  get currentLoss() { return currentLoss; },
  get lossHistory() { return lossHistory; },
  get tokensPerSecond() { return tokensPerSecond; },
  get trainingError() { return trainingError; },
  get loraWeights() { return loraWeights; },
  get totalTokens() { return totalTokens; },
  get canStartTraining() { return canStartTraining; },

  // Setters
  set maxSteps(v: number) { maxSteps = v; },
  set batchSize(v: number) { batchSize = v; },
  set seqLength(v: number) { seqLength = v; },
  set learningRate(v: number) { learningRate = v; },
  set useAMP(v: boolean) { useAMP = v; },
  set useCheckpointing(v: boolean) { useCheckpointing = v; },

  // Actions
  addTrainingData,
  removeTrainingData,
  clearTrainingData,
  startTraining,
  stopTraining,
  downloadLoRA,
};
