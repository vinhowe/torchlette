/**
 * Training store -- config, data, training loop, LoRA export.
 */

import { LoRATrainer, type TrainingConfig } from "$lib/torchlette/trainer";
import { serializeLoRAToSafetensors } from "$lib/torchlette/weights";
import { modelStore } from "./model.svelte";

const TINY_SHAKESPEARE_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

// Config
let rank = $state(64);
let alpha = $state(16);
let maxSteps = $state(200);
let batchSize = $state(1);
let seqLen = $state(128);
let lr = $state(5e-4);
let useAMP = $state(true);
let useCheckpointing = $state(true);

// Data
let dataSource = $state("");
let dataText = $state("");
let dataTokenCount = $state(0);
let dataLoading = $state(false);

// Training progress
let running = $state(false);
let step = $state(0);
let loss = $state(0);
let lossHistory = $state<number[]>([]);
// biome-ignore lint/style/useConst: Svelte $state runes require reassignment
let memoryHistory = $state<number[]>([]);
// biome-ignore lint/style/useConst: Svelte $state runes require reassignment
let memoryMB = $state(0);
let tokPerSec = $state(0);
let error = $state("");

// Export
let loraBlob = $state<ArrayBuffer | null>(null);
let trainer = $state<LoRATrainer | null>(null);

// Stop signal
let shouldStop = false;

const canTrain = $derived(
  dataTokenCount >= batchSize * (seqLen + 1) && !running && modelStore.isReady,
);

/**
 * Auto-fetch TinyShakespeare on first load.
 */
async function fetchShakespeare(): Promise<void> {
  if (dataText) return;
  dataLoading = true;
  try {
    const resp = await fetch(TINY_SHAKESPEARE_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();
    setData(text, "tinyshakespeare (1.1 MB)");
  } catch (e) {
    error = `Failed to fetch Shakespeare: ${e instanceof Error ? e.message : e}`;
  } finally {
    dataLoading = false;
  }
}

function setData(text: string, source: string): void {
  dataText = text;
  dataSource = source;
  if (modelStore.tokenizer) {
    dataTokenCount = modelStore.tokenizer.encode(text).length;
  } else {
    // rough estimate: ~4 chars per token
    dataTokenCount = Math.floor(text.length / 4);
  }
}

function handleFileDrop(file: File): void {
  const reader = new FileReader();
  reader.onload = () => {
    const text = reader.result as string;
    setData(text, `${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
  };
  reader.readAsText(file);
}

async function startTraining(): Promise<void> {
  if (
    !canTrain ||
    !modelStore.api ||
    !modelStore.model ||
    !modelStore.tokenizer
  )
    return;

  running = true;
  shouldStop = false;
  step = 0;
  loss = 0;
  lossHistory = [];
  memoryHistory = [];
  memoryMB = 0;
  tokPerSec = 0;
  error = "";
  loraBlob = null;

  try {
    trainer = new LoRATrainer(
      modelStore.api,
      modelStore.model,
      modelStore.tokenizer,
    );

    const config: TrainingConfig = {
      maxSteps,
      batchSize,
      seqLength: seqLen,
      learningRate: lr,
      useAMP,
      useCheckpointing,
    };

    await trainer.train(dataText, config, {
      onStepStart: (s) => {
        step = s;
      },
      onStepEnd: (s, l, timeMs, mem) => {
        step = s + 1;
        loss = l;
        lossHistory = [...lossHistory, l];
        if (mem !== undefined) {
          memoryMB = mem;
          memoryHistory = [...memoryHistory, mem];
        }
        tokPerSec = (batchSize * seqLen) / (timeMs / 1000);
      },
      shouldStop: () => shouldStop,
    });

    // Export weights
    const weights = await trainer.exportLoRAWeights();
    loraBlob = serializeLoRAToSafetensors(weights, {
      rank: String(rank),
      alpha: String(alpha),
      base_model: "openai-community/gpt2",
      target_modules: "c_attn",
    });
  } catch (e) {
    error = e instanceof Error ? e.message : "Training failed";
  } finally {
    running = false;
  }
}

function stopTraining(): void {
  shouldStop = true;
}

function downloadLoRA(): void {
  if (!loraBlob) return;
  const blob = new Blob([loraBlob], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "gpt2-lora.safetensors";
  a.click();
  URL.revokeObjectURL(url);
}

export const trainingStore = {
  // Config
  get rank() {
    return rank;
  },
  set rank(v: number) {
    rank = v;
  },
  get alpha() {
    return alpha;
  },
  set alpha(v: number) {
    alpha = v;
  },
  get maxSteps() {
    return maxSteps;
  },
  set maxSteps(v: number) {
    maxSteps = v;
  },
  get batchSize() {
    return batchSize;
  },
  set batchSize(v: number) {
    batchSize = v;
  },
  get seqLen() {
    return seqLen;
  },
  set seqLen(v: number) {
    seqLen = v;
  },
  get lr() {
    return lr;
  },
  set lr(v: number) {
    lr = v;
  },
  get useAMP() {
    return useAMP;
  },
  set useAMP(v: boolean) {
    useAMP = v;
  },
  get useCheckpointing() {
    return useCheckpointing;
  },
  set useCheckpointing(v: boolean) {
    useCheckpointing = v;
  },

  // Data
  get dataSource() {
    return dataSource;
  },
  get dataText() {
    return dataText;
  },
  get dataTokenCount() {
    return dataTokenCount;
  },
  get dataLoading() {
    return dataLoading;
  },

  // Progress
  get running() {
    return running;
  },
  get step() {
    return step;
  },
  get loss() {
    return loss;
  },
  get lossHistory() {
    return lossHistory;
  },
  get memoryHistory() {
    return memoryHistory;
  },
  get memoryMB() {
    return memoryMB;
  },
  get tokPerSec() {
    return tokPerSec;
  },
  get error() {
    return error;
  },
  get canTrain() {
    return canTrain;
  },

  // Export
  get loraBlob() {
    return loraBlob;
  },

  // Actions
  fetchShakespeare,
  setData,
  handleFileDrop,
  startTraining,
  stopTraining,
  downloadLoRA,
};
