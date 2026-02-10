/**
 * Model loading state store using Svelte 5 runes.
 */

import { Torchlette, initWebGPU, getWebGPUInitError } from 'torchlette/browser';
import { GPT2WithLoRA, GPT2_SMALL_CONFIG } from '$lib/torchlette/gpt2-lora';
import { createLoRAConfig } from '$lib/torchlette/lora';
import { GPT2Tokenizer } from '$lib/torchlette/tokenizer';
import { fetchGPT2Weights, fetchTokenizer } from '$lib/torchlette/weights';

// State
let isLoading = $state(false);
let isLoaded = $state(false);
let loadProgress = $state(0);
let loadStatus = $state('');
let error = $state<string | null>(null);
let webgpuSupported = $state<boolean | null>(null);

// Model instances
let api = $state<Torchlette | null>(null);
let model = $state<GPT2WithLoRA | null>(null);
let tokenizer = $state<GPT2Tokenizer | null>(null);

// LoRA config
let loraRank = $state(8);
let loraAlpha = $state(16);

/**
 * Check if WebGPU is supported.
 */
async function checkWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined') return false;
  if (!('gpu' in navigator)) return false;

  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Load the GPT-2 model with LoRA.
 */
async function loadModel(): Promise<void> {
  if (isLoading || isLoaded) return;

  isLoading = true;
  error = null;
  loadProgress = 0;
  loadStatus = 'Checking WebGPU support...';

  try {
    // Check WebGPU
    webgpuSupported = await checkWebGPU();
    if (!webgpuSupported) {
      throw new Error('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
    }

    // Initialize WebGPU
    loadStatus = 'Initializing WebGPU...';
    loadProgress = 5;
    const ok = await initWebGPU();
    if (!ok) {
      throw new Error(getWebGPUInitError() || 'Failed to initialize WebGPU');
    }

    // Create Torchlette instance with optimizations enabled
    // - enableFusion: fuse elementwise ops into single kernels
    // - enableMemoryPlanning: reuse buffers to reduce memory
    api = new Torchlette('webgpu', {
      enableFusion: true,
      enableMemoryPlanning: true,
    });
    loadProgress = 10;

    // Load tokenizer
    loadStatus = 'Loading tokenizer...';
    const tokenizerData = await fetchTokenizer((loaded, total, status) => {
      loadStatus = status;
    });

    tokenizer = new GPT2Tokenizer();
    tokenizer.load(tokenizerData.vocab, tokenizerData.merges);
    loadProgress = 15;

    // Load weights
    loadStatus = 'Downloading model weights (~500MB)...';
    const weights = await fetchGPT2Weights((loaded, total, status) => {
      loadStatus = status;
      if (total > 0) {
        loadProgress = 15 + Math.round((loaded / total) * 75);
      }
    });

    loadProgress = 90;
    loadStatus = 'Initializing model...';

    // Create model with LoRA
    const loraConfig = createLoRAConfig(loraRank, loraAlpha);
    model = new GPT2WithLoRA(api, GPT2_SMALL_CONFIG, loraConfig, 'webgpu');

    // Load base weights
    model.loadBaseWeights(weights);

    loadProgress = 100;
    loadStatus = 'Ready!';
    isLoaded = true;
  } catch (e) {
    error = e instanceof Error ? e.message : 'Unknown error occurred';
    loadProgress = 0;
    loadStatus = '';
  } finally {
    isLoading = false;
  }
}

/**
 * Reset the model (for retraining with different config).
 */
function resetModel(): void {
  model = null;
  isLoaded = false;
  loadProgress = 0;
  loadStatus = '';
  error = null;
}

// Export store
export const modelStore = {
  // Getters
  get isLoading() { return isLoading; },
  get isLoaded() { return isLoaded; },
  get loadProgress() { return loadProgress; },
  get loadStatus() { return loadStatus; },
  get error() { return error; },
  get webgpuSupported() { return webgpuSupported; },
  get api() { return api; },
  get model() { return model; },
  get tokenizer() { return tokenizer; },
  get loraRank() { return loraRank; },
  get loraAlpha() { return loraAlpha; },

  // Setters
  set loraRank(v: number) { loraRank = v; },
  set loraAlpha(v: number) { loraAlpha = v; },

  // Actions
  loadModel,
  resetModel,
  checkWebGPU,
};
