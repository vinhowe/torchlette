/**
 * Model store -- WebGPU init, weight download, model + tokenizer creation.
 */

import { getWebGPUInitError, initWebGPU, Torchlette } from "torchlette/browser";
import { GPT2_SMALL_CONFIG, GPT2WithLoRA } from "$lib/torchlette/gpt2-lora";
import { createLoRAConfig } from "$lib/torchlette/lora";
import { GPT2Tokenizer } from "$lib/torchlette/tokenizer";
import { fetchGPT2Weights, fetchTokenizer } from "$lib/torchlette/weights";

type Status = "idle" | "loading" | "ready" | "error";

let status = $state<Status>("idle");
let progress = $state(0); // 0-100
let progressText = $state("");
let errorMsg = $state("");
let webgpuOk = $state<boolean | null>(null);

let api = $state<Torchlette | null>(null);
let model = $state<GPT2WithLoRA | null>(null);
let tokenizer = $state<GPT2Tokenizer | null>(null);

async function checkWebGPU(): Promise<boolean> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return (await navigator.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

async function load(loraRank: number, loraAlpha: number): Promise<void> {
  if (status === "loading" || status === "ready") return;

  status = "loading";
  errorMsg = "";
  progress = 0;
  progressText = "Checking WebGPU...";

  try {
    webgpuOk = await checkWebGPU();
    if (!webgpuOk)
      throw new Error("WebGPU not supported. Use Chrome 113+ or Edge 113+.");

    progressText = "Initializing WebGPU...";
    progress = 5;
    const ok = await initWebGPU();
    if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

    api = new Torchlette("webgpu", {
      enableFusion: true,
      enableMemoryPlanning: true,
    });
    progress = 10;

    progressText = "Loading tokenizer...";
    const tokData = await fetchTokenizer((_, __, s) => {
      progressText = s;
    });
    tokenizer = new GPT2Tokenizer();
    tokenizer.load(tokData.vocab, tokData.merges);
    progress = 15;

    progressText = "Downloading weights (~500 MB)...";
    const weights = await fetchGPT2Weights((loaded, total, s) => {
      progressText = s;
      if (total > 0) progress = 15 + Math.round((loaded / total) * 75);
    });

    progress = 90;
    progressText = "Initializing model...";

    const loraConfig = createLoRAConfig(loraRank, loraAlpha);
    model = new GPT2WithLoRA(api, GPT2_SMALL_CONFIG, loraConfig, "webgpu");
    model.loadBaseWeights(weights);

    progress = 100;
    progressText = "Ready";
    status = "ready";
  } catch (e) {
    errorMsg = e instanceof Error ? e.message : String(e);
    status = "error";
    progress = 0;
    progressText = "";
  }
}

export const modelStore = {
  get status() {
    return status;
  },
  get progress() {
    return progress;
  },
  get progressText() {
    return progressText;
  },
  get error() {
    return errorMsg;
  },
  get webgpuOk() {
    return webgpuOk;
  },
  get api() {
    return api;
  },
  get model() {
    return model;
  },
  get tokenizer() {
    return tokenizer;
  },
  get isReady() {
    return status === "ready";
  },

  load,
  checkWebGPU,
};
