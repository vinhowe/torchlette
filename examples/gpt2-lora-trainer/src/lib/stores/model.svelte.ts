/**
 * Model store -- WebGPU init, weight download, model + tokenizer creation.
 */

import {
  getWebGPUInitError,
  initWebGPU,
  setGPUMemoryLimit,
  Torchlette,
} from "torchlette";
import { GPT2_SMALL_CONFIG, GPT2WithLoRA } from "gpt2-browser";
import { createLoRAConfig } from "gpt2-browser";
import { GPT2Tokenizer } from "gpt2-browser";
import { fetchGPT2Weights, fetchTokenizer } from "gpt2-browser";

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

    // Limit GPU memory to 8 GB to avoid browser OOM
    setGPUMemoryLimit(8 * 1024 * 1024 * 1024);

    api = new Torchlette("webgpu", {
      enableFusion: true,
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

    // Materialize all weight tensors on GPU before declaring ready.
    // Without this, the lazy copy_ nodes from loadBaseWeights remain pending
    // and can fail with "Input not ready" when training starts later.
    await api.markStep();

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

/**
 * Load model for pretraining from scratch.
 * Skips 500MB weight download. Initializes randomly with seed 42
 * (matching V100 DiLoCo agent for identical starting weights).
 */
async function loadForPretraining(): Promise<void> {
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
    progress = 10;
    const ok = await initWebGPU();
    if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

    setGPUMemoryLimit(8 * 1024 * 1024 * 1024);

    api = new Torchlette("webgpu", { enableFusion: true });
    api.manualSeed(42); // Must match V100 agent seed
    progress = 20;

    progressText = "Loading tokenizer...";
    const tokData = await fetchTokenizer((_, __, s) => {
      progressText = s;
    });
    tokenizer = new GPT2Tokenizer();
    tokenizer.load(tokData.vocab, tokData.merges);
    progress = 40;

    progressText = "Initializing random weights (seed 42)...";
    const loraConfig = createLoRAConfig(1, 1); // rank=1 alpha=1 to match V100
    // GPT-2 Small (124M) — must match V100 agent config (tools/diloco-webrtc-agent.ts)
    const PRETRAIN_CONFIG = {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 12,
      numHeads: 12,
      embedDim: 768,
      dropoutRate: 0,
    };
    model = new GPT2WithLoRA(api, PRETRAIN_CONFIG, loraConfig, "webgpu");

    // Random init matching V100: normal_(0, 0.02) for 2D+ params
    const { normal_ } = await import("../../../../../src/nn/init");
    const params = model.getAllParameters();
    for (const p of params) {
      if (p.shape.length >= 2) {
        normal_(api, p, 0, 0.02);
      }
    }
    progress = 80;

    model.setFullFinetuning(true);
    await api._runtime().forceAllPending();
    const p0Init = await params[0].cpu();
    console.log(
      "[init-check] browser param[0] first4:",
      Array.from(p0Init.slice(0, 4)).map((v: number) => v.toFixed(8)),
    );
    console.log(
      "[init-check] expected V100:  [-0.00575741,-0.02528063,0.01242364,-0.00770933]",
    );
    console.log("[init-check] nParams:", params.length);
    await api.markStep();

    progress = 100;
    progressText = "Ready (from scratch)";
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
  loadForPretraining,
  checkWebGPU,
};
