/**
 * Model engine: instantiate a GPT-2 at a Menagerie preset, load weights from an
 * HF repo, and serialize back to HF-layout safetensors. LoRA is disabled — we
 * train and snapshot the full base model (see exportBaseWeights / getBaseParameters).
 */
import type { Torchlette } from "torchlette";
import {
  GPT2WithLoRA,
  type GPT2Config,
  createLoRAConfig,
  parseSafetensors,
  serializeLoRAToSafetensors,
  type WeightData,
  type ProgressCallback,
} from "gpt2-browser";
import { resolveUrl } from "$lib/hf/repo";
import type { ModelConfig } from "$lib/hf/types";

export function toGpt2Config(c: ModelConfig): GPT2Config {
  return {
    vocabSize: c.vocabSize,
    blockSize: c.blockSize,
    numLayers: c.numLayers,
    numHeads: c.numHeads,
    embedDim: c.embedDim,
    dropoutRate: 0,
  };
}

/**
 * Instantiate a full-finetuning GPT-2 (LoRA disabled). If `weights` is given,
 * loads them; otherwise the model keeps its random init ("from scratch").
 */
export function createModel(
  api: Torchlette,
  config: ModelConfig,
  weights?: Map<string, WeightData>,
): GPT2WithLoRA {
  const model = new GPT2WithLoRA(api, toGpt2Config(config), createLoRAConfig(8), "webgpu");
  if (weights) {
    model.loadBaseWeights(weights);
  } else {
    model.initWeightsGPT2(); // from-scratch: nanoGPT-style init (not N(0,1))
  }
  model.disableAllLora();
  model.setFullFinetuning(true);
  return model;
}

/**
 * Fetch + parse a repo's `model.safetensors` (at a branch or pinned SHA).
 * Streams the download so `onProgress` can drive a progress bar: it reports
 * bytes during download (status starts "Downloading") then tensor counts during
 * parse — in both phases `loaded/total` is a 0..1 fraction.
 */
export async function loadWeightsFromRepo(
  repo: string,
  rev = "main",
  onProgress?: ProgressCallback,
): Promise<Map<string, WeightData>> {
  const res = await fetch(resolveUrl(repo, "model.safetensors", rev));
  if (!res.ok) throw new Error(`fetch weights failed: ${res.status} ${res.statusText}`);

  const total = Number(res.headers.get("content-length") || 0);
  let buffer: ArrayBuffer;
  if (res.body && total > 0) {
    const reader = res.body.getReader();
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress?.(
        loaded,
        total,
        `Downloading ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`,
      );
    }
    const merged = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) {
      merged.set(c, off);
      off += c.length;
    }
    buffer = merged.buffer;
  } else {
    buffer = await res.arrayBuffer();
  }

  onProgress?.(0, 1, "Parsing weights…");
  return parseSafetensors(buffer, onProgress);
}

/** Serialize a trained model to an HF-layout safetensors Blob, ready to upload. */
export async function serializeModel(model: GPT2WithLoRA): Promise<Blob> {
  const tensors = await model.exportBaseWeights();
  const buffer = serializeLoRAToSafetensors(tensors, { format: "menagerie", framework: "torchlette" });
  return new Blob([buffer], { type: "application/octet-stream" });
}
