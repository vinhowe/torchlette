/**
 * In-browser steering engine, running inside a Web Worker: WebGPU init, weight
 * streaming, contrastive-vector computation, and steered generation all happen
 * off the main thread — the UI cannot freeze, and engine failures surface as
 * messages instead of dead tabs.
 *
 * Protocol (worker ← main):
 *   { type: "load", modelId, weightDtype? }
 *   { type: "computeVector", id, posPrompt, negPrompt, layer }
 *   { type: "generate", id, prompt, alpha, maxNewTokens? }
 * (worker → main):
 *   { type: "progress", loaded, total, status }
 *   { type: "loaded", modelId, weightDtype, numLayers, hiddenSize }
 *   { type: "tensor", ev }
 *   { type: "vector", id, layer, hiddenSize, posPrompt, negPrompt }
 *   { type: "delta", id, delta } | { type: "replace", id, text }
 *   { type: "done", id, stats }
 *   { type: "error", id?, error }
 */

import { AutoTokenizer } from "@huggingface/transformers";
import {
  buildChatPrompt,
  loadQwen3FromUrl,
  QWEN3_STOP_TOKENS,
  sampleFromTopK,
  type Qwen3,
} from "qwen3-browser";
import {
  getGpuUncapturedErrorCount,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  Torchlette,
} from "torchlette";
import {
  computeSteeringVector,
  makeResidualHook,
  type SteeringVector,
} from "./steering";

let api: Torchlette | null = null;
let model: Qwen3 | null = null;
let tokenizerLike:
  | {
      encode: (t: string) => number[];
      decode: (ids: number[], o?: { skip_special_tokens?: boolean }) => string;
    }
  | null = null;
// Most recently computed steering vector — persisted on the GPU across
// generations (do NOT dispose between runs; the persistence contract holds it).
let steeringVec: SteeringVector | null = null;

const post = (msg: Record<string, unknown>) => {
  (self as unknown as Worker).postMessage(msg);
};

function heapMB(): string {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number } })
    .memory;
  return m ? ` · heap ${(m.usedJSHeapSize / 1e9).toFixed(1)}GB` : "";
}

async function handleLoad(modelId: string, requestedDtype?: "f16" | "f32") {
  if (!("gpu" in self.navigator)) {
    throw new Error(
      "WebGPU not available (need Chrome/Edge 121+ or Safari 18+).",
    );
  }
  const adapter = await (
    self.navigator as unknown as {
      gpu: { requestAdapter(): Promise<{ features: Set<string> } | null> };
    }
  ).gpu.requestAdapter();
  const weightDtype: "f16" | "f32" =
    requestedDtype ?? (adapter?.features?.has("shader-f16") ? "f16" : "f32");

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  api = new Torchlette("webgpu", { enableFusion: true });

  const device = getWebGPUDevice() as unknown as {
    lost?: Promise<{ reason: string; message: string }>;
    limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number };
  } | null;
  device?.lost?.then((info) =>
    post({
      type: "error",
      error: `GPU device lost (${info.reason}): ${info.message}`,
    }),
  );
  const limits = device?.limits;
  post({
    type: "progress",
    loaded: 0,
    total: 0,
    status:
      `Loading tokenizer… (weights as ${weightDtype}; maxBuffer ` +
      `${((limits?.maxBufferSize ?? 0) / 1e9).toFixed(1)}GB, maxBinding ` +
      `${((limits?.maxStorageBufferBindingSize ?? 0) / 1e9).toFixed(1)}GB)`,
  });
  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  tokenizerLike = {
    encode: (text: string) => tokenizer.encode(text) as number[],
    decode: (ids: number[], options?: { skip_special_tokens?: boolean }) =>
      tokenizer.decode(ids, options) as string,
  };

  model = await loadQwen3FromUrl(
    api,
    `https://huggingface.co/${modelId}/resolve/main`,
    {
      maxSeqLen: 2048,
      weightDtype,
      onProgress: (loaded, total, status) => {
        const errs = getGpuUncapturedErrorCount();
        post({
          type: "progress",
          loaded,
          total,
          status:
            status +
            heapMB() +
            (errs > 0 ? ` · ⚠ ${errs} GPU errors (see console)` : ""),
        });
      },
      onTensorEvent: (ev) => post({ type: "tensor", ev }),
    },
  );
  const errs = getGpuUncapturedErrorCount();
  if (errs > 0) {
    throw new Error(
      `Model loaded but ${errs} GPU validation errors occurred — weights are suspect (see worker console).`,
    );
  }
  // Loading a new model invalidates any prior steering vector.
  steeringVec = null;
  post({
    type: "loaded",
    modelId,
    weightDtype,
    numLayers: model.config.numLayers,
    hiddenSize: model.config.hiddenSize,
  });
}

async function handleComputeVector(
  id: number,
  posPrompt: string,
  negPrompt: string,
  layer: number,
) {
  if (!api || !model || !tokenizerLike) throw new Error("Model not loaded");
  steeringVec = await computeSteeringVector(
    api,
    model,
    tokenizerLike,
    posPrompt,
    negPrompt,
    layer,
  );
  post({
    type: "vector",
    id,
    layer: steeringVec.layer,
    hiddenSize: steeringVec.hiddenSize,
    posPrompt,
    negPrompt,
  });
}

/**
 * Chat generation: the prompt is wrapped in the Qwen3 chat template (user turn
 * + thinking-off assistant header) before prefill — Qwen3 is an INSTRUCT model
 * and degenerates into repetition on raw text. Then KV-decode with the static
 * cache. `alpha` builds a residualHook from the current steering vector
 * (alpha=0 or no vector → unsteered baseline). The steering VECTOR itself is
 * still derived from raw concept prompts (steering.ts) — that's the concept's
 * residual direction, independent of prompt format.
 */
async function handleGenerate(
  id: number,
  prompt: string,
  alpha: number,
  maxNewTokens: number,
) {
  if (!api || !model || !tokenizerLike) throw new Error("Model not loaded");
  const a = api;
  const m = model;
  const tok = tokenizerLike;
  const maxSeq = m.config.maxSeqLen;
  const vocab = m.config.vocabSize;
  const promptIds = tok.encode(
    buildChatPrompt([{ role: "user", content: prompt }]),
  );
  if (promptIds.length + 8 >= maxSeq) throw new Error("Prompt too long");
  const maxNew = Math.min(maxNewTokens, maxSeq - promptIds.length - 1);

  const residualHook = makeResidualHook(a, steeringVec, alpha);
  const temperature = 0.7;
  const topK = 20;
  const topP = 0.95;
  const K_PREFILTER = 64;

  const genIds: number[] = [];
  let prevText = "";
  const emit = (t: number) => {
    genIds.push(t);
    const text = tok.decode(genIds, { skip_special_tokens: true });
    if (text.startsWith(prevText)) {
      const delta = text.slice(prevText.length);
      if (delta) post({ type: "delta", id, delta });
    } else {
      post({ type: "replace", id, text });
    }
    prevText = text;
  };

  const t0 = Date.now();
  const staticKV = m.allocStaticKV(maxSeq);
  const prevScope = a.setStepScopedCleanup(true);
  try {
    let nextTok: number;
    {
      const idx = a.tensorFromArray(promptIds, [1, promptIds.length]);
      const { logits } = a.noGrad(() =>
        m.forward(idx, { staticKV, residualHook }),
      );
      const top = await a.readTopK(logits, K_PREFILTER, {
        offset: (promptIds.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      nextTok = sampleFromTopK(top.values, top.indices, temperature, topK, topP);
      await a.markStep();
    }
    const prefillMs = Date.now() - t0;

    // Per-token phase accounting: build = lazy graph construction (JS);
    // lower = readTopK's synchronous prefix (plan build/encode/submit — this
    // is where a NON-replayed plan rebuild shows up); fence = awaiting the GPU
    // + the tiny top-K readback; sample = CPU sampling; step = markStep cleanup.
    let tBuild = 0, tLower = 0, tFence = 0, tSample = 0, tStep = 0;
    let count = 0;
    while (count < maxNew && !QWEN3_STOP_TOKENS.has(nextTok)) {
      emit(nextTok);
      count++;
      const b0 = performance.now();
      const idx = a.tensorFromArray([nextTok], [1, 1]);
      const { logits } = a.noGrad(() =>
        m.forward(idx, { staticKV, residualHook }),
      );
      const b1 = performance.now();
      const readP = a.readTopK(logits, K_PREFILTER, { length: vocab }); // sync prefix = plan/encode/submit
      const b2 = performance.now();
      const top = await readP; // await = GPU exec + readback
      const b3 = performance.now();
      logits.dispose();
      nextTok = sampleFromTopK(top.values, top.indices, temperature, topK, topP);
      const b4 = performance.now();
      await a.markStep();
      const b5 = performance.now();
      tBuild += b1 - b0; tLower += b2 - b1; tFence += b3 - b2;
      tSample += b4 - b3; tStep += b5 - b4;
    }
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await a.markStep();

    const seconds = (Date.now() - t0) / 1000;
    const per = (t: number) => Number((t / Math.max(count, 1)).toFixed(1));
    post({
      type: "done",
      id,
      stats: {
        promptTokens: promptIds.length,
        newTokens: count,
        prefillMs,
        seconds: Number(seconds.toFixed(2)),
        tokPerSec: Number(
          (count / Math.max(seconds - prefillMs / 1000, 0.001)).toFixed(1),
        ),
        alpha,
        steered: residualHook !== undefined,
        decodeBreakdown: {
          buildMs: per(tBuild),
          lowerMs: per(tLower),
          fenceMs: per(tFence),
          sampleMs: per(tSample),
          stepMs: per(tStep),
        },
      },
    });
  } finally {
    a.setStepScopedCleanup(prevScope);
  }
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await handleLoad(msg.modelId, msg.weightDtype);
    } else if (msg.type === "computeVector") {
      await handleComputeVector(msg.id, msg.posPrompt, msg.negPrompt, msg.layer);
    } else if (msg.type === "generate") {
      await handleGenerate(
        msg.id,
        msg.prompt,
        msg.alpha ?? 0,
        msg.maxNewTokens ?? 80,
      );
    }
  } catch (err) {
    post({ type: "error", id: msg.id, error: String(err) });
  }
};
