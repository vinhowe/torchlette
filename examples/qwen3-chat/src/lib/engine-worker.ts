/**
 * In-browser inference engine, running inside a Web Worker: WebGPU init,
 * weight streaming, and generation all happen off the main thread — the UI
 * cannot freeze, and engine failures surface as messages instead of dead tabs.
 *
 * Protocol (worker ← main):
 *   { type: "load", modelId, weightDtype? }
 *   { type: "generate", id, messages }
 * (worker → main):
 *   { type: "progress", loaded, total, status }
 *   { type: "loaded", modelId, weightDtype }
 *   { type: "delta", id, delta } | { type: "replace", id, text }
 *   { type: "done", id, stats }
 *   { type: "error", id?, error }
 */

import { AutoTokenizer } from "@huggingface/transformers";
import {
  generateChat,
  loadQwen3FromUrl,
  type ChatMessage,
  type Qwen3,
} from "qwen3-browser";
import {
  getGpuUncapturedErrorCount,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  Torchlette,
} from "torchlette";

let api: Torchlette | null = null;
let model: Qwen3 | null = null;
let tokenizerLike: {
  encode: (t: string) => number[];
  decode: (ids: number[], o?: { skip_special_tokens?: boolean }) => string;
} | null = null;

const post = (msg: Record<string, unknown>) => {
  (self as unknown as Worker).postMessage(msg);
};

function heapMB(): string {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
  return m ? ` · heap ${(m.usedJSHeapSize / 1e9).toFixed(1)}GB` : "";
}

async function handleLoad(modelId: string, requestedDtype?: "f16" | "f32") {
  if (!("gpu" in self.navigator)) {
    throw new Error("WebGPU not available (need Chrome/Edge 121+ or Safari 18+).");
  }
  const adapter = await (self.navigator as unknown as { gpu: { requestAdapter(): Promise<{ features: Set<string> } | null> } }).gpu.requestAdapter();
  const weightDtype: "f16" | "f32" =
    requestedDtype ?? (adapter?.features?.has("shader-f16") ? "f16" : "f32");

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  api = new Torchlette("webgpu", { enableFusion: true });

  // Surface silent GPU death: a lost device otherwise looks like an
  // infinite hang (fences never resolve).
  const device = getWebGPUDevice() as unknown as {
    lost?: Promise<{ reason: string; message: string }>;
    limits?: { maxBufferSize?: number; maxStorageBufferBindingSize?: number };
  } | null;
  device?.lost?.then((info) =>
    post({ type: "error", error: `GPU device lost (${info.reason}): ${info.message}` }),
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

  model = await loadQwen3FromUrl(api, `https://huggingface.co/${modelId}/resolve/main`, {
    maxSeqLen: 2048,
    weightDtype,
    onProgress: (loaded, total, status) => {
      const errs = getGpuUncapturedErrorCount();
      post({
        type: "progress",
        loaded,
        total,
        status: status + heapMB() + (errs > 0 ? ` · ⚠ ${errs} GPU errors (see console)` : ""),
      });
    },
    onTensorEvent: (ev) => post({ type: "tensor", ev }),
  });
  const errs = getGpuUncapturedErrorCount();
  if (errs > 0) {
    throw new Error(`Model loaded but ${errs} GPU validation errors occurred — weights are suspect (see worker console).`);
  }
  post({ type: "loaded", modelId, weightDtype });
}

async function handleGenerate(id: number, messages: ChatMessage[]) {
  if (!api || !model || !tokenizerLike) throw new Error("Model not loaded");
  const stats = await generateChat(api, model, tokenizerLike, messages, {
    onDelta: (delta) => post({ type: "delta", id, delta }),
    onReplace: (text) => post({ type: "replace", id, text }),
  });
  post({ type: "done", id, stats });
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await handleLoad(msg.modelId, msg.weightDtype);
    } else if (msg.type === "generate") {
      await handleGenerate(msg.id, msg.messages);
    }
  } catch (err) {
    post({ type: "error", id: msg.id, error: String(err) });
  }
};
