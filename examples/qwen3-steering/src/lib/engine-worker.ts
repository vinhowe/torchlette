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

import "./tape-flag"; // MUST be first: sets TORCHLETTE_STEP_TAPE before torchlette evaluates
import { AutoTokenizer } from "@huggingface/transformers";
import { stReplayStats } from "torchlette";
import {
  generateChat,
  loadQwen3FromUrl,
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
 * Generation = ONE generateChat call (packages/qwen3-browser). The worker
 * previously carried a hand-rolled copy of the decode loop; that copy (a)
 * never went through capture()/the step-tape, and (b) still contained the
 * #79 arming-baseline ratchet leak that 0000ee15 fixed in generateChat only.
 * Delegating inherits: the chat template, capture()-taped decode (with
 * ./tape-flag setting TORCHLETTE_STEP_TAPE=1 before module eval), the
 * per-token phase breakdown, and the leak-free generation lifecycle.
 * The steering hook is frozen-by-closure per call — sound, since each
 * generateChat call captures its own fn (one generation lifetime).
 */
async function handleGenerate(
  id: number,
  prompt: string,
  alpha: number,
  maxNewTokens: number,
) {
  if (!api || !model || !tokenizerLike) throw new Error("Model not loaded");
  const residualHook = makeResidualHook(api, steeringVec, alpha);
  const tape0 = stReplayStats();
  const stats = await generateChat(
    api,
    model,
    tokenizerLike,
    [{ role: "user", content: prompt }],
    {
      onDelta: (delta) => post({ type: "delta", id, delta }),
      onReplace: (text) => post({ type: "replace", id, text }),
    },
    { maxNewTokens, temperature: 0.7, topK: 20, topP: 0.95, residualHook },
  );
  const t = stReplayStats();
  post({
    type: "done",
    id,
    stats: {
      ...stats,
      alpha,
      steered: residualHook !== undefined,
      tape: { hits: t.hits - tape0.hits, replays: t.replays - tape0.replays },
    },
  });
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
