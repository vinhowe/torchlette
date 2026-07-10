/**
 * In-browser SAE steering engine (Web Worker): WebGPU init, Gemma-2-2B weight
 * streaming, Gemma Scope SAE loading, feature inspection, and steered generation
 * — all off the main thread.
 *
 * Protocol (worker ← main):
 *   { type: "load", modelId, saeBaseUrl, weightDtype? }
 *   { type: "inspect", id, prompt, topK? }
 *   { type: "generate", id, prompt, steer, maxNewTokens?, temperature? }
 *       steer: { feature, alpha }[]
 * (worker → main):
 *   { type: "progress", loaded, total, status }
 *   { type: "loaded", modelId, weightDtype, numLayers, hiddenSize, saeLayer,
 *            numFeatures, neuronpediaSaeId, tapeFlagSet, tapeOn }
 *   { type: "tensor", ev }
 *   { type: "features", id, report }            (FeatureReport)
 *   { type: "delta", id, delta } | { type: "replace", id, text }
 *   { type: "done", id, stats }
 *   { type: "error", id?, error }
 */

import "./tape-flag"; // MUST be first: sets TORCHLETTE_STEP_TAPE before torchlette evaluates
import { AutoTokenizer } from "@huggingface/transformers";
import { STEP_TAPE_REPLAY } from "torchlette";
import {
  generateChat,
  loadGemma2FromUrl,
  type Gemma2,
} from "gemma2-browser";
import { GemmaScopeSAE, type SAEConfig, type SAEParams } from "gemma-scope-sae";
import {
  getGpuUncapturedErrorCount,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  Torchlette,
} from "torchlette";
import {
  inspectFeatures,
  makeSAEResidualHook,
  type SteerSpec,
} from "./sae-steering";

let api: Torchlette | null = null;
let model: Gemma2 | null = null;
let sae: GemmaScopeSAE | null = null;
let tokenizerLike:
  | {
      encode: (t: string) => number[];
      decode: (ids: number[], o?: { skip_special_tokens?: boolean }) => string;
    }
  | null = null;

const post = (msg: Record<string, unknown>) => {
  (self as unknown as Worker).postMessage(msg);
};

function heapMB(): string {
  const m = (performance as unknown as { memory?: { usedJSHeapSize: number } })
    .memory;
  return m ? ` · heap ${(m.usedJSHeapSize / 1e9).toFixed(1)}GB` : "";
}

/** Fetch the SAE .bin files from a static base URL into SAEParams. */
async function loadSAE(
  baseUrl: string,
  onStatus: (s: string) => void,
): Promise<GemmaScopeSAE> {
  onStatus("Loading SAE manifest…");
  const manifest = await (await fetch(`${baseUrl}/sae.json`)).json();
  const F = manifest.numFeatures as number;
  const d = manifest.dModel as number;
  const rd = async (name: string, expect: number): Promise<Float32Array> => {
    onStatus(`Loading SAE ${name}…`);
    const buf = await (
      await fetch(`${baseUrl}/${manifest.files[name]}`)
    ).arrayBuffer();
    const a = new Float32Array(buf);
    if (a.length !== expect)
      throw new Error(`SAE ${name}: ${a.length} floats, expected ${expect}`);
    return a;
  };
  const params: SAEParams = {
    W_enc: await rd("W_enc", d * F),
    b_enc: await rd("b_enc", F),
    W_dec: await rd("W_dec", F * d),
    b_dec: await rd("b_dec", d),
    threshold: await rd("threshold", F),
  };
  const config: SAEConfig = {
    dModel: d,
    numFeatures: F,
    layer: manifest.layer,
    neuronpediaSaeId: manifest.neuronpediaSaeId,
  };
  if (!api) throw new Error("api not ready");
  return GemmaScopeSAE.load(api, config, params, { dtype: "f32" });
}

async function handleLoad(
  modelId: string,
  saeBaseUrl: string,
  requestedDtype?: "f16" | "f32",
) {
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

  model = await loadGemma2FromUrl(
    api,
    `https://huggingface.co/${modelId}/resolve/main`,
    {
      maxSeqLen: 1024,
      weightDtype,
      onProgress: (loaded, total, status) => {
        const errs = getGpuUncapturedErrorCount();
        post({
          type: "progress",
          loaded,
          total,
          status:
            status + heapMB() + (errs > 0 ? ` · ⚠ ${errs} GPU errors` : ""),
        });
      },
      onTensorEvent: (ev) => post({ type: "tensor", ev }),
    },
  );

  sae = await loadSAE(saeBaseUrl, (status) =>
    post({ type: "progress", loaded: 0, total: 0, status: status + heapMB() }),
  );

  const errs = getGpuUncapturedErrorCount();
  if (errs > 0) {
    throw new Error(
      `Model+SAE loaded but ${errs} GPU validation errors occurred (see worker console).`,
    );
  }
  post({
    type: "loaded",
    modelId,
    weightDtype,
    numLayers: model.config.numLayers,
    hiddenSize: model.config.hiddenSize,
    saeLayer: sae.config.layer,
    numFeatures: sae.config.numFeatures,
    neuronpediaSaeId: sae.config.neuronpediaSaeId,
    tapeFlagSet: !!(globalThis as { __TORCHLETTE_ENV__?: unknown })
      .__TORCHLETTE_ENV__,
    tapeOn: STEP_TAPE_REPLAY,
  });
}

async function handleInspect(id: number, prompt: string, topK?: number) {
  if (!api || !model || !sae || !tokenizerLike) throw new Error("Not loaded");
  const report = await inspectFeatures(
    api,
    model,
    sae,
    tokenizerLike,
    prompt,
    topK ?? 20,
  );
  post({ type: "features", id, report });
}

async function handleGenerate(
  id: number,
  prompt: string,
  steer: SteerSpec[],
  maxNewTokens: number,
  temperature: number,
) {
  if (!api || !model || !sae || !tokenizerLike) throw new Error("Not loaded");
  const residualHook = makeSAEResidualHook(api, sae, steer);
  const stats = await generateChat(
    api,
    model,
    tokenizerLike,
    [{ role: "user", content: prompt }],
    {
      onDelta: (delta) => post({ type: "delta", id, delta }),
      onReplace: (text) => post({ type: "replace", id, text }),
    },
    {
      maxNewTokens,
      temperature,
      topK: 40,
      topP: 0.95,
      chat: false,
      residualHook,
    },
  );
  post({
    type: "done",
    id,
    stats: { ...stats, steered: residualHook !== undefined, steer },
  });
}

self.onmessage = async (e: MessageEvent) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await handleLoad(msg.modelId, msg.saeBaseUrl, msg.weightDtype);
    } else if (msg.type === "inspect") {
      await handleInspect(msg.id, msg.prompt, msg.topK);
    } else if (msg.type === "generate") {
      await handleGenerate(
        msg.id,
        msg.prompt,
        msg.steer ?? [],
        msg.maxNewTokens ?? 80,
        msg.temperature ?? 0.7,
      );
    }
  } catch (err) {
    post({ type: "error", id: msg.id, error: String(err) });
  }
};
