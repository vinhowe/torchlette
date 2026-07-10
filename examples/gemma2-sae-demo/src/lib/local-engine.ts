/**
 * Main-thread facade over the SAE steering engine worker (engine-worker.ts).
 * All GPU work runs in the worker; this relays messages and exposes a small
 * request/response + streaming API.
 */

import type { TensorLoadEvent } from "gemma2-browser";
import type { FeatureReport, SteerSpec } from "./sae-steering";

export type { TensorLoadEvent, FeatureReport, SteerSpec };

export type GenStats = {
  promptTokens: number;
  newTokens: number;
  prefillMs: number;
  seconds: number;
  tokPerSec: number;
  steered: boolean;
  steer: SteerSpec[];
  tape?: {
    hits: number;
    calls: number;
    traces: number;
    coldMisses: number;
    invalidations: number;
    ready: boolean;
  };
  decodeBreakdown?: {
    buildMs: number;
    lowerMs: number;
    fenceMs: number;
    sampleMs: number;
    stepMs: number;
  };
};

export type GenEvent =
  | { delta: string }
  | { replace: string }
  | { error: string }
  | { done: true; stats: GenStats };

export type ModelInfo = {
  modelId: string;
  weightDtype: "f16" | "f32";
  numLayers: number;
  hiddenSize: number;
  saeLayer: number;
  numFeatures: number;
  neuronpediaSaeId: string;
  tapeFlagSet?: boolean;
  tapeOn?: boolean;
};

export type LoadProgress = (
  loaded: number,
  total: number,
  status: string,
) => void;

export type SAEEngine = {
  info: ModelInfo;
  inspect(prompt: string, topK?: number): Promise<FeatureReport>;
  generate(
    prompt: string,
    steer: SteerSpec[],
    maxNewTokens: number,
    temperature: number,
    onEvent: (e: GenEvent) => void,
  ): Promise<void>;
};

/** The one model we ship: base gemma-2-2b (ungated unsloth mirror). f16 weights
 *  ~4.9GB + f32 embedding 2.36GB + SAE ~0.3GB → comfortable on a 16GB Mac. */
export const MODEL = {
  id: "unsloth/gemma-2-2b",
  label: "Gemma-2-2B",
  approxGB: 8,
};

export async function createSAEEngine(
  modelId: string,
  saeBaseUrl: string,
  onProgress: LoadProgress,
  onTensorEvent?: (ev: TensorLoadEvent) => void,
  onEngineError?: (error: string) => void,
): Promise<SAEEngine> {
  const worker = new Worker(new URL("./engine-worker.ts", import.meta.url), {
    type: "module",
  });

  let nextId = 1;
  const inflight = new Map<number, (msg: Record<string, unknown>) => void>();

  const info = await new Promise<ModelInfo>((resolve, reject) => {
    worker.onerror = (e) =>
      reject(new Error(`Engine worker failed: ${e.message}`));
    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress")
        onProgress(msg.loaded, msg.total, msg.status);
      else if (msg.type === "tensor") onTensorEvent?.(msg.ev);
      else if (msg.type === "loaded")
        resolve({
          modelId: msg.modelId,
          weightDtype: msg.weightDtype,
          numLayers: msg.numLayers,
          hiddenSize: msg.hiddenSize,
          saeLayer: msg.saeLayer,
          numFeatures: msg.numFeatures,
          neuronpediaSaeId: msg.neuronpediaSaeId,
          tapeFlagSet: msg.tapeFlagSet,
          tapeOn: msg.tapeOn,
        });
      else if (msg.type === "error") reject(new Error(msg.error));
    };
    worker.postMessage({ type: "load", modelId, saeBaseUrl });
  });

  worker.onmessage = (e) => {
    const msg = e.data;
    const handler = msg.id !== undefined ? inflight.get(msg.id) : undefined;
    if (!handler) {
      if (msg.type === "error") onEngineError?.(msg.error);
      return;
    }
    handler(msg);
  };
  worker.onerror = (e) => {
    for (const handler of inflight.values())
      handler({ type: "error", error: `Engine worker crashed: ${e.message}` });
    inflight.clear();
  };

  return {
    info,
    inspect(prompt, topK) {
      const id = nextId++;
      return new Promise<FeatureReport>((resolve, reject) => {
        inflight.set(id, (msg) => {
          if (msg.type === "features") {
            inflight.delete(id);
            resolve(msg.report as FeatureReport);
          } else if (msg.type === "error") {
            inflight.delete(id);
            reject(new Error(msg.error as string));
          }
        });
        worker.postMessage({ type: "inspect", id, prompt, topK });
      });
    },
    generate(prompt, steer, maxNewTokens, temperature, onEvent) {
      const id = nextId++;
      return new Promise<void>((resolve) => {
        inflight.set(id, (msg) => {
          if (msg.type === "delta") onEvent({ delta: msg.delta as string });
          else if (msg.type === "replace")
            onEvent({ replace: msg.text as string });
          else if (msg.type === "error") {
            onEvent({ error: msg.error as string });
            inflight.delete(id);
            resolve();
          } else if (msg.type === "done") {
            onEvent({ done: true, stats: msg.stats as GenStats });
            inflight.delete(id);
            resolve();
          }
        });
        worker.postMessage({
          type: "generate",
          id,
          prompt,
          steer,
          maxNewTokens,
          temperature,
        });
      });
    },
  };
}
