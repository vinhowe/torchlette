/**
 * Main-thread facade over the steering engine worker (see engine-worker.ts).
 * All GPU work runs in the worker; this relays messages and exposes a small
 * request/response + streaming API.
 */

import type { TensorLoadEvent } from "qwen3-browser";

export type { TensorLoadEvent };

export type GenStats = {
  promptTokens: number;
  newTokens: number;
  prefillMs: number;
  seconds: number;
  tokPerSec: number;
  alpha: number;
  steered: boolean;
  /** Step-tape replay counters for this generation (§6 observability). */
  tape?: {
    hits: number; calls: number; traces: number; coldMisses: number;
    invalidations: number; ready: boolean;
    recorder?: { eligiblePairs: number; refusals: number; structureMisses: number; loweredPairs: number; boundaryResets: number; boundaryReasons?: Record<string, number>; lastRefusal: string };
  };
  /** Per-token decode phase averages (ms): build=lazy graph, lower=plan/
   *  encode/submit, fence=GPU+readback, sample=CPU, step=markStep. */
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

export type VectorInfo = {
  layer: number;
  hiddenSize: number;
  posPrompt: string;
  negPrompt: string;
};

export type ModelInfo = {
  modelId: string;
  weightDtype: "f16" | "f32";
  numLayers: number;
  hiddenSize: number;
  /** Step-tape ground truth from the worker (diagnostics). */
  tapeFlagSet?: boolean;
  tapeOn?: boolean;
};

export type LoadProgress = (
  loaded: number,
  total: number,
  status: string,
) => void;

export type SteeringEngine = {
  info: ModelInfo;
  computeVector(
    posPrompt: string,
    negPrompt: string,
    layer: number,
  ): Promise<VectorInfo>;
  generate(
    prompt: string,
    alpha: number,
    maxNewTokens: number,
    onEvent: (e: GenEvent) => void,
  ): Promise<void>;
};

export const LOCAL_MODELS = [
  { id: "Qwen/Qwen3-0.6B", label: "0.6B", approxGB: 1.6 },
  { id: "Qwen/Qwen3-1.7B", label: "1.7B", approxGB: 4.1 },
  // ~9.5-10GB in memory (f16 weights ~8GB + f32 embedding ~1.5GB + KV/pool):
  // wants a ≥24-32GB unified-memory Mac; may OOM Chrome's GPU budget on 16GB,
  // and likely too big for the IndexedDB cache (re-downloads each session).
  // Golden Gate layer/alpha are tuned for 1.7B — 4B (~36 layers) needs a re-sweep.
  { id: "Qwen/Qwen3-4B", label: "4B", approxGB: 10 },
];

export async function createSteeringEngine(
  modelId: string,
  onProgress: LoadProgress,
  onTensorEvent?: (ev: TensorLoadEvent) => void,
  onEngineError?: (error: string) => void,
): Promise<SteeringEngine> {
  const worker = new Worker(new URL("./engine-worker.ts", import.meta.url), {
    type: "module",
  });

  let nextId = 1;
  // id → per-request handler (generation streams; compute resolves once).
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
          tapeFlagSet: msg.tapeFlagSet,
          tapeOn: msg.tapeOn,
        });
      else if (msg.type === "error") reject(new Error(msg.error));
    };
    worker.postMessage({ type: "load", modelId });
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
    computeVector(posPrompt, negPrompt, layer) {
      const id = nextId++;
      return new Promise<VectorInfo>((resolve, reject) => {
        inflight.set(id, (msg) => {
          if (msg.type === "vector") {
            inflight.delete(id);
            resolve({
              layer: msg.layer as number,
              hiddenSize: msg.hiddenSize as number,
              posPrompt: msg.posPrompt as string,
              negPrompt: msg.negPrompt as string,
            });
          } else if (msg.type === "error") {
            inflight.delete(id);
            reject(new Error(msg.error as string));
          }
        });
        worker.postMessage({ type: "computeVector", id, posPrompt, negPrompt, layer });
      });
    },
    generate(prompt, alpha, maxNewTokens, onEvent) {
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
        worker.postMessage({ type: "generate", id, prompt, alpha, maxNewTokens });
      });
    },
  };
}
