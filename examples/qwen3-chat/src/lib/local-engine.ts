/**
 * Main-thread facade over the engine worker (see engine-worker.ts). All GPU
 * and conversion work runs in the worker; this just relays messages.
 */

import type { ChatMessage, GenerateStats } from "qwen3-browser";

export type EngineEvent =
  | { delta: string }
  | { replace: string }
  | { error: string }
  | { done: true; stats: GenerateStats };

export type LoadProgress = (loaded: number, total: number, status: string) => void;

export type LocalEngine = {
  modelId: string;
  generate(messages: ChatMessage[], onEvent: (e: EngineEvent) => void): Promise<void>;
};

export const LOCAL_MODELS = [
  { id: "Qwen/Qwen3-0.6B", label: "0.6B", approxGB: 1.6 },
  { id: "Qwen/Qwen3-1.7B", label: "1.7B", approxGB: 4.1 },
];

export async function createLocalEngine(
  modelId: string,
  onProgress: LoadProgress,
): Promise<LocalEngine> {
  const worker = new Worker(new URL("./engine-worker.ts", import.meta.url), {
    type: "module",
  });

  let nextId = 1;
  const inflight = new Map<number, (e: EngineEvent) => void>();

  await new Promise<void>((resolve, reject) => {
    worker.onerror = (e) => reject(new Error(`Engine worker failed: ${e.message}`));
    worker.onmessage = (e) => {
      const msg = e.data;
      if (msg.type === "progress") onProgress(msg.loaded, msg.total, msg.status);
      else if (msg.type === "loaded") resolve();
      else if (msg.type === "error") reject(new Error(msg.error));
    };
    worker.postMessage({ type: "load", modelId });
  });

  // Steady-state message routing (post-load).
  worker.onmessage = (e) => {
    const msg = e.data;
    const handler = msg.id !== undefined ? inflight.get(msg.id) : undefined;
    if (!handler) return;
    if (msg.type === "delta") handler({ delta: msg.delta });
    else if (msg.type === "replace") handler({ replace: msg.text });
    else if (msg.type === "error") {
      handler({ error: msg.error });
      inflight.delete(msg.id);
    } else if (msg.type === "done") {
      handler({ done: true, stats: msg.stats });
      inflight.delete(msg.id);
    }
  };
  worker.onerror = (e) => {
    for (const handler of inflight.values()) {
      handler({ error: `Engine worker crashed: ${e.message}` });
    }
    inflight.clear();
  };

  return {
    modelId,
    async generate(messages, onEvent) {
      const id = nextId++;
      await new Promise<void>((resolve) => {
        inflight.set(id, (event) => {
          onEvent(event);
          if ("done" in event || "error" in event) resolve();
        });
        worker.postMessage({ type: "generate", id, messages });
      });
    },
  };
}
