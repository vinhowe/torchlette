/**
 * Shared generation loop (Node + browser) for Gemma-2: prefill + static-KV
 * decode, temperature/top-k/top-p sampling, incremental text emission. Ported
 * from packages/qwen3-browser/src/generate.ts.
 *
 * Two prompt modes, because we run the BASE gemma-2-2b (not the -it chat model):
 *  - raw completion (DEFAULT): <bos> + prompt, continue. Base models follow
 *    plain continuations; the chat template makes them drift.
 *  - chat: the Gemma-2 turn template (kept for completeness / -it checkpoints).
 *
 * The residualHook option is the SAE steering seam — threaded verbatim into the
 * prefill forward AND every taped decode step.
 */

import type { FrontendTensor as Tensor, Torchlette } from "torchlette";
import { stStats } from "torchlette";
import type { Gemma2, ResidualHook } from "./model";
import { kvBucketLen } from "./model";

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type TokenizerLike = {
  encode(text: string): number[];
  decode(ids: number[], options?: { skip_special_tokens?: boolean }): string;
};

export type GenerateEvents = {
  /** Incremental text (append). */
  onDelta?: (delta: string) => void;
  /** Rare BPE re-merge: replace the full generated text. */
  onReplace?: (text: string) => void;
};

export type GenerateOptions = {
  temperature?: number;
  topK?: number;
  topP?: number;
  maxNewTokens?: number;
  isAborted?: () => boolean;
  /** true → apply the Gemma chat turn template; false/undefined → raw
   *  completion (base-model mode). */
  chat?: boolean;
  /**
   * Residual-stream steering hook, forwarded verbatim to model.forward on BOTH
   * the prefill and every KV-decode step. Undefined → unsteered baseline.
   */
  residualHook?: ResidualHook;
};

export type GenerateStats = {
  promptTokens: number;
  newTokens: number;
  seconds: number;
  prefillMs: number;
  tokPerSec: number;
  decodeBreakdown?: {
    buildMs: number;
    lowerMs: number;
    fenceMs: number;
    sampleMs: number;
    stepMs: number;
  };
  tape?: {
    hits: number;
    calls: number;
    traces: number;
    coldMisses: number;
    invalidations: number;
    ready: boolean;
  };
};

/** Gemma-2 stop tokens: <eos>=1, <end_of_turn>=107. */
export const GEMMA2_STOP_TOKENS = new Set([1, 107]);
/** <bos> is id 2 in the Gemma tokenizer. */
const BOS = 2;

/** Gemma-2 chat turn template (for -it checkpoints). Base uses raw mode. */
export function buildChatPrompt(messages: ChatMessage[]): string {
  let s = "";
  for (const m of messages) {
    const role = m.role === "assistant" ? "model" : "user";
    s += `<start_of_turn>${role}\n${m.content}<end_of_turn>\n`;
  }
  s += "<start_of_turn>model\n";
  return s;
}

/**
 * Encode the prompt to ids. `chat` uses the turn template; raw mode passes the
 * text straight through the tokenizer. A leading <bos> is ensured either way
 * (Gemma requires it) — tokenizers usually add it, but we guard against a
 * tokenizer configured with add_bos_token=false.
 */
function encodePrompt(
  tokenizer: TokenizerLike,
  messages: ChatMessage[],
  chat: boolean,
): number[] {
  const text = chat
    ? buildChatPrompt(messages)
    : messages.map((m) => m.content).join("");
  const ids = tokenizer.encode(text);
  if (ids[0] !== BOS) ids.unshift(BOS);
  return ids;
}

/**
 * Temperature + top-k + top-p sampling over a top-K prefilter result
 * (api.readTopK: values sorted desc). Avoids reading the full vocab row.
 */
export function sampleFromTopK(
  values: Float32Array,
  indices: Int32Array,
  temperature = 0.7,
  topK = 20,
  topP = 0.95,
): number {
  const k = Math.min(topK, values.length);
  if (temperature <= 0) return indices[0];
  const mx = values[0];
  const exps = new Float64Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    exps[i] = Math.exp((values[i] - mx) / temperature);
    sum += exps[i];
  }
  let cut = k;
  let cum = 0;
  for (let i = 0; i < k; i++) {
    cum += exps[i] / sum;
    if (cum >= topP) {
      cut = i + 1;
      break;
    }
  }
  let cutSum = 0;
  for (let i = 0; i < cut; i++) cutSum += exps[i];
  let r = Math.random() * cutSum;
  for (let i = 0; i < cut; i++) {
    r -= exps[i];
    if (r <= 0) return indices[i];
  }
  return indices[0];
}

/**
 * Generate a completion. Emits incremental text via `events`, returns final
 * stats. Prefills the full prompt, then decodes with the static KV cache. The
 * residualHook (SAE steering) is threaded into prefill + every decode step.
 */
export async function generateChat(
  api: Torchlette,
  model: Gemma2,
  tokenizer: TokenizerLike,
  messages: ChatMessage[],
  events: GenerateEvents,
  options?: GenerateOptions,
): Promise<GenerateStats> {
  const maxSeq = model.config.maxSeqLen;
  const promptIds = encodePrompt(tokenizer, messages, options?.chat ?? false);
  if (promptIds.length + 8 >= maxSeq) {
    throw new Error(
      `Prompt too long (${promptIds.length} tokens, max ~${maxSeq})`,
    );
  }
  const vocab = model.config.vocabSize;
  const maxNew = Math.min(
    options?.maxNewTokens ?? 200,
    maxSeq - promptIds.length - 1,
  );
  const isAborted = options?.isAborted ?? (() => false);
  const { temperature, topK, topP, residualHook } = options ?? {};

  const genIds: number[] = [];
  let prevText = "";
  const emit = (tok: number) => {
    genIds.push(tok);
    const text = tokenizer.decode(genIds, { skip_special_tokens: true });
    if (text.startsWith(prevText)) {
      const delta = text.slice(prevText.length);
      if (delta) events.onDelta?.(delta);
    } else {
      events.onReplace?.(text);
    }
    prevText = text;
  };

  const t0 = Date.now();
  const staticKV = model.allocStaticKV(maxSeq);
  const prevStepScope = api.setStepScopedCleanup(true);
  try {
    // Baseline boundary before prefill (see qwen3 generate.ts rationale).
    await api.markStep();
    const K_PREFILTER = Math.max(64, topK ?? 20);
    let nextTok: number;
    {
      const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
      const { logits } = api.noGrad(() =>
        model.forward(idx, { staticKV, residualHook }),
      );
      const top = await api.readTopK(logits, K_PREFILTER, {
        offset: (promptIds.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      nextTok = sampleFromTopK(top.values, top.indices, temperature, topK, topP);
      await api.markStep();
    }
    const prefillMs = Date.now() - t0;

    let count = 0;
    let tBuild = 0;
    let tLower = 0;
    let tFence = 0;
    let tSample = 0;
    let tStep = 0;
    // Taped decode: the residualHook is closure-captured and FROZEN for this
    // CapturedFn's lifetime — sound by construction (one generateChat call =
    // one hook/α = one fresh trace). The bucket length is the structural
    // discriminator; attnModKey guards cross-model tape replay.
    const decode = api.capture(
      (idx: Tensor) =>
        api.noGrad(() => model.forward(idx, { staticKV, residualHook }).logits),
      {
        key: () =>
          `kv:bkt${kvBucketLen(staticKV.len + 1, maxSeq)}:mod${model.attnModKey}`,
      },
    );
    while (count < maxNew && !GEMMA2_STOP_TOKENS.has(nextTok) && !isAborted()) {
      emit(nextTok);
      count++;
      const tb0 = performance.now();
      const logits = (await decode(
        api.tensorFromArray([nextTok], [1, 1]),
      )) as Tensor;
      const tb1 = performance.now();
      const readback = api.readTopK(logits, K_PREFILTER, { length: vocab });
      const tb2 = performance.now();
      const top = await readback;
      const tb3 = performance.now();
      logits.dispose();
      nextTok = sampleFromTopK(top.values, top.indices, temperature, topK, topP);
      const tb4 = performance.now();
      await api.markStep();
      const tb5 = performance.now();
      tBuild += tb1 - tb0;
      tLower += tb2 - tb1;
      tFence += tb3 - tb2;
      tSample += tb4 - tb3;
      tStep += tb5 - tb4;
    }
    // Deterministic KV lifetime end (see qwen3 generate.ts).
    for (const t of staticKV.k) t.dispose();
    for (const t of staticKV.v) t.dispose();
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await api.markStep();

    const seconds = (Date.now() - t0) / 1000;
    const per = (t: number) => Number((t / Math.max(count, 1)).toFixed(1));
    const c = decode.stats();
    void stStats();
    return {
      promptTokens: promptIds.length,
      newTokens: count,
      seconds: Number(seconds.toFixed(2)),
      prefillMs,
      tokPerSec: Number(
        (count / Math.max(seconds - prefillMs / 1000, 0.001)).toFixed(1),
      ),
      tape: {
        hits: c.hits,
        calls: c.calls,
        traces: c.traces,
        coldMisses: c.coldMisses,
        invalidations: c.invalidations,
        ready: c.ready,
      },
      decodeBreakdown: {
        buildMs: per(tBuild),
        lowerMs: per(tLower),
        fenceMs: per(tFence),
        sampleMs: per(tSample),
        stepMs: per(tStep),
      },
    };
  } finally {
    api.setStepScopedCleanup(prevStepScope);
  }
}
