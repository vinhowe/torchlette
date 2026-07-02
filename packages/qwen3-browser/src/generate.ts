/**
 * Shared chat generation loop (Node server + browser): prefill + KV-cache
 * decode, temperature/top-k/top-p sampling, incremental text emission.
 *
 * The tokenizer is passed as a minimal interface so both @huggingface/
 * transformers (Node and browser) satisfy it.
 */

import type { Torchlette } from "torchlette";
import type { KVCache, Qwen3 } from "./model";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

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
};

export type GenerateStats = {
  promptTokens: number;
  newTokens: number;
  seconds: number;
  prefillMs: number;
  tokPerSec: number;
  /**
   * Per-token decode averages (ms): where each token's wall time goes.
   * build = lazy graph construction (JS); lower = plan/lower/encode — the
   * synchronous prefix of cpu() (JS); fence = awaiting the GPU + readback;
   * sample = CPU sampling; step = markStep cleanup + fence.
   */
  decodeBreakdown?: {
    buildMs: number;
    lowerMs: number;
    fenceMs: number;
    sampleMs: number;
    stepMs: number;
  };
};

export const QWEN3_STOP_TOKENS = new Set([151645 /* <|im_end|> */, 151643 /* <|endoftext|> */]);

/** Qwen3 chat format, thinking disabled (empty think block, per the official template). */
export function buildChatPrompt(messages: ChatMessage[]): string {
  let s = "";
  for (const m of messages) {
    s += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  }
  s += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  return s;
}

/** Temperature + top-k + top-p sampling over one position's logits. */
export function sampleToken(
  logits: Float32Array,
  offset: number,
  vocab: number,
  temperature = 0.7,
  topK = 20,
  topP = 0.95,
): number {
  const best: { v: number; l: number }[] = [];
  for (let v = 0; v < vocab; v++) {
    const l = logits[offset + v];
    if (best.length < topK) {
      best.push({ v, l });
      if (best.length === topK) best.sort((a, b) => b.l - a.l);
    } else if (l > best[topK - 1].l) {
      best[topK - 1] = { v, l };
      for (let i = topK - 1; i > 0 && best[i - 1].l < best[i].l; i--) {
        [best[i - 1], best[i]] = [best[i], best[i - 1]];
      }
    }
  }
  if (best.length < topK) best.sort((a, b) => b.l - a.l);

  const mx = best[0].l;
  const exps = best.map((b) => Math.exp((b.l - mx) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);

  let cut = exps.length;
  let cum = 0;
  for (let i = 0; i < exps.length; i++) {
    cum += exps[i] / sum;
    if (cum >= topP) {
      cut = i + 1;
      break;
    }
  }
  const cutSum = exps.slice(0, cut).reduce((a, b) => a + b, 0);
  let r = Math.random() * cutSum;
  for (let i = 0; i < cut; i++) {
    r -= exps[i];
    if (r <= 0) return best[i].v;
  }
  return best[0].v;
}

/**
 * Generate a chat completion. Emits incremental text via `events`, returns
 * final stats. Prefills the full prompt, then decodes with the KV cache.
 */
export async function generateChat(
  api: Torchlette,
  model: Qwen3,
  tokenizer: TokenizerLike,
  messages: ChatMessage[],
  events: GenerateEvents,
  options?: GenerateOptions,
): Promise<GenerateStats> {
  const maxSeq = model.config.maxSeqLen;
  const prompt = buildChatPrompt(messages);
  const promptIds = tokenizer.encode(prompt);
  if (promptIds.length + 8 >= maxSeq) {
    throw new Error(`Conversation too long (${promptIds.length} tokens, max ~${maxSeq})`);
  }
  const vocab = model.config.vocabSize;
  const maxNew = Math.min(options?.maxNewTokens ?? 400, maxSeq - promptIds.length - 1);
  const isAborted = options?.isAborted ?? (() => false);
  const { temperature, topK, topP } = options ?? {};

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
  let kv: KVCache[];
  let nextTok: number;
  {
    const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
    const { logits, presentKVs } = api.noGrad(() => model.forward(idx));
    kv = presentKVs;
    const flat = new Float32Array(await logits.cpu());
    logits.dispose();
    nextTok = sampleToken(flat, (promptIds.length - 1) * vocab, vocab, temperature, topK, topP);
    await api.markStep();
  }
  const prefillMs = Date.now() - t0;

  let count = 0;
  let tBuild = 0;
  let tLower = 0;
  let tFence = 0;
  let tSample = 0;
  let tStep = 0;
  while (count < maxNew && !QWEN3_STOP_TOKENS.has(nextTok) && !isAborted()) {
    emit(nextTok);
    count++;
    const posOffset = promptIds.length + count - 1;
    const t0 = performance.now();
    const idx = api.tensorFromArray([nextTok], [1, 1]);
    const { logits, presentKVs } = api.noGrad(() =>
      model.forward(idx, { pastKVs: kv, posOffset }),
    );
    kv = presentKVs;
    const t1 = performance.now();
    // cpu()'s synchronous prefix is the plan/lower/encode JS; the await is
    // the GPU fence + readback.
    const readback = logits.cpu();
    const t2 = performance.now();
    const flat = new Float32Array(await readback);
    const t3 = performance.now();
    logits.dispose();
    nextTok = sampleToken(flat, 0, vocab, temperature, topK, topP);
    const t4 = performance.now();
    await api.markStep();
    const t5 = performance.now();
    tBuild += t1 - t0;
    tLower += t2 - t1;
    tFence += t3 - t2;
    tSample += t4 - t3;
    tStep += t5 - t4;
  }
  // Drop KV refs, then flush so the cache buffers get reclaimed.
  kv = [];
  await api.markStep();

  const seconds = (Date.now() - t0) / 1000;
  const per = (t: number) => Number((t / Math.max(count, 1)).toFixed(1));
  return {
    promptTokens: promptIds.length,
    newTokens: count,
    seconds: Number(seconds.toFixed(2)),
    prefillMs,
    tokPerSec: Number((count / Math.max(seconds - prefillMs / 1000, 0.001)).toFixed(1)),
    decodeBreakdown: {
      buildMs: per(tBuild),
      lowerMs: per(tLower),
      fenceMs: per(tFence),
      sampleMs: per(tSample),
      stepMs: per(tStep),
    },
  };
}
