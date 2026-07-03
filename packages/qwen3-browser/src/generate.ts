/**
 * Shared chat generation loop (Node server + browser): prefill + KV-cache
 * decode, temperature/top-k/top-p sampling, incremental text emission.
 *
 * The tokenizer is passed as a minimal interface so both @huggingface/
 * transformers (Node and browser) satisfy it.
 */

import type { Torchlette } from "torchlette";
import type { Qwen3 } from "./model";

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

export const QWEN3_STOP_TOKENS = new Set([
  151645 /* <|im_end|> */, 151643 /* <|endoftext|> */,
]);

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
 * Temperature + top-k + top-p sampling over a top-K prefilter result
 * (api.readTopK: values sorted desc, ties by index asc — the same ordering
 * sampleToken's partial select produces). Avoids reading the full vocab
 * logits row to the CPU.
 */
export function sampleFromTopK(
  values: Float32Array,
  indices: Int32Array,
  temperature = 0.7,
  topK = 20,
  topP = 0.95,
): number {
  const k = Math.min(topK, values.length);
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
    throw new Error(
      `Conversation too long (${promptIds.length} tokens, max ~${maxSeq})`,
    );
  }
  const vocab = model.config.vocabSize;
  const maxNew = Math.min(
    options?.maxNewTokens ?? 400,
    maxSeq - promptIds.length - 1,
  );
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
  // Static KV cache: preallocated per-layer buffers updated in place —
  // decode steps within a length bucket share one plan template, which the
  // recurring-plan replay machinery accelerates automatically.
  const staticKV = model.allocStaticKV(maxSeq);
  // Ceremony-free step-scoped cleanup: each markStep reclaims the interval's
  // graph temporaries deterministically (without it, decode leaks ~1 storage
  // handle per graph node per token until V8 GC collects the wrappers and
  // the markStep sweep cost grows unboundedly). Safe here: static-KV decode
  // holds NO tensors created inside an interval across markStep (cache slots
  // are updated in place via copy_; logits are read before markStep).
  // Restored on exit so surrounding app code (e.g. steering/analysis holding
  // tensors across steps) keeps default markStep semantics.
  const prevStepScope = api.setStepScopedCleanup(true);
  try {
    // Top-K prefilter readback: K=64 covers any sensible sampling topK and the
    // GPU kernel reduces the per-token readback from the full vocab row (600KB)
    // to 64 (value, index) pairs (512B). Greedy = indices[0], bit-identical to
    // a full-logits argmax (gated by examples/qwen3/topk-equivalence.ts).
    const K_PREFILTER = Math.max(64, topK ?? 20);
    let nextTok: number;
    {
      const idx = api.tensorFromArray(promptIds, [1, promptIds.length]);
      const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
      const top = await api.readTopK(logits, K_PREFILTER, {
        offset: (promptIds.length - 1) * vocab,
        length: vocab,
      });
      logits.dispose();
      nextTok = sampleFromTopK(
        top.values,
        top.indices,
        temperature,
        topK,
        topP,
      );
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
      const t0 = performance.now();
      const idx = api.tensorFromArray([nextTok], [1, 1]);
      const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
      const t1 = performance.now();
      // readTopK's synchronous prefix is the plan/lower/encode JS; the await is
      // the GPU fence + the 512B top-K readback.
      const readback = api.readTopK(logits, K_PREFILTER, { length: vocab });
      const t2 = performance.now();
      const top = await readback;
      const t3 = performance.now();
      logits.dispose();
      nextTok = sampleFromTopK(
        top.values,
        top.indices,
        temperature,
        topK,
        topP,
      );
      const t4 = performance.now();
      api.endStep();
      await api.markStep();
      const t5 = performance.now();
      tBuild += t1 - t0;
      tLower += t2 - t1;
      tFence += t3 - t2;
      tSample += t4 - t3;
      tStep += t5 - t4;
    }
    // Drop KV cache refs, then flush so the buffers get reclaimed.
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await api.markStep();

    const seconds = (Date.now() - t0) / 1000;
    const per = (t: number) => Number((t / Math.max(count, 1)).toFixed(1));
    return {
      promptTokens: promptIds.length,
      newTokens: count,
      seconds: Number(seconds.toFixed(2)),
      prefillMs,
      tokPerSec: Number(
        (count / Math.max(seconds - prefillMs / 1000, 0.001)).toFixed(1),
      ),
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
