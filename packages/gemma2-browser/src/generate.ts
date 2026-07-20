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
import type { Gemma2, ResidualHook, StaticKV } from "./model";
import { KV_BUCKET, kvBucketLen } from "./model";

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
  /** Seed for the block's on-device Gumbel-max sampler (§3.5, pure-temperature
   *  path). Fixed per generation ⇒ byte-reproducible. Undefined ⇒ fresh random
   *  seed. Only consulted when the block Gumbel path is active. */
  seed?: number;
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

// ============================================================================
// Unrolled-K decode block (CRYSTAL-1 P4 cutover) — mechanical port of the
// qwen3-browser static-KV surface (design P1 status: "gemma2 wiring is a
// mechanical port, deferred to P4 cutover"). decodeBlock is generic over the
// StaticDecodeModel interface; Gemma2 satisfies it. See packages/qwen3-browser/
// src/generate.ts for the full contract + the flag's sunset. Kept in-package
// (not shared into src/): packages/ is not the src diamond, and gemma2-browser
// has no qwen3-browser dependency.
// ============================================================================

/** DEFAULT-ON (opt-out via TORCHLETTE_UNROLLED_K=0/1). K sweet spot {4,8}; K=4
 *  default (best compiled win + finest streaming; K=16 regresses). */
export const UNROLLED_K_DEFAULT = 4;
function unrolledKRaw(): string | undefined {
  const env: Record<string, string | undefined> =
    typeof process !== "undefined" && process.env
      ? (process.env as Record<string, string | undefined>)
      : {};
  const g = globalThis as { __TORCHLETTE_ENV__?: Record<string, string> };
  const raw =
    g.__TORCHLETTE_ENV__?.TORCHLETTE_UNROLLED_K ?? env.TORCHLETTE_UNROLLED_K;
  return raw === undefined || raw === "" ? undefined : raw;
}
export function unrolledKFromEnv(): number {
  const raw = unrolledKRaw();
  if (raw === undefined) return UNROLLED_K_DEFAULT;
  const k = Number(raw);
  return Number.isFinite(k) && k >= 2 ? Math.floor(k) : 0;
}
/** Clip K so all K forwards share one static-KV bucket template (§3.4). */
export function clipBlockToBucket(
  len: number,
  K: number,
  maxSeqLen: number,
): number {
  const nextBoundary = Math.ceil((len + 1) / KV_BUCKET) * KV_BUCKET;
  const roomInBucket = nextBoundary - len;
  const roomInSeq = maxSeqLen - len;
  return Math.max(1, Math.min(K, roomInBucket, roomInSeq));
}

/** Deterministic per-position uniform noise (seed-as-DATA, §3.5) — mulberry32,
 *  the single source both decodeBlock and any parity reference draw from. */
export function gumbelUniform(seed: number, n: number): Float32Array {
  let a = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const r = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    out[i] = r === 0 ? 1e-7 : r;
  }
  return out;
}

/** Minimal static-KV model surface decodeBlock needs (Gemma2 satisfies it). */
export interface StaticDecodeModel {
  forward(
    idx: Tensor,
    options: { staticKV: StaticKV; residualHook?: ResidualHook },
  ): { logits: Tensor };
  config: { vocabSize: number; maxSeqLen: number };
}

/**
 * UNROLLED-K DECODE BLOCK (S_dec) on the static-KV path — greedy (argmax) or
 * on-device Gumbel-max (opts.sample.temperature>0). Runs K steps as ONE lazy
 * graph whose per-token feedback closes on-device (each step's selection tensor
 * feeds the next gather); ONE readback of the K ids at the boundary; host
 * truncation on the first stop token. See qwen3-browser for the full contract.
 */
export async function decodeBlock(
  api: Torchlette,
  model: StaticDecodeModel,
  staticKV: StaticKV,
  lastTok: number,
  K: number,
  opts?: {
    residualHook?: ResidualHook;
    stopTokens?: Set<number>;
    sample?: { temperature: number; seed: number };
  },
): Promise<{ ids: number[]; stopIndex: number }> {
  const maxSeq = model.config.maxSeqLen;
  const Kc = clipBlockToBucket(staticKV.len, K, maxSeq);
  const residualHook = opts?.residualHook;
  const sample = opts?.sample;
  const stochastic = sample !== undefined && sample.temperature > 0;
  const startLen = staticKV.len;
  const V = model.config.vocabSize;

  const idTensors: Tensor[] = [];
  let idx: Tensor = api.tensorFromArray([lastTok], [1, 1]);
  for (let j = 0; j < Kc; j++) {
    const logits = api.noGrad(
      () => model.forward(idx, { staticKV, residualHook }).logits,
    );
    const id = api.noGrad(() => {
      if (!stochastic) {
        return api.argmax(logits, { dim: -1, keepdim: false });
      }
      const uArr = gumbelUniform(sample.seed + startLen + j, V);
      const u = api.tensorFromArray(uArr, [1, 1, V]);
      const g = api.neg(api.log(api.neg(api.log(u))));
      const scaled = api.add(api.div(logits, sample.temperature), g);
      return api.argmax(scaled, { dim: -1, keepdim: false });
    });
    idTensors.push(id);
    idx = api.reshape(id, [1, 1]);
  }
  const stacked = api.cat(
    idTensors.map((t) => api.reshape(t, [1, 1])),
    1,
  );
  const rawIds = await api.cpu(stacked);
  const ids: number[] = [];
  for (let i = 0; i < Kc; i++) ids.push(Math.round(rawIds[i]));

  const stopSet = opts?.stopTokens;
  let stopIndex = Kc;
  if (stopSet) {
    for (let i = 0; i < Kc; i++) {
      if (stopSet.has(ids[i])) {
        stopIndex = i;
        break;
      }
    }
  }
  return { ids, stopIndex };
}

/** Prefill's next token via on-device Gumbel-max (full vocab, one readback) —
 *  keeps the sampled block's prefill token drawn the same way (§3.5). */
async function gumbelPrefillToken(
  api: Torchlette,
  logits: Tensor,
  lastPos: number,
  vocab: number,
  sample: { temperature: number; seed: number },
): Promise<number> {
  const id = api.noGrad(() => {
    const row = api.contiguous(
      api.narrow(api.narrow(logits, 1, lastPos, 1), 2, 0, vocab),
    );
    const uArr = gumbelUniform(sample.seed, vocab);
    const u = api.tensorFromArray(uArr, [1, 1, vocab]);
    const g = api.neg(api.log(api.neg(api.log(u))));
    const scaled = api.add(api.div(row, sample.temperature), g);
    return api.argmax(scaled, { dim: -1, keepdim: false });
  });
  const out = await api.cpu(api.reshape(id, [1, 1]));
  return Math.round(out[0]);
}

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
  // P4 CUTOVER routing (see qwen3 generate.ts). Block = greedy (argmax) or
  // pure-temperature (Gumbel, §4); residue (host per-token loop) = any top-k /
  // top-p, or opt-out (TORCHLETTE_UNROLLED_K=0/1).
  const greedy = temperature === 0;
  const topKActive = topK !== undefined && topK < vocab;
  const topPActive = topP !== undefined && topP < 1;
  // Gumbel sampled block is DEFAULT-eligible (2026-07-20, see qwen3 generate.ts):
  // the sampled-path external-destroy transient is fixed, so pure-temperature
  // sampling routes through the on-device Gumbel block by default (still under
  // the global opt-out TORCHLETTE_UNROLLED_K=0/1).
  const gumbelEligible =
    !greedy && (temperature ?? 0) > 0 && !topKActive && !topPActive;
  const flagK = unrolledKFromEnv();
  const useBlock = flagK >= 2 && (greedy || gumbelEligible);
  const blockK = useBlock ? flagK : 0;
  const blockSample =
    useBlock && gumbelEligible
      ? {
          temperature: temperature as number,
          seed: options?.seed ?? (Math.random() * 0x7fffffff) >>> 0,
        }
      : undefined;

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
      if (blockSample) {
        nextTok = await gumbelPrefillToken(
          api,
          logits,
          promptIds.length - 1,
          vocab,
          {
            temperature: blockSample.temperature,
            seed: blockSample.seed + promptIds.length - 1,
          },
        );
        logits.dispose();
      } else {
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
      }
      await api.markStep();
    }
    const prefillMs = Date.now() - t0;

    let count = 0;
    let tBuild = 0;
    let tLower = 0;
    let tFence = 0;
    let tSample = 0;
    let tStep = 0;
    // decode CapturedFn is a per-token-path construct; null on the block path.
    let decode: ReturnType<Torchlette["capture"]> | null = null;
    if (blockK >= 2) {
      // UNROLLED-K BLOCK PATH (P4 default for covered samplers). K tokens per
      // GPU-boundary readback via decodeBlock (greedy or on-device Gumbel);
      // streaming granularity K; host truncation on the first stop token.
      emit(nextTok);
      count++;
      while (
        count < maxNew &&
        !GEMMA2_STOP_TOKENS.has(nextTok) &&
        !isAborted()
      ) {
        const tb0 = performance.now();
        const { ids, stopIndex } = await decodeBlock(
          api,
          model,
          staticKV,
          nextTok,
          blockK,
          { residualHook, stopTokens: GEMMA2_STOP_TOKENS, sample: blockSample },
        );
        const tb1 = performance.now();
        await api.markStep();
        const tb2 = performance.now();
        let brk = false;
        for (let i = 0; i < ids.length; i++) {
          if (i === stopIndex || count >= maxNew) {
            brk = true;
            break;
          }
          emit(ids[i]);
          count++;
        }
        nextTok = ids[ids.length - 1];
        tBuild += tb1 - tb0;
        tStep += tb2 - tb1;
        if (brk) break;
      }
    } else {
      // Taped per-token decode (the §4 host residue + the opt-out path): the
      // residualHook is closure-captured and FROZEN for this CapturedFn's
      // lifetime — sound by construction (one generateChat call = one hook/α =
      // one fresh trace). The bucket length is the structural discriminator;
      // attnModKey guards cross-model tape replay.
      decode = api.capture(
        (idx: Tensor) =>
          api.noGrad(
            () => model.forward(idx, { staticKV, residualHook }).logits,
          ),
        {
          key: () =>
            `kv:bkt${kvBucketLen(staticKV.len + 1, maxSeq)}:mod${model.attnModKey}`,
        },
      );
      while (
        count < maxNew &&
        !GEMMA2_STOP_TOKENS.has(nextTok) &&
        !isAborted()
      ) {
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
        nextTok = sampleFromTopK(
          top.values,
          top.indices,
          temperature,
          topK,
          topP,
        );
        const tb4 = performance.now();
        await api.markStep();
        const tb5 = performance.now();
        tBuild += tb1 - tb0;
        tLower += tb2 - tb1;
        tFence += tb3 - tb2;
        tSample += tb4 - tb3;
        tStep += tb5 - tb4;
      }
    }
    // Deterministic KV lifetime end (see qwen3 generate.ts).
    for (const t of staticKV.k) t.dispose();
    for (const t of staticKV.v) t.dispose();
    staticKV.k.length = 0;
    staticKV.v.length = 0;
    await api.markStep();

    const seconds = (Date.now() - t0) / 1000;
    const per = (t: number) => Number((t / Math.max(count, 1)).toFixed(1));
    void stStats();
    return {
      promptTokens: promptIds.length,
      newTokens: count,
      seconds: Number(seconds.toFixed(2)),
      prefillMs,
      tokPerSec: Number(
        (count / Math.max(seconds - prefillMs / 1000, 0.001)).toFixed(1),
      ),
      tape: (() => {
        if (!decode) return undefined; // block path: no per-token CapturedFn
        const c = decode.stats();
        return {
          hits: c.hits,
          calls: c.calls,
          traces: c.traces,
          coldMisses: c.coldMisses,
          invalidations: c.invalidations,
          ready: c.ready,
        };
      })(),
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
