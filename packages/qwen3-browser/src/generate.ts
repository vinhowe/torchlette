/**
 * Shared chat generation loop (Node server + browser): prefill + KV-cache
 * decode, temperature/top-k/top-p sampling, incremental text emission.
 *
 * The tokenizer is passed as a minimal interface so both @huggingface/
 * transformers (Node and browser) satisfy it.
 */

import type { FrontendTensor as Tensor, Torchlette } from "torchlette";
import { stStats } from "torchlette";
import type { Qwen3, ResidualHook, StaticKV } from "./model";
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
  /**
   * Residual-stream steering hook, forwarded verbatim to model.forward on BOTH
   * the prefill and every KV-decode step (the static-KV fast path threads it
   * through unchanged). Undefined → unsteered baseline (identity), so the
   * existing unsteered call sites keep working with no change.
   */
  residualHook?: ResidualHook;
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
  /** Step-tape counters from THIS generation's CapturedFn (same module
   *  instance as the capture — immune to bundler duplication). traces/
   *  coldMisses/ready discriminate WHY a tape isn't hitting. */
  tape?: {
    hits: number;
    calls: number;
    traces: number;
    coldMisses: number;
    invalidations: number;
    ready: boolean;
  };
};

export const QWEN3_STOP_TOKENS = new Set([
  151645 /* <|im_end|> */, 151643 /* <|endoftext|> */,
]);

// ============================================================================
// Unrolled-K greedy decode block (CRYSTAL-1 P1) — the static-KV product surface
// ============================================================================
//
// FLAG: TORCHLETTE_UNROLLED_K = <K> (integer). Unset / <2 → OFF (per-token host
// loop, the existing decode). K>=2 → generateChat decodes K greedy tokens per
// GPU-boundary readback via decodeBlock() (greedy only; any sampler falls back
// to the per-token loop — the §4 typed boundary).
//
// SUNSET: born 2026-07-19. Soak behind the opt-in flag → default-on (P4 cutover,
// byte-identical stream gate) → the K=1 host loop is removed for greedy and the
// flag/knob dies. A flag that outlives P6 is debt (house policy).
//
// Browser-safe env read: process may be absent; globalThis.__TORCHLETTE_ENV__ is
// the browser hook (mirrors src/core/env.ts).
export function unrolledKFromEnv(): number {
  const env: Record<string, string | undefined> =
    typeof process !== "undefined" && process.env
      ? (process.env as Record<string, string | undefined>)
      : {};
  const g = globalThis as { __TORCHLETTE_ENV__?: Record<string, string> };
  const raw =
    g.__TORCHLETTE_ENV__?.TORCHLETTE_UNROLLED_K ?? env.TORCHLETTE_UNROLLED_K;
  const k = Number(raw ?? "0");
  return Number.isFinite(k) && k >= 2 ? Math.floor(k) : 0;
}

/**
 * Clip a requested block size K to the largest count that keeps every step in
 * ONE static-KV template: all K forwards must share a bucket length (§3.4). The
 * K forwards write cache positions len..len+K-1; kvBucketLen(len+1) must equal
 * kvBucketLen(len+K). Also clipped so the last write stays inside maxSeqLen.
 */
export function clipBlockToBucket(
  len: number,
  K: number,
  maxSeqLen: number,
): number {
  const nextBoundary = Math.ceil((len + 1) / KV_BUCKET) * KV_BUCKET;
  const roomInBucket = nextBoundary - len; // forwards until the bucket edge
  const roomInSeq = maxSeqLen - len; // last write at len+K-1 <= maxSeqLen-1
  return Math.max(1, Math.min(K, roomInBucket, roomInSeq));
}

/** Minimal static-KV model surface decodeBlock needs (Qwen3 / Gemma2 satisfy). */
export interface StaticDecodeModel {
  forward(
    idx: Tensor,
    options: { staticKV: StaticKV; residualHook?: ResidualHook },
  ): { logits: Tensor };
  config: { vocabSize: number; maxSeqLen: number };
}

/**
 * UNROLLED-K GREEDY DECODE BLOCK (S_dec) on the static-KV path.
 *
 * Runs `K` greedy decode steps as ONE lazy graph whose per-token sampling
 * feedback closes ON-DEVICE: each step's `argmax(logits)` is a token-id TENSOR
 * fed straight into the next step's embedding gather — no host readback between
 * tokens. The KV cache scatters at static positions (advanced host-side at graph
 * BUILD time, so the per-step rope/scatter/mask upload tensors carry the right
 * positions as DATA). K is clipped to the bucket edge so the block is a single
 * stable template. Exactly ONE readback of the K ids happens at the boundary;
 * the host truncates on the first stop token (§3.3 — compute all K, truncate at
 * readback). Greedy only (P3 adds on-device sampling).
 *
 * `lastTok` is the token fed into the FIRST step (a host value, exactly like the
 * per-token loop's `nextTok`); the returned `ids` are the K greedy tokens it
 * produces. `staticKV.len` advances by the clipped K.
 *
 * Forces the block with a SINGLE readback — the caller owns step boundaries
 * (markStep) and the KV lifetime, as in generateChat.
 */
export async function decodeBlock(
  api: Torchlette,
  model: StaticDecodeModel,
  staticKV: StaticKV,
  lastTok: number,
  K: number,
  opts?: { residualHook?: ResidualHook; stopTokens?: Set<number> },
): Promise<{ ids: number[]; stopIndex: number }> {
  const maxSeq = model.config.maxSeqLen;
  const Kc = clipBlockToBucket(staticKV.len, K, maxSeq);
  const residualHook = opts?.residualHook;

  const idTensors: Tensor[] = [];
  // First input is the host token; every subsequent input is the on-device
  // argmax id tensor (the argmax output IS the next gather index).
  let idx: Tensor = api.tensorFromArray([lastTok], [1, 1]);
  for (let j = 0; j < Kc; j++) {
    const logits = api.noGrad(
      () => model.forward(idx, { staticKV, residualHook }).logits,
    ); // [1,1,vocab] — the decode row is contiguous
    const id = api.noGrad(() =>
      api.argmax(logits, { dim: -1, keepdim: false }),
    ); // [1,1] LAZY f32 token id
    idTensors.push(id);
    idx = api.reshape(id, [1, 1]); // feed on-device into the next gather
  }
  // ONE readback of the whole block: cat the K ids and force once.
  const stacked = api.cat(
    idTensors.map((t) => api.reshape(t, [1, 1])),
    1,
  ); // [1,Kc]
  const rawIds = await api.cpu(stacked);
  const ids: number[] = [];
  for (let i = 0; i < Kc; i++) ids.push(Math.round(rawIds[i]));

  // Host truncation: index of the first stop token (or Kc if none).
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
  const { temperature, topK, topP, residualHook } = options ?? {};
  // Unrolled-K greedy block decode (TORCHLETTE_UNROLLED_K>=K). Greedy ONLY
  // (temperature===0): any stochastic sampler stays on the per-token host loop
  // (the §4 typed boundary — sampling moves on-device in P3). When blockK<2 the
  // per-token path below is byte-identical to before.
  const greedy = temperature === 0;
  const blockK = greedy ? unrolledKFromEnv() : 0;

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
    // BASELINE BOUNDARY (task #79). The first markStep after arming
    // step-scoped cleanup snapshots the tensors alive at that moment as
    // "persistent". Take it HERE — before prefill — so the baseline captures
    // ONLY genuinely-persistent state (params + the just-materialized KV
    // slots; forceAllPending inside markStep realizes the lazy `zeros`), NOT
    // the prefill forward's transient activations. Without this the prefill's
    // full working set (~1900 storages) was alive at the prefill markStep and
    // got frozen into the baseline, exempt from reclamation forever — and
    // re-absorbed by every subsequent generation's baseline (a linear
    // cross-generation ratchet: +~1900 reachable storages/generation, growing
    // the markStep sweep unboundedly). With the baseline established first,
    // the prefill markStep becomes a RECLAIMING boundary that releases those
    // transients.
    await api.markStep();
    // Top-K prefilter readback: K=64 covers any sensible sampling topK and the
    // GPU kernel reduces the per-token readback from the full vocab row (600KB)
    // to 64 (value, index) pairs (512B). Greedy = indices[0], bit-identical to
    // a full-logits argmax (gated by examples/qwen3/topk-equivalence.ts).
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
      nextTok = greedy
        ? top.indices[0] // greedy = argmax = top-K index 0 (bit-identical)
        : sampleFromTopK(top.values, top.indices, temperature, topK, topP);
      await api.markStep();
    }
    const prefillMs = Date.now() - t0;

    let count = 0;
    let tBuild = 0;
    let tLower = 0;
    let tFence = 0;
    let tSample = 0;
    let tStep = 0;
    // [capture 2a] The decode body wrapped in a CapturedFn: it traces on early
    // tokens (the step-tape recorder observes the plan), then replays via the
    // tape once a per-bucket skeleton is ready and every derived guard passes —
    // skipping the ~1600-node graph build/fingerprint/CSE/rewrite each token.
    // ARG-BOUNDARY CONTRACT: the token enters as a TENSOR arg (the warm
    // per-step slot — its fresh value re-dresses the skeleton every call);
    // rope/scatter/mask are built INSIDE fn from staticKV.len and derived by
    // capture's upload interceptor. No hand-rolled appKey, no hand-built upload
    // payloads, no manual cache.len advance. The residualHook is
    // closure-captured and therefore FROZEN for this CapturedFn's lifetime —
    // sound here BY CONSTRUCTION: the CapturedFn's lifetime is one generateChat
    // call and the hook/α are fixed per generation (a new generation = a new
    // CapturedFn = a fresh trace with the new hook). The bucket length is the
    // one structural discriminator the arg surface can't express, so it is the
    // key's discriminator. When TORCHLETTE_STEP_TAPE is off the CapturedFn is a
    // transparent pass-through (identical behavior to before).
    // decode CapturedFn is a per-token-path construct; null on the block path.
    let decode: ReturnType<Torchlette["capture"]> | null = null;
    if (blockK >= 2) {
      // UNROLLED-K GREEDY BLOCK PATH (TORCHLETTE_UNROLLED_K). Emit the prefill
      // token once, then decode K greedy tokens per GPU-boundary readback via
      // decodeBlock (on-device argmax->gather feedback, one readback/block,
      // bucket-clipped). Streaming granularity is K (§3.3); host truncation on
      // the first stop token.
      emit(nextTok);
      count++;
      while (
        count < maxNew &&
        !QWEN3_STOP_TOKENS.has(nextTok) &&
        !isAborted()
      ) {
        const t0 = performance.now();
        const { ids, stopIndex } = await decodeBlock(
          api,
          model,
          staticKV,
          nextTok,
          blockK,
          { residualHook, stopTokens: QWEN3_STOP_TOKENS },
        );
        const t1 = performance.now();
        await api.markStep();
        const t2 = performance.now();
        let brk = false;
        for (let i = 0; i < ids.length; i++) {
          if (i === stopIndex || count >= maxNew) {
            brk = true;
            break;
          }
          emit(ids[i]);
          count++;
        }
        // Feed the LAST produced id into the next block (its KV is written
        // there); it was already emitted this iteration, so no double-emit.
        nextTok = ids[ids.length - 1];
        tBuild += t1 - t0;
        tStep += t2 - t1;
        if (brk) break;
      }
    } else {
      decode = api.capture(
        (idx: Tensor) =>
          api.noGrad(
            () => model.forward(idx, { staticKV, residualHook }).logits,
          ),
        // The modifier key is a STRUCTURAL discriminator like the bucket
        // length: a model whose attention modifiers differ must never replay
        // this model's tape ("" for the null modifier — key byte-stable).
        {
          key: () =>
            `kv:bkt${kvBucketLen(staticKV.len + 1, maxSeq)}${
              model.attnModKey ? `:mod${model.attnModKey}` : ""
            }`,
        },
      );
      while (count < maxNew && !QWEN3_STOP_TOKENS.has(nextTok) && !isAborted()) {
        emit(nextTok);
        count++;
        const t0 = performance.now();
        const logits = (await decode(
          api.tensorFromArray([nextTok], [1, 1]),
        )) as Tensor;
        const t1 = performance.now();
        // readTopK's synchronous prefix is the plan/lower/encode JS; the await
        // is the GPU fence + the 512B top-K readback.
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
        // markStep alone is the boundary: under setStepScopedCleanup its
        // end-snapshot handles reclamation, and the tape's guard-5 treats an
        // explicit endStep() as a REGIME PERTURBATION (comparator reset) — a
        // per-token endStep here kept the tape permanently ineligible in the
        // browser (the ceremony remnant f651365 missed; found via the
        // boundary-reason counters: {"endStep": 80}).
        await api.markStep();
        const t5 = performance.now();
        tBuild += t1 - t0;
        tLower += t2 - t1;
        tFence += t3 - t2;
        tSample += t4 - t3;
        tStep += t5 - t4;
      }
    }
    // End the per-generation KV lifetime DETERMINISTICALLY (task #79). The KV
    // slots are baseline-persistent (§6 static KV: root-scoped for the
    // generation), so merely dropping JS refs leaves them rc>0 until GC — and
    // under step-scoped cleanup they'd be re-snapshotted persistent, leaking
    // 56 [1,H,maxSeq,D] buffers (+ their views) per generation. generateChat
    // owns the KV cache, so it declares its death: dispose releases each
    // slot's rc; the markStep below then reclaims them (destruction is
    // fence-gated, so no UAF — any live view retains its base until it too is
    // released this boundary).
    for (const t of staticKV.k) t.dispose();
    for (const t of staticKV.v) t.dispose();
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
      tape: (() => {
        if (!decode) return undefined; // block path: no per-token CapturedFn
        const c = decode.stats();
        const r = stStats();
        return {
          hits: c.hits,
          calls: c.calls,
          traces: c.traces,
          coldMisses: c.coldMisses,
          invalidations: c.invalidations,
          ready: c.ready,
          recorder: {
            eligiblePairs: r.eligiblePairs,
            refusals: r.refusals,
            structureMisses: r.structureMisses,
            loweredPairs: r.loweredPairs,
            boundaryResets: r.boundaryResets,
            boundaryReasons: r.boundaryReasons,
            lastRefusal:
              r.refusalDiagnostics[r.refusalDiagnostics.length - 1] ?? "",
          },
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
