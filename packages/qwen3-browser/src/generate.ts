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
  /**
   * Seed for the on-device Gumbel-max sampler (the block's pure-temperature
   * path, §3.5 seed-as-DATA). Fixed per generation so the sampled stream is
   * byte-reproducible. Only consulted when the block Gumbel path is active
   * (temperature>0 with no top-k/top-p restriction and the flag on); the host
   * top-k/top-p residue keeps its own Math.random draw. Undefined → a fresh
   * random seed per generation (run-to-run variety, matching the host sampler).
   */
  seed?: number;
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
// Unrolled-K decode block (CRYSTAL-1) — the static-KV product surface
// ============================================================================
//
// FLAG (P4 CUTOVER — now DEFAULT-ON, opt-OUT): TORCHLETTE_UNROLLED_K = <K>.
//   - UNSET → the block is THE decode path, K = UNROLLED_K_DEFAULT.
//   - "0" or "1" → OFF (opt out to the per-token host loop, the pre-cutover
//     decode) — the soak escape hatch.
//   - ">=2" → the block with that explicit K.
// generateChat routes GREEDY (argmax) decode through the block by DEFAULT. The
// on-device GUMBEL sampled block (pure-temperature, §4) is opt-IN via an
// explicit flag (unrolledKExplicit) — NOT default-on: the sampled path carries a
// pre-existing dropped-submit transient (see unrolledKExplicit). The typed
// residue that STAYS on the per-token host loop: any top-k / top-p / nucleus
// (§4 — a full-distribution host read the block cannot express), default
// (unflagged) temperature sampling, and any non-static-KV path. So the shipped
// top-k+top-p demo samplers are the residue; the block is byte-identical to the
// host loop for the covered samplers (the mother gate) with the steering hook
// composed (t-uk-steering-diff).
//
// K DEFAULT = 4. The measured multiplier (design §P3'): K=4 gives the best
// compiled win (host/def 1.43×, low/def 2.55×) AND the finest streaming
// granularity; K=8 is the other sweet-spot cell (host/def 1.12×); K=16
// REGRESSES (host/def 0.50×) — so the default is clamped to the low end of the
// {4,8} sweet spot.
//
// SUNSET: born 2026-07-19 (opt-in), default-on 2026-07-20 (P4). P6 removes the
// K=1 host loop for the covered samplers and retires the flag/knob. A flag that
// outlives P6 is debt (house policy).
//
// Browser-safe env read: process may be absent; globalThis.__TORCHLETTE_ENV__ is
// the browser hook (mirrors src/core/env.ts).
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
  if (raw === undefined) return UNROLLED_K_DEFAULT; // default-on (greedy)
  const k = Number(raw);
  return Number.isFinite(k) && k >= 2 ? Math.floor(k) : 0; // "0"/"1" → off
}
/**
 * Was TORCHLETTE_UNROLLED_K set EXPLICITLY (≥2)? The GREEDY block is default-on
 * (opt-out); the on-device GUMBEL sampled block is default-OFF, opt-IN via an
 * explicit flag. Reason (P4-discovered, 2026-07-20): the sampled block path
 * carries a PRE-EXISTING dropped-submit transient (a materialized first-token
 * upload whose registry buffer is destroyed without parking — `_lastHarvestIds`
 * tracks harvest RESULTS, not external-input uploads; visible as "used in submit
 * while destroyed" in t-uk-gumbel-parity, which never asserted zero GPU errors).
 * It sometimes corrupts (t-uk-steering-diff surfaced it). Greedy is unaffected
 * (its idx feed IS a harvest result → parked, block-diff STRICT_GPU-clean). So
 * we do NOT ship the sampled block default-on; it activates only when a caller
 * explicitly opts in with the flag, pending the transient fix (named blocker).
 */
export function unrolledKExplicit(): boolean {
  return unrolledKRaw() !== undefined;
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

/**
 * Deterministic per-position uniform noise for on-device Gumbel-max — the
 * seed-as-DATA channel (§3.5): the host computes `u ~ U(0,1)^V` from a
 * position-canonical seed and it is uploaded as an input tensor, so a replayed
 * (or lowered) block draws the SAME stream — byte-reproducible, no on-device RNG
 * data-source (whose lifetime is the frozen-uniform / buffer-destroy hazard).
 * mulberry32 — a fast, well-distributed 32-bit PRNG; the SINGLE SOURCE both
 * decodeBlock and the parity reference (t-uk-gumbel-parity) draw from, so the
 * on-device selection and any host reference are byte-identical by construction.
 */
export function gumbelUniform(seed: number, n: number): Float32Array {
  let a = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    // (0,1): nudge exact 0 up so -log(-log(u)) stays finite (matched host-side).
    const r = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    out[i] = r === 0 ? 1e-7 : r;
  }
  return out;
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
 * readback).
 *
 * SAMPLING (P3): with `opts.sample = { temperature>0, seed }` the feedback
 * selection becomes on-device GUMBEL-MAX — `argmax(logits/temperature + g)`,
 * `g = -log(-log(u))`. The uniform `u` is the seed-as-DATA channel (§3.5): drawn
 * HOST-side from a position-canonical seed (`gumbelUniform(seed +
 * absolutePosition, V)`, the single source the parity reference also uses) and
 * uploaded as a tensor, while the transform + argmax close ON-DEVICE (no
 * per-token readback inside the block). So the stream is byte-reproducible: same
 * seed ⇒ same uniforms ⇒ same ids, and a per-token host loop drawing the same
 * per-position seeds produces byte-identical ids (the parity gate,
 * t-uk-gumbel-parity). `temperature===0` (or no `sample`) is the greedy
 * `argmax(logits)` path, unchanged. Full-vocab top-p / arbitrary host samplers
 * stay the K=1 host-loop residue (§4) — not expressible as a block node.
 *
 * `lastTok` is the token fed into the FIRST step (a host value, exactly like the
 * per-token loop's `nextTok`); the returned `ids` are the K tokens it produces.
 * `staticKV.len` advances by the clipped K.
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
  const startLen = staticKV.len; // absolute position of this block's first step
  const V = model.config.vocabSize;

  const idTensors: Tensor[] = [];
  // First input is the host token; every subsequent input is the on-device
  // argmax/sample id tensor (the selection output IS the next gather index).
  let idx: Tensor = api.tensorFromArray([lastTok], [1, 1]);
  for (let j = 0; j < Kc; j++) {
    const logits = api.noGrad(
      () => model.forward(idx, { staticKV, residualHook }).logits,
    ); // [1,1,vocab] — the decode row is contiguous
    const id = api.noGrad(() => {
      if (!stochastic) {
        return api.argmax(logits, { dim: -1, keepdim: false }); // greedy
      }
      // Gumbel-max: id = argmax(logits/temp + (-log(-log(u)))). The uniform is
      // drawn HOST-side from a position-canonical seed (seed + absolutePosition)
      // and uploaded as DATA (§3.5), so the draw is a deterministic function of
      // position — replay- and reference-byte-reproducible — while the transform
      // + selection close ON-DEVICE (no per-token readback inside the block).
      const uArr = gumbelUniform(sample.seed + startLen + j, V);
      const u = api.tensorFromArray(uArr, [1, 1, V]);
      const g = api.neg(api.log(api.neg(api.log(u)))); // -log(-log(u))
      const scaled = api.add(api.div(logits, sample.temperature), g);
      return api.argmax(scaled, { dim: -1, keepdim: false });
    }); // [1,1] LAZY f32 token id
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

/**
 * Prefill's next token via on-device GUMBEL-max over the full vocab, one
 * readback. Mirrors decodeBlock's feedback selection so the sampled block's
 * prefill token is drawn the same way (full-vocab, seed-as-DATA, position
 * canonical) — the whole sampled stream stays on-device-consistent and
 * byte-reproducible under a fixed seed. `lastPos` is the last prompt position
 * (whose logits row predicts the token); the row is forced contiguous to honor
 * the arg-reduce contiguity seam (§2 Probe 1 sharp edge).
 */
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
    ); // [1,1,V] contiguous
    const uArr = gumbelUniform(sample.seed, vocab);
    const u = api.tensorFromArray(uArr, [1, 1, vocab]);
    const g = api.neg(api.log(api.neg(api.log(u)))); // -log(-log(u))
    const scaled = api.add(api.div(row, sample.temperature), g);
    return api.argmax(scaled, { dim: -1, keepdim: false });
  });
  const out = await api.cpu(api.reshape(id, [1, 1]));
  return Math.round(out[0]);
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
  // P4 CUTOVER routing. The unrolled-K block is the DEFAULT decode path for the
  // samplers it covers on-device: GREEDY (argmax) and pure-temperature
  // (GUMBEL-max, §4). The TYPED RESIDUE that stays on the per-token host loop —
  // any top-k restriction OR top-p / nucleus (a full-distribution host read the
  // block cannot express, §4). TORCHLETTE_UNROLLED_K=0/1 opts the whole thing
  // out (blockK=0 → the pre-cutover host loop, byte-identical to before).
  const greedy = temperature === 0;
  const topKActive = topK !== undefined && topK < vocab;
  const topPActive = topP !== undefined && topP < 1;
  // Gumbel sampled block is opt-IN (explicit flag) — the sampled path's
  // pre-existing dropped-submit transient (see unrolledKExplicit) is not shipped
  // default-on. Greedy is default-on.
  const gumbelEligible =
    !greedy &&
    (temperature ?? 0) > 0 &&
    !topKActive &&
    !topPActive &&
    unrolledKExplicit();
  const flagK = unrolledKFromEnv(); // 0 ⇒ opt-out; else the block size
  const useBlock = flagK >= 2 && (greedy || gumbelEligible);
  const blockK = useBlock ? flagK : 0;
  // Gumbel sampling for the block: the per-block seed flows as DATA (§3.5),
  // position-canonical INSIDE decodeBlock (seed+absolutePosition). Fixed per
  // generation so replay/reference are byte-reproducible; undefined ⇒ a fresh
  // random seed (run-to-run variety, matching the host sampler's Math.random).
  const blockSample =
    useBlock && gumbelEligible
      ? {
          temperature: temperature as number,
          seed: options?.seed ?? ((Math.random() * 0x7fffffff) >>> 0),
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
      if (blockSample) {
        // Gumbel-max prefill token (full-vocab, on-device) so the whole sampled
        // stream stays on-device-consistent + deterministic. Seeded at the last
        // prompt position (promptIds.length-1); decodeBlock's ids[j] then draw
        // seed+promptIds.length+j — a clean monotonic per-position seed sequence.
        nextTok = await gumbelPrefillToken(api, logits, promptIds.length - 1, vocab, {
          temperature: blockSample.temperature,
          seed: blockSample.seed + promptIds.length - 1,
        });
        logits.dispose();
      } else {
        const top = await api.readTopK(logits, K_PREFILTER, {
          offset: (promptIds.length - 1) * vocab,
          length: vocab,
        });
        logits.dispose();
        nextTok = greedy
          ? top.indices[0] // greedy = argmax = top-K index 0 (bit-identical)
          : sampleFromTopK(top.values, top.indices, temperature, topK, topP);
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
      // UNROLLED-K BLOCK PATH (the P4 default for covered samplers). Emit the
      // prefill token once, then decode K tokens per GPU-boundary readback via
      // decodeBlock (on-device argmax/Gumbel->gather feedback, one readback/
      // block, bucket-clipped). Greedy when blockSample is undefined; on-device
      // Gumbel-max when set (pure-temperature, §4). Streaming granularity is K
      // (§3.3); host truncation on the first stop token.
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
          { residualHook, stopTokens: QWEN3_STOP_TOKENS, sample: blockSample },
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
      while (
        count < maxNew &&
        !QWEN3_STOP_TOKENS.has(nextTok) &&
        !isAborted()
      ) {
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
