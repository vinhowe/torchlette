/**
 * Tiny char-level transformer. ~30K params total.
 *
 * 2 transformer blocks, pre-norm, manual multi-head attention via
 * api.scaledDotProductAttention. Learned token + positional embeddings.
 */

import type { Tensor, Torchlette } from "../../../src/frontend/torchlette.ts";

export interface ModelConfig {
  vocabSize: number;
  blockSize: number; // max seq length (context window)
  embedDim: number;
  numHeads: number;
  numLayers: number;
  mlpRatio: number; // MLP hidden = embedDim * mlpRatio
  device?: "cpu" | "webgpu";
}

/** Holds model parameters — client-side Tensor handles. */
export interface TransformerModel {
  config: ModelConfig;
  // Embeddings
  tokEmb: Tensor; // [vocab, D]
  posEmb: Tensor; // [blockSize, D]
  // Per-layer params
  layers: Array<{
    ln1w: Tensor; // [D]
    ln1b: Tensor;
    qkvW: Tensor; // [D, 3*D]
    qkvB: Tensor; // [3*D]
    projW: Tensor; // [D, D]
    projB: Tensor; // [D]
    ln2w: Tensor;
    ln2b: Tensor;
    mlp1W: Tensor; // [D, H]
    mlp1B: Tensor; // [H]
    mlp2W: Tensor; // [H, D]
    mlp2B: Tensor; // [D]
  }>;
  // Final layer norm + head
  lnFw: Tensor;
  lnFb: Tensor;
  headW: Tensor; // [D, vocab]  (we transpose-multiply instead of tying)
  headB: Tensor;
}

/** All parameters in a flat list (for optimizer). */
export function parameters(m: TransformerModel): Tensor[] {
  const ps: Tensor[] = [m.tokEmb, m.posEmb];
  for (const L of m.layers) {
    ps.push(
      L.ln1w, L.ln1b,
      L.qkvW, L.qkvB,
      L.projW, L.projB,
      L.ln2w, L.ln2b,
      L.mlp1W, L.mlp1B,
      L.mlp2W, L.mlp2B,
    );
  }
  ps.push(m.lnFw, m.lnFb, m.headW, m.headB);
  return ps;
}

// ============================================================================
// Initialization — deterministic, seeded
// ============================================================================

function makePrng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };
}

export function createModel(
  api: Torchlette,
  config: ModelConfig,
  seed = 1337,
): TransformerModel {
  const { vocabSize, blockSize, embedDim: D, numLayers, mlpRatio } = config;
  const H = Math.floor(D * mlpRatio);
  const rng = makePrng(seed);
  const opts = { device: config.device ?? ("cpu" as const) };

  const scale = 1 / Math.sqrt(D);
  const draw = (n: number) => Array.from({ length: n }, () => rng() * scale);
  const zeros = (n: number) => new Array<number>(n).fill(0);

  const mkParam = (values: number[], shape: number[]) =>
    api.tensorFromArray(values, shape, opts).requires_grad_(true);

  const tokEmb = mkParam(draw(vocabSize * D), [vocabSize, D]);
  const posEmb = mkParam(draw(blockSize * D), [blockSize, D]);

  const layers: TransformerModel["layers"] = [];
  for (let l = 0; l < numLayers; l++) {
    layers.push({
      ln1w: mkParam(new Array<number>(D).fill(1), [D]),
      ln1b: mkParam(zeros(D), [D]),
      qkvW: mkParam(draw(D * 3 * D), [D, 3 * D]),
      qkvB: mkParam(zeros(3 * D), [3 * D]),
      projW: mkParam(draw(D * D), [D, D]),
      projB: mkParam(zeros(D), [D]),
      ln2w: mkParam(new Array<number>(D).fill(1), [D]),
      ln2b: mkParam(zeros(D), [D]),
      mlp1W: mkParam(draw(D * H), [D, H]),
      mlp1B: mkParam(zeros(H), [H]),
      mlp2W: mkParam(draw(H * D), [H, D]),
      mlp2B: mkParam(zeros(D), [D]),
    });
  }

  return {
    config,
    tokEmb,
    posEmb,
    layers,
    lnFw: mkParam(new Array<number>(D).fill(1), [D]),
    lnFb: mkParam(zeros(D), [D]),
    headW: mkParam(draw(D * vocabSize), [D, vocabSize]),
    headB: mkParam(zeros(vocabSize), [vocabSize]),
  };
}

// ============================================================================
// Layer norm (manual, works on CPU via decomposed path)
// ============================================================================

function layerNorm(
  api: Torchlette,
  x: Tensor, // [..., D]
  w: Tensor, // [D]
  b: Tensor, // [D]
  eps = 1e-5,
): Tensor {
  // mean / var along last dim
  const mean = api.mean(x, { dim: -1, keepdim: true });
  if (typeof mean === "number") throw new Error("mean returned scalar");
  const centered = api.sub(x, mean);
  const sq = api.mul(centered, centered);
  const variance = api.mean(sq, { dim: -1, keepdim: true });
  if (typeof variance === "number") throw new Error("variance scalar");
  const epsT = api.add(variance, eps);
  const invStd = api.rsqrt(epsT);
  const normalized = api.mul(centered, invStd);
  // scale + shift with broadcasting
  return api.add(api.mul(normalized, w), b);
}

// ============================================================================
// Forward pass
// ============================================================================

export function forward(
  api: Torchlette,
  m: TransformerModel,
  tokenIds: Tensor, // [B, T] int32
): Tensor /* [B, T, vocab] */ {
  const { embedDim: D, numHeads: H, blockSize: T } = m.config;
  const headDim = D / H;
  const [B, seq] = tokenIds.shape;

  // Token + pos embeddings
  const tokE = api.embedding(m.tokEmb, tokenIds); // [B, T, D]

  // Positional: slice first `seq` rows, add to tok
  // posEmb is [blockSize, D]; we need [seq, D] then broadcast to [B, seq, D]
  const posIdsArr = Array.from({ length: seq }, (_, i) => i);
  const posIds = api.tensorFromArray(posIdsArr, [seq], { device: m.config.device ?? "cpu", dtype: "i32" });
  const posE = api.embedding(m.posEmb, posIds); // [seq, D]
  // Broadcast: add via expand
  const posEExpanded = api.expand(api.reshape(posE, [1, seq, D]), [B, seq, D]);
  let x = api.add(tokE, posEExpanded);

  // Transformer blocks
  for (const L of m.layers) {
    // Attention (pre-norm, residual)
    const nx = layerNorm(api, x, L.ln1w, L.ln1b);
    // qkv = nx @ qkvW + qkvB : [B, seq, D] @ [D, 3D] → [B, seq, 3D]
    const qkvFull = api.add(api.matmul(nx, L.qkvW), L.qkvB);
    // Split qkv along last dim via narrow.
    const q = api.narrow(qkvFull, 2, 0, D); // [B, seq, D]
    const k = api.narrow(qkvFull, 2, D, D);
    const v = api.narrow(qkvFull, 2, 2 * D, D);
    // Reshape to [B, H, seq, head_dim]
    const toMh = (t: Tensor) =>
      api.permute(api.reshape(t, [B, seq, H, headDim]), [0, 2, 1, 3]);
    const attnOut = api.scaledDotProductAttention(
      toMh(q),
      toMh(k),
      toMh(v),
      1 / Math.sqrt(headDim),
      true, // causal
    );
    // Merge heads: [B, H, seq, hd] → [B, seq, H, hd] → [B, seq, D]
    const merged = api.reshape(
      api.contiguous(api.permute(attnOut, [0, 2, 1, 3])),
      [B, seq, D],
    );
    const attnProj = api.add(api.matmul(merged, L.projW), L.projB);
    x = api.add(x, attnProj);

    // MLP (pre-norm, residual)
    const mx = layerNorm(api, x, L.ln2w, L.ln2b);
    const mh1 = api.relu(api.add(api.matmul(mx, L.mlp1W), L.mlp1B));
    const mh2 = api.add(api.matmul(mh1, L.mlp2W), L.mlp2B);
    x = api.add(x, mh2);
  }

  const xn = layerNorm(api, x, m.lnFw, m.lnFb);
  // Head: [B, seq, D] @ [D, V] + [V] → [B, seq, V]
  const logits = api.add(api.matmul(xn, m.headW), m.headB);
  // Silence unused T
  void T;
  return logits;
}

// ============================================================================
// Data pipeline
// ============================================================================

export interface Dataset {
  vocabSize: number;
  chars: string[]; // index → char
  encode: (s: string) => number[];
  decode: (ids: number[]) => string;
  text: string;
  tokens: number[];
}

export function buildCharDataset(text: string): Dataset {
  const unique = Array.from(new Set(text.split("")));
  unique.sort();
  const charToIdx = new Map<string, number>();
  unique.forEach((c, i) => charToIdx.set(c, i));

  const encode = (s: string): number[] =>
    s.split("").map((c) => charToIdx.get(c) ?? 0);
  const decode = (ids: number[]): string =>
    ids.map((i) => unique[i] ?? "?").join("");

  return {
    vocabSize: unique.length,
    chars: unique,
    encode,
    decode,
    text,
    tokens: encode(text),
  };
}

export function sampleBatch(
  ds: Dataset,
  batchSize: number,
  seqLen: number,
  rng: () => number,
): { inputs: number[]; targets: number[] } {
  const n = ds.tokens.length - seqLen - 1;
  const inputs = new Array<number>(batchSize * seqLen);
  const targets = new Array<number>(batchSize * seqLen);
  for (let b = 0; b < batchSize; b++) {
    // rng returns [-1, 1], shift to [0, 1)
    const u = (rng() + 1) / 2;
    const offset = Math.floor(u * n);
    for (let t = 0; t < seqLen; t++) {
      inputs[b * seqLen + t] = ds.tokens[offset + t];
      targets[b * seqLen + t] = ds.tokens[offset + t + 1];
    }
  }
  return { inputs, targets };
}
