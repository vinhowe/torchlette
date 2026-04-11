/**
 * XOR Disambiguation toy task.
 *
 * Each example samples fresh (F1, F2) ∈ {0,1}². Each compartment has its
 * own 2-element "bit" vocabulary that carries one partial observation:
 *
 *   - Compartment A's bit token encodes  obs_A = F1 ⊕ F2   (the XOR bit)
 *   - Compartment B's bit token encodes  obs_B = F1        (the F1 bit)
 *
 * The supervised target is always F2, written with shared output bit tokens.
 *
 * Sequence types (all end with ASK_F2 → bit_F2):
 *   single-A:  [BOS, bit_A(obs_A), ASK_F2, bit_F2(F2)]   4 tokens
 *   single-B:  [BOS, bit_B(obs_B), ASK_F2, bit_F2(F2)]   4 tokens
 *   mixed:     [BOS, bit_A(obs_A), bit_B(obs_B), ASK_F2, bit_F2(F2)]   5 tokens
 *
 * Information content:
 *   H(F2 | obs_A) = H(F2 | obs_B) = log 2 ≈ 0.693 nats     (fundamental)
 *   H(F2 | obs_A, obs_B) = 0                               (F2 = obs_A ⊕ obs_B)
 *
 * So a model trained only on single-comp examples hits a log(2) floor on
 * its ASK_F2 prediction (for either single-comp or mixed test). A model
 * trained on mixed examples can drive mixed-test loss → 0 while still
 * floor-limited on single-comp test.
 *
 * The "compartmentalization" here is SURFACE: comp A's bits live in a
 * different vocab slot than comp B's. A transformer that only sees one
 * vocab in a training example can't learn to combine them; mixed training
 * forces a joint XOR circuit across the per-comp embeddings.
 */

import { seededRandom } from './bio-data';

export type XorConfig = {
  nCompartments: number;   // must be 2 for the canonical setup
};

export type XorWorld = {
  config: XorConfig;
  vocabSize: number;
  /** bitTokensPerComp[c] = [token for obs=0, token for obs=1] */
  bitTokensPerComp: Array<[number, number]>;
  /** Output bit tokens (shared): bitOut[bit] */
  bitOut: [number, number];
  bosToken: number;
  askF2Token: number;

  /**
   * Training batch with configurable mix. The three fractions are independent
   * probabilities and normalized internally (they need not sum to 1).
   */
  generateTrainingBatch(
    batchSize: number, seqLen: number,
    singleAPct: number, singleBPct: number, mixedPct: number,
  ): { tokens: Uint32Array; targets: Int32Array };

  /** Eval batch of each sequence type separately. */
  generateEvalBatches(nPerType: number): {
    singleA: { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
    singleB: { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
    mixed:   { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
  };

  reseedBatches(batchSeed: number): void;
};

export function createXorWorld(config: XorConfig, seed = 42): XorWorld {
  const { nCompartments } = config;
  if (nCompartments < 2) throw new Error('XOR task needs nCompartments >= 2');

  let batchRng: () => number = seededRandom(seed);

  // Token layout: [bit_0[0], bit_0[1], bit_1[0], bit_1[1], ..., bit_out[0], bit_out[1], BOS, ASK_F2]
  const bitTokensPerComp: Array<[number, number]> = [];
  for (let c = 0; c < nCompartments; c++) {
    bitTokensPerComp.push([c * 2, c * 2 + 1]);
  }
  const sharedBase = nCompartments * 2;
  const bitOut: [number, number] = [sharedBase, sharedBase + 1];
  const bosToken = sharedBase + 2;
  const askF2Token = sharedBase + 3;
  const vocabSize = sharedBase + 4;

  /** Sample one example. Returns the token sequence. */
  function singleASeq(f1: number, f2: number): number[] {
    const xor = f1 ^ f2;
    return [bosToken, bitTokensPerComp[0][xor], askF2Token, bitOut[f2]];
  }
  function singleBSeq(f1: number, f2: number): number[] {
    return [bosToken, bitTokensPerComp[1][f1], askF2Token, bitOut[f2]];
  }
  function mixedSeq(f1: number, f2: number): number[] {
    const xor = f1 ^ f2;
    // Randomly order A and B observations.
    const aFirst = batchRng() < 0.5;
    const aTok = bitTokensPerComp[0][xor];
    const bTok = bitTokensPerComp[1][f1];
    return aFirst
      ? [bosToken, aTok, bTok, askF2Token, bitOut[f2]]
      : [bosToken, bTok, aTok, askF2Token, bitOut[f2]];
  }

  function sampleFeatures(): [number, number] {
    const f1 = batchRng() < 0.5 ? 0 : 1;
    const f2 = batchRng() < 0.5 ? 0 : 1;
    return [f1, f2];
  }

  function generateTrainingBatch(
    batchSize: number, seqLen: number,
    singleAPct: number, singleBPct: number, mixedPct: number,
  ): { tokens: Uint32Array; targets: Int32Array } {
    const total = Math.max(1e-9, singleAPct + singleBPct + mixedPct);
    const aCut = singleAPct / total;
    const bCut = (singleAPct + singleBPct) / total;

    const tokens = new Uint32Array(batchSize * seqLen);
    const targets = new Int32Array(batchSize * seqLen).fill(-1);

    for (let b = 0; b < batchSize; b++) {
      const r = batchRng();
      const [f1, f2] = sampleFeatures();
      const seq = r < aCut
        ? singleASeq(f1, f2)
        : r < bCut
          ? singleBSeq(f1, f2)
          : mixedSeq(f1, f2);
      const len = Math.min(seq.length, seqLen);
      for (let i = 0; i < len; i++) tokens[b * seqLen + i] = seq[i];
      for (let i = 0; i < len - 1; i++) targets[b * seqLen + i] = seq[i + 1];
    }
    return { tokens, targets };
  }

  function buildEval(n: number, gen: (f1: number, f2: number) => number[]) {
    // First example to learn sequence length.
    const probe = gen(0, 0);
    const seqLen = probe.length;
    // Prompt = everything before the F2 answer token (last token).
    // We predict the answer at logits[promptLen-1] → token at position promptLen.
    const promptLen = seqLen - 1;
    const tokens = new Uint32Array(n * seqLen);
    const targets = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const [f1, f2] = sampleFeatures();
      const seq = gen(f1, f2);
      for (let t = 0; t < seqLen; t++) tokens[i * seqLen + t] = seq[t];
      targets[i] = f2;
    }
    return { tokens, targets, seqLen, promptLen };
  }

  function generateEvalBatches(nPerType: number): {
    singleA: { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
    singleB: { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
    mixed:   { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
  } {
    return {
      singleA: buildEval(nPerType, singleASeq),
      singleB: buildEval(nPerType, singleBSeq),
      mixed:   buildEval(nPerType, mixedSeq),
    };
  }

  function reseedBatches(batchSeed: number) { batchRng = seededRandom(batchSeed); }

  return {
    config, vocabSize, bitTokensPerComp, bitOut, bosToken, askF2Token,
    generateTrainingBatch, generateEvalBatches, reseedBatches,
  };
}
