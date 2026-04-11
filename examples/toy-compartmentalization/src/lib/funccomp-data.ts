/**
 * Function Composition toy task — tests generalization across compartments.
 *
 * Two fixed functions defined at world creation:
 *   f: {0..N-1} → {0..N-1}   a random permutation (complex, must be memorized)
 *   g: {0..N-1} → {0..M-1}   g(y) = y mod M (trivial)
 *
 * Compartment A teaches f densely (all x shown), in A's vocabulary:
 *   [BOS, x_A(i), ASK_F, y_A(f(i))]
 *
 * Compartment B teaches the composite g∘f sparsely (only x ∈ TRAIN_X), in B's
 * vocabulary with a shared z output:
 *   [BOS, x_B(i), ASK_GF, z(g(f(i)))]
 *
 * Translation examples link x_A[i] and x_B[i] (same underlying input i):
 *   [TR, x_A(i), x_B(i)]
 *
 * Eval (the generalization test): for x ∈ TEST_X held out from comp B
 * training, ask [BOS, x_B(i), ASK_GF] and measure whether the model
 * predicts z(g(f(i))) correctly.
 *
 * Unified model: learns f from A, aligns x_A/x_B via translation, applies g
 * to f's output → generalizes to held-out x.
 * Compartmentalized model: can only memorize B's (x_B, z) training pairs
 * → test accuracy at chance (1/M) for held-out inputs.
 */

import { seededRandom } from './bio-data';

export type FuncCompConfig = {
  nInputs: number;       // N: size of f's domain and codomain
  gModulus: number;      // M: g(y) = y mod M, output space size
  bCoveragePct: number;  // % of inputs used in comp B training (TRAIN_X)
  /** If true, x_A[i] and x_B[i] share the same token id (no vocab split). */
  sharedInputVocab?: boolean;
};

export type FuncCompWorld = {
  config: FuncCompConfig;
  vocabSize: number;

  /** Ground truth: f[i] = f(i), a permutation of [0..N-1]. */
  f: number[];
  /** Inputs used for comp B training (sorted). */
  trainX: number[];
  /** Inputs held out from comp B training (sorted). */
  testX: number[];

  // Token vocabularies
  xTokensA: number[];    // x_A[i] = token id
  yTokensA: number[];    // y_A[i] = token id
  xTokensB: number[];    // x_B[i] = token id
  zTokens: number[];     // z[j] for j in [0..M-1] (shared)
  bosToken: number;
  askFToken: number;
  askGfToken: number;
  trToken: number;

  /**
   * Training batch. All sequences padded to seqLen with pad=0, targets=-1.
   * The three percentages are independent and normalized internally.
   */
  generateTrainingBatch(
    batchSize: number, seqLen: number,
    compAPct: number, compBPct: number, translationPct: number,
  ): { tokens: Uint32Array; targets: Int32Array };

  /** Eval for comp A: measure f accuracy on all inputs. */
  generateCompAEvalBatch(): { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
  /** Eval for comp B trained inputs. */
  generateCompBTrainEvalBatch(): { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };
  /** Eval for comp B held-out inputs — the headline generalization test. */
  generateCompBHeldoutEvalBatch(): { tokens: Uint32Array; targets: Uint32Array; seqLen: number; promptLen: number };

  reseedBatches(batchSeed: number): void;
};

export function createFuncCompWorld(config: FuncCompConfig, seed = 42): FuncCompWorld {
  const { nInputs: N, gModulus: M, bCoveragePct } = config;
  if (N < 2) throw new Error('nInputs must be >= 2');
  if (M < 2 || M > N) throw new Error('gModulus must be in [2, nInputs]');

  const rng = seededRandom(seed);
  let batchRng: () => number = seededRandom(seed);

  // --- Random permutation f ---
  const f: number[] = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [f[i], f[j]] = [f[j], f[i]];
  }

  // --- Deterministic train/test split (first k inputs as train) ---
  const k = Math.max(1, Math.min(N - 1, Math.floor((N * bCoveragePct) / 100)));
  const trainX = Array.from({ length: k }, (_, i) => i);
  const testX = Array.from({ length: N - k }, (_, i) => k + i);

  // --- Token layout ---
  // [x_A] [y_A] [x_B (or alias to x_A)] [z] [BOS, ASK_F, ASK_GF, TR]
  const shared = config.sharedInputVocab === true;
  const xTokensA: number[] = [];
  const yTokensA: number[] = [];
  const xTokensB: number[] = [];
  const zTokens: number[] = [];
  let next = 0;
  for (let i = 0; i < N; i++) xTokensA.push(next++);
  for (let i = 0; i < N; i++) yTokensA.push(next++);
  if (shared) {
    // B reuses A's x-tokens — no vocab split.
    for (let i = 0; i < N; i++) xTokensB.push(xTokensA[i]);
  } else {
    for (let i = 0; i < N; i++) xTokensB.push(next++);
  }
  for (let i = 0; i < M; i++) zTokens.push(next++);
  const bosToken = next++;
  const askFToken = next++;
  const askGfToken = next++;
  const trToken = next++;
  const vocabSize = next;

  function g(y: number): number { return y % M; }

  // --- Sequence builders ---
  function compASeq(i: number): number[] {
    return [bosToken, xTokensA[i], askFToken, yTokensA[f[i]]];
  }
  function compBSeq(i: number): number[] {
    return [bosToken, xTokensB[i], askGfToken, zTokens[g(f[i])]];
  }
  function translationSeq(i: number): number[] {
    return [trToken, xTokensA[i], xTokensB[i]];
  }

  function generateTrainingBatch(
    batchSize: number, seqLen: number,
    compAPct: number, compBPct: number, translationPct: number,
  ): { tokens: Uint32Array; targets: Int32Array } {
    const total = Math.max(1e-9, compAPct + compBPct + translationPct);
    const aCut = compAPct / total;
    const bCut = (compAPct + compBPct) / total;

    const tokens = new Uint32Array(batchSize * seqLen);
    const targets = new Int32Array(batchSize * seqLen).fill(-1);

    for (let b = 0; b < batchSize; b++) {
      const r = batchRng();
      let seq: number[];
      if (r < aCut) {
        // Comp A: dense, any input
        const i = Math.floor(batchRng() * N);
        seq = compASeq(i);
      } else if (r < bCut) {
        // Comp B: sparse, only trainX
        const i = trainX[Math.floor(batchRng() * trainX.length)];
        seq = compBSeq(i);
      } else {
        // Translation: any input
        const i = Math.floor(batchRng() * N);
        seq = translationSeq(i);
      }
      const len = Math.min(seq.length, seqLen);
      for (let t = 0; t < len; t++) tokens[b * seqLen + t] = seq[t];
      for (let t = 0; t < len - 1; t++) targets[b * seqLen + t] = seq[t + 1];
    }
    return { tokens, targets };
  }

  // --- Eval helpers: 4-token sequences, predict target at position promptLen-1=2 ---
  function buildEvalFromIndices(
    inputs: number[], seqBuilder: (i: number) => number[], targetFn: (i: number) => number,
  ) {
    const n = inputs.length;
    const seqLen = 4;        // [BOS, x_*, ASK_*, ans]
    const promptLen = 3;     // predict ans from position 2
    const tokens = new Uint32Array(n * seqLen);
    const targets = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      const seq = seqBuilder(inputs[i]);
      for (let t = 0; t < seqLen; t++) tokens[i * seqLen + t] = seq[t];
      targets[i] = targetFn(inputs[i]);
    }
    return { tokens, targets, seqLen, promptLen };
  }

  function generateCompAEvalBatch() {
    const inputs = Array.from({ length: N }, (_, i) => i);
    return buildEvalFromIndices(inputs, compASeq, (i) => yTokensA[f[i]]);
  }
  function generateCompBTrainEvalBatch() {
    return buildEvalFromIndices(trainX, compBSeq, (i) => zTokens[g(f[i])]);
  }
  function generateCompBHeldoutEvalBatch() {
    return buildEvalFromIndices(testX, compBSeq, (i) => zTokens[g(f[i])]);
  }

  function reseedBatches(batchSeed: number) { batchRng = seededRandom(batchSeed); }

  return {
    config, vocabSize, f, trainX, testX,
    xTokensA, yTokensA, xTokensB, zTokens,
    bosToken, askFToken, askGfToken, trToken,
    generateTrainingBatch,
    generateCompAEvalBatch, generateCompBTrainEvalBatch, generateCompBHeldoutEvalBatch,
    reseedBatches,
  };
}
