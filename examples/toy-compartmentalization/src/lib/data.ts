/**
 * MESS3: Mixed-state Environment with Shared Structure and Stochastic Switching.
 *
 * A 3-hidden-state HMM emitting tokens from {A, B, C}.
 * Joint transition/emission matrices T^(x)_{ij} = Pr(emit x, go to state j | in state i).
 *
 * From: "Transformers Represent Belief State Geometry in their Residual Stream"
 * (Shai, Riechers, et al., 2024)
 */

// Token IDs
export const TOK_A = 0;
export const TOK_B = 1;
export const TOK_C = 2;
export const TOK_TASK1 = 3;
export const TOK_TASK2 = 4;
export const VOCAB_SIZE = 5; // {A, B, C, TASK1, TASK2}
export const VOCAB_SIZE_DATA = 3; // data tokens only (for HMM)
export const NUM_STATES = 3;

// Paper's exact matrices
const PAPER_T_A = [
  [0.765,   0.00375, 0.00375],
  [0.0425,  0.0675,  0.00375],
  [0.0425,  0.00375, 0.0675 ],
];
const PAPER_T_B = [
  [0.0675,  0.0425,  0.00375],
  [0.00375, 0.765,   0.00375],
  [0.00375, 0.0425,  0.0675 ],
];
const PAPER_T_C = [
  [0.0675,  0.00375, 0.0425 ],
  [0.00375, 0.0675,  0.0425 ],
  [0.00375, 0.00375, 0.765  ],
];

/**
 * Build transition matrices by interpolating the paper's matrices toward uniform.
 * selfLoop=0.765 reproduces the paper exactly.
 * Lower values blend toward uniform transitions (faster mixing, coarser fractal).
 */
export function buildTransitionMatrices(selfLoop = 0.765): number[][][] {
  if (Math.abs(selfLoop - 0.765) < 0.001) {
    return [PAPER_T_A, PAPER_T_B, PAPER_T_C];
  }
  // Scale the paper matrices: lerp between uniform (1/9 each) and paper
  const t = selfLoop / 0.765; // 1.0 = paper, 0 = uniform
  const uniform = 1 / 9; // 3 tokens × 3 states, each row sums to 1
  const paper = [PAPER_T_A, PAPER_T_B, PAPER_T_C];
  const mats: number[][][] = [];
  for (let x = 0; x < 3; x++) {
    const M: number[][] = [];
    for (let i = 0; i < 3; i++) {
      const row: number[] = [];
      for (let j = 0; j < 3; j++) {
        row.push(uniform + t * (paper[x][i][j] - uniform));
      }
      M.push(row);
    }
    mats.push(M);
  }
  return mats;
}

export let TRANSITION_MATRICES = buildTransitionMatrices(0.765);

// Task 2 matrices: permuted version of task 1 (rotate state labels)
export let TRANSITION_MATRICES_2 = buildTask2Matrices(TRANSITION_MATRICES);

function buildTask2Matrices(t1: number[][][]): number[][][] {
  // Permute states: 0→1, 1→2, 2→0
  // This makes task 2 have the same structure but different state-token associations
  const perm = [1, 2, 0];
  const t2: number[][][] = [];
  for (let x = 0; x < VOCAB_SIZE_DATA; x++) {
    const M: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        // T2^(x)[perm[i]][perm[j]] = T1^(x)[i][j]
        M[perm[i]][perm[j]] = t1[x][i][j];
      }
    }
    t2.push(M);
  }
  return t2;
}

/** Control how much task 2 shares with task 1.
 *  sharedFrac=1.0: task 2 is identical to task 1 (same matrices).
 *  sharedFrac=0.0: task 2 is fully permuted (different state-token mapping).
 */
export let SHARED_FRACTION = 0.0;

export function buildBlendedTask2(t1: number[][][], sharedFrac: number): number[][][] {
  const permuted = buildTask2Matrices(t1);
  if (sharedFrac >= 1) return t1.map(m => m.map(r => [...r]));
  if (sharedFrac <= 0) return permuted;
  // Interpolate: t2 = sharedFrac * t1 + (1-sharedFrac) * permuted
  const t2: number[][][] = [];
  for (let x = 0; x < VOCAB_SIZE_DATA; x++) {
    const M: number[][] = [];
    for (let i = 0; i < 3; i++) {
      const row: number[] = [];
      for (let j = 0; j < 3; j++) {
        row.push(sharedFrac * t1[x][i][j] + (1 - sharedFrac) * permuted[x][i][j]);
      }
      M.push(row);
    }
    t2.push(M);
  }
  return t2;
}

export function setTransitionMatrices(selfLoop: number, sharedFrac = 0.0) {
  TRANSITION_MATRICES = buildTransitionMatrices(selfLoop);
  TRANSITION_MATRICES_2 = buildBlendedTask2(TRANSITION_MATRICES, sharedFrac);
  SHARED_FRACTION = sharedFrac;
  STATIONARY_DIST = computeStationaryDist();
}

export function emissionProbs(state: number): [number, number, number] {
  return [
    TRANSITION_MATRICES[0][state][0] + TRANSITION_MATRICES[0][state][1] + TRANSITION_MATRICES[0][state][2],
    TRANSITION_MATRICES[1][state][0] + TRANSITION_MATRICES[1][state][1] + TRANSITION_MATRICES[1][state][2],
    TRANSITION_MATRICES[2][state][0] + TRANSITION_MATRICES[2][state][1] + TRANSITION_MATRICES[2][state][2],
  ];
}

export function computeStationaryDist(): Float64Array {
  const T: number[][] = [];
  for (let i = 0; i < NUM_STATES; i++) {
    T.push([]);
    for (let j = 0; j < NUM_STATES; j++) {
      let s = 0;
      for (let x = 0; x < VOCAB_SIZE_DATA; x++) s += TRANSITION_MATRICES[x][i][j];
      T[i].push(s);
    }
  }
  let pi = new Float64Array([1/3, 1/3, 1/3]);
  for (let iter = 0; iter < 500; iter++) {
    const next = new Float64Array(NUM_STATES);
    for (let i = 0; i < NUM_STATES; i++)
      for (let j = 0; j < NUM_STATES; j++) next[j] += pi[i] * T[i][j];
    pi = next;
  }
  return pi;
}

export let STATIONARY_DIST = computeStationaryDist();

/**
 * Deterministic exploration of the reachable belief simplex.
 * Starting from the stationary distribution, recursively apply all 3 token
 * transitions up to `depth` levels. Deduplicate via coordinate quantization.
 * Returns all reachable belief states — no sampling needed.
 */
export function exploreBeliefSimplex(depth = 9): { b0: number; b1: number; b2: number }[] {
  const pts: { b0: number; b1: number; b2: number }[] = [];
  const seen = new Set<string>();

  function explore(b0: number, b1: number, b2: number, d: number) {
    const key = `${(b0 * 500 + 0.5) | 0},${(b1 * 500 + 0.5) | 0}`;
    if (seen.has(key)) return;
    seen.add(key);
    if (d > 0) pts.push({ b0, b1, b2 });
    if (d >= depth) return;
    for (let tok = 0; tok < VOCAB_SIZE_DATA; tok++) {
      const T = TRANSITION_MATRICES[tok];
      let n0 = 0, n1 = 0, n2 = 0;
      for (let j = 0; j < NUM_STATES; j++) {
        n0 += (j === 0 ? b0 : j === 1 ? b1 : b2) * T[j][0];
        n1 += (j === 0 ? b0 : j === 1 ? b1 : b2) * T[j][1];
        n2 += (j === 0 ? b0 : j === 1 ? b1 : b2) * T[j][2];
      }
      const s = n0 + n1 + n2;
      if (s > 1e-12) explore(n0 / s, n1 / s, n2 / s, d + 1);
    }
  }

  explore(STATIONARY_DIST[0], STATIONARY_DIST[1], STATIONARY_DIST[2], 0);
  return pts;
}

/**
 * Update belief state after observing token x.
 * η' = η T^(x) / (η T^(x) 1)
 */
export function updateBelief(belief: Float64Array, token: number): Float64Array {
  const T = TRANSITION_MATRICES[token];
  const next = new Float64Array(NUM_STATES);
  let sum = 0;
  for (let j = 0; j < NUM_STATES; j++) {
    for (let i = 0; i < NUM_STATES; i++) {
      next[j] += belief[i] * T[i][j];
    }
    sum += next[j];
  }
  for (let j = 0; j < NUM_STATES; j++) next[j] /= sum;
  return next;
}

/**
 * Compute the full belief trajectory for a token sequence.
 * Returns beliefs[t] = belief state AFTER seeing tokens[0..t-1].
 * beliefs[0] = stationary distribution (prior).
 */
export function beliefTrajectory(tokens: number[], seqLen: number): Float64Array[] {
  const beliefs: Float64Array[] = [new Float64Array(STATIONARY_DIST)];
  for (let t = 0; t < seqLen; t++) {
    beliefs.push(updateBelief(beliefs[t], tokens[t]));
  }
  return beliefs;
}

/**
 * Theoretical entropy: H = -sum_i pi_i sum_x P(x|i) log P(x|i)
 */
export function theoreticalEntropy(): number {
  let h = 0;
  for (let i = 0; i < NUM_STATES; i++) {
    const ep = emissionProbs(i);
    for (let x = 0; x < VOCAB_SIZE_DATA; x++) {
      if (ep[x] > 1e-12) h -= STATIONARY_DIST[i] * ep[x] * Math.log(ep[x]);
    }
  }
  return h;
}

export type DataConfig = {
  seqLen: number;
  batchSize: number;
};

export type Batch = {
  tokens: Uint32Array;  // [batch, seqLen] token IDs
  targets: Uint32Array; // [batch, seqLen-1] next-token targets
  beliefs: Float64Array[]; // [batch * seqLen] belief states for visualization
};

/** Fast token-only generation for training (no belief computation) */
/** Generate tokens for single-task training (task 1 only) */
export function generateTokens(batchSize: number, seqLen: number): Uint32Array {
  const tokens = new Uint32Array(batchSize * seqLen);
  for (let b = 0; b < batchSize; b++) {
    let state = sampleCategorical(STATIONARY_DIST);
    for (let t = 0; t < seqLen; t++) {
      const { token, nextState } = sampleTransition(state);
      tokens[b * seqLen + t] = token;
      state = nextState;
    }
  }
  return tokens;
}

/**
 * Generate dual-task tokens: [TASK_PREFIX, tok0, tok1, ...].
 * Half the batch is task 1, half is task 2.
 * For paired batches (activation penalty), same class sequences for both tasks.
 * Returns { tokens, tasks } where tasks[b] = 0 or 1.
 */
export function generateDualTaskTokens(
  batchSize: number, seqLen: number, paired = false,
): { tokens: Uint32Array; tasks: Uint8Array } {
  // seqLen includes the prefix token: [PREFIX, data0, data1, ..., data_{seqLen-2}]
  const dataLen = seqLen - 1;
  const tokens = new Uint32Array(batchSize * seqLen);
  const tasks = new Uint8Array(batchSize);
  const half = batchSize >> 1;

  for (let b = 0; b < batchSize; b++) {
    const task = b < half ? 0 : 1;
    tasks[b] = task;
    const mats = task === 0 ? TRANSITION_MATRICES : TRANSITION_MATRICES_2;
    const prefix = task === 0 ? TOK_TASK1 : TOK_TASK2;
    tokens[b * seqLen] = prefix;

    let state: number;
    if (paired && task === 1 && b - half < half) {
      // For paired: task 2 uses same initial state as corresponding task 1 sequence
      // (we re-seed from the class sequence of the paired task 1 batch element)
      state = sampleCategorical(STATIONARY_DIST);
    } else {
      state = sampleCategorical(STATIONARY_DIST);
    }

    for (let t = 0; t < dataLen; t++) {
      const { token, nextState } = sampleTransitionWith(state, mats);
      tokens[b * seqLen + 1 + t] = token;
      state = nextState;
    }
  }
  return { tokens, tasks };
}

function sampleTransitionWith(state: number, mats: number[][][]): { token: number; nextState: number } {
  const r = Math.random();
  let cum = 0;
  for (let x = 0; x < VOCAB_SIZE_DATA; x++) {
    const T = mats[x];
    for (let j = 0; j < NUM_STATES; j++) {
      cum += T[state][j];
      if (r < cum) return { token: x, nextState: j };
    }
  }
  return { token: VOCAB_SIZE_DATA - 1, nextState: NUM_STATES - 1 };
}

export function generateBatch(config: DataConfig): Batch {
  const { seqLen, batchSize } = config;
  const tokens = new Uint32Array(batchSize * seqLen);
  const targets = new Uint32Array(batchSize * (seqLen - 1));
  const beliefs: Float64Array[] = [];

  for (let b = 0; b < batchSize; b++) {
    // Sample initial state from stationary distribution
    let state = sampleCategorical(STATIONARY_DIST);

    for (let t = 0; t < seqLen; t++) {
      // Sample emission + next state jointly from T^(x)[state][:]
      const { token, nextState } = sampleTransition(state);
      tokens[b * seqLen + t] = token;
      if (t < seqLen - 1) {
        // Target at position t is the token at position t+1
        // (filled on next iteration)
      }
      state = nextState;
    }

    // Fill targets (shifted by 1)
    for (let t = 0; t < seqLen - 1; t++) {
      targets[b * (seqLen - 1) + t] = tokens[b * seqLen + t + 1];
    }

    // Compute belief trajectory for this sequence
    const seq = Array.from(tokens.slice(b * seqLen, (b + 1) * seqLen));
    const bTraj = beliefTrajectory(seq, seqLen);
    // Store beliefs for positions 0..seqLen-2 (where we predict next token)
    for (let t = 0; t < seqLen - 1; t++) {
      beliefs.push(bTraj[t + 1]); // belief AFTER seeing token t
    }
  }

  return { tokens, targets, beliefs };
}

function sampleCategorical(probs: Float64Array | number[]): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r < cum) return i;
  }
  return probs.length - 1;
}

/**
 * Generate a batch where each sequence is randomly assigned a compartment.
 * Compartment c remaps tokens: token' = token + c * VOCAB_SIZE_DATA.
 * The underlying HMM is identical — only the surface tokens differ.
 * vocabSize for the model = VOCAB_SIZE_DATA * nCompartments.
 */
export function generateBatchWithCompartments(
  config: DataConfig, nCompartments: number,
): Batch & { comps: Uint8Array } {
  const { seqLen, batchSize } = config;
  const tokens = new Uint32Array(batchSize * seqLen);
  const targets = new Uint32Array(batchSize * (seqLen - 1));
  const beliefs: Float64Array[] = [];
  const comps = new Uint8Array(batchSize);

  for (let b = 0; b < batchSize; b++) {
    const comp = Math.floor(Math.random() * nCompartments);
    comps[b] = comp;
    const offset = comp * VOCAB_SIZE_DATA;
    let state = sampleCategorical(STATIONARY_DIST);

    for (let t = 0; t < seqLen; t++) {
      const { token, nextState } = sampleTransition(state);
      tokens[b * seqLen + t] = token + offset;
      state = nextState;
    }
    for (let t = 0; t < seqLen - 1; t++) {
      targets[b * (seqLen - 1) + t] = tokens[b * seqLen + t + 1];
    }
    const seq = Array.from(tokens.slice(b * seqLen, (b + 1) * seqLen)).map(t => t - offset);
    const bTraj = beliefTrajectory(seq, seqLen);
    for (let t = 0; t < seqLen - 1; t++) {
      beliefs.push(bTraj[t + 1]);
    }
  }
  return { tokens, targets, beliefs, comps };
}

/** Generate a batch for a SPECIFIC compartment (for eval). */
export function generateBatchForComp(
  config: DataConfig, comp: number,
): Batch {
  const { seqLen, batchSize } = config;
  const tokens = new Uint32Array(batchSize * seqLen);
  const targets = new Uint32Array(batchSize * (seqLen - 1));
  const beliefs: Float64Array[] = [];
  const offset = comp * VOCAB_SIZE_DATA;

  for (let b = 0; b < batchSize; b++) {
    let state = sampleCategorical(STATIONARY_DIST);
    for (let t = 0; t < seqLen; t++) {
      const { token, nextState } = sampleTransition(state);
      tokens[b * seqLen + t] = token + offset;
      state = nextState;
    }
    for (let t = 0; t < seqLen - 1; t++) {
      targets[b * (seqLen - 1) + t] = tokens[b * seqLen + t + 1];
    }
    const seq = Array.from(tokens.slice(b * seqLen, (b + 1) * seqLen)).map(t => t - offset);
    const bTraj = beliefTrajectory(seq, seqLen);
    for (let t = 0; t < seqLen - 1; t++) {
      beliefs.push(bTraj[t + 1]);
    }
  }
  return { tokens, targets, beliefs };
}

/**
 * Generate paired (translation) sequences: same HMM trajectory rendered in
 * two different compartments, bracketed by TR tokens (like bio mirror mode).
 *
 * Format: [TR, comp_A_tok0, ..., comp_A_tokN, TR, comp_B_tok0, ..., comp_B_tokN]
 * Length = 2 * halfLen + 2, padded/truncated to seqLen.
 *
 * The TR token id is passed in (typically vocabSize - 1).
 * Targets: next-token on all positions including TR transitions.
 */
export function generatePairedBatch(
  config: DataConfig, nCompartments: number, trToken: number,
): Batch & { comps: Uint8Array } {
  const { seqLen, batchSize } = config;
  const tokens = new Uint32Array(batchSize * seqLen);
  const targets = new Uint32Array(batchSize * (seqLen - 1));
  const beliefs: Float64Array[] = [];
  const comps = new Uint8Array(batchSize);

  const halfLen = Math.floor((seqLen - 2) / 2); // content tokens per half

  for (let b = 0; b < batchSize; b++) {
    // Pick two different compartments.
    const compA = Math.floor(Math.random() * nCompartments);
    let compB = compA;
    if (nCompartments > 1) {
      const off = 1 + Math.floor(Math.random() * (nCompartments - 1));
      compB = (compA + off) % nCompartments;
    }
    comps[b] = compA;
    const offA = compA * VOCAB_SIZE_DATA;
    const offB = compB * VOCAB_SIZE_DATA;

    // Sample a single HMM trajectory.
    const baseTokens: number[] = [];
    let state = sampleCategorical(STATIONARY_DIST);
    for (let t = 0; t < halfLen; t++) {
      const { token, nextState } = sampleTransition(state);
      baseTokens.push(token);
      state = nextState;
    }

    // Build paired sequence: [TR, A-tokens, TR, B-tokens]
    const seq: number[] = [trToken];
    for (let t = 0; t < halfLen; t++) seq.push(baseTokens[t] + offA);
    seq.push(trToken);
    for (let t = 0; t < halfLen; t++) seq.push(baseTokens[t] + offB);

    const len = Math.min(seq.length, seqLen);
    for (let t = 0; t < len; t++) tokens[b * seqLen + t] = seq[t];
    for (let t = 0; t < len - 1; t++) targets[b * (seqLen - 1) + t] = seq[t + 1];

    // Beliefs: TR positions get stationary dist, content positions get trajectory beliefs.
    const bTraj = beliefTrajectory(baseTokens, halfLen);
    for (let t = 0; t < seqLen - 1; t++) {
      const seqIdx = t; // position in seq[]
      if (seqIdx === 0) {
        // TR → first A token
        beliefs.push(new Float64Array(STATIONARY_DIST));
      } else if (seqIdx <= halfLen) {
        // A-half content
        beliefs.push(bTraj[seqIdx]);
      } else if (seqIdx === halfLen + 1) {
        // second TR
        beliefs.push(bTraj[halfLen]);
      } else {
        // B-half content (same trajectory, restarted)
        const bt = seqIdx - halfLen - 2; // 0-based into B-half
        if (bt < halfLen) {
          beliefs.push(bTraj[bt + 1]);
        } else {
          beliefs.push(new Float64Array(STATIONARY_DIST));
        }
      }
    }
  }
  return { tokens, targets, beliefs, comps };
}

function sampleTransition(state: number): { token: number; nextState: number } {
  // Build flat probability vector over (token, nextState) pairs
  const r = Math.random();
  let cum = 0;
  for (let x = 0; x < VOCAB_SIZE_DATA; x++) {
    const T = TRANSITION_MATRICES[x];
    for (let j = 0; j < NUM_STATES; j++) {
      cum += T[state][j];
      if (r < cum) return { token: x, nextState: j };
    }
  }
  return { token: VOCAB_SIZE_DATA - 1, nextState: NUM_STATES - 1 };
}
