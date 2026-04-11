/**
 * Bracket Matching toy task — shared structural skill across per-comp vocabs.
 *
 * Each compartment c has its own set of bracket pair types. A pair type p in
 * compartment c gets tokens open[c][p], close[c][p]. Matching rule: a closer
 * must match BOTH the pair type AND the compartment of the open on top of the
 * stack.
 *
 * Three training sequence types:
 *   single-A: all opens/closes from comp A (mix of pair types allowed)
 *   single-B: all opens/closes from comp B
 *   mixed:    each opener picks comp AND pair type freshly
 *
 * Task is next-token CE. Diagnostic: close-accuracy — at every close position,
 * does the model's argmax over all close tokens match the ground truth?
 */

import { seededRandom } from './bio-data';

export type BracketsConfig = {
  contentLen: number;     // number of content tokens (must be even)
  maxDepth: number;       // max stack depth
  nPairTypes: number;     // pair types per compartment (1..4)
  nCompartments?: number; // fixed at 2 for now (per-comp story)
};

export type BracketsWorld = {
  config: BracketsConfig;
  vocabSize: number;
  nCompartments: number;
  nPairTypes: number;
  /** openTok[c][p] */
  openTok: number[][];
  /** closeTok[c][p] */
  closeTok: number[][];
  /** Flat list of all close-token ids (for argmax restriction). */
  allCloseTokens: number[];
  bosToken: number;

  /**
   * Training batch. singlePct is split evenly across all compartments.
   * mixedPct produces sequences where each opener picks comp uniformly.
   */
  generateTrainingBatch(
    batchSize: number, seqLen: number,
    singlePct: number, mixedPct: number,
  ): { tokens: Uint32Array; targets: Int32Array };

  /** Eval sets: one per compartment (single-comp) plus one mixed. */
  generateEvalBatches(nPerType: number): {
    perComp: BracketEvalSet[];   // perComp[c] = single-comp-c eval
    mixed: BracketEvalSet;
  };

  sampleSequence(which: 'A' | 'B' | 'mix'): { tokens: number[]; comps: number[]; types: number[] };

  /** UI helpers: classify a token. */
  tokenInfo(tokenId: number): { kind: 'bos' | 'open' | 'close'; comp: number; type: number } | null;

  /**
   * Generate n structurally-identical sequences rendered in each compartment.
   * Used for cosine-similarity measurement: same depth/type structure, different tokens.
   * Returns perComp[c] = Uint32Array of [n * seqLen].
   */
  generatePairedEvalBatch(n: number): { perComp: Uint32Array[]; seqLen: number };

  reseedBatches(batchSeed: number): void;
};

export type BracketEvalSet = {
  tokens: Uint32Array;
  seqLen: number;
  closePositions: Array<Array<{ pos: number; correct: number }>>;
};

export function createBracketsWorld(config: BracketsConfig, seed = 42): BracketsWorld {
  const { contentLen, maxDepth } = config;
  const nPairTypes = Math.max(1, Math.min(4, config.nPairTypes ?? 1));
  const nCompartments = config.nCompartments ?? 2;
  if (contentLen % 2 !== 0) throw new Error('contentLen must be even');
  if (contentLen < 2) throw new Error('contentLen must be >= 2');
  if (nCompartments < 1) throw new Error('nCompartments must be >= 1');

  // Token layout (flat):
  //   for c in 0..nCompartments:
  //     for p in 0..nPairTypes:
  //       open[c][p], close[c][p]
  //   BOS
  const openTok: number[][] = [];
  const closeTok: number[][] = [];
  const allCloseTokens: number[] = [];
  let next = 0;
  for (let c = 0; c < nCompartments; c++) {
    const openRow: number[] = [];
    const closeRow: number[] = [];
    for (let p = 0; p < nPairTypes; p++) {
      openRow.push(next++);
      const cl = next++;
      closeRow.push(cl);
      allCloseTokens.push(cl);
    }
    openTok.push(openRow);
    closeTok.push(closeRow);
  }
  const bosToken = next++;
  const vocabSize = next;

  // Reverse lookup: tokenId → {kind, comp, type}
  const tokenInfoMap = new Map<number, { kind: 'bos' | 'open' | 'close'; comp: number; type: number }>();
  tokenInfoMap.set(bosToken, { kind: 'bos', comp: -1, type: -1 });
  for (let c = 0; c < nCompartments; c++) {
    for (let p = 0; p < nPairTypes; p++) {
      tokenInfoMap.set(openTok[c][p], { kind: 'open', comp: c, type: p });
      tokenInfoMap.set(closeTok[c][p], { kind: 'close', comp: c, type: p });
    }
  }
  function tokenInfo(id: number) { return tokenInfoMap.get(id) ?? null; }

  let batchRng: () => number = seededRandom(seed);

  function genContent(compFn: () => number): { content: number[]; comps: number[]; types: number[] } {
    // Stack holds (comp, type) for each unmatched opener.
    const stack: Array<{ c: number; p: number }> = [];
    const content: number[] = [];
    const comps: number[] = [];
    const types: number[] = [];
    for (let i = 0; i < contentLen; i++) {
      const remaining = contentLen - i;
      const mustClose = (stack.length >= maxDepth) || (remaining === stack.length);
      const mustOpen = stack.length === 0;
      let open: boolean;
      if (mustClose) open = false;
      else if (mustOpen) open = true;
      else open = batchRng() < 0.5;
      if (open) {
        const c = compFn();
        const p = Math.floor(batchRng() * nPairTypes);
        content.push(openTok[c][p]);
        comps.push(c); types.push(p);
        stack.push({ c, p });
      } else {
        const top = stack.pop()!;
        content.push(closeTok[top.c][top.p]);
        comps.push(top.c); types.push(top.p);
      }
    }
    return { content, comps, types };
  }

  /** Build a balanced sequence. comp = specific compartment index, or -1 for mixed. */
  function buildSeq(comp: number) {
    const compFn = comp >= 0
      ? () => comp
      : () => Math.floor(batchRng() * nCompartments);
    const { content, comps, types } = genContent(compFn);
    return { tokens: [bosToken, ...content], comps: [-1, ...comps], types: [-1, ...types] };
  }

  function sampleSequence(which: 'A' | 'B' | 'mix') {
    return buildSeq(which === 'A' ? 0 : which === 'B' ? Math.min(1, nCompartments - 1) : -1);
  }

  function generateTrainingBatch(
    batchSize: number, seqLen: number,
    singlePct: number, mixedPct: number,
  ): { tokens: Uint32Array; targets: Int32Array } {
    const total = Math.max(1e-9, singlePct + mixedPct);
    const singleCut = singlePct / total;

    const tokens = new Uint32Array(batchSize * seqLen);
    const targets = new Int32Array(batchSize * seqLen).fill(-1);
    for (let b = 0; b < batchSize; b++) {
      const r = batchRng();
      let comp: number;
      if (r < singleCut) {
        // Single-comp: pick one uniformly at random.
        comp = Math.floor(batchRng() * nCompartments);
      } else {
        comp = -1; // mixed
      }
      const { tokens: seq } = buildSeq(comp);
      const len = Math.min(seq.length, seqLen);
      for (let t = 0; t < len; t++) tokens[b * seqLen + t] = seq[t];
      for (let t = 0; t < len - 1; t++) targets[b * seqLen + t] = seq[t + 1];
    }
    return { tokens, targets };
  }

  const closeTokenSet = new Set(allCloseTokens);
  function buildEval(n: number, comp: number): BracketEvalSet {
    const seqLen = contentLen + 1;
    const tokens = new Uint32Array(n * seqLen);
    const closePositions: Array<Array<{ pos: number; correct: number }>> = [];
    for (let i = 0; i < n; i++) {
      const { tokens: seq } = buildSeq(comp);
      for (let t = 0; t < seqLen; t++) tokens[i * seqLen + t] = seq[t];
      const closes: Array<{ pos: number; correct: number }> = [];
      for (let t = 1; t < seq.length; t++) {
        const tok = seq[t];
        if (closeTokenSet.has(tok)) closes.push({ pos: t - 1, correct: tok });
      }
      closePositions.push(closes);
    }
    return { tokens, seqLen, closePositions };
  }

  function generateEvalBatches(nPerType: number) {
    const perComp: BracketEvalSet[] = [];
    for (let c = 0; c < nCompartments; c++) perComp.push(buildEval(nPerType, c));
    return { perComp, mixed: buildEval(nPerType, -1) };
  }

  /** Translate a sequence from one comp to another (same structure, different tokens). */
  function translateSeq(tokens: number[], toComp: number): number[] {
    return tokens.map(t => {
      const info = tokenInfoMap.get(t);
      if (!info || info.kind === 'bos') return t;
      if (info.kind === 'open') return openTok[toComp][info.type];
      return closeTok[toComp][info.type];
    });
  }

  function generatePairedEvalBatch(n: number): { perComp: Uint32Array[]; seqLen: number } {
    const seqLen = contentLen + 1;
    // Generate n sequences in comp 0, then translate to each comp.
    const base: number[][] = [];
    for (let i = 0; i < n; i++) {
      const { tokens } = buildSeq(0);
      base.push(tokens);
    }
    const perComp: Uint32Array[] = [];
    for (let c = 0; c < nCompartments; c++) {
      const arr = new Uint32Array(n * seqLen);
      for (let i = 0; i < n; i++) {
        const translated = c === 0 ? base[i] : translateSeq(base[i], c);
        for (let t = 0; t < seqLen; t++) arr[i * seqLen + t] = translated[t];
      }
      perComp.push(arr);
    }
    return { perComp, seqLen };
  }

  function reseedBatches(batchSeed: number) { batchRng = seededRandom(batchSeed); }

  return {
    config, vocabSize, nCompartments, nPairTypes, openTok, closeTok, allCloseTokens, bosToken,
    tokenInfo,
    generateTrainingBatch, generateEvalBatches, generatePairedEvalBatch, sampleSequence, reseedBatches,
  };
}
