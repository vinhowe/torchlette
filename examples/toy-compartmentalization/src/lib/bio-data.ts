/**
 * Bio3-style synthetic factual data for compartmentalization experiments.
 *
 * Entities have attributes with values. Multiple "compartments" encode the
 * same facts using disjoint token sets. The model must decide whether to
 * learn separate fact stores per compartment or a unified representation.
 *
 * Token layout per compartment c:
 *   Entity names:  entityTokens[c][entityId][0..tokensPerEntity-1]
 *   Attr templates: attrTokens[c][attrId]  (single token: "color is")
 *   Values:        valueTokens[c][valueId][0..tokensPerValue-1]
 *   Separator:     sepToken (shared across compartments)
 *   QA prompt:     qaTokens[c][attrId]
 *   Translation:   TR (shared across compartments)
 */

export type BioConfig = {
  nEntities: number;       // e.g. 100
  nAttributes: number;     // e.g. 6
  nValues: number;         // e.g. 20 (per attribute)
  nCompartments: number;   // e.g. 2
  tokensPerEntity: number; // 1-3
  tokensPerValue: number;  // 1-2
  bankSize: number;        // base token bank size per compartment (for multi-token names)
  /**
   * If true, each (entity, attr, value, sep) tuple within a bio paragraph
   * samples an independent compartment. Each tuple is still internally
   * compartment-coherent (entity/attr/value/sep all share the same comp)
   * but consecutive tuples in the same bio may differ. Provides direct
   * cross-compartment supervision in a single sequence without TR markers.
   */
  mixCompartments?: boolean;
};

/**
 * Translation-style supervision signals for cross-compartment learning.
 *
 *  - 'mirror': full fact restated in both compartments, separated by TR.
 *      [TR] entityA attrA valueA [TR] entityB attrB valueB
 *    Strongest signal; model can attend back to A to predict B.
 *  - 'continuation': fact starts in compartment A, switches mid-fact to B.
 *      e.g. [entityA attrA TR valueB] or [entityA TR attrB valueB]
 *    Split point randomized each sample. Forces mid-sequence transfer.
 *  - 'dictionary': single token-pair translation with no fact context.
 *      [TR] entityTokensA entityTokensB   (or attr-pair, or value-pair)
 *    Pure token-to-token mapping — no semantic/fact information.
 */
export type TranslationMode = 'mirror' | 'continuation' | 'dictionary';

export const DEFAULT_BIO_CONFIG: BioConfig = {
  nEntities: 100,
  nAttributes: 6,
  nValues: 20,
  nCompartments: 2,
  tokensPerEntity: 1,
  tokensPerValue: 1,
  bankSize: 200,
  mixCompartments: false,
};

export type BioWorld = {
  config: BioConfig;
  vocabSize: number;

  /** Ground truth: facts[entityId][attrId] = valueId */
  facts: number[][];

  /** Token IDs for entities per compartment: entityTokens[comp][entityId] = tokenId[] */
  entityTokens: number[][][];
  /** Token IDs for attribute templates: attrTokens[comp][attrId] = tokenId */
  attrTokens: number[][];
  /** Token IDs for values: valueTokens[comp][valueId] = tokenId[] */
  valueTokens: number[][][];
  /** Separator token (shared across compartments — structural, not content) */
  sepToken: number;
  /** QA prompt token per compartment per attribute */
  qaTokens: number[][];
  /** Translation marker (shared) */
  trToken: number;

  /** Generate a bio paragraph for an entity in a compartment */
  generateBio(entityId: number, comp: number): number[];
  /** Generate a QA pair: [prompt tokens] and target value token(s) */
  generateQA(entityId: number, attrId: number, comp: number): { prompt: number[]; target: number[] };
  /** Generate a translation pair between two compartments for a fact */
  generateTranslation(entityId: number, attrId: number, compA: number, compB: number, mode?: TranslationMode): number[];
  /** Generate a training batch: 50% bios, remainder QA + translation per translationFrac (0..0.5) */
  generateTrainingBatch(batchSize: number, seqLen: number, translationFrac: number, translationMode?: TranslationMode): {
    tokens: Uint32Array; // [batchSize, seqLen]
    targets: Int32Array;  // [batchSize, seqLen] (-1 = no loss)
    comps: Uint8Array;    // [batchSize] which compartment
  };
  /** Reseed the internal batch-sampling RNG (for reproducible training runs). */
  reseedBatches(batchSeed: number): void;
  /** Generate eval QA batch: same entities, one tokens/targets slab per compartment. */
  generateEvalBatch(entityIds: number[], attrId: number): {
    /** `tokens[c]` is [n*seqLen] for compartment c */
    tokens: Uint32Array[];
    /** `targets[c]` is [n*targetLen] for compartment c */
    targets: Uint32Array[];
    seqLen: number;
    promptLen: number;
    targetLen: number;
  };
};

/** LCG-based deterministic rng. Returns a function that yields values in [0, 1). */
export function seededRandom(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

export function createBioWorld(config: BioConfig, seed = 42): BioWorld {
  const { nEntities, nAttributes, nValues, nCompartments, tokensPerEntity, tokensPerValue, bankSize } = config;

  // Validate constraints
  if (tokensPerEntity === 1 && bankSize < nEntities) {
    throw new Error(`bankSize (${bankSize}) must be >= nEntities (${nEntities}) when tokensPerEntity=1`);
  }
  if (tokensPerValue === 1 && bankSize < nValues) {
    throw new Error(`bankSize (${bankSize}) must be >= nValues (${nValues}) when tokensPerValue=1`);
  }

  // Seeded PRNG for world construction (token assignment, facts).
  const rng = seededRandom(seed);
  function rngInt(n: number) { return Math.floor(rng() * n); }

  // Separate RNG for training-batch sampling. Defaults to the same seed as
  // world construction but can be reseeded via reseedBatches() to get
  // reproducible training runs independent of world seeding.
  let batchRng: () => number = seededRandom(seed);

  // --- Token ID allocation ---
  // Layout: [compartment 0 bank] [compartment 1 bank] ... [shared tokens]
  // Each compartment gets `bankSize` base tokens for multi-token sequences,
  // plus nAttributes attr template tokens and nAttributes QA tokens.
  // Shared (structural) tokens live past all compartments: TR + SEP.
  const specialPerComp = nAttributes + nAttributes; // attr templates + qa prompts
  const tokensPerComp = bankSize + specialPerComp;
  const sharedBase = nCompartments * tokensPerComp;
  const trTokenId = sharedBase;       // translation marker (shared)
  const sepTokenId = sharedBase + 1;  // tuple separator (shared)
  const vocabSize = sharedBase + 2;

  function compBankStart(c: number) { return c * tokensPerComp; }
  function compSpecialStart(c: number) { return c * tokensPerComp + bankSize; }

  // --- Generate entity/value token sequences from bank ---
  function sampleTokenSeq(comp: number, length: number, usedSet: Set<string>): number[] {
    const base = compBankStart(comp);
    for (let attempt = 0; attempt < 1000; attempt++) {
      const seq: number[] = [];
      for (let i = 0; i < length; i++) {
        seq.push(base + rngInt(bankSize));
      }
      const key = seq.join(',');
      if (!usedSet.has(key)) {
        usedSet.add(key);
        return seq;
      }
    }
    throw new Error('Failed to generate unique token sequence');
  }

  // For single-token mode, use sequential assignment (guaranteed unique)
  function assignSingleTokens(comp: number, count: number, offset: number): number[][] {
    const base = compBankStart(comp);
    const result: number[][] = [];
    for (let i = 0; i < count; i++) {
      result.push([base + offset + i]);
    }
    return result;
  }

  // --- Assign tokens per compartment ---
  const entityTokens: number[][][] = [];
  const valueTokens: number[][][] = [];
  const attrTokens: number[][] = [];
  const qaTokens: number[][] = [];

  for (let c = 0; c < nCompartments; c++) {
    const specBase = compSpecialStart(c);

    // Attribute template tokens: one per attribute
    const at: number[] = [];
    for (let a = 0; a < nAttributes; a++) at.push(specBase + a);
    attrTokens.push(at);

    // QA prompt tokens: one per attribute
    const qa: number[] = [];
    for (let a = 0; a < nAttributes; a++) qa.push(specBase + nAttributes + a);
    qaTokens.push(qa);

    // Entity tokens
    if (tokensPerEntity === 1) {
      entityTokens.push(assignSingleTokens(c, nEntities, 0));
    } else {
      const used = new Set<string>();
      const ets: number[][] = [];
      for (let i = 0; i < nEntities; i++) {
        ets.push(sampleTokenSeq(c, tokensPerEntity, used));
      }
      entityTokens.push(ets);
    }

    // Value tokens
    if (tokensPerValue === 1) {
      // Offset past entity tokens to avoid collision
      const valOffset = tokensPerEntity === 1 ? nEntities : 0;
      valueTokens.push(assignSingleTokens(c, nValues, valOffset));
    } else {
      const used = new Set<string>();
      const vts: number[][] = [];
      for (let i = 0; i < nValues; i++) {
        vts.push(sampleTokenSeq(c, tokensPerValue, used));
      }
      valueTokens.push(vts);
    }
  }

  // --- Generate ground-truth facts ---
  const facts: number[][] = [];
  for (let e = 0; e < nEntities; e++) {
    const entityFacts: number[] = [];
    for (let a = 0; a < nAttributes; a++) {
      entityFacts.push(rngInt(nValues));
    }
    facts.push(entityFacts);
  }

  // --- Sequence generators ---

  function generateBio(entityId: number, comp: number): number[] {
    const seq: number[] = [];
    // Shuffle attribute order
    const order = Array.from({ length: nAttributes }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(batchRng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const mix = config.mixCompartments === true && nCompartments > 1;
    for (const attrId of order) {
      const valueId = facts[entityId][attrId];
      // Each tuple is compartment-coherent (entity/attr/value/sep share c),
      // but when mixCompartments is on, each tuple samples its own c.
      const tupleComp = mix ? Math.floor(batchRng() * nCompartments) : comp;
      seq.push(...entityTokens[tupleComp][entityId]);
      seq.push(attrTokens[tupleComp][attrId]);
      seq.push(...valueTokens[tupleComp][valueId]);
      seq.push(sepTokenId);
    }
    return seq;
  }

  function generateQA(entityId: number, attrId: number, comp: number): { prompt: number[]; target: number[] } {
    const valueId = facts[entityId][attrId];
    const prompt = [...entityTokens[comp][entityId], qaTokens[comp][attrId]];
    const target = [...valueTokens[comp][valueId]];
    return { prompt, target };
  }

  function generateTranslation(
    entityId: number, attrId: number, compA: number, compB: number,
    mode: TranslationMode = 'mirror',
  ): number[] {
    const valueId = facts[entityId][attrId];
    const partA = [...entityTokens[compA][entityId], attrTokens[compA][attrId], ...valueTokens[compA][valueId]];
    const partB = [...entityTokens[compB][entityId], attrTokens[compB][attrId], ...valueTokens[compB][valueId]];

    if (mode === 'mirror') {
      return [trTokenId, ...partA, trTokenId, ...partB];
    }
    if (mode === 'continuation') {
      // Random split point in [1, partA.length-1]: switch mid-fact to compartment B.
      const splitIdx = 1 + Math.floor(batchRng() * (partA.length - 1));
      return [...partA.slice(0, splitIdx), trTokenId, ...partB.slice(splitIdx)];
    }
    // 'dictionary': single token-pair translation, no fact context.
    const pairKind = Math.floor(batchRng() * 3);
    if (pairKind === 0) {
      return [trTokenId, ...entityTokens[compA][entityId], ...entityTokens[compB][entityId]];
    } else if (pairKind === 1) {
      return [trTokenId, attrTokens[compA][attrId], attrTokens[compB][attrId]];
    } else {
      return [trTokenId, ...valueTokens[compA][valueId], ...valueTokens[compB][valueId]];
    }
  }

  function generateTrainingBatch(
    batchSize: number, seqLen: number,
    translationFrac: number,
    translationMode: TranslationMode = 'mirror',
  ): {
    tokens: Uint32Array; targets: Int32Array; comps: Uint8Array;
  } {
    // Mix: translation takes `t` of the batch; non-translation is split 70/30
    // between bio and QA (matching the Physics of Language Models convention).
    //   bio   = 0.7 * (1 − t)
    //   qa    = 0.3 * (1 − t)
    //   trans = t
    const tFrac = Math.max(0, Math.min(1, translationFrac));
    const bioThreshold = 0.7 * (1 - tFrac);
    const qaThreshold = (1 - tFrac); // bio+qa share

    const tokens = new Uint32Array(batchSize * seqLen);
    const targets = new Int32Array(batchSize * seqLen).fill(-1); // -1 = no loss
    const comps = new Uint8Array(batchSize);

    for (let b = 0; b < batchSize; b++) {
      const comp = Math.floor(batchRng() * nCompartments);
      comps[b] = comp;
      const r = batchRng();

      let seq: number[];
      let tgt: Int32Array;

      if (r < bioThreshold) {
        // Bio paragraph
        const entityId = Math.floor(batchRng() * nEntities);
        seq = generateBio(entityId, comp);
        // Next-token prediction on the whole bio
        tgt = new Int32Array(seq.length).fill(-1);
        for (let i = 0; i < seq.length - 1; i++) tgt[i] = seq[i + 1];
      } else if (r < qaThreshold) {
        // QA
        const entityId = Math.floor(batchRng() * nEntities);
        const attrId = Math.floor(batchRng() * nAttributes);
        const qa = generateQA(entityId, attrId, comp);
        seq = [...qa.prompt, ...qa.target];
        tgt = new Int32Array(seq.length).fill(-1);
        // Loss only on the value tokens (after prompt)
        for (let i = qa.prompt.length; i < seq.length; i++) {
          tgt[i - 1] = seq[i]; // predict token i from position i-1
        }
      } else {
        // Translation
        const entityId = Math.floor(batchRng() * nEntities);
        const attrId = Math.floor(batchRng() * nAttributes);
        // Pick any compartment other than `comp` uniformly at random.
        let compB = comp;
        if (nCompartments > 1) {
          const offset = 1 + Math.floor(batchRng() * (nCompartments - 1));
          compB = (comp + offset) % nCompartments;
        }
        seq = generateTranslation(entityId, attrId, comp, compB, translationMode);
        tgt = new Int32Array(seq.length).fill(-1);
        for (let i = 0; i < seq.length - 1; i++) tgt[i] = seq[i + 1];
      }

      // Pad or truncate to seqLen
      const len = Math.min(seq.length, seqLen);
      for (let i = 0; i < len; i++) {
        tokens[b * seqLen + i] = seq[i];
        if (i < tgt.length) targets[b * seqLen + i] = tgt[i];
      }
    }

    return { tokens, targets, comps };
  }

  function generateEvalBatch(entityIds: number[], attrId: number): {
    tokens: Uint32Array[]; targets: Uint32Array[]; seqLen: number;
    promptLen: number; targetLen: number;
  } {
    const n = entityIds.length;
    // Figure out promptLen/targetLen/maxLen from comp 0 (all comps have same shape).
    const probe = generateQA(entityIds[0] ?? 0, attrId, 0);
    const promptLen = probe.prompt.length;
    const targetLen = probe.target.length;
    const seqLen = promptLen + targetLen;

    const tokens: Uint32Array[] = [];
    const targets: Uint32Array[] = [];
    for (let c = 0; c < nCompartments; c++) {
      const tok = new Uint32Array(n * seqLen);
      const tgt = new Uint32Array(n * targetLen);
      for (let i = 0; i < n; i++) {
        const qa = generateQA(entityIds[i], attrId, c);
        for (let t = 0; t < promptLen; t++) tok[i * seqLen + t] = qa.prompt[t];
        for (let t = 0; t < targetLen; t++) {
          tok[i * seqLen + promptLen + t] = qa.target[t];
          tgt[i * targetLen + t] = qa.target[t];
        }
      }
      tokens.push(tok);
      targets.push(tgt);
    }
    return { tokens, targets, seqLen, promptLen, targetLen };
  }

  /**
   * Translation eval batch (mirror mode only, deterministic).
   *
   * For each directed compartment pair (compA, compB) with compA≠compB, builds
   * one example per entityId:
   *   [TR, partA_entity, partA_attr, partA_value, TR, partB_entity, partB_attr, partB_value]
   *
   * The first `promptLen` tokens (up to and including the second TR) are the
   * prompt; the remaining `targetLen` tokens are the translation targets.
   *
   * Caller computes CE/accuracy over the target positions only.
   */
  function generateTranslationEvalBatch(entityIds: number[], attrId: number): {
    tokens: Uint32Array[]; targets: Uint32Array[];
    pairs: Array<{ from: number; to: number }>;
    seqLen: number; promptLen: number; targetLen: number;
  } {
    const n = entityIds.length;
    // Probe shapes from comp 0→1 (all pairs have identical shape with mirror).
    const probe = generateTranslation(entityIds[0] ?? 0, attrId, 0, Math.min(1, nCompartments - 1), 'mirror');
    const seqLen = probe.length;
    // partA.length = tokensPerEntity + 1 + tokensPerValue; same for partB.
    const partALen = tokensPerEntity + 1 + tokensPerValue;
    const promptLen = 1 + partALen + 1;    // TR + partA + TR
    const targetLen = seqLen - promptLen;   // partB

    const tokens: Uint32Array[] = [];
    const targets: Uint32Array[] = [];
    const pairs: Array<{ from: number; to: number }> = [];
    for (let a = 0; a < nCompartments; a++) {
      for (let b = 0; b < nCompartments; b++) {
        if (a === b) continue;
        const tok = new Uint32Array(n * seqLen);
        const tgt = new Uint32Array(n * targetLen);
        for (let i = 0; i < n; i++) {
          const seq = generateTranslation(entityIds[i], attrId, a, b, 'mirror');
          for (let t = 0; t < seqLen; t++) tok[i * seqLen + t] = seq[t];
          for (let t = 0; t < targetLen; t++) tgt[i * targetLen + t] = seq[promptLen + t];
        }
        tokens.push(tok); targets.push(tgt);
        pairs.push({ from: a, to: b });
      }
    }
    return { tokens, targets, pairs, seqLen, promptLen, targetLen };
  }

  function reseedBatches(batchSeed: number) {
    batchRng = seededRandom(batchSeed);
  }

  return {
    config, vocabSize, facts,
    entityTokens, attrTokens, valueTokens, sepToken: sepTokenId, qaTokens,
    trToken: trTokenId,
    generateBio, generateQA, generateTranslation,
    generateTrainingBatch, generateEvalBatch, generateTranslationEvalBatch,
    reseedBatches,
  };
}
