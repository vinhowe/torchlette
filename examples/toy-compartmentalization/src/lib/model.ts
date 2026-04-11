/**
 * Tiny causal transformer for MESS3 belief state geometry experiments.
 *
 * 4 layers, 1 head, 64-dim, ReLU MLP (256), context 10.
 * Matches the architecture from "Transformers Represent Belief State Geometry
 * in their Residual Stream" (Shai et al., 2024).
 *
 * Exposes residual stream activations at each layer for belief state probing.
 */

export type ModelConfig = {
  vocabSize: number;
  seqLen: number;
  embedDim: number;
  numHeads: number;
  headDim?: number; // if set, overrides embedDim/numHeads
  numLayers: number;
  mlpDim: number;
  /** Positional encoding: learned wpe table (default) or RoPE applied to Q/K. */
  posEncoding?: 'learned' | 'rope';
  /** RoPE base (default 10000). */
  ropeBase?: number;
};

export const MESS3_CONFIG: ModelConfig = {
  vocabSize: 3, seqLen: 10, embedDim: 64, numHeads: 1,
  headDim: 8, numLayers: 4, mlpDim: 256,
};

export function createModel(api: any, nn: any, config: ModelConfig) {
  const { vocabSize, seqLen, embedDim, numHeads, numLayers, mlpDim } = config;
  const headDim = config.headDim ?? (embedDim / numHeads);
  const attnDim = numHeads * headDim; // total attention dimension (may be < embedDim)
  const useRoPE = config.posEncoding === 'rope';
  const ropeBase = config.ropeBase ?? 10000;
  if (useRoPE && headDim % 2 !== 0) {
    throw new Error(`RoPE requires even headDim, got ${headDim}`);
  }

  const wte = new nn.Embedding(api, vocabSize, embedDim);
  const wpe = useRoPE ? null : new nn.Embedding(api, seqLen, embedDim);

  // Precomputed RoPE cos/sin tables: shape [1, 1, seqLen, headDim/2].
  // Broadcasts against Q/K of shape [batch, numHeads, seqLen, headDim].
  let ropeCos: any = null;
  let ropeSin: any = null;
  if (useRoPE) {
    const half = headDim / 2;
    const cosData = new Float32Array(seqLen * half);
    const sinData = new Float32Array(seqLen * half);
    for (let m = 0; m < seqLen; m++) {
      for (let i = 0; i < half; i++) {
        const theta = m * Math.pow(ropeBase, -(2 * i) / headDim);
        cosData[m * half + i] = Math.cos(theta);
        sinData[m * half + i] = Math.sin(theta);
      }
    }
    ropeCos = api.tensorFromArray(cosData, [seqLen, half]);
    ropeSin = api.tensorFromArray(sinData, [seqLen, half]);
  }

  // RoPE is implemented as a single fused tile-IR kernel via api.applyRoPE.
  // Cached cos/sin have shape [seqLen, headDim/2].

  const layers: {
    ln1: any; qkv: any; outProj: any;
    ln2: any; fc1: any; fc2: any;
  }[] = [];

  for (let i = 0; i < numLayers; i++) {
    layers.push({
      ln1: new nn.LayerNorm(api, embedDim),
      qkv: new nn.Linear(api, embedDim, 3 * attnDim),
      outProj: new nn.Linear(api, attnDim, embedDim),
      ln2: new nn.LayerNorm(api, embedDim),
      fc1: new nn.Linear(api, embedDim, mlpDim),
      fc2: new nn.Linear(api, mlpDim, embedDim),
    });
  }

  const lnF = new nn.LayerNorm(api, embedDim);
  const lmHead = new nn.Linear(api, embedDim, vocabSize, { bias: false });

  const posIndices = useRoPE ? null : api.arange(seqLen).reshape([1, seqLen]);

  function parameters(): any[] {
    const params: any[] = [...wte.parameters()];
    if (wpe) params.push(...wpe.parameters());
    for (const layer of layers) {
      params.push(
        ...layer.ln1.parameters(), ...layer.qkv.parameters(), ...layer.outProj.parameters(),
        ...layer.ln2.parameters(), ...layer.fc1.parameters(), ...layer.fc2.parameters(),
      );
    }
    params.push(...lnF.parameters(), ...lmHead.parameters());
    return params;
  }

  /**
   * Forward pass. Returns logits and per-layer residual stream activations.
   * residuals[i] = residual stream AFTER layer i (before final LN).
   */
  function forward(tokens: any): { logits: any; residuals: any[] } {
    const [batch, sl] = tokens.shape;

    const tokEmb = wte.forward(tokens);
    let x = tokEmb;
    if (wpe && posIndices) {
      const posIdx = posIndices.narrow(1, 0, sl);
      const posEmb = wpe.forward(posIdx);
      x = api.add(tokEmb, posEmb);
    }

    const residuals: any[] = [];

    for (const layer of layers) {
      // Pre-norm attention
      const normed1 = layer.ln1.forward(x);
      const qkvOut = layer.qkv.forward(normed1);
      const [qFlat, kFlat, vFlat] = qkvOut.chunk(3, -1);
      const reshape = (t: any) =>
        t.reshape([batch, sl, numHeads, headDim])
         .permute([0, 2, 1, 3])
         .contiguous();
      let q = reshape(qFlat);
      let k = reshape(kFlat);
      const v = reshape(vFlat);
      if (useRoPE) {
        // Narrow cos/sin to current seq length, apply fused RoPE kernel.
        const cosCur = ropeCos.narrow(0, 0, sl);
        const sinCur = ropeSin.narrow(0, 0, sl);
        q = api.applyRoPE(q, cosCur, sinCur);
        k = api.applyRoPE(k, cosCur, sinCur);
      }

      const scale = 1.0 / Math.sqrt(headDim);
      const attnOut = api.scaledDotProductAttention(q, k, v, scale, true);
      const attnConcat = attnOut
        .permute([0, 2, 1, 3])
        .contiguous()
        .reshape([batch, sl, attnDim]);
      const attnProj = layer.outProj.forward(attnConcat);
      x = api.add(x, attnProj);

      // Pre-norm MLP with ReLU (paper uses ReLU, not GELU)
      const normed2 = layer.ln2.forward(x);
      const mlpHidden = api.relu(layer.fc1.forward(normed2));
      const mlpOut = layer.fc2.forward(mlpHidden);
      x = api.add(x, mlpOut);

      residuals.push(x);
    }

    const normedFinal = lnF.forward(x);
    const logits = lmHead.forward(normedFinal);

    return { logits, residuals };
  }

  /** All non-parameter persistent tensors (RoPE tables, pos indices). */
  function persistentTensors(): any[] {
    const out: any[] = [];
    if (ropeCos) out.push(ropeCos);
    if (ropeSin) out.push(ropeSin);
    if (posIndices) out.push(posIndices);
    return out;
  }

  return { wte, wpe, layers, lnF, lmHead, parameters, persistentTensors, forward, config };
}

export type MESS3Model = ReturnType<typeof createModel>;
