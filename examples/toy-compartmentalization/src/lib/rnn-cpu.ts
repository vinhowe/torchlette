/**
 * Pure CPU RNN: forward, backward (BPTT), and Adam — all in raw JS.
 * Configurable hidden dimension H.
 *
 * Architecture:
 *   h_t = tanh(E[token_t] + W_hh @ h_{t-1} + b_hh)
 *   logits_t = W_out @ h_t + b_out
 *   loss = cross_entropy(logits, targets)
 */

export type CPURNNParams = {
  V: number;
  Vout: number;
  H: number;
  E: Float64Array;      // [V * H]
  Whh: Float64Array;    // [H * H]
  bhh: Float64Array;    // [H]
  Wout: Float64Array;   // [Vout * H]
  bout: Float64Array;   // [Vout]
};

export type AdamState = {
  m: Float64Array[];
  v: Float64Array[];
  t: number;
};

export function initParams(V = 5, Vout = 3, H = 3): CPURNNParams {
  const scale = 0.1;
  const rand = () => (Math.random() - 0.5) * 2 * scale;
  const fill = (n: number) => { const a = new Float64Array(n); for (let i = 0; i < n; i++) a[i] = rand(); return a; };
  return { V, Vout, H, E: fill(V * H), Whh: fill(H * H), bhh: new Float64Array(H), Wout: fill(Vout * H), bout: new Float64Array(Vout) };
}

function getParamArrays(p: CPURNNParams): Float64Array[] {
  return [p.E, p.Whh, p.bhh, p.Wout, p.bout];
}

export function initAdam(params: CPURNNParams): AdamState {
  const all = getParamArrays(params);
  return { m: all.map(a => new Float64Array(a.length)), v: all.map(a => new Float64Array(a.length)), t: 0 };
}

export function paramCount(p: CPURNNParams): number {
  return getParamArrays(p).reduce((s, a) => s + a.length, 0);
}

export function trainBatch(
  params: CPURNNParams,
  adamState: AdamState,
  tokens: Uint32Array,
  batchSize: number,
  seqLen: number,
  lr: number,
): { loss: number } {
  const { V, Vout, H, E, Whh, bhh, Wout, bout } = params;
  const nTargets = seqLen - 1;

  const dE = new Float64Array(V * H);
  const dWhh = new Float64Array(H * H);
  const dbhh = new Float64Array(H);
  const dWout = new Float64Array(Vout * H);
  const dbout = new Float64Array(Vout);

  const hBuf = new Float64Array(seqLen * H);
  const probsBuf = new Float64Array(seqLen * Vout);

  let totalLoss = 0;

  for (let b = 0; b < batchSize; b++) {
    const tokOff = b * seqLen;
    const hPrev = new Float64Array(H);

    // --- Forward ---
    for (let t = 0; t < seqLen; t++) {
      const tok = tokens[tokOff + t];
      const eOff = tok * H;
      const hOff = t * H;

      for (let i = 0; i < H; i++) {
        let s = E[eOff + i] + bhh[i];
        for (let j = 0; j < H; j++) s += Whh[i * H + j] * hPrev[j];
        hBuf[hOff + i] = Math.tanh(s);
      }
      for (let i = 0; i < H; i++) hPrev[i] = hBuf[hOff + i];

      const pOff = t * Vout;
      let maxL = -Infinity;
      for (let i = 0; i < Vout; i++) {
        let s = bout[i];
        for (let j = 0; j < H; j++) s += Wout[i * H + j] * hBuf[hOff + j];
        probsBuf[pOff + i] = s;
        if (s > maxL) maxL = s;
      }
      let sumExp = 0;
      for (let i = 0; i < Vout; i++) { probsBuf[pOff + i] = Math.exp(probsBuf[pOff + i] - maxL); sumExp += probsBuf[pOff + i]; }
      for (let i = 0; i < Vout; i++) probsBuf[pOff + i] /= sumExp;
    }

    // --- Loss ---
    let loss = 0;
    for (let t = 0; t < nTargets; t++) {
      const target = tokens[tokOff + t + 1];
      if (target < Vout) loss -= Math.log(probsBuf[t * Vout + target] + 1e-12);
    }
    totalLoss += loss / nTargets;

    // --- Backward ---
    const dhNext = new Float64Array(H);

    for (let t = nTargets - 1; t >= 0; t--) {
      const target = tokens[tokOff + t + 1];
      const hOff = t * H;
      const pOff = t * Vout;
      const tok = tokens[tokOff + t];
      const eOff = tok * H;
      const invN = 1 / nTargets;

      // dLogits
      const dl = new Float64Array(Vout);
      if (target < Vout) {
        for (let i = 0; i < Vout; i++) dl[i] = probsBuf[pOff + i] * invN;
        dl[target] -= invN;
      }

      // Grads for Wout, bout
      for (let i = 0; i < Vout; i++) {
        dbout[i] += dl[i];
        for (let j = 0; j < H; j++) dWout[i * H + j] += dl[i] * hBuf[hOff + j];
      }

      // dh from output + from next timestep
      const dh = new Float64Array(H);
      for (let j = 0; j < H; j++) {
        let s = dhNext[j];
        for (let i = 0; i < Vout; i++) s += Wout[i * H + j] * dl[i];
        dh[j] = s;
      }

      // Through tanh
      const dp = new Float64Array(H);
      for (let i = 0; i < H; i++) {
        const hi = hBuf[hOff + i];
        dp[i] = dh[i] * (1 - hi * hi);
      }

      // Grads for E, bhh, Whh
      for (let i = 0; i < H; i++) {
        dE[eOff + i] += dp[i];
        dbhh[i] += dp[i];
      }
      const prevOff = t > 0 ? (t - 1) * H : -1;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) {
          const ph = prevOff >= 0 ? hBuf[prevOff + j] : 0;
          dWhh[i * H + j] += dp[i] * ph;
        }
      }

      // dhNext = Whh^T @ dp
      for (let j = 0; j < H; j++) {
        let s = 0;
        for (let i = 0; i < H; i++) s += Whh[i * H + j] * dp[i];
        dhNext[j] = s;
      }
    }
  }

  // Adam
  const invB = 1 / batchSize;
  const grads = [dE, dWhh, dbhh, dWout, dbout];
  const all = getParamArrays(params);
  adamState.t++;
  const bc1 = 1 - 0.9 ** adamState.t;
  const bc2 = 1 - 0.999 ** adamState.t;
  for (let p = 0; p < all.length; p++) {
    const param = all[p], grad = grads[p], m = adamState.m[p], v = adamState.v[p];
    for (let i = 0; i < param.length; i++) {
      const g = grad[i] * invB;
      m[i] = 0.9 * m[i] + 0.1 * g;
      v[i] = 0.999 * v[i] + 0.001 * g * g;
      param[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + 1e-8);
    }
  }

  return { loss: totalLoss / batchSize };
}

export function evalForward(
  params: CPURNNParams,
  tokens: Uint32Array,
  batchSize: number,
  seqLen: number,
): Float64Array {
  const { E, Whh, bhh, H } = params;
  const hiddens = new Float64Array(batchSize * seqLen * H);
  for (let b = 0; b < batchSize; b++) {
    const h = new Float64Array(H);
    const tokOff = b * seqLen;
    for (let t = 0; t < seqLen; t++) {
      const tok = tokens[tokOff + t];
      const eOff = tok * H;
      for (let i = 0; i < H; i++) {
        let s = E[eOff + i] + bhh[i];
        for (let j = 0; j < H; j++) s += Whh[i * H + j] * h[j];
        h[i] = Math.tanh(s);
      }
      const off = (b * seqLen + t) * H;
      for (let i = 0; i < H; i++) hiddens[off + i] = h[i];
    }
  }
  return hiddens;
}
