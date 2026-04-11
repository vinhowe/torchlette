/**
 * Tiny 3-hidden-unit Elman RNN for MESS3 belief state tracking.
 *
 * h_t = tanh(E[token_t] + W_hh @ h_{t-1} + b_h)
 * logits_t = W_out @ h_t + b_out
 *
 * Hidden state is 3D — directly plottable on the belief simplex.
 * Training: unrolled BPTT through torchlette's autograd (GPU).
 * Eval: raw JS CPU forward for instant simplex visualization.
 */

export type RNNConfig = {
  vocabSize: number;
  hiddenSize: number;
};

export const MESS3_RNN_CONFIG: RNNConfig = {
  vocabSize: 3,
  hiddenSize: 3,
};

export function createRNN(api: any, nn: any, config: RNNConfig) {
  const { vocabSize, hiddenSize } = config;

  const embed = new nn.Embedding(api, vocabSize, hiddenSize);
  const wHH = new nn.Linear(api, hiddenSize, hiddenSize);
  const wOut = new nn.Linear(api, hiddenSize, vocabSize);

  function parameters(): any[] {
    return [...embed.parameters(), ...wHH.parameters(), ...wOut.parameters()];
  }

  /** GPU forward for training (autograd-enabled, unrolled BPTT) */
  function forward(tokens: any, seqLen: number): { logitsList: any[]; allH: any[] } {
    const [batch] = tokens.shape;
    let h = api.zeros([batch, hiddenSize]);
    const logitsList: any[] = [];
    const allH: any[] = [];

    for (let t = 0; t < seqLen; t++) {
      const tokT = tokens.narrow(1, t, 1).reshape([batch]);
      const xEmb = embed.forward(tokT);
      const hhOut = wHH.forward(h);
      h = api.tanh(api.add(xEmb, hhOut));
      allH.push(h);
      logitsList.push(wOut.forward(h));
    }

    return { logitsList, allH };
  }

  /**
   * CPU forward for eval — instant, no GPU overhead.
   * Reads current weights from GPU, then runs the full RNN in JS.
   * Returns raw hidden states for simplex plotting.
   */
  async function forwardCPU(
    tokens: Uint32Array, batch: number, seqLen: number,
  ): Promise<Float32Array> {
    // Read weights to CPU
    const E = await embed.weight.cpu();       // [vocabSize, hiddenSize]
    const Whh = await wHH.weight.cpu();       // [hiddenSize, hiddenSize]
    const bhh = await wHH.bias.cpu();         // [hiddenSize]
    const H = hiddenSize;

    // Output: [batch * seqLen * H]
    const hiddens = new Float32Array(batch * seqLen * H);

    for (let b = 0; b < batch; b++) {
      // Hidden state for this sequence
      const h = new Float32Array(H); // zeros

      for (let t = 0; t < seqLen; t++) {
        const tok = tokens[b * seqLen + t];

        // h_new[i] = tanh(E[tok][i] + sum_j(Whh[i][j] * h[j]) + bhh[i])
        const hNew = new Float32Array(H);
        for (let i = 0; i < H; i++) {
          let sum = E[tok * H + i] + bhh[i];
          for (let j = 0; j < H; j++) {
            sum += Whh[i * H + j] * h[j];
          }
          hNew[i] = Math.tanh(sum);
        }

        // Store and update
        const off = (b * seqLen + t) * H;
        for (let i = 0; i < H; i++) {
          hiddens[off + i] = hNew[i];
          h[i] = hNew[i];
        }
      }
    }

    return hiddens;
  }

  return { embed, wHH, wOut, parameters, forward, forwardCPU, config };
}

export type RNNModel = ReturnType<typeof createRNN>;
