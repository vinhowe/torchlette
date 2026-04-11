/**
 * Fused RNN forward kernel via tile-IR.
 *
 * One dispatch does the full RNN forward for all batch elements × all timesteps.
 * Each thread handles one batch element, loops over time sequentially.
 *
 * h_t = tanh(E[token_t] + W_hh @ h_{t-1} + b_hh)
 * logits_t = W_out @ h_t + b_out
 *
 * Bindings:
 *   tokens:  [batch, seqLen] i32
 *   embed:   [vocabSize, hiddenSize] f32  (embedding matrix)
 *   w_hh:    [hiddenSize, hiddenSize] f32
 *   b_hh:    [hiddenSize] f32
 *   w_out:   [vocabSize, hiddenSize] f32
 *   b_out:   [vocabSize] f32
 *   hiddens: [batch, seqLen, hiddenSize] f32  (output)
 *   logits:  [batch, seqLen, vocabSize] f32   (output)
 *
 * Uniforms: batch, seqLen, hiddenSize, vocabSize
 */

import type { TileKernelSpec } from "../../src/backend/webgpu/tile-ir";

export function makeRNNForwardSpec(
  hiddenSize: number,
  vocabSize: number,
): TileKernelSpec {
  return {
    name: "rnn_forward",
    workgroupSize: 64,
    bindings: {
      tokens:  { type: "i32", access: "read" },
      embed:   { type: "f32", access: "read" },
      w_hh:    { type: "f32", access: "read" },
      b_hh:    { type: "f32", access: "read" },
      w_out:   { type: "f32", access: "read" },
      b_out:   { type: "f32", access: "read" },
      hiddens: { type: "f32", access: "write" },
      logits:  { type: "f32", access: "write" },
    },
    uniforms: {
      batch: "u32",
      seqLen: "u32",
    },
    grid: (uniforms) => {
      const batch = uniforms.batch;
      return [Math.ceil(batch / 64), 1, 1];
    },
    kernel(ctx) {
      const H = hiddenSize;
      const V = vocabSize;

      const batchId = ctx.globalId(0);
      const batch = ctx.uniform("batch");
      const seqLen = ctx.uniform("seqLen");

      // Early exit for out-of-bounds threads
      ctx.ifThen(batchId.gte(batch), () => {
        ctx.emitReturn();
      });

      // Mutable hidden state in registers (H vars)
      const hVars: ReturnType<typeof ctx.emitVar>[] = [];
      for (let i = 0; i < H; i++) {
        hVars.push(ctx.emitVar(`h${i}`, "f32", ctx.f32(0)));
      }

      // Loop over timesteps
      ctx.forRange(ctx.u32(0), seqLen, (t) => {
        // Load token: tokens[batchId * seqLen + t]
        const tokIdx = batchId.mul(seqLen).add(t);
        const tok = ctx.load("tokens", tokIdx);

        // Compute new hidden state: h_new[i] = tanh(embed[tok*H+i] + sum_j(w_hh[i*H+j] * h[j]) + b_hh[i])
        const hNewVars: ReturnType<typeof ctx.emitVar>[] = [];
        for (let i = 0; i < H; i++) {
          // Start with embedding + bias
          const embIdx = tok.cast("u32").mul(ctx.u32(H)).add(ctx.u32(i));
          let acc = ctx.load("embed", embIdx).add(ctx.load("b_hh", ctx.u32(i)));

          // W_hh @ h: dot product of row i with hidden state
          for (let j = 0; j < H; j++) {
            const wIdx = ctx.u32(i * H + j);
            acc = acc.add(ctx.load("w_hh", wIdx).mul(hVars[j].expr()));
          }

          hNewVars.push(ctx.emitVar(`hn${i}`, "f32", acc.tanh()));
        }

        // Update hidden state vars
        for (let i = 0; i < H; i++) {
          hVars[i].set(hNewVars[i].expr());
        }

        // Store hidden state: hiddens[(batchId * seqLen + t) * H + i]
        const hBaseIdx = batchId.mul(seqLen).add(t).mul(ctx.u32(H));
        for (let i = 0; i < H; i++) {
          ctx.emitStore("hiddens", hBaseIdx.add(ctx.u32(i)), hVars[i].expr());
        }

        // Compute and store logits: logits[(batchId * seqLen + t) * V + i]
        const lBaseIdx = batchId.mul(seqLen).add(t).mul(ctx.u32(V));
        for (let i = 0; i < V; i++) {
          let logit = ctx.load("b_out", ctx.u32(i));
          for (let j = 0; j < H; j++) {
            const wIdx = ctx.u32(i * H + j);
            logit = logit.add(ctx.load("w_out", wIdx).mul(hVars[j].expr()));
          }
          ctx.emitStore("logits", lBaseIdx.add(ctx.u32(i)), logit);
        }
      });
    },
  };
}
