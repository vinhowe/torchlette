/**
 * CPU reference arm for the flagship training loop (task #94, item 1).
 *
 * The CPU backend is torchlette's GPU-vs-CPU differential reference — a house
 * correctness instrument (CLAUDE.md "differentially test optimized paths against
 * the naive one"). It was silently broken for the CANONICAL loop
 * (autocast + checkpoint + GradScaler + clipGradNorm_ + AdamW +
 * CosineAnnealingLR — the tools/t-fp-spike-probe.ts loop), by two webgpu-era
 * mechanisms leaking onto the CPU path:
 *
 *   (a) the memory-planner `captureActionLayouts` crash — CPU backend tensors
 *       carry no `.buffer.size`; that capture is consumed ONLY by the webgpu
 *       stream generator, so it now early-returns for non-webgpu backends; and
 *   (b) fusion (epilogue/prologue/row-program directives + fused kernels) being
 *       built on CPU because the caller requested enableFusion:true — the
 *       backend's capability (isFusedBackend), not the request, now gates it, so
 *       CPU cleanly takes the sequential path.
 *
 * This gate runs the FULL canonical loop on CPU for a few steps against a tiny
 * from-scratch GPT-2 and asserts it runs end-to-end (finite, descending loss) —
 * a broken reference arm is the real damage, so it must never silently rot.
 *
 * The un-awaited `scaler.update()` "Engine is busy" re-entrancy (caveat item
 * 1b) is a USAGE bug, not a framework bug: on the CPU/elementwise path update()
 * holds the exec lock across an item() readback (the fused path defers it), so a
 * missing await overlaps the next item(). We await it here, as the API
 * documents; the fp-spike probe was fixed the same way.
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { Adam, CosineAnnealingLR, GradScaler } from "../src/optim/index";
import { clipGradNorm_ } from "../src/nn/index";

const TIMEOUT = 120_000;

// Tiny from-scratch GPT-2 — small enough to run many CPU steps fast, but it
// exercises embeddings, attention, layernorm, MLP, weight-tied lm_head and the
// cross-entropy loss (the whole forward/backward the reference arm must cover).
const TINY_CONFIG: GPT2Config = {
  vocabSize: 128,
  blockSize: 32,
  numLayers: 2,
  numHeads: 2,
  embedDim: 16,
  dropoutRate: 0,
};

describe("CPU canonical training loop (reference arm, task #94)", () => {
  it(
    "runs autocast+checkpoint+GradScaler+clip+AdamW+CosineAnnealingLR end-to-end on CPU with descending loss",
    async () => {
      const STEPS = 3;
      const BATCH = 2;
      const SEQ = 16;
      const api = new Torchlette("cpu", {
        // Deliberately request fusion + planner: the fix is that these are
        // capability-gated, so requesting them on CPU must NOT crash (the
        // exact configuration the fp-spike probe passes).
        enableFusion: true,
        enableMemoryPlanning: true,
        enableCheckpointSegmentation: true,
      });

      const model = new GPT2(api, TINY_CONFIG, { device: "cpu" });
      model.train(true);
      const params = model.parameters();
      const opt = new Adam(
        params,
        { lr: 1e-3, weightDecay: 0.01, adamW: true },
        api,
      );
      const sched = new CosineAnnealingLR(opt, STEPS, 1e-5);
      const scaler = new GradScaler(api, { initScale: 1024.0 });

      const V = TINY_CONFIG.vocabSize;
      let seed = 1234;
      const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
      const inp = new Int32Array(BATCH * SEQ);
      const tgt = new Int32Array(BATCH * SEQ);

      const losses: number[] = [];
      for (let stp = 0; stp < STEPS; stp++) {
        await scaler.resolveDeferred();
        for (let i = 0; i < BATCH * SEQ; i++) {
          inp[i] = rnd() % V;
          tgt[i] = rnd() % V;
        }
        const input = api.tensorFromArray(Array.from(inp), [BATCH, SEQ], {
          device: "cpu",
        });
        const target = api.tensorFromArray(Array.from(tgt), [BATCH, SEQ], {
          device: "cpu",
        });

        const l = api.tidy(() => {
          const ll = api.autocast(
            () =>
              model.forwardWithLoss(input, target, { useCheckpoint: true })
                .loss!,
          );
          api.keep(ll);
          return ll;
        });
        const lossOut = api.noGrad(() => api.mul(l, 1));
        const scaled = scaler.scale(l);
        await scaled.backward();
        scaler.unscale_(opt);
        clipGradNorm_(api, params, 1.0);
        scaler.step(opt);
        await scaler.update(); // async — see file header (item 1b)
        opt.zeroGrad();
        scaled.dispose();
        const lossVal = await lossOut.item();
        losses.push(lossVal);
        lossOut.dispose();
        input.dispose();
        target.dispose();
        sched.step();
      }

      // Reference-arm acceptance: every step finite (no crash, no NaN/Inf from a
      // webgpu-era path leaking onto CPU) and the loop actually optimizes.
      for (const v of losses) expect(Number.isFinite(v)).toBe(true);
      expect(losses[losses.length - 1]).toBeLessThan(losses[0]);
    },
    TIMEOUT,
  );
});
