/**
 * Smoke test for WebGPUGPT2Trainer.
 *
 * Verifies that the Trainer-interface adapter wired against a tiny real
 * GPT-2 model runs end-to-end: initialize, setAnchor, innerSteps,
 * pseudograd, applyOuterStep, revertToAnchor, snapshotAnchor, applyF16W,
 * resetOptimState. We're not asserting convergence — only that the
 * pieces compose without crashing and that anchor/params move
 * consistently.
 *
 * Tiny model: 2 layers, 2 heads, 64 embed dim, vocab 256, batch=1,
 * seq=32, 2 inner steps per "round." Runs in a few seconds.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../../src/backend/webgpu/index";
import { WebGPUGPT2Trainer } from "../../src/distributed/protocol/webgpu-gpt2-trainer.ts";
import type { TokenSource } from "../../src/distributed/protocol/webgpu-gpt2-trainer.ts";
import { Torchlette } from "../../src/frontend/torchlette";
import { cpuOnly } from "../helpers/webgpu";

class RandomTokenSource implements TokenSource {
  constructor(private readonly vocab: number) {}
  async fetch(minTokens: number): Promise<number[]> {
    const n = Math.max(minTokens, 1024);
    const out = new Array<number>(n);
    for (let i = 0; i < n; i++) out[i] = Math.floor(Math.random() * this.vocab);
    return out;
  }
}

function l2(a: Float32Array[], b: Float32Array[]): number {
  let s = 0;
  for (let t = 0; t < a.length; t++) {
    for (let i = 0; i < a[t].length; i++) {
      const d = a[t][i] - b[t][i];
      s += d * d;
    }
  }
  return Math.sqrt(s);
}

describe.skipIf(cpuOnly)("WebGPUGPT2Trainer adapter", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU not available");
  });

  it(
    "initialize → setAnchor → inner → pseudograd → outer → revert → F16W cycle",
    async () => {
      const api = new Torchlette("webgpu", { enableFusion: true });
      api.manualSeed(42);
      const trainer = new WebGPUGPT2Trainer({
        api,
        modelConfig: {
          vocabSize: 256,
          blockSize: 64,
          numLayers: 2,
          numHeads: 2,
          embedDim: 64,
          dropoutRate: 0,
        },
        tokenSource: new RandomTokenSource(256),
        innerLr: 1e-3,
        outerLr: 0.7,
        outerMu: 0.9,
        innerSteps: 2,
        batchSize: 1,
        seqLen: 32,
        accumSteps: 1,
        weightDecay: 0,
        fullFinetuning: true,
        checkpointing: true,
      });

      await trainer.initialize();
      expect(trainer.totalParamCount()).toBeGreaterThan(0);

      // Snapshot the initial params as anchor — pseudograd should now be 0.
      await trainer.setAnchor();
      const grad0 = await trainer.pseudograd();
      const zerosLike = grad0.map((g) => new Float32Array(g.length));
      expect(l2(grad0, zerosLike)).toBeCloseTo(0, 5);

      // One round of inner training should leave a nonzero pseudograd.
      const avgLoss = await trainer.innerSteps(0);
      expect(Number.isFinite(avgLoss)).toBe(true);
      const grad1 = await trainer.pseudograd();
      expect(l2(grad1, zerosLike)).toBeGreaterThan(0);

      // Apply an outer step using our own grad as the "average."
      const anchorBefore = await trainer.snapshotAnchor();
      await trainer.applyOuterStep(grad1);
      const anchorAfter = await trainer.snapshotAnchor();
      // Outer step moved the anchor (because the grad was nonzero).
      expect(l2(anchorBefore, anchorAfter)).toBeGreaterThan(0);
      // And pseudograd is back to zero (params == anchor after applyOuterStep).
      const gradAfter = await trainer.pseudograd();
      expect(l2(gradAfter, zerosLike)).toBeLessThan(1e-3);

      // revertToAnchor is a no-op right after applyOuterStep (params already
      // match anchor); train another step then revert and verify reset.
      await trainer.innerSteps(1);
      const driftedGrad = await trainer.pseudograd();
      expect(l2(driftedGrad, zerosLike)).toBeGreaterThan(0);
      await trainer.revertToAnchor();
      const revertedGrad = await trainer.pseudograd();
      expect(l2(revertedGrad, zerosLike)).toBeLessThan(1e-3);

      // F16W: take a snapshot, scramble params with another inner step,
      // then re-apply the snapshot and verify we're back at the snapshot.
      const f16wPayload = await trainer.snapshotAnchor();
      await trainer.innerSteps(2);
      await trainer.applyF16W(f16wPayload);
      const postF16W = await trainer.snapshotAnchor();
      expect(l2(postF16W, f16wPayload)).toBeCloseTo(0, 5);

      // resetOptimState shouldn't crash.
      await trainer.resetOptimState();

      // Cleanup: Dawn needs explicit exit; we leave that to test framework.
    },
    60_000,
  );
});
