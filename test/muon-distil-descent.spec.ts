/**
 * Crystal-3 TAIL — the Muon OPTIMIZER (realized from MUON_PROGRAM alone: the
 * contraction node + unrolled Newton-Schulz, no hand kernel, no hand grads)
 * trains DistilGPT-2 end-to-end.
 *
 * Loads pretrained DistilGPT-2, runs 20 full-finetune steps on a fixed sequence
 * with Muon (2D weight matrices orthogonalized; embeddings + 1D params on the
 * internal AdamW), and asserts SANE MONOTONE-ish DESCENT: every loss finite and
 * the loss falls substantially over the trajectory. That a whole optimizer
 * reaching ABOVE the elementwise algebra (into the `mm` contraction) needed no
 * new engine is the payoff. (The numeric + orthogonality cross-checks vs a JS
 * reference live in test/semantic-optimizer.spec.ts.)
 */

import * as path from "node:path";
import { beforeAll, describe, expect, test } from "vitest";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import type { GPT2 } from "../examples/gpt2/model";
import { Torchlette } from "../src/frontend/torchlette";
import { Muon } from "../src/optim";

const FIXED_TOKENS: number[] = [
  2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30, 198, 1986, 280,
  1242, 517, 8855, 290, 517, 29815, 13, 198, 49, 619, 9985, 466, 13508, 262,
  38482, 31007, 286, 1737,
];

const NUM_STEPS = 20;

describe(
  "Muon trains DistilGPT-2 (Crystal-3 contraction dividend)",
  { timeout: 300_000 },
  () => {
    let webgpuAvailable = false;
    let api: Torchlette;
    let model: GPT2;

    beforeAll(async () => {
      const { canUseWebGPU } = await import("./helpers/webgpu");
      if (!(await canUseWebGPU())) {
        console.warn("WebGPU not available — Muon distil test skipped");
        return;
      }
      webgpuAvailable = true;
      api = new Torchlette("webgpu", {
        enableFusion: true,
        enableMemoryPlanning: true,
        enableCheckpointSegmentation: true,
      });
      const modelDir = path.join(process.cwd(), "models", "distilgpt2");
      model = await loadPretrainedGPT2(
        api,
        modelDir,
        { dropoutRate: 0.0 },
        { device: "webgpu" },
      );
    }, 120_000);

    test("20 Muon steps: sane descent, all finite", async () => {
      if (!webgpuAvailable) {
        console.log("Skipping: WebGPU not available");
        return;
      }
      model.train();
      // Muon: EVERY 2D weight matrix (attention + MLP) orthogonalized; the
      // embeddings + 1D params route to the internal AdamW (standard Muon
      // practice — orthogonalization needs a weight matrix). The `evalOptTerm`
      // memoization keeps each param's Newton-Schulz chain LINEAR in the step
      // count (the shared X/A subterms realize once), so the full 2D surface
      // fits comfortably.
      const optimizer = new Muon(model.parameters(), {
        lr: 2e-3,
        momentum: 0.95,
        weightDecay: 0.0,
        nsSteps: 5,
        adamwParams: [model.wte.weight, model.wpe.weight],
        adamwLr: 3e-4,
      });
      // The 2D weight matrices (4 per layer × 6 layers) ARE being orthogonalized.
      expect(optimizer.numMuonParams).toBe(24);

      const inputData = FIXED_TOKENS.slice(0, -1);
      const targetData = FIXED_TOKENS.slice(1);
      const losses: number[] = [];

      for (let step = 0; step < NUM_STEPS; step++) {
        await api.beginStep();
        const input = api.tensorFromArray(inputData, [1, inputData.length], {
          device: "webgpu",
        });
        const target = api.tensorFromArray(targetData, [1, targetData.length], {
          device: "webgpu",
        });
        const result = model.forwardWithLoss(input, target, {
          useCheckpoint: true,
        });
        if (!result.loss) throw new Error("Loss is null");
        const lossValue = await result.loss.item();
        await result.loss.backward();
        optimizer.step();
        optimizer.zeroGrad();
        result.loss.dispose();
        input.dispose();
        target.dispose();
        api.endStep();
        await api.markStep();
        losses.push(lossValue);
      }

      console.log(
        `\nMuon distil losses: [${losses.map((l) => l.toFixed(4)).join(", ")}]`,
      );

      for (const l of losses) expect(Number.isFinite(l)).toBe(true);
      // Sane descent: the loss should fall meaningfully over 20 steps.
      expect(losses[NUM_STEPS - 1]).toBeLessThan(losses[0]);
      expect(losses[NUM_STEPS - 1]).toBeLessThan(losses[0] - 0.3);
    });
  },
);
