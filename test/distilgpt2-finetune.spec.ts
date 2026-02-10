/**
 * DistilGPT-2 Finetuning Regression Test
 *
 * Loads pretrained DistilGPT-2 weights, runs 5 training steps on a fixed
 * token sequence, and compares loss values against ground truth constants.
 * Reports per-step timing breakdown.
 *
 * Run with: npx vitest run test/distilgpt2-finetune.spec.ts
 * (WebGPU auto-detected; skip with TORCHLETTE_CPU_ONLY=1)
 */

import { describe, expect, test, beforeAll } from "vitest";
import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  getSubmitCount,
  resetSubmitCount,
  setProfilePhase,
  readGpuTimestamps,
  printProfileSummary,
  resetProfileStats,
  isProfilingEnabled,
  writeProfileJSON,
} from "../src/backend/webgpu";
import { GPT2 } from "../examples/gpt2/model";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";

// First 32 tokens of Shakespeare sonnets tokenized by GPT-2 tokenizer:
// "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate.\nRough"
const FIXED_TOKENS: number[] = [
  2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30,
  198, 1986, 280, 1242, 517, 8855, 290, 517, 29815, 13,
  198, 49, 619, 9985, 466, 13508, 262, 38482, 31007, 286, 1737,
];

const NUM_STEPS = 5;

// Ground truth losses — captured from initial run, tolerance ±0.05
const EXPECTED_LOSSES: number[] = [7.886621, 6.118543, 4.959344, 3.926172, 2.909729];

const LOSS_TOLERANCE = 0.05;

describe("DistilGPT-2 Finetuning Regression", { timeout: 300_000 }, () => {
  let webgpuAvailable = false;
  let api: Torchlette;
  let model: GPT2;

  beforeAll(async () => {
    const { canUseWebGPU } = await import("./helpers/webgpu");
    const success = await canUseWebGPU();
    webgpuAvailable = success;
    if (!success) {
      console.warn("WebGPU not available — tests will be skipped");
      return;
    }

    api = new Torchlette("webgpu", {
      enableFusion: true,
      enableMemoryPlanning: true,
      enableCheckpointSegmentation: true,
    });

    const modelDir = path.join(process.cwd(), "models", "distilgpt2");
    model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  }, 120_000);

  test("loss values match ground truth over 5 steps", async () => {
    if (!webgpuAvailable) {
      console.log("Skipping: WebGPU not available");
      return;
    }

    model.train();
    const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
    const scaler = new GradScaler(api, { initScale: 1024.0 });

    const compiledForward = api.compile((input: Tensor, target: Tensor) => {
      return api.autocast(() => {
        const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
        if (!result.loss) throw new Error("Loss is null");
        return result.loss;
      });
    });

    const inputData = FIXED_TOKENS.slice(0, -1); // 31 tokens
    const targetData = FIXED_TOKENS.slice(1); // 31 tokens

    const losses: number[] = [];
    const timings: { fwd: number; bwd: number; opt: number; cleanup: number; submits: number }[] = [];
    let lastCumulativeFusionStats: any = null;

    for (let step = 0; step < NUM_STEPS; step++) {
      await scaler.resolveDeferred();
      await api.beginStep();
      const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
      const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

      resetSubmitCount();
      resetProfileStats();
      setProfilePhase("forward");
      const t0 = performance.now();
      const loss = compiledForward(input, target);
      const lossValue = await loss.item();
      const t1 = performance.now();

      setProfilePhase("backward");
      const scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();
      const t2 = performance.now();

      setProfilePhase("optimizer");
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
      optimizer.zeroGrad();
      const t3 = performance.now();

      // Capture fusion stats before markStep() resets them
      const cumulativeStats = api._runtime().getCumulativeFusionStats();
      const lastStats = api._runtime().getLastFusionStats();
      lastCumulativeFusionStats = cumulativeStats;

      setProfilePhase("cleanup");
      scaledLoss.dispose();
      loss.dispose();
      input.dispose();
      target.dispose();
      api.endStep();
      await api.markStep();
      const t4 = performance.now();

      // Read GPU timestamps and print profiling summary
      if (isProfilingEnabled()) {
        await readGpuTimestamps();
        printProfileSummary(`step ${step}`);
        writeProfileJSON(`/tmp/torchlette-profile-step${step}.json`);
      }

      const stepSubmits = getSubmitCount();
      losses.push(lossValue);
      timings.push({ fwd: t1 - t0, bwd: t2 - t1, opt: t3 - t2, cleanup: t4 - t3, submits: stepSubmits });

      // Print fusion stats if available (cumulative captures all force() calls in a step)
      if (step === NUM_STEPS - 1) {
        if (cumulativeStats) {
          console.log(`\nCumulative fusion stats (step ${step}): ${JSON.stringify(cumulativeStats)}`);
        }
        if (lastStats) {
          console.log(`Last fusion stats (step ${step}): ${JSON.stringify(lastStats)}`);
        }
      }
    }

    // Print results table
    console.log("\n| Step | Loss     | Fwd(ms) | Bwd(ms) | Opt(ms) | Cleanup(ms) | Submits |");
    console.log("|------|----------|---------|---------|---------|-------------|---------|");
    for (let i = 0; i < NUM_STEPS; i++) {
      const t = timings[i];
      console.log(
        `| ${i}    | ${losses[i].toFixed(4)} | ${t.fwd.toFixed(0).padStart(7)} | ${t.bwd.toFixed(0).padStart(7)} | ${t.opt.toFixed(0).padStart(7)} | ${t.cleanup.toFixed(0).padStart(11)} | ${String(t.submits).padStart(7)} |`,
      );
    }

    // Print losses for capture mode
    console.log(`\nCaptured losses: [${losses.map((l) => l.toFixed(6)).join(", ")}]`);

    // Verify losses match ground truth (skip if in capture mode)
    if (EXPECTED_LOSSES.length === NUM_STEPS) {
      for (let i = 0; i < NUM_STEPS; i++) {
        expect(
          Math.abs(losses[i] - EXPECTED_LOSSES[i]),
          `Step ${i}: loss ${losses[i].toFixed(4)} differs from expected ${EXPECTED_LOSSES[i].toFixed(4)} by more than ${LOSS_TOLERANCE}`,
        ).toBeLessThanOrEqual(LOSS_TOLERANCE);
      }
    } else {
      console.log("\n⚠ CAPTURE MODE: No ground truth values set. Embed the captured losses above into EXPECTED_LOSSES.");
    }

    // Loss should decrease over 5 steps
    expect(losses[4]).toBeLessThan(losses[0]);

    // Verify fusion is actually happening with AMP cast ops
    // Use stats captured before the last markStep() reset them
    if (lastCumulativeFusionStats) {
      expect(lastCumulativeFusionStats.fusionGroups, "Expected fusion groups > 0").toBeGreaterThan(0);
      expect(lastCumulativeFusionStats.fusedNodes, "Expected fused nodes > 0").toBeGreaterThan(0);
    }
  });
});
