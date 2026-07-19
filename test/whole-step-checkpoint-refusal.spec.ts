/**
 * D3 checkpoint-arena compile REFUSAL — the typed, SUNSET-BOUND decline.
 *
 * Global CE-from-IR contiguity coverage (docs/step-function-compiler-design.md
 * P3) removes the ACCIDENTAL safety that kept the eager two-plan checkpoint
 * forward/recompute/grad plans lowered (their CE-narrow operand was uncovered).
 * Covering CE would let those plans compile with the arena on — the b66ead78
 * hazard: a compiled forward plan reclaims activations a SEPARATE
 * checkpoint-recompute plan still needs → silent corruption. The refusal
 * restores "eager checkpoint plans stay lowered" AS A DECLARATION.
 *
 * This spec asserts the refusal is scoped to the hazard class PRECISELY:
 *   - it FIRES for a checkpointed EAGER backward (getCompileRefusalCount > 0);
 *   - it does NOT fire for a NON-checkpoint step (count == 0 — the P2 reference
 *     compile path is untouched);
 *   - it does NOT fire for a WHOLE-STEP REMAT step (count == 0 — the merged
 *     plan the planner packs the recompute into is safe and must compile). This
 *     case only runs when TORCHLETTE_WHOLE_STEP=1 (the scope machinery is a
 *     module-load const); it is skipped under the plain suite and exercised by
 *     the explicit gate.
 *
 * SUNSET: dies with the bypass (setBufferArenaDisabled + TORCHLETTE_CHECKPOINT_ARENA)
 * when whole-step training defaults and the eager two-plan path is deleted (P4).
 */
import { beforeAll, describe, expect, it } from "vitest";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { gpuMemoryTracker } from "../src/backend/webgpu/memory-tracker";
import { WHOLE_STEP_TRACE } from "../src/core/step-tape";
import {
  getCompileRefusalCount,
  resetCompileRefusalCount,
} from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";
import {
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "../src/graph/node-factory";
import { resetBaseIdCounter } from "../src/runtime/tensor";
import { canUseWebGPU } from "./helpers/webgpu";

const CONFIG: GPT2Config = {
  vocabSize: 256,
  blockSize: 64,
  numLayers: 3,
  numHeads: 4,
  embedDim: 128,
  dropoutRate: 0.0,
};
const SEQ = 32;
const WARMUP = 4; // reach recurrence (build-from-IR reach ≥2)
const MEASURE = 2;

type Mode = "eager-ckpt" | "eager-nockpt" | "remat-ckpt";

/** Run WARMUP+MEASURE steps; return the refusal count accrued over the last
 *  MEASURE steps (counter reset after warmup). */
async function measureRefusals(mode: Mode): Promise<number> {
  gpuMemoryTracker.reset();
  resetNodeIdCounter();
  resetStorageIdCounter();
  resetBaseIdCounter();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.train();
  const useCheckpoint = mode !== "eager-nockpt";
  const ckptOpts = { useCheckpoint, selectiveCheckpoint: false };

  const inputData = Array.from(
    { length: SEQ },
    (_, i) => (i * 7 + 3) % CONFIG.vocabSize,
  );

  const oneStep = async () => {
    const input = api.tensorFromArray(inputData, [1, SEQ], {
      device: "webgpu",
    });
    await api.beginStep();
    const run = async () => {
      const out = model.forward(input, ckptOpts);
      const loss = out.sum();
      if (typeof loss === "number") throw new Error("expected tensor");
      await loss.backward();
      loss.dispose();
    };
    if (mode === "remat-ckpt") {
      await api.wholeStep(run);
    } else {
      await run();
    }
    api.endStep();
    await api.markStep();
    input.dispose();
  };

  const total = WARMUP + MEASURE;
  for (let step = 0; step < total; step++) {
    if (step === WARMUP) resetCompileRefusalCount();
    await oneStep();
  }
  return getCompileRefusalCount();
}

describe("D3 checkpoint-arena compile refusal", { timeout: 300000 }, () => {
  let ok = false;
  beforeAll(async () => {
    ok = await canUseWebGPU();
    if (!ok) console.warn("WebGPU not available - tests will be skipped");
  });

  it("FIRES for an eager checkpointed backward (the hazard class)", async () => {
    if (!ok) return;
    const refusals = await measureRefusals("eager-ckpt");
    console.log(
      `[refusal] eager-ckpt refusals over ${MEASURE} steps: ${refusals}`,
    );
    expect(refusals).toBeGreaterThan(0);
  });

  it("does NOT fire for a non-checkpoint step (P2 reference path untouched)", async () => {
    if (!ok) return;
    const refusals = await measureRefusals("eager-nockpt");
    console.log(`[refusal] eager-nockpt refusals: ${refusals}`);
    expect(refusals).toBe(0);
  });

  // The remat merged plan is safe (one plan, planner packs the recompute) and
  // must compile — the refusal must NOT fire. Needs the whole-step scope active
  // (TORCHLETTE_WHOLE_STEP=1, a module-load const); skipped otherwise.
  it.skipIf(!WHOLE_STEP_TRACE)(
    "does NOT fire for a whole-step remat step (merged plan must compile)",
    async () => {
      if (!ok) return;
      const refusals = await measureRefusals("remat-ckpt");
      console.log(`[refusal] remat-ckpt refusals: ${refusals}`);
      expect(refusals).toBe(0);
    },
  );
});
