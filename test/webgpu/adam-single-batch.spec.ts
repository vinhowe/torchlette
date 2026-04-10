/**
 * Regression: an isolated single adamStep node must execute correctly
 * across many plan executions, including the liveness/arena slot reuse
 * patterns that cleared `node.result` in the original "repro252" bug.
 *
 * Bug history: an `adam-batch` action was only emitted when there were ≥2
 * consecutive adamStep nodes in the plan (`if (adamCount > 1)` in
 * lowered-plan.ts). A solitary adamStep would fall through to the generic
 * "sequential" classification and dispatch via executeOpSync →
 * executeAdamStep, which intentionally sets
 * `node.results = [null, mStorage, vStorage]` because at op-handler time
 * the param result hasn't been wrapped yet (the wrapper backfills
 * `node.result` later but never `node.results[0]`). After ANY plan that
 * cleared `node.result` for that node — typically via liveness release
 * with arena slot reuse — the next plan reading `outputIndex === 0` would
 * fall through to `node.results[0] === null` and throw
 * `Input not ready: ... op=adamStep[0]`.
 *
 * The structural fix in lowered-plan.ts now ALWAYS routes adamStep
 * through `adam-batch`, even for a single node. `executeAdamBatchInner`
 * → `assignPackedAdamResult` / `assignPerParamAdamResult` set
 * `node.result` AND `node.results[0]` to the same StorageHandle.
 *
 * This test reproduces the original failing pattern: a tiny model with
 * a SINGLE Adam parameter (so the adam-batch optimization has only one
 * node, exercising the size-1 path), interleaved with no-grad probe
 * forward passes that force their own plans (the trigger for the
 * liveness/arena slot reuse that cleared `node.result`).
 */

import { beforeAll, describe, expect, it } from "vitest";

import { initWebGPU } from "../../src/backend/webgpu/index";
import { Torchlette } from "../../src/frontend/torchlette";
import { Adam } from "../../src/optim/adam";
import {
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "../../src/graph/node-factory";
import { resetBaseIdCounter } from "../../src/runtime/tensor";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("regression: single isolated adamStep", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU not available");
  });

  it("solo adamStep node survives probe-interleaved training", async () => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    const api = new Torchlette("webgpu", { enableFusion: true });

    // ONE parameter — the trigger condition. lowered-plan groups
    // *consecutive* adamStep nodes into adam-batch actions; with a single
    // param there's only ever one adamStep node so the unfixed code
    // dropped it into the generic "sequential" classification path.
    const w = api.tensorFromArray(
      new Float32Array(64).fill(0.5),
      [64],
      { requiresGrad: true },
    );
    const opt = new Adam([w], { lr: 0.01 }, api);

    // Many train steps with an interleaved no-grad probe that forces
    // its own plan. The probe triggers exactly the liveness/arena slot
    // reuse that originally cleared the previous adamStep's
    // `node.result`. The original bug crashed at the 6th probe-plan;
    // 20 iterations is comfortably past that threshold.
    for (let step = 0; step < 20; step++) {
      await api.beginStep();

      const loss = w.mul(w).sum();
      if (typeof loss === "number") throw new Error("expected tensor");
      const v = await loss.item();
      expect(Number.isFinite(v)).toBe(true);
      await loss.backward();
      loss.dispose();
      opt.step();
      opt.zeroGrad();

      // Probe: a separate force boundary that creates lazy nodes
      // referencing the model param (now pending to the new adamStep)
      // and reads them back. Without the fix, the plan executed by
      // `await probe.cpu()` clears `node.result` of the previous
      // adamStep via the liveness/arena interaction; the next training
      // step's beginStep then crashes with "Input not ready: adamStep[0]".
      const probe = api.noGrad(() => w.mul(w));
      if (typeof probe === "number") throw new Error("expected tensor");
      await probe.cpu();
      probe.dispose();

      await api.endStep();
    }

    // After all steps, the param must remain readable. Before the fix,
    // the test crashed inside one of the beginStep / probe.cpu() calls
    // above with "Input not ready: adamStep[0]" long before reaching this
    // assertion.
    const final = await w.cpu();
    expect(final.length).toBe(64);
    for (let i = 0; i < final.length; i++) {
      expect(Number.isFinite(final[i])).toBe(true);
    }
  });
});
