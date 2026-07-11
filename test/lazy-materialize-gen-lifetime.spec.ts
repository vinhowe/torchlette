/**
 * Regression gate for the LAZILY-MATERIALIZED-PERSISTENT-PARAM claim-attribution
 * FALSE POSITIVE (task #86 — "the last STRICT transient" on the implied-boundary
 * path).
 *
 * A user param created eagerly via `tensorFromArray` is a PENDING tensor: its
 * upload is a lazy op node that materializes generations later, INSIDE a step.
 * The step tracker stamps a wrapper's generation (`_wrapperGen`) — the signal
 * that separates "this step's tensors" from the next step's lazily-built work.
 * Pre-fix, that stamp was set at the FIRST `trackTensor`, i.e. at MATERIALIZE
 * time. So a persistent param born before the loop carried a generation LATER
 * than the closing step boundary and was filtered OUT of the persistent
 * snapshot in `snapshotForStep`. `releaseStepTemps` then reaped its storage
 * while the live wrapper still pointed at it — a reclaimed-read that
 * TORCHLETTE_STRICT_LIFETIME=1 turns into a hard throw (and a genuine UAF the
 * moment the pooled buffer is reused).
 *
 * The fix stamps `_wrapperGen` at OBJECT CONSTRUCTION (Tensor ctor →
 * storageTracker.stampWrapperGen), tying the generation to when the wrapper was
 * born rather than when its storage first materialized. It cannot mask a real
 * step-temp UAF: a genuine mid-step temporary is still born in the current
 * step's generation and still reaped; only the mis-attribution of an
 * eagerly-created persistent param is corrected.
 *
 * This is a minimal SGD training loop with NO explicit markStep (implied
 * boundaries). Under strict, pre-fix it THROWS on the 2nd step's read of the
 * param `w`; post-fix it trains correctly to w = 10 - 3*grad = 7.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import {
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "../src/graph/node-factory";
import { SGD } from "../src/optim";
import { resetBaseIdCounter } from "../src/runtime/tensor";

let hasGPU = false;
beforeAll(async () => {
  hasGPU = await initWebGPU();
});

describe("lazily-materialized persistent param lifetime (task #86)", () => {
  it("minimal SGD loop reads the param across implied boundaries under STRICT_LIFETIME", async () => {
    if (!hasGPU) return;
    // Force the reclaimed-read guard to THROW so a mis-attributed persistent
    // param is a HARD failure here, not a swallowed warning.
    const prevStrict = process.env.TORCHLETTE_STRICT_LIFETIME;
    process.env.TORCHLETTE_STRICT_LIFETIME = "1";
    try {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const w = api.tensorFromArray([10, 10, 10, 10], [4], {
        requiresGrad: true,
      });
      const optimizer = new SGD([w], { lr: 1.0 }, api);

      // Three implied-boundary steps: zeroGrad → sum → backward → step().
      // Pre-fix, step 2's read of `w` throws (its storage was reaped at the
      // step-1 boundary because `w`'s late materialize-time gen filtered it out
      // of the persistent snapshot).
      for (let i = 0; i < 3; i++) {
        optimizer.zeroGrad();
        const currentW = optimizer.getParams()[0];
        const loss = currentW.sum();
        if (typeof loss === "number") throw new Error("Expected tensor");
        await loss.backward();
        optimizer.step();
      }

      const finalW = optimizer.getParams()[0];
      const finalWeights = await finalW.cpu();
      // lr=1, grad=1 (d/dw of sum(w)), 3 steps → 10 - 3 = 7.
      for (let k = 0; k < 4; k++) {
        expect(finalWeights[k]).toBeCloseTo(7, 4);
      }
    } finally {
      if (prevStrict === undefined)
        delete process.env.TORCHLETTE_STRICT_LIFETIME;
      else process.env.TORCHLETTE_STRICT_LIFETIME = prevStrict;
    }
  }, 60_000);
});
