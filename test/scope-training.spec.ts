/**
 * Training INSIDE api.scope() with the default (fused) Adam optimizer.
 *
 * This is the artifact proving api.scope() is a real beginStep/endStep
 * replacement for a training loop (docs/scoped-memory-design.md §2, §6, §9):
 * forward + backward + optimizer.step() run inside one scope, and the scope's
 * close IS the reclamation boundary — no beginStep/endStep/markStep ceremony,
 * no per-step leak.
 *
 * Regression it guards (TASK #68): fused Adam's moment state (m/v) is created
 * lazily and first MATERIALIZES mid-scope, so it is not in the scope-entry
 * snapshot. Without the optimizer adopting it (runtime.persist, §6), scope
 * close demoted m/v as step temporaries — their buffers pooled while the
 * optimizer still pointed at them → silent UAF → NaN on the second step. Under
 * markStep the next beginStep re-snapshots m/v, hiding the bug; the scope
 * surface has a fixed entry snapshot, so it surfaced there. The fix persists
 * m/v at the fused step; this test pins scope-training == markStep-training and
 * storage-flat so the persistence can't silently regress.
 *
 * Runs on webgpu when available (exercises the FUSED kernel — the path the bug
 * lived in), else cpu (elementwise Adam; the scope reclamation is
 * backend-agnostic, so both projects exercise the boundary).
 */

import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import type { Tensor } from "../src/frontend/tensor";
import { storageTracker } from "../src/graph/storage-tracker";
import { Adam } from "../src/optim/adam";
import { canUseWebGPU } from "./helpers/webgpu";

const TIMEOUT = 120_000;

const D_IN = 8;
const D_H = 16;
const D_OUT = 4;
const BATCH = 2;

function initData(scale: number, n: number, offset = 0): number[] {
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = Math.sin(i * 12.9898 + offset) * scale;
  }
  return out;
}

interface Model {
  w1: Tensor;
  w2: Tensor;
  params: Tensor[];
}

function makeModel(api: Torchlette, device: "cpu" | "webgpu"): Model {
  const w1 = api
    .tensorFromArray(initData(0.3, D_IN * D_H, 1), [D_IN, D_H], { device })
    .requires_grad_();
  const w2 = api
    .tensorFromArray(initData(0.3, D_H * D_OUT, 2), [D_H, D_OUT], { device })
    .requires_grad_();
  return { w1, w2, params: [w1, w2] };
}

function forwardLoss(api: Torchlette, m: Model, x: Tensor, t: Tensor): Tensor {
  return api.tidy(() => {
    const h = api.relu(api.matmul(x, m.w1));
    const pred = api.matmul(h, m.w2);
    const diff = api.sub(pred, t);
    const loss = api.mean(api.mul(diff, diff));
    // Under an open scope keep() escapes to ROOT; tidy's return-escape keeps
    // the loss usable inside the scope without leaking. Only keep() outside a
    // scope (the markStep reference loop).
    return loss;
  });
}

function makeBatch(api: Torchlette, device: "cpu" | "webgpu", step: number) {
  const x = api.tensorFromArray(
    initData(1.0, BATCH * D_IN, 100 + step),
    [BATCH, D_IN],
    { device },
  );
  const t = api.tensorFromArray(
    initData(1.0, BATCH * D_OUT, 200 + step),
    [BATCH, D_OUT],
    { device },
  );
  return { x, t };
}

/** Reference: explicit beginStep/endStep/markStep ceremony. */
async function trainMarkStep(
  api: Torchlette,
  device: "cpu" | "webgpu",
  steps: number,
): Promise<number[]> {
  const m = makeModel(api, device);
  const opt = new Adam(m.params, { lr: 1e-2 }, api);
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    await api.beginStep();
    // Fixed batch every step: the model overfits one example, so the loss
    // must monotonically decrease (a NaN or UAF corruption breaks that).
    const { x, t } = makeBatch(api, device, 0);
    const loss = forwardLoss(api, m, x, t);
    api.keep(loss);
    losses.push(await loss.item());
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    x.dispose();
    t.dispose();
    api.endStep();
    await api.markStep();
  }
  return losses;
}

/** Under test: forward + backward + step inside api.scope(), no ceremony. */
async function trainScope(
  api: Torchlette,
  device: "cpu" | "webgpu",
  steps: number,
  onStep?: (s: number) => void,
): Promise<number[]> {
  const m = makeModel(api, device);
  const opt = new Adam(m.params, { lr: 1e-2 }, api);
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    let lossVal = NaN;
    await api.scope(async () => {
      const { x, t } = makeBatch(api, device, 0);
      const loss = forwardLoss(api, m, x, t);
      lossVal = await loss.item();
      await loss.backward();
      opt.step();
      opt.zeroGrad();
      x.dispose();
      t.dispose();
    });
    losses.push(lossVal);
    onStep?.(s);
  }
  return losses;
}

describe("training inside api.scope() (default fused Adam)", () => {
  let device: "cpu" | "webgpu" = "cpu";
  beforeAll(async () => {
    device = (await canUseWebGPU()) ? "webgpu" : "cpu";
  });

  it(
    "loss decreases and matches the markStep reference",
    async () => {
      const ref = await trainMarkStep(new Torchlette(device), device, 12);
      const scoped = await trainScope(new Torchlette(device), device, 12);

      // Finite + monotone-ish decrease (the whole point: no step-2 NaN).
      for (const l of scoped) expect(Number.isFinite(l)).toBe(true);
      expect(scoped[scoped.length - 1]).toBeLessThan(scoped[0]);

      // scope() is a faithful beginStep/endStep replacement: same trajectory.
      expect(scoped.length).toBe(ref.length);
      for (let i = 0; i < ref.length; i++) {
        expect(Math.abs(scoped[i] - ref[i])).toBeLessThan(1e-4);
      }
    },
    TIMEOUT,
  );

  it(
    "keeps reachable storage flat across scope steps (no per-step leak/UAF)",
    async () => {
      const api = new Torchlette(device);
      const reachable: number[] = [];
      await trainScope(api, device, 12, () => {
        reachable.push(storageTracker.stats().reachableStorages);
      });
      // Persistent set (params + Adam m/v) is fixed; the scope reclaims each
      // step's temporaries at close. Reachable count is flat once warm — any
      // per-step leak of live state would grow it monotonically.
      const growth = reachable[11] - reachable[4];
      expect(growth).toBeLessThanOrEqual(2);
    },
    TIMEOUT,
  );
});
