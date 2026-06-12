/**
 * Implied step boundaries — the minimal-training-loop contract.
 *
 * A training loop should need NO beginStep/endStep/markStep ceremony:
 *
 *   forward → loss.backward() → optimizer.step()
 *
 * optimizer.step() queues a DEFERRED boundary (Torchlette.queueStepBoundary);
 * it commits at the next backward() or explicit markStep()/beginStep(). The
 * deferral is what makes the ergonomics safe:
 *  - loss.item() AFTER step() still reads the correct value (cleanup hasn't
 *    run yet),
 *  - multiple optimizers stepping back-to-back share ONE boundary,
 *  - gradient accumulation needs nothing special (no step() → no boundary).
 *
 * Generation-scoping (storage-tracker stepGen stamps) keeps the commit safe
 * even though the next iteration's forward graph is lazily built — and its
 * storages possibly materialized via loss.item() — BEFORE the commit runs:
 * post-boundary storages are excluded from both the demotion sweep and the
 * persistence snapshot.
 *
 * The differential here: a minimal loop must produce the SAME losses as the
 * canonical explicit loop, with flat storage counts.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { storageTracker } from "../src/graph/storage-tracker";
import type { Tensor } from "../src/frontend/tensor";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim/adam";
import { canUseWebGPU } from "./helpers/webgpu";

const TIMEOUT = 120_000;

const D_IN = 8;
const D_H = 16;
const D_OUT = 4;
const BATCH = 2;

function initData(scale: number, n: number, offset = 0): number[] {
  // Deterministic pseudo-random init, identical across models.
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
    api.keep(loss);
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

/** Canonical explicit-ceremony loop. */
async function trainExplicit(
  api: Torchlette,
  device: "cpu" | "webgpu",
  steps: number,
): Promise<number[]> {
  const m = makeModel(api, device);
  const opt = new Adam(m.params, { lr: 1e-2 }, api);
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    await api.beginStep();
    const { x, t } = makeBatch(api, device, s);
    const loss = forwardLoss(api, m, x, t);
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

/** Minimal loop: no beginStep/endStep/markStep, no dispose. */
async function trainMinimal(
  api: Torchlette,
  device: "cpu" | "webgpu",
  steps: number,
  onStep?: (s: number) => void,
): Promise<number[]> {
  const m = makeModel(api, device);
  const opt = new Adam(m.params, { lr: 1e-2 }, api);
  const losses: number[] = [];
  for (let s = 0; s < steps; s++) {
    const { x, t } = makeBatch(api, device, s);
    const loss = forwardLoss(api, m, x, t);
    losses.push(await loss.item());
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    onStep?.(s);
  }
  return losses;
}

describe("implied step boundaries (minimal training loops)", () => {
  let device: "cpu" | "webgpu" = "cpu";
  beforeAll(async () => {
    device = (await canUseWebGPU()) ? "webgpu" : "cpu";
  });

  it(
    "minimal loop matches the explicit-ceremony loop exactly",
    async () => {
      const api = new Torchlette(device);
      const explicit = await trainExplicit(api, device, 10);
      const minimal = await trainMinimal(api, device, 10);
      expect(minimal.length).toBe(explicit.length);
      for (let i = 0; i < explicit.length; i++) {
        expect(Math.abs(minimal[i] - explicit[i])).toBeLessThan(1e-5);
      }
      // And the run is sane (finite throughout).
      for (const l of minimal) expect(Number.isFinite(l)).toBe(true);
    },
    TIMEOUT,
  );

  it(
    "minimal loop keeps storage counts flat (boundary cleanup really runs)",
    async () => {
      const api = new Torchlette(device);
      const counts: number[] = [];
      await trainMinimal(api, device, 12, () => {
        counts.push(storageTracker.stats().totalStorages);
      });
      // Steady state by step 4; commits lag one backward, so compare a
      // mid-run sample against the end. Any per-step leak of even one
      // storage would diverge by ≥7 over this window.
      const growth = counts[11] - counts[4];
      expect(growth).toBeLessThanOrEqual(3);
    },
    TIMEOUT,
  );

  it(
    "reading a step temp after optimizer.step() still works (deferred commit)",
    async () => {
      // The boundary is QUEUED at step() but commits at the next backward —
      // so a step temporary read after step() must still hold its value. An
      // eager commit would demote its storage here. The probe must live
      // OUTSIDE the autograd graph: reading graph-derived tensors (even
      // detached+kept ones) after backward() is a separate, pre-existing
      // limitation — backward disposes the forward graph.
      const api = new Torchlette(device);
      const m = makeModel(api, device);
      const opt = new Adam(m.params, { lr: 1e-2 }, api);
      for (let s = 0; s < 3; s++) {
        const { x, t } = makeBatch(api, device, s);
        const { loss, metric } = api.tidy(() => {
          const h = api.relu(api.matmul(x, m.w1));
          const pred = api.matmul(h, m.w2);
          const diff = api.sub(pred, t);
          const l = api.mean(api.mul(diff, diff));
          const metricT = api.mean(x);
          api.keep(l);
          api.keep(metricT);
          return { loss: l, metric: metricT };
        });
        const before = await metric.item();
        await loss.backward();
        opt.step();
        // Boundary queued, not committed — the read must still work.
        const after = await metric.item();
        opt.zeroGrad();
        expect(after).toBeCloseTo(before, 6);
      }
    },
    TIMEOUT,
  );

  it(
    "two optimizers stepping back-to-back share one boundary",
    async () => {
      // Adam updates are per-param, so two Adams over disjoint param sets
      // must produce the SAME trajectory as one Adam over all params — IF
      // the implied boundary doesn't fire between opt1.step() and
      // opt2.step() (a boundary there would demote opt2's grads).
      const api = new Torchlette(device);
      const reference = await trainExplicit(api, device, 8);
      const m = makeModel(api, device);
      const opt1 = new Adam([m.w1], { lr: 1e-2 }, api);
      const opt2 = new Adam([m.w2], { lr: 1e-2 }, api);
      const losses: number[] = [];
      for (let s = 0; s < 8; s++) {
        const { x, t } = makeBatch(api, device, s);
        const loss = forwardLoss(api, m, x, t);
        losses.push(await loss.item());
        await loss.backward();
        opt1.step();
        opt2.step();
        opt1.zeroGrad();
        opt2.zeroGrad();
      }
      expect(
        losses.map((l, i) => Math.abs(l - reference[i])),
        `two-opt ${JSON.stringify(losses)} vs ref ${JSON.stringify(reference)}`,
      ).toSatisfy((d: number[]) => d.every((x) => x < 1e-5));
    },
    TIMEOUT,
  );

  it(
    "gradient accumulation: boundaries only at step()",
    async () => {
      const api = new Torchlette(device);
      const m = makeModel(api, device);
      const opt = new Adam(m.params, { lr: 1e-2 }, api);
      const losses: number[] = [];
      for (let s = 0; s < 6; s++) {
        let last = 0;
        for (let micro = 0; micro < 2; micro++) {
          const { x, t } = makeBatch(api, device, s * 2 + micro);
          const loss = forwardLoss(api, m, x, t);
          last = await loss.item();
          await loss.backward();
        }
        losses.push(last);
        opt.step();
        opt.zeroGrad();
      }
      for (const l of losses) expect(Number.isFinite(l)).toBe(true);
    },
    TIMEOUT,
  );

  it(
    "mixed usage: explicit markStep after step() supersedes the queued boundary",
    async () => {
      // Same data as the explicit loop → identical losses required.
      const api = new Torchlette(device);
      const reference = await trainExplicit(api, device, 6);
      const m = makeModel(api, device);
      const opt = new Adam(m.params, { lr: 1e-2 }, api);
      const losses: number[] = [];
      for (let s = 0; s < 6; s++) {
        const { x, t } = makeBatch(api, device, s);
        const loss = forwardLoss(api, m, x, t);
        losses.push(await loss.item());
        await loss.backward();
        opt.step();
        opt.zeroGrad();
        await api.markStep(); // explicit boundary — must not double-clean
      }
      for (let i = 0; i < reference.length; i++) {
        expect(Math.abs(losses[i] - reference[i])).toBeLessThan(1e-5);
      }
    },
    TIMEOUT,
  );
});
