/**
 * Scoped-memory stage 1: the autograd graph INDEPENDENTLY retains its
 * saved-for-backward tensors (PyTorch semantics). A forward run inside a
 * reclamation scope (tidy / step) or an intermediate whose user handle is
 * disposed must still produce correct gradients when backward() runs later.
 *
 * Differential contract: gradients from forward-in-scope + backward-outside
 * must be numerically identical to the same computation with NO scope.
 *
 * The `manual dispose of an intermediate` case is the documented-limitation
 * fix ("disposing intermediates breaks autograd"): it FAILS on the pre-change
 * code (null / wrong gradient — the graph shared lifecycle with the user
 * handle) and PASSES now that the graph owns its saved values.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import type { DeviceKind } from "../src/backend/types";
import { initWebGPU } from "../src/backend/webgpu";
import { cpuOnly } from "./helpers/webgpu";

const TIMEOUT = 120_000;

function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i += 1) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function runSuite(device: DeviceKind) {
  const ATOL = device === "webgpu" ? 1e-4 : 1e-5;

  // Deterministic small MLP: y = gelu(x @ W1) @ W2 ; loss = sum(y*y).
  // biome-ignore lint/suspicious/noExplicitAny: test tensors typed loosely
  function mlp(t: Torchlette, W1: any, W2: any, x: any) {
    const h = t.matmul(x, W1);
    const g = t.gelu(h);
    const y = t.matmul(g, W2);
    return t.sum(t.mul(y, y));
  }

  function makeMLP(t: Torchlette) {
    const W1 = t.tensorFromArray(
      Array.from({ length: 12 }, (_, i) => Math.sin(i * 0.7) * 0.4),
      [3, 4],
      { requiresGrad: true, device },
    );
    const W2 = t.tensorFromArray(
      Array.from({ length: 8 }, (_, i) => Math.cos(i * 0.5) * 0.3),
      [4, 2],
      { requiresGrad: true, device },
    );
    const x = t.tensorFromArray(
      Array.from({ length: 6 }, (_, i) => Math.sin(i * 1.3) * 0.5),
      [2, 3],
      { device },
    );
    return { W1, W2, x };
  }

  async function grads(t: Torchlette, W1: any, W2: any): Promise<number[]> {
    const g1 = await W1.grad.cpu();
    const g2 = await W2.grad.cpu();
    return [...g1, ...g2];
  }

  it(
    "MLP: forward in tidy, backward outside == no-tidy",
    async () => {
      const t = new Torchlette(device);
      const { W1, W2, x } = makeMLP(t);

      // Reference: no scope.
      const lossRef = mlp(t, W1, W2, x);
      await lossRef.backward();
      const ref = await grads(t, W1, W2);
      W1.zeroGrad();
      W2.zeroGrad();

      // Forward INSIDE tidy, returning only the loss; backward OUTSIDE.
      const loss = t.tidy(() => mlp(t, W1, W2, x));
      await loss.backward();
      const got = await grads(t, W1, W2);

      expect(maxAbsDiff(ref, got)).toBeLessThan(ATOL);
    },
    TIMEOUT,
  );

  it(
    "MLP: forward before markStep, backward after (step scope)",
    async () => {
      const t = new Torchlette(device);
      const { W1, W2, x } = makeMLP(t);

      const lossRef = mlp(t, W1, W2, x);
      await lossRef.backward();
      const ref = await grads(t, W1, W2);
      W1.zeroGrad();
      W2.zeroGrad();

      // Step-scoped reclamation between forward and backward.
      t.setStepScopedCleanup(true);
      await t.beginStep();
      const loss = mlp(t, W1, W2, x);
      await loss.item(); // materialize saved activations mid-step
      await t.markStep(); // reclamation boundary before backward
      await loss.backward();
      const got = await grads(t, W1, W2);

      expect(maxAbsDiff(ref, got)).toBeLessThan(ATOL);
    },
    TIMEOUT,
  );

  it(
    "attention/matmul chain (saved qkv): forward in tidy, backward outside",
    async () => {
      const t = new Torchlette(device);
      const mk = (seed: number) =>
        t.tensorFromArray(
          Array.from({ length: 32 }, (_, i) => Math.sin((i + seed) * 0.31) * 0.5),
          [1, 2, 4, 4], // [B, H, S, D]
          { requiresGrad: true, device },
        );
      const Wq = mk(1);
      const Wk = mk(2);
      const Wv = mk(3);
      const xin = t.tensorFromArray(
        Array.from({ length: 32 }, (_, i) => Math.cos((i + 5) * 0.23) * 0.5),
        [1, 2, 4, 4],
        { device },
      );

      const forward = () => {
        const q = t.mul(Wq, xin);
        const k = t.mul(Wk, xin);
        const v = t.mul(Wv, xin);
        const o = t.scaledDotProductAttention(q, k, v);
        return t.sum(t.mul(o, o));
      };

      const lossRef = forward();
      await lossRef.backward();
      const ref = [
        ...(await Wq.grad!.cpu()),
        ...(await Wk.grad!.cpu()),
        ...(await Wv.grad!.cpu()),
      ];
      Wq.zeroGrad();
      Wk.zeroGrad();
      Wv.zeroGrad();

      const loss = t.tidy(() => forward());
      await loss.backward();
      const got = [
        ...(await Wq.grad!.cpu()),
        ...(await Wk.grad!.cpu()),
        ...(await Wv.grad!.cpu()),
      ];

      expect(maxAbsDiff(ref, got)).toBeLessThan(ATOL);
    },
    TIMEOUT,
  );

  it(
    "manual dispose of an intermediate does not break autograd (limitation fix)",
    async () => {
      const t = new Torchlette(device);
      const { W1, W2, x } = makeMLP(t);

      const lossRef = mlp(t, W1, W2, x);
      await lossRef.backward();
      const ref = await grads(t, W1, W2);
      W1.zeroGrad();
      W2.zeroGrad();

      // Recompute, disposing an intermediate handle before backward.
      const h = t.matmul(x, W1);
      const g = t.gelu(h);
      const y = t.matmul(g, W2);
      const loss = t.sum(t.mul(y, y));
      h.dispose(); // the saved intermediate handle
      g.dispose();
      await loss.backward();
      const got = await grads(t, W1, W2);

      expect(maxAbsDiff(ref, got)).toBeLessThan(ATOL);
    },
    TIMEOUT,
  );
}

describe("autograd scope lifetime (CPU)", () => {
  runSuite("cpu");
});

describe.skipIf(cpuOnly)("autograd scope lifetime (WebGPU)", () => {
  beforeAll(async () => {
    await initWebGPU();
  });
  runSuite("webgpu");
});
