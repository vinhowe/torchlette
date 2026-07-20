/**
 * P5 oracle: torchlette's DERIVED optimizers (adam.ts / lion.ts realizing
 * OPTIMIZER_PROGRAMS) vs a REAL torch.optim trajectory over 20 steps.
 *
 * The toy problem L = sum((p − target)²) gives grad = 2(p − target) — a
 * deterministic, autograd-exact trajectory. torch runs torch.optim.AdamW /
 * .Adam / .SGD; torchlette runs the identical loop with its derived optimizer.
 * Per-step param agreement over the FULL trajectory is the strong end-to-end
 * check that the composition-derived update == PyTorch's (design §14 P5 gate).
 */

import { describe, expect, test } from "vitest";

import { Adam, SGD, Torchlette } from "../../src";
import { runTorchOracleBatch } from "./torch-oracle";

const N = 16;
const STEPS = 20;

function randData(n: number, seed: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * 2 - 1);
  }
  return out;
}

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

/** Run torchlette's optimizer over the toy problem; return the flat [STEPS*N]
 *  trajectory (p AFTER each step, matching torch's record order). */
async function torchletteTrajectory(
  api: Torchlette,
  p: import("../../src").Tensor,
  target: import("../../src").Tensor,
  opt: { step: () => unknown; zeroGrad: () => void },
): Promise<number[]> {
  const traj: number[] = [];
  for (let step = 0; step < STEPS; step++) {
    await api.beginStep();
    const diff = api.sub(p, target);
    const loss = api.sum(api.mul(diff, diff));
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    api.endStep();
    await api.markStep();
    traj.push(...(await p.cpu()));
  }
  return traj;
}

describe("P5 oracle: derived optimizer == torch.optim (20-step trajectory)", () => {
  test(
    "AdamW (decoupled wd=0.1) tracks torch.optim.AdamW",
    { timeout: 60000 },
    async () => {
      const lr = 0.05;
      const betas: [number, number] = [0.9, 0.999];
      const eps = 1e-8;
      const wd = 0.1;
      const pData = randData(N, 101);
      const tData = randData(N, 202);

      const api = new Torchlette("cpu");
      const p = api.tensorFromArray(pData.slice(), [N], { requiresGrad: true });
      const target = api.tensorFromArray(tData, [N]);
      const opt = new Adam(
        [p],
        { lr, betas, eps, weightDecay: wd, adamW: true },
        api,
      );
      const tl = await torchletteTrajectory(api, p, target, opt);

      const [oracle] = await runTorchOracleBatch([
        {
          op: "optimizer_trajectory",
          caseName: "adamw.traj",
          inputs: [
            { values: pData, shape: [N] },
            { values: tData, shape: [N] },
          ],
          options: {
            optimizer: "adamw",
            lr,
            betas,
            eps,
            weightDecay: wd,
            steps: STEPS,
          },
        },
      ]);
      expect(maxAbs(tl, oracle.values)).toBeLessThan(1e-6);
    },
  );

  test(
    "Adam (L2 wd=0.1) tracks torch.optim.Adam",
    { timeout: 60000 },
    async () => {
      const lr = 0.05;
      const betas: [number, number] = [0.9, 0.999];
      const eps = 1e-8;
      const wd = 0.1;
      const pData = randData(N, 303);
      const tData = randData(N, 404);

      const api = new Torchlette("cpu");
      const p = api.tensorFromArray(pData.slice(), [N], { requiresGrad: true });
      const target = api.tensorFromArray(tData, [N]);
      const opt = new Adam(
        [p],
        { lr, betas, eps, weightDecay: wd, adamW: false },
        api,
      );
      const tl = await torchletteTrajectory(api, p, target, opt);

      const [oracle] = await runTorchOracleBatch([
        {
          op: "optimizer_trajectory",
          caseName: "adam.traj",
          inputs: [
            { values: pData, shape: [N] },
            { values: tData, shape: [N] },
          ],
          options: {
            optimizer: "adam",
            lr,
            betas,
            eps,
            weightDecay: wd,
            steps: STEPS,
          },
        },
      ]);
      expect(maxAbs(tl, oracle.values)).toBeLessThan(1e-6);
    },
  );

  test("SGD+momentum tracks torch.optim.SGD", { timeout: 60000 }, async () => {
    const lr = 0.02;
    const momentum = 0.9;
    const pData = randData(N, 505);
    const tData = randData(N, 606);

    const api = new Torchlette("cpu");
    const p = api.tensorFromArray(pData.slice(), [N], { requiresGrad: true });
    const target = api.tensorFromArray(tData, [N]);
    const opt = new SGD([p], { lr, momentum }, api);
    const tl = await torchletteTrajectory(api, p, target, opt);

    const [oracle] = await runTorchOracleBatch([
      {
        op: "optimizer_trajectory",
        caseName: "sgd.traj",
        inputs: [
          { values: pData, shape: [N] },
          { values: tData, shape: [N] },
        ],
        options: { optimizer: "sgd", lr, momentum, steps: STEPS },
      },
    ]);
    expect(maxAbs(tl, oracle.values)).toBeLessThan(1e-6);
  });
});
