/**
 * Semantic Derivation — P5 OPTIMIZERS AS PROGRAMS (design §4.6, §14 P5).
 *
 * An optimizer's update is a SEMANTIC COMPOSITION (`OptimizerProgram`): a
 * dataflow of the primitive algebra over its state/hyper roles. This gate proves
 * (1) every program is DATA (schema gate — the covenant/R22 defense),
 * (2) the AdamW / SGD+momentum / Lion programs INTERPRET to their reference math
 *     (an independent plain-JS implementation — the Probe-4 shape), and
 * (3) the Lion OPTIMIZER (realized from LION_PROGRAM alone — no hand kernel)
 *     tracks a JS Lion reference over a trajectory (the generality dividend).
 *
 * The fused adamStep kernel is asserted == the derived program at the trajectory
 * level by test/optim/fused-vs-elementwise.spec.ts (the elementwise + foreach
 * paths now DERIVE from ADAMW_PROGRAM, so that differential IS the RT3 seam).
 */

import { describe, expect, it } from "vitest";
import { Lion, Torchlette } from "../src";
import {
  ADAMW_PROGRAM,
  assertNoOptimizerProgramBody,
  evalOptTensor,
  LION_PROGRAM,
  MUON_DEFERRED,
  OPTIMIZER_PROGRAMS,
  type OptRoles,
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
} from "../src/ops/semantic";

const api = new Torchlette("cpu");

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

const N = 24;

describe("P5 — optimizers as programs (schema + reference)", () => {
  it("schema gate: every program is DATA; a smuggled JS body is unconstructible", () => {
    for (const prog of OPTIMIZER_PROGRAMS) {
      expect(() => assertNoOptimizerProgramBody(prog)).not.toThrow();
    }
    // A function leaf smuggled behind a term must be rejected.
    expect(() =>
      assertNoOptimizerProgramBody({
        name: "evil",
        state: ["m"],
        // A function leaf smuggled behind a term (the negative test).
        stateUpdates: [{ slot: "m", expr: (() => 0) as never }],
        paramUpdate: { k: "role", name: "p" },
        hyperRoles: [],
      }),
    ).toThrow(/schema gate/);
    // A state update to an undeclared slot is rejected.
    expect(() =>
      assertNoOptimizerProgramBody({
        name: "bad-slot",
        state: ["m"],
        stateUpdates: [{ slot: "z", expr: { k: "role", name: "g" } }],
        paramUpdate: { k: "role", name: "p" },
        hyperRoles: [],
      }),
    ).toThrow(/unknown state slot/);
  });

  it("AdamW program interprets to the reference math (one step, byte-close)", async () => {
    const lr = 0.01;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;
    const wd = 0.1;
    const t = 1;
    const pData = randData(N, 3);
    const gData = randData(N, 5);
    const rt = api.runtime;

    // Interpret ADAMW_PROGRAM over the runtime: state (m,v start at 0) then p'.
    const p = api.tensorFromArray(pData, [N]);
    const g = api.tensorFromArray(gData, [N]);
    const m0 = api.tensorFromArray(new Array(N).fill(0), [N]);
    const v0 = api.tensorFromArray(new Array(N).fill(0), [N]);
    const bc1 = 1 - beta1 ** t;
    const bc2 = 1 - beta2 ** t;
    const stateRoles: OptRoles = {
      m: m0._unwrap(),
      v: v0._unwrap(),
      g: g._unwrap(),
      beta1,
      om_beta1: 1 - beta1,
      beta2,
      om_beta2: 1 - beta2,
    };
    const mNew = api._wrap(
      evalOptTensor(ADAMW_PROGRAM.stateUpdates[0].expr, rt, stateRoles),
    );
    const vNew = api._wrap(
      evalOptTensor(ADAMW_PROGRAM.stateUpdates[1].expr, rt, stateRoles),
    );
    const paramRoles: OptRoles = {
      p: p._unwrap(),
      m: mNew._unwrap(),
      v: vNew._unwrap(),
      lr,
      eps,
      wd,
      bc1,
      bc2,
    };
    const pNew = api._wrap(
      evalOptTensor(ADAMW_PROGRAM.paramUpdate, rt, paramRoles),
    );

    // Independent JS AdamW (decoupled) reference.
    const F = (v: number) => Math.fround(v);
    const jsM = gData.map((gi) => F((1 - beta1) * gi));
    const jsV = gData.map((gi) => F((1 - beta2) * gi * gi));
    const jsP = pData.map((pi, i) => {
      const mHat = jsM[i] / bc1;
      const vHat = jsV[i] / bc2;
      const scaled = (mHat / (Math.sqrt(vHat) + eps)) * lr;
      return F(pi - (scaled + lr * wd * pi));
    });

    expect(maxAbs(await mNew.cpu(), jsM)).toBeLessThan(1e-6);
    expect(maxAbs(await vNew.cpu(), jsV)).toBeLessThan(1e-6);
    expect(maxAbs(await pNew.cpu(), jsP)).toBeLessThan(1e-6);
  });

  it("SGD+momentum program interprets to the reference math", async () => {
    const lr = 0.1;
    const mu = 0.9;
    const rt = api.runtime;
    const pData = randData(N, 7);
    const gData = randData(N, 9);
    const vData = randData(N, 11); // an existing velocity
    const p = api.tensorFromArray(pData, [N]);
    const g = api.tensorFromArray(gData, [N]);
    const v = api.tensorFromArray(vData, [N]);
    const roles: OptRoles = {
      p: p._unwrap(),
      v: v._unwrap(),
      g: g._unwrap(),
      lr,
      mu,
    };
    const vNew = api._wrap(
      evalOptTensor(SGD_MOMENTUM_PROGRAM.stateUpdates[0].expr, rt, roles),
    );
    const pRoles: OptRoles = { p: p._unwrap(), v: vNew._unwrap(), lr };
    const pNew = api._wrap(
      evalOptTensor(SGD_MOMENTUM_PROGRAM.paramUpdate, rt, pRoles),
    );
    const F = (x: number) => Math.fround(x);
    const jsV = gData.map((gi, i) => F(mu * vData[i] + gi));
    const jsP = pData.map((pi, i) => F(pi - lr * jsV[i]));
    expect(maxAbs(await vNew.cpu(), jsV)).toBeLessThan(1e-6);
    expect(maxAbs(await pNew.cpu(), jsP)).toBeLessThan(1e-6);
    // Plain SGD too.
    const sgdP = api._wrap(
      evalOptTensor(SGD_PROGRAM.paramUpdate, api.runtime, {
        p: p._unwrap(),
        g: g._unwrap(),
        lr,
      }),
    );
    const jsSgd = pData.map((pi, i) => F(pi - lr * gData[i]));
    expect(maxAbs(await sgdP.cpu(), jsSgd)).toBeLessThan(1e-6);
  });

  it("Lion program interprets to the reference math (sign step + β2 EMA)", async () => {
    const lr = 0.01;
    const beta1 = 0.9;
    const beta2 = 0.99;
    const wd = 0.1;
    const rt = api.runtime;
    const pData = randData(N, 13);
    const gData = randData(N, 17);
    const mData = randData(N, 19);
    const p = api.tensorFromArray(pData, [N]);
    const g = api.tensorFromArray(gData, [N]);
    const m = api.tensorFromArray(mData, [N]);
    const roles: OptRoles = {
      p: p._unwrap(),
      m: m._unwrap(),
      g: g._unwrap(),
      lr,
      wd,
      beta1,
      om_beta1: 1 - beta1,
      beta2,
      om_beta2: 1 - beta2,
    };
    const pNew = api._wrap(evalOptTensor(LION_PROGRAM.paramUpdate, rt, roles));
    const mNew = api._wrap(
      evalOptTensor(LION_PROGRAM.stateUpdates[0].expr, rt, roles),
    );
    const F = (x: number) => Math.fround(x);
    const jsP = pData.map((pi, i) => {
      const c = beta1 * mData[i] + (1 - beta1) * gData[i];
      return F(pi - lr * (Math.sign(c) + wd * pi));
    });
    const jsM = gData.map((gi, i) => F(beta2 * mData[i] + (1 - beta2) * gi));
    expect(maxAbs(await pNew.cpu(), jsP)).toBeLessThan(1e-6);
    expect(maxAbs(await mNew.cpu(), jsM)).toBeLessThan(1e-6);
  });

  it("Lion OPTIMIZER (realized from the definition alone) tracks a JS reference", async () => {
    // Toy problem: L = sum((p − target)²) ⇒ grad = 2(p − target), deterministic.
    const lr = 0.02;
    const beta1 = 0.9;
    const beta2 = 0.99;
    const wd = 0.0;
    const STEPS = 8;
    const pData = randData(N, 23);
    const tData = randData(N, 29);

    const p = api.tensorFromArray(pData.slice(), [N], { requiresGrad: true });
    const target = api.tensorFromArray(tData, [N]);
    const opt = new Lion(
      [p],
      { lr, betas: [beta1, beta2], weightDecay: wd },
      api,
    );

    // JS Lion reference (decoupled wd).
    const jsP = pData.slice();
    const jsM = new Array(N).fill(0);

    for (let step = 0; step < STEPS; step++) {
      await api.beginStep();
      const diff = api.sub(p, target);
      const loss = api.sum(api.mul(diff, diff));
      await loss.backward();
      opt.step();
      opt.zeroGrad();
      api.endStep();
      await api.markStep();

      // Reference step.
      for (let i = 0; i < N; i++) {
        const gi = 2 * (jsP[i] - tData[i]);
        const c = beta1 * jsM[i] + (1 - beta1) * gi;
        jsP[i] = jsP[i] - lr * (Math.sign(c) + wd * jsP[i]);
        jsM[i] = beta2 * jsM[i] + (1 - beta2) * gi;
      }
    }
    // fp32 vs fp64 reference drift accumulates over 8 steps; loose but tight
    // enough to catch a wrong update rule (sign, wrong beta, wrong direction).
    expect(maxAbs(await p.cpu(), jsP)).toBeLessThan(1e-4);
  });

  it("MUON is declared but DEFERRED (honest scope)", () => {
    expect(MUON_DEFERRED.name).toBe("muon");
    expect(MUON_DEFERRED.reason).toMatch(/contraction/);
    // It is NOT in the realizable catalog.
    expect(OPTIMIZER_PROGRAMS.some((p) => p.name === "muon")).toBe(false);
  });
});
