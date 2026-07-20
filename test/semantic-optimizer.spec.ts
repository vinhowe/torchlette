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
  buildMuonOrtho,
  evalOptTensor,
  LION_PROGRAM,
  MUON_NS_COEFFS,
  MUON_ORTHO,
  MUON_PROGRAM,
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

// ---- Muon helpers: an honest JS Newton-Schulz + a Jacobi eigensolver --------

/** 2D matmul; `tb` transposes B (`A·Bᵀ`). Rows-major nested arrays. */
function jsMM(A: number[][], B: number[][], tb = false): number[][] {
  const ar = A.length;
  const ac = A[0].length;
  if (tb) {
    const br = B.length; // B is [br, ac]; result [ar, br]
    const out = Array.from({ length: ar }, () => new Array(br).fill(0));
    for (let i = 0; i < ar; i++)
      for (let j = 0; j < br; j++) {
        let s = 0;
        for (let k = 0; k < ac; k++) s += A[i][k] * B[j][k];
        out[i][j] = s;
      }
    return out;
  }
  const bc = B[0].length;
  const out = Array.from({ length: ar }, () => new Array(bc).fill(0));
  for (let i = 0; i < ar; i++)
    for (let j = 0; j < bc; j++) {
      let s = 0;
      for (let k = 0; k < ac; k++) s += A[i][k] * B[k][j];
      out[i][j] = s;
    }
  return out;
}

/** The reference Newton-Schulz quintic (same coeffs as MUON_NS_COEFFS). */
function jsNewtonSchulz(
  X0: number[][],
  scale: number,
  steps: number,
): number[][] {
  const { a, b, c } = MUON_NS_COEFFS;
  let X = X0.map((row) => row.map((v) => v * scale));
  for (let it = 0; it < steps; it++) {
    const A = jsMM(X, X, true); // X·Xᵀ
    const A2 = jsMM(A, A);
    const B = A.map((row, i) => row.map((v, j) => b * v + c * A2[i][j]));
    const BX = jsMM(B, X);
    X = X.map((row, i) => row.map((v, j) => a * v + BX[i][j]));
  }
  return X;
}

/** Ascending eigenvalues of a symmetric matrix (cyclic Jacobi). */
function eigSym(Ain: number[][]): number[] {
  const n = Ain.length;
  const A = Ain.map((r) => r.slice());
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-16) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t =
          Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cs = 1 / Math.sqrt(t * t + 1);
        const sn = t * cs;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = cs * akp - sn * akq;
          A[k][q] = sn * akp + cs * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = cs * apk - sn * aqk;
          A[q][k] = sn * apk + cs * aqk;
        }
      }
  }
  return A.map((_, i) => A[i][i]).sort((x, y) => x - y);
}

/** Singular values of a [m,n] matrix (m≤n) via eigenvalues of X·Xᵀ. */
function singularValues(X: number[][]): number[] {
  const G = jsMM(X, X, true); // X·Xᵀ  [m,m]
  return eigSym(G).map((e) => Math.sqrt(Math.max(0, e)));
}

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

  it("MUON is now a REALIZABLE program (the contraction node exists)", () => {
    // Muon joined the catalog (no longer deferred); it schema-gates like the rest.
    expect(OPTIMIZER_PROGRAMS.some((p) => p.name === "muon")).toBe(true);
    expect(() => assertNoOptimizerProgramBody(MUON_PROGRAM)).not.toThrow();
    // A contraction (`mm`) node with a smuggled function operand is unconstructible.
    expect(() =>
      assertNoOptimizerProgramBody({
        name: "evil-mm",
        state: [],
        stateUpdates: [],
        paramUpdate: {
          k: "mm",
          a: { k: "role", name: "p" },
          b: (() => 0) as never,
          ta: false,
          tb: false,
        },
        hyperRoles: [],
      }),
    ).toThrow(/schema gate/);
  });

  it("MUON orthogonalization interprets == the JS Newton-Schulz reference", async () => {
    // A random, non-orthogonal wide matrix (rows ≤ cols for full-rank X·Xᵀ).
    const M = 12;
    const K = 32;
    const STEPS = 5;
    const data = randData(M * K, 101);
    const X2d: number[][] = [];
    for (let i = 0; i < M; i++) X2d.push(data.slice(i * K, (i + 1) * K));
    let ss = 0;
    for (const v of data) ss += v * v;
    const scale = 1 / (Math.sqrt(ss) + 1e-7);

    const x = api.tensorFromArray(data, [M, K]);
    const Xk = api._wrap(
      evalOptTensor(buildMuonOrtho(STEPS), api.runtime, {
        m: x._unwrap(),
        ns_scale: scale,
      } as OptRoles),
    );
    const got = Array.from(await Xk.cpu());
    const ref = jsNewtonSchulz(X2d, scale, STEPS).flat();
    // The contraction composition reproduces the reference NS (fp32 vs fp64).
    expect(maxAbs(got, ref)).toBeLessThan(1e-4);
  });

  it("MUON orthogonality: X·Xᵀ ≈ I on the singular-value scale (spectral property)", async () => {
    // A DELIBERATELY ill-conditioned input (singular values spanning ~40×), so
    // the assertion tests real orthogonalization, not an already-orthogonal input.
    const M = 12;
    const K = 32;
    const STEPS = 5;
    const raw = randData(M * K, 202);
    // Skew the rows' magnitudes to widen the singular spectrum.
    const data: number[] = [];
    for (let i = 0; i < M; i++)
      for (let j = 0; j < K; j++)
        data.push(raw[i * K + j] * (0.05 + (i / M) * 2));
    const X2d: number[][] = [];
    for (let i = 0; i < M; i++) X2d.push(data.slice(i * K, (i + 1) * K));
    let ss = 0;
    for (const v of data) ss += v * v;
    const scale = 1 / (Math.sqrt(ss) + 1e-7);

    // Input singular values: wide spread (poorly conditioned).
    const inSv = singularValues(X2d.map((r) => r.map((v) => v * scale)));
    const inCond = inSv[inSv.length - 1] / inSv[0];
    expect(inCond).toBeGreaterThan(5); // genuinely ill-conditioned input

    // MUON_ORTHO (the default 5-step term) over the runtime.
    const x = api.tensorFromArray(data, [M, K]);
    const Xk = api._wrap(
      evalOptTensor(MUON_ORTHO, api.runtime, {
        m: x._unwrap(),
        ns_scale: scale,
      } as OptRoles),
    );
    const flat = Array.from(await Xk.cpu());
    const Xmat: number[][] = [];
    for (let i = 0; i < M; i++) Xmat.push(flat.slice(i * K, (i + 1) * K));

    // The spectral property: every singular value is O(1) (the quintic band),
    // i.e. X·Xᵀ ≈ I up to the singular-value scale — a well-conditioned output.
    const outSv = singularValues(Xmat);
    const outCond = outSv[outSv.length - 1] / outSv[0];
    expect(outSv[0]).toBeGreaterThan(0.6); // no collapsed singular value
    expect(outSv[outSv.length - 1]).toBeLessThan(1.4); // none blown up
    expect(outCond).toBeLessThan(2); // orthogonalized (cond → ~1)
    expect(outCond).toBeLessThan(inCond / 3); // dramatic improvement vs input

    // And directly: the off-diagonal of the row-Gram is small relative to the
    // diagonal (rows are ~orthogonal).
    const G = jsMM(Xmat, Xmat, true);
    let offMax = 0;
    let diagMin = Infinity;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < M; j++) {
        if (i === j) diagMin = Math.min(diagMin, G[i][j]);
        else offMax = Math.max(offMax, Math.abs(G[i][j]));
      }
    // Rows are ~orthogonal: off-diagonal well under the diagonal. (The rigorous
    // spectral statement is the outCond<2 band above; this is a direct corollary.)
    expect(offMax).toBeLessThan(0.5 * diagMin);
  });
});
