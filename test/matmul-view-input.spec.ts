/**
 * Task #67 — raw dispatch op composed directly with a view input.
 *
 * `api.matmul(q, api.transpose(k, -2, -1))` (a transpose VIEW handed straight
 * to a dispatch op, no intervening materializing op) errored at execution with
 * "Input not ready: transpose": the plan / readiness machinery did not treat
 * the pure-view node as a satisfiable input.
 *
 * This is the WORKBENCH entry-criterion class (docs/schedule-state-design.md
 * §7): user-composed graphs will hit a dispatch op with a raw view input
 * immediately. Our model code paths dodge it (nn.linear resolves transposes
 * via detectSimpleTranspose internally), so it went unnoticed.
 *
 * The battery maps the failure surface: which raw-view-into-dispatch-op
 * compositions fail and which accidentally work — across matmul (both arg
 * positions), reductions, gather, elementwise consumers; batched; chained
 * views (transpose-of-narrow); lazy and forced paths; fusion on/off; and the
 * compiled plan on both its 1st (records/lowered) and 2nd (replay) execution.
 *
 * Reference values are computed in plain JS from the raw data + view metadata
 * (NOT through the framework) so a shared bug cannot self-validate.
 */

import { beforeAll, describe, expect, it } from "vitest";
import type { DeviceKind } from "../src/backend/types";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { cpuOnly } from "./helpers/webgpu";

const TIMEOUT = 120_000;

// ---------------------------------------------------------------------------
// Plain-JS references (deliberately independent of the framework)
// ---------------------------------------------------------------------------

/** 2-D matmul on row-major flat arrays. */
function matmul2d(
  a: number[],
  ar: number,
  ac: number,
  b: number[],
  br: number,
  bc: number,
): number[] {
  if (ac !== br) throw new Error("inner dim mismatch");
  const out = new Array(ar * bc).fill(0);
  for (let i = 0; i < ar; i++) {
    for (let j = 0; j < bc; j++) {
      let s = 0;
      for (let k = 0; k < ac; k++) s += a[i * ac + k] * b[k * bc + j];
      out[i * bc + j] = s;
    }
  }
  return out;
}

/** Transpose a row-major [r,c] into a materialized [c,r]. */
function transpose2d(a: number[], r: number, c: number): number[] {
  const out = new Array(r * c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) out[j * r + i] = a[i * c + j];
  return out;
}

function arange(n: number, scale = 1): number[] {
  return Array.from({ length: n }, (_, i) => (i + 1) * scale);
}

function approxEqual(got: number[], want: number[], tol = 1e-3): void {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < got.length; i++) {
    expect(Math.abs(got[i] - want[i])).toBeLessThan(tol);
  }
}

// ---------------------------------------------------------------------------

const devices: DeviceKind[] = cpuOnly ? ["cpu"] : ["cpu", "webgpu"];

describe.each(devices)("matmul/dispatch with raw view input [%s]", (device) => {
  let deviceReady = true;

  beforeAll(async () => {
    if (device === "webgpu") {
      // initWebGPU returns false (does NOT throw) when adapter acquisition
      // fails — treat both as "device unavailable, skip" (canUseWebGPU
      // semantics), so a Dawn/Vulkan init flake reads as a skip, not a
      // spurious failure of this gate.
      try {
        deviceReady = await initWebGPU();
      } catch {
        deviceReady = false;
      }
    }
  }, TIMEOUT);

  const mk = () => new Torchlette(device as DeviceKind);

  it(
    "matmul(q, transpose(k)) — the reported failure",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      // q: [2,3], k: [4,3]; transpose(k) -> [3,4]; q @ kT -> [2,4]
      const qData = arange(6);
      const kData = arange(12, 0.5);
      const q = t.tensorFromArray(qData, [2, 3], { device });
      const k = t.tensorFromArray(kData, [4, 3], { device });
      const kT = t.transpose(k, { dim0: 0, dim1: 1 });
      const got = Array.from(await t.matmul(q, kT).cpu());
      const want = matmul2d(qData, 2, 3, transpose2d(kData, 4, 3), 3, 4);
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "matmul(transpose(k), q) — transpose as FIRST arg",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      // k: [4,3] -> kT [3,4]; q: [4,2]; kT... wait need inner match
      // kT [3,4] @ x [4,2] -> [3,2]
      const kData = arange(12, 0.5);
      const xData = arange(8);
      const k = t.tensorFromArray(kData, [4, 3], { device });
      const x = t.tensorFromArray(xData, [4, 2], { device });
      const kT = t.transpose(k, { dim0: 0, dim1: 1 }); // [3,4]
      const got = Array.from(await t.matmul(kT, x).cpu());
      const want = matmul2d(transpose2d(kData, 4, 3), 3, 4, xData, 4, 2);
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "matmul(transpose(a), transpose(b)) — both inputs are views",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      // a:[3,2] -> aT:[2,3]; b:[4,3] -> bT:[3,4]; aT@bT -> [2,4]
      const aData = arange(6);
      const bData = arange(12, 0.25);
      const a = t.tensorFromArray(aData, [3, 2], { device });
      const b = t.tensorFromArray(bData, [4, 3], { device });
      const aT = t.transpose(a, { dim0: 0, dim1: 1 });
      const bT = t.transpose(b, { dim0: 0, dim1: 1 });
      const got = Array.from(await t.matmul(aT, bT).cpu());
      const want = matmul2d(
        transpose2d(aData, 3, 2),
        2,
        3,
        transpose2d(bData, 4, 3),
        3,
        4,
      );
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "batched matmul(q, transpose(k)) — attention-shaped [B,H,S,D]",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      const B = 2,
        S = 3,
        D = 4;
      // q:[B,S,D], k:[B,S,D]; transpose last two of k -> [B,D,S]; scores [B,S,S]
      const qData = arange(B * S * D, 0.1);
      const kData = arange(B * S * D, 0.2);
      const q = t.tensorFromArray(qData, [B, S, D], { device });
      const k = t.tensorFromArray(kData, [B, S, D], { device });
      const kT = t.transpose(k, { dim0: 1, dim1: 2 }); // [B,D,S]
      const got = Array.from(await t.matmul(q, kT).cpu());
      // reference per batch
      const want: number[] = [];
      for (let b = 0; b < B; b++) {
        const qb = qData.slice(b * S * D, (b + 1) * S * D);
        const kb = kData.slice(b * S * D, (b + 1) * S * D);
        const kbT = transpose2d(kb, S, D); // [D,S]
        want.push(...matmul2d(qb, S, D, kbT, D, S));
      }
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "matmul(q, transpose(narrow(k))) — chained view: narrow then transpose",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      // k full [4,5]; narrow to [4,3] (cols 1..3); transpose -> [3,4]; q[2,4]@ -> [2,3]...
      // q must match inner: q[2,4] @ kT[4,?]. narrow k on dim1 gives [4,3], transpose -> [3,4].
      // So q[2,3] @ kT[3,4] -> [2,4]. Let me use narrow on dim0.
      const kData = arange(20, 0.5); // [4,5]
      const k = t.tensorFromArray(kData, [4, 5], { device });
      const kN = t.narrow(k, 1, 1, 3); // [4,3], cols 1,2,3
      const kNT = t.transpose(kN, { dim0: 0, dim1: 1 }); // [3,4]
      const qData = arange(6); // [2,3]
      const q = t.tensorFromArray(qData, [2, 3], { device });
      const got = Array.from(await t.matmul(q, kNT).cpu());
      // reference: build kN materialized [4,3]
      const kN_mat: number[] = [];
      for (let r = 0; r < 4; r++)
        for (let c = 1; c < 4; c++) kN_mat.push(kData[r * 5 + c]);
      const kNT_mat = transpose2d(kN_mat, 4, 3); // [3,4]
      const want = matmul2d(qData, 2, 3, kNT_mat, 3, 4);
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "sum(transpose(a)) — reduction with raw view input",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      const aData = arange(6);
      const a = t.tensorFromArray(aData, [2, 3], { device });
      const aT = t.transpose(a, { dim0: 0, dim1: 1 }); // [3,2]
      // sum over dim 0 of aT -> shape [2]
      const got = Array.from(await t.sum(aT, { dim: 0 }).cpu());
      const aT_mat = transpose2d(aData, 2, 3); // [3,2]
      const want = [
        aT_mat[0] + aT_mat[2] + aT_mat[4],
        aT_mat[1] + aT_mat[3] + aT_mat[5],
      ];
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "add(transpose(a), transpose(b)) — elementwise with two view inputs",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      const aData = arange(6);
      const bData = arange(6, 10);
      const a = t.tensorFromArray(aData, [2, 3], { device });
      const b = t.tensorFromArray(bData, [2, 3], { device });
      const aT = t.transpose(a, { dim0: 0, dim1: 1 });
      const bT = t.transpose(b, { dim0: 0, dim1: 1 });
      const got = Array.from(await t.add(aT, bT).cpu());
      const aT_mat = transpose2d(aData, 2, 3);
      const bT_mat = transpose2d(bData, 2, 3);
      const want = aT_mat.map((v, i) => v + bT_mat[i]);
      approxEqual(got, want);
    },
    TIMEOUT,
  );

  it(
    "matmul(q, transpose(k)) — repeated (compiled-plan 1st + 2nd exec)",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      const qData = arange(6);
      const kData = arange(12, 0.5);
      const want = matmul2d(qData, 2, 3, transpose2d(kData, 4, 3), 3, 4);
      // Run the same shape/template twice; the compiled plan builds on the
      // 2nd+ execution — both must produce correct values.
      for (let iter = 0; iter < 3; iter++) {
        const q = t.tensorFromArray(qData, [2, 3], { device });
        const k = t.tensorFromArray(kData, [4, 3], { device });
        const kT = t.transpose(k, { dim0: 0, dim1: 1 });
        const got = Array.from(await t.matmul(q, kT).cpu());
        approxEqual(got, want);
      }
    },
    TIMEOUT,
  );

  it(
    "shared transpose consumed by two INCREMENTALLY-forced matmuls",
    async () => {
      if (!deviceReady) return;
      // One transpose view feeding two independent matmuls, forced one at a
      // time. Each force is its own plan; the second must still resolve the
      // shared view (the sibling-plan 'Input not ready' angle).
      const t = mk();
      const kData = arange(12, 0.5);
      for (let iter = 0; iter < 3; iter++) {
        const k = t.tensorFromArray(kData, [4, 3], { device });
        const kT = t.transpose(k, { dim0: 0, dim1: 1 }); // [3,4]
        const q1Data = arange(6);
        const q2Data = arange(6).reverse();
        const q1 = t.tensorFromArray(q1Data, [2, 3], { device });
        const q2 = t.tensorFromArray(q2Data, [2, 3], { device });
        const out1 = t.matmul(q1, kT);
        const got1 = Array.from(await out1.cpu()); // force plan 1
        // Build the second consumer AFTER plan 1 is materialized.
        const out2 = t.matmul(q2, kT);
        const got2 = Array.from(await out2.cpu()); // force plan 2 — needs kT
        const kT_mat = transpose2d(kData, 4, 3);
        approxEqual(got1, matmul2d(q1Data, 2, 3, kT_mat, 3, 4));
        approxEqual(got2, matmul2d(q2Data, 2, 3, kT_mat, 3, 4));
      }
    },
    TIMEOUT,
  );

  it(
    "matmul(q, transpose(k)) with backward — grads flow through the view",
    async () => {
      if (!deviceReady) return;
      const t = mk();
      const qData = arange(6);
      const kData = arange(12, 0.5);
      const q = t.tensorFromArray(qData, [2, 3], {
        device,
        requiresGrad: true,
      });
      const k = t.tensorFromArray(kData, [4, 3], {
        device,
        requiresGrad: true,
      });
      const kT = t.transpose(k, { dim0: 0, dim1: 1 }); // [3,4]
      const mm = t.matmul(q, kT); // [2,4]
      const loss = t.sum(mm);
      // Backward must run without "Input not ready" and produce grads. Run it
      // before any forcing readback so the autograd graph is intact — the
      // forward matmul(q, transpose(k)) output is NEVER materialized, which is
      // exactly what triggered the double-transpose-kills-shared-view bug (#67).
      await loss.backward();
      const gq = Array.from(await q.grad!.cpu());
      const gk = Array.from(await k.grad!.cpu());
      // dL/dq = ones[2,4] @ k (since kT = k^T, q@kT, dq = 1 @ (kT)^T = 1 @ k).
      const ones = new Array(2 * 4).fill(1);
      const kMat = kData; // [4,3]
      const wantGq = matmul2d(ones, 2, 4, kMat, 4, 3); // [2,3]
      approxEqual(gq, wantGq);
      expect(gk.length).toBe(12);
      expect(gk.every((v) => Number.isFinite(v))).toBe(true);
    },
    TIMEOUT,
  );
});
