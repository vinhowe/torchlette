/**
 * GATE (task #80 inc-2a, "capturable-optimizer contract"): the in-kernel Adam
 * bias-correction derivation from an on-device step counter `t` must match the
 * f64 reference to the trajectory band.
 *
 * inc-2a moves the bias-corrected `step_size` OUT of a JS-computed node payload
 * and INTO the fused WGSL kernel (and the foreach/elementwise graph paths),
 * derived from a persistent on-device `t` tensor. The probe (design doc
 * `docs/staged-execution-phase2b.md`, inc-2a section) measured that the NAIVE
 * form `1 - pow(beta, t)` sits AT the 1e-5 trajectory band at small t
 * (catastrophic cancellation — worst 1.95e-5 rel err @ t=2), while the
 * `-expm1(t*ln beta)` form with a 5-term Horner series for |y| < 0.25 measured
 * 6.0e-7. This test is the executable form of that measurement: it emulates the
 * exact WGSL f32 arithmetic (Math.fround chains) the kernel emits and asserts
 * rel err < 2e-6 vs f64, AND that the naive form would fail — so a regression
 * to the naive derivation cannot ship silently.
 *
 * The emulated `expm1F32` / `biasCorrectedStepSizeF32` below MUST mirror
 * `src/backend/webgpu/adam-kernel.ts`'s in-kernel derivation exactly. If the
 * kernel formula changes, change this emulation with it (single source at the
 * seam: the WGSL and this emulation are two sides that must agree).
 */
import { describe, expect, it } from "vitest";

const f = Math.fround;

/**
 * Emulated WGSL f32 expm1(y) for y = t * ln(beta) (y <= 0).
 * |y| < 0.25: 5-term Horner series y*(1 + y*(1/2 + y*(1/6 + y*(1/24 + y/120)))).
 * else:       exp(y) - 1.
 */
function expm1F32(y: number): number {
  const ay = Math.abs(y);
  if (ay < 0.25) {
    let r = f(1 / 120);
    r = f(f(1 / 24) + f(y * r));
    r = f(f(1 / 6) + f(y * r));
    r = f(f(1 / 2) + f(y * r));
    r = f(1 + f(y * r));
    return f(y * r);
  }
  return f(Math.exp(y) - 1);
}

/**
 * Emulated kernel derivation. `lnBeta*` are precomputed f64->f32 on the CPU
 * (static uniforms); everything else is f32.
 */
function biasCorrectedStepSizeF32(
  t: number,
  lr: number,
  beta1: number,
  beta2: number,
): { stepSize: number; bc1: number; bc2: number } {
  const lnB1 = f(Math.log(beta1));
  const lnB2 = f(Math.log(beta2));
  const bc1 = f(-expm1F32(f(t * lnB1)));
  const bc2 = f(-expm1F32(f(t * lnB2)));
  const sqrtBc2 = f(Math.sqrt(bc2));
  return { stepSize: f(f(lr * sqrtBc2) / bc1), bc1, bc2 };
}

/** Naive f32 `1 - pow(beta, t)` (the form the kernel must NOT use). */
function bcNaiveF32(t: number, beta: number): number {
  return f(1 - f(Math.pow(f(beta), f(t))));
}

function refStepSize(
  t: number,
  lr: number,
  beta1: number,
  beta2: number,
): { stepSize: number; bc1: number; bc2: number } {
  const bc1 = 1 - beta1 ** t;
  const bc2 = 1 - beta2 ** t;
  return { stepSize: (lr * Math.sqrt(bc2)) / bc1, bc1, bc2 };
}

const T_VALUES = [1, 2, 3, 10, 100, 1000, 10000, 200000];
const LR = 1e-3;
const B1 = 0.9;
const B2 = 0.999;

describe("Adam in-kernel bias-correction derivation (inc-2a)", () => {
  it("expm1 form matches f64 to < 2e-6 rel err across t=1..200000", () => {
    let worstSS = 0;
    let worstBc2 = 0;
    for (const t of T_VALUES) {
      const a = biasCorrectedStepSizeF32(t, LR, B1, B2);
      const r = refStepSize(t, LR, B1, B2);
      const eSS = Math.abs(a.stepSize - r.stepSize) / Math.abs(r.stepSize);
      const eBc2 = Math.abs(a.bc2 - r.bc2) / Math.abs(r.bc2);
      worstSS = Math.max(worstSS, eSS);
      worstBc2 = Math.max(worstBc2, eBc2);
    }
    expect(worstSS).toBeLessThan(2e-6);
    expect(worstBc2).toBeLessThan(2e-6);
  });

  it("naive 1-pow(beta,t) form would FAIL the band (why expm1 is required)", () => {
    let worstNaive = 0;
    for (const t of T_VALUES) {
      const r = refStepSize(t, LR, B1, B2);
      const nb2 = bcNaiveF32(t, B2);
      worstNaive = Math.max(worstNaive, Math.abs(nb2 - r.bc2) / Math.abs(r.bc2));
    }
    // The naive form's worst error is ~1.95e-5 @ t=2 — above the 2e-6 gate.
    expect(worstNaive).toBeGreaterThan(2e-6);
  });
});
