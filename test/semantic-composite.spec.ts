/**
 * Semantic Derivation — the P2 reduction-composite reference gate (Probe-4
 * shape, design §4.4). Each composite's DEFINITION is a `Composition` DATA term
 * (ops/semantic/composite.ts); this gate proves the hand decomposed/fused
 * forward AGREES with that single source, on CPU. The fused kernel is NOT
 * re-derived (§4.4) — the composition is its REFERENCE, met at the schedule-
 * state SemanticRegionUid seam and checked there by the existing differential.
 *
 * Two independent sources: the op forward (api.softmax / .rmsnorm / .layernorm)
 * and a plain-JS row-wise reference — both compared against the ONE composition
 * interpreted over the runtime engine.
 */

import { describe, expect, it } from "vitest";
import {
  assertNoCompositionBody,
  interpretComposition,
  LAYERNORM_DEF,
  LOG_SOFTMAX_DEF,
  REDUCTION_COMPOSITE_DEFS,
  RMSNORM_DEF,
  SIMPLIFICATION_LEMMAS,
  SOFTMAX_DEF,
} from "../src/ops/semantic";
import { Torchlette } from "../src/frontend/torchlette";

const api = new Torchlette("cpu");

// Deterministic pseudo-random inputs.
function randData(n: number, seed: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * 4 - 2); // [-2, 2)
  }
  return out;
}

const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

// Plain-JS row-wise references (the independent second source), f32-rounded.
const F = (v: number) => Math.fround(v);
function rowsOf(data: number[], rows: number, cols: number): number[][] {
  const r: number[][] = [];
  for (let i = 0; i < rows; i++) r.push(data.slice(i * cols, (i + 1) * cols));
  return r;
}
function jsSoftmax(data: number[], rows: number, cols: number): number[] {
  const out: number[] = [];
  for (const row of rowsOf(data, rows, cols)) {
    const m = Math.max(...row);
    const ex = row.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    for (const e of ex) out.push(F(e / s));
  }
  return out;
}
function jsLogSoftmax(data: number[], rows: number, cols: number): number[] {
  const out: number[] = [];
  for (const row of rowsOf(data, rows, cols)) {
    const m = Math.max(...row);
    const lse = Math.log(row.reduce((a, b) => a + Math.exp(b - m), 0));
    for (const v of row) out.push(F(v - m - lse));
  }
  return out;
}
function jsRmsnorm(
  data: number[],
  w: number[],
  rows: number,
  cols: number,
  eps: number,
): number[] {
  const out: number[] = [];
  for (const row of rowsOf(data, rows, cols)) {
    const ms = row.reduce((a, v) => a + v * v, 0) / cols;
    const inv = 1 / Math.sqrt(ms + eps);
    row.forEach((v, j) => out.push(F(v * inv * w[j])));
  }
  return out;
}
function jsLayernorm(
  data: number[],
  w: number[],
  b: number[],
  rows: number,
  cols: number,
  eps: number,
): number[] {
  const out: number[] = [];
  for (const row of rowsOf(data, rows, cols)) {
    const mean = row.reduce((a, v) => a + v, 0) / cols;
    const varr = row.reduce((a, v) => a + (v - mean) ** 2, 0) / cols;
    const std = Math.sqrt(varr + eps);
    row.forEach((v, j) => out.push(F(((v - mean) / std) * w[j] + b[j])));
  }
  return out;
}

const ROWS = 5;
const COLS = 8;
const N = ROWS * COLS;
const DIM = 1; // last dim (rank 2)

describe("Semantic derivation — P2 reduction composites (Probe-4 reference)", () => {
  it("schema gate: every reduction-composite term is DATA", () => {
    for (const d of REDUCTION_COMPOSITE_DEFS) {
      expect(() => assertNoCompositionBody(d.root)).not.toThrow();
    }
    // The declared simplification lemmas (softmax backward, T1) are DATA too.
    for (const l of SIMPLIFICATION_LEMMAS) {
      expect(() => assertNoCompositionBody(l.root)).not.toThrow();
    }
    // A smuggled JS-body leaf must be unconstructible.
    expect(() =>
      // biome-ignore lint: intentional bad leaf for the negative test
      assertNoCompositionBody({ k: "u", op: "exp", a: (() => 0) as never }),
    ).toThrow(/schema gate/);
  });

  it("softmax: composition == op forward AND == the JS reference", async () => {
    const xData = randData(N, 11);
    const xT = api.tensorFromArray(xData, [ROWS, COLS]);
    const op = await api.softmax(xT, -1).cpu();
    const comp = await api
      ._wrap(interpretComposition(SOFTMAX_DEF, api.runtime, DIM, { x: xT._unwrap() }))
      .cpu();
    const js = jsSoftmax(xData, ROWS, COLS);
    expect(maxAbs(comp, op)).toBeLessThan(1e-6);
    expect(maxAbs(comp, js)).toBeLessThan(1e-6);
  });

  it("log_softmax: composition == the JS reference", async () => {
    const xData = randData(N, 23);
    const xT = api.tensorFromArray(xData, [ROWS, COLS]);
    const comp = await api
      ._wrap(
        interpretComposition(LOG_SOFTMAX_DEF, api.runtime, DIM, {
          x: xT._unwrap(),
        }),
      )
      .cpu();
    expect(maxAbs(comp, jsLogSoftmax(xData, ROWS, COLS))).toBeLessThan(1e-5);
  });

  it("rmsnorm: composition == op forward AND == the JS reference", async () => {
    const eps = RMSNORM_DEF.eps as number;
    const xData = randData(N, 37);
    const wData = randData(COLS, 41);
    const xT = api.tensorFromArray(xData, [ROWS, COLS]);
    const wT = api.tensorFromArray(wData, [COLS]);
    const op = await xT.rmsnorm(wT, eps).cpu();
    const comp = await api
      ._wrap(
        interpretComposition(RMSNORM_DEF, api.runtime, DIM, {
          x: xT._unwrap(),
          w: wT._unwrap(),
          eps,
        }),
      )
      .cpu();
    const js = jsRmsnorm(xData, wData, ROWS, COLS, eps);
    expect(maxAbs(comp, op)).toBeLessThan(1e-6);
    expect(maxAbs(comp, js)).toBeLessThan(1e-5);
  });

  it("layernorm: composition == op forward AND == the JS reference", async () => {
    const eps = LAYERNORM_DEF.eps as number;
    const xData = randData(N, 53);
    const wData = randData(COLS, 59);
    const bData = randData(COLS, 61);
    const xT = api.tensorFromArray(xData, [ROWS, COLS]);
    const wT = api.tensorFromArray(wData, [COLS]);
    const bT = api.tensorFromArray(bData, [COLS]);
    const op = await xT.layernorm(wT, bT, eps).cpu();
    const comp = await api
      ._wrap(
        interpretComposition(LAYERNORM_DEF, api.runtime, DIM, {
          x: xT._unwrap(),
          w: wT._unwrap(),
          b: bT._unwrap(),
          eps,
        }),
      )
      .cpu();
    const js = jsLayernorm(xData, wData, bData, ROWS, COLS, eps);
    expect(maxAbs(comp, op)).toBeLessThan(1e-6);
    expect(maxAbs(comp, js)).toBeLessThan(1e-5);
  });
});
