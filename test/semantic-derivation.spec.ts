/**
 * Semantic Derivation — the productized Probe 2/3 differential (Crystal
 * Campaign 3, Phase 0). The design's feasibility probes
 * (tools/semantic-derivation-probe.ts) run here against the LANDED stratum
 * (src/ops/semantic), so the "one source" claim is a standing gate, not a
 * one-off tool that can rot.
 *
 * S1 (reference): the derived CPU body (interpret of the definition) reproduces
 *   the hand `numeric.ts` bodies byte-for-byte.
 * S2 (adjoint):   the derived VJP (adjoint of the definition, normalized, with
 *   guards) reproduces the hand `registry.ts` grad table byte-for-byte — the 3
 *   probe divergences RESOLVED (div.dB via the conservative normalizer; log/sqrt
 *   via the explicit denomEps guard).
 * Schema gate:    every definition is DATA (no smuggled op body).
 *
 * The hand surfaces below are transcribed INDEPENDENTLY (as functions / Exprs),
 * so this is a genuine two-source differential, not a self-comparison.
 */

import { describe, expect, it } from "vitest";
import {
  assertNoDefinitionBody,
  BINARY_DEFS,
  compileBinary,
  compileUnary,
  evalScalar,
  UNARY_DEFS,
  vjpBinary,
  vjpUnary,
} from "../src/ops/semantic";
import {
  add,
  c,
  cos,
  div,
  type Expr,
  exp,
  g,
  ge,
  gt,
  le,
  lt,
  mul,
  neg,
  recip,
  sign,
  sin,
  sqrt,
  sub,
  tanh,
  x,
  y,
} from "../src/ops/semantic/expr";

// f32 byte comparison helpers (numeric.ts store-once-round discipline).
const f32 = (v: number) => Math.fround(v);
const F = new Float32Array(1);
const U = new Uint32Array(F.buffer);
const bits = (v: number) => {
  F[0] = v;
  return U[0];
};
const byteEqual = (a: number, b: number) => {
  const fa = f32(a);
  const fb = f32(b);
  return bits(fa) === bits(fb) || (Number.isNaN(fa) && Number.isNaN(fb));
};

// Hand CPU bodies — transcribed from src/backend/cpu/numeric.ts UNARY_OPS/BINARY_OPS.
const HAND_UNARY: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt,
  exp: Math.exp,
  log: Math.log,
  neg: (x) => -x,
  abs: Math.abs,
  tanh: Math.tanh,
  sigmoid: (x) => 1.0 / (1.0 + Math.exp(-x)),
  silu: (x) => x / (1.0 + Math.exp(-x)),
  sin: Math.sin,
  cos: Math.cos,
  rsqrt: (x) => 1.0 / Math.sqrt(x),
  floor: Math.floor,
  ceil: Math.ceil,
  round: Math.round,
  sign: Math.sign,
  isfinite: (x) => (Number.isFinite(x) ? 1.0 : 0.0),
  relu: (x) => (x > 0 ? x : 0),
};
const HAND_BINARY: Record<string, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  sub: (x, y) => x - y,
  mul: (x, y) => x * y,
  div: (x, y) => x / y,
  pow: Math.pow,
  minimum: Math.min,
  maximum: Math.max,
};

// Hand grad table — transcribed from src/ops/registry.ts, as Exprs.
const sig = recip(add(c(1), exp(neg(x))));
const HAND_UNARY_GRAD: Record<string, Expr> = {
  relu: mul(g, gt(x, c(0))),
  silu: mul(g, add(sig, mul(x, mul(sig, sub(c(1), sig))))),
  sigmoid: mul(mul(sig, sub(c(1), sig)), g),
  tanh: mul(sub(c(1), mul(tanh(x), tanh(x))), g),
  neg: neg(g),
  abs: mul(g, sign(x)),
  exp: mul(g, exp(x)),
  log: div(g, add(x, c(1e-8))), // note the epsilon guard
  sqrt: mul(g, div(c(0.5), add(sqrt(x), c(1e-8)))), // note the epsilon guard
  rsqrt: mul(
    g,
    mul(c(-0.5), mul(recip(sqrt(x)), mul(recip(sqrt(x)), recip(sqrt(x))))),
  ),
  sin: mul(g, cos(x)),
  cos: mul(g, neg(sin(x))),
};
const HAND_BINARY_GRAD: Record<string, [Expr, Expr]> = {
  add: [g, g],
  mul: [mul(g, y), mul(g, x)],
  div: [div(g, y), mul(g, div(neg(x), mul(y, y)))],
  minimum: [mul(g, le(x, y)), mul(g, gt(x, y))],
  maximum: [mul(g, ge(x, y)), mul(g, lt(x, y))],
};

const UNARY_SAMPLES = [
  -8, -3.5, -1.0, -0.5, -0.0001, 0.0, 0.0001, 0.3, 0.7, 1.0, 2.5, 5.0, 7.3,
  12.0,
];
const BINARY_SAMPLES: [number, number][] = [];
for (const a of [-5, -1.3, -0.2, 0.0, 0.4, 1.1, 3.7]) {
  for (const b of [-4.2, -0.7, 0.5, 1.9, 6.1]) BINARY_SAMPLES.push([a, b]);
}
const GRAD_UPSTREAM = [1.0, -2.3, 0.5];
const posDomain = (op: string) =>
  op === "sqrt" || op === "rsqrt" || op === "log";

describe("Semantic derivation — the definition is the single source", () => {
  it("schema gate: every elementwise definition is DATA (no body leaf)", () => {
    for (const d of [...UNARY_DEFS, ...BINARY_DEFS]) {
      expect(() => assertNoDefinitionBody(d.expr)).not.toThrow();
    }
    // A smuggled JS-body leaf must be unconstructible.
    expect(() =>
      // biome-ignore lint: intentional bad leaf for the negative test
      assertNoDefinitionBody({ k: "neg", a: ((v: number) => v) as never }),
    ).toThrow(/schema gate/);
  });

  describe("S1 — derived CPU reference reproduces the hand body byte-for-byte", () => {
    for (const d of UNARY_DEFS) {
      it(`${d.name} (unary)`, () => {
        const body = compileUnary(d.expr);
        for (const xv of UNARY_SAMPLES) {
          if (posDomain(d.name) && xv <= 0) continue;
          expect(byteEqual(body(xv), HAND_UNARY[d.name](xv))).toBe(true);
        }
      });
    }
    for (const d of BINARY_DEFS) {
      it(`${d.name} (binary)`, () => {
        const body = compileBinary(d.expr);
        for (const [xv, yv] of BINARY_SAMPLES) {
          expect(byteEqual(body(xv, yv), HAND_BINARY[d.name](xv, yv))).toBe(
            true,
          );
        }
      });
    }
  });

  describe("S2 — derived adjoint reproduces the grad table byte-for-byte", () => {
    for (const d of UNARY_DEFS) {
      if (d.gradPolicy !== "derive") continue;
      const hand = HAND_UNARY_GRAD[d.name];
      if (!hand) continue;
      it(`${d.name} grad`, () => {
        const vjp = vjpUnary(d.expr, d.gradGuard);
        for (const xv of UNARY_SAMPLES) {
          if (posDomain(d.name) && xv <= 0) continue;
          for (const gv of GRAD_UPSTREAM) {
            const a = evalScalar(vjp, { x: xv, y: 0, g: gv });
            const b = evalScalar(hand, { x: xv, y: 0, g: gv });
            expect(byteEqual(a, b)).toBe(true);
          }
        }
      });
    }
    for (const d of BINARY_DEFS) {
      if (d.gradPolicy !== "derive") continue;
      const hand = HAND_BINARY_GRAD[d.name];
      if (!hand) continue;
      it(`${d.name} grad (dA, dB)`, () => {
        const [dA, dB] = vjpBinary(d.expr);
        for (const [idx, der] of [
          [0, dA],
          [1, dB],
        ] as [number, Expr][]) {
          for (const [xv, yv] of BINARY_SAMPLES) {
            for (const gv of GRAD_UPSTREAM) {
              const a = evalScalar(der, { x: xv, y: yv, g: gv });
              const b = evalScalar(hand[idx], { x: xv, y: yv, g: gv });
              expect(byteEqual(a, b)).toBe(true);
            }
          }
        }
      });
    }
  });
});
