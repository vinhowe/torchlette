/**
 * Semantic Derivation — the REDUCTION monoid differential (Crystal Campaign 3,
 * Phase 1). The productized reduction analogue of semantic-derivation.spec.ts:
 * the derived monoid (identity + combine + epilogue, from the ONE definition)
 * reproduces an INDEPENDENT hand reduce loop BYTE-for-byte, so the "one source"
 * claim for reductions is a standing gate.
 *
 * S1 (reference): derived reduce(identity, combine[, epilogue]) == a hand loop
 *   over random tensors, byte-exact (f32 store-once discipline).
 * Monoid projection: reduceMonoidOf(def) == the declared label.
 * Indexed monoid: the derived `isBetter` == the hand argmax/argmin comparison.
 * Schema gate: every reduction definition is DATA (no smuggled loop body).
 *
 * The hand loops below are transcribed INDEPENDENTLY of numeric.ts, so this is a
 * genuine two-source differential, not a self-comparison.
 */

import { describe, expect, it } from "vitest";
import {
  ARGMAX_DEF,
  ARGMIN_DEF,
  assertNoReductionDefinitionBody,
  compileArgBetter,
  compileReduceCombine,
  compileReduceEpilogue,
  isStreamableMonoid,
  MAX_DEF,
  MEAN_DEF,
  MIN_DEF,
  REDUCTION_DEFS,
  reduceIdentity,
  reduceMonoidOf,
  SUM_DEF,
} from "../src/ops/semantic";

// f32 byte comparison (numeric.ts store-once-round discipline).
const F = new Float32Array(1);
const U = new Uint32Array(F.buffer);
const bits = (v: number) => {
  F[0] = v;
  return U[0];
};
const byteEqual = (a: number, b: number) => {
  F[0] = a;
  const fa = F[0];
  F[0] = b;
  const fb = F[0];
  return bits(fa) === bits(fb) || (Number.isNaN(fa) && Number.isNaN(fb));
};

// Independent hand reduce loops (transcribed from the reduction MEANING, NOT
// from numeric.ts) — the second source the derived monoid is diffed against.
const handSum = (v: number[]) => {
  const out = new Float32Array(1); // seed 0
  for (const e of v) out[0] = out[0] + e;
  return out[0];
};
const handMean = (v: number[]) => {
  const out = new Float32Array(1);
  for (const e of v) out[0] = out[0] + e;
  out[0] = out[0] / v.length;
  return out[0];
};
const handMax = (v: number[]) => {
  const out = new Float32Array(1);
  out[0] = Number.NEGATIVE_INFINITY;
  for (const e of v) if (e > out[0]) out[0] = e; // left-biased on ties
  return out[0];
};
const handMin = (v: number[]) => {
  const out = new Float32Array(1);
  out[0] = Number.POSITIVE_INFINITY;
  for (const e of v) if (e < out[0]) out[0] = e;
  return out[0];
};

// A derived reduce over the monoid definition (the derived S1 body shape).
function derivedReduce(
  def: (typeof REDUCTION_DEFS)[number],
  values: number[],
): number {
  const combine = compileReduceCombine(def);
  const epilogue = compileReduceEpilogue(def);
  const out = new Float32Array(1);
  out[0] = reduceIdentity(def);
  for (const e of values) out[0] = combine(out[0], e);
  if (epilogue) out[0] = epilogue(out[0], values.length);
  return out[0];
}

const HAND: Record<string, (v: number[]) => number> = {
  sum: handSum,
  mean: handMean,
  max: handMax,
  min: handMin,
};

// Sample vectors, incl. ties, ±0, and mixed signs (the loop's tie policy is the
// discriminating case a bare max/min primitive would get wrong).
function sampleVectors(): number[][] {
  const vs: number[][] = [
    [3, 1, 4, 1, 5, 9, 2, 6],
    [-2.5, -2.5, 1.0, 1.0],
    [0.0, -0.0, 0.0, -0.0],
    [-0.0, 0.0],
    [7.3],
    [-1, -3, -2],
    [1e-4, -1e-4, 12.0, -8.0, 0.3],
  ];
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return (seed / 0x7fffffff) * 20 - 10;
  };
  for (let n = 0; n < 8; n++) {
    const len = 1 + ((seed >> 3) % 17);
    vs.push(Array.from({ length: Math.max(1, len) }, () => Math.fround(rnd())));
  }
  return vs;
}

describe("Semantic reduction — the monoid is the single source", () => {
  it("schema gate: every reduction definition is DATA (no body leaf)", () => {
    for (const d of REDUCTION_DEFS) {
      expect(() => assertNoReductionDefinitionBody(d)).not.toThrow();
    }
    // A smuggled JS combine must be unconstructible.
    expect(() =>
      assertNoReductionDefinitionBody({
        ...SUM_DEF,
        // biome-ignore lint: intentional bad leaf for the negative test
        combine: { k: "add", a: { k: "x" }, b: ((v: number) => v) as never },
      }),
    ).toThrow(/schema gate/);
  });

  it("monoid projection: reduceMonoidOf reads the combine's root", () => {
    expect(reduceMonoidOf(SUM_DEF)).toBe("sum");
    expect(reduceMonoidOf(MEAN_DEF)).toBe("sum"); // mean is the sum monoid + epilogue
    expect(reduceMonoidOf(MAX_DEF)).toBe("max");
    expect(reduceMonoidOf(MIN_DEF)).toBe("min");
    expect(reduceMonoidOf(ARGMAX_DEF)).toBe("max"); // value monoid of argmax
    expect(reduceMonoidOf(ARGMIN_DEF)).toBe("min");
  });

  it("streamability: every monoid streams (referenced two-stage form)", () => {
    for (const d of REDUCTION_DEFS) expect(isStreamableMonoid(d)).toBe(true);
  });

  describe("S1 — derived monoid reduce reproduces the hand loop byte-for-byte", () => {
    for (const d of [SUM_DEF, MEAN_DEF, MAX_DEF, MIN_DEF]) {
      it(`${d.name}`, () => {
        for (const v of sampleVectors()) {
          expect(byteEqual(derivedReduce(d, v), HAND[d.name](v))).toBe(true);
        }
      });
    }
  });

  describe("indexed monoid — derived isBetter reproduces the arg comparison", () => {
    for (const [def, cmp] of [
      [ARGMAX_DEF, (e: number, b: number) => e > b],
      [ARGMIN_DEF, (e: number, b: number) => e < b],
    ] as const) {
      it(`${def.name}`, () => {
        const better = compileArgBetter(def);
        // Simulate the arg-reduce scan: seed=identity, track first winner.
        for (const v of sampleVectors()) {
          let bestD = reduceIdentity(def);
          let idxD = 0;
          let bestH = reduceIdentity(def);
          let idxH = 0;
          v.forEach((e, i) => {
            if (better(e, bestD)) {
              bestD = e;
              idxD = i;
            }
            if (cmp(e, bestH)) {
              bestH = e;
              idxH = i;
            }
          });
          expect(idxD).toBe(idxH);
        }
      });
    }
  });
});
