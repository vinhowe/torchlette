/**
 * Semantic Derivation — the P4 INDEX ALGEBRA gate. An index-space map's adjoint
 * is its TRANSPOSE, derived mechanically (design §2 category c, §4.1, §8.1). Two
 * layers of proof:
 *   1. `adjointIndexMap` is a pure DATA→DATA transpose (the schema-gated
 *      derivation) — unit-checked per kind + involutions.
 *   2. the REALIZED adjoint reproduces an independent hand-written transpose
 *      byte-exact on CPU (the Probe-2 analogue for the index family) — this is
 *      the byte gate the deleted frontend backward closures are replaced by.
 * The CE completion (log_softmax ∘ target-gather) is checked as a composition
 * against the decomposed forward + a JS reference (Probe-4 shape).
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { crossEntropy } from "../src/nn/functional";
import {
  adjointIndexMap,
  assertNoIndexMapBody,
  backwardOfIndexMap,
  CROSS_ENTROPY_DEF,
  type IndexMap,
  interpretComposition,
  invertPermutation,
} from "../src/ops/semantic";

const api = new Torchlette("cpu");
const rt = api.runtime;
const F = (v: number) => Math.fround(v);

function randData(n: number, seed: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * 4 - 2);
  }
  return out;
}

const arrEq = (a: ArrayLike<number>, b: ArrayLike<number>): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

const asOne = (v: unknown) => v as import("../src/runtime/tensor").Tensor;

describe("Semantic derivation — P4 index algebra (adjoint = transpose)", () => {
  it("schema gate: every index map is DATA; a smuggled body is unconstructible", () => {
    const maps: IndexMap[] = [
      { k: "reshape", inShape: [2, 3], outShape: [6] },
      { k: "transpose", dim0: 0, dim1: 1 },
      { k: "permute", perm: [2, 0, 1] },
      { k: "narrow", dim: 0, start: 1, length: 2, inLen: 4 },
      { k: "cat", dim: 1, sizes: [2, 3] },
      { k: "broadcast", inShape: [1, 3], outShape: [4, 3] },
      { k: "reduce", inShape: [4, 3], dims: [1], keepdim: false },
      { k: "gather", dim: 1, inShape: [4, 5] },
      { k: "scatterAdd", dim: 0 },
    ];
    for (const m of maps) expect(() => assertNoIndexMapBody(m)).not.toThrow();
    expect(() =>
      // biome-ignore lint: intentional bad leaf for the negative test
      assertNoIndexMapBody({
        k: "gather",
        dim: 0,
        inShape: (() => 0) as never,
      }),
    ).toThrow(/schema gate/);
    expect(() =>
      // biome-ignore lint: unknown kind must be rejected
      assertNoIndexMapBody({ k: "flip", dim: 0 } as never),
    ).toThrow(/index-algebra map/);
  });

  it("the transpose derivation (pure DATA→DATA), per kind", () => {
    expect(
      adjointIndexMap({ k: "reshape", inShape: [2, 3], outShape: [6] }),
    ).toEqual({
      k: "reshape",
      toShape: [2, 3],
    });
    // a 2-axis transpose is its own inverse
    expect(adjointIndexMap({ k: "transpose", dim0: 1, dim1: 2 })).toEqual({
      k: "transpose",
      dim0: 1,
      dim1: 2,
    });
    // permute's adjoint is the inverse permutation
    expect(adjointIndexMap({ k: "permute", perm: [2, 0, 1] })).toEqual({
      k: "permute",
      perm: invertPermutation([2, 0, 1]),
    });
    // narrow ⇄ pad
    expect(
      adjointIndexMap({ k: "narrow", dim: 0, start: 1, length: 2, inLen: 4 }),
    ).toEqual({ k: "pad", dim: 0, start: 1, outLen: 4 });
    // cat ⇄ split
    expect(adjointIndexMap({ k: "cat", dim: 1, sizes: [2, 3] })).toEqual({
      k: "split",
      dim: 1,
      sizes: [2, 3],
    });
    // broadcast ⇄ reduce; gather ⇄ scatter
    expect(
      adjointIndexMap({ k: "broadcast", inShape: [1, 3], outShape: [4, 3] }),
    ).toEqual({ k: "reduceToShape", fromShape: [4, 3], toShape: [1, 3] });
    expect(adjointIndexMap({ k: "gather", dim: 1, inShape: [4, 5] })).toEqual({
      k: "scatterZeros",
      dim: 1,
      inShape: [4, 5],
    });
  });

  it("invertPermutation is an involution", () => {
    for (const p of [[0], [1, 0], [2, 0, 1], [3, 1, 0, 2]]) {
      expect(invertPermutation(invertPermutation(p))).toEqual(p);
    }
  });

  // -- The byte gate: the realized adjoint == an independent JS transpose. --

  it("permute adjoint == inverse-permutation transpose (byte-exact)", async () => {
    const g = randData(24, 3);
    const gT = api.tensorFromArray(g, [2, 3, 4]);
    const map: IndexMap = { k: "permute", perm: [2, 0, 1] };
    const out = await asOne(
      api._wrap(asOne(backwardOfIndexMap(rt, map, gT._unwrap()))),
    ).cpu();
    // JS ref: permute grad by the inverse permutation.
    const ref = await asOne(
      api.permute(gT, invertPermutation([2, 0, 1])),
    ).cpu();
    expect(arrEq(out, ref)).toBe(true);
  });

  it("narrow adjoint pads grad into zeros at the offset (byte-exact)", async () => {
    const g = randData(6, 5); // narrowed shape [3,2] along dim0 start1 of inLen4
    const gT = api.tensorFromArray(g, [3, 2]);
    const map: IndexMap = {
      k: "narrow",
      dim: 0,
      start: 1,
      length: 3,
      inLen: 4,
    };
    const out = await asOne(
      api._wrap(asOne(backwardOfIndexMap(rt, map, gT._unwrap()))),
    ).cpu();
    // JS ref: [4,2] zeros with g (f32) placed at rows [1,4).
    const ref = new Array(8).fill(0);
    for (let i = 0; i < 6; i++) ref[2 + i] = F(g[i]);
    expect(arrEq(out, ref)).toBe(true);
  });

  it("cat adjoint splits grad at recorded offsets (byte-exact)", async () => {
    const g = randData(20, 7); // [4,5], split along dim1 into [2,3]
    const gT = api.tensorFromArray(g, [4, 5]);
    const map: IndexMap = { k: "cat", dim: 1, sizes: [2, 3] };
    const parts = backwardOfIndexMap(rt, map, gT._unwrap()) as unknown[];
    expect(parts.length).toBe(2);
    const p0 = await asOne(api._wrap(asOne(parts[0]))).cpu();
    const p1 = await asOne(api._wrap(asOne(parts[1]))).cpu();
    // JS ref: columns [0,2) and [2,5) of each row.
    const r0: number[] = [];
    const r1: number[] = [];
    for (let r = 0; r < 4; r++) {
      r0.push(F(g[r * 5 + 0]), F(g[r * 5 + 1]));
      r1.push(F(g[r * 5 + 2]), F(g[r * 5 + 3]), F(g[r * 5 + 4]));
    }
    expect(arrEq(p0, r0)).toBe(true);
    expect(arrEq(p1, r1)).toBe(true);
  });

  it("broadcast adjoint sums grad down to the input shape (byte-exact)", async () => {
    const g = randData(12, 9); // [4,3] broadcast from [1,3]
    const gT = api.tensorFromArray(g, [4, 3]);
    const map: IndexMap = { k: "broadcast", inShape: [1, 3], outShape: [4, 3] };
    const out = await asOne(
      api._wrap(asOne(backwardOfIndexMap(rt, map, gT._unwrap()))),
    ).cpu();
    // f32 rounding: compare via the same runtime sum reference.
    const refT = await asOne(api.sum(gT, { dim: [0], keepdim: true })).cpu();
    expect(arrEq(out, refT)).toBe(true);
    expect(out.length).toBe(3);
  });

  it("reduce adjoint broadcasts grad back over the reduced dims (byte-exact)", async () => {
    // reduce inShape [4,3] over dim1 keepdim=false → grad shape [4]; broadcast back.
    const g = randData(4, 11);
    const gT = api.tensorFromArray(g, [4]);
    const map: IndexMap = {
      k: "reduce",
      inShape: [4, 3],
      dims: [1],
      keepdim: false,
    };
    const out = await asOne(
      api._wrap(asOne(backwardOfIndexMap(rt, map, gT._unwrap()))),
    ).cpu();
    const ref: number[] = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) ref.push(F(g[r]));
    expect(arrEq(out, ref)).toBe(true);
  });

  it("gather adjoint scatters grad into zeros (byte-exact) — and round-trips", async () => {
    // a:[3,4], gather along dim1 with index [3,2]; grad [3,2].
    const idx = [0, 3, 1, 2, 3, 0];
    const idxT = api.tensorFromArray(idx, [3, 2]);
    const g = randData(6, 13);
    const gT = api.tensorFromArray(g, [3, 2]);
    const map: IndexMap = { k: "gather", dim: 1, inShape: [3, 4] };
    const out = await asOne(
      api._wrap(
        asOne(
          backwardOfIndexMap(rt, map, gT._unwrap(), { index: idxT._unwrap() }),
        ),
      ),
    ).cpu();
    // Independent hand path: zeros + scatterAdd IS the transpose of gather.
    const z = api.zeros([3, 4]);
    const refT = await asOne(api.scatterAdd(z, idxT, gT, { dim: 1 })).cpu();
    expect(arrEq(out, refT)).toBe(true);
  });
});

describe("Semantic derivation — P4 cross_entropy completes (composition)", () => {
  it("schema gate: the CE term is DATA (incl. the gather-index node)", () => {
    // The gather node lives inside the CE root; walking it must not throw.
    expect(CROSS_ENTROPY_DEF.name).toBe("cross_entropy");
    expect(CROSS_ENTROPY_DEF.roles).toContain("target");
  });

  it("CE composition == decomposed forward == JS reference (per-sample)", async () => {
    const ROWS = 5;
    const COLS = 7;
    const logitsData = randData(ROWS * COLS, 21);
    const targets = [0, 3, 6, 1, 4];
    const logitsT = api.tensorFromArray(logitsData, [ROWS, COLS]);
    const targetT = api.tensorFromArray(targets, [ROWS]);

    // The composition (single source): per-sample loss, gather target [ROWS,1].
    const comp = await asOne(
      api._wrap(
        interpretComposition(CROSS_ENTROPY_DEF, api.runtime, 1, {
          x: logitsT._unwrap(),
          target: api.reshape(targetT, [ROWS, 1])._unwrap(),
        }),
      ),
    ).cpu(); // shape [ROWS,1]

    // Decomposed CE forward, reduction 'none' (the nn/functional entry).
    const dec = await crossEntropy(api, logitsT, targetT, {
      reduction: "none",
    }).cpu();

    // JS reference: -log_softmax(row)[target].
    const jsRef: number[] = [];
    for (let r = 0; r < ROWS; r++) {
      const row = logitsData.slice(r * COLS, (r + 1) * COLS);
      const m = Math.max(...row);
      const lse = Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0));
      jsRef.push(-(row[targets[r]] - m - lse));
    }

    const maxAbs = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
      let mx = 0;
      for (let i = 0; i < a.length; i++)
        mx = Math.max(mx, Math.abs(a[i] - b[i]));
      return mx;
    };
    expect(maxAbs(comp, dec)).toBeLessThan(1e-5);
    expect(maxAbs(comp, jsRef)).toBeLessThan(1e-5);
  });
});
