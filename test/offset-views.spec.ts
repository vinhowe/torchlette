/**
 * Offset-view correctness (task #58) — the "silently wrong data" class.
 *
 * Views (narrow/slice) share a buffer with a nonzero element offset (+
 * strides). Elementwise kernels honor the offset via generated indexing, but
 * cpu() readback, contiguous(), and raw-bind kernel paths (matmul, fused
 * RoPE, reductions, cat, tril, ...) historically read from buffer element 0,
 * silently returning the WRONG REGION with the correct shape.
 *
 * This suite runs a battery of view shapes through every consumer seam and
 * asserts the result equals the same computation on an INDEPENDENT JS
 * reference (computed from the raw data + view metadata mirrored in plain
 * JS — not through the framework, so a shared bug can't self-validate).
 *
 * Includes the two known model-level victims specifically:
 *  - the RoPE-table-slice pattern (cos/sin = narrow of a persistent table);
 *  - the causal-mask-narrow pattern (mask = narrow of a persistent mask).
 */

import { beforeAll, describe, expect, it } from "vitest";
import type { DeviceKind } from "../src/backend/types";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { cpuOnly } from "./helpers/webgpu";

const TIMEOUT = 120_000;

// ---------------------------------------------------------------------------
// Independent JS reference: a strided view evaluator (second implementation
// of the view math — deliberately NOT importing view-meta.ts).
// ---------------------------------------------------------------------------

interface JsView {
  data: number[];
  shape: number[];
  strides: number[];
  offset: number;
}

function contiguousStrides(shape: number[]): number[] {
  const s = new Array(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

function jsBase(data: number[], shape: number[]): JsView {
  return { data, shape, strides: contiguousStrides(shape), offset: 0 };
}

function jsNarrow(
  v: JsView,
  dim: number,
  start: number,
  length: number,
): JsView {
  const shape = v.shape.slice();
  shape[dim] = length;
  return {
    data: v.data,
    shape,
    strides: v.strides.slice(),
    offset: v.offset + start * v.strides[dim],
  };
}

function jsTranspose(v: JsView, d0: number, d1: number): JsView {
  const shape = v.shape.slice();
  const strides = v.strides.slice();
  [shape[d0], shape[d1]] = [shape[d1], shape[d0]];
  [strides[d0], strides[d1]] = [strides[d1], strides[d0]];
  return { data: v.data, shape, strides, offset: v.offset };
}

/** Materialize a JsView to a flat row-major array. */
function jsRead(v: JsView): number[] {
  const size = v.shape.reduce((a, b) => a * b, 1);
  const out = new Array<number>(size);
  const rank = v.shape.length;
  const coords = new Array(rank).fill(0);
  for (let i = 0; i < size; i++) {
    let idx = v.offset;
    for (let d = 0; d < rank; d++) idx += coords[d] * v.strides[d];
    out[i] = v.data[idx];
    for (let d = rank - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < v.shape[d]) break;
      coords[d] = 0;
    }
  }
  return out;
}

function jsMatmul(
  a: number[],
  b: number[],
  m: number,
  k: number,
  n: number,
): number[] {
  const out = new Array<number>(m * n).fill(0);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += a[i * k + p] * b[p * n + j];
      out[i * n + j] = acc;
    }
  return out;
}

function maxAbsDiff(a: number[], b: number[]): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

function runSuite(device: DeviceKind) {
  const ATOL = device === "webgpu" ? 1e-4 : 1e-6;
  const api = () => new Torchlette(device);

  // Base tensor [4, 6], values 0..23 (every element distinct so a wrong
  // region can never accidentally match).
  const BASE_SHAPE = [4, 6];
  const BASE_DATA = Array.from({ length: 24 }, (_, i) => i);

  /** The view battery: framework builder + independent JS mirror. */
  const battery: Array<{
    name: string;
    // biome-ignore lint/suspicious/noExplicitAny: test tensors typed loosely
    build: (t: Torchlette) => any;
    ref: () => JsView;
  }> = [
    {
      name: "narrow(0, 2, 2) — contiguous strides, offset>0",
      build: (t) =>
        t.tensorFromArray(BASE_DATA, BASE_SHAPE, { device }).narrow(0, 2, 2),
      ref: () => jsNarrow(jsBase(BASE_DATA, BASE_SHAPE), 0, 2, 2),
    },
    {
      name: "narrow(1, 2, 3) — strided, offset>0",
      build: (t) =>
        t.tensorFromArray(BASE_DATA, BASE_SHAPE, { device }).narrow(1, 2, 3),
      ref: () => jsNarrow(jsBase(BASE_DATA, BASE_SHAPE), 1, 2, 3),
    },
    {
      name: "narrow(0,1,3).narrow(1,1,4) — chained narrows",
      build: (t) =>
        t
          .tensorFromArray(BASE_DATA, BASE_SHAPE, { device })
          .narrow(0, 1, 3)
          .narrow(1, 1, 4),
      ref: () =>
        jsNarrow(jsNarrow(jsBase(BASE_DATA, BASE_SHAPE), 0, 1, 3), 1, 1, 4),
    },
    {
      name: "transpose.narrow — strided transpose view with offset",
      build: (t) =>
        t
          .tensorFromArray(BASE_DATA, BASE_SHAPE, { device })
          .transpose({ dim0: 0, dim1: 1 })
          .narrow(0, 2, 3),
      ref: () =>
        jsNarrow(jsTranspose(jsBase(BASE_DATA, BASE_SHAPE), 0, 1), 0, 2, 3),
    },
    {
      name: "narrow.transpose — offset view then transpose",
      build: (t) =>
        t
          .tensorFromArray(BASE_DATA, BASE_SHAPE, { device })
          .narrow(0, 1, 2)
          .transpose({ dim0: 0, dim1: 1 }),
      ref: () =>
        jsTranspose(jsNarrow(jsBase(BASE_DATA, BASE_SHAPE), 0, 1, 2), 0, 1),
    },
  ];

  for (const { name, build, ref } of battery) {
    describe(name, () => {
      it(
        "cpu() returns the view's region",
        async () => {
          const t = api();
          const got = Array.from(await build(t).cpu());
          expect(maxAbsDiff(got, jsRead(ref()))).toBeLessThan(ATOL);
        },
        TIMEOUT,
      );

      it(
        "contiguous().cpu() returns the view's region",
        async () => {
          const t = api();
          const got = Array.from(await build(t).contiguous().cpu());
          expect(maxAbsDiff(got, jsRead(ref()))).toBeLessThan(ATOL);
        },
        TIMEOUT,
      );

      it(
        "elementwise (add 0) preserves the view's region",
        async () => {
          const t = api();
          const v = build(t);
          const zeros = t.tensorFromArray(
            new Array(jsRead(ref()).length).fill(0),
            ref().shape,
            { device },
          );
          const got = Array.from(await t.add(v, zeros).cpu());
          expect(maxAbsDiff(got, jsRead(ref()))).toBeLessThan(ATOL);
        },
        TIMEOUT,
      );

      it(
        "sum() reduces the view's region (full)",
        async () => {
          const t = api();
          const got = await t.sum(build(t)).item();
          const want = jsRead(ref()).reduce((a, b) => a + b, 0);
          expect(Math.abs(got - want)).toBeLessThan(ATOL * 10);
        },
        TIMEOUT,
      );

      it(
        "sum(dim=-1) reduces the view's region (dim)",
        async () => {
          const t = api();
          const got = Array.from(await t.sum(build(t), { dim: -1 }).cpu());
          const r = ref();
          const flat = jsRead(r);
          const inner = r.shape[r.shape.length - 1];
          const want: number[] = [];
          for (let i = 0; i < flat.length; i += inner) {
            let acc = 0;
            for (let j = 0; j < inner; j++) acc += flat[i + j];
            want.push(acc);
          }
          expect(maxAbsDiff(got, want)).toBeLessThan(ATOL * 10);
        },
        TIMEOUT,
      );

      it(
        "matmul(view, W) computes on the view's region",
        async () => {
          const t = api();
          const r = ref();
          const [m, k] = r.shape;
          const n = 3;
          const wData = Array.from({ length: k * n }, (_, i) =>
            Math.sin(i * 0.7),
          );
          const W = t.tensorFromArray(wData, [k, n], { device });
          const got = Array.from(await t.matmul(build(t), W).cpu());
          const want = jsMatmul(jsRead(r), wData, m, k, n);
          expect(maxAbsDiff(got, want)).toBeLessThan(ATOL * 100);
        },
        TIMEOUT,
      );

      it(
        "cat([view, view]) copies the view's region",
        async () => {
          const t = api();
          const v1 = build(t);
          const v2 = build(t);
          const got = Array.from(await t.cat([v1, v2], 0).cpu());
          const want = [...jsRead(ref()), ...jsRead(ref())];
          expect(maxAbsDiff(got, want)).toBeLessThan(ATOL);
        },
        TIMEOUT,
      );
    });
  }

  it(
    "item() on a narrowed scalar reads the right element",
    async () => {
      const t = api();
      const base = t.tensorFromArray(BASE_DATA, [24], { device });
      const v = base.narrow(0, 17, 1);
      expect(Math.abs((await v.item()) - 17)).toBeLessThan(ATOL);
    },
    TIMEOUT,
  );

  it(
    "tril on an offset view zeroes the right region",
    async () => {
      const t = api();
      const base = t.tensorFromArray(BASE_DATA, [4, 6], { device });
      const v = base.narrow(0, 1, 3).narrow(1, 1, 3); // [3,3] offset view
      const got = Array.from(await t.tril(v).cpu());
      const flat = jsRead(
        jsNarrow(jsNarrow(jsBase(BASE_DATA, BASE_SHAPE), 0, 1, 3), 1, 1, 3),
      );
      const want = flat.map((x, i) => {
        const row = Math.floor(i / 3);
        const col = i % 3;
        return col <= row ? x : 0;
      });
      expect(maxAbsDiff(got, want)).toBeLessThan(ATOL);
    },
    TIMEOUT,
  );

  it(
    "causal-mask-narrow pattern: scores + narrow(mask) == scores + uploaded mask",
    async () => {
      // Persistent [maxS, maxS] additive causal mask; per-step slice via narrow
      // (rows [qStart, S), cols [0, kvS)) must equal the CPU-built mask upload.
      const t = api();
      const maxS = 8;
      const S = 2;
      const kvS = 6;
      const qStart = kvS - S;
      const maskData = new Array<number>(maxS * maxS).fill(0);
      for (let i = 0; i < maxS; i++)
        for (let j = i + 1; j < maxS; j++) maskData[i * maxS + j] = -1e9;
      const maskFull = t.tensorFromArray(maskData, [maxS, maxS], { device });
      const maskView = maskFull.narrow(0, qStart, S).narrow(1, 0, kvS);

      const scoresData = Array.from({ length: S * kvS }, (_, i) => Math.cos(i));
      const scores = t.tensorFromArray(scoresData, [S, kvS], { device });

      const got = Array.from(await t.add(scores, maskView).cpu());
      const want = scoresData.map((x, i) => {
        const row = Math.floor(i / kvS);
        const col = i % kvS;
        return col > qStart + row ? x + -1e9 : x;
      });
      expect(maxAbsDiff(got, want)).toBeLessThan(1);
    },
    TIMEOUT,
  );

  if (device === "webgpu") {
    it(
      "RoPE-table-slice pattern: applyRoPE(narrow(table)) == applyRoPE(uploaded slice)",
      async () => {
        // Persistent [maxS, half] cos/sin tables; per-step row slice via narrow
        // must equal uploading the slice fresh (the deleted model dodge).
        const t = api();
        const maxS = 16;
        const half = 4;
        const headDim = half * 2;
        const S = 3;
        const pos = 5;
        const cosData = Array.from({ length: maxS * half }, (_, i) =>
          Math.cos(i * 0.13),
        );
        const sinData = Array.from({ length: maxS * half }, (_, i) =>
          Math.sin(i * 0.29),
        );
        const cosT = t.tensorFromArray(cosData, [maxS, half], { device });
        const sinT = t.tensorFromArray(sinData, [maxS, half], { device });

        const qData = Array.from({ length: 2 * S * headDim }, (_, i) =>
          Math.sin(i * 0.51),
        );
        const mkQ = () =>
          t.tensorFromArray(qData, [1, 2, S, headDim], { device });

        // Reference: uploaded contiguous slices (the old dodge).
        const cosUp = t.tensorFromArray(
          cosData.slice(pos * half, (pos + S) * half),
          [S, half],
          { device },
        );
        const sinUp = t.tensorFromArray(
          sinData.slice(pos * half, (pos + S) * half),
          [S, half],
          { device },
        );
        const want = Array.from(await t.applyRoPE(mkQ(), cosUp, sinUp).cpu());

        // Natural: GPU-side narrow of the persistent tables.
        const got = Array.from(
          await t
            .applyRoPE(mkQ(), cosT.narrow(0, pos, S), sinT.narrow(0, pos, S))
            .cpu(),
        );
        expect(maxAbsDiff(got, want)).toBeLessThan(ATOL);
      },
      TIMEOUT,
    );

    it(
      "f16 narrow with odd element offset reads the right region",
      async () => {
        const t = api();
        const base = t.tensorFromArray(BASE_DATA, [24], { device });
        const v = base.toDtype("f16").narrow(0, 7, 4); // odd offset → 2-byte, non-4-aligned
        const got = Array.from(await v.cpu());
        expect(maxAbsDiff(got, [7, 8, 9, 10])).toBeLessThan(0.01);
      },
      TIMEOUT,
    );
  }
}

describe("offset views (CPU)", () => {
  runSuite("cpu");
});

describe.skipIf(cpuOnly)("offset views (WebGPU)", () => {
  beforeAll(async () => {
    await initWebGPU();
  });
  runSuite("webgpu");
});
