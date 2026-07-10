/**
 * OP CONFORMANCE HARNESS — the catalog + generator.
 *
 * Derives the test matrix from op metadata instead of hand-enumerating cases,
 * so every catalogued op is automatically swept through the bug classes this
 * repo has actually been bitten by:
 *
 *   #58 offset-views   — views share a buffer at an element offset; raw-bind
 *                        consumers historically read from element 0. The 5-view
 *                        battery (generalized from offset-views.spec.ts) flows
 *                        every op through offset/strided/chained/transposed
 *                        views.
 *   #59 dtype-oblivious — f16 (and integer) kernels silently wrong. Every op is
 *                        swept across its supported dtypes.
 *   pow(x<0) class     — CPU (**) vs WGSL (exp2·log2) disagree silently. The
 *                        GPU-vs-CPU differential is the structural defense; a
 *                        signed-base pow case exercises it directly.
 *   broadcast/odd-size — odd shapes (33) catch tile/vec remainder bugs;
 *                        broadcast pairs catch indexing bugs.
 *
 * Two comparison oracles, per vitest project:
 *   - CPU project  (device="cpu",  no oracle): CPU backend vs an INDEPENDENT
 *     JS f64 reference (elementwise/reduction refs below) for f32; structural
 *     op(view)==op(contig) for f16 / ref-less ops.
 *   - WebGPU proj  (device="webgpu", oracle=CPU): GPU vs the CPU backend — the
 *     full cross-implementation differential (the headline defense).
 *
 * The JS strided-view evaluator here is a deliberate SECOND implementation of
 * the view math (not importing view-meta.ts) so a shared bug can't self-validate.
 */

import { describe, expect, it } from "vitest";
import type { DType } from "../../src/backend/types";
import type { Tensor } from "../../src/frontend/tensor";
import type { Torchlette } from "../../src/frontend/torchlette";

const TIMEOUT = 120_000;

// ---------------------------------------------------------------------------
// Independent JS strided-view evaluator (second impl of the view math).
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
function jsNarrow(v: JsView, dim: number, start: number, len: number): JsView {
  const shape = v.shape.slice();
  shape[dim] = len;
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

// ---------------------------------------------------------------------------
// Deterministic, domain-respecting input generation.
// ---------------------------------------------------------------------------

type Domain = "any" | "nonneg" | "positive" | "signed" | "unit";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** n deterministic values in the op's domain. Seeded by a stable string hash. */
function genInputs(n: number, domain: Domain, seedStr: string): number[] {
  let h = 2166136261;
  for (let i = 0; i < seedStr.length; i++) {
    h ^= seedStr.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const rng = mulberry32(h);
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const u = rng();
    switch (domain) {
      case "positive":
        out[i] = 0.15 + u * 3.0; // (0, 3.2]
        break;
      case "nonneg":
        out[i] = u * 3.0;
        break;
      case "unit":
        out[i] = u < 0.5 ? 0 : 1; // boolean-ish for where/comparisons
        break;
      case "signed":
        out[i] = u < 0.5 ? -(0.2 + u * 2.5) : 0.2 + u * 2.5; // no near-zero
        break;
      default:
        out[i] = (u - 0.5) * 4.0; // [-2, 2]
    }
    // Avoid exact half-integers (round() half-to-even vs half-up divergence).
    if (Math.abs(out[i] - Math.round(out[i])) < 1e-6) out[i] += 0.13;
  }
  return out;
}

// ---------------------------------------------------------------------------
// The catalog.
// ---------------------------------------------------------------------------

type TolClass = "exact" | "f32" | "loose";

interface UnaryOp {
  name: string;
  domain: Domain;
  tol: TolClass;
  dtypes: DType[];
  /** JS f64 reference; omitted when no clean independent ref exists. */
  ref?: (x: number) => number;
}

const FLOAT = ["f32", "f16"] as DType[];

export const UNARY_OPS: UnaryOp[] = [
  { name: "relu", domain: "any", tol: "exact", dtypes: FLOAT, ref: (x) => (x > 0 ? x : 0) },
  { name: "neg", domain: "any", tol: "exact", dtypes: FLOAT, ref: (x) => -x },
  { name: "abs", domain: "any", tol: "exact", dtypes: FLOAT, ref: Math.abs },
  { name: "sign", domain: "signed", tol: "exact", dtypes: FLOAT, ref: Math.sign },
  { name: "floor", domain: "any", tol: "exact", dtypes: FLOAT, ref: Math.floor },
  { name: "ceil", domain: "any", tol: "exact", dtypes: FLOAT, ref: Math.ceil },
  { name: "round", domain: "any", tol: "exact", dtypes: FLOAT, ref: Math.round },
  { name: "exp", domain: "any", tol: "loose", dtypes: FLOAT, ref: Math.exp },
  { name: "log", domain: "positive", tol: "loose", dtypes: FLOAT, ref: Math.log },
  { name: "sqrt", domain: "nonneg", tol: "f32", dtypes: FLOAT, ref: Math.sqrt },
  { name: "rsqrt", domain: "positive", tol: "f32", dtypes: FLOAT, ref: (x) => 1 / Math.sqrt(x) },
  { name: "sin", domain: "any", tol: "f32", dtypes: FLOAT, ref: Math.sin },
  { name: "cos", domain: "any", tol: "f32", dtypes: FLOAT, ref: Math.cos },
  { name: "tanh", domain: "any", tol: "f32", dtypes: FLOAT, ref: Math.tanh },
  { name: "sigmoid", domain: "any", tol: "f32", dtypes: FLOAT, ref: (x) => 1 / (1 + Math.exp(-x)) },
  { name: "silu", domain: "any", tol: "f32", dtypes: FLOAT, ref: (x) => x / (1 + Math.exp(-x)) },
  {
    name: "gelu",
    domain: "any",
    tol: "loose",
    dtypes: FLOAT,
    ref: (x) => x * 0.5 * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
  },
  // NOTE: gelu_erf is in OP_REGISTRY but has no table-installed method (reached
  // only via gelu(a, {approximate:"erf"})) — a metadata/dispatch gap, not swept
  // here. softplus likewise lacks a table method. See report's "metadata gaps".
  // isfinite swept f32+f16: an f16 input upcasts to f32 before the bitmask test
  // and the output binding is f32 (always_f32) (#59 FINDING #A fixed).
  { name: "isfinite", domain: "any", tol: "exact", dtypes: ["f32", "f16"], ref: (x) => (Number.isFinite(x) ? 1 : 0) },
];

interface BinaryOp {
  name: string;
  domain: Domain;
  domainB: Domain;
  tol: TolClass;
  dtypes: DType[];
  ref: (a: number, b: number) => number;
}

export const BINARY_OPS: BinaryOp[] = [
  { name: "add", domain: "any", domainB: "any", tol: "f32", dtypes: FLOAT, ref: (a, b) => a + b },
  { name: "mul", domain: "any", domainB: "any", tol: "f32", dtypes: FLOAT, ref: (a, b) => a * b },
  { name: "div", domain: "any", domainB: "positive", tol: "f32", dtypes: FLOAT, ref: (a, b) => a / b },
  { name: "minimum", domain: "any", domainB: "any", tol: "exact", dtypes: FLOAT, ref: Math.min },
  { name: "maximum", domain: "any", domainB: "any", tol: "exact", dtypes: FLOAT, ref: Math.max },
  // pow with POSITIVE base only — negative-base tensor^tensor is a documented
  // WGSL limitation (exp2·log2 → NaN); covered as a targeted case separately.
  { name: "pow", domain: "positive", domainB: "unit", tol: "loose", dtypes: ["f32"], ref: Math.pow },
];

export const COMPARISON_OPS = ["gt", "lt", "ge", "le", "eq", "ne"] as const;
const CMP_REF: Record<string, (a: number, b: number) => number> = {
  gt: (a, b) => (a > b ? 1 : 0),
  lt: (a, b) => (a < b ? 1 : 0),
  ge: (a, b) => (a >= b ? 1 : 0),
  le: (a, b) => (a <= b ? 1 : 0),
  eq: (a, b) => (a === b ? 1 : 0),
  ne: (a, b) => (a !== b ? 1 : 0),
};

export const REDUCTION_OPS = ["sum", "mean", "max", "min"] as const;
export const ARGREDUCE_OPS = ["argmax", "argmin"] as const;

// ---------------------------------------------------------------------------
// Comparison utilities.
// ---------------------------------------------------------------------------

function maxAbs(a: number[]): number {
  let m = 0;
  for (const x of a) m = Math.max(m, Math.abs(x));
  return m;
}
function maxAbsDiff(a: number[], b: number[]): number {
  expect(a.length, "length mismatch").toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    // A NaN diff (e.g. GPU pow(x<0) → NaN vs a finite CPU value) is a definite
    // divergence; NaN comparisons are always false, so surface it as Infinity
    // rather than let it be silently skipped.
    if (Number.isNaN(d)) return Number.POSITIVE_INFINITY;
    if (d > m) m = d;
  }
  return m;
}

/** Tolerance for comparing `got` to `want`, given op class, dtype, and n. */
function tolerance(tol: TolClass, dtype: DType, want: number[], n = 1): number {
  const scale = 1 + maxAbs(want);
  if (dtype === "f16") {
    // ~10-bit mantissa: relative error ~1e-3, plus input-rounding slack.
    const base = tol === "loose" ? 8e-3 : 5e-3;
    return (base + base * maxAbs(want)) * Math.max(1, Math.sqrt(n));
  }
  const base = tol === "exact" ? 1e-5 : tol === "loose" ? 3e-4 : 3e-5;
  return base * scale * Math.max(1, Math.sqrt(n));
}

async function toArr(x: number | Tensor): Promise<number[]> {
  return typeof x === "number" ? [x] : Array.from(await x.cpu());
}

// ---------------------------------------------------------------------------
// View battery: builds a tensor of `dtype` on `device` and its JS mirror.
// ---------------------------------------------------------------------------

interface Case {
  name: string;
  build: (api: Torchlette) => Tensor; // the tensor under test
  ref: JsView; // independent JS mirror (row-major values via jsRead)
}

function unaryBattery(dtype: DType, domain: Domain, full: boolean): Case[] {
  const BASE = [4, 6];
  const data = genInputs(24, domain, `u:${domain}:${dtype}`);
  const mk = (api: Torchlette) =>
    api.tensorFromArray(data, BASE).toDtype(dtype);
  const odd = genInputs(33, domain, `u33:${domain}:${dtype}`);
  const cases: Case[] = [
    { name: "contiguous[4,6]", build: mk, ref: jsBase(data, BASE) },
    {
      name: "odd-size[33]",
      build: (api) => api.tensorFromArray(odd, [33]).toDtype(dtype),
      ref: jsBase(odd, [33]),
    },
    {
      name: "narrow(0,2,2) offset view",
      build: (api) => mk(api).narrow(0, 2, 2),
      ref: jsNarrow(jsBase(data, BASE), 0, 2, 2),
    },
  ];
  if (full) {
    cases.push(
      {
        name: "transpose.narrow strided+offset view",
        build: (api) => mk(api).transpose({ dim0: 0, dim1: 1 }).narrow(0, 2, 3),
        ref: jsNarrow(jsTranspose(jsBase(data, BASE), 0, 1), 0, 2, 3),
      },
      {
        name: "chained narrows view",
        build: (api) => mk(api).narrow(0, 1, 3).narrow(1, 1, 4),
        ref: jsNarrow(jsNarrow(jsBase(data, BASE), 0, 1, 3), 1, 1, 4),
      },
    );
  }
  return cases;
}

// ---------------------------------------------------------------------------
// The generator — registers describe/it blocks. Call inside a describe that has
// already initialized the device (webgpu: beforeAll(initWebGPU)).
// ---------------------------------------------------------------------------

export interface ConformanceCfg {
  device: "cpu" | "webgpu";
  makeApi: () => Torchlette;
  /** CPU oracle for the GPU-vs-CPU differential (webgpu project only). */
  makeOracle?: () => Torchlette;
  /**
   * getGpuUncapturedErrorCount, passed by the webgpu spec only (keeps this
   * helper webgpu-free for the CPU project). When present, each differential
   * asserts the op raised no uncaptured GPU error — a dropped submit silently
   * leaves stale buffer data, which value comparison alone can miss when the
   * stale bytes coincidentally match (exactly how isfinite[f16] hid).
   */
  gpuErrorCount?: () => number;
}

export function registerConformance(cfg: ConformanceCfg): void {
  // device is informational (each api builds on its own backend); not read here.
  const { makeApi, makeOracle, gpuErrorCount } = cfg;

  /**
   * Core assertion: run `op` on the target device, then verify against
   * (a) the CPU oracle differential [webgpu], and/or (b) the independent JS
   * reference `want` [f32 only], falling back to a structural op(contig) check.
   */
  async function check(
    label: string,
    tol: TolClass,
    dtype: DType,
    n: number,
    want: number[] | null,
    run: (api: Torchlette) => number | Tensor,
    contigFallback?: (api: Torchlette) => number | Tensor,
  ): Promise<void> {
    const errBefore = gpuErrorCount ? gpuErrorCount() : 0;
    const got = await toArr(run(makeApi()));
    if (gpuErrorCount) {
      expect(
        gpuErrorCount() - errBefore,
        `${label} :: raised uncaptured GPU error(s) — submit dropped, result is stale`,
      ).toBe(0);
    }
    let checked = false;
    if (makeOracle) {
      const oracle = await toArr(run(makeOracle()));
      const t = tolerance(tol, dtype, oracle, n);
      expect(maxAbsDiff(got, oracle), `${label} :: GPU-vs-CPU differential`).toBeLessThanOrEqual(t);
      checked = true;
    }
    if (want && dtype === "f32") {
      const t = tolerance(tol, dtype, want, n);
      expect(maxAbsDiff(got, want), `${label} :: vs JS reference`).toBeLessThanOrEqual(t);
      checked = true;
    }
    if (!checked && contigFallback) {
      const contig = await toArr(contigFallback(makeApi()));
      const t = tolerance(tol, dtype, contig, n);
      expect(maxAbsDiff(got, contig), `${label} :: op(view)==op(contig) structural`).toBeLessThanOrEqual(t);
    }
  }

  // ---- Unary ops --------------------------------------------------------
  describe("unary", () => {
    for (const op of UNARY_OPS) {
      for (const dtype of op.dtypes) {
        it(
          `${op.name} [${dtype}]`,
          async () => {
            const cases = unaryBattery(dtype, op.domain, dtype === "f32");
            for (const c of cases) {
              const flat = jsRead(c.ref);
              const want = op.ref ? flat.map(op.ref) : null;
              await check(
                `${op.name}[${dtype}] ${c.name}`,
                op.tol,
                dtype,
                1,
                want,
                (api) => (api as any)[op.name](c.build(api)),
                (api) =>
                  (api as any)[op.name](
                    api.tensorFromArray(flat, c.ref.shape).toDtype(dtype),
                  ),
              );
            }
          },
          TIMEOUT,
        );
      }
    }
  });

  // ---- Binary ops (same-shape + broadcast + odd + offset-view) -----------
  describe("binary", () => {
    for (const op of BINARY_OPS) {
      for (const dtype of op.dtypes) {
        it(
          `${op.name} [${dtype}]`,
          async () => {
            const A = [4, 6];
            const aData = genInputs(24, op.domain, `ba:${op.name}:${dtype}`);
            const bFull = genInputs(24, op.domainB, `bb:${op.name}:${dtype}`);
            const bRow = genInputs(6, op.domainB, `brow:${op.name}:${dtype}`);
            const oddA = genInputs(33, op.domain, `boa:${op.name}:${dtype}`);
            const oddB = genInputs(33, op.domainB, `bob:${op.name}:${dtype}`);
            const T = (api: Torchlette, d: number[], s: number[]) =>
              api.tensorFromArray(d, s).toDtype(dtype);

            // want computed via explicit broadcast in JS f64.
            const bcast = (
              a: number[],
              sa: number[],
              b: number[],
              sb: number[],
            ): number[] => {
              const va = jsBase(a, sa);
              const vb = jsBase(b, sb);
              const outShape = A.slice();
              const out: number[] = [];
              for (let i = 0; i < outShape[0]; i++)
                for (let j = 0; j < outShape[1]; j++) {
                  const ai =
                    va.offset +
                    (sa[0] === 1 ? 0 : i) * va.strides[0] +
                    (sa[1] === 1 ? 0 : j) * va.strides[1];
                  const bi =
                    vb.offset +
                    (sb[0] === 1 ? 0 : i) * vb.strides[0] +
                    (sb[1] === 1 ? 0 : j) * vb.strides[1];
                  out.push(op.ref(a[ai], b[bi]));
                }
              return out;
            };

            const variants: Array<{ name: string; run: (api: Torchlette) => Tensor; want: number[] }> = [
              {
                name: "same-shape",
                run: (api) => (api as any)[op.name](T(api, aData, A), T(api, bFull, A)),
                want: aData.map((x, i) => op.ref(x, bFull[i])),
              },
              {
                name: "broadcast [4,6] vs [6]",
                run: (api) => (api as any)[op.name](T(api, aData, A), T(api, bRow, [6])),
                want: bcast(aData, A, bRow, [1, 6]),
              },
              {
                name: "odd-size [33]",
                run: (api) => (api as any)[op.name](T(api, oddA, [33]), T(api, oddB, [33])),
                want: oddA.map((x, i) => op.ref(x, oddB[i])),
              },
              {
                name: "offset-view A (narrow) vs contig",
                run: (api) =>
                  (api as any)[op.name](
                    T(api, aData, A).narrow(0, 2, 2),
                    T(api, bFull.slice(0, 12), [2, 6]),
                  ),
                want: jsRead(jsNarrow(jsBase(aData, A), 0, 2, 2)).map((x, i) =>
                  op.ref(x, bFull[i]),
                ),
              },
            ];
            for (const v of variants) {
              await check(
                `${op.name}[${dtype}] ${v.name}`,
                op.tol,
                dtype,
                1,
                v.want,
                v.run,
              );
            }
          },
          TIMEOUT,
        );
      }
    }
  });

  // ---- Comparison ops ---------------------------------------------------
  describe("comparison", () => {
    for (const name of COMPARISON_OPS) {
      it(
        `${name} [f32]`,
        async () => {
          const A = [4, 6];
          // Overlapping value sets so eq/ne actually fire.
          const aData = genInputs(24, "any", `cmp-a:${name}`).map((x) => Math.round(x));
          const bData = genInputs(24, "any", `cmp-b:${name}`).map((x) => Math.round(x));
          const T = (api: Torchlette, d: number[], s: number[]) =>
            api.tensorFromArray(d, s);
          const ref = CMP_REF[name];
          await check(
            `${name} same-shape`,
            "exact",
            "f32",
            1,
            aData.map((x, i) => ref(x, bData[i])),
            (api) => (api as any)[name](T(api, aData, A), T(api, bData, A)),
          );
          await check(
            `${name} offset-view`,
            "exact",
            "f32",
            1,
            jsRead(jsNarrow(jsBase(aData, A), 0, 1, 2)).map((x, i) => ref(x, bData[i])),
            (api) =>
              (api as any)[name](
                T(api, aData, A).narrow(0, 1, 2),
                T(api, bData.slice(0, 12), [2, 6]),
              ),
          );
        },
        TIMEOUT,
      );
    }
  });

  // ---- Ternary (where) --------------------------------------------------
  describe("ternary", () => {
    it(
      "where [f32]",
      async () => {
        const A = [4, 6];
        const cond = genInputs(24, "unit", "w-cond");
        const xData = genInputs(24, "any", "w-x");
        const yData = genInputs(24, "any", "w-y");
        const T = (api: Torchlette, d: number[]) =>
          api.tensorFromArray(d, A);
        await check(
          "where same-shape",
          "exact",
          "f32",
          1,
          cond.map((c, i) => (c ? xData[i] : yData[i])),
          (api) => api.where(T(api, cond), T(api, xData), T(api, yData)),
        );
      },
      TIMEOUT,
    );
  });

  // ---- Reductions -------------------------------------------------------
  // All reductions swept f32+f16. sum/mean/max/min all upcast f16→f32 before
  // reducing (the f32_required rule), so the kernel's f32 input binding reads
  // the right lanes (#59 FINDING #B fixed).
  describe("reduction", () => {
    for (const name of REDUCTION_OPS) {
      const dtypes: DType[] = ["f32", "f16"];
      for (const dtype of dtypes) {
        it(
          `${name} [${dtype}]`,
          async () => {
            const A = [4, 6];
            const data = genInputs(24, "any", `r:${name}:${dtype}`);
            const T = (api: Torchlette) =>
              api.tensorFromArray(data, A).toDtype(dtype);
            const base = jsBase(data, A);
            const reduce = (xs: number[]): number => {
              if (name === "sum") return xs.reduce((a, b) => a + b, 0);
              if (name === "mean") return xs.reduce((a, b) => a + b, 0) / xs.length;
              if (name === "max") return Math.max(...xs);
              return Math.min(...xs);
            };
            // full reduction
            await check(
              `${name}[${dtype}] full`,
              "f32",
              dtype,
              24,
              [reduce(jsRead(base))],
              (api) => (api as any)[name](T(api)),
            );
            // dim=-1 (rows)
            const rows: number[] = [];
            for (let i = 0; i < 4; i++) rows.push(reduce(data.slice(i * 6, i * 6 + 6)));
            await check(
              `${name}[${dtype}] dim=-1`,
              "f32",
              dtype,
              6,
              rows,
              (api) => (api as any)[name](T(api), { dim: -1 }),
            );
            // dim=0 (cols)
            const cols: number[] = [];
            for (let j = 0; j < 6; j++) {
              const col: number[] = [];
              for (let i = 0; i < 4; i++) col.push(data[i * 6 + j]);
              cols.push(reduce(col));
            }
            await check(
              `${name}[${dtype}] dim=0`,
              "f32",
              dtype,
              4,
              cols,
              (api) => (api as any)[name](T(api), { dim: 0 }),
            );
            // reduce over an offset view (dim=-1)
            const vRef = jsNarrow(base, 0, 1, 3);
            const vFlat = jsRead(vRef);
            const vRows: number[] = [];
            for (let i = 0; i < 3; i++) vRows.push(reduce(vFlat.slice(i * 6, i * 6 + 6)));
            await check(
              `${name}[${dtype}] view dim=-1`,
              "f32",
              dtype,
              6,
              vRows,
              (api) => (api as any)[name](T(api).narrow(0, 1, 3), { dim: -1 }),
            );
          },
          TIMEOUT,
        );
      }
    }
  });

  // ---- Arg-reductions (exact integer indices, distinct values → no ties) --
  describe("argreduce", () => {
    for (const name of ARGREDUCE_OPS) {
      it(
        `${name} [f32]`,
        async () => {
          const A = [4, 6];
          // Distinct values via a permutation → unambiguous argmax/argmin.
          const data = Array.from({ length: 24 }, (_, i) => ((i * 7 + 3) % 24) - 12 + i * 0.01);
          const T = (api: Torchlette) => api.tensorFromArray(data, A);
          const pick = (xs: number[]): number => {
            let bi = 0;
            for (let i = 1; i < xs.length; i++) {
              if (name === "argmax" ? xs[i] > xs[bi] : xs[i] < xs[bi]) bi = i;
            }
            return bi;
          };
          const rows: number[] = [];
          for (let i = 0; i < 4; i++) rows.push(pick(data.slice(i * 6, i * 6 + 6)));
          await check(
            `${name} dim=-1`,
            "exact",
            "f32",
            1,
            rows,
            (api) => (api as any)[name](T(api), { dim: -1 }),
          );
        },
        TIMEOUT,
      );
    }
  });

  // ---- pow signed-base scalar-exponent (the pow(x<0) class) --------------
  // The frontend lowers pow(tensor, non-neg int) to a mul-chain, which is exact
  // for any sign — unlike WGSL exp2·log2 (NaN for x<0). This case is the
  // structural guard for that lowering; the GPU-vs-CPU differential catches a
  // regression to the transcendental path.
  describe("pow-signed-base", () => {
    for (const exp of [2, 3]) {
      it(
        `pow(signed, ${exp})`,
        async () => {
          const data = genInputs(24, "signed", `pow-signed:${exp}`);
          await check(
            `pow(signed,${exp})`,
            "loose",
            "f32",
            1,
            data.map((x) => x ** exp),
            (api) => api.pow(api.tensorFromArray(data, [4, 6]), exp),
          );
        },
        TIMEOUT,
      );
    }
  });
}
