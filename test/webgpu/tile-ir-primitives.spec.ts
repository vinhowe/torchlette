/**
 * Tile IR primitive tests (task #62 stage 1): the three primitives the GEMV
 * implementation identified as gaps.
 *
 *   1. ctx.loadVec4 — vec4 loads from STORAGE bindings in imperative mode
 *      (binding declared array<vec4<T>>, scalar accesses rewritten to
 *      component indexing, f16 widened on load).
 *   2. ctx.wgReduceSegmented — reduce within GROUPS of lanes inside one
 *      workgroup (subgroup butterfly / subgroup+smem tree / smem fallback).
 *   3. rowGrid2d + ctx.rowIndex2d — per-row grids beyond 65535 workgroups
 *      without hand-rolled 2D mapping.
 *
 * Each primitive has WGSL compilation tests (no GPU) and GPU dispatch tests.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  beginSharedEncoder,
  flushSharedEncoder,
  getWebGPUDevice,
  initWebGPU,
  syncWebGPU,
} from "../../src/backend/webgpu";
import type {
  GPUBuffer,
  GPUDevice,
  GPUQueue,
} from "../../src/backend/webgpu/gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../../src/backend/webgpu/gpu-types";
import {
  clearSubgroupSupport,
  getSubgroupSupport,
  type SubgroupSupport,
  setSubgroupSupport,
} from "../../src/backend/webgpu/matmul/types";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import { createTileKernelDispatcher } from "../../src/backend/webgpu/tile-dispatch";
import {
  rowGrid2d,
  type SeamFn,
  splitWorkgroups2d,
  type TileKernelSpec,
} from "../../src/backend/webgpu/tile-ir";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

let device: GPUDevice;
let queue: GPUQueue;

function makeF32Buffer(data: Float32Array, usage: number): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

/** Pack f32 values to IEEE f16 bit patterns (via Float16Array-less rounding). */
function f32ToF16Bits(v: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 1 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    const half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    return sign | (half + (rem > 1 << (shift - 1) ? 1 : 0));
  }
  let half = (e << 10) | (mant >>> 13);
  if (mant & 0x1000) half += 1; // round-to-nearest
  return sign | half;
}

function f16BitsToF32(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  if (exp === 0) {
    if (mant === 0) u32[0] = sign;
    else {
      // subnormal
      let e = -1;
      let m = mant;
      do {
        e++;
        m <<= 1;
      } while (!(m & 0x400));
      u32[0] = sign | ((127 - 15 - e) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp === 0x1f) {
    u32[0] = sign | 0x7f800000 | (mant << 13);
  } else {
    u32[0] = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  return f32[0];
}

function makeF16Buffer(data: Float32Array, usage: number): GPUBuffer {
  const bits = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) bits[i] = f32ToF16Bits(data[i]);
  const buf = device.createBuffer({
    size: bits.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint16Array(buf.getMappedRange()).set(bits);
  buf.unmap();
  return buf;
}

function makeOutputBuffer(numElements: number): GPUBuffer {
  return device.createBuffer({
    size: numElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

async function readF32Buffer(
  buf: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// ---------------------------------------------------------------------------
// Primitive 1: loadVec4 (vec4 loads from storage bindings, imperative mode)
// ---------------------------------------------------------------------------

/** out[t] = dot(a[4t..4t+3], b[4t..4t+3]) — one vec4 dot per thread. */
function vec4DotSpec(bDtype: "f32" | "f16"): TileKernelSpec {
  return {
    name: `vec4Dot_${bDtype}`,
    workgroupSize: 64,
    enableF16: bDtype === "f16",
    bindings: {
      a: { storage: "read", type: "f32" },
      b: { storage: "read", type: bDtype },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { n_vec: "u32" },
    grid: (u) => [Math.ceil(u.n_vec / 64)],
    kernel(ctx) {
      const t = ctx.globalId(0);
      const nVec = ctx.uniform("n_vec");
      ctx.ifThen(t.lt(nVec), () => {
        const base = ctx.emitLet("base", t.shl(2));
        const av = ctx.loadVec4("a", base);
        const bv = ctx.loadVec4("b", base);
        ctx.emitStore("out", t, av.vec4Dot(bv));
      });
    },
  };
}

/** Mixed access: vec4-load a, ALSO scalar-load a (component rewrite). */
const mixedAccessSpec: TileKernelSpec = {
  name: "vec4MixedAccess",
  workgroupSize: 64,
  bindings: {
    a: { storage: "read", type: "f32" },
    out: { storage: "read_write", type: "f32" },
  },
  uniforms: { n_vec: "u32" },
  grid: (u) => [Math.ceil(u.n_vec / 64)],
  kernel(ctx) {
    const t = ctx.globalId(0);
    const nVec = ctx.uniform("n_vec");
    ctx.ifThen(t.lt(nVec), () => {
      const base = ctx.emitLet("base", t.shl(2));
      const av = ctx.loadVec4("a", base);
      // Scalar access to the SAME vec4-declared binding: a[4t+1]
      const scalar = ctx.load("a", base.add(ctx.u32(1)));
      // out[t] = (sum of vec4 lanes) + a[4t+1]
      const ones = ctx.vec4Splat(ctx.f32(1.0));
      ctx.emitStore("out", t, av.vec4Dot(ones).add(scalar));
    });
  },
};

describe("Primitive 1: loadVec4 (WGSL)", () => {
  it("declares vec4-loaded bindings as array<vec4<T>>, others scalar", () => {
    const wgsl = compileTileKernel(vec4DotSpec("f32"));
    expect(wgsl).toContain("var<storage, read> a: array<vec4<f32>>;");
    expect(wgsl).toContain("var<storage, read> b: array<vec4<f32>>;");
    // out is not vec4-loaded — stays scalar
    expect(wgsl).toContain("var<storage, read_write> out: array<f32>;");
    // vec4 load indexing
    expect(wgsl).toContain(">> 2u]");
    expect(wgsl).toContain("dot(");
  });

  it("widens f16 vec4 loads to vec4<f32>", () => {
    const wgsl = compileTileKernel(vec4DotSpec("f16"));
    expect(wgsl).toContain("enable f16;");
    expect(wgsl).toContain("var<storage, read> b: array<vec4<f16>>;");
    expect(wgsl).toMatch(/vec4<f32>\(b\[\(.*\) >> 2u\]\)/);
  });

  it("rewrites scalar access to a vec4-declared binding to component indexing", () => {
    const wgsl = compileTileKernel(mixedAccessSpec);
    expect(wgsl).toContain("array<vec4<f32>>");
    // scalar load a[base+1] becomes a[(idx) >> 2u][(idx) & 3u]
    expect(wgsl).toMatch(/a\[\(.*\) >> 2u\]\[\(.*\) & 3u\]/);
  });

  it("throws when combined with whole-kernel vectorize mode", () => {
    const bad: TileKernelSpec = {
      ...vec4DotSpec("f32"),
      name: "vec4BadVectorize",
      vectorize: 4,
    };
    expect(() => compileTileKernel(bad)).toThrow(/loadVec4/);
  });

  it("throws on atomic bindings", () => {
    const bad: TileKernelSpec = {
      name: "vec4BadAtomic",
      workgroupSize: 64,
      bindings: { flag: { storage: "atomic", type: "u32" } },
      uniforms: { n_vec: "u32" },
      grid: () => [1],
      kernel(ctx) {
        ctx.loadVec4("flag", ctx.u32(0));
      },
    };
    expect(() => compileTileKernel(bad)).toThrow(/atomic/);
  });
});

describe.runIf(isWebGPUEnabled)("Primitive 1: loadVec4 (GPU)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  it("f32×f32 vec4 dot matches CPU", async () => {
    const NV = 200; // 800 elements
    const N = NV * 4;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = Math.sin(i * 0.37);
      bData[i] = Math.cos(i * 0.21);
    }
    const expected = new Float32Array(NV);
    for (let t = 0; t < NV; t++) {
      for (let c = 0; c < 4; c++)
        expected[t] += aData[4 * t + c] * bData[4 * t + c];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(NV);

    const kernel = createTileKernelDispatcher(vec4DotSpec("f32"));
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, out: outBuf }, { n_vec: NV });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, NV);
    for (let t = 0; t < NV; t++) expect(result[t]).toBeCloseTo(expected[t], 3);

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("f32×f16 vec4 dot widens and matches CPU (f16 grid values)", async () => {
    const NV = 64;
    const N = NV * 4;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = ((i % 13) - 6) * 0.25;
      bData[i] = ((i % 7) - 3) * 0.5;
    }
    const expected = new Float32Array(NV);
    for (let t = 0; t < NV; t++) {
      for (let c = 0; c < 4; c++) {
        const bQ = f16BitsToF32(f32ToF16Bits(bData[4 * t + c]));
        expected[t] += aData[4 * t + c] * bQ;
      }
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF16Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(NV);

    const kernel = createTileKernelDispatcher(vec4DotSpec("f16"));
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, out: outBuf }, { n_vec: NV });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, NV);
    for (let t = 0; t < NV; t++) expect(result[t]).toBeCloseTo(expected[t], 3);

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("mixed vec4 + scalar access to the same binding", async () => {
    const NV = 100;
    const N = NV * 4;
    const aData = new Float32Array(N);
    for (let i = 0; i < N; i++) aData[i] = i * 0.01;
    const expected = new Float32Array(NV);
    for (let t = 0; t < NV; t++) {
      let s = 0;
      for (let c = 0; c < 4; c++) s += aData[4 * t + c];
      expected[t] = s + aData[4 * t + 1];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(NV);

    const kernel = createTileKernelDispatcher(mixedAccessSpec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, out: outBuf }, { n_vec: NV });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, NV);
    for (let t = 0; t < NV; t++) expect(result[t]).toBeCloseTo(expected[t], 3);

    aBuf.destroy();
    outBuf.destroy();
  });
});

// ---------------------------------------------------------------------------
// Primitive 2: wgReduceSegmented
// ---------------------------------------------------------------------------

/**
 * out[tid_global] = reduction of x over THIS lane's segment. Every lane
 * writes, validating the all-lanes-broadcast return semantics.
 */
function segmentedReduceSpec(
  op: "sum" | "max" | "min",
  wgSize: number,
  groups: number,
): TileKernelSpec {
  return {
    name: `segReduce_${op}_${wgSize}_${groups}`,
    workgroupSize: wgSize,
    bindings: {
      x: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { n: "u32" },
    grid: (u) => [Math.ceil(u.n / wgSize)],
    kernel(ctx) {
      const tid = ctx.localIndex();
      const n = ctx.uniform("n");
      const gid = ctx.emitLet(
        "g",
        ctx.programId(0).mul(ctx.u32(wgSize)).add(tid),
      );
      const val = ctx.emitLet("v", ctx.load("x", gid));
      const red = ctx.wgReduceSegmented(op, val, tid, wgSize, groups);
      ctx.guardedStore("out", gid.lt(n), gid, red);
    },
  };
}

function cpuSegmentedReduce(
  op: "sum" | "max" | "min",
  x: Float32Array,
  wgSize: number,
  groups: number,
): Float32Array {
  const groupSize = wgSize / groups;
  const out = new Float32Array(x.length);
  for (let base = 0; base < x.length; base += groupSize) {
    let acc = op === "sum" ? 0 : x[base];
    for (let i = 0; i < groupSize; i++) {
      const v = x[base + i];
      acc =
        op === "sum"
          ? acc + v
          : op === "max"
            ? Math.max(acc, v)
            : Math.min(acc, v);
    }
    for (let i = 0; i < groupSize; i++) out[base + i] = acc;
  }
  return out;
}

describe("Primitive 2: wgReduceSegmented (WGSL)", () => {
  let saved: SubgroupSupport | null;
  beforeAll(() => {
    saved = getSubgroupSupport();
  });
  afterAll(() => {
    if (saved) setSubgroupSupport(saved);
    else clearSubgroupSupport();
  });

  it("groupSize <= subgroupSize: register butterfly, no shared memory", () => {
    setSubgroupSupport({ supported: true, subgroupSize: 32 });
    const wgsl = compileTileKernel(segmentedReduceSpec("sum", 64, 8)); // gs=8
    expect(wgsl).toContain("subgroupShuffleXor(");
    expect(wgsl).not.toContain("var<workgroup>");
    expect(wgsl).not.toContain("workgroupBarrier");
    expect(wgsl).toContain("enable subgroups;");
  });

  it("groupSize > subgroupSize: subgroup intrinsic + per-segment smem tree", () => {
    setSubgroupSupport({ supported: true, subgroupSize: 32 });
    const wgsl = compileTileKernel(segmentedReduceSpec("sum", 256, 4)); // gs=64
    expect(wgsl).toContain("subgroupAdd(");
    expect(wgsl).toContain("var<workgroup>");
    expect(wgsl).toContain("workgroupBarrier");
  });

  it("no subgroups: full shared-memory tree per segment", () => {
    clearSubgroupSupport();
    const wgsl = compileTileKernel(segmentedReduceSpec("sum", 256, 4));
    expect(wgsl).not.toContain("subgroup");
    expect(wgsl).toContain("var<workgroup>");
    expect(wgsl).toContain("workgroupBarrier");
  });

  it("rejects non-dividing groups and non-power-of-two segments", () => {
    expect(() => compileTileKernel(segmentedReduceSpec("sum", 64, 3))).toThrow(
      /divide/,
    );
    expect(
      () => compileTileKernel(segmentedReduceSpec("sum", 96, 2)), // gs=48
    ).toThrow(/power of two/);
  });
});

describe.runIf(isWebGPUEnabled)("Primitive 2: wgReduceSegmented (GPU)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  const cases: Array<{
    op: "sum" | "max" | "min";
    wg: number;
    groups: number;
  }> = [
    { op: "sum", wg: 64, groups: 8 }, // gs=8 (butterfly when subgroups)
    { op: "sum", wg: 256, groups: 4 }, // gs=64 (intrinsic + smem tree)
    { op: "sum", wg: 256, groups: 1 }, // gs=256 (degenerate = wgReduce)
    { op: "max", wg: 128, groups: 4 }, // gs=32
    { op: "min", wg: 64, groups: 2 }, // gs=32
  ];

  for (const { op, wg, groups } of cases) {
    it(`${op} wg=${wg} groups=${groups}: every lane sees its segment's reduction`, async () => {
      const numWg = 3;
      const N = numWg * wg;
      const xData = new Float32Array(N);
      for (let i = 0; i < N; i++) xData[i] = Math.sin(i * 1.7) * 4;
      const expected = cpuSegmentedReduce(op, xData, wg, groups);

      const xBuf = makeF32Buffer(xData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(N);

      const kernel = createTileKernelDispatcher(
        segmentedReduceSpec(op, wg, groups),
      );
      beginSharedEncoder();
      kernel.dispatch({ x: xBuf, out: outBuf }, { n: N });
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, N);
      for (let i = 0; i < N; i++) expect(result[i]).toBeCloseTo(expected[i], 3);

      xBuf.destroy();
      outBuf.destroy();
    });
  }
});

// ---------------------------------------------------------------------------
// Primitive 3: rowGrid2d + rowIndex2d (rows beyond 65535 workgroups)
// ---------------------------------------------------------------------------

const ROW_KERNEL_WG = 64;

/** out[row] = row * 2 + 1, one workgroup per row via rowGrid2d/rowIndex2d. */
const bigRowSpec: TileKernelSpec = {
  name: "bigRowKernel",
  workgroupSize: ROW_KERNEL_WG,
  bindings: {
    out: { storage: "read_write", type: "f32" },
  },
  uniforms: { num_rows: "u32" },
  grid: rowGrid2d("num_rows"),
  kernel(ctx) {
    const row = ctx.emitLet("row", ctx.rowIndex2d());
    const n = ctx.uniform("num_rows");
    const tid = ctx.localIndex();
    ctx.ifThen(tid.eq(ctx.u32(0)).and(row.lt(n)), () => {
      ctx.emitStore("out", row, row.toF32().mul(2).add(1));
    });
  },
};

describe("Primitive 3: rowGrid2d (grid math)", () => {
  it("stays 1D under the limit", () => {
    expect(splitWorkgroups2d(100)).toEqual([100, 1]);
    expect(splitWorkgroups2d(65535)).toEqual([65535, 1]);
    expect(rowGrid2d()({ num_rows: 4000 })).toEqual([4000, 1]);
  });

  it("splits into a balanced 2D grid beyond the limit", () => {
    const [gx, gy] = splitWorkgroups2d(70000);
    expect(gy).toBe(2);
    expect(gx).toBe(35000);
    expect(gx * gy).toBeGreaterThanOrEqual(70000);
    expect(gx).toBeLessThanOrEqual(65535);
    // lm_head-sized: 151936 rows
    const [gx2, gy2] = splitWorkgroups2d(151936);
    expect(gy2).toBe(3);
    expect(gx2 * gy2).toBeGreaterThanOrEqual(151936);
    expect(gx2 * gy2 - 151936).toBeLessThan(gy2);
  });

  it("applies rowsPerWorkgroup before splitting", () => {
    expect(rowGrid2d("num_rows", 4)({ num_rows: 151936 })).toEqual([37984, 1]);
  });

  it("kernel WGSL uses workgroup ids and num_workgroups", () => {
    const wgsl = compileTileKernel(bigRowSpec);
    expect(wgsl).toContain("workgroup_id");
    expect(wgsl).toContain("num_workgroups");
  });
});

// ---------------------------------------------------------------------------
// Expression seams: TileKernelSpec.seams + ctx.applySeam
// ---------------------------------------------------------------------------

/** out[i] = seam(x[i] * 2), seam receives outIndex. */
function seamedScaleSpec(seam?: SeamFn): TileKernelSpec {
  return {
    name: "seamedScale",
    workgroupSize: 64,
    bindings: {
      x: { storage: "read", type: "f32" },
      bias: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { n: "u32" },
    grid: (u) => [Math.ceil(u.n / 64)],
    seams: seam ? { epilogue: seam } : undefined,
    kernel(ctx) {
      const idx = ctx.globalId(0);
      const n = ctx.uniform("n");
      ctx.ifThen(idx.lt(n), () => {
        // "epilogue" seam point on the carrier value, with the output index.
        const val = ctx.applySeam("epilogue", ctx.load("x", idx).mul(2), {
          outIndex: idx,
        });
        ctx.emitStore("out", idx, val);
      });
    },
  };
}

describe("Expression seams (WGSL)", () => {
  it("identity when no seam is injected (bias binding unused)", () => {
    const wgsl = compileTileKernel(seamedScaleSpec());
    expect(wgsl).toContain("out[");
    expect(wgsl).not.toContain("bias[");
  });

  it("injected expression is spliced in with seam args", () => {
    const seam: SeamFn = (ctx, value, args) =>
      value.add(ctx.load("bias", args.outIndex)).tanh();
    const wgsl = compileTileKernel(seamedScaleSpec(seam));
    expect(wgsl).toContain("bias[");
    expect(wgsl).toContain("tanh(");
  });
});

describe.runIf(isWebGPUEnabled)("Expression seams (GPU)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  it("seam-injected bias+tanh matches CPU", async () => {
    const N = 200;
    const xData = new Float32Array(N);
    const biasData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      xData[i] = Math.sin(i * 0.5);
      biasData[i] = (i % 5) * 0.1;
      expected[i] = Math.tanh(xData[i] * 2 + biasData[i]);
    }

    const xBuf = makeF32Buffer(xData, GPUBufferUsage.STORAGE);
    const biasBuf = makeF32Buffer(biasData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const seam: SeamFn = (ctx, value, args) =>
      value.add(ctx.load("bias", args.outIndex)).tanh();
    const kernel = createTileKernelDispatcher(seamedScaleSpec(seam));
    beginSharedEncoder();
    kernel.dispatch({ x: xBuf, bias: biasBuf, out: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) expect(result[i]).toBeCloseTo(expected[i], 4);

    xBuf.destroy();
    biasBuf.destroy();
    outBuf.destroy();
  });
});

describe.runIf(isWebGPUEnabled)("Primitive 3: rowGrid2d (GPU)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  it("covers 70000 rows (> 65535) with correct row indices", async () => {
    const ROWS = 70000;
    const outBuf = makeOutputBuffer(ROWS);

    const kernel = createTileKernelDispatcher(bigRowSpec);
    beginSharedEncoder();
    kernel.dispatch({ out: outBuf }, { num_rows: ROWS });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, ROWS);
    // Spot-check across the whole range including both grid rows and the tail
    for (const r of [0, 1, 12345, 35000, 65534, 65535, 69998, 69999]) {
      expect(result[r]).toBe(r * 2 + 1);
    }
    let bad = 0;
    for (let r = 0; r < ROWS; r++) if (result[r] !== r * 2 + 1) bad++;
    expect(bad).toBe(0);

    outBuf.destroy();
  });
});
