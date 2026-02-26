/**
 * Tile IR gap tests.
 *
 * Validates the 6 feature gaps that make tile-IR the universal kernel language:
 *   Gap 1: Elementwise dispatch (globalId)
 *   Gap 2: Multiple output stores (imperative)
 *   Gap 3: Bitcast
 *   Gap 4: Atomics
 *   Gap 5: Subgroup operations
 *   Gap 6: Auto-vectorization
 *
 * Each gap has WGSL compilation tests (no GPU needed) and GPU dispatch tests.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  getWebGPUDevice,
  initWebGPU,
  syncWebGPU,
  beginSharedEncoder,
  flushSharedEncoder,
} from "../../src/backend/webgpu";
import { createTileKernelDispatcher } from "../../src/backend/webgpu/tile-dispatch";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import type { TileKernelSpec } from "../../src/backend/webgpu/tile-ir";
import type { GPUBuffer, GPUDevice, GPUQueue } from "../../src/backend/webgpu/gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../../src/backend/webgpu/gpu-types";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

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

function makeU32Buffer(data: Uint32Array, usage: number): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

function makeOutputBuffer(numElements: number, bytesPerElement = 4): GPUBuffer {
  return device.createBuffer({
    size: numElements * bytesPerElement,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

async function readF32Buffer(buf: GPUBuffer, count: number): Promise<Float32Array> {
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

async function readU32Buffer(buf: GPUBuffer, count: number): Promise<Uint32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Uint32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// ---------------------------------------------------------------------------
// Gap 1: Elementwise dispatch (globalId)
// ---------------------------------------------------------------------------

/** Elementwise add kernel using globalId — imperative mode. */
const elementwiseAddSpec: TileKernelSpec = {
  name: "elementwiseAdd",
  workgroupSize: 64,
  bindings: {
    a:      { storage: "read", type: "f32" },
    b:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { n: "u32" },
  grid: (u) => [Math.ceil(u.n / 64)],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const a = ctx.load("a", idx);
      const b = ctx.load("b", idx);
      ctx.guardedStore("output", idx.lt(n), idx, a.add(b));
    });
  },
};

/** Elementwise mul using globalId (2D grid). */
const elementwiseMul2DSpec: TileKernelSpec = {
  name: "elementwiseMul2D",
  workgroupSize: [8, 8],
  bindings: {
    a:      { storage: "read", type: "f32" },
    b:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { rows: "u32", cols: "u32" },
  grid: (u) => [Math.ceil(u.cols / 8), Math.ceil(u.rows / 8)],
  kernel(ctx) {
    const col = ctx.globalId(0);
    const row = ctx.globalId(1);
    const rows = ctx.uniform("rows");
    const cols = ctx.uniform("cols");
    const inBounds = row.lt(rows).and(col.lt(cols));
    ctx.ifThen(new (Object.getPrototypeOf(row).constructor)(inBounds.node), () => {
      const idx = row.mul(cols).add(col);
      const a = ctx.load("a", idx);
      const b = ctx.load("b", idx);
      ctx.guardedStore("output", inBounds, idx, a.mul(b));
    });
  },
};

describe("Gap 1: Elementwise dispatch (globalId)", () => {
  // -- WGSL compilation tests (no GPU needed) --

  it("emits global_invocation_id and no local_invocation_id", () => {
    const wgsl = compileTileKernel(elementwiseAddSpec);
    expect(wgsl).toContain("global_invocation_id");
    expect(wgsl).not.toContain("local_invocation_id");
    expect(wgsl).not.toContain("workgroup_id");
    expect(wgsl).toContain("gid.x");
  });

  it("emits gid.x and gid.y for 2D globalId", () => {
    const wgsl = compileTileKernel(elementwiseMul2DSpec);
    expect(wgsl).toContain("global_invocation_id");
    expect(wgsl).toContain("gid.x");
    expect(wgsl).toContain("gid.y");
  });

  it("still emits workgroup_id for matmul-like kernels", () => {
    const matmulLikeSpec: TileKernelSpec = {
      name: "matmulLike",
      workgroupSize: [8, 8],
      bindings: { output: { storage: "read_write", type: "f32" } },
      uniforms: { n: "u32" },
      grid: (u) => [1],
      kernel(ctx) {
        const wid = ctx.programId(0);
        const tid = ctx.threadIdx(0);
        ctx.emitLet("x", wid.add(tid));
      },
    };
    const wgsl = compileTileKernel(matmulLikeSpec);
    expect(wgsl).toContain("workgroup_id");
    expect(wgsl).toContain("local_invocation_id");
    expect(wgsl).not.toContain("global_invocation_id");
  });
});

describe.runIf(isWebGPUEnabled)("Gap 1: Elementwise GPU dispatch", () => {
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

  it("elementwise add: 1000 elements", async () => {
    const N = 1000;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i * 0.1;
      bData[i] = (N - i) * 0.01;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(elementwiseAddSpec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("elementwise add: non-multiple of workgroup size", async () => {
    const N = 137; // not a multiple of 64
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i;
      bData[i] = -i * 0.5;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(elementwiseAddSpec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });
});

// ---------------------------------------------------------------------------
// Gap 2: Multiple output stores (auto-phase)
// ---------------------------------------------------------------------------

/**
 * Auto-phase kernel that computes:
 *   mean_out[row] = mean(x[row*D .. row*D+D-1])
 *   output[row*D+i] = x[row*D+i] - mean
 *
 * Two stores: scalar (mean_out) and block (output), different blockRanges.
 */
const meanAndNormalizeSpec: TileKernelSpec = {
  name: "meanAndNormalize",
  workgroupSize: 256,
  bindings: {
    x:        { storage: "read", type: "f32" },
    mean_out: { storage: "read_write", type: "f32" },
    output:   { storage: "read_write", type: "f32" },
  },
  uniforms: { numRows: "u32", featureDim: "u32" },
  grid: (u) => [u.numRows],
  kernel(ctx) {
    const row = ctx.programId(0);
    const tid = ctx.localIndex();
    const D = ctx.uniform("featureDim");
    const Df = D.toF32();
    const base = row.mul(D);

    // Sum reduction for mean via wgReduce
    const mean = ctx.emitLet("mean",
      ctx.wgReduce("sum", tid, D, 256, (i) => ctx.load("x", base.add(i))).div(Df));

    // Store scalar mean (only thread 0)
    ctx.ifThen(tid.eq(ctx.u32(0)), () => {
      ctx.emitStore("mean_out", row, mean);
    });

    // Store normalized block values
    ctx.stridedFor(tid, D, 256, (i) => {
      ctx.emitStore("output", base.add(i), ctx.load("x", base.add(i)).sub(mean));
    });
  },
};

describe("Gap 2: Multiple output stores (imperative)", () => {
  it("compiles kernel with scalar and block stores", () => {
    const wgsl = compileTileKernel(meanAndNormalizeSpec);
    // Should have both stores
    expect(wgsl).toContain("mean_out[");
    expect(wgsl).toContain("output[");
    // Scalar store guarded by thread 0 check
    expect(wgsl).toContain("local_idx == 0u");
    // Uses shared memory for reduction
    expect(wgsl).toContain("var<workgroup>");
  });
});

describe.runIf(isWebGPUEnabled)("Gap 2: Multiple output stores GPU dispatch", () => {
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

  it("mean + normalize: correct scalar and block outputs", async () => {
    const numRows = 4;
    const D = 128;
    const N = numRows * D;

    // Create input data
    const xData = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      xData[i] = (i % 7) - 3; // values in -3..3
    }

    // CPU reference
    const expectedMean = new Float32Array(numRows);
    const expectedNorm = new Float32Array(N);
    for (let r = 0; r < numRows; r++) {
      let sum = 0;
      for (let j = 0; j < D; j++) sum += xData[r * D + j];
      expectedMean[r] = sum / D;
      for (let j = 0; j < D; j++) {
        expectedNorm[r * D + j] = xData[r * D + j] - expectedMean[r];
      }
    }

    const xBuf = makeF32Buffer(xData, GPUBufferUsage.STORAGE);
    const meanBuf = makeOutputBuffer(numRows);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(meanAndNormalizeSpec);
    beginSharedEncoder();
    kernel.dispatch(
      { x: xBuf, mean_out: meanBuf, output: outBuf },
      { numRows, featureDim: D },
    );
    flushSharedEncoder();

    const meanResult = await readF32Buffer(meanBuf, numRows);
    const normResult = await readF32Buffer(outBuf, N);

    // Check means
    for (let r = 0; r < numRows; r++) {
      expect(meanResult[r]).toBeCloseTo(expectedMean[r], 3);
    }

    // Check normalized values
    for (let i = 0; i < N; i++) {
      expect(normResult[i]).toBeCloseTo(expectedNorm[i], 3);
    }

    xBuf.destroy();
    meanBuf.destroy();
    outBuf.destroy();
  });
});

// ---------------------------------------------------------------------------
// Gap 3: Bitcast
// ---------------------------------------------------------------------------

/**
 * Imperative kernel: bitcast f32→u32, extract exponent, check if inf/NaN.
 * For IEEE 754 f32: exponent bits [30:23]. exponent == 0xFF → inf or NaN.
 * Output: 1u if inf/NaN, 0u otherwise.
 */
const isInfNanSpec: TileKernelSpec = {
  name: "isInfNan",
  workgroupSize: 64,
  bindings: {
    x:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "u32" },
  },
  uniforms: { n: "u32" },
  grid: (u) => [Math.ceil(u.n / 64)],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const val = ctx.load("x", idx);
      const bits = val.bitcastTo("u32");
      // Extract exponent: (bits >> 23) & 0xFF
      const exponent = bits.shr(ctx.const(23, "u32")).and(ctx.const(0xFF, "u32"));
      const isInfBool = exponent.eq(ctx.const(0xFF, "u32"));
      const flag = isInfBool.select(ctx.const(1, "u32"), ctx.const(0, "u32"));
      ctx.guardedStore("output", idx.lt(n), idx, flag);
    });
  },
};

describe("Gap 3: Bitcast", () => {
  it("emits bitcast<u32> in WGSL", () => {
    const wgsl = compileTileKernel(isInfNanSpec);
    expect(wgsl).toContain("bitcast<u32>");
  });
});

describe.runIf(isWebGPUEnabled)("Gap 3: Bitcast GPU dispatch", () => {
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

  it("detects inf/NaN via bitcast", async () => {
    const values = [1.0, Infinity, -Infinity, NaN, 0.0, -0.0, 3.14];
    const N = values.length;
    // Expected: 0 for normal, 1 for inf/NaN
    const expected = values.map(v => (!isFinite(v) || isNaN(v)) ? 1 : 0);

    const xBuf = makeF32Buffer(new Float32Array(values), GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(isInfNanSpec);
    beginSharedEncoder();
    kernel.dispatch({ x: xBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readU32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBe(expected[i]);
    }

    xBuf.destroy();
    outBuf.destroy();
  });
});

// ---------------------------------------------------------------------------
// Gap 4: Atomics
// ---------------------------------------------------------------------------

/**
 * Imperative kernel: scan array for inf/NaN, set atomic flag.
 * Uses bitcast (Gap 3) + atomicMax on a single u32 flag.
 */
const hasInfNanSpec: TileKernelSpec = {
  name: "hasInfNan",
  workgroupSize: 64,
  bindings: {
    x:    { storage: "read", type: "f32" },
    flag: { storage: "atomic", type: "u32" },
  },
  uniforms: { n: "u32" },
  grid: (u) => [Math.ceil(u.n / 64)],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const val = ctx.load("x", idx);
      const bits = val.bitcastTo("u32");
      const exponent = bits.shr(ctx.const(23, "u32")).and(ctx.const(0xFF, "u32"));
      const isInf = exponent.eq(ctx.const(0xFF, "u32"));
      // Convert bool to u32: select(0u, 1u, cond)
      const flag = isInf.select(ctx.const(1, "u32"), ctx.const(0, "u32"));
      ctx.atomicOp("flag", ctx.const(0, "u32"), "max", flag);
    });
  },
};

describe("Gap 4: Atomics", () => {
  it("emits atomic<u32> binding and atomicMax call", () => {
    const wgsl = compileTileKernel(hasInfNanSpec);
    expect(wgsl).toContain("array<atomic<u32>>");
    expect(wgsl).toContain("atomicMax(");
  });
});

describe.runIf(isWebGPUEnabled)("Gap 4: Atomics GPU dispatch", () => {
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

  it("atomic flag set when inf present", async () => {
    const values = [1.0, 2.0, Infinity, 4.0];
    const N = values.length;

    const xBuf = makeF32Buffer(new Float32Array(values), GPUBufferUsage.STORAGE);
    const flagBuf = makeU32Buffer(new Uint32Array([0]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const kernel = createTileKernelDispatcher(hasInfNanSpec);
    beginSharedEncoder();
    kernel.dispatch({ x: xBuf, flag: flagBuf }, { n: N });
    flushSharedEncoder();

    const result = await readU32Buffer(flagBuf, 1);
    expect(result[0]).toBe(1);

    xBuf.destroy();
    flagBuf.destroy();
  });

  it("atomic flag stays 0 when no inf", async () => {
    const values = [1.0, 2.0, 3.0, 4.0];
    const N = values.length;

    const xBuf = makeF32Buffer(new Float32Array(values), GPUBufferUsage.STORAGE);
    const flagBuf = makeU32Buffer(new Uint32Array([0]), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    const kernel = createTileKernelDispatcher(hasInfNanSpec);
    beginSharedEncoder();
    kernel.dispatch({ x: xBuf, flag: flagBuf }, { n: N });
    flushSharedEncoder();

    const result = await readU32Buffer(flagBuf, 1);
    expect(result[0]).toBe(0);

    xBuf.destroy();
    flagBuf.destroy();
  });
});

// ---------------------------------------------------------------------------
// Gap 5: Subgroup operations
// ---------------------------------------------------------------------------

/**
 * Subgroup XOR reduction kernel — sums 4 values using subgroupShuffleXor tree.
 * Each thread starts with its value, then reduces via XOR masks 1 and 2.
 */
const subgroupXorSumSpec: TileKernelSpec = {
  name: "subgroupXorSum",
  workgroupSize: 4,
  enableSubgroups: true,
  bindings: {
    input:  { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { n: "u32" },
  grid: () => [1],
  kernel(ctx) {
    const tid = ctx.localIndex();
    const val = ctx.load("input", tid);
    // XOR tree reduction: 1, 2
    const s1 = val.add(val.subgroupShuffleXor(ctx.const(1, "u32")));
    const s2 = s1.add(s1.subgroupShuffleXor(ctx.const(2, "u32")));
    // Thread 0 writes the result
    ctx.ifThen(tid.eq(ctx.const(0, "u32")), () => {
      ctx.guardedStore("output", tid.eq(ctx.const(0, "u32")), ctx.const(0, "u32"), s2);
    });
  },
};

describe("Gap 5: Subgroup operations", () => {
  it("emits enable subgroups and subgroupShuffleXor", () => {
    const wgsl = compileTileKernel(subgroupXorSumSpec);
    expect(wgsl).toContain("enable subgroups;");
    expect(wgsl).toContain("subgroupShuffleXor(");
  });

  it("does not emit enable subgroups when not requested", () => {
    const wgsl = compileTileKernel(elementwiseAddSpec);
    expect(wgsl).not.toContain("enable subgroups;");
  });
});

// ---------------------------------------------------------------------------
// Gap 6: Auto-vectorization
// ---------------------------------------------------------------------------

/** Vec4 elementwise add — each thread processes 4 elements. */
const elementwiseAddVec4Spec: TileKernelSpec = {
  name: "elementwiseAddVec4",
  workgroupSize: 64,
  vectorize: 4,
  bindings: {
    a:      { storage: "read", type: "f32" },
    b:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { n: "u32" },
  // Grid adjusted for vec4: each thread handles 4 elements
  grid: (u) => [Math.ceil(u.n / (64 * 4))],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const a = ctx.load("a", idx);
      const b = ctx.load("b", idx);
      ctx.guardedStore("output", idx.lt(n), idx, a.add(b));
    });
  },
};

describe("Gap 6: Auto-vectorization", () => {
  it("unrolls body 4 times with base+offset pattern", () => {
    const wgsl = compileTileKernel(elementwiseAddVec4Spec);
    // Should have base computation
    expect(wgsl).toContain("gid.x * 4u");
    // Should have 4 copies of the body with different offsets
    expect(wgsl).toContain("(_base + 0u)");
    expect(wgsl).toContain("(_base + 1u)");
    expect(wgsl).toContain("(_base + 2u)");
    expect(wgsl).toContain("(_base + 3u)");
  });

  it("non-vectorized kernel does not have _base", () => {
    const wgsl = compileTileKernel(elementwiseAddSpec);
    expect(wgsl).not.toContain("_base");
    expect(wgsl).toContain("gid.x");
  });
});

describe.runIf(isWebGPUEnabled)("Gap 6: Auto-vectorization GPU dispatch", () => {
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

  it("vec4 add: 1024 elements (multiple of 4*64)", async () => {
    const N = 1024;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i * 0.1;
      bData[i] = (N - i) * 0.01;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(elementwiseAddVec4Spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("vec4 add: non-multiple of 4 (boundary check)", async () => {
    const N = 137; // not divisible by 4
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i;
      bData[i] = -i * 0.5;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    // Allocate slightly more to avoid out-of-bounds
    const outBuf = makeOutputBuffer(Math.ceil(N / 4) * 4);

    const kernel = createTileKernelDispatcher(elementwiseAddVec4Spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });
});

// ---------------------------------------------------------------------------
// Masked auto-phase elementwise (Triton-like)
// ---------------------------------------------------------------------------

/** Imperative elementwise add with bounds-checked store. */
const maskedElementwiseAddSpec: TileKernelSpec = {
  name: "maskedElementwiseAdd",
  workgroupSize: 256,
  bindings: {
    a:      { storage: "read", type: "f32" },
    b:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { n: "u32" },
  grid: (u) => [Math.ceil(u.n / 256)],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const a = ctx.load("a", idx);
      const b = ctx.load("b", idx);
      ctx.emitStore("output", idx, a.add(b));
    });
  },
};

/** Imperative elementwise relu with bounds-checked store. */
const maskedElementwiseReluSpec: TileKernelSpec = {
  name: "maskedElementwiseRelu",
  workgroupSize: 64,
  bindings: {
    x:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { n: "u32" },
  grid: (u) => [Math.ceil(u.n / 64)],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const x = ctx.load("x", idx);
      const zero = ctx.f32(0.0);
      const result = x.gt(zero).select(x, zero);
      ctx.emitStore("output", idx, result);
    });
  },
};

/** Imperative elementwise add with vectorize=4. */
const maskedElementwiseAddVec4Spec: TileKernelSpec = {
  name: "maskedElementwiseAddVec4",
  workgroupSize: 64,
  vectorize: 4,
  bindings: {
    a:      { storage: "read", type: "f32" },
    b:      { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { n: "u32" },
  grid: (u) => [Math.ceil(u.n / (64 * 4))],
  kernel(ctx) {
    const idx = ctx.globalId(0);
    const n = ctx.uniform("n");
    ctx.ifThen(idx.lt(n), () => {
      const a = ctx.load("a", idx);
      const b = ctx.load("b", idx);
      ctx.guardedStore("output", idx.lt(n), idx, a.add(b));
    });
  },
};

describe("Imperative elementwise (WGSL)", () => {
  it("emits global_invocation_id, no shared memory", () => {
    const wgsl = compileTileKernel(maskedElementwiseAddSpec);
    expect(wgsl).toContain("global_invocation_id");
    expect(wgsl).not.toContain("var<workgroup>");
  });

  it("emits if-guard for bounds check", () => {
    const wgsl = compileTileKernel(maskedElementwiseAddSpec);
    expect(wgsl).toContain("if (");
    expect(wgsl).toContain("output[");
  });

  it("relu kernel emits select", () => {
    const wgsl = compileTileKernel(maskedElementwiseReluSpec);
    expect(wgsl).toContain("global_invocation_id");
    expect(wgsl).toContain("select(");
  });

  it("vectorize works with imperative kernels", () => {
    const wgsl = compileTileKernel(maskedElementwiseAddVec4Spec);
    expect(wgsl).toContain("* 4u");
    expect(wgsl).toContain("(_base + 0u)");
    expect(wgsl).toContain("(_base + 3u)");
  });
});

describe.runIf(isWebGPUEnabled)("Masked auto-phase elementwise GPU dispatch", () => {
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

  it("masked add: 1000 elements", async () => {
    const N = 1000;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i * 0.1;
      bData[i] = (N - i) * 0.01;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(maskedElementwiseAddSpec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("masked add: non-multiple of workgroup size (137)", async () => {
    const N = 137;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i;
      bData[i] = -i * 0.5;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(maskedElementwiseAddSpec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("masked relu: correct output", async () => {
    const N = 500;
    const xData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      xData[i] = (i % 7) - 3; // -3..3
      expected[i] = Math.max(0, xData[i]);
    }

    const xBuf = makeF32Buffer(xData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(maskedElementwiseReluSpec);
    beginSharedEncoder();
    kernel.dispatch({ x: xBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    xBuf.destroy();
    outBuf.destroy();
  });

  it("masked vec4 add: 1024 elements", async () => {
    const N = 1024;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i * 0.01;
      bData[i] = -i * 0.005;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const kernel = createTileKernelDispatcher(maskedElementwiseAddVec4Spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("masked vec4 add: non-multiple-of-4 (137)", async () => {
    const N = 137;
    const aData = new Float32Array(N);
    const bData = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      aData[i] = i;
      bData[i] = -i * 0.5;
      expected[i] = aData[i] + bData[i];
    }

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(Math.ceil(N / 4) * 4);

    const kernel = createTileKernelDispatcher(maskedElementwiseAddVec4Spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { n: N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });
});

// ============================================================================
// Constant folding for guardedStore conditions
// ============================================================================

describe("Compiler: constant-fold guardedStore", () => {
  it("const-true guard (1u == 1u) emits unconditional store", () => {
    const spec: TileKernelSpec = {
      name: "constTrueGuard",
      workgroupSize: 64,
      bindings: {
        a: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { n: "u32" },
      grid: (u) => [Math.ceil(u.n / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("a", gid);
        // Always-true guard: should be constant-folded to unconditional store
        ctx.guardedStore("out", ctx.u32(1).eq(ctx.u32(1)), gid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should NOT have an `if` guard around the store
    expect(wgsl).not.toContain("if ((1u == 1u))");
    // Should have the unconditional store
    expect(wgsl).toContain("out[gid.x] =");
  });

  it("const-false guard (0u == 1u) emits guarded store", () => {
    const spec: TileKernelSpec = {
      name: "constFalseGuard",
      workgroupSize: 64,
      bindings: {
        a: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { n: "u32" },
      grid: (u) => [Math.ceil(u.n / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("a", gid);
        // Always-false guard: should remain guarded
        ctx.guardedStore("out", ctx.u32(0).eq(ctx.u32(1)), gid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should keep the `if` guard (cmp expression has wrapping parens)
    expect(wgsl).toContain("if ((0u == 1u))");
  });

  it("dynamic guard remains conditional", () => {
    const spec: TileKernelSpec = {
      name: "dynGuard",
      workgroupSize: 64,
      bindings: {
        a: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { n: "u32" },
      grid: (u) => [Math.ceil(u.n / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const n = ctx.uniform("n");
        const val = ctx.load("a", gid);
        ctx.guardedStore("out", gid.lt(n), gid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should keep the `if` guard for dynamic condition (cmp has wrapping parens)
    expect(wgsl).toContain("if ((gid.x < config.n))");
  });
});
