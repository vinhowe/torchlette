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

  it("const-false guard (0u == 1u) is eliminated entirely", () => {
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
        // Always-false guard: should be eliminated by constant folding
        ctx.guardedStore("out", ctx.u32(0).eq(ctx.u32(1)), gid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Constant folding folds 0u == 1u → false, guardedStore is eliminated
    expect(wgsl).not.toContain("out[");
    expect(wgsl).not.toContain("if (");
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

// ============================================================================
// Triton Gap: blockWhere + blockRange
// ============================================================================

describe("Triton Gap: blockWhere", () => {
  it("produces select() in WGSL output", () => {
    const spec: TileKernelSpec = {
      name: "blockWhereTest",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        const zero = ctx.f32(0.0);
        const result = ctx.blockWhere(val.gt(zero), val, zero);
        ctx.emitStore("output", gid, result);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("select(");
  });
});

describe("Triton Gap: blockRange", () => {
  it("produces programId * blockSize + localIndex + base", () => {
    const spec: TileKernelSpec = {
      name: "blockRangeTest",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const N = ctx.uniform("N");
        const base = ctx.u32(0);
        const idx = ctx.blockRange(base, 64);
        const val = ctx.load("input", idx);
        ctx.guardedStore("output", idx.lt(N), idx, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("workgroup_id");
    expect(wgsl).toContain("local_invocation_index");
    expect(wgsl).toContain("64u");
  });
});

describe.runIf(isWebGPUEnabled)("Triton Gap: blockWhere + blockRange GPU", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => { await syncWebGPU(); });

  it("blockWhere relu: x > 0 ? x : 0", async () => {
    const N = 256;
    const data = new Float32Array(N);
    const expected = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      data[i] = i - 128; // -128..127
      expected[i] = Math.max(0, data[i]);
    }

    const inBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N);

    const spec: TileKernelSpec = {
      name: "blockWhereRelu",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const N = ctx.uniform("N");
        const val = ctx.load("input", gid);
        const zero = ctx.f32(0.0);
        ctx.guardedStore("output", gid.lt(N), gid, ctx.blockWhere(val.gt(zero), val, zero));
      },
    };

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inBuf, output: outBuf }, { N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 4);
    }

    inBuf.destroy();
    outBuf.destroy();
  });

  it("blockRange elementwise add: matches globalId approach", async () => {
    const N = 256;
    const WG = 64;
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

    const spec: TileKernelSpec = {
      name: "blockRangeAdd",
      workgroupSize: WG,
      bindings: {
        a:      { storage: "read", type: "f32" },
        b:      { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / WG)],
      kernel(ctx) {
        const Nval = ctx.uniform("N");
        const idx = ctx.blockRange(ctx.u32(0), WG);
        const val = ctx.load("a", idx).add(ctx.load("b", idx));
        ctx.guardedStore("output", idx.lt(Nval), idx, val);
      },
    };

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { N });
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
// Triton Gap: Scan Primitives
// ============================================================================

describe("Triton Gap: Scan primitives (WGSL)", () => {
  it("inclusiveScan emits correct number of barriers", () => {
    const WG = 64;
    const spec: TileKernelSpec = {
      name: "inclusiveScanTest",
      workgroupSize: WG,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("scan", WG, "f32");
        smem.write(tid, ctx.load("input", tid));
        ctx.inclusiveScan(smem, tid, WG, "sum");
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };
    const wgsl = compileTileKernel(spec);
    // log2(64) = 6 strides → 6 barriers inside the scan + 1 final barrier = 7
    // plus the initial barrier before scan = total varies
    const barrierCount = (wgsl.match(/workgroupBarrier/g) || []).length;
    expect(barrierCount).toBeGreaterThanOrEqual(7);
  });
});

describe.runIf(isWebGPUEnabled)("Triton Gap: Scan primitives GPU", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => { await syncWebGPU(); });

  it("inclusive prefix sum of [1,1,...,1] → [1,2,3,...,N]", async () => {
    const WG = 64;
    const data = new Float32Array(WG).fill(1.0);

    const inBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(WG);

    const spec: TileKernelSpec = {
      name: "inclusivePrefixSum",
      workgroupSize: WG,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("scan", WG, "f32");
        smem.write(tid, ctx.load("input", tid));
        ctx.inclusiveScan(smem, tid, WG, "sum");
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, WG);
    for (let i = 0; i < WG; i++) {
      expect(result[i]).toBeCloseTo(i + 1, 4);
    }

    inBuf.destroy();
    outBuf.destroy();
  });

  it("exclusive prefix sum of [1,1,...,1] → [0,1,2,...,N-1]", async () => {
    const WG = 64;
    const data = new Float32Array(WG).fill(1.0);

    const inBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(WG);

    const spec: TileKernelSpec = {
      name: "exclusivePrefixSum",
      workgroupSize: WG,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("scan", WG, "f32");
        smem.write(tid, ctx.load("input", tid));
        ctx.exclusiveScan(smem, tid, WG, "sum");
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, WG);
    for (let i = 0; i < WG; i++) {
      expect(result[i]).toBeCloseTo(i, 4);
    }

    inBuf.destroy();
    outBuf.destroy();
  });

  it("inclusive max scan", async () => {
    const WG = 16;
    const data = new Float32Array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]);

    const inBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(WG);

    const spec: TileKernelSpec = {
      name: "inclusiveMaxScan",
      workgroupSize: WG,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("scan", WG, "f32");
        smem.write(tid, ctx.load("input", tid));
        ctx.inclusiveScan(smem, tid, WG, "max");
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, WG);
    // Expected: running max
    let runningMax = -Infinity;
    for (let i = 0; i < WG; i++) {
      runningMax = Math.max(runningMax, data[i]);
      expect(result[i]).toBeCloseTo(runningMax, 4);
    }

    inBuf.destroy();
    outBuf.destroy();
  });
});

// ============================================================================
// Triton Gap: Automatic Barrier Insertion
// ============================================================================

import { insertBarriers, validateBarriers, hoistLoopInvariants } from "../../src/backend/webgpu/tile-compiler";
import { buildKernelIR } from "../../src/backend/webgpu/tile-ir";

describe("Triton Gap: Automatic barrier insertion", () => {
  it("inserts barrier between sharedWrite and sharedRead", () => {
    const spec: TileKernelSpec = {
      name: "autoBarrierSimple",
      workgroupSize: 64,
      autoBarriers: true,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("s", 64, "f32");
        smem.write(tid, ctx.load("input", tid));
        // No manual barrier — autoBarriers should insert one
        const val = smem.read(tid);
        ctx.emitStore("output", tid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("workgroupBarrier");
  });

  it("does not double-barrier when manual barrier present", () => {
    const spec: TileKernelSpec = {
      name: "noDoubleBarrier",
      workgroupSize: 64,
      autoBarriers: true,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("s", 64, "f32");
        smem.write(tid, ctx.load("input", tid));
        ctx.barrier(); // Manual barrier
        const val = smem.read(tid);
        ctx.emitStore("output", tid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should have exactly 1 barrier, not 2
    const barrierCount = (wgsl.match(/workgroupBarrier/g) || []).length;
    expect(barrierCount).toBe(1);
  });

  it("no barriers added when no shared memory", () => {
    const spec: TileKernelSpec = {
      name: "noSharedMem",
      workgroupSize: 64,
      autoBarriers: true,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        ctx.emitStore("output", gid, ctx.load("input", gid));
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).not.toContain("workgroupBarrier");
  });

  it("forRange body: barrier at loop boundary when write-then-read", () => {
    const spec: TileKernelSpec = {
      name: "loopBarrier",
      workgroupSize: 64,
      autoBarriers: true,
      bindings: {
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("s", 64, "f32");
        smem.write(tid, ctx.f32(0.0));
        ctx.forRange(ctx.u32(0), ctx.u32(4), (_k) => {
          // Read from shared (written in previous iteration)
          const val = smem.read(tid);
          smem.write(tid, val.add(ctx.f32(1.0)));
        });
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should have barriers inside the loop
    expect(wgsl).toContain("workgroupBarrier");
  });

  it("validateBarriers reports missing barrier", () => {
    const spec: TileKernelSpec = {
      name: "validateMissing",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("s", 64, "f32");
        smem.write(tid, ctx.load("input", tid));
        // No barrier — validateBarriers should warn
        const val = smem.read(tid);
        ctx.emitStore("output", tid, val);
      },
    };
    const ir = buildKernelIR(spec);
    const warnings = validateBarriers(ir.statements);
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0]).toContain("Missing barrier");
  });
});

// ============================================================================
// Triton Gap: LICM (Loop-Invariant Code Motion)
// ============================================================================

describe("Triton Gap: LICM", () => {
  it("hoists uniform-dependent let out of forRange", () => {
    const spec: TileKernelSpec = {
      name: "licmBasic",
      workgroupSize: 64,
      bindings: {
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        ctx.forRange(ctx.u32(0), ctx.u32(4), (_k) => {
          // This let only depends on uniform N, should be hoisted
          const n = ctx.emitLet("n_val", ctx.uniform("N").toF32());
          ctx.emitStore("output", tid, n);
        });
      },
    };
    const wgsl = compileTileKernel(spec);
    // The let should appear before the for loop, not inside it
    const letPos = wgsl.indexOf("n_val");
    const forPos = wgsl.indexOf("for (");
    expect(letPos).toBeLessThan(forPos);
  });

  it("does NOT hoist loop-variable-dependent let", () => {
    const spec: TileKernelSpec = {
      name: "licmNoHoist",
      workgroupSize: 64,
      bindings: {
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        ctx.forRange(ctx.u32(0), ctx.u32(4), (k) => {
          // This depends on loop var k, should NOT be hoisted
          const val = ctx.emitLet("k_val", k.add(ctx.u32(1)));
          ctx.emitStore("output", tid.add(k), val.toF32());
        });
      },
    };
    const wgsl = compileTileKernel(spec);
    // The let should be inside the for loop
    const letPos = wgsl.indexOf("k_val");
    const forPos = wgsl.indexOf("for (");
    const forEnd = wgsl.indexOf("}", forPos);
    expect(letPos).toBeGreaterThan(forPos);
    expect(letPos).toBeLessThan(forEnd);
  });

  it("does NOT hoist sharedRead when shared array is written in loop", () => {
    const spec: TileKernelSpec = {
      name: "licmNoHoistSmem",
      workgroupSize: 64,
      bindings: {
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("s", 64, "f32");
        smem.write(tid, ctx.f32(0.0));
        ctx.barrier();
        ctx.forRange(ctx.u32(0), ctx.u32(4), (_k) => {
          // Reads smem which is written in this loop → should NOT hoist
          const val = ctx.emitLet("s_val", smem.read(ctx.u32(0)));
          smem.write(tid, val.add(ctx.f32(1.0)));
          ctx.barrier();
        });
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };
    const wgsl = compileTileKernel(spec);
    // s_val should be inside the for loop
    const letPos = wgsl.indexOf("s_val");
    const forPos = wgsl.indexOf("for (");
    expect(letPos).toBeGreaterThan(forPos);
  });
});

// ============================================================================
// Triton Gap: Richer Access Analysis (divisibility)
// ============================================================================

import { analyzeAccessPatterns } from "../../src/backend/webgpu/tile-access-analysis";

describe("Triton Gap: Richer access analysis", () => {
  it("globalId(0) * 4 has divisibility tracking", () => {
    const spec: TileKernelSpec = {
      name: "divTest4",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const idx = gid.mul(ctx.u32(4));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      },
    };
    const patterns = analyzeAccessPatterns(spec);
    const load = patterns.find(p => p.accessType === "load");
    expect(load).toBeDefined();
    expect(load!.innerStride).toBe(4);
    expect(load!.baseDivisibility).toBe(4);
  });

  it("globalId(0) * 2 + 1 has odd constant term", () => {
    const spec: TileKernelSpec = {
      name: "divTestOdd",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const idx = gid.mul(ctx.u32(2)).add(ctx.u32(1));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      },
    };
    const patterns = analyzeAccessPatterns(spec);
    const load = patterns.find(p => p.accessType === "load");
    expect(load).toBeDefined();
    expect(load!.baseConstantTerm).toBe(1);
  });

  it("globalId(0) + const(0) has constantTerm 0", () => {
    const spec: TileKernelSpec = {
      name: "divTestZeroConst",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(0));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      },
    };
    const patterns = analyzeAccessPatterns(spec);
    const load = patterns.find(p => p.accessType === "load");
    expect(load).toBeDefined();
    expect(load!.baseConstantTerm).toBe(0);
    expect(load!.maxVecWidth).toBe(4);
  });

  it("programId(0) * 128 + localIndex has divisibility tracking", () => {
    const spec: TileKernelSpec = {
      name: "divTestProgramId",
      workgroupSize: 64,
      bindings: {
        input:  { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const wid = ctx.programId(0);
        const tid = ctx.localIndex();
        const idx = wid.mul(ctx.u32(128)).add(tid);
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      },
    };
    const patterns = analyzeAccessPatterns(spec);
    const load = patterns.find(p => p.accessType === "load");
    expect(load).toBeDefined();
    expect(load!.innerStride).toBe(1);
    expect(load!.isCoalesced).toBe(true);
  });
});

// ============================================================================
// Round 2 — Remaining Closable Triton Gaps
// ============================================================================

describe("Triton Gap: numPrograms (WGSL)", () => {
  it("numPrograms(0) emits num_wg.x builtin", () => {
    const spec: TileKernelSpec = {
      name: "numProgramsTest",
      workgroupSize: 64,
      bindings: {
        output: { storage: "read_write", type: "u32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const numWg = ctx.numPrograms(0);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, numWg);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("@builtin(num_workgroups) num_wg");
    expect(wgsl).toContain("num_wg.x");
  });

  it("numPrograms(1) emits num_wg.y", () => {
    const spec: TileKernelSpec = {
      name: "numProgramsY",
      workgroupSize: 64,
      bindings: {
        output: { storage: "read_write", type: "u32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [u.N, 4],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const numWgY = ctx.numPrograms(1);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, numWgY);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("num_wg.y");
  });
});

describe("Triton Gap: exp2/log2 (WGSL)", () => {
  it("exp2() emits exp2()", () => {
    const spec: TileKernelSpec = {
      name: "exp2Test",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, val.exp2());
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("exp2(");
  });

  it("log2() emits log2()", () => {
    const spec: TileKernelSpec = {
      name: "log2Test",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, val.log2());
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("log2(");
  });
});

describe("Triton Gap: sigmoid/clamp/fma/erf (WGSL)", () => {
  it("sigmoid() produces exp pattern", () => {
    const spec: TileKernelSpec = {
      name: "sigmoidTest",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, val.sigmoid());
      },
    };
    const wgsl = compileTileKernel(spec);
    // sigmoid(x) = 1 / (1 + exp(-x)) — should contain exp and division
    expect(wgsl).toContain("exp(");
  });

  it("clamp() produces min/max pattern", () => {
    const spec: TileKernelSpec = {
      name: "clampTest",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, val.clamp(ctx.f32(0), ctx.f32(1)));
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("max(");
    expect(wgsl).toContain("min(");
  });

  it("erf() produces polynomial approximation", () => {
    const spec: TileKernelSpec = {
      name: "erfTest",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, val.erf());
      },
    };
    const wgsl = compileTileKernel(spec);
    // erf uses sign, abs, exp, and polynomial coefficients
    expect(wgsl).toContain("sign(");
    expect(wgsl).toContain("exp(");
    expect(wgsl).toContain("abs(");
  });
});

describe("Triton Gap: argmax/argmin (WGSL)", () => {
  it("argmax generates parallel reduction with index tracking", () => {
    const spec: TileKernelSpec = {
      name: "argmaxWGSL",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        const valSmem = ctx.sharedArray("vals", 64, "f32");
        const idxSmem = ctx.sharedArray("idxs", 64, "f32");
        valSmem.write(tid, val);
        idxSmem.write(tid, gid.toF32());
        ctx.treeReduceArgmax(valSmem, idxSmem, tid, 64);
        ctx.ifThen(tid.eq(ctx.u32(0)), () => {
          ctx.emitStore("output", ctx.programId(0), idxSmem.read(ctx.u32(0)));
        });
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should contain shared memory declarations and comparison logic
    expect(wgsl).toContain("var<workgroup> vals");
    expect(wgsl).toContain("var<workgroup> idxs");
    // Should contain workgroupBarrier for tree reduction
    expect(wgsl).toContain("workgroupBarrier()");
    // Should contain comparison (gt) for max finding
    expect(wgsl).toContain(">");
  });
});

describe("Triton Gap: generic associativeScan (WGSL)", () => {
  it("associativeScan generates Hillis-Steele pattern with correct barrier count", () => {
    const spec: TileKernelSpec = {
      name: "assocScanWGSL",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const val = ctx.load("input", tid);
        const smem = ctx.sharedArray("data", 64, "f32");
        smem.write(tid, val);
        // User-defined combine: product
        ctx.associativeScan(smem, tid, 64, (a, b) => a.mul(b));
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should have 2*log2(64)=12 barriers from the scan + 1 final = 13
    const barrierCount = (wgsl.match(/workgroupBarrier\(\)/g) || []).length;
    expect(barrierCount).toBe(13); // 2*6 inside scan + 1 final
  });
});

// ============================================================================
// Round 3: bitwise XOR, cdiv/floorDiv, wgReduceGeneric, atomics, RNG
// ============================================================================

describe("Triton Gap: bitwise XOR (WGSL)", () => {
  it("a.xor(b) compiles to (a ^ b)", () => {
    const spec: TileKernelSpec = {
      name: "xorTest",
      workgroupSize: 64,
      bindings: { input: { storage: "read", type: "u32" }, output: { storage: "read_write", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val.xor(ctx.u32(0xFF)));
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("^");
  });

  it("constant folds xor(5, 3) → 6", () => {
    const spec: TileKernelSpec = {
      name: "xorFold",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        ctx.emitStore("output", gid, ctx.u32(5).xor(ctx.u32(3)));
      },
    };
    const wgsl = compileTileKernel(spec);
    // 5 ^ 3 = 6, should be folded to constant 6
    expect(wgsl).toContain("6u");
    expect(wgsl).not.toContain("^");
  });

  it("x ^ 0 simplifies to x (algebraic identity)", () => {
    const spec: TileKernelSpec = {
      name: "xorZero",
      workgroupSize: 64,
      bindings: { input: { storage: "read", type: "u32" }, output: { storage: "read_write", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val.xor(0));
      },
    };
    const wgsl = compileTileKernel(spec);
    // x ^ 0 should simplify to x, no ^ operator
    expect(wgsl).not.toContain("^");
  });
});

describe("Triton Gap: cdiv / floorDiv (WGSL)", () => {
  it("cdiv(7, 3) constant folds to 3", () => {
    const spec: TileKernelSpec = {
      name: "cdivConst",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        ctx.emitStore("output", gid, ctx.u32(7).cdiv(ctx.u32(3)));
      },
    };
    const wgsl = compileTileKernel(spec);
    // ceil(7/3) = 3
    expect(wgsl).toContain("3u");
  });

  it("cdiv(x, 4) produces (x + 3) / 4 pattern", () => {
    const spec: TileKernelSpec = {
      name: "cdivExpr",
      workgroupSize: 64,
      bindings: { input: { storage: "read", type: "u32" }, output: { storage: "read_write", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val.cdiv(ctx.u32(4)));
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should produce (val + 3u) / 4u
    expect(wgsl).toContain("3u");
    expect(wgsl).toContain("/ 4u");
  });

  it("floorDiv produces division + floor", () => {
    const spec: TileKernelSpec = {
      name: "floorDivExpr",
      workgroupSize: 64,
      bindings: { input: { storage: "read", type: "f32" }, output: { storage: "read_write", type: "f32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val.floorDiv(ctx.f32(3.0)));
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("floor");
  });
});

describe("Triton Gap: atomicXor / atomicExchange / atomicCAS (WGSL)", () => {
  it("atomicXor emits atomicXor WGSL", () => {
    const spec: TileKernelSpec = {
      name: "atomicXorTest",
      workgroupSize: 64,
      bindings: { flags: { storage: "atomic", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const tid = ctx.localIndex();
        ctx.atomicOp("flags", ctx.u32(0), "xor", tid);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("atomicXor");
  });

  it("atomicExchange emits atomicExchange WGSL", () => {
    const spec: TileKernelSpec = {
      name: "atomicExchTest",
      workgroupSize: 64,
      bindings: { flags: { storage: "atomic", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const tid = ctx.localIndex();
        ctx.atomicOp("flags", ctx.u32(0), "exchange", tid);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("atomicExchange");
  });

  it("atomicCAS emits atomicCompareExchangeWeak WGSL", () => {
    const spec: TileKernelSpec = {
      name: "atomicCASTest",
      workgroupSize: 64,
      bindings: { flags: { storage: "atomic", type: "u32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const tid = ctx.localIndex();
        const result = ctx.atomicCAS("flags", ctx.u32(0), ctx.u32(0), tid);
        // Use result to ensure it's not DCE'd
        ctx.atomicOp("flags", ctx.u32(1), "add", result.oldValue);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("atomicCompareExchangeWeak");
    expect(wgsl).toContain(".old_value");
    expect(wgsl).toContain(".exchanged");
  });
});

describe("Triton Gap: wgReduceGeneric (WGSL)", () => {
  it("treeReduceGeneric with product generates multiplication", () => {
    const spec: TileKernelSpec = {
      name: "treeReduceProduct",
      workgroupSize: 64,
      bindings: { input: { storage: "read", type: "f32" }, output: { storage: "read_write", type: "f32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("s", 64);
        smem.write(tid, ctx.load("input", tid));
        ctx.barrier();
        ctx.treeReduceGeneric(smem, tid, 64, (a, b) => a.mul(b));
        ctx.emitStore("output", ctx.u32(0), smem.read(ctx.u32(0)));
      },
    };
    const wgsl = compileTileKernel(spec);
    // Tree reduction with product should contain multiplication
    expect(wgsl).toContain("*");
  });

  it("wgReduceGeneric with min produces min reduction", () => {
    const spec: TileKernelSpec = {
      name: "wgReduceMin",
      workgroupSize: 64,
      bindings: { input: { storage: "read", type: "f32" }, output: { storage: "read_write", type: "f32" } },
      uniforms: {},
      grid: () => [1],
      kernel: (ctx) => {
        const tid = ctx.localIndex();
        const result = ctx.wgReduceGeneric(
          tid, ctx.u32(64), 64,
          ctx.f32(3.402823e+38), // +inf identity
          (i) => ctx.load("input", i),
          (a, b) => a.min(b),
        );
        ctx.emitStore("output", ctx.u32(0), result);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("min(");
  });
});

describe("Triton Gap: Philox RNG (WGSL)", () => {
  it("philox2x32 generates Philox mixing rounds", () => {
    const spec: TileKernelSpec = {
      name: "philoxTest",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "u32" } },
      uniforms: { seed: "u32" },
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const seed = ctx.uniform("seed");
        const [r0] = ctx.philox2x32(seed, gid);
        ctx.emitStore("output", gid, r0);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should contain Philox multiplier constant and XOR operations
    expect(wgsl).toContain("3528905107u"); // PHILOX_M = 0xD256D193
    expect(wgsl).toContain("^");            // XOR mixing
    // Should have 10 rounds of mixing (10 mulhi calls)
    expect(wgsl).toContain("_phi_r9");     // last round
  });

  it("randF32 produces f32 output with u32→f32 cast", () => {
    const spec: TileKernelSpec = {
      name: "randF32Test",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "f32" } },
      uniforms: { seed: "u32" },
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const seed = ctx.uniform("seed");
        const val = ctx.randF32(seed, gid);
        ctx.emitStore("output", gid, val);
      },
    };
    const wgsl = compileTileKernel(spec);
    expect(wgsl).toContain("f32(");  // cast to f32
  });

  it("randF32x2 produces two independent f32 values", () => {
    const spec: TileKernelSpec = {
      name: "randF32x2Test",
      workgroupSize: 64,
      bindings: { out0: { storage: "read_write", type: "f32" }, out1: { storage: "read_write", type: "f32" } },
      uniforms: { seed: "u32" },
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const seed = ctx.uniform("seed");
        const [r0, r1] = ctx.randF32x2(seed, gid);
        ctx.emitStore("out0", gid, r0);
        ctx.emitStore("out1", gid, r1);
      },
    };
    const wgsl = compileTileKernel(spec);
    // Should write to both output bindings
    expect(wgsl).toContain("out0[");
    expect(wgsl).toContain("out1[");
  });
});

// ============================================================================
// GPU tests for Round 2 gaps
// ============================================================================

describe.runIf(isWebGPUEnabled)("Triton Gap Round 2: GPU tests", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  afterAll(async () => { await syncWebGPU(); });

  it("numPrograms returns correct grid dimension", async () => {
    const N = 128;
    const numWg = Math.ceil(N / 64); // = 2

    const spec: TileKernelSpec = {
      name: "numProgramsGPU",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "u32" } },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, ctx.numPrograms(0));
      },
    };
    const kernel = createTileKernelDispatcher(spec);
    const buf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    beginSharedEncoder(); kernel.dispatch({ output: buf }, { N }); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, readBuf, 0, N * 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(readBuf.getMappedRange());
    for (let i = 0; i < N; i++) expect(data[i]).toBe(numWg);
    readBuf.unmap(); buf.destroy(); readBuf.destroy();
  });

  it("exp2: 2^x for x in [0,1,2,3]", async () => {
    const N = 4;
    const spec: TileKernelSpec = {
      name: "exp2GPU",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, ctx.load("input", gid).exp2());
      },
    };
    const inputBuf = makeF32Buffer(new Float32Array([0, 1, 2, 3]), GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ input: inputBuf, output: outBuf }, { N }); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(readBuf.getMappedRange());
    expect(r[0]).toBeCloseTo(1, 5);
    expect(r[1]).toBeCloseTo(2, 5);
    expect(r[2]).toBeCloseTo(4, 5);
    expect(r[3]).toBeCloseTo(8, 5);
    readBuf.unmap(); inputBuf.destroy(); outBuf.destroy(); readBuf.destroy();
  });

  it("sigmoid: sigmoid(0)=0.5, sigmoid(1)≈0.731", async () => {
    const N = 4;
    const spec: TileKernelSpec = {
      name: "sigmoidGPU",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, ctx.load("input", gid).sigmoid());
      },
    };
    const inputBuf = makeF32Buffer(new Float32Array([0, 1, -1, 10]), GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ input: inputBuf, output: outBuf }, { N }); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(readBuf.getMappedRange());
    expect(r[0]).toBeCloseTo(0.5, 4);
    expect(r[1]).toBeCloseTo(0.7311, 3);
    expect(r[2]).toBeCloseTo(0.2689, 3);
    expect(r[3]).toBeCloseTo(1.0, 3);
    readBuf.unmap(); inputBuf.destroy(); outBuf.destroy(); readBuf.destroy();
  });

  it("erf: erf(0)=0, erf(1)≈0.843", async () => {
    const N = 4;
    const spec: TileKernelSpec = {
      name: "erfGPU",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const gid = ctx.globalId(0);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, ctx.load("input", gid).erf());
      },
    };
    const inputBuf = makeF32Buffer(new Float32Array([0, 1, -1, 0.5]), GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ input: inputBuf, output: outBuf }, { N }); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(readBuf.getMappedRange());
    expect(r[0]).toBeCloseTo(0, 4);
    expect(r[1]).toBeCloseTo(0.8427, 3);
    expect(r[2]).toBeCloseTo(-0.8427, 3);
    expect(r[3]).toBeCloseTo(0.5205, 3);
    readBuf.unmap(); inputBuf.destroy(); outBuf.destroy(); readBuf.destroy();
  });

  it("argmax finds index of maximum value", async () => {
    const N = 64;
    const spec: TileKernelSpec = {
      name: "argmaxGPU",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const valSmem = ctx.sharedArray("vals", 64, "f32");
        const idxSmem = ctx.sharedArray("idxs", 64, "f32");
        valSmem.write(tid, ctx.load("input", tid));
        idxSmem.write(tid, tid.toF32());
        ctx.treeReduceArgmax(valSmem, idxSmem, tid, 64);
        ctx.ifThen(tid.eq(ctx.u32(0)), () => {
          ctx.emitStore("output", ctx.u32(0), idxSmem.read(ctx.u32(0)));
        });
      },
    };
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) inputData[i] = i * 0.1;
    inputData[42] = 999.0;
    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ input: inputBuf, output: outBuf }, {}); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    expect(new Float32Array(readBuf.getMappedRange())[0]).toBe(42.0);
    readBuf.unmap(); inputBuf.destroy(); outBuf.destroy(); readBuf.destroy();
  });

  it("argmin finds index of minimum value", async () => {
    const N = 64;
    const spec: TileKernelSpec = {
      name: "argminGPU",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const valSmem = ctx.sharedArray("vals", 64, "f32");
        const idxSmem = ctx.sharedArray("idxs", 64, "f32");
        valSmem.write(tid, ctx.load("input", tid));
        idxSmem.write(tid, tid.toF32());
        ctx.treeReduceArgmin(valSmem, idxSmem, tid, 64);
        ctx.ifThen(tid.eq(ctx.u32(0)), () => {
          ctx.emitStore("output", ctx.u32(0), idxSmem.read(ctx.u32(0)));
        });
      },
    };
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) inputData[i] = i * 0.1 + 10;
    inputData[17] = -999.0;
    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ input: inputBuf, output: outBuf }, {}); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    expect(new Float32Array(readBuf.getMappedRange())[0]).toBe(17.0);
    readBuf.unmap(); inputBuf.destroy(); outBuf.destroy(); readBuf.destroy();
  });

  it("associativeScan with min combine", async () => {
    const N = 64;
    const spec: TileKernelSpec = {
      name: "assocScanMinGPU",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const tid = ctx.localIndex();
        const smem = ctx.sharedArray("data", 64, "f32");
        smem.write(tid, ctx.load("input", tid));
        ctx.associativeScan(smem, tid, 64, (a, b) => a.min(b));
        ctx.emitStore("output", tid, smem.read(tid));
      },
    };
    // Input: [64, 63, 62, ..., 1]
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) inputData[i] = N - i;
    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ input: inputBuf, output: outBuf }, {}); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(readBuf.getMappedRange());
    // Inclusive min-scan of [64, 63, 62, ..., 1]: output[i] = min(input[0..=i])
    // Since input is decreasing, min-scan: output[i] = input[i] = N - i
    for (let i = 0; i < N; i++) expect(r[i]).toBeCloseTo(N - i, 5);
    readBuf.unmap(); inputBuf.destroy(); outBuf.destroy(); readBuf.destroy();
  });
});

// ============================================================================
// GPU tests for Round 3: Philox RNG
// ============================================================================

describe.runIf(isWebGPUEnabled)("Triton Gap Round 3: Philox RNG GPU tests", () => {
  let device: any;
  let queue: any;

  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) throw new Error("WebGPU init failed");
    const ctx = getWebGPUDevice();
    if (!ctx) throw new Error("No WebGPU device");
    device = ctx.device;
    queue = ctx.queue;
  });

  it("randF32 produces values in [0, 1) with reasonable distribution", async () => {
    const N = 1024;
    const spec: TileKernelSpec = {
      name: "randF32GPU",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "f32" } },
      uniforms: { seed: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const seed = ctx.uniform("seed");
        const val = ctx.randF32(seed, gid);
        ctx.guardedStore("output", gid.lt(ctx.uniform("N")), gid, val);
      },
    };
    // Need to declare N uniform too
    spec.uniforms.N = "u32";

    const outBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder(); kernel.dispatch({ output: outBuf }, { seed: 42, N }); flushSharedEncoder(); await syncWebGPU();
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const r = new Float32Array(readBuf.getMappedRange());

    // Check all values are in [0, 1)
    let minVal = Infinity, maxVal = -Infinity, sum = 0;
    for (let i = 0; i < N; i++) {
      expect(r[i]).toBeGreaterThanOrEqual(0);
      expect(r[i]).toBeLessThan(1);
      minVal = Math.min(minVal, r[i]);
      maxVal = Math.max(maxVal, r[i]);
      sum += r[i];
    }
    // Mean should be roughly 0.5 (within reasonable tolerance for N=1024)
    const mean = sum / N;
    expect(mean).toBeGreaterThan(0.3);
    expect(mean).toBeLessThan(0.7);
    // Values should span a good range (not all clustered)
    expect(minVal).toBeLessThan(0.1);
    expect(maxVal).toBeGreaterThan(0.9);

    readBuf.unmap(); outBuf.destroy(); readBuf.destroy();
  });

  it("different seeds produce different sequences", async () => {
    const N = 64;
    const spec: TileKernelSpec = {
      name: "randSeedDiff",
      workgroupSize: 64,
      bindings: { output: { storage: "read_write", type: "f32" } },
      uniforms: { seed: "u32" },
      grid: () => [1],
      kernel: (ctx) => {
        const gid = ctx.globalId(0);
        const seed = ctx.uniform("seed");
        ctx.emitStore("output", gid, ctx.randF32(seed, gid));
      },
    };

    // Generate with seed 1
    const out1 = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read1 = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const k1 = createTileKernelDispatcher(spec);
    beginSharedEncoder(); k1.dispatch({ output: out1 }, { seed: 1 }); flushSharedEncoder(); await syncWebGPU();
    let enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(out1, 0, read1, 0, N * 4);
    queue.submit([enc.finish()]);
    await read1.mapAsync(GPUMapMode.READ);
    const r1 = new Float32Array(read1.getMappedRange().slice(0));
    read1.unmap();

    // Generate with seed 2
    const out2 = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read2 = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const k2 = createTileKernelDispatcher(spec);
    beginSharedEncoder(); k2.dispatch({ output: out2 }, { seed: 2 }); flushSharedEncoder(); await syncWebGPU();
    enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(out2, 0, read2, 0, N * 4);
    queue.submit([enc.finish()]);
    await read2.mapAsync(GPUMapMode.READ);
    const r2 = new Float32Array(read2.getMappedRange().slice(0));
    read2.unmap();

    // At least some values should differ
    let diffCount = 0;
    for (let i = 0; i < N; i++) {
      if (Math.abs(r1[i] - r2[i]) > 1e-7) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(N / 2); // Most values should differ

    out1.destroy(); read1.destroy(); out2.destroy(); read2.destroy();
  });
});
