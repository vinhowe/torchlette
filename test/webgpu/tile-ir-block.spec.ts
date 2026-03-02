/**
 * Tile IR Block API tests.
 *
 * Tests the unified Block type with automatic placement:
 *   1. Block allocation (zeros, full)
 *   2. Block load (thread ptr → register, tile ptr → shared)
 *   3. Block store (register → global)
 *   4. Block dot (register × shared^T, register × shared)
 *   5. Block reduce (max, sum along axis)
 *   6. Block arithmetic with broadcasting
 *   7. Block unary operations
 *   8. WGSL inspection for vec4 patterns
 *
 * Each test has WGSL compilation tests and GPU dispatch tests.
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
import { createTiledMatmulKernel } from "../../src/backend/webgpu/matmul/tile-matmul";
import type { EpilogueConfig } from "../../src/backend/webgpu/matmul/types";
import { DEFAULT_CONFIG } from "../../src/backend/webgpu/matmul/types";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import { createTileKernelDispatcher } from "../../src/backend/webgpu/tile-dispatch";
import type { TileKernelSpec } from "../../src/backend/webgpu/tile-ir";
import { BlockOps } from "../../src/backend/webgpu/tile-ops";

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

// ============================================================================
// Test specs using Block API
// ============================================================================

/**
 * Simple load→store: each thread loads 1 row of D elements, stores them back.
 * Tests: blockLoad (thread ptr) → blockStore.
 */
function makeLoadStoreSpec(D: number): TileKernelSpec {
  return {
    name: `blockLoadStore_D${D}`,
    workgroupSize: 64,
    bindings: {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { N: "u32", D: "u32" },
    grid: (u) => [Math.ceil(u.N / 64)],
    kernel(ctx) {
      const ops = new BlockOps(ctx, { wgSize: 64 });
      const tid = ctx.localIndex();
      const wid = ctx.programId(0);
      const row = wid.mul(ctx.u32(64)).add(tid);
      const N = ctx.uniform("N");
      const Dim = ctx.u32(D);
      const valid = row.lt(N);
      const base = row.mul(Dim);

      // Load one row per thread
      const data = ops.load(
        "input",
        { kind: "thread", base, stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );

      // Store it back
      ops.store("output", data, { base, stride: Dim }, { guard: valid });
    },
  };
}

/**
 * Dot product: Q[1×D] @ K[BC×D]^T = scores[1×BC].
 * Tests: blockLoad (thread + tile) → blockDot (register × shared^T).
 */
function makeDotQKTSpec(D: number, BC: number): TileKernelSpec {
  return {
    name: "blockDotQKT",
    workgroupSize: 64,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { N: "u32", D: "u32", BC: "u32" },
    grid: (u) => [Math.ceil(u.N / 64)],
    kernel(ctx) {
      const ops = new BlockOps(ctx, { wgSize: 64 });
      const tid = ctx.localIndex();
      const wid = ctx.programId(0);
      const row = wid.mul(ctx.u32(64)).add(tid);
      const N = ctx.uniform("N");
      const Dim = ctx.u32(D);
      const valid = row.lt(N);

      // Q row in registers [1×D]
      const q = ops.load(
        "Q",
        { kind: "thread", base: row.mul(Dim), stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );

      // K tile in shared memory [BC×D]
      const kRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo = {
        base: ctx.u32(0).node,
        size: BC,
      };
      const dRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo = {
        base: ctx.u32(0).node,
        size: D,
      };
      const k = ops.load(
        "K",
        {
          kind: "tile",
          baseOffset: ctx.u32(0),
          outerRange: kRange,
          innerRange: dRange,
          outerStride: Dim,
          innerStride: ctx.u32(1),
          outerBound: ctx.u32(BC),
          innerBound: Dim,
        },
        { rows: BC, cols: D },
      );
      ctx.barrier();

      // scores = Q @ K^T  [1×D] × [BC×D]^T → [1×BC]
      const scores = ops.dot(q, k.T());

      // Store scores
      ops.store(
        "output",
        scores,
        { base: row.mul(ctx.u32(BC)), stride: ctx.u32(BC) },
        { guard: valid },
      );
    },
  };
}

/**
 * Dot product: P[1×BC] @ V[BC×D] = out[1×D].
 * Tests: blockDot (register × shared, no transpose).
 */
function makeDotPVSpec(D: number, BC: number): TileKernelSpec {
  return {
    name: "blockDotPV",
    workgroupSize: 64,
    bindings: {
      P: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { N: "u32", D: "u32", BC: "u32" },
    grid: (u) => [Math.ceil(u.N / 64)],
    kernel(ctx) {
      const ops = new BlockOps(ctx, { wgSize: 64 });
      const tid = ctx.localIndex();
      const wid = ctx.programId(0);
      const row = wid.mul(ctx.u32(64)).add(tid);
      const N = ctx.uniform("N");
      const BCu = ctx.u32(BC);
      const Dim = ctx.u32(D);
      const valid = row.lt(N);

      // P row in registers [1×BC]
      const p = ops.load(
        "P",
        { kind: "thread", base: row.mul(BCu), stride: BCu },
        { rows: 1, cols: BC, guard: valid },
      );

      // V tile in shared [BC×D]
      const bRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo = {
        base: ctx.u32(0).node,
        size: BC,
      };
      const dRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo = {
        base: ctx.u32(0).node,
        size: D,
      };
      const v = ops.load(
        "V",
        {
          kind: "tile",
          baseOffset: ctx.u32(0),
          outerRange: bRange,
          innerRange: dRange,
          outerStride: Dim,
          innerStride: ctx.u32(1),
          outerBound: ctx.u32(BC),
          innerBound: Dim,
        },
        { rows: BC, cols: D },
      );
      ctx.barrier();

      // out = P @ V  [1×BC] × [BC×D] → [1×D]
      const out = ops.dot(p, v);

      // Store output
      ops.store(
        "output",
        out,
        { base: row.mul(Dim), stride: Dim },
        { guard: valid },
      );
    },
  };
}

/**
 * Reduce + arithmetic: load row, compute max, subtract max, exp, sum, divide.
 * Tests: blockReduce, blockBinary (broadcasting), blockUnary.
 */
function makeSoftmaxSpec(D: number): TileKernelSpec {
  return {
    name: "blockSoftmax",
    workgroupSize: 64,
    bindings: {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { N: "u32", D: "u32" },
    grid: (u) => [Math.ceil(u.N / 64)],
    kernel(ctx) {
      const ops = new BlockOps(ctx, { wgSize: 64 });
      const tid = ctx.localIndex();
      const wid = ctx.programId(0);
      const row = wid.mul(ctx.u32(64)).add(tid);
      const N = ctx.uniform("N");
      const Dim = ctx.u32(D);
      const valid = row.lt(N);
      const base = row.mul(Dim);

      // Load row [1×D]
      const x = ops.load(
        "input",
        { kind: "thread", base, stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );

      // Softmax: max → subtract → exp → sum → divide
      const maxVal = x.max(1); // [1×1]
      const shifted = x.sub(maxVal); // [1×D] - [1×1] broadcasts
      const exped = shifted.exp(); // [1×D] exp (not in-place, returns new)
      // Wait, .exp() returns new Block, but we need in-place for plan matching.
      // Actually for this test both work. Let's use the functional style.
      const sumVal = exped.sum(1); // [1×1]
      const result = exped.div(sumVal); // [1×D] / [1×1] broadcasts

      ops.store("output", result, { base, stride: Dim }, { guard: valid });
    },
  };
}

// ============================================================================
// WGSL Compilation Tests
// ============================================================================

describe("Block API WGSL compilation", () => {
  it("load→store roundtrip compiles", () => {
    const wgsl = compileTileKernel(makeLoadStoreSpec(8));
    // Should have a register array and per-thread load/store loops
    expect(wgsl).toContain("var blk_0: array<f32,");
    expect(wgsl).toContain("input[");
    expect(wgsl).toContain("output[");
  });

  it("dot QK^T compiles with shared memory", () => {
    const wgsl = compileTileKernel(makeDotQKTSpec(8, 4));
    // Should have shared memory for K tile
    expect(wgsl).toContain("var<workgroup>");
    // Should have the dot inner product accumulation
    expect(wgsl).toContain("_s"); // accumulator variable
  });

  it("dot PV compiles", () => {
    const wgsl = compileTileKernel(makeDotPVSpec(8, 4));
    expect(wgsl).toContain("var<workgroup>");
    // Should have the PV inner product loop reading p
    expect(wgsl).toContain("_p"); // p scalar read
  });

  it("softmax compiles with reduce + broadcast", () => {
    const wgsl = compileTileKernel(makeSoftmaxSpec(8));
    // Should have max, exp, sum operations
    expect(wgsl).toContain("max(");
    expect(wgsl).toContain("exp(");
  });
});

// ============================================================================
// GPU Dispatch Tests
// ============================================================================

describe.runIf(isWebGPUEnabled)("Block API GPU dispatch", () => {
  beforeAll(async () => {
    await initWebGPU();
    const d = getWebGPUDevice();
    device = d.device;
    queue = d.queue;
  });

  afterAll(() => {
    syncWebGPU();
  });

  it("load→store roundtrip: 64 rows × 8 cols", async () => {
    const N = 64;
    const D = 8;
    const inputData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) inputData[i] = i * 0.1;

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N * D);

    const spec = makeLoadStoreSpec(D);
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outputBuf }, { N, D });
    flushSharedEncoder();

    const result = await readF32Buffer(outputBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(inputData[i], 3);
    }

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("load→store roundtrip: non-aligned count (37 rows × 16 cols)", async () => {
    const N = 37;
    const D = 16;
    const inputData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) inputData[i] = Math.sin(i * 0.01);

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N * D);

    const spec = makeLoadStoreSpec(D);
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outputBuf }, { N, D });
    flushSharedEncoder();

    const result = await readF32Buffer(outputBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(inputData[i], 3);
    }

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("dot QK^T: Q[1×D] @ K[BC×D]^T → scores[1×BC]", async () => {
    const N = 4;
    const D = 8;
    const BC = 4;

    // Q: N rows of D elements
    const qData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) qData[i] = (i % D) * 0.1;

    // K: BC rows of D elements (shared tile)
    const kData = new Float32Array(BC * D);
    for (let r = 0; r < BC; r++) {
      for (let d = 0; d < D; d++) {
        kData[r * D + d] = r === d % BC ? 1.0 : 0.0;
      }
    }

    // Expected: scores[n][j] = Σ_d Q[n,d] * K[j,d]
    const expected = new Float32Array(N * BC);
    for (let n = 0; n < N; n++) {
      for (let j = 0; j < BC; j++) {
        let s = 0;
        for (let d = 0; d < D; d++) {
          s += qData[n * D + d] * kData[j * D + d];
        }
        expected[n * BC + j] = s;
      }
    }

    const qBuf = makeF32Buffer(qData, GPUBufferUsage.STORAGE);
    const kBuf = makeF32Buffer(kData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N * BC);

    const spec = makeDotQKTSpec(D, BC);
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ Q: qBuf, K: kBuf, output: outBuf }, { N, D, BC });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N * BC);
    for (let i = 0; i < N * BC; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    qBuf.destroy();
    kBuf.destroy();
    outBuf.destroy();
  });

  it("dot PV: P[1×BC] @ V[BC×D] → out[1×D]", async () => {
    const N = 4;
    const D = 8;
    const BC = 4;

    // P: N rows of BC elements (attention weights)
    const pData = new Float32Array(N * BC);
    for (let n = 0; n < N; n++) {
      // Simple uniform attention weights
      for (let j = 0; j < BC; j++) {
        pData[n * BC + j] = 1.0 / BC;
      }
    }

    // V: BC rows of D elements
    const vData = new Float32Array(BC * D);
    for (let r = 0; r < BC; r++) {
      for (let d = 0; d < D; d++) {
        vData[r * D + d] = r * D + d;
      }
    }

    // Expected: out[n][d] = Σ_j P[n,j] * V[j,d]
    const expected = new Float32Array(N * D);
    for (let n = 0; n < N; n++) {
      for (let d = 0; d < D; d++) {
        let s = 0;
        for (let j = 0; j < BC; j++) {
          s += pData[n * BC + j] * vData[j * D + d];
        }
        expected[n * D + d] = s;
      }
    }

    const pBuf = makeF32Buffer(pData, GPUBufferUsage.STORAGE);
    const vBuf = makeF32Buffer(vData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N * D);

    const spec = makeDotPVSpec(D, BC);
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ P: pBuf, V: vBuf, output: outBuf }, { N, D, BC });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 2);
    }

    pBuf.destroy();
    vBuf.destroy();
    outBuf.destroy();
  });

  it("softmax: reduce + broadcast arithmetic", async () => {
    const N = 8;
    const D = 16;

    const inputData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) inputData[i] = Math.random() * 2 - 1;

    // CPU reference softmax
    const expected = new Float32Array(N * D);
    for (let n = 0; n < N; n++) {
      let maxVal = -Infinity;
      for (let d = 0; d < D; d++)
        maxVal = Math.max(maxVal, inputData[n * D + d]);
      let sum = 0;
      for (let d = 0; d < D; d++) {
        expected[n * D + d] = Math.exp(inputData[n * D + d] - maxVal);
        sum += expected[n * D + d];
      }
      for (let d = 0; d < D; d++) expected[n * D + d] /= sum;
    }

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N * D);

    const spec = makeSoftmaxSpec(D);
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outputBuf }, { N, D });
    flushSharedEncoder();

    const result = await readF32Buffer(outputBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("in-place arithmetic: mul_, sub_, exp_", async () => {
    const N = 4;
    const D = 8;

    const inputData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) inputData[i] = i * 0.1;

    // Kernel: load, mul by 2.0, sub 0.5, exp, store
    const spec: TileKernelSpec = {
      name: "blockInPlace",
      workgroupSize: 64,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 64 });
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const row = wid.mul(ctx.u32(64)).add(tid);
        const N_ = ctx.uniform("N");
        const valid = row.lt(N_);
        const base = row.mul(ctx.u32(D));

        const x = ops.load(
          "input",
          { kind: "thread", base, stride: ctx.u32(D) },
          { rows: 1, cols: D, guard: valid },
        );

        x.mul_(ctx.f32(2.0)); // x *= 2.0
        x.sub_(ctx.f32(0.5)); // x -= 0.5
        x.exp_(); // x = exp(x)

        ops.store("output", x, { base, stride: ctx.u32(D) }, { guard: valid });
      },
    };

    // CPU reference
    const expected = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) {
      expected[i] = Math.exp(inputData[i] * 2.0 - 0.5);
    }

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N * D);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outputBuf }, { N });
    flushSharedEncoder();

    const result = await readF32Buffer(outputBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 2);
    }

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("assign and addAssign", async () => {
    const N = 4;
    const D = 8;

    const aData = new Float32Array(N * D);
    const bData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) {
      aData[i] = i * 0.1;
      bData[i] = (N * D - i) * 0.01;
    }

    // Kernel: load a and b, assign a = b, then addAssign a += b, store
    const spec: TileKernelSpec = {
      name: "blockAssign",
      workgroupSize: 64,
      bindings: {
        a: { storage: "read", type: "f32" },
        b: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 64 });
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const row = wid.mul(ctx.u32(64)).add(tid);
        const N_ = ctx.uniform("N");
        const valid = row.lt(N_);
        const Dim = ctx.u32(D);
        const base = row.mul(Dim);

        const aBlock = ops.load(
          "a",
          { kind: "thread", base, stride: Dim },
          { rows: 1, cols: D, guard: valid },
        );
        const bBlock = ops.load(
          "b",
          { kind: "thread", base, stride: Dim },
          { rows: 1, cols: D, guard: valid },
        );

        aBlock.assign(bBlock); // a = b
        aBlock.addAssign(bBlock); // a += b → a = 2*b

        ops.store("output", aBlock, { base, stride: Dim }, { guard: valid });
      },
    };

    // Expected: 2 * bData
    const expected = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) expected[i] = 2 * bData[i];

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N * D);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, { N });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 3);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("dotAccum: accumulate over multiple tiles", async () => {
    const N = 4;
    const D = 8;
    const BC = 4;

    // Two sets of P and V tiles — result should be sum of two matmuls
    const p1Data = new Float32Array(N * BC);
    const p2Data = new Float32Array(N * BC);
    const v1Data = new Float32Array(BC * D);
    const v2Data = new Float32Array(BC * D);

    for (let i = 0; i < N * BC; i++) {
      p1Data[i] = 0.25; // uniform weights
      p2Data[i] = 0.25;
    }
    for (let i = 0; i < BC * D; i++) {
      v1Data[i] = i * 0.1;
      v2Data[i] = i * 0.01;
    }

    // Expected: P1 @ V1 + P2 @ V2
    const expected = new Float32Array(N * D);
    for (let n = 0; n < N; n++) {
      for (let d = 0; d < D; d++) {
        let s = 0;
        for (let j = 0; j < BC; j++) {
          s += p1Data[n * BC + j] * v1Data[j * D + d];
          s += p2Data[n * BC + j] * v2Data[j * D + d];
        }
        expected[n * D + d] = s;
      }
    }

    // Kernel with dotAccum
    const spec: TileKernelSpec = {
      name: "blockDotAccum",
      workgroupSize: 64,
      bindings: {
        P1: { storage: "read", type: "f32" },
        V1: { storage: "read", type: "f32" },
        P2: { storage: "read", type: "f32" },
        V2: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { N: "u32" },
      grid: (u) => [Math.ceil(u.N / 64)],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 64 });
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const row = wid.mul(ctx.u32(64)).add(tid);
        const N_ = ctx.uniform("N");
        const valid = row.lt(N_);
        const BCu = ctx.u32(BC);
        const Dim = ctx.u32(D);

        // Load P1, P2 per-thread
        const p1 = ops.load(
          "P1",
          { kind: "thread", base: row.mul(BCu), stride: BCu },
          { rows: 1, cols: BC, guard: valid },
        );
        const p2 = ops.load(
          "P2",
          { kind: "thread", base: row.mul(BCu), stride: BCu },
          { rows: 1, cols: BC, guard: valid },
        );

        // V1 tile
        const bRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo =
          {
            base: ctx.u32(0).node,
            size: BC,
          };
        const dRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo =
          {
            base: ctx.u32(0).node,
            size: D,
          };
        const v1 = ops.load(
          "V1",
          {
            kind: "tile",
            baseOffset: ctx.u32(0),
            outerRange: bRange,
            innerRange: dRange,
            outerStride: Dim,
            innerStride: ctx.u32(1),
            outerBound: ctx.u32(BC),
            innerBound: Dim,
          },
          { rows: BC, cols: D },
        );
        ctx.barrier();

        // acc = P1 @ V1
        const acc = ops.dot(p1, v1);
        ctx.barrier();

        // Load V2
        const v2 = ops.load(
          "V2",
          {
            kind: "tile",
            baseOffset: ctx.u32(0),
            outerRange: { base: ctx.u32(0).node, size: BC },
            innerRange: { base: ctx.u32(0).node, size: D },
            outerStride: Dim,
            innerStride: ctx.u32(1),
            outerBound: ctx.u32(BC),
            innerBound: Dim,
          },
          { rows: BC, cols: D },
        );
        ctx.barrier();

        // acc += P2 @ V2
        ops.dotAccum(p2, v2, acc);

        ops.store(
          "output",
          acc,
          { base: row.mul(Dim), stride: Dim },
          { guard: valid },
        );
      },
    };

    const p1Buf = makeF32Buffer(p1Data, GPUBufferUsage.STORAGE);
    const v1Buf = makeF32Buffer(v1Data, GPUBufferUsage.STORAGE);
    const p2Buf = makeF32Buffer(p2Data, GPUBufferUsage.STORAGE);
    const v2Buf = makeF32Buffer(v2Data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(N * D);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch(
      { P1: p1Buf, V1: v1Buf, P2: p2Buf, V2: v2Buf, output: outBuf },
      { N },
    );
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, N * D);
    for (let i = 0; i < N * D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 2);
    }

    p1Buf.destroy();
    v1Buf.destroy();
    p2Buf.destroy();
    v2Buf.destroy();
    outBuf.destroy();
  });

  it("shared×shared dot: 32×32×16 outer product (t4×4)", async () => {
    const tileM = 32,
      tileN = 32,
      tileK = 16;
    const ttM = 4,
      ttN = 4;
    const wgX = tileN / ttN; // 8
    const wgY = tileM / ttM; // 8
    const M = tileM,
      N = tileN,
      K = tileK;

    const aData = new Float32Array(M * K);
    const bData = new Float32Array(K * N);
    for (let i = 0; i < M * K; i++) aData[i] = Math.sin(i * 0.1) * 0.5;
    for (let i = 0; i < K * N; i++) bData[i] = Math.cos(i * 0.13) * 0.5;

    // CPU reference: C = A @ B
    const expected = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += aData[m * K + k] * bData[k * N + n];
        expected[m * N + n] = s;
      }
    }

    const spec: TileKernelSpec = {
      name: "blockSharedDot",
      workgroupSize: [wgX, wgY],
      bindings: {
        A: { storage: "read", type: "f32" },
        B: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, {
          wgSize: [wgX, wgY],
          threadTile: [ttM, ttN],
        });

        const threadRow = ctx.emitLet("thread_row", ctx.threadIdx(1));
        const threadCol = ctx.emitLet("thread_col", ctx.threadIdx(0));

        // Cooperative load A [tileM × tileK] → shared
        const aRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo =
          {
            base: ctx.u32(0).node,
            size: tileM,
          };
        const kRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo =
          {
            base: ctx.u32(0).node,
            size: tileK,
          };
        const a = ops.load(
          "A",
          {
            kind: "tile",
            baseOffset: ctx.u32(0),
            outerRange: aRange,
            innerRange: kRange,
            outerStride: ctx.u32(K),
            innerStride: ctx.u32(1),
            outerBound: ctx.u32(M),
            innerBound: ctx.u32(K),
          },
          { rows: tileM, cols: tileK },
        );

        // Cooperative load B [tileK × tileN] → shared
        const bRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo =
          {
            base: ctx.u32(0).node,
            size: tileK,
          };
        const nRange: import("../../src/backend/webgpu/tile-ir").TileRangeInfo =
          {
            base: ctx.u32(0).node,
            size: tileN,
          };
        const b = ops.load(
          "B",
          {
            kind: "tile",
            baseOffset: ctx.u32(0),
            outerRange: bRange,
            innerRange: nRange,
            outerStride: ctx.u32(N),
            innerStride: ctx.u32(1),
            outerBound: ctx.u32(K),
            innerBound: ctx.u32(N),
          },
          { rows: tileK, cols: tileN },
        );

        // Outer product: acc = A @ B
        const acc = ops.dot(a, b);

        // Store: each thread stores its [ttM × ttN] tile
        ctx.forRange(ctx.u32(0), ctx.u32(ttM), (tm) => {
          ctx.forRange(ctx.u32(0), ctx.u32(ttN), (tn) => {
            const outRow = threadRow.mul(ctx.u32(ttM)).add(tm);
            const outCol = threadCol.mul(ctx.u32(ttN)).add(tn);
            const outIdx = outRow.mul(ctx.u32(N)).add(outCol);
            const val = acc.get(tm, tn);
            ctx.pushStatement({
              kind: "indexAssign",
              arrayName: "out",
              idx: outIdx.node,
              value: val.node,
            });
          });
        });
      },
    };

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(M * N);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ A: aBuf, B: bBuf, out: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, M * N);
    let maxErr = 0;
    for (let i = 0; i < M * N; i++) {
      maxErr = Math.max(maxErr, Math.abs(result[i] - expected[i]));
    }
    expect(maxErr).toBeLessThan(1e-4);

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("shared×shared dotAccum: K-loop over multiple tiles", async () => {
    const tileM = 32,
      tileN = 32,
      tileK = 16;
    const ttM = 4,
      ttN = 4;
    const wgX = tileN / ttN; // 8
    const wgY = tileM / ttM; // 8
    const M = tileM,
      N = tileN,
      K = 48; // K > tileK, needs multiple tiles

    const aData = new Float32Array(M * K);
    const bData = new Float32Array(K * N);
    for (let i = 0; i < M * K; i++) aData[i] = Math.sin(i * 0.07) * 0.3;
    for (let i = 0; i < K * N; i++) bData[i] = Math.cos(i * 0.11) * 0.3;

    // CPU reference
    const expected = new Float32Array(M * N);
    for (let m = 0; m < M; m++) {
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += aData[m * K + k] * bData[k * N + n];
        expected[m * N + n] = s;
      }
    }

    const spec: TileKernelSpec = {
      name: "blockSharedDotAccum",
      workgroupSize: [wgX, wgY],
      bindings: {
        A: { storage: "read", type: "f32" },
        B: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: { M: "u32", N: "u32", K: "u32" },
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, {
          wgSize: [wgX, wgY],
          threadTile: [ttM, ttN],
        });
        const threadRow = ctx.emitLet("thread_row", ctx.threadIdx(1));
        const threadCol = ctx.emitLet("thread_col", ctx.threadIdx(0));
        const Ku = ctx.uniform("K");
        const numTiles = ctx.emitLet(
          "num_tiles",
          Ku.add(ctx.u32(tileK - 1)).div(ctx.u32(tileK)),
        );

        const acc = ops.zeros(ttM, ttN);

        ctx.forRange(ctx.u32(0), numTiles, (kTile) => {
          const kOff = kTile.mul(ctx.u32(tileK));
          const a = ops.load(
            "A",
            {
              kind: "tile",
              baseOffset: ctx.u32(0),
              outerRange: { base: ctx.u32(0).node, size: tileM },
              innerRange: { base: kOff.node, size: tileK },
              outerStride: Ku,
              innerStride: ctx.u32(1),
              outerBound: ctx.u32(M),
              innerBound: Ku,
            },
            { rows: tileM, cols: tileK },
          );

          const b = ops.load(
            "B",
            {
              kind: "tile",
              baseOffset: ctx.u32(0),
              outerRange: { base: kOff.node, size: tileK },
              innerRange: { base: ctx.u32(0).node, size: tileN },
              outerStride: ctx.u32(N),
              innerStride: ctx.u32(1),
              outerBound: Ku,
              innerBound: ctx.u32(N),
            },
            { rows: tileK, cols: tileN },
          );

          ops.dotAccum(a, b, acc);
        });

        // Store
        ctx.forRange(ctx.u32(0), ctx.u32(ttM), (tm) => {
          ctx.forRange(ctx.u32(0), ctx.u32(ttN), (tn) => {
            const outRow = threadRow.mul(ctx.u32(ttM)).add(tm);
            const outCol = threadCol.mul(ctx.u32(ttN)).add(tn);
            const outIdx = outRow.mul(ctx.u32(N)).add(outCol);
            ctx.pushStatement({
              kind: "indexAssign",
              arrayName: "out",
              idx: outIdx.node,
              value: acc.get(tm, tn).node,
            });
          });
        });
      },
    };

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(M * N);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ A: aBuf, B: bBuf, out: outBuf }, { M, N, K });
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, M * N);
    let maxErr = 0;
    for (let i = 0; i < M * N; i++) {
      maxErr = Math.max(maxErr, Math.abs(result[i] - expected[i]));
    }
    expect(maxErr).toBeLessThan(1e-3);

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("axis=0 reduce (sum columns)", async () => {
    const N = 8;
    const D = 16;

    const inputData = new Float32Array(N * D);
    for (let i = 0; i < N * D; i++) inputData[i] = (i % D) * 0.1;

    // axis=0: reduce rows → [1×D]
    // But Block axis=0 reduce operates on a single block — we load a block [N×D] per thread
    // Actually for register blocks, multi-row blocks need multi-row loads
    // Let's test with a small block that fits in registers

    // Simpler test: load a [4×8] block per thread, reduce axis=0 → [1×8]
    const ROWS = 4,
      COLS = 8;
    const data = new Float32Array(ROWS * COLS);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        data[r * COLS + c] = r + c * 0.1;
      }
    }
    const expected = new Float32Array(COLS);
    for (let c = 0; c < COLS; c++) {
      let s = 0;
      for (let r = 0; r < ROWS; r++) s += data[r * COLS + c];
      expected[c] = s;
    }

    const spec: TileKernelSpec = {
      name: "blockReduceAxis0",
      workgroupSize: 1,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 1 });
        // Load all ROWS×COLS into a register block
        const block = ops.load(
          "input",
          { kind: "thread", base: ctx.u32(0), stride: ctx.u32(COLS) },
          { rows: ROWS, cols: COLS },
        );
        const reduced = block.sum(0); // [1×COLS]
        ops.store("output", reduced, {
          base: ctx.u32(0),
          stride: ctx.u32(COLS),
        });
      },
    };

    const inputBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(COLS);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, COLS);
    for (let c = 0; c < COLS; c++) {
      expect(result[c]).toBeCloseTo(expected[c], 3);
    }

    inputBuf.destroy();
    outBuf.destroy();
  });

  it("axis=0 reduce (max columns)", async () => {
    const ROWS = 4,
      COLS = 8;
    const data = new Float32Array(ROWS * COLS);
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        data[r * COLS + c] = Math.sin(r * 3.0 + c * 0.7);
      }
    }
    const expected = new Float32Array(COLS);
    for (let c = 0; c < COLS; c++) {
      let m = -Infinity;
      for (let r = 0; r < ROWS; r++) m = Math.max(m, data[r * COLS + c]);
      expected[c] = m;
    }

    const spec: TileKernelSpec = {
      name: "blockReduceAxis0Max",
      workgroupSize: 1,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 1 });
        const block = ops.load(
          "input",
          { kind: "thread", base: ctx.u32(0), stride: ctx.u32(COLS) },
          { rows: ROWS, cols: COLS },
        );
        const reduced = block.max(0); // [1×COLS]
        ops.store("output", reduced, {
          base: ctx.u32(0),
          stride: ctx.u32(COLS),
        });
      },
    };

    const inputBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(COLS);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, COLS);
    for (let c = 0; c < COLS; c++) {
      expect(result[c]).toBeCloseTo(expected[c], 3);
    }

    inputBuf.destroy();
    outBuf.destroy();
  });

  it("neg() and log() unary ops", async () => {
    const D = 8;
    const data = new Float32Array(D);
    for (let i = 0; i < D; i++) data[i] = (i + 1) * 0.5; // positive for log

    const expectedNeg = new Float32Array(D);
    const expectedLog = new Float32Array(D);
    for (let i = 0; i < D; i++) {
      expectedNeg[i] = -data[i];
      expectedLog[i] = Math.log(data[i]);
    }

    const spec: TileKernelSpec = {
      name: "blockNegLog",
      workgroupSize: 1,
      bindings: {
        input: { storage: "read", type: "f32" },
        outNeg: { storage: "read_write", type: "f32" },
        outLog: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 1 });
        const x = ops.load(
          "input",
          { kind: "thread", base: ctx.u32(0), stride: ctx.u32(D) },
          { rows: 1, cols: D },
        );
        const n = x.neg();
        const l = x.log();
        ops.store("outNeg", n, { base: ctx.u32(0), stride: ctx.u32(D) });
        ops.store("outLog", l, { base: ctx.u32(0), stride: ctx.u32(D) });
      },
    };

    const inputBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const negBuf = makeOutputBuffer(D);
    const logBuf = makeOutputBuffer(D);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, outNeg: negBuf, outLog: logBuf }, {});
    flushSharedEncoder();

    const negResult = await readF32Buffer(negBuf, D);
    const logResult = await readF32Buffer(logBuf, D);
    for (let i = 0; i < D; i++) {
      expect(negResult[i]).toBeCloseTo(expectedNeg[i], 4);
      expect(logResult[i]).toBeCloseTo(expectedLog[i], 4);
    }

    inputBuf.destroy();
    negBuf.destroy();
    logBuf.destroy();
  });

  it("elementwise max(Block, Block)", async () => {
    const D = 8;
    const aData = new Float32Array(D);
    const bData = new Float32Array(D);
    for (let i = 0; i < D; i++) {
      aData[i] = Math.sin(i);
      bData[i] = Math.cos(i);
    }
    const expected = new Float32Array(D);
    for (let i = 0; i < D; i++) expected[i] = Math.max(aData[i], bData[i]);

    const spec: TileKernelSpec = {
      name: "blockElemMax",
      workgroupSize: 1,
      bindings: {
        a: { storage: "read", type: "f32" },
        b: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 1 });
        const aBlock = ops.load(
          "a",
          { kind: "thread", base: ctx.u32(0), stride: ctx.u32(D) },
          { rows: 1, cols: D },
        );
        const bBlock = ops.load(
          "b",
          { kind: "thread", base: ctx.u32(0), stride: ctx.u32(D) },
          { rows: 1, cols: D },
        );
        const result = aBlock.max(bBlock);
        ops.store("output", result, { base: ctx.u32(0), stride: ctx.u32(D) });
      },
    };

    const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
    const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(D);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ a: aBuf, b: bBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, D);
    for (let i = 0; i < D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 4);
    }

    aBuf.destroy();
    bBuf.destroy();
    outBuf.destroy();
  });

  it("full() constant initialization", async () => {
    const D = 8;
    const fillVal = 42.5;
    const expected = new Float32Array(D);
    for (let i = 0; i < D; i++) expected[i] = fillVal;

    const spec: TileKernelSpec = {
      name: "blockFull",
      workgroupSize: 1,
      bindings: {
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 1 });
        const block = ops.full(1, D, fillVal);
        ops.store("output", block, { base: ctx.u32(0), stride: ctx.u32(D) });
      },
    };

    const outBuf = makeOutputBuffer(D);
    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, D);
    for (let i = 0; i < D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 4);
    }
    outBuf.destroy();
  });

  it("apply_() with relu", async () => {
    const D = 8;
    const data = new Float32Array(D);
    for (let i = 0; i < D; i++) data[i] = (i - 4) * 0.5; // mix of positive and negative

    const expected = new Float32Array(D);
    for (let i = 0; i < D; i++) expected[i] = Math.max(0, data[i]);

    const spec: TileKernelSpec = {
      name: "blockApplyRelu",
      workgroupSize: 1,
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: {},
      grid: () => [1],
      kernel(ctx) {
        const ops = new BlockOps(ctx, { wgSize: 1 });
        const x = ops.load(
          "input",
          { kind: "thread", base: ctx.u32(0), stride: ctx.u32(D) },
          { rows: 1, cols: D },
        );
        x.apply_((val) => val.gt(ctx.f32(0.0)).select(val, ctx.f32(0.0)));
        ops.store("output", x, { base: ctx.u32(0), stride: ctx.u32(D) });
      },
    };

    const inputBuf = makeF32Buffer(data, GPUBufferUsage.STORAGE);
    const outBuf = makeOutputBuffer(D);

    const kernel = createTileKernelDispatcher(spec);
    beginSharedEncoder();
    kernel.dispatch({ input: inputBuf, output: outBuf }, {});
    flushSharedEncoder();

    const result = await readF32Buffer(outBuf, D);
    for (let i = 0; i < D; i++) {
      expect(result[i]).toBeCloseTo(expected[i], 4);
    }

    inputBuf.destroy();
    outBuf.destroy();
  });
});

// ============================================================================
// Block API Matmul Epilogue Tests
// ============================================================================

describe.runIf(isWebGPUEnabled)(
  "Block API matmul epilogue (BlockOps path)",
  () => {
    const origEnv = process.env.TORCHLETTE_BLOCK_MATMUL;

    beforeAll(async () => {
      process.env.TORCHLETTE_BLOCK_MATMUL = "1";
      if (!device) {
        const ok = await initWebGPU();
        if (ok) {
          device = getWebGPUDevice() as GPUDevice;
          queue = device.queue;
        }
      }
    });

    afterAll(() => {
      if (origEnv === undefined) delete process.env.TORCHLETTE_BLOCK_MATMUL;
      else process.env.TORCHLETTE_BLOCK_MATMUL = origEnv;
    });

    // Tile-aligned size: 32×32 fits exactly in one workgroup with DEFAULT_CONFIG
    const M = 32,
      N = 32,
      K = 32;
    const config = DEFAULT_CONFIG;
    const baseUniforms = {
      m: M,
      n: N,
      k: K,
      lda: K,
      ldb: N,
      ldc: N,
      alpha: 1.0,
      batchSize: 1,
      batchStrideA: 0,
      batchStrideB: 0,
      batchStrideC: 0,
    };

    function cpuMatmul(aData: Float32Array, bData: Float32Array): Float32Array {
      const out = new Float32Array(M * N);
      for (let m = 0; m < M; m++) {
        for (let n = 0; n < N; n++) {
          let s = 0;
          for (let k = 0; k < K; k++) s += aData[m * K + k] * bData[k * N + n];
          out[m * N + n] = s;
        }
      }
      return out;
    }

    function makeTestData() {
      const aData = new Float32Array(M * K);
      const bData = new Float32Array(K * N);
      for (let i = 0; i < M * K; i++) aData[i] = Math.sin(i * 0.07) * 0.3;
      for (let i = 0; i < K * N; i++) bData[i] = Math.cos(i * 0.11) * 0.3;
      return { aData, bData, expected: cpuMatmul(aData, bData) };
    }

    function maxError(actual: Float32Array, expected: Float32Array): number {
      let maxErr = 0;
      for (let i = 0; i < actual.length; i++) {
        maxErr = Math.max(maxErr, Math.abs(actual[i] - expected[i]));
      }
      return maxErr;
    }

    it("matmul + bias epilogue", async () => {
      const { aData, bData, expected } = makeTestData();
      const biasData = new Float32Array(N);
      for (let i = 0; i < N; i++) biasData[i] = (i - N / 2) * 0.1;
      // Apply bias to reference
      const ref = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++) ref[i] = expected[i] + biasData[i % N];

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };
      const spec = createTiledMatmulKernel({
        config,
        transposeMode: "NN",
        dtype: "f32",
        epilogue,
      });

      const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
      const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
      const biasBuf = makeF32Buffer(biasData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(M * N);

      const kernel = createTileKernelDispatcher(spec);
      beginSharedEncoder();
      kernel.dispatch(
        { a: aBuf, b: bBuf, out: outBuf, epilogue_in0: biasBuf },
        baseUniforms,
      );
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, M * N);
      expect(maxError(result, ref)).toBeLessThan(1e-3);

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      outBuf.destroy();
    });

    it("matmul + relu epilogue", async () => {
      const { aData, bData, expected } = makeTestData();
      const ref = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++) ref[i] = Math.max(0, expected[i]);

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "unary", op: "relu" }],
        additionalInputCount: 0,
        outputDtype: "f32",
      };
      const spec = createTiledMatmulKernel({
        config,
        transposeMode: "NN",
        dtype: "f32",
        epilogue,
      });

      const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
      const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(M * N);

      const kernel = createTileKernelDispatcher(spec);
      beginSharedEncoder();
      kernel.dispatch({ a: aBuf, b: bBuf, out: outBuf }, baseUniforms);
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, M * N);
      expect(maxError(result, ref)).toBeLessThan(1e-3);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });

    it("matmul + bias + relu chain", async () => {
      const { aData, bData, expected } = makeTestData();
      const biasData = new Float32Array(N);
      for (let i = 0; i < N; i++) biasData[i] = (i - N / 2) * 0.1;
      const ref = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++)
        ref[i] = Math.max(0, expected[i] + biasData[i % N]);

      const epilogue: EpilogueConfig = {
        ops: [
          { kind: "bias", inputIndex: 0 },
          { kind: "unary", op: "relu" },
        ],
        additionalInputCount: 1,
        outputDtype: "f32",
      };
      const spec = createTiledMatmulKernel({
        config,
        transposeMode: "NN",
        dtype: "f32",
        epilogue,
      });

      const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
      const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
      const biasBuf = makeF32Buffer(biasData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(M * N);

      const kernel = createTileKernelDispatcher(spec);
      beginSharedEncoder();
      kernel.dispatch(
        { a: aBuf, b: bBuf, out: outBuf, epilogue_in0: biasBuf },
        baseUniforms,
      );
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, M * N);
      expect(maxError(result, ref)).toBeLessThan(1e-3);

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      outBuf.destroy();
    });

    it("matmul + binary add (residual)", async () => {
      const { aData, bData, expected } = makeTestData();
      // Residual tensor: same shape as output (M×N)
      const residualData = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++)
        residualData[i] = Math.sin(i * 0.13) * 0.5;
      const ref = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++) ref[i] = expected[i] + residualData[i];

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "binary", op: "add", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };
      const spec = createTiledMatmulKernel({
        config,
        transposeMode: "NN",
        dtype: "f32",
        epilogue,
      });

      const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
      const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
      const resBuf = makeF32Buffer(residualData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(M * N);

      const kernel = createTileKernelDispatcher(spec);
      beginSharedEncoder();
      kernel.dispatch(
        { a: aBuf, b: bBuf, out: outBuf, epilogue_in0: resBuf },
        baseUniforms,
      );
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, M * N);
      expect(maxError(result, ref)).toBeLessThan(1e-3);

      aBuf.destroy();
      bBuf.destroy();
      resBuf.destroy();
      outBuf.destroy();
    });

    it("matmul + bias + binary add + relu chain", async () => {
      const { aData, bData, expected } = makeTestData();
      const biasData = new Float32Array(N);
      for (let i = 0; i < N; i++) biasData[i] = (i - N / 2) * 0.1;
      const residualData = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++)
        residualData[i] = Math.sin(i * 0.13) * 0.5;
      // Reference: matmul → +bias → +residual → relu
      const ref = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++) {
        ref[i] = Math.max(0, expected[i] + biasData[i % N] + residualData[i]);
      }

      const epilogue: EpilogueConfig = {
        ops: [
          { kind: "bias", inputIndex: 0 },
          { kind: "binary", op: "add", inputIndex: 1 },
          { kind: "unary", op: "relu" },
        ],
        additionalInputCount: 2,
        outputDtype: "f32",
      };
      const spec = createTiledMatmulKernel({
        config,
        transposeMode: "NN",
        dtype: "f32",
        epilogue,
      });

      const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
      const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
      const biasBuf = makeF32Buffer(biasData, GPUBufferUsage.STORAGE);
      const resBuf = makeF32Buffer(residualData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(M * N);

      const kernel = createTileKernelDispatcher(spec);
      beginSharedEncoder();
      kernel.dispatch(
        {
          a: aBuf,
          b: bBuf,
          out: outBuf,
          epilogue_in0: biasBuf,
          epilogue_in1: resBuf,
        },
        baseUniforms,
      );
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, M * N);
      expect(maxError(result, ref)).toBeLessThan(1e-3);

      aBuf.destroy();
      bBuf.destroy();
      biasBuf.destroy();
      resBuf.destroy();
      outBuf.destroy();
    });

    it("matmul + gelu epilogue", async () => {
      const { aData, bData, expected } = makeTestData();
      const ref = new Float32Array(M * N);
      for (let i = 0; i < M * N; i++) {
        const x = expected[i];
        const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        ref[i] = 0.5 * x * (1 + Math.tanh(inner));
      }

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "unary", op: "gelu" }],
        additionalInputCount: 0,
        outputDtype: "f32",
      };
      const spec = createTiledMatmulKernel({
        config,
        transposeMode: "NN",
        dtype: "f32",
        epilogue,
      });

      const aBuf = makeF32Buffer(aData, GPUBufferUsage.STORAGE);
      const bBuf = makeF32Buffer(bData, GPUBufferUsage.STORAGE);
      const outBuf = makeOutputBuffer(M * N);

      const kernel = createTileKernelDispatcher(spec);
      beginSharedEncoder();
      kernel.dispatch({ a: aBuf, b: bBuf, out: outBuf }, baseUniforms);
      flushSharedEncoder();

      const result = await readF32Buffer(outBuf, M * N);
      expect(maxError(result, ref)).toBeLessThan(1e-3);

      aBuf.destroy();
      bBuf.destroy();
      outBuf.destroy();
    });
  },
);
