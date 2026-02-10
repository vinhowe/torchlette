import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { cpuBackend } from "../../src/backend/cpu";
import {
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

function makeValues(size: number, offset = 0): number[] {
  const values = new Array<number>(size);
  for (let i = 0; i < size; i++) {
    // Use small values to keep products manageable
    values[i] = ((i + offset) % 7) - 3;
  }
  return values;
}

function matmulReference(
  a: number[],
  b: number[],
  m: number,
  n: number,
  k: number,
  transA = false,
  transB = false,
): number[] {
  const out = new Array<number>(m * n).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let t = 0; t < k; t++) {
        const aIdx = transA ? t * m + i : i * k + t;
        const bIdx = transB ? j * k + t : t * n + j;
        sum += a[aIdx] * b[bIdx];
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

function arraysClose(
  actual: number[],
  expected: number[],
  rtol = 1e-4,
  atol = 1e-4,
): boolean {
  if (actual.length !== expected.length) return false;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const bound = atol + rtol * Math.abs(expected[i]);
    if (diff > bound) {
      console.log(
        `Mismatch at ${i}: actual=${actual[i]}, expected=${expected[i]}, diff=${diff}, bound=${bound}`,
      );
      return false;
    }
  }
  return true;
}

describe.runIf(isWebGPUEnabled)("tiled matmul (webgpu)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) {
      console.warn("WebGPU init failed:", getWebGPUInitError());
    }
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  describe("basic correctness", () => {
    it("computes 32x32 matmul correctly", async () => {
      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("computes 64x64 matmul correctly", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("computes 128x128 matmul correctly", async () => {
      const m = 128,
        n = 128,
        k = 128;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("computes 256x256 matmul correctly", async () => {
      const m = 256,
        n = 256,
        k = 256;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });
  });

  describe("non-square matrices", () => {
    it("computes tall-skinny (256x64x256) matmul correctly", async () => {
      const m = 256,
        n = 256,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("computes short-wide (64x256x64) matmul correctly", async () => {
      const m = 64,
        n = 256,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("computes non-tile-aligned (100x150x80) matmul correctly", async () => {
      const m = 100,
        n = 150,
        k = 80;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = matmulReference(aVals, bVals, m, n, k);

      expect(arraysClose(result, expected)).toBe(true);
    });
  });

  describe("cpu vs webgpu parity", () => {
    it("matches cpu backend for 64x64 matmul", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 42);
      const bVals = makeValues(k * n, 17);

      // CPU
      const aCpu = cpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const bCpu = cpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const cCpu = cpuBackend.ops.matmul(aCpu, bCpu);
      const cpuResult = cCpu.toArray();

      // WebGPU
      const aGpu = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const bGpu = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const cGpu = webgpuBackend.ops.matmul(aGpu, bGpu);

      await syncWebGPU();
      const gpuResult = await webgpuBackend.ops.read(cGpu);

      expect(arraysClose(gpuResult, cpuResult)).toBe(true);
    });

    it("matches cpu backend for non-square (128x64x256) matmul", async () => {
      const m = 128,
        n = 256,
        k = 64;
      const aVals = makeValues(m * k, 42);
      const bVals = makeValues(k * n, 17);

      // CPU
      const aCpu = cpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const bCpu = cpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const cCpu = cpuBackend.ops.matmul(aCpu, bCpu);
      const cpuResult = cCpu.toArray();

      // WebGPU
      const aGpu = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const bGpu = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);
      const cGpu = webgpuBackend.ops.matmul(aGpu, bGpu);

      await syncWebGPU();
      const gpuResult = await webgpuBackend.ops.read(cGpu);

      expect(arraysClose(gpuResult, cpuResult)).toBe(true);
    });
  });
});
