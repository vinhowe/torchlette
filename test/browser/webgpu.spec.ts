/**
 * Browser-specific WebGPU tests.
 * These tests run in a real browser via Playwright and use native WebGPU.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";

function makeValues(size: number, offset = 0): number[] {
  const values = new Array<number>(size);
  for (let i = 0; i < size; i++) {
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
): number[] {
  const out = new Array<number>(m * n).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let t = 0; t < k; t++) {
        sum += a[i * k + t] * b[t * n + j];
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

describe("WebGPU browser tests", () => {
  let webgpuAvailable = false;

  beforeAll(async () => {
    webgpuAvailable = await initWebGPU();
    if (!webgpuAvailable) {
      console.warn("WebGPU not available:", getWebGPUInitError());
    }
  });

  afterAll(async () => {
    if (webgpuAvailable) {
      await syncWebGPU();
    }
  });

  describe("elementwise operations", () => {
    it("runs add operation", async () => {
      if (!webgpuAvailable) {
        return;
      }

      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]);
      const b = webgpuBackend.ops.tensorFromArray([5, 6, 7, 8], [4]);
      const c = webgpuBackend.ops.add(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      expect(result).toEqual([6, 8, 10, 12]);
    });

    it("runs mul operation", async () => {
      if (!webgpuAvailable) {
        return;
      }

      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]);
      const b = webgpuBackend.ops.tensorFromArray([2, 3, 4, 5], [4]);
      const c = webgpuBackend.ops.mul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      expect(result).toEqual([2, 6, 12, 20]);
    });

    it("runs relu operation", async () => {
      if (!webgpuAvailable) {
        return;
      }

      const a = webgpuBackend.ops.tensorFromArray([-2, -1, 0, 1, 2], [5]);
      const b = webgpuBackend.ops.relu(a);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(b);

      expect(result).toEqual([0, 0, 0, 1, 2]);
    });

    it("supports broadcast", async () => {
      if (!webgpuAvailable) {
        return;
      }

      const a = webgpuBackend.ops.tensorFromArray([1, 2], [2, 1]);
      const b = webgpuBackend.ops.tensorFromArray([10, 20, 30], [1, 3]);
      const c = webgpuBackend.ops.add(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      expect(result).toEqual([11, 21, 31, 12, 22, 32]);
    });
  });

  describe("matmul operations", () => {
    it("computes 32x32 matmul correctly", async () => {
      if (!webgpuAvailable) {
        return;
      }

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
      if (!webgpuAvailable) {
        return;
      }

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
      if (!webgpuAvailable) {
        return;
      }

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

    it("computes non-square (64x128x32) matmul correctly", async () => {
      if (!webgpuAvailable) {
        return;
      }

      const m = 64,
        n = 128,
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
  });

  describe("chained operations", () => {
    it("runs matmul followed by elementwise ops", async () => {
      if (!webgpuAvailable) {
        return;
      }

      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]);

      // matmul -> relu -> mul by 2
      const c = webgpuBackend.ops.matmul(a, b);
      const d = webgpuBackend.ops.relu(c);
      const two = webgpuBackend.ops.tensorFromArray(new Array(m * n).fill(2), [
        m,
        n,
      ]);
      const e = webgpuBackend.ops.mul(d, two);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(e);

      // Verify the result manually for a few elements
      const expected = matmulReference(aVals, bVals, m, n, k);
      const expectedProcessed = expected.map((v) => Math.max(0, v) * 2);

      expect(arraysClose(result, expectedProcessed)).toBe(true);
    });
  });
});
