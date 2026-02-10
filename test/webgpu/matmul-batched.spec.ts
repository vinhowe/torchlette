/**
 * Tests for batched matmul with ND broadcasting support.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
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
    values[i] = ((i + offset) % 7) - 3;
  }
  return values;
}

/**
 * Reference batched matmul implementation for testing.
 */
function referenceBatchedMatmul(
  aData: number[],
  bData: number[],
  aShape: number[],
  bShape: number[],
  outShape: number[],
): number[] {
  const aRank = aShape.length;
  const bRank = bShape.length;

  const m = aShape[aRank - 2];
  const k = aShape[aRank - 1];
  const n = bShape[bRank - 1];

  const aBatchDims = aShape.slice(0, -2);
  const bBatchDims = bShape.slice(0, -2);
  const outBatchDims = outShape.slice(0, -2);

  const outBatchSize = outBatchDims.reduce((a, b) => a * b, 1);
  const aBatchSize = aBatchDims.reduce((a, b) => a * b, 1);
  const bBatchSize = bBatchDims.reduce((a, b) => a * b, 1);

  const result = new Array(outBatchSize * m * n).fill(0);

  for (let batch = 0; batch < outBatchSize; batch++) {
    // Compute batch indices for A and B with broadcasting
    const aBatch = aBatchSize === 1 ? 0 : batch % aBatchSize;
    const bBatch = bBatchSize === 1 ? 0 : batch % bBatchSize;

    const aOffset = aBatch * m * k;
    const bOffset = bBatch * k * n;
    const outOffset = batch * m * n;

    // Standard matmul for this batch
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let p = 0; p < k; p++) {
          sum += aData[aOffset + i * k + p] * bData[bOffset + p * n + j];
        }
        result[outOffset + i * n + j] = sum;
      }
    }
  }

  return result;
}

function arraysClose(
  actual: number[],
  expected: number[],
  rtol = 1e-4,
  atol = 1e-4,
): boolean {
  if (actual.length !== expected.length) {
    console.log(
      `Length mismatch: actual=${actual.length}, expected=${expected.length}`,
    );
    return false;
  }
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

describe.runIf(isWebGPUEnabled)("batched matmul (webgpu)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) {
      console.warn("WebGPU init failed:", getWebGPUInitError());
    }
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  describe("3D batched matmul", () => {
    it("computes [2, 32, 32] @ [2, 32, 32] correctly", async () => {
      const batchSize = 2;
      const m = 32,
        k = 32,
        n = 32;

      const aData = makeValues(batchSize * m * k, 0);
      const bData = makeValues(batchSize * k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [batchSize, m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [batchSize, k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [batchSize, m, k],
        [batchSize, k, n],
        [batchSize, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("computes [4, 64, 32] @ [4, 32, 64] correctly", async () => {
      const batchSize = 4;
      const m = 64,
        k = 32,
        n = 64;

      const aData = makeValues(batchSize * m * k, 0);
      const bData = makeValues(batchSize * k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [batchSize, m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [batchSize, k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [batchSize, m, k],
        [batchSize, k, n],
        [batchSize, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });
  });

  describe("broadcast batched matmul", () => {
    it("broadcasts [8, 32, 32] @ [32, 32] correctly", async () => {
      const batchSize = 8;
      const m = 32,
        k = 32,
        n = 32;

      const aData = makeValues(batchSize * m * k, 0);
      const bData = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [batchSize, m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [batchSize, m, k],
        [k, n],
        [batchSize, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("broadcasts [32, 32] @ [8, 32, 32] correctly", async () => {
      const batchSize = 8;
      const m = 32,
        k = 32,
        n = 32;

      const aData = makeValues(m * k, 0);
      const bData = makeValues(batchSize * k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [batchSize, k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [m, k],
        [batchSize, k, n],
        [batchSize, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("broadcasts [1, 64, 32] @ [4, 32, 64] correctly", async () => {
      const m = 64,
        k = 32,
        n = 64;

      const aData = makeValues(1 * m * k, 0);
      const bData = makeValues(4 * k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [1, m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [4, k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [1, m, k],
        [4, k, n],
        [4, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });
  });

  describe("4D batched matmul", () => {
    it("computes [2, 2, 32, 32] @ [2, 2, 32, 32] correctly", async () => {
      const b1 = 2,
        b2 = 2;
      const m = 32,
        k = 32,
        n = 32;
      const totalBatch = b1 * b2;

      const aData = makeValues(totalBatch * m * k, 0);
      const bData = makeValues(totalBatch * k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [b1, b2, m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [b1, b2, k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [b1, b2, m, k],
        [b1, b2, k, n],
        [b1, b2, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });
  });

  describe("non-tile-aligned batched", () => {
    it("computes [3, 50, 60] @ [3, 60, 70] correctly", async () => {
      const batchSize = 3;
      const m = 50,
        k = 60,
        n = 70;

      const aData = makeValues(batchSize * m * k, 0);
      const bData = makeValues(batchSize * k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aData, [batchSize, m, k]);
      const b = webgpuBackend.ops.tensorFromArray(bData, [batchSize, k, n]);
      const c = webgpuBackend.ops.matmul(a, b);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);
      const expected = referenceBatchedMatmul(
        aData,
        bData,
        [batchSize, m, k],
        [batchSize, k, n],
        [batchSize, m, n],
      );

      expect(arraysClose(result, expected)).toBe(true);
    });
  });
});
