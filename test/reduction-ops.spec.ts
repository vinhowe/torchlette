/**
 * Tests for reduction operations: max, softmax
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend";

describe("Reduction ops (CPU)", () => {
  const api = new Torchlette("cpu");

  describe("max", () => {
    it("computes full max reduction", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2], [4]);
      const result = a.max();
      if (typeof result === "number") {
        expect(result).toBe(5);
      } else {
        const data = await result.cpu();
        expect(data[0]).toBe(5);
      }
    });

    it("computes max along dim 0", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2, 4, 6], [2, 3]);
      // Shape [2,3] means: Row 0: [1, 5, 3], Row 1: [2, 4, 6]
      const result = a.max({ dim: 0 });
      if (typeof result === "number") throw new Error("Expected tensor");
      const data = await result.cpu();
      // Max of each column: [max(1,2), max(5,4), max(3,6)]
      expect(data).toEqual([2, 5, 6]);
    });

    it("computes max along dim 1", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2, 4, 6], [2, 3]);
      const result = a.max({ dim: 1 });
      if (typeof result === "number") throw new Error("Expected tensor");
      const data = await result.cpu();
      // Max of each row: [max(1,5,3), max(2,4,6)]
      expect(data).toEqual([5, 6]);
    });

    it("computes max with keepdim", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2, 4, 6], [2, 3]);
      const result = a.max({ dim: 1, keepdim: true });
      if (typeof result === "number") throw new Error("Expected tensor");
      expect(result.shape).toEqual([2, 1]);
      const data = await result.cpu();
      expect(data).toEqual([5, 6]);
    });

    it("handles negative dim", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2, 4, 6], [2, 3]);
      const result = a.max({ dim: -1 });
      if (typeof result === "number") throw new Error("Expected tensor");
      const data = await result.cpu();
      expect(data).toEqual([5, 6]);
    });

    it("handles negative values", async () => {
      const a = api.tensorFromArray([-5, -2, -8, -1], [4]);
      const result = a.max();
      if (typeof result === "number") {
        expect(result).toBe(-1);
      } else {
        const data = await result.cpu();
        expect(data[0]).toBe(-1);
      }
    });
  });

  describe("softmax", () => {
    it("computes softmax along last dimension", async () => {
      const a = api.tensorFromArray([1, 2, 3], [3]);
      const result = a.softmax(-1);
      const data = await result.cpu();

      // Softmax values should sum to 1
      const sum = data.reduce((acc, val) => acc + val, 0);
      expect(sum).toBeCloseTo(1.0, 5);

      // Verify relative ordering: softmax preserves ordering
      expect(data[2]).toBeGreaterThan(data[1]);
      expect(data[1]).toBeGreaterThan(data[0]);
    });

    it("computes 2D softmax along dim 1", async () => {
      const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const result = a.softmax(1);
      expect(result.shape).toEqual([2, 3]);
      const data = await result.cpu();

      // Each row should sum to 1
      const row0Sum = data[0] + data[1] + data[2];
      const row1Sum = data[3] + data[4] + data[5];
      expect(row0Sum).toBeCloseTo(1.0, 5);
      expect(row1Sum).toBeCloseTo(1.0, 5);
    });

    it("is numerically stable with large values", async () => {
      const a = api.tensorFromArray([1000, 1001, 1002], [3]);
      const result = a.softmax(-1);
      const data = await result.cpu();

      // Should not overflow - values should be valid
      expect(data[0]).toBeGreaterThan(0);
      expect(data[1]).toBeGreaterThan(0);
      expect(data[2]).toBeGreaterThan(0);

      // Should sum to 1
      const sum = data.reduce((acc, val) => acc + val, 0);
      expect(sum).toBeCloseTo(1.0, 5);
    });

    it("softmax backward pass", async () => {
      const a = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
      const result = a.softmax(-1);
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // Sum of gradients for softmax loss = sum(softmax) should be 0
      // because softmax outputs sum to 1 and d/dx(sum(softmax(x))) = softmax*(1 - sum(softmax*1)) = 0
      const gradSum = grad?.reduce((acc, val) => acc + val, 0);
      expect(gradSum).toBeCloseTo(0, 5);
    });

    it("handles batched softmax", async () => {
      const a = api.tensorFromArray([
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
      ], [3, 3]);
      const result = a.softmax(-1);
      expect(result.shape).toEqual([3, 3]);

      const data = await result.cpu();
      // Each row should sum to 1
      for (let i = 0; i < 3; i++) {
        const rowSum = data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2];
        expect(rowSum).toBeCloseTo(1.0, 5);
      }
    });
  });
});
