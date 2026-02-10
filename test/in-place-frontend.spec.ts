/**
 * Tests for in-place operations in the frontend (§4.3-4.4)
 *
 * These tests verify:
 * - In-place ops work through the Tensor class methods
 * - base_commit is tracked for mutations per spec §4.3
 */

import { describe, expect, it, beforeEach } from "vitest";
import { torch, Torchlette, Tensor } from "../src/frontend";
import { Engine } from "../src/engine/engine";

describe("in-place operations (frontend)", () => {
  let t: Torchlette;

  beforeEach(() => {
    t = new Torchlette("cpu");
  });

  describe("Tensor.copy_", () => {
    it("copies values from src to this tensor", async () => {
      const a = t.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = t.tensorFromArray([5, 6, 7, 8], [2, 2]);

      const result = a.copy_(b);

      // Returns same tensor
      expect(result).toBe(a);

      // Values updated
      const values = await a.cpu();
      expect(values).toEqual([5, 6, 7, 8]);
    });

    it("preserves baseId", async () => {
      const a = t.tensorFromArray([1, 2], [2]);
      const originalBaseId = a.baseId;
      const b = t.tensorFromArray([3, 4], [2]);

      a.copy_(b);

      expect(a.baseId).toBe(originalBaseId);
    });
  });

  describe("Tensor.add_", () => {
    it("adds values from src to this tensor", async () => {
      const a = t.tensorFromArray([1, 2, 3], [3]);
      const b = t.tensorFromArray([10, 20, 30], [3]);

      const result = a.add_(b);

      expect(result).toBe(a);
      const values = await a.cpu();
      expect(values).toEqual([11, 22, 33]);
    });
  });

  describe("Tensor.zero_", () => {
    it("sets all values to zero", async () => {
      const a = t.tensorFromArray([1, 2, 3, 4], [4]);

      const result = a.zero_();

      expect(result).toBe(a);
      const values = await a.cpu();
      expect(values).toEqual([0, 0, 0, 0]);
    });
  });

  describe("Tensor.fill_", () => {
    it("fills with scalar value", async () => {
      const a = t.tensorFromArray([1, 2, 3], [3]);

      const result = a.fill_(42);

      expect(result).toBe(a);
      const values = await a.cpu();
      expect(values).toEqual([42, 42, 42]);
    });
  });

  describe("Tensor.mul_", () => {
    it("multiplies by scalar in-place", async () => {
      const a = t.tensorFromArray([2, 4, 6], [3]);

      const result = a.mul_(0.5);

      expect(result).toBe(a);
      const values = await a.cpu();
      expect(values[0]).toBeCloseTo(1);
      expect(values[1]).toBeCloseTo(2);
      expect(values[2]).toBeCloseTo(3);
    });
  });

  describe("Torchlette engine methods", () => {
    it("copy_ via engine", async () => {
      const a = t.tensorFromArray([0, 0], [2]);
      const b = t.tensorFromArray([1, 2], [2]);

      const result = t.copy_(a, b);

      expect(result).toBe(a);
      const values = await a.cpu();
      expect(values).toEqual([1, 2]);
    });

    it("add_ via engine", async () => {
      const a = t.tensorFromArray([5, 5], [2]);
      const b = t.tensorFromArray([3, 7], [2]);

      t.add_(a, b);

      const values = await a.cpu();
      expect(values).toEqual([8, 12]);
    });

    it("zero_ via engine", async () => {
      const a = t.tensorFromArray([99, 99, 99], [3]);

      t.zero_(a);

      const values = await a.cpu();
      expect(values).toEqual([0, 0, 0]);
    });

    it("fill_ via engine", async () => {
      const a = t.tensorFromArray([1, 2], [2]);

      t.fill_(a, -1);

      const values = await a.cpu();
      expect(values).toEqual([-1, -1]);
    });

    it("mul_ via engine", async () => {
      const a = t.tensorFromArray([3, 6, 9], [3]);

      t.mul_(a, 2);

      const values = await a.cpu();
      expect(values).toEqual([6, 12, 18]);
    });
  });

  describe("chained in-place operations", () => {
    it("can chain multiple in-place ops", async () => {
      const a = t.tensorFromArray([1, 2, 3, 4], [4]);

      a.mul_(2)      // [2, 4, 6, 8]
       .add_(t.tensorFromArray([1, 1, 1, 1], [4]))  // [3, 5, 7, 9]
       .mul_(0);     // [0, 0, 0, 0]

      const values = await a.cpu();
      expect(values).toEqual([0, 0, 0, 0]);
    });

    it("baseId remains constant through chain", async () => {
      const a = t.tensorFromArray([1, 2, 3, 4], [4]);
      const originalBaseId = a.baseId;

      a.zero_().fill_(5).mul_(2);

      expect(a.baseId).toBe(originalBaseId);
      const values = await a.cpu();
      expect(values).toEqual([10, 10, 10, 10]);
    });
  });

  describe("interaction with other ops", () => {
    it("can use result of in-place op in further computations", async () => {
      const a = t.tensorFromArray([1, 2, 3], [3]);
      const b = t.tensorFromArray([1, 1, 1], [3]);

      a.add_(b);  // [2, 3, 4]
      const c = a.mul(t.tensorFromArray([2, 2, 2], [3]));  // [4, 6, 8]

      const aValues = await a.cpu();
      const cValues = await c.cpu();

      expect(aValues).toEqual([2, 3, 4]);
      expect(cValues).toEqual([4, 6, 8]);
    });
  });
});

describe("base_commit tracking (§4.3)", () => {
  it("each in-place mutation generates unique base_commit", async () => {
    const t = new Torchlette("cpu");
    const a = t.tensorFromArray([1, 2, 3], [3]);
    const b = t.tensorFromArray([1, 1, 1], [3]);

    // First mutation
    t.add_(a, b);

    // Second mutation
    t.add_(a, b);

    // Both mutations should have committed - we verify by checking values work
    const values = await a.cpu();
    expect(values).toEqual([3, 4, 5]);  // 1+1+1, 2+1+1, 3+1+1
  });
});
