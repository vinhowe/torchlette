/**
 * Tests for in-place operations (§4.3-4.4)
 *
 * These tests verify:
 * - copy_: copies values from src to dst in-place
 * - add_: adds values from src to dst in-place
 * - zero_: sets all values to zero
 * - fill_: fills with a scalar value
 * - mul_: multiplies by a scalar in-place
 * - base_commit tracking for mutations
 */

import { describe, expect, it, beforeEach } from "vitest";
import { RuntimeEngine } from "../src/runtime/engine";
import {
  tensorFromArray,
  copy_,
  add_,
  zero_,
  fill_,
  mul_,
} from "../src/runtime/engine-facade";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

describe("in-place operations (runtime)", () => {
  let engine: RuntimeEngine;

  beforeEach(() => {
    engine = new RuntimeEngine("cpu");
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
  });

  describe("copy_", () => {
    it("copies values from src to dst", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const src = engine.tensorFromArray([5, 6, 7, 8], [2, 2]);

      const result = engine.copy_(dst, src);

      // Should return same tensor object
      expect(result).toBe(dst);

      // Force and verify values
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([5, 6, 7, 8]);
    });

    it("preserves baseId after copy_", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const originalBaseId = dst.baseId;
      const src = engine.tensorFromArray([5, 6, 7, 8], [2, 2]);

      engine.copy_(dst, src);

      expect(dst.baseId).toBe(originalBaseId);
    });

    it("rejects shape mismatch", () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const src = engine.tensorFromArray([1, 2, 3], [3]);

      expect(() => engine.copy_(dst, src)).toThrow(/shape mismatch/);
    });

    it("works with 1D tensors", async () => {
      const dst = engine.tensorFromArray([1, 2, 3], [3]);
      const src = engine.tensorFromArray([10, 20, 30], [3]);

      engine.copy_(dst, src);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([10, 20, 30]);
    });

    it("works with scalars (0D)", async () => {
      const dst = engine.tensorFromArray([5], []);
      const src = engine.tensorFromArray([42], []);

      engine.copy_(dst, src);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([42]);
    });
  });

  describe("add_", () => {
    it("adds values from src to dst in-place", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const src = engine.tensorFromArray([10, 20, 30, 40], [2, 2]);

      const result = engine.add_(dst, src);

      expect(result).toBe(dst);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([11, 22, 33, 44]);
    });

    it("preserves baseId after add_", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const originalBaseId = dst.baseId;
      const src = engine.tensorFromArray([1, 1, 1, 1], [2, 2]);

      engine.add_(dst, src);

      expect(dst.baseId).toBe(originalBaseId);
    });

    it("rejects shape mismatch", () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const src = engine.tensorFromArray([1, 2], [2]);

      expect(() => engine.add_(dst, src)).toThrow(/shape mismatch/);
    });

    it("can chain multiple add_ operations", async () => {
      const dst = engine.tensorFromArray([0, 0, 0, 0], [4]);
      const ones = engine.tensorFromArray([1, 1, 1, 1], [4]);

      engine.add_(dst, ones);
      engine.add_(dst, ones);
      engine.add_(dst, ones);

      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([3, 3, 3, 3]);
    });
  });

  describe("zero_", () => {
    it("sets all values to zero", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);

      const result = engine.zero_(dst);

      expect(result).toBe(dst);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([0, 0, 0, 0]);
    });

    it("preserves baseId after zero_", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const originalBaseId = dst.baseId;

      engine.zero_(dst);

      expect(dst.baseId).toBe(originalBaseId);
    });

    it("works on 3D tensors", async () => {
      const dst = engine.tensorFromArray(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [2, 2, 2],
      );

      engine.zero_(dst);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
    });
  });

  describe("fill_", () => {
    it("fills tensor with scalar value", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);

      const result = engine.fill_(dst, 7);

      expect(result).toBe(dst);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([7, 7, 7, 7]);
    });

    it("works with negative values", async () => {
      const dst = engine.tensorFromArray([1, 2, 3], [3]);

      engine.fill_(dst, -5);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([-5, -5, -5]);
    });

    it("works with floating point values", async () => {
      const dst = engine.tensorFromArray([1, 2, 3], [3]);

      engine.fill_(dst, 3.14);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values[0]).toBeCloseTo(3.14);
      expect(values[1]).toBeCloseTo(3.14);
      expect(values[2]).toBeCloseTo(3.14);
    });
  });

  describe("mul_", () => {
    it("multiplies tensor by scalar in-place", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);

      const result = engine.mul_(dst, 2);

      expect(result).toBe(dst);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([2, 4, 6, 8]);
    });

    it("works with zero", async () => {
      const dst = engine.tensorFromArray([1, 2, 3, 4], [4]);

      engine.mul_(dst, 0);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([0, 0, 0, 0]);
    });

    it("works with negative scalars", async () => {
      const dst = engine.tensorFromArray([1, 2, 3], [3]);

      engine.mul_(dst, -3);
      await engine.force(dst);
      const values = await engine.cpu(dst);
      expect(values).toEqual([-3, -6, -9]);
    });
  });

  describe("module-level functions", () => {
    it("copy_ function works", async () => {
      const dst = tensorFromArray([1, 2], [2]);
      const src = tensorFromArray([3, 4], [2]);

      copy_(dst, src);
      await new RuntimeEngine().force(dst);
      const values = dst.toArray();
      expect(values).toEqual([3, 4]);
    });

    it("add_ function works", async () => {
      const dst = tensorFromArray([1, 2], [2]);
      const src = tensorFromArray([10, 20], [2]);

      add_(dst, src);
      await new RuntimeEngine().force(dst);
      const values = dst.toArray();
      expect(values).toEqual([11, 22]);
    });

    it("zero_ function works", async () => {
      const dst = tensorFromArray([1, 2, 3], [3]);

      zero_(dst);
      await new RuntimeEngine().force(dst);
      const values = dst.toArray();
      expect(values).toEqual([0, 0, 0]);
    });

    it("fill_ function works", async () => {
      const dst = tensorFromArray([1, 2], [2]);

      fill_(dst, 99);
      await new RuntimeEngine().force(dst);
      const values = dst.toArray();
      expect(values).toEqual([99, 99]);
    });

    it("mul_ function works", async () => {
      const dst = tensorFromArray([2, 4], [2]);

      mul_(dst, 3);
      await new RuntimeEngine().force(dst);
      const values = dst.toArray();
      expect(values).toEqual([6, 12]);
    });
  });
});
