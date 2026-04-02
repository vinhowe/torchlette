/**
 * Edge case tests: empty tensors, cross-device errors, matmul shape validation.
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";

const api = new Torchlette("cpu");

describe("edge cases", () => {
  // =====================================================================
  // Matmul inner dimension validation
  // =====================================================================

  it("matmul throws on inner dimension mismatch", () => {
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = api.tensorFromArray([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    expect(() => api.matmul(a, b)).toThrow(/inner dimension mismatch/);
  });

  it("matmul throws with helpful shape info", () => {
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = api.tensorFromArray([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
    expect(() => api.matmul(a, b)).toThrow(/2,3.*4,2/);
  });

  it("matmul accepts valid shapes", () => {
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = api.tensorFromArray([1, 2, 3, 4, 5, 6], [3, 2]);
    expect(() => api.matmul(a, b)).not.toThrow();
  });

  it("matmul accepts 1D @ 1D (dot product)", () => {
    const a = api.tensorFromArray([1, 2, 3], [3]);
    const b = api.tensorFromArray([4, 5, 6], [3]);
    expect(() => api.matmul(a, b)).not.toThrow();
  });

  it("matmul rejects 1D @ 1D with different lengths", () => {
    const a = api.tensorFromArray([1, 2, 3], [3]);
    const b = api.tensorFromArray([4, 5, 6, 7], [4]);
    expect(() => api.matmul(a, b)).toThrow(/inner dimension mismatch/);
  });

  // =====================================================================
  // Empty tensors
  // =====================================================================

  it("creates empty tensor with shape [0]", () => {
    const t = api.tensorFromArray([], [0]);
    expect(t.shape).toEqual([0]);
  });

  it("sum of empty tensor returns 0", async () => {
    const t = api.tensorFromArray([], [0]);
    const s = api.sum(t);
    const val = await s.item();
    expect(val).toBe(0);
  });

  it("reshape empty tensor to different empty shape", () => {
    const t = api.tensorFromArray([], [0]);
    const r = api.reshape(t, [0, 5]);
    expect(r.shape).toEqual([0, 5]);
  });

  // =====================================================================
  // Scalar tensors
  // =====================================================================

  it("scalar tensor has shape []", () => {
    const t = api.tensorFromArray([42], []);
    expect(t.shape).toEqual([]);
  });

  it("scalar tensor item() returns value", async () => {
    const t = api.tensorFromArray([42], []);
    expect(await t.item()).toBe(42);
  });

  it("scalar + vector broadcasts correctly", async () => {
    const scalar = api.tensorFromArray([10], []);
    const vec = api.tensorFromArray([1, 2, 3], [3]);
    const result = api.add(scalar, vec);
    expect(result.shape).toEqual([3]);
    const data = await result.cpu();
    expect(Array.from(data)).toEqual([11, 12, 13]);
  });

  // =====================================================================
  // Cross-device validation
  // =====================================================================

  it("operations on same device succeed", () => {
    const a = api.tensorFromArray([1, 2], [2]);
    const b = api.tensorFromArray([3, 4], [2]);
    expect(() => api.add(a, b)).not.toThrow();
  });
});
