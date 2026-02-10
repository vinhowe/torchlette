/**
 * Tests for Sprint 3 operations:
 * - Comparison ops (gt, lt, ge, le, eq, ne)
 * - Argmax/argmin
 * - Layernorm
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend";

describe("Sprint 3 ops (CPU)", () => {
  const api = new Torchlette("cpu");

  describe("Comparison ops", () => {
    it("gt: compares element-wise greater than", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2], [4]);
      const b = api.tensorFromArray([2, 3, 3, 4], [4]);
      const result = a.gt(b);
      const data = await result.cpu();
      // [1>2, 5>3, 3>3, 2>4] = [0, 1, 0, 0]
      expect(data).toEqual([0, 1, 0, 0]);
    });

    it("lt: compares element-wise less than", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2], [4]);
      const b = api.tensorFromArray([2, 3, 3, 4], [4]);
      const result = a.lt(b);
      const data = await result.cpu();
      // [1<2, 5<3, 3<3, 2<4] = [1, 0, 0, 1]
      expect(data).toEqual([1, 0, 0, 1]);
    });

    it("ge: compares element-wise greater than or equal", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2], [4]);
      const b = api.tensorFromArray([2, 3, 3, 4], [4]);
      const result = a.ge(b);
      const data = await result.cpu();
      // [1>=2, 5>=3, 3>=3, 2>=4] = [0, 1, 1, 0]
      expect(data).toEqual([0, 1, 1, 0]);
    });

    it("le: compares element-wise less than or equal", async () => {
      const a = api.tensorFromArray([1, 5, 3, 2], [4]);
      const b = api.tensorFromArray([2, 3, 3, 4], [4]);
      const result = a.le(b);
      const data = await result.cpu();
      // [1<=2, 5<=3, 3<=3, 2<=4] = [1, 0, 1, 1]
      expect(data).toEqual([1, 0, 1, 1]);
    });

    it("eq: compares element-wise equality", async () => {
      const a = api.tensorFromArray([1, 3, 3, 4], [4]);
      const b = api.tensorFromArray([2, 3, 4, 4], [4]);
      const result = a.eq(b);
      const data = await result.cpu();
      // [1==2, 3==3, 3==4, 4==4] = [0, 1, 0, 1]
      expect(data).toEqual([0, 1, 0, 1]);
    });

    it("ne: compares element-wise inequality", async () => {
      const a = api.tensorFromArray([1, 3, 3, 4], [4]);
      const b = api.tensorFromArray([2, 3, 4, 4], [4]);
      const result = a.ne(b);
      const data = await result.cpu();
      // [1!=2, 3!=3, 3!=4, 4!=4] = [1, 0, 1, 0]
      expect(data).toEqual([1, 0, 1, 0]);
    });

    it("handles broadcasting", async () => {
      const a = api.tensorFromArray([1, 5, 3], [3]);
      const b = api.tensorFromArray([3], [1]);
      const result = a.gt(b);
      const data = await result.cpu();
      // [1>3, 5>3, 3>3] = [0, 1, 0]
      expect(data).toEqual([0, 1, 0]);
    });

    it("handles 2D tensors", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4], [2, 2]);
      const b = api.tensorFromArray([3, 3, 3, 3], [2, 2]);
      const result = a.gt(b);
      expect(result.shape).toEqual([2, 2]);
      const data = await result.cpu();
      // [[1>3, 5>3], [2>3, 4>3]] = [[0, 1], [0, 1]]
      expect(data).toEqual([0, 1, 0, 1]);
    });
  });

  describe("argmax", () => {
    it("finds index of max along dim 0", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      // Row 0: [1, 5, 2], Row 1: [4, 3, 6]
      const result = a.argmax({ dim: 0 });
      expect(result.shape).toEqual([3]);
      const data = await result.cpu();
      // argmax along columns: [4>1->1, 5>3->0, 6>2->1]
      expect(data).toEqual([1, 0, 1]);
    });

    it("finds index of max along dim 1", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      const result = a.argmax({ dim: 1 });
      expect(result.shape).toEqual([2]);
      const data = await result.cpu();
      // Row 0: max at index 1 (value 5), Row 1: max at index 2 (value 6)
      expect(data).toEqual([1, 2]);
    });

    it("handles keepdim", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      const result = a.argmax({ dim: 1, keepdim: true });
      expect(result.shape).toEqual([2, 1]);
      const data = await result.cpu();
      expect(data).toEqual([1, 2]);
    });

    it("handles negative dim", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      const result = a.argmax({ dim: -1 });
      expect(result.shape).toEqual([2]);
      const data = await result.cpu();
      expect(data).toEqual([1, 2]);
    });

    it("handles 1D tensor", async () => {
      const a = api.tensorFromArray([3, 1, 4, 1, 5, 9, 2, 6], [8]);
      const result = a.argmax({ dim: 0 });
      expect(result.shape).toEqual([]);
      const data = await result.cpu();
      expect(data[0]).toBe(5); // Index of 9
    });
  });

  describe("argmin", () => {
    it("finds index of min along dim 0", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      const result = a.argmin({ dim: 0 });
      expect(result.shape).toEqual([3]);
      const data = await result.cpu();
      // argmin along columns: [1<4->0, 3<5->1, 2<6->0]
      expect(data).toEqual([0, 1, 0]);
    });

    it("finds index of min along dim 1", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      const result = a.argmin({ dim: 1 });
      expect(result.shape).toEqual([2]);
      const data = await result.cpu();
      // Row 0: min at index 0 (value 1), Row 1: min at index 1 (value 3)
      expect(data).toEqual([0, 1]);
    });

    it("handles keepdim", async () => {
      const a = api.tensorFromArray([1, 5, 2, 4, 3, 6], [2, 3]);
      const result = a.argmin({ dim: 1, keepdim: true });
      expect(result.shape).toEqual([2, 1]);
      const data = await result.cpu();
      expect(data).toEqual([0, 1]);
    });
  });

  describe("layernorm", () => {
    it("normalizes along last dimension", async () => {
      const x = api.tensorFromArray([1, 2, 3, 4], [4]);
      const weight = api.tensorFromArray([1, 1, 1, 1], [4]);
      const bias = api.tensorFromArray([0, 0, 0, 0], [4]);
      const result = x.layernorm(weight, bias);
      expect(result.shape).toEqual([4]);
      const data = await result.cpu();

      // For unit weight and zero bias, output should have mean ~0 and std ~1
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(mean).toBeCloseTo(0, 4);

      const variance = data.reduce((a, b) => a + b * b, 0) / data.length;
      expect(variance).toBeCloseTo(1, 4);
    });

    it("applies affine transformation", async () => {
      const x = api.tensorFromArray([1, 2, 3, 4], [4]);
      const weight = api.tensorFromArray([2, 2, 2, 2], [4]);
      const bias = api.tensorFromArray([1, 1, 1, 1], [4]);
      const result = x.layernorm(weight, bias);
      expect(result.shape).toEqual([4]);
      const data = await result.cpu();

      // Mean should be 1 (the bias), since normalized mean is 0
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(mean).toBeCloseTo(1, 4);
    });

    it("handles 2D input (batch)", async () => {
      const x = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const weight = api.tensorFromArray([1, 1, 1], [3]);
      const bias = api.tensorFromArray([0, 0, 0], [3]);
      const result = x.layernorm(weight, bias);
      expect(result.shape).toEqual([2, 3]);
      const data = await result.cpu();

      // Each row should have mean ~0
      const row0Mean = (data[0] + data[1] + data[2]) / 3;
      const row1Mean = (data[3] + data[4] + data[5]) / 3;
      expect(row0Mean).toBeCloseTo(0, 4);
      expect(row1Mean).toBeCloseTo(0, 4);
    });

    it("handles 3D input (transformer style)", async () => {
      // [batch, seq, hidden]
      const x = api.tensorFromArray([
        1, 2, 3, 4,  // seq 0
        5, 6, 7, 8,  // seq 1
      ], [1, 2, 4]);
      const weight = api.tensorFromArray([1, 1, 1, 1], [4]);
      const bias = api.tensorFromArray([0, 0, 0, 0], [4]);
      const result = x.layernorm(weight, bias);
      expect(result.shape).toEqual([1, 2, 4]);
      const data = await result.cpu();

      // Each of the 2 sequences should have mean ~0 (normalized independently)
      const seq0Mean = (data[0] + data[1] + data[2] + data[3]) / 4;
      const seq1Mean = (data[4] + data[5] + data[6] + data[7]) / 4;
      expect(seq0Mean).toBeCloseTo(0, 4);
      expect(seq1Mean).toBeCloseTo(0, 4);
    });

    it("backward pass computes gradients", async () => {
      const x = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      const weight = api.tensorFromArray([1, 1, 1, 1], [4], { requiresGrad: true });
      const bias = api.tensorFromArray([0, 0, 0, 0], [4], { requiresGrad: true });
      const result = x.layernorm(weight, bias);
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      // Check that gradients exist
      expect(x.grad).not.toBeNull();
      expect(weight.grad).not.toBeNull();
      expect(bias.grad).not.toBeNull();

      // Bias gradient should equal grad output (sum of ones = shape of output)
      const biasGrad = await bias.grad?.cpu();
      const sumBiasGrad = biasGrad?.reduce((a, b) => a + b, 0);
      expect(sumBiasGrad).toBeCloseTo(4, 4); // 4 elements, each gets grad 1
    });
  });

  describe("relu backward with comparison ops", () => {
    it("computes correct gradient", async () => {
      const a = api.tensorFromArray([-1, 0, 1, 2], [4], { requiresGrad: true });
      const result = a.relu();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // grad = [0, 0, 1, 1] (relu grad is 0 for x<=0, 1 for x>0)
      expect(grad).toEqual([0, 0, 1, 1]);
    });
  });

  describe("sqrt backward without toArray", () => {
    it("computes correct gradient", async () => {
      const a = api.tensorFromArray([1, 4, 9, 16], [4], { requiresGrad: true });
      const result = a.sqrt();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx sqrt(x) = 0.5 / sqrt(x)
      // For x=[1, 4, 9, 16], sqrt(x)=[1, 2, 3, 4]
      // grad = [0.5/1, 0.5/2, 0.5/3, 0.5/4] = [0.5, 0.25, 0.1667, 0.125]
      expect(grad?.[0]).toBeCloseTo(0.5, 4);
      expect(grad?.[1]).toBeCloseTo(0.25, 4);
      expect(grad?.[2]).toBeCloseTo(1/6, 4);
      expect(grad?.[3]).toBeCloseTo(0.125, 4);
    });
  });
});
