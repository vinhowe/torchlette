import { describe, expect, it } from "vitest";

import {
  computeContiguousStrides,
  checkContiguous,
  transposeStrides,
  transposeShape,
  type ViewMeta,
} from "../src/backend/types";

import { sizeOf } from "../src/core/shape";

import {
  Tensor,
  tensorFromArray,
  transpose,
  contiguous,
} from "../src/backend/cpu/numeric";

describe("ViewMeta utilities (§4.2-4.4)", () => {
  describe("computeContiguousStrides", () => {
    it("computes strides for 1D tensor", () => {
      expect(computeContiguousStrides([10])).toEqual([1]);
    });

    it("computes strides for 2D tensor", () => {
      // For [3, 4]: strides should be [4, 1] (row-major)
      expect(computeContiguousStrides([3, 4])).toEqual([4, 1]);
    });

    it("computes strides for 3D tensor", () => {
      // For [2, 3, 4]: strides should be [12, 4, 1]
      expect(computeContiguousStrides([2, 3, 4])).toEqual([12, 4, 1]);
    });

    it("handles scalar (empty shape)", () => {
      expect(computeContiguousStrides([])).toEqual([]);
    });

    it("handles size-1 dimensions", () => {
      expect(computeContiguousStrides([1, 4])).toEqual([4, 1]);
      expect(computeContiguousStrides([3, 1])).toEqual([1, 1]);
      expect(computeContiguousStrides([1, 1, 4])).toEqual([4, 4, 1]);
    });
  });

  describe("checkContiguous", () => {
    it("returns true for contiguous strides", () => {
      expect(checkContiguous([3, 4], [4, 1])).toBe(true);
      expect(checkContiguous([2, 3, 4], [12, 4, 1])).toBe(true);
    });

    it("returns false for non-contiguous strides", () => {
      // Transposed [3, 4] has strides [1, 3] instead of [4, 1]
      expect(checkContiguous([4, 3], [1, 4])).toBe(false);
    });

    it("ignores size-1 dimensions", () => {
      // Size-1 dims don't matter for contiguity
      expect(checkContiguous([1, 4], [100, 1])).toBe(true); // stride[0] doesn't matter
      expect(checkContiguous([3, 1], [1, 999])).toBe(true); // stride[1] doesn't matter
    });

    it("returns false for mismatched lengths", () => {
      expect(checkContiguous([3, 4], [4])).toBe(false);
    });
  });

  describe("sizeOf", () => {
    it("computes size of 1D tensor", () => {
      expect(sizeOf([10])).toBe(10);
    });

    it("computes size of 2D tensor", () => {
      expect(sizeOf([3, 4])).toBe(12);
    });

    it("computes size of 3D tensor", () => {
      expect(sizeOf([2, 3, 4])).toBe(24);
    });

    it("returns 1 for scalar", () => {
      expect(sizeOf([])).toBe(1);
    });
  });

  describe("transposeStrides", () => {
    it("swaps strides for 2D transpose", () => {
      // [3, 4] with strides [4, 1] -> transpose(0, 1) -> strides [1, 4]
      expect(transposeStrides([4, 1], 0, 1)).toEqual([1, 4]);
    });

    it("swaps strides for 3D transpose", () => {
      // [2, 3, 4] with strides [12, 4, 1] -> transpose(0, 2) -> strides [1, 4, 12]
      expect(transposeStrides([12, 4, 1], 0, 2)).toEqual([1, 4, 12]);
    });

    it("handles no-op transpose", () => {
      expect(transposeStrides([4, 1], 0, 0)).toEqual([4, 1]);
    });
  });

  describe("transposeShape", () => {
    it("swaps shape dims for 2D transpose", () => {
      expect(transposeShape([3, 4], 0, 1)).toEqual([4, 3]);
    });

    it("swaps shape dims for 3D transpose", () => {
      expect(transposeShape([2, 3, 4], 0, 2)).toEqual([4, 3, 2]);
    });
  });

  describe("ViewMeta type usage", () => {
    it("can create a contiguous view", () => {
      const shape = [3, 4];
      const meta: ViewMeta = {
        baseId: 1,
        offset: 0,
        shape,
        strides: computeContiguousStrides(shape),
        isContiguous: true,
      };

      expect(meta.strides).toEqual([4, 1]);
      expect(checkContiguous(meta.shape, meta.strides)).toBe(true);
    });

    it("can create a transposed view", () => {
      const originalShape = [3, 4];
      const originalStrides = computeContiguousStrides(originalShape);

      // Transpose dims 0 and 1
      const transposedShape = transposeShape(originalShape, 0, 1);
      const transposedStrides = transposeStrides(originalStrides, 0, 1);

      const meta: ViewMeta = {
        baseId: 1,
        offset: 0,
        shape: transposedShape,
        strides: transposedStrides,
        isContiguous: checkContiguous(transposedShape, transposedStrides),
      };

      expect(meta.shape).toEqual([4, 3]);
      expect(meta.strides).toEqual([1, 4]);
      expect(meta.isContiguous).toBe(false); // Transposed is not contiguous
    });
  });
});

describe("CPU Backend Stride Tracking", () => {
  describe("transpose", () => {
    it("creates non-contiguous view with swapped strides", () => {
      // Create a 3x4 tensor
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = tensorFromArray(data, [3, 4]);

      expect(a.strides).toEqual([4, 1]);
      expect(a.isContiguous()).toBe(true);

      // Transpose to 4x3
      const t = transpose(a, { dim0: 0, dim1: 1 });

      expect(t.shape).toEqual([4, 3]);
      expect(t.strides).toEqual([1, 4]); // Strides swapped
      expect(t.isContiguous()).toBe(false); // Not contiguous
    });

    it("shares underlying data with original", () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = tensorFromArray(data, [3, 4]);
      const t = transpose(a, { dim0: 0, dim1: 1 });

      // Same underlying data
      expect(t.data).toBe(a.data);
    });

    it("reads elements correctly via strides", () => {
      // Original: [[0,1,2,3], [4,5,6,7], [8,9,10,11]]
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = tensorFromArray(data, [3, 4]);
      const t = transpose(a, { dim0: 0, dim1: 1 });

      // Transposed: [[0,4,8], [1,5,9], [2,6,10], [3,7,11]]
      const arr = t.toArray();
      expect(arr).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    });
  });

  describe("contiguous", () => {
    it("returns same tensor if already contiguous", () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = tensorFromArray(data, [3, 4]);

      const c = contiguous(a);

      expect(c).toBe(a); // Same object
    });

    it("materializes non-contiguous tensor to new storage", () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = tensorFromArray(data, [3, 4]);
      const t = transpose(a, { dim0: 0, dim1: 1 });

      expect(t.isContiguous()).toBe(false);

      const c = contiguous(t);

      expect(c.isContiguous()).toBe(true);
      expect(c.shape).toEqual([4, 3]);
      expect(c.strides).toEqual([3, 1]); // Contiguous strides for 4x3
      expect(c.data).not.toBe(t.data); // Different underlying data
    });

    it("preserves element values when materializing", () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = tensorFromArray(data, [3, 4]);
      const t = transpose(a, { dim0: 0, dim1: 1 });
      const c = contiguous(t);

      // Should have same logical values
      expect(c.toArray()).toEqual(t.toArray());
    });
  });

  describe("isContiguous method", () => {
    it("returns true for freshly created tensor", () => {
      const a = tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      expect(a.isContiguous()).toBe(true);
    });

    it("returns false for transposed tensor", () => {
      const a = tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const t = transpose(a, { dim0: 0, dim1: 1 });
      expect(t.isContiguous()).toBe(false);
    });

    it("returns true for contiguous() result", () => {
      const a = tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const t = transpose(a, { dim0: 0, dim1: 1 });
      const c = contiguous(t);
      expect(c.isContiguous()).toBe(true);
    });
  });
});
