import { describe, it, expect, beforeAll } from "vitest";
import {
  initWebGPU,
  tensorFromArrayWithDtype,
  isF16Supported,
} from "../../src/backend/webgpu/index";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("WebGPU dtype support", () => {
  beforeAll(async () => {
    const success = await initWebGPU();
    if (!success) {
      throw new Error("WebGPU not available");
    }
  });

  describe("i32 dtype", () => {
    it("can create i32 tensor", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
      expect(tensor.shape).toEqual([2, 2]);
      expect(tensor.dtype).toBe("i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const data = await webgpuBackend.ops.read(tensor);
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("can add i32 tensors", async () => {
      const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
      const b = tensorFromArrayWithDtype([10, 20, 30, 40], [2, 2], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.add(a, b);
      expect(result.dtype).toBe("i32");

      const data = await webgpuBackend.ops.read(result);
      expect(data).toEqual([11, 22, 33, 44]);
    });

    it("can subtract i32 tensors", async () => {
      const a = tensorFromArrayWithDtype([10, 20, 30, 40], [2, 2], "i32");
      const b = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.sub(a, b);
      expect(result.dtype).toBe("i32");

      const data = await webgpuBackend.ops.read(result);
      expect(data).toEqual([9, 18, 27, 36]);
    });

    it("can multiply i32 tensors", async () => {
      const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
      const b = tensorFromArrayWithDtype([2, 3, 4, 5], [2, 2], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.mul(a, b);
      expect(result.dtype).toBe("i32");

      const data = await webgpuBackend.ops.read(result);
      expect(data).toEqual([2, 6, 12, 20]);
    });

    it("can divide i32 tensors", async () => {
      const a = tensorFromArrayWithDtype([10, 20, 30, 40], [2, 2], "i32");
      const b = tensorFromArrayWithDtype([2, 4, 5, 8], [2, 2], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.div(a, b);
      expect(result.dtype).toBe("i32");

      const data = await webgpuBackend.ops.read(result);
      // Integer division
      expect(data).toEqual([5, 5, 6, 5]);
    });

    it("can broadcast i32 tensors", async () => {
      const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
      const b = tensorFromArrayWithDtype([10], [1], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.add(a, b);
      expect(result.dtype).toBe("i32");
      expect(result.shape).toEqual([2, 2]);

      const data = await webgpuBackend.ops.read(result);
      expect(data).toEqual([11, 12, 13, 14]);
    });

    it("reshape preserves i32 dtype", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4, 5, 6], [2, 3], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const reshaped = webgpuBackend.ops.reshape(tensor, [3, 2]);
      expect(reshaped.dtype).toBe("i32");
      expect(reshaped.shape).toEqual([3, 2]);

      const data = await webgpuBackend.ops.read(reshaped);
      expect(data).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("transpose preserves i32 dtype", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4, 5, 6], [2, 3], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const transposed = webgpuBackend.ops.transpose(tensor, { dim0: 0, dim1: 1 });
      expect(transposed.dtype).toBe("i32");
      expect(transposed.shape).toEqual([3, 2]);

      const data = await webgpuBackend.ops.read(transposed);
      // Original: [[1,2,3], [4,5,6]] -> Transposed: [[1,4], [2,5], [3,6]]
      expect(data).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it("expand preserves i32 dtype", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3], [1, 3], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const expanded = webgpuBackend.ops.expand(tensor, [2, 3]);
      expect(expanded.dtype).toBe("i32");
      expect(expanded.shape).toEqual([2, 3]);

      // Need to make contiguous to read expanded view
      const contiguous = webgpuBackend.ops.contiguous(expanded);
      const data = await webgpuBackend.ops.read(contiguous);
      expect(data).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it("throws on dtype mismatch", async () => {
      const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
      const b = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0], [2, 2], "f32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      expect(() => webgpuBackend.ops.add(a, b)).toThrow(/mismatched dtypes/);
    });
  });

  describe("u32 dtype", () => {
    it("can create u32 tensor", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "u32");
      expect(tensor.shape).toEqual([2, 2]);
      expect(tensor.dtype).toBe("u32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const data = await webgpuBackend.ops.read(tensor);
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("can add u32 tensors", async () => {
      const a = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "u32");
      const b = tensorFromArrayWithDtype([10, 20, 30, 40], [2, 2], "u32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.add(a, b);
      expect(result.dtype).toBe("u32");

      const data = await webgpuBackend.ops.read(result);
      expect(data).toEqual([11, 22, 33, 44]);
    });
  });

  describe("f32 dtype (default)", () => {
    it("creates f32 tensor by default", async () => {
      const tensor = tensorFromArrayWithDtype([1.5, 2.5, 3.5, 4.5], [2, 2], "f32");
      expect(tensor.dtype).toBe("f32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const data = await webgpuBackend.ops.read(tensor);
      expect(data[0]).toBeCloseTo(1.5);
      expect(data[1]).toBeCloseTo(2.5);
      expect(data[2]).toBeCloseTo(3.5);
      expect(data[3]).toBeCloseTo(4.5);
    });

    it("standard tensorFromArray creates f32", async () => {
      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const tensor = webgpuBackend.ops.tensorFromArray([1.5, 2.5, 3.5], [3]);
      expect(tensor.dtype).toBe("f32");

      const data = await webgpuBackend.ops.read(tensor);
      expect(data[0]).toBeCloseTo(1.5);
    });
  });

  describe("f16 dtype", () => {
    it("can create f16 tensor when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const tensor = tensorFromArrayWithDtype([1.5, 2.5, 3.5, 4.5], [2, 2], "f16");
      expect(tensor.shape).toEqual([2, 2]);
      expect(tensor.dtype).toBe("f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const data = await webgpuBackend.ops.read(tensor);
      // f16 has less precision, so use larger tolerance
      expect(data[0]).toBeCloseTo(1.5, 2);
      expect(data[1]).toBeCloseTo(2.5, 2);
      expect(data[2]).toBeCloseTo(3.5, 2);
      expect(data[3]).toBeCloseTo(4.5, 2);
    });

    it("can add f16 tensors when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const a = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0], [2, 2], "f16");
      const b = tensorFromArrayWithDtype([0.5, 1.5, 2.5, 3.5], [2, 2], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.add(a, b);
      expect(result.dtype).toBe("f16");

      const data = await webgpuBackend.ops.read(result);
      expect(data[0]).toBeCloseTo(1.5, 2);
      expect(data[1]).toBeCloseTo(3.5, 2);
      expect(data[2]).toBeCloseTo(5.5, 2);
      expect(data[3]).toBeCloseTo(7.5, 2);
    });

    it("can subtract f16 tensors when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const a = tensorFromArrayWithDtype([10.0, 20.0, 30.0, 40.0], [2, 2], "f16");
      const b = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0], [2, 2], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.sub(a, b);
      expect(result.dtype).toBe("f16");

      const data = await webgpuBackend.ops.read(result);
      expect(data[0]).toBeCloseTo(9.0, 2);
      expect(data[1]).toBeCloseTo(18.0, 2);
      expect(data[2]).toBeCloseTo(27.0, 2);
      expect(data[3]).toBeCloseTo(36.0, 2);
    });

    it("can multiply f16 tensors when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const a = tensorFromArrayWithDtype([2.0, 3.0, 4.0, 5.0], [2, 2], "f16");
      const b = tensorFromArrayWithDtype([1.5, 2.0, 2.5, 3.0], [2, 2], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.mul(a, b);
      expect(result.dtype).toBe("f16");

      const data = await webgpuBackend.ops.read(result);
      expect(data[0]).toBeCloseTo(3.0, 2);
      expect(data[1]).toBeCloseTo(6.0, 2);
      expect(data[2]).toBeCloseTo(10.0, 2);
      expect(data[3]).toBeCloseTo(15.0, 2);
    });

    it("can divide f16 tensors when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const a = tensorFromArrayWithDtype([10.0, 20.0, 30.0, 40.0], [2, 2], "f16");
      const b = tensorFromArrayWithDtype([2.0, 4.0, 5.0, 8.0], [2, 2], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.div(a, b);
      expect(result.dtype).toBe("f16");

      const data = await webgpuBackend.ops.read(result);
      expect(data[0]).toBeCloseTo(5.0, 2);
      expect(data[1]).toBeCloseTo(5.0, 2);
      expect(data[2]).toBeCloseTo(6.0, 2);
      expect(data[3]).toBeCloseTo(5.0, 2);
    });

    it("can broadcast f16 tensors when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const a = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0], [2, 2], "f16");
      const b = tensorFromArrayWithDtype([10.0], [1], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const result = webgpuBackend.ops.add(a, b);
      expect(result.dtype).toBe("f16");
      expect(result.shape).toEqual([2, 2]);

      const data = await webgpuBackend.ops.read(result);
      expect(data[0]).toBeCloseTo(11.0, 2);
      expect(data[1]).toBeCloseTo(12.0, 2);
      expect(data[2]).toBeCloseTo(13.0, 2);
      expect(data[3]).toBeCloseTo(14.0, 2);
    });

    it("reshape preserves f16 dtype when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const tensor = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const reshaped = webgpuBackend.ops.reshape(tensor, [3, 2]);
      expect(reshaped.dtype).toBe("f16");
      expect(reshaped.shape).toEqual([3, 2]);

      const data = await webgpuBackend.ops.read(reshaped);
      expect(data[0]).toBeCloseTo(1.0, 2);
      expect(data[5]).toBeCloseTo(6.0, 2);
    });

    it("transpose preserves f16 dtype when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const tensor = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const transposed = webgpuBackend.ops.transpose(tensor, { dim0: 0, dim1: 1 });
      expect(transposed.dtype).toBe("f16");
      expect(transposed.shape).toEqual([3, 2]);

      const data = await webgpuBackend.ops.read(transposed);
      // Original: [[1,2,3], [4,5,6]] -> Transposed: [[1,4], [2,5], [3,6]]
      expect(data[0]).toBeCloseTo(1.0, 2);
      expect(data[1]).toBeCloseTo(4.0, 2);
    });

    it("throws when f16 not supported", async () => {
      if (isF16Supported()) {
        console.log("Skipping f16 not-supported test: f16 is actually supported");
        return;
      }

      expect(() => tensorFromArrayWithDtype([1.0, 2.0], [2], "f16")).toThrow(/shader-f16/);
    });
  });

  describe("dtype casting", () => {
    it("casts f32 to i32", async () => {
      const tensor = tensorFromArrayWithDtype([1.5, 2.7, 3.1, 4.9], [2, 2], "f32");
      expect(tensor.dtype).toBe("f32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "i32");
      expect(casted.dtype).toBe("i32");
      expect(casted.shape).toEqual([2, 2]);

      const data = await webgpuBackend.ops.read(casted);
      // f32 to i32 truncates
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("casts i32 to f32", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");
      expect(tensor.dtype).toBe("i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "f32");
      expect(casted.dtype).toBe("f32");

      const data = await webgpuBackend.ops.read(casted);
      expect(data[0]).toBeCloseTo(1.0);
      expect(data[1]).toBeCloseTo(2.0);
      expect(data[2]).toBeCloseTo(3.0);
      expect(data[3]).toBeCloseTo(4.0);
    });

    it("casts f32 to u32", async () => {
      const tensor = tensorFromArrayWithDtype([1.5, 2.7, 3.1, 4.9], [2, 2], "f32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "u32");
      expect(casted.dtype).toBe("u32");

      const data = await webgpuBackend.ops.read(casted);
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("casts i32 to u32", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "u32");
      expect(casted.dtype).toBe("u32");

      const data = await webgpuBackend.ops.read(casted);
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("cast to same dtype returns same tensor", async () => {
      const tensor = tensorFromArrayWithDtype([1, 2, 3, 4], [2, 2], "i32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "i32");
      // Should return same tensor (no-op)
      expect(casted).toBe(tensor);
    });

    it("casts f32 to f16 when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 cast test: shader-f16 not supported");
        return;
      }

      const tensor = tensorFromArrayWithDtype([1.5, 2.5, 3.5, 4.5], [2, 2], "f32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "f16");
      expect(casted.dtype).toBe("f16");

      const data = await webgpuBackend.ops.read(casted);
      expect(data[0]).toBeCloseTo(1.5, 2);
      expect(data[1]).toBeCloseTo(2.5, 2);
      expect(data[2]).toBeCloseTo(3.5, 2);
      expect(data[3]).toBeCloseTo(4.5, 2);
    });

    it("casts f16 to f32 when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 cast test: shader-f16 not supported");
        return;
      }

      const tensor = tensorFromArrayWithDtype([1.5, 2.5, 3.5, 4.5], [2, 2], "f16");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const casted = webgpuBackend.ops.cast!(tensor, "f32");
      expect(casted.dtype).toBe("f32");

      const data = await webgpuBackend.ops.read(casted);
      expect(data[0]).toBeCloseTo(1.5, 2);
      expect(data[1]).toBeCloseTo(2.5, 2);
    });

    it("casts non-contiguous tensor (transposed)", async () => {
      const tensor = tensorFromArrayWithDtype([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], "f32");

      const { webgpuBackend } = await import("../../src/backend/webgpu/index");
      const transposed = webgpuBackend.ops.transpose(tensor, { dim0: 0, dim1: 1 });
      expect(transposed.isContiguous).toBe(false);

      const casted = webgpuBackend.ops.cast!(transposed, "i32");
      expect(casted.dtype).toBe("i32");
      expect(casted.shape).toEqual([3, 2]);
      expect(casted.isContiguous).toBe(true); // Cast materializes to contiguous

      const data = await webgpuBackend.ops.read(casted);
      // Transposed: [[1,4], [2,5], [3,6]]
      expect(data).toEqual([1, 4, 2, 5, 3, 6]);
    });
  });
});
