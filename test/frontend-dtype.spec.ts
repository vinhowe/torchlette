/**
 * Frontend dtype casting tests
 */

import { describe, expect, it, beforeAll } from "vitest";
import { Torchlette } from "../src/frontend";
import { initWebGPU, isF16Supported } from "../src/backend/webgpu/index";
import { cpuOnly } from "./helpers/webgpu";

describe("Frontend dtype casting", () => {
  describe("CPU backend", () => {
    const api = new Torchlette("cpu");

    it("toDtype method exists", () => {
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(typeof tensor.toDtype).toBe("function");
    });

    it("half() convenience method exists", () => {
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(typeof tensor.half).toBe("function");
    });

    it("float() convenience method exists", () => {
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(typeof tensor.float).toBe("function");
    });

    it("int() convenience method exists", () => {
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(typeof tensor.int).toBe("function");
    });

    it("toDtype returns tensor with same shape", async () => {
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const casted = tensor.toDtype("i32");
      expect(casted.shape).toEqual([2, 2]);

      // CPU backend doesn't actually change dtype, but the API should work
      const data = await casted.cpu();
      expect(data).toEqual([1, 2, 3, 4]);
    });
  });

  describe.skipIf(cpuOnly)("WebGPU backend", () => {
    beforeAll(async () => {
      const success = await initWebGPU();
      if (!success) {
        throw new Error("WebGPU not available");
      }
    });

    it("toDtype casts f32 to i32", async () => {
      const api = new Torchlette("webgpu");
      const tensor = api.tensorFromArray([1.5, 2.7, 3.1, 4.9], [2, 2]);
      const casted = tensor.toDtype("i32");

      expect(casted.shape).toEqual([2, 2]);

      const data = await casted.cpu();
      // f32 to i32 truncates
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("int() convenience method works", async () => {
      const api = new Torchlette("webgpu");
      const tensor = api.tensorFromArray([1.5, 2.7, 3.1, 4.9], [2, 2]);
      const casted = tensor.int();

      const data = await casted.cpu();
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("float() returns f32 tensor", async () => {
      const api = new Torchlette("webgpu");
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const casted = tensor.float();

      const data = await casted.cpu();
      expect(data[0]).toBeCloseTo(1.0);
      expect(data[1]).toBeCloseTo(2.0);
    });

    it("half() casts to f16 when supported", async () => {
      if (!isF16Supported()) {
        console.log("Skipping f16 test: shader-f16 not supported");
        return;
      }

      const api = new Torchlette("webgpu");
      const tensor = api.tensorFromArray([1.5, 2.5, 3.5, 4.5], [2, 2]);
      const casted = tensor.half();

      const data = await casted.cpu();
      expect(data[0]).toBeCloseTo(1.5, 2);
      expect(data[1]).toBeCloseTo(2.5, 2);
    });

    it("dtype cast does not preserve autograd", async () => {
      const api = new Torchlette("webgpu");
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
      const casted = tensor.toDtype("i32");

      // Casted tensor should not require grad (autograd detaches)
      expect(casted.requiresGrad).toBe(false);
    });
  });
});
