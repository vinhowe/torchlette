/**
 * Frontend dtype casting tests
 */

import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU, isF16Supported } from "../src/backend/webgpu/index";
import { Torchlette } from "../src/frontend/torchlette";
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
      const tensor = api.tensorFromArray([1, 2, 3, 4], [2, 2], {
        requiresGrad: true,
      });
      const casted = tensor.toDtype("i32");

      // Casted tensor should not require grad (autograd detaches)
      expect(casted.requiresGrad).toBe(false);
    });
  });

  // ==========================================================================
  // TensorCreateOptions.dtype — dtype-on-creation for i32/u32 index tensors
  // ==========================================================================

  describe("creation with dtype option", () => {
    describe("CPU backend", () => {
      const api = new Torchlette("cpu");

      it("tensorFromArray with dtype='i32' preserves dtype metadata", async () => {
        const t = api.tensorFromArray([0, 1, 2, -1], [4], { dtype: "i32" });
        expect(t.dtype).toBe("i32");
        expect(await t.cpu()).toEqual([0, 1, 2, -1]);
      });

      it("zeros with dtype='i32' produces i32 tensor", async () => {
        const t = api.zeros([3], { dtype: "i32" });
        expect(t.dtype).toBe("i32");
        expect(await t.cpu()).toEqual([0, 0, 0]);
      });

      it("full with dtype='i32' preserves dtype", async () => {
        const t = api.full([3], -100, { dtype: "i32" });
        expect(t.dtype).toBe("i32");
        expect(await t.cpu()).toEqual([-100, -100, -100]);
      });

      it("tensorFromArray default is f32", async () => {
        const t = api.tensorFromArray([1, 2, 3], [3]);
        expect(t.dtype).toBe("f32");
      });
    });

    describe.skipIf(cpuOnly)("WebGPU backend", () => {
      beforeAll(async () => {
        const success = await initWebGPU();
        if (!success) throw new Error("WebGPU not available");
      });

      it("tensorFromArray with dtype='i32' creates i32 buffer", async () => {
        const api = new Torchlette("webgpu");
        const t = api.tensorFromArray([0, 1, 2, -1], [4], { dtype: "i32" });
        expect(t.dtype).toBe("i32");
        const data = await t.cpu();
        expect(data).toEqual([0, 1, 2, -1]);
      });

      it("tensorFromArray with dtype='i32' handles negative sentinel", async () => {
        const api = new Torchlette("webgpu");
        const t = api.tensorFromArray([-100, 0, -1, 42], [4], {
          dtype: "i32",
        });
        expect(t.dtype).toBe("i32");
        expect(await t.cpu()).toEqual([-100, 0, -1, 42]);
      });

      it("zeros with dtype='i32' allocates i32 buffer", async () => {
        const api = new Torchlette("webgpu");
        const t = api.zeros([5], { dtype: "i32" });
        expect(t.dtype).toBe("i32");
        expect(await t.cpu()).toEqual([0, 0, 0, 0, 0]);
      });

      it("full with dtype='i32' fills with integer value", async () => {
        const api = new Torchlette("webgpu");
        const t = api.full([4], -100, { dtype: "i32" });
        expect(t.dtype).toBe("i32");
        expect(await t.cpu()).toEqual([-100, -100, -100, -100]);
      });

      it("tensorFromArray default is f32", async () => {
        const api = new Torchlette("webgpu");
        const t = api.tensorFromArray([1, 2, 3], [3]);
        expect(t.dtype).toBe("f32");
      });

      it("crossEntropy with i32 targets produces correct loss", async () => {
        const { crossEntropy } = await import("../src/nn/functional");
        const api = new Torchlette("webgpu");
        const logits = api.tensorFromArray(
          [2.0, 1.0, 0.5, 3.0, 0.1, 0.2, 0.3, 0.4],
          [2, 4],
        );
        const targets = api.tensorFromArray([0, 2], [2], { dtype: "i32" });
        const loss = crossEntropy(api, logits, targets, { reduction: "mean" });
        const val = await loss.item();

        // Compare to f32-targets path
        const targetsF32 = api.tensorFromArray([0, 2], [2]);
        const lossF32 = crossEntropy(api, logits, targetsF32, {
          reduction: "mean",
        });
        const valF32 = await lossF32.item();
        expect(Math.abs(val - valF32)).toBeLessThan(1e-5);

        api.markStep();
      });

      it("crossEntropy with i32 targets and ignoreIndex=-1 works", async () => {
        const { crossEntropy } = await import("../src/nn/functional");
        const api = new Torchlette("webgpu");
        // 3 samples, 2 classes; middle sample is ignored.
        const logits = api.tensorFromArray(
          [1.0, 2.0, 3.0, 1.0, 0.5, 0.5],
          [3, 2],
        );
        const targets = api.tensorFromArray([0, -1, 0], [3], { dtype: "i32" });
        const loss = crossEntropy(api, logits, targets, {
          reduction: "none",
          ignoreIndex: -1,
        });
        const perSample = await loss.cpu();
        // Ignored row should be 0.
        expect(perSample[1]).toBe(0);
        // Other rows should be > 0.
        expect(perSample[0]).toBeGreaterThan(0);
        expect(perSample[2]).toBeGreaterThan(0);

        api.markStep();
      });

      it("crossEntropy with i32 targets and ignoreIndex=-100 (default-ish) works", async () => {
        const { crossEntropy } = await import("../src/nn/functional");
        const api = new Torchlette("webgpu");
        const logits = api.tensorFromArray(
          [1.0, 2.0, 3.0, 1.0, 0.5, 0.5, 2.0, 1.0],
          [4, 2],
        );
        const targets = api.tensorFromArray([0, -100, 1, 0], [4], {
          dtype: "i32",
        });
        const loss = crossEntropy(api, logits, targets, {
          reduction: "none",
          ignoreIndex: -100,
        });
        const perSample = await loss.cpu();
        expect(perSample[1]).toBe(0);
        expect(perSample[0]).toBeGreaterThan(0);
        expect(perSample[2]).toBeGreaterThan(0);
        expect(perSample[3]).toBeGreaterThan(0);

        api.markStep();
      });

      it("crossEntropy with i32 targets: gradients are zero for ignored rows", async () => {
        const { crossEntropy } = await import("../src/nn/functional");
        const api = new Torchlette("webgpu");
        const logits = api.tensorFromArray(
          [1.0, 2.0, 3.0, 1.0, 0.5, 0.5],
          [3, 2],
          { requiresGrad: true },
        );
        const targets = api.tensorFromArray([0, -1, 0], [3], { dtype: "i32" });
        const loss = crossEntropy(api, logits, targets, {
          reduction: "sum",
          ignoreIndex: -1,
        });
        await loss.backward();
        const grad = await logits.grad?.cpu();
        // Gradients for ignored row (index 1) should be zero.
        expect(grad?.[2]).toBe(0);
        expect(grad?.[3]).toBe(0);

        api.markStep();
      });
    });
  });
});
