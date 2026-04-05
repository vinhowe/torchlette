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

      it("gather with i32 indices returns correct values", async () => {
        const api = new Torchlette("webgpu");
        // data [3,4]: row 0 = [1,2,3,4], row 1 = [5,6,7,8], row 2 = [9,10,11,12]
        const data = api.tensorFromArray(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [3, 4],
        );
        // Gather along dim=0 — indices shape must match data rank.
        const idx = api.tensorFromArray([2, 0, 1, 2, 0, 1, 2, 0], [2, 4], {
          dtype: "i32",
        });
        const gathered = api.gather(data, idx, { dim: 0 });
        expect(gathered.dtype).toBe("f32");
        expect(gathered.shape).toEqual([2, 4]);
        const out = await gathered.cpu();
        // row 0 picks from rows [2,0,1,2] at cols [0,1,2,3] = [9,2,7,12]
        // row 1 picks from rows [0,1,2,0] at cols [0,1,2,3] = [1,6,11,4]
        expect(out).toEqual([9, 2, 7, 12, 1, 6, 11, 4]);
        api.markStep();
      });

      it("gather with u32 indices returns correct values", async () => {
        const api = new Torchlette("webgpu");
        const data = api.tensorFromArray(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
          [3, 4],
        );
        const idx = api.tensorFromArray([2, 0, 1, 2, 0, 1, 2, 0], [2, 4], {
          dtype: "u32",
        });
        const gathered = api.gather(data, idx, { dim: 0 });
        expect(gathered.shape).toEqual([2, 4]);
        const out = await gathered.cpu();
        expect(out).toEqual([9, 2, 7, 12, 1, 6, 11, 4]);
        api.markStep();
      });

      it("gather matches across i32/u32/f32 index dtypes", async () => {
        const api = new Torchlette("webgpu");
        const data = api.tensorFromArray(
          [10, 20, 30, 40, 50, 60, 70, 80],
          [2, 4],
        );
        const idxValues = [2, 0, 3, 1, 0, 2, 1, 3];
        const idxF32 = api.tensorFromArray(idxValues, [2, 4]);
        const idxI32 = api.tensorFromArray(idxValues, [2, 4], { dtype: "i32" });
        const idxU32 = api.tensorFromArray(idxValues, [2, 4], { dtype: "u32" });

        const gF32 = await api
          .gather(data, idxF32, { dim: 1 })
          .cpu();
        const gI32 = await api
          .gather(data, idxI32, { dim: 1 })
          .cpu();
        const gU32 = await api
          .gather(data, idxU32, { dim: 1 })
          .cpu();
        expect(gI32).toEqual(gF32);
        expect(gU32).toEqual(gF32);
        api.markStep();
      });

      it("api.embedding with i32 tokens returns correct embeddings", async () => {
        const api = new Torchlette("webgpu");
        // Known weight: 5 embeddings × 3 dims, row i = [i,i,i].
        const weight = api.tensorFromArray(
          [
            0, 0, 0, //
            1, 1, 1, //
            2, 2, 2, //
            3, 3, 3, //
            4, 4, 4, //
          ],
          [5, 3],
        );
        const tokens = api.tensorFromArray([0, 2, 4, 1], [4], { dtype: "i32" });
        const out = api.embedding(weight, tokens);
        expect(out.shape).toEqual([4, 3]);
        const data = await out.cpu();
        expect(data).toEqual([0, 0, 0, 2, 2, 2, 4, 4, 4, 1, 1, 1]);
        api.markStep();
      });

      it("api.embedding with i32 tokens: no i32→f32 cast in lazy graph", async () => {
        const api = new Torchlette("webgpu");
        const weight = api.randn([8, 4]);
        const tokens = api.tensorFromArray([0, 1, 2, 3], [4], { dtype: "i32" });
        const out = api.embedding(weight, tokens);

        // Walk the lazy IR graph backward from the output and verify no cast
        // node converts i32→f32 on the index path.
        const visited = new Set<number>();
        let sawI32ToF32Cast = false;
        const queue: any[] = [];
        const startRef = (out._unwrap() as any).lazyRef;
        if (startRef?.kind === "pending") queue.push(startRef.node);
        while (queue.length > 0) {
          const node = queue.shift();
          if (!node || visited.has(node.id)) continue;
          visited.add(node.id);
          if (node.op === "cast" && node.dtype === "f32") {
            // Check if input was i32/u32 (an index-path cast)
            for (const inp of node.inputs ?? []) {
              if (inp?.kind === "pending" && inp.node?.dtype !== "f32") {
                // Skip f16→f32 casts (AMP); only flag integer→f32
                const inDt = inp.node.dtype;
                if (inDt === "i32" || inDt === "u32") sawI32ToF32Cast = true;
              } else if (
                inp?.kind === "materialized" &&
                inp.storage?.dtype !== "f32" &&
                (inp.storage?.dtype === "i32" || inp.storage?.dtype === "u32")
              ) {
                sawI32ToF32Cast = true;
              }
            }
          }
          for (const inp of node.inputs ?? []) {
            if (inp?.kind === "pending" && inp.node) queue.push(inp.node);
          }
        }
        expect(sawI32ToF32Cast).toBe(false);
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
        // Gradients for ignored row (index 1) should be zero. (±0 both acceptable.)
        expect(grad?.[2]).toBeCloseTo(0, 10);
        expect(grad?.[3]).toBeCloseTo(0, 10);

        api.markStep();
      });
    });
  });
});
