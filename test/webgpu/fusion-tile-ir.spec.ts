/**
 * Tile-IR Elementwise Fusion Tests
 *
 * Verifies that the tile-IR fusion codegen path produces correct WGSL
 * and computes the same results as the string-template codegen.
 */
import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";
import {
  dispatchFusedKernel,
  resetFusionCache,
} from "../../src/backend/webgpu/fusion-dispatch";
import {
  generateFusedKernel,
  computeKernelMeta,
  type FusedKernelRecipe,
  type FusedNode,
  type FusedInput,
  type FusedOutput,
  type KernelGenOptions,
} from "../../src/backend/webgpu/fusion-codegen";
import { generateFusedKernelTileIR } from "../../src/backend/webgpu/fusion-tile-ir";
import { sizeOf } from "../../src/core/shape";

import { cpuOnly } from "../helpers/webgpu";

const SKIP = cpuOnly;

// Helper to build a simple unary recipe
function unaryRecipe(op: string, shape: number[], dtype: "f32" | "f16" = "f32"): FusedKernelRecipe {
  return {
    id: `test_${op}`,
    nodes: [{ id: 1, op, inputs: [-1], shape, dtype, isOutput: true }],
    inputs: [{ id: 100, index: 0, shape, dtype }],
    outputs: [{ nodeId: 1, index: 0, shape, dtype }],
  };
}

// Helper to build a simple binary recipe
function binaryRecipe(op: string, shape: number[], dtype: "f32" | "f16" = "f32"): FusedKernelRecipe {
  return {
    id: `test_${op}`,
    nodes: [{ id: 1, op, inputs: [-1, -2], shape, dtype, isOutput: true }],
    inputs: [
      { id: 100, index: 0, shape, dtype },
      { id: 101, index: 1, shape, dtype },
    ],
    outputs: [{ nodeId: 1, index: 0, shape, dtype }],
  };
}

describe("Tile-IR Fusion Codegen", () => {
  describe("WGSL generation", () => {
    it("generates valid WGSL for relu", () => {
      const recipe = unaryRecipe("relu", [16]);
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("@compute");
      expect(kernel.source).toContain("fn main");
      expect(kernel.source).toContain("select");
    });

    it("generates valid WGSL for add", () => {
      const recipe = binaryRecipe("add", [32]);
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("@compute");
      expect(kernel.source).toContain("fn main");
    });

    it("generates valid WGSL for gelu", () => {
      const recipe = unaryRecipe("gelu", [64]);
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("tanh");
    });

    it("generates valid WGSL for sigmoid", () => {
      const recipe = unaryRecipe("sigmoid", [64]);
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("exp");
    });

    it("generates valid WGSL for cast_f16", () => {
      const recipe: FusedKernelRecipe = {
        id: "test_cast_f16",
        nodes: [{ id: 1, op: "cast_f16", inputs: [-1], shape: [16], dtype: "f16", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [16], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [16], dtype: "f16" }],
      };
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("f16");
    });

    it("generates valid WGSL for chained ops", () => {
      const recipe: FusedKernelRecipe = {
        id: "test_chain",
        nodes: [
          { id: 1, op: "mul", inputs: [-1, -2], shape: [16], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [16], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [16], dtype: "f32" },
          { id: 101, index: 1, shape: [16], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [16], dtype: "f32" }],
      };
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("@compute");
      expect(kernel.source).toContain("select");
    });

    it("generates valid WGSL for multi-output", () => {
      const recipe: FusedKernelRecipe = {
        id: "test_multi_out",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [16], dtype: "f32", isOutput: true },
          { id: 2, op: "neg", inputs: [-1], shape: [16], dtype: "f32", isOutput: true },
        ],
        inputs: [{ id: 100, index: 0, shape: [16], dtype: "f32" }],
        outputs: [
          { nodeId: 1, index: 0, shape: [16], dtype: "f32" },
          { nodeId: 2, index: 1, shape: [16], dtype: "f32" },
        ],
      };
      const kernel = generateFusedKernelTileIR(recipe, {});
      // Should have two output bindings
      expect(kernel.source).toContain("out0");
      expect(kernel.source).toContain("out1");
    });

    it("generates valid WGSL for where (ternary)", () => {
      const recipe: FusedKernelRecipe = {
        id: "test_where",
        nodes: [{ id: 1, op: "where", inputs: [-1, -2, -3], shape: [16], dtype: "f32", isOutput: true }],
        inputs: [
          { id: 100, index: 0, shape: [16], dtype: "f32" },
          { id: 101, index: 1, shape: [16], dtype: "f32" },
          { id: 102, index: 2, shape: [16], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [16], dtype: "f32" }],
      };
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("select");
    });

    it("generates valid WGSL for isfinite", () => {
      const recipe = unaryRecipe("isfinite", [16]);
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("bitcast");
      // tile-IR emits 0x7F800000 as decimal 2139095040u
      expect(kernel.source).toContain("2139095040u");
    });

    it("handles inlined constants", () => {
      const recipe: FusedKernelRecipe = {
        id: "test_inlined",
        nodes: [{ id: 1, op: "add", inputs: [-1, -2], shape: [16], dtype: "f32", isOutput: true }],
        inputs: [
          { id: 100, index: 0, shape: [16], dtype: "f32" },
          { id: 101, index: 1, shape: [1], dtype: "f32", isInlinedConstant: true, inlinedValue: 2.0 },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [16], dtype: "f32" }],
      };
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.inputBindings).toBe(1); // Only 1 non-inlined input
      expect(kernel.source).toContain("2.0");
    });

    it("handles broadcasting", () => {
      const recipe: FusedKernelRecipe = {
        id: "test_broadcast",
        nodes: [{ id: 1, op: "add", inputs: [-1, -2], shape: [4, 4], dtype: "f32", isOutput: true }],
        inputs: [
          { id: 100, index: 0, shape: [4, 4], dtype: "f32" },
          { id: 101, index: 1, shape: [1, 4], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [4, 4], dtype: "f32" }],
      };
      const kernel = generateFusedKernelTileIR(recipe, {});
      expect(kernel.source).toContain("@compute");
    });

    it("returns same metadata as computeKernelMeta", () => {
      const recipe = unaryRecipe("relu", [1024]);
      const options: KernelGenOptions = { vectorize: true };
      const meta = computeKernelMeta(recipe, options);
      const kernel = generateFusedKernelTileIR(recipe, options);
      expect(kernel.cacheKey).toBe(meta.cacheKey);
      expect(kernel.vectorWidth).toBe(meta.vectorWidth);
      expect(kernel.workItems).toBe(meta.workItems);
      expect(kernel.workgroupSize).toBe(meta.workgroupSize);
      expect(kernel.gridSizeX).toBe(meta.gridSizeX);
    });

    // Test every op in the registry compiles
    const allOps = [
      // Unary
      "relu", "gelu", "gelu_tanh", "gelu_erf", "silu", "sigmoid", "tanh", "softplus",
      "neg", "abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "floor", "ceil",
      "round", "sign", "isfinite",
      // Casts
      "cast_f32", "cast_i32", "cast_u32",
    ];

    for (const op of allOps) {
      it(`compiles ${op} to valid WGSL`, () => {
        const recipe = unaryRecipe(op, [64]);
        const kernel = generateFusedKernelTileIR(recipe, {});
        expect(kernel.source).toContain("@compute");
        expect(kernel.source).toContain("fn main");
      });
    }

    const binaryOps = ["add", "sub", "mul", "div", "pow", "min", "max", "mod"];
    for (const op of binaryOps) {
      it(`compiles binary ${op} to valid WGSL`, () => {
        const recipe = binaryRecipe(op, [64]);
        const kernel = generateFusedKernelTileIR(recipe, {});
        expect(kernel.source).toContain("@compute");
      });
    }

    const cmpOps = ["eq", "ne", "lt", "le", "gt", "ge"];
    for (const op of cmpOps) {
      it(`compiles comparison ${op} to valid WGSL`, () => {
        const recipe = binaryRecipe(op, [64]);
        const kernel = generateFusedKernelTileIR(recipe, {});
        expect(kernel.source).toContain("@compute");
      });
    }
  });

  // GPU dispatch tests — only run when WebGPU is available
  describe.skipIf(SKIP)("GPU dispatch correctness", () => {
    beforeAll(async () => {
      await initWebGPU();
    });

    async function readBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
      const device = webgpuBackend.device!;
      const staging = device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return result;
    }

    async function dispatchAndRead(
      recipe: FusedKernelRecipe,
      inputArrays: Float32Array[],
      options: { vectorize?: boolean } = {},
    ): Promise<Float32Array[]> {
      const device = webgpuBackend.device!;
      const inputs = inputArrays.map((arr, i) => {
        const recipeInput = recipe.inputs.filter(inp => !inp.isInlinedConstant)[i];
        const buf = device.createBuffer({
          size: arr.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          mappedAtCreation: true,
        });
        new Float32Array(buf.getMappedRange()).set(arr);
        buf.unmap();
        return { buffer: buf, shape: recipeInput.shape, dtype: recipeInput.dtype as "f32" };
      });

      const result = dispatchFusedKernel(device, recipe, inputs, {
        vectorize: options.vectorize ?? false,
      });

      // Force GPU work
      device.queue.submit([]);
      await device.queue.onSubmittedWorkDone();

      const outputs: Float32Array[] = [];
      for (const out of result.outputs) {
        const data = await readBuffer(out.buffer, sizeOf(out.shape) * 4);
        outputs.push(data);
      }

      // Cleanup input buffers
      for (const inp of inputs) inp.buffer.destroy();

      return outputs;
    }

    it("relu via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe = unaryRecipe("relu", [8]);
      const input = new Float32Array([-2, -1, 0, 0.5, 1, 2, -0.5, 3]);
      const [result] = await dispatchAndRead(recipe, [input]);
      expect(Array.from(result)).toEqual([0, 0, 0, 0.5, 1, 2, 0, 3]);
    });

    it("add via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe = binaryRecipe("add", [4]);
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([10, 20, 30, 40]);
      const [result] = await dispatchAndRead(recipe, [a, b]);
      expect(Array.from(result)).toEqual([11, 22, 33, 44]);
    });

    it("mul + relu chain via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe: FusedKernelRecipe = {
        id: "test_mul_relu",
        nodes: [
          { id: 1, op: "mul", inputs: [-1, -2], shape: [4], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [4], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [4], dtype: "f32" },
          { id: 101, index: 1, shape: [4], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [4], dtype: "f32" }],
      };
      const a = new Float32Array([1, -2, 3, -4]);
      const b = new Float32Array([2, 3, -1, -2]);
      const [result] = await dispatchAndRead(recipe, [a, b]);
      // mul: [2, -6, -3, 8], relu: [2, 0, 0, 8]
      expect(Array.from(result)).toEqual([2, 0, 0, 8]);
    });

    it("where (ternary) via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe: FusedKernelRecipe = {
        id: "test_where_dispatch",
        nodes: [{ id: 1, op: "where", inputs: [-1, -2, -3], shape: [4], dtype: "f32", isOutput: true }],
        inputs: [
          { id: 100, index: 0, shape: [4], dtype: "f32" },
          { id: 101, index: 1, shape: [4], dtype: "f32" },
          { id: 102, index: 2, shape: [4], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [4], dtype: "f32" }],
      };
      const cond = new Float32Array([1, 0, 1, 0]); // >0 = true
      const a = new Float32Array([10, 20, 30, 40]);
      const b = new Float32Array([100, 200, 300, 400]);
      const [result] = await dispatchAndRead(recipe, [cond, a, b]);
      expect(Array.from(result)).toEqual([10, 200, 30, 400]);
    });

    it("broadcasting via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe: FusedKernelRecipe = {
        id: "test_broadcast_dispatch",
        nodes: [{ id: 1, op: "add", inputs: [-1, -2], shape: [2, 3], dtype: "f32", isOutput: true }],
        inputs: [
          { id: 100, index: 0, shape: [2, 3], dtype: "f32" },
          { id: 101, index: 1, shape: [1, 3], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [2, 3], dtype: "f32" }],
      };
      const a = new Float32Array([1, 2, 3, 4, 5, 6]);
      const bias = new Float32Array([10, 20, 30]);
      const [result] = await dispatchAndRead(recipe, [a, bias]);
      expect(Array.from(result)).toEqual([11, 22, 33, 14, 25, 36]);
    });

    it("inlined constants via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe: FusedKernelRecipe = {
        id: "test_inlined_dispatch",
        nodes: [{ id: 1, op: "mul", inputs: [-1, -2], shape: [4], dtype: "f32", isOutput: true }],
        inputs: [
          { id: 100, index: 0, shape: [4], dtype: "f32" },
          { id: 101, index: 1, shape: [1], dtype: "f32", isInlinedConstant: true, inlinedValue: 3.0 },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [4], dtype: "f32" }],
      };
      const a = new Float32Array([1, 2, 3, 4]);
      const [result] = await dispatchAndRead(recipe, [a]);
      expect(Array.from(result)).toEqual([3, 6, 9, 12]);
    });

    it("multi-output via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe: FusedKernelRecipe = {
        id: "test_multi_out_dispatch",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [4], dtype: "f32", isOutput: true },
          { id: 2, op: "neg", inputs: [-1], shape: [4], dtype: "f32", isOutput: true },
        ],
        inputs: [{ id: 100, index: 0, shape: [4], dtype: "f32" }],
        outputs: [
          { nodeId: 1, index: 0, shape: [4], dtype: "f32" },
          { nodeId: 2, index: 1, shape: [4], dtype: "f32" },
        ],
      };
      const input = new Float32Array([-1, 2, -3, 4]);
      const [out0, out1] = await dispatchAndRead(recipe, [input]);
      expect(Array.from(out0)).toEqual([0, 2, 0, 4]); // relu
      expect(Array.from(out1)).toEqual([1, -2, 3, -4]); // neg
    });

    it("sigmoid via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe = unaryRecipe("sigmoid", [4]);
      const input = new Float32Array([0, 1, -1, 10]);
      const [result] = await dispatchAndRead(recipe, [input]);
      // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269, sigmoid(10) ≈ 1.0
      expect(result[0]).toBeCloseTo(0.5, 4);
      expect(result[1]).toBeCloseTo(0.7310586, 4);
      expect(result[2]).toBeCloseTo(0.2689414, 4);
      expect(result[3]).toBeCloseTo(1.0, 3);
    });

    it("gelu via tile-IR dispatch", async () => {
      resetFusionCache();
      const recipe = unaryRecipe("gelu", [4]);
      const input = new Float32Array([0, 1, -1, 2]);
      const [result] = await dispatchAndRead(recipe, [input]);
      // GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159, GELU(2) ≈ 1.955
      expect(result[0]).toBeCloseTo(0, 3);
      expect(result[1]).toBeCloseTo(0.841, 2);
      expect(result[2]).toBeCloseTo(-0.159, 2);
      expect(result[3]).toBeCloseTo(1.955, 2);
    });

    it("vectorized relu via tile-IR dispatch (vec4)", async () => {
      resetFusionCache();
      const recipe = unaryRecipe("relu", [32]);
      const input = new Float32Array(32);
      for (let i = 0; i < 32; i++) input[i] = i - 16;
      const [result] = await dispatchAndRead(recipe, [input], { vectorize: true });
      for (let i = 0; i < 32; i++) {
        expect(result[i]).toBe(Math.max(0, i - 16));
      }
    });

    it("larger tensor via tile-IR dispatch", async () => {
      resetFusionCache();
      const N = 4096;
      const recipe = unaryRecipe("neg", [N]);
      const input = new Float32Array(N);
      for (let i = 0; i < N; i++) input[i] = i;
      const [result] = await dispatchAndRead(recipe, [input]);
      for (let i = 0; i < N; i++) {
        expect(result[i]).toBe(-i);
      }
    });
  });
});
