/**
 * Fusion Integration Tests (§15)
 *
 * Tests the complete fusion pipeline from detection through dispatch:
 * - §15.1: Elementwise fusion (single output)
 * - §15.2: Multi-output fusion
 * - §15.3: Memory coalescing via vectorization
 * - §15.4: Random ops as fusion barriers
 */
import { describe, expect, it, beforeAll, afterAll } from "vitest";
import { Torchlette } from "../../src/frontend";
import {
  initWebGPU,
  webgpuBackend,
  getWebGPUDevice,
} from "../../src/backend/webgpu";
import {
  dispatchFusedKernel,
  FusionKernelCache,
  getFusionCache,
  resetFusionCache,
} from "../../src/backend/webgpu/fusion-dispatch";
import {
  generateFusedKernel,
  buildRecipeFromIR,
  type FusedKernelRecipe,
} from "../../src/backend/webgpu/fusion-codegen";
import type { IRNode } from "../../src/engine/ir";

import { cpuOnly } from "../helpers/webgpu";

const SKIP = cpuOnly;

// Type for WebGPU backend tensors
type WebGPUTensor = ReturnType<typeof webgpuBackend.ops.tensorFromArray> & {
  buffer: GPUBuffer;
  shape: number[];
  dtype: string;
};

describe.skipIf(SKIP)("Fusion Integration (§15)", () => {
  let api: Torchlette;

  beforeAll(async () => {
    await initWebGPU();
    api = new Torchlette("webgpu");
  });

  afterAll(() => {
    resetFusionCache();
  });

  describe("Backend Integration", () => {
    it("webgpuBackend exposes dispatchFusedKernel", () => {
      expect(webgpuBackend.dispatchFusedKernel).toBeDefined();
      expect(typeof webgpuBackend.dispatchFusedKernel).toBe("function");
    });

    it("webgpuBackend exposes device property", () => {
      expect("device" in webgpuBackend).toBe(true);
      const device = webgpuBackend.device;
      expect(device).not.toBeNull();
    });

    it("getWebGPUDevice returns device and queue", () => {
      const result = getWebGPUDevice();
      expect(result).not.toBeNull();
      expect(result!.device).toBeDefined();
      expect(result!.queue).toBeDefined();
    });
  });

  describe("FusionKernelCache", () => {
    it("caches pipelines by key", () => {
      const cache = new FusionKernelCache();
      const device = webgpuBackend.device!;

      const recipe: FusedKernelRecipe = {
        id: "test_cache",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [16], dtype: "f32", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [16], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [16], dtype: "f32" }],
      };

      const result1 = cache.getOrCreate(device, recipe);
      const result2 = cache.getOrCreate(device, recipe);

      // Same pipeline should be returned
      expect(result1.pipeline).toBe(result2.pipeline);
      expect(cache.stats().size).toBe(1);
    });

    it("evicts oldest on overflow", () => {
      const cache = new FusionKernelCache(2); // max 2 entries
      const device = webgpuBackend.device!;

      const makeRecipe = (id: number): FusedKernelRecipe => ({
        id: `test_${id}`,
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [id * 16], dtype: "f32", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [id * 16], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [id * 16], dtype: "f32" }],
      });

      cache.getOrCreate(device, makeRecipe(1));
      cache.getOrCreate(device, makeRecipe(2));
      expect(cache.stats().size).toBe(2);

      // Adding 3rd should evict oldest
      cache.getOrCreate(device, makeRecipe(3));
      expect(cache.stats().size).toBe(2);
    });

    it("global cache is accessible", () => {
      const cache = getFusionCache();
      expect(cache).toBeInstanceOf(FusionKernelCache);
    });
  });

  describe("§15.1 Elementwise Fusion", () => {
    it("fuses add + relu into single kernel", async () => {
      const device = webgpuBackend.device!;

      // Create input tensors using backend directly
      const a = webgpuBackend.ops.tensorFromArray([1, -2, 3, -4], [4]) as WebGPUTensor;
      const b = webgpuBackend.ops.tensorFromArray([1, 1, 1, 1], [4]) as WebGPUTensor;

      // Create recipe for add -> relu
      const recipe: FusedKernelRecipe = {
        id: "add_relu",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [4], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [4], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [4], dtype: "f32" },
          { id: 101, index: 1, shape: [4], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [4], dtype: "f32" }],
      };

      const result = dispatchFusedKernel(device, recipe, [
        { buffer: a.buffer, shape: [4], dtype: "f32" },
        { buffer: b.buffer, shape: [4], dtype: "f32" },
      ]);

      expect(result.shape).toEqual([4]);
      expect(result.dtype).toBe("f32");

      // Read result: relu(add([1,-2,3,-4], [1,1,1,1])) = relu([2,-1,4,-3]) = [2,0,4,0]
      const resultBuffer = result.buffer as GPUBuffer;
      const stagingBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      stagingBuffer.destroy();

      expect(Array.from(data)).toEqual([2, 0, 4, 0]);
    });

    it("fuses chain of 3+ ops", async () => {
      const device = webgpuBackend.device!;

      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]) as WebGPUTensor;

      // neg -> abs -> relu (should be identity for positive after abs)
      const recipe: FusedKernelRecipe = {
        id: "chain_3",
        nodes: [
          { id: 1, op: "neg", inputs: [-1], shape: [4], dtype: "f32" },
          { id: 2, op: "abs", inputs: [1], shape: [4], dtype: "f32" },
          { id: 3, op: "relu", inputs: [2], shape: [4], dtype: "f32", isOutput: true },
        ],
        inputs: [{ id: 100, index: 0, shape: [4], dtype: "f32" }],
        outputs: [{ nodeId: 3, index: 0, shape: [4], dtype: "f32" }],
      };

      const result = dispatchFusedKernel(device, recipe, [
        { buffer: a.buffer, shape: [4], dtype: "f32" },
      ]);

      // neg([1,2,3,4]) = [-1,-2,-3,-4]
      // abs([-1,-2,-3,-4]) = [1,2,3,4]
      // relu([1,2,3,4]) = [1,2,3,4]
      const resultBuffer = result.buffer as GPUBuffer;
      const stagingBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      stagingBuffer.destroy();

      expect(Array.from(data)).toEqual([1, 2, 3, 4]);
    });

    it("handles broadcasting in fused kernels", async () => {
      const device = webgpuBackend.device!;

      // [2, 4] + [4] with broadcast
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]) as WebGPUTensor;
      const b = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40], [4]) as WebGPUTensor;

      const recipe: FusedKernelRecipe = {
        id: "broadcast_add",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [2, 4], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [2, 4], dtype: "f32" },
          { id: 101, index: 1, shape: [4], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [2, 4], dtype: "f32" }],
      };

      const result = dispatchFusedKernel(device, recipe, [
        { buffer: a.buffer, shape: [2, 4], dtype: "f32" },
        { buffer: b.buffer, shape: [4], dtype: "f32" },
      ]);

      expect(result.shape).toEqual([2, 4]);

      const resultBuffer = result.buffer as GPUBuffer;
      const stagingBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 32);
      device.queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();
      stagingBuffer.destroy();

      // [1+10, 2+20, 3+30, 4+40, 5+10, 6+20, 7+30, 8+40]
      expect(Array.from(data)).toEqual([11, 22, 33, 44, 15, 26, 37, 48]);
    });
  });

  describe("§15.2 Multi-Output Fusion", () => {
    it("generates kernel with multiple outputs", async () => {
      const device = webgpuBackend.device!;

      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]) as WebGPUTensor;

      // Two outputs sharing common input
      const recipe: FusedKernelRecipe = {
        id: "multi_output",
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

      const result = dispatchFusedKernel(device, recipe, [
        { buffer: a.buffer, shape: [4], dtype: "f32" },
      ]);

      expect(result.outputs).toHaveLength(2);
      expect(result.outputs[0].shape).toEqual([4]);
      expect(result.outputs[1].shape).toEqual([4]);

      // Read both outputs
      const readOutput = async (buffer: GPUBuffer): Promise<Float32Array> => {
        const staging = device.createBuffer({
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(buffer, 0, staging, 0, 16);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
      };

      const out0 = await readOutput(result.outputs[0].buffer as GPUBuffer);
      const out1 = await readOutput(result.outputs[1].buffer as GPUBuffer);

      // out0 = relu([1,2,3,4]) = [1,2,3,4]
      // out1 = neg([1,2,3,4]) = [-1,-2,-3,-4]
      expect(Array.from(out0)).toEqual([1, 2, 3, 4]);
      expect(Array.from(out1)).toEqual([-1, -2, -3, -4]);
    });
  });

  describe("§15.3 Vectorization", () => {
    it("uses vec4 for aligned shapes", () => {
      const recipe: FusedKernelRecipe = {
        id: "vec4_test",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [256], dtype: "f32", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [256], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(4);
      expect(kernel.workItems).toBe(64); // 256 / 4
      expect(kernel.source).toContain("vec4");
    });

    it("uses vec2 for 2-aligned shapes", () => {
      const recipe: FusedKernelRecipe = {
        id: "vec2_test",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [66], dtype: "f32", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [66], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [66], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(2);
      expect(kernel.workItems).toBe(33); // 66 / 2
    });

    it("falls back to scalar for odd shapes", () => {
      const recipe: FusedKernelRecipe = {
        id: "scalar_test",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [33], dtype: "f32", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [33], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [33], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(1);
      expect(kernel.workItems).toBe(33);
    });

    it("vectorized dispatch produces correct results", async () => {
      const device = webgpuBackend.device!;

      // Use 256 elements to ensure vec4
      const inputData = new Array(256).fill(0).map((_, i) => i - 128);
      const a = webgpuBackend.ops.tensorFromArray(inputData, [256]) as WebGPUTensor;

      const recipe: FusedKernelRecipe = {
        id: "vec4_relu",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [256], dtype: "f32", isOutput: true }],
        inputs: [{ id: 100, index: 0, shape: [256], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      const result = dispatchFusedKernel(device, recipe, [
        { buffer: a.buffer, shape: [256], dtype: "f32" },
      ], { vectorize: true });

      const staging = device.createBuffer({
        size: 1024,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(result.buffer as GPUBuffer, 0, staging, 0, 1024);
      device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();

      // Verify relu: max(0, x)
      const expected = inputData.map(x => Math.max(0, x));
      for (let i = 0; i < 256; i++) {
        expect(data[i]).toBeCloseTo(expected[i], 5);
      }
    });
  });

  describe("§15.4 Fusion Barriers", () => {
    it("fusion detection identifies fusible chains", () => {
      // Create mock IR nodes
      const nodes: IRNode[] = [
        { id: 0, op: "input", inputs: [], shape: [4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
        { id: 1, op: "relu", inputs: [0], shape: [4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
        { id: 2, op: "neg", inputs: [1], shape: [4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
      ];

      const nodeById = new Map(nodes.map(n => [n.id, n]));

      // Build recipe - relu and neg should fuse
      const recipe = buildRecipeFromIR([1, 2], nodeById, [0]);

      expect(recipe.nodes).toHaveLength(2);
      expect(recipe.nodes[0].op).toBe("relu");
      expect(recipe.nodes[1].op).toBe("neg");
    });

    it("non-fusible ops break chains", () => {
      // Matmul is not fusible
      const nodes: IRNode[] = [
        { id: 0, op: "input", inputs: [], shape: [4, 4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
        { id: 1, op: "relu", inputs: [0], shape: [4, 4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
        { id: 2, op: "matmul", inputs: [1, 0], shape: [4, 4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
        { id: 3, op: "neg", inputs: [2], shape: [4, 4], dtype: "f32", epoch: 1, kind: "lazy_op" as const },
      ];

      // Can't fuse across matmul
      const nodeById = new Map(nodes.map(n => [n.id, n]));

      // Only relu can fuse with input
      const recipe1 = buildRecipeFromIR([1], nodeById, [0]);
      expect(recipe1.nodes).toHaveLength(1);

      // Only neg can be after matmul
      const recipe2 = buildRecipeFromIR([3], nodeById, [2]);
      expect(recipe2.nodes).toHaveLength(1);
    });
  });

  describe("End-to-End Fusion", () => {
    it("fusion produces same results as sequential execution", async () => {
      const device = webgpuBackend.device!;

      // Input data
      const aData = [1, -2, 3, -4, 5, -6, 7, -8];
      const bData = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

      // Expected: relu(add(a, b)) = relu([1.5, -1.5, 3.5, -3.5, 5.5, -5.5, 7.5, -7.5])
      //                           = [1.5, 0, 3.5, 0, 5.5, 0, 7.5, 0]
      const expectedOutput = aData.map((a, i) => Math.max(0, a + bData[i]));

      // Create backend tensors
      const a = webgpuBackend.ops.tensorFromArray(aData, [8]) as WebGPUTensor;
      const b = webgpuBackend.ops.tensorFromArray(bData, [8]) as WebGPUTensor;

      // Fused: single kernel for add + relu
      const recipe: FusedKernelRecipe = {
        id: "add_relu_e2e",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [8], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [8], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [8], dtype: "f32" },
          { id: 101, index: 1, shape: [8], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [8], dtype: "f32" }],
      };

      const fusedResult = dispatchFusedKernel(device, recipe, [
        { buffer: a.buffer, shape: [8], dtype: "f32" },
        { buffer: b.buffer, shape: [8], dtype: "f32" },
      ]);

      // Read fused result
      const staging = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(fusedResult.buffer as GPUBuffer, 0, staging, 0, 32);
      device.queue.submit([encoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const fusedOutput = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();

      // Compare with expected values
      for (let i = 0; i < 8; i++) {
        expect(fusedOutput[i]).toBeCloseTo(expectedOutput[i], 5);
      }
    });

    it("fused kernel via frontend matches direct backend execution", async () => {
      // This test verifies that the fusion infrastructure integrates correctly
      // by comparing frontend lazy execution with expected values
      const aData = [1, -2, 3, -4];
      const bData = [1, 1, 1, 1];

      // Create frontend tensors and compute lazily
      const a = api.tensorFromArray(aData, [4]);
      const b = api.tensorFromArray(bData, [4]);
      const result = api.add(a, b).relu();

      // Force execution and get values
      const output = await result.cpu();

      // Expected: relu(add([1,-2,3,-4], [1,1,1,1])) = relu([2,-1,4,-3]) = [2,0,4,0]
      expect(Array.from(output)).toEqual([2, 0, 4, 0]);
    });
  });
});
