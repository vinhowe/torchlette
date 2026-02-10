/**
 * Matmul Subgroup Support Tests
 *
 * Tests the subgroup-accelerated matmul variant. Subgroups allow efficient
 * register-to-register communication within a workgroup, providing significant
 * performance improvements on GPUs that support them.
 *
 * Tests are gated by:
 * - WebGPU auto-detection (skip with TORCHLETTE_CPU_ONLY=1)
 * - Runtime subgroup support detection (GPU feature availability)
 */
import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  getWebGPUDevice,
} from "../../src/backend/webgpu";
import {
  generateSubgroupMatmulShader,
  getSubgroupShaderCacheKey,
  type SubgroupCodegenOptions,
} from "../../src/backend/webgpu/matmul/subgroup";
import {
  getSubgroupSupport,
  setSubgroupSupport,
  clearSubgroupSupport,
  DEFAULT_CONFIG,
  validateConfig,
  type MatmulKernelConfig,
  type SubgroupSupport,
} from "../../src/backend/webgpu/matmul/types";
import { dispatchTiledMatmul } from "../../src/backend/webgpu/matmul/dispatch";

import { cpuOnly } from "../helpers/webgpu";

const SKIP = cpuOnly;

describe.skipIf(SKIP)("Matmul Subgroup Support", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  // Helper to check subgroup support - must be called inside test
  const hasSubgroupSupport = () => getSubgroupSupport()?.supported ?? false;

  describe("Subgroup Support Detection", () => {
    it("getSubgroupSupport returns detection result", () => {
      const support = getSubgroupSupport();

      expect(support).not.toBeNull();
      expect(support).toHaveProperty("supported");

      if (support?.supported) {
        expect(support.subgroupSize).toBeDefined();
        expect(support.subgroupSize).toBeGreaterThan(0);
      }
    });

    it("setSubgroupSupport updates cached value", () => {
      const originalSupport = getSubgroupSupport();

      // Set mock value
      setSubgroupSupport({ supported: true, subgroupSize: 16 });
      expect(getSubgroupSupport()).toEqual({ supported: true, subgroupSize: 16 });

      // Restore original
      if (originalSupport) {
        setSubgroupSupport(originalSupport);
      } else {
        clearSubgroupSupport();
      }
    });

    it("clearSubgroupSupport clears cached value", () => {
      const originalSupport = getSubgroupSupport();

      setSubgroupSupport({ supported: true, subgroupSize: 32 });
      clearSubgroupSupport();
      expect(getSubgroupSupport()).toBeNull();

      // Restore for other tests
      if (originalSupport) {
        setSubgroupSupport(originalSupport);
      }
    });
  });

  describe("Shader Generation", () => {
    const baseConfig: MatmulKernelConfig = {
      ...DEFAULT_CONFIG,
      useSubgroups: true,
    };

    it("generates valid shader code", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("enable subgroups;");
      expect(shader).toContain("@compute @workgroup_size");
      expect(shader).toContain("@builtin(subgroup_invocation_id)");
      expect(shader).toContain("@builtin(subgroup_size)");
      expect(shader).toContain("SUBGROUP_SIZE: u32 = 32u");
    });

    it("generates shader for all transpose modes", () => {
      const transposeModes = ["NN", "NT", "TN", "TT"] as const;

      for (const mode of transposeModes) {
        const options: SubgroupCodegenOptions = {
          config: baseConfig,
          transposeMode: mode,
          dtype: "f32",
          subgroupSize: 32,
        };

        const shader = generateSubgroupMatmulShader(options);

        expect(shader).toContain(`Transpose mode: ${mode}`);
        expect(shader).toContain("fn main(");
      }
    });

    it("generates shader with f16 support when needed", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f16",
        subgroupSize: 32,
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("enable f16;");
      expect(shader).toContain("enable subgroups;");
      expect(shader).toContain("Dtype: f16");
    });

    it("generates batched shader variant", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
        batched: true,
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("let batch_idx = wg_id.z;");
      expect(shader).toContain("batchStrideA");
      expect(shader).toContain("batchStrideB");
      expect(shader).toContain("batchStrideC");
    });

    it("generates non-batched shader variant", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
        batched: false,
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("let batch_idx = 0u;");
    });

    it("generates shader with epilogue ops", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
        epilogue: {
          ops: [
            { kind: "bias", inputIndex: 0 },
            { kind: "relu" },
          ],
          additionalInputCount: 1,
        },
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("epilogue_in0");
      expect(shader).toContain("biasVal0");
      expect(shader).toContain("select(0.0,"); // relu
    });

    it("generates shader with GELU epilogue", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
        epilogue: {
          ops: [{ kind: "gelu" }],
          additionalInputCount: 0,
        },
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("gelu_x");
      expect(shader).toContain("gelu_inner");
      expect(shader).toContain("tanh(gelu_inner)");
    });

    it("generates shader with SiLU epilogue", () => {
      const options: SubgroupCodegenOptions = {
        config: baseConfig,
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
        epilogue: {
          ops: [{ kind: "silu" }],
          additionalInputCount: 0,
        },
      };

      const shader = generateSubgroupMatmulShader(options);

      expect(shader).toContain("silu_x");
      expect(shader).toContain("exp(-silu_x)");
    });

    it("generates shader with different subgroup sizes", () => {
      for (const subgroupSize of [16, 32, 64]) {
        const options: SubgroupCodegenOptions = {
          config: baseConfig,
          transposeMode: "NN",
          dtype: "f32",
          subgroupSize,
        };

        const shader = generateSubgroupMatmulShader(options);

        expect(shader).toContain(`Subgroup size: ${subgroupSize}`);
        expect(shader).toContain(`SUBGROUP_SIZE: u32 = ${subgroupSize}u`);
      }
    });

    it("generates shader with different tile sizes", () => {
      const configs: MatmulKernelConfig[] = [
        { ...baseConfig, tileM: 32, tileN: 32, tileK: 8 },
        { ...baseConfig, tileM: 64, tileN: 64, tileK: 16 },
        { ...baseConfig, tileM: 64, tileN: 32, tileK: 16 },
      ];

      for (const config of configs) {
        const options: SubgroupCodegenOptions = {
          config,
          transposeMode: "NN",
          dtype: "f32",
          subgroupSize: 32,
        };

        const shader = generateSubgroupMatmulShader(options);

        expect(shader).toContain(`TILE_M=${config.tileM}`);
        expect(shader).toContain(`TILE_N=${config.tileN}`);
        expect(shader).toContain(`TILE_K=${config.tileK}`);
      }
    });
  });

  describe("Shader Cache Keys", () => {
    it("generates unique keys for different configs", () => {
      const options1: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const options2: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true, tileM: 64 },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const key1 = getSubgroupShaderCacheKey(options1);
      const key2 = getSubgroupShaderCacheKey(options2);

      expect(key1).not.toBe(key2);
    });

    it("generates unique keys for different transpose modes", () => {
      const baseOptions: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const keys = (["NN", "NT", "TN", "TT"] as const).map((mode) =>
        getSubgroupShaderCacheKey({ ...baseOptions, transposeMode: mode })
      );

      const uniqueKeys = new Set(keys);
      expect(uniqueKeys.size).toBe(4);
    });

    it("generates unique keys for different dtypes", () => {
      const baseOptions: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const keyF32 = getSubgroupShaderCacheKey(baseOptions);
      const keyF16 = getSubgroupShaderCacheKey({ ...baseOptions, dtype: "f16" });

      expect(keyF32).not.toBe(keyF16);
    });

    it("generates unique keys for batched vs non-batched", () => {
      const baseOptions: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const keyNotBatched = getSubgroupShaderCacheKey({
        ...baseOptions,
        batched: false,
      });
      const keyBatched = getSubgroupShaderCacheKey({
        ...baseOptions,
        batched: true,
      });

      expect(keyNotBatched).not.toBe(keyBatched);
    });

    it("includes subgroup size in key", () => {
      const baseOptions: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const key32 = getSubgroupShaderCacheKey(baseOptions);
      const key16 = getSubgroupShaderCacheKey({ ...baseOptions, subgroupSize: 16 });

      expect(key32).not.toBe(key16);
      expect(key32).toContain("sg32");
      expect(key16).toContain("sg16");
    });

    it("cache key starts with 'subgroup'", () => {
      const options: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: 32,
      };

      const key = getSubgroupShaderCacheKey(options);
      expect(key.startsWith("subgroup")).toBe(true);
    });
  });

  describe("GPU Dispatch (requires subgroup support)", () => {
    it("compiles and runs subgroup shader", async () => {
      if (!hasSubgroupSupport()) {
        return; // Skip if no subgroup support
      }

      const gpuContext = getWebGPUDevice();
      if (!gpuContext) {
        return;
      }

      const { device, queue } = gpuContext;
      const M = 64;
      const N = 64;
      const K = 64;

      // Create test matrices filled with 1s
      const aData = new Float32Array(M * K).fill(1.0);
      const bData = new Float32Array(K * N).fill(1.0);

      const aBuffer = device.createBuffer({
        size: aData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(aBuffer.getMappedRange()).set(aData);
      aBuffer.unmap();

      const bBuffer = device.createBuffer({
        size: bData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(bBuffer.getMappedRange()).set(bData);
      bBuffer.unmap();

      // Create output buffer
      const outBuffer = device.createBuffer({
        size: M * N * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Dispatch with subgroup config
      const subgroupConfig: MatmulKernelConfig = {
        ...DEFAULT_CONFIG,
        useSubgroups: true,
      };

      dispatchTiledMatmul({
        device,
        queue,
        a: aBuffer,
        b: bBuffer,
        out: outBuffer,
        m: M,
        n: N,
        k: K,
        config: subgroupConfig,
        dtype: "f32",
      });

      // Read back result
      const stagingBuffer = device.createBuffer({
        size: M * N * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(outBuffer, 0, stagingBuffer, 0, M * N * 4);
      queue.submit([encoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();

      // Verify: 1s @ 1s with K=64 should give all 64s
      for (let i = 0; i < M * N; i++) {
        expect(result[i]).toBeCloseTo(K, 1);
      }

      // Cleanup
      aBuffer.destroy();
      bBuffer.destroy();
      outBuffer.destroy();
      stagingBuffer.destroy();
    });

    it("subgroup matmul matches non-subgroup result", async () => {
      if (!hasSubgroupSupport()) {
        return; // Skip if no subgroup support
      }

      const gpuContext = getWebGPUDevice();
      if (!gpuContext) {
        return;
      }

      const { device, queue } = gpuContext;
      const M = 64;
      const N = 64;
      const K = 64;

      // Create random test matrices
      const aData = new Float32Array(M * K);
      const bData = new Float32Array(K * N);
      for (let i = 0; i < M * K; i++) aData[i] = Math.random() * 2 - 1;
      for (let i = 0; i < K * N; i++) bData[i] = Math.random() * 2 - 1;

      const aBuffer = device.createBuffer({
        size: aData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(aBuffer.getMappedRange()).set(aData);
      aBuffer.unmap();

      const bBuffer = device.createBuffer({
        size: bData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(bBuffer.getMappedRange()).set(bData);
      bBuffer.unmap();

      // Create output buffers
      const normalOut = device.createBuffer({
        size: M * N * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const subgroupOut = device.createBuffer({
        size: M * N * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Non-subgroup dispatch
      const normalConfig: MatmulKernelConfig = {
        ...DEFAULT_CONFIG,
        useSubgroups: false,
      };
      dispatchTiledMatmul({
        device,
        queue,
        a: aBuffer,
        b: bBuffer,
        out: normalOut,
        m: M,
        n: N,
        k: K,
        config: normalConfig,
        dtype: "f32",
      });

      // Subgroup dispatch
      const subgroupConfig: MatmulKernelConfig = {
        ...DEFAULT_CONFIG,
        useSubgroups: true,
      };
      dispatchTiledMatmul({
        device,
        queue,
        a: aBuffer,
        b: bBuffer,
        out: subgroupOut,
        m: M,
        n: N,
        k: K,
        config: subgroupConfig,
        dtype: "f32",
      });

      // Read both results
      const readBuffer = async (buf: GPUBuffer): Promise<Float32Array> => {
        const staging = device.createBuffer({
          size: M * N * 4,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(buf, 0, staging, 0, M * N * 4);
        queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
      };

      const normalResult = await readBuffer(normalOut);
      const subgroupResult = await readBuffer(subgroupOut);

      // Results should match within tolerance
      for (let i = 0; i < M * N; i++) {
        expect(subgroupResult[i]).toBeCloseTo(normalResult[i], 3);
      }

      // Cleanup
      aBuffer.destroy();
      bBuffer.destroy();
      normalOut.destroy();
      subgroupOut.destroy();
    });
  });

  describe("Shader Validation", () => {
    it("generated shader compiles on GPU", async () => {
      if (!hasSubgroupSupport()) {
        return; // Skip if no subgroup support
      }

      const gpuContext = getWebGPUDevice();
      if (!gpuContext) {
        return;
      }

      const { device } = gpuContext;
      const support = getSubgroupSupport();
      const options: SubgroupCodegenOptions = {
        config: { ...DEFAULT_CONFIG, useSubgroups: true },
        transposeMode: "NN",
        dtype: "f32",
        subgroupSize: support?.subgroupSize ?? 32,
      };

      const shader = generateSubgroupMatmulShader(options);

      // Try to compile the shader
      const shaderModule = device.createShaderModule({
        code: shader,
      });

      // Check for compilation errors
      const info = await shaderModule.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === "error");

      expect(errors.length).toBe(0);
    });

    it("reports when subgroups not supported", () => {
      const support = getSubgroupSupport();

      if (!support?.supported) {
        // This is expected on some hardware
        expect(support?.supported).toBe(false);
      } else {
        expect(support.subgroupSize).toBeGreaterThan(0);
      }
    });
  });
});
