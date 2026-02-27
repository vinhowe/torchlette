/**
 * Tile-IR Autotuning Framework tests.
 *
 * Tests config generation (pure logic, no GPU) and autotuning dispatch (GPU).
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  getWebGPUDevice,
  initWebGPU,
  syncWebGPU,
  beginSharedEncoder,
  flushSharedEncoder,
} from "../../src/backend/webgpu";
import { createTileKernelDispatcher, createAutoTileKernelDispatcher } from "../../src/backend/webgpu/tile-dispatch";
import type { TileKernelSpec, AutotuneConfig, TuneParam } from "../../src/backend/webgpu/tile-ir";
import type { GPUBuffer, GPUDevice, GPUQueue } from "../../src/backend/webgpu/gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../../src/backend/webgpu/gpu-types";
import {
  generateTileConfigs,
  getDefaultConfig,
  autotuneTileKernel,
  clearTileAutotuneCache,
  exportTileAutotuneCache,
  importTileAutotuneCache,
} from "../../src/backend/webgpu/tile-autotune";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

// ============================================================================
// GPU helpers
// ============================================================================

let device: GPUDevice;
let queue: GPUQueue;

function makeF32Buffer(data: Float32Array, usage: number): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

function makeOutputBuffer(numElements: number): GPUBuffer {
  return device.createBuffer({
    size: numElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

async function readF32Buffer(buf: GPUBuffer, count: number): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// ============================================================================
// Simple tunable kernel: elementwise scale with configurable workgroup size
// ============================================================================

function makeScaleSpec(wgSize: number): TileKernelSpec {
  return {
    name: `scale_wg${wgSize}`,
    workgroupSize: wgSize,
    bindings: {
      input:  { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { N: "u32", scale: "f32" },
    grid: (u) => [Math.ceil(u.N / wgSize)],
    kernel(ctx) {
      const gid = ctx.globalId(0);
      const N = ctx.uniform("N");
      const scale = ctx.uniform("scale", "f32");
      const guard = gid.lt(N);
      const val = ctx.load("input", gid);
      ctx.guardedStore("output", guard, gid, val.mul(scale));
    },
  };
}

const scaleAutoConfig: AutotuneConfig = {
  factory: (config) => makeScaleSpec(config.wgSize),
  params: {
    wgSize: { values: [32, 64, 128, 256], default: 64 },
  },
};

// ============================================================================
// Config generation tests (no GPU needed)
// ============================================================================

describe("tile-autotune config generation", () => {
  it("generates Cartesian product of params", () => {
    const autoConfig: AutotuneConfig = {
      factory: (config) => makeScaleSpec(config.wgSize),
      params: {
        wgSize: { values: [32, 64, 128], default: 64 },
      },
    };

    const configs = generateTileConfigs(autoConfig);
    expect(configs).toHaveLength(3);
    expect(configs.map(c => c.wgSize)).toEqual([32, 64, 128]);
  });

  it("applies constraints to filter invalid configs", () => {
    const autoConfig: AutotuneConfig = {
      factory: (config) => makeScaleSpec(config.wgSize),
      params: {
        wgSize: { values: [32, 64, 128, 256, 512], default: 64 },
      },
      constraints: [
        (c) => c.wgSize <= 256, // Exclude 512
      ],
    };

    const configs = generateTileConfigs(autoConfig);
    expect(configs).toHaveLength(4);
    expect(configs.every(c => c.wgSize <= 256)).toBe(true);
  });

  it("handles multi-param Cartesian product", () => {
    const autoConfig: AutotuneConfig = {
      factory: (config) => makeScaleSpec(config.wgSize),
      params: {
        wgSize: { values: [32, 64], default: 64 },
        vecWidth: { values: [1, 4], default: 1 },
      },
    };

    const configs = generateTileConfigs(autoConfig);
    expect(configs).toHaveLength(4); // 2 × 2
    expect(configs).toEqual([
      { wgSize: 32, vecWidth: 1 },
      { wgSize: 32, vecWidth: 4 },
      { wgSize: 64, vecWidth: 1 },
      { wgSize: 64, vecWidth: 4 },
    ]);
  });

  it("multi-param with constraint", () => {
    const autoConfig: AutotuneConfig = {
      factory: (config) => makeScaleSpec(config.wgSize),
      params: {
        wgSize: { values: [32, 64, 128], default: 64 },
        vecWidth: { values: [1, 2, 4], default: 1 },
      },
      constraints: [
        (c) => c.wgSize * c.vecWidth <= 256,
      ],
    };

    const configs = generateTileConfigs(autoConfig);
    // 32*1=32, 32*2=64, 32*4=128, 64*1=64, 64*2=128, 64*4=256, 128*1=128, 128*2=256, 128*4=512 (excluded)
    expect(configs).toHaveLength(8);
    expect(configs.every(c => c.wgSize * c.vecWidth <= 256)).toBe(true);
  });

  it("returns empty array when all configs are filtered", () => {
    const autoConfig: AutotuneConfig = {
      factory: (config) => makeScaleSpec(config.wgSize),
      params: {
        wgSize: { values: [32, 64], default: 64 },
      },
      constraints: [
        (_c) => false, // Reject everything
      ],
    };

    const configs = generateTileConfigs(autoConfig);
    expect(configs).toHaveLength(0);
  });

  it("pruneForShape narrows param space", () => {
    const autoConfig: AutotuneConfig = {
      factory: (config) => makeScaleSpec(config.wgSize),
      params: {
        wgSize: { values: [32, 64, 128, 256], default: 64 },
      },
      pruneForShape: (uniforms) => {
        // For small N, only try small workgroup sizes
        if (uniforms.N < 128) {
          return { wgSize: { values: [32, 64], default: 32 } };
        }
        return { wgSize: { values: [32, 64, 128, 256], default: 64 } };
      },
    };

    const smallConfigs = generateTileConfigs(autoConfig, { N: 64 });
    expect(smallConfigs).toHaveLength(2);

    const largeConfigs = generateTileConfigs(autoConfig, { N: 1024 });
    expect(largeConfigs).toHaveLength(4);
  });

  it("getDefaultConfig returns correct defaults", () => {
    const config = getDefaultConfig(scaleAutoConfig);
    expect(config).toEqual({ wgSize: 64 });
  });
});

// ============================================================================
// Cache round-trip tests (no GPU needed)
// ============================================================================

describe("tile-autotune cache", () => {
  it("export/import round-trip preserves entries", () => {
    clearTileAutotuneCache();

    // Manually populate via import
    const entries = [
      ["testKernel:N=256", { config: { wgSize: 128 }, medianMs: 0.5 }],
    ];
    importTileAutotuneCache(JSON.stringify(entries));

    const exported = exportTileAutotuneCache();
    const parsed = JSON.parse(exported);
    expect(parsed).toHaveLength(1);
    expect(parsed[0][0]).toBe("testKernel:N=256");
    expect(parsed[0][1].config.wgSize).toBe(128);
    expect(parsed[0][1].medianMs).toBe(0.5);

    clearTileAutotuneCache();
  });
});

// ============================================================================
// GPU tests: actual autotuning
// ============================================================================

describe.skipIf(!isWebGPUEnabled)("tile-autotune GPU", () => {
  beforeAll(async () => {
    await initWebGPU();
    const ctx = getWebGPUDevice();
    device = ctx.device;
    queue = ctx.queue;
    clearTileAutotuneCache();
  });

  afterAll(() => {
    clearTileAutotuneCache();
  });

  it("autotuneTileKernel selects a config and caches it", async () => {
    const N = 1024;
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) inputData[i] = i;

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N);

    const result = await autotuneTileKernel(
      scaleAutoConfig,
      { input: inputBuf, output: outputBuf },
      { N, scale: 2.0 },
      { warmupIters: 1, timingIters: 2 },
    );

    // Should pick one of the valid configs
    expect([32, 64, 128, 256]).toContain(result.config.wgSize);
    expect(result.medianMs).toBeGreaterThan(0);
    expect(result.medianMs).toBeLessThan(100); // Sanity check — should be fast

    // Second call should use cache (fast)
    const result2 = await autotuneTileKernel(
      scaleAutoConfig,
      { input: inputBuf, output: outputBuf },
      { N, scale: 2.0 },
    );
    expect(result2.config.wgSize).toBe(result.config.wgSize);
    expect(result2.medianMs).toBe(result.medianMs); // Exact same (cached)

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("createAutoTileKernelDispatcher dispatches correctly", async () => {
    const N = 256;
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) inputData[i] = i + 1;

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N);

    const autoDispatcher = createAutoTileKernelDispatcher(scaleAutoConfig);

    // Dispatch with default config
    beginSharedEncoder();
    autoDispatcher.dispatch(
      { input: inputBuf, output: outputBuf },
      { N, scale: 3.0 },
    );
    flushSharedEncoder();
    await syncWebGPU();

    const result = await readF32Buffer(outputBuf, N);
    // Each element should be multiplied by 3
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo((i + 1) * 3.0, 4);
    }

    // Verify getConfig returns default
    const config = autoDispatcher.getConfig();
    expect(config.wgSize).toBe(64);

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("auto dispatcher tune() changes config", async () => {
    const N = 4096;
    const inputData = new Float32Array(N);
    for (let i = 0; i < N; i++) inputData[i] = 1.0;

    const inputBuf = makeF32Buffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuf = makeOutputBuffer(N);

    clearTileAutotuneCache();
    const autoDispatcher = createAutoTileKernelDispatcher(scaleAutoConfig);

    // Tune
    await autoDispatcher.tune(
      { input: inputBuf, output: outputBuf },
      { N, scale: 1.0 },
      { warmupIters: 1, timingIters: 2 },
    );

    // Config should be a valid option (may be any of 32, 64, 128, 256)
    const config = autoDispatcher.getConfig();
    expect([32, 64, 128, 256]).toContain(config.wgSize);

    // Dispatch with tuned config still produces correct results
    beginSharedEncoder();
    autoDispatcher.dispatch(
      { input: inputBuf, output: outputBuf },
      { N, scale: 5.0 },
    );
    flushSharedEncoder();
    await syncWebGPU();

    const result = await readF32Buffer(outputBuf, N);
    for (let i = 0; i < N; i++) {
      expect(result[i]).toBeCloseTo(5.0, 4);
    }

    inputBuf.destroy();
    outputBuf.destroy();
  });

  it("reset() returns to default config", () => {
    const autoDispatcher = createAutoTileKernelDispatcher(scaleAutoConfig);
    autoDispatcher.reset();
    expect(autoDispatcher.getConfig().wgSize).toBe(64);
  });
});
