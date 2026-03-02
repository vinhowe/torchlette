/**
 * Tile-IR Autotuning Framework tests.
 *
 * Tests config generation (pure logic, no GPU) and autotuning dispatch (GPU).
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  beginSharedEncoder,
  flushSharedEncoder,
  getWebGPUDevice,
  initWebGPU,
  syncWebGPU,
} from "../../src/backend/webgpu";
import type {
  GPUBuffer,
  GPUDevice,
  GPUQueue,
} from "../../src/backend/webgpu/gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../../src/backend/webgpu/gpu-types";
import {
  autotuneTileKernel,
  clearTileAutotuneCache,
  exportTileAutotuneCache,
  generateTileConfigs,
  getDefaultConfig,
  importTileAutotuneCache,
} from "../../src/backend/webgpu/tile-autotune";
import { createAutoTileKernelDispatcher } from "../../src/backend/webgpu/tile-dispatch";
import type {
  AutotuneConfig,
  TileKernelSpec,
} from "../../src/backend/webgpu/tile-ir";
import {
  ceilDivGrid,
  elementwiseGrid,
  inferGrid,
  perRowGrid,
  productGrid,
  resolveGrid,
  singleWorkgroup,
  tiledGrid,
} from "../../src/backend/webgpu/tile-ir";

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

async function readF32Buffer(
  buf: GPUBuffer,
  count: number,
): Promise<Float32Array> {
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
      input: { storage: "read", type: "f32" },
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
    expect(configs.map((c) => c.wgSize)).toEqual([32, 64, 128]);
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
    expect(configs.every((c) => c.wgSize <= 256)).toBe(true);
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
      constraints: [(c) => c.wgSize * c.vecWidth <= 256],
    };

    const configs = generateTileConfigs(autoConfig);
    // 32*1=32, 32*2=64, 32*4=128, 64*1=64, 64*2=128, 64*4=256, 128*1=128, 128*2=256, 128*4=512 (excluded)
    expect(configs).toHaveLength(8);
    expect(configs.every((c) => c.wgSize * c.vecWidth <= 256)).toBe(true);
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

// ============================================================================
// Grid helper tests (no GPU needed)
// ============================================================================

describe("tile-ir grid helpers", () => {
  describe("elementwiseGrid", () => {
    it("returns 1D grid for small element counts", () => {
      const grid = elementwiseGrid(256, { elementUniform: "size" });
      expect(grid({ size: 1024 })).toEqual([4]);
    });

    it("returns 2D grid when workgroups exceed 65535", () => {
      const grid = elementwiseGrid(256, { elementUniform: "size" });
      // 256 * 70000 = 17.92M elements → 70000 workgroups > 65535
      const result = grid({ size: 256 * 70000 });
      expect(result).toHaveLength(2);
      expect(result[0]).toBe(65535);
      expect(result[1]).toBe(Math.ceil(70000 / 65535));
    });

    it("accounts for vectorization width", () => {
      const grid = elementwiseGrid(256, { vecWidth: 4, elementUniform: "n" });
      // 4096 / (256 * 4) = 4 workgroups
      expect(grid({ n: 4096 })).toEqual([4]);
    });

    it("defaults to total_elements uniform", () => {
      const grid = elementwiseGrid(256);
      expect(grid({ total_elements: 512 })).toEqual([2]);
    });
  });

  describe("perRowGrid", () => {
    it("dispatches one workgroup per row", () => {
      const grid = perRowGrid("num_rows");
      expect(grid({ num_rows: 128 })).toEqual([128]);
    });

    it("defaults to num_rows uniform", () => {
      const grid = perRowGrid();
      expect(grid({ num_rows: 64 })).toEqual([64]);
    });
  });

  describe("ceilDivGrid", () => {
    it("ceil-divides element count by divisor", () => {
      const grid = ceilDivGrid(256, "n");
      expect(grid({ n: 1000 })).toEqual([4]); // ceil(1000/256) = 4
    });
  });

  describe("singleWorkgroup", () => {
    it("always returns [1]", () => {
      const grid = singleWorkgroup();
      expect(grid({})).toEqual([1]);
      expect(grid({ anything: 999 })).toEqual([1]);
    });
  });

  describe("tiledGrid", () => {
    it("creates 1D tiled grid", () => {
      const grid = tiledGrid({ x: { uniform: "N", tileSize: 32 } });
      expect(grid({ N: 128 })).toEqual([4]); // ceil(128/32)
    });

    it("creates 2D tiled grid", () => {
      const grid = tiledGrid({
        x: { uniform: "N", tileSize: 32 },
        y: "num_heads",
      });
      expect(grid({ N: 128, num_heads: 12 })).toEqual([4, 12]);
    });

    it("creates 3D tiled grid (attention pattern)", () => {
      const grid = tiledGrid({
        x: { uniform: "seq_len", tileSize: 64 },
        y: "num_heads",
        z: "batch_size",
      });
      expect(grid({ seq_len: 512, num_heads: 12, batch_size: 2 })).toEqual([
        8, 12, 2,
      ]);
    });

    it("handles non-divisible tile sizes", () => {
      const grid = tiledGrid({ x: { uniform: "N", tileSize: 64 } });
      expect(grid({ N: 100 })).toEqual([2]); // ceil(100/64)
    });

    it("supports raw uniform dims (tileSize=1)", () => {
      const grid = tiledGrid({ x: "count" });
      expect(grid({ count: 42 })).toEqual([42]);
    });
  });

  describe("productGrid", () => {
    it("multiplies uniform values", () => {
      const grid = productGrid("batch", "heads", "seq");
      expect(grid({ batch: 2, heads: 12, seq: 512 })).toEqual([12288]);
    });

    it("handles single uniform", () => {
      const grid = productGrid("n");
      expect(grid({ n: 100 })).toEqual([100]);
    });
  });

  describe("inferGrid", () => {
    it("infers elementwise grid from 'size' uniform", () => {
      const spec: TileKernelSpec = {
        name: "test",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { size: "u32" },
        kernel() {},
      };
      const grid = inferGrid(spec);
      expect(grid).not.toBeNull();
      expect(grid?.({ size: 1024 })).toEqual([4]);
    });

    it("infers from 'outSize' uniform", () => {
      const spec: TileKernelSpec = {
        name: "test",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { outSize: "u32", reductionSize: "u32" },
        kernel() {},
      };
      const grid = inferGrid(spec);
      expect(grid).not.toBeNull();
      expect(grid?.({ outSize: 512 })).toEqual([2]);
    });

    it("returns null when no matching uniform", () => {
      const spec: TileKernelSpec = {
        name: "test",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { rows: "u32", cols: "u32" },
        kernel() {},
      };
      expect(inferGrid(spec)).toBeNull();
    });

    it("ignores f32 uniforms named 'size'", () => {
      const spec: TileKernelSpec = {
        name: "test",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { size: "f32" },
        kernel() {},
      };
      expect(inferGrid(spec)).toBeNull();
    });
  });

  describe("resolveGrid", () => {
    it("returns explicit grid when provided", () => {
      const explicitGrid = singleWorkgroup();
      const spec: TileKernelSpec = {
        name: "test",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { size: "u32" },
        grid: explicitGrid,
        kernel() {},
      };
      expect(resolveGrid(spec)).toBe(explicitGrid);
    });

    it("falls back to inference when grid omitted", () => {
      const spec: TileKernelSpec = {
        name: "test",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { size: "u32" },
        kernel() {},
      };
      const grid = resolveGrid(spec);
      expect(grid({ size: 1024 })).toEqual([4]);
    });

    it("throws when grid omitted and no inference matches", () => {
      const spec: TileKernelSpec = {
        name: "mykernel",
        workgroupSize: 256,
        bindings: { out: { storage: "read_write", type: "f32" } },
        uniforms: { rows: "u32" },
        kernel() {},
      };
      expect(() => resolveGrid(spec)).toThrow(
        /no grid function for kernel "mykernel"/,
      );
    });
  });
});
