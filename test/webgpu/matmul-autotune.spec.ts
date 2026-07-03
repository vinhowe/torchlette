/**
 * Matmul Autotuning Tests
 *
 * Tests the autotuning infrastructure for finding optimal kernel configurations.
 * These tests exercise:
 * - Config generation and validation
 * - Shape classification
 * - Tuning cache operations
 * - Full autotune with mock benchmarks
 */
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { getWebGPUDevice, initWebGPU } from "../../src/backend/webgpu";
import {
  autotune,
  type BenchmarkFn,
  cacheTuningResult,
  clearTuningCache,
  exportTuningCache,
  generateTuningConfigs,
  getCachedTuningResult,
  getDefaultConfigForShape,
  importTuningCache,
  quickAutotune,
} from "../../src/backend/webgpu/matmul/autotune";
import {
  clearPerShapeMatmulChoices,
  dispatchTiledMatmul,
  getAutotuneRunCount,
  getPerShapeMatmulChoice,
  planTiledMatmul,
  pretuneMatmulShapes,
  seedPerShapeMatmulChoice,
  setAutotuneEnabled,
} from "../../src/backend/webgpu/matmul/dispatch";
import { GEMV_WG_SIZE_PARAM } from "../../src/backend/webgpu/matmul/gemv";
import {
  classifyShape,
  DEFAULT_CONFIG,
  type ShapeClass,
  TUNING_SPACE,
  type TuneResult,
  validateConfig,
} from "../../src/backend/webgpu/matmul/types";
import {
  getMatmulVariant,
  MATMUL_VARIANTS,
  type MatmulVariantContext,
} from "../../src/backend/webgpu/matmul/variants";

import { cpuOnly } from "../helpers/webgpu";

const SKIP = cpuOnly;

describe.skipIf(SKIP)("Matmul Autotuning", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  beforeEach(() => {
    clearTuningCache();
  });

  afterEach(() => {
    clearTuningCache();
  });

  describe("generateTuningConfigs", () => {
    it("generates configs without subgroups", () => {
      const configs = generateTuningConfigs(false);

      expect(configs.length).toBeGreaterThan(0);

      // All configs should have useSubgroups = false
      for (const config of configs) {
        expect(config.useSubgroups).toBe(false);
      }

      // Check that configs are valid
      for (const config of configs) {
        expect(() => validateConfig(config)).not.toThrow();
      }
    });

    it("generates configs with subgroups when enabled", () => {
      const configs = generateTuningConfigs(true);

      expect(configs.length).toBeGreaterThan(0);

      // Should have mix of subgroup and non-subgroup configs
      const withSubgroups = configs.filter((c) => c.useSubgroups);
      const withoutSubgroups = configs.filter((c) => !c.useSubgroups);

      expect(withSubgroups.length).toBeGreaterThan(0);
      expect(withoutSubgroups.length).toBeGreaterThan(0);
    });

    it("generates more configs with subgroups than without", () => {
      const withoutSub = generateTuningConfigs(false);
      const withSub = generateTuningConfigs(true);

      // With subgroups should have roughly 2x configs
      expect(withSub.length).toBeGreaterThan(withoutSub.length);
    });

    it("all generated configs have valid structure", () => {
      const configs = generateTuningConfigs(true);

      for (const config of configs) {
        expect(config).toHaveProperty("tileM");
        expect(config).toHaveProperty("tileN");
        expect(config).toHaveProperty("tileK");
        expect(config).toHaveProperty("threadTileM");
        expect(config).toHaveProperty("threadTileN");
        expect(config).toHaveProperty("vectorWidth");
        expect(config).toHaveProperty("useSubgroups");

        // Values should be from tuning space
        expect(TUNING_SPACE.tileM).toContain(config.tileM);
        expect(TUNING_SPACE.tileN).toContain(config.tileN);
        expect(TUNING_SPACE.tileK).toContain(config.tileK);
        expect(TUNING_SPACE.threadTileM).toContain(config.threadTileM);
        expect(TUNING_SPACE.threadTileN).toContain(config.threadTileN);
        expect(TUNING_SPACE.vectorWidth).toContain(config.vectorWidth);
      }
    });
  });

  describe("classifyShape", () => {
    it("classifies GEMV shapes", () => {
      // M=1 (row-vector × matrix) has its own class: full-thread config +
      // K-split occupancy — decode steps, probes. N=1 keeps "gemv".
      expect(classifyShape(1, 100, 100, 1)).toBe("gemv_row");
      expect(classifyShape(100, 1, 100, 1)).toBe("gemv");
    });

    it("classifies small square matrices", () => {
      expect(classifyShape(256, 256, 256, 1)).toBe("square_small");
      expect(classifyShape(128, 128, 128, 1)).toBe("square_small");
    });

    it("classifies medium square matrices", () => {
      expect(classifyShape(1024, 1024, 1024, 1)).toBe("square_medium");
      expect(classifyShape(512, 512, 512, 1)).toBe("square_medium");
    });

    it("classifies large square matrices", () => {
      expect(classifyShape(2048, 2048, 2048, 1)).toBe("square_large");
      expect(classifyShape(4096, 4096, 4096, 1)).toBe("square_large");
    });

    it("classifies tall-skinny matrices", () => {
      expect(classifyShape(4096, 256, 256, 1)).toBe("tall_skinny");
      expect(classifyShape(8192, 512, 512, 1)).toBe("tall_skinny");
    });

    it("classifies short-wide matrices", () => {
      expect(classifyShape(256, 4096, 256, 1)).toBe("short_wide");
      expect(classifyShape(512, 8192, 512, 1)).toBe("short_wide");
    });

    it("classifies large-K matrices", () => {
      expect(classifyShape(512, 768, 50304, 1)).toBe("large_k");
      expect(classifyShape(256, 256, 4096, 1)).toBe("large_k");
    });

    it("classifies batched small matrices", () => {
      expect(classifyShape(64, 64, 64, 8)).toBe("batched_small");
      expect(classifyShape(128, 128, 128, 4)).toBe("batched_small");
    });
  });

  describe("getDefaultConfigForShape", () => {
    const shapeClasses: ShapeClass[] = [
      "square_small",
      "square_medium",
      "square_large",
      "tall_skinny",
      "short_wide",
      "large_k",
      "gemv",
      "batched_small",
    ];

    for (const shapeClass of shapeClasses) {
      it(`returns valid config for ${shapeClass}`, () => {
        const config = getDefaultConfigForShape(shapeClass);

        expect(() => validateConfig(config)).not.toThrow();
        expect(config.tileM).toBeGreaterThan(0);
        expect(config.tileN).toBeGreaterThan(0);
        expect(config.tileK).toBeGreaterThan(0);
      });
    }

    it("square_large uses larger tiles", () => {
      const large = getDefaultConfigForShape("square_large");
      const small = getDefaultConfigForShape("square_small");

      expect(large.tileM).toBeGreaterThanOrEqual(small.tileM);
      expect(large.tileN).toBeGreaterThanOrEqual(small.tileN);
    });

    it("tall_skinny uses large tiles for arithmetic intensity", () => {
      const tallSkinny = getDefaultConfigForShape("tall_skinny");

      // Large tiles (64×128) maximize arithmetic intensity for tall matrices
      // with moderate K (e.g. lm_head backward dW: M=50304, N=768, K=512)
      expect(tallSkinny.tileM).toBeGreaterThanOrEqual(64);
      expect(tallSkinny.tileN).toBeGreaterThanOrEqual(64);
    });

    it("short_wide favors N dimension", () => {
      const shortWide = getDefaultConfigForShape("short_wide");

      expect(shortWide.tileN).toBeGreaterThanOrEqual(shortWide.tileM);
    });

    it("large_k uses maximum tile size for arithmetic intensity", () => {
      const largeK = getDefaultConfigForShape("large_k");

      // 128×128 tiles maximize compute-to-memory ratio for large K
      expect(largeK.tileM).toBe(128);
      expect(largeK.tileN).toBe(128);
    });

    it("square_large bare uses larger thread tiles (t8x4) with tileK=16", () => {
      const config = getDefaultConfigForShape("square_large", false);
      expect(config.tileM).toBe(64);
      expect(config.tileN).toBe(128);
      expect(config.tileK).toBe(16);
      expect(config.threadTileM).toBe(8);
      expect(config.threadTileN).toBe(4);
    });

    it("square_large epilogue uses smaller thread tiles (t4x4) with tileK=16", () => {
      const config = getDefaultConfigForShape("square_large", true);
      expect(config.tileM).toBe(64);
      expect(config.tileN).toBe(64);
      expect(config.tileK).toBe(16);
      expect(config.threadTileM).toBe(4);
      expect(config.threadTileN).toBe(4);
    });

    it("square_medium bare uses 64x64x16 config", () => {
      const config = getDefaultConfigForShape("square_medium", false);
      expect(config.tileM).toBe(64);
      expect(config.tileN).toBe(64);
      expect(config.tileK).toBe(16);
    });

    it("square_medium epilogue uses 64x64x16 config", () => {
      const config = getDefaultConfigForShape("square_medium", true);
      expect(config.tileM).toBe(64);
      expect(config.tileN).toBe(64);
      expect(config.tileK).toBe(16);
    });

    it("tall_skinny and short_wide have epilogue-specific configs", () => {
      for (const sc of ["tall_skinny", "short_wide"] as ShapeClass[]) {
        const bare = getDefaultConfigForShape(sc, false);
        const epilogue = getDefaultConfigForShape(sc, true);
        // Epilogue: conservative t4x4 to avoid register pressure
        expect(epilogue.threadTileM).toBe(4);
        expect(epilogue.threadTileN).toBe(4);
        // Bare: larger thread tiles for better register reuse
        expect(bare.threadTileM).toBe(8);
        expect(bare.threadTileN).toBe(4);
      }
    });

    it("other shape classes return same config regardless of hasEpilogue", () => {
      const otherClasses: ShapeClass[] = [
        "large_k",
        "gemv",
        "batched_small",
        "square_small",
      ];
      for (const sc of otherClasses) {
        const bare = getDefaultConfigForShape(sc, false);
        const epilogue = getDefaultConfigForShape(sc, true);
        expect(bare).toEqual(epilogue);
      }
    });
  });

  describe("Tuning Cache", () => {
    it("caches and retrieves tuning results", () => {
      const result: TuneResult = {
        config: DEFAULT_CONFIG,
        gflopsPerSec: 100,
        medianMs: 1.0,
        shapeClass: "square_medium",
        dtype: "f32",
      };

      cacheTuningResult(result);

      const cached = getCachedTuningResult("square_medium", "f32");
      expect(cached).toEqual(result);
    });

    it("returns undefined for uncached results", () => {
      const cached = getCachedTuningResult("square_large", "f32");
      expect(cached).toBeUndefined();
    });

    it("clears cache correctly", () => {
      cacheTuningResult({
        config: DEFAULT_CONFIG,
        gflopsPerSec: 100,
        medianMs: 1.0,
        shapeClass: "square_medium",
        dtype: "f32",
      });

      expect(getCachedTuningResult("square_medium", "f32")).toBeDefined();

      clearTuningCache();

      expect(getCachedTuningResult("square_medium", "f32")).toBeUndefined();
    });

    it("exports and imports cache", () => {
      const result1: TuneResult = {
        config: DEFAULT_CONFIG,
        gflopsPerSec: 100,
        medianMs: 1.0,
        shapeClass: "square_medium",
        dtype: "f32",
      };
      const result2: TuneResult = {
        config: { ...DEFAULT_CONFIG, tileM: 64 },
        gflopsPerSec: 150,
        medianMs: 0.7,
        shapeClass: "square_large",
        dtype: "f32",
      };

      cacheTuningResult(result1);
      cacheTuningResult(result2);

      const exported = exportTuningCache();
      expect(typeof exported).toBe("string");

      clearTuningCache();
      expect(getCachedTuningResult("square_medium", "f32")).toBeUndefined();

      importTuningCache(exported);
      expect(getCachedTuningResult("square_medium", "f32")).toEqual(result1);
      expect(getCachedTuningResult("square_large", "f32")).toEqual(result2);
    });

    it("throws on invalid import JSON", () => {
      expect(() => importTuningCache("invalid json")).toThrow(
        "Invalid tuning cache JSON",
      );
    });

    it("separates f16 and f32 results", () => {
      const f32Result: TuneResult = {
        config: DEFAULT_CONFIG,
        gflopsPerSec: 100,
        medianMs: 1.0,
        shapeClass: "square_medium",
        dtype: "f32",
      };
      const f16Result: TuneResult = {
        config: { ...DEFAULT_CONFIG, tileM: 64 },
        gflopsPerSec: 200,
        medianMs: 0.5,
        shapeClass: "square_medium",
        dtype: "f16",
      };

      cacheTuningResult(f32Result);
      cacheTuningResult(f16Result);

      const cachedF32 = getCachedTuningResult("square_medium", "f32");
      const cachedF16 = getCachedTuningResult("square_medium", "f16");

      expect(cachedF32).toEqual(f32Result);
      expect(cachedF16).toEqual(f16Result);
      expect(cachedF32).not.toEqual(cachedF16);
    });
  });

  describe("autotune", () => {
    it("runs autotune with mock benchmark", async () => {
      // Mock benchmark that returns decreasing time for configs with larger tiles
      const mockBenchmark: BenchmarkFn = async (config, _warmup, _iters) => {
        // Larger tiles = better performance (lower time)
        const baseTime = 10.0;
        const tileBonus = (config.tileM + config.tileN) / 256;
        return baseTime / (1 + tileBonus);
      };

      const result = await autotune(mockBenchmark, 1024, 1024, 1024, "f32", {
        maxTrials: 5,
        warmupIters: 1,
        timingIters: 2,
      });

      expect(result).toHaveProperty("config");
      expect(result).toHaveProperty("gflopsPerSec");
      expect(result).toHaveProperty("medianMs");
      expect(result).toHaveProperty("shapeClass");
      expect(result).toHaveProperty("dtype");

      expect(result.dtype).toBe("f32");
      expect(result.gflopsPerSec).toBeGreaterThan(0);
    });

    it("uses cached result on second call", async () => {
      let callCount = 0;
      const mockBenchmark: BenchmarkFn = async () => {
        callCount++;
        return 5.0;
      };

      // First call - should benchmark
      await autotune(mockBenchmark, 512, 512, 512, "f32", { maxTrials: 2 });
      const firstCallCount = callCount;

      // Second call - should use cache
      await autotune(mockBenchmark, 512, 512, 512, "f32", { maxTrials: 2 });

      expect(callCount).toBe(firstCallCount); // No additional benchmark calls
    });

    it("handles benchmark failures gracefully", async () => {
      let failures = 0;
      const failingBenchmark: BenchmarkFn = async () => {
        failures++;
        if (failures < 3) {
          throw new Error("Simulated failure");
        }
        return 5.0;
      };

      const result = await autotune(failingBenchmark, 256, 256, 256, "f32", {
        maxTrials: 5,
      });

      // Should still return a result (either from successful benchmark or default)
      expect(result).toBeDefined();
      expect(result.config).toBeDefined();
    });

    it("returns default config when all benchmarks fail", async () => {
      const alwaysFails: BenchmarkFn = async () => {
        throw new Error("Always fails");
      };

      const result = await autotune(alwaysFails, 256, 256, 256, "f32", {
        maxTrials: 3,
      });

      expect(result.gflopsPerSec).toBe(0);
      expect(result.medianMs).toBe(Infinity);
    });
  });

  describe("quickAutotune", () => {
    it("runs quick autotune with fewer configs", async () => {
      let configsTested = 0;
      const countingBenchmark: BenchmarkFn = async () => {
        configsTested++;
        return 5.0;
      };

      await quickAutotune(countingBenchmark, 512, 512, 512, "f32");

      // Quick autotune should test fewer than full autotune
      expect(configsTested).toBeLessThanOrEqual(5);
      expect(configsTested).toBeGreaterThan(0);
    });

    it("returns best config from quick search", async () => {
      const mockBenchmark: BenchmarkFn = async (config) => {
        // Return best time for 64x64 tiles
        if (config.tileM === 64 && config.tileN === 64) {
          return 2.0;
        }
        return 5.0;
      };

      const result = await quickAutotune(
        mockBenchmark,
        1024,
        1024,
        1024,
        "f32",
      );

      expect(result.config.tileM).toBe(64);
      expect(result.config.tileN).toBe(64);
    });

    it("caches quick autotune results", async () => {
      let callCount = 0;
      const mockBenchmark: BenchmarkFn = async () => {
        callCount++;
        return 5.0;
      };

      await quickAutotune(mockBenchmark, 512, 512, 512, "f32");
      const _afterFirst = callCount;

      // Clear and run autotune - should use same cache key
      const cached = getCachedTuningResult("square_medium", "f32");
      expect(cached).toBeDefined();
    });
  });

  describe("Variant Registry", () => {
    const mkCtx = (
      over: Partial<MatmulVariantContext> = {},
    ): MatmulVariantContext => ({
      m: 1,
      n: 768,
      k: 768,
      batchSize: 1,
      dtypeA: "f32",
      dtypeB: "f32",
      transA: false,
      transB: true,
      hasEpilogue: false,
      epiloguePresent: false,
      hasInputCast: false,
      hasExplicitConfig: false,
      subgroupSupported: false,
      ...over,
    });

    afterEach(() => {
      clearPerShapeMatmulChoices();
      setAutotuneEnabled(false);
    });

    it("registry is ordered gemv-before-tiled and tiled is always applicable", () => {
      expect(MATMUL_VARIANTS.map((v) => v.name)).toEqual(["gemv", "tiled"]);
      expect(getMatmulVariant("tiled").isApplicable(mkCtx())).toBe(true);
      expect(
        getMatmulVariant("tiled").isApplicable(
          mkCtx({ m: 512, batchSize: 8, epiloguePresent: true }),
        ),
      ).toBe(true);
    });

    it("gemv applicability filters on geometry/epilogue/cast/explicit-config", () => {
      const gemv = getMatmulVariant("gemv");
      expect(gemv.isApplicable(mkCtx())).toBe(true);
      expect(gemv.isApplicable(mkCtx({ m: 2 }))).toBe(false);
      expect(gemv.isApplicable(mkCtx({ batchSize: 2 }))).toBe(false);
      expect(gemv.isApplicable(mkCtx({ epiloguePresent: true }))).toBe(false);
      expect(gemv.isApplicable(mkCtx({ hasInputCast: true }))).toBe(false);
      expect(gemv.isApplicable(mkCtx({ hasExplicitConfig: true }))).toBe(false);
      // Geometry the GEMV route itself rejects (NN small grid: gx*splitK < 16)
      expect(gemv.isApplicable(mkCtx({ transB: false, n: 128, k: 1024 }))).toBe(
        false,
      );
    });

    it("TORCHLETTE_GEMV=0 opt-out makes gemv inapplicable", () => {
      const gemv = getMatmulVariant("gemv");
      const saved = process.env.TORCHLETTE_GEMV;
      try {
        process.env.TORCHLETTE_GEMV = "0";
        expect(gemv.isApplicable(mkCtx())).toBe(false);
      } finally {
        if (saved === undefined) delete process.env.TORCHLETTE_GEMV;
        else process.env.TORCHLETTE_GEMV = saved;
      }
      expect(gemv.isApplicable(mkCtx())).toBe(true);
    });

    it("gemv candidates vary wgSize from the shared TuneParam", () => {
      const gemv = getMatmulVariant("gemv");
      const candidates = gemv.candidates(mkCtx());
      expect(candidates.length).toBeGreaterThan(1);
      const wgSizes = candidates.map((c) =>
        c.variant === "gemv" ? c.wgSize : -1,
      );
      expect(new Set(wgSizes).size).toBe(candidates.length);
      for (const w of wgSizes) {
        expect(GEMV_WG_SIZE_PARAM.values).toContain(w);
      }
      // default choice is included
      const def = gemv.defaultChoice(mkCtx());
      expect(def.variant).toBe("gemv");
      expect(wgSizes).toContain(def.variant === "gemv" ? def.wgSize : -1);
    });

    it("tiled candidates include the heuristic base and DEFAULT_CONFIG", () => {
      const tiled = getMatmulVariant("tiled");
      const ctx = mkCtx({ m: 512, n: 768, k: 768, transB: false });
      const base = tiled.defaultChoice(ctx);
      const candidates = tiled.candidates(ctx);
      expect(candidates.length).toBeGreaterThan(5);
      const key = (c: (typeof candidates)[number]) =>
        c.variant === "tiled" ? JSON.stringify(c.config) : "";
      expect(candidates.map(key)).toContain(key(base as never));
      expect(candidates.map(key)).toContain(JSON.stringify(DEFAULT_CONFIG));
      for (const c of candidates) {
        expect(c.variant).toBe("tiled");
        if (c.variant === "tiled") {
          expect(() => validateConfig(c.config)).not.toThrow();
        }
      }
    });

    it("seeded per-shape gemv winner short-circuits planning and its wgSize flows into dispatch dims", () => {
      const gpuContext = getWebGPUDevice();
      if (!gpuContext) return;
      const { device, queue } = gpuContext;
      // NN m=1 n=3072 k=768: wgSize=256 → gx=12,splitK=3; wgSize=128 → gx=24,splitK=6
      seedPerShapeMatmulChoice(1, 3072, 768, "f32", {
        variant: "gemv",
        wgSize: 128,
      });
      const plan = planTiledMatmul({
        device,
        queue,
        a: undefined as never,
        b: undefined as never,
        out: undefined as never,
        m: 1,
        n: 3072,
        k: 768,
      });
      expect(plan.label).toBe("_gemv");
      expect(plan.kSplit).toBe(true);
      if (plan.kSplit) {
        expect(plan.ksplitDispatch).toEqual([24, 6, 1]);
      }
      // Default (unseeded) choice uses wgSize=256 → different grid
      clearPerShapeMatmulChoices();
      const defPlan = planTiledMatmul({
        device,
        queue,
        a: undefined as never,
        b: undefined as never,
        out: undefined as never,
        m: 1,
        n: 3072,
        k: 768,
      });
      expect(defPlan.label).toBe("_gemv");
      if (defPlan.kSplit) {
        expect(defPlan.ksplitDispatch).toEqual([12, 3, 1]);
      }
    });

    it("seeded winner is re-gated on applicability (batched ctx falls back to tiled)", () => {
      const gpuContext = getWebGPUDevice();
      if (!gpuContext) return;
      const { device, queue } = gpuContext;
      seedPerShapeMatmulChoice(1, 3072, 768, "f32", {
        variant: "gemv",
        wgSize: 128,
      });
      const plan = planTiledMatmul({
        device,
        queue,
        a: undefined as never,
        b: undefined as never,
        out: undefined as never,
        m: 1,
        n: 3072,
        k: 768,
        batchSize: 4,
      });
      expect(plan.label).toBeUndefined(); // tiled, not _gemv
    });

    it("real GPU autotune searches across variants, caches (variant, choice), and second run hits the cache", async () => {
      const gpuContext = getWebGPUDevice();
      if (!gpuContext) return;
      const { device, queue } = gpuContext;

      setAutotuneEnabled(true);
      const runsBefore = getAutotuneRunCount();
      // m=1 shape where BOTH variants apply, plus a tiled-only shape
      await pretuneMatmulShapes(
        device,
        queue,
        [
          [1, 4096, 512],
          [256, 256, 256],
        ],
        "f32",
      );
      expect(getAutotuneRunCount()).toBe(runsBefore + 2);

      const gemvShapeChoice = getPerShapeMatmulChoice(1, 4096, 512, "f32");
      expect(gemvShapeChoice).toBeDefined();
      expect(["gemv", "tiled"]).toContain(gemvShapeChoice?.variant);
      const tiledShapeChoice = getPerShapeMatmulChoice(256, 256, 256, "f32");
      expect(tiledShapeChoice?.variant).toBe("tiled");

      // Second run of the same shapes: pure cache hits, no new searches
      await pretuneMatmulShapes(
        device,
        queue,
        [
          [1, 4096, 512],
          [256, 256, 256],
        ],
        "f32",
      );
      expect(getAutotuneRunCount()).toBe(runsBefore + 2);

      // The cached winner short-circuits planning without error
      const plan = planTiledMatmul({
        device,
        queue,
        a: undefined as never,
        b: undefined as never,
        out: undefined as never,
        m: 1,
        n: 4096,
        k: 512,
      });
      if (gemvShapeChoice?.variant === "gemv") {
        expect(plan.label).toBe("_gemv");
      }
    }, 60000);
  });

  describe("Real GPU Autotune", () => {
    it("autotunes real matmul kernel", async () => {
      const gpuContext = getWebGPUDevice();
      if (!gpuContext) {
        return; // Skip if no GPU
      }

      const { device, queue } = gpuContext;
      const M = 256;
      const N = 256;
      const K = 256;

      // Create test matrices
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

      // Create reusable output buffer
      const outBuffer = device.createBuffer({
        size: M * N * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });

      // Real benchmark function using actual matmul dispatch
      const realBenchmark: BenchmarkFn = async (config, warmup, iters) => {
        // Warmup
        for (let i = 0; i < warmup; i++) {
          dispatchTiledMatmul({
            device,
            queue,
            a: aBuffer,
            b: bBuffer,
            out: outBuffer,
            m: M,
            n: N,
            k: K,
            config,
            dtype: "f32",
          });
        }
        await device.queue.onSubmittedWorkDone();

        // Timing
        const start = performance.now();
        for (let i = 0; i < iters; i++) {
          dispatchTiledMatmul({
            device,
            queue,
            a: aBuffer,
            b: bBuffer,
            out: outBuffer,
            m: M,
            n: N,
            k: K,
            config,
            dtype: "f32",
          });
        }
        await device.queue.onSubmittedWorkDone();
        const end = performance.now();

        return (end - start) / iters;
      };

      const result = await autotune(realBenchmark, M, N, K, "f32", {
        maxTrials: 3,
        warmupIters: 1,
        timingIters: 2,
      });

      expect(result.gflopsPerSec).toBeGreaterThan(0);
      expect(result.medianMs).toBeGreaterThan(0);
      expect(result.medianMs).toBeLessThan(Infinity);

      // Cleanup
      aBuffer.destroy();
      bBuffer.destroy();
      outBuffer.destroy();
    });
  });
});
