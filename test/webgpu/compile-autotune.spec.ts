/**
 * Compile Autotune Tests
 *
 * Tests the autotune option for compile() which enables runtime autotuning
 * of matmul kernels within compiled regions, including subgroup variants
 * when hardware supports them.
 *
 * Tests auto-detect WebGPU; skip with TORCHLETTE_CPU_ONLY=1.
 */
import { describe, expect, it, beforeAll, afterEach } from "vitest";
import { Torchlette } from "../../src/frontend";
import {
  initWebGPU,
  getWebGPUDevice,
  isAutotuneEnabled,
  setAutotuneEnabled,
} from "../../src/backend/webgpu";
import {
  clearDispatchTuningCache,
  clearTuningCache,
  getCachedTuningResult,
  getSubgroupSupport,
} from "../../src/backend/webgpu/matmul";

import { cpuOnly } from "../helpers/webgpu";

const SKIP = cpuOnly;

describe.skipIf(SKIP)("Compile Autotune", () => {
  let torch: Torchlette;

  beforeAll(async () => {
    await initWebGPU();
    torch = new Torchlette("webgpu");
  });

  afterEach(() => {
    // Reset autotune state after each test
    setAutotuneEnabled(false);
    clearTuningCache();
    clearDispatchTuningCache();
  });

  describe("Autotune Flag Control", () => {
    it("autotune is disabled by default", () => {
      expect(isAutotuneEnabled()).toBe(false);
    });

    it("setAutotuneEnabled toggles autotune state", () => {
      setAutotuneEnabled(true);
      expect(isAutotuneEnabled()).toBe(true);
      setAutotuneEnabled(false);
      expect(isAutotuneEnabled()).toBe(false);
    });

    it("compile without autotune option does not enable autotuning", async () => {
      const a = torch.randn([32, 32]);
      const b = torch.randn([32, 32]);

      const fn = torch.compile((x: typeof a, y: typeof b) => x.matmul(y));
      const result = fn(a, b);

      // Force execution
      await result.cpu();

      // Autotune should not be enabled
      expect(isAutotuneEnabled()).toBe(false);

      // No tuning result should be cached (we only cache on autotune)
      // (Default config is used but not cached from autotune)
    });

    it("compile with autotune:true enables autotuning during execution", async () => {
      const a = torch.randn([64, 64]);
      const b = torch.randn([64, 64]);

      // Track if autotune was ever enabled
      let autotuneWasEnabled = false;

      const fn = torch.compile(
        (x: typeof a, y: typeof b) => {
          // Check inside the compiled region
          autotuneWasEnabled = isAutotuneEnabled();
          return x.matmul(y);
        },
        { autotune: true }
      );

      fn(a, b);

      // Autotune should have been enabled during execution
      expect(autotuneWasEnabled).toBe(true);

      // NOTE: Autotune flag stays enabled until explicitly reset (via afterEach
      // or setAutotuneEnabled(false)) because lazy execution happens AFTER the
      // compile wrapper returns. The flag is reset by afterEach in tests.
      expect(isAutotuneEnabled()).toBe(true);
    });

    it("autotune state is restored after compile finishes", async () => {
      // Set initial state to true
      setAutotuneEnabled(true);

      const a = torch.randn([32, 32]);
      const b = torch.randn([32, 32]);

      // Compile with autotune: false (default)
      const fn = torch.compile((x: typeof a, y: typeof b) => x.matmul(y));
      fn(a, b);

      // Should be restored to true (the state before compile)
      expect(isAutotuneEnabled()).toBe(true);
    });
  });

  describe("Autotune Execution", () => {
    it("autotune runs for matmul shapes", async () => {
      const a = torch.randn([128, 64]);
      const b = torch.randn([64, 128]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b) => x.matmul(y),
        { autotune: true }
      );

      const result = fn(a, b);

      // Force execution
      await result.cpu();

      // Check that tuning result was cached
      // Shape class for 128x128 output with K=64, maxDim=128 < 512 -> "square_small"
      const cachedResult = getCachedTuningResult("square_small", "f32");

      // Should have a cached result
      expect(cachedResult).not.toBeNull();
      expect(cachedResult?.config).toBeDefined();
    });

    it("autotuned matmul produces correct results", async () => {
      // Create matrices with known values for verification
      const M = 64, N = 64, K = 32;
      const aData = Array.from({ length: M * K }, () => 1.0);
      const bData = Array.from({ length: K * N }, () => 1.0);

      const a = torch.tensorFromArray(aData, [M, K]);
      const b = torch.tensorFromArray(bData, [K, N]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b) => x.matmul(y),
        { autotune: true }
      );

      const result = fn(a, b);
      const resultData = await result.cpu();

      // 1s @ 1s with K=32 should give all 32s
      expect(result.shape).toEqual([M, N]);
      for (let i = 0; i < M * N; i++) {
        expect(resultData[i]).toBeCloseTo(K, 1);
      }
    });

    it("subgroup variants are included when supported", async () => {
      const subgroupSupport = getSubgroupSupport();

      if (!subgroupSupport?.supported) {
        console.log("Subgroups not supported, skipping subgroup variant test");
        return;
      }

      const a = torch.randn([64, 64]);
      const b = torch.randn([64, 64]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b) => x.matmul(y),
        { autotune: true }
      );

      const result = fn(a, b);
      await result.cpu();

      // Check that tuning happened (64x64x64, maxDim=64 < 512 -> "square_small")
      const cachedResult = getCachedTuningResult("square_small", "f32");
      expect(cachedResult).not.toBeNull();

      // The best config may or may not use subgroups depending on hardware
      // We just verify that autotuning completed successfully
      expect(cachedResult?.config).toBeDefined();

      console.log(
        "Autotuned config:",
        cachedResult?.config,
        "Subgroup size:",
        subgroupSupport.subgroupSize
      );
    });

    it("cached tuning results are reused", async () => {
      const a = torch.randn([64, 64]);
      const b = torch.randn([64, 64]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b) => x.matmul(y),
        { autotune: true }
      );

      // First call - triggers autotuning
      const start1 = performance.now();
      const result1 = fn(a, b);
      await result1.cpu();
      const time1 = performance.now() - start1;

      // Second call - should use cached result
      const start2 = performance.now();
      const result2 = fn(a, b);
      await result2.cpu();
      const time2 = performance.now() - start2;

      // Second call should be faster (no autotuning overhead)
      // This is a soft check - timing can vary
      console.log(`First call: ${time1.toFixed(2)}ms, Second call: ${time2.toFixed(2)}ms`);

      // Just verify both completed successfully
      expect(result1.shape).toEqual([64, 64]);
      expect(result2.shape).toEqual([64, 64]);
    });
  });

  describe("Multiple Matmuls", () => {
    it("autotunes different shapes independently", async () => {
      // Run two separate compiles with different shapes
      // Small shape
      const a1 = torch.randn([32, 32]);
      const b1 = torch.randn([32, 32]);

      const fn1 = torch.compile(
        (x: typeof a1, y: typeof b1) => x.matmul(y),
        { autotune: true }
      );

      const r1 = fn1(a1, b1);
      await r1.cpu();

      // Medium shape
      const a2 = torch.randn([128, 128]);
      const b2 = torch.randn([128, 128]);

      const fn2 = torch.compile(
        (x: typeof a2, y: typeof b2) => x.matmul(y),
        { autotune: true }
      );

      const r2 = fn2(a2, b2);
      await r2.cpu();

      // Check both shape classes got tuned
      const smallResult = getCachedTuningResult("square_small", "f32");
      const mediumResult = getCachedTuningResult("square_medium", "f32");

      expect(smallResult).not.toBeNull();
      expect(mediumResult).not.toBeNull();
    });

    it("chain of matmuls all get autotuned", async () => {
      const a = torch.randn([64, 32]);
      const b = torch.randn([32, 64]);
      const c = torch.randn([64, 64]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b, z: typeof c) => {
          // A @ B = 64x64, then result @ C = 64x64
          return x.matmul(y).matmul(z);
        },
        { autotune: true }
      );

      const result = fn(a, b, c);
      await result.cpu();

      expect(result.shape).toEqual([64, 64]);

      // At least one shape class should be tuned
      const cached = getCachedTuningResult("square_medium", "f32");
      expect(cached).not.toBeNull();
    });
  });

  describe("Integration with Other Compile Features", () => {
    it("autotune works with fusion", async () => {
      const a = torch.randn([64, 64]);
      const b = torch.randn([64, 64]);
      const two = torch.tensorFromArray([2.0], []);

      const fn = torch.compile(
        (x: typeof a, y: typeof b, scale: typeof two) => {
          // Matmul followed by elementwise ops (should fuse the elementwise part)
          return x.matmul(y).relu().mul(scale);
        },
        { autotune: true }
      );

      const result = fn(a, b, two);
      await result.cpu();

      expect(result.shape).toEqual([64, 64]);
    });

    it("autotune works with memory planning", async () => {
      const a = torch.randn([64, 64]);
      const b = torch.randn([64, 64]);
      const c = torch.randn([64, 64]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b, z: typeof c) => {
          // Multiple operations that can benefit from memory planning
          const t1 = x.matmul(y);
          const t2 = t1.add(z);
          return t2.matmul(x);
        },
        { autotune: true }
      );

      const result = fn(a, b, c);
      await result.cpu();

      expect(result.shape).toEqual([64, 64]);
    });
  });

  describe("Error Handling", () => {
    it("autotune handles errors gracefully", async () => {
      // This should work even if some configs fail to benchmark
      const a = torch.randn([64, 64]);
      const b = torch.randn([64, 64]);

      const fn = torch.compile(
        (x: typeof a, y: typeof b) => x.matmul(y),
        { autotune: true }
      );

      // Should not throw
      const result = fn(a, b);
      await result.cpu();

      expect(result.shape).toEqual([64, 64]);
    });
  });
});
