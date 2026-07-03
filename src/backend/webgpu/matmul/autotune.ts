/**
 * Autotuning infrastructure for matmul kernels.
 *
 * Provides runtime autotuning to find the best kernel configuration
 * for different matrix shapes and dtypes on the current GPU.
 */

import type {
  DType,
  MatmulKernelConfig,
  ShapeClass,
  TuneResult,
} from "./types";
import {
  classifyShape,
  DEFAULT_CONFIG,
  TUNING_SPACE,
  validateConfig,
} from "./types";

/**
 * Autotuner options.
 */
type AutotuneOptions = {
  /** Maximum number of configs to try (default: 27) */
  maxTrials?: number;
  /** Warmup iterations before timing (default: 3) */
  warmupIters?: number;
  /** Timing iterations (default: 5) */
  timingIters?: number;
  /** Timeout per trial in ms (default: 5000) */
  trialTimeoutMs?: number;
};

/**
 * Benchmark function type passed to autotuner.
 * Returns execution time in milliseconds for the given config.
 */
export type BenchmarkFn = (
  config: MatmulKernelConfig,
  warmup: number,
  iters: number,
) => Promise<number>;

/**
 * Generate all valid configurations from the tuning space.
 */
export function generateTuningConfigs(
  includeSubgroups: boolean,
): MatmulKernelConfig[] {
  const configs: MatmulKernelConfig[] = [];

  for (const tileM of TUNING_SPACE.tileM) {
    for (const tileN of TUNING_SPACE.tileN) {
      for (const tileK of TUNING_SPACE.tileK) {
        for (const threadTileM of TUNING_SPACE.threadTileM) {
          for (const threadTileN of TUNING_SPACE.threadTileN) {
            for (const vectorWidth of TUNING_SPACE.vectorWidth) {
              // Skip subgroups if not supported
              const subgroupOptions = includeSubgroups
                ? TUNING_SPACE.useSubgroups
                : [false as const];

              for (const useSubgroups of subgroupOptions) {
                const config: MatmulKernelConfig = {
                  tileM,
                  tileN,
                  tileK,
                  threadTileM,
                  threadTileN,
                  vectorWidth,
                  useSubgroups,
                };

                // Validate config (skip invalid ones)
                try {
                  validateConfig(config);
                  configs.push(config);
                } catch {
                  // Skip invalid configs
                }
              }
            }
          }
        }
      }
    }
  }

  return configs;
}

/**
 * Get a sensible default config for a shape class.
 * These are educated guesses before autotuning.
 */
export function getDefaultConfigForShape(
  shapeClass: ShapeClass,
  hasEpilogue: boolean = false,
  m?: number,
  n?: number,
  k?: number,
): MatmulKernelConfig {
  switch (shapeClass) {
    case "square_large":
      if (hasEpilogue) {
        // Epilogue matmuls: t4x4 avoids register pressure from extra per-element ops
        // tileK=16 halves K-loop iterations while keeping 8KB smem budget
        // Benchmarked: 64x64 t4x4 is +50% faster than 64x128 t8x4 with epilogue
        return {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 16,
        };
      }
      // Bare matmuls with small M and large K (backward dX shapes like 512×1024×3072):
      // 64×64×16 t4×4 is 29-31% faster — higher occupancy from smaller tiles wins
      // when the output grid is modest but K-loop is long.
      // Tall shapes (dW like 3072×1024×512) still prefer 64×128×16 t8×4.
      if (m !== undefined && k !== undefined && m <= 512 && k >= m * 2) {
        return {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 16,
        };
      }
      // Default bare: larger thread tiles (t8x4) give better register reuse
      // tileK=16 halves K-loop iterations and barriers for large-K backward matmuls
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 128,
        tileK: 16,
        threadTileM: 8,
        threadTileN: 4,
      };

    case "square_medium":
      if (hasEpilogue) {
        // Epilogue matmuls: 64x64x16 t4x4 outperforms 32x32x16 by 30% on 1024-embed shapes
        // (e.g. attn.cProj fwd 512x1024x1024: 0.32ms vs 0.45ms). The epilogue overhead
        // from cast+bias is small enough that larger tiles win via better throughput.
        return {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 16,
        };
      }
      // Bare matmuls: step up from 32x32x16 for better throughput
      // tileK=16 halves K-loop iterations; same shared memory as 64x64x8
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 64,
        tileK: 16,
      };

    case "large_k":
      // Large K dimension (e.g. lm_head backward dX: M=512, N=768, K=50304)
      // tileK=8 halves shared memory (4KB vs 8KB), doubling GPU occupancy.
      // K-split factor stays the same; occupancy gain outweighs 2× K-loop iterations.
      // Benchmarked: 128×128×8 t8×8 is +13% faster than 128×128×16 t8×8 on lm_head dX
      return {
        ...DEFAULT_CONFIG,
        tileM: 128,
        tileN: 128,
        tileK: 8,
        threadTileM: 8,
        threadTileN: 8,
      };

    case "tall_skinny":
      if (hasEpilogue) {
        // Epilogue: conservative thread tiles to avoid register pressure
        return {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 8,
        };
      }
      // Tall matrices (e.g. lm_head backward dW: M=50304, N=768, K=512)
      // tileK=16 halves K-loop iterations (32→16 for K=512); 6KB shared memory
      // Benchmarked: 64×128×16 t8×4 is +6% faster than 64×128×8 t8×4 on lm_head dW
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 128,
        tileK: 16,
        threadTileM: 8,
        threadTileN: 4,
      };

    case "short_wide":
      if (hasEpilogue) {
        // Epilogue: conservative thread tiles to avoid register pressure
        return {
          ...DEFAULT_CONFIG,
          tileM: 64,
          tileN: 64,
          tileK: 8,
        };
      }
      // Wide matrices (e.g. lm_head: M=512, N=50304, K=768)
      // Large output tile with tileK=16 for fewer K-loop iterations; 256 threads
      // f16 shared memory halves tile footprint, making tileK=16 fit in same budget as tileK=8 with f32
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 128,
        tileK: 16,
        threadTileM: 8,
        threadTileN: 4,
      };

    case "small_k":
      // Small K dimension with large output (e.g. lm_head backward dW: K=seq_len)
      // tileK=32 covers K<=32 in single tile iteration; 64x64 output reduces workgroup count
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 64,
        tileK: 32,
        threadTileM: 8,
        threadTileN: 8,
      };

    case "gemv":
      // Vector operations - smaller tiles, more parallelism
      return {
        ...DEFAULT_CONFIG,
        tileM: 32,
        tileN: 32,
        tileK: 32,
        threadTileM: 4,
        threadTileN: 4,
      };

    case "gemv_row":
      // M=1 row-vector × matrix (decode steps, probes): a single real output
      // row, so tileM=1/threadTileM=1 puts every thread on real work (the
      // generic tile config idles (tileM-1)/tileM of the workgroup). One
      // thread per output column, 64 threads/workgroup; occupancy comes from
      // N/64 workgroups × K-split (these shapes are exactly the K-split
      // sweet spot: tiny output grid, large K).
      return {
        ...DEFAULT_CONFIG,
        tileM: 1,
        tileN: 64,
        tileK: 32,
        threadTileM: 1,
        threadTileN: 1,
      };

    case "batched_small":
      // Small batched - conservative config
      return DEFAULT_CONFIG;

    default:
      return DEFAULT_CONFIG;
  }
}

/**
 * In-memory tuning cache.
 * Maps (shapeClass, dtype) -> best config
 */
const tuningCache = new Map<string, TuneResult>();

/**
 * Get cache key for tuning result.
 */
function getTuningCacheKey(shapeClass: ShapeClass, dtype: DType): string {
  return `${shapeClass}_${dtype}`;
}

/**
 * Get cached tuning result if available.
 */
export function getCachedTuningResult(
  shapeClass: ShapeClass,
  dtype: DType,
): TuneResult | undefined {
  const key = getTuningCacheKey(shapeClass, dtype);
  return tuningCache.get(key);
}

/**
 * Cache a tuning result.
 */
export function cacheTuningResult(result: TuneResult): void {
  const key = getTuningCacheKey(result.shapeClass, result.dtype);
  tuningCache.set(key, result);
}

/**
 * Clear the tuning cache.
 */
export function clearTuningCache(): void {
  tuningCache.clear();
}

/**
 * Export the tuning cache as JSON.
 */
export function exportTuningCache(): string {
  const entries: Array<[string, TuneResult]> = [];
  for (const [key, value] of tuningCache) {
    entries.push([key, value]);
  }
  return JSON.stringify(entries, null, 2);
}

/**
 * Import tuning results from JSON.
 */
export function importTuningCache(json: string): void {
  try {
    const entries = JSON.parse(json) as Array<[string, TuneResult]>;
    for (const [key, value] of entries) {
      tuningCache.set(key, value);
    }
  } catch {
    throw new Error("Invalid tuning cache JSON");
  }
}

/**
 * Run autotuning for a specific shape class.
 *
 * @param benchmarkFn Function to benchmark a config
 * @param m Matrix M dimension
 * @param n Matrix N dimension
 * @param k Matrix K dimension
 * @param dtype Data type
 * @param options Tuning options
 * @param includeSubgroups Whether to include subgroup variants
 * @returns Best tuning result
 */
export async function autotune(
  benchmarkFn: BenchmarkFn,
  m: number,
  n: number,
  k: number,
  dtype: DType = "f32",
  options: AutotuneOptions = {},
  includeSubgroups = false,
): Promise<TuneResult> {
  const { maxTrials = 27, warmupIters = 3, timingIters = 5 } = options;

  const shapeClass = classifyShape(m, n, k, 1);
  const flops = 2 * m * n * k;

  // Check cache first
  const cached = getCachedTuningResult(shapeClass, dtype);
  if (cached) {
    return cached;
  }

  // Generate configs to try
  const allConfigs = generateTuningConfigs(includeSubgroups);

  // Limit trials
  const configs = allConfigs.slice(0, maxTrials);

  // Track best result
  let bestResult: TuneResult | null = null;
  let bestGflops = 0;

  for (const config of configs) {
    try {
      // Benchmark this config
      const medianMs = await benchmarkFn(config, warmupIters, timingIters);

      // Calculate GFLOPs/s
      const gflops = flops / (medianMs * 1e6);

      if (gflops > bestGflops) {
        bestGflops = gflops;
        bestResult = {
          config,
          gflopsPerSec: gflops,
          medianMs,
          shapeClass,
          dtype,
        };
      }
    } catch {
      // Config failed to compile or dispatch — skip and try next
    }
  }

  // If no config worked, use default
  if (!bestResult) {
    const defaultConfig = getDefaultConfigForShape(shapeClass);
    bestResult = {
      config: defaultConfig,
      gflopsPerSec: 0,
      medianMs: Infinity,
      shapeClass,
      dtype,
    };
  }

  // Cache the result
  cacheTuningResult(bestResult);

  return bestResult;
}

/**
 * Generate neighbor configs around a base config for focused autotuning.
 * Produces ~10-15 candidates by varying tile dimensions, thread tile sizes,
 * and optionally subgroup usage around the known-good base config.
 */
export function generateNeighborConfigs(
  baseConfig: MatmulKernelConfig,
  includeSubgroups: boolean,
): MatmulKernelConfig[] {
  const configs: MatmulKernelConfig[] = [baseConfig];
  const { tileM, tileN, tileK } = baseConfig;

  // Tile dimension neighbors
  for (const newTileM of [32, 64, 128] as const) {
    if (newTileM !== tileM) configs.push({ ...baseConfig, tileM: newTileM });
  }
  for (const newTileN of [32, 64, 128] as const) {
    if (newTileN !== tileN) configs.push({ ...baseConfig, tileN: newTileN });
  }
  for (const newTileK of [8, 16, 32] as const) {
    if (newTileK !== tileK) configs.push({ ...baseConfig, tileK: newTileK });
  }
  // Thread tile variants
  for (const [ttm, ttn] of [
    [4, 4],
    [8, 4],
    [4, 8],
    [8, 8],
  ] as const) {
    if (ttm !== baseConfig.threadTileM || ttn !== baseConfig.threadTileN) {
      configs.push({ ...baseConfig, threadTileM: ttm, threadTileN: ttn });
    }
  }
  // Subgroup variant
  if (includeSubgroups && !baseConfig.useSubgroups) {
    configs.push({ ...baseConfig, useSubgroups: true });
  }

  // Validate and deduplicate
  const seen = new Set<string>();
  return configs.filter((c) => {
    const key = `${c.tileM}_${c.tileN}_${c.tileK}_${c.threadTileM}_${c.threadTileN}_${c.vectorWidth}_${c.useSubgroups}`;
    if (seen.has(key)) return false;
    seen.add(key);
    try {
      validateConfig(c);
      return true;
    } catch {
      return false;
    }
  });
}

/**
 * Quick autotune: try a small subset of representative configs.
 * Faster than full autotune but may not find the absolute best config.
 */
export async function quickAutotune(
  benchmarkFn: BenchmarkFn,
  m: number,
  n: number,
  k: number,
  dtype: DType = "f32",
): Promise<TuneResult> {
  const shapeClass = classifyShape(m, n, k, 1);

  // Representative configs to try
  const quickConfigs: MatmulKernelConfig[] = [
    DEFAULT_CONFIG,
    getDefaultConfigForShape(shapeClass),
    {
      ...DEFAULT_CONFIG,
      tileM: 64,
      tileN: 64,
    },
    {
      ...DEFAULT_CONFIG,
      tileM: 32,
      tileN: 64,
    },
    {
      ...DEFAULT_CONFIG,
      tileM: 64,
      tileN: 32,
    },
  ];

  // Deduplicate
  const seen = new Set<string>();
  const uniqueConfigs = quickConfigs.filter((config) => {
    const key = JSON.stringify(config);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const flops = 2 * m * n * k;
  let bestResult: TuneResult | null = null;
  let bestGflops = 0;

  for (const config of uniqueConfigs) {
    try {
      validateConfig(config);
      const medianMs = await benchmarkFn(config, 2, 3);
      const gflops = flops / (medianMs * 1e6);

      if (gflops > bestGflops) {
        bestGflops = gflops;
        bestResult = {
          config,
          gflopsPerSec: gflops,
          medianMs,
          shapeClass,
          dtype,
        };
      }
    } catch {
      // Config failed to compile or dispatch — skip and try next
    }
  }

  if (!bestResult) {
    bestResult = {
      config: DEFAULT_CONFIG,
      gflopsPerSec: 0,
      medianMs: Infinity,
      shapeClass,
      dtype,
    };
  }

  cacheTuningResult(bestResult);
  return bestResult;
}
