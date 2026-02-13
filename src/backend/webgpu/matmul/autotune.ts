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
export type AutotuneOptions = {
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
): MatmulKernelConfig {
  switch (shapeClass) {
    case "square_large":
      // Large matrices benefit from larger tiles
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 64,
        tileK: 16,
      };

    case "square_medium":
      // Medium matrices - balanced config
      return {
        ...DEFAULT_CONFIG,
        tileM: 32,
        tileN: 32,
        tileK: 16,
      };

    case "tall_skinny":
      // Tall matrices - favor M dimension
      return {
        ...DEFAULT_CONFIG,
        tileM: 64,
        tileN: 32,
        tileK: 16,
      };

    case "short_wide":
      // Wide matrices - favor N dimension
      return {
        ...DEFAULT_CONFIG,
        tileM: 32,
        tileN: 64,
        tileK: 16,
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
    } catch {}
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
    } catch {}
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
