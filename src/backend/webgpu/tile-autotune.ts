/**
 * Generic Tile-IR Autotuning Framework
 *
 * Provides runtime autotuning for any tile-IR kernel via AutotuneConfig.
 * Generates configs from a parameter space, benchmarks each on the GPU,
 * and caches the best config per kernel + shape combination.
 *
 * Equivalent to Triton's @triton.autotune decorator but for our tile-IR.
 */

import type { GPUBuffer } from "./gpu-types";
import { beginSharedEncoder, flushSharedEncoder } from "./shared-encoder";
import { createTileKernelDispatcher } from "./tile-dispatch";
import type { AutotuneConfig } from "./tile-ir";
import { requireContext } from "./webgpu-state";

// ============================================================================
// Config Generation
// ============================================================================

/**
 * Generate all valid configs from an AutotuneConfig's parameter space.
 * Applies Cartesian product over all param values, filtered by constraints.
 * If `uniforms` is provided and `pruneForShape` is defined, narrows the
 * parameter space first.
 */
export function generateTileConfigs(
  autoConfig: AutotuneConfig,
  uniforms?: Record<string, number>,
): Record<string, number>[] {
  // Optionally narrow params based on shape
  const params =
    uniforms && autoConfig.pruneForShape
      ? autoConfig.pruneForShape(uniforms)
      : autoConfig.params;

  const paramNames = Object.keys(params);
  const paramValues = paramNames.map((name) => params[name].values);

  // Cartesian product
  const configs: Record<string, number>[] = [];
  const indices = new Array(paramNames.length).fill(0);

  outer: while (true) {
    // Build config from current indices
    const config: Record<string, number> = {};
    for (let i = 0; i < paramNames.length; i++) {
      config[paramNames[i]] = paramValues[i][indices[i]];
    }

    // Apply constraints
    const valid =
      !autoConfig.constraints ||
      autoConfig.constraints.every((fn) => fn(config));
    if (valid) {
      configs.push(config);
    }

    // Advance indices (odometer-style)
    for (let i = paramNames.length - 1; i >= 0; i--) {
      indices[i]++;
      if (indices[i] < paramValues[i].length) continue outer;
      indices[i] = 0;
    }
    break; // All combinations exhausted
  }

  return configs;
}

/**
 * Get the default config from an AutotuneConfig (using each param's default).
 */
export function getDefaultConfig(
  autoConfig: AutotuneConfig,
): Record<string, number> {
  const config: Record<string, number> = {};
  for (const [name, param] of Object.entries(autoConfig.params)) {
    config[name] = param.default;
  }
  return config;
}

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

export interface AutotuneOptions {
  /** Maximum configs to try (default: 20). */
  maxTrials?: number;
  /** Warmup iterations before timing (default: 3). */
  warmupIters?: number;
  /** Timing iterations (default: 5). */
  timingIters?: number;
}

/**
 * Autotune a tile kernel: benchmark candidate configs and return the best.
 *
 * Uses `performance.now()` + `queue.onSubmittedWorkDone()` for GPU timing
 * (same approach as the matmul autotuner). Each config gets its own
 * compile→dispatch→time cycle.
 *
 * @returns The best config and its median execution time.
 */
export async function autotuneTileKernel(
  autoConfig: AutotuneConfig,
  buffers: Record<string, GPUBuffer>,
  uniforms: Record<string, number>,
  options?: AutotuneOptions,
): Promise<{ config: Record<string, number>; medianMs: number }> {
  const { maxTrials = 20, warmupIters = 3, timingIters = 5 } = options ?? {};

  // Check cache first
  const cacheKey = buildCacheKey(autoConfig, uniforms);
  const cached = tuneCache.get(cacheKey);
  if (cached) return cached;

  const ctx = requireContext();
  const queue = ctx.queue;

  // Generate candidates
  let configs = generateTileConfigs(autoConfig, uniforms);
  if (configs.length > maxTrials) {
    configs = configs.slice(0, maxTrials);
  }

  // If no valid configs, fall back to default
  if (configs.length === 0) {
    const result = { config: getDefaultConfig(autoConfig), medianMs: Infinity };
    tuneCache.set(cacheKey, result);
    return result;
  }

  let bestConfig = configs[0];
  let bestMedian = Infinity;

  for (const config of configs) {
    try {
      const spec = autoConfig.factory(config);
      const dispatcher = createTileKernelDispatcher(spec);

      // Warmup
      for (let i = 0; i < warmupIters; i++) {
        beginSharedEncoder();
        dispatcher.dispatch(buffers, uniforms);
        flushSharedEncoder();
      }
      if (queue.onSubmittedWorkDone) await queue.onSubmittedWorkDone();

      // Timed iterations
      const times: number[] = [];
      for (let i = 0; i < timingIters; i++) {
        const start = performance.now();
        beginSharedEncoder();
        dispatcher.dispatch(buffers, uniforms);
        flushSharedEncoder();
        if (queue.onSubmittedWorkDone) await queue.onSubmittedWorkDone();
        times.push(performance.now() - start);
      }

      times.sort((a, b) => a - b);
      const median = times[Math.floor(times.length / 2)];

      if (median < bestMedian) {
        bestMedian = median;
        bestConfig = config;
      }
    } catch {
      // Skip configs that fail to compile or dispatch
    }
  }

  const result = { config: bestConfig, medianMs: bestMedian };
  tuneCache.set(cacheKey, result);
  return result;
}

// ============================================================================
// Cache
// ============================================================================

/** In-memory cache: specName:uniformKey → best config + timing. */
const tuneCache = new Map<
  string,
  { config: Record<string, number>; medianMs: number }
>();

function buildCacheKey(
  autoConfig: AutotuneConfig,
  uniforms: Record<string, number>,
): string {
  // Use the factory's default config name as the spec identifier
  const defaultConfig = getDefaultConfig(autoConfig);
  const specName = autoConfig.factory(defaultConfig).name;
  // Include all integer uniforms in the key (shape-dependent)
  const uParts = Object.entries(uniforms)
    .filter(([, v]) => Number.isInteger(v))
    .map(([k, v]) => `${k}=${v}`)
    .join(",");
  return `${specName}:${uParts}`;
}

/** Clear the autotuning cache. */
export function clearTileAutotuneCache(): void {
  tuneCache.clear();
}

/** Export the autotuning cache as JSON. */
export function exportTileAutotuneCache(): string {
  return JSON.stringify(Array.from(tuneCache.entries()), null, 2);
}

/** Import autotuning results from JSON. */
export function importTileAutotuneCache(json: string): void {
  const entries = JSON.parse(json) as Array<
    [string, { config: Record<string, number>; medianMs: number }]
  >;
  for (const [key, value] of entries) {
    tuneCache.set(key, value);
  }
}
