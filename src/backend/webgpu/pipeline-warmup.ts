/**
 * Pipeline Cache Warmup
 *
 * Records pipeline creation during execution, then replays them in parallel
 * via createComputePipelineAsync to cut step-0 compilation time.
 *
 * Usage:
 * ```typescript
 * // Record during step 0
 * startPipelineRecording();
 * trainStep();
 * const registry = stopPipelineRecording();
 *
 * // On future runs, pre-compile in parallel before step 0
 * await warmupPipelines(device, registry, getPipelineCaches());
 * ```
 */

import type { GPUComputePipeline, GPUDevice } from "./gpu-types";

// ============================================================================
// Warmup Cache
// ============================================================================

/**
 * Pipelines pre-compiled during warmup. Dispatch paths check this on cache miss
 * to avoid synchronous recompilation.
 */
const warmupCache = new Map<string, GPUComputePipeline>();

/** Look up a pre-compiled pipeline from the warmup cache. */
export function getWarmupPipeline(key: string): GPUComputePipeline | undefined {
  return warmupCache.get(key);
}

/** Clear the warmup pipeline cache. */
export function clearWarmupCache(): void {
  warmupCache.clear();
}

// ============================================================================
// Recording
// ============================================================================

let recording = false;
const registry: Array<{ key: string; wgsl: string }> = [];
const registryKeys = new Set<string>();

/** Start recording pipeline creations. */
export function startPipelineRecording(): void {
  recording = true;
  registry.length = 0;
  registryKeys.clear();
}

/** Stop recording and return all recorded entries. */
export function stopPipelineRecording(): Array<{ key: string; wgsl: string }> {
  recording = false;
  return [...registry];
}

/** Record a pipeline creation. Called from getPipeline() and other creation points on cache miss. */
export function recordPipeline(key: string, wgsl: string): void {
  if (!recording) return;
  if (registryKeys.has(key)) return;
  registryKeys.add(key);
  registry.push({ key, wgsl });
}

// ============================================================================
// Warmup
// ============================================================================

/**
 * Pre-compile pipelines in parallel using createComputePipelineAsync.
 * Populates the provided cache maps. Skips entries already present in any cache.
 *
 * Falls back to synchronous createComputePipeline if async is not available.
 */
export async function warmupPipelines(
  device: GPUDevice,
  entries: Array<{ key: string; wgsl: string }>,
): Promise<{ compiled: number; skipped: number; timeMs: number }> {
  const t0 = performance.now();
  let skipped = 0;

  // Filter to entries not already in the warmup cache
  const toCompile: Array<{ key: string; wgsl: string }> = [];
  for (const entry of entries) {
    if (warmupCache.has(entry.key)) {
      skipped++;
    } else {
      toCompile.push(entry);
    }
  }

  if (toCompile.length === 0) {
    return { compiled: 0, skipped, timeMs: performance.now() - t0 };
  }

  if (device.createComputePipelineAsync) {
    // Parallel async compilation
    const promises = toCompile.map(async (entry) => {
      const module = device.createShaderModule({ code: entry.wgsl });
      const pipeline = await device.createComputePipelineAsync?.({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      if (pipeline) warmupCache.set(entry.key, pipeline);
    });
    await Promise.all(promises);
  } else {
    // Fallback: synchronous compilation
    for (const entry of toCompile) {
      const module = device.createShaderModule({ code: entry.wgsl });
      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      warmupCache.set(entry.key, pipeline);
    }
  }

  return {
    compiled: toCompile.length,
    skipped,
    timeMs: performance.now() - t0,
  };
}

// ============================================================================
// Serialization
// ============================================================================

/** Serialize a pipeline registry to JSON for cross-session persistence. */
export function serializeRegistry(
  entries: Array<{ key: string; wgsl: string }>,
): string {
  return JSON.stringify(entries);
}

/** Deserialize a pipeline registry from JSON. */
export function deserializeRegistry(
  json: string,
): Array<{ key: string; wgsl: string }> {
  return JSON.parse(json) as Array<{ key: string; wgsl: string }>;
}
