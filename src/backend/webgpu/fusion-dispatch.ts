/**
 * Fused Kernel Dispatch
 *
 * Handles compilation, caching, and execution of fused elementwise kernels.
 * Supports vectorized memory coalescing (§15.3) for improved bandwidth.
 */

import { submitOrCollect, getSharedEncoderInstance, getCurrentOpLabel, createParamsBuffer, releaseParamsBuffer, allocateOutputBuffer, trackSharedEncoderWrite, cachedCreateBindGroup, type RecordedDispatch, getAndClearLastBindGroupBuffers } from "./index";

/** Module-level recording buffer (shared with index.ts recording system). */
let fusionRecordingBuffer: RecordedDispatch[] | null = null;
export function setFusionRecordingBuffer(buf: RecordedDispatch[] | null): void {
  fusionRecordingBuffer = buf;
}

import type { DType } from "../types";
import { dtypeBytes } from "./shape-utils";
import { profileApiCall, getTimestampWrites, getProfileModule } from "./profiler";
import {
  type FusedKernelRecipe,
  type GeneratedKernel,
  generateFusedKernel,
  computeKernelMeta,
  type KernelGenOptions,
} from "./fusion-codegen";
import type { GPUBuffer, GPUDevice, GPUComputePipeline } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";

// ============================================================================
// Kernel Cache
// ============================================================================

interface CachedPipeline {
  pipeline: GPUComputePipeline;
  kernel: GeneratedKernel;
  createdAt: number;
}

/**
 * Cache for compiled fusion pipelines.
 */
export class FusionKernelCache {
  private cache = new Map<string, CachedPipeline>();
  private maxSize: number;

  constructor(maxSize = 1024) {
    this.maxSize = maxSize;
  }

  /**
   * Get or create a pipeline for a fusion recipe.
   * Computes a cheap cache key first; only does full WGSL codegen on miss.
   */
  getOrCreate(
    device: GPUDevice,
    recipe: FusedKernelRecipe,
    options: KernelGenOptions = {},
  ): { pipeline: GPUComputePipeline; kernel: GeneratedKernel } {
    // Cheap: compute cache key + metadata without generating WGSL
    const meta = computeKernelMeta(recipe, options);
    const cached = this.cache.get(meta.cacheKey);

    if (cached) {
      return { pipeline: cached.pipeline, kernel: cached.kernel };
    }

    // Cache miss: do full WGSL codegen + shader compilation
    const kernel = generateFusedKernel(recipe, options);

    const module = device.createShaderModule({ code: kernel.source });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    // Cache it
    if (this.cache.size >= this.maxSize) {
      // Evict oldest entry
      let oldest: string | null = null;
      let oldestTime = Infinity;
      for (const [key, entry] of this.cache) {
        if (entry.createdAt < oldestTime) {
          oldestTime = entry.createdAt;
          oldest = key;
        }
      }
      if (oldest) {
        this.cache.delete(oldest);
      }
    }

    this.cache.set(meta.cacheKey, {
      pipeline,
      kernel,
      createdAt: Date.now(),
    });

    return { pipeline, kernel };
  }

  /**
   * Clear the cache.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics.
   */
  stats(): { size: number; maxSize: number } {
    return { size: this.cache.size, maxSize: this.maxSize };
  }
}

// Global kernel cache
let globalKernelCache: FusionKernelCache | null = null;

/**
 * Get the global fusion kernel cache.
 */
export function getFusionCache(): FusionKernelCache {
  if (!globalKernelCache) {
    globalKernelCache = new FusionKernelCache();
  }
  return globalKernelCache;
}

/**
 * Reset the global fusion kernel cache.
 */
export function resetFusionCache(): void {
  globalKernelCache?.clear();
  globalKernelCache = null;
}

// ============================================================================
// Dispatch
// ============================================================================

/**
 * Input tensor for fused kernel dispatch.
 */
interface FusedInputTensor {
  buffer: GPUBuffer;
  shape: number[];
  dtype: DType;
}

/**
 * Single output tensor from fused kernel dispatch.
 */
interface FusedOutputTensor {
  buffer: GPUBuffer;
  shape: number[];
  dtype: DType;
}

/**
 * Result of fused kernel dispatch.
 * Supports single or multi-output fusion (§15.2).
 */
interface FusedDispatchResult {
  /** All output buffers (supports multiple for §15.2) */
  outputs: FusedOutputTensor[];
  /** Primary output (first output, for backwards compatibility) */
  buffer: GPUBuffer;
  shape: number[];
  dtype: DType;
}

/**
 * Options for fused kernel dispatch.
 */
interface FusedDispatchOptions {
  /** Optional kernel cache (uses global if not provided) */
  cache?: FusionKernelCache;
  /** Enable vectorized loads/stores for memory coalescing (§15.3) */
  vectorize?: boolean;
}

/**
 * Dispatch a fused elementwise kernel.
 * Supports single or multi-output fusion (§15.2).
 *
 * @param device - GPU device
 * @param recipe - Fusion recipe describing the computation
 * @param inputs - Input tensors (must match recipe.inputs order)
 * @param options - Dispatch options (cache, vectorize)
 */
export function dispatchFusedKernel(
  device: GPUDevice,
  recipe: FusedKernelRecipe,
  inputs: FusedInputTensor[],
  options: FusedDispatchOptions = {},
): FusedDispatchResult {
  // Count non-inlined inputs (inlined constants don't need buffer bindings)
  const nonInlinedCount = recipe.inputs.filter(inp => !inp.isInlinedConstant).length;

  // Validate inputs — caller should pass only non-inlined input buffers
  if (inputs.length !== nonInlinedCount) {
    throw new Error(
      `Expected ${nonInlinedCount} non-inlined inputs, got ${inputs.length}`,
    );
  }

  // Check storage buffer binding count against device limits BEFORE creating pipeline.
  const storageBindingCount = nonInlinedCount + recipe.outputs.length;
  const maxStorageBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (storageBindingCount > maxStorageBuffers) {
    throw new Error(
      `Fused kernel requires ${storageBindingCount} storage buffers but device limit is ${maxStorageBuffers}`,
    );
  }

  const kernelCache = options.cache ?? getFusionCache();
  const { pipeline, kernel } = kernelCache.getOrCreate(device, recipe, {
    vectorize: options.vectorize ?? true,
  });

  // All outputs must have the same shape (required for elementwise fusion)
  const primaryOutput = recipe.outputs[0];
  const totalElements = primaryOutput.shape.reduce((a, b) => a * b, 1);

  // Create output buffers for all outputs (via shared buffer pool)
  const outputBuffers: GPUBuffer[] = [];
  const outputTensors: FusedOutputTensor[] = [];
  for (const output of recipe.outputs) {
    const outputBytes = totalElements * dtypeBytes(output.dtype);
    const buffer = allocateOutputBuffer(outputBytes);
    trackSharedEncoderWrite(buffer);
    outputBuffers.push(buffer);
    outputTensors.push({
      buffer,
      shape: output.shape.slice(),
      dtype: output.dtype,
    });
  }

  // Create uniform buffer for params via shared pool
  const paramsBuffer = createParamsBuffer(device, new Uint32Array([totalElements]));

  // Build flat buffer array: non-inlined inputs, then outputs, then params
  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < inputs.length; i++) {
    bgBuffers.push(inputs[i].buffer);
  }
  for (let i = 0; i < outputBuffers.length; i++) {
    bgBuffers.push(outputBuffers[i]);
  }
  bgBuffers.push(paramsBuffer);
  const bindGroup = cachedCreateBindGroup(device, pipeline, bgBuffers);

  // Dispatch (batch/shared-encoder mode aware)
  // Use 2D dispatch when workgroups exceed WebGPU per-dimension limit (65535)
  const totalWorkgroups = Math.ceil(kernel.workItems / kernel.workgroupSize);
  const MAX_WG_DIM = 65535;
  const workgroupsX = Math.min(totalWorkgroups, MAX_WG_DIM);
  const workgroupsY = totalWorkgroups <= MAX_WG_DIM ? 1 : Math.ceil(totalWorkgroups / MAX_WG_DIM);

  // Record dispatch if recording is active
  if (fusionRecordingBuffer) {
    fusionRecordingBuffer.push({
      pipeline,
      bindGroup,
      workgroupsX,
      workgroupsY,
      workgroupsZ: 1,
      buffers: getAndClearLastBindGroupBuffers(),
      label: getCurrentOpLabel() ?? undefined,
      module: getProfileModule(),
    });
  }

  const sharedEnc = getSharedEncoderInstance();
  const opLabel = getCurrentOpLabel() ?? "fused";
  if (sharedEnc) {
    const tsWrites = getTimestampWrites(opLabel);
    const pass = sharedEnc.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);
    pass.end();
  } else {
    const encoder = device.createCommandEncoder();
    const tsWrites = getTimestampWrites(opLabel);
    const pass = encoder.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY);
    pass.end();
    submitOrCollect(encoder.finish());
  }

  // Release the params uniform buffer via the params buffer pool (handles deferred destruction)
  releaseParamsBuffer(paramsBuffer);

  // Return result with backwards-compatible primary output fields
  return {
    outputs: outputTensors,
    buffer: outputBuffers[0],
    shape: primaryOutput.shape.slice(),
    dtype: primaryOutput.dtype,
  };
}
