/**
 * Fused Kernel Dispatch
 *
 * Handles compilation, caching, and execution of fused elementwise kernels.
 * Supports vectorized memory coalescing (§15.3) for improved bandwidth.
 */

import { sizeOf } from "../../core/shape";
import type { DType } from "../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  releaseParamsBuffer,
} from "./bind-group-cache";
import { allocateOutputBuffer, resolveOutputBuffer } from "./buffer-arena";
import { dispatchComputePass } from "./dispatch";
import { generateFusedKernelTileIR } from "./fusion-tile-ir";
import {
  computeKernelMeta,
  type FusedKernelRecipe,
  type GeneratedKernel,
  type KernelGenOptions,
} from "./fusion-types";
import type { GPUBuffer, GPUComputePipeline, GPUDevice } from "./gpu-types";
import { getWarmupPipeline, recordPipeline } from "./pipeline-warmup";
import { dtypeBytes, MAX_WORKGROUPS_PER_DIM } from "./shape-utils";
import { onTeardown, trackSharedEncoderWrite } from "./webgpu-state";

// ============================================================================
// Kernel Cache
// ============================================================================

interface CachedPipeline {
  pipeline: GPUComputePipeline;
  kernel: GeneratedKernel;
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
    const kernel = generateFusedKernelTileIR(recipe, options);

    // Check warmup cache before synchronous compilation
    let pipeline = getWarmupPipeline(meta.cacheKey);
    if (!pipeline) {
      recordPipeline(meta.cacheKey, kernel.source);
      const module = device.createShaderModule({ code: kernel.source });
      pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }

    // Cache it (Map preserves insertion order — first key is oldest)
    if (this.cache.size >= this.maxSize) {
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }

    this.cache.set(meta.cacheKey, { pipeline, kernel });

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
onTeardown(resetFusionCache);

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
  /**
   * BUFFER DONATION (see KernelGenOptions.donatedInput): index into
   * recipe.inputs whose buffer becomes the primary output IN PLACE. The
   * caller guarantees liveness (this dispatch is the input's last reader)
   * and shape/dtype equality with out0; single-output recipes only.
   */
  donatedInput?: number;
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
  const nonInlinedCount = recipe.inputs.filter(
    (inp) => !inp.isInlinedConstant,
  ).length;

  // Validate inputs — caller should pass only non-inlined input buffers
  if (inputs.length !== nonInlinedCount) {
    throw new Error(
      `Expected ${nonInlinedCount} non-inlined inputs, got ${inputs.length}`,
    );
  }

  // Donation: the donated input shares out0's binding (one fewer binding).
  const donated = options.donatedInput;
  // Position of the donated input within the non-inlined `inputs` array.
  let donatedPos = -1;
  if (donated !== undefined) {
    let pos = 0;
    for (let i = 0; i < recipe.inputs.length; i++) {
      if (recipe.inputs[i].isInlinedConstant) continue;
      if (i === donated) {
        donatedPos = pos;
        break;
      }
      pos++;
    }
    if (donatedPos < 0) throw new Error(`donatedInput ${donated} not found`);
  }

  // Check storage buffer binding count against device limits BEFORE creating pipeline.
  const storageBindingCount =
    nonInlinedCount - (donated !== undefined ? 1 : 0) + recipe.outputs.length;
  const maxStorageBuffers = device.limits.maxStorageBuffersPerShaderStage;
  if (storageBindingCount > maxStorageBuffers!) {
    throw new Error(
      `Fused kernel requires ${storageBindingCount} storage buffers but device limit is ${maxStorageBuffers}`,
    );
  }

  const kernelCache = options.cache ?? getFusionCache();
  const { pipeline, kernel } = kernelCache.getOrCreate(device, recipe, {
    vectorize: options.vectorize ?? true,
    donatedInput: donated,
  });

  // All outputs must have the same shape (required for elementwise fusion)
  const primaryOutput = recipe.outputs[0];
  const totalElements = sizeOf(primaryOutput.shape);

  // Create output buffers for all outputs.
  // Use resolveOutputBuffer (not allocateOutputBuffer) so the arena aliasing
  // check prevents returning an input buffer as the output — that would create
  // a read/read_write conflict within the same compute pass.
  const inputGPUBuffers = inputs.map((inp) => inp.buffer);
  const outputBuffers: GPUBuffer[] = [];
  const outputTensors: FusedOutputTensor[] = [];
  for (let oi = 0; oi < recipe.outputs.length; oi++) {
    const output = recipe.outputs[oi];
    let buffer: GPUBuffer;
    if (donated !== undefined && oi === 0) {
      // In-place: write OUT0 (only!) into the donated input's buffer. No
      // allocation, no recordAlloc — the compiled-plan recording resolves
      // this binding to the input's existing slot (in-place discipline,
      // like adamStep). Other outputs allocate normally; giving them the
      // donated buffer too binds one buffer at multiple writable indices —
      // a WebGPU validation error that rejects the ENTIRE submit (the
      // multi-output foreach freeze).
      buffer = inputs[donatedPos].buffer;
    } else {
      const outputBytes = totalElements * dtypeBytes(output.dtype);
      buffer = resolveOutputBuffer(device, outputBytes, inputGPUBuffers);
    }
    trackSharedEncoderWrite(buffer);
    outputBuffers.push(buffer);
    outputTensors.push({
      buffer,
      shape: output.shape.slice(),
      dtype: output.dtype,
    });
  }

  // Create uniform buffer for params via shared pool
  const paramsBuffer = createParamsBuffer(
    device,
    new Uint32Array([totalElements]),
  );

  // Build flat buffer array: non-inlined inputs (minus the donated one,
  // which is bound as out0), then outputs, then params
  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < inputs.length; i++) {
    if (i === donatedPos) continue;
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
  const workgroupsX = Math.min(totalWorkgroups, MAX_WORKGROUPS_PER_DIM);
  const workgroupsY =
    totalWorkgroups <= MAX_WORKGROUPS_PER_DIM
      ? 1
      : Math.ceil(totalWorkgroups / MAX_WORKGROUPS_PER_DIM);

  dispatchComputePass(pipeline, bindGroup, workgroupsX, workgroupsY, 1);

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
