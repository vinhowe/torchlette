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
/**
 * Stage-4 plan/encode split: everything dispatchFusedKernel decides BEFORE
 * touching the GPU — pipeline, the abstract binding layout (in bind-group
 * order), output allocation descriptors, workgroups, params bytes. The
 * stream generator consumes this to emit ALLOC + DISPATCH commands without
 * executing; dispatchFusedKernel consumes the same plan to allocate + bind
 * + dispatch, so the two cannot disagree on binding order or workgroups.
 */
export type FusedBinding =
  | { kind: "input"; pos: number } // inputs[pos].buffer
  | { kind: "output"; index: number } // outputBuffers[index] (allocated)
  | { kind: "params" };

export interface FusedKernelPlan {
  pipeline: GPUComputePipeline;
  /** Bind-group order: inputs (minus donated) then outputs then params. */
  bindings: FusedBinding[];
  /** Per output: bytes to allocate, or the input pos whose buffer it reuses
   *  in-place (donation — output 0 only). */
  outputs: Array<{ bytes: number } | { donatedPos: number }>;
  workgroups: [number, number, number];
  paramsData: Uint32Array;
  /** Resolved donated position within the non-inlined inputs (or -1). */
  donatedPos: number;
}

export function planFusedKernel(
  device: GPUDevice,
  recipe: FusedKernelRecipe,
  inputs: FusedInputTensor[],
  options: FusedDispatchOptions = {},
): FusedKernelPlan {
  const nonInlinedCount = recipe.inputs.filter(
    (inp) => !inp.isInlinedConstant,
  ).length;
  if (inputs.length !== nonInlinedCount) {
    throw new Error(
      `Expected ${nonInlinedCount} non-inlined inputs, got ${inputs.length}`,
    );
  }

  const donated = options.donatedInput;
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

  const primaryOutput = recipe.outputs[0];
  const totalElements = sizeOf(primaryOutput.shape);

  const outputs: Array<{ bytes: number } | { donatedPos: number }> = [];
  for (let oi = 0; oi < recipe.outputs.length; oi++) {
    if (donated !== undefined && oi === 0) {
      outputs.push({ donatedPos });
    } else {
      outputs.push({
        bytes: totalElements * dtypeBytes(recipe.outputs[oi].dtype),
      });
    }
  }

  // Bind order: non-inlined inputs (minus donated) → outputs → params.
  const bindings: FusedBinding[] = [];
  for (let i = 0; i < inputs.length; i++) {
    if (i === donatedPos) continue;
    bindings.push({ kind: "input", pos: i });
  }
  for (let oi = 0; oi < recipe.outputs.length; oi++) {
    bindings.push({ kind: "output", index: oi });
  }
  bindings.push({ kind: "params" });

  const totalWorkgroups = Math.ceil(kernel.workItems / kernel.workgroupSize);
  const workgroupsX = Math.min(totalWorkgroups, MAX_WORKGROUPS_PER_DIM);
  const workgroupsY =
    totalWorkgroups <= MAX_WORKGROUPS_PER_DIM
      ? 1
      : Math.ceil(totalWorkgroups / MAX_WORKGROUPS_PER_DIM);

  return {
    pipeline,
    bindings,
    outputs,
    workgroups: [workgroupsX, workgroupsY, 1],
    paramsData: new Uint32Array([totalElements]),
    donatedPos,
  };
}

export function dispatchFusedKernel(
  device: GPUDevice,
  recipe: FusedKernelRecipe,
  inputs: FusedInputTensor[],
  options: FusedDispatchOptions = {},
): FusedDispatchResult {
  const plan = planFusedKernel(device, recipe, inputs, options);
  const { pipeline, donatedPos } = plan;

  // All outputs must have the same shape (required for elementwise fusion)
  const primaryOutput = recipe.outputs[0];

  // Create output buffers for all outputs (per the plan's descriptors).
  // resolveOutputBuffer's aliasing check prevents returning an input buffer
  // as the output — that would be a read/read_write conflict in one pass.
  const inputGPUBuffers = inputs.map((inp) => inp.buffer);
  const outputBuffers: GPUBuffer[] = [];
  const outputTensors: FusedOutputTensor[] = [];
  for (let oi = 0; oi < recipe.outputs.length; oi++) {
    const output = recipe.outputs[oi];
    const desc = plan.outputs[oi];
    // In-place out0 into the donated input's buffer (no alloc, no
    // recordAlloc — the recording resolves this binding to the input's
    // existing slot, in-place discipline like adamStep).
    const buffer =
      "donatedPos" in desc
        ? inputs[desc.donatedPos].buffer
        : resolveOutputBuffer(device, desc.bytes, inputGPUBuffers);
    trackSharedEncoderWrite(buffer);
    outputBuffers.push(buffer);
    outputTensors.push({
      buffer,
      shape: output.shape.slice(),
      dtype: output.dtype,
    });
  }

  // Create uniform buffer for params via shared pool
  const paramsBuffer = createParamsBuffer(device, plan.paramsData);

  // Build the bind-group buffer array from the plan's binding order.
  const bgBuffers: GPUBuffer[] = plan.bindings.map((b) => {
    if (b.kind === "input") return inputs[b.pos].buffer;
    if (b.kind === "output") return outputBuffers[b.index];
    return paramsBuffer;
  });
  const bindGroup = cachedCreateBindGroup(device, pipeline, bgBuffers);

  dispatchComputePass(
    pipeline,
    bindGroup,
    plan.workgroups[0],
    plan.workgroups[1],
    plan.workgroups[2],
  );

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
