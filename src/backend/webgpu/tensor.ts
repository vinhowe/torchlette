/**
 * Tensor construction helpers: createTensor, createTrackedBuffer, createBufferWithData.
 */

import {
  getSlot,
  isCompilationRecordingActive,
} from "../../executor/compiled-plan";
import { getSizeClass, getSizeForClass } from "../../graph/lifetime-analysis";
import type { DType } from "../types";
import { bufAcquire, bufRelease, bufRegister } from "./buffer-debug";
import { bufferPool } from "./buffer-pool";
import type { GPUBuffer, GPUDevice, GPUQueue, WebGPUTensor } from "./gpu-types";
import { GPUBufferUsage, STORAGE_BUFFER_USAGE } from "./gpu-types";
import { gpuMemoryTracker } from "./memory-tracker";
import { profileApiCall } from "./profiler";
import {
  alignBufferSize,
  checkContiguousStrides,
  contiguousStrides,
  sizeOf,
} from "./shape-utils";
import { arenaBufferSet, pinnedBufferSet, sharedEncoderActive } from "./webgpu-state";

/**
 * Create a WebGPU tensor with optional stride info.
 * If strides not provided, assumes contiguous layout.
 * @param ownsBuffer - If true, this tensor owns the buffer and will destroy it.
 *                     If false, this is a view that borrows the buffer.
 */
export function createTensor(
  shape: number[],
  buffer: GPUBuffer,
  strides?: number[],
  offset = 0,
  dtype: DType = "f32",
  ownsBuffer = true,
): WebGPUTensor {
  // Structural invariant: any owning storage tensor produced while a plan
  // is being recorded must come from a buffer that the compiled-plan
  // recorder knows about. The canonical path is allocateOutputBuffer in
  // buffer-arena.ts, which calls recordAlloc() to register the buffer with
  // bufferToSlot. Skipping that path (e.g. calling device.createBuffer
  // directly) historically led to silent plan invalidation and bind-group
  // cache misses. Catching it here fails fast at the dev's keyboard
  // instead of by `console.warn` from the executor after the plan was
  // already wasted.
  if (
    ownsBuffer &&
    isCompilationRecordingActive() &&
    buffer.usage === STORAGE_BUFFER_USAGE &&
    getSlot(buffer) === undefined
  ) {
    throw new Error(
      `createTensor: storage buffer (size ${buffer.size}) is not registered with the compiled-plan recorder. ` +
        `Allocate result buffers via allocateOutputBuffer() from backend/webgpu/buffer-arena.ts instead of ` +
        `device.createBuffer(). See src/executor/compiled-plan.ts for the recording protocol.`,
    );
  }

  const computedStrides = strides ?? contiguousStrides(shape);
  const isContiguousLayout =
    strides === undefined || checkContiguousStrides(shape, computedStrides);

  // Track if already destroyed to prevent double-destroy
  let destroyed = false;

  // Get buffer size and usage for pool release
  const bufferSize = buffer.size;
  const bufferUsage = buffer.usage;

  // Check if this buffer size matches pool expectations (power-of-2)
  const sizeClass = getSizeClass(bufferSize);
  const expectedPoolSize = getSizeForClass(sizeClass);
  const isPoolCompatible =
    bufferUsage === STORAGE_BUFFER_USAGE && bufferSize === expectedPoolSize;

  // Track owning reference for buffer liveness (refcount)
  if (ownsBuffer) {
    bufferPool.incRef(buffer);
  }

  return {
    shape: shape.slice(),
    size: sizeOf(shape),
    buffer,
    strides: computedStrides,
    offset,
    isContiguous: isContiguousLayout,
    dtype,
    ownsBuffer,
    toArray: () => {
      throw new Error("Use cpu() to read back WebGPU tensors");
    },
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      // Only release/destroy if we own the buffer (not a view)
      if (ownsBuffer) {
        bufRelease(buffer, "tensor.destroy");
        // Release our owning reference before pool/destroy
        bufferPool.decRef(buffer);

        // Arena buffers are owned by the arena, not the pool.
        // Don't release or destroy them — the arena will reuse them next step.
        // Pinned buffers are owned by a compiled plan's recorded assignment —
        // destroying or pooling one poisons every future replay submit.
        if (arenaBufferSet.has(buffer) || pinnedBufferSet.has(buffer)) {
          return;
        }

        // Don't track deallocation here - buffer isn't actually destroyed yet.
        // It goes to either pendingRelease (for pool reuse) or pendingDestroy
        // (for deferred destruction after GPU fence). Deallocation is tracked
        // when the buffer is ACTUALLY destroyed in flushPending() or clear().
        //
        // Try to release to pool for reuse; only if buffer is pool-compatible
        const pooled =
          isPoolCompatible &&
          bufferPool.release(buffer, bufferSize, bufferUsage);
        if (!pooled) {
          // Can't pool - use deferred destruction to wait for GPU fence
          // This prevents "buffer destroyed while in use" validation errors
          bufferPool.deferredDestroy(buffer, bufferSize);
        }
      }
    },
  };
}

/**
 * Create a GPU buffer with memory tracking.
 * This ensures all buffer allocations respect the memory limit.
 * @throws GPUMemoryLimitExceededError if allocation would exceed the limit
 */
export function createTrackedBuffer(
  device: GPUDevice,
  descriptor: { size: number; usage: number; mappedAtCreation?: boolean },
  preferredBuffer?: GPUBuffer,
): GPUBuffer {
  const alignedSize = alignBufferSize(descriptor.size);

  // Try to acquire from pool ONLY for storage buffers without mappedAtCreation
  // Pool only contains buffers with STORAGE | COPY_SRC | COPY_DST usage
  const isStorageBuffer = descriptor.usage === STORAGE_BUFFER_USAGE;
  if (isStorageBuffer && !descriptor.mappedAtCreation) {
    // Try preferred buffer first (for bind group cache stability)
    if (preferredBuffer) {
      const pooled = bufferPool.acquirePreferred(alignedSize, preferredBuffer);
      if (pooled) {
        bufAcquire(pooled, "createTrackedBuffer.preferred");
        return pooled;
      }
    }
    const pooled = bufferPool.acquire(alignedSize);
    if (pooled) {
      bufAcquire(pooled, "createTrackedBuffer.acquire");
      return pooled;
    }
  }

  if (
    process.env.TORCHLETTE_DEBUG_BIGALLOC &&
    alignedSize > 64 * 1024 * 1024
  ) {
    console.log(
      `[bigalloc] createTrackedBuffer ${(alignedSize / 1e6).toFixed(0)}MB usage=${descriptor.usage} at ${new Error().stack?.split("\n").slice(2, 5).join(" | ")}`,
    );
  }

  // Check if this is a small UNIFORM-only buffer (params buffers)
  // These are temporary and don't need memory tracking to avoid memory leaks
  const isUniformOnly =
    (descriptor.usage & GPUBufferUsage.UNIFORM) !== 0 &&
    (descriptor.usage & GPUBufferUsage.STORAGE) === 0;
  const isSmallBuffer = alignedSize <= 64; // Params buffers are typically 4-32 bytes
  const skipTracking = isUniformOnly && isSmallBuffer;

  // For poolable buffers, allocate at full size class size for better reuse
  // For non-poolable buffers, use exact aligned size to save memory
  // Cap pool size at maxStorageBufferBindingSize to prevent oversized buffers
  // that would fail bind group validation when bound without explicit size
  let actualSize: number;
  if (isStorageBuffer) {
    // Use pool size class for all storage buffers (including mappedAtCreation)
    // so they can be reused from the pool after release.
    const pooledSize = getSizeForClass(getSizeClass(alignedSize));
    const limits = device.limits;
    const maxBinding = limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
    actualSize = pooledSize <= maxBinding ? pooledSize : alignedSize;
  } else {
    actualSize = alignedSize;
  }

  // Track allocation before creating buffer (skip for small uniform buffers)
  if (!skipTracking) {
    // Check if PHYSICAL GPU memory (tracked + pool-held) would exceed the limit.
    // The memory tracker undercounts actual GPU usage because pool-released buffers
    // are tracked as deallocated but still occupy GPU memory. We must account for
    // pool-held bytes to prevent Vulkan OOM errors.
    const poolHeldBytes = bufferPool.getTotalHeldBytes();
    const physicalUsage =
      gpuMemoryTracker.getCurrentAllocatedBytes() + poolHeldBytes;
    if (physicalUsage + actualSize > gpuMemoryTracker.getMemoryLimit()) {
      // Evict enough pool buffers to make room. evictBuffers actually destroys
      // the GPU buffers (not just drops references), freeing physical GPU memory.
      bufferPool.evictBuffers(actualSize * 2);
    }
    gpuMemoryTracker.trackAllocation(null, actualSize);
  }

  try {
    const buffer = profileApiCall("createBuffer", () =>
      device.createBuffer({
        ...descriptor,
        size: actualSize,
      }),
    );
    // Re-track with the actual buffer reference for deallocation tracking
    if (!skipTracking) {
      gpuMemoryTracker.trackDeallocation(null);
      gpuMemoryTracker.trackAllocation(buffer, actualSize);
    }
    // Track storage buffers in pool for future reuse
    // Note: mappedAtCreation buffers are pool-compatible after unmap() — they
    // behave identically to regular storage buffers and can be reused via writeBuffer().
    if (isStorageBuffer) {
      bufferPool.trackAllocation(buffer, actualSize);
      bufferPool.markAsFromPool(buffer); // Mark so we can release later
    }
    // Track that this is a NEW allocation (not from pool)
    if (!skipTracking) {
      bufferPool.trackNewAllocation(actualSize);
    }
    bufRegister(buffer, actualSize, "createTrackedBuffer");
    return buffer;
  } catch (e) {
    // If buffer creation fails, undo the tracking
    if (!skipTracking) {
      gpuMemoryTracker.trackDeallocation(null);
    }
    throw e;
  }
}

export function createBufferWithData(
  device: GPUDevice,
  data: Float32Array | Int32Array | Uint32Array | Uint16Array,
  queue?: GPUQueue,
): GPUBuffer {
  if (data.byteLength === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }

  const alignedSize = alignBufferSize(data.byteLength);
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

  // Pool uses power-of-2 size classes - must use the full class size
  // Cap at maxStorageBufferBindingSize to avoid oversized pooled buffers
  const rawPoolSize = getSizeForClass(getSizeClass(alignedSize));
  const devLimits = device.limits;
  const maxBindingSizeForPool =
    devLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const poolSize =
    rawPoolSize <= maxBindingSizeForPool ? rawPoolSize : alignedSize;

  // Try to acquire from pool first (only if pool size is power-of-2, i.e. not capped)
  const pooled =
    rawPoolSize <= maxBindingSizeForPool ? bufferPool.acquire(poolSize) : null;
  // Don't reuse pooled buffers for non-f32 data when a shared encoder is
  // active. The fused kernel dispatch allocates output buffers from the same
  // pool; if the size class matches, it may return the same buffer (pool
  // acquire races with createTrackedBuffer). Use mappedAtCreation for safety.
  const isNonF32 = data instanceof Int32Array || data instanceof Uint32Array || data instanceof Uint16Array;
  if (pooled && queue && !(isNonF32 && sharedEncoderActive)) {
    // Reusing a pooled buffer - write data via queue
    // NOTE: Don't call trackAllocation here! The buffer was already tracked
    // when it was first created, and we don't call trackDeallocation when
    // releasing to pool. Memory tracking stays consistent throughout pool lifecycle.
    profileApiCall("writeBuffer", () => queue.writeBuffer(pooled, 0, data));
    return pooled;
  }
  // Release back to pool if we acquired but didn't use
  if (pooled) {
    bufferPool.deferredDestroy(pooled, pooled.size);
  }

  // Create new buffer with mappedAtCreation for efficient initial write.
  // createTrackedBuffer will round up to pool size class and track it in the pool,
  // so this buffer can be reused after release (mappedAtCreation buffers are
  // pool-compatible after unmap).
  const buffer = createTrackedBuffer(device, {
    size: alignedSize,
    usage,
    mappedAtCreation: true,
  });
  // Create appropriate view based on input type
  if (data instanceof Int32Array) {
    new Int32Array(buffer.getMappedRange()).set(data);
  } else if (data instanceof Uint32Array) {
    new Uint32Array(buffer.getMappedRange()).set(data);
  } else if (data instanceof Uint16Array) {
    new Uint16Array(buffer.getMappedRange()).set(data);
  } else {
    new Float32Array(buffer.getMappedRange()).set(data);
  }
  buffer.unmap();
  return buffer;
}
