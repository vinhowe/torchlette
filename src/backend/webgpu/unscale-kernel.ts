/**
 * Fused GradScaler Unscale Kernel
 *
 * A single WGSL compute shader per parameter that:
 * 1. Reads gradient element
 * 2. Multiplies by invScale (unscale)
 * 3. Checks if result is finite
 * 4. If non-finite: atomically sets shared inf flag, writes 0.0 to grad
 * 5. If finite: writes unscaled value to grad
 *
 * The inf flag is a shared 4-byte GPUBuffer (atomic<u32>) across all parameter
 * dispatches within a single unscale_() call.
 *
 * Handles large buffers (>maxStorageBufferBindingSize) via tile-IR dispatchChunked.
 */

import { awaitDeferredFence } from "./buffer-pool";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import { getMaxStorageBufferBindingSize, requireContext } from "./gpu-context";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage, GPUMapMode } from "./gpu-types";
import { gpuMemoryTracker } from "./memory-tracker";
import { isProfilingEnabled } from "./profiler";
import {
  F32_ONE_BITS,
  MAX_WORKGROUPS_PER_DIM,
  WORKGROUP_SIZE,
} from "./shape-utils";
import { trackSharedEncoderWrite } from "./shared-encoder";
import {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./tile-dispatch";
import type { TileKernelSpec } from "./tile-ir";

// ============================================================================
// Tile-IR Unscale Spec
// ============================================================================

function makeUnscaleSpec(): TileKernelSpec {
  return {
    name: "unscaleGrad",
    workgroupSize: WORKGROUP_SIZE,
    bindings: {
      grad_in: { storage: "read", type: "f32" },
      grad_out: { storage: "read_write", type: "f32" },
      inf_flag: { storage: "atomic", type: "u32" },
    },
    uniforms: {
      inv_scale: "f32",
      num_elements: "u32",
      grid_stride: "u32",
      _pad0: "u32",
    },
    uniformBindingIndex: 3,
    grid: (u) => {
      const wg = Math.ceil(u.num_elements / WORKGROUP_SIZE);
      if (wg <= MAX_WORKGROUPS_PER_DIM) return [wg];
      const x = Math.min(wg, MAX_WORKGROUPS_PER_DIM);
      return [x, Math.ceil(wg / x)];
    },
    kernel(ctx) {
      // 2D-safe indexing: grid_stride = gridSizeX * WORKGROUP_SIZE.
      // For 1D dispatch, globalId(1) = 0 so grid_stride is irrelevant.
      const gridStride = ctx.uniform("grid_stride");
      const idx = ctx.emitLet(
        "idx",
        ctx.globalId(0).add(ctx.globalId(1).mul(gridStride)),
      );
      const numElements = ctx.uniform("num_elements");
      ctx.ifThen(idx.ge(numElements), () => {
        ctx.emitReturn();
      });

      const invScale = ctx.uniform("inv_scale").bitcastTo("f32");
      const g = ctx.emitLet("g", ctx.load("grad_in", idx).mul(invScale));

      // Check finite via bit pattern
      const bits = g.bitcastTo("u32");
      const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xff));
      const isFiniteVal = exponent.ne(ctx.u32(0xff));

      ctx.ifThenElse(
        isFiniteVal,
        () => {
          ctx.emitStore("grad_out", idx, g);
        },
        () => {
          ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(F32_ONE_BITS));
          ctx.emitStore("grad_out", idx, ctx.f32(0.0));
        },
      );
    },
  };
}

// ============================================================================
// Dispatcher (singleton, created on first use)
// ============================================================================

let unscaleDispatcher: TileKernelInstance | null = null;

function getUnscaleDispatcher(): TileKernelInstance {
  if (!unscaleDispatcher) {
    unscaleDispatcher = createTileKernelDispatcher(makeUnscaleSpec());
  }
  return unscaleDispatcher;
}

// ============================================================================
// Inf Flag Buffer
// ============================================================================

/**
 * Persistent 4-byte atomic buffer for inf detection.
 * Allocated once and zeroed via writeBuffer each step to stabilize
 * buffer identity for bind group caching.
 */
let persistentInfFlagBuffer: GPUBuffer | null = null;
const infFlagZeroData = new Float32Array([0.0]);

/**
 * Get or allocate a 4-byte atomic buffer for inf detection, initialized to 0.0.
 * Shared across all parameter dispatches in one unscale_() call.
 *
 * Uses a persistent buffer: allocated once, zeroed via writeBuffer each step.
 * This stabilizes buffer identity for bind group caching (76 Adam dispatches
 * all reference the same GPUBuffer across steps).
 */
export function allocateInfFlagBuffer(device?: GPUDevice): GPUBuffer {
  if (persistentInfFlagBuffer) {
    const dev = device ?? requireContext().device;
    dev.queue.writeBuffer(persistentInfFlagBuffer, 0, infFlagZeroData);
    return persistentInfFlagBuffer;
  }
  const dev = device ?? requireContext().device;
  persistentInfFlagBuffer = dev.createBuffer({
    size: 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  }) as GPUBuffer;
  dev.queue.writeBuffer(persistentInfFlagBuffer, 0, infFlagZeroData);
  return persistentInfFlagBuffer;
}

/**
 * Read the inf flag buffer and destroy it.
 * Returns 0.0 (all finite) or 1.0 (inf/NaN found).
 *
 * Flushes the shared encoder first to ensure all kernel writes are submitted.
 */
export async function readInfFlag(infFlagBuffer: GPUBuffer): Promise<number> {
  const ctx = requireContext();
  const device = ctx.device;

  // Consume the deferred fence from markStep() BEFORE any GPU sync.
  // resolveDeferred() always calls markStep() → issueDeferredFence() before
  // calling readInfFlag(). The pending onSubmittedWorkDone from issueDeferredFence
  // causes mapAsync to deadlock on V100/Dawn when both are in-flight concurrently.
  // Awaiting it first ensures no concurrent GPU sync operations.
  await awaitDeferredFence();

  // All pending GPU work (including unscale kernel writes) was submitted by
  // markStep() and confirmed complete by awaitDeferredFence. No need to
  // flushSharedEncoder.

  // Create staging buffer for readback
  const staging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(infFlagBuffer, 0, staging, 0, 4);
  device.queue.submit([encoder.finish()]);

  // On V100/Dawn, mapAsync on buffers populated only by copyBufferToBuffer
  // can deadlock after timestamp query operations. Work around by issuing a
  // queue.writeBuffer to a separate fence buffer first — mapAsync correctly
  // tracks writeBuffer (direct queue op), and its completion confirms all
  // prior GPU work (including the copy) has finished.
  if (isProfilingEnabled()) {
    const fenceBuf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    device.queue.writeBuffer(fenceBuf, 0, new Uint8Array([1, 2, 3, 4]));
    await fenceBuf.mapAsync(GPUMapMode.READ);
    fenceBuf.unmap();
    fenceBuf.destroy();
  }

  // V100/Dawn workaround: mapAsync on copyBufferToBuffer destinations deadlocks
  // permanently after GPU timestamp query operations. Use timeout to detect.
  // Returns 1.0 (inf found) as the safe default — this skips the optimizer step
  // and decreases the loss scale, preventing NaN propagation.
  const MAPASYNC_TIMEOUT_MS = 2000;
  let mapOk = false;
  try {
    const mapResult = await Promise.race([
      staging.mapAsync(GPUMapMode.READ).then(() => true),
      new Promise<false>((resolve) =>
        setTimeout(() => resolve(false), MAPASYNC_TIMEOUT_MS),
      ),
    ]);
    mapOk = mapResult;
  } catch (_e) {
    staging.destroy();
    return 1.0; // Safe default: assume inf found
  }

  if (!mapOk) {
    console.warn(
      "[readInfFlag] mapAsync timed out (V100/Dawn timestamp corruption). Assuming inf found (safe default).",
    );
    // Destroy the staging buffer to cancel the pending mapAsync in Dawn's queue.
    // Leaving it alive corrupts subsequent mapAsync calls.
    staging.destroy();
    // Return 1.0 (inf found) as safe default — GradScaler will skip the optimizer
    // step and decrease the loss scale. This prevents NaN propagation that occurs
    // when returning 0 (no inf) with actually-infinite gradients.
    return 1.0;
  }

  const mapped = staging.getMappedRange();
  const value = new Float32Array(mapped)[0];
  staging.unmap();
  staging.destroy();

  // Do NOT destroy the inf flag buffer — it's persistent and reused across steps.

  return value;
}

/**
 * Destroy the persistent inf flag buffer (cleanup on WebGPU teardown).
 */
function destroyPersistentInfFlagBuffer(): void {
  if (persistentInfFlagBuffer) {
    persistentInfFlagBuffer.destroy();
    persistentInfFlagBuffer = null;
  }
}

// ============================================================================
// Output Buffer Allocation
// ============================================================================

function allocateFreshOutputBuffer(
  device: GPUDevice,
  sizeBytes: number,
): GPUBuffer {
  const buf = device.createBuffer({
    size: sizeBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  gpuMemoryTracker.trackAllocation(buf, sizeBytes);
  trackSharedEncoderWrite(buf);
  return buf;
}

// ============================================================================
// Dispatch
// ============================================================================

interface UnscaleGradResult {
  gradOutBuffer: GPUBuffer;
}

/**
 * Dispatch the fused unscale+inf-check kernel.
 *
 * Uses tile-IR dispatchChunked for buffers exceeding maxStorageBufferBindingSize.
 * All chunks share the same infFlagBuffer (atomic writes).
 */
export function dispatchUnscaleGrad(
  gradBuffer: GPUBuffer,
  numElements: number,
  invScale: number,
  infFlagBuffer: GPUBuffer,
): UnscaleGradResult {
  const bytesPerElement = 4; // f32
  const totalBytes = numElements * bytesPerElement;
  const maxBindingSize = getMaxStorageBufferBindingSize();
  const needsChunking = totalBytes > maxBindingSize;

  // Allocate output buffer (fresh, no pool reuse)
  const alignedBytes = roundUpToPowerOfTwo(totalBytes);
  const gradOut = allocateFreshOutputBuffer(
    requireContext().device,
    alignedBytes,
  );

  // Compute grid_stride for 2D-safe indexing based on per-chunk element count
  const elemPerChunk = needsChunking
    ? computeFlatChunkLayout(numElements, bytesPerElement, maxBindingSize, 256)
        .elementsPerChunk
    : numElements;
  const workgroups = Math.ceil(elemPerChunk / WORKGROUP_SIZE);
  const gridSizeX = Math.min(workgroups, MAX_WORKGROUPS_PER_DIM);
  const gridStride = gridSizeX * WORKGROUP_SIZE;

  const dispatcher = getUnscaleDispatcher();
  const buffers = {
    grad_in: gradBuffer,
    grad_out: gradOut,
    inf_flag: infFlagBuffer,
  };
  const uniforms = {
    inv_scale: invScale,
    num_elements: numElements,
    grid_stride: gridStride,
    _pad0: 0,
  };

  if (needsChunking) {
    dispatcher.dispatchChunked(buffers, uniforms, {
      modes: { grad_in: "chunked", grad_out: "chunked", inf_flag: "scalar" },
      sizeUniform: "num_elements",
      totalElements: numElements,
      maxBytesPerElement: bytesPerElement,
    });
  } else {
    dispatcher.dispatch(buffers, uniforms);
  }

  return { gradOutBuffer: gradOut };
}

// ============================================================================
// Helpers
// ============================================================================

/** Round up buffer size to next power of 2 (matching pool size classes) */
function roundUpToPowerOfTwo(size: number): number {
  if (size <= 256) return 256;
  let s = 1;
  while (s < size) s <<= 1;
  return s;
}

/**
 * Reset all module-local mutable state (dispatcher, persistent inf flag buffer).
 */
export function resetUnscaleKernelState(): void {
  destroyPersistentInfFlagBuffer();
  if (unscaleDispatcher) {
    unscaleDispatcher.reset();
    unscaleDispatcher = null;
  }
}
