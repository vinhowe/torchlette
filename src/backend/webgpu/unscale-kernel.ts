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
 * Handles large buffers (>maxStorageBufferBindingSize) via chunked dispatch.
 */

import {
  dispatchComputePass,
  getMaxStorageBufferBindingSize,
  trackSharedEncoderWrite,
  createParamsBuffer,
  releaseParamsBuffer,
  flushSharedEncoder,
  cachedCreateBindGroup,
  awaitDeferredFence,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";
import { isProfilingEnabled } from "./profiler";
import { gpuMemoryTracker } from "./memory-tracker";
import type { GPUBuffer, GPUBindGroup, GPUDevice, GPUCommandEncoder } from "./gpu-types";
import { GPUBufferUsage, GPUMapMode } from "./gpu-types";
import { WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM } from "./shape-utils";

// ============================================================================
// WGSL Shader
// ============================================================================

function unscaleGradShader(use2D: boolean, gridSizeX: number): string {
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  return `
struct UnscaleConfig {
  inv_scale: f32,
  num_elements: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> grad_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> inf_flag: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> config: UnscaleConfig;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= config.num_elements) { return; }

  let g = grad_in[idx] * config.inv_scale;

  // Check finite via bit pattern: f32 exponent bits are [30:23].
  // Inf and NaN have all exponent bits set (exponent = 0xFF = 255).
  // Finite values have exponent < 255.
  let bits = bitcast<u32>(g);
  let exponent = (bits >> 23u) & 0xFFu;
  let is_finite = (exponent != 0xFFu);
  if (!is_finite) {
    // Set flag to 1.0f bit pattern via atomicMax
    // bitcast<u32>(1.0f) = 0x3F800000 = 1065353216
    atomicMax(&inf_flag[0], 1065353216u);
    grad_out[idx] = 0.0;
  } else {
    grad_out[idx] = g;
  }
}
`;
}

// ============================================================================
// Config Buffer
// ============================================================================

function createUnscaleConfigBuffer(
  device: GPUDevice,
  invScale: number,
  numElements: number,
): GPUBuffer {
  // 4 x f32/u32 = 16 bytes
  const data = new ArrayBuffer(16);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);

  f32[0] = invScale;
  u32[1] = numElements;
  u32[2] = 0; // pad
  u32[3] = 0; // pad

  return createParamsBuffer(device, u32);
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
    const dev = device ?? (requireContext().device);
    dev.queue.writeBuffer(persistentInfFlagBuffer, 0, infFlagZeroData);
    return persistentInfFlagBuffer;
  }
  const dev = device ?? (requireContext().device);
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
      new Promise<false>((resolve) => setTimeout(() => resolve(false), MAPASYNC_TIMEOUT_MS)),
    ]);
    mapOk = mapResult;
  } catch (e) {
    staging.destroy();
    return 1.0; // Safe default: assume inf found
  }

  if (!mapOk) {
    console.warn("[readInfFlag] mapAsync timed out (V100/Dawn timestamp corruption). Assuming inf found (safe default).");
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
export function destroyPersistentInfFlagBuffer(): void {
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
 * Handles chunking for buffers larger than maxStorageBufferBindingSize.
 * All chunks share the same infFlagBuffer (atomic writes).
 */
export function dispatchUnscaleGrad(
  gradBuffer: GPUBuffer,
  numElements: number,
  invScale: number,
  infFlagBuffer: GPUBuffer,
): UnscaleGradResult {
  const ctx = requireContext();
  const device = ctx.device;

  const bytesPerElement = 4; // f32
  const totalBytes = numElements * bytesPerElement;
  const maxBindingSize = getMaxStorageBufferBindingSize();

  // Determine if chunking is needed
  const needsChunking = totalBytes > maxBindingSize;

  // Align chunk size for sub-range bindings
  const minAlignment = 256; // minStorageBufferOffsetAlignment
  const elementsPerAlignment = minAlignment / bytesPerElement; // 64
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk = needsChunking
    ? Math.floor(maxElementsPerChunk / elementsPerAlignment) *
      elementsPerAlignment
    : numElements;
  const numChunks = Math.ceil(numElements / elementsPerChunk);

  // Allocate output buffer (fresh, no pool reuse)
  const alignedBytes = roundUpToPowerOfTwo(totalBytes);
  const gradOut = allocateFreshOutputBuffer(device, alignedBytes);

  // Determine dispatch dimensions
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Get or create pipeline
  const key = `unscaleGrad:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const code = unscaleGradShader(use2D, gridSizeX);
  const pipeline = getPipeline(ctx, key, code);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, numElements);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    // Create config buffer for this chunk
    const configBuf = createUnscaleConfigBuffer(device, invScale, chunkSize);

    // Build bind group entries with sub-range bindings for chunked access
    const mkBinding = (buf: GPUBuffer) =>
      needsChunking
        ? { buffer: buf, offset: chunkByteOffset, size: chunkByteSize }
        : { buffer: buf };

    const entries = [
      { binding: 0, resource: mkBinding(gradBuffer) },
      { binding: 1, resource: mkBinding(gradOut) },
      { binding: 2, resource: { buffer: infFlagBuffer } }, // always full 4 bytes
      { binding: 3, resource: { buffer: configBuf } },
    ];

    let bindGroup: GPUBindGroup;
    if (needsChunking) {
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
    } else {
      bindGroup = cachedCreateBindGroup(device, pipeline,
        [gradBuffer, gradOut, infFlagBuffer, configBuf]);
    }

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D
      ? Math.ceil(chunkWorkgroups / dispatchX)
      : 1;

    dispatchComputePass(
      pipeline,
      bindGroup,
      dispatchX,
      dispatchY,
    );

    // Config buffer deferred destruction (shared encoder still active)
    releaseParamsBuffer(configBuf);
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
 * Reset all module-local mutable state (pipeline cache, persistent inf flag buffer).
 */
export function resetUnscaleKernelState(): void {
  destroyPersistentInfFlagBuffer();
}
