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
  getWebGPUDevice,
  getMaxStorageBufferBindingSize,
  trackSharedEncoderWrite,
  createParamsBuffer,
  releaseParamsBuffer,
  flushSharedEncoder,
  cachedCreateBindGroup,
} from "./index";
import { gpuMemoryTracker } from "./memory-tracker";

// WebGPU type definitions (runtime, not importable at compile time)
type GPUBuffer = {
  size: number;
  usage: number;
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
};

type GPUDevice = {
  createShaderModule(descriptor: { code: string }): GPUShaderModule;
  createComputePipeline(descriptor: {
    layout: "auto";
    compute: { module: GPUShaderModule; entryPoint: string };
  }): GPUComputePipeline;
  createBindGroup(descriptor: {
    layout: GPUBindGroupLayout;
    entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer; offset?: number; size?: number };
    }>;
  }): GPUBindGroup;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer;
  createCommandEncoder(): GPUCommandEncoder;
  queue: {
    writeBuffer(
      buffer: GPUBuffer,
      offset: number,
      data: ArrayBufferView,
    ): void;
    submit(commandBuffers: GPUCommandBuffer[]): void;
    onSubmittedWorkDone?(): Promise<void>;
  };
};

type GPUShaderModule = object;
type GPUComputePipeline = {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
};
type GPUBindGroupLayout = object;
type GPUBindGroup = object;
type GPUCommandEncoder = {
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): GPUCommandBuffer;
};
type GPUCommandBuffer = object;

const GPUBufferUsage = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

const GPUMapMode = {
  READ: 0x0001,
};

const WORKGROUP_SIZE = 256;
const MAX_WORKGROUPS_PER_DIM = 65535;

// ============================================================================
// Pipeline Cache
// ============================================================================

const pipelineCache = new Map<string, GPUComputePipeline>();

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  code: string,
): GPUComputePipeline {
  let pipeline = pipelineCache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({ code });
    pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    pipelineCache.set(key, pipeline);
  }
  return pipeline;
}

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

  return createParamsBuffer(device as any, u32);
}

// ============================================================================
// Inf Flag Buffer
// ============================================================================

/**
 * Allocate a 4-byte atomic buffer for inf detection, initialized to 0.0.
 * Shared across all parameter dispatches in one unscale_() call.
 *
 * Uses Float32Array([0.0]) to write zero bits. The kernel writes
 * bitcast<u32>(1.0f) = 0x3F800000 via atomicMax, so readback interprets
 * the raw bytes as Float32Array → 0.0 (clean) or 1.0 (inf found).
 */
export function allocateInfFlagBuffer(device?: GPUDevice): GPUBuffer {
  const dev = device ?? (getWebGPUDevice()!.device as unknown as GPUDevice);
  const buffer = dev.createBuffer({
    size: 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  dev.queue.writeBuffer(buffer, 0, new Float32Array([0.0]));
  return buffer;
}

/**
 * Read the inf flag buffer and destroy it.
 * Returns 0.0 (all finite) or 1.0 (inf/NaN found).
 *
 * Flushes the shared encoder first to ensure all kernel writes are submitted.
 */
export async function readInfFlag(infFlagBuffer: GPUBuffer): Promise<number> {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  // Flush shared encoder to ensure kernel writes are submitted
  flushSharedEncoder();

  // Create staging buffer for readback
  const staging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(infFlagBuffer, 0, staging, 0, 4);
  device.queue.submit([encoder.finish()]);

  if (typeof device.queue.onSubmittedWorkDone === "function") {
    await device.queue.onSubmittedWorkDone();
  }

  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  const value = new Float32Array(mapped.slice(0, 4))[0];
  staging.unmap();
  staging.destroy();

  // Destroy the inf flag buffer
  infFlagBuffer.destroy();

  return value;
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
  gpuMemoryTracker.trackAllocation(buf as any, sizeBytes);
  trackSharedEncoderWrite(buf as any);
  return buf;
}

// ============================================================================
// Dispatch
// ============================================================================

export interface UnscaleGradResult {
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
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

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
  const alignedBytes = alignBufferSize(totalBytes);
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
  const pipeline = getOrCreatePipeline(device, key, code);

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

    let bindGroup: any;
    if (needsChunking) {
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
    } else {
      bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
        [gradBuffer, gradOut, infFlagBuffer, configBuf] as any);
    }

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D
      ? Math.ceil(chunkWorkgroups / dispatchX)
      : 1;

    dispatchComputePass(
      pipeline as any,
      bindGroup as any,
      dispatchX,
      dispatchY,
    );

    // Config buffer deferred destruction (shared encoder still active)
    releaseParamsBuffer(configBuf as any);
  }

  return { gradOutBuffer: gradOut };
}

// ============================================================================
// Helpers
// ============================================================================

/** Round up buffer size to next power of 2 (matching pool size classes) */
function alignBufferSize(size: number): number {
  if (size <= 256) return 256;
  let s = 1;
  while (s < size) s <<= 1;
  return s;
}
