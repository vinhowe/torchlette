/**
 * GPU context management: initialization, device lifecycle, f16 support.
 * Extracted from index.ts — purely structural refactoring.
 */

import type {
  GPUBuffer,
  GPUDevice,
  GPUQueue,
  GPUAdapter,
  GPUAdapterLimits,
  GPUComputePipeline,
  WebGPUProvider,
  WebGPUModule,
  WebGPUContext,
  WebGPUTensor,
} from "./gpu-types";
import { gpuContext, setGpuContext, requireContext } from "./webgpu-state";
import { bufferPool } from "./buffer-pool";
import { isProfilingEnabled, initGpuTimestamps } from "./profiler";
import { setSubgroupSupport, type SubgroupSupport } from "./matmul";
import { registerWebGPUDonation } from "../../engine/memory-planned-executor";
import { destroyPersistentInfFlagBuffer } from "./unscale-kernel";
import { destroyProfilingFenceBuffer } from "./buffer-pool";

import { donateBuffer, getBufferSize } from "./buffer-arena";
import { setSharedEncoderEnabled } from "./shared-encoder";
import { clearBindGroupCache } from "./bind-group-cache";

// Re-exports from webgpu-state for backward compatibility
export { gpuContext as context, requireContext } from "./webgpu-state";
let lastInitError: string | null = null;

// ============================================================================
// F16 Weight Cache
// ============================================================================

/**
 * Cache of f16 weight buffers produced by the Adam kernel's dual-write.
 * Keyed by the f32 param GPUBuffer → corresponding f16 GPUBuffer.
 * Checked in cast() to skip standalone f32→f16 dispatches for AMP weights.
 */
export const f16WeightCache = new Map<GPUBuffer, GPUBuffer>();

/** Set an entry in the f16 weight cache (used by packed Adam). */
export function setF16WeightCacheEntry(paramBuffer: GPUBuffer, f16Buffer: GPUBuffer): void {
  f16WeightCache.set(paramBuffer, f16Buffer);
}

/** Evict and optionally destroy an f16 weight cache entry (used by packed Adam). */
export function evictF16WeightCacheEntry(paramBuffer: GPUBuffer): GPUBuffer | undefined {
  const old = f16WeightCache.get(paramBuffer);
  if (old) {
    f16WeightCache.delete(paramBuffer);
  }
  return old;
}

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Check if we're running in a browser environment with native WebGPU.
 */
function isBrowserWithWebGPU(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof (navigator as { gpu?: unknown }).gpu !== "undefined"
  );
}

async function loadWebGPU(): Promise<WebGPUModule | null> {
  // In browser, we use navigator.gpu directly, so skip the Node.js module
  if (isBrowserWithWebGPU()) {
    return null;
  }
  try {
    const mod = (await import("webgpu")) as WebGPUModule;
    return mod;
  } catch {
    return null;
  }
}

function parseWebGPUOptions(): string[] {
  // In browser, we don't have process.env
  if (typeof process === "undefined") {
    return [];
  }
  const raw = process.env.TORCHLETTE_WEBGPU_OPTS ?? "";
  const options = raw
    .split(",")
    .map((value) => value.trim())
    .filter((value) => value.length > 0);
  if (
    process.platform === "darwin" &&
    !options.some((value) => value.startsWith("backend="))
  ) {
    options.unshift("backend=metal");
  }
  // Enable f16 on NVIDIA Vulkan (Dawn blocks it by default due to CTS test issues,
  // but the hardware supports it fine for compute workloads)
  if (
    process.platform === "linux" &&
    !options.some((value) => value.includes("vulkan_enable_f16_on_nvidia"))
  ) {
    options.push("enable-dawn-features=vulkan_enable_f16_on_nvidia");
  }
  return options;
}

// ============================================================================
// Public API: Initialization Error
// ============================================================================

export function getWebGPUInitError(): string | null {
  return lastInitError;
}

// ============================================================================
// F16 Support Check
// ============================================================================

/**
 * Check if f16 (half precision) is supported on the current device.
 * Returns false if WebGPU is not initialized.
 */
export function isF16Supported(): boolean {
  return gpuContext?.f16Supported ?? false;
}

// ============================================================================
// F16 Conversion Functions
// ============================================================================

/**
 * Convert a f32 value to f16 (IEEE 754 half-precision).
 * Returns a 16-bit unsigned integer representing the f16 value.
 */
export function f32ToF16(value: number): number {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);

  floatView[0] = value;
  const f = int32View[0];

  const sign = (f >>> 31) & 0x1;
  const exp = (f >>> 23) & 0xff;
  const frac = f & 0x7fffff;

  let newExp: number;
  let newFrac: number;

  if (exp === 0xff) {
    // Inf or NaN
    newExp = 0x1f;
    newFrac = frac ? 0x200 : 0; // NaN preserves some bits, Inf is 0
  } else if (exp === 0) {
    // Zero or denormal - becomes zero in f16
    newExp = 0;
    newFrac = 0;
  } else {
    // Normalized value
    const unbiasedExp = exp - 127;
    if (unbiasedExp < -24) {
      // Too small, becomes zero
      newExp = 0;
      newFrac = 0;
    } else if (unbiasedExp < -14) {
      // Denormalized f16
      newExp = 0;
      const shift = -14 - unbiasedExp;
      newFrac = (0x400 | (frac >>> 13)) >>> shift;
    } else if (unbiasedExp > 15) {
      // Overflow to infinity
      newExp = 0x1f;
      newFrac = 0;
    } else {
      // Normal f16
      newExp = unbiasedExp + 15;
      newFrac = frac >>> 13;
    }
  }

  return (sign << 15) | (newExp << 10) | newFrac;
}

/**
 * Convert a f16 value (16-bit unsigned int) to f32.
 */
export function f16ToF32(h: number): number {
  const sign = (h >>> 15) & 0x1;
  const exp = (h >>> 10) & 0x1f;
  const frac = h & 0x3ff;

  let f: number;
  if (exp === 0) {
    if (frac === 0) {
      // Zero
      f = sign ? -0 : 0;
    } else {
      // Denormalized
      f = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
    }
  } else if (exp === 0x1f) {
    if (frac === 0) {
      // Infinity
      f = sign ? -Infinity : Infinity;
    } else {
      // NaN
      f = NaN;
    }
  } else {
    // Normalized
    f = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }
  return f;
}

/**
 * Convert an array of f32 values to a Uint16Array of f16 values.
 */
export function f32ArrayToF16Array(values: number[]): Uint16Array {
  const result = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    result[i] = f32ToF16(values[i]);
  }
  return result;
}

/**
 * Convert a Uint16Array of f16 values to an array of f32 values.
 */
export function f16ArrayToF32Array(data: Uint16Array): number[] {
  const result = new Array<number>(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = f16ToF32(data[i]);
  }
  return result;
}

// ============================================================================
// initWebGPU
// ============================================================================

export async function initWebGPU(): Promise<boolean> {
  if (gpuContext) {
    return true;
  }
  lastInitError = null;

  let adapter: GPUAdapter | null = null;
  let provider: WebGPUProvider;

  // Try browser-native WebGPU first
  if (isBrowserWithWebGPU()) {
    const gpu = (navigator as { gpu: WebGPUProvider }).gpu;
    provider = gpu;
    try {
      adapter = await gpu.requestAdapter();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      lastInitError = `WebGPU requestAdapter failed: ${message}`;
      return false;
    }
  } else {
    // Fall back to Node.js webgpu module
    const mod = await loadWebGPU();
    if (!mod) {
      lastInitError = "webgpu module not available";
      return false;
    }
    Object.assign(globalThis, mod.globals);
    const options = parseWebGPUOptions();
    const nodeProvider = mod.create(options);
    if (!nodeProvider) {
      lastInitError = "webgpu create() returned no provider";
      return false;
    }
    provider = nodeProvider;
    try {
      adapter = await nodeProvider.requestAdapter();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      lastInitError = `WebGPU requestAdapter failed: ${message}`;
      return false;
    }
    if (!adapter) {
      lastInitError =
        `No WebGPU adapter found` +
        (options.length > 0 ? ` (options: ${options.join(", ")})` : "");
      return false;
    }
  }

  if (!adapter) {
    lastInitError = "No WebGPU adapter found";
    return false;
  }
  // Check for subgroup support
  const subgroupSupport = detectSubgroupSupport(adapter);
  setSubgroupSupport(subgroupSupport);

  // Check for shader-f16 support
  const f16Supported = adapter.features?.has("shader-f16") ?? false;

  let device: GPUDevice;
  let actualF16Supported = f16Supported;

  // Request higher maxStorageBufferBindingSize to support large model buffers
  // Default is 128MB, but GPT-2 embeddings can be 150MB+
  const adapterMaxStorage =
    adapter.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const adapterMaxBuffer =
    (adapter.limits as Record<string, number>)?.maxBufferSize ?? 256 * 1024 * 1024;
  const adapterMaxStorageBuffers =
    (adapter.limits as Record<string, number>)?.maxStorageBuffersPerShaderStage ?? 8;

  console.log(`[WebGPU] Adapter limits: maxStorageBufferBindingSize=${adapterMaxStorage}, maxBufferSize=${adapterMaxBuffer}, maxStorageBuffersPerShaderStage=${adapterMaxStorageBuffers}`);

  const requiredLimits: Record<string, number> = {
    maxStorageBufferBindingSize: adapterMaxStorage,
    maxBufferSize: adapterMaxBuffer,
    maxStorageBuffersPerShaderStage: adapterMaxStorageBuffers,
  };

  try {
    // Request device with optional features and higher limits
    const requiredFeatures: string[] = [];
    if (subgroupSupport.supported) {
      requiredFeatures.push("subgroups");
    }
    if (f16Supported) {
      requiredFeatures.push("shader-f16");
    }
    if (isProfilingEnabled() && adapter.features?.has("timestamp-query")) {
      requiredFeatures.push("timestamp-query");
    }
    device = await adapter.requestDevice({
      requiredFeatures:
        requiredFeatures.length > 0 ? requiredFeatures : undefined,
      requiredLimits,
    });
  } catch (error) {
    // If features failed, try with fewer features
    try {
      // Try without f16 first
      const fallbackFeatures: string[] = [];
      if (subgroupSupport.supported) {
        fallbackFeatures.push("subgroups");
      }
      device = await adapter.requestDevice({
        requiredFeatures:
          fallbackFeatures.length > 0 ? fallbackFeatures : undefined,
        requiredLimits,
      });
      actualF16Supported = false;
    } catch (retryError) {
      // Try without any features but still with higher limits
      try {
        device = await adapter.requestDevice({ requiredLimits });
        setSubgroupSupport({ supported: false });
        actualF16Supported = false;
      } catch (finalError) {
        const message =
          finalError instanceof Error ? finalError.message : "Unknown error";
        lastInitError = `WebGPU requestDevice failed: ${message}`;
        return false;
      }
    }
  }
  setGpuContext({
    provider,
    device,
    queue: device.queue,
    pipelines: new Map(),
    f16Supported: actualF16Supported,
  });
  // Set queue on buffer pool for fence integration (§14)
  bufferPool.setQueue(device.queue);

  // Apply pool budget from env var if set (e.g. TORCHLETTE_POOL_BUDGET_MB=2000)
  if (typeof process !== "undefined" && process.env?.TORCHLETTE_POOL_BUDGET_MB) {
    const mb = Number(process.env.TORCHLETTE_POOL_BUDGET_MB);
    if (Number.isFinite(mb) && mb > 0) {
      bufferPool.setMaxPoolBytes(mb * 1024 * 1024);
    }
  }

  // Initialize GPU timestamp profiling if supported
  if (isProfilingEnabled() && device.features.has("timestamp-query")) {
    initGpuTimestamps(device);
  }

  // Register donation functions for memory planning
  registerWebGPUDonation(donateBuffer, getBufferSize);

  // Enable shared encoder for command buffer consolidation
  // Set TORCHLETTE_BATCH_SUBMITS=0 to disable for debugging
  const batchSubmits = typeof process !== "undefined"
    ? process.env?.TORCHLETTE_BATCH_SUBMITS
    : undefined;
  setSharedEncoderEnabled(batchSubmits !== "0");

  return true;
}

// ============================================================================
// Subgroup Detection
// ============================================================================

/**
 * Detect subgroup support from the GPU adapter.
 */
function detectSubgroupSupport(adapter: GPUAdapter): SubgroupSupport {
  // Check if adapter has features and if subgroups is in the set
  if (adapter.features?.has("subgroups")) {
    // Typical subgroup sizes: 32 (NVIDIA/AMD), 16 (Intel/mobile)
    // We assume 32 as default since that's most common
    return { supported: true, subgroupSize: 32 };
  }
  return { supported: false };
}

// ============================================================================
// syncWebGPU / destroyWebGPU / getWebGPUDevice / requireContext
// ============================================================================

export async function syncWebGPU(): Promise<void> {
  const ctx = requireContext();
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  bufferPool.flushPendingToAvailable();
}

/**
 * Destroy the WebGPU device and release all GPU resources.
 * After calling this, the Node.js process can exit cleanly without process.exit().
 * Safe to call multiple times (no-op if already destroyed or never initialized).
 */
export function destroyWebGPU(): void {
  if (!gpuContext) return;
  // Destroy cached f16 weight buffers
  for (const buf of f16WeightCache.values()) {
    buf.destroy();
  }
  f16WeightCache.clear();
  clearBindGroupCache();
  destroyPersistentInfFlagBuffer();
  destroyProfilingFenceBuffer();
  gpuContext.device.destroy();
  gpuContext.pipelines.clear();
  setGpuContext(null);
}

/**
 * Get the raw WebGPU device and queue for advanced use cases (benchmarking, etc).
 * Returns null if WebGPU is not initialized.
 */
export function getWebGPUDevice(): {
  device: GPUDevice;
  queue: GPUQueue;
} | null {
  if (!gpuContext) return null;
  return { device: gpuContext.device, queue: gpuContext.queue };
}

/**
 * Get the maximum storage buffer binding size from the device.
 * Used to determine when chunked operations are needed for large tensors.
 */
export function getMaxStorageBufferBindingSize(): number {
  const ctx = requireContext();
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  return limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
}
