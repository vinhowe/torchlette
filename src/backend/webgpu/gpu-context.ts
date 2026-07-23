/**
 * GPU context management: initialization, device lifecycle, f16 support.
 */

import { ENV } from "../../core/env";
import { clearBindGroupCache } from "./bind-group-cache";
import { bufferPool, destroyProfilingFenceBuffer } from "./buffer-pool";
import { DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE } from "./shape-utils";
import { advanceEpoch } from "./epoch";
import type {
  GPUAdapter,
  GPUBuffer,
  GPUDevice,
  GPUQueue,
  WebGPUModule,
  WebGPUProvider,
} from "./gpu-types";
import { type SubgroupSupport, setSubgroupSupport } from "./matmul/types";
import {
  clearWarmupCache,
  startPipelineRecording,
  stopPipelineRecording,
  warmupPipelines,
} from "./pipeline-warmup";
import { initGpuTimestamps, isProfilingEnabled } from "./profiler";
import { setSharedEncoderEnabled } from "./shared-encoder";
import {
  gpuContext,
  requireContext,
  runTeardownCallbacks,
  setGpuContext,
} from "./webgpu-state";

// Re-export from webgpu-state
export { requireContext } from "./webgpu-state";

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
    const mod = (await import("webgpu")) as unknown as WebGPUModule;
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
  const raw = ENV.TORCHLETTE_WEBGPU_OPTS ?? "";
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
function f32ToF16(value: number): number {
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
function f16ToF32(h: number): number {
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
      f = (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
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
    f = (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
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

/**
 * Acquire a GPU adapter from the browser or Node.js WebGPU module.
 * Returns the adapter and provider, or sets lastInitError and returns null.
 */
async function acquireAdapter(): Promise<{
  adapter: GPUAdapter;
  provider: WebGPUProvider;
} | null> {
  if (isBrowserWithWebGPU()) {
    const gpu = (navigator as { gpu: WebGPUProvider }).gpu;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        lastInitError = "No WebGPU adapter found";
        return null;
      }
      return { adapter, provider: gpu };
    } catch (error) {
      lastInitError = `WebGPU requestAdapter failed: ${error instanceof Error ? error.message : "Unknown error"}`;
      return null;
    }
  }

  // Fall back to Node.js webgpu module
  const mod = await loadWebGPU();
  if (!mod) {
    lastInitError = "webgpu module not available";
    return null;
  }
  Object.assign(globalThis, mod.globals);
  const options = parseWebGPUOptions();
  const nodeProvider = mod.create(options);
  if (!nodeProvider) {
    lastInitError = "webgpu create() returned no provider";
    return null;
  }
  try {
    const adapter = await nodeProvider.requestAdapter();
    if (!adapter) {
      lastInitError =
        `No WebGPU adapter found` +
        (options.length > 0 ? ` (options: ${options.join(", ")})` : "");
      return null;
    }
    return { adapter, provider: nodeProvider };
  } catch (error) {
    lastInitError = `WebGPU requestAdapter failed: ${error instanceof Error ? error.message : "Unknown error"}`;
    return null;
  }
}

/**
 * Request a GPU device with cascading feature fallback.
 * Tries all features first, then without f16, then without any features.
 */
async function requestDeviceWithFallback(
  adapter: GPUAdapter,
  f16Supported: boolean,
  subgroupSupport: SubgroupSupport,
): Promise<{ device: GPUDevice; actualF16Supported: boolean } | null> {
  const adapterMaxStorage =
    adapter.limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  const adapterMaxBuffer = adapter.limits?.maxBufferSize ?? 256 * 1024 * 1024;
  const adapterMaxStorageBuffers =
    adapter.limits?.maxStorageBuffersPerShaderStage ?? 8;
  // The default maxComputeWorkgroupStorageSize is 16KB. The flash-attention
  // kernel's shared-memory tiles scale with headDim; head_dim=256 (Gemma-2)
  // needs 32KB and OVERFLOWS the 16KB default → the pipeline is invalid and
  // every attention submit is DROPPED (silently zeroed downstream). Request the
  // adapter's supported max (A100/V100 report 48KB). Harmless where headDim is
  // small (headDim=128 fits 16KB); required for Gemma-2's 256-dim heads.
  const adapterMaxWgStorage =
    adapter.limits?.maxComputeWorkgroupStorageSize ?? 16384;

  if (isProfilingEnabled()) {
    console.log(
      `[WebGPU] Adapter limits: maxStorageBufferBindingSize=${adapterMaxStorage}, maxBufferSize=${adapterMaxBuffer}, maxStorageBuffersPerShaderStage=${adapterMaxStorageBuffers}, maxComputeWorkgroupStorageSize=${adapterMaxWgStorage}`,
    );
  }

  const requiredLimits: Record<string, number> = {
    maxStorageBufferBindingSize: adapterMaxStorage,
    maxBufferSize: adapterMaxBuffer,
    maxStorageBuffersPerShaderStage: adapterMaxStorageBuffers,
    maxComputeWorkgroupStorageSize: adapterMaxWgStorage,
  };

  // Attempt 1: all features
  try {
    const requiredFeatures: string[] = [];
    if (subgroupSupport.supported) requiredFeatures.push("subgroups");
    if (f16Supported) requiredFeatures.push("shader-f16");
    // Always request timestamp-query if available — profiling can be
    // enabled at runtime (e.g., from the browser console) and the device
    // can't be re-created with new features after initialization.
    if (adapter.features?.has("timestamp-query")) {
      requiredFeatures.push("timestamp-query");
    }
    const device = await adapter.requestDevice({
      requiredFeatures:
        requiredFeatures.length > 0 ? requiredFeatures : undefined,
      requiredLimits,
    });
    return { device, actualF16Supported: f16Supported };
  } catch {
    // Fall through
  }

  // Attempt 2: without f16
  try {
    const fallbackFeatures: string[] = [];
    if (subgroupSupport.supported) fallbackFeatures.push("subgroups");
    const device = await adapter.requestDevice({
      requiredFeatures:
        fallbackFeatures.length > 0 ? fallbackFeatures : undefined,
      requiredLimits,
    });
    return { device, actualF16Supported: false };
  } catch {
    // Fall through
  }

  // Attempt 3: no features, just limits
  try {
    const device = await adapter.requestDevice({ requiredLimits });
    setSubgroupSupport({ supported: false });
    return { device, actualF16Supported: false };
  } catch (finalError) {
    lastInitError = `WebGPU requestDevice failed: ${finalError instanceof Error ? finalError.message : "Unknown error"}`;
    return null;
  }
}

let _gpuUncapturedErrorCount = 0;
// High-water mark for the step-boundary dropped-submit guard (task #94, item 3).
let _lastCheckedUncapturedErrorCount = 0;

/** Total uncaptured GPU device errors since init (each one = a dropped submit). */
export function getGpuUncapturedErrorCount(): number {
  return _gpuUncapturedErrorCount;
}

/**
 * LOUD dropped-submit guard (task #94, item 3).
 *
 * A memory-pressured device (VkOOM) drops the submit an uncaptured error occurs
 * in: none of that work runs, downstream reads see stale/all-zero data, and
 * training silently continues on garbage (the VULKAN_DEVICE_INDEX=1 incident —
 * all-zero runs, no crash). The onuncapturederror handler counts these, but its
 * in-callback throw does NOT propagate to the training loop's control flow
 * (it fires out-of-band on the event loop), so under memory pressure the failure
 * was silent zeros rather than an error.
 *
 * This is the synchronous, in-band detector: called at fence / readback points
 * (after submits have been observed by the device), it compares the uncaptured
 * error count to the last check and — under TORCHLETTE_STRICT_GPU — THROWS,
 * naming device pressure, at a deterministic point in the loop. It rides the
 * existing STRICT_GPU flag (no new env flag). Without STRICT_GPU it silently
 * advances the high-water mark (the existing console.error already logs each
 * error), preserving the default behavior.
 */
export function assertNoDroppedSubmits(context: string): void {
  if (_gpuUncapturedErrorCount === _lastCheckedUncapturedErrorCount) return;
  const dropped = _gpuUncapturedErrorCount - _lastCheckedUncapturedErrorCount;
  _lastCheckedUncapturedErrorCount = _gpuUncapturedErrorCount;
  if (ENV.TORCHLETTE_STRICT_GPU === "1") {
    throw new Error(
      `TORCHLETTE_STRICT_GPU: ${dropped} GPU submit(s) were DROPPED before ${context} ` +
        `(uncaptured device error — almost always device memory pressure / VkOOM). ` +
        `Downstream reads would see stale/all-zero data; failing loudly instead of ` +
        `training silently on garbage. Free device memory (fewer tenants / smaller ` +
        `model / lower batch) or select a free device (VULKAN_DEVICE_INDEX + tools/vk-shim).`,
    );
  }
}

/**
 * TEST-ONLY: simulate a dropped submit by bumping the uncaptured-error count,
 * so the item-3 submit-drop guard can be exercised without a real VkOOM. Not
 * used in production paths.
 */
export function _simulateDroppedSubmitForTest(): void {
  _gpuUncapturedErrorCount++;
}

export type InitWebGPUOptions = {
  /**
   * Run torchlette on a caller-owned GPUDevice instead of requesting one.
   * This is the interop path: WebGPU resources cannot cross devices, so a
   * renderer that wants to bind torchlette tensor buffers (or vice versa)
   * must share ONE device. Torchlette will not destroy an external device at
   * teardown, and it chains — not clobbers — any onuncapturederror handler
   * already installed.
   *
   * Create the device with `webgpuDeviceRequirements(adapter)` merged into
   * your descriptor, or torchlette runs degraded: without "shader-f16" all
   * f16 paths fall back to f32, and with default limits (256MB maxBufferSize)
   * large-model weights cannot be allocated.
   */
  device?: GPUDevice;
};

/**
 * The device-descriptor pieces a renderer should merge into its own
 * `adapter.requestDevice()` call so the shared device satisfies torchlette:
 * features gated on adapter support (subgroups, shader-f16, timestamp-query)
 * and the storage/buffer limits raised to the adapter maximums.
 */
export function webgpuDeviceRequirements(adapter: GPUAdapter): {
  requiredFeatures: string[];
  requiredLimits: Record<string, number>;
} {
  const requiredFeatures: string[] = [];
  if (adapter.features?.has("subgroups")) requiredFeatures.push("subgroups");
  if (adapter.features?.has("shader-f16")) requiredFeatures.push("shader-f16");
  if (adapter.features?.has("timestamp-query")) {
    requiredFeatures.push("timestamp-query");
  }
  return {
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize:
        adapter.limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE,
      maxBufferSize: adapter.limits?.maxBufferSize ?? 256 * 1024 * 1024,
      maxStorageBuffersPerShaderStage:
        adapter.limits?.maxStorageBuffersPerShaderStage ?? 8,
      // head_dim=256 attention needs 32KB shared memory (> the 16KB default).
      maxComputeWorkgroupStorageSize:
        adapter.limits?.maxComputeWorkgroupStorageSize ?? 16384,
    },
  };
}

type UncapturedErrorHandler =
  | ((ev: { error: { message: string } }) => void)
  | null;

/** External device's pre-existing error handler, restored at destroyWebGPU. */
let _chainedPriorErrorHandler: UncapturedErrorHandler = null;

export async function initWebGPU(
  options?: InitWebGPUOptions,
): Promise<boolean> {
  if (gpuContext) {
    if (options?.device && gpuContext.device !== options.device) {
      throw new Error(
        "initWebGPU: already initialized with a different device. " +
          "Call destroyWebGPU() first, or initialize once with the shared device.",
      );
    }
    return true;
  }
  lastInitError = null;

  let device: GPUDevice;
  let provider: WebGPUProvider | null;
  let actualF16Supported: boolean;
  let externalDevice = false;

  if (options?.device) {
    // External (caller-owned) device: derive capabilities from the DEVICE's
    // enabled features — the adapter is not ours to inspect.
    device = options.device;
    provider = null;
    externalDevice = true;
    const features = (
      device as unknown as { features?: { has(s: string): boolean } }
    ).features;
    const subgroupSupport: SubgroupSupport = features?.has("subgroups")
      ? { supported: true, subgroupSize: 32 }
      : { supported: false };
    setSubgroupSupport(subgroupSupport);
    actualF16Supported = features?.has("shader-f16") ?? false;
  } else {
    const acquired = await acquireAdapter();
    if (!acquired) return false;
    const { adapter } = acquired;
    provider = acquired.provider;

    const subgroupSupport: SubgroupSupport = adapter.features?.has("subgroups")
      ? { supported: true, subgroupSize: 32 }
      : { supported: false };
    setSubgroupSupport(subgroupSupport);

    const f16Supported = adapter.features?.has("shader-f16") ?? false;

    const deviceResult = await requestDeviceWithFallback(
      adapter,
      f16Supported,
      subgroupSupport,
    );
    if (!deviceResult) return false;
    device = deviceResult.device;
    actualF16Supported = deviceResult.actualF16Supported;
  }

  setGpuContext({
    provider,
    device,
    queue: device.queue,
    pipelines: new Map(),
    f16Supported: actualF16Supported,
    externalDevice,
  });

  // LOUD GPU ERRORS. A WebGPU validation error rejects the ENTIRE submit it
  // occurs in — none of that work executes, buffers keep their old contents,
  // and training silently continues on stale values. Dawn prints details to
  // stderr but the process never learns anything happened; four separate
  // silent-training-corruption bugs (destroyed-buffer binds, read/rw
  // aliasing, multi-writable-binding aliasing) were each found only by loss
  // archaeology. This handler makes the class observable: every uncaptured
  // device error is counted and clearly attributed, and
  // TORCHLETTE_STRICT_GPU=1 turns the first one into a crash.
  const deviceWithHandler = device as unknown as {
    onuncapturederror: UncapturedErrorHandler;
  };
  // On a SHARED device the renderer may already have a handler installed —
  // chain it (torchlette first for counting/strict, then theirs), and restore
  // it at destroyWebGPU. Clobbering it would silence the host app's own
  // error reporting.
  const priorHandler = externalDevice ? deviceWithHandler.onuncapturederror : null;
  _chainedPriorErrorHandler = priorHandler;
  deviceWithHandler.onuncapturederror = (ev) => {
    _gpuUncapturedErrorCount++;
    if (_gpuUncapturedErrorCount <= 10) {
      console.error(
        `[torchlette] GPU UNCAPTURED ERROR #${_gpuUncapturedErrorCount} (the enclosing submit was DROPPED — downstream reads see stale data): ${ev.error?.message ?? ev.error}`,
      );
      if (_gpuUncapturedErrorCount === 10) {
        console.error(
          "[torchlette] further GPU errors will be counted but not logged (see getGpuUncapturedErrorCount())",
        );
      }
    }
    priorHandler?.(ev);
    if (
      ENV.TORCHLETTE_STRICT_GPU === "1"
    ) {
      throw new Error(
        `TORCHLETTE_STRICT_GPU: uncaptured GPU error: ${ev.error?.message ?? ev.error}`,
      );
    }
  };

  if (
    ENV.TORCHLETTE_POOL_BUDGET_MB
  ) {
    const mb = Number(ENV.TORCHLETTE_POOL_BUDGET_MB);
    if (Number.isFinite(mb) && mb > 0) {
      bufferPool.setMaxPoolBytes(mb * 1024 * 1024);
    }
  }

  if (
    isProfilingEnabled() &&
    (
      device as unknown as { features: { has(s: string): boolean } }
    ).features.has("timestamp-query")
  ) {
    initGpuTimestamps(device);
  }

  const batchSubmits =
    typeof process !== "undefined"
      ? ENV.TORCHLETTE_BATCH_SUBMITS
      : undefined;
  setSharedEncoderEnabled(batchSubmits !== "0");

  return true;
}

// ============================================================================
// syncWebGPU / destroyWebGPU / getWebGPUDevice / requireContext
// ============================================================================

export async function syncWebGPU(): Promise<void> {
  const ctx = requireContext();
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  // Fence settled: surface any dropped submit loudly under STRICT_GPU rather
  // than letting the loop proceed on stale data (task #94, item 3).
  assertNoDroppedSubmits("syncWebGPU fence");
  // Fence completed — quiescent point: advance the engine epoch
  // (flushes pending pool buffers).
  advanceEpoch("syncWebGPU");
}

/**
 * Destroy the WebGPU device and release all GPU resources.
 * After calling this, the Node.js process can exit cleanly without process.exit().
 * Safe to call multiple times (no-op if already destroyed or never initialized).
 */
export function destroyWebGPU(): void {
  if (!gpuContext) return;
  // Destroy cached f16 weight buffers (teardown — the device is going away, so
  // immediate destroy is correct; no pending submit survives device destruction).
  for (const buf of f16WeightCache.values()) {
    buf.destroy();
  }
  f16WeightCache.clear();
  clearBindGroupCache();
  runTeardownCallbacks();
  clearWarmupCache();
  destroyProfilingFenceBuffer();
  if (gpuContext.externalDevice) {
    // Caller-owned device: never destroy it; hand its error handler back.
    (
      gpuContext.device as unknown as {
        onuncapturederror: UncapturedErrorHandler;
      }
    ).onuncapturederror = _chainedPriorErrorHandler;
    _chainedPriorErrorHandler = null;
  } else {
    (gpuContext.device as unknown as { destroy(): void }).destroy();
  }
  gpuContext.pipelines.clear();
  setGpuContext(null);
  // A destroyed device's error count must not bleed into the next device's
  // dropped-submit guard (task #94, item 2/3 multi-engine reclaim).
  _gpuUncapturedErrorCount = 0;
  _lastCheckedUncapturedErrorCount = 0;
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
  const limits = ctx.device.limits;
  return limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
}

// ============================================================================
// Pipeline Warmup
// ============================================================================

/**
 * Run fn() once (typically a full training step), recording all pipeline
 * compilations. Then return the registry for future warmup.
 *
 * On the first call, pipelines compile synchronously as normal.
 * Pass the returned registry to warmupPipelines() before step 0 on future runs.
 */
export async function warmupFromStep(
  fn: () => void | Promise<void>,
): Promise<Array<{ key: string; wgsl: string }>> {
  startPipelineRecording();
  await fn();
  return stopPipelineRecording();
}

/**
 * Pre-compile all pipelines from a registry in parallel.
 * Convenience wrapper that calls warmupPipelines with the current device.
 */
export async function warmupFromRegistry(
  entries: Array<{ key: string; wgsl: string }>,
): Promise<{ compiled: number; skipped: number; timeMs: number }> {
  const ctx = requireContext();
  return warmupPipelines(ctx.device, entries);
}
