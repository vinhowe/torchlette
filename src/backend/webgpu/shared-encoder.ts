/**
 * Shared Encoder — Command Buffer Consolidation
 *
 * Multiple compute passes are encoded into a shared GPUCommandEncoder to reduce
 * queue.submit() calls. A write-set tracks all buffers written during the current
 * encoder scope so the buffer pool never returns a buffer that was already written
 * in this scope (preventing the aliasing corruption that broke previous attempts).
 *
 * Ops that create their own encoders (matmul, reductions) flush the shared encoder
 * first, then their command buffer is collected and submitted together at the end.
 *
 * Also includes Batch Execution Context for true segmented execution and
 * step-level scope management (beginStep/endStep).
 */

import type { GPUBuffer, GPUCommandBuffer, GPUCommandEncoder, GPUComputePipeline, GPUBindGroup, WebGPUContext } from "./gpu-types";
import { requireContext } from "./gpu-context";
import { bufferPool, awaitDeferredFence } from "./buffer-pool";
import { resolveGpuTimestamps, profileApiCall } from "./profiler";
import { getSizeClass } from "../../engine/memory-planning";

import { replayPinnedBufferSet } from "./dispatch-recording";
import { paramsBufferSizeClass, paramsBufferPools, MAX_PARAMS_POOL_SIZE_PER_CLASS, resetDispatchSequence } from "./bind-group-cache";
import { prePinOutputBuffers, pinnedOutputBuffers } from "./buffer-arena";

// ============================================================================
// Batch Execution Context for True Segmented Execution
// ============================================================================

/**
 * Batch execution context - collects command buffers for deferred submission.
 *
 * Instead of using a single shared encoder (which causes validation errors
 * when ops have data dependencies), we collect separate command buffers
 * and submit them all together at the end.
 */
export interface BatchExecutionContext {
  /** Collected command buffers to submit together */
  commandBuffers: GPUCommandBuffer[];
  /** Buffers to destroy after the batch submits (deferred from mid-batch destroy calls) */
  deferredDestroyBuffers: GPUBuffer[];
}

/** Active batch context (null when in immediate mode) */
export let activeBatch: BatchExecutionContext | null = null;

/**
 * Begin batched execution. Subsequent ops will collect command buffers
 * instead of submitting immediately. Call endBatchExecution() to submit
 * all collected work and wait for GPU completion.
 */
export function beginBatchExecution(): void {
  if (activeBatch) {
    throw new Error("Batch execution already active");
  }
  activeBatch = {
    commandBuffers: [],
    deferredDestroyBuffers: [],
  };
}

/**
 * End batch, submit all collected command buffers, and WAIT for GPU completion.
 * This is the key synchronization point for true segmented execution.
 */
export async function endBatchExecution(): Promise<void> {
  if (!activeBatch) {
    throw new Error("No active batch execution");
  }
  const ctx = requireContext();
  const batch = activeBatch;
  activeBatch = null;

  // Submit each command buffer individually to avoid synchronization scope conflicts.
  // A single queue.submit([cb1, cb2, ...]) puts all CBs in the same scope, causing
  // validation errors if a buffer transitions between read and write across CBs.
  for (const cb of batch.commandBuffers) {
    profileApiCall("queue.submit", () => ctx.queue.submit([cb]));
    gpuSubmitCount++;
  }

  // Wait for GPU to finish - this is the sync point
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }

  // Now safe to destroy deferred buffers (GPU is done with them)
  for (const buffer of batch.deferredDestroyBuffers) {
    // Replay-pinned buffers must survive — referenced by recorded bind groups.
    if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) continue;
    buffer.destroy();
  }
  // Also flush any pending pool buffers since GPU work is complete
  bufferPool.flushPendingToAvailable();
}

/**
 * Check if batch execution is currently active.
 */
export function isBatchActive(): boolean {
  return activeBatch !== null;
}

/**
 * Abort batch without submitting (for error recovery).
 */
export function abortBatch(): void {
  activeBatch = null;
}

/**
 * Get the current batch encoder if active, otherwise null.
 * DEPRECATED: Use dispatchComputePass or submitOrCollect instead.
 */
export function getActiveBatchEncoder(): GPUCommandEncoder | null {
  // Return null to indicate batch mode uses collected buffers, not shared encoder
  return null;
}

// ============================================================================
// Shared Encoder — Command Buffer Consolidation
// ============================================================================
//
// Multiple compute passes are encoded into a shared GPUCommandEncoder to reduce
// queue.submit() calls. A write-set tracks all buffers written during the current
// encoder scope so the buffer pool never returns a buffer that was already written
// in this scope (preventing the aliasing corruption that broke previous attempts).
//
// Ops that create their own encoders (matmul, reductions) flush the shared encoder
// first, then their command buffer is collected and submitted alongside the
// shared encoder's CB at flush/end time.

// True Shared Encoder: a single GPUCommandEncoder is shared across multiple ops.
// Each op encodes its compute pass directly onto the shared encoder instead of
// creating its own encoder + command buffer. This eliminates ~3700 encoder
// creations per DistilGPT-2 training step.
//
// The write set tracks buffers written during the current shared encoder scope
// to prevent buffer pool aliasing — ensuring a buffer written by an earlier op
// is not reused as output for a later op within the same scope.
//
export let sharedEncoder: boolean = false;
let sharedEncoderEnabled = false;
let sharedEncoderDepth = 0;
let sharedEncoderInstance: GPUCommandEncoder | null = null;
let sharedEncoderPassCount = 0;
export let collectedCommandBuffers: GPUCommandBuffer[] = [];
export let sharedEncoderWriteSet: Set<GPUBuffer> = new Set();
let sharedEncoderDeferredUniformBuffers: GPUBuffer[] = [];
export let stepLevelScope = false; // true between beginStep() and endStep()

// Auto-flush threshold: finish current encoder and start a new one after N passes.
// This bounds the size of individual command buffers and the write/read sets.
export const SHARED_ENCODER_MAX_PASSES = 2000;

// Auto-flush when deferred uniform buffers exceed this threshold.
// With paramsSequenceBuffers caching, most params are reused directly and never
// deferred. Only evicted buffers enter the deferred list, which is typically <10
// per step in steady state. Threshold matches SHARED_ENCODER_MAX_PASSES.
export const PARAMS_FLUSH_THRESHOLD = 2000;

// Debug flag: set TORCHLETTE_DEBUG_SHARED_ENCODER=1 to enable verbose logging
export const DEBUG_SHARED_ENCODER = typeof process !== "undefined" && !!process.env?.TORCHLETTE_DEBUG_SHARED_ENCODER;

// Current op label for GPU timestamp profiling (set from lazy.ts)
let currentOpLabel: string | null = null;
export function setCurrentOpLabel(label: string | null): void { currentOpLabel = label; }
export function getCurrentOpLabel(): string | null { return currentOpLabel; }

// Adam batch mode: when true, adamStep() skips its pre-dispatch flushSharedEncoder().
// The caller (lazy.ts) is responsible for a single flush before the Adam batch.
let adamBatchMode = false;
export function setAdamBatchMode(active: boolean): void { adamBatchMode = active; }
export function isAdamBatchMode(): boolean { return adamBatchMode; }

export function beginSharedEncoder(): void {
  if (!sharedEncoderEnabled) return;
  if (sharedEncoderDepth === 0) {
    const ctx = requireContext();
    sharedEncoder = true;
    sharedEncoderInstance = ctx.device.createCommandEncoder();
    sharedEncoderPassCount = 0;
    collectedCommandBuffers = [];
    sharedEncoderWriteSet = new Set();

    sharedEncoderDeferredUniformBuffers = [];
  }
  sharedEncoderDepth++;
}

/**
 * Flush the shared encoder: finish current encoder into a CB, combine with
 * any collected CBs, submit all, then create a fresh encoder.
 */
export function flushSharedEncoder(): void {
  if (!sharedEncoder) return;
  const ctx = requireContext();

  // Finish the shared encoder into a command buffer
  const cbs: GPUCommandBuffer[] = [];
  if (sharedEncoderInstance) {
    resolveGpuTimestamps(sharedEncoderInstance);
    cbs.push(sharedEncoderInstance.finish());
  }
  // Add any collected CBs (from ops that bypassed the shared encoder)
  cbs.push(...collectedCommandBuffers);

  if (DEBUG_SHARED_ENCODER && cbs.length > 0) {
    console.log(`[shared-enc] FLUSH: ${sharedEncoderPassCount} passes on encoder, ${collectedCommandBuffers.length} collected CBs, ${sharedEncoderWriteSet.size} writes`);
  }

  if (cbs.length > 0) {
    profileApiCall("queue.submit", () => ctx.queue.submit(cbs));
    gpuSubmitCount++;
  }

  // Reset state and create fresh encoder
  collectedCommandBuffers = [];
  sharedEncoderWriteSet = new Set();
  sharedEncoderInstance = ctx.device.createCommandEncoder();
  sharedEncoderPassCount = 0;

  // Return deferred uniform buffers to pool so subsequent passes can reuse them.
  // This is critical for step-level scope where the encoder stays open across flushes.
  // Reverse iteration: buffers were deferred in forward order (A,B,C...), so pushing
  // them in reverse (C,B,A) means the next LIFO pop() sequence returns A,B,C —
  // matching the original acquisition order for bind group cache stability.
  for (let i = sharedEncoderDeferredUniformBuffers.length - 1; i >= 0; i--) {
    const buf = sharedEncoderDeferredUniformBuffers[i];
    // Replay-pinned buffers must stay alive — referenced by recorded bind groups.
    // Don't return them to the pool (prevents reuse and data corruption).
    if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buf)) continue;
    const sc = paramsBufferSizeClass(buf.size);
    const pool = paramsBufferPools.get(sc);
    if (pool) {
      if (pool.length < MAX_PARAMS_POOL_SIZE_PER_CLASS) {
        pool.push(buf);
      } else {
        bufferPool.deferredDestroyUntracked(buf);
      }
    } else {
      paramsBufferPools.set(sc, [buf]);
    }
  }
  sharedEncoderDeferredUniformBuffers = [];

  // NOTE: Do NOT flush pendingRelease to pool here (§14.1). Mid-step buffer
  // reclamation causes corruption — buffers released during a step (e.g.
  // forward-pass intermediates) may still be in-flight on the GPU when
  // reacquired by a subsequent op in the same step. Pending buffers are
  // safely flushed at endSharedEncoder() (end-of-step).
}

export function endSharedEncoder(): void {
  if (!sharedEncoderEnabled) return;
  sharedEncoderDepth--;
  if (sharedEncoderDepth === 0 && sharedEncoder) {
    // If step-level scope is active, don't submit — keep encoder open.
    // Bump depth back to 1 so inner begin/end pairs still work.
    if (stepLevelScope) {
      sharedEncoderDepth = 1;
      return;
    }

    sharedEncoder = false;
    const ctx = requireContext();

    // Finish the shared encoder and submit everything
    const cbs: GPUCommandBuffer[] = [];
    if (sharedEncoderInstance) {
      resolveGpuTimestamps(sharedEncoderInstance);
      cbs.push(sharedEncoderInstance.finish());
      sharedEncoderInstance = null;
    }
    cbs.push(...collectedCommandBuffers);

    if (DEBUG_SHARED_ENCODER && cbs.length > 0) {
      console.log(`[shared-enc] END: ${sharedEncoderPassCount} passes on encoder, ${collectedCommandBuffers.length} collected CBs, ${sharedEncoderWriteSet.size} writes`);
    }

    if (cbs.length > 0) {
      profileApiCall("queue.submit", () => ctx.queue.submit(cbs));
      gpuSubmitCount++;
    }

    collectedCommandBuffers = [];
    sharedEncoderWriteSet = new Set();

    sharedEncoderPassCount = 0;

    // Return deferred uniform buffers to pool now that all CBs are submitted.
    // Reverse iteration for LIFO stability (see flushSharedEncoder comment).
    for (let i = sharedEncoderDeferredUniformBuffers.length - 1; i >= 0; i--) {
      const buf = sharedEncoderDeferredUniformBuffers[i];
      // Replay-pinned buffers must stay alive — referenced by recorded bind groups.
      if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buf)) continue;
      const sc = paramsBufferSizeClass(buf.size);
      const pool = paramsBufferPools.get(sc);
      if (pool) {
        if (pool.length < MAX_PARAMS_POOL_SIZE_PER_CLASS) {
          pool.push(buf);
        } else {
          bufferPool.deferredDestroyUntracked(buf);
        }
      } else {
        paramsBufferPools.set(sc, [buf]);
      }
    }
    sharedEncoderDeferredUniformBuffers = [];

    // Flush storage buffer pendingRelease → main pool. The encoder was just
    // submitted, so buffers released by earlier passes are safe to reuse.
    bufferPool.flushPendingToAvailable();
    bufferPool.beginWindow();  // End-of-step is also a window boundary
  }
}

/**
 * Begin a step-level shared encoder scope.
 * Keeps the GPU command encoder open across force() boundaries,
 * only flushing at hard sync points (item() readback) and endStep().
 */
export async function beginStep(): Promise<void> {
  if (stepLevelScope) return; // already in step scope
  stepLevelScope = true;
  // Await any deferred fence from the previous markStep BEFORE opening the
  // shared encoder.  This flushes pendingRelease buffers into the main pool
  // so the upcoming training step can reuse them, eliminating the 2-step lag
  // that previously caused hundreds of unnecessary createBuffer() calls.
  await awaitDeferredFence();
  // End the previous step's window tracking (if any) and compute reservation.
  // This must happen AFTER awaitDeferredFence (which may trigger pool operations)
  // and BEFORE reserve() (which uses the computed reservation).
  // We end tracking here (not in endStep) so that post-step cleanup acquires
  // (from markStep/GC) are captured in the recording.
  bufferPool.endWindowTracking();
  // Reserve buffers based on window-demand recording from previous step.
  // Falls back to prewarm() on the first step (no recording yet).
  // Must happen BEFORE opening the shared encoder (no writeSet conflicts).
  bufferPool.reserve(requireContext().device);
  bufferPool.sortPoolBuckets(); // deterministic acquire order for bind group cache
  prePinOutputBuffers(); // pre-extract hinted output buffers before any dispatches
  bufferPool.beginWindowTracking();  // Start recording this step's demand
  beginSharedEncoder(); // open the shared encoder for the whole step
  resetDispatchSequence(); // reset bind group cache sequence counter for this step
}

/**
 * End the step-level shared encoder scope.
 * Submits all remaining encoded work.
 */
export function endStep(): void {
  if (!stepLevelScope) return;
  stepLevelScope = false;
  // Return any unconsumed pre-pinned buffers to the pool
  for (let i = 0; i < pinnedOutputBuffers.length; i++) {
    const buf = pinnedOutputBuffers[i];
    if (buf !== null && buf !== undefined) {
      bufferPool.returnToPool(buf, getSizeClass(buf.size));
      pinnedOutputBuffers[i] = null;
    }
  }
  endSharedEncoder(); // depth goes to 0 and actually submits
  // Note: endWindowTracking() is called at the start of the NEXT beginStep(),
  // not here. This ensures post-step cleanup acquires (from markStep/GC)
  // are captured in the window-demand recording.
}

export function isSharedEncoderActive(): boolean {
  return sharedEncoder;
}


export function setSharedEncoderEnabled(enabled: boolean): void {
  sharedEncoderEnabled = enabled;
}

/**
 * Get the shared encoder instance for use by external dispatch functions (e.g., matmul).
 * Returns null if no shared encoder is active.
 */
export function getSharedEncoderInstance(): GPUCommandEncoder | null {
  return sharedEncoderInstance;
}

/** Increment the shared encoder pass count (for use from modules that import). */
export function incrementSharedEncoderPassCount(): void {
  sharedEncoderPassCount++;
}

/**
 * Auto-flush the shared encoder if it has too many passes.
 * This prevents extremely large command buffers and bounds the write/read sets.
 */
export function autoFlushSharedEncoder(): void {
  if (sharedEncoder && (
    sharedEncoderPassCount >= SHARED_ENCODER_MAX_PASSES ||
    sharedEncoderDeferredUniformBuffers.length >= PARAMS_FLUSH_THRESHOLD
  )) {
    flushSharedEncoder();
  }
}

/** Defer a uniform buffer for destruction at end of shared encoder scope. */
export function deferUniformBufferForSharedEncoder(buffer: GPUBuffer): void {
  sharedEncoderDeferredUniformBuffers.push(buffer);
}

/**
 * Track a buffer as written during the current shared encoder scope.
 * The buffer pool must not return this buffer for the rest of the scope.
 */
export function trackSharedEncoderWrite(buffer: GPUBuffer): void {
  if (sharedEncoder) {
    sharedEncoderWriteSet.add(buffer);
  }
}

/**
 * Check if a buffer was written during the current shared encoder scope.
 */
export function isInSharedEncoderWriteSet(buffer: GPUBuffer): boolean {
  return sharedEncoderWriteSet.has(buffer);
}

// ============================================================================
// GPU Submit Counter (Profiling)
// ============================================================================

export let gpuSubmitCount = 0;

/**
 * Get the number of queue.submit() calls since last reset.
 */
export function getSubmitCount(): number {
  return gpuSubmitCount;
}

/**
 * Increment the GPU submit counter (for use from modules that import gpuSubmitCount).
 */
export function incrementSubmitCount(): void {
  gpuSubmitCount++;
}

/**
 * Reset the GPU submit counter.
 */
export function resetSubmitCount(): void {
  gpuSubmitCount = 0;
}

/**
 * Submit a command buffer immediately or collect it for batch submission.
 * Use this for ops that bypass the shared encoder (e.g., ops needing custom encoder logic).
 * When shared encoder is active, these CBs are collected and submitted alongside the
 * shared encoder's CB at flush/end time.
 */
export function submitOrCollect(commandBuffer: GPUCommandBuffer): void {
  if (sharedEncoder) {
    // Collect for later submission at flush/end of shared encoder scope
    collectedCommandBuffers.push(commandBuffer);
  } else if (activeBatch) {
    activeBatch.commandBuffers.push(commandBuffer);
  } else {
    const ctx = requireContext();
    profileApiCall("queue.submit", () => ctx.queue.submit([commandBuffer]));
    gpuSubmitCount++;
  }
}
