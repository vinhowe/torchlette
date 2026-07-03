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

import { ENV } from "../../core/env";
import { getSizeClass } from "../../graph/lifetime-analysis";
import { resetDispatchSequence } from "./bind-group-cache";
import { pinnedOutputBuffers, prePinOutputBuffers } from "./buffer-arena";
import { bufBeginScope, bufEndScope } from "./buffer-debug";
import { awaitDeferredFence, bufferPool } from "./buffer-pool";
import { advanceEpoch } from "./epoch";
import type {
  GPUBuffer,
  GPUCommandBuffer,
  GPUCommandEncoder,
} from "./gpu-types";
import { profileApiCall, resolveGpuTimestamps } from "./profiler";
import {
  activeBatch,
  incrementSubmitCount,
  MAX_PARAMS_POOL_SIZE_PER_CLASS,
  paramsBufferPools,
  paramsBufferSizeClass,
  requireContext,
  resetSharedEncoderWriteSet,
  setActiveBatch,
  setSharedEncoderActive,
  sharedEncoderActive,
  sharedEncoderWriteSet,
} from "./webgpu-state";

// Re-exports from webgpu-state
export {
  getSubmitCount,
  incrementSubmitCount,
  resetSubmitCount,
  trackSharedEncoderWrite,
} from "./webgpu-state";

// ============================================================================
// Batch Execution Context for True Segmented Execution
// ============================================================================

/**
 * Begin batched execution. Subsequent ops will collect command buffers
 * instead of submitting immediately. Call endBatchExecution() to submit
 * all collected work and wait for GPU completion.
 */
export function beginBatchExecution(): void {
  if (activeBatch) {
    throw new Error("Batch execution already active");
  }
  setActiveBatch({
    commandBuffers: [],
    deferredDestroyBuffers: [],
  });
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
  setActiveBatch(null);

  // Submit each command buffer individually to avoid synchronization scope conflicts.
  // A single queue.submit([cb1, cb2, ...]) puts all CBs in the same scope, causing
  // validation errors if a buffer transitions between read and write across CBs.
  for (const cb of batch.commandBuffers) {
    profileApiCall("queue.submit", () => ctx.queue.submit([cb]));
    incrementSubmitCount();
  }

  // Wait for GPU to finish - this is the sync point
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }

  // Now safe to destroy deferred buffers (GPU is done with them)
  for (const buffer of batch.deferredDestroyBuffers) {
    buffer.destroy();
  }
  // GPU work is complete — quiescent point: advance the engine epoch
  // (flushes pending pool buffers).
  advanceEpoch("endBatch");
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
  setActiveBatch(null);
}

// ============================================================================
// Shared Encoder — Command Buffer Consolidation
// ============================================================================

/**
 * Shared encoder mutable state — groups all module-level let variables
 * into a single typed object for debuggability and explicit reset.
 */
interface EncoderState {
  enabled: boolean;
  depth: number;
  instance: GPUCommandEncoder | null;
  passCount: number;
  collectedCommandBuffers: GPUCommandBuffer[];
  deferredUniformBuffers: GPUBuffer[];
  stepLevelScope: boolean;
}

const encoderState: EncoderState = {
  enabled: false,
  depth: 0,
  instance: null,
  passCount: 0,
  collectedCommandBuffers: [],
  deferredUniformBuffers: [],
  stepLevelScope: false,
};

// Auto-flush threshold: finish current encoder and start a new one after N passes.
// This bounds the size of individual command buffers and the write/read sets.
const SHARED_ENCODER_MAX_PASSES = 2000;

// Auto-flush when deferred uniform buffers exceed this threshold.
// With paramsSequenceBuffers caching, most params are reused directly and never
// deferred. Only evicted buffers enter the deferred list, which is typically <10
// per step in steady state. Threshold matches SHARED_ENCODER_MAX_PASSES.
const PARAMS_FLUSH_THRESHOLD = 2000;

// Debug flag: set TORCHLETTE_DEBUG_SHARED_ENCODER=1 to enable verbose logging
const DEBUG_SHARED_ENCODER =
  typeof process !== "undefined" &&
  !!ENV.TORCHLETTE_DEBUG_SHARED_ENCODER;

// currentOpLabel + get/set moved to webgpu-state.ts, re-exported here for backward compat.
export { getCurrentOpLabel, setCurrentOpLabel } from "./webgpu-state";

export function beginSharedEncoder(): void {
  if (!encoderState.enabled) return;
  if (encoderState.depth === 0) {
    const ctx = requireContext();
    setSharedEncoderActive(true);
    encoderState.instance = ctx.device.createCommandEncoder();
    encoderState.passCount = 0;
    encoderState.collectedCommandBuffers = [];
    resetSharedEncoderWriteSet();

    encoderState.deferredUniformBuffers = [];
    bufBeginScope("shared-encoder");
  }
  encoderState.depth++;
}

/**
 * Finish current encoder, combine with collected CBs, submit, and reset state.
 * When createNew is true (flush), a fresh encoder is created for continued encoding.
 * When createNew is false (end), the encoder is nulled out.
 */
function finishAndSubmitEncoder(createNew: boolean): void {
  const ctx = requireContext();
  const cbs: GPUCommandBuffer[] = [];
  if (encoderState.instance) {
    resolveGpuTimestamps();
    cbs.push(encoderState.instance.finish());
    encoderState.instance = createNew
      ? ctx.device.createCommandEncoder()
      : null;
  }
  cbs.push(...encoderState.collectedCommandBuffers);

  if (DEBUG_SHARED_ENCODER && cbs.length > 0) {
    console.log(
      `[shared-enc] ${createNew ? "FLUSH" : "END"}: ${encoderState.passCount} passes, ${encoderState.collectedCommandBuffers.length} collected CBs, ${sharedEncoderWriteSet.size} writes`,
    );
  }

  if (cbs.length > 0) {
    profileApiCall("queue.submit", () => ctx.queue.submit(cbs));
    incrementSubmitCount();
  }

  encoderState.collectedCommandBuffers = [];
  resetSharedEncoderWriteSet();
  encoderState.passCount = 0;
  // Return deferred uniform buffers to pool. Reverse iteration preserves LIFO
  // acquisition order for bind group cache stability.
  for (let i = encoderState.deferredUniformBuffers.length - 1; i >= 0; i--) {
    const buf = encoderState.deferredUniformBuffers[i];
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
  encoderState.deferredUniformBuffers = [];
}

/**
 * Flush the shared encoder: finish current encoder into a CB, combine with
 * any collected CBs, submit all, then create a fresh encoder.
 *
 * NOTE: Does NOT flush pendingRelease to pool (§14.1). Mid-step buffer
 * reclamation causes corruption — buffers released during a step may still
 * be in-flight on the GPU when reacquired. Pending buffers are safely
 * flushed at endSharedEncoder() (the epoch advance at encoder close).
 */
export function flushSharedEncoder(): void {
  if (!sharedEncoderActive) return;
  finishAndSubmitEncoder(true);
}

export function endSharedEncoder(): void {
  if (!encoderState.enabled) return;
  encoderState.depth--;
  if (encoderState.depth === 0 && sharedEncoderActive) {
    // If step-level scope is active, don't submit — keep encoder open.
    // Bump depth back to 1 so inner begin/end pairs still work.
    if (encoderState.stepLevelScope) {
      encoderState.depth = 1;
      return;
    }

    setSharedEncoderActive(false);
    finishAndSubmitEncoder(false);
    bufEndScope("shared-encoder");

    // The encoder was just submitted, so buffers released by earlier passes
    // are safe to reuse: this is the buffer-pool epoch boundary. Advance the
    // engine epoch (flushes pendingRelease → main pool). Steps consume this
    // epoch — encoder close IS the quiescent point, not "end of step".
    advanceEpoch("endSharedEncoder");
    bufferPool.beginWindow(); // Encoder close is also a window boundary
  }
}

/**
 * Begin a step-level shared encoder scope.
 * Keeps the GPU command encoder open across force() boundaries,
 * only flushing at hard sync points (item() readback) and endStep().
 */
export async function beginStep(): Promise<void> {
  if (encoderState.stepLevelScope) return; // already in step scope
  encoderState.stepLevelScope = true;

  // Fence is already awaited by markStep — just do buffer pool maintenance.
  // awaitDeferredFence() is a no-op here (fence already consumed), but call
  // it for safety in case beginStep is called without a prior markStep.
  await awaitDeferredFence();
  bufferPool.destroyPendingBuffers();
  bufferPool.endWindowTracking();
  bufferPool.reserve(requireContext().device);
  bufferPool.sortPoolBuckets();
  prePinOutputBuffers();
  bufferPool.beginWindowTracking();
  beginSharedEncoder();
  resetDispatchSequence();
}

/**
 * End the step-level shared encoder scope.
 * Submits all remaining encoded work.
 */
export function endStep(): void {
  if (!encoderState.stepLevelScope) return;
  encoderState.stepLevelScope = false;
  // Return any unconsumed pre-pinned buffers to the pool
  for (let i = 0; i < pinnedOutputBuffers.length; i++) {
    const buf = pinnedOutputBuffers[i];
    if (buf != null) {
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
  return sharedEncoderActive;
}

export function setSharedEncoderEnabled(enabled: boolean): void {
  encoderState.enabled = enabled;
}

/**
 * Get the shared encoder instance for use by external dispatch functions (e.g., matmul).
 * Returns null if no shared encoder is active.
 */
export function getSharedEncoderInstance(): GPUCommandEncoder | null {
  return encoderState.instance;
}

/** Increment the shared encoder pass count (for use from modules that import). */
export function incrementSharedEncoderPassCount(): void {
  encoderState.passCount++;
}

/**
 * Auto-flush the shared encoder if it has too many passes.
 * This prevents extremely large command buffers and bounds the write/read sets.
 */
export function autoFlushSharedEncoder(): void {
  if (
    sharedEncoderActive &&
    (encoderState.passCount >= SHARED_ENCODER_MAX_PASSES ||
      encoderState.deferredUniformBuffers.length >= PARAMS_FLUSH_THRESHOLD)
  ) {
    flushSharedEncoder();
  }
}

/** Defer a uniform buffer for destruction at end of shared encoder scope. */
export function deferUniformBufferForSharedEncoder(buffer: GPUBuffer): void {
  encoderState.deferredUniformBuffers.push(buffer);
}

/**
 * Submit a command buffer immediately or collect it for batch submission.
 * Use this for ops that bypass the shared encoder (e.g., ops needing custom encoder logic).
 * When shared encoder is active, these CBs are collected and submitted alongside the
 * shared encoder's CB at flush/end time.
 */
export function submitOrCollect(commandBuffer: GPUCommandBuffer): void {
  if (sharedEncoderActive) {
    // Collect for later submission at flush/end of shared encoder scope
    encoderState.collectedCommandBuffers.push(commandBuffer);
  } else if (activeBatch) {
    activeBatch.commandBuffers.push(commandBuffer);
  } else {
    const ctx = requireContext();
    profileApiCall("queue.submit", () => ctx.queue.submit([commandBuffer]));
    incrementSubmitCount();
  }
}
