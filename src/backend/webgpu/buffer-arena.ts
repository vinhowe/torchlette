/**
 * Buffer Arena — per-lowered-plan persistent GPU buffer management.
 *
 * Extracted from index.ts. Each cached lowered plan gets its own buffer arena:
 * a set of GPUBuffers that persist across steps (never returned to pool). This
 * stabilizes buffer object identities so that bind group cache keys match across
 * steps (~100% hit rate).
 *
 * When an arena is active, resolveOutputBuffer and allocateOutputBuffer return
 * arena buffers instead of pool-allocated ones. The arena has its own output
 * index counter, independent of the global outputSeqIndex (multiple plans
 * per step each have their own arena).
 */

import type { BackendTensor } from "../types";
import type { GPUBuffer, GPUDevice, WebGPUTensor } from "./gpu-types";
import { GPUBufferUsage, STORAGE_BUFFER_USAGE } from "./gpu-types";
import {
  arenaBufferSet, trackSharedEncoderWrite, requireContext,
  replayPinnedBufferSet, paramsSequenceSet,
  outputSeqIndex, getOutputSeqIndex, setOutputSeqIndex,
} from "./webgpu-state";
import { createTrackedBuffer } from "./tensor";
import { bufferPool } from "./buffer-pool";
import { alignBufferSize } from "./shape-utils";
import { getSizeClass, getSizeForClass } from "../../engine/memory-planning";
import { gpuMemoryTracker } from "./memory-tracker";
import { profileApiCall } from "./profiler";

// Re-export from webgpu-state for backward compatibility
export { arenaBufferSet } from "./webgpu-state";
export { outputSeqIndex, getOutputSeqIndex, setOutputSeqIndex } from "./webgpu-state";

// ============================================================================
// Output allocation functions
// ============================================================================

/**
 * Allocate a new tracked buffer (from pool or fresh).
 * Use this for op output allocation when pool-based reuse is desired.
 */
export function allocateOutputBuffer(
  sizeBytes: number,
): GPUBuffer {
  // Arena fast path: if a buffer arena is active, use it.
  // This path is used by fused kernels (dispatchFusedKernel) which bypass
  // resolveOutputBuffer and call allocateOutputBuffer directly.
  if (arenaLocal.active) {
    const idx = arenaLocal.allocIndex++;
    const arenaBuffer = arenaAllocAt(arenaLocal.active.alloc, idx, sizeBytes);
    if (arenaBuffer) {
      // Check if this arena buffer aliases with an external plan input.
      // This happens when the same template is reused across layers: the
      // previous execution's output (an arena buffer) becomes the next
      // execution's external input. Without this check, the fused kernel
      // would overwrite the input data before it's fully consumed.
      if (arenaLocal.externalInputBuffers && arenaLocal.externalInputBuffers.has(arenaBuffer)) {
        // Replace the arena slot with a fresh buffer. The old buffer stays
        // alive (referenced by the external input tensor) but is no longer
        // in the arena. This permanently resolves the conflict for this slot.
        arenaBufferSet.delete(arenaBuffer);
        arenaLocal.active.alloc[idx] = undefined as any;
        const freshBuffer = arenaAllocAt(arenaLocal.active.alloc, idx, sizeBytes);
        arenaLocal.conflictDetected = true;
        return freshBuffer!;
      }
      return arenaBuffer;
    }
  } else if (arenaLocal.active) {
    arenaLocal.allocIndex++; // Keep counter in sync
  }

  const ctx = requireContext();
  const alignedSize = alignBufferSize(sizeBytes);
  const buffer = createTrackedBuffer(ctx.device, {
    size: alignedSize,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  }) as GPUBuffer;
  return buffer;
}

/**
 * Donate a tensor's buffer for reuse by another op.
 * Transfers ownership: the donor tensor will no longer destroy the buffer.
 * Returns the buffer if donation is possible, null otherwise.
 *
 * Donation requirements:
 * - Tensor must own its buffer (not a view)
 * - Buffer must have pool-compatible usage flags
 *
 * @param tensor The tensor to donate from
 * @returns The GPU buffer if donation succeeded, null otherwise
 */
export function donateBuffer(
  tensor: BackendTensor,
): GPUBuffer | null {
  const t = tensor as WebGPUTensor;

  // Can only donate if tensor owns the buffer
  if (!t.ownsBuffer) {
    return null;
  }

  // Check buffer has pool-compatible usage (STORAGE | COPY_SRC | COPY_DST)
  const bufferUsage = t.buffer.usage ?? 0;
  if (bufferUsage !== STORAGE_BUFFER_USAGE) {
    return null;
  }

  // Transfer ownership: decRef for this tensor, prevent closure from releasing
  bufferPool.decRef(t.buffer);
  (t as any).ownsBuffer = false;
  t.destroy = () => {}; // Prevent closure from firing (closure captures old ownsBuffer=true)

  return t.buffer as GPUBuffer;
}

/**
 * Get the size of a tensor's underlying buffer in bytes.
 */
export function getBufferSize(
  tensor: BackendTensor,
): number {
  const t = tensor as WebGPUTensor;
  return t.buffer.size ?? 0;
}

// ============================================================================
// Sequence-indexed output buffer hints
// ============================================================================
// Record which GPUBuffer was used at each resolveOutputBuffer position.
// On the next step, try to acquire the same buffer from the pool for bind
// group cache stability.

// outputSeqIndex, getOutputSeqIndex, setOutputSeqIndex are in webgpu-state.ts and re-exported above.
export const outputSequenceHints: Array<GPUBuffer | null> = [];

// Pre-pinned output buffers: extracted from pool at step start before any dispatches.
// Eliminates contention where multiple dispatch positions hint the same buffer.
export const pinnedOutputBuffers: Array<GPUBuffer | null> = [];

// ============================================================================
// Per-Lowered-Plan Buffer Arena
// ============================================================================
// Each cached lowered plan gets its own buffer arena: a set of GPUBuffers that
// persist across steps (never returned to pool). This stabilizes buffer object
// identities so that bind group cache keys match across steps (~100% hit rate).
//
// When an arena is active, resolveOutputBuffer and allocateOutputBuffer return
// arena buffers instead of pool-allocated ones. The arena has its own output
// index counter, independent of the global outputSeqIndex (multiple plans
// per step each have their own arena).

/**
 * Buffer arena: two separate arrays indexed by output position within a plan.
 * `resolve` tracks resolveOutputBuffer positions (non-fused ops, matmul).
 * `alloc` tracks allocateOutputBuffer positions (fused kernels).
 * Separate arrays are needed because these two allocation paths have
 * independent position sequences that may interleave differently between
 * the first execution (executePlanOptimized) and subsequent executions
 * (executeLoweredPlan).
 */
export interface BufferArena {
  resolve: GPUBuffer[];
  alloc: GPUBuffer[];
}

// arenaBufferSet is in webgpu-state.ts and re-exported above.

/**
 * Arena mutable state — groups all module-level let variables into a single
 * typed object for debuggability, testability, and explicit reset.
 */
interface ArenaLocalState {
  /** Currently active arena (set during lowered plan execution). */
  active: BufferArena | null;
  /** Resolve-path output index within the current arena. */
  resolveIndex: number;
  /** Alloc-path output index within the current arena. */
  allocIndex: number;
  /** Stats: arena resolve hits. */
  resolveHits: number;
  /** Stats: arena resolve aliased (fell through to pool). */
  resolveAliased: number;
  /** Stats: arena resolve with no arena active. */
  resolveNoArena: number;
  /** External input buffers for conflict detection. */
  externalInputBuffers: Set<GPUBuffer> | null;
  /** Flag set when an arena buffer was replaced due to external input conflict. */
  conflictDetected: boolean;
}

const arenaLocal: ArenaLocalState = {
  active: null,
  resolveIndex: 0,
  allocIndex: 0,
  resolveHits: 0,
  resolveAliased: 0,
  resolveNoArena: 0,
  externalInputBuffers: null,
  conflictDetected: false,
};

/** Get the currently active arena (for cross-module access). */
export function getActiveArena(): BufferArena | null { return arenaLocal.active; }

/** Register external input buffers for arena conflict detection. */
export function setArenaExternalInputBuffers(buffers: GPUBuffer[]): void {
  arenaLocal.externalInputBuffers = new Set(buffers);
}

/** Clear external input buffers tracking. */
export function clearArenaExternalInputBuffers(): void {
  arenaLocal.externalInputBuffers = null;
}

/** Check if any arena conflict was detected during the current plan execution. */
export function getArenaConflictDetected(): boolean { return arenaLocal.conflictDetected; }

/** Clear the arena conflict flag. */
export function clearArenaConflictDetected(): void { arenaLocal.conflictDetected = false; }

/**
 * Pre-check if any arena buffers conflict with external input buffers.
 * Used before replay to determine if the cached bind groups are still valid.
 */
export function hasArenaExternalConflicts(arena: BufferArena, extBufs: Set<GPUBuffer>): boolean {
  for (const buf of arena.resolve) {
    if (buf && extBufs.has(buf)) return true;
  }
  for (const buf of arena.alloc) {
    if (buf && extBufs.has(buf)) return true;
  }
  return false;
}

/**
 * Activate a buffer arena for the duration of a lowered plan execution.
 * All subsequent resolveOutputBuffer/allocateOutputBuffer calls will use
 * the arena instead of the pool, stabilizing buffer identities.
 */
export function setActiveArena(arena: BufferArena): void {
  arenaLocal.active = arena;
  arenaLocal.resolveIndex = 0;
  arenaLocal.allocIndex = 0;
  arenaLocal.conflictDetected = false;
}

/**
 * Deactivate the current buffer arena.
 */
export function clearActiveArena(): void {
  arenaLocal.active = null;
  arenaLocal.resolveIndex = 0;
  arenaLocal.allocIndex = 0;
}

/** Get the current arena resolve index (for dispatch replay recording). */
export function getArenaResolveIndex(): number { return arenaLocal.resolveIndex; }

/** Set the arena resolve index to a specific value (for dispatch replay restore). */
export function setArenaResolveIndexTo(idx: number): void { arenaLocal.resolveIndex = idx; }

/** Check if a buffer is owned by an arena (should not be released to pool). */
export function isArenaBuffer(buffer: GPUBuffer): boolean {
  return arenaBufferSet.has(buffer);
}

/** Classify a buffer for replay debugging. */
export function classifyBuffer(buffer: GPUBuffer): string {
  if (arenaBufferSet.has(buffer)) return "arena";
  if (paramsSequenceSet.has(buffer)) return "params-seq";
  return `other(size=${buffer.size})`;
}

/**
 * Destroy all buffers in an arena and remove them from the arena set.
 * Called when a lowered plan is evicted from the fusion analysis cache.
 */
export function destroyArena(arena: BufferArena): void {
  for (const arr of [arena.resolve, arena.alloc]) {
    for (const buffer of arr) {
      if (buffer) {
        arenaBufferSet.delete(buffer);
        // Only destroy if not referenced by a live tensor
        if (!bufferPool.isLive(buffer)) {
          // Replay-pinned buffers must survive
          if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) continue;
          gpuMemoryTracker.trackDeallocation(buffer);
          buffer.destroy();
        }
      }
    }
    arr.length = 0;
  }
}

/** Allocate or reuse an arena buffer at the given position in the given array. */
export function arenaAllocAt(arr: GPUBuffer[], idx: number, sizeBytes: number): GPUBuffer | null {
  const alignedSize = alignBufferSize(sizeBytes);
  const neededSizeClass = getSizeClass(alignedSize);

  // Check if we already have a buffer at this position from a previous step
  const existing = arr[idx];
  if (existing) {
    const existingSizeClass = getSizeClass(existing.size);
    if (existingSizeClass === neededSizeClass && !bufferPool.isLive(existing)) {
      // Perfect match, not referenced by any live tensor — safe to reuse
      trackSharedEncoderWrite(existing);
      return existing;
    }
    // Either size class changed, or buffer is still live (referenced by a persistent
    // tensor, e.g. model weight loaded by a shared plan template). In either case,
    // remove from arena set and allocate a fresh buffer below.
    arenaBufferSet.delete(existing);
    if (bufferPool.isLive(existing)) {
      // Live buffer — don't destroy. The owning tensor still references it.
      arenaLocal.conflictDetected = true;
    } else {
      // Dead buffer with wrong size class — destroy if not replay-pinned
      if (replayPinnedBufferSet === null || !replayPinnedBufferSet.has(existing)) {
        gpuMemoryTracker.trackDeallocation(existing);
        existing.destroy();
      }
    }
  }

  // Allocate a new buffer for this arena position
  const ctx = requireContext();
  const pooledSize = getSizeForClass(neededSizeClass);
  const usage = STORAGE_BUFFER_USAGE;
  const buffer = profileApiCall("createBuffer", () =>
    ctx.device.createBuffer({ size: pooledSize, usage })
  );
  gpuMemoryTracker.trackAllocation(buffer, pooledSize);
  // Track in pool's allocation tracking for proper refcounting
  bufferPool.trackAllocation(buffer, pooledSize);
  bufferPool.markAsFromPool(buffer);

  arr[idx] = buffer;
  arenaBufferSet.add(buffer);
  trackSharedEncoderWrite(buffer);
  return buffer;
}

/**
 * Pre-extract hinted output buffers from the pool before any dispatches run.
 * This ensures each dispatch position gets its hinted buffer without contention,
 * cascading bind group cache hits to downstream dispatches that read stable inputs.
 */
export function prePinOutputBuffers(): void {
  pinnedOutputBuffers.length = 0;
  for (let i = 0; i < outputSequenceHints.length; i++) {
    const hint = outputSequenceHints[i];
    if (hint === null || hint === undefined) {
      pinnedOutputBuffers[i] = null;
      continue;
    }
    // Try to extract this specific buffer from the pool
    const sizeClass = getSizeClass(hint.size);
    const bucket = bufferPool.getPoolBucket(sizeClass);
    if (bucket) {
      const idx = bucket.indexOf(hint);
      if (idx !== -1) {
        bucket.splice(idx, 1);
        const actualSize = getSizeForClass(sizeClass);
        bufferPool.adjustPooledBytes(-actualSize);
        bufferPool.recordAcquireForPin(sizeClass);
        pinnedOutputBuffers[i] = hint;
        continue;
      }
    }
    pinnedOutputBuffers[i] = null;
  }
}

export function resolveOutputBuffer(
  device: GPUDevice,
  sizeBytes: number,
  inputBuffers: GPUBuffer[],
  providedOutBuffer?: GPUBuffer,
): GPUBuffer {
  // Arena fast path: if a buffer arena is active, use it for output allocation.
  // Arena buffers persist across steps, giving 100% stable buffer identities
  // for bind group cache hits.
  if (!providedOutBuffer && arenaLocal.active) {
    const idx = arenaLocal.resolveIndex++;
    const arenaBuffer = arenaAllocAt(arenaLocal.active.resolve, idx, sizeBytes);
    if (arenaBuffer) {
      // Check for external input conflict first (structural — replace arena slot).
      if (arenaLocal.externalInputBuffers && arenaLocal.externalInputBuffers.has(arenaBuffer)) {
        // Replace the arena slot with a fresh buffer. The old buffer stays
        // alive (referenced by the external input tensor).
        arenaBufferSet.delete(arenaBuffer);
        arenaLocal.active.resolve[idx] = undefined as any;
        const freshBuffer = arenaAllocAt(arenaLocal.active.resolve, idx, sizeBytes);
        arenaLocal.conflictDetected = true;
        // Still need to check direct aliasing on the fresh buffer
        if (freshBuffer && !inputBuffers.some(b => b === freshBuffer)) {
          arenaLocal.resolveHits++;
          return freshBuffer;
        }
        // Fresh buffer aliased with direct input — fall through to normal path
        arenaLocal.resolveAliased++;
      } else if (!inputBuffers.some(b => b === arenaBuffer)) {
        // No conflict, no aliasing — use arena buffer directly
        arenaLocal.resolveHits++;
        return arenaBuffer;
      } else {
        // Direct aliasing with current op's input — fall through to normal path.
        // Don't replace arena slot; this is a normal within-plan aliasing situation.
        arenaLocal.resolveAliased++;
      }
    }
  } else if (!providedOutBuffer) {
    arenaLocal.resolveNoArena++;
  }

  const alignedSize = alignBufferSize(sizeBytes);
  // Use pool-rounded size for replacement allocations so they match the size
  // class exactly. Without this, undersized buffers enter the pool and can
  // later be acquired by tensors needing the full size class, causing overflow.
  const pooledSize = getSizeForClass(getSizeClass(alignedSize));
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outIdx = getOutputSeqIndex();
  setOutputSeqIndex(outIdx + 1);

  // Fast path: check pre-pinned buffer (extracted from pool at step start).
  // Pre-pinning eliminates contention — each dispatch position gets its hinted
  // buffer without racing other positions for the same pool slot.
  if (!providedOutBuffer) {
    const pinned = pinnedOutputBuffers[outIdx];
    if (pinned !== null && pinned !== undefined) {
      const neededSizeClass = getSizeClass(alignedSize);
      const pinnedSizeClass = getSizeClass(pinned.size);
      if (pinnedSizeClass === neededSizeClass && !inputBuffers.some(b => b === pinned)) {
        // Use pre-pinned buffer directly (already extracted from pool)
        trackSharedEncoderWrite(pinned);
        gpuMemoryTracker.trackAllocation(pinned, getSizeForClass(pinnedSizeClass));
        bufferPool.markAsFromPool(pinned);
        outputSequenceHints[outIdx] = pinned;
        pinnedOutputBuffers[outIdx] = null; // consumed
        return pinned;
      }
      // Pin didn't match (size class changed or aliased with input) — return to pool
      bufferPool.returnToPool(pinned, getSizeClass(pinned.size));
      pinnedOutputBuffers[outIdx] = null;
    }
  }

  const preferredBuf = providedOutBuffer ? undefined : (outputSequenceHints[outIdx] ?? undefined);
  let outBuffer = providedOutBuffer ?? createTrackedBuffer(device, { size: alignedSize, usage }, preferredBuf);

  if (!providedOutBuffer && inputBuffers.some(b => b === outBuffer)) {
    const released = bufferPool.release(outBuffer, alignedSize, usage);
    if (!released) {
      bufferPool.deferredDestroy(outBuffer, alignedSize);
    }
    outBuffer = device.createBuffer({ size: pooledSize, usage });
    gpuMemoryTracker.trackAllocation(outBuffer, pooledSize);
    bufferPool.trackAllocation(outBuffer, pooledSize);
    bufferPool.markAsFromPool(outBuffer);
    bufferPool.trackNewAllocation(pooledSize);
  }

  // Track this buffer as written in the current shared encoder scope
  trackSharedEncoderWrite(outBuffer);

  // Record for next step's preferred-buffer hint
  if (!providedOutBuffer) {
    outputSequenceHints[outIdx] = outBuffer;
  }

  return outBuffer;
}

// ============================================================================
// Arena stats (accessed by bind-group-cache reset)
// ============================================================================

export function getArenaResolveStats(): { hits: number; aliased: number; noArena: number } {
  return { hits: arenaLocal.resolveHits, aliased: arenaLocal.resolveAliased, noArena: arenaLocal.resolveNoArena };
}

export function resetArenaResolveStats(): void {
  arenaLocal.resolveHits = 0;
  arenaLocal.resolveAliased = 0;
  arenaLocal.resolveNoArena = 0;
}

/**
 * Reset arena-related mutable state. Called by resetBindGroupSequenceCache
 * in the bind-group-cache or index module during step transitions.
 */
export function resetArenaState(): void {
  outputSequenceHints.length = 0;
  pinnedOutputBuffers.length = 0;
  setOutputSeqIndex(0);
  arenaLocal.active = null;
  arenaLocal.resolveIndex = 0;
  arenaLocal.allocIndex = 0;
}
