/**
 * Buffer Arena — per-lowered-plan persistent GPU buffer management.
 *
 * Each cached lowered plan gets its own buffer arena:
 * a set of GPUBuffers that persist across steps (never returned to pool). This
 * stabilizes buffer object identities so that bind group cache keys match across
 * steps (~100% hit rate).
 *
 * When an arena is active, resolveOutputBuffer and allocateOutputBuffer return
 * arena buffers instead of pool-allocated ones. The arena has its own output
 * index counter, independent of the global outputSeqIndex (multiple plans
 * per step each have their own arena).
 */

import { ENV } from "../../core/env";
import { recordAlloc } from "../../executor/compiled-plan";
import { getSizeClass, getSizeForClass } from "../../graph/lifetime-analysis";
import type { BackendTensor } from "../types";
import { bufAcquire, bufRegister } from "./buffer-debug";
import { bufferPool } from "./buffer-pool";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { asGPUTensor, GPUBufferUsage, STORAGE_BUFFER_USAGE } from "./gpu-types";
import { gpuMemoryTracker } from "./memory-tracker";
import { profileApiCall } from "./profiler";
import { alignBufferSize } from "./shape-utils";
import { createTrackedBuffer } from "./tensor";
import {
  arenaBufferSet,
  pinnedBufferSet,
  getOutputSeqIndex,
  requireContext,
  setOutputSeqIndex,
  trackSharedEncoderWrite,
} from "./webgpu-state";

// ============================================================================
// Output allocation functions
// ============================================================================

/**
 * Allocate a new tracked buffer (from pool or fresh).
 * Use this for op output allocation when pool-based reuse is desired.
 */
export function allocateOutputBuffer(sizeBytes: number): GPUBuffer {
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
      if (arenaLocal.externalInputBuffers?.has(arenaBuffer)) {
        // Replace the arena slot with a fresh buffer. The old buffer stays
        // alive — it's referenced by the external input tensor — but it's
        // no longer owned by the arena. Just remove from arenaBufferSet;
        // when the input tensor's destroy() eventually runs it will see
        // the buffer is no longer in arenaBufferSet and route it through
        // the normal pool release chain (decRef → pendingRelease → pool).
        // Calling release() here would create a DUPLICATE pendingRelease
        // entry that's pinned by the still-live owner's refcount, which
        // never resolves and accumulates as a per-step leak.
        arenaBufferSet.delete(arenaBuffer);
        arenaLocal.active.alloc[idx] = undefined;
        const freshBuffer = arenaAllocAt(
          arenaLocal.active.alloc,
          idx,
          sizeBytes,
        );
        arenaLocal.conflictDetected = true;
        recordAlloc(freshBuffer as GPUBuffer, sizeBytes, 1, []);
        return freshBuffer as GPUBuffer;
      }
      recordAlloc(arenaBuffer, sizeBytes, 1, []);
      return arenaBuffer;
    }
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
  recordAlloc(buffer, sizeBytes, 1, []);
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
export function donateBuffer(tensor: BackendTensor): GPUBuffer | null {
  const t = asGPUTensor(tensor);

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
  t.ownsBuffer = false;
  t.destroy = () => {}; // Prevent closure from firing (closure captures old ownsBuffer=true)

  return t.buffer as GPUBuffer;
}

/**
 * Get the size of a tensor's underlying buffer in bytes.
 */
export function getBufferSize(tensor: BackendTensor): number {
  const t = asGPUTensor(tensor);
  return t.buffer.size ?? 0;
}

// ============================================================================
// Sequence-indexed output buffer hints
// ============================================================================
// Record which GPUBuffer was used at each resolveOutputBuffer position.
// On the next step, try to acquire the same buffer from the pool for bind
// group cache stability.

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
  resolve: (GPUBuffer | undefined)[];
  alloc: (GPUBuffer | undefined)[];
}

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
export function getActiveArena(): BufferArena | null {
  return arenaLocal.active;
}

/** Register external input buffers for arena conflict detection. */
export function setArenaExternalInputBuffers(buffers: GPUBuffer[]): void {
  arenaLocal.externalInputBuffers = new Set(buffers);
}

/** Clear external input buffers tracking. */
export function clearArenaExternalInputBuffers(): void {
  arenaLocal.externalInputBuffers = null;
}

/** Check if any arena conflict was detected during the current plan execution. */
export function getArenaConflictDetected(): boolean {
  return arenaLocal.conflictDetected;
}

/** Clear the arena conflict flag. */
export function clearArenaConflictDetected(): void {
  arenaLocal.conflictDetected = false;
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

/** Check if a buffer is owned by an arena (should not be released to pool). */
export function isArenaBuffer(buffer: GPUBuffer): boolean {
  return arenaBufferSet.has(buffer);
}

/**
 * Destroy all buffers in an arena and remove them from the arena set.
 * Called when a lowered plan is evicted from the fusion analysis cache.
 *
 * @param force - if true, destroy buffers unconditionally. The default
 *   path skips destroying buffers still referenced by a live tensor
 *   (mid-step eviction safety), but cross-session recycle paths must
 *   pass force=true: by then, every "live" reference is stale state
 *   from the previous session and refusing to destroy leaks the buffer
 *   permanently — ~20 MiB per session, observed empirically.
 */
export function destroyArena(arena: BufferArena, force = false): void {
  for (const arr of [arena.resolve, arena.alloc]) {
    for (const buffer of arr) {
      if (buffer) {
        arenaBufferSet.delete(buffer);
        // Pinned by a compiled plan's recorded assignment — the plan
        // outlives this arena teardown; destroyCompiledPlanBuffers disposes.
        if (pinnedBufferSet.has(buffer)) continue;
        if (force || bufferPool.canRecycle(buffer)) {
          gpuMemoryTracker.trackDeallocation(buffer);
          try {
            buffer.destroy();
          } catch {
            /* already destroyed */
          }
        }
      }
    }
    arr.length = 0;
  }
}

/**
 * Bounded-arena mode (TORCHLETTE_ARENA_LIVENESS=1): the per-position arena is
 * UNBUDGETED — one persistent buffer per dispatch position sums to ~17GB for
 * 124M GPT-2, OOMing a 32GB V100. Under this flag, buffers larger than a
 * threshold SPILL to the budgeted buffer POOL instead of the arena: the pool
 * already reclaims dead buffers SAFELY (released buffers wait in pendingRelease
 * and are only reused after a GPU fence — so a buffer read by a still-queued op
 * is never handed out, the hazard that a naive arena-side reuse hits). This
 * bounds memory to the pool's live working set while keeping the many small
 * buffers arena-resident for bind-group stability. Compiled-plan replay works
 * in this mode via PLANNED BUFFERS (see compiledPlannedEnabled below): the
 * recorded pool-buffer assignment is pinned to the plan and replays bind it
 * directly — compiled speed at bounded memory.
 * Default off. Threshold via TORCHLETTE_ARENA_MAX_BUFFER_MB (default 2MB).
 */
let _arenaLiveness: boolean | null = null;
export function arenaLivenessEnabled(): boolean {
  if (_arenaLiveness === null) {
    // DEFAULT ON (2026-06-11). The bounded arena + planned compiled buffers
    // is strictly better than the unbudgeted per-position arena on every
    // workload measured: same compiled speed at roughly half the memory
    // (distil@512 5.0GB vs 9.1GB; Medium@512 13.8GB vs 28.6GB — the default
    // arena barely fit a 32GB V100), and the production 124M DiLoCo runs
    // live here. Validated: the forced-liveness full suite is green, all
    // parity/regression gates pass in this mode, and the 4-peer soak
    // converges better than the unbudgeted baseline.
    // TORCHLETTE_ARENA_LIVENESS=0 opts back into the unbudgeted arena;
    // the globalThis escape hatch still forces it on for browser tests.
    _arenaLiveness =
      ENV.TORCHLETTE_ARENA_LIVENESS !== "0" ||
      !!(globalThis as { __torchletteArenaLiveness?: boolean })
        .__torchletteArenaLiveness;
  }
  return _arenaLiveness;
}

/**
 * Planned compiled buffers: under the bounded (liveness) arena, compiled
 * plans pin and replay the recorded pool-buffer assignment instead of
 * disabling compilation. ON by default whenever liveness mode is on
 * (measured: GPT-2 Medium @512 13.8GB at compiled speed, vs ~525ms/step
 * lowered); TORCHLETTE_COMPILED_PLANNED=0 opts out (lowered execution, the
 * pre-2026-06-10 liveness behavior). No effect in default-arena mode.
 */
export function compiledPlannedEnabled(): boolean {
  return (
    arenaLivenessEnabled() && ENV.TORCHLETTE_COMPILED_PLANNED !== "0"
  );
}
let _arenaSpillBytes: number | null = null;
function arenaSpillThreshold(): number {
  if (_arenaSpillBytes === null) {
    const mb = ENV.TORCHLETTE_ARENA_MAX_BUFFER_MB;
    _arenaSpillBytes = (mb ? parseFloat(mb) : 2) * 1024 * 1024;
  }
  return _arenaSpillBytes;
}
/** Test/diagnostic hook: re-read the flag + threshold from env. */
export function resetArenaLivenessCache(): void {
  _arenaLiveness = null;
  _arenaSpillBytes = null;
}

/** Allocate or reuse an arena buffer at the given position in the given array. */
function arenaAllocAt(
  arr: (GPUBuffer | undefined)[],
  idx: number,
  sizeBytes: number,
): GPUBuffer | null {
  // Bounded-arena spill: large buffers bypass the unbudgeted arena and fall
  // through to the budgeted, fence-safe pool (caller treats null as "use pool").
  // Only valid because the compiled plan is disabled under this flag — otherwise
  // its replay would reuse stale recorded identities for spilled positions.
  if (arenaLivenessEnabled() && sizeBytes > arenaSpillThreshold()) return null;
  const alignedSize = alignBufferSize(sizeBytes);
  const neededSizeClass = getSizeClass(alignedSize);

  // Check if we already have a buffer at this position from a previous step.
  // Reuse only when (a) the size class matches and (b) the buffer is safe
  // to recycle right now — both ownership and in-flight encoder claims must
  // be clear. See bufferPool.canRecycle for why both checks are required.
  const existing = arr[idx];
  if (existing) {
    if (
      getSizeClass(existing.size) === neededSizeClass &&
      bufferPool.canRecycle(existing)
    ) {
      trackSharedEncoderWrite(existing);
      return existing;
    }
    // Can't reuse here. Drop from arena set and let the normal release
    // chain handle it: a live buffer's owner will release via tensor.destroy
    // when the tensor goes away; an in-flight orphaned buffer is already in
    // pendingRelease (queued by tensor.destroy) and will hit the pool after
    // the next fence. Calling release() / deferredDestroy() here would
    // double-track the buffer.
    arenaBufferSet.delete(existing);
    if (bufferPool.isLive(existing)) {
      arenaLocal.conflictDetected = true;
    }
  }

  // Allocate a buffer for this arena position. Try the POOL first: an
  // in-place-updated persistent tensor (copy_ into optimizer state / params)
  // migrates its storage into this plan's output position every step, so
  // the position's previous buffer is a live EXTERNAL INPUT next step → the
  // conflict path vacates the position and the old state buffer lands in
  // the pool when the tensor's storage moves on. Without pool reuse here,
  // every such position allocates a FRESH buffer each step (e.g. the
  // foreach optimizer's 328MB packed m/v leaked +640MB/step); with it, the
  // producer/consumer ping-pong settles into two alternating pooled buffers.
  if (ENV.TORCHLETTE_DEBUG_BIGALLOC && sizeBytes > 64 * 1024 * 1024) {
    console.log(
      `[bigalloc] arena idx=${idx} bytes=${(sizeBytes / 1e6).toFixed(0)}MB existing=${existing ? `yes(class ${getSizeClass(existing.size)} vs ${neededSizeClass}, recyclable=${bufferPool.canRecycle(existing)})` : "no"}`,
    );
  }
  const ctx = requireContext();
  const pooledSize = getSizeForClass(neededSizeClass);
  // EXPERIMENTAL, default OFF: acquiring pool buffers into arena positions
  // broke compiled replays (a pool buffer recorded at one slot can be handed
  // to another → slot aliasing → training divergence). The in-place copy_
  // fast path removed the allocation churn this was meant to fix.
  const pooled =
    ENV.TORCHLETTE_ARENA_POOL_ACQUIRE === "1"
      ? bufferPool.acquire(alignedSize)
      : null;
  if (pooled) {
    arr[idx] = pooled;
    arenaBufferSet.add(pooled);
    bufRegister(pooled, pooledSize, "arena");
    trackSharedEncoderWrite(pooled);
    return pooled;
  }
  const usage = STORAGE_BUFFER_USAGE;
  const buffer = profileApiCall("createBuffer", () =>
    ctx.device.createBuffer({ size: pooledSize, usage }),
  );
  gpuMemoryTracker.trackAllocation(buffer, pooledSize);
  // Track in pool's allocation tracking for proper refcounting
  bufferPool.trackAllocation(buffer, pooledSize);
  bufferPool.markAsFromPool(buffer);

  arr[idx] = buffer;
  arenaBufferSet.add(buffer);
  bufRegister(buffer, pooledSize, "arena");
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
    if (hint == null) {
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
      if (arenaLocal.externalInputBuffers?.has(arenaBuffer)) {
        // Replace the arena slot with a fresh buffer. The old buffer stays
        // alive — referenced by the external input tensor — but it's no
        // longer owned by the arena. Just remove from arenaBufferSet; the
        // owner's tensor.destroy() will route it through the pool release
        // chain later. Calling release() here would create a duplicate
        // pendingRelease entry pinned by the live refcount → leak.
        arenaBufferSet.delete(arenaBuffer);
        arenaLocal.active.resolve[idx] = undefined;
        const freshBuffer = arenaAllocAt(
          arenaLocal.active.resolve,
          idx,
          sizeBytes,
        );
        arenaLocal.conflictDetected = true;
        // Still need to check direct aliasing on the fresh buffer
        if (freshBuffer && !inputBuffers.some((b) => b === freshBuffer)) {
          arenaLocal.resolveHits++;
          recordAlloc(freshBuffer, sizeBytes, 0, inputBuffers);
          return freshBuffer;
        }
        // Fresh buffer aliased with direct input — fall through to normal path
        arenaLocal.resolveAliased++;
      } else if (!inputBuffers.some((b) => b === arenaBuffer)) {
        // No conflict, no aliasing — use arena buffer directly
        arenaLocal.resolveHits++;
        recordAlloc(arenaBuffer, sizeBytes, 0, inputBuffers);
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
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outIdx = getOutputSeqIndex();
  setOutputSeqIndex(outIdx + 1);

  // Fast path: check pre-pinned buffer (extracted from pool at step start).
  // Pre-pinning eliminates contention — each dispatch position gets its hinted
  // buffer without racing other positions for the same pool slot.
  if (!providedOutBuffer) {
    const pinned = pinnedOutputBuffers[outIdx];
    if (pinned != null) {
      const neededSizeClass = getSizeClass(alignedSize);
      const pinnedSizeClass = getSizeClass(pinned.size);
      if (
        pinnedSizeClass === neededSizeClass &&
        !inputBuffers.some((b) => b === pinned)
      ) {
        trackSharedEncoderWrite(pinned);
        gpuMemoryTracker.trackAllocation(
          pinned,
          getSizeForClass(pinnedSizeClass),
        );
        bufferPool.markAsFromPool(pinned);
        bufAcquire(pinned, "resolveOutputBuffer.pinned");
        outputSequenceHints[outIdx] = pinned;
        pinnedOutputBuffers[outIdx] = null; // consumed
        recordAlloc(pinned, sizeBytes, 0, inputBuffers);
        return pinned;
      }
      // Pin didn't match (size class changed or aliased with input) — return to pool
      bufferPool.returnToPool(pinned, getSizeClass(pinned.size));
      pinnedOutputBuffers[outIdx] = null;
    }
  }

  const preferredBuf = providedOutBuffer
    ? undefined
    : (outputSequenceHints[outIdx] ?? undefined);
  let outBuffer =
    providedOutBuffer ??
    createTrackedBuffer(device, { size: alignedSize, usage }, preferredBuf);

  if (!providedOutBuffer && inputBuffers.some((b) => b === outBuffer)) {
    const released = bufferPool.release(outBuffer, alignedSize, usage);
    if (!released) {
      bufferPool.deferredDestroy(outBuffer, alignedSize);
    }
    outBuffer = device.createBuffer({ size: pooledSize, usage });
    gpuMemoryTracker.trackAllocation(outBuffer, pooledSize);
    bufferPool.trackAllocation(outBuffer, pooledSize);
    bufferPool.markAsFromPool(outBuffer);
    bufferPool.trackNewAllocation(pooledSize);
    bufRegister(outBuffer, pooledSize, "resolveOutputBuffer.alias-replace");
  }

  // Track this buffer as written in the current shared encoder scope
  trackSharedEncoderWrite(outBuffer);

  // Record for next step's preferred-buffer hint
  if (!providedOutBuffer) {
    outputSequenceHints[outIdx] = outBuffer;
  }

  recordAlloc(outBuffer, sizeBytes, 0, inputBuffers);
  return outBuffer;
}

// ============================================================================
// Arena stats (accessed by bind-group-cache reset)
// ============================================================================

export function getArenaResolveStats(): {
  hits: number;
  aliased: number;
  noArena: number;
} {
  return {
    hits: arenaLocal.resolveHits,
    aliased: arenaLocal.resolveAliased,
    noArena: arenaLocal.resolveNoArena,
  };
}

/** Get the current resolve-path output index within the active arena. */
export function getArenaResolveIndex(): number {
  return arenaLocal.resolveIndex;
}

/** Set the resolve-path output index to a specific value. */
export function setArenaResolveIndexTo(value: number): void {
  arenaLocal.resolveIndex = value;
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
