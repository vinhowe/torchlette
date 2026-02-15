import type { FusionRecipe } from "../../engine/fusion";
import type { IRGraph, IRNode } from "../../engine/ir";
import { storageTracker } from "../../engine/lazy";
import { getSizeClass, getSizeForClass } from "../../engine/memory-planning";
import { registerWebGPUDonation } from "../../engine/memory-planned-executor";
import { registerBackend } from "../registry";
import { getExpr as getExprFromRegistry, isUnaryOp as isUnaryOpFromRegistry, getArity as getArityFromRegistry } from "./ops/registry";
import { dispatchAdamStep as dispatchAdamStepKernel } from "./adam-kernel";
import { dispatchUnscaleGrad as dispatchUnscaleGradKernel, allocateInfFlagBuffer, readInfFlag, destroyPersistentInfFlagBuffer } from "./unscale-kernel";
import { dispatchCrossEntropyForward as dispatchCEForwardKernel, dispatchCrossEntropyBackward as dispatchCEBackwardKernel } from "./cross-entropy-kernel";
import { dispatchLayerNormForward as dispatchLNForwardKernel, dispatchLayerNormBackwardGradX as dispatchLNBwdGradXKernel, dispatchLayerNormBackwardGradWeightBias as dispatchLNBwdGradWBKernel } from "./layernorm-kernel";
import type {
  Backend,
  BackendTensor,
  DivOptions,
  DType,
  GatherOptions,
  GeluOptions,
  MeanOptions,
  ScatterAddOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
} from "../types";
import {
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  dispatchTiledMatmul,
  type EpilogueConfig,
  isAutotuneEnabled,
  pretuneMatmulShapes as pretuneShapes,
  setAutotuneEnabled,
  setSubgroupSupport,
  type SubgroupSupport,
} from "./matmul";

// Re-export autotune control functions
export { isAutotuneEnabled, pretuneShapes as pretuneMatmulShapes, setAutotuneEnabled };
import {
  gpuMemoryTracker,
  GPUMemoryLimitExceededError,
  setGPUMemoryLimit,
  getGPUMemoryLimit,
  getGPUMemoryStats,
  getGPUAllocationHistogram,
  enableLargeAllocDebug,
  getLargeAllocLog,
  clearLargeAllocLog,
  enableAllAllocDebug,
  disableAllAllocDebug,
  clearAllocStacks,
  setAllocStep,
  snapshotLeakedAllocs,
  snapshotLeakedAllocsForStep,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  getAndResetFlowCounters,
  getTrackedBuffers,
  getLeakedSizeHistogramForStep,
} from "./memory-tracker";
import { dispatchFusedKernel } from "./fusion-dispatch";
import {
  isProfilingEnabled,
  profileApiCall,
  initGpuTimestamps,
  getTimestampWrites,
  resolveGpuTimestamps,
  profileSubOpBegin,
  profileSubOpEnd,
} from "./profiler";

// Re-export memory tracking functions
export {
  gpuMemoryTracker,
  GPUMemoryLimitExceededError,
  setGPUMemoryLimit,
  getGPUMemoryLimit,
  getGPUMemoryStats,
  getGPUAllocationHistogram,
  enableLargeAllocDebug,
  getLargeAllocLog,
  clearLargeAllocLog,
  enableAllAllocDebug,
  disableAllAllocDebug,
  clearAllocStacks,
  setAllocStep,
  snapshotLeakedAllocs,
  snapshotLeakedAllocsForStep,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  getAndResetFlowCounters,
  getTrackedBuffers,
  getLeakedSizeHistogramForStep,
};

// Re-export profiler functions for use in tests
export {
  setProfilePhase,
  setProfileModule,
  getProfileModule,
  readGpuTimestamps,
  printProfileSummary,
  resetProfileStats,
  isProfilingEnabled,
  getProfileJSON,
  writeProfileJSON,
  recordPlanAnalysis,
  type PlanAnalysis,
} from "./profiler";

// ============================================================================
// Buffer Pool for GPU Buffer Reuse (§14)
// ============================================================================

/**
 * Storage buffer usage flags - must match for pool compatibility.
 * Uniform buffers and other usages are NOT pooled.
 */
const STORAGE_BUFFER_USAGE = 0x0080 | 0x0004 | 0x0008; // STORAGE | COPY_SRC | COPY_DST

/**
 * Buffer pool with fence integration per spec §14.
 * Buffers are not immediately available for reuse - they go through a pending
 * queue and become available only after GPU work completes.
 * Only pools storage buffers with STORAGE | COPY_SRC | COPY_DST usage.
 */
class SimpleBufferPool {
  private pool = new Map<number, GPUBuffer[]>();
  private pooledBytes = 0;
  private pooledBufferSet = new Set<GPUBuffer>(); // track which buffers are from pool
  private reuseCount = 0;
  private allocCount = 0;
  private enabled = true;
  private debugTrace = false;
  // Track new allocations by size class for pool pre-warming
  private newAllocsByClass = new Map<number, number>();
  // Stable buffer identity for deterministic pool ordering
  private nextBufferPoolId = 1;
  private bufferPoolIdMap = new WeakMap<GPUBuffer, number>();

  // Window-based demand tracking for reservation
  private windowTracking = false;
  private currentWindowId = 0;
  private windowDemand: Array<Map<number, { acquires: number; releases: number }>> = [];
  private reservation: Map<number, number> | null = null;

  // Total byte budget for pooled + pending buffers.
  // Default: no limit (Infinity). Like PyTorch's CUDA caching allocator, the
  // pool grows as needed and relies on the memory tracker's eviction mechanism
  // (in createTrackedBuffer) for pressure relief. Users can set an explicit
  // cap via setMaxPoolBytes() or the TORCHLETTE_POOL_BUDGET_MB env var.
  private maxPoolBytes = Infinity;

  // Fence integration (§14): buffers pending GPU completion
  private pendingRelease: Array<{
    buffer: GPUBuffer;
    sizeClass: number;
    size: number;
  }> = [];
  private pendingReleaseBytes = 0;
  private queue: GPUQueue | null = null;
  private fencePromise: Promise<void> | null = null;

  // Reference counting: track how many owning tensors reference each GPUBuffer.
  // Only owning tensors (ownsBuffer=true) participate. When refcount drops to 0,
  // the buffer is eligible for pool promotion or reuse from pendingRelease.
  private bufferLiveCount = new Map<GPUBuffer, number>();

  // Stats for understanding buffer reuse patterns
  private acquireFromPool = 0;
  private acquireFromPending = 0;
  private acquireNew = 0;
  private releaseToPool = 0;
  private releaseToDestroy = 0;

  setDebugTrace(enabled: boolean): void {
    this.debugTrace = enabled;
  }

  getPendingReleaseCount(): number {
    return this.pendingRelease.length;
  }

  getDetailedStats() {
    return {
      acquireFromPool: this.acquireFromPool,
      acquireFromPending: this.acquireFromPending,
      acquireNew: this.acquireNew,
      releaseToPool: this.releaseToPool,
      releaseToDestroy: this.releaseToDestroy,
      pendingReleaseCount: this.pendingRelease.length,
      pendingReleaseBytes: this.pendingRelease.reduce((sum, p) => sum + p.size, 0),
    };
  }

  resetDetailedStats(): void {
    this.acquireFromPool = 0;
    this.acquireFromPending = 0;
    this.acquireNew = 0;
    this.releaseToPool = 0;
    this.releaseToDestroy = 0;
  }

  /** Increment reference count for a buffer (called when an owning tensor is created). */
  incRef(buffer: GPUBuffer): void {
    this.bufferLiveCount.set(buffer, (this.bufferLiveCount.get(buffer) ?? 0) + 1);
  }

  /** Decrement reference count for a buffer (called when an owning tensor is destroyed or ownership transferred). */
  decRef(buffer: GPUBuffer): void {
    const c = this.bufferLiveCount.get(buffer);
    if (c === undefined) return;
    if (c <= 1) this.bufferLiveCount.delete(buffer);
    else this.bufferLiveCount.set(buffer, c - 1);
  }

  /** Check if a buffer is still referenced by any owning tensor. */
  isLive(buffer: GPUBuffer): boolean {
    return this.bufferLiveCount.has(buffer);
  }

  /**
   * Set the GPU queue for fence integration.
   * Must be called after WebGPU initialization.
   */
  setQueue(queue: GPUQueue): void {
    this.queue = queue;
  }

  /**
   * Try to acquire a storage buffer from the pool, or return null if none available.
   * Only returns buffers with STORAGE | COPY_SRC | COPY_DST usage.
   *
   * Checks both the main pool AND pendingRelease queue. Buffers in pendingRelease
   * are safe to reuse within the same execution because WebGPU guarantees
   * submission-order execution - writes to reused buffers happen after reads.
   */
  /**
   * Try to acquire a specific preferred buffer from the pool.
   * Returns the preferred buffer if found and not in writeSet, else null.
   * Used by resolveOutputBuffer to stabilize buffer assignments across steps,
   * enabling bind group cache hits.
   */
  acquirePreferred(sizeBytes: number, preferred: GPUBuffer): GPUBuffer | null {
    if (!this.enabled) return null;
    // writeSet check removed: a buffer in the writeSet was already acquired (removed
    // from pool) by a prior dispatch — it cannot also be in the pool. Verified via
    // diagnostic counters (always 0 hits).

    const sizeClass = getSizeClass(sizeBytes);
    const pooledBuffers = this.pool.get(sizeClass);
    if (!pooledBuffers) return null;

    const idx = pooledBuffers.indexOf(preferred);
    if (idx === -1) return null;

    pooledBuffers.splice(idx, 1);
    const actualSize = getSizeForClass(sizeClass);
    this.pooledBytes -= actualSize;
    this.reuseCount++;
    this.acquireFromPool++;
    this.recordAcquire(sizeClass);
    this.pooledBufferSet.add(preferred);
    gpuMemoryTracker.trackAllocation(preferred, actualSize);
    return preferred;
  }

  /**
   * Sort each pool bucket by stable buffer ID for deterministic acquire order.
   * Call at step start and after each flush to ensure the same set of buffers
   * yields the same LIFO order for bind group cache stability.
   */
  sortPoolBuckets(): void {
    const idMap = this.bufferPoolIdMap;
    for (const bucket of this.pool.values()) {
      if (bucket.length > 1) {
        bucket.sort((a, b) => (idMap.get(a) ?? 0) - (idMap.get(b) ?? 0));
      }
    }
  }

  acquire(sizeBytes: number): GPUBuffer | null {
    if (!this.enabled) return null;

    const sizeClass = getSizeClass(sizeBytes);

    // Check the main pool (buffers from previous executions).
    // writeSet check removed: pool-resident buffers cannot be in the writeSet because
    // they haven't been acquired yet this encoder scope. Verified via diagnostic counters.
    const pooledBuffers = this.pool.get(sizeClass);
    if (pooledBuffers && pooledBuffers.length > 0) {
      // Take from end (LIFO) for deterministic order after sortPoolBuckets()
      const buffer = pooledBuffers.pop()!;
      const actualSize = getSizeForClass(sizeClass);
      this.pooledBytes -= actualSize;
      this.reuseCount++;
      this.acquireFromPool++;
      this.recordAcquire(sizeClass);
      // Track that this buffer is from pool for release
      this.pooledBufferSet.add(buffer);
      // Re-track allocation (was deallocated when released to pool)
      gpuMemoryTracker.trackAllocation(buffer, actualSize);
      if (this.debugTrace) {
        console.log(`[pool] acquire from POOL: ${(actualSize / 1e6).toFixed(2)} MB`);
      }
      return buffer;
    }

    // Then check pendingRelease for same-execution reuse.
    // This enables actual memory savings within a single execution.
    // Skip when batching — pending buffers may still be referenced by
    // collected command buffers that haven't been submitted yet.
    // Also skip when a deferred fence is outstanding — those pending buffers
    // are from a previous step and the GPU may still be using them.
    // Also skip during shared encoder scope — pending buffers may have been
    // written by earlier passes and their command buffers not yet submitted.
    if (!activeBatch && !pendingFencePromise && !sharedEncoder) {
      const pendingIdx = this.pendingRelease.findIndex(
        (p) => p.sizeClass === sizeClass && !this.bufferLiveCount.has(p.buffer),
      );
      if (pendingIdx !== -1) {
        const { buffer, size } = this.pendingRelease.splice(pendingIdx, 1)[0];
        this.pendingReleaseBytes -= size;
        this.reuseCount++;
        this.acquireFromPending++;
        this.recordAcquire(sizeClass);
        this.pooledBufferSet.add(buffer);
        // Re-track allocation (was deallocated when added to pending)
        gpuMemoryTracker.trackAllocation(buffer, size);
        if (this.debugTrace) {
          console.log(`[pool] acquire from PENDING: ${(size / 1e6).toFixed(2)} MB (${this.pendingRelease.length} remaining)`);
        }
        return buffer;
      }
    }

    return null;
  }

  /**
   * Called when a new buffer is allocated (not from pool).
   */
  trackNewAllocation(sizeBytes: number): void {
    this.acquireNew++;
    const sc = getSizeClass(sizeBytes);
    this.newAllocsByClass.set(sc, (this.newAllocsByClass.get(sc) ?? 0) + 1);
    this.recordAcquire(sc);
    if (this.debugTrace) {
      console.log(`[pool] NEW allocation: ${(sizeBytes / 1e6).toFixed(2)} MB`);
    }
  }

  /**
   * Pre-warm the pool by creating buffers for size classes that had misses last step.
   * Call at the start of each step (before opening the shared encoder).
   */
  prewarm(device: GPUDevice): void {
    for (const [sizeClass, count] of this.newAllocsByClass) {
      const size = getSizeForClass(sizeClass);
      for (let i = 0; i < count; i++) {
        if (this.pooledBytes + this.pendingReleaseBytes + size > this.maxPoolBytes) break;
        const buf = device.createBuffer({ size, usage: STORAGE_BUFFER_USAGE });
        let bucket = this.pool.get(sizeClass);
        if (!bucket) { bucket = []; this.pool.set(sizeClass, bucket); }
        bucket.push(buf);
        this.pooledBytes += size;
      }
    }
    this.newAllocsByClass.clear();
  }

  /** Start recording window demand for this step. */
  beginWindowTracking(): void {
    this.windowTracking = true;
    this.currentWindowId = 0;
    this.windowDemand = [new Map()];
  }

  /** Advance to the next reclaim window. Call at each flushBufferPool boundary. */
  beginWindow(): void {
    if (!this.windowTracking) return;
    this.currentWindowId++;
    while (this.windowDemand.length <= this.currentWindowId) {
      this.windowDemand.push(new Map());
    }
  }

  /** Stop recording and compute reservation for next step. */
  endWindowTracking(): void {
    if (!this.windowTracking) return;
    this.windowTracking = false;
    this.computeReservation();
  }

  private recordAcquire(sizeClass: number): void {
    if (!this.windowTracking) return;
    const wm = this.windowDemand[this.currentWindowId];
    if (!wm) return;
    const e = wm.get(sizeClass) ?? { acquires: 0, releases: 0 };
    e.acquires++;
    wm.set(sizeClass, e);
  }

  private recordRelease(sizeClass: number): void {
    if (!this.windowTracking) return;
    const wm = this.windowDemand[this.currentWindowId];
    if (!wm) return;
    const e = wm.get(sizeClass) ?? { acquires: 0, releases: 0 };
    e.releases++;
    wm.set(sizeClass, e);
  }

  private computeReservation(): void {
    const allSc = new Set<number>();
    for (const wm of this.windowDemand) {
      for (const sc of wm.keys()) allSc.add(sc);
    }

    const reservation = new Map<number, number>();
    for (const sc of allSc) {
      let cumAcq = 0, cumRel = 0, maxDeficit = 0;
      for (let w = 0; w < this.windowDemand.length; w++) {
        const e = this.windowDemand[w].get(sc);
        cumAcq += e?.acquires ?? 0;
        // Deficit at window w = cumAcquires[0..w] - cumReleases[0..w-1]
        maxDeficit = Math.max(maxDeficit, cumAcq - cumRel);
        cumRel += e?.releases ?? 0;
      }
      reservation.set(sc, maxDeficit);
    }
    this.reservation = reservation;
  }

  /**
   * Reserve buffers to match the computed window-demand reservation.
   * Only creates buffers for size classes where the pool has a deficit.
   * Replaces prewarm() — call at beginStep() BEFORE opening shared encoder.
   */
  reserve(device: GPUDevice): void {
    if (!this.reservation) {
      // No reservation yet (step 0): fall back to prewarm
      this.prewarm(device);
      return;
    }

    for (const [sizeClass, needed] of this.reservation) {
      const size = getSizeForClass(sizeClass);
      const bucket = this.pool.get(sizeClass);
      const have = bucket?.length ?? 0;
      const deficit = needed - have;
      if (deficit <= 0) continue;

      for (let i = 0; i < deficit; i++) {
        if (this.pooledBytes + this.pendingReleaseBytes + size > this.maxPoolBytes) break;
        const buf = device.createBuffer({ size, usage: STORAGE_BUFFER_USAGE });
        let b = this.pool.get(sizeClass);
        if (!b) { b = []; this.pool.set(sizeClass, b); }
        b.push(buf);
        this.pooledBytes += size;
      }
    }
    this.newAllocsByClass.clear();
  }

  /**
   * Release a buffer back to the pool for reuse.
   * Per spec §14, the buffer goes to a pending queue first and only becomes
   * available after GPU work completes (fence signaled).
   * Returns true if the buffer was queued for pooling, false if it should be destroyed.
   */
  release(buffer: GPUBuffer, sizeBytes: number, usage: number): boolean {
    if (!this.enabled) return false;

    // Only pool storage buffers with compatible usage
    if (usage !== STORAGE_BUFFER_USAGE) {
      this.releaseToDestroy++;
      if (this.debugTrace) {
        console.log(`[pool] release INCOMPATIBLE (wrong usage): ${(sizeBytes / 1e6).toFixed(2)} MB`);
      }
      return false;
    }

    const sizeClass = getSizeClass(sizeBytes);
    const actualSize = getSizeForClass(sizeClass);

    // Remove from active tracking
    this.pooledBufferSet.delete(buffer);

    // Check total byte budget: pooledBytes tracks main pool, pendingReleaseBytes tracks pending
    if (this.pooledBytes + this.pendingReleaseBytes + actualSize > this.maxPoolBytes) {
      this.releaseToDestroy++;
      if (this.debugTrace) {
        console.log(`[pool] release BUDGET (${((this.pooledBytes + this.pendingReleaseBytes) / 1e6).toFixed(1)}MB + ${(actualSize / 1e6).toFixed(2)}MB > ${(this.maxPoolBytes / 1e6).toFixed(0)}MB): ${(actualSize / 1e6).toFixed(2)} MB`);
      }
      return false; // Don't pool, destroy instead
    }

    // Add to pending queue - will be available after GPU work completes
    this.pendingRelease.push({ buffer, sizeClass, size: actualSize });
    this.pendingReleaseBytes += actualSize;
    this.releaseToPool++;
    // Note: recordRelease is called in flushPendingToPool() when the buffer
    // actually promotes to the pool, NOT here. Buffers blocked by liveCount
    // filtering never become pool-available, so recording here would cause
    // the reservation formula to overestimate available releases.

    // Track deallocation immediately so user-facing stats reflect tensor usage
    // (not pool cache). The pool holds the buffer for reuse but from the
    // tracker's perspective, the memory is "logically" freed.
    gpuMemoryTracker.trackDeallocation(buffer);

    if (this.debugTrace) {
      console.log(`[pool] release to PENDING: ${(actualSize / 1e6).toFixed(2)} MB (now ${this.pendingRelease.length} pending)`);
    }

    // Schedule fence if we have a queue and no pending fence
    this.scheduleFence();

    return true;
  }

  /** Buffers to destroy after fence (not pool, just destroy safely) */
  private pendingDestroy: Array<{ buffer: GPUBuffer; size: number }> = [];

  /**
   * Queue a buffer for destruction after GPU work completes.
   * Use this for buffers that can't be pooled but need safe destruction.
   * This prevents "buffer destroyed while in use" validation errors.
   *
   * @param buffer The GPU buffer to destroy
   * @param size The buffer size in bytes (for memory tracking)
   */
  deferredDestroy(buffer: GPUBuffer, size: number): void {
    // Arena buffers are owned by the arena — never destroy them.
    // This check covers ALL callers (direct pool calls and wrapper).
    if (arenaBufferSet.has(buffer)) return;
    // Replay-pinned buffers are referenced by recorded bind groups — never destroy.
    if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) return;
    // Track deallocation immediately - the memory is now "freeable"
    gpuMemoryTracker.trackDeallocation(buffer);
    // When batching or shared encoder active, defer until after submit
    if (activeBatch) {
      activeBatch.deferredDestroyBuffers.push(buffer);
      return;
    }
    this.pendingDestroy.push({ buffer, size });
    this.scheduleFence();
  }

  /**
   * Queue an untracked buffer for deferred destruction.
   * Use for tiny params buffers that are not in the memory tracker.
   */
  deferredDestroyUntracked(buffer: GPUBuffer): void {
    // Replay-pinned buffers are referenced by recorded bind groups — never destroy.
    if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) return;
    // When batching, command buffers haven't been submitted yet.
    // scheduleFence()'s onSubmittedWorkDone would resolve before the batch submits,
    // destroying the buffer while it's still referenced by collected command buffers.
    if (activeBatch) {
      activeBatch.deferredDestroyBuffers.push(buffer);
      return;
    }
    this.pendingDestroy.push({ buffer, size: 0 });
    this.scheduleFence();
  }

  /**
   * Schedule a fence to move pending buffers to the pool.
   * Uses onSubmittedWorkDone to wait for GPU completion.
   *
   * IMPORTANT: We snapshot the current pending arrays at fence creation time.
   * The fence's onSubmittedWorkDone() only covers GPU work submitted BEFORE the
   * fence was created. Buffers added to pending AFTER the fence was created may
   * still have in-flight GPU work, so they must NOT be processed by this fence.
   * They'll be handled by a subsequent fence.
   */
  private scheduleFence(): void {
    // Don't use async onSubmittedWorkDone() fences - they fire at unpredictable
    // times via microtask queue and cause "buffer destroyed while in use" errors
    // when the fence callback destroys a buffer still referenced by pending GPU work.
    //
    // Instead, buffers stay in pendingRelease/pendingDestroy and are only processed:
    // 1. pendingRelease: acquire() can reuse them immediately (safe because WebGPU
    //    guarantees submission-order execution between queue.submit calls)
    // 2. pendingDestroy: processed at explicit sync points (waitAndFlushPending,
    //    markStep, or read()) where we know GPU work is complete
    //
    // This trades slightly higher memory usage for correctness.
    return;
  }

  /**
   * Move all pending buffers to the available pool and destroy deferred buffers.
   * Only used when fence support is unavailable (synchronous fallback).
   */
  private flushPending(): void {
    this.flushPendingToPool();
    // Note: pendingDestroy is NOT processed here. Use destroyPendingBuffers()
    // after GPU sync to safely destroy those buffers.
  }

  /**
   * Move pending-release buffers into the available pool.
   * Always safe to call — these buffers are reusable after WebGPU submission ordering.
   */
  private flushPendingToPool(): void {
    // Move pending buffers to pool, but skip any that are still referenced
    // by live owning tensors (refcount > 0). Those stay in pendingRelease
    // until the owning tensor is destroyed.
    const remaining: typeof this.pendingRelease = [];
    let remainingBytes = 0;
    for (const entry of this.pendingRelease) {
      if (this.bufferLiveCount.has(entry.buffer)) {
        remaining.push(entry);
        remainingBytes += entry.size;
        continue;
      }
      let bucket = this.pool.get(entry.sizeClass);
      if (!bucket) { bucket = []; this.pool.set(entry.sizeClass, bucket); }
      bucket.push(entry.buffer);
      this.pooledBytes += entry.size;
      // Record at promotion time (not release time) so the reservation formula
      // only counts buffers that actually become pool-available.
      this.recordRelease(entry.sizeClass);
    }
    this.pendingRelease = remaining;
    this.pendingReleaseBytes = remainingBytes;
  }

  /**
   * Destroy all pending-destroy buffers. Only call after GPU sync
   * (queue.onSubmittedWorkDone()) to avoid "buffer destroyed while in use" errors.
   */
  destroyPendingBuffers(): void {
    for (const { buffer } of this.pendingDestroy) {
      // Replay-pinned buffers must survive
      if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) continue;
      try {
        buffer.destroy();
      } catch {
        // Ignore destroy errors (e.g. buffer already destroyed or invalid)
      }
    }
    this.pendingDestroy = [];
  }

  /**
   * Track a newly allocated buffer (for stats).
   */
  trackAllocation(_buffer: GPUBuffer, _sizeBytes: number): void {
    this.allocCount++;
  }

  /**
   * Check if a buffer is from the pool.
   */
  isFromPool(buffer: GPUBuffer): boolean {
    return this.pooledBufferSet.has(buffer);
  }

  /**
   * Mark a buffer as from pool (for release tracking).
   */
  markAsFromPool(buffer: GPUBuffer): void {
    this.pooledBufferSet.add(buffer);
    if (!this.bufferPoolIdMap.has(buffer)) {
      this.bufferPoolIdMap.set(buffer, this.nextBufferPoolId++);
    }
  }

  /**
   * Get the pool bucket for a given size class. Used by pre-pinning to extract
   * specific buffers before dispatches run.
   */
  getPoolBucket(sizeClass: number): GPUBuffer[] | undefined {
    return this.pool.get(sizeClass);
  }

  /**
   * Adjust pooledBytes tracking (e.g. when pre-pinning extracts buffers from pool).
   */
  adjustPooledBytes(delta: number): void {
    this.pooledBytes += delta;
  }

  /**
   * Return a buffer directly to the pool bucket (bypassing pendingRelease).
   * Used when a pre-pinned buffer wasn't consumed and needs to go back.
   */
  returnToPool(buffer: GPUBuffer, sizeClass: number): void {
    let bucket = this.pool.get(sizeClass);
    if (!bucket) { bucket = []; this.pool.set(sizeClass, bucket); }
    bucket.push(buffer);
    this.pooledBytes += getSizeForClass(sizeClass);
  }

  /**
   * Record a pool acquire for window-demand tracking (used by pre-pin path).
   */
  recordAcquireForPin(sizeClass: number): void {
    this.reuseCount++;
    this.acquireFromPool++;
    this.recordAcquire(sizeClass);
  }

  /**
   * Get pool statistics.
   */
  stats(): {
    pooledBuffers: number;
    pendingBuffers: number;
    pooledBytes: number;
    reuseCount: number;
    allocCount: number;
    reuseRate: number;
  } {
    let totalPooled = 0;
    for (const buffers of this.pool.values()) {
      totalPooled += buffers.length;
    }
    const total = this.reuseCount + this.allocCount;
    return {
      pooledBuffers: totalPooled,
      pendingBuffers: this.pendingRelease.length,
      pooledBytes: this.pooledBytes,
      reuseCount: this.reuseCount,
      allocCount: this.allocCount,
      reuseRate: total > 0 ? this.reuseCount / total : 0,
      pendingRelease: this.pendingRelease.length,
      pendingDestroy: this.pendingDestroy.length,
    };
  }

  /**
   * Clear the pool and destroy all pooled buffers.
   */
  clear(): void {
    // Don't call GPUBuffer.destroy() - Dawn's async timing causes
    // "buffer destroyed while in use" errors. Just drop references.
    this.pool.clear();
    this.pooledBytes = 0;
    this.pooledBufferSet.clear();
    this.pendingRelease = [];
    this.pendingReleaseBytes = 0;
    this.pendingDestroy = [];
    this.bufferLiveCount.clear();
  }

  /**
   * Evict pooled buffers to free up memory.
   * Called automatically when memory pressure is high.
   * Returns the number of bytes freed.
   *
   * Note: This only evicts from the main pool (buffers that have passed the GPU fence).
   * Pending buffers are NOT destroyed here as they may still be in use by GPU work.
   * If you need to free pending buffers, call waitAndFlushPending() first.
   */
  evictBuffers(bytesNeeded: number): number {
    let bytesFreed = 0;

    // Evict from largest size classes first (most memory savings)
    const sizeClasses = Array.from(this.pool.keys()).sort((a, b) => b - a);

    for (const sizeClass of sizeClasses) {
      const buffers = this.pool.get(sizeClass);
      if (!buffers || buffers.length === 0) continue;

      const sizePerBuffer = getSizeForClass(sizeClass);

      while (buffers.length > 0 && bytesFreed < bytesNeeded) {
        const buffer = buffers.pop()!;
        // Replay-pinned buffers must survive — push back and skip.
        if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) {
          buffers.push(buffer);
          break; // Can't evict from this size class
        }
        this.pooledBufferSet.delete(buffer);
        this.pooledBytes -= sizePerBuffer;
        // Track deallocation and destroy the buffer to actually free GPU memory.
        // These buffers have already passed the GPU fence (promoted from pending
        // to free pool), so they're safe to destroy.
        gpuMemoryTracker.trackDeallocation(buffer);
        try { buffer.destroy(); } catch { /* already destroyed */ }
        bytesFreed += sizePerBuffer;
      }

      if (buffers.length === 0) {
        this.pool.delete(sizeClass);
      }

      if (bytesFreed >= bytesNeeded) break;
    }

    return bytesFreed;
  }

  /**
   * Enable or disable buffer pooling.
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    if (!enabled) {
      this.clear();
    }
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Set the maximum byte budget for pooled + pending buffers.
   * Pass Infinity (or null) for no limit (PyTorch-like behavior).
   * When set, release() will destroy buffers rather than pooling them
   * if the budget would be exceeded, and reserve() will stop pre-creating
   * buffers at the budget boundary.
   */
  setMaxPoolBytes(bytes: number | null): void {
    this.maxPoolBytes = bytes == null ? Infinity : bytes;
  }

  getMaxPoolBytes(): number {
    return this.maxPoolBytes;
  }

  /**
   * Flush pending buffers to make them immediately available for reuse.
   *
   * This is safe to call when we know GPU work has completed (e.g., at the
   * start of backward pass, since backward depends on forward results being
   * ready). This enables checkpoint memory savings by making buffers available
   * for recomputation before the async onSubmittedWorkDone() callback runs.
   */
  flushPendingToAvailable(): void {
    this.flushPending();
  }

  /**
   * Flush pending-release to pool AND destroy pending-destroy buffers,
   * after awaiting GPU sync. Safe to call from any context.
   */
  async flushAndDestroyPending(): Promise<void> {
    if (!this.queue) {
      // No queue — just flush pool, skip destroy (can't sync)
      this.flushPendingToPool();
      return;
    }
    await this.queue.onSubmittedWorkDone();
    this.flushPendingToPool();
    this.destroyPendingBuffers();
  }

  /**
   * Wait for GPU work to complete and then flush pending buffers.
   * This is a synchronous wait - use sparingly when memory pressure is critical.
   * After this, evictBuffers() can be called to free more memory.
   */
  async waitAndFlushPending(): Promise<void> {
    if (!this.queue) return;
    // Wait for all GPU work to complete
    await this.queue.onSubmittedWorkDone();
    // Now safe to flush pending buffers and destroy deferred buffers
    this.flushPendingToPool();
    this.destroyPendingBuffers();
  }

  /**
   * Get bytes in pending queues (not yet safe to destroy).
   */
  getPendingBytes(): number {
    const pendingReleaseBytes = this.pendingRelease.reduce((sum, p) => sum + p.size, 0);
    const pendingDestroyBytes = this.pendingDestroy.reduce((sum, p) => sum + p.size, 0);
    return pendingReleaseBytes + pendingDestroyBytes;
  }

  /**
   * Get total bytes held by the pool (free + pending release).
   * These bytes are in GPU memory but not tracked by the memory tracker
   * (trackDeallocation was called when buffers were released to pool).
   * Used by createTrackedBuffer to check physical GPU memory pressure.
   */
  getTotalHeldBytes(): number {
    return this.pooledBytes + this.pendingReleaseBytes;
  }
}

/** Global buffer pool instance */
const bufferPool = new SimpleBufferPool();

/**
 * Get buffer pool statistics.
 */
export function getBufferPoolStats(): ReturnType<SimpleBufferPool["stats"]> {
  return bufferPool.stats();
}

/**
 * Enable or disable buffer pooling.
 */
export function setBufferPoolEnabled(enabled: boolean): void {
  bufferPool.setEnabled(enabled);
}

/**
 * Clear the buffer pool (destroy all pooled buffers).
 */
export function clearBufferPool(): void {
  bufferPool.clear();
}

/**
 * Set the maximum byte budget for the buffer pool.
 *
 * Like PyTorch's CUDA caching allocator, the default is no limit —
 * the pool grows to cache all released buffers, and memory pressure
 * is handled by the eviction mechanism in createTrackedBuffer.
 *
 * Set an explicit limit to cap how much GPU memory the pool holds:
 *   setBufferPoolBudget(2 * 1024 * 1024 * 1024)  // 2 GB
 *   setBufferPoolBudget(null)                      // no limit (default)
 *
 * Can also be set via TORCHLETTE_POOL_BUDGET_MB environment variable.
 */
export function setBufferPoolBudget(bytes: number | null): void {
  bufferPool.setMaxPoolBytes(bytes);
}

/**
 * Get the current buffer pool budget in bytes (Infinity if no limit).
 */
export function getBufferPoolBudget(): number {
  return bufferPool.getMaxPoolBytes();
}

/**
 * Flush pending buffers to make them immediately available for reuse.
 *
 * Call this when GPU work is known to be complete (e.g., at the start of
 * backward pass). This enables checkpoint memory savings by making buffers
 * available before the async onSubmittedWorkDone() callback runs.
 */
export function flushBufferPool(): void {
  bufferPool.flushPendingToAvailable();
  bufferPool.sortPoolBuckets(); // deterministic acquire order for bind group cache
  bufferPool.beginWindow();  // Advance window counter for demand tracking
}

/**
 * Flush pending buffers to pool AND safely destroy pending-destroy buffers
 * after GPU sync. Use this at markStep() to prevent memory leaks from
 * non-poolable buffers (mappedAtCreation, staging buffers, etc.).
 */
export async function flushBufferPoolWithSync(): Promise<void> {
  await bufferPool.flushAndDestroyPending();
}

/**
 * Decrement the owning-tensor reference count for a GPUBuffer.
 * Use this when transferring buffer ownership outside the normal
 * createTensor/destroy lifecycle (e.g., external donation consumers).
 */
export function decRefBuffer(buffer: GPUBuffer): void {
  bufferPool.decRef(buffer);
}

// ============================================================================
// Deferred GPU Fence (Phase 2: Async markStep)
// ============================================================================

/**
 * Pending fence promise from a previous markStep. When set, the GPU was still
 * working when the previous markStep returned. The next markStep must await
 * this before destroying any buffers.
 */
let pendingFencePromise: Promise<void> | null = null;

/**
 * Buffers queued for destruction after the pending fence resolves.
 * These are from destroyUnreachable() — we can't destroy them until
 * the GPU is done with them.
 */
let deferredPendingRelease: boolean = false;

/**
 * Issue a GPU fence without awaiting it. The fence promise is stored
 * and will be awaited at the start of the next markStep.
 */
export function issueDeferredFence(): void {
  const ctx = context;
  if (!ctx) return;
  if (typeof ctx.queue.onSubmittedWorkDone !== "function") return;

  // Issue the fence — don't await
  pendingFencePromise = ctx.queue.onSubmittedWorkDone();
  deferredPendingRelease = true;
}

/**
 * Await the pending fence from a previous markStep, then flush/destroy
 * pending buffers. Must be called before any buffer reuse in a new step.
 */
export async function awaitDeferredFence(): Promise<void> {
  if (pendingFencePromise) {
    await pendingFencePromise;
    pendingFencePromise = null;
  }
  if (deferredPendingRelease) {
    bufferPool.flushPendingToPool();
    bufferPool.destroyPendingBuffers();
    deferredPendingRelease = false;
  }
}

/**
 * Check if there's a pending deferred fence.
 */
export function hasDeferredFence(): boolean {
  return pendingFencePromise !== null;
}

/**
 * Get detailed buffer pool statistics for debugging.
 */
export function getBufferPoolDetailedStats(): ReturnType<
  SimpleBufferPool["getDetailedStats"]
> {
  return bufferPool.getDetailedStats();
}

/**
 * Reset detailed buffer pool statistics.
 */
export function resetBufferPoolDetailedStats(): void {
  bufferPool.resetDetailedStats();
}

/**
 * Enable or disable debug tracing for buffer pool operations.
 */
export function setBufferPoolDebugTrace(enabled: boolean): void {
  bufferPool.setDebugTrace(enabled);
}

/**
 * Queue a buffer for deferred destruction after GPU work completes.
 * Tracks deallocation immediately in the memory tracker.
 * Used by engine layer for buffers not managed by WebGPUTensor.destroy().
 */
export function deferredDestroyBuffer(buffer: GPUBuffer, size: number): void {
  // Arena buffers are owned by the arena — don't destroy them.
  // The arena will reuse the buffer on the next step.
  if (arenaBufferSet.has(buffer)) return;
  // Replay-pinned buffers are referenced by recorded bind groups — don't destroy.
  if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) return;
  bufferPool.deferredDestroy(buffer, size);
}

/**
 * Acquire a buffer from the pool for output allocation.
 * Returns null if no suitable buffer is available.
 * The caller is responsible for returning the buffer to the pool.
 */
export function acquirePooledBuffer(sizeBytes: number): GPUBuffer | null {
  return bufferPool.acquire(sizeBytes) as GPUBuffer | null;
}

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
  if (activeArena) {
    const idx = arenaAllocIndex++;
    const arenaBuffer = arenaAllocAt(activeArena.alloc, idx, sizeBytes);
    if (arenaBuffer) return arenaBuffer;
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
  tensor: import("../types").BackendTensor,
): GPUBuffer | null {
  const t = tensor as WebGPUTensor;

  // Can only donate if tensor owns the buffer
  if (!t.ownsBuffer) {
    return null;
  }

  // Check buffer has pool-compatible usage (STORAGE | COPY_SRC | COPY_DST)
  const bufferUsage = (t.buffer as any).usage ?? 0;
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
  tensor: import("../types").BackendTensor,
): number {
  const t = tensor as WebGPUTensor;
  return (t.buffer as any).size ?? 0;
}

type GPUBuffer = {
  getMappedRange(): ArrayBuffer;
  mapAsync(mode: number): Promise<void>;
  unmap(): void;
};

type GPUComputePipeline = {
  getBindGroupLayout(index: number): unknown;
};

type GPUComputePass = {
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
  setBindGroup(index: number, group: unknown): void;
  setPipeline(pipeline: GPUComputePipeline): void;
};

type GPUCommandEncoder = {
  beginComputePass(): GPUComputePass;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): unknown;
};

type GPUQueue = {
  onSubmittedWorkDone?: () => Promise<void>;
  submit(commands: unknown[]): void;
  writeBuffer(buffer: GPUBuffer, offset: number, data: Uint32Array): void;
};

type GPUDevice = {
  createBindGroup(descriptor: {
    layout: unknown;
    entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }>;
  }): unknown;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer;
  createCommandEncoder(): GPUCommandEncoder;
  createComputePipeline(descriptor: {
    layout: "auto";
    compute: { module: unknown; entryPoint: string };
  }): GPUComputePipeline;
  createShaderModule(descriptor: { code: string }): unknown;
  queue: GPUQueue;
};

type GPUAdapterLimits = {
  maxStorageBufferBindingSize?: number;
  maxStorageBuffersPerShaderStage?: number;
};

type GPUAdapter = {
  features?: Set<string>;
  limits?: GPUAdapterLimits;
  requestDevice(descriptor?: {
    requiredFeatures?: string[];
    requiredLimits?: GPUAdapterLimits;
  }): Promise<GPUDevice>;
};

type WebGPUProvider = {
  requestAdapter(): Promise<GPUAdapter | null>;
};

type WebGPUModule = {
  create: (args: string[]) => WebGPUProvider;
  globals: Record<string, unknown>;
};

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

// Maximum workgroups per dimension in WebGPU (per spec)
const MAX_WORKGROUPS_PER_DIM = 65535;

/**
 * Compute 2D dispatch dimensions for large workloads.
 * WebGPU has a limit of 65535 workgroups per dimension.
 * For large tensors, we use 2D dispatch (x, y).
 */
function compute2DDispatch(totalWorkgroups: number): {
  x: number;
  y: number;
  gridSizeX: number;
} {
  if (totalWorkgroups <= MAX_WORKGROUPS_PER_DIM) {
    return { x: totalWorkgroups, y: 1, gridSizeX: totalWorkgroups };
  }
  // Split into 2D grid
  const x = MAX_WORKGROUPS_PER_DIM;
  const y = Math.ceil(totalWorkgroups / MAX_WORKGROUPS_PER_DIM);
  return { x, y, gridSizeX: x };
}

/**
 * Generate WGSL code to compute flat index from 2D gid.
 * Use this in shaders when dispatch might use 2D workgroups.
 */
function flatIndexFromGid(gridSizeX: number): string {
  if (gridSizeX <= MAX_WORKGROUPS_PER_DIM) {
    return `let idx = gid.x;`;
  }
  return `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`;
}

// Legacy matmul constants (kept for reference, now using tiled matmul)
const MATMUL_WORKGROUP_X = 8;
const MATMUL_WORKGROUP_Y = 8;

type WebGPUTensor = BackendTensor & {
  buffer: GPUBuffer;
  size: number;
  /** Strides in elements for each dimension */
  strides: number[];
  /** Offset in elements from start of buffer */
  offset: number;
  /** True if memory is contiguous (enables fast paths) */
  isContiguous: boolean;
  /** Data type of the tensor (defaults to f32 for backwards compatibility) */
  dtype: DType;
  /** True if this tensor owns the buffer (should destroy it) vs borrowing (view) */
  ownsBuffer: boolean;
  /** Destroy the GPU buffer and free memory */
  destroy(): void;
};

type WebGPUContext = {
  provider: WebGPUProvider;
  device: GPUDevice;
  queue: GPUQueue;
  pipelines: Map<string, GPUComputePipeline>;
  /** Whether shader-f16 feature is enabled */
  f16Supported: boolean;
};

let context: WebGPUContext | null = null;
let lastInitError: string | null = null;

/**
 * Cache of f16 weight buffers produced by the Adam kernel's dual-write.
 * Keyed by the f32 param GPUBuffer → corresponding f16 GPUBuffer.
 * Checked in cast() to skip standalone f32→f16 dispatches for AMP weights.
 */
const f16WeightCache = new Map<GPUBuffer, GPUBuffer>();

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

export function getWebGPUInitError(): string | null {
  return lastInitError;
}

/**
 * Check if f16 (half precision) is supported on the current device.
 * Returns false if WebGPU is not initialized.
 */
export function isF16Supported(): boolean {
  return context?.f16Supported ?? false;
}

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
function f32ArrayToF16Array(values: number[]): Uint16Array {
  const result = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    result[i] = f32ToF16(values[i]);
  }
  return result;
}

/**
 * Convert a Uint16Array of f16 values to an array of f32 values.
 */
function f16ArrayToF32Array(data: Uint16Array): number[] {
  const result = new Array<number>(data.length);
  for (let i = 0; i < data.length; i++) {
    result[i] = f16ToF32(data[i]);
  }
  return result;
}

export async function initWebGPU(): Promise<boolean> {
  if (context) {
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
  context = {
    provider,
    device,
    queue: device.queue,
    pipelines: new Map(),
    f16Supported: actualF16Supported,
  };
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

  registerBackend(webgpuBackend);
  return true;
}

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
  if (!context) return;
  // Destroy cached f16 weight buffers
  for (const buf of f16WeightCache.values()) {
    buf.destroy();
  }
  f16WeightCache.clear();
  clearBindGroupCache();
  destroyPersistentInfFlagBuffer();
  context.device.destroy();
  context.pipelines.clear();
  context = null;
}

/**
 * Get the raw WebGPU device and queue for advanced use cases (benchmarking, etc).
 * Returns null if WebGPU is not initialized.
 */
export function getWebGPUDevice(): {
  device: GPUDevice;
  queue: GPUQueue;
} | null {
  if (!context) return null;
  return { device: context.device, queue: context.queue };
}

function requireContext(): WebGPUContext {
  if (!context) {
    throw new Error("WebGPU backend not initialized; call initWebGPU()");
  }
  return context;
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
interface BatchExecutionContext {
  /** Collected command buffers to submit together */
  commandBuffers: GPUCommandBuffer[];
  /** Buffers to destroy after the batch submits (deferred from mid-batch destroy calls) */
  deferredDestroyBuffers: GPUBuffer[];
}

/** Active batch context (null when in immediate mode) */
let activeBatch: BatchExecutionContext | null = null;

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
// first, then their command buffer is collected and submitted together at the end.

// True Shared Encoder: a single GPUCommandEncoder is shared across multiple ops.
// Each op encodes its compute pass directly onto the shared encoder instead of
// creating its own encoder + command buffer. This eliminates ~3700 encoder
// creations per DistilGPT-2 training step.
//
// The write set tracks buffers written during the current shared encoder scope
// to prevent buffer pool aliasing — ensuring a buffer written by an earlier op
// is not reused as output for a later op within the same scope.
//
let sharedEncoder: boolean = false;
let sharedEncoderEnabled = false;
let sharedEncoderDepth = 0;
let sharedEncoderInstance: GPUCommandEncoder | null = null;
let sharedEncoderPassCount = 0;
let collectedCommandBuffers: GPUCommandBuffer[] = [];
let sharedEncoderWriteSet: Set<GPUBuffer> = new Set();
let sharedEncoderDeferredUniformBuffers: GPUBuffer[] = [];
let stepLevelScope = false; // true between beginStep() and endStep()

// Auto-flush threshold: finish current encoder and start a new one after N passes.
// This bounds the size of individual command buffers and the write/read sets.
const SHARED_ENCODER_MAX_PASSES = 2000;

// Auto-flush when deferred uniform buffers exceed this threshold.
// With paramsSequenceBuffers caching, most params are reused directly and never
// deferred. Only evicted buffers enter the deferred list, which is typically <10
// per step in steady state. Threshold matches SHARED_ENCODER_MAX_PASSES.
const PARAMS_FLUSH_THRESHOLD = 2000;

// Debug flag: set TORCHLETTE_DEBUG_SHARED_ENCODER=1 to enable verbose logging
const DEBUG_SHARED_ENCODER = typeof process !== "undefined" && !!process.env?.TORCHLETTE_DEBUG_SHARED_ENCODER;

// Current op label for GPU timestamp profiling (set from lazy.ts)
let currentOpLabel: string | null = null;
export function setCurrentOpLabel(label: string | null): void { currentOpLabel = label; }
export function getCurrentOpLabel(): string | null { return currentOpLabel; }

// Adam batch mode: when true, adamStep() skips its pre-dispatch flushSharedEncoder().
// The caller (lazy.ts) is responsible for a single flush before the Adam batch.
let adamBatchMode = false;
export function setAdamBatchMode(active: boolean): void { adamBatchMode = active; }

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

/**
 * Auto-flush the shared encoder if it has too many passes.
 * This prevents extremely large command buffers and bounds the write/read sets.
 */
function autoFlushSharedEncoder(): void {
  if (sharedEncoder && (
    sharedEncoderPassCount >= SHARED_ENCODER_MAX_PASSES ||
    sharedEncoderDeferredUniformBuffers.length >= PARAMS_FLUSH_THRESHOLD
  )) {
    flushSharedEncoder();
  }
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

let gpuSubmitCount = 0;

/**
 * Get the number of queue.submit() calls since last reset.
 */
export function getSubmitCount(): number {
  return gpuSubmitCount;
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

/**
 * Dispatch a compute pass.
 *
 * - If shared encoder is active: encodes pass directly onto the shared encoder (no new encoder/CB)
 * - If batch mode is active: creates encoder, finishes it, and collects the buffer
 * - If immediate mode: creates encoder, submits, and returns
 */
// ============================================================================
// Dispatch Recording for Replay Cache
// ============================================================================

/** Recorded dispatch entry for replay. */
export interface RecordedDispatch {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroupsX: number;
  workgroupsY: number;
  workgroupsZ: number;
  /** GPUBuffers referenced by this bind group. Populated during recording for pinning. */
  buffers?: GPUBuffer[];
}

/** Active recording buffer (null = not recording). */
let dispatchRecordingBuffer: RecordedDispatch[] | null = null;

/** Last bind group's buffer list — captured during recording for pinning. */
let lastBindGroupBuffers: GPUBuffer[] | null = null;

/** Get and clear the last bind group's buffer list. Used by matmul/fusion recording.
 *  Also immediately pins the buffers to prevent pool recycling during recording. */
export function getAndClearLastBindGroupBuffers(): GPUBuffer[] | undefined {
  const bufs = lastBindGroupBuffers ?? undefined;
  lastBindGroupBuffers = null;
  // Immediately pin buffers during recording to prevent pool recycling.
  // Without this, a params buffer could be released to the pool, reused by a
  // later op, and overwritten — making the recorded bind group reference stale data.
  if (bufs && replayPinnedBufferSet) {
    for (const b of bufs) replayPinnedBufferSet.add(b);
  }
  return bufs;
}

/** Start recording dispatches into the given buffer. */
export function startDispatchRecording(buffer: RecordedDispatch[]): void {
  dispatchRecordingBuffer = buffer;
  // Initialize pin set at recording start so buffers are pinned immediately
  // as they're captured (not deferred until after recording completes).
  if (!replayPinnedBufferSet) {
    replayPinnedBufferSet = new Set();
  }
}

/** Stop recording dispatches. */
export function stopDispatchRecording(): void {
  dispatchRecordingBuffer = null;
}

/**
 * Replay a sequence of recorded dispatches directly onto the shared encoder.
 * Skips all JS-level dispatch logic (pipeline lookup, params, bind group creation).
 * Caller must ensure the shared encoder is active.
 */
export function replayDispatches(dispatches: RecordedDispatch[]): void {
  if (!sharedEncoderInstance) {
    throw new Error("replayDispatches requires an active shared encoder");
  }
  for (let i = 0; i < dispatches.length; i++) {
    const d = dispatches[i];
    const pass = sharedEncoderInstance.beginComputePass(undefined);
    pass.setPipeline(d.pipeline);
    pass.setBindGroup(0, d.bindGroup);
    pass.dispatchWorkgroups(d.workgroupsX, d.workgroupsY, d.workgroupsZ);
    pass.end();
    sharedEncoderPassCount++;
    autoFlushSharedEncoder();
  }
}

export function dispatchComputePass(
  pipeline: GPUComputePipeline,
  bindGroup: unknown,
  workgroupsX: number,
  workgroupsY: number = 1,
  workgroupsZ: number = 1,
): void {
  const ctx = requireContext();

  // Record dispatch if recording is active
  if (dispatchRecordingBuffer) {
    dispatchRecordingBuffer.push({
      pipeline,
      bindGroup: bindGroup as GPUBindGroup,
      workgroupsX,
      workgroupsY,
      workgroupsZ,
      buffers: getAndClearLastBindGroupBuffers(),
    });
  }

  if (sharedEncoderInstance) {
    // Encode directly onto the shared encoder — no new encoder or CB
    const tsWrites = getTimestampWrites(currentOpLabel ?? "unknown");
    const pass = sharedEncoderInstance.beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    sharedEncoderPassCount++;
    autoFlushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    const tsWrites = getTimestampWrites(currentOpLabel ?? "unknown");
    const pass = encoder.beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    submitOrCollect(encoder.finish());
  }

}

function sizeOf(shape: number[]): number {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

/**
 * Greatest common divisor using Euclidean algorithm.
 * Used for buffer alignment calculations.
 */
function gcd(a: number, b: number): number {
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

function lcm(a: number, b: number): number {
  return (a * b) / gcd(a, b);
}

function broadcastShapes(a: number[], b: number[]): number[] {
  const outRank = Math.max(a.length, b.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim);
  }
  return out;
}

function toIndexShape(shape: number[]): number[] {
  return shape.length === 0 ? [1] : shape;
}

function contiguousStrides(shape: number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i -= 1) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

function broadcastStrides(shape: number[], outShape: number[]): number[] {
  if (shape.length > outShape.length) {
    throw new Error("webgpu broadcast target has fewer dimensions than input");
  }
  const pad = outShape.length - shape.length;
  const inStrides = contiguousStrides(shape);
  const outStrides = new Array<number>(outShape.length);
  for (let axis = 0; axis < outShape.length; axis += 1) {
    const inAxis = axis - pad;
    if (inAxis < 0) {
      outStrides[axis] = 0;
      continue;
    }
    const inDim = shape[inAxis];
    const outDim = outShape[axis];
    if (inDim === outDim) {
      outStrides[axis] = inStrides[inAxis];
    } else if (inDim === 1) {
      outStrides[axis] = 0;
    } else {
      throw new Error("webgpu broadcast target shape is incompatible");
    }
  }
  return outStrides;
}

/**
 * Compute effective broadcast strides for a tensor with existing strides.
 * This handles non-contiguous tensors (e.g., transposed, expanded views).
 */
function computeEffectiveBroadcastStrides(
  tensor: WebGPUTensor,
  outShape: number[],
): number[] {
  const shape = tensor.shape;
  const strides = tensor.strides;

  if (shape.length > outShape.length) {
    throw new Error("webgpu broadcast target has fewer dimensions than input");
  }

  const pad = outShape.length - shape.length;
  const outStrides = new Array<number>(outShape.length);

  for (let axis = 0; axis < outShape.length; axis += 1) {
    const inAxis = axis - pad;
    if (inAxis < 0) {
      // Leading dimension not in input - broadcast with stride 0
      outStrides[axis] = 0;
      continue;
    }
    const inDim = shape[inAxis];
    const outDim = outShape[axis];
    if (inDim === outDim) {
      // Use the tensor's actual stride
      outStrides[axis] = strides[inAxis];
    } else if (inDim === 1) {
      // Broadcast: stride = 0
      outStrides[axis] = 0;
    } else {
      throw new Error("webgpu broadcast target shape is incompatible");
    }
  }
  return outStrides;
}

function wgslArray(values: number[]): string {
  return values.map((value) => `${value}u`).join(", ");
}

function buildBroadcastIndexing(
  indexShape: number[],
  inputStrides: number[][],
): {
  declarations: string;
  compute: string;
  offsets: string[];
} {
  const rank = indexShape.length;
  const outShapeDecl = `const OUT_SHAPE: array<u32, ${rank}> = array<u32, ${rank}>(${wgslArray(indexShape)});`;
  const strideDecls = inputStrides.map(
    (strides, index) =>
      `const IN${index}_STRIDES: array<u32, ${rank}> = array<u32, ${rank}>(${wgslArray(strides)});`,
  );
  const compute = `
  var remaining = idx;
  var coords: array<u32, ${rank}>;
  for (var axis = 0u; axis < ${rank}u; axis = axis + 1u) {
    let rev = ${rank}u - 1u - axis;
    let dim = OUT_SHAPE[rev];
    let coord = remaining % dim;
    coords[rev] = coord;
    remaining = remaining / dim;
  }
`;
  const offsets = inputStrides.map((_, index) => {
    const terms = indexShape.map(
      (_, axis) => `coords[${axis}u] * IN${index}_STRIDES[${axis}u]`,
    );
    return `  let offset${index} = ${terms.join(" + ")};`;
  });
  return {
    declarations: [outShapeDecl, ...strideDecls].join("\n"),
    compute,
    offsets,
  };
}

function toArrayUnsupported(): number[] {
  throw new Error("Use cpu() to read back WebGPU tensors");
}

/**
 * Create a WebGPU tensor with optional stride info.
 * If strides not provided, assumes contiguous layout.
 * @param ownsBuffer - If true, this tensor owns the buffer and will destroy it.
 *                     If false, this is a view that borrows the buffer.
 */
function createTensor(
  shape: number[],
  buffer: GPUBuffer,
  strides?: number[],
  offset = 0,
  dtype: DType = "f32",
  ownsBuffer = true,
): WebGPUTensor {
  const computedStrides = strides ?? contiguousStrides(shape);
  const isContiguousLayout =
    strides === undefined || checkContiguousStrides(shape, computedStrides);

  // Track if already destroyed to prevent double-destroy
  let destroyed = false;

  // Get buffer size and usage for pool release (WebGPU buffers have .size and .usage properties)
  const bufferSize = (buffer as any).size ?? sizeOf(shape) * dtypeBytes(dtype);
  const bufferUsage = (buffer as any).usage ?? 0;

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
    toArray: toArrayUnsupported,
    destroy(): void {
      if (destroyed) return;
      destroyed = true;
      // Only release/destroy if we own the buffer (not a view)
      if (ownsBuffer) {
        // Release our owning reference before pool/destroy
        bufferPool.decRef(buffer);

        // Arena buffers are owned by the arena, not the pool.
        // Don't release or destroy them — the arena will reuse them next step.
        if (arenaBufferSet.has(buffer)) {
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
 * Check if strides represent contiguous memory layout.
 */
function checkContiguousStrides(shape: number[], strides: number[]): boolean {
  const expected = contiguousStrides(shape);
  for (let i = 0; i < shape.length; i++) {
    // Size-1 dims don't affect contiguity
    if (shape[i] <= 1) continue;
    if (strides[i] !== expected[i]) return false;
  }
  return true;
}

function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  return a.every((dim, index) => dim === b[index]);
}

/**
 * Get bytes per element for a dtype.
 */
function dtypeBytes(dtype: DType): number {
  switch (dtype) {
    case "f16":
      return 2;
    case "f32":
      return 4;
    case "i32":
      return 4;
    case "u32":
      return 4;
    case "bool":
      return 1;
    default:
      return 4;
  }
}

/**
 * Align buffer size to 4 bytes (WebGPU requirement).
 */
function alignBufferSize(bytes: number): number {
  return Math.ceil(bytes / 4) * 4;
}

/**
 * Convert dtype to WGSL type string.
 */
function dtypeToWgsl(dtype: DType): string {
  switch (dtype) {
    case "f16":
      return "f16";
    case "f32":
      return "f32";
    case "i32":
      return "i32";
    case "u32":
      return "u32";
    case "bool":
      return "bool";
    default:
      return "f32";
  }
}

/**
 * Create a GPU buffer with memory tracking.
 * This ensures all buffer allocations respect the memory limit.
 * @throws GPUMemoryLimitExceededError if allocation would exceed the limit
 */
function createTrackedBuffer(
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
        return pooled;
      }
    }
    const pooled = bufferPool.acquire(alignedSize);
    if (pooled) {
      return pooled;
    }
  }

  // Check if this is a small UNIFORM-only buffer (params buffers)
  // These are temporary and don't need memory tracking to avoid memory leaks
  const isUniformOnly = (descriptor.usage & GPUBufferUsage.UNIFORM) !== 0 &&
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
    const limits = (device as unknown as { limits?: Record<string, number> }).limits;
    const maxBinding = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
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
    const physicalUsage = gpuMemoryTracker.getCurrentAllocatedBytes() + poolHeldBytes;
    if (physicalUsage + actualSize > gpuMemoryTracker.getMemoryLimit()) {
      // Evict enough pool buffers to make room. evictBuffers actually destroys
      // the GPU buffers (not just drops references), freeing physical GPU memory.
      bufferPool.evictBuffers(actualSize * 2);
    }
    gpuMemoryTracker.trackAllocation(null, actualSize);
  }

  try {
    const buffer = profileApiCall("createBuffer", () => device.createBuffer({
      ...descriptor,
      size: actualSize,
    }));
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
    return buffer;
  } catch (e) {
    // If buffer creation fails, undo the tracking
    if (!skipTracking) {
      gpuMemoryTracker.trackDeallocation(null);
    }
    throw e;
  }
}

/**
 * Allocate an output buffer ensuring no aliasing with input buffers.
 * The buffer pool may return an input buffer when sizes share a size class.
 * WebGPU forbids the same buffer as both read and read_write in one pass.
 */
// Sequence-indexed output buffer hints: record which GPUBuffer was used at each
// resolveOutputBuffer position. On the next step, try to acquire the same buffer
// from the pool for bind group cache stability.
let outputSeqIndex = 0;
const outputSequenceHints: Array<GPUBuffer | null> = [];

// Pre-pinned output buffers: extracted from pool at step start before any dispatches.
// Eliminates contention where multiple dispatch positions hint the same buffer.
const pinnedOutputBuffers: Array<GPUBuffer | null> = [];

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

/** The set of all buffers owned by any active arena (for release interception). */
const arenaBufferSet = new Set<GPUBuffer>();

/** Currently active arena (set during lowered plan execution). */
let activeArena: BufferArena | null = null;
let arenaResolveIndex = 0;
let arenaResolveHits = 0;
let arenaResolveAliased = 0;
let arenaResolveNoArena = 0;
let arenaAllocIndex = 0;

/**
 * Activate a buffer arena for the duration of a lowered plan execution.
 * All subsequent resolveOutputBuffer/allocateOutputBuffer calls will use
 * the arena instead of the pool, stabilizing buffer identities.
 */
export function setActiveArena(arena: BufferArena): void {
  activeArena = arena;
  arenaResolveIndex = 0;
  arenaAllocIndex = 0;
}

/**
 * Deactivate the current buffer arena.
 */
export function clearActiveArena(): void {
  activeArena = null;
  arenaResolveIndex = 0;
  arenaAllocIndex = 0;
}

/** Get the current arena resolve index (for dispatch replay recording). */
export function getArenaResolveIndex(): number { return arenaResolveIndex; }

/** Set the arena resolve index to a specific value (for dispatch replay restore). */
export function setArenaResolveIndexTo(idx: number): void { arenaResolveIndex = idx; }

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
function arenaAllocAt(arr: GPUBuffer[], idx: number, sizeBytes: number): GPUBuffer | null {
  const alignedSize = alignBufferSize(sizeBytes);
  const neededSizeClass = getSizeClass(alignedSize);

  // Check if we already have a buffer at this position from a previous step
  const existing = arr[idx];
  if (existing) {
    const existingSizeClass = getSizeClass(existing.size);
    if (existingSizeClass === neededSizeClass) {
      // Perfect match — reuse the same GPUBuffer object
      trackSharedEncoderWrite(existing);
      return existing;
    }
    // Size class changed — destroy old, allocate new below
    arenaBufferSet.delete(existing);
    if (!bufferPool.isLive(existing)) {
      // Replay-pinned buffers must survive
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
function prePinOutputBuffers(): void {
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

function resolveOutputBuffer(
  device: GPUDevice,
  sizeBytes: number,
  inputBuffers: GPUBuffer[],
  providedOutBuffer?: GPUBuffer,
): GPUBuffer {
  // Arena fast path: if a buffer arena is active, use it for output allocation.
  // Arena buffers persist across steps, giving 100% stable buffer identities
  // for bind group cache hits.
  if (!providedOutBuffer && activeArena) {
    const idx = arenaResolveIndex++;
    const arenaBuffer = arenaAllocAt(activeArena.resolve, idx, sizeBytes);
    if (arenaBuffer) {
      // Still need to check no input aliasing (WebGPU forbids same buffer as read+write)
      if (!inputBuffers.some(b => b === arenaBuffer)) {
        arenaResolveHits++;
        return arenaBuffer;
      }
      // Aliased with input — fall through to normal path for this position.
      // The arena buffer stays allocated for future steps; we just can't use it here.
      arenaResolveAliased++;
    }
  } else if (!providedOutBuffer) {
    arenaResolveNoArena++;
  }

  const alignedSize = alignBufferSize(sizeBytes);
  // Use pool-rounded size for replacement allocations so they match the size
  // class exactly. Without this, undersized buffers enter the pool and can
  // later be acquired by tensors needing the full size class, causing overflow.
  const pooledSize = getSizeForClass(getSizeClass(alignedSize));
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outIdx = outputSeqIndex++;

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

function createBufferWithData(
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
  const devLimits = (device as unknown as { limits?: Record<string, number> }).limits;
  const maxBindingSizeForPool = devLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const poolSize = rawPoolSize <= maxBindingSizeForPool ? rawPoolSize : alignedSize;

  // Try to acquire from pool first (only if pool size is power-of-2, i.e. not capped)
  const pooled = rawPoolSize <= maxBindingSizeForPool ? bufferPool.acquire(poolSize) : null;
  if (pooled && queue) {
    // Reusing a pooled buffer - write data via queue
    // NOTE: Don't call trackAllocation here! The buffer was already tracked
    // when it was first created, and we don't call trackDeallocation when
    // releasing to pool. Memory tracking stays consistent throughout pool lifecycle.
    profileApiCall("writeBuffer", () => queue.writeBuffer(pooled, 0, data));
    return pooled;
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

// General-purpose params buffer pool — pools uniform buffers by size class (4, 8, 16, 32, 48, 64).
// Replaces the old 4-byte-only uniformBufferPool with support for arbitrary param sizes.
const paramsBufferPools: Map<number, GPUBuffer[]> = new Map();
const MAX_PARAMS_POOL_SIZE_PER_CLASS = 256;

function paramsBufferSizeClass(byteLength: number): number {
  if (byteLength <= 4) return 4;
  if (byteLength <= 8) return 8;
  if (byteLength <= 16) return 16;
  if (byteLength <= 32) return 32;
  if (byteLength <= 48) return 48;
  return 64;
}

// Pre-allocated Uint32Array pool to avoid ~700 short-lived allocations per step.
// Each op dispatch creates a params array; reusing pre-allocated arrays reduces GC pressure.
const paramsArrayPool: Uint32Array[] = [];
for (let i = 0; i <= 8; i++) {
  paramsArrayPool.push(new Uint32Array(i));
}

/** Get a reusable Uint32Array of the given word count. Caller must set values before use. */
export function getParamsArray(wordCount: number): Uint32Array {
  if (wordCount <= 8) return paramsArrayPool[wordCount];
  return new Uint32Array(wordCount); // rare: allocate for large params
}

// Convenience helpers that fill and return pooled arrays (avoids new Uint32Array([...]) allocations)
function params1(a: number): Uint32Array { const p = paramsArrayPool[1]; p[0] = a; return p; }
function params2(a: number, b: number): Uint32Array { const p = paramsArrayPool[2]; p[0] = a; p[1] = b; return p; }
function params3(a: number, b: number, c: number): Uint32Array { const p = paramsArrayPool[3]; p[0] = a; p[1] = b; p[2] = c; return p; }
function params4(a: number, b: number, c: number, d: number): Uint32Array { const p = paramsArrayPool[4]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; return p; }
function params5(a: number, b: number, c: number, d: number, e: number): Uint32Array { const p = paramsArrayPool[5]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; p[4] = e; return p; }
function params6(a: number, b: number, c: number, d: number, e: number, f: number): Uint32Array { const p = paramsArrayPool[6]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; p[4] = e; p[5] = f; return p; }
function params7(a: number, b: number, c: number, d: number, e: number, f: number, g: number): Uint32Array { const p = paramsArrayPool[7]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; p[4] = e; p[5] = f; p[6] = g; return p; }

// Sequence-indexed params buffer cache: reuse the same GPUBuffer at each dispatch
// position across steps. This enables bind group cache hits since the params buffer
// pointer stays stable for a given dispatch position.
let paramsSeqIndex = 0;
const paramsSequenceBuffers: Array<{ buffer: GPUBuffer; sizeClass: number; data: Uint32Array } | null> = [];

export function createParamsBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const sizeClass = paramsBufferSizeClass(data.byteLength);
  const idx = paramsSeqIndex++;

  // Try to reuse the buffer from the same dispatch position (previous step).
  // This keeps the GPUBuffer pointer stable so bind group caching can hit.
  if (!activeBatch) {
    const cached = paramsSequenceBuffers[idx];
    if (cached !== undefined && cached !== null && cached.sizeClass === sizeClass) {
      // Fast path: skip writeBuffer if data is identical (params derived from
      // tensor shapes which are constant across steps).
      if (cached.data.length === data.length) {
        let same = true;
        for (let i = 0; i < data.length; i++) {
          if (cached.data[i] !== data[i]) { same = false; break; }
        }
        if (same) {
          return cached.buffer;  // Skip writeBuffer entirely
        }
      }
      // Data changed — write new data, update cached copy
      profileApiCall("writeBuffer", () => device.queue.writeBuffer(cached.buffer, 0, data));
      cached.data.set(data);
      return cached.buffer;
    }

    // Fallback: try pool
    const pool = paramsBufferPools.get(sizeClass);
    if (pool && pool.length > 0) {
      const buffer = pool.pop()!;
      profileApiCall("writeBuffer", () => device.queue.writeBuffer(buffer, 0, data));
      paramsSequenceBuffers[idx] = { buffer, sizeClass, data: data.slice() };
      paramsSequenceSet.add(buffer);
      return buffer;
    }
  }

  const buffer = profileApiCall("createBuffer", () => device.createBuffer({
    size: sizeClass,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  }));
  profileApiCall("writeBuffer", () => device.queue.writeBuffer(buffer, 0, data));
  if (!activeBatch) {
    paramsSequenceBuffers[idx] = { buffer, sizeClass, data: data.slice() };
    paramsSequenceSet.add(buffer);
  }
  return buffer;
}

// Track which buffers are pinned to sequence positions (not returnable to pool)
const paramsSequenceSet = new Set<GPUBuffer>();

export function releaseParamsBuffer(buffer: GPUBuffer): void {
  // Sequence-cached params buffers are reused across steps — don't return to pool
  if (paramsSequenceSet.has(buffer)) return;
  // Replay-pinned buffers must stay alive — referenced by recorded bind groups
  if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) return;

  if (activeBatch) {
    activeBatch.deferredDestroyBuffers.push(buffer);
    return;
  }
  if (sharedEncoder) {
    sharedEncoderDeferredUniformBuffers.push(buffer);
    autoFlushSharedEncoder();
    return;
  }
  const sizeClass = paramsBufferSizeClass(buffer.size);
  const pool = paramsBufferPools.get(sizeClass);
  if (pool) {
    if (pool.length < MAX_PARAMS_POOL_SIZE_PER_CLASS) {
      pool.push(buffer);
    } else {
      bufferPool.deferredDestroyUntracked(buffer);
    }
  } else {
    paramsBufferPools.set(sizeClass, [buffer]);
  }
}

// Backward-compatible wrappers for existing callsites
function createUniformBuffer(device: GPUDevice, size: number): GPUBuffer {
  return createParamsBuffer(device, params1(size));
}

function releaseUniformBuffer(buffer: GPUBuffer): void {
  releaseParamsBuffer(buffer);
}

// Profiled helpers for hot-path WebGPU API calls
function profiledWriteBuffer(queue: GPUQueue, buffer: GPUBuffer, offset: number, data: ArrayBufferView | ArrayBuffer): void {
  profileApiCall("writeBuffer", () => queue.writeBuffer(buffer, offset, data as ArrayBufferView));
}
function profiledCreateBindGroup(device: GPUDevice, descriptor: any): GPUBindGroup {
  const bg = profileApiCall("createBindGroup", () => device.createBindGroup(descriptor));
  // When recording, capture buffer references from the descriptor for replay pinning
  if (dispatchRecordingBuffer && descriptor.entries) {
    const bufs: GPUBuffer[] = [];
    for (const e of descriptor.entries) {
      const r = e.resource;
      if (r && typeof r === "object" && "buffer" in r) bufs.push(r.buffer);
    }
    lastBindGroupBuffers = bufs;
  }
  return bg;
}


// --- Sequence-Indexed Bind Group Cache ---
// Plans execute the same ops in the same order each step. Dispatch #i in step N
// corresponds to dispatch #i in step N+1. Rather than building string keys from
// unstable buffer IDs, index the cache by dispatch sequence position and validate
// by GPUBuffer/pipeline pointer equality.
let dispatchIndex = 0;
const sequenceEntries: Array<{
  bindGroup: GPUBindGroup;
  pipeline: GPUComputePipeline;
  buffers: GPUBuffer[];
} | null> = [];
let seqCacheHits = 0;
let seqCacheMisses = 0;
let seqCacheMissLog: Array<{ idx: number; reason: string; label: string | null; details: string }> = [];

/**
 * Create or retrieve a cached bind group for simple (no offset/size) buffer bindings.
 * Entries are built internally: binding i → { buffer: buffers[i] }.
 *
 * Uses sequence-indexed caching: each dispatch position in a step maps to the same
 * position in the next step. Validation is by pointer equality on pipeline + buffers.
 */
export function cachedCreateBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: GPUBuffer[],
): GPUBindGroup {
  const idx = dispatchIndex++;
  const entry = sequenceEntries[idx];

  if (entry !== undefined && entry !== null
      && entry.pipeline === pipeline
      && entry.buffers.length === buffers.length) {
    let match = true;
    for (let i = 0; i < buffers.length; i++) {
      if (entry.buffers[i] !== buffers[i]) { match = false; break; }
    }
    if (match) {
      seqCacheHits++;
      if (dispatchRecordingBuffer) lastBindGroupBuffers = entry.buffers;
      return entry.bindGroup;
    }
  }

  seqCacheMisses++;
  if (seqCacheMissLog.length < 200) {
    let reason = "new";
    let details = "";
    if (entry !== undefined && entry !== null) {
      if (entry.pipeline !== pipeline) reason = "pipeline";
      else if (entry.buffers.length !== buffers.length) reason = "buf-count";
      else {
        const changed: number[] = [];
        for (let i = 0; i < buffers.length; i++) {
          if (entry.buffers[i] !== buffers[i]) changed.push(i);
        }
        reason = `buf[${changed.join(",")}]`;
        // Log sizes and arena status for changed buffers
        const parts: string[] = [];
        for (const ci of changed) {
          const oldB = entry.buffers[ci];
          const newB = buffers[ci];
          const oldArena = arenaBufferSet.has(oldB) ? "A" : "P";
          const newArena = arenaBufferSet.has(newB) ? "A" : "P";
          parts.push(`${ci}:${oldB.size}${oldArena}->${newB.size}${newArena}`);
        }
        details = parts.join(" ");
      }
    }
    seqCacheMissLog.push({ idx, reason, label: currentOpLabel, details });
  }
  const entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }> = [];
  for (let i = 0; i < buffers.length; i++) {
    entries.push({ binding: i, resource: { buffer: buffers[i] } });
  }
  const bindGroup = profileApiCall("createBindGroup", () =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    })
  );

  const bufCopy = buffers.slice();
  sequenceEntries[idx] = { bindGroup, pipeline, buffers: bufCopy };
  if (dispatchRecordingBuffer) lastBindGroupBuffers = bufCopy;
  return bindGroup;
}

/** Reset the dispatch sequence counter to 0. Call at the start of each step. */
export function resetDispatchSequence(): void {
  dispatchIndex = 0;
  paramsSeqIndex = 0;
  outputSeqIndex = 0;
}

/** Set dispatch sequence counters to specific positions (for replay cache). */
export function setDispatchSequenceCounters(dispatch: number, params: number, output: number): void {
  dispatchIndex = dispatch;
  paramsSeqIndex = params;
  outputSeqIndex = output;
}

/** Get current dispatch sequence counters (for recording). */
export function getDispatchSequenceCounters(): { dispatch: number; params: number; output: number } {
  return { dispatch: dispatchIndex, params: paramsSeqIndex, output: outputSeqIndex };
}

export function clearBindGroupCache(): void {
  sequenceEntries.length = 0;
  dispatchIndex = 0;
  seqCacheHits = 0;
  seqCacheMisses = 0;
  paramsSequenceBuffers.length = 0;
  paramsSequenceSet.clear();
  paramsSeqIndex = 0;
  outputSequenceHints.length = 0;
  pinnedOutputBuffers.length = 0;
  outputSeqIndex = 0;
  // Clear arena state (arenas themselves are cleaned up by plan cache eviction)
  activeArena = null;
  arenaResolveIndex = 0;
  arenaAllocIndex = 0;
}

export function getBindGroupCacheStats(): { hits: number; misses: number; size: number; hitRate: number } {
  const total = seqCacheHits + seqCacheMisses;
  return { hits: seqCacheHits, misses: seqCacheMisses, size: sequenceEntries.length, hitRate: total > 0 ? seqCacheHits / total : 0 };
}

export function resetBindGroupCacheStats(): void {
  seqCacheHits = 0;
  seqCacheMisses = 0;
  seqCacheMissLog = [];
  arenaResolveHits = 0;
  arenaResolveAliased = 0;
  arenaResolveNoArena = 0;
}

export function getArenaResolveStats(): { hits: number; aliased: number; noArena: number } {
  return { hits: arenaResolveHits, aliased: arenaResolveAliased, noArena: arenaResolveNoArena };
}

export function getBindGroupCacheMissLog(): Array<{ idx: number; reason: string; label: string | null; details: string }> {
  return seqCacheMissLog;
}

/**
 * Get the buffer list for a specific dispatch sequence index.
 * Returns null if the index has no cached entry.
 * Used by dispatch replay to collect buffers that need pinning.
 */
export function getSequenceEntryBuffers(idx: number): GPUBuffer[] | null {
  const entry = sequenceEntries[idx];
  return entry ? entry.buffers : null;
}

// --- Replay Buffer Pinning ---
// When dispatch replay caches exist, buffers referenced by recorded bind groups
// must not be destroyed between steps. This set accumulates all such buffers
// across all plans (forward, backward, optimizer).
// Checked in deferredDestroy/deferredDestroyUntracked/deferredDestroyBuffer.
let replayPinnedBufferSet: Set<GPUBuffer> | null = null;


/** Add buffers to the replay pinned set. Called when a replay cache is built. */
export function addReplayPinnedBuffers(pins: Set<GPUBuffer>): void {
  if (!replayPinnedBufferSet) {
    replayPinnedBufferSet = new Set(pins);
  } else {
    for (const b of pins) replayPinnedBufferSet.add(b);
  }
}

/**
 * Unified dispatch helper for elementwise compute shaders.
 * Collapses the common 6-step boilerplate (pipeline, output, params, bindgroup, dispatch, release)
 * into a single call.
 */
function dispatchElementwise(desc: {
  key: string;
  shader: string;
  inputs: GPUBuffer[];
  outputSizeBytes: number;
  params: Uint32Array;
  outBuffer?: GPUBuffer;
  dispatchX: number;
  dispatchY?: number;
}): GPUBuffer {
  const ctx = requireContext();

  const pipeline = getPipeline(ctx, desc.key, desc.shader);

  const outBuffer = desc.outBuffer
    ?? resolveOutputBuffer(ctx.device, desc.outputSizeBytes, desc.inputs);

  const paramsBuffer = createParamsBuffer(ctx.device, desc.params);

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < desc.inputs.length; i++) {
    bgBuffers.push(desc.inputs[i]);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(paramsBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

  dispatchComputePass(pipeline, bindGroup, desc.dispatchX, desc.dispatchY ?? 1);

  releaseParamsBuffer(paramsBuffer);

  return outBuffer;
}

function binaryBroadcastShader(
  op: string,
  indexShape: number[],
  aStrides: number[],
  bStrides: number[],
  aOffset: number,
  bOffset: number,
  dtype: DType = "f32",
  gridSizeX?: number,
): string {
  const indexing = buildBroadcastIndexing(indexShape, [aStrides, bStrides]);
  const wgslType = dtypeToWgsl(dtype);
  const enableF16 = dtype === "f16" ? "enable f16;\n" : "";
  // Support 2D dispatch for large tensors
  const use2D = gridSizeX !== undefined && gridSizeX > 0;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read> b: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(3) var<uniform> params: Params;

${indexing.declarations}
const A_OFFSET: u32 = ${aOffset}u;
const B_OFFSET: u32 = ${bOffset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
  out[idx] = a[offset0 + A_OFFSET] ${op} b[offset1 + B_OFFSET];
}
`;
}

/**
 * Generate unary shader with stride support for non-contiguous tensors.
 */
function unaryStridedShader(
  expr: string,
  shape: number[],
  strides: number[],
  offset: number,
  dtype: DType = "f32",
  gridSizeX?: number,
): string {
  const rank = shape.length;
  const wgslType = dtypeToWgsl(dtype);
  const enableF16 = dtype === "f16" ? "enable f16;\n" : "";
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  if (rank === 0) {
    // Scalar case
    return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  let x = a[${offset}u];
  out[idx] = ${expr};
}
`;
  }

  const shapeArray = `array<u32, ${rank}>(${shape.map((s) => `${s}u`).join(", ")})`;
  const stridesArray = `array<u32, ${rank}>(${strides.map((s) => `${s}u`).join(", ")})`;

  return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const SHAPE = ${shapeArray};
const STRIDES = ${stridesArray};
const OFFSET: u32 = ${offset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert flat index to strided offset
  var remaining = idx;
  var inputOffset = OFFSET;
  for (var d = 0u; d < RANK; d = d + 1u) {
    var dimSize = 1u;
    for (var j = d + 1u; j < RANK; j = j + 1u) {
      dimSize = dimSize * SHAPE[j];
    }
    let coord = remaining / dimSize;
    remaining = remaining % dimSize;
    inputOffset = inputOffset + coord * STRIDES[d];
  }

  let x = a[inputOffset];
  out[idx] = ${expr};
}
`;
}

const FUSED_UNARY_OPS = new Map<string, (value: string) => string>([
  ["neg", (value) => `-(${value})`],
  ["abs", (value) => `abs(${value})`],
  ["exp", (value) => `exp(${value})`],
  ["log", (value) => `log(${value})`],
  ["relu", (value) => `select(0.0, ${value}, ${value} > 0.0)`],
  ["sqrt", (value) => `sqrt(${value})`],
]);

const FUSED_BINARY_OPS = new Map<
  string,
  (left: string, right: string) => string
>([
  ["add", (left, right) => `(${left} + ${right})`],
  ["sub", (left, right) => `(${left} - ${right})`],
  ["mul", (left, right) => `(${left} * ${right})`],
  ["div", (left, right) => `(${left} / ${right})`],
]);

function buildFusedExpression(op: string, inputs: string[]): string {
  const binary = FUSED_BINARY_OPS.get(op);
  if (binary) {
    if (inputs.length !== 2) {
      throw new Error(`fused op ${op} expects 2 inputs`);
    }
    return binary(inputs[0], inputs[1]);
  }
  const unary = FUSED_UNARY_OPS.get(op);
  if (unary) {
    if (inputs.length !== 1) {
      throw new Error(`fused op ${op} expects 1 input`);
    }
    return unary(inputs[0]);
  }
  throw new Error(`fused op ${op} is not supported`);
}

function requireFusionNode(
  nodeById: Map<number, IRNode>,
  nodeId: number,
): IRNode {
  const node = nodeById.get(nodeId);
  if (!node) {
    throw new Error(`fusion recipe missing node ${nodeId}`);
  }
  return node;
}

function requireShape(node: IRNode): number[] {
  if (!node.shape) {
    throw new Error(`fusion recipe missing shape for node ${node.id}`);
  }
  return node.shape;
}

function buildFusedElementwiseShader(
  graph: IRGraph,
  recipe: FusionRecipe,
  use2D?: boolean,
  shaderGridSizeX?: number,
): string {
  if (recipe.outputs.length !== 1) {
    throw new Error("fused elementwise expects a single output");
  }
  const outputDescriptor = recipe.outputDescriptors[0];
  if (!outputDescriptor) {
    throw new Error("fusion recipe has no output descriptors");
  }
  const outShape = outputDescriptor.shape.slice();
  const indexShape = toIndexShape(outShape);
  const nodeById = new Map<number, IRNode>();
  for (const node of graph.nodes) {
    nodeById.set(node.id, node);
  }

  const inputDecls: string[] = [];
  const inputStrides: number[][] = [];
  const valueById = new Map<number, string>();
  for (let i = 0; i < recipe.inputs.length; i += 1) {
    const name = `in${i}`;
    const inputNode = requireFusionNode(nodeById, recipe.inputs[i]);
    const inputShape = requireShape(inputNode);
    inputStrides.push(broadcastStrides(inputShape, indexShape));
    inputDecls.push(
      `@group(0) @binding(${i}) var<storage, read> ${name}: array<f32>;`,
    );
  }

  const indexing = buildBroadcastIndexing(indexShape, inputStrides);
  for (let i = 0; i < recipe.inputs.length; i += 1) {
    const name = `in${i}`;
    valueById.set(recipe.inputs[i], `${name}[offset${i}]`);
  }

  const statements: string[] = [];
  let varIndex = 0;
  for (const nodeId of recipe.nodeIds) {
    const node = requireFusionNode(nodeById, nodeId);
    const inputExprs = node.inputs.map((inputId) => {
      const expr = valueById.get(inputId);
      if (!expr) {
        throw new Error(`fusion recipe missing input ${inputId}`);
      }
      return expr;
    });
    const expr = buildFusedExpression(node.op, inputExprs);
    const varName = `v${varIndex}`;
    varIndex += 1;
    statements.push(`  let ${varName} = ${expr};`);
    valueById.set(nodeId, varName);
  }

  const outputExpr = valueById.get(recipe.outputs[0]);
  if (!outputExpr) {
    throw new Error(`fusion recipe missing output ${recipe.outputs[0]}`);
  }

  const outputBinding = recipe.inputs.length;
  const paramsBinding = recipe.inputs.length + 1;
  const body = statements.join("\n");

  return `
struct Params {
  size: u32,
};

${inputDecls.join("\n")}
@group(0) @binding(${outputBinding}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${paramsBinding}) var<uniform> params: Params;

${indexing.declarations}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${use2D ? `gid.x + gid.y * ${shaderGridSizeX}u` : "gid.x"};
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
${body}
  out[idx] = ${outputExpr};
}
`;
}

function getPipeline(
  context: WebGPUContext,
  key: string,
  code: string,
): GPUComputePipeline {
  const cached = context.pipelines.get(key);
  if (cached) {
    return cached;
  }
  const module = context.device.createShaderModule({ code });
  const pipeline = context.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  context.pipelines.set(key, pipeline);
  return pipeline;
}

/**
 * Dispatch binary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
function dispatchBinary(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);
  if (outSize === 0) {
    throw new Error("webgpu ops do not support empty tensors yet");
  }
  const ctx = requireContext();

  // Determine output dtype - both inputs must have same dtype
  const dtype = a.dtype;
  if (b.dtype !== dtype) {
    throw new Error(
      `webgpu binary op: mismatched dtypes ${a.dtype} and ${b.dtype}`,
    );
  }

  // Check if any buffer exceeds max binding size
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const bytesPerElement = dtypeBytes(dtype);

  const aSizeBytes = a.size * bytesPerElement;
  const bSizeBytes = b.size * bytesPerElement;
  const outSizeBytes = outSize * bytesPerElement;

  // Check if chunking is needed
  if (aSizeBytes > maxBindingSize || bSizeBytes > maxBindingSize || outSizeBytes > maxBindingSize) {
    // Check for simple case: same shape, both contiguous, or one is scalar
    const aIsScalar = a.size === 1;
    const bIsScalar = b.size === 1;
    const sameShape = a.shape.length === b.shape.length &&
      a.shape.every((d, i) => d === b.shape[i]);

    if ((sameShape && a.isContiguous && b.isContiguous) ||
        (aIsScalar && b.isContiguous) ||
        (bIsScalar && a.isContiguous)) {
      return dispatchBinaryChunked(op, a, b, options);
    }

    // Handle case where one or both tensors are non-contiguous
    // Materialize them to contiguous first, then use chunked path
    if (sameShape) {
      const aContiguous = a.isContiguous ? a : ensureContiguous(a);
      const bContiguous = b.isContiguous ? b : ensureContiguous(b);
      const result = dispatchBinaryChunked(op, aContiguous, bContiguous, options);
      // Destroy contiguous copies to prevent memory leaks
      if (aContiguous !== a) aContiguous.destroy?.();
      if (bContiguous !== b) bContiguous.destroy?.();
      return result;
    }

    // For broadcast cases with non-contiguous tensors, materialize the large non-contiguous one
    if (!a.isContiguous && aSizeBytes > maxBindingSize) {
      const aContiguous = ensureContiguous(a);
      const result = dispatchBinary(op, aContiguous, b, options);
      aContiguous.destroy?.();
      return result;
    }
    if (!b.isContiguous && bSizeBytes > maxBindingSize) {
      const bContiguous = ensureContiguous(b);
      const result = dispatchBinary(op, a, bContiguous, options);
      bContiguous.destroy?.();
      return result;
    }

    // Fall through to direct dispatch - may fail for complex cases
  }

  return dispatchBinaryDirect(op, a, b, options);
}

/**
 * Direct binary dispatch for small tensors (no chunking).
 */
function dispatchBinaryDirect(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);
  const dtype = a.dtype;

  const aStrides = computeEffectiveBroadcastStrides(a, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(b, indexShape);

  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = binaryBroadcastShader(
    op, indexShape, aStrides, bStrides, a.offset, b.offset, dtype,
    use2D ? dispatch.gridSizeX : undefined,
  );
  const key = `binary:${op}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${a.offset}:${b.offset}:${dtype}:${use2D ? dispatch.gridSizeX : "1d"}`;
  const bytesPerElement = dtypeBytes(dtype);

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [a.buffer, b.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params1(outSize),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Chunked binary dispatch for large tensors.
 * Handles: same-shape contiguous tensors, or scalar + large tensor.
 */
function dispatchBinaryChunked(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Determine output shape and size
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize = sizeOf(outShape);
  const outSizeBytes = outSize * bytesPerElement;

  // Check if a or b is scalar
  const aIsScalar = a.size === 1;
  const bIsScalar = b.size === 1;

  // Calculate chunk size
  const elementsPerAlignment = minAlignment / bytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const numChunks = Math.ceil(outSize / elementsPerChunk);

  // Create output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSizeBytes,
    [a.buffer, b.buffer],
    options?.outBuffer,
  );

  // Determine 2D dispatch dimensions
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Build shader for chunked binary op
  const wgslType = dtype === "f16" ? "f16" : dtype === "i32" ? "i32" : dtype === "u32" ? "u32" : "f32";
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Shader handles scalar broadcast
  const aAccess = aIsScalar ? "a[0]" : "a[idx]";
  const bAccess = bIsScalar ? "b[0]" : "b[idx]";

  const code = `
@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read> b: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> out: array<${wgslType}>;

struct Params {
  chunkSize: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  out[idx] = ${aAccess} ${op} ${bAccess};
}
`;

  const key = `binaryChunked:${op}:${dtype}:${aIsScalar}:${bIsScalar}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  // Process each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, outSize);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    // Each chunk needs its own uniform buffer to avoid data corruption
    // when batching (queue.writeBuffer overwrites in-place, but all command
    // buffers reference the same GPU buffer).
    const paramsBuffer = createUniformBuffer(ctx.device, chunkSize);

    // Bind chunk regions (or full buffer for scalars)
    const aBinding = aIsScalar
      ? { buffer: a.buffer }
      : { buffer: a.buffer, offset: chunkByteOffset, size: chunkByteSize };
    const bBinding = bIsScalar
      ? { buffer: b.buffer }
      : { buffer: b.buffer, offset: chunkByteOffset, size: chunkByteSize };
    const outBinding = { buffer: outBuffer, offset: chunkByteOffset, size: chunkByteSize };

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: aBinding },
        { binding: 1, resource: bBinding },
        { binding: 2, resource: outBinding },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);

    // Return uniform buffer to pool (deferred if batching)
    releaseUniformBuffer(paramsBuffer);
  }

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Dispatch unary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
function dispatchUnary(
  opKey: string,
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  // Check if chunking is needed for large contiguous tensors
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const aSizeBytes = a.size * bytesPerElement;

  if (aSizeBytes > maxBindingSize && a.isContiguous) {
    return dispatchUnaryChunked(opKey, expr, a, options);
  }

  return dispatchUnaryDirect(opKey, expr, a, options);
}

/**
 * Direct unary dispatch for small tensors.
 */
function dispatchUnaryDirect(
  opKey: string,
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const dtype = a.dtype;

  const totalWorkgroups = Math.ceil(a.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = unaryStridedShader(
    expr, a.shape, a.strides, a.offset, dtype,
    use2D ? dispatch.gridSizeX : undefined,
  );
  const key = `unary:${opKey}:${a.shape.join("x")}:${a.strides.join(",")}:${a.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const bytesPerElement = dtypeBytes(dtype);

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [a.buffer],
    outputSizeBytes: a.size * bytesPerElement,
    params: params1(a.size),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Chunked unary dispatch for large contiguous tensors.
 */
function dispatchUnaryChunked(
  opKey: string,
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  const outSize = a.size;
  const outSizeBytes = outSize * bytesPerElement;

  // Calculate chunk size
  const elementsPerAlignment = minAlignment / bytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const numChunks = Math.ceil(outSize / elementsPerChunk);

  // Create output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSizeBytes,
    [a.buffer],
    options?.outBuffer,
  );

  // Determine 2D dispatch dimensions
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Build shader for chunked unary op
  const wgslType = dtype === "f16" ? "f16" : dtype === "i32" ? "i32" : dtype === "u32" ? "u32" : "f32";
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Replace 'x' in expr with array access
  const exprWithAccess = expr.replace(/\bx\b/g, "a[idx]");

  const code = `
@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;

struct Params {
  chunkSize: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  out[idx] = ${exprWithAccess};
}
`;

  const key = `unaryChunked:${opKey}:${dtype}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  // Process each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, outSize);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    // Each chunk needs its own uniform buffer to avoid data corruption
    // when batching (queue.writeBuffer overwrites in-place).
    const paramsBuffer = createUniformBuffer(ctx.device, chunkSize);

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: a.buffer, offset: chunkByteOffset, size: chunkByteSize } },
        { binding: 1, resource: { buffer: outBuffer, offset: chunkByteOffset, size: chunkByteSize } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);

    // Return uniform buffer to pool (deferred if batching)
    releaseUniformBuffer(paramsBuffer);
  }

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

function dispatchMatmul(
  a: WebGPUTensor,
  b: WebGPUTensor,
  transA = false,
  transB = false,
  donatedBuffer?: GPUBuffer,
): WebGPUTensor {
  const ctx = requireContext();

  // Try to detect simple last-2-dim transposes to avoid contiguous() materialization.
  // If detected, we use the original contiguous buffer and flip the transpose flag.
  let effectiveA: WebGPUTensor = a;
  let effectiveTransA = transA;
  let aWasCopied = false;

  const aOrigShape = !transA ? detectSimpleTranspose(a) : null;
  if (aOrigShape) {
    // Use original buffer with swapped shape and flipped transpose flag
    effectiveA = createTensor(aOrigShape, a.buffer, undefined, 0, a.dtype, false);
    effectiveTransA = true;
  } else {
    effectiveA = ensureContiguous(a);
    aWasCopied = effectiveA !== a;
  }

  let effectiveB: WebGPUTensor = b;
  let effectiveTransB = transB;
  let bWasCopied = false;

  const bOrigShape = !transB ? detectSimpleTranspose(b) : null;
  if (bOrigShape) {
    effectiveB = createTensor(bOrigShape, b.buffer, undefined, 0, b.dtype, false);
    effectiveTransB = true;
  } else {
    effectiveB = ensureContiguous(b);
    bWasCopied = effectiveB !== b;
  }

  // Compute output shape with transpose and batch broadcasting
  const outShape = computeMatmulOutputShape(
    effectiveA.shape,
    effectiveB.shape,
    effectiveTransA,
    effectiveTransB,
  );

  // Extract matrix dimensions
  const aRank = effectiveA.shape.length;
  const bRank = effectiveB.shape.length;

  let m: number, k: number, n: number;
  if (effectiveTransA) {
    k = effectiveA.shape[aRank - 2];
    m = effectiveA.shape[aRank - 1];
  } else {
    m = effectiveA.shape[aRank - 2];
    k = effectiveA.shape[aRank - 1];
  }
  if (effectiveTransB) {
    n = effectiveB.shape[bRank - 2];
  } else {
    n = effectiveB.shape[bRank - 1];
  }

  // Compute batch size and strides
  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    effectiveA.shape,
    effectiveB.shape,
    batchDims,
    m,
    n,
    k,
  );

  // Derive per-input dtypes; output is always the higher-precision type
  const dtypeA = effectiveA.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const dtypeB = effectiveB.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const outputDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
  const bytesPerElement = outputDtype === "f16" ? 2 : 4;

  // Create or use donated output buffer
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);
  const requiredSize = outSize * bytesPerElement;
  const useDonated =
    donatedBuffer && (donatedBuffer as any).size >= requiredSize;
  const outBuffer = useDonated
    ? donatedBuffer
    : resolveOutputBuffer(ctx.device, requiredSize,
        [effectiveA.buffer, effectiveB.buffer]);

  // Dispatch tiled matmul
  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: effectiveA.buffer,
    b: effectiveB.buffer,
    out: outBuffer,
    m,
    n,
    k,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    transA: effectiveTransA,
    transB: effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
  });

  // Destroy contiguous copies if they were created (deferred for GPU fence)
  if (aWasCopied) {
    bufferPool.decRef(effectiveA.buffer);
    bufferPool.deferredDestroy(effectiveA.buffer, effectiveA.size * (a.dtype === "f16" ? 2 : 4));
  }
  if (bWasCopied) {
    bufferPool.decRef(effectiveB.buffer);
    bufferPool.deferredDestroy(effectiveB.buffer, effectiveB.size * (b.dtype === "f16" ? 2 : 4));
  }

  // Output tensor always owns the buffer (donated or new)
  return createTensor(outShape, outBuffer, undefined, 0, outputDtype, true);
}

/**
 * Dispatch matmul with fused epilogue operations.
 *
 * This function runs matmul with additional elementwise operations
 * (like bias, relu, gelu) fused into the output write loop for better performance.
 */
export function dispatchMatmulWithEpilogue(
  a: WebGPUTensor,
  b: WebGPUTensor,
  epilogue: EpilogueConfig,
  epilogueInputs: WebGPUTensor[],
  transA = false,
  transB = false,
  inputCastA?: DType,
  inputCastB?: DType,
): WebGPUTensor {
  const ctx = requireContext();

  // Try to detect simple last-2-dim transposes to avoid contiguous() materialization.
  let effectiveA: WebGPUTensor = a;
  let effectiveTransA = transA;
  let aWasCopied = false;

  const aOrigShape = !transA ? detectSimpleTranspose(a) : null;
  if (aOrigShape) {
    effectiveA = createTensor(aOrigShape, a.buffer, undefined, 0, a.dtype, false);
    effectiveTransA = true;
  } else {
    effectiveA = ensureContiguous(a);
    aWasCopied = effectiveA !== a;
  }

  let effectiveB: WebGPUTensor = b;
  let effectiveTransB = transB;
  let bWasCopied = false;

  const bOrigShape = !transB ? detectSimpleTranspose(b) : null;
  if (bOrigShape) {
    effectiveB = createTensor(bOrigShape, b.buffer, undefined, 0, b.dtype, false);
    effectiveTransB = true;
  } else {
    effectiveB = ensureContiguous(b);
    bWasCopied = effectiveB !== b;
  }

  // Compute output shape with transpose and batch broadcasting
  const outShape = computeMatmulOutputShape(
    effectiveA.shape,
    effectiveB.shape,
    effectiveTransA,
    effectiveTransB,
  );

  // Extract matrix dimensions
  const aRank = effectiveA.shape.length;
  const bRank = effectiveB.shape.length;

  let m: number, k: number, n: number;
  if (effectiveTransA) {
    k = effectiveA.shape[aRank - 2];
    m = effectiveA.shape[aRank - 1];
  } else {
    m = effectiveA.shape[aRank - 2];
    k = effectiveA.shape[aRank - 1];
  }
  if (effectiveTransB) {
    n = effectiveB.shape[bRank - 2];
  } else {
    n = effectiveB.shape[bRank - 1];
  }

  // Compute batch size and strides
  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    effectiveA.shape,
    effectiveB.shape,
    batchDims,
    m,
    n,
    k,
  );

  // Derive per-input dtypes; output dtype from epilogue or promoted type.
  // When inputCastA/B is set, the actual buffer dtype is wider (e.g. f32) but
  // the matmul computes in the target dtype (e.g. f16) by casting during tile load.
  // The "compute dtype" for codegen is the post-cast dtype.
  const rawDtypeA = effectiveA.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const rawDtypeB = effectiveB.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const dtypeA: "f16" | "f32" = inputCastA === "f16" && isF16Supported() ? "f16" : rawDtypeA;
  const dtypeB: "f16" | "f32" = inputCastB === "f16" && isF16Supported() ? "f16" : rawDtypeB;
  const promotedDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
  const outputDtype = epilogue.outputDtype ?? promotedDtype;
  const bytesPerElement = outputDtype === "f16" ? 2 : 4;

  // Extract GPU buffers from epilogue input tensors
  const epilogueBuffers = epilogueInputs.map((t) => t.buffer);

  // Create output buffer (routed through arena for stable bind group cache)
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bytesPerElement,
    [effectiveA.buffer, effectiveB.buffer, ...epilogueBuffers]);

  // Dispatch tiled matmul with epilogue
  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: effectiveA.buffer,
    b: effectiveB.buffer,
    out: outBuffer,
    m,
    n,
    k,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    transA: effectiveTransA,
    transB: effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
    epilogue,
    epilogueInputs: epilogueBuffers,
    inputCastA,
    inputCastB,
  });

  // Destroy contiguous copies if they were created (deferred for GPU fence)
  if (aWasCopied) {
    bufferPool.decRef(effectiveA.buffer);
    bufferPool.deferredDestroy(effectiveA.buffer, effectiveA.size * (a.dtype === "f16" ? 2 : 4));
  }
  if (bWasCopied) {
    bufferPool.decRef(effectiveB.buffer);
    bufferPool.deferredDestroy(effectiveB.buffer, effectiveB.size * (b.dtype === "f16" ? 2 : 4));
  }

  return createTensor(outShape, outBuffer, undefined, 0, outputDtype);
}

/**
 * Direct matmul dispatch with pre-computed geometry.
 *
 * Used by the lowered plan fast path to skip shape computation,
 * transpose detection, contiguous checks, and dtype resolution.
 * All structural decisions were computed on the first execution and cached.
 */
export function dispatchMatmulDirect(
  bufA: GPUBuffer,
  bufB: GPUBuffer,
  config: {
    m: number; n: number; k: number;
    transA: boolean; transB: boolean;
    batchSize: number;
    batchStrideA: number; batchStrideB: number; batchStrideC: number;
    outShape: number[];
    dtypeA: "f16" | "f32";
    dtypeB?: "f16" | "f32";
    outputDtype: DType;
    epilogueConfig: any;
    epilogueBuffers: GPUBuffer[];
    inputCastA?: DType;
    inputCastB?: DType;
  },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = config.outputDtype === "f16" ? 2 : 4;
  const outSize = config.outShape.reduce((a, b) => a * b, 1);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [bufA, bufB, ...config.epilogueBuffers],
  );

  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: bufA,
    b: bufB,
    out: outBuffer,
    m: config.m,
    n: config.n,
    k: config.k,
    batchSize: config.batchSize,
    batchStrideA: config.batchStrideA,
    batchStrideB: config.batchStrideB,
    batchStrideC: config.batchStrideC,
    transA: config.transA,
    transB: config.transB,
    dtype: config.dtypeA,
    dtypeB: config.dtypeB,
    epilogue: config.epilogueConfig,
    epilogueInputs: config.epilogueBuffers,
    inputCastA: config.inputCastA,
    inputCastB: config.inputCastB,
  });

  return createTensor(config.outShape, outBuffer, undefined, 0, config.outputDtype);
}

export function runFusedElementwise(
  graph: IRGraph,
  recipe: FusionRecipe,
  inputs: BackendTensor[],
): BackendTensor | null {
  // Get output dtype from first output descriptor (single-output case)
  const outputDescriptor = recipe.outputDescriptors[0];
  if (!outputDescriptor) {
    throw new Error("fusion recipe has no output descriptors");
  }
  if (outputDescriptor.dtype !== "f32") {
    // Non-f32 fusion not yet supported; return null to fall back to sequential execution
    return null;
  }
  if (inputs.length !== recipe.inputs.length) {
    throw new Error(
      `fusion recipe expects ${recipe.inputs.length} inputs, got ${inputs.length}`,
    );
  }

  const ctx = requireContext();
  const outShape = outputDescriptor.shape.slice();
  const size = sizeOf(outShape);
  if (size === 0) {
    throw new Error("fused elementwise does not support empty tensors yet");
  }
  const inputTensors = inputs as WebGPUTensor[];
  const nodeById = new Map<number, IRNode>();
  for (const node of graph.nodes) {
    nodeById.set(node.id, node);
  }
  recipe.inputs.forEach((inputId, index) => {
    const node = requireFusionNode(nodeById, inputId);
    const expectedShape = requireShape(node);
    if (!shapesEqual(inputTensors[index].shape, expectedShape)) {
      throw new Error(
        `fused elementwise input shape mismatch for node ${inputId}`,
      );
    }
  });

  const totalWorkgroups = Math.ceil(size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const shaderGridSizeX = dispatch.x * WORKGROUP_SIZE;
  const code = buildFusedElementwiseShader(graph, recipe, use2D, shaderGridSizeX);
  const pipeline = getPipeline(ctx, `fused:${code}`, code);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const params = createUniformBuffer(ctx.device, size);
  const bgBuffers: GPUBuffer[] = inputTensors.map(t => t.buffer);
  bgBuffers.push(outBuffer);
  bgBuffers.push(params);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);
  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(params);
  return createTensor(outShape, outBuffer);
}

function tensorFromArray(values: number[], shape: number[]): WebGPUTensor {
  const ctx = requireContext();
  const expected = sizeOf(shape);
  if (expected !== values.length) {
    throw new Error("Tensor data length does not match shape");
  }
  const f32data = Float32Array.from(values);
  // Arena fast path: use resolveOutputBuffer for stable buffer identity across steps.
  // This eliminates bind group cache misses from data-source ops in lowered plans.
  if (activeArena) {
    const buffer = resolveOutputBuffer(ctx.device, f32data.byteLength, []);
    profileApiCall("writeBuffer", () => ctx.queue.writeBuffer(buffer, 0, f32data));
    return createTensor(shape, buffer, undefined, 0, "f32");
  }
  const buffer = createBufferWithData(
    ctx.device,
    f32data,
    ctx.queue,
  );
  return createTensor(shape, buffer, undefined, 0, "f32");
}

/**
 * Generate WGSL shader for GPU-side fill.
 * Fills an output buffer with a constant value using a compute shader.
 */
function fillShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
struct Params {
  size: u32,
  value: f32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  out[idx] = params.value;
}
`;
}

/**
 * Create a zero-filled tensor efficiently.
 * Allocates a GPU buffer directly — no JS array, no upload.
 * If the buffer comes from the pool (stale data), clears it with clearBuffer.
 * Fresh buffers are zero-initialized by the WebGPU spec.
 */
function zeros(shape: number[]): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }
  const sizeBytes = numElements * 4; // f32
  const alignedSize = alignBufferSize(sizeBytes);
  // Arena-aware output allocation for stable buffer identity across steps
  const buffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Arena and pooled buffers may contain stale data — always clear to zero on GPU.
  // Fresh buffers are already zero, so this is a no-op on the GPU side.
  if (bufferPool.isFromPool(buffer) || arenaBufferSet.has(buffer)) {
    if (sharedEncoderInstance) {
      sharedEncoderInstance.clearBuffer(buffer, 0, alignedSize);
    } else {
      const encoder = ctx.device.createCommandEncoder();
      encoder.clearBuffer(buffer, 0, alignedSize);
      submitOrCollect(encoder.finish());
    }
  }

  return createTensor(shape, buffer, undefined, 0, "f32");
}

/**
 * Create a tensor filled with a constant value.
 * Uses a GPU compute shader to fill the buffer — no JS array allocation.
 * fillValue === 0 is special-cased to use the zero-cost zeros() path.
 */
function full(shape: number[], fillValue: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }

  // Special case: fillValue === 0 → use zeros path (WebGPU auto-zeros or clearBuffer)
  if (fillValue === 0) {
    return zeros(shape);
  }

  const sizeBytes = numElements * 4; // f32
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `fill_${gridSizeX}`;
  const shader = fillShader(gridSizeX);
  const pipeline = getPipeline(ctx, shaderKey, shader);

  // Arena-aware output allocation for stable buffer identity across steps
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Params: [numElements as u32, fillValue as f32 (reinterpreted as u32 bits)]
  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  new Float32Array(paramsData.buffer, 4, 1)[0] = fillValue;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer, undefined, 0, "f32");
}

/**
 * Generate WGSL shader for GPU-side arange.
 * Fills output with start + idx * step.
 */
function arangeShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
struct Params {
  size: u32,
  start: f32,
  step: f32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  out[idx] = params.start + f32(idx) * params.step;
}
`;
}

/**
 * Create a 1-D tensor of evenly spaced values on the GPU.
 * No JS array allocation — values are computed directly by the GPU.
 */
function arange(end: number, start = 0, step = 1): WebGPUTensor {
  const ctx = requireContext();
  const numElements = Math.max(0, Math.ceil((end - start) / step));
  if (numElements === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }

  const sizeBytes = numElements * 4; // f32
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `arange_${gridSizeX}`;
  const shader = arangeShader(gridSizeX);
  const pipeline = getPipeline(ctx, shaderKey, shader);

  // Arena-aware output allocation for stable buffer identity across steps
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Params: [numElements as u32, start as f32, step as f32]
  const paramsData = new Uint32Array(3);
  paramsData[0] = numElements;
  new Float32Array(paramsData.buffer, 4, 1)[0] = start;
  new Float32Array(paramsData.buffer, 8, 1)[0] = step;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor([numElements], outBuffer, undefined, 0, "f32");
}

/**
 * Generate WGSL shader for tril/triu.
 * A single shader template parameterized by upper (inlined at compile time).
 * For tril: zero where col > row + k
 * For triu: zero where col < row + k
 */
function triangularShader(gridSizeX: number, upper: boolean): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  // Inlined condition: tril zeros above k-th diagonal, triu zeros below
  const zeroCondition = upper
    ? `col < row + k` // triu: zero below
    : `col > row + k`; // tril: zero above
  return `
struct Params {
  num_elements: u32,
  H: u32,
  W: u32,
  k: i32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.num_elements) {
    return;
  }
  let row = i32((idx / params.W) % params.H);
  let col = i32(idx % params.W);
  let k = params.k;
  if (${zeroCondition}) {
    output[idx] = 0.0;
  } else {
    output[idx] = input[idx];
  }
}
`;
}

/**
 * Triangular operation: zero elements above (tril) or below (triu) a diagonal.
 * Operates on the last 2 dimensions; supports arbitrary batch dimensions.
 */
function triangularOp(a: WebGPUTensor, k: number, upper: boolean): WebGPUTensor {
  const ctx = requireContext();
  if (a.shape.length < 2) throw new Error("tril/triu requires at least 2 dimensions");

  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const numElements = sizeOf(a.shape);

  // Ensure contiguous input
  const input = a.isContiguous ? a : contiguous(a);

  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const tag = upper ? "triu" : "tril";
  const shaderKey = `${tag}_${gridSizeX}`;
  const shader = triangularShader(gridSizeX, upper);
  const pipeline = getPipeline(ctx, shaderKey, shader);

  const sizeBytes = numElements * 4;
  const alignedSize = alignBufferSize(sizeBytes);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = createTrackedBuffer(ctx.device, { size: alignedSize, usage });

  // Params: [numElements as u32, H as u32, W as u32, k as i32]
  const paramsData = new Int32Array(4);
  new Uint32Array(paramsData.buffer, 0, 1)[0] = numElements;
  new Uint32Array(paramsData.buffer, 4, 1)[0] = H;
  new Uint32Array(paramsData.buffer, 8, 1)[0] = W;
  paramsData[3] = k;
  const paramsBuffer = createParamsBuffer(ctx.device, new Uint32Array(paramsData.buffer));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [input.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  // Destroy contiguous copy if one was created (deferred for GPU fence)
  if (input !== a) {
    bufferPool.decRef((input as WebGPUTensor).buffer);
    bufferPool.deferredDestroy((input as WebGPUTensor).buffer, numElements * dtypeBytes(a.dtype));
  }

  return createTensor(a.shape.slice(), outBuffer, undefined, 0, a.dtype);
}

function tril(a: WebGPUTensor, k = 0): WebGPUTensor {
  return triangularOp(a, k, false);
}

function triu(a: WebGPUTensor, k = 0): WebGPUTensor {
  return triangularOp(a, k, true);
}

/**
 * Create a tensor from an array with a specific dtype.
 * Supports f32 (default), i32, u32, and f16 (if device supports shader-f16).
 */
export function tensorFromArrayWithDtype(
  values: number[],
  shape: number[],
  dtype: DType,
): WebGPUTensor {
  const ctx = requireContext();
  const expected = sizeOf(shape);
  if (expected !== values.length) {
    throw new Error("Tensor data length does not match shape");
  }

  // Check f16 support
  if (dtype === "f16" && !ctx.f16Supported) {
    throw new Error(
      "f16 dtype requires shader-f16 device feature which is not available",
    );
  }

  let typedData: Float32Array | Int32Array | Uint32Array | Uint16Array;
  switch (dtype) {
    case "i32":
      typedData = Int32Array.from(values);
      break;
    case "u32":
      typedData = Uint32Array.from(values);
      break;
    case "f16":
      typedData = f32ArrayToF16Array(values);
      break;
    case "f32":
    default:
      typedData = Float32Array.from(values);
      break;
  }

  // Arena fast path: use resolveOutputBuffer for stable buffer identity across steps
  if (activeArena) {
    const buffer = resolveOutputBuffer(ctx.device, typedData.byteLength, []);
    profileApiCall("writeBuffer", () => ctx.queue.writeBuffer(buffer, 0, typedData));
    return createTensor(shape, buffer, undefined, 0, dtype);
  }
  const buffer = createBufferWithData(ctx.device, typedData, ctx.queue);
  return createTensor(shape, buffer, undefined, 0, dtype);
}

function add(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("+", a as WebGPUTensor, b as WebGPUTensor, options);
}

function sub(
  a: BackendTensor,
  b: BackendTensor,
  options?: SubOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("-", a as WebGPUTensor, b as WebGPUTensor, options);
}

function div(
  a: BackendTensor,
  b: BackendTensor,
  options?: DivOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("/", a as WebGPUTensor, b as WebGPUTensor, options);
}

function mul(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("*", a as WebGPUTensor, b as WebGPUTensor, options);
}

function sqrt(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("sqrt", "sqrt(x)", a as WebGPUTensor, options);
}

function relu(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary(
    "relu",
    "select(0.0, x, x > 0.0)",
    a as WebGPUTensor,
    options,
  );
}

function exp(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("exp", "exp(x)", a as WebGPUTensor, options);
}

function log(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("log", "log(x)", a as WebGPUTensor, options);
}

function neg(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("neg", "-x", a as WebGPUTensor, options);
}

function abs(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("abs", "abs(x)", a as WebGPUTensor, options);
}

function tanh(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("tanh", "tanh(x)", a as WebGPUTensor, options);
}

function sigmoid(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary(
    "sigmoid",
    "(1.0 / (1.0 + exp(-x)))",
    a as WebGPUTensor,
    options,
  );
}

function gelu(
  a: BackendTensor,
  options?: GeluOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  const approximate = options?.approximate ?? "tanh";

  if (approximate === "tanh") {
    // Tanh approximation (GPT-2 "new GELU"): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Clamp tanh input to [-10, 10] to avoid overflow (tanh saturates to ±1 beyond this)
    return dispatchUnary(
      "gelu_tanh",
      "(x * 0.5 * (1.0 + tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0))))",
      a as WebGPUTensor,
      { outBuffer: options?.outBuffer },
    );
  } else {
    // Exact formula using erf: x * 0.5 * (1 + erf(x / sqrt(2)))
    // WGSL doesn't have erf, so we use a polynomial approximation
    // Abramowitz and Stegun approximation 7.1.26 (max error ~1.5e-7)
    // Single expression with all computations inlined
    return dispatchUnary(
      "gelu_erf",
      "(x * 0.5 * (1.0 + sign(x) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + -1.453152027) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + -0.284496736) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) * exp(-x * x * 0.5)))))",
      a as WebGPUTensor,
      { outBuffer: options?.outBuffer },
    );
  }
}

function silu(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  // SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
  return dispatchUnary(
    "silu",
    "(x / (1.0 + exp(-x)))",
    a as WebGPUTensor,
    options,
  );
}

/**
 * Check if values are finite (not NaN and not Inf).
 * Returns 1.0 where finite, 0.0 where NaN or Inf.
 * Uses arithmetic checks since not all WGSL implementations support isinf/isnan.
 */
function isfinite(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  // Use bitcast to check IEEE 754 exponent bits directly.
  // A f32 is Inf or NaN when all exponent bits (bits 23-30) are set.
  // Exponent mask: 0x7F800000. If (bits & mask) == mask, value is non-finite.
  // This is robust against GPU compiler optimizations that may fold
  // arithmetic checks like x * 0.0 == 0.0 or x - x == 0.0.
  return dispatchUnary(
    "isfinite",
    "select(0.0, 1.0, (bitcast<u32>(x) & 0x7F800000u) != 0x7F800000u)",
    a as WebGPUTensor,
    options,
  );
}

/**
 * Generate shader for dtype casting with stride support.
 */
function castShader(
  srcDtype: DType,
  dstDtype: DType,
  shape: number[],
  strides: number[],
  offset: number,
  gridSizeX?: number,
): string {
  const rank = shape.length;
  const srcWgsl = dtypeToWgsl(srcDtype);
  const dstWgsl = dtypeToWgsl(dstDtype);
  const enableF16 =
    srcDtype === "f16" || dstDtype === "f16" ? "enable f16;\n" : "";
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Generate cast expression based on dtype pair
  let castExpr: string;
  if (srcDtype === dstDtype) {
    castExpr = "x";
  } else if (dstDtype === "f32") {
    castExpr = "f32(x)";
  } else if (dstDtype === "f16") {
    castExpr = "f16(x)";
  } else if (dstDtype === "i32") {
    castExpr = "i32(x)";
  } else if (dstDtype === "u32") {
    castExpr = "u32(x)";
  } else {
    castExpr = `${dstWgsl}(x)`;
  }

  if (rank === 0) {
    // Scalar case
    return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${srcWgsl}>;
@group(0) @binding(1) var<storage, read_write> out: array<${dstWgsl}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  let x = a[${offset}u];
  out[idx] = ${castExpr};
}
`;
  }

  const shapeArray = `array<u32, ${rank}>(${shape.map((s) => `${s}u`).join(", ")})`;
  const stridesArray = `array<u32, ${rank}>(${strides.map((s) => `${s}u`).join(", ")})`;

  return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${srcWgsl}>;
@group(0) @binding(1) var<storage, read_write> out: array<${dstWgsl}>;
@group(0) @binding(2) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const SHAPE = ${shapeArray};
const STRIDES = ${stridesArray};
const OFFSET: u32 = ${offset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert flat index to strided offset
  var remaining = idx;
  var inputOffset = OFFSET;
  for (var d = 0u; d < RANK; d = d + 1u) {
    var dimSize = 1u;
    for (var j = d + 1u; j < RANK; j = j + 1u) {
      dimSize = dimSize * SHAPE[j];
    }
    let coord = remaining / dimSize;
    remaining = remaining % dimSize;
    inputOffset = inputOffset + coord * STRIDES[d];
  }

  let x = a[inputOffset];
  out[idx] = ${castExpr};
}
`;
}

/**
 * Cast tensor to a different dtype.
 * Returns same tensor if already the target dtype.
 */
function cast(a: BackendTensor, dtype: DType): BackendTensor {
  const tensor = a as WebGPUTensor;
  const ctx = requireContext();

  // No-op if already the target dtype
  if (tensor.dtype === dtype) {
    return tensor;
  }

  // Check f16 weight cache (populated by Adam dual-write)
  if (dtype === "f16" && tensor.dtype === "f32") {
    const cached = f16WeightCache.get(tensor.buffer);
    if (cached) {
      if (tensor.isContiguous && tensor.offset === 0) {
        // Contiguous: direct return of cached f16 buffer
        return createTensor(tensor.shape, cached, undefined, 0, "f16", false);
      }
      // Non-contiguous view (e.g., transpose) of a cached buffer: return an
      // f16 view with the same strides/offset. The f16 buffer has the same
      // element layout as the f32 buffer (contiguous), so strided access works
      // identically — just with f16 elements instead of f32.
      if (tensor.offset === 0) {
        return createTensor(
          tensor.shape,
          cached,
          tensor.strides,
          0,
          "f16",
          false,
        );
      }
    }
  }

  // Check f16 support
  if (dtype === "f16" && !ctx.f16Supported) {
    throw new Error(
      "f16 dtype requires shader-f16 device feature which is not available",
    );
  }

  // Check if buffer exceeds maxStorageBufferBindingSize — use chunked path
  const srcBytesPerElement = dtypeBytes(tensor.dtype);
  const dstBytesPerElement = dtypeBytes(dtype);
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const inputDataBytes = tensor.size * srcBytesPerElement;
  const outputDataBytes = tensor.size * dstBytesPerElement;

  if (inputDataBytes > maxBindingSize || outputDataBytes > maxBindingSize) {
    // Chunked cast requires contiguous input
    let src = tensor;
    let contiguousCopy: WebGPUTensor | null = null;
    if (!tensor.isContiguous || tensor.offset > 0) {
      src = contiguous(tensor) as WebGPUTensor;
      contiguousCopy = src;
    }
    const result = castChunked(src, dtype, ctx, maxBindingSize, limits);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(tensor.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = castShader(
    tensor.dtype,
    dtype,
    tensor.shape,
    tensor.strides,
    tensor.offset,
    use2D ? dispatch.gridSizeX : undefined,
  );
  const key = `cast:${tensor.dtype}->${dtype}:${tensor.shape.join("x")}:${tensor.strides.join(",")}:${tensor.offset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const bytesPerElement = dtypeBytes(dtype);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    tensor.size * bytesPerElement,
    [tensor.buffer],
  );
  const params = createUniformBuffer(ctx.device, tensor.size);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, params]);
  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(params);
  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

/**
 * Chunked cast dispatch for tensors exceeding maxStorageBufferBindingSize.
 * Input must be contiguous with offset 0.
 */
function castChunked(
  tensor: WebGPUTensor,
  dtype: DType,
  ctx: ReturnType<typeof requireContext>,
  maxBindingSize: number,
  limits: Record<string, number>,
): BackendTensor {
  const srcBytesPerElement = dtypeBytes(tensor.dtype);
  const dstBytesPerElement = dtypeBytes(dtype);
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Alignment must satisfy both src and dst offset alignment requirements
  const srcElemsPerAlign = minAlignment / srcBytesPerElement;
  const dstElemsPerAlign = minAlignment / dstBytesPerElement;
  const elementsPerAlignment = lcm(srcElemsPerAlign, dstElemsPerAlign);

  // Max elements per chunk: limited by binding size for both src and dst
  const maxByBinding = Math.floor(maxBindingSize / Math.max(srcBytesPerElement, dstBytesPerElement));
  const elementsPerChunk = Math.floor(maxByBinding / elementsPerAlignment) * elementsPerAlignment;

  const totalElements = tensor.size;
  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  // Create full output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * dstBytesPerElement,
    [tensor.buffer],
  );

  // Determine 2D dispatch dimensions for max chunk size
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Build shader for chunked cast (contiguous, no strides/offset)
  const srcWgslType = tensor.dtype === "f16" ? "f16" : tensor.dtype === "i32" ? "i32" : tensor.dtype === "u32" ? "u32" : "f32";
  const dstWgslType = dtype === "f16" ? "f16" : dtype === "i32" ? "i32" : dtype === "u32" ? "u32" : "f32";
  const f16Enable = (tensor.dtype === "f16" || dtype === "f16") ? "enable f16;\n" : "";
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `${f16Enable}
@group(0) @binding(0) var<storage, read> a: array<${srcWgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${dstWgslType}>;

struct Params {
  chunkSize: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  out[idx] = ${dstWgslType}(a[idx]);
}
`;

  const key = `castChunked:${tensor.dtype}->${dtype}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  // Process each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;

    const srcByteOffset = chunkStart * srcBytesPerElement;
    const srcByteSize = chunkSize * srcBytesPerElement;
    const dstByteOffset = chunkStart * dstBytesPerElement;
    const dstByteSize = chunkSize * dstBytesPerElement;

    const paramsBuffer = createUniformBuffer(ctx.device, chunkSize);

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer, offset: srcByteOffset, size: srcByteSize } },
        { binding: 1, resource: { buffer: outBuffer, offset: dstByteOffset, size: dstByteSize } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
    releaseUniformBuffer(paramsBuffer);
  }

  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

/**
 * Infer strides for a new shape given old shape/strides, without copying data.
 * Returns null if the reshape requires a contiguous copy.
 * Implements PyTorch's computeStride algorithm.
 */
function inferReshapeStrides(
  oldShape: number[],
  oldStrides: number[],
  newShape: number[],
): number[] | null {
  if (newShape.length === 0) return [];
  if (oldShape.length === 0) return contiguousStrides(newShape);

  const newStrides = new Array<number>(newShape.length);

  // Work through old and new dims left-to-right, grouping contiguous chunks
  let oldIdx = 0;
  let newIdx = 0;
  const oldN = oldShape.length;
  const newN = newShape.length;

  while (newIdx < newN) {
    // Skip size-1 dims in new shape (stride is irrelevant)
    if (newShape[newIdx] === 1) {
      // Use a sensible stride for size-1 dims
      newStrides[newIdx] = newIdx + 1 < newN ? newStrides[newIdx + 1] || 1 : 1;
      newIdx++;
      continue;
    }

    // Skip size-1 dims in old shape
    while (oldIdx < oldN && oldShape[oldIdx] === 1) oldIdx++;
    if (oldIdx >= oldN) return null;

    // Accumulate a group of old dims and match to new dims
    let oldProduct = oldShape[oldIdx];
    let newProduct = newShape[newIdx];

    // Collect old dims until oldProduct >= newProduct
    const groupStart = oldIdx;
    while (oldProduct < newProduct && oldIdx + 1 < oldN) {
      // Check contiguity between consecutive old dims
      if (oldStrides[oldIdx] !== oldStrides[oldIdx + 1] * oldShape[oldIdx + 1]) {
        return null; // Non-contiguous boundary
      }
      oldIdx++;
      // Skip size-1 old dims within group
      if (oldShape[oldIdx] === 1) continue;
      oldProduct *= oldShape[oldIdx];
    }

    // Collect new dims until newProduct >= oldProduct
    const newGroupStart = newIdx;
    while (newProduct < oldProduct && newIdx + 1 < newN) {
      newIdx++;
      if (newShape[newIdx] === 1) {
        newStrides[newIdx] = 1;
        continue;
      }
      newProduct *= newShape[newIdx];
    }

    if (oldProduct !== newProduct) return null;

    // Assign strides for the new dims in this group (right-to-left)
    // The rightmost new dim gets the stride of the rightmost old dim
    let stride = oldStrides[oldIdx];
    for (let i = newIdx; i >= newGroupStart; i--) {
      if (newShape[i] === 1) {
        newStrides[i] = stride;
        continue;
      }
      newStrides[i] = stride;
      stride *= newShape[i];
    }

    oldIdx++;
    newIdx++;
  }

  // Check remaining old dims are all size 1
  while (oldIdx < oldN) {
    if (oldShape[oldIdx] !== 1) return null;
    oldIdx++;
  }

  return newStrides;
}

function reshape(a: BackendTensor, shape: number[]): BackendTensor {
  const tensor = a as WebGPUTensor;
  const expected = sizeOf(shape);
  if (expected !== tensor.size) {
    throw new Error("View shape does not match tensor size");
  }

  if (tensor.isContiguous) {
    // Fast path: contiguous input → contiguous output view
    return createTensor(shape, tensor.buffer, undefined, tensor.offset, tensor.dtype, false);
  }

  // Non-contiguous: try to compute valid strides for new shape
  const newStrides = inferReshapeStrides(tensor.shape, tensor.strides, shape);
  if (newStrides !== null) {
    // Compatible layout: return view with computed strides (zero-cost)
    return createTensor(shape, tensor.buffer, newStrides, tensor.offset, tensor.dtype, false);
  }

  // Incompatible: must materialize first, transfer buffer ownership to result
  const contig = contiguous(tensor) as WebGPUTensor;
  bufferPool.decRef(contig.buffer); // Transfer ownership to result tensor
  return createTensor(shape, contig.buffer, undefined,
    contig.offset, tensor.dtype, true);
}

/**
 * Expand returns a VIEW - no data copy, just metadata change.
 * Broadcast dimensions get stride=0 (same element repeated).
 */
function expand(a: BackendTensor, shape: number[]): BackendTensor {
  const tensor = a as WebGPUTensor;
  const inputShape = tensor.shape;
  const inputStrides = tensor.strides;

  // Validate shapes are compatible for broadcasting
  if (shape.length < inputShape.length) {
    throw new Error(
      "expand: target shape must have at least as many dimensions as input",
    );
  }

  // Compute output strides with broadcasting
  // For broadcast dims (input dim = 1, output dim > 1), stride = 0
  // For leading dims (not in input), stride = 0
  // For matching dims, use input stride
  const outStrides: number[] = [];
  const padded = shape.length - inputShape.length;

  for (let i = 0; i < shape.length; i++) {
    if (i < padded) {
      // Leading dimension not in input - broadcast with stride 0
      outStrides.push(0);
    } else {
      const inputIdx = i - padded;
      const inputDim = inputShape[inputIdx];
      const outputDim = shape[i];

      if (inputDim === 1 && outputDim > 1) {
        // Broadcast: stride = 0 (repeat same element)
        outStrides.push(0);
      } else if (inputDim === outputDim) {
        // Same size: use existing stride
        outStrides.push(inputStrides[inputIdx]);
      } else {
        throw new Error(
          `expand: incompatible shapes at dimension ${i} (input ${inputDim} vs output ${outputDim})`,
        );
      }
    }
  }

  // Return a view sharing the same buffer
  // Note: expand views are never contiguous (they have stride=0 somewhere)
  // View - does not own the buffer
  return createTensor(
    shape,
    tensor.buffer,
    outStrides,
    tensor.offset,
    tensor.dtype,
    false,
  );
}

/**
 * Materialize a non-contiguous tensor to a new contiguous buffer.
 * If already contiguous, returns the same tensor (no-op).
 * Handles large tensors by processing in chunks.
 */
function contiguous(a: BackendTensor): BackendTensor {
  const tensor = a as WebGPUTensor;

  // Fast path: already contiguous - return a non-owning view.
  // Returning the same tensor object would cause the executor to create a
  // second StorageHandle for the same GPUBuffer.  When the intermediate
  // handle becomes unreachable, destroyUnreachable() would destroy the
  // buffer while the original tensor still references it.
  if (tensor.isContiguous) {
    return createTensor(
      tensor.shape,
      tensor.buffer,
      tensor.strides,
      tensor.offset,
      tensor.dtype,
      false, // ownsBuffer = false → won't destroy buffer on cleanup
    );
  }

  const ctx = requireContext();
  const shape = tensor.shape;
  const rank = shape.length;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  if (outSize === 0) {
    throw new Error("contiguous: empty tensors not supported");
  }

  // Check if input buffer exceeds max binding size
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Get actual buffer size (the backing storage might be larger than the view)
  const inputBufferSize = (tensor.buffer as { size: number }).size;

  if (inputBufferSize > maxBindingSize) {
    // Use chunked contiguous for large input buffers
    return contiguousChunked(tensor, maxBindingSize, minAlignment);
  }

  // Fast path: input fits in binding limit
  return contiguousDirect(tensor);
}

/**
 * Direct contiguous implementation for tensors within buffer binding limits.
 */
function contiguousDirect(tensor: WebGPUTensor): WebGPUTensor {
  const ctx = requireContext();
  const shape = tensor.shape;
  const rank = shape.length;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const wgslType = dtypeToWgsl(dtype);
  const bytesPerElement = dtypeBytes(dtype);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Create new contiguous buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [tensor.buffer],
  );

  // Generate shader that reads with strides and writes contiguous
  const shapeArray = `array<u32, ${rank}>(${shape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${tensor.strides.map((s) => `${s}u`).join(", ")})`;
  const outStridesArray = `array<u32, ${rank}>(${contiguousStrides(shape)
    .map((s) => `${s}u`)
    .join(", ")})`;
  const enableF16 = dtype === "f16" ? "enable f16;\n" : "";
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `${enableF16}
struct Params {
  size: u32,
  offset: u32,
};

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const shape = ${shapeArray};
const inputStrides = ${inputStridesArray};
const outStrides = ${outStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert output flat index to coordinates
  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }

  // Compute input offset using strides
  var inputOffset = params.offset;
  for (var d = 0u; d < RANK; d = d + 1u) {
    inputOffset = inputOffset + coords[d] * inputStrides[d];
  }

  out[idx] = input[inputOffset];
}
`;

  const key = `contiguous:${shape.join(",")}:${tensor.strides.join(",")}:${tensor.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;

  dispatchElementwise({
    key, shader: code,
    inputs: [tensor.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params2(outSize, tensor.offset),
    outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  return createTensor(shape, outBuffer, undefined, 0, dtype);
}

/**
 * Chunked contiguous for large input buffers.
 * For transposed 2D tensors [K, N] with strides [1, K], processes by output columns.
 * Each output column reads from a contiguous section of input.
 */
function contiguousChunked(
  tensor: WebGPUTensor,
  maxBindingSize: number,
  minAlignment: number,
): WebGPUTensor {
  const ctx = requireContext();
  const shape = tensor.shape;
  const rank = shape.length;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const wgslType = dtypeToWgsl(dtype);
  const bytesPerElement = dtypeBytes(dtype);

  // Currently optimized for 2D transposed tensors
  // General case would need more sophisticated chunking
  if (rank !== 2) {
    throw new Error("Chunked contiguous currently only supports 2D tensors");
  }

  const [K, N] = shape;
  const [strideK, strideN] = tensor.strides;

  // Create new contiguous buffer
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: alignBufferSize(outSize * bytesPerElement),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // For transposed tensor [K, N] with strides [1, K]:
  // Output column j reads from input[j*K : j*K+K] (contiguous in input)
  // We chunk by both input columns AND output rows to stay within binding limits
  if (strideK === 1 && strideN === K) {
    // This is a simple transpose - output column j reads input rows j
    // Challenge: output is also large and needs chunking
    // Strategy: process in row chunks where each row chunk handles all columns in sub-chunks

    const bytesPerOutputRow = N * bytesPerElement; // Output row [K elements across N columns for one K-row]
    const bytesPerInputRow = K * bytesPerElement;  // One column of input (K elements)

    // How many output rows fit in binding limit?
    const maxOutputRows = Math.floor(maxBindingSize / bytesPerOutputRow);

    // How many input columns (= K elements each) fit in binding limit?
    const maxInputCols = Math.floor(maxBindingSize / bytesPerInputRow);

    // Align row counts for buffer offsets
    const outputRowAlignment = minAlignment / gcd(bytesPerOutputRow, minAlignment);
    const inputColAlignment = minAlignment / gcd(bytesPerInputRow, minAlignment);

    const alignedOutputRows = Math.max(
      outputRowAlignment,
      Math.floor(maxOutputRows / outputRowAlignment) * outputRowAlignment
    );
    const alignedInputCols = Math.max(
      inputColAlignment,
      Math.floor(maxInputCols / inputColAlignment) * inputColAlignment
    );

    // Process output in row chunks, and within each row chunk, process input in column chunks
    const numOutputRowChunks = Math.ceil(K / alignedOutputRows);
    const numInputColChunks = Math.ceil(N / alignedInputCols);

    // Shader processes a tile: output rows [rowStart, rowEnd) and columns [colStart, colEnd)
    const code = `
struct Params {
  K: u32,
  N: u32,
  rowStart: u32,
  rowEnd: u32,
  colStart: u32,
  colEnd: u32,
  gridStride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.gridStride + gid.x;
  let numRows = params.rowEnd - params.rowStart;
  let numCols = params.colEnd - params.colStart;
  let tileSize = numRows * numCols;
  if (idx >= tileSize) { return; }

  let localRow = idx / numCols;
  let localCol = idx % numCols;
  let globalRow = params.rowStart + localRow;
  let globalCol = params.colStart + localCol;

  // Input: transposed view element [globalRow, globalCol] = buffer[globalCol * K + globalRow]
  // With offset binding at colStart * K, local index is localCol * K + globalRow
  // But we need to account for the fact that globalRow is the actual row index
  let inputIdx = localCol * params.K + globalRow;

  // Output: row chunk is bound with offset, so local row index is localRow
  // Column index is the global column
  let outputIdx = localRow * params.N + globalCol;

  output[outputIdx] = input[inputIdx];
}
`;

    const key = `contiguous_chunked_transpose_tiled:${K}:${N}:${dtype}`;
    const pipeline = getPipeline(ctx, key, code);

    for (let rowChunk = 0; rowChunk < numOutputRowChunks; rowChunk++) {
      const rowStart = rowChunk * alignedOutputRows;
      const rowEnd = Math.min(rowStart + alignedOutputRows, K);
      const numRows = rowEnd - rowStart;

      // Output row chunk binding
      const outputByteOffset = rowStart * bytesPerOutputRow;
      const outputChunkSize = numRows * bytesPerOutputRow;

      for (let colChunk = 0; colChunk < numInputColChunks; colChunk++) {
        const colStart = colChunk * alignedInputCols;
        const colEnd = Math.min(colStart + alignedInputCols, N);
        const numCols = colEnd - colStart;

        // Input column chunk binding: columns [colStart, colEnd) means buffer positions [colStart*K, colEnd*K)
        const inputByteOffset = (tensor.offset + colStart * K) * bytesPerElement;
        const inputChunkSize = numCols * K * bytesPerElement;

        const tileSize = numRows * numCols;
        const dispatch = compute2DDispatch(Math.ceil(tileSize / WORKGROUP_SIZE));

        const paramsBuffer = createParamsBuffer(ctx.device, params7(K, N, rowStart, rowEnd, colStart, colEnd, dispatch.gridSizeX * WORKGROUP_SIZE));

        const bindGroup = profiledCreateBindGroup(ctx.device,{
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            {
              binding: 0,
              resource: {
                buffer: tensor.buffer,
                offset: inputByteOffset,
                size: inputChunkSize,
              } as GPUBufferBinding,
            },
            {
              binding: 1,
              resource: {
                buffer: outBuffer,
                offset: outputByteOffset,
                size: outputChunkSize,
              } as GPUBufferBinding,
            },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });

        dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

        releaseParamsBuffer(paramsBuffer);
      }
    }

    return createTensor(shape, outBuffer, undefined, 0, dtype);
  }

  // For other stride patterns, use a more general approach
  // Process output in chunks and bind input sections as needed
  // This is a fallback that may not be optimal for all patterns
  throw new Error(
    `Chunked contiguous not yet implemented for stride pattern [${tensor.strides.join(", ")}]`
  );
}

/**
 * Detect if a tensor is a simple last-2-dim transpose of a contiguous buffer.
 * If so, we can pass the original contiguous buffer to the matmul shader with a flipped
 * transpose flag, avoiding a contiguous() materialization dispatch.
 *
 * Returns the "original" contiguous shape (last 2 dims swapped back) if detected,
 * or null if the tensor is not a simple transpose.
 */
function detectSimpleTranspose(tensor: WebGPUTensor): number[] | null {
  if (tensor.isContiguous) return null; // Already contiguous, no transpose to detect
  if (tensor.offset !== 0) return null; // Non-zero offset not supported
  const rank = tensor.shape.length;
  if (rank < 2) return null;

  const strides = tensor.strides;
  const shape = tensor.shape;

  // For a last-2-dim transpose of a contiguous buffer:
  // strides[-1] should equal shape[-2] (the original inner dim stride)
  // strides[-2] should equal 1 (the original innermost stride)
  if (strides[rank - 2] !== 1) return null;
  if (strides[rank - 1] !== shape[rank - 2]) return null;

  // Check batch dimensions are contiguous: each batch stride should equal
  // the product of all inner dimensions' sizes in the original layout.
  // Original shape has last 2 dims swapped: [...batch, shape[-1], shape[-2]]
  let expectedStride = shape[rank - 1] * shape[rank - 2]; // innermost 2D block
  for (let i = rank - 3; i >= 0; i--) {
    if (strides[i] !== expectedStride) return null;
    expectedStride *= shape[i];
  }

  // Construct the original contiguous shape (swap last 2 dims back)
  const originalShape = shape.slice();
  originalShape[rank - 2] = shape[rank - 1];
  originalShape[rank - 1] = shape[rank - 2];
  return originalShape;
}

/**
 * Helper to ensure a tensor is contiguous, materializing if needed.
 */
function ensureContiguous(tensor: WebGPUTensor): WebGPUTensor {
  return (tensor.isContiguous ? tensor : contiguous(tensor)) as WebGPUTensor;
}

/**
 * Select a contiguous sub-range along one dimension. Returns a view (zero GPU cost).
 * The returned tensor shares the same buffer with an adjusted offset.
 */
function narrow(a: BackendTensor, dim: number, start: number, length: number): BackendTensor {
  const tensor = a as WebGPUTensor;
  const rank = tensor.shape.length;
  if (dim < 0 || dim >= rank) {
    throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
  }
  if (start < 0 || start + length > tensor.shape[dim]) {
    throw new Error(`narrow: range [${start}, ${start + length}) out of bounds for dim size ${tensor.shape[dim]}`);
  }
  const newShape = tensor.shape.slice();
  newShape[dim] = length;
  const newOffset = tensor.offset + start * tensor.strides[dim];
  return createTensor(newShape, tensor.buffer, tensor.strides.slice(), newOffset, tensor.dtype, false);
}

/**
 * Backward for narrow: pad gradient back to original shape.
 * Writes grad into [start, start+length) along dim, zeros elsewhere.
 */
function narrowBackward(grad: BackendTensor, dim: number, start: number, originalLength: number): BackendTensor {
  const gradTensor = ensureContiguous(grad as WebGPUTensor);
  const ctx = requireContext();

  const outShape = gradTensor.shape.slice();
  outShape[dim] = originalLength;
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const dtype = gradTensor.dtype;
  const bytesPerElement = dtype === "f16" ? 2 : 4;

  const outerSize = outShape.slice(0, dim).reduce((a, b) => a * b, 1);
  const innerSize = outShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const gradDimSize = gradTensor.shape[dim]; // = length from narrow
  const outDimSize = originalLength;

  const wgslType = dtype === "f16" ? "f16" : "f32";
  const WG = WORKGROUP_SIZE;

  const totalWorkgroups = Math.ceil(outSize / WG);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const gridSizeX = dispatch.x * WG;

  const shaderCode = `
${dtype === "f16" ? "enable f16;\n" : ""}
@group(0) @binding(0) var<storage, read> grad: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // outerSize, innerSize, gradDimSize, start

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${use2D ? `gid.x + gid.y * ${gridSizeX}u` : "gid.x"};
  let total = ${outSize}u;
  if (idx >= total) { return; }

  let innerSize = params.y;
  let outDimSize = ${outDimSize}u;
  let outerIdx = idx / (outDimSize * innerSize);
  let remainder = idx % (outDimSize * innerSize);
  let dimIdx = remainder / innerSize;
  let innerIdx = remainder % innerSize;

  let startOffset = params.w;
  if (dimIdx >= startOffset && dimIdx < startOffset + params.z) {
    let gradDimIdx = dimIdx - startOffset;
    let gradIdx = outerIdx * params.z * innerSize + gradDimIdx * innerSize + innerIdx;
    out[idx] = grad[gradIdx];
  } else {
    out[idx] = ${wgslType}(0.0);
  }
}
`;

  const key = `narrowBackward:${outDimSize}:${gradDimSize}:${start}:${outSize}:${dtype}:${use2D ? `2d:${gridSizeX}` : "1d"}`;

  const outBuffer = dispatchElementwise({
    key, shader: shaderCode,
    inputs: [gradTensor.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params4(outerSize, innerSize, gradDimSize, start),
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  if (gradTensor !== (grad as WebGPUTensor)) {
    bufferPool.decRef(gradTensor.buffer);
    bufferPool.deferredDestroy(gradTensor.buffer, gradTensor.size * bytesPerElement);
  }

  return createTensor(outShape, outBuffer, undefined, 0, dtype);
}

/**
 * Transpose returns a VIEW - no data copy, just metadata change.
 * The returned tensor shares the same buffer but with swapped strides.
 */
function transpose(a: BackendTensor, options: TransposeOptions): BackendTensor {
  const tensor = a as WebGPUTensor;
  const { dim0, dim1 } = options;
  const inputShape = tensor.shape;
  const rank = inputShape.length;

  if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
    throw new Error("transpose: dimension out of range");
  }

  // Swap shape dimensions
  const outShape = inputShape.slice();
  outShape[dim0] = inputShape[dim1];
  outShape[dim1] = inputShape[dim0];

  // Swap strides - this is the key to view-based transpose
  const outStrides = tensor.strides.slice();
  outStrides[dim0] = tensor.strides[dim1];
  outStrides[dim1] = tensor.strides[dim0];

  // Return a view sharing the same buffer
  // Note: createTensor will correctly compute isContiguous=false for transposed strides
  // View - does not own the buffer
  return createTensor(
    outShape,
    tensor.buffer,
    outStrides,
    tensor.offset,
    tensor.dtype,
    false,
  );
}

/**
 * Permute dimensions according to the given order.
 * Returns a view sharing the same buffer (no data copy).
 *
 * Example: permute([2, 3, 4], [2, 0, 1]) -> [4, 2, 3]
 */
function permute(a: BackendTensor, dims: number[]): BackendTensor {
  const tensor = a as WebGPUTensor;
  const inputShape = tensor.shape;
  const rank = inputShape.length;

  // Validate dims
  if (dims.length !== rank) {
    throw new Error(
      `permute: dims length ${dims.length} doesn't match tensor rank ${rank}`,
    );
  }

  // Check for valid permutation (each dim appears exactly once)
  const seen = new Set<number>();
  for (const d of dims) {
    if (d < 0 || d >= rank) {
      throw new Error(`permute: dimension ${d} out of range for rank ${rank}`);
    }
    if (seen.has(d)) {
      throw new Error(`permute: duplicate dimension ${d}`);
    }
    seen.add(d);
  }

  // Reorder shape and strides according to dims
  const outShape = dims.map((d) => inputShape[d]);
  const outStrides = dims.map((d) => tensor.strides[d]);

  // Return a view sharing the same buffer
  // View - does not own the buffer
  return createTensor(
    outShape,
    tensor.buffer,
    outStrides,
    tensor.offset,
    tensor.dtype,
    false,
  );
}

function matmul(
  _a: BackendTensor,
  _b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const ctx = requireContext();
  const a = _a as WebGPUTensor;
  const b = _b as WebGPUTensor;

  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Check if B matrix exceeds max buffer binding size
  const bSizeBytes = b.size * dtypeBytes(b.dtype);

  if (bSizeBytes > maxBindingSize) {
    // Chunked path for large B matrices
    return matmulChunked(a, b, maxBindingSize);
  }

  // Compute output size to check if it exceeds limit
  // Output shape: broadcast(a[:-1], b[:-2]) + [a[-2], b[-1]]
  const aShape = a.shape;
  const bShape = b.shape;
  const M = aShape[aShape.length - 2];
  const N = bShape[bShape.length - 1];
  // Compute batch dimensions
  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch.length > 0 ? aBatch : [1], bBatch.length > 0 ? bBatch : [1]);
  const outShape = [...batchShape, M, N];
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outputDtype = (a.dtype === "f32" || b.dtype === "f32") ? "f32" as const : a.dtype;
  const outSizeBytes = outSize * dtypeBytes(outputDtype);

  if (outSizeBytes > maxBindingSize) {
    // Chunked path for large output
    return matmulChunkedOutput(a, b, maxBindingSize);
  }

  // Fast path: existing implementation
  return dispatchMatmul(a, b, false, false, options?.outBuffer);
}

/**
 * Extract column slice from a 2D matrix to a contiguous buffer.
 * Input: [K, N], Output: [K, colEnd - colStart]
 * Handles large input matrices by processing in row chunks.
 */
function sliceColumns(
  input: WebGPUTensor,
  colStart: number,
  colEnd: number,
): WebGPUTensor {
  const ctx = requireContext();
  const [K, N] = input.shape;
  const sliceWidth = colEnd - colStart;
  const outSize = K * sliceWidth;

  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Check if input exceeds binding limit
  const inputSizeBytes = input.size * 4;
  const needsChunking = inputSizeBytes > maxBindingSize;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Shader to copy column slice (with row offset support for chunking)
  const shaderCode = `
    struct Params {
      numRows: u32,
      N: u32,
      colStart: u32,
      sliceWidth: u32,
      rowStart: u32,
      gridStride: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.y * params.gridStride + gid.x;
      let totalSize = params.numRows * params.sliceWidth;
      if (idx >= totalSize) { return; }

      let localRow = idx / params.sliceWidth;
      let col = idx % params.sliceWidth;
      let srcCol = params.colStart + col;
      // Input offset is relative to chunk start (row 0 of bound range)
      let srcIdx = localRow * params.N + srcCol;
      // Output offset accounts for rowStart
      let dstIdx = (params.rowStart + localRow) * params.sliceWidth + col;

      output[dstIdx] = input[srcIdx];
    }
  `;

  const module = ctx.device.createShaderModule({ code: shaderCode });
  const pipeline = ctx.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  if (!needsChunking) {
    // Fast path: single dispatch
    const dispatch = compute2DDispatch(Math.ceil(outSize / WORKGROUP_SIZE));
    const paramsBuffer = createParamsBuffer(ctx.device, params6(K, N, colStart, sliceWidth, 0, dispatch.gridSizeX * WORKGROUP_SIZE));

    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [input.buffer, outBuffer, paramsBuffer]);

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

    releaseParamsBuffer(paramsBuffer);
  } else {
    // Chunked path: process rows in chunks that fit in binding limit
    const bytesPerRow = N * 4;

    // Calculate how many rows must group together for aligned offsets
    // We need rowStart * bytesPerRow to be divisible by minAlignment
    // Find the smallest rowAlignment where rowAlignment * bytesPerRow % minAlignment == 0
    const g = gcd(bytesPerRow, minAlignment);
    const rowAlignment = minAlignment / g;

    // How many rows fit in maxBindingSize?
    const maxRowsUnaligned = Math.floor(maxBindingSize / bytesPerRow);
    // Round down to nearest multiple of rowAlignment
    const rowsPerChunk = Math.max(rowAlignment, Math.floor(maxRowsUnaligned / rowAlignment) * rowAlignment);

    const numRowChunks = Math.ceil(K / rowsPerChunk);

    for (let chunk = 0; chunk < numRowChunks; chunk++) {
      const rowStart = chunk * rowsPerChunk;
      const rowEnd = Math.min(rowStart + rowsPerChunk, K);
      const numRows = rowEnd - rowStart;

      // Calculate byte offset and size for this chunk
      const byteOffset = rowStart * bytesPerRow;
      const chunkByteSize = numRows * bytesPerRow;

      const chunkSize = numRows * sliceWidth;
      const dispatch = compute2DDispatch(Math.ceil(chunkSize / WORKGROUP_SIZE));

      const paramsBuffer = createParamsBuffer(ctx.device, params6(numRows, N, colStart, sliceWidth, rowStart, dispatch.gridSizeX * WORKGROUP_SIZE));

      const bindGroup = profiledCreateBindGroup(ctx.device,{
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: input.buffer,
              offset: byteOffset,
              size: chunkByteSize,
            } as GPUBufferBinding,
          },
          { binding: 1, resource: { buffer: outBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

      releaseParamsBuffer(paramsBuffer);
    }
  }

  return createTensor([K, sliceWidth], outBuffer, undefined, 0, "f32", true);
}

/**
 * Write partial matmul result to columns of output buffer.
 * partial: [M, sliceWidth], output: [M, N], writes to columns [colStart, colStart+sliceWidth)
 * Handles large output buffers by processing in row chunks.
 */
function scatterColumnsToOutput(
  partial: WebGPUTensor,
  outBuffer: GPUBuffer,
  M: number,
  N: number,
  colStart: number,
): void {
  const ctx = requireContext();
  const sliceWidth = partial.shape[partial.shape.length - 1];
  const totalRows = partial.size / sliceWidth; // M (could be M*batch for batched)

  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Check if output buffer exceeds limit
  const outputBufferSize = (outBuffer as { size: number }).size;
  const inputBufferSize = (partial.buffer as { size: number }).size;

  const needsChunking = outputBufferSize > maxBindingSize || inputBufferSize > maxBindingSize;

  const shaderCode = `
    struct Params {
      numRows: u32,
      N: u32,
      colStart: u32,
      sliceWidth: u32,
      rowStart: u32,
      inputRowStart: u32,
      gridStride: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.y * params.gridStride + gid.x;
      let totalSize = params.numRows * params.sliceWidth;
      if (idx >= totalSize) { return; }

      let localRow = idx / params.sliceWidth;
      let col = idx % params.sliceWidth;

      // Input index: relative to bound chunk
      let inputIdx = (params.inputRowStart + localRow) * params.sliceWidth + col;

      // Output: write to row (rowStart + localRow), column (colStart + col)
      // Output offset is relative to bound chunk
      let outputIdx = localRow * params.N + (params.colStart + col);

      output[outputIdx] = input[inputIdx];
    }
  `;

  const module = ctx.device.createShaderModule({ code: shaderCode });
  const pipeline = ctx.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  if (!needsChunking) {
    // Fast path: single dispatch
    const totalSize = totalRows * sliceWidth;
    const dispatch = compute2DDispatch(Math.ceil(totalSize / WORKGROUP_SIZE));
    const paramsBuffer = createParamsBuffer(ctx.device, params7(totalRows, N, colStart, sliceWidth, 0, 0, dispatch.gridSizeX * WORKGROUP_SIZE));

    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [partial.buffer, outBuffer, paramsBuffer]);

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

    releaseParamsBuffer(paramsBuffer);
  } else {
    // Chunked path: process rows in chunks
    // Output row size in bytes = N * 4
    // Input row size in bytes = sliceWidth * 4
    const outputBytesPerRow = N * 4;
    const inputBytesPerRow = sliceWidth * 4;

    // Find row chunk size that fits both input and output in binding limit
    const maxOutputRows = Math.floor(maxBindingSize / outputBytesPerRow);
    const maxInputRows = Math.floor(maxBindingSize / inputBytesPerRow);
    const maxRowsPerChunk = Math.min(maxOutputRows, maxInputRows);

    // Align for buffer offsets
    const outputG = gcd(outputBytesPerRow, minAlignment);
    const inputG = gcd(inputBytesPerRow, minAlignment);
    const outputRowAlignment = minAlignment / outputG;
    const inputRowAlignment = minAlignment / inputG;
    const rowAlignment = Math.max(outputRowAlignment, inputRowAlignment);

    const alignedRowsPerChunk = Math.max(
      rowAlignment,
      Math.floor(maxRowsPerChunk / rowAlignment) * rowAlignment
    );

    const numChunks = Math.ceil(totalRows / alignedRowsPerChunk);

    for (let chunk = 0; chunk < numChunks; chunk++) {
      const rowStart = chunk * alignedRowsPerChunk;
      const rowEnd = Math.min(rowStart + alignedRowsPerChunk, totalRows);
      const numRows = rowEnd - rowStart;

      // Calculate byte offsets
      const outputByteOffset = rowStart * outputBytesPerRow;
      const inputByteOffset = rowStart * inputBytesPerRow;

      const outputChunkSize = numRows * outputBytesPerRow;
      const inputChunkSize = numRows * inputBytesPerRow;

      const chunkSize = numRows * sliceWidth;
      const dispatch = compute2DDispatch(Math.ceil(chunkSize / WORKGROUP_SIZE));

      const paramsBuffer = createParamsBuffer(ctx.device, params7(numRows, N, colStart, sliceWidth, 0, 0, dispatch.gridSizeX * WORKGROUP_SIZE));

      const bindGroup = profiledCreateBindGroup(ctx.device,{
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: partial.buffer,
              offset: inputByteOffset,
              size: inputChunkSize,
            } as GPUBufferBinding,
          },
          {
            binding: 1,
            resource: {
              buffer: outBuffer,
              offset: outputByteOffset,
              size: outputChunkSize,
            } as GPUBufferBinding,
          },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

      releaseParamsBuffer(paramsBuffer);
    }
  }
}

/**
 * Chunked matmul for large B matrices that exceed buffer binding limits.
 *
 * Strategy: Use the underlying buffer of B directly (even if B is a transposed view)
 * and chunk by BUFFER rows (which are contiguous), using transB=true.
 * This avoids the need to materialize a large contiguous copy.
 */
function matmulChunked(
  a: WebGPUTensor,
  b: WebGPUTensor,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Ensure A is contiguous (A is typically small)
  const aContiguous = ensureContiguous(a);
  const aWasCopied = aContiguous !== a;

  // Get dimensions from the logical shapes
  const aRank = aContiguous.shape.length;
  const bRank = b.shape.length;

  if (bRank !== 2) {
    if (aWasCopied) aContiguous.destroy?.();
    throw new Error("Chunked matmul currently only supports 2D B matrix");
  }

  const M = aContiguous.shape[aRank - 2];
  const K_a = aContiguous.shape[aRank - 1];
  const [K_b, N] = b.shape;

  if (K_a !== K_b) {
    if (aWasCopied) aContiguous.destroy?.();
    throw new Error(`Matmul dimension mismatch: A[...,${K_a}] vs B[${K_b},${N}]`);
  }
  const K = K_a;

  // Compute batch dimensions from A
  const batchDims = aContiguous.shape.slice(0, -2);
  const batchSize = batchDims.reduce((acc, d) => acc * d, 1) || 1;

  // Output shape: [...batchDims, M, N]
  const outShape = [...batchDims, M, N];

  // Check if B is a transposed view of a contiguous buffer
  // Transposed [K, N] from original [N, K] has strides [1, K]
  const isTransposedView = b.strides[0] === 1 && b.strides[1] === K;

  if (isTransposedView) {
    const result = matmulChunkedTransposed(
      aContiguous,
      b,
      M,
      K,
      N,
      batchSize,
      batchDims,
      outShape,
      maxBindingSize,
      minAlignment
    );
    if (aWasCopied) aContiguous.destroy?.();
    return result;
  }

  // B is contiguous or has a different stride pattern
  // For contiguous B [K, N], chunk by rows (each row is N elements)
  if (!b.isContiguous) {
    if (aWasCopied) aContiguous.destroy?.();
    throw new Error("Chunked matmul for non-contiguous non-transposed B not yet implemented");
  }

  const result = matmulChunkedContiguous(
    aContiguous,
    b,
    M,
    K,
    N,
    batchSize,
    batchDims,
    outShape,
    maxBindingSize,
    minAlignment
  );
  if (aWasCopied) aContiguous.destroy?.();
  return result;
}

/**
 * Chunked matmul for transposed B.
 * B is a view [K, N] of underlying buffer [N, K].
 * We chunk the buffer by rows (= logical columns) and use transB=true.
 */
function matmulChunkedTransposed(
  a: WebGPUTensor,
  b: WebGPUTensor,
  M: number,
  K: number,
  N: number,
  batchSize: number,
  batchDims: number[],
  outShape: number[],
  maxBindingSize: number,
  minAlignment: number,
): WebGPUTensor {
  const ctx = requireContext();

  // The underlying buffer is [N, K] = N rows of K elements each
  // Each buffer row corresponds to one logical column of B
  const bytesPerBufferRow = K * 4;

  // How many buffer rows fit in one binding?
  const maxBufferRowsPerChunk = Math.floor(maxBindingSize / bytesPerBufferRow);

  // Ensure chunk boundaries are alignment-friendly
  const g = gcd(bytesPerBufferRow, minAlignment);
  const rowAlignment = minAlignment / g;
  const alignedRowsPerChunk = Math.max(
    rowAlignment,
    Math.floor(maxBufferRowsPerChunk / rowAlignment) * rowAlignment
  );

  const numChunks = Math.ceil(N / alignedRowsPerChunk);

  // We'll process each chunk and accumulate partial outputs
  // Output is [...batchDims, M, N] and each chunk contributes columns [colStart, colEnd)
  const partialOutputs: { tensor: WebGPUTensor; colStart: number; colEnd: number }[] = [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const bufferRowStart = chunk * alignedRowsPerChunk;
    const bufferRowEnd = Math.min(bufferRowStart + alignedRowsPerChunk, N);
    const numBufferRows = bufferRowEnd - bufferRowStart;

    // These buffer rows correspond to logical columns [bufferRowStart, bufferRowEnd)
    const colStart = bufferRowStart;
    const colEnd = bufferRowEnd;
    const chunkWidth = numBufferRows;

    // Create a view of the buffer chunk: [numBufferRows, K]
    const byteOffset = (b.offset + bufferRowStart * K) * 4;
    const chunkByteSize = numBufferRows * K * 4;

    // Create a temporary tensor that wraps just this chunk of the buffer
    // This is [chunkWidth, K] in buffer layout
    const bChunkBuffer = createTrackedBuffer(ctx.device, {
      size: chunkByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Copy the chunk from the original buffer
    if (sharedEncoderInstance) {
      sharedEncoderInstance.copyBufferToBuffer(b.buffer, byteOffset, bChunkBuffer, 0, chunkByteSize);
    } else {
      const encoder = ctx.device.createCommandEncoder();
      encoder.copyBufferToBuffer(b.buffer, byteOffset, bChunkBuffer, 0, chunkByteSize);
      submitOrCollect(encoder.finish());
    }

    // Create tensor for the chunk: [chunkWidth, K]
    const bChunk = createTensor([chunkWidth, K], bChunkBuffer, undefined, 0, "f32", true);

    // Matmul: A [M, K] @ bChunk.T [K, chunkWidth] = [M, chunkWidth]
    // Use transB=true since bChunk is [chunkWidth, K] and we want [K, chunkWidth]
    const partialResult = dispatchMatmul(a, bChunk, false, true);

    // Store the partial result tensor (owns the buffer)
    partialOutputs.push({
      tensor: partialResult,
      colStart,
      colEnd,
    });

    // Destroy bChunk - its buffer data has been consumed by the matmul dispatch
    bChunk.destroy?.();
  }

  // Now assemble the final output from partial results
  // Each partial is [batchSize * M, chunkWidth] and goes to columns [colStart, colEnd)
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  for (const partial of partialOutputs) {
    const { tensor: partialTensor, colStart, colEnd } = partial;
    const chunkWidth = colEnd - colStart;

    // Scatter columns from partial [batchSize*M, chunkWidth] to output [batchSize*M, N]
    scatterColumnsToOutput(
      partialTensor,
      outBuffer,
      batchSize * M,
      N,
      colStart
    );

    // Destroy the partial result buffer (deferred destruction waits for GPU fence)
    partialTensor.destroy?.();
  }

  return createTensor(outShape, outBuffer, undefined, 0, "f32", true);
}

/**
 * Chunked matmul for contiguous B [K, N].
 * Chunks B by columns (which requires copying column slices).
 */
function matmulChunkedContiguous(
  a: WebGPUTensor,
  b: WebGPUTensor,
  M: number,
  K: number,
  N: number,
  batchSize: number,
  batchDims: number[],
  outShape: number[],
  maxBindingSize: number,
  minAlignment: number,
): WebGPUTensor {
  const ctx = requireContext();

  // For contiguous B [K, N], columns are NOT contiguous
  // We need to extract column slices to separate buffers

  // Each column is K elements = K * 4 bytes
  const bytesPerColumn = K * 4;
  const maxColumnsPerChunk = Math.floor(maxBindingSize / bytesPerColumn);

  // Ensure alignment
  const g = gcd(bytesPerColumn, minAlignment);
  const colAlignment = minAlignment / g;
  const alignedColumnsPerChunk = Math.max(
    colAlignment,
    Math.floor(maxColumnsPerChunk / colAlignment) * colAlignment
  );

  const numChunks = Math.ceil(N / alignedColumnsPerChunk);

  const partialOutputs: { tensor: WebGPUTensor; colStart: number; colEnd: number }[] = [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const colStart = chunk * alignedColumnsPerChunk;
    const colEnd = Math.min(colStart + alignedColumnsPerChunk, N);
    const chunkWidth = colEnd - colStart;

    // Extract column slice using sliceColumns
    const bSlice = sliceColumns(b, colStart, colEnd);

    // Matmul: A [M, K] @ bSlice [K, chunkWidth] = [M, chunkWidth]
    const partialResult = dispatchMatmul(a, bSlice, false, false);

    partialOutputs.push({
      tensor: partialResult,
      colStart,
      colEnd,
    });

    // Destroy bSlice after matmul dispatch - destroy() uses deferred destruction
    // which waits for GPU fence before actually freeing the buffer
    bSlice.destroy?.();
  }

  // Assemble final output
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  for (const partial of partialOutputs) {
    const { tensor: partialTensor, colStart, colEnd } = partial;
    const chunkWidth = colEnd - colStart;

    scatterColumnsToOutput(
      partialTensor,
      outBuffer,
      batchSize * M,
      N,
      colStart
    );

    // Destroy the partial result buffer (deferred destruction waits for GPU fence)
    partialTensor.destroy?.();
  }

  return createTensor(outShape, outBuffer, undefined, 0, "f32", true);
}

/**
 * Chunked matmul when the OUTPUT exceeds the buffer binding limit.
 * This happens when A and B are small but their product is large.
 * Strategy: chunk along the N (columns) dimension of B and output.
 */
function matmulChunkedOutput(
  a: WebGPUTensor,
  b: WebGPUTensor,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Get shapes
  const aShape = a.shape;
  const bShape = b.shape;
  const M = aShape[aShape.length - 2];
  const K = aShape[aShape.length - 1];
  const N = bShape[bShape.length - 1];

  // Compute batch dimensions
  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch.length > 0 ? aBatch : [1], bBatch.length > 0 ? bBatch : [1]);
  const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
  const outShape = [...batchShape, M, N];

  // Calculate how many columns we can output per chunk
  // Each output column is batchSize * M elements
  const elementsPerColumn = batchSize * M;
  const bytesPerColumn = elementsPerColumn * 4;
  const maxColumnsPerChunk = Math.floor(maxBindingSize / bytesPerColumn);

  // Ensure alignment for B slicing
  const bBytesPerColumn = K * 4;
  const g = gcd(bBytesPerColumn, minAlignment);
  const colAlignment = Math.max(1, minAlignment / g);
  const alignedColumnsPerChunk = Math.max(
    colAlignment,
    Math.floor(maxColumnsPerChunk / colAlignment) * colAlignment
  );

  const numChunks = Math.ceil(N / alignedColumnsPerChunk);

  // Create output buffer
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Reshape A to [batchSize, M, K] for consistent processing
  const aReshaped = a.shape.length === 2
    ? createTensor([1, M, K], a.buffer, a.strides ? [0, ...a.strides] : undefined, a.offset, "f32", false)
    : a;

  // Reshape B to [batchSize, K, N] for consistent processing
  const bReshaped = b.shape.length === 2
    ? createTensor([1, K, N], b.buffer, b.strides ? [0, ...b.strides] : undefined, b.offset, "f32", false)
    : b;

  // Process each column chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const colStart = chunk * alignedColumnsPerChunk;
    const colEnd = Math.min(colStart + alignedColumnsPerChunk, N);
    const chunkWidth = colEnd - colStart;

    // Slice B to get columns [colStart:colEnd] -> shape [batchSize, K, chunkWidth]
    // For contiguous B [batchSize, K, N], columns aren't contiguous, so we need to copy
    const bSlice = sliceBColumns(bReshaped, colStart, colEnd);

    // Compute partial matmul: [batchSize, M, K] @ [batchSize, K, chunkWidth] = [batchSize, M, chunkWidth]
    const partialResult = dispatchMatmul(aReshaped, bSlice, false, false);

    // Copy partial result to output buffer at the right column offset
    scatterColumnsToOutput(
      partialResult,
      outBuffer,
      batchSize * M,
      N,
      colStart
    );

    // Destroy temporary buffers after scattering (deferred destruction waits for GPU fence)
    bSlice.destroy?.();
    partialResult.destroy?.();
  }

  return createTensor(outShape, outBuffer, undefined, 0, "f32", true);
}

/**
 * Slice columns from B matrix for chunked output matmul.
 * Input shape: [batch, K, N], Output shape: [batch, K, colEnd - colStart]
 */
function sliceBColumns(
  b: WebGPUTensor,
  colStart: number,
  colEnd: number,
): WebGPUTensor {
  const ctx = requireContext();
  const [batch, K, N] = b.shape;
  const chunkWidth = colEnd - colStart;
  const outSize = batch * K * chunkWidth;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Shader to copy column slice
  const sliceTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const sliceDispatch = compute2DDispatch(sliceTotalWG);
  const sliceUse2D = sliceDispatch.y > 1;
  const sliceGridSizeX = sliceDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  batch: u32,
  K: u32,
  N: u32,
  colStart: u32,
  chunkWidth: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${sliceUse2D ? `gid.x + gid.y * ${sliceGridSizeX}u` : "gid.x"};
  let totalSize = params.batch * params.K * params.chunkWidth;
  if (idx >= totalSize) { return; }

  // Convert flat idx to (b, k, c) in output space
  let c = idx % params.chunkWidth;
  let k = (idx / params.chunkWidth) % params.K;
  let batchIdx = idx / (params.K * params.chunkWidth);

  // Compute input offset: (batchIdx, k, colStart + c)
  let inputOffset = batchIdx * params.K * params.N + k * params.N + params.colStart + c;

  output[idx] = input[inputOffset];
}
`;

  const key = `sliceBColumns:${batch}:${K}:${N}:${WORKGROUP_SIZE}:${sliceUse2D ? "2d" : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const paramsBuffer = createParamsBuffer(ctx.device, params5(batch, K, N, colStart, chunkWidth));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [b.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, sliceDispatch.x, sliceDispatch.y);

  releaseParamsBuffer(paramsBuffer);

  return createTensor([batch, K, chunkWidth], outBuffer);
}

function gather(
  a: BackendTensor,
  index: BackendTensor,
  options: GatherOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensorA = a as WebGPUTensor;
  const tensorIndex = index as WebGPUTensor;
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("gather: dimension out of range");
  }

  // Check if input tensor exceeds max buffer binding size
  const inputSizeBytes = tensorA.size * 4; // f32 = 4 bytes
  const gatherLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = gatherLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  if (inputSizeBytes > maxBindingSize) {
    // Chunked path for large input tensors
    return gatherChunked(tensorA, tensorIndex, options, maxBindingSize);
  }

  // Fast path: existing implementation for normal-sized tensors
  return gatherDirect(tensorA, tensorIndex, options);
}

/**
 * Direct gather implementation for tensors within buffer binding limits.
 */
function gatherDirect(
  tensorA: WebGPUTensor,
  tensorIndex: WebGPUTensor,
  options: GatherOptions,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = tensorA.shape;
  const indexShape = tensorIndex.shape;
  const rank = inputShape.length;

  // Output shape is same as index shape
  const outShape = indexShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * 4,
    [tensorA.buffer, tensorIndex.buffer],
  );

  // Compute input strides
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  // Compute index strides
  const indexStrides: number[] = [];
  for (let i = 0; i < indexShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < indexShape.length; j++) {
      stride *= indexShape[j];
    }
    indexStrides.push(stride);
  }

  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const indexShapeArray = `array<u32, ${rank}>(${indexShape.map((s) => `${s}u`).join(", ")})`;
  const indexStridesArray = `array<u32, ${rank}>(${indexStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const inputStrides = ${inputStridesArray};
const indexShape = ${indexShapeArray};
const indexStrides = ${indexStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert flat index to coordinates in index tensor
  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / indexStrides[d];
    remaining = remaining % indexStrides[d];
  }

  // Get the gather index for the specified dimension
  let gatherIdx = u32(indices[idx]);

  // Compute input offset using coords, but replace dim with gatherIdx
  var inputOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      inputOffset = inputOffset + gatherIdx * inputStrides[d];
    } else {
      inputOffset = inputOffset + coords[d] * inputStrides[d];
    }
  }

  out[idx] = input[inputOffset];
}
`;

  const pipeline = getPipeline(
    ctx,
    `gather:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );
  const uniformBuffer = createUniformBuffer(ctx.device, outSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensorA.buffer, tensorIndex.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Chunked gather implementation for tensors exceeding buffer binding limits.
 *
 * Strategy: Partition dispatch by which indices fall into which chunk of the input.
 * Each dispatch binds a different chunk of the input buffer (using WebGPU's offset binding).
 * Each dispatch only processes indices that fall within its chunk's range.
 * Output accumulates across dispatches (each writes to non-overlapping output positions).
 */
function gatherChunked(
  input: WebGPUTensor,
  index: WebGPUTensor,
  options: GatherOptions,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = input.shape;
  const indexShape = index.shape;
  const rank = inputShape.length;
  const dimSize = inputShape[dim];

  // Output shape is same as index shape
  const outShape = indexShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);

  // Calculate slice size (elements per entry along the gather dimension)
  // For dim=0 on [vocabSize, embedDim], this is embedDim
  const elementsPerSlice = inputShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const bytesPerSlice = elementsPerSlice * 4; // f32 = 4 bytes

  // WebGPU requires buffer binding offsets to be aligned (typically 256 bytes)
  const deviceLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;

  // Ensure chunk boundaries are aligned by adjusting slices per chunk
  // We need (slicesPerChunk * bytesPerSlice) to be a multiple of minAlignment
  let maxSlicesPerChunk = Math.floor(maxBindingSize / bytesPerSlice);

  // If bytesPerSlice isn't a multiple of minAlignment, we need to adjust
  if (bytesPerSlice % minAlignment !== 0) {
    // Find how many slices we need for aligned chunk boundaries
    const slicesForAlignment = minAlignment / gcd(bytesPerSlice, minAlignment);
    // Round down maxSlicesPerChunk to a multiple of slicesForAlignment
    maxSlicesPerChunk = Math.floor(maxSlicesPerChunk / slicesForAlignment) * slicesForAlignment;
    if (maxSlicesPerChunk === 0) {
      maxSlicesPerChunk = slicesForAlignment; // At least one aligned group
    }
  }

  const numChunks = Math.ceil(dimSize / maxSlicesPerChunk);

  // Create output buffer
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Compute input strides
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  // Compute index strides
  const indexStrides: number[] = [];
  for (let i = 0; i < indexShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < indexShape.length; j++) {
      stride *= indexShape[j];
    }
    indexStrides.push(stride);
  }

  // Compute 2D dispatch for large output
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const indexShapeArray = `array<u32, ${rank}>(${indexShape.map((s) => `${s}u`).join(", ")})`;
  const indexStridesArray = `array<u32, ${rank}>(${indexStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Shader with chunk bounds checking
  const code = `
struct Params {
  size: u32,
  chunkStart: u32,
  chunkEnd: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const inputStrides = ${inputStridesArray};
const indexShape = ${indexShapeArray};
const indexStrides = ${indexStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Get the gather index for the specified dimension
  let gatherIdx = u32(indices[idx]);

  // Only process if this index falls in our chunk
  if (gatherIdx < params.chunkStart || gatherIdx >= params.chunkEnd) {
    return;  // Skip - will be handled by another chunk's dispatch
  }

  // Convert flat index to coordinates in index tensor
  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / indexStrides[d];
    remaining = remaining % indexStrides[d];
  }

  // Adjust index to be relative to chunk start
  let localIdx = gatherIdx - params.chunkStart;

  // Compute input offset using coords, but replace dim with localIdx
  var inputOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      inputOffset = inputOffset + localIdx * inputStrides[d];
    } else {
      inputOffset = inputOffset + coords[d] * inputStrides[d];
    }
  }

  out[idx] = input[inputOffset];
}
`;

  const pipeline = getPipeline(
    ctx,
    `gatherChunked:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );

  // Dispatch for each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * bytesPerSlice;

    const uniformBuffer = createParamsBuffer(ctx.device, params4(outSize, chunkStart, chunkEnd, 0));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: input.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          } as GPUBufferBinding,
        },
        { binding: 1, resource: { buffer: index.buffer } },
        { binding: 2, resource: { buffer: outBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseParamsBuffer(uniformBuffer);
  }

  return createTensor(outShape, outBuffer);
}

function scatterAdd(
  a: BackendTensor,
  index: BackendTensor,
  src: BackendTensor,
  options: ScatterAddOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensorA = a as WebGPUTensor;
  const tensorIndex = index as WebGPUTensor;
  const tensorSrc = src as WebGPUTensor;
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("scatterAdd: dimension out of range");
  }

  // Check if output tensor exceeds max buffer binding size
  const outputSizeBytes = tensorA.size * 4; // f32 = 4 bytes
  const scatterLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = scatterLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  if (outputSizeBytes > maxBindingSize) {
    // Chunked path for large output tensors
    return scatterAddChunked(tensorA, tensorIndex, tensorSrc, options, maxBindingSize);
  }

  // Fast path: existing implementation for normal-sized tensors
  return scatterAddDirect(tensorA, tensorIndex, tensorSrc, options);
}

/**
 * Direct scatterAdd implementation for tensors within buffer binding limits.
 */
function scatterAddDirect(
  tensorA: WebGPUTensor,
  tensorIndex: WebGPUTensor,
  tensorSrc: WebGPUTensor,
  options: ScatterAddOptions,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;

  // Output shape is same as input shape
  const outShape = inputShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const srcSize = tensorSrc.shape.reduce((a, b) => a * b, 1);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // First, copy input to output
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * 4,
    [tensorA.buffer, tensorIndex.buffer, tensorSrc.buffer],
  );

  {
    if (sharedEncoderInstance) {
      sharedEncoderInstance.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
    } else {
      const enc = ctx.device.createCommandEncoder();
      enc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
      submitOrCollect(enc.finish());
    }
  }

  // Compute strides for output and src
  const outStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  const srcStrides: number[] = [];
  for (let i = 0; i < tensorSrc.shape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < tensorSrc.shape.length; j++) {
      stride *= tensorSrc.shape[j];
    }
    srcStrides.push(stride);
  }

  const outStridesArray = `array<u32, ${rank}>(${outStrides.map((s) => `${s}u`).join(", ")})`;
  const srcShapeArray = `array<u32, ${rank}>(${tensorSrc.shape.map((s) => `${s}u`).join(", ")})`;
  const srcStridesArray = `array<u32, ${rank}>(${srcStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let srcIdx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let srcIdx = gid.x;`;

  // Note: scatterAdd with atomics would be more correct, but for simplicity
  // we use a loop-based approach that processes each src element sequentially
  // This is less parallel but handles the general case correctly
  const code = `
struct Params {
  srcSize: u32,
};

@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const outStrides = ${outStridesArray};
const srcShape = ${srcShapeArray};
const srcStrides = ${srcStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (srcIdx >= params.srcSize) {
    return;
  }

  // Convert src flat index to coordinates
  var coords: array<u32, ${rank}>;
  var remaining = srcIdx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / srcStrides[d];
    remaining = remaining % srcStrides[d];
  }

  // Get the scatter index for the specified dimension
  let scatterIdx = u32(indices[srcIdx]);

  // Compute output offset using coords, but replace dim with scatterIdx
  var outOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      outOffset = outOffset + scatterIdx * outStrides[d];
    } else {
      outOffset = outOffset + coords[d] * outStrides[d];
    }
  }

  // Atomic add would be ideal here, but f32 atomics aren't widely supported
  // For now, we accept potential race conditions for overlapping indices
  out[outOffset] = out[outOffset] + src[srcIdx];
}
`;

  const pipeline = getPipeline(
    ctx,
    `scatterAdd:${inputShape.join(",")}:${tensorSrc.shape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );
  const uniformBuffer = createUniformBuffer(ctx.device, srcSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensorIndex.buffer, tensorSrc.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Chunked scatterAdd implementation for output tensors exceeding buffer binding limits.
 *
 * Strategy: Partition dispatch by which scatter targets fall into which chunk of the output.
 * Each dispatch binds a different chunk of the output buffer (using WebGPU's offset binding).
 * Each dispatch only processes source elements whose scatter target falls in its chunk's range.
 */
function scatterAddChunked(
  tensorA: WebGPUTensor,
  tensorIndex: WebGPUTensor,
  tensorSrc: WebGPUTensor,
  options: ScatterAddOptions,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;
  const dimSize = inputShape[dim];

  // Output shape is same as input shape
  const outShape = inputShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const srcSize = tensorSrc.shape.reduce((a, b) => a * b, 1);

  // Calculate slice size (elements per entry along the scatter dimension)
  const elementsPerSlice = inputShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const bytesPerSlice = elementsPerSlice * 4; // f32 = 4 bytes

  // WebGPU requires buffer binding offsets to be aligned (typically 256 bytes)
  const deviceLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;

  // Ensure chunk boundaries are aligned by adjusting slices per chunk
  // We need (slicesPerChunk * bytesPerSlice) to be a multiple of minAlignment
  let maxSlicesPerChunk = Math.floor(maxBindingSize / bytesPerSlice);

  // If bytesPerSlice isn't a multiple of minAlignment, we need to adjust
  if (bytesPerSlice % minAlignment !== 0) {
    // Find how many slices we need for aligned chunk boundaries
    const slicesForAlignment = minAlignment / gcd(bytesPerSlice, minAlignment);
    // Round down maxSlicesPerChunk to a multiple of slicesForAlignment
    maxSlicesPerChunk = Math.floor(maxSlicesPerChunk / slicesForAlignment) * slicesForAlignment;
    if (maxSlicesPerChunk === 0) {
      maxSlicesPerChunk = slicesForAlignment; // At least one aligned group
    }
  }

  const numChunks = Math.ceil(dimSize / maxSlicesPerChunk);

  // Create output buffer (full size - we'll bind chunks of it)
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // Copy input to output first
  {
    if (sharedEncoderInstance) {
      sharedEncoderInstance.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
    } else {
      const enc = ctx.device.createCommandEncoder();
      enc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
      submitOrCollect(enc.finish());
    }
  }

  // Compute strides for output and src
  const outStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  const srcStrides: number[] = [];
  for (let i = 0; i < tensorSrc.shape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < tensorSrc.shape.length; j++) {
      stride *= tensorSrc.shape[j];
    }
    srcStrides.push(stride);
  }

  // Compute 2D dispatch for large src tensors
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const outStridesArray = `array<u32, ${rank}>(${outStrides.map((s) => `${s}u`).join(", ")})`;
  const srcShapeArray = `array<u32, ${rank}>(${tensorSrc.shape.map((s) => `${s}u`).join(", ")})`;
  const srcStridesArray = `array<u32, ${rank}>(${srcStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let srcIdx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let srcIdx = gid.x;`;

  // Shader with chunk bounds checking
  const code = `
struct Params {
  srcSize: u32,
  chunkStart: u32,
  chunkEnd: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const outStrides = ${outStridesArray};
const srcShape = ${srcShapeArray};
const srcStrides = ${srcStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (srcIdx >= params.srcSize) {
    return;
  }

  // Get the scatter index for the specified dimension
  let scatterIdx = u32(indices[srcIdx]);

  // Only process if scatter target falls in our output chunk
  if (scatterIdx < params.chunkStart || scatterIdx >= params.chunkEnd) {
    return;
  }

  // Convert src flat index to coordinates
  var coords: array<u32, ${rank}>;
  var remaining = srcIdx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / srcStrides[d];
    remaining = remaining % srcStrides[d];
  }

  // Adjust index to be relative to chunk start
  let localIdx = scatterIdx - params.chunkStart;

  // Compute output offset using coords, but replace dim with localIdx
  var outOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      outOffset = outOffset + localIdx * outStrides[d];
    } else {
      outOffset = outOffset + coords[d] * outStrides[d];
    }
  }

  // Atomic add would be ideal here, but f32 atomics aren't widely supported
  // For now, we accept potential race conditions for overlapping indices
  out[outOffset] = out[outOffset] + src[srcIdx];
}
`;

  const pipeline = getPipeline(
    ctx,
    `scatterAddChunked:${inputShape.join(",")}:${tensorSrc.shape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );

  // Dispatch for each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * bytesPerSlice;

    const uniformBuffer = createParamsBuffer(ctx.device, params4(srcSize, chunkStart, chunkEnd, 0));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensorIndex.buffer } },
        { binding: 1, resource: { buffer: tensorSrc.buffer } },
        {
          binding: 2,
          resource: {
            buffer: outBuffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          } as GPUBufferBinding,
        },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseParamsBuffer(uniformBuffer);
  }

  return createTensor(outShape, outBuffer);
}

function sum(a: BackendTensor, options?: SumOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = a as WebGPUTensor;
  let contiguousCopy: WebGPUTensor | null = null;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  // The sum kernels assume contiguous layout for index computation
  if (!tensor.isContiguous) {
    tensor = contiguous(tensor) as WebGPUTensor;
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;

  // Handle full reduction (no dim specified or dim is null)
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    // Full reduction - returns 0-d tensor (shape [])
    const result = sumFullReduction(ctx, tensor);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  // Dimension-wise reduction
  const dims = Array.isArray(dim) ? dim : [dim];

  // Normalize negative dimensions
  const rank = inputShape.length;
  const normalizedDims = dims.map((d) => (d < 0 ? d + rank : d));

  // Validate dimensions
  for (const d of normalizedDims) {
    if (d < 0 || d >= rank) {
      throw new Error(`sum: dimension ${d} out of range`);
    }
  }

  // Compute output shape
  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) {
        outShape.push(1);
      }
    } else {
      outShape.push(inputShape[i]);
    }
  }

  // If output is scalar, handle specially
  if (outShape.length === 0) {
    const result = sumFullReduction(ctx, tensor);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  // Compute strides
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  const outStrides: number[] = [];
  for (let i = 0; i < outShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < outShape.length; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  // Compute reduction size (product of reduced dimensions)
  let reductionSize = 1;
  for (const d of normalizedDims) {
    reductionSize *= inputShape[d];
  }

  // Build mapping from input dimension to output dimension (or -1 if reduced without keepdim)
  // This tells us which output coordinate corresponds to each input dimension
  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) {
        inputToOutDim.push(outDimIdx); // maps to a size-1 dim in output
        outDimIdx++;
      } else {
        inputToOutDim.push(-1); // not in output
      }
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const outShapeArray =
    outShape.length > 0
      ? `array<u32, ${outShape.length}>(${outShape.map((s) => `${s}u`).join(", ")})`
      : "";
  const outStridesArray =
    outStrides.length > 0
      ? `array<u32, ${outStrides.length}>(${outStrides.map((s) => `${s}u`).join(", ")})`
      : "";
  const reduceDimsArray = `array<u32, ${normalizedDims.length}>(${normalizedDims.map((d) => `${d}u`).join(", ")})`;
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d) => `${d}i`).join(", ")})`;

  const sumTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const sumDispatch = compute2DDispatch(sumTotalWG);
  const sumUse2D = sumDispatch.y > 1;
  const sumGridSizeX = sumDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  outSize: u32,
  reductionSize: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const NUM_REDUCE_DIMS: u32 = ${normalizedDims.length}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const reduceDims = ${reduceDimsArray};
const inputToOutDim = ${inputToOutDimArray};

fn isReduceDim(d: u32) -> bool {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return true;
    }
  }
  return false;
}

fn getReduceDimIndex(d: u32) -> u32 {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return i;
    }
  }
  return 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = ${sumUse2D ? `gid.x + gid.y * ${sumGridSizeX}u` : "gid.x"};
  if (outIdx >= params.outSize) {
    return;
  }

  // Convert output index to output coordinates
  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${
    outShape.length > 0
      ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  `
      : ""
  }

  // Sum over all reduction indices
  var total = 0.0;
  for (var reduceIdx = 0u; reduceIdx < params.reductionSize; reduceIdx = reduceIdx + 1u) {
    // Convert reduceIdx to coordinates in reduced dimensions
    var reduceCoords: array<u32, ${Math.max(normalizedDims.length, 1)}>;
    var rRemaining = reduceIdx;
    ${
      normalizedDims.length > 0
        ? normalizedDims
            .map(
              (_, i) => `
    {
      var rDimSize = 1u;
      for (var j = ${i + 1}u; j < NUM_REDUCE_DIMS; j = j + 1u) {
        rDimSize = rDimSize * inputShape[reduceDims[j]];
      }
      reduceCoords[${i}u] = rRemaining / rDimSize;
      rRemaining = rRemaining % rDimSize;
    }
    `,
            )
            .join("")
        : ""
    }

    // Build full input coordinates
    var inputOffset = 0u;
    for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
      var coord = 0u;
      if (isReduceDim(d)) {
        // Use the reduce coordinate
        let rIdx = getReduceDimIndex(d);
        coord = reduceCoords[rIdx];
      } else {
        // Use the output coordinate - find which output dim this maps to
        let outD = inputToOutDim[d];
        if (outD >= 0i) {
          coord = outCoords[u32(outD)];
        }
      }
      inputOffset = inputOffset + coord * inputStrides[d];
    }

    total = total + input[inputOffset];
  }

  out[outIdx] = total;
}
`;

  const pipeline = getPipeline(
    ctx,
    `sum:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim}:${sumUse2D ? "2d" : "1d"}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(outSize, reductionSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, sumDispatch.x, sumDispatch.y);

  releaseParamsBuffer(paramsBuffer);
  if (contiguousCopy) contiguousCopy.destroy();

  return createTensor(outShape, outBuffer);
}

/**
 * Dimension-wise sum with a fused elementwise preamble.
 * Instead of `total += input[offset]`, computes `total += preambleExpr(input0[offset], input1[offset])`.
 * This eliminates the need for a separate elementwise kernel before the reduction.
 */
export function sumDimWithPreamble(
  inputs: BackendTensor[],
  preambleOp: string,
  sumOptions: SumOptions,
): BackendTensor {
  const getExprFn = getExprFromRegistry;
  const isUnaryOpFn = isUnaryOpFromRegistry;
  const ctx = requireContext();

  // All inputs must be contiguous and same shape
  const tensor0 = inputs[0] as WebGPUTensor;
  const inputShape = tensor0.shape;
  const rank = inputShape.length;

  const dim = sumOptions?.dim;
  const keepdim = sumOptions?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    // Full reduction with preamble
    return sumFullReductionWithPreamble(ctx, inputs, preambleOp);
  }

  const dims = Array.isArray(dim) ? dim : [dim];
  const normalizedDims = dims.map((d: number) => (d < 0 ? d + rank : d));

  // Compute output shape
  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) outShape.push(1);
    } else {
      outShape.push(inputShape[i]);
    }
  }

  if (outShape.length === 0) {
    return sumFullReductionWithPreamble(ctx, inputs, preambleOp);
  }

  const outSize = outShape.reduce((acc: number, d: number) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Compute strides
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) stride *= inputShape[j];
    inputStrides.push(stride);
  }
  const outStrides: number[] = [];
  for (let i = 0; i < outShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < outShape.length; j++) stride *= outShape[j];
    outStrides.push(stride);
  }

  let reductionSize = 1;
  for (const d of normalizedDims) reductionSize *= inputShape[d];

  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) { inputToOutDim.push(outDimIdx); outDimIdx++; }
      else inputToOutDim.push(-1);
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

  const isUnary = isUnaryOpFn(preambleOp);
  const arity = isUnary ? 1 : 2;

  // Build input bindings
  const inputBindings = inputs.map((inp: BackendTensor, i: number) =>
    `@group(0) @binding(${i}) var<storage, read> input${i}: array<f32>;`
  ).join("\n");

  // Build preamble expression
  const inputExprs = Array.from({ length: arity }, (_, i) => `input${i}[inputOffset]`);
  const preambleExpr = getExprFn(preambleOp, inputExprs);

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s: number) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s: number) => `${s}u`).join(", ")})`;
  const outShapeArray = outShape.length > 0
    ? `array<u32, ${outShape.length}>(${outShape.map((s: number) => `${s}u`).join(", ")})` : "";
  const outStridesArray = outStrides.length > 0
    ? `array<u32, ${outStrides.length}>(${outStrides.map((s: number) => `${s}u`).join(", ")})` : "";
  const reduceDimsArray = `array<u32, ${normalizedDims.length}>(${normalizedDims.map((d: number) => `${d}u`).join(", ")})`;
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d: number) => `${d}i`).join(", ")})`;

  const spTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const spDispatch = compute2DDispatch(spTotalWG);
  const spUse2D = spDispatch.y > 1;
  const spGridSizeX = spDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  outSize: u32,
  reductionSize: u32,
};

${inputBindings}
@group(0) @binding(${arity}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${arity + 1}) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const NUM_REDUCE_DIMS: u32 = ${normalizedDims.length}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const reduceDims = ${reduceDimsArray};
const inputToOutDim = ${inputToOutDimArray};

fn isReduceDim(d: u32) -> bool {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) { return true; }
  }
  return false;
}

fn getReduceDimIndex(d: u32) -> u32 {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) { return i; }
  }
  return 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = ${spUse2D ? `gid.x + gid.y * ${spGridSizeX}u` : "gid.x"};
  if (outIdx >= params.outSize) { return; }

  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${outShape.length > 0 ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  ` : ""}

  var total = 0.0;
  for (var reduceIdx = 0u; reduceIdx < params.reductionSize; reduceIdx = reduceIdx + 1u) {
    var reduceCoords: array<u32, ${Math.max(normalizedDims.length, 1)}>;
    var rRemaining = reduceIdx;
    ${normalizedDims.map((_: number, i: number) => `
    {
      var rDimSize = 1u;
      for (var j = ${i + 1}u; j < NUM_REDUCE_DIMS; j = j + 1u) {
        rDimSize = rDimSize * inputShape[reduceDims[j]];
      }
      reduceCoords[${i}u] = rRemaining / rDimSize;
      rRemaining = rRemaining % rDimSize;
    }
    `).join("")}

    var inputOffset = 0u;
    for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
      var coord = 0u;
      if (isReduceDim(d)) {
        let rIdx = getReduceDimIndex(d);
        coord = reduceCoords[rIdx];
      } else {
        let outD = inputToOutDim[d];
        if (outD >= 0i) { coord = outCoords[u32(outD)]; }
      }
      inputOffset = inputOffset + coord * inputStrides[d];
    }

    total = total + ${preambleExpr};
  }

  out[outIdx] = total;
}
`;

  const cacheKey = `sumPreamble:${preambleOp}:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim}:${spUse2D ? "2d" : "1d"}`;
  const pipeline = getPipeline(ctx, cacheKey, code);
  const paramsBuffer = createParamsBuffer(ctx.device, params2(outSize, reductionSize));

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < arity; i++) {
    bgBuffers.push((inputs[i] as WebGPUTensor).buffer);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(paramsBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

  dispatchComputePass(pipeline, bindGroup, spDispatch.x, spDispatch.y);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Full reduction sum with fused elementwise preamble.
 */
function sumFullReductionWithPreamble(
  ctx: WebGPUContext,
  inputs: BackendTensor[],
  preambleOp: string,
): WebGPUTensor {
  const getExprFn = getExprFromRegistry;
  const isUnaryOpFn = isUnaryOpFromRegistry;
  const tensor0 = inputs[0] as WebGPUTensor;
  const inputSize = tensor0.size;

  const isUnary = isUnaryOpFn(preambleOp);
  const arity = isUnary ? 1 : 2;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const inputBindings = inputs.slice(0, arity).map((_: BackendTensor, i: number) =>
    `@group(0) @binding(${i}) var<storage, read> input${i}: array<f32>;`
  ).join("\n");
  const inputExprs = Array.from({ length: arity }, (_, i) => `input${i}[i]`);
  const preambleExpr = getExprFn(preambleOp, inputExprs);

  const code = `
struct Params {
  size: u32,
};

${inputBindings}
@group(0) @binding(${arity}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${arity + 1}) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.size; i = i + 1u) {
    sum = sum + ${preambleExpr};
  }
  out[0] = sum;
}
`;

  const cacheKey = `sumFullPreamble:${preambleOp}:${inputSize}`;
  const pipeline = getPipeline(ctx, cacheKey, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < arity; i++) {
    bgBuffers.push((inputs[i] as WebGPUTensor).buffer);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(uniformBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

  dispatchComputePass(pipeline, bindGroup, 1);
  releaseUniformBuffer(uniformBuffer);

  return createTensor([], outBuffer);
}

// Helper for full reduction to scalar
async function sumFullReductionAsync(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): Promise<number> {
  const inputSize = tensor.size;

  // Simple sequential reduction for now
  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;

  // Each thread loads and sums its elements
  var sum = 0.0;
  var i = idx;
  while (i < params.size) {
    sum = sum + input[i];
    i = i + ${WORKGROUP_SIZE}u * ${Math.ceil(inputSize / WORKGROUP_SIZE)}u;
  }
  shared[localIdx] = sum;

  workgroupBarrier();

  // Parallel reduction in shared memory
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localIdx < stride) {
      shared[localIdx] = shared[localIdx] + shared[localIdx + stride];
    }
    workgroupBarrier();
  }

  // First thread writes result
  if (localIdx == 0u) {
    out[wid.x] = shared[0];
  }
}
`;

  // For simplicity, do a two-pass reduction
  const numWorkgroups = Math.ceil(inputSize / WORKGROUP_SIZE);
  const intermediateBuffer = createTrackedBuffer(ctx.device, {
    size: Math.max(numWorkgroups * 4, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const pipeline = getPipeline(ctx, `sumFull:${inputSize}`, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, intermediateBuffer, uniformBuffer]);

  // Flush shared encoder before sync readback — we need to submit all
  // prior work and then do a standalone encoder for the readback.
  if (sharedEncoder) {
    flushSharedEncoder();
  }

  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();

  // Read back intermediate results and sum on CPU (for small number of workgroups)
  const readBuffer = createTrackedBuffer(ctx.device, {
    size: numWorkgroups * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  encoder.copyBufferToBuffer(
    intermediateBuffer,
    0,
    readBuffer,
    0,
    numWorkgroups * 4,
  );
  profileApiCall("queue.submit", () => ctx.queue.submit([encoder.finish()]));
  gpuSubmitCount++;

  await readBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuffer.getMappedRange());
  let total = 0;
  for (let i = 0; i < numWorkgroups; i++) {
    total += data[i];
  }
  readBuffer.unmap();

  // Destroy temporary buffers to prevent memory leaks
  bufferPool.deferredDestroy(intermediateBuffer, (intermediateBuffer as any).size ?? numWorkgroups * 4);
  bufferPool.deferredDestroy(readBuffer, (readBuffer as any).size ?? numWorkgroups * 4);
  releaseUniformBuffer(uniformBuffer);

  return total;
}

/**
 * Full reduction sum - returns a 0-d tensor (shape []) with the sum.
 * Use item() to extract the scalar value asynchronously.
 */
function sumFullReduction(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): WebGPUTensor {
  const inputSize = tensor.size;
  const bytesPerElement = dtypeBytes(tensor.dtype);

  // Check if input buffer exceeds max binding size
  const limits = (ctx.device as unknown as { limits?: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const inputBufferSize = (tensor.buffer as { size: number }).size;

  if (inputBufferSize > maxBindingSize || inputSize * bytesPerElement > maxBindingSize) {
    return sumFullReductionChunked(ctx, tensor, maxBindingSize);
  }

  // Create output buffer with single element
  const outBuffer = resolveOutputBuffer(ctx.device, 4, [tensor.buffer]);

  // Use a simple sequential sum shader
  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.size; i = i + 1u) {
    sum = sum + input[i];
  }
  out[0] = sum;
}
`;

  const pipeline = getPipeline(ctx, `sumFullSeq:${inputSize}`, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, 1);
  releaseUniformBuffer(uniformBuffer);

  // Return 0-d tensor (shape [])
  return createTensor([], outBuffer);
}

/**
 * Chunked full reduction sum for tensors exceeding maxStorageBufferBindingSize.
 * Computes partial sums per chunk, then sums the partials.
 */
function sumFullReductionChunked(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
  maxBindingSize: number,
): WebGPUTensor {
  const bytesPerElement = dtypeBytes(tensor.dtype);
  const limits = (ctx.device as unknown as { limits?: Record<string, number> }).limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;
  const elementsPerAlignment = minAlignment / bytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const totalElements = tensor.size;
  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  // Create buffer for partial sums (one f32 per chunk)
  const partialsBuffer = createTrackedBuffer(ctx.device, {
    size: alignBufferSize(numChunks * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Shader: sequential sum of a chunk, output to out[chunkIdx]
  const code = `
struct Params {
  chunkSize: u32,
  chunkIdx: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.chunkSize; i = i + 1u) {
    sum = sum + input[i];
  }
  out[params.chunkIdx] = sum;
}
`;

  const pipeline = getPipeline(ctx, `sumFullChunked`, code);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    const paramsBuffer = createParamsBuffer(ctx.device, params2(chunkSize, chunk));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer, offset: chunkByteOffset, size: chunkByteSize } },
        { binding: 1, resource: { buffer: partialsBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, 1);
    releaseParamsBuffer(paramsBuffer);
  }

  // Now sum the partials (small buffer, fits in one binding)
  if (numChunks === 1) {
    return createTensor([], partialsBuffer);
  }

  // Final reduction of partials
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const finalCode = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.size; i = i + 1u) {
    sum = sum + input[i];
  }
  out[0] = sum;
}
`;

  const finalPipeline = getPipeline(ctx, `sumFullSeq:${numChunks}`, finalCode);
  const finalParams = createUniformBuffer(ctx.device, numChunks);

  const finalBindGroup = cachedCreateBindGroup(ctx.device, finalPipeline, [partialsBuffer, outBuffer, finalParams]);

  dispatchComputePass(finalPipeline, finalBindGroup, 1);
  releaseUniformBuffer(finalParams);

  // Destroy the intermediate partials buffer — it's been consumed by the final reduction
  bufferPool.deferredDestroy(partialsBuffer, alignBufferSize(numChunks * 4));

  return createTensor([], outBuffer);
}

type MaxOptions = { dim?: number | number[] | null; keepdim?: boolean };

/**
 * Full reduction max - returns a 0-d tensor (shape []) with the maximum.
 */
function maxFullReduction(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): WebGPUTensor {
  const inputSize = tensor.size;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var maxVal = input[0];
  for (var i = 1u; i < params.size; i = i + 1u) {
    maxVal = max(maxVal, input[i]);
  }
  out[0] = maxVal;
}
`;

  const pipeline = getPipeline(ctx, `maxFullSeq:${inputSize}`, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, 1);
  releaseUniformBuffer(uniformBuffer);

  return createTensor([], outBuffer);
}

function max(a: BackendTensor, options?: MaxOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = a as WebGPUTensor;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  if (!tensor.isContiguous) {
    tensor = contiguous(tensor) as WebGPUTensor;
  }

  const inputShape = tensor.shape;

  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    return maxFullReduction(ctx, tensor);
  }

  const dims = Array.isArray(dim) ? dim : [dim];
  const rank = inputShape.length;
  const normalizedDims = dims.map((d) => (d < 0 ? d + rank : d));

  for (const d of normalizedDims) {
    if (d < 0 || d >= rank) {
      throw new Error(`max: dimension ${d} out of range`);
    }
  }

  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) {
        outShape.push(1);
      }
    } else {
      outShape.push(inputShape[i]);
    }
  }

  if (outShape.length === 0) {
    return maxFullReduction(ctx, tensor);
  }

  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  const outStrides: number[] = [];
  for (let i = 0; i < outShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < outShape.length; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  let reductionSize = 1;
  for (const d of normalizedDims) {
    reductionSize *= inputShape[d];
  }

  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) {
        inputToOutDim.push(outDimIdx);
        outDimIdx++;
      } else {
        inputToOutDim.push(-1);
      }
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const outShapeArray =
    outShape.length > 0
      ? `array<u32, ${outShape.length}>(${outShape.map((s) => `${s}u`).join(", ")})`
      : "";
  const outStridesArray =
    outStrides.length > 0
      ? `array<u32, ${outStrides.length}>(${outStrides.map((s) => `${s}u`).join(", ")})`
      : "";
  const reduceDimsArray = `array<u32, ${normalizedDims.length}>(${normalizedDims.map((d) => `${d}u`).join(", ")})`;
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d) => `${d}i`).join(", ")})`;

  const maxTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const maxDispatch = compute2DDispatch(maxTotalWG);
  const maxUse2D = maxDispatch.y > 1;
  const maxGridSizeX = maxDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  outSize: u32,
  reductionSize: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const NUM_REDUCE_DIMS: u32 = ${normalizedDims.length}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const reduceDims = ${reduceDimsArray};
const inputToOutDim = ${inputToOutDimArray};

fn isReduceDim(d: u32) -> bool {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return true;
    }
  }
  return false;
}

fn getReduceDimIndex(d: u32) -> u32 {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return i;
    }
  }
  return 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = ${maxUse2D ? `gid.x + gid.y * ${maxGridSizeX}u` : "gid.x"};
  if (outIdx >= params.outSize) {
    return;
  }

  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${
    outShape.length > 0
      ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  `
      : ""
  }

  // Find max over all reduction indices
  var maxVal = -3.402823466e+38; // -FLT_MAX
  for (var reduceIdx = 0u; reduceIdx < params.reductionSize; reduceIdx = reduceIdx + 1u) {
    var reduceCoords: array<u32, ${Math.max(normalizedDims.length, 1)}>;
    var rRemaining = reduceIdx;
    ${
      normalizedDims.length > 0
        ? normalizedDims
            .map(
              (_, i) => `
    {
      var rDimSize = 1u;
      for (var j = ${i + 1}u; j < NUM_REDUCE_DIMS; j = j + 1u) {
        rDimSize = rDimSize * inputShape[reduceDims[j]];
      }
      reduceCoords[${i}u] = rRemaining / rDimSize;
      rRemaining = rRemaining % rDimSize;
    }
    `,
            )
            .join("")
        : ""
    }

    var inputOffset = 0u;
    for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
      var coord = 0u;
      if (isReduceDim(d)) {
        let rIdx = getReduceDimIndex(d);
        coord = reduceCoords[rIdx];
      } else {
        let outD = inputToOutDim[d];
        if (outD >= 0i) {
          coord = outCoords[u32(outD)];
        }
      }
      inputOffset = inputOffset + coord * inputStrides[d];
    }

    maxVal = max(maxVal, input[inputOffset]);
  }

  out[outIdx] = maxVal;
}
`;

  const pipeline = getPipeline(
    ctx,
    `max:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim}:${maxUse2D ? "2d" : "1d"}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(outSize, reductionSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, maxDispatch.x, maxDispatch.y);

  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}

function mean(a: BackendTensor, options?: MeanOptions): BackendTensor {
  let tensor = a as WebGPUTensor;
  let contiguousCopy: WebGPUTensor | null = null;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  // Mean uses sum internally which requires contiguous layout
  if (!tensor.isContiguous) {
    tensor = contiguous(tensor) as WebGPUTensor;
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;

  const dim = options?.dim;

  // Compute the count of elements being averaged
  let count: number;
  if (dim === undefined || dim === null) {
    count = tensor.size;
  } else {
    const dims = Array.isArray(dim) ? dim : [dim];
    const rank = inputShape.length;
    count = dims.reduce((acc, d) => {
      const nd = d < 0 ? d + rank : d;
      return acc * inputShape[nd];
    }, 1);
  }

  // Get sum result (always a tensor, possibly 0-d)
  const sumTensor = sum(a, options) as WebGPUTensor;

  // Divide by count
  const ctx = requireContext();
  const outSize = sumTensor.size;

  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [sumTensor.buffer]);

  const meanTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const meanDispatch = compute2DDispatch(meanTotalWG);
  const meanUse2D = meanDispatch.y > 1;
  const meanGridSizeX = meanDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  size: u32,
  count: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${meanUse2D ? `gid.x + gid.y * ${meanGridSizeX}u` : "gid.x"};
  if (idx >= params.size) {
    return;
  }
  out[idx] = input[idx] / params.count;
}
`;

  const pipeline = getPipeline(ctx, `meanDiv:${outSize}:${count}:${meanUse2D ? "2d" : "1d"}`, code);

  // Pack mixed u32 + f32 params into a single Uint32Array
  const meanParamsData = new ArrayBuffer(8);
  new Uint32Array(meanParamsData, 0, 1)[0] = outSize;
  new Float32Array(meanParamsData, 4, 1)[0] = count;
  const paramsBuffer = createParamsBuffer(ctx.device, new Uint32Array(meanParamsData));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [sumTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, meanDispatch.x, meanDispatch.y);

  releaseParamsBuffer(paramsBuffer);

  // Destroy intermediate sum tensor — its buffer was only needed as input to the
  // division kernel above. Without this, the 256B buffer leaks every mean() call.
  sumTensor.destroy();

  // Destroy contiguous copy if one was created
  if (contiguousCopy) {
    contiguousCopy.destroy();
  }

  return createTensor(sumTensor.shape, outBuffer);
}

// ============================================================================
// Comparison ops - return 1.0 for true, 0.0 for false
// ============================================================================

function comparisonOp(
  opName: string,
  wgslOp: string,
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const aTensor = a as WebGPUTensor;
  const bTensor = b as WebGPUTensor;

  const outShape = broadcastShapes(aTensor.shape, bTensor.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);

  const aStrides = computeEffectiveBroadcastStrides(aTensor, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(bTensor, indexShape);

  const indexing = buildBroadcastIndexing(indexShape, [aStrides, bStrides]);

  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

${indexing.declarations}
const A_OFFSET: u32 = ${aTensor.offset}u;
const B_OFFSET: u32 = ${bTensor.offset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
  let aVal = a[offset0 + A_OFFSET];
  let bVal = b[offset1 + B_OFFSET];
  out[idx] = select(0.0, 1.0, aVal ${wgslOp} bVal);
}
`;

  const key = `${opName}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${aTensor.offset}:${bTensor.offset}:${use2D ? dispatch.gridSizeX : "1d"}`;

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [aTensor.buffer, bTensor.buffer],
    outputSizeBytes: outSize * 4,
    params: params1(outSize),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  return createTensor(outShape, outBuffer);
}

function gt(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("gt", ">", a, b, options);
}

function lt(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("lt", "<", a, b, options);
}

function ge(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("ge", ">=", a, b, options);
}

function le(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("le", "<=", a, b, options);
}

function eq(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("eq", "==", a, b, options);
}

function ne(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("ne", "!=", a, b, options);
}

// ============================================================================
// ArgMax/ArgMin - return indices of max/min values
// ============================================================================

type ArgReduceOptions = { dim: number; keepdim?: boolean };

function argmax(a: BackendTensor, options: ArgReduceOptions): BackendTensor {
  return argReduceOp("argmax", ">", a, options);
}

function argmin(a: BackendTensor, options: ArgReduceOptions): BackendTensor {
  return argReduceOp("argmin", "<", a, options);
}

function argReduceOp(
  opName: string,
  compareOp: string,
  a: BackendTensor,
  options: ArgReduceOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensor = a as WebGPUTensor;
  const inputShape = tensor.shape;
  const rank = inputShape.length;

  const dim = options.dim < 0 ? options.dim + rank : options.dim;
  if (dim < 0 || dim >= rank) {
    throw new Error(
      `${opName}: dim ${options.dim} out of range for tensor of rank ${rank}`,
    );
  }
  const keepdim = options.keepdim ?? false;

  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (i === dim) {
      if (keepdim) {
        outShape.push(1);
      }
    } else {
      outShape.push(inputShape[i]);
    }
  }

  const outSize = outShape.reduce((acc, d) => acc * d, 1) || 1;
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  const outStrides: number[] = [];
  for (let i = 0; i < outShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < outShape.length; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  const dimSize = inputShape[dim];
  const dimStride = inputStrides[dim];

  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (i === dim) {
      if (keepdim) {
        inputToOutDim.push(outDimIdx);
        outDimIdx++;
      } else {
        inputToOutDim.push(-1);
      }
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const outShapeArray =
    outShape.length > 0
      ? `array<u32, ${outShape.length}>(${outShape.map((s) => `${s}u`).join(", ")})`
      : "";
  const outStridesArray =
    outStrides.length > 0
      ? `array<u32, ${outStrides.length}>(${outStrides.map((s) => `${s}u`).join(", ")})`
      : "";
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d) => `${d}i`).join(", ")})`;

  const initVal = compareOp === ">" ? "-3.402823466e+38" : "3.402823466e+38";

  const code = `
struct Params {
  outSize: u32,
  dimSize: u32,
  dimStride: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const REDUCE_DIM: u32 = ${dim}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const inputToOutDim = ${inputToOutDimArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = gid.x;
  if (outIdx >= params.outSize) {
    return;
  }

  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${
    outShape.length > 0
      ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  `
      : ""
  }

  // Compute base offset in input (with reduce dim = 0)
  var baseOffset = 0u;
  for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
    if (d != REDUCE_DIM) {
      let outD = inputToOutDim[d];
      if (outD >= 0i) {
        baseOffset = baseOffset + outCoords[u32(outD)] * inputStrides[d];
      }
    }
  }

  // Find argmax/argmin along the reduce dimension
  var bestVal = ${initVal};
  var bestIdx = 0u;
  for (var i = 0u; i < params.dimSize; i = i + 1u) {
    let val = input[baseOffset + i * params.dimStride];
    if (val ${compareOp} bestVal) {
      bestVal = val;
      bestIdx = i;
    }
  }

  out[outIdx] = f32(bestIdx);
}
`;

  const pipeline = getPipeline(
    ctx,
    `${opName}:${inputShape.join(",")}:${dim}:${keepdim}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params3(outSize, dimSize, dimStride));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, Math.ceil(outSize / WORKGROUP_SIZE));

  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Generate ternary shader for where(condition, x, y).
 * Returns x where condition != 0, else y.
 */
function ternaryWhereShader(
  indexShape: number[],
  condStrides: number[],
  xStrides: number[],
  yStrides: number[],
  condOffset: number,
  xOffset: number,
  yOffset: number,
): string {
  const indexing = buildBroadcastIndexing(indexShape, [
    condStrides,
    xStrides,
    yStrides,
  ]);
  return `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

${indexing.declarations}
const COND_OFFSET: u32 = ${condOffset}u;
const X_OFFSET: u32 = ${xOffset}u;
const Y_OFFSET: u32 = ${yOffset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
  let condVal = cond[offset0 + COND_OFFSET];
  let xVal = x[offset1 + X_OFFSET];
  let yVal = y[offset2 + Y_OFFSET];
  out[idx] = select(yVal, xVal, condVal != 0.0);
}
`;
}

/**
 * Broadcast three shapes to a common output shape.
 */
function broadcastThreeShapes(a: number[], b: number[], c: number[]): number[] {
  const outRank = Math.max(a.length, b.length, c.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    const cDim = c[c.length - 1 - i] ?? 1;
    // Check all pairs for broadcast compatibility
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    if (aDim !== cDim && aDim !== 1 && cDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    if (bDim !== cDim && bDim !== 1 && cDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim, cDim);
  }
  return out;
}

/**
 * where(condition, x, y): returns x where condition is true (non-zero), else y.
 */
function where(
  condition: BackendTensor,
  x: BackendTensor,
  y: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const condTensor = condition as WebGPUTensor;
  const xTensor = x as WebGPUTensor;
  const yTensor = y as WebGPUTensor;

  const outShape = broadcastThreeShapes(
    condTensor.shape,
    xTensor.shape,
    yTensor.shape,
  );
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);

  if (outSize === 0) {
    throw new Error("webgpu where does not support empty tensors yet");
  }

  const ctx = requireContext();

  // Check if chunking is needed for large contiguous tensors
  const bytesPerElement = 4; // f32
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const outSizeBytes = outSize * bytesPerElement;

  // Use chunked dispatch when output exceeds binding limit and inputs are chunkable:
  // Each input must be either scalar (0-d, stride=0 broadcast) or contiguous.
  if (outSizeBytes > maxBindingSize) {
    const condIsScalar = condTensor.size <= 1;
    const xIsScalar = xTensor.size <= 1;
    const yIsScalar = yTensor.size <= 1;
    const condChunkable = condIsScalar || condTensor.isContiguous;
    const xChunkable = xIsScalar || xTensor.isContiguous;
    const yChunkable = yIsScalar || yTensor.isContiguous;

    if (condChunkable && xChunkable && yChunkable) {
      return whereChunked(condTensor, xTensor, yTensor, outShape, outSize, options);
    }
  }

  return whereDirect(condTensor, xTensor, yTensor, outShape, indexShape, outSize, options);
}

/**
 * Direct (non-chunked) where dispatch using broadcast indexing.
 */
function whereDirect(
  condTensor: WebGPUTensor,
  xTensor: WebGPUTensor,
  yTensor: WebGPUTensor,
  outShape: number[],
  indexShape: number[],
  outSize: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const condStrides = computeEffectiveBroadcastStrides(condTensor, indexShape);
  const xStrides = computeEffectiveBroadcastStrides(xTensor, indexShape);
  const yStrides = computeEffectiveBroadcastStrides(yTensor, indexShape);

  const code = ternaryWhereShader(
    indexShape, condStrides, xStrides, yStrides,
    condTensor.offset, xTensor.offset, yTensor.offset,
  );
  const key = `where:${indexShape.join("x")}:${condStrides.join(",")}:${xStrides.join(",")}:${yStrides.join(",")}:${condTensor.offset}:${xTensor.offset}:${yTensor.offset}`;

  const providedOut = options?.outBuffer && options.outBuffer.size >= outSize * 4
    ? options.outBuffer
    : undefined;

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    outputSizeBytes: outSize * 4,
    params: params1(outSize),
    outBuffer: providedOut,
    dispatchX: Math.ceil(outSize / WORKGROUP_SIZE),
  });

  return createTensor(outShape, outBuffer);
}

/**
 * Chunked where dispatch for large contiguous tensors.
 * Each input is either scalar (bound fully each chunk) or contiguous (bound as sub-range).
 */
function whereChunked(
  condTensor: WebGPUTensor,
  xTensor: WebGPUTensor,
  yTensor: WebGPUTensor,
  outShape: number[],
  outSize: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = 4; // f32

  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Calculate chunk size
  const elementsPerAlignment = minAlignment / bytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const numChunks = Math.ceil(outSize / elementsPerChunk);
  const outSizeBytes = outSize * bytesPerElement;

  // Create output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSizeBytes,
    [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    options?.outBuffer,
  );

  const condIsScalar = condTensor.size <= 1;
  const xIsScalar = xTensor.size <= 1;
  const yIsScalar = yTensor.size <= 1;

  // Determine 2D dispatch dimensions for large chunks
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Build a flat chunked shader — no broadcast indexing needed since
  // scalar inputs are read at [0] and contiguous inputs are read at [idx].
  const condAccess = condIsScalar ? "cond[0]" : "cond[idx]";
  const xAccess = xIsScalar ? "x[0]" : "x[idx]";
  const yAccess = yIsScalar ? "y[0]" : "y[idx]";

  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `
struct Params {
  chunkSize: u32,
};

@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  let condVal = ${condAccess};
  let xVal = ${xAccess};
  let yVal = ${yAccess};
  out[idx] = select(yVal, xVal, condVal != 0.0);
}
`;

  const key = `whereChunked:${condIsScalar}:${xIsScalar}:${yIsScalar}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, outSize);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    const paramsBuffer = createUniformBuffer(ctx.device, chunkSize);

    // Scalar inputs: bind the full (small) buffer. Contiguous inputs: bind the chunk sub-range.
    const condBinding = condIsScalar
      ? { buffer: condTensor.buffer }
      : { buffer: condTensor.buffer, offset: chunkByteOffset, size: chunkByteSize };
    const xBinding = xIsScalar
      ? { buffer: xTensor.buffer }
      : { buffer: xTensor.buffer, offset: chunkByteOffset, size: chunkByteSize };
    const yBinding = yIsScalar
      ? { buffer: yTensor.buffer }
      : { buffer: yTensor.buffer, offset: chunkByteOffset, size: chunkByteSize };

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM) : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: condBinding },
        { binding: 1, resource: xBinding },
        { binding: 2, resource: yBinding },
        { binding: 3, resource: { buffer: outBuffer, offset: chunkByteOffset, size: chunkByteSize } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
    releaseUniformBuffer(paramsBuffer);
  }

  return createTensor(outShape, outBuffer);
}

/**
 * Generate shader for strided scatter copy.
 * Copies src values into base tensor at positions defined by view strides.
 */
function stridedScatterCopyShader(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
  gridSizeX?: number,
): string {
  const rank = viewShape.length;
  const viewSize = viewShape.reduce((a, b) => a * b, 1);
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Build coordinate calculation code
  let coordCode = "";
  let baseOffsetCode = `var baseOffset: u32 = ${viewOffset}u;\n`;
  let srcOffsetCode = `var srcOffset: u32 = ${srcOffset}u;\n`;

  for (let d = 0; d < rank; d++) {
    const shapeStride = viewShape.slice(d + 1).reduce((a, b) => a * b, 1);
    coordCode += `    let coord${d} = (remainder / ${shapeStride}u) % ${viewShape[d]}u;\n`;
    if (d < rank - 1) {
      coordCode += `    remainder = remainder % ${shapeStride}u;\n`;
    }
    baseOffsetCode += `    baseOffset += coord${d} * ${viewStrides[d]}u;\n`;
    srcOffsetCode += `    srcOffset += coord${d} * ${srcStrides[d]}u;\n`;
  }

  return `
struct Params {
  baseSize: u32,
  viewSize: u32,
};

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}

  // First pass: copy base to output (all threads that fit in baseSize)
  if (idx < params.baseSize) {
    out[idx] = base[idx];
  }

  workgroupBarrier();

  // Second pass: scatter src values into output at view positions
  if (idx < params.viewSize) {
    var remainder = idx;
${coordCode}
${baseOffsetCode}
${srcOffsetCode}
    out[baseOffset] = src[srcOffset];
  }
}
`;
}

/**
 * Copy src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
function stridedScatterCopy(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const baseTensor = base as WebGPUTensor;
  const srcTensor = src as WebGPUTensor;
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  if (baseSize === 0) {
    throw new Error("stridedScatterCopy: empty base tensor");
  }

  const ctx = requireContext();
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Check if any buffer exceeds max binding size
  const baseSizeBytes = baseSize * 4;
  const srcSizeBytes = srcTensor.size * 4;

  if (baseSizeBytes > maxBindingSize || srcSizeBytes > maxBindingSize) {
    // Check for simple full copy: viewSize == baseSize, offset == 0, both contiguous
    const isFullCopy = viewSize === baseSize && offset === 0;

    // Check if viewStrides are contiguous
    let viewContiguous = true;
    let expectedStride = 1;
    for (let d = viewShape.length - 1; d >= 0; d--) {
      if (viewStrides[d] !== expectedStride) {
        viewContiguous = false;
        break;
      }
      expectedStride *= viewShape[d];
    }

    // Check if src strides are contiguous
    const srcStrides = srcTensor.strides;
    let srcContiguous = srcTensor.isContiguous;
    if (!srcContiguous) {
      // Double-check with stride calculation
      srcContiguous = true;
      expectedStride = 1;
      for (let d = srcTensor.shape.length - 1; d >= 0; d--) {
        if (srcStrides[d] !== expectedStride) {
          srcContiguous = false;
          break;
        }
        expectedStride *= srcTensor.shape[d];
      }
    }

    if (isFullCopy && baseTensor.isContiguous && srcContiguous && viewContiguous) {
      // Fast path: simple chunked copy from src to output (no need to copy base first since all elements overwritten)
      return stridedScatterCopyChunkedSimple(
        baseTensor,
        srcTensor,
        maxBindingSize,
      );
    }

    // General chunked case for complex stride patterns - not yet implemented
    // For now, fall through to original implementation which will fail with validation error
    // This case is rare (non-contiguous large tensors)
  }

  return stridedScatterCopyDirect(baseTensor, srcTensor, options);
}

/**
 * Direct implementation of stridedScatterCopy for small tensors.
 */
function stridedScatterCopyDirect(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  const ctx = requireContext();

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const maxSize = Math.max(baseSize, viewSize);
  const totalWorkgroups = Math.ceil(maxSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Ensure base is contiguous for simplicity
  const contiguousBase = baseTensor.isContiguous
    ? baseTensor
    : ensureContiguous(baseTensor);

  // Compute src strides for reading
  const srcStrides = srcTensor.strides;
  const srcOffset = srcTensor.offset;

  const code = stridedScatterCopyShader(
    baseSize,
    viewShape,
    viewStrides,
    offset,
    srcStrides,
    srcOffset,
    use2D ? dispatch.gridSizeX : undefined,
  );

  const key = `stridedScatterCopy:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    baseSize * 4,
    [contiguousBase.buffer, srcTensor.buffer],
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(baseSize, viewSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [contiguousBase.buffer, srcTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) {
    bufferPool.decRef(contiguousBase.buffer);
    bufferPool.deferredDestroy(contiguousBase.buffer, contiguousBase.size * 4);
  }

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Chunked implementation for simple full contiguous copy (src -> output).
 * Used when both src and dest exceed buffer binding limit but are contiguous
 * and the copy covers all elements.
 */
function stridedScatterCopyChunkedSimple(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  maxBindingSize: number,
): BackendTensor {
  const ctx = requireContext();
  const totalElements = baseTensor.size;
  const totalBytes = totalElements * 4;

  // Calculate chunk size (in elements) that fits within binding limit
  // Also need to account for alignment requirements
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Elements per aligned chunk (must be multiple of alignment / 4 bytes)
  const elementsPerAlignment = minAlignment / 4;

  // Max elements that fit in binding size
  const maxElementsPerChunk = Math.floor(maxBindingSize / 4);

  // Round down to alignment boundary
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  // Create output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalBytes,
    [baseTensor.buffer, srcTensor.buffer],
  );

  // Determine if 2D dispatch is needed (chunks can be large)
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;

  // For 2D dispatch, compute grid dimensions
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;
  const gridSizeY = use2D ? Math.ceil(maxWorkgroups / gridSizeX) : 1;

  // Simple copy shader with optional 2D indexing
  const idxCompute = use2D
    ? `let localIdx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let localIdx = gid.x;`;

  const code = `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params {
  chunkSize: u32,
  totalSize: u32,
  chunkOffset: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (localIdx >= params.chunkSize) { return; }

  let globalIdx = params.chunkOffset + localIdx;
  if (globalIdx >= params.totalSize) { return; }

  out[localIdx] = src[localIdx];
}
`;

  const key = `stridedScatterCopyChunkedSimple:${WORKGROUP_SIZE}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  // Process each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * 4;
    const chunkByteSize = chunkSize * 4;

    const paramsBuffer = createParamsBuffer(ctx.device, params3(chunkSize, totalElements, chunkStart));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: srcTensor.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: outBuffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          },
        },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    // Calculate dispatch dimensions for this chunk
    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
    releaseParamsBuffer(paramsBuffer);
  }

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Generate shader for strided scatter add.
 * Adds src values into base tensor at positions defined by view strides.
 */
function stridedScatterAddShader(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
  gridSizeX?: number,
): string {
  const rank = viewShape.length;
  const viewSize = viewShape.reduce((a, b) => a * b, 1);
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Build coordinate calculation code
  let coordCode = "";
  let baseOffsetCode = `var baseOffset: u32 = ${viewOffset}u;\n`;
  let srcOffsetCode = `var srcOffset: u32 = ${srcOffset}u;\n`;

  for (let d = 0; d < rank; d++) {
    const shapeStride = viewShape.slice(d + 1).reduce((a, b) => a * b, 1);
    coordCode += `    let coord${d} = (remainder / ${shapeStride}u) % ${viewShape[d]}u;\n`;
    if (d < rank - 1) {
      coordCode += `    remainder = remainder % ${shapeStride}u;\n`;
    }
    baseOffsetCode += `    baseOffset += coord${d} * ${viewStrides[d]}u;\n`;
    srcOffsetCode += `    srcOffset += coord${d} * ${srcStrides[d]}u;\n`;
  }

  return `
struct Params {
  baseSize: u32,
  viewSize: u32,
};

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}

  // First pass: copy base to output
  if (idx < params.baseSize) {
    out[idx] = base[idx];
  }

  workgroupBarrier();

  // Second pass: add src values into output at view positions
  if (idx < params.viewSize) {
    var remainder = idx;
${coordCode}
${baseOffsetCode}
${srcOffsetCode}
    out[baseOffset] = out[baseOffset] + src[srcOffset];
  }
}
`;
}

/**
 * Add src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
function stridedScatterAdd(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const baseTensor = base as WebGPUTensor;
  const srcTensor = src as WebGPUTensor;
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  if (baseSize === 0) {
    throw new Error("stridedScatterAdd: empty base tensor");
  }

  const ctx = requireContext();
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Check if any buffer exceeds max binding size
  const baseSizeBytes = baseSize * 4;
  const srcSizeBytes = srcTensor.size * 4;

  if (baseSizeBytes > maxBindingSize || srcSizeBytes > maxBindingSize) {
    // Check for simple full add: viewSize == baseSize, offset == 0, both contiguous
    const isFullAdd = viewSize === baseSize && offset === 0;

    // Check if viewStrides are contiguous
    let viewContiguous = true;
    let expectedStride = 1;
    for (let d = viewShape.length - 1; d >= 0; d--) {
      if (viewStrides[d] !== expectedStride) {
        viewContiguous = false;
        break;
      }
      expectedStride *= viewShape[d];
    }

    // Check if src strides are contiguous
    const srcStrides = srcTensor.strides;
    let srcContiguous = srcTensor.isContiguous;
    if (!srcContiguous) {
      srcContiguous = true;
      expectedStride = 1;
      for (let d = srcTensor.shape.length - 1; d >= 0; d--) {
        if (srcStrides[d] !== expectedStride) {
          srcContiguous = false;
          break;
        }
        expectedStride *= srcTensor.shape[d];
      }
    }

    if (isFullAdd && baseTensor.isContiguous && srcContiguous && viewContiguous) {
      // Fast path: simple chunked add
      return stridedScatterAddChunkedSimple(
        baseTensor,
        srcTensor,
        maxBindingSize,
      );
    }

    // General chunked case - not yet implemented
  }

  return stridedScatterAddDirect(baseTensor, srcTensor, options);
}

/**
 * Direct implementation of stridedScatterAdd for small tensors.
 */
function stridedScatterAddDirect(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  const ctx = requireContext();

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const maxSize = Math.max(baseSize, viewSize);
  const totalWorkgroups = Math.ceil(maxSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Ensure base is contiguous for simplicity
  const contiguousBase = baseTensor.isContiguous
    ? baseTensor
    : ensureContiguous(baseTensor);

  // Compute src strides for reading
  const srcStrides = srcTensor.strides;
  const srcOffset = srcTensor.offset;

  const code = stridedScatterAddShader(
    baseSize,
    viewShape,
    viewStrides,
    offset,
    srcStrides,
    srcOffset,
    use2D ? dispatch.gridSizeX : undefined,
  );

  const key = `stridedScatterAdd:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    baseSize * 4,
    [contiguousBase.buffer, srcTensor.buffer],
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(baseSize, viewSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [contiguousBase.buffer, srcTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) {
    bufferPool.decRef(contiguousBase.buffer);
    bufferPool.deferredDestroy(contiguousBase.buffer, contiguousBase.size * 4);
  }

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Chunked implementation for simple full contiguous add (base + src -> output).
 * Used when both base and src exceed buffer binding limit but are contiguous
 * and the add covers all elements.
 */
function stridedScatterAddChunkedSimple(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  maxBindingSize: number,
): BackendTensor {
  const ctx = requireContext();
  const totalElements = baseTensor.size;
  const totalBytes = totalElements * 4;

  // Calculate chunk size (in elements) that fits within binding limit
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Elements per aligned chunk
  const elementsPerAlignment = minAlignment / 4;
  const maxElementsPerChunk = Math.floor(maxBindingSize / 4);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  // Create output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalBytes,
    [baseTensor.buffer, srcTensor.buffer],
  );

  // Determine if 2D dispatch is needed
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;

  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Simple add shader with optional 2D indexing
  const idxCompute = use2D
    ? `let localIdx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let localIdx = gid.x;`;

  const code = `
@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
  chunkSize: u32,
  totalSize: u32,
  chunkOffset: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (localIdx >= params.chunkSize) { return; }

  let globalIdx = params.chunkOffset + localIdx;
  if (globalIdx >= params.totalSize) { return; }

  out[localIdx] = base[localIdx] + src[localIdx];
}
`;

  const key = `stridedScatterAddChunkedSimple:${WORKGROUP_SIZE}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  // Process each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * 4;
    const chunkByteSize = chunkSize * 4;

    const paramsBuffer = createParamsBuffer(ctx.device, params3(chunkSize, totalElements, chunkStart));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: baseTensor.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: srcTensor.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: outBuffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          },
        },
        { binding: 3, resource: { buffer: paramsBuffer } },
      ],
    });

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
    releaseParamsBuffer(paramsBuffer);
  }

  return createTensor(baseTensor.shape, outBuffer);
}

// ============================================================================
// Fused Adam/AdamW Step
// ============================================================================

async function adamStep(
  grad: BackendTensor,
  param: BackendTensor,
  m: BackendTensor,
  v: BackendTensor,
  config: import("../types").AdamStepConfig,
): Promise<{ param: BackendTensor; m: BackendTensor; v: BackendTensor }> {
  // Suppress memory limit checks for the entire adam step.
  // During optimizer steps, old param/m/v buffers haven't been released yet
  // while new output buffers are allocated — temporary 2x peak is expected.
  gpuMemoryTracker.suppressLimitCheck();
  try {
    let _st = profileSubOpBegin();
    const gradT = ensureContiguous(grad as WebGPUTensor);
    const paramT = ensureContiguous(param as WebGPUTensor);
    const mT = ensureContiguous(m as WebGPUTensor);
    const vT = ensureContiguous(v as WebGPUTensor);
    profileSubOpEnd("adam.ensureContig", _st);

    // Evict stale f16 cache entry for the old param buffer before dispatch.
    // Use deferred destruction since the old f16 buffer may still be referenced
    // by cast() results from the forward pass in the same command encoder batch.
    _st = profileSubOpBegin();
    const oldF16 = f16WeightCache.get(paramT.buffer);
    if (oldF16) {
      bufferPool.deferredDestroy(oldF16, oldF16.size);
      f16WeightCache.delete(paramT.buffer);
    }
    profileSubOpEnd("adam.f16evict", _st);

    const bpe = (t: WebGPUTensor) => dtypeBytes(t.dtype);

    // Flush shared encoder before dispatching the in-place Adam kernel.
    // The forward/backward pass may have used param/m/v buffers as read-only
    // inputs in compute passes. The Adam kernel binds them as read_write,
    // which conflicts within the same command encoder synchronization scope.
    // Flushing ensures prior passes are submitted before the in-place writes.
    // In Adam batch mode, the caller already flushed once for the entire batch.
    if (!adamBatchMode) {
      flushSharedEncoder();
    }

    // Re-protect adam's input buffers in the write set.
    trackSharedEncoderWrite(paramT.buffer);
    trackSharedEncoderWrite(mT.buffer);
    trackSharedEncoderWrite(vT.buffer);
    trackSharedEncoderWrite(gradT.buffer);

    // WebGPU forbids binding the same buffer as read_write at multiple
    // binding points. De-alias input buffers if pool recycling caused
    // any to share the same GPUBuffer (common for same-size tensors).
    let paramBuf = paramT.buffer;
    let mBuf = mT.buffer;
    let vBuf = vT.buffer;
    const gradBuf = gradT.buffer;
    const bufSize = gradT.size * bpe(gradT);
    const copyToFresh = (src: GPUBuffer): GPUBuffer => {
      const dst = allocateOutputBuffer(bufSize);
      // Use the shared encoder for the copy so it's ordered before the
      // Adam compute pass in the same command buffer submission.
      const enc = getSharedEncoderInstance();
      if (enc) {
        (enc as any).copyBufferToBuffer(src, 0, dst, 0, bufSize);
      } else {
        const ctx2 = requireContext();
        const tmpEnc = (ctx2.device as any).createCommandEncoder();
        tmpEnc.copyBufferToBuffer(src, 0, dst, 0, bufSize);
        (ctx2.queue as any).submit([tmpEnc.finish()]);
      }
      return dst;
    };
    if (paramBuf === gradBuf || paramBuf === mBuf || paramBuf === vBuf) {
      paramBuf = copyToFresh(paramBuf);
    }
    if (mBuf === gradBuf || mBuf === vBuf) {
      mBuf = copyToFresh(mBuf);
    }
    if (vBuf === gradBuf) {
      vBuf = copyToFresh(vBuf);
    }

    // If any copies were made, flush again so the copy commands are submitted
    // before the Adam kernel binds the original buffers as read_write.
    if (paramBuf !== paramT.buffer || mBuf !== mT.buffer || vBuf !== vT.buffer) {
      flushSharedEncoder();
    }

    const emitF16 = config.emitF16 ?? false;
    const infFlagBuf = (config.infFlagBuffer as any) ?? null;
    const numElements = gradT.size;

    const result = dispatchAdamStepKernel(
      gradBuf,
      paramBuf,
      mBuf,
      vBuf,
      numElements,
      config,
      emitF16,
      infFlagBuf,
    );

    // Post-dispatch flush only needed outside shared encoder mode.
    // In shared encoder mode, all buffer releases are deferred (deferredDestroy for grad,
    // sharedEncoderDeferredUniformBuffers for config, noop'd destroy for param/m/v).
    // No buffer destruction occurs during plan execution.
    if (!isSharedEncoderActive()) {
      flushSharedEncoder();
    }

    // Destroy contiguous copy for grad only (read-only input, safe to free).
    if (gradT !== (grad as WebGPUTensor)) {
      bufferPool.decRef(gradT.buffer);
      bufferPool.deferredDestroy(gradT.buffer, gradT.size * bpe(gradT));
    }

    // Cache the f16 param buffer (keyed by the param buffer, same as before)
    if (result.paramF16Buffer) {
      f16WeightCache.set(result.paramBuffer, result.paramF16Buffer);
    }

    // In-place: the kernel wrote to the input buffers (or de-aliased copies).
    // Create new owning WebGPUTensor wrappers for the result. Replace the
    // input tensors' destroy() with a no-op so that when the engine disposes
    // the old storage handles, the buffer won't be released to the pool
    // (the result tensors now own it). This prevents double-free.
    _st = profileSubOpBegin();
    // Explicitly decRef before noop-ing destroy — ownership transfers to result tensors.
    if (paramT.ownsBuffer) bufferPool.decRef(paramT.buffer);
    if (mT.ownsBuffer) bufferPool.decRef(mT.buffer);
    if (vT.ownsBuffer) bufferPool.decRef(vT.buffer);
    const noop = () => {};
    paramT.destroy = noop;
    mT.destroy = noop;
    vT.destroy = noop;
    const ret = {
      param: createTensor(paramT.shape, result.paramBuffer, undefined, 0, paramT.dtype),
      m: createTensor(mT.shape, result.mBuffer, undefined, 0, mT.dtype),
      v: createTensor(vT.shape, result.vBuffer, undefined, 0, vT.dtype),
    };
    profileSubOpEnd("adam.createTensor", _st);
    return ret;
  } finally {
    gpuMemoryTracker.unsuppressLimitCheck();
  }
}

// ============================================================================
// Fused Unscale + Inf-Check (GradScaler)
// ============================================================================

function unscaleGrad(
  grad: BackendTensor,
  invScale: number,
  infFlagBuffer: unknown,
): BackendTensor {
  const gradT = ensureContiguous(grad as WebGPUTensor);
  const numElements = gradT.size;
  const result = dispatchUnscaleGradKernel(
    gradT.buffer,
    numElements,
    invScale,
    infFlagBuffer as any,
  );
  // Destroy contiguous copy if one was created (deferred for GPU fence)
  if (gradT !== (grad as WebGPUTensor)) {
    bufferPool.decRef(gradT.buffer);
    bufferPool.deferredDestroy(gradT.buffer, gradT.size * dtypeBytes(gradT.dtype));
  }
  return createTensor(gradT.shape, result.gradOutBuffer);
}

function createInfCountBuffer(): unknown {
  return allocateInfFlagBuffer();
}

async function readAndDestroyInfCount(buffer: unknown): Promise<number> {
  return readInfFlag(buffer as any);
}

// ============================================================================
// Fused Cross-Entropy (Forward + Backward)
// ============================================================================

function fusedCrossEntropyForward(
  logits: BackendTensor,
  targets: BackendTensor,
  config: import("../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = ensureContiguous(logits as WebGPUTensor);
  const targetsT = ensureContiguous(targets as WebGPUTensor);
  const outBuf = dispatchCEForwardKernel(
    logitsT.buffer, targetsT.buffer, config.batchSize, config.vocabSize,
  );
  return createTensor([config.batchSize], outBuf, undefined, 0, "f32");
}

function fusedCrossEntropyBackward(
  logits: BackendTensor,
  targets: BackendTensor,
  gradOutput: BackendTensor,
  config: import("../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = ensureContiguous(logits as WebGPUTensor);
  const targetsT = ensureContiguous(targets as WebGPUTensor);
  const gradT = ensureContiguous(gradOutput as WebGPUTensor);
  const outBuf = dispatchCEBackwardKernel(
    logitsT.buffer, targetsT.buffer, gradT.buffer,
    config.batchSize, config.vocabSize,
  );
  return createTensor([config.batchSize, config.vocabSize], outBuf, undefined, 0, "f32");
}

// ============================================================================
// Fused LayerNorm (Forward + Backward)
// ============================================================================

function fusedLayerNormForward(
  x: BackendTensor,
  weight: BackendTensor,
  bias: BackendTensor,
  config: import("../types").FusedLayerNormConfig,
): BackendTensor {
  const xT = ensureContiguous(x as WebGPUTensor);
  const weightT = ensureContiguous(weight as WebGPUTensor);
  const biasT = ensureContiguous(bias as WebGPUTensor);
  const outBuf = dispatchLNForwardKernel(
    xT.buffer, weightT.buffer, biasT.buffer,
    config.numRows, config.featureDim, config.eps,
  );
  const outShape: number[] = [];
  // Reconstruct shape: [...batch_dims, featureDim] = same as input
  for (let i = 0; i < xT.shape.length; i++) outShape.push(xT.shape[i]);
  return createTensor(outShape, outBuf, undefined, 0, "f32");
}

function fusedLayerNormBackwardGradX(
  gradOutput: BackendTensor,
  x: BackendTensor,
  weight: BackendTensor,
  config: import("../types").FusedLayerNormConfig,
): BackendTensor {
  const gradT = ensureContiguous(gradOutput as WebGPUTensor);
  const xT = ensureContiguous(x as WebGPUTensor);
  const weightT = ensureContiguous(weight as WebGPUTensor);

  const gradXBuf = dispatchLNBwdGradXKernel(
    gradT.buffer, xT.buffer, weightT.buffer,
    config.numRows, config.featureDim, config.eps,
  );

  const gradXShape: number[] = [];
  for (let i = 0; i < xT.shape.length; i++) gradXShape.push(xT.shape[i]);
  return createTensor(gradXShape, gradXBuf, undefined, 0, "f32");
}

function fusedLayerNormBackwardGradWeightBias(
  gradOutput: BackendTensor,
  x: BackendTensor,
  config: import("../types").FusedLayerNormConfig,
): { gradWeight: BackendTensor; gradBias: BackendTensor } {
  const gradT = ensureContiguous(gradOutput as WebGPUTensor);
  const xT = ensureContiguous(x as WebGPUTensor);
  const result = dispatchLNBwdGradWBKernel(
    gradT.buffer, xT.buffer,
    config.numRows, config.featureDim, config.eps,
  );
  const shape = [config.featureDim];
  return {
    gradWeight: createTensor(shape, result.gradWeightBuffer, undefined, 0, "f32", true),
    gradBias: createTensor(shape, result.gradBiasBuffer, undefined, 0, "f32", true),
  };
}

async function read(a: BackendTensor): Promise<number[]> {
  const ctx = requireContext();
  let tensor = a as WebGPUTensor;
  if (tensor.size === 0) {
    return [];
  }

  // Flush shared encoder before readback — all prior GPU work must be submitted.
  if (sharedEncoder) {
    flushSharedEncoder();
  }

  // Materialize non-contiguous tensors before reading
  const originalTensor = tensor;
  if (!tensor.isContiguous) {
    tensor = ensureContiguous(tensor);
  }

  const bytesPerElement = dtypeBytes(tensor.dtype);
  const totalBytes = tensor.size * bytesPerElement;
  const alignedBytes = alignBufferSize(totalBytes);
  const readBuffer = createTrackedBuffer(ctx.device, {
    size: alignedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  // Use the shared encoder for the copy if active, otherwise create a standalone one
  if (sharedEncoderInstance) {
    sharedEncoderInstance.copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
    // Flush to submit the copy command
    flushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
    profileApiCall("queue.submit", () => ctx.queue.submit([encoder.finish()]));
    gpuSubmitCount++;
  }
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  // GPU work is complete - safe to flush pending buffers
  bufferPool.flushPendingToAvailable();
  await readBuffer.mapAsync(GPUMapMode.READ);
  const mapped = readBuffer.getMappedRange();

  // Create appropriate TypedArray view based on dtype
  // Slice only the actual data bytes to exclude padding
  let result: number[];
  switch (tensor.dtype) {
    case "i32":
      result = Array.from(new Int32Array(mapped.slice(0, totalBytes)));
      break;
    case "u32":
      result = Array.from(new Uint32Array(mapped.slice(0, totalBytes)));
      break;
    case "f16":
      // Convert f16 (Uint16Array) back to f32 numbers
      result = f16ArrayToF32Array(new Uint16Array(mapped.slice(0, totalBytes)));
      break;
    case "f32":
    default:
      result = Array.from(new Float32Array(mapped.slice(0, totalBytes)));
      break;
  }
  readBuffer.unmap();

  // Destroy staging buffer to prevent memory leaks
  bufferPool.deferredDestroy(readBuffer, (readBuffer as any).size ?? alignedBytes);

  // Destroy contiguous copy if we created one
  if (tensor !== originalTensor && tensor.destroy) {
    tensor.destroy();
  }

  return result;
}

/**
 * Wait for all submitted GPU work to complete.
 * Call this before destroying buffers to ensure GPU is done using them.
 */
export async function waitForGPU(): Promise<void> {
  const ctx = context;
  if (!ctx) return;
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  // GPU work is complete - safe to flush pending buffers
  bufferPool.flushPendingToAvailable();
}

// ============================================================================
// In-place Scalar Operations (for optimizer updates)
// ============================================================================

/**
 * Multiply tensor by a scalar in-place (for gradient unscaling).
 * This is a simple in-place multiplication without NaN checking.
 */
function mulScalarInPlace(tensor: BackendTensor, scalar: number): void {
  const ctx = requireContext();
  const a = tensor as WebGPUTensor;
  const size = a.size;

  // Pack mixed f32 + u32 params
  const fillParamsData = new ArrayBuffer(8);
  new Float32Array(fillParamsData, 0, 1)[0] = scalar;
  new Uint32Array(fillParamsData, 4, 1)[0] = size;
  const uniformBuffer = createParamsBuffer(ctx.device, new Uint32Array(fillParamsData));

  const code = `
struct Params {
  scalar: f32,
  size: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) {
    return;
  }
  data[idx] = data[idx] * params.scalar;
}
`;

  const key = `mul_scalar_inplace:${size}`;
  const pipeline = getPipeline(ctx, key, code);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [a.buffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, Math.ceil(size / WORKGROUP_SIZE));

  releaseParamsBuffer(uniformBuffer);
}

export const webgpuBackend: Backend & {
  waitForGPU: typeof waitForGPU;
  mulScalarInPlace: typeof mulScalarInPlace;
  dispatchFusedKernel: typeof dispatchFusedKernel;
  beginStep: typeof beginStep;
  endStep: typeof endStep;
  device: GPUDevice | null;
} = {
  name: "webgpu",
  waitForGPU,
  // Expose device for fusion dispatch (§15)
  get device() {
    return context?.device ?? null;
  },
  // Fusion dispatch (§15.1, §15.2, §15.3)
  dispatchFusedKernel,
  ops: {
    tensorFromArray,
    zeros,
    full,
    arange,
    tril,
    triu,
    add,
    sub,
    div,
    mul,
    matmul,
    sqrt,
    relu,
    exp,
    log,
    neg,
    abs,
    tanh,
    sigmoid,
    gelu,
    silu,
    isfinite,
    expand,
    reshape,
    transpose,
    permute,
    narrow,
    narrowBackward,
    contiguous,
    cast,
    gather,
    scatterAdd,
    sum,
    max,
    mean,
    argmax,
    argmin,
    gt,
    lt,
    ge,
    le,
    eq,
    ne,
    where,
    stridedScatterCopy,
    stridedScatterAdd,
    adamStep,
    unscaleGrad,
    fusedCrossEntropyForward,
    fusedCrossEntropyBackward,
    fusedLayerNormForward,
    fusedLayerNormBackwardGradX,
    fusedLayerNormBackwardGradWeightBias,
    createInfCountBuffer,
    readAndDestroyInfCount,
    read,
  },
  mulScalarInPlace,
  beginStep,
  endStep,
  // Pretune matmul shapes for autotuning (used by compile with autotune: true)
  async pretuneMatmulShapes(shapes: Array<[number, number, number]>): Promise<void> {
    const ctx = context;
    if (!ctx) {
      return; // WebGPU not initialized
    }
    await pretuneShapes(ctx.device, ctx.queue, shapes, "f32");
  },
};
