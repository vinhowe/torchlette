/**
 * Buffer Pool for GPU Buffer Reuse (spec section 14).
 *
 * Contains the SimpleBufferPool class, the global bufferPool singleton,
 * pool accessor functions, the deferred GPU fence mechanism, and buffer
 * lifecycle helpers (deferredDestroy, acquirePooledBuffer).
 *
 * Cross-module state (activeBatch, sharedEncoderActive, gpuContext,
 * arenaBufferSet) is accessed via direct imports from webgpu-state.ts
 * (a zero-dependency leaf module).
 */

import { getSizeClass, getSizeForClass } from "../../graph/lifetime-analysis";
import type { DType } from "../types";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { STORAGE_BUFFER_USAGE } from "./gpu-types";
import { gpuMemoryTracker } from "./memory-tracker";
import { isProfilingEnabled } from "./profiler";
import { dtypeBytes } from "./shape-utils";
import {
  activeBatch,
  arenaBufferSet,
  gpuContext,
  sharedEncoderActive,
} from "./webgpu-state";

// ============================================================================
// Buffer Pool for GPU Buffer Reuse (section 14)
// ============================================================================

/**
 * Buffer pool with fence integration per spec section 14.
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
  private windowDemand: Array<
    Map<number, { acquires: number; releases: number }>
  > = [];
  private reservation: Map<number, number> | null = null;

  // Total byte budget for pooled + pending buffers.
  // Default: no limit (Infinity). Like PyTorch's CUDA caching allocator, the
  // pool grows as needed and relies on the memory tracker's eviction mechanism
  // (in createTrackedBuffer) for pressure relief. Users can set an explicit
  // cap via setMaxPoolBytes() or the TORCHLETTE_POOL_BUDGET_MB env var.
  private maxPoolBytes = Infinity;

  // Fence integration (section 14): buffers pending GPU completion
  private pendingRelease: Array<{
    buffer: GPUBuffer;
    sizeClass: number;
    size: number;
  }> = [];
  private pendingReleaseBytes = 0;
  // Reference counting: track how many owning tensors reference each GPUBuffer.
  // Only owning tensors (ownsBuffer=true) participate. When refcount drops to 0,
  // the buffer is eligible for pool promotion or reuse from pendingRelease.
  private bufferLiveCount = new Map<GPUBuffer, number>();

  // Buffers released by liveness-based early release within a plan. These are
  // provably safe to reuse during the shared encoder — liveness guarantees no
  // future action in the plan reads them. Within a single WebGPU queue,
  // dispatches execute sequentially, so the releasing dispatch has completed
  // by the time the acquiring dispatch runs.
  private livenessSafeBuffers = new Set<GPUBuffer>();

  // Stats for understanding buffer reuse patterns
  private acquireFromPool = 0;
  private acquireFromPending = 0;
  private acquireNew = 0;
  private releaseToPool = 0;
  private releaseToDestroy = 0;

  getDetailedStats() {
    return {
      acquireFromPool: this.acquireFromPool,
      acquireFromPending: this.acquireFromPending,
      acquireNew: this.acquireNew,
      releaseToPool: this.releaseToPool,
      releaseToDestroy: this.releaseToDestroy,
      pendingReleaseCount: this.pendingRelease.length,
      pendingReleaseBytes: this.pendingReleaseBytes,
    };
  }

  resetDetailedStats(): void {
    this.acquireFromPool = 0;
    this.acquireFromPending = 0;
    this.acquireNew = 0;
    this.releaseToPool = 0;
    this.releaseToDestroy = 0;
  }

  /** Get or create a pool bucket for a size class. */
  private getBucket(sizeClass: number): GPUBuffer[] {
    let bucket = this.pool.get(sizeClass);
    if (!bucket) {
      bucket = [];
      this.pool.set(sizeClass, bucket);
    }
    return bucket;
  }

  /** Increment reference count for a buffer (called when an owning tensor is created). */
  incRef(buffer: GPUBuffer): void {
    this.bufferLiveCount.set(
      buffer,
      (this.bufferLiveCount.get(buffer) ?? 0) + 1,
    );
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

  /** Mark a buffer as liveness-safe for immediate reuse during shared encoder. */
  markLivenessSafe(buffer: GPUBuffer): void {
    this.livenessSafeBuffers.add(buffer);
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
    this.recordWindowDemand(sizeClass, "acquires");
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
      const buffer = pooledBuffers.pop() as GPUBuffer;
      const actualSize = getSizeForClass(sizeClass);
      this.pooledBytes -= actualSize;
      this.reuseCount++;
      this.acquireFromPool++;
      this.recordWindowDemand(sizeClass, "acquires");
      // Track that this buffer is from pool for release
      this.pooledBufferSet.add(buffer);
      // Re-track allocation (was deallocated when released to pool)
      gpuMemoryTracker.trackAllocation(buffer, actualSize);
      if (this.debugTrace) {
        console.log(
          `[pool] acquire from POOL: ${(actualSize / 1e6).toFixed(2)} MB`,
        );
      }
      return buffer;
    }

    // Then check pendingRelease for same-execution reuse.
    // This enables actual memory savings within a single execution.
    // Skip when batching — pending buffers may still be referenced by
    // collected command buffers that haven't been submitted yet.
    // Also skip when a deferred fence is outstanding — those pending buffers
    // are from a previous step and the GPU may still be using them.
    //
    // During shared encoder: only allow liveness-safe buffers (released by
    // the liveness analysis within the current plan — provably dead).
    if (!activeBatch && !fenceState.pendingFencePromise) {
      const livenessOnly = sharedEncoderActive;
      const pendingIdx = this.pendingRelease.findIndex(
        (p) =>
          p.sizeClass === sizeClass &&
          !this.bufferLiveCount.has(p.buffer) &&
          (!livenessOnly || this.livenessSafeBuffers.has(p.buffer)),
      );
      if (pendingIdx !== -1) {
        const { buffer, size } = this.pendingRelease.splice(pendingIdx, 1)[0];
        this.pendingReleaseBytes -= size;
        this.reuseCount++;
        this.acquireFromPending++;
        this.recordWindowDemand(sizeClass, "acquires");
        this.pooledBufferSet.add(buffer);
        this.livenessSafeBuffers.delete(buffer);
        // Re-track allocation (was deallocated when added to pending)
        gpuMemoryTracker.trackAllocation(buffer, size);
        if (this.debugTrace) {
          console.log(
            `[pool] acquire from PENDING${livenessOnly ? " (liveness-safe)" : ""}: ${(size / 1e6).toFixed(2)} MB (${this.pendingRelease.length} remaining)`,
          );
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
    this.recordWindowDemand(sc, "acquires");
    if (this.debugTrace) {
      console.log(`[pool] NEW allocation: ${(sizeBytes / 1e6).toFixed(2)} MB`);
    }
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

  private recordWindowDemand(
    sizeClass: number,
    field: "acquires" | "releases",
  ): void {
    if (!this.windowTracking) return;
    const wm = this.windowDemand[this.currentWindowId];
    if (!wm) return;
    const e = wm.get(sizeClass) ?? { acquires: 0, releases: 0 };
    e[field]++;
    wm.set(sizeClass, e);
  }

  private computeReservation(): void {
    const allSc = new Set<number>();
    for (const wm of this.windowDemand) {
      for (const sc of wm.keys()) allSc.add(sc);
    }

    const reservation = new Map<number, number>();
    for (const sc of allSc) {
      let cumAcq = 0,
        cumRel = 0,
        maxDeficit = 0;
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
    // Iterate either the window-based reservation (step 1+) or the
    // new-allocs-by-class map (step 0 prewarm fallback).
    const source: Iterable<[number, number]> =
      this.reservation ?? this.newAllocsByClass;

    for (const [sizeClass, needed] of source) {
      const size = getSizeForClass(sizeClass);
      const deficit = this.reservation
        ? needed - (this.pool.get(sizeClass)?.length ?? 0)
        : needed;
      if (deficit <= 0) continue;

      for (let i = 0; i < deficit; i++) {
        if (
          this.pooledBytes + this.pendingReleaseBytes + size >
          this.maxPoolBytes
        )
          break;
        const buf = device.createBuffer({ size, usage: STORAGE_BUFFER_USAGE });
        this.getBucket(sizeClass).push(buf);
        this.pooledBytes += size;
      }
    }
    this.newAllocsByClass.clear();
  }

  /**
   * Release a buffer back to the pool for reuse.
   * Per spec section 14, the buffer goes to a pending queue first and only becomes
   * available after GPU work completes (fence signaled).
   * Returns true if the buffer was queued for pooling, false if it should be destroyed.
   */
  release(buffer: GPUBuffer, sizeBytes: number, usage: number): boolean {
    if (!this.enabled) return false;

    // Only pool storage buffers with compatible usage
    if (usage !== STORAGE_BUFFER_USAGE) {
      this.releaseToDestroy++;
      if (this.debugTrace) {
        console.log(
          `[pool] release INCOMPATIBLE (wrong usage): ${(sizeBytes / 1e6).toFixed(2)} MB`,
        );
      }
      return false;
    }

    const sizeClass = getSizeClass(sizeBytes);
    const actualSize = getSizeForClass(sizeClass);

    // Remove from active tracking
    this.pooledBufferSet.delete(buffer);

    // Check total byte budget: pooledBytes tracks main pool, pendingReleaseBytes tracks pending
    if (
      this.pooledBytes + this.pendingReleaseBytes + actualSize >
      this.maxPoolBytes
    ) {
      this.releaseToDestroy++;
      if (this.debugTrace) {
        console.log(
          `[pool] release BUDGET (${((this.pooledBytes + this.pendingReleaseBytes) / 1e6).toFixed(1)}MB + ${(actualSize / 1e6).toFixed(2)}MB > ${(this.maxPoolBytes / 1e6).toFixed(0)}MB): ${(actualSize / 1e6).toFixed(2)} MB`,
        );
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
      console.log(
        `[pool] release to PENDING: ${(actualSize / 1e6).toFixed(2)} MB (now ${this.pendingRelease.length} pending)`,
      );
    }

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
    // Track deallocation immediately - the memory is now "freeable"
    gpuMemoryTracker.trackDeallocation(buffer);
    // When batching or shared encoder active, defer until after submit
    if (activeBatch) {
      activeBatch.deferredDestroyBuffers.push(buffer);
      return;
    }
    this.pendingDestroy.push({ buffer, size });
  }

  /**
   * Queue an untracked buffer for deferred destruction.
   * Use for tiny params buffers that are not in the memory tracker.
   */
  deferredDestroyUntracked(buffer: GPUBuffer): void {
    // When batching, command buffers haven't been submitted yet.
    if (activeBatch) {
      activeBatch.deferredDestroyBuffers.push(buffer);
      return;
    }
    this.pendingDestroy.push({ buffer, size: 0 });
  }

  /**
   * Move pending-release buffers into the available pool.
   * Always safe to call — these buffers are reusable after WebGPU submission ordering.
   *
   * NOTE: This method is accessed directly by awaitDeferredFence() in this module,
   * so it cannot be private despite being an implementation detail.
   */
  flushPendingToPool(): void {
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
      this.livenessSafeBuffers.delete(entry.buffer);
      this.getBucket(entry.sizeClass).push(entry.buffer);
      this.pooledBytes += entry.size;
      // Record at promotion time (not release time) so the reservation formula
      // only counts buffers that actually become pool-available.
      this.recordWindowDemand(entry.sizeClass, "releases");
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
    this.getBucket(sizeClass).push(buffer);
    this.pooledBytes += getSizeForClass(sizeClass);
  }

  /**
   * Record a pool acquire for window-demand tracking (used by pre-pin path).
   */
  recordAcquireForPin(sizeClass: number): void {
    this.reuseCount++;
    this.acquireFromPool++;
    this.recordWindowDemand(sizeClass, "acquires");
  }

  /** Debug: dump pool size-class breakdown. Keys are log2(bytes) classes. */
  debugDumpSizeClasses(): void {
    const classes: Array<[number, number, number]> = []; // [sizeClass, count, bytes]
    for (const [sizeClass, buffers] of this.pool) {
      const bufSize = getSizeForClass(sizeClass);
      classes.push([sizeClass, buffers.length, bufSize * buffers.length]);
    }
    classes.sort((a, b) => b[2] - a[2]);
    let totalBytes = 0;
    let totalCount = 0;
    for (const [, count, bytes] of classes) {
      totalBytes += bytes;
      totalCount += count;
    }
    console.log("Pool size-class breakdown (top 20):");
    for (const [sizeClass, count, bytes] of classes.slice(0, 20)) {
      const bufSize = getSizeForClass(sizeClass);
      const sizeStr =
        bufSize >= 1024 * 1024
          ? `${(bufSize / 1024 / 1024).toFixed(0)}MB`
          : bufSize >= 1024
            ? `${(bufSize / 1024).toFixed(0)}KB`
            : `${bufSize}B`;
      console.log(
        `  ${sizeStr.padStart(10)} × ${String(count).padStart(5)} = ${(bytes / 1024 / 1024).toFixed(1).padStart(8)}MB (${((bytes / totalBytes) * 100).toFixed(1)}%)`,
      );
    }
    console.log(
      `Total: ${totalCount} buffers, ${(totalBytes / 1024 / 1024).toFixed(0)}MB across ${classes.length} size classes`,
    );
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
    pendingRelease: number;
    pendingDestroy: number;
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
    this.livenessSafeBuffers.clear();
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
        const buffer = buffers.pop() as GPUBuffer;
        this.pooledBufferSet.delete(buffer);
        this.pooledBytes -= sizePerBuffer;
        // Track deallocation and destroy the buffer to actually free GPU memory.
        // These buffers have already passed the GPU fence (promoted from pending
        // to free pool), so they're safe to destroy.
        gpuMemoryTracker.trackDeallocation(buffer);
        try {
          buffer.destroy();
        } catch {
          /* already destroyed */
        }
        bytesFreed += sizePerBuffer;
      }

      if (buffers.length === 0) {
        this.pool.delete(sizeClass);
      }

      if (bytesFreed >= bytesNeeded) break;
    }

    return bytesFreed;
  }

  /** Evict ALL pooled buffers, freeing GPU memory. */
  evictAll(): number {
    return this.evictBuffers(Infinity);
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
    this.flushPendingToPool();
  }

  /**
   * Get bytes in pending queues (not yet safe to destroy).
   */
  getPendingBytes(): number {
    let destroyBytes = 0;
    for (const p of this.pendingDestroy) destroyBytes += p.size;
    return this.pendingReleaseBytes + destroyBytes;
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

// ============================================================================
// Global buffer pool instance
// ============================================================================

/** Global buffer pool instance */
const bufferPool = new SimpleBufferPool();
export { bufferPool };

// ============================================================================
// Pool accessor functions
// ============================================================================

/**
 * Get buffer pool statistics.
 */
export function getBufferPoolStats(): ReturnType<SimpleBufferPool["stats"]> {
  return bufferPool.stats();
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
 * Flush pending buffers to make them immediately available for reuse.
 *
 * Call this when GPU work is known to be complete (e.g., at the start of
 * backward pass). This enables checkpoint memory savings by making buffers
 * available before the async onSubmittedWorkDone() callback runs.
 */
export function flushBufferPool(): void {
  bufferPool.flushPendingToAvailable();
  bufferPool.sortPoolBuckets(); // deterministic acquire order for bind group cache
  bufferPool.beginWindow(); // Advance window counter for demand tracking
}

/** Destroy all buffers in the pending-destroy queue. */
export function destroyPendingGPUBuffers(): void {
  bufferPool.destroyPendingBuffers();
}

/** Evict all pooled buffers to free GPU memory. Call after GPU fence. */
export function evictAllPoolBuffers(): number {
  return bufferPool.evictAll();
}

/**
 * Release a contiguous copy's buffer if one was created.
 * Common pattern: `if (copy !== original) destroyCopy(copy)`.
 */
export function destroyCopy(tensor: {
  buffer: GPUBuffer;
  size: number;
  dtype: DType;
}): void {
  const bytes = tensor.size * dtypeBytes(tensor.dtype);
  bufferPool.decRef(tensor.buffer);
  bufferPool.deferredDestroy(tensor.buffer, bytes);
}

// ============================================================================
// Deferred GPU Fence (Phase 2: Async markStep)
// ============================================================================

/**
 * Deferred GPU fence state — groups the pending fence promise, pending-release
 * flag, and profiling fence buffer into a single typed object for debuggability
 * and explicit reset.
 */
interface FenceState {
  /** Pending fence promise from a previous markStep. */
  pendingFencePromise: Promise<void> | null;
  /** Whether to flush pending-release buffers after fence resolves. */
  deferredPendingRelease: boolean;
  /** Persistent fence buffer for the writeBuffer+mapAsync fence workaround. */
  profilingFenceBuffer: GPUBuffer | null;
}

const fenceState: FenceState = {
  pendingFencePromise: null,
  deferredPendingRelease: false,
  profilingFenceBuffer: null,
};

const profilingFenceData = new Uint8Array([1, 2, 3, 4]);

/**
 * Issue a GPU fence without awaiting it. The fence promise is stored
 * and will be awaited at the start of the next markStep.
 *
 * V100/Dawn workaround: when timestamp-query is enabled (profiling mode),
 * onSubmittedWorkDone deadlocks after backward pass dispatches. Use
 * writeBuffer+mapAsync instead — on Dawn/Vulkan, MAP_READ buffers are
 * host-visible so writeBuffer is a CPU memcpy and mapAsync resolves
 * immediately. This is not a real GPU fence, but works because arena
 * buffers are stable across steps (no actual buffer recycling needed).
 */
export function issueDeferredFence(): void {
  const ctx = gpuContext;
  if (!ctx) return;

  if (isProfilingEnabled()) {
    // Profiling mode: use writeBuffer+mapAsync (CPU-only, no deadlock)
    if (!fenceState.profilingFenceBuffer) {
      fenceState.profilingFenceBuffer = ctx.device.createBuffer({
        size: 4,
        usage: 0x0008 | 0x0001, // COPY_DST | MAP_READ
      });
    }
    ctx.queue.writeBuffer(
      fenceState.profilingFenceBuffer,
      0,
      profilingFenceData,
    );
    fenceState.pendingFencePromise = fenceState.profilingFenceBuffer
      .mapAsync(0x0001 /* MAP_READ */)
      .then(() => {
        fenceState.profilingFenceBuffer?.unmap();
      });
    fenceState.deferredPendingRelease = true;
    return;
  }

  if (typeof ctx.queue.onSubmittedWorkDone !== "function") return;
  fenceState.pendingFencePromise = ctx.queue.onSubmittedWorkDone();
  fenceState.deferredPendingRelease = true;
}

/**
 * Await the pending fence from a previous markStep, then flush/destroy
 * pending buffers. Must be called before any buffer reuse in a new step.
 */
export async function awaitDeferredFence(): Promise<void> {
  if (fenceState.pendingFencePromise) {
    await fenceState.pendingFencePromise;
    fenceState.pendingFencePromise = null;
  }
  if (fenceState.deferredPendingRelease) {
    bufferPool.flushPendingToPool();
    bufferPool.destroyPendingBuffers();
    fenceState.deferredPendingRelease = false;
  }
}

// ============================================================================
// Detailed stats, debug trace, and buffer lifecycle helpers
// ============================================================================

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
 * Queue a buffer for deferred destruction after GPU work completes.
 * Tracks deallocation immediately in the memory tracker.
 * Used by engine layer for buffers not managed by WebGPUTensor.destroy().
 */
export function deferredDestroyBuffer(buffer: GPUBuffer, size: number): void {
  bufferPool.deferredDestroy(buffer, size); // arena guard is inside deferredDestroy
}

// ============================================================================
// Profiling fence buffer cleanup
// ============================================================================

/**
 * Destroy the persistent profiling fence buffer.
 * Called during WebGPU context teardown (destroyWebGPU).
 */
export function destroyProfilingFenceBuffer(): void {
  if (fenceState.profilingFenceBuffer) {
    fenceState.profilingFenceBuffer.destroy();
    fenceState.profilingFenceBuffer = null;
  }
}
