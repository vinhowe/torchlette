/**
 * GPU Memory Tracker
 *
 * Centralized tracking of GPU memory allocations with configurable limits.
 * This ensures ALL buffer allocations respect the memory limit.
 */

/**
 * Default memory limit: 32GB (matches V100 GPU)
 * This is a soft limit to prevent runaway memory usage.
 */
const DEFAULT_MEMORY_LIMIT_BYTES = 32 * 1024 * 1024 * 1024;

/**
 * Error thrown when GPU memory allocation would exceed the limit.
 */
export class GPUMemoryLimitExceededError extends Error {
  constructor(
    public readonly requestedBytes: number,
    public readonly currentBytes: number,
    public readonly limitBytes: number,
  ) {
    super(
      `GPU memory limit exceeded: requested ${formatBytes(requestedBytes)}, ` +
        `current usage ${formatBytes(currentBytes)}, ` +
        `limit ${formatBytes(limitBytes)}`,
    );
    this.name = "GPUMemoryLimitExceededError";
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)}MB`;
  }
  if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)}KB`;
  }
  return `${bytes}B`;
}

/**
 * GPU Memory Tracker - singleton that tracks all GPU buffer allocations.
 */
class GPUMemoryTracker {
  private memoryLimitBytes: number = DEFAULT_MEMORY_LIMIT_BYTES;
  private currentAllocatedBytes = 0;
  private bufferSizes = new Map<unknown, number>();
  private allocationCount = 0;
  private peakUsageBytes = 0;

  /**
   * Get the current memory limit in bytes.
   */
  getMemoryLimit(): number {
    return this.memoryLimitBytes;
  }

  /**
   * Set the memory limit in bytes.
   */
  setMemoryLimit(limitBytes: number): void {
    if (limitBytes <= 0) {
      throw new Error("Memory limit must be positive");
    }
    this.memoryLimitBytes = limitBytes;
  }

  /**
   * Get the current total allocated memory in bytes.
   */
  getCurrentAllocatedBytes(): number {
    return this.currentAllocatedBytes;
  }

  /**
   * Get the peak memory usage in bytes.
   */
  getPeakUsageBytes(): number {
    return this.peakUsageBytes;
  }

  /**
   * Get memory usage as a percentage of the limit.
   */
  getUsagePercent(): number {
    return (this.currentAllocatedBytes / this.memoryLimitBytes) * 100;
  }

  /**
   * Get the number of active allocations.
   */
  getAllocationCount(): number {
    return this.allocationCount;
  }

  /**
   * Track a buffer allocation.
   * @throws GPUMemoryLimitExceededError if allocation would exceed the limit
   */
  // Debug: track large allocation sources
  private _debugLargeAllocLog: string[] = [];
  private _debugLargeAllocEnabled = false;

  enableLargeAllocDebug(): void { this._debugLargeAllocEnabled = true; }
  getLargeAllocLog(): string[] { return this._debugLargeAllocLog; }
  clearLargeAllocLog(): void { this._debugLargeAllocLog = []; }

  // Debug: track ALL allocation stack traces for leak detection
  private _debugAllAllocEnabled = false;
  private _allocStacks = new Map<unknown, { size: number; stack: string; timestamp: number; step: number }>();
  private _currentStep = 0;

  // Debug: per-step allocation/deallocation flow counters
  private _debugAllocCount = 0;
  private _debugDeallocCount = 0;
  private _debugDeallocMissCount = 0; // trackDeallocation called but buffer not in bufferSizes
  private _debugDoubleTrackCount = 0; // trackAllocation called for already-tracked buffer

  enableAllAllocDebug(): void { this._debugAllAllocEnabled = true; }
  disableAllAllocDebug(): void { this._debugAllAllocEnabled = false; }
  clearAllocStacks(): void { this._allocStacks.clear(); }
  setAllocStep(step: number): void { this._currentStep = step; }

  /**
   * Snapshot current unmatched allocations, grouped by call site.
   */
  snapshotLeakedAllocs(): Map<string, { count: number; totalBytes: number; exampleStack: string }> {
    const grouped = new Map<string, { count: number; totalBytes: number; exampleStack: string }>();
    for (const [, info] of this._allocStacks) {
      const key = info.stack.split('\n').slice(0, 3).join('\n');
      const existing = grouped.get(key) || { count: 0, totalBytes: 0, exampleStack: info.stack };
      existing.count++;
      existing.totalBytes += info.size;
      grouped.set(key, existing);
    }
    return grouped;
  }

  /**
   * Snapshot unmatched allocations from a specific step only.
   */
  snapshotLeakedAllocsForStep(step: number): Map<string, { count: number; totalBytes: number; exampleStack: string }> {
    const grouped = new Map<string, { count: number; totalBytes: number; exampleStack: string }>();
    for (const [, info] of this._allocStacks) {
      if (info.step !== step) continue;
      const key = info.stack.split('\n').slice(0, 3).join('\n');
      const existing = grouped.get(key) || { count: 0, totalBytes: 0, exampleStack: info.stack };
      existing.count++;
      existing.totalBytes += info.size;
      grouped.set(key, existing);
    }
    return grouped;
  }

  /**
   * Count unmatched allocations from a specific step.
   */
  getLeakedAllocCountForStep(step: number): number {
    let count = 0;
    for (const [, info] of this._allocStacks) {
      if (info.step === step) count++;
    }
    return count;
  }

  /**
   * Get the number of tracked-but-not-deallocated allocations.
   */
  getLeakedAllocCount(): number {
    return this._allocStacks.size;
  }

  /**
   * Get the actual buffer objects tracked in bufferSizes (for cross-referencing).
   */
  getTrackedBuffers(): Set<unknown> {
    return new Set(this.bufferSizes.keys());
  }

  /**
   * Get size histogram of leaked allocations for a specific step.
   */
  getLeakedSizeHistogramForStep(step: number): Map<number, number> {
    const histogram = new Map<number, number>();
    for (const [, info] of this._allocStacks) {
      if (info.step !== step) continue;
      const count = histogram.get(info.size) || 0;
      histogram.set(info.size, count + 1);
    }
    return histogram;
  }

  /**
   * Get and reset per-step allocation/deallocation flow counters.
   */
  getAndResetFlowCounters(): { allocs: number; deallocs: number; deallocMisses: number; doubleTracked: number } {
    const result = {
      allocs: this._debugAllocCount,
      deallocs: this._debugDeallocCount,
      deallocMisses: this._debugDeallocMissCount,
      doubleTracked: this._debugDoubleTrackCount,
    };
    this._debugAllocCount = 0;
    this._debugDeallocCount = 0;
    this._debugDeallocMissCount = 0;
    this._debugDoubleTrackCount = 0;
    return result;
  }

  /**
   * Scoped limit-check suppression. When > 0, trackAllocation skips
   * the limit check (behaves like trackAllocationForced).
   * Used by the optimizer where temporary 2x peak memory is expected.
   */
  private _suppressLimitCheckDepth = 0;

  suppressLimitCheck(): void { this._suppressLimitCheckDepth++; }
  unsuppressLimitCheck(): void { this._suppressLimitCheckDepth = Math.max(0, this._suppressLimitCheckDepth - 1); }

  trackAllocation(buffer: unknown, sizeBytes: number): void {
    if (this._suppressLimitCheckDepth === 0 && this.currentAllocatedBytes + sizeBytes > this.memoryLimitBytes) {
      throw new GPUMemoryLimitExceededError(
        sizeBytes,
        this.currentAllocatedBytes,
        this.memoryLimitBytes,
      );
    }
    this._trackAllocationInner(buffer, sizeBytes);
  }

  /**
   * Track allocation without throwing on limit exceeded.
   * Used by the optimizer where old buffers haven't been released yet
   * but will be shortly after (temporary peak is expected and safe).
   */
  trackAllocationForced(buffer: unknown, sizeBytes: number): void {
    this._trackAllocationInner(buffer, sizeBytes);
  }

  private _trackAllocationInner(buffer: unknown, sizeBytes: number): void {
    if (this._debugAllAllocEnabled && buffer !== null && this.bufferSizes.has(buffer)) {
      this._debugDoubleTrackCount++;
    }
    this.bufferSizes.set(buffer, sizeBytes);
    this.currentAllocatedBytes += sizeBytes;
    this.allocationCount++;

    if (this._debugLargeAllocEnabled && buffer !== null && sizeBytes > 16 * 1024 * 1024) {
      const stack = new Error().stack?.split('\n').slice(1, 6).join('\n') ?? 'no stack';
      this._debugLargeAllocLog.push(`ALLOC ${(sizeBytes / 1e6).toFixed(2)}MB:\n${stack}`);
    }

    if (this._debugAllAllocEnabled && buffer !== null) {
      const stack = new Error().stack?.split('\n').slice(1, 15).join('\n') ?? 'no stack';
      this._allocStacks.set(buffer, { size: sizeBytes, stack, timestamp: Date.now(), step: this._currentStep });
    }

    if (this.currentAllocatedBytes > this.peakUsageBytes) {
      this.peakUsageBytes = this.currentAllocatedBytes;
    }

    if (this._debugAllAllocEnabled) {
      this._debugAllocCount++;
    }
  }

  /**
   * Track a buffer deallocation.
   */
  trackDeallocation(buffer: unknown): void {
    const size = this.bufferSizes.get(buffer);
    if (size !== undefined) {
      this.currentAllocatedBytes -= size;
      this.bufferSizes.delete(buffer);
      this.allocationCount--;
    }

    if (this._debugAllAllocEnabled) {
      this._allocStacks.delete(buffer);
    }

    if (this._debugAllAllocEnabled) {
      this._debugDeallocCount++;
      if (size === undefined) {
        this._debugDeallocMissCount++;
      }
    }

    if (size === undefined && this._debugLargeAllocEnabled && buffer !== null) {
      const stack = new Error().stack?.split('\n').slice(1, 4).join('\n') ?? 'no stack';
      this._debugLargeAllocLog.push(`DEALLOC_MISS (buffer not found):\n${stack}`);
    }
  }

  /**
   * Check if an allocation of the given size would exceed the limit.
   * @param freeableBytes - Bytes that can be freed if needed (e.g., pending buffers)
   */
  wouldExceedLimit(sizeBytes: number, freeableBytes = 0): boolean {
    const effectiveAllocated = this.currentAllocatedBytes - freeableBytes;
    return effectiveAllocated + sizeBytes > this.memoryLimitBytes;
  }

  /**
   * Get statistics about memory usage.
   */
  stats(): {
    currentBytes: number;
    peakBytes: number;
    limitBytes: number;
    usagePercent: number;
    allocationCount: number;
    bufferSizesCount: number;
    availableBytes: number;
  } {
    return {
      currentBytes: this.currentAllocatedBytes,
      peakBytes: this.peakUsageBytes,
      limitBytes: this.memoryLimitBytes,
      usagePercent: this.getUsagePercent(),
      allocationCount: this.allocationCount,
      bufferSizesCount: this.bufferSizes.size,
      availableBytes: this.memoryLimitBytes - this.currentAllocatedBytes,
    };
  }

  /**
   * Reset all tracking state (for testing).
   */
  reset(): void {
    this.currentAllocatedBytes = 0;
    this.bufferSizes.clear();
    this.allocationCount = 0;
    this.peakUsageBytes = 0;
  }

  /**
   * Get a histogram of allocation sizes for diagnostics.
   * Returns a map of size bucket labels to counts.
   */
  getAllocationSizeHistogram(): Map<string, { count: number; totalBytes: number }> {
    const buckets = new Map<string, { count: number; totalBytes: number }>();
    for (const [, size] of this.bufferSizes) {
      let label: string;
      if (size <= 256) label = "<=256B";
      else if (size <= 1024) label = "<=1KB";
      else if (size <= 4096) label = "<=4KB";
      else if (size <= 16384) label = "<=16KB";
      else if (size <= 65536) label = "<=64KB";
      else if (size <= 262144) label = "<=256KB";
      else if (size <= 1048576) label = "<=1MB";
      else if (size <= 4194304) label = "<=4MB";
      else if (size <= 16777216) label = "<=16MB";
      else label = ">16MB";

      const existing = buckets.get(label) || { count: 0, totalBytes: 0 };
      existing.count++;
      existing.totalBytes += size;
      buckets.set(label, existing);
    }
    return buckets;
  }
}

/**
 * Global GPU memory tracker instance.
 */
export const gpuMemoryTracker = new GPUMemoryTracker();

/**
 * Set the GPU memory limit.
 * This is a convenience function that delegates to the global tracker.
 */
export function setGPUMemoryLimit(limitBytes: number): void {
  gpuMemoryTracker.setMemoryLimit(limitBytes);
}

/**
 * Get the current GPU memory limit.
 */
export function getGPUMemoryLimit(): number {
  return gpuMemoryTracker.getMemoryLimit();
}

/**
 * Get GPU memory statistics.
 */
export function getGPUMemoryStats(): ReturnType<GPUMemoryTracker["stats"]> {
  return gpuMemoryTracker.stats();
}

/**
 * Get allocation size histogram for diagnostics.
 */
export function getGPUAllocationHistogram(): Map<string, { count: number; totalBytes: number }> {
  return gpuMemoryTracker.getAllocationSizeHistogram();
}

export function enableLargeAllocDebug(): void {
  gpuMemoryTracker.enableLargeAllocDebug();
}

export function getLargeAllocLog(): string[] {
  return gpuMemoryTracker.getLargeAllocLog();
}

export function clearLargeAllocLog(): void {
  gpuMemoryTracker.clearLargeAllocLog();
}

export function enableAllAllocDebug(): void {
  gpuMemoryTracker.enableAllAllocDebug();
}

export function disableAllAllocDebug(): void {
  gpuMemoryTracker.disableAllAllocDebug();
}

export function clearAllocStacks(): void {
  gpuMemoryTracker.clearAllocStacks();
}

export function setAllocStep(step: number): void {
  gpuMemoryTracker.setAllocStep(step);
}

export function snapshotLeakedAllocs(): Map<string, { count: number; totalBytes: number; exampleStack: string }> {
  return gpuMemoryTracker.snapshotLeakedAllocs();
}

export function snapshotLeakedAllocsForStep(step: number): Map<string, { count: number; totalBytes: number; exampleStack: string }> {
  return gpuMemoryTracker.snapshotLeakedAllocsForStep(step);
}

export function getLeakedAllocCount(): number {
  return gpuMemoryTracker.getLeakedAllocCount();
}

export function getLeakedAllocCountForStep(step: number): number {
  return gpuMemoryTracker.getLeakedAllocCountForStep(step);
}

export function getAndResetFlowCounters(): ReturnType<GPUMemoryTracker["getAndResetFlowCounters"]> {
  return gpuMemoryTracker.getAndResetFlowCounters();
}

export function getTrackedBuffers(): Set<unknown> {
  return gpuMemoryTracker.getTrackedBuffers();
}

export function getLeakedSizeHistogramForStep(step: number): Map<number, number> {
  return gpuMemoryTracker.getLeakedSizeHistogramForStep(step);
}
