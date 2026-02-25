import type { StorageHandle } from "./lazy-types";
import { getNextStorageId } from "./node-factory";

/**
 * Step-scoped storage tracker for memory management (§14).
 *
 * Tracks all StorageHandles created during execution and which are
 * externally reachable (linked to user-visible Tensors).
 * At markStep(), unreachable storages can be destroyed.
 */
class StorageTracker {
  /** All storages created and not yet destroyed */
  private allStorages = new Map<number, StorageHandle>();

  /** Storage IDs that are externally reachable (linked to Tensors) */
  private externallyReachable = new Set<number>();

  /** WeakRefs to owning tensors — used to detect GC'd tensors at cleanup time */
  private tensorWeakRefs = new Map<number, WeakRef<object>>();

  /** Storage IDs that recently became unreachable (for incremental scanning) */
  private recentlyUnreachable = new Set<number>();

  /** Debug counters for tracking reachability changes per step */
  private _debugRegisterCount = 0;
  private _debugReachableCount = 0;
  private _debugUnreachableCount = 0;
  private _debugDestroyCount = 0;

  /**
   * Register a newly created storage.
   */
  register(storage: StorageHandle): void {
    this.allStorages.set(storage.id, storage);
    this._debugRegisterCount++;
  }

  /**
   * Mark a storage as externally reachable (linked to a user-visible Tensor).
   * Optionally pass the owning tensor object so we can track it via WeakRef
   * and detect when it has been garbage collected.
   */
  markReachable(storageId: number, tensorRef?: object): void {
    const wasNew = !this.externallyReachable.has(storageId);
    this.externallyReachable.add(storageId);
    if (wasNew) this._debugReachableCount++;
    if (tensorRef) {
      this.tensorWeakRefs.set(storageId, new WeakRef(tensorRef));
    }
  }

  /**
   * Mark a storage as no longer externally reachable (Tensor disposed).
   */
  markUnreachable(storageId: number): void {
    const wasReachable = this.externallyReachable.has(storageId);
    this.externallyReachable.delete(storageId);
    this.tensorWeakRefs.delete(storageId);
    if (wasReachable) {
      this._debugUnreachableCount++;
      this.recentlyUnreachable.add(storageId);
    }
  }

  /**
   * Check if a storage is externally reachable.
   */
  isReachable(storageId: number): boolean {
    return this.externallyReachable.has(storageId);
  }

  /**
   * Unregister a storage (after it's been destroyed).
   */
  unregister(storageId: number): void {
    this.allStorages.delete(storageId);
    this.externallyReachable.delete(storageId);
    this.tensorWeakRefs.delete(storageId);
  }

  /**
   * Destroy all unreachable storages (called at markStep after GPU fence).
   * Returns the number of storages destroyed.
   *
   * Note: Only destroys storages whose backend tensors own their buffers.
   * Views (tensors that borrow buffers) are unregistered but not destroyed.
   * Base storages that are needed by reachable views are kept alive.
   */
  destroyUnreachable(): number {
    let destroyedCount = 0;

    // Step 0: Check WeakRefs — if the owning tensor was GC'd, demote to unreachable
    for (const [id, ref] of this.tensorWeakRefs) {
      if (ref.deref() === undefined) {
        this.externallyReachable.delete(id);
        this.tensorWeakRefs.delete(id);
        this.recentlyUnreachable.add(id);
      }
    }

    // Early exit: if all storages are reachable and none recently unreachable, skip scan
    if (this.recentlyUnreachable.size === 0 && this.allStorages.size === this.externallyReachable.size) {
      return 0;
    }

    // Step 1: Find all base storage IDs transitively needed by reachable view storages
    // If A is a view of B, and B is a view of C, then both B and C must stay alive
    const neededByViews = new Set<number>();
    const worklist = [...this.externallyReachable];
    while (worklist.length > 0) {
      const id = worklist.pop()!;
      const storage = this.allStorages.get(id);
      if (
        storage?.baseStorageId !== undefined &&
        !neededByViews.has(storage.baseStorageId)
      ) {
        neededByViews.add(storage.baseStorageId);
        worklist.push(storage.baseStorageId);
      }
    }

    // Step 2: Collect storages to destroy (unreachable and not needed by views)
    const toDestroy: number[] = [];
    for (const [id] of this.allStorages) {
      if (!this.externallyReachable.has(id) && !neededByViews.has(id)) {
        toDestroy.push(id);
      }
    }

    // Clear the recently unreachable set since we've scanned everything
    this.recentlyUnreachable.clear();

    // Step 3: Destroy collected storages
    for (const id of toDestroy) {
      const storage = this.allStorages.get(id);
      if (storage) {
        // Check if the backend tensor owns its buffer (not a view)
        const tensor = storage.backendTensor as {
          ownsBuffer?: boolean;
          destroy?: () => void;
        };

        // Only destroy if the tensor owns its buffer
        if (tensor.ownsBuffer !== false && tensor.destroy) {
          tensor.destroy();
        }
        this.allStorages.delete(id);
        destroyedCount++;
        this._debugDestroyCount++;
      }
    }

    return destroyedCount;
  }

  /**
   * Get the next storage ID that will be assigned.
   * Used to scope destroyUnreachableSince() to only affect newly created storages.
   */
  getNextStorageId(): number {
    return getNextStorageId();
  }

  /**
   * Destroy unreachable storages created at or after the given storage ID.
   * This is a scoped version of destroyUnreachable() that only affects
   * storages created within a specific time window (e.g., during backward pass
   * gradient computation). Pre-existing unreachable storages are left alone.
   * Returns the number of storages destroyed.
   */
  destroyUnreachableSince(sinceId: number): number {
    let destroyedCount = 0;

    // Step 1: Find all base storage IDs transitively needed by reachable view storages
    const neededByViews = new Set<number>();
    const worklist = [...this.externallyReachable];
    while (worklist.length > 0) {
      const id = worklist.pop()!;
      const storage = this.allStorages.get(id);
      if (
        storage?.baseStorageId !== undefined &&
        !neededByViews.has(storage.baseStorageId)
      ) {
        neededByViews.add(storage.baseStorageId);
        worklist.push(storage.baseStorageId);
      }
    }

    // Step 2: Collect storages to destroy (unreachable, not needed by views, created since sinceId)
    const toDestroy: number[] = [];
    for (const [id, storage] of this.allStorages) {
      if (id >= sinceId && !this.externallyReachable.has(id) && !neededByViews.has(id)) {
        toDestroy.push(id);
      }
    }

    // Step 3: Destroy collected storages
    for (const id of toDestroy) {
      const storage = this.allStorages.get(id);
      if (storage) {
        const tensor = storage.backendTensor as {
          ownsBuffer?: boolean;
          destroy?: () => void;
        };
        if (tensor.ownsBuffer !== false && tensor.destroy) {
          tensor.destroy();
        }
        this.allStorages.delete(id);
        destroyedCount++;
        this._debugDestroyCount++;
      }
    }

    return destroyedCount;
  }

  /**
   * Get statistics about tracked storages.
   */
  stats(): {
    totalStorages: number;
    reachableStorages: number;
    unreachableStorages: number;
  } {
    return {
      totalStorages: this.allStorages.size,
      reachableStorages: this.externallyReachable.size,
      unreachableStorages:
        this.allStorages.size - this.externallyReachable.size,
    };
  }

  /**
   * Get and reset debug counters.
   */
  debugCounters(): { registered: number; reachable: number; unreachable: number; destroyed: number } {
    const result = {
      registered: this._debugRegisterCount,
      reachable: this._debugReachableCount,
      unreachable: this._debugUnreachableCount,
      destroyed: this._debugDestroyCount,
    };
    this._debugRegisterCount = 0;
    this._debugReachableCount = 0;
    this._debugUnreachableCount = 0;
    this._debugDestroyCount = 0;
    return result;
  }

  /**
   * Reset the tracker (for testing).
   */
  reset(): void {
    this.allStorages.clear();
    this.externallyReachable.clear();
    this.tensorWeakRefs.clear();
    this.recentlyUnreachable.clear();
  }

  /**
   * Get the set of externally reachable storage IDs.
   */
  getReachableIds(): Set<number> {
    return new Set(this.externallyReachable);
  }

  /**
   * Check if a storage has a live (not GC'd) tensor WeakRef.
   */
  hasLiveTensorRef(storageId: number): boolean {
    const ref = this.tensorWeakRefs.get(storageId);
    if (!ref) return false;
    return ref.deref() !== undefined;
  }

  /**
   * Get a storage by ID.
   */
  getStorage(storageId: number): StorageHandle | undefined {
    return this.allStorages.get(storageId);
  }

  /**
   * Get debug info about the tensor ref holding a storage reachable.
   * Returns shape/dtype if the ref is a RuntimeTensor, or a description otherwise.
   */
  getTensorRefDebugInfo(storageId: number): { shape?: number[]; dtype?: string; type: string; disposed?: boolean } | null {
    const ref = this.tensorWeakRefs.get(storageId);
    if (!ref) return null;
    const obj = ref.deref();
    if (!obj) return null;
    // Check if it's a RuntimeTensor (has shape and dtype fields)
    if ('shape' in obj && 'dtype' in obj) {
      const t = obj as { shape: number[]; dtype: string; disposed?: boolean; _disposed?: boolean };
      return {
        shape: t.shape,
        dtype: t.dtype,
        type: 'tensor',
        disposed: t.disposed ?? t._disposed,
      };
    }
    // It's a sideOutputs object or other ref
    if ('m' in obj && 'v' in obj) {
      return { type: 'adamSideOutputs' };
    }
    return { type: typeof obj };
  }

  /**
   * Get storages that became reachable since a given snapshot.
   * Returns entries for IDs present in current reachable set but absent from prevIds.
   */
  getNewReachableSince(prevIds: Set<number>): Array<{
    id: number;
    hasLiveTensorRef: boolean;
    debugInfo: ReturnType<StorageTracker["getTensorRefDebugInfo"]>;
  }> {
    const result: Array<{
      id: number;
      hasLiveTensorRef: boolean;
      debugInfo: ReturnType<StorageTracker["getTensorRefDebugInfo"]>;
    }> = [];
    for (const id of this.externallyReachable) {
      if (!prevIds.has(id)) {
        result.push({
          id,
          hasLiveTensorRef: this.hasLiveTensorRef(id),
          debugInfo: this.getTensorRefDebugInfo(id),
        });
      }
    }
    return result;
  }

  /**
   * Get all buffer objects from live storages that own their buffer.
   * For cross-referencing with memory tracker to find orphaned buffers.
   */
  getLiveOwnedBuffers(): Set<unknown> {
    const buffers = new Set<unknown>();
    for (const [, storage] of this.allStorages) {
      const tensor = storage.backendTensor as { ownsBuffer?: boolean; buffer?: unknown };
      if (tensor.ownsBuffer !== false && tensor.buffer) {
        buffers.add(tensor.buffer);
      }
    }
    return buffers;
  }
}

/** Global storage tracker instance */
export const storageTracker = new StorageTracker();

// ============================================================================
// Early Release Helpers for Memory-Aware Execution
// ============================================================================

/**
 * Check if a storage can be safely released during plan execution.
 *
 * A storage can be safely released if:
 * 1. It's not externally reachable (not linked to a user-visible Tensor)
 * 2. No other active storage uses it as a base (view aliasing)
 *
 * @param storage - The storage to check
 * @param activeStorages - Map of all storages that are still active in the plan
 */
export function canSafelyRelease(
  storage: StorageHandle,
  activeStorages: Map<number, StorageHandle>,
): boolean {
  // Cannot release if externally reachable (linked to user tensor)
  if (storageTracker.isReachable(storage.id)) {
    return false;
  }

  // Cannot release if this storage is the base for another active storage (view aliasing)
  for (const [, activeStorage] of activeStorages) {
    if (activeStorage.baseStorageId === storage.id) {
      return false;
    }
  }

  return true;
}

/**
 * Release a buffer during plan execution.
 *
 * For WebGPU: The buffer pool uses deferred destruction - buffers are queued
 * for destruction after the GPU fence signals that work is complete. This
 * prevents "buffer destroyed while in use" validation errors.
 *
 * For CPU: Buffers are destroyed immediately since operations are synchronous.
 *
 * @param storage - The storage handle to release
 */
export function releaseBufferImmediate(storage: StorageHandle): void {
  const tensor = storage.backendTensor as {
    ownsBuffer?: boolean;
    destroy?: () => void;
  };

  // Don't release views (they don't own buffers)
  if (tensor.ownsBuffer === false) {
    return;
  }

  // Unregister from storage tracker to prevent double-free at markStep
  storageTracker.unregister(storage.id);

  // Call destroy() - for WebGPU, this uses deferred destruction via the buffer pool
  // which waits for the GPU fence before actually destroying the buffer
  if (tensor.destroy) {
    tensor.destroy();
  }
}
