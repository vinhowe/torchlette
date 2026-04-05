import {
  findDeadTensorsAtStep,
  type TensorLifetime,
} from "./lifetime-analysis";
import { getNextStorageId } from "./node-factory";
import { rcDelete, rcGet, rcRelease } from "./refcount";
import type { StorageHandle } from "./types";

/**
 * Storage tracker — manages StorageHandle lifecycle via reference counting.
 *
 * Liveness is determined by rc (reference count):
 *   rc > 0  →  alive (some tensor or view holds a reference)
 *   rc <= 0 →  dead (eligible for destruction)
 *
 * Views keep their base alive through rcRetain on the base storage.
 * No separate "needed by views" walk is required.
 *
 * WeakRefs are kept as a safety net: when a tensor is GC'd without being
 * disposed, the WeakRef scan in destroyUnreachable() detects this and
 * releases the tensor's claim (rcRelease). This ensures storages are
 * eventually freed even without explicit dispose() or `using`.
 */
class StorageTracker {
  /** All storages created and not yet destroyed */
  private allStorages = new Map<number, StorageHandle>();

  /** WeakRefs to owning tensors — safety net for GC'd undisposed tensors */
  private tensorWeakRefs = new Map<number, WeakRef<object>>();

  /** Tensors alive at beginStep — these are "persistent" across the step.
   *  A tensor's storage can change within a step (via _updateLazyRef), but
   *  the tensor object is stable. Tracking the tensor (not the storage id)
   *  ensures persistence survives storage replacement (e.g., Adam m/v updates). */
  private _stepStartTensors: WeakSet<object> | null = null;

  /** Debug counters */
  private _debugDestroyCount = 0;

  /** Register a newly created storage. */
  register(storage: StorageHandle): void {
    this.allStorages.set(storage.id, storage);
  }

  /**
   * Register a WeakRef to the tensor that owns this storage.
   * Used by destroyUnreachable() to detect GC'd tensors.
   */
  trackTensor(storageId: number, tensorRef: object): void {
    this.tensorWeakRefs.set(storageId, new WeakRef(tensorRef));
  }

  /** Unregister a storage (after it's been destroyed or early-released). */
  unregister(storageId: number): void {
    this.allStorages.delete(storageId);
    this.tensorWeakRefs.delete(storageId);
  }

  /**
   * Destroy all storages with rc <= 0.
   *
   * First scans WeakRefs to detect GC'd tensors and release their claims,
   * which may push more storages to rc=0. Then collects and destroys all
   * dead storages.
   */
  destroyUnreachable(): number {
    // Safety net: detect GC'd tensors and release their claims
    for (const [id, ref] of this.tensorWeakRefs) {
      if (ref.deref() === undefined) {
        this.tensorWeakRefs.delete(id);
        if (rcGet(id) > 0) rcRelease(id, "gc");
      }
    }

    // Collect dead storages (rc <= 0), but protect view base chains.
    // A live storage (rc > 0) with baseStorageId keeps its base alive
    // even if the base itself has rc=0 (e.g., unclaimed intermediate base
    // only kept alive by the view's retain). This mirrors the old
    // findNeededByViews check.
    const protectedBases = new Set<number>();
    for (const [, storage] of this.allStorages) {
      if (storage.baseStorageId !== undefined) {
        let baseId: number | undefined = storage.baseStorageId;
        while (baseId !== undefined && !protectedBases.has(baseId)) {
          protectedBases.add(baseId);
          baseId = this.allStorages.get(baseId)?.baseStorageId;
        }
      }
    }

    const toDestroy: number[] = [];
    for (const [id] of this.allStorages) {
      if (rcGet(id) <= 0 && !protectedBases.has(id)) {
        toDestroy.push(id);
      }
    }

    if (toDestroy.length === 0) return 0;
    return this.destroyStorageIds(toDestroy);
  }

  /** Destroy storages with rc <= 0, scoped to ids >= sinceId. */
  destroyUnreachableSince(sinceId: number): number {
    const toDestroy: number[] = [];
    for (const [id] of this.allStorages) {
      if (id < sinceId) continue;
      if (rcGet(id) <= 0) {
        toDestroy.push(id);
      }
    }
    if (toDestroy.length === 0) return 0;
    return this.destroyStorageIds(toDestroy);
  }

  /** Destroy a list of storage IDs. Returns count destroyed. */
  private destroyStorageIds(ids: number[]): number {
    let count = 0;
    for (const id of ids) {
      const storage = this.allStorages.get(id);
      if (!storage) continue;

      // Release view base ref (view is being destroyed → base loses a reference)
      if (storage.baseStorageId !== undefined) {
        rcRelease(storage.baseStorageId, "view.destroyed");
      }
      rcDelete(id);

      const bt = storage.backendTensor;
      if (bt.ownsBuffer !== false && bt.destroy) {
        bt.destroy();
      }
      this.allStorages.delete(id);
      this.tensorWeakRefs.delete(id);
      count++;
      this._debugDestroyCount++;
    }
    return count;
  }

  /**
   * Snapshot tensors alive at the start of the step. These are "persistent"
   * (model params, optimizer state). Tensors created during the step are
   * step-scoped temporaries and will have their refs released at markStep.
   */
  snapshotForStep(): void {
    this._stepStartTensors = new WeakSet();
    for (const [, ref] of this.tensorWeakRefs) {
      const target = ref.deref();
      if (target) this._stepStartTensors.add(target);
    }
  }

  /**
   * Release tensor refs for step-scoped temporaries.
   * For each tracked storage: if its owning tensor is alive but was NOT
   * in the beginStep snapshot, release the ref — it's a step-scoped temp.
   * Persistent tensors whose storages changed within the step (e.g., Adam
   * m/v via _updateLazyRef) stay alive because the tensor OBJECT is in
   * the snapshot, even if the storage id is new.
   */
  releaseStepTemps(): number {
    if (!this._stepStartTensors) return 0;
    let released = 0;
    for (const [id, ref] of this.tensorWeakRefs) {
      const target = ref.deref();
      // GC'd tensor: skip (will be handled by FR or gc scan in destroyUnreachable)
      if (target === undefined) continue;
      // Persistent tensor: skip
      if (this._stepStartTensors.has(target)) continue;
      // Step-scoped tensor with live ref: release its claim
      if (rcGet(id) > 0) {
        rcRelease(id, "stepScoped");
        released++;
      }
    }
    this._stepStartTensors = null;
    return released;
  }

  getNextStorageId(): number {
    return getNextStorageId();
  }

  stats(): {
    totalStorages: number;
    reachableStorages: number;
    unreachableStorages: number;
  } {
    let reachable = 0;
    for (const [id] of this.allStorages) {
      if (rcGet(id) > 0) reachable++;
    }
    return {
      totalStorages: this.allStorages.size,
      reachableStorages: reachable,
      unreachableStorages: this.allStorages.size - reachable,
    };
  }

  debugCounters(): { destroyed: number } {
    const d = this._debugDestroyCount;
    this._debugDestroyCount = 0;
    return { destroyed: d };
  }

  reset(): void {
    this.allStorages.clear();
    this.tensorWeakRefs.clear();
  }

  getStorage(storageId: number): StorageHandle | undefined {
    return this.allStorages.get(storageId);
  }

  getLiveOwnedBuffers(): Set<unknown> {
    const buffers = new Set<unknown>();
    for (const [, storage] of this.allStorages) {
      const bt = storage.backendTensor;
      const buf = (bt as { buffer?: unknown }).buffer;
      if (bt.ownsBuffer !== false && buf) {
        buffers.add(buf);
      }
    }
    return buffers;
  }
}

/** Global storage tracker instance */
export const storageTracker = new StorageTracker();

// ============================================================================
// Early Release Helpers
// ============================================================================

/**
 * Check if a storage can be safely released during plan execution.
 * Uses refcount: rc <= 0 means no one references it.
 * Also checks view aliasing (another active storage uses this as base).
 */
export function canSafelyRelease(
  storage: StorageHandle,
  activeStorages: Map<number, StorageHandle>,
): boolean {
  if (rcGet(storage.id) > 0) return false;
  for (const [, active] of activeStorages) {
    if (active.baseStorageId === storage.id) return false;
  }
  return true;
}

/**
 * Release a buffer during plan execution.
 */
export function releaseBufferImmediate(storage: StorageHandle): void {
  const bt = storage.backendTensor;

  // Views: release base ref but don't destroy (don't own buffer).
  // Don't unregister — the view may still be referenced by the shared encoder.
  // Clear baseStorageId to prevent double-release when destroyStorageIds
  // later processes this view at destroyUnreachable time.
  if (bt.ownsBuffer === false) {
    if (storage.baseStorageId !== undefined) {
      rcRelease(storage.baseStorageId, "earlyRelease.viewBase");
      storage.baseStorageId = undefined;
    }
    return;
  }

  // Owning storage: release base ref if view, unregister, destroy.
  if (storage.baseStorageId !== undefined) {
    rcRelease(storage.baseStorageId, "earlyRelease.base");
    storage.baseStorageId = undefined;
  }
  rcDelete(storage.id);
  storageTracker.unregister(storage.id);
  if (bt.destroy) bt.destroy();
}

/**
 * Find and release all dead tensors at a given execution step.
 */
export function releaseDeadTensors(
  lifetimes: Map<number, TensorLifetime> | null,
  step: number,
  outputNodeIds: Set<number> | null,
  alreadyReleased: Set<number>,
  nodeToStorage: Map<number, StorageHandle>,
): void {
  if (!lifetimes || !outputNodeIds) return;
  const deadNodeIds = findDeadTensorsAtStep(
    lifetimes,
    step,
    outputNodeIds,
    alreadyReleased,
  );
  for (const deadId of deadNodeIds) {
    const storage = nodeToStorage.get(deadId);
    if (storage && canSafelyRelease(storage, nodeToStorage)) {
      releaseBufferImmediate(storage);
      nodeToStorage.delete(deadId);
      alreadyReleased.add(deadId);
    }
  }
}
