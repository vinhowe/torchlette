import {
  bumpEpoch,
  currentEpoch,
  epochTrace,
  epochTraceEnabled,
} from "../core/epoch";
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

  /** Implied-step-boundary generation stamps (see
   *  Torchlette.queueStepBoundary). Keyed by the engine EPOCH id
   *  (src/core/epoch.ts): each owning TENSOR OBJECT is stamped with the
   *  epoch current when it is first seen; a queued step boundary records
   *  the current epoch as its closing generation and bumps the counter.
   *  Stamps are compared only relatively (`stamp > boundaryEpoch`), so the
   *  epoch also advancing at other quiesce points is harmless — monotonic
   *  time-order is all that's used. Gen-scoped cleanup uses the stamps to
   *  separate "this step's tensors" from work the NEXT step has already
   *  lazily built by the time the boundary commits. The stamp is
   *  wrapper-level, like persistence itself: a persistent tensor's STORAGE
   *  can be replaced mid-step (fused Adam's m/v side outputs re-point the
   *  wrapper every step), so a storage-level stamp would mis-classify
   *  long-lived state as next-step work and demote it — live optimizer
   *  state corrupted, the persistence-UAF class. */
  private _wrapperGen = new WeakMap<object, number>();

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
    if (!this._wrapperGen.has(tensorRef)) {
      this._wrapperGen.set(tensorRef, currentEpoch());
    }
  }

  /** Mark a step boundary (called when one is queued). Returns the epoch
   *  that closes — wrappers stamped <= this value belong to the step being
   *  ended — and bumps the engine epoch so everything created afterwards
   *  is stamped strictly greater. */
  bumpStepGen(): number {
    const closing = currentEpoch();
    bumpEpoch("stepBoundary");
    return closing;
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

    // Worklist-based cascading destruction. Each destroy can release a base
    // ref (rcRelease "view.destroyed"), potentially making the base dead too.
    // We re-collect dead storages until fixed point — single call, no caller loop.
    let totalDestroyed = 0;
    while (true) {
      // Only LIVE views protect their bases — dead views will release the
      // base retain when destroyed in this pass.
      const protectedBases = new Set<number>();
      for (const [id, storage] of this.allStorages) {
        if (storage.baseStorageId !== undefined && rcGet(id) > 0) {
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

      if (toDestroy.length === 0) break;
      totalDestroyed += this.destroyStorageIds(toDestroy);
    }
    if (epochTraceEnabled()) {
      epochTrace(`sweep destroyed=${totalDestroyed}`);
    }
    return totalDestroyed;
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

      // Release view base ref (view is being destroyed → base loses a reference).
      // EXCEPT for compiled-plan harvested views whose base retain is owned by
      // the plan (planOwnedBaseRetain): the plan releases that retain itself
      // (at the next harvest / teardown), so releasing here too would
      // double-free the base. See StorageHandle.planOwnedBaseRetain and
      // compiled-plan.ts harvest.
      if (
        storage.baseStorageId !== undefined &&
        !storage.planOwnedBaseRetain
      ) {
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
  snapshotForStep(maxGen?: number): void {
    this._stepStartTensors = new WeakSet();
    for (const [, ref] of this.tensorWeakRefs) {
      const target = ref.deref();
      if (!target) continue;
      // Gen-scoped snapshot (implied boundaries): tensors created AFTER the
      // boundary belong to the next step's lazily-built work — treating
      // them as persistent would exempt them from cleanup forever.
      if (
        maxGen !== undefined &&
        (this._wrapperGen.get(target) ?? 0) > maxGen
      ) {
        continue;
      }
      this._stepStartTensors.add(target);
    }
  }

  /**
   * Mark a tensor created MID-STEP as persistent: add it to the active
   * step snapshot so releaseStepTemps will not reclaim its storage at
   * markStep. This is THE supported way to create long-lived state inside
   * a step (e.g. lazily-initialized optimizer state): without it, the
   * snapshot-membership rule treats every mid-step tensor as a temporary —
   * its buffer is reclaimed into the pool while the live tensor still
   * points at it, and later writes corrupt it silently (the per-param Adam
   * first-param m/v corruption). No-op when no step is active (tensors
   * created between steps are captured by the next snapshot).
   */
  adoptIntoSnapshot(tensor: object): void {
    this._stepStartTensors?.add(tensor);
  }

  // ── Scope-surface support (api.scope / api.openScope) ─────────────────────
  // The async scope() surface (docs/scoped-memory-design.md §2-4) reuses the
  // SAME persistent-snapshot reclamation the step path uses: scope entry
  // snapshots the currently-alive tensors as this scope's "existed before"
  // baseline; scope exit adopts escapees into that baseline then
  // releaseStepTemps() releases everything created in-scope that didn't
  // escape. Nesting is LIFO save/restore of the single active snapshot: a
  // child scope steals the parent's snapshot (peekSnapshot), installs a fresh
  // one, and restores the parent's on close. This is the ONE reclamation path
  // — no second mechanism.

  /** The active persistent snapshot (opaque token; save it across a nested
   *  scope and restore it on the child's close). */
  peekSnapshot(): object | null {
    return this._stepStartTensors;
  }

  /** Install a previously-captured snapshot (restore the parent scope's on a
   *  child scope's close). */
  installSnapshot(snapshot: object | null): void {
    this._stepStartTensors = (snapshot as WeakSet<object> | null) ?? null;
  }

  /** Adopt a tensor into a SPECIFIC (possibly non-active) snapshot token —
   *  used by keep()/persist() to re-parent to an ancestor scope so a tensor
   *  survives every enclosing scope (escape-to-root). No-op for null tokens. */
  adoptIntoSnapshotToken(snapshot: object | null, tensor: object): void {
    (snapshot as WeakSet<object> | null)?.add(tensor);
  }

  /**
   * Drop the active step snapshot without releasing anything. Used when
   * step-scoped markStep cleanup (Torchlette.setStepScopedCleanup) is
   * DISABLED: its end-of-markStep snapshot must not linger, or a later bare
   * markStep — back on historical semantics — would consume it and reclaim
   * tensors created after the disable.
   */
  clearStepSnapshot(): void {
    this._stepStartTensors = null;
  }

  /**
   * Whether a storage has been DESTROYED (reclaimed). Reading a destroyed
   * storage through a stale ref is the silent-UAF class — its buffer is
   * back in the pool and may hold another op's data.
   */
  isDestroyed(storageId: number): boolean {
    return !this.allStorages.has(storageId);
  }

  /**
   * Release tensor refs for step-scoped temporaries.
   * For each tracked storage: if its owning tensor is alive but was NOT
   * in the beginStep snapshot, release the ref — it's a step-scoped temp.
   * Persistent tensors whose storages changed within the step (e.g., Adam
   * m/v via _updateLazyRef) stay alive because the tensor OBJECT is in
   * the snapshot, even if the storage id is new.
   */
  releaseStepTemps(maxGen?: number): number {
    if (!this._stepStartTensors) return 0;
    let released = 0;
    for (const [id, ref] of this.tensorWeakRefs) {
      const target = ref.deref();
      // Gen-scoped release (implied boundaries): tensors created after the
      // boundary are the NEXT step's — they get their own boundary.
      if (
        maxGen !== undefined &&
        target !== undefined &&
        (this._wrapperGen.get(target) ?? 0) > maxGen
      ) {
        continue;
      }
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
    if (epochTraceEnabled()) {
      epochTrace(`releaseStepTemps released=${released}`);
    }
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

  /**
   * Debug: snapshot the REACHABLE (rc>0) storage set for cross-boundary
   * leak diffs (task #79). Returns per-storage id/shape/dtype/view info so a
   * caller can diff two snapshots and histogram the delta by shape. Pure read;
   * cheap enough to call between generations, not on a hot path.
   */
  debugLiveSet(): Array<{
    id: number;
    shape: readonly number[];
    dtype: string;
    view: boolean;
    base?: number;
  }> {
    const out: Array<{
      id: number;
      shape: readonly number[];
      dtype: string;
      view: boolean;
      base?: number;
    }> = [];
    for (const [id, storage] of this.allStorages) {
      if (rcGet(id) <= 0) continue;
      const bt = storage.backendTensor;
      out.push({
        id,
        shape: bt.shape ?? [],
        dtype: bt.dtype ?? "?",
        view: bt.ownsBuffer === false,
        base: storage.baseStorageId,
      });
    }
    return out;
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
