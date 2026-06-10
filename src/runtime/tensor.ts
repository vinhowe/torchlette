import type { BackendTensor, DeviceKind, DType, Shape } from "../backend/types";
import { storageTracker } from "../graph/storage-tracker";
import { rcGet, rcRelease, rcRetain } from "../graph/refcount";
import type { LazyRef, StorageHandle } from "../graph/types";
import { isMaterialized } from "../graph/types";

export type BaseId = number;

/**
 * FinalizationRegistry for automatic GPU memory cleanup (§1.4).
 * When a RuntimeTensor is garbage collected, we mark its storage as unreachable.
 * This ensures that tensors which go out of scope (like optimizer intermediates)
 * are automatically cleaned up without requiring explicit disposal.
 */
const tensorFinalizationRegistry = new FinalizationRegistry<{
  storageId: number;
  pendingNodeId: number | null;
}>((held) => {
  // Release tensor's claim when GC'd (safety net for undisposed tensors).
  // Guard: the WeakRef scan in destroyUnreachable() may have already released.
  if (held.storageId >= 0 && rcGet(held.storageId) > 0) {
    rcRelease(held.storageId, "gc.fr");
  }
  // Note: pending tensor unregistration is handled in dispose()
  // By the time finalization runs, the tensor is already gone from pending registry
});

/**
 * Global registry of pending tensors by their LazyIRNode ID.
 * Used to materialize all tensors whose nodes were executed during force().
 */
const pendingTensorsByNodeId = new Map<number, Set<Tensor>>();

/**
 * Node IDs whose pending tensors were disposed (e.g., by checkpoint's tidy()).
 * These IDs must still appear in getPendingNodeIds() so fusion analysis treats
 * them as external, but we don't keep full Tensor references (that would leak).
 * Cleared after each plan execution via clearDisposedPendingNodeIds().
 */
const disposedPendingNodeIds = new Set<number>();

/**
 * Register a tensor as pending on a node ID.
 */
function registerPendingTensor(nodeId: number, tensor: Tensor): void {
  let tensors = pendingTensorsByNodeId.get(nodeId);
  if (!tensors) {
    tensors = new Set();
    pendingTensorsByNodeId.set(nodeId, tensors);
  }
  tensors.add(tensor);
}

/**
 * Unregister a tensor from its pending node.
 */
function unregisterPendingTensor(nodeId: number, tensor: Tensor): void {
  const tensors = pendingTensorsByNodeId.get(nodeId);
  if (tensors) {
    tensors.delete(tensor);
    if (tensors.size === 0) {
      pendingTensorsByNodeId.delete(nodeId);
    }
  }
}

/**
 * Materialize all tensors pending on a node with the given storage.
 * Called after plan execution when a node's result is set.
 */
export function materializePendingTensors(
  nodeId: number,
  storage: StorageHandle,
  allResults?: StorageHandle[],
): void {
  const tensors = pendingTensorsByNodeId.get(nodeId);
  if (tensors) {
    for (const tensor of tensors) {
      if (!tensor.isMaterialized() && !tensor.disposed) {
        // Multi-output: use the correct result based on outputIndex
        const ref = tensor.lazyRef;
        const outputIdx = ref.kind === "pending" ? (ref.outputIndex ?? 0) : 0;
        const targetStorage =
          outputIdx > 0 && allResults?.[outputIdx]
            ? allResults[outputIdx]
            : storage;
        tensor._materialize(targetStorage);
      }
    }
    // Don't delete from map - tensors will unregister themselves
  }
}

/**
 * Get all pending tensors (for forcing before cleanup).
 */
export function getAllPendingTensors(): Tensor[] {
  const result: Tensor[] = [];
  for (const tensors of pendingTensorsByNodeId.values()) {
    for (const tensor of tensors) {
      if (!tensor.disposed) {
        result.push(tensor);
      }
    }
  }
  return result;
}

/**
 * Check if there are any pending tensors.
 */
export function hasPendingTensors(): boolean {
  return pendingTensorsByNodeId.size > 0;
}

/**
 * Get the set of node IDs that have pending tensors (live or recently disposed).
 * Used by fusion detection to avoid fusing across saved-for-backward boundaries.
 * Includes disposedPendingNodeIds so checkpoint's tidy() doesn't hide nodes.
 */
export function getPendingNodeIds(): Set<number> {
  const ids = new Set(pendingTensorsByNodeId.keys());
  for (const id of disposedPendingNodeIds) ids.add(id);
  return ids;
}

/**
 * Get node IDs with live (non-disposed) pending tensors only.
 * Unlike getPendingNodeIds(), excludes disposed tensors — their buffers can be
 * reused after their last in-plan consumer. Used by liveness-based buffer release.
 */
export function getLivePendingNodeIds(): Set<number> {
  return new Set(pendingTensorsByNodeId.keys());
}

/**
 * Live pending ROOT NODES (the LazyIRNodes that live, undisposed tensors
 * point at). Used by the executor's liveness analysis to walk the parts of
 * the pending graph OUTSIDE the current plan: a plan node consumed by such
 * an external root's closure must not be released or donated — a LATER plan
 * in the same step will read it (the foreach update graph splits across
 * plans; releasing its cross-plan intermediates was sound only by the
 * accident of pendingRelease fence gating, and donation made the
 * corruption deterministic).
 */
export function getLivePendingRootNodes(): object[] {
  const roots: object[] = [];
  for (const tensors of pendingTensorsByNodeId.values()) {
    for (const t of tensors) {
      const ref = (t as { lazyRef?: { kind?: string; node?: object } })
        .lazyRef;
      if (ref?.kind === "pending" && ref.node) {
        roots.push(ref.node);
        break; // one node object per registry entry
      }
    }
  }
  return roots;
}

/**
 * Clear the disposed pending node IDs set.
 * Called after plan execution to prevent unbounded growth.
 */
export function clearDisposedPendingNodeIds(): void {
  disposedPendingNodeIds.clear();
}

/** Debug: track live tensors to find leaks. Enable via Tensor._debugTracking = true. */
// biome-ignore lint/style/useLet: toggled at runtime
let _debugTracking = false;
const _debugLiveTensors = new Set<Tensor>();

/** Snapshot live tensor shapes for leak analysis. */
export function getDebugLiveTensors(): Array<{
  shape: string;
  dtype: string;
  materialized: boolean;
  disposed: boolean;
  creationSite?: string;
}> {
  const result: Array<{
    shape: string;
    dtype: string;
    materialized: boolean;
    disposed: boolean;
    creationSite?: string;
  }> = [];
  for (const t of _debugLiveTensors) {
    result.push({
      shape: t.shape.join("x") || "scalar",
      dtype: t.dtype,
      materialized: t.isMaterialized(),
      disposed: t.disposed,
      creationSite: t._debugCreationSite,
    });
  }
  return result;
}

export function setDebugTracking(enabled: boolean): void {
  _debugTracking = enabled;
  if (!enabled) _debugLiveTensors.clear();
}

export class Tensor {
  readonly baseId: BaseId;
  readonly device: DeviceKind;
  readonly shape: Shape;
  readonly dtype: DType;
  private _lazyRef: LazyRef;
  private _pendingNodeId: number | null = null;
  /** Mutable object for FinalizationRegistry - updated when tensor materializes */
  private readonly _held: { storageId: number; pendingNodeId: number | null };
  /** Debug: creation callsite (only when debug tracking is enabled) */
  _debugCreationSite?: string;

  constructor(
    baseId: BaseId,
    lazyRef: LazyRef,
    shape: number[],
    device: DeviceKind,
    dtype: DType = "f32",
  ) {
    this.baseId = baseId;
    this._lazyRef = lazyRef;
    this.device = device;
    this.shape = shape.slice();
    this.dtype = dtype;
    if (_debugTracking) {
      _debugLiveTensors.add(this);
      // Capture creation stack for leak tracing
      const stack = new Error().stack;
      if (stack) {
        // Skip "Error" + constructor frame, take next 6 useful frames
        this._debugCreationSite = stack
          .split("\n")
          .slice(2, 8)
          .map((l) => l.trim())
          .join(" | ");
      }
    }

    // Initialize held value for finalization registry
    // storageId = -1 means not yet materialized
    this._held = {
      storageId: lazyRef.kind === "materialized" ? lazyRef.storage.id : -1,
      pendingNodeId: lazyRef.kind === "pending" ? lazyRef.node.id : null,
    };

    // Register with FinalizationRegistry for automatic cleanup on GC
    tensorFinalizationRegistry.register(this, this._held, this);
    tensorCreatedCount++;

    // Register as pending if this tensor has a pending lazy ref
    if (lazyRef.kind === "pending") {
      this._pendingNodeId = lazyRef.node.id;
      registerPendingTensor(lazyRef.node.id, this);
    } else if (lazyRef.kind === "materialized") {
      rcRetain(lazyRef.storage.id, "tensor.ctor");
      storageTracker.trackTensor(lazyRef.storage.id, this);
    }
  }

  get lazyRef(): LazyRef {
    return this._lazyRef;
  }

  get backendTensor(): BackendTensor {
    if (!isMaterialized(this._lazyRef)) {
      throw new Error(
        "Tensor not materialized. Call cpu() or item() first to force execution.",
      );
    }
    return this._lazyRef.storage.backendTensor;
  }

  isMaterialized(): boolean {
    return isMaterialized(this._lazyRef);
  }

  /**
   * Materialize this tensor with the given storage.
   * Called by RuntimeEngine.force() after plan execution.
   */
  _materialize(storage: StorageHandle): void {
    // Defense-in-depth: never materialize a disposed tensor.
    // A disposed tensor should never mark a storage reachable, because
    // no one will later mark it unreachable (the tensor is already disposed).
    if (this._disposed) {
      // Still unregister from pending to clean up bookkeeping
      if (this._pendingNodeId !== null) {
        unregisterPendingTensor(this._pendingNodeId, this);
        this._pendingNodeId = null;
      }
      return;
    }
    // Unregister from pending tensors if we were pending
    if (this._pendingNodeId !== null) {
      unregisterPendingTensor(this._pendingNodeId, this);
      this._pendingNodeId = null;
    }
    this._lazyRef = { kind: "materialized", storage };
    // Update held value so FinalizationRegistry knows which storage to clean up
    this._held.storageId = storage.id;
    rcRetain(storage.id, "tensor.materialize");
    storageTracker.trackTensor(storage.id, this);
    tensorMaterializedCount++;
  }

  /**
   * Update this tensor's lazy ref to point to a new pending node.
   * Used by in-place operations (§4.3-4.4).
   */
  _updateLazyRef(newRef: LazyRef): void {
    // If the old ref was materialized, mark its storage as unreachable.
    // In-place operations (copy_, add_, etc.) create a new lazy node that
    // will produce a NEW storage when executed. The old storage is no longer
    // referenced by this tensor, so it must be marked unreachable to avoid
    // permanently orphaning it in externallyReachable.
    if (isMaterialized(this._lazyRef)) {
      rcRelease(this._lazyRef.storage.id, "tensor.updateRef");
      this._held.storageId = -1;
    }

    // Unregister from old pending node if any
    if (this._pendingNodeId !== null) {
      unregisterPendingTensor(this._pendingNodeId, this);
      this._pendingNodeId = null;
    }

    this._lazyRef = newRef;

    // Register for new pending node if applicable
    if (newRef.kind === "pending") {
      this._pendingNodeId = newRef.node.id;
      registerPendingTensor(newRef.node.id, this);
    }
  }

  toArray(): number[] {
    return this.backendTensor.toArray();
  }

  view(shape: number[]): Tensor {
    const backend = this.backendTensor as BackendTensor & {
      view?: (shape: Shape) => BackendTensor;
    };
    if (!backend.view) {
      throw new Error("view is not supported for this backend tensor");
    }
    const backendView = backend.view(shape);
    // View shares the same lazy ref (already materialized)
    return new Tensor(
      this.baseId,
      {
        kind: "materialized",
        storage: { id: -1, device: this.device, backendTensor: backendView },
      },
      backendView.shape,
      this.device,
      this.dtype,
    );
  }

  private _disposed = false;

  /**
   * Dispose this tensor and free GPU resources if applicable.
   * Safe to call multiple times.
   *
   * Per spec §1.4, disposal is cleanup-only and must respect allocator fencing.
   * GPU buffers are NOT destroyed immediately - they're marked for cleanup
   * at the next markStep() after GPU completion (allocator fencing).
   *
   * NOTE: Due to lazy execution, GPU work may not have been submitted yet
   * when dispose() is called. Destroying buffers immediately would cause
   * "buffer used while destroyed" errors. Destruction is deferred to markStep().
   */
  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    tensorDisposedCount++;
    if (_debugTracking) _debugLiveTensors.delete(this);

    // Unregister from FinalizationRegistry to prevent double cleanup
    tensorFinalizationRegistry.unregister(this);
    // Clear held value so finalization callback is a no-op if it runs anyway
    this._held.storageId = -1;

    // Unregister from pending tensors if still pending.
    // Also record the node ID in disposedPendingNodeIds so that
    // getPendingNodeIds() still includes it for fusion analysis.
    // Without this, checkpoint's tidy() would make nodes invisible
    // to the matmul epilogue detector, causing incorrect fusion.
    if (this._pendingNodeId !== null) {
      disposedPendingNodeIds.add(this._pendingNodeId);
      unregisterPendingTensor(this._pendingNodeId, this);
      this._pendingNodeId = null;
    }

    if (isMaterialized(this._lazyRef)) {
      const storage = this._lazyRef.storage;
      // Mark as unreachable - will be destroyed at next markStep() after GPU fence
      // Per §1.4: "Any reclamation of underlying buffers must respect allocator
      // fencing and in-flight plan retention (§14)"
      rcRelease(storage.id, "tensor.dispose");
      // Note: Do NOT destroy immediately - lazy execution means GPU work may
      // not have been submitted yet. Destruction is deferred to markStep().
    }
    // Note: If not materialized yet, there's no GPU buffer to free.
    // The lazy computation graph will be garbage collected.
  }

  /**
   * Check if this tensor has been disposed.
   */
  get disposed(): boolean {
    return this._disposed;
  }
}

let nextBaseId = 1;

// Debug counters for tensor creation/disposal
let tensorCreatedCount = 0;
let tensorDisposedCount = 0;
let tensorMaterializedCount = 0;

export function getTensorDebugStats() {
  return {
    created: tensorCreatedCount,
    disposed: tensorDisposedCount,
    materialized: tensorMaterializedCount,
  };
}

export function resetTensorDebugStats() {
  tensorCreatedCount = 0;
  tensorDisposedCount = 0;
  tensorMaterializedCount = 0;
}

export function createBaseId(): BaseId {
  return nextBaseId++;
}

export function resetBaseIdCounter(): void {
  nextBaseId = 1;
}
