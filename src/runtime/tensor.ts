import type { BackendTensor, DeviceKind, DType, Shape } from "../backend/types";
import type { LazyRef, StorageHandle } from "../engine/lazy";
import { isMaterialized, storageTracker } from "../engine/lazy";

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
  // Mark storage as unreachable when tensor is GC'd
  if (held.storageId >= 0) {
    storageTracker.markUnreachable(held.storageId);
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
): void {
  const tensors = pendingTensorsByNodeId.get(nodeId);
  if (tensors) {
    for (const tensor of tensors) {
      if (!tensor.isMaterialized() && !tensor.disposed) {
        tensor._materialize(storage);
      }
    }
    // Don't delete from map - tensors will unregister themselves
  }
}

/**
 * Get the number of pending tensors (for debugging/testing).
 */
function getPendingTensorCount(): number {
  let count = 0;
  for (const tensors of pendingTensorsByNodeId.values()) {
    count += tensors.size;
  }
  return count;
}

/**
 * Get all pending tensors (for forcing before cleanup).
 */
export function getAllPendingTensors(): Tensor[] {
  const result: Tensor[] = [];
  for (const tensors of pendingTensorsByNodeId.values()) {
    for (const tensor of tensors) {
      result.push(tensor);
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
 * Get the set of node IDs that have live pending tensors.
 * Used by fusion detection to avoid fusing across saved-for-backward boundaries.
 */
export function getPendingNodeIds(): Set<number> {
  return new Set(pendingTensorsByNodeId.keys());
}

export class Tensor {
  readonly baseId: BaseId;
  readonly device: DeviceKind;
  readonly shape: Shape;
  readonly dtype: DType;
  private _lazyRef: LazyRef;
  private _pendingNodeId: number | null = null;
  /** Mutable object for FinalizationRegistry - updated when tensor materializes */
  private readonly _held: { storageId: number };

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

    // Initialize held value for finalization registry
    // storageId = -1 means not yet materialized
    this._held = {
      storageId: lazyRef.kind === "materialized" ? lazyRef.storage.id : -1,
    };

    // Register with FinalizationRegistry for automatic cleanup on GC
    tensorFinalizationRegistry.register(this, this._held, this);
    tensorCreatedCount++;

    // Register as pending if this tensor has a pending lazy ref
    if (lazyRef.kind === "pending") {
      this._pendingNodeId = lazyRef.node.id;
      registerPendingTensor(lazyRef.node.id, this);
    } else if (lazyRef.kind === "materialized") {
      // Already materialized - mark storage as externally reachable
      storageTracker.markReachable(lazyRef.storage.id, this);
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
    // Mark storage as externally reachable (linked to user-visible Tensor)
    // Pass `this` so StorageTracker can track via WeakRef and detect GC'd tensors
    storageTracker.markReachable(storage.id, this);
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
      storageTracker.markUnreachable(this._lazyRef.storage.id);
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

    // Unregister from FinalizationRegistry to prevent double cleanup
    tensorFinalizationRegistry.unregister(this);
    // Clear held value so finalization callback is a no-op if it runs anyway
    this._held.storageId = -1;

    // Unregister from pending tensors if still pending
    if (this._pendingNodeId !== null) {
      unregisterPendingTensor(this._pendingNodeId, this);
      this._pendingNodeId = null;
    }

    if (isMaterialized(this._lazyRef)) {
      const storage = this._lazyRef.storage;
      // Mark as unreachable - will be destroyed at next markStep() after GPU fence
      // Per §1.4: "Any reclamation of underlying buffers must respect allocator
      // fencing and in-flight plan retention (§14)"
      storageTracker.markUnreachable(storage.id);
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
  return { created: tensorCreatedCount, disposed: tensorDisposedCount, materialized: tensorMaterializedCount };
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
