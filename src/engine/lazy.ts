import type {
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
  GeluOptions,
} from "../backend/types";
import { getBackend } from "../backend/registry";
import {
  beginBatchExecution,
  endBatchExecution,
  isBatchActive,
  abortBatch,
  flushBufferPool,
  flushSharedEncoder,
  beginSharedEncoder,
  endSharedEncoder,
  setCurrentOpLabel,
  setAdamBatchMode,
  setActiveArena,
  clearActiveArena,
  type BufferArena,
  startDispatchRecording,
  stopDispatchRecording,
  replayDispatches,
  type RecordedDispatch,
  setDispatchSequenceCounters,
  getDispatchSequenceCounters,
  addReplayPinnedBuffers,
  getArenaResolveIndex,
  setArenaResolveIndexTo,
} from "../backend/webgpu";
import { profileOpBegin, profileOpEnd, isProfilingEnabled, recordPlanAnalysis, type PlanAnalysis, setProfileModule, getProfileModule, recordFusionFallback } from "../backend/webgpu/profiler";
import type { Token } from "./tokens";
import {
  computePlanFingerprint,
  detectFusionGroups,
  groupToRecipe,
  hasFusionOpportunities,
  hasFusionPotential,
  isFusibleOp,
  reorderPlanForFusion,
  segmentPlanForExecution,
  type ExecutionSegment,
  type FusionGroup,
} from "./fusion-detect";
import {
  analyzeLifetimes,
  computeBufferSize,
  findDeadTensorsAtStep,
  type TensorLifetime,
} from "./memory-planning";
import {
  type LoweredPlan,
  LoweredPlanBuilder,
  isDataSourceOp,
  isViewOp,
  ENCODER_COPY_OPS,
  type DispatchReplayCache,
  type ReplayEntry,
  type ReplayNodeResult,
} from "./lowered-plan";

export type LazyOpCode =
  | "add"
  | "sub"
  | "mul"
  | "div"
  | "matmul"
  | "sqrt"
  | "relu"
  | "reshape"
  | "expand"
  | "transpose"
  | "permute"
  | "contiguous"
  | "gather"
  | "scatterAdd"
  | "sum"
  | "mean"
  | "max"
  | "argmax"
  | "argmin"
  | "gt"
  | "lt"
  | "ge"
  | "le"
  | "eq"
  | "ne"
  | "where"
  | "stridedScatterCopy"
  | "stridedScatterAdd"
  | "tensorFromArray"
  | "transfer"
  | "neg"
  | "abs"
  | "exp"
  | "log"
  | "tanh"
  | "sigmoid"
  | "gelu"
  | "silu"
  | "cast"
  | "pow"
  | "zeros"
  | "full"
  | "arange"
  | "tril"
  | "triu"
  | "isfinite"
  | "narrow"
  | "narrowBackward"
  | "adamStep"
  | "unscaleGrad"
  | "fusedAttentionForward"
  | "extractAttentionLogsumexp"
  | "fusedAttentionBackward"
  | "extractAttentionDK"
  | "extractAttentionDV"
  | "fusedCrossEntropyForward"
  | "fusedCrossEntropyBackward"
  | "fusedLayerNormForward"
  | "fusedLayerNormBackwardGradX"
  | "fusedLayerNormBackwardGradWeightBias"
  | "extractLnBwdGradBias";

/** GELU approximation type matching PyTorch's nn.GELU */
export type GeluApproximate = "none" | "tanh";

export interface StorageHandle {
  id: number;
  device: DeviceKind;
  backendTensor: BackendTensor;
  /** For views: ID of the storage that owns the buffer. Views don't destroy buffers. */
  baseStorageId?: number;
}

export interface LazyIRNode {
  id: number;
  op: LazyOpCode;
  inputs: LazyRef[];
  shape: number[];
  dtype: DType;
  device: DeviceKind;
  tokenIn?: Token;
  tokenOut?: Token;
  result?: StorageHandle;
  payload?: unknown;
  /** Module label for profiling (set via setProfileModule during graph construction) */
  module?: string;
  /**
   * Marks this node as a checkpoint boundary. When executing with segmented
   * checkpoint execution, the executor will flush the buffer pool after this
   * node completes, making released buffers available for subsequent segments.
   * This enables memory savings for large models that don't fit in GPU memory.
   */
  isCheckpointBoundary?: boolean;
}

export type LazyRef =
  | { kind: "pending"; node: LazyIRNode }
  | { kind: "materialized"; storage: StorageHandle }
  | { kind: "scalar"; value: number; dtype: DType };

export function createPendingRef(node: LazyIRNode): LazyRef {
  return { kind: "pending", node };
}

export function createMaterializedRef(storage: StorageHandle): LazyRef {
  return { kind: "materialized", storage };
}

export function createScalarRef(value: number, dtype: DType): LazyRef {
  return { kind: "scalar", value, dtype };
}

export function isPending(
  ref: LazyRef,
): ref is { kind: "pending"; node: LazyIRNode } {
  return ref.kind === "pending";
}

export function isMaterialized(
  ref: LazyRef,
): ref is { kind: "materialized"; storage: StorageHandle } {
  return ref.kind === "materialized";
}

export function isScalarRef(
  ref: LazyRef,
): ref is { kind: "scalar"; value: number; dtype: DType } {
  return ref.kind === "scalar";
}

export interface ExecutionPlan {
  nodes: LazyIRNode[];
}

/**
 * Options for plan execution.
 */
/** Ops safe to execute during tape replay fill-in: pure views + data sources. */
const FILL_IN_OPS: ReadonlySet<string> = new Set([
  // Pure view ops (no GPU dispatch, same buffer)
  "reshape", "transpose", "permute", "expand", "narrow", "contiguous",
  // Data source ops (create new buffers from host data)
  "tensorFromArray", "zeros", "full", "arange",
]);

export interface ExecutePlanOptions {
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
  /** Enable segmented execution at checkpoint boundaries */
  enableCheckpointSegmentation?: boolean;
  /** Skip nodes that already have a .result (from tape replay). */
  skipPreExecuted?: boolean;
  /** Only execute view/data-source ops; skip all compute ops. Used by tape replay fill-in. */
  viewOpsOnly?: boolean;
}

/**
 * Mark a LazyIRNode as a checkpoint boundary.
 * During segmented execution, the buffer pool will be flushed after this node.
 */
export function markAsCheckpointBoundary(node: LazyIRNode): void {
  node.isCheckpointBoundary = true;
}

/**
 * Segment a plan at checkpoint boundaries.
 * Returns an array of segments, where each segment ends at a checkpoint boundary
 * (or the end of the plan for the last segment).
 */
export function segmentPlanAtCheckpoints(
  plan: ExecutionPlan,
): ExecutionPlan[] {
  const segments: ExecutionPlan[] = [];
  let currentSegment: LazyIRNode[] = [];

  for (const node of plan.nodes) {
    currentSegment.push(node);
    if (node.isCheckpointBoundary) {
      segments.push({ nodes: currentSegment });
      currentSegment = [];
    }
  }

  // Add remaining nodes as final segment
  if (currentSegment.length > 0) {
    segments.push({ nodes: currentSegment });
  }

  return segments;
}

export function buildPlan(root: LazyIRNode): ExecutionPlan {
  const nodes: LazyIRNode[] = [];
  const visited = new Set<LazyIRNode>();

  const visit = (ref: LazyRef) => {
    if (ref.kind !== "pending") return;
    if (visited.has(ref.node)) return;
    visited.add(ref.node);

    for (const input of ref.node.inputs) {
      visit(input);
    }
    nodes.push(ref.node);
  };

  visit({ kind: "pending", node: root });

  return { nodes };
}

/**
 * Build an execution plan from multiple root nodes.
 * Used to merge recomputations from multiple checkpoint regions into a single plan.
 *
 * This enables unified backward execution where all checkpointed layers'
 * recomputations are executed in one plan with proper segmentation at
 * checkpoint boundaries.
 *
 * @param roots - Array of LazyIRNode roots to include in the plan
 * @returns A single ExecutionPlan containing all nodes from all roots
 */
export function buildMergedPlan(roots: LazyIRNode[], skipExecuted = false): ExecutionPlan {
  const nodes: LazyIRNode[] = [];
  // Use object identity for visited tracking instead of numeric IDs.
  // Node IDs can collide when resetNodeIdCounter() is called between
  // API instances (e.g., in tests), but lingering pending tensors from
  // a previous instance still reference old node objects. ID-based
  // deduplication would incorrectly merge distinct node objects.
  const visited = new Set<LazyIRNode>();

  const visit = (ref: LazyRef) => {
    if (ref.kind !== "pending") return;
    // Skip nodes that were already executed (have results from a previous force call).
    // Their results are accessible via ref.node.result in getInputStorage().
    if (skipExecuted && ref.node.result) return;
    if (visited.has(ref.node)) return;
    visited.add(ref.node);

    for (const input of ref.node.inputs) {
      visit(input);
    }
    nodes.push(ref.node);
  };

  for (const root of roots) {
    visit({ kind: "pending", node: root });
  }

  return { nodes };
}

// ============================================================================
// Lazy-initialized WebGPU imports (avoids circular deps; loaded once on first use)
// ============================================================================

let _webgpuMatmulImports: {
  dispatchMatmulWithEpilogue: typeof import("../backend/webgpu/index").dispatchMatmulWithEpilogue;
  dispatchMatmulDirect: typeof import("../backend/webgpu/index").dispatchMatmulDirect;
  deferredDestroyBuffer: typeof import("../backend/webgpu/index").deferredDestroyBuffer;
} | null = null;

let _webgpuMatmulGeomImports: {
  computeMatmulOutputShape: typeof import("../backend/webgpu/matmul/dispatch").computeMatmulOutputShape;
  computeBatchSize: typeof import("../backend/webgpu/matmul/dispatch").computeBatchSize;
  computeBatchStrides: typeof import("../backend/webgpu/matmul/dispatch").computeBatchStrides;
} | null = null;

async function ensureWebGPUMatmulImports() {
  if (!_webgpuMatmulImports) {
    const mod = await import("../backend/webgpu/index");
    _webgpuMatmulImports = {
      dispatchMatmulWithEpilogue: mod.dispatchMatmulWithEpilogue,
      dispatchMatmulDirect: mod.dispatchMatmulDirect,
      deferredDestroyBuffer: mod.deferredDestroyBuffer,
    };
  }
  if (!_webgpuMatmulGeomImports) {
    const mod = await import("../backend/webgpu/matmul/dispatch");
    _webgpuMatmulGeomImports = {
      computeMatmulOutputShape: mod.computeMatmulOutputShape,
      computeBatchSize: mod.computeBatchSize,
      computeBatchStrides: mod.computeBatchStrides,
    };
  }
}

let nextNodeId = 1;

export function resetNodeIdCounter(): void {
  nextNodeId = 1;
}

export function createLazyIRNode(
  op: LazyOpCode,
  inputs: LazyRef[],
  shape: number[],
  dtype: DType,
  device: DeviceKind,
  payload?: unknown,
): LazyIRNode {
  const node: LazyIRNode = {
    id: nextNodeId++,
    op,
    inputs,
    shape,
    dtype,
    device,
    payload,
  };
  // Capture module context for profiling (zero-cost when profiling disabled)
  const mod = getProfileModule();
  if (mod !== "unknown") node.module = mod;
  return node;
}

/** @deprecated use createLazyIRNode */
export const createIRNode = createLazyIRNode;

let nextStorageId = 1;

export function resetStorageIdCounter(): void {
  nextStorageId = 1;
}

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
    return nextStorageId;
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
      return {
        shape: (obj as any).shape,
        dtype: (obj as any).dtype,
        type: 'tensor',
        disposed: (obj as any).disposed ?? (obj as any)._disposed,
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

/**
 * Extract execution metadata from a plan for lifetime analysis.
 */
export function extractPlanMetadata(plan: ExecutionPlan): {
  nodeOrder: number[];
  nodeInputs: Map<number, number[]>;
  nodeSizes: Map<number, number>;
} {
  const nodeOrder: number[] = [];
  const nodeInputs = new Map<number, number[]>();
  const nodeSizes = new Map<number, number>();

  for (const node of plan.nodes) {
    nodeOrder.push(node.id);

    // Extract input node IDs (from pending refs)
    const inputIds: number[] = [];
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        inputIds.push(input.node.id);
      }
    }
    nodeInputs.set(node.id, inputIds);

    // Compute buffer size
    const bytesPerElement =
      node.dtype === "f32" || node.dtype === "i32"
        ? 4
        : node.dtype === "f16"
          ? 2
          : 1;
    const numElements = node.shape.reduce((a, b) => a * b, 1);
    nodeSizes.set(node.id, numElements * bytesPerElement);
  }

  return { nodeOrder, nodeInputs, nodeSizes };
}

export function createStorageHandle(
  device: DeviceKind,
  backendTensor: BackendTensor,
  baseStorageId?: number,
): StorageHandle {
  const storage: StorageHandle = {
    id: nextStorageId++,
    device,
    backendTensor,
    baseStorageId,
  };
  // Register in the global tracker
  storageTracker.register(storage);
  return storage;
}

function getInputStorage(ref: LazyRef, backend?: Backend): StorageHandle {
  if (ref.kind === "materialized") {
    return ref.storage;
  }
  if (ref.kind === "scalar") {
    // Materialize scalar ref on-the-fly for non-fused execution
    const b = backend ?? getBackend("cpu");
    if (!b) throw new Error("No backend available to materialize scalar ref");
    const bt = b.ops.tensorFromArray([ref.value], []);
    return createStorageHandle("cpu", bt);
  }
  if (ref.node.result) {
    return ref.node.result;
  }
  throw new Error(`Input not ready: node id=${ref.node.id} op=${ref.node.op} shape=[${ref.node.shape}] caller=${new Error().stack?.split("\n")[2]?.trim()}`);
}

/**
 * Extract matmul shapes from a plan for pre-tuning.
 * Returns array of [M, N, K] tuples.
 */
function extractMatmulShapes(plan: ExecutionPlan): Array<[number, number, number]> {
  const shapes: Array<[number, number, number]> = [];

  for (const node of plan.nodes) {
    if (node.op === "matmul") {
      // Get input shapes from the node inputs
      const aRef = node.inputs[0];
      const bRef = node.inputs[1];

      // Try to get shapes from materialized refs or pending nodes
      let aShape: number[] | undefined;
      let bShape: number[] | undefined;

      if (aRef.kind === "materialized") {
        aShape = aRef.storage.backendTensor.shape;
      } else if (aRef.kind === "pending" && aRef.node.shape) {
        aShape = aRef.node.shape;
      }

      if (bRef.kind === "materialized") {
        bShape = bRef.storage.backendTensor.shape;
      } else if (bRef.kind === "pending" && bRef.node.shape) {
        bShape = bRef.node.shape;
      }

      if (aShape && bShape && aShape.length >= 2 && bShape.length >= 2) {
        // Extract M, K from A (last two dims)
        const m = aShape[aShape.length - 2];
        const k = aShape[aShape.length - 1];
        // Extract N from B (last dim)
        const n = bShape[bShape.length - 1];
        shapes.push([m, n, k]);
      }
    }
  }

  return shapes;
}

/** Cache of already-pretuned matmul shape signatures. */
const pretunedShapeSignatures = new Set<string>();

/**
 * Pre-tune matmul shapes before plan execution.
 * Only runs if the backend supports pretuning.
 * Caches tuned shape signatures to skip redundant pretune calls on subsequent steps.
 */
export async function pretunePlanMatmuls(
  plan: ExecutionPlan,
  backend: Backend,
): Promise<void> {
  if (!backend.pretuneMatmulShapes) {
    return;
  }

  const shapes = extractMatmulShapes(plan);
  // Filter to only shapes not yet pretuned
  const untunedShapes = shapes.filter(s => {
    const sig = `${s[0]},${s[1]},${s[2]}`;
    return !pretunedShapeSignatures.has(sig);
  });
  if (untunedShapes.length === 0) return;

  await backend.pretuneMatmulShapes(untunedShapes);
  for (const s of untunedShapes) {
    pretunedShapeSignatures.add(`${s[0]},${s[1]},${s[2]}`);
  }
}

export async function executePlan(
  plan: ExecutionPlan,
  backend: Backend,
  options?: ExecutePlanOptions,
): Promise<StorageHandle> {
  if (plan.nodes.length === 0) {
    throw new Error("Cannot execute empty plan");
  }

  // Pre-tune matmul shapes if backend supports it
  await pretunePlanMatmuls(plan, backend);

  // Set up lifetime analysis if early release is enabled
  let lifetimes: Map<number, TensorLifetime> | null = null;
  let outputNodeIds: Set<number> | null = null;
  const alreadyReleased = new Set<number>();
  const nodeToStorage = new Map<number, StorageHandle>();

  if (options?.enableEarlyRelease) {
    const { nodeOrder, nodeInputs, nodeSizes } = extractPlanMetadata(plan);
    const lastNodeId = plan.nodes[plan.nodes.length - 1].id;
    outputNodeIds = new Set([lastNodeId]);
    // Protect externally-referenced nodes (saved for backward, user-held tensors)
    // from early release — later plans need their buffers intact.
    try {
      const { getPendingNodeIds } = await import("../runtime/tensor");
      for (const id of getPendingNodeIds()) outputNodeIds.add(id);
    } catch { /* runtime/tensor not available */ }
    lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, outputNodeIds, nodeSizes);
  }

  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {

  const viewOnly = options?.viewOpsOnly === true;

  for (let step = 0; step < plan.nodes.length; step++) {
    const node = plan.nodes[step];

    // Skip nodes that already have results (from a prior plan execution within this step).
    if (node.result) continue;

    // In view-only mode (tape replay fill-in), skip compute ops.
    // Only view ops and data-source ops need execution; compute ops are
    // either already handled by replay or are intermediate fused nodes.
    if (viewOnly && !FILL_IN_OPS.has(node.op)) continue;

    // For multi-device graphs, use the node's device backend
    const nodeBackend = getBackend(node.device) ?? backend;

    const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
    const backendInputs = inputs.map((s) => s.backendTensor);

    let resultTensor: BackendTensor;

    setCurrentOpLabel(node.op);
    setProfileModule(node.module ?? "unknown");
    const _profT0 = profileOpBegin(node.op);

    switch (node.op) {
      case "tensorFromArray": {
        const payload = node.payload as { values: number[] } | undefined;
        if (!payload?.values) {
          throw new Error("tensorFromArray requires values in payload");
        }
        resultTensor = nodeBackend.ops.tensorFromArray(
          payload.values,
          node.shape,
        );
        break;
      }

      case "zeros": {
        if (nodeBackend.ops.zeros) {
          resultTensor = nodeBackend.ops.zeros(node.shape);
        } else {
          // Fallback for backends without dedicated zeros op
          const numElements = node.shape.reduce((a, b) => a * b, 1);
          resultTensor = nodeBackend.ops.tensorFromArray(
            new Array(numElements).fill(0),
            node.shape,
          );
        }
        break;
      }

      case "full": {
        const fullPayload = node.payload as { fillValue: number };
        if (nodeBackend.ops.full) {
          resultTensor = nodeBackend.ops.full(node.shape, fullPayload.fillValue);
        } else {
          // Fallback: create array filled with value
          const numElements = node.shape.reduce((a, b) => a * b, 1);
          resultTensor = nodeBackend.ops.tensorFromArray(
            new Array(numElements).fill(fullPayload.fillValue),
            node.shape,
          );
        }
        break;
      }

      case "arange": {
        const arangePayload = node.payload as { end: number; start: number; step: number };
        if (nodeBackend.ops.arange) {
          resultTensor = nodeBackend.ops.arange(arangePayload.end, arangePayload.start, arangePayload.step);
        } else {
          // Fallback: create array on CPU
          const n = Math.max(0, Math.ceil((arangePayload.end - arangePayload.start) / arangePayload.step));
          const vals = new Array(n);
          for (let i = 0; i < n; i++) vals[i] = arangePayload.start + i * arangePayload.step;
          resultTensor = nodeBackend.ops.tensorFromArray(vals, node.shape);
        }
        break;
      }

      case "tril": {
        if (!nodeBackend.ops.tril) throw new Error("tril not supported by backend");
        const trilK = (node.payload as { k: number })?.k ?? 0;
        resultTensor = nodeBackend.ops.tril(backendInputs[0], trilK);
        break;
      }

      case "triu": {
        if (!nodeBackend.ops.triu) throw new Error("triu not supported by backend");
        const triuK = (node.payload as { k: number })?.k ?? 0;
        resultTensor = nodeBackend.ops.triu(backendInputs[0], triuK);
        break;
      }

      case "add":
        resultTensor = nodeBackend.ops.add(backendInputs[0], backendInputs[1]);
        break;

      case "sub": {
        const subPayload = node.payload as { alpha?: number } | undefined;
        resultTensor = nodeBackend.ops.sub(
          backendInputs[0],
          backendInputs[1],
          subPayload,
        );
        break;
      }

      case "mul":
        resultTensor = nodeBackend.ops.mul(backendInputs[0], backendInputs[1]);
        break;

      case "div": {
        const divPayload = node.payload as
          | { roundingMode?: "trunc" | "floor" }
          | undefined;
        resultTensor = nodeBackend.ops.div(
          backendInputs[0],
          backendInputs[1],
          divPayload,
        );
        break;
      }

      case "matmul":
        resultTensor = nodeBackend.ops.matmul(
          backendInputs[0],
          backendInputs[1],
        );
        break;

      case "sqrt":
        resultTensor = nodeBackend.ops.sqrt(backendInputs[0]);
        break;

      case "relu":
        resultTensor = nodeBackend.ops.relu(backendInputs[0]);
        break;

      case "exp":
        if (!nodeBackend.ops.exp)
          throw new Error("exp not supported by backend");
        resultTensor = nodeBackend.ops.exp(backendInputs[0]);
        break;

      case "log":
        if (!nodeBackend.ops.log)
          throw new Error("log not supported by backend");
        resultTensor = nodeBackend.ops.log(backendInputs[0]);
        break;

      case "neg":
        if (!nodeBackend.ops.neg)
          throw new Error("neg not supported by backend");
        resultTensor = nodeBackend.ops.neg(backendInputs[0]);
        break;

      case "abs":
        if (!nodeBackend.ops.abs)
          throw new Error("abs not supported by backend");
        resultTensor = nodeBackend.ops.abs(backendInputs[0]);
        break;

      case "tanh":
        if (!nodeBackend.ops.tanh)
          throw new Error("tanh not supported by backend");
        resultTensor = nodeBackend.ops.tanh(backendInputs[0]);
        break;

      case "sigmoid":
        if (!nodeBackend.ops.sigmoid)
          throw new Error("sigmoid not supported by backend");
        resultTensor = nodeBackend.ops.sigmoid(backendInputs[0]);
        break;

      case "gelu": {
        if (!nodeBackend.ops.gelu)
          throw new Error("gelu not supported by backend");
        const geluOpts = node.payload as GeluOptions | undefined;
        resultTensor = nodeBackend.ops.gelu(backendInputs[0], geluOpts);
        break;
      }

      case "silu":
        if (!nodeBackend.ops.silu)
          throw new Error("silu not supported by backend");
        resultTensor = nodeBackend.ops.silu(backendInputs[0]);
        break;

      case "isfinite":
        if (!nodeBackend.ops.isfinite)
          throw new Error("isfinite not supported by backend");
        resultTensor = nodeBackend.ops.isfinite(backendInputs[0]);
        break;

      case "reshape": {
        const payload = node.payload as { targetShape: number[] } | undefined;
        const targetShape = payload?.targetShape ?? node.shape;
        resultTensor = nodeBackend.ops.reshape(backendInputs[0], targetShape);
        break;
      }

      case "expand":
        resultTensor = nodeBackend.ops.expand(backendInputs[0], node.shape);
        break;

      case "transpose": {
        const payload = node.payload as
          | { dim0: number; dim1: number }
          | undefined;
        if (!payload) {
          throw new Error("transpose requires dim0 and dim1 in payload");
        }
        resultTensor = nodeBackend.ops.transpose(backendInputs[0], payload);
        break;
      }

      case "permute": {
        const payload = node.payload as { dims: number[] } | undefined;
        if (!payload) {
          throw new Error("permute requires dims in payload");
        }
        resultTensor = nodeBackend.ops.permute(backendInputs[0], payload.dims);
        break;
      }

      case "contiguous":
        resultTensor = nodeBackend.ops.contiguous(backendInputs[0]);
        break;

      case "narrow": {
        const p = node.payload as { dim: number; start: number; length: number };
        if (!nodeBackend.ops.narrow) throw new Error("narrow not supported by backend");
        resultTensor = nodeBackend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
        break;
      }

      case "narrowBackward": {
        const p = node.payload as { dim: number; start: number; originalLength: number };
        if (!nodeBackend.ops.narrowBackward) throw new Error("narrowBackward not supported by backend");
        resultTensor = nodeBackend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
        break;
      }

      case "cast": {
        const payload = node.payload as
          | { dtype: import("../backend/types").DType }
          | undefined;
        if (!payload) {
          throw new Error("cast requires dtype in payload");
        }
        if (!nodeBackend.ops.cast) {
          throw new Error("cast not supported by backend");
        }
        resultTensor = nodeBackend.ops.cast(backendInputs[0], payload.dtype);
        break;
      }

      case "gather": {
        const payload = node.payload as { dim: number } | undefined;
        if (!payload) {
          throw new Error("gather requires dim in payload");
        }
        resultTensor = nodeBackend.ops.gather(
          backendInputs[0],
          backendInputs[1],
          payload,
        );
        break;
      }

      case "scatterAdd": {
        const payload = node.payload as { dim: number } | undefined;
        if (!payload) {
          throw new Error("scatterAdd requires dim in payload");
        }
        resultTensor = nodeBackend.ops.scatterAdd(
          backendInputs[0],
          backendInputs[1],
          backendInputs[2],
          payload,
        );
        break;
      }

      case "sum": {
        const payload = node.payload as
          | { dim?: number | number[] | null; keepdim?: boolean }
          | undefined;
        resultTensor = nodeBackend.ops.sum(backendInputs[0], payload);
        break;
      }

      case "max": {
        const payload = node.payload as
          | { dim?: number | number[] | null; keepdim?: boolean }
          | undefined;
        resultTensor = nodeBackend.ops.max(backendInputs[0], payload);
        break;
      }

      case "mean": {
        const payload = node.payload as
          | { dim?: number | number[] | null; keepdim?: boolean }
          | undefined;
        resultTensor = nodeBackend.ops.mean(backendInputs[0], payload);
        break;
      }

      case "argmax": {
        const payload = node.payload as { dim: number; keepdim?: boolean };
        if (!nodeBackend.ops.argmax)
          throw new Error("argmax not supported by backend");
        resultTensor = nodeBackend.ops.argmax(backendInputs[0], payload);
        break;
      }

      case "argmin": {
        const payload = node.payload as { dim: number; keepdim?: boolean };
        if (!nodeBackend.ops.argmin)
          throw new Error("argmin not supported by backend");
        resultTensor = nodeBackend.ops.argmin(backendInputs[0], payload);
        break;
      }

      case "gt":
        if (!nodeBackend.ops.gt) throw new Error("gt not supported by backend");
        resultTensor = nodeBackend.ops.gt(backendInputs[0], backendInputs[1]);
        break;

      case "lt":
        if (!nodeBackend.ops.lt) throw new Error("lt not supported by backend");
        resultTensor = nodeBackend.ops.lt(backendInputs[0], backendInputs[1]);
        break;

      case "ge":
        if (!nodeBackend.ops.ge) throw new Error("ge not supported by backend");
        resultTensor = nodeBackend.ops.ge(backendInputs[0], backendInputs[1]);
        break;

      case "le":
        if (!nodeBackend.ops.le) throw new Error("le not supported by backend");
        resultTensor = nodeBackend.ops.le(backendInputs[0], backendInputs[1]);
        break;

      case "eq":
        if (!nodeBackend.ops.eq) throw new Error("eq not supported by backend");
        resultTensor = nodeBackend.ops.eq(backendInputs[0], backendInputs[1]);
        break;

      case "ne":
        if (!nodeBackend.ops.ne) throw new Error("ne not supported by backend");
        resultTensor = nodeBackend.ops.ne(backendInputs[0], backendInputs[1]);
        break;

      case "where":
        resultTensor = nodeBackend.ops.where(
          backendInputs[0],
          backendInputs[1],
          backendInputs[2],
        );
        break;

      case "stridedScatterCopy": {
        const payload = node.payload as {
          offset: number;
          viewShape: number[];
          viewStrides: number[];
        };
        if (!payload) {
          throw new Error("stridedScatterCopy requires options in payload");
        }
        resultTensor = nodeBackend.ops.stridedScatterCopy(
          backendInputs[0],
          backendInputs[1],
          payload,
        );
        break;
      }

      case "stridedScatterAdd": {
        const payload = node.payload as {
          offset: number;
          viewShape: number[];
          viewStrides: number[];
        };
        if (!payload) {
          throw new Error("stridedScatterAdd requires options in payload");
        }
        resultTensor = nodeBackend.ops.stridedScatterAdd(
          backendInputs[0],
          backendInputs[1],
          payload,
        );
        break;
      }

      case "adamStep": {
        const adamPayload = node.payload as import("../backend/types").AdamStepConfig;
        if (!nodeBackend.ops.adamStep) {
          throw new Error("adamStep not supported by backend");
        }
        const adamResult = await nodeBackend.ops.adamStep(
          backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3],
          adamPayload,
        );
        resultTensor = adamResult.param;
        // Wrap side outputs in StorageHandles so they're tracked by the storage system
        // and won't be destroyed prematurely by the buffer pool.
        // Mark them as externally reachable so forceAllPending's destroyUnreachableSince
        // won't reclaim them before the optimizer extracts them on the next step.
        // Pass the sideOutputs object as the WeakRef target so destroyUnreachable()
        // can detect orphaned side outputs via GC (the node holds _adamSideOutputs
        // alive, and Adam._pendingNodes holds the node alive until _resolvePendingState).
        const mStorage = createStorageHandle(node.device, adamResult.m);
        const vStorage = createStorageHandle(node.device, adamResult.v);
        const sideOutputs = { m: mStorage, v: vStorage };
        storageTracker.markReachable(mStorage.id, sideOutputs);
        storageTracker.markReachable(vStorage.id, sideOutputs);
        (node as any)._adamSideOutputs = sideOutputs;
        break;
      }

      case "unscaleGrad": {
        const unscalePayload = node.payload as { invScale: number; infFlagBuffer: unknown };
        if (!nodeBackend.ops.unscaleGrad) throw new Error("unscaleGrad not supported by backend");
        resultTensor = nodeBackend.ops.unscaleGrad(
          backendInputs[0], unscalePayload.invScale, unscalePayload.infFlagBuffer,
        );
        break;
      }

      case "fusedAttentionForward": {
        const faPayload = node.payload as import("../backend/types").FusedAttentionConfig;
        if (!nodeBackend.ops.fusedAttentionForward) throw new Error("fusedAttentionForward not supported by backend");
        const faResult = nodeBackend.ops.fusedAttentionForward(
          backendInputs[0], backendInputs[1], backendInputs[2], faPayload,
        );
        resultTensor = faResult.output;
        // Wrap logsumexp in a StorageHandle so destroyUnreachable() can clean it up
        // if extractAttentionLogsumexp never executes (e.g., checkpoint disposes the
        // consumer before the plan runs). Without this, the raw GPUBuffer leaks because
        // allocateOutputBuffer called trackAllocation but nothing calls trackDeallocation.
        const lseSH = createStorageHandle(node.device, faResult.logsumexp);
        (node as any)._attnSideOutput = lseSH;
        break;
      }

      case "extractAttentionLogsumexp": {
        const parentNodeFA = node.inputs[0].node;
        const lseSH = (parentNodeFA as any)._attnSideOutput as StorageHandle | undefined;
        if (!lseSH) {
          throw new Error("extractAttentionLogsumexp: parent node has no _attnSideOutput");
        }
        resultTensor = lseSH.backendTensor;
        // Unregister the StorageHandle — the engine will create a new one for this node's output
        storageTracker.unregister(lseSH.id);
        (parentNodeFA as any)._attnSideOutput = undefined;
        break;
      }

      case "fusedAttentionBackward": {
        const faBwdPayload = node.payload as import("../backend/types").FusedAttentionConfig;
        if (!nodeBackend.ops.fusedAttentionBackward) throw new Error("fusedAttentionBackward not supported by backend");
        const faBwdResult = nodeBackend.ops.fusedAttentionBackward(
          backendInputs[0], backendInputs[1], backendInputs[2],
          backendInputs[3], backendInputs[4], backendInputs[5], faBwdPayload,
        );
        resultTensor = faBwdResult.dQ;
        // Wrap dK/dV in StorageHandles for proper cleanup (same reason as logsumexp above)
        const dkSH = createStorageHandle(node.device, faBwdResult.dK);
        const dvSH = createStorageHandle(node.device, faBwdResult.dV);
        (node as any)._attnBwdDK = dkSH;
        (node as any)._attnBwdDV = dvSH;
        break;
      }

      case "extractAttentionDK": {
        const parentNodeDK = node.inputs[0].node;
        const dkSH = (parentNodeDK as any)._attnBwdDK as StorageHandle | undefined;
        if (!dkSH) {
          throw new Error("extractAttentionDK: parent node has no _attnBwdDK");
        }
        resultTensor = dkSH.backendTensor;
        storageTracker.unregister(dkSH.id);
        (parentNodeDK as any)._attnBwdDK = undefined;
        break;
      }

      case "extractAttentionDV": {
        const parentNodeDV = node.inputs[0].node;
        const dvSH = (parentNodeDV as any)._attnBwdDV as StorageHandle | undefined;
        if (!dvSH) {
          throw new Error("extractAttentionDV: parent node has no _attnBwdDV");
        }
        resultTensor = dvSH.backendTensor;
        storageTracker.unregister(dvSH.id);
        (parentNodeDV as any)._attnBwdDV = undefined;
        break;
      }

      case "fusedCrossEntropyForward": {
        const cePayload = node.payload as import("../backend/types").FusedCrossEntropyConfig;
        if (!nodeBackend.ops.fusedCrossEntropyForward) throw new Error("fusedCrossEntropyForward not supported by backend");
        resultTensor = nodeBackend.ops.fusedCrossEntropyForward(
          backendInputs[0], backendInputs[1], cePayload,
        );
        break;
      }

      case "fusedCrossEntropyBackward": {
        const cePayload2 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
        if (!nodeBackend.ops.fusedCrossEntropyBackward) throw new Error("fusedCrossEntropyBackward not supported by backend");
        resultTensor = nodeBackend.ops.fusedCrossEntropyBackward(
          backendInputs[0], backendInputs[1], backendInputs[2], cePayload2,
        );
        break;
      }

      case "fusedLayerNormForward": {
        const lnPayload = node.payload as import("../backend/types").FusedLayerNormConfig;
        if (!nodeBackend.ops.fusedLayerNormForward) throw new Error("fusedLayerNormForward not supported by backend");
        resultTensor = nodeBackend.ops.fusedLayerNormForward(
          backendInputs[0], backendInputs[1], backendInputs[2], lnPayload,
        );
        break;
      }

      case "fusedLayerNormBackwardGradX": {
        const lnPayload2 = node.payload as import("../backend/types").FusedLayerNormConfig;
        if (!nodeBackend.ops.fusedLayerNormBackwardGradX) throw new Error("fusedLayerNormBackwardGradX not supported by backend");
        resultTensor = nodeBackend.ops.fusedLayerNormBackwardGradX(
          backendInputs[0], backendInputs[1], backendInputs[2], lnPayload2,
        );
        break;
      }

      case "fusedLayerNormBackwardGradWeightBias": {
        const lnGWBPayload = node.payload as import("../backend/types").FusedLayerNormConfig;
        if (!nodeBackend.ops.fusedLayerNormBackwardGradWeightBias) {
          throw new Error("fusedLayerNormBackwardGradWeightBias not supported by backend");
        }
        const gwResult = nodeBackend.ops.fusedLayerNormBackwardGradWeightBias(
          backendInputs[0], backendInputs[1], lnGWBPayload,
        );
        resultTensor = gwResult.gradWeight;
        // Store raw gradBias BackendTensor for extractLnBwdGradBias
        (node as any)._lnBwdSideOutput = gwResult.gradBias;
        break;
      }

      case "extractLnBwdGradBias": {
        // Input[0] is the gradWeight tensor from fusedLayerNormBackwardGradWeightBias
        const parentNode = node.inputs[0].node;
        const sideOutputBT = (parentNode as any)._lnBwdSideOutput as BackendTensor | undefined;
        if (!sideOutputBT) {
          throw new Error("extractLnBwdGradBias: parent node has no _lnBwdSideOutput");
        }
        resultTensor = sideOutputBT;
        (parentNode as any)._lnBwdSideOutput = undefined;
        break;
      }

      case "transfer": {
        // Transfer from source device to target device
        const sourceStorage = inputs[0];
        const targetDevice = node.device;
        const sourceDevice = sourceStorage.device;

        if (sourceDevice === targetDevice) {
          // No transfer needed
          resultTensor = sourceStorage.backendTensor;
        } else {
          // Get target backend and transfer via CPU
          const targetBackend = getBackend(targetDevice);
          if (!targetBackend) {
            throw new Error(
              `Transfer failed: backend not available for ${targetDevice}`,
            );
          }

          // Read from source, create on target
          const sourceBackend = getBackend(sourceDevice);
          if (!sourceBackend) {
            throw new Error(
              `Transfer failed: backend not available for ${sourceDevice}`,
            );
          }

          const values = await sourceBackend.ops.read(
            sourceStorage.backendTensor,
          );
          resultTensor = targetBackend.ops.tensorFromArray(values, node.shape);
        }
        break;
      }

      default:
        throw new Error(`Unknown op: ${node.op}`);
    }

    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");

    // Safety: if a backend op returned the exact same tensor object as one of
    // its inputs (e.g. contiguous on an already-contiguous tensor), creating a
    // separate owning StorageHandle would double-free the underlying buffer.
    // Detect this and wrap the result as a non-owning view.
    const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
    if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
      // Clone the tensor object so we don't mutate the input's ownsBuffer field.
      // Only applies to backends with explicit ownsBuffer (e.g. WebGPU).
      resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
    }
    const isView =
      (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
    const baseStorageId =
      isView && inputs.length > 0
        ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
        : undefined;
    node.result = createStorageHandle(node.device, resultTensor, baseStorageId);

    // Track storage for early release
    if (options?.enableEarlyRelease) {
      nodeToStorage.set(node.id, node.result);

      // Release dead buffers after each step
      if (lifetimes && outputNodeIds) {
        const deadNodeIds = findDeadTensorsAtStep(
          lifetimes,
          step + 1, // We just completed this step
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
    }
  }

  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }

  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Execution failed: no result for last node");
  }

  // Clear results for nodes whose buffers were destroyed by early release.
  // Later plans skip nodes with results (if (node.result) continue), so stale
  // results pointing to destroyed buffers would cause silent data corruption.
  if (alreadyReleased.size > 0) {
    for (const node of plan.nodes) {
      if (alreadyReleased.has(node.id)) {
        node.result = undefined;
      }
    }
  }

  return lastNode.result;
}

// ============================================================================
// Segmented Execution for Checkpointing
// ============================================================================

/**
 * Execute a plan with segmentation at checkpoint boundaries.
 *
 * This enables memory savings for large models by:
 * 1. Executing each segment (up to a checkpoint boundary)
 * 2. Flushing the buffer pool after each segment
 * 3. Making released buffers available for subsequent segments
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Execution options
 * @param flushBufferPool - Callback to flush the buffer pool (backend-specific)
 */
export async function executePlanWithCheckpointSegments(
  plan: ExecutionPlan,
  backend: Backend,
  options: ExecutePlanOptions | undefined,
  flushBufferPool: () => void,
): Promise<StorageHandle> {
  // Check if plan has any checkpoint boundaries
  const hasCheckpointBoundaries = plan.nodes.some((n) => n.isCheckpointBoundary);

  if (!hasCheckpointBoundaries) {
    // No segmentation needed - use regular execution
    return executePlan(plan, backend, options);
  }

  // Segment the plan at checkpoint boundaries
  const segments = segmentPlanAtCheckpoints(plan);

  if (segments.length === 1) {
    // Only one segment - use regular execution
    return executePlan(plan, backend, options);
  }

  // Execute each segment, flushing buffers between them
  let lastResult: StorageHandle | null = null;

  // Track all materialized storages across segments
  const materializedStorages = new Map<number, StorageHandle>();

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];

    // Clone node inputs before mutation to preserve plan immutability.
    // Nodes are shared across steps; mutating inputs in-place would cause
    // stale storage refs to accumulate, producing NaN after ~25 steps.
    for (const node of segment.nodes) {
      let needsClone = false;
      for (let j = 0; j < node.inputs.length; j++) {
        const input = node.inputs[j];
        if (input.kind === "pending") {
          const materialized = materializedStorages.get(input.node.id);
          if (materialized) {
            if (!needsClone) {
              node.inputs = [...node.inputs];
              needsClone = true;
            }
            node.inputs[j] = { kind: "materialized", storage: materialized };
          }
        }
      }
    }

    // Execute this segment
    lastResult = await executePlan(segment, backend, options);

    // Track all materialized results from this segment
    for (const node of segment.nodes) {
      if (node.result) {
        materializedStorages.set(node.id, node.result);
      }
    }

    // Flush buffer pool after each segment (except the last)
    // This makes released buffers available for the next segment
    if (i < segments.length - 1) {
      flushBufferPool();
    }
  }

  if (!lastResult) {
    throw new Error("Segmented execution failed: no result");
  }

  return lastResult;
}

// ============================================================================
// True Segmented Execution with GPU Synchronization
// ============================================================================

/**
 * Execute a plan with true segmented execution using GPU synchronization.
 *
 * Unlike executePlanWithCheckpointSegments which just flushes the buffer pool,
 * this version:
 * 1. Batches all ops in a segment into a single command buffer
 * 2. Submits and waits for GPU completion between segments
 * 3. Actually frees GPU memory before next segment starts
 *
 * This enables running models that wouldn't fit in GPU memory when using
 * checkpoint-based training.
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Execution options
 */
export async function executePlanWithTrueSegments(
  plan: ExecutionPlan,
  backend: Backend,
  options?: ExecutePlanOptions,
): Promise<StorageHandle> {
  // Check if plan has any checkpoint boundaries
  const hasCheckpointBoundaries = plan.nodes.some((n) => n.isCheckpointBoundary);

  if (!hasCheckpointBoundaries) {
    // No segmentation needed - use regular execution
    return executePlan(plan, backend, options);
  }

  // Segment the plan at checkpoint boundaries
  const segments = segmentPlanAtCheckpoints(plan);

  if (segments.length === 1) {
    // Only one segment - use regular execution
    return executePlan(plan, backend, options);
  }

  // Track cross-segment data flow
  const materializedStorages = new Map<number, StorageHandle>();
  let lastResult: StorageHandle | null = null;
  const finalOutputId = plan.nodes[plan.nodes.length - 1].id;

  for (let segIdx = 0; segIdx < segments.length; segIdx++) {
    const segment = segments[segIdx];
    const isLastSegment = segIdx === segments.length - 1;

    // Find outputs needed by later segments
    const survivingNodeIds = findSurvivingOutputs(
      segment,
      segments.slice(segIdx + 1),
      finalOutputId,
    );

    // Clone node inputs before mutation to preserve plan immutability.
    for (const node of segment.nodes) {
      let needsClone = false;
      for (let j = 0; j < node.inputs.length; j++) {
        const input = node.inputs[j];
        if (input.kind === "pending") {
          const materialized = materializedStorages.get(input.node.id);
          if (materialized) {
            if (!needsClone) {
              node.inputs = [...node.inputs];
              needsClone = true;
            }
            node.inputs[j] = { kind: "materialized", storage: materialized };
          }
        }
      }
    }

    // Begin batched execution - all ops encode to shared command buffer
    beginBatchExecution();

    try {
      const nodeToStorage = new Map<number, StorageHandle>();

      // Execute all ops in segment (encode to shared encoder, no GPU submit yet)
      for (const node of segment.nodes) {
        const nodeBackend = getBackend(node.device) ?? backend;
        const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
        const backendInputs = inputs.map((s) => s.backendTensor);

        let resultTensor = await executeOpInternal(
          node,
          backendInputs,
          nodeBackend,
        );

        // Safety: detect aliased result (e.g. contiguous on contiguous tensor)
        const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
        if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
          resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
        }
        const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
        const baseStorageId = isView && inputs.length > 0
          ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
          : undefined;
        node.result = createStorageHandleInternal(node.device, resultTensor, baseStorageId);
        nodeToStorage.set(node.id, node.result);
        materializedStorages.set(node.id, node.result);
      }

      // End batch - submits command buffer and WAITS for GPU completion
      await endBatchExecution();

      // NOW safe to release dead buffers (GPU work is complete)
      if (!isLastSegment) {
        for (const node of segment.nodes) {
          if (!survivingNodeIds.has(node.id)) {
            const storage = nodeToStorage.get(node.id);
            if (storage && canSafelyRelease(storage, nodeToStorage)) {
              releaseBufferImmediate(storage);
              nodeToStorage.delete(node.id);
              materializedStorages.delete(node.id);
            }
          }
        }

        // Flush buffer pool - buffers now available for next segment
        flushBufferPool();
      }

      lastResult = segment.nodes[segment.nodes.length - 1].result!;

    } catch (error) {
      // Clean up batch on error
      if (isBatchActive()) {
        abortBatch();
      }
      throw error;
    }
  }

  if (!lastResult) {
    throw new Error("True segmented execution failed: no result");
  }

  return lastResult;
}

/**
 * Find node IDs that must survive this segment (used by later segments).
 */
function findSurvivingOutputs(
  segment: ExecutionPlan,
  laterSegments: ExecutionPlan[],
  finalOutputId: number,
): Set<number> {
  const surviving = new Set<number>();
  surviving.add(finalOutputId);

  // Find all nodes from this segment that are used as inputs in later segments
  for (const laterSegment of laterSegments) {
    for (const node of laterSegment.nodes) {
      for (const input of node.inputs) {
        if (input.kind === "pending") {
          surviving.add(input.node.id);
        }
      }
    }
  }

  return surviving;
}

/**
 * Internal helper to execute a single op.
 * This is a subset of the switch statement in executePlan, factored out for reuse.
 */
async function executeOpInternal(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  nodeBackend: Backend,
): Promise<BackendTensor> {
  setCurrentOpLabel(node.op);
  setProfileModule(node.module ?? "unknown");
  const _profT0 = profileOpBegin(node.op);
  try {
  switch (node.op) {
    case "tensorFromArray": {
      const payload = node.payload as { values: number[] } | undefined;
      if (!payload?.values) {
        throw new Error("tensorFromArray requires values in payload");
      }
      return nodeBackend.ops.tensorFromArray(payload.values, node.shape);
    }

    case "zeros": {
      if (nodeBackend.ops.zeros) {
        return nodeBackend.ops.zeros(node.shape);
      }
      const numEl = node.shape.reduce((a: number, b: number) => a * b, 1);
      return nodeBackend.ops.tensorFromArray(new Array(numEl).fill(0), node.shape);
    }

    case "full": {
      const fullPayload = node.payload as { fillValue: number };
      if (nodeBackend.ops.full) {
        return nodeBackend.ops.full(node.shape, fullPayload.fillValue);
      }
      const numElFull = node.shape.reduce((a: number, b: number) => a * b, 1);
      return nodeBackend.ops.tensorFromArray(new Array(numElFull).fill(fullPayload.fillValue), node.shape);
    }

    case "arange": {
      const ap = node.payload as { end: number; start: number; step: number };
      if (nodeBackend.ops.arange) {
        return nodeBackend.ops.arange(ap.end, ap.start, ap.step);
      }
      const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
      return nodeBackend.ops.tensorFromArray(vals, node.shape);
    }

    case "tril": {
      if (!nodeBackend.ops.tril) throw new Error("tril not supported by backend");
      return nodeBackend.ops.tril(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "triu": {
      if (!nodeBackend.ops.triu) throw new Error("triu not supported by backend");
      return nodeBackend.ops.triu(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "add":
      return nodeBackend.ops.add(backendInputs[0], backendInputs[1]);

    case "sub": {
      const subPayload = node.payload as { alpha?: number } | undefined;
      return nodeBackend.ops.sub(backendInputs[0], backendInputs[1], subPayload);
    }

    case "mul":
      return nodeBackend.ops.mul(backendInputs[0], backendInputs[1]);

    case "div": {
      const divPayload = node.payload as { roundingMode?: "trunc" | "floor" } | undefined;
      return nodeBackend.ops.div(backendInputs[0], backendInputs[1], divPayload);
    }

    case "matmul":
      return nodeBackend.ops.matmul(backendInputs[0], backendInputs[1]);

    case "sqrt":
      return nodeBackend.ops.sqrt(backendInputs[0]);

    case "relu":
      return nodeBackend.ops.relu(backendInputs[0]);

    case "exp":
      if (!nodeBackend.ops.exp) throw new Error("exp not supported on this backend");
      return nodeBackend.ops.exp(backendInputs[0]);

    case "log":
      if (!nodeBackend.ops.log) throw new Error("log not supported on this backend");
      return nodeBackend.ops.log(backendInputs[0]);

    case "neg":
      if (!nodeBackend.ops.neg) throw new Error("neg not supported on this backend");
      return nodeBackend.ops.neg(backendInputs[0]);

    case "abs":
      if (!nodeBackend.ops.abs) throw new Error("abs not supported on this backend");
      return nodeBackend.ops.abs(backendInputs[0]);

    case "tanh":
      if (!nodeBackend.ops.tanh) throw new Error("tanh not supported on this backend");
      return nodeBackend.ops.tanh(backendInputs[0]);

    case "sigmoid":
      if (!nodeBackend.ops.sigmoid) throw new Error("sigmoid not supported on this backend");
      return nodeBackend.ops.sigmoid(backendInputs[0]);

    case "gelu": {
      if (!nodeBackend.ops.gelu) throw new Error("gelu not supported on this backend");
      const geluPayload = node.payload as GeluOptions | undefined;
      return nodeBackend.ops.gelu(backendInputs[0], geluPayload);
    }

    case "silu":
      if (!nodeBackend.ops.silu) throw new Error("silu not supported on this backend");
      return nodeBackend.ops.silu(backendInputs[0]);

    case "cast": {
      if (!nodeBackend.ops.cast) throw new Error("cast not supported on this backend");
      const castPayload = node.payload as { dtype: DType } | undefined;
      if (!castPayload?.dtype) throw new Error("cast requires dtype in payload");
      return nodeBackend.ops.cast(backendInputs[0], castPayload.dtype);
    }

    case "pow":
      if (!nodeBackend.ops.pow) throw new Error("pow not supported on this backend");
      return nodeBackend.ops.pow(backendInputs[0], backendInputs[1]);

    case "reshape":
      return nodeBackend.ops.reshape(backendInputs[0], node.shape);

    case "expand":
      return nodeBackend.ops.expand(backendInputs[0], node.shape);

    case "transpose": {
      const transposePayload = node.payload as { dim0: number; dim1: number } | undefined;
      if (!transposePayload) throw new Error("transpose requires dim0, dim1 in payload");
      return nodeBackend.ops.transpose(backendInputs[0], transposePayload);
    }

    case "permute": {
      const permutePayload = node.payload as { dims: number[] } | undefined;
      if (!permutePayload?.dims) throw new Error("permute requires dims in payload");
      return nodeBackend.ops.permute(backendInputs[0], permutePayload.dims);
    }

    case "contiguous":
      return nodeBackend.ops.contiguous(backendInputs[0]);

    case "narrow": {
      const p = node.payload as { dim: number; start: number; length: number };
      if (!nodeBackend.ops.narrow) throw new Error("narrow not supported by backend");
      return nodeBackend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
    }

    case "narrowBackward": {
      const p = node.payload as { dim: number; start: number; originalLength: number };
      if (!nodeBackend.ops.narrowBackward) throw new Error("narrowBackward not supported by backend");
      return nodeBackend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
    }

    case "gather": {
      const gatherPayload = node.payload as { dim: number } | undefined;
      if (gatherPayload?.dim === undefined) throw new Error("gather requires dim in payload");
      return nodeBackend.ops.gather(backendInputs[0], backendInputs[1], gatherPayload);
    }

    case "scatterAdd": {
      const scatterAddPayload = node.payload as { dim: number } | undefined;
      if (scatterAddPayload?.dim === undefined) throw new Error("scatterAdd requires dim in payload");
      return nodeBackend.ops.scatterAdd(backendInputs[0], backendInputs[1], backendInputs[2], scatterAddPayload);
    }

    case "sum": {
      const sumPayload = node.payload as { dim?: number | number[] | null; keepdim?: boolean } | undefined;
      return nodeBackend.ops.sum(backendInputs[0], sumPayload);
    }

    case "mean": {
      const meanPayload = node.payload as { dim?: number | number[] | null; keepdim?: boolean } | undefined;
      return nodeBackend.ops.mean(backendInputs[0], meanPayload);
    }

    case "max": {
      const maxPayload = node.payload as { dim?: number | number[] | null; keepdim?: boolean } | undefined;
      return nodeBackend.ops.max(backendInputs[0], maxPayload);
    }

    case "argmax": {
      const argmaxPayload = node.payload as { dim: number; keepdim?: boolean } | undefined;
      if (argmaxPayload?.dim === undefined) throw new Error("argmax requires dim in payload");
      return nodeBackend.ops.argmax(backendInputs[0], argmaxPayload);
    }

    case "argmin": {
      const argminPayload = node.payload as { dim: number; keepdim?: boolean } | undefined;
      if (argminPayload?.dim === undefined) throw new Error("argmin requires dim in payload");
      return nodeBackend.ops.argmin(backendInputs[0], argminPayload);
    }

    case "gt":
      return nodeBackend.ops.gt(backendInputs[0], backendInputs[1]);

    case "lt":
      return nodeBackend.ops.lt(backendInputs[0], backendInputs[1]);

    case "ge":
      return nodeBackend.ops.ge(backendInputs[0], backendInputs[1]);

    case "le":
      return nodeBackend.ops.le(backendInputs[0], backendInputs[1]);

    case "eq":
      return nodeBackend.ops.eq(backendInputs[0], backendInputs[1]);

    case "ne":
      return nodeBackend.ops.ne(backendInputs[0], backendInputs[1]);

    case "where":
      return nodeBackend.ops.where(backendInputs[0], backendInputs[1], backendInputs[2]);

    case "stridedScatterCopy": {
      const scatterCopyPayload = node.payload as { offset: number; viewShape: number[]; viewStrides: number[] } | undefined;
      if (!scatterCopyPayload) throw new Error("stridedScatterCopy requires offset, viewShape, viewStrides in payload");
      return nodeBackend.ops.stridedScatterCopy(backendInputs[0], backendInputs[1], scatterCopyPayload);
    }

    case "stridedScatterAdd": {
      const scatterAddPayload = node.payload as { offset: number; viewShape: number[]; viewStrides: number[] } | undefined;
      if (!scatterAddPayload) throw new Error("stridedScatterAdd requires offset, viewShape, viewStrides in payload");
      return nodeBackend.ops.stridedScatterAdd(backendInputs[0], backendInputs[1], scatterAddPayload);
    }

    case "adamStep": {
      const adamPayload = node.payload as import("../backend/types").AdamStepConfig;
      if (!nodeBackend.ops.adamStep) throw new Error("adamStep not supported by backend");
      const adamResult = await nodeBackend.ops.adamStep(
        backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3],
        adamPayload,
      );
      const mStorage2 = createStorageHandle(node.device, adamResult.m);
      const vStorage2 = createStorageHandle(node.device, adamResult.v);
      const sideOutputs2 = { m: mStorage2, v: vStorage2 };
      storageTracker.markReachable(mStorage2.id, sideOutputs2);
      storageTracker.markReachable(vStorage2.id, sideOutputs2);
      (node as any)._adamSideOutputs = sideOutputs2;
      return adamResult.param;
    }

    case "unscaleGrad": {
      const unscalePayload2 = node.payload as { invScale: number; infFlagBuffer: unknown };
      if (!nodeBackend.ops.unscaleGrad) throw new Error("unscaleGrad not supported by backend");
      return nodeBackend.ops.unscaleGrad(
        backendInputs[0], unscalePayload2.invScale, unscalePayload2.infFlagBuffer,
      );
    }

    case "fusedAttentionForward": {
      const faPayload2 = node.payload as import("../backend/types").FusedAttentionConfig;
      if (!nodeBackend.ops.fusedAttentionForward) throw new Error("fusedAttentionForward not supported by backend");
      const faResult2 = nodeBackend.ops.fusedAttentionForward(
        backendInputs[0], backendInputs[1], backendInputs[2], faPayload2,
      );
      const lseSH2 = createStorageHandle(node.device, faResult2.logsumexp);
      (node as any)._attnSideOutput = lseSH2;
      return faResult2.output;
    }

    case "extractAttentionLogsumexp": {
      const parentNodeFA2 = node.inputs[0].node;
      const lseSH2 = (parentNodeFA2 as any)._attnSideOutput as StorageHandle | undefined;
      if (!lseSH2) throw new Error("extractAttentionLogsumexp: parent has no _attnSideOutput");
      const lseResult2 = lseSH2.backendTensor;
      storageTracker.unregister(lseSH2.id);
      (parentNodeFA2 as any)._attnSideOutput = undefined;
      return lseResult2;
    }

    case "fusedAttentionBackward": {
      const faBwdPayload2 = node.payload as import("../backend/types").FusedAttentionConfig;
      if (!nodeBackend.ops.fusedAttentionBackward) throw new Error("fusedAttentionBackward not supported by backend");
      const faBwdResult2 = nodeBackend.ops.fusedAttentionBackward(
        backendInputs[0], backendInputs[1], backendInputs[2],
        backendInputs[3], backendInputs[4], backendInputs[5], faBwdPayload2,
      );
      const dkSH2 = createStorageHandle(node.device, faBwdResult2.dK);
      const dvSH2 = createStorageHandle(node.device, faBwdResult2.dV);
      (node as any)._attnBwdDK = dkSH2;
      (node as any)._attnBwdDV = dvSH2;
      return faBwdResult2.dQ;
    }

    case "extractAttentionDK": {
      const parentDK2 = node.inputs[0].node;
      const dkSH2 = (parentDK2 as any)._attnBwdDK as StorageHandle | undefined;
      if (!dkSH2) throw new Error("extractAttentionDK: parent has no _attnBwdDK");
      const dkResult2 = dkSH2.backendTensor;
      storageTracker.unregister(dkSH2.id);
      (parentDK2 as any)._attnBwdDK = undefined;
      return dkResult2;
    }

    case "extractAttentionDV": {
      const parentDV2 = node.inputs[0].node;
      const dvSH2 = (parentDV2 as any)._attnBwdDV as StorageHandle | undefined;
      if (!dvSH2) throw new Error("extractAttentionDV: parent has no _attnBwdDV");
      const dvResult2 = dvSH2.backendTensor;
      storageTracker.unregister(dvSH2.id);
      (parentDV2 as any)._attnBwdDV = undefined;
      return dvResult2;
    }

    case "fusedCrossEntropyForward": {
      const cePayload3 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
      if (!nodeBackend.ops.fusedCrossEntropyForward) throw new Error("fusedCrossEntropyForward not supported by backend");
      return nodeBackend.ops.fusedCrossEntropyForward(
        backendInputs[0], backendInputs[1], cePayload3,
      );
    }

    case "fusedCrossEntropyBackward": {
      const cePayload4 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
      if (!nodeBackend.ops.fusedCrossEntropyBackward) throw new Error("fusedCrossEntropyBackward not supported by backend");
      return nodeBackend.ops.fusedCrossEntropyBackward(
        backendInputs[0], backendInputs[1], backendInputs[2], cePayload4,
      );
    }

    case "fusedLayerNormForward": {
      const lnPayload3 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!nodeBackend.ops.fusedLayerNormForward) throw new Error("fusedLayerNormForward not supported by backend");
      return nodeBackend.ops.fusedLayerNormForward(
        backendInputs[0], backendInputs[1], backendInputs[2], lnPayload3,
      );
    }

    case "fusedLayerNormBackwardGradX": {
      const lnPayload4 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!nodeBackend.ops.fusedLayerNormBackwardGradX) throw new Error("fusedLayerNormBackwardGradX not supported by backend");
      return nodeBackend.ops.fusedLayerNormBackwardGradX(
        backendInputs[0], backendInputs[1], backendInputs[2], lnPayload4,
      );
    }

    case "fusedLayerNormBackwardGradWeightBias": {
      const lnGWBPayload2 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!nodeBackend.ops.fusedLayerNormBackwardGradWeightBias) {
        throw new Error("fusedLayerNormBackwardGradWeightBias not supported by backend");
      }
      const gwResult2 = nodeBackend.ops.fusedLayerNormBackwardGradWeightBias(
        backendInputs[0], backendInputs[1], lnGWBPayload2,
      );
      // Store raw gradBias BackendTensor for extractLnBwdGradBias
      (node as any)._lnBwdSideOutput = gwResult2.gradBias;
      return gwResult2.gradWeight;
    }

    case "extractLnBwdGradBias": {
      const parentNode2 = node.inputs[0].node;
      const sideOutputBT2 = (parentNode2 as any)._lnBwdSideOutput as BackendTensor | undefined;
      if (!sideOutputBT2) {
        throw new Error("extractLnBwdGradBias: parent node has no _lnBwdSideOutput");
      }
      (parentNode2 as any)._lnBwdSideOutput = undefined;
      return sideOutputBT2;
    }

    case "transfer": {
      const transferPayload = node.payload as { targetDevice: DeviceKind } | undefined;
      if (!transferPayload?.targetDevice) throw new Error("transfer requires targetDevice in payload");
      const targetBackend = getBackend(transferPayload.targetDevice);
      if (!targetBackend) throw new Error(`No backend found for device: ${transferPayload.targetDevice}`);
      // Read from source backend, create in target backend
      const data = await nodeBackend.ops.read(backendInputs[0]);
      const sourceShape = (backendInputs[0] as { shape: number[] }).shape;
      return targetBackend.ops.tensorFromArray(data, sourceShape);
    }

    default:
      throw new Error(`Unknown op: ${node.op}`);
  }
  } finally {
    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}

/**
 * Internal helper to create a storage handle during execution.
 */
function createStorageHandleInternal(
  device: DeviceKind,
  backendTensor: BackendTensor,
  baseStorageId?: number,
): StorageHandle {
  const handle: StorageHandle = {
    id: nextStorageId++,
    device,
    backendTensor,
    baseStorageId,
  };
  storageTracker.register(handle);
  return handle;
}

// ============================================================================
// Optimized Execution with Fusion (§15)
// ============================================================================

/**
 * Options for optimized plan execution.
 */
export interface OptimizedExecutionOptions {
  /** Enable elementwise fusion (default: true for WebGPU) */
  enableFusion?: boolean;
  /** Enable vectorization for fused kernels (default: true) */
  enableVectorization?: boolean;
  /** Minimum ops required to trigger fusion (default: 2) */
  minFusionSize?: number;
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
  /** Flush buffer pool every N nodes to reclaim dead buffers mid-plan (default: 50, 0=disabled) */
  reclaimInterval?: number;
}

/** Default reclaim interval, overridable via TORCHLETTE_RECLAIM_INTERVAL env var. */
const DEFAULT_RECLAIM_INTERVAL =
  typeof process !== "undefined" && process.env?.TORCHLETTE_RECLAIM_INTERVAL
    ? parseInt(process.env.TORCHLETTE_RECLAIM_INTERVAL, 10)
    : 100;

/**
 * Statistics from optimized execution.
 */
export interface OptimizedExecutionStats {
  totalNodes: number;
  fusedNodes: number;
  sequentialNodes: number;
  fusionGroups: number;
  fusionEnabled: boolean;
}

/**
 * Result of optimized execution.
 */
export interface OptimizedExecutionResult {
  result: StorageHandle;
  stats: OptimizedExecutionStats;
}


// ============================================================================
// Fusion Analysis Cache
// ============================================================================

/**
 * Cached fusion analysis template. Stores the analysis result as position-based
 * indices that can be applied to any plan with the same structural fingerprint.
 */
interface FusionAnalysisTemplate {
  /** Maps final plan position → original plan position.
   *  finalPlan[i] = originalPlan[finalPerm[i]] */
  finalPerm: number[];

  /** Segment pattern using positions in the final plan. */
  segments: CachedSegmentDesc[];

  /** Original plan positions that are epilogue-claimed. */
  epilogueClaimedOrigPoss: number[];
  /** Original plan positions that are prologue-claimed. */
  prologueClaimedOrigPoss: number[];

  /** Matmul epilogue chains: [origPos, [epilogueOrigPoss]]. */
  epilogueChains: Array<[number, number[]]>;

  /** Matmul prologues: [origPos, [{inputIndex, castOrigPos, fromDtype, toDtype}]]. */
  prologueDescs: Array<[number, Array<{
    inputIndex: 0 | 1;
    castOrigPos: number;
    fromDtype: DType;
    toDtype: DType;
  }>]>;

  /** Cached lifetime analysis (position-based). */
  lifetimeTemplate?: Array<{ firstUse: number; lastUse: number; isOutput: boolean; isInput: boolean; bufferSize: number }>;

  /** Cached lowered execution plan (built during first execution on cache miss). */
  loweredPlan?: LoweredPlan;

  /** Per-plan buffer arena: GPUBuffers that persist across steps for bind group cache stability. */
  bufferArena?: BufferArena;
}

type CachedSegmentDesc =
  | { kind: "sequential"; finalPoss: number[] }
  | {
      kind: "fused";
      /** All group node positions in final plan. */
      finalPoss: number[];
      /** Output node position in final plan. */
      outputFinalPos: number;
      /** Additional output positions in final plan. */
      additionalOutputFinalPoss: number[];
      /** Needed intermediate positions in final plan. */
      neededIntermediateFinalPoss: number[];
    };

/**
 * Module-level cache for fusion analysis results.
 * Keyed by structural fingerprint (FNV-1a hash).
 * Typically holds <10 entries (one per unique plan structure).
 */
const fusionAnalysisCache = new Map<number, FusionAnalysisTemplate>();

/** Get a cached fusion analysis template by fingerprint. */
export function getFusionAnalysisTemplate(fingerprint: number): FusionAnalysisTemplate | undefined {
  return fusionAnalysisCache.get(fingerprint);
}

/**
 * Execute a plan with automatic fusion optimization.
 *
 * Per spec §15, this:
 * - Detects fusible elementwise op chains
 * - Dispatches fused WebGPU kernels when beneficial
 * - Falls back to sequential execution for non-fusible ops
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Optimization options
 */
export async function executePlanOptimized(
  plan: ExecutionPlan,
  backend: Backend,
  options: OptimizedExecutionOptions = {},
): Promise<OptimizedExecutionResult> {
  if (plan.nodes.length === 0) {
    throw new Error("Cannot execute empty plan");
  }

  // Pre-tune matmul shapes if backend supports it
  await pretunePlanMatmuls(plan, backend);

  const {
    enableFusion = backend.name === "webgpu",
    enableVectorization = true,
    minFusionSize = 2,
    enableEarlyRelease = false,
    reclaimInterval = DEFAULT_RECLAIM_INTERVAL,
  } = options;

  const stats: OptimizedExecutionStats = {
    totalNodes: plan.nodes.length,
    fusedNodes: 0,
    sequentialNodes: 0,
    fusionGroups: 0,
    fusionEnabled: enableFusion,
  };

  // Fall back to simple sequential execution when fusion is disabled entirely.
  // When fusion IS enabled, always go through the full analysis path even if
  // no fusible ops exist — this creates lowered plan templates (for adam batching,
  // arena allocation, bind group cache stability) that benefit all plan types.
  if (!enableFusion) {
    const result = await executePlan(plan, backend, { enableEarlyRelease });
    stats.sequentialNodes = plan.nodes.length;
    return { result, stats };
  }

  // Get node IDs with live external tensors (e.g., saved-for-backward)
  let externalNodeIds: Set<number> | undefined;
  try {
    const { getPendingNodeIds } = await import("../runtime/tensor");
    const pending = getPendingNodeIds();
    if (pending.size > 0) {
      externalNodeIds = pending;
    }
  } catch {
    // If runtime/tensor is not available, skip external node tracking
  }
  // Lifetime analysis is set up after analysis (below)
  let lifetimes: Map<number, TensorLifetime> | null = null;
  let outputNodeIds: Set<number> | null = null;
  const alreadyReleased = new Set<number>();
  const nodeToStorage = new Map<number, StorageHandle>();

  // Query device storage buffer limit to constrain fusion group size.
  const maxStorageBuffers: number | undefined =
    (backend as any).device?.limits?.maxStorageBuffersPerShaderStage;

  // Compute structural fingerprint for fusion analysis caching.
  // Plans with the same fingerprint have identical structure (ops, shapes,
  // dtypes, dependency graph) and can reuse cached analysis results.
  const fingerprint = computePlanFingerprint(plan.nodes, externalNodeIds);
  const cachedTemplate = fusionAnalysisCache.get(fingerprint);

  let planNodes: LazyIRNode[];
  let segments: ExecutionSegment[];
  const epilogueClaimedIds = new Set<number>();
  const prologueClaimedIds = new Set<number>();
  const matmulEpilogueChains = new Map<number, number[]>();
  const matmulPrologues = new Map<number, MatmulPrologueInfo[]>();

  if (cachedTemplate) {
    // ── Cache hit: reconstruct from template ──
    planNodes = cachedTemplate.finalPerm.map(i => plan.nodes[i]);

    // Reconstruct epilogue/prologue ID sets
    for (const pos of cachedTemplate.epilogueClaimedOrigPoss) {
      epilogueClaimedIds.add(plan.nodes[pos].id);
    }
    for (const pos of cachedTemplate.prologueClaimedOrigPoss) {
      prologueClaimedIds.add(plan.nodes[pos].id);
    }
    for (const [matmulPos, epiloguePoss] of cachedTemplate.epilogueChains) {
      matmulEpilogueChains.set(
        plan.nodes[matmulPos].id,
        epiloguePoss.map(p => plan.nodes[p].id),
      );
    }
    for (const [matmulPos, descs] of cachedTemplate.prologueDescs) {
      matmulPrologues.set(plan.nodes[matmulPos].id, descs.map(d => ({
        inputIndex: d.inputIndex,
        castNodeId: plan.nodes[d.castOrigPos].id,
        originalInputRef: plan.nodes[d.castOrigPos].inputs[0],
        fromDtype: d.fromDtype,
        toDtype: d.toDtype,
      })));
    }

    // Reconstruct lifetime analysis from cached template (avoids extractPlanMetadata + analyzeLifetimes)
    if (cachedTemplate.lifetimeTemplate && enableEarlyRelease) {
      lifetimes = new Map();
      for (let i = 0; i < planNodes.length; i++) {
        const t = cachedTemplate.lifetimeTemplate[i];
        lifetimes.set(planNodes[i].id, {
          nodeId: planNodes[i].id,
          firstUse: t.firstUse,
          lastUse: t.lastUse,
          isOutput: t.isOutput,
          isInput: t.isInput,
          bufferSize: t.bufferSize,
        });
      }
      outputNodeIds = new Set([planNodes[planNodes.length - 1].id]);
      if (externalNodeIds) {
        for (const id of externalNodeIds) outputNodeIds.add(id);
      }
    }

    // Reconstruct segments from cached pattern
    segments = cachedTemplate.segments.map(seg => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          nodes: seg.finalPoss.map(i => planNodes[i]),
        };
      }
      // Reconstruct FusionGroup
      const groupNodes = seg.finalPoss.map(i => planNodes[i]);
      const groupNodeIds = new Set(groupNodes.map(n => n.id));
      const extInputs: LazyRef[] = [];
      for (const node of groupNodes) {
        for (const inp of node.inputs) {
          if (inp.kind === "pending") {
            if (!groupNodeIds.has(inp.node.id) &&
                !extInputs.some(ei => ei.kind === "pending" && (ei as any).node.id === inp.node.id)) {
              extInputs.push(inp);
            }
          } else if (inp.kind === "scalar") {
            // Deduplicate scalar inputs by value+dtype
            if (!extInputs.some(ei => ei.kind === "scalar" && ei.value === inp.value && ei.dtype === inp.dtype)) {
              extInputs.push(inp);
            }
          } else {
            if (!extInputs.some(ei => ei.kind === "materialized" && ei.storage.id === inp.storage.id)) {
              extInputs.push(inp);
            }
          }
        }
      }
      const group: FusionGroup = {
        nodes: groupNodes,
        planIndices: seg.finalPoss,
        externalInputs: extInputs,
        outputNode: planNodes[seg.outputFinalPos],
        additionalOutputNodes: seg.additionalOutputFinalPoss.length > 0
          ? seg.additionalOutputFinalPoss.map(i => planNodes[i]) : undefined,
        neededIntermediates: seg.neededIntermediateFinalPoss.length > 0
          ? seg.neededIntermediateFinalPoss.map(i => planNodes[i]) : undefined,
      };
      const recipe = groupToRecipe(group);
      return { kind: "fused" as const, group, recipe };
    });
  } else {
    // ── Cache miss: run full analysis ──

    // Reorder plan to cluster fusible chains together
    planNodes = plan.nodes;
    if (planNodes.length > 2) {
      planNodes = reorderPlanForFusion(planNodes);
    }

    // Pre-scan: detect matmul epilogue chains AND input-side cast prologues
    if (backend.name === "webgpu") {
      // Build consumer map: nodeId → list of consumer nodes
      const consumers = new Map<number, LazyIRNode[]>();
      const consumerCount = new Map<number, number>();
      for (const node of planNodes) {
        for (const input of node.inputs) {
          if (input.kind === "pending") {
            const producerId = input.node.id;
            consumerCount.set(producerId, (consumerCount.get(producerId) ?? 0) + 1);
            if (!consumers.has(producerId)) consumers.set(producerId, []);
            consumers.get(producerId)!.push(node);
          }
        }
      }

      const nodePosition = new Map<number, number>();
      for (let i = 0; i < planNodes.length; i++) {
        nodePosition.set(planNodes[i].id, i);
      }

      const nodeById = new Map<number, LazyIRNode>();
      for (const n of planNodes) nodeById.set(n.id, n);


      for (let mi = 0; mi < planNodes.length; mi++) {
        const node = planNodes[mi];
        if (node.op !== "matmul") continue;

        const matmulPos = mi;
        const chainIds: number[] = [];
        let current = node;
        let additionalInputCount = 0;

        for (let depth = 0; depth < 4; depth++) {
          const cc = consumerCount.get(current.id) ?? 0;
          if (cc !== 1) break;
          // Skip external check for nodes already absorbed into the chain
          // (e.g., reshape nodes with single-consumer). Their live pending
          // tensors will be disposed by the caller after plan execution.
          if (externalNodeIds?.has(current.id) && !chainIds.includes(current.id)) {
            break;
          }

          const nexts = consumers.get(current.id);
          if (!nexts || nexts.length !== 1) break;
          const next = nexts[0];

          if (next.inputs.length === 0) break;
          let chainInputIdx = 0;
          const primary = next.inputs[0];
          if (primary.kind !== "pending" || primary.node.id !== current.id) {
            // For commutative binary ops, check if the chain continues via inputs[1]
            if ((next.op === "add" || next.op === "mul") && next.inputs.length === 2) {
              const alt = next.inputs[1];
              if (alt.kind === "pending" && alt.node.id === current.id) {
                chainInputIdx = 1;
              } else {
                break;
              }
            } else {
              break;
            }
          }

          // Skip reshape that only removes leading size-1 dims (e.g. [1,M,N]→[M,N])
          if (next.op === "reshape") {
            const curShape = current.shape;
            const nextShape = next.shape;
            if (curShape.length === nextShape.length + 1
                && curShape[0] === 1
                && curShape.slice(1).every((d: number, i: number) => d === nextShape[i])) {
              chainIds.push(next.id);
              current = next;
              continue; // don't increment depth — reshape is free
            }
            break;
          }

          let ok = false;
          if (next.op === "cast") ok = true;
          else if ((next.op === "add" || next.op === "mul") && next.inputs.length === 2) {
            if (additionalInputCount >= 4) break;
            const secondary = next.inputs[chainInputIdx === 0 ? 1 : 0];
            if (secondary.kind === "materialized") {
              ok = true;
            } else if (secondary.kind === "pending") {
              const secPos = nodePosition.get(secondary.node.id);
              if (secPos !== undefined && secPos < matmulPos) {
                ok = true;
              }
            }
            if (ok) additionalInputCount++;
          } else if (next.op === "relu" || next.op === "silu" || next.op === "sigmoid" || next.op === "tanh" || next.op === "gelu") {
            ok = true;
          }

          if (!ok) break;

          chainIds.push(next.id);
          current = next;
        }

        if (chainIds.length > 0) {
          matmulEpilogueChains.set(node.id, chainIds);
          for (const id of chainIds) epilogueClaimedIds.add(id);
        }

        // Detect input-side cast prologues (inference only)
        if (!externalNodeIds || externalNodeIds.size === 0) {
          const prologuesForNode: MatmulPrologueInfo[] = [];
          for (const idx of [0, 1] as const) {
            const inputRef = node.inputs[idx];
            if (inputRef.kind !== "pending") continue;
            const castNode = inputRef.node;
            if (castNode.op !== "cast") continue;
            if ((consumerCount.get(castNode.id) ?? 0) !== 1) continue;
            const castPayload = castNode.payload as { dtype: DType } | undefined;
            if (!castPayload) continue;
            const toDtype = castPayload.dtype;
            const castInput = castNode.inputs[0];
            if (!castInput) continue;
            let fromDtype: DType;
            if (castInput.kind === "pending") {
              fromDtype = castInput.node.dtype;
            } else if (castInput.kind === "materialized") {
              fromDtype = (castInput.storage.backendTensor as any).dtype ?? "f32";
            } else {
              continue;
            }
            if (fromDtype !== "f32" || toDtype !== "f16") continue;

            prologuesForNode.push({
              inputIndex: idx,
              castNodeId: castNode.id,
              originalInputRef: castInput,
              fromDtype,
              toDtype,
            });
            prologueClaimedIds.add(castNode.id);
          }
          if (prologuesForNode.length > 0) {
            matmulPrologues.set(node.id, prologuesForNode);
          }
        }
      }

      // Relocate epilogue chain nodes after their matmul
      if (epilogueClaimedIds.size > 0) {
        const claimedSet = epilogueClaimedIds;
        const unclaimed = planNodes.filter(n => !claimedSet.has(n.id));
        const relocated: LazyIRNode[] = [];
        for (const n of unclaimed) {
          relocated.push(n);
          const chain = matmulEpilogueChains.get(n.id);
          if (chain) {
            for (const id of chain) {
              relocated.push(nodeById.get(id)!);
            }
          }
        }
        planNodes = relocated;
      }
    }

    // Segment the reordered plan into fusible and sequential parts
    let allClaimedIds: Set<number> | undefined;
    if (epilogueClaimedIds.size > 0 || prologueClaimedIds.size > 0) {
      allClaimedIds = new Set([...epilogueClaimedIds, ...prologueClaimedIds]);
    }
    segments = segmentPlanForExecution(planNodes, externalNodeIds, {
      maxStorageBuffers,
      enableMultiOutput: true,
      epilogueClaimedIds: allClaimedIds,
    });

    // ── Build template and cache it ──
    const origIdToPos = new Map<number, number>();
    for (let i = 0; i < plan.nodes.length; i++) {
      origIdToPos.set(plan.nodes[i].id, i);
    }
    const finalPerm = planNodes.map(n => origIdToPos.get(n.id)!);

    const finalIdToPos = new Map<number, number>();
    for (let i = 0; i < planNodes.length; i++) {
      finalIdToPos.set(planNodes[i].id, i);
    }

    const cachedSegments: CachedSegmentDesc[] = segments.map(seg => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          finalPoss: seg.nodes.map(n => finalIdToPos.get(n.id)!),
        };
      }
      return {
        kind: "fused" as const,
        finalPoss: seg.group.nodes.map(n => finalIdToPos.get(n.id)!),
        outputFinalPos: finalIdToPos.get(seg.group.outputNode.id)!,
        additionalOutputFinalPoss: (seg.group.additionalOutputNodes ?? [])
          .map(n => finalIdToPos.get(n.id)!),
        neededIntermediateFinalPoss: (seg.group.neededIntermediates ?? [])
          .map(n => finalIdToPos.get(n.id)!),
      };
    });

    const template: FusionAnalysisTemplate = {
      finalPerm,
      segments: cachedSegments,
      epilogueClaimedOrigPoss: [...epilogueClaimedIds].map(id => origIdToPos.get(id)!),
      prologueClaimedOrigPoss: [...prologueClaimedIds].map(id => origIdToPos.get(id)!),
      epilogueChains: [...matmulEpilogueChains].map(([mmId, epilogueIds]) => [
        origIdToPos.get(mmId)!,
        epilogueIds.map(id => origIdToPos.get(id)!),
      ] as [number, number[]]),
      prologueDescs: [...matmulPrologues].map(([mmId, prologues]) => [
        origIdToPos.get(mmId)!,
        prologues.map(p => ({
          inputIndex: p.inputIndex,
          castOrigPos: origIdToPos.get(p.castNodeId)!,
          fromDtype: p.fromDtype,
          toDtype: p.toDtype,
        })),
      ] as [number, Array<{ inputIndex: 0 | 1; castOrigPos: number; fromDtype: DType; toDtype: DType }>]),
    };
    fusionAnalysisCache.set(fingerprint, template);
  }

  // Build a lowered plan during cache miss execution (first run).
  // On cache hit, the lowered plan is already attached to the template.
  let loweredPlanBuilder: LoweredPlanBuilder | null = null;
  const nodeIdToFinalPos = new Map<number, number>();
  for (let i = 0; i < planNodes.length; i++) {
    nodeIdToFinalPos.set(planNodes[i].id, i);
  }
  if (!cachedTemplate) {
    loweredPlanBuilder = new LoweredPlanBuilder(planNodes.length);
  }

  // Set up lifetime analysis after final plan order is determined.
  // Skip if already reconstructed from cached template (cache hit path above).
  if (enableEarlyRelease && !lifetimes) {
    const reorderedPlan = { nodes: planNodes };
    const { nodeOrder, nodeInputs, nodeSizes } = extractPlanMetadata(reorderedPlan);
    const lastNodeId = plan.nodes[plan.nodes.length - 1].id;
    outputNodeIds = new Set([lastNodeId]);
    if (externalNodeIds) {
      for (const id of externalNodeIds) outputNodeIds.add(id);
    }
    lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, outputNodeIds, nodeSizes);

    // Store lifetime template in the fusion analysis cache for future hits
    const cached = fusionAnalysisCache.get(fingerprint);
    if (cached && !cached.lifetimeTemplate) {
      cached.lifetimeTemplate = planNodes.map((node) => {
        const lt = lifetimes!.get(node.id)!;
        return { firstUse: lt.firstUse, lastUse: lt.lastUse, isOutput: lt.isOutput, isInput: lt.isInput, bufferSize: lt.bufferSize };
      });
    }
  }


  // Collect plan analysis for profiling
  let planAnalysisRef: PlanAnalysis | null = null;
  if (isProfilingEnabled()) {
    let fusedSegCount = 0;
    let seqSegCount = 0;
    let fusedNodeCount = 0;
    let fusionGroupCount = 0;
    const sequentialOps: Record<string, number> = {};
    const unfusedByShape: Record<string, { count: number; ops: Record<string, number> }> = {};

    for (const segment of segments) {
      if (segment.kind === "fused" && segment.group.nodes.length >= minFusionSize) {
        fusedSegCount++;
        fusedNodeCount += segment.group.nodes.length;
        fusionGroupCount++;
      } else if (segment.kind === "fused") {
        seqSegCount++;
        for (const node of segment.group.nodes) {
          sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
          if (isFusibleOp(node.op)) {
            const shapeKey = node.shape.join(",");
            let bucket = unfusedByShape[shapeKey];
            if (!bucket) { bucket = { count: 0, ops: {} }; unfusedByShape[shapeKey] = bucket; }
            bucket.count++;
            bucket.ops[node.op] = (bucket.ops[node.op] ?? 0) + 1;
          }
        }
      } else {
        seqSegCount++;
        for (const node of segment.nodes) {
          sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
          if (isFusibleOp(node.op) && !epilogueClaimedIds.has(node.id) && !prologueClaimedIds.has(node.id)) {
            const shapeKey = node.shape.join(",");
            let bucket = unfusedByShape[shapeKey];
            if (!bucket) { bucket = { count: 0, ops: {} }; unfusedByShape[shapeKey] = bucket; }
            bucket.count++;
            bucket.ops[node.op] = (bucket.ops[node.op] ?? 0) + 1;
          }
        }
      }
    }

    // Count reduction preamble opportunities from sequential segments
    let reductionFusionEstimate = 0;
    for (const segment of segments) {
      if (segment.kind !== "sequential") continue;
      for (let i = 0; i < segment.nodes.length - 1; i++) {
        const cur = segment.nodes[i];
        const next = segment.nodes[i + 1];
        if (isFusibleOp(cur.op) && cur.op !== "cast" && (next.op === "sum" || next.op === "mean")) {
          reductionFusionEstimate++;
        }
      }
    }

    planAnalysisRef = {
      planIndex: 0, // assigned by recordPlanAnalysis
      totalNodes: plan.nodes.length,
      segments: { fused: fusedSegCount, sequential: seqSegCount },
      fusedNodes: fusedNodeCount,
      fusionGroups: fusionGroupCount,
      epilogueFusions: matmulEpilogueChains.size,
      reductionFusions: reductionFusionEstimate,
      sequentialOps,
      unfusedByShape,
    };
    recordPlanAnalysis(planAnalysisRef);
  }

  // Track overall step index for early release
  let overallStep = 0;

  // Pre-build consumer count map once for the entire plan.
  // This is used by both reduction preamble detection and matmul epilogue detection.
  // Building it once here instead of per-segment saves ~90 redundant rebuilds.
  let planConsumerCount: Map<number, number> | undefined;
  if (backend.name === "webgpu") {
    planConsumerCount = new Map<number, number>();
    for (const n of planNodes) {
      for (const ref of n.inputs) {
        if (ref.kind === "pending") {
          planConsumerCount.set(ref.node.id, (planConsumerCount.get(ref.node.id) ?? 0) + 1);
        }
      }
    }
  }

  // Wrap the entire segment loop in a top-level shared encoder scope.
  // Inner begin/end calls in executeSequentialSegment become nested no-ops
  // thanks to the depth counter. This lets elementwise ops from consecutive
  // segments accumulate into a single batch, reducing queue.submit() calls.
  const useTopLevelSharedEncoder = backend.name === "webgpu";
  if (useTopLevelSharedEncoder) beginSharedEncoder();

  // Activate buffer arena if available from cached template.
  // This is critical for the fallback path: when executeLoweredPlan fails and
  // we fall back to executePlanOptimized, the arena still provides stable buffer
  // identities for bind group cache hits.
  const useArenaFallback = useTopLevelSharedEncoder && cachedTemplate?.bufferArena
    && process.env.TORCHLETTE_USE_ARENA !== "0";
  if (useArenaFallback) {
    setActiveArena(cachedTemplate!.bufferArena!);
  }

  try {

  // Track dispatched nodes for periodic buffer reclamation.
  // When the shared encoder is active, released buffers go to pendingRelease
  // and can't be reused. Periodically flushing moves them back to the main pool.
  let nodesSinceReclaim = 0;

  // Execute each segment
  for (const segment of segments) {
    if (
      segment.kind === "fused" &&
      segment.group.nodes.length >= minFusionSize
    ) {
      // Execute fused segment
      await executeFusedSegment(
        segment.group,
        segment.recipe,
        backend,
        enableVectorization,
      );
      stats.fusedNodes += segment.group.nodes.length;
      stats.fusionGroups++;

      // Record fused action in lowered plan builder
      if (loweredPlanBuilder) {
        loweredPlanBuilder.recordFused(
          segment.group.nodes.map(n => nodeIdToFinalPos.get(n.id)!),
          nodeIdToFinalPos.get(segment.group.outputNode.id)!,
          (segment.group.additionalOutputNodes ?? []).map(n => nodeIdToFinalPos.get(n.id)!),
          (segment.group.neededIntermediates ?? []).map(n => nodeIdToFinalPos.get(n.id)!),
          segment.recipe,
          enableVectorization,
        );
      }

      // Track storages and release dead buffers for fused nodes
      if (enableEarlyRelease) {
        for (const node of segment.group.nodes) {
          if (node.result) {
            nodeToStorage.set(node.id, node.result);
          }
          overallStep++;
          if (lifetimes && outputNodeIds) {
            const deadNodeIds = findDeadTensorsAtStep(
              lifetimes,
              overallStep,
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
        }
      }
      nodesSinceReclaim += segment.group.nodes.length;
    } else if (segment.kind === "fused") {
      // Too small for fusion - execute sequentially
      await executeSequentialSegmentWithEarlyRelease(
        segment.group.nodes,
        backend,
        enableEarlyRelease,
        lifetimes,
        outputNodeIds,
        alreadyReleased,
        nodeToStorage,
        overallStep,
        externalNodeIds,
        planNodes,
        matmulPrologues.size > 0 ? matmulPrologues : undefined,
        prologueClaimedIds.size > 0 ? prologueClaimedIds : undefined,
        planConsumerCount,
        loweredPlanBuilder,
        nodeIdToFinalPos,
      );
      stats.sequentialNodes += segment.group.nodes.length;
      overallStep += segment.group.nodes.length;
      nodesSinceReclaim += segment.group.nodes.length;
    } else {
      // Execute sequential segment
      await executeSequentialSegmentWithEarlyRelease(
        segment.nodes,
        backend,
        enableEarlyRelease,
        lifetimes,
        outputNodeIds,
        alreadyReleased,
        nodeToStorage,
        overallStep,
        externalNodeIds,
        planNodes,
        matmulPrologues.size > 0 ? matmulPrologues : undefined,
        prologueClaimedIds.size > 0 ? prologueClaimedIds : undefined,
        planConsumerCount,
        loweredPlanBuilder,
        nodeIdToFinalPos,
      );
      stats.sequentialNodes += segment.nodes.length;
      overallStep += segment.nodes.length;
      nodesSinceReclaim += segment.nodes.length;
    }

    // Periodic buffer reclamation: flush the shared encoder and buffer pool
    // so that dead buffers in pendingRelease become available for reuse.
    if (useTopLevelSharedEncoder && reclaimInterval > 0 && nodesSinceReclaim >= reclaimInterval) {
      flushSharedEncoder();
      flushBufferPool();
      if (loweredPlanBuilder) loweredPlanBuilder.recordReclaim();
      nodesSinceReclaim = 0;
    }
  }

  } finally {
    if (useArenaFallback) {
      clearActiveArena();
    }
    if (useTopLevelSharedEncoder) {
      endSharedEncoder();
    }
  }

  // Save the lowered plan to the fusion analysis template (first execution only).
  if (loweredPlanBuilder) {
    const cached = fusionAnalysisCache.get(fingerprint);
    if (cached && !cached.loweredPlan) {
      cached.loweredPlan = loweredPlanBuilder.build();
    }
  }

  // Get the result from the last node
  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Execution failed: no result for last node");
  }

  // Clear results for nodes whose buffers were destroyed by early release.
  if (alreadyReleased.size > 0) {
    for (const node of plan.nodes) {
      if (alreadyReleased.has(node.id)) {
        node.result = undefined;
      }
    }
  }

  return { result: lastNode.result, stats };
}

/**
 * Execute a lowered plan (cached dispatch sequence).
 *
 * Replaces the full executePlanOptimized() path on cache hits when the
 * lowered plan is available. Skips fusion detection, plan reordering,
 * segmentation, matmul epilogue detection, reduction preamble detection —
 * all of those decisions were recorded during the first execution and are
 * replayed here.
 *
 * @param plan - The original execution plan
 * @param planNodes - Reordered plan nodes (from template.finalPerm)
 * @param loweredPlan - The cached lowered plan from the template
 * @param backend - The backend to use
 * @param options - Execution options
 * @returns The result and execution stats
 */
export async function executeLoweredPlan(
  plan: ExecutionPlan,
  planNodes: LazyIRNode[],
  loweredPlan: LoweredPlan,
  backend: Backend,
  options: { enableEarlyRelease?: boolean; enableVectorization?: boolean; bufferArena?: BufferArena; enableReplay?: boolean } = {},
): Promise<OptimizedExecutionResult> {
  const { enableEarlyRelease = false, enableVectorization = true, enableReplay = true } = options;

  // Validate plan node count matches
  if (planNodes.length !== loweredPlan.planNodeCount) {
    throw new Error(
      `Lowered plan node count mismatch: plan has ${planNodes.length}, lowered expects ${loweredPlan.planNodeCount}`,
    );
  }

  // Pre-tune matmul shapes (same as normal path)
  await pretunePlanMatmuls({ nodes: planNodes }, backend);

  // Ensure matmul imports are loaded (cached — only first call does actual import)
  if (backend.name === "webgpu") {
    await ensureWebGPUMatmulImports();
  }

  // Track stats for consistency with executePlanOptimized
  const stats: OptimizedExecutionStats = {
    totalNodes: planNodes.length,
    fusedNodes: 0,
    sequentialNodes: 0,
    fusionGroups: 0,
    fusionEnabled: true,
  };

  const useTopLevelSharedEncoder = backend.name === "webgpu";
  // Dispatch replay cache: enabled by default for compiled plans where arena
  // stabilizes buffer identities. Disabled for non-compiled plans (backward,
  // optimizer) whose external inputs (saved-for-backward tensors, gradient
  // seeds) may not have stable buffer identities across steps.
  // Disable globally with TORCHLETTE_DISPATCH_REPLAY=0.
  const useReplayCache = enableReplay && process.env.TORCHLETTE_DISPATCH_REPLAY !== "0";

  // =========================================================================
  // FAST PATH: Dispatch Replay
  // =========================================================================
  // If we have a valid dispatch replay cache, skip all JS dispatch logic and
  // replay recorded GPU dispatches directly. Only data sources, view ops, and
  // encoder-copy ops (scatterAdd) are re-executed (their data or encoder
  // commands change per step).
  if (useReplayCache && loweredPlan.dispatchCache?.valid && useTopLevelSharedEncoder && options.bufferArena) {
    const cache = loweredPlan.dispatchCache;
    if (useTopLevelSharedEncoder) beginSharedEncoder();
    setActiveArena(options.bufferArena);

    // Compute stats from lowered plan structure (same as normal path would)
    for (const act of loweredPlan.actions) {
      if (act.kind === "fused") {
        stats.fusedNodes += act.coveredNodeIndices.length;
        stats.fusionGroups++;
      } else if (act.kind === "sequential" || act.kind === "data-source"
                 || act.kind === "view" || act.kind === "prologue-skip") {
        stats.sequentialNodes++;
      }
    }

    // Replay timing instrumentation (controlled by env var)
    const _replayTiming = process.env.TORCHLETTE_REPLAY_TIMING === "1";
    let _tReplayDispatch = 0, _tDataSource = 0, _tView = 0, _tSequential = 0;
    let _tAdamBatch = 0, _tReclaim = 0, _tResult = 0, _tLoopOverhead = 0;
    let _nDispatches = 0, _nDataSources = 0, _nViews = 0, _nSequential = 0;
    let _nAdamNodes = 0, _nReclaims = 0, _nResults = 0;
    const _tReplayStart = _replayTiming ? performance.now() : 0;

    try {
      // Batch consecutive dispatch entries for efficient replay
      const dispatchBatch: RecordedDispatch[] = [];
      const flushDispatchBatch = () => {
        if (dispatchBatch.length > 0) {
          const _t0 = _replayTiming ? performance.now() : 0;
          replayDispatches(dispatchBatch);
          if (_replayTiming) { _tReplayDispatch += performance.now() - _t0; _nDispatches += dispatchBatch.length; }
          dispatchBatch.length = 0;
        }
      };

      for (const entry of cache.entries) {
        switch (entry.kind) {
          case "dispatch":
            dispatchBatch.push(entry.dispatch);
            break;
          case "data-source": {
            flushDispatchBatch();
            const _dsT0 = _replayTiming ? performance.now() : 0;
            // Restore arena resolve index and sequence counters to match recording position
            setArenaResolveIndexTo(entry.arenaResolveIdx);
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            const dsNode = planNodes[entry.nodeIndex];
            if (dsNode.result) { if (_replayTiming) { _tDataSource += performance.now() - _dsT0; _nDataSources++; } break; }
            const dsBackend = getBackend(dsNode.device) ?? backend;
            const dsInputs = dsNode.inputs.map(ref => getInputStorage(ref, dsBackend));
            const dsBackendInputs = dsInputs.map(s => s.backendTensor);
            const dsResult = await executeOp(dsNode, dsBackendInputs, dsBackend);
            dsNode.result = createStorageHandle(dsNode.device, dsResult);
            if (_replayTiming) { _tDataSource += performance.now() - _dsT0; _nDataSources++; }
            break;
          }
          case "view": {
            flushDispatchBatch();
            const _vT0 = _replayTiming ? performance.now() : 0;
            const vNode = planNodes[entry.nodeIndex];
            if (vNode.result) { if (_replayTiming) { _tView += performance.now() - _vT0; _nViews++; } break; }
            if (entry.cachedResult) {
              // Fast path: reconstruct from cached metadata (arena buffers stable)
              setArenaResolveIndexTo(entry.arenaResolveIdxAfter ?? entry.arenaResolveIdx);
              const cr = entry.cachedResult;
              vNode.result = createStorageHandle(vNode.device, {
                buffer: cr.buffer,
                shape: cr.shape,
                dtype: cr.dtype,
                size: cr.size,
                strides: cr.strides,
                offset: cr.offset,
                isContiguous: cr.isContiguous,
                ownsBuffer: false,
                destroy() {},
              } as BackendTensor);
            } else {
              // Slow path: re-execute (first replay before cache populated)
              setArenaResolveIndexTo(entry.arenaResolveIdx);
              const vNodeBackend = getBackend(vNode.device) ?? backend;
              const vInputs = vNode.inputs.map(ref => getInputStorage(ref, vNodeBackend));
              const vBackendInputs = vInputs.map(s => s.backendTensor);
              let vResultTensor = await executeOp(vNode, vBackendInputs, vNodeBackend);
              const vAliasedInputIdx = vBackendInputs.findIndex(b => b === vResultTensor);
              if (vAliasedInputIdx >= 0 && (vResultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
                vResultTensor = { ...vResultTensor, ownsBuffer: false } as BackendTensor;
              }
              vNode.result = createStorageHandle(vNode.device, vResultTensor);
            }
            if (_replayTiming) { _tView += performance.now() - _vT0; _nViews++; }
            break;
          }
          case "sequential": {
            flushDispatchBatch();
            const _seqT0 = _replayTiming ? performance.now() : 0;
            // Restore arena resolve index and sequence counters to match recording position
            setArenaResolveIndexTo(entry.arenaResolveIdx);
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            const node = planNodes[entry.nodeIndex];
            if (node.result) { if (_replayTiming) { _tSequential += performance.now() - _seqT0; _nSequential++; } break; }
            const nodeBackend = getBackend(node.device) ?? backend;
            const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
            const backendInputs = inputs.map(s => s.backendTensor);
            let resultTensor = await executeOp(node, backendInputs, nodeBackend);
            const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
            if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
              resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
            }
            const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
            const baseStorageId = isView && inputs.length > 0
              ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
              : undefined;
            node.result = createStorageHandle(node.device, resultTensor, baseStorageId);
            if (_replayTiming) { _tSequential += performance.now() - _seqT0; _nSequential++; }
            break;
          }
          case "adam-batch": {
            flushDispatchBatch();
            const _adamT0 = _replayTiming ? performance.now() : 0;
            flushSharedEncoder();
            flushBufferPool();
            // Set sequence counters so adam dispatches hit the correct cache positions
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            setAdamBatchMode(true);
            try {
              const adamBackend = backend;
              const adamOp = adamBackend.ops.adamStep!;
              // Profile the entire adam batch as one op
              setCurrentOpLabel("adamStep");
              const _adamBatchT0 = profileOpBegin("adamStep");
              for (const nodeIdx of entry.nodeIndices) {
                const adamNode = planNodes[nodeIdx];
                if (adamNode.result) continue;
                // Inline getInputStorage + backendTensor extraction
                const inputs = adamNode.inputs;
                const s0 = getInputStorage(inputs[0], adamBackend);
                const s1 = getInputStorage(inputs[1], adamBackend);
                const s2 = getInputStorage(inputs[2], adamBackend);
                const s3 = getInputStorage(inputs[3], adamBackend);
                // Call adamStep directly, bypassing executeOp switch dispatch
                const adamPayload = adamNode.payload as import("../backend/types").AdamStepConfig;
                const adamResult = await adamOp(
                  s0.backendTensor, s1.backendTensor, s2.backendTensor, s3.backendTensor,
                  adamPayload,
                );
                // Adam is in-place: param buffer is returned as result.
                // Set ownsBuffer: false since the buffer is shared with the input.
                const paramResult = adamResult.param;
                const paramOwns = (paramResult as { ownsBuffer?: boolean }).ownsBuffer;
                const finalResult = paramOwns === true
                  ? { ...paramResult, ownsBuffer: false } as BackendTensor
                  : paramResult;
                // Side outputs (m, v) — create storage handles
                const mStorage = createStorageHandle(adamNode.device, adamResult.m);
                const vStorage = createStorageHandle(adamNode.device, adamResult.v);
                const sideOutputs = { m: mStorage, v: vStorage };
                storageTracker.markReachable(mStorage.id, sideOutputs);
                storageTracker.markReachable(vStorage.id, sideOutputs);
                (adamNode as any)._adamSideOutputs = sideOutputs;
                // Use param input (index 1) as base storage for the view result
                adamNode.result = createStorageHandle(adamNode.device, finalResult, s1.id);
              }
              profileOpEnd("adamStep", _adamBatchT0);
            } finally {
              setAdamBatchMode(false);
            }
            if (_replayTiming) { _tAdamBatch += performance.now() - _adamT0; _nAdamNodes += entry.nodeIndices.length; }
            break;
          }
          case "reclaim": {
            // During replay, flush encoder to submit pending dispatches but
            // skip pool reclamation — arena buffers are persistent.
            flushDispatchBatch();
            const _rclT0 = _replayTiming ? performance.now() : 0;
            flushSharedEncoder();
            if (_replayTiming) { _tReclaim += performance.now() - _rclT0; _nReclaims++; }
            break;
          }
          case "pre-adam-reclaim": {
            flushDispatchBatch();
            const _parT0 = _replayTiming ? performance.now() : 0;
            // Still need pre-adam flush for Adam's flushSharedEncoder requirement
            flushSharedEncoder();
            flushBufferPool();
            if (_replayTiming) { _tReclaim += performance.now() - _parT0; _nReclaims++; }
            break;
          }
          case "result": {
            flushDispatchBatch();
            const _rsT0 = _replayTiming ? performance.now() : 0;
            // Assign node result from cached metadata (arena buffers are stable).
            // Arena buffers persist across steps — ownsBuffer: false, no-op destroy.
            const nr = entry.nodeResult;
            const node = planNodes[nr.nodeIndex];
            if (!node.result) {
              node.result = createStorageHandle(node.device, {
                buffer: nr.buffer,
                shape: nr.shape,
                dtype: nr.dtype,
                size: nr.size,
                strides: nr.strides,
                offset: 0,
                isContiguous: true,
                ownsBuffer: false,
                destroy() {},
              } as BackendTensor);
            }
            if (_replayTiming) { _tResult += performance.now() - _rsT0; _nResults++; }
            break;
          }
          case "side-output": {
            // Restore _attnSideOutput on fusedAttentionForward nodes so a later
            // plan (backward Phase B) can skip re-executing fusedAttentionForward
            // and extractAttentionLogsumexp can read the side output directly.
            const soNode = planNodes[entry.nodeIndex];
            if (!(soNode as any)._attnSideOutput) {
              (soNode as any)._attnSideOutput = createStorageHandle(soNode.device, {
                buffer: entry.buffer,
                shape: entry.shape,
                dtype: entry.dtype,
                size: entry.size,
                strides: entry.strides,
                offset: 0,
                isContiguous: true,
                ownsBuffer: false,
                destroy() {},
              } as BackendTensor);
            }
            break;
          }
        }
      }
      flushDispatchBatch(); // Flush remaining batched dispatches
    } finally {
      clearActiveArena();
      const _tEndT0 = _replayTiming ? performance.now() : 0;
      if (useTopLevelSharedEncoder) endSharedEncoder();
      const _tEnd = _replayTiming ? performance.now() - _tEndT0 : 0;
      if (_replayTiming) {
        const _tTotal = performance.now() - _tReplayStart;
        const _tAccounted = _tReplayDispatch + _tDataSource + _tView + _tSequential + _tAdamBatch + _tReclaim + _tResult + _tEnd;
        console.log(`[replay-timing] nodes=${plan.nodes.length} entries=${cache.entries.length} | total=${_tTotal.toFixed(1)}ms | dispatch=${_tReplayDispatch.toFixed(1)}ms(${_nDispatches}) dataSrc=${_tDataSource.toFixed(1)}ms(${_nDataSources}) view=${_tView.toFixed(1)}ms(${_nViews}) seq=${_tSequential.toFixed(1)}ms(${_nSequential}) adam=${_tAdamBatch.toFixed(1)}ms(${_nAdamNodes}) reclaim=${_tReclaim.toFixed(1)}ms(${_nReclaims}) result=${_tResult.toFixed(1)}ms(${_nResults}) endEnc=${_tEnd.toFixed(1)}ms | unaccounted=${(_tTotal - _tAccounted).toFixed(1)}ms`);
      }
    }

    // Get the result from the last original plan node
    const lastNode = plan.nodes[plan.nodes.length - 1];
    if (!lastNode.result) {
      throw new Error("Dispatch replay failed: no result for last node");
    }
    return { result: lastNode.result, stats };
  }

  // =========================================================================
  // NORMAL PATH (with optional recording for dispatch replay cache)
  // =========================================================================

  // Shared encoder scope
  if (useTopLevelSharedEncoder) beginSharedEncoder();

  // Activate buffer arena for this plan (stabilizes buffer identities for bind group cache)
  if (options.bufferArena && useTopLevelSharedEncoder) {
    setActiveArena(options.bufferArena);
  }

  // Set up dispatch recording if we have an arena (needed for stable bind groups)
  // and no replay cache yet. We only record after the first execution (arena needs
  // to be populated first), so we check if the arena already has buffers.
  const shouldRecord = useReplayCache && useTopLevelSharedEncoder && options.bufferArena
    && !loweredPlan.dispatchCache
    && options.bufferArena.resolve.length > 0; // Arena populated from prior execution
  const recordingBuffer: RecordedDispatch[] = [];
  const replayEntries: ReplayEntry[] = [];
  let recordingDispatchIdx = 0;

  if (shouldRecord) {
    startDispatchRecording(recordingBuffer);
    // Also set up matmul and fusion recording buffers (imports cached from prior calls)
    const [matmulMod, fusionMod] = await Promise.all([
      import("../backend/webgpu/matmul/dispatch"),
      import("../backend/webgpu/fusion-dispatch"),
    ]);
    matmulMod.setMatmulRecordingBuffer(recordingBuffer);
    fusionMod.setFusionRecordingBuffer(recordingBuffer);
  }

  try {
    for (const action of loweredPlan.actions) {
      switch (action.kind) {
        case "fused": {
          // Reconstruct FusionGroup from plan nodes
          const groupNodes = action.coveredNodeIndices.map(i => planNodes[i]);
          const outputNode = planNodes[action.outputNodeIndex];
          const additionalOutputNodes = action.additionalOutputNodeIndices.map(i => planNodes[i]);
          const neededIntermediates = action.neededIntermediateNodeIndices.map(i => planNodes[i]);

          // Reconstruct external inputs using cached pattern (O(n) instead of O(n²))
          let extInputs: LazyRef[];
          if (action.cachedExternalInputPattern) {
            // Fast path: use pre-computed pattern
            extInputs = action.cachedExternalInputPattern.map(p =>
              groupNodes[p.nodeLocalIdx].inputs[p.inputIdx],
            );
          } else {
            // First execution: compute pattern with dedup, then cache it
            const groupNodeIds = new Set(groupNodes.map(n => n.id));
            extInputs = [];
            const pattern: Array<{ nodeLocalIdx: number; inputIdx: number }> = [];
            for (let ni = 0; ni < groupNodes.length; ni++) {
              const node = groupNodes[ni];
              for (let ii = 0; ii < node.inputs.length; ii++) {
                const inp = node.inputs[ii];
                if (inp.kind === "pending") {
                  if (!groupNodeIds.has(inp.node.id) &&
                      !extInputs.some(ei => ei.kind === "pending" && (ei as any).node.id === inp.node.id)) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                } else if (inp.kind === "scalar") {
                  if (!extInputs.some(ei => ei.kind === "scalar" && ei.value === inp.value && ei.dtype === inp.dtype)) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                } else {
                  if (!extInputs.some(ei => ei.kind === "materialized" && ei.storage.id === inp.storage.id)) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                }
              }
            }
            action.cachedExternalInputPattern = pattern;
          }

          const group: FusionGroup = {
            nodes: groupNodes,
            planIndices: action.coveredNodeIndices,
            externalInputs: extInputs,
            outputNode,
            additionalOutputNodes: additionalOutputNodes.length > 0 ? additionalOutputNodes : undefined,
            neededIntermediates: neededIntermediates.length > 0 ? neededIntermediates : undefined,
          };

          await executeFusedSegment(
            group,
            action.recipe,
            backend,
            action.enableVectorization,
          );
          stats.fusedNodes += groupNodes.length;
          stats.fusionGroups++;

          // Record dispatches and node results for replay cache
          if (shouldRecord) {
            // Capture dispatches emitted during this action
            while (recordingDispatchIdx < recordingBuffer.length) {
              replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
              recordingDispatchIdx++;
            }
            // Record output node result (inline for ordering)
            if (outputNode.result) {
              const bt = outputNode.result.backendTensor as any;
              replayEntries.push({ kind: "result", nodeResult: {
                nodeIndex: action.outputNodeIndex,
                buffer: bt.buffer,
                shape: bt.shape?.slice() ?? [],
                dtype: bt.dtype ?? "f32",
                size: bt.size ?? 0,
                strides: bt.strides?.slice() ?? [],
                isView: false,
              }});
            }
            // Record additional output node results
            for (const addIdx of action.additionalOutputNodeIndices) {
              const addNode = planNodes[addIdx];
              if (addNode.result) {
                const bt = addNode.result.backendTensor as any;
                replayEntries.push({ kind: "result", nodeResult: {
                  nodeIndex: addIdx,
                  buffer: bt.buffer,
                  shape: bt.shape?.slice() ?? [],
                  dtype: bt.dtype ?? "f32",
                  size: bt.size ?? 0,
                  strides: bt.strides?.slice() ?? [],
                  isView: false,
                }});
              }
            }
          }
          break;
        }

        case "matmul-epilogue": {
          const matmulNode = planNodes[action.matmulNodeIndex];
          const outputNode = planNodes[action.outputNodeIndex];

          // Cache the label string (structural, same across steps)
          let epLabel = action.cachedLabel;
          if (!epLabel) {
            const prologueLabel = action.prologues ? "prologue+" : "";
            const epilogueLabel = action.epilogueOps.length > 0
              ? "+" + action.epilogueOps.map(o => o.kind).join("+")
              : "";
            epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(/\+$/, "");
            action.cachedLabel = epLabel;
          }

          // ── FAST PATH: use cached dispatch config (2nd+ lowered plan execution) ──
          if (action.cachedDispatchConfig) {
            const cfg = action.cachedDispatchConfig;
            setCurrentOpLabel(epLabel);
            setProfileModule(matmulNode.module ?? "unknown");
            const _profT0 = profileOpBegin(epLabel);
            try {
              // Resolve GPU buffers from cached paths (plan-node-relative lookups)
              const refA = planNodes[cfg.inputAPath.planNodeIndex].inputs[cfg.inputAPath.inputIndex];
              const refB = planNodes[cfg.inputBPath.planNodeIndex].inputs[cfg.inputBPath.inputIndex];
              const bufA = (getInputStorage(refA).backendTensor as any).buffer;
              const bufB = (getInputStorage(refB).backendTensor as any).buffer;
              const epilogueBuffers: any[] = [];
              for (const path of cfg.epilogueInputPaths) {
                const ref = planNodes[path.planNodeIndex].inputs[path.inputIndex];
                epilogueBuffers.push((getInputStorage(ref).backendTensor as any).buffer);
              }

              // Dispatch directly — skips shape computation, transpose detection,
              // contiguous checks, prologue resolution, dynamic imports
              const resultTensor = _webgpuMatmulImports!.dispatchMatmulDirect(
                bufA, bufB, {
                  m: cfg.m, n: cfg.n, k: cfg.k,
                  transA: cfg.transA, transB: cfg.transB,
                  batchSize: cfg.batchSize,
                  batchStrideA: cfg.batchStrideA,
                  batchStrideB: cfg.batchStrideB,
                  batchStrideC: cfg.batchStrideC,
                  outShape: cfg.outShape,
                  dtypeA: cfg.dtypeA, dtypeB: cfg.dtypeB,
                  outputDtype: cfg.outputDtype,
                  epilogueConfig: cfg.epilogueConfig,
                  epilogueBuffers,
                  inputCastA: cfg.inputCastA,
                  inputCastB: cfg.inputCastB,
                },
              );
              // Fix up shape if epilogue absorbed a reshape (e.g. [1,M,N]→[M,N])
              const fastOutShape = outputNode.shape;
              if (!shapesEqual(resultTensor.shape, fastOutShape)) {
                (resultTensor as any).shape = fastOutShape;
                const newStrides = new Array(fastOutShape.length);
                let stride = 1;
                for (let i = fastOutShape.length - 1; i >= 0; i--) {
                  newStrides[i] = stride;
                  stride *= fastOutShape[i];
                }
                (resultTensor as any).strides = newStrides;
              }
              outputNode.result = createStorageHandle(outputNode.device, resultTensor);
            } finally {
              profileOpEnd(epLabel, _profT0);
              setCurrentOpLabel(null);
              setProfileModule("unknown");
            }

            // Record dispatches and node results (for replay cache)
            if (shouldRecord) {
              while (recordingDispatchIdx < recordingBuffer.length) {
                replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
                recordingDispatchIdx++;
              }
              if (outputNode.result) {
                const bt = outputNode.result.backendTensor as any;
                replayEntries.push({ kind: "result", nodeResult: {
                  nodeIndex: action.outputNodeIndex,
                  buffer: bt.buffer,
                  shape: bt.shape?.slice() ?? [],
                  dtype: bt.dtype ?? "f32",
                  size: bt.size ?? 0,
                  strides: bt.strides?.slice() ?? [],
                  isView: false,
                }});
              }
            }
            break;
          }

          // ── SLOW PATH: first lowered plan execution — full reconstruction ──

          // Reconstruct the epilogue input refs from the covered nodes.
          // For each add/mul in the chain, the "external" input (not from the chain)
          // is the epilogue input. The chain may connect via inputs[0] or inputs[1]
          // (commutative ops), so we check which input is the previous chain node.
          const epilogueInputRefs: LazyRef[] = [];
          const epilogueInputPaths: Array<{ planNodeIndex: number; inputIndex: number }> = [];
          for (let ci = 1; ci < action.coveredNodeIndices.length; ci++) {
            const chainNode = planNodes[action.coveredNodeIndices[ci]];
            if ((chainNode.op === "add" || chainNode.op === "mul") && chainNode.inputs.length === 2) {
              const prevChainNodeId = planNodes[action.coveredNodeIndices[ci - 1]].id;
              const inp0IsChain = chainNode.inputs[0].kind === "pending"
                && chainNode.inputs[0].node.id === prevChainNodeId;
              const externalIdx = inp0IsChain ? 1 : 0;
              epilogueInputRefs.push(chainNode.inputs[externalIdx]);
              epilogueInputPaths.push({ planNodeIndex: action.coveredNodeIndices[ci], inputIndex: externalIdx });
            }
          }

          // Reconstruct prologues and resolve prologue decisions
          let prologues: MatmulPrologueInfo[] | undefined;
          let inputCastA: DType | undefined;
          let inputCastB: DType | undefined;
          let resolvedInputRefA = matmulNode.inputs[0];
          let resolvedInputRefB = matmulNode.inputs[1];
          // Track paths for caching (plan-node-relative, stable across steps)
          let inputAPath = { planNodeIndex: action.matmulNodeIndex, inputIndex: 0 };
          let inputBPath = { planNodeIndex: action.matmulNodeIndex, inputIndex: 1 };

          if (action.prologues && action.prologues.length > 0) {
            prologues = action.prologues.map(p => ({
              inputIndex: p.inputIndex,
              castNodeId: planNodes[p.castNodeIndex].id,
              originalInputRef: planNodes[p.castNodeIndex].inputs[0],
              fromDtype: p.fromDtype,
              toDtype: p.toDtype,
            }));

            // Resolve prologue decisions (same logic as executeMatmulWithEpilogue)
            for (const p of action.prologues) {
              const castRef = p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
              const castAlreadyRan = castRef.kind === "pending" && castRef.node.result != null;
              if (!castAlreadyRan) {
                if (p.inputIndex === 0) {
                  resolvedInputRefA = planNodes[p.castNodeIndex].inputs[0];
                  inputCastA = p.toDtype;
                  inputAPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
                } else {
                  resolvedInputRefB = planNodes[p.castNodeIndex].inputs[0];
                  inputCastB = p.toDtype;
                  inputBPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
                }
              }
            }
          }

          const epiloguePlan: MatmulEpiloguePlan = {
            consumedCount: action.consumedCount,
            epilogueOps: action.epilogueOps,
            epilogueInputRefs,
            outputDtype: action.outputDtype,
            outputNode,
            prologues,
          };

          setCurrentOpLabel(epLabel);
          setProfileModule(matmulNode.module ?? "unknown");
          const _profT0 = profileOpBegin(epLabel);
          try {
            await executeMatmulWithEpilogue(matmulNode, epiloguePlan, backend);
          } finally {
            profileOpEnd(epLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // ── Cache dispatch config for next step ──
          // Compute matmul geometry from the resolved inputs (shapes are stable across steps).
          // Must replicate dispatchMatmulWithEpilogue's transpose detection logic exactly.
          {
            const { computeMatmulOutputShape, computeBatchSize, computeBatchStrides } = _webgpuMatmulGeomImports!;
            const { isF16Supported } = await import("../backend/webgpu/index");

            const tensorA = getInputStorage(resolvedInputRefA).backendTensor as any;
            const tensorB = getInputStorage(resolvedInputRefB).backendTensor as any;

            // Detect simple last-2-dim transposes (matching detectSimpleTranspose in index.ts).
            // If detected, use original contiguous shape and flip transpose flag.
            const detA = _detectTransposeView(tensorA);
            const detB = _detectTransposeView(tensorB);
            const effectiveShapeA = detA.shape;
            const effectiveShapeB = detB.shape;
            const transA = detA.transposed;
            const transB = detB.transposed;

            const outShape = computeMatmulOutputShape(effectiveShapeA, effectiveShapeB, transA, transB);
            const aRank = effectiveShapeA.length;
            const bRank = effectiveShapeB.length;
            const m = transA ? effectiveShapeA[aRank - 1] : effectiveShapeA[aRank - 2];
            const k = transA ? effectiveShapeA[aRank - 2] : effectiveShapeA[aRank - 1];
            const n = transB ? effectiveShapeB[bRank - 2] : effectiveShapeB[bRank - 1];

            const batchDims = outShape.slice(0, -2);
            const batchSize = computeBatchSize(batchDims);
            const { strideA, strideB, strideC } = computeBatchStrides(effectiveShapeA, effectiveShapeB, batchDims, m, n, k);

            // Compute dtypes (matching dispatchMatmulWithEpilogue logic)
            const f16ok = isF16Supported();
            const rawDtypeA = tensorA.dtype === "f16" && f16ok ? "f16" as const : "f32" as const;
            const rawDtypeB = tensorB.dtype === "f16" && f16ok ? "f16" as const : "f32" as const;
            const dtypeA: "f16" | "f32" = inputCastA === "f16" && f16ok ? "f16" : rawDtypeA;
            const dtypeB: "f16" | "f32" = inputCastB === "f16" && f16ok ? "f16" : rawDtypeB;
            const promotedDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
            const outputDtype = action.outputDtype ?? promotedDtype;

            const epilogueConfig = {
              ops: action.epilogueOps,
              additionalInputCount: epilogueInputRefs.length,
              outputDtype: action.outputDtype,
            };

            action.cachedDispatchConfig = {
              inputAPath,
              inputBPath,
              epilogueInputPaths,
              inputCastA,
              inputCastB,
              m, k, n,
              transA, transB,
              batchSize,
              batchStrideA: strideA,
              batchStrideB: strideB,
              batchStrideC: strideC,
              outShape,
              dtypeA,
              dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
              outputDtype,
              epilogueConfig,
            };
          }

          // Record dispatches and node results
          if (shouldRecord) {
            while (recordingDispatchIdx < recordingBuffer.length) {
              replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
              recordingDispatchIdx++;
            }
            if (outputNode.result) {
              const bt = outputNode.result.backendTensor as any;
              replayEntries.push({ kind: "result", nodeResult: {
                nodeIndex: action.outputNodeIndex,
                buffer: bt.buffer,
                shape: bt.shape?.slice() ?? [],
                dtype: bt.dtype ?? "f32",
                size: bt.size ?? 0,
                strides: bt.strides?.slice() ?? [],
                isView: false,
              }});
            }
          }
          break;
        }

        case "reduction-preamble": {
          const preambleNode = planNodes[action.preambleNodeIndex];
          const reductionNode = planNodes[action.reductionNodeIndex];
          const reductionPlan: ReductionPreamblePlan = {
            preambleNode,
            reductionNode,
            op: preambleNode.op,
            arity: preambleNode.inputs.length,
            isMean: reductionNode.op === "mean",
          };
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.op}`;
          setCurrentOpLabel(rpLabel);
          setProfileModule(preambleNode.module ?? "unknown");
          const _profT0 = profileOpBegin(rpLabel);
          try {
            await executeReductionWithPreamble(reductionPlan, backend);
          } finally {
            profileOpEnd(rpLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // Record dispatches and node results
          if (shouldRecord) {
            while (recordingDispatchIdx < recordingBuffer.length) {
              replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
              recordingDispatchIdx++;
            }
            // Record reduction node result
            if (reductionNode.result) {
              const bt = reductionNode.result.backendTensor as any;
              replayEntries.push({ kind: "result", nodeResult: {
                nodeIndex: action.reductionNodeIndex,
                buffer: bt.buffer,
                shape: bt.shape?.slice() ?? [],
                dtype: bt.dtype ?? "f32",
                size: bt.size ?? 0,
                strides: bt.strides?.slice() ?? [],
                isView: false,
              }});
            }
          }
          break;
        }

        case "adam-batch": {
          const useSharedEncoder = backend.name === "webgpu";
          if (useSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }

          // Record pre-adam reclaim and capture counter positions
          let adamSeqCountersBefore: { dispatch: number; params: number; output: number } | undefined;
          if (shouldRecord && useSharedEncoder) {
            replayEntries.push({ kind: "pre-adam-reclaim" });
            adamSeqCountersBefore = getDispatchSequenceCounters();
          }

          setAdamBatchMode(true);
          try {
            const adamBackend = getBackend(planNodes[action.nodeIndices[0]].device) ?? backend;
            const adamOp = adamBackend.ops.adamStep!;
            // Single profile begin/end for entire batch (not per-node)
            setCurrentOpLabel("adamStep");
            setProfileModule("optimizer.step");
            const _adamBatchT0 = profileOpBegin("adamStep");
            for (const nodeIdx of action.nodeIndices) {
              const adamNode = planNodes[nodeIdx];
              if (adamNode.result) continue;
              // Indexed getInputStorage (no .map() array allocation)
              const inputs = adamNode.inputs;
              const s0 = getInputStorage(inputs[0], adamBackend);
              const s1 = getInputStorage(inputs[1], adamBackend);
              const s2 = getInputStorage(inputs[2], adamBackend);
              const s3 = getInputStorage(inputs[3], adamBackend);
              // Direct adamOp call (bypasses executeOp switch)
              const adamPayload = adamNode.payload as import("../backend/types").AdamStepConfig;
              const adamResult = await adamOp(
                s0.backendTensor, s1.backendTensor, s2.backendTensor, s3.backendTensor,
                adamPayload,
              );
              // Adam is in-place: param buffer is returned as result.
              // Set ownsBuffer: false since the buffer is shared with the input.
              const paramResult = adamResult.param;
              const paramOwns = (paramResult as { ownsBuffer?: boolean }).ownsBuffer;
              const finalResult = paramOwns === true
                ? { ...paramResult, ownsBuffer: false } as BackendTensor
                : paramResult;
              // Side outputs (m, v) — create storage handles
              const mStorage = createStorageHandle(adamNode.device, adamResult.m);
              const vStorage = createStorageHandle(adamNode.device, adamResult.v);
              const sideOutputs = { m: mStorage, v: vStorage };
              storageTracker.markReachable(mStorage.id, sideOutputs);
              storageTracker.markReachable(vStorage.id, sideOutputs);
              (adamNode as any)._adamSideOutputs = sideOutputs;
              // Adam param is always input[1] — use s1.id directly (no findIndex)
              adamNode.result = createStorageHandle(adamNode.device, finalResult, s1.id);
            }
            profileOpEnd("adamStep", _adamBatchT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          } finally {
            setAdamBatchMode(false);
          }

          // Record adam batch as a single replay entry (must re-execute each step).
          // Capture the sequence counter positions so adam dispatches hit the correct
          // cache positions during replay.
          if (shouldRecord) {
            while (recordingDispatchIdx < recordingBuffer.length) {
              recordingDispatchIdx++;
            }
            replayEntries.push({
              kind: "adam-batch",
              nodeIndices: action.nodeIndices,
              seqCounters: adamSeqCountersBefore!,
            });
          }
          break;
        }

        case "sequential":
        case "view":
        case "data-source": {
          const nodeIdx = action.nodeIndex;
          const node = planNodes[nodeIdx];
          if (node.result) break;

          // Capture arena resolve index and sequence counters BEFORE execution
          const arenaResolveIdxBefore = shouldRecord ? getArenaResolveIndex() : 0;
          const seqCountersBefore = shouldRecord ? getDispatchSequenceCounters() : undefined;

          const nodeBackend = getBackend(node.device) ?? backend;
          const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
          const backendInputs = inputs.map(s => s.backendTensor);

          let resultTensor = await executeOp(node, backendInputs, nodeBackend);
          const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
          if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
            resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
          }
          const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
          const baseStorageId = isView && inputs.length > 0
            ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
            : undefined;
          node.result = createStorageHandle(node.device, resultTensor, baseStorageId);
          stats.sequentialNodes++;

          // Record for replay cache
          if (shouldRecord) {
            if (action.kind === "data-source") {
              // Data sources must re-execute each step (host data changes).
              replayEntries.push({ kind: "data-source", nodeIndex: nodeIdx, arenaResolveIdx: arenaResolveIdxBefore, seqCounters: seqCountersBefore! });
              while (recordingDispatchIdx < recordingBuffer.length) {
                recordingDispatchIdx++;
              }
            } else if (action.kind === "view") {
              // Views produce deterministic results (same arena buffer, same
              // shape/strides/offset). Cache the result to skip re-execution.
              const bt = node.result!.backendTensor as any;
              replayEntries.push({
                kind: "view",
                nodeIndex: nodeIdx,
                arenaResolveIdx: arenaResolveIdxBefore,
                arenaResolveIdxAfter: getArenaResolveIndex(),
                cachedResult: {
                  buffer: bt.buffer,
                  shape: bt.shape?.slice() ?? [],
                  dtype: bt.dtype ?? "f32",
                  size: bt.size ?? 0,
                  strides: bt.strides?.slice() ?? [],
                  offset: bt.offset ?? 0,
                  isContiguous: bt.isContiguous ?? true,
                },
              });
              while (recordingDispatchIdx < recordingBuffer.length) {
                recordingDispatchIdx++;
              }
            } else if (ENCODER_COPY_OPS.has(node.op)) {
              // Ops that use encoder copy commands (copyBufferToBuffer) alongside
              // compute dispatches must be re-executed during replay — the copy
              // commands are invisible to the compute dispatch recording mechanism.
              replayEntries.push({ kind: "sequential", nodeIndex: nodeIdx, arenaResolveIdx: arenaResolveIdxBefore, seqCounters: seqCountersBefore! });
              while (recordingDispatchIdx < recordingBuffer.length) {
                recordingDispatchIdx++;
              }
            } else {
              // With arena active, all buffer identities are stable (arena returns
              // same GPUBuffer object on subsequent steps). Data-source-consuming
              // ops can be safely replayed — their bind groups reference arena
              // buffers that don't change identity.
              // Safe to replay: capture dispatches
              while (recordingDispatchIdx < recordingBuffer.length) {
                replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
                recordingDispatchIdx++;
              }
              // Record node result (inline for ordering)
              if (node.result) {
                const bt = node.result.backendTensor as any;
                replayEntries.push({ kind: "result", nodeResult: {
                  nodeIndex: nodeIdx,
                  buffer: bt.buffer,
                  shape: bt.shape?.slice() ?? [],
                  dtype: bt.dtype ?? "f32",
                  size: bt.size ?? 0,
                  strides: bt.strides?.slice() ?? [],
                  isView: isView,
                }});
              }
              // Record side output for fusedAttentionForward so replay
              // restores _attnSideOutput (needed by extractAttentionLogsumexp
              // in a later plan that skips re-executing this node).
              if ((node as any)._attnSideOutput) {
                const soSH = (node as any)._attnSideOutput as StorageHandle;
                const sobt = soSH.backendTensor as any;
                replayEntries.push({ kind: "side-output", nodeIndex: nodeIdx,
                  buffer: sobt.buffer,
                  shape: sobt.shape?.slice() ?? [],
                  dtype: sobt.dtype ?? "f32",
                  size: sobt.size ?? 0,
                  strides: sobt.strides?.slice() ?? [],
                });
              }
            }
          }
          break;
        }

        case "prologue-skip": {
          // Prologue-claimed cast nodes are skipped — their work is absorbed
          // into the matmul tile load.
          break;
        }

        case "reclaim": {
          if (useTopLevelSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }
          if (shouldRecord) {
            replayEntries.push({ kind: "reclaim" });
          }
          break;
        }
      }
    }
  } finally {
    // Stop recording
    if (shouldRecord) {
      stopDispatchRecording();
      const [matmulMod, fusionMod] = await Promise.all([
        import("../backend/webgpu/matmul/dispatch"),
        import("../backend/webgpu/fusion-dispatch"),
      ]);
      matmulMod.setMatmulRecordingBuffer(null);
      fusionMod.setFusionRecordingBuffer(null);

      // Collect all GPUBuffers referenced by recorded bind groups.
      // These must be pinned (not destroyed) between steps so replay bind groups stay valid.
      const pinnedBuffers = new Set<any>();
      for (const entry of replayEntries) {
        if (entry.kind === "dispatch" && entry.dispatch.buffers) {
          for (const b of entry.dispatch.buffers) pinnedBuffers.add(b);
        }
        if (entry.kind === "result" && entry.nodeResult.buffer) {
          pinnedBuffers.add(entry.nodeResult.buffer);
        }
        if (entry.kind === "side-output" && entry.buffer) {
          pinnedBuffers.add(entry.buffer);
        }
      }

      // Activate pinning globally — buffers survive across markStep() boundaries.
      // Multiple plans accumulate into the same global set.
      addReplayPinnedBuffers(pinnedBuffers);

      // Build and store the dispatch replay cache
      loweredPlan.dispatchCache = {
        entries: replayEntries,
        valid: true,
        pinnedBuffers,
      };
    }

    // Deactivate buffer arena before closing encoder
    if (options.bufferArena && useTopLevelSharedEncoder) {
      clearActiveArena();
    }
    if (useTopLevelSharedEncoder) {
      endSharedEncoder();
    }
  }

  // Get the result from the last original plan node
  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Lowered plan execution failed: no result for last node");
  }

  return { result: lastNode.result, stats };
}

/**
 * Execute a fused segment using a fused kernel.
 */
async function executeFusedSegment(
  group: FusionGroup,
  recipe: ReturnType<typeof groupToRecipe>,
  backend: Backend,
  enableVectorization: boolean,
): Promise<void> {
  // For WebGPU, use the fused kernel dispatcher
  if (backend.name === "webgpu" && "dispatchFusedKernel" in backend) {
    await executeFusedWebGPU(
      group,
      recipe,
      backend as any,
      enableVectorization,
    );
    return;
  }

  // For CPU or other backends, fall back to sequential execution
  await executeSequentialSegment(group.nodes, backend);
}

/**
 * Execute a fused segment on WebGPU using generated kernels.
 */
async function executeFusedWebGPU(
  group: FusionGroup,
  recipe: ReturnType<typeof groupToRecipe>,
  backend: Backend & { device?: unknown },
  enableVectorization: boolean,
): Promise<void> {
  // Import fusion dispatch and buffer lifecycle helpers (cached on first call)
  const fusionDispatch = await import("../backend/webgpu/fusion-dispatch");
  const { dispatchFusedKernel } = fusionDispatch;
  await ensureWebGPUMatmulImports();
  const { deferredDestroyBuffer } = _webgpuMatmulImports!;

  // Get WebGPU device from backend
  const device = (backend as any).device;
  if (!device) {
    // No device available - fall back to sequential
    recordFusionFallback("no_device", group.nodes.length);
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  // Check storage buffer limit before attempting fusion.
  // Each fused kernel needs inputs + outputs storage bindings.
  // Inlined constants don't consume binding slots.
  // If we'd exceed the device limit, skip fusion silently (no console.warn spam).
  const maxStorageBuffers = device.limits?.maxStorageBuffersPerShaderStage ?? 8;
  const numOutputs = recipe.outputs?.length ?? 1;
  const nonInlinedInputCount = recipe.inputs.filter((inp: any) => !inp.isInlinedConstant).length;
  const requiredBindings = nonInlinedInputCount + numOutputs;
  if (requiredBindings > maxStorageBuffers) {
    recordFusionFallback("binding_limit", group.nodes.length, { required: requiredBindings, max: maxStorageBuffers });
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  // Prepare inputs from external refs, skipping inlined constants
  const inputs: Array<{ buffer: unknown; shape: number[]; dtype: DType }> = [];
  const tempContiguousCopies: Array<{ destroy?: () => void }> = [];
  for (let inputIdx = 0; inputIdx < group.externalInputs.length; inputIdx++) {
    // Skip inlined constants — their values are baked into the shader
    // This handles both scalar LazyRefs and pending nodes detected as inlinable
    if (recipe.inputs[inputIdx]?.isInlinedConstant) {
      continue;
    }

    const inputRef = group.externalInputs[inputIdx];
    // Scalar refs should always be inlined — this is a safety fallback
    if (inputRef.kind === "scalar") {
      continue;
    }
    const storage =
      inputRef.kind === "materialized"
        ? inputRef.storage
        : inputRef.node.result;

    if (!storage) {
      // Input not materialized - fall back to sequential
      recordFusionFallback("not_materialized", group.nodes.length);
      await executeSequentialSegment(group.nodes, backend);
      return;
    }

    const tensor = storage.backendTensor as any;
    // Fusion requires contiguous inputs — strided/offset layouts not supported by codegen
    if (tensor.isContiguous === false || (tensor.offset != null && tensor.offset > 0)) {
      // Auto-materialize to contiguous rather than abandoning fusion
      if (backend.ops.contiguous) {
        const contig = backend.ops.contiguous(tensor) as any;
        tempContiguousCopies.push(contig);
        inputs.push({
          buffer: contig.buffer,
          shape: contig.shape ?? tensor.shape ?? [1],
          dtype: (contig.dtype as DType) ?? (tensor.dtype as DType) ?? "f32",
        });
        continue;
      }
      // No contiguous op — fall back
      recordFusionFallback("non_contiguous", group.nodes.length, {
        shape: tensor.shape, isContiguous: tensor.isContiguous, offset: tensor.offset,
      });
      await executeSequentialSegment(group.nodes, backend);
      return;
    }
    inputs.push({
      buffer: tensor.buffer,
      shape: tensor.shape ?? [1],
      dtype: (tensor.dtype as DType) ?? "f32",
    });
  }

  // Check if any input buffer exceeds maxStorageBufferBindingSize
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const hasOversizedBuffer = inputs.some(
    (inp) => (inp.buffer as { size?: number }).size! > maxBindingSize,
  );
  if (hasOversizedBuffer) {
    recordFusionFallback("oversized_buffer", group.nodes.length, { maxBindingSize });
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  try {
    // Set module context for profiling from the output node
    setProfileModule(group.outputNode.module ?? "unknown");
    // Dispatch the fused kernel
    const result = dispatchFusedKernel(device, recipe, inputs as any, {
      vectorize: enableVectorization,
    });

    // Store the result in the output node
    const outputNode = group.outputNode;
    const fusionBuffer = result.buffer as GPUBuffer;
    const fusionBufferSize = (fusionBuffer as unknown as { size: number }).size ?? 0;
    let fusionDestroyed = false;
    outputNode.result = createStorageHandle(outputNode.device, {
      buffer: result.buffer,
      shape: result.shape,
      dtype: result.dtype,
      size: result.shape.reduce((a, b) => a * b, 1),
      strides: computeContiguousStrides(result.shape),
      offset: 0,
      isContiguous: true,
      ownsBuffer: true,
      destroy() {
        if (fusionDestroyed) return;
        fusionDestroyed = true;
        deferredDestroyBuffer(fusionBuffer, fusionBufferSize);
      },
    } as BackendTensor);

    // Multi-output: store results for additional output nodes (§15.2)
    if (group.additionalOutputNodes && result.outputs) {
      for (let i = 0; i < group.additionalOutputNodes.length; i++) {
        const addNode = group.additionalOutputNodes[i];
        const addOutput = result.outputs[i + 1]; // +1: primary is at index 0
        if (addOutput) {
          const addBuffer = addOutput.buffer as GPUBuffer;
          const addBufferSize = (addBuffer as unknown as { size: number }).size ?? 0;
          let addDestroyed = false;
          addNode.result = createStorageHandle(addNode.device, {
            buffer: addOutput.buffer,
            shape: addOutput.shape,
            dtype: addOutput.dtype,
            size: addOutput.shape.reduce((a, b) => a * b, 1),
            strides: computeContiguousStrides(addOutput.shape),
            offset: 0,
            isContiguous: true,
            ownsBuffer: true,
            destroy() {
              if (addDestroyed) return;
              addDestroyed = true;
              deferredDestroyBuffer(addBuffer, addBufferSize);
            },
          } as BackendTensor);
        }
      }
    }
    // Clean up temporary contiguous copies (deferred destroy for GPU fence)
    for (const temp of tempContiguousCopies) {
      if (temp.destroy) temp.destroy();
    }

    // Re-execute intermediates that are consumed outside the group but
    // couldn't be promoted to additional outputs (shape mismatch / binding limit).
    // The fused kernel computed the chain inline; we re-execute just the needed
    // nodes so external consumers can access their results.
    if (group.neededIntermediates && group.neededIntermediates.length > 0) {
      await executeSequentialSegment(group.neededIntermediates, backend);
    }
  } catch (e) {
    // Clean up temporary contiguous copies on failure too
    for (const temp of tempContiguousCopies) {
      if (temp.destroy) temp.destroy();
    }
    // Fusion failed - fall back to sequential
    recordFusionFallback("exception", group.nodes.length, { error: String(e) });
    console.warn("Fusion dispatch failed, falling back to sequential:", e);
    await executeSequentialSegment(group.nodes, backend);
  }
}

// Note: OPS_NEEDING_FLUSH was removed. All ops (including matmul, reductions,
// gather, etc.) go through submitOrCollect() which respects the shared encoder
// collection. No flush is needed before them — CBs are submitted in order at
// the end of the scope or at read() boundaries.

/**
 * Execute nodes sequentially (standard execution).
 */
async function executeSequentialSegment(
  nodes: LazyIRNode[],
  backend: Backend,
  externalNodeIds?: Set<number>,
  allPlanNodes?: LazyIRNode[],
): Promise<void> {
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    // Build consumer count map for reduction preamble detection
    const reductionConsumerCount = new Map<number, number>();
    if (backend.name === "webgpu") {
      for (const n of (allPlanNodes ?? nodes)) {
        for (const ref of n.inputs) {
          if (ref.kind === "pending") {
            reductionConsumerCount.set(ref.node.id, (reductionConsumerCount.get(ref.node.id) ?? 0) + 1);
          }
        }
      }
    }

    for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
      const node = nodes[nodeIdx];
      if (node.result) {
        continue;
      }

      // Try matmul epilogue fusion (Phase 1)
      if (node.op === "matmul" && backend.name === "webgpu") {
        const epiloguePlan = detectMatmulEpilogue(nodes, nodeIdx, allPlanNodes ?? nodes, externalNodeIds);
        if (epiloguePlan) {
          const epLabel = "matmul+" + epiloguePlan.epilogueOps.map(o => o.kind).join("+");
          setCurrentOpLabel(epLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(epLabel);
          try {
            await executeMatmulWithEpilogue(node, epiloguePlan, backend);
          } finally {
            profileOpEnd(epLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }
          nodeIdx += epiloguePlan.consumedCount - 1;
          continue;
        }
      }

      // Try reduction preamble fusion (Phase 3)
      if (isFusibleOp(node.op) && backend.name === "webgpu") {
        const reductionPlan = detectReductionPreamble(nodes, nodeIdx, reductionConsumerCount);
        if (reductionPlan) {
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.op}`;
          setCurrentOpLabel(rpLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(rpLabel);
          try {
            await executeReductionWithPreamble(reductionPlan, backend);
          } finally {
            profileOpEnd(rpLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }
          nodeIdx += 1; // Skip the reduction node (consumed 2 nodes total)
          continue;
        }
      }

      const nodeBackend = getBackend(node.device) ?? backend;
      const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
      const backendInputs = inputs.map((s) => s.backendTensor);

      let resultTensor = await executeOp(node, backendInputs, nodeBackend);
      const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
      if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
        resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
      }
      const isView =
        (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
      const baseStorageId =
        isView && inputs.length > 0
          ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
          : undefined;
      node.result = createStorageHandle(node.device, resultTensor, baseStorageId);
    }
  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }
}

/**
 * Execute nodes sequentially with early buffer release support.
 */
async function executeSequentialSegmentWithEarlyRelease(
  nodes: LazyIRNode[],
  backend: Backend,
  enableEarlyRelease: boolean,
  lifetimes: Map<number, TensorLifetime> | null,
  outputNodeIds: Set<number> | null,
  alreadyReleased: Set<number>,
  nodeToStorage: Map<number, StorageHandle>,
  startStep: number,
  externalNodeIds?: Set<number>,
  allPlanNodes?: LazyIRNode[],
  matmulPrologueMap?: Map<number, MatmulPrologueInfo[]>,
  prologueSkipIds?: Set<number>,
  prebuiltConsumerCount?: Map<number, number>,
  loweredPlanBuilder?: LoweredPlanBuilder | null,
  nodeIdToFinalPos?: Map<number, number>,
): Promise<void> {
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    // Use pre-built consumer count if provided, otherwise build it.
    // The consumer count is used for both reduction preamble and matmul epilogue detection.
    // Building it once per plan (in the caller) instead of once per segment saves ~15ms.
    const reductionConsumerCount = prebuiltConsumerCount ?? new Map<number, number>();
    if (!prebuiltConsumerCount && backend.name === "webgpu") {
      for (const n of (allPlanNodes ?? nodes)) {
        for (const ref of n.inputs) {
          if (ref.kind === "pending") {
            reductionConsumerCount.set(ref.node.id, (reductionConsumerCount.get(ref.node.id) ?? 0) + 1);
          }
        }
      }
    }

    // Intra-segment periodic reclamation: flush pending buffers to main pool
    // every N nodes so freed intermediates can be reused within the same segment.
    // Uses the safe pattern: flushSharedEncoder() submits all encoded work, then
    // flushBufferPool() moves pending→pool. Subsequent dispatches encode on a
    // fresh encoder, and WebGPU queue ordering guarantees prior work completes first.
    const INTRA_SEGMENT_RECLAIM_INTERVAL = DEFAULT_RECLAIM_INTERVAL;
    let nodesSinceIntraReclaim = 0;

    let step = startStep;
    for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
      const node = nodes[nodeIdx];
      if (node.result) {
        if (enableEarlyRelease) {
          nodeToStorage.set(node.id, node.result);
        }
        step++;
        continue;
      }

      // Skip prologue-claimed cast nodes — their work is absorbed into the
      // matmul tile load. They stay in the plan for topological ordering but
      // don't execute. No result is set since the only consumer (the matmul)
      // uses the pre-cast input via prologue info.
      if (prologueSkipIds?.has(node.id)) {
        if (loweredPlanBuilder && nodeIdToFinalPos) {
          loweredPlanBuilder.recordPrologueSkip(nodeIdToFinalPos.get(node.id)!);
        }
        step++;
        continue;
      }

      // Try matmul epilogue/prologue fusion (Phase 1)
      if (node.op === "matmul" && backend.name === "webgpu") {
        let epiloguePlan = prebuiltConsumerCount
          ? detectMatmulEpilogueCore(nodes, nodeIdx, prebuiltConsumerCount, externalNodeIds)
          : detectMatmulEpilogue(nodes, nodeIdx, allPlanNodes ?? nodes, externalNodeIds);
        const prologues = matmulPrologueMap?.get(node.id);

        // If we have prologues but no epilogue, create a minimal (empty) epilogue plan
        // so the matmul goes through the epilogue dispatch path with prologue support.
        if (!epiloguePlan && prologues && prologues.length > 0) {
          epiloguePlan = {
            consumedCount: 1, // just the matmul itself
            epilogueOps: [],
            epilogueInputRefs: [],
            outputDtype: node.dtype,
            outputNode: node,
          };
        }

        // Attach prologues to the plan
        if (epiloguePlan && prologues && prologues.length > 0) {
          epiloguePlan.prologues = prologues;
        }

        if (epiloguePlan) {
          const prologueLabel = epiloguePlan.prologues ? "prologue+" : "";
          const epilogueLabel = epiloguePlan.epilogueOps.length > 0
            ? "+" + epiloguePlan.epilogueOps.map(o => o.kind).join("+")
            : "";
          const epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(/\+$/, "");
          setCurrentOpLabel(epLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(epLabel);
          try {
            await executeMatmulWithEpilogue(node, epiloguePlan, backend);
          } finally {
            profileOpEnd(epLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // Track storages for all consumed nodes and release dead buffers
          if (enableEarlyRelease) {
            for (let skip = 0; skip < epiloguePlan.consumedCount; skip++) {
              const consumedNode = nodes[nodeIdx + skip];
              if (consumedNode.result) {
                nodeToStorage.set(consumedNode.id, consumedNode.result);
              }
              step++;
              if (lifetimes && outputNodeIds) {
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
            }
          } else {
            step += epiloguePlan.consumedCount;
          }

          // Record matmul epilogue action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const covered: number[] = [];
            for (let skip = 0; skip < epiloguePlan.consumedCount; skip++) {
              covered.push(nodeIdToFinalPos.get(nodes[nodeIdx + skip].id)!);
            }
            loweredPlanBuilder.recordMatmulEpilogue(
              nodeIdToFinalPos.get(node.id)!,
              covered,
              nodeIdToFinalPos.get(epiloguePlan.outputNode.id)!,
              epiloguePlan.epilogueOps,
              epiloguePlan.epilogueInputRefs.length,
              epiloguePlan.outputDtype,
              epiloguePlan.consumedCount,
              epiloguePlan.prologues?.map(p => ({
                inputIndex: p.inputIndex,
                castNodeIndex: nodeIdToFinalPos.get(p.castNodeId)!,
                fromDtype: p.fromDtype,
                toDtype: p.toDtype,
              })),
            );
          }

          nodeIdx += epiloguePlan.consumedCount - 1;
          continue;
        }
      }

      // Try reduction preamble fusion (Phase 3)
      if (isFusibleOp(node.op) && backend.name === "webgpu") {
        const reductionPlan = detectReductionPreamble(nodes, nodeIdx, reductionConsumerCount);
        if (reductionPlan) {
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.op}`;
          setCurrentOpLabel(rpLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(rpLabel);
          try {
            await executeReductionWithPreamble(reductionPlan, backend);
          } finally {
            profileOpEnd(rpLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // Track storages for both consumed nodes (preamble + reduction)
          if (enableEarlyRelease) {
            // Preamble node: no result (consumed), but step still advances
            step++;
            if (lifetimes && outputNodeIds) {
              const deadNodeIds = findDeadTensorsAtStep(lifetimes, step, outputNodeIds, alreadyReleased);
              for (const deadId of deadNodeIds) {
                const storage = nodeToStorage.get(deadId);
                if (storage && canSafelyRelease(storage, nodeToStorage)) {
                  releaseBufferImmediate(storage);
                  nodeToStorage.delete(deadId);
                  alreadyReleased.add(deadId);
                }
              }
            }
            // Reduction node: has the result
            const reductionNode = nodes[nodeIdx + 1];
            if (reductionNode.result) {
              nodeToStorage.set(reductionNode.id, reductionNode.result);
            }
            step++;
            if (lifetimes && outputNodeIds) {
              const deadNodeIds = findDeadTensorsAtStep(lifetimes, step, outputNodeIds, alreadyReleased);
              for (const deadId of deadNodeIds) {
                const storage = nodeToStorage.get(deadId);
                if (storage && canSafelyRelease(storage, nodeToStorage)) {
                  releaseBufferImmediate(storage);
                  nodeToStorage.delete(deadId);
                  alreadyReleased.add(deadId);
                }
              }
            }
          } else {
            step += 2;
          }

          // Record reduction preamble action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            loweredPlanBuilder.recordReductionPreamble(
              nodeIdToFinalPos.get(node.id)!,
              nodeIdToFinalPos.get(nodes[nodeIdx + 1].id)!,
            );
          }

          nodeIdx += 1; // Skip the reduction node (consumed 2 nodes total)
          continue;
        }
      }

      // Batch consecutive adamStep nodes: flush once before the batch, then
      // execute all Adam nodes without per-op flushes. All 76 params are
      // independent — a single pre-flush resolves all read→read_write conflicts.
      if (node.op === "adamStep" && useSharedEncoder) {
        // Count consecutive adamStep nodes
        let adamCount = 1;
        for (let j = nodeIdx + 1; j < nodes.length; j++) {
          if (nodes[j].op === "adamStep" && !nodes[j].result) adamCount++;
          else break;
        }

        if (adamCount > 1) {
          // Single flush before the entire Adam batch
          flushSharedEncoder();
          flushBufferPool(); // Make released fwd/bwd buffers available for Adam allocs

          setAdamBatchMode(true);
          try {
            for (let a = 0; a < adamCount; a++) {
              const adamNode = nodes[nodeIdx + a];
              if (adamNode.result) {
                if (enableEarlyRelease) nodeToStorage.set(adamNode.id, adamNode.result);
                step++;
                continue;
              }

              const adamBackend = getBackend(adamNode.device) ?? backend;
              const adamInputs = adamNode.inputs.map(ref => getInputStorage(ref, adamBackend));
              const adamBackendInputs = adamInputs.map((s) => s.backendTensor);

              setCurrentOpLabel("adamStep");
              setProfileModule(adamNode.module ?? "unknown");
              const _profT0 = profileOpBegin("adamStep");
              let adamResult: BackendTensor;
              try {
                adamResult = await executeOp(adamNode, adamBackendInputs, adamBackend);
              } finally {
                profileOpEnd("adamStep", _profT0);
                setCurrentOpLabel(null);
                setProfileModule("unknown");
              }

              const adamAliasedIdx = adamBackendInputs.findIndex(b => b === adamResult);
              if (adamAliasedIdx >= 0 && (adamResult as { ownsBuffer?: boolean }).ownsBuffer === true) {
                adamResult = { ...adamResult, ownsBuffer: false } as BackendTensor;
              }
              const adamIsView = (adamResult as { ownsBuffer?: boolean }).ownsBuffer === false;
              const adamBaseId = adamIsView && adamInputs.length > 0
                ? adamInputs[adamAliasedIdx >= 0 ? adamAliasedIdx : 0].id
                : undefined;
              adamNode.result = createStorageHandle(adamNode.device, adamResult, adamBaseId);

              if (enableEarlyRelease) {
                nodeToStorage.set(adamNode.id, adamNode.result);
                step++;
                if (lifetimes && outputNodeIds) {
                  const deadNodeIds = findDeadTensorsAtStep(lifetimes, step, outputNodeIds, alreadyReleased);
                  for (const deadId of deadNodeIds) {
                    const storage = nodeToStorage.get(deadId);
                    if (storage && canSafelyRelease(storage, nodeToStorage)) {
                      releaseBufferImmediate(storage);
                      nodeToStorage.delete(deadId);
                      alreadyReleased.add(deadId);
                    }
                  }
                }
              } else {
                step++;
              }
            }
          } finally {
            setAdamBatchMode(false);
          }

          // Record adam batch action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const adamIndices: number[] = [];
            for (let a = 0; a < adamCount; a++) {
              adamIndices.push(nodeIdToFinalPos.get(nodes[nodeIdx + a].id)!);
            }
            loweredPlanBuilder.recordAdamBatch(adamIndices);
          }

          nodeIdx += adamCount - 1;
          continue;
        }
      }

      const nodeBackend = getBackend(node.device) ?? backend;
      const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
      const backendInputs = inputs.map((s) => s.backendTensor);

      let resultTensor = await executeOp(node, backendInputs, nodeBackend);
      const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
      if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
        resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
      }
      const isView =
        (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
      const baseStorageId =
        isView && inputs.length > 0
          ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
          : undefined;
      node.result = createStorageHandle(node.device, resultTensor, baseStorageId);

      // Record action in lowered plan builder
      if (loweredPlanBuilder && nodeIdToFinalPos) {
        const finalPos = nodeIdToFinalPos.get(node.id)!;
        if (isDataSourceOp(node.op)) {
          loweredPlanBuilder.recordDataSource(finalPos);
        } else if (isViewOp(node.op)) {
          loweredPlanBuilder.recordView(finalPos);
        } else {
          loweredPlanBuilder.recordSequential(finalPos);
        }
      }

      // Track storage and release dead buffers
      if (enableEarlyRelease) {
        nodeToStorage.set(node.id, node.result);
        step++;

        if (lifetimes && outputNodeIds) {
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
      } else {
        step++;
      }

      // Periodic intra-segment reclamation
      nodesSinceIntraReclaim++;
      if (useSharedEncoder && enableEarlyRelease && nodesSinceIntraReclaim >= INTRA_SEGMENT_RECLAIM_INTERVAL) {
        flushSharedEncoder();
        flushBufferPool();
        if (loweredPlanBuilder) loweredPlanBuilder.recordReclaim();
        nodesSinceIntraReclaim = 0;
      }
    }
  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }
}

// ============================================================================
// Matmul Epilogue Fusion (Phase 1)
// Detects matmul → cast/bias/activation chains and fuses them into a single
// dispatchMatmulWithEpilogue call, eliminating intermediate buffers.
// ============================================================================

interface MatmulPrologueInfo {
  /** Which matmul input has the cast (0 = A, 1 = B) */
  inputIndex: 0 | 1;
  /** The cast node ID (for tracking) */
  castNodeId: number;
  /** The cast's input ref (the original f32 tensor) */
  originalInputRef: LazyRef;
  /** Source dtype of the cast input (e.g. "f32") */
  fromDtype: DType;
  /** Target dtype the matmul expects (e.g. "f16") */
  toDtype: DType;
}

interface MatmulEpiloguePlan {
  /** Number of nodes consumed (matmul + epilogue ops) */
  consumedCount: number;
  /** Epilogue operations to fuse */
  epilogueOps: Array<{ kind: string; toDtype?: DType; inputIndex?: number; op?: string }>;
  /** Additional inputs required by epilogue (e.g. bias tensor) */
  epilogueInputRefs: LazyRef[];
  /** Output dtype after epilogue */
  outputDtype: DType;
  /** The final output node in the chain */
  outputNode: LazyIRNode;
  /** Input-side cast prologues absorbed into the matmul */
  prologues?: MatmulPrologueInfo[];
}

/**
 * Core matmul epilogue chain detection logic.
 * Starting from a matmul node at `startIdx` in `nodes`, walks forward to find
 * a chain of fusible ops (cast, bias add, activations) that can be merged
 * into a single matmul dispatch with epilogue.
 *
 * @param nodes - Array of nodes to walk (full plan or segment)
 * @param startIdx - Index of the matmul node
 * @param consumerCount - Pre-computed map of node ID → number of consumers
 * @param externalNodeIds - Node IDs with external references (saved-for-backward etc.)
 */
function detectMatmulEpilogueCore(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): MatmulEpiloguePlan | null {
  const matmulNode = nodes[startIdx];
  if (matmulNode.op !== "matmul") return null;

  const epilogueOps: MatmulEpiloguePlan["epilogueOps"] = [];
  const epilogueInputRefs: LazyRef[] = [];
  let additionalInputCount = 0;
  let chainLength = 0;
  let currentNode = matmulNode;
  let outputDtype = matmulNode.dtype;

  // Walk forward from matmul, matching epilogue-compatible ops
  let currentIsReshapeSkip = false;
  for (let i = startIdx + 1; i < nodes.length && chainLength < 4; i++) {
    const nextNode = nodes[i];

    // Check that the candidate node's primary input (input[0]) is a pending ref
    // to the current chain node
    if (nextNode.inputs.length === 0) break;
    let chainInputIdx = 0;
    const primaryInput = nextNode.inputs[0];
    if (primaryInput.kind !== "pending" || primaryInput.node.id !== currentNode.id) {
      // For commutative binary ops, check if the chain continues via inputs[1]
      if ((nextNode.op === "add" || nextNode.op === "mul") && nextNode.inputs.length === 2) {
        const altInput = nextNode.inputs[1];
        if (altInput.kind === "pending" && altInput.node.id === currentNode.id) {
          chainInputIdx = 1;
        } else {
          break;
        }
      } else {
        break;
      }
    }

    // Check that the current chain node is NOT externally referenced
    // (must be used only by this next node).
    // Skip this check for nodes already absorbed into the chain as reshape
    // skips — their live pending tensors will be disposed by the caller.
    if (externalNodeIds?.has(currentNode.id) && !currentIsReshapeSkip) break;
    const consumers = consumerCount.get(currentNode.id) ?? 0;
    if (consumers > 1) break;

    // Skip reshape that only removes leading size-1 dims (e.g. [1,M,N]→[M,N])
    if (nextNode.op === "reshape") {
      const curShape = currentNode.shape;
      const nextShape = nextNode.shape;
      if (curShape.length === nextShape.length + 1
          && curShape[0] === 1
          && curShape.slice(1).every((d: number, i: number) => d === nextShape[i])) {
        chainLength++;
        currentNode = nextNode;
        currentIsReshapeSkip = true;
        outputDtype = nextNode.dtype || outputDtype;
        continue;
      }
      break;
    }
    currentIsReshapeSkip = false;

    // Match the op type
    let matched = false;

    if (nextNode.op === "cast") {
      const payload = nextNode.payload as { dtype: DType } | undefined;
      if (payload) {
        epilogueOps.push({ kind: "cast", toDtype: payload.dtype });
        outputDtype = payload.dtype;
        matched = true;
      }
    } else if (nextNode.op === "add" && nextNode.inputs.length === 2) {
      // Check if second input is a 1D bias (size matches matmul N dimension)
      const secondInput = nextNode.inputs[chainInputIdx === 0 ? 1 : 0];
      let secondShape: number[] | undefined;
      if (secondInput.kind === "materialized") {
        secondShape = secondInput.storage.backendTensor.shape;
      } else if (secondInput.kind === "pending") {
        secondShape = secondInput.node.shape;
      }

      if (secondShape && secondShape.length === 1) {
        // 1D bias — broadcast add
        if (additionalInputCount >= 4) break; // binding limit
        epilogueOps.push({ kind: "bias", inputIndex: additionalInputCount });
        epilogueInputRefs.push(secondInput);
        additionalInputCount++;
        matched = true;
      } else {
        // General binary add
        if (additionalInputCount >= 4) break;
        epilogueOps.push({ kind: "binary", op: "add", inputIndex: additionalInputCount });
        epilogueInputRefs.push(secondInput);
        additionalInputCount++;
        matched = true;
      }
    } else if (nextNode.op === "mul" && nextNode.inputs.length === 2) {
      if (additionalInputCount >= 4) break;
      epilogueOps.push({ kind: "binary", op: "mul", inputIndex: additionalInputCount });
      epilogueInputRefs.push(nextNode.inputs[chainInputIdx === 0 ? 1 : 0]);
      additionalInputCount++;
      matched = true;
    } else if (nextNode.op === "relu" || nextNode.op === "silu" || nextNode.op === "sigmoid" || nextNode.op === "tanh") {
      epilogueOps.push({ kind: "unary", op: nextNode.op });
      matched = true;
    } else if (nextNode.op === "gelu") {
      const geluPayload = nextNode.payload as { approximate?: string } | undefined;
      if (geluPayload?.approximate === "tanh") {
        epilogueOps.push({ kind: "gelu" });
      } else {
        // gelu with approximate="none" uses erf — expressed as unary "gelu_erf"
        epilogueOps.push({ kind: "unary", op: "gelu_erf" });
      }
      matched = true;
    }

    if (!matched) break;

    chainLength++;
    currentNode = nextNode;
    outputDtype = nextNode.dtype || outputDtype;
  }

  // Need at least one epilogue op to be worthwhile
  if (chainLength === 0) return null;

  return {
    consumedCount: 1 + chainLength, // matmul + epilogue ops
    epilogueOps,
    epilogueInputRefs,
    outputDtype,
    outputNode: currentNode,
  };
}

/**
 * Pre-scan variant: detect matmul epilogue from full plan (for pre-segmentation scan).
 */
function detectMatmulEpilogueFromPlan(
  planNodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): MatmulEpiloguePlan | null {
  return detectMatmulEpilogueCore(planNodes, startIdx, consumerCount, externalNodeIds);
}

/**
 * Segment variant: detect matmul epilogue within a segment.
 * Also checks the full plan for consumer counts.
 */
function detectMatmulEpilogue(
  nodes: LazyIRNode[],
  startIdx: number,
  allPlanNodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
): MatmulEpiloguePlan | null {
  // Build consumer count from full plan for accurate "only one consumer" checks
  const consumerCount = new Map<number, number>();
  for (const node of allPlanNodes) {
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        consumerCount.set(input.node.id, (consumerCount.get(input.node.id) ?? 0) + 1);
      }
    }
  }
  return detectMatmulEpilogueCore(nodes, startIdx, consumerCount, externalNodeIds);
}

/**
 * Detect simple last-2-dim transpose views (mirrors detectSimpleTranspose in index.ts).
 * Returns the effective contiguous shape and whether a transpose was detected.
 * Used by the matmul epilogue cache to compute correct geometry.
 */
function _detectTransposeView(tensor: any): { shape: number[]; transposed: boolean } {
  const shape = tensor.shape as number[];
  const strides = tensor.strides as number[] | undefined;
  const rank = shape.length;

  if (tensor.isContiguous || (tensor.offset ?? 0) !== 0 || rank < 2 || !strides) {
    return { shape: shape.slice(), transposed: false };
  }

  // Check for last-2-dim transpose: strides[-2] === 1 and strides[-1] === shape[-2]
  if (strides[rank - 2] === 1 && strides[rank - 1] === shape[rank - 2]) {
    // Verify batch dimensions are contiguous
    let expectedStride = shape[rank - 1] * shape[rank - 2];
    let valid = true;
    for (let i = rank - 3; i >= 0; i--) {
      if (strides[i] !== expectedStride) { valid = false; break; }
      expectedStride *= shape[i];
    }
    if (valid) {
      // Swap last 2 dims back to get original contiguous shape
      const origShape = shape.slice();
      origShape[rank - 2] = shape[rank - 1];
      origShape[rank - 1] = shape[rank - 2];
      return { shape: origShape, transposed: true };
    }
  }

  return { shape: shape.slice(), transposed: false };
}

/**
 * Execute a matmul with fused epilogue operations.
 * Uses the existing dispatchMatmulWithEpilogue from the WebGPU backend.
 */
async function executeMatmulWithEpilogue(
  matmulNode: LazyIRNode,
  plan: MatmulEpiloguePlan,
  backend: Backend,
): Promise<void> {
  // Use cached imports (initialized once at executeLoweredPlan/executePlanOptimized entry)
  await ensureWebGPUMatmulImports();
  const { dispatchMatmulWithEpilogue } = _webgpuMatmulImports!;

  // Determine which inputs have prologue casts.
  // If the prologue cast was skipped (no result), use the original pre-cast input.
  // If the cast ran (e.g., in a fusion group), use the normal cast output instead.
  let inputCastA: DType | undefined;
  let inputCastB: DType | undefined;
  let resolvedInputRefA = matmulNode.inputs[0];
  let resolvedInputRefB = matmulNode.inputs[1];
  if (plan.prologues) {
    for (const p of plan.prologues) {
      // Check if the cast node's result was computed (e.g., via fusion group)
      const castRef = p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
      const castAlreadyRan = castRef.kind === "pending" && castRef.node.result != null;
      if (!castAlreadyRan) {
        // Cast was skipped — use the pre-cast f32 input and tell codegen about the cast
        if (p.inputIndex === 0) {
          resolvedInputRefA = p.originalInputRef;
          inputCastA = p.toDtype;
        } else {
          resolvedInputRefB = p.originalInputRef;
          inputCastB = p.toDtype;
        }
      }
      // If cast already ran, just use the normal matmul input (cast's f16 output)
    }
  }

  // Resolve matmul inputs
  const matmulInputA = getInputStorage(resolvedInputRefA);
  const matmulInputB = getInputStorage(resolvedInputRefB);

  // Resolve epilogue input refs
  const epilogueInputTensors: BackendTensor[] = [];
  for (const ref of plan.epilogueInputRefs) {
    const storage = getInputStorage(ref);
    epilogueInputTensors.push(storage.backendTensor);
  }

  // Build EpilogueConfig
  const epilogueConfig = {
    ops: plan.epilogueOps,
    additionalInputCount: plan.epilogueInputRefs.length,
    outputDtype: plan.outputDtype,
  };

  // Call dispatchMatmulWithEpilogue
  const resultTensor = dispatchMatmulWithEpilogue(
    matmulInputA.backendTensor as any,
    matmulInputB.backendTensor as any,
    epilogueConfig,
    epilogueInputTensors as any[],
    false, // transA
    false, // transB
    inputCastA,
    inputCastB,
  );

  // Store result on the final output node.
  // If epilogue absorbed a reshape (e.g. [1,M,N]→[M,N]), the matmul output shape
  // may differ from outputNode.shape. Fix up by mutating the tensor's shape/strides.
  const outNodeShape = plan.outputNode.shape;
  if (!shapesEqual(resultTensor.shape, outNodeShape)) {
    (resultTensor as any).shape = outNodeShape;
    // Recompute strides for the new shape (contiguous row-major)
    const newStrides = new Array(outNodeShape.length);
    let stride = 1;
    for (let i = outNodeShape.length - 1; i >= 0; i--) {
      newStrides[i] = stride;
      stride *= outNodeShape[i];
    }
    (resultTensor as any).strides = newStrides;
  }
  plan.outputNode.result = createStorageHandle(plan.outputNode.device, resultTensor);
}

// ============================================================================
// Reduction Preamble Fusion (Phase 3)
// Detects elementwise → sum/mean patterns and fuses them into a single
// sumDimWithPreamble call, eliminating the intermediate elementwise buffer.
// ============================================================================

interface ReductionPreamblePlan {
  /** The elementwise preamble node */
  preambleNode: LazyIRNode;
  /** The sum or mean reduction node */
  reductionNode: LazyIRNode;
  /** The elementwise op name (e.g., "mul", "exp", "add") */
  op: string;
  /** Number of inputs to the elementwise op (1=unary, 2=binary) */
  arity: number;
  /** Whether the reduction is a mean (divide by count after sum) */
  isMean: boolean;
}

/**
 * Detect an elementwise → sum/mean pattern suitable for reduction preamble fusion.
 *
 * Constraints:
 * - nodes[startIdx] must be a fusible elementwise op (unary or binary, not ternary)
 * - nodes[startIdx + 1] must be "sum" or "mean"
 * - The reduction's input must be a pending ref to the elementwise node
 * - The elementwise node must not be referenced by anything else
 * - All elementwise inputs must have the same shape as the elementwise output
 * - All inputs must be f32 dtype (preamble shader hardcodes f32)
 */
function detectReductionPreamble(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
): ReductionPreamblePlan | null {
  if (startIdx + 1 >= nodes.length) return null;

  const elemNode = nodes[startIdx];
  const nextNode = nodes[startIdx + 1];

  // 1. Check elementwise op is fusible and unary/binary (not ternary)
  if (!isFusibleOp(elemNode.op)) return null;
  if (elemNode.inputs.length > 2) return null; // Skip ternary (where)
  if (elemNode.op === "cast") return null; // Cast doesn't benefit from fusion here

  // 2. Check next node is sum or mean
  if (nextNode.op !== "sum" && nextNode.op !== "mean") return null;

  // 3. Check the reduction's primary input is a pending ref to the elementwise node
  if (nextNode.inputs.length < 1) return null;
  const reductionInput = nextNode.inputs[0];
  if (reductionInput.kind !== "pending" || reductionInput.node !== elemNode) return null;

  // 4. Check the elementwise output is only consumed by the reduction
  const consumers = consumerCount.get(elemNode.id) ?? 0;
  if (consumers !== 1) return null;

  // 5. All elementwise inputs must have the same shape as the elementwise output
  const elemShape = elemNode.shape;
  for (const ref of elemNode.inputs) {
    const inputNode = ref.kind === "pending" ? ref.node : null;
    if (inputNode && !shapesEqual(inputNode.shape, elemShape)) return null;
    // For materialized refs, we can't easily check shape at detection time.
    // The backend tensor will have the correct shape, so we rely on the
    // constraint that the lazy engine produces correct shapes.
  }

  // 6. All inputs must be f32 dtype (preamble shader hardcodes f32)
  if (elemNode.dtype !== "f32") return null;
  for (const ref of elemNode.inputs) {
    if (ref.kind === "pending" && ref.node.dtype !== "f32") return null;
  }

  // 7. The elementwise node must not already have a result
  if (elemNode.result) return null;

  const arity = elemNode.inputs.length;

  return {
    preambleNode: elemNode,
    reductionNode: nextNode,
    op: elemNode.op,
    arity,
    isMean: nextNode.op === "mean",
  };
}

function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Execute a fused reduction-with-preamble operation.
 */
async function executeReductionWithPreamble(
  plan: ReductionPreamblePlan,
  backend: Backend,
): Promise<void> {
  const { sumDimWithPreamble } = await import("../backend/webgpu/index");

  // Resolve all elementwise inputs
  const elemInputStorages = plan.preambleNode.inputs.map(ref => getInputStorage(ref, backend));
  const elemInputTensors = elemInputStorages.map(s => s.backendTensor);

  // Get sum options from the reduction node's payload
  const payload = plan.reductionNode.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;

  // Call sumDimWithPreamble
  let resultTensor = sumDimWithPreamble(elemInputTensors, plan.op, payload ?? {});

  // If this is a mean, divide by reduction size
  if (plan.isMean) {
    const inputShape = plan.preambleNode.shape;
    const dim = payload?.dim;
    let reductionSize: number;
    if (dim === undefined || dim === null) {
      reductionSize = inputShape.reduce((a, b) => a * b, 1);
    } else {
      const dims = Array.isArray(dim) ? dim : [dim];
      const rank = inputShape.length;
      reductionSize = dims.reduce((acc, d) => acc * inputShape[d < 0 ? d + rank : d], 1);
    }
    // Divide by reduction size using backend mul with scalar (1/reductionSize)
    const invSize = 1.0 / reductionSize;
    const sumResult = resultTensor;
    const invSizeTensor = backend.ops.tensorFromArray([invSize], []);
    resultTensor = backend.ops.mul(sumResult, invSizeTensor);
    // Destroy intermediate backend tensors (sum output + scalar) to prevent buffer leak
    (sumResult as { destroy?: () => void }).destroy?.();
    (invSizeTensor as { destroy?: () => void }).destroy?.();
  }

  // Store result on the reduction node
  plan.reductionNode.result = createStorageHandle(plan.reductionNode.device, resultTensor);
}

/**
 * Execute a single op on the backend.
 */
async function executeOp(
  node: LazyIRNode,
  backendInputs: BackendTensor[],
  backend: Backend,
): Promise<BackendTensor> {
  setCurrentOpLabel(node.op);
  const _profT0 = profileOpBegin(node.op);
  try {
  switch (node.op) {
    case "tensorFromArray": {
      const payload = node.payload as { values: number[] } | undefined;
      if (!payload?.values) {
        throw new Error("tensorFromArray requires values in payload");
      }
      return backend.ops.tensorFromArray(payload.values, node.shape);
    }

    case "zeros": {
      if (backend.ops.zeros) {
        return backend.ops.zeros(node.shape);
      }
      const numEl = node.shape.reduce((a: number, b: number) => a * b, 1);
      return backend.ops.tensorFromArray(new Array(numEl).fill(0), node.shape);
    }

    case "full": {
      const fullPayload = node.payload as { fillValue: number };
      if (backend.ops.full) {
        return backend.ops.full(node.shape, fullPayload.fillValue);
      }
      const numElFull = node.shape.reduce((a: number, b: number) => a * b, 1);
      return backend.ops.tensorFromArray(new Array(numElFull).fill(fullPayload.fillValue), node.shape);
    }

    case "arange": {
      const ap = node.payload as { end: number; start: number; step: number };
      if (backend.ops.arange) {
        return backend.ops.arange(ap.end, ap.start, ap.step);
      }
      const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
      const vals = new Array(n);
      for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
      return backend.ops.tensorFromArray(vals, node.shape);
    }

    case "tril": {
      if (!backend.ops.tril) throw new Error("tril not supported by backend");
      return backend.ops.tril(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "triu": {
      if (!backend.ops.triu) throw new Error("triu not supported by backend");
      return backend.ops.triu(backendInputs[0], (node.payload as { k: number })?.k ?? 0);
    }

    case "add":
      return backend.ops.add(backendInputs[0], backendInputs[1]);

    case "sub": {
      const subPayload = node.payload as { alpha?: number } | undefined;
      return backend.ops.sub(backendInputs[0], backendInputs[1], subPayload);
    }

    case "mul":
      return backend.ops.mul(backendInputs[0], backendInputs[1]);

    case "div": {
      const divPayload = node.payload as
        | { roundingMode?: "trunc" | "floor" }
        | undefined;
      return backend.ops.div(backendInputs[0], backendInputs[1], divPayload);
    }

    case "matmul":
      return backend.ops.matmul(backendInputs[0], backendInputs[1]);

    case "sqrt":
      return backend.ops.sqrt(backendInputs[0]);

    case "relu":
      return backend.ops.relu(backendInputs[0]);

    case "exp":
      if (!backend.ops.exp) throw new Error("exp not supported by backend");
      return backend.ops.exp(backendInputs[0]);

    case "log":
      if (!backend.ops.log) throw new Error("log not supported by backend");
      return backend.ops.log(backendInputs[0]);

    case "neg":
      if (!backend.ops.neg) throw new Error("neg not supported by backend");
      return backend.ops.neg(backendInputs[0]);

    case "abs":
      if (!backend.ops.abs) throw new Error("abs not supported by backend");
      return backend.ops.abs(backendInputs[0]);

    case "tanh":
      if (!backend.ops.tanh) throw new Error("tanh not supported by backend");
      return backend.ops.tanh(backendInputs[0]);

    case "sigmoid":
      if (!backend.ops.sigmoid)
        throw new Error("sigmoid not supported by backend");
      return backend.ops.sigmoid(backendInputs[0]);

    case "gelu": {
      if (!backend.ops.gelu) throw new Error("gelu not supported by backend");
      const geluOpts = node.payload as GeluOptions | undefined;
      return backend.ops.gelu(backendInputs[0], geluOpts);
    }

    case "silu":
      if (!backend.ops.silu) throw new Error("silu not supported by backend");
      return backend.ops.silu(backendInputs[0]);

    case "isfinite":
      if (!backend.ops.isfinite)
        throw new Error("isfinite not supported by backend");
      return backend.ops.isfinite(backendInputs[0]);

    case "reshape": {
      const payload = node.payload as { targetShape: number[] } | undefined;
      const targetShape = payload?.targetShape ?? node.shape;
      return backend.ops.reshape(backendInputs[0], targetShape);
    }

    case "expand":
      return backend.ops.expand(backendInputs[0], node.shape);

    case "transpose": {
      const payload = node.payload as
        | { dim0: number; dim1: number }
        | undefined;
      if (!payload) {
        throw new Error("transpose requires dim0 and dim1 in payload");
      }
      return backend.ops.transpose(backendInputs[0], payload);
    }

    case "permute": {
      const payload = node.payload as { dims: number[] } | undefined;
      if (!payload) {
        throw new Error("permute requires dims in payload");
      }
      return backend.ops.permute(backendInputs[0], payload.dims);
    }

    case "contiguous":
      return backend.ops.contiguous(backendInputs[0]);

    case "narrow": {
      const p = node.payload as { dim: number; start: number; length: number };
      if (!backend.ops.narrow) throw new Error("narrow not supported by backend");
      return backend.ops.narrow(backendInputs[0], p.dim, p.start, p.length);
    }

    case "narrowBackward": {
      const p = node.payload as { dim: number; start: number; originalLength: number };
      if (!backend.ops.narrowBackward) throw new Error("narrowBackward not supported by backend");
      return backend.ops.narrowBackward(backendInputs[0], p.dim, p.start, p.originalLength);
    }

    case "cast": {
      const payload = node.payload as
        | { dtype: import("../backend/types").DType }
        | undefined;
      if (!payload) {
        throw new Error("cast requires dtype in payload");
      }
      if (!backend.ops.cast) {
        throw new Error("cast not supported by backend");
      }
      return backend.ops.cast(backendInputs[0], payload.dtype);
    }

    case "gather": {
      const payload = node.payload as { dim: number } | undefined;
      if (!payload) {
        throw new Error("gather requires dim in payload");
      }
      return backend.ops.gather(backendInputs[0], backendInputs[1], payload);
    }

    case "scatterAdd": {
      const payload = node.payload as { dim: number } | undefined;
      if (!payload) {
        throw new Error("scatterAdd requires dim in payload");
      }
      return backend.ops.scatterAdd(
        backendInputs[0],
        backendInputs[1],
        backendInputs[2],
        payload,
      );
    }

    case "sum": {
      const payload = node.payload as
        | { dim?: number | number[] | null; keepdim?: boolean }
        | undefined;
      return backend.ops.sum(backendInputs[0], payload);
    }

    case "max": {
      const payload = node.payload as
        | { dim?: number | number[] | null; keepdim?: boolean }
        | undefined;
      return backend.ops.max(backendInputs[0], payload);
    }

    case "mean": {
      const payload = node.payload as
        | { dim?: number | number[] | null; keepdim?: boolean }
        | undefined;
      return backend.ops.mean(backendInputs[0], payload);
    }

    case "argmax": {
      const payload = node.payload as { dim: number; keepdim?: boolean };
      if (!backend.ops.argmax)
        throw new Error("argmax not supported by backend");
      return backend.ops.argmax(backendInputs[0], payload);
    }

    case "argmin": {
      const payload = node.payload as { dim: number; keepdim?: boolean };
      if (!backend.ops.argmin)
        throw new Error("argmin not supported by backend");
      return backend.ops.argmin(backendInputs[0], payload);
    }

    case "gt":
      if (!backend.ops.gt) throw new Error("gt not supported by backend");
      return backend.ops.gt(backendInputs[0], backendInputs[1]);

    case "lt":
      if (!backend.ops.lt) throw new Error("lt not supported by backend");
      return backend.ops.lt(backendInputs[0], backendInputs[1]);

    case "ge":
      if (!backend.ops.ge) throw new Error("ge not supported by backend");
      return backend.ops.ge(backendInputs[0], backendInputs[1]);

    case "le":
      if (!backend.ops.le) throw new Error("le not supported by backend");
      return backend.ops.le(backendInputs[0], backendInputs[1]);

    case "eq":
      if (!backend.ops.eq) throw new Error("eq not supported by backend");
      return backend.ops.eq(backendInputs[0], backendInputs[1]);

    case "ne":
      if (!backend.ops.ne) throw new Error("ne not supported by backend");
      return backend.ops.ne(backendInputs[0], backendInputs[1]);

    case "where":
      return backend.ops.where(
        backendInputs[0],
        backendInputs[1],
        backendInputs[2],
      );

    case "stridedScatterCopy": {
      const payload = node.payload as {
        offset: number;
        viewShape: number[];
        viewStrides: number[];
      };
      if (!payload) {
        throw new Error("stridedScatterCopy requires options in payload");
      }
      return backend.ops.stridedScatterCopy(
        backendInputs[0],
        backendInputs[1],
        payload,
      );
    }

    case "stridedScatterAdd": {
      const payload = node.payload as {
        offset: number;
        viewShape: number[];
        viewStrides: number[];
      };
      if (!payload) {
        throw new Error("stridedScatterAdd requires options in payload");
      }
      return backend.ops.stridedScatterAdd(
        backendInputs[0],
        backendInputs[1],
        payload,
      );
    }

    case "adamStep": {
      if (!backend.ops.adamStep) throw new Error("adamStep not supported by backend");
      const adamPayload = node.payload as import("../backend/types").AdamStepConfig;
      const adamResult = await backend.ops.adamStep(
        backendInputs[0], backendInputs[1], backendInputs[2], backendInputs[3],
        adamPayload,
      );
      const mStorage3 = createStorageHandle(node.device, adamResult.m);
      const vStorage3 = createStorageHandle(node.device, adamResult.v);
      const sideOutputs3 = { m: mStorage3, v: vStorage3 };
      storageTracker.markReachable(mStorage3.id, sideOutputs3);
      storageTracker.markReachable(vStorage3.id, sideOutputs3);
      (node as any)._adamSideOutputs = sideOutputs3;
      return adamResult.param;
    }

    case "unscaleGrad": {
      const unscalePayload3 = node.payload as { invScale: number; infFlagBuffer: unknown };
      if (!backend.ops.unscaleGrad) throw new Error("unscaleGrad not supported by backend");
      return backend.ops.unscaleGrad(
        backendInputs[0], unscalePayload3.invScale, unscalePayload3.infFlagBuffer,
      );
    }

    case "fusedAttentionForward": {
      const faPayload3 = node.payload as import("../backend/types").FusedAttentionConfig;
      if (!backend.ops.fusedAttentionForward) throw new Error("fusedAttentionForward not supported by backend");
      const faResult3 = backend.ops.fusedAttentionForward(
        backendInputs[0], backendInputs[1], backendInputs[2], faPayload3,
      );
      const lseSH3 = createStorageHandle(node.device, faResult3.logsumexp);
      (node as any)._attnSideOutput = lseSH3;
      return faResult3.output;
    }

    case "extractAttentionLogsumexp": {
      const parentFA3 = node.inputs[0].node;
      const lseSH3 = (parentFA3 as any)._attnSideOutput as StorageHandle | undefined;
      if (!lseSH3) throw new Error("extractAttentionLogsumexp: parent has no _attnSideOutput");
      const lseResult3 = lseSH3.backendTensor;
      storageTracker.unregister(lseSH3.id);
      (parentFA3 as any)._attnSideOutput = undefined;
      return lseResult3;
    }

    case "fusedAttentionBackward": {
      const faBwdPayload3 = node.payload as import("../backend/types").FusedAttentionConfig;
      if (!backend.ops.fusedAttentionBackward) throw new Error("fusedAttentionBackward not supported by backend");
      const faBwdResult3 = backend.ops.fusedAttentionBackward(
        backendInputs[0], backendInputs[1], backendInputs[2],
        backendInputs[3], backendInputs[4], backendInputs[5], faBwdPayload3,
      );
      const dkSH3 = createStorageHandle(node.device, faBwdResult3.dK);
      const dvSH3 = createStorageHandle(node.device, faBwdResult3.dV);
      (node as any)._attnBwdDK = dkSH3;
      (node as any)._attnBwdDV = dvSH3;
      return faBwdResult3.dQ;
    }

    case "extractAttentionDK": {
      const parentDK3 = node.inputs[0].node;
      const dkSH3 = (parentDK3 as any)._attnBwdDK as StorageHandle | undefined;
      if (!dkSH3) throw new Error("extractAttentionDK: parent node has no _attnBwdDK");
      const dkResult3 = dkSH3.backendTensor;
      storageTracker.unregister(dkSH3.id);
      (parentDK3 as any)._attnBwdDK = undefined;
      return dkResult3;
    }

    case "extractAttentionDV": {
      const parentDV3 = node.inputs[0].node;
      const dvSH3 = (parentDV3 as any)._attnBwdDV as StorageHandle | undefined;
      if (!dvSH3) throw new Error("extractAttentionDV: parent node has no _attnBwdDV");
      const dvResult3 = dvSH3.backendTensor;
      storageTracker.unregister(dvSH3.id);
      (parentDV3 as any)._attnBwdDV = undefined;
      return dvResult3;
    }

    case "fusedCrossEntropyForward": {
      const cePayload5 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
      if (!backend.ops.fusedCrossEntropyForward) throw new Error("fusedCrossEntropyForward not supported by backend");
      return backend.ops.fusedCrossEntropyForward(
        backendInputs[0], backendInputs[1], cePayload5,
      );
    }

    case "fusedCrossEntropyBackward": {
      const cePayload6 = node.payload as import("../backend/types").FusedCrossEntropyConfig;
      if (!backend.ops.fusedCrossEntropyBackward) throw new Error("fusedCrossEntropyBackward not supported by backend");
      return backend.ops.fusedCrossEntropyBackward(
        backendInputs[0], backendInputs[1], backendInputs[2], cePayload6,
      );
    }

    case "fusedLayerNormForward": {
      const lnPayload5 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!backend.ops.fusedLayerNormForward) throw new Error("fusedLayerNormForward not supported by backend");
      return backend.ops.fusedLayerNormForward(
        backendInputs[0], backendInputs[1], backendInputs[2], lnPayload5,
      );
    }

    case "fusedLayerNormBackwardGradX": {
      const lnPayload6 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!backend.ops.fusedLayerNormBackwardGradX) throw new Error("fusedLayerNormBackwardGradX not supported by backend");
      return backend.ops.fusedLayerNormBackwardGradX(
        backendInputs[0], backendInputs[1], backendInputs[2], lnPayload6,
      );
    }

    case "fusedLayerNormBackwardGradWeightBias": {
      const lnGWBPayload3 = node.payload as import("../backend/types").FusedLayerNormConfig;
      if (!backend.ops.fusedLayerNormBackwardGradWeightBias) {
        throw new Error("fusedLayerNormBackwardGradWeightBias not supported by backend");
      }
      const gwResult3 = backend.ops.fusedLayerNormBackwardGradWeightBias(
        backendInputs[0], backendInputs[1], lnGWBPayload3,
      );
      // Store raw gradBias BackendTensor for extractLnBwdGradBias
      (node as any)._lnBwdSideOutput = gwResult3.gradBias;
      return gwResult3.gradWeight;
    }

    case "extractLnBwdGradBias": {
      const parentNode3 = node.inputs[0].node;
      const sideOutputBT3 = (parentNode3 as any)._lnBwdSideOutput as BackendTensor | undefined;
      if (!sideOutputBT3) {
        throw new Error("extractLnBwdGradBias: parent node has no _lnBwdSideOutput");
      }
      (parentNode3 as any)._lnBwdSideOutput = undefined;
      return sideOutputBT3;
    }

    case "transfer": {
      // For transfer ops, we need the source storage
      throw new Error("Transfer ops should be handled in executePlan");
    }

    default:
      throw new Error(`Unknown op: ${node.op}`);
  }
  } finally {
    profileOpEnd(node.op, _profT0);
    setCurrentOpLabel(null);
    setProfileModule("unknown");
  }
}

/**
 * Compute contiguous strides for a shape.
 */
function computeContiguousStrides(shape: number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}
