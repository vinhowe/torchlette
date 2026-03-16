import { OP_DTYPE_RULES, promoteDtype } from "../backend/op-registry";
import { getActiveBackend, getBackend } from "../backend/registry";
import {
  type ArgReduceOptions,
  type Backend,
  type CatOptions,
  type DeviceKind,
  type DivOptions,
  type DType,
  type FusedAttentionConfig,
  type FusedCrossEntropyConfig,
  type FusedLayerNormConfig,
  type FusedRMSNormConfig,
  type GatherOptions,
  type GeluOptions,
  type MaxOptions,
  type MeanOptions,
  normalizeDim,
  type ScatterAddOptions,
  type StridedScatterOptions,
  type SubOptions,
  type SumOptions,
  type TransposeOptions,
} from "../backend/types";
import {
  broadcastShapes,
  broadcastThreeShapes,
  contiguousStrides,
} from "../core/shape";
import type { AutocastContext } from "../engine/amp";
import {
  executePlanOptimized,
  type OptimizedExecutionStats,
} from "../engine/executor-lowered";
import {
  executePlanSegmented,
  executePlanSequential,
} from "../engine/executor-sequential";
import {
  createMaterializedRef,
  createPendingRef,
  createScalarRef,
  type LazyIRNode,
  type LazyOpCode,
  type LazyRef,
} from "../engine/lazy-types";
import { isDataSourceOp } from "../engine/lowered-plan";
import { createLazyIRNode } from "../engine/node-factory";
import { buildMergedPlan } from "../engine/plan-builder";
import { storageTracker } from "../engine/storage-tracker";
import { type BaseId, createBaseId, Tensor } from "./tensor";

// Re-export all types from engine-types (EngineTensor, SavedTensorRecord, etc.)
export * from "../engine/engine-types";

// Local imports from engine-types (used by Engine methods merged into RuntimeEngine)
import {
  type AsyncScope,
  type BaseState,
  type BaseStateInfo,
  type BaseId as EngineBaseId,
  type EngineMemoryStats,
  EngineTensor,
  type ExecLock,
  type MemorySnapshot,
  type MemoryStatsProvider,
  type RngBasis,
  type RngDrawRecord,
  type RngDrawResult,
  type SavedTensorInfo,
  type SavedTensorRecord,
  type TensorOrigin,
  type TidyScope,
} from "../engine/engine-types";

// ── Engine helpers ──────────────────────────────────────────────────────────
function collectTensorHandles(value: unknown): EngineTensor[] {
  const out: EngineTensor[] = [];
  const seen = new Set<unknown>();
  const visit = (current: unknown) => {
    if (current == null) return;
    if (current instanceof EngineTensor) {
      out.push(current);
      return;
    }
    if (typeof current !== "object") return;
    if (seen.has(current)) return;
    seen.add(current);
    if (Array.isArray(current)) {
      for (const entry of current) visit(entry);
      return;
    }
    for (const entry of Object.values(current as Record<string, unknown>))
      visit(entry);
  };
  visit(value);
  return out;
}

function computeRngValue(
  basis: RngBasis,
  opNonce: number,
  drawNonce: number,
): number {
  const seed = basis.seed >>> 0;
  const algo = basis.algorithmId >>> 0;
  const op = opNonce >>> 0;
  const draw = drawNonce >>> 0;
  let v =
    (seed ^ Math.imul(algo, 0x9e3779b9) ^ op ^ Math.imul(draw, 0x85ebca6b)) >>>
    0;
  v ^= v >>> 16;
  v = Math.imul(v, 0x7feb352d);
  v ^= v >>> 15;
  v = Math.imul(v, 0x846ca68b);
  v ^= v >>> 16;
  return (v >>> 0) / 2 ** 32;
}

// ── Error classes (from Engine) ─────────────────────────────────────────────
export class EngineBusyError extends Error {
  name = "EngineBusyError";
}
export class AsyncInCompileError extends Error {
  name = "AsyncInCompileError";
}
export class SavedTensorModifiedError extends Error {
  name = "SavedTensorModifiedError";
}
export class NonReentrantBackwardError extends Error {
  name = "NonReentrantBackwardError";
}
export class PoisonedEngineError extends Error {
  name = "PoisonedEngineError";
}
export class RngReplayExhaustedError extends Error {
  name = "RngReplayExhaustedError";
}
export class RngReplayMismatchError extends Error {
  name = "RngReplayMismatchError";
}

// ── Engine types (merged from engine-types.ts) ─────────────────────────────
/** A tensor or a numeric scalar (will be inlined as a constant in fused kernels). */
export type TensorOrScalar = Tensor | number;

/** Lightweight reference to a lazy computation with shape/dtype metadata. */
type OpInput = Pick<Tensor, "lazyRef" | "shape" | "dtype" | "device">;

/** Dispatch mode: receives notifications when RuntimeTensors are created or escape a scope. */
export interface DispatchMode {
  onTensorCreated(tensor: Tensor): void;
  onTensorEscaped?(tensor: Tensor): void;
}

/** Dispatch mode that tracks RuntimeTensors created within a tidy scope. */
export class TidyDispatchMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  readonly escaped = new Set<Tensor>();
  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }
  onTensorEscaped(tensor: Tensor): void {
    this.escaped.add(tensor);
  }
  disposeNonEscaped(): void {
    for (const t of this.tracked) {
      if (!this.escaped.has(t) && !t.disposed) t.dispose();
    }
  }
}

export interface RuntimeEngineOptions {
  enableFusion?: boolean;
  enableVectorization?: boolean;
  enableEarlyRelease?: boolean;
  enableCheckpointSegmentation?: boolean;
  enableTrueSegmentation?: boolean;
}

/** Internal dispatch mode used by startIntermediateTracking/stopIntermediateTracking. */
class IntermediateTrackingMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }
}

// ── Shape helpers (merged from shape-helpers.ts) ────────────────────────────
export { broadcastShapes };

function matmulShape(a: number[], b: number[]): number[] {
  if (a.length < 1 || b.length < 1)
    throw new Error("matmul requires at least 1D tensors");
  // Validate inner dimension match
  const kA = a[a.length - 1];
  const kB = b.length === 1 ? b[0] : b[b.length - 2];
  if (kA !== kB) {
    throw new Error(
      `matmul: inner dimension mismatch — [${a.join(",")}] @ [${b.join(",")}] (${kA} != ${kB})`,
    );
  }
  if (a.length === 1 && b.length === 1) return [];
  if (a.length === 1) return [...b.slice(0, -2), b[b.length - 1]];
  if (b.length === 1) return a.slice(0, -1);
  const m = a[a.length - 2];
  const n = b[b.length - 1];
  const batch = broadcastShapes(a.slice(0, -2), b.slice(0, -2));
  return [...batch, m, n];
}

function transposeShape(shape: number[], options: TransposeOptions): number[] {
  const result = shape.slice();
  const temp = result[options.dim0];
  result[options.dim0] = result[options.dim1];
  result[options.dim1] = temp;
  return result;
}

function reduceShape(
  shape: number[],
  dim: number | number[] | null | undefined,
  keepdim: boolean,
): number[] {
  if (dim == null) return keepdim ? shape.map(() => 1) : [];
  const dims = Array.isArray(dim) ? dim : [dim];
  const normalizedDims = dims.map((d) => (d < 0 ? shape.length + d : d));
  if (keepdim) return shape.map((s, i) => (normalizedDims.includes(i) ? 1 : s));
  return shape.filter((_, i) => !normalizedDims.includes(i));
}

/** Build a cast LazyRef without creating a Tensor (no lifecycle, no registration). */
function castRef(input: OpInput, toDtype: DType): OpInput {
  const node = createLazyIRNode(
    "cast",
    [input.lazyRef],
    input.shape.slice(),
    toDtype,
    input.device,
    { dtype: toDtype },
  );
  return {
    lazyRef: createPendingRef(node),
    shape: input.shape,
    dtype: toDtype,
    device: input.device,
  };
}

// ── Force-method helpers ────────────────────────────────────────────────────

/** Collect pending lazy nodes from tensors that still need execution. */
function collectPendingRoots(tensors: readonly Tensor[]): LazyIRNode[] {
  const roots: LazyIRNode[] = [];
  for (const t of tensors) {
    if (t.isMaterialized() || t.disposed) continue;
    const ref = t.lazyRef;
    if (ref.kind === "pending") roots.push(ref.node);
  }
  return roots;
}

/** Find the device of the first non-materialized tensor (defaults to "cpu"). */
function resolveDeviceFromTensors(tensors: readonly Tensor[]): DeviceKind {
  for (const t of tensors) {
    if (!t.isMaterialized()) return t.device;
  }
  return "cpu";
}

/**
 * Materialize tensors whose lazy nodes have results but haven't been materialized yet.
 * Returns any nodes that were materialized (for cleanup of skipped nodes).
 */
function materializeRemaining(tensors: readonly Tensor[]): LazyIRNode[] {
  const materialized: LazyIRNode[] = [];
  for (const t of tensors) {
    if (t.isMaterialized() || t.disposed) continue;
    const ref = t.lazyRef;
    if (ref.kind === "pending") {
      const idx = ref.outputIndex ?? 0;
      const storage = idx === 0 ? ref.node.result : ref.node.results?.[idx];
      if (storage) {
        t._materialize(storage);
        materialized.push(ref.node);
      }
    }
  }
  return materialized;
}

export class RuntimeEngine {
  private defaultDevice: DeviceKind | null = null;
  private fusionEnabled = false;
  private vectorizationEnabled = false;
  private lastFusionStats: OptimizedExecutionStats | null = null;
  private cumulativeFusionStats: OptimizedExecutionStats | null = null;
  private earlyReleaseEnabled = false;
  private checkpointSegmentationEnabled = false;
  private _rngCounter = 0;
  private trueSegmentationEnabled = false;
  private _webgpuFlushBufferPool: (() => void) | null = null;

  /** Cache plan fingerprints by cheap structural key to avoid O(N) hashing. */

  /** Stack of active dispatch modes (notified on tensor creation/escape). */
  private dispatchModes: DispatchMode[] = [];

  // ── Engine fields (merged from Engine class) ──────────────────────────────
  private readonly _baseState = new Map<EngineBaseId, BaseState>();
  private readonly _execLock: ExecLock = { held: false, ownerId: 0, depth: 0 };
  private _nextOwnerId = 1;
  private _poisoned = false;
  private _nextSavedTensorId = 1;
  private _backwardActive = false;
  private _nextTensorId = 1;
  private _nextBaseId = 1;
  private _nextScopeId = 1;
  private _nextMutIdValue = 1;
  private readonly _tidyScopes: TidyScope[] = [];
  private _asyncScopeContext: AsyncScope | null = null;
  private readonly _basePinCount = new Map<EngineBaseId, number>();
  private _rngBasis: RngBasis = { algorithmId: 0, seed: 0 };
  private _rngDrawNonce = 0;
  private _rngCheckpointMode: "record" | "replay" | null = null;
  private _rngCheckpointDraws: RngDrawRecord[] = [];
  private _rngCheckpointIndex = 0;
  private _autocastContext: AutocastContext | null = null;
  private readonly _savedTensors = new Map<number, SavedTensorInfo>();
  private readonly _memorySnapshots: MemorySnapshot[] = [];
  private _memoryStatsProvider: MemoryStatsProvider | null = null;

  constructor(backendName?: DeviceKind, options?: RuntimeEngineOptions) {
    if (backendName) {
      this.defaultDevice = backendName;
    }
    if (options?.enableFusion !== undefined) {
      this.fusionEnabled = options.enableFusion;
    }
    if (options?.enableVectorization !== undefined) {
      this.vectorizationEnabled = options.enableVectorization;
    }
    if (options?.enableEarlyRelease !== undefined) {
      this.earlyReleaseEnabled = options.enableEarlyRelease;
    }
    if (options?.enableCheckpointSegmentation !== undefined) {
      this.checkpointSegmentationEnabled = options.enableCheckpointSegmentation;
    }
    if (options?.enableTrueSegmentation !== undefined) {
      this.trueSegmentationEnabled = options.enableTrueSegmentation;
    }
  }

  /**
   * Enable or disable fusion optimization (§15).
   */
  setFusionEnabled(enabled: boolean): void {
    this.fusionEnabled = enabled;
  }

  /**
   * Check if fusion is enabled.
   */
  isFusionEnabled(): boolean {
    return this.fusionEnabled;
  }

  /**
   * Get statistics from the last optimized execution.
   */
  getLastFusionStats(): OptimizedExecutionStats | null {
    return this.lastFusionStats;
  }

  /**
   * Get cumulative fusion statistics across all force() calls since the last
   * markStep(). This is useful because lastFusionStats only captures the most
   * recent force() call, which may be the optimizer step (with no fusion).
   */
  getCumulativeFusionStats(): OptimizedExecutionStats | null {
    return this.cumulativeFusionStats;
  }

  /**
   * Reset cumulative fusion stats. Called at markStep() boundaries.
   */
  resetCumulativeFusionStats(): void {
    this.cumulativeFusionStats = null;
  }

  /**
   * Accumulate fusion stats from a single execution into the cumulative total.
   */
  private accumulateFusionStats(stats: OptimizedExecutionStats): void {
    if (!this.cumulativeFusionStats) {
      this.cumulativeFusionStats = { ...stats };
    } else {
      this.cumulativeFusionStats.totalNodes += stats.totalNodes;
      this.cumulativeFusionStats.fusedNodes += stats.fusedNodes;
      this.cumulativeFusionStats.sequentialNodes += stats.sequentialNodes;
      this.cumulativeFusionStats.fusionGroups += stats.fusionGroups;
      this.cumulativeFusionStats.fusionEnabled = stats.fusionEnabled;
    }
  }

  /**
   * Enable or disable early buffer release during execution.
   * This releases intermediate buffers as soon as they're no longer needed,
   * allowing for better memory reuse within a single plan execution.
   */
  setEarlyReleaseEnabled(enabled: boolean): void {
    this.earlyReleaseEnabled = enabled;
  }

  /**
   * Check if early release is enabled.
   */
  isEarlyReleaseEnabled(): boolean {
    return this.earlyReleaseEnabled;
  }

  // ============================================================================
  // Dispatch Mode Stack
  // ============================================================================

  pushDispatchMode(mode: DispatchMode): void {
    this.dispatchModes.push(mode);
  }

  popDispatchMode(): DispatchMode {
    const mode = this.dispatchModes.pop();
    if (!mode) throw new Error("No dispatch mode to pop");
    return mode;
  }

  hasDispatchMode(): boolean {
    return this.dispatchModes.length > 0;
  }

  markEscaped(tensor: Tensor): void {
    for (const mode of this.dispatchModes) {
      mode.onTensorEscaped?.(tensor);
    }
  }

  // ============================================================================
  // Intermediate Tensor Tracking (backward pass memory management)
  // ============================================================================

  /**
   * Start tracking intermediate tensors created during backward computations.
   * Implemented as a dispatch mode — supports nesting.
   */
  startIntermediateTracking(): void {
    this.pushDispatchMode(new IntermediateTrackingMode());
  }

  /**
   * Stop tracking and return all tracked intermediate tensors.
   * The caller should dispose tensors that are not needed (e.g., not returned gradients).
   */
  stopIntermediateTracking(): Set<Tensor> {
    const mode = this.popDispatchMode();
    if (!(mode instanceof IntermediateTrackingMode)) {
      throw new Error(
        "Expected IntermediateTrackingMode on dispatch mode stack",
      );
    }
    return mode.tracked;
  }

  /**
   * Create a tensor and track it as an intermediate if tracking is active.
   * Helper to reduce boilerplate in all op implementations.
   */
  private createAndTrack(
    baseId: BaseId,
    lazyRef: LazyRef,
    shape: number[],
    device: DeviceKind,
    dtype: DType = "f32",
  ): Tensor {
    const tensor = new Tensor(baseId, lazyRef, shape, device, dtype);
    for (let i = 0; i < this.dispatchModes.length; i++) {
      this.dispatchModes[i].onTensorCreated(tensor);
    }
    return tensor;
  }

  /** Shorthand: create a tracked tensor from a lazy ref. */
  private trackRef(
    ref: LazyRef,
    shape: number[],
    device: DeviceKind,
    dtype?: DType,
  ): Tensor {
    return this.createAndTrack(createBaseId(), ref, shape, device, dtype);
  }

  /** Shorthand: create a tracked tensor from a lazy IR node. */
  private trackNode(
    node: LazyIRNode,
    shape: number[],
    device: DeviceKind,
    dtype?: DType,
  ): Tensor {
    return this.trackRef(createPendingRef(node), shape, device, dtype);
  }

  setBackend(name: DeviceKind): Backend {
    const backend = getBackend(name);
    if (!backend) {
      throw new Error(`Unknown backend: ${name}`);
    }
    this.defaultDevice = name;
    return backend;
  }

  /**
   * Get the current default device.
   */
  get currentDefaultDevice(): DeviceKind {
    return this.defaultDevice ?? getActiveBackend().name;
  }

  getBackend(device?: DeviceKind): Backend {
    const resolved = device ?? this.defaultDevice ?? getActiveBackend().name;
    const backend = getBackend(resolved);
    if (!backend) {
      throw new Error(`Unknown backend: ${resolved}`);
    }
    return backend;
  }

  private getDevice(device?: DeviceKind): DeviceKind {
    return device ?? this.defaultDevice ?? getActiveBackend().name;
  }

  private assertSameDevice(...tensors: Tensor[]): DeviceKind {
    const device = tensors[0]?.device;
    if (!device) {
      throw new Error("Missing tensor device");
    }
    for (const tensor of tensors) {
      if (tensor.device !== device) {
        throw new Error("Tensors must be on the same device");
      }
    }
    return device;
  }

  private assertShapeMatch(op: string, dst: Tensor, src: Tensor): void {
    if (dst.shape.length !== src.shape.length) {
      throw new Error(
        `${op}: shape mismatch - dst has rank ${dst.shape.length}, src has rank ${src.shape.length}`,
      );
    }
    for (let i = 0; i < dst.shape.length; i++) {
      if (dst.shape[i] !== src.shape[i]) {
        throw new Error(
          `${op}: shape mismatch at dim ${i} - dst has ${dst.shape[i]}, src has ${src.shape[i]}`,
        );
      }
    }
  }

  /**
   * Force a tensor to materialize by executing its computation graph.
   */
  async force(tensor: Tensor): Promise<void> {
    if (tensor.isMaterialized() || tensor.disposed) return;
    if (tensor.lazyRef.kind !== "pending") return;
    await this.forceAllMerged(tensor);
  }

  /**
   * Get the buffer pool flush function for the given device.
   * Returns a no-op for devices without buffer pools.
   */
  private getBufferPoolFlushFn(device: DeviceKind): () => void {
    if (device === "webgpu") {
      if (this._webgpuFlushBufferPool) return this._webgpuFlushBufferPool;
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const webgpu = require("../backend/webgpu");
        this._webgpuFlushBufferPool = webgpu.flushBufferPool ?? (() => {});
        return this._webgpuFlushBufferPool!;
      } catch {
        return () => {};
      }
    }
    return () => {};
  }

  /**
   * Force multiple tensors to materialize using a single merged execution plan.
   *
   * This is critical for unified backward execution: when multiple checkpointed
   * layers' recomputations are collected, this method builds ONE merged plan
   * containing all their computation graphs. If true segmentation is enabled
   * and checkpoint boundaries exist in the merged plan, the execution will
   * segment at those boundaries with GPU sync between segments, enabling
   * memory savings.
   */
  async forceAllMerged(...tensors: Tensor[]): Promise<void> {
    const pendingRoots = collectPendingRoots(tensors);
    if (pendingRoots.length === 0) return;

    const plan = buildMergedPlan(pendingRoots);
    if (plan.nodes.length === 0) return;

    const device = resolveDeviceFromTensors(tensors);

    // Data-source-only plans skip the template/arena/lowered-plan path.
    const allDataSource = plan.nodes.every((n) => isDataSourceOp(n.op));

    const backend = this.getBackend(device);

    // Check if plan has checkpoint boundaries - only segment if checkpointing is used
    const hasCheckpointBoundaries = plan.nodes.some(
      (n) => n.isCheckpointBoundary,
    );

    // Use fusion when enabled and on WebGPU, regardless of segmentation settings.
    // executePlanOptimized supports enableEarlyRelease for memory management.
    if (this.fusionEnabled && !allDataSource) {
      const optimizedResult = await executePlanOptimized(plan, backend, {
        enableFusion: true,
        enableVectorization: this.vectorizationEnabled,
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
      this.lastFusionStats = optimizedResult.stats;
      this.accumulateFusionStats(optimizedResult.stats);
    } else if (
      this.trueSegmentationEnabled &&
      hasCheckpointBoundaries &&
      device === "webgpu"
    ) {
      // True segmentation with GPU sync between segments
      await executePlanSegmented(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
        gpuSync: true,
      });
    } else if (this.checkpointSegmentationEnabled && hasCheckpointBoundaries) {
      // Checkpoint segmentation (buffer pool flush only, no GPU sync)
      await executePlanSegmented(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
        flushBufferPoolFn: this.getBufferPoolFlushFn(device),
      });
    } else {
      // Standard execution - no segmentation overhead
      await executePlanSequential(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
    }

    // Materialize ALL tensors that were pending on executed nodes.
    // This ensures all user-held tensors get their storages marked as externally reachable.
    const { materializePendingTensors } = await import("./tensor");
    for (const node of plan.nodes) {
      if (node.result) {
        materializePendingTensors(node.id, node.result, node.results);
      }
    }

    // Materialize any remaining tensors whose nodes were already executed
    materializeRemaining(tensors);
  }

  /**
   * Force all pending tensors to materialize.
   * Called at markStep to ensure all pending work is done before cleanup.
   *
   * Uses a single merged plan for efficiency. Skips nodes that were already
   * executed (have results from previous force() calls) since their results
   * are still accessible via node.result in getInputStorage().
   */
  async forceAllPending(): Promise<void> {
    const { getAllPendingTensors, materializePendingTensors: materialize } =
      await import("./tensor");
    const pendingTensors = getAllPendingTensors();
    if (pendingTensors.length === 0) {
      return;
    }

    const pendingRoots = collectPendingRoots(pendingTensors);
    if (pendingRoots.length === 0) return;

    const plan = buildMergedPlan(pendingRoots, /* skipExecuted */ true);
    if (plan.nodes.length === 0) return;

    // Destroy storages from prior steps (e.g., old Adam m/v states
    // disposed by _resolvePendingState). Frees GPU buffers into pool
    // before the new plan allocates output buffers.
    storageTracker.destroyUnreachable();

    const device = resolveDeviceFromTensors(pendingTensors);

    // Flush recycled GPU buffers from pendingRelease to the main pool.
    // Must happen BEFORE the shared encoder opens (executePlanSequential opens one),
    // so that acquire() can find them in the main pool during plan execution
    // (acquire() skips pendingRelease when sharedEncoder is active).
    const flushBufferPool = this.getBufferPoolFlushFn(device);
    flushBufferPool();

    // Snapshot storage ID before execution to scope cleanup of orphaned intermediates
    const storageSnapshot = storageTracker.getNextStorageId();

    const backend = this.getBackend(device);

    // Use optimized execution with fusion when enabled
    if (this.fusionEnabled) {
      const optimizedResult = await executePlanOptimized(plan, backend, {
        enableFusion: true,
        enableVectorization: this.vectorizationEnabled,
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
      this.lastFusionStats = optimizedResult.stats;
      this.accumulateFusionStats(optimizedResult.stats);
    } else {
      await executePlanSequential(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
    }

    // Materialize all tensors pending on executed nodes
    for (const node of plan.nodes) {
      if (node.result) {
        materialize(node.id, node.result, node.results);
      }
    }
    // Materialize remaining tensors and collect skipped nodes (already executed
    // by prior force() calls but not in plan.nodes) for result cleanup.
    const skippedNodes = materializeRemaining(pendingTensors);

    // Drop node.result references to allow GC of unclaimed intermediate storages.
    // Note: node.results is NOT cleared here — multi-output ops (e.g., adamStep)
    // store side outputs (m/v) in node.results that the optimizer reads on the
    // next step via _resolvePendingState(). Clearing results would make those
    // storages unreachable (killing their WeakRef anchor) before they're consumed.
    for (const node of plan.nodes) {
      node.result = undefined;
    }
    for (const node of skippedNodes) {
      node.result = undefined;
    }

    // Destroy orphaned intermediates created during this plan execution.
    // At markStep time, all pending tensors have been materialized, so
    // any unreachable storages created during execution are truly orphaned.
    storageTracker.destroyUnreachableSince(storageSnapshot);
  }

  tensorFromArray(
    values: number[] | Float32Array,
    shape: number[],
    device?: DeviceKind,
  ): Tensor {
    const resolvedDevice = this.getDevice(device);
    // For Float32Array, skip the defensive copy — the typed array buffer will be consumed
    // by the GPU backend at the next force boundary. Avoids doubling memory for large models.
    // For number[], copy to prevent aliasing issues with mutable JS arrays.
    const payload = values instanceof Float32Array ? values : values.slice();
    const node = createLazyIRNode(
      "tensorFromArray",
      [],
      shape,
      "f32",
      resolvedDevice,
      { values: payload },
    );
    return this.trackNode(node, shape, resolvedDevice);
  }

  /** Helper: create a data-source op (no inputs, f32 output). */
  private _creationOp(
    op: LazyOpCode,
    shape: number[],
    device?: DeviceKind,
    payload?: unknown,
  ): Tensor {
    const resolvedDevice = this.getDevice(device);
    const node = createLazyIRNode(
      op,
      [],
      shape,
      "f32",
      resolvedDevice,
      payload,
    );
    return this.trackNode(node, shape, resolvedDevice);
  }

  zeros(shape: number[], device?: DeviceKind): Tensor {
    return this._creationOp("zeros", shape, device);
  }

  full(shape: number[], fillValue: number, device?: DeviceKind): Tensor {
    return this._creationOp("full", shape, device, { fillValue });
  }

  arange(end: number, start = 0, step = 1, device?: DeviceKind): Tensor {
    const numElements = Math.max(0, Math.ceil((end - start) / step));
    return this._creationOp("arange", [numElements], device, {
      end,
      start,
      step,
    });
  }

  tril(a: Tensor, k = 0): Tensor {
    if (a.shape.length < 2)
      throw new Error("tril requires at least 2 dimensions");
    return this._unaryOp("tril", a, { k });
  }

  triu(a: Tensor, k = 0): Tensor {
    if (a.shape.length < 2)
      throw new Error("triu requires at least 2 dimensions");
    return this._unaryOp("triu", a, { k });
  }

  rand(shape: number[], device?: DeviceKind): Tensor {
    return this._creationOp("rand", shape, device, {
      seed: this._rngCounter++,
    });
  }

  randn(shape: number[], device?: DeviceKind): Tensor {
    return this._creationOp("randn", shape, device, {
      seed: this._rngCounter++,
    });
  }

  bernoulli(shape: number[], p: number, device?: DeviceKind): Tensor {
    return this._creationOp("bernoulli", shape, device, {
      seed: this._rngCounter++,
      p,
    });
  }

  /** Helper: create a simple unary lazy op node.
   *  Automatically handles dtype safety from OP_DTYPE_RULES:
   *  - f32_required: casts f16 input to f32
   *  - always_f32: forces f32 output dtype
   */
  _unaryOp(op: LazyOpCode, a: OpInput, payload?: unknown): Tensor {
    let input: OpInput = a;
    let outDtype = a.dtype;
    const rule = OP_DTYPE_RULES[op];
    if (rule) {
      if (rule.category === "f32_required" && a.dtype === "f16") {
        input = castRef(a, "f32");
        outDtype = "f32";
      } else if (rule.category === "always_f32") {
        outDtype = "f32";
      }
    }
    const node = createLazyIRNode(
      op,
      [input.lazyRef],
      a.shape.slice(),
      outDtype,
      a.device,
      payload,
    );
    return this.trackNode(node, a.shape.slice(), a.device, outDtype);
  }

  /**
   * Ensure dtype safety for an op based on the centralized Op Dtype Registry.
   * - promote_inputs: promote mismatched dtypes to f32
   * - f32_required: upcast f16 inputs to f32
   *
   * Returns OpInput[] (lightweight refs) rather than Tensor[] to avoid creating
   * lifecycle-managed Tensor objects for internal dtype casts. Callers only need
   * lazyRef/shape/dtype/device, which OpInput provides.
   */
  private ensureDtypeSafety(op: LazyOpCode, inputs: OpInput[]): OpInput[] {
    const rule = OP_DTYPE_RULES[op];
    if (!rule) return inputs;

    if (rule.category === "promote_inputs" && inputs.length >= 2) {
      const [a, b] = inputs;
      if (a.dtype !== b.dtype) {
        const target = promoteDtype(a.dtype, b.dtype);
        return [
          a.dtype === target ? a : castRef(a, target),
          b.dtype === target ? b : castRef(b, target),
          ...inputs.slice(2),
        ];
      }
    }

    if (rule.category === "f32_required") {
      return inputs.map((t) => (t.dtype === "f16" ? castRef(t, "f32") : t));
    }

    return inputs;
  }

  /**
   * Resolve a TensorOrScalar operand to a LazyRef.
   * Numbers become scalar LazyRefs (no graph node, no GPU buffer).
   * The refTensor provides dtype/device context for the scalar.
   */
  private resolveOperand(
    value: OpInput | number,
    ref: OpInput,
  ): { ref: LazyRef; shape: number[] } {
    if (typeof value === "number") {
      return { ref: createScalarRef(value, ref.dtype), shape: [] };
    }
    return { ref: value.lazyRef, shape: value.shape.slice() };
  }

  /**
   * Get the reference input (for dtype/device) from a pair of operands.
   * At least one must be a non-number.
   */
  private getRefInput(a: OpInput | number, b: OpInput | number): OpInput {
    if (typeof a !== "number") return a;
    if (typeof b !== "number") return b;
    throw new Error("At least one operand must be a Tensor");
  }

  /**
   * Helper for binary ops that accept TensorOrScalar.
   * Resolves operands, handles dtype safety, and returns the pieces needed to build a node.
   */
  private resolveBinaryOp(
    op: LazyOpCode,
    a: TensorOrScalar,
    b: TensorOrScalar,
  ): {
    refA: LazyRef;
    refB: LazyRef;
    shape: number[];
    dtype: DType;
    device: DeviceKind;
  } {
    // Apply dtype safety only when both operands are tensors
    let opA: OpInput | number = a;
    let opB: OpInput | number = b;
    if (typeof a !== "number" && typeof b !== "number") {
      this.assertSameDevice(a, b);
      [opA, opB] = this.ensureDtypeSafety(op, [a, b]);
    }
    const ref = this.getRefInput(opA, opB);
    const resA = this.resolveOperand(opA, ref);
    const resB = this.resolveOperand(opB, ref);
    const shape = broadcastShapes(resA.shape, resB.shape);
    // Check if this op always produces f32 (comparisons, etc.)
    const rule = OP_DTYPE_RULES[op];
    const dtype = rule?.category === "always_f32" ? "f32" : ref.dtype;
    return {
      refA: resA.ref,
      refB: resB.ref,
      shape,
      dtype,
      device: ref.device,
    };
  }

  _binaryOp(
    op: LazyOpCode,
    a: TensorOrScalar,
    b: TensorOrScalar,
    payload?: unknown,
  ): Tensor {
    const { refA, refB, shape, dtype, device } = this.resolveBinaryOp(op, a, b);
    const node = createLazyIRNode(
      op,
      [refA, refB],
      shape,
      dtype,
      device,
      payload,
    );
    return this.trackNode(node, shape, device, dtype);
  }

  add(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._binaryOp("add", a, b);
  }
  sub(a: TensorOrScalar, b: TensorOrScalar, options?: SubOptions): Tensor {
    return this._binaryOp("sub", a, b, options);
  }
  div(a: TensorOrScalar, b: TensorOrScalar, options?: DivOptions): Tensor {
    return this._binaryOp("div", a, b, options);
  }
  mul(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._binaryOp("mul", a, b);
  }
  pow(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._binaryOp("pow", a, b);
  }

  /** Create a view op node that shares baseId with the input tensor. */
  private _viewOp(
    op: LazyOpCode,
    a: Tensor,
    shape: number[],
    payload?: unknown,
  ): Tensor {
    const node = createLazyIRNode(
      op,
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      payload,
    );
    return this.createAndTrack(
      a.baseId,
      createPendingRef(node),
      shape,
      a.device,
      a.dtype,
    );
  }

  view(a: Tensor, shape: number[]): Tensor {
    return this.reshape(a, shape);
  }

  reshape(a: Tensor, shape: number[]): Tensor {
    return this._viewOp("reshape", a, shape, { targetShape: shape });
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    const device = this.assertSameDevice(a, b);
    const shape = matmulShape(a.shape, b.shape);
    // Output dtype = higher precision of inputs (f32 if mixed)
    const dtype =
      a.dtype === "f32" || b.dtype === "f32" ? ("f32" as const) : a.dtype;
    const node = createLazyIRNode(
      "matmul",
      [a.lazyRef, b.lazyRef],
      shape,
      dtype,
      device,
    );
    return this.trackNode(node, shape, device, dtype);
  }

  gelu(a: Tensor, options?: GeluOptions): Tensor {
    return this._unaryOp("gelu", a, options);
  }
  clamp(a: Tensor, min: number | null, max: number | null): Tensor {
    return this._unaryOp("clamp", a, { min, max });
  }

  expand(a: Tensor, shape: number[]): Tensor {
    return this._viewOp("expand", a, shape, { targetShape: shape });
  }

  transpose(a: Tensor, options: TransposeOptions): Tensor {
    return this._viewOp(
      "transpose",
      a,
      transposeShape(a.shape, options),
      options,
    );
  }

  permute(a: Tensor, dims: number[]): Tensor {
    // Validate and compute output shape
    const rank = a.shape.length;
    if (dims.length !== rank) {
      throw new Error(
        `permute: dims length ${dims.length} doesn't match tensor rank ${rank}`,
      );
    }
    const seen = new Set<number>();
    for (const d of dims) {
      const nd = normalizeDim(d, rank);
      if (nd < 0 || nd >= rank) {
        throw new Error(
          `permute: dimension ${d} out of range for rank ${rank}`,
        );
      }
      if (seen.has(nd)) {
        throw new Error(`permute: duplicate dimension ${d}`);
      }
      seen.add(nd);
    }
    const normalizedDims = dims.map((d) => normalizeDim(d, rank));
    const shape = normalizedDims.map((d) => a.shape[d]);

    return this._viewOp("permute", a, shape, { dims: normalizedDims });
  }

  contiguous(a: Tensor): Tensor {
    return this._unaryOp("contiguous", a);
  }

  narrow(a: Tensor, dim: number, start: number, length: number): Tensor {
    const rank = a.shape.length;
    if (dim < 0 || dim >= rank) {
      throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
    }
    if (start < 0 || start + length > a.shape[dim]) {
      throw new Error(
        `narrow: range [${start}, ${start + length}) out of bounds for dim size ${a.shape[dim]}`,
      );
    }
    const shape = a.shape.slice();
    shape[dim] = length;
    return this._viewOp("narrow", a, shape, { dim, start, length });
  }

  narrowBackward(
    grad: Tensor,
    dim: number,
    start: number,
    originalLength: number,
  ): Tensor {
    const shape = grad.shape.slice();
    shape[dim] = originalLength;
    const node = createLazyIRNode(
      "narrowBackward",
      [grad.lazyRef],
      shape,
      grad.dtype,
      grad.device,
      { dim, start, originalLength },
    );
    return this.trackNode(node, shape, grad.device, grad.dtype);
  }

  cast(a: Tensor, dtype: DType): Tensor {
    const node = createLazyIRNode(
      "cast",
      [a.lazyRef],
      a.shape.slice(),
      dtype,
      a.device,
      { dtype },
    );
    return this.trackNode(node, a.shape.slice(), a.device, dtype);
  }

  gather(a: Tensor, index: Tensor, options: GatherOptions): Tensor {
    const device = this.assertSameDevice(a, index);
    const shape = index.shape.slice();
    const node = createLazyIRNode(
      "gather",
      [a.lazyRef, index.lazyRef],
      shape,
      a.dtype,
      device,
      options,
    );
    return this.trackNode(node, shape, device, a.dtype);
  }

  scatterAdd(
    a: Tensor,
    index: Tensor,
    src: Tensor,
    options: ScatterAddOptions,
  ): Tensor {
    const device = this.assertSameDevice(a, index, src);
    // ScatterAdd output shape matches input a shape
    const shape = a.shape.slice();
    const dtype = a.dtype;
    const node = createLazyIRNode(
      "scatterAdd",
      [a.lazyRef, index.lazyRef, src.lazyRef],
      shape,
      dtype,
      device,
      options,
    );
    return this.trackNode(node, shape, device, dtype);
  }

  cat(tensors: Tensor[], options?: CatOptions): Tensor {
    if (tensors.length === 0) throw new Error("cat: empty tensor list");
    const dim = options?.dim ?? 0;
    const rank = tensors[0].shape.length;
    const d = dim < 0 ? dim + rank : dim;
    // Compute output shape
    const outShape = tensors[0].shape.slice();
    for (let i = 1; i < tensors.length; i++) {
      outShape[d] += tensors[i].shape[d];
    }
    const device = tensors[0].device;
    const dtype = tensors[0].dtype;
    const node = createLazyIRNode(
      "cat",
      tensors.map((t) => t.lazyRef),
      outShape,
      dtype,
      device,
      { dim: d } satisfies CatOptions,
    );
    return this.trackNode(node, outShape, device, dtype);
  }

  private _reductionOp(
    op: LazyOpCode,
    a: Tensor,
    options?: { dim?: number | number[] | null; keepdim?: boolean },
  ): Tensor {
    const [safe] = this.ensureDtypeSafety(op, [a]);
    const shape = reduceShape(
      safe.shape,
      options?.dim,
      options?.keepdim ?? false,
    );
    const node = createLazyIRNode(
      op,
      [safe.lazyRef],
      shape,
      safe.dtype,
      safe.device,
      options,
    );
    return this.trackNode(node, shape, safe.device, safe.dtype);
  }

  sum(a: Tensor, options?: SumOptions): Tensor {
    return this._reductionOp("sum", a, options);
  }
  max(a: Tensor, options?: MaxOptions): number | Tensor {
    return this._reductionOp("max", a, options);
  }
  min(a: Tensor, options?: MaxOptions): number | Tensor {
    return this._reductionOp("min", a, options);
  }
  mean(a: Tensor, options?: MeanOptions): number | Tensor {
    return this._reductionOp("mean", a, options);
  }

  private _argReduceOp(
    op: LazyOpCode,
    a: Tensor,
    options: ArgReduceOptions,
  ): Tensor {
    const dim = options.dim < 0 ? a.shape.length + options.dim : options.dim;
    const shape = options.keepdim
      ? a.shape.map((s, i) => (i === dim ? 1 : s))
      : a.shape.filter((_, i) => i !== dim);
    const node = createLazyIRNode(op, [a.lazyRef], shape, "f32", a.device, {
      dim,
      keepdim: options.keepdim,
    });
    return this.trackNode(node, shape, a.device);
  }

  argmax(a: Tensor, options: ArgReduceOptions): Tensor {
    return this._argReduceOp("argmax", a, options);
  }
  argmin(a: Tensor, options: ArgReduceOptions): Tensor {
    return this._argReduceOp("argmin", a, options);
  }

  where(condition: Tensor, x: TensorOrScalar, y: TensorOrScalar): Tensor {
    const refT =
      typeof x !== "number" ? x : typeof y !== "number" ? y : condition;
    if (typeof x !== "number" && typeof y !== "number") {
      this.assertSameDevice(condition, x, y);
    }
    const device = condition.device;
    const resX = this.resolveOperand(x, refT);
    const resY = this.resolveOperand(y, refT);
    const shape = broadcastThreeShapes(condition.shape, resX.shape, resY.shape);
    const dtype = refT.dtype;
    const node = createLazyIRNode(
      "where",
      [condition.lazyRef, resX.ref, resY.ref],
      shape,
      dtype,
      device,
    );
    return this.trackNode(node, shape, device, dtype);
  }

  async cpu(a: Tensor): Promise<number[]> {
    await this.force(a);
    const backend = this.getBackend(a.device);
    return backend.ops.read(a.backendTensor);
  }

  async item(a: Tensor): Promise<number> {
    const values = await this.cpu(a);
    if (values.length !== 1) {
      throw new Error("item() requires a single-element tensor");
    }
    return values[0];
  }

  /**
   * Transfer a tensor to a different device (lazy).
   * Creates a lazy transfer node that will be executed when forced.
   */
  transfer(a: Tensor, device: DeviceKind): Tensor {
    if (a.device === device) {
      return a;
    }
    const node = createLazyIRNode(
      "transfer",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      device,
      { sourceDevice: a.device },
    );
    // Transfer creates a new tensor with new baseId on target device
    return this.trackNode(node, a.shape.slice(), device, a.dtype);
  }

  /**
   * Force transfer and return the transferred tensor.
   * Use this when you need immediate transfer (e.g., for cross-device ops).
   */
  async transferNow(a: Tensor, device: DeviceKind): Promise<Tensor> {
    if (a.device === device) {
      return a;
    }
    const transferred = this.transfer(a, device);
    await this.force(transferred);
    return transferred;
  }

  // ============================================================================
  // In-place operations (§4.3-4.4)
  // ============================================================================

  /** Shared logic for in-place scatter ops (copy_, add_). */
  private _scatterInPlace(
    op: "stridedScatterCopy" | "stridedScatterAdd",
    label: string,
    dst: Tensor,
    src: Tensor,
  ): Tensor {
    this.assertSameDevice(dst, src);
    this.assertShapeMatch(label, dst, src);
    const payload: StridedScatterOptions = {
      offset: 0,
      viewShape: dst.shape.slice(),
      viewStrides: contiguousStrides(dst.shape),
    };
    const node = createLazyIRNode(
      op,
      [dst.lazyRef, src.lazyRef],
      dst.shape,
      dst.dtype,
      dst.device,
      payload,
    );
    dst._updateLazyRef(createPendingRef(node));
    return dst;
  }

  copy_(dst: Tensor, src: Tensor): Tensor {
    return this._scatterInPlace("stridedScatterCopy", "copy_", dst, src);
  }
  add_(dst: Tensor, src: Tensor): Tensor {
    return this._scatterInPlace("stridedScatterAdd", "add_", dst, src);
  }

  zero_(dst: Tensor): Tensor {
    return this.copy_(dst, this.zeros(dst.shape, dst.device));
  }
  fill_(dst: Tensor, value: number): Tensor {
    return this.copy_(dst, this.full(dst.shape, value, dst.device));
  }
  mul_(dst: Tensor, value: number): Tensor {
    return this.copy_(dst, this.mul(dst, value));
  }

  async beginStep(): Promise<void> {
    await this.getBackend().beginStep?.();
  }
  endStep(): void {
    this.getBackend().endStep?.();
  }

  /**
   * Wrap an existing StorageHandle into a tracked RuntimeTensor.
   * Reuses the existing storage without creating a new one.
   * Used by the fused Adam kernel to create tensors for side outputs (m, v).
   */
  createFromStorageHandle(
    storage: import("../engine/lazy-types").StorageHandle,
    shape: number[],
    device: DeviceKind,
    dtype: DType = "f32",
  ): Tensor {
    const ref: LazyRef = createMaterializedRef(storage);
    return this.trackRef(ref, shape, device, dtype);
  }

  /** Create a fused kernel op node and track it as a new Tensor. */
  private fusedOp(
    op: LazyOpCode,
    tensors: Tensor[],
    shape: number[],
    device: DeviceKind,
    config?: unknown,
  ): Tensor {
    const node = createLazyIRNode(
      op,
      tensors.map((t) => t.lazyRef),
      shape,
      "f32",
      device,
      config,
    );
    return this.trackRef(createPendingRef(node), shape, device);
  }

  fusedCrossEntropyForward(
    logits: Tensor,
    targets: Tensor,
    config: FusedCrossEntropyConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedCrossEntropyForward",
      [logits, targets],
      [config.batchSize],
      logits.device,
      config,
    );
  }

  fusedCrossEntropyBackward(
    logits: Tensor,
    targets: Tensor,
    gradOutput: Tensor,
    config: FusedCrossEntropyConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedCrossEntropyBackward",
      [logits, targets, gradOutput],
      [config.batchSize, config.vocabSize],
      logits.device,
      config,
    );
  }

  fusedLayerNormForward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedLayerNormForward",
      [x, weight, bias],
      x.shape.slice(),
      x.device,
      config,
    );
  }

  fusedLayerNormBackwardGradX(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedLayerNormBackwardGradX",
      [gradOutput, x, weight],
      x.shape.slice(),
      x.device,
      config,
    );
  }

  fusedLayerNormBackwardGradWeightBias(
    gradOutput: Tensor,
    x: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedLayerNormBackwardGradWeightBias",
      [gradOutput, x],
      [config.featureDim],
      gradOutput.device,
      config,
    );
  }

  fusedRMSNormForward(
    x: Tensor,
    weight: Tensor,
    config: FusedRMSNormConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedRMSNormForward",
      [x, weight],
      x.shape.slice(),
      x.device,
      config,
    );
  }

  fusedRMSNormBackwardGradX(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedRMSNormConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedRMSNormBackwardGradX",
      [gradOutput, x, weight],
      x.shape.slice(),
      x.device,
      config,
    );
  }

  fusedRMSNormBackwardGradWeight(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedRMSNormConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedRMSNormBackwardGradWeight",
      [gradOutput, x, weight],
      [config.featureDim],
      x.device,
      config,
    );
  }

  fusedAttentionForward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedAttentionForward",
      [q, k, v],
      [config.batchSize, config.numHeads, config.seqLen, config.headDim],
      q.device,
      config,
    );
  }

  extractAttentionLogsumexp(
    fwdOutput: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    // Multi-output: logsumexp is output index 1 of fusedAttentionForward
    const parentRef = fwdOutput.lazyRef;
    if (parentRef.kind !== "pending")
      throw new Error("extractAttentionLogsumexp: expected pending ref");
    const ref = createPendingRef(parentRef.node, 1);
    return this.trackRef(
      ref,
      [config.batchSize, config.numHeads, config.seqLen],
      fwdOutput.device,
    );
  }

  fusedAttentionBackward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    logsumexp: Tensor,
    dO: Tensor,
    output: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    return this.fusedOp(
      "fusedAttentionBackward",
      [q, k, v, logsumexp, dO, output],
      [config.batchSize, config.numHeads, config.seqLen, config.headDim],
      q.device,
      config,
    );
  }

  extractAttentionDK(bwdDQ: Tensor, config: FusedAttentionConfig): Tensor {
    // Multi-output: dK is output index 1 of fusedAttentionBackward
    const parentRef = bwdDQ.lazyRef;
    if (parentRef.kind !== "pending")
      throw new Error("extractAttentionDK: expected pending ref");
    const ref = createPendingRef(parentRef.node, 1);
    return this.trackRef(
      ref,
      [config.batchSize, config.numHeads, config.seqLen, config.headDim],
      bwdDQ.device,
    );
  }

  extractAttentionDV(bwdDQ: Tensor, config: FusedAttentionConfig): Tensor {
    // Multi-output: dV is output index 2 of fusedAttentionBackward
    const parentRef = bwdDQ.lazyRef;
    if (parentRef.kind !== "pending")
      throw new Error("extractAttentionDV: expected pending ref");
    const ref = createPendingRef(parentRef.node, 2);
    return this.trackRef(
      ref,
      [config.batchSize, config.numHeads, config.seqLen, config.headDim],
      bwdDQ.device,
    );
  }

  extractLnBwdGradBias(gradWeight: Tensor, featureDim: number): Tensor {
    // Multi-output: gradBias is output index 1 of fusedLayerNormBackwardGradWeightBias
    const parentRef = gradWeight.lazyRef;
    if (parentRef.kind !== "pending")
      throw new Error("extractLnBwdGradBias: expected pending ref");
    const ref = createPendingRef(parentRef.node, 1);
    return this.trackRef(ref, [featureDim], gradWeight.device);
  }

  // ============================================================================
  // Engine methods (merged from Engine class)
  // ============================================================================

  /**
   * Set the autocast context for AMP transforms in compiled regions.
   * This should be called by the frontend when entering an autocast block.
   */
  setAutocastContext(ctx: AutocastContext | null): void {
    this._autocastContext = ctx;
  }

  /**
   * Get the current autocast context.
   */
  getAutocastContext(): AutocastContext | null {
    return this._autocastContext;
  }

  _debug_publishSave(_baseId: EngineBaseId): void {
    // No-op: token algebra removed. Kept for frontend.ts call site.
  }

  _debug_setRngBasis(basis: RngBasis): void {
    this._rngBasis = { ...basis };
    this._rngDrawNonce = 0;
  }

  _debug_getRngDrawNonce(): number {
    return this._rngDrawNonce;
  }

  _debug_startCheckpointRecord(): void {
    if (this._rngCheckpointMode) {
      throw new Error("Checkpoint RNG already active");
    }
    this._rngCheckpointMode = "record";
    this._rngCheckpointDraws = [];
    this._rngCheckpointIndex = 0;
  }

  _debug_finishCheckpointRecord(): RngDrawRecord[] {
    if (this._rngCheckpointMode !== "record") {
      throw new Error("Checkpoint RNG record not active");
    }
    const draws = this._rngCheckpointDraws.slice();
    this._rngCheckpointMode = null;
    this._rngCheckpointDraws = [];
    this._rngCheckpointIndex = 0;
    return draws;
  }

  _debug_startCheckpointReplay(draws: RngDrawRecord[]): void {
    if (this._rngCheckpointMode) {
      throw new Error("Checkpoint RNG already active");
    }
    this._rngCheckpointMode = "replay";
    this._rngCheckpointDraws = draws.slice();
    this._rngCheckpointIndex = 0;
  }

  _debug_finishCheckpointReplay(): void {
    if (this._rngCheckpointMode !== "replay") {
      throw new Error("Checkpoint RNG replay not active");
    }
    this._rngCheckpointMode = null;
    this._rngCheckpointDraws = [];
    this._rngCheckpointIndex = 0;
  }

  _debug_random(opNonce: number, drawNonce?: number): RngDrawResult {
    if (this._rngCheckpointMode === "replay") {
      const record = this._rngCheckpointDraws[this._rngCheckpointIndex];
      if (!record) {
        throw new RngReplayExhaustedError("RNG replay exhausted");
      }
      if (record.opNonce !== opNonce) {
        throw new RngReplayMismatchError("RNG replay opNonce mismatch");
      }
      if (drawNonce !== undefined && drawNonce !== record.drawNonce) {
        throw new RngReplayMismatchError("RNG replay drawNonce mismatch");
      }
      this._rngCheckpointIndex += 1;
      return { drawNonce: record.drawNonce, value: record.value };
    }

    const assignedDrawNonce = drawNonce ?? this._nextRngDrawNonce();
    if (drawNonce !== undefined && drawNonce > this._rngDrawNonce) {
      this._rngDrawNonce = drawNonce;
    }
    const value = computeRngValue(this._rngBasis, opNonce, assignedDrawNonce);

    if (this._rngCheckpointMode === "record") {
      this._rngCheckpointDraws.push({
        opNonce,
        drawNonce: assignedDrawNonce,
        value,
      });
    }

    return { drawNonce: assignedDrawNonce, value };
  }

  _debug_backward<T>(fn: () => T): T {
    if (this._backwardActive) {
      throw new NonReentrantBackwardError("Backward is already running");
    }
    this._backwardActive = true;
    try {
      return fn();
    } finally {
      this._backwardActive = false;
    }
  }

  _debug_saveForBackward(baseId: EngineBaseId): SavedTensorRecord {
    const state = this._getOrCreateBaseState(baseId);
    const id = this._nextSavedTensorId++;
    const record = {
      id,
      baseId,
      baseCommitVersionAtSave: state.baseCommitVersion,
    };

    // Track for visibility
    this._savedTensors.set(id, {
      id,
      baseId,
      commitVersionAtSave: state.baseCommitVersion,
      savedAt: Date.now(),
    });

    return record;
  }

  _debug_useSavedTensor(record: SavedTensorRecord): void {
    const state = this._getOrCreateBaseState(record.baseId);
    if (state.baseCommitVersion !== record.baseCommitVersionAtSave) {
      throw new SavedTensorModifiedError(
        `Saved tensor modified for baseId ${record.baseId}`,
      );
    }
  }

  /**
   * Release a saved tensor from tracking (called after backward completes).
   */
  _debug_releaseSavedTensor(id: number): void {
    this._savedTensors.delete(id);
  }

  /**
   * Clear all saved tensor tracking (for cleanup after backward).
   */
  _debug_clearSavedTensors(): void {
    this._savedTensors.clear();
  }

  compile<Args extends unknown[], R>(
    fn: (...args: Args) => R,
  ): (...args: Args) => R {
    return (...args: Args) => {
      let result: R;
      result = fn(...args);

      if (
        result &&
        typeof (result as unknown as Promise<unknown>).then === "function"
      ) {
        throw new AsyncInCompileError("Async work is not allowed in compile");
      }

      return result;
    };
  }

  createTensor(baseId?: EngineBaseId): EngineTensor {
    const resolvedBaseId = baseId ?? this._nextBaseId++;
    if (resolvedBaseId >= this._nextBaseId) {
      this._nextBaseId = resolvedBaseId + 1;
    }
    const origin: TensorOrigin =
      this._tidyScopes.length > 0
        ? {
            kind: "tidy",
            scopeId: this._tidyScopes[this._tidyScopes.length - 1].id,
          }
        : { kind: "global" };
    const tensor = new EngineTensor(
      this._nextTensorId++,
      resolvedBaseId,
      origin,
    );
    this._registerTensor(tensor);
    return tensor;
  }

  forceRead(_baseId: EngineBaseId): void {
    // No-op: token algebra and loc bindings removed.
    // Callers use this as a barrier; actual read is done by the runtime.
  }

  tidy<T>(fn: () => T): T {
    const scope: TidyScope = { id: this._nextScopeId++, tensors: new Set() };
    this._tidyScopes.push(scope);
    let result!: T;
    let succeeded = false;

    try {
      result = fn();
      succeeded = true;
    } finally {
      const escaped = succeeded ? collectTensorHandles(result) : [];
      for (const tensor of escaped) {
        tensor.escapes = true;
      }

      // Convert to array first since dispose() modifies scope.tensors
      const tensorsToProcess = Array.from(scope.tensors);
      for (const tensor of tensorsToProcess) {
        if (!tensor.escapes) {
          this.dispose(tensor);
        }
      }
      this._tidyScopes.pop();
    }

    return result;
  }

  keep(tensor: EngineTensor): void {
    if (tensor.disposed) {
      return;
    }
    tensor.escapes = true;
  }

  dispose(tensor: EngineTensor): void {
    if (tensor.disposed) {
      return;
    }
    tensor.disposed = true;
    // Call disposal callback to free backend resources (GPU buffers, etc.)
    if (tensor.onDispose) {
      tensor.onDispose();
    }
    const count = this._basePinCount.get(tensor.baseId) ?? 0;
    if (count <= 1) {
      this._basePinCount.delete(tensor.baseId);
    } else {
      this._basePinCount.set(tensor.baseId, count - 1);
    }
    for (const scope of this._tidyScopes) {
      scope.tensors.delete(tensor);
    }
  }

  async markStep(): Promise<void> {
    this._debug_runEntryPoint(() => {
      // No-op: token algebra and loc bindings removed.
      // Runtime handles plan execution directly.
    });
  }

  private _acquireExecLock(): void {
    if (this._execLock.held) throw new EngineBusyError("Engine is busy");
    this._execLock.held = true;
    this._execLock.ownerId = this._nextOwnerId++;
    this._execLock.depth = 1;
    this._ensureNotPoisoned();
  }

  private _releaseExecLock(): void {
    this._execLock.held = false;
    this._execLock.ownerId = 0;
    this._execLock.depth = 0;
  }

  _debug_runEntryPoint<T>(fn: () => T): T {
    this._acquireExecLock();
    try {
      return fn();
    } finally {
      this._releaseExecLock();
    }
  }

  async runEntryPoint<T>(fn: () => Promise<T>): Promise<T> {
    this._acquireExecLock();
    try {
      return await fn();
    } finally {
      this._releaseExecLock();
    }
  }

  /**
   * Run an async function with an async scope context that tracks tensors
   * across await boundaries. Tensors created during the async operation
   * (but not within a synchronous tidy scope) are tracked and disposed
   * when the scope exits, unless explicitly kept.
   */
  async runWithAsyncScope<T>(fn: () => Promise<T>): Promise<T> {
    const scope: AsyncScope = {
      id: this._nextScopeId++,
      tensors: new Set(),
    };

    const previous = this._asyncScopeContext;
    this._asyncScopeContext = scope;

    try {
      return await fn();
    } finally {
      this._asyncScopeContext = previous;

      // Dispose all tensors in this scope that weren't kept
      for (const tensor of scope.tensors) {
        if (!tensor.escapes && !tensor.disposed) {
          this.dispose(tensor);
        }
      }
    }
  }

  _debug_poison(): void {
    this._poisoned = true;
  }

  _debug_getBasePinCount(baseId: EngineBaseId): number {
    return this._basePinCount.get(baseId) ?? 0;
  }

  /**
   * Get the next unique mutation ID for in-place operations.
   * Per spec §4.3, each in-place mutation needs a unique mutId.
   */
  nextMutId(): number {
    return this._nextMutIdValue++;
  }

  _debug_baseCommit(baseId: EngineBaseId, mutId: number): void {
    const state = this._getOrCreateBaseState(baseId);
    if (state.committed.has(mutId)) {
      throw new Error(`base_commit already recorded for mutId ${mutId}`);
    }
    state.committed.add(mutId);
    state.baseCommitVersion += 1;
  }

  private _ensureNotPoisoned(): void {
    if (this._poisoned) {
      throw new PoisonedEngineError("Engine is poisoned");
    }
  }

  private _nextRngDrawNonce(): number {
    this._rngDrawNonce += 1;
    return this._rngDrawNonce;
  }

  private _registerTensor(tensor: EngineTensor): void {
    const count = this._basePinCount.get(tensor.baseId) ?? 0;
    this._basePinCount.set(tensor.baseId, count + 1);

    // Register in all active tidy scopes
    for (const scope of this._tidyScopes) {
      scope.tensors.add(tensor);
    }

    // If no tidy scope is active but async scope is, register there
    if (this._tidyScopes.length === 0 && this._asyncScopeContext !== null) {
      this._asyncScopeContext.tensors.add(tensor);
    }
  }

  private _getOrCreateBaseState(baseId: EngineBaseId): BaseState {
    const existing = this._baseState.get(baseId);
    if (existing) {
      return existing;
    }
    const initial: BaseState = {
      baseCommitVersion: 0,
      committed: new Set<number>(),
    };
    this._baseState.set(baseId, initial);
    return initial;
  }

  // ============================================================================
  // Engine Visibility Methods (Phase 1)
  // ============================================================================

  /**
   * Set the memory stats provider for external memory tracking.
   * This should be called by the runtime during initialization.
   */
  setMemoryStatsProvider(provider: MemoryStatsProvider): void {
    this._memoryStatsProvider = provider;
  }

  /**
   * Get comprehensive memory statistics.
   */
  _debug_getMemoryStats(): EngineMemoryStats {
    const p = this._memoryStatsProvider;
    const gpu = p?.getGPUStats?.() ?? {
      currentBytes: 0,
      peakBytes: 0,
      limitBytes: 0,
    };
    const pool = p?.getBufferPoolStats?.() ?? {
      pooledBuffers: 0,
      inUseBuffers: 0,
      pendingFenceBuffers: 0,
    };
    const plan = p?.getPlanStats?.() ?? { activePlans: 0, completedPlans: 0 };
    let totalPinCount = 0;
    for (const count of this._basePinCount.values()) totalPinCount += count;

    return {
      gpuCurrentBytes: gpu.currentBytes,
      gpuPeakBytes: gpu.peakBytes,
      gpuLimitBytes: gpu.limitBytes,
      pooledBuffers: pool.pooledBuffers,
      inUseBuffers: pool.inUseBuffers,
      pendingFenceBuffers: pool.pendingFenceBuffers,
      activeBases: this._basePinCount.size,
      totalPinCount,
      savedTensorCount: this._savedTensors.size,
      pendingTensorCount: p?.getPendingTensorCount?.() ?? 0,
      activePlans: plan.activePlans,
      completedPlans: plan.completedPlans,
    };
  }

  /**
   * Get information about all currently saved tensors.
   */
  _debug_getSavedTensorsInfo(): SavedTensorInfo[] {
    return Array.from(this._savedTensors.values());
  }

  /**
   * Get information about all base states.
   */
  _debug_getBaseStatesInfo(): BaseStateInfo[] {
    const infos: BaseStateInfo[] = [];

    for (const [baseId, count] of this._basePinCount) {
      infos.push({
        baseId,
        pinCount: count,
        binding: "ssa",
        locId: null,
        hasValue: false,
        commitVersion: this._baseState.get(baseId)?.baseCommitVersion ?? 0,
      });
    }

    return infos;
  }

  /**
   * Take a memory snapshot with a label.
   */
  _debug_takeMemorySnapshot(label: string): void {
    this._memorySnapshots.push({
      label,
      timestamp: Date.now(),
      stats: this._debug_getMemoryStats(),
    });
  }

  /**
   * Get all memory snapshots.
   */
  _debug_getMemorySnapshots(): MemorySnapshot[] {
    return this._memorySnapshots.slice();
  }

  /**
   * Clear all memory snapshots.
   */
  _debug_clearMemorySnapshots(): void {
    this._memorySnapshots.length = 0;
  }
}

// ============================================================================
// Table-driven method generation — typed declarations + prototype installation
// ============================================================================

// Simple unary ops: all delegate to _unaryOp with no payload.
const SIMPLE_UNARY_OPS = [
  "sqrt",
  "relu",
  "exp",
  "log",
  "neg",
  "abs",
  "tanh",
  "sigmoid",
  "silu",
  "sin",
  "cos",
  "rsqrt",
  "floor",
  "ceil",
  "round",
  "sign",
  "isfinite",
  "contiguous",
] as const;

interface RuntimeEngine {
  sqrt(a: Tensor): Tensor;
  relu(a: Tensor): Tensor;
  exp(a: Tensor): Tensor;
  log(a: Tensor): Tensor;
  neg(a: Tensor): Tensor;
  abs(a: Tensor): Tensor;
  tanh(a: Tensor): Tensor;
  sigmoid(a: Tensor): Tensor;
  silu(a: Tensor): Tensor;
  sin(a: Tensor): Tensor;
  cos(a: Tensor): Tensor;
  rsqrt(a: Tensor): Tensor;
  floor(a: Tensor): Tensor;
  ceil(a: Tensor): Tensor;
  round(a: Tensor): Tensor;
  sign(a: Tensor): Tensor;
  isfinite(a: Tensor): Tensor;
  contiguous(a: Tensor): Tensor;
}

for (const op of SIMPLE_UNARY_OPS) {
  (RuntimeEngine.prototype as any)[op] = function (
    this: RuntimeEngine,
    a: Tensor,
  ) {
    return this._unaryOp(op, a);
  };
}

// Comparison ops: all delegate to _binaryOp.
const COMPARISON_OPS = ["gt", "lt", "ge", "le", "eq", "ne"] as const;

interface RuntimeEngine {
  gt(a: TensorOrScalar, b: TensorOrScalar): Tensor;
  lt(a: TensorOrScalar, b: TensorOrScalar): Tensor;
  ge(a: TensorOrScalar, b: TensorOrScalar): Tensor;
  le(a: TensorOrScalar, b: TensorOrScalar): Tensor;
  eq(a: TensorOrScalar, b: TensorOrScalar): Tensor;
  ne(a: TensorOrScalar, b: TensorOrScalar): Tensor;
}

for (const op of COMPARISON_OPS) {
  (RuntimeEngine.prototype as any)[op] = function (
    this: RuntimeEngine,
    a: TensorOrScalar,
    b: TensorOrScalar,
  ) {
    return this._binaryOp(op, a, b);
  };
}
