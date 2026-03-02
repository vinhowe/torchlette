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
import { OP_DTYPE_RULES, promoteDtype } from "../engine/dtype-rules";
import { computePlanFingerprint } from "../engine/fusion-detect";
import {
  buildMergedPlan,
  createLazyIRNode,
  createMaterializedRef,
  createPendingRef,
  createScalarRef,
  executeLoweredPlan,
  executePlan,
  executePlanOptimized,
  executePlanSegmented,
  getFusionAnalysisTemplate,
  type LazyIRNode,
  type LazyOpCode,
  type LazyRef,
  type OptimizedExecutionStats,
  storageTracker,
} from "../engine/lazy";
import { isDataSourceOp } from "../engine/lowered-plan";
import {
  extractAttentionDKOp,
  extractAttentionDVOp,
  extractAttentionLogsumexpOp,
  extractLnBwdGradBiasOp,
  fusedAttentionBackwardOp,
  fusedAttentionForwardOp,
  fusedCrossEntropyBackwardOp,
  fusedCrossEntropyForwardOp,
  fusedLayerNormBackwardGradWeightBiasOp,
  fusedLayerNormBackwardGradXOp,
  fusedLayerNormForwardOp,
  fusedRMSNormBackwardGradWeightOp,
  fusedRMSNormBackwardGradXOp,
  fusedRMSNormForwardOp,
} from "./engine-fused";
import { type BaseId, createBaseId, Tensor } from "./tensor";

// ── Engine types (merged from engine-types.ts) ─────────────────────────────
/** A tensor or a numeric scalar (will be inlined as a constant in fused kernels). */
export type TensorOrScalar = Tensor | number;

/** Lightweight reference to a lazy computation with shape/dtype metadata. */
export type OpInput = Pick<Tensor, "lazyRef" | "shape" | "dtype" | "device">;

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
export class IntermediateTrackingMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }
}

// ── Shape helpers (merged from shape-helpers.ts) ────────────────────────────
export { broadcastShapes };

export function matmulShape(a: number[], b: number[]): number[] {
  if (a.length < 1 || b.length < 1)
    throw new Error("matmul requires at least 1D tensors");
  if (a.length === 1 && b.length === 1) return [];
  if (a.length === 1) return [...b.slice(0, -2), b[b.length - 1]];
  if (b.length === 1) return a.slice(0, -1);
  const m = a[a.length - 2];
  const n = b[b.length - 1];
  const batch = broadcastShapes(a.slice(0, -2), b.slice(0, -2));
  return [...batch, m, n];
}

export function transposeShape(
  shape: number[],
  options: TransposeOptions,
): number[] {
  const result = shape.slice();
  const temp = result[options.dim0];
  result[options.dim0] = result[options.dim1];
  result[options.dim1] = temp;
  return result;
}

export function reduceShape(
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
  private planFingerprintCache = new Map<string, number>();

  /** Stack of active dispatch modes (notified on tensor creation/escape). */
  private dispatchModes: DispatchMode[] = [];

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
        return this._webgpuFlushBufferPool;
      } catch {
        return () => {};
      }
    }
    return () => {};
  }

  // --- Lowered Plan Execution ---

  /**
   * Try to execute a plan using a cached lowered execution plan.
   * On cache hit with a lowered plan, executes the plan and materializes all
   * tensors, returning true. On miss, returns false for fallback to normal execution.
   */
  private async tryLoweredPlanExecution(
    plan: { nodes: LazyIRNode[] },
    tensors: Tensor[],
    device: DeviceKind,
  ): Promise<boolean> {
    // Quick structural key for fingerprint cache lookup
    const structKey = `${plan.nodes.length}:${plan.nodes[plan.nodes.length - 1].op}`;
    let fingerprint = this.planFingerprintCache.get(structKey);

    if (fingerprint === undefined) {
      // First time — compute full O(N) fingerprint
      let externalNodeIds: Set<number> | undefined;
      try {
        const { getPendingNodeIds } = await import("./tensor");
        const pending = getPendingNodeIds();
        if (pending.size > 0) {
          externalNodeIds = pending;
        }
      } catch {
        // tensor module not available
      }
      fingerprint = computePlanFingerprint(plan.nodes, externalNodeIds);
    }

    const template = getFusionAnalysisTemplate(fingerprint);
    if (!template?.loweredPlan) {
      return false;
    }

    // Validate plan node count
    if (plan.nodes.length !== template.finalPerm.length) {
      return false;
    }

    // Reconstruct reordered plan nodes from template
    const planNodes = template.finalPerm.map((i) => plan.nodes[i]);

    const backend = this.getBackend(device);

    // Get or create buffer arena for this plan (persists across steps).
    // Arena stabilizes buffer identities for bind group cache hits.
    const useArena = process.env.TORCHLETTE_USE_ARENA !== "0";
    if (useArena && !template.bufferArena) {
      template.bufferArena = { resolve: [], alloc: [] };
    }

    // Save dispatch sequence counters before attempting lowered plan execution.
    // If the lowered plan fails, we restore them so the fallback executePlanOptimized
    // starts at the same dispatch position — preserving bind group cache alignment.
    let savedCounters:
      | { dispatch: number; params: number; output: number }
      | undefined;
    if (device === "webgpu") {
      try {
        const { getDispatchSequenceCounters } = await import(
          "../backend/webgpu/index"
        );
        savedCounters = getDispatchSequenceCounters();
      } catch {
        /* non-WebGPU runtime */
      }
    }

    try {
      const { stats: loweredStats } = await executeLoweredPlan(
        plan as { nodes: LazyIRNode[] },
        planNodes,
        template.loweredPlan,
        backend,
        {
          enableEarlyRelease: this.earlyReleaseEnabled,
          enableVectorization: this.vectorizationEnabled,
          bufferArena: useArena ? template.bufferArena : undefined,
          enableReplay: true,
        },
      );

      // Update fusion stats for consistency with normal execution path
      this.lastFusionStats = loweredStats;
      this.accumulateFusionStats(loweredStats);

      // Materialize ALL tensors that were pending on executed nodes
      const { materializePendingTensors: materialize } = await import(
        "./tensor"
      );
      for (const node of plan.nodes) {
        if (node.result) {
          materialize(node.id, node.result);
        }
      }

      // Materialize the primary tensors
      for (const tensor of tensors) {
        if (!tensor.isMaterialized() && !tensor.disposed) {
          const lazyRef = tensor.lazyRef;
          if (lazyRef.kind === "pending" && lazyRef.node.result) {
            tensor._materialize(lazyRef.node.result);
          }
        }
      }

      // Cache fingerprint for this structural key on success
      this.planFingerprintCache.set(structKey, fingerprint);

      return true;
    } catch {
      // Lowered plan execution failed — clean up and fall through.
      // Disable lowered plan for this fingerprint so we don't retry every step.
      // The template's bufferArena is preserved so executePlanOptimized can use it.
      template.loweredPlan = undefined;

      // Restore dispatch sequence counters so the fallback executePlanOptimized
      // starts at the same position, keeping bind group cache indices aligned.
      if (savedCounters && device === "webgpu") {
        try {
          const { setDispatchSequenceCounters } = await import(
            "../backend/webgpu/index"
          );
          setDispatchSequenceCounters(
            savedCounters.dispatch,
            savedCounters.params,
            savedCounters.output,
          );
        } catch {
          /* non-WebGPU runtime */
        }
      }

      for (const node of plan.nodes) {
        node.result = undefined;
      }
      return false;
    }
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
    // Collect all pending nodes
    const pendingRoots: LazyIRNode[] = [];

    for (const tensor of tensors) {
      if (tensor.isMaterialized() || tensor.disposed) {
        continue;
      }
      const lazyRef = tensor.lazyRef;
      if (lazyRef.kind === "pending") {
        pendingRoots.push(lazyRef.node);
      }
    }

    if (pendingRoots.length === 0) {
      return;
    }

    // Build ONE merged plan from all pending roots
    const plan = buildMergedPlan(pendingRoots);

    if (plan.nodes.length === 0) {
      return;
    }

    // Determine device from first pending tensor
    let device: DeviceKind = "cpu";
    for (const tensor of tensors) {
      if (!tensor.isMaterialized()) {
        device = tensor.device;
        break;
      }
    }

    // Data-source-only plans skip the template/arena/lowered-plan path.
    const allDataSource = plan.nodes.every((n) => isDataSourceOp(n.op));

    // Lowered plan fast-path for forceAllMerged
    if (!allDataSource && device === "webgpu" && this.fusionEnabled) {
      const executed = await this.tryLoweredPlanExecution(
        plan,
        tensors,
        device,
      );
      if (executed) return;
    }

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
      await executePlan(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
    }

    // Materialize ALL tensors that were pending on executed nodes.
    // This ensures all user-held tensors get their storages marked as externally reachable.
    const { materializePendingTensors } = await import("./tensor");
    for (const node of plan.nodes) {
      if (node.result) {
        materializePendingTensors(node.id, node.result);
      }
    }

    // Update tensor refs to point to results for any tensors not yet materialized
    // Use isMaterialized() to check (not lazyRef.kind) because materializePendingTensors
    // above may have already updated some tensors.
    // IMPORTANT: Skip disposed tensors - materializing a disposed tensor marks its storage
    // reachable, but since the tensor is disposed, no one will ever mark it unreachable,
    // causing a permanent storage leak.
    for (const tensor of tensors) {
      if (!tensor.isMaterialized() && !tensor.disposed) {
        const lazyRef = tensor.lazyRef;
        if (lazyRef.kind === "pending" && lazyRef.node.result) {
          tensor._materialize(lazyRef.node.result);
        }
      }
    }
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

    // Collect pending roots
    const pendingRoots: LazyIRNode[] = [];
    for (const tensor of pendingTensors) {
      if (tensor.isMaterialized() || tensor.disposed) continue;
      const lazyRef = tensor.lazyRef;
      if (lazyRef.kind === "pending") {
        pendingRoots.push(lazyRef.node);
      }
    }
    if (pendingRoots.length === 0) return;

    // Build merged plan, skipping already-executed nodes
    const plan = buildMergedPlan(pendingRoots, /* skipExecuted */ true);
    if (plan.nodes.length === 0) return;

    // Destroy storages from prior steps (e.g., old Adam m/v states
    // disposed by _resolvePendingState). Frees GPU buffers into pool
    // before the new plan allocates output buffers.
    storageTracker.destroyUnreachable();

    let device: DeviceKind = "cpu";
    for (const tensor of pendingTensors) {
      if (!tensor.isMaterialized()) {
        device = tensor.device;
        break;
      }
    }

    // Flush recycled GPU buffers from pendingRelease to the main pool.
    // Must happen BEFORE the shared encoder opens (executePlan opens one),
    // so that acquire() can find them in the main pool during plan execution
    // (acquire() skips pendingRelease when sharedEncoder is active).
    const flushBufferPool = this.getBufferPoolFlushFn(device);
    flushBufferPool();

    // Snapshot storage ID before execution to scope cleanup of orphaned intermediates
    const storageSnapshot = storageTracker.getNextStorageId();

    // Lowered plan fast-path for forceAllPending
    if (device === "webgpu" && this.fusionEnabled) {
      const executed = await this.tryLoweredPlanExecution(
        plan,
        pendingTensors,
        device,
      );
      if (executed) {
        // Handle skipped nodes (already executed from prior force() calls)
        for (const tensor of pendingTensors) {
          if (!tensor.isMaterialized() && !tensor.disposed) {
            const lazyRef = tensor.lazyRef;
            if (lazyRef.kind === "pending" && lazyRef.node.result) {
              tensor._materialize(lazyRef.node.result);
            }
          }
        }
        storageTracker.destroyUnreachableSince(storageSnapshot);
        return;
      }
    }

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
      await executePlan(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
    }

    // Materialize all tensors pending on executed nodes
    for (const node of plan.nodes) {
      if (node.result) {
        materialize(node.id, node.result);
      }
    }
    // Materialize any remaining tensors whose nodes were already executed
    // (skipped by buildMergedPlan because they already had results).
    // Also collect these skipped nodes so we can clear their results below.
    const skippedNodes: LazyIRNode[] = [];
    for (const tensor of pendingTensors) {
      if (!tensor.isMaterialized() && !tensor.disposed) {
        const lazyRef = tensor.lazyRef;
        if (lazyRef.kind === "pending" && lazyRef.node.result) {
          tensor._materialize(lazyRef.node.result);
          skippedNodes.push(lazyRef.node);
        }
      }
    }

    // Drop node.result references to allow GC of unclaimed intermediate storages.
    // Without this, lazy nodes keep StorageHandles alive via node.result references,
    // preventing storageTracker.destroyUnreachable() from freeing their buffers.
    // Must clear BOTH plan nodes AND skipped nodes (which were already executed
    // by a prior force() call but not in plan.nodes).
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
    const lazyRef: LazyRef = createPendingRef(node);
    return this.createAndTrack(createBaseId(), lazyRef, shape, resolvedDevice);
  }

  /**
   * Create a zero-filled tensor efficiently without allocating a large JS array.
   * Uses the "zeros" lazy op so the backend can create a zero-initialized buffer directly.
   */
  zeros(shape: number[], device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const node = createLazyIRNode("zeros", [], shape, "f32", resolvedDevice);
    const lazyRef: LazyRef = createPendingRef(node);
    return this.createAndTrack(createBaseId(), lazyRef, shape, resolvedDevice);
  }

  /**
   * Create a tensor filled with a constant value.
   * Uses the "full" lazy op so the backend can create the buffer directly.
   */
  full(shape: number[], fillValue: number, device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const node = createLazyIRNode("full", [], shape, "f32", resolvedDevice, {
      fillValue,
    });
    const lazyRef: LazyRef = createPendingRef(node);
    return this.createAndTrack(createBaseId(), lazyRef, shape, resolvedDevice);
  }

  /**
   * Create a 1-D tensor of evenly spaced values.
   * Uses the "arange" lazy op so the backend can compute values on device.
   */
  arange(end: number, start = 0, step = 1, device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const numElements = Math.max(0, Math.ceil((end - start) / step));
    const shape = [numElements];
    const node = createLazyIRNode("arange", [], shape, "f32", resolvedDevice, {
      end,
      start,
      step,
    });
    const lazyRef: LazyRef = createPendingRef(node);
    return this.createAndTrack(createBaseId(), lazyRef, shape, resolvedDevice);
  }

  /**
   * Return the lower-triangular part of a matrix (or batch of matrices).
   * Elements above the k-th diagonal are zeroed.
   */
  tril(a: Tensor, k = 0): Tensor {
    if (a.shape.length < 2)
      throw new Error("tril requires at least 2 dimensions");
    const node = createLazyIRNode(
      "tril",
      [a.lazyRef],
      a.shape,
      a.dtype,
      a.device,
      { k },
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape,
      a.device,
      a.dtype,
    );
  }

  /**
   * Return the upper-triangular part of a matrix (or batch of matrices).
   * Elements below the k-th diagonal are zeroed.
   */
  triu(a: Tensor, k = 0): Tensor {
    if (a.shape.length < 2)
      throw new Error("triu requires at least 2 dimensions");
    const node = createLazyIRNode(
      "triu",
      [a.lazyRef],
      a.shape,
      a.dtype,
      a.device,
      { k },
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape,
      a.device,
      a.dtype,
    );
  }

  rand(shape: number[], device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const seed = this._rngCounter++;
    const node = createLazyIRNode("rand", [], shape, "f32", resolvedDevice, {
      seed,
    });
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      resolvedDevice,
    );
  }

  randn(shape: number[], device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const seed = this._rngCounter++;
    const node = createLazyIRNode("randn", [], shape, "f32", resolvedDevice, {
      seed,
    });
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      resolvedDevice,
    );
  }

  bernoulli(shape: number[], p: number, device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const seed = this._rngCounter++;
    const node = createLazyIRNode(
      "bernoulli",
      [],
      shape,
      "f32",
      resolvedDevice,
      { seed, p },
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      resolvedDevice,
    );
  }

  /** Helper: create a simple binary lazy op node (comparison ops output f32). */
  private _comparisonOp(
    op: LazyOpCode,
    a: TensorOrScalar,
    b: TensorOrScalar,
  ): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp(op, a, b);
    const node = createLazyIRNode(op, [refA, refB], shape, "f32", device);
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      device,
    );
  }

  /** Helper: create a simple unary lazy op node. */
  private _unaryOp(op: LazyOpCode, a: OpInput, payload?: unknown): Tensor {
    const node = createLazyIRNode(
      op,
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
      payload,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
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
    return {
      refA: resA.ref,
      refB: resB.ref,
      shape,
      dtype: ref.dtype,
      device: ref.device,
    };
  }

  private _binaryOp(
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      device,
      dtype,
    );
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

  view(a: Tensor, shape: number[]): Tensor {
    return this.reshape(a, shape);
  }

  reshape(a: Tensor, shape: number[]): Tensor {
    const node = createLazyIRNode(
      "reshape",
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      { targetShape: shape },
    );
    return this.createAndTrack(
      a.baseId,
      createPendingRef(node),
      shape,
      a.device,
      a.dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      device,
      dtype,
    );
  }

  sqrt(a: Tensor): Tensor {
    return this._unaryOp("sqrt", a);
  }
  relu(a: Tensor): Tensor {
    return this._unaryOp("relu", a);
  }

  exp(a: Tensor): Tensor {
    const [op] = this.ensureDtypeSafety("exp", [a]);
    return this._unaryOp("exp", op);
  }

  log(a: Tensor): Tensor {
    const [op] = this.ensureDtypeSafety("log", [a]);
    return this._unaryOp("log", op);
  }

  neg(a: Tensor): Tensor {
    return this._unaryOp("neg", a);
  }
  abs(a: Tensor): Tensor {
    return this._unaryOp("abs", a);
  }
  tanh(a: Tensor): Tensor {
    return this._unaryOp("tanh", a);
  }
  sigmoid(a: Tensor): Tensor {
    return this._unaryOp("sigmoid", a);
  }

  gelu(a: Tensor, options?: GeluOptions): Tensor {
    return this._unaryOp("gelu", a, options);
  }
  silu(a: Tensor): Tensor {
    return this._unaryOp("silu", a);
  }

  sin(a: Tensor): Tensor {
    return this._unaryOp("sin", a);
  }
  cos(a: Tensor): Tensor {
    return this._unaryOp("cos", a);
  }
  rsqrt(a: Tensor): Tensor {
    return this._unaryOp("rsqrt", a);
  }
  floor(a: Tensor): Tensor {
    return this._unaryOp("floor", a);
  }
  ceil(a: Tensor): Tensor {
    return this._unaryOp("ceil", a);
  }
  round(a: Tensor): Tensor {
    return this._unaryOp("round", a);
  }
  sign(a: Tensor): Tensor {
    return this._unaryOp("sign", a);
  }

  clamp(a: Tensor, min: number | null, max: number | null): Tensor {
    return this._unaryOp("clamp", a, { min, max });
  }

  /** Returns 1.0 where finite, 0.0 where NaN or Inf. */
  isfinite(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "isfinite",
      [a.lazyRef],
      a.shape.slice(),
      "f32",
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
    );
  }

  expand(a: Tensor, shape: number[]): Tensor {
    const node = createLazyIRNode(
      "expand",
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      { targetShape: shape },
    );
    return this.createAndTrack(
      a.baseId,
      createPendingRef(node),
      shape,
      a.device,
      a.dtype,
    );
  }

  transpose(a: Tensor, options: TransposeOptions): Tensor {
    const shape = transposeShape(a.shape, options);
    const node = createLazyIRNode(
      "transpose",
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      options,
    );
    return this.createAndTrack(
      a.baseId,
      createPendingRef(node),
      shape,
      a.device,
      a.dtype,
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

    const node = createLazyIRNode(
      "permute",
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      { dims: normalizedDims },
    );
    // Permute shares baseId with input (it's a view)
    return this.createAndTrack(
      a.baseId,
      createPendingRef(node),
      shape,
      a.device,
      a.dtype,
    );
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
    const node = createLazyIRNode(
      "narrow",
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      { dim, start, length },
    );
    // narrow is a view, shares baseId with input
    return this.createAndTrack(
      a.baseId,
      createPendingRef(node),
      shape,
      a.device,
      a.dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      grad.device,
      grad.dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      device,
      a.dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      device,
      dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      outShape,
      device,
      dtype,
    );
  }

  private _reductionOp(
    op: LazyOpCode,
    a: Tensor,
    options?: { dim?: number | number[]; keepdim?: boolean },
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      safe.device,
      safe.dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      a.device,
    );
  }

  argmax(a: Tensor, options: ArgReduceOptions): Tensor {
    return this._argReduceOp("argmax", a, options);
  }
  argmin(a: Tensor, options: ArgReduceOptions): Tensor {
    return this._argReduceOp("argmin", a, options);
  }

  gt(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._comparisonOp("gt", a, b);
  }
  lt(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._comparisonOp("lt", a, b);
  }
  ge(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._comparisonOp("ge", a, b);
  }
  le(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._comparisonOp("le", a, b);
  }
  eq(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._comparisonOp("eq", a, b);
  }
  ne(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    return this._comparisonOp("ne", a, b);
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      shape,
      device,
      dtype,
    );
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
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      device,
      a.dtype,
    );
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
    storage: import("../engine/lazy").StorageHandle,
    shape: number[],
    device: DeviceKind,
    dtype: DType = "f32",
  ): Tensor {
    const ref: LazyRef = createMaterializedRef(storage);
    return this.createAndTrack(createBaseId(), ref, shape, device, dtype);
  }

  fusedCrossEntropyForward(
    logits: Tensor,
    targets: Tensor,
    config: FusedCrossEntropyConfig,
  ): Tensor {
    const { ref, shape } = fusedCrossEntropyForwardOp(
      logits.lazyRef,
      targets.lazyRef,
      logits.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, logits.device);
  }

  fusedCrossEntropyBackward(
    logits: Tensor,
    targets: Tensor,
    gradOutput: Tensor,
    config: FusedCrossEntropyConfig,
  ): Tensor {
    const { ref, shape } = fusedCrossEntropyBackwardOp(
      logits.lazyRef,
      targets.lazyRef,
      gradOutput.lazyRef,
      logits.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, logits.device);
  }

  fusedLayerNormForward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    const { ref, shape } = fusedLayerNormForwardOp(
      x.lazyRef,
      weight.lazyRef,
      bias.lazyRef,
      x.shape,
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, x.device);
  }

  fusedLayerNormBackwardGradX(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    const { ref, shape } = fusedLayerNormBackwardGradXOp(
      gradOutput.lazyRef,
      x.lazyRef,
      weight.lazyRef,
      x.shape,
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, x.device);
  }

  fusedLayerNormBackwardGradWeightBias(
    gradOutput: Tensor,
    x: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    const { ref, shape } = fusedLayerNormBackwardGradWeightBiasOp(
      gradOutput.lazyRef,
      x.lazyRef,
      gradOutput.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, gradOutput.device);
  }

  fusedRMSNormForward(
    x: Tensor,
    weight: Tensor,
    config: FusedRMSNormConfig,
  ): Tensor {
    const { ref, shape } = fusedRMSNormForwardOp(
      x.lazyRef,
      weight.lazyRef,
      x.shape,
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, x.device);
  }

  fusedRMSNormBackwardGradX(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedRMSNormConfig,
  ): Tensor {
    const { ref, shape } = fusedRMSNormBackwardGradXOp(
      gradOutput.lazyRef,
      x.lazyRef,
      weight.lazyRef,
      x.shape,
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, x.device);
  }

  fusedRMSNormBackwardGradWeight(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedRMSNormConfig,
  ): Tensor {
    const { ref, shape } = fusedRMSNormBackwardGradWeightOp(
      gradOutput.lazyRef,
      x.lazyRef,
      weight.lazyRef,
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, x.device);
  }

  fusedAttentionForward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    const { ref, shape } = fusedAttentionForwardOp(
      q.lazyRef,
      k.lazyRef,
      v.lazyRef,
      q.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, q.device);
  }

  extractAttentionLogsumexp(
    fwdOutput: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    const { ref, shape } = extractAttentionLogsumexpOp(
      fwdOutput.lazyRef,
      fwdOutput.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, fwdOutput.device);
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
    const { ref, shape } = fusedAttentionBackwardOp(
      q.lazyRef,
      k.lazyRef,
      v.lazyRef,
      logsumexp.lazyRef,
      dO.lazyRef,
      output.lazyRef,
      q.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, q.device);
  }

  extractAttentionDK(bwdDQ: Tensor, config: FusedAttentionConfig): Tensor {
    const { ref, shape } = extractAttentionDKOp(
      bwdDQ.lazyRef,
      bwdDQ.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, bwdDQ.device);
  }

  extractAttentionDV(bwdDQ: Tensor, config: FusedAttentionConfig): Tensor {
    const { ref, shape } = extractAttentionDVOp(
      bwdDQ.lazyRef,
      bwdDQ.device,
      config,
    );
    return this.createAndTrack(createBaseId(), ref, shape, bwdDQ.device);
  }

  extractLnBwdGradBias(gradWeight: Tensor, featureDim: number): Tensor {
    const { ref, shape } = extractLnBwdGradBiasOp(
      gradWeight.lazyRef,
      gradWeight.device,
      featureDim,
    );
    return this.createAndTrack(createBaseId(), ref, shape, gradWeight.device);
  }
}
