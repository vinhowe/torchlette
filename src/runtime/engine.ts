import { getActiveBackend, getBackend } from "../backend/registry";
import {
  type ArgReduceOptions,
  type Backend,
  type BackendTensor,
  computeContiguousStrides,
  type DeviceKind,
  type DivOptions,
  type DType,
  type GatherOptions,
  type GeluOptions,
  type MaxOptions,
  type MeanOptions,
  type ScatterAddOptions,
  type StridedScatterOptions,
  type FusedAttentionConfig,
  type FusedCrossEntropyConfig,
  type FusedLayerNormConfig,
  type SubOptions,
  type SumOptions,
  type TransposeOptions,
} from "../backend/types";
import {
  buildMergedPlan,
  buildPlan,
  createLazyIRNode,
  createMaterializedRef,
  createPendingRef,
  createScalarRef,
  createStorageHandle,
  executePlan,
  executePlanOptimized,
  executePlanWithCheckpointSegments,
  executePlanWithTrueSegments,
  type LazyIRNode,
  type LazyOpCode,
  type LazyRef,
  type OptimizedExecutionStats,
  storageTracker,
  executeLoweredPlan,
  getFusionAnalysisTemplate,
} from "../engine/lazy";
import { computePlanFingerprint } from "../engine/fusion-detect";
import { OP_DTYPE_RULES, promoteDtype } from "../engine/dtype-rules";
import {
  executeWithMemoryPlanning,
  getMemoryPlannerStats,
  type MemoryPlanningStats,
} from "../engine/memory-planned-executor";
import { type BaseId, createBaseId, materializePendingTensors, Tensor } from "./tensor";

/** A tensor or a numeric scalar (will be inlined as a constant in fused kernels). */
export type TensorOrScalar = Tensor | number;

/**
 * Lightweight reference to a lazy computation with shape/dtype metadata.
 * Used internally by ensureDtypeSafety to build cast nodes without creating
 * full Tensor objects (which would register in pendingTensorsByNodeId and leak).
 * Tensor satisfies this interface, so existing Tensors can be used directly.
 */
type OpInput = Pick<Tensor, "lazyRef" | "shape" | "dtype" | "device">;

/** Build a cast LazyRef without creating a Tensor (no lifecycle, no registration). */
function castRef(input: OpInput, toDtype: DType): OpInput {
  const node = createLazyIRNode("cast", [input.lazyRef], input.shape.slice(), toDtype, input.device, { dtype: toDtype });
  return { lazyRef: createPendingRef(node), shape: input.shape, dtype: toDtype, device: input.device };
}

/**
 * Dispatch mode: receives notifications when RuntimeTensors are created or escape a scope.
 * Pushed/popped on a stack on RuntimeEngine. All active modes are notified.
 */
export interface DispatchMode {
  onTensorCreated(tensor: Tensor): void;
  onTensorEscaped?(tensor: Tensor): void;
}

/**
 * Dispatch mode that tracks RuntimeTensors created within a tidy scope.
 * Tensors that are wrapped (via markEscaped) are exempt from disposal.
 * At scope exit, disposeNonEscaped() cleans up unwrapped intermediates.
 */
export class TidyDispatchMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  readonly escaped = new Set<Tensor>();

  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }

  onTensorEscaped(tensor: Tensor): void {
    this.escaped.add(tensor);
  }

  /** Dispose all RuntimeTensors that weren't wrapped (didn't escape). */
  disposeNonEscaped(): void {
    for (const t of this.tracked) {
      if (!this.escaped.has(t) && !t.disposed) {
        t.dispose();
      }
    }
  }
}

export interface RuntimeEngineOptions {
  /** Enable memory planning for buffer pooling and reuse */
  enableMemoryPlanning?: boolean;
  /** Enable buffer donation (reuse input buffers for outputs) */
  enableDonation?: boolean;
  /** Track memory planning statistics */
  trackStats?: boolean;
  /** Enable automatic fusion of elementwise ops (§15) */
  enableFusion?: boolean;
  /** Enable vectorization for fused kernels */
  enableVectorization?: boolean;
  /** Enable early buffer release during execution for memory savings */
  enableEarlyRelease?: boolean;
  /**
   * Enable segmented execution at checkpoint boundaries.
   * This enables memory savings for large models by executing segments
   * separately and flushing buffers between them.
   */
  enableCheckpointSegmentation?: boolean;
  /**
   * Enable true segmented execution with GPU synchronization.
   * Provides actual memory savings for checkpointed models at cost of
   * increased latency due to GPU sync points between segments.
   * This is more effective than enableCheckpointSegmentation but slower.
   */
  enableTrueSegmentation?: boolean;
}

/** Internal dispatch mode used by startIntermediateTracking/stopIntermediateTracking. */
class IntermediateTrackingMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }
}

export class RuntimeEngine {
  private defaultDevice: DeviceKind | null = null;
  private memoryPlanningEnabled = false;
  private donationEnabled = true;
  private trackStats = false;
  private lastStats: MemoryPlanningStats | null = null;
  private fusionEnabled = false;
  private vectorizationEnabled = false;
  private lastFusionStats: OptimizedExecutionStats | null = null;
  private cumulativeFusionStats: OptimizedExecutionStats | null = null;
  private earlyReleaseEnabled = false;
  private checkpointSegmentationEnabled = false;
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
    if (options?.enableMemoryPlanning) {
      this.memoryPlanningEnabled = true;
    }
    if (options?.enableDonation !== undefined) {
      this.donationEnabled = options.enableDonation;
    }
    if (options?.trackStats) {
      this.trackStats = true;
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
   * Enable or disable memory planning.
   */
  setMemoryPlanning(enabled: boolean): void {
    this.memoryPlanningEnabled = enabled;
  }

  /**
   * Check if memory planning is enabled.
   */
  isMemoryPlanningEnabled(): boolean {
    return this.memoryPlanningEnabled;
  }

  /**
   * Get statistics from the last memory-planned execution.
   */
  getLastMemoryStats(): MemoryPlanningStats | null {
    return this.lastStats;
  }

  /**
   * Get overall memory planner statistics.
   */
  getMemoryPlannerStats(): ReturnType<typeof getMemoryPlannerStats> {
    return getMemoryPlannerStats();
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
   * Enable or disable vectorization for fused kernels.
   */
  setVectorizationEnabled(enabled: boolean): void {
    this.vectorizationEnabled = enabled;
  }

  /**
   * Check if vectorization is enabled.
   */
  isVectorizationEnabled(): boolean {
    return this.vectorizationEnabled;
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

  /**
   * Enable or disable segmented execution at checkpoint boundaries.
   * When enabled, the executor will flush buffers between checkpoint segments,
   * enabling memory savings for large models that don't fit in GPU memory.
   */
  setCheckpointSegmentationEnabled(enabled: boolean): void {
    this.checkpointSegmentationEnabled = enabled;
  }

  /**
   * Check if checkpoint segmentation is enabled.
   */
  isCheckpointSegmentationEnabled(): boolean {
    return this.checkpointSegmentationEnabled;
  }

  /**
   * Enable or disable true segmented execution with GPU synchronization.
   * This provides actual memory savings for checkpointed models by waiting
   * for GPU completion between segments before releasing buffers.
   */
  setTrueSegmentationEnabled(enabled: boolean): void {
    this.trueSegmentationEnabled = enabled;
  }

  /**
   * Check if true segmentation is enabled.
   */
  isTrueSegmentationEnabled(): boolean {
    return this.trueSegmentationEnabled;
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
      throw new Error("Expected IntermediateTrackingMode on dispatch mode stack");
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

  /**
   * Ensure all tensors are on the same device, auto-transferring if needed.
   * Returns the tensors (possibly transferred) and the target device.
   */
  private ensureSameDevice(...tensors: Tensor[]): {
    tensors: Tensor[];
    device: DeviceKind;
  } {
    if (tensors.length === 0) {
      return { tensors: [], device: this.getDevice() };
    }

    // Determine target device - prefer GPU if any tensor is on GPU
    let targetDevice = tensors[0].device;
    for (const tensor of tensors) {
      if (tensor.device === "webgpu") {
        targetDevice = "webgpu";
        break;
      }
    }

    // Transfer any tensors not on target device
    const result: Tensor[] = [];
    for (const tensor of tensors) {
      if (tensor.device !== targetDevice) {
        result.push(this.transfer(tensor, targetDevice));
      } else {
        result.push(tensor);
      }
    }

    return { tensors: result, device: targetDevice };
  }

  /**
   * Force a tensor to materialize by executing its computation graph.
   */
  async force(tensor: Tensor): Promise<void> {
    if (tensor.isMaterialized() || tensor.disposed) {
      return;
    }

    const lazyRef = tensor.lazyRef;
    if (lazyRef.kind !== "pending") {
      return;
    }

    const _forceTiming = process.env.TORCHLETTE_REPLAY_TIMING === "1";
    const _forceT0 = _forceTiming ? performance.now() : 0;
    const plan = buildPlan(lazyRef.node);
    const _buildT = _forceTiming ? performance.now() - _forceT0 : 0;
    const backend = this.getBackend(tensor.device);

    // --- Lowered Plan Fast-Path ---
    // Try cached lowered execution plan first.
    if (tensor.device === "webgpu" && this.fusionEnabled) {
      const _lpT0 = _forceTiming ? performance.now() : 0;
      const executed = await this.tryLoweredPlanExecution(plan, [tensor], tensor.device);
      if (_forceTiming) console.log(`[force-timing] nodes=${plan.nodes.length} buildPlan=${_buildT.toFixed(1)}ms lowered=${executed ? "HIT" : "MISS"} loweredTime=${(performance.now() - _lpT0).toFixed(1)}ms`);
      if (executed) return;
    }

    // Get buffer pool flush function for checkpoint segmentation
    const flushBufferPool = this.getBufferPoolFlushFn(tensor.device);

    let result;
    if (this.fusionEnabled) {
      // Use optimized execution with fusion (§15)
      // executePlanOptimized supports enableEarlyRelease for memory management
      // On non-WebGPU backends, fusion is automatically skipped inside executePlanOptimized
      const optimizedResult = await executePlanOptimized(plan, backend, {
        enableFusion: true,
        enableVectorization: this.vectorizationEnabled,
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
      result = optimizedResult.result;
      this.lastFusionStats = optimizedResult.stats;
      this.accumulateFusionStats(optimizedResult.stats);
    } else if (this.memoryPlanningEnabled) {
      // Use memory-planned execution (for non-fusion or CPU)
      const memResult = await executeWithMemoryPlanning(plan, backend, {
        enableDonation: this.donationEnabled,
        trackStats: this.trackStats,
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
      result = memResult.result;
      this.lastStats = memResult.stats;
    } else if (this.trueSegmentationEnabled && tensor.device === "webgpu") {
      // Use true segmented execution with GPU synchronization
      // This provides actual memory savings by waiting for GPU completion
      // between segments before releasing buffers
      result = await executePlanWithTrueSegments(
        plan,
        backend,
        { enableEarlyRelease: this.earlyReleaseEnabled },
      );
    } else if (this.checkpointSegmentationEnabled) {
      // Use segmented execution at checkpoint boundaries
      result = await executePlanWithCheckpointSegments(
        plan,
        backend,
        { enableEarlyRelease: this.earlyReleaseEnabled },
        flushBufferPool,
      );
    } else {
      // Use standard execution
      result = await executePlan(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
    }

    // Materialize ALL tensors that were pending on executed nodes.
    // This ensures model parameters and other user-held tensors get their
    // storages marked as externally reachable.
    for (const node of plan.nodes) {
      if (node.result) {
        materializePendingTensors(node.id, node.result);
      }
    }

    // Materialize this tensor (in case it wasn't already handled above)
    if (!tensor.isMaterialized()) {
      tensor._materialize(result);
    }

    // Drop node.result references to allow GC of unclaimed intermediate storages
    for (const node of plan.nodes) {
      node.result = undefined;
    }

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
    const planNodes = template.finalPerm.map(i => plan.nodes[i]);

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
    let savedCounters: { dispatch: number; params: number; output: number } | undefined;
    if (device === "webgpu") {
      try {
        const { getDispatchSequenceCounters } = await import("../backend/webgpu/index");
        savedCounters = getDispatchSequenceCounters();
      } catch { /* non-WebGPU runtime */ }
    }

    try {
      const { result, stats: loweredStats } = await executeLoweredPlan(
        plan as { nodes: LazyIRNode[] },
        planNodes,
        template.loweredPlan,
        backend,
        {
          enableEarlyRelease: this.earlyReleaseEnabled,
          enableVectorization: this.vectorizationEnabled,
          bufferArena: useArena ? template.bufferArena : undefined,
        },
      );

      // Update fusion stats for consistency with normal execution path
      this.lastFusionStats = loweredStats;
      this.accumulateFusionStats(loweredStats);

      // Materialize ALL tensors that were pending on executed nodes
      const { materializePendingTensors: materialize } = await import("./tensor");
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

      // Drop node.result references to allow GC
      for (const node of plan.nodes) {
        node.result = undefined;
      }

      return true;
    } catch (err) {
      // Lowered plan execution failed — clean up and fall through.
      // Disable lowered plan for this fingerprint so we don't retry every step.
      // The template's bufferArena is preserved so executePlanOptimized can use it.
      if (process.env.TORCHLETTE_REPLAY_TIMING === "1") {
        console.log(`[lowered-plan-fail] nodes=${plan.nodes.length} fingerprint=${fingerprint} error=${err instanceof Error ? err.message : String(err)}`);
      }
      template.loweredPlan = undefined;

      // Restore dispatch sequence counters so the fallback executePlanOptimized
      // starts at the same position, keeping bind group cache indices aligned.
      if (savedCounters && device === "webgpu") {
        try {
          const { setDispatchSequenceCounters } = await import("../backend/webgpu/index");
          setDispatchSequenceCounters(savedCounters.dispatch, savedCounters.params, savedCounters.output);
        } catch { /* non-WebGPU runtime */ }
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
    const _famTiming = process.env.TORCHLETTE_REPLAY_TIMING === "1";
    const _famT0 = _famTiming ? performance.now() : 0;
    const plan = buildMergedPlan(pendingRoots);
    const _famBuildT = _famTiming ? performance.now() - _famT0 : 0;

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

    // Lowered plan fast-path for forceAllMerged
    if (device === "webgpu" && this.fusionEnabled) {
      const _famLpT0 = _famTiming ? performance.now() : 0;
      const executed = await this.tryLoweredPlanExecution(plan, tensors, device);
      if (_famTiming) console.log(`[forceAllMerged-timing] nodes=${plan.nodes.length} pendingRoots=${pendingRoots.length} buildPlan=${_famBuildT.toFixed(1)}ms lowered=${executed ? "HIT" : "MISS"} loweredTime=${(performance.now() - _famLpT0).toFixed(1)}ms`);
      if (executed) return;
    }

    const backend = this.getBackend(device);

    // Check if plan has checkpoint boundaries - only segment if checkpointing is used
    const hasCheckpointBoundaries = plan.nodes.some((n) => n.isCheckpointBoundary);

    // Use fusion when enabled and on WebGPU, regardless of segmentation settings.
    // executePlanOptimized supports enableEarlyRelease for memory management.
    if (this.fusionEnabled) {
      const optimizedResult = await executePlanOptimized(plan, backend, {
        enableFusion: true,
        enableVectorization: this.vectorizationEnabled,
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
      this.lastFusionStats = optimizedResult.stats;
      this.accumulateFusionStats(optimizedResult.stats);
    } else if (this.trueSegmentationEnabled && hasCheckpointBoundaries && device === "webgpu") {
      // True segmentation with GPU sync between segments
      await executePlanWithTrueSegments(plan, backend, {
        enableEarlyRelease: this.earlyReleaseEnabled,
      });
    } else if (this.checkpointSegmentationEnabled && hasCheckpointBoundaries) {
      // Use checkpoint segmentation (buffer pool flush only, no GPU sync)
      const flushBufferPool = this.getBufferPoolFlushFn(device);
      await executePlanWithCheckpointSegments(
        plan,
        backend,
        { enableEarlyRelease: this.earlyReleaseEnabled },
        flushBufferPool,
      );
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

    // Drop node.result references to allow GC of unclaimed intermediate storages
    for (const node of plan.nodes) {
      node.result = undefined;
    }

  }

  /**
   * Force multiple tensors to materialize using a single merged execution plan.
   * This is always more efficient than forcing individually, and enables proper
   * checkpoint segmentation when checkpoint boundaries are present.
   */
  async forceAll(...tensors: Tensor[]): Promise<void> {
    await this.forceAllMerged(...tensors);
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
    const { getAllPendingTensors, materializePendingTensors: materialize } = await import("./tensor");
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
      const executed = await this.tryLoweredPlanExecution(plan, pendingTensors, device);
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
    values: number[],
    shape: number[],
    device?: DeviceKind,
  ): Tensor {
    const resolvedDevice = this.getDevice(device);
    const node = createLazyIRNode(
      "tensorFromArray",
      [],
      shape,
      "f32",
      resolvedDevice,
      { values: values.slice() },
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
    const node = createLazyIRNode(
      "zeros",
      [],
      shape,
      "f32",
      resolvedDevice,
    );
    const lazyRef: LazyRef = createPendingRef(node);
    return this.createAndTrack(createBaseId(), lazyRef, shape, resolvedDevice);
  }

  /**
   * Create a tensor filled with a constant value.
   * Uses the "full" lazy op so the backend can create the buffer directly.
   */
  full(shape: number[], fillValue: number, device?: DeviceKind): Tensor {
    const resolvedDevice = this.getDevice(device);
    const node = createLazyIRNode(
      "full",
      [],
      shape,
      "f32",
      resolvedDevice,
      { fillValue },
    );
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
    const node = createLazyIRNode(
      "arange",
      [],
      shape,
      "f32",
      resolvedDevice,
      { end, start, step },
    );
    const lazyRef: LazyRef = createPendingRef(node);
    return this.createAndTrack(createBaseId(), lazyRef, shape, resolvedDevice);
  }

  /**
   * Return the lower-triangular part of a matrix (or batch of matrices).
   * Elements above the k-th diagonal are zeroed.
   */
  tril(a: Tensor, k = 0): Tensor {
    if (a.shape.length < 2) throw new Error("tril requires at least 2 dimensions");
    const node = createLazyIRNode("tril", [a.lazyRef], a.shape, a.dtype, a.device, { k });
    return this.createAndTrack(createBaseId(), createPendingRef(node), a.shape, a.device, a.dtype);
  }

  /**
   * Return the upper-triangular part of a matrix (or batch of matrices).
   * Elements below the k-th diagonal are zeroed.
   */
  triu(a: Tensor, k = 0): Tensor {
    if (a.shape.length < 2) throw new Error("triu requires at least 2 dimensions");
    const node = createLazyIRNode("triu", [a.lazyRef], a.shape, a.dtype, a.device, { k });
    return this.createAndTrack(createBaseId(), createPendingRef(node), a.shape, a.device, a.dtype);
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
      return inputs.map(t => t.dtype === "f16" ? castRef(t, "f32") : t);
    }

    return inputs;
  }

  /**
   * Resolve a TensorOrScalar operand to a LazyRef.
   * Numbers become scalar LazyRefs (no graph node, no GPU buffer).
   * The refTensor provides dtype/device context for the scalar.
   */
  private resolveOperand(value: OpInput | number, ref: OpInput): { ref: LazyRef; shape: number[] } {
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
  ): { refA: LazyRef; refB: LazyRef; shape: number[]; dtype: DType; device: DeviceKind } {
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
    return { refA: resA.ref, refB: resB.ref, shape, dtype: ref.dtype, device: ref.device };
  }

  add(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, dtype, device } = this.resolveBinaryOp("add", a, b);
    const node = createLazyIRNode("add", [refA, refB], shape, dtype, device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
  }

  sub(a: TensorOrScalar, b: TensorOrScalar, options?: SubOptions): Tensor {
    const { refA, refB, shape, dtype, device } = this.resolveBinaryOp("sub", a, b);
    const node = createLazyIRNode("sub", [refA, refB], shape, dtype, device, options);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
  }

  div(a: TensorOrScalar, b: TensorOrScalar, options?: DivOptions): Tensor {
    const { refA, refB, shape, dtype, device } = this.resolveBinaryOp("div", a, b);
    const node = createLazyIRNode("div", [refA, refB], shape, dtype, device, options);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
  }

  mul(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, dtype, device } = this.resolveBinaryOp("mul", a, b);
    const node = createLazyIRNode("mul", [refA, refB], shape, dtype, device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
  }

  view(a: Tensor, shape: number[]): Tensor {
    const node = createLazyIRNode(
      "reshape",
      [a.lazyRef],
      shape,
      a.dtype,
      a.device,
      { targetShape: shape },
    );
    // View shares baseId with input
    return this.createAndTrack(a.baseId, createPendingRef(node), shape, a.device, a.dtype);
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
    // Reshape shares baseId with input
    return this.createAndTrack(a.baseId, createPendingRef(node), shape, a.device, a.dtype);
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    const device = this.assertSameDevice(a, b);
    const shape = matmulShape(a.shape, b.shape);
    // Output dtype = higher precision of inputs (f32 if mixed)
    const dtype = (a.dtype === "f32" || b.dtype === "f32") ? "f32" as const : a.dtype;
    const node = createLazyIRNode(
      "matmul",
      [a.lazyRef, b.lazyRef],
      shape,
      dtype,
      device,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
  }

  sqrt(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "sqrt",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  relu(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "relu",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  exp(a: Tensor): Tensor {
    const [op] = this.ensureDtypeSafety("exp", [a]);
    const node = createLazyIRNode(
      "exp",
      [op.lazyRef],
      op.shape.slice(),
      op.dtype,
      op.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      op.shape.slice(),
      op.device,
      op.dtype,
    );
  }

  log(a: Tensor): Tensor {
    const [op] = this.ensureDtypeSafety("log", [a]);
    const node = createLazyIRNode(
      "log",
      [op.lazyRef],
      op.shape.slice(),
      op.dtype,
      op.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      op.shape.slice(),
      op.device,
      op.dtype,
    );
  }

  neg(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "neg",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  abs(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "abs",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  tanh(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "tanh",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  sigmoid(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "sigmoid",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  gelu(a: Tensor, options?: GeluOptions): Tensor {
    const node = createLazyIRNode(
      "gelu",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
      options, // Pass options as payload
    );
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape.slice(),
      a.device,
      a.dtype,
    );
  }

  silu(a: Tensor): Tensor {
    const node = createLazyIRNode(
      "silu",
      [a.lazyRef],
      a.shape.slice(),
      a.dtype,
      a.device,
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
   * Check if values are finite (not NaN and not Inf).
   * Returns 1.0 where finite, 0.0 where NaN or Inf.
   */
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
    // Expand shares baseId with input
    return this.createAndTrack(a.baseId, createPendingRef(node), shape, a.device, a.dtype);
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
    // Transpose shares baseId with input
    return this.createAndTrack(a.baseId, createPendingRef(node), shape, a.device, a.dtype);
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
      const nd = d < 0 ? d + rank : d;
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
    const normalizedDims = dims.map((d) => (d < 0 ? d + rank : d));
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
    return this.createAndTrack(a.baseId, createPendingRef(node), shape, a.device, a.dtype);
  }

  contiguous(a: Tensor): Tensor {
    // contiguous materializes non-contiguous tensors to new contiguous buffer
    const node = createLazyIRNode(
      "contiguous",
      [a.lazyRef],
      a.shape,
      a.dtype,
      a.device,
      undefined,
    );
    // contiguous may create new storage, so gets new baseId
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape,
      a.device,
      a.dtype,
    );
  }

  narrow(a: Tensor, dim: number, start: number, length: number): Tensor {
    const rank = a.shape.length;
    if (dim < 0 || dim >= rank) {
      throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
    }
    if (start < 0 || start + length > a.shape[dim]) {
      throw new Error(`narrow: range [${start}, ${start + length}) out of bounds for dim size ${a.shape[dim]}`);
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
    return this.createAndTrack(a.baseId, createPendingRef(node), shape, a.device, a.dtype);
  }

  narrowBackward(grad: Tensor, dim: number, start: number, originalLength: number): Tensor {
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
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, grad.device, grad.dtype);
  }

  /**
   * Cast tensor to a different dtype.
   * Returns a new tensor with the specified dtype.
   */
  cast(a: Tensor, dtype: DType): Tensor {
    const node = createLazyIRNode(
      "cast",
      [a.lazyRef],
      a.shape,
      dtype,
      a.device,
      { dtype },
    );
    // cast creates new storage with different dtype
    return this.createAndTrack(
      createBaseId(),
      createPendingRef(node),
      a.shape,
      a.device,
      dtype,
    );
  }

  gather(a: Tensor, index: Tensor, options: GatherOptions): Tensor {
    const device = this.assertSameDevice(a, index);
    // Gather output shape matches index shape
    const shape = index.shape.slice();
    const dtype = a.dtype;
    const node = createLazyIRNode(
      "gather",
      [a.lazyRef, index.lazyRef],
      shape,
      dtype,
      device,
      options,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
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
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
  }

  sum(a: Tensor, options?: SumOptions): Tensor {
    const [op] = this.ensureDtypeSafety("sum", [a]);
    const shape = reduceShape(op.shape, options?.dim, options?.keepdim ?? false);

    // If reducing to scalar and no keepdim, return number
    // But we can't know the actual value without forcing - return a tensor
    const dtype = op.dtype;
    const node = createLazyIRNode(
      "sum",
      [op.lazyRef],
      shape,
      dtype,
      op.device,
      options,
    );

    // If output is scalar [], we still return a Tensor for lazy evaluation
    // The frontend will handle scalar conversion at force time
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, op.device, dtype);
  }

  max(a: Tensor, options?: MaxOptions): number | Tensor {
    const [op] = this.ensureDtypeSafety("max", [a]);
    const shape = reduceShape(op.shape, options?.dim, options?.keepdim ?? false);
    const dtype = op.dtype;

    const node = createLazyIRNode(
      "max",
      [op.lazyRef],
      shape,
      dtype,
      op.device,
      options,
    );

    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, op.device, dtype);
  }

  mean(a: Tensor, options?: MeanOptions): number | Tensor {
    const [op] = this.ensureDtypeSafety("mean", [a]);
    const shape = reduceShape(op.shape, options?.dim, options?.keepdim ?? false);
    const dtype = op.dtype;

    const node = createLazyIRNode(
      "mean",
      [op.lazyRef],
      shape,
      dtype,
      op.device,
      options,
    );

    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, op.device, dtype);
  }

  argmax(a: Tensor, options: ArgReduceOptions): Tensor {
    const normalizedDim =
      options.dim < 0 ? a.shape.length + options.dim : options.dim;
    const shape = options.keepdim
      ? a.shape.map((s, i) => (i === normalizedDim ? 1 : s))
      : a.shape.filter((_, i) => i !== normalizedDim);

    const node = createLazyIRNode(
      "argmax",
      [a.lazyRef],
      shape,
      "f32",
      a.device,
      { dim: normalizedDim, keepdim: options.keepdim },
    );

    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, a.device);
  }

  argmin(a: Tensor, options: ArgReduceOptions): Tensor {
    const normalizedDim =
      options.dim < 0 ? a.shape.length + options.dim : options.dim;
    const shape = options.keepdim
      ? a.shape.map((s, i) => (i === normalizedDim ? 1 : s))
      : a.shape.filter((_, i) => i !== normalizedDim);

    const node = createLazyIRNode(
      "argmin",
      [a.lazyRef],
      shape,
      "f32",
      a.device,
      { dim: normalizedDim, keepdim: options.keepdim },
    );

    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, a.device);
  }

  gt(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp("gt", a, b);
    const node = createLazyIRNode("gt", [refA, refB], shape, "f32", device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device);
  }

  lt(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp("lt", a, b);
    const node = createLazyIRNode("lt", [refA, refB], shape, "f32", device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device);
  }

  ge(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp("ge", a, b);
    const node = createLazyIRNode("ge", [refA, refB], shape, "f32", device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device);
  }

  le(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp("le", a, b);
    const node = createLazyIRNode("le", [refA, refB], shape, "f32", device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device);
  }

  eq(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp("eq", a, b);
    const node = createLazyIRNode("eq", [refA, refB], shape, "f32", device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device);
  }

  ne(a: TensorOrScalar, b: TensorOrScalar): Tensor {
    const { refA, refB, shape, device } = this.resolveBinaryOp("ne", a, b);
    const node = createLazyIRNode("ne", [refA, refB], shape, "f32", device);
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device);
  }

  where(condition: Tensor, x: TensorOrScalar, y: TensorOrScalar): Tensor {
    const refT = typeof x !== "number" ? x : typeof y !== "number" ? y : condition;
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
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, device, dtype);
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

  /**
   * Copy src values into dst tensor in-place.
   * Returns dst (same tensor object) with updated lazyRef.
   *
   * Per spec §4.4, this uses strided_scatter_copy to copy src into dst.
   */
  copy_(dst: Tensor, src: Tensor): Tensor {
    this.assertSameDevice(dst, src);

    // Validate shapes match
    if (dst.shape.length !== src.shape.length) {
      throw new Error(
        `copy_: shape mismatch - dst has rank ${dst.shape.length}, src has rank ${src.shape.length}`,
      );
    }
    for (let i = 0; i < dst.shape.length; i++) {
      if (dst.shape[i] !== src.shape[i]) {
        throw new Error(
          `copy_: shape mismatch at dim ${i} - dst has ${dst.shape[i]}, src has ${src.shape[i]}`,
        );
      }
    }

    // Create strided scatter copy node
    // For full tensor copy: offset=0, viewShape=dst.shape, viewStrides=contiguous
    const viewStrides = computeContiguousStrides(dst.shape);
    const payload: StridedScatterOptions = {
      offset: 0,
      viewShape: dst.shape.slice(),
      viewStrides,
    };

    const node = createLazyIRNode(
      "stridedScatterCopy",
      [dst.lazyRef, src.lazyRef],
      dst.shape,
      dst.dtype,
      dst.device,
      payload,
    );

    // Update dst's lazyRef to the new node
    dst._updateLazyRef(createPendingRef(node));

    return dst;
  }

  /**
   * Add src values to dst tensor in-place.
   * Returns dst (same tensor object) with updated lazyRef.
   *
   * Per spec §4.4, this uses strided_scatter_add to add src into dst.
   */
  add_(dst: Tensor, src: Tensor): Tensor {
    this.assertSameDevice(dst, src);

    // Validate shapes match
    if (dst.shape.length !== src.shape.length) {
      throw new Error(
        `add_: shape mismatch - dst has rank ${dst.shape.length}, src has rank ${src.shape.length}`,
      );
    }
    for (let i = 0; i < dst.shape.length; i++) {
      if (dst.shape[i] !== src.shape[i]) {
        throw new Error(
          `add_: shape mismatch at dim ${i} - dst has ${dst.shape[i]}, src has ${src.shape[i]}`,
        );
      }
    }

    // Create strided scatter add node
    const viewStrides = computeContiguousStrides(dst.shape);
    const payload: StridedScatterOptions = {
      offset: 0,
      viewShape: dst.shape.slice(),
      viewStrides,
    };

    const node = createLazyIRNode(
      "stridedScatterAdd",
      [dst.lazyRef, src.lazyRef],
      dst.shape,
      dst.dtype,
      dst.device,
      payload,
    );

    // Update dst's lazyRef to the new node
    dst._updateLazyRef(createPendingRef(node));

    return dst;
  }

  /**
   * Zero out a tensor in-place.
   * Returns dst (same tensor object) with updated lazyRef.
   */
  zero_(dst: Tensor): Tensor {
    // Create a zeros tensor and copy it into dst
    const zerosTensor = this.zeros(dst.shape, dst.device);
    return this.copy_(dst, zerosTensor);
  }

  /**
   * Fill a tensor with a scalar value in-place.
   * Returns dst (same tensor object) with updated lazyRef.
   */
  fill_(dst: Tensor, value: number): Tensor {
    const valuesTensor = this.full(dst.shape, value, dst.device);
    return this.copy_(dst, valuesTensor);
  }

  /**
   * Multiply tensor by scalar in-place.
   * Returns dst (same tensor object) with updated lazyRef.
   */
  mul_(dst: Tensor, value: number): Tensor {
    // Multiply by scalar and copy result back to dst
    const scaledTensor = this.mul(dst, value);
    return this.copy_(dst, scaledTensor);
  }

  /**
   * Begin step-level shared encoder scope.
   * Keeps GPU command encoder open across force() boundaries.
   */
  async beginStep(): Promise<void> {
    await this.getBackend().beginStep?.();
  }

  /**
   * End step-level shared encoder scope.
   * Submits all remaining encoded GPU work.
   */
  endStep(): void {
    this.getBackend().endStep?.();
  }

  /**
   * Wrap a raw BackendTensor into a tracked RuntimeTensor.
   * Creates a new StorageHandle for tracking.
   */
  createFromBackendTensor(
    bt: BackendTensor,
    shape: number[],
    device: DeviceKind,
    dtype: DType = "f32",
  ): Tensor {
    const storage = createStorageHandle(device, bt);
    const ref: LazyRef = createMaterializedRef(storage);
    return this.createAndTrack(createBaseId(), ref, shape, device, dtype);
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

  /**
   * Fused cross-entropy forward: logits [B,V] + targets [B] → per-sample loss [B].
   */
  fusedCrossEntropyForward(
    logits: Tensor,
    targets: Tensor,
    config: FusedCrossEntropyConfig,
  ): Tensor {
    const node = createLazyIRNode(
      "fusedCrossEntropyForward",
      [logits.lazyRef, targets.lazyRef],
      [config.batchSize],
      "f32",
      logits.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), [config.batchSize], logits.device);
  }

  /**
   * Fused cross-entropy backward: logits [B,V] + targets [B] + grad [B] → grad_logits [B,V].
   */
  fusedCrossEntropyBackward(
    logits: Tensor,
    targets: Tensor,
    gradOutput: Tensor,
    config: FusedCrossEntropyConfig,
  ): Tensor {
    const shape = [config.batchSize, config.vocabSize];
    const node = createLazyIRNode(
      "fusedCrossEntropyBackward",
      [logits.lazyRef, targets.lazyRef, gradOutput.lazyRef],
      shape,
      "f32",
      logits.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, logits.device);
  }

  /**
   * Fused LayerNorm forward: x [N,D] + weight [D] + bias [D] → output [N,D].
   */
  fusedLayerNormForward(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    const node = createLazyIRNode(
      "fusedLayerNormForward",
      [x.lazyRef, weight.lazyRef, bias.lazyRef],
      x.shape.slice(),
      "f32",
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), x.shape.slice(), x.device);
  }

  /**
   * Fused LayerNorm backward gradX: grad [N,D] + x [N,D] + weight [D] → gradX [N,D].
   */
  fusedLayerNormBackwardGradX(
    gradOutput: Tensor,
    x: Tensor,
    weight: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    const node = createLazyIRNode(
      "fusedLayerNormBackwardGradX",
      [gradOutput.lazyRef, x.lazyRef, weight.lazyRef],
      x.shape.slice(),
      "f32",
      x.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), x.shape.slice(), x.device);
  }

  /**
   * Fused LayerNorm backward gradWeight+gradBias.
   * Returns gradWeight as main output. gradBias extracted via extractLnBwdGradBias.
   */
  fusedLayerNormBackwardGradWeightBias(
    gradOutput: Tensor,
    x: Tensor,
    config: FusedLayerNormConfig,
  ): Tensor {
    const shape = [config.featureDim];
    const node = createLazyIRNode(
      "fusedLayerNormBackwardGradWeightBias",
      [gradOutput.lazyRef, x.lazyRef],
      shape,
      "f32",
      gradOutput.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, gradOutput.device);
  }

  /**
   * Fused attention forward: Q,K,V [B,H,N,D] → O [B,H,N,D].
   * Logsumexp is extracted via extractAttentionLogsumexp.
   */
  fusedAttentionForward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
    const node = createLazyIRNode(
      "fusedAttentionForward",
      [q.lazyRef, k.lazyRef, v.lazyRef],
      shape,
      "f32",
      q.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, q.device);
  }

  /**
   * Extract logsumexp side output [B,H,N] from fusedAttentionForward.
   */
  extractAttentionLogsumexp(fwdOutput: Tensor, config: FusedAttentionConfig): Tensor {
    const shape = [config.batchSize, config.numHeads, config.seqLen];
    const node = createLazyIRNode(
      "extractAttentionLogsumexp",
      [fwdOutput.lazyRef],
      shape,
      "f32",
      fwdOutput.device,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, fwdOutput.device);
  }

  /**
   * Fused attention backward: Q,K,V,L,dO,O → dQ [B,H,N,D].
   * dK and dV are extracted via extractAttentionDK/DV.
   */
  fusedAttentionBackward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    logsumexp: Tensor,
    dO: Tensor,
    output: Tensor,
    config: FusedAttentionConfig,
  ): Tensor {
    const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
    const node = createLazyIRNode(
      "fusedAttentionBackward",
      [q.lazyRef, k.lazyRef, v.lazyRef, logsumexp.lazyRef, dO.lazyRef, output.lazyRef],
      shape,
      "f32",
      q.device,
      config,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, q.device);
  }

  /**
   * Extract dK side output from fusedAttentionBackward.
   */
  extractAttentionDK(bwdDQ: Tensor, config: FusedAttentionConfig): Tensor {
    const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
    const node = createLazyIRNode(
      "extractAttentionDK",
      [bwdDQ.lazyRef],
      shape,
      "f32",
      bwdDQ.device,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, bwdDQ.device);
  }

  /**
   * Extract dV side output from fusedAttentionBackward.
   */
  extractAttentionDV(bwdDQ: Tensor, config: FusedAttentionConfig): Tensor {
    const shape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
    const node = createLazyIRNode(
      "extractAttentionDV",
      [bwdDQ.lazyRef],
      shape,
      "f32",
      bwdDQ.device,
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, bwdDQ.device);
  }

  /**
   * Extract the gradBias side output from a fusedLayerNormBackwardGradWeightBias node.
   */
  extractLnBwdGradBias(gradWeight: Tensor, featureDim: number): Tensor {
    const shape = [featureDim];
    const node = createLazyIRNode(
      "extractLnBwdGradBias",
      [gradWeight.lazyRef],
      shape,
      "f32",
      gradWeight.device,
      { featureDim },
    );
    return this.createAndTrack(createBaseId(), createPendingRef(node), shape, gradWeight.device);
  }

}

export const defaultEngine = new RuntimeEngine();

export function tensorFromArray(
  values: number[],
  shape: number[],
  device?: DeviceKind,
): Tensor {
  return defaultEngine.tensorFromArray(values, shape, device);
}

export function add(a: TensorOrScalar, b: TensorOrScalar): Tensor {
  return defaultEngine.add(a, b);
}

export function sub(a: TensorOrScalar, b: TensorOrScalar, options?: SubOptions): Tensor {
  return defaultEngine.sub(a, b, options);
}

export function div(a: TensorOrScalar, b: TensorOrScalar, options?: DivOptions): Tensor {
  return defaultEngine.div(a, b, options);
}

export function mul(a: TensorOrScalar, b: TensorOrScalar): Tensor {
  return defaultEngine.mul(a, b);
}

export function view(a: Tensor, shape: number[]): Tensor {
  return defaultEngine.view(a, shape);
}

export function reshape(a: Tensor, shape: number[]): Tensor {
  return defaultEngine.reshape(a, shape);
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.matmul(a, b);
}

export function sqrt(a: Tensor): Tensor {
  return defaultEngine.sqrt(a);
}

export function relu(a: Tensor): Tensor {
  return defaultEngine.relu(a);
}

export function exp(a: Tensor): Tensor {
  return defaultEngine.exp(a);
}

export function log(a: Tensor): Tensor {
  return defaultEngine.log(a);
}

export function neg(a: Tensor): Tensor {
  return defaultEngine.neg(a);
}

export function abs(a: Tensor): Tensor {
  return defaultEngine.abs(a);
}

export function tanh(a: Tensor): Tensor {
  return defaultEngine.tanh(a);
}

export function sigmoid(a: Tensor): Tensor {
  return defaultEngine.sigmoid(a);
}

export function gelu(a: Tensor, options?: GeluOptions): Tensor {
  return defaultEngine.gelu(a, options);
}

export function silu(a: Tensor): Tensor {
  return defaultEngine.silu(a);
}

export function isfinite(a: Tensor): Tensor {
  return defaultEngine.isfinite(a);
}

export function expand(a: Tensor, shape: number[]): Tensor {
  return defaultEngine.expand(a, shape);
}

export function transpose(a: Tensor, options: TransposeOptions): Tensor {
  return defaultEngine.transpose(a, options);
}

export function permute(a: Tensor, dims: number[]): Tensor {
  return defaultEngine.permute(a, dims);
}

export function contiguous(a: Tensor): Tensor {
  return defaultEngine.contiguous(a);
}

export function cast(a: Tensor, dtype: DType): Tensor {
  return defaultEngine.cast(a, dtype);
}

export function gather(
  a: Tensor,
  index: Tensor,
  options: GatherOptions,
): Tensor {
  return defaultEngine.gather(a, index, options);
}

export function scatterAdd(
  a: Tensor,
  index: Tensor,
  src: Tensor,
  options: ScatterAddOptions,
): Tensor {
  return defaultEngine.scatterAdd(a, index, src, options);
}

export function sum(a: Tensor, options?: SumOptions): number | Tensor {
  return defaultEngine.sum(a, options);
}

export function max(a: Tensor, options?: MaxOptions): number | Tensor {
  return defaultEngine.max(a, options);
}

export function mean(a: Tensor, options?: MeanOptions): number | Tensor {
  return defaultEngine.mean(a, options);
}

export function argmax(a: Tensor, options: ArgReduceOptions): Tensor {
  return defaultEngine.argmax(a, options);
}

export function argmin(a: Tensor, options: ArgReduceOptions): Tensor {
  return defaultEngine.argmin(a, options);
}

export function gt(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.gt(a, b);
}

export function lt(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.lt(a, b);
}

export function ge(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.ge(a, b);
}

export function le(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.le(a, b);
}

export function eq(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.eq(a, b);
}

export function ne(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.ne(a, b);
}

export function where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
  return defaultEngine.where(condition, x, y);
}

export async function cpu(a: Tensor): Promise<number[]> {
  return defaultEngine.cpu(a);
}

export async function item(a: Tensor): Promise<number> {
  return defaultEngine.item(a);
}

// In-place operations

export function copy_(dst: Tensor, src: Tensor): Tensor {
  return defaultEngine.copy_(dst, src);
}

export function add_(dst: Tensor, src: Tensor): Tensor {
  return defaultEngine.add_(dst, src);
}

export function zero_(dst: Tensor): Tensor {
  return defaultEngine.zero_(dst);
}

export function fill_(dst: Tensor, value: number): Tensor {
  return defaultEngine.fill_(dst, value);
}

export function mul_(dst: Tensor, value: number): Tensor {
  return defaultEngine.mul_(dst, value);
}

// Helper functions for shape inference

function broadcastShapes(a: number[], b: number[]): number[] {
  const outRank = Math.max(a.length, b.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i++) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim);
  }
  return out;
}

function matmulShape(a: number[], b: number[]): number[] {
  if (a.length < 1 || b.length < 1) {
    throw new Error("matmul requires at least 1D tensors");
  }

  // Handle 1D cases
  if (a.length === 1 && b.length === 1) {
    // Vector dot product: [n] @ [n] -> []
    return [];
  }
  if (a.length === 1) {
    // [n] @ [..., n, m] -> [..., m]
    return [...b.slice(0, -2), b[b.length - 1]];
  }
  if (b.length === 1) {
    // [..., m, n] @ [n] -> [..., m]
    return a.slice(0, -1);
  }

  // 2D+ matmul
  const m = a[a.length - 2];
  const n = b[b.length - 1];

  // Broadcast batch dimensions
  const aBatch = a.slice(0, -2);
  const bBatch = b.slice(0, -2);
  const batch = broadcastShapes(aBatch, bBatch);

  return [...batch, m, n];
}

function transposeShape(shape: number[], options: TransposeOptions): number[] {
  const { dim0, dim1 } = options;
  const result = shape.slice();
  const temp = result[dim0];
  result[dim0] = result[dim1];
  result[dim1] = temp;
  return result;
}

function reduceShape(
  shape: number[],
  dim: number | number[] | null | undefined,
  keepdim: boolean,
): number[] {
  if (dim == null) {
    // Reduce all dimensions
    return keepdim ? shape.map(() => 1) : [];
  }

  const dims = Array.isArray(dim) ? dim : [dim];
  const normalizedDims = dims.map((d) => (d < 0 ? shape.length + d : d));

  if (keepdim) {
    return shape.map((s, i) => (normalizedDims.includes(i) ? 1 : s));
  }

  return shape.filter((_, i) => !normalizedDims.includes(i));
}

function broadcastThreeShapes(a: number[], b: number[], c: number[]): number[] {
  const outRank = Math.max(a.length, b.length, c.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i++) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    const cDim = c[c.length - 1 - i] ?? 1;
    // Check all pairs for broadcast compatibility
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    }
    if (aDim !== cDim && aDim !== 1 && cDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    }
    if (bDim !== cDim && bDim !== 1 && cDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim, cDim);
  }
  return out;
}
