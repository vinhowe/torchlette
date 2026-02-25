/**
 * Memory Planning System
 *
 * Implements:
 * - In-flight plan strong rooting (§14): Plans hold refs to all touched tensors
 * - Allocator fencing: Track GPU completion before buffer reuse
 * - Lifetime analysis: Determine when tensors can be freed
 * - Memory donation: Reuse input buffers for outputs when safe
 * - Buffer pooling: Pre-allocate and reuse buffers by size class
 */

import type { DType } from "../backend/types";
import { sizeOf } from "../core/shape";

// ============================================================================
// Core Types
// ============================================================================

/**
 * Unique identifier for a memory buffer.
 */
export type BufferId = number;

/**
 * Unique identifier for an in-flight execution plan.
 */
export type PlanId = number;

/**
 * Size class for buffer pooling.
 * Buffers are grouped by power-of-2 sizes for efficient reuse.
 */
export type SizeClass = number;

/**
 * Memory buffer metadata.
 */
export interface BufferInfo {
  id: BufferId;
  sizeBytes: number;
  sizeClass: SizeClass;
  dtype: DType;
  shape: number[];
  createdAt: number;
  lastUsedAt: number;
  inUseByPlan: PlanId | null;
  fenceId: number | null; // GPU fence for completion tracking
}

/**
 * In-flight plan with strong references to all buffers it uses.
 */
export interface InFlightPlan {
  id: PlanId;
  inputBuffers: BufferId[];
  outputBuffers: BufferId[];
  intermediateBuffers: BufferId[];
  startedAt: number;
  completedAt: number | null;
  fenceId: number | null;
}

/**
 * Donation decision for a buffer.
 */
export interface DonationDecision {
  canDonate: boolean;
  sourceBufferId: BufferId;
  targetNodeId: number;
  reason: string;
}

/**
 * Lifetime information for a tensor in an execution plan.
 */
export interface TensorLifetime {
  nodeId: number;
  firstUse: number; // Step index
  lastUse: number; // Step index
  isOutput: boolean;
  isInput: boolean;
  bufferSize: number;
}

// ============================================================================
// Size Class Utilities
// ============================================================================

/**
 * Minimum buffer size (256 bytes) to avoid tiny allocations.
 */
const MIN_BUFFER_SIZE = 256;

/**
 * Compute the size class for a given byte size.
 * Size class is the ceiling log2, giving power-of-2 buckets.
 */
export function getSizeClass(sizeBytes: number): SizeClass {
  const size = Math.max(sizeBytes, MIN_BUFFER_SIZE);
  return Math.ceil(Math.log2(size));
}

/**
 * Get the actual buffer size for a size class.
 */
export function getSizeForClass(sizeClass: SizeClass): number {
  return Math.pow(2, sizeClass);
}

/**
 * Compute buffer size in bytes for a tensor.
 */
export function computeBufferSize(shape: number[], dtype: DType): number {
  const elements = sizeOf(shape);
  const bytesPerElement =
    dtype === "f32" || dtype === "i32" ? 4 : dtype === "f16" ? 2 : 1;
  return elements * bytesPerElement;
}

// ============================================================================
// Lifetime Analysis
// ============================================================================

/**
 * Analyze tensor lifetimes in an execution plan.
 *
 * @param nodeOrder - Node IDs in execution order
 * @param nodeInputs - Map from node ID to its input node IDs
 * @param nodeOutputs - Set of node IDs that are plan outputs
 * @param nodeSizes - Map from node ID to buffer size
 */
export function analyzeLifetimes(
  nodeOrder: number[],
  nodeInputs: Map<number, number[]>,
  nodeOutputs: Set<number>,
  nodeSizes: Map<number, number>,
): Map<number, TensorLifetime> {
  const lifetimes = new Map<number, TensorLifetime>();
  const stepIndex = new Map<number, number>();

  // Build step index
  nodeOrder.forEach((nodeId, idx) => {
    stepIndex.set(nodeId, idx);
  });

  // Initialize lifetimes
  for (const nodeId of nodeOrder) {
    lifetimes.set(nodeId, {
      nodeId,
      firstUse: stepIndex.get(nodeId)!,
      lastUse: stepIndex.get(nodeId)!,
      isOutput: nodeOutputs.has(nodeId),
      isInput: !nodeInputs.has(nodeId) || nodeInputs.get(nodeId)!.length === 0,
      bufferSize: nodeSizes.get(nodeId) ?? 0,
    });
  }

  // Extend lifetimes based on uses
  for (const [nodeId, inputs] of nodeInputs) {
    const nodeStep = stepIndex.get(nodeId)!;
    for (const inputId of inputs) {
      const lifetime = lifetimes.get(inputId);
      if (lifetime && nodeStep > lifetime.lastUse) {
        lifetime.lastUse = nodeStep;
      }
    }
  }

  // Outputs live until end of plan
  const lastStep = nodeOrder.length - 1;
  for (const outputId of nodeOutputs) {
    const lifetime = lifetimes.get(outputId);
    if (lifetime) {
      lifetime.lastUse = lastStep;
    }
  }

  return lifetimes;
}

/**
 * Find tensors that are dead (no longer needed) at a given step.
 */
export function findDeadTensors(
  lifetimes: Map<number, TensorLifetime>,
  currentStep: number,
): number[] {
  const dead: number[] = [];
  for (const [nodeId, lifetime] of lifetimes) {
    if (lifetime.lastUse < currentStep && !lifetime.isOutput) {
      dead.push(nodeId);
    }
  }
  return dead;
}

/**
 * Find tensors that become dead at a specific step during execution.
 * Returns node IDs whose lastUse equals the previous step (currentStep - 1),
 * meaning they were last used in the previous step and are now dead.
 *
 * This is used for incremental buffer release during plan execution.
 *
 * @param lifetimes - Lifetime information for all tensors
 * @param currentStep - The step we just completed (0-indexed)
 * @param outputNodeIds - Set of node IDs that are plan outputs (never release these)
 * @param alreadyReleased - Set of node IDs already released (to avoid double-release)
 */
export function findDeadTensorsAtStep(
  lifetimes: Map<number, TensorLifetime>,
  currentStep: number,
  outputNodeIds: Set<number>,
  alreadyReleased: Set<number>,
): number[] {
  const dead: number[] = [];
  for (const [nodeId, lifetime] of lifetimes) {
    // Skip outputs - they must live until plan completion
    if (outputNodeIds.has(nodeId)) {
      continue;
    }
    // Skip already released
    if (alreadyReleased.has(nodeId)) {
      continue;
    }
    // A tensor is dead if its lastUse was before currentStep
    // (i.e., it was last used in a previous step)
    if (lifetime.lastUse < currentStep) {
      dead.push(nodeId);
    }
  }
  return dead;
}

// ============================================================================
// Donation Analysis
// ============================================================================

/**
 * Check if a buffer can be donated (reused) for a target operation.
 *
 * Donation is safe when:
 * 1. Source tensor is dead after this step
 * 2. Source buffer is large enough for target
 * 3. Source is not an output of the plan
 * 4. Dtypes are compatible (same size or target is smaller)
 */
export function canDonateBuffer(
  sourceLifetime: TensorLifetime,
  targetSize: number,
  currentStep: number,
): DonationDecision {
  const sourceBufferId = sourceLifetime.nodeId as BufferId;

  if (sourceLifetime.isOutput) {
    return {
      canDonate: false,
      sourceBufferId,
      targetNodeId: -1,
      reason: "Source is a plan output",
    };
  }

  if (sourceLifetime.lastUse > currentStep) {
    return {
      canDonate: false,
      sourceBufferId,
      targetNodeId: -1,
      reason: "Source is still alive",
    };
  }

  if (sourceLifetime.bufferSize < targetSize) {
    return {
      canDonate: false,
      sourceBufferId,
      targetNodeId: -1,
      reason: "Source buffer too small",
    };
  }

  return {
    canDonate: true,
    sourceBufferId,
    targetNodeId: -1,
    reason: "Eligible for donation",
  };
}

/**
 * Find all donation opportunities in a plan.
 */
export function findDonationOpportunities(
  nodeOrder: number[],
  lifetimes: Map<number, TensorLifetime>,
  nodeSizes: Map<number, number>,
): Map<number, number> {
  // Maps target node ID -> source node ID for donation
  const donations = new Map<number, number>();
  const usedForDonation = new Set<number>();

  for (let step = 0; step < nodeOrder.length; step++) {
    const nodeId = nodeOrder[step];
    const targetSize = nodeSizes.get(nodeId) ?? 0;

    if (targetSize === 0) continue;

    // Find dead tensors that could donate their buffer
    const deadTensors = findDeadTensors(lifetimes, step);

    // Sort by size (prefer exact match, then smallest sufficient)
    const candidates = deadTensors
      .filter((id) => !usedForDonation.has(id))
      .map((id) => ({
        id,
        lifetime: lifetimes.get(id)!,
      }))
      .filter((c) => c.lifetime.bufferSize >= targetSize)
      .sort((a, b) => a.lifetime.bufferSize - b.lifetime.bufferSize);

    if (candidates.length > 0) {
      const donor = candidates[0];
      donations.set(nodeId, donor.id);
      usedForDonation.add(donor.id);
    }
  }

  return donations;
}

// ============================================================================
// Buffer Pool
// ============================================================================

/**
 * Default memory limit: 10GB
 */
const DEFAULT_MEMORY_LIMIT_BYTES = 10 * 1024 * 1024 * 1024;

/**
 * Options for configuring the buffer pool.
 */
interface BufferPoolOptions {
  /**
   * Maximum memory limit in bytes.
   * Default: 10GB (10 * 1024 * 1024 * 1024)
   */
  memoryLimitBytes?: number;
}

/**
 * Error thrown when memory allocation would exceed the limit.
 */
export class MemoryLimitExceededError extends Error {
  constructor(
    public readonly requestedBytes: number,
    public readonly currentBytes: number,
    public readonly limitBytes: number,
  ) {
    super(
      `Memory limit exceeded: requested ${formatBytesForError(requestedBytes)}, ` +
        `current usage ${formatBytesForError(currentBytes)}, ` +
        `limit ${formatBytesForError(limitBytes)}`,
    );
    this.name = "MemoryLimitExceededError";
  }
}

function formatBytesForError(bytes: number): string {
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
 * A pool of reusable buffers organized by size class.
 */
export class BufferPool {
  private pools = new Map<SizeClass, BufferId[]>();
  private bufferInfo = new Map<BufferId, BufferInfo>();
  private nextBufferId = 1;
  private nextFenceId = 1;
  private memoryLimitBytes: number;
  private currentAllocatedBytes = 0;

  constructor(options?: BufferPoolOptions) {
    this.memoryLimitBytes =
      options?.memoryLimitBytes ?? DEFAULT_MEMORY_LIMIT_BYTES;
  }

  /**
   * Get the current memory limit in bytes.
   */
  getMemoryLimit(): number {
    return this.memoryLimitBytes;
  }

  /**
   * Set the memory limit in bytes.
   * Note: This does not free existing allocations if they exceed the new limit.
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
   * Allocate or reuse a buffer of the given size.
   * @throws MemoryLimitExceededError if allocation would exceed memory limit
   */
  allocate(sizeBytes: number, dtype: DType, shape: number[]): BufferInfo {
    const sizeClass = getSizeClass(sizeBytes);
    const pool = this.pools.get(sizeClass) ?? [];

    // Try to find a free buffer in the pool
    for (let i = 0; i < pool.length; i++) {
      const bufferId = pool[i];
      const info = this.bufferInfo.get(bufferId);
      if (info && info.inUseByPlan === null && info.fenceId === null) {
        // Reuse this buffer (no new memory allocated)
        pool.splice(i, 1);
        info.lastUsedAt = Date.now();
        info.dtype = dtype;
        info.shape = shape.slice();
        return info;
      }
    }

    // Need to allocate new buffer - check memory limit
    const actualSize = getSizeForClass(sizeClass);
    if (this.currentAllocatedBytes + actualSize > this.memoryLimitBytes) {
      throw new MemoryLimitExceededError(
        actualSize,
        this.currentAllocatedBytes,
        this.memoryLimitBytes,
      );
    }

    // Allocate new buffer
    const info: BufferInfo = {
      id: this.nextBufferId++,
      sizeBytes: actualSize,
      sizeClass,
      dtype,
      shape: shape.slice(),
      createdAt: Date.now(),
      lastUsedAt: Date.now(),
      inUseByPlan: null,
      fenceId: null,
    };
    this.bufferInfo.set(info.id, info);
    this.currentAllocatedBytes += actualSize;
    return info;
  }

  /**
   * Mark a buffer as in-use by a plan.
   */
  markInUse(bufferId: BufferId, planId: PlanId): void {
    const info = this.bufferInfo.get(bufferId);
    if (info) {
      info.inUseByPlan = planId;
      info.lastUsedAt = Date.now();
    }
  }

  /**
   * Mark a buffer as pending GPU fence.
   * If fenceId is provided, use it. Otherwise generate a new one.
   */
  markPendingFence(bufferId: BufferId, fenceId?: number): number {
    const info = this.bufferInfo.get(bufferId);
    if (info) {
      const actualFenceId = fenceId ?? this.nextFenceId++;
      info.fenceId = actualFenceId;
      info.inUseByPlan = null;
      return actualFenceId;
    }
    return -1;
  }

  /**
   * Signal that a fence has completed, making buffers available.
   */
  signalFence(fenceId: number): void {
    for (const [bufferId, info] of this.bufferInfo) {
      if (info.fenceId === fenceId) {
        info.fenceId = null;
        // Return to pool
        const pool = this.pools.get(info.sizeClass) ?? [];
        pool.push(bufferId);
        this.pools.set(info.sizeClass, pool);
      }
    }
  }

  /**
   * Release a buffer back to the pool (if no fence pending).
   */
  release(bufferId: BufferId): void {
    const info = this.bufferInfo.get(bufferId);
    if (info) {
      info.inUseByPlan = null;
      if (info.fenceId === null) {
        const pool = this.pools.get(info.sizeClass) ?? [];
        pool.push(bufferId);
        this.pools.set(info.sizeClass, pool);
      }
    }
  }

  /**
   * Get buffer info.
   */
  getInfo(bufferId: BufferId): BufferInfo | undefined {
    return this.bufferInfo.get(bufferId);
  }

  /**
   * Get pool statistics.
   */
  stats(): {
    totalBuffers: number;
    pooledBuffers: number;
    inUseBuffers: number;
    pendingFenceBuffers: number;
    totalBytes: number;
    pooledBytes: number;
    memoryLimitBytes: number;
    memoryUsagePercent: number;
  } {
    let pooledBuffers = 0;
    let pooledBytes = 0;
    for (const pool of this.pools.values()) {
      pooledBuffers += pool.length;
      for (const id of pool) {
        pooledBytes += this.bufferInfo.get(id)?.sizeBytes ?? 0;
      }
    }

    let inUseBuffers = 0;
    let pendingFenceBuffers = 0;
    let totalBytes = 0;
    for (const info of this.bufferInfo.values()) {
      totalBytes += info.sizeBytes;
      if (info.inUseByPlan !== null) inUseBuffers++;
      if (info.fenceId !== null) pendingFenceBuffers++;
    }

    return {
      totalBuffers: this.bufferInfo.size,
      pooledBuffers,
      inUseBuffers,
      pendingFenceBuffers,
      totalBytes,
      pooledBytes,
      memoryLimitBytes: this.memoryLimitBytes,
      memoryUsagePercent:
        (this.currentAllocatedBytes / this.memoryLimitBytes) * 100,
    };
  }

  /**
   * Clear all buffers (for testing or shutdown).
   */
  clear(): void {
    this.pools.clear();
    this.bufferInfo.clear();
    this.nextBufferId = 1;
    this.nextFenceId = 1;
    this.currentAllocatedBytes = 0;
  }
}

// ============================================================================
// In-Flight Plan Manager
// ============================================================================

/**
 * Manages in-flight execution plans and their buffer references.
 * Implements §14 strong rooting requirement.
 */
export class InFlightPlanManager {
  private plans = new Map<PlanId, InFlightPlan>();
  private nextPlanId = 1;
  private bufferPool: BufferPool;

  constructor(bufferPool: BufferPool) {
    this.bufferPool = bufferPool;
  }

  /**
   * Register a new in-flight plan with its buffer requirements.
   * This creates strong references to all buffers.
   */
  registerPlan(
    inputBuffers: BufferId[],
    outputBuffers: BufferId[],
    intermediateBuffers: BufferId[],
  ): PlanId {
    const planId = this.nextPlanId++;
    const plan: InFlightPlan = {
      id: planId,
      inputBuffers: inputBuffers.slice(),
      outputBuffers: outputBuffers.slice(),
      intermediateBuffers: intermediateBuffers.slice(),
      startedAt: Date.now(),
      completedAt: null,
      fenceId: null,
    };
    this.plans.set(planId, plan);

    // Mark all buffers as in-use by this plan
    for (const bufferId of [
      ...inputBuffers,
      ...outputBuffers,
      ...intermediateBuffers,
    ]) {
      this.bufferPool.markInUse(bufferId, planId);
    }

    return planId;
  }

  /**
   * Mark a plan as completed and optionally set a fence.
   */
  completePlan(planId: PlanId, fenceId?: number): void {
    const plan = this.plans.get(planId);
    if (!plan) return;

    plan.completedAt = Date.now();
    plan.fenceId = fenceId ?? null;

    // Release intermediate buffers (they're no longer needed)
    for (const bufferId of plan.intermediateBuffers) {
      if (fenceId !== undefined) {
        this.bufferPool.markPendingFence(bufferId, fenceId);
      } else {
        this.bufferPool.release(bufferId);
      }
    }

    // Input buffers can be released if not outputs
    const outputSet = new Set(plan.outputBuffers);
    for (const bufferId of plan.inputBuffers) {
      if (!outputSet.has(bufferId)) {
        if (fenceId !== undefined) {
          this.bufferPool.markPendingFence(bufferId, fenceId);
        } else {
          this.bufferPool.release(bufferId);
        }
      }
    }
  }

  /**
   * Signal fence completion, allowing buffer reuse.
   */
  signalFence(fenceId: number): void {
    this.bufferPool.signalFence(fenceId);
  }

  /**
   * Get a plan by ID.
   */
  getPlan(planId: PlanId): InFlightPlan | undefined {
    return this.plans.get(planId);
  }

  /**
   * Get all active (not completed) plans.
   */
  getActivePlans(): InFlightPlan[] {
    return Array.from(this.plans.values()).filter(
      (p) => p.completedAt === null,
    );
  }

  /**
   * Check if any buffer is still in use by any plan.
   */
  isBufferInUse(bufferId: BufferId): boolean {
    for (const plan of this.plans.values()) {
      if (plan.completedAt !== null) continue;
      if (
        plan.inputBuffers.includes(bufferId) ||
        plan.outputBuffers.includes(bufferId) ||
        plan.intermediateBuffers.includes(bufferId)
      ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Clean up completed plans older than maxAge.
   */
  cleanup(maxAgeMs = 60000): number {
    const now = Date.now();
    let cleaned = 0;
    for (const [planId, plan] of this.plans) {
      if (plan.completedAt !== null && now - plan.completedAt >= maxAgeMs) {
        this.plans.delete(planId);
        cleaned++;
      }
    }
    return cleaned;
  }

  /**
   * Get statistics.
   */
  stats(): {
    totalPlans: number;
    activePlans: number;
    completedPlans: number;
  } {
    let activePlans = 0;
    let completedPlans = 0;
    for (const plan of this.plans.values()) {
      if (plan.completedAt === null) {
        activePlans++;
      } else {
        completedPlans++;
      }
    }
    return {
      totalPlans: this.plans.size,
      activePlans,
      completedPlans,
    };
  }
}

// ============================================================================
// Memory Planner
// ============================================================================

/**
 * Memory allocation plan for an execution.
 */
export interface MemoryPlan {
  allocations: Map<number, BufferId>; // nodeId -> bufferId
  donations: Map<number, number>; // targetNodeId -> sourceNodeId
  totalAllocatedBytes: number;
  reusedBytes: number;
  newAllocations: number;
  reusedAllocations: number;
}

/**
 * Options for the memory planner.
 */
interface MemoryPlannerOptions {
  /**
   * Maximum memory limit in bytes.
   * Default: 10GB (10 * 1024 * 1024 * 1024)
   */
  memoryLimitBytes?: number;
}

/**
 * High-level memory planner that combines all features.
 */
export class MemoryPlanner {
  private bufferPool: BufferPool;
  private planManager: InFlightPlanManager;

  constructor(options?: MemoryPlannerOptions) {
    this.bufferPool = new BufferPool({
      memoryLimitBytes: options?.memoryLimitBytes,
    });
    this.planManager = new InFlightPlanManager(this.bufferPool);
  }

  /**
   * Get the current memory limit in bytes.
   */
  getMemoryLimit(): number {
    return this.bufferPool.getMemoryLimit();
  }

  /**
   * Set the memory limit in bytes.
   */
  setMemoryLimit(limitBytes: number): void {
    this.bufferPool.setMemoryLimit(limitBytes);
  }

  /**
   * Get the current total allocated memory in bytes.
   */
  getCurrentAllocatedBytes(): number {
    return this.bufferPool.getCurrentAllocatedBytes();
  }

  /**
   * Plan memory allocation for an execution.
   */
  planExecution(
    nodeOrder: number[],
    nodeInputs: Map<number, number[]>,
    outputNodeIds: number[],
    nodeShapes: Map<number, number[]>,
    nodeDtypes: Map<number, DType>,
  ): MemoryPlan {
    // Compute sizes
    const nodeSizes = new Map<number, number>();
    for (const nodeId of nodeOrder) {
      const shape = nodeShapes.get(nodeId) ?? [1];
      const dtype = nodeDtypes.get(nodeId) ?? "f32";
      nodeSizes.set(nodeId, computeBufferSize(shape, dtype));
    }

    // Analyze lifetimes
    const outputSet = new Set(outputNodeIds);
    const lifetimes = analyzeLifetimes(
      nodeOrder,
      nodeInputs,
      outputSet,
      nodeSizes,
    );

    // Find donation opportunities
    const donations = findDonationOpportunities(
      nodeOrder,
      lifetimes,
      nodeSizes,
    );

    // Allocate buffers
    const allocations = new Map<number, BufferId>();
    let totalAllocatedBytes = 0;
    let reusedBytes = 0;
    let newAllocations = 0;
    let reusedAllocations = 0;

    for (const nodeId of nodeOrder) {
      const shape = nodeShapes.get(nodeId) ?? [1];
      const dtype = nodeDtypes.get(nodeId) ?? "f32";
      const size = nodeSizes.get(nodeId) ?? 0;

      // Check if we can use a donated buffer
      const donorId = donations.get(nodeId);
      if (donorId !== undefined) {
        const donorBufferId = allocations.get(donorId);
        if (donorBufferId !== undefined) {
          allocations.set(nodeId, donorBufferId);
          reusedBytes += size;
          reusedAllocations++;
          continue;
        }
      }

      // Allocate from pool
      const bufferInfo = this.bufferPool.allocate(size, dtype, shape);
      allocations.set(nodeId, bufferInfo.id);
      totalAllocatedBytes += bufferInfo.sizeBytes;
      newAllocations++;
    }

    return {
      allocations,
      donations,
      totalAllocatedBytes,
      reusedBytes,
      newAllocations,
      reusedAllocations,
    };
  }

  /**
   * Register an execution plan (for in-flight tracking).
   */
  registerExecution(memoryPlan: MemoryPlan, outputNodeIds: number[]): PlanId {
    const allBufferIds = Array.from(new Set(memoryPlan.allocations.values()));
    const outputBufferIds = outputNodeIds
      .map((id) => memoryPlan.allocations.get(id))
      .filter((id): id is BufferId => id !== undefined);
    const intermediateBufferIds = allBufferIds.filter(
      (id) => !outputBufferIds.includes(id),
    );

    return this.planManager.registerPlan(
      [],
      outputBufferIds,
      intermediateBufferIds,
    );
  }

  /**
   * Complete an execution plan.
   */
  completeExecution(planId: PlanId, fenceId?: number): void {
    this.planManager.completePlan(planId, fenceId);
  }

  /**
   * Signal GPU fence completion.
   */
  signalFence(fenceId: number): void {
    this.planManager.signalFence(fenceId);
  }

  /**
   * Get buffer pool.
   */
  getBufferPool(): BufferPool {
    return this.bufferPool;
  }

  /**
   * Get plan manager.
   */
  getPlanManager(): InFlightPlanManager {
    return this.planManager;
  }

  /**
   * Get combined statistics.
   */
  stats(): {
    bufferPool: ReturnType<BufferPool["stats"]>;
    planManager: ReturnType<InFlightPlanManager["stats"]>;
  } {
    return {
      bufferPool: this.bufferPool.stats(),
      planManager: this.planManager.stats(),
    };
  }

  /**
   * Clear all state (for testing).
   */
  clear(): void {
    this.bufferPool.clear();
  }
}
