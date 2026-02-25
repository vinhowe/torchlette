/**
 * Memory-Planned Executor
 *
 * Executes plans with memory planning integration:
 * - Lifetime analysis for buffer reuse
 * - Buffer donation (reuse input buffers for outputs)
 * - Buffer pooling with fence-based release
 * - In-flight plan strong rooting (§14)
 */

import { getBackend } from "../backend/registry";
import type {
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
} from "../backend/types";
import type { ExecutionPlan, StorageHandle } from "./lazy";
import {
  canSafelyRelease,
  createStorageHandle,
  pretunePlanMatmuls,
  releaseBufferImmediate,
} from "./lazy";
import { executeOp, getInputStorage } from "./op-dispatch";
import {
  analyzeLifetimes,
  computeBufferSize,
  findDeadTensorsAtStep,
  findDonationOpportunities,
  MemoryLimitExceededError,
  type MemoryPlan,
  MemoryPlanner,
  type PlanId,
  type TensorLifetime,
} from "./memory-planning";

// Re-export for convenience
export { MemoryLimitExceededError };

// WebGPU donation functions (registered by initWebGPU)
let webgpuDonateBuffer: ((tensor: BackendTensor) => unknown) | null = null;
let webgpuGetBufferSize: ((tensor: BackendTensor) => number) | null = null;

/**
 * Register WebGPU donation functions.
 * Called by initWebGPU() to enable donation for WebGPU tensors.
 */
export function registerWebGPUDonation(
  donateBuffer: (tensor: BackendTensor) => unknown,
  getBufferSize: (tensor: BackendTensor) => number,
): void {
  webgpuDonateBuffer = donateBuffer;
  webgpuGetBufferSize = getBufferSize;
}

/**
 * Try to donate a buffer from a tensor for reuse.
 * Returns the buffer if donation succeeded, null otherwise.
 * Only works for WebGPU tensors with compatible buffers.
 */
function tryDonateBuffer(
  tensor: BackendTensor,
  device: DeviceKind,
): unknown | null {
  if (device !== "webgpu") {
    return null;
  }

  // WebGPU donation functions must be loaded synchronously
  // Use the globally registered functions from initWebGPUDonation()
  if (webgpuDonateBuffer === null) {
    return null;
  }

  return webgpuDonateBuffer(tensor);
}

/**
 * Global memory planner instance.
 */
let globalMemoryPlanner: MemoryPlanner | null = null;

/**
 * Default memory limit: 10GB
 */
const DEFAULT_MEMORY_LIMIT_BYTES = 10 * 1024 * 1024 * 1024;

/**
 * Current memory limit configuration.
 */
let memoryLimitConfig = DEFAULT_MEMORY_LIMIT_BYTES;

/**
 * Configure the memory limit for the global memory planner.
 * This must be called before any memory planning operations, or
 * call resetMemoryPlanner() first to apply the new limit.
 */
export function setMemoryLimit(limitBytes: number): void {
  if (limitBytes <= 0) {
    throw new Error("Memory limit must be positive");
  }
  memoryLimitConfig = limitBytes;
  if (globalMemoryPlanner) {
    globalMemoryPlanner.setMemoryLimit(limitBytes);
  }
}

/**
 * Get the current memory limit.
 */
export function getMemoryLimit(): number {
  return memoryLimitConfig;
}

/**
 * Get or create the global memory planner.
 */
export function getMemoryPlanner(): MemoryPlanner {
  if (!globalMemoryPlanner) {
    globalMemoryPlanner = new MemoryPlanner({
      memoryLimitBytes: memoryLimitConfig,
    });
  }
  return globalMemoryPlanner;
}

/**
 * Reset the global memory planner (for testing).
 */
export function resetMemoryPlanner(): void {
  if (globalMemoryPlanner) {
    globalMemoryPlanner.clear();
  }
  globalMemoryPlanner = null;
}

/**
 * Execution statistics from memory-planned execution.
 */
export interface MemoryPlanningStats {
  totalNodes: number;
  totalAllocatedBytes: number;
  reusedBytes: number;
  newAllocations: number;
  reusedAllocations: number;
  donationCount: number;
}

/**
 * Result of memory-planned execution.
 */
export interface MemoryPlannedResult {
  result: StorageHandle;
  stats: MemoryPlanningStats;
  planId: PlanId;
}

/**
 * Extract execution metadata from a plan for memory planning.
 */
function extractPlanMetadata(plan: ExecutionPlan): {
  nodeOrder: number[];
  nodeInputs: Map<number, number[]>;
  nodeShapes: Map<number, number[]>;
  nodeDtypes: Map<number, DType>;
  outputNodeId: number;
} {
  const nodeOrder: number[] = [];
  const nodeInputs = new Map<number, number[]>();
  const nodeShapes = new Map<number, number[]>();
  const nodeDtypes = new Map<number, DType>();

  for (const node of plan.nodes) {
    nodeOrder.push(node.id);
    nodeShapes.set(node.id, node.shape);
    nodeDtypes.set(node.id, node.dtype);

    // Extract input node IDs (from pending refs)
    const inputIds: number[] = [];
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        inputIds.push(input.node.id);
      }
    }
    nodeInputs.set(node.id, inputIds);
  }

  const outputNodeId = plan.nodes[plan.nodes.length - 1].id;

  return {
    nodeOrder,
    nodeInputs,
    nodeShapes,
    nodeDtypes,
    outputNodeId,
  };
}

/**
 * Create a memory plan for an execution plan.
 */
export function createMemoryPlanForExecution(plan: ExecutionPlan): {
  memoryPlan: MemoryPlan;
  lifetimes: Map<number, TensorLifetime>;
  donations: Map<number, number>;
} {
  const planner = getMemoryPlanner();
  const { nodeOrder, nodeInputs, nodeShapes, nodeDtypes, outputNodeId } =
    extractPlanMetadata(plan);

  const memoryPlan = planner.planExecution(
    nodeOrder,
    nodeInputs,
    [outputNodeId],
    nodeShapes,
    nodeDtypes,
  );

  // Also compute lifetimes and donations for stats
  const nodeSizes = new Map<number, number>();
  for (const nodeId of nodeOrder) {
    const shape = nodeShapes.get(nodeId) ?? [1];
    const dtype = nodeDtypes.get(nodeId) ?? "f32";
    nodeSizes.set(nodeId, computeBufferSize(shape, dtype));
  }

  const outputSet = new Set([outputNodeId]);
  const lifetimes = analyzeLifetimes(
    nodeOrder,
    nodeInputs,
    outputSet,
    nodeSizes,
  );
  const donations = findDonationOpportunities(nodeOrder, lifetimes, nodeSizes);

  return { memoryPlan, lifetimes, donations };
}

/**
 * Execute a plan with memory planning.
 *
 * This is an enhanced version of executePlan that:
 * 1. Analyzes tensor lifetimes
 * 2. Finds donation opportunities
 * 3. Tracks buffer allocations
 * 4. Registers the plan for in-flight tracking
 * 5. Optionally releases buffers early during execution
 */
export async function executeWithMemoryPlanning(
  plan: ExecutionPlan,
  backend: Backend,
  options?: {
    enableDonation?: boolean;
    trackStats?: boolean;
    enableEarlyRelease?: boolean;
  },
): Promise<MemoryPlannedResult> {
  const enableDonation = options?.enableDonation ?? true;
  const trackStats = options?.trackStats ?? true;
  const enableEarlyRelease = options?.enableEarlyRelease ?? false;

  if (plan.nodes.length === 0) {
    throw new Error("Cannot execute empty plan");
  }

  // Pre-tune matmul shapes if backend supports it
  await pretunePlanMatmuls(plan, backend);

  const planner = getMemoryPlanner();
  const { memoryPlan, lifetimes, donations } =
    createMemoryPlanForExecution(plan);

  // Register this execution for in-flight tracking
  const { nodeOrder, nodeInputs, nodeShapes, nodeDtypes, outputNodeId } =
    extractPlanMetadata(plan);
  const planId = planner.registerExecution(memoryPlan, [outputNodeId]);

  // Track which node results are available for donation
  const nodeResults = new Map<number, StorageHandle>();
  const donatedBuffers = new Set<number>(); // nodeIds whose buffers were donated

  // Set up early release tracking
  const outputNodeIds = new Set([outputNodeId]);
  const alreadyReleased = new Set<number>();

  // Execute nodes in order
  for (let step = 0; step < plan.nodes.length; step++) {
    const node = plan.nodes[step];
    const inputs = node.inputs.map(getInputStorage);
    const backendInputs = inputs.map((s) => s.backendTensor);

    // For multi-device graphs, use the node's device backend
    const nodeBackend = getBackend(node.device) ?? backend;

    let resultTensor: BackendTensor;

    // Check if we can donate a buffer for this node's output
    let donatedBuffer: unknown | null = null;
    if (enableDonation) {
      const donorNodeId = donations.get(node.id);
      if (donorNodeId !== undefined && !donatedBuffers.has(donorNodeId)) {
        const donorResult = nodeResults.get(donorNodeId);
        if (donorResult) {
          // Try to donate the buffer
          donatedBuffer = tryDonateBuffer(
            donorResult.backendTensor,
            node.device,
          );
          if (donatedBuffer) {
            // Mark as donated so we don't reuse it again
            donatedBuffers.add(donorNodeId);
          }
        }
      }
    }

    // Options for ops that support output buffer donation
    const donationOpts = donatedBuffer
      ? { outBuffer: donatedBuffer }
      : undefined;

    // Execute the operation via canonical dispatch (with donation support)
    resultTensor = await executeOp(node, backendInputs, nodeBackend, donationOpts);

    // Safety: if a backend op returned the exact same tensor object as one of
    // its inputs (e.g. contiguous on an already-contiguous tensor), creating a
    // separate owning StorageHandle would double-free the underlying buffer.
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
    const storage = createStorageHandle(node.device, resultTensor, baseStorageId);
    node.result = storage;
    nodeResults.set(node.id, storage);

    // Release dead buffers after each step if early release is enabled
    if (enableEarlyRelease) {
      const deadNodeIds = findDeadTensorsAtStep(
        lifetimes,
        step + 1, // We just completed this step
        outputNodeIds,
        alreadyReleased,
      );
      for (const deadId of deadNodeIds) {
        const deadStorage = nodeResults.get(deadId);
        if (deadStorage && canSafelyRelease(deadStorage, nodeResults)) {
          releaseBufferImmediate(deadStorage);
          nodeResults.delete(deadId);
          alreadyReleased.add(deadId);
        }
      }
    }
  }

  // Complete the execution tracking
  // For WebGPU, we'd create a fence here to track GPU completion
  planner.completeExecution(planId);

  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Execution failed: no result for last node");
  }

  // Note: Intermediate buffer cleanup is handled by tensor disposal via onDispose callbacks.
  nodeResults.clear();

  const stats: MemoryPlanningStats = {
    totalNodes: plan.nodes.length,
    totalAllocatedBytes: memoryPlan.totalAllocatedBytes,
    reusedBytes: memoryPlan.reusedBytes,
    newAllocations: memoryPlan.newAllocations,
    reusedAllocations: memoryPlan.reusedAllocations,
    donationCount: donatedBuffers.size,
  };

  return {
    result: lastNode.result,
    stats,
    planId,
  };
}

/**
 * Get memory planner statistics.
 */
export function getMemoryPlannerStats(): {
  bufferPool: {
    totalBuffers: number;
    pooledBuffers: number;
    inUseBuffers: number;
    pendingFenceBuffers: number;
    totalBytes: number;
    pooledBytes: number;
  };
  planManager: {
    totalPlans: number;
    activePlans: number;
    completedPlans: number;
  };
} {
  return getMemoryPlanner().stats();
}
