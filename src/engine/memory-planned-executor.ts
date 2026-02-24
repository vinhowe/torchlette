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
import type { ExecutionPlan, LazyRef, StorageHandle } from "./lazy";
import {
  canSafelyRelease,
  createStorageHandle,
  pretunePlanMatmuls,
  releaseBufferImmediate,
} from "./lazy";
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
 * Get input storage handle from a lazy ref.
 */
function getInputStorage(ref: LazyRef): StorageHandle {
  if (ref.kind === "materialized") {
    return ref.storage;
  }
  if (ref.node.result) {
    return ref.node.result;
  }
  throw new Error("Input not ready");
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

    // Execute the operation
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

      case "arange": {
        const ap = node.payload as { end: number; start: number; step: number };
        if (nodeBackend.ops.arange) {
          resultTensor = nodeBackend.ops.arange(ap.end, ap.start, ap.step);
        } else {
          const n = Math.max(0, Math.ceil((ap.end - ap.start) / ap.step));
          const vals = new Array(n);
          for (let i = 0; i < n; i++) vals[i] = ap.start + i * ap.step;
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
        resultTensor = nodeBackend.ops.add(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "sub": {
        const subPayload = node.payload as { alpha?: number } | undefined;
        resultTensor = nodeBackend.ops.sub(
          backendInputs[0],
          backendInputs[1],
          { ...subPayload, ...donationOpts },
        );
        break;
      }

      case "mul":
        resultTensor = nodeBackend.ops.mul(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "div": {
        const divPayload = node.payload as
          | { roundingMode?: "trunc" | "floor" }
          | undefined;
        resultTensor = nodeBackend.ops.div(
          backendInputs[0],
          backendInputs[1],
          { ...divPayload, ...donationOpts },
        );
        break;
      }

      case "matmul":
        resultTensor = nodeBackend.ops.matmul(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "sqrt":
        resultTensor = nodeBackend.ops.sqrt(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "relu":
        resultTensor = nodeBackend.ops.relu(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "exp":
        if (!nodeBackend.ops.exp)
          throw new Error("exp not supported by backend");
        resultTensor = nodeBackend.ops.exp(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "log":
        if (!nodeBackend.ops.log)
          throw new Error("log not supported by backend");
        resultTensor = nodeBackend.ops.log(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "neg":
        if (!nodeBackend.ops.neg)
          throw new Error("neg not supported by backend");
        resultTensor = nodeBackend.ops.neg(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "abs":
        if (!nodeBackend.ops.abs)
          throw new Error("abs not supported by backend");
        resultTensor = nodeBackend.ops.abs(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "tanh":
        if (!nodeBackend.ops.tanh)
          throw new Error("tanh not supported by backend");
        resultTensor = nodeBackend.ops.tanh(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "sigmoid":
        if (!nodeBackend.ops.sigmoid)
          throw new Error("sigmoid not supported by backend");
        resultTensor = nodeBackend.ops.sigmoid(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "gelu":
        if (!nodeBackend.ops.gelu)
          throw new Error("gelu not supported by backend");
        resultTensor = nodeBackend.ops.gelu(
          backendInputs[0],
          donationOpts,
        );
        break;

      case "silu":
        if (!nodeBackend.ops.silu)
          throw new Error("silu not supported by backend");
        resultTensor = nodeBackend.ops.silu(
          backendInputs[0],
          donationOpts,
        );
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
        resultTensor = nodeBackend.ops.gt(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "lt":
        if (!nodeBackend.ops.lt) throw new Error("lt not supported by backend");
        resultTensor = nodeBackend.ops.lt(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "ge":
        if (!nodeBackend.ops.ge) throw new Error("ge not supported by backend");
        resultTensor = nodeBackend.ops.ge(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "le":
        if (!nodeBackend.ops.le) throw new Error("le not supported by backend");
        resultTensor = nodeBackend.ops.le(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "eq":
        if (!nodeBackend.ops.eq) throw new Error("eq not supported by backend");
        resultTensor = nodeBackend.ops.eq(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "ne":
        if (!nodeBackend.ops.ne) throw new Error("ne not supported by backend");
        resultTensor = nodeBackend.ops.ne(
          backendInputs[0],
          backendInputs[1],
          donationOpts,
        );
        break;

      case "where":
        resultTensor = nodeBackend.ops.where(
          backendInputs[0],
          backendInputs[1],
          backendInputs[2],
          donationOpts,
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

      case "transfer": {
        const sourceStorage = inputs[0];
        const targetDevice = node.device;
        const sourceDevice = sourceStorage.device;

        if (sourceDevice === targetDevice) {
          resultTensor = sourceStorage.backendTensor;
        } else {
          const targetBackend = getBackend(targetDevice);
          if (!targetBackend) {
            throw new Error(
              `Transfer failed: backend not available for ${targetDevice}`,
            );
          }

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
