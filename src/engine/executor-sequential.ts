import { getBackend } from "../backend/registry";
import type { Backend } from "../backend/types";
import {
  abortBatch,
  beginBatchExecution,
  beginSharedEncoder,
  endBatchExecution,
  endSharedEncoder,
  flushBufferPool,
  isBatchActive,
} from "../backend/webgpu";
import type {
  ExecutePlanOptions,
  ExecutionPlan,
  LazyIRNode,
  StorageHandle,
} from "./lazy-types";
import { analyzeLifetimes, type TensorLifetime } from "./lifetime-analysis";
import { wrapResultAsStorage } from "./node-factory";
import { executeOp, getInputStorage } from "./op-dispatch";
import {
  extractPlanMetadata,
  pretunePlanMatmuls,
  segmentPlanAtCheckpoints,
} from "./plan-builder";
import {
  canSafelyRelease,
  releaseBufferImmediate,
  releaseDeadTensors,
} from "./storage-tracker";

/**
 * Replace pending input refs with materialized storages from previous segments.
 * Clone-on-write: only copies the inputs array if a substitution is needed,
 * preserving plan immutability across steps.
 */
function materializeSegmentInputs(
  nodes: LazyIRNode[],
  materializedStorages: Map<number, StorageHandle>,
): void {
  for (const node of nodes) {
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
}

// ============================================================================
// Sequential Plan Execution
// ============================================================================

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
    } catch {
      /* runtime/tensor not available */
    }
    lifetimes = analyzeLifetimes(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeSizes,
    );
  }

  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    for (let step = 0; step < plan.nodes.length; step++) {
      const node = plan.nodes[step];

      // Skip nodes that already have results (from a prior plan execution within this step).
      if (node.result) continue;

      // For multi-device graphs, use the node's device backend
      const nodeBackend = getBackend(node.device) ?? backend;

      const inputs = node.inputs.map((ref) =>
        getInputStorage(ref, nodeBackend),
      );
      const backendInputs = inputs.map((s) => s.backendTensor);

      const resultTensor = await executeOp(node, backendInputs, nodeBackend);
      node.result = wrapResultAsStorage(
        node.device,
        resultTensor,
        backendInputs,
        inputs,
      );

      // Track storage for early release
      if (options?.enableEarlyRelease) {
        nodeToStorage.set(node.id, node.result);
        releaseDeadTensors(
          lifetimes,
          step + 1,
          outputNodeIds,
          alreadyReleased,
          nodeToStorage,
        );
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

/** Options for segmented plan execution at checkpoint boundaries. */
interface SegmentedPlanOptions extends ExecutePlanOptions {
  /**
   * When true, uses GPU synchronization between segments: batches all ops
   * into a single command buffer, submits and waits for GPU completion,
   * then frees dead buffers before the next segment. Enables running models
   * that wouldn't fit in GPU memory.
   *
   * When false (default), just flushes the buffer pool between segments
   * without GPU sync — lighter weight but doesn't actually free GPU memory.
   */
  gpuSync?: boolean;
  /** Callback to flush the buffer pool between segments. Required when gpuSync is false. */
  flushBufferPoolFn?: () => void;
}

/**
 * Execute a plan with segmentation at checkpoint boundaries.
 *
 * Splits the plan at checkpoint boundaries and executes each segment,
 * freeing buffers between segments for memory savings.
 */
export async function executePlanSegmented(
  plan: ExecutionPlan,
  backend: Backend,
  options?: SegmentedPlanOptions,
): Promise<StorageHandle> {
  // Check if plan has any checkpoint boundaries
  const hasCheckpointBoundaries = plan.nodes.some(
    (n) => n.isCheckpointBoundary,
  );

  if (!hasCheckpointBoundaries) {
    return executePlan(plan, backend, options);
  }

  const segments = segmentPlanAtCheckpoints(plan);

  if (segments.length === 1) {
    return executePlan(plan, backend, options);
  }

  const gpuSync = options?.gpuSync ?? false;
  const materializedStorages = new Map<number, StorageHandle>();
  let lastResult: StorageHandle | null = null;
  const finalOutputId = gpuSync ? plan.nodes[plan.nodes.length - 1].id : 0;

  for (let segIdx = 0; segIdx < segments.length; segIdx++) {
    const segment = segments[segIdx];
    const isLastSegment = segIdx === segments.length - 1;

    materializeSegmentInputs(segment.nodes, materializedStorages);

    if (gpuSync) {
      // GPU sync path: batch ops, submit, wait, then release dead buffers
      const survivingNodeIds = findSurvivingOutputs(
        segment,
        segments.slice(segIdx + 1),
        finalOutputId,
      );

      beginBatchExecution();

      try {
        const nodeToStorage = new Map<number, StorageHandle>();

        for (const node of segment.nodes) {
          const nodeBackend = getBackend(node.device) ?? backend;
          const inputs = node.inputs.map((ref) =>
            getInputStorage(ref, nodeBackend),
          );
          const backendInputs = inputs.map((s) => s.backendTensor);

          const resultTensor = await executeOp(
            node,
            backendInputs,
            nodeBackend,
          );
          node.result = wrapResultAsStorage(
            node.device,
            resultTensor,
            backendInputs,
            inputs,
          );
          nodeToStorage.set(node.id, node.result);
          materializedStorages.set(node.id, node.result);
        }

        await endBatchExecution();

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
          flushBufferPool();
        }

        const lastNode = segment.nodes[segment.nodes.length - 1];
        lastResult = lastNode.result ?? undefined;
      } catch (error) {
        if (isBatchActive()) {
          abortBatch();
        }
        throw error;
      }
    } else {
      // Lightweight path: execute via executePlan, flush pool between segments
      lastResult = await executePlan(segment, backend, options);

      for (const node of segment.nodes) {
        if (node.result) {
          materializedStorages.set(node.id, node.result);
        }
      }

      if (!isLastSegment && options?.flushBufferPoolFn) {
        options.flushBufferPoolFn();
      }
    }
  }

  if (!lastResult) {
    throw new Error("Segmented execution failed: no result");
  }

  return lastResult;
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * Find node IDs that must survive this segment (used by later segments).
 */
function findSurvivingOutputs(
  _segment: ExecutionPlan,
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
