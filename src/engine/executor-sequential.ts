import type { Backend } from "../backend/types";
import { getBackend } from "../backend/registry";
import {
  beginBatchExecution,
  endBatchExecution,
  isBatchActive,
  abortBatch,
  flushBufferPool,
  beginSharedEncoder,
  endSharedEncoder,
} from "../backend/webgpu";
import {
  analyzeLifetimes,
  type TensorLifetime,
} from "./memory-planning";
import type { LazyIRNode, StorageHandle, ExecutionPlan, ExecutePlanOptions } from "./lazy-types";
import { wrapResultAsStorage } from "./node-factory";
import { canSafelyRelease, releaseBufferImmediate, releaseDeadTensors } from "./storage-tracker";
import { extractPlanMetadata, pretunePlanMatmuls, segmentPlanAtCheckpoints } from "./plan-builder";
import { getInputStorage, executeOp } from "./op-dispatch";

// ============================================================================
// Constants
// ============================================================================

/** Ops safe to execute during tape replay fill-in: pure views + data sources. */
export const FILL_IN_OPS: ReadonlySet<string> = new Set([
  // Pure view ops (no GPU dispatch, same buffer)
  "reshape", "transpose", "permute", "expand", "narrow", "contiguous",
  // Data source ops (create new buffers from host data)
  "tensorFromArray", "zeros", "full", "arange",
]);

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

    const resultTensor = await executeOp(node, backendInputs, nodeBackend);
    node.result = wrapResultAsStorage(node.device, resultTensor, backendInputs, inputs);

    // Track storage for early release
    if (options?.enableEarlyRelease) {
      nodeToStorage.set(node.id, node.result);
      releaseDeadTensors(lifetimes, step + 1, outputNodeIds, alreadyReleased, nodeToStorage);
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

        const resultTensor = await executeOp(
          node,
          backendInputs,
          nodeBackend,
        );

        node.result = wrapResultAsStorage(node.device, resultTensor, backendInputs, inputs);
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

// ============================================================================
// Helpers
// ============================================================================

/**
 * Find node IDs that must survive this segment (used by later segments).
 */
export function findSurvivingOutputs(
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
