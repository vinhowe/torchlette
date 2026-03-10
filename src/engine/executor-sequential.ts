import { getBackend } from "../backend/registry";
import type { Backend } from "../backend/types";
import { isFusedBackend } from "../backend/types";
import type {
  ExecutePlanOptions,
  ExecutionPlan,
  LazyIRNode,
  StorageHandle,
} from "./lazy-types";
import type { TensorLifetime } from "./lifetime-analysis";
import { wrapResultAsStorage } from "./node-factory";
import { executeOp, getInputStorage } from "./op-dispatch";
import {
  initLifetimeAnalysis,
  pretunePlanMatmuls,
  segmentPlanAtCheckpoints,
} from "./plan-builder";
import {
  canSafelyRelease,
  releaseBufferImmediate,
  releaseDeadTensors,
} from "./storage-tracker";

// ============================================================================
// Sequential Plan Execution
// ============================================================================

/** Execute a single node: resolve inputs → dispatch op → store result. */
export async function executeNode(
  node: LazyIRNode,
  backend: Backend,
): Promise<void> {
  const nodeBackend = getBackend(node.device) ?? backend;
  const inputs = node.inputs.map((ref) => getInputStorage(ref, nodeBackend));
  const backendInputs = inputs.map((s) => s.backendTensor);
  const resultTensor = await executeOp(node, backendInputs, nodeBackend);
  node.result = wrapResultAsStorage(
    node.device,
    resultTensor,
    backendInputs,
    inputs,
  );
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
    // Resolve externally-referenced nodes (saved for backward, user-held tensors)
    let externalNodeIds: Set<number> | undefined;
    try {
      const { getPendingNodeIds } = await import("../runtime/tensor");
      const pending = getPendingNodeIds();
      if (pending.size > 0) externalNodeIds = pending;
    } catch {
      /* runtime/tensor not available */
    }
    ({ lifetimes, outputNodeIds } = initLifetimeAnalysis(
      plan.nodes,
      externalNodeIds,
    ));
  }

  const fused = isFusedBackend(backend) ? backend : null;
  if (fused) fused.beginSharedEncoder();

  try {
    for (let step = 0; step < plan.nodes.length; step++) {
      const node = plan.nodes[step];

      // Skip nodes that already have results (from a prior plan execution within this step).
      if (node.result) continue;

      await executeNode(node, backend);

      // Track storage for early release
      if (options?.enableEarlyRelease) {
        nodeToStorage.set(node.id, node.result!);
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
    if (fused) fused.endSharedEncoder();
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

    // Replace pending input refs with materialized storages from previous segments
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

    if (gpuSync) {
      // GPU sync path: batch ops, submit, wait, then release dead buffers
      // Find node IDs that must survive this segment (used by later segments)
      const survivingNodeIds = new Set<number>([finalOutputId]);
      for (const laterSegment of segments.slice(segIdx + 1)) {
        for (const laterNode of laterSegment.nodes) {
          for (const input of laterNode.inputs) {
            if (input.kind === "pending") {
              survivingNodeIds.add(input.node.id);
            }
          }
        }
      }

      const fusedBe = isFusedBackend(backend) ? backend : null;
      fusedBe?.beginBatchExecution();

      try {
        const nodeToStorage = new Map<number, StorageHandle>();

        for (const node of segment.nodes) {
          await executeNode(node, backend);
          nodeToStorage.set(node.id, node.result!);
          materializedStorages.set(node.id, node.result!);
        }

        await fusedBe?.endBatchExecution();

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
          fusedBe?.flushBufferPool();
        }

        const lastNode = segment.nodes[segment.nodes.length - 1];
        lastResult = lastNode.result ?? null;
      } catch (error) {
        if (fusedBe?.isBatchActive()) {
          fusedBe.abortBatch();
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
