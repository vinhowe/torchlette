import type { Backend } from "../backend/types";
import type { ExecutionPlan, LazyIRNode, LazyRef } from "./lazy-types";
import { sizeOf } from "../core/shape";

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
    const numElements = sizeOf(node.shape);
    nodeSizes.set(node.id, numElements * bytesPerElement);
  }

  return { nodeOrder, nodeInputs, nodeSizes };
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
