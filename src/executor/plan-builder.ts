import type { Backend } from "../backend/types";
import { sizeOf } from "../core/shape";
import type { ExecutionPlan, LazyIRNode, LazyRef } from "../graph/types";
import { isFusibleOp } from "../compiler/fusion-detect";
import { analyzeLifetimes, type TensorLifetime } from "../graph/lifetime-analysis";
import { NumMinHeap } from "../graph/num-min-heap";

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
export function segmentPlanAtCheckpoints(plan: ExecutionPlan): ExecutionPlan[] {
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

/**
 * Build an execution plan from a single root node.
 */
export function buildPlan(root: LazyIRNode): ExecutionPlan {
  return buildMergedPlan([root]);
}

/**
 * Build an execution plan from multiple root nodes.
 * Used to merge recomputations from multiple checkpoint regions into a single plan.
 *
 * Uses object identity for visited tracking instead of numeric IDs.
 * Node IDs can collide when resetNodeIdCounter() is called between
 * API instances (e.g., in tests), but lingering pending tensors from
 * a previous instance still reference old node objects.
 *
 * @param roots - Array of LazyIRNode roots to include in the plan
 * @param skipExecuted - Skip nodes that already have results from a previous force call
 * @returns A single ExecutionPlan containing all nodes from all roots
 */
export function buildMergedPlan(
  roots: LazyIRNode[],
  skipExecuted = false,
): ExecutionPlan {
  const nodes: LazyIRNode[] = [];
  const visited = new Set<LazyIRNode>();

  // Iterative DFS postorder (was recursive `visit`): whole-step graphs reach
  // 5–15k nodes with dependency chains thousands deep, which overflows the JS
  // call stack. An explicit work-stack of {node, next-input-index} frames emits
  // each node after all its inputs — byte-identical order to the old recursion
  // (same input-index traversal, same shared `visited`, same root order).
  const shouldVisit = (ref: LazyRef): boolean => {
    if (ref.kind !== "pending") return false;
    if (skipExecuted && (ref.node.result || ref.node._executed)) return false;
    return !visited.has(ref.node);
  };
  const stack: Array<{ node: LazyIRNode; i: number }> = [];
  for (const root of roots) {
    if (!shouldVisit({ kind: "pending", node: root })) continue;
    visited.add(root);
    stack.push({ node: root, i: 0 });
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      if (frame.i < frame.node.inputs.length) {
        const input = frame.node.inputs[frame.i++];
        if (shouldVisit(input)) {
          // input is pending here (shouldVisit gated it).
          const child = (input as { node: LazyIRNode }).node;
          visited.add(child);
          stack.push({ node: child, i: 0 });
        }
      } else {
        nodes.push(frame.node);
        stack.pop();
      }
    }
  }

  return { nodes: enforceWriteAfterReadOrder(nodes) };
}

/** In-place ops and the input positions whose refs they OVERWRITE. */
const IN_PLACE_DST_INPUTS: Record<string, number[]> = {
  stridedScatterCopy: [0],
  stridedScatterAdd: [0],
  // adamStep updates param/m/v (inputs 1..3) in place; grad (input 0) is read-only.
  adamStep: [1, 2, 3],
};

/**
 * WAR (write-after-read) ordering: an in-place node that OVERWRITES the
 * buffer behind ref R must execute after every other plan node that READS R.
 * DFS postorder guarantees def-before-use but says nothing about sibling
 * readers vs an in-place writer of the same ref — the schedule was only
 * correct by traversal luck (e.g. zeroGrad's zero_ scatters racing the
 * optimizer's grad readers in the same plan; foreach's packed staging lost
 * that race deterministically). Kahn's algorithm over the normal dependency
 * edges PLUS anti-dependency edges (reader → in-place writer), tie-broken
 * by original position so conflict-free plans keep their exact order
 * (template/compiled-plan stability).
 */
export function enforceWriteAfterReadOrder(nodes: LazyIRNode[]): LazyIRNode[] {
  // Fast path: no in-place writers, keep the array untouched.
  let hasInPlace = false;
  for (const n of nodes) {
    if (IN_PLACE_DST_INPUTS[n.op]) {
      hasInPlace = true;
      break;
    }
  }
  if (!hasInPlace) return nodes;

  const indexOf = new Map<LazyIRNode, number>();
  for (let i = 0; i < nodes.length; i++) indexOf.set(nodes[i], i);

  // readers per ref key (producer node object + outputIndex)
  const refKey = (ref: { node: LazyIRNode; outputIndex?: number }) => ref.node;
  const readers = new Map<LazyIRNode, Map<number, number[]>>(); // prod → oi → reader idxs
  for (let i = 0; i < nodes.length; i++) {
    for (const ref of nodes[i].inputs) {
      if (ref.kind !== "pending") continue;
      const prod = refKey(ref);
      let byOi = readers.get(prod);
      if (!byOi) {
        byOi = new Map();
        readers.set(prod, byOi);
      }
      const oi = ref.outputIndex ?? 0;
      let list = byOi.get(oi);
      if (!list) {
        byOi.set(oi, (list = []));
      }
      list.push(i);
    }
  }

  // adjacency: normal edges (producer → consumer) + WAR edges (reader → writer)
  const succ: number[][] = nodes.map(() => []);
  const indeg = new Array<number>(nodes.length).fill(0);
  const addEdge = (from: number, to: number) => {
    succ[from].push(to);
    indeg[to]++;
  };
  for (let i = 0; i < nodes.length; i++) {
    for (const ref of nodes[i].inputs) {
      if (ref.kind !== "pending") continue;
      const pi = indexOf.get(ref.node);
      if (pi !== undefined) addEdge(pi, i);
    }
  }
  let warEdges = 0;
  // CHECKPOINT BOUNDARIES are order BARRIERS: segmented execution splits the
  // plan at these nodes (segmentPlanAtCheckpoints) and runs pool flushes
  // between segments — nodes must not migrate across a boundary (the
  // affinity tie-break otherwise regroups same-op nodes across segments,
  // breaking segment-local execution assumptions: "Input not ready" in
  // checkpoint mode). Constrain every node to its original side.
  //
  // The barrier is encoded with O(n) segment edges, not the old O(B·n)
  // all-pairs (every node before b → b, b → every node after): each node gets
  // one edge FROM the previous boundary and one edge TO the boundary ending its
  // segment. Boundaries chain (prev→next) through this same rule, so the
  // transitive order is identical to all-pairs — same partial order → same
  // deterministic Kahn output — at a fraction of the edges.
  const boundaryIdx: number[] = [];
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].isCheckpointBoundary) boundaryIdx.push(i);
  }
  if (boundaryIdx.length > 0) {
    let bp = 0; // first boundary at index >= i
    let prevBoundary = -1; // last boundary at index < i
    for (let i = 0; i < nodes.length; i++) {
      while (bp < boundaryIdx.length && boundaryIdx[bp] < i) {
        prevBoundary = boundaryIdx[bp];
        bp++;
      }
      const endBoundary = bp < boundaryIdx.length ? boundaryIdx[bp] : -1;
      if (prevBoundary >= 0) addEdge(prevBoundary, i);
      if (endBoundary >= 0 && endBoundary !== i) addEdge(i, endBoundary);
    }
  }
  for (let i = 0; i < nodes.length; i++) {
    const dstInputs = IN_PLACE_DST_INPUTS[nodes[i].op];
    if (!dstInputs) continue;
    for (const di of dstInputs) {
      const ref = nodes[i].inputs[di];
      if (!ref || ref.kind !== "pending") continue;
      const list = readers.get(ref.node)?.get(ref.outputIndex ?? 0);
      if (!list) continue;
      for (const ri of list) {
        if (ri !== i) {
          addEdge(ri, i);
          warEdges++;
        }
      }
    }
  }
  // NOTE: no early-return on warEdges === 0 — the Kahn pass also applies the
  // same-op AFFINITY tie-break (below), which regroups independent
  // same-kernel sequential dispatches (adamStep runs) even when no WAR
  // conflict exists. Order is preserved exactly wherever neither edges nor
  // affinity apply (min-original-index tie-break).

  // Kahn with min-original-index tie-break (deterministic, order-preserving
  // when constraints allow). Ready nodes live in a min-heap by index (was a
  // sorted-array `splice`, O(ready) per op → O(n²) on the whole-step graph's
  // O(n)-wide frontier: the backward/optimizer fan-out puts thousands of nodes
  // ready at once). A per-op min-heap serves the affinity tie-break in O(log n)
  // instead of the old O(ready) scan; both share an `emitted` flag for lazy
  // deletion of stale tops.
  const globalHeap = new NumMinHeap();
  const opHeaps = new Map<string, NumMinHeap>();
  const emitted = new Uint8Array(nodes.length);
  const pushReady = (i: number) => {
    globalHeap.push(i);
    const op = nodes[i].op;
    let oh = opHeaps.get(op);
    if (!oh) opHeaps.set(op, (oh = new NumMinHeap()));
    oh.push(i);
  };
  for (let i = 0; i < nodes.length; i++) if (indeg[i] === 0) pushReady(i);
  const order: LazyIRNode[] = [];
  let lastOp: string | null = null;
  let lastOpAffine = false;
  while (order.length < nodes.length) {
    // AFFINITY tie-break (generic same-op batching): when the last emitted
    // node is a NON-FUSIBLE op (sequential dispatch — adamStep, scatters,
    // unscaleGrad, ...) and another ready node has the same op, continue the
    // run. This keeps independent same-kernel dispatches adjacent so the
    // action builder's consecutive-run scans batch them — subsuming the old
    // op-name hoisting whitelist (ADAM_HOISTABLE_OPS) with a mechanism that
    // works for any op. Fusible ops keep strict original order: their
    // adjacency IS the fusion-reorder's chain layout, which must not be
    // perturbed. Deterministic: first (min-index) same-op ready node wins.
    let i = -1;
    if (lastOpAffine) {
      const oh = opHeaps.get(lastOp as string);
      if (oh) {
        while (oh.size > 0 && emitted[oh.peek()]) oh.pop();
        if (oh.size > 0) i = oh.pop();
      }
    }
    if (i === -1) {
      while (globalHeap.size > 0 && emitted[globalHeap.peek()]) globalHeap.pop();
      if (globalHeap.size === 0) break; // cycle — caught by length check below
      i = globalHeap.pop();
    }
    emitted[i] = 1;
    order.push(nodes[i]);
    lastOp = nodes[i].op;
    lastOpAffine = !isFusibleOp(lastOp);
    for (const j of succ[i]) {
      if (--indeg[j] === 0) pushReady(j);
    }
  }
  if (order.length !== nodes.length) {
    // A WAR edge formed a cycle (a reader of the old value that also depends
    // on the new value) — unsatisfiable; fall back to the original order and
    // say so rather than silently dropping nodes.
    console.warn(
      `[plan-builder] WAR ordering cycle (${nodes.length - order.length} nodes); keeping DFS order — in-place ops may race their readers`,
    );
    return nodes;
  }
  return order;
}

/**
 * Tag a plan with the indices of nodes whose results are needed by the caller.
 *
 * Called after `buildMergedPlan` to populate `plan.outputIndices` from the set
 * of node IDs that have live RuntimeTensors. The executor reads `outputIndices`
 * for liveness analysis and scoped result production; the serializer carries
 * it through to remote executors.
 */
export function tagPlanOutputs(
  plan: ExecutionPlan,
  liveNodeIds: Set<number>,
): void {
  plan.outputIndices = [];
  for (let i = 0; i < plan.nodes.length; i++) {
    if (liveNodeIds.has(plan.nodes[i].id)) plan.outputIndices.push(i);
  }
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
 * Initialize lifetime analysis for early buffer release.
 * Combines extractPlanMetadata + analyzeLifetimes with output node protection.
 */
export function initLifetimeAnalysis(
  planNodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
): { lifetimes: Map<number, TensorLifetime>; outputNodeIds: Set<number> } {
  const { nodeOrder, nodeInputs, nodeSizes } = extractPlanMetadata({
    nodes: planNodes,
  });
  const outputNodeIds = new Set([planNodes[planNodes.length - 1].id]);
  if (externalNodeIds) {
    for (const id of externalNodeIds) outputNodeIds.add(id);
  }
  const lifetimes = analyzeLifetimes(
    nodeOrder,
    nodeInputs,
    outputNodeIds,
    nodeSizes,
  );
  return { lifetimes, outputNodeIds };
}

/**
 * Extract matmul shapes from a plan for pre-tuning.
 * Returns array of [M, N, K] tuples.
 */
function extractMatmulShapes(
  plan: ExecutionPlan,
): Array<[number, number, number]> {
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
  const untunedShapes = shapes.filter((s) => {
    const sig = `${s[0]},${s[1]},${s[2]}`;
    return !pretunedShapeSignatures.has(sig);
  });
  if (untunedShapes.length === 0) return;

  await backend.pretuneMatmulShapes(untunedShapes);
  for (const s of untunedShapes) {
    pretunedShapeSignatures.add(`${s[0]},${s[1]},${s[2]}`);
  }
}
