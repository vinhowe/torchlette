/**
 * Row-Program Detection
 *
 * Detects multi-reduction subgraphs eligible for row-program fusion.
 * A row program fuses elementwise ops + multiple reductions along the same
 * last dim into a single perRowKernel dispatch.
 *
 * Replaces the hardcoded compound pattern detection for softmax/log_softmax
 * with a general-purpose system.
 *
 * Detection algorithm:
 * 1. For each unclaimed last-dim reduction, expand backward/forward
 *    collecting elementwise ops and additional same-dim reductions
 * 2. Validate constraints (shapes, consumers, bindings)
 * 3. Build RowProgram IR with expression trees per phase
 */

import type { DType } from "../backend/types";
import type { LazyIRNode, LazyRef } from "../graph/types";
import { isFusibleOp } from "./fusion-detect";
import type {
  RowProgram,
  RowProgramMatch,
  RPExpr,
  RPPhase,
} from "./row-program-types";
import { isRPValue } from "./row-program-types";

/**
 * Whether an expression reads any per-element input buffer (vs. only reduction
 * results and constants). A write expression with NO per-element input produces
 * one value PER ROW (reduced shape [...,1]) — e.g. `rsqrt(mean(xc²)+eps)`. Such
 * outputs MUST use the scalar-output write path; emitting them full-width writes
 * an [R,D] buffer that a downstream [R,1]-shaped consumer misreads (every row
 * collapses onto row 0's block). See tools/compile-ln-repro.ts.
 */
function exprReadsInput(expr: RPExpr): boolean {
  if (isRPValue(expr)) return expr.kind === "input";
  return expr.inputs.some(exprReadsInput);
}

// ============================================================================
// Helpers
// ============================================================================

/** Reduction ops that can seed a row program. */
const REDUCE_OPS = new Set(["sum", "mean", "max"]);

/** Max unique input bindings (WebGPU limit 8 minus 1 for output). */
const MAX_INPUT_BINDINGS = 7;

/** Max expression tree nodes per phase (prevent WGSL explosion). */
const MAX_EXPR_NODES = 30;

/** Extract normalized reduction dim from a reduction node's payload. */
function getReductionDim(node: LazyIRNode): number | null {
  const payload = node.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;
  if (!payload) return null;
  const dim = payload.dim;
  if (dim == null) return null; // full reduction — not a row program
  if (Array.isArray(dim)) {
    if (dim.length !== 1) return null; // multi-dim reduction — too complex
    return normalizeDim(
      dim[0],
      node.inputs[0]?.kind === "pending"
        ? node.inputs[0].node.shape.length
        : node.shape.length,
    );
  }
  const rank =
    node.inputs[0]?.kind === "pending"
      ? node.inputs[0].node.shape.length
      : node.shape.length;
  return normalizeDim(dim, rank);
}

function normalizeDim(dim: number, rank: number): number {
  return dim < 0 ? dim + rank : dim;
}

/** Check if a reduction is along the last dimension. */
function isLastDimReduction(node: LazyIRNode): boolean {
  const dim = getReductionDim(node);
  if (dim === null) return false;
  const rank =
    node.inputs[0]?.kind === "pending"
      ? node.inputs[0].node.shape.length
      : node.shape.length;
  return dim === rank - 1;
}

/** Get dtype from a LazyRef. */
function getRefDtype(ref: LazyRef): DType {
  if (ref.kind === "pending") return ref.node.dtype;
  if (ref.kind === "materialized")
    return (ref.storage.backendTensor as { dtype?: DType }).dtype || "f32";
  return ref.dtype;
}

// ============================================================================
// Subgraph Expansion
// ============================================================================

/**
 * Expand a subgraph from a seed reduction node, collecting elementwise
 * ops and additional same-dim reductions.
 *
 * Uses fixed-point iteration: keep expanding until no new nodes can be added.
 * This handles DAG patterns like softmax where exp feeds both sum and div,
 * and neither can be added until the other is in the subgraph.
 *
 * Returns the set of node IDs in the subgraph, or null if invalid.
 */
function expandSubgraph(
  seedNode: LazyIRNode,
  seedDim: number,
  nodeById: Map<number, LazyIRNode>,
  consumerCount: Map<number, number>,
  consumers: Map<number, LazyIRNode[]>,
  externalNodeIds: Set<number> | undefined,
  alreadyClaimedIds: Set<number> | undefined,
): Set<number> | null {
  const subgraph = new Set<number>();
  const reductionNodes = new Set<number>();

  // Start with the seed reduction
  subgraph.add(seedNode.id);
  reductionNodes.add(seedNode.id);

  // Collect candidate nodes reachable from the seed
  const candidates = new Set<number>();
  collectCandidates(
    seedNode,
    seedDim,
    nodeById,
    consumerCount,
    consumers,
    externalNodeIds,
    alreadyClaimedIds,
    candidates,
    reductionNodes,
  );

  // Fixed-point: keep adding candidates whose constraints are satisfied
  let changed = true;
  while (changed) {
    changed = false;
    for (const candId of candidates) {
      if (subgraph.has(candId)) continue;
      const candNode = nodeById.get(candId)!;

      if (REDUCE_OPS.has(candNode.op)) {
        // Reduction: add if its primary input is in the subgraph
        const primaryInput = candNode.inputs[0];
        if (
          primaryInput?.kind === "pending" &&
          subgraph.has(primaryInput.node.id)
        ) {
          subgraph.add(candId);
          changed = true;
        }
        continue;
      }

      // Elementwise: add if connected to the subgraph (via input OR consumer)
      // AND all of this node's consumers that are candidates are accounted for.
      const hasSubgraphInput = candNode.inputs.some(
        (ref) => ref.kind === "pending" && subgraph.has(ref.node.id),
      );
      const hasSubgraphConsumer = (consumers.get(candId) ?? []).some((c) =>
        subgraph.has(c.id),
      );
      if (!hasSubgraphInput && !hasSubgraphConsumer) continue;

      const cc = consumerCount.get(candId) ?? 0;
      if (cc <= 1) {
        subgraph.add(candId);
        changed = true;
      } else {
        // Multi-consumer: only add if all consumers are either in subgraph or candidates
        const nodeConsumers = consumers.get(candId) ?? [];
        const allConsumersAccountedFor = nodeConsumers.every(
          (c) => subgraph.has(c.id) || candidates.has(c.id),
        );
        if (allConsumersAccountedFor) {
          subgraph.add(candId);
          changed = true;
        }
      }
    }
  }

  // Must have at least 1 reduction + elementwise ops to be worthwhile.
  // A bare reduction with no preamble/epilogue is better handled by the
  // standard reduction dispatch (which supports K-split, chunking, etc.).
  let reductionCount = 0;
  let elemCount = 0;
  for (const id of subgraph) {
    const n = nodeById.get(id)!;
    if (REDUCE_OPS.has(n.op)) reductionCount++;
    else elemCount++;
  }
  if (reductionCount < 1) return null;
  if (reductionCount < 2 && elemCount < 1) return null;

  // Verify: no intermediate node has consumers outside the subgraph
  // (except the output node, which is allowed)
  let outputCount = 0;
  for (const nodeId of subgraph) {
    if (externalNodeIds?.has(nodeId)) {
      const nodeConsumers = consumers.get(nodeId) ?? [];
      if (nodeConsumers.some((c) => !subgraph.has(c.id))) {
        return null; // intermediate with external ref
      }
    }
    const cc = consumerCount.get(nodeId) ?? 0;
    if (cc > 0) {
      const nodeConsumers = consumers.get(nodeId) ?? [];
      if (nodeConsumers.some((c) => !subgraph.has(c.id))) {
        outputCount++;
      }
    } else {
      // Zero consumers = output
      outputCount++;
    }
  }
  if (outputCount > 1) return null; // multiple outputs not supported

  return subgraph;
}

/**
 * Collect candidate nodes reachable from the seed via forward expansion.
 * This is a relaxed BFS that collects nodes that COULD be part of the program.
 */
function collectCandidates(
  seedNode: LazyIRNode,
  seedDim: number,
  nodeById: Map<number, LazyIRNode>,
  consumerCount: Map<number, number>,
  consumers: Map<number, LazyIRNode[]>,
  externalNodeIds: Set<number> | undefined,
  alreadyClaimedIds: Set<number> | undefined,
  candidates: Set<number>,
  reductionNodes: Set<number>,
): void {
  const queue: number[] = [seedNode.id];
  candidates.add(seedNode.id);

  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    const node = nodeById.get(nodeId)!;

    // Expand forward: consumers
    const nodeConsumers = consumers.get(nodeId) ?? [];
    for (const consumer of nodeConsumers) {
      if (candidates.has(consumer.id)) continue;
      if (alreadyClaimedIds?.has(consumer.id)) continue;
      if (consumer.result) continue;
      if (externalNodeIds?.has(consumer.id)) continue;

      if (REDUCE_OPS.has(consumer.op) && isLastDimReduction(consumer)) {
        const consumerDim = getReductionDim(consumer);
        if (consumerDim === seedDim) {
          candidates.add(consumer.id);
          reductionNodes.add(consumer.id);
          queue.push(consumer.id);
          continue;
        }
      }

      if (isFusibleOp(consumer.op) && consumer.inputs.length <= 3) {
        candidates.add(consumer.id);
        queue.push(consumer.id);
      }
    }

    // Expand backward: inputs that feed into the subgraph
    // (e.g., 'x' in softmax feeds both max and sub)
    for (const ref of node.inputs) {
      if (ref.kind !== "pending") continue;
      const inputNode = ref.node;
      if (candidates.has(inputNode.id)) continue;
      if (alreadyClaimedIds?.has(inputNode.id)) continue;
      if (inputNode.result) continue;
      if (externalNodeIds?.has(inputNode.id)) continue;

      if (REDUCE_OPS.has(inputNode.op) && isLastDimReduction(inputNode)) {
        const inputDim = getReductionDim(inputNode);
        if (inputDim === seedDim) {
          candidates.add(inputNode.id);
          reductionNodes.add(inputNode.id);
          queue.push(inputNode.id);
          continue;
        }
      }

      if (isFusibleOp(inputNode.op) && inputNode.inputs.length <= 3) {
        candidates.add(inputNode.id);
        queue.push(inputNode.id);
      }
    }
  }
}

// ============================================================================
// Expression Tree Builder
// ============================================================================

/**
 * Build an RPExpr for a node within a row program phase.
 *
 * - External inputs → RPValue { kind: "input", bufferIndex }
 * - Completed reductions → RPValue { kind: "reduceResult", phaseIndex }
 * - Elementwise ops → RPExpr { op, inputs: [...recurse...] }
 */
function buildExpr(
  node: LazyIRNode,
  subgraphIds: Set<number>,
  completedReducePhases: Map<number, number>, // nodeId → phaseIndex
  inputRefMap: Map<string, number>, // refKey → bufferIndex
  inputRefs: LazyRef[], // ordered input refs
  inputDtypes: DType[], // ordered input dtypes
  nodeCount: { value: number },
  inputRefConsumers: Array<{ nodeId: number; inputIndex: number }>,
): RPExpr | null {
  if (nodeCount.value > MAX_EXPR_NODES) return null;
  nodeCount.value++;

  // Check if this node is a completed reduction
  const reducePhase = completedReducePhases.get(node.id);
  if (reducePhase !== undefined) {
    return { kind: "reduceResult", phaseIndex: reducePhase };
  }

  // This node must be a fusible elementwise op
  const inputExprs: RPExpr[] = [];
  for (let i = 0; i < node.inputs.length; i++) {
    const ref = node.inputs[i];
    const expr = buildRefExpr(
      ref,
      subgraphIds,
      completedReducePhases,
      inputRefMap,
      inputRefs,
      inputDtypes,
      nodeCount,
      inputRefConsumers,
      { nodeId: node.id, inputIndex: i },
    );
    if (!expr) return null;
    inputExprs.push(expr);
  }

  // Map op name for applyFusedOp compatibility
  let opName = node.op as string;
  if (opName === "cast") {
    const payload = node.payload as { dtype?: DType } | undefined;
    if (payload?.dtype) opName = `cast_${payload.dtype}`;
  }

  return { op: opName, inputs: inputExprs };
}

function buildRefExpr(
  ref: LazyRef,
  subgraphIds: Set<number>,
  completedReducePhases: Map<number, number>,
  inputRefMap: Map<string, number>,
  inputRefs: LazyRef[],
  inputDtypes: DType[],
  nodeCount: { value: number },
  inputRefConsumers: Array<{ nodeId: number; inputIndex: number }>,
  consumer: { nodeId: number; inputIndex: number },
): RPExpr | null {
  if (ref.kind === "scalar") {
    return { kind: "const", value: ref.value };
  }

  if (ref.kind === "materialized") {
    // External materialized input. Record CONSUMER provenance (which subgraph
    // node's input this ref is): the ref itself is a lowering-time SNAPSHOT
    // that goes stale on template reuse (its storage is a previous step's —
    // the clipGradNorm_ clipCoef class), while the consuming node is re-created
    // fresh every step. Consumers of this match resolve the CURRENT ref as
    // planNodes[consumerPos].inputs[inputIndex] — single source, never the
    // snapshot.
    const key = `m:${ref.storage.id}`;
    let idx = inputRefMap.get(key);
    if (idx === undefined) {
      if (inputRefs.length >= MAX_INPUT_BINDINGS) return null;
      idx = inputRefs.length;
      inputRefMap.set(key, idx);
      inputRefs.push(ref);
      inputDtypes.push(getRefDtype(ref));
      inputRefConsumers.push(consumer);
    }
    return { kind: "input", bufferIndex: idx };
  }

  // Pending ref
  const inputNode = ref.node;

  if (!subgraphIds.has(inputNode.id)) {
    // External pending input
    const key = `p:${inputNode.id}`;
    let idx = inputRefMap.get(key);
    if (idx === undefined) {
      if (inputRefs.length >= MAX_INPUT_BINDINGS) return null;
      idx = inputRefs.length;
      inputRefMap.set(key, idx);
      inputRefs.push(ref);
      inputDtypes.push(getRefDtype(ref));
      inputRefConsumers.push(consumer);
    }
    return { kind: "input", bufferIndex: idx };
  }

  // Internal node — recurse
  return buildExpr(
    inputNode,
    subgraphIds,
    completedReducePhases,
    inputRefMap,
    inputRefs,
    inputDtypes,
    nodeCount,
    inputRefConsumers,
  );
}

// ============================================================================
// RowProgram Builder
// ============================================================================

/**
 * Build a RowProgram from a validated subgraph.
 *
 * Topologically sorts the subgraph, partitions into phases at each
 * reduction boundary, and builds expression trees for each phase.
 */
function buildRowProgram(
  subgraphIds: Set<number>,
  nodeById: Map<number, LazyIRNode>,
  consumers: Map<number, LazyIRNode[]>,
  consumerCount: Map<number, number>,
  dim: number,
): {
  program: RowProgram;
  inputRefs: LazyRef[];
  inputRefConsumers: Array<{ nodeId: number; inputIndex: number }>;
  outputNodeId: number;
} | null {
  // Topological sort of subgraph nodes
  const sorted = topoSortSubgraph(subgraphIds, nodeById);
  if (!sorted) return null;

  // Identify the output node: the subgraph node with consumers outside
  let outputNode: LazyIRNode | null = null;
  for (const nodeId of subgraphIds) {
    const node = nodeById.get(nodeId)!;
    const cc = consumerCount.get(nodeId) ?? 0;
    const nodeConsumers = consumers.get(nodeId) ?? [];
    const hasOutsideConsumer =
      cc === 0 || nodeConsumers.some((c) => !subgraphIds.has(c.id));
    if (hasOutsideConsumer) {
      if (outputNode && outputNode.id !== node.id) {
        // Multiple output nodes — not supported
        return null;
      }
      outputNode = node;
    }
  }
  if (!outputNode) return null;

  // Build phases: walk sorted nodes, emit ReducePhase at each reduction
  const phases: RPPhase[] = [];
  const completedReducePhases = new Map<number, number>(); // nodeId → phaseIndex
  const inputRefMap = new Map<string, number>(); // refKey → bufferIndex
  const inputRefs: LazyRef[] = [];
  const inputRefConsumers: Array<{ nodeId: number; inputIndex: number }> = [];
  const inputDtypes: DType[] = [];

  for (const node of sorted) {
    if (!REDUCE_OPS.has(node.op)) continue;

    // This is a reduction node — build a ReducePhase
    const reduceInput = node.inputs[0];
    if (!reduceInput || reduceInput.kind !== "pending") return null;

    const bodyExpr = buildRefExpr(
      reduceInput,
      subgraphIds,
      completedReducePhases,
      inputRefMap,
      inputRefs,
      inputDtypes,
      { value: 0 },
      inputRefConsumers,
      { nodeId: node.id, inputIndex: 0 },
    );
    if (!bodyExpr) return null;

    const reduceOp = node.op === "mean" ? "sum" : (node.op as "sum" | "max");
    phases.push({
      kind: "reduce",
      reduceOp,
      bodyExpr,
      isMean: node.op === "mean" || undefined,
    });
    completedReducePhases.set(node.id, phases.length - 1);
  }

  // Build WritePhase for the output node
  let writeExpr: RPExpr | null;
  if (REDUCE_OPS.has(outputNode.op)) {
    // Output IS a reduction — the write is just the last reduce result
    const phaseIdx = completedReducePhases.get(outputNode.id);
    if (phaseIdx === undefined) return null;
    writeExpr = { kind: "reduceResult", phaseIndex: phaseIdx };
  } else {
    // Output is an elementwise node — build its expression tree
    writeExpr = buildExpr(
      outputNode,
      subgraphIds,
      completedReducePhases,
      inputRefMap,
      inputRefs,
      inputDtypes,
      { value: 0 },
      inputRefConsumers,
    );
  }
  if (!writeExpr) return null;

  // Per-row scalar output iff the write expression reads no per-element input
  // (only reduceResults/consts). This covers both a bare reduction output AND
  // elementwise transforms of reductions like rsqrt(mean(...)+eps). The old
  // `REDUCE_OPS.has(outputNode.op)` test missed the latter, emitting a full
  // [R,D] write that the [R,1] consumer collapsed onto row 0.
  const isScalarOutput = !exprReadsInput(writeExpr);
  phases.push({
    kind: "write",
    bodyExpr: writeExpr,
    scalarOutput: isScalarOutput || undefined,
  });

  // Build structural cache key
  const cacheKey = buildCacheKey(phases, inputDtypes);

  const program: RowProgram = {
    inputs: inputDtypes.map((dtype) => ({ dtype })),
    output: { dtype: outputNode.dtype },
    dim,
    phases,
    cacheKey,
  };

  return { program, inputRefs, inputRefConsumers, outputNodeId: outputNode.id };
}

// ============================================================================
// Cache Key
// ============================================================================

/** Build a structural cache key from phases and input dtypes. */
function buildCacheKey(phases: RPPhase[], inputDtypes: DType[]): string {
  const parts: string[] = [`i:${inputDtypes.join(",")}`];
  for (let i = 0; i < phases.length; i++) {
    const p = phases[i];
    if (p.kind === "reduce") {
      parts.push(
        `R:${p.reduceOp}${p.isMean ? ":m" : ""}:${exprKey(p.bodyExpr)}`,
      );
    } else {
      parts.push(`W:${exprKey(p.bodyExpr)}`);
    }
  }
  return parts.join("|");
}

function exprKey(e: RPExpr): string {
  if ("kind" in e) {
    switch (e.kind) {
      case "input":
        return `I${e.bufferIndex}`;
      case "reduceResult":
        return `R${e.phaseIndex}`;
      case "const":
        return `C${e.value}`;
    }
  }
  return `(${e.op} ${e.inputs.map(exprKey).join(" ")})`;
}

// ============================================================================
// Topological Sort
// ============================================================================

function topoSortSubgraph(
  subgraphIds: Set<number>,
  nodeById: Map<number, LazyIRNode>,
): LazyIRNode[] | null {
  const inDegree = new Map<number, number>();
  for (const id of subgraphIds) inDegree.set(id, 0);

  for (const id of subgraphIds) {
    const node = nodeById.get(id)!;
    for (const ref of node.inputs) {
      if (ref.kind === "pending" && subgraphIds.has(ref.node.id)) {
        inDegree.set(id, (inDegree.get(id) ?? 0) + 1);
      }
    }
  }

  const queue: number[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }

  const sorted: LazyIRNode[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    sorted.push(nodeById.get(id)!);

    // Find consumers of this node within subgraph and decrement their in-degree.
    // Must decrement for EACH input ref (not once per pair), since in-degree
    // counts total incoming refs (e.g., mul(x, x) has in-degree 2 from x).
    for (const otherId of subgraphIds) {
      if (otherId === id) continue;
      const other = nodeById.get(otherId)!;
      for (const ref of other.inputs) {
        if (ref.kind === "pending" && ref.node.id === id) {
          const newDeg = (inDegree.get(otherId) ?? 0) - 1;
          inDegree.set(otherId, newDeg);
          if (newDeg === 0) queue.push(otherId);
        }
      }
    }
  }

  if (sorted.length !== subgraphIds.size) return null; // cycle detected
  return sorted;
}

// ============================================================================
// Main Detection Entry Point
// ============================================================================

/**
 * Detect multi-reduction row-program subgraphs in a plan.
 *
 * @param planNodes       - Plan nodes in topological order
 * @param consumerCount   - Map from node ID → number of consumers
 * @param consumers       - Map from node ID → list of consumer nodes
 * @param externalNodeIds - Node IDs with external references (saved-for-backward)
 * @param alreadyClaimedIds - Node IDs already claimed by higher-priority patterns
 * @returns Array of RowProgramMatch (may be empty)
 */
export function detectRowPrograms(
  planNodes: LazyIRNode[],
  consumerCount: Map<number, number>,
  consumers: Map<number, LazyIRNode[]>,
  externalNodeIds?: Set<number>,
  alreadyClaimedIds?: Set<number>,
): RowProgramMatch[] {
  const nodeById = new Map<number, LazyIRNode>();
  for (const node of planNodes) nodeById.set(node.id, node);

  const matches: RowProgramMatch[] = [];
  const globalClaimedIds = new Set(alreadyClaimedIds);

  for (const node of planNodes) {
    if (globalClaimedIds.has(node.id)) continue;
    if (!REDUCE_OPS.has(node.op)) continue;
    if (node.result) continue;
    if (!isLastDimReduction(node)) continue;

    const dim = getReductionDim(node)!;

    const subgraphIds = expandSubgraph(
      node,
      dim,
      nodeById,
      consumerCount,
      consumers,
      externalNodeIds,
      globalClaimedIds,
    );
    if (!subgraphIds) continue;

    const result = buildRowProgram(
      subgraphIds,
      nodeById,
      consumers,
      consumerCount,
      dim,
    );
    if (!result) continue;

    // Compute geometry from the first reduction's input shape
    const inputShape =
      node.inputs[0]?.kind === "pending"
        ? node.inputs[0].node.shape
        : node.shape;
    let numRows = 1;
    for (let d = 0; d < dim; d++) numRows *= inputShape[d];
    const dimSize = inputShape[dim];

    // Claim all nodes
    const coveredNodeIds: number[] = [];
    for (const id of subgraphIds) {
      coveredNodeIds.push(id);
      globalClaimedIds.add(id);
    }

    matches.push({
      coveredNodeIds,
      outputNodeId: result.outputNodeId,
      inputRefs: result.inputRefs,
      inputRefConsumers: result.inputRefConsumers,
      dim: result.program.dim,
      numRows,
      dimSize,
      program: result.program,
    });
  }

  return matches;
}
