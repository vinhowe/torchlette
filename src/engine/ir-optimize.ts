/**
 * IR Optimization inside compiled regions (§15)
 *
 * Implements:
 * - CSE (Common Subexpression Elimination) for pure ops
 * - tok_after insertion for redundant ordered loads
 * - RNG non-CSE enforcement (random ops are never CSE'd)
 * - Dead code elimination
 */

import type { IRGraph, IRNode } from "./ir";

// ============================================================================
// Op Classification
// ============================================================================

/**
 * Random ops that must NOT be CSE'd.
 * Per spec §15: "random ops non-CSE"
 */
const RANDOM_OPS = new Set([
  "rand",
  "randn",
  "randint",
  "bernoulli",
  "multinomial",
  "normal",
  "uniform",
  "dropout",
]);

/**
 * Pure elementwise ops that can be CSE'd.
 */
const PURE_ELEMENTWISE_OPS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "neg",
  "abs",
  "exp",
  "log",
  "sqrt",
  "relu",
  "sigmoid",
  "tanh",
  "gelu",
  "silu",
]);

/**
 * Pure reduction ops that can be CSE'd.
 */
const PURE_REDUCTION_OPS = new Set(["sum", "mean", "max", "min", "prod"]);

/**
 * Pure shape ops that can be CSE'd.
 */
const PURE_SHAPE_OPS = new Set([
  "reshape",
  "view",
  "transpose",
  "permute",
  "expand",
  "squeeze",
  "unsqueeze",
  "flatten",
]);

/**
 * Ops that are pure and can be CSE'd (subject to RNG exclusion).
 */
export function isPureOp(op: string): boolean {
  return (
    PURE_ELEMENTWISE_OPS.has(op) ||
    PURE_REDUCTION_OPS.has(op) ||
    PURE_SHAPE_OPS.has(op) ||
    op === "matmul" ||
    op === "gather" ||
    op === "scatterAdd"
  );
}

/**
 * Check if an op is a random op (non-CSE-able).
 */
export function isRandomOp(op: string): boolean {
  return RANDOM_OPS.has(op);
}

/**
 * Check if an op can be CSE'd.
 * Random ops are excluded even if otherwise pure.
 */
export function isCSEable(op: string): boolean {
  return isPureOp(op) && !isRandomOp(op);
}

/**
 * Check if an op is an ordered load (requires tok_after handling).
 */
export function isOrderedLoad(op: string): boolean {
  return op === "loc_load" || op === "state_load";
}

/**
 * Check if an op is effectful (can't be DCE'd if reachable).
 */
export function isEffectful(op: string): boolean {
  return (
    op.endsWith("_") || // In-place mutations
    op === "loc_store" ||
    op === "state_store" ||
    op === "print" ||
    op === "assert"
  );
}

// ============================================================================
// CSE Key Generation
// ============================================================================

/**
 * Generate a CSE key for a node.
 * Nodes with the same key can potentially be merged.
 */
export function generateCSEKey(
  node: IRNode,
  nodeIdToCSEId: Map<number, number>,
): string {
  // Random ops get unique keys (never CSE'd)
  if (isRandomOp(node.op)) {
    return `random:${node.id}`;
  }

  // Impure ops get unique keys
  if (!isPureOp(node.op)) {
    return `impure:${node.id}`;
  }

  // Normalize input references using CSE ids
  const normalizedInputs = node.inputs
    .map((id) => nodeIdToCSEId.get(id) ?? id)
    .join(",");

  // Include shape and dtype in key for precision
  const shapePart = node.shape ? node.shape.join("x") : "?";
  const dtypePart = node.dtype ?? "?";

  return `${node.op}[${normalizedInputs}](${shapePart},${dtypePart})`;
}

// ============================================================================
// Common Subexpression Elimination
// ============================================================================

/**
 * Result of CSE optimization.
 */
export type CSEResult = {
  optimizedGraph: IRGraph;
  eliminatedNodes: number[]; // IDs of nodes that were eliminated
  cseMapping: Map<number, number>; // Original ID -> CSE'd ID
  stats: {
    originalNodeCount: number;
    optimizedNodeCount: number;
    eliminatedCount: number;
  };
};

/**
 * Perform Common Subexpression Elimination on an IR graph.
 *
 * Rules:
 * 1. Pure ops with identical (op, inputs, shape, dtype) are merged
 * 2. Random ops are NEVER merged (§15)
 * 3. Effectful ops are kept separate
 */
export function performCSE(graph: IRGraph): CSEResult {
  const cseKeyToNodeId = new Map<string, number>();
  const nodeIdToCSEId = new Map<number, number>();
  const eliminatedNodes: number[] = [];
  const keptNodes: IRNode[] = [];

  // First pass: assign CSE IDs
  for (const node of graph.nodes) {
    const key = generateCSEKey(node, nodeIdToCSEId);

    if (cseKeyToNodeId.has(key)) {
      // This is a duplicate - map to existing node
      const existingId = cseKeyToNodeId.get(key)!;
      nodeIdToCSEId.set(node.id, existingId);
      eliminatedNodes.push(node.id);
    } else {
      // This is new - keep it
      cseKeyToNodeId.set(key, node.id);
      nodeIdToCSEId.set(node.id, node.id);
      keptNodes.push(node);
    }
  }

  // Second pass: rewrite inputs in kept nodes
  const optimizedNodes = keptNodes.map((node) => ({
    ...node,
    inputs: node.inputs.map((id) => nodeIdToCSEId.get(id) ?? id),
  }));

  // Rebuild fusion groups with remapped IDs
  const optimizedFusionGroups = graph.fusionGroups
    .map((group) => ({
      ...group,
      nodeIds: group.nodeIds
        .map((id) => nodeIdToCSEId.get(id) ?? id)
        .filter((id) => !eliminatedNodes.includes(id)),
    }))
    .filter((group) => group.nodeIds.length > 1);

  return {
    optimizedGraph: {
      epoch: graph.epoch,
      nodes: optimizedNodes,
      fusionGroups: optimizedFusionGroups,
    },
    eliminatedNodes,
    cseMapping: nodeIdToCSEId,
    stats: {
      originalNodeCount: graph.nodes.length,
      optimizedNodeCount: optimizedNodes.length,
      eliminatedCount: eliminatedNodes.length,
    },
  };
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

/**
 * Result of DCE optimization.
 */
export type DCEResult = {
  optimizedGraph: IRGraph;
  eliminatedNodes: number[];
  stats: {
    originalNodeCount: number;
    optimizedNodeCount: number;
    eliminatedCount: number;
  };
};

/**
 * Perform Dead Code Elimination on an IR graph.
 *
 * Rules:
 * 1. Start from outputs (nodes with no dependents that are outputs)
 * 2. Keep all nodes transitively reachable from outputs
 * 3. Keep all effectful nodes
 */
export function performDCE(graph: IRGraph, outputNodeIds: number[]): DCEResult {
  const outputSet = new Set(outputNodeIds);
  const nodeById = new Map(graph.nodes.map((n) => [n.id, n]));
  const reachable = new Set<number>();

  // Mark effectful nodes as always reachable
  for (const node of graph.nodes) {
    if (isEffectful(node.op)) {
      reachable.add(node.id);
    }
  }

  // Mark outputs as reachable
  for (const id of outputNodeIds) {
    reachable.add(id);
  }

  // Propagate reachability backwards
  let changed = true;
  while (changed) {
    changed = false;
    for (const node of graph.nodes) {
      if (reachable.has(node.id)) {
        for (const inputId of node.inputs) {
          if (!reachable.has(inputId)) {
            reachable.add(inputId);
            changed = true;
          }
        }
      }
    }
  }

  // Filter to reachable nodes
  const keptNodes = graph.nodes.filter((n) => reachable.has(n.id));
  const eliminatedNodes = graph.nodes
    .filter((n) => !reachable.has(n.id))
    .map((n) => n.id);

  // Filter fusion groups
  const optimizedFusionGroups = graph.fusionGroups
    .map((group) => ({
      ...group,
      nodeIds: group.nodeIds.filter((id) => reachable.has(id)),
    }))
    .filter((group) => group.nodeIds.length > 1);

  return {
    optimizedGraph: {
      epoch: graph.epoch,
      nodes: keptNodes,
      fusionGroups: optimizedFusionGroups,
    },
    eliminatedNodes,
    stats: {
      originalNodeCount: graph.nodes.length,
      optimizedNodeCount: keptNodes.length,
      eliminatedCount: eliminatedNodes.length,
    },
  };
}

// ============================================================================
// tok_after Optimization
// ============================================================================

/**
 * A tok_after node that replaces a redundant ordered load.
 */
export type TokAfterNode = {
  id: number;
  originalLoadId: number;
  inputTokenId: number;
  outputTokenId: number;
};

/**
 * Result of tok_after optimization.
 */
export type TokAfterResult = {
  optimizedGraph: IRGraph;
  tokAfterNodes: TokAfterNode[];
  replacedLoads: number[];
};

/**
 * Identify opportunities for tok_after optimization.
 *
 * Per spec §15: "redundant ordered-load replacement with tok_after
 * only if throw behavior cannot change"
 *
 * A load is redundant if:
 * 1. The same loc was loaded earlier in the same region
 * 2. No store to that loc occurred between the loads
 * 3. The throw behavior is identical (same potential exceptions)
 */
export function analyzeTokAfterOpportunities(
  graph: IRGraph,
  locLoads: Map<number, number[]>, // locId -> [nodeIds that load from it]
  locStores: Map<number, number[]>, // locId -> [nodeIds that store to it]
): { redundantLoads: number[]; firstLoadMap: Map<number, number> } {
  const redundantLoads: number[] = [];
  const firstLoadMap = new Map<number, number>(); // redundant load -> first load

  const nodeOrder = new Map(graph.nodes.map((n, i) => [n.id, i]));

  for (const [locId, loadNodeIds] of locLoads) {
    if (loadNodeIds.length <= 1) continue;

    // Sort by order in graph
    const sortedLoads = loadNodeIds.slice().sort((a, b) => {
      return (nodeOrder.get(a) ?? 0) - (nodeOrder.get(b) ?? 0);
    });

    const storeNodeIds = locStores.get(locId) ?? [];
    const storeOrders = storeNodeIds.map((id) => nodeOrder.get(id) ?? Infinity);

    // First load is the canonical one
    const firstLoad = sortedLoads[0];
    const firstLoadOrder = nodeOrder.get(firstLoad) ?? 0;

    // Check subsequent loads
    for (let i = 1; i < sortedLoads.length; i++) {
      const loadId = sortedLoads[i];
      const loadOrder = nodeOrder.get(loadId) ?? 0;

      // Check if any store happened between first load and this load
      const hasInterveningStore = storeOrders.some(
        (storeOrder) => storeOrder > firstLoadOrder && storeOrder < loadOrder,
      );

      if (!hasInterveningStore) {
        redundantLoads.push(loadId);
        firstLoadMap.set(loadId, firstLoad);
      }
    }
  }

  return { redundantLoads, firstLoadMap };
}

// ============================================================================
// Full IR Optimization Pipeline
// ============================================================================

/**
 * Options for IR optimization.
 */
export type OptimizeOptions = {
  enableCSE?: boolean;
  enableDCE?: boolean;
  enableTokAfter?: boolean;
  outputNodeIds?: number[];
};

/**
 * Result of full optimization pipeline.
 */
export type OptimizeResult = {
  optimizedGraph: IRGraph;
  cse?: CSEResult;
  dce?: DCEResult;
  stats: {
    originalNodeCount: number;
    finalNodeCount: number;
    cseEliminated: number;
    dceEliminated: number;
  };
};

/**
 * Run the full IR optimization pipeline.
 */
export function optimizeIR(
  graph: IRGraph,
  options: OptimizeOptions = {},
): OptimizeResult {
  const { enableCSE = true, enableDCE = true, outputNodeIds = [] } = options;

  let currentGraph = graph;
  let cseResult: CSEResult | undefined;
  let dceResult: DCEResult | undefined;

  // Step 1: CSE
  if (enableCSE) {
    cseResult = performCSE(currentGraph);
    currentGraph = cseResult.optimizedGraph;
  }

  // Step 2: DCE (if outputs are known)
  if (enableDCE && outputNodeIds.length > 0) {
    // Remap output IDs if CSE was performed
    const remappedOutputs = cseResult
      ? outputNodeIds.map((id) => cseResult!.cseMapping.get(id) ?? id)
      : outputNodeIds;

    dceResult = performDCE(currentGraph, remappedOutputs);
    currentGraph = dceResult.optimizedGraph;
  }

  return {
    optimizedGraph: currentGraph,
    cse: cseResult,
    dce: dceResult,
    stats: {
      originalNodeCount: graph.nodes.length,
      finalNodeCount: currentGraph.nodes.length,
      cseEliminated: cseResult?.stats.eliminatedCount ?? 0,
      dceEliminated: dceResult?.stats.eliminatedCount ?? 0,
    },
  };
}
