/**
 * Compound Pattern Recognition
 *
 * Detects higher-level computation patterns (softmax, log_softmax) from
 * decomposed LazyIRNode graphs and marks them for optimized execution.
 *
 * Runs during the analysis phase of executePlanOptimized(), after matmul
 * epilogue pre-scan and before elementwise fusion detection. Claimed nodes
 * are excluded from fusion groups.
 *
 * Pattern recognition walks the dependency graph forward from candidate
 * start nodes (e.g., `max` for softmax), verifying the exact op/shape/
 * connection pattern.
 */

import type { LazyIRNode, LazyRef } from "./lazy-types";

// ============================================================================
// Helpers
// ============================================================================

/** Get the single consumer of a node, or null if it has != 1 consumers. */
function getSoleConsumer(
  nodeId: number,
  consumers: Map<number, LazyIRNode[]>,
  consumerCount: Map<number, number>,
): LazyIRNode | null {
  if ((consumerCount.get(nodeId) ?? 0) !== 1) return null;
  const list = consumers.get(nodeId);
  return list?.length === 1 ? list[0] : null;
}

/** Check if a LazyRef is a pending ref pointing to a specific node. */
function isPendingFrom(ref: LazyRef, nodeId: number): boolean {
  return ref.kind === "pending" && ref.node.id === nodeId;
}

/** Extract the reduction dim from a keepdim=true reduction payload, or null. */
function getKeepdimReductionDim(
  node: LazyIRNode,
  expectedDim?: number,
): number | null {
  const p = node.payload as { dim?: number; keepdim?: boolean } | undefined;
  if (!p?.keepdim || p.dim === undefined) return null;
  if (expectedDim !== undefined && p.dim !== expectedDim) return null;
  return p.dim;
}

/** Return true if any node in the list is externally referenced. */
function hasExternalNode(
  nodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
): boolean {
  return (
    externalNodeIds != null && nodes.some((n) => externalNodeIds.has(n.id))
  );
}

/** Extract node ID and materialized status from a ref. */
function refInfo(
  ref: LazyRef | undefined,
): { nodeId: number; isMaterialized: boolean } | null {
  if (!ref) return null;
  return {
    nodeId: ref.kind === "pending" ? ref.node.id : -1,
    isMaterialized: ref.kind === "materialized",
  };
}

/** Find a consumer by op from a list. */
function findByOp(nodes: LazyIRNode[], op: string): LazyIRNode | null {
  for (const n of nodes) if (n.op === op) return n;
  return null;
}

// ============================================================================
// Types
// ============================================================================

export interface CompoundMatch {
  /** Pattern name: "softmax" or "log_softmax" */
  name: string;
  /** Plan-node IDs consumed by this pattern. */
  coveredNodeIds: number[];
  /** Node ID of the final output. */
  outputNodeId: number;
  /** Reduction dimension (normalized). */
  dim: number;
  /** Node ID of the input tensor to the pattern. */
  inputNodeId: number;
  /** Whether the input comes from a materialized ref (not a pending node). */
  inputIsMaterialized: boolean;
}

// ============================================================================
// Softmax Pattern
// ============================================================================

/**
 * Softmax decomposition pattern:
 *   max(x, dim, keepdim=true) → sub(x, maxVal) → exp(shifted) → sum(exps, dim, keepdim=true) → div(exps, sumVal)
 *
 * Requirements:
 * - All intermediate nodes have exactly 1 consumer (except exp which has 2: sum + div)
 * - x (input) feeds both max and sub
 * - All nodes are f32
 * - max and sum use the same dim with keepdim=true
 */
function matchSoftmax(
  maxNode: LazyIRNode,
  consumers: Map<number, LazyIRNode[]>,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): CompoundMatch | null {
  // 1. maxNode must be "max" with keepdim=true
  if (maxNode.op !== "max") return null;
  if (maxNode.dtype !== "f32") return null;
  const dim = getKeepdimReductionDim(maxNode);
  if (dim === null) return null;

  // 2. max → sub (single consumer)
  const subNode = getSoleConsumer(maxNode.id, consumers, consumerCount);
  if (!subNode || subNode.op !== "sub") return null;

  // 3. sub must take (x, maxVal) where x is also the input to max
  const xInfo = refInfo(maxNode.inputs[0]);
  if (!xInfo) return null;
  const { nodeId: xNodeId, isMaterialized: xIsMaterialized } = xInfo;

  if (subNode.inputs.length < 2) return null;
  if (!isPendingFrom(subNode.inputs[1], maxNode.id)) return null;
  if (xIsMaterialized) {
    if (subNode.inputs[0].kind !== "materialized") return null;
  } else {
    if (!isPendingFrom(subNode.inputs[0], xNodeId)) return null;
  }

  // 4. sub → exp (single consumer)
  const expNode = getSoleConsumer(subNode.id, consumers, consumerCount);
  if (!expNode || expNode.op !== "exp") return null;

  // 5. exp must have exactly 2 consumers → sum and div
  if ((consumerCount.get(expNode.id) ?? 0) !== 2) return null;
  const expConsumers = consumers.get(expNode.id);
  if (!expConsumers || expConsumers.length !== 2) return null;

  const sumNode = findByOp(expConsumers, "sum");
  const divNode = findByOp(expConsumers, "div");
  if (!sumNode || !divNode) return null;

  // 6. sum must use the same dim with keepdim=true, input from exp
  if (getKeepdimReductionDim(sumNode, dim) === null) return null;
  if (!sumNode.inputs[0] || !isPendingFrom(sumNode.inputs[0], expNode.id))
    return null;

  // 7. sum → div (single consumer)
  const sumSoleConsumer = getSoleConsumer(sumNode.id, consumers, consumerCount);
  if (!sumSoleConsumer || sumSoleConsumer.id !== divNode.id) return null;

  // 8. div must take (exp, sum) as inputs
  if (divNode.inputs.length < 2) return null;
  if (!isPendingFrom(divNode.inputs[0], expNode.id)) return null;
  if (!isPendingFrom(divNode.inputs[1], sumNode.id)) return null;

  // 9. Don't claim externally referenced intermediates
  if (hasExternalNode([maxNode, subNode, expNode, sumNode], externalNodeIds))
    return null;

  // All checks pass — this is a softmax pattern
  return {
    name: "softmax",
    coveredNodeIds: [
      maxNode.id,
      subNode.id,
      expNode.id,
      sumNode.id,
      divNode.id,
    ],
    outputNodeId: divNode.id,
    dim,
    inputNodeId: xNodeId,
    inputIsMaterialized: xIsMaterialized,
  };
}

// ============================================================================
// Log-Softmax Pattern
// ============================================================================

/**
 * Log-softmax decomposition pattern:
 *   max(x, dim, keepdim=true) → sub(x, maxVal) → exp(shifted) → sum(exps, dim, keepdim=true) → log(sumVal) → sub(shifted, logSumVal)
 *
 * Requirements:
 * - max, sub₁, exp, sum have single consumers (except sub₁ which has 2: exp + sub₂)
 * - x feeds both max and sub₁
 * - Same dim for max and sum
 */
function matchLogSoftmax(
  maxNode: LazyIRNode,
  consumers: Map<number, LazyIRNode[]>,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): CompoundMatch | null {
  // 1. maxNode must be "max" with keepdim=true
  if (maxNode.op !== "max") return null;
  if (maxNode.dtype !== "f32") return null;
  const dim = getKeepdimReductionDim(maxNode);
  if (dim === null) return null;

  // 2. max → sub₁ (single consumer)
  const sub1Node = getSoleConsumer(maxNode.id, consumers, consumerCount);
  if (!sub1Node || sub1Node.op !== "sub") return null;

  // 3. Verify sub₁(x, maxVal)
  const xInfo = refInfo(maxNode.inputs[0]);
  if (!xInfo) return null;
  const { nodeId: xNodeId, isMaterialized: xIsMaterialized } = xInfo;

  if (sub1Node.inputs.length < 2) return null;
  if (!isPendingFrom(sub1Node.inputs[1], maxNode.id)) return null;
  if (xIsMaterialized) {
    if (sub1Node.inputs[0].kind !== "materialized") return null;
  } else {
    if (!isPendingFrom(sub1Node.inputs[0], xNodeId)) return null;
  }

  // 4. sub₁ must have exactly 2 consumers → exp + sub₂
  if ((consumerCount.get(sub1Node.id) ?? 0) !== 2) return null;
  const sub1Consumers = consumers.get(sub1Node.id);
  if (!sub1Consumers || sub1Consumers.length !== 2) return null;

  const expNode = findByOp(sub1Consumers, "exp");
  const sub2Node = findByOp(sub1Consumers, "sub");
  if (!expNode || !sub2Node) return null;

  // 5. exp → sum (single consumer)
  const sumNode = getSoleConsumer(expNode.id, consumers, consumerCount);
  if (!sumNode || sumNode.op !== "sum") return null;

  // 6. sum must use same dim, keepdim=true, input from exp
  if (getKeepdimReductionDim(sumNode, dim) === null) return null;
  if (!isPendingFrom(sumNode.inputs[0], expNode.id)) return null;

  // 7. sum → log (single consumer)
  const logNode = getSoleConsumer(sumNode.id, consumers, consumerCount);
  if (!logNode || logNode.op !== "log") return null;

  // 8. log → sub₂ (single consumer, must be our sub₂)
  const logSoleConsumer = getSoleConsumer(logNode.id, consumers, consumerCount);
  if (!logSoleConsumer || logSoleConsumer.id !== sub2Node.id) return null;

  // 9. sub₂ must take (shifted, logSumVal)
  if (sub2Node.inputs.length < 2) return null;
  if (!isPendingFrom(sub2Node.inputs[0], sub1Node.id)) return null;
  if (!isPendingFrom(sub2Node.inputs[1], logNode.id)) return null;

  // 10. Don't claim externally referenced intermediates
  if (
    hasExternalNode(
      [maxNode, sub1Node, expNode, sumNode, logNode],
      externalNodeIds,
    )
  )
    return null;

  return {
    name: "log_softmax",
    coveredNodeIds: [
      maxNode.id,
      sub1Node.id,
      expNode.id,
      sumNode.id,
      logNode.id,
      sub2Node.id,
    ],
    outputNodeId: sub2Node.id,
    dim,
    inputNodeId: xNodeId,
    inputIsMaterialized: xIsMaterialized,
  };
}

// ============================================================================
// Main Detection Entry Point
// ============================================================================

/**
 * Detect compound patterns in a plan.
 *
 * @param planNodes       Plan nodes in topological order.
 * @param consumerCount   Map from node ID → number of consumers in the plan.
 * @param consumers       Map from node ID → list of consumer nodes.
 * @param externalNodeIds Set of node IDs that are externally referenced.
 * @returns Array of matched compound patterns (may be empty).
 */
export function detectCompoundPatterns(
  planNodes: LazyIRNode[],
  consumerCount: Map<number, number>,
  consumers: Map<number, LazyIRNode[]>,
  externalNodeIds?: Set<number>,
): CompoundMatch[] {
  const matches: CompoundMatch[] = [];
  const claimedIds = new Set<number>();

  for (const node of planNodes) {
    if (claimedIds.has(node.id)) continue;
    if (node.op !== "max") continue;

    // Try log_softmax first (it's a superset of softmax pattern start)
    let match = matchLogSoftmax(
      node,
      consumers,
      consumerCount,
      externalNodeIds,
    );
    if (!match) {
      match = matchSoftmax(node, consumers, consumerCount, externalNodeIds);
    }

    if (match) {
      // Verify no nodes are already claimed
      const conflict = match.coveredNodeIds.some((id) => claimedIds.has(id));
      if (!conflict) {
        matches.push(match);
        for (const id of match.coveredNodeIds) claimedIds.add(id);
      }
    }
  }

  return matches;
}
