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
  const maxPayload = maxNode.payload as { dim?: number; keepdim?: boolean } | undefined;
  if (!maxPayload || !maxPayload.keepdim) return null;
  const dim = maxPayload.dim;
  if (dim === undefined) return null;

  // 2. max must have exactly 1 consumer → sub
  if ((consumerCount.get(maxNode.id) ?? 0) !== 1) return null;
  const maxConsumers = consumers.get(maxNode.id);
  if (!maxConsumers || maxConsumers.length !== 1) return null;
  const subNode = maxConsumers[0];
  if (subNode.op !== "sub") return null;

  // 3. sub must take (x, maxVal) where x is also the input to max
  const maxInputRef = maxNode.inputs[0];
  if (!maxInputRef) return null;
  const xNodeId = maxInputRef.kind === "pending" ? maxInputRef.node.id : -1;
  const xIsMaterialized = maxInputRef.kind === "materialized";

  // Verify sub(x, maxVal): sub.inputs[0] = x, sub.inputs[1] = maxVal
  if (subNode.inputs.length < 2) return null;
  const subInput0 = subNode.inputs[0];
  const subInput1 = subNode.inputs[1];
  if (subInput1.kind !== "pending" || subInput1.node.id !== maxNode.id) return null;
  // sub.inputs[0] should be the same x as max.inputs[0]
  if (xIsMaterialized) {
    if (subInput0.kind !== "materialized") return null;
    // Can't verify identity of materialized refs — rely on structural match
  } else {
    if (subInput0.kind !== "pending" || subInput0.node.id !== xNodeId) return null;
  }

  // 4. sub must have exactly 1 consumer → exp
  if ((consumerCount.get(subNode.id) ?? 0) !== 1) return null;
  const subConsumers = consumers.get(subNode.id);
  if (!subConsumers || subConsumers.length !== 1) return null;
  const expNode = subConsumers[0];
  if (expNode.op !== "exp") return null;

  // 5. exp must have exactly 2 consumers → sum and div
  if ((consumerCount.get(expNode.id) ?? 0) !== 2) return null;
  const expConsumers = consumers.get(expNode.id);
  if (!expConsumers || expConsumers.length !== 2) return null;

  // Find sum and div among exp's consumers
  let sumNode: LazyIRNode | null = null;
  let divNode: LazyIRNode | null = null;
  for (const c of expConsumers) {
    if (c.op === "sum") sumNode = c;
    else if (c.op === "div") divNode = c;
  }
  if (!sumNode || !divNode) return null;

  // 6. sum must use the same dim with keepdim=true
  const sumPayload = sumNode.payload as { dim?: number; keepdim?: boolean } | undefined;
  if (!sumPayload || !sumPayload.keepdim || sumPayload.dim !== dim) return null;
  // sum's input must be exp
  if (sumNode.inputs.length < 1) return null;
  if (sumNode.inputs[0].kind !== "pending" || sumNode.inputs[0].node.id !== expNode.id) return null;

  // 7. sum must have exactly 1 consumer → div
  if ((consumerCount.get(sumNode.id) ?? 0) !== 1) return null;
  const sumConsumers = consumers.get(sumNode.id);
  if (!sumConsumers || sumConsumers.length !== 1) return null;
  if (sumConsumers[0].id !== divNode.id) return null;

  // 8. div must take (exp, sum) as inputs
  if (divNode.inputs.length < 2) return null;
  const divInput0 = divNode.inputs[0];
  const divInput1 = divNode.inputs[1];
  if (divInput0.kind !== "pending" || divInput0.node.id !== expNode.id) return null;
  if (divInput1.kind !== "pending" || divInput1.node.id !== sumNode.id) return null;

  // 9. Don't claim nodes that are externally referenced (except the output)
  const intermediates = [maxNode, subNode, expNode, sumNode];
  for (const n of intermediates) {
    if (externalNodeIds?.has(n.id)) return null;
  }

  // All checks pass — this is a softmax pattern
  return {
    name: "softmax",
    coveredNodeIds: [maxNode.id, subNode.id, expNode.id, sumNode.id, divNode.id],
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
  const maxPayload = maxNode.payload as { dim?: number; keepdim?: boolean } | undefined;
  if (!maxPayload || !maxPayload.keepdim) return null;
  const dim = maxPayload.dim;
  if (dim === undefined) return null;

  // 2. max → sub₁
  if ((consumerCount.get(maxNode.id) ?? 0) !== 1) return null;
  const maxConsumers = consumers.get(maxNode.id);
  if (!maxConsumers || maxConsumers.length !== 1) return null;
  const sub1Node = maxConsumers[0];
  if (sub1Node.op !== "sub") return null;

  // 3. Verify sub₁(x, maxVal)
  const maxInputRef = maxNode.inputs[0];
  if (!maxInputRef) return null;
  const xNodeId = maxInputRef.kind === "pending" ? maxInputRef.node.id : -1;
  const xIsMaterialized = maxInputRef.kind === "materialized";

  if (sub1Node.inputs.length < 2) return null;
  if (sub1Node.inputs[1].kind !== "pending" || sub1Node.inputs[1].node.id !== maxNode.id) return null;
  if (xIsMaterialized) {
    if (sub1Node.inputs[0].kind !== "materialized") return null;
  } else {
    if (sub1Node.inputs[0].kind !== "pending" || sub1Node.inputs[0].node.id !== xNodeId) return null;
  }

  // 4. sub₁ must have exactly 2 consumers → exp + sub₂
  if ((consumerCount.get(sub1Node.id) ?? 0) !== 2) return null;
  const sub1Consumers = consumers.get(sub1Node.id);
  if (!sub1Consumers || sub1Consumers.length !== 2) return null;

  let expNode: LazyIRNode | null = null;
  let sub2Node: LazyIRNode | null = null;
  for (const c of sub1Consumers) {
    if (c.op === "exp") expNode = c;
    else if (c.op === "sub") sub2Node = c;
  }
  if (!expNode || !sub2Node) return null;

  // 5. exp → sum (exactly 1 consumer)
  if ((consumerCount.get(expNode.id) ?? 0) !== 1) return null;
  const expConsumers = consumers.get(expNode.id);
  if (!expConsumers || expConsumers.length !== 1) return null;
  const sumNode = expConsumers[0];
  if (sumNode.op !== "sum") return null;

  // 6. sum must use same dim, keepdim=true
  const sumPayload = sumNode.payload as { dim?: number; keepdim?: boolean } | undefined;
  if (!sumPayload || !sumPayload.keepdim || sumPayload.dim !== dim) return null;
  if (sumNode.inputs[0].kind !== "pending" || sumNode.inputs[0].node.id !== expNode.id) return null;

  // 7. sum → log (exactly 1 consumer)
  if ((consumerCount.get(sumNode.id) ?? 0) !== 1) return null;
  const sumConsumers = consumers.get(sumNode.id);
  if (!sumConsumers || sumConsumers.length !== 1) return null;
  const logNode = sumConsumers[0];
  if (logNode.op !== "log") return null;

  // 8. log → sub₂ (exactly 1 consumer)
  if ((consumerCount.get(logNode.id) ?? 0) !== 1) return null;
  const logConsumers = consumers.get(logNode.id);
  if (!logConsumers || logConsumers.length !== 1) return null;
  if (logConsumers[0].id !== sub2Node.id) return null;

  // 9. sub₂ must take (shifted, logSumVal)
  if (sub2Node.inputs.length < 2) return null;
  if (sub2Node.inputs[0].kind !== "pending" || sub2Node.inputs[0].node.id !== sub1Node.id) return null;
  if (sub2Node.inputs[1].kind !== "pending" || sub2Node.inputs[1].node.id !== logNode.id) return null;

  // 10. Don't claim externally referenced intermediates
  const intermediates = [maxNode, sub1Node, expNode, sumNode, logNode];
  for (const n of intermediates) {
    if (externalNodeIds?.has(n.id)) return null;
  }

  return {
    name: "log_softmax",
    coveredNodeIds: [maxNode.id, sub1Node.id, expNode.id, sumNode.id, logNode.id, sub2Node.id],
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
    let match = matchLogSoftmax(node, consumers, consumerCount, externalNodeIds);
    if (!match) {
      match = matchSoftmax(node, consumers, consumerCount, externalNodeIds);
    }

    if (match) {
      // Verify no nodes are already claimed
      const conflict = match.coveredNodeIds.some(id => claimedIds.has(id));
      if (!conflict) {
        matches.push(match);
        for (const id of match.coveredNodeIds) claimedIds.add(id);
      }
    }
  }

  return matches;
}
