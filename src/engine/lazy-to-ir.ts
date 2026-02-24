/**
 * Convert lazy execution plans to IR graphs for optimization.
 *
 * This module bridges the lazy execution system (LazyIRNode) with the
 * IR optimization infrastructure (IRGraph, IRNode).
 *
 * Per spec §15: Optimization passes (CSE, DCE, fusion) operate on IRGraph.
 */

import type { DType } from "../backend/types";
import type { IRFusionGroup, IRGraph, IRNode } from "./ir";
import type { ExecutionPlan, LazyIRNode } from "./lazy";

/**
 * Mapping from LazyIRNode IDs to IRNode IDs.
 */
type NodeIdMapping = Map<number, number>;

/**
 * Result of converting a lazy plan to IR.
 */
type LazyToIRResult = {
  graph: IRGraph;
  nodeMapping: NodeIdMapping;
  outputNodeIds: number[];
  scalarsByNode: Map<number, number[]>;  // IRNode id → scalar input values (§8.2.1)
};

/**
 * Set of ops that are fusible elementwise operations.
 */
const FUSIBLE_ELEMENTWISE_OPS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "sqrt",
  "relu",
  "neg",
  "abs",
  "exp",
  "log",
  "sigmoid",
  "tanh",
  "gelu",
  "silu",
]);

/**
 * Check if an op is a fusible elementwise operation.
 */
export function isFusibleElementwise(op: string): boolean {
  return FUSIBLE_ELEMENTWISE_OPS.has(op);
}

/**
 * Convert a lazy execution plan to an IR graph.
 *
 * @param plan - The execution plan containing LazyIRNodes
 * @returns IRGraph suitable for optimization passes
 */
export function lazyPlanToIR(plan: ExecutionPlan): LazyToIRResult {
  const nodes: IRNode[] = [];
  const nodeMapping: NodeIdMapping = new Map();
  const scalarsByNode = new Map<number, number[]>();
  const epoch = Date.now(); // Use timestamp as epoch for unique identification

  // First pass: create IRNodes from LazyIRNodes, collecting scalar values
  for (const lazyNode of plan.nodes) {
    const irNode = lazyNodeToIRNode(lazyNode, nodeMapping, epoch);
    nodes.push(irNode);
    nodeMapping.set(lazyNode.id, irNode.id);

    // Collect scalar inputs for this node (§8.2.1)
    const scalars: number[] = [];
    for (const inputRef of lazyNode.inputs) {
      if (inputRef.kind === "scalar") {
        scalars.push(inputRef.value);
      }
    }
    if (scalars.length > 0) {
      scalarsByNode.set(irNode.id, scalars);
      irNode.scalarValues = scalars;
    }
  }

  // Detect fusion groups
  const fusionGroups = detectFusionGroups(nodes);

  // The output is the last node in the plan
  const outputNodeIds = plan.nodes.length > 0
    ? [nodeMapping.get(plan.nodes[plan.nodes.length - 1].id)!]
    : [];

  return {
    graph: {
      epoch,
      nodes,
      fusionGroups,
    },
    nodeMapping,
    outputNodeIds,
    scalarsByNode,
  };
}

/**
 * Convert a single LazyIRNode to an IRNode.
 */
function lazyNodeToIRNode(
  lazyNode: LazyIRNode,
  nodeMapping: NodeIdMapping,
  epoch: number,
): IRNode {
  // Map input LazyRef IDs to IRNode IDs
  const inputs: number[] = [];
  for (const inputRef of lazyNode.inputs) {
    if (inputRef.kind === "pending") {
      const mappedId = nodeMapping.get(inputRef.node.id);
      if (mappedId !== undefined) {
        inputs.push(mappedId);
      }
    }
    // Materialized inputs are not part of the IR graph (they're external)
  }

  return {
    id: lazyNode.id,
    op: lazyNode.op,
    epoch,
    kind: "lazy_op",
    inputs,
    shape: lazyNode.shape?.slice(),
    dtype: lazyNode.dtype as DType,
  };
}

/**
 * Detect fusion groups in an IR graph.
 *
 * A fusion group is a sequence of consecutive fusible elementwise ops
 * that can be combined into a single kernel.
 *
 * Fusion barriers:
 * - Non-elementwise ops (matmul, reduce, gather, etc.)
 * - Random ops (rand, randn, dropout)
 * - Ops with side effects
 */
export function detectFusionGroups(nodes: IRNode[]): IRFusionGroup[] {
  const groups: IRFusionGroup[] = [];
  let currentGroup: number[] = [];

  for (const node of nodes) {
    if (isFusibleElementwise(node.op)) {
      currentGroup.push(node.id);
    } else {
      // Non-fusible op: finalize current group if it has 2+ ops
      if (currentGroup.length >= 2) {
        groups.push({
          id: groups.length,
          kind: "elementwise",
          nodeIds: currentGroup,
        });
      }
      currentGroup = [];
    }
  }

  // Finalize any remaining group
  if (currentGroup.length >= 2) {
    groups.push({
      id: groups.length,
      kind: "elementwise",
      nodeIds: currentGroup,
    });
  }

  return groups;
}

