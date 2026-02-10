/**
 * AMP IR Transforms (§12)
 *
 * Transforms an IR graph to insert dtype cast nodes for AMP.
 * This is applied to compiled region IR before execution.
 *
 * The transform:
 * 1. Identifies ops that benefit from f16 (matmul, linear, etc.)
 * 2. Inserts cast-to-f16 nodes for inputs to those ops
 * 3. Inserts cast-to-f32 nodes for ops that require f32 (sum, mean, etc.)
 * 4. Preserves output dtypes according to the select-gated policy
 */

import type { DType } from "../backend/types";
import {
  type AMPPolicy,
  type AutocastContext,
  computeInputCasts,
  computeSelectGatedDtype,
  F16_ELIGIBLE_OPS,
  F32_REQUIRED_OPS,
} from "./amp";
import type { IRGraph, IRNode } from "./ir";

/**
 * Result of the AMP IR transform.
 */
export type AMPTransformResult = {
  /** The transformed graph with cast nodes inserted */
  graph: IRGraph;
  /** Number of cast nodes inserted */
  castsInserted: number;
  /** Map from original node ID to new node ID (for node remapping) */
  nodeIdMap: Map<number, number>;
  /** Whether any transforms were applied */
  modified: boolean;
};

/**
 * A cast node to be inserted into the graph.
 */
type PendingCast = {
  inputNodeId: number;
  fromDtype: DType;
  toDtype: DType;
  shape: number[];
};

/**
 * Apply AMP transforms to an IR graph.
 *
 * This inserts cast nodes for:
 * - f16-eligible ops: cast f32 inputs to f16
 * - f32-required ops: cast f16 inputs to f32
 *
 * @param graph The IR graph to transform
 * @param ctx The autocast context (contains AMP policy)
 * @returns The transformed graph and metadata
 */
export function applyAMPTransform(
  graph: IRGraph,
  ctx: AutocastContext,
): AMPTransformResult {
  // If autocast is disabled, return unchanged
  if (!ctx.current.enabled) {
    return {
      graph,
      castsInserted: 0,
      nodeIdMap: new Map(),
      modified: false,
    };
  }

  const policy = ctx.current.policy;
  const newNodes: IRNode[] = [];
  const nodeIdMap = new Map<number, number>();
  let nextId = Math.max(...graph.nodes.map((n) => n.id)) + 1;
  let castsInserted = 0;

  // First pass: copy nodes and identify where casts are needed
  const pendingCasts = new Map<number, PendingCast[]>(); // nodeId -> casts needed before it

  for (const node of graph.nodes) {
    // Compute what casts are needed for this node's inputs
    if (node.inputs.length > 0) {
      const inputDtypes = node.inputs.map((id) => {
        const inputNode = graph.nodes.find((n) => n.id === id);
        return inputNode?.dtype ?? "f32";
      });

      const casts = computeInputCastsForIR(node, inputDtypes, policy);
      if (casts.length > 0) {
        pendingCasts.set(node.id, casts);
      }
    }
  }

  // Second pass: build new graph with cast nodes inserted
  const inputRemapping = new Map<number, number>(); // old input id -> new input id (after cast)

  for (const node of graph.nodes) {
    // Insert any pending casts for this node's inputs
    const castsForNode = pendingCasts.get(node.id);
    if (castsForNode) {
      for (const cast of castsForNode) {
        const castNode: IRNode = {
          id: nextId++,
          op: "cast",
          epoch: node.epoch,
          kind: "lazy_op",
          inputs: [cast.inputNodeId],
          shape: cast.shape.slice(),
          dtype: cast.toDtype,
        };
        newNodes.push(castNode);
        inputRemapping.set(cast.inputNodeId, castNode.id);
        castsInserted++;
      }
    }

    // Copy the node with remapped inputs
    const remappedInputs = node.inputs.map(
      (id) => inputRemapping.get(id) ?? id,
    );

    // Determine output dtype based on select-gated logic
    const inputDtypes = remappedInputs.map((id) => {
      const inputNode = newNodes.find((n) => n.id === id);
      if (inputNode) return inputNode.dtype ?? "f32";
      const origNode = graph.nodes.find((n) => n.id === id);
      return origNode?.dtype ?? "f32";
    });

    const gatedResult = computeSelectGatedDtype(node.op, inputDtypes, ctx);

    const newNode: IRNode = {
      ...node,
      inputs: remappedInputs,
      dtype: gatedResult.outputDtype,
    };
    newNodes.push(newNode);
    nodeIdMap.set(node.id, newNode.id);

    // Clear input remapping for non-persistent casts
    // (each cast is only for the specific input use)
    for (const cast of castsForNode ?? []) {
      inputRemapping.delete(cast.inputNodeId);
    }
  }

  return {
    graph: {
      ...graph,
      nodes: newNodes,
    },
    castsInserted,
    nodeIdMap,
    modified: castsInserted > 0,
  };
}

/**
 * Compute input casts needed for an IR node based on AMP policy.
 */
function computeInputCastsForIR(
  node: IRNode,
  inputDtypes: DType[],
  policy: AMPPolicy,
): PendingCast[] {
  if (!policy.enabled) {
    return [];
  }

  const casts: PendingCast[] = [];
  const op = node.op;

  // For f16-eligible ops, cast f32 inputs to f16
  if (F16_ELIGIBLE_OPS.has(op) && policy.computeDtype === "f16") {
    for (let i = 0; i < node.inputs.length; i++) {
      if (inputDtypes[i] === "f32") {
        // Find the input node to get its shape
        const inputShape = node.shape?.slice() ?? [];
        casts.push({
          inputNodeId: node.inputs[i],
          fromDtype: "f32",
          toDtype: "f16",
          shape: inputShape,
        });
      }
    }
  }

  // For f32-required ops, cast f16 inputs to f32
  if (F32_REQUIRED_OPS.has(op)) {
    for (let i = 0; i < node.inputs.length; i++) {
      if (inputDtypes[i] === "f16") {
        const inputShape = node.shape?.slice() ?? [];
        casts.push({
          inputNodeId: node.inputs[i],
          fromDtype: "f16",
          toDtype: "f32",
          shape: inputShape,
        });
      }
    }
  }

  return casts;
}

/**
 * Check if a graph has been AMP-transformed.
 * Looks for cast nodes that were inserted by the transform.
 */
export function isAMPTransformed(graph: IRGraph): boolean {
  return graph.nodes.some((node) => node.op === "cast");
}

/**
 * Get AMP statistics for a transformed graph.
 */
export function getAMPStats(graph: IRGraph): {
  castCount: number;
  f16NodeCount: number;
  f32NodeCount: number;
} {
  let castCount = 0;
  let f16NodeCount = 0;
  let f32NodeCount = 0;

  for (const node of graph.nodes) {
    if (node.op === "cast") {
      castCount++;
    }
    if (node.dtype === "f16") {
      f16NodeCount++;
    } else if (node.dtype === "f32") {
      f32NodeCount++;
    }
  }

  return { castCount, f16NodeCount, f32NodeCount };
}
