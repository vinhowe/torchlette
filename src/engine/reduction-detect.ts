/**
 * Reduction Group Detection
 *
 * Detects elementwise → reduction and reduction → elementwise patterns
 * and packages them as first-class ReductionGroup objects for segment-level
 * execution. Replaces the per-node directive approach of reduction-preamble.ts.
 *
 * Three patterns are recognized:
 *  - Preamble only:  [elem chain] → sum/mean
 *  - Epilogue only:  sum/mean/max → [elem chain]
 *  - Combined:       [elem chain] → sum/mean → [elem chain]
 */

import type { DType } from "../backend/types";
import type { PreambleChainKernelOp } from "../backend/webgpu/reduction-tile-ir";
import { shapesEqual } from "../core/shape";
import { isFusibleOp } from "./fusion-detect";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import {
  type EpilogueOp,
  findChainInput,
  getEpilogueOpName,
} from "./matmul-epilogue";

// ============================================================================
// Types
// ============================================================================

/** A detected reduction group: preamble → reduction → epilogue. */
export interface ReductionGroup {
  /** All nodes consumed (preamble + reduction + epilogue), in plan order. */
  nodes: LazyIRNode[];
  /** Plan indices for all consumed nodes. */
  planIndices: number[];

  /** The reduction node. */
  reductionNode: LazyIRNode;
  /** Preamble chain nodes (before reduction), may be empty. */
  preambleNodes: LazyIRNode[];
  /** Epilogue chain nodes (after reduction), may be empty. */
  epilogueNodes: LazyIRNode[];

  /** Output node (last epilogue, or reduction if no epilogue). */
  outputNode: LazyIRNode;

  /** PreambleChainKernelOp descriptors for tile-IR codegen. */
  preambleOps: PreambleChainKernelOp[];
  /** External input refs for preamble chain. */
  preambleInputRefs: LazyRef[];
  /** Dtypes for each preamble external input. */
  preambleInputDtypes: DType[];

  /** Epilogue operations to apply after the reduction. */
  epilogueOps: EpilogueOp[];
  /** External input refs for epilogue binary ops. */
  epilogueInputRefs: LazyRef[];

  /** Output dtype after epilogue chain. */
  outputDtype: DType;
  /** Whether the reduction is a mean (needs /count). */
  isMean: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

/** Extract the shape from a LazyRef (pending or materialized). */
function getRefShape(ref: LazyRef): number[] | undefined {
  if (ref.kind === "pending") return ref.node.shape;
  if (ref.kind === "materialized") return ref.storage.backendTensor.shape;
  return undefined;
}

/** Get the dtype of a LazyRef. */
function getRefDtype(ref: LazyRef): DType {
  if (ref.kind === "pending") return ref.node.dtype;
  if (ref.kind === "materialized") {
    return (ref.storage.backendTensor as { dtype?: DType }).dtype || "f32";
  }
  return "f32";
}

// ============================================================================
// Preamble Detection (elementwise → reduction)
// ============================================================================

/**
 * Walk forward from startIdx collecting fusible elementwise ops until
 * hitting a sum/mean reduction. Returns the preamble chain + reduction,
 * or null if no pattern found.
 */
function detectPreamble(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
): {
  chain: LazyIRNode[];
  chainOps: PreambleChainKernelOp[];
  chainInputRefs: LazyRef[];
  chainInputDtypes: DType[];
  reductionNode: LazyIRNode;
  reductionIdx: number;
} | null {
  if (startIdx + 1 >= nodes.length) return null;

  const firstNode = nodes[startIdx];
  if (!isFusibleOp(firstNode.op)) return null;
  if (firstNode.inputs.length > 2) return null;
  if (firstNode.result) return null;

  // Build chain of fusible elem ops ending at a sum/mean
  const chain: LazyIRNode[] = [firstNode];
  let reductionNode: LazyIRNode | null = null;
  let reductionIdx = -1;
  const MAX_CHAIN = 4;

  for (
    let idx = startIdx + 1;
    idx < nodes.length && chain.length <= MAX_CHAIN;
    idx++
  ) {
    const nextNode = nodes[idx];
    const lastChainNode = chain[chain.length - 1];

    // Check if nextNode is a reduction consuming the last chain node
    if (nextNode.op === "sum" || nextNode.op === "mean") {
      if (
        nextNode.inputs.length >= 1 &&
        nextNode.inputs[0].kind === "pending" &&
        nextNode.inputs[0].node === lastChainNode
      ) {
        if ((consumerCount.get(lastChainNode.id) ?? 0) !== 1) return null;
        reductionNode = nextNode;
        reductionIdx = idx;
        break;
      }
      break;
    }

    // Check if nextNode can extend the chain
    if (!isFusibleOp(nextNode.op)) break;
    if (nextNode.inputs.length > 2) break;
    if (nextNode.result) break;

    const consumesLast = nextNode.inputs.some(
      (inp) => inp.kind === "pending" && inp.node === lastChainNode,
    );
    if (!consumesLast) break;
    if ((consumerCount.get(lastChainNode.id) ?? 0) !== 1) break;

    chain.push(nextNode);
  }

  if (!reductionNode) return null;

  // Single-op chain: skip standalone cast (no benefit)
  if (chain.length === 1) {
    if (firstNode.op === "cast") return null;

    const elemShape = firstNode.shape;
    for (const ref of firstNode.inputs) {
      const inputNode = ref.kind === "pending" ? ref.node : null;
      if (inputNode && !shapesEqual(inputNode.shape, elemShape)) return null;
    }
    if (firstNode.dtype !== "f32") return null;
    for (const ref of firstNode.inputs) {
      if (ref.kind === "pending" && ref.node.dtype !== "f32") return null;
    }

    return {
      chain: [firstNode],
      chainOps: [{ op: firstNode.op, arity: firstNode.inputs.length }],
      chainInputRefs: [...firstNode.inputs],
      chainInputDtypes: firstNode.inputs.map(getRefDtype),
      reductionNode,
      reductionIdx,
    };
  }

  // Multi-op chain: collect external input refs and build chain descriptors
  const externalInputRefs: LazyRef[] = [];
  const externalInputDtypes: DType[] = [];
  const chainOps: PreambleChainKernelOp[] = [];
  const firstShape = firstNode.shape;

  // First node: all inputs are external
  for (const ref of firstNode.inputs) {
    const refShape = getRefShape(ref);
    if (refShape && !shapesEqual(refShape, firstShape)) return null;
    externalInputRefs.push(ref);
    externalInputDtypes.push(getRefDtype(ref));
  }
  chainOps.push({
    op: getEpilogueOpName(firstNode),
    arity: firstNode.inputs.length,
  });

  // Subsequent nodes: identify chain input vs external input
  for (let i = 1; i < chain.length; i++) {
    const node = chain[i];
    const prevNode = chain[i - 1];

    if (node.inputs.length === 1) {
      chainOps.push({ op: getEpilogueOpName(node), arity: 1 });
    } else {
      const inp0IsChain =
        node.inputs[0].kind === "pending" && node.inputs[0].node === prevNode;
      const externalPos = inp0IsChain ? 1 : 0;
      const chainPos: 0 | 1 = inp0IsChain ? 0 : 1;

      const extRef = node.inputs[externalPos];
      const refShape = getRefShape(extRef);
      if (refShape && !shapesEqual(refShape, firstShape)) return null;

      externalInputRefs.push(extRef);
      externalInputDtypes.push(getRefDtype(extRef));
      chainOps.push({
        op: getEpilogueOpName(node),
        arity: 2,
        chainInputPos: chainPos,
      });
    }
  }

  // Validate: chain output must be f32 (for accumulator)
  const lastChainNode = chain[chain.length - 1];
  if (lastChainNode.dtype !== "f32") return null;

  return {
    chain,
    chainOps,
    chainInputRefs: externalInputRefs,
    chainInputDtypes: externalInputDtypes,
    reductionNode,
    reductionIdx,
  };
}

// ============================================================================
// Epilogue Detection (reduction → elementwise)
// ============================================================================

/**
 * Walk forward from a reduction node collecting fusible elementwise ops.
 * Returns the epilogue chain or null if no pattern found.
 */
function detectEpilogue(
  nodes: LazyIRNode[],
  reductionIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): {
  chain: LazyIRNode[];
  epilogueOps: EpilogueOp[];
  epilogueInputRefs: LazyRef[];
  outputDtype: DType;
  outputNode: LazyIRNode;
} | null {
  const reductionNode = nodes[reductionIdx];
  const epilogueOps: EpilogueOp[] = [];
  const epilogueInputRefs: LazyRef[] = [];
  let additionalInputCount = 0;
  let chainLength = 0;
  let currentNode = reductionNode;
  let outputDtype = reductionNode.dtype;
  const chain: LazyIRNode[] = [];

  for (let i = reductionIdx + 1; i < nodes.length && chainLength < 4; i++) {
    const nextNode = nodes[i];
    if (nextNode.inputs.length === 0) break;

    const chainInputIdx = findChainInput(nextNode, currentNode.id);
    if (chainInputIdx < 0) break;

    if (externalNodeIds?.has(currentNode.id)) break;
    const consumers = consumerCount.get(currentNode.id) ?? 0;
    if (consumers > 1) break;

    let matched = false;

    if (nextNode.op === "cast") {
      const payload = nextNode.payload as { dtype: DType } | undefined;
      if (payload) {
        epilogueOps.push({ kind: "cast", toDtype: payload.dtype });
        outputDtype = payload.dtype;
        matched = true;
      }
    } else if (
      (nextNode.op === "add" || nextNode.op === "mul") &&
      nextNode.inputs.length === 2
    ) {
      if (additionalInputCount >= 4) break;
      const secondInput = nextNode.inputs[chainInputIdx === 0 ? 1 : 0];
      const secondShape = getRefShape(secondInput);
      if (secondShape && !shapesEqual(secondShape, currentNode.shape)) break;

      epilogueOps.push({
        kind: "binary",
        op: nextNode.op,
        inputIndex: additionalInputCount,
      });
      epilogueInputRefs.push(secondInput);
      additionalInputCount++;
      matched = true;
    } else if (
      isFusibleOp(nextNode.op) &&
      nextNode.inputs.length === 1 &&
      (nextNode.op as string) !== "cast"
    ) {
      epilogueOps.push({ kind: "unary", op: getEpilogueOpName(nextNode) });
      matched = true;
    }

    if (!matched) break;

    chainLength++;
    chain.push(nextNode);
    currentNode = nextNode;
    outputDtype = nextNode.dtype || outputDtype;
  }

  if (chainLength === 0) return null;

  return {
    chain,
    epilogueOps,
    epilogueInputRefs,
    outputDtype,
    outputNode: currentNode,
  };
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Detect reduction groups from a plan.
 *
 * Scans for three patterns:
 *  1. Combined preamble+epilogue: [elem] → sum/mean → [elem]
 *  2. Preamble only: [elem] → sum/mean
 *  3. Epilogue only: sum/mean/max → [elem]
 *
 * @returns Detected groups and the set of claimed node IDs.
 */
export function detectReductionGroups(
  nodes: LazyIRNode[],
  consumerCount: Map<number, number>,
  alreadyClaimedIds: Set<number>,
  externalNodeIds?: Set<number>,
): { groups: ReductionGroup[]; claimedIds: Set<number> } {
  const groups: ReductionGroup[] = [];
  const claimedIds = new Set<number>();

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    if (claimedIds.has(node.id)) continue;
    if (alreadyClaimedIds.has(node.id)) continue;

    // --- Try preamble-first patterns (starts at fusible op) ---
    if (isFusibleOp(node.op)) {
      const preamble = detectPreamble(nodes, i, consumerCount);
      if (preamble) {
        // Only support sum/mean for combined fusion
        const canHaveEpilogue =
          preamble.reductionNode.op === "sum" ||
          preamble.reductionNode.op === "mean";

        // Try combined preamble+epilogue
        let epilogue: ReturnType<typeof detectEpilogue> = null;
        if (canHaveEpilogue) {
          epilogue = detectEpilogue(
            nodes,
            preamble.reductionIdx,
            consumerCount,
            externalNodeIds,
          );
        }

        if (epilogue) {
          // Combined preamble + epilogue
          const allNodes = [
            ...preamble.chain,
            preamble.reductionNode,
            ...epilogue.chain,
          ];
          const startPlanIdx = i;
          const planIndices: number[] = [];
          for (let j = 0; j < allNodes.length; j++) {
            planIndices.push(startPlanIdx + j);
          }

          groups.push({
            nodes: allNodes,
            planIndices,
            reductionNode: preamble.reductionNode,
            preambleNodes: preamble.chain,
            epilogueNodes: epilogue.chain,
            outputNode: epilogue.outputNode,
            preambleOps: preamble.chainOps,
            preambleInputRefs: preamble.chainInputRefs,
            preambleInputDtypes: preamble.chainInputDtypes,
            epilogueOps: epilogue.epilogueOps,
            epilogueInputRefs: epilogue.epilogueInputRefs,
            outputDtype: epilogue.outputDtype,
            isMean: preamble.reductionNode.op === "mean",
          });

          for (const n of preamble.chain) claimedIds.add(n.id);
          for (const n of epilogue.chain) claimedIds.add(n.id);
          // Don't claim the reduction node — it's the anchor in the plan
          continue;
        }

        // Preamble-only
        const allNodes = [...preamble.chain, preamble.reductionNode];
        const planIndices: number[] = [];
        for (let j = 0; j < allNodes.length; j++) {
          planIndices.push(i + j);
        }

        groups.push({
          nodes: allNodes,
          planIndices,
          reductionNode: preamble.reductionNode,
          preambleNodes: preamble.chain,
          epilogueNodes: [],
          outputNode: preamble.reductionNode,
          preambleOps: preamble.chainOps,
          preambleInputRefs: preamble.chainInputRefs,
          preambleInputDtypes: preamble.chainInputDtypes,
          epilogueOps: [],
          epilogueInputRefs: [],
          outputDtype: preamble.reductionNode.dtype,
          isMean: preamble.reductionNode.op === "mean",
        });

        for (const n of preamble.chain) claimedIds.add(n.id);
        continue;
      }
    }

    // --- Try epilogue-only (starts at reduction node) ---
    if (node.op === "sum" || node.op === "mean" || node.op === "max") {
      const epilogue = detectEpilogue(nodes, i, consumerCount, externalNodeIds);
      if (epilogue) {
        const allNodes = [node, ...epilogue.chain];
        const planIndices: number[] = [];
        for (let j = 0; j < allNodes.length; j++) {
          planIndices.push(i + j);
        }

        groups.push({
          nodes: allNodes,
          planIndices,
          reductionNode: node,
          preambleNodes: [],
          epilogueNodes: epilogue.chain,
          outputNode: epilogue.outputNode,
          preambleOps: [],
          preambleInputRefs: [],
          preambleInputDtypes: [],
          epilogueOps: epilogue.epilogueOps,
          epilogueInputRefs: epilogue.epilogueInputRefs,
          outputDtype: epilogue.outputDtype,
          isMean: node.op === "mean",
        });

        // Claim epilogue chain nodes (not the reduction itself — it's the anchor)
        for (const n of epilogue.chain) claimedIds.add(n.id);
      }
    }
  }

  return { groups, claimedIds };
}
