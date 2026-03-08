import {
  type Backend,
  type BackendTensor,
  type DType,
  normalizeDim,
} from "../backend/types";
import type { PreambleChainKernelOp } from "../backend/webgpu/reduction-tile-ir";
import { shapesEqual } from "../core/shape";
import { isFusibleOp } from "./fusion-detect";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import {
  type EpilogueOp,
  findChainInput,
  getEpilogueOpName as getChainOpName,
} from "./matmul-epilogue";
import { createStorageHandle } from "./node-factory";
import { getInputStorage } from "./op-dispatch";

/** Reduction payload shape shared across sum/mean/max. */
type ReductionPayload = { dim?: number | number[] | null; keepdim?: boolean };

// ============================================================================
// Reduction Preamble Fusion (Phase 3)
// Detects elementwise → sum/mean patterns and fuses them into a single
// sumDimWithPreamble call, eliminating the intermediate elementwise buffer.
// ============================================================================

export interface ReductionPreamblePlan {
  /** The elementwise preamble node (first in chain) */
  preambleNode: LazyIRNode;
  /** The sum or mean reduction node */
  reductionNode: LazyIRNode;
  /** Whether the reduction is a mean (divide by count after sum) */
  isMean: boolean;
  /** Total nodes consumed (chain length + 1 for reduction) */
  consumedCount: number;
  /** All preamble nodes in chain order */
  preambleChain: LazyIRNode[];
  /** Kernel op descriptors for the chain */
  chainOps: PreambleChainKernelOp[];
  /** External input refs for the entire chain */
  chainInputRefs: LazyRef[];
  /** Dtypes for each external input */
  chainInputDtypes: DType[];
}

/** Extract the shape from a LazyRef (pending or materialized). */
function getRefShape(ref: LazyRef): number[] | undefined {
  if (ref.kind === "pending") return ref.node.shape;
  if (ref.kind === "materialized") return ref.storage.backendTensor.shape;
  return undefined;
}

/** Get the dtype of a LazyRef's backing tensor. */
function getRefDtype(ref: LazyRef): DType {
  if (ref.kind === "pending") return ref.node.dtype;
  if (ref.kind === "materialized") {
    return (ref.storage.backendTensor as { dtype?: DType }).dtype || "f32";
  }
  return "f32";
}

/**
 * Detect an elementwise chain → sum/mean pattern suitable for reduction preamble fusion.
 *
 * Walks forward from startIdx collecting fusible elementwise ops (max depth 4)
 * until a sum/mean reduction is found that consumes the last chain node.
 *
 * Single-op chains use the existing fast path. Multi-op chains (e.g., cast → mul → sum)
 * generate a fused kernel that applies the entire chain in the accumulation loop body.
 *
 * Constraints:
 * - All chain nodes must be fusible, unary/binary (not ternary), no existing result
 * - Each intermediate node must have exactly 1 consumer (the next chain/reduction node)
 * - All external inputs must have the same shape as the first node's output
 * - The final chain output dtype must be f32 (for accumulator)
 * - Single-op chains exclude standalone cast (no benefit)
 */
export function detectReductionPreamble(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
): ReductionPreamblePlan | null {
  if (startIdx + 1 >= nodes.length) return null;

  const firstNode = nodes[startIdx];
  if (!isFusibleOp(firstNode.op)) return null;
  if (firstNode.inputs.length > 2) return null;
  if (firstNode.result) return null;

  // Build chain of fusible elem ops ending at a sum/mean
  const chain: LazyIRNode[] = [firstNode];
  let reductionNode: LazyIRNode | null = null;
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
        // Validate last chain node has exactly 1 consumer
        if ((consumerCount.get(lastChainNode.id) ?? 0) !== 1) return null;
        reductionNode = nextNode;
        break;
      }
      break; // Reduction doesn't consume our chain
    }

    // Check if nextNode can extend the chain
    if (!isFusibleOp(nextNode.op)) break;
    if (nextNode.inputs.length > 2) break;
    if (nextNode.result) break;

    // Check that nextNode consumes the last chain node
    const consumesLast = nextNode.inputs.some(
      (inp) => inp.kind === "pending" && inp.node === lastChainNode,
    );
    if (!consumesLast) break;

    // Check single consumer on last chain node
    if ((consumerCount.get(lastChainNode.id) ?? 0) !== 1) break;

    chain.push(nextNode);
  }

  if (!reductionNode) return null;

  // Single-op chain: use existing single-op constraints
  if (chain.length === 1) {
    if (firstNode.op === "cast") return null; // Cast alone doesn't benefit

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
      preambleNode: firstNode,
      reductionNode,
      isMean: reductionNode.op === "mean",
      consumedCount: 2,
      preambleChain: [firstNode],
      chainOps: [{ op: firstNode.op, arity: firstNode.inputs.length }],
      chainInputRefs: [...firstNode.inputs],
      chainInputDtypes: firstNode.inputs.map(getRefDtype),
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
    op: getChainOpName(firstNode),
    arity: firstNode.inputs.length,
  });

  // Subsequent nodes: identify chain input vs external input
  for (let i = 1; i < chain.length; i++) {
    const node = chain[i];
    const prevNode = chain[i - 1];

    if (node.inputs.length === 1) {
      chainOps.push({ op: getChainOpName(node), arity: 1 });
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
        op: getChainOpName(node),
        arity: 2,
        chainInputPos: chainPos,
      });
    }
  }

  // Validate: chain output must be f32 (for accumulator)
  const lastChainNode = chain[chain.length - 1];
  if (lastChainNode.dtype !== "f32") return null;

  return {
    preambleNode: firstNode,
    reductionNode,
    isMean: reductionNode.op === "mean",
    consumedCount: chain.length + 1,
    preambleChain: chain,
    chainOps,
    chainInputRefs: externalInputRefs,
    chainInputDtypes: externalInputDtypes,
  };
}

/**
 * Execute a fused reduction-with-preamble operation.
 * Handles both single-op preambles (existing path) and multi-op chains.
 */
export async function executeReductionWithPreamble(
  plan: ReductionPreamblePlan,
  backend: Backend,
): Promise<void> {
  // Get sum options from the reduction node's payload
  const payload = plan.reductionNode.payload as ReductionPayload | undefined;

  const { sumDimWithPreambleChain } = await import("../backend/webgpu/index");
  const inputStorages = plan.chainInputRefs.map((ref) =>
    getInputStorage(ref, backend),
  );
  const inputTensors = inputStorages.map((s) => s.backendTensor);
  let resultTensor: BackendTensor = sumDimWithPreambleChain(
    inputTensors,
    plan.chainOps,
    plan.chainInputDtypes,
    payload ?? {},
  );

  // If this is a mean, divide by reduction size
  if (plan.isMean) {
    const inputShape = plan.preambleNode.shape;
    const dim = payload?.dim;
    let reductionSize: number;
    if (dim == null) {
      reductionSize = inputShape.reduce((a, b) => a * b, 1);
    } else {
      const dims = Array.isArray(dim) ? dim : [dim];
      const rank = inputShape.length;
      reductionSize = dims.reduce(
        (acc, d) => acc * inputShape[normalizeDim(d, rank)],
        1,
      );
    }
    const invSize = 1.0 / reductionSize;
    const sumResult = resultTensor;
    const invSizeTensor = backend.ops.full
      ? backend.ops.full([], invSize)
      : backend.ops.tensorFromArray([invSize], []);
    resultTensor = backend.ops.mul(sumResult, invSizeTensor);
    (sumResult as { destroy?: () => void }).destroy?.();
    (invSizeTensor as { destroy?: () => void }).destroy?.();
  }

  // Store result on the reduction node
  plan.reductionNode.result = createStorageHandle(
    plan.reductionNode.device,
    resultTensor,
  );
}

// ============================================================================
// Reduction Epilogue Fusion (Phase 4)
// Detects sum/mean/max → elementwise chain patterns and fuses them into a
// single kernel, eliminating intermediate buffers between reduction and
// subsequent elementwise ops. Mirrors matmul epilogue detection.
// ============================================================================

/** Format an array of epilogue ops into a profile label fragment (e.g. "cast+add+relu"). */
export function formatEpilogueLabel(ops: EpilogueOp[]): string {
  return ops
    .map((o) =>
      o.kind === "binary" ? o.op : o.kind === "cast" ? "cast" : o.op || o.kind,
    )
    .join("+");
}

export interface ReductionEpiloguePlan {
  /** The reduction node (sum, mean, or max) */
  reductionNode: LazyIRNode;
  /** Epilogue operations to fuse after the reduction */
  epilogueOps: EpilogueOp[];
  /** Additional inputs required by epilogue binary ops */
  epilogueInputRefs: LazyRef[];
  /** Output dtype after epilogue chain */
  outputDtype: DType;
  /** The final output node in the chain */
  outputNode: LazyIRNode;
  /** Total nodes consumed (reduction + epilogue ops) */
  consumedCount: number;
}

/**
 * Detect a reduction → elementwise chain pattern suitable for epilogue fusion.
 *
 * Starting from a sum/mean/max node, walks forward through single-consumer
 * elementwise ops that can be applied in-register after the reduction.
 */
export function detectReductionEpilogue(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): ReductionEpiloguePlan | null {
  const reductionNode = nodes[startIdx];
  if (
    reductionNode.op !== "sum" &&
    reductionNode.op !== "mean" &&
    reductionNode.op !== "max"
  ) {
    return null;
  }

  const epilogueOps: EpilogueOp[] = [];
  const epilogueInputRefs: LazyRef[] = [];
  let additionalInputCount = 0;
  let chainLength = 0;
  let currentNode = reductionNode;
  let outputDtype = reductionNode.dtype;

  for (let i = startIdx + 1; i < nodes.length && chainLength < 4; i++) {
    const nextNode = nodes[i];
    if (nextNode.inputs.length === 0) break;

    // Check that the candidate's input continues the chain from currentNode
    const chainInputIdx = findChainInput(nextNode, currentNode.id);
    if (chainInputIdx < 0) break;

    // Current chain node must have exactly 1 consumer and not be external
    if (externalNodeIds?.has(currentNode.id)) break;
    const consumers = consumerCount.get(currentNode.id) ?? 0;
    if (consumers > 1) break;

    // Match the op type (same patterns as matmul epilogue)
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
      // The external input must have the same shape as the reduction output
      // (it gets indexed at the output position)
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
      nextNode.op !== "cast"
    ) {
      epilogueOps.push({ kind: "unary", op: getEpilogueOpName(nextNode) });
      matched = true;
    }

    if (!matched) break;

    chainLength++;
    currentNode = nextNode;
    outputDtype = nextNode.dtype || outputDtype;
  }

  if (chainLength === 0) return null;

  return {
    reductionNode,
    epilogueOps,
    epilogueInputRefs,
    outputDtype,
    outputNode: currentNode,
    consumedCount: 1 + chainLength,
  };
}

/**
 * Execute a fused reduction-with-epilogue operation.
 */
export async function executeReductionWithEpilogue(
  plan: ReductionEpiloguePlan,
  backend: Backend,
): Promise<void> {
  const { reduction, meanWithEpilogue } = await import(
    "../backend/webgpu/index"
  );

  // Resolve reduction input
  const reductionInputStorage = getInputStorage(
    plan.reductionNode.inputs[0],
    backend,
  );
  const reductionInputTensor = reductionInputStorage.backendTensor;

  // Resolve epilogue external inputs
  const epilogueInputTensors = plan.epilogueInputRefs.map(
    (ref) => getInputStorage(ref, backend).backendTensor,
  );

  const payload = plan.reductionNode.payload as ReductionPayload | undefined;

  // Mean has special handling (prepends mul(1/count) to epilogue), others go directly
  const resultTensor =
    plan.reductionNode.op === "mean"
      ? meanWithEpilogue(
          reductionInputTensor,
          payload ?? {},
          plan.epilogueOps,
          epilogueInputTensors,
          plan.outputDtype,
        )
      : reduction(
          plan.reductionNode.op as "sum" | "max",
          reductionInputTensor,
          payload ?? {},
          plan.epilogueOps,
          epilogueInputTensors,
          plan.outputDtype,
        );

  // Store result on the FINAL output node (not the reduction node)
  plan.outputNode.result = createStorageHandle(
    plan.outputNode.device,
    resultTensor,
  );
}

// ============================================================================
// Combined Preamble + Epilogue Fusion (Phase 5)
// Detects elem → reduction → elem patterns and fuses them into a single
// kernel that applies the preamble chain in the accumulation loop and the
// epilogue chain on the result.
// ============================================================================

export interface ReductionFusionPlan {
  /** All preamble nodes in chain order */
  preambleChain: LazyIRNode[];
  /** The reduction node (sum or mean) */
  reductionNode: LazyIRNode;
  /** Epilogue chain nodes (after reduction) */
  epilogueChain: LazyIRNode[];
  /** The final output node (last epilogue node, or reduction if no epilogue) */
  outputNode: LazyIRNode;
  /** Whether the reduction is a mean */
  isMean: boolean;
  /** Kernel op descriptors for the preamble chain */
  preambleOps: PreambleChainKernelOp[];
  /** External input refs for the preamble chain */
  preambleInputRefs: LazyRef[];
  /** Dtypes for each preamble external input */
  preambleInputDtypes: DType[];
  /** Epilogue operations to apply after the reduction */
  epilogueOps: EpilogueOp[];
  /** External input refs for the epilogue binary ops */
  epilogueInputRefs: LazyRef[];
  /** Output dtype after epilogue chain */
  outputDtype: DType;
  /** Total nodes consumed (preamble chain + reduction + epilogue chain) */
  consumedCount: number;
}

/**
 * Detect a combined [elem chain] → reduction → [elem chain] pattern.
 *
 * First calls detectReductionPreamble to find the preamble→reduction.
 * Then, from the reduction node, walks forward to find an epilogue chain
 * (same logic as detectReductionEpilogue but starting from the preamble's
 * reduction node position).
 *
 * Returns null if:
 * - No preamble is found (caller should try standalone epilogue)
 * - A preamble is found but no epilogue (caller should use preamble-only)
 */
export function detectReductionFusion(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): ReductionFusionPlan | null {
  // First, detect the preamble
  const preamblePlan = detectReductionPreamble(nodes, startIdx, consumerCount);
  if (!preamblePlan) return null;

  // Find the reduction node's position in the node array
  const reductionIdx = startIdx + preamblePlan.consumedCount - 1;
  if (reductionIdx >= nodes.length) return null;
  if (nodes[reductionIdx] !== preamblePlan.reductionNode) return null;

  // Only support sum/mean for combined fusion (not max — max has different accumulator semantics)
  if (
    preamblePlan.reductionNode.op !== "sum" &&
    preamblePlan.reductionNode.op !== "mean"
  ) {
    return null;
  }

  // Try to detect an epilogue starting from the reduction node
  const epiloguePlan = detectReductionEpilogue(
    nodes,
    reductionIdx,
    consumerCount,
    externalNodeIds,
  );
  if (!epiloguePlan) return null;

  // Collect epilogue chain nodes
  const epilogueChain: LazyIRNode[] = [];
  for (let i = 1; i < epiloguePlan.consumedCount; i++) {
    epilogueChain.push(nodes[reductionIdx + i]);
  }

  return {
    preambleChain: preamblePlan.preambleChain,
    reductionNode: preamblePlan.reductionNode,
    epilogueChain,
    outputNode: epiloguePlan.outputNode,
    isMean: preamblePlan.isMean,
    preambleOps: preamblePlan.chainOps,
    preambleInputRefs: preamblePlan.chainInputRefs,
    preambleInputDtypes: preamblePlan.chainInputDtypes,
    epilogueOps: epiloguePlan.epilogueOps,
    epilogueInputRefs: epiloguePlan.epilogueInputRefs,
    outputDtype: epiloguePlan.outputDtype,
    consumedCount: preamblePlan.consumedCount + epiloguePlan.consumedCount - 1, // -1: reduction counted in both
  };
}

/**
 * Execute a combined preamble + epilogue reduction fusion.
 *
 * For mean: the dispatch function internally prepends mul(1/count) to the
 * epilogue chain with a scalar buffer, matching the meanWithEpilogue pattern.
 */
export async function executeReductionWithFusion(
  plan: ReductionFusionPlan,
  backend: Backend,
): Promise<void> {
  const { sumWithPreambleEpilogue } = await import("../backend/webgpu/index");

  // Resolve preamble input storages
  const preambleInputTensors = plan.preambleInputRefs.map(
    (ref) => getInputStorage(ref, backend).backendTensor,
  );

  // Resolve epilogue input storages
  const epilogueInputTensors = plan.epilogueInputRefs.map(
    (ref) => getInputStorage(ref, backend).backendTensor,
  );

  const payload = plan.reductionNode.payload as ReductionPayload | undefined;

  const resultTensor = sumWithPreambleEpilogue(
    preambleInputTensors,
    plan.preambleOps,
    plan.preambleInputDtypes,
    plan.epilogueOps,
    epilogueInputTensors,
    plan.outputDtype,
    payload ?? {},
    plan.isMean,
  );

  // Store result on the FINAL output node
  plan.outputNode.result = createStorageHandle(
    plan.outputNode.device,
    resultTensor,
  );
}
