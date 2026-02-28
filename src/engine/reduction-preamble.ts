import { normalizeDim, type Backend, type BackendTensor, type DType } from "../backend/types";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import { createStorageHandle } from "./node-factory";
import { getInputStorage } from "./op-dispatch";
import { shapesEqual } from "./matmul-epilogue";
import { isFusibleOp } from "./fusion-detect";

// ============================================================================
// Reduction Preamble Fusion (Phase 3)
// Detects elementwise → sum/mean patterns and fuses them into a single
// sumDimWithPreamble call, eliminating the intermediate elementwise buffer.
// ============================================================================

export interface ReductionPreamblePlan {
  /** The elementwise preamble node */
  preambleNode: LazyIRNode;
  /** The sum or mean reduction node */
  reductionNode: LazyIRNode;
  /** The elementwise op name (e.g., "mul", "exp", "add") */
  op: string;
  /** Number of inputs to the elementwise op (1=unary, 2=binary) */
  arity: number;
  /** Whether the reduction is a mean (divide by count after sum) */
  isMean: boolean;
}

/**
 * Detect an elementwise → sum/mean pattern suitable for reduction preamble fusion.
 *
 * Constraints:
 * - nodes[startIdx] must be a fusible elementwise op (unary or binary, not ternary)
 * - nodes[startIdx + 1] must be "sum" or "mean"
 * - The reduction's input must be a pending ref to the elementwise node
 * - The elementwise node must not be referenced by anything else
 * - All elementwise inputs must have the same shape as the elementwise output
 * - All inputs must be f32 dtype (preamble shader hardcodes f32)
 */
export function detectReductionPreamble(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
): ReductionPreamblePlan | null {
  if (startIdx + 1 >= nodes.length) return null;

  const elemNode = nodes[startIdx];
  const nextNode = nodes[startIdx + 1];

  // 1. Check elementwise op is fusible and unary/binary (not ternary)
  if (!isFusibleOp(elemNode.op)) return null;
  if (elemNode.inputs.length > 2) return null; // Skip ternary (where)
  if (elemNode.op === "cast") return null; // Cast doesn't benefit from fusion here

  // 2. Check next node is sum or mean
  if (nextNode.op !== "sum" && nextNode.op !== "mean") return null;

  // 3. Check the reduction's primary input is a pending ref to the elementwise node
  if (nextNode.inputs.length < 1) return null;
  const reductionInput = nextNode.inputs[0];
  if (reductionInput.kind !== "pending" || reductionInput.node !== elemNode) return null;

  // 4. Check the elementwise output is only consumed by the reduction
  const consumers = consumerCount.get(elemNode.id) ?? 0;
  if (consumers !== 1) return null;

  // 5. All elementwise inputs must have the same shape as the elementwise output
  const elemShape = elemNode.shape;
  for (const ref of elemNode.inputs) {
    const inputNode = ref.kind === "pending" ? ref.node : null;
    if (inputNode && !shapesEqual(inputNode.shape, elemShape)) return null;
    // For materialized refs, we can't easily check shape at detection time.
    // The backend tensor will have the correct shape, so we rely on the
    // constraint that the lazy engine produces correct shapes.
  }

  // 6. All inputs must be f32 dtype (preamble shader hardcodes f32)
  if (elemNode.dtype !== "f32") return null;
  for (const ref of elemNode.inputs) {
    if (ref.kind === "pending" && ref.node.dtype !== "f32") return null;
  }

  // 7. The elementwise node must not already have a result
  if (elemNode.result) return null;

  const arity = elemNode.inputs.length;

  return {
    preambleNode: elemNode,
    reductionNode: nextNode,
    op: elemNode.op,
    arity,
    isMean: nextNode.op === "mean",
  };
}

/**
 * Execute a fused reduction-with-preamble operation.
 */
export async function executeReductionWithPreamble(
  plan: ReductionPreamblePlan,
  backend: Backend,
): Promise<void> {
  const { sumDimWithPreamble } = await import("../backend/webgpu/index");

  // Resolve all elementwise inputs
  const elemInputStorages = plan.preambleNode.inputs.map(ref => getInputStorage(ref, backend));
  const elemInputTensors = elemInputStorages.map(s => s.backendTensor);

  // Get sum options from the reduction node's payload
  const payload = plan.reductionNode.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;

  // Call sumDimWithPreamble
  let resultTensor = sumDimWithPreamble(elemInputTensors, plan.op, payload ?? {});

  // If this is a mean, divide by reduction size
  if (plan.isMean) {
    const inputShape = plan.preambleNode.shape;
    const dim = payload?.dim;
    let reductionSize: number;
    if (dim === undefined || dim === null) {
      reductionSize = inputShape.reduce((a, b) => a * b, 1);
    } else {
      const dims = Array.isArray(dim) ? dim : [dim];
      const rank = inputShape.length;
      reductionSize = dims.reduce((acc, d) => acc * inputShape[normalizeDim(d, rank)], 1);
    }
    // Divide by reduction size using backend mul with scalar (1/reductionSize)
    const invSize = 1.0 / reductionSize;
    const sumResult = resultTensor;
    const invSizeTensor = backend.ops.full ? backend.ops.full([], invSize) : backend.ops.tensorFromArray([invSize], []);
    resultTensor = backend.ops.mul(sumResult, invSizeTensor);
    // Destroy intermediate backend tensors (sum output + scalar) to prevent buffer leak
    (sumResult as { destroy?: () => void }).destroy?.();
    (invSizeTensor as { destroy?: () => void }).destroy?.();
  }

  // Store result on the reduction node
  plan.reductionNode.result = createStorageHandle(plan.reductionNode.device, resultTensor);
}

// ============================================================================
// Reduction Epilogue Fusion (Phase 4)
// Detects sum/mean/max → elementwise chain patterns and fuses them into a
// single kernel, eliminating intermediate buffers between reduction and
// subsequent elementwise ops. Mirrors matmul epilogue detection.
// ============================================================================

/** Describes a single epilogue op to apply after the reduction. */
export type ReductionEpilogueOp = {
  kind: string;
  toDtype?: DType;
  inputIndex?: number;
  op?: string;
};

export interface ReductionEpiloguePlan {
  /** The reduction node (sum, mean, or max) */
  reductionNode: LazyIRNode;
  /** Epilogue operations to fuse after the reduction */
  epilogueOps: ReductionEpilogueOp[];
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
  if (reductionNode.op !== "sum" && reductionNode.op !== "mean" && reductionNode.op !== "max") {
    return null;
  }

  const epilogueOps: ReductionEpilogueOp[] = [];
  const epilogueInputRefs: LazyRef[] = [];
  let additionalInputCount = 0;
  let chainLength = 0;
  let currentNode = reductionNode;
  let outputDtype = reductionNode.dtype;

  for (let i = startIdx + 1; i < nodes.length && chainLength < 4; i++) {
    const nextNode = nodes[i];
    if (nextNode.inputs.length === 0) break;

    // Check that the candidate's input is a pending ref to the current chain node
    let chainInputIdx = 0;
    const primaryInput = nextNode.inputs[0];
    if (primaryInput.kind !== "pending" || primaryInput.node.id !== currentNode.id) {
      // For commutative binary ops, check inputs[1]
      if ((nextNode.op === "add" || nextNode.op === "mul") && nextNode.inputs.length === 2) {
        const altInput = nextNode.inputs[1];
        if (altInput.kind === "pending" && altInput.node.id === currentNode.id) {
          chainInputIdx = 1;
        } else {
          break;
        }
      } else {
        break;
      }
    }

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
    } else if ((nextNode.op === "add" || nextNode.op === "mul") && nextNode.inputs.length === 2) {
      if (additionalInputCount >= 4) break;
      const secondInput = nextNode.inputs[chainInputIdx === 0 ? 1 : 0];
      // The external input must have the same shape as the reduction output
      // (it gets indexed at the output position)
      let secondShape: number[] | undefined;
      if (secondInput.kind === "materialized") {
        secondShape = secondInput.storage.backendTensor.shape;
      } else if (secondInput.kind === "pending") {
        secondShape = secondInput.node.shape;
      }
      if (secondShape && !shapesEqual(secondShape, currentNode.shape)) break;

      epilogueOps.push({ kind: "binary", op: nextNode.op, inputIndex: additionalInputCount });
      epilogueInputRefs.push(secondInput);
      additionalInputCount++;
      matched = true;
    } else if (
      nextNode.op === "relu" || nextNode.op === "silu" || nextNode.op === "sigmoid" ||
      nextNode.op === "tanh" || nextNode.op === "neg" || nextNode.op === "abs" ||
      nextNode.op === "exp" || nextNode.op === "log" || nextNode.op === "sqrt"
    ) {
      epilogueOps.push({ kind: "unary", op: nextNode.op });
      matched = true;
    } else if (nextNode.op === "gelu") {
      const geluPayload = nextNode.payload as { approximate?: string } | undefined;
      if (geluPayload?.approximate === "tanh") {
        epilogueOps.push({ kind: "gelu" });
      } else {
        epilogueOps.push({ kind: "unary", op: "gelu_erf" });
      }
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
  const { sumWithEpilogue, maxWithEpilogue, meanWithEpilogue } = await import("../backend/webgpu/index");

  // Resolve reduction input
  const reductionInputStorage = getInputStorage(plan.reductionNode.inputs[0], backend);
  const reductionInputTensor = reductionInputStorage.backendTensor;

  // Resolve epilogue external inputs
  const epilogueInputTensors = plan.epilogueInputRefs.map(
    ref => getInputStorage(ref, backend).backendTensor,
  );

  const payload = plan.reductionNode.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;

  let resultTensor: BackendTensor;
  if (plan.reductionNode.op === "max") {
    resultTensor = maxWithEpilogue(
      reductionInputTensor, payload ?? {},
      plan.epilogueOps, epilogueInputTensors, plan.outputDtype,
    );
  } else if (plan.reductionNode.op === "mean") {
    resultTensor = meanWithEpilogue(
      reductionInputTensor, payload ?? {},
      plan.epilogueOps, epilogueInputTensors, plan.outputDtype,
    );
  } else {
    resultTensor = sumWithEpilogue(
      reductionInputTensor, payload ?? {},
      plan.epilogueOps, epilogueInputTensors, plan.outputDtype,
    );
  }

  // Store result on the FINAL output node (not the reduction node)
  plan.outputNode.result = createStorageHandle(plan.outputNode.device, resultTensor);
}
