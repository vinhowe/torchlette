import type { Backend, BackendTensor } from "../backend/types";
import type { LazyIRNode } from "./lazy-types";
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
      reductionSize = dims.reduce((acc, d) => acc * inputShape[d < 0 ? d + rank : d], 1);
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
