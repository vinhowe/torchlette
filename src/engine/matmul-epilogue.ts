import type { Backend, BackendTensor, DType } from "../backend/types";
import { asGPUTensor, type WebGPUTensor } from "../backend/webgpu/gpu-types";
import type { LazyIRNode, LazyRef, StorageHandle } from "./lazy-types";
import { createStorageHandle, ensureWebGPUMatmulImports, _webgpuMatmulImports } from "./node-factory";
import { getInputStorage } from "./op-dispatch";
import { shapesEqual } from "../core/shape";
export { shapesEqual };

// ============================================================================
// Matmul Epilogue Fusion (Phase 1)
// Detects matmul → cast/bias/activation chains and fuses them into a single
// dispatchMatmulWithEpilogue call, eliminating intermediate buffers.
// ============================================================================

export interface MatmulPrologueInfo {
  /** Which matmul input has the cast (0 = A, 1 = B) */
  inputIndex: 0 | 1;
  /** The cast node ID (for tracking) */
  castNodeId: number;
  /** The cast's input ref (the original f32 tensor) */
  originalInputRef: LazyRef;
  /** Source dtype of the cast input (e.g. "f32") */
  fromDtype: DType;
  /** Target dtype the matmul expects (e.g. "f16") */
  toDtype: DType;
}

export interface MatmulEpiloguePlan {
  /** Number of nodes consumed (matmul + epilogue ops) */
  consumedCount: number;
  /** Epilogue operations to fuse */
  epilogueOps: Array<{ kind: string; toDtype?: DType; inputIndex?: number; op?: string }>;
  /** Additional inputs required by epilogue (e.g. bias tensor) */
  epilogueInputRefs: LazyRef[];
  /** Output dtype after epilogue */
  outputDtype: DType;
  /** The final output node in the chain */
  outputNode: LazyIRNode;
  /** Input-side cast prologues absorbed into the matmul */
  prologues?: MatmulPrologueInfo[];
}

/**
 * Core matmul epilogue chain detection logic.
 * Starting from a matmul node at `startIdx` in `nodes`, walks forward to find
 * a chain of fusible ops (cast, bias add, activations) that can be merged
 * into a single matmul dispatch with epilogue.
 *
 * @param nodes - Array of nodes to walk (full plan or segment)
 * @param startIdx - Index of the matmul node
 * @param consumerCount - Pre-computed map of node ID → number of consumers
 * @param externalNodeIds - Node IDs with external references (saved-for-backward etc.)
 */
export function detectMatmulEpilogueCore(
  nodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): MatmulEpiloguePlan | null {
  const matmulNode = nodes[startIdx];
  if (matmulNode.op !== "matmul") return null;

  const epilogueOps: MatmulEpiloguePlan["epilogueOps"] = [];
  const epilogueInputRefs: LazyRef[] = [];
  let additionalInputCount = 0;
  let chainLength = 0;
  let currentNode = matmulNode;
  let outputDtype = matmulNode.dtype;

  // Walk forward from matmul, matching epilogue-compatible ops
  let currentIsReshapeSkip = false;
  for (let i = startIdx + 1; i < nodes.length && chainLength < 4; i++) {
    const nextNode = nodes[i];

    // Check that the candidate node's primary input (input[0]) is a pending ref
    // to the current chain node
    if (nextNode.inputs.length === 0) break;
    let chainInputIdx = 0;
    const primaryInput = nextNode.inputs[0];
    if (primaryInput.kind !== "pending" || primaryInput.node.id !== currentNode.id) {
      // For commutative binary ops, check if the chain continues via inputs[1]
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

    // Check that the current chain node is NOT externally referenced
    // (must be used only by this next node).
    // Skip this check for nodes already absorbed into the chain as reshape
    // skips — their live pending tensors will be disposed by the caller.
    if (externalNodeIds?.has(currentNode.id) && !currentIsReshapeSkip) break;
    const consumers = consumerCount.get(currentNode.id) ?? 0;
    if (consumers > 1) break;

    // Skip reshape that only removes leading size-1 dims (e.g. [1,M,N]→[M,N])
    if (nextNode.op === "reshape") {
      const curShape = currentNode.shape;
      const nextShape = nextNode.shape;
      if (curShape.length === nextShape.length + 1
          && curShape[0] === 1
          && curShape.slice(1).every((d: number, i: number) => d === nextShape[i])) {
        chainLength++;
        currentNode = nextNode;
        currentIsReshapeSkip = true;
        outputDtype = nextNode.dtype || outputDtype;
        continue;
      }
      break;
    }
    currentIsReshapeSkip = false;

    // Match the op type
    let matched = false;

    if (nextNode.op === "cast") {
      const payload = nextNode.payload as { dtype: DType } | undefined;
      if (payload) {
        epilogueOps.push({ kind: "cast", toDtype: payload.dtype });
        outputDtype = payload.dtype;
        matched = true;
      }
    } else if (nextNode.op === "add" && nextNode.inputs.length === 2) {
      // Check if second input is a 1D bias (size matches matmul N dimension)
      const secondInput = nextNode.inputs[chainInputIdx === 0 ? 1 : 0];
      let secondShape: number[] | undefined;
      if (secondInput.kind === "materialized") {
        secondShape = secondInput.storage.backendTensor.shape;
      } else if (secondInput.kind === "pending") {
        secondShape = secondInput.node.shape;
      }

      if (secondShape && secondShape.length === 1) {
        // 1D bias — broadcast add
        if (additionalInputCount >= 4) break; // binding limit
        epilogueOps.push({ kind: "bias", inputIndex: additionalInputCount });
        epilogueInputRefs.push(secondInput);
        additionalInputCount++;
        matched = true;
      } else {
        // General binary add
        if (additionalInputCount >= 4) break;
        epilogueOps.push({ kind: "binary", op: "add", inputIndex: additionalInputCount });
        epilogueInputRefs.push(secondInput);
        additionalInputCount++;
        matched = true;
      }
    } else if (nextNode.op === "mul" && nextNode.inputs.length === 2) {
      if (additionalInputCount >= 4) break;
      epilogueOps.push({ kind: "binary", op: "mul", inputIndex: additionalInputCount });
      epilogueInputRefs.push(nextNode.inputs[chainInputIdx === 0 ? 1 : 0]);
      additionalInputCount++;
      matched = true;
    } else if (nextNode.op === "relu" || nextNode.op === "silu" || nextNode.op === "sigmoid" || nextNode.op === "tanh") {
      epilogueOps.push({ kind: "unary", op: nextNode.op });
      matched = true;
    } else if (nextNode.op === "gelu") {
      const geluPayload = nextNode.payload as { approximate?: string } | undefined;
      if (geluPayload?.approximate === "tanh") {
        epilogueOps.push({ kind: "gelu" });
      } else {
        // gelu with approximate="none" uses erf — expressed as unary "gelu_erf"
        epilogueOps.push({ kind: "unary", op: "gelu_erf" });
      }
      matched = true;
    }

    if (!matched) break;

    chainLength++;
    currentNode = nextNode;
    outputDtype = nextNode.dtype || outputDtype;
  }

  // Need at least one epilogue op to be worthwhile
  if (chainLength === 0) return null;

  return {
    consumedCount: 1 + chainLength, // matmul + epilogue ops
    epilogueOps,
    epilogueInputRefs,
    outputDtype,
    outputNode: currentNode,
  };
}

/**
 * Pre-scan variant: detect matmul epilogue from full plan (for pre-segmentation scan).
 */
export function detectMatmulEpilogueFromPlan(
  planNodes: LazyIRNode[],
  startIdx: number,
  consumerCount: Map<number, number>,
  externalNodeIds?: Set<number>,
): MatmulEpiloguePlan | null {
  return detectMatmulEpilogueCore(planNodes, startIdx, consumerCount, externalNodeIds);
}

/**
 * Segment variant: detect matmul epilogue within a segment.
 * Also checks the full plan for consumer counts.
 */
export function detectMatmulEpilogue(
  nodes: LazyIRNode[],
  startIdx: number,
  allPlanNodes: LazyIRNode[],
  externalNodeIds?: Set<number>,
): MatmulEpiloguePlan | null {
  // Build consumer count from full plan for accurate "only one consumer" checks
  const consumerCount = new Map<number, number>();
  for (const node of allPlanNodes) {
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        consumerCount.set(input.node.id, (consumerCount.get(input.node.id) ?? 0) + 1);
      }
    }
  }
  return detectMatmulEpilogueCore(nodes, startIdx, consumerCount, externalNodeIds);
}

/**
 * Detect simple last-2-dim transpose views (mirrors detectSimpleTranspose in index.ts).
 * Returns the effective contiguous shape and whether a transpose was detected.
 * Used by the matmul epilogue cache to compute correct geometry.
 */
export function _detectTransposeView(tensor: { shape: number[]; strides?: number[]; isContiguous?: boolean; offset?: number }): { shape: number[]; transposed: boolean } {
  const shape = tensor.shape;
  const strides = tensor.strides;
  const rank = shape.length;

  if (tensor.isContiguous || (tensor.offset ?? 0) !== 0 || rank < 2 || !strides) {
    return { shape: shape.slice(), transposed: false };
  }

  // Check for last-2-dim transpose: strides[-2] === 1 and strides[-1] === shape[-2]
  if (strides[rank - 2] === 1 && strides[rank - 1] === shape[rank - 2]) {
    // Verify batch dimensions are contiguous
    let expectedStride = shape[rank - 1] * shape[rank - 2];
    let valid = true;
    for (let i = rank - 3; i >= 0; i--) {
      if (strides[i] !== expectedStride) { valid = false; break; }
      expectedStride *= shape[i];
    }
    if (valid) {
      // Swap last 2 dims back to get original contiguous shape
      const origShape = shape.slice();
      origShape[rank - 2] = shape[rank - 1];
      origShape[rank - 1] = shape[rank - 2];
      return { shape: origShape, transposed: true };
    }
  }

  return { shape: shape.slice(), transposed: false };
}

/**
 * Execute a matmul with fused epilogue operations.
 * Uses the existing dispatchMatmulWithEpilogue from the WebGPU backend.
 */
export async function executeMatmulWithEpilogue(
  matmulNode: LazyIRNode,
  plan: MatmulEpiloguePlan,
  backend: Backend,
): Promise<void> {
  // Use cached imports (initialized once at executeLoweredPlan/executePlanOptimized entry)
  await ensureWebGPUMatmulImports();
  const { dispatchMatmulWithEpilogue } = _webgpuMatmulImports!;

  // Determine which inputs have prologue casts.
  // If the prologue cast was skipped (no result), use the original pre-cast input.
  // If the cast ran (e.g., in a fusion group), use the normal cast output instead.
  let inputCastA: DType | undefined;
  let inputCastB: DType | undefined;
  let resolvedInputRefA = matmulNode.inputs[0];
  let resolvedInputRefB = matmulNode.inputs[1];
  if (plan.prologues) {
    for (const p of plan.prologues) {
      // Check if the cast node's result was computed (e.g., via fusion group)
      const castRef = p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
      const castAlreadyRan = castRef.kind === "pending" && castRef.node.result != null;
      if (!castAlreadyRan) {
        // Cast was skipped — use the pre-cast f32 input and tell codegen about the cast
        if (p.inputIndex === 0) {
          resolvedInputRefA = p.originalInputRef;
          inputCastA = p.toDtype;
        } else {
          resolvedInputRefB = p.originalInputRef;
          inputCastB = p.toDtype;
        }
      }
      // If cast already ran, just use the normal matmul input (cast's f16 output)
    }
  }

  // Resolve matmul inputs
  const matmulInputA = getInputStorage(resolvedInputRefA);
  const matmulInputB = getInputStorage(resolvedInputRefB);

  // Resolve epilogue input refs
  const epilogueInputTensors: BackendTensor[] = [];
  for (const ref of plan.epilogueInputRefs) {
    const storage = getInputStorage(ref);
    epilogueInputTensors.push(storage.backendTensor);
  }

  // Build EpilogueConfig
  const epilogueConfig = {
    ops: plan.epilogueOps,
    additionalInputCount: plan.epilogueInputRefs.length,
    outputDtype: plan.outputDtype,
  };

  // Call dispatchMatmulWithEpilogue
  const resultTensor = dispatchMatmulWithEpilogue(
    asGPUTensor(matmulInputA.backendTensor),
    asGPUTensor(matmulInputB.backendTensor),
    epilogueConfig,
    epilogueInputTensors.map(t => asGPUTensor(t)),
    false, // transA
    false, // transB
    inputCastA,
    inputCastB,
  );

  // Store result on the final output node.
  // If epilogue absorbed a reshape (e.g. [1,M,N]→[M,N]), the matmul output shape
  // may differ from outputNode.shape. Fix up by mutating the tensor's shape/strides.
  const outNodeShape = plan.outputNode.shape;
  if (!shapesEqual(resultTensor.shape, outNodeShape)) {
    const gpuT = asGPUTensor(resultTensor);
    gpuT.shape = outNodeShape;
    // Recompute strides for the new shape (contiguous row-major)
    const newStrides = new Array(outNodeShape.length);
    let stride = 1;
    for (let i = outNodeShape.length - 1; i >= 0; i--) {
      newStrides[i] = stride;
      stride *= outNodeShape[i];
    }
    gpuT.strides = newStrides;
  }
  plan.outputNode.result = createStorageHandle(plan.outputNode.device, resultTensor);
}

// shapesEqual re-exported from core/shape
