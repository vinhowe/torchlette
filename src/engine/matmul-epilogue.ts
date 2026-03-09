import type { BackendTensor, DType } from "../backend/types";
import { asGPUTensor } from "../backend/webgpu/gpu-types";
import type { EpilogueConfig } from "../backend/webgpu/matmul/types";
import { contiguousStrides, shapesEqual } from "../core/shape";
import { isFusibleOp } from "./fusion-detect";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import {
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
} from "./node-factory";
import { getInputStorage } from "./op-dispatch";

// ============================================================================
// Shared epilogue chain helpers (used by matmul-epilogue and reduction-preamble)
// ============================================================================

/** Epilogue operation descriptor for fused kernel chains. */
export type EpilogueOp = {
  kind: string;
  toDtype?: DType;
  inputIndex?: number;
  op?: string;
};

/**
 * Find which input of `nextNode` continues a chain from `currentNode`.
 * Returns the input index (0 or 1) if found, or -1 if the chain breaks.
 * For commutative ops (add, mul), checks both inputs.
 */
export function findChainInput(
  nextNode: LazyIRNode,
  currentNodeId: number,
): number {
  const primary = nextNode.inputs[0];
  if (primary.kind === "pending" && primary.node.id === currentNodeId) return 0;
  if (
    (nextNode.op === "add" || nextNode.op === "mul") &&
    nextNode.inputs.length === 2
  ) {
    const alt = nextNode.inputs[1];
    if (alt.kind === "pending" && alt.node.id === currentNodeId) return 1;
  }
  return -1;
}

/**
 * Get the applyFusedOp-compatible operation name for a node.
 * Handles payload inspection for gelu (approximate variant) and cast (dtype suffix).
 * For all other ops, node.op is used directly.
 */
export function getEpilogueOpName(node: LazyIRNode): string {
  if (node.op === "gelu") {
    const p = node.payload as { approximate?: string } | undefined;
    return p?.approximate === "tanh" ? "gelu" : "gelu_erf";
  }
  if (node.op === "cast") {
    const p = node.payload as { dtype?: string } | undefined;
    return `cast_${p?.dtype || "f32"}`;
  }
  return node.op;
}

/** Format an array of epilogue ops into a profile label fragment (e.g. "cast+add+relu"). */
export function formatEpilogueLabel(ops: EpilogueOp[]): string {
  return ops
    .map((o) =>
      o.kind === "binary" ? o.op : o.kind === "cast" ? "cast" : o.op || o.kind,
    )
    .join("+");
}

// ============================================================================
// Matmul Epilogue Fusion (Phase 1)
// Detects matmul → cast/bias/activation chains and fuses them into a single
// dispatchMatmul call with epilogue options, eliminating intermediate buffers.
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
  epilogueOps: EpilogueOp[];
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

  const epilogueOps: EpilogueOp[] = [];
  const epilogueInputRefs: LazyRef[] = [];
  let additionalInputCount = 0;
  let chainLength = 0;
  let currentNode = matmulNode;
  let outputDtype = matmulNode.dtype;

  // Walk forward from matmul, matching epilogue-compatible ops
  let currentIsReshapeSkip = false;
  for (let i = startIdx + 1; i < nodes.length && chainLength < 4; i++) {
    const nextNode = nodes[i];

    // Check that the candidate's input continues the chain from currentNode
    if (nextNode.inputs.length === 0) break;
    const chainInputIdx = findChainInput(nextNode, currentNode.id);
    if (chainInputIdx < 0) break;

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
      if (
        curShape.length === nextShape.length + 1 &&
        curShape[0] === 1 &&
        curShape.slice(1).every((d: number, i: number) => d === nextShape[i])
      ) {
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
        epilogueOps.push({
          kind: "binary",
          op: "add",
          inputIndex: additionalInputCount,
        });
        epilogueInputRefs.push(secondInput);
        additionalInputCount++;
        matched = true;
      }
    } else if (
      (nextNode.op === "mul" || nextNode.op === "sub") &&
      nextNode.inputs.length === 2
    ) {
      if (additionalInputCount >= 4) break;
      epilogueOps.push({
        kind: "binary",
        op: nextNode.op,
        inputIndex: additionalInputCount,
      });
      epilogueInputRefs.push(nextNode.inputs[chainInputIdx === 0 ? 1 : 0]);
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
 * Detect simple last-2-dim transpose views (mirrors detectSimpleTranspose in index.ts).
 * Returns the effective contiguous shape and whether a transpose was detected.
 * Used by the matmul epilogue cache to compute correct geometry.
 */
export function _detectTransposeView(tensor: {
  shape: number[];
  strides?: number[];
  isContiguous?: boolean;
  offset?: number;
}): { shape: number[]; transposed: boolean } {
  const shape = tensor.shape;
  const strides = tensor.strides;
  const rank = shape.length;

  if (
    tensor.isContiguous ||
    (tensor.offset ?? 0) !== 0 ||
    rank < 2 ||
    !strides
  ) {
    return { shape: shape.slice(), transposed: false };
  }

  // Check for last-2-dim transpose: strides[-2] === 1 and strides[-1] === shape[-2]
  if (strides[rank - 2] === 1 && strides[rank - 1] === shape[rank - 2]) {
    // Verify batch dimensions are contiguous
    let expectedStride = shape[rank - 1] * shape[rank - 2];
    let valid = true;
    for (let i = rank - 3; i >= 0; i--) {
      if (strides[i] !== expectedStride) {
        valid = false;
        break;
      }
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
 * Uses dispatchMatmul from the WebGPU backend with epilogue options.
 */
export async function executeMatmulWithEpilogue(
  matmulNode: LazyIRNode,
  plan: MatmulEpiloguePlan,
): Promise<void> {
  // Use cached imports (initialized once at executeLoweredPlan/executePlanOptimized entry)
  await ensureWebGPUMatmulImports();
  const { dispatchMatmul } = _webgpuMatmulImports as NonNullable<
    typeof _webgpuMatmulImports
  >;

  // Determine which inputs have prologue casts.
  // If the prologue cast was skipped (no result), use the original pre-cast input.
  // If the cast ran (e.g., in a fusion group), use the normal cast output instead.
  let inputCastA: "f16" | "f32" | undefined;
  let inputCastB: "f16" | "f32" | undefined;
  let resolvedInputRefA = matmulNode.inputs[0];
  let resolvedInputRefB = matmulNode.inputs[1];
  if (plan.prologues) {
    for (const p of plan.prologues) {
      // Check if the cast node's result was computed (e.g., via fusion group)
      const castRef =
        p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
      const castAlreadyRan =
        castRef.kind === "pending" && castRef.node.result != null;
      if (!castAlreadyRan) {
        // Cast was skipped — use the pre-cast f32 input and tell codegen about the cast
        if (p.inputIndex === 0) {
          resolvedInputRefA = p.originalInputRef;
          inputCastA = p.toDtype as "f16" | "f32";
        } else {
          resolvedInputRefB = p.originalInputRef;
          inputCastB = p.toDtype as "f16" | "f32";
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

  // Build EpilogueConfig (cast loose EpilogueOp[] to matmul's strict discriminated union)
  const epilogueConfig = {
    ops: plan.epilogueOps,
    additionalInputCount: plan.epilogueInputRefs.length,
    outputDtype: plan.outputDtype,
  } as EpilogueConfig;

  // Call dispatchMatmul with epilogue options
  const resultTensor = dispatchMatmul(
    asGPUTensor(matmulInputA.backendTensor),
    asGPUTensor(matmulInputB.backendTensor),
    false, // transA
    false, // transB
    undefined, // donatedBuffer
    {
      epilogue: epilogueConfig,
      epilogueInputs: epilogueInputTensors.map((t) => asGPUTensor(t)),
      inputCastA,
      inputCastB,
    },
  );

  // Store result on the final output node.
  // If epilogue absorbed a reshape (e.g. [1,M,N]→[M,N]), the matmul output shape
  // may differ from outputNode.shape. Fix up by mutating the tensor's shape/strides.
  const outNodeShape = plan.outputNode.shape;
  if (!shapesEqual(resultTensor.shape, outNodeShape)) {
    const gpuT = asGPUTensor(resultTensor);
    gpuT.shape = outNodeShape;
    gpuT.strides = contiguousStrides(outNodeShape);
  }
  plan.outputNode.result = createStorageHandle(
    plan.outputNode.device,
    resultTensor,
  );
}
