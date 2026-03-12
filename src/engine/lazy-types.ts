import type { BackendTensor, DeviceKind, DType } from "../backend/types";
import type { Token } from "./tokens";

export type LazyOpCode =
  | "add"
  | "sub"
  | "mul"
  | "div"
  | "matmul"
  | "sqrt"
  | "relu"
  | "reshape"
  | "expand"
  | "transpose"
  | "permute"
  | "contiguous"
  | "gather"
  | "scatterAdd"
  | "cat"
  | "sum"
  | "mean"
  | "max"
  | "min"
  | "argmax"
  | "argmin"
  | "gt"
  | "lt"
  | "ge"
  | "le"
  | "eq"
  | "ne"
  | "where"
  | "stridedScatterCopy"
  | "stridedScatterAdd"
  | "tensorFromArray"
  | "transfer"
  | "neg"
  | "abs"
  | "exp"
  | "log"
  | "tanh"
  | "sigmoid"
  | "gelu"
  | "silu"
  | "cast"
  | "pow"
  | "zeros"
  | "full"
  | "arange"
  | "tril"
  | "triu"
  | "rand"
  | "randn"
  | "bernoulli"
  | "sin"
  | "cos"
  | "rsqrt"
  | "floor"
  | "ceil"
  | "round"
  | "sign"
  | "clamp"
  | "isfinite"
  | "narrow"
  | "narrowBackward"
  | "adamStep"
  | "unscaleGrad"
  | "fusedAttentionForward"
  | "fusedAttentionBackward"
  | "fusedCrossEntropyForward"
  | "fusedCrossEntropyBackward"
  | "fusedLayerNormForward"
  | "fusedLayerNormBackwardGradX"
  | "fusedLayerNormBackwardGradWeightBias"
  | "fusedRMSNormForward"
  | "fusedRMSNormBackwardGradX"
  | "fusedRMSNormBackwardGradWeight";

export interface StorageHandle {
  id: number;
  device: DeviceKind;
  backendTensor: BackendTensor;
  /** For views: ID of the storage that owns the buffer. Views don't destroy buffers. */
  baseStorageId?: number;
}

export interface LazyIRNode {
  id: number;
  op: LazyOpCode;
  inputs: LazyRef[];
  shape: number[];
  dtype: DType;
  device: DeviceKind;
  tokenIn?: Token;
  tokenOut?: Token;
  /** Primary output (index 0). Set after execution. */
  result?: StorageHandle;
  /** All outputs for multi-output ops. results[0] === result. */
  results?: StorageHandle[];
  payload?: unknown;
  /** Module label for profiling (set via setProfileModule during graph construction) */
  module?: string;
  /**
   * Marks this node as a checkpoint boundary. When executing with segmented
   * checkpoint execution, the executor will flush the buffer pool after this
   * node completes, making released buffers available for subsequent segments.
   * This enables memory savings for large models that don't fit in GPU memory.
   */
  isCheckpointBoundary?: boolean;
}

export type LazyRef =
  | { kind: "pending"; node: LazyIRNode; outputIndex?: number }
  | { kind: "materialized"; storage: StorageHandle }
  | { kind: "scalar"; value: number; dtype: DType };

export function createPendingRef(
  node: LazyIRNode,
  outputIndex?: number,
): LazyRef {
  if (outputIndex !== undefined && outputIndex !== 0) {
    return { kind: "pending", node, outputIndex };
  }
  return { kind: "pending", node };
}

export function createMaterializedRef(storage: StorageHandle): LazyRef {
  return { kind: "materialized", storage };
}

export function createScalarRef(value: number, dtype: DType): LazyRef {
  return { kind: "scalar", value, dtype };
}

export function isPending(
  ref: LazyRef,
): ref is { kind: "pending"; node: LazyIRNode } {
  return ref.kind === "pending";
}

export function isMaterialized(
  ref: LazyRef,
): ref is { kind: "materialized"; storage: StorageHandle } {
  return ref.kind === "materialized";
}

export interface ExecutionPlan {
  nodes: LazyIRNode[];
}

/**
 * Options for plan execution.
 */
export interface ExecutePlanOptions {
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
}
