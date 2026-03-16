import type { OP_REGISTRY } from "../ops/registry";
import type { BackendTensor, DeviceKind, DType } from "../backend/types";

/**
 * All valid lazy op codes.
 *
 * Elementwise ops (add, mul, relu, etc.) are derived automatically from
 * OP_REGISTRY keys — adding a new entry there makes it a valid LazyOpCode.
 * Non-elementwise ops (matmul, reductions, views, fused kernels) are listed
 * explicitly below since they're not in OP_REGISTRY.
 */
export type LazyOpCode =
  | keyof typeof OP_REGISTRY
  // Structural / view ops
  | "matmul"
  | "reshape"
  | "expand"
  | "transpose"
  | "permute"
  | "contiguous"
  | "narrow"
  | "narrowBackward"
  | "cast"
  // Reductions
  | "sum"
  | "mean"
  | "max"
  | "min"
  | "argmax"
  | "argmin"
  // Creation ops
  | "tensorFromArray"
  | "zeros"
  | "full"
  | "arange"
  | "tril"
  | "triu"
  | "rand"
  | "randn"
  | "bernoulli"
  // Scatter / gather
  | "gather"
  | "scatterAdd"
  | "cat"
  | "stridedScatterCopy"
  | "stridedScatterAdd"
  // Special
  | "clamp"
  | "transfer"
  | "adamStep"
  | "unscaleGrad"
  // Fused GPU kernels
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
