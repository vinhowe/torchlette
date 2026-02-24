import type {
  BackendTensor,
  DeviceKind,
  DType,
} from "../backend/types";
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
  | "sum"
  | "mean"
  | "max"
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
  | "isfinite"
  | "narrow"
  | "narrowBackward"
  | "adamStep"
  | "unscaleGrad"
  | "fusedAttentionForward"
  | "extractAttentionLogsumexp"
  | "fusedAttentionBackward"
  | "extractAttentionDK"
  | "extractAttentionDV"
  | "fusedCrossEntropyForward"
  | "fusedCrossEntropyBackward"
  | "fusedLayerNormForward"
  | "fusedLayerNormBackwardGradX"
  | "fusedLayerNormBackwardGradWeightBias"
  | "extractLnBwdGradBias";

/** GELU approximation type matching PyTorch's nn.GELU */
export type GeluApproximate = "none" | "tanh";

export interface StorageHandle {
  id: number;
  device: DeviceKind;
  backendTensor: BackendTensor;
  /** For views: ID of the storage that owns the buffer. Views don't destroy buffers. */
  baseStorageId?: number;
}

/** Side outputs for multi-output ops (attention, layernorm, adam). */
export interface NodeSideOutputs {
  /** fusedAttentionForward → logsumexp StorageHandle */
  attnLogsumexp?: StorageHandle;
  /** fusedAttentionBackward → dK StorageHandle */
  attnBwdDK?: StorageHandle;
  /** fusedAttentionBackward → dV StorageHandle */
  attnBwdDV?: StorageHandle;
  /** fusedLayerNormBackwardGradWeightBias → gradBias BackendTensor */
  lnBwdGradBias?: BackendTensor;
  /** adamStep → updated m and v StorageHandles */
  adamMV?: { m: StorageHandle; v: StorageHandle };
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
  result?: StorageHandle;
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
  /** Side outputs for multi-output ops (attention, layernorm, adam). */
  _sideOutputs?: NodeSideOutputs;
}

export type LazyRef =
  | { kind: "pending"; node: LazyIRNode }
  | { kind: "materialized"; storage: StorageHandle }
  | { kind: "scalar"; value: number; dtype: DType };

export function createPendingRef(node: LazyIRNode): LazyRef {
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
  /** Enable segmented execution at checkpoint boundaries */
  enableCheckpointSegmentation?: boolean;
  /** Skip nodes that already have a .result (from tape replay). */
  skipPreExecuted?: boolean;
  /** Only execute view/data-source ops; skip all compute ops. Used by tape replay fill-in. */
  viewOpsOnly?: boolean;
}
