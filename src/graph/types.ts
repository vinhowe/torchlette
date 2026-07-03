import type { BackendTensor, DeviceKind, DType } from "../backend/types";
import type { OP_REGISTRY } from "../ops/registry";

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
  | "maximum"
  | "minimum"
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
  | "fusedRMSNormBackwardGradWeight"
  | "fusedRoPE";

export interface StorageHandle {
  id: number;
  device: DeviceKind;
  backendTensor: BackendTensor;
  /** For views: ID of the storage that owns the buffer. Views don't destroy buffers. */
  baseStorageId?: number;
  /**
   * Set on VIEW handles created by the compiled-plan harvest whose base
   * retain is OWNED by the plan (tracked in CompiledPlan._viewBaseRetains and
   * released at the next harvest / plan teardown). For such handles
   * destroyStorageIds must NOT also fire the "view.destroyed" base release —
   * that would double-release the base (the plan releases it exactly once).
   * The plan is the SOLE owner of the base retain for harvested views,
   * decoupling the base's refcount from the harvested handle's lifecycle
   * (which may be destroyed at markStep OR leak-retained across replays).
   */
  planOwnedBaseRetain?: boolean;
}

/**
 * A lazy IR node — one entry in the deferred-execution graph.
 *
 * `result` is a derived view of `results[0]`, not an independent field.
 * Reading `node.result` returns `node.results?.[0]`; assigning `node.result`
 * mutates `node.results[0]` in place. This makes it impossible for the two
 * to diverge — the historical "Input not ready: adamStep[0]" bug class came
 * from code paths that cleared `node.result` while leaving a stale handle in
 * `node.results[0]`, which the read fallback then silently used.
 *
 * Multi-output ops (`adamStep`, `fusedAttention*`, `fusedLayerNormBackwardGradWeightBias`)
 * populate additional slots `results[1..N]` for their side outputs. Those
 * slots have INDEPENDENT lifecycles — clearing `node.result` only nulls
 * `results[0]`, never the side outputs.
 */
export class LazyIRNode {
  /**
   * Brand field — makes LazyIRNode nominally typed under TypeScript's
   * structural type system. Without this, code could create an object
   * literal `{...} as LazyIRNode` and skip the constructor — which would
   * also skip the `result` getter/setter, breaking the invariant that
   * `node.result === node.results?.[0]`. The brand forces all instances
   * to come from `new LazyIRNode(...)` (or via `createLazyIRNode`).
   */
  // biome-ignore lint/correctness/noUnusedPrivateClassMembers: nominal-typing brand
  private _brand!: undefined;

  id: number;
  op: LazyOpCode;
  inputs: LazyRef[];
  shape: number[];
  dtype: DType;
  device: DeviceKind;
  /**
   * All outputs. `results[0]` is the primary output (= `node.result`).
   * Slots may be sparse: `results[i]` can be undefined while a later slot
   * is still set (e.g., after liveness clears the primary buffer).
   */
  results?: (StorageHandle | undefined)[];
  /** True while input rc is retained by the plan executor (prevents double-retain). */
  _inputsRetained?: boolean;
  /**
   * Set after execution. Survives node.result cleanup so buildMergedPlan's
   * skipExecuted can distinguish "executed but result cleared" from "never executed".
   */
  _executed?: boolean;
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

  constructor(
    id: number,
    op: LazyOpCode,
    inputs: LazyRef[],
    shape: number[],
    dtype: DType,
    device: DeviceKind,
    payload?: unknown,
  ) {
    this.id = id;
    this.op = op;
    this.inputs = inputs;
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
    this.payload = payload;
  }

  /** Primary output (index 0). Derived view of `results[0]`. */
  get result(): StorageHandle | undefined {
    return this.results?.[0];
  }

  set result(value: StorageHandle | undefined) {
    if (value === undefined) {
      if (this.results) this.results[0] = undefined;
    } else if (!this.results) {
      this.results = [value];
    } else {
      this.results[0] = value;
    }
  }
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
  /**
   * Indices into `nodes[]` whose results the caller needs.
   *
   * Set by the engine from live RuntimeTensor tracking before execution.
   * Flows through serialization to remote executors. Used by the optimized
   * executor for buffer lifetime analysis, epilogue safety, and scoped
   * result production (only output nodes are guaranteed to have `node.result`
   * after execution).
   *
   * If undefined, all nodes are treated as outputs (conservative fallback).
   */
  outputIndices?: number[];
}

/**
 * Options for plan execution.
 */
export interface ExecutePlanOptions {
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
}
