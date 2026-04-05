/**
 * Wire format for remote plan execution.
 *
 * Parallels src/graph/types.ts (LazyIRNode / LazyRef / ExecutionPlan), with:
 *   - Process-global node ids replaced by plan-local indices (0..N-1, dense).
 *   - Object references (LazyRef.pending.node) replaced by numeric node indices.
 *   - Runtime-only fields stripped (result, results — these are execution outputs).
 *   - Float32Array payloads carried as explicit byte-arrays with a tagged wrapper.
 *
 * This is the payload of `execute` RPCs. Everything here must JSON-round-trip.
 */

import type { DeviceKind, DType } from "../backend/types";
import type { LazyOpCode } from "../graph/types";

/**
 * Opaque, server-issued identifier for a storage resident on the server.
 * Used to reference materialized tensors (model weights, inputs uploaded via
 * `upload`, outputs of previous `execute` calls) from within a plan.
 *
 * The client treats these as opaque strings; only the server interprets them.
 */
export type HandleRef = string;

/**
 * Plan-local node index (0..nodes.length-1). Nodes are emitted in topological
 * order, so any pending ref's nodeIdx is always less than the referring node's
 * index.
 */
export type NodeIdx = number;

/** Tagged reference to another node's output within the plan. */
export interface SerializedPendingRef {
  kind: "pending";
  nodeIdx: NodeIdx;
  outputIndex?: number;
}

/**
 * Reference to a materialized tensor that already exists server-side.
 * The `handle` must have been returned by a prior `execute`, `upload`,
 * or `restore` call on the same session.
 */
export interface SerializedMaterializedRef {
  kind: "materialized";
  handle: HandleRef;
}

/** Inline scalar constant. */
export interface SerializedScalarRef {
  kind: "scalar";
  value: number;
  dtype: DType;
}

export type SerializedRef =
  | SerializedPendingRef
  | SerializedMaterializedRef
  | SerializedScalarRef;

/**
 * Tagged wrapper for a dense tensor carried inline in a payload.
 *
 * Used for `tensorFromArray` nodes whose `values` field is a Float32Array.
 * For anything non-trivial, clients should `upload` out-of-band and reference
 * via a materialized ref instead.
 */
export interface InlineTensorBytes {
  __inlineTensor: true;
  dtype: DType;
  /** Array of numbers — sufficient for f32/i32/u32/bool. */
  values: number[];
}

export interface SerializedNode {
  /** Plan-local index. Redundant with position in `nodes[]` but explicit for readability. */
  idx: NodeIdx;
  op: LazyOpCode;
  inputs: SerializedRef[];
  shape: number[];
  dtype: DType;
  device: DeviceKind;
  /**
   * Op-specific payload. All values must be JSON-safe after applying the
   * InlineTensorBytes transform to Float32Arrays. See `src/remote/serialize.ts`
   * for the transform.
   */
  payload?: unknown;
  /** Profiler label, if any. */
  module?: string;
  isCheckpointBoundary?: boolean;
}

export interface SerializedPlan {
  /** Format version — bump on breaking changes. */
  version: 1;
  /** Nodes in topological order. nodes[i].idx === i. */
  nodes: SerializedNode[];
  /**
   * External handles referenced by the plan. The server uses this to sanity-
   * check that every materialized ref resolves before executing, and to
   * maintain a reference count during execution.
   */
  externalHandles: HandleRef[];
  /**
   * Indices of nodes whose outputs the client wants back. The server returns a
   * handle for each, which the client stores in its own runtime tensors.
   *
   * If undefined, the server returns handles for ALL nodes (conservative).
   * In practice the client will pass a minimal set matching user-held tensors.
   */
  outputNodes?: NodeIdx[];
}

/**
 * Server response to `execute`.
 */
export interface ExecuteResponse {
  /** For each requested output node idx: the server's handle for its result. */
  outputs: Record<NodeIdx, HandleRef>;
}
