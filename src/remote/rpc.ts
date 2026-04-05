/**
 * RPC envelope and method schemas for the remote-training protocol.
 *
 * Transport-agnostic (WebSocket or HTTP or in-process). Each request carries an
 * id and a method name; each response echoes the id and carries either `result`
 * or `error`. No streaming in v0 — every RPC is request/response.
 */

import type { DType } from "../backend/types";
import type { HandleRef, NodeIdx, SerializedPlan } from "./wire";

export interface RpcRequest<M extends string = string, P = unknown> {
  id: number;
  method: M;
  params: P;
}

export interface RpcSuccess<R = unknown> {
  id: number;
  result: R;
}

export interface RpcError {
  id: number;
  error: { message: string; stack?: string };
}

export type RpcResponse<R = unknown> = RpcSuccess<R> | RpcError;

export function isRpcError(r: RpcResponse): r is RpcError {
  return "error" in r;
}

// ============================================================================
// Method params / results
// ============================================================================

/** Execute a plan on the server, registering output handles. */
export interface ExecuteParams {
  plan: SerializedPlan;
}
export interface ExecuteResult {
  /** For each output node idx (from plan.outputNodes, or all nodes), the
   * server-allocated HandleRef. */
  outputs: Record<NodeIdx, HandleRef>;
}

/** Upload a tensor to the server, registering its handle. */
export interface UploadParams {
  values: number[];
  shape: number[];
  dtype: DType;
}
export interface UploadResult {
  handle: HandleRef;
}

/** Download tensor bytes for a handle. */
export interface DownloadParams {
  handle: HandleRef;
}
export interface DownloadResult {
  values: number[];
}

/** Read a single scalar (shape []) without incurring full-tensor download. */
export interface ReadScalarParams {
  handle: HandleRef;
}
export interface ReadScalarResult {
  value: number;
}

/** Release handles the client no longer references. */
export interface ReleaseParams {
  handles: HandleRef[];
}
export interface ReleaseResult {
  releasedCount: number;
}

/** Server → client greeting. */
export interface HelloResult {
  sessionId: string;
  protocolVersion: 1;
}

// ============================================================================
// Typed method union (for server dispatch)
// ============================================================================

export type RpcMethod =
  | "execute"
  | "upload"
  | "download"
  | "readScalar"
  | "release";
