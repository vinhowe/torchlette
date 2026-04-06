/**
 * Pattern AST: structural queries over LazyIR.
 *
 * A Pattern describes the shape of a sub-tree to match in the lazy graph.
 * Patterns are plain data (no runtime state) — they're matched against
 * LazyRefs by the matcher, which produces a Bindings map.
 *
 * Design principles:
 *   - Patterns are small, composable values (no classes, no state).
 *   - Captures bind names to LazyRefs — the matcher never descends into
 *     an already-captured ref's internals.
 *   - Per-node predicates (`where`) run during traversal.
 *   - Cross-capture constraints are expressed as rule-level predicates,
 *     not inside patterns.
 */
import type { DType } from "../../backend/types";
import type { LazyOpCode, LazyIRNode, LazyRef } from "../../graph/types";

/** A pattern matches a single LazyRef; sub-patterns match the inputs of
 *  the node that ref points to. */
export type Pattern =
  | OpPattern
  | CapturePattern
  | AnyPattern
  | ScalarPattern
  | MaterializedPattern;

/** Match a pending-ref that points to a node with matching op and inputs. */
export interface OpPattern {
  readonly kind: "op";
  readonly op: LazyOpCode;
  /** Sub-patterns for this node's inputs, in order. Must match the full
   *  input arity exactly (short-match is an error in the rule). */
  readonly inputs?: readonly Pattern[];
  /** Per-node predicate. Receives the matched node; returns true to accept. */
  readonly where?: (node: LazyIRNode) => boolean;
}

/** Bind `name` to whatever LazyRef matched here. If `inner` is provided,
 *  the ref must ALSO match `inner` for the capture to succeed. */
export interface CapturePattern {
  readonly kind: "capture";
  readonly name: string;
  readonly inner?: Pattern;
}

/** Match any ref. */
export interface AnyPattern {
  readonly kind: "any";
}

/** Match a scalar ref (LazyRef of kind "scalar"). */
export interface ScalarPattern {
  readonly kind: "scalar";
  readonly where?: (value: number) => boolean;
}

/** Match a materialized ref (external input to the plan). */
export interface MaterializedPattern {
  readonly kind: "materialized";
}

// ============================================================================
// Builders — terse, tree-literal-style API
// ============================================================================

/** Match any ref, without binding. */
export const any: AnyPattern = { kind: "any" };

/** Bind `name` to whatever ref matched here. */
export function capture(name: string, inner?: Pattern): CapturePattern {
  return { kind: "capture", name, inner };
}

/** Match a node with a specific op code. */
export function op(
  opCode: LazyOpCode,
  options?: {
    inputs?: readonly Pattern[];
    where?: (node: LazyIRNode) => boolean;
  },
): OpPattern {
  return { kind: "op", op: opCode, inputs: options?.inputs, where: options?.where };
}

/** Match a scalar ref (optional value predicate). */
export function scalar(where?: (value: number) => boolean): ScalarPattern {
  return { kind: "scalar", where };
}

/** Match a materialized (external) ref. */
export const materialized: MaterializedPattern = { kind: "materialized" };

// ============================================================================
// Bindings
// ============================================================================

/** Map from capture name to matched LazyRef. */
export type Bindings = ReadonlyMap<string, LazyRef>;

/** Mutable version used internally by the matcher. */
export type MutableBindings = Map<string, LazyRef>;

// ============================================================================
// Ref inspection helpers — for use inside check/rewrite functions
// ============================================================================

/** Get the static shape of whatever `ref` points to. Returns null for
 *  refs that don't have a shape (should be unreachable in practice). */
export function refShape(ref: LazyRef): number[] | null {
  if (ref.kind === "pending") return ref.node.shape;
  if (ref.kind === "materialized") return ref.storage.backendTensor.shape;
  if (ref.kind === "scalar") return [];
  return null;
}

/** Get the dtype of whatever `ref` points to. */
export function refDtype(ref: LazyRef): DType | null {
  if (ref.kind === "pending") return ref.node.dtype;
  if (ref.kind === "materialized") return ref.storage.backendTensor.dtype;
  if (ref.kind === "scalar") return ref.dtype;
  return null;
}

/** Get the node behind a pending ref, or null. */
export function refNode(ref: LazyRef): LazyIRNode | null {
  return ref.kind === "pending" ? ref.node : null;
}
