/**
 * Row-Program IR Types
 *
 * A row program is a computation where one workgroup processes one row:
 * load elements, apply elementwise ops, reduce to per-row scalars, use
 * those scalars in more ops, optionally reduce again, and write output.
 *
 * Between reduce phases, only scalars survive — per-element values are
 * recomputed (matching hand-written softmax/layernorm kernels).
 */

import type { DType } from "../backend/types";
import type { LazyRef } from "../graph/types";

// ============================================================================
// Expression IR
// ============================================================================

/** Leaf value in an expression tree. */
export type RPValue =
  | { kind: "input"; bufferIndex: number } // External buffer load at [base+i]
  | { kind: "reduceResult"; phaseIndex: number } // Scalar from prior reduce
  | { kind: "const"; value: number }; // Inline constant

/** Expression tree node. Leaves are RPValue, interior nodes apply ops. */
export type RPExpr = RPValue | { op: string; inputs: RPExpr[] };

/** Type guard: is this expression a leaf value? */
export function isRPValue(e: RPExpr): e is RPValue {
  return "kind" in e;
}

// ============================================================================
// Phases
// ============================================================================

/** A phase that computes a workgroup-cooperative reduction. */
export interface ReducePhase {
  kind: "reduce";
  reduceOp: "sum" | "max" | "min";
  /** Expression computing the per-element value to reduce. */
  bodyExpr: RPExpr;
  /** For mean: divide result by element count after reduction. */
  isMean?: boolean;
}

/** A phase that writes output values. */
export interface WritePhase {
  kind: "write";
  /** Expression computing each output element value. */
  bodyExpr: RPExpr;
  /** If true, output is a scalar per row (1 element, not D elements).
   *  Used when the output IS a reduction result with no epilogue. */
  scalarOutput?: boolean;
}

export type RPPhase = ReducePhase | WritePhase;

// ============================================================================
// Complete Program
// ============================================================================

/** Full row-program specification. */
export interface RowProgram {
  /** Input buffer descriptors (index = bufferIndex in RPValue). */
  inputs: Array<{ dtype: DType }>;
  /** Output buffer descriptor. */
  output: { dtype: DType };
  /** Reduction dimension (normalized). Must be the last dim for perRowKernel. */
  dim: number;
  /** Ordered sequence of phases. Must end with a WritePhase. */
  phases: RPPhase[];
  /** Structural cache key for cross-step kernel caching. */
  cacheKey: string;
}

// ============================================================================
// Detection Output
// ============================================================================

/** A detected row-program match from the graph compiler. */
export interface RowProgramMatch {
  /** All node IDs consumed by this row program. */
  coveredNodeIds: number[];
  /** The output node ID. */
  outputNodeId: number;
  /** External input refs (for resolving buffers at execution time). */
  inputRefs: LazyRef[];
  /** CONSUMER provenance per input ref: which subgraph node's input slot the
   *  ref was captured from. The refs above are a lowering-time SNAPSHOT that
   *  goes stale on template reuse (a materialized ref's storage is a previous
   *  step's swept temp — the clipGradNorm_ clipCoef class); the consuming node
   *  is re-created fresh every step, so the CURRENT ref is
   *  planNodes[posOf(nodeId)].inputs[inputIndex] — the single source. */
  inputRefConsumers: Array<{ nodeId: number; inputIndex: number }>;
  /** Reduction dimension (normalized). */
  dim: number;
  /** Number of rows (product of dims before reduction dim in the first reduction's input). */
  numRows: number;
  /** Size of the reduction dimension (feature dim for perRowKernel). */
  dimSize: number;
  /** The constructed RowProgram IR. */
  program: RowProgram;
}
