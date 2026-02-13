/**
 * Lowered Execution Plan
 *
 * Replaces the dispatch tape replay system with plan-level caching.
 * Instead of recording raw GPU dispatches and replaying them (with a costly
 * fill-in pass for fused intermediates), we record the full dispatch sequence
 * as a sequence of typed actions during the first execution of a plan with a
 * given structural fingerprint, then re-execute from the cached lowered plan
 * on subsequent steps.
 *
 * The lowered plan captures:
 * - Fused kernel dispatches (recipe + covered node indices)
 * - Sequential op dispatches (op + node index)
 * - Matmul+epilogue chains (matmul + epilogue config + covered nodes)
 * - Reduction preamble fusions (preamble + reduction nodes)
 * - View ops (metadata only, no GPU dispatch)
 * - Data source ops (tensorFromArray, zeros, full, arange)
 * - Adam batch segments (consecutive adamStep nodes)
 * - Buffer reclaim points
 *
 * Buffer references use plan-node indices, not GPU buffer pointers.
 * Persistent buffers (model params) resolve through the lazy graph's
 * materialized refs. Pool buffers are allocated fresh each step via
 * existing sequence-hinted pool.
 */

import type { DType } from "../backend/types";
import type { FusedKernelRecipe } from "../backend/webgpu/fusion-codegen";
import type { LazyIRNode, LazyRef } from "./lazy";

// ============================================================================
// Lowered Action Types
// ============================================================================

/** A fused elementwise kernel dispatch covering multiple plan nodes. */
export interface LoweredFusedAction {
  kind: "fused";
  /** Plan-node indices covered by this fused kernel (in group order). */
  coveredNodeIndices: number[];
  /** Index of the primary output node in the plan. */
  outputNodeIndex: number;
  /** Indices of additional output nodes (§15.2 multi-output). */
  additionalOutputNodeIndices: number[];
  /** Indices of needed intermediate nodes that require re-execution. */
  neededIntermediateNodeIndices: number[];
  /** The fusion recipe (cached — same object reused across steps). */
  recipe: FusedKernelRecipe;
  /** Whether vectorization was enabled for this dispatch. */
  enableVectorization: boolean;
  /**
   * Cached external input extraction pattern. On the first execution,
   * we compute which (nodeIdx, inputIdx) pairs constitute external inputs.
   * On subsequent executions, we skip the O(n²) dedup and walk this pattern.
   * Format: Array of { nodeLocalIdx, inputIdx, kind } where nodeLocalIdx
   * indexes into coveredNodeIndices.
   */
  cachedExternalInputPattern?: Array<{ nodeLocalIdx: number; inputIdx: number }>;
}

/** A single non-fused op dispatch. */
export interface LoweredSequentialAction {
  kind: "sequential";
  /** Plan-node index for this op. */
  nodeIndex: number;
}

/** A matmul + epilogue chain dispatch. */
export interface LoweredMatmulEpilogueAction {
  kind: "matmul-epilogue";
  /** Plan-node index of the matmul node. */
  matmulNodeIndex: number;
  /** Plan-node indices consumed by the epilogue (matmul + chain). */
  coveredNodeIndices: number[];
  /** Index of the final output node. */
  outputNodeIndex: number;
  /** Cached epilogue operations (structural, same across steps). */
  epilogueOps: Array<{ kind: string; toDtype?: DType; inputIndex?: number; op?: string }>;
  /** Number of additional epilogue inputs (e.g., bias tensors). */
  epilogueInputCount: number;
  /** Output dtype after epilogue chain. */
  outputDtype: DType;
  /** Number of nodes consumed (matmul + epilogue chain). */
  consumedCount: number;
  /** Prologue info: which matmul inputs have absorbed casts. */
  prologues?: Array<{
    inputIndex: 0 | 1;
    castNodeIndex: number;  // Plan-node index of the cast node
    fromDtype: DType;
    toDtype: DType;
  }>;
  /** Cached label string (computed once, reused across steps). */
  cachedLabel?: string;

  /** Cached dispatch config computed on first lowered plan execution.
   *  On subsequent steps, the replay path resolves buffers and dispatches
   *  directly via dispatchMatmulDirect, skipping all intermediate functions.
   *
   *  IMPORTANT: Input refs are stored as plan-node-relative paths
   *  (planNodeIndex + inputIndex) rather than LazyRef objects. LazyRef objects
   *  become stale across steps because each step creates new LazyIRNode objects.
   *  The paths are resolved against the current step's planNodes array. */
  cachedDispatchConfig?: {
    /** Path to resolve input A: planNodes[planNodeIndex].inputs[inputIndex]. */
    inputAPath: { planNodeIndex: number; inputIndex: number };
    /** Path to resolve input B. */
    inputBPath: { planNodeIndex: number; inputIndex: number };
    /** Paths to resolve epilogue inputs (e.g., bias tensors). */
    epilogueInputPaths: Array<{ planNodeIndex: number; inputIndex: number }>;
    /** Prologue cast for input A (if cast was absorbed into matmul codegen). */
    inputCastA?: DType;
    /** Prologue cast for input B. */
    inputCastB?: DType;
    /** Pre-computed matmul geometry — stable across steps for same plan fingerprint. */
    m: number;
    k: number;
    n: number;
    transA: boolean;
    transB: boolean;
    batchSize: number;
    batchStrideA: number;
    batchStrideB: number;
    batchStrideC: number;
    outShape: number[];
    /** Dtype for matmul input A (post-prologue). */
    dtypeA: "f16" | "f32";
    /** Dtype for matmul input B (if different from A). */
    dtypeB?: "f16" | "f32";
    /** Output dtype after epilogue chain. */
    outputDtype: DType;
    /** Pre-computed epilogue config (structural, same across steps). */
    epilogueConfig: { ops: Array<{ kind: string; toDtype?: DType; inputIndex?: number; op?: string }>; additionalInputCount: number; outputDtype?: DType };
  };
}

/** A reduction with elementwise preamble fusion. */
export interface LoweredReductionPreambleAction {
  kind: "reduction-preamble";
  /** Plan-node index of the preamble (elementwise) node. */
  preambleNodeIndex: number;
  /** Plan-node index of the reduction (sum/mean) node. */
  reductionNodeIndex: number;
}

/** A view op (metadata only, no GPU dispatch). */
export interface LoweredViewAction {
  kind: "view";
  /** Plan-node index. */
  nodeIndex: number;
}

/** A data source op (tensorFromArray, zeros, full, arange). */
export interface LoweredDataSourceAction {
  kind: "data-source";
  /** Plan-node index. */
  nodeIndex: number;
}

/** A prologue-skipped cast node (absorbed into matmul). */
export interface LoweredPrologueSkipAction {
  kind: "prologue-skip";
  /** Plan-node index. */
  nodeIndex: number;
}

/** A batch of consecutive adamStep nodes. */
export interface LoweredAdamBatchAction {
  kind: "adam-batch";
  /** Plan-node indices for all consecutive adamStep nodes in this batch. */
  nodeIndices: number[];
}

/** A buffer reclaim point (flushSharedEncoder + flushBufferPool). */
export interface LoweredReclaimAction {
  kind: "reclaim";
}

/** Union of all lowered action types. */
export type LoweredAction =
  | LoweredFusedAction
  | LoweredSequentialAction
  | LoweredMatmulEpilogueAction
  | LoweredReductionPreambleAction
  | LoweredViewAction
  | LoweredDataSourceAction
  | LoweredPrologueSkipAction
  | LoweredAdamBatchAction
  | LoweredReclaimAction;

// ============================================================================
// Lowered Plan
// ============================================================================

/** A cached lowered execution plan. */
export interface LoweredPlan {
  /** Full action sequence in execution order. */
  actions: LoweredAction[];
  /** Total plan node count (for validation). */
  planNodeCount: number;
  /** Dispatch replay cache: recorded GPU dispatches for fast replay. */
  dispatchCache?: DispatchReplayCache;
}

// ============================================================================
// Dispatch Replay Cache
// ============================================================================

/**
 * A recorded GPU dispatch for replay. Contains everything needed to encode
 * a compute pass without any JS-level op dispatch logic.
 */
export interface ReplayDispatch {
  pipeline: any;  // GPUComputePipeline
  bindGroup: any; // GPUBindGroup
  workgroupsX: number;
  workgroupsY: number;
  workgroupsZ: number;
}

/**
 * Recorded node result metadata for reconstructing StorageHandles during replay.
 * Arena buffers are stable across steps, so we record the buffer reference directly.
 */
export interface ReplayNodeResult {
  nodeIndex: number;
  buffer: any;       // GPUBuffer (arena-stable)
  shape: number[];
  dtype: DType;
  size: number;
  strides: number[];
  /** If this result aliases an input (view ops), the aliased input's node index or -1 for materialized refs. */
  isView: boolean;
  baseNodeIndex?: number;
}

/**
 * A replay entry: either a GPU dispatch, a data source re-execution,
 * a view re-execution, a reclaim point, or a result assignment.
 */
export type ReplayEntry =
  | { kind: "dispatch"; dispatch: ReplayDispatch }
  | { kind: "data-source"; nodeIndex: number }
  | { kind: "view"; nodeIndex: number }
  | { kind: "sequential"; nodeIndex: number }
  | { kind: "result"; nodeResult: ReplayNodeResult }
  | { kind: "adam-batch"; nodeIndices: number[];
      /** Sequence counter positions at adam batch start (for correct cache indexing). */
      seqCounters: { dispatch: number; params: number; output: number } }
  | { kind: "reclaim" }
  | { kind: "pre-adam-reclaim" };

/**
 * Dispatch replay cache: records the full dispatch sequence during the first
 * execution and replays it on subsequent steps, bypassing all JS dispatch logic.
 */
export interface DispatchReplayCache {
  /** Ordered replay entries (dispatches, data sources, reclaims, result assignments). */
  entries: ReplayEntry[];
  /** Whether this cache is valid for replay. Set to false on invalidation. */
  valid: boolean;
}

// ============================================================================
// Data Source Ops
// ============================================================================

const DATA_SOURCE_OPS: ReadonlySet<string> = new Set([
  "tensorFromArray", "zeros", "full", "arange",
]);

/** View ops that produce views (no GPU dispatch, metadata only). */
const VIEW_OPS: ReadonlySet<string> = new Set([
  "reshape", "transpose", "permute", "expand", "narrow",
]);

// ============================================================================
// Lowered Plan Builder
// ============================================================================

/**
 * Builds a LoweredPlan by observing execution as it happens.
 * Call record*() methods during the first execution of a plan;
 * the builder captures the action sequence for replay.
 */
export class LoweredPlanBuilder {
  private actions: LoweredAction[] = [];
  private planNodeCount: number;

  constructor(planNodeCount: number) {
    this.planNodeCount = planNodeCount;
  }

  /** Record a fused elementwise kernel dispatch. */
  recordFused(
    coveredNodeIndices: number[],
    outputNodeIndex: number,
    additionalOutputNodeIndices: number[],
    neededIntermediateNodeIndices: number[],
    recipe: FusedKernelRecipe,
    enableVectorization: boolean,
  ): void {
    this.actions.push({
      kind: "fused",
      coveredNodeIndices,
      outputNodeIndex,
      additionalOutputNodeIndices,
      neededIntermediateNodeIndices,
      recipe,
      enableVectorization,
    });
  }

  /** Record a single sequential op dispatch. */
  recordSequential(nodeIndex: number): void {
    this.actions.push({
      kind: "sequential",
      nodeIndex,
    });
  }

  /** Record a matmul + epilogue chain. */
  recordMatmulEpilogue(
    matmulNodeIndex: number,
    coveredNodeIndices: number[],
    outputNodeIndex: number,
    epilogueOps: Array<{ kind: string; toDtype?: DType; inputIndex?: number; op?: string }>,
    epilogueInputCount: number,
    outputDtype: DType,
    consumedCount: number,
    prologues?: Array<{
      inputIndex: 0 | 1;
      castNodeIndex: number;
      fromDtype: DType;
      toDtype: DType;
    }>,
  ): void {
    this.actions.push({
      kind: "matmul-epilogue",
      matmulNodeIndex,
      coveredNodeIndices,
      outputNodeIndex,
      epilogueOps,
      epilogueInputCount,
      outputDtype,
      consumedCount,
      prologues,
    });
  }

  /** Record a reduction with elementwise preamble. */
  recordReductionPreamble(
    preambleNodeIndex: number,
    reductionNodeIndex: number,
  ): void {
    this.actions.push({
      kind: "reduction-preamble",
      preambleNodeIndex,
      reductionNodeIndex,
    });
  }

  /** Record a view op (metadata only). */
  recordView(nodeIndex: number): void {
    this.actions.push({
      kind: "view",
      nodeIndex,
    });
  }

  /** Record a data source op. */
  recordDataSource(nodeIndex: number): void {
    this.actions.push({
      kind: "data-source",
      nodeIndex,
    });
  }

  /** Record a prologue-skipped cast node. */
  recordPrologueSkip(nodeIndex: number): void {
    this.actions.push({
      kind: "prologue-skip",
      nodeIndex,
    });
  }

  /** Record an Adam batch (consecutive adamStep nodes). */
  recordAdamBatch(nodeIndices: number[]): void {
    this.actions.push({
      kind: "adam-batch",
      nodeIndices,
    });
  }

  /** Record a buffer reclaim point. */
  recordReclaim(): void {
    this.actions.push({
      kind: "reclaim",
    });
  }

  /** Build the final lowered plan. */
  build(): LoweredPlan {
    return {
      actions: this.actions,
      planNodeCount: this.planNodeCount,
    };
  }
}

// ============================================================================
// Helpers
// ============================================================================

/** Check if an op is a data source (creates buffers from host data). */
export function isDataSourceOp(op: string): boolean {
  return DATA_SOURCE_OPS.has(op);
}

/** Check if an op is a pure view (metadata only, no GPU dispatch). */
export function isViewOp(op: string): boolean {
  return VIEW_OPS.has(op);
}
