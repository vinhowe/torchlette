/**
 * Lowered Execution Plan
 *
 * A cached sequence of typed actions derived from graph analysis. On the first
 * execution with a given structural fingerprint, the lowered plan is built from
 * segments, matmul directives, compound patterns, etc. On subsequent steps it
 * is re-executed directly, skipping all analysis. A compiled plan (flat GPU
 * command sequence) is built after the second execution for even faster replay.
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
import type { FusedKernelRecipe } from "../backend/webgpu/fusion-types";
import type { EpilogueOp } from "./matmul-epilogue";

/** A path to resolve a tensor reference: planNodes[planNodeIndex].inputs[inputIndex]. */
type PlanNodePath = { planNodeIndex: number; inputIndex: number };

// ============================================================================
// Lowered Action Types
// ============================================================================

/** A fused elementwise kernel dispatch covering multiple plan nodes. */
interface LoweredFusedAction {
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
  cachedExternalInputPattern?: Array<{
    nodeLocalIdx: number;
    inputIdx: number;
  }>;
}

/** A single-node action (sequential op, view, data source, or prologue skip). */
interface LoweredNodeAction {
  kind: "sequential" | "view" | "data-source" | "prologue-skip";
  /** Plan-node index for this op. */
  nodeIndex: number;
}

/** A matmul + epilogue chain dispatch. */
interface LoweredMatmulEpilogueAction {
  kind: "matmul-epilogue";
  /** Plan-node index of the matmul node. */
  matmulNodeIndex: number;
  /** Plan-node indices consumed by the epilogue (matmul + chain). */
  coveredNodeIndices: number[];
  /** Index of the final output node. */
  outputNodeIndex: number;
  /** Cached epilogue operations (structural, same across steps). */
  epilogueOps: EpilogueOp[];
  /** Output dtype after epilogue chain. */
  outputDtype: DType;
  /** Number of nodes consumed (matmul + epilogue chain). */
  consumedCount: number;
  /** Prologue info: which matmul inputs have absorbed casts. */
  prologues?: Array<{
    inputIndex: 0 | 1;
    castNodeIndex: number; // Plan-node index of the cast node
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
    inputAPath: PlanNodePath;
    /** Path to resolve input B. */
    inputBPath: PlanNodePath;
    /** Paths to resolve epilogue inputs (e.g., bias tensors). */
    epilogueInputPaths: PlanNodePath[];
    /** Prologue cast for input A (if cast was absorbed into matmul codegen). */
    inputCastA?: "f16" | "f32";
    /** Prologue cast for input B. */
    inputCastB?: "f16" | "f32";
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
    epilogueConfig: {
      ops: EpilogueOp[];
      additionalInputCount: number;
      outputDtype?: DType;
    };
  };
}

/** A reduction with elementwise preamble fusion. */
interface LoweredReductionPreambleAction {
  kind: "reduction-preamble";
  /** Plan-node index of the preamble (elementwise) node. */
  preambleNodeIndex: number;
  /** Plan-node index of the reduction (sum/mean) node. */
  reductionNodeIndex: number;
  /** Plan-node indices for all chain nodes. */
  chainNodeIndices: number[];
  /** Kernel op descriptors for the chain. */
  chainOps: Array<{ op: string; arity: number; chainInputPos?: 0 | 1 }>;
  /** Dtypes for each external input. */
  chainInputDtypes: DType[];
  /** Total nodes consumed (chain + reduction). */
  consumedCount: number;
}

/** A reduction with elementwise epilogue fusion (sum/mean/max → cast/add/mul/activation chain). */
interface LoweredReductionEpilogueAction {
  kind: "reduction-epilogue";
  /** Plan-node index of the reduction (sum/mean/max) node. */
  reductionNodeIndex: number;
  /** Plan-node indices consumed by this pattern (reduction + epilogue chain). */
  coveredNodeIndices: number[];
  /** Index of the final output node in the plan. */
  outputNodeIndex: number;
  /** Epilogue operations to apply after the reduction. */
  epilogueOps: EpilogueOp[];
  /** Output dtype after epilogue chain. */
  outputDtype: DType;
  /** Number of nodes consumed (reduction + epilogue chain). */
  consumedCount: number;
}

/** A combined preamble + epilogue reduction fusion (elem → sum/mean → elem in one kernel). */
interface LoweredReductionFusionAction {
  kind: "reduction-fusion";
  /** Plan-node indices for all preamble chain nodes. */
  preambleNodeIndices: number[];
  /** Plan-node index of the reduction (sum/mean) node. */
  reductionNodeIndex: number;
  /** Plan-node indices for all epilogue chain nodes. */
  epilogueNodeIndices: number[];
  /** Index of the final output node in the plan. */
  outputNodeIndex: number;
  /** Kernel op descriptors for the preamble chain. */
  preambleOps: Array<{ op: string; arity: number; chainInputPos?: 0 | 1 }>;
  /** Dtypes for each preamble external input. */
  preambleInputDtypes: DType[];
  /** Epilogue operations to apply after the reduction. */
  epilogueOps: EpilogueOp[];
  /** Output dtype after epilogue chain. */
  outputDtype: DType;
  /** Total nodes consumed (preamble chain + reduction + epilogue chain). */
  consumedCount: number;
  /** Whether the reduction is a mean. */
  isMean: boolean;
}

/** A batch of consecutive adamStep nodes. */
interface LoweredAdamBatchAction {
  kind: "adam-batch";
  /** Plan-node indices for all consecutive adamStep nodes in this batch. */
  nodeIndices: number[];
}

/** A buffer reclaim point (flushSharedEncoder + flushBufferPool). */
interface LoweredReclaimAction {
  kind: "reclaim";
}

/** A compound pattern (softmax, log_softmax) replacing multiple decomposed ops. */
interface LoweredCompoundAction {
  kind: "compound";
  /** Pattern name (e.g. "softmax", "log_softmax"). */
  name: string;
  /** Plan-node indices consumed by this pattern. */
  coveredNodeIndices: number[];
  /** Index of the final output node. */
  outputNodeIndex: number;
  /** Reduction dimension. */
  dim: number;
}

/** A row-program fusion: multi-reduction + elementwise → single perRowKernel. */
interface LoweredRowProgramAction {
  kind: "row-program";
  /** Plan-node indices consumed by this row program. */
  coveredNodeIndices: number[];
  /** Index of the final output node. */
  outputNodeIndex: number;
  /** Reduction dimension (normalized, last dim). */
  dim: number;
  /** The RowProgram specification (structural, reusable across steps). */
  program: RowProgram;
  /** External input refs for resolving buffers at execution time. */
  inputRefs: LazyRef[];
}

/** Union of all lowered action types. */
type LoweredAction =
  | LoweredFusedAction
  | LoweredNodeAction
  | LoweredMatmulEpilogueAction
  | LoweredReductionPreambleAction
  | LoweredReductionEpilogueAction
  | LoweredReductionFusionAction
  | LoweredAdamBatchAction
  | LoweredReclaimAction
  | LoweredCompoundAction
  | LoweredRowProgramAction;

// ============================================================================
// Lowered Plan
// ============================================================================

/** A cached lowered execution plan. */
export interface LoweredPlan {
  /** Full action sequence in execution order. */
  actions: LoweredAction[];
  /** Total plan node count (for validation). */
  planNodeCount: number;
  /** Compiled execution plan: flat GPU command sequence. */
  compiledPlan?: import("./compiled-plan").CompiledPlan;
}

// ============================================================================
// Data Source Ops
// ============================================================================

const DATA_SOURCE_OPS: ReadonlySet<string> = new Set([
  "tensorFromArray",
  "zeros",
  "full",
  "arange",
  "rand",
  "randn",
  "bernoulli",
]);

/** View ops that produce views (no GPU dispatch, metadata only). */
const VIEW_OPS: ReadonlySet<string> = new Set([
  "reshape",
  "transpose",
  "permute",
  "expand",
  "narrow",
]);

/**
 * Sequential ops that use encoder copy commands (copyBufferToBuffer) alongside
 * compute dispatches. These must be re-executed during dispatch replay because
 * the copy commands are invisible to the compute dispatch recording mechanism.
 */
export const ENCODER_COPY_OPS: ReadonlySet<string> = new Set(["scatterAdd"]);

// (LoweredPlanBuilder removed — plans are now built from analysis alone
// via buildLoweredPlanFromAnalysis() below.)

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

// ============================================================================
// Analysis-Driven Lowered Plan Builder
// ============================================================================

import type { CompoundMatch } from "./compound-patterns";
import type { ExecutionSegment } from "./fusion-detect";
import type { LazyIRNode, LazyRef } from "./lazy-types";
import type { MatmulEpiloguePlan } from "./matmul-epilogue";
import type { ReductionGroup } from "./reduction-detect";
import type { RowProgram, RowProgramMatch } from "./row-program-types";

/** Default reclaim interval, overridable via TORCHLETTE_RECLAIM_INTERVAL env var. */
export const DEFAULT_RECLAIM_INTERVAL =
  typeof process !== "undefined" && process.env?.TORCHLETTE_RECLAIM_INTERVAL
    ? parseInt(process.env.TORCHLETTE_RECLAIM_INTERVAL, 10)
    : 10000;

interface BuildFromAnalysisInput {
  segments: ExecutionSegment[];
  planNodes: LazyIRNode[];
  nodeIdToFinalPos: Map<number, number>;
  prologueClaimedIds: Set<number>;
  compoundMatches: CompoundMatch[];
  rowProgramMatches: RowProgramMatch[];
  matmulDirectives: Map<number, MatmulEpiloguePlan>;
  enableVectorization: boolean;
  reclaimInterval?: number;
}

/**
 * Build a LoweredPlan purely from graph analysis results — no execution needed.
 * This is the sole path for building lowered plans. The action sequence is
 * derived entirely from the graph analysis (segments, matmul directives,
 * compound patterns, etc.) without executing any ops.
 *
 * Reclaim actions are inserted every `reclaimInterval` nodes to flush the
 * shared encoder and buffer pool periodically.
 */
export function buildLoweredPlanFromAnalysis(
  input: BuildFromAnalysisInput,
): LoweredPlan {
  const {
    segments,
    planNodes,
    nodeIdToFinalPos,
    prologueClaimedIds,
    compoundMatches,
    rowProgramMatches,
    matmulDirectives,
    enableVectorization,
    reclaimInterval = DEFAULT_RECLAIM_INTERVAL,
  } = input;

  const actions: LoweredAction[] = [];
  let nodesSinceReclaim = 0;

  /** Insert a reclaim action if enough nodes have been processed. */
  const maybeReclaim = (nodeCount: number) => {
    nodesSinceReclaim += nodeCount;
    if (nodesSinceReclaim >= reclaimInterval) {
      actions.push({ kind: "reclaim" });
      nodesSinceReclaim = 0;
    }
  };

  // Build compound match map
  let compoundMatchMap:
    | Map<
        number,
        {
          name: string;
          coveredNodeIds: Set<number>;
          outputNodeId: number;
          dim: number;
        }
      >
    | undefined;
  if (compoundMatches.length > 0) {
    compoundMatchMap = new Map();
    for (const match of compoundMatches) {
      let firstPos = Infinity;
      let firstId = match.coveredNodeIds[0];
      for (const id of match.coveredNodeIds) {
        const pos = nodeIdToFinalPos.get(id) ?? Infinity;
        if (pos < firstPos) {
          firstPos = pos;
          firstId = id;
        }
      }
      const coveredSet = new Set(match.coveredNodeIds);
      const desc = {
        name: match.name,
        coveredNodeIds: coveredSet,
        outputNodeId: match.outputNodeId,
        dim: match.dim,
      };
      compoundMatchMap.set(firstId, desc);
      for (const id of match.coveredNodeIds) {
        if (id !== firstId) compoundMatchMap.set(id, { ...desc, name: "" });
      }
    }
  }

  // Build row-program match map (same structure as compound matches)
  let rowProgramMatchMap:
    | Map<
        number,
        {
          match: RowProgramMatch;
          isFirst: boolean;
        }
      >
    | undefined;
  if (rowProgramMatches.length > 0) {
    rowProgramMatchMap = new Map();
    for (const match of rowProgramMatches) {
      let firstPos = Infinity;
      let firstId = match.coveredNodeIds[0];
      for (const id of match.coveredNodeIds) {
        const pos = nodeIdToFinalPos.get(id) ?? Infinity;
        if (pos < firstPos) {
          firstPos = pos;
          firstId = id;
        }
      }
      for (const id of match.coveredNodeIds) {
        rowProgramMatchMap.set(id, {
          match,
          isFirst: id === firstId,
        });
      }
    }
  }

  for (const segment of segments) {
    if (segment.kind === "reduction") {
      const nodeCount = segment.group.nodes.length;
      emitReductionActions(actions, segment.group, nodeIdToFinalPos);
      maybeReclaim(nodeCount);
    } else if (segment.kind === "fused" && segment.group.nodes.length >= 2) {
      const nodeCount = segment.group.nodes.length;
      emitFusedActions(actions, segment, nodeIdToFinalPos, enableVectorization);
      maybeReclaim(nodeCount);
    } else {
      // Sequential segment (or small fused group treated as sequential)
      const seqNodes =
        segment.kind === "fused" ? segment.group.nodes : segment.nodes;
      emitSequentialActions(
        actions,
        seqNodes,
        nodeIdToFinalPos,
        prologueClaimedIds,
        compoundMatchMap,
        rowProgramMatchMap,
        matmulDirectives,
        maybeReclaim,
      );
    }
  }

  return {
    actions,
    planNodeCount: planNodes.length,
  };
}

function emitReductionActions(
  actions: LoweredAction[],
  rg: ReductionGroup,
  posMap: Map<number, number>,
): void {
  const hasPreamble = rg.preambleNodes.length > 0;
  const hasEpilogue = rg.epilogueOps.length > 0;

  if (hasPreamble && hasEpilogue) {
    actions.push({
      kind: "reduction-fusion",
      preambleNodeIndices: rg.preambleNodes.map(
        (n) => posMap.get(n.id) as number,
      ),
      reductionNodeIndex: posMap.get(rg.reductionNode.id) as number,
      epilogueNodeIndices: rg.epilogueNodes.map(
        (n) => posMap.get(n.id) as number,
      ),
      outputNodeIndex: posMap.get(rg.outputNode.id) as number,
      preambleOps: rg.preambleOps,
      preambleInputDtypes: rg.preambleInputDtypes,
      epilogueOps: rg.epilogueOps,
      outputDtype: rg.outputDtype,
      consumedCount: rg.nodes.length,
      isMean: rg.isMean,
    });
  } else if (hasPreamble) {
    actions.push({
      kind: "reduction-preamble",
      preambleNodeIndex: posMap.get(rg.preambleNodes[0].id) as number,
      reductionNodeIndex: posMap.get(rg.reductionNode.id) as number,
      chainNodeIndices: rg.preambleNodes.map((n) => posMap.get(n.id) as number),
      chainOps: rg.preambleOps,
      chainInputDtypes: rg.preambleInputDtypes,
      consumedCount: rg.nodes.length,
    });
  } else if (hasEpilogue) {
    actions.push({
      kind: "reduction-epilogue",
      reductionNodeIndex: posMap.get(rg.reductionNode.id) as number,
      coveredNodeIndices: rg.nodes.map((n) => posMap.get(n.id) as number),
      outputNodeIndex: posMap.get(rg.outputNode.id) as number,
      epilogueOps: rg.epilogueOps,
      outputDtype: rg.outputDtype,
      consumedCount: rg.nodes.length,
    });
  }
}

function emitFusedActions(
  actions: LoweredAction[],
  segment: Extract<ExecutionSegment, { kind: "fused" }>,
  posMap: Map<number, number>,
  enableVectorization: boolean,
): void {
  actions.push({
    kind: "fused",
    coveredNodeIndices: segment.group.nodes.map(
      (n) => posMap.get(n.id) as number,
    ),
    outputNodeIndex: posMap.get(segment.group.outputNode.id) as number,
    additionalOutputNodeIndices: (
      segment.group.additionalOutputNodes ?? []
    ).map((n) => posMap.get(n.id) as number),
    neededIntermediateNodeIndices: (
      segment.group.neededIntermediates ?? []
    ).map((n) => posMap.get(n.id) as number),
    recipe: segment.recipe,
    enableVectorization,
  });
}

function emitSequentialActions(
  actions: LoweredAction[],
  nodes: LazyIRNode[],
  posMap: Map<number, number>,
  prologueSkipIds: Set<number>,
  compoundMatchMap:
    | Map<
        number,
        {
          name: string;
          coveredNodeIds: Set<number>;
          outputNodeId: number;
          dim: number;
        }
      >
    | undefined,
  rowProgramMatchMap:
    | Map<number, { match: RowProgramMatch; isFirst: boolean }>
    | undefined,
  matmulDirectives: Map<number, MatmulEpiloguePlan>,
  maybeReclaim: (count: number) => void,
): void {
  for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
    const node = nodes[nodeIdx];

    // Already materialized — skip (matches `if (node.result) continue` in execution)
    if (node.result) continue;

    // Prologue-claimed cast: absorbed into matmul
    if (prologueSkipIds.has(node.id)) {
      actions.push({
        kind: "prologue-skip",
        nodeIndex: posMap.get(node.id) as number,
      });
      maybeReclaim(1);
      continue;
    }

    // Row-program fusion (multi-reduction → single perRowKernel)
    if (rowProgramMatchMap?.has(node.id)) {
      const entry = rowProgramMatchMap.get(node.id)!;
      if (!entry.isFirst) {
        // Intermediate node — skip (covered by the row-program action)
        actions.push({
          kind: "prologue-skip",
          nodeIndex: posMap.get(node.id) as number,
        });
        continue;
      }
      const m = entry.match;
      actions.push({
        kind: "row-program",
        coveredNodeIndices: m.coveredNodeIds.map(
          (id) => posMap.get(id) as number,
        ),
        outputNodeIndex: posMap.get(m.outputNodeId) as number,
        dim: m.dim,
        program: m.program,
        inputRefs: m.inputRefs,
      });
      maybeReclaim(m.coveredNodeIds.length);
      continue;
    }

    // Compound patterns (softmax, log_softmax)
    if (compoundMatchMap?.has(node.id)) {
      const match = compoundMatchMap.get(node.id)!;
      if (match.name === "") {
        // Intermediate covered node — skip
        actions.push({
          kind: "prologue-skip",
          nodeIndex: posMap.get(node.id) as number,
        });
        continue;
      }

      const coveredCount = match.coveredNodeIds.size;
      const coveredPoss: number[] = [];
      for (let c = 0; c < coveredCount; c++) {
        coveredPoss.push(posMap.get(nodes[nodeIdx + c].id) as number);
      }
      actions.push({
        kind: "compound",
        name: match.name,
        coveredNodeIndices: coveredPoss,
        outputNodeIndex: posMap.get(match.outputNodeId) as number,
        dim: match.dim,
      });
      maybeReclaim(coveredCount);
      nodeIdx += coveredCount - 1;
      continue;
    }

    // Matmul with epilogue/prologue
    if (node.op === "matmul") {
      const epiloguePlan = matmulDirectives.get(node.id);
      if (epiloguePlan) {
        const covered: number[] = [];
        for (let c = 0; c < epiloguePlan.consumedCount; c++) {
          covered.push(posMap.get(nodes[nodeIdx + c].id) as number);
        }
        actions.push({
          kind: "matmul-epilogue",
          matmulNodeIndex: posMap.get(node.id) as number,
          coveredNodeIndices: covered,
          outputNodeIndex: posMap.get(epiloguePlan.outputNode.id) as number,
          epilogueOps: epiloguePlan.epilogueOps,
          outputDtype: epiloguePlan.outputDtype,
          consumedCount: epiloguePlan.consumedCount,
          prologues: epiloguePlan.prologues?.map((p) => ({
            inputIndex: p.inputIndex,
            castNodeIndex: posMap.get(p.castNodeId) as number,
            fromDtype: p.fromDtype,
            toDtype: p.toDtype,
          })),
        });
        maybeReclaim(epiloguePlan.consumedCount);
        nodeIdx += epiloguePlan.consumedCount - 1;
        continue;
      }
    }

    // Adam batch: count consecutive adamStep nodes
    if (node.op === "adamStep") {
      let adamCount = 1;
      for (let j = nodeIdx + 1; j < nodes.length; j++) {
        if (nodes[j].op === "adamStep" && !nodes[j].result) adamCount++;
        else break;
      }
      if (adamCount > 1) {
        const adamIndices: number[] = [];
        for (let c = 0; c < adamCount; c++) {
          adamIndices.push(posMap.get(nodes[nodeIdx + c].id) as number);
        }
        actions.push({ kind: "adam-batch", nodeIndices: adamIndices });
        maybeReclaim(adamCount);
        nodeIdx += adamCount - 1;
        continue;
      }
    }

    // Regular node: classify as data-source, view, or sequential
    const kind = isDataSourceOp(node.op)
      ? ("data-source" as const)
      : isViewOp(node.op)
        ? ("view" as const)
        : ("sequential" as const);
    actions.push({ kind, nodeIndex: posMap.get(node.id) as number });
    maybeReclaim(1);
  }
}
