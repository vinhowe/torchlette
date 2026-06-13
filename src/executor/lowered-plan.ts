/**
 * Lowered Execution Plan
 *
 * A cached sequence of typed actions derived from graph analysis. On the first
 * execution with a given structural fingerprint, the lowered plan is built from
 * segments, matmul directives, row programs, etc. On subsequent steps it
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

import { ENV } from "../core/env";
import type { DType } from "../backend/types";
import type { FusedKernelRecipe } from "../backend/webgpu/fusion-types";
import type { EpilogueOp } from "../compiler/matmul-epilogue";
import { collectScalarSlots } from "./scalar-table";

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
  /**
   * Recipe input indices demoted from inlined constants to runtime inputs
   * because their values were observed to CHANGE across template reuses
   * (frozen-scalar adaptation — see the executor's fused case).
   */
  runtimeScalarInputs?: Set<number>;
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
  /** Stage-4 phase-3: bare-matmul geometry captured at first (live)
   *  execution, read by the stream generator post-hoc (matmul inputs are
   *  liveness-released by plan-build). A string is a bail reason
   *  (contiguous-prologue / ksplit) — uncovered, keeps record/replay.
   *  Geometry is shape/dtype-pure → invariant within a template. */
  cachedMatmulPlan?: import("../backend/webgpu/dispatch").BareMatmulPlan | string;
  /** Stage-4 phase-3: input shapes captured at lowering for ops whose
   *  generator can't derive them post-hoc (released multi-output inputs);
   *  e.g. narrowBackward's grad dim size. Template-invariant. */
  cachedInputShapes?: number[][];
  /** Stage-4 phase-3: whether every input was contiguous at lowering
   *  (attention ops asContiguous internally; a copy prologue isn't yet
   *  generated, so the generator bails when false). Template-invariant. */
  cachedAllInputsContig?: boolean;
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

/** A batch of consecutive adamStep nodes. */
interface LoweredAdamBatchAction {
  kind: "adam-batch";
  /** Plan-node indices for all consecutive adamStep nodes in this batch. */
  nodeIndices: number[];
}

/** A batch of consecutive same-config reduction nodes. */
interface LoweredBatchedReductionAction {
  kind: "batched-reduction";
  /** Plan-node indices for all reduction nodes in this batch. */
  nodeIndices: number[];
  /** Reduction op (sum, max, min). */
  reduceOp: string;
  /** Reduction config from the first node's payload. */
  payload: { dim: number | number[]; keepdim?: boolean };
}

/** A buffer reclaim point (flushSharedEncoder + flushBufferPool). */
interface LoweredReclaimAction {
  kind: "reclaim";
}

/** A row-program fusion: reduction(s) + elementwise → single perRowKernel. */
interface LoweredRowProgramAction {
  kind: "row-program";
  /** Plan-node indices consumed by this row program. */
  coveredNodeIndices: number[];
  /** Index of the final output node. */
  outputNodeIndex: number;
  /** Reduction dimension (normalized, last dim). */
  dim: number;
  /** Number of rows (from the first reduction's input shape). */
  numRows: number;
  /** Size of the reduction dimension. */
  dimSize: number;
  /** The RowProgram specification (structural, reusable across steps). */
  program: RowProgram;
  /** External input refs for resolving buffers at execution time. */
  inputRefs: LazyRef[];
  /**
   * Plan-node positions for each pending inputRef.
   * Maps inputRef index → plan-node index (or -1 if materialized/scalar).
   * Used by executeLoweredPlan to remap stale inputRefs to current-step nodes.
   */
  inputRefPositions: number[];
}

/** Union of all lowered action types. */
type LoweredAction =
  | LoweredFusedAction
  | LoweredNodeAction
  | LoweredMatmulEpilogueAction
  | LoweredAdamBatchAction
  | LoweredBatchedReductionAction
  | LoweredReclaimAction
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
  /** Cached fusion stats from the normal-path execution that built the compiled plan.
   *  Returned by the compiled plan fast path so cumulative stats stay accurate. */
  cachedStats?: import("./executor").OptimizedExecutionStats;
  /** Compiled execution plan: flat GPU command sequence. */
  compiledPlan?: import("./compiled-plan").CompiledPlan;
  /** Structural positions of f32 scalar refs — refreshed as DATA every
   *  execution so template/compiled caches stay value-independent
   *  (see scalar-table.ts and docs/architecture-debt.md stage 1). */
  scalarSlots?: import("./scalar-table").ScalarSlot[];
  /** Per-template scalar buffers (created lazily by the executor). */
  scalarTable?: import("./scalar-table").PlanScalarTable;
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

/**
 * Merge non-adjacent same-config sequential reduction actions into batched
 * dispatches. Safe when no intervening action consumes the output of an
 * earlier reduction in the group (i.e., the reductions are independent).
 *
 * Replaces individual sequential actions with a single batched-reduction
 * action at the position of the LAST member. Earlier members become
 * prologue-skip (no-op). This preserves topological order — the batch
 * executes only after all its members' inputs are ready.
 */
function mergeBatchableReductions(
  actions: LoweredAction[],
  planNodes: LazyIRNode[],
): void {
  // 1. Find sequential reduction actions and group by config key
  const groups = new Map<
    string,
    Array<{ actionIndex: number; nodeIndex: number }>
  >();
  for (let i = 0; i < actions.length; i++) {
    const a = actions[i];
    if (a.kind !== "sequential") continue;
    const node = planNodes[a.nodeIndex];
    if (!isReductionOp(node.op) || node.payload?.dim == null) continue;
    const key = reductionDimKey(node);
    let group = groups.get(key);
    if (!group) {
      group = [];
      groups.set(key, group);
    }
    group.push({ actionIndex: i, nodeIndex: a.nodeIndex });
  }

  // 2. For each group with ≥2 members, check independence and merge
  for (const [, members] of groups) {
    if (members.length < 2) continue;

    // Collect node indices in this group
    const groupNodeIndices = new Set(members.map((m) => m.nodeIndex));

    // Check: no action between first and last member reads from any member's output.
    // An action reads from a node if any of its plan nodes' inputs reference a group member.
    const firstIdx = members[0].actionIndex;
    const lastIdx = members[members.length - 1].actionIndex;
    let safe = true;
    for (let i = firstIdx + 1; i < lastIdx && safe; i++) {
      const a = actions[i];
      const nodeIndices = getActionNodeIndices(a);
      for (const ni of nodeIndices) {
        if (groupNodeIndices.has(ni)) continue; // skip self
        const node = planNodes[ni];
        if (!node) continue;
        for (const inp of node.inputs) {
          if (inp.kind === "pending" && groupNodeIndices.has(inp.node.id)) {
            // This fails if node IDs don't match plan indices.
            // Node IDs and plan indices are different — need to map.
            // Actually, inp.node is the LazyIRNode itself, and we stored
            // planNode indices in groupNodeIndices. Let's check planNodes instead.
          }
        }
      }
    }

    // Safety check: no action between first and last member can depend
    // on any group member's output. This means the sum outputs must only
    // be consumed AFTER the last member's position (e.g., by the optimizer).
    safe = true;
    for (let mi = 0; mi < members.length - 1 && safe; mi++) {
      const memberNode = planNodes[members[mi].nodeIndex];
      for (let ai = members[mi].actionIndex + 1; ai <= lastIdx; ai++) {
        for (const ni of getActionNodeIndices(actions[ai])) {
          if (groupNodeIndices.has(ni)) continue;
          const depNode = planNodes[ni];
          if (!depNode) continue;
          for (const inp of depNode.inputs) {
            if (inp.kind === "pending" && inp.node === memberNode) {
              safe = false;
              break;
            }
          }
          if (!safe) break;
        }
        if (!safe) break;
      }
    }

    if (!safe) continue;

    // 3. Merge: replace last member with batch, earlier members become no-ops
    const firstNode = planNodes[members[0].nodeIndex];
    actions[lastIdx] = {
      kind: "batched-reduction",
      nodeIndices: members.map((m) => m.nodeIndex),
      reduceOp: firstNode.op,
      payload: {
        dim: firstNode.payload.dim,
        keepdim: firstNode.payload.keepdim,
      },
    };
    for (let mi = 0; mi < members.length - 1; mi++) {
      actions[members[mi].actionIndex] = {
        kind: "prologue-skip",
        nodeIndex: members[mi].nodeIndex,
      };
    }
  }
}

/** Get all plan-node indices referenced by an action. */
export function getActionNodeIndices(action: LoweredAction): number[] {
  switch (action.kind) {
    case "sequential":
    case "view":
    case "data-source":
    case "prologue-skip":
      return [action.nodeIndex];
    case "fused":
      return action.coveredNodeIndices;
    case "matmul-epilogue":
      return action.coveredNodeIndices;
    case "adam-batch":
      return action.nodeIndices;
    case "batched-reduction":
      return action.nodeIndices;
    case "row-program":
      return action.coveredNodeIndices;
    case "reclaim":
      return [];
  }
}

// (LoweredPlanBuilder removed — plans are now built from analysis alone
// via buildLoweredPlanFromAnalysis() below.)

// ============================================================================
// Helpers
// ============================================================================

const REDUCTION_OPS: ReadonlySet<string> = new Set([
  "sum",
  "max",
  "min",
  "mean",
]);

function isReductionOp(op: string): boolean {
  return REDUCTION_OPS.has(op);
}

/** Stable key for matching consecutive reductions with the same config. */
function reductionDimKey(node: {
  op: string;
  payload?: any;
  shape: number[];
}): string {
  const p = node.payload;
  const dims = Array.isArray(p?.dim) ? p.dim.join(",") : String(p?.dim);
  return `${node.op}:${dims}:${p?.keepdim ?? false}:${node.shape.join("x")}`;
}

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

import type { ExecutionSegment } from "../compiler/fusion-detect";
import type { MatmulEpiloguePlan } from "../compiler/matmul-epilogue";
import type {
  RowProgram,
  RowProgramMatch,
} from "../compiler/row-program-types";
import type { LazyIRNode, LazyRef } from "../graph/types";

/** Default reclaim interval, overridable via TORCHLETTE_RECLAIM_INTERVAL env var. */
export const DEFAULT_RECLAIM_INTERVAL =
  ENV.TORCHLETTE_RECLAIM_INTERVAL
    ? parseInt(ENV.TORCHLETTE_RECLAIM_INTERVAL, 10)
    : 10000;

interface BuildFromAnalysisInput {
  segments: ExecutionSegment[];
  planNodes: LazyIRNode[];
  nodeIdToFinalPos: Map<number, number>;
  prologueClaimedIds: Set<number>;
  rowProgramMatches: RowProgramMatch[];
  matmulDirectives: Map<number, MatmulEpiloguePlan>;
  enableVectorization: boolean;
  reclaimInterval?: number;
}

/**
 * Build a LoweredPlan purely from graph analysis results — no execution needed.
 * This is the sole path for building lowered plans. The action sequence is
 * derived entirely from the graph analysis (segments, matmul directives,
 * row programs, etc.) without executing any ops.
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
    rowProgramMatches,
    matmulDirectives,
    enableVectorization,
    reclaimInterval = DEFAULT_RECLAIM_INTERVAL,
  } = input;

  const actions: LoweredAction[] = [];
  let nodesSinceReclaim = 0;

  if (
    ENV.TORCHLETTE_DEBUG_OPTPLAN &&
    planNodes.some((n) => n.op === "adamStep")
  ) {
    console.log(`[optplan] ${planNodes.map((n) => n.op).join(",")}`);
  }

  /** Insert a reclaim action if enough nodes have been processed. */
  const maybeReclaim = (nodeCount: number) => {
    nodesSinceReclaim += nodeCount;
    if (nodesSinceReclaim >= reclaimInterval) {
      actions.push({ kind: "reclaim" });
      nodesSinceReclaim = 0;
    }
  };

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
    if (segment.kind === "fused" && segment.group.nodes.length >= 2) {
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
        rowProgramMatchMap,
        matmulDirectives,
        maybeReclaim,
      );
    }
  }

  // Post-process: merge non-adjacent same-config reduction actions into batches.
  // Only merges sums whose outputs have no consumers between group members.
  // Currently fires for plans where sums are naturally adjacent. For the common
  // bias-gradient pattern (sum interleaved with matmul + reshape), the view
  // dependents prevent merging. To enable: fuse sumToShape into a single node.
  mergeBatchableReductions(actions, planNodes);

  // Collect f32 scalar-ref positions: their VALUES may change every step
  // (optimizer bias correction, scheduled LR) while the template fingerprint
  // deliberately ignores them. The executor refreshes these as DATA each
  // execution (scalar-table.ts) so every cache keyed under the fingerprint
  // stays value-independent.
  const scalarSlots = collectScalarSlots(planNodes);

  return {
    actions,
    planNodeCount: planNodes.length,
    scalarSlots: scalarSlots.length > 0 ? scalarSlots : undefined,
  };
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
      // Map each inputRef to its plan-node position so executeLoweredPlan
      // can remap stale refs to the current step's nodes on template reuse.
      const inputRefPositions = m.inputRefs.map((ref) => {
        if (ref.kind === "pending") {
          return posMap.get(ref.node.id) ?? -1;
        }
        return -1; // materialized or scalar — no remapping needed
      });
      actions.push({
        kind: "row-program",
        coveredNodeIndices: m.coveredNodeIds.map(
          (id) => posMap.get(id) as number,
        ),
        outputNodeIndex: posMap.get(m.outputNodeId) as number,
        dim: m.dim,
        numRows: m.numRows,
        dimSize: m.dimSize,
        program: m.program,
        inputRefs: m.inputRefs,
        inputRefPositions,
      });
      maybeReclaim(m.coveredNodeIds.length);
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

    // Adam batch: collect the CONSECUTIVE run of adamStep nodes.
    //
    // Always emit an adam-batch action for adamStep, even when there's only
    // one node. adam-batch routes through backend.ops.adamStepBatch, which
    // returns BackendTensors that the executor wraps via assignNodeResult
    // (the shared multi-output protocol). This ensures node.result and
    // node.results[0] are set together — the historical "Input not ready:
    // adamStep[0]" bug came from a generic-sequential path that left
    // node.results[0] = null until a later wrap-and-backfill that wasn't
    // always reached.
    //
    // Adjacency comes from the GENERIC same-op affinity tie-break in the
    // plan builder's scheduling pass (enforceWriteAfterReadOrder): the DFS
    // order interleaves each param's producer (unscaleGrad, clip's
    // copy-back) with its adamStep, and the scheduler regroups independent
    // same-op sequential dispatches adjacently for ANY op — no op-name
    // whitelist in the executor (the old ADAM_HOISTABLE_OPS hoisting).
    if (node.op === "adamStep") {
      const adamIndices: number[] = [];
      let consumed = 0;
      for (let j = nodeIdx; j < nodes.length; j++) {
        const nj = nodes[j];
        if (nj.op !== "adamStep") break;
        if (!nj.result) {
          adamIndices.push(posMap.get(nj.id) as number);
        }
        consumed++;
      }
      actions.push({ kind: "adam-batch", nodeIndices: adamIndices });
      maybeReclaim(consumed);
      nodeIdx += consumed - 1;
      continue;
    }

    // Batched reduction: count consecutive same-config sum/max/min nodes
    if (isReductionOp(node.op) && node.payload?.dim != null) {
      const dimKey = reductionDimKey(node);
      let batchCount = 1;
      for (let j = nodeIdx + 1; j < nodes.length; j++) {
        const next = nodes[j];
        if (
          next.op === node.op &&
          !next.result &&
          next.payload?.dim != null &&
          reductionDimKey(next) === dimKey
        )
          batchCount++;
        else break;
      }
      if (batchCount > 1) {
        const indices: number[] = [];
        for (let c = 0; c < batchCount; c++) {
          indices.push(posMap.get(nodes[nodeIdx + c].id) as number);
        }
        actions.push({
          kind: "batched-reduction",
          nodeIndices: indices,
          reduceOp: node.op,
          payload: { dim: node.payload.dim, keepdim: node.payload.keepdim },
        });
        maybeReclaim(batchCount);
        nodeIdx += batchCount - 1;
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
