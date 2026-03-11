import type { Backend, DType } from "../backend/types";
import { isFusedBackend } from "../backend/types";
import { executeLoweredPlan } from "./executor-lowered";
import { executePlan } from "./executor-sequential";
import {
  buildIdPositionMap,
  collectExternalInputs,
  computePlanFingerprint,
  type ExecutionSegment,
  groupToRecipe,
  isFusibleOp,
} from "./fusion-detect";
import { analyzeGraph } from "./graph-compiler";
import type { ExecutionPlan, LazyIRNode, StorageHandle } from "./lazy-types";
import { buildLoweredPlanFromAnalysis, type LoweredPlan } from "./lowered-plan";
import { pretunePlanMatmuls } from "./plan-builder";
import {
  isProfilingEnabled,
  type PlanAnalysis,
  recordPlanAnalysis,
} from "./profiler";

/**
 * Options for optimized plan execution.
 */
interface OptimizedExecutionOptions {
  /** Enable elementwise fusion (default: true for WebGPU) */
  enableFusion?: boolean;
  /** Enable vectorization for fused kernels (default: true) */
  enableVectorization?: boolean;
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
}

/**
 * Statistics from optimized execution.
 */
export interface OptimizedExecutionStats {
  totalNodes: number;
  fusedNodes: number;
  sequentialNodes: number;
  fusionGroups: number;
  fusionEnabled: boolean;
}

/**
 * Result of optimized execution.
 */
export interface OptimizedExecutionResult {
  result: StorageHandle;
  stats: OptimizedExecutionStats;
}

// ============================================================================
// Fusion Analysis Cache
// ============================================================================

/**
 * Cached fusion analysis template. Stores the analysis result as position-based
 * indices that can be applied to any plan with the same structural fingerprint.
 */
export interface FusionAnalysisTemplate {
  /** Maps final plan position → original plan position.
   *  finalPlan[i] = originalPlan[finalPerm[i]] */
  finalPerm: number[];

  /** Segment pattern using positions in the final plan. */
  segments: CachedSegmentDesc[];

  /** Original plan positions that are epilogue-claimed. */
  epilogueClaimedOrigPoss: number[];
  /** Original plan positions that are prologue-claimed. */
  prologueClaimedOrigPoss: number[];

  /** Matmul epilogue chains: [origPos, [epilogueOrigPoss]]. */
  epilogueChains: Array<[number, number[]]>;

  /** Matmul prologues: [origPos, [{inputIndex, castOrigPos, fromDtype, toDtype}]]. */
  prologueDescs: Array<
    [
      number,
      Array<{
        inputIndex: 0 | 1;
        castOrigPos: number;
        fromDtype: DType;
        toDtype: DType;
      }>,
    ]
  >;

  /** Compound pattern matches (position-based: coveredOrigPoss, outputOrigPos, dim). */
  compoundDescs?: Array<{
    name: string;
    coveredOrigPoss: number[];
    outputOrigPos: number;
    dim: number;
  }>;

  /** Cached lifetime analysis (position-based). */
  lifetimeTemplate?: Array<{
    firstUse: number;
    lastUse: number;
    isOutput: boolean;
    isInput: boolean;
    bufferSize: number;
  }>;

  /** Cached lowered execution plan (built from graph analysis). */
  loweredPlan?: LoweredPlan;

  /** Per-plan buffer arena: GPUBuffers that persist across steps for bind group cache stability. */
  bufferArena?: unknown;
}

type CachedSegmentDesc =
  | { kind: "sequential"; finalPoss: number[] }
  | {
      kind: "fused";
      /** All group node positions in final plan. */
      finalPoss: number[];
      /** Output node position in final plan. */
      outputFinalPos: number;
      /** Additional output positions in final plan. */
      additionalOutputFinalPoss: number[];
      /** Needed intermediate positions in final plan. */
      neededIntermediateFinalPoss: number[];
    }
  | {
      kind: "reduction";
      /** All group node positions in final plan. */
      finalPoss: number[];
      /** Reduction node position. */
      reductionFinalPos: number;
      /** Preamble node positions. */
      preambleFinalPoss: number[];
      /** Epilogue node positions. */
      epilogueFinalPoss: number[];
      /** Output node position. */
      outputFinalPos: number;
      /** Serialized preamble ops. */
      preambleOps: Array<{ op: string; arity: number; chainInputPos?: 0 | 1 }>;
      /** Preamble input dtypes. */
      preambleInputDtypes: DType[];
      /** Serialized epilogue ops. */
      epilogueOps: Array<{
        kind: string;
        toDtype?: DType;
        inputIndex?: number;
        op?: string;
      }>;
      /** Output dtype. */
      outputDtype: DType;
      /** Whether this is a mean reduction. */
      isMean: boolean;
    };

/**
 * Module-level cache for fusion analysis results.
 * Keyed by structural fingerprint (FNV-1a hash).
 * Typically holds <10 entries (one per unique plan structure).
 */
const fusionAnalysisCache = new Map<number, FusionAnalysisTemplate>();

/** Get a cached fusion analysis template by fingerprint. */
export function getFusionAnalysisTemplate(
  fingerprint: number,
): FusionAnalysisTemplate | undefined {
  return fusionAnalysisCache.get(fingerprint);
}

/**
 * Execute a plan with automatic fusion optimization.
 *
 * Pipeline: analyze graph → build lowered plan → execute lowered plan.
 * The lowered plan is built purely from graph analysis (no execution needed)
 * and cached for subsequent steps. executeLoweredPlan() is the sole execution
 * engine for both first-run and replay paths.
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Optimization options
 */
export async function executePlanOptimized(
  plan: ExecutionPlan,
  backend: Backend & {
    device?: { limits?: { maxStorageBuffersPerShaderStage?: number } };
  },
  options: OptimizedExecutionOptions = {},
): Promise<OptimizedExecutionResult> {
  if (plan.nodes.length === 0) {
    throw new Error("Cannot execute empty plan");
  }

  // Pre-tune matmul shapes if backend supports it
  await pretunePlanMatmuls(plan, backend);

  const { enableFusion = isFusedBackend(backend), enableVectorization = true } =
    options;

  // Fall back to simple sequential execution when fusion is disabled entirely.
  if (!enableFusion) {
    const result = await executePlan(plan, backend, {
      enableEarlyRelease: options.enableEarlyRelease,
    });
    const stats: OptimizedExecutionStats = {
      totalNodes: plan.nodes.length,
      fusedNodes: 0,
      sequentialNodes: plan.nodes.length,
      fusionGroups: 0,
      fusionEnabled: enableFusion,
    };
    return { result, stats };
  }

  // Get node IDs with live external tensors (e.g., saved-for-backward)
  let externalNodeIds: Set<number> | undefined;
  try {
    const { getPendingNodeIds } = await import("../runtime/tensor");
    const pending = getPendingNodeIds();
    if (pending.size > 0) {
      externalNodeIds = pending;
    }
  } catch {
    // If runtime/tensor is not available, skip external node tracking
  }

  // Query device storage buffer limit to constrain fusion group size.
  const maxStorageBuffers: number | undefined =
    backend.device?.limits?.maxStorageBuffersPerShaderStage;

  // Compute structural fingerprint for fusion analysis caching.
  const fingerprint = computePlanFingerprint(plan.nodes, externalNodeIds);
  const cachedTemplate = fusionAnalysisCache.get(fingerprint);

  let planNodes: LazyIRNode[];
  let loweredPlan: LoweredPlan;

  if (cachedTemplate?.loweredPlan) {
    // ── Cache hit: reuse existing lowered plan ──
    planNodes = cachedTemplate.finalPerm.map((i) => plan.nodes[i]);
    loweredPlan = cachedTemplate.loweredPlan;
  } else {
    // ── Cache miss: run full analysis + build lowered plan ──

    const analysis = analyzeGraph(
      plan.nodes,
      externalNodeIds,
      maxStorageBuffers,
    );
    planNodes = analysis.planNodes;

    // Build template and cache it
    const origIdToPos = buildIdPositionMap(plan.nodes);
    const finalPerm = planNodes.map((n) => origIdToPos.get(n.id) as number);
    const finalIdToPos = buildIdPositionMap(planNodes);

    const cachedSegments: CachedSegmentDesc[] = analysis.segments.map((seg) => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          finalPoss: seg.nodes.map((n) => finalIdToPos.get(n.id) as number),
        };
      }
      if (seg.kind === "reduction") {
        const rg = seg.group;
        return {
          kind: "reduction" as const,
          finalPoss: rg.nodes.map((n) => finalIdToPos.get(n.id) as number),
          reductionFinalPos: finalIdToPos.get(rg.reductionNode.id) as number,
          preambleFinalPoss: rg.preambleNodes.map(
            (n) => finalIdToPos.get(n.id) as number,
          ),
          epilogueFinalPoss: rg.epilogueNodes.map(
            (n) => finalIdToPos.get(n.id) as number,
          ),
          outputFinalPos: finalIdToPos.get(rg.outputNode.id) as number,
          preambleOps: rg.preambleOps,
          preambleInputDtypes: rg.preambleInputDtypes,
          epilogueOps: rg.epilogueOps,
          outputDtype: rg.outputDtype,
          isMean: rg.isMean,
        };
      }
      return {
        kind: "fused" as const,
        finalPoss: seg.group.nodes.map((n) => finalIdToPos.get(n.id) as number),
        outputFinalPos: finalIdToPos.get(seg.group.outputNode.id) as number,
        additionalOutputFinalPoss: (seg.group.additionalOutputNodes ?? []).map(
          (n) => finalIdToPos.get(n.id) as number,
        ),
        neededIntermediateFinalPoss: (seg.group.neededIntermediates ?? []).map(
          (n) => finalIdToPos.get(n.id) as number,
        ),
      };
    });

    const template: FusionAnalysisTemplate = {
      finalPerm,
      segments: cachedSegments,
      epilogueClaimedOrigPoss: [...analysis.epilogueClaimedIds].map(
        (id) => origIdToPos.get(id) as number,
      ),
      prologueClaimedOrigPoss: [...analysis.prologueClaimedIds].map(
        (id) => origIdToPos.get(id) as number,
      ),
      epilogueChains: [...analysis.matmulEpilogueChains].map(
        ([mmId, epilogueIds]) =>
          [
            origIdToPos.get(mmId) as number,
            epilogueIds.map((id) => origIdToPos.get(id) as number),
          ] as [number, number[]],
      ),
      prologueDescs: [...analysis.matmulPrologues].map(
        ([mmId, prologues]) =>
          [
            origIdToPos.get(mmId) as number,
            prologues.map((p) => ({
              inputIndex: p.inputIndex,
              castOrigPos: origIdToPos.get(p.castNodeId) as number,
              fromDtype: p.fromDtype,
              toDtype: p.toDtype,
            })),
          ] as [
            number,
            Array<{
              inputIndex: 0 | 1;
              castOrigPos: number;
              fromDtype: DType;
              toDtype: DType;
            }>,
          ],
      ),
      compoundDescs:
        analysis.compoundMatches.length > 0
          ? analysis.compoundMatches.map((m) => ({
              name: m.name,
              coveredOrigPoss: m.coveredNodeIds.map(
                (id) => origIdToPos.get(id) as number,
              ),
              outputOrigPos: origIdToPos.get(m.outputNodeId) as number,
              dim: m.dim,
            }))
          : undefined,
    };

    // Build lowered plan from analysis (the sole plan-building path)
    loweredPlan = buildLoweredPlanFromAnalysis({
      segments: analysis.segments,
      planNodes,
      nodeIdToFinalPos: finalIdToPos,
      prologueClaimedIds: analysis.prologueClaimedIds,
      compoundMatches: analysis.compoundMatches,
      matmulDirectives: analysis.matmulDirectives,
      enableVectorization,
    });
    template.loweredPlan = loweredPlan;

    fusionAnalysisCache.set(fingerprint, template);

    // Collect plan analysis for profiling (structural, no execution needed)
    if (isProfilingEnabled()) {
      collectProfilingStats(
        analysis.segments,
        analysis.epilogueClaimedIds,
        analysis.prologueClaimedIds,
        analysis.matmulEpilogueChains,
        plan.nodes.length,
      );
    }
  }

  // Execute via the lowered plan — the sole execution engine
  const bufferArena = (cachedTemplate ?? fusionAnalysisCache.get(fingerprint))
    ?.bufferArena;
  return executeLoweredPlan(plan, planNodes, loweredPlan, backend, {
    bufferArena: bufferArena as
      | import("../backend/webgpu").BufferArena
      | undefined,
  });
}

// ============================================================================
// Profiling (structural analysis, no execution needed)
// ============================================================================

function collectProfilingStats(
  segments: ExecutionSegment[],
  epilogueClaimedIds: Set<number>,
  prologueClaimedIds: Set<number>,
  matmulEpilogueChains: Map<number, number[]>,
  totalNodes: number,
): void {
  let fusedSegCount = 0;
  let seqSegCount = 0;
  let fusedNodeCount = 0;
  let fusionGroupCount = 0;
  const sequentialOps: Record<string, number> = {};
  const unfusedByShape: Record<
    string,
    { count: number; ops: Record<string, number> }
  > = {};

  const recordUnfused = (node: LazyIRNode) => {
    sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
    if (isFusibleOp(node.op)) {
      const shapeKey = node.shape.join(",");
      let bucket = unfusedByShape[shapeKey];
      if (!bucket) {
        bucket = { count: 0, ops: {} };
        unfusedByShape[shapeKey] = bucket;
      }
      bucket.count++;
      bucket.ops[node.op] = (bucket.ops[node.op] ?? 0) + 1;
    }
  };

  for (const segment of segments) {
    if (segment.kind === "fused" && segment.group.nodes.length >= 2) {
      fusedSegCount++;
      fusedNodeCount += segment.group.nodes.length;
      fusionGroupCount++;
    } else if (segment.kind === "fused") {
      seqSegCount++;
      for (const node of segment.group.nodes) recordUnfused(node);
    } else if (segment.kind === "sequential") {
      seqSegCount++;
      for (const node of segment.nodes) {
        if (
          !epilogueClaimedIds.has(node.id) &&
          !prologueClaimedIds.has(node.id)
        ) {
          recordUnfused(node);
        } else {
          sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
        }
      }
    }
  }

  // Count reduction preamble opportunities from sequential segments
  let reductionFusionEstimate = 0;
  for (const segment of segments) {
    if (segment.kind !== "sequential") continue;
    for (let i = 0; i < segment.nodes.length - 1; i++) {
      const cur = segment.nodes[i];
      const next = segment.nodes[i + 1];
      if (
        isFusibleOp(cur.op) &&
        cur.op !== "cast" &&
        (next.op === "sum" || next.op === "mean")
      ) {
        reductionFusionEstimate++;
      }
    }
  }

  const planAnalysisRef: PlanAnalysis = {
    planIndex: 0, // assigned by recordPlanAnalysis
    totalNodes,
    segments: { fused: fusedSegCount, sequential: seqSegCount },
    fusedNodes: fusedNodeCount,
    fusionGroups: fusionGroupCount,
    epilogueFusions: matmulEpilogueChains.size,
    reductionFusions: reductionFusionEstimate,
    sequentialOps,
    unfusedByShape,
  };
  recordPlanAnalysis(planAnalysisRef);
}
