import type { Backend, DType } from "../backend/types";
import {
  type BufferArena,
  beginSharedEncoder,
  clearActiveArena,
  clearArenaExternalInputBuffers,
  endSharedEncoder,
  setActiveArena,
  setArenaExternalInputBuffers,
} from "../backend/webgpu";
import {
  isProfilingEnabled,
  type PlanAnalysis,
  recordPlanAnalysis,
} from "../backend/webgpu/profiler";
import type { CompoundMatch } from "./compound-patterns";
import { collectExternalInputBuffers } from "./executor-lowered";
import { executePlan } from "./executor-sequential";
import {
  buildIdPositionMap,
  collectExternalInputs,
  computePlanFingerprint,
  type ExecutionSegment,
  type FusionGroup,
  groupToRecipe,
  isFusibleOp,
} from "./fusion-detect";
import { analyzeGraph } from "./graph-compiler";
import type { ExecutionPlan, LazyIRNode, StorageHandle } from "./lazy-types";
import type { TensorLifetime } from "./lifetime-analysis";
import { type LoweredPlan, LoweredPlanBuilder } from "./lowered-plan";
import type { MatmulPrologueInfo } from "./matmul-epilogue";
import { initLifetimeAnalysis, pretunePlanMatmuls } from "./plan-builder";
import type { ReductionGroup } from "./reduction-detect";
import {
  type CompoundMatchExec,
  createReclaimController,
  DEFAULT_RECLAIM_INTERVAL,
  executeFusedSegment,
  executeReductionSegment,
  executeSequentialSegmentWithEarlyRelease,
} from "./segment-executors";
import { releaseDeadTensors } from "./storage-tracker";

/**
 * Collect external input refs from a chain of nodes.
 * For each node, inputs that don't reference the previous chain node are external.
 * Used when reconstructing ReductionGroup from cached template on cache hit.
 */
function collectChainExternalRefsFromNodes(
  chainNodes: LazyIRNode[],
  skipNodeId?: number,
): LazyIRNode["inputs"] {
  const refs: LazyIRNode["inputs"] = [];
  const chainNodeIds = new Set(chainNodes.map((n) => n.id));
  if (skipNodeId !== undefined) chainNodeIds.add(skipNodeId);
  for (const node of chainNodes) {
    for (const ref of node.inputs) {
      if (ref.kind === "pending" && chainNodeIds.has(ref.node.id)) continue;
      refs.push(ref);
    }
  }
  return refs;
}

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

  /** Cached lowered execution plan (built during first execution on cache miss). */
  loweredPlan?: LoweredPlan;

  /** Per-plan buffer arena: GPUBuffers that persist across steps for bind group cache stability. */
  bufferArena?: BufferArena;
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
 * Per spec §15, this:
 * - Detects fusible elementwise op chains
 * - Dispatches fused WebGPU kernels when beneficial
 * - Falls back to sequential execution for non-fusible ops
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

  const {
    enableFusion = backend.name === "webgpu",
    enableVectorization = true,
    enableEarlyRelease = false,
  } = options;

  const stats: OptimizedExecutionStats = {
    totalNodes: plan.nodes.length,
    fusedNodes: 0,
    sequentialNodes: 0,
    fusionGroups: 0,
    fusionEnabled: enableFusion,
  };

  // Fall back to simple sequential execution when fusion is disabled entirely.
  // When fusion IS enabled, always go through the full analysis path even if
  // no fusible ops exist — this creates lowered plan templates (for adam batching,
  // arena allocation, bind group cache stability) that benefit all plan types.
  if (!enableFusion) {
    const result = await executePlan(plan, backend, { enableEarlyRelease });
    stats.sequentialNodes = plan.nodes.length;
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
  // Lifetime analysis is set up after analysis (below)
  let lifetimes: Map<number, TensorLifetime> | null = null;
  let outputNodeIds: Set<number> | null = null;
  const alreadyReleased = new Set<number>();
  const nodeToStorage = new Map<number, StorageHandle>();

  // Query device storage buffer limit to constrain fusion group size.
  const maxStorageBuffers: number | undefined =
    backend.device?.limits?.maxStorageBuffersPerShaderStage;

  // Compute structural fingerprint for fusion analysis caching.
  // Plans with the same fingerprint have identical structure (ops, shapes,
  // dtypes, dependency graph) and can reuse cached analysis results.
  const fingerprint = computePlanFingerprint(plan.nodes, externalNodeIds);
  const cachedTemplate = fusionAnalysisCache.get(fingerprint);

  let planNodes: LazyIRNode[];
  let segments: ExecutionSegment[];
  const epilogueClaimedIds = new Set<number>();
  const prologueClaimedIds = new Set<number>();
  const matmulEpilogueChains = new Map<number, number[]>();
  const matmulPrologues = new Map<number, MatmulPrologueInfo[]>();
  const compoundClaimedIds = new Set<number>();
  let compoundMatches: CompoundMatch[] = [];
  let matmulDirectives:
    | ReturnType<typeof analyzeGraph>["matmulDirectives"]
    | undefined;

  if (cachedTemplate) {
    // ── Cache hit: reconstruct from template ──
    planNodes = cachedTemplate.finalPerm.map((i) => plan.nodes[i]);

    // Reconstruct epilogue/prologue ID sets
    for (const pos of cachedTemplate.epilogueClaimedOrigPoss) {
      epilogueClaimedIds.add(plan.nodes[pos].id);
    }
    for (const pos of cachedTemplate.prologueClaimedOrigPoss) {
      prologueClaimedIds.add(plan.nodes[pos].id);
    }
    for (const [matmulPos, epiloguePoss] of cachedTemplate.epilogueChains) {
      matmulEpilogueChains.set(
        plan.nodes[matmulPos].id,
        epiloguePoss.map((p) => plan.nodes[p].id),
      );
    }
    for (const [matmulPos, descs] of cachedTemplate.prologueDescs) {
      matmulPrologues.set(
        plan.nodes[matmulPos].id,
        descs.map((d) => ({
          inputIndex: d.inputIndex,
          castNodeId: plan.nodes[d.castOrigPos].id,
          originalInputRef: plan.nodes[d.castOrigPos].inputs[0],
          fromDtype: d.fromDtype,
          toDtype: d.toDtype,
        })),
      );
    }

    // Reconstruct compound pattern matches from cached template
    if (cachedTemplate.compoundDescs) {
      for (const desc of cachedTemplate.compoundDescs) {
        const coveredIds = desc.coveredOrigPoss.map((p) => plan.nodes[p].id);
        compoundMatches.push({
          name: desc.name,
          coveredNodeIds: coveredIds,
          outputNodeId: plan.nodes[desc.outputOrigPos].id,
          dim: desc.dim,
          inputNodeId: -1, // Not needed for execution
          inputIsMaterialized: false,
        });
        for (const id of coveredIds) compoundClaimedIds.add(id);
      }
    }

    // Reconstruct lifetime analysis from cached template (avoids initLifetimeAnalysis)
    if (cachedTemplate.lifetimeTemplate && enableEarlyRelease) {
      lifetimes = new Map();
      for (let i = 0; i < planNodes.length; i++) {
        const t = cachedTemplate.lifetimeTemplate[i];
        lifetimes.set(planNodes[i].id, {
          nodeId: planNodes[i].id,
          firstUse: t.firstUse,
          lastUse: t.lastUse,
          isOutput: t.isOutput,
          isInput: t.isInput,
          bufferSize: t.bufferSize,
        });
      }
      outputNodeIds = new Set([planNodes[planNodes.length - 1].id]);
      if (externalNodeIds) {
        for (const id of externalNodeIds) outputNodeIds.add(id);
      }
    }

    // Reconstruct segments from cached pattern
    segments = cachedTemplate.segments.map((seg) => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          nodes: seg.finalPoss.map((i) => planNodes[i]),
        };
      }
      if (seg.kind === "reduction") {
        // Reconstruct ReductionGroup from cached positions
        const groupNodes = seg.finalPoss.map((i) => planNodes[i]);
        const preambleNodes = seg.preambleFinalPoss.map((i) => planNodes[i]);
        const epilogueNodes = seg.epilogueFinalPoss.map((i) => planNodes[i]);
        const reductionNode = planNodes[seg.reductionFinalPos];
        const outputNode = planNodes[seg.outputFinalPos];

        // Reconstruct external input refs from preamble/epilogue nodes
        const preambleInputRefs =
          collectChainExternalRefsFromNodes(preambleNodes);
        const epilogueInputRefs = collectChainExternalRefsFromNodes(
          epilogueNodes,
          preambleNodes.length > 0 ? undefined : reductionNode.id,
        );

        const group: ReductionGroup = {
          nodes: groupNodes,
          planIndices: seg.finalPoss,
          reductionNode,
          preambleNodes,
          epilogueNodes,
          outputNode,
          preambleOps: seg.preambleOps,
          preambleInputRefs,
          preambleInputDtypes: seg.preambleInputDtypes,
          epilogueOps: seg.epilogueOps,
          epilogueInputRefs,
          outputDtype: seg.outputDtype,
          isMean: seg.isMean,
        };
        return { kind: "reduction" as const, group };
      }
      // Reconstruct FusionGroup
      const groupNodes = seg.finalPoss.map((i) => planNodes[i]);
      const groupNodeIds = new Set(groupNodes.map((n) => n.id));
      const extInputs = collectExternalInputs(groupNodes, groupNodeIds);
      const group: FusionGroup = {
        nodes: groupNodes,
        planIndices: seg.finalPoss,
        externalInputs: extInputs,
        outputNode: planNodes[seg.outputFinalPos],
        additionalOutputNodes:
          seg.additionalOutputFinalPoss.length > 0
            ? seg.additionalOutputFinalPoss.map((i) => planNodes[i])
            : undefined,
        neededIntermediates:
          seg.neededIntermediateFinalPoss.length > 0
            ? seg.neededIntermediateFinalPoss.map((i) => planNodes[i])
            : undefined,
      };
      const recipe = groupToRecipe(group);
      return { kind: "fused" as const, group, recipe };
    });
  } else {
    // ── Cache miss: run full analysis via unified graph compiler ──

    const analysis = analyzeGraph(
      plan.nodes,
      externalNodeIds,
      maxStorageBuffers,
    );
    planNodes = analysis.planNodes;
    segments = analysis.segments;

    // Copy analysis results into local variables
    for (const id of analysis.epilogueClaimedIds) epilogueClaimedIds.add(id);
    for (const id of analysis.prologueClaimedIds) prologueClaimedIds.add(id);
    for (const id of analysis.compoundClaimedIds) compoundClaimedIds.add(id);
    for (const [mmId, chain] of analysis.matmulEpilogueChains)
      matmulEpilogueChains.set(mmId, chain);
    for (const [mmId, prologues] of analysis.matmulPrologues)
      matmulPrologues.set(mmId, prologues);
    compoundMatches = analysis.compoundMatches;
    matmulDirectives = analysis.matmulDirectives;

    // ── Build template and cache it ──
    const origIdToPos = buildIdPositionMap(plan.nodes);
    const finalPerm = planNodes.map((n) => origIdToPos.get(n.id) as number);
    const finalIdToPos = buildIdPositionMap(planNodes);

    const cachedSegments: CachedSegmentDesc[] = segments.map((seg) => {
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
      epilogueClaimedOrigPoss: [...epilogueClaimedIds].map(
        (id) => origIdToPos.get(id) as number,
      ),
      prologueClaimedOrigPoss: [...prologueClaimedIds].map(
        (id) => origIdToPos.get(id) as number,
      ),
      epilogueChains: [...matmulEpilogueChains].map(
        ([mmId, epilogueIds]) =>
          [
            origIdToPos.get(mmId) as number,
            epilogueIds.map((id) => origIdToPos.get(id) as number),
          ] as [number, number[]],
      ),
      prologueDescs: [...matmulPrologues].map(
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
        compoundMatches.length > 0
          ? compoundMatches.map((m) => ({
              name: m.name,
              coveredOrigPoss: m.coveredNodeIds.map(
                (id) => origIdToPos.get(id) as number,
              ),
              outputOrigPos: origIdToPos.get(m.outputNodeId) as number,
              dim: m.dim,
            }))
          : undefined,
    };
    fusionAnalysisCache.set(fingerprint, template);
  }

  // Build a lowered plan during cache miss execution (first run).
  // On cache hit, the lowered plan is already attached to the template.
  let loweredPlanBuilder: LoweredPlanBuilder | null = null;
  const nodeIdToFinalPos = buildIdPositionMap(planNodes);
  if (!cachedTemplate) {
    loweredPlanBuilder = new LoweredPlanBuilder(planNodes.length);
  }

  // Set up lifetime analysis after final plan order is determined.
  // Skip if already reconstructed from cached template (cache hit path above).
  if (enableEarlyRelease && !lifetimes) {
    ({ lifetimes, outputNodeIds } = initLifetimeAnalysis(
      planNodes,
      externalNodeIds,
    ));

    // Store lifetime template in the fusion analysis cache for future hits
    const cached = fusionAnalysisCache.get(fingerprint);
    if (cached && !cached.lifetimeTemplate) {
      cached.lifetimeTemplate = planNodes.map((node) => {
        const lt = (lifetimes as Map<number, TensorLifetime>).get(
          node.id,
        ) as TensorLifetime;
        return {
          firstUse: lt.firstUse,
          lastUse: lt.lastUse,
          isOutput: lt.isOutput,
          isInput: lt.isInput,
          bufferSize: lt.bufferSize,
        };
      });
    }
  }

  // Collect plan analysis for profiling
  let planAnalysisRef: PlanAnalysis | null = null;
  if (isProfilingEnabled()) {
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
      } else {
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

    planAnalysisRef = {
      planIndex: 0, // assigned by recordPlanAnalysis
      totalNodes: plan.nodes.length,
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

  // Track overall step index for early release
  let overallStep = 0;

  // Pre-build consumer count map once for the entire plan.
  // This is used by both reduction preamble detection and matmul epilogue detection.
  // Wrap the entire segment loop in a top-level shared encoder scope.
  // Inner begin/end calls in executeSequentialSegment become nested no-ops
  // thanks to the depth counter. This lets elementwise ops from consecutive
  // segments accumulate into a single batch, reducing queue.submit() calls.
  const useTopLevelSharedEncoder = backend.name === "webgpu";
  if (useTopLevelSharedEncoder) beginSharedEncoder();

  // Activate buffer arena if available from cached template.
  // This is critical for the fallback path: when executeLoweredPlan fails and
  // we fall back to executePlanOptimized, the arena still provides stable buffer
  // identities for bind group cache hits.
  const useArenaFallback =
    useTopLevelSharedEncoder &&
    cachedTemplate?.bufferArena &&
    process.env.TORCHLETTE_USE_ARENA !== "0";
  if (useArenaFallback) {
    setActiveArena(
      (cachedTemplate as FusionAnalysisTemplate).bufferArena as BufferArena,
    );
    setArenaExternalInputBuffers(collectExternalInputBuffers(planNodes));
  }

  try {
    let compoundMatchMap: Map<number, CompoundMatchExec> | undefined;
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

    // Track dispatched nodes for periodic buffer reclamation.
    // When the shared encoder is active, released buffers go to pendingRelease
    // and can't be reused. Periodically flushing moves them back to the main pool.
    const reclaim = createReclaimController(
      DEFAULT_RECLAIM_INTERVAL,
      loweredPlanBuilder,
    );

    // Execute each segment
    for (const segment of segments) {
      if (segment.kind === "reduction") {
        // Execute reduction segment
        const rg = segment.group;
        const reductionLabel =
          rg.preambleNodes.length > 0
            ? `${rg.isMean ? "mean" : rg.reductionNode.op}+${rg.preambleNodes.map((n) => n.op).join("+")}${rg.epilogueOps.length > 0 ? "+" + rg.epilogueOps.map((o) => o.op || o.kind).join("+") : ""}`
            : `${rg.reductionNode.op}+${rg.epilogueOps.map((o) => o.op || o.kind).join("+")}`;

        const { withProfileContext } = await import("./op-dispatch");
        await withProfileContext(reductionLabel, rg.nodes[0].module, () =>
          executeReductionSegment(rg, backend),
        );

        // Record reduction action in lowered plan builder
        if (loweredPlanBuilder) {
          if (rg.preambleNodes.length > 0 && rg.epilogueOps.length > 0) {
            loweredPlanBuilder.recordReductionFusion(
              rg.preambleNodes.map((n) => nodeIdToFinalPos.get(n.id) as number),
              nodeIdToFinalPos.get(rg.reductionNode.id) as number,
              rg.epilogueNodes.map((n) => nodeIdToFinalPos.get(n.id) as number),
              nodeIdToFinalPos.get(rg.outputNode.id) as number,
              rg.preambleOps,
              rg.preambleInputDtypes,
              rg.epilogueOps,
              rg.outputDtype,
              rg.nodes.length,
              rg.isMean,
            );
          } else if (rg.preambleNodes.length > 0) {
            loweredPlanBuilder.recordReductionPreamble(
              nodeIdToFinalPos.get(rg.preambleNodes[0].id) as number,
              nodeIdToFinalPos.get(rg.reductionNode.id) as number,
              rg.preambleNodes.map((n) => nodeIdToFinalPos.get(n.id) as number),
              rg.preambleOps,
              rg.preambleInputDtypes,
              rg.nodes.length,
            );
          } else {
            const covered = rg.nodes.map(
              (n) => nodeIdToFinalPos.get(n.id) as number,
            );
            loweredPlanBuilder.recordReductionEpilogue(
              nodeIdToFinalPos.get(rg.reductionNode.id) as number,
              covered,
              nodeIdToFinalPos.get(rg.outputNode.id) as number,
              rg.epilogueOps,
              rg.outputDtype,
              rg.nodes.length,
            );
          }
        }

        // Track storages and release dead buffers
        if (enableEarlyRelease) {
          for (const node of rg.nodes) {
            if (node.result) nodeToStorage.set(node.id, node.result);
            overallStep++;
            releaseDeadTensors(
              lifetimes,
              overallStep,
              outputNodeIds,
              alreadyReleased,
              nodeToStorage,
            );
          }
        }
        reclaim.advance(rg.nodes.length);
      } else if (segment.kind === "fused" && segment.group.nodes.length >= 2) {
        // Execute fused segment
        await executeFusedSegment(
          segment.group,
          segment.recipe,
          backend,
          enableVectorization,
        );
        stats.fusedNodes += segment.group.nodes.length;
        stats.fusionGroups++;

        // Record fused action in lowered plan builder
        if (loweredPlanBuilder) {
          loweredPlanBuilder.recordFused(
            segment.group.nodes.map(
              (n) => nodeIdToFinalPos.get(n.id) as number,
            ),
            nodeIdToFinalPos.get(segment.group.outputNode.id) as number,
            (segment.group.additionalOutputNodes ?? []).map(
              (n) => nodeIdToFinalPos.get(n.id) as number,
            ),
            (segment.group.neededIntermediates ?? []).map(
              (n) => nodeIdToFinalPos.get(n.id) as number,
            ),
            segment.recipe,
            enableVectorization,
          );
        }

        // Track storages and release dead buffers for fused nodes
        if (enableEarlyRelease) {
          for (const node of segment.group.nodes) {
            if (node.result) {
              nodeToStorage.set(node.id, node.result);
            }
            overallStep++;
            releaseDeadTensors(
              lifetimes,
              overallStep,
              outputNodeIds,
              alreadyReleased,
              nodeToStorage,
            );
          }
        }
        reclaim.advance(segment.group.nodes.length);
      } else {
        // Execute sequentially (too-small fusion groups or sequential segments)
        const seqNodes =
          segment.kind === "fused" ? segment.group.nodes : segment.nodes;
        await executeSequentialSegmentWithEarlyRelease(seqNodes, backend, {
          enableEarlyRelease,
          lifetimes,
          outputNodeIds,
          alreadyReleased,
          nodeToStorage,
          startStep: overallStep,
          prologueSkipIds:
            prologueClaimedIds.size > 0 ? prologueClaimedIds : undefined,
          loweredPlanBuilder,
          nodeIdToFinalPos,
          compoundMatchMap,
          matmulDirectives,
        });
        stats.sequentialNodes += seqNodes.length;
        overallStep += seqNodes.length;
        reclaim.advance(seqNodes.length);
      }

      // Periodic buffer reclamation
      reclaim.maybeFlush(useTopLevelSharedEncoder);
    }
  } finally {
    if (useArenaFallback) {
      clearActiveArena();
      clearArenaExternalInputBuffers();
    }
    if (useTopLevelSharedEncoder) {
      endSharedEncoder();
    }
  }

  // Save the lowered plan to the fusion analysis template (first execution only).
  if (loweredPlanBuilder) {
    const cached = fusionAnalysisCache.get(fingerprint);
    if (cached && !cached.loweredPlan) {
      cached.loweredPlan = loweredPlanBuilder.build();
    }
  }

  // Get the result from the last node
  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Execution failed: no result for last node");
  }

  // Clear results for nodes whose buffers were destroyed by early release.
  if (alreadyReleased.size > 0) {
    for (const node of plan.nodes) {
      if (alreadyReleased.has(node.id)) {
        node.result = undefined;
      }
    }
  }

  return { result: lastNode.result, stats };
}
