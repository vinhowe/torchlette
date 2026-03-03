import type { Backend, DType } from "../backend/types";
import {
  type BufferArena,
  beginSharedEncoder,
  clearActiveArena,
  clearArenaExternalInputBuffers,
  endSharedEncoder,
  flushBufferPool,
  flushSharedEncoder,
  setActiveArena,
  setArenaExternalInputBuffers,
} from "../backend/webgpu";
import { type GPUBuffer, gpuBuffer } from "../backend/webgpu/gpu-types";
import {
  isProfilingEnabled,
  type PlanAnalysis,
  recordPlanAnalysis,
} from "../backend/webgpu/profiler";
import type { CompoundMatch } from "./compound-patterns";
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
import { analyzeLifetimes, type TensorLifetime } from "./lifetime-analysis";
import { type LoweredPlan, LoweredPlanBuilder } from "./lowered-plan";
import type { MatmulPrologueInfo } from "./matmul-epilogue";
import { extractPlanMetadata, pretunePlanMatmuls } from "./plan-builder";
import {
  buildConsumerCount,
  type CompoundMatchExec,
  DEFAULT_RECLAIM_INTERVAL,
  executeFusedSegment,
  executeSequentialSegmentWithEarlyRelease,
} from "./segment-executors";
import { releaseDeadTensors } from "./storage-tracker";

/**
 * Options for optimized plan execution.
 */
interface OptimizedExecutionOptions {
  /** Enable elementwise fusion (default: true for WebGPU) */
  enableFusion?: boolean;
  /** Enable vectorization for fused kernels (default: true) */
  enableVectorization?: boolean;
  /** Minimum ops required to trigger fusion (default: 2) */
  minFusionSize?: number;
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
  /** Flush buffer pool every N nodes to reclaim dead buffers mid-plan (default: 50, 0=disabled) */
  reclaimInterval?: number;
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

  /** Original plan positions of graph-rewrite-bypassed nodes (identity casts, etc.). */
  rewriteBypassedOrigPoss?: number[];

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
    minFusionSize = 2,
    enableEarlyRelease = false,
    reclaimInterval = DEFAULT_RECLAIM_INTERVAL,
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
  let analysisConsumerCount: Map<number, number> | undefined;

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

    // Reconstruct lifetime analysis from cached template (avoids extractPlanMetadata + analyzeLifetimes)
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
    analysisConsumerCount = analysis.consumerCount;

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
      rewriteBypassedOrigPoss:
        analysis.rewriteBypassedIds.size > 0
          ? [...analysis.rewriteBypassedIds]
              .map((id) => origIdToPos.get(id) as number)
              .filter((p) => p !== undefined)
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
    const reorderedPlan = { nodes: planNodes };
    const { nodeOrder, nodeInputs, nodeSizes } =
      extractPlanMetadata(reorderedPlan);
    const lastNodeId = plan.nodes[plan.nodes.length - 1].id;
    outputNodeIds = new Set([lastNodeId]);
    if (externalNodeIds) {
      for (const id of externalNodeIds) outputNodeIds.add(id);
    }
    lifetimes = analyzeLifetimes(
      nodeOrder,
      nodeInputs,
      outputNodeIds,
      nodeSizes,
    );

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
      if (
        segment.kind === "fused" &&
        segment.group.nodes.length >= minFusionSize
      ) {
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
  // On cache miss, reuse the consumer count from the graph analysis.
  // On cache hit, build it from the reconstructed plan nodes.
  let planConsumerCount: Map<number, number> | undefined;
  if (backend.name === "webgpu") {
    planConsumerCount =
      !cachedTemplate && analysisConsumerCount
        ? analysisConsumerCount
        : buildConsumerCount(planNodes);
  }

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
    // Register external input buffers for arena conflict detection
    const extBufs: GPUBuffer[] = [];
    for (const node of planNodes) {
      for (const ref of node.inputs) {
        if (ref.kind === "materialized") {
          const buf = gpuBuffer(ref.storage.backendTensor);
          if (buf) extBufs.push(buf);
        } else if (ref.kind === "pending" && ref.node.result) {
          const buf = gpuBuffer(ref.node.result.backendTensor);
          if (buf) extBufs.push(buf);
        }
      }
    }
    setArenaExternalInputBuffers(extBufs);
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
    let nodesSinceReclaim = 0;

    // Execute each segment
    for (const segment of segments) {
      if (
        segment.kind === "fused" &&
        segment.group.nodes.length >= minFusionSize
      ) {
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
        nodesSinceReclaim += segment.group.nodes.length;
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
          externalNodeIds,
          allPlanNodes: planNodes,
          matmulPrologueMap:
            matmulPrologues.size > 0 ? matmulPrologues : undefined,
          prologueSkipIds:
            prologueClaimedIds.size > 0 ? prologueClaimedIds : undefined,
          prebuiltConsumerCount: planConsumerCount,
          loweredPlanBuilder,
          nodeIdToFinalPos,
          compoundMatchMap,
        });
        stats.sequentialNodes += seqNodes.length;
        overallStep += seqNodes.length;
        nodesSinceReclaim += seqNodes.length;
      }

      // Periodic buffer reclamation: flush the shared encoder and buffer pool
      // so that dead buffers in pendingRelease become available for reuse.
      if (
        useTopLevelSharedEncoder &&
        reclaimInterval > 0 &&
        nodesSinceReclaim >= reclaimInterval
      ) {
        flushSharedEncoder();
        flushBufferPool();
        if (loweredPlanBuilder) loweredPlanBuilder.recordReclaim();
        nodesSinceReclaim = 0;
      }
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
