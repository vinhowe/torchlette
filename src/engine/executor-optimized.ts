import type { Backend, BackendTensor, DType } from "../backend/types";
import { gpuBuffer, type GPUBuffer } from "../backend/webgpu/gpu-types";
import { getBackend } from "../backend/registry";
import {
  flushBufferPool,
  flushSharedEncoder,
  beginSharedEncoder,
  endSharedEncoder,
  setCurrentOpLabel,
  setActiveArena,
  clearActiveArena,
  setArenaExternalInputBuffers,
  clearArenaExternalInputBuffers,
  type BufferArena,
} from "../backend/webgpu";
import { profileOpBegin, profileOpEnd, isProfilingEnabled, recordPlanAnalysis, type PlanAnalysis, setProfileModule, recordFusionFallback } from "../backend/webgpu/profiler";
import {
  computePlanFingerprint,
  detectFusionGroups,
  groupToRecipe,
  hasFusionOpportunities,
  hasFusionPotential,
  isFusibleOp,
  reorderPlanForFusion,
  segmentPlanForExecution,
  type ExecutionSegment,
  type FusionGroup,
} from "./fusion-detect";
import {
  analyzeLifetimes,
  type TensorLifetime,
} from "./memory-planning";
import {
  type LoweredPlan,
  LoweredPlanBuilder,
} from "./lowered-plan";
import type { LazyIRNode, LazyRef, StorageHandle, ExecutionPlan } from "./lazy-types";
import { storageTracker, releaseDeadTensors } from "./storage-tracker";
import { getInputStorage } from "./op-dispatch";
import { extractPlanMetadata, pretunePlanMatmuls } from "./plan-builder";
import { executePlan } from "./executor-sequential";
import { executeFusedSegment, executeSequentialSegmentWithEarlyRelease, DEFAULT_RECLAIM_INTERVAL } from "./segment-executors";
import type { MatmulPrologueInfo } from "./matmul-epilogue";

/**
 * Options for optimized plan execution.
 */
export interface OptimizedExecutionOptions {
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
  prologueDescs: Array<[number, Array<{
    inputIndex: 0 | 1;
    castOrigPos: number;
    fromDtype: DType;
    toDtype: DType;
  }>]>;

  /** Cached lifetime analysis (position-based). */
  lifetimeTemplate?: Array<{ firstUse: number; lastUse: number; isOutput: boolean; isInput: boolean; bufferSize: number }>;

  /** Cached lowered execution plan (built during first execution on cache miss). */
  loweredPlan?: LoweredPlan;

  /** Per-plan buffer arena: GPUBuffers that persist across steps for bind group cache stability. */
  bufferArena?: BufferArena;
}

export type CachedSegmentDesc =
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
export const fusionAnalysisCache = new Map<number, FusionAnalysisTemplate>();

/** Get a cached fusion analysis template by fingerprint. */
export function getFusionAnalysisTemplate(fingerprint: number): FusionAnalysisTemplate | undefined {
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
  backend: Backend & { device?: { limits?: { maxStorageBuffersPerShaderStage?: number } } },
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

  if (cachedTemplate) {
    // ── Cache hit: reconstruct from template ──
    planNodes = cachedTemplate.finalPerm.map(i => plan.nodes[i]);

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
        epiloguePoss.map(p => plan.nodes[p].id),
      );
    }
    for (const [matmulPos, descs] of cachedTemplate.prologueDescs) {
      matmulPrologues.set(plan.nodes[matmulPos].id, descs.map(d => ({
        inputIndex: d.inputIndex,
        castNodeId: plan.nodes[d.castOrigPos].id,
        originalInputRef: plan.nodes[d.castOrigPos].inputs[0],
        fromDtype: d.fromDtype,
        toDtype: d.toDtype,
      })));
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
    segments = cachedTemplate.segments.map(seg => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          nodes: seg.finalPoss.map(i => planNodes[i]),
        };
      }
      // Reconstruct FusionGroup
      const groupNodes = seg.finalPoss.map(i => planNodes[i]);
      const groupNodeIds = new Set(groupNodes.map(n => n.id));
      const extInputs: LazyRef[] = [];
      for (const node of groupNodes) {
        for (const inp of node.inputs) {
          if (inp.kind === "pending") {
            if (!groupNodeIds.has(inp.node.id) &&
                !extInputs.some(ei => ei.kind === "pending" && ei.node.id === inp.node.id)) {
              extInputs.push(inp);
            }
          } else if (inp.kind === "scalar") {
            // Deduplicate scalar inputs by value+dtype
            if (!extInputs.some(ei => ei.kind === "scalar" && ei.value === inp.value && ei.dtype === inp.dtype)) {
              extInputs.push(inp);
            }
          } else {
            if (!extInputs.some(ei => ei.kind === "materialized" && ei.storage.id === inp.storage.id)) {
              extInputs.push(inp);
            }
          }
        }
      }
      const group: FusionGroup = {
        nodes: groupNodes,
        planIndices: seg.finalPoss,
        externalInputs: extInputs,
        outputNode: planNodes[seg.outputFinalPos],
        additionalOutputNodes: seg.additionalOutputFinalPoss.length > 0
          ? seg.additionalOutputFinalPoss.map(i => planNodes[i]) : undefined,
        neededIntermediates: seg.neededIntermediateFinalPoss.length > 0
          ? seg.neededIntermediateFinalPoss.map(i => planNodes[i]) : undefined,
      };
      const recipe = groupToRecipe(group);
      return { kind: "fused" as const, group, recipe };
    });
  } else {
    // ── Cache miss: run full analysis ──

    // Reorder plan to cluster fusible chains together
    planNodes = plan.nodes;
    if (planNodes.length > 2) {
      planNodes = reorderPlanForFusion(planNodes);
    }

    // Pre-scan: detect matmul epilogue chains AND input-side cast prologues
    if (backend.name === "webgpu") {
      // Build consumer map: nodeId → list of consumer nodes
      const consumers = new Map<number, LazyIRNode[]>();
      const consumerCount = new Map<number, number>();
      for (const node of planNodes) {
        for (const input of node.inputs) {
          if (input.kind === "pending") {
            const producerId = input.node.id;
            consumerCount.set(producerId, (consumerCount.get(producerId) ?? 0) + 1);
            if (!consumers.has(producerId)) consumers.set(producerId, []);
            consumers.get(producerId)!.push(node);
          }
        }
      }

      const nodePosition = new Map<number, number>();
      for (let i = 0; i < planNodes.length; i++) {
        nodePosition.set(planNodes[i].id, i);
      }

      const nodeById = new Map<number, LazyIRNode>();
      for (const n of planNodes) nodeById.set(n.id, n);


      for (let mi = 0; mi < planNodes.length; mi++) {
        const node = planNodes[mi];
        if (node.op !== "matmul") continue;

        const matmulPos = mi;
        const chainIds: number[] = [];
        let current = node;
        let additionalInputCount = 0;

        for (let depth = 0; depth < 4; depth++) {
          const cc = consumerCount.get(current.id) ?? 0;
          if (cc !== 1) break;
          // Skip external check for nodes already absorbed into the chain
          // (e.g., reshape nodes with single-consumer). Their live pending
          // tensors will be disposed by the caller after plan execution.
          if (externalNodeIds?.has(current.id) && !chainIds.includes(current.id)) {
            break;
          }

          const nexts = consumers.get(current.id);
          if (!nexts || nexts.length !== 1) break;
          const next = nexts[0];

          if (next.inputs.length === 0) break;
          let chainInputIdx = 0;
          const primary = next.inputs[0];
          if (primary.kind !== "pending" || primary.node.id !== current.id) {
            // For commutative binary ops, check if the chain continues via inputs[1]
            if ((next.op === "add" || next.op === "mul") && next.inputs.length === 2) {
              const alt = next.inputs[1];
              if (alt.kind === "pending" && alt.node.id === current.id) {
                chainInputIdx = 1;
              } else {
                break;
              }
            } else {
              break;
            }
          }

          // Skip reshape that only removes leading size-1 dims (e.g. [1,M,N]→[M,N])
          if (next.op === "reshape") {
            const curShape = current.shape;
            const nextShape = next.shape;
            if (curShape.length === nextShape.length + 1
                && curShape[0] === 1
                && curShape.slice(1).every((d: number, i: number) => d === nextShape[i])) {
              chainIds.push(next.id);
              current = next;
              continue; // don't increment depth — reshape is free
            }
            break;
          }

          let ok = false;
          if (next.op === "cast") ok = true;
          else if ((next.op === "add" || next.op === "mul") && next.inputs.length === 2) {
            if (additionalInputCount >= 4) break;
            const secondary = next.inputs[chainInputIdx === 0 ? 1 : 0];
            if (secondary.kind === "materialized") {
              ok = true;
            } else if (secondary.kind === "pending") {
              const secPos = nodePosition.get(secondary.node.id);
              if (secPos !== undefined && secPos < matmulPos) {
                ok = true;
              }
            }
            if (ok) additionalInputCount++;
          } else if (next.op === "relu" || next.op === "silu" || next.op === "sigmoid" || next.op === "tanh" || next.op === "gelu") {
            ok = true;
          }

          if (!ok) break;

          chainIds.push(next.id);
          current = next;
        }

        if (chainIds.length > 0) {
          matmulEpilogueChains.set(node.id, chainIds);
          for (const id of chainIds) epilogueClaimedIds.add(id);
        }

        // Detect input-side cast prologues (inference only)
        if (!externalNodeIds || externalNodeIds.size === 0) {
          const prologuesForNode: MatmulPrologueInfo[] = [];
          for (const idx of [0, 1] as const) {
            const inputRef = node.inputs[idx];
            if (inputRef.kind !== "pending") continue;
            const castNode = inputRef.node;
            if (castNode.op !== "cast") continue;
            if ((consumerCount.get(castNode.id) ?? 0) !== 1) continue;
            const castPayload = castNode.payload as { dtype: DType } | undefined;
            if (!castPayload) continue;
            const toDtype = castPayload.dtype;
            const castInput = castNode.inputs[0];
            if (!castInput) continue;
            let fromDtype: DType;
            if (castInput.kind === "pending") {
              fromDtype = castInput.node.dtype;
            } else if (castInput.kind === "materialized") {
              fromDtype = castInput.storage.backendTensor.dtype ?? "f32";
            } else {
              continue;
            }
            if (fromDtype !== "f32" || toDtype !== "f16") continue;

            prologuesForNode.push({
              inputIndex: idx,
              castNodeId: castNode.id,
              originalInputRef: castInput,
              fromDtype,
              toDtype,
            });
            prologueClaimedIds.add(castNode.id);
          }
          if (prologuesForNode.length > 0) {
            matmulPrologues.set(node.id, prologuesForNode);
          }
        }
      }

      // Relocate epilogue chain nodes after their matmul
      if (epilogueClaimedIds.size > 0) {
        const claimedSet = epilogueClaimedIds;
        const unclaimed = planNodes.filter(n => !claimedSet.has(n.id));
        const relocated: LazyIRNode[] = [];
        for (const n of unclaimed) {
          relocated.push(n);
          const chain = matmulEpilogueChains.get(n.id);
          if (chain) {
            for (const id of chain) {
              relocated.push(nodeById.get(id)!);
            }
          }
        }
        planNodes = relocated;
      }
    }

    // Segment the reordered plan into fusible and sequential parts
    let allClaimedIds: Set<number> | undefined;
    if (epilogueClaimedIds.size > 0 || prologueClaimedIds.size > 0) {
      allClaimedIds = new Set([...epilogueClaimedIds, ...prologueClaimedIds]);
    }
    segments = segmentPlanForExecution(planNodes, externalNodeIds, {
      maxStorageBuffers,
      enableMultiOutput: true,
      epilogueClaimedIds: allClaimedIds,
    });

    // ── Build template and cache it ──
    const origIdToPos = new Map<number, number>();
    for (let i = 0; i < plan.nodes.length; i++) {
      origIdToPos.set(plan.nodes[i].id, i);
    }
    const finalPerm = planNodes.map(n => origIdToPos.get(n.id)!);

    const finalIdToPos = new Map<number, number>();
    for (let i = 0; i < planNodes.length; i++) {
      finalIdToPos.set(planNodes[i].id, i);
    }

    const cachedSegments: CachedSegmentDesc[] = segments.map(seg => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          finalPoss: seg.nodes.map(n => finalIdToPos.get(n.id)!),
        };
      }
      return {
        kind: "fused" as const,
        finalPoss: seg.group.nodes.map(n => finalIdToPos.get(n.id)!),
        outputFinalPos: finalIdToPos.get(seg.group.outputNode.id)!,
        additionalOutputFinalPoss: (seg.group.additionalOutputNodes ?? [])
          .map(n => finalIdToPos.get(n.id)!),
        neededIntermediateFinalPoss: (seg.group.neededIntermediates ?? [])
          .map(n => finalIdToPos.get(n.id)!),
      };
    });

    const template: FusionAnalysisTemplate = {
      finalPerm,
      segments: cachedSegments,
      epilogueClaimedOrigPoss: [...epilogueClaimedIds].map(id => origIdToPos.get(id)!),
      prologueClaimedOrigPoss: [...prologueClaimedIds].map(id => origIdToPos.get(id)!),
      epilogueChains: [...matmulEpilogueChains].map(([mmId, epilogueIds]) => [
        origIdToPos.get(mmId)!,
        epilogueIds.map(id => origIdToPos.get(id)!),
      ] as [number, number[]]),
      prologueDescs: [...matmulPrologues].map(([mmId, prologues]) => [
        origIdToPos.get(mmId)!,
        prologues.map(p => ({
          inputIndex: p.inputIndex,
          castOrigPos: origIdToPos.get(p.castNodeId)!,
          fromDtype: p.fromDtype,
          toDtype: p.toDtype,
        })),
      ] as [number, Array<{ inputIndex: 0 | 1; castOrigPos: number; fromDtype: DType; toDtype: DType }>]),
    };
    fusionAnalysisCache.set(fingerprint, template);
  }

  // Build a lowered plan during cache miss execution (first run).
  // On cache hit, the lowered plan is already attached to the template.
  let loweredPlanBuilder: LoweredPlanBuilder | null = null;
  const nodeIdToFinalPos = new Map<number, number>();
  for (let i = 0; i < planNodes.length; i++) {
    nodeIdToFinalPos.set(planNodes[i].id, i);
  }
  if (!cachedTemplate) {
    loweredPlanBuilder = new LoweredPlanBuilder(planNodes.length);
  }

  // Set up lifetime analysis after final plan order is determined.
  // Skip if already reconstructed from cached template (cache hit path above).
  if (enableEarlyRelease && !lifetimes) {
    const reorderedPlan = { nodes: planNodes };
    const { nodeOrder, nodeInputs, nodeSizes } = extractPlanMetadata(reorderedPlan);
    const lastNodeId = plan.nodes[plan.nodes.length - 1].id;
    outputNodeIds = new Set([lastNodeId]);
    if (externalNodeIds) {
      for (const id of externalNodeIds) outputNodeIds.add(id);
    }
    lifetimes = analyzeLifetimes(nodeOrder, nodeInputs, outputNodeIds, nodeSizes);

    // Store lifetime template in the fusion analysis cache for future hits
    const cached = fusionAnalysisCache.get(fingerprint);
    if (cached && !cached.lifetimeTemplate) {
      cached.lifetimeTemplate = planNodes.map((node) => {
        const lt = lifetimes!.get(node.id)!;
        return { firstUse: lt.firstUse, lastUse: lt.lastUse, isOutput: lt.isOutput, isInput: lt.isInput, bufferSize: lt.bufferSize };
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
    const unfusedByShape: Record<string, { count: number; ops: Record<string, number> }> = {};

    for (const segment of segments) {
      if (segment.kind === "fused" && segment.group.nodes.length >= minFusionSize) {
        fusedSegCount++;
        fusedNodeCount += segment.group.nodes.length;
        fusionGroupCount++;
      } else if (segment.kind === "fused") {
        seqSegCount++;
        for (const node of segment.group.nodes) {
          sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
          if (isFusibleOp(node.op)) {
            const shapeKey = node.shape.join(",");
            let bucket = unfusedByShape[shapeKey];
            if (!bucket) { bucket = { count: 0, ops: {} }; unfusedByShape[shapeKey] = bucket; }
            bucket.count++;
            bucket.ops[node.op] = (bucket.ops[node.op] ?? 0) + 1;
          }
        }
      } else {
        seqSegCount++;
        for (const node of segment.nodes) {
          sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
          if (isFusibleOp(node.op) && !epilogueClaimedIds.has(node.id) && !prologueClaimedIds.has(node.id)) {
            const shapeKey = node.shape.join(",");
            let bucket = unfusedByShape[shapeKey];
            if (!bucket) { bucket = { count: 0, ops: {} }; unfusedByShape[shapeKey] = bucket; }
            bucket.count++;
            bucket.ops[node.op] = (bucket.ops[node.op] ?? 0) + 1;
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
        if (isFusibleOp(cur.op) && cur.op !== "cast" && (next.op === "sum" || next.op === "mean")) {
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
  // Building it once here instead of per-segment saves ~90 redundant rebuilds.
  let planConsumerCount: Map<number, number> | undefined;
  if (backend.name === "webgpu") {
    planConsumerCount = new Map<number, number>();
    for (const n of planNodes) {
      for (const ref of n.inputs) {
        if (ref.kind === "pending") {
          planConsumerCount.set(ref.node.id, (planConsumerCount.get(ref.node.id) ?? 0) + 1);
        }
      }
    }
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
  const useArenaFallback = useTopLevelSharedEncoder && cachedTemplate?.bufferArena
    && process.env.TORCHLETTE_USE_ARENA !== "0";
  if (useArenaFallback) {
    setActiveArena(cachedTemplate!.bufferArena!);
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
          segment.group.nodes.map(n => nodeIdToFinalPos.get(n.id)!),
          nodeIdToFinalPos.get(segment.group.outputNode.id)!,
          (segment.group.additionalOutputNodes ?? []).map(n => nodeIdToFinalPos.get(n.id)!),
          (segment.group.neededIntermediates ?? []).map(n => nodeIdToFinalPos.get(n.id)!),
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
          releaseDeadTensors(lifetimes, overallStep, outputNodeIds, alreadyReleased, nodeToStorage);
        }
      }
      nodesSinceReclaim += segment.group.nodes.length;
    } else if (segment.kind === "fused") {
      // Too small for fusion - execute sequentially
      await executeSequentialSegmentWithEarlyRelease(
        segment.group.nodes,
        backend,
        enableEarlyRelease,
        lifetimes,
        outputNodeIds,
        alreadyReleased,
        nodeToStorage,
        overallStep,
        externalNodeIds,
        planNodes,
        matmulPrologues.size > 0 ? matmulPrologues : undefined,
        prologueClaimedIds.size > 0 ? prologueClaimedIds : undefined,
        planConsumerCount,
        loweredPlanBuilder,
        nodeIdToFinalPos,
      );
      stats.sequentialNodes += segment.group.nodes.length;
      overallStep += segment.group.nodes.length;
      nodesSinceReclaim += segment.group.nodes.length;
    } else {
      // Execute sequential segment
      await executeSequentialSegmentWithEarlyRelease(
        segment.nodes,
        backend,
        enableEarlyRelease,
        lifetimes,
        outputNodeIds,
        alreadyReleased,
        nodeToStorage,
        overallStep,
        externalNodeIds,
        planNodes,
        matmulPrologues.size > 0 ? matmulPrologues : undefined,
        prologueClaimedIds.size > 0 ? prologueClaimedIds : undefined,
        planConsumerCount,
        loweredPlanBuilder,
        nodeIdToFinalPos,
      );
      stats.sequentialNodes += segment.nodes.length;
      overallStep += segment.nodes.length;
      nodesSinceReclaim += segment.nodes.length;
    }

    // Periodic buffer reclamation: flush the shared encoder and buffer pool
    // so that dead buffers in pendingRelease become available for reuse.
    if (useTopLevelSharedEncoder && reclaimInterval > 0 && nodesSinceReclaim >= reclaimInterval) {
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
