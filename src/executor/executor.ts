import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DType,
} from "../backend/types";
import { isFusedBackend } from "../backend/types";
import {
  type BufferArena,
  arenaLivenessEnabled,
  beginSharedEncoder,
  clearActiveArena,
  clearArenaConflictDetected,
  clearArenaExternalInputBuffers,
  compiledPlannedEnabled,
  destroyArena,
  endSharedEncoder,
  flushBufferPool,
  flushSharedEncoder,
  getArenaConflictDetected,
  setActiveArena,
  setArenaExternalInputBuffers,
  setCurrentOpLabel,
} from "../backend/webgpu";
import { getDispatchSequenceCounters } from "../backend/webgpu/bind-group-cache";
import { bufferPool } from "../backend/webgpu/buffer-pool";
import {
  asGPUTensor,
  type GPUBuffer,
  gpuBuffer,
} from "../backend/webgpu/gpu-types";
import type { EpilogueConfig } from "../backend/webgpu/matmul/types";
import type { FusionGroup } from "../compiler/fusion-detect";
import {
  buildIdPositionMap,
  computePlanFingerprint,
  type ExecutionSegment,
  groupToRecipe,
  isFusibleOp,
  isInlinableScalar,
} from "../compiler/fusion-detect";
import { analyzeGraph } from "../compiler/graph-compiler";
import { runPasses, SIMPLIFICATION_PASSES } from "../compiler/graph-rewrites";
import type { MatmulPrologueInfo } from "../compiler/matmul-epilogue";
import {
  _detectTransposeView,
  executeMatmulWithEpilogue,
} from "../compiler/matmul-epilogue";
import { contiguousStrides, shapesEqual } from "../core/shape";
import {
  _webgpuMatmulGeomImports,
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
  wrapResultAsStorage,
} from "../graph/node-factory";
import {
  getPlanAnalysisGeneration,
  isProfilingEnabled,
  type PlanAnalysis,
  profileOpBegin,
  profileOpEnd,
  recordPlanAnalysis,
  setProfileModule,
} from "../graph/profiler";
import {
  canSafelyRelease,
  releaseBufferImmediate,
} from "../graph/storage-tracker";
import type {
  ExecutionPlan,
  LazyIRNode,
  LazyRef,
  StorageHandle,
} from "../graph/types";
/** Mark a storage's GPU buffer as liveness-safe for immediate pool reuse. */
function markLivenessSafe(storage: StorageHandle): void {
  const buf = (storage.backendTensor as { buffer?: GPUBuffer }).buffer;
  if (buf) bufferPool.markLivenessSafe(buf);
}

import {
  getLivePendingNodeIds,
  getLivePendingRootNodes,
} from "../runtime/tensor";
import {
  assignSlot,
  buildCompiledPlan,
  destroyCompiledPlanBuffers,
  executeCompiledPlan,
  setRecordingNodeIndex,
} from "./compiled-plan";
import {
  isCompilationRecordingActive,
  type NodeResult,
  recordBarrier,
  recordWrite,
  startCompilationRecording,
  stopCompilationRecording,
} from "./compiled-plan";
import {
  clearActiveScalarTable,
  destroyScalarTable,
  refreshScalarTable,
} from "./scalar-table";
import {
  buildLoweredPlanFromAnalysis,
  getActionNodeIndices,
  type LoweredPlan,
} from "./lowered-plan";
import {
  assignNodeResult,
  executeOpSync,
  getInputStorage,
  withProfileContext,
} from "./op-dispatch";
import { pretunePlanMatmuls } from "./plan-builder";
import {
  ensureFusionImports,
  executeFusedSegment,
  executeRowProgram,
} from "./segment-executors";
import { executePlanSequential } from "./sequential";

// ============================================================================
// Optimized Execution Types
// ============================================================================

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

  /** Cached plan analysis for profiling (stored on first miss, replayed on hits). */
  planAnalysis?: PlanAnalysis;
  /** Generation at which planAnalysis was last replayed (prevents per-step accumulation). */
  replayedGeneration?: number;
  /** Secondary fingerprint hash — used for collision detection on cache hit. */
  fingerprintSecondary: number;
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
 * Destroy all buffer arenas across cached templates, freeing GPU memory.
 * The arenas will be rebuilt on the next execution (one slow step, then
 * back to compiled plan speed). Call between training rounds to prevent
 * unbounded GPU memory growth in long-running training.
 *
 * @param force - pass true from cross-session recycle paths to destroy
 *   buffers even if they appear "live" (the liveness state is stale
 *   between sessions and refusing to destroy leaks ~20 MiB per session).
 */
export function evictAllArenas(force = false): void {
  // Import is at top of file — destroyArena is already available
  for (const [, template] of fusionAnalysisCache) {
    if (template.bufferArena) {
      destroyArena(template.bufferArena, force);
      template.bufferArena = undefined;
    }
    // Also invalidate compiled plan (it depends on arena buffer identities)
    if (template.loweredPlan?.compiledPlan) {
      destroyCompiledPlanBuffers(template.loweredPlan.compiledPlan);
      template.loweredPlan.compiledPlan = undefined;
    }
    if (template.loweredPlan) destroyScalarTable(template.loweredPlan);
  }
}

/**
 * Invalidate compiled plans without destroying arena buffers.
 * Use after param buffers change (e.g., outer optimizer step) to force
 * recompilation on the next step. Lighter than evictAllArenas() — avoids
 * the transient OOM from arena rebuild.
 */
export function invalidateCompiledPlans(): void {
  for (const [, template] of fusionAnalysisCache) {
    if (template.loweredPlan?.compiledPlan) {
      destroyCompiledPlanBuffers(template.loweredPlan.compiledPlan);
      template.loweredPlan.compiledPlan = undefined;
    }
    if (template.loweredPlan) destroyScalarTable(template.loweredPlan);
    // Clear arena too — its bind groups reference stale param buffers
    if (template.bufferArena) {
      destroyArena(template.bufferArena);
      template.bufferArena = undefined;
    }
  }
}

/** Collect GPU buffers from all external (materialized/already-resolved) plan inputs. */
/**
 * Check whether any INLINED scalar constant in a fused-action recipe differs
 * from the current step's value. Cheap scan: only fused actions with a built
 * external-input pattern (always true once a compiled plan exists) and only
 * their inlined-constant inputs. Used to invalidate a compiled plan whose
 * recorded kernels baked a value that has since changed.
 */
function inlinedFusionScalarsStale(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
): boolean {
  for (const action of loweredPlan.actions) {
    if (action.kind !== "fused") continue;
    const pattern = action.cachedExternalInputPattern;
    if (!pattern) continue;
    const rin = action.recipe.inputs;
    for (let i = 0; i < rin.length && i < pattern.length; i++) {
      const inp = rin[i];
      if (!inp?.isInlinedConstant) continue;
      const p = pattern[i];
      const node = planNodes[action.coveredNodeIndices[p.nodeLocalIdx]];
      const ref = node?.inputs[p.inputIdx];
      if (!ref) continue;
      if (ref.kind === "scalar") {
        if (ref.value !== inp.inlinedValue) return true;
      } else {
        const cur = isInlinableScalar(ref);
        if (cur.inlinable && cur.value !== inp.inlinedValue) return true;
      }
    }
  }
  return false;
}

export function collectExternalInputBuffers(
  planNodes: LazyIRNode[],
): GPUBuffer[] {
  const bufs: GPUBuffer[] = [];
  for (const node of planNodes) {
    for (const ref of node.inputs) {
      if (ref.kind === "materialized") {
        const buf = gpuBuffer(ref.storage.backendTensor);
        if (buf) bufs.push(buf);
      } else if (ref.kind === "pending" && ref.node.result) {
        const buf = gpuBuffer(ref.node.result.backendTensor);
        if (buf) bufs.push(buf);
      }
    }
  }
  return bufs;
}

// ============================================================================
// Matmul-epilogue slow-path helpers (extracted for readability)
// ============================================================================

type PlanNodePath = { planNodeIndex: number; inputIndex: number };

/**
 * Walk the matmul-epilogue covered node chain to find external epilogue inputs.
 * For each add/mul in the chain, identifies which input comes from the chain
 * vs. which is an external input (e.g., bias tensor).
 */
function reconstructEpilogueInputs(
  planNodes: LazyIRNode[],
  coveredNodeIndices: number[],
): { epilogueInputRefs: LazyRef[]; epilogueInputPaths: PlanNodePath[] } {
  const epilogueInputRefs: LazyRef[] = [];
  const epilogueInputPaths: PlanNodePath[] = [];
  for (let ci = 1; ci < coveredNodeIndices.length; ci++) {
    const chainNode = planNodes[coveredNodeIndices[ci]];
    if (
      (chainNode.op === "add" || chainNode.op === "mul") &&
      chainNode.inputs.length === 2
    ) {
      const prevChainNodeId = planNodes[coveredNodeIndices[ci - 1]].id;
      const inp0IsChain =
        chainNode.inputs[0].kind === "pending" &&
        chainNode.inputs[0].node.id === prevChainNodeId;
      const externalIdx = inp0IsChain ? 1 : 0;
      epilogueInputRefs.push(chainNode.inputs[externalIdx]);
      epilogueInputPaths.push({
        planNodeIndex: coveredNodeIndices[ci],
        inputIndex: externalIdx,
      });
    }
  }
  return { epilogueInputRefs, epilogueInputPaths };
}

/**
 * Resolve prologue cast decisions: for each prologue, check whether the cast
 * already ran (fused in an earlier group). If not, redirect the matmul input
 * to the pre-cast tensor and tell codegen to cast inline.
 */
function resolvePrologueInputs(
  planNodes: LazyIRNode[],
  matmulNode: LazyIRNode,
  matmulNodeIndex: number,
  actionPrologues: Array<{
    inputIndex: 0 | 1;
    castNodeIndex: number;
    fromDtype: DType;
    toDtype: DType;
  }>,
): {
  prologues: MatmulPrologueInfo[];
  inputCastA: "f16" | "f32" | undefined;
  inputCastB: "f16" | "f32" | undefined;
  resolvedInputRefA: LazyRef;
  resolvedInputRefB: LazyRef;
  inputAPath: PlanNodePath;
  inputBPath: PlanNodePath;
} {
  let inputCastA: "f16" | "f32" | undefined;
  let inputCastB: "f16" | "f32" | undefined;
  let resolvedInputRefA = matmulNode.inputs[0];
  let resolvedInputRefB = matmulNode.inputs[1];
  let inputAPath: PlanNodePath = {
    planNodeIndex: matmulNodeIndex,
    inputIndex: 0,
  };
  let inputBPath: PlanNodePath = {
    planNodeIndex: matmulNodeIndex,
    inputIndex: 1,
  };

  const prologues = actionPrologues.map((p) => ({
    inputIndex: p.inputIndex,
    castNodeId: planNodes[p.castNodeIndex].id,
    originalInputRef: planNodes[p.castNodeIndex].inputs[0],
    fromDtype: p.fromDtype,
    toDtype: p.toDtype,
  }));

  for (const p of actionPrologues) {
    const castRef =
      p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
    const castAlreadyRan =
      castRef.kind === "pending" && castRef.node.result != null;
    if (!castAlreadyRan) {
      if (p.inputIndex === 0) {
        resolvedInputRefA = planNodes[p.castNodeIndex].inputs[0];
        inputCastA = p.toDtype as "f16" | "f32";
        inputAPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
      } else {
        resolvedInputRefB = planNodes[p.castNodeIndex].inputs[0];
        inputCastB = p.toDtype as "f16" | "f32";
        inputBPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
      }
    }
  }

  return {
    prologues,
    inputCastA,
    inputCastB,
    resolvedInputRefA,
    resolvedInputRefB,
    inputAPath,
    inputBPath,
  };
}

/**
 * Compute matmul geometry from resolved inputs and store the dispatch config
 * on the action for fast-path replay on subsequent steps.
 */
async function buildAndCacheDispatchConfig(
  action: {
    cachedDispatchConfig?: unknown;
    epilogueOps: unknown[];
    outputDtype?: DType;
  },
  resolvedInputRefA: LazyRef,
  resolvedInputRefB: LazyRef,
  inputCastA: "f16" | "f32" | undefined,
  inputCastB: "f16" | "f32" | undefined,
  inputAPath: PlanNodePath,
  inputBPath: PlanNodePath,
  epilogueInputRefs: LazyRef[],
  epilogueInputPaths: PlanNodePath[],
): Promise<void> {
  const { computeMatmulOutputShape, computeBatchSize, computeBatchStrides } =
    _webgpuMatmulGeomImports as NonNullable<typeof _webgpuMatmulGeomImports>;
  const { isF16Supported } = await import("../backend/webgpu/index");

  const tensorA = asGPUTensor(getInputStorage(resolvedInputRefA).backendTensor);
  const tensorB = asGPUTensor(getInputStorage(resolvedInputRefB).backendTensor);

  const detA = _detectTransposeView(tensorA);
  const detB = _detectTransposeView(tensorB);
  const transA = detA.transposed;
  const transB = detB.transposed;

  const outShape = computeMatmulOutputShape(
    detA.shape,
    detB.shape,
    transA,
    transB,
  );
  const aRank = detA.shape.length;
  const bRank = detB.shape.length;
  const m = transA ? detA.shape[aRank - 1] : detA.shape[aRank - 2];
  const k = transA ? detA.shape[aRank - 2] : detA.shape[aRank - 1];
  const n = transB ? detB.shape[bRank - 2] : detB.shape[bRank - 1];

  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    detA.shape,
    detB.shape,
    batchDims,
    m,
    n,
    k,
  );

  const f16ok = isF16Supported();
  const rawDtypeA =
    tensorA.dtype === "f16" && f16ok ? ("f16" as const) : ("f32" as const);
  const rawDtypeB =
    tensorB.dtype === "f16" && f16ok ? ("f16" as const) : ("f32" as const);
  const dtypeA: "f16" | "f32" =
    inputCastA === "f16" && f16ok ? "f16" : rawDtypeA;
  const dtypeB: "f16" | "f32" =
    inputCastB === "f16" && f16ok ? "f16" : rawDtypeB;
  const promotedDtype =
    dtypeA === "f32" || dtypeB === "f32" ? ("f32" as const) : dtypeA;
  const outputDtype = action.outputDtype ?? promotedDtype;

  action.cachedDispatchConfig = {
    inputAPath,
    inputBPath,
    epilogueInputPaths,
    inputCastA,
    inputCastB,
    m,
    k,
    n,
    transA,
    transB,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    outShape,
    dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
    outputDtype,
    epilogueConfig: {
      ops: action.epilogueOps,
      additionalInputCount: epilogueInputRefs.length,
      outputDtype: action.outputDtype,
    },
  };
}

/**
 * Execute a lowered plan (cached dispatch sequence).
 *
 * Replaces the full executePlanOptimized() path on cache hits when the
 * lowered plan is available. Skips fusion detection, plan reordering,
 * segmentation, matmul epilogue detection, reduction preamble detection —
 * all of those decisions were recorded during the first execution and are
 * replayed here.
 *
 * @param plan - The original execution plan
 * @param planNodes - Reordered plan nodes (from template.finalPerm)
 * @param loweredPlan - The cached lowered plan from the template
 * @param backend - The backend to use
 * @param options - Execution options
 * @returns The result and execution stats
 */
export async function executeLoweredPlan(
  plan: ExecutionPlan,
  planNodes: LazyIRNode[],
  loweredPlan: LoweredPlan,
  backend: Backend,
  options: {
    bufferArena?: BufferArena;
  } = {},
): Promise<OptimizedExecutionResult> {
  // Validate plan node count matches
  if (planNodes.length !== loweredPlan.planNodeCount) {
    throw new Error(
      `Lowered plan node count mismatch: plan has ${planNodes.length}, lowered expects ${loweredPlan.planNodeCount}`,
    );
  }

  // Pre-tune matmul shapes (same as normal path)
  await pretunePlanMatmuls({ nodes: planNodes }, backend);

  // Ensure matmul imports are loaded (cached — only first call does actual import)
  if (backend.name === "webgpu") {
    await ensureWebGPUMatmulImports();
  }

  // Track stats for consistency with executePlanOptimized
  const stats: OptimizedExecutionStats = {
    totalNodes: planNodes.length,
    fusedNodes: 0,
    sequentialNodes: 0,
    fusionGroups: 0,
    fusionEnabled: true,
  };

  const useTopLevelSharedEncoder = backend.name === "webgpu";

  // Refresh per-step scalar values as DATA before ANY execution path —
  // lowered actions and compiled replays both read scalar refs through the
  // table buffers (getInputStorage / recorded dispatch bindings), so this
  // single refresh keeps every value-independent cache honest. See
  // scalar-table.ts.
  refreshScalarTable(loweredPlan, planNodes, backend);

  // Late-varying inlined scalars: a scalar whose value was constant through
  // the recording executions is INLINED in fused-recipe WGSL (and thus in the
  // recorded compiled plan). If such a value changes later — an LR schedule
  // leaving its warmup plateau — the replay would compute with the baked
  // value and the lowered-path adaptation could never fire (the fast path
  // bypasses the fused actions). Detect it here and drop the compiled plan;
  // the lowered execution below adapts the recipe (demotes the scalar to a
  // runtime input) and re-records.
  if (
    loweredPlan.compiledPlan?.valid &&
    inlinedFusionScalarsStale(loweredPlan, planNodes)
  ) {
    destroyCompiledPlanBuffers(loweredPlan.compiledPlan);
    loweredPlan.compiledPlan = undefined;
  }

  // =========================================================================
  // FAST PATH: Compiled Plan Execution
  // =========================================================================
  // If we have a valid compiled plan, execute it directly.
  // The compiled plan is a flat sequence of GPU primitives (alloc, dispatch,
  // copy, write, barrier) with abstract slot indices instead of concrete buffers.
  // Compiled plan: enabled by default. Disable with TORCHLETTE_COMPILED_PLAN=0.
  if (
    loweredPlan.compiledPlan?.valid &&
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    process.env.TORCHLETTE_COMPILED_PLAN !== "0" &&
    // Liveness-aware arena reuses one buffer across multiple positions, so
    // compiled replay needs the EXPERIMENTAL planned-buffer mode (such plans
    // carry allocBuffers — the pinned, lifetime-split assignment recorded
    // from the pool-reusing execution). See buildCompiledPlan for status.
    (!arenaLivenessEnabled() || compiledPlannedEnabled())
  ) {
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      console.log(
        `[exec] COMPILED nodes=${planNodes.length} cmds=${loweredPlan.compiledPlan.commands.length}`,
      );
    }
    const externalInputBuffers = collectExternalInputBuffers(planNodes);
    try {
      await executeCompiledPlan(
        loweredPlan.compiledPlan,
        planNodes,
        options.bufferArena,
        backend,
        externalInputBuffers,
      );
    } finally {
      clearActiveScalarTable();
    }
    return {
      result: planNodes[planNodes.length - 1].result!,
      stats: loweredPlan.cachedStats ?? stats,
    };
  }

  // =========================================================================
  // NORMAL PATH (with optional compilation recording for compiled plan)
  // =========================================================================

  if (process.env.TORCHLETTE_DEBUG_COMPILED) {
    console.log(
      `[exec] NORMAL nodes=${planNodes.length} hasCompiledPlan=${!!loweredPlan.compiledPlan?.valid} hasArena=${!!options.bufferArena} arenaLen=${options.bufferArena?.resolve?.length ?? "n/a"}`,
    );
  }

  // Shared encoder scope
  if (useTopLevelSharedEncoder) beginSharedEncoder();

  // Activate buffer arena for this plan (stabilizes buffer identities for bind group cache)
  if (options.bufferArena && useTopLevelSharedEncoder) {
    setActiveArena(options.bufferArena);
  }

  // Register external input buffers so the arena can detect conflicts.
  // When the same template is reused (e.g., same transformer block structure
  // across layers), the previous execution's output arena buffer becomes the
  // next execution's external input. Without this, the arena would return the
  // same buffer for both reading (external input) and writing (fused output),
  // causing data corruption.
  if (options.bufferArena && useTopLevelSharedEncoder) {
    setArenaExternalInputBuffers(collectExternalInputBuffers(planNodes));
  }

  // Start compilation recording for compiled plan.
  // Bounded-arena (liveness) mode is supported via PLANNED BUFFERS: slot
  // resolution happens at record time (lifetime splitting in recordAlloc),
  // and buildCompiledPlan pins the recorded pool-buffer assignment to the
  // plan — replays bind the exact lifetime-reusing assignment the recording
  // proved safe, with memory bounded to the recording's working set.
  const shouldCompile =
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    !loweredPlan.compiledPlan &&
    options.bufferArena.resolve.length > 0 && // Arena populated from prior execution
    // Bounded-arena mode: compile only under the experimental planned-buffer
    // flag (see buildCompiledPlan), else stay on the lowered path as before.
    (!arenaLivenessEnabled() || compiledPlannedEnabled()) &&
    // Debug bisection: only compile plans up to N nodes.
    (!process.env.TORCHLETTE_COMPILED_MAX_NODES ||
      planNodes.length <=
        parseInt(process.env.TORCHLETTE_COMPILED_MAX_NODES, 10)) &&
    // Debug bisection: only compile plans whose node count is in the list.
    (!process.env.TORCHLETTE_COMPILED_ONLY_NODES ||
      process.env.TORCHLETTE_COMPILED_ONLY_NODES.split(",").includes(
        String(planNodes.length),
      ));
  let compilationRecording: ReturnType<
    typeof startCompilationRecording
  > | null = null;
  if (shouldCompile) {
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      console.log(`[exec] RECORDING nodes=${planNodes.length}`);
    }
    compilationRecording = startCompilationRecording();
    // Pre-assign external input slots for inputs that are already materialized
    // (results from prior plan executions). Inputs produced within this plan
    // are NOT yet materialized and will get slots during recording via arena alloc.
    for (let i = 0; i < planNodes.length; i++) {
      const node = planNodes[i];
      for (let j = 0; j < node.inputs.length; j++) {
        const ref = node.inputs[j];
        // Only assign if the input is already materialized
        let storage: StorageHandle | undefined;
        if (ref.kind === "materialized") {
          storage = ref.storage;
        } else if (ref.kind === "scalar") {
          continue; // Scalar constants don't have GPU buffers
        } else {
          const idx = ref.outputIndex ?? 0;
          storage = idx === 0 ? ref.node.result : ref.node.results?.[idx];
        }
        if (!storage) continue;
        const buf = gpuBuffer(storage.backendTensor);
        assignSlot(buf, {
          kind: "external",
          planNodeIndex: i,
          inputIndex: j,
        });
      }
    }
  }

  // Pre-resolve dynamic imports before the action loop so subsequent calls
  // to executeFusedSegment avoid per-call async import overhead.
  await ensureFusionImports();

  // =========================================================================
  // Liveness analysis: free intermediate buffers as soon as their last
  // consuming action completes. After release, buffers go to the pool's
  // pendingRelease queue and become reusable at the next reclaim point.
  //
  // Key insight: the lowered plan's action order does NOT match plan-node
  // step order (fused groups can cover non-contiguous positions). So we
  // track the last ACTION INDEX that reads each node, not the last step.
  // =========================================================================
  // Liveness-based buffer release: free intermediate buffers mid-plan execution.
  // Enabled by default; opt out with TORCHLETTE_LIVENESS_RELEASE=0.
  const enableLivenessRelease = process.env.TORCHLETTE_LIVENESS_RELEASE !== "0";
  let livenessOutputIds: Set<number> | null = null;
  let livenessReleased: Set<number> | null = null;
  let livenessDeferred: Set<number> | null = null;
  let livenessNodeToStorage: Map<number, StorageHandle> | null = null;
  let livenessNodeIdToIndex: Map<number, number> | null = null;
  // nodeId → last action index that reads from this node
  let livenessLastAction: Map<number, number> | null = null;
  // actionIndex → [nodeIds to release after this action]
  let livenessReleaseSchedule: Map<number, number[]> | null = null;

  if (enableLivenessRelease) {
    const nodeIdSet = new Set(planNodes.map((n) => n.id));

    // Protected nodes: plan terminals + already-materialized + live RuntimeTensors.
    // Materialized nodes (node.result) MUST stay protected — their buffers may be
    // read by subsequent plan actions even after the liveness analysis thinks they're
    // dead (the analysis only sees plan-internal references, not external ones like
    // compiled plan cache slots or arena bindings).
    livenessOutputIds = new Set<number>();
    livenessOutputIds.add(planNodes[planNodes.length - 1].id);
    // Trust plan.outputIndices first — this is the explicit contract from the
    // engine ("the caller needs these results"). Without this, multi-output ops
    // like adamStep can have node.result cleared mid-plan even though the new
    // adamStep node tied to a model parameter still needs it for materialization.
    if (plan.outputIndices) {
      for (const idx of plan.outputIndices) {
        livenessOutputIds.add(plan.nodes[idx].id);
      }
    }
    // Legacy fallback for callers that don't set outputIndices.
    const livePendingIds = getLivePendingNodeIds();
    for (const node of planNodes) {
      if (node.result) livenessOutputIds.add(node.id);
      if (livePendingIds.has(node.id)) livenessOutputIds.add(node.id);
    }

    // CROSS-PLAN CONSUMERS: a step's pending graph can split across several
    // plans (e.g. the foreach optimizer's update graph). A node in THIS plan
    // that an external live root's closure references will be read by a
    // LATER plan — it must be protected from liveness release and from
    // buffer donation. Walk the pending graph from every live root that is
    // NOT itself in this plan and protect any plan node encountered.
    {
      const visited = new Set<number>();
      const stack: LazyIRNode[] = [];
      for (const rootObj of getLivePendingRootNodes()) {
        const root = rootObj as LazyIRNode;
        if (nodeIdSet.has(root.id)) continue; // this plan's own output
        if (!visited.has(root.id)) {
          visited.add(root.id);
          stack.push(root);
        }
      }
      while (stack.length > 0) {
        const n = stack.pop()!;
        if (nodeIdSet.has(n.id)) {
          // Reached INTO the plan from outside: protect, don't descend —
          // the node's own inputs are protected transitively only if some
          // external path reads them; in-plan reads are already covered by
          // livenessLastAction.
          livenessOutputIds.add(n.id);
          continue;
        }
        if (n.result) continue; // materialized boundary — replays from storage
        for (const ref of n.inputs) {
          if (ref.kind !== "pending") continue;
          const child = ref.node as LazyIRNode;
          if (visited.has(child.id)) continue;
          visited.add(child.id);
          stack.push(child);
        }
      }
    }

    // Map nodeId → plan index for O(1) lookup
    livenessNodeIdToIndex = new Map();
    for (let i = 0; i < planNodes.length; i++) {
      livenessNodeIdToIndex.set(planNodes[i].id, i);
    }

    // For each plan node, find the last action that reads it as an input.
    // This operates in ACTION-INDEX space (not plan-step space), which is
    // correct because actions can cover non-contiguous plan positions.
    livenessLastAction = new Map<number, number>();
    for (let ai = 0; ai < loweredPlan.actions.length; ai++) {
      const actionNodeIndices = getActionNodeIndices(loweredPlan.actions[ai]);
      for (const ni of actionNodeIndices) {
        const node = planNodes[ni];
        for (const ref of node.inputs) {
          if (ref.kind === "pending" && nodeIdSet.has(ref.node.id)) {
            const cur = livenessLastAction.get(ref.node.id);
            if (cur === undefined || ai > cur) {
              livenessLastAction.set(ref.node.id, ai);
            }
          }
        }
      }
    }

    // Build release schedule: after action[i] completes, release these nodes
    livenessReleaseSchedule = new Map();
    for (const [nodeId, lastAi] of livenessLastAction) {
      if (livenessOutputIds.has(nodeId)) continue;
      let list = livenessReleaseSchedule.get(lastAi);
      if (!list) {
        list = [];
        livenessReleaseSchedule.set(lastAi, list);
      }
      list.push(nodeId);
    }

    livenessReleased = new Set();
    livenessDeferred = new Set();
    livenessNodeToStorage = new Map();
  }

  try {
    for (
      let actionIndex = 0;
      actionIndex < loweredPlan.actions.length;
      actionIndex++
    ) {
      const action = loweredPlan.actions[actionIndex];
      switch (action.kind) {
        case "fused": {
          // Reconstruct FusionGroup from plan nodes
          const groupNodes = action.coveredNodeIndices.map((i) => planNodes[i]);
          const outputNode = planNodes[action.outputNodeIndex];
          const additionalOutputNodes = action.additionalOutputNodeIndices.map(
            (i) => planNodes[i],
          );
          const neededIntermediates = action.neededIntermediateNodeIndices.map(
            (i) => planNodes[i],
          );

          // Reconstruct external inputs using cached pattern (O(n) instead of O(n²))
          let extInputs: LazyRef[];
          if (action.cachedExternalInputPattern) {
            // Fast path: use pre-computed pattern
            extInputs = action.cachedExternalInputPattern.map(
              (p) => groupNodes[p.nodeLocalIdx].inputs[p.inputIdx],
            );
          } else {
            // First execution: compute pattern with dedup, then cache it
            const groupNodeIds = new Set(groupNodes.map((n) => n.id));
            extInputs = [];
            const pattern: Array<{ nodeLocalIdx: number; inputIdx: number }> =
              [];
            for (let ni = 0; ni < groupNodes.length; ni++) {
              const node = groupNodes[ni];
              for (let ii = 0; ii < node.inputs.length; ii++) {
                const inp = node.inputs[ii];
                if (inp.kind === "pending") {
                  if (
                    !groupNodeIds.has(inp.node.id) &&
                    !extInputs.some(
                      (ei) =>
                        ei.kind === "pending" && ei.node.id === inp.node.id,
                    )
                  ) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                } else if (inp.kind === "scalar") {
                  // Positional — never value-deduped (must mirror
                  // collectExternalInputs / groupToRecipe; equal-today
                  // values may diverge across steps).
                  extInputs.push(inp);
                  pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                } else {
                  if (
                    !extInputs.some(
                      (ei) =>
                        ei.kind === "materialized" &&
                        ei.storage.id === inp.storage.id,
                    )
                  ) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                }
              }
            }
            action.cachedExternalInputPattern = pattern;
          }

          const group: FusionGroup = {
            nodes: groupNodes,
            planIndices: action.coveredNodeIndices,
            externalInputs: extInputs,
            outputNode,
            additionalOutputNodes:
              additionalOutputNodes.length > 0
                ? additionalOutputNodes
                : undefined,
            neededIntermediates:
              neededIntermediates.length > 0 ? neededIntermediates : undefined,
          };

          // Frozen-scalar adaptation: the cached recipe bakes inlined scalar
          // VALUES; the template fingerprint deliberately ignores them. If a
          // scalar's current value differs from the baked one (per-step
          // coefficient, scheduled LR), demote it to a RUNTIME input — its
          // value then flows as data through the scalar table on every
          // execution. The rebuilt recipe replaces the cached one and any
          // compiled plan recorded against the old kernel is invalidated
          // (adaptation fires on the recording execution — the first one
          // where the value can differ — so the re-recording uses the new
          // kernel). Inlining is kept for scalars that never change.
          {
            let stale: Set<number> | null = null;
            const rin = action.recipe.inputs;
            for (let i = 0; i < rin.length; i++) {
              const inp = rin[i];
              if (!inp?.isInlinedConstant) continue;
              const ref = extInputs[i];
              if (!ref) continue;
              const cur =
                ref.kind === "scalar"
                  ? { inlinable: true as const, value: ref.value }
                  : isInlinableScalar(ref);
              if (cur.inlinable && cur.value !== inp.inlinedValue) {
                (stale ??= new Set(action.runtimeScalarInputs)).add(i);
              }
            }
            if (stale) {
              action.runtimeScalarInputs = stale;
              action.recipe = groupToRecipe(group, stale);
              if (loweredPlan.compiledPlan) {
                destroyCompiledPlanBuffers(loweredPlan.compiledPlan);
                loweredPlan.compiledPlan = undefined;
              }
            }
          }

          // BUFFER DONATION eligibility: external inputs whose LAST reader
          // is this action and which are not plan outputs / externally
          // referenced — their buffers may be overwritten in place by the
          // fused kernel's primary output (segment-executors picks one).
          let donatableInputIds: Set<number> | undefined;
          if (livenessLastAction && livenessOutputIds) {
            for (const ref of group.externalInputs) {
              if (!ref || ref.kind !== "pending") continue;
              const nid = ref.node.id;
              if (
                livenessLastAction.get(nid) === actionIndex &&
                !livenessOutputIds.has(nid)
              ) {
                (donatableInputIds ??= new Set()).add(nid);
              }
            }
          }
          await executeFusedSegment(
            group,
            action.recipe,
            backend,
            action.enableVectorization,
            donatableInputIds,
          );
          stats.fusedNodes += groupNodes.length;
          stats.fusionGroups++;
          break;
        }

        case "matmul-epilogue": {
          const matmulNode = planNodes[action.matmulNodeIndex];
          const outputNode = planNodes[action.outputNodeIndex];

          // Cache the label string (structural, same across steps)
          let epLabel = action.cachedLabel;
          if (!epLabel) {
            const prologueLabel = action.prologues ? "prologue+" : "";
            const epilogueLabel =
              action.epilogueOps.length > 0
                ? "+" + action.epilogueOps.map((o) => o.kind).join("+")
                : "";
            epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(
              /\+$/,
              "",
            );
            action.cachedLabel = epLabel;
          }

          // ── FAST PATH: use cached dispatch config (2nd+ lowered plan execution) ──
          if (action.cachedDispatchConfig) {
            const cfg = action.cachedDispatchConfig;
            setCurrentOpLabel(epLabel);
            setProfileModule(matmulNode.module ?? "unknown");
            const _profT0 = profileOpBegin(epLabel);
            try {
              // Resolve GPU buffers from cached paths (plan-node-relative lookups)
              const refA =
                planNodes[cfg.inputAPath.planNodeIndex].inputs[
                  cfg.inputAPath.inputIndex
                ];
              const refB =
                planNodes[cfg.inputBPath.planNodeIndex].inputs[
                  cfg.inputBPath.inputIndex
                ];
              const bufA = gpuBuffer(getInputStorage(refA).backendTensor);
              const bufB = gpuBuffer(getInputStorage(refB).backendTensor);
              const epilogueBuffers: GPUBuffer[] = [];
              for (const path of cfg.epilogueInputPaths) {
                const ref =
                  planNodes[path.planNodeIndex].inputs[path.inputIndex];
                epilogueBuffers.push(
                  gpuBuffer(getInputStorage(ref).backendTensor),
                );
              }

              // Dispatch directly — skips shape computation, transpose detection,
              // contiguous checks, prologue resolution, dynamic imports
              const resultTensor = _webgpuMatmulImports!.dispatchMatmulDirect(
                bufA,
                bufB,
                {
                  m: cfg.m,
                  n: cfg.n,
                  k: cfg.k,
                  transA: cfg.transA,
                  transB: cfg.transB,
                  batchSize: cfg.batchSize,
                  batchStrideA: cfg.batchStrideA,
                  batchStrideB: cfg.batchStrideB,
                  batchStrideC: cfg.batchStrideC,
                  outShape: cfg.outShape,
                  dtypeA: cfg.dtypeA,
                  dtypeB: cfg.dtypeB,
                  outputDtype: cfg.outputDtype,
                  epilogueConfig: cfg.epilogueConfig as
                    | EpilogueConfig
                    | undefined,
                  epilogueBuffers,
                  inputCastA: cfg.inputCastA,
                  inputCastB: cfg.inputCastB,
                },
              );
              // Fix up shape if epilogue absorbed a reshape (e.g. [1,M,N]→[M,N])
              const fastOutShape = outputNode.shape;
              if (!shapesEqual(resultTensor.shape, fastOutShape)) {
                const gpuT = asGPUTensor(resultTensor);
                gpuT.shape = fastOutShape;
                gpuT.strides = contiguousStrides(fastOutShape);
              }
              outputNode.result = createStorageHandle(
                outputNode.device,
                resultTensor,
              );
            } finally {
              profileOpEnd(epLabel, _profT0);
              setCurrentOpLabel(null);
              setProfileModule("unknown");
            }

            break;
          }

          // ── SLOW PATH: first lowered plan execution — full reconstruction ──

          const { epilogueInputRefs, epilogueInputPaths } =
            reconstructEpilogueInputs(planNodes, action.coveredNodeIndices);

          // Reconstruct prologues and resolve prologue decisions
          let prologues: MatmulPrologueInfo[] | undefined;
          let inputCastA: "f16" | "f32" | undefined;
          let inputCastB: "f16" | "f32" | undefined;
          let resolvedInputRefA = matmulNode.inputs[0];
          let resolvedInputRefB = matmulNode.inputs[1];
          let inputAPath: PlanNodePath = {
            planNodeIndex: action.matmulNodeIndex,
            inputIndex: 0,
          };
          let inputBPath: PlanNodePath = {
            planNodeIndex: action.matmulNodeIndex,
            inputIndex: 1,
          };

          if (action.prologues && action.prologues.length > 0) {
            const resolved = resolvePrologueInputs(
              planNodes,
              matmulNode,
              action.matmulNodeIndex,
              action.prologues,
            );
            prologues = resolved.prologues;
            inputCastA = resolved.inputCastA;
            inputCastB = resolved.inputCastB;
            resolvedInputRefA = resolved.resolvedInputRefA;
            resolvedInputRefB = resolved.resolvedInputRefB;
            inputAPath = resolved.inputAPath;
            inputBPath = resolved.inputBPath;
          }

          await withProfileContext(epLabel, matmulNode.module, () =>
            executeMatmulWithEpilogue(matmulNode, {
              consumedCount: action.consumedCount,
              epilogueOps: action.epilogueOps,
              epilogueInputRefs,
              outputDtype: action.outputDtype,
              outputNode,
              prologues,
            }),
          );

          // Cache dispatch config for next step
          await buildAndCacheDispatchConfig(
            action,
            resolvedInputRefA,
            resolvedInputRefB,
            inputCastA,
            inputCastB,
            inputAPath,
            inputBPath,
            epilogueInputRefs,
            epilogueInputPaths,
          );

          break;
        }

        case "adam-batch": {
          const useSharedEncoder = backend.name === "webgpu";
          if (useSharedEncoder) {
            // Flush the shared encoder BEFORE returning pending-release
            // buffers to the pool — otherwise an encoded-but-not-submitted
            // command could still be referencing one. backend.ops.adamStepBatch
            // also flushes internally; this earlier flush exists to make
            // flushBufferPool() safe.
            flushSharedEncoder();
            flushBufferPool();
            recordBarrier();
          }

          const adamBackend =
            getBackend(planNodes[action.nodeIndices[0]].device) ?? backend;
          const adamStepBatch = adamBackend.ops.adamStepBatch;
          if (!adamStepBatch) {
            throw new Error(
              "adamStepBatch not supported by backend; cannot execute adam-batch action",
            );
          }

          // Resolve inputs for every adamStep node, then call the backend op
          // once. The backend handles flushing, packing, and per-item fallback;
          // it returns BackendTensors which we wrap as StorageHandles via the
          // shared multi-output protocol.
          await withProfileContext("adamStep", "optimizer.step", async () => {
            const nodes: LazyIRNode[] = [];
            const inputStorages: StorageHandle[][] = [];
            const items: import("../backend/types").AdamBatchItem[] = [];
            let firstNodeIdx = -1;
            for (const nodeIdx of action.nodeIndices) {
              const node = planNodes[nodeIdx];
              if (node.result) continue;
              if (firstNodeIdx < 0) firstNodeIdx = nodeIdx;
              const inputs = node.inputs.map((ref) =>
                getInputStorage(ref, adamBackend),
              );
              const config = node.payload as AdamStepConfig;
              nodes.push(node);
              inputStorages.push(inputs);
              items.push({
                grad: inputs[0].backendTensor,
                param: inputs[1].backendTensor,
                m: inputs[2].backendTensor,
                v: inputs[3].backendTensor,
                config,
              });
            }
            if (items.length === 0) return;
            // Attribute volatile uniform recordings (Adam config) to the first
            // adamStep node — all items in a batch share hyperparameters by
            // construction, so any node is a valid replay-time config source.
            if (compilationRecording) setRecordingNodeIndex(firstNodeIdx);
            const results = adamStepBatch(items);
            for (let i = 0; i < nodes.length; i++) {
              const r = results[i];
              assignNodeResult(
                nodes[i],
                { primary: r.param, extras: [r.m, r.v] },
                inputStorages[i].map((s) => s.backendTensor),
                inputStorages[i],
              );
            }
          });
          break;
        }

        case "batched-reduction": {
          const batchNodes = action.nodeIndices.map((i) => planNodes[i]);
          const batchBackend = getBackend(batchNodes[0].device) ?? backend;
          const batchInputs = batchNodes.map((node) => {
            const storage = getInputStorage(node.inputs[0], batchBackend);
            return storage.backendTensor;
          });
          const batchedSum = batchBackend.ops.batchedReduction;
          if (batchedSum) {
            const results = batchedSum(
              action.reduceOp,
              batchInputs,
              action.payload.dim,
              action.payload.keepdim,
            );
            for (let bi = 0; bi < batchNodes.length; bi++) {
              batchNodes[bi].result = wrapResultAsStorage(
                batchNodes[bi].device,
                results[bi],
                [batchInputs[bi]],
                [],
              );
            }
            stats.sequentialNodes += batchNodes.length;
          } else {
            // Fallback: dispatch individually
            for (let bi = 0; bi < batchNodes.length; bi++) {
              const node = batchNodes[bi];
              const inputs = node.inputs.map((ref) =>
                getInputStorage(ref, batchBackend),
              );
              const backendInputs = inputs.map((s) => s.backendTensor);
              const result = executeOpSync(node, backendInputs, batchBackend);
              const handlerResult =
                result instanceof Promise ? await result : result;
              assignNodeResult(node, handlerResult, backendInputs, inputs);
              stats.sequentialNodes++;
            }
          }
          break;
        }

        case "sequential":
        case "view":
        case "data-source": {
          const nodeIdx = action.nodeIndex;
          const node = planNodes[nodeIdx];
          if (node.result) break;

          // Tag the node for recording hooks (volatile uniform attribution).
          if (compilationRecording) setRecordingNodeIndex(nodeIdx);

                  setProfileModule(node.module ?? "unknown");
          let inputs;
          try {
            inputs = node.inputs.map((ref) =>
              getInputStorage(ref, backend),
            );
          } catch (e) {
            // Dump context for debugging
            console.error(`[lowered-plan] Action #${actionIndex} kind=${action.kind} nodeIdx=${nodeIdx} node=${node.id}:${node.op} shape=${JSON.stringify(node.shape)}`);
            for (let ii = 0; ii < node.inputs.length; ii++) {
              const ref = node.inputs[ii];
              if (ref.kind === "pending") {
                const depPlanIdx = planNodes.findIndex(n => n.id === ref.node.id);
                console.error(`[lowered-plan]   input[${ii}]: pending node=${ref.node.id}:${ref.node.op} result=${!!ref.node.result} planIdx=${depPlanIdx}`);
              }
            }
            throw e;
          }
          const backendInputs = inputs.map((s) => s.backendTensor);

          // Use synchronous dispatch to avoid microtask overhead (~5-15µs/await).
          // Only adamStep and transfer are truly async; adamStep never reaches
          // this branch because lowered-plan.ts always emits an adam-batch
          // action for it (see the comment there).
          const resultOrPromise = executeOpSync(node, backendInputs, backend);
          const handlerResult =
            resultOrPromise instanceof Promise
              ? await resultOrPromise
              : resultOrPromise;
          assignNodeResult(node, handlerResult, backendInputs, inputs);
          stats.sequentialNodes++;

          // Record data-source for compiled plan (re-executed each step)
          if (action.kind === "data-source" && isCompilationRecordingActive()) {
            recordWrite(gpuBuffer(node.result!.backendTensor), nodeIdx);
          }
          break;
        }

        case "prologue-skip": {
          // Prologue-claimed cast nodes are skipped — their work is absorbed
          // into the matmul tile load.
          break;
        }

        case "row-program": {
          // Remap inputRefs to current step's nodes (template reuse makes
          // the original refs stale — they point to cleared previous-step nodes).
          const remappedRefs = action.inputRefs.map((ref, i) => {
            const pos = action.inputRefPositions[i];
            if (pos >= 0 && ref.kind === "pending") {
              const currentNode = planNodes[pos];
              return {
                kind: "pending" as const,
                node: currentNode,
                outputIndex: ref.outputIndex,
              };
            }
            return ref;
          });

          await executeRowProgram(
            action.program,
            remappedRefs,
            planNodes[action.outputNodeIndex],
            action.numRows,
            action.dimSize,
            planNodes.filter((_n, i) => action.coveredNodeIndices.includes(i)),
            backend,
          );
          break;
        }

        case "reclaim": {
          if (useTopLevelSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }
          recordBarrier();
          break;
        }
      }

      // Liveness-based early release: free dead buffers after each action.
      // Uses the pre-built release schedule (action-index based, not step based).
      // Liveness releases stay ON during recording in planned-buffer mode:
      // the pool reuse they enable IS the planned-buffer assignment the
      // compiled plan captures (recordAlloc splits lifetimes into distinct
      // slots). In default (unbounded-arena) mode, releases are skipped
      // during recording as before — arena identities are already stable.
      if (
        livenessReleaseSchedule &&
        (!compilationRecording ||
          compiledPlannedEnabled())
      ) {
        // Register newly-produced storages for release tracking
        const covered = getActionNodeIndices(action);
        for (const ni of covered) {
          const node = planNodes[ni];
          if (node.result) {
            livenessNodeToStorage!.set(node.id, node.result);
          }
        }

        // Release nodes scheduled for this action index
        const toRelease = livenessReleaseSchedule.get(actionIndex);
        if (toRelease) {
          for (const nodeId of toRelease) {
            if (livenessReleased!.has(nodeId)) continue;
            const storage = livenessNodeToStorage!.get(nodeId);
            if (storage && canSafelyRelease(storage, livenessNodeToStorage!)) {
              markLivenessSafe(storage);
              releaseBufferImmediate(storage);
              livenessNodeToStorage!.delete(nodeId);
              const idx = livenessNodeIdToIndex!.get(nodeId);
              if (idx !== undefined) planNodes[idx].result = undefined;
              livenessReleased!.add(nodeId);
            } else if (storage) {
              livenessDeferred!.add(nodeId);
            }
          }
        }

        // Retry deferred nodes whose rc may have dropped since last attempt.
        if (livenessDeferred!.size > 0) {
          for (const nodeId of livenessDeferred!) {
            if (livenessReleased!.has(nodeId)) {
              livenessDeferred!.delete(nodeId);
              continue;
            }
            const storage = livenessNodeToStorage!.get(nodeId);
            if (storage && canSafelyRelease(storage, livenessNodeToStorage!)) {
              markLivenessSafe(storage);
              releaseBufferImmediate(storage);
              livenessNodeToStorage!.delete(nodeId);
              const idx = livenessNodeIdToIndex!.get(nodeId);
              if (idx !== undefined) planNodes[idx].result = undefined;
              livenessReleased!.add(nodeId);
              livenessDeferred!.delete(nodeId);
            }
          }
        }
      }
    }
    if (livenessReleased?.size && process.env.TORCHLETTE_DEBUG_LIVENESS) {
      const outputCount = livenessOutputIds?.size ?? 0;
      const totalNodes = planNodes.length;
      const reclaimCount = loweredPlan.actions.filter(
        (a) => a.kind === "reclaim",
      ).length;
      console.log(
        `[liveness] released=${livenessReleased.size}/${totalNodes} (${outputCount} protected, ${reclaimCount} reclaims)`,
      );
    }
  } finally {
    // Build compiled plan from recording
    if (compilationRecording) {
      stopCompilationRecording();
      // Collect node results for the compiled plan.
      //
      // The compiled plan is a dispatch-level replay: it can only reproduce results
      // that were produced by GPU dispatches recorded during this execution. If a
      // backend op returned a cached/pre-existing buffer (no dispatch), that buffer
      // won't be in bufferToSlot and the compiled plan can't reproduce it.
      //
      // Backend ops must check isCompilationRecording() (from webgpu-state) and
      // bypass optimization caches during recording. If any result still slips
      // through, we invalidate the compiled plan rather than silently dropping it.
      const nodeResults: NodeResult[] = [];
      const unrecordedNodes: string[] = [];
      for (let i = 0; i < planNodes.length; i++) {
        const node = planNodes[i];
        if (node.results && node.results.length > 0) {
          for (let oi = 0; oi < node.results.length; oi++) {
            const sh = node.results[oi];
            if (!sh) continue;
            const bt = asGPUTensor(sh.backendTensor);
            const slot = compilationRecording.bufferToSlot.get(bt.buffer);
            if (slot !== undefined) {
              nodeResults.push({
                nodeIndex: i,
                outputIndex: oi,
                slot,
                shape: bt.shape.slice(),
                strides: bt.strides.slice(),
                dtype: bt.dtype,
                offset: bt.offset,
              });
            } else {
              unrecordedNodes.push(
                `node[${i}].results[${oi}] op=${node.op} shape=${JSON.stringify(node.shape)} dtype=${bt.dtype}`,
              );
            }
          }
        } else if (node.result) {
          const bt = asGPUTensor(node.result.backendTensor);
          const slot = compilationRecording.bufferToSlot.get(bt.buffer);
          if (slot !== undefined) {
            nodeResults.push({
              nodeIndex: i,
              outputIndex: 0,
              slot,
              shape: bt.shape.slice(),
              strides: bt.strides.slice(),
              dtype: bt.dtype,
              offset: bt.offset,
            });
          } else {
            unrecordedNodes.push(
              `node[${i}] op=${node.op} shape=${JSON.stringify(node.shape)} dtype=${bt.dtype}`,
            );
          }
        }
      }
      const compiled = buildCompiledPlan({
        commandLog: compilationRecording.commandLog,
        arena: options.bufferArena!,
        planNodes,
        bufferToSlot: compilationRecording.bufferToSlot,
        slotSources: compilationRecording.slotSources,
        nodeResults,
      });
      if (unrecordedNodes.length > 0) {
        // A backend op returned a cached buffer that wasn't tracked during
        // recording. The op needs to check isCompilationRecording() and bypass
        // its cache. Log details so the developer knows exactly which op to fix.
        console.warn(
          `[compiled-plan] Invalidated: ${unrecordedNodes.length} node(s) have result buffers not tracked during recording.\n` +
            `  This means a backend op returned a cached/pre-existing buffer instead of dispatching.\n` +
            `  Fix: check isCompilationRecording() from webgpu-state and bypass the cache.\n` +
            unrecordedNodes.map((s) => `  - ${s}`).join("\n"),
        );
        compiled.valid = false;
      }
      // Arena conflicts no longer block compiled plan storage. The arena
      // hands evicted buffers back to the pool's release chain (see
      // buffer-arena.ts arenaAllocAt + allocateOutputBuffer + resolveOutputBuffer),
      // so orphaned buffers from conflict resolution get destroyed once their
      // owners release. The compiled plan's per-command bind group cache
      // (compiled-plan.ts) handles buffer pointer changes between replays.
      if (compiled.valid) {
        compiled.endCounters = getDispatchSequenceCounters();
        loweredPlan.compiledPlan = compiled;
      }
      compilationRecording = null;
    }

    // Clear arena conflict flag
    if (getArenaConflictDetected()) {
      clearArenaConflictDetected();
    }

    // Deactivate the scalar-ref table for this plan execution
    clearActiveScalarTable();

    // Deactivate buffer arena
    if (options.bufferArena && useTopLevelSharedEncoder) {
      clearActiveArena();
      clearArenaExternalInputBuffers();
    }

    // Scoped result guarantee: only re-execute output nodes that optimization
    // passes absorbed without producing individual results. Uses recursive
    // dependency resolution since absorbed intermediates may depend on each other.
    // Non-output nodes are internal to the plan — their results are consumed
    // by fusion groups or epilogues, so no external caller needs them.
    {
      const outputNodeIds = plan.outputIndices
        ? new Set(plan.outputIndices.map((i) => plan.nodes[i].id))
        : null; // null = all nodes (conservative fallback)
      const { executeNode: execNode } = await import("./sequential");
      const forceWithDeps = async (node: LazyIRNode): Promise<void> => {
        if (node.result) return;
        for (const inp of node.inputs) {
          if (inp.kind === "pending" && !inp.node.result) {
            await forceWithDeps(inp.node);
          }
        }
        await execNode(node, backend);
      };
      for (const node of planNodes) {
        if (!node.result && (!outputNodeIds || outputNodeIds.has(node.id))) {
          await forceWithDeps(node);
        }
      }
    }

    if (useTopLevelSharedEncoder) {
      endSharedEncoder();
    }
  }

  // Get the result from the last original plan node
  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Lowered plan execution failed: no result for last node");
  }

  // Cache stats on the lowered plan so the compiled path can return them
  loweredPlan.cachedStats = stats;

  return { result: lastNode.result, stats };
}

// ============================================================================
// Optimized Plan Execution (entry point)
// ============================================================================

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
    const result = await executePlanSequential(plan, backend, {
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

  // Derive external node IDs from plan.outputIndices (set by the engine or
  // deserialized from the wire). This replaces the getPendingNodeIds()
  // side-channel — the plan carries its own output contract.
  let externalNodeIds: Set<number> | undefined;
  if (plan.outputIndices) {
    const ids = new Set<number>();
    for (const idx of plan.outputIndices) ids.add(plan.nodes[idx].id);
    if (ids.size > 0) externalNodeIds = ids;
  } else {
    // Fallback for callers that don't set outputIndices (legacy / tests).
    try {
      const { getPendingNodeIds } = await import("../runtime/tensor");
      const pending = getPendingNodeIds();
      if (pending.size > 0) externalNodeIds = pending;
    } catch {
      // runtime/tensor not available — no external node tracking
    }
  }

  // Query device storage buffer limit to constrain fusion group size.
  const maxStorageBuffers: number | undefined =
    backend.device?.limits?.maxStorageBuffersPerShaderStage;

  // Compute 64-bit structural fingerprint (two parallel FNV-1a hashes).
  // Primary is the cache key for O(1) lookup; secondary is checked on hit
  // to detect collisions (effective collision probability: 2^-64).
  const fingerprint = computePlanFingerprint(plan.nodes, externalNodeIds);
  const cachedTemplate = fusionAnalysisCache.get(fingerprint.primary);

  let planNodes: LazyIRNode[];
  let loweredPlan: LoweredPlan;

  // Validate cache hit via secondary fingerprint.
  let validatedTemplate = cachedTemplate?.loweredPlan ? cachedTemplate : undefined;
  if (validatedTemplate && validatedTemplate.fingerprintSecondary !== fingerprint.secondary) {
    if (validatedTemplate.loweredPlan?.compiledPlan) {
      destroyCompiledPlanBuffers(validatedTemplate.loweredPlan.compiledPlan);
    }
    if (validatedTemplate.loweredPlan) {
      destroyScalarTable(validatedTemplate.loweredPlan);
    }
    validatedTemplate = undefined;
    fusionAnalysisCache.delete(fingerprint.primary);
  }

  if (validatedTemplate) {
    // ── Cache hit: reuse existing lowered plan ──
    // If a cross-session recycle (or arena eviction) cleared the
    // bufferArena, recreate it as an empty arena that the executor
    // will populate during the upcoming plan execution. Without this,
    // the cache-hit path would proceed with options.bufferArena =
    // undefined, the shouldCompile check would fail, and compiled-plan
    // recording would never engage for the rest of the process — every
    // subsequent session would run 3-4× slower than the first because
    // every plan would go through the slow normal path forever.
    if (backend.name === "webgpu" && !validatedTemplate.bufferArena) {
      validatedTemplate.bufferArena = { resolve: [], alloc: [] };
    }
    planNodes = validatedTemplate.finalPerm.map((i) => plan.nodes[i]);

    // Re-apply graph rewrites to fresh nodes. The lowered plan was built from
    // rewritten refs (identity-cast bypass, sum-reshape fusion, CSE, etc.).
    // Fresh nodes from deserialization or new steps have original refs. Re-running
    // the cheap O(n) passes ensures input refs match the lowered plan's assumptions.
    {
      const consumers = new Map<number, LazyIRNode[]>();
      const consumerCount = new Map<number, number>();
      for (const node of planNodes) {
        for (const inp of node.inputs) {
          if (inp.kind === "pending") {
            consumerCount.set(inp.node.id, (consumerCount.get(inp.node.id) ?? 0) + 1);
            if (!consumers.has(inp.node.id)) consumers.set(inp.node.id, []);
            consumers.get(inp.node.id)!.push(node);
          }
        }
      }
      runPasses({ planNodes, consumers, consumerCount }, new Set(), SIMPLIFICATION_PASSES);
    }

    loweredPlan = validatedTemplate.loweredPlan!;
    // Replay stored plan analysis on cache hits when profiling is enabled.
    // Only replay once per profiler reset (tracked via generation counter).
    if (isProfilingEnabled() && validatedTemplate.planAnalysis) {
      const gen = getPlanAnalysisGeneration();
      if (validatedTemplate.replayedGeneration !== gen) {
        validatedTemplate.replayedGeneration = gen;
        recordPlanAnalysis({ ...validatedTemplate.planAnalysis });
      }
    }
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      const extCount = externalNodeIds?.size ?? 0;
      const n0 = plan.nodes[0];
      const n0Info = n0
        ? `n0=${n0.op}(${n0.dtype},${JSON.stringify(n0.shape)})`
        : "";
      console.log(
        `[template] HIT fp=0x${fingerprint.primary.toString(16)} nodes=${plan.nodes.length} ext=${extCount} compiled=${!!loweredPlan.compiledPlan?.valid} ${n0Info}`,
      );
    }
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
          finalPoss: seg.nodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
        };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((seg as any).kind === "reduction") {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const rg = (seg as any).group;
        return {
          kind: "reduction" as const,
          finalPoss: rg.nodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
          reductionFinalPos: finalIdToPos.get(rg.reductionNode.id) as number,
          preambleFinalPoss: rg.preambleNodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
          epilogueFinalPoss: rg.epilogueNodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
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
      fingerprintSecondary: fingerprint.secondary,
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
    };

    // Build lowered plan from analysis (the sole plan-building path)
    loweredPlan = buildLoweredPlanFromAnalysis({
      segments: analysis.segments,
      planNodes,
      nodeIdToFinalPos: finalIdToPos,
      prologueClaimedIds: analysis.prologueClaimedIds,
      rowProgramMatches: analysis.rowProgramMatches,
      matmulDirectives: analysis.matmulDirectives,
      enableVectorization,
      // Insert reclaim points so released buffers get promoted to the available pool.
      // Matches the enableLivenessRelease default-on behavior.
      reclaimInterval:
        process.env.TORCHLETTE_LIVENESS_RELEASE !== "0" ? 300 : undefined,
    });
    template.loweredPlan = loweredPlan;
    // Create empty arena for WebGPU — populated during first execution,
    // reused on subsequent cache hits for stable buffer identities.
    if (backend.name === "webgpu") {
      template.bufferArena = { resolve: [], alloc: [] };
    }

    fusionAnalysisCache.set(fingerprint.primary, template);

    // Evict old templates if cache grows too large (each has a GPU arena)
    const MAX_CACHED_TEMPLATES = 16;
    if (fusionAnalysisCache.size > MAX_CACHED_TEMPLATES) {
      // Remove oldest entries (first in map insertion order)
      const toRemove = fusionAnalysisCache.size - MAX_CACHED_TEMPLATES;
      let removed = 0;
      for (const [fp, tmpl] of fusionAnalysisCache) {
        if (removed >= toRemove) break;
        if (fp === fingerprint.primary) continue; // don't evict the one we just added
        if (tmpl.bufferArena) {
          destroyArena(tmpl.bufferArena);
        }
        if (tmpl.loweredPlan?.compiledPlan) {
          destroyCompiledPlanBuffers(tmpl.loweredPlan.compiledPlan);
        }
        if (tmpl.loweredPlan) destroyScalarTable(tmpl.loweredPlan);
        fusionAnalysisCache.delete(fp);
        removed++;
      }
    }

    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      // Log op histogram for structural comparison
      const opCounts = new Map<string, number>();
      for (const n of plan.nodes)
        opCounts.set(n.op, (opCounts.get(n.op) ?? 0) + 1);
      const opSummary = [...opCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([op, c]) => `${op}:${c}`)
        .join(" ");
      const extCount = externalNodeIds?.size ?? 0;
      // Check if this is a near-miss with an existing template (same node count ±2)
      let nearMiss = "";
      for (const [
        existingFp,
        existingTemplate,
      ] of fusionAnalysisCache.entries()) {
        if (existingFp !== fingerprint.primary) {
          const nOld = existingTemplate.finalPerm?.length ?? 0;
          const diff = plan.nodes.length - nOld;
          if (Math.abs(diff) <= 2 && diff !== 0) {
            nearMiss += ` [near: fp=0x${(existingFp >>> 0).toString(16)} nodes=${nOld} diff=${diff > 0 ? "+" : ""}${diff}]`;
            // Diff the op sequences to find the extra node
            // Reconstruct old op list from the cached template
            // We can't access old plan.nodes, but we have the current plan.nodes
            // and the old externalNodeIds. Let's find which nodes are new externals.
            if (externalNodeIds && diff === 1) {
              // Find external nodes that could be the extra
              const extNodes = plan.nodes.filter((n) =>
                externalNodeIds.has(n.id),
              );
              const castExts = extNodes.filter((n) => n.op === "cast");
              if (castExts.length > 0) {
                const c = castExts[castExts.length - 1]; // likely the new one
                const cIdx = plan.nodes.indexOf(c);
                const inp = c.inputs[0];
                const inputOp =
                  inp.kind === "pending" ? inp.node.op : "(materialized)";
                const inputDtype =
                  inp.kind === "pending"
                    ? inp.node.dtype
                    : inp.kind === "materialized"
                      ? inp.storage.backendTensor.dtype
                      : "?";
                console.log(
                  `  EXTRA CAST: node[${cIdx}] cast ${inputDtype}->${c.dtype} shape=${JSON.stringify(c.shape)} input=${inputOp} isExt=${externalNodeIds.has(c.id)}`,
                );
                // Also show the consumer of this cast
                const consumers = plan.nodes.filter((n) =>
                  n.inputs.some(
                    (i) => i.kind === "pending" && i.node.id === c.id,
                  ),
                );
                for (const con of consumers) {
                  console.log(
                    `    consumer: ${con.op} shape=${JSON.stringify(con.shape)} dtype=${con.dtype}`,
                  );
                }
              }
            }
          }
        }
      }
      console.log(
        `[template] MISS fp=0x${fingerprint.primary.toString(16)} nodes=${plan.nodes.length} ext=${extCount} actions=${loweredPlan.actions.length} cacheSize=${fusionAnalysisCache.size}${nearMiss}\n  ops: ${opSummary}`,
      );
    }

    // Collect plan analysis for profiling (structural, no execution needed)
    // Store on template so cache hits can replay it when profiling is enabled later.
    const pa = collectProfilingStats(
      analysis.segments,
      analysis.epilogueClaimedIds,
      analysis.prologueClaimedIds,
      analysis.matmulEpilogueChains,
      plan.nodes.length,
    );
    const storedTemplate = fusionAnalysisCache.get(fingerprint.primary);
    if (storedTemplate) storedTemplate.planAnalysis = pa;
  }

  // Execute via the lowered plan — the sole execution engine.
  // The buffer arena (created at template construction) provides stable buffer
  // identities across steps for bind group cache hits.
  // Allow runtime toggling from the browser via globalThis (env vars don't
  // exist in browsers and Vite replaces process.env at compile time).
  const arenaDisabled =
    !!process.env.TORCHLETTE_NO_ARENA ||
    !!(globalThis as { __torchletteNoArena?: boolean }).__torchletteNoArena;
  const bufferArena = arenaDisabled
    ? undefined
    : ((cachedTemplate ?? fusionAnalysisCache.get(fingerprint.primary))
        ?.bufferArena as BufferArena | undefined);
  return executeLoweredPlan(plan, planNodes, loweredPlan, backend, {
    bufferArena,
  });
}

// ============================================================================
// Profiling (structural analysis, no execution needed)
// ============================================================================

function collectProfilingStats(
  segments: ExecutionSegment[],
  excludedIds: Set<number>,
  prologueClaimedIds: Set<number>,
  matmulEpilogueChains: Map<number, number[]>,
  totalNodes: number,
): PlanAnalysis {
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
        if (!excludedIds.has(node.id) && !prologueClaimedIds.has(node.id)) {
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
  return planAnalysisRef;
}
