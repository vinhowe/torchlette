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
  beginSharedEncoder,
  clearActiveArena,
  clearArenaConflictDetected,
  clearArenaExternalInputBuffers,
  endSharedEncoder,
  flushBufferPool,
  flushSharedEncoder,
  getArenaConflictDetected,
  setActiveArena,
  setAdamBatchMode,
  setArenaExternalInputBuffers,
  setCurrentOpLabel,
} from "../backend/webgpu";
import { dispatchAdamStep } from "../backend/webgpu/adam-kernel";
import { getDispatchSequenceCounters } from "../backend/webgpu/bind-group-cache";
import { bufferPool } from "../backend/webgpu/buffer-pool";
import { f16WeightCache } from "../backend/webgpu/gpu-context";
import {
  asGPUTensor,
  type GPUBuffer,
  gpuBuffer,
} from "../backend/webgpu/gpu-types";
import type { EpilogueConfig } from "../backend/webgpu/matmul/types";
import { createTensor } from "../backend/webgpu/tensor";
import type { FusionGroup } from "../compiler/fusion-detect";
import {
  buildIdPositionMap,
  computePlanFingerprint,
  type ExecutionSegment,
  isFusibleOp,
} from "../compiler/fusion-detect";
import { analyzeGraph } from "../compiler/graph-compiler";
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
  isProfilingEnabled,
  type PlanAnalysis,
  profileOpBegin,
  profileOpEnd,
  recordPlanAnalysis,
  setProfileModule,
} from "../graph/profiler";
import { storageTracker } from "../graph/storage-tracker";
import type {
  ExecutionPlan,
  LazyIRNode,
  LazyRef,
  StorageHandle,
} from "../graph/types";
import {
  dispatchPackedOptimizer,
  type PackedOptimizerItem,
} from "../optim/packed-dispatch";
import {
  assignSlot,
  buildCompiledPlan,
  executeCompiledPlan,
  isCompilationRecordingActive,
  type NodeResult,
  recordBarrier,
  recordWrite,
  startCompilationRecording,
  stopCompilationRecording,
} from "./compiled-plan";
import { buildLoweredPlanFromAnalysis, type LoweredPlan } from "./lowered-plan";
import {
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

type AdamStepFn = NonNullable<
  import("../backend/types").Backend["ops"]["adamStep"]
>;

/**
 * Execute a batch of adamStep nodes using packed dispatch for groups of same-size params.
 * Groups params by element count and dispatches one Adam kernel per group (instead of one
 * per param). Single-param groups fall through to per-param adamOp.
 */
async function executeAdamBatchInner(
  planNodes: LazyIRNode[],
  nodeIndices: number[],
  adamOp: AdamStepFn,
  getStorage: (ref: LazyRef) => StorageHandle,
): Promise<void> {
  // Pre-resolve all storages and build packed items list
  const packedItems: PackedOptimizerItem[] = [];
  const nodes: LazyIRNode[] = [];
  const storages: Array<
    [StorageHandle, StorageHandle, StorageHandle, StorageHandle]
  > = [];

  for (const nodeIdx of nodeIndices) {
    const adamNode = planNodes[nodeIdx];
    if (adamNode.result) continue;
    const inputs = adamNode.inputs;
    const s0 = getStorage(inputs[0]);
    const s1 = getStorage(inputs[1]);
    const s2 = getStorage(inputs[2]);
    const s3 = getStorage(inputs[3]);
    const numElements = adamNode.shape.reduce(
      (a: number, b: number) => a * b,
      1,
    );

    packedItems.push({
      buffers: [
        gpuBuffer(s0.backendTensor),
        gpuBuffer(s1.backendTensor),
        gpuBuffer(s2.backendTensor),
        gpuBuffer(s3.backendTensor),
      ],
      numElements,
    });
    nodes.push(adamNode);
    storages.push([s0, s1, s2, s3]);
  }

  if (packedItems.length === 0) return;

  // Packed dispatch: scatter→dispatch→gather for groups of ≥2 same-size params
  const config = nodes[0].payload as AdamStepConfig;
  const infFlagBuffer = (config.infFlagBuffer as GPUBuffer | null) ?? null;
  const handled = dispatchPackedOptimizer({
    items: packedItems,
    gatherIndices: [1, 2, 3], // gather param, m, v (not grad)
    dispatch(packed, totalElements) {
      dispatchAdamStep(
        packed[0],
        packed[1],
        packed[2],
        packed[3],
        totalElements,
        config,
        false,
        infFlagBuffer,
      );
    },
    label: "packedAdam",
  });

  // Create result StorageHandles for packed items (data updated in-place via gather)
  for (const i of handled) {
    assignPackedAdamResult(nodes[i], storages[i]);
    // Evict stale f16 weight cache entries (param data changed in-place)
    const paramBuf = packedItems[i].buffers[1];
    const oldF16 = f16WeightCache.get(paramBuf);
    if (oldF16) {
      bufferPool.deferredDestroy(oldF16, oldF16.size);
      f16WeightCache.delete(paramBuf);
    }
  }

  // Fall through: dispatch remaining singles via per-param adamOp
  for (let i = 0; i < packedItems.length; i++) {
    if (handled.has(i)) continue;
    const adamNode = nodes[i];
    const [s0, s1, s2, s3] = storages[i];
    const adamPayload = adamNode.payload as AdamStepConfig;
    const adamResult = await adamOp(
      s0.backendTensor,
      s1.backendTensor,
      s2.backendTensor,
      s3.backendTensor,
      adamPayload,
    );
    assignPerParamAdamResult(adamNode, s1, adamResult);
  }
}

/** Assign result + side outputs for a param updated by packed dispatch (in-place via gather copy). */
function assignPackedAdamResult(
  adamNode: LazyIRNode,
  [, s1, s2, s3]: [StorageHandle, StorageHandle, StorageHandle, StorageHandle],
): void {
  // Param: same buffer, ownsBuffer: false since buffer is shared with s1
  adamNode.result = createStorageHandle(
    adamNode.device,
    { ...(s1.backendTensor as Record<string, unknown>), ownsBuffer: false },
    s1.id,
  );

  // m/v: transfer ownership from old tensors to new ones
  // 1. DecRef old (undoes the incRef from the original createTensor)
  // 2. Noop old destroy (prevents double-free when old storage is GC'd)
  // 3. Create new tensor wrapping same buffer (does incRef)
  const mBT = s2.backendTensor as {
    buffer: GPUBuffer;
    shape: number[];
    dtype: DType;
    ownsBuffer?: boolean;
    destroy?: () => void;
  };
  const vBT = s3.backendTensor as {
    buffer: GPUBuffer;
    shape: number[];
    dtype: DType;
    ownsBuffer?: boolean;
    destroy?: () => void;
  };

  if (mBT.ownsBuffer) bufferPool.decRef(mBT.buffer);
  if (vBT.ownsBuffer) bufferPool.decRef(vBT.buffer);
  mBT.destroy = () => {};
  vBT.destroy = () => {};

  const newM = createTensor(mBT.shape, mBT.buffer, undefined, 0, mBT.dtype);
  const newV = createTensor(vBT.shape, vBT.buffer, undefined, 0, vBT.dtype);

  const mStorage = createStorageHandle(adamNode.device, newM);
  const vStorage = createStorageHandle(adamNode.device, newV);
  adamNode.results = [adamNode.result!, mStorage, vStorage];
  storageTracker.markReachable(mStorage.id, adamNode.results);
  storageTracker.markReachable(vStorage.id, adamNode.results);
}

/** Assign result + side outputs for a param updated by per-param adamOp (standard path). */
function assignPerParamAdamResult(
  adamNode: LazyIRNode,
  s1: StorageHandle,
  adamResult: { param: BackendTensor; m: BackendTensor; v: BackendTensor },
): void {
  const paramResult = adamResult.param;
  const paramOwns = (paramResult as { ownsBuffer?: boolean }).ownsBuffer;
  const finalResult =
    paramOwns === true ? { ...paramResult, ownsBuffer: false } : paramResult;
  adamNode.result = createStorageHandle(adamNode.device, finalResult, s1.id);
  const mStorage = createStorageHandle(adamNode.device, adamResult.m);
  const vStorage = createStorageHandle(adamNode.device, adamResult.v);
  adamNode.results = [adamNode.result, mStorage, vStorage];
  storageTracker.markReachable(mStorage.id, adamNode.results);
  storageTracker.markReachable(vStorage.id, adamNode.results);
}

/** Collect GPU buffers from all external (materialized/already-resolved) plan inputs. */
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
    process.env.TORCHLETTE_COMPILED_PLAN !== "0"
  ) {
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      console.log(
        `[exec] COMPILED nodes=${planNodes.length} cmds=${loweredPlan.compiledPlan.commands.length}`,
      );
    }
    const externalInputBuffers = collectExternalInputBuffers(planNodes);
    await executeCompiledPlan(
      loweredPlan.compiledPlan,
      planNodes,
      options.bufferArena,
      backend,
      externalInputBuffers,
    );
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

  // Start compilation recording for compiled plan
  const shouldCompile =
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    !loweredPlan.compiledPlan &&
    options.bufferArena.resolve.length > 0; // Arena populated from prior execution
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

  try {
    for (const action of loweredPlan.actions) {
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
                  if (
                    !extInputs.some(
                      (ei) =>
                        ei.kind === "scalar" &&
                        ei.value === inp.value &&
                        ei.dtype === inp.dtype,
                    )
                  ) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
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

          await executeFusedSegment(
            group,
            action.recipe,
            backend,
            action.enableVectorization,
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
            flushSharedEncoder();
            flushBufferPool();
            recordBarrier();
          }

          setAdamBatchMode(true);
          try {
            const adamBackend =
              getBackend(planNodes[action.nodeIndices[0]].device) ?? backend;
            const adamOp = adamBackend.ops.adamStep as NonNullable<
              typeof adamBackend.ops.adamStep
            >;
            await withProfileContext("adamStep", "optimizer.step", () =>
              executeAdamBatchInner(
                planNodes,
                action.nodeIndices,
                adamOp,
                (ref) => getInputStorage(ref, adamBackend),
              ),
            );
          } finally {
            setAdamBatchMode(false);
          }
          break;
        }

        case "sequential":
        case "view":
        case "data-source": {
          const nodeIdx = action.nodeIndex;
          const node = planNodes[nodeIdx];
          if (node.result) break;

          setProfileModule(node.module ?? "unknown");
          const inputs = node.inputs.map((ref) =>
            getInputStorage(ref, backend),
          );
          const backendInputs = inputs.map((s) => s.backendTensor);

          // Use synchronous dispatch to avoid microtask overhead (~5-15µs/await).
          // Only adamStep and transfer are truly async; neither appears in
          // sequential/view/data-source actions.
          const resultOrPromise = executeOpSync(node, backendInputs, backend);
          const resultTensor =
            resultOrPromise instanceof Promise
              ? await resultOrPromise
              : resultOrPromise;
          node.result = wrapResultAsStorage(
            node.device,
            resultTensor,
            backendInputs,
            inputs,
          );
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
          await executeRowProgram(
            action.program,
            action.inputRefs,
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
      if (compiled.valid && !getArenaConflictDetected()) {
        compiled.endCounters = getDispatchSequenceCounters();
        loweredPlan.compiledPlan = compiled;
      }
      compilationRecording = null;
    }

    // Clear arena conflict flag
    if (getArenaConflictDetected()) {
      clearArenaConflictDetected();
    }

    // Deactivate buffer arena before closing encoder
    if (options.bufferArena && useTopLevelSharedEncoder) {
      clearActiveArena();
      clearArenaExternalInputBuffers();
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
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      const extCount = externalNodeIds?.size ?? 0;
      const n0 = plan.nodes[0];
      const n0Info = n0
        ? `n0=${n0.op}(${n0.dtype},${JSON.stringify(n0.shape)})`
        : "";
      console.log(
        `[template] HIT fp=0x${(fingerprint >>> 0).toString(16)} nodes=${plan.nodes.length} ext=${extCount} compiled=${!!loweredPlan.compiledPlan?.valid} ${n0Info}`,
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
    });
    template.loweredPlan = loweredPlan;

    fusionAnalysisCache.set(fingerprint, template);

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
        if (existingFp !== fingerprint) {
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
        `[template] MISS fp=0x${(fingerprint >>> 0).toString(16)} nodes=${plan.nodes.length} ext=${extCount} actions=${loweredPlan.actions.length} cacheSize=${fusionAnalysisCache.size}${nearMiss}\n  ops: ${opSummary}`,
      );
    }

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
    bufferArena: bufferArena as BufferArena | undefined,
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
