import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DType,
} from "../backend/types";
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
import { contiguousStrides, shapesEqual } from "../core/shape";
import {
  dispatchPackedOptimizer,
  type PackedOptimizerItem,
} from "../optim/packed-dispatch";
import {
  assignSlot,
  buildCompiledPlan,
  type CompiledPlan,
  executeCompiledPlan,
  isCompilationRecordingActive,
  type NodeResult,
  recordBarrier,
  recordWrite,
  startCompilationRecording,
  stopCompilationRecording,
} from "./compiled-plan";
import type {
  OptimizedExecutionResult,
  OptimizedExecutionStats,
} from "./executor-optimized";
import type { FusionGroup } from "./fusion-detect";
import type {
  ExecutionPlan,
  LazyIRNode,
  LazyRef,
  StorageHandle,
} from "./lazy-types";
import type { LoweredPlan } from "./lowered-plan";
import type { MatmulEpiloguePlan, MatmulPrologueInfo } from "./matmul-epilogue";
import {
  _detectTransposeView,
  executeMatmulWithEpilogue,
  formatEpilogueLabel,
} from "./matmul-epilogue";
import {
  _webgpuMatmulGeomImports,
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
  wrapResultAsStorage,
} from "./node-factory";
import {
  executeOpSync,
  getInputStorage,
  withProfileContext,
} from "./op-dispatch";
import { pretunePlanMatmuls } from "./plan-builder";
import { profileOpBegin, profileOpEnd, setProfileModule } from "./profiler";
import type { ReductionGroup } from "./reduction-detect";
import {
  ensureFusionImports,
  executeCompoundSoftmax,
  executeFusedSegment,
  executeReductionSegment,
  executeRowProgram,
} from "./segment-executors";
import { storageTracker } from "./storage-tracker";

type AdamStepFn = NonNullable<
  import("../backend/types").Backend["ops"]["adamStep"]
>;

/**
 * Walk a chain of ops collecting external (non-chain) input refs.
 * When `prevId` is given (epilogue): the first node's chain input comes from prevId.
 * When `prevId` is undefined (preamble): all first-node inputs are external.
 */
function collectChainExternalRefs(
  chainNodes: LazyIRNode[],
  prevId?: number,
): LazyRef[] {
  const refs: LazyRef[] = [];
  const startIdx = prevId !== undefined ? 0 : 1;
  if (prevId === undefined) {
    for (const ref of chainNodes[0].inputs) refs.push(ref);
  }
  for (let ci = startIdx; ci < chainNodes.length; ci++) {
    const node = chainNodes[ci];
    if (node.inputs.length === 2) {
      const prev = ci === 0 ? prevId! : chainNodes[ci - 1].id;
      const inp0IsChain =
        node.inputs[0].kind === "pending" && node.inputs[0].node.id === prev;
      refs.push(node.inputs[inp0IsChain ? 1 : 0]);
    }
  }
  return refs;
}

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

        case "reduction-preamble": {
          const rpChainNodes = action.chainNodeIndices.map((i) => planNodes[i]);
          const rpReductionNode = planNodes[action.reductionNodeIndex];
          const rpGroup: ReductionGroup = {
            nodes: [...rpChainNodes, rpReductionNode],
            planIndices: [
              ...action.chainNodeIndices,
              action.reductionNodeIndex,
            ],
            reductionNode: rpReductionNode,
            preambleNodes: rpChainNodes,
            epilogueNodes: [],
            outputNode: rpReductionNode,
            preambleOps: action.chainOps,
            preambleInputRefs: collectChainExternalRefs(rpChainNodes),
            preambleInputDtypes: action.chainInputDtypes,
            epilogueOps: [],
            epilogueInputRefs: [],
            outputDtype: rpReductionNode.dtype,
            isMean: rpReductionNode.op === "mean",
          };

          const rpLabel = `${rpGroup.isMean ? "mean" : "sum"}+${rpChainNodes.map((n) => n.op).join("+")}`;
          await withProfileContext(rpLabel, rpChainNodes[0].module, () =>
            executeReductionSegment(rpGroup, backend),
          );

          break;
        }

        case "reduction-epilogue": {
          const reNode = planNodes[action.reductionNodeIndex];
          const reOutputNode = planNodes[action.outputNodeIndex];
          const reEpilogueChainNodes = action.coveredNodeIndices
            .slice(1)
            .map((i) => planNodes[i]);

          const reGroup: ReductionGroup = {
            nodes: action.coveredNodeIndices.map((i) => planNodes[i]),
            planIndices: action.coveredNodeIndices,
            reductionNode: reNode,
            preambleNodes: [],
            epilogueNodes: reEpilogueChainNodes,
            outputNode: reOutputNode,
            preambleOps: [],
            preambleInputRefs: [],
            preambleInputDtypes: [],
            epilogueOps: action.epilogueOps,
            epilogueInputRefs: collectChainExternalRefs(
              reEpilogueChainNodes,
              reNode.id,
            ),
            outputDtype: action.outputDtype,
            isMean: reNode.op === "mean",
          };

          const reLabel = `${reNode.op}+${formatEpilogueLabel(action.epilogueOps)}`;
          await withProfileContext(reLabel, reNode.module, () =>
            executeReductionSegment(reGroup, backend),
          );

          break;
        }

        case "reduction-fusion": {
          const rfReductionNode = planNodes[action.reductionNodeIndex];
          const rfOutputNode = planNodes[action.outputNodeIndex];
          const rfPreambleNodes = action.preambleNodeIndices.map(
            (i) => planNodes[i],
          );
          const rfEpilogueNodes = action.epilogueNodeIndices.map(
            (i) => planNodes[i],
          );

          const rfGroup: ReductionGroup = {
            nodes: [...rfPreambleNodes, rfReductionNode, ...rfEpilogueNodes],
            planIndices: [
              ...action.preambleNodeIndices,
              action.reductionNodeIndex,
              ...action.epilogueNodeIndices,
            ],
            reductionNode: rfReductionNode,
            preambleNodes: rfPreambleNodes,
            epilogueNodes: rfEpilogueNodes,
            outputNode: rfOutputNode,
            preambleOps: action.preambleOps,
            preambleInputRefs: collectChainExternalRefs(rfPreambleNodes),
            preambleInputDtypes: action.preambleInputDtypes,
            epilogueOps: action.epilogueOps,
            epilogueInputRefs: collectChainExternalRefs(
              rfEpilogueNodes,
              rfReductionNode.id,
            ),
            outputDtype: action.outputDtype,
            isMean: action.isMean,
          };

          const rfLabel = `${action.isMean ? "mean" : "sum"}+${rfPreambleNodes
            .map((n) => n.op)
            .join("+")}+${formatEpilogueLabel(action.epilogueOps)}`;
          await withProfileContext(rfLabel, rfPreambleNodes[0].module, () =>
            executeReductionSegment(rfGroup, backend),
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

        case "compound": {
          const compOutNode = planNodes[action.outputNodeIndex];
          const firstCoveredNode = planNodes[action.coveredNodeIndices[0]];
          await executeCompoundSoftmax(
            firstCoveredNode,
            compOutNode,
            action.dim,
            action.name,
            backend,
          );
          break;
        }

        case "row-program": {
          await executeRowProgram(
            action.program,
            action.inputRefs,
            planNodes[action.outputNodeIndex],
            action.dim,
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
