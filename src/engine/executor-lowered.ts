import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
} from "../backend/types";
import {
  addReplayPinnedBuffers,
  type BufferArena,
  beginSharedEncoder,
  clearActiveArena,
  clearArenaConflictDetected,
  clearArenaExternalInputBuffers,
  endSharedEncoder,
  flushBufferPool,
  flushSharedEncoder,
  getArenaConflictDetected,
  getArenaResolveIndex,
  getDispatchSequenceCounters,
  hasArenaExternalConflicts,
  type RecordedDispatch,
  replayDispatches,
  setActiveArena,
  setAdamBatchMode,
  setArenaExternalInputBuffers,
  setArenaResolveIndexTo,
  setCurrentOpLabel,
  setDispatchSequenceCounters,
  startDispatchRecording,
  stopDispatchRecording,
} from "../backend/webgpu";
import { dispatchAdamStep } from "../backend/webgpu/adam-kernel";
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
import {
  ENCODER_COPY_OPS,
  type LoweredPlan,
  type ReplayEntry,
  type SeqCounters,
} from "./lowered-plan";
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
import { executeOp, getInputStorage, withProfileContext } from "./op-dispatch";
import { pretunePlanMatmuls } from "./plan-builder";
import { profileOpBegin, profileOpEnd, setProfileModule } from "./profiler";
import type { ReductionGroup } from "./reduction-detect";
import {
  executeCompoundSoftmax,
  executeFusedSegment,
  executeReductionSegment,
} from "./segment-executors";
import { storageTracker } from "./storage-tracker";

type AdamStepFn = NonNullable<
  import("../backend/types").Backend["ops"]["adamStep"]
>;

/** Restore arena + dispatch counters from a replay entry. */
function restoreReplayCounters(entry: {
  arenaResolveIdx: number;
  seqCounters: SeqCounters;
}): void {
  setArenaResolveIndexTo(entry.arenaResolveIdx);
  setDispatchSequenceCounters(
    entry.seqCounters.dispatch,
    entry.seqCounters.params,
    entry.seqCounters.output,
  );
}

/** Execute an op node: resolve backend, get inputs, run executeOp. */
async function executeNodeOp(
  node: LazyIRNode,
  backend: Backend,
): Promise<{
  result: BackendTensor;
  backendInputs: BackendTensor[];
  inputStorages: StorageHandle[];
}> {
  const nodeBackend = getBackend(node.device) ?? backend;
  const inputStorages = node.inputs.map((ref) =>
    getInputStorage(ref, nodeBackend),
  );
  const backendInputs = inputStorages.map((s) => s.backendTensor);
  const result = await executeOp(node, backendInputs, nodeBackend);
  return { result, backendInputs, inputStorages };
}

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
  const sideOutputs = { m: mStorage, v: vStorage };
  storageTracker.markReachable(mStorage.id, sideOutputs);
  storageTracker.markReachable(vStorage.id, sideOutputs);
  if (!adamNode._sideOutputs) adamNode._sideOutputs = {};
  adamNode._sideOutputs.adamMV = sideOutputs;
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
  const mStorage = createStorageHandle(adamNode.device, adamResult.m);
  const vStorage = createStorageHandle(adamNode.device, adamResult.v);
  const sideOutputs = { m: mStorage, v: vStorage };
  storageTracker.markReachable(mStorage.id, sideOutputs);
  storageTracker.markReachable(vStorage.id, sideOutputs);
  if (!adamNode._sideOutputs) adamNode._sideOutputs = {};
  adamNode._sideOutputs.adamMV = sideOutputs;
  adamNode.result = createStorageHandle(adamNode.device, finalResult, s1.id);
}

/** Create a StorageHandle from arena buffer metadata (ownsBuffer: false, no-op destroy). */
function arenaStorage(
  device: DeviceKind,
  meta: {
    buffer: GPUBuffer;
    shape: number[];
    dtype: DType;
    size: number;
    strides: number[];
    offset?: number;
    isContiguous?: boolean;
  },
): StorageHandle {
  return createStorageHandle(device, {
    buffer: meta.buffer,
    shape: meta.shape,
    dtype: meta.dtype,
    size: meta.size,
    strides: meta.strides,
    offset: meta.offset ?? 0,
    isContiguous: meta.isContiguous ?? true,
    ownsBuffer: false,
    destroy() {},
  } as unknown as BackendTensor);
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
  // Dispatch replay cache: enabled by default. Disable with TORCHLETTE_DISPATCH_REPLAY=0.
  const useReplayCache = process.env.TORCHLETTE_DISPATCH_REPLAY !== "0";

  // =========================================================================
  // FAST PATH: Dispatch Replay
  // =========================================================================
  // If we have a valid dispatch replay cache, skip all JS dispatch logic and
  // replay recorded GPU dispatches directly. Only data sources, view ops, and
  // encoder-copy ops (scatterAdd) are re-executed (their data or encoder
  // commands change per step).
  // Pre-check: collect external input buffers for arena conflict detection.
  // If any arena buffer matches an external input, replay bind groups are stale
  // (they reference the old buffer). We must skip replay and use normal execution
  // which replaces the conflicting arena slot with a fresh buffer.
  let extInputBufSet: Set<GPUBuffer> | null = null;
  if (
    useReplayCache &&
    loweredPlan.dispatchCache?.valid &&
    options.bufferArena
  ) {
    extInputBufSet = new Set(collectExternalInputBuffers(planNodes));
    if (hasArenaExternalConflicts(options.bufferArena, extInputBufSet)) {
      // Arena buffers conflict with external inputs — invalidate replay cache.
      // Normal execution path will replace the conflicting arena slots.
      loweredPlan.dispatchCache.valid = false;
    }
  }

  if (
    useReplayCache &&
    loweredPlan.dispatchCache?.valid &&
    useTopLevelSharedEncoder &&
    options.bufferArena
  ) {
    const cache = loweredPlan.dispatchCache;
    beginSharedEncoder();
    setActiveArena(options.bufferArena);

    // Register external input buffers for arena conflict detection
    setArenaExternalInputBuffers(collectExternalInputBuffers(planNodes));

    // Compute stats from lowered plan structure (same as normal path would)
    for (const act of loweredPlan.actions) {
      if (act.kind === "fused") {
        stats.fusedNodes += act.coveredNodeIndices.length;
        stats.fusionGroups++;
      } else if (
        act.kind === "sequential" ||
        act.kind === "data-source" ||
        act.kind === "view" ||
        act.kind === "prologue-skip"
      ) {
        stats.sequentialNodes++;
      }
    }

    try {
      // Batch consecutive dispatch entries for efficient replay
      const dispatchBatch: RecordedDispatch[] = [];
      const flushDispatchBatch = () => {
        if (dispatchBatch.length > 0) {
          replayDispatches(dispatchBatch);
          dispatchBatch.length = 0;
        }
      };

      for (const entry of cache.entries) {
        switch (entry.kind) {
          case "dispatch":
            dispatchBatch.push(entry.dispatch);
            break;
          case "data-source": {
            flushDispatchBatch();
            restoreReplayCounters(entry);
            const dsNode = planNodes[entry.nodeIndex];
            if (dsNode.result) break;
            const { result: dsResult } = await executeNodeOp(dsNode, backend);
            dsNode.result = createStorageHandle(dsNode.device, dsResult);
            break;
          }
          case "view": {
            flushDispatchBatch();
            const vNode = planNodes[entry.nodeIndex];
            if (vNode.result) break;
            if (entry.cachedResult) {
              // Fast path: reconstruct from cached metadata (arena buffers stable)
              setArenaResolveIndexTo(
                entry.arenaResolveIdxAfter ?? entry.arenaResolveIdx,
              );
              vNode.result = arenaStorage(vNode.device, entry.cachedResult);
            } else {
              // Slow path: re-execute (first replay before cache populated)
              setArenaResolveIndexTo(entry.arenaResolveIdx);
              const {
                result: vResult,
                backendInputs: vBI,
                inputStorages: vIS,
              } = await executeNodeOp(vNode, backend);
              vNode.result = wrapResultAsStorage(
                vNode.device,
                vResult,
                vBI,
                vIS,
              );
            }
            break;
          }
          case "sequential": {
            flushDispatchBatch();
            restoreReplayCounters(entry);
            const node = planNodes[entry.nodeIndex];
            if (node.result) break;
            const { result, backendInputs, inputStorages } =
              await executeNodeOp(node, backend);
            node.result = wrapResultAsStorage(
              node.device,
              result,
              backendInputs,
              inputStorages,
            );
            break;
          }
          case "adam-batch": {
            flushDispatchBatch();
            flushSharedEncoder();
            flushBufferPool();
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            setAdamBatchMode(true);
            try {
              const adamOp = backend.ops.adamStep as NonNullable<
                typeof backend.ops.adamStep
              >;
              setCurrentOpLabel("adamStep");
              const _adamBatchT0 = profileOpBegin("adamStep");
              await executeAdamBatchInner(
                planNodes,
                entry.nodeIndices,
                adamOp,
                (ref) => getInputStorage(ref, backend),
              );
              profileOpEnd("adamStep", _adamBatchT0);
            } finally {
              setAdamBatchMode(false);
            }
            break;
          }
          case "reclaim": {
            flushDispatchBatch();
            flushSharedEncoder();
            break;
          }
          case "pre-adam-reclaim": {
            flushDispatchBatch();
            flushSharedEncoder();
            flushBufferPool();
            break;
          }
          case "result": {
            flushDispatchBatch();
            const nr = entry.nodeResult;
            const node = planNodes[nr.nodeIndex];
            if (!node.result) {
              node.result = arenaStorage(node.device, nr);
            }
            break;
          }
          case "side-output": {
            // Restore attnLogsumexp on fusedAttentionForward nodes so a later
            // plan (backward Phase B) can skip re-executing fusedAttentionForward
            // and extractAttentionLogsumexp can read the side output directly.
            const soNode = planNodes[entry.nodeIndex];
            if (!soNode._sideOutputs?.attnLogsumexp) {
              if (!soNode._sideOutputs) soNode._sideOutputs = {};
              soNode._sideOutputs.attnLogsumexp = arenaStorage(
                soNode.device,
                entry,
              );
            }
            break;
          }
        }
      }
      flushDispatchBatch(); // Flush remaining batched dispatches
    } finally {
      clearActiveArena();
      clearArenaExternalInputBuffers();
      if (useTopLevelSharedEncoder) endSharedEncoder();
    }

    // Get the result from the last original plan node
    const lastNode = plan.nodes[plan.nodes.length - 1];
    if (!lastNode.result) {
      throw new Error("Dispatch replay failed: no result for last node");
    }
    return { result: lastNode.result, stats };
  }

  // =========================================================================
  // NORMAL PATH (with optional recording for dispatch replay cache)
  // =========================================================================

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

  // Set up dispatch recording if we have an arena (needed for stable bind groups)
  // and no replay cache yet. We only record after the first execution (arena needs
  // to be populated first), so we check if the arena already has buffers.
  const shouldRecord =
    useReplayCache &&
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    !loweredPlan.dispatchCache &&
    options.bufferArena.resolve.length > 0; // Arena populated from prior execution
  const recordingBuffer: RecordedDispatch[] = [];
  const replayEntries: ReplayEntry[] = [];
  let recordingDispatchIdx = 0;

  /** Capture all new dispatches since last call and push to replay entries. */
  const captureDispatches = () => {
    while (recordingDispatchIdx < recordingBuffer.length) {
      replayEntries.push({
        kind: "dispatch",
        dispatch: recordingBuffer[recordingDispatchIdx++],
      });
    }
  };
  /** Skip all new dispatches since last call (for ops that must re-execute). */
  const skipDispatches = () => {
    recordingDispatchIdx = recordingBuffer.length;
  };
  /** Record a node result for replay. */
  const recordResult = (
    nodeIndex: number,
    bt: {
      buffer: GPUBuffer;
      shape: number[];
      dtype: DType;
      size: number;
      strides: number[];
    },
  ) => {
    replayEntries.push({
      kind: "result",
      nodeResult: {
        nodeIndex,
        buffer: bt.buffer,
        shape: bt.shape.slice(),
        dtype: bt.dtype,
        size: bt.size,
        strides: bt.strides.slice(),
      },
    });
  };

  /** Capture dispatches and record the output node result (common suffix for action cases). */
  const captureAndRecordResult = (nodeIndex: number, node: LazyIRNode) => {
    if (!shouldRecord) return;
    captureDispatches();
    if (node.result) {
      recordResult(nodeIndex, asGPUTensor(node.result.backendTensor));
    }
  };

  if (shouldRecord) {
    startDispatchRecording(recordingBuffer);
    // Also set up matmul and fusion recording buffers (imports cached from prior calls)
    const [matmulMod, fusionMod] = await Promise.all([
      import("../backend/webgpu/matmul/dispatch"),
      import("../backend/webgpu/fusion-dispatch"),
    ]);
    matmulMod.setMatmulRecordingBuffer(recordingBuffer);
    fusionMod.setFusionRecordingBuffer(recordingBuffer);
  }

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

          // Record dispatches and node results for replay cache
          if (shouldRecord) {
            // Capture dispatches emitted during this action
            captureDispatches();
            // Record output node result (inline for ordering)
            if (outputNode.result) {
              recordResult(
                action.outputNodeIndex,
                asGPUTensor(outputNode.result.backendTensor),
              );
            }
            // Record additional output node results
            for (const addIdx of action.additionalOutputNodeIndices) {
              const addNode = planNodes[addIdx];
              if (addNode.result) {
                recordResult(addIdx, asGPUTensor(addNode.result.backendTensor));
              }
            }
          }
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

            captureAndRecordResult(action.outputNodeIndex, outputNode);
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

          captureAndRecordResult(action.outputNodeIndex, outputNode);
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

          captureAndRecordResult(action.reductionNodeIndex, rpReductionNode);
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

          captureAndRecordResult(action.outputNodeIndex, reOutputNode);
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

          captureAndRecordResult(action.outputNodeIndex, rfOutputNode);
          break;
        }

        case "adam-batch": {
          const useSharedEncoder = backend.name === "webgpu";
          if (useSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }

          // Record pre-adam reclaim and capture counter positions
          let adamSeqCountersBefore: SeqCounters | undefined;
          if (shouldRecord && useSharedEncoder) {
            replayEntries.push({ kind: "pre-adam-reclaim" });
            adamSeqCountersBefore = getDispatchSequenceCounters();
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

          // Record adam batch as a single replay entry (must re-execute each step).
          // Capture the sequence counter positions so adam dispatches hit the correct
          // cache positions during replay.
          if (shouldRecord) {
            skipDispatches();
            replayEntries.push({
              kind: "adam-batch",
              nodeIndices: action.nodeIndices,
              seqCounters: adamSeqCountersBefore as SeqCounters,
            });
          }
          break;
        }

        case "sequential":
        case "view":
        case "data-source": {
          const nodeIdx = action.nodeIndex;
          const node = planNodes[nodeIdx];
          if (node.result) break;

          // Capture arena resolve index and sequence counters BEFORE execution
          const arenaResolveIdxBefore = shouldRecord
            ? getArenaResolveIndex()
            : 0;
          const seqCountersBefore = shouldRecord
            ? getDispatchSequenceCounters()
            : undefined;

          const nodeBackend = getBackend(node.device) ?? backend;
          setProfileModule(node.module ?? "unknown");
          const inputs = node.inputs.map((ref) =>
            getInputStorage(ref, nodeBackend),
          );
          const backendInputs = inputs.map((s) => s.backendTensor);

          const resultTensor = await executeOp(
            node,
            backendInputs,
            nodeBackend,
          );
          node.result = wrapResultAsStorage(
            node.device,
            resultTensor,
            backendInputs,
            inputs,
          );
          stats.sequentialNodes++;

          // Record for replay cache
          if (shouldRecord) {
            if (action.kind === "data-source") {
              // Data sources must re-execute each step (host data changes).
              replayEntries.push({
                kind: "data-source",
                nodeIndex: nodeIdx,
                arenaResolveIdx: arenaResolveIdxBefore,
                seqCounters: seqCountersBefore as SeqCounters,
              });
              skipDispatches();
            } else if (action.kind === "view") {
              // Views produce deterministic results (same arena buffer, same
              // shape/strides/offset). Cache the result to skip re-execution.
              const bt = asGPUTensor(node.result?.backendTensor);
              replayEntries.push({
                kind: "view",
                nodeIndex: nodeIdx,
                arenaResolveIdx: arenaResolveIdxBefore,
                arenaResolveIdxAfter: getArenaResolveIndex(),
                cachedResult: {
                  buffer: bt.buffer,
                  shape: bt.shape.slice(),
                  dtype: bt.dtype,
                  size: bt.size,
                  strides: bt.strides.slice(),
                  offset: bt.offset,
                  isContiguous: bt.isContiguous,
                },
              });
              skipDispatches();
            } else if (ENCODER_COPY_OPS.has(node.op)) {
              // Ops that use encoder copy commands (copyBufferToBuffer) alongside
              // compute dispatches must be re-executed during replay — the copy
              // commands are invisible to the compute dispatch recording mechanism.
              replayEntries.push({
                kind: "sequential",
                nodeIndex: nodeIdx,
                arenaResolveIdx: arenaResolveIdxBefore,
                seqCounters: seqCountersBefore as SeqCounters,
              });
              skipDispatches();
            } else {
              // With arena active, all buffer identities are stable (arena returns
              // same GPUBuffer object on subsequent steps). Data-source-consuming
              // ops can be safely replayed — their bind groups reference arena
              // buffers that don't change identity.
              // Safe to replay: capture dispatches
              captureDispatches();
              // Record node result (inline for ordering)
              if (node.result) {
                recordResult(nodeIdx, asGPUTensor(node.result.backendTensor));
              }
              // Record side output for fusedAttentionForward so replay
              // restores attnLogsumexp (needed by extractAttentionLogsumexp
              // in a later plan that skips re-executing this node).
              if (node._sideOutputs?.attnLogsumexp) {
                const soSH = node._sideOutputs.attnLogsumexp;
                const sobt = asGPUTensor(soSH.backendTensor);
                replayEntries.push({
                  kind: "side-output",
                  nodeIndex: nodeIdx,
                  buffer: sobt.buffer,
                  shape: sobt.shape.slice(),
                  dtype: sobt.dtype,
                  size: sobt.size,
                  strides: sobt.strides.slice(),
                });
              }
            }
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
          captureAndRecordResult(action.outputNodeIndex, compOutNode);
          break;
        }

        case "reclaim": {
          if (useTopLevelSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }
          if (shouldRecord) {
            replayEntries.push({ kind: "reclaim" });
          }
          break;
        }
      }
    }
  } finally {
    // Stop recording
    if (shouldRecord) {
      stopDispatchRecording();
      const [matmulMod, fusionMod] = await Promise.all([
        import("../backend/webgpu/matmul/dispatch"),
        import("../backend/webgpu/fusion-dispatch"),
      ]);
      matmulMod.setMatmulRecordingBuffer(null);
      fusionMod.setFusionRecordingBuffer(null);

      // Collect all GPUBuffers referenced by recorded bind groups.
      // These must be pinned (not destroyed) between steps so replay bind groups stay valid.
      const pinnedBuffers = new Set<GPUBuffer>();
      for (const entry of replayEntries) {
        if (entry.kind === "dispatch" && entry.dispatch.buffers) {
          for (const b of entry.dispatch.buffers) pinnedBuffers.add(b);
        }
        if (entry.kind === "result" && entry.nodeResult.buffer) {
          pinnedBuffers.add(entry.nodeResult.buffer);
        }
        if (entry.kind === "side-output" && entry.buffer) {
          pinnedBuffers.add(entry.buffer);
        }
      }

      // Activate pinning globally — buffers survive across markStep() boundaries.
      // Multiple plans accumulate into the same global set.
      addReplayPinnedBuffers(pinnedBuffers);

      // Build and store the dispatch replay cache — but only if no arena
      // conflicts were detected. Conflicts mean bind groups reference replaced
      // buffers and cannot be replayed safely.
      if (getArenaConflictDetected()) {
        // Don't cache — bind groups reference stale (replaced) arena buffers
        clearArenaConflictDetected();
      } else {
        loweredPlan.dispatchCache = {
          entries: replayEntries,
          valid: true,
          pinnedBuffers,
        };
      }
    }

    // If arena conflicts were detected during non-recording execution,
    // invalidate any existing dispatch cache.
    if (getArenaConflictDetected()) {
      if (loweredPlan.dispatchCache) {
        loweredPlan.dispatchCache.valid = false;
      }
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

  return { result: lastNode.result, stats };
}
