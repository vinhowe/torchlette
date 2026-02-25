import type { AdamStepConfig, Backend, BackendTensor, DType } from "../backend/types";
import { gpuBuffer, asGPUTensor, type GPUBuffer } from "../backend/webgpu/gpu-types";
import { getBackend } from "../backend/registry";
import {
  flushBufferPool,
  flushSharedEncoder,
  beginSharedEncoder,
  endSharedEncoder,
  setCurrentOpLabel,
  setAdamBatchMode,
  setActiveArena,
  clearActiveArena,
  setArenaExternalInputBuffers,
  clearArenaExternalInputBuffers,
  getArenaConflictDetected,
  clearArenaConflictDetected,
  hasArenaExternalConflicts,
  type BufferArena,
  startDispatchRecording,
  stopDispatchRecording,
  replayDispatches,
  type RecordedDispatch,
  setDispatchSequenceCounters,
  getDispatchSequenceCounters,
  addReplayPinnedBuffers,
  getArenaResolveIndex,
  setArenaResolveIndexTo,
} from "../backend/webgpu";
import { profileOpBegin, profileOpEnd, setProfileModule } from "../backend/webgpu/profiler";
import type { FusionGroup } from "./fusion-detect";
import { groupToRecipe } from "./fusion-detect";
import {
  type LoweredPlan,
  type DispatchReplayCache,
  type ReplayEntry,
  type ReplayNodeResult,
  type RecordedDispatch as LPRecordedDispatch,
  ENCODER_COPY_OPS,
} from "./lowered-plan";
import type { LazyIRNode, LazyRef, StorageHandle, ExecutionPlan } from "./lazy-types";
import { createStorageHandle, ensureWebGPUMatmulImports, _webgpuMatmulImports, _webgpuMatmulGeomImports } from "./node-factory";
import { storageTracker } from "./storage-tracker";
import { getInputStorage, executeOp } from "./op-dispatch";
import { pretunePlanMatmuls } from "./plan-builder";
import { executeFusedSegment } from "./segment-executors";
import type { MatmulEpiloguePlan, MatmulPrologueInfo } from "./matmul-epilogue";
import { executeMatmulWithEpilogue, _detectTransposeView, shapesEqual } from "./matmul-epilogue";
import type { ReductionPreamblePlan } from "./reduction-preamble";
import { executeReductionWithPreamble } from "./reduction-preamble";
import type { OptimizedExecutionStats, OptimizedExecutionResult } from "./executor-optimized";

type AdamStepFn = NonNullable<import("../backend/types").Backend["ops"]["adamStep"]>;

/**
 * Execute a batch of adamStep nodes using a direct adamOp call (bypasses executeOp switch).
 * Shared by both the lowered-plan replay path and the normal lowered plan execution path.
 */
async function executeAdamBatchInner(
  planNodes: LazyIRNode[],
  nodeIndices: number[],
  adamOp: AdamStepFn,
  getStorage: (ref: LazyRef) => StorageHandle,
): Promise<void> {
  for (const nodeIdx of nodeIndices) {
    const adamNode = planNodes[nodeIdx];
    if (adamNode.result) continue;
    const inputs = adamNode.inputs;
    const s0 = getStorage(inputs[0]);
    const s1 = getStorage(inputs[1]);
    const s2 = getStorage(inputs[2]);
    const s3 = getStorage(inputs[3]);
    const adamPayload = adamNode.payload as AdamStepConfig;
    const adamResult = await adamOp(
      s0.backendTensor, s1.backendTensor, s2.backendTensor, s3.backendTensor,
      adamPayload,
    );
    // Adam is in-place: param buffer is returned as result.
    // Set ownsBuffer: false since the buffer is shared with the input.
    const paramResult = adamResult.param;
    const paramOwns = (paramResult as { ownsBuffer?: boolean }).ownsBuffer;
    const finalResult = paramOwns === true
      ? { ...paramResult, ownsBuffer: false } as BackendTensor
      : paramResult;
    // Side outputs (m, v) — create storage handles
    const mStorage = createStorageHandle(adamNode.device, adamResult.m);
    const vStorage = createStorageHandle(adamNode.device, adamResult.v);
    const sideOutputs = { m: mStorage, v: vStorage };
    storageTracker.markReachable(mStorage.id, sideOutputs);
    storageTracker.markReachable(vStorage.id, sideOutputs);
    if (!adamNode._sideOutputs) adamNode._sideOutputs = {};
    adamNode._sideOutputs.adamMV = sideOutputs;
    // Adam param is always input[1] — use s1.id directly (no findIndex)
    adamNode.result = createStorageHandle(adamNode.device, finalResult, s1.id);
  }
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
  options: { enableEarlyRelease?: boolean; enableVectorization?: boolean; bufferArena?: BufferArena; enableReplay?: boolean } = {},
): Promise<OptimizedExecutionResult> {
  const { enableEarlyRelease = false, enableVectorization = true, enableReplay = true } = options;

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
  // Dispatch replay cache: enabled by default for compiled plans where arena
  // stabilizes buffer identities. Disabled for non-compiled plans (backward,
  // optimizer) whose external inputs (saved-for-backward tensors, gradient
  // seeds) may not have stable buffer identities across steps.
  // Disable globally with TORCHLETTE_DISPATCH_REPLAY=0.
  const useReplayCache = enableReplay && process.env.TORCHLETTE_DISPATCH_REPLAY !== "0";

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
  if (useReplayCache && loweredPlan.dispatchCache?.valid && options.bufferArena) {
    extInputBufSet = new Set<GPUBuffer>();
    for (const node of planNodes) {
      for (const ref of node.inputs) {
        if (ref.kind === "materialized") {
          const buf = gpuBuffer(ref.storage.backendTensor);
          if (buf) extInputBufSet.add(buf);
        } else if (ref.kind === "pending" && ref.node.result) {
          const buf = gpuBuffer(ref.node.result.backendTensor);
          if (buf) extInputBufSet.add(buf);
        }
      }
    }
    if (hasArenaExternalConflicts(options.bufferArena, extInputBufSet)) {
      // Arena buffers conflict with external inputs — invalidate replay cache.
      // Normal execution path will replace the conflicting arena slots.
      loweredPlan.dispatchCache.valid = false;
    }
  }

  if (useReplayCache && loweredPlan.dispatchCache?.valid && useTopLevelSharedEncoder && options.bufferArena) {
    const cache = loweredPlan.dispatchCache;
    if (useTopLevelSharedEncoder) beginSharedEncoder();
    setActiveArena(options.bufferArena);

    // Register external input buffers for arena conflict detection
    {
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

    // Compute stats from lowered plan structure (same as normal path would)
    for (const act of loweredPlan.actions) {
      if (act.kind === "fused") {
        stats.fusedNodes += act.coveredNodeIndices.length;
        stats.fusionGroups++;
      } else if (act.kind === "sequential" || act.kind === "data-source"
                 || act.kind === "view" || act.kind === "prologue-skip") {
        stats.sequentialNodes++;
      }
    }

    // Replay timing instrumentation (controlled by env var)
    const _replayTiming = process.env.TORCHLETTE_REPLAY_TIMING === "1";
    let _tReplayDispatch = 0, _tDataSource = 0, _tView = 0, _tSequential = 0;
    let _tAdamBatch = 0, _tReclaim = 0, _tResult = 0, _tLoopOverhead = 0;
    let _nDispatches = 0, _nDataSources = 0, _nViews = 0, _nSequential = 0;
    let _nAdamNodes = 0, _nReclaims = 0, _nResults = 0;
    const _tReplayStart = _replayTiming ? performance.now() : 0;

    try {
      // Batch consecutive dispatch entries for efficient replay
      const dispatchBatch: RecordedDispatch[] = [];
      const flushDispatchBatch = () => {
        if (dispatchBatch.length > 0) {
          const _t0 = _replayTiming ? performance.now() : 0;
          replayDispatches(dispatchBatch);
          if (_replayTiming) { _tReplayDispatch += performance.now() - _t0; _nDispatches += dispatchBatch.length; }
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
            const _dsT0 = _replayTiming ? performance.now() : 0;
            // Restore arena resolve index and sequence counters to match recording position
            setArenaResolveIndexTo(entry.arenaResolveIdx);
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            const dsNode = planNodes[entry.nodeIndex];
            if (dsNode.result) { if (_replayTiming) { _tDataSource += performance.now() - _dsT0; _nDataSources++; } break; }
            const dsBackend = getBackend(dsNode.device) ?? backend;
            const dsInputs = dsNode.inputs.map(ref => getInputStorage(ref, dsBackend));
            const dsBackendInputs = dsInputs.map(s => s.backendTensor);
            const dsResult = await executeOp(dsNode, dsBackendInputs, dsBackend);
            dsNode.result = createStorageHandle(dsNode.device, dsResult);
            if (_replayTiming) { _tDataSource += performance.now() - _dsT0; _nDataSources++; }
            break;
          }
          case "view": {
            flushDispatchBatch();
            const _vT0 = _replayTiming ? performance.now() : 0;
            const vNode = planNodes[entry.nodeIndex];
            if (vNode.result) { if (_replayTiming) { _tView += performance.now() - _vT0; _nViews++; } break; }
            if (entry.cachedResult) {
              // Fast path: reconstruct from cached metadata (arena buffers stable)
              setArenaResolveIndexTo(entry.arenaResolveIdxAfter ?? entry.arenaResolveIdx);
              const cr = entry.cachedResult;
              vNode.result = createStorageHandle(vNode.device, {
                buffer: cr.buffer,
                shape: cr.shape,
                dtype: cr.dtype,
                size: cr.size,
                strides: cr.strides,
                offset: cr.offset,
                isContiguous: cr.isContiguous,
                ownsBuffer: false,
                destroy() {},
              } as BackendTensor);
            } else {
              // Slow path: re-execute (first replay before cache populated)
              setArenaResolveIndexTo(entry.arenaResolveIdx);
              const vNodeBackend = getBackend(vNode.device) ?? backend;
              const vInputs = vNode.inputs.map(ref => getInputStorage(ref, vNodeBackend));
              const vBackendInputs = vInputs.map(s => s.backendTensor);
              let vResultTensor = await executeOp(vNode, vBackendInputs, vNodeBackend);
              const vAliasedInputIdx = vBackendInputs.findIndex(b => b === vResultTensor);
              if (vAliasedInputIdx >= 0 && (vResultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
                vResultTensor = { ...vResultTensor, ownsBuffer: false } as BackendTensor;
              }
              vNode.result = createStorageHandle(vNode.device, vResultTensor);
            }
            if (_replayTiming) { _tView += performance.now() - _vT0; _nViews++; }
            break;
          }
          case "sequential": {
            flushDispatchBatch();
            const _seqT0 = _replayTiming ? performance.now() : 0;
            // Restore arena resolve index and sequence counters to match recording position
            setArenaResolveIndexTo(entry.arenaResolveIdx);
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            const node = planNodes[entry.nodeIndex];
            if (node.result) { if (_replayTiming) { _tSequential += performance.now() - _seqT0; _nSequential++; } break; }
            const nodeBackend = getBackend(node.device) ?? backend;
            const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
            const backendInputs = inputs.map(s => s.backendTensor);
            let resultTensor = await executeOp(node, backendInputs, nodeBackend);
            const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
            if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
              resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
            }
            const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
            const baseStorageId = isView && inputs.length > 0
              ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
              : undefined;
            node.result = createStorageHandle(node.device, resultTensor, baseStorageId);
            if (_replayTiming) { _tSequential += performance.now() - _seqT0; _nSequential++; }
            break;
          }
          case "adam-batch": {
            flushDispatchBatch();
            const _adamT0 = _replayTiming ? performance.now() : 0;
            flushSharedEncoder();
            flushBufferPool();
            // Set sequence counters so adam dispatches hit the correct cache positions
            setDispatchSequenceCounters(
              entry.seqCounters.dispatch,
              entry.seqCounters.params,
              entry.seqCounters.output,
            );
            setAdamBatchMode(true);
            try {
              const adamOp = backend.ops.adamStep!;
              setCurrentOpLabel("adamStep");
              const _adamBatchT0 = profileOpBegin("adamStep");
              await executeAdamBatchInner(planNodes, entry.nodeIndices, adamOp, (ref) => getInputStorage(ref, backend));
              profileOpEnd("adamStep", _adamBatchT0);
            } finally {
              setAdamBatchMode(false);
            }
            if (_replayTiming) { _tAdamBatch += performance.now() - _adamT0; _nAdamNodes += entry.nodeIndices.length; }
            break;
          }
          case "reclaim": {
            // During replay, flush encoder to submit pending dispatches but
            // skip pool reclamation — arena buffers are persistent.
            flushDispatchBatch();
            const _rclT0 = _replayTiming ? performance.now() : 0;
            flushSharedEncoder();
            if (_replayTiming) { _tReclaim += performance.now() - _rclT0; _nReclaims++; }
            break;
          }
          case "pre-adam-reclaim": {
            flushDispatchBatch();
            const _parT0 = _replayTiming ? performance.now() : 0;
            // Still need pre-adam flush for Adam's flushSharedEncoder requirement
            flushSharedEncoder();
            flushBufferPool();
            if (_replayTiming) { _tReclaim += performance.now() - _parT0; _nReclaims++; }
            break;
          }
          case "result": {
            flushDispatchBatch();
            const _rsT0 = _replayTiming ? performance.now() : 0;
            // Assign node result from cached metadata (arena buffers are stable).
            // Arena buffers persist across steps — ownsBuffer: false, no-op destroy.
            const nr = entry.nodeResult;
            const node = planNodes[nr.nodeIndex];
            if (!node.result) {
              node.result = createStorageHandle(node.device, {
                buffer: nr.buffer,
                shape: nr.shape,
                dtype: nr.dtype,
                size: nr.size,
                strides: nr.strides,
                offset: 0,
                isContiguous: true,
                ownsBuffer: false,
                destroy() {},
              } as BackendTensor);
            }
            if (_replayTiming) { _tResult += performance.now() - _rsT0; _nResults++; }
            break;
          }
          case "side-output": {
            // Restore attnLogsumexp on fusedAttentionForward nodes so a later
            // plan (backward Phase B) can skip re-executing fusedAttentionForward
            // and extractAttentionLogsumexp can read the side output directly.
            const soNode = planNodes[entry.nodeIndex];
            if (!soNode._sideOutputs?.attnLogsumexp) {
              if (!soNode._sideOutputs) soNode._sideOutputs = {};
              soNode._sideOutputs.attnLogsumexp = createStorageHandle(soNode.device, {
                buffer: entry.buffer,
                shape: entry.shape,
                dtype: entry.dtype,
                size: entry.size,
                strides: entry.strides,
                offset: 0,
                isContiguous: true,
                ownsBuffer: false,
                destroy() {},
              } as BackendTensor);
            }
            break;
          }
        }
      }
      flushDispatchBatch(); // Flush remaining batched dispatches
    } finally {
      clearActiveArena();
      clearArenaExternalInputBuffers();
      const _tEndT0 = _replayTiming ? performance.now() : 0;
      if (useTopLevelSharedEncoder) endSharedEncoder();
      const _tEnd = _replayTiming ? performance.now() - _tEndT0 : 0;
      if (_replayTiming) {
        const _tTotal = performance.now() - _tReplayStart;
        const _tAccounted = _tReplayDispatch + _tDataSource + _tView + _tSequential + _tAdamBatch + _tReclaim + _tResult + _tEnd;
        console.log(`[replay-timing] nodes=${plan.nodes.length} entries=${cache.entries.length} | total=${_tTotal.toFixed(1)}ms | dispatch=${_tReplayDispatch.toFixed(1)}ms(${_nDispatches}) dataSrc=${_tDataSource.toFixed(1)}ms(${_nDataSources}) view=${_tView.toFixed(1)}ms(${_nViews}) seq=${_tSequential.toFixed(1)}ms(${_nSequential}) adam=${_tAdamBatch.toFixed(1)}ms(${_nAdamNodes}) reclaim=${_tReclaim.toFixed(1)}ms(${_nReclaims}) result=${_tResult.toFixed(1)}ms(${_nResults}) endEnc=${_tEnd.toFixed(1)}ms | unaccounted=${(_tTotal - _tAccounted).toFixed(1)}ms`);
      }
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

  // Set up dispatch recording if we have an arena (needed for stable bind groups)
  // and no replay cache yet. We only record after the first execution (arena needs
  // to be populated first), so we check if the arena already has buffers.
  const shouldRecord = useReplayCache && useTopLevelSharedEncoder && options.bufferArena
    && !loweredPlan.dispatchCache
    && options.bufferArena.resolve.length > 0; // Arena populated from prior execution
  const recordingBuffer: RecordedDispatch[] = [];
  const replayEntries: ReplayEntry[] = [];
  let recordingDispatchIdx = 0;

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
          const groupNodes = action.coveredNodeIndices.map(i => planNodes[i]);
          const outputNode = planNodes[action.outputNodeIndex];
          const additionalOutputNodes = action.additionalOutputNodeIndices.map(i => planNodes[i]);
          const neededIntermediates = action.neededIntermediateNodeIndices.map(i => planNodes[i]);

          // Reconstruct external inputs using cached pattern (O(n) instead of O(n²))
          let extInputs: LazyRef[];
          if (action.cachedExternalInputPattern) {
            // Fast path: use pre-computed pattern
            extInputs = action.cachedExternalInputPattern.map(p =>
              groupNodes[p.nodeLocalIdx].inputs[p.inputIdx],
            );
          } else {
            // First execution: compute pattern with dedup, then cache it
            const groupNodeIds = new Set(groupNodes.map(n => n.id));
            extInputs = [];
            const pattern: Array<{ nodeLocalIdx: number; inputIdx: number }> = [];
            for (let ni = 0; ni < groupNodes.length; ni++) {
              const node = groupNodes[ni];
              for (let ii = 0; ii < node.inputs.length; ii++) {
                const inp = node.inputs[ii];
                if (inp.kind === "pending") {
                  if (!groupNodeIds.has(inp.node.id) &&
                      !extInputs.some(ei => ei.kind === "pending" && ei.node.id === inp.node.id)) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                } else if (inp.kind === "scalar") {
                  if (!extInputs.some(ei => ei.kind === "scalar" && ei.value === inp.value && ei.dtype === inp.dtype)) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                } else {
                  if (!extInputs.some(ei => ei.kind === "materialized" && ei.storage.id === inp.storage.id)) {
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
            additionalOutputNodes: additionalOutputNodes.length > 0 ? additionalOutputNodes : undefined,
            neededIntermediates: neededIntermediates.length > 0 ? neededIntermediates : undefined,
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
            while (recordingDispatchIdx < recordingBuffer.length) {
              replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
              recordingDispatchIdx++;
            }
            // Record output node result (inline for ordering)
            if (outputNode.result) {
              const bt = asGPUTensor(outputNode.result.backendTensor);
              replayEntries.push({ kind: "result", nodeResult: {
                nodeIndex: action.outputNodeIndex,
                buffer: bt.buffer,
                shape: bt.shape.slice(),
                dtype: bt.dtype,
                size: bt.size,
                strides: bt.strides.slice(),
                isView: false,
              }});
            }
            // Record additional output node results
            for (const addIdx of action.additionalOutputNodeIndices) {
              const addNode = planNodes[addIdx];
              if (addNode.result) {
                const bt = asGPUTensor(addNode.result.backendTensor);
                replayEntries.push({ kind: "result", nodeResult: {
                  nodeIndex: addIdx,
                  buffer: bt.buffer,
                  shape: bt.shape.slice(),
                  dtype: bt.dtype,
                  size: bt.size,
                  strides: bt.strides.slice(),
                  isView: false,
                }});
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
            const epilogueLabel = action.epilogueOps.length > 0
              ? "+" + action.epilogueOps.map(o => o.kind).join("+")
              : "";
            epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(/\+$/, "");
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
              const refA = planNodes[cfg.inputAPath.planNodeIndex].inputs[cfg.inputAPath.inputIndex];
              const refB = planNodes[cfg.inputBPath.planNodeIndex].inputs[cfg.inputBPath.inputIndex];
              const bufA = gpuBuffer(getInputStorage(refA).backendTensor);
              const bufB = gpuBuffer(getInputStorage(refB).backendTensor);
              const epilogueBuffers: GPUBuffer[] = [];
              for (const path of cfg.epilogueInputPaths) {
                const ref = planNodes[path.planNodeIndex].inputs[path.inputIndex];
                epilogueBuffers.push(gpuBuffer(getInputStorage(ref).backendTensor));
              }

              // Dispatch directly — skips shape computation, transpose detection,
              // contiguous checks, prologue resolution, dynamic imports
              const resultTensor = _webgpuMatmulImports!.dispatchMatmulDirect(
                bufA, bufB, {
                  m: cfg.m, n: cfg.n, k: cfg.k,
                  transA: cfg.transA, transB: cfg.transB,
                  batchSize: cfg.batchSize,
                  batchStrideA: cfg.batchStrideA,
                  batchStrideB: cfg.batchStrideB,
                  batchStrideC: cfg.batchStrideC,
                  outShape: cfg.outShape,
                  dtypeA: cfg.dtypeA, dtypeB: cfg.dtypeB,
                  outputDtype: cfg.outputDtype,
                  epilogueConfig: cfg.epilogueConfig,
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
                const newStrides = new Array(fastOutShape.length);
                let stride = 1;
                for (let i = fastOutShape.length - 1; i >= 0; i--) {
                  newStrides[i] = stride;
                  stride *= fastOutShape[i];
                }
                gpuT.strides = newStrides;
              }
              outputNode.result = createStorageHandle(outputNode.device, resultTensor);
            } finally {
              profileOpEnd(epLabel, _profT0);
              setCurrentOpLabel(null);
              setProfileModule("unknown");
            }

            // Record dispatches and node results (for replay cache)
            if (shouldRecord) {
              while (recordingDispatchIdx < recordingBuffer.length) {
                replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
                recordingDispatchIdx++;
              }
              if (outputNode.result) {
                const bt = asGPUTensor(outputNode.result.backendTensor);
                replayEntries.push({ kind: "result", nodeResult: {
                  nodeIndex: action.outputNodeIndex,
                  buffer: bt.buffer,
                  shape: bt.shape.slice(),
                  dtype: bt.dtype,
                  size: bt.size,
                  strides: bt.strides.slice(),
                  isView: false,
                }});
              }
            }
            break;
          }

          // ── SLOW PATH: first lowered plan execution — full reconstruction ──

          // Reconstruct the epilogue input refs from the covered nodes.
          // For each add/mul in the chain, the "external" input (not from the chain)
          // is the epilogue input. The chain may connect via inputs[0] or inputs[1]
          // (commutative ops), so we check which input is the previous chain node.
          const epilogueInputRefs: LazyRef[] = [];
          const epilogueInputPaths: Array<{ planNodeIndex: number; inputIndex: number }> = [];
          for (let ci = 1; ci < action.coveredNodeIndices.length; ci++) {
            const chainNode = planNodes[action.coveredNodeIndices[ci]];
            if ((chainNode.op === "add" || chainNode.op === "mul") && chainNode.inputs.length === 2) {
              const prevChainNodeId = planNodes[action.coveredNodeIndices[ci - 1]].id;
              const inp0IsChain = chainNode.inputs[0].kind === "pending"
                && chainNode.inputs[0].node.id === prevChainNodeId;
              const externalIdx = inp0IsChain ? 1 : 0;
              epilogueInputRefs.push(chainNode.inputs[externalIdx]);
              epilogueInputPaths.push({ planNodeIndex: action.coveredNodeIndices[ci], inputIndex: externalIdx });
            }
          }

          // Reconstruct prologues and resolve prologue decisions
          let prologues: MatmulPrologueInfo[] | undefined;
          let inputCastA: DType | undefined;
          let inputCastB: DType | undefined;
          let resolvedInputRefA = matmulNode.inputs[0];
          let resolvedInputRefB = matmulNode.inputs[1];
          // Track paths for caching (plan-node-relative, stable across steps)
          let inputAPath = { planNodeIndex: action.matmulNodeIndex, inputIndex: 0 };
          let inputBPath = { planNodeIndex: action.matmulNodeIndex, inputIndex: 1 };

          if (action.prologues && action.prologues.length > 0) {
            prologues = action.prologues.map(p => ({
              inputIndex: p.inputIndex,
              castNodeId: planNodes[p.castNodeIndex].id,
              originalInputRef: planNodes[p.castNodeIndex].inputs[0],
              fromDtype: p.fromDtype,
              toDtype: p.toDtype,
            }));

            // Resolve prologue decisions (same logic as executeMatmulWithEpilogue)
            for (const p of action.prologues) {
              const castRef = p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
              const castAlreadyRan = castRef.kind === "pending" && castRef.node.result != null;
              if (!castAlreadyRan) {
                if (p.inputIndex === 0) {
                  resolvedInputRefA = planNodes[p.castNodeIndex].inputs[0];
                  inputCastA = p.toDtype;
                  inputAPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
                } else {
                  resolvedInputRefB = planNodes[p.castNodeIndex].inputs[0];
                  inputCastB = p.toDtype;
                  inputBPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
                }
              }
            }
          }

          const epiloguePlan: MatmulEpiloguePlan = {
            consumedCount: action.consumedCount,
            epilogueOps: action.epilogueOps,
            epilogueInputRefs,
            outputDtype: action.outputDtype,
            outputNode,
            prologues,
          };

          setCurrentOpLabel(epLabel);
          setProfileModule(matmulNode.module ?? "unknown");
          const _profT0 = profileOpBegin(epLabel);
          try {
            await executeMatmulWithEpilogue(matmulNode, epiloguePlan, backend);
          } finally {
            profileOpEnd(epLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // ── Cache dispatch config for next step ──
          // Compute matmul geometry from the resolved inputs (shapes are stable across steps).
          // Must replicate dispatchMatmulWithEpilogue's transpose detection logic exactly.
          {
            const { computeMatmulOutputShape, computeBatchSize, computeBatchStrides } = _webgpuMatmulGeomImports!;
            const { isF16Supported } = await import("../backend/webgpu/index");

            const tensorA = asGPUTensor(getInputStorage(resolvedInputRefA).backendTensor);
            const tensorB = asGPUTensor(getInputStorage(resolvedInputRefB).backendTensor);

            // Detect simple last-2-dim transposes (matching detectSimpleTranspose in index.ts).
            // If detected, use original contiguous shape and flip transpose flag.
            const detA = _detectTransposeView(tensorA);
            const detB = _detectTransposeView(tensorB);
            const effectiveShapeA = detA.shape;
            const effectiveShapeB = detB.shape;
            const transA = detA.transposed;
            const transB = detB.transposed;

            const outShape = computeMatmulOutputShape(effectiveShapeA, effectiveShapeB, transA, transB);
            const aRank = effectiveShapeA.length;
            const bRank = effectiveShapeB.length;
            const m = transA ? effectiveShapeA[aRank - 1] : effectiveShapeA[aRank - 2];
            const k = transA ? effectiveShapeA[aRank - 2] : effectiveShapeA[aRank - 1];
            const n = transB ? effectiveShapeB[bRank - 2] : effectiveShapeB[bRank - 1];

            const batchDims = outShape.slice(0, -2);
            const batchSize = computeBatchSize(batchDims);
            const { strideA, strideB, strideC } = computeBatchStrides(effectiveShapeA, effectiveShapeB, batchDims, m, n, k);

            // Compute dtypes (matching dispatchMatmulWithEpilogue logic)
            const f16ok = isF16Supported();
            const rawDtypeA = tensorA.dtype === "f16" && f16ok ? "f16" as const : "f32" as const;
            const rawDtypeB = tensorB.dtype === "f16" && f16ok ? "f16" as const : "f32" as const;
            const dtypeA: "f16" | "f32" = inputCastA === "f16" && f16ok ? "f16" : rawDtypeA;
            const dtypeB: "f16" | "f32" = inputCastB === "f16" && f16ok ? "f16" : rawDtypeB;
            const promotedDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
            const outputDtype = action.outputDtype ?? promotedDtype;

            const epilogueConfig = {
              ops: action.epilogueOps,
              additionalInputCount: epilogueInputRefs.length,
              outputDtype: action.outputDtype,
            };

            action.cachedDispatchConfig = {
              inputAPath,
              inputBPath,
              epilogueInputPaths,
              inputCastA,
              inputCastB,
              m, k, n,
              transA, transB,
              batchSize,
              batchStrideA: strideA,
              batchStrideB: strideB,
              batchStrideC: strideC,
              outShape,
              dtypeA,
              dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
              outputDtype,
              epilogueConfig,
            };
          }

          // Record dispatches and node results
          if (shouldRecord) {
            while (recordingDispatchIdx < recordingBuffer.length) {
              replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
              recordingDispatchIdx++;
            }
            if (outputNode.result) {
              const bt = asGPUTensor(outputNode.result.backendTensor);
              replayEntries.push({ kind: "result", nodeResult: {
                nodeIndex: action.outputNodeIndex,
                buffer: bt.buffer,
                shape: bt.shape.slice(),
                dtype: bt.dtype,
                size: bt.size,
                strides: bt.strides.slice(),
                isView: false,
              }});
            }
          }
          break;
        }

        case "reduction-preamble": {
          const preambleNode = planNodes[action.preambleNodeIndex];
          const reductionNode = planNodes[action.reductionNodeIndex];
          const reductionPlan: ReductionPreamblePlan = {
            preambleNode,
            reductionNode,
            op: preambleNode.op,
            arity: preambleNode.inputs.length,
            isMean: reductionNode.op === "mean",
          };
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.op}`;
          setCurrentOpLabel(rpLabel);
          setProfileModule(preambleNode.module ?? "unknown");
          const _profT0 = profileOpBegin(rpLabel);
          try {
            await executeReductionWithPreamble(reductionPlan, backend);
          } finally {
            profileOpEnd(rpLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // Record dispatches and node results
          if (shouldRecord) {
            while (recordingDispatchIdx < recordingBuffer.length) {
              replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
              recordingDispatchIdx++;
            }
            // Record reduction node result
            if (reductionNode.result) {
              const bt = asGPUTensor(reductionNode.result.backendTensor);
              replayEntries.push({ kind: "result", nodeResult: {
                nodeIndex: action.reductionNodeIndex,
                buffer: bt.buffer,
                shape: bt.shape.slice(),
                dtype: bt.dtype,
                size: bt.size,
                strides: bt.strides.slice(),
                isView: false,
              }});
            }
          }
          break;
        }

        case "adam-batch": {
          const useSharedEncoder = backend.name === "webgpu";
          if (useSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }

          // Record pre-adam reclaim and capture counter positions
          let adamSeqCountersBefore: { dispatch: number; params: number; output: number } | undefined;
          if (shouldRecord && useSharedEncoder) {
            replayEntries.push({ kind: "pre-adam-reclaim" });
            adamSeqCountersBefore = getDispatchSequenceCounters();
          }

          setAdamBatchMode(true);
          try {
            const adamBackend = getBackend(planNodes[action.nodeIndices[0]].device) ?? backend;
            const adamOp = adamBackend.ops.adamStep!;
            setCurrentOpLabel("adamStep");
            setProfileModule("optimizer.step");
            const _adamBatchT0 = profileOpBegin("adamStep");
            await executeAdamBatchInner(planNodes, action.nodeIndices, adamOp, (ref) => getInputStorage(ref, adamBackend));
            profileOpEnd("adamStep", _adamBatchT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          } finally {
            setAdamBatchMode(false);
          }

          // Record adam batch as a single replay entry (must re-execute each step).
          // Capture the sequence counter positions so adam dispatches hit the correct
          // cache positions during replay.
          if (shouldRecord) {
            while (recordingDispatchIdx < recordingBuffer.length) {
              recordingDispatchIdx++;
            }
            replayEntries.push({
              kind: "adam-batch",
              nodeIndices: action.nodeIndices,
              seqCounters: adamSeqCountersBefore!,
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
          const arenaResolveIdxBefore = shouldRecord ? getArenaResolveIndex() : 0;
          const seqCountersBefore = shouldRecord ? getDispatchSequenceCounters() : undefined;

          const nodeBackend = getBackend(node.device) ?? backend;
          const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
          const backendInputs = inputs.map(s => s.backendTensor);

          let resultTensor = await executeOp(node, backendInputs, nodeBackend);
          const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
          if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
            resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
          }
          const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
          const baseStorageId = isView && inputs.length > 0
            ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
            : undefined;
          node.result = createStorageHandle(node.device, resultTensor, baseStorageId);
          stats.sequentialNodes++;

          // Record for replay cache
          if (shouldRecord) {
            if (action.kind === "data-source") {
              // Data sources must re-execute each step (host data changes).
              replayEntries.push({ kind: "data-source", nodeIndex: nodeIdx, arenaResolveIdx: arenaResolveIdxBefore, seqCounters: seqCountersBefore! });
              while (recordingDispatchIdx < recordingBuffer.length) {
                recordingDispatchIdx++;
              }
            } else if (action.kind === "view") {
              // Views produce deterministic results (same arena buffer, same
              // shape/strides/offset). Cache the result to skip re-execution.
              const bt = asGPUTensor(node.result!.backendTensor);
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
              while (recordingDispatchIdx < recordingBuffer.length) {
                recordingDispatchIdx++;
              }
            } else if (ENCODER_COPY_OPS.has(node.op)) {
              // Ops that use encoder copy commands (copyBufferToBuffer) alongside
              // compute dispatches must be re-executed during replay — the copy
              // commands are invisible to the compute dispatch recording mechanism.
              replayEntries.push({ kind: "sequential", nodeIndex: nodeIdx, arenaResolveIdx: arenaResolveIdxBefore, seqCounters: seqCountersBefore! });
              while (recordingDispatchIdx < recordingBuffer.length) {
                recordingDispatchIdx++;
              }
            } else {
              // With arena active, all buffer identities are stable (arena returns
              // same GPUBuffer object on subsequent steps). Data-source-consuming
              // ops can be safely replayed — their bind groups reference arena
              // buffers that don't change identity.
              // Safe to replay: capture dispatches
              while (recordingDispatchIdx < recordingBuffer.length) {
                replayEntries.push({ kind: "dispatch", dispatch: recordingBuffer[recordingDispatchIdx] });
                recordingDispatchIdx++;
              }
              // Record node result (inline for ordering)
              if (node.result) {
                const bt = asGPUTensor(node.result.backendTensor);
                replayEntries.push({ kind: "result", nodeResult: {
                  nodeIndex: nodeIdx,
                  buffer: bt.buffer,
                  shape: bt.shape.slice(),
                  dtype: bt.dtype,
                  size: bt.size,
                  strides: bt.strides.slice(),
                  isView: isView,
                }});
              }
              // Record side output for fusedAttentionForward so replay
              // restores attnLogsumexp (needed by extractAttentionLogsumexp
              // in a later plan that skips re-executing this node).
              if (node._sideOutputs?.attnLogsumexp) {
                const soSH = node._sideOutputs.attnLogsumexp;
                const sobt = asGPUTensor(soSH.backendTensor);
                replayEntries.push({ kind: "side-output", nodeIndex: nodeIdx,
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
      const pinnedBuffers = new Set<any>();
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
