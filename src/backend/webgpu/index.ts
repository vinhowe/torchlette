// ============================================================================
// Re-exports from extracted utility modules
// ============================================================================
export {
  sizeOf,
  gcd,
  lcm,
  broadcastShapes,
  toIndexShape,
  contiguousStrides,
  broadcastStrides,
  computeEffectiveBroadcastStrides,
  wgslArray,
  buildBroadcastIndexing,
  checkContiguousStrides,
  shapesEqual,
  dtypeBytes,
  alignBufferSize,
  dtypeToWgsl,
  compute2DDispatch,
  WORKGROUP_SIZE,
  MAX_WORKGROUPS_PER_DIM,
} from "./shape-utils";

export type { WebGPUTensor, WebGPUContext } from "./gpu-types";
export { GPUBufferUsage, GPUMapMode, STORAGE_BUFFER_USAGE, MATMUL_WORKGROUP_X, MATMUL_WORKGROUP_Y } from "./gpu-types";

export {
  bufferPool, getBufferPoolStats, setBufferPoolEnabled, clearBufferPool,
  setBufferPoolBudget, getBufferPoolBudget, flushBufferPool, flushBufferPoolWithSync,
  decRefBuffer, issueDeferredFence, awaitDeferredFence, hasDeferredFence,
  getBufferPoolDetailedStats, resetBufferPoolDetailedStats, setBufferPoolDebugTrace,
  deferredDestroyBuffer, acquirePooledBuffer,
} from "./buffer-pool";

export {
  initWebGPU, destroyWebGPU, syncWebGPU, getWebGPUDevice, isF16Supported,
  getWebGPUInitError, getMaxStorageBufferBindingSize, setF16WeightCacheEntry,
  evictF16WeightCacheEntry, f32ToF16, f16ToF32, f32ArrayToF16Array, f16ArrayToF32Array,
  resetAllKernelCaches,
} from "./gpu-context";

export type { BatchExecutionContext } from "./shared-encoder";
export {
  beginBatchExecution, endBatchExecution, isBatchActive, abortBatch,
  getActiveBatchEncoder, beginSharedEncoder, flushSharedEncoder, endSharedEncoder,
  beginStep, endStep, isSharedEncoderActive, setSharedEncoderEnabled,
  getSharedEncoderInstance, trackSharedEncoderWrite, isInSharedEncoderWriteSet,
  getSubmitCount, resetSubmitCount, submitOrCollect,
  setCurrentOpLabel, getCurrentOpLabel, setAdamBatchMode,
} from "./shared-encoder";

export type { RecordedDispatch } from "./dispatch-recording";
export {
  replayPinnedBufferSet, getAndClearLastBindGroupBuffers,
  startDispatchRecording, stopDispatchRecording, replayDispatches,
  addReplayPinnedBuffers,
} from "./dispatch-recording";

export type { BufferArena } from "./buffer-arena";
export {
  arenaBufferSet, pinnedOutputBuffers,
  allocateOutputBuffer, donateBuffer, getBufferSize,
  setArenaExternalInputBuffers, clearArenaExternalInputBuffers,
  getArenaConflictDetected, clearArenaConflictDetected,
  hasArenaExternalConflicts, setActiveArena, clearActiveArena,
  getArenaResolveIndex, setArenaResolveIndexTo, isArenaBuffer, classifyBuffer,
  destroyArena, arenaAllocAt, prePinOutputBuffers, resolveOutputBuffer,
  getArenaResolveStats,
} from "./buffer-arena";

export {
  paramsBufferPools, MAX_PARAMS_POOL_SIZE_PER_CLASS, paramsBufferSizeClass,
  getParamsArray, createParamsBuffer, releaseParamsBuffer,
  createUniformBuffer, releaseUniformBuffer,
  profiledCreateBindGroup, cachedCreateBindGroup,
  resetDispatchSequence, setDispatchSequenceCounters, getDispatchSequenceCounters,
  clearBindGroupCache, getBindGroupCacheStats, resetBindGroupCacheStats,
  getBindGroupCacheMissLog, getSequenceEntryBuffers,
} from "./bind-group-cache";

export { createTensor, createTrackedBuffer, createBufferWithData } from "./tensor";

export {
  dispatchComputePass, dispatchElementwise,
  binaryBroadcastShader, unaryStridedShader,
  getPipeline,
  dispatchBinary, dispatchBinaryDirect, dispatchBinaryChunked,
  dispatchUnary, dispatchUnaryDirect, dispatchUnaryChunked,
  dispatchMatmul,
  dispatchMatmulWithEpilogue, dispatchMatmulDirect,
  runFusedElementwise,
} from "./dispatch";

// Re-export memory tracking functions
export {
  gpuMemoryTracker,
  GPUMemoryLimitExceededError,
  setGPUMemoryLimit,
  getGPUMemoryLimit,
  getGPUMemoryStats,
  getGPUAllocationHistogram,
  enableLargeAllocDebug,
  getLargeAllocLog,
  clearLargeAllocLog,
  enableAllAllocDebug,
  disableAllAllocDebug,
  clearAllocStacks,
  setAllocStep,
  snapshotLeakedAllocs,
  snapshotLeakedAllocsForStep,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  getAndResetFlowCounters,
  getTrackedBuffers,
  getLeakedSizeHistogramForStep,
} from "./memory-tracker";

// Re-export profiler functions for use in tests
export {
  setProfilePhase,
  setProfileModule,
  getProfileModule,
  readGpuTimestamps,
  flushAndReadGpuTimestamps,
  printProfileSummary,
  resetProfileStats,
  isProfilingEnabled,
  setTimestampsEnabled,
  getProfileJSON,
  writeProfileJSON,
  recordPlanAnalysis,
  type PlanAnalysis,
} from "./profiler";

// Re-export autotune control functions
export { isAutotuneEnabled, pretuneMatmulShapes, setAutotuneEnabled } from "./matmul";

// Re-export public functions from ops modules
export { tensorFromArrayWithDtype } from "./ops/creation";
export { sumDimWithPreamble } from "./ops/reductions";
export { waitForGPU } from "./ops/fused";
export { detectSimpleTranspose, ensureContiguous } from "./ops/views";

// ============================================================================
// Imports used locally (for webgpuBackend object and remaining wiring)
// ============================================================================
import type { Backend } from "../types";
import type { GPUDevice } from "./gpu-types";
import { gpuContext } from "./webgpu-state";
import { registerBackend } from "../registry";
import { pretuneMatmulShapes as pretuneShapes } from "./matmul";
import { dispatchFusedKernel } from "./fusion-dispatch";
import { beginStep, endStep } from "./shared-encoder";

// ============================================================================
// Ops imports (used in webgpuBackend.ops)
// ============================================================================
import { gt, lt, ge, le, eq, ne, argmax, argmin } from "./ops/comparison";
import { where } from "./ops/where";
import { tensorFromArray, zeros, full, arange, tril, triu, rand, randn, bernoulli, _setContiguous } from "./ops/creation";
import { add, sub, div, mul, sqrt, relu, exp, log, neg, abs, tanh, sigmoid, gelu, silu, isfinite } from "./ops/elementwise";
import { cast, reshape, expand, contiguous, narrow, narrowBackward, transpose, permute } from "./ops/views";
import { matmul } from "./ops/matmul-ops";
import { gather, scatterAdd } from "./ops/gather-scatter";
import { sum, max, mean } from "./ops/reductions";
import { stridedScatterCopy, stridedScatterAdd } from "./ops/strided-scatter";
import { adamStep, unscaleGrad, createInfCountBuffer, readAndDestroyInfCount, fusedCrossEntropyForward, fusedCrossEntropyBackward, fusedLayerNormForward, fusedLayerNormBackwardGradX, fusedLayerNormBackwardGradWeightBias, fusedAttentionForward, fusedAttentionBackward, read, waitForGPU, mulScalarInPlace } from "./ops/fused";

// Wire up creation.ts contiguous injection callback (the sole remaining callback —
// all other injection callbacks were eliminated by the webgpu-state.ts refactoring).
_setContiguous(contiguous);

// ============================================================================
// webgpuBackend object
// ============================================================================
export const webgpuBackend: Backend & {
  waitForGPU: typeof waitForGPU;
  mulScalarInPlace: typeof mulScalarInPlace;
  dispatchFusedKernel: typeof dispatchFusedKernel;
  beginStep: typeof beginStep;
  endStep: typeof endStep;
  device: GPUDevice | null;
} = {
  name: "webgpu",
  waitForGPU,
  // Expose device for fusion dispatch (§15)
  get device() {
    return gpuContext?.device ?? null;
  },
  // Fusion dispatch (§15.1, §15.2, §15.3)
  dispatchFusedKernel,
  ops: {
    tensorFromArray,
    zeros,
    full,
    arange,
    tril,
    triu,
    rand,
    randn,
    bernoulli,
    add,
    sub,
    div,
    mul,
    matmul,
    sqrt,
    relu,
    exp,
    log,
    neg,
    abs,
    tanh,
    sigmoid,
    gelu,
    silu,
    isfinite,
    expand,
    reshape,
    transpose,
    permute,
    narrow,
    narrowBackward,
    contiguous,
    cast,
    gather,
    scatterAdd,
    sum,
    max,
    mean,
    argmax,
    argmin,
    gt,
    lt,
    ge,
    le,
    eq,
    ne,
    where,
    stridedScatterCopy,
    stridedScatterAdd,
    adamStep,
    unscaleGrad,
    fusedAttentionForward,
    fusedAttentionBackward,
    fusedCrossEntropyForward,
    fusedCrossEntropyBackward,
    fusedLayerNormForward,
    fusedLayerNormBackwardGradX,
    fusedLayerNormBackwardGradWeightBias,
    createInfCountBuffer,
    readAndDestroyInfCount,
    read,
  },
  mulScalarInPlace,
  beginStep,
  endStep,
  // Pretune matmul shapes for autotuning (used by compile with autotune: true)
  async pretuneMatmulShapes(shapes: Array<[number, number, number]>): Promise<void> {
    const ctx = gpuContext;
    if (!ctx) {
      return; // WebGPU not initialized
    }
    await pretuneShapes(ctx.device, ctx.queue, shapes, "f32");
  },
};

// Register the WebGPU backend with the backend registry at module load time.
// Previously this was inside initWebGPU() in gpu-context.ts, which created a
// circular dependency (gpu-context → index). registerBackend() just adds to a
// map — it's safe to call before initWebGPU(). The backend's ops all require
// an initialized GPU context (via requireContext()), so calling them before
// initWebGPU() still produces a clear error.
registerBackend(webgpuBackend);
