// ============================================================================
// Re-exports from extracted utility modules
// ============================================================================

export {
  getDispatchSequenceCounters,
  setDispatchSequenceCounters,
} from "./bind-group-cache";
export type { BufferArena } from "./buffer-arena";
export {
  allocateOutputBuffer,
  arenaLivenessEnabled,
  clearActiveArena,
  clearArenaConflictDetected,
  clearArenaExternalInputBuffers,
  destroyArena,
  donateBuffer,
  getArenaConflictDetected,
  getArenaResolveIndex,
  getBufferSize,
  isArenaBuffer,
  resolveOutputBuffer,
  setActiveArena,
  setArenaExternalInputBuffers,
  setArenaResolveIndexTo,
} from "./buffer-arena";

export {
  clearBufferPool,
  deferredDestroyBuffer,
  destroyPendingGPUBuffers,
  evictAllPoolBuffers,
  flushBufferPool,
  getBufferPoolDetailedStats,
  getBufferPoolStats,
  resetBufferPoolDetailedStats,
} from "./buffer-pool";
export {
  dispatchMatmul,
  dispatchMatmulDirect,
  getPipeline,
} from "./dispatch";
export {
  destroyWebGPU,
  getMaxStorageBufferBindingSize,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  isF16Supported,
  syncWebGPU,
  warmupFromRegistry,
  warmupFromStep,
} from "./gpu-context";
export type { WebGPUTensor } from "./gpu-types";
// Re-export autotune control functions
export {
  isAutotuneEnabled,
  setAutotuneEnabled,
} from "./matmul/dispatch";
// Re-export memory tracking functions
export {
  enableAllAllocDebug,
  getAndResetFlowCounters,
  getGPUAllocationHistogram,
  getGPUMemoryStats,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  setAllocStep,
  setGPUMemoryLimit,
  snapshotLeakedAllocs,
} from "./memory-tracker";
// Re-export public functions from ops modules
export { tensorFromArrayWithDtype } from "./ops/creation";
// Re-export reduction fusion functions (used by engine/reduction-preamble.ts via dynamic import)
export {
  meanWithEpilogue,
  reduction,
  sumDimWithPreambleChain,
  sumWithPreambleEpilogue,
} from "./ops/reductions";
export {
  clearWarmupCache,
  deserializeRegistry,
  serializeRegistry,
  startPipelineRecording,
  stopPipelineRecording,
  warmupPipelines,
} from "./pipeline-warmup";
// Re-export profiler functions for use in tests and tools
export {
  disableProfiling,
  enableProfiling,
  getProfileJSON,
  initGpuTimestamps,
  isProfilingEnabled,
  printProfileSummary,
  readGpuTimestamps,
  resetProfileStats,
  setProfileModule,
  setProfilePhase,
  setTimestampsEnabled,
  writeProfileJSON,
} from "./profiler";
export {
  abortBatch,
  beginBatchExecution,
  beginSharedEncoder,
  endBatchExecution,
  endSharedEncoder,
  flushSharedEncoder,
  getCurrentOpLabel,
  getSubmitCount,
  isBatchActive,
  resetSubmitCount,
  setCurrentOpLabel,
} from "./shared-encoder";
export { compileTileKernel } from "./tile-compiler";
export {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./tile-dispatch";
// Re-export tile-IR public API for custom kernels
export {
  type BindingSpec,
  ceilDivGrid,
  type DataType,
  elementwiseGrid,
  KernelContext,
  singleWorkgroup,
  type TileKernelSpec,
  type UniformType,
} from "./tile-ir";

import { registerBackend } from "../registry";
// ============================================================================
// Imports used locally (for webgpuBackend object and remaining wiring)
// ============================================================================
import type { FusedBackend } from "../types";
import {
  clearActiveArena,
  clearArenaExternalInputBuffers,
  setActiveArena,
  setArenaExternalInputBuffers,
} from "./buffer-arena";
import { flushBufferPool } from "./buffer-pool";
import { dispatchFusedKernel } from "./fusion-dispatch";
import type { GPUDevice } from "./gpu-types";
import { pretuneMatmulShapes as pretuneShapes } from "./matmul/dispatch";
// ============================================================================
// Ops imports (used in webgpuBackend.ops)
// ============================================================================
import { argmax, argmin, eq, ge, gt, le, lt, ne } from "./ops/comparison";
import { conv2d } from "./ops/conv2d";
import {
  _setContiguous,
  arange,
  bernoulli,
  full,
  rand,
  randn,
  tensorFromArray,
  tril,
  triu,
  zeros,
} from "./ops/creation";
import {
  abs,
  add,
  ceil,
  clamp,
  cos,
  div,
  exp,
  floor,
  gelu,
  isfinite,
  log,
  maximum,
  minimum,
  mul,
  neg,
  pow,
  relu,
  round,
  rsqrt,
  sigmoid,
  sign,
  silu,
  sin,
  sqrt,
  sub,
  tanh,
} from "./ops/elementwise";
import {
  adamStep,
  adamStepBatch,
  createInfCountBuffer,
  fusedAttentionBackward,
  fusedAttentionForward,
  fusedCrossEntropyBackward,
  fusedCrossEntropyForward,
  fusedLayerNormBackwardGradWeightBias,
  fusedLayerNormBackwardGradX,
  fusedLayerNormForward,
  fusedRMSNormBackwardGradWeight,
  fusedRMSNormBackwardGradX,
  fusedRMSNormForward,
  fusedRoPE,
  mulScalarInPlace,
  read,
  readAndDestroyInfCount,
  startScalarReadback,
  unscaleGrad,
  waitForGPU,
} from "./ops/fused";
import { cat, gather, scatterAdd } from "./ops/gather-scatter";
import { matmul } from "./ops/matmul-ops";
import { batchedReduction, max, mean, min, sum } from "./ops/reductions";
import { stridedScatterAdd, stridedScatterCopy } from "./ops/strided-scatter";
import {
  cast,
  contiguous,
  expand,
  narrow,
  narrowBackward,
  permute,
  reshape,
  transpose,
} from "./ops/views";
import { where } from "./ops/where";
import {
  abortBatch,
  beginBatchExecution,
  beginSharedEncoder,
  beginStep,
  endBatchExecution,
  endSharedEncoder,
  endStep,
  flushSharedEncoder,
  isBatchActive,
} from "./shared-encoder";
import { gpuContext } from "./webgpu-state";

// Wire up creation.ts contiguous injection callback (the sole remaining callback —
// all other injection callbacks were eliminated by the webgpu-state.ts refactoring).
// eslint-disable-next-line @typescript-eslint/no-explicit-any
_setContiguous(contiguous as any);

// ============================================================================
// webgpuBackend object
// ============================================================================
export const webgpuBackend: FusedBackend & {
  waitForGPU: typeof waitForGPU;
  mulScalarInPlace: typeof mulScalarInPlace;
  dispatchFusedKernel: typeof dispatchFusedKernel;
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

  // FusedBackend: shared encoder lifecycle
  beginSharedEncoder,
  endSharedEncoder,
  flushSharedEncoder,
  flushBufferPool,

  // FusedBackend: buffer arena management
  setActiveArena,
  clearActiveArena,
  setArenaExternalInputBuffers,
  clearArenaExternalInputBuffers,

  // FusedBackend: batch execution (checkpoint segmentation)
  beginBatchExecution,
  endBatchExecution,
  isBatchActive,
  abortBatch,
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
    minimum,
    maximum,
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
    sin,
    cos,
    rsqrt,
    floor,
    ceil,
    round,
    sign,
    clamp,
    pow,
    isfinite,
    expand,
    reshape,
    transpose,
    permute,
    narrow,
    narrowBackward,
    contiguous,
    cast,
    conv2d,
    gather,
    scatterAdd,
    cat,
    sum,
    max,
    min,
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
    batchedReduction,
    adamStep,
    adamStepBatch,
    unscaleGrad,
    fusedAttentionForward,
    fusedAttentionBackward,
    fusedCrossEntropyForward,
    fusedCrossEntropyBackward,
    fusedLayerNormForward,
    fusedLayerNormBackwardGradX,
    fusedLayerNormBackwardGradWeightBias,
    fusedRMSNormForward,
    fusedRMSNormBackwardGradX,
    fusedRMSNormBackwardGradWeight,
    fusedRoPE,
    createInfCountBuffer,
    readAndDestroyInfCount,
    read,
    startScalarReadback,
  },
  mulScalarInPlace,
  beginStep,
  endStep,
  // Pretune matmul shapes for autotuning (used by compile with autotune: true)
  async pretuneMatmulShapes(
    shapes: Array<[number, number, number]>,
  ): Promise<void> {
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
