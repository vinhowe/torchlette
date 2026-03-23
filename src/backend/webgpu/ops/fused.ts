/**
 * Fused kernel wrappers and I/O ops: adamStep, unscaleGrad, fusedCrossEntropy,
 * fusedLayerNorm, fusedAttention, read, waitForGPU, mulScalarInPlace.
 */
import type { BackendTensor } from "../../types";
import { allocateOutputBuffer } from "../buffer-arena";
import { bufferPool, destroyCopy } from "../buffer-pool";
import {
  f16ArrayToF32Array,
  f16WeightCache,
  requireContext,
} from "../gpu-context";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { asGPUTensor, GPUBufferUsage, GPUMapMode } from "../gpu-types";
import { gpuMemoryTracker } from "../memory-tracker";
import {
  profileApiCall,
  profileSubOpBegin,
  profileSubOpEnd,
} from "../profiler";
import { alignBufferSize, dtypeBytes, WORKGROUP_SIZE } from "../shape-utils";
import {
  flushSharedEncoder,
  getSharedEncoderInstance,
  incrementSubmitCount,
  isAdamBatchMode,
  isSharedEncoderActive,
  trackSharedEncoderWrite,
} from "../shared-encoder";
import { createTensor, createTrackedBuffer } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";
import type { TileKernelSpec } from "../tile-ir";
import { elementwiseGrid } from "../tile-ir";
import { gpuContext, sharedEncoderActive } from "../webgpu-state";
import { asContiguous, ensureContiguous } from "./views";

/** Destroy contiguous copies that differ from their originals. */
function cleanupContiguous(...pairs: [BackendTensor, WebGPUTensor][]) {
  for (const [orig, copy] of pairs) {
    if (copy !== orig) destroyCopy(copy);
  }
}

import { dispatchAdamStep as dispatchAdamStepKernel } from "../adam-kernel";
import {
  dispatchFlashAttentionBackwardD as dispatchFABwdDKernel,
  dispatchFlashAttentionBackwardDKV as dispatchFABwdDKVKernel,
  dispatchFlashAttentionBackwardDQ as dispatchFABwdDQKernel,
  dispatchFlashAttentionForward as dispatchFAForwardKernel,
} from "../attention-kernel";
import {
  dispatchCrossEntropyBackward as dispatchCEBackwardKernel,
  dispatchCrossEntropyForward as dispatchCEForwardKernel,
} from "../cross-entropy-kernel";
import {
  dispatchLayerNormBackwardGradWeightBias as dispatchLNBwdGradWBKernel,
  dispatchLayerNormBackwardGradX as dispatchLNBwdGradXKernel,
  dispatchLayerNormForward as dispatchLNForwardKernel,
} from "../layernorm-kernel";
import {
  dispatchRMSNormBackwardGradWeight as dispatchRMSBwdGradWKernel,
  dispatchRMSNormBackwardGradX as dispatchRMSBwdGradXKernel,
  dispatchRMSNormForward as dispatchRMSForwardKernel,
} from "../rmsnorm-kernel";
import {
  allocateInfFlagBuffer,
  dispatchUnscaleGrad as dispatchUnscaleGradKernel,
  readInfFlag,
} from "../unscale-kernel";

// ============================================================================
// Fused Adam/AdamW Step
// ============================================================================

export async function adamStep(
  grad: BackendTensor,
  param: BackendTensor,
  m: BackendTensor,
  v: BackendTensor,
  config: import("../../types").AdamStepConfig,
): Promise<{ param: BackendTensor; m: BackendTensor; v: BackendTensor }> {
  // Suppress memory limit checks for the entire adam step.
  // During optimizer steps, old param/m/v buffers haven't been released yet
  // while new output buffers are allocated — temporary 2x peak is expected.
  gpuMemoryTracker.suppressLimitCheck();
  try {
    let _st = profileSubOpBegin();
    const gradT = asContiguous(grad);
    const paramT = asContiguous(param);
    const mT = asContiguous(m);
    const vT = asContiguous(v);
    profileSubOpEnd("adam.ensureContig", _st);

    // Evict stale f16 cache entry for the old param buffer before dispatch.
    // Use deferred destruction since the old f16 buffer may still be referenced
    // by cast() results from the forward pass in the same command encoder batch.
    _st = profileSubOpBegin();
    const oldF16 = f16WeightCache.get(paramT.buffer);
    if (oldF16) {
      bufferPool.deferredDestroy(oldF16, oldF16.size);
      f16WeightCache.delete(paramT.buffer);
    }
    profileSubOpEnd("adam.f16evict", _st);

    // Flush shared encoder before dispatching the Adam kernel.
    // The forward/backward pass may have used param buffers as read-only
    // inputs in compute passes. The Adam kernel binds param as read_write,
    // which conflicts within the same command encoder synchronization scope.
    // Flushing ensures prior passes are submitted before the in-place writes.
    // In Adam batch mode, the caller already flushed once for the entire batch.
    if (!isAdamBatchMode()) {
      flushSharedEncoder();
    }

    // Re-protect adam's input buffers in the write set.
    trackSharedEncoderWrite(paramT.buffer);
    trackSharedEncoderWrite(mT.buffer);
    trackSharedEncoderWrite(vT.buffer);
    trackSharedEncoderWrite(gradT.buffer);

    // WebGPU forbids binding the same buffer as read_write at multiple
    // binding points. param is read_write, so de-alias it from read-only
    // inputs (grad, m, v) if pool recycling caused buffer sharing.
    // m_in/v_in are read-only, so they can safely alias each other or grad.
    // m_out/v_out are fresh allocations and cannot alias.
    let paramBuf = paramT.buffer;
    const mBuf = mT.buffer;
    const vBuf = vT.buffer;
    const gradBuf = gradT.buffer;
    const bufSize = gradT.size * dtypeBytes(gradT.dtype);
    if (paramBuf === gradBuf || paramBuf === mBuf || paramBuf === vBuf) {
      const dst = allocateOutputBuffer(bufSize);
      const enc = getSharedEncoderInstance();
      if (enc) {
        enc.copyBufferToBuffer(paramBuf, 0, dst, 0, bufSize);
      } else {
        const ctx2 = requireContext();
        const tmpEnc = ctx2.device.createCommandEncoder();
        tmpEnc.copyBufferToBuffer(paramBuf, 0, dst, 0, bufSize);
        ctx2.queue.submit([tmpEnc.finish()]);
      }
      paramBuf = dst;
      flushSharedEncoder();
    }

    const emitF16 = config.emitF16 ?? false;
    const infFlagBuf = (config.infFlagBuffer as GPUBuffer | null) ?? null;
    const numElements = gradT.size;

    const result = dispatchAdamStepKernel(
      gradBuf,
      paramBuf,
      mBuf,
      vBuf,
      numElements,
      config,
      emitF16,
      infFlagBuf,
    );

    // Post-dispatch flush only needed outside shared encoder mode.
    // In shared encoder mode, all buffer releases are deferred (deferredDestroy for grad,
    // sharedEncoderDeferredUniformBuffers for config, noop'd destroy for param).
    // No buffer destruction occurs during plan execution.
    if (!isSharedEncoderActive()) {
      flushSharedEncoder();
    }

    cleanupContiguous([grad, gradT]);

    // Cache the f16 param buffer (keyed by the param buffer, same as before)
    if (result.paramF16Buffer) {
      f16WeightCache.set(result.paramBuffer, result.paramF16Buffer);
    }

    // param is updated in-place — transfer ownership from old tensor to result.
    // m/v are fresh output buffers — no ownership transfer needed.
    _st = profileSubOpBegin();
    if (paramT.ownsBuffer) bufferPool.decRef(paramT.buffer);
    paramT.destroy = () => {};
    const ret = {
      param: createTensor(
        paramT.shape,
        result.paramBuffer,
        undefined,
        0,
        paramT.dtype,
      ),
      m: createTensor(mT.shape, result.mBuffer, undefined, 0, mT.dtype),
      v: createTensor(vT.shape, result.vBuffer, undefined, 0, vT.dtype),
    };
    profileSubOpEnd("adam.createTensor", _st);
    return ret;
  } finally {
    gpuMemoryTracker.unsuppressLimitCheck();
  }
}

// ============================================================================
// Fused Unscale + Inf-Check (GradScaler)
// ============================================================================

export function unscaleGrad(
  grad: BackendTensor,
  invScale: number,
  infFlagBuffer: unknown,
): BackendTensor {
  const gradT = asContiguous(grad);
  const numElements = gradT.size;
  const result = dispatchUnscaleGradKernel(
    gradT.buffer,
    numElements,
    invScale,
    infFlagBuffer as GPUBuffer,
  );
  cleanupContiguous([grad, gradT]);
  return createTensor(gradT.shape, result.gradOutBuffer);
}

export function createInfCountBuffer(): unknown {
  return allocateInfFlagBuffer();
}

export async function readAndDestroyInfCount(buffer: unknown): Promise<number> {
  return readInfFlag(buffer as GPUBuffer);
}

// ============================================================================
// Fused Cross-Entropy (Forward + Backward)
// ============================================================================

export function fusedCrossEntropyForward(
  logits: BackendTensor,
  targets: BackendTensor,
  config: import("../../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = asContiguous(logits);
  const targetsT = asContiguous(targets);
  const outBuf = dispatchCEForwardKernel(
    logitsT.buffer,
    targetsT.buffer,
    config.batchSize,
    config.vocabSize,
  );
  cleanupContiguous([logits, logitsT], [targets, targetsT]);
  return createTensor([config.batchSize], outBuf);
}

export function fusedCrossEntropyBackward(
  logits: BackendTensor,
  targets: BackendTensor,
  gradOutput: BackendTensor,
  config: import("../../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = asContiguous(logits);
  const targetsT = asContiguous(targets);
  const gradT = asContiguous(gradOutput);
  const outBuf = dispatchCEBackwardKernel(
    logitsT.buffer,
    targetsT.buffer,
    gradT.buffer,
    config.batchSize,
    config.vocabSize,
  );
  cleanupContiguous(
    [logits, logitsT],
    [targets, targetsT],
    [gradOutput, gradT],
  );
  return createTensor([config.batchSize, config.vocabSize], outBuf);
}

// ============================================================================
// Fused LayerNorm (Forward + Backward)
// ============================================================================

export function fusedLayerNormForward(
  x: BackendTensor,
  weight: BackendTensor,
  bias: BackendTensor,
  config: import("../../types").FusedLayerNormConfig,
): BackendTensor {
  const xT = asContiguous(x);
  const weightT = asContiguous(weight);
  const biasT = asContiguous(bias);
  const outBuf = dispatchLNForwardKernel(
    xT.buffer,
    weightT.buffer,
    biasT.buffer,
    config.numRows,
    config.featureDim,
    config.eps,
  );
  cleanupContiguous([x, xT], [weight, weightT], [bias, biasT]);
  return createTensor(xT.shape.slice(), outBuf);
}

export function fusedLayerNormBackwardGradX(
  gradOutput: BackendTensor,
  x: BackendTensor,
  weight: BackendTensor,
  config: import("../../types").FusedLayerNormConfig,
): BackendTensor {
  const gradT = asContiguous(gradOutput);
  const xT = asContiguous(x);
  const weightT = asContiguous(weight);

  const gradXBuf = dispatchLNBwdGradXKernel(
    gradT.buffer,
    xT.buffer,
    weightT.buffer,
    config.numRows,
    config.featureDim,
    config.eps,
  );
  cleanupContiguous([gradOutput, gradT], [x, xT], [weight, weightT]);
  return createTensor(xT.shape.slice(), gradXBuf);
}

export function fusedLayerNormBackwardGradWeightBias(
  gradOutput: BackendTensor,
  x: BackendTensor,
  config: import("../../types").FusedLayerNormConfig,
): { gradWeight: BackendTensor; gradBias: BackendTensor } {
  const gradT = asContiguous(gradOutput);
  const xT = asContiguous(x);
  const result = dispatchLNBwdGradWBKernel(
    gradT.buffer,
    xT.buffer,
    config.numRows,
    config.featureDim,
    config.eps,
  );
  cleanupContiguous([gradOutput, gradT], [x, xT]);
  const shape = [config.featureDim];
  return {
    gradWeight: createTensor(shape, result.gradWeightBuffer),
    gradBias: createTensor(shape, result.gradBiasBuffer),
  };
}

// ============================================================================
// Fused RMSNorm (Forward)
// ============================================================================

export function fusedRMSNormForward(
  x: BackendTensor,
  weight: BackendTensor,
  config: import("../../types").FusedRMSNormConfig,
): BackendTensor {
  const xT = asContiguous(x);
  const weightT = asContiguous(weight);
  const outBuf = dispatchRMSForwardKernel(
    xT.buffer,
    weightT.buffer,
    config.numRows,
    config.featureDim,
    config.eps,
  );
  cleanupContiguous([x, xT], [weight, weightT]);
  return createTensor(xT.shape.slice(), outBuf);
}

export function fusedRMSNormBackwardGradX(
  gradOutput: BackendTensor,
  x: BackendTensor,
  weight: BackendTensor,
  config: import("../../types").FusedRMSNormConfig,
): BackendTensor {
  const goT = asContiguous(gradOutput);
  const xT = asContiguous(x);
  const wT = asContiguous(weight);
  const outBuf = dispatchRMSBwdGradXKernel(
    goT.buffer,
    xT.buffer,
    wT.buffer,
    config.numRows,
    config.featureDim,
    config.eps,
  );
  cleanupContiguous([gradOutput, goT], [x, xT], [weight, wT]);
  return createTensor(xT.shape.slice(), outBuf);
}

export function fusedRMSNormBackwardGradWeight(
  gradOutput: BackendTensor,
  x: BackendTensor,
  _weight: BackendTensor,
  config: import("../../types").FusedRMSNormConfig,
): BackendTensor {
  const goT = asContiguous(gradOutput);
  const xT = asContiguous(x);
  const outBuf = dispatchRMSBwdGradWKernel(
    goT.buffer,
    xT.buffer,
    config.numRows,
    config.featureDim,
    config.eps,
  );
  cleanupContiguous([gradOutput, goT], [x, xT]);
  return createTensor([config.featureDim], outBuf);
}

// ============================================================================
// Fused FlashAttention (Forward + Backward)
// ============================================================================

export function fusedAttentionForward(
  q: BackendTensor,
  k: BackendTensor,
  v: BackendTensor,
  config: import("../../types").FusedAttentionConfig,
): { output: BackendTensor; logsumexp: BackendTensor } {
  const qT = asContiguous(q);
  const kT = asContiguous(k);
  const vT = asContiguous(v);
  const { outputBuffer, logsumexpBuffer } = dispatchFAForwardKernel(
    qT.buffer,
    kT.buffer,
    vT.buffer,
    config.batchSize,
    config.numHeads,
    config.seqLen,
    config.headDim,
    config.scale,
    config.isCausal,
  );

  cleanupContiguous([q, qT], [k, kT], [v, vT]);
  const outShape = [
    config.batchSize,
    config.numHeads,
    config.seqLen,
    config.headDim,
  ];
  const lseShape = [config.batchSize, config.numHeads, config.seqLen];
  return {
    output: createTensor(outShape, outputBuffer),
    logsumexp: createTensor(lseShape, logsumexpBuffer),
  };
}

export function fusedAttentionBackward(
  q: BackendTensor,
  k: BackendTensor,
  v: BackendTensor,
  logsumexp: BackendTensor,
  dO: BackendTensor,
  output: BackendTensor,
  config: import("../../types").FusedAttentionConfig,
): { dQ: BackendTensor; dK: BackendTensor; dV: BackendTensor } {
  const qT = asContiguous(q);
  const kT = asContiguous(k);
  const vT = asContiguous(v);
  const lseT = asContiguous(logsumexp);
  const dOT = asContiguous(dO);
  const oT = asContiguous(output);

  // Step 1: Compute D[i] = rowsum(dO[i,:] * O[i,:])
  const dBuf = dispatchFABwdDKernel(
    dOT.buffer,
    oT.buffer,
    config.batchSize,
    config.numHeads,
    config.seqLen,
    config.headDim,
    config.scale,
    config.isCausal,
  );

  // Step 2: Compute dQ
  const dQBuf = dispatchFABwdDQKernel(
    qT.buffer,
    kT.buffer,
    vT.buffer,
    lseT.buffer,
    dBuf,
    dOT.buffer,
    config.batchSize,
    config.numHeads,
    config.seqLen,
    config.headDim,
    config.scale,
    config.isCausal,
  );

  // Step 3: Compute dK, dV
  const { dKBuffer, dVBuffer } = dispatchFABwdDKVKernel(
    qT.buffer,
    kT.buffer,
    vT.buffer,
    lseT.buffer,
    dBuf,
    dOT.buffer,
    config.batchSize,
    config.numHeads,
    config.seqLen,
    config.headDim,
    config.scale,
    config.isCausal,
  );

  // D buffer is an intermediate — release it back to the pool now
  const dBufSize = config.batchSize * config.numHeads * config.seqLen * 4;
  bufferPool.deferredDestroy(dBuf, dBufSize);

  cleanupContiguous(
    [q, qT],
    [k, kT],
    [v, vT],
    [logsumexp, lseT],
    [dO, dOT],
    [output, oT],
  );
  const gradShape = [
    config.batchSize,
    config.numHeads,
    config.seqLen,
    config.headDim,
  ];
  return {
    dQ: createTensor(gradShape, dQBuf),
    dK: createTensor(gradShape, dKBuffer),
    dV: createTensor(gradShape, dVBuffer),
  };
}

export async function read(a: BackendTensor): Promise<number[]> {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);
  if (tensor.size === 0) {
    return [];
  }

  // Flush shared encoder before readback — all prior GPU work must be submitted.
  if (sharedEncoderActive) {
    flushSharedEncoder();
  }

  // Materialize non-contiguous tensors before reading
  const originalTensor = tensor;
  if (!tensor.isContiguous) {
    tensor = ensureContiguous(tensor);
  }

  const bytesPerElement = dtypeBytes(tensor.dtype);
  const totalBytes = tensor.size * bytesPerElement;
  const alignedBytes = alignBufferSize(totalBytes);
  const readBuffer = createTrackedBuffer(ctx.device, {
    size: alignedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  // Use the shared encoder for the copy if active, otherwise create a standalone one
  const sharedEnc = getSharedEncoderInstance();
  if (sharedEnc) {
    sharedEnc.copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
    // Flush to submit the copy command
    flushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
    profileApiCall("queue.submit", () => ctx.queue.submit([encoder.finish()]));
    incrementSubmitCount();
  }
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  // GPU work is complete - safe to flush pending buffers
  bufferPool.flushPendingToAvailable();
  await readBuffer.mapAsync(GPUMapMode.READ);
  const mapped = readBuffer.getMappedRange();

  // Create appropriate TypedArray view based on dtype
  // Slice only the actual data bytes to exclude padding
  let result: number[];
  switch (tensor.dtype) {
    case "i32":
      result = Array.from(new Int32Array(mapped.slice(0, totalBytes)));
      break;
    case "u32":
      result = Array.from(new Uint32Array(mapped.slice(0, totalBytes)));
      break;
    case "f16":
      // Convert f16 (Uint16Array) back to f32 numbers
      result = f16ArrayToF32Array(new Uint16Array(mapped.slice(0, totalBytes)));
      break;
    default:
      result = Array.from(new Float32Array(mapped.slice(0, totalBytes)));
      break;
  }
  readBuffer.unmap();

  // Destroy staging buffer to prevent memory leaks
  bufferPool.deferredDestroy(readBuffer, readBuffer.size ?? alignedBytes);

  // Destroy contiguous copy if we created one
  if (tensor !== originalTensor && tensor.destroy) {
    tensor.destroy();
  }

  return result;
}

/**
 * Wait for all submitted GPU work to complete.
 * Call this before destroying buffers to ensure GPU is done using them.
 */
export async function waitForGPU(): Promise<void> {
  const ctx = gpuContext;
  if (!ctx) return;
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  // GPU work is complete - safe to flush pending buffers
  bufferPool.flushPendingToAvailable();
}

// ============================================================================
// In-place Scalar Operations (for optimizer updates)
// ============================================================================

const WG_MUL = WORKGROUP_SIZE;

const mulScalarInPlaceSpec: TileKernelSpec = {
  name: "mulScalarInPlace",
  workgroupSize: WG_MUL,
  bindings: {
    data: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    scalar: "f32",
    size: "u32",
  },
  grid: elementwiseGrid(WG_MUL, { elementUniform: "size" }),
  kernel(ctx) {
    const idx = ctx.elementIndex(WG_MUL);
    const scalar = ctx.uniform("scalar").toF32();
    ctx.emitStore("data", idx, ctx.load("data", idx).mul(scalar));
  },
};

const mulScalarKernel = createTileKernelDispatcher(mulScalarInPlaceSpec);

/**
 * Multiply tensor by a scalar in-place (for gradient unscaling).
 * This is a simple in-place multiplication without NaN checking.
 */
export function mulScalarInPlace(tensor: BackendTensor, scalar: number): void {
  const a = asGPUTensor(tensor);
  mulScalarKernel.dispatch({ data: a.buffer }, { scalar, size: a.size });
}
