/**
 * Fused kernel wrappers and I/O ops: adamStep, unscaleGrad, fusedCrossEntropy,
 * fusedLayerNorm, fusedAttention, read, waitForGPU, mulScalarInPlace.
 * Extracted from index.ts — purely structural refactoring.
 */
import type { BackendTensor } from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../gpu-types";
import {
  sizeOf, WORKGROUP_SIZE, dtypeBytes, alignBufferSize,
} from "../shape-utils";
import {
  requireContext, context, isF16Supported,
  f16WeightCache, f16ArrayToF32Array,
} from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import { bufferPool } from "../buffer-pool";
import { allocateOutputBuffer, resolveOutputBuffer } from "../buffer-arena";
import {
  cachedCreateBindGroup, createParamsBuffer, releaseParamsBuffer,
  profiledCreateBindGroup,
} from "../bind-group-cache";
import {
  sharedEncoder as sharedEncoderFlag,
  flushSharedEncoder, isSharedEncoderActive,
  getSharedEncoderInstance, trackSharedEncoderWrite,
  isAdamBatchMode, incrementSubmitCount,
} from "../shared-encoder";
import {
  profileApiCall, profileSubOpBegin, profileSubOpEnd,
} from "../profiler";
import { gpuMemoryTracker } from "../memory-tracker";
import { ensureContiguous } from "./views";
import { dispatchAdamStep as dispatchAdamStepKernel } from "../adam-kernel";
import { dispatchUnscaleGrad as dispatchUnscaleGradKernel, allocateInfFlagBuffer, readInfFlag } from "../unscale-kernel";
import { dispatchCrossEntropyForward as dispatchCEForwardKernel, dispatchCrossEntropyBackward as dispatchCEBackwardKernel } from "../cross-entropy-kernel";
import { dispatchFlashAttentionForward as dispatchFAForwardKernel, dispatchFlashAttentionBackwardD as dispatchFABwdDKernel, dispatchFlashAttentionBackwardDQ as dispatchFABwdDQKernel, dispatchFlashAttentionBackwardDKV as dispatchFABwdDKVKernel } from "../attention-kernel";
import { dispatchLayerNormForward as dispatchLNForwardKernel, dispatchLayerNormBackwardGradX as dispatchLNBwdGradXKernel, dispatchLayerNormBackwardGradWeightBias as dispatchLNBwdGradWBKernel } from "../layernorm-kernel";

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
    const gradT = ensureContiguous(grad as WebGPUTensor);
    const paramT = ensureContiguous(param as WebGPUTensor);
    const mT = ensureContiguous(m as WebGPUTensor);
    const vT = ensureContiguous(v as WebGPUTensor);
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

    const bpe = (t: WebGPUTensor) => dtypeBytes(t.dtype);

    // Flush shared encoder before dispatching the in-place Adam kernel.
    // The forward/backward pass may have used param/m/v buffers as read-only
    // inputs in compute passes. The Adam kernel binds them as read_write,
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
    // binding points. De-alias input buffers if pool recycling caused
    // any to share the same GPUBuffer (common for same-size tensors).
    let paramBuf = paramT.buffer;
    let mBuf = mT.buffer;
    let vBuf = vT.buffer;
    const gradBuf = gradT.buffer;
    const bufSize = gradT.size * bpe(gradT);
    const copyToFresh = (src: GPUBuffer): GPUBuffer => {
      const dst = allocateOutputBuffer(bufSize);
      // Use the shared encoder for the copy so it's ordered before the
      // Adam compute pass in the same command buffer submission.
      const enc = getSharedEncoderInstance();
      if (enc) {
        (enc as any).copyBufferToBuffer(src, 0, dst, 0, bufSize);
      } else {
        const ctx2 = requireContext();
        const tmpEnc = ctx2.device.createCommandEncoder();
        tmpEnc.copyBufferToBuffer(src, 0, dst, 0, bufSize);
        ctx2.queue.submit([tmpEnc.finish()]);
      }
      return dst;
    };
    if (paramBuf === gradBuf || paramBuf === mBuf || paramBuf === vBuf) {
      paramBuf = copyToFresh(paramBuf);
    }
    if (mBuf === gradBuf || mBuf === vBuf) {
      mBuf = copyToFresh(mBuf);
    }
    if (vBuf === gradBuf) {
      vBuf = copyToFresh(vBuf);
    }

    // If any copies were made, flush again so the copy commands are submitted
    // before the Adam kernel binds the original buffers as read_write.
    if (paramBuf !== paramT.buffer || mBuf !== mT.buffer || vBuf !== vT.buffer) {
      flushSharedEncoder();
    }

    const emitF16 = config.emitF16 ?? false;
    const infFlagBuf = (config.infFlagBuffer as any) ?? null;
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
    // sharedEncoderDeferredUniformBuffers for config, noop'd destroy for param/m/v).
    // No buffer destruction occurs during plan execution.
    if (!isSharedEncoderActive()) {
      flushSharedEncoder();
    }

    // Destroy contiguous copy for grad only (read-only input, safe to free).
    if (gradT !== (grad as WebGPUTensor)) {
      bufferPool.decRef(gradT.buffer);
      bufferPool.deferredDestroy(gradT.buffer, gradT.size * bpe(gradT));
    }

    // Cache the f16 param buffer (keyed by the param buffer, same as before)
    if (result.paramF16Buffer) {
      f16WeightCache.set(result.paramBuffer, result.paramF16Buffer);
    }

    // In-place: the kernel wrote to the input buffers (or de-aliased copies).
    // Create new owning WebGPUTensor wrappers for the result. Replace the
    // input tensors' destroy() with a no-op so that when the engine disposes
    // the old storage handles, the buffer won't be released to the pool
    // (the result tensors now own it). This prevents double-free.
    _st = profileSubOpBegin();
    // Explicitly decRef before noop-ing destroy — ownership transfers to result tensors.
    if (paramT.ownsBuffer) bufferPool.decRef(paramT.buffer);
    if (mT.ownsBuffer) bufferPool.decRef(mT.buffer);
    if (vT.ownsBuffer) bufferPool.decRef(vT.buffer);
    const noop = () => {};
    paramT.destroy = noop;
    mT.destroy = noop;
    vT.destroy = noop;
    const ret = {
      param: createTensor(paramT.shape, result.paramBuffer, undefined, 0, paramT.dtype),
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
  const gradT = ensureContiguous(grad as WebGPUTensor);
  const numElements = gradT.size;
  const result = dispatchUnscaleGradKernel(
    gradT.buffer,
    numElements,
    invScale,
    infFlagBuffer as any,
  );
  // Destroy contiguous copy if one was created (deferred for GPU fence)
  if (gradT !== (grad as WebGPUTensor)) {
    bufferPool.decRef(gradT.buffer);
    bufferPool.deferredDestroy(gradT.buffer, gradT.size * dtypeBytes(gradT.dtype));
  }
  return createTensor(gradT.shape, result.gradOutBuffer);
}

export function createInfCountBuffer(): unknown {
  return allocateInfFlagBuffer();
}

export async function readAndDestroyInfCount(buffer: unknown): Promise<number> {
  return readInfFlag(buffer as any);
}

// ============================================================================
// Fused Cross-Entropy (Forward + Backward)
// ============================================================================

export function fusedCrossEntropyForward(
  logits: BackendTensor,
  targets: BackendTensor,
  config: import("../../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = ensureContiguous(logits as WebGPUTensor);
  const targetsT = ensureContiguous(targets as WebGPUTensor);
  const outBuf = dispatchCEForwardKernel(
    logitsT.buffer, targetsT.buffer, config.batchSize, config.vocabSize,
  );
  // Destroy contiguous copies if created (deferred for GPU fence)
  if (logitsT !== logits) { bufferPool.decRef(logitsT.buffer); bufferPool.deferredDestroy(logitsT.buffer, logitsT.size * dtypeBytes(logitsT.dtype)); }
  if (targetsT !== targets) { bufferPool.decRef(targetsT.buffer); bufferPool.deferredDestroy(targetsT.buffer, targetsT.size * dtypeBytes(targetsT.dtype)); }
  return createTensor([config.batchSize], outBuf, undefined, 0, "f32");
}

export function fusedCrossEntropyBackward(
  logits: BackendTensor,
  targets: BackendTensor,
  gradOutput: BackendTensor,
  config: import("../../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = ensureContiguous(logits as WebGPUTensor);
  const targetsT = ensureContiguous(targets as WebGPUTensor);
  const gradT = ensureContiguous(gradOutput as WebGPUTensor);
  const outBuf = dispatchCEBackwardKernel(
    logitsT.buffer, targetsT.buffer, gradT.buffer,
    config.batchSize, config.vocabSize,
  );
  // Destroy contiguous copies if created (deferred for GPU fence)
  if (logitsT !== logits) { bufferPool.decRef(logitsT.buffer); bufferPool.deferredDestroy(logitsT.buffer, logitsT.size * dtypeBytes(logitsT.dtype)); }
  if (targetsT !== targets) { bufferPool.decRef(targetsT.buffer); bufferPool.deferredDestroy(targetsT.buffer, targetsT.size * dtypeBytes(targetsT.dtype)); }
  if (gradT !== gradOutput) { bufferPool.decRef(gradT.buffer); bufferPool.deferredDestroy(gradT.buffer, gradT.size * dtypeBytes(gradT.dtype)); }
  return createTensor([config.batchSize, config.vocabSize], outBuf, undefined, 0, "f32");
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
  const xT = ensureContiguous(x as WebGPUTensor);
  const weightT = ensureContiguous(weight as WebGPUTensor);
  const biasT = ensureContiguous(bias as WebGPUTensor);
  const outBuf = dispatchLNForwardKernel(
    xT.buffer, weightT.buffer, biasT.buffer,
    config.numRows, config.featureDim, config.eps,
  );
  // Destroy contiguous copies if created (deferred for GPU fence)
  if (xT !== x) { bufferPool.decRef(xT.buffer); bufferPool.deferredDestroy(xT.buffer, xT.size * dtypeBytes(xT.dtype)); }
  if (weightT !== weight) { bufferPool.decRef(weightT.buffer); bufferPool.deferredDestroy(weightT.buffer, weightT.size * dtypeBytes(weightT.dtype)); }
  if (biasT !== bias) { bufferPool.decRef(biasT.buffer); bufferPool.deferredDestroy(biasT.buffer, biasT.size * dtypeBytes(biasT.dtype)); }
  const outShape: number[] = [];
  // Reconstruct shape: [...batch_dims, featureDim] = same as input
  for (let i = 0; i < xT.shape.length; i++) outShape.push(xT.shape[i]);
  return createTensor(outShape, outBuf, undefined, 0, "f32");
}

export function fusedLayerNormBackwardGradX(
  gradOutput: BackendTensor,
  x: BackendTensor,
  weight: BackendTensor,
  config: import("../../types").FusedLayerNormConfig,
): BackendTensor {
  const gradT = ensureContiguous(gradOutput as WebGPUTensor);
  const xT = ensureContiguous(x as WebGPUTensor);
  const weightT = ensureContiguous(weight as WebGPUTensor);

  const gradXBuf = dispatchLNBwdGradXKernel(
    gradT.buffer, xT.buffer, weightT.buffer,
    config.numRows, config.featureDim, config.eps,
  );

  // Destroy contiguous copies if created (deferred for GPU fence)
  if (gradT !== gradOutput) { bufferPool.decRef(gradT.buffer); bufferPool.deferredDestroy(gradT.buffer, gradT.size * dtypeBytes(gradT.dtype)); }
  if (xT !== x) { bufferPool.decRef(xT.buffer); bufferPool.deferredDestroy(xT.buffer, xT.size * dtypeBytes(xT.dtype)); }
  if (weightT !== weight) { bufferPool.decRef(weightT.buffer); bufferPool.deferredDestroy(weightT.buffer, weightT.size * dtypeBytes(weightT.dtype)); }

  const gradXShape: number[] = [];
  for (let i = 0; i < xT.shape.length; i++) gradXShape.push(xT.shape[i]);
  return createTensor(gradXShape, gradXBuf, undefined, 0, "f32");
}

export function fusedLayerNormBackwardGradWeightBias(
  gradOutput: BackendTensor,
  x: BackendTensor,
  config: import("../../types").FusedLayerNormConfig,
): { gradWeight: BackendTensor; gradBias: BackendTensor } {
  const gradT = ensureContiguous(gradOutput as WebGPUTensor);
  const xT = ensureContiguous(x as WebGPUTensor);
  const result = dispatchLNBwdGradWBKernel(
    gradT.buffer, xT.buffer,
    config.numRows, config.featureDim, config.eps,
  );

  // Destroy contiguous copies if created (deferred for GPU fence)
  if (gradT !== gradOutput) { bufferPool.decRef(gradT.buffer); bufferPool.deferredDestroy(gradT.buffer, gradT.size * dtypeBytes(gradT.dtype)); }
  if (xT !== x) { bufferPool.decRef(xT.buffer); bufferPool.deferredDestroy(xT.buffer, xT.size * dtypeBytes(xT.dtype)); }

  const shape = [config.featureDim];
  return {
    gradWeight: createTensor(shape, result.gradWeightBuffer, undefined, 0, "f32", true),
    gradBias: createTensor(shape, result.gradBiasBuffer, undefined, 0, "f32", true),
  };
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
  const qT = ensureContiguous(q as WebGPUTensor);
  const kT = ensureContiguous(k as WebGPUTensor);
  const vT = ensureContiguous(v as WebGPUTensor);
  const { outputBuffer, logsumexpBuffer } = dispatchFAForwardKernel(
    qT.buffer, kT.buffer, vT.buffer,
    config.batchSize, config.numHeads, config.seqLen, config.headDim,
    config.scale, config.isCausal,
  );

  // Destroy contiguous copies if created (deferred for GPU fence)
  const bpe = 4; // f32
  if (qT !== q) { bufferPool.decRef(qT.buffer); bufferPool.deferredDestroy(qT.buffer, qT.size * bpe); }
  if (kT !== k) { bufferPool.decRef(kT.buffer); bufferPool.deferredDestroy(kT.buffer, kT.size * bpe); }
  if (vT !== v) { bufferPool.decRef(vT.buffer); bufferPool.deferredDestroy(vT.buffer, vT.size * bpe); }

  const outShape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
  const lseShape = [config.batchSize, config.numHeads, config.seqLen];
  return {
    output: createTensor(outShape, outputBuffer, undefined, 0, "f32"),
    logsumexp: createTensor(lseShape, logsumexpBuffer, undefined, 0, "f32"),
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
  const qT = ensureContiguous(q as WebGPUTensor);
  const kT = ensureContiguous(k as WebGPUTensor);
  const vT = ensureContiguous(v as WebGPUTensor);
  const lseT = ensureContiguous(logsumexp as WebGPUTensor);
  const dOT = ensureContiguous(dO as WebGPUTensor);
  const oT = ensureContiguous(output as WebGPUTensor);

  // Step 1: Compute D[i] = rowsum(dO[i,:] * O[i,:])
  const dBuf = dispatchFABwdDKernel(
    dOT.buffer, oT.buffer,
    config.batchSize, config.numHeads, config.seqLen, config.headDim,
    config.scale, config.isCausal,
  );

  // Step 2: Compute dQ
  const dQBuf = dispatchFABwdDQKernel(
    qT.buffer, kT.buffer, vT.buffer, lseT.buffer, dBuf, dOT.buffer,
    config.batchSize, config.numHeads, config.seqLen, config.headDim,
    config.scale, config.isCausal,
  );

  // Step 3: Compute dK, dV
  const { dKBuffer, dVBuffer } = dispatchFABwdDKVKernel(
    qT.buffer, kT.buffer, vT.buffer, lseT.buffer, dBuf, dOT.buffer,
    config.batchSize, config.numHeads, config.seqLen, config.headDim,
    config.scale, config.isCausal,
  );

  // D buffer is an intermediate — release it back to the pool now
  const dBufSize = config.batchSize * config.numHeads * config.seqLen * 4;
  bufferPool.deferredDestroy(dBuf, dBufSize);

  // Destroy contiguous copies if created (deferred for GPU fence)
  const bpe = 4; // f32
  if (qT !== q) { bufferPool.decRef(qT.buffer); bufferPool.deferredDestroy(qT.buffer, qT.size * bpe); }
  if (kT !== k) { bufferPool.decRef(kT.buffer); bufferPool.deferredDestroy(kT.buffer, kT.size * bpe); }
  if (vT !== v) { bufferPool.decRef(vT.buffer); bufferPool.deferredDestroy(vT.buffer, vT.size * bpe); }
  if (lseT !== logsumexp) { bufferPool.decRef(lseT.buffer); bufferPool.deferredDestroy(lseT.buffer, lseT.size * bpe); }
  if (dOT !== dO) { bufferPool.decRef(dOT.buffer); bufferPool.deferredDestroy(dOT.buffer, dOT.size * bpe); }
  if (oT !== output) { bufferPool.decRef(oT.buffer); bufferPool.deferredDestroy(oT.buffer, oT.size * bpe); }

  const gradShape = [config.batchSize, config.numHeads, config.seqLen, config.headDim];
  return {
    dQ: createTensor(gradShape, dQBuf, undefined, 0, "f32"),
    dK: createTensor(gradShape, dKBuffer, undefined, 0, "f32"),
    dV: createTensor(gradShape, dVBuffer, undefined, 0, "f32"),
  };
}

export async function read(a: BackendTensor): Promise<number[]> {
  const ctx = requireContext();
  let tensor = a as WebGPUTensor;
  if (tensor.size === 0) {
    return [];
  }

  // Flush shared encoder before readback — all prior GPU work must be submitted.
  if (sharedEncoderFlag) {
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
  if (getSharedEncoderInstance()) {
    getSharedEncoderInstance().copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
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
    case "f32":
    default:
      result = Array.from(new Float32Array(mapped.slice(0, totalBytes)));
      break;
  }
  readBuffer.unmap();

  // Destroy staging buffer to prevent memory leaks
  bufferPool.deferredDestroy(readBuffer, (readBuffer as any).size ?? alignedBytes);

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
  const ctx = context;
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

/**
 * Multiply tensor by a scalar in-place (for gradient unscaling).
 * This is a simple in-place multiplication without NaN checking.
 */
export function mulScalarInPlace(tensor: BackendTensor, scalar: number): void {
  const ctx = requireContext();
  const a = tensor as WebGPUTensor;
  const size = a.size;

  // Pack mixed f32 + u32 params
  const fillParamsData = new ArrayBuffer(8);
  new Float32Array(fillParamsData, 0, 1)[0] = scalar;
  new Uint32Array(fillParamsData, 4, 1)[0] = size;
  const uniformBuffer = createParamsBuffer(ctx.device, new Uint32Array(fillParamsData));

  const code = `
struct Params {
  scalar: f32,
  size: u32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) {
    return;
  }
  data[idx] = data[idx] * params.scalar;
}
`;

  const key = `mul_scalar_inplace:${size}`;
  const pipeline = getPipeline(ctx, key, code);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [a.buffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, Math.ceil(size / WORKGROUP_SIZE));

  releaseParamsBuffer(uniformBuffer);
}
