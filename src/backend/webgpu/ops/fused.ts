/**
 * Fused kernel wrappers and I/O ops: adamStep, unscaleGrad, fusedCrossEntropy,
 * fusedLayerNorm, fusedAttention, read, waitForGPU, mulScalarInPlace.
 */
import { ENV } from "../../../core/env";
import type { BackendTensor } from "../../types";
import { allocateOutputBuffer } from "../buffer-arena";
import { bufferPool, destroyCopy } from "../buffer-pool";
import { advanceEpoch } from "../epoch";
import {
  assertNoDroppedSubmits,
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
  isSharedEncoderActive,
  trackSharedEncoderWrite,
} from "../shared-encoder";
import { createTensor, createTrackedBuffer } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";
import { dispatchDeviceTopK } from "../topk-kernel";
import type { TileKernelSpec } from "../tile-ir";
import { elementwiseGrid } from "../tile-ir";
import { gpuContext, sharedEncoderActive } from "../webgpu-state";
import {
  asContiguous,
  assertRawBindable,
  cast as castOp,
  ensureContiguous,
} from "./views";

/** Destroy contiguous copies that differ from their originals. */
function cleanupContiguous(...pairs: [BackendTensor, WebGPUTensor][]) {
  for (const [orig, copy] of pairs) {
    if (copy !== orig) destroyCopy(copy);
  }
}

import {
  dispatchPackedOptimizer,
  type PackedOptimizerItem,
} from "../../../optim/packed-dispatch";
import type { AdamBatchItem, AdamBatchResult } from "../../types";
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
import { dispatchRoPE as dispatchRoPEKernel } from "../rope-kernel";
import {
  allocateInfFlagBuffer,
  readAndDestroyInfSnapshot,
  dispatchUnscaleGrad as dispatchUnscaleGradKernel,
  readInfFlag,
  snapshotAndResetInfFlag,
} from "../unscale-kernel";

// ============================================================================
// Fused Adam/AdamW Step
// ============================================================================

/**
 * Per-item Adam step implementation. Does NOT call flushSharedEncoder — the
 * caller is responsible for ensuring previously-recorded passes that read
 * the param/m/v buffers have been submitted before invoking this. Use
 * `adamStep` (which flushes once per call) for standalone calls; use
 * `adamStepBatch` (which flushes once per batch) for grouped calls.
 */
function adamStepInner(
  grad: BackendTensor,
  param: BackendTensor,
  m: BackendTensor,
  v: BackendTensor,
  t: BackendTensor,
  lr: BackendTensor,
  config: import("../../types").AdamStepConfig,
): { param: BackendTensor; m: BackendTensor; v: BackendTensor } {
  let _st = profileSubOpBegin();
  const gradT = asContiguous(grad);
  const paramT = asContiguous(param);
  const mT = asContiguous(m);
  const vT = asContiguous(v);
  // t/lr are read-only 1-element inputs — resolve their buffers (contiguous).
  const tT = asContiguous(t);
  const lrT = asContiguous(lr);
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

  // Re-protect adam's input buffers in the write set.
  trackSharedEncoderWrite(paramT.buffer);
  trackSharedEncoderWrite(mT.buffer);
  trackSharedEncoderWrite(vT.buffer);
  trackSharedEncoderWrite(gradT.buffer);

  // WebGPU forbids binding the same buffer as read_write at multiple
  // binding points and forbids read_write ↔ read aliasing. All three of
  // param/m/v are read_write, grad is read — so each pair must have a
  // distinct buffer. Pool recycling can cause collisions.
  let paramBuf = paramT.buffer;
  let mBuf = mT.buffer;
  let vBuf = vT.buffer;
  const gradBuf = gradT.buffer;
  const bufSize = gradT.size * dtypeBytes(gradT.dtype);
  const copyTo = (src: GPUBuffer): GPUBuffer => {
    const dst = allocateOutputBuffer(bufSize);
    const enc = getSharedEncoderInstance();
    if (enc) {
      enc.copyBufferToBuffer(src, 0, dst, 0, bufSize);
    } else {
      const ctx2 = requireContext();
      const tmpEnc = ctx2.device.createCommandEncoder();
      tmpEnc.copyBufferToBuffer(src, 0, dst, 0, bufSize);
      ctx2.queue.submit([tmpEnc.finish()]);
    }
    flushSharedEncoder();
    return dst;
  };
  if (paramBuf === gradBuf || paramBuf === mBuf || paramBuf === vBuf) {
    paramBuf = copyTo(paramBuf);
  }
  if (mBuf === gradBuf || mBuf === vBuf) {
    mBuf = copyTo(mBuf);
  }
  if (vBuf === gradBuf) {
    vBuf = copyTo(vBuf);
  }

  const emitF16 = config.emitF16 ?? false;
  const infFlagBuf = (config.infFlagBuffer as GPUBuffer | null) ?? null;
  const numElements = gradT.size;

  const result = dispatchAdamStepKernel(
    gradBuf,
    paramBuf,
    mBuf,
    vBuf,
    tT.buffer,
    lrT.buffer,
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
  // t/lr are persistent optimizer state owned by the caller — only free the
  // contiguous COPY if asContiguous made one (cleanupContiguous no-ops when
  // tT===t). The original t/lr buffers are never destroyed here.
  cleanupContiguous([t, tT]);
  cleanupContiguous([lr, lrT]);

  // Cache the f16 param buffer (keyed by the param buffer, same as before)
  if (result.paramF16Buffer) {
    f16WeightCache.set(result.paramBuffer, result.paramF16Buffer);
  }

  // param/m/v are all updated in-place — transfer ownership from the old
  // BackendTensors to the results (noop their destroy, decRef their buffers).
  _st = profileSubOpBegin();
  if (paramT.ownsBuffer) bufferPool.decRef(paramT.buffer);
  paramT.destroy = () => {};
  if (mT.ownsBuffer) bufferPool.decRef(mT.buffer);
  mT.destroy = () => {};
  if (vT.ownsBuffer) bufferPool.decRef(vT.buffer);
  vT.destroy = () => {};
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
}

export async function adamStep(
  grad: BackendTensor,
  param: BackendTensor,
  m: BackendTensor,
  v: BackendTensor,
  t: BackendTensor,
  lr: BackendTensor,
  config: import("../../types").AdamStepConfig,
): Promise<{ param: BackendTensor; m: BackendTensor; v: BackendTensor }> {
  // Suppress memory limit checks for the entire adam step.
  // During optimizer steps, old param/m/v buffers haven't been released yet
  // while new output buffers are allocated — temporary 2x peak is expected.
  gpuMemoryTracker.suppressLimitCheck();
  try {
    // Flush shared encoder before dispatching the Adam kernel.
    // The forward/backward pass may have used param buffers as read-only
    // inputs in compute passes. The Adam kernel binds param as read_write,
    // which conflicts within the same command encoder synchronization scope.
    // Flushing ensures prior passes are submitted before the in-place writes.
    flushSharedEncoder();
    return adamStepInner(grad, param, m, v, t, lr, config);
  } finally {
    gpuMemoryTracker.unsuppressLimitCheck();
  }
}

/**
 * Batched Adam step. Groups same-element-count items into packed dispatches
 * (one kernel per size class instead of one per param), with per-item fallback
 * via `adamStepInner` for any items not handled by packing. The shared encoder
 * is flushed exactly ONCE for the whole batch, not per item.
 *
 * Returns results in the same order as `items`.
 */
export function adamStepBatch(items: AdamBatchItem[]): AdamBatchResult[] {
  if (items.length === 0) return [];
  gpuMemoryTracker.suppressLimitCheck();
  try {
    // Single flush before any item — all items share one logical batch.
    flushSharedEncoder();

    const results: Array<AdamBatchResult | null> = new Array(items.length).fill(
      null,
    );

    // Try packed dispatch for groups of size ≥ 2 with the same element count.
    // dispatchPackedOptimizer requires items.length > 1 to even attempt.
    // TORCHLETTE_PACKED_ADAM=0: measurement kill switch (per-item dispatches,
    // still one flush per batch) — quantifies what packing is worth.
    if (items.length > 1 && ENV.TORCHLETTE_PACKED_ADAM !== "0") {
      // Pre-resolve contiguous tensors per item, evict f16 caches, build the
      // packed item descriptors. We hold onto the contiguous tensors so the
      // fallback path can claim ownership uniformly with adamStepInner.
      const contig: Array<{
        gradT: WebGPUTensor;
        paramT: WebGPUTensor;
        mT: WebGPUTensor;
        vT: WebGPUTensor;
      }> = items.map((item) => ({
        gradT: asContiguous(item.grad),
        paramT: asContiguous(item.param),
        mT: asContiguous(item.m),
        vT: asContiguous(item.v),
      }));

      for (const { paramT } of contig) {
        const oldF16 = f16WeightCache.get(paramT.buffer);
        if (oldF16) {
          bufferPool.deferredDestroy(oldF16, oldF16.size);
          f16WeightCache.delete(paramT.buffer);
        }
      }

      const packedItems: PackedOptimizerItem[] = contig.map((c) => ({
        buffers: [c.gradT.buffer, c.paramT.buffer, c.mT.buffer, c.vT.buffer],
        numElements: c.gradT.size,
      }));

      // Packed kernel uses the FIRST item's config AND the FIRST item's t/lr
      // tensors. This is SOUND because the executor's adam-batch grouping key
      // (inc-2a) breaks the batch on lr-tensor identity: every item in one
      // batch shares the same static config AND the same persistent t/lr
      // tensors by construction. t/lr are 1-element shared bindings — they are
      // NOT scatter/gathered (packed-dispatch assumes per-item numElements
      // sizing); they are bound ONCE via the dispatch closure.
      const config = items[0].config;
      const infFlagBuffer = (config.infFlagBuffer as GPUBuffer | null) ?? null;
      const tBuf = (asContiguous(items[0].t) as WebGPUTensor).buffer;
      const lrBuf = (asContiguous(items[0].lr) as WebGPUTensor).buffer;

      const handled = dispatchPackedOptimizer({
        items: packedItems,
        gatherIndices: [1, 2, 3], // gather param, m, v back into original buffers
        dispatch(packed, totalElements) {
          dispatchAdamStepKernel(
            packed[0],
            packed[1],
            packed[2],
            packed[3],
            tBuf,
            lrBuf,
            totalElements,
            config,
            false,
            infFlagBuffer,
          );
        },
        label: "packedAdam",
      });

      // For items handled by packed dispatch: param/m/v buffers were updated
      // in place via the gather copy. Transfer ownership from the input BTs
      // to fresh BTs wrapping the same buffers — same pattern adamStepInner
      // uses for its single-item path.
      for (const i of handled) {
        const { gradT, paramT, mT, vT } = contig[i];
        const item = items[i];

        cleanupContiguous([item.grad, gradT]);

        if (paramT.ownsBuffer) bufferPool.decRef(paramT.buffer);
        paramT.destroy = () => {};
        if (mT.ownsBuffer) bufferPool.decRef(mT.buffer);
        mT.destroy = () => {};
        if (vT.ownsBuffer) bufferPool.decRef(vT.buffer);
        vT.destroy = () => {};

        results[i] = {
          param: createTensor(
            paramT.shape,
            paramT.buffer,
            undefined,
            0,
            paramT.dtype,
          ),
          m: createTensor(mT.shape, mT.buffer, undefined, 0, mT.dtype),
          v: createTensor(vT.shape, vT.buffer, undefined, 0, vT.dtype),
        };
      }
    }

    // Fallback path: for items not handled by packed dispatch (or all items
    // when items.length === 1), call adamStepInner per item. The encoder was
    // already flushed at the top, so adamStepInner skipping its own flush is
    // exactly what we want.
    for (let i = 0; i < items.length; i++) {
      if (results[i] !== null) continue;
      const item = items[i];
      results[i] = adamStepInner(
        item.grad,
        item.param,
        item.m,
        item.v,
        item.t,
        item.lr,
        item.config,
      );
    }

    return results as AdamBatchResult[];
  } finally {
    gpuMemoryTracker.unsuppressLimitCheck();
  }
}

// ============================================================================
// Fused Unscale + Inf-Check (GradScaler)
// ============================================================================

export function unscaleGrad(
  grad: BackendTensor,
  scale: BackendTensor,
  infFlagBuffer: unknown,
): BackendTensor {
  const gradT = asContiguous(grad);
  const numElements = gradT.size;
  // scaler-as-tensor: scale is a persistent 1-element f32 tensor (the
  // GradScaler's `_scaleLive` LiveScalar buffer) read LIVE from a storage
  // binding, not a frozen uniform number (invScale reciprocated in-kernel).
  const scaleT = scale as WebGPUTensor;
  const result = dispatchUnscaleGradKernel(
    gradT.buffer,
    numElements,
    scaleT.buffer,
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

/** [inc-3 ring] Per-step found-inf report isolation — see unscale-kernel.ts. */
export function snapshotInfFlag(): unknown | null {
  return snapshotAndResetInfFlag();
}

export async function readInfSnapshot(snapshot: unknown): Promise<number> {
  return readAndDestroyInfSnapshot(snapshot as GPUBuffer);
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
  const targetsRaw = asContiguous(targets);
  const targetsT = ensureI32Targets(targetsRaw);
  const outBuf = dispatchCEForwardKernel(
    logitsT.buffer,
    targetsT.buffer,
    config.batchSize,
    config.vocabSize,
    config.ignoreIndex ?? -100,
  );
  cleanupContiguous([logits, logitsT], [targets, targetsRaw]);
  if (targetsT !== targetsRaw) destroyCopy(targetsT);
  return createTensor([config.batchSize], outBuf);
}

export function fusedCrossEntropyBackward(
  logits: BackendTensor,
  targets: BackendTensor,
  gradOutput: BackendTensor,
  config: import("../../types").FusedCrossEntropyConfig,
): BackendTensor {
  const logitsT = asContiguous(logits);
  const targetsRaw = asContiguous(targets);
  const targetsT = ensureI32Targets(targetsRaw);
  const gradT = asContiguous(gradOutput);
  const outBuf = dispatchCEBackwardKernel(
    logitsT.buffer,
    targetsT.buffer,
    gradT.buffer,
    config.batchSize,
    config.vocabSize,
    config.ignoreIndex ?? -100,
  );
  cleanupContiguous(
    [logits, logitsT],
    [targets, targetsRaw],
    [gradOutput, gradT],
  );
  if (targetsT !== targetsRaw) destroyCopy(targetsT);
  return createTensor([config.batchSize, config.vocabSize], outBuf);
}

/**
 * Transitional compatibility shim: CE kernels expect i32 targets, but some
 * callers may still pass f32 (legacy tests, user code that predates the
 * dtype option on creation ops). If targets arrive as f32, cast to i32.
 */
function ensureI32Targets(targets: WebGPUTensor): WebGPUTensor {
  if (targets.dtype === "i32" || targets.dtype === "u32") return targets;
  return asGPUTensor(castOp(targets, "i32"));
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

// ============================================================================
// Device top-K prefilter (lazy) — the on-device sampling support selector
// ============================================================================

/**
 * Lazy top-K over a single logits row: `[.., V]` → packed `[1, 2, k]` (row 0 =
 * the K values descending, row 1 = the K token ids as f32 values). Stays
 * on-device (no readback) so decodeBlock composes top-p + Gumbel-max over the
 * survivors without a per-token host roundtrip. Reuses the SAME tile-IR passes
 * (and tie-break) as `readTopK`, so the device top-k SET is byte-identical to
 * the host `sampleFromTopK` reference. Input is raw-bound flat-from-0
 * (asContiguous), which also honors the arg-reduce contiguity seam downstream.
 */
export function deviceTopK(
  logits: BackendTensor,
  config: { k: number },
): BackendTensor {
  const lt = asContiguous(logits);
  const gt = asGPUTensor(lt);
  const length = gt.size; // single row [.., V] → V
  const out = dispatchDeviceTopK(gt.buffer, gt.offset ?? 0, length, config.k);
  cleanupContiguous([logits, lt]);
  return createTensor([1, 2, config.k], out);
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
// Fused RoPE
// ============================================================================

export function fusedRoPE(
  qk: BackendTensor,
  cos: BackendTensor,
  sin: BackendTensor,
  config: import("../../types").FusedRoPEConfig,
): BackendTensor {
  const qkT = asContiguous(qk);
  // cos/sin: a narrow() row-slice of a persistent [maxSeqLen, D/2] table has
  // contiguous strides + element offset — fold the offset into the kernel's
  // table indexing (zero-copy) instead of materializing. Non-contiguous
  // strides still materialize.
  const cosG = asGPUTensor(cos);
  const sinG = asGPUTensor(sin);
  const cosT = cosG.isContiguous ? cosG : asContiguous(cos);
  const sinT = sinG.isContiguous ? sinG : asContiguous(sin);
  const outBuf = dispatchRoPEKernel(
    qkT.buffer,
    cosT.buffer,
    sinT.buffer,
    config.total,
    config.seqLen,
    config.headDim,
    config.sinScale,
    cosT.offset,
    sinT.offset,
  );
  cleanupContiguous([qk, qkT], [cos, cosT], [sin, sinT]);
  return createTensor(qkT.shape.slice(), outBuf);
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
    config.modifier,
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
    config.modifier,
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
    config.modifier,
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
    config.modifier,
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

/**
 * Start a scalar readback: copy the tensor buffer to a staging buffer and submit.
 * Returns a finish function that mapAsync's the staging buffer and returns the value.
 * This allows backward to run between start and finish, overlapping GPU readback
 * with CPU graph construction.
 */
export function startScalarReadback(a: BackendTensor): () => Promise<number> {
  const ctx = requireContext();
  // Offset/strided views must not be read from byte 0 (offset-view class —
  // task #58): materialize first. Scalars are tiny, so the copy is cheap.
  const original = asGPUTensor(a);
  const tensor = ensureContiguous(original);
  const bytesPerElement = dtypeBytes(tensor.dtype);
  const totalBytes = tensor.size * bytesPerElement;
  const alignedBytes = alignBufferSize(totalBytes);

  const readBuffer = createTrackedBuffer(ctx.device, {
    size: alignedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Copy to staging buffer on the shared encoder and flush
  const sharedEnc = getSharedEncoderInstance();
  if (sharedEnc) {
    sharedEnc.copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
    flushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.buffer, 0, readBuffer, 0, alignedBytes);
    profileApiCall("queue.submit", () => ctx.queue.submit([encoder.finish()]));
    incrementSubmitCount();
  }
  if (tensor !== original) destroyCopy(tensor);

  // Return a finish function that maps the staging buffer and reads the scalar
  const dtype = tensor.dtype;
  return async (): Promise<number> => {
    await readBuffer.mapAsync(GPUMapMode.READ);
    const mapped = readBuffer.getMappedRange();
    let value: number;
    switch (dtype) {
      case "f16": {
        const u16 = new Uint16Array(mapped.slice(0, 2));
        value = f16ArrayToF32Array(u16)[0];
        break;
      }
      case "i32":
        value = new Int32Array(mapped.slice(0, 4))[0];
        break;
      case "u32":
        value = new Uint32Array(mapped.slice(0, 4))[0];
        break;
      default:
        value = new Float32Array(mapped.slice(0, 4))[0];
        break;
    }
    readBuffer.unmap();
    bufferPool.deferredDestroy(readBuffer, readBuffer.size ?? alignedBytes);
    return value;
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

  // Materialize views before reading. A narrow view can have CONTIGUOUS
  // strides with a non-zero element OFFSET — copying its buffer from byte 0
  // silently returns the wrong region (offset-view class, task #58). When
  // the view is contiguous-strided and the offset is 4-byte aligned we can
  // copy directly from the byte offset; otherwise materialize via
  // contiguous() (which folds offset/strides in its kernel).
  const originalTensor = tensor;
  const bytesPerElement = dtypeBytes(tensor.dtype);
  const totalBytes = tensor.size * bytesPerElement;
  let srcByteOffset = 0;
  if (!tensor.isContiguous) {
    tensor = ensureContiguous(tensor);
  } else if (tensor.offset > 0) {
    const offsetBytes = tensor.offset * bytesPerElement;
    if (
      offsetBytes % 4 === 0 &&
      offsetBytes + alignBufferSize(totalBytes) <= tensor.buffer.size
    ) {
      srcByteOffset = offsetBytes;
    } else {
      tensor = ensureContiguous(tensor);
    }
  }

  const alignedBytes = alignBufferSize(totalBytes);
  const readBuffer = createTrackedBuffer(ctx.device, {
    size: alignedBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  // Use the shared encoder for the copy if active, otherwise create a standalone one
  const sharedEnc = getSharedEncoderInstance();
  if (sharedEnc) {
    sharedEnc.copyBufferToBuffer(
      tensor.buffer,
      srcByteOffset,
      readBuffer,
      0,
      alignedBytes,
    );
    // Flush to submit the copy command
    flushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    encoder.copyBufferToBuffer(
      tensor.buffer,
      srcByteOffset,
      readBuffer,
      0,
      alignedBytes,
    );
    profileApiCall("queue.submit", () => ctx.queue.submit([encoder.finish()]));
    incrementSubmitCount();
  }
  if (typeof ctx.queue.onSubmittedWorkDone === "function") {
    await ctx.queue.onSubmittedWorkDone();
  }
  // The fence has settled — any dropped submit's uncaptured error has been
  // delivered by now. Under STRICT_GPU, throw (naming device pressure) rather
  // than read back the stale/all-zero data a dropped submit leaves behind
  // (task #94, item 3 — the VULKAN_DEVICE_INDEX=1 all-zero incident).
  assertNoDroppedSubmits("tensor readback");
  // GPU work is complete — quiescent point: advance the engine epoch
  // (flushes pending pool buffers).
  advanceEpoch("readback");
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
  // GPU work is complete — quiescent point: advance the engine epoch
  // (flushes pending pool buffers).
  advanceEpoch("waitForGPU");
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
  // In-place op binding the buffer flat from element 0: an offset/strided
  // view here would CORRUPT the base's leading region — auto-materialize is
  // not an option for in-place semantics, so throw loudly (offset-view
  // class, task #58).
  assertRawBindable(a, "mulScalarInPlace");
  mulScalarKernel.dispatch({ data: a.buffer }, { scalar, size: a.size });
}
