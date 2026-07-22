/**
 * Fused kernel wrappers and I/O ops: optStep, unscaleGrad, fusedCrossEntropy,
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
import type { OptStepBatchItem, OptStepBatchResult } from "../../types";
import { dispatchOptStep as dispatchOptStepKernel } from "../opt-step-kernel";
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
// Fused Optimizer Step (Adam/Lion/SGD — spec-driven, R5b de-naming)
// ============================================================================

/**
 * Variadic in-place aliasing guard (R5b — the variadic buffer-lifetime class).
 * Given the read_write set `[param, ...states]` and the read-only `gradBuf`,
 * return one buffer per rw tensor such that all rw buffers are distinct from
 * `gradBuf` AND from each other (WebGPU forbids rw↔rw and rw↔read aliasing),
 * copying via `copyTo` on collision. In the real-training case (distinct
 * persistent buffers) NO copy fires — byte-identical to the prior param/m/v
 * hand-unrolled guard.
 */
function distinctRwBuffers(
  rwTensors: WebGPUTensor[],
  gradBuf: GPUBuffer,
  copyTo: (src: GPUBuffer) => GPUBuffer,
): GPUBuffer[] {
  const out: GPUBuffer[] = [];
  for (const t of rwTensors) {
    let b = t.buffer;
    if (b === gradBuf || out.includes(b)) b = copyTo(b);
    out.push(b);
  }
  return out;
}

/**
 * Variadic in-place ownership transfer (R5b). For each read_write tensor
 * (param + optimizer states), decRef its buffer and neuter the old destroy —
 * the fused kernel wrote the update in place and the fresh result BackendTensor
 * now owns the buffer. The generalization of the hard-coded param/m/v transfer
 * to per-optimizer state arity. Idempotent-safe: only decRefs owners.
 */
function transferInPlaceOwnership(rwTensors: WebGPUTensor[]): void {
  for (const t of rwTensors) {
    if (t.ownsBuffer) bufferPool.decRef(t.buffer);
    t.destroy = () => {};
  }
}

/**
 * Per-item fused optimizer step implementation. Does NOT call flushSharedEncoder
 * — the caller is responsible for ensuring previously-recorded passes that read
 * the param/state buffers have been submitted before invoking this. Use
 * `optStep` (which flushes once per call) for standalone calls; use
 * `optStepBatch` (which flushes once per batch) for grouped calls. `states` and
 * `scalars` are variadic per the config's spec (Adam: states [m,v], scalars [bc,lr]).
 */
function optStepInner(
  grad: BackendTensor,
  param: BackendTensor,
  states: BackendTensor[],
  scalars: BackendTensor[],
  config: import("../../types").OptStepConfig,
): { param: BackendTensor; states: BackendTensor[] } {
  let _st = profileSubOpBegin();
  const gradT = asContiguous(grad);
  const paramT = asContiguous(param);
  const stateTs = states.map((s) => asContiguous(s));
  // scalars (bc/lr) are read-only shared inputs — resolve their buffers.
  const scalarTs = scalars.map((s) => asContiguous(s));
  profileSubOpEnd("optStep.ensureContig", _st);

  // Evict stale f16 cache entry for the old param buffer before dispatch.
  // Use deferred destruction since the old f16 buffer may still be referenced
  // by cast() results from the forward pass in the same command encoder batch.
  _st = profileSubOpBegin();
  const oldF16 = f16WeightCache.get(paramT.buffer);
  if (oldF16) {
    bufferPool.deferredDestroy(oldF16, oldF16.size);
    f16WeightCache.delete(paramT.buffer);
  }
  profileSubOpEnd("optStep.f16evict", _st);

  // Re-protect the optimizer's input buffers in the write set.
  trackSharedEncoderWrite(paramT.buffer);
  for (const s of stateTs) trackSharedEncoderWrite(s.buffer);
  trackSharedEncoderWrite(gradT.buffer);
  for (const s of scalarTs) trackSharedEncoderWrite(s.buffer);

  // WebGPU forbids binding the same buffer as read_write at multiple
  // binding points and forbids read_write ↔ read aliasing. param and every
  // state slot are read_write, grad is read — so each rw buffer must be
  // distinct from grad AND from every other rw buffer. Pool recycling can
  // cause collisions.
  //
  // R5b — the variadic in-place buffer-lifetime class: the guard loops over
  // the read_write set `[param, ...states]` instead of hard-coding param/m/v,
  // so a per-optimizer state arity (Adam [m,v], Lion [m], SGD+mom [v], SGD [])
  // is handled by the same code. With distinct persistent buffers (the real-
  // training case) NO copies fire — byte-identical to the prior hand-unroll.
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
  const rwTensors: WebGPUTensor[] = [paramT, ...stateTs];
  const rwBufs = distinctRwBuffers(rwTensors, gradBuf, copyTo);
  const paramBuf = rwBufs[0];
  const stateBufs = rwBufs.slice(1);

  const emitF16 = config.emitF16 ?? false;
  const infFlagBuf = (config.infFlagBuffer as GPUBuffer | null) ?? null;
  const numElements = gradT.size;

  const result = dispatchOptStepKernel(
    gradBuf,
    paramBuf,
    stateBufs,
    scalarTs.map((t) => t.buffer),
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
  // scalars (bc/lr) are persistent optimizer state owned by the caller — only
  // free the contiguous COPY if asContiguous made one (cleanupContiguous no-ops
  // when the copy === original). The originals are never destroyed here.
  for (let i = 0; i < scalars.length; i++)
    cleanupContiguous([scalars[i], scalarTs[i]]);

  // Cache the f16 param buffer (keyed by the param buffer, same as before)
  if (result.paramF16Buffer) {
    f16WeightCache.set(result.paramBuffer, result.paramF16Buffer);
  }

  // param + states are all updated in-place — transfer ownership from the old
  // BackendTensors to the results (noop their destroy, decRef their buffers).
  _st = profileSubOpBegin();
  transferInPlaceOwnership(rwTensors);
  const ret = {
    param: createTensor(
      paramT.shape,
      result.paramBuffer,
      undefined,
      0,
      paramT.dtype,
    ),
    states: stateTs.map((st, i) =>
      createTensor(st.shape, result.stateBuffers[i], undefined, 0, st.dtype),
    ),
  };
  profileSubOpEnd("optStep.createTensor", _st);
  return ret;
}

export async function optStep(
  grad: BackendTensor,
  param: BackendTensor,
  states: BackendTensor[],
  scalars: BackendTensor[],
  config: import("../../types").OptStepConfig,
): Promise<{ param: BackendTensor; states: BackendTensor[] }> {
  // Suppress memory limit checks for the entire optimizer step.
  // During optimizer steps, old param/state buffers haven't been released yet
  // while new output buffers are allocated — temporary 2x peak is expected.
  gpuMemoryTracker.suppressLimitCheck();
  try {
    // Flush shared encoder before dispatching the optimizer kernel.
    // The forward/backward pass may have used param buffers as read-only
    // inputs in compute passes. The kernel binds param as read_write,
    // which conflicts within the same command encoder synchronization scope.
    // Flushing ensures prior passes are submitted before the in-place writes.
    flushSharedEncoder();
    return optStepInner(grad, param, states, scalars, config);
  } finally {
    gpuMemoryTracker.unsuppressLimitCheck();
  }
}

/**
 * Batched fused optimizer step. Groups same-element-count items into packed
 * dispatches (one kernel per size class instead of one per param), with per-item
 * fallback via `optStepInner` for any items not handled by packing. The shared
 * encoder is flushed exactly ONCE for the whole batch, not per item.
 *
 * Returns results in the same order as `items`.
 */
export function optStepBatch(items: OptStepBatchItem[]): OptStepBatchResult[] {
  if (items.length === 0) return [];
  gpuMemoryTracker.suppressLimitCheck();
  try {
    // Single flush before any item — all items share one logical batch.
    flushSharedEncoder();

    const results: Array<OptStepBatchResult | null> = new Array(
      items.length,
    ).fill(null);

    // Try packed dispatch for groups of size ≥ 2 with the same element count.
    // dispatchPackedOptimizer requires items.length > 1 to even attempt.
    // TORCHLETTE_PACKED_OPT=0: measurement kill switch (per-item dispatches,
    // still one flush per batch) — quantifies what packing is worth.
    if (items.length > 1 && ENV.TORCHLETTE_PACKED_OPT !== "0") {
      // Pre-resolve contiguous tensors per item, evict f16 caches, build the
      // packed item descriptors. We hold onto the contiguous tensors so the
      // fallback path can claim ownership uniformly with optStepInner.
      const contig: Array<{
        gradT: WebGPUTensor;
        paramT: WebGPUTensor;
        stateTs: WebGPUTensor[];
      }> = items.map((item) => ({
        gradT: asContiguous(item.grad),
        paramT: asContiguous(item.param),
        stateTs: item.states.map((s) => asContiguous(s)),
      }));

      for (const { paramT } of contig) {
        const oldF16 = f16WeightCache.get(paramT.buffer);
        if (oldF16) {
          bufferPool.deferredDestroy(oldF16, oldF16.size);
          f16WeightCache.delete(paramT.buffer);
        }
      }

      // Packed roles: grad · param · state slots (the chunked buffers). Scalar-
      // DATA inputs (bc/lr) are 1-/2-element shared bindings — NOT scatter/
      // gathered; bound ONCE via the dispatch closure.
      const packedItems: PackedOptimizerItem[] = contig.map((c) => ({
        buffers: [c.gradT.buffer, c.paramT.buffer, ...c.stateTs.map((t) => t.buffer)],
        numElements: c.gradT.size,
      }));

      // Packed kernel uses the FIRST item's config AND the FIRST item's scalars.
      // This is SOUND because the executor's opt-batch grouping key breaks the
      // batch on scalar-tensor identity: every item in one batch shares the same
      // static config AND the same persistent scalar tensors by construction.
      const config = items[0].config;
      const infFlagBuffer = (config.infFlagBuffer as GPUBuffer | null) ?? null;
      const scalarBufs = items[0].scalars.map(
        (s) => (asContiguous(s) as WebGPUTensor).buffer,
      );
      const nState = config.stateSlots.length;
      // gather param(1) + state slots (2..1+nState) back into original buffers.
      const gatherIndices = Array.from({ length: 1 + nState }, (_, i) => 1 + i);

      const handled = dispatchPackedOptimizer({
        items: packedItems,
        gatherIndices,
        dispatch(packed, totalElements) {
          dispatchOptStepKernel(
            packed[0],
            packed[1],
            packed.slice(2, 2 + nState),
            scalarBufs,
            totalElements,
            config,
            false,
            infFlagBuffer,
          );
        },
        label: "packedOptStep",
      });

      // For items handled by packed dispatch: param/state buffers were updated
      // in place via the gather copy. Transfer ownership from the input BTs
      // to fresh BTs wrapping the same buffers — same pattern optStepInner
      // uses for its single-item path.
      for (const i of handled) {
        const { gradT, paramT, stateTs } = contig[i];
        const item = items[i];

        cleanupContiguous([item.grad, gradT]);

        transferInPlaceOwnership([paramT, ...stateTs]);

        results[i] = {
          param: createTensor(
            paramT.shape,
            paramT.buffer,
            undefined,
            0,
            paramT.dtype,
          ),
          states: stateTs.map((st) =>
            createTensor(st.shape, st.buffer, undefined, 0, st.dtype),
          ),
        };
      }
    }

    // Fallback path: for items not handled by packed dispatch (or all items
    // when items.length === 1), call optStepInner per item. The encoder was
    // already flushed at the top, so optStepInner skipping its own flush is
    // exactly what we want.
    for (let i = 0; i < items.length; i++) {
      if (results[i] !== null) continue;
      const item = items[i];
      results[i] = optStepInner(
        item.grad,
        item.param,
        item.states,
        item.scalars,
        item.config,
      );
    }

    return results as OptStepBatchResult[];
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
