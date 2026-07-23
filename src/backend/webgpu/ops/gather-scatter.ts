/**
 * Gather and scatter ops with integrated chunking for large tensors.
 *
 * Each op handles both the normal (direct) and chunked paths internally,
 * branching at the dispatch level after shared setup (shapes, strides, shader).
 */

import { sizeOf } from "../../../core/shape";
import { recordedCopyBufferToBuffer } from "../../../executor/compiled-plan";
import type {
  BackendTensor,
  CatOptions,
  GatherOptions,
  ScatterAddOptions,
} from "../../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  profiledCreateBindGroup,
  releaseParamsBuffer,
} from "../bind-group-cache";
import { resolveOutputBuffer } from "../buffer-arena";
import { destroyCopy } from "../buffer-pool";
import { computeDimChunkLayout } from "../chunked-dispatch";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { requireContext } from "../gpu-context";
import type { GPUBufferBinding } from "../gpu-types";
import { asGPUTensor, GPUBufferUsage } from "../gpu-types";
import {
  compute2DDispatch,
  dtypeBytes,
  F32_BYTES,
  WORKGROUP_SIZE,
  DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE,
} from "../shape-utils";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import { createTensor, createTrackedBuffer } from "../tensor";
import {
  chunkedGatherTileIR,
  chunkedScatterAddTileIR,
  gatherTileIR,
  scatterAddTileIR,
} from "./ops-tile-ir";
import { ensureContiguous } from "./views";

// ============================================================================
// Gather
// ============================================================================

/**
 * Stage-4 plan/encode for the DIRECT gather path (no chunking). Geometry is
 * derivable from a.shape + index.shape + dim + index dtype — the stream
 * generator computes it from the node post-hoc (a = embedding table, index =
 * token ids; both live at lowering, contiguous). Returns null on the chunked
 * route (table > maxBindingSize). Binding order [a, index, out, params];
 * output is resolveOutputBuffer (allocKind 0, inputs [a, index]).
 */
export function planGatherDirect(
  inputShape: number[],
  indexShape: number[],
  dim: number,
  indexDtype: DType,
  dataDtype: DType = "f32",
): {
  key: string;
  shader: string;
  paramsData: Uint32Array;
  dispatchX: number;
  dispatchY: number;
  outShape: number[];
  outputBytes: number;
} | null {
  const ctx = requireContext();
  // gather is dtype-preserving: out dtype == source (a) dtype. Buffer sizing
  // and the maxBinding check must use the SOURCE element size, not a hardcoded
  // F32 (an f16 table is half the bytes — the #59 fix).
  const elemBytes = dtypeBytes(dataDtype);
  const inputSizeBytes = sizeOf(inputShape) * elemBytes;
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  if (inputSizeBytes > maxBindingSize) return null; // chunked
  if (indexDtype !== "f32" && indexDtype !== "i32" && indexDtype !== "u32") {
    return null;
  }
  const outShape = indexShape.slice();
  const outSize = sizeOf(outShape);
  const dispatch = compute2DDispatch(Math.ceil(outSize / WORKGROUP_SIZE));
  const shader = gatherTileIR(inputShape, indexShape, dim, indexDtype, dataDtype);
  return {
    // Pipeline cache key IS the WGSL text (tile-dispatch's canonical
    // `getPipeline(ctx, wgsl, wgsl)`). A structural key that bakes the shader
    // in a separate string can silently drift from codegen and serve a stale
    // pipeline (the single-source-at-seams rule). `use2D` is folded into the
    // shader by gatherTileIR, so the code already distinguishes the variants.
    key: shader,
    shader,
    paramsData: params(outSize),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
    outShape,
    outputBytes: outSize * elemBytes,
  };
}

export function gather(
  a: BackendTensor,
  index: BackendTensor,
  options: GatherOptions,
): BackendTensor {
  const ctx = requireContext();
  // Input is indexed as a flat contiguous array from element 0 (both the
  // direct kernel and the chunked path's byte-offset slicing assume it);
  // materialize offset/strided views first (offset-view class, task #58).
  const tensorAOrig = asGPUTensor(a);
  const tensorA = ensureContiguous(tensorAOrig);
  // Index is read as a flat contiguous array; materialize any view first.
  const tensorIndex = ensureContiguous(asGPUTensor(index));
  const { dim } = options;
  const inputShape = tensorA.shape;
  const indexShape = tensorIndex.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("gather: dimension out of range");
  }

  // gather is dtype-preserving (out dtype == source dtype). All buffer sizing,
  // the maxBinding check, and the chunk layout below must use the SOURCE
  // element size — an f16 table is half the bytes (the #59 fix; a f16 gemma
  // embedding table both halves residency AND stops chunking sooner).
  const dataDtype = tensorA.dtype;
  const elemBytes = dtypeBytes(dataDtype);

  // Determine chunking
  const inputSizeBytes = tensorA.size * elemBytes;
  const deviceLimits = ctx.device.limits;
  const maxBindingSize =
    deviceLimits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  const chunked = inputSizeBytes > maxBindingSize;

  // Shared setup
  const outShape = indexShape.slice();
  const outSize = sizeOf(outShape);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);

  // Index dtype: gather kernel reads indices as native i32/u32/f32 to avoid
  // round-trip casts at call sites (e.g. i32 token ids from embedding).
  const indexDtype = tensorIndex.dtype;
  if (indexDtype !== "f32" && indexDtype !== "i32" && indexDtype !== "u32") {
    throw new Error(
      `gather: index dtype must be f32/i32/u32, got ${indexDtype}`,
    );
  }

  // --- Direct path — single-sourced with the stream generator. ---
  if (!chunked) {
    const plan = planGatherDirect(
      inputShape,
      indexShape,
      dim,
      indexDtype,
      dataDtype,
    );
    if (!plan) throw new Error("gather: direct path but planGatherDirect null");
    const pipeline = getPipeline(ctx, plan.key, plan.shader);
    const outBuffer = resolveOutputBuffer(ctx.device, plan.outputBytes, [
      tensorA.buffer,
      tensorIndex.buffer,
    ]);
    const uniformBuf = createParamsBuffer(ctx.device, plan.paramsData);
    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
      tensorA.buffer,
      tensorIndex.buffer,
      outBuffer,
      uniformBuf,
    ]);
    dispatchComputePass(pipeline, bindGroup, plan.dispatchX, plan.dispatchY);
    releaseParamsBuffer(uniformBuf);
    if (tensorA !== tensorAOrig) destroyCopy(tensorA);
    return createTensor(plan.outShape, outBuffer, undefined, 0, dataDtype);
  }

  // Generate shader for the chunked path.
  const code = chunkedGatherTileIR(
    inputShape,
    indexShape,
    dim,
    indexDtype,
    dataDtype,
  );
  // Key IS the code (see planGatherDirect) — no structural key to drift.
  const pipeline = getPipeline(ctx, code, code);

  // --- Chunked path ---
  const dimSize = inputShape[dim];
  const elementsPerSlice = sizeOf(inputShape.slice(dim + 1));
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;
  const layout = computeDimChunkLayout(
    dimSize,
    elementsPerSlice,
    maxBindingSize,
    minAlignment,
    elemBytes,
  );

  // Route through resolveOutputBuffer (like the direct path above) so the
  // output is registered with the compiled-plan recorder. createTrackedBuffer
  // (raw device.createBuffer) is invisible to the recorder, which throws under
  // any compiled-plan/capture recording (e.g. a >2GB embedding gather inside a
  // repeated decode template — Gemma-2's 256k-vocab table forces this path).
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * elemBytes, [
    tensorA.buffer,
    tensorIndex.buffer,
  ]);

  for (let chunk = 0; chunk < layout.numChunks; chunk++) {
    const chunkStart = chunk * layout.maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + layout.maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * layout.bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * layout.bytesPerSlice;

    const uniformBuffer = createParamsBuffer(
      ctx.device,
      params(outSize, chunkStart, chunkEnd, 0),
    );
    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: tensorA.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          } as GPUBufferBinding,
        },
        { binding: 1, resource: { buffer: tensorIndex.buffer } },
        { binding: 2, resource: { buffer: outBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });
    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseParamsBuffer(uniformBuffer);
  }

  if (tensorA !== tensorAOrig) destroyCopy(tensorA);
  return createTensor(outShape, outBuffer, undefined, 0, dataDtype);
}

// ============================================================================
// ScatterAdd
// ============================================================================

/**
 * Stage-4 plan/encode for the DIRECT scatterAdd path. Geometry from
 * a.shape (=out) + src.shape + dim + index dtype. The op is ALLOC(out,
 * allocKind 0, [a,index,src]) + COPY(a→out) + DISPATCH[index,src,out,params]
 * (the copy seeds the accumulator — recorded, replayed). Null on chunked.
 */
export function planScatterAddDirect(
  inputShape: number[],
  srcShape: number[],
  dim: number,
  indexDtype: DType,
): {
  key: string;
  shader: string;
  paramsData: Uint32Array;
  dispatchX: number;
  dispatchY: number;
  outShape: number[];
  outputBytes: number;
} | null {
  const ctx = requireContext();
  const outSize = sizeOf(inputShape);
  if (
    outSize * F32_BYTES >
    (ctx.device.limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE)
  ) {
    return null; // chunked
  }
  if (indexDtype !== "f32" && indexDtype !== "i32" && indexDtype !== "u32") {
    return null;
  }
  const srcSize = sizeOf(srcShape);
  const dispatch = compute2DDispatch(Math.ceil(srcSize / WORKGROUP_SIZE));
  const shader = scatterAddTileIR(inputShape, srcShape, dim, indexDtype);
  return {
    // Key IS the WGSL text (see planGatherDirect) — single-source at the seam.
    key: shader,
    shader,
    paramsData: params(srcSize),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
    outShape: inputShape.slice(),
    outputBytes: outSize * F32_BYTES,
  };
}

export function scatterAdd(
  a: BackendTensor,
  index: BackendTensor,
  src: BackendTensor,
  options: ScatterAddOptions,
): BackendTensor {
  const ctx = requireContext();
  // The accumulator seed is a flat buffer copy of `a` from byte 0 — an
  // offset/strided view of `a` must be materialized first (offset-view
  // class, task #58).
  const tensorA = ensureContiguous(asGPUTensor(a));
  // Kernel reads `src` and `indices` as flat contiguous arrays. If the caller
  // passes a broadcasted/strided view (common when gradients come in from an
  // expand()/sum-backward path), force materialization first.
  const tensorIndex = ensureContiguous(asGPUTensor(index));
  const tensorSrc = ensureContiguous(asGPUTensor(src));
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("scatterAdd: dimension out of range");
  }

  // scatterAdd accumulates via an f32 atomic CAS loop (atomicAddF32 over
  // atomic<u32>). WebGPU has NO f16/bf16 atomic, so the kernel is inherently
  // f32-only; the accumulator and src bytes are sized as f32. Embedding
  // BACKWARD (this op's only caller) runs its grads in f32 under autocast, so
  // f32 is the contract. Guard LOUDLY rather than silently mis-size an f16
  // input (the #59 seam — extending to f16 here would need f16 atomics we
  // don't have; out of scope, not silently wrong).
  if (tensorA.dtype !== "f32" || tensorSrc.dtype !== "f32") {
    throw new Error(
      `scatterAdd: f32-only (atomic CAS has no f16 equivalent); got a=${tensorA.dtype} src=${tensorSrc.dtype}`,
    );
  }

  // Determine chunking
  const outputSizeBytes = tensorA.size * F32_BYTES;
  const deviceLimits = ctx.device.limits;
  const maxBindingSize =
    deviceLimits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  const chunked = outputSizeBytes > maxBindingSize;

  // Shared setup
  const outShape = inputShape.slice();
  const outSize = sizeOf(outShape);
  const srcSize = sizeOf(tensorSrc.shape);
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);

  // Index dtype: scatterAdd kernel reads indices as native i32/u32/f32.
  const indexDtype = tensorIndex.dtype;
  if (indexDtype !== "f32" && indexDtype !== "i32" && indexDtype !== "u32") {
    throw new Error(
      `scatterAdd: index dtype must be f32/i32/u32, got ${indexDtype}`,
    );
  }

  // --- Direct path — single-sourced with the stream generator. ---
  if (!chunked) {
    const plan = planScatterAddDirect(
      inputShape,
      tensorSrc.shape,
      dim,
      indexDtype,
    );
    if (!plan)
      throw new Error("scatterAdd: direct path but planScatterAddDirect null");
    const pipeline = getPipeline(ctx, plan.key, plan.shader);
    const outBuffer = resolveOutputBuffer(ctx.device, plan.outputBytes, [
      tensorA.buffer,
      tensorIndex.buffer,
      tensorSrc.buffer,
    ]);
    // scatterAdd accumulates: out = copy(a); out[idx] += src. The kernel ADDS
    // onto the copied content, so this copy is INTRA-PLAN and MUST be replayed
    // by the compiled plan — otherwise `out` keeps the previous replay's result
    // and the grad inflates +1x per step (embedding-grad accumulation bug).
    const enc = getSharedEncoderInstance();
    if (enc) {
      recordedCopyBufferToBuffer(
        enc,
        tensorA.buffer,
        0,
        outBuffer,
        0,
        plan.outputBytes,
      );
    } else {
      const cmdEnc = ctx.device.createCommandEncoder();
      cmdEnc.copyBufferToBuffer(
        tensorA.buffer,
        0,
        outBuffer,
        0,
        plan.outputBytes,
      );
      submitOrCollect(cmdEnc.finish());
    }
    const uniformBuf = createParamsBuffer(ctx.device, plan.paramsData);
    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
      tensorIndex.buffer,
      tensorSrc.buffer,
      outBuffer,
      uniformBuf,
    ]);
    dispatchComputePass(pipeline, bindGroup, plan.dispatchX, plan.dispatchY);
    releaseParamsBuffer(uniformBuf);
    return createTensor(plan.outShape, outBuffer);
  }

  // Generate shader for the chunked path.
  const code = chunkedScatterAddTileIR(
    inputShape,
    tensorSrc.shape,
    dim,
    indexDtype,
  );
  // Key IS the code (see planScatterAddDirect) — no structural key to drift.
  const pipeline = getPipeline(ctx, code, code);

  // Chunked uses createTrackedBuffer for out (untracked → the executor
  // invalidates the compiled plan → lowered path), so the seed copy below
  // needs no recording.
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * F32_BYTES,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  {
    const enc = getSharedEncoderInstance();
    if (enc) {
      recordedCopyBufferToBuffer(
        enc,
        tensorA.buffer,
        0,
        outBuffer,
        0,
        outSize * F32_BYTES,
      );
    } else {
      const cmdEnc = ctx.device.createCommandEncoder();
      cmdEnc.copyBufferToBuffer(
        tensorA.buffer,
        0,
        outBuffer,
        0,
        outSize * F32_BYTES,
      );
      submitOrCollect(cmdEnc.finish());
    }
  }

  // --- Chunked path ---
  const dimSize = inputShape[dim];
  const elementsPerSlice = sizeOf(inputShape.slice(dim + 1));
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;
  const layout = computeDimChunkLayout(
    dimSize,
    elementsPerSlice,
    maxBindingSize,
    minAlignment,
  );

  for (let chunk = 0; chunk < layout.numChunks; chunk++) {
    const chunkStart = chunk * layout.maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + layout.maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * layout.bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * layout.bytesPerSlice;

    const uniformBuffer = createParamsBuffer(
      ctx.device,
      params(srcSize, chunkStart, chunkEnd, 0),
    );
    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensorIndex.buffer } },
        { binding: 1, resource: { buffer: tensorSrc.buffer } },
        {
          binding: 2,
          resource: {
            buffer: outBuffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          } as GPUBufferBinding,
        },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });
    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseParamsBuffer(uniformBuffer);
  }

  return createTensor(outShape, outBuffer);
}

/**
 * Concatenate tensors along an existing dimension.
 * Uses copyBufferToBuffer for efficient GPU-side data movement.
 */
export function cat(
  tensors: BackendTensor[],
  options: CatOptions,
): BackendTensor {
  if (tensors.length === 0) throw new Error("cat: empty tensor list");
  if (tensors.length === 1) return tensors[0];

  const ctx = requireContext();
  // The copy offsets below are computed from SHAPES assuming each input is a
  // contiguous block. A view input with an element OFFSET (narrow) is folded
  // into srcByteOffset when 4-byte aligned; non-contiguous strides (or
  // unaligned offsets, e.g. odd-element f16) are materialized first
  // (offset-view class, task #58).
  const originals = tensors.map((t) => asGPUTensor(t));
  const bytesPerElement = dtypeBytes(originals[0].dtype);
  const gpuTensors = originals.map((t) =>
    t.isContiguous && (t.offset * bytesPerElement) % 4 === 0
      ? t
      : ensureContiguous(t),
  );

  // Compute output shape
  const dim = options.dim;
  const rank = gpuTensors[0].shape.length;
  const outShape = gpuTensors[0].shape.slice();
  for (let i = 1; i < gpuTensors.length; i++) {
    outShape[dim] += gpuTensors[i].shape[dim];
  }
  const outSize = sizeOf(outShape);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    gpuTensors.map((t) => t.buffer),
  );

  // Compute outer/inner sizes for strided copy
  const innerSize = dim === rank - 1 ? 1 : sizeOf(outShape.slice(dim + 1));
  const outerSize = dim === 0 ? 1 : sizeOf(outShape.slice(0, dim));
  const outDimSize = outShape[dim];

  const enc = getSharedEncoderInstance();
  let dimOffset = 0;
  for (const t of gpuTensors) {
    const tDimSize = t.shape[dim];
    const copyBytes = tDimSize * innerSize * bytesPerElement;
    const tBaseByteOffset = t.offset * bytesPerElement;
    for (let o = 0; o < outerSize; o++) {
      const srcByteOffset =
        tBaseByteOffset + o * tDimSize * innerSize * bytesPerElement;
      const dstByteOffset =
        (o * outDimSize + dimOffset) * innerSize * bytesPerElement;
      if (enc) {
        // INTRA-PLAN copies (the assembled tensor is read by later dispatches
        // in the same plan) — must be recorded or compiled replays reuse the
        // destination's stale contents (e.g. the foreach optimizer's packed
        // grads freezing at the recording step's values).
        recordedCopyBufferToBuffer(
          enc,
          t.buffer,
          srcByteOffset,
          outBuffer,
          dstByteOffset,
          copyBytes,
        );
      } else {
        const cmdEnc = ctx.device.createCommandEncoder();
        cmdEnc.copyBufferToBuffer(
          t.buffer,
          srcByteOffset,
          outBuffer,
          dstByteOffset,
          copyBytes,
        );
        submitOrCollect(cmdEnc.finish());
      }
    }
    dimOffset += tDimSize;
  }

  for (let i = 0; i < gpuTensors.length; i++) {
    if (gpuTensors[i] !== originals[i]) destroyCopy(gpuTensors[i]);
  }
  return createTensor(outShape, outBuffer, undefined, 0, gpuTensors[0].dtype);
}
