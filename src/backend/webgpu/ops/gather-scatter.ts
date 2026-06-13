/**
 * Gather and scatter ops with integrated chunking for large tensors.
 *
 * Each op handles both the normal (direct) and chunked paths internally,
 * branching at the dispatch level after shared setup (shapes, strides, shader).
 */

import { sizeOf } from "../../../core/shape";
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
import { recordedCopyBufferToBuffer } from "../../../executor/compiled-plan";
import { computeDimChunkLayout } from "../chunked-dispatch";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { requireContext } from "../gpu-context";
import type { GPUBufferBinding } from "../gpu-types";
import { asGPUTensor, GPUBufferUsage } from "../gpu-types";
import { compute2DDispatch, dtypeBytes, F32_BYTES, WORKGROUP_SIZE } from "../shape-utils";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import { createTensor, createTrackedBuffer } from "../tensor";
import { ensureContiguous } from "./views";
import {
  chunkedGatherTileIR,
  chunkedScatterAddTileIR,
  gatherTileIR,
  scatterAddTileIR,
} from "./ops-tile-ir";

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
  const inputSizeBytes = sizeOf(inputShape) * F32_BYTES;
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  if (inputSizeBytes > maxBindingSize) return null; // chunked
  if (indexDtype !== "f32" && indexDtype !== "i32" && indexDtype !== "u32") {
    return null;
  }
  const outShape = indexShape.slice();
  const outSize = sizeOf(outShape);
  const dispatch = compute2DDispatch(Math.ceil(outSize / WORKGROUP_SIZE));
  const use2D = dispatch.y > 1;
  const shader = gatherTileIR(inputShape, indexShape, dim, indexDtype);
  const key = `gather:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${indexDtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  return {
    key,
    shader,
    paramsData: params(outSize),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
    outShape,
    outputBytes: outSize * F32_BYTES,
  };
}

export function gather(
  a: BackendTensor,
  index: BackendTensor,
  options: GatherOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensorA = asGPUTensor(a);
  // Index is read as a flat contiguous array; materialize any view first.
  const tensorIndex = ensureContiguous(asGPUTensor(index));
  const { dim } = options;
  const inputShape = tensorA.shape;
  const indexShape = tensorIndex.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("gather: dimension out of range");
  }

  // Determine chunking
  const inputSizeBytes = tensorA.size * F32_BYTES;
  const deviceLimits = ctx.device.limits;
  const maxBindingSize =
    deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const chunked = inputSizeBytes > maxBindingSize;

  // Shared setup
  const outShape = indexShape.slice();
  const outSize = sizeOf(outShape);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Index dtype: gather kernel reads indices as native i32/u32/f32 to avoid
  // round-trip casts at call sites (e.g. i32 token ids from embedding).
  const indexDtype = tensorIndex.dtype;
  if (
    indexDtype !== "f32" &&
    indexDtype !== "i32" &&
    indexDtype !== "u32"
  ) {
    throw new Error(
      `gather: index dtype must be f32/i32/u32, got ${indexDtype}`,
    );
  }

  // --- Direct path — single-sourced with the stream generator. ---
  if (!chunked) {
    const plan = planGatherDirect(inputShape, indexShape, dim, indexDtype);
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
    return createTensor(plan.outShape, outBuffer);
  }

  // Generate shader for the chunked path.
  const code = chunkedGatherTileIR(inputShape, indexShape, dim, indexDtype);
  const pipelineKey = `gatherChunked:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${indexDtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, pipelineKey, code);

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

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * F32_BYTES,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

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

  return createTensor(outShape, outBuffer);
}

// ============================================================================
// ScatterAdd
// ============================================================================

export function scatterAdd(
  a: BackendTensor,
  index: BackendTensor,
  src: BackendTensor,
  options: ScatterAddOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensorA = asGPUTensor(a);
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

  // Determine chunking
  const outputSizeBytes = tensorA.size * F32_BYTES;
  const deviceLimits = ctx.device.limits;
  const maxBindingSize =
    deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const chunked = outputSizeBytes > maxBindingSize;

  // Shared setup
  const outShape = inputShape.slice();
  const outSize = sizeOf(outShape);
  const srcSize = sizeOf(tensorSrc.shape);
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Index dtype: scatterAdd kernel reads indices as native i32/u32/f32.
  const indexDtype = tensorIndex.dtype;
  if (
    indexDtype !== "f32" &&
    indexDtype !== "i32" &&
    indexDtype !== "u32"
  ) {
    throw new Error(
      `scatterAdd: index dtype must be f32/i32/u32, got ${indexDtype}`,
    );
  }

  // Generate shader via tile-IR (both direct and chunked paths)
  const code = chunked
    ? chunkedScatterAddTileIR(inputShape, tensorSrc.shape, dim, indexDtype)
    : scatterAddTileIR(inputShape, tensorSrc.shape, dim, indexDtype);

  const keyPrefix = chunked ? "scatterAddChunked" : "scatterAdd";
  const pipelineKey = `${keyPrefix}:${inputShape.join(",")}:${tensorSrc.shape.join(",")}:${dim}:${indexDtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, pipelineKey, code);

  // Copy input to output (both paths need this)
  const outBuffer = chunked
    ? createTrackedBuffer(ctx.device, {
        size: outSize * F32_BYTES,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      })
    : resolveOutputBuffer(ctx.device, outSize * F32_BYTES, [
        tensorA.buffer,
        tensorIndex.buffer,
        tensorSrc.buffer,
      ]);

  {
    // scatterAdd accumulates: out = copy(a); out[idx] += src. The kernel ADDS
    // onto the copied content, so this copy is INTRA-PLAN and MUST be replayed
    // by the compiled plan — otherwise `out` keeps the previous replay's result
    // and the grad inflates +1x per step (embedding-grad accumulation bug). Use
    // the recorded-copy helper. (Chunked path uses createTrackedBuffer for out,
    // which is untracked → the executor invalidates the compiled plan → lowered
    // path, so the standalone branch needs no recording.)
    const enc = getSharedEncoderInstance();
    if (enc) {
      recordedCopyBufferToBuffer(enc, tensorA.buffer, 0, outBuffer, 0, outSize * F32_BYTES);
    } else {
      const cmdEnc = ctx.device.createCommandEncoder();
      cmdEnc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * F32_BYTES);
      submitOrCollect(cmdEnc.finish());
    }
  }

  // --- Direct path ---
  if (!chunked) {
    const uniformBuf = createParamsBuffer(ctx.device, params(srcSize));
    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
      tensorIndex.buffer,
      tensorSrc.buffer,
      outBuffer,
      uniformBuf,
    ]);
    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseParamsBuffer(uniformBuf);
    return createTensor(outShape, outBuffer);
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
  const gpuTensors = tensors.map((t) => asGPUTensor(t));
  const bytesPerElement = dtypeBytes(gpuTensors[0].dtype);

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
    for (let o = 0; o < outerSize; o++) {
      const srcByteOffset = o * tDimSize * innerSize * bytesPerElement;
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

  return createTensor(outShape, outBuffer, undefined, 0, gpuTensors[0].dtype);
}
