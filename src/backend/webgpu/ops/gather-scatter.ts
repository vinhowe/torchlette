/**
 * Gather and scatter ops with integrated chunking for large tensors.
 *
 * Each op handles both the normal (direct) and chunked paths internally,
 * branching at the dispatch level after shared setup (shapes, strides, shader).
 */
import type { BackendTensor, CatOptions, GatherOptions, ScatterAddOptions } from "../../types";
import { sizeOf } from "../../../core/shape";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage, asGPUTensor } from "../gpu-types";
import { WORKGROUP_SIZE, compute2DDispatch } from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import { resolveOutputBuffer } from "../buffer-arena";
import { computeDimChunkLayout } from "../chunked-dispatch";
import {
  cachedCreateBindGroup, profiledCreateBindGroup,
  createParamsBuffer, releaseParamsBuffer,
  createUniformBuffer, releaseUniformBuffer,
  params4,
} from "../bind-group-cache";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import { gatherTileIR, scatterAddTileIR, chunkedGatherTileIR, chunkedScatterAddTileIR } from "./ops-tile-ir";

/** Local type alias for GPU buffer binding descriptors with offset/size. */
type GPUBufferBinding = { buffer: GPUBuffer; offset?: number; size?: number };

// ============================================================================
// Gather
// ============================================================================

export function gather(
  a: BackendTensor,
  index: BackendTensor,
  options: GatherOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensorA = asGPUTensor(a);
  const tensorIndex = asGPUTensor(index);
  const { dim } = options;
  const inputShape = tensorA.shape;
  const indexShape = tensorIndex.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("gather: dimension out of range");
  }

  // Determine chunking
  const inputSizeBytes = tensorA.size * 4;
  const deviceLimits = ctx.device.limits;
  const maxBindingSize = deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const chunked = inputSizeBytes > maxBindingSize;

  // Shared setup
  const outShape = indexShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Generate shader via tile-IR (both direct and chunked paths)
  const code = chunked
    ? chunkedGatherTileIR(inputShape, indexShape, dim)
    : gatherTileIR(inputShape, indexShape, dim);

  const keyPrefix = chunked ? "gatherChunked" : "gather";
  const pipelineKey = `${keyPrefix}:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, pipelineKey, code);

  // --- Direct path ---
  if (!chunked) {
    const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensorA.buffer, tensorIndex.buffer]);
    const uniformBuffer = createUniformBuffer(ctx.device, outSize);
    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensorA.buffer, tensorIndex.buffer, outBuffer, uniformBuffer]);
    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseUniformBuffer(uniformBuffer);
    return createTensor(outShape, outBuffer);
  }

  // --- Chunked path ---
  const dimSize = inputShape[dim];
  const elementsPerSlice = inputShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;
  const layout = computeDimChunkLayout(dimSize, elementsPerSlice, maxBindingSize, minAlignment);

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  for (let chunk = 0; chunk < layout.numChunks; chunk++) {
    const chunkStart = chunk * layout.maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + layout.maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * layout.bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * layout.bytesPerSlice;

    const uniformBuffer = createParamsBuffer(ctx.device, params4(outSize, chunkStart, chunkEnd, 0));
    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensorA.buffer, offset: chunkByteOffset, size: chunkByteSize } as GPUBufferBinding },
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
  const tensorIndex = asGPUTensor(index);
  const tensorSrc = asGPUTensor(src);
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("scatterAdd: dimension out of range");
  }

  // Determine chunking
  const outputSizeBytes = tensorA.size * 4;
  const deviceLimits = ctx.device.limits;
  const maxBindingSize = deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const chunked = outputSizeBytes > maxBindingSize;

  // Shared setup
  const outShape = inputShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const srcSize = tensorSrc.shape.reduce((a, b) => a * b, 1);
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Generate shader via tile-IR (both direct and chunked paths)
  const code = chunked
    ? chunkedScatterAddTileIR(inputShape, tensorSrc.shape, dim)
    : scatterAddTileIR(inputShape, tensorSrc.shape, dim);

  const keyPrefix = chunked ? "scatterAddChunked" : "scatterAdd";
  const pipelineKey = `${keyPrefix}:${inputShape.join(",")}:${tensorSrc.shape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, pipelineKey, code);

  // Copy input to output (both paths need this)
  const outBuffer = chunked
    ? createTrackedBuffer(ctx.device, {
        size: outSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
    : resolveOutputBuffer(ctx.device, outSize * 4, [tensorA.buffer, tensorIndex.buffer, tensorSrc.buffer]);

  {
    const enc = getSharedEncoderInstance();
    if (enc) {
      enc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
    } else {
      const cmdEnc = ctx.device.createCommandEncoder();
      cmdEnc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
      submitOrCollect(cmdEnc.finish());
    }
  }

  // --- Direct path ---
  if (!chunked) {
    const uniformBuffer = createUniformBuffer(ctx.device, srcSize);
    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensorIndex.buffer, tensorSrc.buffer, outBuffer, uniformBuffer]);
    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseUniformBuffer(uniformBuffer);
    return createTensor(outShape, outBuffer);
  }

  // --- Chunked path ---
  const dimSize = inputShape[dim];
  const elementsPerSlice = inputShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;
  const layout = computeDimChunkLayout(dimSize, elementsPerSlice, maxBindingSize, minAlignment);

  for (let chunk = 0; chunk < layout.numChunks; chunk++) {
    const chunkStart = chunk * layout.maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + layout.maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * layout.bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * layout.bytesPerSlice;

    const uniformBuffer = createParamsBuffer(ctx.device, params4(srcSize, chunkStart, chunkEnd, 0));
    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensorIndex.buffer } },
        { binding: 1, resource: { buffer: tensorSrc.buffer } },
        { binding: 2, resource: { buffer: outBuffer, offset: chunkByteOffset, size: chunkByteSize } as GPUBufferBinding },
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
export function cat(tensors: BackendTensor[], options: CatOptions): BackendTensor {
  if (tensors.length === 0) throw new Error("cat: empty tensor list");
  if (tensors.length === 1) return tensors[0];

  const ctx = requireContext();
  const gpuTensors = tensors.map(t => asGPUTensor(t));
  const bytesPerElement = gpuTensors[0].dtype === "f16" ? 2 : 4;

  // Compute output shape
  const dim = options.dim;
  const rank = gpuTensors[0].shape.length;
  const outShape = gpuTensors[0].shape.slice();
  for (let i = 1; i < gpuTensors.length; i++) {
    outShape[dim] += gpuTensors[i].shape[dim];
  }
  const outSize = sizeOf(outShape);
  const outBuffer = resolveOutputBuffer(
    ctx.device, outSize * bytesPerElement,
    gpuTensors.map(t => t.buffer),
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
      const dstByteOffset = (o * outDimSize + dimOffset) * innerSize * bytesPerElement;
      if (enc) {
        enc.copyBufferToBuffer(t.buffer, srcByteOffset, outBuffer, dstByteOffset, copyBytes);
      } else {
        const cmdEnc = ctx.device.createCommandEncoder();
        cmdEnc.copyBufferToBuffer(t.buffer, srcByteOffset, outBuffer, dstByteOffset, copyBytes);
        submitOrCollect(cmdEnc.finish());
      }
    }
    dimOffset += tDimSize;
  }

  return createTensor(outShape, outBuffer, undefined, 0, gpuTensors[0].dtype as any);
}
