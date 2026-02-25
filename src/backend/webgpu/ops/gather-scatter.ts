/**
 * Gather and scatter ops with integrated chunking for large tensors.
 *
 * Each op handles both the normal (direct) and chunked paths internally,
 * branching at the dispatch level after shared setup (shapes, strides, shader).
 */
import type { BackendTensor, GatherOptions, ScatterAddOptions } from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { WORKGROUP_SIZE, compute2DDispatch, contiguousStrides, wgslArray } from "../shape-utils";
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
  const tensorA = a as WebGPUTensor;
  const tensorIndex = index as WebGPUTensor;
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
  const inputStrides = contiguousStrides(inputShape);
  const indexStrides = contiguousStrides(indexShape);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // WGSL constants
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const shaderConsts = `
const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const inputStrides = array<u32, ${rank}>(${wgslArray(inputStrides)});
const indexStrides = array<u32, ${rank}>(${wgslArray(indexStrides)});`;

  // Shader body: coords → gather index → input offset → output
  // Chunked variant adds bounds check + local index adjustment.
  const shaderBody = chunked
    ? `  let gatherIdx = u32(indices[idx]);
  if (gatherIdx < params.chunkStart || gatherIdx >= params.chunkEnd) { return; }

  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / indexStrides[d];
    remaining = remaining % indexStrides[d];
  }

  let localIdx = gatherIdx - params.chunkStart;
  var inputOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) { inputOffset = inputOffset + localIdx * inputStrides[d]; }
    else { inputOffset = inputOffset + coords[d] * inputStrides[d]; }
  }
  out[idx] = input[inputOffset];`
    : `  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / indexStrides[d];
    remaining = remaining % indexStrides[d];
  }

  let gatherIdx = u32(indices[idx]);
  var inputOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) { inputOffset = inputOffset + gatherIdx * inputStrides[d]; }
    else { inputOffset = inputOffset + coords[d] * inputStrides[d]; }
  }
  out[idx] = input[inputOffset];`;

  const paramsStruct = chunked
    ? `struct Params { size: u32, chunkStart: u32, chunkEnd: u32, _pad: u32, };`
    : `struct Params { size: u32, };`;

  const code = `
${paramsStruct}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
${shaderConsts}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) { return; }
${shaderBody}
}
`;

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
  const tensorA = a as WebGPUTensor;
  const tensorIndex = index as WebGPUTensor;
  const tensorSrc = src as WebGPUTensor;
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
  const outStrides = contiguousStrides(outShape);
  const srcStrides = contiguousStrides(tensorSrc.shape);
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // WGSL constants
  const idxCompute = use2D
    ? `let srcIdx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let srcIdx = gid.x;`;

  const shaderConsts = `
const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const outStrides = array<u32, ${rank}>(${wgslArray(outStrides)});
const srcStrides = array<u32, ${rank}>(${wgslArray(srcStrides)});`;

  // Shader body: coords → scatter index → output offset → add
  // Chunked variant adds bounds check + local index adjustment.
  const shaderBody = chunked
    ? `  let scatterIdx = u32(indices[srcIdx]);
  if (scatterIdx < params.chunkStart || scatterIdx >= params.chunkEnd) { return; }

  var coords: array<u32, ${rank}>;
  var remaining = srcIdx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / srcStrides[d];
    remaining = remaining % srcStrides[d];
  }

  let localIdx = scatterIdx - params.chunkStart;
  var outOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) { outOffset = outOffset + localIdx * outStrides[d]; }
    else { outOffset = outOffset + coords[d] * outStrides[d]; }
  }
  out[outOffset] = out[outOffset] + src[srcIdx];`
    : `  var coords: array<u32, ${rank}>;
  var remaining = srcIdx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / srcStrides[d];
    remaining = remaining % srcStrides[d];
  }

  let scatterIdx = u32(indices[srcIdx]);
  var outOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) { outOffset = outOffset + scatterIdx * outStrides[d]; }
    else { outOffset = outOffset + coords[d] * outStrides[d]; }
  }
  out[outOffset] = out[outOffset] + src[srcIdx];`;

  const paramsStruct = chunked
    ? `struct Params { srcSize: u32, chunkStart: u32, chunkEnd: u32, _pad: u32, };`
    : `struct Params { srcSize: u32, };`;

  const code = `
${paramsStruct}

@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
${shaderConsts}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (srcIdx >= params.srcSize) { return; }
${shaderBody}
}
`;

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
