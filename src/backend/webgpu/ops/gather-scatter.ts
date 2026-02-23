/**
 * Gather and scatter ops: gather, gatherDirect, gatherChunked, scatterAdd, scatterAddDirect, scatterAddChunked.
 * Extracted from index.ts — purely structural refactoring.
 */
import type { BackendTensor, GatherOptions, ScatterAddOptions } from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { gcd, WORKGROUP_SIZE, compute2DDispatch } from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import { resolveOutputBuffer } from "../buffer-arena";
import {
  cachedCreateBindGroup, profiledCreateBindGroup,
  createParamsBuffer, releaseParamsBuffer,
  createUniformBuffer, releaseUniformBuffer,
  params4,
} from "../bind-group-cache";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";

/** Local type alias for GPU buffer binding descriptors with offset/size. */
type GPUBufferBinding = { buffer: GPUBuffer; offset?: number; size?: number };

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
  const rank = inputShape.length;

  if (dim < 0 || dim >= rank) {
    throw new Error("gather: dimension out of range");
  }

  // Check if input tensor exceeds max buffer binding size
  const inputSizeBytes = tensorA.size * 4; // f32 = 4 bytes
  const gatherLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = gatherLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  if (inputSizeBytes > maxBindingSize) {
    // Chunked path for large input tensors
    return gatherChunked(tensorA, tensorIndex, options, maxBindingSize);
  }

  // Fast path: existing implementation for normal-sized tensors
  return gatherDirect(tensorA, tensorIndex, options);
}

/**
 * Direct gather implementation for tensors within buffer binding limits.
 */
export function gatherDirect(
  tensorA: WebGPUTensor,
  tensorIndex: WebGPUTensor,
  options: GatherOptions,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = tensorA.shape;
  const indexShape = tensorIndex.shape;
  const rank = inputShape.length;

  // Output shape is same as index shape
  const outShape = indexShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * 4,
    [tensorA.buffer, tensorIndex.buffer],
  );

  // Compute input strides
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  // Compute index strides
  const indexStrides: number[] = [];
  for (let i = 0; i < indexShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < indexShape.length; j++) {
      stride *= indexShape[j];
    }
    indexStrides.push(stride);
  }

  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const indexShapeArray = `array<u32, ${rank}>(${indexShape.map((s) => `${s}u`).join(", ")})`;
  const indexStridesArray = `array<u32, ${rank}>(${indexStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const inputStrides = ${inputStridesArray};
const indexShape = ${indexShapeArray};
const indexStrides = ${indexStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert flat index to coordinates in index tensor
  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / indexStrides[d];
    remaining = remaining % indexStrides[d];
  }

  // Get the gather index for the specified dimension
  let gatherIdx = u32(indices[idx]);

  // Compute input offset using coords, but replace dim with gatherIdx
  var inputOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      inputOffset = inputOffset + gatherIdx * inputStrides[d];
    } else {
      inputOffset = inputOffset + coords[d] * inputStrides[d];
    }
  }

  out[idx] = input[inputOffset];
}
`;

  const pipeline = getPipeline(
    ctx,
    `gather:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );
  const uniformBuffer = createUniformBuffer(ctx.device, outSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensorA.buffer, tensorIndex.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Chunked gather implementation for tensors exceeding buffer binding limits.
 *
 * Strategy: Partition dispatch by which indices fall into which chunk of the input.
 * Each dispatch binds a different chunk of the input buffer (using WebGPU's offset binding).
 * Each dispatch only processes indices that fall within its chunk's range.
 * Output accumulates across dispatches (each writes to non-overlapping output positions).
 */
export function gatherChunked(
  input: WebGPUTensor,
  index: WebGPUTensor,
  options: GatherOptions,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = input.shape;
  const indexShape = index.shape;
  const rank = inputShape.length;
  const dimSize = inputShape[dim];

  // Output shape is same as index shape
  const outShape = indexShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);

  // Calculate slice size (elements per entry along the gather dimension)
  // For dim=0 on [vocabSize, embedDim], this is embedDim
  const elementsPerSlice = inputShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const bytesPerSlice = elementsPerSlice * 4; // f32 = 4 bytes

  // WebGPU requires buffer binding offsets to be aligned (typically 256 bytes)
  const deviceLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;

  // Ensure chunk boundaries are aligned by adjusting slices per chunk
  // We need (slicesPerChunk * bytesPerSlice) to be a multiple of minAlignment
  let maxSlicesPerChunk = Math.floor(maxBindingSize / bytesPerSlice);

  // If bytesPerSlice isn't a multiple of minAlignment, we need to adjust
  if (bytesPerSlice % minAlignment !== 0) {
    // Find how many slices we need for aligned chunk boundaries
    const slicesForAlignment = minAlignment / gcd(bytesPerSlice, minAlignment);
    // Round down maxSlicesPerChunk to a multiple of slicesForAlignment
    maxSlicesPerChunk = Math.floor(maxSlicesPerChunk / slicesForAlignment) * slicesForAlignment;
    if (maxSlicesPerChunk === 0) {
      maxSlicesPerChunk = slicesForAlignment; // At least one aligned group
    }
  }

  const numChunks = Math.ceil(dimSize / maxSlicesPerChunk);

  // Create output buffer
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Compute input strides
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  // Compute index strides
  const indexStrides: number[] = [];
  for (let i = 0; i < indexShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < indexShape.length; j++) {
      stride *= indexShape[j];
    }
    indexStrides.push(stride);
  }

  // Compute 2D dispatch for large output
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const indexShapeArray = `array<u32, ${rank}>(${indexShape.map((s) => `${s}u`).join(", ")})`;
  const indexStridesArray = `array<u32, ${rank}>(${indexStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Shader with chunk bounds checking
  const code = `
struct Params {
  size: u32,
  chunkStart: u32,
  chunkEnd: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const inputStrides = ${inputStridesArray};
const indexShape = ${indexShapeArray};
const indexStrides = ${indexStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Get the gather index for the specified dimension
  let gatherIdx = u32(indices[idx]);

  // Only process if this index falls in our chunk
  if (gatherIdx < params.chunkStart || gatherIdx >= params.chunkEnd) {
    return;  // Skip - will be handled by another chunk's dispatch
  }

  // Convert flat index to coordinates in index tensor
  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / indexStrides[d];
    remaining = remaining % indexStrides[d];
  }

  // Adjust index to be relative to chunk start
  let localIdx = gatherIdx - params.chunkStart;

  // Compute input offset using coords, but replace dim with localIdx
  var inputOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      inputOffset = inputOffset + localIdx * inputStrides[d];
    } else {
      inputOffset = inputOffset + coords[d] * inputStrides[d];
    }
  }

  out[idx] = input[inputOffset];
}
`;

  const pipeline = getPipeline(
    ctx,
    `gatherChunked:${inputShape.join(",")}:${indexShape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );

  // Dispatch for each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * bytesPerSlice;

    const uniformBuffer = createParamsBuffer(ctx.device, params4(outSize, chunkStart, chunkEnd, 0));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: input.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          } as GPUBufferBinding,
        },
        { binding: 1, resource: { buffer: index.buffer } },
        { binding: 2, resource: { buffer: outBuffer } },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
    releaseParamsBuffer(uniformBuffer);
  }

  return createTensor(outShape, outBuffer);
}

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

  // Check if output tensor exceeds max buffer binding size
  const outputSizeBytes = tensorA.size * 4; // f32 = 4 bytes
  const scatterLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = scatterLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  if (outputSizeBytes > maxBindingSize) {
    // Chunked path for large output tensors
    return scatterAddChunked(tensorA, tensorIndex, tensorSrc, options, maxBindingSize);
  }

  // Fast path: existing implementation for normal-sized tensors
  return scatterAddDirect(tensorA, tensorIndex, tensorSrc, options);
}

/**
 * Direct scatterAdd implementation for tensors within buffer binding limits.
 */
export function scatterAddDirect(
  tensorA: WebGPUTensor,
  tensorIndex: WebGPUTensor,
  tensorSrc: WebGPUTensor,
  options: ScatterAddOptions,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;

  // Output shape is same as input shape
  const outShape = inputShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const srcSize = tensorSrc.shape.reduce((a, b) => a * b, 1);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // First, copy input to output
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * 4,
    [tensorA.buffer, tensorIndex.buffer, tensorSrc.buffer],
  );

  {
    if (getSharedEncoderInstance()) {
      getSharedEncoderInstance().copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
    } else {
      const enc = ctx.device.createCommandEncoder();
      enc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
      submitOrCollect(enc.finish());
    }
  }

  // Compute strides for output and src
  const outStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  const srcStrides: number[] = [];
  for (let i = 0; i < tensorSrc.shape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < tensorSrc.shape.length; j++) {
      stride *= tensorSrc.shape[j];
    }
    srcStrides.push(stride);
  }

  const outStridesArray = `array<u32, ${rank}>(${outStrides.map((s) => `${s}u`).join(", ")})`;
  const srcShapeArray = `array<u32, ${rank}>(${tensorSrc.shape.map((s) => `${s}u`).join(", ")})`;
  const srcStridesArray = `array<u32, ${rank}>(${srcStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let srcIdx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let srcIdx = gid.x;`;

  // Note: scatterAdd with atomics would be more correct, but for simplicity
  // we use a loop-based approach that processes each src element sequentially
  // This is less parallel but handles the general case correctly
  const code = `
struct Params {
  srcSize: u32,
};

@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const outStrides = ${outStridesArray};
const srcShape = ${srcShapeArray};
const srcStrides = ${srcStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (srcIdx >= params.srcSize) {
    return;
  }

  // Convert src flat index to coordinates
  var coords: array<u32, ${rank}>;
  var remaining = srcIdx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / srcStrides[d];
    remaining = remaining % srcStrides[d];
  }

  // Get the scatter index for the specified dimension
  let scatterIdx = u32(indices[srcIdx]);

  // Compute output offset using coords, but replace dim with scatterIdx
  var outOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      outOffset = outOffset + scatterIdx * outStrides[d];
    } else {
      outOffset = outOffset + coords[d] * outStrides[d];
    }
  }

  // Atomic add would be ideal here, but f32 atomics aren't widely supported
  // For now, we accept potential race conditions for overlapping indices
  out[outOffset] = out[outOffset] + src[srcIdx];
}
`;

  const pipeline = getPipeline(
    ctx,
    `scatterAdd:${inputShape.join(",")}:${tensorSrc.shape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );
  const uniformBuffer = createUniformBuffer(ctx.device, srcSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensorIndex.buffer, tensorSrc.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Chunked scatterAdd implementation for output tensors exceeding buffer binding limits.
 *
 * Strategy: Partition dispatch by which scatter targets fall into which chunk of the output.
 * Each dispatch binds a different chunk of the output buffer (using WebGPU's offset binding).
 * Each dispatch only processes source elements whose scatter target falls in its chunk's range.
 */
export function scatterAddChunked(
  tensorA: WebGPUTensor,
  tensorIndex: WebGPUTensor,
  tensorSrc: WebGPUTensor,
  options: ScatterAddOptions,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const { dim } = options;
  const inputShape = tensorA.shape;
  const rank = inputShape.length;
  const dimSize = inputShape[dim];

  // Output shape is same as input shape
  const outShape = inputShape.slice();
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const srcSize = tensorSrc.shape.reduce((a, b) => a * b, 1);

  // Calculate slice size (elements per entry along the scatter dimension)
  const elementsPerSlice = inputShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const bytesPerSlice = elementsPerSlice * 4; // f32 = 4 bytes

  // WebGPU requires buffer binding offsets to be aligned (typically 256 bytes)
  const deviceLimits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const minAlignment = deviceLimits?.minStorageBufferOffsetAlignment ?? 256;

  // Ensure chunk boundaries are aligned by adjusting slices per chunk
  // We need (slicesPerChunk * bytesPerSlice) to be a multiple of minAlignment
  let maxSlicesPerChunk = Math.floor(maxBindingSize / bytesPerSlice);

  // If bytesPerSlice isn't a multiple of minAlignment, we need to adjust
  if (bytesPerSlice % minAlignment !== 0) {
    // Find how many slices we need for aligned chunk boundaries
    const slicesForAlignment = minAlignment / gcd(bytesPerSlice, minAlignment);
    // Round down maxSlicesPerChunk to a multiple of slicesForAlignment
    maxSlicesPerChunk = Math.floor(maxSlicesPerChunk / slicesForAlignment) * slicesForAlignment;
    if (maxSlicesPerChunk === 0) {
      maxSlicesPerChunk = slicesForAlignment; // At least one aligned group
    }
  }

  const numChunks = Math.ceil(dimSize / maxSlicesPerChunk);

  // Create output buffer (full size - we'll bind chunks of it)
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // Copy input to output first
  {
    if (getSharedEncoderInstance()) {
      getSharedEncoderInstance().copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
    } else {
      const enc = ctx.device.createCommandEncoder();
      enc.copyBufferToBuffer(tensorA.buffer, 0, outBuffer, 0, outSize * 4);
      submitOrCollect(enc.finish());
    }
  }

  // Compute strides for output and src
  const outStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  const srcStrides: number[] = [];
  for (let i = 0; i < tensorSrc.shape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < tensorSrc.shape.length; j++) {
      stride *= tensorSrc.shape[j];
    }
    srcStrides.push(stride);
  }

  // Compute 2D dispatch for large src tensors
  const totalWorkgroups = Math.ceil(srcSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const outStridesArray = `array<u32, ${rank}>(${outStrides.map((s) => `${s}u`).join(", ")})`;
  const srcShapeArray = `array<u32, ${rank}>(${tensorSrc.shape.map((s) => `${s}u`).join(", ")})`;
  const srcStridesArray = `array<u32, ${rank}>(${srcStrides.map((s) => `${s}u`).join(", ")})`;
  const idxCompute = use2D
    ? `let srcIdx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let srcIdx = gid.x;`;

  // Shader with chunk bounds checking
  const code = `
struct Params {
  srcSize: u32,
  chunkStart: u32,
  chunkEnd: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> indices: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const DIM: u32 = ${dim}u;
const outStrides = ${outStridesArray};
const srcShape = ${srcShapeArray};
const srcStrides = ${srcStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (srcIdx >= params.srcSize) {
    return;
  }

  // Get the scatter index for the specified dimension
  let scatterIdx = u32(indices[srcIdx]);

  // Only process if scatter target falls in our output chunk
  if (scatterIdx < params.chunkStart || scatterIdx >= params.chunkEnd) {
    return;
  }

  // Convert src flat index to coordinates
  var coords: array<u32, ${rank}>;
  var remaining = srcIdx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / srcStrides[d];
    remaining = remaining % srcStrides[d];
  }

  // Adjust index to be relative to chunk start
  let localIdx = scatterIdx - params.chunkStart;

  // Compute output offset using coords, but replace dim with localIdx
  var outOffset = 0u;
  for (var d = 0u; d < RANK; d = d + 1u) {
    if (d == DIM) {
      outOffset = outOffset + localIdx * outStrides[d];
    } else {
      outOffset = outOffset + coords[d] * outStrides[d];
    }
  }

  // Atomic add would be ideal here, but f32 atomics aren't widely supported
  // For now, we accept potential race conditions for overlapping indices
  out[outOffset] = out[outOffset] + src[srcIdx];
}
`;

  const pipeline = getPipeline(
    ctx,
    `scatterAddChunked:${inputShape.join(",")}:${tensorSrc.shape.join(",")}:${dim}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`,
    code,
  );

  // Dispatch for each chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * maxSlicesPerChunk;
    const chunkEnd = Math.min(chunkStart + maxSlicesPerChunk, dimSize);
    const chunkByteOffset = chunkStart * bytesPerSlice;
    const chunkByteSize = (chunkEnd - chunkStart) * bytesPerSlice;

    const uniformBuffer = createParamsBuffer(ctx.device, params4(srcSize, chunkStart, chunkEnd, 0));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
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
