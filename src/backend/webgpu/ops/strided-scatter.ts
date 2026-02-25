/**
 * Strided scatter ops: stridedScatterCopy (+ chunked), stridedScatterAdd (+ chunked).
 * Extracted from index.ts — purely structural refactoring.
 */
import type { BackendTensor } from "../../types";
import type { GPUDevice, WebGPUTensor } from "../gpu-types";
import { sizeOf, WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM, compute2DDispatch } from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor } from "../tensor";
import { bufferPool } from "../buffer-pool";
import { resolveOutputBuffer } from "../buffer-arena";
import {
  cachedCreateBindGroup, createParamsBuffer, releaseParamsBuffer,
  params2, params3,
} from "../bind-group-cache";
import { ensureContiguous } from "./views";
import { computeFlatChunkLayout, dispatchFlatChunked } from "../chunked-dispatch";

function stridedScatterCopyShader(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
  gridSizeX?: number,
): string {
  const rank = viewShape.length;
  const viewSize = viewShape.reduce((a, b) => a * b, 1);
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Build coordinate calculation code
  let coordCode = "";
  let baseOffsetCode = `var baseOffset: u32 = ${viewOffset}u;\n`;
  let srcOffsetCode = `var srcOffset: u32 = ${srcOffset}u;\n`;

  for (let d = 0; d < rank; d++) {
    const shapeStride = viewShape.slice(d + 1).reduce((a, b) => a * b, 1);
    coordCode += `    let coord${d} = (remainder / ${shapeStride}u) % ${viewShape[d]}u;\n`;
    if (d < rank - 1) {
      coordCode += `    remainder = remainder % ${shapeStride}u;\n`;
    }
    baseOffsetCode += `    baseOffset += coord${d} * ${viewStrides[d]}u;\n`;
    srcOffsetCode += `    srcOffset += coord${d} * ${srcStrides[d]}u;\n`;
  }

  return `
struct Params {
  baseSize: u32,
  viewSize: u32,
};

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}

  // First pass: copy base to output (all threads that fit in baseSize)
  if (idx < params.baseSize) {
    out[idx] = base[idx];
  }

  workgroupBarrier();

  // Second pass: scatter src values into output at view positions
  if (idx < params.viewSize) {
    var remainder = idx;
${coordCode}
${baseOffsetCode}
${srcOffsetCode}
    out[baseOffset] = src[srcOffset];
  }
}
`;
}

/**
 * Copy src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
export function stridedScatterCopy(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const baseTensor = base as WebGPUTensor;
  const srcTensor = src as WebGPUTensor;
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  if (baseSize === 0) {
    throw new Error("stridedScatterCopy: empty base tensor");
  }

  const ctx = requireContext();
  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Check if any buffer exceeds max binding size
  const baseSizeBytes = baseSize * 4;
  const srcSizeBytes = srcTensor.size * 4;

  if (baseSizeBytes > maxBindingSize || srcSizeBytes > maxBindingSize) {
    // Check for simple full copy: viewSize == baseSize, offset == 0, both contiguous
    const isFullCopy = viewSize === baseSize && offset === 0;

    // Check if viewStrides are contiguous
    let viewContiguous = true;
    let expectedStride = 1;
    for (let d = viewShape.length - 1; d >= 0; d--) {
      if (viewStrides[d] !== expectedStride) {
        viewContiguous = false;
        break;
      }
      expectedStride *= viewShape[d];
    }

    // Check if src strides are contiguous
    const srcStrides = srcTensor.strides;
    let srcContiguous = srcTensor.isContiguous;
    if (!srcContiguous) {
      // Double-check with stride calculation
      srcContiguous = true;
      expectedStride = 1;
      for (let d = srcTensor.shape.length - 1; d >= 0; d--) {
        if (srcStrides[d] !== expectedStride) {
          srcContiguous = false;
          break;
        }
        expectedStride *= srcTensor.shape[d];
      }
    }

    if (isFullCopy && baseTensor.isContiguous && srcContiguous && viewContiguous) {
      // Fast path: simple chunked copy from src to output (no need to copy base first since all elements overwritten)
      return stridedScatterCopyChunkedSimple(
        baseTensor,
        srcTensor,
        maxBindingSize,
      );
    }

    // General chunked case for complex stride patterns - not yet implemented
    // For now, fall through to original implementation which will fail with validation error
    // This case is rare (non-contiguous large tensors)
  }

  return stridedScatterCopyDirect(baseTensor, srcTensor, options);
}

/**
 * Direct implementation of stridedScatterCopy for small tensors.
 */
function stridedScatterCopyDirect(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  const ctx = requireContext();

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const maxSize = Math.max(baseSize, viewSize);
  const totalWorkgroups = Math.ceil(maxSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Ensure base is contiguous for simplicity
  const contiguousBase = baseTensor.isContiguous
    ? baseTensor
    : ensureContiguous(baseTensor);

  // Compute src strides for reading
  const srcStrides = srcTensor.strides;
  const srcOffset = srcTensor.offset;

  const code = stridedScatterCopyShader(
    baseSize,
    viewShape,
    viewStrides,
    offset,
    srcStrides,
    srcOffset,
    use2D ? dispatch.gridSizeX : undefined,
  );

  const key = `stridedScatterCopy:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    baseSize * 4,
    [contiguousBase.buffer, srcTensor.buffer],
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(baseSize, viewSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [contiguousBase.buffer, srcTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) {
    bufferPool.decRef(contiguousBase.buffer);
    bufferPool.deferredDestroy(contiguousBase.buffer, contiguousBase.size * 4);
  }

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Chunked implementation for simple full contiguous copy (src -> output).
 * Used when both src and dest exceed buffer binding limit but are contiguous
 * and the copy covers all elements.
 */
function stridedScatterCopyChunkedSimple(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  maxBindingSize: number,
): BackendTensor {
  const ctx = requireContext();
  const totalElements = baseTensor.size;
  const bytesPerElement = 4; // f32

  const limits = ctx.device.limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  const layout = computeFlatChunkLayout(totalElements, bytesPerElement, maxBindingSize, minAlignment);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * bytesPerElement,
    [baseTensor.buffer, srcTensor.buffer],
  );

  const idxCompute = layout.use2D
    ? `let localIdx = gid.x + gid.y * ${layout.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let localIdx = gid.x;`;

  const code = `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Params {
  chunkSize: u32,
  totalSize: u32,
  chunkOffset: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (localIdx >= params.chunkSize) { return; }

  let globalIdx = params.chunkOffset + localIdx;
  if (globalIdx >= params.totalSize) { return; }

  out[localIdx] = src[localIdx];
}
`;

  const key = `stridedScatterCopyChunkedSimple:${WORKGROUP_SIZE}:${layout.use2D ? `2d:${layout.gridSizeX}` : "1d"}`;

  dispatchFlatChunked({
    key, shader: code, layout,
    inputs: [{ buffer: srcTensor.buffer, mode: "chunked" }],
    outBuffer, outBytesPerElement: bytesPerElement, totalElements,
    createChunkParams: (device: GPUDevice, chunkSize: number, chunkStart: number) =>
      createParamsBuffer(device, params3(chunkSize, totalElements, chunkStart)),
    releaseChunkParams: releaseParamsBuffer,
  });

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Generate shader for strided scatter add.
 * Adds src values into base tensor at positions defined by view strides.
 */
function stridedScatterAddShader(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
  gridSizeX?: number,
): string {
  const rank = viewShape.length;
  const viewSize = viewShape.reduce((a, b) => a * b, 1);
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Build coordinate calculation code
  let coordCode = "";
  let baseOffsetCode = `var baseOffset: u32 = ${viewOffset}u;\n`;
  let srcOffsetCode = `var srcOffset: u32 = ${srcOffset}u;\n`;

  for (let d = 0; d < rank; d++) {
    const shapeStride = viewShape.slice(d + 1).reduce((a, b) => a * b, 1);
    coordCode += `    let coord${d} = (remainder / ${shapeStride}u) % ${viewShape[d]}u;\n`;
    if (d < rank - 1) {
      coordCode += `    remainder = remainder % ${shapeStride}u;\n`;
    }
    baseOffsetCode += `    baseOffset += coord${d} * ${viewStrides[d]}u;\n`;
    srcOffsetCode += `    srcOffset += coord${d} * ${srcStrides[d]}u;\n`;
  }

  return `
struct Params {
  baseSize: u32,
  viewSize: u32,
};

@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}

  // First pass: copy base to output
  if (idx < params.baseSize) {
    out[idx] = base[idx];
  }

  workgroupBarrier();

  // Second pass: add src values into output at view positions
  if (idx < params.viewSize) {
    var remainder = idx;
${coordCode}
${baseOffsetCode}
${srcOffsetCode}
    out[baseOffset] = out[baseOffset] + src[srcOffset];
  }
}
`;
}

/**
 * Add src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
export function stridedScatterAdd(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const baseTensor = base as WebGPUTensor;
  const srcTensor = src as WebGPUTensor;
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  if (baseSize === 0) {
    throw new Error("stridedScatterAdd: empty base tensor");
  }

  const ctx = requireContext();
  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Check if any buffer exceeds max binding size
  const baseSizeBytes = baseSize * 4;
  const srcSizeBytes = srcTensor.size * 4;

  if (baseSizeBytes > maxBindingSize || srcSizeBytes > maxBindingSize) {
    // Check for simple full add: viewSize == baseSize, offset == 0, both contiguous
    const isFullAdd = viewSize === baseSize && offset === 0;

    // Check if viewStrides are contiguous
    let viewContiguous = true;
    let expectedStride = 1;
    for (let d = viewShape.length - 1; d >= 0; d--) {
      if (viewStrides[d] !== expectedStride) {
        viewContiguous = false;
        break;
      }
      expectedStride *= viewShape[d];
    }

    // Check if src strides are contiguous
    const srcStrides = srcTensor.strides;
    let srcContiguous = srcTensor.isContiguous;
    if (!srcContiguous) {
      srcContiguous = true;
      expectedStride = 1;
      for (let d = srcTensor.shape.length - 1; d >= 0; d--) {
        if (srcStrides[d] !== expectedStride) {
          srcContiguous = false;
          break;
        }
        expectedStride *= srcTensor.shape[d];
      }
    }

    if (isFullAdd && baseTensor.isContiguous && srcContiguous && viewContiguous) {
      // Fast path: simple chunked add
      return stridedScatterAddChunkedSimple(
        baseTensor,
        srcTensor,
        maxBindingSize,
      );
    }

    // General chunked case - not yet implemented
  }

  return stridedScatterAddDirect(baseTensor, srcTensor, options);
}

/**
 * Direct implementation of stridedScatterAdd for small tensors.
 */
function stridedScatterAddDirect(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const { offset, viewShape, viewStrides } = options;

  const baseSize = sizeOf(baseTensor.shape);
  const viewSize = sizeOf(viewShape);

  const ctx = requireContext();

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const maxSize = Math.max(baseSize, viewSize);
  const totalWorkgroups = Math.ceil(maxSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Ensure base is contiguous for simplicity
  const contiguousBase = baseTensor.isContiguous
    ? baseTensor
    : ensureContiguous(baseTensor);

  // Compute src strides for reading
  const srcStrides = srcTensor.strides;
  const srcOffset = srcTensor.offset;

  const code = stridedScatterAddShader(
    baseSize,
    viewShape,
    viewStrides,
    offset,
    srcStrides,
    srcOffset,
    use2D ? dispatch.gridSizeX : undefined,
  );

  const key = `stridedScatterAdd:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    baseSize * 4,
    [contiguousBase.buffer, srcTensor.buffer],
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(baseSize, viewSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [contiguousBase.buffer, srcTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) {
    bufferPool.decRef(contiguousBase.buffer);
    bufferPool.deferredDestroy(contiguousBase.buffer, contiguousBase.size * 4);
  }

  return createTensor(baseTensor.shape, outBuffer);
}

/**
 * Chunked implementation for simple full contiguous add (base + src -> output).
 * Used when both base and src exceed buffer binding limit but are contiguous
 * and the add covers all elements.
 */
function stridedScatterAddChunkedSimple(
  baseTensor: WebGPUTensor,
  srcTensor: WebGPUTensor,
  maxBindingSize: number,
): BackendTensor {
  const ctx = requireContext();
  const totalElements = baseTensor.size;
  const bytesPerElement = 4; // f32

  const limits = ctx.device.limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  const layout = computeFlatChunkLayout(totalElements, bytesPerElement, maxBindingSize, minAlignment);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * bytesPerElement,
    [baseTensor.buffer, srcTensor.buffer],
  );

  const idxCompute = layout.use2D
    ? `let localIdx = gid.x + gid.y * ${layout.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let localIdx = gid.x;`;

  const code = `
@group(0) @binding(0) var<storage, read> base: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Params {
  chunkSize: u32,
  totalSize: u32,
  chunkOffset: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (localIdx >= params.chunkSize) { return; }

  let globalIdx = params.chunkOffset + localIdx;
  if (globalIdx >= params.totalSize) { return; }

  out[localIdx] = base[localIdx] + src[localIdx];
}
`;

  const key = `stridedScatterAddChunkedSimple:${WORKGROUP_SIZE}:${layout.use2D ? `2d:${layout.gridSizeX}` : "1d"}`;

  dispatchFlatChunked({
    key, shader: code, layout,
    inputs: [
      { buffer: baseTensor.buffer, mode: "chunked" },
      { buffer: srcTensor.buffer, mode: "chunked" },
    ],
    outBuffer, outBytesPerElement: bytesPerElement, totalElements,
    createChunkParams: (device: GPUDevice, chunkSize: number, chunkStart: number) =>
      createParamsBuffer(device, params3(chunkSize, totalElements, chunkStart)),
    releaseChunkParams: releaseParamsBuffer,
  });

  return createTensor(baseTensor.shape, outBuffer);
}
