/**
 * Strided scatter ops: stridedScatterCopy (+ chunked), stridedScatterAdd (+ chunked).
 */
import type { BackendTensor } from "../../types";
import type { WebGPUTensor } from "../gpu-types";
import { asGPUTensor } from "../gpu-types";
import { sizeOf, WORKGROUP_SIZE, compute2DDispatch } from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor } from "../tensor";
import { destroyCopy } from "../buffer-pool";
import { resolveOutputBuffer } from "../buffer-arena";
import {
  cachedCreateBindGroup, createParamsBuffer, releaseParamsBuffer,
  params,
} from "../bind-group-cache";
import { ensureContiguous } from "./views";
import { stridedScatterCopyTileIR, stridedScatterAddTileIR, flatCopySpec, flatAddSpec } from "./ops-tile-ir";
import { createTileKernelDispatcher } from "../tile-dispatch";


/**
 * Copy src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
export function stridedScatterCopy(
  base: BackendTensor,
  src: BackendTensor,
  options: { offset: number; viewShape: number[]; viewStrides: number[] },
): BackendTensor {
  const baseTensor = asGPUTensor(base);
  const srcTensor = asGPUTensor(src);
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

  const code = stridedScatterCopyTileIR(baseSize, viewShape, viewStrides, offset, srcStrides, srcOffset);
  const key = `stridedScatterCopy:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    baseSize * 4,
    [contiguousBase.buffer, srcTensor.buffer],
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params(baseSize, viewSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [contiguousBase.buffer, srcTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) destroyCopy(contiguousBase);

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
  _maxBindingSize: number,
): BackendTensor {
  const ctx = requireContext();
  const totalElements = baseTensor.size;
  const bytesPerElement = 4; // f32

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * bytesPerElement,
    [baseTensor.buffer, srcTensor.buffer],
  );

  const dispatcher = createTileKernelDispatcher(flatCopySpec());
  dispatcher.dispatchChunked(
    { src: srcTensor.buffer, out: outBuffer },
    { size: totalElements },
    {
      modes: { src: "chunked", out: "chunked" },
      sizeUniform: "size",
      totalElements,
      maxBytesPerElement: bytesPerElement,
    },
  );

  return createTensor(baseTensor.shape, outBuffer);
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
  const baseTensor = asGPUTensor(base);
  const srcTensor = asGPUTensor(src);
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

  const code = stridedScatterAddTileIR(baseSize, viewShape, viewStrides, offset, srcStrides, srcOffset);
  const key = `stridedScatterAdd:${baseSize}:${viewShape.join("x")}:${viewStrides.join(",")}:${offset}:${srcStrides.join(",")}:${srcOffset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    baseSize * 4,
    [contiguousBase.buffer, srcTensor.buffer],
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params(baseSize, viewSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [contiguousBase.buffer, srcTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

  releaseParamsBuffer(paramsBuffer);

  if (contiguousBase !== baseTensor) destroyCopy(contiguousBase);

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
  _maxBindingSize: number,
): BackendTensor {
  const ctx = requireContext();
  const totalElements = baseTensor.size;
  const bytesPerElement = 4; // f32

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * bytesPerElement,
    [baseTensor.buffer, srcTensor.buffer],
  );

  const dispatcher = createTileKernelDispatcher(flatAddSpec());
  dispatcher.dispatchChunked(
    { base: baseTensor.buffer, src: srcTensor.buffer, out: outBuffer },
    { size: totalElements },
    {
      modes: { base: "chunked", src: "chunked", out: "chunked" },
      sizeUniform: "size",
      totalElements,
      maxBytesPerElement: bytesPerElement,
    },
  );

  return createTensor(baseTensor.shape, outBuffer);
}
