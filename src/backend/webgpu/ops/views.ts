/**
 * View and cast ops: cast, reshape, expand, contiguous, narrow, transpose, permute,
 * detectSimpleTranspose, ensureContiguous.
 */
import { inferReshapeStrides } from "../../../core/shape";
import type { BackendTensor, DType, TransposeOptions } from "../../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  profiledCreateBindGroup,
  releaseParamsBuffer,
} from "../bind-group-cache";
import { resolveOutputBuffer } from "../buffer-arena";
import { bufferPool, destroyCopy } from "../buffer-pool";
import {
  dispatchComputePass,
  dispatchElementwise,
  getPipeline,
} from "../dispatch";
import { f16WeightCache, requireContext } from "../gpu-context";
import type { GPUBufferBinding, WebGPUTensor } from "../gpu-types";
import { asGPUTensor, GPUBufferUsage } from "../gpu-types";
import {
  alignBufferSize,
  alignedChunkSize,
  compute2DDispatch,
  dtypeBytes,
  dtypeToWgsl,
  lcm,
  sizeOf,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { createTensor, createTrackedBuffer } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";
import {
  castSpec,
  castTileIR,
  chunkedTransposeTileIR,
  contiguousTileIR,
  narrowBackwardTileIR,
} from "./ops-tile-ir";

/**
 * Cast tensor to a different dtype.
 * Returns same tensor if already the target dtype.
 */
export function cast(a: BackendTensor, dtype: DType): BackendTensor {
  const tensor = asGPUTensor(a);
  const ctx = requireContext();

  // No-op if already the target dtype
  if (tensor.dtype === dtype) {
    return tensor;
  }

  // Check f16 weight cache (populated by Adam dual-write)
  if (dtype === "f16" && tensor.dtype === "f32") {
    const cached = f16WeightCache.get(tensor.buffer);
    if (cached) {
      if (tensor.isContiguous && tensor.offset === 0) {
        // Contiguous: direct return of cached f16 buffer
        return createTensor(tensor.shape, cached, undefined, 0, "f16", false);
      }
      // Non-contiguous view (e.g., transpose) of a cached buffer: return an
      // f16 view with the same strides/offset. The f16 buffer has the same
      // element layout as the f32 buffer (contiguous), so strided access works
      // identically — just with f16 elements instead of f32.
      if (tensor.offset === 0) {
        return createTensor(
          tensor.shape,
          cached,
          tensor.strides,
          0,
          "f16",
          false,
        );
      }
    }
  }

  // Check f16 support
  if (dtype === "f16" && !ctx.f16Supported) {
    throw new Error(
      "f16 dtype requires shader-f16 device feature which is not available",
    );
  }

  // Check if buffer exceeds maxStorageBufferBindingSize — use chunked path
  const srcBytesPerElement = dtypeBytes(tensor.dtype);
  const dstBytesPerElement = dtypeBytes(dtype);
  const limits = ctx.device.limits;
  const maxBindingSize =
    limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const inputDataBytes = tensor.size * srcBytesPerElement;
  const outputDataBytes = tensor.size * dstBytesPerElement;

  if (inputDataBytes > maxBindingSize || outputDataBytes > maxBindingSize) {
    // Chunked cast requires contiguous input
    let src = tensor;
    let contiguousCopy: WebGPUTensor | null = null;
    if (!tensor.isContiguous || tensor.offset > 0) {
      src = asGPUTensor(contiguous(tensor));
      contiguousCopy = src;
    }
    const result = castChunked(src, dtype, ctx, limits);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(tensor.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = castTileIR(
    tensor.dtype as "f32" | "f16" | "i32" | "u32",
    dtype as "f32" | "f16" | "i32" | "u32",
    tensor.shape,
    tensor.strides,
    tensor.offset,
  );
  const key = `cast:${tensor.dtype}->${dtype}:${tensor.shape.join("x")}:${tensor.strides.join(",")}:${tensor.offset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const bytesPerElement = dtypeBytes(dtype);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    tensor.size * bytesPerElement,
    [tensor.buffer],
  );
  const uniformBuf = createParamsBuffer(ctx.device, params(tensor.size));
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    tensor.buffer,
    outBuffer,
    uniformBuf,
  ]);
  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseParamsBuffer(uniformBuf);
  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

/**
 * Chunked cast dispatch for tensors exceeding maxStorageBufferBindingSize.
 * Uses tile-IR spec + dispatchChunked. Input must be contiguous with offset 0.
 */
function castChunked(
  tensor: WebGPUTensor,
  dtype: DType,
  ctx: ReturnType<typeof requireContext>,
  limits: Record<string, number>,
): BackendTensor {
  const srcBytesPerElement = dtypeBytes(tensor.dtype);
  const dstBytesPerElement = dtypeBytes(dtype);
  const minAlignment =
    (limits as Record<string, number>)?.minStorageBufferOffsetAlignment ?? 256;

  // Alignment must satisfy both src and dst offset alignment requirements
  const srcElemsPerAlign = minAlignment / srcBytesPerElement;
  const dstElemsPerAlign = minAlignment / dstBytesPerElement;
  const elementsPerAlignment = lcm(srcElemsPerAlign, dstElemsPerAlign);

  const totalElements = tensor.size;
  const maxBpe = Math.max(srcBytesPerElement, dstBytesPerElement);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * dstBytesPerElement,
    [tensor.buffer],
  );

  // Build flat tile-IR spec: contiguous input, stride 1, offset 0
  type DT = "f32" | "f16" | "i32" | "u32";
  const spec = castSpec(
    tensor.dtype as DT,
    dtype as DT,
    [totalElements],
    [1],
    0,
  );
  const dispatcher = createTileKernelDispatcher(spec);
  dispatcher.dispatchChunked(
    { a: tensor.buffer, out: outBuffer },
    { size: totalElements },
    {
      modes: { a: "chunked", out: "chunked" },
      bytesPerElement: { a: srcBytesPerElement, out: dstBytesPerElement },
      sizeUniform: "size",
      totalElements,
      maxBytesPerElement: maxBpe,
      elementsPerAlignment,
    },
  );

  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

export function reshape(a: BackendTensor, shape: number[]): BackendTensor {
  const tensor = asGPUTensor(a);
  const expected = sizeOf(shape);
  if (expected !== tensor.size) {
    throw new Error("View shape does not match tensor size");
  }

  if (tensor.isContiguous) {
    // Fast path: contiguous input → contiguous output view
    return createTensor(
      shape,
      tensor.buffer,
      undefined,
      tensor.offset,
      tensor.dtype,
      false,
    );
  }

  // Non-contiguous: try to compute valid strides for new shape
  const newStrides = inferReshapeStrides(tensor.shape, tensor.strides, shape);
  if (newStrides !== null) {
    // Compatible layout: return view with computed strides (zero-cost)
    return createTensor(
      shape,
      tensor.buffer,
      newStrides,
      tensor.offset,
      tensor.dtype,
      false,
    );
  }

  // Incompatible: must materialize first, transfer buffer ownership to result
  const contig = asGPUTensor(contiguous(tensor));
  bufferPool.decRef(contig.buffer); // Transfer ownership to result tensor
  return createTensor(
    shape,
    contig.buffer,
    undefined,
    contig.offset,
    tensor.dtype,
    true,
  );
}

/**
 * Expand returns a VIEW - no data copy, just metadata change.
 * Broadcast dimensions get stride=0 (same element repeated).
 */
export function expand(a: BackendTensor, shape: number[]): BackendTensor {
  const tensor = asGPUTensor(a);
  const inputShape = tensor.shape;
  const inputStrides = tensor.strides;

  // Validate shapes are compatible for broadcasting
  if (shape.length < inputShape.length) {
    throw new Error(
      "expand: target shape must have at least as many dimensions as input",
    );
  }

  // Compute output strides with broadcasting
  // For broadcast dims (input dim = 1, output dim > 1), stride = 0
  // For leading dims (not in input), stride = 0
  // For matching dims, use input stride
  const outStrides: number[] = [];
  const padded = shape.length - inputShape.length;

  for (let i = 0; i < shape.length; i++) {
    if (i < padded) {
      // Leading dimension not in input - broadcast with stride 0
      outStrides.push(0);
    } else {
      const inputIdx = i - padded;
      const inputDim = inputShape[inputIdx];
      const outputDim = shape[i];

      if (inputDim === 1 && outputDim > 1) {
        // Broadcast: stride = 0 (repeat same element)
        outStrides.push(0);
      } else if (inputDim === outputDim) {
        // Same size: use existing stride
        outStrides.push(inputStrides[inputIdx]);
      } else {
        throw new Error(
          `expand: incompatible shapes at dimension ${i} (input ${inputDim} vs output ${outputDim})`,
        );
      }
    }
  }

  // Return a view sharing the same buffer
  // Note: expand views are never contiguous (they have stride=0 somewhere)
  // View - does not own the buffer
  return createTensor(
    shape,
    tensor.buffer,
    outStrides,
    tensor.offset,
    tensor.dtype,
    false,
  );
}

/**
 * Materialize a non-contiguous tensor to a new contiguous buffer.
 * If already contiguous, returns the same tensor (no-op).
 * Handles large tensors by processing in chunks.
 */
export function contiguous(a: BackendTensor): BackendTensor {
  const tensor = asGPUTensor(a);

  // Fast path: already contiguous - return a non-owning view.
  // Returning the same tensor object would cause the executor to create a
  // second StorageHandle for the same GPUBuffer.  When the intermediate
  // handle becomes unreachable, destroyUnreachable() would destroy the
  // buffer while the original tensor still references it.
  if (tensor.isContiguous) {
    return createTensor(
      tensor.shape,
      tensor.buffer,
      tensor.strides,
      tensor.offset,
      tensor.dtype,
      false, // ownsBuffer = false → won't destroy buffer on cleanup
    );
  }

  const shape = tensor.shape;
  const outSize = sizeOf(shape);

  if (outSize === 0) {
    throw new Error("contiguous: empty tensors not supported");
  }

  // Check if input buffer exceeds max binding size
  const ctx = requireContext();
  const limits = ctx.device.limits;
  const maxBindingSize =
    limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Get actual buffer size (the backing storage might be larger than the view)
  const inputBufferSize = tensor.buffer.size;

  if (inputBufferSize > maxBindingSize) {
    // Use chunked contiguous for large input buffers
    return contiguousChunked(tensor, maxBindingSize, minAlignment);
  }

  // Fast path: input fits in binding limit
  return contiguousDirect(tensor);
}

/**
 * Direct contiguous implementation for tensors within buffer binding limits.
 */
function contiguousDirect(tensor: WebGPUTensor): WebGPUTensor {
  const ctx = requireContext();
  const shape = tensor.shape;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Create new contiguous buffer
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bytesPerElement, [
    tensor.buffer,
  ]);

  const code = contiguousTileIR(
    shape,
    tensor.strides,
    tensor.offset,
    dtype as "f32" | "f16" | "i32" | "u32",
  );
  const key = `contiguous:${shape.join(",")}:${tensor.strides.join(",")}:${tensor.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;

  dispatchElementwise({
    key,
    shader: code,
    inputs: [tensor.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params(outSize),
    outBuffer,
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  });

  return createTensor(shape, outBuffer, undefined, 0, dtype);
}

/**
 * Chunked contiguous for large input buffers.
 * For transposed 2D tensors [K, N] with strides [1, K], processes by output columns.
 * Each output column reads from a contiguous section of input.
 */
function contiguousChunked(
  tensor: WebGPUTensor,
  maxBindingSize: number,
  minAlignment: number,
): WebGPUTensor {
  const ctx = requireContext();
  const shape = tensor.shape;
  const rank = shape.length;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const wgslType = dtypeToWgsl(dtype);
  const bytesPerElement = dtypeBytes(dtype);

  // Currently optimized for 2D transposed tensors
  // General case would need more sophisticated chunking
  if (rank !== 2) {
    throw new Error("Chunked contiguous currently only supports 2D tensors");
  }

  const [K, N] = shape;
  const [strideK, strideN] = tensor.strides;

  // Create new contiguous buffer
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: alignBufferSize(outSize * bytesPerElement),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // For transposed tensor [K, N] with strides [1, K]:
  // Output column j reads from input[j*K : j*K+K] (contiguous in input)
  // We chunk by both input columns AND output rows to stay within binding limits
  if (strideK === 1 && strideN === K) {
    // This is a simple transpose - output column j reads input rows j
    // Challenge: output is also large and needs chunking
    // Strategy: process in row chunks where each row chunk handles all columns in sub-chunks

    const bytesPerOutputRow = N * bytesPerElement; // Output row [K elements across N columns for one K-row]
    const bytesPerInputRow = K * bytesPerElement; // One column of input (K elements)

    // How many output rows fit in binding limit?
    const maxOutputRows = Math.floor(maxBindingSize / bytesPerOutputRow);

    // How many input columns (= K elements each) fit in binding limit?
    const maxInputCols = Math.floor(maxBindingSize / bytesPerInputRow);

    const alignedOutputRows = alignedChunkSize(
      bytesPerOutputRow,
      maxOutputRows,
      minAlignment,
    );
    const alignedInputCols = alignedChunkSize(
      bytesPerInputRow,
      maxInputCols,
      minAlignment,
    );

    // Process output in row chunks, and within each row chunk, process input in column chunks
    const numOutputRowChunks = Math.ceil(K / alignedOutputRows);
    const numInputColChunks = Math.ceil(N / alignedInputCols);

    // Generate shader via tile-IR
    const code = chunkedTransposeTileIR(wgslType as "f32" | "u32" | "i32");

    const key = `contiguous_chunked_transpose_tiled:${K}:${N}:${dtype}`;
    const pipeline = getPipeline(ctx, key, code);

    for (let rowChunk = 0; rowChunk < numOutputRowChunks; rowChunk++) {
      const rowStart = rowChunk * alignedOutputRows;
      const rowEnd = Math.min(rowStart + alignedOutputRows, K);
      const numRows = rowEnd - rowStart;

      // Output row chunk binding
      const outputByteOffset = rowStart * bytesPerOutputRow;
      const outputChunkSize = numRows * bytesPerOutputRow;

      for (let colChunk = 0; colChunk < numInputColChunks; colChunk++) {
        const colStart = colChunk * alignedInputCols;
        const colEnd = Math.min(colStart + alignedInputCols, N);
        const numCols = colEnd - colStart;

        // Input column chunk binding: columns [colStart, colEnd) means buffer positions [colStart*K, colEnd*K)
        const inputByteOffset =
          (tensor.offset + colStart * K) * bytesPerElement;
        const inputChunkSize = numCols * K * bytesPerElement;

        const tileSize = numRows * numCols;
        const dispatch = compute2DDispatch(
          Math.ceil(tileSize / WORKGROUP_SIZE),
        );

        const paramsBuffer = createParamsBuffer(
          ctx.device,
          params(
            K,
            N,
            rowStart,
            rowEnd,
            colStart,
            colEnd,
            dispatch.gridSizeX * WORKGROUP_SIZE,
          ),
        );

        const bindGroup = profiledCreateBindGroup(ctx.device, {
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            {
              binding: 0,
              resource: {
                buffer: tensor.buffer,
                offset: inputByteOffset,
                size: inputChunkSize,
              } as GPUBufferBinding,
            },
            {
              binding: 1,
              resource: {
                buffer: outBuffer,
                offset: outputByteOffset,
                size: outputChunkSize,
              } as GPUBufferBinding,
            },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });

        dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

        releaseParamsBuffer(paramsBuffer);
      }
    }

    return createTensor(shape, outBuffer, undefined, 0, dtype);
  }

  // For other stride patterns, use a more general approach
  // Process output in chunks and bind input sections as needed
  // This is a fallback that may not be optimal for all patterns
  throw new Error(
    `Chunked contiguous not yet implemented for stride pattern [${tensor.strides.join(", ")}]`,
  );
}

/**
 * Detect if a tensor is a simple last-2-dim transpose of a contiguous buffer.
 * If so, we can pass the original contiguous buffer to the matmul shader with a flipped
 * transpose flag, avoiding a contiguous() materialization dispatch.
 *
 * Returns the "original" contiguous shape (last 2 dims swapped back) if detected,
 * or null if the tensor is not a simple transpose.
 */
export function detectSimpleTranspose(tensor: WebGPUTensor): number[] | null {
  if (tensor.isContiguous) return null; // Already contiguous, no transpose to detect
  if (tensor.offset !== 0) return null; // Non-zero offset not supported
  const rank = tensor.shape.length;
  if (rank < 2) return null;

  const strides = tensor.strides;
  const shape = tensor.shape;

  // For a last-2-dim transpose of a contiguous buffer:
  // strides[-1] should equal shape[-2] (the original inner dim stride)
  // strides[-2] should equal 1 (the original innermost stride)
  if (strides[rank - 2] !== 1) return null;
  if (strides[rank - 1] !== shape[rank - 2]) return null;

  // Check batch dimensions are contiguous: each batch stride should equal
  // the product of all inner dimensions' sizes in the original layout.
  // Original shape has last 2 dims swapped: [...batch, shape[-1], shape[-2]]
  let expectedStride = shape[rank - 1] * shape[rank - 2]; // innermost 2D block
  for (let i = rank - 3; i >= 0; i--) {
    if (strides[i] !== expectedStride) return null;
    expectedStride *= shape[i];
  }

  // Construct the original contiguous shape (swap last 2 dims back)
  const originalShape = shape.slice();
  originalShape[rank - 2] = shape[rank - 1];
  originalShape[rank - 1] = shape[rank - 2];
  return originalShape;
}

/**
 * Helper to ensure a tensor is contiguous, materializing if needed.
 */
export function ensureContiguous(tensor: WebGPUTensor): WebGPUTensor {
  if (tensor.isContiguous) return tensor;
  return asGPUTensor(contiguous(tensor));
}

/** Shorthand: cast to WebGPUTensor + ensure contiguous in one call. */
export function asContiguous(
  bt: import("../../types").BackendTensor,
): WebGPUTensor {
  return ensureContiguous(asGPUTensor(bt));
}

/**
 * Select a contiguous sub-range along one dimension. Returns a view (zero GPU cost).
 * The returned tensor shares the same buffer with an adjusted offset.
 */
export function narrow(
  a: BackendTensor,
  dim: number,
  start: number,
  length: number,
): BackendTensor {
  const tensor = asGPUTensor(a);
  const rank = tensor.shape.length;
  if (dim < 0 || dim >= rank) {
    throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
  }
  if (start < 0 || start + length > tensor.shape[dim]) {
    throw new Error(
      `narrow: range [${start}, ${start + length}) out of bounds for dim size ${tensor.shape[dim]}`,
    );
  }
  const newShape = tensor.shape.slice();
  newShape[dim] = length;
  const newOffset = tensor.offset + start * tensor.strides[dim];
  return createTensor(
    newShape,
    tensor.buffer,
    tensor.strides.slice(),
    newOffset,
    tensor.dtype,
    false,
  );
}

/**
 * Backward for narrow: pad gradient back to original shape.
 * Writes grad into [start, start+length) along dim, zeros elsewhere.
 */
export function narrowBackward(
  grad: BackendTensor,
  dim: number,
  start: number,
  originalLength: number,
): BackendTensor {
  const gradTensor = ensureContiguous(asGPUTensor(grad));

  const outShape = gradTensor.shape.slice();
  outShape[dim] = originalLength;
  const outSize = sizeOf(outShape);
  const dtype = gradTensor.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  const outerSize = sizeOf(outShape.slice(0, dim));
  const innerSize = sizeOf(outShape.slice(dim + 1));
  const gradDimSize = gradTensor.shape[dim]; // = length from narrow
  const outDimSize = originalLength;

  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const gridSizeX = dispatch.x * WORKGROUP_SIZE;

  const shaderCode = narrowBackwardTileIR(
    outDimSize,
    outSize,
    dtype as "f32" | "f16" | "i32" | "u32",
  );
  const key = `narrowBackward:${outDimSize}:${gradDimSize}:${start}:${outSize}:${dtype}:${use2D ? `2d:${gridSizeX}` : "1d"}`;

  const outBuffer = dispatchElementwise({
    key,
    shader: shaderCode,
    inputs: [gradTensor.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params(outerSize, innerSize, gradDimSize, start),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  });

  if (gradTensor !== asGPUTensor(grad)) destroyCopy(gradTensor);

  return createTensor(outShape, outBuffer, undefined, 0, dtype);
}

/**
 * Transpose returns a VIEW - no data copy, just metadata change.
 * Delegates to permute with swapped dimension indices.
 */
export function transpose(
  a: BackendTensor,
  options: TransposeOptions,
): BackendTensor {
  const { dim0, dim1 } = options;
  const rank = (a as WebGPUTensor).shape.length;
  if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
    throw new Error("transpose: dimension out of range");
  }
  const dims = Array.from({ length: rank }, (_, i) => i);
  dims[dim0] = dim1;
  dims[dim1] = dim0;
  return permute(a, dims);
}

/**
 * Permute dimensions according to the given order.
 * Returns a view sharing the same buffer (no data copy).
 */
export function permute(a: BackendTensor, dims: number[]): BackendTensor {
  const tensor = asGPUTensor(a);
  const inputShape = tensor.shape;
  const rank = inputShape.length;

  // Validate dims
  if (dims.length !== rank) {
    throw new Error(
      `permute: dims length ${dims.length} doesn't match tensor rank ${rank}`,
    );
  }

  // Check for valid permutation (each dim appears exactly once)
  const seen = new Set<number>();
  for (const d of dims) {
    if (d < 0 || d >= rank) {
      throw new Error(`permute: dimension ${d} out of range for rank ${rank}`);
    }
    if (seen.has(d)) {
      throw new Error(`permute: duplicate dimension ${d}`);
    }
    seen.add(d);
  }

  // Reorder shape and strides according to dims
  const outShape = dims.map((d) => inputShape[d]);
  const outStrides = dims.map((d) => tensor.strides[d]);

  // Return a view sharing the same buffer
  // View - does not own the buffer
  return createTensor(
    outShape,
    tensor.buffer,
    outStrides,
    tensor.offset,
    tensor.dtype,
    false,
  );
}
