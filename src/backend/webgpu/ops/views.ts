/**
 * View and cast ops: cast, reshape, expand, contiguous, narrow, transpose, permute,
 * detectSimpleTranspose, ensureContiguous.
 */
import { inferReshapeStrides } from "../../../core/shape";
import { recordedCopyBufferToBuffer } from "../../../executor/compiled-plan";
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
  type ElementwiseDirectPlan,
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
  lcm,
  sizeOf,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import { createTensor, createTrackedBuffer } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";
import {
  isCompilationRecording,
  trackSharedEncoderWrite,
} from "../webgpu-state";
import {
  castSpec,
  castTileIR,
  chunkedTransposeTileIR,
  contiguousTileIR,
  narrowBackwardTileIR,
} from "./ops-tile-ir";
import { expandMeta, narrowMeta, permuteMeta } from "./view-meta";

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

  // Check f16 weight cache (populated by Adam dual-write).
  // Skip during compilation recording: cached buffers weren't allocated during
  // this plan so they won't be in bufferToSlot, causing the result to be silently
  // dropped from compiled.results. Forcing a real dispatch ensures the output
  // buffer is recorded and the saved-for-backward tensor gets materialized.
  if (dtype === "f16" && tensor.dtype === "f32" && !isCompilationRecording()) {
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
    const result = castChunked(
      src,
      dtype,
      ctx,
      limits as Record<string, number>,
    );
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  // Direct dispatch — single-sourced with the stream generator.
  const plan = planCastDirectCore(tensor, dtype);
  const pipeline = getPipeline(ctx, plan.key, plan.shader);
  const outBuffer = resolveOutputBuffer(ctx.device, plan.outputSizeBytes, [
    tensor.buffer,
  ]);
  const uniformBuf = createParamsBuffer(ctx.device, plan.paramsData);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    tensor.buffer,
    outBuffer,
    uniformBuf,
  ]);
  dispatchComputePass(pipeline, bindGroup, plan.dispatchX, plan.dispatchY);
  releaseParamsBuffer(uniformBuf);
  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

/** Stage-4 plan/encode split for the DIRECT cast path. Null where cast()
 *  routes to chunked dispatch (with a possible contiguous-copy prologue). */
export function planCastDirect(
  tensor: WebGPUTensor,
  dtype: DType,
): ElementwiseDirectPlan | null {
  const ctx = requireContext();
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  if (
    tensor.size * dtypeBytes(tensor.dtype) > maxBindingSize ||
    tensor.size * dtypeBytes(dtype) > maxBindingSize
  ) {
    return null;
  }
  return planCastDirectCore(tensor, dtype);
}

/** Unguarded cast plan core — cast()'s direct tail (post-routing). */
export function planCastDirectCore(
  tensor: WebGPUTensor,
  dtype: DType,
): ElementwiseDirectPlan {
  const totalWorkgroups = Math.ceil(tensor.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const shader = castTileIR(
    tensor.dtype as "f32" | "f16" | "i32" | "u32",
    dtype as "f32" | "f16" | "i32" | "u32",
    tensor.shape,
    tensor.strides,
    tensor.offset,
  );
  const key = `cast:${tensor.dtype}->${dtype}:${tensor.shape.join("x")}:${tensor.strides.join(",")}:${tensor.offset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  return {
    key,
    shader,
    outputSizeBytes: tensor.size * dtypeBytes(dtype),
    paramsData: params(tensor.size),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  };
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

  // Validate broadcast compatibility, then use the shared single-source meta.
  // (expandMeta assumes already-validated shapes; keep the explicit check here
  // so the error message is preserved for callers.)
  const padded = shape.length - inputShape.length;
  for (let i = padded; i < shape.length; i++) {
    const inputDim = inputShape[i - padded];
    const outputDim = shape[i];
    if (inputDim !== 1 && inputDim !== outputDim) {
      throw new Error(
        `expand: incompatible shapes at dimension ${i} (input ${inputDim} vs output ${outputDim})`,
      );
    }
  }
  const m = expandMeta(
    { shape: inputShape, strides: inputStrides, offset: tensor.offset },
    shape,
  );

  // Return a view sharing the same buffer
  // Note: expand views are never contiguous (they have stride=0 somewhere)
  // View - does not own the buffer
  return createTensor(
    m.shape,
    tensor.buffer,
    m.strides,
    m.offset,
    tensor.dtype,
    false,
  );
}

/**
 * THE raw-bind safety predicate (single source of truth at the seam).
 *
 * `isContiguous` is computed from STRIDES ONLY — a narrow(dim0, start>0) view
 * has contiguous strides but a non-zero element OFFSET into its base buffer.
 * Any path that binds `t.buffer` starting at element 0 (readback copies,
 * fused kernels taking raw GPUBuffers, flat tile-IR kernels) must use THIS
 * predicate, never bare `isContiguous`: treating offset>0 views as bindable
 * silently reads the WRONG REGION with the right shape (task #58).
 */
export function isRawBindable(t: WebGPUTensor): boolean {
  return t.isContiguous && (t.offset ?? 0) === 0;
}

/**
 * LOUD guard for raw-bind sites: throws if the tensor cannot be bound from
 * element 0. Place after the site's materialization/offset-fold so that any
 * future regression of the prep logic fails loudly instead of silently
 * reading the wrong region.
 */
export function assertRawBindable(t: WebGPUTensor, site: string): WebGPUTensor {
  if (!isRawBindable(t)) {
    throw new Error(
      `[offset-view] ${site}: tensor bound raw from element 0 but has ` +
        `offset=${t.offset} strides=[${t.strides.join(",")}] shape=[${t.shape.join(",")}]. ` +
        `Route through ensureContiguous()/asContiguous() or fold the offset into the kernel's indexing.`,
    );
  }
  return t;
}

/**
 * Materialize a non-contiguous OR offset (element offset > 0) tensor to a new
 * contiguous buffer at offset 0. If already raw-bindable, returns a
 * non-owning view (no-op).
 * Handles large tensors by processing in chunks.
 */
export function contiguous(a: BackendTensor): BackendTensor {
  const tensor = asGPUTensor(a);

  // Fast path: already contiguous AND offset 0 - return a non-owning view.
  // Returning the same tensor object would cause the executor to create a
  // second StorageHandle for the same GPUBuffer.  When the intermediate
  // handle becomes unreachable, destroyUnreachable() would destroy the
  // buffer while the original tensor still references it.
  // NOTE offset>0 views with contiguous strides do NOT take the fast path:
  // contiguous() is the designated materialization point for raw-bind
  // consumers, so its result must start at element 0 of its buffer.
  if (isRawBindable(tensor)) {
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

  // Offset-only view (contiguous strides, offset>0) of a base buffer LARGER
  // than maxStorageBufferBindingSize: pure DMA copy from the byte offset —
  // copyBufferToBuffer is not subject to the binding limit, covering narrow
  // views of huge bases the chunked shader path can't (it only supports 2D
  // transposes). Requires 4-byte-aligned offset+size per the WebGPU copy
  // rules. Deliberately NOT used at normal sizes: the stream generator
  // mirrors contiguous() as a contiguousTileIR DISPATCH (planContigCopy /
  // the contiguous direct-op handler), and routing small views through a
  // COPY here would diverge recording vs generation at that seam.
  {
    const bpe = dtypeBytes(tensor.dtype);
    const offsetBytes = tensor.offset * bpe;
    const dataBytes = outSize * bpe;
    const maxBinding =
      requireContext().device.limits?.maxStorageBufferBindingSize ??
      128 * 1024 * 1024;
    if (
      tensor.isContiguous &&
      tensor.offset > 0 &&
      tensor.buffer.size > maxBinding &&
      offsetBytes % 4 === 0 &&
      dataBytes % 4 === 0 &&
      offsetBytes + dataBytes <= tensor.buffer.size
    ) {
      const ctx = requireContext();
      const outBuffer = resolveOutputBuffer(ctx.device, dataBytes, [
        tensor.buffer,
      ]);
      trackSharedEncoderWrite(outBuffer);
      const enc = getSharedEncoderInstance();
      if (enc) {
        recordedCopyBufferToBuffer(
          enc,
          tensor.buffer,
          offsetBytes,
          outBuffer,
          0,
          dataBytes,
        );
      } else {
        const cmdEnc = ctx.device.createCommandEncoder();
        cmdEnc.copyBufferToBuffer(
          tensor.buffer,
          offsetBytes,
          outBuffer,
          0,
          dataBytes,
        );
        submitOrCollect(cmdEnc.finish());
      }
      return createTensor(shape, outBuffer, undefined, 0, tensor.dtype);
    }
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
  // Single-sourced with the stream generator.
  const plan = planContiguousDirectCore(tensor);
  const outBuffer = resolveOutputBuffer(ctx.device, plan.outputSizeBytes, [
    tensor.buffer,
  ]);
  dispatchElementwise({
    key: plan.key,
    shader: plan.shader,
    inputs: [tensor.buffer],
    outputSizeBytes: plan.outputSizeBytes,
    params: plan.paramsData,
    outBuffer,
    dispatchX: plan.dispatchX,
    dispatchY: plan.dispatchY,
  });
  return createTensor(tensor.shape, outBuffer, undefined, 0, tensor.dtype);
}

/** Stage-4 plan/encode split for the DIRECT contiguous path. Null where
 *  contiguous() routes to chunked dispatch. The already-contiguous fast
 *  path (no commands at all) is the CALLER's to mirror. */
export function planContiguousDirect(
  tensor: WebGPUTensor,
): ElementwiseDirectPlan | null {
  const ctx = requireContext();
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  if (tensor.buffer.size > maxBindingSize) return null;
  return planContiguousDirectCore(tensor);
}

/** Unguarded contiguous plan core — contiguousDirect (post-routing). */
export function planContiguousDirectCore(
  tensor: WebGPUTensor,
): ElementwiseDirectPlan {
  const shape = tensor.shape;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const bytesPerElement = dtypeBytes(dtype);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const shader = contiguousTileIR(
    shape,
    tensor.strides,
    tensor.offset,
    dtype as "f32" | "f16" | "i32" | "u32",
  );
  const key = `contiguous:${shape.join(",")}:${tensor.strides.join(",")}:${tensor.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  return {
    key,
    shader,
    outputSizeBytes: outSize * bytesPerElement,
    paramsData: params(outSize),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  };
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
  const wgslType = dtype;
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
 * Helper to ensure a tensor is RAW-BINDABLE (contiguous strides AND element
 * offset 0), materializing if needed. This is the sanctioned prep for any
 * kernel that binds the buffer from element 0 — a contiguous-strides view
 * with offset>0 (e.g. narrow along dim 0) MUST be materialized here, or the
 * kernel silently reads the wrong region.
 */
export function ensureContiguous(tensor: WebGPUTensor): WebGPUTensor {
  if (isRawBindable(tensor)) return tensor;
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
  const m = narrowMeta(
    { shape: tensor.shape, strides: tensor.strides, offset: tensor.offset },
    dim,
    start,
    length,
  );
  return createTensor(
    m.shape,
    tensor.buffer,
    m.strides,
    m.offset,
    tensor.dtype,
    false,
  );
}

/**
 * Backward for narrow: pad gradient back to original shape.
 * Writes grad into [start, start+length) along dim, zeros elsewhere.
 */
/**
 * Stage-4 plan/encode for narrowBackward (pad a narrowed grad back to the
 * original length). Geometry is fully derivable from the grad SHAPE +
 * dim/start/originalLength + dtype — no live buffer needed, so the stream
 * generator computes it from the node (input shape + payload) post-hoc.
 * Output ALLOC is resolveOutputBuffer (allocKind 0) with the grad as the
 * aliasing input; binding order [grad, out, params].
 */
export function planNarrowBackward(
  gradShape: number[],
  dim: number,
  start: number,
  originalLength: number,
  dtype: DType,
): ElementwiseDirectPlan & { outShape: number[] } {
  const outShape = gradShape.slice();
  outShape[dim] = originalLength;
  const outSize = sizeOf(outShape);
  const bytesPerElement = dtypeBytes(dtype);
  const outerSize = sizeOf(outShape.slice(0, dim));
  const innerSize = sizeOf(outShape.slice(dim + 1));
  const gradDimSize = gradShape[dim]; // = length from narrow
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const gridSizeX = dispatch.x * WORKGROUP_SIZE;
  const shader = narrowBackwardTileIR(
    originalLength,
    outSize,
    dtype as "f32" | "f16" | "i32" | "u32",
  );
  const key = `narrowBackward:${originalLength}:${gradDimSize}:${start}:${outSize}:${dtype}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  return {
    key,
    shader,
    outputSizeBytes: outSize * bytesPerElement,
    paramsData: params(outerSize, innerSize, gradDimSize, start),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
    outShape,
  };
}

export function narrowBackward(
  grad: BackendTensor,
  dim: number,
  start: number,
  originalLength: number,
): BackendTensor {
  const gradTensor = ensureContiguous(asGPUTensor(grad));
  const dtype = gradTensor.dtype;
  const plan = planNarrowBackward(
    gradTensor.shape,
    dim,
    start,
    originalLength,
    dtype,
  );

  const outBuffer = dispatchElementwise({
    key: plan.key,
    shader: plan.shader,
    inputs: [gradTensor.buffer],
    outputSizeBytes: plan.outputSizeBytes,
    params: plan.paramsData,
    dispatchX: plan.dispatchX,
    dispatchY: plan.dispatchY,
  });

  if (gradTensor !== asGPUTensor(grad)) destroyCopy(gradTensor);

  return createTensor(plan.outShape, outBuffer, undefined, 0, dtype);
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

  // Reorder shape and strides according to dims (shared single-source meta).
  const m = permuteMeta(
    { shape: inputShape, strides: tensor.strides, offset: tensor.offset },
    dims,
  );

  // Return a view sharing the same buffer
  // View - does not own the buffer
  return createTensor(
    m.shape,
    tensor.buffer,
    m.strides,
    m.offset,
    tensor.dtype,
    false,
  );
}
