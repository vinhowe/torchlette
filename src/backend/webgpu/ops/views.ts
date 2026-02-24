/**
 * View and cast ops: cast, reshape, expand, contiguous, narrow, transpose, permute,
 * detectSimpleTranspose, ensureContiguous.
 * Extracted from index.ts — purely structural refactoring.
 */
import type { BackendTensor, DType, TransposeOptions } from "../../types";
import type { GPUBuffer, GPUDevice, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage, STORAGE_BUFFER_USAGE } from "../gpu-types";
import {
  sizeOf, WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM, compute2DDispatch,
  contiguousStrides, checkContiguousStrides, dtypeBytes, dtypeToWgsl,
  alignBufferSize, gcd, lcm,
} from "../shape-utils";
import { requireContext, isF16Supported, f16WeightCache, f32ArrayToF16Array, f16ArrayToF32Array, f16ToF32, f32ToF16 } from "../gpu-context";
import { dispatchComputePass, dispatchElementwise, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import { bufferPool } from "../buffer-pool";
import { resolveOutputBuffer } from "../buffer-arena";
import {
  cachedCreateBindGroup, createParamsBuffer, releaseParamsBuffer,
  createUniformBuffer, releaseUniformBuffer, profiledCreateBindGroup,
  params1, params2, params4, params7,
} from "../bind-group-cache";
import { profileApiCall } from "../profiler";
import { computeFlatChunkLayout, dispatchFlatChunked } from "../chunked-dispatch";

function castShader(
  srcDtype: DType,
  dstDtype: DType,
  shape: number[],
  strides: number[],
  offset: number,
  gridSizeX?: number,
): string {
  const rank = shape.length;
  const srcWgsl = dtypeToWgsl(srcDtype);
  const dstWgsl = dtypeToWgsl(dstDtype);
  const enableF16 =
    srcDtype === "f16" || dstDtype === "f16" ? "enable f16;\n" : "";
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  // Generate cast expression based on dtype pair
  let castExpr: string;
  if (srcDtype === dstDtype) {
    castExpr = "x";
  } else if (dstDtype === "f32") {
    castExpr = "f32(x)";
  } else if (dstDtype === "f16") {
    castExpr = "f16(x)";
  } else if (dstDtype === "i32") {
    castExpr = "i32(x)";
  } else if (dstDtype === "u32") {
    castExpr = "u32(x)";
  } else {
    castExpr = `${dstWgsl}(x)`;
  }

  if (rank === 0) {
    // Scalar case
    return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${srcWgsl}>;
@group(0) @binding(1) var<storage, read_write> out: array<${dstWgsl}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  let x = a[${offset}u];
  out[idx] = ${castExpr};
}
`;
  }

  const shapeArray = `array<u32, ${rank}>(${shape.map((s) => `${s}u`).join(", ")})`;
  const stridesArray = `array<u32, ${rank}>(${strides.map((s) => `${s}u`).join(", ")})`;

  return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${srcWgsl}>;
@group(0) @binding(1) var<storage, read_write> out: array<${dstWgsl}>;
@group(0) @binding(2) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const SHAPE = ${shapeArray};
const STRIDES = ${stridesArray};
const OFFSET: u32 = ${offset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert flat index to strided offset
  var remaining = idx;
  var inputOffset = OFFSET;
  for (var d = 0u; d < RANK; d = d + 1u) {
    var dimSize = 1u;
    for (var j = d + 1u; j < RANK; j = j + 1u) {
      dimSize = dimSize * SHAPE[j];
    }
    let coord = remaining / dimSize;
    remaining = remaining % dimSize;
    inputOffset = inputOffset + coord * STRIDES[d];
  }

  let x = a[inputOffset];
  out[idx] = ${castExpr};
}
`;
}

/**
 * Cast tensor to a different dtype.
 * Returns same tensor if already the target dtype.
 */
export function cast(a: BackendTensor, dtype: DType): BackendTensor {
  const tensor = a as WebGPUTensor;
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
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const inputDataBytes = tensor.size * srcBytesPerElement;
  const outputDataBytes = tensor.size * dstBytesPerElement;

  if (inputDataBytes > maxBindingSize || outputDataBytes > maxBindingSize) {
    // Chunked cast requires contiguous input
    let src = tensor;
    let contiguousCopy: WebGPUTensor | null = null;
    if (!tensor.isContiguous || tensor.offset > 0) {
      src = contiguous(tensor) as WebGPUTensor;
      contiguousCopy = src;
    }
    const result = castChunked(src, dtype, ctx, maxBindingSize, limits);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(tensor.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = castShader(
    tensor.dtype,
    dtype,
    tensor.shape,
    tensor.strides,
    tensor.offset,
    use2D ? dispatch.gridSizeX : undefined,
  );
  const key = `cast:${tensor.dtype}->${dtype}:${tensor.shape.join("x")}:${tensor.strides.join(",")}:${tensor.offset}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const bytesPerElement = dtypeBytes(dtype);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    tensor.size * bytesPerElement,
    [tensor.buffer],
  );
  const params = createUniformBuffer(ctx.device, tensor.size);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, params]);
  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(params);
  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

/**
 * Chunked cast dispatch for tensors exceeding maxStorageBufferBindingSize.
 * Input must be contiguous with offset 0.
 */
function castChunked(
  tensor: WebGPUTensor,
  dtype: DType,
  ctx: ReturnType<typeof requireContext>,
  maxBindingSize: number,
  limits: Record<string, number>,
): BackendTensor {
  const srcBytesPerElement = dtypeBytes(tensor.dtype);
  const dstBytesPerElement = dtypeBytes(dtype);
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Alignment must satisfy both src and dst offset alignment requirements
  const srcElemsPerAlign = minAlignment / srcBytesPerElement;
  const dstElemsPerAlign = minAlignment / dstBytesPerElement;
  const elementsPerAlignment = lcm(srcElemsPerAlign, dstElemsPerAlign);

  const totalElements = tensor.size;
  const maxBpe = Math.max(srcBytesPerElement, dstBytesPerElement);

  const layout = computeFlatChunkLayout(
    totalElements, maxBpe, maxBindingSize, minAlignment, elementsPerAlignment,
  );

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    totalElements * dstBytesPerElement,
    [tensor.buffer],
  );

  // Build shader for chunked cast (contiguous, no strides/offset)
  const srcWgslType = dtypeToWgsl(tensor.dtype);
  const dstWgslType = dtypeToWgsl(dtype);
  const f16Enable = (tensor.dtype === "f16" || dtype === "f16") ? "enable f16;\n" : "";
  const idxCompute = layout.use2D
    ? `let idx = gid.x + gid.y * ${layout.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `${f16Enable}
@group(0) @binding(0) var<storage, read> a: array<${srcWgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${dstWgslType}>;

struct Params {
  chunkSize: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  out[idx] = ${dstWgslType}(a[idx]);
}
`;

  const key = `castChunked:${tensor.dtype}->${dtype}:${layout.use2D ? `2d:${layout.gridSizeX}` : "1d"}`;

  dispatchFlatChunked({
    key, shader: code, layout,
    inputs: [{ buffer: tensor.buffer, mode: "chunked", bytesPerElement: srcBytesPerElement }],
    outBuffer, outBytesPerElement: dstBytesPerElement, totalElements,
  });

  return createTensor(tensor.shape, outBuffer, undefined, 0, dtype);
}

/**
 * Infer strides for a new shape given old shape/strides, without copying data.
 * Returns null if the reshape requires a contiguous copy.
 * Implements PyTorch's computeStride algorithm.
 */
export function inferReshapeStrides(
  oldShape: number[],
  oldStrides: number[],
  newShape: number[],
): number[] | null {
  if (newShape.length === 0) return [];
  if (oldShape.length === 0) return contiguousStrides(newShape);

  const newStrides = new Array<number>(newShape.length);

  // Work through old and new dims left-to-right, grouping contiguous chunks
  let oldIdx = 0;
  let newIdx = 0;
  const oldN = oldShape.length;
  const newN = newShape.length;

  while (newIdx < newN) {
    // Skip size-1 dims in new shape (stride is irrelevant)
    if (newShape[newIdx] === 1) {
      // Use a sensible stride for size-1 dims
      newStrides[newIdx] = newIdx + 1 < newN ? newStrides[newIdx + 1] || 1 : 1;
      newIdx++;
      continue;
    }

    // Skip size-1 dims in old shape
    while (oldIdx < oldN && oldShape[oldIdx] === 1) oldIdx++;
    if (oldIdx >= oldN) return null;

    // Accumulate a group of old dims and match to new dims
    let oldProduct = oldShape[oldIdx];
    let newProduct = newShape[newIdx];

    // Collect old dims until oldProduct >= newProduct
    const groupStart = oldIdx;
    while (oldProduct < newProduct && oldIdx + 1 < oldN) {
      // Check contiguity between consecutive old dims
      if (oldStrides[oldIdx] !== oldStrides[oldIdx + 1] * oldShape[oldIdx + 1]) {
        return null; // Non-contiguous boundary
      }
      oldIdx++;
      // Skip size-1 old dims within group
      if (oldShape[oldIdx] === 1) continue;
      oldProduct *= oldShape[oldIdx];
    }

    // Collect new dims until newProduct >= oldProduct
    const newGroupStart = newIdx;
    while (newProduct < oldProduct && newIdx + 1 < newN) {
      newIdx++;
      if (newShape[newIdx] === 1) {
        newStrides[newIdx] = 1;
        continue;
      }
      newProduct *= newShape[newIdx];
    }

    if (oldProduct !== newProduct) return null;

    // Assign strides for the new dims in this group (right-to-left)
    // The rightmost new dim gets the stride of the rightmost old dim
    let stride = oldStrides[oldIdx];
    for (let i = newIdx; i >= newGroupStart; i--) {
      if (newShape[i] === 1) {
        newStrides[i] = stride;
        continue;
      }
      newStrides[i] = stride;
      stride *= newShape[i];
    }

    oldIdx++;
    newIdx++;
  }

  // Check remaining old dims are all size 1
  while (oldIdx < oldN) {
    if (oldShape[oldIdx] !== 1) return null;
    oldIdx++;
  }

  return newStrides;
}

export function reshape(a: BackendTensor, shape: number[]): BackendTensor {
  const tensor = a as WebGPUTensor;
  const expected = sizeOf(shape);
  if (expected !== tensor.size) {
    throw new Error("View shape does not match tensor size");
  }

  if (tensor.isContiguous) {
    // Fast path: contiguous input → contiguous output view
    return createTensor(shape, tensor.buffer, undefined, tensor.offset, tensor.dtype, false);
  }

  // Non-contiguous: try to compute valid strides for new shape
  const newStrides = inferReshapeStrides(tensor.shape, tensor.strides, shape);
  if (newStrides !== null) {
    // Compatible layout: return view with computed strides (zero-cost)
    return createTensor(shape, tensor.buffer, newStrides, tensor.offset, tensor.dtype, false);
  }

  // Incompatible: must materialize first, transfer buffer ownership to result
  const contig = contiguous(tensor) as WebGPUTensor;
  bufferPool.decRef(contig.buffer); // Transfer ownership to result tensor
  return createTensor(shape, contig.buffer, undefined,
    contig.offset, tensor.dtype, true);
}

/**
 * Expand returns a VIEW - no data copy, just metadata change.
 * Broadcast dimensions get stride=0 (same element repeated).
 */
export function expand(a: BackendTensor, shape: number[]): BackendTensor {
  const tensor = a as WebGPUTensor;
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
  const tensor = a as WebGPUTensor;

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

  const ctx = requireContext();
  const shape = tensor.shape;
  const rank = shape.length;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  if (outSize === 0) {
    throw new Error("contiguous: empty tensors not supported");
  }

  // Check if input buffer exceeds max binding size
  const limits = (ctx.device as unknown as { limits: Record<string, number> }).limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Get actual buffer size (the backing storage might be larger than the view)
  const inputBufferSize = (tensor.buffer as { size: number }).size;

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
  const rank = shape.length;
  const outSize = sizeOf(shape);
  const dtype = tensor.dtype;
  const wgslType = dtypeToWgsl(dtype);
  const bytesPerElement = dtypeBytes(dtype);

  // Compute 2D dispatch for large tensors (> 65535 workgroups)
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  // Create new contiguous buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [tensor.buffer],
  );

  // Generate shader that reads with strides and writes contiguous
  const shapeArray = `array<u32, ${rank}>(${shape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${tensor.strides.map((s) => `${s}u`).join(", ")})`;
  const outStridesArray = `array<u32, ${rank}>(${contiguousStrides(shape)
    .map((s) => `${s}u`)
    .join(", ")})`;
  const enableF16 = dtype === "f16" ? "enable f16;\n" : "";
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `${enableF16}
struct Params {
  size: u32,
  offset: u32,
};

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const shape = ${shapeArray};
const inputStrides = ${inputStridesArray};
const outStrides = ${outStridesArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert output flat index to coordinates
  var coords: array<u32, ${rank}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }

  // Compute input offset using strides
  var inputOffset = params.offset;
  for (var d = 0u; d < RANK; d = d + 1u) {
    inputOffset = inputOffset + coords[d] * inputStrides[d];
  }

  out[idx] = input[inputOffset];
}
`;

  const key = `contiguous:${shape.join(",")}:${tensor.strides.join(",")}:${tensor.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;

  dispatchElementwise({
    key, shader: code,
    inputs: [tensor.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params2(outSize, tensor.offset),
    outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // For transposed tensor [K, N] with strides [1, K]:
  // Output column j reads from input[j*K : j*K+K] (contiguous in input)
  // We chunk by both input columns AND output rows to stay within binding limits
  if (strideK === 1 && strideN === K) {
    // This is a simple transpose - output column j reads input rows j
    // Challenge: output is also large and needs chunking
    // Strategy: process in row chunks where each row chunk handles all columns in sub-chunks

    const bytesPerOutputRow = N * bytesPerElement; // Output row [K elements across N columns for one K-row]
    const bytesPerInputRow = K * bytesPerElement;  // One column of input (K elements)

    // How many output rows fit in binding limit?
    const maxOutputRows = Math.floor(maxBindingSize / bytesPerOutputRow);

    // How many input columns (= K elements each) fit in binding limit?
    const maxInputCols = Math.floor(maxBindingSize / bytesPerInputRow);

    // Align row counts for buffer offsets
    const outputRowAlignment = minAlignment / gcd(bytesPerOutputRow, minAlignment);
    const inputColAlignment = minAlignment / gcd(bytesPerInputRow, minAlignment);

    const alignedOutputRows = Math.max(
      outputRowAlignment,
      Math.floor(maxOutputRows / outputRowAlignment) * outputRowAlignment
    );
    const alignedInputCols = Math.max(
      inputColAlignment,
      Math.floor(maxInputCols / inputColAlignment) * inputColAlignment
    );

    // Process output in row chunks, and within each row chunk, process input in column chunks
    const numOutputRowChunks = Math.ceil(K / alignedOutputRows);
    const numInputColChunks = Math.ceil(N / alignedInputCols);

    // Shader processes a tile: output rows [rowStart, rowEnd) and columns [colStart, colEnd)
    const code = `
struct Params {
  K: u32,
  N: u32,
  rowStart: u32,
  rowEnd: u32,
  colStart: u32,
  colEnd: u32,
  gridStride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> output: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.gridStride + gid.x;
  let numRows = params.rowEnd - params.rowStart;
  let numCols = params.colEnd - params.colStart;
  let tileSize = numRows * numCols;
  if (idx >= tileSize) { return; }

  let localRow = idx / numCols;
  let localCol = idx % numCols;
  let globalRow = params.rowStart + localRow;
  let globalCol = params.colStart + localCol;

  // Input: transposed view element [globalRow, globalCol] = buffer[globalCol * K + globalRow]
  // With offset binding at colStart * K, local index is localCol * K + globalRow
  // But we need to account for the fact that globalRow is the actual row index
  let inputIdx = localCol * params.K + globalRow;

  // Output: row chunk is bound with offset, so local row index is localRow
  // Column index is the global column
  let outputIdx = localRow * params.N + globalCol;

  output[outputIdx] = input[inputIdx];
}
`;

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
        const inputByteOffset = (tensor.offset + colStart * K) * bytesPerElement;
        const inputChunkSize = numCols * K * bytesPerElement;

        const tileSize = numRows * numCols;
        const dispatch = compute2DDispatch(Math.ceil(tileSize / WORKGROUP_SIZE));

        const paramsBuffer = createParamsBuffer(ctx.device, params7(K, N, rowStart, rowEnd, colStart, colEnd, dispatch.gridSizeX * WORKGROUP_SIZE));

        const bindGroup = profiledCreateBindGroup(ctx.device,{
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
    `Chunked contiguous not yet implemented for stride pattern [${tensor.strides.join(", ")}]`
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
  return contiguous(tensor) as WebGPUTensor;
}

/**
 * Select a contiguous sub-range along one dimension. Returns a view (zero GPU cost).
 * The returned tensor shares the same buffer with an adjusted offset.
 */
export function narrow(a: BackendTensor, dim: number, start: number, length: number): BackendTensor {
  const tensor = a as WebGPUTensor;
  const rank = tensor.shape.length;
  if (dim < 0 || dim >= rank) {
    throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
  }
  if (start < 0 || start + length > tensor.shape[dim]) {
    throw new Error(`narrow: range [${start}, ${start + length}) out of bounds for dim size ${tensor.shape[dim]}`);
  }
  const newShape = tensor.shape.slice();
  newShape[dim] = length;
  const newOffset = tensor.offset + start * tensor.strides[dim];
  return createTensor(newShape, tensor.buffer, tensor.strides.slice(), newOffset, tensor.dtype, false);
}

/**
 * Backward for narrow: pad gradient back to original shape.
 * Writes grad into [start, start+length) along dim, zeros elsewhere.
 */
export function narrowBackward(grad: BackendTensor, dim: number, start: number, originalLength: number): BackendTensor {
  const gradTensor = ensureContiguous(grad as WebGPUTensor);
  const ctx = requireContext();

  const outShape = gradTensor.shape.slice();
  outShape[dim] = originalLength;
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const dtype = gradTensor.dtype;
  const bytesPerElement = dtype === "f16" ? 2 : 4;

  const outerSize = outShape.slice(0, dim).reduce((a, b) => a * b, 1);
  const innerSize = outShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const gradDimSize = gradTensor.shape[dim]; // = length from narrow
  const outDimSize = originalLength;

  const wgslType = dtype === "f16" ? "f16" : "f32";
  const WG = WORKGROUP_SIZE;

  const totalWorkgroups = Math.ceil(outSize / WG);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const gridSizeX = dispatch.x * WG;

  const shaderCode = `
${dtype === "f16" ? "enable f16;\n" : ""}
@group(0) @binding(0) var<storage, read> grad: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // outerSize, innerSize, gradDimSize, start

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${use2D ? `gid.x + gid.y * ${gridSizeX}u` : "gid.x"};
  let total = ${outSize}u;
  if (idx >= total) { return; }

  let innerSize = params.y;
  let outDimSize = ${outDimSize}u;
  let outerIdx = idx / (outDimSize * innerSize);
  let remainder = idx % (outDimSize * innerSize);
  let dimIdx = remainder / innerSize;
  let innerIdx = remainder % innerSize;

  let startOffset = params.w;
  if (dimIdx >= startOffset && dimIdx < startOffset + params.z) {
    let gradDimIdx = dimIdx - startOffset;
    let gradIdx = outerIdx * params.z * innerSize + gradDimIdx * innerSize + innerIdx;
    out[idx] = grad[gradIdx];
  } else {
    out[idx] = ${wgslType}(0.0);
  }
}
`;

  const key = `narrowBackward:${outDimSize}:${gradDimSize}:${start}:${outSize}:${dtype}:${use2D ? `2d:${gridSizeX}` : "1d"}`;

  const outBuffer = dispatchElementwise({
    key, shader: shaderCode,
    inputs: [gradTensor.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params4(outerSize, innerSize, gradDimSize, start),
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  if (gradTensor !== (grad as WebGPUTensor)) {
    bufferPool.decRef(gradTensor.buffer);
    bufferPool.deferredDestroy(gradTensor.buffer, gradTensor.size * bytesPerElement);
  }

  return createTensor(outShape, outBuffer, undefined, 0, dtype);
}

/**
 * Transpose returns a VIEW - no data copy, just metadata change.
 * The returned tensor shares the same buffer but with swapped strides.
 */
export function transpose(a: BackendTensor, options: TransposeOptions): BackendTensor {
  const tensor = a as WebGPUTensor;
  const { dim0, dim1 } = options;
  const inputShape = tensor.shape;
  const rank = inputShape.length;

  if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
    throw new Error("transpose: dimension out of range");
  }

  // Swap shape dimensions
  const outShape = inputShape.slice();
  outShape[dim0] = inputShape[dim1];
  outShape[dim1] = inputShape[dim0];

  // Swap strides - this is the key to view-based transpose
  const outStrides = tensor.strides.slice();
  outStrides[dim0] = tensor.strides[dim1];
  outStrides[dim1] = tensor.strides[dim0];

  // Return a view sharing the same buffer
  // Note: createTensor will correctly compute isContiguous=false for transposed strides
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

/**
 * Permute dimensions according to the given order.
 * Returns a view sharing the same buffer (no data copy).
 *
 * Example: permute([2, 3, 4], [2, 0, 1]) -> [4, 2, 3]
 */
export function permute(a: BackendTensor, dims: number[]): BackendTensor {
  const tensor = a as WebGPUTensor;
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
