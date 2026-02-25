/**
 * Matrix multiplication ops: matmul (wrapper + chunked variants), sliceColumns, scatterColumnsToOutput.
 * Extracted from index.ts — purely structural refactoring.
 */

import type { BackendTensor } from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { WORKGROUP_SIZE, compute2DDispatch, dtypeBytes, broadcastShapes, gcd } from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchComputePass, dispatchMatmul, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import {
  cachedCreateBindGroup,
  profiledCreateBindGroup,
  createParamsBuffer,
  releaseParamsBuffer,
  params5,
  params6,
  params7,
} from "../bind-group-cache";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import { ensureContiguous } from "./views";

/** Local type alias for GPU buffer binding descriptors with offset/size. */
type GPUBufferBinding = { buffer: GPUBuffer; offset?: number; size?: number };

export function matmul(
  _a: BackendTensor,
  _b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const ctx = requireContext();
  const a = _a as WebGPUTensor;
  const b = _b as WebGPUTensor;

  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Check if B matrix exceeds max buffer binding size
  const bSizeBytes = b.size * dtypeBytes(b.dtype);

  if (bSizeBytes > maxBindingSize) {
    // Chunked path for large B matrices
    return matmulChunked(a, b, maxBindingSize);
  }

  // Compute output size to check if it exceeds limit
  // Output shape: broadcast(a[:-1], b[:-2]) + [a[-2], b[-1]]
  const aShape = a.shape;
  const bShape = b.shape;
  const M = aShape[aShape.length - 2];
  const N = bShape[bShape.length - 1];
  // Compute batch dimensions
  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch.length > 0 ? aBatch : [1], bBatch.length > 0 ? bBatch : [1]);
  const outShape = [...batchShape, M, N];
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outputDtype = (a.dtype === "f32" || b.dtype === "f32") ? "f32" as const : a.dtype;
  const outSizeBytes = outSize * dtypeBytes(outputDtype);

  if (outSizeBytes > maxBindingSize) {
    // Chunked path for large output
    return matmulChunkedOutput(a, b, maxBindingSize);
  }

  // Fast path: existing implementation
  return dispatchMatmul(a, b, false, false, options?.outBuffer);
}

/**
 * Extract column slice from a 2D matrix to a contiguous buffer.
 * Input: [K, N], Output: [K, colEnd - colStart]
 * Handles large input matrices by processing in row chunks.
 */
function sliceColumns(
  input: WebGPUTensor,
  colStart: number,
  colEnd: number,
): WebGPUTensor {
  const ctx = requireContext();
  const [K, N] = input.shape;
  const sliceWidth = colEnd - colStart;
  const outSize = K * sliceWidth;

  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Check if input exceeds binding limit
  const inputSizeBytes = input.size * 4;
  const needsChunking = inputSizeBytes > maxBindingSize;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Shader to copy column slice (with row offset support for chunking)
  const shaderCode = `
    struct Params {
      numRows: u32,
      N: u32,
      colStart: u32,
      sliceWidth: u32,
      rowStart: u32,
      gridStride: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.y * params.gridStride + gid.x;
      let totalSize = params.numRows * params.sliceWidth;
      if (idx >= totalSize) { return; }

      let localRow = idx / params.sliceWidth;
      let col = idx % params.sliceWidth;
      let srcCol = params.colStart + col;
      // Input offset is relative to chunk start (row 0 of bound range)
      let srcIdx = localRow * params.N + srcCol;
      // Output offset accounts for rowStart
      let dstIdx = (params.rowStart + localRow) * params.sliceWidth + col;

      output[dstIdx] = input[srcIdx];
    }
  `;

  const module = ctx.device.createShaderModule({ code: shaderCode });
  const pipeline = ctx.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  if (!needsChunking) {
    // Fast path: single dispatch
    const dispatch = compute2DDispatch(Math.ceil(outSize / WORKGROUP_SIZE));
    const paramsBuffer = createParamsBuffer(ctx.device, params6(K, N, colStart, sliceWidth, 0, dispatch.gridSizeX * WORKGROUP_SIZE));

    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [input.buffer, outBuffer, paramsBuffer]);

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

    releaseParamsBuffer(paramsBuffer);
  } else {
    // Chunked path: process rows in chunks that fit in binding limit
    const bytesPerRow = N * 4;

    // Calculate how many rows must group together for aligned offsets
    // We need rowStart * bytesPerRow to be divisible by minAlignment
    // Find the smallest rowAlignment where rowAlignment * bytesPerRow % minAlignment == 0
    const g_ = gcd(bytesPerRow, minAlignment);
    const rowAlignment = minAlignment / g_;

    // How many rows fit in maxBindingSize?
    const maxRowsUnaligned = Math.floor(maxBindingSize / bytesPerRow);
    // Round down to nearest multiple of rowAlignment
    const rowsPerChunk = Math.max(rowAlignment, Math.floor(maxRowsUnaligned / rowAlignment) * rowAlignment);

    const numRowChunks = Math.ceil(K / rowsPerChunk);

    for (let chunk = 0; chunk < numRowChunks; chunk++) {
      const rowStart = chunk * rowsPerChunk;
      const rowEnd = Math.min(rowStart + rowsPerChunk, K);
      const numRows = rowEnd - rowStart;

      // Calculate byte offset and size for this chunk
      const byteOffset = rowStart * bytesPerRow;
      const chunkByteSize = numRows * bytesPerRow;

      const chunkSize = numRows * sliceWidth;
      const dispatch = compute2DDispatch(Math.ceil(chunkSize / WORKGROUP_SIZE));

      const paramsBuffer = createParamsBuffer(ctx.device, params6(numRows, N, colStart, sliceWidth, rowStart, dispatch.gridSizeX * WORKGROUP_SIZE));

      const bindGroup = profiledCreateBindGroup(ctx.device,{
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: input.buffer,
              offset: byteOffset,
              size: chunkByteSize,
            } as GPUBufferBinding,
          },
          { binding: 1, resource: { buffer: outBuffer } },
          { binding: 2, resource: { buffer: paramsBuffer } },
        ],
      });

      dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

      releaseParamsBuffer(paramsBuffer);
    }
  }

  return createTensor([K, sliceWidth], outBuffer, undefined, 0, "f32", true);
}

/**
 * Write partial matmul result to columns of output buffer.
 * partial: [M, sliceWidth], output: [M, N], writes to columns [colStart, colStart+sliceWidth)
 * Handles large output buffers by processing in row chunks.
 */
function scatterColumnsToOutput(
  partial: WebGPUTensor,
  outBuffer: GPUBuffer,
  M: number,
  N: number,
  colStart: number,
): void {
  const ctx = requireContext();
  const sliceWidth = partial.shape[partial.shape.length - 1];
  const totalRows = partial.size / sliceWidth; // M (could be M*batch for batched)

  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Check if output buffer exceeds limit
  const outputBufferSize = outBuffer.size;
  const inputBufferSize = partial.buffer.size;

  const needsChunking = outputBufferSize > maxBindingSize || inputBufferSize > maxBindingSize;

  const shaderCode = `
    struct Params {
      numRows: u32,
      N: u32,
      colStart: u32,
      sliceWidth: u32,
      rowStart: u32,
      inputRowStart: u32,
      gridStride: u32,
    }

    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform> params: Params;

    @compute @workgroup_size(${WORKGROUP_SIZE})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx = gid.y * params.gridStride + gid.x;
      let totalSize = params.numRows * params.sliceWidth;
      if (idx >= totalSize) { return; }

      let localRow = idx / params.sliceWidth;
      let col = idx % params.sliceWidth;

      // Input index: relative to bound chunk
      let inputIdx = (params.inputRowStart + localRow) * params.sliceWidth + col;

      // Output: write to row (rowStart + localRow), column (colStart + col)
      // Output offset is relative to bound chunk
      let outputIdx = localRow * params.N + (params.colStart + col);

      output[outputIdx] = input[inputIdx];
    }
  `;

  const module = ctx.device.createShaderModule({ code: shaderCode });
  const pipeline = ctx.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  if (!needsChunking) {
    // Fast path: single dispatch
    const totalSize = totalRows * sliceWidth;
    const dispatch = compute2DDispatch(Math.ceil(totalSize / WORKGROUP_SIZE));
    const paramsBuffer = createParamsBuffer(ctx.device, params7(totalRows, N, colStart, sliceWidth, 0, 0, dispatch.gridSizeX * WORKGROUP_SIZE));

    const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [partial.buffer, outBuffer, paramsBuffer]);

    dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);

    releaseParamsBuffer(paramsBuffer);
  } else {
    // Chunked path: process rows in chunks
    // Output row size in bytes = N * 4
    // Input row size in bytes = sliceWidth * 4
    const outputBytesPerRow = N * 4;
    const inputBytesPerRow = sliceWidth * 4;

    // Find row chunk size that fits both input and output in binding limit
    const maxOutputRows = Math.floor(maxBindingSize / outputBytesPerRow);
    const maxInputRows = Math.floor(maxBindingSize / inputBytesPerRow);
    const maxRowsPerChunk = Math.min(maxOutputRows, maxInputRows);

    // Align for buffer offsets
    const outputG = gcd(outputBytesPerRow, minAlignment);
    const inputG = gcd(inputBytesPerRow, minAlignment);
    const outputRowAlignment = minAlignment / outputG;
    const inputRowAlignment = minAlignment / inputG;
    const rowAlignment = Math.max(outputRowAlignment, inputRowAlignment);

    const alignedRowsPerChunk = Math.max(
      rowAlignment,
      Math.floor(maxRowsPerChunk / rowAlignment) * rowAlignment
    );

    const numChunks = Math.ceil(totalRows / alignedRowsPerChunk);

    for (let chunk = 0; chunk < numChunks; chunk++) {
      const rowStart = chunk * alignedRowsPerChunk;
      const rowEnd = Math.min(rowStart + alignedRowsPerChunk, totalRows);
      const numRows = rowEnd - rowStart;

      // Calculate byte offsets
      const outputByteOffset = rowStart * outputBytesPerRow;
      const inputByteOffset = rowStart * inputBytesPerRow;

      const outputChunkSize = numRows * outputBytesPerRow;
      const inputChunkSize = numRows * inputBytesPerRow;

      const chunkSize = numRows * sliceWidth;
      const dispatch = compute2DDispatch(Math.ceil(chunkSize / WORKGROUP_SIZE));

      const paramsBuffer = createParamsBuffer(ctx.device, params7(numRows, N, colStart, sliceWidth, 0, 0, dispatch.gridSizeX * WORKGROUP_SIZE));

      const bindGroup = profiledCreateBindGroup(ctx.device,{
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: partial.buffer,
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
}

/**
 * Chunked matmul for large B matrices that exceed buffer binding limits.
 *
 * Strategy: Use the underlying buffer of B directly (even if B is a transposed view)
 * and chunk by BUFFER rows (which are contiguous), using transB=true.
 * This avoids the need to materialize a large contiguous copy.
 */
function matmulChunked(
  a: WebGPUTensor,
  b: WebGPUTensor,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const limits = ctx.device.limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Ensure A is contiguous (A is typically small)
  const aContiguous = ensureContiguous(a);
  const aWasCopied = aContiguous !== a;

  // Get dimensions from the logical shapes
  const aRank = aContiguous.shape.length;
  const bRank = b.shape.length;

  if (bRank !== 2) {
    if (aWasCopied) aContiguous.destroy?.();
    throw new Error("Chunked matmul currently only supports 2D B matrix");
  }

  const M = aContiguous.shape[aRank - 2];
  const K_a = aContiguous.shape[aRank - 1];
  const [K_b, N] = b.shape;

  if (K_a !== K_b) {
    if (aWasCopied) aContiguous.destroy?.();
    throw new Error(`Matmul dimension mismatch: A[...,${K_a}] vs B[${K_b},${N}]`);
  }
  const K = K_a;

  // Compute batch dimensions from A
  const batchDims = aContiguous.shape.slice(0, -2);
  const batchSize = batchDims.reduce((acc, d) => acc * d, 1) || 1;

  // Output shape: [...batchDims, M, N]
  const outShape = [...batchDims, M, N];

  // Check if B is a transposed view of a contiguous buffer
  // Transposed [K, N] from original [N, K] has strides [1, K]
  const isTransposedView = b.strides[0] === 1 && b.strides[1] === K;

  if (isTransposedView) {
    const result = matmulChunkedTransposed(
      aContiguous,
      b,
      M,
      K,
      N,
      batchSize,
      batchDims,
      outShape,
      maxBindingSize,
      minAlignment
    );
    if (aWasCopied) aContiguous.destroy?.();
    return result;
  }

  // B is contiguous or has a different stride pattern
  // For contiguous B [K, N], chunk by rows (each row is N elements)
  if (!b.isContiguous) {
    if (aWasCopied) aContiguous.destroy?.();
    throw new Error("Chunked matmul for non-contiguous non-transposed B not yet implemented");
  }

  const result = matmulChunkedContiguous(
    aContiguous,
    b,
    M,
    K,
    N,
    batchSize,
    batchDims,
    outShape,
    maxBindingSize,
    minAlignment
  );
  if (aWasCopied) aContiguous.destroy?.();
  return result;
}

/**
 * Chunked matmul for transposed B.
 * B is a view [K, N] of underlying buffer [N, K].
 * We chunk the buffer by rows (= logical columns) and use transB=true.
 */
function matmulChunkedTransposed(
  a: WebGPUTensor,
  b: WebGPUTensor,
  M: number,
  K: number,
  N: number,
  batchSize: number,
  batchDims: number[],
  outShape: number[],
  maxBindingSize: number,
  minAlignment: number,
): WebGPUTensor {
  const ctx = requireContext();

  // The underlying buffer is [N, K] = N rows of K elements each
  // Each buffer row corresponds to one logical column of B
  const bytesPerBufferRow = K * 4;

  // How many buffer rows fit in one binding?
  const maxBufferRowsPerChunk = Math.floor(maxBindingSize / bytesPerBufferRow);

  // Ensure chunk boundaries are alignment-friendly
  const g_ = gcd(bytesPerBufferRow, minAlignment);
  const rowAlignment = minAlignment / g_;
  const alignedRowsPerChunk = Math.max(
    rowAlignment,
    Math.floor(maxBufferRowsPerChunk / rowAlignment) * rowAlignment
  );

  const numChunks = Math.ceil(N / alignedRowsPerChunk);

  // We'll process each chunk and accumulate partial outputs
  // Output is [...batchDims, M, N] and each chunk contributes columns [colStart, colEnd)
  const partialOutputs: { tensor: WebGPUTensor; colStart: number; colEnd: number }[] = [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const bufferRowStart = chunk * alignedRowsPerChunk;
    const bufferRowEnd = Math.min(bufferRowStart + alignedRowsPerChunk, N);
    const numBufferRows = bufferRowEnd - bufferRowStart;

    // These buffer rows correspond to logical columns [bufferRowStart, bufferRowEnd)
    const colStart = bufferRowStart;
    const colEnd = bufferRowEnd;
    const chunkWidth = numBufferRows;

    // Create a view of the buffer chunk: [numBufferRows, K]
    const byteOffset = (b.offset + bufferRowStart * K) * 4;
    const chunkByteSize = numBufferRows * K * 4;

    // Create a temporary tensor that wraps just this chunk of the buffer
    // This is [chunkWidth, K] in buffer layout
    const bChunkBuffer = createTrackedBuffer(ctx.device, {
      size: chunkByteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Copy the chunk from the original buffer
    if (getSharedEncoderInstance()) {
      getSharedEncoderInstance()!.copyBufferToBuffer(b.buffer, byteOffset, bChunkBuffer, 0, chunkByteSize);
    } else {
      const encoder = ctx.device.createCommandEncoder();
      encoder.copyBufferToBuffer(b.buffer, byteOffset, bChunkBuffer, 0, chunkByteSize);
      submitOrCollect(encoder.finish());
    }

    // Create tensor for the chunk: [chunkWidth, K]
    const bChunk = createTensor([chunkWidth, K], bChunkBuffer, undefined, 0, "f32", true);

    // Matmul: A [M, K] @ bChunk.T [K, chunkWidth] = [M, chunkWidth]
    // Use transB=true since bChunk is [chunkWidth, K] and we want [K, chunkWidth]
    const partialResult = dispatchMatmul(a, bChunk, false, true);

    // Store the partial result tensor (owns the buffer)
    partialOutputs.push({
      tensor: partialResult,
      colStart,
      colEnd,
    });

    // Destroy bChunk - its buffer data has been consumed by the matmul dispatch
    bChunk.destroy?.();
  }

  // Now assemble the final output from partial results
  // Each partial is [batchSize * M, chunkWidth] and goes to columns [colStart, colEnd)
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  for (const partial of partialOutputs) {
    const { tensor: partialTensor, colStart, colEnd } = partial;
    const chunkWidth = colEnd - colStart;

    // Scatter columns from partial [batchSize*M, chunkWidth] to output [batchSize*M, N]
    scatterColumnsToOutput(
      partialTensor,
      outBuffer,
      batchSize * M,
      N,
      colStart
    );

    // Destroy the partial result buffer (deferred destruction waits for GPU fence)
    partialTensor.destroy?.();
  }

  return createTensor(outShape, outBuffer, undefined, 0, "f32", true);
}

/**
 * Chunked matmul for contiguous B [K, N].
 * Chunks B by columns (which requires copying column slices).
 */
function matmulChunkedContiguous(
  a: WebGPUTensor,
  b: WebGPUTensor,
  M: number,
  K: number,
  N: number,
  batchSize: number,
  batchDims: number[],
  outShape: number[],
  maxBindingSize: number,
  minAlignment: number,
): WebGPUTensor {
  const ctx = requireContext();

  // For contiguous B [K, N], columns are NOT contiguous
  // We need to extract column slices to separate buffers

  // Each column is K elements = K * 4 bytes
  const bytesPerColumn = K * 4;
  const maxColumnsPerChunk = Math.floor(maxBindingSize / bytesPerColumn);

  // Ensure alignment
  const g_ = gcd(bytesPerColumn, minAlignment);
  const colAlignment = minAlignment / g_;
  const alignedColumnsPerChunk = Math.max(
    colAlignment,
    Math.floor(maxColumnsPerChunk / colAlignment) * colAlignment
  );

  const numChunks = Math.ceil(N / alignedColumnsPerChunk);

  const partialOutputs: { tensor: WebGPUTensor; colStart: number; colEnd: number }[] = [];

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const colStart = chunk * alignedColumnsPerChunk;
    const colEnd = Math.min(colStart + alignedColumnsPerChunk, N);
    const chunkWidth = colEnd - colStart;

    // Extract column slice using sliceColumns
    const bSlice = sliceColumns(b, colStart, colEnd);

    // Matmul: A [M, K] @ bSlice [K, chunkWidth] = [M, chunkWidth]
    const partialResult = dispatchMatmul(a, bSlice, false, false);

    partialOutputs.push({
      tensor: partialResult,
      colStart,
      colEnd,
    });

    // Destroy bSlice after matmul dispatch - destroy() uses deferred destruction
    // which waits for GPU fence before actually freeing the buffer
    bSlice.destroy?.();
  }

  // Assemble final output
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  for (const partial of partialOutputs) {
    const { tensor: partialTensor, colStart, colEnd } = partial;
    const chunkWidth = colEnd - colStart;

    scatterColumnsToOutput(
      partialTensor,
      outBuffer,
      batchSize * M,
      N,
      colStart
    );

    // Destroy the partial result buffer (deferred destruction waits for GPU fence)
    partialTensor.destroy?.();
  }

  return createTensor(outShape, outBuffer, undefined, 0, "f32", true);
}

/**
 * Chunked matmul when the OUTPUT exceeds the buffer binding limit.
 * This happens when A and B are small but their product is large.
 * Strategy: chunk along the N (columns) dimension of B and output.
 */
function matmulChunkedOutput(
  a: WebGPUTensor,
  b: WebGPUTensor,
  maxBindingSize: number,
): WebGPUTensor {
  const ctx = requireContext();
  const limits = ctx.device.limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Get shapes
  const aShape = a.shape;
  const bShape = b.shape;
  const M = aShape[aShape.length - 2];
  const K = aShape[aShape.length - 1];
  const N = bShape[bShape.length - 1];

  // Compute batch dimensions
  const aBatch = aShape.slice(0, -2);
  const bBatch = bShape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch.length > 0 ? aBatch : [1], bBatch.length > 0 ? bBatch : [1]);
  const batchSize = batchShape.reduce((acc, d) => acc * d, 1);
  const outShape = [...batchShape, M, N];

  // Calculate how many columns we can output per chunk
  // Each output column is batchSize * M elements
  const elementsPerColumn = batchSize * M;
  const bytesPerColumn = elementsPerColumn * 4;
  const maxColumnsPerChunk = Math.floor(maxBindingSize / bytesPerColumn);

  // Ensure alignment for B slicing
  const bBytesPerColumn = K * 4;
  const g_ = gcd(bBytesPerColumn, minAlignment);
  const colAlignment = Math.max(1, minAlignment / g_);
  const alignedColumnsPerChunk = Math.max(
    colAlignment,
    Math.floor(maxColumnsPerChunk / colAlignment) * colAlignment
  );

  const numChunks = Math.ceil(N / alignedColumnsPerChunk);

  // Create output buffer
  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Reshape A to [batchSize, M, K] for consistent processing
  const aReshaped = a.shape.length === 2
    ? createTensor([1, M, K], a.buffer, a.strides ? [0, ...a.strides] : undefined, a.offset, "f32", false)
    : a;

  // Reshape B to [batchSize, K, N] for consistent processing
  const bReshaped = b.shape.length === 2
    ? createTensor([1, K, N], b.buffer, b.strides ? [0, ...b.strides] : undefined, b.offset, "f32", false)
    : b;

  // Process each column chunk
  for (let chunk = 0; chunk < numChunks; chunk++) {
    const colStart = chunk * alignedColumnsPerChunk;
    const colEnd = Math.min(colStart + alignedColumnsPerChunk, N);
    const chunkWidth = colEnd - colStart;

    // Slice B to get columns [colStart:colEnd] -> shape [batchSize, K, chunkWidth]
    // For contiguous B [batchSize, K, N], columns aren't contiguous, so we need to copy
    const bSlice = sliceBColumns(bReshaped, colStart, colEnd);

    // Compute partial matmul: [batchSize, M, K] @ [batchSize, K, chunkWidth] = [batchSize, M, chunkWidth]
    const partialResult = dispatchMatmul(aReshaped, bSlice, false, false);

    // Copy partial result to output buffer at the right column offset
    scatterColumnsToOutput(
      partialResult,
      outBuffer,
      batchSize * M,
      N,
      colStart
    );

    // Destroy temporary buffers after scattering (deferred destruction waits for GPU fence)
    bSlice.destroy?.();
    partialResult.destroy?.();
  }

  return createTensor(outShape, outBuffer, undefined, 0, "f32", true);
}

/**
 * Slice columns from B matrix for chunked output matmul.
 * Input shape: [batch, K, N], Output shape: [batch, K, colEnd - colStart]
 */
function sliceBColumns(
  b: WebGPUTensor,
  colStart: number,
  colEnd: number,
): WebGPUTensor {
  const ctx = requireContext();
  const [batch, K, N] = b.shape;
  const chunkWidth = colEnd - colStart;
  const outSize = batch * K * chunkWidth;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Shader to copy column slice
  const sliceTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const sliceDispatch = compute2DDispatch(sliceTotalWG);
  const sliceUse2D = sliceDispatch.y > 1;
  const sliceGridSizeX = sliceDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  batch: u32,
  K: u32,
  N: u32,
  colStart: u32,
  chunkWidth: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${sliceUse2D ? `gid.x + gid.y * ${sliceGridSizeX}u` : "gid.x"};
  let totalSize = params.batch * params.K * params.chunkWidth;
  if (idx >= totalSize) { return; }

  // Convert flat idx to (b, k, c) in output space
  let c = idx % params.chunkWidth;
  let k = (idx / params.chunkWidth) % params.K;
  let batchIdx = idx / (params.K * params.chunkWidth);

  // Compute input offset: (batchIdx, k, colStart + c)
  let inputOffset = batchIdx * params.K * params.N + k * params.N + params.colStart + c;

  output[idx] = input[inputOffset];
}
`;

  const key = `sliceBColumns:${batch}:${K}:${N}:${WORKGROUP_SIZE}:${sliceUse2D ? "2d" : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  const paramsBuffer = createParamsBuffer(ctx.device, params5(batch, K, N, colStart, chunkWidth));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [b.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, sliceDispatch.x, sliceDispatch.y);

  releaseParamsBuffer(paramsBuffer);

  return createTensor([batch, K, chunkWidth], outBuffer);
}
