/**
 * Reduction ops: sum, max, mean, and supporting helpers.
 *
 * All WGSL shaders are generated via tile-IR (reduction-tile-ir.ts).
 * This file handles shape analysis, contiguity, buffer allocation,
 * and dispatching through the tile-IR pipeline.
 */

import { normalizeDim, type BackendTensor, type SumOptions, type MeanOptions, type MaxOptions } from "../../types";
import { sizeOf } from "../../../core/shape";
import type { GPUBuffer, WebGPUTensor, WebGPUContext } from "../gpu-types";
import { GPUBufferUsage, asGPUTensor } from "../gpu-types";
import {
  dtypeBytes,
  alignBufferSize,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import { bufferPool } from "../buffer-pool";
import { resolveOutputBuffer } from "../buffer-arena";
import {
  cachedCreateBindGroup,
  profiledCreateBindGroup,
  createParamsBuffer,
  releaseParamsBuffer,
  params2,
} from "../bind-group-cache";
import { isUnaryOp as isUnaryOpFromRegistry } from "./registry";
import { contiguous } from "./views";
import { createTileKernelDispatcher } from "../tile-dispatch";
import {
  makeSumDimSpec,
  makeSumFullSpec,
  makeMaxDimSpec,
  makeMaxFullSpec,
  makeMeanDivSpec,
  makeSumDimWithPreambleSpec,
  makeSumFullWithPreambleSpec,
  getChunkedSumWGSL,
  getFinalSumWGSL,
} from "../reduction-tile-ir";

// ============================================================================
// Shared Metadata
// ============================================================================

/**
 * Compute strides, reduction size, and input→output dimension mapping
 * shared by sum(), max(), and sumDimWithPreamble().
 */
function buildReductionMetadata(
  inputShape: number[],
  normalizedDims: number[],
  outShape: number[],
  keepdim: boolean,
): {
  inputStrides: number[];
  outStrides: number[];
  reductionSize: number;
  inputToOutDim: number[];
} {
  const rank = inputShape.length;
  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) stride *= inputShape[j];
    inputStrides.push(stride);
  }

  const outStrides: number[] = [];
  for (let i = 0; i < outShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < outShape.length; j++) stride *= outShape[j];
    outStrides.push(stride);
  }

  let reductionSize = 1;
  for (const d of normalizedDims) reductionSize *= inputShape[d];

  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) { inputToOutDim.push(outDimIdx); outDimIdx++; }
      else inputToOutDim.push(-1);
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

  return { inputStrides, outStrides, reductionSize, inputToOutDim };
}

/**
 * Shared preamble for dim-wise reductions.
 * Normalizes dims, computes outShape, outSize, metadata.
 * Returns null if all dims are reduced (caller should use full-reduction path).
 */
interface DimReductionSetup {
  normalizedDims: number[];
  rank: number;
  outShape: number[];
  outSize: number;
  reductionSize: number;
  inputStrides: number[];
  outStrides: number[];
  inputToOutDim: number[];
}

function prepareDimReduction(
  inputShape: number[],
  dim: number | number[],
  keepdim: boolean,
): DimReductionSetup | null {
  const dims = Array.isArray(dim) ? dim : [dim];
  const rank = inputShape.length;
  const normalizedDims = dims.map((d) => normalizeDim(d, rank));

  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) outShape.push(1);
    } else {
      outShape.push(inputShape[i]);
    }
  }

  if (outShape.length === 0) return null;

  const outSize = sizeOf(outShape);
  const { inputStrides, outStrides, reductionSize, inputToOutDim } =
    buildReductionMetadata(inputShape, normalizedDims, outShape, keepdim);

  return {
    normalizedDims, rank, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim,
  };
}

// ============================================================================
// Cached dispatchers for shape-independent kernels
// ============================================================================

let sumFullDispatcher: ReturnType<typeof createTileKernelDispatcher> | null = null;
let maxFullDispatcher: ReturnType<typeof createTileKernelDispatcher> | null = null;
let meanDivDispatcher: ReturnType<typeof createTileKernelDispatcher> | null = null;

function getSumFullDispatcher() {
  if (!sumFullDispatcher) sumFullDispatcher = createTileKernelDispatcher(makeSumFullSpec());
  return sumFullDispatcher;
}

function getMaxFullDispatcher() {
  if (!maxFullDispatcher) maxFullDispatcher = createTileKernelDispatcher(makeMaxFullSpec());
  return maxFullDispatcher;
}

function getMeanDivDispatcher() {
  if (!meanDivDispatcher) meanDivDispatcher = createTileKernelDispatcher(makeMeanDivSpec());
  return meanDivDispatcher;
}

// ============================================================================
// Sum
// ============================================================================

export function sum(a: BackendTensor, options?: SumOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);
  let contiguousCopy: WebGPUTensor | null = null;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  // The sum kernels assume contiguous layout for index computation
  if (!tensor.isContiguous) {
    tensor = asGPUTensor(contiguous(tensor));
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    const result = sumFullReduction(ctx, tensor);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  const setup = prepareDimReduction(inputShape, dim, keepdim);
  if (!setup) {
    const result = sumFullReduction(ctx, tensor);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  for (const d of setup.normalizedDims) {
    if (d < 0 || d >= setup.rank) {
      throw new Error(`sum: dimension ${d} out of range`);
    }
  }

  const { normalizedDims, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  const useParallel = reductionSize > 64;
  const spec = makeSumDimSpec(inputShape, inputStrides, normalizedDims,
    outShape, outStrides, inputToOutDim, useParallel);
  const dispatcher = createTileKernelDispatcher(spec);

  dispatcher.dispatch(
    { input: tensor.buffer, out: outBuffer },
    { outSize, reductionSize },
  );

  if (contiguousCopy) contiguousCopy.destroy();
  return createTensor(outShape, outBuffer);
}

// ============================================================================
// Sum Full Reduction
// ============================================================================

function sumFullReduction(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): WebGPUTensor {
  const inputSize = tensor.size;
  const bytesPerElement = dtypeBytes(tensor.dtype);

  // Check if input buffer exceeds max binding size
  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const inputBufferSize = tensor.buffer.size;

  if (inputBufferSize > maxBindingSize || inputSize * bytesPerElement > maxBindingSize) {
    return sumFullReductionChunked(ctx, tensor, maxBindingSize);
  }

  const outBuffer = resolveOutputBuffer(ctx.device, 4, [tensor.buffer]);

  getSumFullDispatcher().dispatch(
    { input: tensor.buffer, out: outBuffer },
    { size: inputSize },
  );

  return createTensor([], outBuffer);
}

// ============================================================================
// Sum Full Reduction Chunked
// ============================================================================

/**
 * Chunked full reduction sum for tensors exceeding maxStorageBufferBindingSize.
 * Computes partial sums per chunk, then sums the partials.
 *
 * Uses tile-IR generated WGSL but dispatches manually (per-chunk bind groups
 * need buffer offset/size entries which tile-dispatch doesn't support directly).
 */
function sumFullReductionChunked(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
  maxBindingSize: number,
): WebGPUTensor {
  const bytesPerElement = dtypeBytes(tensor.dtype);
  const limits = ctx.device.limits;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;
  const elementsPerAlignment = minAlignment / bytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const totalElements = tensor.size;
  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  // Create buffer for partial sums (one f32 per chunk)
  const partialsBuffer = createTrackedBuffer(ctx.device, {
    size: alignBufferSize(numChunks * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  // Per-chunk kernel: tile-IR generated WGSL, manual bind groups for offset/size
  const chunkWGSL = getChunkedSumWGSL();
  const pipeline = getPipeline(ctx, chunkWGSL, chunkWGSL);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    const paramsBuffer = createParamsBuffer(ctx.device, params2(chunkSize, chunk));

    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tensor.buffer, offset: chunkByteOffset, size: chunkByteSize } },
        { binding: 1, resource: { buffer: partialsBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, 1);
    releaseParamsBuffer(paramsBuffer);
  }

  // Sum the partials
  if (numChunks === 1) {
    return createTensor([], partialsBuffer);
  }

  // Final reduction of partials: tile-IR generated WGSL
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const finalWGSL = getFinalSumWGSL();
  const finalPipeline = getPipeline(ctx, finalWGSL, finalWGSL);
  const finalParamsData = new Uint32Array([numChunks, 0, 0, 0]); // 16-byte aligned
  const finalParamsBuffer = createParamsBuffer(ctx.device, finalParamsData);

  const finalBindGroup = cachedCreateBindGroup(ctx.device, finalPipeline, [partialsBuffer, outBuffer, finalParamsBuffer]);

  dispatchComputePass(finalPipeline, finalBindGroup, 1);
  releaseParamsBuffer(finalParamsBuffer);

  // Destroy the intermediate partials buffer
  bufferPool.deferredDestroy(partialsBuffer, alignBufferSize(numChunks * 4));

  return createTensor([], outBuffer);
}

// ============================================================================
// Sum Dim With Preamble (fused elementwise + reduce)
// ============================================================================

/**
 * Dimension-wise sum with a fused elementwise preamble.
 * Instead of `total += input[offset]`, computes `total += preambleExpr(input0[offset], input1[offset])`.
 * This eliminates the need for a separate elementwise kernel before the reduction.
 */
export function sumDimWithPreamble(
  inputs: BackendTensor[],
  preambleOp: string,
  sumOptions: SumOptions,
): BackendTensor {
  const tensor0 = asGPUTensor(inputs[0]);
  const inputShape = tensor0.shape;

  const dim = sumOptions?.dim;
  const keepdim = sumOptions?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    return sumFullReductionWithPreamble(inputs, preambleOp);
  }

  const setup = prepareDimReduction(inputShape, dim, keepdim);
  if (!setup) {
    return sumFullReductionWithPreamble(inputs, preambleOp);
  }

  const isUnary = isUnaryOpFromRegistry(preambleOp);
  const arity = isUnary ? 1 : 2;

  const { normalizedDims, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim } = setup;

  const outBuffer = createTrackedBuffer(requireContext().device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const spec = makeSumDimWithPreambleSpec(
    preambleOp, arity, inputShape, inputStrides,
    normalizedDims, outShape, outStrides, inputToOutDim,
  );
  const dispatcher = createTileKernelDispatcher(spec);

  const buffers: Record<string, GPUBuffer> = { out: outBuffer };
  for (let i = 0; i < arity; i++) {
    buffers[`in${i}`] = asGPUTensor(inputs[i]).buffer;
  }

  dispatcher.dispatch(buffers, { outSize, reductionSize });

  return createTensor(outShape, outBuffer);
}

/**
 * Full reduction sum with fused elementwise preamble.
 */
function sumFullReductionWithPreamble(
  inputs: BackendTensor[],
  preambleOp: string,
): WebGPUTensor {
  const isUnary = isUnaryOpFromRegistry(preambleOp);
  const arity = isUnary ? 1 : 2;
  const tensor0 = asGPUTensor(inputs[0]);
  const inputSize = tensor0.size;

  const outBuffer = createTrackedBuffer(requireContext().device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const spec = makeSumFullWithPreambleSpec(preambleOp, arity);
  const dispatcher = createTileKernelDispatcher(spec);

  const buffers: Record<string, GPUBuffer> = { out: outBuffer };
  for (let i = 0; i < arity; i++) {
    buffers[`in${i}`] = asGPUTensor(inputs[i]).buffer;
  }

  dispatcher.dispatch(buffers, { size: inputSize });

  return createTensor([], outBuffer);
}

// ============================================================================
// Max
// ============================================================================

export function max(a: BackendTensor, options?: MaxOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);

  if (!tensor.isContiguous) {
    tensor = asGPUTensor(contiguous(tensor));
  }

  const inputShape = tensor.shape;
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    return maxFullReduction(ctx, tensor);
  }

  const setup = prepareDimReduction(inputShape, dim, keepdim);
  if (!setup) {
    return maxFullReduction(ctx, tensor);
  }

  for (const d of setup.normalizedDims) {
    if (d < 0 || d >= setup.rank) {
      throw new Error(`max: dimension ${d} out of range`);
    }
  }

  const { normalizedDims, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  // Use parallel reduction for large reduction dims (new capability from tile-IR!)
  const useParallel = reductionSize > 64;
  const spec = makeMaxDimSpec(inputShape, inputStrides, normalizedDims,
    outShape, outStrides, inputToOutDim, useParallel);
  const dispatcher = createTileKernelDispatcher(spec);

  dispatcher.dispatch(
    { input: tensor.buffer, out: outBuffer },
    { outSize, reductionSize },
  );

  return createTensor(outShape, outBuffer);
}

function maxFullReduction(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): WebGPUTensor {
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  getMaxFullDispatcher().dispatch(
    { input: tensor.buffer, out: outBuffer },
    { size: tensor.size },
  );

  return createTensor([], outBuffer);
}

// ============================================================================
// Mean
// ============================================================================

export function mean(a: BackendTensor, options?: MeanOptions): BackendTensor {
  let tensor = asGPUTensor(a);
  let contiguousCopy: WebGPUTensor | null = null;

  if (!tensor.isContiguous) {
    tensor = asGPUTensor(contiguous(tensor));
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;
  const dim = options?.dim;

  // Compute the count of elements being averaged
  let count: number;
  if (dim === undefined || dim === null) {
    count = tensor.size;
  } else {
    const dims = Array.isArray(dim) ? dim : [dim];
    const rank = inputShape.length;
    count = dims.reduce((acc, d) => acc * inputShape[normalizeDim(d, rank)], 1);
  }

  // Get sum result (always a tensor, possibly 0-d)
  const sumTensor = asGPUTensor(sum(a, options));

  // Divide by count via tile-IR
  const ctx = requireContext();
  const outSize = sumTensor.size;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [sumTensor.buffer]);

  getMeanDivDispatcher().dispatch(
    { input: sumTensor.buffer, out: outBuffer },
    { size: outSize, count },
  );

  // Destroy intermediate sum tensor
  sumTensor.destroy();

  if (contiguousCopy) {
    contiguousCopy.destroy();
  }

  return createTensor(sumTensor.shape, outBuffer);
}
