/**
 * Reduction ops: sum, max, mean, and supporting helpers.
 *
 * All WGSL shaders are generated via tile-IR (reduction-tile-ir.ts).
 * This file handles shape analysis, contiguity, buffer allocation,
 * and dispatching through the tile-IR pipeline.
 */

import { normalizeDim, type BackendTensor, type DType, type SumOptions, type MeanOptions, type MaxOptions } from "../../types";
import { sizeOf } from "../../../core/shape";
import type { GPUBuffer, WebGPUTensor, WebGPUContext } from "../gpu-types";
import { GPUBufferUsage, asGPUTensor } from "../gpu-types";
import {
  contiguousStrides,
  dtypeBytes,
  alignBufferSize,
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
  params,
} from "../bind-group-cache";
import { isUnaryOp as isUnaryOpFromRegistry } from "./registry";
import { contiguous } from "./views";
import { createTileKernelDispatcher } from "../tile-dispatch";
import {
  makeReductionSpec,
  dimInfo,
  makeMeanDivSpec,
  getChunkedSumWGSL,
  getFinalSumWGSL,
  type ReductionEpilogueOpDesc,
  type PreambleChainKernelOp,
} from "../reduction-tile-ir";

// ============================================================================
// Shared Metadata
// ============================================================================

/** Compute the count of elements being reduced across given dims. */
function reductionCount(inputShape: number[], dim: number | number[] | undefined | null): number {
  if (dim === undefined || dim === null) return sizeOf(inputShape);
  const dims = Array.isArray(dim) ? dim : [dim];
  const rank = inputShape.length;
  return dims.reduce((acc, d) => acc * inputShape[normalizeDim(d, rank)], 1);
}

/**
 * Shared preamble for dim-wise reductions.
 * Normalizes dims, computes outShape, outSize, strides, inputToOutDim.
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

  const inputStrides = contiguousStrides(inputShape);
  const outStrides = contiguousStrides(outShape);
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

  return {
    normalizedDims, rank, outShape, outSize: sizeOf(outShape), reductionSize,
    inputStrides, outStrides, inputToOutDim,
  };
}

// ============================================================================
// Cached dispatchers + helpers
// ============================================================================

const dispatcherCache = new Map<string, ReturnType<typeof createTileKernelDispatcher>>();
function getCachedDispatcher(key: string, specFactory: () => ReturnType<typeof makeReductionSpec>) {
  let d = dispatcherCache.get(key);
  if (!d) { d = createTileKernelDispatcher(specFactory()); dispatcherCache.set(key, d); }
  return d;
}

/** Build `{ in0: buf0, in1: buf1, ... }` map for preamble input buffers. */
function inputBufferMap(inputs: BackendTensor[]): Record<string, GPUBuffer> {
  const m: Record<string, GPUBuffer> = {};
  for (let i = 0; i < inputs.length; i++) m[`in${i}`] = asGPUTensor(inputs[i]).buffer;
  return m;
}

/** Create a 1/count scalar buffer and prepend mul(1/count) to an epilogue chain (for mean). */
function createInvCountEpilogue(
  ctx: WebGPUContext, count: number,
  epilogueOps: ReductionEpilogueOpDesc[], epilogueInputs: BackendTensor[],
): { ops: ReductionEpilogueOpDesc[]; inputs: BackendTensor[]; invCountBuffer: GPUBuffer } {
  const invCountBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  ctx.device.queue.writeBuffer(invCountBuffer, 0, new Float32Array([1.0 / count]));
  return {
    ops: [{ kind: "binary", op: "mul", inputIndex: epilogueInputs.length }, ...epilogueOps],
    inputs: [...epilogueInputs, { buffer: invCountBuffer } as unknown as BackendTensor],
    invCountBuffer,
  };
}

/** Add epilogue binary input buffers to a bindings map. */
function addEpilogueBindings(
  buffers: Record<string, GPUBuffer>,
  epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[],
): void {
  for (const eop of epilogueOps) {
    if (eop.kind === "binary" && eop.inputIndex !== undefined) {
      buffers[`ep_in${eop.inputIndex}`] = asGPUTensor(epilogueInputs[eop.inputIndex]).buffer;
    }
  }
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
  const spec = makeReductionSpec({ reduceOp: "sum",
    dim: dimInfo(inputShape, inputStrides, normalizedDims, outShape, outStrides, inputToOutDim, useParallel),
  });
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

  getCachedDispatcher("sumFull", () => makeReductionSpec({ reduceOp: "sum" })).dispatch(
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

    const paramsBuffer = createParamsBuffer(ctx.device, params(chunkSize, chunk));

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
 * Dimension-wise sum with a single fused elementwise preamble.
 * Delegates to sumDimWithPreambleChain with a single-op chain.
 */
export function sumDimWithPreamble(
  inputs: BackendTensor[],
  preambleOp: string,
  sumOptions: SumOptions,
): BackendTensor {
  const arity = isUnaryOpFromRegistry(preambleOp) ? 1 : 2;
  return sumDimWithPreambleChain(inputs, [{ op: preambleOp, arity }], [], sumOptions);
}

/**
 * Dimension-wise sum with a fused multi-op preamble chain.
 * Applies a chain of elementwise ops (cast → mul → ...) in the accumulation
 * loop body, eliminating intermediate buffers between all preamble ops.
 */
export function sumDimWithPreambleChain(
  inputs: BackendTensor[],
  chainOps: PreambleChainKernelOp[],
  inputDtypes: DType[],
  sumOptions: SumOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensor0 = asGPUTensor(inputs[0]);
  const inputShape = tensor0.shape;
  const dim = sumOptions?.dim;
  const keepdim = sumOptions?.keepdim ?? false;

  const setup = (dim !== undefined && dim !== null)
    ? prepareDimReduction(inputShape, dim, keepdim) : null;

  const inputBuffers = inputs.map(inp => asGPUTensor(inp).buffer);
  const preamble = { chainOps, totalInputs: inputs.length, inputDtypes };

  if (!setup) {
    const outBuffer = resolveOutputBuffer(ctx.device, 4, inputBuffers);
    const spec = makeReductionSpec({ reduceOp: "sum", preamble });
    createTileKernelDispatcher(spec).dispatch(
      { ...inputBufferMap(inputs), out: outBuffer },
      { size: tensor0.size },
    );
    return createTensor([], outBuffer);
  }

  const { normalizedDims, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, inputBuffers);
  const useParallel = reductionSize > 64;
  const spec = makeReductionSpec({ reduceOp: "sum",
    dim: dimInfo(inputShape, inputStrides, normalizedDims, outShape, outStrides, inputToOutDim, useParallel),
    preamble,
  });
  createTileKernelDispatcher(spec).dispatch(
    { ...inputBufferMap(inputs), out: outBuffer },
    { outSize, reductionSize },
  );
  return createTensor(outShape, outBuffer);
}

// ============================================================================
// Sum With Preamble Chain + Epilogue (cross-reduction fusion)
// ============================================================================

/**
 * Sum with both preamble chain and epilogue chain fused into one kernel.
 * Preamble ops are applied in the accumulation loop body,
 * epilogue ops are applied to the reduced result before storing.
 */
export function sumWithPreambleEpilogue(
  preambleInputs: BackendTensor[],
  chainOps: PreambleChainKernelOp[],
  preambleInputDtypes: DType[],
  epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[],
  outputDtype: DType,
  sumOptions: SumOptions,
  isMean?: boolean,
): BackendTensor {
  const ctx = requireContext();
  const tensor0 = asGPUTensor(preambleInputs[0]);
  const inputShape = tensor0.shape;
  const dim = sumOptions?.dim;
  const keepdim = sumOptions?.keepdim ?? false;

  // For mean: prepend mul(1/count) to epilogue chain with a scalar buffer
  let effectiveEpilogueOps = epilogueOps;
  let effectiveEpilogueInputs = epilogueInputs;
  let invCountBuffer: GPUBuffer | null = null;
  if (isMean) {
    const count = reductionCount(inputShape, dim);
    const mean = createInvCountEpilogue(ctx, count, epilogueOps, epilogueInputs);
    effectiveEpilogueOps = mean.ops;
    effectiveEpilogueInputs = mean.inputs;
    invCountBuffer = mean.invCountBuffer;
  }

  const bpe = dtypeBytes(outputDtype);
  const setup = (dim !== undefined && dim !== null)
    ? prepareDimReduction(inputShape, dim, keepdim) : null;
  const preamble = { chainOps, totalInputs: preambleInputs.length, inputDtypes: preambleInputDtypes };
  const epilogue = { ops: effectiveEpilogueOps, outputDtype };
  const allInputBuffers = [
    ...preambleInputs.map(inp => asGPUTensor(inp).buffer),
    ...effectiveEpilogueInputs.map(inp => asGPUTensor(inp).buffer),
  ];
  const buffers: Record<string, GPUBuffer> = { ...inputBufferMap(preambleInputs) };
  addEpilogueBindings(buffers, effectiveEpilogueOps, effectiveEpilogueInputs);

  let result: BackendTensor;
  if (!setup) {
    const outBuffer = resolveOutputBuffer(ctx.device, bpe, allInputBuffers);
    buffers.out = outBuffer;
    createTileKernelDispatcher(makeReductionSpec({ reduceOp: "sum", preamble, epilogue }))
      .dispatch(buffers, { size: tensor0.size });
    result = createTensor([], outBuffer, undefined, 0, outputDtype);
  } else {
    const { normalizedDims, outShape, outSize, reductionSize,
      inputStrides, outStrides, inputToOutDim } = setup;
    const outBuffer = resolveOutputBuffer(ctx.device, outSize * bpe, allInputBuffers);
    buffers.out = outBuffer;
    const useParallel = reductionSize > 64;
    createTileKernelDispatcher(makeReductionSpec({ reduceOp: "sum",
      dim: dimInfo(inputShape, inputStrides, normalizedDims, outShape, outStrides, inputToOutDim, useParallel),
      preamble, epilogue,
    })).dispatch(buffers, { outSize, reductionSize });
    result = createTensor(outShape, outBuffer, undefined, 0, outputDtype);
  }

  if (invCountBuffer) invCountBuffer.destroy();
  return result;
}

// ============================================================================
// Max / Min (unified implementation)
// ============================================================================

type MaxMinOp = "max" | "min";

function maxMinReduction(op: MaxMinOp, a: BackendTensor, options?: MaxOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);
  if (!tensor.isContiguous) tensor = asGPUTensor(contiguous(tensor));

  const inputShape = tensor.shape;
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    return maxMinFullReduction(op, ctx, tensor);
  }

  const setup = prepareDimReduction(inputShape, dim, keepdim);
  if (!setup) return maxMinFullReduction(op, ctx, tensor);

  for (const d of setup.normalizedDims) {
    if (d < 0 || d >= setup.rank) throw new Error(`${op}: dimension ${d} out of range`);
  }

  const { normalizedDims, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  const useParallel = reductionSize > 64;
  const spec = makeReductionSpec({ reduceOp: op,
    dim: dimInfo(inputShape, inputStrides, normalizedDims, outShape, outStrides, inputToOutDim, useParallel),
  });
  const dispatcher = createTileKernelDispatcher(spec);
  dispatcher.dispatch({ input: tensor.buffer, out: outBuffer }, { outSize, reductionSize });

  return createTensor(outShape, outBuffer);
}

function maxMinFullReduction(op: MaxMinOp, ctx: WebGPUContext, tensor: WebGPUTensor): WebGPUTensor {
  const outBuffer = resolveOutputBuffer(ctx.device, 4, [tensor.buffer]);
  getCachedDispatcher(`${op}Full`, () => makeReductionSpec({ reduceOp: op })).dispatch(
    { input: tensor.buffer, out: outBuffer },
    { size: tensor.size },
  );
  return createTensor([], outBuffer);
}

export function max(a: BackendTensor, options?: MaxOptions): BackendTensor {
  return maxMinReduction("max", a, options);
}

export function min(a: BackendTensor, options?: MaxOptions): BackendTensor {
  return maxMinReduction("min", a, options);
}

// ============================================================================
// Mean
// ============================================================================

export function mean(a: BackendTensor, options?: MeanOptions): BackendTensor {
  const tensor = asGPUTensor(a);
  const count = reductionCount(tensor.shape, options?.dim);

  // sum() handles contiguity internally
  const sumTensor = asGPUTensor(sum(a, options));

  // Divide by count via tile-IR
  const ctx = requireContext();
  const outSize = sumTensor.size;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [sumTensor.buffer]);

  getCachedDispatcher("meanDiv", makeMeanDivSpec).dispatch(
    { input: sumTensor.buffer, out: outBuffer },
    { size: outSize, count },
  );

  sumTensor.destroy();
  return createTensor(sumTensor.shape, outBuffer);
}

// ============================================================================
// Sum With Epilogue
// ============================================================================

/** Shared implementation for sumWithEpilogue / maxWithEpilogue. */
function reductionWithEpilogue(
  reduceOp: "sum" | "max",
  a: BackendTensor,
  options: SumOptions | MaxOptions,
  epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[],
  outputDtype: DType,
): BackendTensor {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);
  let contiguousCopy: WebGPUTensor | null = null;

  if (!tensor.isContiguous) {
    tensor = asGPUTensor(contiguous(tensor));
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;
  const bpe = dtypeBytes(outputDtype);

  const setup = (dim !== undefined && dim !== null)
    ? prepareDimReduction(inputShape, dim, keepdim) : null;

  if (!setup) {
    const outBuffer = resolveOutputBuffer(ctx.device, bpe, [tensor.buffer]);
    const spec = makeReductionSpec({ reduceOp, epilogue: { ops: epilogueOps, outputDtype } });
    const dispatcher = createTileKernelDispatcher(spec);
    const buffers: Record<string, GPUBuffer> = { input: tensor.buffer, out: outBuffer };
    addEpilogueBindings(buffers, epilogueOps, epilogueInputs);
    dispatcher.dispatch(buffers, { size: tensor.size });
    if (contiguousCopy) contiguousCopy.destroy();
    return createTensor([], outBuffer, undefined, 0, outputDtype);
  }

  const { normalizedDims, outShape, outSize, reductionSize,
    inputStrides, outStrides, inputToOutDim } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bpe, [tensor.buffer]);

  const useParallel = reductionSize > 64;
  const spec = makeReductionSpec({ reduceOp,
    dim: dimInfo(inputShape, inputStrides, normalizedDims, outShape, outStrides, inputToOutDim, useParallel),
    epilogue: { ops: epilogueOps, outputDtype },
  });
  const dispatcher = createTileKernelDispatcher(spec);

  const buffers: Record<string, GPUBuffer> = { input: tensor.buffer, out: outBuffer };
  addEpilogueBindings(buffers, epilogueOps, epilogueInputs);
  dispatcher.dispatch(buffers, { outSize, reductionSize });

  if (contiguousCopy) contiguousCopy.destroy();
  return createTensor(outShape, outBuffer, undefined, 0, outputDtype);
}

export function sumWithEpilogue(
  a: BackendTensor, options: SumOptions, epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[], outputDtype: DType,
): BackendTensor {
  return reductionWithEpilogue("sum", a, options, epilogueOps, epilogueInputs, outputDtype);
}

export function maxWithEpilogue(
  a: BackendTensor, options: MaxOptions, epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[], outputDtype: DType,
): BackendTensor {
  return reductionWithEpilogue("max", a, options, epilogueOps, epilogueInputs, outputDtype);
}

// ============================================================================
// Mean With Epilogue
// ============================================================================

/**
 * mean = sum / count. Fuses the division into the epilogue chain:
 * sum → mul(1/count) → user epilogue ops, all in one kernel.
 * A scalar buffer with 1/count is created as an extra epilogue binary input.
 */
export function meanWithEpilogue(
  a: BackendTensor,
  options: MeanOptions,
  epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[],
  outputDtype: DType,
): BackendTensor {
  const tensor = asGPUTensor(a);
  const count = reductionCount(tensor.shape, options?.dim);
  const mean = createInvCountEpilogue(requireContext(), count, epilogueOps, epilogueInputs);
  const result = sumWithEpilogue(a, options, mean.ops, mean.inputs, outputDtype);
  mean.invCountBuffer.destroy();
  return result;
}
