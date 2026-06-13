/**
 * Reduction ops: sum, max, mean, and supporting helpers.
 *
 * All WGSL shaders are generated via tile-IR (reduction-tile-ir.ts).
 * This file handles shape analysis, contiguity, buffer allocation,
 * and dispatching through the tile-IR pipeline.
 */

import { sizeOf } from "../../../core/shape";
import {
  type BackendTensor,
  type DType,
  type MaxOptions,
  type MeanOptions,
  normalizeDim,
  type SumOptions,
} from "../../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  profiledCreateBindGroup,
  releaseParamsBuffer,
} from "../bind-group-cache";
import { resolveOutputBuffer } from "../buffer-arena";
import { bufferPool } from "../buffer-pool";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { requireContext } from "../gpu-context";
import type { GPUBuffer, WebGPUContext, WebGPUTensor } from "../gpu-types";
import { asGPUTensor, GPUBufferUsage } from "../gpu-types";
import {
  dimInfo,
  getChunkedSumWGSL,
  getFinalSumWGSL,
  makeMeanDivSpec,
  makeReductionSpec,
  type PreambleChainKernelOp,
  type ReductionEpilogueOpDesc,
} from "../reduction-tile-ir";
import { alignBufferSize, contiguousStrides, dtypeBytes } from "../shape-utils";
import { createTensor, createTrackedBuffer } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";

import { contiguous } from "./views";

// ============================================================================
// Shared Metadata
// ============================================================================

/** Compute the count of elements being reduced across given dims. */
function reductionCount(
  inputShape: number[],
  dim: number | number[] | undefined | null,
): number {
  if (dim == null) return sizeOf(inputShape);
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
      if (keepdim) {
        inputToOutDim.push(outDimIdx);
        outDimIdx++;
      } else inputToOutDim.push(-1);
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

  return {
    normalizedDims,
    rank,
    outShape,
    outSize: sizeOf(outShape),
    reductionSize,
    inputStrides,
    outStrides,
    inputToOutDim,
  };
}

// ============================================================================
// Cached dispatchers + helpers
// ============================================================================

const dispatcherCache = new Map<
  string,
  ReturnType<typeof createTileKernelDispatcher>
>();
function getCachedDispatcher(
  key: string,
  specFactory: () => ReturnType<typeof makeReductionSpec>,
) {
  let d = dispatcherCache.get(key);
  if (!d) {
    d = createTileKernelDispatcher(specFactory());
    dispatcherCache.set(key, d);
  }
  return d;
}

/** Build `{ in0: buf0, in1: buf1, ... }` map for preamble input buffers. */
function inputBufferMap(inputs: BackendTensor[]): Record<string, GPUBuffer> {
  const m: Record<string, GPUBuffer> = {};
  for (let i = 0; i < inputs.length; i++)
    m[`in${i}`] = asGPUTensor(inputs[i]).buffer;
  return m;
}

/** Create a 1/count scalar buffer and prepend mul(1/count) to an epilogue chain (for mean). */
function createInvCountEpilogue(
  ctx: WebGPUContext,
  count: number,
  epilogueOps: ReductionEpilogueOpDesc[],
  epilogueInputs: BackendTensor[],
): {
  ops: ReductionEpilogueOpDesc[];
  inputs: BackendTensor[];
  invCountBuffer: GPUBuffer;
} {
  const invCountBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  ctx.device.queue.writeBuffer(
    invCountBuffer,
    0,
    new Float32Array([1.0 / count]),
  );
  return {
    ops: [
      { kind: "binary", op: "mul", inputIndex: epilogueInputs.length },
      ...epilogueOps,
    ],
    inputs: [
      ...epilogueInputs,
      { buffer: invCountBuffer, shape: [], toArray: () => [] } as BackendTensor,
    ],
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
      buffers[`ep_in${eop.inputIndex}`] = asGPUTensor(
        epilogueInputs[eop.inputIndex],
      ).buffer;
    }
  }
}

// ============================================================================
// Unified reduction dispatch (sum, max, min) with optional epilogue
// ============================================================================

type ReduceOp = "sum" | "max" | "min";

export function reduction(
  op: ReduceOp,
  a: BackendTensor,
  options?: SumOptions | MaxOptions,
  epilogueOps?: ReductionEpilogueOpDesc[],
  epilogueInputs?: BackendTensor[],
  outputDtype?: DType,
): BackendTensor {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);
  let contiguousCopy: ReturnType<typeof asGPUTensor> | null = null;

  if (!tensor.isContiguous) {
    tensor = asGPUTensor(contiguous(tensor));
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;
  const hasEpilogue = epilogueOps != null && epilogueOps.length > 0;
  const bpe = outputDtype ? dtypeBytes(outputDtype) : 4;
  const epilogue = hasEpilogue
    ? { ops: epilogueOps!, outputDtype: outputDtype! }
    : undefined;

  const setup =
    dim != null ? prepareDimReduction(inputShape, dim, keepdim) : null;

  // Full reduction (no dim or all dims reduced)
  if (!setup) {
    if (!hasEpilogue) {
      // Non-epilogue uses cached/chunked fullReduction path
      const result = fullReduction(op, ctx, tensor);
      if (contiguousCopy) contiguousCopy.destroy();
      return result;
    }
    const outBuffer = resolveOutputBuffer(ctx.device, bpe, [tensor.buffer]);
    const buffers: Record<string, GPUBuffer> = {
      input: tensor.buffer,
      out: outBuffer,
    };
    addEpilogueBindings(buffers, epilogueOps!, epilogueInputs!);
    createTileKernelDispatcher(
      makeReductionSpec({ reduceOp: op, epilogue }),
    ).dispatch(buffers, { size: tensor.size });
    if (contiguousCopy) contiguousCopy.destroy();
    return createTensor([], outBuffer, undefined, 0, outputDtype);
  }

  // Dim reduction
  for (const d of setup.normalizedDims) {
    if (d < 0 || d >= setup.rank) {
      throw new Error(`${op}: dimension ${d} out of range`);
    }
  }

  const { normalizedDims, outShape, outSize, reductionSize } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bpe, [
    tensor.buffer,
  ]);
  const buffers: Record<string, GPUBuffer> = {
    input: tensor.buffer,
    out: outBuffer,
  };
  if (hasEpilogue) addEpilogueBindings(buffers, epilogueOps!, epilogueInputs!);
  // Cached dispatcher (shared config buffer → stable identity for stream
  // generation, and no per-call config-buffer allocation). The spec is fully
  // determined by (op, inputShape, dims, keepdim, epilogue) because the input
  // was forced contiguous above, so strides/outShape/useParallel are derived.
  getDimReductionDispatcher(
    op,
    inputShape,
    setup,
    keepdim,
    epilogueOps,
  ).dispatch(buffers, { outSize, reductionSize });

  if (contiguousCopy) contiguousCopy.destroy();
  return createTensor(outShape, outBuffer, undefined, 0, outputDtype);
}

/** Build the dim-reduction tile spec from a prepared setup (single source for
 *  the dispatch path and planDimReductionDispatch). */
function buildDimReductionSpec(
  op: ReduceOp,
  inputShape: number[],
  setup: DimReductionSetup,
  epilogue: { ops: ReductionEpilogueOpDesc[]; outputDtype: DType } | undefined,
): ReturnType<typeof makeReductionSpec> {
  const useParallel = setup.reductionSize > 64;
  return makeReductionSpec({
    reduceOp: op,
    dim: dimInfo(
      inputShape,
      setup.inputStrides,
      setup.normalizedDims,
      setup.outShape,
      setup.outStrides,
      setup.inputToOutDim,
      useParallel,
    ),
    epilogue,
  });
}

/** A cache key that fully identifies a dim-reduction spec. Input is always
 *  contiguous at the dispatch site, so shape+dims+keepdim derive strides,
 *  outShape, and useParallel; the epilogue op chain (not its buffers) decides
 *  the rest of the WGSL. */
function dimReductionKey(
  op: ReduceOp,
  inputShape: number[],
  normalizedDims: number[],
  keepdim: boolean,
  epilogueOps: ReductionEpilogueOpDesc[] | undefined,
): string {
  const epiSig = epilogueOps
    ? epilogueOps
        .map((e) => `${e.kind}:${(e as { op?: string }).op ?? ""}:${(e as { inputIndex?: number }).inputIndex ?? ""}`)
        .join("|")
    : "";
  return `${op}Dim:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim ? 1 : 0}:${epiSig}`;
}

/** Cached dim-reduction dispatcher. The epilogue chain is part of the cache
 *  key (it changes the WGSL); epilogue BUFFERS are passed per-dispatch and are
 *  not part of identity. */
function getDimReductionDispatcher(
  op: ReduceOp,
  inputShape: number[],
  setup: DimReductionSetup,
  keepdim: boolean,
  epilogueOps: ReductionEpilogueOpDesc[] | undefined,
) {
  const epilogue =
    epilogueOps && epilogueOps.length > 0
      ? { ops: epilogueOps, outputDtype: "f32" as DType }
      : undefined;
  return getCachedDispatcher(
    dimReductionKey(op, inputShape, setup.normalizedDims, keepdim, epilogueOps),
    () => buildDimReductionSpec(op, inputShape, setup, epilogue),
  );
}

/** Stage-4 plan/encode: the dim-reduction dispatch plan for a NON-epilogue
 *  reduction (plain sum/max/min over dims — e.g. bias gradients), derived from
 *  the SAME cached dispatcher reduction() uses. Epilogue reductions (mean over
 *  dim) are excluded: their invCount buffer is created fresh per call and isn't
 *  generatable here. */
export function planDimReductionDispatch(
  op: ReduceOp,
  inputShape: number[],
  dim: number | number[],
  keepdim: boolean,
): import("../tile-dispatch").TileKernelPlan | null {
  const setup = prepareDimReduction(inputShape, dim, keepdim);
  if (!setup) return null; // all dims reduced → caller should use full reduction
  return getDimReductionDispatcher(op, inputShape, setup, keepdim, undefined).plan({
    outSize: setup.outSize,
    reductionSize: setup.reductionSize,
  });
}

// ============================================================================
// Full Reduction (all dims)
// ============================================================================

function fullReduction(
  op: ReduceOp,
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): WebGPUTensor {
  // Only sum needs the chunked path for tensors exceeding maxStorageBufferBindingSize
  if (op === "sum") {
    const bytesPerElement = dtypeBytes(tensor.dtype);
    const limits = ctx.device.limits;
    const maxBindingSize =
      limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    if (
      tensor.buffer.size > maxBindingSize ||
      tensor.size * bytesPerElement > maxBindingSize
    ) {
      return sumFullReductionChunked(ctx, tensor, maxBindingSize);
    }
  }

  const outBuffer = resolveOutputBuffer(ctx.device, 4, [tensor.buffer]);
  getCachedDispatcher(`${op}Full`, () =>
    makeReductionSpec({ reduceOp: op }),
  ).dispatch({ input: tensor.buffer, out: outBuffer }, { size: tensor.size });
  return createTensor([], outBuffer);
}

export function sum(a: BackendTensor, options?: SumOptions): BackendTensor {
  return reduction("sum", a, options);
}

/** Stage-4 plan/encode: the FULL-reduction dispatch plan, derived from the
 *  SAME cached dispatcher instance fullReduction() uses (shared config
 *  cache → shared config buffer identity). */
export function planFullReductionDispatch(
  op: ReduceOp,
  size: number,
): import("../tile-dispatch").TileKernelPlan {
  return getCachedDispatcher(`${op}Full`, () =>
    makeReductionSpec({ reduceOp: op }),
  ).plan({ size });
}

/** Stage-4 plan/encode: the meanDiv (sum→÷count) dispatch plan, from the SAME
 *  cached dispatcher mean() uses. Both uniforms (size, count) are shape-derived
 *  constants, so the config buffer is stable — no volatile repack needed. */
export function planMeanDivDispatch(
  size: number,
  count: number,
): import("../tile-dispatch").TileKernelPlan {
  return getCachedDispatcher("meanDiv", makeMeanDivSpec).plan({ size, count });
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
    Math.floor(maxElementsPerChunk / elementsPerAlignment) *
    elementsPerAlignment;

  const totalElements = tensor.size;
  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  // Create buffer for partial sums (one f32 per chunk)
  const partialsBuffer = createTrackedBuffer(ctx.device, {
    size: alignBufferSize(numChunks * 4),
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
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

    const paramsBuffer = createParamsBuffer(
      ctx.device,
      params(chunkSize, chunk),
    );

    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: {
            buffer: tensor.buffer,
            offset: chunkByteOffset,
            size: chunkByteSize,
          },
        },
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
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  const finalWGSL = getFinalSumWGSL();
  const finalPipeline = getPipeline(ctx, finalWGSL, finalWGSL);
  const finalParamsData = new Uint32Array([numChunks, 0, 0, 0]); // 16-byte aligned
  const finalParamsBuffer = createParamsBuffer(ctx.device, finalParamsData);

  const finalBindGroup = cachedCreateBindGroup(ctx.device, finalPipeline, [
    partialsBuffer,
    outBuffer,
    finalParamsBuffer,
  ]);

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

  const setup =
    dim != null ? prepareDimReduction(inputShape, dim, keepdim) : null;

  const inputBuffers = inputs.map((inp) => asGPUTensor(inp).buffer);
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

  const {
    normalizedDims,
    outShape,
    outSize,
    reductionSize,
    inputStrides,
    outStrides,
    inputToOutDim,
  } = setup;
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, inputBuffers);
  const useParallel = reductionSize > 64;
  const spec = makeReductionSpec({
    reduceOp: "sum",
    dim: dimInfo(
      inputShape,
      inputStrides,
      normalizedDims,
      outShape,
      outStrides,
      inputToOutDim,
      useParallel,
    ),
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
    const mean = createInvCountEpilogue(
      ctx,
      count,
      epilogueOps,
      epilogueInputs,
    );
    effectiveEpilogueOps = mean.ops;
    effectiveEpilogueInputs = mean.inputs;
    invCountBuffer = mean.invCountBuffer;
  }

  const bpe = dtypeBytes(outputDtype);
  const setup =
    dim != null ? prepareDimReduction(inputShape, dim, keepdim) : null;
  const preamble = {
    chainOps,
    totalInputs: preambleInputs.length,
    inputDtypes: preambleInputDtypes,
  };
  const epilogue = { ops: effectiveEpilogueOps, outputDtype };
  const allInputBuffers = [
    ...preambleInputs.map((inp) => asGPUTensor(inp).buffer),
    ...effectiveEpilogueInputs.map((inp) => asGPUTensor(inp).buffer),
  ];
  const buffers: Record<string, GPUBuffer> = {
    ...inputBufferMap(preambleInputs),
  };
  addEpilogueBindings(buffers, effectiveEpilogueOps, effectiveEpilogueInputs);

  let result: BackendTensor;
  if (!setup) {
    const outBuffer = resolveOutputBuffer(ctx.device, bpe, allInputBuffers);
    buffers.out = outBuffer;
    createTileKernelDispatcher(
      makeReductionSpec({ reduceOp: "sum", preamble, epilogue }),
    ).dispatch(buffers, { size: tensor0.size });
    result = createTensor([], outBuffer, undefined, 0, outputDtype);
  } else {
    const {
      normalizedDims,
      outShape,
      outSize,
      reductionSize,
      inputStrides,
      outStrides,
      inputToOutDim,
    } = setup;
    const outBuffer = resolveOutputBuffer(
      ctx.device,
      outSize * bpe,
      allInputBuffers,
    );
    buffers.out = outBuffer;
    const useParallel = reductionSize > 64;
    createTileKernelDispatcher(
      makeReductionSpec({
        reduceOp: "sum",
        dim: dimInfo(
          inputShape,
          inputStrides,
          normalizedDims,
          outShape,
          outStrides,
          inputToOutDim,
          useParallel,
        ),
        preamble,
        epilogue,
      }),
    ).dispatch(buffers, { outSize, reductionSize });
    result = createTensor(outShape, outBuffer, undefined, 0, outputDtype);
  }

  if (invCountBuffer) releaseParamsBuffer(invCountBuffer);
  return result;
}

export function max(a: BackendTensor, options?: MaxOptions): BackendTensor {
  return reduction("max", a, options);
}

export function min(a: BackendTensor, options?: MaxOptions): BackendTensor {
  return reduction("min", a, options);
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
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [
    sumTensor.buffer,
  ]);

  getCachedDispatcher("meanDiv", makeMeanDivSpec).dispatch(
    { input: sumTensor.buffer, out: outBuffer },
    { size: outSize, count },
  );

  sumTensor.destroy();
  return createTensor(sumTensor.shape, outBuffer);
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
  const mean = createInvCountEpilogue(
    requireContext(),
    count,
    epilogueOps,
    epilogueInputs,
  );
  const result = reduction(
    "sum",
    a,
    options,
    mean.ops,
    mean.inputs,
    outputDtype,
  );
  releaseParamsBuffer(mean.invCountBuffer);
  return result;
}

// ============================================================================
// Batched Reduction
// ============================================================================

/**
 * Dispatch N independent same-shape reductions in a single kernel.
 *
 * All inputs must have the same shape and be contiguous. All reductions
 * use the same dim and reduce op. Returns N output BackendTensors.
 *
 * The batch size is clamped by the device's maxStorageBuffersPerShaderStage:
 * 2*N (inputs + outputs) + 1 (uniforms) ≤ limit. Excess items are split
 * into multiple dispatches automatically.
 *
 * @param op - Reduction operation ("sum", "max", "min")
 * @param inputs - N input tensors (same shape, contiguous)
 * @param dim - Reduction dimension(s)
 * @param keepdim - Whether to keep reduced dimensions
 * @returns N output tensors
 */
export function batchedReduction(
  op: ReduceOp,
  inputs: BackendTensor[],
  dim: number | number[],
  keepdim = false,
): BackendTensor[] {
  if (inputs.length === 0) return [];
  if (inputs.length === 1) return [reduction(op, inputs[0], { dim, keepdim })];

  const ctx = requireContext();
  const tensors = inputs.map((t) => {
    const gpu = asGPUTensor(t);
    return gpu.isContiguous ? gpu : asGPUTensor(contiguous(gpu));
  });

  const inputShape = tensors[0].shape;
  const setup = prepareDimReduction(inputShape, dim, keepdim);
  if (!setup) {
    // Full reduction — fall back to individual dispatches
    return inputs.map((t) => reduction(op, t, { dim, keepdim }));
  }

  const {
    normalizedDims,
    outShape,
    outSize,
    reductionSize,
    inputStrides,
    outStrides,
    inputToOutDim,
  } = setup;

  // Batched path only supports dim-sequential (small reduction size).
  // Parallel reductions (large reduction size) fall back to individual.
  if (reductionSize > 64) {
    return inputs.map((t) => reduction(op, t, { dim, keepdim }));
  }

  // Device binding limit: 2*N + 1 (uniform) ≤ maxStorageBuffersPerShaderStage
  const maxStorage = ctx.device.limits?.maxStorageBuffersPerShaderStage ?? 8;
  const maxBatch = Math.max(1, Math.floor((maxStorage - 1) / 2));

  const results: BackendTensor[] = [];

  for (let start = 0; start < tensors.length; start += maxBatch) {
    const batch = tensors.slice(
      start,
      Math.min(start + maxBatch, tensors.length),
    );
    const count = batch.length;

    if (count === 1) {
      results.push(reduction(op, inputs[start], { dim, keepdim }));
      continue;
    }

    // Build batched spec via unified factory
    const spec = makeReductionSpec({
      reduceOp: op,
      dim: dimInfo(
        inputShape,
        inputStrides,
        normalizedDims,
        outShape,
        outStrides,
        inputToOutDim,
        false, // sequential
      ),
      count,
    });

    // Allocate output buffers + build binding map
    const inputBuffers = batch.map((t) => t.buffer);
    const buffers: Record<string, GPUBuffer> = {};
    for (let i = 0; i < count; i++) {
      buffers[`in${i}`] = batch[i].buffer;
    }
    for (let i = 0; i < count; i++) {
      buffers[`out${i}`] = resolveOutputBuffer(
        ctx.device,
        outSize * 4,
        inputBuffers,
      );
    }

    createTileKernelDispatcher(spec).dispatch(buffers, {
      outSize,
      reductionSize,
    });

    for (let i = 0; i < count; i++) {
      results.push(createTensor(outShape, buffers[`out${i}`]));
    }
  }

  return results;
}
