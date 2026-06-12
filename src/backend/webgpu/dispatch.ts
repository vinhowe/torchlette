/**
 * Core dispatch infrastructure: compute pass execution, pipeline caching,
 * binary/unary/matmul dispatch.
 */

import {
  getAndClearLastBindGroupBuffers,
  isCompilationRecordingActive,
  recordDispatch as recordCompiledDispatch,
} from "../../executor/compiled-plan";
import type { DType } from "../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  releaseParamsBuffer,
} from "./bind-group-cache";
import { resolveOutputBuffer } from "./buffer-arena";
import { destroyCopy } from "./buffer-pool";
import { isF16Supported, requireContext } from "./gpu-context";
import type {
  GPUBindGroup,
  GPUBuffer,
  GPUComputePipeline,
  WebGPUContext,
  WebGPUTensor,
} from "./gpu-types";
import {
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  dispatchTiledMatmul,
} from "./matmul/dispatch";
import type { EpilogueConfig, DType as MatmulDType } from "./matmul/types";
import {
  binaryBroadcastSpec,
  binaryBroadcastTileIR,
  unaryStridedSpec,
  unaryStridedTileIR,
} from "./ops/ops-tile-ir";
import { detectSimpleTranspose, ensureContiguous } from "./ops/views";
import { getWarmupPipeline, recordPipeline } from "./pipeline-warmup";
import { getProfileModule, getTimestampWrites } from "./profiler";
import {
  broadcastShapes,
  compute2DDispatch,
  computeEffectiveBroadcastStrides,
  dtypeBytes,
  sizeOf,
  toIndexShape,
  WORKGROUP_SIZE,
} from "./shape-utils";
import {
  autoFlushSharedEncoder,
  getCurrentOpLabel,
  getSharedEncoderInstance,
  incrementSharedEncoderPassCount,
  submitOrCollect,
} from "./shared-encoder";

// Tensor construction helpers (extracted to tensor.ts)
import { createTensor } from "./tensor";
import { createTileKernelDispatcher } from "./tile-dispatch";

export function dispatchComputePass(
  pipeline: GPUComputePipeline,
  bindGroup: unknown,
  workgroupsX: number,
  workgroupsY: number = 1,
  workgroupsZ: number = 1,
  labelOverride?: string,
): void {
  const ctx = requireContext();
  const label = labelOverride ?? getCurrentOpLabel();

  // Record dispatch for compiled plan (label/module restore profiler
  // attribution during replay — without them all replayed GPU time shows
  // as "unknown").
  if (isCompilationRecordingActive()) {
    recordCompiledDispatch({
      pipeline,
      bindGroup: bindGroup as GPUBindGroup,
      workgroupsX,
      workgroupsY,
      workgroupsZ,
      buffers: getAndClearLastBindGroupBuffers(),
      label: label ?? undefined,
      module: getProfileModule(),
    });
  }

  const sharedEnc = getSharedEncoderInstance();
  const tsWrites = getTimestampWrites(label ?? "unknown");
  const encoder = sharedEnc ?? ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass(
    tsWrites ? { timestampWrites: tsWrites } : undefined,
  );
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup as GPUBindGroup);
  pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
  pass.end();
  if (sharedEnc) {
    incrementSharedEncoderPassCount();
    autoFlushSharedEncoder();
  } else {
    submitOrCollect(encoder.finish());
  }
}

/**
 * Unified dispatch helper for elementwise compute shaders.
 * Collapses the common 6-step boilerplate (pipeline, output, params, bindgroup, dispatch, release)
 * into a single call.
 */
export function dispatchElementwise(desc: {
  key: string;
  shader: string;
  inputs: GPUBuffer[];
  outputSizeBytes: number;
  params: Uint32Array;
  outBuffer?: GPUBuffer;
  dispatchX: number;
  dispatchY?: number;
}): GPUBuffer {
  const ctx = requireContext();

  const pipeline = getPipeline(ctx, desc.key, desc.shader);

  const outBuffer =
    desc.outBuffer ??
    resolveOutputBuffer(ctx.device, desc.outputSizeBytes, desc.inputs);

  const paramsBuffer = createParamsBuffer(ctx.device, desc.params);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    ...desc.inputs,
    outBuffer,
    paramsBuffer,
  ]);

  dispatchComputePass(pipeline, bindGroup, desc.dispatchX, desc.dispatchY ?? 1);

  releaseParamsBuffer(paramsBuffer);

  return outBuffer;
}

export function getPipeline(
  context: WebGPUContext,
  key: string,
  code: string,
): GPUComputePipeline {
  const cached = context.pipelines.get(key);
  if (cached) {
    return cached;
  }
  // Check warmup cache (pre-compiled via createComputePipelineAsync)
  const warmed = getWarmupPipeline(key);
  if (warmed) {
    context.pipelines.set(key, warmed);
    return warmed;
  }
  recordPipeline(key, code);
  const module = context.device.createShaderModule({ code });
  const pipeline = context.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  context.pipelines.set(key, pipeline);
  return pipeline;
}

/**
 * Dispatch binary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
/**
 * Stage-4 plan/encode split: everything the DIRECT binary dispatch decides,
 * computed from tensor METADATA only (no GPU calls, no allocation). Returns
 * null when the dispatcher would take a non-direct path (chunked dispatch
 * or contiguous-copy insertion for oversized strided operands) — callers
 * generating streams treat null as "uncovered". dispatchBinary's direct
 * tail consumes the SAME plan, so the generator and the executor cannot
 * silently disagree.
 */
export interface ElementwiseDirectPlan {
  key: string;
  shader: string;
  outputSizeBytes: number;
  paramsData: Uint32Array;
  dispatchX: number;
  dispatchY: number;
}

export function planBinaryDirect(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
): ElementwiseDirectPlan | null {
  // Generator-facing guard: bail where dispatchBinary takes a NON-direct
  // path (chunked dispatch or contiguous-copy insertion). dispatchBinary's
  // own tail calls the unguarded core — it has already routed those cases.
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize = sizeOf(outShape);
  if (outSize === 0) return null;
  const dtype = a.dtype;
  if (b.dtype !== dtype) return null;
  const bytesPerElement = dtypeBytes(dtype);
  const ctx = requireContext();
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  if (
    a.size * bytesPerElement > maxBindingSize ||
    b.size * bytesPerElement > maxBindingSize ||
    outSize * bytesPerElement > maxBindingSize
  ) {
    return null; // chunked / contiguous-insertion territory
  }
  return planBinaryDirectCore(op, a, b, outShape, outSize, bytesPerElement);
}

/** Unguarded plan core — dispatchBinary's direct tail (post-routing). */
export function planBinaryDirectCore(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  outShape: number[],
  outSize: number,
  bytesPerElement: number,
): ElementwiseDirectPlan {
  const dtype = a.dtype;
  const indexShape = toIndexShape(outShape);
  const aStrides = computeEffectiveBroadcastStrides(a, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(b, indexShape);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const shader = binaryBroadcastTileIR(
    op,
    indexShape,
    aStrides,
    bStrides,
    a.offset,
    b.offset,
    dtype as "f32" | "f16" | "u32" | "i32",
  );
  const key = `binary:${op}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${a.offset}:${b.offset}:${dtype}:${use2D ? dispatch.gridSizeX : "1d"}`;
  return {
    key,
    shader,
    outputSizeBytes: outSize * bytesPerElement,
    paramsData: params(outSize),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  };
}

export function dispatchBinary(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize = sizeOf(outShape);
  if (outSize === 0) {
    throw new Error("webgpu ops do not support empty tensors yet");
  }
  const ctx = requireContext();
  const dtype = a.dtype;
  if (b.dtype !== dtype) {
    throw new Error(
      `webgpu binary op: mismatched dtypes ${a.dtype} and ${b.dtype}`,
    );
  }

  const bytesPerElement = dtypeBytes(dtype);
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const aSizeBytes = a.size * bytesPerElement;
  const bSizeBytes = b.size * bytesPerElement;
  const outSizeBytes = outSize * bytesPerElement;

  // Large tensor chunked dispatch path
  if (
    aSizeBytes > maxBindingSize ||
    bSizeBytes > maxBindingSize ||
    outSizeBytes > maxBindingSize
  ) {
    const aIsScalar = a.size === 1;
    const bIsScalar = b.size === 1;
    const sameShape =
      a.shape.length === b.shape.length &&
      a.shape.every((d, i) => d === b.shape[i]);

    if (
      (sameShape && a.isContiguous && b.isContiguous) ||
      (aIsScalar && b.isContiguous) ||
      (bIsScalar && a.isContiguous)
    ) {
      return binaryChunked(
        op,
        a,
        b,
        aIsScalar,
        bIsScalar,
        outShape,
        outSize,
        dtype,
        bytesPerElement,
        options,
      );
    }

    if (sameShape) {
      const aC = a.isContiguous ? a : ensureContiguous(a);
      const bC = b.isContiguous ? b : ensureContiguous(b);
      const result = binaryChunked(
        op,
        aC,
        bC,
        aC.size === 1,
        bC.size === 1,
        outShape,
        outSize,
        dtype,
        bytesPerElement,
        options,
      );
      if (aC !== a) aC.destroy?.();
      if (bC !== b) bC.destroy?.();
      return result;
    }

    if (!a.isContiguous && aSizeBytes > maxBindingSize) {
      const aC = ensureContiguous(a);
      const result = dispatchBinary(op, aC, b, options);
      aC.destroy?.();
      return result;
    }
    if (!b.isContiguous && bSizeBytes > maxBindingSize) {
      const bC = ensureContiguous(b);
      const result = dispatchBinary(op, a, bC, options);
      bC.destroy?.();
      return result;
    }
    // Fall through to direct dispatch
  }

  // Direct dispatch — decisions single-sourced with the stage-4 stream
  // generator (planBinaryDirect guards + this core; stream-generate.ts).
  const plan = planBinaryDirectCore(
    op,
    a,
    b,
    outShape,
    outSize,
    bytesPerElement,
  );

  const outBuffer = dispatchElementwise({
    key: plan.key,
    shader: plan.shader,
    inputs: [a.buffer, b.buffer],
    outputSizeBytes: plan.outputSizeBytes,
    params: plan.paramsData,
    outBuffer: options?.outBuffer,
    dispatchX: plan.dispatchX,
    dispatchY: plan.dispatchY,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/** Chunked binary dispatch for large contiguous tensors exceeding maxStorageBufferBindingSize. */
function binaryChunked(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  aIsScalar: boolean,
  bIsScalar: boolean,
  outShape: number[],
  outSize: number,
  dtype: DType,
  bytesPerElement: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [a.buffer, b.buffer],
    options?.outBuffer,
  );
  const spec = binaryBroadcastSpec(
    op,
    [outSize],
    aIsScalar ? [0] : [1],
    bIsScalar ? [0] : [1],
    0,
    0,
    dtype as "f32" | "f16" | "u32" | "i32",
  );
  const dispatcher = createTileKernelDispatcher(spec);
  dispatcher.dispatchChunked(
    { a: a.buffer, b: b.buffer, out: outBuffer },
    { size: outSize },
    {
      modes: {
        a: aIsScalar ? "scalar" : "chunked",
        b: bIsScalar ? "scalar" : "chunked",
        out: "chunked",
      },
      sizeUniform: "size",
      totalElements: outSize,
      maxBytesPerElement: bytesPerElement,
    },
  );
  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Dispatch unary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
export function dispatchUnary(
  opKey: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Chunked dispatch for large contiguous tensors
  if (a.size * bytesPerElement > maxBindingSize && a.isContiguous) {
    const outSize = a.size;
    const outBuffer = resolveOutputBuffer(
      ctx.device,
      outSize * bytesPerElement,
      [a.buffer],
      options?.outBuffer,
    );
    const spec = unaryStridedSpec(
      opKey,
      [outSize],
      [1],
      0,
      dtype as "f32" | "f16" | "u32" | "i32",
    );
    const dispatcher = createTileKernelDispatcher(spec);
    dispatcher.dispatchChunked(
      { a: a.buffer, out: outBuffer },
      { size: outSize },
      {
        modes: { a: "chunked", out: "chunked" },
        sizeUniform: "size",
        totalElements: outSize,
        maxBytesPerElement: bytesPerElement,
      },
    );
    const ownsBuffer = options?.outBuffer === undefined;
    return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
  }

  // Direct dispatch
  const totalWorkgroups = Math.ceil(a.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const code = unaryStridedTileIR(
    opKey,
    a.shape,
    a.strides,
    a.offset,
    dtype as "f32" | "f16" | "u32" | "i32",
  );
  const key = `unary:${opKey}:${a.shape.join("x")}:${a.strides.join(",")}:${a.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;

  const outBuffer = dispatchElementwise({
    key,
    shader: code,
    inputs: [a.buffer],
    outputSizeBytes: a.size * bytesPerElement,
    params: params(a.size),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Resolve a matmul input: detect simple transpose to avoid materialization,
 * or ensure contiguity.  Returns the effective tensor, transpose flag, and
 * whether a contiguous copy was allocated.
 */
function resolveMatmulInput(t: WebGPUTensor, trans: boolean) {
  const origShape = !trans ? detectSimpleTranspose(t) : null;
  if (origShape) {
    return {
      tensor: createTensor(origShape, t.buffer, undefined, 0, t.dtype, false),
      trans: true,
      wasCopied: false,
    };
  }
  const contig = ensureContiguous(t);
  return { tensor: contig, trans, wasCopied: contig !== t };
}

/**
 * Prepare matmul inputs: detect transposes, ensure contiguity,
 * compute output shape, extract M/K/N, and compute batch strides.
 */
function prepareMatmulInputs(
  a: WebGPUTensor,
  b: WebGPUTensor,
  transA: boolean,
  transB: boolean,
) {
  const rA = resolveMatmulInput(a, transA);
  const rB = resolveMatmulInput(b, transB);
  const effectiveA = rA.tensor,
    effectiveTransA = rA.trans,
    aWasCopied = rA.wasCopied;
  const effectiveB = rB.tensor,
    effectiveTransB = rB.trans,
    bWasCopied = rB.wasCopied;

  // Compute output shape with transpose and batch broadcasting
  const outShape = computeMatmulOutputShape(
    effectiveA.shape,
    effectiveB.shape,
    effectiveTransA,
    effectiveTransB,
  );

  // Extract matrix dimensions
  const aRank = effectiveA.shape.length;
  const bRank = effectiveB.shape.length;

  let m: number, k: number, n: number;
  if (effectiveTransA) {
    k = effectiveA.shape[aRank - 2];
    m = effectiveA.shape[aRank - 1];
  } else {
    m = effectiveA.shape[aRank - 2];
    k = effectiveA.shape[aRank - 1];
  }
  if (effectiveTransB) {
    n = effectiveB.shape[bRank - 2];
  } else {
    n = effectiveB.shape[bRank - 1];
  }

  // Compute batch size and strides
  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    effectiveA.shape,
    effectiveB.shape,
    batchDims,
    m,
    n,
    k,
  );

  return {
    effectiveA,
    effectiveB,
    effectiveTransA,
    effectiveTransB,
    aWasCopied,
    bWasCopied,
    outShape,
    m,
    k,
    n,
    batchDims,
    batchSize,
    strideA,
    strideB,
    strideC,
  };
}

export function dispatchMatmul(
  a: WebGPUTensor,
  b: WebGPUTensor,
  transA = false,
  transB = false,
  donatedBuffer?: GPUBuffer,
  epilogueOpts?: {
    epilogue: EpilogueConfig;
    epilogueInputs: WebGPUTensor[];
    inputCastA?: MatmulDType;
    inputCastB?: MatmulDType;
  },
): WebGPUTensor {
  const ctx = requireContext();

  const {
    effectiveA,
    effectiveB,
    effectiveTransA,
    effectiveTransB,
    aWasCopied,
    bWasCopied,
    outShape,
    m,
    k,
    n,
    batchSize,
    strideA,
    strideB,
    strideC,
  } = prepareMatmulInputs(a, b, transA, transB);

  // Derive per-input dtypes.
  // When inputCast is set, buffer is wider (e.g. f32) but matmul computes in target (e.g. f16).
  const rawDtypeA =
    effectiveA.dtype === "f16" && isF16Supported()
      ? ("f16" as const)
      : ("f32" as const);
  const rawDtypeB =
    effectiveB.dtype === "f16" && isF16Supported()
      ? ("f16" as const)
      : ("f32" as const);
  const dtypeA: "f16" | "f32" =
    epilogueOpts?.inputCastA === "f16" && isF16Supported() ? "f16" : rawDtypeA;
  const dtypeB: "f16" | "f32" =
    epilogueOpts?.inputCastB === "f16" && isF16Supported() ? "f16" : rawDtypeB;
  const promotedDtype =
    dtypeA === "f32" || dtypeB === "f32" ? ("f32" as const) : dtypeA;
  const outputDtype = epilogueOpts?.epilogue.outputDtype ?? promotedDtype;
  const bytesPerElement = dtypeBytes(outputDtype);

  const epilogueBuffers =
    epilogueOpts?.epilogueInputs.map((t) => t.buffer) ?? [];

  // Create or use donated output buffer
  const outSize = sizeOf(outShape);
  const requiredSize = outSize * bytesPerElement;
  const useDonated = donatedBuffer && donatedBuffer.size >= requiredSize;
  const outBuffer = useDonated
    ? donatedBuffer
    : resolveOutputBuffer(ctx.device, requiredSize, [
        effectiveA.buffer,
        effectiveB.buffer,
        ...epilogueBuffers,
      ]);

  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: effectiveA.buffer,
    b: effectiveB.buffer,
    out: outBuffer,
    m,
    n,
    k,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    transA: effectiveTransA,
    transB: effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
    epilogue: epilogueOpts?.epilogue,
    epilogueInputs: epilogueBuffers.length > 0 ? epilogueBuffers : undefined,
    inputCastA: epilogueOpts?.inputCastA as MatmulDType | undefined,
    inputCastB: epilogueOpts?.inputCastB as MatmulDType | undefined,
  });

  if (aWasCopied) destroyCopy(effectiveA);
  if (bWasCopied) destroyCopy(effectiveB);

  return createTensor(
    outShape,
    outBuffer,
    undefined,
    0,
    outputDtype,
    useDonated || !epilogueOpts,
  );
}

/**
 * Direct matmul dispatch with pre-computed geometry.
 *
 * Used by the lowered plan fast path to skip shape computation,
 * transpose detection, contiguous checks, and dtype resolution.
 * All structural decisions were computed on the first execution and cached.
 */
export function dispatchMatmulDirect(
  bufA: GPUBuffer,
  bufB: GPUBuffer,
  config: {
    m: number;
    n: number;
    k: number;
    transA: boolean;
    transB: boolean;
    batchSize: number;
    batchStrideA: number;
    batchStrideB: number;
    batchStrideC: number;
    outShape: number[];
    dtypeA: "f16" | "f32";
    dtypeB?: "f16" | "f32";
    outputDtype: DType;
    epilogueConfig: EpilogueConfig | undefined;
    epilogueBuffers: GPUBuffer[];
    inputCastA?: MatmulDType;
    inputCastB?: MatmulDType;
  },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = dtypeBytes(config.outputDtype);
  const outSize = sizeOf(config.outShape);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bytesPerElement, [
    bufA,
    bufB,
    ...config.epilogueBuffers,
  ]);

  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: bufA,
    b: bufB,
    out: outBuffer,
    m: config.m,
    n: config.n,
    k: config.k,
    batchSize: config.batchSize,
    batchStrideA: config.batchStrideA,
    batchStrideB: config.batchStrideB,
    batchStrideC: config.batchStrideC,
    transA: config.transA,
    transB: config.transB,
    dtype: config.dtypeA,
    dtypeB: config.dtypeB,
    epilogue: config.epilogueConfig,
    epilogueInputs: config.epilogueBuffers,
    inputCastA: config.inputCastA,
    inputCastB: config.inputCastB,
  });

  return createTensor(
    config.outShape,
    outBuffer,
    undefined,
    0,
    config.outputDtype,
  );
}
