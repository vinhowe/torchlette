/**
 * Core dispatch infrastructure: compute pass execution, pipeline caching,
 * binary/unary/matmul dispatch.
 * Extracted from index.ts — purely structural refactoring.
 */

import {
  sizeOf,
  broadcastShapes,
  toIndexShape,
  computeEffectiveBroadcastStrides,
  dtypeBytes,
  compute2DDispatch,
  WORKGROUP_SIZE,
} from "./shape-utils";
import type { DType } from "../types";
import type { GPUBuffer, GPUComputePipeline, GPUBindGroup, WebGPUContext, WebGPUTensor } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { recordPipeline, getWarmupPipeline } from "./pipeline-warmup";
import { requireContext, isF16Supported } from "./gpu-context";
import { destroyCopy } from "./buffer-pool";
import {
  getSharedEncoderInstance,
  autoFlushSharedEncoder,
  incrementSharedEncoderPassCount,
  submitOrCollect,
  getCurrentOpLabel,
} from "./shared-encoder";
import { dispatchRecordingBuffer, getAndClearLastBindGroupBuffers, type RecordedDispatch } from "./dispatch-recording";
import { resolveOutputBuffer } from "./buffer-arena";
import {
  createParamsBuffer,
  releaseParamsBuffer,
  cachedCreateBindGroup,
  params,
} from "./bind-group-cache";
import { getTimestampWrites, getProfileModule } from "./profiler";
import { binaryBroadcastTileIR, binaryBroadcastSpec, unaryStridedTileIR, unaryStridedSpec } from "./ops/ops-tile-ir";
import { createTileKernelDispatcher } from "./tile-dispatch";

import {
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  dispatchTiledMatmul,
} from "./matmul/dispatch";
import type { EpilogueConfig } from "./matmul/types";

// Tensor construction helpers (extracted to tensor.ts)
import { createTensor } from "./tensor";
import { ensureContiguous, detectSimpleTranspose } from "./ops/views";

export function dispatchComputePass(
  pipeline: GPUComputePipeline,
  bindGroup: unknown,
  workgroupsX: number,
  workgroupsY: number = 1,
  workgroupsZ: number = 1,
  recordingBuffer?: RecordedDispatch[] | null,
  labelOverride?: string,
): void {
  const ctx = requireContext();
  const label = labelOverride ?? getCurrentOpLabel();
  const recBuf = recordingBuffer !== undefined ? recordingBuffer : dispatchRecordingBuffer;

  // Record dispatch if recording is active
  if (recBuf) {
    recBuf.push({
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
  if (sharedEnc) {
    // Encode directly onto the shared encoder — no new encoder or CB
    const pass = sharedEnc.beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    incrementSharedEncoderPassCount();
    autoFlushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    const pass = encoder.beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
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

  const outBuffer = desc.outBuffer
    ?? resolveOutputBuffer(ctx.device, desc.outputSizeBytes, desc.inputs);

  const paramsBuffer = createParamsBuffer(ctx.device, desc.params);

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < desc.inputs.length; i++) {
    bgBuffers.push(desc.inputs[i]);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(paramsBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

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
  const maxBindingSize = ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const aSizeBytes = a.size * bytesPerElement;
  const bSizeBytes = b.size * bytesPerElement;
  const outSizeBytes = outSize * bytesPerElement;

  // Large tensor chunked dispatch path
  if (aSizeBytes > maxBindingSize || bSizeBytes > maxBindingSize || outSizeBytes > maxBindingSize) {
    const aIsScalar = a.size === 1;
    const bIsScalar = b.size === 1;
    const sameShape = a.shape.length === b.shape.length &&
      a.shape.every((d, i) => d === b.shape[i]);

    if ((sameShape && a.isContiguous && b.isContiguous) ||
        (aIsScalar && b.isContiguous) ||
        (bIsScalar && a.isContiguous)) {
      return binaryChunked(op, a, b, aIsScalar, bIsScalar, outShape, outSize, dtype, bytesPerElement, options);
    }

    if (sameShape) {
      const aC = a.isContiguous ? a : ensureContiguous(a);
      const bC = b.isContiguous ? b : ensureContiguous(b);
      const result = binaryChunked(op, aC, bC, aC.size === 1, bC.size === 1, outShape, outSize, dtype, bytesPerElement, options);
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

  // Direct dispatch
  const indexShape = toIndexShape(outShape);
  const aStrides = computeEffectiveBroadcastStrides(a, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(b, indexShape);
  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const code = binaryBroadcastTileIR(op, indexShape, aStrides, bStrides, a.offset, b.offset, dtype);
  const key = `binary:${op}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${a.offset}:${b.offset}:${dtype}:${use2D ? dispatch.gridSizeX : "1d"}`;

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [a.buffer, b.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params(outSize),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/** Chunked binary dispatch for large contiguous tensors exceeding maxStorageBufferBindingSize. */
function binaryChunked(
  op: string, a: WebGPUTensor, b: WebGPUTensor,
  aIsScalar: boolean, bIsScalar: boolean,
  outShape: number[], outSize: number, dtype: DType, bytesPerElement: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bytesPerElement, [a.buffer, b.buffer], options?.outBuffer);
  const spec = binaryBroadcastSpec(
    op, [outSize],
    aIsScalar ? [0] : [1], bIsScalar ? [0] : [1],
    0, 0, dtype as "f32" | "f16" | "u32" | "i32",
  );
  const dispatcher = createTileKernelDispatcher(spec);
  dispatcher.dispatchChunked(
    { a: a.buffer, b: b.buffer, out: outBuffer },
    { size: outSize },
    {
      modes: { a: aIsScalar ? "scalar" : "chunked", b: bIsScalar ? "scalar" : "chunked", out: "chunked" },
      sizeUniform: "size", totalElements: outSize, maxBytesPerElement: bytesPerElement,
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
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);
  const maxBindingSize = ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Chunked dispatch for large contiguous tensors
  if (a.size * bytesPerElement > maxBindingSize && a.isContiguous) {
    const outSize = a.size;
    const outBuffer = resolveOutputBuffer(ctx.device, outSize * bytesPerElement, [a.buffer], options?.outBuffer);
    const spec = unaryStridedSpec(opKey, [outSize], [1], 0, dtype as "f32" | "f16" | "u32" | "i32");
    const dispatcher = createTileKernelDispatcher(spec);
    dispatcher.dispatchChunked(
      { a: a.buffer, out: outBuffer },
      { size: outSize },
      { modes: { a: "chunked", out: "chunked" }, sizeUniform: "size", totalElements: outSize, maxBytesPerElement: bytesPerElement },
    );
    const ownsBuffer = options?.outBuffer === undefined;
    return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
  }

  // Direct dispatch
  const totalWorkgroups = Math.ceil(a.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const code = unaryStridedTileIR(opKey, a.shape, a.strides, a.offset, dtype);
  const key = `unary:${opKey}:${a.shape.join("x")}:${a.strides.join(",")}:${a.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [a.buffer],
    outputSizeBytes: a.size * bytesPerElement,
    params: params(a.size),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
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
  // Try to detect simple last-2-dim transposes to avoid contiguous() materialization.
  // If detected, we use the original contiguous buffer and flip the transpose flag.
  let effectiveA: WebGPUTensor = a;
  let effectiveTransA = transA;
  let aWasCopied = false;

  const aOrigShape = !transA ? detectSimpleTranspose(a) : null;
  if (aOrigShape) {
    effectiveA = createTensor(aOrigShape, a.buffer, undefined, 0, a.dtype, false);
    effectiveTransA = true;
  } else {
    effectiveA = ensureContiguous(a);
    aWasCopied = effectiveA !== a;
  }

  let effectiveB: WebGPUTensor = b;
  let effectiveTransB = transB;
  let bWasCopied = false;

  const bOrigShape = !transB ? detectSimpleTranspose(b) : null;
  if (bOrigShape) {
    effectiveB = createTensor(bOrigShape, b.buffer, undefined, 0, b.dtype, false);
    effectiveTransB = true;
  } else {
    effectiveB = ensureContiguous(b);
    bWasCopied = effectiveB !== b;
  }

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
    effectiveA, effectiveB, effectiveTransA, effectiveTransB,
    aWasCopied, bWasCopied, outShape, m, k, n, batchDims, batchSize,
    strideA, strideB, strideC,
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
    inputCastA?: DType;
    inputCastB?: DType;
  },
): WebGPUTensor {
  const ctx = requireContext();

  const {
    effectiveA, effectiveB, effectiveTransA, effectiveTransB,
    aWasCopied, bWasCopied, outShape, m, k, n, batchSize,
    strideA, strideB, strideC,
  } = prepareMatmulInputs(a, b, transA, transB);

  // Derive per-input dtypes.
  // When inputCast is set, buffer is wider (e.g. f32) but matmul computes in target (e.g. f16).
  const rawDtypeA = effectiveA.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const rawDtypeB = effectiveB.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const dtypeA: "f16" | "f32" = epilogueOpts?.inputCastA === "f16" && isF16Supported() ? "f16" : rawDtypeA;
  const dtypeB: "f16" | "f32" = epilogueOpts?.inputCastB === "f16" && isF16Supported() ? "f16" : rawDtypeB;
  const promotedDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
  const outputDtype = epilogueOpts?.epilogue.outputDtype ?? promotedDtype;
  const bytesPerElement = outputDtype === "f16" ? 2 : 4;

  const epilogueBuffers = epilogueOpts?.epilogueInputs.map((t) => t.buffer) ?? [];

  // Create or use donated output buffer
  const outSize = sizeOf(outShape);
  const requiredSize = outSize * bytesPerElement;
  const useDonated = donatedBuffer && donatedBuffer.size >= requiredSize;
  const outBuffer = useDonated
    ? donatedBuffer
    : resolveOutputBuffer(ctx.device, requiredSize,
        [effectiveA.buffer, effectiveB.buffer, ...epilogueBuffers]);

  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: effectiveA.buffer,
    b: effectiveB.buffer,
    out: outBuffer,
    m, n, k, batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    transA: effectiveTransA,
    transB: effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
    epilogue: epilogueOpts?.epilogue,
    epilogueInputs: epilogueBuffers.length > 0 ? epilogueBuffers : undefined,
    inputCastA: epilogueOpts?.inputCastA,
    inputCastB: epilogueOpts?.inputCastB,
  });

  if (aWasCopied) destroyCopy(effectiveA);
  if (bWasCopied) destroyCopy(effectiveB);

  return createTensor(outShape, outBuffer, undefined, 0, outputDtype, useDonated || !epilogueOpts);
}

/** @deprecated Use dispatchMatmul with epilogueOpts parameter instead. */
export function dispatchMatmulWithEpilogue(
  a: WebGPUTensor,
  b: WebGPUTensor,
  epilogue: EpilogueConfig,
  epilogueInputs: WebGPUTensor[],
  transA = false,
  transB = false,
  inputCastA?: DType,
  inputCastB?: DType,
): WebGPUTensor {
  return dispatchMatmul(a, b, transA, transB, undefined, {
    epilogue, epilogueInputs, inputCastA, inputCastB,
  });
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
    m: number; n: number; k: number;
    transA: boolean; transB: boolean;
    batchSize: number;
    batchStrideA: number; batchStrideB: number; batchStrideC: number;
    outShape: number[];
    dtypeA: "f16" | "f32";
    dtypeB?: "f16" | "f32";
    outputDtype: DType;
    epilogueConfig: EpilogueConfig | undefined;
    epilogueBuffers: GPUBuffer[];
    inputCastA?: DType;
    inputCastB?: DType;
  },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = config.outputDtype === "f16" ? 2 : 4;
  const outSize = sizeOf(config.outShape);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [bufA, bufB, ...config.epilogueBuffers],
  );

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

  return createTensor(config.outShape, outBuffer, undefined, 0, config.outputDtype);
}
