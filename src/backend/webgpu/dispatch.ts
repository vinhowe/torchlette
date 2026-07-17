/**
 * Core dispatch infrastructure: compute pass execution, pipeline caching,
 * binary/unary/matmul dispatch.
 */

import type { DType } from "../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  profiledCreateBindGroup,
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
  planTiledMatmul,
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
import { getTimestampWrites } from "./profiler";
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
import type {
  ChunkedBindingConfig,
  TileKernelChunkedPlan,
} from "./tile-dispatch";
import { createTileKernelDispatcher } from "./tile-dispatch";
import type { TileKernelSpec } from "./tile-ir";

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
  const shader = binaryBroadcastTileIR(
    op,
    indexShape,
    aStrides,
    bStrides,
    a.offset,
    b.offset,
    dtype as "f32" | "f16" | "u32" | "i32",
  );
  return {
    // Key IS the WGSL (single-source-at-seams). dispatchBinary's direct tail
    // and the stream generator BOTH consume this plan; keying on code keeps
    // them from silently disagreeing. See planCastDirectCore.
    key: shader,
    shader,
    outputSizeBytes: outSize * bytesPerElement,
    // Task #71: a/b offsets ride paramsData after size, in uniform-declaration
    // order (size, a_offset, b_offset). Compiled replay re-derives both from
    // the current step's input view offsets (volatile repack).
    paramsData: params(outSize, a.offset, b.offset),
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

    // Chunked binding slices buffers from element 0 — require offset 0 in
    // addition to contiguous strides (offset-view class, task #58).
    const aRB = a.isContiguous && (a.offset ?? 0) === 0;
    const bRB = b.isContiguous && (b.offset ?? 0) === 0;
    if (
      (sameShape && aRB && bRB) ||
      (aIsScalar && aRB && bRB) ||
      (bIsScalar && bRB && aRB)
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
      const aC = ensureContiguous(a);
      const bC = ensureContiguous(b);
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

    if (
      !(a.isContiguous && (a.offset ?? 0) === 0) &&
      aSizeBytes > maxBindingSize
    ) {
      const aC = ensureContiguous(a);
      const result = dispatchBinary(op, aC, b, options);
      aC.destroy?.();
      return result;
    }
    if (
      !(b.isContiguous && (b.offset ?? 0) === 0) &&
      bSizeBytes > maxBindingSize
    ) {
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

/** The spec + uniforms + chunking config for a chunked binary elementwise op.
 *  SINGLE SOURCE consumed by both the execution path (binaryChunked →
 *  dispatchChunked) and the stream generator (planChunkedBinary → planChunked),
 *  so the recorded and generated command streams cannot drift. Task #71: chunked
 *  binds each operand from element 0 → offsets are 0. */
function chunkedBinaryConfig(
  op: string,
  aIsScalar: boolean,
  bIsScalar: boolean,
  outSize: number,
  dtype: DType,
  bytesPerElement: number,
): {
  spec: TileKernelSpec;
  uniforms: Record<string, number>;
  chunking: ChunkedBindingConfig;
} {
  return {
    spec: binaryBroadcastSpec(
      op,
      [outSize],
      aIsScalar ? [0] : [1],
      bIsScalar ? [0] : [1],
      0,
      0,
      dtype as "f32" | "f16" | "u32" | "i32",
    ),
    uniforms: { size: outSize, a_offset: 0, b_offset: 0 },
    chunking: {
      modes: {
        a: aIsScalar ? "scalar" : "chunked",
        b: bIsScalar ? "scalar" : "chunked",
        out: "chunked",
      },
      sizeUniform: "size",
      totalElements: outSize,
      maxBytesPerElement: bytesPerElement,
    },
  };
}

/** Stage-4 chunked plan: the chunked binary elementwise dispatch fully described
 *  WITHOUT encoding, derived from the SAME chunkedBinaryConfig the execution path
 *  dispatches. The stream generator (generateChunkedBinary) consumes this. */
export function planChunkedBinary(
  op: string,
  aIsScalar: boolean,
  bIsScalar: boolean,
  outSize: number,
  dtype: DType,
): TileKernelChunkedPlan {
  const bytesPerElement = dtypeBytes(dtype);
  const { spec, uniforms, chunking } = chunkedBinaryConfig(
    op,
    aIsScalar,
    bIsScalar,
    outSize,
    dtype,
    bytesPerElement,
  );
  return createTileKernelDispatcher(spec).planChunked(uniforms, chunking);
}

/**
 * Execute a chunked elementwise op from a TileKernelChunkedPlan. Uses a FRESH
 * per-chunk params buffer (createParamsBuffer → recorded as a self-contained
 * `params` slot) rather than the tile-dispatch config cache (a `persistent`
 * config buffer the build-from-IR replay cannot own) — exactly the
 * sumFullReductionChunked pattern, and the reason chunked elementwise is now
 * generatable. Both execution and the stream generator consume `plan`, so their
 * command streams cannot drift. `buffers` maps spec binding names (a/b/out) to
 * GPUBuffers; the uniform config slot (null in bindingOrder) binds the params.
 */
function dispatchChunkedElementwise(
  plan: TileKernelChunkedPlan,
  buffers: Record<string, GPUBuffer>,
): void {
  const ctx = requireContext();
  for (const chunk of plan.chunks) {
    const u32 = new Uint32Array(
      chunk.paramsData.buffer,
      chunk.paramsData.byteOffset,
      chunk.paramsData.byteLength >> 2,
    );
    const paramsBuffer = createParamsBuffer(ctx.device, u32);
    const entries = plan.bindingOrder.map((name, i) => {
      if (name === null) {
        return { binding: i, resource: { buffer: paramsBuffer } };
      }
      const range = chunk.ranges[i];
      const buf = buffers[name];
      return {
        binding: i,
        resource: range
          ? { buffer: buf, offset: range.offset, size: range.size }
          : { buffer: buf },
      };
    });
    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: plan.pipeline.getBindGroupLayout(0),
      entries,
    });
    dispatchComputePass(
      plan.pipeline,
      bindGroup,
      chunk.grid[0],
      chunk.grid[1],
      chunk.grid[2],
    );
    releaseParamsBuffer(paramsBuffer);
  }
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
  const plan = planChunkedBinary(op, aIsScalar, bIsScalar, outSize, dtype);
  dispatchChunkedElementwise(plan, {
    a: a.buffer,
    b: b.buffer,
    out: outBuffer,
  });
  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Dispatch unary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
/** Stage-4 plan/encode split for the DIRECT unary path (see
 *  planBinaryDirect for the pattern). Returns null where dispatchUnary
 *  routes to chunked dispatch. */
export function planUnaryDirect(
  opKey: string,
  a: WebGPUTensor,
): ElementwiseDirectPlan | null {
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);
  const ctx = requireContext();
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  if (a.size * bytesPerElement > maxBindingSize) return null;
  return planUnaryDirectCore(opKey, a, bytesPerElement);
}

/** Unguarded unary plan core — dispatchUnary's direct tail (post-routing). */
export function planUnaryDirectCore(
  opKey: string,
  a: WebGPUTensor,
  bytesPerElement: number,
): ElementwiseDirectPlan {
  const dtype = a.dtype;
  const outDtype = unaryOutputDtype(opKey, dtype);
  const outBytesPerElement =
    outDtype === dtype ? bytesPerElement : dtypeBytes(outDtype);
  const totalWorkgroups = Math.ceil(a.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const shader = unaryStridedTileIR(
    opKey,
    a.shape,
    a.strides,
    a.offset,
    dtype as "f32" | "f16" | "u32" | "i32",
    outDtype as "f32" | "f16" | "u32" | "i32",
  );
  return {
    // Key IS the WGSL (single-source-at-seams; consumed by dispatchUnary and
    // the stream generator). See planCastDirectCore.
    key: shader,
    shader,
    outputSizeBytes: a.size * outBytesPerElement,
    // Task #71: offset via uniform (after size). See planBinaryDirectCore.
    paramsData: params(a.size, a.offset),
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  };
}

/** Output dtype for a unary op. isfinite is the only always_f32 unary (it
 *  returns a 0/1 mask regardless of input dtype); everything else preserves
 *  the input dtype. Kept local (the backend does not import the op registry);
 *  the runtime's always_f32 rule is the single source of truth for the tensor
 *  dtype — this must agree with it. */
function unaryOutputDtype(opKey: string, inDtype: DType): DType {
  return opKey === "isfinite" ? "f32" : inDtype;
}

/** The spec + uniforms + chunking config for a chunked unary elementwise op.
 *  SINGLE SOURCE consumed by both the execution path (dispatchUnary's chunked
 *  branch) and the stream generator (planChunkedUnary → planChunked). Task #71:
 *  chunked binds from element 0 → base_offset is 0. The chunk element stride
 *  uses the LARGER of in/out bytes so a dtype-widening op never over-runs. */
function chunkedUnaryConfig(
  opKey: string,
  outSize: number,
  dtype: DType,
  outDtype: DType,
  bytesPerElement: number,
  outBytesPerElement: number,
): {
  spec: TileKernelSpec;
  uniforms: Record<string, number>;
  chunking: ChunkedBindingConfig;
} {
  return {
    spec: unaryStridedSpec(
      opKey,
      [outSize],
      [1],
      0,
      dtype as "f32" | "f16" | "u32" | "i32",
      outDtype as "f32" | "f16" | "u32" | "i32",
    ),
    uniforms: { size: outSize, base_offset: 0 },
    chunking: {
      modes: { a: "chunked", out: "chunked" },
      sizeUniform: "size",
      totalElements: outSize,
      maxBytesPerElement: Math.max(bytesPerElement, outBytesPerElement),
    },
  };
}

/** Stage-4 chunked plan: the chunked unary elementwise dispatch fully described
 *  WITHOUT encoding, derived from the SAME chunkedUnaryConfig the execution path
 *  dispatches. The stream generator (generateChunkedUnary) consumes this. */
export function planChunkedUnary(
  opKey: string,
  outSize: number,
  dtype: DType,
): TileKernelChunkedPlan {
  const bytesPerElement = dtypeBytes(dtype);
  const outDtype = unaryOutputDtype(opKey, dtype);
  const outBytesPerElement =
    outDtype === dtype ? bytesPerElement : dtypeBytes(outDtype);
  const { spec, uniforms, chunking } = chunkedUnaryConfig(
    opKey,
    outSize,
    dtype,
    outDtype,
    bytesPerElement,
    outBytesPerElement,
  );
  return createTileKernelDispatcher(spec).planChunked(uniforms, chunking);
}

export function dispatchUnary(
  opKey: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);
  // Output dtype may differ from input (isfinite is always_f32). Size the
  // output buffer and stamp the returned tensor from it, not the input dtype.
  const outDtype = unaryOutputDtype(opKey, dtype);
  const outBytesPerElement =
    outDtype === dtype ? bytesPerElement : dtypeBytes(outDtype);
  const maxBindingSize =
    ctx.device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;

  // Chunked dispatch for large contiguous tensors (offset 0 required: the
  // chunked binding slices the buffer from element 0 — offset-view class,
  // task #58). The chunk element stride uses the LARGER of in/out bytes so a
  // dtype-widening op (f16→f32 isfinite) never over-runs a binding.
  if (
    a.size * bytesPerElement > maxBindingSize &&
    a.isContiguous &&
    (a.offset ?? 0) === 0
  ) {
    const outSize = a.size;
    const outBuffer = resolveOutputBuffer(
      ctx.device,
      outSize * outBytesPerElement,
      [a.buffer],
      options?.outBuffer,
    );
    const plan = planChunkedUnary(opKey, outSize, dtype);
    dispatchChunkedElementwise(plan, { a: a.buffer, out: outBuffer });
    const ownsBuffer = options?.outBuffer === undefined;
    return createTensor(a.shape, outBuffer, undefined, 0, outDtype, ownsBuffer);
  }

  // Direct dispatch — single-sourced with the stream generator.
  const plan = planUnaryDirectCore(opKey, a, bytesPerElement);

  const outBuffer = dispatchElementwise({
    key: plan.key,
    shader: plan.shader,
    inputs: [a.buffer],
    outputSizeBytes: plan.outputSizeBytes,
    params: plan.paramsData,
    outBuffer: options?.outBuffer,
    dispatchX: plan.dispatchX,
    dispatchY: plan.dispatchY,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, outDtype, ownsBuffer);
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

/**
 * Stage-4 phase-3: capture a BARE matmul's resolved dispatch plan from LIVE
 * inputs (no epilogue, no donation — the dispatchMatmul slow path). Runs the
 * same prepareMatmulInputs + planTiledMatmul the dispatcher uses. Returns a
 * reason string when the generated stream can't faithfully reproduce it:
 *  - "contiguous-prologue": an input needed a contiguous copy (an extra
 *    prologue dispatch the generator doesn't yet emit);
 *  - "ksplit": the K-split path (op-internal temp + reduction).
 * On success: a simple-transpose input is a non-owning view on the SAME
 * buffer and a contiguous input is itself — so the matmul binds the input
 * buffers directly, and the generator resolves their slots from the node's
 * input refs (works even after the inputs are liveness-released). The result
 * is geometry-only (no live buffers) so it is safe to CACHE on the lowered
 * action and read at generation time. */
export interface BareMatmulPlan {
  outShape: number[];
  outputDtype: DType;
  /** Standard single-dispatch plan, or the two-dispatch K-split plan. */
  matmul:
    | import("./matmul/dispatch").MatmulStandardPlan
    | import("./matmul/dispatch").MatmulKSplitPlan;
}

export function planBareMatmul(
  a: WebGPUTensor,
  b: WebGPUTensor,
): BareMatmulPlan | string {
  const ctx = requireContext();
  // A packed-int (quantized) B operand is routed by the matmul seam
  // (matmulQuantizedB) to a fused-dequant GEMV NT / explicit-dequant kernel —
  // a capability the plain tiled-matmul stream generator does NOT reproduce
  // (it would read the packed u32 buffer as a plain f16 [N,K] weight). The
  // graph compiler already keeps a quantized matmul out of the directive /
  // sequential-capture path (see graph-compiler.ts matmulHasQuantizedB), so
  // this guard is belt-and-suspenders for any other bare-matmul capture site.
  if (b.format?.packing) return "quantized-operand";
  const prep = prepareMatmulInputs(a, b, false, false);
  if (prep.aWasCopied || prep.bWasCopied) {
    if (prep.aWasCopied) destroyCopy(prep.effectiveA);
    if (prep.bWasCopied) destroyCopy(prep.effectiveB);
    return "contiguous-prologue";
  }
  const dtypeA: "f16" | "f32" =
    prep.effectiveA.dtype === "f16" && isF16Supported() ? "f16" : "f32";
  const dtypeB: "f16" | "f32" =
    prep.effectiveB.dtype === "f16" && isF16Supported() ? "f16" : "f32";
  const outputDtype = dtypeA === "f32" || dtypeB === "f32" ? "f32" : dtypeA;
  const plan = planTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: prep.effectiveA.buffer,
    b: prep.effectiveB.buffer,
    out: undefined as unknown as GPUBuffer,
    m: prep.m,
    n: prep.n,
    k: prep.k,
    batchSize: prep.batchSize,
    batchStrideA: prep.strideA,
    batchStrideB: prep.strideB,
    batchStrideC: prep.strideC,
    transA: prep.effectiveTransA,
    transB: prep.effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
  });
  return { outShape: prep.outShape, outputDtype, matmul: plan };
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
