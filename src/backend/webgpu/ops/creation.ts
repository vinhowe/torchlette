/**
 * Tensor creation ops: tensorFromArray, zeros, full, arange, tril/triu, rand/randn/bernoulli.
 */

import type { WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { sizeOf, WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM, compute2DDispatch, alignBufferSize } from "../shape-utils";
import { fillWGSL, arangeWGSL, triangularWGSL, randWGSL, randnWGSL, bernoulliWGSL } from "./ops-tile-ir";
import { requireContext, f32ArrayToF16Array } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer, createBufferWithData } from "../tensor";
import { resolveOutputBuffer, getActiveArena, arenaBufferSet } from "../buffer-arena";
import { cachedCreateBindGroup, createParamsBuffer, releaseParamsBuffer } from "../bind-group-cache";
import { profileApiCall } from "../profiler";
import { bufferPool, destroyCopy } from "../buffer-pool";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import type { DType } from "../../types";

// Lazy import to avoid circular dependency: contiguous is defined in ../index.ts
// and ../index.ts imports from ./ops/creation.ts
let _contiguous: ((a: WebGPUTensor) => WebGPUTensor) | null = null;
export function _setContiguous(fn: (a: WebGPUTensor) => WebGPUTensor): void {
  _contiguous = fn;
}
function getContiguous(a: WebGPUTensor): WebGPUTensor {
  if (!_contiguous) throw new Error("contiguous not wired up — call _setContiguous first");
  return _contiguous(a);
}

export function tensorFromArray(values: number[] | Float32Array, shape: number[]): WebGPUTensor {
  const ctx = requireContext();
  const expected = sizeOf(shape);
  if (expected !== values.length) {
    throw new Error("Tensor data length does not match shape");
  }
  const f32data = values instanceof Float32Array ? values : Float32Array.from(values);
  // Arena fast path: use resolveOutputBuffer for stable buffer identity across steps.
  // This eliminates bind group cache misses from data-source ops in lowered plans.
  if (getActiveArena()) {
    const buffer = resolveOutputBuffer(ctx.device, f32data.byteLength, []);
    profileApiCall("writeBuffer", () => ctx.queue.writeBuffer(buffer, 0, f32data));
    return createTensor(shape, buffer);
  }
  const buffer = createBufferWithData(
    ctx.device,
    f32data,
    ctx.queue,
  );
  return createTensor(shape, buffer);
}

/** Cached fill WGSL generated via tile-IR. */
let _fillWGSL: string | null = null;
function getFillWGSL(): string {
  return _fillWGSL ?? (_fillWGSL = fillWGSL());
}

/**
 * Create a zero-filled tensor efficiently.
 * Allocates a GPU buffer directly — no JS array, no upload.
 * If the buffer comes from the pool (stale data), clears it with clearBuffer.
 * Fresh buffers are zero-initialized by the WebGPU spec.
 */
export function zeros(shape: number[]): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }
  const sizeBytes = numElements * 4; // f32
  const alignedSize = alignBufferSize(sizeBytes);
  // Arena-aware output allocation for stable buffer identity across steps
  const buffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Arena and pooled buffers may contain stale data — always clear to zero on GPU.
  // Fresh buffers are already zero, so this is a no-op on the GPU side.
  if (bufferPool.isFromPool(buffer) || arenaBufferSet.has(buffer)) {
    if (getSharedEncoderInstance()) {
      getSharedEncoderInstance().clearBuffer(buffer, 0, alignedSize);
    } else {
      const encoder = ctx.device.createCommandEncoder();
      encoder.clearBuffer(buffer, 0, alignedSize);
      submitOrCollect(encoder.finish());
    }
  }

  return createTensor(shape, buffer);
}

/**
 * Create a tensor filled with a constant value.
 * Uses a GPU compute shader to fill the buffer — no JS array allocation.
 * fillValue === 0 is special-cased to use the zero-cost zeros() path.
 */
export function full(shape: number[], fillValue: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }

  // Special case: fillValue === 0 → use zeros path (WebGPU auto-zeros or clearBuffer)
  if (fillValue === 0) {
    return zeros(shape);
  }

  const sizeBytes = numElements * 4; // f32
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const shader = getFillWGSL();
  const pipeline = getPipeline(ctx, "fill_tile", shader);

  // Arena-aware output allocation for stable buffer identity across steps
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Params: [numElements as u32, fillValue as f32 (reinterpreted as u32 bits)]
  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  new Float32Array(paramsData.buffer, 4, 1)[0] = fillValue;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer);
}

/** Cached arange WGSL generated via tile-IR. */
let _arangeWGSL: string | null = null;
function getArangeWGSL(): string {
  return _arangeWGSL ?? (_arangeWGSL = arangeWGSL());
}

/**
 * Create a 1-D tensor of evenly spaced values on the GPU.
 * No JS array allocation — values are computed directly by the GPU.
 */
export function arange(end: number, start = 0, step = 1): WebGPUTensor {
  const ctx = requireContext();
  const numElements = Math.max(0, Math.ceil((end - start) / step));
  if (numElements === 0) {
    throw new Error("webgpu tensors cannot be empty yet");
  }

  const sizeBytes = numElements * 4; // f32
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const shader = getArangeWGSL();
  const pipeline = getPipeline(ctx, "arange_tile", shader);

  // Arena-aware output allocation for stable buffer identity across steps
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Params: [numElements as u32, start as f32, step as f32]
  const paramsData = new Uint32Array(3);
  paramsData[0] = numElements;
  new Float32Array(paramsData.buffer, 4, 1)[0] = start;
  new Float32Array(paramsData.buffer, 8, 1)[0] = step;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor([numElements], outBuffer);
}

/** Cached triangular WGSL (tril/triu) generated via tile-IR. */
let _trilWGSL: string | null = null;
let _triuWGSL: string | null = null;
function getTriangularWGSL(upper: boolean): string {
  if (upper) return _triuWGSL ?? (_triuWGSL = triangularWGSL(true));
  return _trilWGSL ?? (_trilWGSL = triangularWGSL(false));
}

/**
 * Triangular operation: zero elements above (tril) or below (triu) a diagonal.
 * Operates on the last 2 dimensions; supports arbitrary batch dimensions.
 */
function triangularOp(a: WebGPUTensor, k: number, upper: boolean): WebGPUTensor {
  const ctx = requireContext();
  if (a.shape.length < 2) throw new Error("tril/triu requires at least 2 dimensions");

  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const numElements = sizeOf(a.shape);

  // Ensure contiguous input
  const input = a.isContiguous ? a : getContiguous(a);

  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const tag = upper ? "triu" : "tril";
  const shader = getTriangularWGSL(upper);
  const pipeline = getPipeline(ctx, `${tag}_tile`, shader);

  const sizeBytes = numElements * 4;
  const alignedSize = alignBufferSize(sizeBytes);
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = createTrackedBuffer(ctx.device, { size: alignedSize, usage });

  // Params: [numElements as u32, H as u32, W as u32, k as i32]
  const paramsData = new Int32Array(4);
  new Uint32Array(paramsData.buffer, 0, 1)[0] = numElements;
  new Uint32Array(paramsData.buffer, 4, 1)[0] = H;
  new Uint32Array(paramsData.buffer, 8, 1)[0] = W;
  paramsData[3] = k;
  const paramsBuffer = createParamsBuffer(ctx.device, new Uint32Array(paramsData.buffer));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [input.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  // Destroy contiguous copy if one was created (deferred for GPU fence)
  if (input !== a) destroyCopy(input);

  return createTensor(a.shape.slice(), outBuffer, undefined, 0, a.dtype);
}

export function tril(a: WebGPUTensor, k = 0): WebGPUTensor {
  return triangularOp(a, k, false);
}

export function triu(a: WebGPUTensor, k = 0): WebGPUTensor {
  return triangularOp(a, k, true);
}

// ============================================================================
// GPU RNG (Philox 2x32-10 via tile-IR)
// ============================================================================

let _randWGSL: string | null = null;
function getRandWGSL(): string { return _randWGSL ?? (_randWGSL = randWGSL()); }

let _randnWGSL: string | null = null;
function getRandnWGSL(): string { return _randnWGSL ?? (_randnWGSL = randnWGSL()); }

let _bernoulliWGSL: string | null = null;
function getBernoulliWGSL(): string { return _bernoulliWGSL ?? (_bernoulliWGSL = bernoulliWGSL()); }

export function rand(shape: number[], seed: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const sizeBytes = numElements * 4;
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const pipeline = getPipeline(ctx, "rand_tile", getRandWGSL());
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);
  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer);
}

export function randn(shape: number[], seed: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const sizeBytes = numElements * 4;
  const numThreads = Math.ceil(numElements / 2);
  const totalWorkgroups = Math.ceil(numThreads / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const pipeline = getPipeline(ctx, "randn_tile", getRandnWGSL());
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Shader uniforms: {size: u32, seed: u32}
  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);
  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer);
}

export function bernoulli(shape: number[], p: number, seed: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const sizeBytes = numElements * 4;
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const pipeline = getPipeline(ctx, "bernoulli_tile", getBernoulliWGSL());
  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Params: [size as u32, seed as u32, prob as f32] — 3 words, padded to 4 for alignment
  const paramsData = new Uint32Array(4);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  new Float32Array(paramsData.buffer, 8, 1)[0] = p;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);
  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer);
}

/**
 * Create a tensor from an array with a specific dtype.
 * Supports f32 (default), i32, u32, and f16 (if device supports shader-f16).
 */
export function tensorFromArrayWithDtype(
  values: number[],
  shape: number[],
  dtype: DType,
): WebGPUTensor {
  const ctx = requireContext();
  const expected = sizeOf(shape);
  if (expected !== values.length) {
    throw new Error("Tensor data length does not match shape");
  }

  // Check f16 support
  if (dtype === "f16" && !ctx.f16Supported) {
    throw new Error(
      "f16 dtype requires shader-f16 device feature which is not available",
    );
  }

  let typedData: Float32Array | Int32Array | Uint32Array | Uint16Array;
  switch (dtype) {
    case "i32":
      typedData = Int32Array.from(values);
      break;
    case "u32":
      typedData = Uint32Array.from(values);
      break;
    case "f16":
      typedData = f32ArrayToF16Array(values);
      break;
    case "f32":
    default:
      typedData = Float32Array.from(values);
      break;
  }

  // Arena fast path: use resolveOutputBuffer for stable buffer identity across steps
  if (getActiveArena()) {
    const buffer = resolveOutputBuffer(ctx.device, typedData.byteLength, []);
    profileApiCall("writeBuffer", () => ctx.queue.writeBuffer(buffer, 0, typedData));
    return createTensor(shape, buffer, undefined, 0, dtype);
  }
  const buffer = createBufferWithData(ctx.device, typedData, ctx.queue);
  return createTensor(shape, buffer, undefined, 0, dtype);
}
