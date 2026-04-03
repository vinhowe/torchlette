/**
 * Tensor creation ops: tensorFromArray, zeros, full, arange, tril/triu, rand/randn/bernoulli.
 */

import type { DType } from "../../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  releaseParamsBuffer,
} from "../bind-group-cache";
import { getActiveArena, resolveOutputBuffer } from "../buffer-arena";
import { bufferPool, destroyCopy } from "../buffer-pool";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { f32ArrayToF16Array, requireContext } from "../gpu-context";
import type { GPUBuffer as LocalGPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { profileApiCall } from "../profiler";
import {
  alignBufferSize,
  compute2DDispatch,
  F32_BYTES,
  sizeOf,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { getSharedEncoderInstance, submitOrCollect } from "../shared-encoder";
import {
  createBufferWithData,
  createTensor,
  createTrackedBuffer,
} from "../tensor";
import { arenaBufferSet } from "../webgpu-state";
import {
  arangeWGSL,
  bernoulliWGSL,
  fillWGSL,
  randnWGSL,
  randWGSL,
  triangularWGSL,
} from "./ops-tile-ir";

// Lazy import to avoid circular dependency: contiguous is defined in ../index.ts
// and ../index.ts imports from ./ops/creation.ts
let _contiguous: ((a: WebGPUTensor) => WebGPUTensor) | null = null;
export function _setContiguous(fn: (a: WebGPUTensor) => WebGPUTensor): void {
  _contiguous = fn;
}
function getContiguous(a: WebGPUTensor): WebGPUTensor {
  if (!_contiguous)
    throw new Error("contiguous not wired up — call _setContiguous first");
  return _contiguous(a);
}

export function tensorFromArray(
  values: number[] | Float32Array,
  shape: number[],
): WebGPUTensor {
  const ctx = requireContext();
  const expected = sizeOf(shape);
  if (expected !== values.length) {
    throw new Error("Tensor data length does not match shape");
  }
  const f32data =
    values instanceof Float32Array ? values : Float32Array.from(values);
  // Arena fast path: use resolveOutputBuffer for stable buffer identity across steps.
  // This eliminates bind group cache misses from data-source ops in lowered plans.
  if (getActiveArena()) {
    const buffer = resolveOutputBuffer(ctx.device, f32data.byteLength, []);
    profileApiCall("writeBuffer", () =>
      ctx.queue.writeBuffer(buffer, 0, f32data),
    );
    return createTensor(shape, buffer);
  }
  const buffer = createBufferWithData(ctx.device, f32data, ctx.queue);
  return createTensor(shape, buffer);
}

/** Lazy WGSL cache: generates once on first call. */
function cachedWGSL(gen: () => string): () => string {
  let wgsl: string | null = null;
  return () => {
    if (wgsl === null) wgsl = gen();
    return wgsl;
  };
}

const getFillWGSL = cachedWGSL(fillWGSL);

/** Dispatch a creation kernel: pipeline + params + bind group + dispatch + release. */
function dispatchCreationKernel(
  cacheKey: string,
  shader: string,
  numElements: number,
  paramsData: Uint32Array,
  workgroupThreads?: number,
): LocalGPUBuffer {
  const ctx = requireContext();
  const threads = workgroupThreads ?? numElements;
  const totalWorkgroups = Math.ceil(threads / WORKGROUP_SIZE);
  const { x, y } = compute2DDispatch(totalWorkgroups);
  const pipeline = getPipeline(ctx, cacheKey, shader);
  const outBuffer = resolveOutputBuffer(ctx.device, numElements * F32_BYTES, []);
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    outBuffer,
    paramsBuffer,
  ]);
  dispatchComputePass(pipeline, bindGroup, x, y);
  releaseParamsBuffer(paramsBuffer);
  return outBuffer;
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
  const sizeBytes = numElements * F32_BYTES;
  const alignedSize = alignBufferSize(sizeBytes);
  // Arena-aware output allocation for stable buffer identity across steps
  const buffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Arena and pooled buffers may contain stale data — always clear to zero on GPU.
  // Fresh buffers are already zero, so this is a no-op on the GPU side.
  if (bufferPool.isFromPool(buffer) || arenaBufferSet.has(buffer)) {
    const sharedEnc = getSharedEncoderInstance();
    if (sharedEnc) {
      sharedEnc.clearBuffer(buffer, 0, alignedSize);
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
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");
  if (fillValue === 0) return zeros(shape);

  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  new Float32Array(paramsData.buffer, 4, 1)[0] = fillValue;
  return createTensor(
    shape,
    dispatchCreationKernel("fill_tile", getFillWGSL(), numElements, paramsData),
  );
}

const getArangeWGSL = cachedWGSL(arangeWGSL);

/**
 * Create a 1-D tensor of evenly spaced values on the GPU.
 * No JS array allocation — values are computed directly by the GPU.
 */
export function arange(end: number, start = 0, step = 1): WebGPUTensor {
  const numElements = Math.max(0, Math.ceil((end - start) / step));
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const paramsData = new Uint32Array(3);
  paramsData[0] = numElements;
  new Float32Array(paramsData.buffer, 4, 1)[0] = start;
  new Float32Array(paramsData.buffer, 8, 1)[0] = step;
  return createTensor(
    [numElements],
    dispatchCreationKernel(
      "arange_tile",
      getArangeWGSL(),
      numElements,
      paramsData,
    ),
  );
}

const getTrilWGSL = cachedWGSL(() => triangularWGSL(false));
const getTriuWGSL = cachedWGSL(() => triangularWGSL(true));

/**
 * Triangular operation: zero elements above (tril) or below (triu) a diagonal.
 * Operates on the last 2 dimensions; supports arbitrary batch dimensions.
 */
function triangularOp(
  a: WebGPUTensor,
  k: number,
  upper: boolean,
): WebGPUTensor {
  const ctx = requireContext();
  if (a.shape.length < 2)
    throw new Error("tril/triu requires at least 2 dimensions");

  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const numElements = sizeOf(a.shape);

  // Ensure contiguous input
  const input = a.isContiguous ? a : getContiguous(a);

  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY } = compute2DDispatch(totalWorkgroups);

  const shader = upper ? getTriuWGSL() : getTrilWGSL();
  const pipeline = getPipeline(ctx, upper ? "triu_tile" : "tril_tile", shader);

  const sizeBytes = numElements * F32_BYTES;
  const alignedSize = alignBufferSize(sizeBytes);
  const usage =
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: alignedSize,
    usage,
  });

  // Params: [numElements as u32, H as u32, W as u32, k as i32]
  const paramsData = new Int32Array(4);
  new Uint32Array(paramsData.buffer, 0, 1)[0] = numElements;
  new Uint32Array(paramsData.buffer, 4, 1)[0] = H;
  new Uint32Array(paramsData.buffer, 8, 1)[0] = W;
  paramsData[3] = k;
  const paramsBuffer = createParamsBuffer(
    ctx.device,
    new Uint32Array(paramsData.buffer),
  );

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    input.buffer,
    outBuffer,
    paramsBuffer,
  ]);

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

const getRandWGSL = cachedWGSL(randWGSL);
const getRandnWGSL = cachedWGSL(randnWGSL);
const getBernoulliWGSL = cachedWGSL(bernoulliWGSL);

/** Pack seed params: [numElements, seed]. */
function seedParams(numElements: number, seed: number): Uint32Array {
  const p = new Uint32Array(2);
  p[0] = numElements;
  p[1] = seed >>> 0;
  return p;
}

export function rand(shape: number[], seed: number): WebGPUTensor {
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");
  return createTensor(
    shape,
    dispatchCreationKernel(
      "rand_tile",
      getRandWGSL(),
      numElements,
      seedParams(numElements, seed),
    ),
  );
}

export function randn(shape: number[], seed: number): WebGPUTensor {
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");
  // randn uses numElements/2 threads (each produces 2 values via Box-Muller)
  return createTensor(
    shape,
    dispatchCreationKernel(
      "randn_tile",
      getRandnWGSL(),
      numElements,
      seedParams(numElements, seed),
      Math.ceil(numElements / 2),
    ),
  );
}

export function bernoulli(
  shape: number[],
  p: number,
  seed: number,
): WebGPUTensor {
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");
  const paramsData = new Uint32Array(4);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  new Float32Array(paramsData.buffer, 8, 1)[0] = p;
  return createTensor(
    shape,
    dispatchCreationKernel(
      "bernoulli_tile",
      getBernoulliWGSL(),
      numElements,
      paramsData,
    ),
  );
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
    default:
      typedData = Float32Array.from(values);
      break;
  }

  // Arena fast path: use resolveOutputBuffer for stable buffer identity across steps
  if (getActiveArena()) {
    const buffer = resolveOutputBuffer(ctx.device, typedData.byteLength, []);
    profileApiCall("writeBuffer", () =>
      ctx.queue.writeBuffer(buffer, 0, typedData),
    );
    return createTensor(shape, buffer, undefined, 0, dtype);
  }
  const buffer = createBufferWithData(ctx.device, typedData, ctx.queue);
  return createTensor(shape, buffer, undefined, 0, dtype);
}
