/**
 * Tensor creation ops: tensorFromArray, zeros, full, arange, tril/triu, rand/randn/bernoulli.
 * Extracted from index.ts — purely structural refactoring.
 */

import type { WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import { sizeOf, WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM, compute2DDispatch, dtypeBytes, alignBufferSize } from "../shape-utils";
import { requireContext, f32ArrayToF16Array } from "../gpu-context";
import { dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer, createBufferWithData } from "../tensor";
import { resolveOutputBuffer, activeArena, arenaBufferSet } from "../buffer-arena";
import { cachedCreateBindGroup, createParamsBuffer, releaseParamsBuffer } from "../bind-group-cache";
import { profileApiCall } from "../profiler";
import { bufferPool } from "../buffer-pool";
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
  if (activeArena) {
    const buffer = resolveOutputBuffer(ctx.device, f32data.byteLength, []);
    profileApiCall("writeBuffer", () => ctx.queue.writeBuffer(buffer, 0, f32data));
    return createTensor(shape, buffer, undefined, 0, "f32");
  }
  const buffer = createBufferWithData(
    ctx.device,
    f32data,
    ctx.queue,
  );
  return createTensor(shape, buffer, undefined, 0, "f32");
}

/**
 * Generate WGSL shader for GPU-side fill.
 * Fills an output buffer with a constant value using a compute shader.
 */
export function fillShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
struct Params {
  size: u32,
  value: f32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  out[idx] = params.value;
}
`;
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

  return createTensor(shape, buffer, undefined, 0, "f32");
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
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `fill_${gridSizeX}`;
  const shader = fillShader(gridSizeX);
  const pipeline = getPipeline(ctx, shaderKey, shader);

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

  return createTensor(shape, outBuffer, undefined, 0, "f32");
}

/**
 * Generate WGSL shader for GPU-side arange.
 * Fills output with start + idx * step.
 */
export function arangeShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
struct Params {
  size: u32,
  start: f32,
  step: f32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  out[idx] = params.start + f32(idx) * params.step;
}
`;
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
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `arange_${gridSizeX}`;
  const shader = arangeShader(gridSizeX);
  const pipeline = getPipeline(ctx, shaderKey, shader);

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

  return createTensor([numElements], outBuffer, undefined, 0, "f32");
}

/**
 * Generate WGSL shader for tril/triu.
 * A single shader template parameterized by upper (inlined at compile time).
 * For tril: zero where col > row + k
 * For triu: zero where col < row + k
 */
export function triangularShader(gridSizeX: number, upper: boolean): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  // Inlined condition: tril zeros above k-th diagonal, triu zeros below
  const zeroCondition = upper
    ? `col < row + k` // triu: zero below
    : `col > row + k`; // tril: zero above
  return `
struct Params {
  num_elements: u32,
  H: u32,
  W: u32,
  k: i32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.num_elements) {
    return;
  }
  let row = i32((idx / params.W) % params.H);
  let col = i32(idx % params.W);
  let k = params.k;
  if (${zeroCondition}) {
    output[idx] = 0.0;
  } else {
    output[idx] = input[idx];
  }
}
`;
}

/**
 * Triangular operation: zero elements above (tril) or below (triu) a diagonal.
 * Operates on the last 2 dimensions; supports arbitrary batch dimensions.
 */
export function triangularOp(a: WebGPUTensor, k: number, upper: boolean): WebGPUTensor {
  const ctx = requireContext();
  if (a.shape.length < 2) throw new Error("tril/triu requires at least 2 dimensions");

  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const numElements = sizeOf(a.shape);

  // Ensure contiguous input
  const input = a.isContiguous ? a : getContiguous(a);

  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const tag = upper ? "triu" : "tril";
  const shaderKey = `${tag}_${gridSizeX}`;
  const shader = triangularShader(gridSizeX, upper);
  const pipeline = getPipeline(ctx, shaderKey, shader);

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
  if (input !== a) {
    bufferPool.decRef((input as WebGPUTensor).buffer);
    bufferPool.deferredDestroy((input as WebGPUTensor).buffer, numElements * dtypeBytes(a.dtype));
  }

  return createTensor(a.shape.slice(), outBuffer, undefined, 0, a.dtype);
}

export function tril(a: WebGPUTensor, k = 0): WebGPUTensor {
  return triangularOp(a, k, false);
}

export function triu(a: WebGPUTensor, k = 0): WebGPUTensor {
  return triangularOp(a, k, true);
}

// ============================================================================
// GPU RNG (PCG32)
// ============================================================================

export const PCG32_WGSL = `
fn pcg32(state: ptr<function, u32>) -> u32 {
  let old = *state;
  *state = old * 747796405u + 2891336453u;
  let word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
  return (word >> 22u) ^ word;
}

fn pcg32_init(seed: u32, seq: u32) -> u32 {
  var state = 0u;
  state = state * 747796405u + ((seq << 1u) | 1u);
  state = state + seed;
  state = state * 747796405u + ((seq << 1u) | 1u);
  return state;
}
`;

export function randShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
${PCG32_WGSL}

struct Params {
  size: u32,
  seed: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) { return; }
  var state = pcg32_init(params.seed, idx);
  let bits = pcg32(&state);
  out[idx] = f32(bits) / 4294967296.0;
}
`;
}

export function randnShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
${PCG32_WGSL}

const PI: f32 = 3.14159265358979323846;

struct Params {
  size: u32,
  seed: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  // Process pairs: thread idx handles element idx*2 and idx*2+1
  let outIdx = idx * 2u;
  if (outIdx >= params.size) { return; }
  var state = pcg32_init(params.seed, idx);
  let bits1 = pcg32(&state);
  let bits2 = pcg32(&state);
  // Map to (0, 1] to avoid log(0)
  let u1 = (f32(bits1) + 1.0) / 4294967297.0;
  let u2 = f32(bits2) / 4294967296.0;
  let r = sqrt(-2.0 * log(u1));
  let theta = 2.0 * PI * u2;
  out[outIdx] = r * cos(theta);
  if (outIdx + 1u < params.size) {
    out[outIdx + 1u] = r * sin(theta);
  }
}
`;
}

export function bernoulliShader(gridSizeX: number): string {
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `
${PCG32_WGSL}

struct Params {
  size: u32,
  seed: u32,
  prob: f32,
};

@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) { return; }
  var state = pcg32_init(params.seed, idx);
  let bits = pcg32(&state);
  let u = f32(bits) / 4294967296.0;
  out[idx] = select(0.0, 1.0, u < params.prob);
}
`;
}

export function rand(shape: number[], seed: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const sizeBytes = numElements * 4;
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `rand_${gridSizeX}`;
  const pipeline = getPipeline(ctx, shaderKey, randShader(gridSizeX));

  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);
  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer, undefined, 0, "f32");
}

export function randn(shape: number[], seed: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const sizeBytes = numElements * 4;
  // randn processes pairs, so dispatch half the threads (rounded up)
  const numThreads = Math.ceil(numElements / 2);
  const totalWorkgroups = Math.ceil(numThreads / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `randn_${gridSizeX}`;
  const pipeline = getPipeline(ctx, shaderKey, randnShader(gridSizeX));

  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  const paramsData = new Uint32Array(2);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);
  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer, undefined, 0, "f32");
}

export function bernoulli(shape: number[], p: number, seed: number): WebGPUTensor {
  const ctx = requireContext();
  const numElements = sizeOf(shape);
  if (numElements === 0) throw new Error("webgpu tensors cannot be empty yet");

  const sizeBytes = numElements * 4;
  const totalWorkgroups = Math.ceil(numElements / WORKGROUP_SIZE);
  const { x: dispatchX, y: dispatchY, gridSizeX } = compute2DDispatch(totalWorkgroups);

  const shaderKey = `bernoulli_${gridSizeX}`;
  const pipeline = getPipeline(ctx, shaderKey, bernoulliShader(gridSizeX));

  const outBuffer = resolveOutputBuffer(ctx.device, sizeBytes, []);

  // Params: [size as u32, seed as u32, prob as f32]  — 3 words, padded to 4 for alignment
  const paramsData = new Uint32Array(4);
  paramsData[0] = numElements;
  paramsData[1] = seed >>> 0;
  new Float32Array(paramsData.buffer, 8, 1)[0] = p;
  const paramsBuffer = createParamsBuffer(ctx.device, paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [outBuffer, paramsBuffer]);
  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(shape, outBuffer, undefined, 0, "f32");
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
  if (activeArena) {
    const buffer = resolveOutputBuffer(ctx.device, typedData.byteLength, []);
    profileApiCall("writeBuffer", () => ctx.queue.writeBuffer(buffer, 0, typedData));
    return createTensor(shape, buffer, undefined, 0, dtype);
  }
  const buffer = createBufferWithData(ctx.device, typedData, ctx.queue);
  return createTensor(shape, buffer, undefined, 0, dtype);
}
