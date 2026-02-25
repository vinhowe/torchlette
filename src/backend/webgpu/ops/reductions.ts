/**
 * Reduction ops: sum, max, mean, and supporting helpers.
 * Extracted from index.ts — purely structural refactoring.
 */

import { normalizeDim, type BackendTensor, type SumOptions, type MeanOptions, type MaxOptions } from "../../types";
import type { GPUBuffer, WebGPUTensor, WebGPUContext } from "../gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../gpu-types";
import {
  dtypeBytes,
  alignBufferSize,
  compute2DDispatch,
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
  createUniformBuffer,
  releaseUniformBuffer,
  params2,
} from "../bind-group-cache";
import { profileApiCall } from "../profiler";
import {
  sharedEncoder as sharedEncoderFlag,
  flushSharedEncoder,
  incrementSubmitCount,
} from "../shared-encoder";
import { getExpr as getExprFromRegistry, isUnaryOp as isUnaryOpFromRegistry } from "./registry";
import { contiguous } from "./views";

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

export function sum(a: BackendTensor, options?: SumOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = a as WebGPUTensor;
  let contiguousCopy: WebGPUTensor | null = null;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  // The sum kernels assume contiguous layout for index computation
  if (!tensor.isContiguous) {
    tensor = contiguous(tensor) as WebGPUTensor;
    contiguousCopy = tensor;
  }

  const inputShape = tensor.shape;

  // Handle full reduction (no dim specified or dim is null)
  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    // Full reduction - returns 0-d tensor (shape [])
    const result = sumFullReduction(ctx, tensor);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  // Dimension-wise reduction
  const dims = Array.isArray(dim) ? dim : [dim];

  // Normalize negative dimensions
  const rank = inputShape.length;
  const normalizedDims = dims.map((d) => normalizeDim(d, rank));

  // Validate dimensions
  for (const d of normalizedDims) {
    if (d < 0 || d >= rank) {
      throw new Error(`sum: dimension ${d} out of range`);
    }
  }

  // Compute output shape
  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) {
        outShape.push(1);
      }
    } else {
      outShape.push(inputShape[i]);
    }
  }

  // If output is scalar, handle specially
  if (outShape.length === 0) {
    const result = sumFullReduction(ctx, tensor);
    if (contiguousCopy) contiguousCopy.destroy();
    return result;
  }

  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  const { inputStrides, outStrides, reductionSize, inputToOutDim } =
    buildReductionMetadata(inputShape, normalizedDims, outShape, keepdim);

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const outShapeArray =
    outShape.length > 0
      ? `array<u32, ${outShape.length}>(${outShape.map((s) => `${s}u`).join(", ")})`
      : "";
  const outStridesArray =
    outStrides.length > 0
      ? `array<u32, ${outStrides.length}>(${outStrides.map((s) => `${s}u`).join(", ")})`
      : "";
  const reduceDimsArray = `array<u32, ${normalizedDims.length}>(${normalizedDims.map((d) => `${d}u`).join(", ")})`;
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d) => `${d}i`).join(", ")})`;

  // Choose between parallel tree reduction (large reductionSize) and sequential (small)
  const useParallelReduction = reductionSize > 64;

  // Common shader fragments for index computation
  const shaderPreamble = `
struct Params {
  outSize: u32,
  reductionSize: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const NUM_REDUCE_DIMS: u32 = ${normalizedDims.length}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const reduceDims = ${reduceDimsArray};
const inputToOutDim = ${inputToOutDimArray};

fn isReduceDim(d: u32) -> bool {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return true;
    }
  }
  return false;
}

fn getReduceDimIndex(d: u32) -> u32 {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return i;
    }
  }
  return 0u;
}

fn computeInputOffset(outIdx: u32, reduceIdx: u32) -> u32 {
  // Convert output index to output coordinates
  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${
    outShape.length > 0
      ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  `
      : ""
  }

  // Convert reduceIdx to coordinates in reduced dimensions
  var reduceCoords: array<u32, ${Math.max(normalizedDims.length, 1)}>;
  var rRemaining = reduceIdx;
  ${
    normalizedDims.length > 0
      ? normalizedDims
          .map(
            (_, i) => `
  {
    var rDimSize = 1u;
    for (var j = ${i + 1}u; j < NUM_REDUCE_DIMS; j = j + 1u) {
      rDimSize = rDimSize * inputShape[reduceDims[j]];
    }
    reduceCoords[${i}u] = rRemaining / rDimSize;
    rRemaining = rRemaining % rDimSize;
  }
  `,
          )
          .join("")
      : ""
  }

  // Build full input offset
  var inputOffset = 0u;
  for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
    var coord = 0u;
    if (isReduceDim(d)) {
      let rIdx = getReduceDimIndex(d);
      coord = reduceCoords[rIdx];
    } else {
      let outD = inputToOutDim[d];
      if (outD >= 0i) {
        coord = outCoords[u32(outD)];
      }
    }
    inputOffset = inputOffset + coord * inputStrides[d];
  }
  return inputOffset;
}
`;

  let code: string;
  let variant: string;
  let dispatchX: number;
  let dispatchY: number;

  if (useParallelReduction) {
    // Parallel tree reduction: one workgroup (256 threads) per output element
    const parDispatch = compute2DDispatch(outSize);
    const parUse2D = parDispatch.y > 1;
    dispatchX = parDispatch.x;
    dispatchY = parDispatch.y;
    variant = "par";

    code = shaderPreamble + `
var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let outIdx = ${parUse2D ? `wid.x + wid.y * ${parDispatch.x}u` : "wid.x"};
  if (outIdx >= params.outSize) {
    return;
  }
  let tid = lid.x;

  // Phase 1: Each thread sums a strided slice of the reduction dimension
  var local_sum = 0.0;
  for (var r = tid; r < params.reductionSize; r = r + ${WORKGROUP_SIZE}u) {
    local_sum = local_sum + input[computeInputOffset(outIdx, r)];
  }
  sdata[tid] = local_sum;
  workgroupBarrier();

  // Phase 2: Tree reduction in shared memory
  for (var s = ${WORKGROUP_SIZE >> 1}u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sdata[tid] = sdata[tid] + sdata[tid + s];
    }
    workgroupBarrier();
  }

  // Thread 0 writes the final sum
  if (tid == 0u) {
    out[outIdx] = sdata[0];
  }
}
`;
  } else {
    // Sequential reduction: each thread handles one output element
    const sumTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
    const sumDispatch = compute2DDispatch(sumTotalWG);
    const sumUse2D = sumDispatch.y > 1;
    const sumGridSizeX = sumDispatch.x * WORKGROUP_SIZE;
    dispatchX = sumDispatch.x;
    dispatchY = sumDispatch.y;
    variant = "seq";

    code = shaderPreamble + `
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = ${sumUse2D ? `gid.x + gid.y * ${sumGridSizeX}u` : "gid.x"};
  if (outIdx >= params.outSize) {
    return;
  }

  var total = 0.0;
  for (var reduceIdx = 0u; reduceIdx < params.reductionSize; reduceIdx = reduceIdx + 1u) {
    total = total + input[computeInputOffset(outIdx, reduceIdx)];
  }

  out[outIdx] = total;
}
`;
  }

  const pipeline = getPipeline(
    ctx,
    `sum:${variant}:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim}:${dispatchY > 1 ? "2d" : "1d"}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(outSize, reductionSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);

  releaseParamsBuffer(paramsBuffer);
  if (contiguousCopy) contiguousCopy.destroy();

  return createTensor(outShape, outBuffer);
}

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
  const getExprFn = getExprFromRegistry;
  const isUnaryOpFn = isUnaryOpFromRegistry;
  const ctx = requireContext();

  // All inputs must be contiguous and same shape
  const tensor0 = inputs[0] as WebGPUTensor;
  const inputShape = tensor0.shape;
  const rank = inputShape.length;

  const dim = sumOptions?.dim;
  const keepdim = sumOptions?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    // Full reduction with preamble
    return sumFullReductionWithPreamble(ctx, inputs, preambleOp);
  }

  const dims = Array.isArray(dim) ? dim : [dim];
  const normalizedDims = dims.map((d: number) => normalizeDim(d, rank));

  // Compute output shape
  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) outShape.push(1);
    } else {
      outShape.push(inputShape[i]);
    }
  }

  if (outShape.length === 0) {
    return sumFullReductionWithPreamble(ctx, inputs, preambleOp);
  }

  const outSize = outShape.reduce((acc: number, d: number) => acc * d, 1);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const { inputStrides, outStrides, reductionSize, inputToOutDim } =
    buildReductionMetadata(inputShape, normalizedDims, outShape, keepdim);

  const isUnary = isUnaryOpFn(preambleOp);
  const arity = isUnary ? 1 : 2;

  // Build input bindings
  const inputBindings = inputs.map((inp: BackendTensor, i: number) =>
    `@group(0) @binding(${i}) var<storage, read> input${i}: array<f32>;`
  ).join("\n");

  // Build preamble expression
  const inputExprs = Array.from({ length: arity }, (_, i) => `input${i}[inputOffset]`);
  const preambleExpr = getExprFn(preambleOp, inputExprs);

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s: number) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s: number) => `${s}u`).join(", ")})`;
  const outShapeArray = outShape.length > 0
    ? `array<u32, ${outShape.length}>(${outShape.map((s: number) => `${s}u`).join(", ")})` : "";
  const outStridesArray = outStrides.length > 0
    ? `array<u32, ${outStrides.length}>(${outStrides.map((s: number) => `${s}u`).join(", ")})` : "";
  const reduceDimsArray = `array<u32, ${normalizedDims.length}>(${normalizedDims.map((d: number) => `${d}u`).join(", ")})`;
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d: number) => `${d}i`).join(", ")})`;

  const spTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const spDispatch = compute2DDispatch(spTotalWG);
  const spUse2D = spDispatch.y > 1;
  const spGridSizeX = spDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  outSize: u32,
  reductionSize: u32,
};

${inputBindings}
@group(0) @binding(${arity}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${arity + 1}) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const NUM_REDUCE_DIMS: u32 = ${normalizedDims.length}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const reduceDims = ${reduceDimsArray};
const inputToOutDim = ${inputToOutDimArray};

fn isReduceDim(d: u32) -> bool {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) { return true; }
  }
  return false;
}

fn getReduceDimIndex(d: u32) -> u32 {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) { return i; }
  }
  return 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = ${spUse2D ? `gid.x + gid.y * ${spGridSizeX}u` : "gid.x"};
  if (outIdx >= params.outSize) { return; }

  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${outShape.length > 0 ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  ` : ""}

  var total = 0.0;
  for (var reduceIdx = 0u; reduceIdx < params.reductionSize; reduceIdx = reduceIdx + 1u) {
    var reduceCoords: array<u32, ${Math.max(normalizedDims.length, 1)}>;
    var rRemaining = reduceIdx;
    ${normalizedDims.map((_: number, i: number) => `
    {
      var rDimSize = 1u;
      for (var j = ${i + 1}u; j < NUM_REDUCE_DIMS; j = j + 1u) {
        rDimSize = rDimSize * inputShape[reduceDims[j]];
      }
      reduceCoords[${i}u] = rRemaining / rDimSize;
      rRemaining = rRemaining % rDimSize;
    }
    `).join("")}

    var inputOffset = 0u;
    for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
      var coord = 0u;
      if (isReduceDim(d)) {
        let rIdx = getReduceDimIndex(d);
        coord = reduceCoords[rIdx];
      } else {
        let outD = inputToOutDim[d];
        if (outD >= 0i) { coord = outCoords[u32(outD)]; }
      }
      inputOffset = inputOffset + coord * inputStrides[d];
    }

    total = total + ${preambleExpr};
  }

  out[outIdx] = total;
}
`;

  const cacheKey = `sumPreamble:${preambleOp}:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim}:${spUse2D ? "2d" : "1d"}`;
  const pipeline = getPipeline(ctx, cacheKey, code);
  const paramsBuffer = createParamsBuffer(ctx.device, params2(outSize, reductionSize));

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < arity; i++) {
    bgBuffers.push((inputs[i] as WebGPUTensor).buffer);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(paramsBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

  dispatchComputePass(pipeline, bindGroup, spDispatch.x, spDispatch.y);
  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}

/**
 * Full reduction sum with fused elementwise preamble.
 */
function sumFullReductionWithPreamble(
  ctx: WebGPUContext,
  inputs: BackendTensor[],
  preambleOp: string,
): WebGPUTensor {
  const getExprFn = getExprFromRegistry;
  const isUnaryOpFn = isUnaryOpFromRegistry;
  const tensor0 = inputs[0] as WebGPUTensor;
  const inputSize = tensor0.size;

  const isUnary = isUnaryOpFn(preambleOp);
  const arity = isUnary ? 1 : 2;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const inputBindings = inputs.slice(0, arity).map((_: BackendTensor, i: number) =>
    `@group(0) @binding(${i}) var<storage, read> input${i}: array<f32>;`
  ).join("\n");
  const inputExprs = Array.from({ length: arity }, (_, i) => `input${i}[i]`);
  const preambleExpr = getExprFn(preambleOp, inputExprs);

  const code = `
struct Params {
  size: u32,
};

${inputBindings}
@group(0) @binding(${arity}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${arity + 1}) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.size; i = i + 1u) {
    sum = sum + ${preambleExpr};
  }
  out[0] = sum;
}
`;

  const cacheKey = `sumFullPreamble:${preambleOp}:${inputSize}`;
  const pipeline = getPipeline(ctx, cacheKey, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < arity; i++) {
    bgBuffers.push((inputs[i] as WebGPUTensor).buffer);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(uniformBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

  dispatchComputePass(pipeline, bindGroup, 1);
  releaseUniformBuffer(uniformBuffer);

  return createTensor([], outBuffer);
}

// Helper for full reduction to scalar
async function sumFullReductionAsync(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): Promise<number> {
  const inputSize = tensor.size;

  // Simple sequential reduction for now
  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let idx = gid.x;
  let localIdx = lid.x;

  // Each thread loads and sums its elements
  var sum = 0.0;
  var i = idx;
  while (i < params.size) {
    sum = sum + input[i];
    i = i + ${WORKGROUP_SIZE}u * ${Math.ceil(inputSize / WORKGROUP_SIZE)}u;
  }
  shared[localIdx] = sum;

  workgroupBarrier();

  // Parallel reduction in shared memory
  for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localIdx < stride) {
      shared[localIdx] = shared[localIdx] + shared[localIdx + stride];
    }
    workgroupBarrier();
  }

  // First thread writes result
  if (localIdx == 0u) {
    out[wid.x] = shared[0];
  }
}
`;

  // For simplicity, do a two-pass reduction
  const numWorkgroups = Math.ceil(inputSize / WORKGROUP_SIZE);
  const intermediateBuffer = createTrackedBuffer(ctx.device, {
    size: Math.max(numWorkgroups * 4, 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const pipeline = getPipeline(ctx, `sumFull:${inputSize}`, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, intermediateBuffer, uniformBuffer]);

  // Flush shared encoder before sync readback — we need to submit all
  // prior work and then do a standalone encoder for the readback.
  if (sharedEncoderFlag) {
    flushSharedEncoder();
  }

  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(numWorkgroups);
  pass.end();

  // Read back intermediate results and sum on CPU (for small number of workgroups)
  const readBuffer = createTrackedBuffer(ctx.device, {
    size: numWorkgroups * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  encoder.copyBufferToBuffer(
    intermediateBuffer,
    0,
    readBuffer,
    0,
    numWorkgroups * 4,
  );
  profileApiCall("queue.submit", () => ctx.queue.submit([encoder.finish()]));
  incrementSubmitCount();

  await readBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readBuffer.getMappedRange());
  let total = 0;
  for (let i = 0; i < numWorkgroups; i++) {
    total += data[i];
  }
  readBuffer.unmap();

  // Destroy temporary buffers to prevent memory leaks
  bufferPool.deferredDestroy(intermediateBuffer, intermediateBuffer.size ?? numWorkgroups * 4);
  bufferPool.deferredDestroy(readBuffer, readBuffer.size ?? numWorkgroups * 4);
  releaseUniformBuffer(uniformBuffer);

  return total;
}

/**
 * Full reduction sum - returns a 0-d tensor (shape []) with the sum.
 * Use item() to extract the scalar value asynchronously.
 */
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

  // Create output buffer with single element
  const outBuffer = resolveOutputBuffer(ctx.device, 4, [tensor.buffer]);

  // Use a simple sequential sum shader
  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.size; i = i + 1u) {
    sum = sum + input[i];
  }
  out[0] = sum;
}
`;

  const pipeline = getPipeline(ctx, `sumFullSeq:${inputSize}`, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, 1);
  releaseUniformBuffer(uniformBuffer);

  // Return 0-d tensor (shape [])
  return createTensor([], outBuffer);
}

/**
 * Chunked full reduction sum for tensors exceeding maxStorageBufferBindingSize.
 * Computes partial sums per chunk, then sums the partials.
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

  // Shader: sequential sum of a chunk, output to out[chunkIdx]
  const code = `
struct Params {
  chunkSize: u32,
  chunkIdx: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.chunkSize; i = i + 1u) {
    sum = sum + input[i];
  }
  out[params.chunkIdx] = sum;
}
`;

  const pipeline = getPipeline(ctx, `sumFullChunked`, code);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, totalElements);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    const paramsBuffer = createParamsBuffer(ctx.device, params2(chunkSize, chunk));

    const bindGroup = profiledCreateBindGroup(ctx.device,{
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

  // Now sum the partials (small buffer, fits in one binding)
  if (numChunks === 1) {
    return createTensor([], partialsBuffer);
  }

  // Final reduction of partials
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const finalCode = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var sum = 0.0;
  for (var i = 0u; i < params.size; i = i + 1u) {
    sum = sum + input[i];
  }
  out[0] = sum;
}
`;

  const finalPipeline = getPipeline(ctx, `sumFullSeq:${numChunks}`, finalCode);
  const finalParams = createUniformBuffer(ctx.device, numChunks);

  const finalBindGroup = cachedCreateBindGroup(ctx.device, finalPipeline, [partialsBuffer, outBuffer, finalParams]);

  dispatchComputePass(finalPipeline, finalBindGroup, 1);
  releaseUniformBuffer(finalParams);

  // Destroy the intermediate partials buffer — it's been consumed by the final reduction
  bufferPool.deferredDestroy(partialsBuffer, alignBufferSize(numChunks * 4));

  return createTensor([], outBuffer);
}

/**
 * Full reduction max - returns a 0-d tensor (shape []) with the maximum.
 */
function maxFullReduction(
  ctx: WebGPUContext,
  tensor: WebGPUTensor,
): WebGPUTensor {
  const inputSize = tensor.size;

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
  var maxVal = input[0];
  for (var i = 1u; i < params.size; i = i + 1u) {
    maxVal = max(maxVal, input[i]);
  }
  out[0] = maxVal;
}
`;

  const pipeline = getPipeline(ctx, `maxFullSeq:${inputSize}`, code);
  const uniformBuffer = createUniformBuffer(ctx.device, inputSize);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, uniformBuffer]);

  dispatchComputePass(pipeline, bindGroup, 1);
  releaseUniformBuffer(uniformBuffer);

  return createTensor([], outBuffer);
}

export function max(a: BackendTensor, options?: MaxOptions): BackendTensor {
  const ctx = requireContext();
  let tensor = a as WebGPUTensor;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  if (!tensor.isContiguous) {
    tensor = contiguous(tensor) as WebGPUTensor;
  }

  const inputShape = tensor.shape;

  const dim = options?.dim;
  const keepdim = options?.keepdim ?? false;

  if (dim === undefined || dim === null) {
    return maxFullReduction(ctx, tensor);
  }

  const dims = Array.isArray(dim) ? dim : [dim];
  const rank = inputShape.length;
  const normalizedDims = dims.map((d) => normalizeDim(d, rank));

  for (const d of normalizedDims) {
    if (d < 0 || d >= rank) {
      throw new Error(`max: dimension ${d} out of range`);
    }
  }

  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) {
        outShape.push(1);
      }
    } else {
      outShape.push(inputShape[i]);
    }
  }

  if (outShape.length === 0) {
    return maxFullReduction(ctx, tensor);
  }

  const outSize = outShape.reduce((acc, d) => acc * d, 1);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [tensor.buffer]);

  const { inputStrides, outStrides, reductionSize, inputToOutDim } =
    buildReductionMetadata(inputShape, normalizedDims, outShape, keepdim);

  const inputShapeArray = `array<u32, ${rank}>(${inputShape.map((s) => `${s}u`).join(", ")})`;
  const inputStridesArray = `array<u32, ${rank}>(${inputStrides.map((s) => `${s}u`).join(", ")})`;
  const outShapeArray =
    outShape.length > 0
      ? `array<u32, ${outShape.length}>(${outShape.map((s) => `${s}u`).join(", ")})`
      : "";
  const outStridesArray =
    outStrides.length > 0
      ? `array<u32, ${outStrides.length}>(${outStrides.map((s) => `${s}u`).join(", ")})`
      : "";
  const reduceDimsArray = `array<u32, ${normalizedDims.length}>(${normalizedDims.map((d) => `${d}u`).join(", ")})`;
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d) => `${d}i`).join(", ")})`;

  const maxTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const maxDispatch = compute2DDispatch(maxTotalWG);
  const maxUse2D = maxDispatch.y > 1;
  const maxGridSizeX = maxDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  outSize: u32,
  reductionSize: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const NUM_REDUCE_DIMS: u32 = ${normalizedDims.length}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const reduceDims = ${reduceDimsArray};
const inputToOutDim = ${inputToOutDimArray};

fn isReduceDim(d: u32) -> bool {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return true;
    }
  }
  return false;
}

fn getReduceDimIndex(d: u32) -> u32 {
  for (var i = 0u; i < NUM_REDUCE_DIMS; i = i + 1u) {
    if (reduceDims[i] == d) {
      return i;
    }
  }
  return 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = ${maxUse2D ? `gid.x + gid.y * ${maxGridSizeX}u` : "gid.x"};
  if (outIdx >= params.outSize) {
    return;
  }

  var outCoords: array<u32, ${Math.max(outShape.length, 1)}>;
  ${
    outShape.length > 0
      ? `
  var remaining = outIdx;
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    outCoords[d] = remaining / outStrides[d];
    remaining = remaining % outStrides[d];
  }
  `
      : ""
  }

  // Find max over all reduction indices
  var maxVal = -3.402823466e+38; // -FLT_MAX
  for (var reduceIdx = 0u; reduceIdx < params.reductionSize; reduceIdx = reduceIdx + 1u) {
    var reduceCoords: array<u32, ${Math.max(normalizedDims.length, 1)}>;
    var rRemaining = reduceIdx;
    ${
      normalizedDims.length > 0
        ? normalizedDims
            .map(
              (_, i) => `
    {
      var rDimSize = 1u;
      for (var j = ${i + 1}u; j < NUM_REDUCE_DIMS; j = j + 1u) {
        rDimSize = rDimSize * inputShape[reduceDims[j]];
      }
      reduceCoords[${i}u] = rRemaining / rDimSize;
      rRemaining = rRemaining % rDimSize;
    }
    `,
            )
            .join("")
        : ""
    }

    var inputOffset = 0u;
    for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
      var coord = 0u;
      if (isReduceDim(d)) {
        let rIdx = getReduceDimIndex(d);
        coord = reduceCoords[rIdx];
      } else {
        let outD = inputToOutDim[d];
        if (outD >= 0i) {
          coord = outCoords[u32(outD)];
        }
      }
      inputOffset = inputOffset + coord * inputStrides[d];
    }

    maxVal = max(maxVal, input[inputOffset]);
  }

  out[outIdx] = maxVal;
}
`;

  const pipeline = getPipeline(
    ctx,
    `max:${inputShape.join(",")}:${normalizedDims.join(",")}:${keepdim}:${maxUse2D ? "2d" : "1d"}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params2(outSize, reductionSize));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, maxDispatch.x, maxDispatch.y);

  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}

export function mean(a: BackendTensor, options?: MeanOptions): BackendTensor {
  let tensor = a as WebGPUTensor;
  let contiguousCopy: WebGPUTensor | null = null;

  // Must materialize non-contiguous tensors first (e.g., expanded views)
  // Mean uses sum internally which requires contiguous layout
  if (!tensor.isContiguous) {
    tensor = contiguous(tensor) as WebGPUTensor;
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
  const sumTensor = sum(a, options) as WebGPUTensor;

  // Divide by count
  const ctx = requireContext();
  const outSize = sumTensor.size;

  const outBuffer = resolveOutputBuffer(ctx.device, outSize * 4, [sumTensor.buffer]);

  const meanTotalWG = Math.ceil(outSize / WORKGROUP_SIZE);
  const meanDispatch = compute2DDispatch(meanTotalWG);
  const meanUse2D = meanDispatch.y > 1;
  const meanGridSizeX = meanDispatch.x * WORKGROUP_SIZE;

  const code = `
struct Params {
  size: u32,
  count: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${meanUse2D ? `gid.x + gid.y * ${meanGridSizeX}u` : "gid.x"};
  if (idx >= params.size) {
    return;
  }
  out[idx] = input[idx] / params.count;
}
`;

  const pipeline = getPipeline(ctx, `meanDiv:${outSize}:${count}:${meanUse2D ? "2d" : "1d"}`, code);

  // Pack mixed u32 + f32 params into a single Uint32Array
  const meanParamsData = new ArrayBuffer(8);
  new Uint32Array(meanParamsData, 0, 1)[0] = outSize;
  new Float32Array(meanParamsData, 4, 1)[0] = count;
  const paramsBuffer = createParamsBuffer(ctx.device, new Uint32Array(meanParamsData));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [sumTensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, meanDispatch.x, meanDispatch.y);

  releaseParamsBuffer(paramsBuffer);

  // Destroy intermediate sum tensor — its buffer was only needed as input to the
  // division kernel above. Without this, the 256B buffer leaks every mean() call.
  sumTensor.destroy();

  // Destroy contiguous copy if one was created
  if (contiguousCopy) {
    contiguousCopy.destroy();
  }

  return createTensor(sumTensor.shape, outBuffer);
}
