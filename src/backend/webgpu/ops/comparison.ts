/**
 * Comparison ops: gt, lt, ge, le, eq, ne, and argmax/argmin.
 * Extracted from index.ts — purely structural refactoring.
 */

import type {
  BackendTensor,
  ArgReduceOptions,
} from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { GPUBufferUsage } from "../gpu-types";
import {
  broadcastShapes,
  toIndexShape,
  sizeOf,
  computeEffectiveBroadcastStrides,
  buildBroadcastIndexing,
  compute2DDispatch,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchElementwise, dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import {
  params1,
  params3,
  createParamsBuffer,
  releaseParamsBuffer,
  cachedCreateBindGroup,
} from "../bind-group-cache";

export function comparisonOp(
  opName: string,
  wgslOp: string,
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const aTensor = a as WebGPUTensor;
  const bTensor = b as WebGPUTensor;

  const outShape = broadcastShapes(aTensor.shape, bTensor.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);

  const aStrides = computeEffectiveBroadcastStrides(aTensor, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(bTensor, indexShape);

  const indexing = buildBroadcastIndexing(indexShape, [aStrides, bStrides]);

  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${dispatch.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

${indexing.declarations}
const A_OFFSET: u32 = ${aTensor.offset}u;
const B_OFFSET: u32 = ${bTensor.offset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
  let aVal = a[offset0 + A_OFFSET];
  let bVal = b[offset1 + B_OFFSET];
  out[idx] = select(0.0, 1.0, aVal ${wgslOp} bVal);
}
`;

  const key = `${opName}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${aTensor.offset}:${bTensor.offset}:${use2D ? dispatch.gridSizeX : "1d"}`;

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [aTensor.buffer, bTensor.buffer],
    outputSizeBytes: outSize * 4,
    params: params1(outSize),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  return createTensor(outShape, outBuffer);
}

export function gt(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("gt", ">", a, b, options);
}

export function lt(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("lt", "<", a, b, options);
}

export function ge(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("ge", ">=", a, b, options);
}

export function le(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("le", "<=", a, b, options);
}

export function eq(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("eq", "==", a, b, options);
}

export function ne(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return comparisonOp("ne", "!=", a, b, options);
}

// ============================================================================
// ArgMax/ArgMin - return indices of max/min values
// ============================================================================

export function argmax(a: BackendTensor, options: ArgReduceOptions): BackendTensor {
  return argReduceOp("argmax", ">", a, options);
}

export function argmin(a: BackendTensor, options: ArgReduceOptions): BackendTensor {
  return argReduceOp("argmin", "<", a, options);
}

export function argReduceOp(
  opName: string,
  compareOp: string,
  a: BackendTensor,
  options: ArgReduceOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensor = a as WebGPUTensor;
  const inputShape = tensor.shape;
  const rank = inputShape.length;

  const dim = options.dim < 0 ? options.dim + rank : options.dim;
  if (dim < 0 || dim >= rank) {
    throw new Error(
      `${opName}: dim ${options.dim} out of range for tensor of rank ${rank}`,
    );
  }
  const keepdim = options.keepdim ?? false;

  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (i === dim) {
      if (keepdim) {
        outShape.push(1);
      }
    } else {
      outShape.push(inputShape[i]);
    }
  }

  const outSize = outShape.reduce((acc, d) => acc * d, 1) || 1;
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const inputStrides: number[] = [];
  for (let i = 0; i < rank; i++) {
    let stride = 1;
    for (let j = i + 1; j < rank; j++) {
      stride *= inputShape[j];
    }
    inputStrides.push(stride);
  }

  const outStrides: number[] = [];
  for (let i = 0; i < outShape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < outShape.length; j++) {
      stride *= outShape[j];
    }
    outStrides.push(stride);
  }

  const dimSize = inputShape[dim];
  const dimStride = inputStrides[dim];

  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (i === dim) {
      if (keepdim) {
        inputToOutDim.push(outDimIdx);
        outDimIdx++;
      } else {
        inputToOutDim.push(-1);
      }
    } else {
      inputToOutDim.push(outDimIdx);
      outDimIdx++;
    }
  }

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
  const inputToOutDimArray = `array<i32, ${rank}>(${inputToOutDim.map((d) => `${d}i`).join(", ")})`;

  const initVal = compareOp === ">" ? "-3.402823466e+38" : "3.402823466e+38";

  const code = `
struct Params {
  outSize: u32,
  dimSize: u32,
  dimStride: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const INPUT_RANK: u32 = ${rank}u;
const OUT_RANK: u32 = ${outShape.length}u;
const REDUCE_DIM: u32 = ${dim}u;
const inputShape = ${inputShapeArray};
const inputStrides = ${inputStridesArray};
${outShape.length > 0 ? `const outShape = ${outShapeArray};` : ""}
${outStrides.length > 0 ? `const outStrides = ${outStridesArray};` : ""}
const inputToOutDim = ${inputToOutDimArray};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = gid.x;
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

  // Compute base offset in input (with reduce dim = 0)
  var baseOffset = 0u;
  for (var d = 0u; d < INPUT_RANK; d = d + 1u) {
    if (d != REDUCE_DIM) {
      let outD = inputToOutDim[d];
      if (outD >= 0i) {
        baseOffset = baseOffset + outCoords[u32(outD)] * inputStrides[d];
      }
    }
  }

  // Find argmax/argmin along the reduce dimension
  var bestVal = ${initVal};
  var bestIdx = 0u;
  for (var i = 0u; i < params.dimSize; i = i + 1u) {
    let val = input[baseOffset + i * params.dimStride];
    if (val ${compareOp} bestVal) {
      bestVal = val;
      bestIdx = i;
    }
  }

  out[outIdx] = f32(bestIdx);
}
`;

  const pipeline = getPipeline(
    ctx,
    `${opName}:${inputShape.join(",")}:${dim}:${keepdim}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params3(outSize, dimSize, dimStride));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, Math.ceil(outSize / WORKGROUP_SIZE));

  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}
