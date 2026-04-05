/**
 * Comparison ops: gt, lt, ge, le, eq, ne, and argmax/argmin.
 */

import type { ArgReduceOptions, BackendTensor } from "../../types";
import {
  cachedCreateBindGroup,
  createParamsBuffer,
  params,
  releaseParamsBuffer,
} from "../bind-group-cache";
import {
  dispatchComputePass,
  dispatchElementwise,
  getPipeline,
} from "../dispatch";
import { requireContext } from "../gpu-context";
import type { GPUBuffer } from "../gpu-types";
import { asGPUTensor, GPUBufferUsage } from "../gpu-types";
import {
  broadcastShapes,
  compute2DDispatch,
  computeEffectiveBroadcastStrides,
  contiguousStrides,
  sizeOf,
  toIndexShape,
  WORKGROUP_SIZE,
  F32_BYTES,
} from "../shape-utils";
import { createTensor, createTrackedBuffer } from "../tensor";
import { argReduceWGSL, comparisonWGSL } from "./ops-tile-ir";

function comparisonOp(
  opName: string,
  wgslOp: string,
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const aTensor = asGPUTensor(a);
  const bTensor = asGPUTensor(b);

  const outShape = broadcastShapes(aTensor.shape, bTensor.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);

  const aStrides = computeEffectiveBroadcastStrides(aTensor, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(bTensor, indexShape);

  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);

  const code = comparisonWGSL(
    wgslOp,
    indexShape,
    aStrides,
    bStrides,
    aTensor.offset,
    bTensor.offset,
    aTensor.dtype,
    bTensor.dtype,
  );

  const key = `${opName}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${aTensor.offset}:${bTensor.offset}:${aTensor.dtype}:${bTensor.dtype}`;

  const outBuffer = dispatchElementwise({
    key,
    shader: code,
    inputs: [aTensor.buffer, bTensor.buffer],
    outputSizeBytes: outSize * 4,
    params: params(outSize),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x,
    dispatchY: dispatch.y,
  });

  return createTensor(outShape, outBuffer);
}

type ComparisonFn = (
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
) => BackendTensor;
const cmp =
  (name: string, op: string): ComparisonFn =>
  (a, b, options) =>
    comparisonOp(name, op, a, b, options);

export const gt = cmp("gt", ">");
export const lt = cmp("lt", "<");
export const ge = cmp("ge", ">=");
export const le = cmp("le", "<=");
export const eq = cmp("eq", "==");
export const ne = cmp("ne", "!=");

// ============================================================================
// ArgMax/ArgMin - return indices of max/min values
// ============================================================================

type ArgReduceFn = (
  a: BackendTensor,
  options: ArgReduceOptions,
) => BackendTensor;
const argReduce =
  (name: string, op: string): ArgReduceFn =>
  (a, options) =>
    argReduceOp(name, op, a, options);

export const argmax = argReduce("argmax", ">");
export const argmin = argReduce("argmin", "<");

function argReduceOp(
  opName: string,
  compareOp: string,
  a: BackendTensor,
  options: ArgReduceOptions,
): BackendTensor {
  const ctx = requireContext();
  const tensor = asGPUTensor(a);
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

  const outSize = sizeOf(outShape) || 1;
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: outSize * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  const inputStrides = contiguousStrides(inputShape);

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

  const code = argReduceWGSL(
    compareOp,
    inputShape,
    inputStrides,
    outShape,
    dim,
    inputToOutDim,
  );

  const pipeline = getPipeline(
    ctx,
    `${opName}:${inputShape.join(",")}:${dim}:${keepdim}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(
    ctx.device,
    params(outSize, dimSize, dimStride),
  );

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    tensor.buffer,
    outBuffer,
    paramsBuffer,
  ]);

  dispatchComputePass(pipeline, bindGroup, Math.ceil(outSize / WORKGROUP_SIZE));

  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}
