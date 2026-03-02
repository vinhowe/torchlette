/**
 * Comparison ops: gt, lt, ge, le, eq, ne, and argmax/argmin.
 */

import type {
  BackendTensor,
  ArgReduceOptions,
} from "../../types";
import type { GPUBuffer } from "../gpu-types";
import { GPUBufferUsage, asGPUTensor } from "../gpu-types";
import {
  broadcastShapes,
  toIndexShape,
  sizeOf,
  contiguousStrides,
  computeEffectiveBroadcastStrides,
  compute2DDispatch,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { comparisonWGSL, argReduceWGSL } from "./ops-tile-ir";
import { requireContext } from "../gpu-context";
import { dispatchElementwise, dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor, createTrackedBuffer } from "../tensor";
import {
  params,
  createParamsBuffer,
  releaseParamsBuffer,
  cachedCreateBindGroup,
} from "../bind-group-cache";

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

  const code = comparisonWGSL(wgslOp, indexShape, aStrides, bStrides, aTensor.offset, bTensor.offset);

  const key = `${opName}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${aTensor.offset}:${bTensor.offset}`;

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [aTensor.buffer, bTensor.buffer],
    outputSizeBytes: outSize * 4,
    params: params(outSize),
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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const inputStrides = contiguousStrides(inputShape);
  const outStrides = contiguousStrides(outShape);

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
    compareOp, inputShape, inputStrides,
    outShape, outStrides, dim, dimSize, dimStride, inputToOutDim,
  );

  const pipeline = getPipeline(
    ctx,
    `${opName}:${inputShape.join(",")}:${dim}:${keepdim}`,
    code,
  );

  const paramsBuffer = createParamsBuffer(ctx.device, params(outSize, dimSize, dimStride));

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [tensor.buffer, outBuffer, paramsBuffer]);

  dispatchComputePass(pipeline, bindGroup, Math.ceil(outSize / WORKGROUP_SIZE));

  releaseParamsBuffer(paramsBuffer);

  return createTensor(outShape, outBuffer);
}
