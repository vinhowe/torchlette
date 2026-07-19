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
import { realizeArgReduceWgsl as argReduceWGSL } from "../../../schedule/reduction-skeleton";
import { createTensor, createTrackedBuffer } from "../tensor";
import { comparisonWGSL } from "./ops-tile-ir";
import { contiguous } from "./views";

function comparisonOp(
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

  const outBuffer = dispatchElementwise({
    // Key IS the WGSL (single-source-at-seams; tile-dispatch canonical).
    key: code,
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
  (op: string): ComparisonFn =>
  (a, b, options) =>
    comparisonOp(op, a, b, options);

export const gt = cmp(">");
export const lt = cmp("<");
export const ge = cmp(">=");
export const le = cmp("<=");
export const eq = cmp("==");
export const ne = cmp("!=");

// ============================================================================
// ArgMax/ArgMin - return indices of max/min values
// ============================================================================

type ArgReduceFn = (
  a: BackendTensor,
  options: ArgReduceOptions,
) => BackendTensor;
const argReduce =
  (name: string, op: ">" | "<"): ArgReduceFn =>
  (a, options) =>
    argReduceOp(name, op, a, options);

export const argmax = argReduce("argmax", ">");
export const argmin = argReduce("argmin", "<");

/**
 * Geometry + WGSL for ONE arg-reduce dispatch (argmax/argmin over a dim), the
 * SINGLE SOURCE both the imperative dispatcher (`argReduceOp`) and the stream
 * generator (`stream-generate.ts` `generateArgReduce`) derive from — mirroring
 * `planGatherDirect` / `planDimReductionDispatch`. Callers pass the CONTIGUOUS,
 * offset-0 input shape (the kernel derives its addressing from
 * `contiguousStrides(inputShape)` and binds flat from element 0; the contiguity
 * is enforced at the seam by each caller). The pipeline cache key IS the WGSL
 * text (`argReduceWGSL` bakes strides/outShape/inputToOutDim in, so a structural
 * key would drift — single-source-at-seams). `dim` must be normalized
 * non-negative.
 */
export function planArgReduceDispatch(
  compareOp: ">" | "<",
  inputShape: readonly number[],
  dim: number,
  keepdim: boolean,
): {
  key: string;
  shader: string;
  paramsData: Uint32Array;
  dispatchX: number;
  dispatchY: number;
  outShape: number[];
  outputBytes: number;
} {
  const rank = inputShape.length;
  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (i === dim) {
      if (keepdim) outShape.push(1);
    } else {
      outShape.push(inputShape[i]);
    }
  }
  const outSize = sizeOf(outShape) || 1;
  const inputStrides = contiguousStrides([...inputShape]);
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
    [...inputShape],
    inputStrides,
    outShape,
    dim,
    inputToOutDim,
  );
  return {
    key: code,
    shader: code,
    paramsData: params(outSize, dimSize, dimStride),
    dispatchX: Math.ceil(outSize / WORKGROUP_SIZE),
    dispatchY: 1,
    outShape,
    outputBytes: outSize * F32_BYTES,
  };
}

function argReduceOp(
  opName: string,
  compareOp: ">" | "<",
  a: BackendTensor,
  options: ArgReduceOptions,
): BackendTensor {
  const ctx = requireContext();
  let tensor = asGPUTensor(a);

  // Materialize non-raw-bindable inputs: the arg-reduce kernel derives its
  // input addressing from `contiguousStrides(inputShape)` and binds the buffer
  // flat from element 0 — so a STRIDED view (narrow) or an OFFSET view (a
  // last-position row out of a multi-token logits tensor) would be indexed with
  // the wrong strides AND from the wrong base, silently returning the wrong
  // index (measured on decode: argmax over a doubly-narrowed logits row gave 40
  // vs the correct 995). Mirror the sum/max reduction guard (reductions.ts,
  // task #58): force contiguous so the kernel's contiguous-stride assumption is
  // TRUE. Cheap — a decode reduction row is a single vector.
  let contiguousCopy: ReturnType<typeof asGPUTensor> | null = null;
  if (!tensor.isContiguous || (tensor.offset ?? 0) !== 0) {
    tensor = asGPUTensor(contiguous(tensor));
    contiguousCopy = tensor;
  }
  // Seam assertion (single source of truth): the kernel assumes a contiguous,
  // offset-0 reduction row. Assert it here rather than let a future non-raw-
  // bindable input mis-index silently (the exact failure this op just fixed).
  if (!tensor.isContiguous || (tensor.offset ?? 0) !== 0) {
    throw new Error(
      `${opName}: reduction input must be contiguous with offset 0 at the ` +
        `dispatch seam (kernel derives addressing from contiguousStrides); got ` +
        `isContiguous=${tensor.isContiguous}, offset=${tensor.offset ?? 0}.`,
    );
  }

  const inputShape = tensor.shape;
  const rank = inputShape.length;

  const dim = options.dim < 0 ? options.dim + rank : options.dim;
  if (dim < 0 || dim >= rank) {
    throw new Error(
      `${opName}: dim ${options.dim} out of range for tensor of rank ${rank}`,
    );
  }
  const keepdim = options.keepdim ?? false;

  // SINGLE SOURCE for geometry + WGSL (shared with the stream generator's
  // generateArgReduce). Key IS the WGSL (argReduceWGSL bakes strides/outShape/
  // inputToOutDim in, so a non-contiguous input with the same shape/dim/keepdim
  // would collide on a stale pipeline — single-source-at-seams).
  const plan = planArgReduceDispatch(compareOp, inputShape, dim, keepdim);

  const outBuffer = createTrackedBuffer(ctx.device, {
    size: plan.outputBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  const pipeline = getPipeline(ctx, plan.key, plan.shader);

  const paramsBuffer = createParamsBuffer(ctx.device, plan.paramsData);

  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    tensor.buffer,
    outBuffer,
    paramsBuffer,
  ]);

  dispatchComputePass(pipeline, bindGroup, plan.dispatchX, plan.dispatchY);

  releaseParamsBuffer(paramsBuffer);
  if (contiguousCopy) contiguousCopy.destroy();

  return createTensor(plan.outShape, outBuffer);
}
