/**
 * Where (ternary select) ops: where, whereDirect, whereChunked.
 */

import { broadcastThreeShapes } from "../../../core/shape";
import type { BackendTensor } from "../../types";
import { params } from "../bind-group-cache";
import { resolveOutputBuffer } from "../buffer-arena";
import { dispatchElementwise } from "../dispatch";
import { requireContext } from "../gpu-context";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { asGPUTensor } from "../gpu-types";
import {
  computeEffectiveBroadcastStrides,
  DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE,
  F32_BYTES,
  sizeOf,
  toIndexShape,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { createTensor } from "../tensor";
import { createTileKernelDispatcher } from "../tile-dispatch";
import { whereSpec, whereWGSL } from "./ops-tile-ir";

/**
 * where(condition, x, y): returns x where condition is true (non-zero), else y.
 */
export function where(
  condition: BackendTensor,
  x: BackendTensor,
  y: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  const condTensor = asGPUTensor(condition);
  const xTensor = asGPUTensor(x);
  const yTensor = asGPUTensor(y);

  const outShape = broadcastThreeShapes(
    condTensor.shape,
    xTensor.shape,
    yTensor.shape,
  );
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);

  if (outSize === 0) {
    throw new Error("webgpu where does not support empty tensors yet");
  }

  const ctx = requireContext();

  // Check if chunking is needed for large contiguous tensors
  const bytesPerElement = 4; // f32
  const limits = ctx.device.limits;
  const maxBindingSize =
    limits?.maxStorageBufferBindingSize ?? DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  const outSizeBytes = outSize * bytesPerElement;

  // Use chunked dispatch when output exceeds binding limit and inputs are chunkable:
  // Each input must be either scalar (0-d, stride=0 broadcast) or contiguous.
  if (outSizeBytes > maxBindingSize) {
    const condIsScalar = condTensor.size <= 1;
    const xIsScalar = xTensor.size <= 1;
    const yIsScalar = yTensor.size <= 1;
    // Chunked binding slices each input by byte ranges from element 0 —
    // offset>0 views are NOT chunkable even with contiguous strides
    // (offset-view class, task #58).
    const condChunkable =
      condIsScalar ||
      (condTensor.isContiguous && (condTensor.offset ?? 0) === 0);
    const xChunkable =
      xIsScalar || (xTensor.isContiguous && (xTensor.offset ?? 0) === 0);
    const yChunkable =
      yIsScalar || (yTensor.isContiguous && (yTensor.offset ?? 0) === 0);

    if (condChunkable && xChunkable && yChunkable) {
      return whereChunked(
        condTensor,
        xTensor,
        yTensor,
        outShape,
        outSize,
        options,
      );
    }
  }

  return whereDirect(
    condTensor,
    xTensor,
    yTensor,
    outShape,
    indexShape,
    outSize,
    options,
  );
}

/** The stream-generate plan shape a direct `where` produces (mirrors
 *  ElementwiseDirectPlan: key IS the WGSL, params ride [size, ...offsets]). */
export interface WhereDirectPlan {
  key: string;
  shader: string;
  outputSizeBytes: number;
  paramsData: ArrayBufferView;
  dispatchX: number;
  dispatchY: number;
}

/**
 * Generator-facing plan builder for a direct (non-chunked) `where`. Returns the
 * exact command geometry `whereDirect` dispatches — key/shader/params/dispatch —
 * or null when `where()` would route away from the direct path (the chunked
 * >128MB branch, which stays uncovered → the plan keeps record/replay). The
 * stream generator consumes this so the generated stream is byte-identical to
 * the recording (single-source-at-seams, like planBinaryDirect).
 */
export function planWhereDirect(
  condTensor: WebGPUTensor,
  xTensor: WebGPUTensor,
  yTensor: WebGPUTensor,
): WhereDirectPlan | null {
  const outShape = broadcastThreeShapes(
    condTensor.shape,
    xTensor.shape,
    yTensor.shape,
  );
  const outSize = sizeOf(outShape);
  if (outSize === 0) return null;
  // `where()` routes to whereChunked when the output exceeds the binding limit;
  // that is a different command shape (chunked dispatcher) — leave it uncovered.
  const maxBindingSize =
    requireContext().device.limits?.maxStorageBufferBindingSize ??
    DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE;
  if (outSize * F32_BYTES > maxBindingSize) return null;
  const indexShape = toIndexShape(outShape);
  const condStrides = computeEffectiveBroadcastStrides(condTensor, indexShape);
  const xStrides = computeEffectiveBroadcastStrides(xTensor, indexShape);
  const yStrides = computeEffectiveBroadcastStrides(yTensor, indexShape);
  const code = whereWGSL(
    indexShape,
    condStrides,
    xStrides,
    yStrides,
    condTensor.offset,
    xTensor.offset,
    yTensor.offset,
  );
  return {
    key: code,
    shader: code,
    outputSizeBytes: outSize * F32_BYTES,
    paramsData: params(
      outSize,
      condTensor.offset,
      xTensor.offset,
      yTensor.offset,
    ),
    dispatchX: Math.ceil(outSize / WORKGROUP_SIZE),
    dispatchY: 1,
  };
}

/**
 * Direct (non-chunked) where dispatch using broadcast indexing.
 */
function whereDirect(
  condTensor: WebGPUTensor,
  xTensor: WebGPUTensor,
  yTensor: WebGPUTensor,
  outShape: number[],
  indexShape: number[],
  outSize: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const condStrides = computeEffectiveBroadcastStrides(condTensor, indexShape);
  const xStrides = computeEffectiveBroadcastStrides(xTensor, indexShape);
  const yStrides = computeEffectiveBroadcastStrides(yTensor, indexShape);

  const code = whereWGSL(
    indexShape,
    condStrides,
    xStrides,
    yStrides,
    condTensor.offset,
    xTensor.offset,
    yTensor.offset,
  );
  const providedOut =
    options?.outBuffer && options.outBuffer.size >= outSize * 4
      ? options.outBuffer
      : undefined;

  const outBuffer = dispatchElementwise({
    // Key IS the WGSL (single-source-at-seams; tile-dispatch canonical).
    key: code,
    shader: code,
    inputs: [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    outputSizeBytes: outSize * 4,
    // Task #71: offsets ride params after size, in uniform-declaration order
    // (size, cond_offset, x_offset, y_offset).
    params: params(
      outSize,
      condTensor.offset,
      xTensor.offset,
      yTensor.offset,
    ),
    outBuffer: providedOut,
    dispatchX: Math.ceil(outSize / WORKGROUP_SIZE),
  });

  return createTensor(outShape, outBuffer);
}

/**
 * Chunked where dispatch for large contiguous tensors.
 * Each input is either scalar (bound fully each chunk) or contiguous (bound as sub-range).
 */
function whereChunked(
  condTensor: WebGPUTensor,
  xTensor: WebGPUTensor,
  yTensor: WebGPUTensor,
  outShape: number[],
  outSize: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = 4; // f32

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    options?.outBuffer,
  );

  const condIsScalar = condTensor.size <= 1;
  const xIsScalar = xTensor.size <= 1;
  const yIsScalar = yTensor.size <= 1;

  // Flat chunked: scalar inputs get stride 0, contiguous inputs get stride 1
  const spec = whereSpec(
    [outSize],
    condIsScalar ? [0] : [1],
    xIsScalar ? [0] : [1],
    yIsScalar ? [0] : [1],
    0,
    0,
    0,
  );
  const dispatcher = createTileKernelDispatcher(spec);
  dispatcher.dispatchChunked(
    {
      cond: condTensor.buffer,
      x: xTensor.buffer,
      y: yTensor.buffer,
      out: outBuffer,
    },
    // Task #71: chunked binds each input from element 0, so all offsets are 0.
    { size: outSize, cond_offset: 0, x_offset: 0, y_offset: 0 },
    {
      modes: {
        cond: condIsScalar ? "scalar" : "chunked",
        x: xIsScalar ? "scalar" : "chunked",
        y: yIsScalar ? "scalar" : "chunked",
        out: "chunked",
      },
      sizeUniform: "size",
      totalElements: outSize,
      maxBytesPerElement: bytesPerElement,
    },
  );

  return createTensor(outShape, outBuffer);
}
