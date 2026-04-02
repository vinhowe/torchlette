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
  sizeOf,
  toIndexShape,
  WORKGROUP_SIZE,
  DEFAULT_MAX_STORAGE_BUFFER_BINDING_SIZE,
  F32_BYTES,
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
    limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const outSizeBytes = outSize * bytesPerElement;

  // Use chunked dispatch when output exceeds binding limit and inputs are chunkable:
  // Each input must be either scalar (0-d, stride=0 broadcast) or contiguous.
  if (outSizeBytes > maxBindingSize) {
    const condIsScalar = condTensor.size <= 1;
    const xIsScalar = xTensor.size <= 1;
    const yIsScalar = yTensor.size <= 1;
    const condChunkable = condIsScalar || condTensor.isContiguous;
    const xChunkable = xIsScalar || xTensor.isContiguous;
    const yChunkable = yIsScalar || yTensor.isContiguous;

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
  const key = `where:${indexShape.join("x")}:${condStrides.join(",")}:${xStrides.join(",")}:${yStrides.join(",")}:${condTensor.offset}:${xTensor.offset}:${yTensor.offset}`;

  const providedOut =
    options?.outBuffer && options.outBuffer.size >= outSize * 4
      ? options.outBuffer
      : undefined;

  const outBuffer = dispatchElementwise({
    key,
    shader: code,
    inputs: [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    outputSizeBytes: outSize * 4,
    params: params(outSize),
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
    { size: outSize },
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
