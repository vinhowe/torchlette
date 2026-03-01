/**
 * Where (ternary select) ops: where, whereDirect, whereChunked.
 * Extracted from index.ts — purely structural refactoring.
 */

import type { BackendTensor } from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { asGPUTensor } from "../gpu-types";
import {
  toIndexShape,
  sizeOf,
  computeEffectiveBroadcastStrides,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { whereWGSL, whereSpec } from "./ops-tile-ir";
import { requireContext } from "../gpu-context";
import { dispatchElementwise } from "../dispatch";
import { createTensor } from "../tensor";
import { resolveOutputBuffer } from "../buffer-arena";
import { params } from "../bind-group-cache";
import { createTileKernelDispatcher } from "../tile-dispatch";

/**
 * Broadcast three shapes to a common output shape.
 */
function broadcastThreeShapes(a: number[], b: number[], c: number[]): number[] {
  const outRank = Math.max(a.length, b.length, c.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    const cDim = c[c.length - 1 - i] ?? 1;
    // Check all pairs for broadcast compatibility
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    if (aDim !== cDim && aDim !== 1 && cDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    if (bDim !== cDim && bDim !== 1 && cDim !== 1) {
      throw new Error("webgpu shapes are not broadcastable");
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim, cDim);
  }
  return out;
}

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
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
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
      return whereChunked(condTensor, xTensor, yTensor, outShape, outSize, options);
    }
  }

  return whereDirect(condTensor, xTensor, yTensor, outShape, indexShape, outSize, options);
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
    indexShape, condStrides, xStrides, yStrides,
    condTensor.offset, xTensor.offset, yTensor.offset,
  );
  const key = `where:${indexShape.join("x")}:${condStrides.join(",")}:${xStrides.join(",")}:${yStrides.join(",")}:${condTensor.offset}:${xTensor.offset}:${yTensor.offset}`;

  const providedOut = options?.outBuffer && options.outBuffer.size >= outSize * 4
    ? options.outBuffer
    : undefined;

  const outBuffer = dispatchElementwise({
    key, shader: code,
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
    0, 0, 0,
  );
  const dispatcher = createTileKernelDispatcher(spec);
  dispatcher.dispatchChunked(
    { cond: condTensor.buffer, x: xTensor.buffer, y: yTensor.buffer, out: outBuffer },
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
