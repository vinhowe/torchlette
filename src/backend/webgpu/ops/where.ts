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
  buildBroadcastIndexing,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchElementwise } from "../dispatch";
import { createTensor } from "../tensor";
import { resolveOutputBuffer } from "../buffer-arena";
import { params1 } from "../bind-group-cache";
import { computeFlatChunkLayout, dispatchFlatChunked } from "../chunked-dispatch";

function ternaryWhereShader(
  indexShape: number[],
  condStrides: number[],
  xStrides: number[],
  yStrides: number[],
  condOffset: number,
  xOffset: number,
  yOffset: number,
): string {
  const indexing = buildBroadcastIndexing(indexShape, [
    condStrides,
    xStrides,
    yStrides,
  ]);
  return `
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

${indexing.declarations}
const COND_OFFSET: u32 = ${condOffset}u;
const X_OFFSET: u32 = ${xOffset}u;
const Y_OFFSET: u32 = ${yOffset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
  let condVal = cond[offset0 + COND_OFFSET];
  let xVal = x[offset1 + X_OFFSET];
  let yVal = y[offset2 + Y_OFFSET];
  out[idx] = select(yVal, xVal, condVal != 0.0);
}
`;
}

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

  const code = ternaryWhereShader(
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
    params: params1(outSize),
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

  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  const layout = computeFlatChunkLayout(outSize, bytesPerElement, maxBindingSize, minAlignment);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    options?.outBuffer,
  );

  const condIsScalar = condTensor.size <= 1;
  const xIsScalar = xTensor.size <= 1;
  const yIsScalar = yTensor.size <= 1;

  // Build a flat chunked shader — no broadcast indexing needed since
  // scalar inputs are read at [0] and contiguous inputs are read at [idx].
  const condAccess = condIsScalar ? "cond[0]" : "cond[idx]";
  const xAccess = xIsScalar ? "x[0]" : "x[idx]";
  const yAccess = yIsScalar ? "y[0]" : "y[idx]";

  const idxCompute = layout.use2D
    ? `let idx = gid.x + gid.y * ${layout.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  const code = `
struct Params {
  chunkSize: u32,
};

@group(0) @binding(0) var<storage, read> cond: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  let condVal = ${condAccess};
  let xVal = ${xAccess};
  let yVal = ${yAccess};
  out[idx] = select(yVal, xVal, condVal != 0.0);
}
`;

  const key = `whereChunked:${condIsScalar}:${xIsScalar}:${yIsScalar}:${layout.use2D ? `2d:${layout.gridSizeX}` : "1d"}`;

  dispatchFlatChunked({
    key, shader: code, layout,
    inputs: [
      { buffer: condTensor.buffer, mode: condIsScalar ? "scalar" : "chunked" },
      { buffer: xTensor.buffer, mode: xIsScalar ? "scalar" : "chunked" },
      { buffer: yTensor.buffer, mode: yIsScalar ? "scalar" : "chunked" },
    ],
    outBuffer, outBytesPerElement: bytesPerElement, totalElements: outSize,
  });

  return createTensor(outShape, outBuffer);
}
