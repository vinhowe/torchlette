/**
 * Where (ternary select) ops: where, whereDirect, whereChunked.
 * Extracted from index.ts — purely structural refactoring.
 */

import type { BackendTensor } from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import {
  toIndexShape,
  sizeOf,
  computeEffectiveBroadcastStrides,
  buildBroadcastIndexing,
  WORKGROUP_SIZE,
  MAX_WORKGROUPS_PER_DIM,
} from "../shape-utils";
import { requireContext } from "../gpu-context";
import { dispatchElementwise, dispatchComputePass, getPipeline } from "../dispatch";
import { createTensor } from "../tensor";
import { resolveOutputBuffer } from "../buffer-arena";
import {
  params1,
  createUniformBuffer,
  releaseUniformBuffer,
  profiledCreateBindGroup,
} from "../bind-group-cache";

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
export function broadcastThreeShapes(a: number[], b: number[], c: number[]): number[] {
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
  const condTensor = condition as WebGPUTensor;
  const xTensor = x as WebGPUTensor;
  const yTensor = y as WebGPUTensor;

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
  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
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
export function whereDirect(
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
export function whereChunked(
  condTensor: WebGPUTensor,
  xTensor: WebGPUTensor,
  yTensor: WebGPUTensor,
  outShape: number[],
  outSize: number,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = 4; // f32

  const limits = (ctx.device as unknown as { limits: Record<string, number> })
    .limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  // Calculate chunk size
  const elementsPerAlignment = minAlignment / bytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk =
    Math.floor(maxElementsPerChunk / elementsPerAlignment) * elementsPerAlignment;

  const numChunks = Math.ceil(outSize / elementsPerChunk);
  const outSizeBytes = outSize * bytesPerElement;

  // Create output buffer
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSizeBytes,
    [condTensor.buffer, xTensor.buffer, yTensor.buffer],
    options?.outBuffer,
  );

  const condIsScalar = condTensor.size <= 1;
  const xIsScalar = xTensor.size <= 1;
  const yIsScalar = yTensor.size <= 1;

  // Determine 2D dispatch dimensions for large chunks
  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Build a flat chunked shader — no broadcast indexing needed since
  // scalar inputs are read at [0] and contiguous inputs are read at [idx].
  const condAccess = condIsScalar ? "cond[0]" : "cond[idx]";
  const xAccess = xIsScalar ? "x[0]" : "x[idx]";
  const yAccess = yIsScalar ? "y[0]" : "y[idx]";

  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
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

  const key = `whereChunked:${condIsScalar}:${xIsScalar}:${yIsScalar}:${use2D ? `2d:${gridSizeX}` : "1d"}`;
  const pipeline = getPipeline(ctx, key, code);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, outSize);
    const chunkSize = chunkEnd - chunkStart;
    const chunkByteOffset = chunkStart * bytesPerElement;
    const chunkByteSize = chunkSize * bytesPerElement;

    const paramsBuffer = createUniformBuffer(ctx.device, chunkSize);

    // Scalar inputs: bind the full (small) buffer. Contiguous inputs: bind the chunk sub-range.
    const condBinding = condIsScalar
      ? { buffer: condTensor.buffer }
      : { buffer: condTensor.buffer, offset: chunkByteOffset, size: chunkByteSize };
    const xBinding = xIsScalar
      ? { buffer: xTensor.buffer }
      : { buffer: xTensor.buffer, offset: chunkByteOffset, size: chunkByteSize };
    const yBinding = yIsScalar
      ? { buffer: yTensor.buffer }
      : { buffer: yTensor.buffer, offset: chunkByteOffset, size: chunkByteSize };

    const chunkWorkgroups = Math.ceil(chunkSize / WORKGROUP_SIZE);
    const dispatchX = use2D ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM) : chunkWorkgroups;
    const dispatchY = use2D ? Math.ceil(chunkWorkgroups / dispatchX) : 1;

    const bindGroup = profiledCreateBindGroup(ctx.device, {
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: condBinding },
        { binding: 1, resource: xBinding },
        { binding: 2, resource: yBinding },
        { binding: 3, resource: { buffer: outBuffer, offset: chunkByteOffset, size: chunkByteSize } },
        { binding: 4, resource: { buffer: paramsBuffer } },
      ],
    });

    dispatchComputePass(pipeline, bindGroup, dispatchX, dispatchY);
    releaseUniformBuffer(paramsBuffer);
  }

  return createTensor(outShape, outBuffer);
}
