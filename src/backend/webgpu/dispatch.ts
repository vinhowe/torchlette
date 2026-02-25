/**
 * Core dispatch infrastructure: compute pass execution, pipeline caching,
 * binary/unary/matmul dispatch, fused elementwise.
 * Extracted from index.ts — purely structural refactoring.
 */

import type { FusionRecipe } from "../../engine/fusion";
import type { IRGraph, IRNode } from "../../engine/ir";
import {
  sizeOf,
  broadcastShapes,
  toIndexShape,
  broadcastStrides,
  computeEffectiveBroadcastStrides,
  buildBroadcastIndexing,
  shapesEqual,
  dtypeBytes,
  dtypeToWgsl,
  compute2DDispatch,
  WORKGROUP_SIZE,
  MAX_WORKGROUPS_PER_DIM,
} from "./shape-utils";
import type { BackendTensor, DType } from "../types";
import type { GPUBuffer, GPUComputePipeline, GPUBindGroup, WebGPUContext, WebGPUTensor } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { requireContext, isF16Supported } from "./gpu-context";
import { bufferPool } from "./buffer-pool";
import {
  getSharedEncoderInstance,
  autoFlushSharedEncoder,
  incrementSharedEncoderPassCount,
  submitOrCollect,
  getCurrentOpLabel,
} from "./shared-encoder";
import { dispatchRecordingBuffer, getAndClearLastBindGroupBuffers } from "./dispatch-recording";
import { resolveOutputBuffer } from "./buffer-arena";
import {
  createParamsBuffer,
  releaseParamsBuffer,
  createUniformBuffer,
  releaseUniformBuffer,
  cachedCreateBindGroup,
  params1,
} from "./bind-group-cache";
import { computeFlatChunkLayout, dispatchFlatChunked } from "./chunked-dispatch";
import { getTimestampWrites, getProfileModule } from "./profiler";
import {
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  dispatchTiledMatmul,
  type EpilogueConfig,
} from "./matmul";

// Tensor construction helpers (extracted to tensor.ts)
import { createTensor, createTrackedBuffer } from "./tensor";
import { ensureContiguous, detectSimpleTranspose } from "./ops/views";

export function dispatchComputePass(
  pipeline: GPUComputePipeline,
  bindGroup: unknown,
  workgroupsX: number,
  workgroupsY: number = 1,
  workgroupsZ: number = 1,
): void {
  const ctx = requireContext();

  // Record dispatch if recording is active
  if (dispatchRecordingBuffer) {
    dispatchRecordingBuffer.push({
      pipeline,
      bindGroup: bindGroup as GPUBindGroup,
      workgroupsX,
      workgroupsY,
      workgroupsZ,
      buffers: getAndClearLastBindGroupBuffers(),
      label: getCurrentOpLabel() ?? undefined,
      module: getProfileModule(),
    });
  }

  if (getSharedEncoderInstance()) {
    // Encode directly onto the shared encoder — no new encoder or CB
    const tsWrites = getTimestampWrites(getCurrentOpLabel() ?? "unknown");
    const pass = getSharedEncoderInstance().beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    incrementSharedEncoderPassCount();
    autoFlushSharedEncoder();
  } else {
    const encoder = ctx.device.createCommandEncoder();
    const tsWrites = getTimestampWrites(getCurrentOpLabel() ?? "unknown");
    const pass = encoder.beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup as GPUBindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    submitOrCollect(encoder.finish());
  }

}

/**
 * Unified dispatch helper for elementwise compute shaders.
 * Collapses the common 6-step boilerplate (pipeline, output, params, bindgroup, dispatch, release)
 * into a single call.
 */
export function dispatchElementwise(desc: {
  key: string;
  shader: string;
  inputs: GPUBuffer[];
  outputSizeBytes: number;
  params: Uint32Array;
  outBuffer?: GPUBuffer;
  dispatchX: number;
  dispatchY?: number;
}): GPUBuffer {
  const ctx = requireContext();

  const pipeline = getPipeline(ctx, desc.key, desc.shader);

  const outBuffer = desc.outBuffer
    ?? resolveOutputBuffer(ctx.device, desc.outputSizeBytes, desc.inputs);

  const paramsBuffer = createParamsBuffer(ctx.device, desc.params);

  const bgBuffers: GPUBuffer[] = [];
  for (let i = 0; i < desc.inputs.length; i++) {
    bgBuffers.push(desc.inputs[i]);
  }
  bgBuffers.push(outBuffer);
  bgBuffers.push(paramsBuffer);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);

  dispatchComputePass(pipeline, bindGroup, desc.dispatchX, desc.dispatchY ?? 1);

  releaseParamsBuffer(paramsBuffer);

  return outBuffer;
}

export function binaryBroadcastShader(
  op: string,
  indexShape: number[],
  aStrides: number[],
  bStrides: number[],
  aOffset: number,
  bOffset: number,
  dtype: DType = "f32",
  gridSizeX?: number,
): string {
  const indexing = buildBroadcastIndexing(indexShape, [aStrides, bStrides]);
  const wgslType = dtypeToWgsl(dtype);
  const enableF16 = dtype === "f16" ? "enable f16;\n" : "";
  // Support 2D dispatch for large tensors
  const use2D = gridSizeX !== undefined && gridSizeX > 0;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read> b: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(3) var<uniform> params: Params;

${indexing.declarations}
const A_OFFSET: u32 = ${aOffset}u;
const B_OFFSET: u32 = ${bOffset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
  out[idx] = a[offset0 + A_OFFSET] ${op} b[offset1 + B_OFFSET];
}
`;
}

/**
 * Generate unary shader with stride support for non-contiguous tensors.
 */
export function unaryStridedShader(
  expr: string,
  shape: number[],
  strides: number[],
  offset: number,
  dtype: DType = "f32",
  gridSizeX?: number,
): string {
  const rank = shape.length;
  const wgslType = dtypeToWgsl(dtype);
  const enableF16 = dtype === "f16" ? "enable f16;\n" : "";
  // Use 2D indexing when gridSizeX > MAX_WORKGROUPS_PER_DIM
  const use2D = gridSizeX !== undefined && gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  const idxCompute = use2D
    ? `let idx = gid.x + gid.y * ${gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;

  if (rank === 0) {
    // Scalar case
    return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }
  let x = a[${offset}u];
  out[idx] = ${expr};
}
`;
  }

  const shapeArray = `array<u32, ${rank}>(${shape.map((s) => `${s}u`).join(", ")})`;
  const stridesArray = `array<u32, ${rank}>(${strides.map((s) => `${s}u`).join(", ")})`;

  return `${enableF16}
struct Params {
  size: u32,
};

@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;
@group(0) @binding(2) var<uniform> params: Params;

const RANK: u32 = ${rank}u;
const SHAPE = ${shapeArray};
const STRIDES = ${stridesArray};
const OFFSET: u32 = ${offset}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.size) {
    return;
  }

  // Convert flat index to strided offset
  var remaining = idx;
  var inputOffset = OFFSET;
  for (var d = 0u; d < RANK; d = d + 1u) {
    var dimSize = 1u;
    for (var j = d + 1u; j < RANK; j = j + 1u) {
      dimSize = dimSize * SHAPE[j];
    }
    let coord = remaining / dimSize;
    remaining = remaining % dimSize;
    inputOffset = inputOffset + coord * STRIDES[d];
  }

  let x = a[inputOffset];
  out[idx] = ${expr};
}
`;
}

const FUSED_UNARY_OPS = new Map<string, (value: string) => string>([
  ["neg", (value) => `-(${value})`],
  ["abs", (value) => `abs(${value})`],
  ["exp", (value) => `exp(${value})`],
  ["log", (value) => `log(${value})`],
  ["relu", (value) => `select(0.0, ${value}, ${value} > 0.0)`],
  ["sqrt", (value) => `sqrt(${value})`],
]);

const FUSED_BINARY_OPS = new Map<
  string,
  (left: string, right: string) => string
>([
  ["add", (left, right) => `(${left} + ${right})`],
  ["sub", (left, right) => `(${left} - ${right})`],
  ["mul", (left, right) => `(${left} * ${right})`],
  ["div", (left, right) => `(${left} / ${right})`],
]);

function buildFusedExpression(op: string, inputs: string[]): string {
  const binary = FUSED_BINARY_OPS.get(op);
  if (binary) {
    if (inputs.length !== 2) {
      throw new Error(`fused op ${op} expects 2 inputs`);
    }
    return binary(inputs[0], inputs[1]);
  }
  const unary = FUSED_UNARY_OPS.get(op);
  if (unary) {
    if (inputs.length !== 1) {
      throw new Error(`fused op ${op} expects 1 input`);
    }
    return unary(inputs[0]);
  }
  throw new Error(`fused op ${op} is not supported`);
}

function requireFusionNode(
  nodeById: Map<number, IRNode>,
  nodeId: number,
): IRNode {
  const node = nodeById.get(nodeId);
  if (!node) {
    throw new Error(`fusion recipe missing node ${nodeId}`);
  }
  return node;
}

function requireShape(node: IRNode): number[] {
  if (!node.shape) {
    throw new Error(`fusion recipe missing shape for node ${node.id}`);
  }
  return node.shape;
}

function buildFusedElementwiseShader(
  graph: IRGraph,
  recipe: FusionRecipe,
  use2D?: boolean,
  shaderGridSizeX?: number,
): string {
  if (recipe.outputs.length !== 1) {
    throw new Error("fused elementwise expects a single output");
  }
  const outputDescriptor = recipe.outputDescriptors[0];
  if (!outputDescriptor) {
    throw new Error("fusion recipe has no output descriptors");
  }
  const outShape = outputDescriptor.shape.slice();
  const indexShape = toIndexShape(outShape);
  const nodeById = new Map<number, IRNode>();
  for (const node of graph.nodes) {
    nodeById.set(node.id, node);
  }

  const inputDecls: string[] = [];
  const inputStrides: number[][] = [];
  const valueById = new Map<number, string>();
  for (let i = 0; i < recipe.inputs.length; i += 1) {
    const name = `in${i}`;
    const inputNode = requireFusionNode(nodeById, recipe.inputs[i]);
    const inputShape = requireShape(inputNode);
    inputStrides.push(broadcastStrides(inputShape, indexShape));
    inputDecls.push(
      `@group(0) @binding(${i}) var<storage, read> ${name}: array<f32>;`,
    );
  }

  const indexing = buildBroadcastIndexing(indexShape, inputStrides);
  for (let i = 0; i < recipe.inputs.length; i += 1) {
    const name = `in${i}`;
    valueById.set(recipe.inputs[i], `${name}[offset${i}]`);
  }

  const statements: string[] = [];
  let varIndex = 0;
  for (const nodeId of recipe.nodeIds) {
    const node = requireFusionNode(nodeById, nodeId);
    const inputExprs = node.inputs.map((inputId) => {
      const expr = valueById.get(inputId);
      if (!expr) {
        throw new Error(`fusion recipe missing input ${inputId}`);
      }
      return expr;
    });
    const expr = buildFusedExpression(node.op, inputExprs);
    const varName = `v${varIndex}`;
    varIndex += 1;
    statements.push(`  let ${varName} = ${expr};`);
    valueById.set(nodeId, varName);
  }

  const outputExpr = valueById.get(recipe.outputs[0]);
  if (!outputExpr) {
    throw new Error(`fusion recipe missing output ${recipe.outputs[0]}`);
  }

  const outputBinding = recipe.inputs.length;
  const paramsBinding = recipe.inputs.length + 1;
  const body = statements.join("\n");

  return `
struct Params {
  size: u32,
};

${inputDecls.join("\n")}
@group(0) @binding(${outputBinding}) var<storage, read_write> out: array<f32>;
@group(0) @binding(${paramsBinding}) var<uniform> params: Params;

${indexing.declarations}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = ${use2D ? `gid.x + gid.y * ${shaderGridSizeX}u` : "gid.x"};
  if (idx >= params.size) {
    return;
  }
${indexing.compute}
${indexing.offsets.join("\n")}
${body}
  out[idx] = ${outputExpr};
}
`;
}

export function getPipeline(
  context: WebGPUContext,
  key: string,
  code: string,
): GPUComputePipeline {
  const cached = context.pipelines.get(key);
  if (cached) {
    return cached;
  }
  const module = context.device.createShaderModule({ code });
  const pipeline = context.device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  context.pipelines.set(key, pipeline);
  return pipeline;
}

/**
 * Dispatch binary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
export function dispatchBinary(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);
  if (outSize === 0) {
    throw new Error("webgpu ops do not support empty tensors yet");
  }
  const ctx = requireContext();

  // Determine output dtype - both inputs must have same dtype
  const dtype = a.dtype;
  if (b.dtype !== dtype) {
    throw new Error(
      `webgpu binary op: mismatched dtypes ${a.dtype} and ${b.dtype}`,
    );
  }

  // Check if any buffer exceeds max binding size
  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const bytesPerElement = dtypeBytes(dtype);

  const aSizeBytes = a.size * bytesPerElement;
  const bSizeBytes = b.size * bytesPerElement;
  const outSizeBytes = outSize * bytesPerElement;

  // Check if chunking is needed
  if (aSizeBytes > maxBindingSize || bSizeBytes > maxBindingSize || outSizeBytes > maxBindingSize) {
    // Check for simple case: same shape, both contiguous, or one is scalar
    const aIsScalar = a.size === 1;
    const bIsScalar = b.size === 1;
    const sameShape = a.shape.length === b.shape.length &&
      a.shape.every((d, i) => d === b.shape[i]);

    if ((sameShape && a.isContiguous && b.isContiguous) ||
        (aIsScalar && b.isContiguous) ||
        (bIsScalar && a.isContiguous)) {
      return dispatchBinaryChunked(op, a, b, options);
    }

    // Handle case where one or both tensors are non-contiguous
    // Materialize them to contiguous first, then use chunked path
    if (sameShape) {
      const aContiguous = a.isContiguous ? a : ensureContiguous(a);
      const bContiguous = b.isContiguous ? b : ensureContiguous(b);
      const result = dispatchBinaryChunked(op, aContiguous, bContiguous, options);
      // Destroy contiguous copies to prevent memory leaks
      if (aContiguous !== a) aContiguous.destroy?.();
      if (bContiguous !== b) bContiguous.destroy?.();
      return result;
    }

    // For broadcast cases with non-contiguous tensors, materialize the large non-contiguous one
    if (!a.isContiguous && aSizeBytes > maxBindingSize) {
      const aContiguous = ensureContiguous(a);
      const result = dispatchBinary(op, aContiguous, b, options);
      aContiguous.destroy?.();
      return result;
    }
    if (!b.isContiguous && bSizeBytes > maxBindingSize) {
      const bContiguous = ensureContiguous(b);
      const result = dispatchBinary(op, a, bContiguous, options);
      bContiguous.destroy?.();
      return result;
    }

    // Fall through to direct dispatch - may fail for complex cases
  }

  return dispatchBinaryDirect(op, a, b, options);
}

/**
 * Direct binary dispatch for small tensors (no chunking).
 */
export function dispatchBinaryDirect(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const indexShape = toIndexShape(outShape);
  const outSize = sizeOf(outShape);
  const dtype = a.dtype;

  const aStrides = computeEffectiveBroadcastStrides(a, indexShape);
  const bStrides = computeEffectiveBroadcastStrides(b, indexShape);

  const totalWorkgroups = Math.ceil(outSize / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = binaryBroadcastShader(
    op, indexShape, aStrides, bStrides, a.offset, b.offset, dtype,
    use2D ? dispatch.gridSizeX : undefined,
  );
  const key = `binary:${op}:${indexShape.join("x")}:${aStrides.join(",")}:${bStrides.join(",")}:${a.offset}:${b.offset}:${dtype}:${use2D ? dispatch.gridSizeX : "1d"}`;
  const bytesPerElement = dtypeBytes(dtype);

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [a.buffer, b.buffer],
    outputSizeBytes: outSize * bytesPerElement,
    params: params1(outSize),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Chunked binary dispatch for large tensors.
 * Handles: same-shape contiguous tensors, or scalar + large tensor.
 */
export function dispatchBinaryChunked(
  op: string,
  a: WebGPUTensor,
  b: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  const outShape = broadcastShapes(a.shape, b.shape);
  const outSize = sizeOf(outShape);

  const aIsScalar = a.size === 1;
  const bIsScalar = b.size === 1;

  const layout = computeFlatChunkLayout(outSize, bytesPerElement, maxBindingSize, minAlignment);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [a.buffer, b.buffer],
    options?.outBuffer,
  );

  // Build shader for chunked binary op
  const wgslType = dtypeToWgsl(dtype);
  const idxCompute = layout.use2D
    ? `let idx = gid.x + gid.y * ${layout.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  const aAccess = aIsScalar ? "a[0]" : "a[idx]";
  const bAccess = bIsScalar ? "b[0]" : "b[idx]";

  const code = `
@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read> b: array<${wgslType}>;
@group(0) @binding(2) var<storage, read_write> out: array<${wgslType}>;

struct Params {
  chunkSize: u32,
};
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  out[idx] = ${aAccess} ${op} ${bAccess};
}
`;

  const key = `binaryChunked:${op}:${dtype}:${aIsScalar}:${bIsScalar}:${layout.use2D ? `2d:${layout.gridSizeX}` : "1d"}`;

  dispatchFlatChunked({
    key, shader: code, layout,
    inputs: [
      { buffer: a.buffer, mode: aIsScalar ? "scalar" : "chunked" },
      { buffer: b.buffer, mode: bIsScalar ? "scalar" : "chunked" },
    ],
    outBuffer, outBytesPerElement: bytesPerElement, totalElements: outSize,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(outShape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Dispatch unary op with full stride support.
 * Handles non-contiguous tensors (transposed, expanded views) directly.
 */
export function dispatchUnary(
  opKey: string,
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  // Check if chunking is needed for large contiguous tensors
  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const aSizeBytes = a.size * bytesPerElement;

  if (aSizeBytes > maxBindingSize && a.isContiguous) {
    return dispatchUnaryChunked(opKey, expr, a, options);
  }

  return dispatchUnaryDirect(opKey, expr, a, options);
}

/**
 * Direct unary dispatch for small tensors.
 */
export function dispatchUnaryDirect(
  opKey: string,
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const dtype = a.dtype;

  const totalWorkgroups = Math.ceil(a.size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;

  const code = unaryStridedShader(
    expr, a.shape, a.strides, a.offset, dtype,
    use2D ? dispatch.gridSizeX : undefined,
  );
  const key = `unary:${opKey}:${a.shape.join("x")}:${a.strides.join(",")}:${a.offset}:${dtype}:${use2D ? `2d:${dispatch.gridSizeX}` : "1d"}`;
  const bytesPerElement = dtypeBytes(dtype);

  const outBuffer = dispatchElementwise({
    key, shader: code,
    inputs: [a.buffer],
    outputSizeBytes: a.size * bytesPerElement,
    params: params1(a.size),
    outBuffer: options?.outBuffer,
    dispatchX: dispatch.x, dispatchY: dispatch.y,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

/**
 * Chunked unary dispatch for large contiguous tensors.
 */
export function dispatchUnaryChunked(
  opKey: string,
  expr: string,
  a: WebGPUTensor,
  options?: { outBuffer?: GPUBuffer },
): WebGPUTensor {
  const ctx = requireContext();
  const dtype = a.dtype;
  const bytesPerElement = dtypeBytes(dtype);

  const limits = ctx.device.limits;
  const maxBindingSize = limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const minAlignment = limits?.minStorageBufferOffsetAlignment ?? 256;

  const outSize = a.size;

  const layout = computeFlatChunkLayout(outSize, bytesPerElement, maxBindingSize, minAlignment);

  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [a.buffer],
    options?.outBuffer,
  );

  // Build shader for chunked unary op
  const wgslType = dtypeToWgsl(dtype);
  const idxCompute = layout.use2D
    ? `let idx = gid.x + gid.y * ${layout.gridSizeX}u * ${WORKGROUP_SIZE}u;`
    : `let idx = gid.x;`;
  const exprWithAccess = expr.replace(/\bx\b/g, "a[idx]");

  const code = `
@group(0) @binding(0) var<storage, read> a: array<${wgslType}>;
@group(0) @binding(1) var<storage, read_write> out: array<${wgslType}>;

struct Params {
  chunkSize: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${idxCompute}
  if (idx >= params.chunkSize) { return; }
  out[idx] = ${exprWithAccess};
}
`;

  const key = `unaryChunked:${opKey}:${dtype}:${layout.use2D ? `2d:${layout.gridSizeX}` : "1d"}`;

  dispatchFlatChunked({
    key, shader: code, layout,
    inputs: [{ buffer: a.buffer, mode: "chunked" }],
    outBuffer, outBytesPerElement: bytesPerElement, totalElements: outSize,
  });

  const ownsBuffer = options?.outBuffer === undefined;
  return createTensor(a.shape, outBuffer, undefined, 0, dtype, ownsBuffer);
}

export function dispatchMatmul(
  a: WebGPUTensor,
  b: WebGPUTensor,
  transA = false,
  transB = false,
  donatedBuffer?: GPUBuffer,
): WebGPUTensor {
  const ctx = requireContext();

  // Try to detect simple last-2-dim transposes to avoid contiguous() materialization.
  // If detected, we use the original contiguous buffer and flip the transpose flag.
  let effectiveA: WebGPUTensor = a;
  let effectiveTransA = transA;
  let aWasCopied = false;

  const aOrigShape = !transA ? detectSimpleTranspose(a) : null;
  if (aOrigShape) {
    // Use original buffer with swapped shape and flipped transpose flag
    effectiveA = createTensor(aOrigShape, a.buffer, undefined, 0, a.dtype, false);
    effectiveTransA = true;
  } else {
    effectiveA = ensureContiguous(a);
    aWasCopied = effectiveA !== a;
  }

  let effectiveB: WebGPUTensor = b;
  let effectiveTransB = transB;
  let bWasCopied = false;

  const bOrigShape = !transB ? detectSimpleTranspose(b) : null;
  if (bOrigShape) {
    effectiveB = createTensor(bOrigShape, b.buffer, undefined, 0, b.dtype, false);
    effectiveTransB = true;
  } else {
    effectiveB = ensureContiguous(b);
    bWasCopied = effectiveB !== b;
  }

  // Compute output shape with transpose and batch broadcasting
  const outShape = computeMatmulOutputShape(
    effectiveA.shape,
    effectiveB.shape,
    effectiveTransA,
    effectiveTransB,
  );

  // Extract matrix dimensions
  const aRank = effectiveA.shape.length;
  const bRank = effectiveB.shape.length;

  let m: number, k: number, n: number;
  if (effectiveTransA) {
    k = effectiveA.shape[aRank - 2];
    m = effectiveA.shape[aRank - 1];
  } else {
    m = effectiveA.shape[aRank - 2];
    k = effectiveA.shape[aRank - 1];
  }
  if (effectiveTransB) {
    n = effectiveB.shape[bRank - 2];
  } else {
    n = effectiveB.shape[bRank - 1];
  }

  // Compute batch size and strides
  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    effectiveA.shape,
    effectiveB.shape,
    batchDims,
    m,
    n,
    k,
  );

  // Derive per-input dtypes; output is always the higher-precision type
  const dtypeA = effectiveA.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const dtypeB = effectiveB.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const outputDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
  const bytesPerElement = outputDtype === "f16" ? 2 : 4;

  // Create or use donated output buffer
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);
  const requiredSize = outSize * bytesPerElement;
  const useDonated =
    donatedBuffer && (donatedBuffer as any).size >= requiredSize;
  const outBuffer = useDonated
    ? donatedBuffer
    : resolveOutputBuffer(ctx.device, requiredSize,
        [effectiveA.buffer, effectiveB.buffer]);

  // Dispatch tiled matmul
  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: effectiveA.buffer,
    b: effectiveB.buffer,
    out: outBuffer,
    m,
    n,
    k,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    transA: effectiveTransA,
    transB: effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
  });

  // Destroy contiguous copies if they were created (deferred for GPU fence)
  if (aWasCopied) {
    bufferPool.decRef(effectiveA.buffer);
    bufferPool.deferredDestroy(effectiveA.buffer, effectiveA.size * (a.dtype === "f16" ? 2 : 4));
  }
  if (bWasCopied) {
    bufferPool.decRef(effectiveB.buffer);
    bufferPool.deferredDestroy(effectiveB.buffer, effectiveB.size * (b.dtype === "f16" ? 2 : 4));
  }

  // Output tensor always owns the buffer (donated or new)
  return createTensor(outShape, outBuffer, undefined, 0, outputDtype, true);
}

/**
 * Dispatch matmul with fused epilogue operations.
 *
 * This function runs matmul with additional elementwise operations
 * (like bias, relu, gelu) fused into the output write loop for better performance.
 */
export function dispatchMatmulWithEpilogue(
  a: WebGPUTensor,
  b: WebGPUTensor,
  epilogue: EpilogueConfig,
  epilogueInputs: WebGPUTensor[],
  transA = false,
  transB = false,
  inputCastA?: DType,
  inputCastB?: DType,
): WebGPUTensor {
  const ctx = requireContext();

  // Try to detect simple last-2-dim transposes to avoid contiguous() materialization.
  let effectiveA: WebGPUTensor = a;
  let effectiveTransA = transA;
  let aWasCopied = false;

  const aOrigShape = !transA ? detectSimpleTranspose(a) : null;
  if (aOrigShape) {
    effectiveA = createTensor(aOrigShape, a.buffer, undefined, 0, a.dtype, false);
    effectiveTransA = true;
  } else {
    effectiveA = ensureContiguous(a);
    aWasCopied = effectiveA !== a;
  }

  let effectiveB: WebGPUTensor = b;
  let effectiveTransB = transB;
  let bWasCopied = false;

  const bOrigShape = !transB ? detectSimpleTranspose(b) : null;
  if (bOrigShape) {
    effectiveB = createTensor(bOrigShape, b.buffer, undefined, 0, b.dtype, false);
    effectiveTransB = true;
  } else {
    effectiveB = ensureContiguous(b);
    bWasCopied = effectiveB !== b;
  }

  // Compute output shape with transpose and batch broadcasting
  const outShape = computeMatmulOutputShape(
    effectiveA.shape,
    effectiveB.shape,
    effectiveTransA,
    effectiveTransB,
  );

  // Extract matrix dimensions
  const aRank = effectiveA.shape.length;
  const bRank = effectiveB.shape.length;

  let m: number, k: number, n: number;
  if (effectiveTransA) {
    k = effectiveA.shape[aRank - 2];
    m = effectiveA.shape[aRank - 1];
  } else {
    m = effectiveA.shape[aRank - 2];
    k = effectiveA.shape[aRank - 1];
  }
  if (effectiveTransB) {
    n = effectiveB.shape[bRank - 2];
  } else {
    n = effectiveB.shape[bRank - 1];
  }

  // Compute batch size and strides
  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    effectiveA.shape,
    effectiveB.shape,
    batchDims,
    m,
    n,
    k,
  );

  // Derive per-input dtypes; output dtype from epilogue or promoted type.
  // When inputCastA/B is set, the actual buffer dtype is wider (e.g. f32) but
  // the matmul computes in the target dtype (e.g. f16) by casting during tile load.
  // The "compute dtype" for codegen is the post-cast dtype.
  const rawDtypeA = effectiveA.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const rawDtypeB = effectiveB.dtype === "f16" && isF16Supported() ? "f16" as const : "f32" as const;
  const dtypeA: "f16" | "f32" = inputCastA === "f16" && isF16Supported() ? "f16" : rawDtypeA;
  const dtypeB: "f16" | "f32" = inputCastB === "f16" && isF16Supported() ? "f16" : rawDtypeB;
  const promotedDtype = (dtypeA === "f32" || dtypeB === "f32") ? "f32" as const : dtypeA;
  const outputDtype = epilogue.outputDtype ?? promotedDtype;
  const bytesPerElement = outputDtype === "f16" ? 2 : 4;

  // Extract GPU buffers from epilogue input tensors
  const epilogueBuffers = epilogueInputs.map((t) => t.buffer);

  // Create output buffer (routed through arena for stable bind group cache)
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);
  const outBuffer = resolveOutputBuffer(ctx.device, outSize * bytesPerElement,
    [effectiveA.buffer, effectiveB.buffer, ...epilogueBuffers]);

  // Dispatch tiled matmul with epilogue
  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: effectiveA.buffer,
    b: effectiveB.buffer,
    out: outBuffer,
    m,
    n,
    k,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    transA: effectiveTransA,
    transB: effectiveTransB,
    dtype: dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
    epilogue,
    epilogueInputs: epilogueBuffers,
    inputCastA,
    inputCastB,
  });

  // Destroy contiguous copies if they were created (deferred for GPU fence)
  if (aWasCopied) {
    bufferPool.decRef(effectiveA.buffer);
    bufferPool.deferredDestroy(effectiveA.buffer, effectiveA.size * (a.dtype === "f16" ? 2 : 4));
  }
  if (bWasCopied) {
    bufferPool.decRef(effectiveB.buffer);
    bufferPool.deferredDestroy(effectiveB.buffer, effectiveB.size * (b.dtype === "f16" ? 2 : 4));
  }

  return createTensor(outShape, outBuffer, undefined, 0, outputDtype);
}

/**
 * Direct matmul dispatch with pre-computed geometry.
 *
 * Used by the lowered plan fast path to skip shape computation,
 * transpose detection, contiguous checks, and dtype resolution.
 * All structural decisions were computed on the first execution and cached.
 */
export function dispatchMatmulDirect(
  bufA: GPUBuffer,
  bufB: GPUBuffer,
  config: {
    m: number; n: number; k: number;
    transA: boolean; transB: boolean;
    batchSize: number;
    batchStrideA: number; batchStrideB: number; batchStrideC: number;
    outShape: number[];
    dtypeA: "f16" | "f32";
    dtypeB?: "f16" | "f32";
    outputDtype: DType;
    epilogueConfig: any;
    epilogueBuffers: GPUBuffer[];
    inputCastA?: DType;
    inputCastB?: DType;
  },
): WebGPUTensor {
  const ctx = requireContext();
  const bytesPerElement = config.outputDtype === "f16" ? 2 : 4;
  const outSize = config.outShape.reduce((a, b) => a * b, 1);
  const outBuffer = resolveOutputBuffer(
    ctx.device,
    outSize * bytesPerElement,
    [bufA, bufB, ...config.epilogueBuffers],
  );

  dispatchTiledMatmul({
    device: ctx.device,
    queue: ctx.queue,
    a: bufA,
    b: bufB,
    out: outBuffer,
    m: config.m,
    n: config.n,
    k: config.k,
    batchSize: config.batchSize,
    batchStrideA: config.batchStrideA,
    batchStrideB: config.batchStrideB,
    batchStrideC: config.batchStrideC,
    transA: config.transA,
    transB: config.transB,
    dtype: config.dtypeA,
    dtypeB: config.dtypeB,
    epilogue: config.epilogueConfig,
    epilogueInputs: config.epilogueBuffers,
    inputCastA: config.inputCastA,
    inputCastB: config.inputCastB,
  });

  return createTensor(config.outShape, outBuffer, undefined, 0, config.outputDtype);
}

export function runFusedElementwise(
  graph: IRGraph,
  recipe: FusionRecipe,
  inputs: BackendTensor[],
): BackendTensor | null {
  // Get output dtype from first output descriptor (single-output case)
  const outputDescriptor = recipe.outputDescriptors[0];
  if (!outputDescriptor) {
    throw new Error("fusion recipe has no output descriptors");
  }
  if (outputDescriptor.dtype !== "f32") {
    // Non-f32 fusion not yet supported; return null to fall back to sequential execution
    return null;
  }
  if (inputs.length !== recipe.inputs.length) {
    throw new Error(
      `fusion recipe expects ${recipe.inputs.length} inputs, got ${inputs.length}`,
    );
  }

  const ctx = requireContext();
  const outShape = outputDescriptor.shape.slice();
  const size = sizeOf(outShape);
  if (size === 0) {
    throw new Error("fused elementwise does not support empty tensors yet");
  }
  const inputTensors = inputs as WebGPUTensor[];
  const nodeById = new Map<number, IRNode>();
  for (const node of graph.nodes) {
    nodeById.set(node.id, node);
  }
  recipe.inputs.forEach((inputId, index) => {
    const node = requireFusionNode(nodeById, inputId);
    const expectedShape = requireShape(node);
    if (!shapesEqual(inputTensors[index].shape, expectedShape)) {
      throw new Error(
        `fused elementwise input shape mismatch for node ${inputId}`,
      );
    }
  });

  const totalWorkgroups = Math.ceil(size / WORKGROUP_SIZE);
  const dispatch = compute2DDispatch(totalWorkgroups);
  const use2D = dispatch.y > 1;
  const shaderGridSizeX = dispatch.x * WORKGROUP_SIZE;
  const code = buildFusedElementwiseShader(graph, recipe, use2D, shaderGridSizeX);
  const pipeline = getPipeline(ctx, `fused:${code}`, code);
  const outBuffer = createTrackedBuffer(ctx.device, {
    size: size * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const params = createUniformBuffer(ctx.device, size);
  const bgBuffers: GPUBuffer[] = inputTensors.map(t => t.buffer);
  bgBuffers.push(outBuffer);
  bgBuffers.push(params);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, bgBuffers);
  dispatchComputePass(pipeline, bindGroup, dispatch.x, dispatch.y);
  releaseUniformBuffer(params);
  return createTensor(outShape, outBuffer);
}
