/**
 * Pure utility functions for shape/stride computation, broadcast indexing,
 * dtype helpers, and WGSL code generation helpers.
 *
 * Zero mutable state. No GPU context dependency.
 */

import type { DType } from "../types";
import { computeContiguousStrides } from "../types";
export { sizeOf, broadcastShapes, shapesEqual } from "../../core/shape";
export { computeContiguousStrides as contiguousStrides } from "../types";

// ============================================================================
// Constants
// ============================================================================

export const WORKGROUP_SIZE = 256;

/** Maximum workgroups per dimension in WebGPU (per spec) */
export const MAX_WORKGROUPS_PER_DIM = 65535;

/**
 * Greatest common divisor using Euclidean algorithm.
 * Used for buffer alignment calculations.
 */
export function gcd(a: number, b: number): number {
  while (b !== 0) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a;
}

export function lcm(a: number, b: number): number {
  return (a * b) / gcd(a, b);
}

// broadcastShapes re-exported from core/shape

export function toIndexShape(shape: number[]): number[] {
  return shape.length === 0 ? [1] : shape;
}

// contiguousStrides re-exported from ../types above

export function broadcastStrides(shape: number[], outShape: number[]): number[] {
  if (shape.length > outShape.length) {
    throw new Error("webgpu broadcast target has fewer dimensions than input");
  }
  const pad = outShape.length - shape.length;
  const inStrides = computeContiguousStrides(shape);
  const outStrides = new Array<number>(outShape.length);
  for (let axis = 0; axis < outShape.length; axis += 1) {
    const inAxis = axis - pad;
    if (inAxis < 0) {
      outStrides[axis] = 0;
      continue;
    }
    const inDim = shape[inAxis];
    const outDim = outShape[axis];
    if (inDim === outDim) {
      outStrides[axis] = inStrides[inAxis];
    } else if (inDim === 1) {
      outStrides[axis] = 0;
    } else {
      throw new Error("webgpu broadcast target shape is incompatible");
    }
  }
  return outStrides;
}

/**
 * Compute effective broadcast strides for a tensor with existing strides.
 * This handles non-contiguous tensors (e.g., transposed, expanded views).
 */
export function computeEffectiveBroadcastStrides(
  tensor: { shape: number[]; strides: number[] },
  outShape: number[],
): number[] {
  const shape = tensor.shape;
  const strides = tensor.strides;

  if (shape.length > outShape.length) {
    throw new Error("webgpu broadcast target has fewer dimensions than input");
  }

  const pad = outShape.length - shape.length;
  const outStrides = new Array<number>(outShape.length);

  for (let axis = 0; axis < outShape.length; axis += 1) {
    const inAxis = axis - pad;
    if (inAxis < 0) {
      // Leading dimension not in input - broadcast with stride 0
      outStrides[axis] = 0;
      continue;
    }
    const inDim = shape[inAxis];
    const outDim = outShape[axis];
    if (inDim === outDim) {
      // Use the tensor's actual stride
      outStrides[axis] = strides[inAxis];
    } else if (inDim === 1) {
      // Broadcast: stride = 0
      outStrides[axis] = 0;
    } else {
      throw new Error("webgpu broadcast target shape is incompatible");
    }
  }
  return outStrides;
}

// ============================================================================
// WGSL Code Generation Helpers
// ============================================================================

export function wgslArray(values: number[]): string {
  return values.map((value) => `${value}u`).join(", ");
}

export function buildBroadcastIndexing(
  indexShape: number[],
  inputStrides: number[][],
): {
  declarations: string;
  compute: string;
  offsets: string[];
} {
  const rank = indexShape.length;
  const outShapeDecl = `const OUT_SHAPE: array<u32, ${rank}> = array<u32, ${rank}>(${wgslArray(indexShape)});`;
  const strideDecls = inputStrides.map(
    (strides, index) =>
      `const IN${index}_STRIDES: array<u32, ${rank}> = array<u32, ${rank}>(${wgslArray(strides)});`,
  );
  const compute = `
  var remaining = idx;
  var coords: array<u32, ${rank}>;
  for (var axis = 0u; axis < ${rank}u; axis = axis + 1u) {
    let rev = ${rank}u - 1u - axis;
    let dim = OUT_SHAPE[rev];
    let coord = remaining % dim;
    coords[rev] = coord;
    remaining = remaining / dim;
  }
`;
  const offsets = inputStrides.map((_, index) => {
    const terms = indexShape.map(
      (_, axis) => `coords[${axis}u] * IN${index}_STRIDES[${axis}u]`,
    );
    return `  let offset${index} = ${terms.join(" + ")};`;
  });
  return {
    declarations: [outShapeDecl, ...strideDecls].join("\n"),
    compute,
    offsets,
  };
}

// ============================================================================
// Shape Comparison & Contiguity Checks
// ============================================================================

/**
 * Check if strides represent contiguous memory layout.
 */
export function checkContiguousStrides(shape: number[], strides: number[]): boolean {
  const expected = computeContiguousStrides(shape);
  for (let i = 0; i < shape.length; i++) {
    // Size-1 dims don't affect contiguity
    if (shape[i] <= 1) continue;
    if (strides[i] !== expected[i]) return false;
  }
  return true;
}

// shapesEqual re-exported from core/shape

// ============================================================================
// Dtype Helpers
// ============================================================================

/**
 * Get bytes per element for a dtype.
 */
export function dtypeBytes(dtype: DType): number {
  switch (dtype) {
    case "f16":
      return 2;
    case "f32":
      return 4;
    case "i32":
      return 4;
    case "u32":
      return 4;
    case "bool":
      return 1;
    default:
      return 4;
  }
}

/**
 * Align buffer size to 4 bytes (WebGPU requirement).
 */
export function alignBufferSize(bytes: number): number {
  return Math.ceil(bytes / 4) * 4;
}

/**
 * Convert dtype to WGSL type string.
 */
export function dtypeToWgsl(dtype: DType): string {
  switch (dtype) {
    case "f16":
      return "f16";
    case "f32":
      return "f32";
    case "i32":
      return "i32";
    case "u32":
      return "u32";
    case "bool":
      return "bool";
    default:
      return "f32";
  }
}

/**
 * Convert dtype to WGSL storage type. Like dtypeToWgsl but maps bool to u32
 * since WGSL doesn't support bool in storage arrays.
 */
export function dtypeToWgslStorage(dtype: DType): string {
  if (dtype === "bool") return "u32";
  return dtypeToWgsl(dtype);
}

// ============================================================================
// Dispatch Helpers
// ============================================================================

/**
 * Compute 2D dispatch dimensions for large workloads.
 * WebGPU has a limit of 65535 workgroups per dimension.
 * For large tensors, we use 2D dispatch (x, y).
 */
export function compute2DDispatch(totalWorkgroups: number): {
  x: number;
  y: number;
  gridSizeX: number;
} {
  if (totalWorkgroups <= MAX_WORKGROUPS_PER_DIM) {
    return { x: totalWorkgroups, y: 1, gridSizeX: totalWorkgroups };
  }
  // Split into 2D grid
  const x = MAX_WORKGROUPS_PER_DIM;
  const y = Math.ceil(totalWorkgroups / MAX_WORKGROUPS_PER_DIM);
  return { x, y, gridSizeX: x };
}
