import type { GeluOptions } from "../types";
import { sizeOf, broadcastShapes } from "../../core/shape";

export type Shape = number[];

export class Tensor {
  readonly shape: Shape;
  readonly strides: number[];
  readonly data: Float32Array;
  readonly offset: number;
  private readonly sizeValue: number;

  constructor(
    shape: Shape,
    data: Float32Array,
    strides?: number[],
    offset = 0,
    validateLength = true,
  ) {
    const expected = sizeOf(shape);
    if (validateLength && expected !== data.length) {
      throw new Error("Tensor data length does not match shape");
    }
    this.shape = shape.slice();
    this.strides = (strides ?? computeStrides(shape)).slice();
    this.data = data;
    this.offset = offset;
    this.sizeValue = expected;
  }

  get size(): number {
    return this.sizeValue;
  }

  view(shape: Shape): Tensor {
    const expected = sizeOf(shape);
    if (expected !== this.size) {
      throw new Error("View shape does not match tensor size");
    }
    if (isContiguous(this)) {
      const strides = computeStrides(shape);
      return new Tensor(shape, this.data, strides, this.offset);
    }
    // Non-contiguous: try to compute valid strides
    const newStrides = inferReshapeStrides(this.shape, this.strides, shape);
    if (newStrides !== null) {
      return new Tensor(shape, this.data, newStrides, this.offset, false);
    }
    // Incompatible: materialize first
    const contig = this.contiguous();
    const strides = computeStrides(shape);
    return new Tensor(shape, contig.data, strides, contig.offset);
  }

  /**
   * Check if this tensor has contiguous memory layout.
   * Returns true if strides match row-major (C-style) layout.
   */
  isContiguous(): boolean {
    return isContiguous(this);
  }

  /**
   * Return a contiguous tensor. If already contiguous, returns self.
   * Otherwise, materializes into a new contiguous tensor.
   */
  contiguous(): Tensor {
    if (isContiguous(this)) {
      return this;
    }
    // Materialize to new contiguous storage
    const out = new Float32Array(this.size);
    const shapeStrides = computeStrides(this.shape);
    for (let i = 0; i < this.size; i++) {
      out[i] = readAtLinear(this, i, shapeStrides);
    }
    return new Tensor(this.shape, out);
  }

  toArray(): number[] {
    const out = new Array<number>(this.sizeValue);
    if (this.sizeValue === 0) {
      return out;
    }
    const shapeStrides = computeStrides(this.shape);
    for (let i = 0; i < this.sizeValue; i += 1) {
      out[i] = readAtLinear(this, i, shapeStrides);
    }
    return out;
  }
}

export function tensorFromArray(values: number[] | Float32Array, shape: Shape): Tensor {
  return new Tensor(shape, values instanceof Float32Array ? values.slice() : Float32Array.from(values));
}

export function full(shape: Shape, fillValue: number): Tensor {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(numElements);
  data.fill(fillValue);
  return new Tensor(shape, data);
}

export function arange(end: number, start = 0, step = 1): Tensor {
  const numElements = Math.max(0, Math.ceil((end - start) / step));
  const data = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    data[i] = start + i * step;
  }
  return new Tensor([numElements], data);
}

export function tril(a: Tensor, k = 0): Tensor {
  if (a.shape.length < 2) throw new Error("tril requires at least 2 dimensions");
  const data = Float32Array.from(a.data);
  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const batchSize = data.length / (H * W);
  for (let b = 0; b < batchSize; b++) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (c > r + k) data[b * H * W + r * W + c] = 0;
      }
    }
  }
  return new Tensor(a.shape.slice(), data);
}

export function triu(a: Tensor, k = 0): Tensor {
  if (a.shape.length < 2) throw new Error("triu requires at least 2 dimensions");
  const data = Float32Array.from(a.data);
  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const batchSize = data.length / (H * W);
  for (let b = 0; b < batchSize; b++) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (c < r + k) data[b * H * W + r * W + c] = 0;
      }
    }
  }
  return new Tensor(a.shape.slice(), data);
}

export function expand(a: Tensor, shape: Shape): Tensor {
  return broadcastTo(a, shape);
}

export type SubOptions = {
  alpha?: number;
};

export function sub(a: Tensor, b: Tensor, options?: SubOptions): Tensor {
  const alpha = options?.alpha ?? 1;
  const outShape = broadcastShapes(a.shape, b.shape);
  const aBroadcast = broadcastTo(a, outShape);
  const bBroadcast = broadcastTo(b, outShape);
  const outSize = sizeOf(outShape);
  const out = new Float32Array(outSize);
  const shapeStrides = computeStrides(outShape);
  for (let i = 0; i < outSize; i += 1) {
    out[i] =
      readAtLinear(aBroadcast, i, shapeStrides) -
      alpha * readAtLinear(bBroadcast, i, shapeStrides);
  }
  return new Tensor(outShape, out);
}

export type DivOptions = {
  roundingMode?: "floor" | "trunc" | null;
};

export function div(a: Tensor, b: Tensor, options?: DivOptions): Tensor {
  const rounding = options?.roundingMode ?? null;
  const outShape = broadcastShapes(a.shape, b.shape);
  const aBroadcast = broadcastTo(a, outShape);
  const bBroadcast = broadcastTo(b, outShape);
  const outSize = sizeOf(outShape);
  const out = new Float32Array(outSize);
  const shapeStrides = computeStrides(outShape);
  for (let i = 0; i < outSize; i += 1) {
    let value =
      readAtLinear(aBroadcast, i, shapeStrides) /
      readAtLinear(bBroadcast, i, shapeStrides);
    if (rounding === "floor") {
      value = Math.floor(value);
    } else if (rounding === "trunc") {
      value = Math.trunc(value);
    }
    out[i] = value;
  }
  return new Tensor(outShape, out);
}

export function reshape(a: Tensor, shape: Shape): Tensor {
  return a.view(shape);
}

/**
 * Return a contiguous tensor. If already contiguous, returns the input.
 * Otherwise, materializes into a new contiguous tensor.
 */
export function contiguous(a: Tensor): Tensor {
  return a.contiguous();
}

/**
 * Select a contiguous sub-range along one dimension. Returns a view.
 */
export function narrow(a: Tensor, dim: number, start: number, length: number): Tensor {
  const rank = a.shape.length;
  if (dim < 0 || dim >= rank) {
    throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
  }
  if (start < 0 || start + length > a.shape[dim]) {
    throw new Error(`narrow: range [${start}, ${start + length}) out of bounds for dim size ${a.shape[dim]}`);
  }
  const newShape = a.shape.slice();
  newShape[dim] = length;
  const newOffset = a.offset + start * a.strides[dim];
  return new Tensor(newShape, a.data, a.strides.slice(), newOffset, false);
}

/**
 * Backward for narrow: pad gradient back to original shape.
 */
export function narrowBackward(grad: Tensor, dim: number, start: number, originalLength: number): Tensor {
  const outShape = grad.shape.slice();
  outShape[dim] = originalLength;
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const result = new Float32Array(outSize);

  // Compute indexing
  const outerSize = outShape.slice(0, dim).reduce((a, b) => a * b, 1);
  const innerSize = outShape.slice(dim + 1).reduce((a, b) => a * b, 1);
  const gradDimSize = grad.shape[dim];

  // Copy grad values into the correct slice
  for (let o = 0; o < outerSize; o++) {
    for (let d = 0; d < gradDimSize; d++) {
      for (let i = 0; i < innerSize; i++) {
        const gradIdx = getStridedIndex(grad, o, d, i, dim);
        const outIdx = o * originalLength * innerSize + (start + d) * innerSize + i;
        result[outIdx] = grad.data[gradIdx];
      }
    }
  }

  return new Tensor(outShape, result);
}

function getStridedIndex(t: Tensor, outer: number, dimIdx: number, inner: number, dim: number): number {
  // Compute strided index for [outer, dimIdx, inner] where dim splits the dimensions
  let idx = t.offset;
  // Outer dimensions
  let outerRemain = outer;
  for (let d = dim - 1; d >= 0; d--) {
    idx += (outerRemain % t.shape[d]) * t.strides[d];
    outerRemain = Math.floor(outerRemain / t.shape[d]);
  }
  // Dim dimension
  idx += dimIdx * t.strides[dim];
  // Inner dimensions
  let innerRemain = inner;
  for (let d = t.shape.length - 1; d > dim; d--) {
    idx += (innerRemain % t.shape[d]) * t.strides[d];
    innerRemain = Math.floor(innerRemain / t.shape[d]);
  }
  return idx;
}

export type TransposeOptions = {
  dim0: number;
  dim1: number;
};

export function transpose(a: Tensor, options: TransposeOptions): Tensor {
  if (!options) {
    throw new Error("transpose requires options.dim0 and options.dim1");
  }
  const rank = a.shape.length;
  const dim0 = normalizeDim(options.dim0, rank);
  const dim1 = normalizeDim(options.dim1, rank);
  if (dim0 === dim1) {
    return a;
  }
  const shape = a.shape.slice();
  const strides = a.strides.slice();
  [shape[dim0], shape[dim1]] = [shape[dim1], shape[dim0]];
  [strides[dim0], strides[dim1]] = [strides[dim1], strides[dim0]];
  return new Tensor(shape, a.data, strides, a.offset, false);
}

/**
 * Permute dimensions according to the given order.
 * Returns a view sharing the same data (no copy).
 */
export function permute(a: Tensor, dims: number[]): Tensor {
  const rank = a.shape.length;

  if (dims.length !== rank) {
    throw new Error(
      `permute: dims length ${dims.length} doesn't match tensor rank ${rank}`,
    );
  }

  // Check for valid permutation
  const seen = new Set<number>();
  for (const d of dims) {
    const nd = d < 0 ? d + rank : d;
    if (nd < 0 || nd >= rank) {
      throw new Error(`permute: dimension ${d} out of range for rank ${rank}`);
    }
    if (seen.has(nd)) {
      throw new Error(`permute: duplicate dimension ${d}`);
    }
    seen.add(nd);
  }

  // Normalize dims
  const normalizedDims = dims.map((d) => (d < 0 ? d + rank : d));

  // Reorder shape and strides
  const newShape = normalizedDims.map((d) => a.shape[d]);
  const newStrides = normalizedDims.map((d) => a.strides[d]);

  return new Tensor(newShape, a.data, newStrides, a.offset, false);
}

export function add(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const aBroadcast = broadcastTo(a, outShape);
  const bBroadcast = broadcastTo(b, outShape);
  const outSize = sizeOf(outShape);
  const out = new Float32Array(outSize);
  const shapeStrides = computeStrides(outShape);
  for (let i = 0; i < outSize; i += 1) {
    out[i] =
      readAtLinear(aBroadcast, i, shapeStrides) +
      readAtLinear(bBroadcast, i, shapeStrides);
  }
  return new Tensor(outShape, out);
}

export function mul(a: Tensor, b: Tensor): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const aBroadcast = broadcastTo(a, outShape);
  const bBroadcast = broadcastTo(b, outShape);
  const outSize = sizeOf(outShape);
  const out = new Float32Array(outSize);
  const shapeStrides = computeStrides(outShape);
  for (let i = 0; i < outSize; i += 1) {
    out[i] =
      readAtLinear(aBroadcast, i, shapeStrides) *
      readAtLinear(bBroadcast, i, shapeStrides);
  }
  return new Tensor(outShape, out);
}

export function relu(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    const value = readAtLinear(a, i, shapeStrides);
    out[i] = value > 0 ? value : 0;
  }
  return new Tensor(a.shape, out);
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  const aRank = a.shape.length;
  const bRank = b.shape.length;
  if (aRank === 0 || bRank === 0) {
    throw new Error("matmul does not support scalar inputs");
  }

  const aWas1d = aRank === 1;
  const bWas1d = bRank === 1;
  const aMatrix = aWas1d
    ? new Tensor([1, a.shape[0]], a.data, [0, a.strides[0]], a.offset, false)
    : a;
  const bMatrix = bWas1d
    ? new Tensor([b.shape[0], 1], b.data, [b.strides[0], 0], b.offset, false)
    : b;

  if (aMatrix.shape.length < 2 || bMatrix.shape.length < 2) {
    throw new Error("matmul requires tensors with at least 1 dimension");
  }

  const m = aMatrix.shape[aMatrix.shape.length - 2];
  const k = aMatrix.shape[aMatrix.shape.length - 1];
  const kB = bMatrix.shape[bMatrix.shape.length - 2];
  const n = bMatrix.shape[bMatrix.shape.length - 1];
  if (k !== kB) {
    throw new Error("matmul dimension mismatch");
  }

  const aBatch = aMatrix.shape.slice(0, -2);
  const bBatch = bMatrix.shape.slice(0, -2);
  const batchShape = broadcastShapes(aBatch, bBatch);
  const aBroadcast = broadcastTo(aMatrix, batchShape.concat([m, k]));
  const bBroadcast = broadcastTo(bMatrix, batchShape.concat([k, n]));

  const batchSize = sizeOf(batchShape);
  const out = new Float32Array(batchSize * m * n);
  const batchRank = batchShape.length;
  const batchShapeStrides = computeStrides(batchShape);
  const aBatchStrides = aBroadcast.strides.slice(0, batchRank);
  const bBatchStrides = bBroadcast.strides.slice(0, batchRank);
  const aRowStride = aBroadcast.strides[batchRank];
  const aInnerStride = aBroadcast.strides[batchRank + 1];
  const bInnerStride = bBroadcast.strides[batchRank];
  const bColStride = bBroadcast.strides[batchRank + 1];

  for (let batch = 0; batch < batchSize; batch += 1) {
    const aBase = linearOffset(
      batch,
      batchShapeStrides,
      aBatchStrides,
      aBroadcast.offset,
    );
    const bBase = linearOffset(
      batch,
      batchShapeStrides,
      bBatchStrides,
      bBroadcast.offset,
    );
    const outBase = batch * m * n;
    for (let row = 0; row < m; row += 1) {
      const rowOffset = aBase + row * aRowStride;
      const outOffset = outBase + row * n;
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        for (let inner = 0; inner < k; inner += 1) {
          acc +=
            aBroadcast.data[rowOffset + inner * aInnerStride] *
            bBroadcast.data[bBase + inner * bInnerStride + col * bColStride];
        }
        out[outOffset + col] = acc;
      }
    }
  }

  const outShape = batchShape.concat([m, n]);
  if (aWas1d && bWas1d) {
    return new Tensor([], out);
  }
  if (aWas1d) {
    return new Tensor(batchShape.concat([n]), out);
  }
  if (bWas1d) {
    return new Tensor(batchShape.concat([m]), out);
  }
  return new Tensor(outShape, out);
}

export function sqrt(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = Math.sqrt(readAtLinear(a, i, shapeStrides));
  }
  return new Tensor(a.shape, out);
}

export function exp(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = Math.exp(readAtLinear(a, i, shapeStrides));
  }
  return new Tensor(a.shape, out);
}

export function log(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = Math.log(readAtLinear(a, i, shapeStrides));
  }
  return new Tensor(a.shape, out);
}

export function neg(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = -readAtLinear(a, i, shapeStrides);
  }
  return new Tensor(a.shape, out);
}

export function abs(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = Math.abs(readAtLinear(a, i, shapeStrides));
  }
  return new Tensor(a.shape, out);
}

export function tanh(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = Math.tanh(readAtLinear(a, i, shapeStrides));
  }
  return new Tensor(a.shape, out);
}

export function sigmoid(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    const x = readAtLinear(a, i, shapeStrides);
    out[i] = 1.0 / (1.0 + Math.exp(-x));
  }
  return new Tensor(a.shape, out);
}

/**
 * Error function approximation using Horner's method.
 * Maximum error ~1.5e-7 in the range [-∞, +∞].
 */
function erf(x: number): number {
  // Abramowitz and Stegun approximation 7.1.26
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return sign * y;
}

export function gelu(a: Tensor, options?: GeluOptions): Tensor {
  const approximate = options?.approximate ?? "tanh";
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);

  if (approximate === "tanh") {
    // Tanh approximation (GPT-2 "new GELU"):
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const sqrt2OverPi = 0.7978845608; // sqrt(2/pi)
    for (let i = 0; i < a.size; i += 1) {
      const x = readAtLinear(a, i, shapeStrides);
      out[i] = x * 0.5 * (1.0 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x * x * x)));
    }
  } else {
    // Exact formula using erf:
    // x * 0.5 * (1 + erf(x / sqrt(2)))
    const sqrt2Inv = 0.7071067811865476; // 1/sqrt(2)
    for (let i = 0; i < a.size; i += 1) {
      const x = readAtLinear(a, i, shapeStrides);
      out[i] = x * 0.5 * (1.0 + erf(x * sqrt2Inv));
    }
  }

  return new Tensor(a.shape, out);
}

export function silu(a: Tensor): Tensor {
  // SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    const x = readAtLinear(a, i, shapeStrides);
    out[i] = x / (1.0 + Math.exp(-x));
  }
  return new Tensor(a.shape, out);
}

/**
 * Check if values are finite (not NaN and not Inf).
 * Returns 1.0 where finite, 0.0 where NaN or Inf.
 */
export function isfinite(a: Tensor): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    const x = readAtLinear(a, i, shapeStrides);
    out[i] = Number.isFinite(x) ? 1.0 : 0.0;
  }
  return new Tensor(a.shape, out);
}

export type SumOptions = {
  dim?: number | number[] | null;
  keepdim?: boolean;
  dtype?: "f16" | "f32" | "i32" | "u32" | "bool" | null;
};

export type MeanOptions = SumOptions;

function computeStrides(shape: Shape): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i -= 1) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// sizeOf imported from core/shape

/**
 * Infer strides for a new shape given old shape/strides, without copying data.
 * Returns null if the reshape requires a contiguous copy.
 */
function inferReshapeStrides(
  oldShape: number[],
  oldStrides: number[],
  newShape: number[],
): number[] | null {
  if (newShape.length === 0) return [];
  if (oldShape.length === 0) return computeStrides(newShape);

  const newStrides = new Array<number>(newShape.length);
  let oldIdx = 0;
  let newIdx = 0;
  const oldN = oldShape.length;
  const newN = newShape.length;

  while (newIdx < newN) {
    if (newShape[newIdx] === 1) {
      newStrides[newIdx] = newIdx + 1 < newN ? newStrides[newIdx + 1] || 1 : 1;
      newIdx++;
      continue;
    }
    while (oldIdx < oldN && oldShape[oldIdx] === 1) oldIdx++;
    if (oldIdx >= oldN) return null;

    let oldProduct = oldShape[oldIdx];
    let newProduct = newShape[newIdx];
    const groupStart = oldIdx;
    while (oldProduct < newProduct && oldIdx + 1 < oldN) {
      if (oldStrides[oldIdx] !== oldStrides[oldIdx + 1] * oldShape[oldIdx + 1]) {
        return null;
      }
      oldIdx++;
      if (oldShape[oldIdx] === 1) continue;
      oldProduct *= oldShape[oldIdx];
    }
    const newGroupStart = newIdx;
    while (newProduct < oldProduct && newIdx + 1 < newN) {
      newIdx++;
      if (newShape[newIdx] === 1) {
        newStrides[newIdx] = 1;
        continue;
      }
      newProduct *= newShape[newIdx];
    }
    if (oldProduct !== newProduct) return null;

    let stride = oldStrides[oldIdx];
    for (let i = newIdx; i >= newGroupStart; i--) {
      if (newShape[i] === 1) {
        newStrides[i] = stride;
        continue;
      }
      newStrides[i] = stride;
      stride *= newShape[i];
    }
    oldIdx++;
    newIdx++;
  }
  while (oldIdx < oldN) {
    if (oldShape[oldIdx] !== 1) return null;
    oldIdx++;
  }
  return newStrides;
}

function isContiguous(tensor: Tensor): boolean {
  const expected = computeStrides(tensor.shape);
  if (expected.length !== tensor.strides.length) {
    return false;
  }
  for (let axis = 0; axis < expected.length; axis += 1) {
    if (tensor.shape[axis] <= 1) {
      continue;
    }
    if (tensor.strides[axis] !== expected[axis]) {
      return false;
    }
  }
  return true;
}

function linearOffset(
  linear: number,
  shapeStrides: number[],
  strides: number[],
  baseOffset: number,
): number {
  let remainder = linear;
  let offset = baseOffset;
  for (let axis = 0; axis < shapeStrides.length; axis += 1) {
    const stride = shapeStrides[axis];
    const coord = stride === 0 ? 0 : Math.floor(remainder / stride);
    remainder -= coord * stride;
    offset += coord * strides[axis];
  }
  return offset;
}

function readAtLinear(
  tensor: Tensor,
  linear: number,
  shapeStrides: number[],
): number {
  const offset = linearOffset(
    linear,
    shapeStrides,
    tensor.strides,
    tensor.offset,
  );
  return tensor.data[offset];
}

// broadcastShapes imported from core/shape

function broadcastTo(a: Tensor, targetShape: Shape): Tensor {
  if (
    a.shape.length === targetShape.length &&
    a.shape.every((dim, index) => dim === targetShape[index])
  ) {
    return a;
  }

  if (a.shape.length > targetShape.length) {
    throw new Error("broadcast target has fewer dimensions than input");
  }

  const pad = targetShape.length - a.shape.length;
  const outStrides = new Array<number>(targetShape.length);

  for (let axis = 0; axis < targetShape.length; axis += 1) {
    const inAxis = axis - pad;
    if (inAxis < 0) {
      outStrides[axis] = 0;
      continue;
    }
    const inDim = a.shape[inAxis];
    const outDim = targetShape[axis];
    if (inDim === outDim) {
      outStrides[axis] = a.strides[inAxis];
    } else if (inDim === 1) {
      outStrides[axis] = 0;
    } else {
      throw new Error("broadcast target shape is incompatible");
    }
  }

  return new Tensor(targetShape, a.data, outStrides, a.offset, false);
}

function normalizeDim(dim: number, rank: number): number {
  const normalized = dim < 0 ? rank + dim : dim;
  if (normalized < 0 || normalized >= rank) {
    throw new Error(`dim out of range: ${dim}`);
  }
  return normalized;
}

function readIndexValue(value: number, limit: number): number {
  if (!Number.isFinite(value) || Math.trunc(value) !== value) {
    throw new Error("index values must be integers");
  }
  if (value < 0 || value >= limit) {
    throw new Error("index out of range");
  }
  return value;
}

export type GatherOptions = {
  dim: number;
};

export function gather(
  a: Tensor,
  index: Tensor,
  options: GatherOptions,
): Tensor {
  if (!options) {
    throw new Error("gather requires options.dim");
  }
  const rank = a.shape.length;
  if (index.shape.length !== rank) {
    throw new Error("gather requires index with the same rank as input");
  }
  const dim = normalizeDim(options.dim, rank);
  for (let i = 0; i < rank; i += 1) {
    if (i !== dim && index.shape[i] > a.shape[i]) {
      throw new Error("gather index shape must be <= input shape");
    }
  }

  const out = new Float32Array(index.size);
  const indexShapeStrides = computeStrides(index.shape);
  const inputStrides = a.strides;

  for (let linear = 0; linear < index.size; linear += 1) {
    let remainder = linear;
    let inputOffset = 0;

    for (let axis = 0; axis < rank; axis += 1) {
      const stride = indexShapeStrides[axis];
      const coord = Math.floor(remainder / stride);
      remainder -= coord * stride;

      if (axis === dim) {
        const idx = readIndexValue(
          readAtLinear(index, linear, indexShapeStrides),
          a.shape[dim],
        );
        inputOffset += idx * inputStrides[axis];
      } else {
        inputOffset += coord * inputStrides[axis];
      }
    }

    out[linear] = a.data[inputOffset + a.offset];
  }

  return new Tensor(index.shape, out);
}

export type ScatterAddOptions = {
  dim: number;
};

export function scatterAdd(
  a: Tensor,
  index: Tensor,
  src: Tensor,
  options: ScatterAddOptions,
): Tensor {
  if (!options) {
    throw new Error("scatterAdd requires options.dim");
  }
  const rank = a.shape.length;
  if (index.shape.length !== rank || src.shape.length !== rank) {
    throw new Error(
      "scatterAdd requires index/src with the same rank as input",
    );
  }
  const dim = normalizeDim(options.dim, rank);
  for (let i = 0; i < rank; i += 1) {
    if (index.shape[i] !== src.shape[i]) {
      throw new Error(
        "scatterAdd requires index and src to have the same shape",
      );
    }
    if (i !== dim && index.shape[i] > a.shape[i]) {
      throw new Error("scatterAdd index shape must be <= input shape");
    }
  }

  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = readAtLinear(a, i, shapeStrides);
  }
  const indexShapeStrides = computeStrides(index.shape);
  const outStrides = computeStrides(a.shape);

  for (let linear = 0; linear < index.size; linear += 1) {
    let remainder = linear;
    let outputOffset = 0;

    for (let axis = 0; axis < rank; axis += 1) {
      const stride = indexShapeStrides[axis];
      const coord = Math.floor(remainder / stride);
      remainder -= coord * stride;

      if (axis === dim) {
        const idx = readIndexValue(
          readAtLinear(index, linear, indexShapeStrides),
          a.shape[dim],
        );
        outputOffset += idx * outStrides[axis];
      } else {
        outputOffset += coord * outStrides[axis];
      }
    }

    out[outputOffset] += readAtLinear(src, linear, indexShapeStrides);
  }

  return new Tensor(a.shape, out);
}

function sumAll(a: Tensor): number {
  let total = 0;
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    total += readAtLinear(a, i, shapeStrides);
  }
  return total;
}

function normalizeDims(dim: number | number[], rank: number): number[] {
  const dims = Array.isArray(dim) ? dim.slice() : [dim];
  const normalized = dims.map((value) => (value < 0 ? rank + value : value));
  const unique = new Set<number>();
  for (const value of normalized) {
    if (value < 0 || value >= rank) {
      throw new Error(`sum dim out of range: ${value}`);
    }
    if (unique.has(value)) {
      throw new Error(`sum dim repeated: ${value}`);
    }
    unique.add(value);
  }
  return Array.from(unique).sort((a, b) => a - b);
}

export function sum(a: Tensor, options?: SumOptions): Tensor {
  if (!options || options.dim == null) {
    // Full reduction - return 0-d tensor
    return new Tensor([], new Float32Array([sumAll(a)]));
  }

  if (options.dtype != null) {
    throw new Error("sum dtype option is not supported yet");
  }

  const rank = a.shape.length;
  const dims = normalizeDims(options.dim, rank);
  const keepdim = options.keepdim ?? false;
  const reduceSet = new Set(dims);

  const outShape = keepdim
    ? a.shape.map((dim, index) => (reduceSet.has(index) ? 1 : dim))
    : a.shape.filter((_, index) => !reduceSet.has(index));

  const outSize = outShape.reduce((acc, dim) => acc * dim, 1) || 1;
  const out = new Float32Array(outSize);
  const inShapeStrides = computeStrides(a.shape);
  const outStrides = computeStrides(outShape);

  for (let linear = 0; linear < a.size; linear += 1) {
    let remainder = linear;
    let outOffset = 0;

    if (keepdim) {
      for (let dim = 0; dim < rank; dim += 1) {
        const stride = inShapeStrides[dim];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (!reduceSet.has(dim)) {
          outOffset += coord * outStrides[dim];
        }
      }
    } else {
      let outDim = 0;
      for (let dim = 0; dim < rank; dim += 1) {
        const stride = inShapeStrides[dim];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (!reduceSet.has(dim)) {
          outOffset += coord * outStrides[outDim];
          outDim += 1;
        }
      }
    }

    out[outOffset] += readAtLinear(a, linear, inShapeStrides);
  }

  return new Tensor(outShape, out);
}

function maxAll(a: Tensor): number {
  let maxVal = -Infinity;
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    const val = readAtLinear(a, i, shapeStrides);
    if (val > maxVal) {
      maxVal = val;
    }
  }
  return maxVal;
}

export type MaxOptions = { dim?: number | number[] | null; keepdim?: boolean };

export function max(a: Tensor, options?: MaxOptions): Tensor {
  if (!options || options.dim == null) {
    // Full reduction - return 0-d tensor
    return new Tensor([], new Float32Array([maxAll(a)]));
  }

  const rank = a.shape.length;
  const dims = normalizeDimsForReduce(options.dim, rank, "max");
  const keepdim = options.keepdim ?? false;
  const reduceSet = new Set(dims);

  const outShape = keepdim
    ? a.shape.map((dim, index) => (reduceSet.has(index) ? 1 : dim))
    : a.shape.filter((_, index) => !reduceSet.has(index));

  const outSize = outShape.reduce((acc, dim) => acc * dim, 1) || 1;
  const out = new Float32Array(outSize);
  out.fill(-Infinity);
  const inShapeStrides = computeStrides(a.shape);
  const outStrides = computeStrides(outShape);

  for (let linear = 0; linear < a.size; linear += 1) {
    let remainder = linear;
    let outOffset = 0;

    if (keepdim) {
      for (let dim = 0; dim < rank; dim += 1) {
        const stride = inShapeStrides[dim];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (!reduceSet.has(dim)) {
          outOffset += coord * outStrides[dim];
        }
      }
    } else {
      let outDim = 0;
      for (let dim = 0; dim < rank; dim += 1) {
        const stride = inShapeStrides[dim];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (!reduceSet.has(dim)) {
          outOffset += coord * outStrides[outDim];
          outDim += 1;
        }
      }
    }

    const val = readAtLinear(a, linear, inShapeStrides);
    if (val > out[outOffset]) {
      out[outOffset] = val;
    }
  }

  return new Tensor(outShape, out);
}

function normalizeDimsForReduce(dim: number | number[], rank: number, opName: string): number[] {
  const dims = Array.isArray(dim) ? dim.slice() : [dim];
  const normalized = dims.map((value) => (value < 0 ? rank + value : value));
  const unique = new Set<number>();
  for (const value of normalized) {
    if (value < 0 || value >= rank) {
      throw new Error(`${opName} dim out of range: ${value}`);
    }
    if (unique.has(value)) {
      throw new Error(`${opName} dim repeated: ${value}`);
    }
    unique.add(value);
  }
  return Array.from(unique).sort((a, b) => a - b);
}

export type ArgReduceOptions = { dim: number; keepdim?: boolean };

export function argmax(a: Tensor, options: ArgReduceOptions): Tensor {
  const rank = a.shape.length;
  const dim = options.dim < 0 ? options.dim + rank : options.dim;
  if (dim < 0 || dim >= rank) {
    throw new Error(`argmax: dim ${options.dim} out of range for tensor of rank ${rank}`);
  }
  const keepdim = options.keepdim ?? false;

  const outShape = keepdim
    ? a.shape.map((d, i) => (i === dim ? 1 : d))
    : a.shape.filter((_, i) => i !== dim);

  const outSize = outShape.reduce((acc, d) => acc * d, 1) || 1;
  const out = new Float32Array(outSize);
  const maxVals = new Float32Array(outSize);
  maxVals.fill(-Infinity);

  const inShapeStrides = computeStrides(a.shape);
  const outStrides = computeStrides(outShape);
  const dimSize = a.shape[dim];

  for (let linear = 0; linear < a.size; linear += 1) {
    let remainder = linear;
    let outOffset = 0;
    let dimCoord = 0;

    if (keepdim) {
      for (let d = 0; d < rank; d += 1) {
        const stride = inShapeStrides[d];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (d === dim) {
          dimCoord = coord;
        } else {
          outOffset += coord * outStrides[d];
        }
      }
    } else {
      let outDim = 0;
      for (let d = 0; d < rank; d += 1) {
        const stride = inShapeStrides[d];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (d === dim) {
          dimCoord = coord;
        } else {
          outOffset += coord * outStrides[outDim];
          outDim += 1;
        }
      }
    }

    const val = readAtLinear(a, linear, inShapeStrides);
    if (val > maxVals[outOffset]) {
      maxVals[outOffset] = val;
      out[outOffset] = dimCoord;
    }
  }

  return new Tensor(outShape, out);
}

export function argmin(a: Tensor, options: ArgReduceOptions): Tensor {
  const rank = a.shape.length;
  const dim = options.dim < 0 ? options.dim + rank : options.dim;
  if (dim < 0 || dim >= rank) {
    throw new Error(`argmin: dim ${options.dim} out of range for tensor of rank ${rank}`);
  }
  const keepdim = options.keepdim ?? false;

  const outShape = keepdim
    ? a.shape.map((d, i) => (i === dim ? 1 : d))
    : a.shape.filter((_, i) => i !== dim);

  const outSize = outShape.reduce((acc, d) => acc * d, 1) || 1;
  const out = new Float32Array(outSize);
  const minVals = new Float32Array(outSize);
  minVals.fill(Infinity);

  const inShapeStrides = computeStrides(a.shape);
  const outStrides = computeStrides(outShape);

  for (let linear = 0; linear < a.size; linear += 1) {
    let remainder = linear;
    let outOffset = 0;
    let dimCoord = 0;

    if (keepdim) {
      for (let d = 0; d < rank; d += 1) {
        const stride = inShapeStrides[d];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (d === dim) {
          dimCoord = coord;
        } else {
          outOffset += coord * outStrides[d];
        }
      }
    } else {
      let outDim = 0;
      for (let d = 0; d < rank; d += 1) {
        const stride = inShapeStrides[d];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (d === dim) {
          dimCoord = coord;
        } else {
          outOffset += coord * outStrides[outDim];
          outDim += 1;
        }
      }
    }

    const val = readAtLinear(a, linear, inShapeStrides);
    if (val < minVals[outOffset]) {
      minVals[outOffset] = val;
      out[outOffset] = dimCoord;
    }
  }

  return new Tensor(outShape, out);
}

// ============================================================================
// Comparison ops - return 1.0 for true, 0.0 for false
// ============================================================================

export function gt(a: Tensor, b: Tensor): Tensor {
  const { aBroadcast, bBroadcast, outShape, outSize, shapeStrides } = broadcastBinary(a, b);
  const out = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const aVal = readAtLinear(aBroadcast, i, shapeStrides);
    const bVal = readAtLinear(bBroadcast, i, shapeStrides);
    out[i] = aVal > bVal ? 1.0 : 0.0;
  }
  return new Tensor(outShape, out);
}

export function lt(a: Tensor, b: Tensor): Tensor {
  const { aBroadcast, bBroadcast, outShape, outSize, shapeStrides } = broadcastBinary(a, b);
  const out = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const aVal = readAtLinear(aBroadcast, i, shapeStrides);
    const bVal = readAtLinear(bBroadcast, i, shapeStrides);
    out[i] = aVal < bVal ? 1.0 : 0.0;
  }
  return new Tensor(outShape, out);
}

export function ge(a: Tensor, b: Tensor): Tensor {
  const { aBroadcast, bBroadcast, outShape, outSize, shapeStrides } = broadcastBinary(a, b);
  const out = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const aVal = readAtLinear(aBroadcast, i, shapeStrides);
    const bVal = readAtLinear(bBroadcast, i, shapeStrides);
    out[i] = aVal >= bVal ? 1.0 : 0.0;
  }
  return new Tensor(outShape, out);
}

export function le(a: Tensor, b: Tensor): Tensor {
  const { aBroadcast, bBroadcast, outShape, outSize, shapeStrides } = broadcastBinary(a, b);
  const out = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const aVal = readAtLinear(aBroadcast, i, shapeStrides);
    const bVal = readAtLinear(bBroadcast, i, shapeStrides);
    out[i] = aVal <= bVal ? 1.0 : 0.0;
  }
  return new Tensor(outShape, out);
}

export function eq(a: Tensor, b: Tensor): Tensor {
  const { aBroadcast, bBroadcast, outShape, outSize, shapeStrides } = broadcastBinary(a, b);
  const out = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const aVal = readAtLinear(aBroadcast, i, shapeStrides);
    const bVal = readAtLinear(bBroadcast, i, shapeStrides);
    out[i] = aVal === bVal ? 1.0 : 0.0;
  }
  return new Tensor(outShape, out);
}

export function ne(a: Tensor, b: Tensor): Tensor {
  const { aBroadcast, bBroadcast, outShape, outSize, shapeStrides } = broadcastBinary(a, b);
  const out = new Float32Array(outSize);
  for (let i = 0; i < outSize; i += 1) {
    const aVal = readAtLinear(aBroadcast, i, shapeStrides);
    const bVal = readAtLinear(bBroadcast, i, shapeStrides);
    out[i] = aVal !== bVal ? 1.0 : 0.0;
  }
  return new Tensor(outShape, out);
}

/** Helper for binary broadcast */
function broadcastBinary(a: Tensor, b: Tensor) {
  const outShape = broadcastShapes(a.shape, b.shape);
  const aBroadcast = broadcastTo(a, outShape);
  const bBroadcast = broadcastTo(b, outShape);
  const outSize = sizeOf(outShape);
  const shapeStrides = computeStrides(outShape);
  return { aBroadcast, bBroadcast, outShape, outSize, shapeStrides };
}

/**
 * where(condition, x, y): returns x where condition is true (non-zero), else y.
 */
export function where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
  // Broadcast all three shapes together
  const outShape = broadcastThreeShapes(condition.shape, x.shape, y.shape);
  const condBroadcast = broadcastTo(condition, outShape);
  const xBroadcast = broadcastTo(x, outShape);
  const yBroadcast = broadcastTo(y, outShape);
  const outSize = sizeOf(outShape);
  const out = new Float32Array(outSize);
  const shapeStrides = computeStrides(outShape);

  for (let i = 0; i < outSize; i += 1) {
    const condVal = readAtLinear(condBroadcast, i, shapeStrides);
    const xVal = readAtLinear(xBroadcast, i, shapeStrides);
    const yVal = readAtLinear(yBroadcast, i, shapeStrides);
    out[i] = condVal !== 0 ? xVal : yVal;
  }

  return new Tensor(outShape, out);
}

/**
 * Broadcast three shapes together.
 */
function broadcastThreeShapes(a: Shape, b: Shape, c: Shape): Shape {
  const outRank = Math.max(a.length, b.length, c.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    const cDim = c[c.length - 1 - i] ?? 1;
    // Check all pairs for broadcast compatibility
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error("shapes are not broadcastable");
    }
    if (aDim !== cDim && aDim !== 1 && cDim !== 1) {
      throw new Error("shapes are not broadcastable");
    }
    if (bDim !== cDim && bDim !== 1 && cDim !== 1) {
      throw new Error("shapes are not broadcastable");
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim, cDim);
  }
  return out;
}

/**
 * Options for strided scatter operations.
 * Describes where within a base tensor to write values.
 */
export type StridedScatterOptions = {
  /** Element offset into the base tensor */
  offset: number;
  /** Shape of the view (must match src shape) */
  viewShape: number[];
  /** Element strides for the view */
  viewStrides: number[];
};

/**
 * Copy src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 *
 * This is the fundamental operation for view mutation lowering (§4.4):
 * When mutating a view, we load the base, scatter-copy the new values
 * into the view positions, and store back.
 */
export function stridedScatterCopy(
  base: Tensor,
  src: Tensor,
  options: StridedScatterOptions,
): Tensor {
  const { offset, viewShape, viewStrides } = options;

  // Validate src shape matches view shape
  if (src.shape.length !== viewShape.length) {
    throw new Error(
      `stridedScatterCopy: src rank ${src.shape.length} doesn't match view rank ${viewShape.length}`,
    );
  }
  for (let i = 0; i < viewShape.length; i++) {
    if (src.shape[i] !== viewShape[i]) {
      throw new Error(
        `stridedScatterCopy: src shape [${src.shape}] doesn't match view shape [${viewShape}]`,
      );
    }
  }

  // Clone base data (we don't mutate the original)
  const out = new Float32Array(base.data.length);
  out.set(base.data);

  // Iterate over src and write each value to the corresponding position in base
  const srcSize = sizeOf(viewShape);
  const srcShapeStrides = computeStrides(viewShape);

  for (let linear = 0; linear < srcSize; linear++) {
    // Convert linear index to coordinates in view shape
    let remainder = linear;
    let baseOffset = offset;
    for (let axis = 0; axis < viewShape.length; axis++) {
      const stride = srcShapeStrides[axis];
      const coord = Math.floor(remainder / stride);
      remainder -= coord * stride;
      // Map to base using view strides
      baseOffset += coord * viewStrides[axis];
    }

    // Read from src (handling src's own strides)
    const srcVal = readAtLinear(src, linear, srcShapeStrides);
    out[baseOffset] = srcVal;
  }

  return new Tensor(base.shape, out, computeStrides(base.shape), 0);
}

/**
 * Add src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 */
export function stridedScatterAdd(
  base: Tensor,
  src: Tensor,
  options: StridedScatterOptions,
): Tensor {
  const { offset, viewShape, viewStrides } = options;

  // Validate src shape matches view shape
  if (src.shape.length !== viewShape.length) {
    throw new Error(
      `stridedScatterAdd: src rank ${src.shape.length} doesn't match view rank ${viewShape.length}`,
    );
  }
  for (let i = 0; i < viewShape.length; i++) {
    if (src.shape[i] !== viewShape[i]) {
      throw new Error(
        `stridedScatterAdd: src shape [${src.shape}] doesn't match view shape [${viewShape}]`,
      );
    }
  }

  // Clone base data (we don't mutate the original)
  const out = new Float32Array(base.data.length);
  out.set(base.data);

  // Iterate over src and add each value to the corresponding position in base
  const srcSize = sizeOf(viewShape);
  const srcShapeStrides = computeStrides(viewShape);

  for (let linear = 0; linear < srcSize; linear++) {
    // Convert linear index to coordinates in view shape
    let remainder = linear;
    let baseOffset = offset;
    for (let axis = 0; axis < viewShape.length; axis++) {
      const stride = srcShapeStrides[axis];
      const coord = Math.floor(remainder / stride);
      remainder -= coord * stride;
      // Map to base using view strides
      baseOffset += coord * viewStrides[axis];
    }

    // Read from src (handling src's own strides)
    const srcVal = readAtLinear(src, linear, srcShapeStrides);
    out[baseOffset] += srcVal;
  }

  return new Tensor(base.shape, out, computeStrides(base.shape), 0);
}

export function mean(a: Tensor, options?: MeanOptions): Tensor {
  if (!options || options.dim == null) {
    // Full reduction - return 0-d tensor
    return new Tensor([], new Float32Array([sumAll(a) / a.size]));
  }

  if (options.dtype != null) {
    throw new Error("mean dtype option is not supported yet");
  }

  const rank = a.shape.length;
  const dims = normalizeDims(options.dim, rank);
  const keepdim = options.keepdim ?? false;
  const reduceSet = new Set(dims);
  const reduceCount = dims.reduce((acc, dim) => acc * a.shape[dim], 1);

  const outShape = keepdim
    ? a.shape.map((dim, index) => (reduceSet.has(index) ? 1 : dim))
    : a.shape.filter((_, index) => !reduceSet.has(index));

  const outSize = outShape.reduce((acc, dim) => acc * dim, 1) || 1;
  const out = new Float32Array(outSize);
  const inShapeStrides = computeStrides(a.shape);
  const outStrides = computeStrides(outShape);

  for (let linear = 0; linear < a.size; linear += 1) {
    let remainder = linear;
    let outOffset = 0;

    if (keepdim) {
      for (let dim = 0; dim < rank; dim += 1) {
        const stride = inShapeStrides[dim];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (!reduceSet.has(dim)) {
          outOffset += coord * outStrides[dim];
        }
      }
    } else {
      let outDim = 0;
      for (let dim = 0; dim < rank; dim += 1) {
        const stride = inShapeStrides[dim];
        const coord = Math.floor(remainder / stride);
        remainder -= coord * stride;
        if (!reduceSet.has(dim)) {
          outOffset += coord * outStrides[outDim];
          outDim += 1;
        }
      }
    }

    out[outOffset] += readAtLinear(a, linear, inShapeStrides);
  }

  const denom = reduceCount || 0;
  for (let i = 0; i < out.length; i += 1) {
    out[i] = out[i] / denom;
  }

  return new Tensor(outShape, out);
}
