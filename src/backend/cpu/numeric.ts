import {
  broadcastShapes,
  broadcastThreeShapes,
  contiguousStrides as computeStrides,
  inferReshapeStrides,
  sizeOf,
} from "../../core/shape";
import type { DType } from "../types";
import { type GeluOptions, normalizeDim as normalizeDimBase } from "../types";

export type Shape = number[];

export class Tensor {
  readonly shape: Shape;
  readonly strides: number[];
  readonly data: Float32Array;
  readonly offset: number;
  readonly dtype: DType;
  private readonly sizeValue: number;

  constructor(
    shape: Shape,
    data: Float32Array,
    strides?: number[],
    offset = 0,
    validateLength = true,
    dtype: DType = "f32",
  ) {
    const expected = sizeOf(shape);
    if (validateLength && expected !== data.length) {
      throw new Error("Tensor data length does not match shape");
    }
    this.shape = shape.slice();
    this.strides = (strides ?? computeStrides(shape)).slice();
    this.data = data;
    this.offset = offset;
    this.dtype = dtype;
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
      return new Tensor(shape, this.data, strides, this.offset, false);
    }
    // Non-contiguous: try to compute valid strides
    const newStrides = inferReshapeStrides(this.shape, this.strides, shape);
    if (newStrides !== null) {
      return new Tensor(shape, this.data, newStrides, this.offset, false);
    }
    // Incompatible: materialize first
    const contig = this.contiguous();
    const strides = computeStrides(shape);
    return new Tensor(shape, contig.data, strides, contig.offset, false);
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

export function tensorFromArray(
  values: number[] | Float32Array | Int32Array | Uint32Array,
  shape: Shape,
  dtype: DType = "f32",
): Tensor {
  const f32 =
    values instanceof Float32Array ? values.slice() : Float32Array.from(values);
  return new Tensor(shape, f32, undefined, 0, true, dtype);
}

export function zeros(shape: Shape, dtype: DType = "f32"): Tensor {
  return new Tensor(
    shape,
    new Float32Array(sizeOf(shape)),
    undefined,
    0,
    true,
    dtype,
  );
}

export function full(
  shape: Shape,
  fillValue: number,
  dtype: DType = "f32",
): Tensor {
  const numElements = sizeOf(shape);
  const data = new Float32Array(numElements);
  data.fill(fillValue);
  return new Tensor(shape, data, undefined, 0, true, dtype);
}

export function rand(shape: Shape): Tensor {
  const n = sizeOf(shape);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.random();
  return new Tensor(shape, data);
}

export function randn(shape: Shape): Tensor {
  const n = sizeOf(shape);
  const data = new Float32Array(n);
  // Box-Muller transform
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    const r = Math.sqrt(-2 * Math.log(u1));
    data[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) data[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return new Tensor(shape, data);
}

export function bernoulli(shape: Shape, p: number, _seed?: number): Tensor {
  const n = sizeOf(shape);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.random() < p ? 1 : 0;
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

/** Shared impl for tril/triu: zero elements where zeroWhen(col, row, k) is true. */
function triangularMask(
  a: Tensor,
  k: number,
  zeroWhen: (c: number, r: number, k: number) => boolean,
): Tensor {
  if (a.shape.length < 2)
    throw new Error("tril/triu requires at least 2 dimensions");
  // toArray() honors strides + offset — reading a.data raw would take the
  // base buffer's leading elements for any view (offset-view class, task #58).
  const data = Float32Array.from(a.toArray());
  const H = a.shape[a.shape.length - 2];
  const W = a.shape[a.shape.length - 1];
  const batchSize = data.length / (H * W);
  for (let b = 0; b < batchSize; b++) {
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        if (zeroWhen(c, r, k)) data[b * H * W + r * W + c] = 0;
      }
    }
  }
  return new Tensor(a.shape.slice(), data);
}

export function tril(a: Tensor, k = 0): Tensor {
  return triangularMask(a, k, (c, r, k) => c > r + k);
}

export function triu(a: Tensor, k = 0): Tensor {
  return triangularMask(a, k, (c, r, k) => c < r + k);
}

export function expand(a: Tensor, shape: Shape): Tensor {
  return broadcastTo(a, shape);
}

export type SubOptions = {
  alpha?: number;
};

export function sub(a: Tensor, b: Tensor, options?: SubOptions): Tensor {
  const alpha = options?.alpha ?? 1;
  return applyBinaryOp(a, b, (x, y) => x - alpha * y);
}

export type DivOptions = {
  roundingMode?: "floor" | "trunc" | null;
};

export function div(a: Tensor, b: Tensor, options?: DivOptions): Tensor {
  const rounding = options?.roundingMode ?? null;
  return applyBinaryOp(a, b, (x, y) => {
    let v = x / y;
    if (rounding === "floor") v = Math.floor(v);
    else if (rounding === "trunc") v = Math.trunc(v);
    return v;
  });
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
export function narrow(
  a: Tensor,
  dim: number,
  start: number,
  length: number,
): Tensor {
  const rank = a.shape.length;
  if (dim < 0 || dim >= rank) {
    throw new Error(`narrow: dim ${dim} out of range for rank ${rank}`);
  }
  if (start < 0 || start + length > a.shape[dim]) {
    throw new Error(
      `narrow: range [${start}, ${start + length}) out of bounds for dim size ${a.shape[dim]}`,
    );
  }
  const newShape = a.shape.slice();
  newShape[dim] = length;
  const newOffset = a.offset + start * a.strides[dim];
  return new Tensor(newShape, a.data, a.strides.slice(), newOffset, false);
}

/**
 * Backward for narrow: pad gradient back to original shape.
 */
export function narrowBackward(
  grad: Tensor,
  dim: number,
  start: number,
  originalLength: number,
): Tensor {
  const outShape = grad.shape.slice();
  outShape[dim] = originalLength;
  const outSize = sizeOf(outShape);
  const result = new Float32Array(outSize);

  // Compute indexing
  const outerSize = sizeOf(outShape.slice(0, dim));
  const innerSize = sizeOf(outShape.slice(dim + 1));
  const gradDimSize = grad.shape[dim];

  // Copy grad values into the correct slice
  for (let o = 0; o < outerSize; o++) {
    for (let d = 0; d < gradDimSize; d++) {
      for (let i = 0; i < innerSize; i++) {
        const gradIdx = getStridedIndex(grad, o, d, i, dim);
        const outIdx =
          o * originalLength * innerSize + (start + d) * innerSize + i;
        result[outIdx] = grad.data[gradIdx];
      }
    }
  }

  return new Tensor(outShape, result);
}

export function cat(tensors: Tensor[], options: { dim: number }): Tensor {
  if (tensors.length === 0) throw new Error("cat: empty tensor list");
  const dim =
    options.dim < 0 ? options.dim + tensors[0].shape.length : options.dim;
  const rank = tensors[0].shape.length;
  // Validate shapes match on all non-cat dims
  for (let i = 1; i < tensors.length; i++) {
    if (tensors[i].shape.length !== rank) throw new Error("cat: rank mismatch");
    for (let d = 0; d < rank; d++) {
      if (d !== dim && tensors[i].shape[d] !== tensors[0].shape[d]) {
        throw new Error(`cat: shape mismatch at dim ${d}`);
      }
    }
  }
  const outShape = tensors[0].shape.slice();
  outShape[dim] = tensors.reduce((s, t) => s + t.shape[dim], 0);
  const outSize = sizeOf(outShape);
  const result = new Float32Array(outSize);
  const outerSize = dim === 0 ? 1 : sizeOf(outShape.slice(0, dim));
  const innerSize = dim === rank - 1 ? 1 : sizeOf(outShape.slice(dim + 1));
  let dimOffset = 0;
  for (const t of tensors) {
    const tDimSize = t.shape[dim];
    for (let o = 0; o < outerSize; o++) {
      for (let d = 0; d < tDimSize; d++) {
        for (let i = 0; i < innerSize; i++) {
          const srcIdx = getStridedIndex(t, o, d, i, dim);
          const outIdx =
            o * outShape[dim] * innerSize + (dimOffset + d) * innerSize + i;
          result[outIdx] = t.data[srcIdx];
        }
      }
    }
    dimOffset += tDimSize;
  }
  return new Tensor(outShape, result);
}

function getStridedIndex(
  t: Tensor,
  outer: number,
  dimIdx: number,
  inner: number,
  dim: number,
): number {
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
    const nd = normalizeDimBase(d, rank);
    if (nd < 0 || nd >= rank) {
      throw new Error(`permute: dimension ${d} out of range for rank ${rank}`);
    }
    if (seen.has(nd)) {
      throw new Error(`permute: duplicate dimension ${d}`);
    }
    seen.add(nd);
  }

  // Normalize dims
  const normalizedDims = dims.map((d) => normalizeDimBase(d, rank));

  // Reorder shape and strides
  const newShape = normalizedDims.map((d) => a.shape[d]);
  const newStrides = normalizedDims.map((d) => a.strides[d]);

  return new Tensor(newShape, a.data, newStrides, a.offset, false);
}

function applyUnaryOp(a: Tensor, fn: (x: number) => number): Tensor {
  const out = new Float32Array(a.size);
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    out[i] = fn(readAtLinear(a, i, shapeStrides));
  }
  return new Tensor(a.shape, out);
}

function applyBinaryOp(
  a: Tensor,
  b: Tensor,
  fn: (x: number, y: number) => number,
): Tensor {
  const outShape = broadcastShapes(a.shape, b.shape);
  const aBroadcast = broadcastTo(a, outShape);
  const bBroadcast = broadcastTo(b, outShape);
  const outSize = sizeOf(outShape);
  const out = new Float32Array(outSize);
  const shapeStrides = computeStrides(outShape);
  for (let i = 0; i < outSize; i += 1) {
    out[i] = fn(
      readAtLinear(aBroadcast, i, shapeStrides),
      readAtLinear(bBroadcast, i, shapeStrides),
    );
  }
  return new Tensor(outShape, out);
}

// ============================================================================
// Table-driven elementwise ops
// ============================================================================

/** Unary CPU implementations: op name → scalar function. */
const UNARY_OPS: Record<string, (x: number) => number> = {
  sqrt: Math.sqrt,
  exp: Math.exp,
  log: Math.log,
  neg: (x) => -x,
  abs: Math.abs,
  tanh: Math.tanh,
  sigmoid: (x) => 1.0 / (1.0 + Math.exp(-x)),
  silu: (x) => x / (1.0 + Math.exp(-x)),
  sin: Math.sin,
  cos: Math.cos,
  rsqrt: (x) => 1.0 / Math.sqrt(x),
  floor: Math.floor,
  ceil: Math.ceil,
  round: Math.round,
  sign: Math.sign,
  isfinite: (x) => (Number.isFinite(x) ? 1.0 : 0.0),
  relu: (x) => (x > 0 ? x : 0),
};

/** Binary CPU implementations: op name → scalar function. */
const BINARY_OPS: Record<string, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  mul: (x, y) => x * y,
  pow: Math.pow,
  minimum: Math.min,
  maximum: Math.max,
};

// Generate exports from tables
export function add(a: Tensor, b: Tensor): Tensor {
  return applyBinaryOp(a, b, BINARY_OPS.add);
}
export function mul(a: Tensor, b: Tensor): Tensor {
  return applyBinaryOp(a, b, BINARY_OPS.mul);
}
export function minimum(a: Tensor, b: Tensor): Tensor {
  return applyBinaryOp(a, b, BINARY_OPS.minimum);
}
export function maximum(a: Tensor, b: Tensor): Tensor {
  return applyBinaryOp(a, b, BINARY_OPS.maximum);
}
export function relu(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.relu);
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
  return applyUnaryOp(a, UNARY_OPS.sqrt);
}
export function exp(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.exp);
}
export function log(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.log);
}
export function neg(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.neg);
}
export function abs(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.abs);
}
export function tanh(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.tanh);
}
export function sigmoid(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.sigmoid);
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
  const y =
    1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

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
      out[i] =
        x * 0.5 * (1.0 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x * x * x)));
    }
  } else {
    // Exact formula using erf:
    // x * 0.5 * (1 + erf(x / sqrt(2)))
    const sqrt2Inv = Math.SQRT1_2; // 1/sqrt(2)
    for (let i = 0; i < a.size; i += 1) {
      const x = readAtLinear(a, i, shapeStrides);
      out[i] = x * 0.5 * (1.0 + erf(x * sqrt2Inv));
    }
  }

  return new Tensor(a.shape, out);
}

export function silu(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.silu);
}
export function sin(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.sin);
}
export function cos(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.cos);
}
export function rsqrt(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.rsqrt);
}
export function floor(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.floor);
}
export function ceil(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.ceil);
}
export function round(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.round);
}
export function sign(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.sign);
}
export function isfinite(a: Tensor): Tensor {
  return applyUnaryOp(a, UNARY_OPS.isfinite);
}

export function clamp(
  a: Tensor,
  min: number | null,
  max: number | null,
): Tensor {
  return applyUnaryOp(a, (v) => {
    if (min !== null && v < min) v = min;
    if (max !== null && v > max) v = max;
    return v;
  });
}

export function pow(a: Tensor, b: Tensor): Tensor {
  return applyBinaryOp(a, b, BINARY_OPS.pow);
}

export type SumOptions = {
  dim?: number | number[] | null;
  keepdim?: boolean;
  dtype?: "f16" | "f32" | "i32" | "u32" | "bool" | null;
};

export type MeanOptions = SumOptions;

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
  const normalized = normalizeDimBase(dim, rank);
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

// ============================================================================
// Conv2d
// ============================================================================

export function conv2d(
  input: Tensor,
  weight: Tensor,
  bias: Tensor | undefined,
  options?: {
    stride?: number | [number, number];
    padding?: number | [number, number];
  },
): Tensor {
  const [N, Cin, H, W] = input.shape;
  const [Cout, CinK, KH, KW] = weight.shape;
  if (Cin !== CinK)
    throw new Error(`conv2d: input channels ${Cin} != weight channels ${CinK}`);

  const stride = normalizePair(options?.stride, 1);
  const padding = normalizePair(options?.padding, 0);
  const [sH, sW] = stride;
  const [pH, pW] = padding;

  const outH = Math.floor((H + 2 * pH - KH) / sH + 1);
  const outW = Math.floor((W + 2 * pW - KW) / sW + 1);
  const out = new Float32Array(N * Cout * outH * outW);

  for (let n = 0; n < N; n++) {
    for (let co = 0; co < Cout; co++) {
      for (let oh = 0; oh < outH; oh++) {
        for (let ow = 0; ow < outW; ow++) {
          let acc = bias ? readElement(bias, [co]) : 0;
          for (let ci = 0; ci < Cin; ci++) {
            for (let ky = 0; ky < KH; ky++) {
              for (let kx = 0; kx < KW; kx++) {
                const ih = oh * sH - pH + ky;
                const iw = ow * sW - pW + kx;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                  acc +=
                    readElement(input, [n, ci, ih, iw]) *
                    readElement(weight, [co, ci, ky, kx]);
                }
              }
            }
          }
          out[n * Cout * outH * outW + co * outH * outW + oh * outW + ow] = acc;
        }
      }
    }
  }

  return new Tensor([N, Cout, outH, outW], out);
}

function normalizePair(
  p: number | [number, number] | undefined,
  fallback: number,
): [number, number] {
  if (p === undefined) return [fallback, fallback];
  if (typeof p === "number") return [p, p];
  return p;
}

function readElement(t: Tensor, indices: number[]): number {
  let offset = t.offset;
  for (let i = 0; i < indices.length; i++) {
    offset += indices[i] * t.strides[i];
  }
  return t.data[offset];
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
  const normalized = dims.map((value) => normalizeDimBase(value, rank));
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

function minAll(a: Tensor): number {
  let minVal = Infinity;
  const shapeStrides = computeStrides(a.shape);
  for (let i = 0; i < a.size; i += 1) {
    const val = readAtLinear(a, i, shapeStrides);
    if (val < minVal) {
      minVal = val;
    }
  }
  return minVal;
}

export function min(a: Tensor, options?: MaxOptions): Tensor {
  if (!options || options.dim == null) {
    return new Tensor([], new Float32Array([minAll(a)]));
  }

  const rank = a.shape.length;
  const dims = normalizeDimsForReduce(options.dim, rank, "min");
  const keepdim = options.keepdim ?? false;
  const reduceSet = new Set(dims);

  const outShape = keepdim
    ? a.shape.map((dim, index) => (reduceSet.has(index) ? 1 : dim))
    : a.shape.filter((_, index) => !reduceSet.has(index));

  const outSize = outShape.reduce((acc, dim) => acc * dim, 1) || 1;
  const out = new Float32Array(outSize);
  out.fill(Infinity);
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
    if (val < out[outOffset]) {
      out[outOffset] = val;
    }
  }

  return new Tensor(outShape, out);
}

function normalizeDimsForReduce(
  dim: number | number[],
  rank: number,
  opName: string,
): number[] {
  const dims = Array.isArray(dim) ? dim.slice() : [dim];
  const normalized = dims.map((value) => normalizeDimBase(value, rank));
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

/** Shared argmax/argmin implementation parameterized by comparison direction. */
function argReduce(
  a: Tensor,
  opName: string,
  options: ArgReduceOptions,
  initVal: number,
  isBetter: (val: number, best: number) => boolean,
): Tensor {
  const rank = a.shape.length;
  const dim = options.dim < 0 ? options.dim + rank : options.dim;
  if (dim < 0 || dim >= rank) {
    throw new Error(
      `${opName}: dim ${options.dim} out of range for tensor of rank ${rank}`,
    );
  }
  const keepdim = options.keepdim ?? false;

  const outShape = keepdim
    ? a.shape.map((d, i) => (i === dim ? 1 : d))
    : a.shape.filter((_, i) => i !== dim);

  const outSize = outShape.reduce((acc, d) => acc * d, 1) || 1;
  const out = new Float32Array(outSize);
  const bestVals = new Float32Array(outSize);
  bestVals.fill(initVal);

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
    if (isBetter(val, bestVals[outOffset])) {
      bestVals[outOffset] = val;
      out[outOffset] = dimCoord;
    }
  }

  return new Tensor(outShape, out);
}

export function argmax(a: Tensor, options: ArgReduceOptions): Tensor {
  return argReduce(a, "argmax", options, -Infinity, (v, b) => v > b);
}

export function argmin(a: Tensor, options: ArgReduceOptions): Tensor {
  return argReduce(a, "argmin", options, Infinity, (v, b) => v < b);
}

// ============================================================================
// Comparison ops - return 1.0 for true, 0.0 for false
// ============================================================================

function comparisonOp(
  a: Tensor,
  b: Tensor,
  cmp: (x: number, y: number) => boolean,
): Tensor {
  return applyBinaryOp(a, b, (x, y) => (cmp(x, y) ? 1.0 : 0.0));
}

export function gt(a: Tensor, b: Tensor): Tensor {
  return comparisonOp(a, b, (x, y) => x > y);
}
export function lt(a: Tensor, b: Tensor): Tensor {
  return comparisonOp(a, b, (x, y) => x < y);
}
export function ge(a: Tensor, b: Tensor): Tensor {
  return comparisonOp(a, b, (x, y) => x >= y);
}
export function le(a: Tensor, b: Tensor): Tensor {
  return comparisonOp(a, b, (x, y) => x <= y);
}
export function eq(a: Tensor, b: Tensor): Tensor {
  return comparisonOp(a, b, (x, y) => x === y);
}
export function ne(a: Tensor, b: Tensor): Tensor {
  return comparisonOp(a, b, (x, y) => x !== y);
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
 * Scatter src values into base tensor at positions defined by view metadata.
 * Returns a new tensor (does not mutate base).
 *
 * This is the fundamental operation for view mutation lowering (§4.4):
 * When mutating a view, we load the base, scatter the new values
 * into the view positions, and store back.
 */
function stridedScatterImpl(
  opName: string,
  base: Tensor,
  src: Tensor,
  options: StridedScatterOptions,
  combine: (existing: number, srcVal: number) => number,
): Tensor {
  const { offset, viewShape, viewStrides } = options;

  if (src.shape.length !== viewShape.length) {
    throw new Error(
      `${opName}: src rank ${src.shape.length} doesn't match view rank ${viewShape.length}`,
    );
  }
  for (let i = 0; i < viewShape.length; i++) {
    if (src.shape[i] !== viewShape[i]) {
      throw new Error(
        `${opName}: src shape [${src.shape}] doesn't match view shape [${viewShape}]`,
      );
    }
  }

  // Create output matching base's logical size, not its backing buffer size.
  // The base may be a view into a larger buffer (offset > 0 or data.length > size).
  const baseSize = sizeOf(base.shape);
  const out = new Float32Array(baseSize);
  // Read base values through strides (handles non-contiguous views)
  const baseStrides = computeStrides(base.shape);
  for (let i = 0; i < baseSize; i++) {
    let remainder = i;
    let idx = base.offset;
    for (let axis = 0; axis < base.shape.length; axis++) {
      const coord = Math.floor(remainder / baseStrides[axis]);
      remainder -= coord * baseStrides[axis];
      idx += coord * base.strides[axis];
    }
    out[i] = base.data[idx];
  }

  const srcSize = sizeOf(viewShape);
  const srcShapeStrides = computeStrides(viewShape);

  for (let linear = 0; linear < srcSize; linear++) {
    let remainder = linear;
    let baseOffset = offset;
    for (let axis = 0; axis < viewShape.length; axis++) {
      const stride = srcShapeStrides[axis];
      const coord = Math.floor(remainder / stride);
      remainder -= coord * stride;
      baseOffset += coord * viewStrides[axis];
    }

    const srcVal = readAtLinear(src, linear, srcShapeStrides);
    out[baseOffset] = combine(out[baseOffset], srcVal);
  }

  return new Tensor(base.shape, out, computeStrides(base.shape), 0);
}

export function stridedScatterCopy(
  base: Tensor,
  src: Tensor,
  options: StridedScatterOptions,
): Tensor {
  return stridedScatterImpl(
    "stridedScatterCopy",
    base,
    src,
    options,
    (_, v) => v,
  );
}

export function stridedScatterAdd(
  base: Tensor,
  src: Tensor,
  options: StridedScatterOptions,
): Tensor {
  return stridedScatterImpl(
    "stridedScatterAdd",
    base,
    src,
    options,
    (e, v) => e + v,
  );
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
