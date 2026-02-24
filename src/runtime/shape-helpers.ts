import type { TransposeOptions } from "../backend/types";

export function broadcastShapes(a: number[], b: number[]): number[] {
  const outRank = Math.max(a.length, b.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i++) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim);
  }
  return out;
}

export function matmulShape(a: number[], b: number[]): number[] {
  if (a.length < 1 || b.length < 1) {
    throw new Error("matmul requires at least 1D tensors");
  }

  // Handle 1D cases
  if (a.length === 1 && b.length === 1) {
    // Vector dot product: [n] @ [n] -> []
    return [];
  }
  if (a.length === 1) {
    // [n] @ [..., n, m] -> [..., m]
    return [...b.slice(0, -2), b[b.length - 1]];
  }
  if (b.length === 1) {
    // [..., m, n] @ [n] -> [..., m]
    return a.slice(0, -1);
  }

  // 2D+ matmul
  const m = a[a.length - 2];
  const n = b[b.length - 1];

  // Broadcast batch dimensions
  const aBatch = a.slice(0, -2);
  const bBatch = b.slice(0, -2);
  const batch = broadcastShapes(aBatch, bBatch);

  return [...batch, m, n];
}

export function transposeShape(shape: number[], options: TransposeOptions): number[] {
  const { dim0, dim1 } = options;
  const result = shape.slice();
  const temp = result[dim0];
  result[dim0] = result[dim1];
  result[dim1] = temp;
  return result;
}

export function reduceShape(
  shape: number[],
  dim: number | number[] | null | undefined,
  keepdim: boolean,
): number[] {
  if (dim == null) {
    // Reduce all dimensions
    return keepdim ? shape.map(() => 1) : [];
  }

  const dims = Array.isArray(dim) ? dim : [dim];
  const normalizedDims = dims.map((d) => (d < 0 ? shape.length + d : d));

  if (keepdim) {
    return shape.map((s, i) => (normalizedDims.includes(i) ? 1 : s));
  }

  return shape.filter((_, i) => !normalizedDims.includes(i));
}

export function broadcastThreeShapes(a: number[], b: number[], c: number[]): number[] {
  const outRank = Math.max(a.length, b.length, c.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i++) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    const cDim = c[c.length - 1 - i] ?? 1;
    // Check all pairs for broadcast compatibility
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    }
    if (aDim !== cDim && aDim !== 1 && cDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    }
    if (bDim !== cDim && bDim !== 1 && cDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim, cDim);
  }
  return out;
}
