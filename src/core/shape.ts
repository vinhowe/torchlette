/**
 * Canonical pure shape utility functions.
 *
 * Zero dependencies — importable from any layer (frontend, engine, backend).
 */

export function sizeOf(shape: number[]): number {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

export function broadcastShapes(a: number[], b: number[]): number[] {
  const outRank = Math.max(a.length, b.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error(`Cannot broadcast shapes [${a}] and [${b}]`);
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim);
  }
  return out;
}

export function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export function broadcastThreeShapes(
  a: number[],
  b: number[],
  c: number[],
): number[] {
  const outRank = Math.max(a.length, b.length, c.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i++) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    const cDim = c[c.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1)
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    if (aDim !== cDim && aDim !== 1 && cDim !== 1)
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    if (bDim !== cDim && bDim !== 1 && cDim !== 1)
      throw new Error(`Cannot broadcast shapes [${a}], [${b}], and [${c}]`);
    out[outRank - 1 - i] = Math.max(aDim, bDim, cDim);
  }
  return out;
}

export function contiguousStrides(shape: number[]): number[] {
  if (shape.length === 0) return [];
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Infer strides for a new shape given old shape/strides, without copying data.
 * Returns null if the reshape requires a contiguous copy.
 * Implements PyTorch's computeStride algorithm.
 */
export function inferReshapeStrides(
  oldShape: number[],
  oldStrides: number[],
  newShape: number[],
): number[] | null {
  if (newShape.length === 0) return [];
  if (oldShape.length === 0) return contiguousStrides(newShape);

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
    while (oldProduct < newProduct && oldIdx + 1 < oldN) {
      if (
        oldStrides[oldIdx] !==
        oldStrides[oldIdx + 1] * oldShape[oldIdx + 1]
      ) {
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
