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
