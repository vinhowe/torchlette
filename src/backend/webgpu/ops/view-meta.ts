/**
 * View metadata transforms — the SINGLE SOURCE for how each view op maps an
 * input {shape, strides, offset} to its output {shape, strides, offset}.
 *
 * Views are metadata-only (no GPU work): their output layout is a pure function
 * of the input layout + the op's params. Two callers need that function:
 *   1. the backend view ops (narrow/permute/transpose/expand/reshape) at
 *      EXECUTION time (they wrap these + the buffer/ownership handling);
 *   2. build-without-execution result-metadata derivation (stage-4 phase 4.4),
 *      which has no live result to read shape/strides/offset from.
 * Keeping the math here — instead of re-deriving it in the executor — avoids a
 * second source of truth that would silently drift (the "single source at
 * seams" rule). `test/webgpu/view-meta.spec.ts` asserts these match the backend
 * ops' actual output (the seam check), and reshape's materialize branch is
 * pinned the same way.
 */
import {
  checkContiguous,
  contiguousStrides,
  inferReshapeStrides,
} from "../../../core/shape";

export interface ViewMeta {
  shape: number[];
  strides: number[];
  offset: number;
}

/** narrow(dim, start, length): shrink one dim; offset shifts by start·stride[dim]. */
export function narrowMeta(
  inp: ViewMeta,
  dim: number,
  start: number,
  length: number,
): ViewMeta {
  const shape = inp.shape.slice();
  shape[dim] = length;
  return {
    shape,
    strides: inp.strides.slice(),
    offset: inp.offset + start * inp.strides[dim],
  };
}

/** permute(dims): reorder shape + strides by dims; offset unchanged. */
export function permuteMeta(inp: ViewMeta, dims: number[]): ViewMeta {
  return {
    shape: dims.map((d) => inp.shape[d]),
    strides: dims.map((d) => inp.strides[d]),
    offset: inp.offset,
  };
}

/** transpose(dim0, dim1): permute with two dims swapped. */
export function transposeMeta(
  inp: ViewMeta,
  dim0: number,
  dim1: number,
): ViewMeta {
  const dims = inp.shape.map((_, i) => i);
  dims[dim0] = dim1;
  dims[dim1] = dim0;
  return permuteMeta(inp, dims);
}

/** expand(shape): leading + size-1→N dims get stride 0; rest keep input stride. */
export function expandMeta(inp: ViewMeta, shape: number[]): ViewMeta {
  const outStrides: number[] = [];
  const padded = shape.length - inp.shape.length;
  for (let i = 0; i < shape.length; i++) {
    if (i < padded) {
      outStrides.push(0);
    } else {
      const ii = i - padded;
      const inDim = inp.shape[ii];
      const outDim = shape[i];
      outStrides.push(inDim === 1 && outDim > 1 ? 0 : inp.strides[ii]);
    }
  }
  return { shape: shape.slice(), strides: outStrides, offset: inp.offset };
}

/**
 * reshape(shape): mirrors backend reshape's three branches —
 *   contiguous input        → contiguous output view, offset preserved;
 *   compatible non-contig    → inferReshapeStrides, offset preserved;
 *   incompatible non-contig  → MATERIALIZE → contiguous, offset 0.
 * `materialized` tells the backend (the only caller that touches buffers) to do
 * the contiguous() copy; the derivation just needs the resulting layout.
 */
export function reshapeMeta(
  inp: ViewMeta,
  shape: number[],
): ViewMeta & { materialized: boolean } {
  if (checkContiguous(inp.shape, inp.strides)) {
    return {
      shape: shape.slice(),
      strides: contiguousStrides(shape),
      offset: inp.offset,
      materialized: false,
    };
  }
  const ns = inferReshapeStrides(inp.shape, inp.strides, shape);
  if (ns !== null) {
    return { shape: shape.slice(), strides: ns, offset: inp.offset, materialized: false };
  }
  return {
    shape: shape.slice(),
    strides: contiguousStrides(shape),
    offset: 0,
    materialized: true,
  };
}

/** View op names this module handles (others are not metadata-only views). */
export const VIEW_META_OPS = new Set([
  "narrow",
  "permute",
  "transpose",
  "reshape",
  "view",
  "expand",
]);

/**
 * Dispatch by op name + payload (the LazyIRNode form). Returns the output
 * ViewMeta, or null if the op isn't a metadata-only view this module handles.
 * For reshape/expand the target shape is the op's output shape (passed as
 * `outShape`); narrow/permute/transpose read their params from `payload`.
 */
export function viewResultMeta(
  op: string,
  inp: ViewMeta,
  outShape: number[],
  payload: unknown,
): ViewMeta | null {
  const p = payload as Record<string, number | number[]> | undefined;
  switch (op) {
    case "narrow":
      return narrowMeta(
        inp,
        p!.dim as number,
        p!.start as number,
        p!.length as number,
      );
    case "permute":
      return permuteMeta(inp, p!.dims as number[]);
    case "transpose":
      return transposeMeta(inp, p!.dim0 as number, p!.dim1 as number);
    case "expand":
      return expandMeta(inp, outShape);
    case "reshape":
    case "view":
      return reshapeMeta(inp, outShape);
    default:
      return null;
  }
}
