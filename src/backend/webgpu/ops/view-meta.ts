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

/** Minimal node surface the offset deriver walks (a LazyIRNode subset). */
export interface MetaNodeLike {
  op: string;
  shape: number[];
  payload?: unknown;
  inputs: ReadonlyArray<{
    kind: string;
    node?: MetaNodeLike;
    outputIndex?: number;
  }>;
}

/** A ref's live output layout when its result tensor still exists (a
 *  materialized base, or a view whose result wasn't released). Full ViewMeta so
 *  the chain can bottom out on a materialized base's real shape/strides. */
export type ResolveLiveMeta = (
  ref: MetaNodeLike["inputs"][number],
) => ViewMeta | undefined;

/**
 * Task #71: derive the ELEMENT OFFSET of a node's output by walking its view
 * chain — the single-source form of "narrow start·stride" that the compiled
 * replay re-derives per step (offset is data, not template identity). Base
 * cases: a live producer supplies its real ViewMeta; a non-view producer with
 * no live meta allocates fresh (contiguous, offset 0). Returns null when the
 * chain is not derivable — the caller keeps the frozen record-time offset.
 */
export function deriveNodeOffset(
  node: MetaNodeLike,
  resolveLive: ResolveLiveMeta,
): number | null {
  const vm = deriveNodeViewMeta(node, resolveLive);
  return vm ? vm.offset : null;
}

/** Full ViewMeta (shape/strides/offset) for a node via its view chain. */
export function deriveNodeViewMeta(
  node: MetaNodeLike,
  resolveLive: ResolveLiveMeta,
): ViewMeta | null {
  if (!VIEW_META_OPS.has(node.op)) {
    // Non-view producer: fresh contiguous allocation (offset 0).
    return {
      shape: node.shape.slice(),
      strides: contiguousStrides(node.shape),
      offset: 0,
    };
  }
  const ref = node.inputs[0];
  if (!ref || (ref.outputIndex ?? 0) !== 0) return null;
  // Prefer the input's live layout (materialized base OR an un-released view).
  let inp = resolveLive(ref);
  if (!inp) {
    if (!ref.node) return null; // materialized-but-released: not derivable.
    inp = deriveNodeViewMeta(ref.node, resolveLive) ?? undefined;
  }
  if (!inp) return null;
  return viewResultMeta(node.op, inp, node.shape, node.payload);
}
