/**
 * Semantic Derivation — the INDEX ALGEBRA (Crystal Campaign 3, Phase 4).
 *
 * An index-space map's *meaning* is a first-class DATA object: a description of
 * how output coordinates read input coordinates (a stride/offset/gather rule),
 * carrying only shapes, dims and offsets — never a closure or a buffer. From this
 * ONE source the op's gradient DERIVES: the adjoint (VJP) of an index map is its
 * TRANSPOSE. Where the forward map moves an element input→output, the transpose
 * moves the upstream grad output→input, SUMMING when the forward was many-to-one.
 * This single fact subsumes what were nine independently hand-authored backward
 * choices (design §2 category c, §4.1 index-space frame, §8.1):
 *
 *   forward map          nature              adjoint = transpose
 *   ------------------   -----------------   -----------------------------------
 *   reshape (flat id)    bijection           reshape back
 *   transpose(d0,d1)     bijection (invol.)  transpose(d0,d1)
 *   permute(perm)        bijection           permute(inverse perm)
 *   narrow(dim,off,len)  injection out⊂in    pad — scatter grad into zeros at off
 *   cat(dim,sizes)       disjoint union      split — narrow grad at each offset
 *   broadcast(in→out)    many-to-one         reduce — sum over broadcast dims
 *   reduce(dims)         many-to-one (fold)  broadcast grad back over reduced dims
 *   gather(dim)          data-indexed        scatterAdd grad into zeros
 *   scatterAdd(dim)      data-indexed        [grad, gather(grad)]
 *
 * The DUALITIES are single-sourced here: narrow⇄pad, cat⇄split, broadcast⇄reduce,
 * gather⇄scatter are each ONE transpose fact, not two hand copies. The transpose
 * is a pure DATA→DATA function (`adjointIndexMap`) — unit-testable, an involution
 * on the bijections/dual-pairs. Its REALIZATION composes the existing runtime
 * kernels (`realizeIndexAdjoint`), never re-owning them (design §4.4): the
 * transpose DECISION derives; the movement is realized by `rt.*`.
 *
 * The index TENSOR of a gather/scatter is NOT part of the map DATA — it is runtime
 * data supplied to the realizer (exactly as operands/eps are supplied to
 * `interpretComposition`), so the map stays pure DATA and the schema gate holds.
 */

import type { RuntimeEngine } from "../../runtime/engine";
import type { Tensor as RuntimeTensor } from "../../runtime/tensor";

// ----------------------------------------------------------------------------
// The forward index-map schema — DATA (shapes / dims / offsets only).
// ----------------------------------------------------------------------------

/** An index-space map as DATA: the forward movement the frontend op performs. */
export type IndexMap =
  | { k: "reshape"; inShape: number[]; outShape: number[] }
  | { k: "transpose"; dim0: number; dim1: number }
  | { k: "permute"; perm: number[] }
  | { k: "narrow"; dim: number; start: number; length: number; inLen: number }
  | { k: "cat"; dim: number; sizes: number[] }
  | { k: "broadcast"; inShape: number[]; outShape: number[] }
  | { k: "reduce"; inShape: number[]; dims: number[]; keepdim: boolean }
  | { k: "gather"; dim: number; inShape: number[] }
  | { k: "scatterAdd"; dim: number };

/**
 * The adjoint (transpose) of an index map — also DATA, realize-oriented. The
 * algebra is closed under transpose (each forward kind maps to an adjoint kind;
 * the dual pairs are involutions), so the derivation is a pure DATA→DATA function.
 */
export type IndexAdjoint =
  | { k: "reshape"; toShape: number[] }
  | { k: "transpose"; dim0: number; dim1: number }
  | { k: "permute"; perm: number[] }
  | { k: "pad"; dim: number; start: number; outLen: number }
  | { k: "split"; dim: number; sizes: number[] }
  | { k: "reduceToShape"; fromShape: number[]; toShape: number[] }
  | { k: "broadcastDims"; inShape: number[]; dims: number[]; keepdim: boolean }
  | { k: "scatterZeros"; dim: number; inShape: number[] }
  | { k: "scatterSelf"; dim: number };

// ----------------------------------------------------------------------------
// The derived fact: the adjoint IS the transpose (pure DATA→DATA).
// ----------------------------------------------------------------------------

/** Invert a permutation: `inverse[perm[i]] = i`. The permute op's adjoint. */
export function invertPermutation(perm: readonly number[]): number[] {
  const inv = new Array<number>(perm.length);
  for (let i = 0; i < perm.length; i++) inv[perm[i]] = i;
  return inv;
}

/**
 * DERIVE the adjoint of an index map — its transpose. This is the single source
 * for every index-space op's gradient: no per-op hand backward, one structural
 * rule (design §4.2 analogue for the index family; §8.1 broadcast-transpose).
 */
export function adjointIndexMap(map: IndexMap): IndexAdjoint {
  switch (map.k) {
    case "reshape":
      // bijection: reshape the grad back to the input shape.
      return { k: "reshape", toShape: map.inShape };
    case "transpose":
      // a 2-axis swap is its own inverse (involution).
      return { k: "transpose", dim0: map.dim0, dim1: map.dim1 };
    case "permute":
      // bijection: the adjoint permutation is the inverse permutation.
      return { k: "permute", perm: invertPermutation(map.perm) };
    case "narrow":
      // injection out⊂in: scatter grad into a zero-filled input at the offset.
      return { k: "pad", dim: map.dim, start: map.start, outLen: map.inLen };
    case "cat":
      // disjoint union: split the grad at the recorded per-input offsets.
      return { k: "split", dim: map.dim, sizes: map.sizes };
    case "broadcast":
      // many-to-one: sum the grad over the broadcast dims back to the input shape.
      return {
        k: "reduceToShape",
        fromShape: map.outShape,
        toShape: map.inShape,
      };
    case "reduce":
      // many-to-one fold: broadcast the grad back over the reduced dims.
      return {
        k: "broadcastDims",
        inShape: map.inShape,
        dims: map.dims,
        keepdim: map.keepdim,
      };
    case "gather":
      // data-indexed: scatterAdd the grad into zeros (gather's transpose).
      return { k: "scatterZeros", dim: map.dim, inShape: map.inShape };
    case "scatterAdd":
      // data-indexed: grad flows unchanged to `a`; `src` gets grad gathered.
      return { k: "scatterSelf", dim: map.dim };
  }
}

// ----------------------------------------------------------------------------
// The realizer — compose runtime kernels for the derived adjoint (design §4.4:
// the movement is REFERENCED, never re-owned). Pure movement, no formula.
// ----------------------------------------------------------------------------

/** Context the data-indexed adjoints need (index tensor for gather/scatter). */
export interface IndexAdjointCtx {
  index?: RuntimeTensor;
}

/**
 * Realize a derived adjoint over the runtime engine, producing the input grad(s).
 * Single-input adjoints return one tensor; `split`/`scatterSelf` return an array
 * (cat's per-input grads / scatterAdd's [a, src] grads).
 */
export function realizeIndexAdjoint(
  rt: RuntimeEngine,
  adj: IndexAdjoint,
  grad: RuntimeTensor,
  ctx?: IndexAdjointCtx,
): RuntimeTensor | RuntimeTensor[] {
  switch (adj.k) {
    case "reshape":
      return rt.reshape(grad, adj.toShape);
    case "transpose":
      return rt.transpose(grad, { dim0: adj.dim0, dim1: adj.dim1 });
    case "permute":
      return rt.permute(grad, adj.perm);
    case "pad":
      return rt.narrowBackward(grad, adj.dim, adj.start, adj.outLen);
    case "split": {
      const out: RuntimeTensor[] = [];
      let offset = 0;
      for (const size of adj.sizes) {
        out.push(rt.narrow(grad, adj.dim, offset, size));
        offset += size;
      }
      return out;
    }
    case "reduceToShape":
      return reduceToShape(rt, grad, adj.toShape);
    case "broadcastDims":
      return broadcastOverDims(rt, grad, adj.inShape, adj.dims, adj.keepdim);
    case "scatterZeros": {
      if (ctx?.index === undefined) {
        throw new Error(
          "realizeIndexAdjoint: gather adjoint (scatterZeros) needs the index tensor.",
        );
      }
      const z = rt.zeros(adj.inShape);
      return rt.scatterAdd(z, ctx.index, grad, { dim: adj.dim });
    }
    case "scatterSelf": {
      if (ctx?.index === undefined) {
        throw new Error(
          "realizeIndexAdjoint: scatterAdd adjoint needs the index tensor.",
        );
      }
      return [grad, rt.gather(grad, ctx.index, { dim: adj.dim })];
    }
  }
}

/**
 * Convenience: derive AND realize an index map's backward in one call — the
 * single entry point the frontend view/gather/scatter backwards route through.
 */
export function backwardOfIndexMap(
  rt: RuntimeEngine,
  map: IndexMap,
  grad: RuntimeTensor,
  ctx?: IndexAdjointCtx,
): RuntimeTensor | RuntimeTensor[] {
  return realizeIndexAdjoint(rt, adjointIndexMap(map), grad, ctx);
}

// ----------------------------------------------------------------------------
// The broadcast⇄reduce transpose movement (the P1 debt, §8.1) — single-sourced
// here. `reduceToShape` is the implicit-broadcast VJP (`_sumToShape`);
// `broadcastOverDims` is the reduction VJP (`_expandGrad`). Both are the SAME
// index-transpose family, realized over `rt`.
// ----------------------------------------------------------------------------

/**
 * The transpose of a broadcast: sum `grad` down to `targetShape` (numpy
 * broadcasting rules — leading rank-pad dims and size-1 dims collapse). This is
 * `_sumToShape` promoted into the index algebra (the broadcast⇄reduce duality).
 */
export function reduceToShape(
  rt: RuntimeEngine,
  grad: RuntimeTensor,
  targetShape: number[],
): RuntimeTensor {
  if (shapesEqual(grad.shape, targetShape)) return grad;
  const gradRank = grad.shape.length;
  const targetRank = targetShape.length;
  const pad = Math.max(0, gradRank - targetRank);
  const paddedTarget = new Array<number>(pad).fill(1).concat(targetShape);
  const dims: number[] = [];
  for (let axis = 0; axis < gradRank; axis += 1) {
    if (axis < pad) {
      dims.push(axis); // rank-pad leading dim — drop it
    } else if (paddedTarget[axis] === 1 && grad.shape[axis] !== 1) {
      dims.push(axis); // broadcast size-1 dim — sum it
    }
  }
  let reduced = grad;
  if (dims.length > 0) {
    const summed = rt.sum(reduced, { dim: dims, keepdim: false });
    reduced =
      typeof summed === "number" ? rt.full([], summed, grad.device) : summed;
  }
  if (!shapesEqual(reduced.shape, targetShape)) {
    reduced = rt.reshape(reduced, targetShape);
  }
  return reduced;
}

/**
 * The transpose of a reduction: broadcast `grad` back over the reduced `dims` to
 * `inputShape`. This is `_expandGrad` promoted into the index algebra (the
 * reduce⇄broadcast duality — a reduction is the transpose of a broadcast, §8.1).
 */
export function broadcastOverDims(
  rt: RuntimeEngine,
  grad: RuntimeTensor,
  inputShape: number[],
  dims: number[],
  keepdim: boolean,
): RuntimeTensor {
  let expanded = grad;
  if (!keepdim && dims.length > 0) {
    const rank = inputShape.length;
    const reduceSet = new Set(dims);
    const nextShape = new Array<number>(rank);
    let gradAxis = 0;
    for (let axis = 0; axis < rank; axis += 1) {
      if (reduceSet.has(axis)) {
        nextShape[axis] = 1;
      } else {
        nextShape[axis] = grad.shape[gradAxis];
        gradAxis += 1;
      }
    }
    expanded = rt.reshape(expanded, nextShape);
  }
  return rt.expand(expanded, inputShape);
}

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ----------------------------------------------------------------------------
// The schema gate — an index map is DATA (the `assertNoDefinitionBody` analogue,
// design §4.1: no leaf is a JS function, string, or buffer; the kind is a member
// of the closed index algebra). A map that hid the old `narrowBackward` closure
// behind an opaque leaf is unconstructible (the covenant/R22 defense).
// ----------------------------------------------------------------------------

const INDEX_MAP_KINDS = new Set<string>([
  "reshape",
  "transpose",
  "permute",
  "narrow",
  "cat",
  "broadcast",
  "reduce",
  "gather",
  "scatterAdd",
]);

/** Prove an `IndexMap` is DATA (numbers / number-arrays / a known kind only). */
export function assertNoIndexMapBody(map: IndexMap, path = "map"): void {
  const walk = (value: unknown, p: string): void => {
    if (value === null || value === undefined) {
      throw new Error(`IndexMap schema gate: ${p} is null/undefined.`);
    }
    const t = typeof value;
    if (t === "function") {
      throw new Error(
        `IndexMap schema gate: ${p} is a function — the map must be DATA (no embedded movement body).`,
      );
    }
    if (t === "string" || t === "number" || t === "boolean") return;
    if (t === "object") {
      if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
        throw new Error(
          `IndexMap schema gate: ${p} is a buffer leaf — a map names shapes/dims, never buffers.`,
        );
      }
      if (Array.isArray(value)) {
        value.forEach((v, i) => {
          walk(v, `${p}[${i}]`);
        });
        return;
      }
      const kind = (value as { k?: unknown }).k;
      if (typeof kind !== "string" || !INDEX_MAP_KINDS.has(kind)) {
        throw new Error(
          `IndexMap schema gate: ${p} has kind ${JSON.stringify(kind)} which is not an index-algebra map.`,
        );
      }
      for (const [key, v] of Object.entries(value as Record<string, unknown>)) {
        if (key === "k") continue;
        walk(v, `${p}.${key}`);
      }
      return;
    }
    throw new Error(`IndexMap schema gate: ${p} has non-data type ${t}.`);
  };
  walk(map, path);
}
