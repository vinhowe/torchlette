/**
 * Tile-IR kernel specs for ops that were previously hand-written WGSL.
 *
 * Each function returns compiled WGSL (via compileTileKernel) that is
 * plug-compatible with the existing dispatch infrastructure: same binding
 * order, same uniform struct layout, same workgroup size.
 *
 * Benefits: constant folding eliminates dead broadcast dimensions,
 * CSE deduplicates common subexpressions, LICM hoists loop-invariant code.
 */

import {
  type TileKernelSpec,
  type KernelContext,
  type BlockExpr,
  elementwiseGrid,
} from "../tile-ir";
import { compileTileKernel } from "../tile-compiler";
import { WORKGROUP_SIZE } from "../shape-utils";

const WG = WORKGROUP_SIZE; // 256

// ============================================================================
// Broadcast Index Helper
// ============================================================================

/**
 * Compute the linear offset into a storage buffer given a flat output index
 * and effective broadcast strides. Coordinates are decomposed from the flat
 * index using indexShape, then each coordinate is multiplied by its stride.
 *
 * Broadcast dimensions have stride=0, and the constant folder eliminates
 * the coord*0 term automatically.
 *
 * @param ctx        - KernelContext
 * @param flatIdx    - The flat output index
 * @param indexShape - Broadcast output shape (with leading 1s removed)
 * @param strides    - Effective broadcast strides for this input
 * @param offset     - Buffer offset for this input
 * @param prefix     - Prefix for let binding names (must be unique per input)
 */
function buildStridedOffset(
  ctx: KernelContext,
  flatIdx: BlockExpr,
  indexShape: number[],
  strides: number[],
  offset: number,
  prefix: string,
): BlockExpr {
  const rank = indexShape.length;
  if (rank === 0) return ctx.u32(offset);

  let rem: BlockExpr = flatIdx;
  let result: BlockExpr = ctx.u32(offset);

  for (let d = 0; d < rank; d++) {
    // Compute stride for coordinate decomposition (product of remaining dims)
    let dimStride = 1;
    for (let j = d + 1; j < rank; j++) dimStride *= indexShape[j];

    const coord = d < rank - 1
      ? ctx.emitLet(`${prefix}_c${d}`, rem.div(ctx.u32(dimStride)))
      : rem;

    if (d < rank - 1) {
      rem = ctx.emitLet(`${prefix}_r${d}`, rem.mod(ctx.u32(dimStride)));
    }

    // offset += coord * inputStride[d]
    // When strides[d] === 0 (broadcast), constant folder eliminates: coord*0 → 0, result+0 → result
    result = result.add(coord.mul(ctx.u32(strides[d])));
  }

  return ctx.emitLet(`${prefix}_off`, result);
}

// ============================================================================
// Fill Kernel
// ============================================================================

export function fillWGSL(): string {
  const spec: TileKernelSpec = {
    name: "fill",
    workgroupSize: WG,
    bindings: {
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      size: "u32",
      value: "f32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      ctx.emitStore("out", idx, ctx.uniform("value"));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Arange Kernel
// ============================================================================

export function arangeWGSL(): string {
  const spec: TileKernelSpec = {
    name: "arange",
    workgroupSize: WG,
    bindings: {
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      size: "u32",
      start: "f32",
      step: "f32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      const val = ctx.uniform("start").add(idx.toF32().mul(ctx.uniform("step")));
      ctx.emitStore("out", idx, val);
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Triangular (tril/triu) Kernel
// ============================================================================

export function triangularWGSL(upper: boolean): string {
  const spec: TileKernelSpec = {
    name: upper ? "triu" : "tril",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      num_elements: "u32",
      H: "u32",
      W: "u32",
      k: "i32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "num_elements" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("num_elements")), () => ctx.emitReturn());
      const W = ctx.uniform("W");
      const row = idx.div(W).mod(ctx.uniform("H")).toI32();
      const col = idx.mod(W).toI32();
      const k = ctx.uniform("k");
      // tril: zero where col > row + k; triu: zero where col < row + k
      const cond = upper ? col.lt(row.add(k)) : col.gt(row.add(k));
      ctx.ifThenElse(cond,
        () => ctx.emitStore("output", idx, ctx.f32(0)),
        () => ctx.emitStore("output", idx, ctx.load("input", idx)),
      );
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Comparison Op Kernel
// ============================================================================

/**
 * Generate WGSL for a broadcast comparison op.
 * out[idx] = select(0.0, 1.0, a[offsetA] wgslOp b[offsetB])
 */
export function comparisonWGSL(
  wgslOp: string,
  indexShape: number[],
  aStrides: number[],
  bStrides: number[],
  aOffset: number,
  bOffset: number,
): string {
  const opToMethod: Record<string, (a: BlockExpr, b: BlockExpr) => BlockExpr> = {
    ">": (a, b) => a.gt(b),
    "<": (a, b) => a.lt(b),
    ">=": (a, b) => a.ge(b),
    "<=": (a, b) => a.le(b),
    "==": (a, b) => a.eq(b),
    "!=": (a, b) => a.ne(b),
  };
  const cmpFn = opToMethod[wgslOp];
  if (!cmpFn) throw new Error(`Unknown comparison op: ${wgslOp}`);

  const spec: TileKernelSpec = {
    name: `cmp_${wgslOp}`,
    workgroupSize: WG,
    bindings: {
      a: { storage: "read", type: "f32" },
      b: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

      const aOff = buildStridedOffset(ctx, idx, indexShape, aStrides, aOffset, "ao");
      const bOff = buildStridedOffset(ctx, idx, indexShape, bStrides, bOffset, "bo");

      const aVal = ctx.load("a", aOff);
      const bVal = ctx.load("b", bOff);
      const result = cmpFn(aVal, bVal).select(ctx.f32(1), ctx.f32(0));
      ctx.emitStore("out", idx, result);
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Argmax/Argmin Kernel
// ============================================================================

/**
 * Generate WGSL for argmax/argmin along a dimension.
 * Sequential loop per output element, finding the index of max/min value.
 */
export function argReduceWGSL(
  compareOp: string, // ">" for argmax, "<" for argmin
  inputShape: number[],
  inputStrides: number[],
  outShape: number[],
  outStrides: number[],
  dim: number,
  dimSize: number,
  dimStride: number,
  inputToOutDim: number[],
): string {
  const rank = inputShape.length;
  const isMax = compareOp === ">";
  const name = isMax ? "argmax" : "argmin";

  const spec: TileKernelSpec = {
    name: `${name}_d${dim}_${inputShape.join("x")}`,
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      outSize: "u32",
      dimSize: "u32",
      dimStride: "u32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const outIdx = ctx.globalId(0);
      ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      // Compute base offset in input (with reduce dim = 0)
      // Decompose outIdx into output coordinates and compute input offset
      let baseOffset: BlockExpr = ctx.u32(0);
      if (outShape.length > 0) {
        let rem = ctx.emitLet("ar_rem", outIdx);
        for (let d = 0; d < outShape.length; d++) {
          const coord = d < outShape.length - 1
            ? ctx.emitLet(`ar_c${d}`, rem.div(ctx.u32(outStrides[d])))
            : rem;
          if (d < outShape.length - 1) {
            rem = ctx.emitLet(`ar_r${d}`, rem.mod(ctx.u32(outStrides[d])));
          }
          // Map output coord back to input dimension
          // Find which input dim this output dim corresponds to
          for (let id = 0; id < rank; id++) {
            if (inputToOutDim[id] === d && id !== dim) {
              baseOffset = baseOffset.add(coord.mul(ctx.u32(inputStrides[id])));
            }
          }
        }
      }
      baseOffset = ctx.emitLet("ar_base", baseOffset);

      // Sequential search for max/min
      const initVal = isMax ? -3.402823466e+38 : 3.402823466e+38;
      const bestVal = ctx.emitVar("bestVal", "f32", ctx.f32(initVal));
      const bestIdx = ctx.emitVar("bestIdx", "u32", ctx.u32(0));

      ctx.forRange(ctx.u32(0), ctx.uniform("dimSize"), (i) => {
        const val = ctx.load("input", baseOffset.add(i.mul(ctx.uniform("dimStride"))));
        const cond = isMax ? val.gt(bestVal.get()) : val.lt(bestVal.get());
        ctx.ifThen(cond, () => {
          bestVal.set(val);
          bestIdx.set(i);
        });
      });

      ctx.emitStore("out", outIdx, bestIdx.get().toF32());
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Where (ternary select) Kernel
// ============================================================================

/**
 * Generate WGSL for broadcast where: out[idx] = cond ? x : y
 */
export function whereWGSL(
  indexShape: number[],
  condStrides: number[],
  xStrides: number[],
  yStrides: number[],
  condOffset: number,
  xOffset: number,
  yOffset: number,
): string {
  const spec: TileKernelSpec = {
    name: "where",
    workgroupSize: WG,
    bindings: {
      cond: { storage: "read", type: "f32" },
      x: { storage: "read", type: "f32" },
      y: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

      const condOff = buildStridedOffset(ctx, idx, indexShape, condStrides, condOffset, "co");
      const xOff = buildStridedOffset(ctx, idx, indexShape, xStrides, xOffset, "xo");
      const yOff = buildStridedOffset(ctx, idx, indexShape, yStrides, yOffset, "yo");

      const condVal = ctx.load("cond", condOff);
      const xVal = ctx.load("x", xOff);
      const yVal = ctx.load("y", yOff);
      // select(falseVal, trueVal, condition) in WGSL
      const result = condVal.ne(ctx.f32(0)).select(xVal, yVal);
      ctx.emitStore("out", idx, result);
    },
  };
  return compileTileKernel(spec);
}

