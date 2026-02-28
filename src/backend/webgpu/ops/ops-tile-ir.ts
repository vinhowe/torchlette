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
  type DataType,
  elementwiseGrid,
} from "../tile-ir";
import { compileTileKernel } from "../tile-compiler";
import { WORKGROUP_SIZE } from "../shape-utils";
import { applyFusedOp } from "../fusion-tile-ir";

const WG = WORKGROUP_SIZE; // 256
const MAX_WG_PER_DIM = 65535;

/** Grid for a compile-time-known element count (no uniform needed). */
function fixedElementGrid(workgroupSize: number, elements: number): (u: Record<string, number>) => [number] | [number, number] {
  const totalWg = Math.ceil(elements / workgroupSize);
  if (totalWg <= MAX_WG_PER_DIM) return () => [totalWg];
  return () => [Math.min(totalWg, MAX_WG_PER_DIM), Math.ceil(totalWg / MAX_WG_PER_DIM)];
}

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
  _prefix: string,
): BlockExpr {
  const rank = indexShape.length;
  if (rank === 0) return ctx.u32(offset);

  const coords = ctx.decomposeIndex(flatIdx, indexShape);
  return ctx.linearizeIndex(coords, strides, offset);
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
      // Decompose outIdx into output coordinates, map to input strides
      let baseOffset: BlockExpr = ctx.u32(0);
      if (outShape.length > 0) {
        const outCoords = ctx.decomposeIndex(outIdx, outShape);
        for (let d = 0; d < outShape.length; d++) {
          // Map output coord back to input dimension
          for (let id = 0; id < rank; id++) {
            if (inputToOutDim[id] === d && id !== dim) {
              baseOffset = baseOffset.add(outCoords[d].mul(ctx.u32(inputStrides[id])));
            }
          }
        }
      }

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
 * Build a TileKernelSpec for broadcast where: out[idx] = cond ? x : y
 * Used by both direct dispatch (compiled to WGSL) and chunked dispatch.
 */
export function whereSpec(
  indexShape: number[],
  condStrides: number[],
  xStrides: number[],
  yStrides: number[],
  condOffset: number,
  xOffset: number,
  yOffset: number,
): TileKernelSpec {
  return {
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
}

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
  return compileTileKernel(whereSpec(indexShape, condStrides, xStrides, yStrides, condOffset, xOffset, yOffset));
}

// ============================================================================
// Binary Broadcast Kernel (tile-IR)
// ============================================================================

/** WGSL infix operator → fusion op name */
const WGSL_OP_TO_FUSION: Record<string, string> = {
  "+": "add", "-": "sub", "*": "mul", "/": "div",
};

/**
 * Build a TileKernelSpec for a binary broadcast op.
 * Used by both direct dispatch (compiled to WGSL) and chunked dispatch.
 */
export function binaryBroadcastSpec(
  op: string,
  indexShape: number[],
  aStrides: number[],
  bStrides: number[],
  aOffset: number,
  bOffset: number,
  dtype: DataType,
): TileKernelSpec {
  const fusionOp = WGSL_OP_TO_FUSION[op];
  if (!fusionOp) throw new Error(`binaryBroadcastSpec: unsupported op "${op}"`);

  return {
    name: `binary_${fusionOp}`,
    workgroupSize: WG,
    enableF16: dtype === "f16",
    bindings: {
      a: { storage: "read", type: dtype },
      b: { storage: "read", type: dtype },
      out: { storage: "read_write", type: dtype },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG);

      const aOff = buildStridedOffset(ctx, idx, indexShape, aStrides, aOffset, "ao");
      const bOff = buildStridedOffset(ctx, idx, indexShape, bStrides, bOffset, "bo");

      const aVal = ctx.load("a", aOff);
      const bVal = ctx.load("b", bOff);
      const result = applyFusedOp(ctx, fusionOp, [aVal, bVal]);
      ctx.emitStore("out", idx, result);
    },
  };
}

/**
 * Generate WGSL for a binary broadcast op via tile-IR.
 * Drop-in replacement for binaryBroadcastShader() in dispatch.ts.
 */
export function binaryBroadcastTileIR(
  op: string,
  indexShape: number[],
  aStrides: number[],
  bStrides: number[],
  aOffset: number,
  bOffset: number,
  dtype: DataType,
): string {
  return compileTileKernel(binaryBroadcastSpec(op, indexShape, aStrides, bStrides, aOffset, bOffset, dtype));
}

// ============================================================================
// Unary Strided Kernel (tile-IR)
// ============================================================================

/**
 * Build a TileKernelSpec for a strided unary op.
 * Used by both direct dispatch (compiled to WGSL) and chunked dispatch.
 */
export function unaryStridedSpec(
  opKey: string,
  shape: number[],
  strides: number[],
  offset: number,
  dtype: DataType,
): TileKernelSpec {
  return {
    name: `unary_${opKey}`,
    workgroupSize: WG,
    enableF16: dtype === "f16",
    bindings: {
      a: { storage: "read", type: dtype },
      out: { storage: "read_write", type: dtype },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG);

      const inputOff = buildStridedOffset(ctx, idx, shape, strides, offset, "in");
      const val = ctx.load("a", inputOff);
      const result = applyFusedOp(ctx, opKey, [val]);
      ctx.emitStore("out", idx, result);
    },
  };
}

/**
 * Generate WGSL for a strided unary op via tile-IR.
 * Drop-in replacement for unaryStridedShader() in dispatch.ts.
 *
 * @param opKey - Fusion op name (e.g. "sqrt", "relu", "neg", "tanh", etc.)
 */
export function unaryStridedTileIR(
  opKey: string,
  shape: number[],
  strides: number[],
  offset: number,
  dtype: DataType,
): string {
  return compileTileKernel(unaryStridedSpec(opKey, shape, strides, offset, dtype));
}

// ============================================================================
// Cast Kernel (tile-IR)
// ============================================================================

/**
 * Build a TileKernelSpec for a strided dtype cast.
 * Used by both direct dispatch (compiled to WGSL) and chunked dispatch.
 */
export function castSpec(
  srcDtype: DataType,
  dstDtype: DataType,
  shape: number[],
  strides: number[],
  offset: number,
): TileKernelSpec {
  const castOp = `cast_${dstDtype}`;

  return {
    name: `cast_${srcDtype}_${dstDtype}`,
    workgroupSize: WG,
    enableF16: srcDtype === "f16" || dstDtype === "f16",
    bindings: {
      a: { storage: "read", type: srcDtype },
      out: { storage: "read_write", type: dstDtype },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG);

      const inputOff = buildStridedOffset(ctx, idx, shape, strides, offset, "in");
      const val = ctx.load("a", inputOff);
      const result = srcDtype === dstDtype ? val : applyFusedOp(ctx, castOp, [val]);
      ctx.emitStore("out", idx, result);
    },
  };
}

/**
 * Generate WGSL for a strided dtype cast via tile-IR.
 * Drop-in replacement for castShader() in views.ts.
 */
export function castTileIR(
  srcDtype: DataType,
  dstDtype: DataType,
  shape: number[],
  strides: number[],
  offset: number,
): string {
  return compileTileKernel(castSpec(srcDtype, dstDtype, shape, strides, offset));
}

// ============================================================================
// Contiguous (strided copy) Kernel
// ============================================================================

/**
 * Generate WGSL for a strided-to-contiguous copy via tile-IR.
 * Drop-in replacement for contiguousDirect() shader in views.ts.
 */
export function contiguousTileIR(
  shape: number[],
  strides: number[],
  offset: number,
  dtype: DataType,
): string {
  const spec: TileKernelSpec = {
    name: `contiguous_${dtype}`,
    workgroupSize: WG,
    enableF16: dtype === "f16",
    bindings: {
      input: { storage: "read", type: dtype },
      out: { storage: "read_write", type: dtype },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG);

      const inputOff = buildStridedOffset(ctx, idx, shape, strides, offset, "in");
      ctx.emitStore("out", idx, ctx.load("input", inputOff));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// NarrowBackward Kernel
// ============================================================================

/**
 * Generate WGSL for narrow backward via tile-IR.
 * Pads gradient back to original shape along dim.
 */
export function narrowBackwardTileIR(
  outDimSize: number,
  outSize: number,
  dtype: DataType,
): string {
  const spec: TileKernelSpec = {
    name: `narrowBackward_${dtype}`,
    workgroupSize: WG,
    enableF16: dtype === "f16",
    bindings: {
      grad: { storage: "read", type: dtype },
      out: { storage: "read_write", type: dtype },
    },
    uniforms: {
      outerSize: "u32",
      innerSize: "u32",
      gradDimSize: "u32",
      start: "u32",
    },
    grid: fixedElementGrid(WG, outSize),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG, ctx.u32(outSize));

      const innerSize = ctx.uniform("innerSize");
      const dimSize = ctx.u32(outDimSize);
      const outerIdx = ctx.emitLet("outerIdx", idx.div(dimSize.mul(innerSize)));
      const remainder = ctx.emitLet("rem", idx.mod(dimSize.mul(innerSize)));
      const dimIdx = ctx.emitLet("dimIdx", remainder.div(innerSize));
      const innerIdx = ctx.emitLet("innerIdx", remainder.mod(innerSize));

      const startU = ctx.uniform("start");
      const gradDimSize = ctx.uniform("gradDimSize");
      const endU = ctx.emitLet("endU", startU.add(gradDimSize));

      // Check if dimIdx is in [start, start + gradDimSize)
      // Use nested ifs to avoid out-of-bounds grad read
      ctx.ifThen(dimIdx.lt(startU), () => {
        const zero = dtype === "f16" ? ctx.f16(0) : ctx.f32(0);
        ctx.emitStore("out", idx, zero);
        ctx.emitReturn();
      });
      ctx.ifThen(dimIdx.ge(endU), () => {
        const zero = dtype === "f16" ? ctx.f16(0) : ctx.f32(0);
        ctx.emitStore("out", idx, zero);
        ctx.emitReturn();
      });

      const gradDimIdx = ctx.emitLet("gradDimIdx", dimIdx.sub(startU));
      const gradIdx = outerIdx.mul(gradDimSize).mul(innerSize)
        .add(gradDimIdx.mul(innerSize))
        .add(innerIdx);
      ctx.emitStore("out", idx, ctx.load("grad", gradIdx));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// StridedScatterCopy Kernel
// ============================================================================

/**
 * Generate WGSL for strided scatter copy via tile-IR.
 * Two-phase: copy base to output, then scatter src into view positions.
 */
export function stridedScatterCopyTileIR(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
): string {
  const viewSize = viewShape.reduce((a, b) => a * b, 1);

  const spec: TileKernelSpec = {
    name: "stridedScatterCopy",
    workgroupSize: WG,
    bindings: {
      base: { storage: "read", type: "f32" },
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      baseSize: "u32",
      viewSize: "u32",
    },
    grid: fixedElementGrid(WG, Math.max(baseSize, viewSize)),
    kernel(ctx) {
      const idx = ctx.flatGlobalId(WG);

      // Phase 1: copy base to output
      ctx.ifThen(idx.lt(ctx.uniform("baseSize")), () => {
        ctx.emitStore("out", idx, ctx.load("base", idx));
      });

      ctx.barrier();

      // Phase 2: scatter src values into output at view positions
      ctx.ifThen(idx.lt(ctx.uniform("viewSize")), () => {
        const coords = ctx.decomposeIndex(idx, viewShape);
        const baseOff = ctx.linearizeIndex(coords, viewStrides, viewOffset);
        const srcOff = ctx.linearizeIndex(coords, srcStrides, srcOffset);
        ctx.emitStore("out", baseOff, ctx.load("src", srcOff));
      });
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// StridedScatterAdd Kernel
// ============================================================================

/**
 * Generate WGSL for strided scatter add via tile-IR.
 * Two-phase: copy base to output, then add src values at view positions.
 */
export function stridedScatterAddTileIR(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
): string {
  const viewSize = viewShape.reduce((a, b) => a * b, 1);

  const spec: TileKernelSpec = {
    name: "stridedScatterAdd",
    workgroupSize: WG,
    bindings: {
      base: { storage: "read", type: "f32" },
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      baseSize: "u32",
      viewSize: "u32",
    },
    grid: fixedElementGrid(WG, Math.max(baseSize, viewSize)),
    kernel(ctx) {
      const idx = ctx.flatGlobalId(WG);

      // Phase 1: copy base to output
      ctx.ifThen(idx.lt(ctx.uniform("baseSize")), () => {
        ctx.emitStore("out", idx, ctx.load("base", idx));
      });

      ctx.barrier();

      // Phase 2: add src values into output at view positions
      ctx.ifThen(idx.lt(ctx.uniform("viewSize")), () => {
        const coords = ctx.decomposeIndex(idx, viewShape);
        const baseOff = ctx.linearizeIndex(coords, viewStrides, viewOffset);
        const srcOff = ctx.linearizeIndex(coords, srcStrides, srcOffset);
        const existing = ctx.load("out", baseOff);
        ctx.emitStore("out", baseOff, existing.add(ctx.load("src", srcOff)));
      });
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Gather Kernel
// ============================================================================

/**
 * Generate WGSL for gather op via tile-IR.
 * out[idx] = input[offset with gather dim replaced by indices[idx]]
 */
export function gatherTileIR(
  inputShape: number[],
  indexShape: number[],
  dim: number,
): string {
  const inputStrides = contiguousStridesForShape(inputShape);

  const spec: TileKernelSpec = {
    name: `gather_d${dim}`,
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      indices: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG);

      // Decompose flat index into multi-dimensional coordinates
      const coords = ctx.decomposeIndex(idx, indexShape);

      // Get gather index from indices buffer
      const gatherIdx = ctx.emitLet("gatherIdx", ctx.load("indices", idx).toU32());

      // Compute input offset: replace dim coordinate with gather index
      const inputCoords = coords.map((c, d) => d === dim ? gatherIdx : c);
      const inputOffset = ctx.linearizeIndex(inputCoords, inputStrides);

      ctx.emitStore("out", idx, ctx.load("input", inputOffset));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// ScatterAdd Kernel
// ============================================================================

/**
 * Generate WGSL for scatter_add op via tile-IR.
 * Caller must copy input→output before dispatching this kernel.
 */
export function scatterAddTileIR(
  inputShape: number[],
  srcShape: number[],
  dim: number,
): string {
  const outStrides = contiguousStridesForShape(inputShape);
  const srcStrides = contiguousStridesForShape(srcShape);

  const spec: TileKernelSpec = {
    name: `scatterAdd_d${dim}`,
    workgroupSize: WG,
    bindings: {
      indices: { storage: "read", type: "f32" },
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { srcSize: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "srcSize" }),
    kernel(ctx) {
      const srcIdx = ctx.elementIndex(WG, "srcSize");

      // Decompose flat src index into multi-dimensional coordinates
      const coords = ctx.decomposeIndex(srcIdx, srcShape);

      // Get scatter index from indices buffer
      const scatterIdx = ctx.emitLet("scatterIdx", ctx.load("indices", srcIdx).toU32());

      // Compute output offset: replace dim coordinate with scatter index
      const outCoords = coords.map((c, d) => d === dim ? scatterIdx : c);
      const outOffset = ctx.linearizeIndex(outCoords, outStrides);

      const existing = ctx.load("out", outOffset);
      ctx.emitStore("out", outOffset, existing.add(ctx.load("src", srcIdx)));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Chunked Gather Kernel
// ============================================================================

/**
 * Generate WGSL for chunked gather op via tile-IR.
 * Each dispatch handles a chunk of the input buffer along the gather dimension.
 * Threads whose gather index falls outside [chunkStart, chunkEnd) are skipped.
 * The gather index is adjusted to local coordinates within the chunk.
 */
export function chunkedGatherTileIR(
  inputShape: number[],
  indexShape: number[],
  dim: number,
): string {
  const inputStrides = contiguousStridesForShape(inputShape);

  const spec: TileKernelSpec = {
    name: `gather_chunked_d${dim}`,
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      indices: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32", chunkStart: "u32", chunkEnd: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG);

      // Get gather index and skip if outside current chunk range
      const gatherIdx = ctx.emitLet("gatherIdx", ctx.load("indices", idx).toU32());
      ctx.ifThen(gatherIdx.lt(ctx.uniform("chunkStart")), () => ctx.emitReturn());
      ctx.ifThen(gatherIdx.ge(ctx.uniform("chunkEnd")), () => ctx.emitReturn());

      // Adjust to chunk-local index
      const localIdx = gatherIdx.sub(ctx.uniform("chunkStart"));

      // Decompose flat index into multi-dimensional coordinates
      const coords = ctx.decomposeIndex(idx, indexShape);

      // Compute input offset: replace dim coordinate with local gather index
      const inputCoords = coords.map((c, d) => d === dim ? localIdx : c);
      const inputOffset = ctx.linearizeIndex(inputCoords, inputStrides);

      ctx.emitStore("out", idx, ctx.load("input", inputOffset));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Chunked ScatterAdd Kernel
// ============================================================================

/**
 * Generate WGSL for chunked scatter_add op via tile-IR.
 * Each dispatch handles a chunk of the output buffer along the scatter dimension.
 * Threads whose scatter index falls outside [chunkStart, chunkEnd) are skipped.
 * The scatter index is adjusted to local coordinates within the chunk.
 */
export function chunkedScatterAddTileIR(
  inputShape: number[],
  srcShape: number[],
  dim: number,
): string {
  const outStrides = contiguousStridesForShape(inputShape);
  const srcStrides = contiguousStridesForShape(srcShape);

  const spec: TileKernelSpec = {
    name: `scatterAdd_chunked_d${dim}`,
    workgroupSize: WG,
    bindings: {
      indices: { storage: "read", type: "f32" },
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { srcSize: "u32", chunkStart: "u32", chunkEnd: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "srcSize" }),
    kernel(ctx) {
      const srcIdx = ctx.elementIndex(WG, "srcSize");

      // Get scatter index and skip if outside current chunk range
      const scatterIdx = ctx.emitLet("scatterIdx", ctx.load("indices", srcIdx).toU32());
      ctx.ifThen(scatterIdx.lt(ctx.uniform("chunkStart")), () => ctx.emitReturn());
      ctx.ifThen(scatterIdx.ge(ctx.uniform("chunkEnd")), () => ctx.emitReturn());

      // Adjust to chunk-local index
      const localIdx = scatterIdx.sub(ctx.uniform("chunkStart"));

      // Decompose flat src index into multi-dimensional coordinates
      const coords = ctx.decomposeIndex(srcIdx, srcShape);

      // Compute output offset: replace dim coordinate with local scatter index
      const outCoords = coords.map((c, d) => d === dim ? localIdx : c);
      const outOffset = ctx.linearizeIndex(outCoords, outStrides);

      const existing = ctx.load("out", outOffset);
      ctx.emitStore("out", outOffset, existing.add(ctx.load("src", srcIdx)));
    },
  };
  return compileTileKernel(spec);
}

/** Helper: compute contiguous strides for a given shape. */
function contiguousStridesForShape(shape: number[]): number[] {
  const strides: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    let stride = 1;
    for (let j = i + 1; j < shape.length; j++) stride *= shape[j];
    strides.push(stride);
  }
  return strides;
}

// ============================================================================
// Chunked Transpose Kernel
// ============================================================================

/**
 * Generate WGSL for chunked 2D transpose via tile-IR.
 * Processes tiles defined by [rowStart, rowEnd) × [colStart, colEnd).
 * For a transposed tensor [K, N] with strides [1, K]:
 *   input[localCol * K + globalRow] → output[localRow * N + globalCol]
 *
 * @param dtype - Element dtype ("f32", "f16", etc.)
 */
export function chunkedTransposeTileIR(dtype: DataType): string {
  const spec: TileKernelSpec = {
    name: "contiguous_chunked_transpose",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: dtype },
      output: { storage: "read_write", type: dtype },
    },
    uniforms: {
      K: "u32", N: "u32",
      rowStart: "u32", rowEnd: "u32",
      colStart: "u32", colEnd: "u32",
      gridStride: "u32",
    },
    kernel(ctx) {
      const numRows = ctx.uniform("rowEnd").sub(ctx.uniform("rowStart"));
      const numCols = ctx.uniform("colEnd").sub(ctx.uniform("colStart"));
      const idx = ctx.elementIndex(WG, numRows.mul(numCols));

      const localRow = idx.div(numCols);
      const localCol = idx.mod(numCols);
      const globalRow = ctx.uniform("rowStart").add(localRow);
      const globalCol = ctx.uniform("colStart").add(localCol);

      // Input: transposed [globalRow, globalCol] = buffer[globalCol * K + globalRow]
      // With chunk offset at colStart * K, local index = localCol * K + globalRow
      const inputIdx = localCol.mul(ctx.uniform("K")).add(globalRow);

      // Output: row chunk is bound with offset
      // localRow * N + globalCol
      const outputIdx = localRow.mul(ctx.uniform("N")).add(globalCol);

      ctx.emitStore("output", outputIdx, ctx.load("input", inputIdx));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Flat Contiguous Copy / Add Specs (for chunked strided-scatter paths)
// ============================================================================

/** Vec4 unroll factor for flat specs. Each thread processes VEC consecutive elements. */
const VEC = 4;

/**
 * Helper: compute a vec4-unrolled flat thread base index.
 * Each thread processes VEC consecutive elements starting at this base.
 * Handles 2D dispatch overflow via programId(1) * numPrograms(0).
 */
function flatVec4Base(ctx: KernelContext): BlockExpr {
  const flatWgId = ctx.programId(0).add(
    ctx.programId(1).mul(ctx.numPrograms(0)),
  );
  return ctx.emitLet("base",
    flatWgId.mul(ctx.u32(WG * VEC)).add(ctx.localIndex().mul(ctx.u32(VEC))),
  );
}

/**
 * TileKernelSpec for a flat contiguous copy: out[idx] = src[idx].
 * Used by chunked stridedScatterCopy when both src and dest are contiguous.
 * Vec4 unrolled: each thread copies VEC consecutive elements.
 */
export function flatCopySpec(): TileKernelSpec {
  return {
    name: "flatCopy",
    workgroupSize: WG,
    bindings: {
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size", vecWidth: VEC }),
    kernel(ctx) {
      const base = flatVec4Base(ctx);
      const size = ctx.uniform("size");
      // Early exit: if base >= size, all VEC elements are out of bounds
      ctx.ifThen(base.ge(size), () => ctx.emitReturn());
      for (let v = 0; v < VEC; v++) {
        const idx = base.add(ctx.u32(v));
        ctx.ifThen(idx.lt(size), () => {
          ctx.emitStore("out", idx, ctx.load("src", idx));
        });
      }
    },
  };
}

/**
 * TileKernelSpec for a flat contiguous add: out[idx] = base[idx] + src[idx].
 * Used by chunked stridedScatterAdd when both base and src are contiguous.
 * Vec4 unrolled: each thread adds VEC consecutive elements.
 */
export function flatAddSpec(): TileKernelSpec {
  return {
    name: "flatAdd",
    workgroupSize: WG,
    bindings: {
      base: { storage: "read", type: "f32" },
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size", vecWidth: VEC }),
    kernel(ctx) {
      const b = flatVec4Base(ctx);
      const size = ctx.uniform("size");
      // Early exit: if base >= size, all VEC elements are out of bounds
      ctx.ifThen(b.ge(size), () => ctx.emitReturn());
      for (let v = 0; v < VEC; v++) {
        const idx = b.add(ctx.u32(v));
        ctx.ifThen(idx.lt(size), () => {
          ctx.emitStore("out", idx, ctx.load("base", idx).add(ctx.load("src", idx)));
        });
      }
    },
  };
}

// ============================================================================
// Matmul Column Slice / Scatter Kernels
// ============================================================================

/**
 * Extract column slice from a 2D matrix to a contiguous buffer.
 * Input: [numRows, N] (may be offset-bound for chunking), Output: [K, sliceWidth].
 * Uniforms: numRows, N, colStart, sliceWidth, rowStart.
 */
export function sliceColumnsTileIR(): string {
  const spec: TileKernelSpec = {
    name: "sliceColumns",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { numRows: "u32", N: "u32", colStart: "u32", sliceWidth: "u32", rowStart: "u32" },
    grid: elementwiseGrid(WG),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG, ctx.uniform("numRows").mul(ctx.uniform("sliceWidth")));

      const localRow = idx.div(ctx.uniform("sliceWidth"));
      const col = idx.mod(ctx.uniform("sliceWidth"));
      const srcCol = ctx.uniform("colStart").add(col);
      // Input offset relative to chunk start (row 0 of bound range)
      const srcIdx = localRow.mul(ctx.uniform("N")).add(srcCol);
      // Output accounts for rowStart
      const dstIdx = ctx.uniform("rowStart").add(localRow).mul(ctx.uniform("sliceWidth")).add(col);

      ctx.emitStore("output", dstIdx, ctx.load("input", srcIdx));
    },
  };
  return compileTileKernel(spec);
}

/**
 * Write partial matmul result to columns of output buffer.
 * Input: [M, sliceWidth], Output: [M, N], writes to columns [colStart, colStart+sliceWidth).
 * Uniforms: numRows, N, colStart, sliceWidth, rowStart (unused, kept for compat), inputRowStart.
 */
export function scatterColumnsTileIR(): string {
  const spec: TileKernelSpec = {
    name: "scatterColumns",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { numRows: "u32", N: "u32", colStart: "u32", sliceWidth: "u32", rowStart: "u32", inputRowStart: "u32" },
    grid: elementwiseGrid(WG),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG, ctx.uniform("numRows").mul(ctx.uniform("sliceWidth")));

      const localRow = idx.div(ctx.uniform("sliceWidth"));
      const col = idx.mod(ctx.uniform("sliceWidth"));

      // Input: relative to bound chunk
      const inputIdx = ctx.uniform("inputRowStart").add(localRow).mul(ctx.uniform("sliceWidth")).add(col);
      // Output: write to local row, column colStart + col (relative to bound chunk)
      const outputIdx = localRow.mul(ctx.uniform("N")).add(ctx.uniform("colStart").add(col));

      ctx.emitStore("output", outputIdx, ctx.load("input", inputIdx));
    },
  };
  return compileTileKernel(spec);
}

/**
 * Slice columns from B matrix for chunked output matmul.
 * Input: [batch, K, N], Output: [batch, K, chunkWidth].
 * Uniforms: batch, K, N, colStart, chunkWidth.
 */
export function sliceBColumnsTileIR(): string {
  const spec: TileKernelSpec = {
    name: "sliceBColumns",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: { batch: "u32", K: "u32", N: "u32", colStart: "u32", chunkWidth: "u32" },
    grid: elementwiseGrid(WG),
    kernel(ctx) {
      const idx = ctx.elementIndex(WG, ctx.uniform("batch").mul(ctx.uniform("K")).mul(ctx.uniform("chunkWidth")));

      // Decompose flat idx to (batchIdx, k, c) in output space
      const cw = ctx.uniform("chunkWidth");
      const c = idx.mod(cw);
      const k = idx.div(cw).mod(ctx.uniform("K"));
      const batchIdx = idx.div(ctx.uniform("K").mul(cw));

      // Input offset: (batchIdx, k, colStart + c) in [batch, K, N]
      const inputOffset = batchIdx.mul(ctx.uniform("K").mul(ctx.uniform("N")))
        .add(k.mul(ctx.uniform("N")))
        .add(ctx.uniform("colStart").add(c));

      ctx.emitStore("output", idx, ctx.load("input", inputOffset));
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// RNG Kernels (Philox 2x32-10)
// ============================================================================

/**
 * Uniform random f32 in [0, 1).
 * Each thread generates one value using Philox 2x32-10.
 */
export function randWGSL(): string {
  const spec: TileKernelSpec = {
    name: "rand",
    workgroupSize: WG,
    bindings: {
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      size: "u32",
      seed: "u32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      ctx.emitStore("out", idx, ctx.randF32(ctx.uniform("seed"), idx));
    },
  };
  return compileTileKernel(spec);
}

/**
 * Standard normal (Gaussian) random f32 via Box-Muller transform.
 * Each thread processes a pair of elements using two Philox outputs.
 */
export function randnWGSL(): string {
  const spec: TileKernelSpec = {
    name: "randn",
    workgroupSize: WG,
    bindings: {
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      size: "u32",
      seed: "u32",
    },
    // Half the threads: each produces 2 values
    grid(u) {
      const numThreads = Math.ceil(u.size / 2);
      const totalWg = Math.ceil(numThreads / WG);
      if (totalWg <= MAX_WG_PER_DIM) return [totalWg];
      return [Math.min(totalWg, MAX_WG_PER_DIM), Math.ceil(totalWg / MAX_WG_PER_DIM)];
    },
    kernel(ctx) {
      const idx = ctx.globalId(0);
      const outIdx = idx.mul(ctx.u32(2));
      const size = ctx.uniform("size");
      ctx.ifThen(outIdx.ge(size), () => ctx.emitReturn());

      // Two uniform randoms from a single Philox call
      const [u1raw, u2] = ctx.randF32x2(ctx.uniform("seed"), idx);
      // Clamp u1 to (0, 1] to avoid log(0): max(u1, epsilon)
      const u1 = u1raw.max(ctx.f32(1.1754944e-38));
      // Box-Muller transform
      const r = u1.log().mul(ctx.f32(-2.0)).sqrt();
      const theta = u2.mul(ctx.f32(2.0 * Math.PI));
      ctx.emitStore("out", outIdx, r.mul(theta.cos()));
      // Second value with bounds check
      ctx.guardedStore("out", outIdx.add(ctx.u32(1)).lt(size), outIdx.add(ctx.u32(1)), r.mul(theta.sin()));
    },
  };
  return compileTileKernel(spec);
}

/**
 * Bernoulli random f32: 1.0 with probability p, 0.0 otherwise.
 */
export function bernoulliWGSL(): string {
  const spec: TileKernelSpec = {
    name: "bernoulli",
    workgroupSize: WG,
    bindings: {
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      size: "u32",
      seed: "u32",
      prob: "f32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      const u = ctx.randF32(ctx.uniform("seed"), idx);
      ctx.emitStore("out", idx, u.lt(ctx.uniform("prob")).select(ctx.f32(1.0), ctx.f32(0.0)));
    },
  };
  return compileTileKernel(spec);
}
