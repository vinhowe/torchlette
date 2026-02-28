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
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

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
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

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
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

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
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

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
      const idx = ctx.flatGlobalId(WG);
      const total = ctx.u32(outSize);
      ctx.ifThen(idx.ge(total), () => ctx.emitReturn());

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
  const rank = viewShape.length;
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
        let rem: BlockExpr = idx;
        let baseOff: BlockExpr = ctx.u32(viewOffset);
        let srcOff: BlockExpr = ctx.u32(srcOffset);

        for (let d = 0; d < rank; d++) {
          let dimStride = 1;
          for (let j = d + 1; j < rank; j++) dimStride *= viewShape[j];

          const coord = d < rank - 1
            ? ctx.emitLet(`sc_c${d}`, rem.div(ctx.u32(dimStride)))
            : rem;

          if (d < rank - 1) {
            rem = ctx.emitLet(`sc_r${d}`, rem.mod(ctx.u32(dimStride)));
          }

          baseOff = baseOff.add(coord.mul(ctx.u32(viewStrides[d])));
          srcOff = srcOff.add(coord.mul(ctx.u32(srcStrides[d])));
        }

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
  const rank = viewShape.length;
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
        let rem: BlockExpr = idx;
        let baseOff: BlockExpr = ctx.u32(viewOffset);
        let srcOff: BlockExpr = ctx.u32(srcOffset);

        for (let d = 0; d < rank; d++) {
          let dimStride = 1;
          for (let j = d + 1; j < rank; j++) dimStride *= viewShape[j];

          const coord = d < rank - 1
            ? ctx.emitLet(`sa_c${d}`, rem.div(ctx.u32(dimStride)))
            : rem;

          if (d < rank - 1) {
            rem = ctx.emitLet(`sa_r${d}`, rem.mod(ctx.u32(dimStride)));
          }

          baseOff = baseOff.add(coord.mul(ctx.u32(viewStrides[d])));
          srcOff = srcOff.add(coord.mul(ctx.u32(srcStrides[d])));
        }

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
  const rank = inputShape.length;
  const inputStrides = contiguousStridesForShape(inputShape);
  const indexStrides = contiguousStridesForShape(indexShape);

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
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());

      // Decompose idx into coordinates using index strides
      let rem: BlockExpr = idx;
      const coords: BlockExpr[] = [];
      for (let d = 0; d < rank; d++) {
        const coord = d < rank - 1
          ? ctx.emitLet(`g_c${d}`, rem.div(ctx.u32(indexStrides[d])))
          : rem;
        if (d < rank - 1) {
          rem = ctx.emitLet(`g_r${d}`, rem.mod(ctx.u32(indexStrides[d])));
        }
        coords.push(coord);
      }

      // Get gather index from indices buffer
      const gatherIdx = ctx.emitLet("gatherIdx", ctx.load("indices", idx).toU32());

      // Compute input offset
      let inputOffset: BlockExpr = ctx.u32(0);
      for (let d = 0; d < rank; d++) {
        if (d === dim) {
          inputOffset = inputOffset.add(gatherIdx.mul(ctx.u32(inputStrides[d])));
        } else {
          inputOffset = inputOffset.add(coords[d].mul(ctx.u32(inputStrides[d])));
        }
      }

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
  const rank = inputShape.length;
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
      const srcIdx = ctx.flatGlobalId(WG);
      ctx.ifThen(srcIdx.ge(ctx.uniform("srcSize")), () => ctx.emitReturn());

      // Decompose srcIdx into coordinates
      let rem: BlockExpr = srcIdx;
      const coords: BlockExpr[] = [];
      for (let d = 0; d < rank; d++) {
        const coord = d < rank - 1
          ? ctx.emitLet(`s_c${d}`, rem.div(ctx.u32(srcStrides[d])))
          : rem;
        if (d < rank - 1) {
          rem = ctx.emitLet(`s_r${d}`, rem.mod(ctx.u32(srcStrides[d])));
        }
        coords.push(coord);
      }

      // Get scatter index from indices buffer
      const scatterIdx = ctx.emitLet("scatterIdx", ctx.load("indices", srcIdx).toU32());

      // Compute output offset
      let outOffset: BlockExpr = ctx.u32(0);
      for (let d = 0; d < rank; d++) {
        if (d === dim) {
          outOffset = outOffset.add(scatterIdx.mul(ctx.u32(outStrides[d])));
        } else {
          outOffset = outOffset.add(coords[d].mul(ctx.u32(outStrides[d])));
        }
      }

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
// Flat Contiguous Copy / Add Specs (for chunked strided-scatter paths)
// ============================================================================

/**
 * TileKernelSpec for a flat contiguous copy: out[idx] = src[idx].
 * Used by chunked stridedScatterCopy when both src and dest are contiguous.
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
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      ctx.emitStore("out", idx, ctx.load("src", idx));
    },
  };
}

/**
 * TileKernelSpec for a flat contiguous add: out[idx] = base[idx] + src[idx].
 * Used by chunked stridedScatterAdd when both base and src are contiguous.
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
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.flatGlobalId(WG);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      ctx.emitStore("out", idx, ctx.load("base", idx).add(ctx.load("src", idx)));
    },
  };
}
