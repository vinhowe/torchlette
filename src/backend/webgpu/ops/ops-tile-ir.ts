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

import { applyFusedOp } from "../fusion-tile-ir";
import {
  contiguousStrides,
  F32_NEG_MAX,
  F32_POS_MAX,
  MAX_WORKGROUPS_PER_DIM,
  sizeOf,
  WORKGROUP_SIZE,
} from "../shape-utils";
import { compileTileKernel } from "../tile-compiler";
import {
  type BlockExpr,
  type DataType,
  elementwiseGrid,
  elementwiseKernel,
  type KernelContext,
  type TileKernelSpec,
} from "../tile-ir";

const WG = WORKGROUP_SIZE; // 256
/** Apply chunked index guard: early-return if index outside [chunkStart, chunkEnd), adjust to chunk-local. */
function applyChunkedGuard(
  ctx: KernelContext,
  rawIdx: BlockExpr,
  chunked: boolean,
): BlockExpr {
  if (!chunked) return rawIdx;
  ctx.ifThen(rawIdx.lt(ctx.uniform("chunkStart")), () => ctx.emitReturn());
  ctx.ifThen(rawIdx.ge(ctx.uniform("chunkEnd")), () => ctx.emitReturn());
  return rawIdx.sub(ctx.uniform("chunkStart"));
}

/** Grid for a compile-time-known element count (no uniform needed). */
function fixedElementGrid(
  workgroupSize: number,
  elements: number,
): (u: Record<string, number>) => [number] | [number, number] {
  const totalWg = Math.ceil(elements / workgroupSize);
  if (totalWg <= MAX_WORKGROUPS_PER_DIM) return () => [totalWg];
  return () => [
    Math.min(totalWg, MAX_WORKGROUPS_PER_DIM),
    Math.ceil(totalWg / MAX_WORKGROUPS_PER_DIM),
  ];
}

// ============================================================================
// Fill Kernel
// ============================================================================

export function fillWGSL(): string {
  return compileTileKernel(
    elementwiseKernel({
      name: "fill",
      bindings: { out: { storage: "read_write", type: "f32" } },
      uniforms: { value: "f32" },
      kernel(ctx, idx) {
        ctx.emitStore("out", idx, ctx.uniform("value"));
      },
    }),
  );
}

// ============================================================================
// Arange Kernel
// ============================================================================

export function arangeWGSL(): string {
  return compileTileKernel(
    elementwiseKernel({
      name: "arange",
      bindings: { out: { storage: "read_write", type: "f32" } },
      uniforms: { start: "f32", step: "f32" },
      kernel(ctx, idx) {
        ctx.emitStore(
          "out",
          idx,
          ctx.uniform("start").add(idx.toF32().mul(ctx.uniform("step"))),
        );
      },
    }),
  );
}

// ============================================================================
// Triangular (tril/triu) Kernel
// ============================================================================

export function triangularWGSL(upper: boolean): string {
  return compileTileKernel(
    elementwiseKernel({
      name: upper ? "triu" : "tril",
      bindings: {
        input: { storage: "read", type: "f32" },
        output: { storage: "read_write", type: "f32" },
      },
      uniforms: { H: "u32", W: "u32", k: "i32" },
      sizeUniform: "num_elements",
      kernel(ctx, idx) {
        const W = ctx.uniform("W");
        const row = idx.div(W).mod(ctx.uniform("H")).toI32();
        const col = idx.mod(W).toI32();
        const k = ctx.uniform("k");
        const cond = upper ? col.lt(row.add(k)) : col.gt(row.add(k));
        ctx.ifThenElse(
          cond,
          () => ctx.emitStore("output", idx, ctx.f32(0)),
          () => ctx.emitStore("output", idx, ctx.load("input", idx)),
        );
      },
    }),
  );
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
  const opToMethod: Record<string, (a: BlockExpr, b: BlockExpr) => BlockExpr> =
    {
      ">": (a, b) => a.gt(b),
      "<": (a, b) => a.lt(b),
      ">=": (a, b) => a.ge(b),
      "<=": (a, b) => a.le(b),
      "==": (a, b) => a.eq(b),
      "!=": (a, b) => a.ne(b),
    };
  const cmpFn = opToMethod[wgslOp];
  if (!cmpFn) throw new Error(`Unknown comparison op: ${wgslOp}`);

  return compileTileKernel(
    elementwiseKernel({
      name: `cmp_${wgslOp}`,
      bindings: {
        a: { storage: "read", type: "f32" },
        b: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      kernel(ctx, idx) {
        const aVal = ctx.stridedLoad("a", idx, indexShape, aStrides, aOffset);
        const bVal = ctx.stridedLoad("b", idx, indexShape, bStrides, bOffset);
        ctx.emitStore(
          "out",
          idx,
          cmpFn(aVal, bVal).select(ctx.f32(1), ctx.f32(0)),
        );
      },
    }),
  );
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
  dim: number,
  inputToOutDim: number[],
): string {
  const rank = inputShape.length;
  const isMax = compareOp === ">";
  const name = isMax ? "argmax" : "argmin";

  const spec = elementwiseKernel({
    name: `${name}_d${dim}_${inputShape.join("x")}`,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { dimSize: "u32", dimStride: "u32" },
    sizeUniform: "outSize",
    kernel(ctx, outIdx) {
      // Compute base offset in input (with reduce dim = 0)
      // Decompose outIdx into output coordinates, map to input strides
      let baseOffset: BlockExpr = ctx.u32(0);
      if (outShape.length > 0) {
        const outCoords = ctx.decomposeIndex(outIdx, outShape);
        for (let d = 0; d < outShape.length; d++) {
          // Map output coord back to input dimension
          for (let id = 0; id < rank; id++) {
            if (inputToOutDim[id] === d && id !== dim) {
              baseOffset = baseOffset.add(
                outCoords[d].mul(ctx.u32(inputStrides[id])),
              );
            }
          }
        }
      }

      // Sequential search for max/min
      const initVal = isMax ? F32_NEG_MAX : F32_POS_MAX;
      const bestVal = ctx.emitVar("bestVal", "f32", ctx.f32(initVal));
      const bestIdx = ctx.emitVar("bestIdx", "u32", ctx.u32(0));

      ctx.forRange(ctx.u32(0), ctx.uniform("dimSize"), (i) => {
        const val = ctx.load(
          "input",
          baseOffset.add(i.mul(ctx.uniform("dimStride"))),
        );
        const cond = isMax ? val.gt(bestVal.get()) : val.lt(bestVal.get());
        ctx.ifThen(cond, () => {
          bestVal.set(val);
          bestIdx.set(i);
        });
      });

      ctx.emitStore("out", outIdx, bestIdx.get().toF32());
    },
  });
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
  return elementwiseKernel({
    name: "where",
    bindings: {
      cond: { storage: "read", type: "f32" },
      x: { storage: "read", type: "f32" },
      y: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    kernel(ctx, idx) {
      const condVal = ctx.stridedLoad(
        "cond",
        idx,
        indexShape,
        condStrides,
        condOffset,
      );
      const xVal = ctx.stridedLoad("x", idx, indexShape, xStrides, xOffset);
      const yVal = ctx.stridedLoad("y", idx, indexShape, yStrides, yOffset);
      ctx.emitStore("out", idx, condVal.ne(ctx.f32(0)).select(xVal, yVal));
    },
  });
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
  return compileTileKernel(
    whereSpec(
      indexShape,
      condStrides,
      xStrides,
      yStrides,
      condOffset,
      xOffset,
      yOffset,
    ),
  );
}

// ============================================================================
// Binary Broadcast Kernel (tile-IR)
// ============================================================================

/** WGSL infix operator → fusion op name */
const WGSL_OP_TO_FUSION: Record<string, string> = {
  "+": "add",
  "-": "sub",
  "*": "mul",
  "/": "div",
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

  return elementwiseKernel({
    name: `binary_${fusionOp}`,
    enableF16: dtype === "f16",
    bindings: {
      a: { storage: "read", type: dtype },
      b: { storage: "read", type: dtype },
      out: { storage: "read_write", type: dtype },
    },
    kernel(ctx, idx) {
      const aVal = ctx.stridedLoad("a", idx, indexShape, aStrides, aOffset);
      const bVal = ctx.stridedLoad("b", idx, indexShape, bStrides, bOffset);
      ctx.emitStore("out", idx, applyFusedOp(ctx, fusionOp, [aVal, bVal]));
    },
  });
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
  return compileTileKernel(
    binaryBroadcastSpec(
      op,
      indexShape,
      aStrides,
      bStrides,
      aOffset,
      bOffset,
      dtype,
    ),
  );
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
  return elementwiseKernel({
    name: `unary_${opKey}`,
    enableF16: dtype === "f16",
    bindings: {
      a: { storage: "read", type: dtype },
      out: { storage: "read_write", type: dtype },
    },
    kernel(ctx, idx) {
      ctx.emitStore(
        "out",
        idx,
        applyFusedOp(ctx, opKey, [
          ctx.stridedLoad("a", idx, shape, strides, offset),
        ]),
      );
    },
  });
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
  return compileTileKernel(
    unaryStridedSpec(opKey, shape, strides, offset, dtype),
  );
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

  return elementwiseKernel({
    name: `cast_${srcDtype}_${dstDtype}`,
    enableF16: srcDtype === "f16" || dstDtype === "f16",
    bindings: {
      a: { storage: "read", type: srcDtype },
      out: { storage: "read_write", type: dstDtype },
    },
    kernel(ctx, idx) {
      const val = ctx.stridedLoad("a", idx, shape, strides, offset);
      ctx.emitStore(
        "out",
        idx,
        srcDtype === dstDtype ? val : applyFusedOp(ctx, castOp, [val]),
      );
    },
  });
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
  return compileTileKernel(
    castSpec(srcDtype, dstDtype, shape, strides, offset),
  );
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
  return compileTileKernel(
    elementwiseKernel({
      name: `contiguous_${dtype}`,
      enableF16: dtype === "f16",
      bindings: {
        input: { storage: "read", type: dtype },
        out: { storage: "read_write", type: dtype },
      },
      kernel(ctx, idx) {
        ctx.emitStore(
          "out",
          idx,
          ctx.stridedLoad("input", idx, shape, strides, offset),
        );
      },
    }),
  );
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
      const zero = dtype === "f16" ? ctx.f16(0) : ctx.f32(0);
      ctx.ifThen(dimIdx.lt(startU), () => {
        ctx.emitStore("out", idx, zero);
        ctx.emitReturn();
      });
      ctx.ifThen(dimIdx.ge(endU), () => {
        ctx.emitStore("out", idx, zero);
        ctx.emitReturn();
      });

      const gradDimIdx = ctx.emitLet("gradDimIdx", dimIdx.sub(startU));
      const gradIdx = outerIdx
        .mul(gradDimSize)
        .mul(innerSize)
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
/**
 * Unified strided scatter kernel: copy or add.
 * Two-phase: copy base to output, then scatter src values at view positions.
 */
function stridedScatterTileIR(
  op: "copy" | "add",
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
): string {
  const viewSize = sizeOf(viewShape);

  const spec: TileKernelSpec = {
    name: op === "copy" ? "stridedScatterCopy" : "stridedScatterAdd",
    workgroupSize: WG,
    bindings: {
      base: { storage: "read", type: "f32" },
      src: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { baseSize: "u32", viewSize: "u32" },
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
        const srcVal = ctx.load("src", srcOff);
        ctx.emitStore(
          "out",
          baseOff,
          op === "add" ? ctx.load("out", baseOff).add(srcVal) : srcVal,
        );
      });
    },
  };
  return compileTileKernel(spec);
}

export function stridedScatterCopyTileIR(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
): string {
  return stridedScatterTileIR(
    "copy",
    baseSize,
    viewShape,
    viewStrides,
    viewOffset,
    srcStrides,
    srcOffset,
  );
}

export function stridedScatterAddTileIR(
  baseSize: number,
  viewShape: number[],
  viewStrides: number[],
  viewOffset: number,
  srcStrides: number[],
  srcOffset: number,
): string {
  return stridedScatterTileIR(
    "add",
    baseSize,
    viewShape,
    viewStrides,
    viewOffset,
    srcStrides,
    srcOffset,
  );
}

// ============================================================================
// Gather Kernel
// ============================================================================

/**
 * Generate WGSL for gather op via tile-IR.
 * out[idx] = input[offset with gather dim replaced by indices[idx]]
 */
/**
 * Unified gather kernel: regular or chunked.
 * Chunked mode adds chunk range guards and adjusts indices to chunk-local coordinates.
 */
function gatherTileIRImpl(
  inputShape: number[],
  indexShape: number[],
  dim: number,
  chunked: boolean,
): string {
  const inputStrides = contiguousStrides(inputShape);
  const uniforms: Record<string, "u32"> = {};
  if (chunked) {
    uniforms.chunkStart = "u32";
    uniforms.chunkEnd = "u32";
  }

  return compileTileKernel(
    elementwiseKernel({
      name: chunked ? `gather_chunked_d${dim}` : `gather_d${dim}`,
      bindings: {
        input: { storage: "read", type: "f32" },
        indices: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms,
      kernel(ctx, idx) {
        const gatherIdx = ctx.emitLet(
          "gatherIdx",
          ctx.load("indices", idx).toU32(),
        );
        const dimIdx = applyChunkedGuard(ctx, gatherIdx, chunked);

        const coords = ctx.decomposeIndex(idx, indexShape);
        const inputCoords = coords.map((c, d) => (d === dim ? dimIdx : c));
        ctx.emitStore(
          "out",
          idx,
          ctx.load("input", ctx.linearizeIndex(inputCoords, inputStrides)),
        );
      },
    }),
  );
}

export function gatherTileIR(
  inputShape: number[],
  indexShape: number[],
  dim: number,
): string {
  return gatherTileIRImpl(inputShape, indexShape, dim, false);
}

export function chunkedGatherTileIR(
  inputShape: number[],
  indexShape: number[],
  dim: number,
): string {
  return gatherTileIRImpl(inputShape, indexShape, dim, true);
}

/**
 * Unified scatter_add kernel: regular or chunked.
 * Caller must copy input→output before dispatching this kernel.
 */
function scatterAddTileIRImpl(
  inputShape: number[],
  srcShape: number[],
  dim: number,
  chunked: boolean,
): string {
  const outStrides = contiguousStrides(inputShape);
  const uniforms: Record<string, "u32"> = {};
  if (chunked) {
    uniforms.chunkStart = "u32";
    uniforms.chunkEnd = "u32";
  }

  return compileTileKernel(
    elementwiseKernel({
      name: chunked ? `scatterAdd_chunked_d${dim}` : `scatterAdd_d${dim}`,
      bindings: {
        indices: { storage: "read", type: "f32" },
        src: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms,
      sizeUniform: "srcSize",
      kernel(ctx, srcIdx) {
        const scatterIdx = ctx.emitLet(
          "scatterIdx",
          ctx.load("indices", srcIdx).toU32(),
        );
        const dimIdx = applyChunkedGuard(ctx, scatterIdx, chunked);

        const coords = ctx.decomposeIndex(srcIdx, srcShape);
        const outCoords = coords.map((c, d) => (d === dim ? dimIdx : c));
        const outOffset = ctx.linearizeIndex(outCoords, outStrides);
        ctx.emitStore(
          "out",
          outOffset,
          ctx.load("out", outOffset).add(ctx.load("src", srcIdx)),
        );
      },
    }),
  );
}

export function scatterAddTileIR(
  inputShape: number[],
  srcShape: number[],
  dim: number,
): string {
  return scatterAddTileIRImpl(inputShape, srcShape, dim, false);
}

export function chunkedScatterAddTileIR(
  inputShape: number[],
  srcShape: number[],
  dim: number,
): string {
  return scatterAddTileIRImpl(inputShape, srcShape, dim, true);
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
      K: "u32",
      N: "u32",
      rowStart: "u32",
      rowEnd: "u32",
      colStart: "u32",
      colEnd: "u32",
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
  const flatWgId = ctx
    .programId(0)
    .add(ctx.programId(1).mul(ctx.numPrograms(0)));
  return ctx.emitLet(
    "base",
    flatWgId.mul(ctx.u32(WG * VEC)).add(ctx.localIndex().mul(ctx.u32(VEC))),
  );
}

/**
 * Unified flat contiguous spec: copy (out=src) or add (out=base+src).
 * Vec4 unrolled: each thread processes VEC consecutive elements.
 */
function flatOpSpec(op: "copy" | "add"): TileKernelSpec {
  const bindings: Record<
    string,
    { storage: "read" | "read_write"; type: "f32" }
  > =
    op === "copy"
      ? {
          src: { storage: "read", type: "f32" },
          out: { storage: "read_write", type: "f32" },
        }
      : {
          base: { storage: "read", type: "f32" },
          src: { storage: "read", type: "f32" },
          out: { storage: "read_write", type: "f32" },
        };

  return {
    name: op === "copy" ? "flatCopy" : "flatAdd",
    workgroupSize: WG,
    bindings,
    uniforms: { size: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "size", vecWidth: VEC }),
    kernel(ctx) {
      const b = flatVec4Base(ctx);
      const size = ctx.uniform("size");
      ctx.ifThen(b.ge(size), () => ctx.emitReturn());
      for (let v = 0; v < VEC; v++) {
        const idx = b.add(ctx.u32(v));
        const val =
          op === "copy"
            ? ctx.blockLoad("src", idx, size)
            : ctx
                .blockLoad("base", idx, size)
                .add(ctx.blockLoad("src", idx, size));
        ctx.blockStore("out", idx, size, val);
      }
    },
  };
}

export function flatCopySpec(): TileKernelSpec {
  return flatOpSpec("copy");
}
export function flatAddSpec(): TileKernelSpec {
  return flatOpSpec("add");
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
    uniforms: {
      numRows: "u32",
      N: "u32",
      colStart: "u32",
      sliceWidth: "u32",
      rowStart: "u32",
    },
    grid: elementwiseGrid(WG),
    kernel(ctx) {
      const idx = ctx.elementIndex(
        WG,
        ctx.uniform("numRows").mul(ctx.uniform("sliceWidth")),
      );

      const localRow = idx.div(ctx.uniform("sliceWidth"));
      const col = idx.mod(ctx.uniform("sliceWidth"));
      const srcCol = ctx.uniform("colStart").add(col);
      // Input offset relative to chunk start (row 0 of bound range)
      const srcIdx = localRow.mul(ctx.uniform("N")).add(srcCol);
      // Output accounts for rowStart
      const dstIdx = ctx
        .uniform("rowStart")
        .add(localRow)
        .mul(ctx.uniform("sliceWidth"))
        .add(col);

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
    uniforms: {
      numRows: "u32",
      N: "u32",
      colStart: "u32",
      sliceWidth: "u32",
      rowStart: "u32",
      inputRowStart: "u32",
    },
    grid: elementwiseGrid(WG),
    kernel(ctx) {
      const idx = ctx.elementIndex(
        WG,
        ctx.uniform("numRows").mul(ctx.uniform("sliceWidth")),
      );

      const localRow = idx.div(ctx.uniform("sliceWidth"));
      const col = idx.mod(ctx.uniform("sliceWidth"));

      // Input: relative to bound chunk
      const inputIdx = ctx
        .uniform("inputRowStart")
        .add(localRow)
        .mul(ctx.uniform("sliceWidth"))
        .add(col);
      // Output: write to local row, column colStart + col (relative to bound chunk)
      const outputIdx = localRow
        .mul(ctx.uniform("N"))
        .add(ctx.uniform("colStart").add(col));

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
    uniforms: {
      batch: "u32",
      K: "u32",
      N: "u32",
      colStart: "u32",
      chunkWidth: "u32",
    },
    grid: elementwiseGrid(WG),
    kernel(ctx) {
      const idx = ctx.elementIndex(
        WG,
        ctx
          .uniform("batch")
          .mul(ctx.uniform("K"))
          .mul(ctx.uniform("chunkWidth")),
      );

      // Decompose flat idx to (batchIdx, k, c) in output space
      const cw = ctx.uniform("chunkWidth");
      const c = idx.mod(cw);
      const k = idx.div(cw).mod(ctx.uniform("K"));
      const batchIdx = idx.div(ctx.uniform("K").mul(cw));

      // Input offset: (batchIdx, k, colStart + c) in [batch, K, N]
      const inputOffset = batchIdx
        .mul(ctx.uniform("K").mul(ctx.uniform("N")))
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
  return compileTileKernel(
    elementwiseKernel({
      name: "rand",
      bindings: { out: { storage: "read_write", type: "f32" } },
      uniforms: { seed: "u32" },
      kernel(ctx, idx) {
        ctx.emitStore("out", idx, ctx.randF32(ctx.uniform("seed"), idx));
      },
    }),
  );
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
      if (totalWg <= MAX_WORKGROUPS_PER_DIM) return [totalWg];
      return [
        Math.min(totalWg, MAX_WORKGROUPS_PER_DIM),
        Math.ceil(totalWg / MAX_WORKGROUPS_PER_DIM),
      ];
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
      // Second value with masked store (like tl.store with mask)
      const outIdx2 = outIdx.add(ctx.u32(1));
      ctx.blockStore("out", outIdx2, size, r.mul(theta.sin()));
    },
  };
  return compileTileKernel(spec);
}

/**
 * Bernoulli random f32: 1.0 with probability p, 0.0 otherwise.
 */
export function bernoulliWGSL(): string {
  return compileTileKernel(
    elementwiseKernel({
      name: "bernoulli",
      bindings: { out: { storage: "read_write", type: "f32" } },
      uniforms: { seed: "u32", prob: "f32" },
      kernel(ctx, idx) {
        const u = ctx.randF32(ctx.uniform("seed"), idx);
        ctx.emitStore(
          "out",
          idx,
          u.lt(ctx.uniform("prob")).select(ctx.f32(1.0), ctx.f32(0.0)),
        );
      },
    }),
  );
}
