/**
 * Tile-IR Reduction Kernel Factories
 *
 * Generates tile-IR kernel specs for all reduction operations:
 * sum (dim/full/chunked/preamble), max (dim/full), mean division.
 *
 * Factory functions take shape metadata and return TileKernelSpec instances.
 * Shapes are baked in as constants — the constant folding pass optimizes
 * aggressively (x*1→x, x/1→x, dead code elimination for rank-1 tensors).
 */

import {
  type TileKernelSpec,
  type KernelContext,
  type BlockExpr,
  type BindingSpec,
  elementwiseGrid,
  singleWorkgroup,
} from "./tile-ir";
import { compileTileKernel } from "./tile-compiler";
import { applyFusedOp } from "./fusion-tile-ir";
import { WORKGROUP_SIZE } from "./shape-utils";

const WG = WORKGROUP_SIZE; // 256

// ============================================================================
// Index Computation
// ============================================================================

/**
 * Build a tile-IR expression that computes the linear input offset
 * from (outIdx, reduceIdx) using shape/stride metadata.
 *
 * All shape values are known at kernel-build time and become constants,
 * so the constant folder eliminates dead coordinates and trivial arithmetic.
 */
function buildInputOffset(
  ctx: KernelContext,
  outIdx: BlockExpr,
  reduceIdx: BlockExpr,
  inputShape: number[],
  inputStrides: number[],
  outShape: number[],
  outStrides: number[],
  normalizedDims: number[],
  inputToOutDim: number[],
  prefix: string,
): BlockExpr {
  const rank = inputShape.length;

  // Decompose outIdx into output coordinates
  const outCoords: BlockExpr[] = [];
  if (outShape.length > 0) {
    let rem = ctx.emitLet(`${prefix}_or`, outIdx);
    for (let d = 0; d < outShape.length; d++) {
      const stride = outStrides[d];
      if (d < outShape.length - 1) {
        outCoords.push(ctx.emitLet(`${prefix}_oc${d}`, rem.div(ctx.u32(stride))));
        rem = ctx.emitLet(`${prefix}_or${d}`, rem.mod(ctx.u32(stride)));
      } else {
        outCoords.push(rem); // last coord is the remainder
      }
    }
  }

  // Decompose reduceIdx into reduction coordinates
  const reduceCoords: BlockExpr[] = [];
  if (normalizedDims.length > 0) {
    let rrem = ctx.emitLet(`${prefix}_rr`, reduceIdx);
    for (let i = 0; i < normalizedDims.length; i++) {
      // Compute size of remaining reduce dims
      let rDimSize = 1;
      for (let j = i + 1; j < normalizedDims.length; j++) {
        rDimSize *= inputShape[normalizedDims[j]];
      }
      if (i < normalizedDims.length - 1) {
        reduceCoords.push(ctx.emitLet(`${prefix}_rc${i}`, rrem.div(ctx.u32(rDimSize))));
        rrem = ctx.emitLet(`${prefix}_rr${i}`, rrem.mod(ctx.u32(rDimSize)));
      } else {
        reduceCoords.push(rrem);
      }
    }
  }

  // Build the linear input offset from coordinates
  let offset: BlockExpr = ctx.u32(0);
  let reduceCoordIdx = 0;
  for (let d = 0; d < rank; d++) {
    let coord: BlockExpr;
    if (normalizedDims.includes(d)) {
      coord = reduceCoords[reduceCoordIdx++];
    } else {
      const outD = inputToOutDim[d];
      if (outD >= 0 && outD < outCoords.length) {
        coord = outCoords[outD];
      } else {
        coord = ctx.u32(0);
      }
    }
    // offset += coord * inputStrides[d]
    offset = offset.add(coord.mul(ctx.u32(inputStrides[d])));
  }

  return ctx.emitLet(`${prefix}_off`, offset);
}

// ============================================================================
// Sum Dim Reduction
// ============================================================================

/**
 * Factory: Dimension-wise sum reduction kernel spec.
 *
 * parallel=true: 1 workgroup (256 threads) per output element, uses wgReduce.
 * parallel=false: 1 thread per output element, sequential accumulation.
 */
export function makeSumDimSpec(
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
): TileKernelSpec {
  const variant = parallel ? "par" : "seq";
  const name = `sumDim_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  if (parallel) {
    // Parallel: 1 workgroup per output element
    return {
      name,
      workgroupSize: WG,
      bindings: {
        input: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: {
        outSize: "u32",
        reductionSize: "u32",
      },
      grid: (u) => {
        const n = u.outSize;
        if (n <= 65535) return [n];
        return [Math.min(n, 65535), Math.ceil(n / 65535)];
      },
      kernel(ctx) {
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const outIdx = ctx.emitLet("outIdx", wid);

        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("sum", tid, reductionSize, WG, (r) => {
          const off = buildInputOffset(ctx, outIdx, r,
            inputShape, inputStrides, outShape, outStrides,
            normalizedDims, inputToOutDim, "sp");
          return ctx.load("input", off);
        });

        ctx.guardedStore("out", tid.eq(ctx.u32(0)), outIdx, result);
      },
    };
  }

  // Sequential: 1 thread per output element
  return {
    name,
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      outSize: "u32",
      reductionSize: "u32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      const reductionSize = ctx.uniform("reductionSize");
      const acc = ctx.emitVar("total", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), reductionSize, (r) => {
        const off = buildInputOffset(ctx, idx, r,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "ss");
        acc.addAssign(ctx.load("input", off));
      });
      ctx.emitStore("out", idx, acc.get());
    },
  };
}

// ============================================================================
// Sum Full Reduction
// ============================================================================

/**
 * Factory: Full reduction sum to a single scalar (0-d tensor).
 * Single workgroup with wgReduce for subgroup optimization.
 */
export function makeSumFullSpec(): TileKernelSpec {
  return {
    name: "sumFull",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("sum", tid, ctx.uniform("size"), WG, (i) =>
        ctx.load("input", i),
      );
      ctx.guardedStore("out", tid.eq(ctx.u32(0)), ctx.u32(0), result);
    },
  };
}

// ============================================================================
// Sum Full Chunked (per-chunk + final kernels)
// ============================================================================

/**
 * Get compiled WGSL for the per-chunk sum kernel.
 * Each workgroup sums one chunk and writes to out[chunkIdx].
 *
 * We compile the WGSL via tile-IR but dispatch manually (the chunked path
 * needs buffer offset/size in bind group entries which tile-dispatch
 * doesn't support directly).
 */
export function getChunkedSumWGSL(): string {
  const spec: TileKernelSpec = {
    name: "sumChunk",
    workgroupSize: 1,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      chunkSize: "u32",
      chunkIdx: "u32",
    },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const acc = ctx.emitVar("sum", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), ctx.uniform("chunkSize"), (i) => {
        acc.addAssign(ctx.load("input", i));
      });
      ctx.emitStore("out", ctx.uniform("chunkIdx"), acc.get());
    },
  };
  return compileTileKernel(spec);
}

/**
 * Get compiled WGSL for the final partials sum kernel (same as sumFull but workgroup_size(1)).
 */
export function getFinalSumWGSL(): string {
  const spec: TileKernelSpec = {
    name: "sumFinalPartials",
    workgroupSize: 1,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const acc = ctx.emitVar("sum", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), ctx.uniform("size"), (i) => {
        acc.addAssign(ctx.load("input", i));
      });
      ctx.emitStore("out", ctx.u32(0), acc.get());
    },
  };
  return compileTileKernel(spec);
}

// ============================================================================
// Max Dim Reduction
// ============================================================================

/**
 * Factory: Dimension-wise max reduction kernel spec.
 *
 * parallel=true: 1 workgroup per output element, uses wgReduce("max").
 * parallel=false: 1 thread per output element, sequential loop.
 */
export function makeMaxDimSpec(
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
): TileKernelSpec {
  const variant = parallel ? "par" : "seq";
  const name = `maxDim_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  if (parallel) {
    return {
      name,
      workgroupSize: WG,
      bindings: {
        input: { storage: "read", type: "f32" },
        out: { storage: "read_write", type: "f32" },
      },
      uniforms: {
        outSize: "u32",
        reductionSize: "u32",
      },
      grid: (u) => {
        const n = u.outSize;
        if (n <= 65535) return [n];
        return [Math.min(n, 65535), Math.ceil(n / 65535)];
      },
      kernel(ctx) {
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const outIdx = ctx.emitLet("outIdx", wid);

        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("max", tid, reductionSize, WG, (r) => {
          const off = buildInputOffset(ctx, outIdx, r,
            inputShape, inputStrides, outShape, outStrides,
            normalizedDims, inputToOutDim, "mp");
          return ctx.load("input", off);
        });

        ctx.guardedStore("out", tid.eq(ctx.u32(0)), outIdx, result);
      },
    };
  }

  // Sequential
  return {
    name,
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      outSize: "u32",
      reductionSize: "u32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      // First element as initial max value
      const firstOff = buildInputOffset(ctx, idx, ctx.u32(0),
        inputShape, inputStrides, outShape, outStrides,
        normalizedDims, inputToOutDim, "ms0");
      const maxVal = ctx.emitVar("maxVal", "f32", ctx.load("input", firstOff));

      ctx.forRange(ctx.u32(1), ctx.uniform("reductionSize"), (r) => {
        const off = buildInputOffset(ctx, idx, r,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "ms");
        maxVal.set(maxVal.get().max(ctx.load("input", off)));
      });
      ctx.emitStore("out", idx, maxVal.get());
    },
  };
}

// ============================================================================
// Max Full Reduction
// ============================================================================

/**
 * Factory: Full reduction max to a single scalar (0-d tensor).
 */
export function makeMaxFullSpec(): TileKernelSpec {
  return {
    name: "maxFull",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("max", tid, ctx.uniform("size"), WG, (i) =>
        ctx.load("input", i),
      );
      ctx.guardedStore("out", tid.eq(ctx.u32(0)), ctx.u32(0), result);
    },
  };
}

// ============================================================================
// Mean Division
// ============================================================================

/**
 * Factory: Elementwise division kernel for mean (out[i] = input[i] / count).
 */
export function makeMeanDivSpec(): TileKernelSpec {
  return {
    name: "meanDiv",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      size: "u32",
      count: "f32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "size" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("size")), () => ctx.emitReturn());
      ctx.emitStore("out", idx, ctx.load("input", idx).div(ctx.uniform("count")));
    },
  };
}

// ============================================================================
// Sum Dim With Preamble (fused elementwise + reduce)
// ============================================================================

/**
 * Factory: Dimension-wise sum with fused elementwise preamble.
 * Instead of `total += input[offset]`, computes `total += preambleOp(in0[offset], in1[offset])`.
 */
export function makeSumDimWithPreambleSpec(
  preambleOp: string,
  arity: number,
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
): TileKernelSpec {
  const name = `sumPreamble_${preambleOp}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < arity; i++) {
    bindings[`in${i}`] = { storage: "read", type: "f32" };
  }
  bindings.out = { storage: "read_write", type: "f32" };

  return {
    name,
    workgroupSize: WG,
    bindings,
    uniforms: {
      outSize: "u32",
      reductionSize: "u32",
    },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      const acc = ctx.emitVar("total", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), ctx.uniform("reductionSize"), (reduceIdx) => {
        const off = buildInputOffset(ctx, idx, reduceIdx,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "sp");
        const inputs: BlockExpr[] = [];
        for (let i = 0; i < arity; i++) {
          inputs.push(ctx.load(`in${i}`, off));
        }
        acc.addAssign(applyFusedOp(ctx, preambleOp, inputs));
      });
      ctx.emitStore("out", idx, acc.get());
    },
  };
}

// ============================================================================
// Sum Full With Preamble
// ============================================================================

/**
 * Factory: Full reduction sum with fused elementwise preamble.
 * Single workgroup, single thread sequential loop.
 */
export function makeSumFullWithPreambleSpec(
  preambleOp: string,
  arity: number,
): TileKernelSpec {
  const name = `sumFullPreamble_${preambleOp}`;

  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < arity; i++) {
    bindings[`in${i}`] = { storage: "read", type: "f32" };
  }
  bindings.out = { storage: "read_write", type: "f32" };

  return {
    name,
    workgroupSize: 1,
    bindings,
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const acc = ctx.emitVar("sum", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), ctx.uniform("size"), (i) => {
        const inputs: BlockExpr[] = [];
        for (let j = 0; j < arity; j++) {
          inputs.push(ctx.load(`in${j}`, i));
        }
        acc.addAssign(applyFusedOp(ctx, preambleOp, inputs));
      });
      ctx.emitStore("out", ctx.u32(0), acc.get());
    },
  };
}
