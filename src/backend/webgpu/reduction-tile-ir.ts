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
  type DataType,
  elementwiseGrid,
  singleWorkgroup,
} from "./tile-ir";
import { compileTileKernel } from "./tile-compiler";
import { applyFusedOp } from "./fusion-tile-ir";
import { WORKGROUP_SIZE } from "./shape-utils";
import type { DType } from "../types";

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
    let rem = outIdx;
    for (let d = 0; d < outShape.length; d++) {
      const stride = outStrides[d];
      if (d < outShape.length - 1) {
        outCoords.push(rem.div(ctx.u32(stride)));
        rem = rem.mod(ctx.u32(stride));
      } else {
        outCoords.push(rem); // last coord is the remainder
      }
    }
  }

  // Decompose reduceIdx into reduction coordinates
  const reduceCoords: BlockExpr[] = [];
  if (normalizedDims.length > 0) {
    let rrem = reduceIdx;
    for (let i = 0; i < normalizedDims.length; i++) {
      // Compute size of remaining reduce dims
      let rDimSize = 1;
      for (let j = i + 1; j < normalizedDims.length; j++) {
        rDimSize *= inputShape[normalizedDims[j]];
      }
      if (i < normalizedDims.length - 1) {
        reduceCoords.push(rrem.div(ctx.u32(rDimSize)));
        rrem = rrem.mod(ctx.u32(rDimSize));
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

  return offset;
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
        const outIdx = wid;

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
        const outIdx = wid;

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
// Min Dim Reduction
// ============================================================================

export function makeMinDimSpec(
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
): TileKernelSpec {
  const variant = parallel ? "par" : "seq";
  const name = `minDim_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

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
        const outIdx = wid;

        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("min", tid, reductionSize, WG, (r) => {
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

      // First element as initial min value
      const firstOff = buildInputOffset(ctx, idx, ctx.u32(0),
        inputShape, inputStrides, outShape, outStrides,
        normalizedDims, inputToOutDim, "ms0");
      const minVal = ctx.emitVar("minVal", "f32", ctx.load("input", firstOff));

      ctx.forRange(ctx.u32(1), ctx.uniform("reductionSize"), (r) => {
        const off = buildInputOffset(ctx, idx, r,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "ms");
        minVal.set(minVal.get().min(ctx.load("input", off)));
      });
      ctx.emitStore("out", idx, minVal.get());
    },
  };
}

// ============================================================================
// Min Full Reduction
// ============================================================================

export function makeMinFullSpec(): TileKernelSpec {
  return {
    name: "minFull",
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: "f32" },
    },
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("min", tid, ctx.uniform("size"), WG, (i) =>
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

// ============================================================================
// Preamble Chain Types & Helper
// ============================================================================

/** Describes a single preamble chain op for kernel generation. */
export type PreambleChainKernelOp = {
  /** applyFusedOp-compatible op name (e.g., "mul", "cast_f32", "relu") */
  op: string;
  /** Number of inputs: 1 for unary, 2 for binary */
  arity: number;
  /** For non-first binary ops: which input position has the chain result (0 or 1) */
  chainInputPos?: 0 | 1;
};

/**
 * Apply a chain of preamble ops in-register within the reduction loop body.
 * External inputs are loaded from in0, in1, ... bindings at the given offset.
 *
 * First op: all inputs are external bindings.
 * Subsequent ops: chain result is one input, optional external is the other.
 */
function applyPreambleChain(
  ctx: KernelContext,
  chainOps: PreambleChainKernelOp[],
  offset: BlockExpr,
): BlockExpr {
  let externalIdx = 0;
  let result: BlockExpr;

  // First op: all inputs are external
  const firstInput = ctx.load(`in${externalIdx++}`, offset);
  if (chainOps[0].arity === 1) {
    result = applyFusedOp(ctx, chainOps[0].op, [firstInput]);
  } else {
    const secondInput = ctx.load(`in${externalIdx++}`, offset);
    result = applyFusedOp(ctx, chainOps[0].op, [firstInput, secondInput]);
  }

  // Remaining ops: chain result + optional external input
  for (let i = 1; i < chainOps.length; i++) {
    const op = chainOps[i];
    if (op.arity === 1) {
      result = applyFusedOp(ctx, op.op, [result]);
    } else {
      const ext = ctx.load(`in${externalIdx++}`, offset);
      if (op.chainInputPos === 1) {
        result = applyFusedOp(ctx, op.op, [ext, result]);
      } else {
        result = applyFusedOp(ctx, op.op, [result, ext]);
      }
    }
  }

  return result;
}

// ============================================================================
// Sum Dim With Preamble Chain
// ============================================================================

/**
 * Factory: Dimension-wise sum with fused multi-op preamble chain.
 * Applies a chain of elementwise ops in the accumulation loop body.
 */
export function makeSumDimWithPreambleChainSpec(
  chainOps: PreambleChainKernelOp[],
  totalExternalInputs: number,
  inputDtypes: DType[],
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
): TileKernelSpec {
  const opNames = chainOps.map(o => o.op).join("+");
  const variant = parallel ? "par" : "seq";
  const name = `sumPC_${opNames}_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < totalExternalInputs; i++) {
    bindings[`in${i}`] = { storage: "read", type: (inputDtypes[i] || "f32") as DataType };
  }
  bindings.out = { storage: "read_write", type: "f32" };

  const needsF16 = inputDtypes.some(d => d === "f16");

  if (parallel) {
    return {
      name,
      workgroupSize: WG,
      bindings,
      enableF16: needsF16,
      uniforms: { outSize: "u32", reductionSize: "u32" },
      grid: (u) => {
        const n = u.outSize;
        if (n <= 65535) return [n];
        return [Math.min(n, 65535), Math.ceil(n / 65535)];
      },
      kernel(ctx) {
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const outIdx = wid;
        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("sum", tid, reductionSize, WG, (r) => {
          const off = buildInputOffset(ctx, outIdx, r,
            inputShape, inputStrides, outShape, outStrides,
            normalizedDims, inputToOutDim, "spc");
          return applyPreambleChain(ctx, chainOps, off);
        });

        ctx.guardedStore("out", tid.eq(ctx.u32(0)), outIdx, result);
      },
    };
  }

  // Sequential
  return {
    name,
    workgroupSize: WG,
    bindings,
    enableF16: needsF16,
    uniforms: { outSize: "u32", reductionSize: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      const acc = ctx.emitVar("total", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), ctx.uniform("reductionSize"), (reduceIdx) => {
        const off = buildInputOffset(ctx, idx, reduceIdx,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "spc");
        acc.addAssign(applyPreambleChain(ctx, chainOps, off));
      });
      ctx.emitStore("out", idx, acc.get());
    },
  };
}

// ============================================================================
// Sum Full With Preamble Chain
// ============================================================================

/**
 * Factory: Full reduction sum with fused multi-op preamble chain.
 * Single workgroup with wgReduce.
 */
export function makeSumFullWithPreambleChainSpec(
  chainOps: PreambleChainKernelOp[],
  totalExternalInputs: number,
  inputDtypes: DType[],
): TileKernelSpec {
  const opNames = chainOps.map(o => o.op).join("+");
  const name = `sumFullPC_${opNames}`;

  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < totalExternalInputs; i++) {
    bindings[`in${i}`] = { storage: "read", type: (inputDtypes[i] || "f32") as DataType };
  }
  bindings.out = { storage: "read_write", type: "f32" };

  const needsF16 = inputDtypes.some(d => d === "f16");

  return {
    name,
    workgroupSize: WG,
    bindings,
    enableF16: needsF16,
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("sum", tid, ctx.uniform("size"), WG, (i) =>
        applyPreambleChain(ctx, chainOps, i),
      );
      ctx.guardedStore("out", tid.eq(ctx.u32(0)), ctx.u32(0), result);
    },
  };
}

// ============================================================================
// Epilogue Chain Helper
// ============================================================================

/** Describes a single epilogue op to apply after a reduction. */
export type ReductionEpilogueOpDesc = {
  kind: string;
  toDtype?: DType;
  inputIndex?: number;
  op?: string;
};

/**
 * Apply a chain of epilogue ops to a reduced value in-register.
 * Binary ops load external inputs from ep_inN bindings at the output index.
 */
function applyEpilogueChain(
  ctx: KernelContext,
  value: BlockExpr,
  epilogueOps: ReductionEpilogueOpDesc[],
  outIdx: BlockExpr,
): BlockExpr {
  let result = value;
  for (const eop of epilogueOps) {
    if (eop.kind === "cast") {
      if (eop.toDtype === "f16") result = result.toF16();
      else if (eop.toDtype === "i32") result = result.toI32();
      else if (eop.toDtype === "u32") result = result.toU32();
      else result = result.toF32();
    } else if (eop.kind === "binary" && eop.op && eop.inputIndex !== undefined) {
      const epInput = ctx.load(`ep_in${eop.inputIndex}`, outIdx);
      result = applyFusedOp(ctx, eop.op, [result, epInput]);
    } else if (eop.kind === "unary" && eop.op) {
      result = applyFusedOp(ctx, eop.op, [result]);
    } else if (eop.kind === "gelu") {
      result = applyFusedOp(ctx, "gelu_tanh", [result]);
    }
  }
  return result;
}

/**
 * Build epilogue bindings for external inputs.
 */
function buildEpilogueBindings(
  epilogueOps: ReductionEpilogueOpDesc[],
): Record<string, BindingSpec> {
  const bindings: Record<string, BindingSpec> = {};
  for (const eop of epilogueOps) {
    if (eop.kind === "binary" && eop.inputIndex !== undefined) {
      bindings[`ep_in${eop.inputIndex}`] = { storage: "read", type: "f32" as DataType };
    }
  }
  return bindings;
}

/**
 * Determine the output DataType for bindings from the final DType.
 */
function outBindingType(dtype: DType): DataType {
  if (dtype === "f16") return "f16";
  if (dtype === "i32") return "i32";
  if (dtype === "u32") return "u32";
  return "f32";
}

// ============================================================================
// Sum Dim With Epilogue
// ============================================================================

/**
 * Factory: Dimension-wise sum with fused epilogue chain.
 * After reducing, applies epilogue ops (cast, add, mul, activations) in-register.
 */
export function makeSumDimWithEpilogueSpec(
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
  epilogueOps: ReductionEpilogueOpDesc[],
  outputDtype: DType,
): TileKernelSpec {
  const variant = parallel ? "par" : "seq";
  const name = `sumDimEp_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  const bindings: Record<string, BindingSpec> = {
    input: { storage: "read", type: "f32" },
    ...buildEpilogueBindings(epilogueOps),
    out: { storage: "read_write", type: outBindingType(outputDtype) },
  };

  if (parallel) {
    return {
      name,
      workgroupSize: WG,
      bindings,
      uniforms: { outSize: "u32", reductionSize: "u32" },
      grid: (u) => {
        const n = u.outSize;
        if (n <= 65535) return [n];
        return [Math.min(n, 65535), Math.ceil(n / 65535)];
      },
      kernel(ctx) {
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const outIdx = wid;
        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("sum", tid, reductionSize, WG, (r) => {
          const off = buildInputOffset(ctx, outIdx, r,
            inputShape, inputStrides, outShape, outStrides,
            normalizedDims, inputToOutDim, "sep");
          return ctx.load("input", off);
        });

        const final = applyEpilogueChain(ctx, result, epilogueOps, outIdx);
        ctx.guardedStore("out", tid.eq(ctx.u32(0)), outIdx, final);
      },
    };
  }

  return {
    name,
    workgroupSize: WG,
    bindings,
    uniforms: { outSize: "u32", reductionSize: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      const reductionSize = ctx.uniform("reductionSize");
      const acc = ctx.emitVar("total", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), reductionSize, (r) => {
        const off = buildInputOffset(ctx, idx, r,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "ses");
        acc.addAssign(ctx.load("input", off));
      });

      const final = applyEpilogueChain(ctx, acc.get(), epilogueOps, idx);
      ctx.emitStore("out", idx, final);
    },
  };
}

// ============================================================================
// Sum Full With Epilogue
// ============================================================================

/**
 * Factory: Full reduction sum with fused epilogue chain.
 */
export function makeSumFullWithEpilogueSpec(
  epilogueOps: ReductionEpilogueOpDesc[],
  outputDtype: DType,
): TileKernelSpec {
  const bindings: Record<string, BindingSpec> = {
    input: { storage: "read", type: "f32" },
    ...buildEpilogueBindings(epilogueOps),
    out: { storage: "read_write", type: outBindingType(outputDtype) },
  };

  return {
    name: "sumFullEp",
    workgroupSize: WG,
    bindings,
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("sum", tid, ctx.uniform("size"), WG, (i) =>
        ctx.load("input", i),
      );
      const final = applyEpilogueChain(ctx, result, epilogueOps, ctx.u32(0));
      ctx.guardedStore("out", tid.eq(ctx.u32(0)), ctx.u32(0), final);
    },
  };
}

// ============================================================================
// Max Dim With Epilogue
// ============================================================================

/**
 * Factory: Dimension-wise max with fused epilogue chain.
 */
export function makeMaxDimWithEpilogueSpec(
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
  epilogueOps: ReductionEpilogueOpDesc[],
  outputDtype: DType,
): TileKernelSpec {
  const variant = parallel ? "par" : "seq";
  const name = `maxDimEp_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  const bindings: Record<string, BindingSpec> = {
    input: { storage: "read", type: "f32" },
    ...buildEpilogueBindings(epilogueOps),
    out: { storage: "read_write", type: outBindingType(outputDtype) },
  };

  if (parallel) {
    return {
      name,
      workgroupSize: WG,
      bindings,
      uniforms: { outSize: "u32", reductionSize: "u32" },
      grid: (u) => {
        const n = u.outSize;
        if (n <= 65535) return [n];
        return [Math.min(n, 65535), Math.ceil(n / 65535)];
      },
      kernel(ctx) {
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const outIdx = wid;
        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("max", tid, reductionSize, WG, (r) => {
          const off = buildInputOffset(ctx, outIdx, r,
            inputShape, inputStrides, outShape, outStrides,
            normalizedDims, inputToOutDim, "mep");
          return ctx.load("input", off);
        });

        const final = applyEpilogueChain(ctx, result, epilogueOps, outIdx);
        ctx.guardedStore("out", tid.eq(ctx.u32(0)), outIdx, final);
      },
    };
  }

  return {
    name,
    workgroupSize: WG,
    bindings,
    uniforms: { outSize: "u32", reductionSize: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      const firstOff = buildInputOffset(ctx, idx, ctx.u32(0),
        inputShape, inputStrides, outShape, outStrides,
        normalizedDims, inputToOutDim, "mes0");
      const maxVal = ctx.emitVar("maxVal", "f32", ctx.load("input", firstOff));

      ctx.forRange(ctx.u32(1), ctx.uniform("reductionSize"), (r) => {
        const off = buildInputOffset(ctx, idx, r,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "mes");
        maxVal.set(maxVal.get().max(ctx.load("input", off)));
      });

      const final = applyEpilogueChain(ctx, maxVal.get(), epilogueOps, idx);
      ctx.emitStore("out", idx, final);
    },
  };
}

// ============================================================================
// Max Full With Epilogue
// ============================================================================

/**
 * Factory: Full reduction max with fused epilogue chain.
 */
export function makeMaxFullWithEpilogueSpec(
  epilogueOps: ReductionEpilogueOpDesc[],
  outputDtype: DType,
): TileKernelSpec {
  const bindings: Record<string, BindingSpec> = {
    input: { storage: "read", type: "f32" },
    ...buildEpilogueBindings(epilogueOps),
    out: { storage: "read_write", type: outBindingType(outputDtype) },
  };

  return {
    name: "maxFullEp",
    workgroupSize: WG,
    bindings,
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("max", tid, ctx.uniform("size"), WG, (i) =>
        ctx.load("input", i),
      );
      const final = applyEpilogueChain(ctx, result, epilogueOps, ctx.u32(0));
      ctx.guardedStore("out", tid.eq(ctx.u32(0)), ctx.u32(0), final);
    },
  };
}

// ============================================================================
// Sum Dim With Preamble Chain + Epilogue (cross-reduction fusion)
// ============================================================================

/**
 * Factory: Dimension-wise sum with both preamble chain and epilogue chain.
 * Preamble ops are applied in the accumulation loop body,
 * epilogue ops are applied to the reduced result before storing.
 */
export function makeSumDimWithPreambleEpilogueSpec(
  chainOps: PreambleChainKernelOp[],
  totalPreambleInputs: number,
  preambleInputDtypes: DType[],
  epilogueOps: ReductionEpilogueOpDesc[],
  outputDtype: DType,
  inputShape: number[],
  inputStrides: number[],
  normalizedDims: number[],
  outShape: number[],
  outStrides: number[],
  inputToOutDim: number[],
  parallel: boolean,
): TileKernelSpec {
  const preNames = chainOps.map(o => o.op).join("+");
  const variant = parallel ? "par" : "seq";
  const name = `sumPCE_${preNames}_${variant}_${inputShape.join("x")}_d${normalizedDims.join(",")}`;

  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < totalPreambleInputs; i++) {
    bindings[`in${i}`] = { storage: "read", type: (preambleInputDtypes[i] || "f32") as DataType };
  }
  Object.assign(bindings, buildEpilogueBindings(epilogueOps));
  bindings.out = { storage: "read_write", type: outBindingType(outputDtype) };

  const needsF16 = preambleInputDtypes.some(d => d === "f16") || outputDtype === "f16";

  if (parallel) {
    return {
      name,
      workgroupSize: WG,
      bindings,
      enableF16: needsF16,
      uniforms: { outSize: "u32", reductionSize: "u32" },
      grid: (u) => {
        const n = u.outSize;
        if (n <= 65535) return [n];
        return [Math.min(n, 65535), Math.ceil(n / 65535)];
      },
      kernel(ctx) {
        const tid = ctx.localIndex();
        const wid = ctx.programId(0);
        const outIdx = wid;
        ctx.ifThen(outIdx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

        const reductionSize = ctx.uniform("reductionSize");
        const result = ctx.wgReduce("sum", tid, reductionSize, WG, (r) => {
          const off = buildInputOffset(ctx, outIdx, r,
            inputShape, inputStrides, outShape, outStrides,
            normalizedDims, inputToOutDim, "spce");
          return applyPreambleChain(ctx, chainOps, off);
        });

        const final = applyEpilogueChain(ctx, result, epilogueOps, outIdx);
        ctx.guardedStore("out", tid.eq(ctx.u32(0)), outIdx, final);
      },
    };
  }

  // Sequential
  return {
    name,
    workgroupSize: WG,
    bindings,
    enableF16: needsF16,
    uniforms: { outSize: "u32", reductionSize: "u32" },
    grid: elementwiseGrid(WG, { elementUniform: "outSize" }),
    kernel(ctx) {
      const idx = ctx.globalId(0);
      ctx.ifThen(idx.ge(ctx.uniform("outSize")), () => ctx.emitReturn());

      const acc = ctx.emitVar("total", "f32", ctx.f32(0));
      ctx.forRange(ctx.u32(0), ctx.uniform("reductionSize"), (reduceIdx) => {
        const off = buildInputOffset(ctx, idx, reduceIdx,
          inputShape, inputStrides, outShape, outStrides,
          normalizedDims, inputToOutDim, "spce");
        acc.addAssign(applyPreambleChain(ctx, chainOps, off));
      });

      const final = applyEpilogueChain(ctx, acc.get(), epilogueOps, idx);
      ctx.emitStore("out", idx, final);
    },
  };
}

// ============================================================================
// Sum Full With Preamble Chain + Epilogue
// ============================================================================

/**
 * Factory: Full reduction sum with both preamble chain and epilogue chain.
 */
export function makeSumFullWithPreambleEpilogueSpec(
  chainOps: PreambleChainKernelOp[],
  totalPreambleInputs: number,
  preambleInputDtypes: DType[],
  epilogueOps: ReductionEpilogueOpDesc[],
  outputDtype: DType,
): TileKernelSpec {
  const preNames = chainOps.map(o => o.op).join("+");
  const name = `sumFullPCE_${preNames}`;

  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < totalPreambleInputs; i++) {
    bindings[`in${i}`] = { storage: "read", type: (preambleInputDtypes[i] || "f32") as DataType };
  }
  Object.assign(bindings, buildEpilogueBindings(epilogueOps));
  bindings.out = { storage: "read_write", type: outBindingType(outputDtype) };

  const needsF16 = preambleInputDtypes.some(d => d === "f16") || outputDtype === "f16";

  return {
    name,
    workgroupSize: WG,
    bindings,
    enableF16: needsF16,
    uniforms: { size: "u32" },
    grid: singleWorkgroup(),
    kernel(ctx) {
      const tid = ctx.localIndex();
      const result = ctx.wgReduce("sum", tid, ctx.uniform("size"), WG, (i) =>
        applyPreambleChain(ctx, chainOps, i),
      );
      const final = applyEpilogueChain(ctx, result, epilogueOps, ctx.u32(0));
      ctx.guardedStore("out", tid.eq(ctx.u32(0)), ctx.u32(0), final);
    },
  };
}
