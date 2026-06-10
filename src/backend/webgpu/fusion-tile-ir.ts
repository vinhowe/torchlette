/**
 * Tile-IR Elementwise Fusion Codegen
 *
 * Tile-IR fusion kernel generator.
 * Converts FusedKernelRecipe → TileKernelSpec → compileTileKernel → WGSL.
 *
 * This puts elementwise fusion on the same tile-IR path as matmul kernels,
 * creating a unified kernel representation for future optimization passes.
 */

import { sizeOf } from "../../core/shape";
import type { DType } from "../types";
import {
  computeKernelMeta,
  type FusedKernelRecipe,
  type GeneratedKernel,
  type KernelGenOptions,
  needsBroadcast,
} from "./fusion-types";
import { compileTileKernel } from "./tile-compiler";
import {
  type BindingSpec,
  type BlockExpr,
  type DataType,
  elementwiseGrid,
  type KernelContext,
  type TileKernelSpec,
} from "./tile-ir";

// ============================================================================
// Op mapping: OP_REGISTRY op name → tile-IR BlockExpr operations
// ============================================================================

/**
 * Apply a fused op to tile-IR BlockExpr inputs, producing a new BlockExpr.
 */
export function applyFusedOp(
  ctx: KernelContext,
  op: string,
  inputs: BlockExpr[],
): BlockExpr {
  const a = inputs[0];

  switch (op) {
    // -- Direct unary (1:1 with tile-IR) --
    case "neg":
      return a.neg();
    case "abs":
      return a.abs();
    case "exp":
      return a.exp();
    case "log":
      return a.log();
    case "sqrt":
      return a.sqrt();
    case "rsqrt":
      return a.rsqrt();
    case "tanh":
      return a.tanh();
    case "floor":
      return a.floor();
    case "ceil":
      return a.ceil();
    case "sin":
      return a.sin();
    case "cos":
      return a.cos();
    case "round":
      return a.round();
    case "sign":
      return a.sign();

    // -- Direct binary (1:1 with tile-IR) --
    case "add":
      return a.add(inputs[1]);
    case "sub":
      return a.sub(inputs[1]);
    case "mul":
      return a.mul(inputs[1]);
    case "div":
      return a.div(inputs[1]);
    case "pow":
      return a.pow(inputs[1]);
    case "min":
      return a.min(inputs[1]);
    case "max":
      return a.max(inputs[1]);
    case "mod":
      return a.mod(inputs[1]);

    // -- Composite activations (built from primitives) --
    case "relu":
      return a.gt(ctx.f32(0)).select(a, ctx.f32(0));

    case "sigmoid":
      // 1 / (1 + exp(-x))
      return ctx.f32(1).div(ctx.f32(1).add(a.neg().exp()));

    case "silu":
      // x / (1 + exp(-x))
      return a.div(ctx.f32(1).add(a.neg().exp()));

    case "softplus":
      // log(1 + exp(x))
      return ctx.f32(1).add(a.exp()).log();

    case "gelu":
    case "gelu_tanh": {
      // x * 0.5 * (1 + tanh(clamp(0.7978845608 * (x + 0.044715 * x^3), -10, 10)))
      const x3 = a.mul(a).mul(a);
      const inner = ctx.f32(0.7978845608).mul(a.add(ctx.f32(0.044715).mul(x3)));
      // clamp via min/max
      const clamped = inner.max(ctx.f32(-10)).min(ctx.f32(10));
      return a.mul(ctx.f32(0.5)).mul(ctx.f32(1).add(clamped.tanh()));
    }

    case "gelu_erf": {
      // x * 0.5 * (1 + erf(x / sqrt(2)))
      // Abramowitz & Stegun polynomial approximation for erf
      const ax = a.abs().mul(ctx.f32(Math.SQRT1_2)); // |x| / sqrt(2)
      const t = ctx.f32(1).div(ctx.f32(1).add(ctx.f32(0.3275911).mul(ax)));
      const poly = ctx
        .f32(1.061405429)
        .mul(t)
        .add(ctx.f32(-1.453152027))
        .mul(t)
        .add(ctx.f32(1.421413741))
        .mul(t)
        .add(ctx.f32(-0.284496736))
        .mul(t)
        .add(ctx.f32(0.254829592))
        .mul(t);
      const erfAbs = ctx.f32(1).sub(poly.mul(ax.neg().mul(ax).exp()));
      const erf = a.sign().mul(erfAbs);
      return a.mul(ctx.f32(0.5)).mul(ctx.f32(1).add(erf));
    }

    // -- Comparisons (return f32 0.0/1.0) --
    case "eq":
      return a.eq(inputs[1]).select(ctx.f32(1), ctx.f32(0));
    case "ne":
      return a.ne(inputs[1]).select(ctx.f32(1), ctx.f32(0));
    case "lt":
      return a.lt(inputs[1]).select(ctx.f32(1), ctx.f32(0));
    case "le":
      return a.le(inputs[1]).select(ctx.f32(1), ctx.f32(0));
    case "gt":
      return a.gt(inputs[1]).select(ctx.f32(1), ctx.f32(0));
    case "ge":
      return a.ge(inputs[1]).select(ctx.f32(1), ctx.f32(0));

    // -- Binary min/max --
    case "minimum":
      return a.min(inputs[1]);
    case "maximum":
      return a.max(inputs[1]);

    // -- Ternary --
    case "where":
      // where(cond, a, b) → select(b, a, cond > 0)
      return inputs[0].gt(ctx.f32(0)).select(inputs[1], inputs[2]);

    // -- Casts --
    case "cast_f32":
      return a.toF32();
    case "cast_f16":
      return a.toF16();
    case "cast_i32":
      return a.toI32();
    case "cast_u32":
      return a.toU32();

    // -- isfinite: (bits & 0x7F800000) != 0x7F800000 → 1.0, else 0.0 --
    case "isfinite": {
      const bits = a.bitcastTo("u32");
      const exponentMask = ctx.u32(0x7f800000);
      const masked = bits.and(exponentMask);
      return masked.ne(exponentMask).select(ctx.f32(1), ctx.f32(0));
    }

    default:
      throw new Error(`Unsupported fusion op for tile-IR: ${op}`);
  }
}

// ============================================================================
// Broadcasting
// ============================================================================

/**
 * Emit tile-IR statements to compute a broadcast index.
 * Decomposes a linear output index into multi-dim coords, maps to input coords,
 * and re-linearizes.
 */
function emitBroadcastIndex(
  ctx: KernelContext,
  outputShape: number[],
  inputShape: number[],
  outputIdx: BlockExpr,
  _inputName: string,
): BlockExpr {
  const inputSize = sizeOf(inputShape);
  if (inputSize === 1) {
    return ctx.u32(0);
  }

  // Same shape — no broadcast needed
  if (
    outputShape.length === inputShape.length &&
    outputShape.every((d, i) => d === inputShape[i])
  ) {
    return outputIdx;
  }

  const rank = outputShape.length;
  const inputRank = inputShape.length;
  const rankDiff = rank - inputRank;

  // Decompose output linear index into coordinates
  const outCoords = ctx.decomposeIndex(outputIdx, outputShape);

  // Map to input coordinates (broadcast dims → 0)
  const inCoords: BlockExpr[] = [];
  for (let i = 0; i < inputRank; i++) {
    const outIdx = i + rankDiff;
    if (inputShape[i] === 1) {
      inCoords.push(ctx.u32(0));
    } else {
      inCoords.push(outCoords[outIdx]);
    }
  }

  // Re-linearize: compute input strides from shape
  const inStrides: number[] = [];
  for (let i = 0; i < inputRank; i++) {
    let s = 1;
    for (let j = i + 1; j < inputRank; j++) s *= inputShape[j];
    inStrides.push(s);
  }
  return ctx.linearizeIndex(inCoords, inStrides);
}

// ============================================================================
// Main entry point
// ============================================================================

/**
 * Generate a fused elementwise kernel using tile-IR.
 * Drop-in replacement for generateFusedKernel().
 */
export function generateFusedKernelTileIR(
  recipe: FusedKernelRecipe,
  options: KernelGenOptions = {},
): GeneratedKernel {
  // Reuse computeKernelMeta for cache key, vector width, grid size
  const meta = computeKernelMeta(recipe, options);
  const { vectorWidth, workItems, workgroupSize, gridSizeX } = meta;

  // Buffer donation: the donated input shares out0's read_write binding —
  // no separate read binding (WebGPU forbids one buffer bound read + rw in a
  // single dispatch). Its loads read `out0` instead.
  const donated = options.donatedInput;
  if (donated !== undefined) {
    const din = recipe.inputs[donated];
    if (!din || din.isInlinedConstant || din.isScalar) {
      throw new Error(`donatedInput ${donated} is not a bindable tensor input`);
    }
    if (sizeOf(din.shape) !== sizeOf(recipe.outputs[0].shape)) {
      throw new Error(`donatedInput ${donated} shape != output shape`);
    }
  }

  // Build physical binding map (skip inlined constants and the donated input)
  const physicalBinding: (number | null)[] = [];
  let nextBinding = 0;
  for (let i = 0; i < recipe.inputs.length; i++) {
    if (recipe.inputs[i].isInlinedConstant || i === donated) {
      physicalBinding.push(null);
    } else {
      physicalBinding.push(nextBinding++);
    }
  }
  const inputBindingCount = nextBinding;

  // Build TileKernelSpec bindings
  const bindings: Record<string, BindingSpec> = {};
  for (let i = 0; i < recipe.inputs.length; i++) {
    if (physicalBinding[i] === null) continue;
    const input = recipe.inputs[i];
    bindings[`in${i}`] = { storage: "read", type: dtypeToTileIR(input.dtype) };
  }
  for (let i = 0; i < recipe.outputs.length; i++) {
    const output = recipe.outputs[i];
    bindings[`out${i}`] = {
      storage: "read_write",
      type: dtypeToTileIR(output.dtype),
    };
  }

  // Uniform binding index: after all storage bindings
  const uniformBindingIndex = inputBindingCount + recipe.outputs.length;

  // Check if f16 is needed
  const needsF16 =
    recipe.inputs.some((inp) => inp.dtype === "f16") ||
    recipe.outputs.some((o) => o.dtype === "f16") ||
    recipe.nodes.some((n) => n.dtype === "f16");

  // Output shape for broadcast index computation (all outputs share the same shape)
  const outputShape = recipe.outputs[0].shape;

  const spec: TileKernelSpec = {
    name: `fused_${recipe.id}`,
    workgroupSize,
    bindings,
    uniforms: { total_elements: "u32" },
    uniformBindingIndex,
    enableF16: needsF16 || undefined,
    vectorize: vectorWidth > 1 ? vectorWidth : undefined,
    grid: elementwiseGrid(workgroupSize, { vecWidth: vectorWidth }),
    kernel: (ctx: KernelContext) => {
      // flatGlobalId handles both 1D and 2D grids, and works correctly
      // with the compiler's true vec4 vectorization path.
      const idx = ctx.elementIndex(workgroupSize, "total_elements");

      // Load inputs
      const inputExprs = new Map<number, BlockExpr>();
      for (let i = 0; i < recipe.inputs.length; i++) {
        const input = recipe.inputs[i];

        if (input.isInlinedConstant && input.inlinedValue !== undefined) {
          // Inlined constant: emit as tile-IR constant
          inputExprs.set(
            i,
            ctx[dtypeToTileIR(input.dtype)](input.inlinedValue),
          );
          continue;
        }

        const inputSize = sizeOf(input.shape);
        const isScalar = inputSize === 1;

        if (i === donated) {
          // Donated input: read through out0's read_write binding (same
          // element index — caller guarantees same shape, no broadcast).
          const val = ctx.emitLet(`v${i}`, ctx.load(`out0`, idx));
          inputExprs.set(i, val);
        } else if (isScalar) {
          // Scalar input: load index 0
          const val = ctx.emitLet(`v${i}`, ctx.load(`in${i}`, ctx.u32(0)));
          inputExprs.set(i, val);
        } else if (needsBroadcast(outputShape, input.shape)) {
          // Broadcast: compute mapped index
          const broadIdx = emitBroadcastIndex(
            ctx,
            outputShape,
            input.shape,
            idx,
            `in${i}`,
          );
          const val = ctx.emitLet(`v${i}`, ctx.load(`in${i}`, broadIdx));
          inputExprs.set(i, val);
        } else {
          // Same shape: direct indexed load
          const val = ctx.emitLet(`v${i}`, ctx.load(`in${i}`, idx));
          inputExprs.set(i, val);
        }
      }

      // Process nodes in topological order
      const nodeExprs = new Map<number, BlockExpr>();
      for (const node of recipe.nodes) {
        // Resolve inputs for this node
        const nodeInputs: BlockExpr[] = [];
        for (const inputId of node.inputs) {
          if (inputId < 0) {
            // External input
            const inputIdx = -inputId - 1;
            const expr = inputExprs.get(inputIdx);
            if (!expr)
              throw new Error(
                `Missing input expr for external input ${inputIdx}`,
              );
            nodeInputs.push(expr);
          } else {
            // Internal node reference
            const expr = nodeExprs.get(inputId);
            if (!expr) throw new Error(`Missing node expr for node ${inputId}`);
            nodeInputs.push(expr);
          }
        }

        // Apply op
        const result = applyFusedOp(ctx, node.op, nodeInputs);
        // Always emit a let for clarity (tile-compiler will inline simple exprs)
        const named = ctx.emitLet(`t${node.id}`, result);
        nodeExprs.set(node.id, named);
      }

      // Store outputs (masked store — like tl.store with mask)
      const totalElements = ctx.uniform("total_elements");
      for (let i = 0; i < recipe.outputs.length; i++) {
        const output = recipe.outputs[i];
        const val = nodeExprs.get(output.nodeId);
        if (!val)
          throw new Error(`Missing output expr for node ${output.nodeId}`);
        ctx.blockStore(`out${i}`, idx, totalElements, val);
      }
    },
  };

  const source = compileTileKernel(spec);

  return {
    source,
    workgroupSize,
    inputBindings: inputBindingCount,
    cacheKey: meta.cacheKey,
    vectorWidth,
    workItems,
    gridSizeX,
  };
}

// ============================================================================
// Helpers
// ============================================================================

export function dtypeToTileIR(dtype: DType): DataType {
  return dtype === "bool" ? "f32" : (dtype as DataType);
}
