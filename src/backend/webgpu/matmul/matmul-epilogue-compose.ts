/**
 * Matmul epilogue composition: converts EpilogueConfig into post-accumulate
 * callbacks that the matmul kernel body can call without knowing about epilogues.
 *
 * This separates the graph-level optimization (fusing elementwise ops after matmul)
 * from the kernel authoring layer (pure tiled matmul).
 */

import type { BlockExpr, KernelContext } from "../tile-ir";
import type { EpilogueConfig } from "./codegen";
import type { DType } from "./types";
import { TileOps, BlockOps, Block, TileRange, type Accumulator } from "../tile-ops";

// ============================================================================
// Addressing types
// ============================================================================

/** Addressing info passed to BlockOps post-accumulate callbacks. */
export interface BlockOpsAddressing {
  offsN: TileRange;
  threadOutBase: BlockExpr;
  threadOutStride: BlockExpr;
}

/** Addressing info passed to TileOps post-accumulate callbacks. */
export interface TileOpsAddressing {
  batchOffC: BlockExpr;
  wgRow: BlockExpr;
  wgCol: BlockExpr;
  threadRow: BlockExpr;
  threadCol: BlockExpr;
  m: BlockExpr;
  n: BlockExpr;
  ldc: BlockExpr;
  offsM: TileRange;
  offsN: TileRange;
}

// ============================================================================
// Callback types
// ============================================================================

/** BlockOps post-accumulate: transforms acc in-place (caller stores). */
export type BlockOpsPostAccFn = (
  ctx: KernelContext,
  ops: BlockOps,
  acc: Block,
  addressing: BlockOpsAddressing,
) => void;

/** TileOps post-accumulate result. */
export type TileOpsPostAcc = {
  fn: (ctx: KernelContext, ops: TileOps, acc: Accumulator, addressing: TileOpsAddressing) => void;
  /** If true, the callback handled the store — caller should NOT store. */
  handlesStore: boolean;
};

// ============================================================================
// Epilogue Expression Helpers
// ============================================================================

function applyGelu(ctx: KernelContext, x: BlockExpr): BlockExpr {
  const inner = ctx.const(0.7978845608).mul(x.add(ctx.const(0.044715).mul(x.mul(x).mul(x))));
  return x.mul(ctx.const(0.5)).mul(ctx.const(1.0).add(inner.tanh()));
}

function applyGeluErf(ctx: KernelContext, x: BlockExpr): BlockExpr {
  const absX = x.abs();
  const t = ctx.const(1.0).div(
    ctx.const(1.0).add(ctx.const(0.3275911).mul(absX.mul(ctx.const(0.7071067811865476))))
  );
  const poly = ctx.const(1.061405429).mul(t)
    .add(ctx.const(-1.453152027)).mul(t)
    .add(ctx.const(1.421413741)).mul(t)
    .add(ctx.const(-0.284496736)).mul(t)
    .add(ctx.const(0.254829592)).mul(t);
  const erfApprox = ctx.const(1.0).sub(poly.mul(x.neg().mul(x).mul(ctx.const(0.5)).exp()));
  const signX = x.ge(ctx.const(0.0)).select(ctx.const(1.0), ctx.const(-1.0));
  return x.mul(ctx.const(0.5)).mul(ctx.const(1.0).add(signX.mul(erfApprox)));
}

function applySilu(ctx: KernelContext, x: BlockExpr): BlockExpr {
  return x.div(ctx.const(1.0).add(x.neg().exp()));
}

function applySigmoid(ctx: KernelContext, x: BlockExpr): BlockExpr {
  return ctx.const(1.0).div(ctx.const(1.0).add(x.neg().exp()));
}

/** Apply a unary epilogue op to a BlockExpr value. */
function applyUnaryOp(ctx: KernelContext, opName: string, x: BlockExpr): BlockExpr {
  switch (opName) {
    case "relu": return x.gt(ctx.const(0.0)).select(x, ctx.const(0.0));
    case "gelu": case "gelu_tanh": return applyGelu(ctx, x);
    case "gelu_erf": return applyGeluErf(ctx, x);
    case "silu": return applySilu(ctx, x);
    case "sigmoid": return applySigmoid(ctx, x);
    case "tanh": return x.tanh();
    case "neg": return x.neg();
    case "abs": return x.abs();
    case "exp": return x.exp();
    case "log": return x.log();
    case "sqrt": return x.sqrt();
    default: throw new Error(`Unsupported unary epilogue op: ${opName}`);
  }
}

// ============================================================================
// Per-element epilogue application (used by TileOps binary path)
// ============================================================================

/** Apply an epilogue op to a result expression (used for binary epilogue fallback). */
function applyEpilogueOp(
  ctx: KernelContext,
  op: EpilogueConfig["ops"][number],
  result: BlockExpr,
  outIdx: BlockExpr,
  outCol: BlockExpr,
): BlockExpr {
  switch (op.kind) {
    case "none": return result;
    case "bias": return result.add(ctx.load(`epilogue_in${op.inputIndex}`, outCol));
    case "unary": return applyUnaryOp(ctx, op.op, result);
    case "binary": {
      const rhs = ctx.load(`epilogue_in${op.inputIndex}`, outIdx);
      switch (op.op) {
        case "add": return result.add(rhs);
        case "sub": return result.sub(rhs);
        case "mul": return result.mul(rhs);
        case "div": return result.div(rhs);
        default: throw new Error(`Unsupported binary epilogue op: ${op.op}`);
      }
    }
    case "cast": return op.toDtype === "f16" ? result.toF16() : result.toF32();
    case "relu": return result.gt(ctx.const(0.0)).select(result, ctx.const(0.0));
    case "gelu": return applyGelu(ctx, result);
    case "silu": return applySilu(ctx, result);
    case "add": return result.add(ctx.load(`epilogue_in${op.inputIndex}`, outIdx));
    case "mul": return result.mul(ctx.load(`epilogue_in${op.inputIndex}`, outIdx));
  }
}

/** Check if epilogue has any binary ops that need the full output index. */
function hasBinaryEpilogue(epilogue: EpilogueConfig): boolean {
  return epilogue.ops.some(o => o.kind === "binary" || o.kind === "add" || o.kind === "mul");
}

// ============================================================================
// TileOps accumulator epilogue (non-binary path)
// ============================================================================

/** Apply a non-binary epilogue op to the accumulator. */
function applyEpilogueToAcc(
  ctx: KernelContext,
  ops: TileOps,
  acc: Accumulator,
  op: EpilogueConfig["ops"][number],
  offsN: TileRange,
): void {
  switch (op.kind) {
    case "none": break;
    case "bias":
      acc.add_(ops.load1d(`epilogue_in${op.inputIndex}`, offsN));
      break;
    case "unary":
      acc.apply_((x) => applyUnaryOp(ctx, op.op, x));
      break;
    case "cast":
      if (op.toDtype === "f16") acc.castTo_("f16");
      // f32 cast is a no-op on the f32 accumulator
      break;
    case "relu":
      acc.apply_((x) => x.gt(ctx.const(0.0)).select(x, ctx.const(0.0)));
      break;
    case "gelu":
      acc.apply_((x) => applyGelu(ctx, x));
      break;
    case "silu":
      acc.apply_((x) => applySilu(ctx, x));
      break;
    // binary/add/mul should not reach here — caller checks hasBinaryEpilogue
  }
}

// ============================================================================
// BlockOps accumulator epilogue
// ============================================================================

/**
 * Apply composable epilogue ops to a Block accumulator using only Block-level operations.
 *
 * Binary epilogue (residual add) collapses to the same path as simple epilogue:
 * load the binary operand as a register Block via ops.load() with thread ptr,
 * then acc.add_(binaryBlock). WebGPU clamped reads (OOB → 0) make this safe;
 * the storeTile mask prevents writing OOB elements.
 */
function applyBlockEpilogue(
  ctx: KernelContext,
  ops: BlockOps,
  acc: Block,
  epilogue: EpilogueConfig,
  addressing: BlockOpsAddressing,
): void {
  const { offsN, threadOutBase, threadOutStride } = addressing;
  for (const op of epilogue.ops) {
    switch (op.kind) {
      case "none": break;
      case "bias":
        acc.add_(ops.load1d(`epilogue_in${op.inputIndex}`, offsN));
        break;
      case "unary":
        acc.apply_((x) => applyUnaryOp(ctx, op.op, x));
        break;
      case "binary": {
        // Load binary operand at same indices the store will write to
        const binBlock = ops.load(`epilogue_in${op.inputIndex}`,
          { kind: "thread", base: threadOutBase, stride: threadOutStride },
          { rows: acc.rows, cols: acc.cols },
        );
        switch (op.op) {
          case "add": acc.add_(binBlock); break;
          case "sub": acc.sub_(binBlock); break;
          case "mul": acc.mul_(binBlock); break;
          default: throw new Error(`Unsupported binary epilogue: ${op.op}`);
        }
        break;
      }
      case "cast":
        if (op.toDtype === "f16") acc.castTo_("f16");
        break;
      // Backward-compat kinds
      case "relu":
        acc.apply_((x) => x.gt(ctx.const(0.0)).select(x, ctx.const(0.0)));
        break;
      case "gelu":
        acc.apply_((x) => applyGelu(ctx, x));
        break;
      case "silu":
        acc.apply_((x) => applySilu(ctx, x));
        break;
      case "add": {
        const b = ops.load(`epilogue_in${op.inputIndex}`,
          { kind: "thread", base: threadOutBase, stride: threadOutStride },
          { rows: acc.rows, cols: acc.cols });
        acc.add_(b);
        break;
      }
      case "mul": {
        const b = ops.load(`epilogue_in${op.inputIndex}`,
          { kind: "thread", base: threadOutBase, stride: threadOutStride },
          { rows: acc.rows, cols: acc.cols });
        acc.mul_(b);
        break;
      }
    }
  }
  if (epilogue.outputDtype === "f16" && !epilogue.ops.some(o => o.kind === "cast")) {
    acc.castTo_("f16");
  }
}

// ============================================================================
// Composition Functions
// ============================================================================

/**
 * Build a BlockOps post-accumulate callback from EpilogueConfig.
 * The callback transforms the accumulator in-place; the caller handles the store.
 */
export function composeBlockOpsEpilogue(
  epilogue: EpilogueConfig,
  _outputDtype: DType,
): BlockOpsPostAccFn {
  return (ctx, ops, acc, addressing) => {
    applyBlockEpilogue(ctx, ops, acc, epilogue, addressing);
  };
}

/**
 * Build a TileOps post-accumulate callback from EpilogueConfig.
 *
 * Binary epilogue: callback does per-element store (handlesStore=true).
 * Non-binary: callback transforms acc (handlesStore=false, caller stores).
 */
export function composeTileOpsEpilogue(
  epilogue: EpilogueConfig,
  outputDtype: DType,
  threadTileM: number,
  threadTileN: number,
): TileOpsPostAcc {
  if (hasBinaryEpilogue(epilogue)) {
    return {
      handlesStore: true,
      fn: (ctx, _ops, acc, addressing) => {
        const { batchOffC, wgRow, wgCol, threadRow, threadCol, m, n, ldc } = addressing;
        ctx.forRange(ctx.const(0, "u32"), ctx.const(threadTileM, "u32"), (tm) => {
          ctx.forRange(ctx.const(0, "u32"), ctx.const(threadTileN, "u32"), (tn) => {
            const outRow = ctx.emitLet("out_row",
              wgRow.add(threadRow.mul(ctx.const(threadTileM, "u32")).add(tm)));
            const outCol = ctx.emitLet("out_col",
              wgCol.add(threadCol.mul(ctx.const(threadTileN, "u32")).add(tn)));
            const inBounds = outRow.lt(m).and(outCol.lt(n));
            ctx.ifThen(inBounds, () => {
              const outIdx = ctx.emitLet("out_idx", batchOffC.add(outRow.mul(ldc).add(outCol)));
              let result = ctx.emitLet("result", acc.get(tm, tn));
              let step = 0;
              for (const op of epilogue.ops) {
                result = ctx.emitLet(`result_e${step++}`, applyEpilogueOp(ctx, op, result, outIdx, outCol));
              }
              if (epilogue.outputDtype === "f16" && !epilogue.ops.some(o => o.kind === "cast")) {
                result = ctx.emitLet("result_final", result.toF16());
              }
              ctx.pushStatement({
                kind: "indexAssign",
                arrayName: "out",
                idx: outIdx.node,
                value: result.node,
              });
            });
          });
        });
      },
    };
  }

  // Non-binary epilogue: transform acc, caller stores
  return {
    handlesStore: false,
    fn: (ctx, ops, acc, addressing) => {
      for (const op of epilogue.ops) {
        applyEpilogueToAcc(ctx, ops, acc, op, addressing.offsN);
      }
      if (epilogue.outputDtype === "f16" && !epilogue.ops.some(o => o.kind === "cast")) {
        acc.castTo_("f16");
      }
    },
  };
}
