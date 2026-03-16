/**
 * Row-Program Codegen
 *
 * Translates a RowProgram IR into a perRowKernel TileKernelSpec.
 * Each ReducePhase → ctx.wgReduce(...), WritePhase → ctx.stridedFor(store).
 * Expression trees are compiled via applyFusedOp.
 */

import type { RowProgram, RPExpr } from "../../compiler/row-program-types";
import { isRPValue } from "../../compiler/row-program-types";
import { applyFusedOp, dtypeToTileIR } from "./fusion-tile-ir";
import { WORKGROUP_SIZE } from "./shape-utils";
import type { BlockExpr, KernelContext, TileKernelSpec } from "./tile-ir";
import { perRowKernel } from "./tile-ir";

const WG = WORKGROUP_SIZE; // 256

/**
 * Generate a perRowKernel TileKernelSpec from a RowProgram.
 */
export function rowProgramToSpec(program: RowProgram): TileKernelSpec {
  // Build bindings: in0, in1, ... for inputs; output for the write target
  const bindings: Record<
    string,
    { storage: "read" | "read_write"; type: string }
  > = {};
  for (let i = 0; i < program.inputs.length; i++) {
    bindings[`in${i}`] = {
      storage: "read",
      type: dtypeToTileIR(program.inputs[i].dtype),
    };
  }
  bindings["output"] = {
    storage: "read_write",
    type: dtypeToTileIR(program.output.dtype),
  };

  const needsF16 =
    program.inputs.some((inp) => inp.dtype === "f16") ||
    program.output.dtype === "f16";

  return perRowKernel({
    name: `rowProg_${program.cacheKey}`,
    bindings,
    enableF16: needsF16 || undefined,

    kernel(ctx, _row, tid, D, base) {
      const reduceResults: BlockExpr[] = [];

      for (const phase of program.phases) {
        if (phase.kind === "reduce") {
          const result = ctx.wgReduce(phase.reduceOp, tid, D, WG, (i) =>
            emitExpr(ctx, phase.bodyExpr, base.add(i), reduceResults),
          );
          let finalResult = result;
          if (phase.isMean) {
            finalResult = result.div(D.toF32());
          }
          const named = ctx.emitLet(`r${reduceResults.length}`, finalResult);
          reduceResults.push(named);
        } else {
          // WritePhase
          if (phase.scalarOutput) {
            // Scalar output: only thread 0 writes one value per row
            ctx.ifThen(tid.eq(ctx.u32(0)), () => {
              const val = emitExpr(
                ctx,
                phase.bodyExpr,
                base, // offset = row index (base = row * D, but output is [numRows])
                reduceResults,
              );
              ctx.emitStore("output", _row, val);
            });
          } else {
            ctx.stridedFor(tid, D, WG, (i) => {
              const val = emitExpr(
                ctx,
                phase.bodyExpr,
                base.add(i),
                reduceResults,
              );
              ctx.emitStore("output", base.add(i), val);
            });
          }
        }
      }
    },
  });
}

/**
 * Recursively emit a tile-IR expression from an RPExpr tree.
 *
 * @param ctx           - Tile-IR kernel context
 * @param expr          - The expression node to emit
 * @param elementOffset - base + i (for per-element buffer loads)
 * @param reduceResults - Array of prior reduction result BlockExprs
 */
function emitExpr(
  ctx: KernelContext,
  expr: RPExpr,
  elementOffset: BlockExpr,
  reduceResults: BlockExpr[],
): BlockExpr {
  if (isRPValue(expr)) {
    switch (expr.kind) {
      case "input":
        return ctx.load(`in${expr.bufferIndex}`, elementOffset);
      case "reduceResult":
        return reduceResults[expr.phaseIndex];
      case "const":
        return ctx.f32(expr.value);
    }
  }

  // Interior node: evaluate inputs recursively, apply op
  const inputExprs = expr.inputs.map((inp) =>
    emitExpr(ctx, inp, elementOffset, reduceResults),
  );
  return applyFusedOp(ctx, expr.op, inputExprs);
}
