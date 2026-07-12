/**
 * Tile IR Matmul: the SPLIT-K REDUCTION pass (family 3b).
 *
 * The tiled matmul kernel body (loop nest / cooperative-load staging / dot /
 * epilogue) was ABSORBED into the schedule object — it now lowers FROM the
 * ScheduleState in `src/schedule/matmul-skeleton.ts` `lowerTiledMatmul`
 * (P2 wave A / Commit B; the block-op body `matmulKernelBlockOps` and
 * `createTiledMatmulKernel` were retired, byte-differential-gated). This module
 * keeps only the split-K reduction pass, which the skeleton realizes via
 * `realizeKSplitReductionWgsl`.
 */

import { ceilDivGrid, type DataType, type TileKernelSpec } from "../tile-ir";
import type { DType } from "./types";

// ============================================================================
// K-Split Reduction Kernel (imperative mode)
// ============================================================================

function createKSplitReductionKernel(
  kSplitCount: number,
  outputDtype: DType,
): TileKernelSpec {
  const needsF16 = outputDtype === "f16";
  return {
    name: "kSplitReduce",
    workgroupSize: 256,
    enableF16: needsF16,
    bindings: {
      partials: { storage: "read", type: "f32" },
      out: { storage: "read_write", type: outputDtype as DataType },
    },
    uniforms: { totalElements: "u32", alpha: "f32" },
    grid: ceilDivGrid(256, "totalElements"),
    kernel(ctx) {
      const idx = ctx.emitLet("idx", ctx.globalId(0));
      const totalElements = ctx.uniform("totalElements");
      ctx.ifThen(idx.ge(totalElements), () => {
        ctx.emitReturn();
      });

      const alpha = ctx.uniform("alpha").bitcastTo("f32");
      const sumVar = ctx.emitVar("sum", "f32", ctx.load("partials", idx));
      for (let p = 1; p < kSplitCount; p++) {
        sumVar.addAssign(
          ctx.load("partials", idx.add(totalElements.mul(ctx.u32(p)))),
        );
      }
      const result = sumVar.get().mul(alpha);
      if (outputDtype === "f16") {
        ctx.emitStore("out", idx, result.toF16());
      } else {
        ctx.emitStore("out", idx, result);
      }
    },
  };
}

// ============================================================================
// Integration: compile to WGSL string
// ============================================================================

import { compileTileKernel } from "../tile-compiler";

export function generateKSplitReductionShaderTileIR(
  kSplitCount: number,
  outputDtype: DType,
): string {
  return compileTileKernel(
    createKSplitReductionKernel(kSplitCount, outputDtype),
  );
}
