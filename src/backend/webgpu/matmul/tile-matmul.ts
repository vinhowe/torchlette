/**
 * Tile IR Matmul: Triton-like 2D tiled matrix multiplication.
 *
 * The kernel author thinks in BLOCK_M × BLOCK_N tiles, never about threads.
 * The compiler handles thread mapping, shared memory, barriers, and cooperative loading.
 *
 * Also includes epilogue composition: converts EpilogueConfig into post-accumulate
 * callbacks that the matmul kernel body can call without knowing about epilogues.
 */

import { applyFusedOp } from "../fusion-tile-ir";
import {
  type BlockExpr,
  ceilDivGrid,
  type DataType,
  type KernelContext,
  singleWorkgroup,
  type TileKernelSpec,
} from "../tile-ir";
import type { Block, BlockOps, TileRange } from "../tile-ops";
import type { CodegenOptions, EpilogueConfig } from "./types";
import { type DType, getWorkgroupSize } from "./types";

// ============================================================================
// Epilogue Composition
// ============================================================================

/** Addressing info passed to post-accumulate callbacks. */
interface BlockOpsAddressing {
  offsN: TileRange;
  threadOutBase: BlockExpr;
  threadOutStride: BlockExpr;
}

/** BlockOps post-accumulate: transforms acc in-place (caller stores). */
type BlockOpsPostAccFn = (
  ctx: KernelContext,
  ops: BlockOps,
  acc: Block,
  addressing: BlockOpsAddressing,
) => void;

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
      case "none":
        break;
      case "bias":
        acc.add_(ops.load1d(`epilogue_in${op.inputIndex}`, offsN));
        break;
      case "unary":
        acc.apply_((x) => applyFusedOp(ctx, op.op!, [x]));
        break;
      case "binary": {
        const binBlock = ops.load(
          `epilogue_in${op.inputIndex}`,
          { kind: "thread", base: threadOutBase, stride: threadOutStride },
          { rows: acc.rows, cols: acc.cols },
        );
        switch (op.op) {
          case "add":
            acc.add_(binBlock);
            break;
          case "sub":
            acc.sub_(binBlock);
            break;
          case "mul":
            acc.mul_(binBlock);
            break;
          default:
            throw new Error(`Unsupported binary epilogue: ${op.op}`);
        }
        break;
      }
      case "cast":
        if (op.toDtype === "f16") acc.castTo_("f16");
        break;
    }
  }
  if (
    epilogue.outputDtype === "f16" &&
    !epilogue.ops.some((o) => o.kind === "cast")
  ) {
    acc.castTo_("f16");
  }
}

// ============================================================================
// Main Kernel Builder
// ============================================================================

/** Create a TileKernelSpec for tiled matrix multiplication. */
export function createTiledMatmulKernel(
  options: CodegenOptions,
): TileKernelSpec {
  const {
    config,
    transposeMode,
    dtype,
    dtypeB: dtypeBOpt,
    epilogue,
    batched,
    inputCastA,
    inputCastB,
    kSplit,
  } = options;
  const { tileM, tileN, tileK, threadTileM, threadTileN } = config;

  const wgSize = getWorkgroupSize(config);
  const wgSizeX = wgSize.x;
  const wgSizeY = wgSize.y;

  const dtypeB = dtypeBOpt ?? dtype;
  const wgslDtypeA: DataType = (inputCastA ? "f32" : dtype) as DataType;
  const wgslDtypeB: DataType = (inputCastB ? "f32" : dtypeB) as DataType;

  const outputDtype: DType = kSplit
    ? "f32"
    : (epilogue?.outputDtype ??
      (dtype === "f32" || dtypeB === "f32" ? "f32" : dtype));
  const needsF16 =
    wgslDtypeA === "f16" || wgslDtypeB === "f16" || outputDtype === "f16";

  // Build bindings
  const bindings: Record<
    string,
    { storage: "read" | "read_write"; type: DataType }
  > = {
    a: { storage: "read", type: wgslDtypeA },
    b: { storage: "read", type: wgslDtypeB },
    out: {
      storage: "read_write",
      type: (epilogue?.outputDtype ?? outputDtype) as DataType,
    },
  };
  if (epilogue) {
    for (let i = 0; i < epilogue.additionalInputCount; i++) {
      bindings[`epilogue_in${i}`] = { storage: "read", type: "f32" };
    }
  }

  // Transpose flags
  const transA = transposeMode === "TN" || transposeMode === "TT";
  const transB = transposeMode === "NT" || transposeMode === "TT";

  // Build post-accumulate callback from epilogue config.
  let postAcc: BlockOpsPostAccFn | undefined;
  if (epilogue && epilogue.ops.length > 0 && !kSplit) {
    postAcc = (ctx, ops, acc, addressing) =>
      applyBlockEpilogue(ctx, ops, acc, epilogue, addressing);
  }

  const params: MatmulKernelParams = {
    tileM,
    tileN,
    tileK,
    threadTileM,
    threadTileN,
    wgSizeX,
    wgSizeY,
    transA,
    transB,
    outputDtype,
  };

  return {
    name: "tiledMatmul",
    workgroupSize: [wgSizeX, wgSizeY],
    enableF16: needsF16,
    uniformBindingIndex: 3,
    bindings,
    uniforms: {
      m: "u32",
      n: "u32",
      k: "u32",
      lda: "u32",
      ldb: "u32",
      ldc: "u32",
      alpha: "f32",
      batchSize: "u32",
      batchStrideA: "u32",
      batchStrideB: "u32",
      batchStrideC: "u32",
    },
    grid: singleWorkgroup(), // Actual grid computed at dispatch time in dispatch.ts

    kernel: (ctx) =>
      matmulKernelBlockOps(ctx, { batched, kSplit }, params, postAcc),
  };
}

interface MatmulKernelParams {
  tileM: number;
  tileN: number;
  tileK: number;
  threadTileM: number;
  threadTileN: number;
  wgSizeX: number;
  wgSizeY: number;
  transA: boolean;
  transB: boolean;
  outputDtype: DType;
}

/** Matmul kernel body — pure matmul with optional post-accumulate callback. */
function matmulKernelBlockOps(
  ctx: KernelContext,
  opts: { batched?: boolean; kSplit?: number },
  p: MatmulKernelParams,
  postAccumulate?: BlockOpsPostAccFn,
): void {
  const { batched, kSplit } = opts;
  const {
    tileM,
    tileN,
    tileK,
    threadTileM,
    threadTileN,
    transA,
    transB,
    outputDtype,
  } = p;

  // Initialize tile context — set thread tile, emit thread position bindings
  const { threadRow, threadCol } = ctx.configureTiles({
    threadTileM,
    threadTileN,
  });

  // 1. Uniforms
  const m = ctx.emitLet("m", ctx.uniform("m"));
  const n = ctx.emitLet("n", ctx.uniform("n"));
  const k = ctx.emitLet("k", ctx.uniform("k"));
  const lda = ctx.emitLet("lda", ctx.uniform("lda"));
  const ldb = ctx.emitLet("ldb", ctx.uniform("ldb"));
  const ldc = ctx.emitLet("ldc", ctx.uniform("ldc"));
  const alpha = ctx.emitLet("alpha", ctx.uniform("alpha"));

  // 2. Batch / K-split offsets
  let batchOffA: BlockExpr;
  let batchOffB: BlockExpr;
  let batchOffC: BlockExpr;
  let splitIdx: BlockExpr | undefined;

  if (kSplit) {
    splitIdx = ctx.emitLet("split_idx", ctx.programId(2));
    batchOffA = ctx.emitLet("batch_offset_a", ctx.const(0, "u32"));
    batchOffB = ctx.emitLet("batch_offset_b", ctx.const(0, "u32"));
    batchOffC = ctx.const(0, "u32");
  } else if (batched) {
    const batchIdx = ctx.emitLet("batch_idx", ctx.programId(2));
    batchOffA = ctx.emitLet(
      "batch_offset_a",
      batchIdx.mul(ctx.uniform("batchStrideA")),
    );
    batchOffB = ctx.emitLet(
      "batch_offset_b",
      batchIdx.mul(ctx.uniform("batchStrideB")),
    );
    batchOffC = ctx.emitLet(
      "batch_offset_c",
      batchIdx.mul(ctx.uniform("batchStrideC")),
    );
  } else {
    const batchIdx = ctx.emitLet("batch_idx", ctx.const(0, "u32"));
    batchOffA = ctx.emitLet(
      "batch_offset_a",
      batchIdx.mul(ctx.uniform("batchStrideA")),
    );
    batchOffB = ctx.emitLet(
      "batch_offset_b",
      batchIdx.mul(ctx.uniform("batchStrideB")),
    );
    batchOffC = ctx.emitLet(
      "batch_offset_c",
      batchIdx.mul(ctx.uniform("batchStrideC")),
    );
  }

  // 3. Workgroup positions
  const wgRow = ctx.emitLet(
    "wg_row",
    ctx.programId(1).mul(ctx.const(tileM, "u32")),
  );
  const wgCol = ctx.emitLet(
    "wg_col",
    ctx.programId(0).mul(ctx.const(tileN, "u32")),
  );

  // 4. Block ranges — ≈ tl.arange(wgRow, wgRow + BLOCK_M)
  const offsM = ctx.arange(wgRow, tileM);
  const offsN = ctx.arange(wgCol, tileN);

  // 5. Transpose strides
  const cOne = ctx.const(1, "u32");
  const strideAm = transA ? cOne : lda;
  const strideAk = transA ? lda : cOne;
  const strideBk = transB ? cOne : ldb;
  const strideBn = transB ? ldb : cOne;

  // 6. Accumulator — ≈ tl.zeros([BLOCK_M, BLOCK_N])
  const acc = ctx.zeros(threadTileM, threadTileN);

  // 7. K-loop bounds
  let kStart: BlockExpr;
  let kEnd: BlockExpr;
  let numKTiles: BlockExpr;
  const cTK = ctx.const(tileK, "u32");

  if (kSplit) {
    const kPerSplit = ctx.emitLet(
      "k_per_split",
      k.add(ctx.const(kSplit - 1, "u32")).div(ctx.const(kSplit, "u32")),
    );
    kStart = ctx.emitLet("k_start", splitIdx!.mul(kPerSplit));
    kEnd = ctx.emitLet("k_end", kStart.add(kPerSplit).min(k));
    numKTiles = ctx.emitLet(
      "num_k_tiles",
      kEnd
        .sub(kStart)
        .add(ctx.const(tileK - 1, "u32"))
        .div(cTK),
    );
  } else {
    numKTiles = ctx.emitLet(
      "num_k_tiles",
      k.add(ctx.const(tileK - 1, "u32")).div(cTK),
    );
    kStart = ctx.emitLet("k_start", ctx.const(0, "u32"));
    kEnd = ctx.emitLet("k_end", k);
  }

  // 8. K-loop — the core matmul
  ctx.forRange(ctx.const(0, "u32"), numKTiles, (kTile) => {
    const kOffset = ctx.emitLet("k_offset", kStart.add(kTile.mul(cTK)));
    const offsK = ctx.arange(kOffset, tileK);

    // Cooperative tile loads — ≈ tl.load(a_ptr, mask=a_mask)
    const aPtr = ctx.tilePtr(
      batchOffA,
      offsM.outer(strideAm),
      offsK.inner(strideAk),
    );
    const aMask = ctx.tileMask(offsM.lt(m), offsK.lt(kEnd));
    const a = ctx.load2D("a", aPtr, aMask);

    const bPtr = ctx.tilePtr(
      batchOffB,
      offsK.outer(strideBk),
      offsN.inner(strideBn),
    );
    const bMask = ctx.tileMask(offsK.lt(kEnd), offsN.lt(n));
    const b = ctx.load2D("b", bPtr, bMask);

    // acc += a @ b — ≈ acc += tl.dot(a, b)
    ctx.dotAccum(a, b, acc);
  });

  // 9. Store — two paths: K-split (raw partials) vs normal (with optional post-accumulate)
  if (kSplit) {
    const splitBase = ctx.emitLet("split_base", splitIdx!.mul(m.mul(n)));
    const outPtr = ctx.tilePtr(splitBase, offsM.outer(ldc), offsN.inner(cOne));
    const outMask = ctx.tileMask(offsM.lt(m), offsN.lt(n));
    ctx.store2D("out", acc, outPtr, outMask);
  } else {
    acc.mul_(alpha.toF32());

    if (postAccumulate) {
      const threadOutBase = ctx.emitLet(
        "thread_out_base",
        batchOffC.add(
          wgRow
            .add(threadRow.mul(ctx.const(threadTileM, "u32")))
            .mul(ldc)
            .add(wgCol.add(threadCol.mul(ctx.const(threadTileN, "u32")))),
        ),
      );
      postAccumulate(ctx, ctx.tileOps, acc, {
        offsN,
        threadOutBase,
        threadOutStride: ldc,
      });
    } else if (outputDtype === "f16") {
      acc.castTo_("f16");
    }

    const outPtr = ctx.tilePtr(batchOffC, offsM.outer(ldc), offsN.inner(cOne));
    const outMask = ctx.tileMask(offsM.lt(m), offsN.lt(n));
    ctx.store2D("out", acc, outPtr, outMask);
  }
}

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

export function generateTiledMatmulShaderTileIR(
  options: CodegenOptions,
): string {
  return compileTileKernel(createTiledMatmulKernel(options));
}

export function generateKSplitReductionShaderTileIR(
  kSplitCount: number,
  outputDtype: DType,
): string {
  return compileTileKernel(
    createKSplitReductionKernel(kSplitCount, outputDtype),
  );
}
