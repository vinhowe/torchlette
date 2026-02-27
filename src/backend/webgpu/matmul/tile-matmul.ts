/**
 * Tile IR Matmul: Triton-like 2D tiled matrix multiplication.
 *
 * Uses the TileOps block-level API: the kernel author thinks in BLOCK_M × BLOCK_N
 * tiles, never about threads. The compiler handles thread mapping, shared memory,
 * barriers, and cooperative loading.
 *
 * The dispatch infrastructure is unchanged — we just swap the WGSL generation.
 */

import { type TileKernelSpec, type DataType, type BlockExpr, type KernelContext, ceilDivGrid } from "../tile-ir";
import type { CodegenOptions } from "./codegen";
import { getWorkgroupSize, type DType } from "./types";
import { TileOps, BlockOps, Block, TileRange, buildPtr, buildMask, type TileConfig } from "../tile-ops";
import {
  composeBlockOpsEpilogue,
  composeTileOpsEpilogue,
  type BlockOpsPostAccFn,
  type TileOpsPostAcc,
} from "./matmul-epilogue-compose";

// ============================================================================
// Main Kernel Builder
// ============================================================================

/**
 * Create a TileKernelSpec for tiled matrix multiplication.
 *
 * Two kernel body implementations:
 * - TileOps (default): Original block-level API with Tile/Accumulator
 * - BlockOps (TORCHLETTE_BLOCK_MATMUL=1): Unified Block API
 *
 * Both produce equivalent WGSL. The dispatch infrastructure is shared.
 */
export function createTiledMatmulKernel(options: CodegenOptions): TileKernelSpec {
  const { config, transposeMode, dtype, dtypeB: dtypeBOpt, epilogue, batched, inputCastA, inputCastB, kSplit } = options;
  const { tileM, tileN, tileK, threadTileM, threadTileN } = config;

  const wgSize = getWorkgroupSize(config);
  const wgSizeX = wgSize.x;
  const wgSizeY = wgSize.y;

  const dtypeB = dtypeBOpt ?? dtype;
  const wgslDtypeA: DataType = (inputCastA ? "f32" : dtype) as DataType;
  const wgslDtypeB: DataType = (inputCastB ? "f32" : dtypeB) as DataType;

  const outputDtype: DType = kSplit ? "f32" : (epilogue?.outputDtype ?? (dtype === "f32" || dtypeB === "f32" ? "f32" : dtype));
  const needsF16 = wgslDtypeA === "f16" || wgslDtypeB === "f16" || outputDtype === "f16";

  // Build bindings
  const bindings: Record<string, { storage: "read" | "read_write"; type: DataType }> = {
    a: { storage: "read", type: wgslDtypeA },
    b: { storage: "read", type: wgslDtypeB },
    out: { storage: "read_write", type: (epilogue?.outputDtype ?? outputDtype) as DataType },
  };
  if (epilogue) {
    for (let i = 0; i < epilogue.additionalInputCount; i++) {
      bindings[`epilogue_in${i}`] = { storage: "read", type: "f32" };
    }
  }

  // Transpose flags
  const transA = transposeMode === "TN" || transposeMode === "TT";
  const transB = transposeMode === "NT" || transposeMode === "TT";

  const useBlockOps = process.env.TORCHLETTE_BLOCK_MATMUL === "1";

  // Build post-accumulate callback from epilogue config.
  // This is the composition boundary: the kernel body receives a callback,
  // not an EpilogueConfig, keeping it epilogue-agnostic.
  let blockOpsPostAcc: BlockOpsPostAccFn | undefined;
  let tileOpsPostAcc: TileOpsPostAcc | undefined;
  if (epilogue && epilogue.ops.length > 0 && !kSplit) {
    if (useBlockOps) {
      blockOpsPostAcc = composeBlockOpsEpilogue(epilogue, outputDtype);
    } else {
      tileOpsPostAcc = composeTileOpsEpilogue(epilogue, outputDtype, threadTileM, threadTileN);
    }
  }

  const params: MatmulKernelParams = { tileM, tileN, tileK, threadTileM, threadTileN, wgSizeX, wgSizeY, transA, transB, outputDtype };

  return {
    name: "tiledMatmul",
    workgroupSize: [wgSizeX, wgSizeY],
    enableF16: needsF16,
    uniformBindingIndex: 3,
    bindings,
    uniforms: {
      m: "u32", n: "u32", k: "u32",
      lda: "u32", ldb: "u32", ldc: "u32",
      alpha: "f32",
      batchSize: "u32",
      batchStrideA: "u32", batchStrideB: "u32", batchStrideC: "u32",
    },
    grid: () => [1],

    kernel: useBlockOps
      ? (ctx) => matmulKernelBlockOps(ctx, { batched, kSplit }, params, blockOpsPostAcc)
      : (ctx) => matmulKernelTileOps(ctx, { batched, kSplit }, params, tileOpsPostAcc),
  };
}

interface MatmulKernelParams {
  tileM: number; tileN: number; tileK: number;
  threadTileM: number; threadTileN: number;
  wgSizeX: number; wgSizeY: number;
  transA: boolean; transB: boolean;
  outputDtype: DType;
}

/** TileOps kernel body — pure matmul with optional post-accumulate callback. */
function matmulKernelTileOps(
  ctx: KernelContext,
  opts: { batched?: boolean; kSplit?: number },
  p: MatmulKernelParams,
  postAccumulate?: TileOpsPostAcc,
): void {
  const { batched, kSplit } = opts;
  const { tileM, tileN, tileK, threadTileM, threadTileN, wgSizeX, wgSizeY, transA, transB, outputDtype } = p;

  const tileConfig: TileConfig = {
    BLOCK_M: tileM, BLOCK_N: tileN, BLOCK_K: tileK,
    threadTileM, threadTileN, wgSizeX, wgSizeY,
  };
  const ops = new TileOps(ctx, tileConfig);

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
    batchOffA = ctx.emitLet("batch_offset_a", batchIdx.mul(ctx.uniform("batchStrideA")));
    batchOffB = ctx.emitLet("batch_offset_b", batchIdx.mul(ctx.uniform("batchStrideB")));
    batchOffC = ctx.emitLet("batch_offset_c", batchIdx.mul(ctx.uniform("batchStrideC")));
  } else {
    const batchIdx = ctx.emitLet("batch_idx", ctx.const(0, "u32"));
    batchOffA = ctx.emitLet("batch_offset_a", batchIdx.mul(ctx.uniform("batchStrideA")));
    batchOffB = ctx.emitLet("batch_offset_b", batchIdx.mul(ctx.uniform("batchStrideB")));
    batchOffC = ctx.emitLet("batch_offset_c", batchIdx.mul(ctx.uniform("batchStrideC")));
  }

  // 3. Workgroup + thread positions
  const wgRow = ctx.emitLet("wg_row", ctx.programId(1).mul(ctx.const(tileM, "u32")));
  const wgCol = ctx.emitLet("wg_col", ctx.programId(0).mul(ctx.const(tileN, "u32")));
  const threadRow = ctx.emitLet("thread_row", ctx.threadIdx(1));
  const threadCol = ctx.emitLet("thread_col", ctx.threadIdx(0));

  // 4. Block ranges
  const offsM = ops.arange(wgRow, tileM);
  const offsN = ops.arange(wgCol, tileN);

  // 5. Transpose strides
  const cOne = ctx.const(1, "u32");
  const strideAm = transA ? cOne : lda;
  const strideAk = transA ? lda : cOne;
  const strideBk = transB ? cOne : ldb;
  const strideBn = transB ? ldb : cOne;

  // 6. Accumulator
  const acc = ops.zeros(tileM, tileN);

  // 7. K-loop bounds
  let kStart: BlockExpr;
  let kEnd: BlockExpr;
  let numKTiles: BlockExpr;
  const cTK = ctx.const(tileK, "u32");

  if (kSplit) {
    const kPerSplit = ctx.emitLet("k_per_split",
      k.add(ctx.const(kSplit - 1, "u32")).div(ctx.const(kSplit, "u32")));
    kStart = ctx.emitLet("k_start", splitIdx!.mul(kPerSplit));
    kEnd = ctx.emitLet("k_end", kStart.add(kPerSplit).min(k));
    numKTiles = ctx.emitLet("num_k_tiles",
      kEnd.sub(kStart).add(ctx.const(tileK - 1, "u32")).div(cTK));
  } else {
    numKTiles = ctx.emitLet("num_k_tiles",
      k.add(ctx.const(tileK - 1, "u32")).div(cTK));
    kStart = ctx.emitLet("k_start", ctx.const(0, "u32"));
    kEnd = ctx.emitLet("k_end", k);
  }

  // 8. K-loop — the core matmul
  ctx.forRange(ctx.const(0, "u32"), numKTiles, (kTile) => {
    const kOffset = ctx.emitLet("k_offset", kStart.add(kTile.mul(cTK)));
    const offsK = ops.arange(kOffset, tileK);

    const aPtr = buildPtr(batchOffA, offsM.outer(strideAm), offsK.inner(strideAk));
    const aMask = buildMask(offsM.lt(m), offsK.lt(kEnd));
    const a = ops.load("a", aPtr, aMask);

    const bPtr = buildPtr(batchOffB, offsK.outer(strideBk), offsN.inner(strideBn));
    const bMask = buildMask(offsK.lt(kEnd), offsN.lt(n));
    const b = ops.load("b", bPtr, bMask);

    ops.dot(a, b, acc);
  });

  // 9. Store — two paths: K-split (raw partials) vs normal (with optional post-accumulate)
  if (kSplit) {
    const splitBase = ctx.emitLet("split_base", splitIdx!.mul(m.mul(n)));
    const outPtr = buildPtr(splitBase, offsM.outer(ldc), offsN.inner(cOne));
    const outMask = buildMask(offsM.lt(m), offsN.lt(n));
    ops.store("out", outPtr, acc, outMask);
  } else {
    acc.mul_(alpha.toF32());

    if (postAccumulate) {
      const addressing = { batchOffC, wgRow, wgCol, threadRow, threadCol, m, n, ldc, offsM, offsN };
      postAccumulate.fn(ctx, ops, acc, addressing);
      if (!postAccumulate.handlesStore) {
        const outPtr = buildPtr(batchOffC, offsM.outer(ldc), offsN.inner(cOne));
        const outMask = buildMask(offsM.lt(m), offsN.lt(n));
        ops.store("out", outPtr, acc, outMask);
      }
    } else {
      if (outputDtype === "f16") {
        acc.castTo_("f16");
      }
      const outPtr = buildPtr(batchOffC, offsM.outer(ldc), offsN.inner(cOne));
      const outMask = buildMask(offsM.lt(m), offsN.lt(n));
      ops.store("out", outPtr, acc, outMask);
    }
  }
}

/** BlockOps kernel body — pure matmul with optional post-accumulate callback. */
function matmulKernelBlockOps(
  ctx: KernelContext,
  opts: { batched?: boolean; kSplit?: number },
  p: MatmulKernelParams,
  postAccumulate?: BlockOpsPostAccFn,
): void {
  const { batched, kSplit } = opts;
  const { tileM, tileN, tileK, threadTileM, threadTileN, wgSizeX, wgSizeY, transA, transB, outputDtype } = p;

  const ops = new BlockOps(ctx, {
    wgSize: [wgSizeX, wgSizeY],
    threadTile: [threadTileM, threadTileN],
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
    batchOffA = ctx.emitLet("batch_offset_a", batchIdx.mul(ctx.uniform("batchStrideA")));
    batchOffB = ctx.emitLet("batch_offset_b", batchIdx.mul(ctx.uniform("batchStrideB")));
    batchOffC = ctx.emitLet("batch_offset_c", batchIdx.mul(ctx.uniform("batchStrideC")));
  } else {
    const batchIdx = ctx.emitLet("batch_idx", ctx.const(0, "u32"));
    batchOffA = ctx.emitLet("batch_offset_a", batchIdx.mul(ctx.uniform("batchStrideA")));
    batchOffB = ctx.emitLet("batch_offset_b", batchIdx.mul(ctx.uniform("batchStrideB")));
    batchOffC = ctx.emitLet("batch_offset_c", batchIdx.mul(ctx.uniform("batchStrideC")));
  }

  // 3. Workgroup + thread positions
  const wgRow = ctx.emitLet("wg_row", ctx.programId(1).mul(ctx.const(tileM, "u32")));
  const wgCol = ctx.emitLet("wg_col", ctx.programId(0).mul(ctx.const(tileN, "u32")));
  const threadRow = ctx.emitLet("thread_row", ctx.threadIdx(1));
  const threadCol = ctx.emitLet("thread_col", ctx.threadIdx(0));

  // 4. Block ranges
  const offsM = ops.arange(wgRow, tileM);
  const offsN = ops.arange(wgCol, tileN);

  // 5. Transpose strides
  const cOne = ctx.const(1, "u32");
  const strideAm = transA ? cOne : lda;
  const strideAk = transA ? lda : cOne;
  const strideBk = transB ? cOne : ldb;
  const strideBn = transB ? ldb : cOne;

  // 6. Accumulator — per-thread [threadTileM × threadTileN]
  const acc = ops.zeros(threadTileM, threadTileN);

  // 7. K-loop bounds
  let kStart: BlockExpr;
  let kEnd: BlockExpr;
  let numKTiles: BlockExpr;
  const cTK = ctx.const(tileK, "u32");

  if (kSplit) {
    const kPerSplit = ctx.emitLet("k_per_split",
      k.add(ctx.const(kSplit - 1, "u32")).div(ctx.const(kSplit, "u32")));
    kStart = ctx.emitLet("k_start", splitIdx!.mul(kPerSplit));
    kEnd = ctx.emitLet("k_end", kStart.add(kPerSplit).min(k));
    numKTiles = ctx.emitLet("num_k_tiles",
      kEnd.sub(kStart).add(ctx.const(tileK - 1, "u32")).div(cTK));
  } else {
    numKTiles = ctx.emitLet("num_k_tiles",
      k.add(ctx.const(tileK - 1, "u32")).div(cTK));
    kStart = ctx.emitLet("k_start", ctx.const(0, "u32"));
    kEnd = ctx.emitLet("k_end", k);
  }

  // 8. K-loop — the core matmul
  ctx.forRange(ctx.const(0, "u32"), numKTiles, (kTile) => {
    const kOffset = ctx.emitLet("k_offset", kStart.add(kTile.mul(cTK)));
    const offsK = ops.arange(kOffset, tileK);

    // Cooperative tile loads using loadTile (→ shared memory)
    const aPtr = buildPtr(batchOffA, offsM.outer(strideAm), offsK.inner(strideAk));
    const aMask = buildMask(offsM.lt(m), offsK.lt(kEnd));
    const a = ops.loadTile("a", aPtr, aMask);

    const bPtr = buildPtr(batchOffB, offsK.outer(strideBk), offsN.inner(strideBn));
    const bMask = buildMask(offsK.lt(kEnd), offsN.lt(n));
    const b = ops.loadTile("b", bPtr, bMask);

    // acc += a @ b — shared × shared outer product
    ops.dotAccum(a, b, acc);
  });

  // 9. Store — two paths: K-split (raw partials) vs normal (with optional post-accumulate)
  if (kSplit) {
    const splitBase = ctx.emitLet("split_base", splitIdx!.mul(m.mul(n)));
    const outPtr = buildPtr(splitBase, offsM.outer(ldc), offsN.inner(cOne));
    const outMask = buildMask(offsM.lt(m), offsN.lt(n));
    ops.storeTile("out", acc, outPtr, outMask);
  } else {
    acc.mul_(alpha.toF32());

    if (postAccumulate) {
      const threadOutBase = ctx.emitLet("thread_out_base", batchOffC.add(
        wgRow.add(threadRow.mul(ctx.const(threadTileM, "u32"))).mul(ldc)
          .add(wgCol.add(threadCol.mul(ctx.const(threadTileN, "u32"))))
      ));
      postAccumulate(ctx, ops, acc, { offsN, threadOutBase, threadOutStride: ldc });
    } else if (outputDtype === "f16") {
      acc.castTo_("f16");
    }

    const outPtr = buildPtr(batchOffC, offsM.outer(ldc), offsN.inner(cOne));
    const outMask = buildMask(offsM.lt(m), offsN.lt(n));
    ops.storeTile("out", acc, outPtr, outMask);
  }
}

// ============================================================================
// K-Split Reduction Kernel (imperative mode)
// ============================================================================

export function createKSplitReductionKernel(kSplitCount: number, outputDtype: DType): TileKernelSpec {
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
      ctx.ifThen(idx.ge(totalElements), () => { ctx.emitReturn(); });

      const alpha = ctx.uniform("alpha").bitcastTo("f32");
      const sumVar = ctx.emitVar("sum", "f32", ctx.load("partials", idx));
      for (let p = 1; p < kSplitCount; p++) {
        sumVar.addAssign(ctx.load("partials", idx.add(totalElements.mul(ctx.u32(p)))));
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

export function generateTiledMatmulShaderTileIR(options: CodegenOptions): string {
  return compileTileKernel(createTiledMatmulKernel(options));
}

export function generateKSplitReductionShaderTileIR(kSplitCount: number, outputDtype: DType): string {
  return compileTileKernel(createKSplitReductionKernel(kSplitCount, outputDtype));
}
