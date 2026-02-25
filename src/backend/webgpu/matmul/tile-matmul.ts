/**
 * Tile IR Matmul: Triton-like 2D tiled matrix multiplication.
 *
 * Uses the TileOps block-level API: the kernel author thinks in BLOCK_M × BLOCK_N
 * tiles, never about threads. The compiler handles thread mapping, shared memory,
 * barriers, and cooperative loading.
 *
 * The dispatch infrastructure is unchanged — we just swap the WGSL generation.
 */

import type { TileKernelSpec, DataType, BlockExpr, KernelContext } from "../tile-ir";
import type { CodegenOptions, EpilogueConfig } from "./codegen";
import { getWorkgroupSize, type DType } from "./types";
import { TileOps, buildPtr, buildMask, type TileConfig } from "../tile-ops";

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

/** Apply a non-binary epilogue op to the accumulator. */
function applyEpilogueToAcc(
  ctx: KernelContext,
  ops: TileOps,
  acc: InstanceType<typeof import("../tile-ops").Accumulator>,
  op: EpilogueConfig["ops"][number],
  offsN: InstanceType<typeof import("../tile-ops").TileRange>,
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
// Main Kernel Builder
// ============================================================================

/**
 * Create a TileKernelSpec for tiled matrix multiplication.
 *
 * The kernel body uses TileOps: block-level loads, dot, and store.
 * The compiler lowers these to cooperative loading loops, barriers,
 * outer products, and bounds-checked stores.
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

  const tileConfig: TileConfig = {
    BLOCK_M: tileM, BLOCK_N: tileN, BLOCK_K: tileK,
    threadTileM, threadTileN, wgSizeX, wgSizeY,
  };

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

    kernel(ctx) {
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

      // 5. Transpose strides — just different index math at build time
      const cOne = ctx.const(1, "u32");
      const strideAm = transA ? cOne : lda;
      const strideAk = transA ? lda : cOne;
      const strideBk = transB ? cOne : ldb;
      const strideBn = transB ? ldb : cOne;

      // 6. Accumulator — block-sized [BLOCK_M, BLOCK_N]
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

        // Cooperative tile loads — transpose is just different strides
        const aPtr = buildPtr(batchOffA, offsM.outer(strideAm), offsK.inner(strideAk));
        const aMask = buildMask(offsM.lt(m), offsK.lt(kEnd));
        const a = ops.load("a", aPtr, aMask);

        const bPtr = buildPtr(batchOffB, offsK.outer(strideBk), offsN.inner(strideBn));
        const bMask = buildMask(offsK.lt(kEnd), offsN.lt(n));
        const b = ops.load("b", bPtr, bMask);

        // acc += a @ b — the compiler generates barriers + outer product
        ops.dot(a, b, acc);
      });

      // 9. Epilogue + Store
      if (kSplit) {
        // K-split: partial sums to temp buffer (no alpha, no epilogue)
        const splitBase = ctx.emitLet("split_base", splitIdx!.mul(m.mul(n)));
        const outPtr = buildPtr(splitBase, offsM.outer(ldc), offsN.inner(cOne));
        const outMask = buildMask(offsM.lt(m), offsN.lt(n));
        ops.store("out", outPtr, acc, outMask);
      } else if (epilogue && hasBinaryEpilogue(epilogue)) {
        // Binary epilogue needs the full output index → imperative store path
        acc.mul_(alpha.toF32());
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
      } else {
        // Simple epilogue path — accumulator ops + tile store
        acc.mul_(alpha.toF32());

        if (epilogue && epilogue.ops.length > 0) {
          for (const op of epilogue.ops) {
            applyEpilogueToAcc(ctx, ops, acc, op, offsN);
          }
          if (epilogue.outputDtype === "f16" && !epilogue.ops.some(o => o.kind === "cast")) {
            acc.castTo_("f16");
          }
        } else if (outputDtype === "f16") {
          acc.castTo_("f16");
        }

        const outPtr = buildPtr(batchOffC, offsM.outer(ldc), offsN.inner(cOne));
        const outMask = buildMask(offsM.lt(m), offsN.lt(n));
        ops.store("out", outPtr, acc, outMask);
      }
    },
  };
}

// ============================================================================
// K-Split Reduction Kernel (unchanged — auto-phase mode)
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
    grid: (u) => [Math.ceil(u.totalElements / 256)],
    kernel(ctx) {
      const gid = ctx.programId(0).mul(ctx.const(256, "u32")).add(ctx.blockRange(ctx.const(256, "u32")));
      const totalElements = ctx.uniform("totalElements");
      const alpha = ctx.uniform("alpha").toF32();
      const baseVal = ctx.load("partials", gid);
      let sum = baseVal;
      for (let p = 1; p < kSplitCount; p++) {
        sum = sum.add(ctx.load("partials", gid.add(totalElements.mul(ctx.const(p, "u32")))));
      }
      const result = sum.mul(alpha);
      if (outputDtype === "f16") {
        ctx.store("out", gid, result.toF16());
      } else {
        ctx.store("out", gid, result);
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
