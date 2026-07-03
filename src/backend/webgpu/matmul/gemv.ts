/**
 * Dedicated GEMV kernels for M=1 matmuls (autoregressive decode), expressed
 * as tile-IR programs (the pilot for matmul-class kernels in the IR).
 *
 * The tiled matmul kernel wastes (tileM-1)/tileM of every workgroup on rows
 * that don't exist when M=1, and its K-split rescue still runs the full 2D
 * tile machinery. These kernels treat A as a K-length vector and read B
 * exactly once, aiming for memory bandwidth:
 *
 *  - NT mode (B stored [N,K] row-major — the api.linear weight layout after
 *    simple-transpose detection; all decode projections + lm_head): one
 *    workgroup per output row computes dot(a, B[row,:]) via
 *    ctx.wgReduce("sum") — strided coalesced row reads, subgroup reduction
 *    when available, shared-memory tree otherwise. Rows beyond one dispatch
 *    dimension (lm_head: N=151936) map onto a 2D grid with a row guard.
 *  - NN mode (B stored [K,N] row-major): one thread per output column,
 *    cooperatively staging a[] into a shared-memory tile in wgSize chunks
 *    (ctx.sharedArray + ctx.barrier), forStride over K; B reads are
 *    coalesced across threads. When the column grid alone can't fill the
 *    GPU, K splits across workgroup.y (split_k uniform) into f32 partials
 *    [split_k, N] reduced by the existing matmul K-split reduction kernel
 *    (same layout: partials[p * N + col]).
 *
 * Mixed dtype (f32 activations × f16 weights) converts on load via per-
 * binding dtypes + .toF32(); accumulation is always f32. Output is f32
 * unless both inputs are f16.
 *
 * N and K are shape-derived uniforms (stable per pipeline); alpha is packed
 * into the params data exactly like the tiled path — nothing per-step-
 * varying is baked into shader code.
 *
 * Known IR gap (noted, not worked around): the imperative tile-IR has no
 * vec4 loads from STORAGE bindings — vec4 exists only for register/shared
 * arrays and the whole-kernel elementwise `vectorize` mode. Loads here are
 * scalar but fully coalesced across threads.
 */

import type { AutotuneConfig, TileKernelSpec, TuneParam } from "../tile-ir";
import { compileTileKernel } from "../tile-compiler";
import type { DataType } from "../tile-ir";
import type { DType } from "./types";

/** Default threads per workgroup (NT: threads cooperating on one row's dot;
 *  NN: output columns handled per workgroup / staging chunk length). */
export const GEMV_DEFAULT_WG_SIZE = 256;

/** Mirror of the dispatch.ts K-split occupancy targets. */
const GEMV_KSPLIT_TARGET_WG = 128;
const GEMV_KSPLIT_MIN_K = 512;
const GEMV_KSPLIT_MAX_FACTOR = 32;
/** WebGPU guaranteed maxComputeWorkgroupsPerDimension. */
const MAX_WG_PER_DIM = 65535;

export interface GemvKernelOptions {
  mode: "nt" | "nn";
  dtypeA: DType;
  dtypeB: DType;
  /** Ignored (partials are f32) when mode === "nn" && kSplit. */
  outputDtype: DType;
  /** NN only: write f32 partials [split_k, N]; the reduction applies alpha. */
  kSplit: boolean;
  /** Threads per workgroup (TuneParam; default GEMV_DEFAULT_WG_SIZE). */
  wgSize?: number;
}

export interface GemvRoute {
  mode: "nt" | "nn";
  /** ≥2 → NN K-split into partials + reduction pass. Always 1 for NT. */
  splitK: number;
  dispatch: [number, number, number];
}

/**
 * Decide whether (and how) an M=1, batch=1, no-epilogue matmul routes to the
 * GEMV kernels. Geometry-only (safe to run at plan/lowering time — the
 * stage-4 stream generator consumes the same plan). Returns null to fall
 * through to the tiled path.
 */
export function computeGemvRoute(
  n: number,
  k: number,
  transB: boolean,
  wgSize: number = GEMV_DEFAULT_WG_SIZE,
): GemvRoute | null {
  if (n <= 1 || k < 1) return null;

  if (transB) {
    // NT: one workgroup per row. 2D grid with a row guard covers N beyond
    // one dispatch dimension (lm_head). gy chosen so overshoot < gy.
    const gy = Math.ceil(n / MAX_WG_PER_DIM);
    if (gy > MAX_WG_PER_DIM) return null;
    const gx = Math.ceil(n / gy);
    return { mode: "nt", splitK: 1, dispatch: [gx, gy, 1] };
  }

  // NN: one thread per output column.
  const gx = Math.ceil(n / wgSize);
  if (gx > MAX_WG_PER_DIM) return null;
  let splitK = 1;
  if (gx < GEMV_KSPLIT_TARGET_WG && k >= GEMV_KSPLIT_MIN_K) {
    splitK = Math.min(
      Math.ceil(GEMV_KSPLIT_TARGET_WG / gx),
      Math.floor(k / wgSize),
      GEMV_KSPLIT_MAX_FACTOR,
    );
    if (splitK < 2) splitK = 1;
  }
  // Measured threshold (V100 battery): with large K but a tiny column grid,
  // splitK is capped by floor(k/wgSize) and the total grid stays single-digit
  // — the tiled gemv_row config + its K-split wins there (e.g. K=1024 N=128:
  // 38µs vs 32µs). Small-K shapes (< MIN_K) stay: the simpler kernel wins on
  // launch overhead even at one workgroup.
  if (k >= GEMV_KSPLIT_MIN_K && gx * splitK < 16) return null;
  return { mode: "nn", splitK, dispatch: [gx, splitK, 1] };
}

export function getGemvShaderCacheKey(o: GemvKernelOptions): string {
  return [
    "gemv",
    o.mode,
    `${o.dtypeA}x${o.dtypeB}`,
    `out_${o.outputDtype}`,
    `wg${o.wgSize ?? GEMV_DEFAULT_WG_SIZE}`,
    o.kSplit ? "ks" : "",
  ]
    .filter(Boolean)
    .join("_");
}

/**
 * Build the GEMV TileKernelSpec.
 *
 * Bindings [a, b, out] + uniform config {n, k, alpha, split_k} at binding 3
 * — the same order as the tiled matmul, so the dispatch tail and the stream
 * generator bind identically for both kernels.
 */
export function createGemvKernel(o: GemvKernelOptions): TileKernelSpec {
  const wgSize = o.wgSize ?? GEMV_DEFAULT_WG_SIZE;
  const outDtype: DType = o.mode === "nn" && o.kSplit ? "f32" : o.outputDtype;
  const needsF16 =
    o.dtypeA === "f16" || o.dtypeB === "f16" || outDtype === "f16";
  const bindings = {
    a: { storage: "read" as const, type: o.dtypeA as DataType },
    b: { storage: "read" as const, type: o.dtypeB as DataType },
    out: { storage: "read_write" as const, type: outDtype as DataType },
  };
  const uniforms = {
    n: "u32" as const,
    k: "u32" as const,
    alpha: "f32" as const,
    split_k: "u32" as const,
  };

  if (o.mode === "nt") {
    return {
      name: "gemvNT",
      workgroupSize: wgSize,
      enableF16: needsF16,
      uniformBindingIndex: 3,
      bindings,
      uniforms,
      // Real grid dims come from computeGemvRoute at plan time (2D split of
      // N over the 65535/dim limit); this mirrors the tiled matmul pattern.
      grid: (u) => {
        const gy = Math.ceil(u.n / MAX_WG_PER_DIM);
        return [Math.ceil(u.n / gy), gy];
      },
      kernel(ctx) {
        const n = ctx.emitLet("n", ctx.uniform("n"));
        const k = ctx.emitLet("k", ctx.uniform("k"));
        const alpha = ctx.uniform("alpha", "f32");
        const tid = ctx.localIndex();
        // Row from the 2D grid; tail workgroups (row ≥ n) recompute the
        // last row (keeps wgReduce's barriers in uniform control flow) and
        // are masked at the store.
        const row = ctx.emitLet(
          "row",
          ctx.programId(1).mul(ctx.numPrograms(0)).add(ctx.programId(0)),
        );
        const rowC = ctx.emitLet("row_c", row.min(n.sub(ctx.u32(1))));
        const rowBase = ctx.emitLet("row_base", rowC.mul(k));
        const dot = ctx.wgReduce("sum", tid, k, wgSize, (i) =>
          ctx
            .load("a", i)
            .toF32()
            .mul(ctx.load("b", rowBase.add(i)).toF32()),
        );
        ctx.ifThen(tid.eq(ctx.u32(0)), () => {
          ctx.ifThen(row.lt(n), () => {
            const val = dot.mul(alpha);
            ctx.emitStore("out", row, outDtype === "f16" ? val.toF16() : val);
          });
        });
      },
    };
  }

  // NN mode: thread-per-column with cooperative a[] staging; optional
  // K-split across workgroup.y into f32 partials.
  return {
    name: o.kSplit ? "gemvNNKSplit" : "gemvNN",
    workgroupSize: wgSize,
    enableF16: needsF16,
    uniformBindingIndex: 3,
    bindings,
    uniforms,
    grid: (u) => [Math.ceil(u.n / wgSize), u.split_k ?? 1],
    kernel(ctx) {
      const n = ctx.emitLet("n", ctx.uniform("n"));
      const k = ctx.emitLet("k", ctx.uniform("k"));
      const alpha = ctx.uniform("alpha", "f32");
      const splitK = ctx.emitLet("split_k", ctx.uniform("split_k"));
      const tid = ctx.localIndex();
      const col = ctx.emitLet("col", ctx.globalId(0));
      // Tail threads (col ≥ n) compute the last column redundantly so the
      // staging barriers stay in uniform control flow; store is masked.
      const colC = ctx.emitLet("col_c", col.min(n.sub(ctx.u32(1))));
      const split = ctx.emitLet("split", ctx.programId(1));
      const chunkLen = ctx.emitLet(
        "chunk_len",
        k.add(splitK.sub(ctx.u32(1))).div(splitK),
      );
      const kStart = ctx.emitLet("k_start", split.mul(chunkLen));
      const kEnd = ctx.emitLet("k_end", kStart.add(chunkLen).min(k));

      const aTile = ctx.sharedArray("a_tile", wgSize);
      const acc = ctx.emitVar("acc", "f32", ctx.f32(0));

      ctx.forStride(kStart, kEnd, wgSize, (kb) => {
        const stageIdx = ctx.emitLet("stage_idx", kb.add(tid));
        ctx.ifThen(stageIdx.lt(kEnd), () => {
          aTile.write(tid, ctx.load("a", stageIdx).toF32());
        });
        ctx.barrier();
        const lim = ctx.emitLet("lim", kEnd.sub(kb).min(ctx.u32(wgSize)));
        ctx.forRange(ctx.u32(0), lim, (j) => {
          acc.addAssign(
            aTile
              .read(j)
              .mul(ctx.load("b", kb.add(j).mul(n).add(colC)).toF32()),
          );
        });
        ctx.barrier();
      });

      ctx.ifThen(col.lt(n), () => {
        if (o.kSplit) {
          // Raw f32 partials [split_k, N]; the shared K-split reduction
          // kernel sums them and applies alpha.
          ctx.emitStore("out", split.mul(n).add(col), acc.get());
        } else {
          const val = acc.get().mul(alpha);
          ctx.emitStore("out", col, outDtype === "f16" ? val.toF16() : val);
        }
      });
    },
  };
}

export function generateGemvShaderTileIR(o: GemvKernelOptions): string {
  return compileTileKernel(createGemvKernel(o));
}

// ============================================================================
// Autotune wiring (searchable later via autotuneTileKernel)
// ============================================================================

/** The GEMV tunable: threads per workgroup. Single source for both the
 *  tile-IR AutotuneConfig below and the variant registry's candidates. */
export const GEMV_WG_SIZE_PARAM: TuneParam = {
  values: [64, 128, 256],
  default: GEMV_DEFAULT_WG_SIZE,
};

/** AutotuneConfig for a GEMV variant: the only tunable is wgSize. The grid
 *  is a function of the uniforms + wgSize (closed over per factory call),
 *  so autotuneTileKernel can dispatch candidates standalone. */
export function gemvAutotuneConfig(
  base: Omit<GemvKernelOptions, "wgSize">,
): AutotuneConfig {
  return {
    factory: (config) => createGemvKernel({ ...base, wgSize: config.wgSize }),
    params: { wgSize: GEMV_WG_SIZE_PARAM },
  };
}
