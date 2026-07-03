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
 *    workgroup handles rowsPerWg output rows — the workgroup is split into
 *    rowsPerWg lane-groups, each computing dot(a, B[row,:]) with vec4
 *    storage loads (ctx.loadVec4, gated on k % 4 === 0) and a segmented
 *    reduction (ctx.wgReduceSegmented — subgroup butterfly / intrinsic +
 *    smem tree when available). Rows beyond one dispatch dimension
 *    (lm_head: N=151936) map onto a 2D grid via rowGrid2d/ctx.rowIndex2d
 *    with a row guard.
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
 * The three IR gaps this kernel originally noted (vec4 storage loads,
 * segmented workgroup reduction, >65535-row grids) are now first-class
 * tile-IR primitives (task #62): ctx.loadVec4, ctx.wgReduceSegmented,
 * rowGrid2d/ctx.rowIndex2d. K not divisible by 4 falls back to scalar
 * (fully coalesced) loads.
 */

import { applyFusedOp } from "../fusion-tile-ir";
import { compileTileKernel } from "../tile-compiler";
import type {
  AutotuneConfig,
  DataType,
  SeamFn,
  TileKernelSpec,
  TuneParam,
} from "../tile-ir";
import { splitWorkgroups2d } from "../tile-ir";
import type { DType, EpilogueConfig } from "./types";

/** Default threads per workgroup (NT: threads cooperating on rowsPerWg rows;
 *  NN: output columns handled per workgroup / staging chunk length). */
export const GEMV_DEFAULT_WG_SIZE = 256;

/** Fallback output rows per workgroup for NT mode when no explicit value
 *  and no shape is available (kernel-spec default). */
export const GEMV_NT_DEFAULT_ROWS_PER_WG = 1;

/** Very-large-N threshold above which multiple rows per workgroup win. */
const GEMV_NT_MULTIROW_MIN_N = 32768;

/**
 * Heuristic NT rowsPerWg for a given N (geometry-only, plan-time safe).
 * Measured (V100 sweep, f32, K∈{2048,6144}): vec4 loads dominate the win;
 * rowsPerWg=1 is best or tied for N ≤ 6144, while the huge-row lm_head
 * shape (N=151936) runs ~6% faster with 4 lane-groups of wgSize/4 (r4
 * 1328µs vs r1 1411µs at wg256+vec4).
 */
export function defaultGemvNtRowsPerWg(n: number): number {
  return n >= GEMV_NT_MULTIROW_MIN_N ? 4 : 1;
}

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
  /** NT only: output rows per workgroup (TuneParam; default
   *  GEMV_NT_DEFAULT_ROWS_PER_WG). Must divide wgSize; both are powers of
   *  two so the lane-group size is too (wgReduceSegmented requirement). */
  rowsPerWg?: number;
  /** NT only: vec4 storage loads on the row dot-product. Requires
   *  k % 4 === 0 (computeGemvRoute gates this — single source). */
  vec4?: boolean;
  /** Simple epilogue (bias / unary activation) injected at the kernel's
   *  "epilogue" seam. Must satisfy gemvSupportsEpilogue(); incompatible
   *  with kSplit (partials must stay raw — the caller gates this). */
  epilogue?: EpilogueConfig;
}

// ============================================================================
// Epilogue seam (composable epilogues via TileKernelSpec.seams)
// ============================================================================

/** Unary activations the GEMV epilogue seam supports — the epilogue-fusion
 *  detector's vocabulary (matmul-epilogue.ts getEpilogueOpName), all
 *  implemented scalar-wise by the ops-registry bridge applyFusedOp. */
const GEMV_EPILOGUE_UNARY_OPS: ReadonlySet<string> = new Set([
  "relu",
  "silu",
  "sigmoid",
  "tanh",
  "gelu",
  "gelu_tanh",
  "gelu_erf",
]);

/**
 * Can the GEMV kernels run this epilogue at their "epilogue" seam?
 * Pilot scope: bias adds and simple unary activations only — no binary
 * (residual) inputs, no mid-chain casts, no input casts (the caller checks
 * those separately). `undefined` (bare matmul) is trivially supported.
 */
export function gemvSupportsEpilogue(epilogue?: EpilogueConfig): boolean {
  if (!epilogue) return true;
  return epilogue.ops.every(
    (op) =>
      op.kind === "none" ||
      op.kind === "bias" ||
      (op.kind === "unary" && GEMV_EPILOGUE_UNARY_OPS.has(op.op)),
  );
}

/**
 * Build the SeamFn for a supported epilogue: bias loads address
 * `epilogue_in{i}` at the output element index (for M=1 the output index IS
 * the bias/N index — same addressing as the tiled path's load1d(offsN));
 * unary activations route through applyFusedOp (the same ops-registry WGSL
 * expression bridge elementwise fusion uses).
 */
function gemvEpilogueSeam(epilogue: EpilogueConfig): SeamFn {
  return (ctx, value, args) => {
    const outIndex = args.outIndex;
    if (!outIndex) throw new Error("gemv epilogue seam requires outIndex");
    let v = value;
    for (const op of epilogue.ops) {
      switch (op.kind) {
        case "none":
          break;
        case "bias":
          v = v.add(ctx.load(`epilogue_in${op.inputIndex}`, outIndex).toF32());
          break;
        case "unary":
          v = applyFusedOp(ctx, op.op, [v]);
          break;
        default:
          throw new Error(
            `gemv epilogue seam: unsupported op kind '${op.kind}' ` +
              `(gemvSupportsEpilogue must gate this)`,
          );
      }
    }
    return v;
  };
}

/** Cache-key fragment describing an epilogue (shape of the injected expr). */
function epilogueKeyFragment(epilogue?: EpilogueConfig): string {
  if (!epilogue || epilogue.ops.length === 0) return "";
  const ops = epilogue.ops
    .map((op) =>
      op.kind === "bias"
        ? `bias${op.inputIndex}`
        : op.kind === "unary"
          ? op.op
          : op.kind,
    )
    .join("+");
  return `epi_${ops}_${epilogue.outputDtype}`;
}

export interface GemvRoute {
  mode: "nt" | "nn";
  /** ≥2 → NN K-split into partials + reduction pass. Always 1 for NT. */
  splitK: number;
  dispatch: [number, number, number];
  /** NT: output rows per workgroup. Always 1 for NN. */
  rowsPerWg: number;
  /** NT: whether the kernel uses vec4 storage loads (k % 4 === 0). */
  vec4: boolean;
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
  rowsPerWg?: number,
): GemvRoute | null {
  if (n <= 1 || k < 1) return null;

  if (transB) {
    // NT: rowsPerWg output rows per workgroup (lane-groups + segmented
    // reduce). 2D grid with a row guard covers N beyond one dispatch
    // dimension (lm_head); overshoot workgroups are masked in-kernel.
    const rpw = rowsPerWg ?? defaultGemvNtRowsPerWg(n);
    if (rpw < 1 || wgSize % rpw !== 0) return null;
    const groupSize = wgSize / rpw;
    if ((groupSize & (groupSize - 1)) !== 0) return null;
    const totalWg = Math.ceil(n / rpw);
    const [gx, gy] = splitWorkgroups2d(totalWg);
    if (gy > MAX_WG_PER_DIM) return null;
    return {
      mode: "nt",
      splitK: 1,
      dispatch: [gx, gy, 1],
      rowsPerWg: rpw,
      vec4: k % 4 === 0,
    };
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
  return {
    mode: "nn",
    splitK,
    dispatch: [gx, splitK, 1],
    rowsPerWg: 1,
    vec4: false,
  };
}

export function getGemvShaderCacheKey(o: GemvKernelOptions): string {
  return [
    "gemv",
    o.mode,
    `${o.dtypeA}x${o.dtypeB}`,
    `out_${o.outputDtype}`,
    `wg${o.wgSize ?? GEMV_DEFAULT_WG_SIZE}`,
    o.mode === "nt" ? `r${o.rowsPerWg ?? GEMV_NT_DEFAULT_ROWS_PER_WG}` : "",
    o.mode === "nt" && o.vec4 ? "v4" : "",
    o.kSplit ? "ks" : "",
    epilogueKeyFragment(o.epilogue),
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
  const bindings: Record<
    string,
    { storage: "read" | "read_write"; type: DataType }
  > = {
    a: { storage: "read", type: o.dtypeA as DataType },
    b: { storage: "read", type: o.dtypeB as DataType },
    out: { storage: "read_write", type: outDtype as DataType },
  };
  // Epilogue seam: extra inputs bind AFTER the uniform slot (binding 4+),
  // matching the tiled plan's [a, b, out, params, ...epilogueInputs] order.
  const epilogue =
    o.epilogue && o.epilogue.ops.length > 0 ? o.epilogue : undefined;
  if (epilogue) {
    if (o.kSplit) {
      throw new Error("gemv: epilogue is incompatible with kSplit partials");
    }
    if (!gemvSupportsEpilogue(epilogue)) {
      throw new Error("gemv: unsupported epilogue (see gemvSupportsEpilogue)");
    }
    for (let i = 0; i < epilogue.additionalInputCount; i++) {
      bindings[`epilogue_in${i}`] = { storage: "read", type: "f32" };
    }
  }
  const seams = epilogue ? { epilogue: gemvEpilogueSeam(epilogue) } : undefined;
  const uniforms = {
    n: "u32" as const,
    k: "u32" as const,
    alpha: "f32" as const,
    split_k: "u32" as const,
  };

  if (o.mode === "nt") {
    const rowsPerWg = o.rowsPerWg ?? GEMV_NT_DEFAULT_ROWS_PER_WG;
    const groupSize = wgSize / rowsPerWg;
    const useVec4 = o.vec4 ?? false;
    return {
      name: "gemvNT",
      workgroupSize: wgSize,
      enableF16: needsF16,
      uniformBindingIndex: 3,
      bindings,
      uniforms,
      seams,
      // Real grid dims come from computeGemvRoute at plan time (2D split of
      // ceil(N/rowsPerWg) over the 65535/dim limit); this mirrors it.
      grid: (u) => splitWorkgroups2d(Math.ceil(u.n / rowsPerWg)),
      kernel(ctx) {
        const n = ctx.emitLet("n", ctx.uniform("n"));
        const k = ctx.emitLet("k", ctx.uniform("k"));
        const alpha = ctx.uniform("alpha", "f32");
        const tid = ctx.localIndex();
        // rowsPerWg lane-groups of groupSize threads; each group owns one
        // output row. Tail groups (row ≥ n) recompute the last row (keeps
        // the segmented reduce's barriers in uniform control flow) and are
        // masked at the store.
        const lane = ctx.emitLet("lane", tid.mod(ctx.u32(groupSize)));
        const row = ctx.emitLet(
          "row",
          ctx
            .rowIndex2d()
            .mul(ctx.u32(rowsPerWg))
            .add(tid.div(ctx.u32(groupSize))),
        );
        const rowC = ctx.emitLet("row_c", row.min(n.sub(ctx.u32(1))));
        const rowBase = ctx.emitLet("row_base", rowC.mul(k));
        const acc = ctx.emitVar("acc", "f32", ctx.f32(0));
        if (useVec4) {
          // vec4 storage loads: k % 4 === 0 (route-gated), so both the
          // a-vector and each B row are 4-element aligned.
          const kVec = ctx.emitLet("k_vec", k.shr(ctx.u32(2)));
          ctx.forStride(lane, kVec, groupSize, (i) => {
            const e4 = ctx.emitLet("e4", i.shl(ctx.u32(2)));
            acc.addAssign(
              ctx.loadVec4("a", e4).vec4Dot(ctx.loadVec4("b", rowBase.add(e4))),
            );
          });
        } else {
          ctx.forStride(lane, k, groupSize, (i) => {
            acc.addAssign(
              ctx
                .load("a", i)
                .toF32()
                .mul(ctx.load("b", rowBase.add(i)).toF32()),
            );
          });
        }
        const dot = ctx.wgReduceSegmented(
          "sum",
          acc.get(),
          tid,
          wgSize,
          rowsPerWg,
        );
        ctx.ifThen(lane.eq(ctx.u32(0)).and(row.lt(n)), () => {
          // "epilogue" seam: injected expression (bias/activation) applied
          // to the accumulated output value, addressed by output index.
          const val = ctx.applySeam("epilogue", dot.mul(alpha), {
            outIndex: row,
          });
          ctx.emitStore("out", row, outDtype === "f16" ? val.toF16() : val);
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
    seams,
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
          // kernel sums them and applies alpha. (No epilogue seam here —
          // partials must stay raw; the plan gates epilogue to splitK=1.)
          ctx.emitStore("out", split.mul(n).add(col), acc.get());
        } else {
          const val = ctx.applySeam("epilogue", acc.get().mul(alpha), {
            outIndex: col,
          });
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

/** The GEMV tunables: threads per workgroup and (NT) rows per workgroup.
 *  Single source for both the tile-IR AutotuneConfig below and the variant
 *  registry's candidates. */
export const GEMV_WG_SIZE_PARAM: TuneParam = {
  values: [64, 128, 256],
  default: GEMV_DEFAULT_WG_SIZE,
};

export const GEMV_NT_ROWS_PER_WG_PARAM: TuneParam = {
  values: [1, 2, 4, 8],
  default: GEMV_NT_DEFAULT_ROWS_PER_WG, // shape-aware: defaultGemvNtRowsPerWg
};

/** AutotuneConfig for a GEMV variant: tunables are wgSize and (NT only)
 *  rowsPerWg. The grid is a function of the uniforms + config (closed over
 *  per factory call), so autotuneTileKernel can dispatch candidates
 *  standalone. */
export function gemvAutotuneConfig(
  base: Omit<GemvKernelOptions, "wgSize" | "rowsPerWg">,
): AutotuneConfig {
  return {
    factory: (config) =>
      createGemvKernel({
        ...base,
        wgSize: config.wgSize,
        ...(base.mode === "nt" ? { rowsPerWg: config.rowsPerWg } : {}),
      }),
    params:
      base.mode === "nt"
        ? {
            wgSize: GEMV_WG_SIZE_PARAM,
            rowsPerWg: GEMV_NT_ROWS_PER_WG_PARAM,
          }
        : { wgSize: GEMV_WG_SIZE_PARAM },
    constraints:
      base.mode === "nt"
        ? [(config) => config.wgSize % config.rowsPerWg === 0]
        : undefined,
  };
}
