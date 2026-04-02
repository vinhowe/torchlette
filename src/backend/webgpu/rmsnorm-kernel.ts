/**
 * Fused RMSNorm Kernels
 *
 * Forward:  x [N, D] + weight [D] → output [N, D]
 * Backward gradX: grad [N, D] + x [N, D] + weight [D] → gradX [N, D]
 * Backward gradWeight: grad [N, D] + x [N, D] + weight [D] → gradWeight [D]
 *
 * Uses 256-thread workgroup-level parallel reduction in shared memory.
 * Each workgroup handles one row (one sample/position in the batch).
 */

import { allocateOutputBuffer } from "./buffer-arena";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { WORKGROUP_SIZE } from "./shape-utils";
import { createTileKernelDispatcher } from "./tile-dispatch";
import {
  ceilDivGrid,
  perRowKernel,
  type TileKernelSpec,
  tiledGrid,
} from "./tile-ir";
import { onTeardown, requireContext } from "./webgpu-state";

const WG = WORKGROUP_SIZE; // 256

// ============================================================================
// Tile IR Forward Kernel
// ============================================================================

const rmsNormFwdSpec = perRowKernel({
  name: "rmsNormFwd",
  bindings: {
    x: { storage: "read", type: "f32" },
    weight: { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { eps: "f32" },

  kernel(ctx, _row, tid, D, base) {
    const Df = D.toF32();

    // Compute mean(x²) → inv_rms = rsqrt(mean(x²) + eps)
    const invRms = ctx.emitLet(
      "inv_rms",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const xi = ctx.load("x", base.add(i));
          return xi.mul(xi);
        })
        .div(Df)
        .add(ctx.uniform("eps").toF32())
        .rsqrt(),
    );

    // Normalize + scale + store: output = x * inv_rms * weight
    ctx.stridedFor(tid, D, WG, (i) => {
      const out = ctx
        .load("x", base.add(i))
        .mul(invRms)
        .mul(ctx.load("weight", i));
      ctx.emitStore("output", base.add(i), out);
    });
  },
});

const fwdTileKernel = createTileKernelDispatcher(rmsNormFwdSpec);

// ============================================================================
// Tile IR Backward Kernels
// ============================================================================

/**
 * RMSNorm backward gradX kernel (tile-IR).
 * One workgroup per row. Recomputes inv_rms from x, then computes gradX.
 *
 * inv_rms = rsqrt(mean(x²) + eps)
 * normalized = x * inv_rms
 * c = mean(grad * weight * normalized, dim=-1)
 * gradX = inv_rms * (grad * weight - normalized * c)
 */
const rmsNormBackwardGradXSpec = perRowKernel({
  name: "rmsNormBwdGradX",
  bindings: {
    grad_output: { storage: "read", type: "f32" },
    x: { storage: "read", type: "f32" },
    weight: { storage: "read", type: "f32" },
    grad_x: { storage: "read_write", type: "f32" },
  },
  uniforms: { eps: "f32" },

  kernel(ctx, _row, tid, D, base) {
    const Df = D.toF32();

    // Recompute inv_rms = rsqrt(mean(x²) + eps)
    const invRms = ctx.emitLet(
      "inv_rms",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const xi = ctx.load("x", base.add(i));
          return xi.mul(xi);
        })
        .div(Df)
        .add(ctx.uniform("eps").toF32())
        .rsqrt(),
    );

    // c = mean(grad * weight * normalized, dim=-1)
    const c = ctx.emitLet(
      "c",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const gw = ctx
            .load("grad_output", base.add(i))
            .mul(ctx.load("weight", i));
          const normI = ctx.load("x", base.add(i)).mul(invRms);
          return gw.mul(normI);
        })
        .div(Df),
    );

    // gradX = inv_rms * (grad * weight - normalized * c)
    ctx.stridedFor(tid, D, WG, (i) => {
      const gw = ctx
        .load("grad_output", base.add(i))
        .mul(ctx.load("weight", i));
      const normI = ctx.load("x", base.add(i)).mul(invRms);
      ctx.emitStore("grad_x", base.add(i), gw.sub(normI.mul(c)).mul(invRms));
    });
  },
});

/**
 * RMSNorm row inv_rms kernel (tile-IR).
 * Computes per-row inv_rms = rsqrt(mean(x²) + eps) for use by gradWeight pass.
 */
const rmsNormRowStatsSpec = perRowKernel({
  name: "rmsNormRowStats",
  bindings: {
    x: { storage: "read", type: "f32" },
    row_inv_rms: { storage: "read_write", type: "f32" },
  },
  uniforms: { eps: "f32" },

  kernel(ctx, row, tid, D, base) {
    const Df = D.toF32();

    const invRms = ctx.emitLet(
      "inv_rms",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const xi = ctx.load("x", base.add(i));
          return xi.mul(xi);
        })
        .div(Df)
        .add(ctx.uniform("eps").toF32())
        .rsqrt(),
    );

    const isThread0 = tid.eq(ctx.u32(0));
    ctx.guardedStore("row_inv_rms", isThread0, row, invRms);
  },
});

/**
 * RMSNorm backward gradWeight: partial accumulation (tile-IR).
 * 2D grid: [ceil(D/WG), numRowTiles]. Each workgroup accumulates over a
 * tile of rows for WG features, writing partials to temp buffer.
 */
const ROWS_PER_TILE = 32;

const rmsNormBwdGradWPartialSpec: TileKernelSpec = {
  name: "rmsNormBwdGradWPartial",
  workgroupSize: WG,
  bindings: {
    grad_output: { storage: "read", type: "f32" },
    x: { storage: "read", type: "f32" },
    row_inv_rms: { storage: "read", type: "f32" },
    partial_gw: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows: "u32",
    feature_dim: "u32",
    num_row_tiles: "u32",
  },
  grid: tiledGrid({
    x: { uniform: "feature_dim", tileSize: WG },
    y: "num_row_tiles",
  }),

  kernel(ctx) {
    const featureIdx = ctx.globalId(0);
    const rowTileIdx = ctx.programId(1);
    const D = ctx.uniform("feature_dim");
    const N = ctx.uniform("num_rows");
    const RPT = ctx.u32(ROWS_PER_TILE);

    ctx.ifThen(featureIdx.ge(D), () => ctx.emitReturn());

    const rowStart = rowTileIdx.mul(RPT);
    const rowEnd = rowStart.add(RPT).min(N);

    const acc = ctx.emitVar("acc", "f32", ctx.f32(0.0));

    ctx.forRange(rowStart, rowEnd, (row) => {
      const base = row.mul(D);
      const go = ctx.load("grad_output", base.add(featureIdx));
      const normalized = ctx
        .load("x", base.add(featureIdx))
        .mul(ctx.load("row_inv_rms", row));
      acc.addAssign(go.mul(normalized));
    });

    const partialIdx = rowTileIdx.mul(D).add(featureIdx);
    ctx.emitStore("partial_gw", partialIdx, acc.get());
  },
};

/**
 * RMSNorm backward gradWeight: reduction (tile-IR).
 * 1D grid: [ceil(D/WG)]. Each thread sums partials across all row tiles.
 */
const rmsNormBwdGradWReduceSpec: TileKernelSpec = {
  name: "rmsNormBwdGradWReduce",
  workgroupSize: WG,
  bindings: {
    partial_gw: { storage: "read", type: "f32" },
    grad_weight: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    feature_dim: "u32",
    num_row_tiles: "u32",
  },
  grid: ceilDivGrid(WG, "feature_dim"),

  kernel(ctx) {
    const featureIdx = ctx.globalId(0);
    const D = ctx.uniform("feature_dim");
    const numTiles = ctx.uniform("num_row_tiles");

    ctx.ifThen(featureIdx.ge(D), () => ctx.emitReturn());

    const sumGW = ctx.emitVar("sum_gw", "f32", ctx.f32(0.0));

    ctx.forRange(ctx.u32(0), numTiles, (t) => {
      sumGW.addAssign(ctx.load("partial_gw", t.mul(D).add(featureIdx)));
    });

    ctx.emitStore("grad_weight", featureIdx, sumGW.get());
  },
};

const gradXTileKernel = createTileKernelDispatcher(rmsNormBackwardGradXSpec);
const rowStatsTileKernel = createTileKernelDispatcher(rmsNormRowStatsSpec);
const gradWPartialTileKernel = createTileKernelDispatcher(
  rmsNormBwdGradWPartialSpec,
);
const gradWReduceTileKernel = createTileKernelDispatcher(
  rmsNormBwdGradWReduceSpec,
);

// ============================================================================
// Temp Buffer Caches (persistent, reused across steps)
// ============================================================================

/** Row inv_rms stats: keyed by numRows. */
const rowStatsTempCache = new Map<number, GPUBuffer>();

function getOrCreateRowStatsTempBuffer(
  device: GPUDevice,
  numRows: number,
): GPUBuffer {
  let buf = rowStatsTempCache.get(numRows);
  if (!buf) {
    buf = device.createBuffer({
      size: numRows * 4, // f32 per row
      usage: GPUBufferUsage.STORAGE,
    });
    rowStatsTempCache.set(numRows, buf);
  }
  return buf;
}

/** GradW partial sums: keyed by `${numRowTiles}:${featureDim}`. */
const gradWPartialCache = new Map<string, GPUBuffer>();

function getOrCreateGradWPartial(
  device: GPUDevice,
  numRowTiles: number,
  featureDim: number,
): GPUBuffer {
  const key = `${numRowTiles}:${featureDim}`;
  let buf = gradWPartialCache.get(key);
  if (!buf) {
    buf = device.createBuffer({
      size: numRowTiles * featureDim * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    gradWPartialCache.set(key, buf);
  }
  return buf;
}

/** Reset all cached pipelines and persistent buffers (called by destroyWebGPU). */
export function resetRMSNormKernelState(): void {
  fwdTileKernel.reset();
  gradXTileKernel.reset();
  rowStatsTileKernel.reset();
  gradWPartialTileKernel.reset();
  gradWReduceTileKernel.reset();
  for (const buf of rowStatsTempCache.values()) buf.destroy();
  rowStatsTempCache.clear();
  for (const buf of gradWPartialCache.values()) buf.destroy();
  gradWPartialCache.clear();
}
onTeardown(resetRMSNormKernelState);

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch fused RMSNorm forward kernel.
 * x [N, D] + weight [D] → output [N, D]
 */
export function dispatchRMSNormForward(
  xBuffer: GPUBuffer,
  weightBuffer: GPUBuffer,
  numRows: number,
  featureDim: number,
  eps: number,
): GPUBuffer {
  const outputSizeBytes = numRows * featureDim * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  fwdTileKernel.dispatch(
    { x: xBuffer, weight: weightBuffer, output: outBuffer },
    { num_rows: numRows, feature_dim: featureDim, eps },
  );

  return outBuffer;
}

/**
 * Dispatch fused RMSNorm backward gradX kernel.
 * grad [N, D] + x [N, D] + weight [D] → gradX [N, D]
 */
export function dispatchRMSNormBackwardGradX(
  gradOutputBuffer: GPUBuffer,
  xBuffer: GPUBuffer,
  weightBuffer: GPUBuffer,
  numRows: number,
  featureDim: number,
  eps: number,
): GPUBuffer {
  const outputSizeBytes = numRows * featureDim * 4;
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  gradXTileKernel.dispatch(
    {
      grad_output: gradOutputBuffer,
      x: xBuffer,
      weight: weightBuffer,
      grad_x: outBuffer,
    },
    { num_rows: numRows, feature_dim: featureDim, eps },
  );

  return outBuffer;
}

/**
 * Dispatch fused RMSNorm backward gradWeight kernel.
 * Three-pass: row stats → 2D partial accumulation → reduce.
 * grad [N, D] + x [N, D] + weight [D] → gradWeight [D]
 */
export function dispatchRMSNormBackwardGradWeight(
  gradOutputBuffer: GPUBuffer,
  xBuffer: GPUBuffer,
  numRows: number,
  featureDim: number,
  eps: number,
): GPUBuffer {
  const ctx = requireContext();
  const device = ctx.device;

  // Pass 1: Compute per-row inv_rms
  const invRmsBuffer = getOrCreateRowStatsTempBuffer(device, numRows);

  rowStatsTileKernel.dispatch(
    { x: xBuffer, row_inv_rms: invRmsBuffer },
    { num_rows: numRows, feature_dim: featureDim, eps },
  );

  const numRowTiles = Math.ceil(numRows / ROWS_PER_TILE);

  // Pass 2: 2D partial accumulation
  const partialGW = getOrCreateGradWPartial(device, numRowTiles, featureDim);

  gradWPartialTileKernel.dispatch(
    {
      grad_output: gradOutputBuffer,
      x: xBuffer,
      row_inv_rms: invRmsBuffer,
      partial_gw: partialGW,
    },
    { num_rows: numRows, feature_dim: featureDim, num_row_tiles: numRowTiles },
  );

  // Pass 3: Reduce partials → final output
  const featureSizeBytes = featureDim * 4;
  const gradWeightBuffer = allocateOutputBuffer(featureSizeBytes);

  gradWReduceTileKernel.dispatch(
    {
      partial_gw: partialGW,
      grad_weight: gradWeightBuffer,
    },
    { feature_dim: featureDim, num_row_tiles: numRowTiles },
  );

  return gradWeightBuffer;
}
