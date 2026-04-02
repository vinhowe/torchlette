/**
 * Fused LayerNorm Kernel
 *
 * Single-dispatch forward and backward kernels that replace the 9-op
 * decomposed LayerNorm (mean, sub, square, mean, add, sqrt, div, mul, add).
 *
 * Uses 256-thread workgroup-level parallel reduction in shared memory.
 * Each workgroup handles one row (one sample/position in the batch).
 *
 * Forward:  x [N, D] + weight [D] + bias [D] → output [N, D]
 * Backward: grad [N, D] + x [N, D] + weight [D] → gradX [N, D] + gradWeight [D] + gradBias [D]
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

// ============================================================================
// Temp Buffer Caches (persistent, reused across steps)
// ============================================================================

/** Row stats (mean, inv_std): keyed by numRows. */
const rowStatsTempCache = new Map<
  number,
  { meanBuffer: GPUBuffer; invStdBuffer: GPUBuffer }
>();

function getOrCreateRowStatsTempBuffers(
  device: GPUDevice,
  numRows: number,
): { meanBuffer: GPUBuffer; invStdBuffer: GPUBuffer } {
  let entry = rowStatsTempCache.get(numRows);
  if (!entry) {
    const size = numRows * 4; // f32 per row
    entry = {
      meanBuffer: device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE,
      }),
      invStdBuffer: device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE,
      }),
    };
    rowStatsTempCache.set(numRows, entry);
  }
  return entry;
}

/** GradWB partial sums: keyed by `${numRowTiles}:${featureDim}`. */
const gradWBPartialCache = new Map<
  string,
  { partialGW: GPUBuffer; partialGB: GPUBuffer }
>();

function getOrCreateGradWBPartials(
  device: GPUDevice,
  numRowTiles: number,
  featureDim: number,
): { partialGW: GPUBuffer; partialGB: GPUBuffer } {
  const key = `${numRowTiles}:${featureDim}`;
  let entry = gradWBPartialCache.get(key);
  if (!entry) {
    const size = numRowTiles * featureDim * 4; // f32
    entry = {
      partialGW: device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE,
      }),
      partialGB: device.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE,
      }),
    };
    gradWBPartialCache.set(key, entry);
  }
  return entry;
}

// ============================================================================
// Tile IR Forward Kernel
// ============================================================================

const WG = WORKGROUP_SIZE; // 256

const layerNormFwdSpec = perRowKernel({
  name: "layerNormFwd",
  bindings: {
    x: { storage: "read", type: "f32" },
    weight: { storage: "read", type: "f32" },
    bias: { storage: "read", type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: { eps: "f32" },

  kernel(ctx, _row, tid, D, base) {
    const Df = D.toF32();

    // Compute mean
    const mean = ctx.emitLet(
      "mean",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => ctx.load("x", base.add(i)))
        .div(Df),
    );

    // Compute variance → inv_std
    const invStd = ctx.emitLet(
      "inv_std",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const diff = ctx.load("x", base.add(i)).sub(mean);
          return diff.mul(diff);
        })
        .div(Df)
        .add(ctx.uniform("eps").toF32())
        .rsqrt(),
    );

    // Normalize + affine + store
    ctx.stridedFor(tid, D, WG, (i) => {
      const normI = ctx.load("x", base.add(i)).sub(mean).mul(invStd);
      const out = normI.mul(ctx.load("weight", i)).add(ctx.load("bias", i));
      ctx.emitStore("output", base.add(i), out);
    });
  },
});

const fwdTileKernel = createTileKernelDispatcher(layerNormFwdSpec);

// ============================================================================
// Tile IR Backward Kernels
// ============================================================================

/**
 * LayerNorm backward gradX kernel (tile-IR).
 * One workgroup per row. Recomputes mean/variance in shared memory,
 * then computes gradX with dual reduction coefficients.
 */
const layerNormBackwardGradXSpec = perRowKernel({
  name: "lnBwdGradX",
  bindings: {
    grad_output: { storage: "read", type: "f32" },
    x: { storage: "read", type: "f32" },
    weight: { storage: "read", type: "f32" },
    grad_x: { storage: "read_write", type: "f32" },
  },
  uniforms: { eps: "f32" },

  kernel(ctx, _row, tid, D, base) {
    const Df = D.toF32();

    // Recompute mean
    const mean = ctx.emitLet(
      "mean",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => ctx.load("x", base.add(i)))
        .div(Df),
    );

    // Recompute variance → inv_std
    const invStd = ctx.emitLet(
      "inv_std",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const diff = ctx.load("x", base.add(i)).sub(mean);
          return diff.mul(diff);
        })
        .div(Df)
        .add(ctx.uniform("eps").toF32())
        .rsqrt(),
    );

    // Dual reduction for c1, c2
    const [sumC1, sumC2] = ctx.dualWgReduce(tid, D, WG, (i) => {
      const gn = ctx
        .load("grad_output", base.add(i))
        .mul(ctx.load("weight", i));
      const normI = ctx.load("x", base.add(i)).sub(mean).mul(invStd);
      return [gn, gn.mul(normI)];
    });
    const c1 = ctx.emitLet("c1", sumC1.div(Df));
    const c2 = ctx.emitLet("c2", sumC2.div(Df));

    // Output gradX
    ctx.stridedFor(tid, D, WG, (i) => {
      const gn = ctx
        .load("grad_output", base.add(i))
        .mul(ctx.load("weight", i));
      const normI = ctx.load("x", base.add(i)).sub(mean).mul(invStd);
      ctx.emitStore(
        "grad_x",
        base.add(i),
        gn.sub(c1).sub(normI.mul(c2)).mul(invStd),
      );
    });
  },
});

/**
 * LayerNorm row stats kernel (tile-IR).
 * Computes per-row mean and inv_std for use by gradWeight/gradBias pass.
 */
const layerNormRowStatsSpec = perRowKernel({
  name: "lnRowStats",
  bindings: {
    x: { storage: "read", type: "f32" },
    row_mean: { storage: "read_write", type: "f32" },
    row_inv_std: { storage: "read_write", type: "f32" },
  },
  uniforms: { eps: "f32" },

  kernel(ctx, row, tid, D, base) {
    const Df = D.toF32();

    // Mean
    const mean = ctx.emitLet(
      "mean",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => ctx.load("x", base.add(i)))
        .div(Df),
    );

    // Variance → inv_std
    const invStd = ctx.emitLet(
      "inv_std",
      ctx
        .wgReduce("sum", tid, D, WG, (i) => {
          const diff = ctx.load("x", base.add(i)).sub(mean);
          return diff.mul(diff);
        })
        .div(Df)
        .add(ctx.uniform("eps").toF32())
        .rsqrt(),
    );

    // Thread 0 writes row stats
    const isThread0 = tid.eq(ctx.u32(0));
    ctx.guardedStore("row_mean", isThread0, row, mean);
    ctx.guardedStore("row_inv_std", isThread0, row, invStd);
  },
});

/**
 * LayerNorm backward gradWeight + gradBias: partial accumulation (tile-IR).
 * 2D grid: [ceil(D/WG), numRowTiles]. Each workgroup accumulates over a
 * tile of rows for WG features, writing partials to temp buffers.
 */
const ROWS_PER_TILE = 32;

const lnBwdGradWBPartialSpec: TileKernelSpec = {
  name: "lnBwdGradWBPartial",
  workgroupSize: WG,
  bindings: {
    grad_output: { storage: "read", type: "f32" },
    x: { storage: "read", type: "f32" },
    row_mean: { storage: "read", type: "f32" },
    row_inv_std: { storage: "read", type: "f32" },
    partial_gw: { storage: "read_write", type: "f32" },
    partial_gb: { storage: "read_write", type: "f32" },
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

    const accGW = ctx.emitVar("acc_gw", "f32", ctx.f32(0.0));
    const accGB = ctx.emitVar("acc_gb", "f32", ctx.f32(0.0));

    ctx.forRange(rowStart, rowEnd, (row) => {
      const base = row.mul(D);
      const go = ctx.load("grad_output", base.add(featureIdx));
      const normalized = ctx
        .load("x", base.add(featureIdx))
        .sub(ctx.load("row_mean", row))
        .mul(ctx.load("row_inv_std", row));
      accGW.addAssign(go.mul(normalized));
      accGB.addAssign(go);
    });

    const partialIdx = rowTileIdx.mul(D).add(featureIdx);
    ctx.emitStore("partial_gw", partialIdx, accGW.get());
    ctx.emitStore("partial_gb", partialIdx, accGB.get());
  },
};

/**
 * LayerNorm backward gradWeight + gradBias: reduction (tile-IR).
 * 1D grid: [ceil(D/WG)]. Each thread sums partials across all row tiles.
 */
const lnBwdGradWBReduceSpec: TileKernelSpec = {
  name: "lnBwdGradWBReduce",
  workgroupSize: WG,
  bindings: {
    partial_gw: { storage: "read", type: "f32" },
    partial_gb: { storage: "read", type: "f32" },
    grad_weight: { storage: "read_write", type: "f32" },
    grad_bias: { storage: "read_write", type: "f32" },
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
    const sumGB = ctx.emitVar("sum_gb", "f32", ctx.f32(0.0));

    ctx.forRange(ctx.u32(0), numTiles, (t) => {
      const idx = t.mul(D).add(featureIdx);
      sumGW.addAssign(ctx.load("partial_gw", idx));
      sumGB.addAssign(ctx.load("partial_gb", idx));
    });

    ctx.emitStore("grad_weight", featureIdx, sumGW.get());
    ctx.emitStore("grad_bias", featureIdx, sumGB.get());
  },
};

const gradXTileKernel = createTileKernelDispatcher(layerNormBackwardGradXSpec);
const rowStatsTileKernel = createTileKernelDispatcher(layerNormRowStatsSpec);
const gradWBPartialTileKernel = createTileKernelDispatcher(
  lnBwdGradWBPartialSpec,
);
const gradWBReduceTileKernel = createTileKernelDispatcher(
  lnBwdGradWBReduceSpec,
);

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch fused LayerNorm forward kernel.
 * x [N, D] + weight [D] + bias [D] → output [N, D]
 *
 * Uses the tile IR compiler to generate the WGSL shader.
 */
export function dispatchLayerNormForward(
  xBuffer: GPUBuffer,
  weightBuffer: GPUBuffer,
  biasBuffer: GPUBuffer,
  numRows: number,
  featureDim: number,
  eps: number,
): GPUBuffer {
  const outputSizeBytes = numRows * featureDim * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  fwdTileKernel.dispatch(
    { x: xBuffer, weight: weightBuffer, bias: biasBuffer, output: outBuffer },
    { num_rows: numRows, feature_dim: featureDim, eps },
  );

  return outBuffer;
}

/**
 * Dispatch fused LayerNorm backward gradX kernel.
 * grad_output [N, D] + x [N, D] + weight [D] → grad_x [N, D]
 */
export function dispatchLayerNormBackwardGradX(
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
 * Dispatch fused LayerNorm backward gradWeight + gradBias kernel.
 * Three-pass approach:
 *   Pass 1: Compute row stats (mean, inv_std) for all N rows
 *   Pass 2: 2D partial accumulation — each workgroup handles ROWS_PER_TILE rows
 *           for WG features, writing partials to temp buffers
 *   Pass 3: Reduce partials across row tiles → final gradWeight/gradBias
 *
 * grad_output [N, D] + x [N, D] → grad_weight [D] + grad_bias [D]
 */
export function dispatchLayerNormBackwardGradWeightBias(
  gradOutputBuffer: GPUBuffer,
  xBuffer: GPUBuffer,
  numRows: number,
  featureDim: number,
  eps: number,
): { gradWeightBuffer: GPUBuffer; gradBiasBuffer: GPUBuffer } {
  const ctx = requireContext();
  const device = ctx.device;

  // Pass 1: Compute row stats (mean[N], inv_std[N])
  const { meanBuffer, invStdBuffer } = getOrCreateRowStatsTempBuffers(
    device,
    numRows,
  );

  rowStatsTileKernel.dispatch(
    { x: xBuffer, row_mean: meanBuffer, row_inv_std: invStdBuffer },
    { num_rows: numRows, feature_dim: featureDim, eps },
  );

  const numRowTiles = Math.ceil(numRows / ROWS_PER_TILE);

  // Pass 2: 2D partial accumulation
  const { partialGW, partialGB } = getOrCreateGradWBPartials(
    device,
    numRowTiles,
    featureDim,
  );

  gradWBPartialTileKernel.dispatch(
    {
      grad_output: gradOutputBuffer,
      x: xBuffer,
      row_mean: meanBuffer,
      row_inv_std: invStdBuffer,
      partial_gw: partialGW,
      partial_gb: partialGB,
    },
    { num_rows: numRows, feature_dim: featureDim, num_row_tiles: numRowTiles },
  );

  // Pass 3: Reduce partials → final output
  const featureSizeBytes = featureDim * 4;
  const gradWeightBuffer = allocateOutputBuffer(featureSizeBytes);
  const gradBiasBuffer = allocateOutputBuffer(featureSizeBytes);

  gradWBReduceTileKernel.dispatch(
    {
      partial_gw: partialGW,
      partial_gb: partialGB,
      grad_weight: gradWeightBuffer,
      grad_bias: gradBiasBuffer,
    },
    { feature_dim: featureDim, num_row_tiles: numRowTiles },
  );

  return { gradWeightBuffer, gradBiasBuffer };
}

/**
 * Reset all module-local mutable state (pipeline cache, row stats temp buffers).
 */
export function resetLayerNormKernelState(): void {
  fwdTileKernel.reset();
  gradXTileKernel.reset();
  rowStatsTileKernel.reset();
  gradWBPartialTileKernel.reset();
  gradWBReduceTileKernel.reset();
  for (const entry of rowStatsTempCache.values()) {
    entry.meanBuffer.destroy();
    entry.invStdBuffer.destroy();
  }
  rowStatsTempCache.clear();
  for (const entry of gradWBPartialCache.values()) {
    entry.partialGW.destroy();
    entry.partialGB.destroy();
  }
  gradWBPartialCache.clear();
}
onTeardown(resetLayerNormKernelState);
