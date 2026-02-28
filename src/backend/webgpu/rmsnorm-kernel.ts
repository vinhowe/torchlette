/**
 * Fused RMSNorm Kernel
 *
 * Single-dispatch forward kernel that replaces the decomposed RMSNorm:
 *   x² → mean(x², dim=-1, keepdim) → add(eps) → rsqrt → mul(x) → mul(weight)
 *
 * Uses 256-thread workgroup-level parallel reduction in shared memory.
 * Each workgroup handles one row (one sample/position in the batch).
 *
 * Forward:  x [N, D] + weight [D] → output [N, D]
 * Backward: decomposed in frontend (recompute from saved x + weight)
 */

import { allocateOutputBuffer } from "./index";
import { WORKGROUP_SIZE } from "./shape-utils";
import { type TileKernelSpec, perRowGrid } from "./tile-ir";
import { createTileKernelDispatcher } from "./tile-dispatch";

const WG = WORKGROUP_SIZE; // 256

// ============================================================================
// Tile IR Forward Kernel
// ============================================================================

const rmsNormFwdSpec: TileKernelSpec = {
  name: "rmsNormFwd",
  workgroupSize: WG,
  bindings: {
    x:      { storage: "read",       type: "f32" },
    weight: { storage: "read",       type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows:    "u32",
    feature_dim: "u32",
    eps:         "f32",
  },
  grid: perRowGrid(),

  kernel(ctx) {
    const row  = ctx.programId(0);
    const tid  = ctx.localIndex();
    const D    = ctx.uniform("feature_dim");
    const Df   = D.toF32();
    const base = row.mul(D);

    // Compute mean(x²) → inv_rms = rsqrt(mean(x²) + eps)
    const invRms = ctx.emitLet("inv_rms",
      ctx.wgReduce("sum", tid, D, WG, (i) => {
        const xi = ctx.load("x", base.add(i));
        return xi.mul(xi);
      }).div(Df).add(ctx.uniform("eps").toF32()).rsqrt());

    // Normalize + scale + store: output = x * inv_rms * weight
    ctx.stridedFor(tid, D, WG, (i) => {
      const out = ctx.load("x", base.add(i)).mul(invRms).mul(ctx.load("weight", i));
      ctx.emitStore("output", base.add(i), out);
    });
  },
};

const fwdTileKernel = createTileKernelDispatcher(rmsNormFwdSpec);

// ============================================================================
// Dispatch Function
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
 * Reset module-local mutable state (pipeline cache).
 */
export function resetRMSNormKernelState(): void {
  fwdTileKernel.reset();
}
