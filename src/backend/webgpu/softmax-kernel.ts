/**
 * Fused Softmax / Log-Softmax Kernels (tile-IR)
 *
 * Replaces the 5-op decomposition (max → sub → exp → sum → div) with a
 * single-dispatch kernel per reduction row. Each workgroup handles one row:
 *   1. Parallel max reduction in shared memory
 *   2. Parallel exp(x - max) + sum reduction
 *   3. Parallel divide by sum (softmax) or subtract log(sum) (log_softmax)
 *
 * Used by compound pattern recognition when the executor detects the
 * softmax/log_softmax decomposition pattern in the lazy IR plan.
 */

import { resolveOutputBuffer } from "./buffer-arena";
import type { GPUBuffer } from "./gpu-types";
import { WORKGROUP_SIZE } from "./shape-utils";
import {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./tile-dispatch";
import { perRowKernel, type TileKernelSpec } from "./tile-ir";
import { requireContext } from "./webgpu-state";

const WG = WORKGROUP_SIZE; // 256

// ============================================================================
// Softmax Kernel Spec
// ============================================================================

function softmaxSpec(isLog: boolean): TileKernelSpec {
  return perRowKernel({
    name: isLog ? "fusedLogSoftmax" : "fusedSoftmax",
    bindings: {
      x: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    dimUniform: "dim_size",

    kernel(ctx, _row, tid, D, base) {
      // Pass 1: Compute max along reduction dimension
      const maxVal = ctx.emitLet(
        "max_val",
        ctx.wgReduce("max", tid, D, WG, (i) => ctx.load("x", base.add(i))),
      );

      // Pass 2: Compute exp(x - max) and sum
      const sumExp = ctx.emitLet(
        "sum_exp",
        ctx.wgReduce("sum", tid, D, WG, (i) =>
          ctx.load("x", base.add(i)).sub(maxVal).exp(),
        ),
      );

      if (isLog) {
        const logSum = ctx.emitLet("log_sum", sumExp.log());
        ctx.stridedFor(tid, D, WG, (i) => {
          ctx.emitStore(
            "output",
            base.add(i),
            ctx.load("x", base.add(i)).sub(maxVal).sub(logSum),
          );
        });
      } else {
        ctx.stridedFor(tid, D, WG, (i) => {
          ctx.emitStore(
            "output",
            base.add(i),
            ctx.load("x", base.add(i)).sub(maxVal).exp().div(sumExp),
          );
        });
      }
    },
  });
}

// ============================================================================
// Cached Kernel Dispatchers
// ============================================================================

let softmaxKernel: TileKernelInstance | null = null;
let logSoftmaxKernel: TileKernelInstance | null = null;

function getSoftmaxKernel(): TileKernelInstance {
  if (!softmaxKernel)
    softmaxKernel = createTileKernelDispatcher(softmaxSpec(false));
  return softmaxKernel;
}

function getLogSoftmaxKernel(): TileKernelInstance {
  if (!logSoftmaxKernel)
    logSoftmaxKernel = createTileKernelDispatcher(softmaxSpec(true));
  return logSoftmaxKernel;
}

export function resetSoftmaxKernelState(): void {
  softmaxKernel?.reset();
  logSoftmaxKernel?.reset();
  softmaxKernel = null;
  logSoftmaxKernel = null;
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch a fused softmax kernel.
 *
 * @param inputBuffer  Input tensor buffer [N, D] (read-only)
 * @param numRows      Number of rows (product of dims before reduction dim)
 * @param dimSize      Size of the reduction dimension
 * @param outShape     Output tensor shape (same as input)
 * @param isLog        If true, compute log_softmax instead
 * @returns Output GPUBuffer
 */
export function dispatchFusedSoftmax(
  inputBuffer: GPUBuffer,
  numRows: number,
  dimSize: number,
  isLog: boolean,
): GPUBuffer {
  const ctx = requireContext();
  const outBuffer = resolveOutputBuffer(ctx.device, numRows * dimSize * 4, [
    inputBuffer,
  ]);

  const kernel = isLog ? getLogSoftmaxKernel() : getSoftmaxKernel();
  kernel.dispatch(
    { x: inputBuffer, output: outBuffer },
    { num_rows: numRows, dim_size: dimSize },
  );

  return outBuffer;
}
