/**
 * Fused Cross-Entropy Kernel
 *
 * Single-dispatch forward and backward kernels that replace the 9-op
 * decomposed cross-entropy (max, sub, exp, sum, log, sub, gather, neg, mean).
 *
 * Uses 256-thread workgroup-level parallel reduction in shared memory.
 * Each workgroup handles one row (one sample in the batch).
 *
 * Forward:  logits [B, V] + targets [B] → loss [B]
 * Backward: logits [B, V] + targets [B] + grad_output [B] → grad_logits [B, V]
 */

import { allocateOutputBuffer } from "./buffer-arena";
import type { GPUBuffer } from "./gpu-types";
import { WORKGROUP_SIZE } from "./shape-utils";
import { createTileKernelDispatcher } from "./tile-dispatch";
import { perRowKernel } from "./tile-ir";
import { onTeardown } from "./webgpu-state";

// ============================================================================
// Tile IR Kernels
// ============================================================================

const WG = WORKGROUP_SIZE; // 256

/**
 * Cross-entropy forward kernel (tile-IR).
 * One workgroup per batch row: max reduction, sum-exp reduction, thread 0 writes loss.
 *
 * Targets are read as native i32 (PyTorch convention). ignore_index is an i32
 * uniform so negative sentinels (e.g. -100, -1) are handled without any f32→u32
 * undefined-behavior conversions.
 */
const crossEntropyForwardSpec = perRowKernel({
  name: "ceFwd",
  bindings: {
    logits: { storage: "read", type: "f32" },
    targets: { storage: "read", type: "i32" },
    loss: { storage: "read_write", type: "f32" },
  },
  rowUniform: "batch_size",
  dimUniform: "vocab_size",
  uniforms: { ignore_index: "i32" },

  kernel(ctx, row, tid, V, base) {
    const tI = ctx.emitLet("t_i", ctx.load("targets", row));
    const ignoreIdx = ctx.uniform("ignore_index", "i32");
    const isIgnored = ctx.emitLet("is_ignored", tI.eq(ignoreIdx));

    // Use tSafe=0 for ignored rows so the OOB read below doesn't happen.
    // All threads participate in the reductions (avoids uniformity violation
    // when barriers/subgroups follow an early return).
    const tSafe = ctx.emitLet(
      "t_safe",
      isIgnored.select(ctx.u32(0), tI.toU32()),
    );

    // Parallel max reduction (runs for all rows)
    const rowMax = ctx.emitLet(
      "row_max",
      ctx.wgReduce("max", tid, V, WG, (i) => ctx.load("logits", base.add(i))),
    );

    // Parallel sum-exp reduction
    const logSumExp = ctx.emitLet(
      "log_sum_exp",
      ctx
        .wgReduce("sum", tid, V, WG, (i) =>
          ctx.load("logits", base.add(i)).sub(rowMax).exp(),
        )
        .log(),
    );

    // Thread 0 writes final loss: 0 if ignored, else -log p(target).
    const rawLoss = ctx
      .load("logits", base.add(tSafe))
      .sub(rowMax)
      .sub(logSumExp)
      .neg();
    ctx.guardedStore(
      "loss",
      tid.eq(ctx.u32(0)),
      row,
      isIgnored.select(ctx.f32(0.0), rawLoss),
    );
  },
});

/**
 * Cross-entropy backward kernel (tile-IR).
 * One workgroup per batch row: recomputes max + sum-exp, writes softmax gradients.
 */
const crossEntropyBackwardSpec = perRowKernel({
  name: "ceBwd",
  bindings: {
    logits: { storage: "read", type: "f32" },
    targets: { storage: "read", type: "i32" },
    grad_output: { storage: "read", type: "f32" },
    grad_logits: { storage: "read_write", type: "f32" },
  },
  rowUniform: "batch_size",
  dimUniform: "vocab_size",
  uniforms: { ignore_index: "i32" },

  kernel(ctx, row, tid, V, base) {
    const tI = ctx.emitLet("t_i", ctx.load("targets", row));
    const ignoreIdx = ctx.uniform("ignore_index", "i32");
    const isIgnored = ctx.emitLet("is_ignored", tI.eq(ignoreIdx));
    // Clamp to 0 so the one-hot i.eq(t) never spuriously matches at a
    // negative/OOB index; gradients are zeroed out via select() below.
    const t = ctx.emitLet("t", isIgnored.select(ctx.u32(0), tI.toU32()));

    // Parallel max reduction
    const rowMax = ctx.emitLet(
      "row_max",
      ctx.wgReduce("max", tid, V, WG, (i) => ctx.load("logits", base.add(i))),
    );

    // Parallel sum-exp reduction
    const logSumExp = ctx.emitLet(
      "log_sum_exp",
      ctx
        .wgReduce("sum", tid, V, WG, (i) =>
          ctx.load("logits", base.add(i)).sub(rowMax).exp(),
        )
        .log(),
    );

    // Write gradients: zero for ignored rows, else grad_output * (softmax - one_hot).
    const g = ctx.emitLet("g", ctx.load("grad_output", row));
    const gScaled = ctx.emitLet(
      "g_scaled",
      isIgnored.select(ctx.f32(0.0), g),
    );
    ctx.stridedFor(tid, V, WG, (i) => {
      const softmaxI = ctx
        .load("logits", base.add(i))
        .sub(rowMax)
        .sub(logSumExp)
        .exp();
      const oneHotI = i.eq(t).select(ctx.f32(1.0), ctx.f32(0.0));
      ctx.emitStore(
        "grad_logits",
        base.add(i),
        gScaled.mul(softmaxI.sub(oneHotI)),
      );
    });
  },
});

const ceFwdTileKernel = createTileKernelDispatcher(crossEntropyForwardSpec);
const ceBwdTileKernel = createTileKernelDispatcher(crossEntropyBackwardSpec);

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch fused cross-entropy forward kernel.
 * logits [B, V] + targets [B] → loss [B]
 */
export function dispatchCrossEntropyForward(
  logitsBuffer: GPUBuffer,
  targetsBuffer: GPUBuffer,
  batchSize: number,
  vocabSize: number,
  ignoreIndex = -100,
): GPUBuffer {
  const outputSizeBytes = batchSize * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  ceFwdTileKernel.dispatch(
    { logits: logitsBuffer, targets: targetsBuffer, loss: outBuffer },
    { batch_size: batchSize, vocab_size: vocabSize, ignore_index: ignoreIndex },
  );

  return outBuffer;
}

/**
 * Dispatch fused cross-entropy backward kernel.
 * logits [B, V] + targets [B] + grad_output [B] → grad_logits [B, V]
 */
export function dispatchCrossEntropyBackward(
  logitsBuffer: GPUBuffer,
  targetsBuffer: GPUBuffer,
  gradOutputBuffer: GPUBuffer,
  batchSize: number,
  vocabSize: number,
  ignoreIndex = -100,
): GPUBuffer {
  const outputSizeBytes = batchSize * vocabSize * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  ceBwdTileKernel.dispatch(
    {
      logits: logitsBuffer,
      targets: targetsBuffer,
      grad_output: gradOutputBuffer,
      grad_logits: outBuffer,
    },
    { batch_size: batchSize, vocab_size: vocabSize, ignore_index: ignoreIndex },
  );

  return outBuffer;
}

/**
 * Stage-4 plan/encode: forward CE dispatch plan, derived from the SAME
 * module-level cached dispatcher dispatchCrossEntropyForward() uses (shared
 * config cache → shared config-buffer identity for stream generation).
 */
export function planCrossEntropyForwardDispatch(
  batchSize: number,
  vocabSize: number,
  ignoreIndex = -100,
): import("./tile-dispatch").TileKernelPlan {
  return ceFwdTileKernel.plan({
    batch_size: batchSize,
    vocab_size: vocabSize,
    ignore_index: ignoreIndex,
  });
}

/** Stage-4 plan/encode: backward CE dispatch plan (same cached dispatcher). */
export function planCrossEntropyBackwardDispatch(
  batchSize: number,
  vocabSize: number,
  ignoreIndex = -100,
): import("./tile-dispatch").TileKernelPlan {
  return ceBwdTileKernel.plan({
    batch_size: batchSize,
    vocab_size: vocabSize,
    ignore_index: ignoreIndex,
  });
}

/**
 * Reset all module-local mutable state (pipeline cache, config buffer cache).
 */
export function resetCrossEntropyKernelState(): void {
  ceFwdTileKernel.reset();
  ceBwdTileKernel.reset();
}
onTeardown(resetCrossEntropyKernelState);
