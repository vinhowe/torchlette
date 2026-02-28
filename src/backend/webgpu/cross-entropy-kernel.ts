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

import {
  allocateOutputBuffer,
} from "./index";
import type { GPUBuffer } from "./gpu-types";
import { WORKGROUP_SIZE } from "./shape-utils";
import { type TileKernelSpec, perRowGrid } from "./tile-ir";
import { createTileKernelDispatcher } from "./tile-dispatch";

// ============================================================================
// Tile IR Kernels
// ============================================================================

const WG = WORKGROUP_SIZE; // 256

/**
 * Cross-entropy forward kernel (tile-IR).
 * One workgroup per batch row: max reduction, sum-exp reduction, thread 0 writes loss.
 */
const crossEntropyForwardSpec: TileKernelSpec = {
  name: "ceFwd",
  workgroupSize: WG,
  bindings: {
    logits:  { storage: "read", type: "f32" },
    targets: { storage: "read", type: "f32" },
    loss:    { storage: "read_write", type: "f32" },
  },
  uniforms: {
    batch_size: "u32",
    vocab_size: "u32",
  },
  grid: perRowGrid("batch_size"),

  kernel(ctx) {
    const tid = ctx.localIndex();
    const row = ctx.programId(0);
    const V = ctx.uniform("vocab_size");
    const base = row.mul(V);

    // Parallel max reduction
    const rowMax = ctx.emitLet("row_max",
      ctx.wgReduce("max", tid, V, WG, (i) => ctx.load("logits", base.add(i))));

    // Parallel sum-exp reduction
    const logSumExp = ctx.emitLet("log_sum_exp",
      ctx.wgReduce("sum", tid, V, WG,
        (i) => ctx.load("logits", base.add(i)).sub(rowMax).exp()).log());

    // Output (thread 0 only)
    const t = ctx.emitLet("t", ctx.load("targets", row).toU32());
    ctx.guardedStore("loss", tid.eq(ctx.u32(0)), row,
      ctx.load("logits", base.add(t)).sub(rowMax).sub(logSumExp).neg());
  },
};

/**
 * Cross-entropy backward kernel (tile-IR).
 * One workgroup per batch row: recomputes max + sum-exp, writes softmax gradients.
 */
const crossEntropyBackwardSpec: TileKernelSpec = {
  name: "ceBwd",
  workgroupSize: WG,
  bindings: {
    logits:      { storage: "read", type: "f32" },
    targets:     { storage: "read", type: "f32" },
    grad_output: { storage: "read", type: "f32" },
    grad_logits: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    batch_size: "u32",
    vocab_size: "u32",
  },
  grid: perRowGrid("batch_size"),

  kernel(ctx) {
    const tid = ctx.localIndex();
    const row = ctx.programId(0);
    const V = ctx.uniform("vocab_size");
    const base = row.mul(V);

    // Parallel max reduction
    const rowMax = ctx.emitLet("row_max",
      ctx.wgReduce("max", tid, V, WG, (i) => ctx.load("logits", base.add(i))));

    // Parallel sum-exp reduction
    const logSumExp = ctx.emitLet("log_sum_exp",
      ctx.wgReduce("sum", tid, V, WG,
        (i) => ctx.load("logits", base.add(i)).sub(rowMax).exp()).log());

    // Write gradients
    const t = ctx.emitLet("t", ctx.load("targets", row).toU32());
    const g = ctx.emitLet("g", ctx.load("grad_output", row));
    ctx.stridedFor(tid, V, WG, (i) => {
      const softmaxI = ctx.load("logits", base.add(i)).sub(rowMax).sub(logSumExp).exp();
      const oneHotI = i.eq(t).select(ctx.f32(1.0), ctx.f32(0.0));
      ctx.emitStore("grad_logits", base.add(i),
        g.mul(softmaxI.sub(oneHotI)));
    });
  },
};

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
): GPUBuffer {
  const outputSizeBytes = batchSize * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  ceFwdTileKernel.dispatch(
    { logits: logitsBuffer, targets: targetsBuffer, loss: outBuffer },
    { batch_size: batchSize, vocab_size: vocabSize },
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
): GPUBuffer {
  const outputSizeBytes = batchSize * vocabSize * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  ceBwdTileKernel.dispatch(
    { logits: logitsBuffer, targets: targetsBuffer, grad_output: gradOutputBuffer,
      grad_logits: outBuffer },
    { batch_size: batchSize, vocab_size: vocabSize },
  );

  return outBuffer;
}

/**
 * Reset all module-local mutable state (pipeline cache, config buffer cache).
 */
export function resetCrossEntropyKernelState(): void {
  ceFwdTileKernel.reset();
  ceBwdTileKernel.reset();
}
