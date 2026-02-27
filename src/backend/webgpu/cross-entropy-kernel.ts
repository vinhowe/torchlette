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
  dispatchComputePass,
  allocateOutputBuffer,
  cachedCreateBindGroup,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { WORKGROUP_SIZE } from "./shape-utils";
import { wgslSumReduction, wgslMaxReduction, trackBuffers } from "./wgsl-helpers";
import { type TileKernelSpec, perRowGrid } from "./tile-ir";
import { createTileKernelDispatcher } from "./tile-dispatch";

// Shared WGSL struct for cross-entropy config (8 bytes, 2 fields)
const WGSL_CE_CONFIG = `struct CEConfig {
  batch_size: u32,
  vocab_size: u32,
};`;

// ============================================================================
// WGSL Shaders
// ============================================================================

function crossEntropyForwardShader(): string {
  return `
${WGSL_CE_CONFIG}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
@group(0) @binding(3) var<uniform> config: CEConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let V = config.vocab_size;
  let base = row * V;

  // Phase 1: Parallel max reduction
  var local_max = -3.402823e+38;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_max = max(local_max, logits[base + i]);
  }
  sdata[tid] = local_max;
  workgroupBarrier();

  ${wgslMaxReduction("sdata", WORKGROUP_SIZE / 2)}
  let row_max = sdata[0];

  // Phase 2: Parallel sum-exp reduction
  var local_sum = 0.0;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_sum += exp(logits[base + i] - row_max);
  }
  sdata[tid] = local_sum;
  workgroupBarrier();

  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let log_sum_exp = log(sdata[0]);

  // Phase 3: Output (thread 0 only)
  if (tid == 0u) {
    let t = u32(targets[row]);
    loss[row] = -(logits[base + t] - row_max - log_sum_exp);
  }
}
`;
}

function crossEntropyBackwardShader(): string {
  return `
${WGSL_CE_CONFIG}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_logits: array<f32>;
@group(0) @binding(4) var<uniform> config: CEConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let V = config.vocab_size;
  let base = row * V;

  // Phase 1: Parallel max reduction (recompute from forward)
  var local_max = -3.402823e+38;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_max = max(local_max, logits[base + i]);
  }
  sdata[tid] = local_max;
  workgroupBarrier();

  ${wgslMaxReduction("sdata", WORKGROUP_SIZE / 2)}
  let row_max = sdata[0];

  // Phase 2: Parallel sum-exp reduction (recompute from forward)
  var local_sum = 0.0;
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    local_sum += exp(logits[base + i] - row_max);
  }
  sdata[tid] = local_sum;
  workgroupBarrier();

  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let log_sum_exp = log(sdata[0]);

  // Phase 3: Write gradients (all threads participate)
  // grad_logits[b,v] = grad_output[b] * (softmax[b,v] - one_hot(target[b])[v])
  let t = u32(targets[row]);
  let g = grad_output[row];
  for (var i = tid; i < V; i += ${WORKGROUP_SIZE}u) {
    let softmax_i = exp(logits[base + i] - row_max - log_sum_exp);
    let one_hot_i = select(0.0, 1.0, i == t);
    grad_logits[base + i] = g * (softmax_i - one_hot_i);
  }
}
`;
}

// ============================================================================
// Tile IR Kernels
// ============================================================================

const USE_TILE_IR_CROSSENTROPY = process.env.TORCHLETTE_TILE_CROSSENTROPY !== "0";

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

const ceFwdTileKernel = USE_TILE_IR_CROSSENTROPY
  ? createTileKernelDispatcher(crossEntropyForwardSpec) : null;
const ceBwdTileKernel = USE_TILE_IR_CROSSENTROPY
  ? createTileKernelDispatcher(crossEntropyBackwardSpec) : null;

// ============================================================================
// Config Buffer (cached per batch_size x vocab_size)
// ============================================================================

const configCache = new Map<string, GPUBuffer>();

function getOrCreateConfigBuffer(
  device: GPUDevice,
  batchSize: number,
  vocabSize: number,
): GPUBuffer {
  const key = `${batchSize}:${vocabSize}`;
  let buf = configCache.get(key);
  if (!buf) {
    buf = device.createBuffer({
      size: 8, // 2 x u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    configCache.set(key, buf);
  }
  const data = new Uint32Array([batchSize, vocabSize]);
  device.queue.writeBuffer(buf, 0, new Uint8Array(data.buffer));
  return buf;
}

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

  if (ceFwdTileKernel) {
    ceFwdTileKernel.dispatch(
      { logits: logitsBuffer, targets: targetsBuffer, loss: outBuffer },
      { batch_size: batchSize, vocab_size: vocabSize },
    );
    return outBuffer;
  }

  const ctx = requireContext();
  const device = ctx.device;

  const configBuf = getOrCreateConfigBuffer(device, batchSize, vocabSize);

  const pipeline = getPipeline(
    ctx,
    "crossEntropyFwd",
    crossEntropyForwardShader(),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [logitsBuffer, targetsBuffer, outBuffer, configBuf]);

  trackBuffers(logitsBuffer, targetsBuffer, outBuffer);

  dispatchComputePass(
    pipeline,
    bindGroup,
    batchSize,
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

  if (ceBwdTileKernel) {
    ceBwdTileKernel.dispatch(
      { logits: logitsBuffer, targets: targetsBuffer, grad_output: gradOutputBuffer,
        grad_logits: outBuffer },
      { batch_size: batchSize, vocab_size: vocabSize },
    );
    return outBuffer;
  }

  const ctx = requireContext();
  const device = ctx.device;

  const configBuf = getOrCreateConfigBuffer(device, batchSize, vocabSize);

  const pipeline = getPipeline(
    ctx,
    "crossEntropyBwd",
    crossEntropyBackwardShader(),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [logitsBuffer, targetsBuffer, gradOutputBuffer, outBuffer, configBuf]);

  trackBuffers(logitsBuffer, targetsBuffer, gradOutputBuffer, outBuffer);

  dispatchComputePass(
    pipeline,
    bindGroup,
    batchSize,
  );

  return outBuffer;
}

/**
 * Reset all module-local mutable state (pipeline cache, config buffer cache).
 */
export function resetCrossEntropyKernelState(): void {
  ceFwdTileKernel?.reset();
  ceBwdTileKernel?.reset();
  configCache.clear();
}
