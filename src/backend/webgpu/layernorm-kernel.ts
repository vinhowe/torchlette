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

import {
  dispatchComputePass,
  getWebGPUDevice,
  trackSharedEncoderWrite,
  allocateOutputBuffer,
  cachedCreateBindGroup,
} from "./index";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { defineKernel } from "./kernel-factory";
import { wgslReduce } from "./wgsl-reduce";

const kernel = defineKernel("layerNorm");

const WORKGROUP_SIZE = 256;

// ============================================================================
// Row Stats Temp Buffer Cache (persistent, keyed by numRows)
// ============================================================================

const rowStatsTempCache = new Map<number, { meanBuffer: GPUBuffer; invStdBuffer: GPUBuffer }>();

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

// ============================================================================
// WGSL Shaders
// ============================================================================

function layerNormForwardShader(): string {
  return `
struct LNConfig {
  num_rows: u32,
  feature_dim: u32,
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> config: LNConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let D = config.feature_dim;
  let base = row * D;

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  smem: "sdata", init: "0.0",
  accumExpr: "x[base + i]", result: "mean",
  transform: "_ / f32(D)",
})}

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  smem: "sdata", init: "0.0",
  loopPreamble: "let diff = x[base + i] - mean;",
  accumExpr: "diff * diff", result: "inv_std",
  transform: "inverseSqrt(_ / f32(D) + config.eps)",
})}

  // Normalize + affine transform
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let normalized = (x[base + i] - mean) * inv_std;
    output[base + i] = normalized * weight[i] + bias[i];
  }
}
`;
}

function layerNormBackwardGradXShader(): string {
  return `
struct LNConfig {
  num_rows: u32,
  feature_dim: u32,
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_x: array<f32>;
@group(0) @binding(4) var<uniform> config: LNConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> sdata2: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let D = config.feature_dim;
  let Df = f32(D);
  let base = row * D;

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  smem: "sdata", init: "0.0",
  accumExpr: "x[base + i]", result: "mean",
  transform: "_ / Df",
})}

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  smem: "sdata", init: "0.0",
  loopPreamble: "let diff = x[base + i] - mean;",
  accumExpr: "diff * diff", result: "inv_std",
  transform: "inverseSqrt(_ / Df + config.eps)",
})}

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  loopPreamble: "let gn = grad_output[base + i] * weight[i];\nlet norm_i = (x[base + i] - mean) * inv_std;",
  channels: [
    { smem: "sdata", init: "0.0", accumExpr: "gn", result: "c1", transform: "_ / Df" },
    { smem: "sdata2", init: "0.0", accumExpr: "gn * norm_i", result: "c2", transform: "_ / Df" },
  ],
})}

  // Output: grad_x[i] = (grad_normalized[i] - c1 - normalized[i] * c2) * inv_std
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let gn = grad_output[base + i] * weight[i];
    let norm_i = (x[base + i] - mean) * inv_std;
    grad_x[base + i] = (gn - c1 - norm_i * c2) * inv_std;
  }
}
`;
}

function layerNormRowStatsShader(): string {
  return `
struct LNConfig {
  num_rows: u32,
  feature_dim: u32,
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> row_mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> row_inv_std: array<f32>;
@group(0) @binding(3) var<uniform> config: LNConfig;

var<workgroup> sdata: array<f32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row = wid.x;
  let tid = lid.x;
  let D = config.feature_dim;
  let Df = f32(D);
  let base = row * D;

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  smem: "sdata", init: "0.0",
  accumExpr: "x[base + i]", result: "mean",
  transform: "_ / Df",
})}

${wgslReduce({ wgSize: WORKGROUP_SIZE, tid: "tid", dim: "D", op: "sum",
  smem: "sdata", init: "0.0",
  loopPreamble: "let diff = x[base + i] - mean;",
  accumExpr: "diff * diff", result: "inv_std",
  transform: "inverseSqrt(_ / Df + config.eps)",
})}

  // Write row stats (only thread 0 writes)
  if (tid == 0u) {
    row_mean[row] = mean;
    row_inv_std[row] = inv_std;
  }
}
`;
}

function layerNormBackwardGradWeightBiasShader(): string {
  return `
struct LNConfig {
  num_rows: u32,
  feature_dim: u32,
  eps: f32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> row_mean: array<f32>;
@group(0) @binding(3) var<storage, read> row_inv_std: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_bias: array<f32>;
@group(0) @binding(6) var<uniform> config: LNConfig;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let feature_idx = wid.x * ${WORKGROUP_SIZE}u + lid.x;
  let D = config.feature_dim;
  let N = config.num_rows;

  if (feature_idx >= D) { return; }

  var acc_gw = 0.0;
  var acc_gb = 0.0;
  for (var row = 0u; row < N; row++) {
    let base = row * D;
    let go = grad_output[base + feature_idx];
    let normalized = (x[base + feature_idx] - row_mean[row]) * row_inv_std[row];
    acc_gw += go * normalized;
    acc_gb += go;
  }
  grad_weight[feature_idx] = acc_gw;
  grad_bias[feature_idx] = acc_gb;
}
`;
}

// Reusable typed arrays for config data packing
const configArrayBuffer = new ArrayBuffer(16);
const configU32View = new Uint32Array(configArrayBuffer);
const configF32View = new Float32Array(configArrayBuffer);
const configU8View = new Uint8Array(configArrayBuffer);

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch fused LayerNorm forward kernel.
 * x [N, D] + weight [D] + bias [D] → output [N, D]
 */
export function dispatchLayerNormForward(
  xBuffer: GPUBuffer,
  weightBuffer: GPUBuffer,
  biasBuffer: GPUBuffer,
  numRows: number,
  featureDim: number,
  eps: number,
): GPUBuffer {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device;

  const outputSizeBytes = numRows * featureDim * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  configU32View[0] = numRows;
  configU32View[1] = featureDim;
  configF32View[2] = eps;
  configU32View[3] = 0;
  const configBuf = kernel.getConfigBuffer(device, `${numRows}:${featureDim}`, 16, configU8View);

  const pipeline = kernel.getPipeline(device, "layerNormFwd", () => layerNormForwardShader());

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [xBuffer, weightBuffer, biasBuffer, outBuffer, configBuf]);

  trackSharedEncoderWrite(xBuffer);
  trackSharedEncoderWrite(weightBuffer);
  trackSharedEncoderWrite(biasBuffer);
  trackSharedEncoderWrite(outBuffer);

  dispatchComputePass(
    pipeline,
    bindGroup,
    numRows, // one workgroup per row
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
  const ctx = getWebGPUDevice()!;
  const device = ctx.device;

  const outputSizeBytes = numRows * featureDim * 4;
  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  configU32View[0] = numRows;
  configU32View[1] = featureDim;
  configF32View[2] = eps;
  configU32View[3] = 0;
  const configBuf = kernel.getConfigBuffer(device, `${numRows}:${featureDim}`, 16, configU8View);

  const pipeline = kernel.getPipeline(device, "layerNormBwdGradX", () => layerNormBackwardGradXShader());

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [gradOutputBuffer, xBuffer, weightBuffer, outBuffer, configBuf]);

  trackSharedEncoderWrite(gradOutputBuffer);
  trackSharedEncoderWrite(xBuffer);
  trackSharedEncoderWrite(weightBuffer);
  trackSharedEncoderWrite(outBuffer);

  dispatchComputePass(
    pipeline,
    bindGroup,
    numRows,
  );

  return outBuffer;
}

/**
 * Dispatch fused LayerNorm backward gradWeight + gradBias kernel.
 * Two-pass approach:
 *   Pass 1: Compute row stats (mean, inv_std) for all N rows
 *   Pass 2: Accumulate gradWeight/gradBias using precomputed stats (no barriers)
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
  const ctx = getWebGPUDevice()!;
  const device = ctx.device;

  configU32View[0] = numRows;
  configU32View[1] = featureDim;
  configF32View[2] = eps;
  configU32View[3] = 0;
  const configBuf = kernel.getConfigBuffer(device, `${numRows}:${featureDim}`, 16, configU8View);

  // Pass 1: Compute row stats (mean[N], inv_std[N])
  // Use persistent cached buffers (same pattern as K-split temp buffers)
  const { meanBuffer, invStdBuffer } = getOrCreateRowStatsTempBuffers(device, numRows);

  const statsPipeline = kernel.getPipeline(device, "layerNormRowStats", () => layerNormRowStatsShader());

  const statsBindGroup = cachedCreateBindGroup(device, statsPipeline,
    [xBuffer, meanBuffer, invStdBuffer, configBuf]);

  trackSharedEncoderWrite(xBuffer);
  trackSharedEncoderWrite(meanBuffer);
  trackSharedEncoderWrite(invStdBuffer);

  dispatchComputePass(
    statsPipeline,
    statsBindGroup,
    numRows, // one workgroup per row
  );

  // Pass 2: Accumulate gradWeight/gradBias using precomputed stats
  const featureSizeBytes = featureDim * 4;
  const gradWeightBuffer = allocateOutputBuffer(featureSizeBytes) as unknown as GPUBuffer;
  const gradBiasBuffer = allocateOutputBuffer(featureSizeBytes) as unknown as GPUBuffer;

  const gradPipeline = kernel.getPipeline(device, "layerNormBwdGradWBv2", () => layerNormBackwardGradWeightBiasShader());

  const numWorkgroups = Math.ceil(featureDim / WORKGROUP_SIZE);

  const gradBindGroup = cachedCreateBindGroup(device, gradPipeline,
    [gradOutputBuffer, xBuffer, meanBuffer, invStdBuffer, gradWeightBuffer, gradBiasBuffer, configBuf]);

  trackSharedEncoderWrite(gradOutputBuffer);
  trackSharedEncoderWrite(gradWeightBuffer);
  trackSharedEncoderWrite(gradBiasBuffer);

  dispatchComputePass(
    gradPipeline,
    gradBindGroup,
    numWorkgroups,
  );

  return { gradWeightBuffer, gradBiasBuffer };
}

/**
 * Reset all module-local mutable state (pipeline cache, row stats temp buffers).
 */
export function resetLayerNormKernelState(): void {
  kernel.reset();
  for (const entry of rowStatsTempCache.values()) {
    entry.meanBuffer.destroy();
    entry.invStdBuffer.destroy();
  }
  rowStatsTempCache.clear();
}
