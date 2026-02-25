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
  allocateOutputBuffer,
  cachedCreateBindGroup,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { WORKGROUP_SIZE } from "./shape-utils";
import { wgslSumReduction, wgslDualSumReduction, trackBuffers } from "./wgsl-helpers";

// Shared WGSL struct for LayerNorm config (16 bytes, 4 fields)
const WGSL_LN_CONFIG = `struct LNConfig {
  num_rows: u32,
  feature_dim: u32,
  eps: f32,
  _pad: u32,
};`;

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
${WGSL_LN_CONFIG}

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

  // Phase 1: Parallel sum for mean
  var local_sum = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    local_sum += x[base + i];
  }
  sdata[tid] = local_sum;
  workgroupBarrier();

  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let mean = sdata[0] / f32(D);

  // Phase 2: Parallel sum for variance
  var local_var = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let diff = x[base + i] - mean;
    local_var += diff * diff;
  }
  sdata[tid] = local_var;
  workgroupBarrier();

  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let inv_std = inverseSqrt(sdata[0] / f32(D) + config.eps);

  // Phase 3: Normalize + affine transform
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let normalized = (x[base + i] - mean) * inv_std;
    output[base + i] = normalized * weight[i] + bias[i];
  }
}
`;
}

function layerNormBackwardGradXShader(): string {
  return `
${WGSL_LN_CONFIG}

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

  // Recompute mean
  var local_sum = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    local_sum += x[base + i];
  }
  sdata[tid] = local_sum;
  workgroupBarrier();
  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let mean = sdata[0] / Df;

  // Recompute variance + inv_std
  var local_var = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let diff = x[base + i] - mean;
    local_var += diff * diff;
  }
  sdata[tid] = local_var;
  workgroupBarrier();
  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let inv_std = inverseSqrt(sdata[0] / Df + config.eps);

  // Compute grad_normalized = grad_output * weight
  // Then reduce: c1 = mean(grad_normalized), c2 = mean(grad_normalized * normalized)
  var local_c1 = 0.0;
  var local_c2 = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let gn = grad_output[base + i] * weight[i];
    let norm_i = (x[base + i] - mean) * inv_std;
    local_c1 += gn;
    local_c2 += gn * norm_i;
  }
  sdata[tid] = local_c1;
  sdata2[tid] = local_c2;
  workgroupBarrier();
  ${wgslDualSumReduction("sdata", "sdata2", WORKGROUP_SIZE / 2)}
  let c1 = sdata[0] / Df;
  let c2 = sdata2[0] / Df;

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
${WGSL_LN_CONFIG}

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

  // Phase 1: Parallel sum for mean
  var local_sum = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    local_sum += x[base + i];
  }
  sdata[tid] = local_sum;
  workgroupBarrier();
  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let mean = sdata[0] / Df;

  // Phase 2: Parallel sum for variance → inv_std
  var local_var = 0.0;
  for (var i = tid; i < D; i += ${WORKGROUP_SIZE}u) {
    let diff = x[base + i] - mean;
    local_var += diff * diff;
  }
  sdata[tid] = local_var;
  workgroupBarrier();
  ${wgslSumReduction("sdata", WORKGROUP_SIZE / 2)}
  let inv_std = inverseSqrt(sdata[0] / Df + config.eps);

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
${WGSL_LN_CONFIG}

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

// ============================================================================
// Config Buffer (cached per numRows x featureDim)
// ============================================================================

const configCache = new Map<string, GPUBuffer>();
// Reusable typed arrays for config writes
const configArrayBuffer = new ArrayBuffer(16);
const configU32View = new Uint32Array(configArrayBuffer);
const configF32View = new Float32Array(configArrayBuffer);
const configU8View = new Uint8Array(configArrayBuffer);

function getOrCreateConfigBuffer(
  device: GPUDevice,
  numRows: number,
  featureDim: number,
  eps: number,
): GPUBuffer {
  const key = `${numRows}:${featureDim}`;
  let buf = configCache.get(key);
  if (!buf) {
    buf = device.createBuffer({
      size: 16, // { num_rows: u32, feature_dim: u32, eps: f32, _pad: u32 }
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    configCache.set(key, buf);
  }
  configU32View[0] = numRows;
  configU32View[1] = featureDim;
  configF32View[2] = eps;
  configU32View[3] = 0; // padding
  device.queue.writeBuffer(buf, 0, configU8View);
  return buf;
}

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
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = numRows * featureDim * 4; // f32
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(device, numRows, featureDim, eps);

  const pipeline = getPipeline(
    ctx,
    "layerNormFwd",
    layerNormForwardShader(),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [xBuffer, weightBuffer, biasBuffer, outBuffer, configBuf]);

  trackBuffers(xBuffer, weightBuffer, biasBuffer, outBuffer);

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
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = numRows * featureDim * 4;
  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(device, numRows, featureDim, eps);

  const pipeline = getPipeline(
    ctx,
    "layerNormBwdGradX",
    layerNormBackwardGradXShader(),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [gradOutputBuffer, xBuffer, weightBuffer, outBuffer, configBuf]);

  trackBuffers(gradOutputBuffer, xBuffer, weightBuffer, outBuffer);

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
  const ctx = requireContext();
  const device = ctx.device;

  const configBuf = getOrCreateConfigBuffer(device, numRows, featureDim, eps);

  // Pass 1: Compute row stats (mean[N], inv_std[N])
  // Use persistent cached buffers (same pattern as K-split temp buffers)
  const { meanBuffer, invStdBuffer } = getOrCreateRowStatsTempBuffers(device, numRows);

  const statsPipeline = getPipeline(
    ctx,
    "layerNormRowStats",
    layerNormRowStatsShader(),
  );

  const statsBindGroup = cachedCreateBindGroup(device, statsPipeline,
    [xBuffer, meanBuffer, invStdBuffer, configBuf]);

  trackBuffers(xBuffer, meanBuffer, invStdBuffer);

  dispatchComputePass(
    statsPipeline,
    statsBindGroup,
    numRows, // one workgroup per row
  );

  // Pass 2: Accumulate gradWeight/gradBias using precomputed stats
  const featureSizeBytes = featureDim * 4;
  const gradWeightBuffer = allocateOutputBuffer(featureSizeBytes);
  const gradBiasBuffer = allocateOutputBuffer(featureSizeBytes);

  const gradPipeline = getPipeline(
    ctx,
    "layerNormBwdGradWBv2",
    layerNormBackwardGradWeightBiasShader(),
  );

  const numWorkgroups = Math.ceil(featureDim / WORKGROUP_SIZE);

  const gradBindGroup = cachedCreateBindGroup(device, gradPipeline,
    [gradOutputBuffer, xBuffer, meanBuffer, invStdBuffer, gradWeightBuffer, gradBiasBuffer, configBuf]);

  trackBuffers(gradOutputBuffer, gradWeightBuffer, gradBiasBuffer);

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
  configCache.clear();
  for (const entry of rowStatsTempCache.values()) {
    entry.meanBuffer.destroy();
    entry.invStdBuffer.destroy();
  }
  rowStatsTempCache.clear();
}
