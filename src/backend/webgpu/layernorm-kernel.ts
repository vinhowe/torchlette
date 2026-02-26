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
import type { TileKernelSpec } from "./tile-ir";
import { createTileKernelDispatcher } from "./tile-dispatch";
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
// Tile IR Forward Kernel
// ============================================================================

const layerNormFwdSpec: TileKernelSpec = {
  name: "layerNormFwd",
  workgroupSize: WORKGROUP_SIZE,
  bindings: {
    x:      { storage: "read",       type: "f32" },
    weight: { storage: "read",       type: "f32" },
    bias:   { storage: "read",       type: "f32" },
    output: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows:    "u32",
    feature_dim: "u32",
    eps:         "f32",
  },
  grid: (u) => [u.num_rows],

  kernel(ctx) {
    const row  = ctx.programId(0);
    const D    = ctx.uniform("feature_dim");
    const Df   = D.toF32();
    const base = row.mul(D);
    const offs = base.add(ctx.blockRange(D));

    // Phase 0: load + reduce to mean
    const xVals = ctx.load("x", offs);
    const mean  = ctx.reduce(xVals, "sum").div(Df);

    // Phase 1: compute variance and inv_std
    const diff     = xVals.sub(mean);
    const variance = ctx.reduce(diff.mul(diff), "sum").div(Df);
    const invStd   = variance.add(ctx.uniform("eps").toF32()).rsqrt();

    // Phase 2: normalize + affine + store
    const normalized = diff.mul(invStd);
    const w = ctx.load("weight", ctx.blockRange(D));
    const b = ctx.load("bias", ctx.blockRange(D));
    ctx.store("output", offs, normalized.mul(w).add(b));
  },
};

const fwdTileKernel = createTileKernelDispatcher(layerNormFwdSpec);

const USE_TILE_IR_LAYERNORM = process.env.TORCHLETTE_TILE_LAYERNORM !== "0";

// ============================================================================
// Tile IR Backward Kernels
// ============================================================================

const WG = WORKGROUP_SIZE; // 256

/** Emit tree sum reduction: sdata[0] = sum(sdata[0..wgSize-1]) */
function emitTreeSumReduction(
  ctx: Parameters<TileKernelSpec["kernel"]>[0],
  smem: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["sharedArray"]>,
  tid: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["u32"]>,
  wgSize: number,
): void {
  for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
    ctx.ifThen(tid.lt(ctx.u32(stride)), () => {
      smem.write(tid, smem.read(tid).add(smem.read(tid.add(ctx.u32(stride)))));
    });
    ctx.barrier();
  }
}

/** Emit dual tree sum reduction for two shared arrays simultaneously */
function emitDualTreeSumReduction(
  ctx: Parameters<TileKernelSpec["kernel"]>[0],
  smem1: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["sharedArray"]>,
  smem2: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["sharedArray"]>,
  tid: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["u32"]>,
  wgSize: number,
): void {
  for (let stride = wgSize >> 1; stride >= 1; stride >>= 1) {
    ctx.ifThen(tid.lt(ctx.u32(stride)), () => {
      smem1.write(tid, smem1.read(tid).add(smem1.read(tid.add(ctx.u32(stride)))));
      smem2.write(tid, smem2.read(tid).add(smem2.read(tid.add(ctx.u32(stride)))));
    });
    ctx.barrier();
  }
}

/** Emit strided loop: for (var i = tid; i < bound; i += wgSize) { body(i); } */
function emitStridedLoop(
  ctx: Parameters<TileKernelSpec["kernel"]>[0],
  tid: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["u32"]>,
  bound: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["u32"]>,
  wgSize: number,
  body: (i: ReturnType<Parameters<TileKernelSpec["kernel"]>[0]["u32"]>) => void,
): void {
  const maxIters = bound.add(ctx.u32(wgSize - 1)).div(ctx.u32(wgSize));
  ctx.forRange(ctx.u32(0), maxIters, (iter) => {
    const i = tid.add(iter.mul(ctx.u32(wgSize)));
    ctx.ifThen(i.lt(bound), () => {
      body(i);
    });
  });
}

/**
 * LayerNorm backward gradX kernel (tile-IR).
 * One workgroup per row. Recomputes mean/variance in shared memory,
 * then computes gradX with dual reduction coefficients.
 */
const layerNormBackwardGradXSpec: TileKernelSpec = {
  name: "lnBwdGradX",
  workgroupSize: WG,
  bindings: {
    grad_output: { storage: "read", type: "f32" },
    x:           { storage: "read", type: "f32" },
    weight:      { storage: "read", type: "f32" },
    grad_x:      { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows:    "u32",
    feature_dim: "u32",
    eps:         "f32",
  },
  grid: (u) => [u.num_rows],

  kernel(ctx) {
    const tid = ctx.localIndex();
    const row = ctx.programId(0);
    const D = ctx.uniform("feature_dim");
    const Df = D.toF32();
    const base = row.mul(D);

    const sdata = ctx.sharedArray("sdata", WG, "f32");
    const sdata2 = ctx.sharedArray("sdata2", WG, "f32");

    // Phase 1: Recompute mean
    const localSum = ctx.emitVar("local_sum", "f32", ctx.f32(0.0));
    emitStridedLoop(ctx, tid, D, WG, (i) => {
      localSum.addAssign(ctx.load("x", base.add(i)));
    });
    sdata.write(tid, localSum.get());
    ctx.barrier();
    emitTreeSumReduction(ctx, sdata, tid, WG);
    const mean = ctx.emitLet("mean", sdata.read(ctx.u32(0)).div(Df));

    // Phase 2: Recompute variance + inv_std
    const localVar = ctx.emitVar("local_var", "f32", ctx.f32(0.0));
    emitStridedLoop(ctx, tid, D, WG, (i) => {
      const diff = ctx.load("x", base.add(i)).sub(mean);
      localVar.addAssign(diff.mul(diff));
    });
    sdata.write(tid, localVar.get());
    ctx.barrier();
    emitTreeSumReduction(ctx, sdata, tid, WG);
    const invStd = ctx.emitLet("inv_std",
      sdata.read(ctx.u32(0)).div(Df).add(ctx.uniform("eps").toF32()).rsqrt());

    // Phase 3: Dual reduction for c1, c2
    const localC1 = ctx.emitVar("local_c1", "f32", ctx.f32(0.0));
    const localC2 = ctx.emitVar("local_c2", "f32", ctx.f32(0.0));
    emitStridedLoop(ctx, tid, D, WG, (i) => {
      const gn = ctx.load("grad_output", base.add(i)).mul(ctx.load("weight", i));
      const normI = ctx.load("x", base.add(i)).sub(mean).mul(invStd);
      localC1.addAssign(gn);
      localC2.addAssign(gn.mul(normI));
    });
    sdata.write(tid, localC1.get());
    sdata2.write(tid, localC2.get());
    ctx.barrier();
    emitDualTreeSumReduction(ctx, sdata, sdata2, tid, WG);
    const c1 = ctx.emitLet("c1", sdata.read(ctx.u32(0)).div(Df));
    const c2 = ctx.emitLet("c2", sdata2.read(ctx.u32(0)).div(Df));

    // Phase 4: Output gradX
    const maxIters4 = D.add(ctx.u32(WG - 1)).div(ctx.u32(WG));
    ctx.forRange(ctx.u32(0), maxIters4, (iter) => {
      const i = tid.add(iter.mul(ctx.u32(WG)));
      const inBounds = i.lt(D);
      const gn = ctx.load("grad_output", base.add(i)).mul(ctx.load("weight", i));
      const normI = ctx.load("x", base.add(i)).sub(mean).mul(invStd);
      ctx.guardedStore("grad_x", inBounds, base.add(i),
        gn.sub(c1).sub(normI.mul(c2)).mul(invStd));
    });
  },
};

/**
 * LayerNorm row stats kernel (tile-IR).
 * Computes per-row mean and inv_std for use by gradWeight/gradBias pass.
 */
const layerNormRowStatsSpec: TileKernelSpec = {
  name: "lnRowStats",
  workgroupSize: WG,
  bindings: {
    x:           { storage: "read", type: "f32" },
    row_mean:    { storage: "read_write", type: "f32" },
    row_inv_std: { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows:    "u32",
    feature_dim: "u32",
    eps:         "f32",
  },
  grid: (u) => [u.num_rows],

  kernel(ctx) {
    const tid = ctx.localIndex();
    const row = ctx.programId(0);
    const D = ctx.uniform("feature_dim");
    const Df = D.toF32();
    const base = row.mul(D);

    const sdata = ctx.sharedArray("sdata", WG, "f32");

    // Phase 1: Mean
    const localSum = ctx.emitVar("local_sum", "f32", ctx.f32(0.0));
    emitStridedLoop(ctx, tid, D, WG, (i) => {
      localSum.addAssign(ctx.load("x", base.add(i)));
    });
    sdata.write(tid, localSum.get());
    ctx.barrier();
    emitTreeSumReduction(ctx, sdata, tid, WG);
    const mean = ctx.emitLet("mean", sdata.read(ctx.u32(0)).div(Df));

    // Phase 2: Variance + inv_std
    const localVar = ctx.emitVar("local_var", "f32", ctx.f32(0.0));
    emitStridedLoop(ctx, tid, D, WG, (i) => {
      const diff = ctx.load("x", base.add(i)).sub(mean);
      localVar.addAssign(diff.mul(diff));
    });
    sdata.write(tid, localVar.get());
    ctx.barrier();
    emitTreeSumReduction(ctx, sdata, tid, WG);
    const invStd = ctx.emitLet("inv_std",
      sdata.read(ctx.u32(0)).div(Df).add(ctx.uniform("eps").toF32()).rsqrt());

    // Thread 0 writes row stats
    const isThread0 = tid.eq(ctx.u32(0));
    ctx.guardedStore("row_mean", isThread0, row, mean);
    ctx.guardedStore("row_inv_std", isThread0, row, invStd);
  },
};

/**
 * LayerNorm backward gradWeight + gradBias kernel (tile-IR).
 * One thread per feature, loops over all rows.
 */
const layerNormBackwardGradWeightBiasSpec: TileKernelSpec = {
  name: "lnBwdGradWB",
  workgroupSize: WG,
  bindings: {
    grad_output: { storage: "read", type: "f32" },
    x:           { storage: "read", type: "f32" },
    row_mean:    { storage: "read", type: "f32" },
    row_inv_std: { storage: "read", type: "f32" },
    grad_weight: { storage: "read_write", type: "f32" },
    grad_bias:   { storage: "read_write", type: "f32" },
  },
  uniforms: {
    num_rows:    "u32",
    feature_dim: "u32",
    eps:         "f32",
  },
  grid: (u) => [Math.ceil(u.feature_dim / WG)],

  kernel(ctx) {
    const featureIdx = ctx.globalId(0);
    const D = ctx.uniform("feature_dim");
    const N = ctx.uniform("num_rows");

    // Early return if out of bounds
    ctx.ifThen(featureIdx.ge(D), () => {
      ctx.emitReturn();
    });

    const accGW = ctx.emitVar("acc_gw", "f32", ctx.f32(0.0));
    const accGB = ctx.emitVar("acc_gb", "f32", ctx.f32(0.0));

    ctx.forRange(ctx.u32(0), N, (row) => {
      const base = row.mul(D);
      const go = ctx.load("grad_output", base.add(featureIdx));
      const normalized = ctx.load("x", base.add(featureIdx))
        .sub(ctx.load("row_mean", row))
        .mul(ctx.load("row_inv_std", row));
      accGW.addAssign(go.mul(normalized));
      accGB.addAssign(go);
    });

    const inBounds = featureIdx.lt(D);
    ctx.guardedStore("grad_weight", inBounds, featureIdx, accGW.get());
    ctx.guardedStore("grad_bias", inBounds, featureIdx, accGB.get());
  },
};

const gradXTileKernel = USE_TILE_IR_LAYERNORM
  ? createTileKernelDispatcher(layerNormBackwardGradXSpec) : null;
const rowStatsTileKernel = USE_TILE_IR_LAYERNORM
  ? createTileKernelDispatcher(layerNormRowStatsSpec) : null;
const gradWBTileKernel = USE_TILE_IR_LAYERNORM
  ? createTileKernelDispatcher(layerNormBackwardGradWeightBiasSpec) : null;

// ============================================================================
// WGSL Shaders (backward kernels — hand-written)
// ============================================================================

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

  if (gradXTileKernel) {
    gradXTileKernel.dispatch(
      { grad_output: gradOutputBuffer, x: xBuffer, weight: weightBuffer, grad_x: outBuffer },
      { num_rows: numRows, feature_dim: featureDim, eps },
    );
    return outBuffer;
  }

  const ctx = requireContext();
  const device = ctx.device;

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

  // Pass 1: Compute row stats (mean[N], inv_std[N])
  const { meanBuffer, invStdBuffer } = getOrCreateRowStatsTempBuffers(device, numRows);

  if (rowStatsTileKernel && gradWBTileKernel) {
    rowStatsTileKernel.dispatch(
      { x: xBuffer, row_mean: meanBuffer, row_inv_std: invStdBuffer },
      { num_rows: numRows, feature_dim: featureDim, eps },
    );

    // Pass 2: Accumulate gradWeight/gradBias using precomputed stats
    const featureSizeBytes = featureDim * 4;
    const gradWeightBuffer = allocateOutputBuffer(featureSizeBytes);
    const gradBiasBuffer = allocateOutputBuffer(featureSizeBytes);

    gradWBTileKernel.dispatch(
      { grad_output: gradOutputBuffer, x: xBuffer, row_mean: meanBuffer,
        row_inv_std: invStdBuffer, grad_weight: gradWeightBuffer, grad_bias: gradBiasBuffer },
      { num_rows: numRows, feature_dim: featureDim, eps },
    );

    return { gradWeightBuffer, gradBiasBuffer };
  }

  const configBuf = getOrCreateConfigBuffer(device, numRows, featureDim, eps);

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
    numRows,
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
  fwdTileKernel.reset();
  gradXTileKernel?.reset();
  rowStatsTileKernel?.reset();
  gradWBTileKernel?.reset();
  configCache.clear();
  for (const entry of rowStatsTempCache.values()) {
    entry.meanBuffer.destroy();
    entry.invStdBuffer.destroy();
  }
  rowStatsTempCache.clear();
}
