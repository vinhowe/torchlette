/**
 * Fused FlashAttention Kernel
 *
 * Single-dispatch forward and backward kernels that replace the decomposed
 * attention path (Q@K^T + scale + mask + softmax + attn@V).
 *
 * Uses online softmax (FlashAttention algorithm) to avoid materializing
 * the full N×N attention matrix.
 *
 * Forward:  Q,K,V [B,H,N,D] → O [B,H,N,D] + L [B,H,N] (logsumexp)
 * Backward: Q,K,V,L,dO → dQ,dK,dV
 *
 * Tiling: BR=64 Q rows per workgroup, BC=32 KV rows per tile.
 * Each thread handles one Q row (workgroup_size = BR = 64).
 */

import {
  dispatchComputePass,
  getWebGPUDevice,
  trackSharedEncoderWrite,
  allocateOutputBuffer,
  cachedCreateBindGroup,
} from "./index";

// WebGPU type definitions (runtime, not importable at compile time)
type GPUBuffer = {
  size: number;
  usage: number;
  destroy(): void;
};

type GPUDevice = {
  createShaderModule(descriptor: { code: string }): GPUShaderModule;
  createComputePipeline(descriptor: {
    layout: "auto";
    compute: { module: GPUShaderModule; entryPoint: string };
  }): GPUComputePipeline;
  createBuffer(descriptor: {
    size: number;
    usage: number;
    mappedAtCreation?: boolean;
  }): GPUBuffer;
  queue: {
    writeBuffer(
      buffer: GPUBuffer,
      offset: number,
      data: ArrayBufferView,
    ): void;
  };
};

type GPUShaderModule = object;
type GPUComputePipeline = {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
};
type GPUBindGroupLayout = object;

const GPUBufferUsage = {
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

// Tiling parameters
const BR = 64;  // Q rows per workgroup (also workgroup size)
const BC = 32;  // KV rows per tile
const BQ_BW = 16; // Q rows per tile in backward dKV kernel

// ============================================================================
// Pipeline Cache
// ============================================================================

const pipelineCache = new Map<string, GPUComputePipeline>();

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  code: string,
): GPUComputePipeline {
  let pipeline = pipelineCache.get(key);
  if (!pipeline) {
    const module = device.createShaderModule({ code });
    pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
    pipelineCache.set(key, pipeline);
  }
  return pipeline;
}

// ============================================================================
// Config Buffer Cache
// ============================================================================

const configCache = new Map<string, GPUBuffer>();

function getOrCreateConfigBuffer(
  device: GPUDevice,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: number,
): GPUBuffer {
  const key = `fa:${batchSize}:${numHeads}:${seqLen}:${headDim}:${scale}:${isCausal}`;
  let buf = configCache.get(key);
  if (!buf) {
    buf = device.createBuffer({
      size: 32, // 8 x u32/f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    configCache.set(key, buf);
  }
  // Pack config: 4 u32s + 1 f32 + 1 u32 + 2 padding
  const data = new ArrayBuffer(32);
  const u32View = new Uint32Array(data);
  const f32View = new Float32Array(data);
  u32View[0] = batchSize;
  u32View[1] = numHeads;
  u32View[2] = seqLen;
  u32View[3] = headDim;
  f32View[4] = scale;
  u32View[5] = isCausal;
  u32View[6] = 0; // pad
  u32View[7] = 0; // pad
  device.queue.writeBuffer(buf, 0, new Uint8Array(data));
  return buf;
}

// ============================================================================
// WGSL Shaders
// ============================================================================

/**
 * FlashAttention Forward Shader
 *
 * One workgroup per (q_block, head, batch). BR threads, each handling one Q row.
 * Loop over KV tiles of size BC. Online softmax accumulation.
 *
 * Input: Q,K,V [B,H,N,D] (flattened as storage arrays)
 * Output: O [B,H,N,D], L [B,H,N] (logsumexp per row)
 */
function flashAttentionForwardShader(headDim: number): string {
  const HD4 = headDim / 4;
  return `
struct FAConfig {
  batch_size: u32,
  num_heads: u32,
  seq_len: u32,
  head_dim: u32,
  scale: f32,
  is_causal: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read_write> L: array<f32>;
@group(0) @binding(5) var<uniform> config: FAConfig;

// Shared memory for K and V tiles: [BC, HD/4] as vec4
var<workgroup> k_tile: array<vec4<f32>, ${BC * HD4}>;
var<workgroup> v_tile: array<vec4<f32>, ${BC * HD4}>;

@compute @workgroup_size(${BR})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let q_block = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let q_row = q_block * ${BR}u + tid;
  let valid = q_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load Q row into vec4 registers
  var q_reg: array<vec4<f32>, ${HD4}>;
  if (valid) {
    let q_base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${HD4}u; d4++) {
      let off = q_base + d4 * 4u;
      q_reg[d4] = vec4<f32>(Q[off], Q[off+1u], Q[off+2u], Q[off+3u]);
    }
  }

  // Running online softmax state
  var m_i = -3.402823e+38f;
  var l_i = 0.0f;
  var o_acc: array<vec4<f32>, ${HD4}>;
  for (var d4 = 0u; d4 < ${HD4}u; d4++) {
    o_acc[d4] = vec4<f32>(0.0);
  }

  let num_kv_tiles = (N + ${BC - 1}u) / ${BC}u;

  for (var tile = 0u; tile < num_kv_tiles; tile++) {
    let kv_start = tile * ${BC}u;

    // Cooperatively load K tile into shared memory as vec4
    for (var row = tid; row < ${BC}u; row += ${BR}u) {
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let k_base = bh_offset + kv_row * D;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          let off = k_base + d4 * 4u;
          k_tile[row * ${HD4}u + d4] = vec4<f32>(K[off], K[off+1u], K[off+2u], K[off+3u]);
        }
      } else {
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          k_tile[row * ${HD4}u + d4] = vec4<f32>(0.0);
        }
      }
    }

    // Cooperatively load V tile into shared memory as vec4
    for (var row = tid; row < ${BC}u; row += ${BR}u) {
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let v_base = bh_offset + kv_row * D;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          let off = v_base + d4 * 4u;
          v_tile[row * ${HD4}u + d4] = vec4<f32>(V[off], V[off+1u], V[off+2u], V[off+3u]);
        }
      } else {
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          v_tile[row * ${HD4}u + d4] = vec4<f32>(0.0);
        }
      }
    }

    workgroupBarrier();

    if (valid) {
      var tile_end = min(${BC}u, N - kv_start);

      var tile_max = -3.402823e+38f;
      var scores: array<f32, ${BC}>;

      for (var j = 0u; j < tile_end; j++) {
        let kv_pos = kv_start + j;
        if (config.is_causal != 0u && kv_pos > q_row) {
          scores[j] = -3.402823e+38f;
          continue;
        }

        // Vec4 dot product: Q_row . K_tile[j]
        var s = 0.0f;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          s += dot(q_reg[d4], k_tile[j * ${HD4}u + d4]);
        }
        scores[j] = s * scale;
        tile_max = max(tile_max, scores[j]);
      }

      for (var j = tile_end; j < ${BC}u; j++) {
        scores[j] = -3.402823e+38f;
      }

      let m_new = max(m_i, tile_max);
      let correction = exp(m_i - m_new);

      l_i = l_i * correction;
      for (var d4 = 0u; d4 < ${HD4}u; d4++) {
        o_acc[d4] = o_acc[d4] * correction;
      }

      for (var j = 0u; j < tile_end; j++) {
        let p = exp(scores[j] - m_new);
        l_i += p;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          o_acc[d4] += p * v_tile[j * ${HD4}u + d4];
        }
      }

      m_i = m_new;
    }

    workgroupBarrier();
  }

  // Final normalization and write output
  if (valid) {
    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
    let o_base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${HD4}u; d4++) {
      let v = o_acc[d4] * inv_l;
      let off = o_base + d4 * 4u;
      O[off] = v.x;
      O[off+1u] = v.y;
      O[off+2u] = v.z;
      O[off+3u] = v.w;
    }

    let logsumexp = m_i + log(max(l_i, 1e-10f));
    L[bh_offset_L + q_row] = logsumexp;
  }
}
`;
}

/**
 * FlashAttention Backward D Precompute Shader
 *
 * D[i] = sum_d(dO[i,d] * O[i,d]) for each row i.
 * One thread per head_dim element, one workgroup per row.
 *
 * Input: dO [B,H,N,D], O [B,H,N,D]
 * Output: D [B,H,N]
 */
function flashAttentionBackwardDShader(headDim: number): string {
  // Use headDim threads per workgroup for reduction
  const WG = Math.max(headDim, 32); // at least 32 for reduction
  return `
struct FAConfig {
  batch_size: u32,
  num_heads: u32,
  seq_len: u32,
  head_dim: u32,
  scale: f32,
  is_causal: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> dO: array<f32>;
@group(0) @binding(1) var<storage, read> Out: array<f32>;
@group(0) @binding(2) var<storage, read_write> D: array<f32>;
@group(0) @binding(3) var<uniform> config: FAConfig;

var<workgroup> sdata: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row_idx = wid.x;  // flattened row in [B*H*N]
  let tid = lid.x;

  let N = config.seq_len;
  let D_dim = config.head_dim;
  let total_rows = config.batch_size * config.num_heads * N;

  if (row_idx >= total_rows) {
    return;
  }

  // Compute base offset for this row
  let base = row_idx * D_dim;

  // Each thread accumulates partial dot product
  var local_sum = 0.0f;
  for (var d = tid; d < D_dim; d += ${WG}u) {
    local_sum += dO[base + d] * Out[base + d];
  }

  sdata[tid] = local_sum;
  workgroupBarrier();

  // Tree reduction
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    workgroupBarrier();
  }

  if (tid == 0u) {
    D[row_idx] = sdata[0];
  }
}
`;
}

/**
 * FlashAttention Backward dQ Shader
 *
 * Same tiling structure as forward. One workgroup per Q block.
 * Loop over KV tiles, recompute attention from saved logsumexp.
 *
 * dS = P * (dO@V^T_row - D[i])  ... but we compute row-wise
 * dQ[i] += sum_j(dS[i,j] * K[j])
 *
 * Input: Q,K,V [B,H,N,D], L [B,H,N], D_buf [B,H,N], dO [B,H,N,D]
 * Output: dQ [B,H,N,D]
 */
function flashAttentionBackwardDQShader(headDim: number): string {
  const HD4 = headDim / 4;
  return `
struct FAConfig {
  batch_size: u32,
  num_heads: u32,
  seq_len: u32,
  head_dim: u32,
  scale: f32,
  is_causal: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> L_buf: array<f32>;
@group(0) @binding(4) var<storage, read> D_buf: array<f32>;
@group(0) @binding(5) var<storage, read> dO: array<f32>;
@group(0) @binding(6) var<storage, read_write> dQ: array<f32>;
@group(0) @binding(7) var<uniform> config: FAConfig;

var<workgroup> k_tile: array<vec4<f32>, ${BC * HD4}>;
var<workgroup> v_tile: array<vec4<f32>, ${BC * HD4}>;

@compute @workgroup_size(${BR})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let q_block = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let q_row = q_block * ${BR}u + tid;
  let valid = q_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load Q row, dO row into vec4 registers
  var q_reg: array<vec4<f32>, ${HD4}>;
  var dO_reg: array<vec4<f32>, ${HD4}>;
  var dq_acc: array<vec4<f32>, ${HD4}>;
  var L_i = 0.0f;
  var D_i = 0.0f;

  if (valid) {
    let base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${HD4}u; d4++) {
      let off = base + d4 * 4u;
      q_reg[d4] = vec4<f32>(Q[off], Q[off+1u], Q[off+2u], Q[off+3u]);
      dO_reg[d4] = vec4<f32>(dO[off], dO[off+1u], dO[off+2u], dO[off+3u]);
      dq_acc[d4] = vec4<f32>(0.0);
    }
    L_i = L_buf[bh_offset_L + q_row];
    D_i = D_buf[bh_offset_L + q_row];
  }

  let num_kv_tiles = (N + ${BC - 1}u) / ${BC}u;

  for (var tile = 0u; tile < num_kv_tiles; tile++) {
    let kv_start = tile * ${BC}u;

    // Cooperatively load K and V tiles as vec4
    for (var row = tid; row < ${BC}u; row += ${BR}u) {
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let k_base = bh_offset + kv_row * D;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          let off = k_base + d4 * 4u;
          k_tile[row * ${HD4}u + d4] = vec4<f32>(K[off], K[off+1u], K[off+2u], K[off+3u]);
          v_tile[row * ${HD4}u + d4] = vec4<f32>(V[off], V[off+1u], V[off+2u], V[off+3u]);
        }
      } else {
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          k_tile[row * ${HD4}u + d4] = vec4<f32>(0.0);
          v_tile[row * ${HD4}u + d4] = vec4<f32>(0.0);
        }
      }
    }

    workgroupBarrier();

    if (valid) {
      let tile_end = min(${BC}u, N - kv_start);

      for (var j = 0u; j < tile_end; j++) {
        let kv_pos = kv_start + j;
        if (config.is_causal != 0u && kv_pos > q_row) {
          continue;
        }

        // Vec4 dot product: Q[i] . K[j] * scale
        var s = 0.0f;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          s += dot(q_reg[d4], k_tile[j * ${HD4}u + d4]);
        }
        s = s * scale;

        let p = exp(s - L_i);

        // Vec4 dot product: dO[i] . V[j]
        var dov = 0.0f;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          dov += dot(dO_reg[d4], v_tile[j * ${HD4}u + d4]);
        }

        let ds = p * (dov - D_i);

        // dQ[i] += dS[i,j] * K[j]
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          dq_acc[d4] += ds * k_tile[j * ${HD4}u + d4];
        }
      }
    }

    workgroupBarrier();
  }

  // Write dQ (scale applied once at write)
  if (valid) {
    let base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${HD4}u; d4++) {
      let v = dq_acc[d4] * scale;
      let off = base + d4 * 4u;
      dQ[off] = v.x;
      dQ[off+1u] = v.y;
      dQ[off+2u] = v.z;
      dQ[off+3u] = v.w;
    }
  }
}
`;
}

/**
 * FlashAttention Backward dKV Shader
 *
 * One workgroup per KV block (BC_BW=64 KV rows).
 * Each thread handles one KV row. Q and dO are loaded in tiles of BQ_BW=16
 * rows via cooperative shared memory loading (all 64 threads participate),
 * reducing barrier count from 2*N to 2*ceil(N/BQ_BW).
 *
 * Input: Q,K,V [B,H,N,D], L [B,H,N], D_buf [B,H,N], dO [B,H,N,D]
 * Output: dK [B,H,N,D], dV [B,H,N,D]
 */
function flashAttentionBackwardDKVShader(headDim: number): string {
  const BC_BW = 64; // KV rows per workgroup for backward
  const HD4 = headDim / 4;
  return `
struct FAConfig {
  batch_size: u32,
  num_heads: u32,
  seq_len: u32,
  head_dim: u32,
  scale: f32,
  is_causal: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> L_buf: array<f32>;
@group(0) @binding(4) var<storage, read> D_buf: array<f32>;
@group(0) @binding(5) var<storage, read> dO: array<f32>;
@group(0) @binding(6) var<storage, read_write> dK: array<f32>;
@group(0) @binding(7) var<storage, read_write> dV: array<f32>;
@group(0) @binding(8) var<uniform> config: FAConfig;

// Tiled shared memory as vec4: BQ_BW=${BQ_BW} Q rows, HD/4 vec4s per row
var<workgroup> q_tile: array<vec4<f32>, ${BQ_BW * HD4}>;
var<workgroup> dO_tile: array<vec4<f32>, ${BQ_BW * HD4}>;
var<workgroup> L_tile: array<f32, ${BQ_BW}>;
var<workgroup> D_tile: array<f32, ${BQ_BW}>;

@compute @workgroup_size(${BC_BW})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let kv_block = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let kv_row = kv_block * ${BC_BW}u + tid;
  let valid = kv_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load this thread's K and V rows into vec4 registers
  var k_reg: array<vec4<f32>, ${HD4}>;
  var v_reg: array<vec4<f32>, ${HD4}>;
  var dk_acc: array<vec4<f32>, ${HD4}>;
  var dv_acc: array<vec4<f32>, ${HD4}>;

  if (valid) {
    let base = bh_offset + kv_row * D;
    for (var d4 = 0u; d4 < ${HD4}u; d4++) {
      let off = base + d4 * 4u;
      k_reg[d4] = vec4<f32>(K[off], K[off+1u], K[off+2u], K[off+3u]);
      v_reg[d4] = vec4<f32>(V[off], V[off+1u], V[off+2u], V[off+3u]);
      dk_acc[d4] = vec4<f32>(0.0);
      dv_acc[d4] = vec4<f32>(0.0);
    }
  }

  // Loop over Q positions in tiles of BQ_BW=${BQ_BW}
  let num_q_tiles = (N + ${BQ_BW - 1}u) / ${BQ_BW}u;
  for (var qt = 0u; qt < num_q_tiles; qt++) {
    let q_start = qt * ${BQ_BW}u;

    // Causal early exit: skip entire tile if max qi in tile < min kv_row in block
    if (config.is_causal != 0u && q_start + ${BQ_BW - 1}u < kv_block * ${BC_BW}u) {
      continue;
    }

    // Cooperative load: all ${BC_BW} threads load vec4s
    for (var idx = tid; idx < ${BQ_BW * HD4}u; idx += ${BC_BW}u) {
      let row = idx / ${HD4}u;
      let d4 = idx % ${HD4}u;
      let qi = q_start + row;
      if (qi < N) {
        let base = bh_offset + qi * D + d4 * 4u;
        q_tile[idx] = vec4<f32>(Q[base], Q[base+1u], Q[base+2u], Q[base+3u]);
        dO_tile[idx] = vec4<f32>(dO[base], dO[base+1u], dO[base+2u], dO[base+3u]);
      } else {
        q_tile[idx] = vec4<f32>(0.0);
        dO_tile[idx] = vec4<f32>(0.0);
      }
    }
    // Load L and D values (first BQ_BW threads)
    if (tid < ${BQ_BW}u) {
      let qi = q_start + tid;
      if (qi < N) {
        L_tile[tid] = L_buf[bh_offset_L + qi];
        D_tile[tid] = D_buf[bh_offset_L + qi];
      }
    }

    workgroupBarrier();

    // Inner loop over BQ_BW Q rows
    if (valid) {
      let tile_end = min(${BQ_BW}u, N - q_start);
      for (var j = 0u; j < tile_end; j++) {
        let qi = q_start + j;
        if (config.is_causal != 0u && kv_row > qi) { continue; }

        // Vec4 dot product: Q[qi] . K[kv_row] * scale
        var s = 0.0f;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          s += dot(q_tile[j * ${HD4}u + d4], k_reg[d4]);
        }
        s = s * scale;

        let p = exp(s - L_tile[j]);

        var dov = 0.0f;
        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          dov += dot(dO_tile[j * ${HD4}u + d4], v_reg[d4]);
        }

        let ds = p * (dov - D_tile[j]);
        let ds_scale = ds * scale;

        for (var d4 = 0u; d4 < ${HD4}u; d4++) {
          dk_acc[d4] += ds_scale * q_tile[j * ${HD4}u + d4];
          dv_acc[d4] += p * dO_tile[j * ${HD4}u + d4];
        }
      }
    }

    workgroupBarrier();
  }

  // Write dK and dV
  if (valid) {
    let base = bh_offset + kv_row * D;
    for (var d4 = 0u; d4 < ${HD4}u; d4++) {
      let off = base + d4 * 4u;
      dK[off] = dk_acc[d4].x;
      dK[off+1u] = dk_acc[d4].y;
      dK[off+2u] = dk_acc[d4].z;
      dK[off+3u] = dk_acc[d4].w;
      dV[off] = dv_acc[d4].x;
      dV[off+1u] = dv_acc[d4].y;
      dV[off+2u] = dv_acc[d4].z;
      dV[off+3u] = dv_acc[d4].w;
    }
  }
}
`;
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/**
 * Dispatch FlashAttention forward kernel.
 * Q,K,V: [B, H, N, D] → O: [B, H, N, D], L: [B, H, N]
 */
export function dispatchFlashAttentionForward(
  qBuffer: GPUBuffer,
  kBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): { outputBuffer: GPUBuffer; logsumexpBuffer: GPUBuffer } {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4; // f32
  const logsumexpSizeBytes = batchSize * numHeads * seqLen * 4; // f32

  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;
  const lseBuffer = allocateOutputBuffer(logsumexpSizeBytes) as unknown as GPUBuffer;

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const pipeline = getOrCreatePipeline(
    device,
    `faFwd:${headDim}`,
    flashAttentionForwardShader(headDim),
  );

  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
    [qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer, configBuf] as any) as any;

  trackSharedEncoderWrite(qBuffer as any);
  trackSharedEncoderWrite(kBuffer as any);
  trackSharedEncoderWrite(vBuffer as any);
  trackSharedEncoderWrite(outBuffer as any);
  trackSharedEncoderWrite(lseBuffer as any);

  const numQBlocks = Math.ceil(seqLen / BR);
  dispatchComputePass(
    pipeline as any,
    bindGroup as any,
    numQBlocks,
    numHeads,
    batchSize,
  );

  return { outputBuffer: outBuffer, logsumexpBuffer: lseBuffer };
}

/**
 * Dispatch FlashAttention backward D precompute kernel.
 * dO,O: [B,H,N,D] → D: [B,H,N]
 */
export function dispatchFlashAttentionBackwardD(
  dOBuffer: GPUBuffer,
  oBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): GPUBuffer {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  const totalRows = batchSize * numHeads * seqLen;
  const outputSizeBytes = totalRows * 4; // f32

  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const WG = Math.max(headDim, 32);
  const pipeline = getOrCreatePipeline(
    device,
    `faBwdD:${headDim}`,
    flashAttentionBackwardDShader(headDim),
  );

  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
    [dOBuffer, oBuffer, outBuffer, configBuf] as any) as any;

  trackSharedEncoderWrite(dOBuffer as any);
  trackSharedEncoderWrite(oBuffer as any);
  trackSharedEncoderWrite(outBuffer as any);

  dispatchComputePass(
    pipeline as any,
    bindGroup as any,
    totalRows,
  );

  return outBuffer;
}

/**
 * Dispatch FlashAttention backward dQ kernel.
 * Q,K,V,L,D,dO → dQ: [B,H,N,D]
 */
export function dispatchFlashAttentionBackwardDQ(
  qBuffer: GPUBuffer,
  kBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  lBuffer: GPUBuffer,
  dBuffer: GPUBuffer,
  dOBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): GPUBuffer {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4;

  const outBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const pipeline = getOrCreatePipeline(
    device,
    `faBwdDQ:${headDim}`,
    flashAttentionBackwardDQShader(headDim),
  );

  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer, configBuf] as any) as any;

  trackSharedEncoderWrite(qBuffer as any);
  trackSharedEncoderWrite(kBuffer as any);
  trackSharedEncoderWrite(vBuffer as any);
  trackSharedEncoderWrite(lBuffer as any);
  trackSharedEncoderWrite(dBuffer as any);
  trackSharedEncoderWrite(dOBuffer as any);
  trackSharedEncoderWrite(outBuffer as any);

  const numQBlocks = Math.ceil(seqLen / BR);
  dispatchComputePass(
    pipeline as any,
    bindGroup as any,
    numQBlocks,
    numHeads,
    batchSize,
  );

  return outBuffer;
}

/**
 * Dispatch FlashAttention backward dKV kernel.
 * Q,K,V,L,D,dO → dK: [B,H,N,D], dV: [B,H,N,D]
 */
export function dispatchFlashAttentionBackwardDKV(
  qBuffer: GPUBuffer,
  kBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  lBuffer: GPUBuffer,
  dBuffer: GPUBuffer,
  dOBuffer: GPUBuffer,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  headDim: number,
  scale: number,
  isCausal: boolean,
): { dKBuffer: GPUBuffer; dVBuffer: GPUBuffer } {
  const ctx = getWebGPUDevice()!;
  const device = ctx.device as unknown as GPUDevice;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4;

  const dKBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;
  const dVBuffer = allocateOutputBuffer(outputSizeBytes) as unknown as GPUBuffer;

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const BC_BW = 64;
  const pipeline = getOrCreatePipeline(
    device,
    `faBwdDKV:${headDim}`,
    flashAttentionBackwardDKVShader(headDim),
  );

  const bindGroup = cachedCreateBindGroup(device as any, pipeline as any,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer, configBuf] as any) as any;

  trackSharedEncoderWrite(qBuffer as any);
  trackSharedEncoderWrite(kBuffer as any);
  trackSharedEncoderWrite(vBuffer as any);
  trackSharedEncoderWrite(lBuffer as any);
  trackSharedEncoderWrite(dBuffer as any);
  trackSharedEncoderWrite(dOBuffer as any);
  trackSharedEncoderWrite(dKBuffer as any);
  trackSharedEncoderWrite(dVBuffer as any);

  const numKVBlocks = Math.ceil(seqLen / BC_BW);
  dispatchComputePass(
    pipeline as any,
    bindGroup as any,
    numKVBlocks,
    numHeads,
    batchSize,
  );

  return { dKBuffer, dVBuffer };
}
