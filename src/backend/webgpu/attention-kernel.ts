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
 * Scalar mode: Each thread handles one row (workgroup_size = 64).
 * Subgroup mode: 4 threads per row (workgroup_size = 256), each thread
 * handles 1/4 of headDim. Dot products reduced via subgroupShuffleXor.
 */

import {
  dispatchComputePass,
  trackSharedEncoderWrite,
  allocateOutputBuffer,
  cachedCreateBindGroup,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";

import { getSubgroupSupport } from "./matmul/types";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";

/** Track multiple buffers in the shared encoder write set. */
function trackBuffers(...buffers: GPUBuffer[]): void {
  for (const buf of buffers) trackSharedEncoderWrite(buf);
}

// Tiling parameters
const BR = 64;  // Q rows per workgroup (also workgroup size in scalar mode)
const BC = 32;  // KV rows per tile
const BQ_BW = 16; // Q rows per tile in backward dKV kernel

// Subgroup cooperative parameters
const THREADS_PER_ROW = 4; // threads cooperating on one row's dot product
const WG_SG = 256;  // workgroup size in subgroup mode (BR * THREADS_PER_ROW)

// Shared WGSL struct for flash attention config (32 bytes, 8 fields)
const WGSL_FA_CONFIG = `struct FAConfig {
  batch_size: u32,
  num_heads: u32,
  seq_len: u32,
  head_dim: u32,
  scale: f32,
  is_causal: u32,
  _pad0: u32,
  _pad1: u32,
};`;

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
// Subgroup support detection for attention kernels
// ============================================================================

/**
 * Check if subgroup cooperative attention is usable for given headDim.
 * Requires: subgroups feature supported AND HD4 divisible by THREADS_PER_ROW.
 */
function useSubgroupAttention(headDim: number): boolean {
  const sg = getSubgroupSupport();
  if (!sg?.supported) return false;
  const HD4 = headDim / 4;
  // Need HD4 divisible by THREADS_PER_ROW so each thread gets equal vec4 elements.
  // Also require at least 4 vec4 elements per thread (headDim >= 64) for the
  // register reduction to meaningfully outweigh shuffle overhead.
  const D4_COUNT = HD4 / THREADS_PER_ROW;
  return HD4 % THREADS_PER_ROW === 0 && D4_COUNT >= 4;
}

// ============================================================================
// WGSL Shaders
// ============================================================================

/**
 * FlashAttention Forward Shader
 *
 * One workgroup per (q_block, head, batch). BR rows per workgroup.
 * Scalar mode: 64 threads, 1 per row.
 * Subgroup mode: 256 threads, 4 per row. Dot products via subgroupShuffleXor.
 *
 * Input: Q,K,V [B,H,N,D] (flattened as storage arrays)
 * Output: O [B,H,N,D], L [B,H,N] (logsumexp per row)
 */
function flashAttentionForwardShader(headDim: number, useSubgroups: boolean): string {
  if (headDim % 4 !== 0) throw new Error(`flashAttentionForwardShader requires headDim divisible by 4, got ${headDim}`);
  const HD4 = headDim / 4;

  if (useSubgroups) {
    const D4_COUNT = HD4 / THREADS_PER_ROW; // vec4 elements per thread
    const WG = WG_SG;
    return `
enable subgroups;

${WGSL_FA_CONFIG}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read_write> L: array<f32>;
@group(0) @binding(5) var<uniform> config: FAConfig;

// Shared memory for K and V tiles: [BC, HD/4] as vec4
var<workgroup> k_tile: array<vec4<f32>, ${BC * HD4}>;
var<workgroup> v_tile: array<vec4<f32>, ${BC * HD4}>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) tidx: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let q_block = wid.x;
  let h = wid.y;
  let b = wid.z;

  let row_lane = tidx / ${THREADS_PER_ROW}u;  // which row (0..63)
  let d4_lane = tidx % ${THREADS_PER_ROW}u;   // which portion of headDim (0..3)
  let d4_start = d4_lane * ${D4_COUNT}u;       // starting vec4 index

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let q_row = q_block * ${BR}u + row_lane;
  let valid = q_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load Q row slice into registers (each thread loads 1/4 of headDim)
  // var is zero-initialized; invalid threads keep zeros → dot products = 0
  var q_reg: array<vec4<f32>, ${D4_COUNT}>;
  if (valid) {
    let q_base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      let off = q_base + (d4_start + d4) * 4u;
      q_reg[d4] = vec4<f32>(Q[off], Q[off+1u], Q[off+2u], Q[off+3u]);
    }
  }

  // Running online softmax state
  var m_i = -3.402823e+38f;
  var l_i = 0.0f;
  var o_acc: array<vec4<f32>, ${D4_COUNT}>;
  for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
    o_acc[d4] = vec4<f32>(0.0);
  }

  let num_kv_tiles = (N + ${BC - 1}u) / ${BC}u;

  for (var tile = 0u; tile < num_kv_tiles; tile++) {
    let kv_start = tile * ${BC}u;

    // Cooperatively load K tile (256 threads loading BC*HD4 elements)
    for (var idx = tidx; idx < ${BC * HD4}u; idx += ${WG}u) {
      let row = idx / ${HD4}u;
      let d4 = idx % ${HD4}u;
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let off = bh_offset + kv_row * D + d4 * 4u;
        k_tile[idx] = vec4<f32>(K[off], K[off+1u], K[off+2u], K[off+3u]);
      } else {
        k_tile[idx] = vec4<f32>(0.0);
      }
    }

    // Cooperatively load V tile
    for (var idx = tidx; idx < ${BC * HD4}u; idx += ${WG}u) {
      let row = idx / ${HD4}u;
      let d4 = idx % ${HD4}u;
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let off = bh_offset + kv_row * D + d4 * 4u;
        v_tile[idx] = vec4<f32>(V[off], V[off+1u], V[off+2u], V[off+3u]);
      } else {
        v_tile[idx] = vec4<f32>(0.0);
      }
    }

    workgroupBarrier();

    // Score computation — ALL threads participate (subgroup-uniform control flow)
    // Invalid/masked threads have q_reg=0 so their partial sums are 0
    var tile_max = -3.402823e+38f;
    var scores: array<f32, ${BC}>;

    for (var j = 0u; j < ${BC}u; j++) {
      let kv_pos = kv_start + j;
      let is_active = valid && kv_pos < N && (config.is_causal == 0u || kv_pos <= q_row);

      // All threads compute partial dot (0 for invalid/masked via zero q_reg)
      var s_partial = 0.0f;
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        s_partial += dot(q_reg[d4], k_tile[j * ${HD4}u + d4_start + d4]);
      }
      // Tree reduction — subgroup-uniform, all threads execute
      s_partial += subgroupShuffleXor(s_partial, 1u);
      s_partial += subgroupShuffleXor(s_partial, 2u);

      let s = s_partial * scale;
      scores[j] = select(-3.402823e+38f, s, is_active);
      tile_max = select(tile_max, max(tile_max, s), is_active);
    }

    // Online softmax update (safe for invalid: m_i=-inf, correction=1, p=0)
    let m_new = max(m_i, tile_max);
    let correction = exp(m_i - m_new);

    l_i = l_i * correction;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      o_acc[d4] = o_acc[d4] * correction;
    }

    for (var j = 0u; j < ${BC}u; j++) {
      let p = exp(scores[j] - m_new);
      l_i += p;
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        o_acc[d4] += p * v_tile[j * ${HD4}u + d4_start + d4];
      }
    }

    m_i = m_new;

    workgroupBarrier();
  }

  // Final normalization and write output
  if (valid) {
    let inv_l = select(0.0, 1.0 / l_i, l_i > 0.0);
    let o_base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      let v = o_acc[d4] * inv_l;
      let off = o_base + (d4_start + d4) * 4u;
      O[off] = v.x;
      O[off+1u] = v.y;
      O[off+2u] = v.z;
      O[off+3u] = v.w;
    }

    // Only one thread per row writes L
    if (d4_lane == 0u) {
      let logsumexp = m_i + log(max(l_i, 1e-10f));
      L[bh_offset_L + q_row] = logsumexp;
    }
  }
}
`;
  }

  // Scalar (non-subgroup) path — original implementation
  return `
${WGSL_FA_CONFIG}

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
${WGSL_FA_CONFIG}

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
function flashAttentionBackwardDQShader(headDim: number, useSubgroups: boolean): string {
  if (headDim % 4 !== 0) throw new Error(`flashAttentionBackwardDQShader requires headDim divisible by 4, got ${headDim}`);
  const HD4 = headDim / 4;

  if (useSubgroups) {
    const D4_COUNT = HD4 / THREADS_PER_ROW;
    const WG = WG_SG;
    return `
enable subgroups;

${WGSL_FA_CONFIG}

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

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) tidx: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let q_block = wid.x;
  let h = wid.y;
  let b = wid.z;

  let row_lane = tidx / ${THREADS_PER_ROW}u;
  let d4_lane = tidx % ${THREADS_PER_ROW}u;
  let d4_start = d4_lane * ${D4_COUNT}u;

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let q_row = q_block * ${BR}u + row_lane;
  let valid = q_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load Q row slice, dO row slice into registers
  var q_reg: array<vec4<f32>, ${D4_COUNT}>;
  var dO_reg: array<vec4<f32>, ${D4_COUNT}>;
  var dq_acc: array<vec4<f32>, ${D4_COUNT}>;
  var L_i = 0.0f;
  var D_i = 0.0f;

  if (valid) {
    let base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      let off = base + (d4_start + d4) * 4u;
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

    // Cooperatively load K and V tiles (256 threads)
    for (var idx = tidx; idx < ${BC * HD4}u; idx += ${WG}u) {
      let row = idx / ${HD4}u;
      let d4 = idx % ${HD4}u;
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let off = bh_offset + kv_row * D + d4 * 4u;
        k_tile[idx] = vec4<f32>(K[off], K[off+1u], K[off+2u], K[off+3u]);
        v_tile[idx] = vec4<f32>(V[off], V[off+1u], V[off+2u], V[off+3u]);
      } else {
        k_tile[idx] = vec4<f32>(0.0);
        v_tile[idx] = vec4<f32>(0.0);
      }
    }

    workgroupBarrier();

    // ALL threads participate in shuffle — subgroup-uniform control flow
    // Invalid/masked threads have q_reg=0 and dO_reg=0 so partials are 0
    for (var j = 0u; j < ${BC}u; j++) {
      let kv_pos = kv_start + j;
      let is_active =valid && kv_pos < N && (config.is_causal == 0u || kv_pos <= q_row);

      // Partial dot product: Q[i] . K[j]
      var s_partial = 0.0f;
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        s_partial += dot(q_reg[d4], k_tile[j * ${HD4}u + d4_start + d4]);
      }
      s_partial += subgroupShuffleXor(s_partial, 1u);
      s_partial += subgroupShuffleXor(s_partial, 2u);
      let s = s_partial * scale;

      // p=0 for masked (s*scale is arbitrary but exp(anything - L_i) for invalid
      // doesn't matter since we gate accumulation below)
      let p = select(0.0f, exp(s - L_i), is_active);

      // Partial dot product: dO[i] . V[j]
      var dov_partial = 0.0f;
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        dov_partial += dot(dO_reg[d4], v_tile[j * ${HD4}u + d4_start + d4]);
      }
      dov_partial += subgroupShuffleXor(dov_partial, 1u);
      dov_partial += subgroupShuffleXor(dov_partial, 2u);

      let ds = p * (dov_partial - D_i);

      // dQ[i] += dS[i,j] * K[j] (each thread accumulates its slice)
      // ds=0 for inactive, so accumulation is a no-op
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        dq_acc[d4] += ds * k_tile[j * ${HD4}u + d4_start + d4];
      }
    }

    workgroupBarrier();
  }

  // Write dQ (each thread writes its slice)
  if (valid) {
    let base = bh_offset + q_row * D;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      let v = dq_acc[d4] * scale;
      let off = base + (d4_start + d4) * 4u;
      dQ[off] = v.x;
      dQ[off+1u] = v.y;
      dQ[off+2u] = v.z;
      dQ[off+3u] = v.w;
    }
  }
}
`;
  }

  // Scalar (non-subgroup) path
  return `
${WGSL_FA_CONFIG}

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
 * Scalar mode: 64 threads, 1 per KV row.
 * Subgroup mode: 256 threads, 4 per KV row. Dot products via subgroupShuffleXor.
 *
 * Input: Q,K,V [B,H,N,D], L [B,H,N], D_buf [B,H,N], dO [B,H,N,D]
 * Output: dK [B,H,N,D], dV [B,H,N,D]
 */
function flashAttentionBackwardDKVShader(headDim: number, useSubgroups: boolean): string {
  if (headDim % 4 !== 0) throw new Error(`flashAttentionBackwardDKVShader requires headDim divisible by 4, got ${headDim}`);
  const BC_BW = 64; // KV rows per workgroup for backward
  const HD4 = headDim / 4;

  if (useSubgroups) {
    const D4_COUNT = HD4 / THREADS_PER_ROW;
    const WG = BC_BW * THREADS_PER_ROW; // 256
    return `
enable subgroups;

${WGSL_FA_CONFIG}

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

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) tidx: u32,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let kv_block = wid.x;
  let h = wid.y;
  let b = wid.z;

  let row_lane = tidx / ${THREADS_PER_ROW}u;  // which KV row (0..63)
  let d4_lane = tidx % ${THREADS_PER_ROW}u;   // which portion (0..3)
  let d4_start = d4_lane * ${D4_COUNT}u;

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let kv_row = kv_block * ${BC_BW}u + row_lane;
  let valid = kv_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load this thread's K and V row slice into registers
  var k_reg: array<vec4<f32>, ${D4_COUNT}>;
  var v_reg: array<vec4<f32>, ${D4_COUNT}>;
  var dk_acc: array<vec4<f32>, ${D4_COUNT}>;
  var dv_acc: array<vec4<f32>, ${D4_COUNT}>;

  if (valid) {
    let base = bh_offset + kv_row * D;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      let off = base + (d4_start + d4) * 4u;
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

    // Cooperative load: all ${WG} threads load vec4s
    for (var idx = tidx; idx < ${BQ_BW * HD4}u; idx += ${WG}u) {
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
    // Load L and D values (first BQ_BW threads); zero for out-of-range
    if (tidx < ${BQ_BW}u) {
      let qi = q_start + tidx;
      if (qi < N) {
        L_tile[tidx] = L_buf[bh_offset_L + qi];
        D_tile[tidx] = D_buf[bh_offset_L + qi];
      } else {
        L_tile[tidx] = 0.0f;
        D_tile[tidx] = 0.0f;
      }
    }

    workgroupBarrier();

    // ALL threads participate in shuffle — subgroup-uniform control flow
    // Invalid/masked threads have k_reg=0 and v_reg=0 so partials are 0
    for (var j = 0u; j < ${BQ_BW}u; j++) {
      let qi = q_start + j;
      let is_active =valid && qi < N && (config.is_causal == 0u || kv_row <= qi);

      // Partial dot product: Q[qi] . K[kv_row]
      var s_partial = 0.0f;
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        s_partial += dot(q_tile[j * ${HD4}u + d4_start + d4], k_reg[d4]);
      }
      s_partial += subgroupShuffleXor(s_partial, 1u);
      s_partial += subgroupShuffleXor(s_partial, 2u);
      let s = s_partial * scale;

      let p = select(0.0f, exp(s - L_tile[j]), is_active);

      // Partial dot product: dO[qi] . V[kv_row]
      var dov_partial = 0.0f;
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        dov_partial += dot(dO_tile[j * ${HD4}u + d4_start + d4], v_reg[d4]);
      }
      dov_partial += subgroupShuffleXor(dov_partial, 1u);
      dov_partial += subgroupShuffleXor(dov_partial, 2u);

      let ds = p * (dov_partial - D_tile[j]);
      let ds_scale = ds * scale;

      // ds=0 for inactive, so accumulation is a no-op
      for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
        dk_acc[d4] += ds_scale * q_tile[j * ${HD4}u + d4_start + d4];
        dv_acc[d4] += p * dO_tile[j * ${HD4}u + d4_start + d4];
      }
    }

    workgroupBarrier();
  }

  // Write dK and dV (each thread writes its slice)
  if (valid) {
    let base = bh_offset + kv_row * D;
    for (var d4 = 0u; d4 < ${D4_COUNT}u; d4++) {
      let off = base + (d4_start + d4) * 4u;
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

  // Scalar (non-subgroup) path
  return `
${WGSL_FA_CONFIG}

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
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4; // f32
  const logsumexpSizeBytes = batchSize * numHeads * seqLen * 4; // f32

  const outBuffer = allocateOutputBuffer(outputSizeBytes);
  const lseBuffer = allocateOutputBuffer(logsumexpSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const sg = useSubgroupAttention(headDim);
  const pipeline = getPipeline(
    ctx,
    `faFwd:${headDim}${sg ? ":sg" : ""}`,
    flashAttentionForwardShader(headDim, sg),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer, configBuf]);

  trackBuffers(qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer);

  const numQBlocks = Math.ceil(seqLen / BR);
  dispatchComputePass(
    pipeline,
    bindGroup,
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
  const ctx = requireContext();
  const device = ctx.device;

  const totalRows = batchSize * numHeads * seqLen;
  const outputSizeBytes = totalRows * 4; // f32

  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const WG = Math.max(headDim, 32);
  const pipeline = getPipeline(
    ctx,
    `faBwdD:${headDim}`,
    flashAttentionBackwardDShader(headDim),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [dOBuffer, oBuffer, outBuffer, configBuf]);

  trackBuffers(dOBuffer, oBuffer, outBuffer);

  dispatchComputePass(
    pipeline,
    bindGroup,
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
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4;

  const outBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const sg = useSubgroupAttention(headDim);
  const pipeline = getPipeline(
    ctx,
    `faBwdDQ:${headDim}${sg ? ":sg" : ""}`,
    flashAttentionBackwardDQShader(headDim, sg),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer, configBuf]);

  trackBuffers(qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer);

  const numQBlocks = Math.ceil(seqLen / BR);
  dispatchComputePass(
    pipeline,
    bindGroup,
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
/**
 * Reset all module-local mutable state (pipeline cache, config buffer cache).
 */
export function resetAttentionKernelState(): void {
  configCache.clear();
}

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
  const ctx = requireContext();
  const device = ctx.device;

  const outputSizeBytes = batchSize * numHeads * seqLen * headDim * 4;

  const dKBuffer = allocateOutputBuffer(outputSizeBytes);
  const dVBuffer = allocateOutputBuffer(outputSizeBytes);

  const configBuf = getOrCreateConfigBuffer(
    device, batchSize, numHeads, seqLen, headDim, scale, isCausal ? 1 : 0,
  );

  const BC_BW = 64;
  const sg = useSubgroupAttention(headDim);
  const pipeline = getPipeline(
    ctx,
    `faBwdDKV:${headDim}${sg ? ":sg" : ""}`,
    flashAttentionBackwardDKVShader(headDim, sg),
  );

  const bindGroup = cachedCreateBindGroup(device, pipeline,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer, configBuf]);

  trackBuffers(qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer);

  const numKVBlocks = Math.ceil(seqLen / BC_BW);
  dispatchComputePass(
    pipeline,
    bindGroup,
    numKVBlocks,
    numHeads,
    batchSize,
  );

  return { dKBuffer, dVBuffer };
}
