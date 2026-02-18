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

// Shared memory for K and V tiles: [BC, HD]
var<workgroup> k_tile: array<f32, ${BC * headDim}>;
var<workgroup> v_tile: array<f32, ${BC * headDim}>;

@compute @workgroup_size(${BR})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let q_block = wid.x;   // which Q block
  let h = wid.y;          // head index
  let b = wid.z;          // batch index
  let tid = lid.x;        // thread = Q row within block

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  // Global Q row index
  let q_row = q_block * ${BR}u + tid;
  let valid = q_row < N;

  // Base offset for this (batch, head) in the flattened [B,H,N,D] layout
  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load Q row into registers
  var q_reg: array<f32, ${headDim}>;
  if (valid) {
    let q_base = bh_offset + q_row * D;
    for (var d = 0u; d < ${headDim}u; d++) {
      q_reg[d] = Q[q_base + d];
    }
  }

  // Running online softmax state
  var m_i = -3.402823e+38f;  // running max
  var l_i = 0.0f;            // running sum of exp
  var o_acc: array<f32, ${headDim}>;  // running output accumulator
  for (var d = 0u; d < ${headDim}u; d++) {
    o_acc[d] = 0.0;
  }

  // Number of KV tiles
  let num_kv_tiles = (N + ${BC - 1}u) / ${BC}u;

  for (var tile = 0u; tile < num_kv_tiles; tile++) {
    let kv_start = tile * ${BC}u;

    // Cooperatively load K tile [BC, D] into shared memory
    // Each of BR threads loads some rows
    for (var row = tid; row < ${BC}u; row += ${BR}u) {
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let k_base = bh_offset + kv_row * D;
        for (var d = 0u; d < ${headDim}u; d++) {
          k_tile[row * ${headDim}u + d] = K[k_base + d];
        }
      } else {
        for (var d = 0u; d < ${headDim}u; d++) {
          k_tile[row * ${headDim}u + d] = 0.0;
        }
      }
    }

    // Cooperatively load V tile [BC, D] into shared memory
    for (var row = tid; row < ${BC}u; row += ${BR}u) {
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let v_base = bh_offset + kv_row * D;
        for (var d = 0u; d < ${headDim}u; d++) {
          v_tile[row * ${headDim}u + d] = V[v_base + d];
        }
      } else {
        for (var d = 0u; d < ${headDim}u; d++) {
          v_tile[row * ${headDim}u + d] = 0.0;
        }
      }
    }

    workgroupBarrier();

    // Compute scores for this tile and update online softmax
    if (valid) {
      // Determine how many KV positions are valid in this tile
      var tile_end = min(${BC}u, N - kv_start);

      // Step 1: Compute scores and find tile max
      var tile_max = -3.402823e+38f;
      var scores: array<f32, ${BC}>;

      for (var j = 0u; j < tile_end; j++) {
        // Causal mask: skip positions where kv_pos > q_pos
        let kv_pos = kv_start + j;
        if (config.is_causal != 0u && kv_pos > q_row) {
          scores[j] = -3.402823e+38f;
          continue;
        }

        // Dot product: Q_row . K_tile[j]
        var dot = 0.0f;
        for (var d = 0u; d < ${headDim}u; d++) {
          dot += q_reg[d] * k_tile[j * ${headDim}u + d];
        }
        scores[j] = dot * scale;
        tile_max = max(tile_max, scores[j]);
      }

      // Mark invalid positions in the tile
      for (var j = tile_end; j < ${BC}u; j++) {
        scores[j] = -3.402823e+38f;
      }

      // Step 2: Online softmax update
      let m_new = max(m_i, tile_max);
      let correction = exp(m_i - m_new);

      // Rescale existing accumulator
      l_i = l_i * correction;
      for (var d = 0u; d < ${headDim}u; d++) {
        o_acc[d] = o_acc[d] * correction;
      }

      // Add new tile contributions
      for (var j = 0u; j < tile_end; j++) {
        let p = exp(scores[j] - m_new);
        l_i += p;
        for (var d = 0u; d < ${headDim}u; d++) {
          o_acc[d] += p * v_tile[j * ${headDim}u + d];
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
    for (var d = 0u; d < ${headDim}u; d++) {
      O[o_base + d] = o_acc[d] * inv_l;
    }

    // Save logsumexp: L = m + log(l)
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

var<workgroup> k_tile: array<f32, ${BC * headDim}>;
var<workgroup> v_tile: array<f32, ${BC * headDim}>;

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

  // Load Q row, dO row, and saved stats into registers
  var q_reg: array<f32, ${headDim}>;
  var dO_reg: array<f32, ${headDim}>;
  var dq_acc: array<f32, ${headDim}>;
  var L_i = 0.0f;
  var D_i = 0.0f;

  if (valid) {
    let base = bh_offset + q_row * D;
    for (var d = 0u; d < ${headDim}u; d++) {
      q_reg[d] = Q[base + d];
      dO_reg[d] = dO[base + d];
      dq_acc[d] = 0.0;
    }
    L_i = L_buf[bh_offset_L + q_row];
    D_i = D_buf[bh_offset_L + q_row];
  }

  let num_kv_tiles = (N + ${BC - 1}u) / ${BC}u;

  for (var tile = 0u; tile < num_kv_tiles; tile++) {
    let kv_start = tile * ${BC}u;

    // Cooperatively load K and V tiles
    for (var row = tid; row < ${BC}u; row += ${BR}u) {
      let kv_row = kv_start + row;
      if (kv_row < N) {
        let k_base = bh_offset + kv_row * D;
        for (var d = 0u; d < ${headDim}u; d++) {
          k_tile[row * ${headDim}u + d] = K[k_base + d];
          v_tile[row * ${headDim}u + d] = V[k_base + d];
        }
      } else {
        for (var d = 0u; d < ${headDim}u; d++) {
          k_tile[row * ${headDim}u + d] = 0.0;
          v_tile[row * ${headDim}u + d] = 0.0;
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

        // Recompute score: s = Q[i] . K[j] * scale
        var s = 0.0f;
        for (var d = 0u; d < ${headDim}u; d++) {
          s += q_reg[d] * k_tile[j * ${headDim}u + d];
        }
        s = s * scale;

        // Recompute P[i,j] = exp(s - L[i])
        let p = exp(s - L_i);

        // Compute dO[i] . V[j]
        var dov = 0.0f;
        for (var d = 0u; d < ${headDim}u; d++) {
          dov += dO_reg[d] * v_tile[j * ${headDim}u + d];
        }

        // dS[i,j] = P[i,j] * (dO[i].V[j] - D[i])
        let ds = p * (dov - D_i);

        // dQ[i] += dS[i,j] * K[j] * scale
        for (var d = 0u; d < ${headDim}u; d++) {
          dq_acc[d] += ds * k_tile[j * ${headDim}u + d];
        }
      }
    }

    workgroupBarrier();
  }

  // Write dQ
  if (valid) {
    let base = bh_offset + q_row * D;
    for (var d = 0u; d < ${headDim}u; d++) {
      dQ[base + d] = dq_acc[d] * scale;
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

// Tiled shared memory: BQ_BW=${BQ_BW} Q rows loaded cooperatively per tile
var<workgroup> q_tile: array<f32, ${BQ_BW * headDim}>;
var<workgroup> dO_tile: array<f32, ${BQ_BW * headDim}>;
var<workgroup> L_tile: array<f32, ${BQ_BW}>;
var<workgroup> D_tile: array<f32, ${BQ_BW}>;

@compute @workgroup_size(${BC_BW})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let kv_block = wid.x;
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;  // this thread's KV row within the block

  let N = config.seq_len;
  let D = config.head_dim;
  let scale = config.scale;

  let kv_row = kv_block * ${BC_BW}u + tid;
  let valid = kv_row < N;

  let bh_offset = (b * config.num_heads + h) * N * D;
  let bh_offset_L = (b * config.num_heads + h) * N;

  // Load this thread's K and V rows into registers
  var k_reg: array<f32, ${headDim}>;
  var v_reg: array<f32, ${headDim}>;
  var dk_acc: array<f32, ${headDim}>;
  var dv_acc: array<f32, ${headDim}>;

  if (valid) {
    let base = bh_offset + kv_row * D;
    for (var d = 0u; d < ${headDim}u; d++) {
      k_reg[d] = K[base + d];
      v_reg[d] = V[base + d];
      dk_acc[d] = 0.0;
      dv_acc[d] = 0.0;
    }
  }

  // Loop over Q positions in tiles of BQ_BW=${BQ_BW}
  let num_q_tiles = (N + ${BQ_BW - 1}u) / ${BQ_BW}u;
  for (var qt = 0u; qt < num_q_tiles; qt++) {
    let q_start = qt * ${BQ_BW}u;

    // Causal early exit: skip entire tile if max qi in tile < min kv_row in block
    // Uses kv_block * BC_BW (uniform across all threads) instead of kv_row
    if (config.is_causal != 0u && q_start + ${BQ_BW - 1}u < kv_block * ${BC_BW}u) {
      continue;
    }

    // Cooperative load: all ${BC_BW} threads load BQ*D/${BC_BW} elements each
    for (var idx = tid; idx < ${BQ_BW * headDim}u; idx += ${BC_BW}u) {
      let row = idx / ${headDim}u;
      let col = idx % ${headDim}u;
      let qi = q_start + row;
      if (qi < N) {
        let base = bh_offset + qi * D + col;
        q_tile[idx] = Q[base];
        dO_tile[idx] = dO[base];
      } else {
        q_tile[idx] = 0.0;
        dO_tile[idx] = 0.0;
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

    // Inner loop over BQ_BW Q rows — shared memory is read-only, no barriers
    if (valid) {
      let tile_end = min(${BQ_BW}u, N - q_start);
      for (var j = 0u; j < tile_end; j++) {
        let qi = q_start + j;
        if (config.is_causal != 0u && kv_row > qi) { continue; }

        // Recompute score: Q[qi] . K[kv_row] * scale
        var s = 0.0f;
        for (var d = 0u; d < ${headDim}u; d++) {
          s += q_tile[j * ${headDim}u + d] * k_reg[d];
        }
        s = s * scale;

        let p = exp(s - L_tile[j]);

        var dov = 0.0f;
        for (var d = 0u; d < ${headDim}u; d++) {
          dov += dO_tile[j * ${headDim}u + d] * v_reg[d];
        }

        let ds = p * (dov - D_tile[j]);

        for (var d = 0u; d < ${headDim}u; d++) {
          dk_acc[d] += ds * q_tile[j * ${headDim}u + d] * scale;
          dv_acc[d] += p * dO_tile[j * ${headDim}u + d];
        }
      }
    }

    workgroupBarrier();
  }

  // Write dK and dV
  if (valid) {
    let base = bh_offset + kv_row * D;
    for (var d = 0u; d < ${headDim}u; d++) {
      dK[base + d] = dk_acc[d];
      dV[base + d] = dv_acc[d];
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
