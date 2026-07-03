/**
 * GPU top-K prefilter for sampling readbacks.
 *
 * Decode-time sampling only needs the top-K logits (K=64), but the naive path
 * reads the FULL logits row to the CPU ([1,1,151936] f32 = 600KB staging copy
 * + Array.from into 152k boxed numbers + a JS partial-select) — ~8ms of pure
 * CPU per token on the Qwen3-1.7B decode path. This kernel reduces the
 * readback to K (value, index) pairs (512B for K=64).
 *
 * Two passes, both iterative selection with a workgroup-level argmax tree
 * reduction (deterministic tie-break: smaller index wins, matching a CPU
 * linear scan that takes the FIRST maximum — greedy argmax is bit-identical
 * to the full-logits argmax):
 *   pass 1: NW workgroups each select the top-K of their contiguous chunk
 *           → partial (value, index) pairs [NW*K]
 *   pass 2: one workgroup merges the NW*K partials → final top-K, sorted
 *           descending (by construction of iterative selection)
 *
 * Runs OUTSIDE the lazy plan, at readback level (like read()): the input is
 * an already-forced contiguous f32 buffer; passes are encoded on the shared
 * encoder after the producing work, then flushed and read back through a
 * persistent MAP_READ staging buffer (excluded from the pool).
 */

import type { BackendTensor } from "../types";
import { cachedCreateBindGroup } from "./bind-group-cache";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, WebGPUTensor } from "./gpu-types";
import { asGPUTensor, GPUBufferUsage, GPUMapMode } from "./gpu-types";
import { ensureContiguous } from "./ops/views";
import { flushSharedEncoder, getSharedEncoderInstance } from "./shared-encoder";
import { onTeardown, requireContext } from "./webgpu-state";

const WG = 256; // workgroup size
const NW = 64; // pass-1 workgroups

// ============================================================================
// WGSL
// ============================================================================

/** Pass 1: per-chunk top-K selection. maxPT = ceil(ceil(length/NW)/WG). */
function pass1WGSL(k: number, maxPT: number): string {
  return `
const WGS: u32 = ${WG}u;
const K: u32 = ${k}u;
const NW: u32 = ${NW}u;
const MAX_PT: u32 = ${maxPT}u;
const NEG_INF: f32 = -3.4028234e38;
const NO_IDX: u32 = 0xffffffffu;

struct Params { offset: u32, length: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> pvals: array<f32>;
@group(0) @binding(2) var<storage, read_write> pidx: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sVal: array<f32, WGS>;
var<workgroup> sIdx: array<u32, WGS>;

@compute @workgroup_size(${WG})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let w = wid.x;
  let tid = lid.x;
  let len = params.length;
  let chunk = (len + NW - 1u) / NW;
  let start = w * chunk;
  let end = min(start + chunk, len);

  // Load this thread's strided slice of the chunk into registers.
  // Element index of private slot s: e = start + s*WGS + tid.
  var vals: array<f32, MAX_PT>;
  for (var s = 0u; s < MAX_PT; s = s + 1u) {
    let e = start + s * WGS + tid;
    if (e < end) {
      vals[s] = input[params.offset + e];
    } else {
      vals[s] = NEG_INF;
    }
  }

  for (var kk = 0u; kk < K; kk = kk + 1u) {
    // Thread-local argmax (ascending s → ascending e, so strict '>' keeps
    // the SMALLEST index among equal values).
    var bv = NEG_INF;
    var bi = NO_IDX;
    for (var s = 0u; s < MAX_PT; s = s + 1u) {
      if (vals[s] > bv) {
        bv = vals[s];
        bi = start + s * WGS + tid;
      }
    }
    sVal[tid] = bv;
    sIdx[tid] = bi;
    workgroupBarrier();
    // Tree reduction; ties resolve to the smaller index.
    var step = WGS / 2u;
    while (step > 0u) {
      if (tid < step) {
        let ov = sVal[tid + step];
        let oi = sIdx[tid + step];
        if (ov > sVal[tid] || (ov == sVal[tid] && oi < sIdx[tid])) {
          sVal[tid] = ov;
          sIdx[tid] = oi;
        }
      }
      workgroupBarrier();
      step = step / 2u;
    }
    let winV = sVal[0];
    let winI = sIdx[0];
    workgroupBarrier(); // everyone has read the winner before sVal is reused

    if (tid == 0u) {
      pvals[w * K + kk] = winV;
      pidx[w * K + kk] = winI;
    }
    // Owner thread removes the winner from its private slice.
    if (winI != NO_IDX) {
      let r = winI - start;
      if (r % WGS == tid) {
        vals[r / WGS] = NEG_INF;
      }
    }
  }
}
`;
}

/** Pass 2: merge NW*K partials → final top-K. ptTwo = (NW*K)/WG. */
function pass2WGSL(k: number): string {
  const count = NW * k;
  const ptTwo = Math.ceil(count / WG);
  return `
const WGS: u32 = ${WG}u;
const K: u32 = ${k}u;
const COUNT: u32 = ${count}u;
const PT: u32 = ${ptTwo}u;
const NEG_INF: f32 = -3.4028234e38;
const NO_IDX: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read> pvals: array<f32>;
@group(0) @binding(1) var<storage, read> pidx: array<u32>;
@group(0) @binding(2) var<storage, read_write> outVals: array<f32>;
@group(0) @binding(3) var<storage, read_write> outIdx: array<u32>;

var<workgroup> sVal: array<f32, WGS>;
var<workgroup> sIdx: array<u32, WGS>;

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;

  // Private copy of this thread's strided slice (value + ORIGINAL index).
  var vals: array<f32, PT>;
  var idxs: array<u32, PT>;
  for (var s = 0u; s < PT; s = s + 1u) {
    let e = s * WGS + tid;
    if (e < COUNT) {
      vals[s] = pvals[e];
      idxs[s] = pidx[e];
    } else {
      vals[s] = NEG_INF;
      idxs[s] = NO_IDX;
    }
  }

  for (var kk = 0u; kk < K; kk = kk + 1u) {
    var bv = NEG_INF;
    var bi = NO_IDX;
    for (var s = 0u; s < PT; s = s + 1u) {
      // Tie-break on the ORIGINAL index (all original indices are distinct).
      if (vals[s] > bv || (vals[s] == bv && idxs[s] < bi)) {
        bv = vals[s];
        bi = idxs[s];
      }
    }
    sVal[tid] = bv;
    sIdx[tid] = bi;
    workgroupBarrier();
    var step = WGS / 2u;
    while (step > 0u) {
      if (tid < step) {
        let ov = sVal[tid + step];
        let oi = sIdx[tid + step];
        if (ov > sVal[tid] || (ov == sVal[tid] && oi < sIdx[tid])) {
          sVal[tid] = ov;
          sIdx[tid] = oi;
        }
      }
      workgroupBarrier();
      step = step / 2u;
    }
    let winV = sVal[0];
    let winI = sIdx[0];
    workgroupBarrier();

    if (tid == 0u) {
      outVals[kk] = winV;
      outIdx[kk] = winI;
    }
    // Removal: the thread holding the winning ORIGINAL index clears it.
    for (var s = 0u; s < PT; s = s + 1u) {
      if (idxs[s] == winI && winI != NO_IDX) {
        vals[s] = NEG_INF;
      }
    }
  }
}
`;
}

// ============================================================================
// Persistent buffers (excluded from the pool; destroyed at backend teardown)
// ============================================================================

type TopKBuffers = {
  pvals: GPUBuffer;
  pidx: GPUBuffer;
  outVals: GPUBuffer;
  outIdx: GPUBuffer;
  staging: GPUBuffer;
  params: GPUBuffer;
  paramsData: Uint32Array;
};

/** Keyed by k. Persistent across steps — a dedicated readback staging path. */
const buffersCache = new Map<number, TopKBuffers>();

function getOrCreateBuffers(k: number): TopKBuffers {
  let b = buffersCache.get(k);
  if (b) return b;
  const { device } = requireContext();
  const partialBytes = NW * k * 4;
  const outBytes = k * 4;
  b = {
    pvals: device.createBuffer({
      size: partialBytes,
      usage: GPUBufferUsage.STORAGE,
    }),
    pidx: device.createBuffer({
      size: partialBytes,
      usage: GPUBufferUsage.STORAGE,
    }),
    outVals: device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    outIdx: device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    }),
    staging: device.createBuffer({
      size: outBytes * 2,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    params: device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    paramsData: new Uint32Array(2),
  };
  buffersCache.set(k, b);
  return b;
}

export function resetTopKKernelState(): void {
  for (const b of buffersCache.values()) {
    b.pvals.destroy();
    b.pidx.destroy();
    b.outVals.destroy();
    b.outIdx.destroy();
    b.staging.destroy();
    b.params.destroy();
  }
  buffersCache.clear();
}
onTeardown(resetTopKKernelState);

// ============================================================================
// Entry point
// ============================================================================

export type TopKResult = { values: Float32Array; indices: Int32Array };

/**
 * Read the top-K (value, index) pairs of a 1-D slice of a forced f32 tensor,
 * sorted descending (ties: smaller index first). `offset`/`length` select the
 * slice in ELEMENTS (default: the whole tensor). Greedy decode = indices[0];
 * temperature/top-p sampling runs CPU-side over the K pairs.
 */
export async function readTopK(
  a: BackendTensor,
  k: number,
  opts?: { offset?: number; length?: number },
): Promise<TopKResult> {
  const ctx = requireContext();
  let tensor: WebGPUTensor = asGPUTensor(a);
  if (tensor.dtype !== "f32") {
    throw new Error(`readTopK: only f32 supported (got ${tensor.dtype})`);
  }
  const originalTensor = tensor;
  if (!tensor.isContiguous) {
    tensor = ensureContiguous(tensor);
  }
  const offset = (opts?.offset ?? 0) + (tensor.offset ?? 0);
  const length = opts?.length ?? tensor.size - (opts?.offset ?? 0);
  if (k > length) {
    throw new Error(`readTopK: k=${k} exceeds slice length ${length}`);
  }

  const bufs = getOrCreateBuffers(k);
  const chunk = Math.ceil(length / NW);
  const maxPT = Math.ceil(chunk / WG);

  // Params upload (queue-ordered: lands before the flush below submits).
  bufs.paramsData[0] = offset;
  bufs.paramsData[1] = length;
  ctx.queue.writeBuffer(bufs.params, 0, bufs.paramsData);

  const p1 = getPipeline(ctx, `topkPass1:${k}:${maxPT}`, pass1WGSL(k, maxPT));
  const p2 = getPipeline(ctx, `topkPass2:${k}`, pass2WGSL(k));

  const bg1 = cachedCreateBindGroup(ctx.device, p1, [
    tensor.buffer,
    bufs.pvals,
    bufs.pidx,
    bufs.params,
  ]);
  dispatchComputePass(p1, bg1, NW, 1, 1, "topkPass1");
  const bg2 = cachedCreateBindGroup(ctx.device, p2, [
    bufs.pvals,
    bufs.pidx,
    bufs.outVals,
    bufs.outIdx,
  ]);
  dispatchComputePass(p2, bg2, 1, 1, 1, "topkPass2");

  // Copy results to the persistent staging buffer and submit.
  const outBytes = k * 4;
  const enc = getSharedEncoderInstance();
  if (enc) {
    enc.copyBufferToBuffer(bufs.outVals, 0, bufs.staging, 0, outBytes);
    enc.copyBufferToBuffer(bufs.outIdx, 0, bufs.staging, outBytes, outBytes);
    flushSharedEncoder();
  } else {
    const e = ctx.device.createCommandEncoder();
    e.copyBufferToBuffer(bufs.outVals, 0, bufs.staging, 0, outBytes);
    e.copyBufferToBuffer(bufs.outIdx, 0, bufs.staging, outBytes, outBytes);
    ctx.queue.submit([e.finish()]);
  }

  // mapAsync waits on the queue timeline for the copies above to complete.
  await bufs.staging.mapAsync(GPUMapMode.READ);
  const mapped = bufs.staging.getMappedRange();
  const values = new Float32Array(mapped.slice(0, outBytes));
  const indices = new Int32Array(
    new Uint32Array(mapped.slice(outBytes, outBytes * 2)),
  );
  bufs.staging.unmap();

  if (tensor !== originalTensor && tensor.destroy) {
    tensor.destroy();
  }
  return { values, indices };
}
