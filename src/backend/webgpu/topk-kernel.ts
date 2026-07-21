/**
 * GPU top-K prefilter for sampling readbacks.
 *
 * Decode-time sampling only needs the top-K logits (K=64), but the naive path
 * reads the FULL logits row to the CPU ([1,1,151936] f32 = 600KB staging copy
 * + Array.from into 152k boxed numbers + a JS partial-select) — ~8ms of pure
 * CPU per token on the Qwen3-1.7B decode path. This kernel reduces the
 * readback to K (value, index) pairs (512B for K=64).
 *
 * Two passes, both iterative selection with a workgroup-level argmax reduction
 * (deterministic tie-break: smaller index wins, matching a CPU linear scan that
 * takes the FIRST maximum — greedy argmax is bit-identical to the full-logits
 * argmax):
 *   pass 1: NW workgroups each select the top-K of their contiguous chunk
 *           → partial (value, index) pairs [NW*K]
 *   pass 2: one workgroup merges the NW*K partials → final top-K, sorted
 *           descending (by construction of iterative selection)
 *
 * Both passes are AUTHORED ON TILE-IR (#65 — the last hand-WGSL kernel migrated
 * onto the substrate). The workgroup argmax is `ctx.pairArgReduce`, the shared
 * value+index reduction primitive added for this migration; the per-thread
 * slice lives in a private register array (`emitVarArray`); the K-round
 * selection loop, winner store and winner removal are the imperative tile-IR
 * statement API (forRange/ifThen/emitVar). Future compression: argmax/argmin
 * could ride pairArgReduce too (deliberately NOT migrated here).
 *
 * Runs OUTSIDE the lazy plan, at readback level (like read()): the input is
 * an already-forced contiguous f32 buffer; passes are encoded on the shared
 * encoder after the producing work, then flushed and read back through a
 * persistent MAP_READ staging buffer (excluded from the pool).
 */

import type { BackendTensor } from "../types";
import { cachedCreateBindGroup } from "./bind-group-cache";
import { allocateOutputBuffer } from "./buffer-arena";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, WebGPUTensor } from "./gpu-types";
import { asGPUTensor, GPUBufferUsage, GPUMapMode } from "./gpu-types";
import { ensureContiguous } from "./ops/views";
import { flushSharedEncoder, getSharedEncoderInstance } from "./shared-encoder";
import { compileTileKernel } from "./tile-compiler";
import type { KernelContext, TileKernelSpec } from "./tile-ir";
import { singleWorkgroup } from "./tile-ir";
import { onTeardown, requireContext } from "./webgpu-state";

const WG = 256; // workgroup size
const NW = 64; // pass-1 workgroups
const NEG_INF = -3.4028234e38;
const NO_IDX = 0xffffffff;

// ============================================================================
// Tile-IR kernel specs
// ============================================================================

/**
 * Pass 1: per-chunk top-K selection. Each of NW workgroups owns a contiguous
 * chunk of the input; each thread holds a strided register slice of maxPT
 * elements; K rounds of workgroup argmax (pairArgReduce) drain the top-K,
 * writing (value, index) partials for the chunk. maxPT = ceil(ceil(len/NW)/WG).
 */
function pass1Spec(
  k: number,
  maxPT: number,
  constParams?: { offset: number; length: number },
): TileKernelSpec {
  return {
    name: `topkPass1_k${k}_pt${maxPT}${
      constParams ? `_o${constParams.offset}_l${constParams.length}` : ""
    }`,
    workgroupSize: WG,
    bindings: {
      input: { storage: "read", type: "f32" },
      pvals: { storage: "read_write", type: "f32" },
      pidx: { storage: "read_write", type: "u32" },
    },
    // Lazy (`dispatchDeviceTopK`) bakes offset/length as WGSL constants so NO
    // params UNIFORM is bound — removing the shared-uniform last-write-wins
    // hazard when K deviceTopK dispatches share one encoder (each block step
    // calls deviceTopK once). readTopK keeps the uniform path unchanged.
    uniforms: constParams ? {} : { offset: "u32", length: "u32" },
    grid: () => [NW],
    kernel(ctx: KernelContext) {
      const w = ctx.programId(0);
      const tid = ctx.localIndex();
      const off = constParams
        ? ctx.u32(constParams.offset)
        : ctx.uniform("offset");
      const len = constParams
        ? ctx.u32(constParams.length)
        : ctx.uniform("length");
      // chunk = ceil(length / NW). In the const path compute it in JS — folding
      // `len.div(NW)` over u32 CONSTANTS emits a float literal (`4.98u`, invalid
      // WGSL); the uniform path keeps runtime integer division.
      const chunk = constParams
        ? ctx.u32(Math.ceil(constParams.length / NW))
        : len.add(ctx.u32(NW - 1)).div(ctx.u32(NW));
      const start = w.mul(chunk);
      const end = start.add(chunk).min(len);

      // Load this thread's strided slice of the chunk into registers.
      // Element index of private slot s: e = start + s*WG + tid.
      const vals = ctx.emitVarArray("vals", "f32", maxPT, true);
      for (let s = 0; s < maxPT; s++) {
        const e = start.add(ctx.u32(s * WG)).add(tid);
        vals.set(
          ctx.u32(s),
          e.lt(end).select(ctx.load("input", off.add(e)), ctx.f32(NEG_INF)),
        );
      }

      ctx.forRange(ctx.u32(0), ctx.u32(k), (kk) => {
        // Thread-local argmax (ascending s → ascending e, so strict '>' keeps
        // the SMALLEST index among equal values).
        const bv = ctx.emitVar("bv", "f32", ctx.f32(NEG_INF));
        const bi = ctx.emitVar("bi", "u32", ctx.u32(NO_IDX));
        for (let s = 0; s < maxPT; s++) {
          const v = vals.get(ctx.u32(s));
          ctx.ifThen(v.gt(bv.get()), () => {
            bv.set(v);
            bi.set(start.add(ctx.u32(s * WG)).add(tid));
          });
        }
        // Workgroup argmax over the per-thread candidate pairs.
        const win = ctx.pairArgReduce("max", tid, WG, bv.get(), bi.get());
        const winV = ctx.emitLet("winV", win.value);
        const winI = ctx.emitLet("winI", win.index);
        ctx.barrier(); // all lanes read the winner before smem is reused

        const slot = w.mul(ctx.u32(k)).add(kk);
        ctx.guardedStore("pvals", tid.eq(ctx.u32(0)), slot, winV);
        ctx.guardedStore("pidx", tid.eq(ctx.u32(0)), slot, winI);

        // Owner thread removes the winner from its private slice.
        ctx.ifThen(winI.ne(ctx.u32(NO_IDX)), () => {
          const r = winI.sub(start);
          ctx.ifThen(r.mod(ctx.u32(WG)).eq(tid), () => {
            // r / WG is a dynamic slot index into the register array.
            vals.set(r.div(ctx.u32(WG)), ctx.f32(NEG_INF));
          });
        });
      });
    },
  };
}

/**
 * Pass 2: merge NW*K partials → final top-K. Each thread holds a strided
 * register slice of ptTwo (value, ORIGINAL-index) pairs; K rounds of workgroup
 * argmax drain the final top-K. ptTwo = ceil((NW*K)/WG).
 */
function pass2Spec(k: number, packed = false): TileKernelSpec {
  const count = NW * k;
  const ptTwo = Math.ceil(count / WG);
  return {
    name: `topkPass2_k${k}${packed ? "_packed" : ""}`,
    workgroupSize: WG,
    // packed (lazy deviceTopK): ONE f32 output [2k] — values at [0,k), token
    // ids as f32 VALUES at [k,2k) (the index rides as f32 like argmax's output,
    // read natively by a downstream gather). Non-packed (readTopK readback):
    // separate f32 values + u32 indices, unchanged.
    bindings: packed
      ? {
          pvals: { storage: "read", type: "f32" },
          pidx: { storage: "read", type: "u32" },
          out: { storage: "read_write", type: "f32" },
        }
      : {
          pvals: { storage: "read", type: "f32" },
          pidx: { storage: "read", type: "u32" },
          outVals: { storage: "read_write", type: "f32" },
          outIdx: { storage: "read_write", type: "u32" },
        },
    uniforms: {},
    grid: singleWorkgroup(),
    kernel(ctx: KernelContext) {
      const tid = ctx.localIndex();

      // Private copy of this thread's strided slice (value + ORIGINAL index).
      const vals = ctx.emitVarArray("vals", "f32", ptTwo, true);
      const idxs = ctx.emitVarArray("idxs", "u32", ptTwo, true);
      for (let s = 0; s < ptTwo; s++) {
        const e = ctx.u32(s * WG).add(tid);
        const inRange = e.lt(ctx.u32(count));
        vals.set(
          ctx.u32(s),
          inRange.select(ctx.load("pvals", e), ctx.f32(NEG_INF)),
        );
        idxs.set(
          ctx.u32(s),
          inRange.select(ctx.load("pidx", e), ctx.u32(NO_IDX)),
        );
      }

      ctx.forRange(ctx.u32(0), ctx.u32(k), (kk) => {
        const bv = ctx.emitVar("bv", "f32", ctx.f32(NEG_INF));
        const bi = ctx.emitVar("bi", "u32", ctx.u32(NO_IDX));
        for (let s = 0; s < ptTwo; s++) {
          const v = vals.get(ctx.u32(s));
          const i = idxs.get(ctx.u32(s));
          // Tie-break on the ORIGINAL index (all original indices distinct).
          ctx.ifThen(
            v.gt(bv.get()).or(v.eq(bv.get()).and(i.lt(bi.get()))),
            () => {
              bv.set(v);
              bi.set(i);
            },
          );
        }
        const win = ctx.pairArgReduce("max", tid, WG, bv.get(), bi.get());
        const winV = ctx.emitLet("winV", win.value);
        const winI = ctx.emitLet("winI", win.index);
        ctx.barrier();

        if (packed) {
          // values → out[kk]; token id as f32 → out[k + kk].
          ctx.guardedStore("out", tid.eq(ctx.u32(0)), kk, winV);
          ctx.guardedStore(
            "out",
            tid.eq(ctx.u32(0)),
            ctx.u32(k).add(kk),
            winI.toF32(),
          );
        } else {
          ctx.guardedStore("outVals", tid.eq(ctx.u32(0)), kk, winV);
          ctx.guardedStore("outIdx", tid.eq(ctx.u32(0)), kk, winI);
        }

        // Removal: the thread holding the winning ORIGINAL index clears it.
        for (let s = 0; s < ptTwo; s++) {
          ctx.ifThen(
            idxs
              .get(ctx.u32(s))
              .eq(winI)
              .and(winI.ne(ctx.u32(NO_IDX))),
            () => {
              vals.set(ctx.u32(s), ctx.f32(NEG_INF));
            },
          );
        }
      });
    },
  };
}

// ============================================================================
// Compiled kernel cache (WGSL compiled once per (k, maxPT) / k)
// ============================================================================

/** Pass-1 WGSL keyed by `${k}:${maxPT}:${offset|u}:${length|u}`, pass-2 by
 *  `${k}:${packed}`. */
const pass1WGSLCache = new Map<string, string>();
const pass2WGSLCache = new Map<string, string>();

function pass1WGSL(
  k: number,
  maxPT: number,
  constParams?: { offset: number; length: number },
): string {
  const key = `${k}:${maxPT}:${constParams ? `${constParams.offset}:${constParams.length}` : "u"}`;
  let wgsl = pass1WGSLCache.get(key);
  if (!wgsl) {
    wgsl = compileTileKernel(pass1Spec(k, maxPT, constParams));
    pass1WGSLCache.set(key, wgsl);
  }
  return wgsl;
}

function pass2WGSL(k: number, packed = false): string {
  const key = `${k}:${packed}`;
  let wgsl = pass2WGSLCache.get(key);
  if (!wgsl) {
    wgsl = compileTileKernel(pass2Spec(k, packed));
    pass2WGSLCache.set(key, wgsl);
  }
  return wgsl;
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
      // Packed as a TileConfig { offset, length } uniform. tile-IR pads the
      // struct to 16 bytes, so the buffer is 16 bytes (offset, length, pad×2).
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
    paramsData: new Uint32Array(4),
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
  pass1WGSLCache.clear();
  pass2WGSLCache.clear();
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

  const wgsl1 = pass1WGSL(k, maxPT);
  const wgsl2 = pass2WGSL(k);
  const p1 = getPipeline(ctx, wgsl1, wgsl1);
  const p2 = getPipeline(ctx, wgsl2, wgsl2);

  // Bind groups: storage bindings in spec declaration order, then the uniform
  // config (tile-IR appends the config after all storage bindings by default).
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

// ============================================================================
// Lazy device top-k (`deviceTopK`) — the on-device sampling prefilter
// ============================================================================

/**
 * Encode the top-K passes into the shared plan encoder, writing a PACKED f32
 * output `[2k]` (row 0 = the K values descending, row 1 = the K token ids as
 * f32 values). Unlike `readTopK` there is NO readback / flush / mapAsync — the
 * output buffer stays on-device and is consumed by downstream lazy ops (the
 * top-p filter + Gumbel-max selection in decodeBlock). offset/length are baked
 * as WGSL constants (no params uniform), so K per-step dispatches sharing one
 * encoder never collide on a shared uniform. Reuses the SAME pass-1 / pass-2
 * tile-IR kernels (and their tie-break: value desc, smaller index first) that
 * `readTopK` uses, so the device support is byte-identical to the host
 * `sampleFromTopK` reference (which draws its top-k set from `readTopK`).
 *
 * `input` is a contiguous f32 buffer holding the single logits row; `offset`/
 * `length` (elements) select the row. Returns the packed output GPUBuffer.
 */
export function dispatchDeviceTopK(
  input: GPUBuffer,
  offset: number,
  length: number,
  k: number,
): GPUBuffer {
  if (k > length) {
    throw new Error(`deviceTopK: k=${k} exceeds slice length ${length}`);
  }
  const ctx = requireContext();
  const bufs = getOrCreateBuffers(k); // reuse pvals/pidx scratch (per-k)
  const chunk = Math.ceil(length / NW);
  const maxPT = Math.ceil(chunk / WG);

  const wgsl1 = pass1WGSL(k, maxPT, { offset, length });
  const wgsl2 = pass2WGSL(k, true); // packed f32 output
  const p1 = getPipeline(ctx, wgsl1, wgsl1);
  const p2 = getPipeline(ctx, wgsl2, wgsl2);

  const out = allocateOutputBuffer(2 * k * 4); // packed [2k] f32

  const bg1 = cachedCreateBindGroup(ctx.device, p1, [
    input,
    bufs.pvals,
    bufs.pidx,
  ]);
  dispatchComputePass(p1, bg1, NW, 1, 1, "deviceTopKPass1");
  const bg2 = cachedCreateBindGroup(ctx.device, p2, [bufs.pvals, bufs.pidx, out]);
  dispatchComputePass(p2, bg2, 1, 1, 1, "deviceTopKPass2");
  return out;
}

