/**
 * SPIKE PART B — naive→flash derivation replay.
 *
 * Four discrete schedule states over the SAME semantic computation
 * softmax(QK^T · scale, causal) · V, fixed config B=2,H=4,N=512,D=64, f32,
 * causal. Each state is a real runnable WGSL dispatch sequence:
 *
 *   S0 NAIVE     : 3 dispatches — matmul(Q,K^T) → masked-softmax → matmul(P,V).
 *                  S materialized [B,H,N,N] in global memory. Naive (1 thread
 *                  per output elem), everything global-resident.
 *   S1 TILED     : same 3-dispatch materialized-S structure, but the two GEMMs
 *                  use a shared-memory K-blocked tile kernel (blocked matmul).
 *                  Still O(N²) S in global memory.
 *   S2 STREAMED  : ONE fused kernel, online softmax across KV blocks with
 *                  running max/sum corrections — the flash lemma. Expressed via
 *                  ScheduleRecord {bc: 16} (smaller KV block) to show tile-size
 *                  is record-data.
 *   S3 FUSED     : the shipped kernel == ScheduleRecord DEFAULT {bc: 32}.
 *
 * For each state: expressible-via-record?, compiles+runs, differential vs S0
 * CPU reference (max abs err + parity-sanity absolute floor), measured
 * median GPU µs, predicted global-memory bytes, and rank-match.
 */

import { initWebGPU, webgpuBackend } from "../src/backend/webgpu";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import type { KernelContext, TileKernelSpec } from "../src/backend/webgpu/tile-ir";
import { tiledGrid } from "../src/backend/webgpu/tile-ir";

const F32_NEG_MAX = -3.4028234663852886e38;

// Fixed config
const B = 2,
  H = 4,
  N = 512,
  D = 64;
const CAUSAL = true;
const SCALE = 1 / Math.sqrt(D);

// ============================================================================
// ScheduleRecord (mirrors tools/spike-schedule-record.ts) — the lifted data.
// ============================================================================
interface ScheduleRecord {
  br: number;
  bc: number;
  kvResidency: "shared" | "global";
  softmax: "online" | "materialized";
  fusedPV: boolean;
}
const DEFAULT_SCHEDULE: ScheduleRecord = {
  br: 64,
  bc: 32,
  kvResidency: "shared",
  softmax: "online",
  fusedPV: true,
};

// ============================================================================
// Fused forward spec parameterized by ScheduleRecord (S2/S3).
// ============================================================================
function makeFusedForwardSpec(
  headDim: number,
  schedule: ScheduleRecord,
  causal: boolean,
): TileKernelSpec {
  const D = headDim;
  const BR = schedule.br;
  const BC = schedule.bc;
  const WG = BR;
  return {
    name: `spikeFused_D${D}_bc${BC}`,
    workgroupSize: WG,
    autoBarriers: true,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      O: { storage: "read_write", type: "f32" },
      L: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),
    kernel(ctx: KernelContext) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);
      const Nn = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");
      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(Nn);
      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(Nn).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(Nn);
      const qBase = bhOff.add(qRow.mul(Dim));
      const Q = ctx.tileLoad(
        "Q",
        { kind: "thread", base: qBase, stride: ctx.u32(1) },
        { rows: 1, cols: D, guard: valid },
      );
      const mPrev = ctx.full(1, 1, F32_NEG_MAX);
      const lPrev = ctx.full(1, 1, 0);
      const oAcc = ctx.zeros(1, D);
      const numKVTiles = Nn.add(ctx.u32(BC - 1)).div(ctx.u32(BC));
      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));
        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(bhOff, offsR.outer(Dim), offsD.inner(ctx.u32(1)));
        const tileMask = ctx.tileMask(offsR.lt(Nn), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);
        const scores = ctx.dot(Q, K.T());
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const active = causal
            ? valid.and(kvPos.lt(Nn)).and(kvPos.le(qRow))
            : valid.and(kvPos.lt(Nn));
          const s = scores.get(j).mul(scale);
          scores.set(j, active.select(s, ctx.f32(F32_NEG_MAX)));
        });
        const mNew = scores.max(1);
        const mMax = mNew.max(mPrev);
        const correction = mPrev.sub(mMax).exp();
        oAcc.mul_(correction);
        lPrev.mul_(correction);
        scores.sub_(mMax);
        scores.exp_();
        lPrev.add_(scores.sum(1));
        mPrev.assign(mMax);
        const V = ctx.load2D("V", tilePtr, tileMask, { reuseShared: K });
        ctx.dotAccum(scores, V, oAcc);
      });
      ctx.ifThen(valid, () => {
        const l = lPrev.get(ctx.u32(0));
        const invL = l.gt(ctx.f32(0)).select(ctx.f32(1).div(l), ctx.f32(0));
        oAcc.mul_(invL);
        ctx.tileStore("O", oAcc, { base: qBase, stride: ctx.u32(1) });
        const m = mPrev.get(ctx.u32(0));
        const lse = m.add(l.max(ctx.f32(1e-10)).log());
        ctx.emitStore("L", bhOffL.add(qRow), lse);
      });
    },
  };
}

// ============================================================================
// S0/S1 building-block kernels — materialized S path.
//   scores = Q @ K^T * scale, then masked-softmax rows, then P @ V.
// Layout: Q,K,V,O are [B,H,N,D] flat; S,P are [B,H,N,N] flat.
// bh = b*H+h is the batch index; grid over bh in z.
// ============================================================================

/** naive matmul: out[bh, i, j] = sum_d A[bh,i,d]*B2[bh,j,d] (B2 is K, so this
 *  is Q@K^T). one thread per (i,j). global-resident. */
function makeNaiveQKSpec(): TileKernelSpec {
  return {
    name: "spikeNaiveQK",
    workgroupSize: 64,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      S: { storage: "read_write", type: "f32" },
    },
    uniforms: { bh: "u32", n: "u32", d: "u32", scale_u32: "u32" },
    grid: (u) => [Math.ceil((u.bh * u.n * u.n) / 64)],
    kernel(ctx) {
      const gid = ctx.globalId(0);
      const total = ctx.uniform("bh").mul(ctx.uniform("n")).mul(ctx.uniform("n"));
      ctx.ifThen(gid.lt(total), () => {
        const Nn = ctx.uniform("n");
        const Dim = ctx.uniform("d");
        const scale = ctx.uniform("scale_u32").bitcastTo("f32");
        const j = gid.mod(Nn);
        const rest = gid.div(Nn);
        const i = rest.mod(Nn);
        const bh = rest.div(Nn);
        const qBase = bh.mul(Nn).mul(Dim).add(i.mul(Dim));
        const kBase = bh.mul(Nn).mul(Dim).add(j.mul(Dim));
        const acc = ctx.emitVar("_acc", "f32", ctx.f32(0));
        ctx.forRange(ctx.u32(0), Dim, (dd) => {
          acc.set(acc.get().add(ctx.load("Q", qBase.add(dd)).mul(ctx.load("K", kBase.add(dd)))));
        });
        ctx.emitStore("S", gid, acc.get().mul(scale));
      });
    },
  };
}

/** masked row-softmax over S[bh,i,:] (N cols), causal mask j<=i. one workgroup
 *  per row (bh*N rows). writes P in place. */
function makeSoftmaxSpec(causal: boolean): TileKernelSpec {
  const WG = 128;
  return {
    name: `spikeSoftmax_${causal ? "c" : "nc"}`,
    workgroupSize: WG,
    bindings: {
      S: { storage: "read", type: "f32" },
      P: { storage: "read_write", type: "f32" },
    },
    uniforms: { bh: "u32", n: "u32" },
    grid: (u) => [u.bh * u.n],
    kernel(ctx) {
      const rowFlat = ctx.programId(0); // bh*i
      const tid = ctx.localIndex();
      const Nn = ctx.uniform("n");
      const i = rowFlat.mod(Nn);
      const rowBase = rowFlat.mul(Nn);
      const activeCol = (j: import("../src/backend/webgpu/tile-ir").BlockExpr) =>
        causal ? j.le(i) : ctx.u32(1).eq(ctx.u32(1));
      // pass 1: max
      const rowMax = ctx.wgReduce("max", tid, Nn, WG, (j) =>
        activeCol(j).select(ctx.load("S", rowBase.add(j)), ctx.f32(F32_NEG_MAX)),
      );
      const rowSum = ctx.wgReduce("sum", tid, Nn, WG, (j) =>
        activeCol(j).select(ctx.load("S", rowBase.add(j)).sub(rowMax).exp(), ctx.f32(0)),
      );
      const invSum = rowSum.gt(ctx.f32(0)).select(ctx.f32(1).div(rowSum), ctx.f32(0));
      ctx.stridedFor(tid, Nn, WG, (j) => {
        const p = activeCol(j).select(
          ctx.load("S", rowBase.add(j)).sub(rowMax).exp().mul(invSum),
          ctx.f32(0),
        );
        ctx.emitStore("P", rowBase.add(j), p);
      });
    },
  };
}

/** TILED QK (S1): one workgroup per (bh, i). Stages Q[bh,i,:] into shared
 *  memory ONCE, then threads compute S[bh,i,j] over j reusing the shared Q row
 *  — a genuine residency/tiling change (Q read from global D times instead of
 *  N·D times). S still materialized to global. */
function makeTiledQKSpec(): TileKernelSpec {
  const WG = 128;
  return {
    name: "spikeTiledQK",
    workgroupSize: WG,
    autoBarriers: true,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      S: { storage: "read_write", type: "f32" },
    },
    uniforms: { bh: "u32", n: "u32", d: "u32", scale_u32: "u32" },
    grid: (u) => [u.bh * u.n],
    kernel(ctx) {
      const rowFlat = ctx.programId(0); // bh*i
      const tid = ctx.localIndex();
      const Nn = ctx.uniform("n");
      const Dim = ctx.uniform("d");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");
      const bh = rowFlat.div(Nn);
      const i = rowFlat.mod(Nn);
      const qBase = bh.mul(Nn).mul(Dim).add(i.mul(Dim));
      // stage Q row into shared memory
      const qsh = ctx.sharedArray("q_row", D, "f32");
      ctx.stridedFor(tid, Dim, WG, (dd) => qsh.write(dd, ctx.load("Q", qBase.add(dd))));
      ctx.barrier();
      // each thread computes S[bh,i,j] for its strided j
      ctx.stridedFor(tid, Nn, WG, (j) => {
        const kBase = bh.mul(Nn).mul(Dim).add(j.mul(Dim));
        const acc = ctx.emitVar("_acc", "f32", ctx.f32(0));
        ctx.forRange(ctx.u32(0), Dim, (dd) => {
          acc.set(acc.get().add(qsh.read(dd).mul(ctx.load("K", kBase.add(dd)))));
        });
        ctx.emitStore("S", rowFlat.mul(Nn).add(j), acc.get().mul(scale));
      });
    },
  };
}

/** naive P@V: out[bh,i,d] = sum_j P[bh,i,j]*V[bh,j,d]. one thread per (i,d). */
function makeNaivePVSpec(): TileKernelSpec {
  return {
    name: "spikeNaivePV",
    workgroupSize: 64,
    bindings: {
      P: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      O: { storage: "read_write", type: "f32" },
    },
    uniforms: { bh: "u32", n: "u32", d: "u32" },
    grid: (u) => [Math.ceil((u.bh * u.n * u.d) / 64)],
    kernel(ctx) {
      const gid = ctx.globalId(0);
      const Nn = ctx.uniform("n");
      const Dim = ctx.uniform("d");
      const total = ctx.uniform("bh").mul(Nn).mul(Dim);
      ctx.ifThen(gid.lt(total), () => {
        const dd = gid.mod(Dim);
        const rest = gid.div(Dim);
        const i = rest.mod(Nn);
        const bh = rest.div(Nn);
        const pBase = bh.mul(Nn).mul(Nn).add(i.mul(Nn));
        const vBase = bh.mul(Nn).mul(Dim);
        const acc = ctx.emitVar("_acc", "f32", ctx.f32(0));
        ctx.forRange(ctx.u32(0), Nn, (j) => {
          acc.set(acc.get().add(ctx.load("P", pBase.add(j)).mul(ctx.load("V", vBase.add(j.mul(Dim)).add(dd)))));
        });
        ctx.emitStore("O", gid, acc.get());
      });
    },
  };
}

// ============================================================================
// GPU driver helpers
// ============================================================================
let device: GPUDevice;

function makePipeline(spec: TileKernelSpec) {
  const wgsl = compileTileKernel(spec);
  const module = device.createShaderModule({ code: wgsl });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  return { pipeline, wgsl };
}

function buf(sizeBytes: number, extraUsage = 0): GPUBuffer {
  return device.createBuffer({
    size: sizeBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
  });
}
function uniformBuf(words: number[]): GPUBuffer {
  const u = device.createBuffer({
    size: Math.max(16, words.length * 4),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(u, 0, new Uint32Array(words));
  return u;
}
function bindGroup(pipeline: GPUComputePipeline, buffers: GPUBuffer[]): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
}
async function readback(b: GPUBuffer, floats: number): Promise<Float32Array> {
  const rb = device.createBuffer({ size: floats * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(b, 0, rb, 0, floats * 4);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}

const scaleU32 = new Uint32Array(new Float32Array([SCALE]).buffer)[0];
const BH = B * H;

// ============================================================================
// CPU reference (S0 semantics) for the differential.
// ============================================================================
function cpuReference(q: Float32Array, k: Float32Array, v: Float32Array): Float32Array {
  const out = new Float32Array(BH * N * D);
  for (let bh = 0; bh < BH; bh++) {
    for (let i = 0; i < N; i++) {
      const scores = new Float32Array(N).fill(-Infinity);
      for (let j = 0; j < N; j++) {
        if (CAUSAL && j > i) continue;
        let dot = 0;
        for (let d = 0; d < D; d++) dot += q[(bh * N + i) * D + d] * k[(bh * N + j) * D + d];
        scores[j] = dot * SCALE;
      }
      let mx = -Infinity;
      for (let j = 0; j <= (CAUSAL ? i : N - 1); j++) if (scores[j] > mx) mx = scores[j];
      let sum = 0;
      const e = new Float32Array(N);
      for (let j = 0; j <= (CAUSAL ? i : N - 1); j++) { e[j] = Math.exp(scores[j] - mx); sum += e[j]; }
      for (let d = 0; d < D; d++) {
        let acc = 0;
        for (let j = 0; j <= (CAUSAL ? i : N - 1); j++) acc += (e[j] / sum) * v[(bh * N + j) * D + d];
        out[(bh * N + i) * D + d] = acc;
      }
    }
  }
  return out;
}

function maxAbsErr(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
function rms(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s / a.length);
}

// GPU timestamp timing of an arbitrary encoded dispatch sequence.
async function timeDispatches(
  record: (enc: GPUCommandEncoder, tsPass?: { querySet: GPUQuerySet; begin: number; end: number }) => void,
  iters: number,
): Promise<number> {
  const hasTs = device.features.has("timestamp-query");
  // warmup
  for (let w = 0; w < 5; w++) {
    const enc = device.createCommandEncoder();
    record(enc);
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const times: number[] = [];
  if (hasTs) {
    // We time the WHOLE sequence via wall-clock around a fenced submit, since
    // multi-dispatch states can't share one begin/end timestamp pair cleanly.
    // Use CPU wall around a fenced submit — median over many iters is stable
    // for GPU-bound work (the sanity discipline: median + spread reported).
  }
  for (let it = 0; it < iters; it++) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    record(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    times.push((performance.now() - t0) * 1000); // µs
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

async function main() {
  if (process.env.SPIKE_COMPILE_ONLY === "1") {
    for (const [n, s] of [
      ["naiveQK", makeNaiveQKSpec()],
      ["tiledQK", makeTiledQKSpec()],
      ["softmax", makeSoftmaxSpec(CAUSAL)],
      ["naivePV", makeNaivePVSpec()],
      ["fusedS2", makeFusedForwardSpec(D, { ...DEFAULT_SCHEDULE, bc: 16 }, CAUSAL)],
      ["fusedS3", makeFusedForwardSpec(D, DEFAULT_SCHEDULE, CAUSAL)],
    ] as const) {
      const w = compileTileKernel(s);
      console.log(`[compile] ${n}: ${w.length} bytes OK`);
    }
    process.exit(0);
  }
  await initWebGPU();
  device = webgpuBackend.device!;
  console.log(`[spike] device features: timestamp=${device.features.has("timestamp-query")}`);
  console.log(`[spike] config B=${B} H=${H} N=${N} D=${D} causal=${CAUSAL} f32`);

  // Deterministic data
  const qkv = BH * N * D;
  const q = new Float32Array(qkv);
  const k = new Float32Array(qkv);
  const v = new Float32Array(qkv);
  for (let i = 0; i < qkv; i++) {
    q[i] = Math.sin(i * 0.011) * 0.3;
    k[i] = Math.cos(i * 0.013) * 0.3;
    v[i] = Math.sin(i * 0.017 + 1) * 0.3;
  }
  const ref = cpuReference(q, k, v);
  const refRms = rms(ref);
  console.log(`[spike] CPU reference RMS = ${refRms.toFixed(6)} (parity-sanity: nonzero => not a mutual-zero pass)`);

  // Shared GPU input buffers
  const qBuf = buf(qkv * 4), kBuf = buf(qkv * 4), vBuf = buf(qkv * 4);
  device.queue.writeBuffer(qBuf, 0, q);
  device.queue.writeBuffer(kBuf, 0, k);
  device.queue.writeBuffer(vBuf, 0, v);

  // ---- Pipelines ----
  const qkP = makePipeline(makeNaiveQKSpec());
  const tqkP = makePipeline(makeTiledQKSpec()); // S1: shared-mem-staged Q row
  const smP = makePipeline(makeSoftmaxSpec(CAUSAL));
  const pvP = makePipeline(makeNaivePVSpec());
  const fusedS2 = makePipeline(makeFusedForwardSpec(D, { ...DEFAULT_SCHEDULE, bc: 16 }, CAUSAL));
  const fusedS3 = makePipeline(makeFusedForwardSpec(D, DEFAULT_SCHEDULE, CAUSAL));

  // Buffers for materialized-S path
  const sBuf = buf(BH * N * N * 4);
  const pBuf = buf(BH * N * N * 4);
  const oBufNaive = buf(qkv * 4);
  const oBufTiled = buf(qkv * 4);
  const lBuf = buf(BH * N * 4);
  const oBufS2 = buf(qkv * 4);
  const oBufS3 = buf(qkv * 4);

  const uQK = uniformBuf([BH, N, D, scaleU32]);
  const uSM = uniformBuf([BH, N]);
  const uPV = uniformBuf([BH, N, D]);
  const uFused = uniformBuf([B, H, N, D, scaleU32]);

  const bgQK = bindGroup(qkP.pipeline, [qBuf, kBuf, sBuf, uQK]);
  const bgTQK = bindGroup(tqkP.pipeline, [qBuf, kBuf, sBuf, uQK]);
  const bgSM = bindGroup(smP.pipeline, [sBuf, pBuf, uSM]);
  const bgPVnaive = bindGroup(pvP.pipeline, [pBuf, vBuf, oBufNaive, uPV]);
  const bgPVtiled = bindGroup(pvP.pipeline, [pBuf, vBuf, oBufTiled, uPV]);
  const bgS2 = bindGroup(fusedS2.pipeline, [qBuf, kBuf, vBuf, oBufS2, lBuf, uFused]);
  const bgS3 = bindGroup(fusedS3.pipeline, [qBuf, kBuf, vBuf, oBufS3, lBuf, uFused]);

  const gridQK = Math.ceil((BH * N * N) / 64);
  const gridTQK = BH * N; // one workgroup per row
  const gridSM = BH * N;
  const gridPV = Math.ceil((BH * N * D) / 64);

  // ---- Record fns ----
  const recNaive = (enc: GPUCommandEncoder) => {
    let p = enc.beginComputePass();
    p.setPipeline(qkP.pipeline); p.setBindGroup(0, bgQK); p.dispatchWorkgroups(gridQK); p.end();
    p = enc.beginComputePass();
    p.setPipeline(smP.pipeline); p.setBindGroup(0, bgSM); p.dispatchWorkgroups(gridSM); p.end();
    p = enc.beginComputePass();
    p.setPipeline(pvP.pipeline); p.setBindGroup(0, bgPVnaive); p.dispatchWorkgroups(gridPV); p.end();
  };
  const recTiled = (enc: GPUCommandEncoder) => {
    let p = enc.beginComputePass();
    p.setPipeline(tqkP.pipeline); p.setBindGroup(0, bgTQK); p.dispatchWorkgroups(gridTQK); p.end();
    p = enc.beginComputePass();
    p.setPipeline(smP.pipeline); p.setBindGroup(0, bgSM); p.dispatchWorkgroups(gridSM); p.end();
    p = enc.beginComputePass();
    p.setPipeline(pvP.pipeline); p.setBindGroup(0, bgPVtiled); p.dispatchWorkgroups(gridPV); p.end();
  };
  const recFused = (pipeline: GPUComputePipeline, bg: GPUBindGroup, br: number) => (enc: GPUCommandEncoder) => {
    const p = enc.beginComputePass();
    p.setPipeline(pipeline); p.setBindGroup(0, bg);
    p.dispatchWorkgroups(Math.ceil(N / br), H, B); p.end();
  };

  // ---- Correctness: run once, read back, diff ----
  async function runOnce(rec: (enc: GPUCommandEncoder) => void) {
    const enc = device.createCommandEncoder();
    rec(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  await runOnce(recNaive);
  const s0Out = await readback(oBufNaive, qkv);
  await runOnce(recTiled);
  const s1Out = await readback(oBufTiled, qkv);
  await runOnce(recFused(fusedS2.pipeline, bgS2, DEFAULT_SCHEDULE.br));
  const s2Out = await readback(oBufS2, qkv);
  await runOnce(recFused(fusedS3.pipeline, bgS3, DEFAULT_SCHEDULE.br));
  const s3Out = await readback(oBufS3, qkv);

  const errS0 = maxAbsErr(s0Out, ref);
  const errS1 = maxAbsErr(s1Out, ref);
  const errS2 = maxAbsErr(s2Out, ref);
  const errS3 = maxAbsErr(s3Out, ref);

  // ---- Timing ----
  const ITERS = 30;
  const tS0 = await timeDispatches(recNaive, ITERS);
  const tS1 = await timeDispatches(recTiled, ITERS);
  const tS2 = await timeDispatches(recFused(fusedS2.pipeline, bgS2, DEFAULT_SCHEDULE.br), ITERS);
  const tS3 = await timeDispatches(recFused(fusedS3.pipeline, bgS3, DEFAULT_SCHEDULE.br), ITERS);

  // ---- Predicted global-memory bytes (napkin transfer model) ----
  // Reads+writes to GLOBAL memory. f32 = 4 bytes.
  const qkvBytes = BH * N * D * 4;
  const sBytes = BH * N * N * 4;
  // S0 naive: QK reads Q (N times per row! — one thread per (i,j) each re-reads
  //   Q[i,:]) and K, writes S; softmax reads/writes S,P; PV reads P,V writes O.
  //   The dominant term is the materialized S/P (N² floats) traffic.
  //   QK global reads: Q read N·D per row-block (naive), K read N·D per row.
  //   Model the materialized-S traffic (dominant): S written + read (softmax) +
  //   P written + read (PV), plus Q,K,V,O once-ish. Naive Q over-read:
  const predS0 =
    /*QK*/ (N * qkvBytes /*Q re-read ~N× naive*/ + qkvBytes /*K*/ + sBytes /*wr S*/) +
    /*softmax*/ (2 * sBytes) +
    /*PV*/ (sBytes /*rd P*/ + N * qkvBytes /*V re-read ~N× naive*/ + qkvBytes /*wr O*/);
  // S1 tiled: Q staged in shared once → Q read D-per-row not N·D. Materialized S
  //   unchanged. Removes the naive Q/V over-read term. Predict << S0, still >> fused.
  const predS1 =
    /*QK*/ (qkvBytes /*Q once*/ + qkvBytes /*K*/ + sBytes) +
    /*softmax*/ (2 * sBytes) +
    /*PV naive V over-read kept (PV not tiled)*/ (sBytes + N * qkvBytes + qkvBytes);
  // S2/S3 fused streamed: NO materialized S. Each Q-block reads all KV blocks:
  //   reads Q (once), K and V (once each per the streaming loop), writes O + L.
  //   Global bytes ≈ Q + K + V + O = 4*qkvBytes (+ tiny L). Independent of bc
  //   for global traffic (bc changes shared-mem staging, not global reads).
  const predFused = 4 * qkvBytes + BH * N * 4;
  const predS2 = predFused;
  const predS3 = predFused;

  // ---- Report table ----
  const rows = [
    { name: "S0 NAIVE     ", expr: "no (multi-dispatch, below this kernel)", err: errS0, ms: tS0, bytes: predS0 },
    { name: "S1 TILED     ", expr: "no (GEMM shared-mem staging, sep kernel)", err: errS1, ms: tS1, bytes: predS1 },
    { name: "S2 STREAMED  ", expr: "YES (record bc=16)", err: errS2, ms: tS2, bytes: predS2 },
    { name: "S3 FUSED     ", expr: "YES (record DEFAULT bc=32)", err: errS3, ms: tS3, bytes: predS3 },
  ];

  console.log("\n=== PART B: DERIVATION REPLAY TABLE ===");
  console.log("state         | record? | maxAbsErr | median µs | pred MB | ");
  for (const r of rows) {
    console.log(
      `${r.name}| ${r.expr.padEnd(42)} | ${r.err.toExponential(2)} | ${r.ms.toFixed(0).padStart(8)} | ${(r.bytes / 1e6).toFixed(2).padStart(7)}`,
    );
  }

  // ---- Rank match ----
  const measuredRank = [...rows].sort((a, b) => a.ms - b.ms).map((r) => r.name.trim());
  const predictedRank = [...rows].sort((a, b) => a.bytes - b.bytes).map((r) => r.name.trim());
  console.log(`\nmeasured  fastest->slowest: ${measuredRank.join(" < ")}`);
  console.log(`predicted (fewest bytes)  : ${predictedRank.join(" < ")}`);
  // Rank-match on the coarse tiers (fused < materialized), ignoring S0/S1 tie
  // and S2/S3 tie (equal predicted bytes).
  const measuredTier = (n: string) => (n.startsWith("S2") || n.startsWith("S3") ? 0 : 1);
  const tiersMatch =
    measuredRank.slice(0, 2).every((n) => measuredTier(n) === 0) &&
    measuredRank.slice(2).every((n) => measuredTier(n) === 1);
  console.log(`RANK-MATCH (fused tier faster than materialized tier): ${tiersMatch ? "YES" : "NO"}`);

  // ---- parity-sanity absolute floor ----
  const s0rms = rms(s0Out), s1rms = rms(s1Out), s2rms = rms(s2Out), s3rms = rms(s3Out);
  console.log(`\n[parity-sanity] RMS ref=${refRms.toFixed(5)} S0=${s0rms.toFixed(5)} S1=${s1rms.toFixed(5)} S2=${s2rms.toFixed(5)} S3=${s3rms.toFixed(5)}`);
  const sane = [refRms, s0rms, s1rms, s2rms, s3rms].every((r) => r > 1e-3);
  console.log(`[parity-sanity] all arms nonzero (no mutual-zero fake pass): ${sane ? "OK" : "FAIL"}`);

  console.log(`\nDIFFERENTIAL vs S0 CPU ref: S0=${errS0.toExponential(2)} S1=${errS1.toExponential(2)} S2=${errS2.toExponential(2)} S3=${errS3.toExponential(2)}`);
  console.log(`ALL STATES CORRECT (<2e-3): ${[errS0, errS1, errS2, errS3].every((e) => e < 2e-3) ? "YES" : "NO"}`);

  process.exit(0);
}

main();
