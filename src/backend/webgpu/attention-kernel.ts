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
 * Kernel specs use the tile-IR block API: cooperative loads, block dot products
 * (auto-vec4), block reductions, and block stores. No manual vec4 or shared
 * memory code. Auto-CSE handles all sub-expression sharing.
 */

import { cachedCreateBindGroup } from "./bind-group-cache";
import { allocateOutputBuffer } from "./buffer-arena";
import { dispatchComputePass, getPipeline } from "./dispatch";
import type { GPUBuffer, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { F32_NEG_MAX, WORKGROUP_SIZE } from "./shape-utils";
import { compileTileKernel } from "./tile-compiler";
import type { TileKernelSpec } from "./tile-ir";
import { tiledGrid } from "./tile-ir";
import { requireContext, trackSharedEncoderWrite } from "./webgpu-state";

// ============================================================================
// Tiling Parameters
// ============================================================================

const BR = 64; // Q rows per workgroup (forward, dQ)
const BC = 32; // KV rows per tile (forward, dQ)
const BQ_BW = 16; // Q rows per tile (backward dKV)
const BC_BW = 64; // KV rows per workgroup (backward dKV)

// ============================================================================
// WGSL Cache & Config Buffer Cache
// ============================================================================

const tileIRWGSLCache = new Map<string, string>();
function getTileIRWGSL(key: string, specFactory: () => TileKernelSpec): string {
  let wgsl = tileIRWGSLCache.get(key);
  if (!wgsl) {
    wgsl = compileTileKernel(specFactory());
    tileIRWGSLCache.set(key, wgsl);
  }
  return wgsl;
}

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
  const key = `${batchSize}:${numHeads}:${seqLen}:${headDim}:${scale}:${isCausal}`;
  let buf = configCache.get(key);
  if (buf) return buf;

  buf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false,
  });

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
// Kernel Specs (tile-IR)
// ============================================================================

function makeForwardAttentionSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnFwd_D${D}`,
    workgroupSize: WG,
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
      is_causal: "u32",
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const qBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        {
          kind: "thread",
          base: qBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const mPrev = ctx.full(1, 1, F32_NEG_MAX);
      const lPrev = ctx.full(1, 1, 0);
      const oAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);

        const scores = ctx.dot(Q, K.T());

        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid
            .and(kvPos.lt(N))
            .and(isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)));
          scores.set(
            j,
            isActive.select(scores.get(j).mul(scale), ctx.f32(F32_NEG_MAX)),
          );
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

        const V = ctx.load2D("V", tilePtr, tileMask);
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

function makeDPrecomputeSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  const WG = WORKGROUP_SIZE;

  return {
    name: `tileAttnDPrecompute_D${D}`,
    workgroupSize: WG,
    bindings: {
      dO: { storage: "read", type: "f32" },
      Out: { storage: "read", type: "f32" },
      D_val: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      is_causal: "u32",
    },
    grid: (u) => [u.batch_size * u.num_heads * u.seq_len],

    kernel(ctx) {
      const row = ctx.programId(0);
      const tid = ctx.localIndex();
      const Dim = ctx.uniform("head_dim");
      const base = row.mul(Dim);

      const dotProd = ctx.wgReduce("sum", tid, Dim, WG, (i) =>
        ctx.load("dO", base.add(i)).mul(ctx.load("Out", base.add(i))),
      );
      ctx.guardedStore("D_val", tid.eq(ctx.u32(0)), row, dotProd);
    },
  };
}

function makeBackwardDQSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnBwdDQ_D${D}`,
    workgroupSize: WG,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read", type: "f32" },
      dQ: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      is_causal: "u32",
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const rowBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const dO = ctx.tileLoad(
        "dO",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const lVar = ctx.emitVar("_Li", "f32", ctx.f32(0));
      const dVar = ctx.emitVar("_Di", "f32", ctx.f32(0));
      ctx.ifThen(valid, () => {
        lVar.set(ctx.load("L_buf", bhOffL.add(qRow)));
        dVar.set(ctx.load("D_buf", bhOffL.add(qRow)));
      });

      const dqAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);
        const V = ctx.load2D("V", tilePtr, tileMask);

        const scores = ctx.dot(Q, K.T());
        const dovs = ctx.dot(dO, V.T());

        const ds = ctx.zeros(1, BC);
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid
            .and(kvPos.lt(N))
            .and(isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)));
          const s = scores.get(j).mul(scale);
          const p = isActive.select(s.sub(lVar.get()).exp(), ctx.f32(0));
          ds.set(j, p.mul(dovs.get(j).sub(dVar.get())));
        });

        ctx.dotAccum(ds, K, dqAcc);
      });

      ctx.ifThen(valid, () => {
        dqAcc.mul_(scale);
        ctx.tileStore("dQ", dqAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}

function makeBackwardDKVSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BC_BW;

  return {
    name: `tileAttnBwdDKV_D${D}`,
    workgroupSize: WG,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read", type: "f32" },
      dK: { storage: "read_write", type: "f32" },
      dV: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      is_causal: "u32",
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BC_BW },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const kvBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const kvRow = kvBlock.mul(ctx.u32(BC_BW)).add(tidx);
      const valid = kvRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const rowBase = bhOff.add(kvRow.mul(Dim));

      const K = ctx.tileLoad(
        "K",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const V = ctx.tileLoad(
        "V",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const dkAcc = ctx.zeros(1, D);
      const dvAcc = ctx.zeros(1, D);

      const lTile = ctx.sharedArray("L_tile", BQ_BW, "f32");
      const dTile = ctx.sharedArray("D_tile", BQ_BW, "f32");

      const numQTiles = N.add(ctx.u32(BQ_BW - 1)).div(ctx.u32(BQ_BW));

      ctx.forRange(ctx.u32(0), numQTiles, (qt) => {
        const qStart = qt.mul(ctx.u32(BQ_BW));

        const skipTile = isCausal
          .ne(ctx.u32(0))
          .and(qStart.add(ctx.u32(BQ_BW - 1)).lt(kvBlock.mul(ctx.u32(BC_BW))));

        ctx.ifThen(skipTile.not(), () => {
          const offsR = ctx.arange(qStart, BQ_BW);
          const offsD = ctx.arange(ctx.u32(0), D);
          const tilePtr = ctx.tilePtr(
            bhOff,
            offsR.outer(Dim),
            offsD.inner(ctx.u32(1)),
          );
          const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
          const QTile = ctx.load2D("Q", tilePtr, tileMask);
          const dOTile = ctx.load2D("dO", tilePtr, tileMask);

          ctx.ifThen(tidx.lt(ctx.u32(BQ_BW)), () => {
            const qi = qStart.add(tidx);
            const inBounds = qi.lt(N);
            const lIdx = bhOffL.add(qi);
            lTile.write(
              tidx,
              inBounds.select(ctx.load("L_buf", lIdx), ctx.f32(0)),
            );
            dTile.write(
              tidx,
              inBounds.select(ctx.load("D_buf", lIdx), ctx.f32(0)),
            );
          });

          ctx.barrier();

          const scores = ctx.dot(K, QTile.T());
          const dovs = ctx.dot(V, dOTile.T());

          const dsBlk = ctx.zeros(1, BQ_BW);
          const pBlk = ctx.zeros(1, BQ_BW);
          ctx.range(0, BQ_BW, (j) => {
            const qi = qStart.add(j);
            const isActive = valid
              .and(qi.lt(N))
              .and(isCausal.eq(ctx.u32(0)).or(kvRow.le(qi)));
            const s = scores.get(j).mul(scale);
            const p = isActive.select(s.sub(lTile.read(j)).exp(), ctx.f32(0));
            const ds = p.mul(dovs.get(j).sub(dTile.read(j)));
            dsBlk.set(j, ds.mul(scale));
            pBlk.set(j, p);
          });

          ctx.dotAccum(dsBlk, QTile, dkAcc);
          ctx.dotAccum(pBlk, dOTile, dvAcc);

          ctx.barrier();
        });
      });

      ctx.ifThen(valid, () => {
        ctx.tileStore("dK", dkAcc, { base: rowBase, stride: ctx.u32(1) });
        ctx.tileStore("dV", dvAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}

// ============================================================================
// Dispatch Functions
// ============================================================================

/** Shared attention dispatch: config buffer + WGSL cache + pipeline + bind group + tracking. */
function dispatchAttention(
  wgslKey: string,
  pipelinePrefix: string,
  specFactory: () => TileKernelSpec,
  headDim: number,
  batchSize: number,
  numHeads: number,
  seqLen: number,
  scale: number,
  isCausal: boolean,
  buffers: GPUBuffer[], // all data buffers (config appended automatically)
  ...grid: number[]
): void {
  const ctx = requireContext();
  const configBuf = getOrCreateConfigBuffer(
    ctx.device,
    batchSize,
    numHeads,
    seqLen,
    headDim,
    scale,
    isCausal ? 1 : 0,
  );
  const wgsl = getTileIRWGSL(wgslKey, specFactory);
  const pipeline = getPipeline(ctx, `${pipelinePrefix}:tile:${headDim}`, wgsl);
  const bindGroup = cachedCreateBindGroup(ctx.device, pipeline, [
    ...buffers,
    configBuf,
  ]);
  for (const b of buffers) trackSharedEncoderWrite(b);
  dispatchComputePass(pipeline, bindGroup, grid[0], grid[1] ?? 1, grid[2] ?? 1);
}

/** Q,K,V: [B, H, N, D] → O: [B, H, N, D], L: [B, H, N] */
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
  const outBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  const lseBuffer = allocateOutputBuffer(batchSize * numHeads * seqLen * 4);
  dispatchAttention(
    `fwd:${headDim}`,
    "faFwd",
    () => makeForwardAttentionSpec(headDim),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    isCausal,
    [qBuffer, kBuffer, vBuffer, outBuffer, lseBuffer],
    Math.ceil(seqLen / BR),
    numHeads,
    batchSize,
  );
  return { outputBuffer: outBuffer, logsumexpBuffer: lseBuffer };
}

/** dO,O: [B,H,N,D] → D: [B,H,N] */
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
  const outBuffer = allocateOutputBuffer(batchSize * numHeads * seqLen * 4);
  dispatchAttention(
    `bwdD:${headDim}`,
    "faBwdD",
    () => makeDPrecomputeSpec(headDim),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    isCausal,
    [dOBuffer, oBuffer, outBuffer],
    batchSize * numHeads * seqLen,
  );
  return outBuffer;
}

/** Q,K,V,L,D,dO → dQ: [B,H,N,D] */
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
  const outBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  dispatchAttention(
    `bwdDQ:${headDim}`,
    "faBwdDQ",
    () => makeBackwardDQSpec(headDim),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    isCausal,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, outBuffer],
    Math.ceil(seqLen / BR),
    numHeads,
    batchSize,
  );
  return outBuffer;
}

/** Q,K,V,L,D,dO → dK: [B,H,N,D], dV: [B,H,N,D] */
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
  const dKBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  const dVBuffer = allocateOutputBuffer(
    batchSize * numHeads * seqLen * headDim * 4,
  );
  dispatchAttention(
    `bwdDKV:${headDim}`,
    "faBwdDKV",
    () => makeBackwardDKVSpec(headDim),
    headDim,
    batchSize,
    numHeads,
    seqLen,
    scale,
    isCausal,
    [qBuffer, kBuffer, vBuffer, lBuffer, dBuffer, dOBuffer, dKBuffer, dVBuffer],
    Math.ceil(seqLen / BC_BW),
    numHeads,
    batchSize,
  );
  return { dKBuffer, dVBuffer };
}

/** Reset all module-local mutable state (pipeline cache, config buffer cache). */
export function resetAttentionKernelState(): void {
  configCache.clear();
  tileIRWGSLCache.clear();
}
