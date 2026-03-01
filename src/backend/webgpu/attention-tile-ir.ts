/**
 * FlashAttention kernels expressed via tile-IR block API.
 *
 * Produces 4 TileKernelSpec objects that the tile compiler lowers to WGSL.
 * Uses the block-level API: cooperative loads, block dot products (auto-vec4),
 * block reductions, and block stores. No manual vec4 or shared memory code.
 *
 * Forward:      Q,K,V → O,L  (online softmax)
 * D-precompute: dO,O → D     (per-row dot product via wgReduce)
 * Backward dQ:  Q,K,V,L,D,dO → dQ
 * Backward dKV: Q,K,V,L,D,dO → dK,dV
 *
 * Auto-CSE handles all sub-expression sharing — no manual emitLet needed.
 */

import type { TileKernelSpec } from "./tile-ir";
import { tiledGrid } from "./tile-ir";
import { F32_NEG_MAX, WORKGROUP_SIZE } from "./shape-utils";

// Tiling parameters
const BR = 64;     // Q rows per workgroup (forward, dQ)
const BC = 32;     // KV rows per tile (forward, dQ)
const BQ_BW = 16;  // Q rows per tile (backward dKV)
const BC_BW = 64;  // KV rows per workgroup (backward dKV)

// ============================================================================
// Forward Attention Kernel
// ============================================================================

export function makeForwardAttentionSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0) throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR; // One thread per Q row

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
    grid: tiledGrid({ x: { uniform: "seq_len", tileSize: BR }, y: "num_heads", z: "batch_size" }),

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

      // Load Q row → register [1 × D]
      const Q = ctx.tileLoad("Q", {
        kind: "thread", base: qBase, stride: ctx.u32(1),
      }, { rows: 1, cols: D, guard: valid });

      // Online softmax state
      const mPrev = ctx.full(1, 1, F32_NEG_MAX);
      const lPrev = ctx.full(1, 1, 0);
      const oAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        // Cooperative load K tile → shared [BC × D]
        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(bhOff, offsR.outer(Dim), offsD.inner(ctx.u32(1)));
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);

        // Scores = Q @ K^T → register [1 × BC]  (compiler generates vec4 dot)
        const scores = ctx.dot(Q, K.T());

        // Scale and apply causal mask
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid.and(kvPos.lt(N)).and(
            isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)),
          );
          scores.set(j, isActive.select(scores.get(j).mul(scale), ctx.f32(F32_NEG_MAX)));
        });

        // Online softmax update
        const mNew = scores.max(1);          // [1×1]
        const mMax = mNew.max(mPrev);        // [1×1]
        const correction = mPrev.sub(mMax).exp(); // [1×1]

        oAcc.mul_(correction);               // broadcasts [1×1] over [1×D]
        lPrev.mul_(correction);

        scores.sub_(mMax);                   // broadcasts [1×1] over [1×BC]
        scores.exp_();
        lPrev.add_(scores.sum(1));
        mPrev.assign(mMax);

        // Cooperative load V tile → shared [BC × D]
        const V = ctx.load2D("V", tilePtr, tileMask);

        // oAcc += scores @ V  (compiler generates vec4 FMA)
        ctx.dotAccum(scores, V, oAcc);
      });

      // Final normalization and write output
      ctx.ifThen(valid, () => {
        const l = lPrev.get(ctx.u32(0));
        const invL = l.gt(ctx.f32(0)).select(ctx.f32(1).div(l), ctx.f32(0));
        oAcc.mul_(invL);
        ctx.tileStore("O", oAcc, { base: qBase, stride: ctx.u32(1) });

        // Write logsumexp
        const m = mPrev.get(ctx.u32(0));
        const lse = m.add(l.max(ctx.f32(1e-10)).log());
        ctx.emitStore("L", bhOffL.add(qRow), lse);
      });
    },
  };
}

// ============================================================================
// D-Precompute Kernel (per-row dot product via wgReduce)
// ============================================================================

export function makeDPrecomputeSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  const WG = WORKGROUP_SIZE; // 256

  return {
    name: `tileAttnDPrecompute_D${D}`,
    workgroupSize: WG,
    bindings: {
      dO:    { storage: "read", type: "f32" },
      Out:   { storage: "read", type: "f32" },
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

// ============================================================================
// Backward dQ Kernel
// ============================================================================

export function makeBackwardDQSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0) throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnBwdDQ_D${D}`,
    workgroupSize: WG,
    bindings: {
      Q:     { storage: "read", type: "f32" },
      K:     { storage: "read", type: "f32" },
      V:     { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO:    { storage: "read", type: "f32" },
      dQ:    { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      is_causal: "u32",
    },
    grid: tiledGrid({ x: { uniform: "seq_len", tileSize: BR }, y: "num_heads", z: "batch_size" }),

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

      // Load Q and dO rows → register [1 × D]
      const Q = ctx.tileLoad("Q", {
        kind: "thread", base: rowBase, stride: ctx.u32(1),
      }, { rows: 1, cols: D, guard: valid });

      const dO = ctx.tileLoad("dO", {
        kind: "thread", base: rowBase, stride: ctx.u32(1),
      }, { rows: 1, cols: D, guard: valid });

      // Load per-row L and D values
      const lVar = ctx.emitVar("_Li", "f32", ctx.f32(0));
      const dVar = ctx.emitVar("_Di", "f32", ctx.f32(0));
      ctx.ifThen(valid, () => {
        lVar.set(ctx.load("L_buf", bhOffL.add(qRow)));
        dVar.set(ctx.load("D_buf", bhOffL.add(qRow)));
      });

      // Accumulator
      const dqAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        // Cooperative load K and V tiles → shared [BC × D]
        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(bhOff, offsR.outer(Dim), offsD.inner(ctx.u32(1)));
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);
        const V = ctx.load2D("V", tilePtr, tileMask);

        // Scores = Q @ K^T → [1 × BC]   (compiler auto-vec4)
        const scores = ctx.dot(Q, K.T());

        // dO · V^T → [1 × BC]  (compiler auto-vec4)
        const dovs = ctx.dot(dO, V.T());

        // Compute dS per element and apply causal mask
        const ds = ctx.zeros(1, BC);
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid.and(kvPos.lt(N)).and(
            isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)),
          );
          const s = scores.get(j).mul(scale);
          const p = isActive.select(s.sub(lVar.get()).exp(), ctx.f32(0));
          ds.set(j, p.mul(dovs.get(j).sub(dVar.get())));
        });

        // dqAcc += ds @ K → [1×BC] @ [BC×D] → [1×D]  (compiler auto-vec4 FMA)
        ctx.dotAccum(ds, K, dqAcc);
      });

      // Write dQ (with scale applied)
      ctx.ifThen(valid, () => {
        dqAcc.mul_(scale);
        ctx.tileStore("dQ", dqAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}

// ============================================================================
// Backward dKV Kernel
// ============================================================================

export function makeBackwardDKVSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0) throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BC_BW; // One thread per KV row

  return {
    name: `tileAttnBwdDKV_D${D}`,
    workgroupSize: WG,
    bindings: {
      Q:     { storage: "read", type: "f32" },
      K:     { storage: "read", type: "f32" },
      V:     { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO:    { storage: "read", type: "f32" },
      dK:    { storage: "read_write", type: "f32" },
      dV:    { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      is_causal: "u32",
    },
    grid: tiledGrid({ x: { uniform: "seq_len", tileSize: BC_BW }, y: "num_heads", z: "batch_size" }),

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

      // Load K and V rows → register [1 × D]
      const K = ctx.tileLoad("K", {
        kind: "thread", base: rowBase, stride: ctx.u32(1),
      }, { rows: 1, cols: D, guard: valid });

      const V = ctx.tileLoad("V", {
        kind: "thread", base: rowBase, stride: ctx.u32(1),
      }, { rows: 1, cols: D, guard: valid });

      // Accumulators
      const dkAcc = ctx.zeros(1, D);
      const dvAcc = ctx.zeros(1, D);

      // Shared arrays for L and D values within Q tiles
      const lTile = ctx.sharedArray("L_tile", BQ_BW, "f32");
      const dTile = ctx.sharedArray("D_tile", BQ_BW, "f32");

      const numQTiles = N.add(ctx.u32(BQ_BW - 1)).div(ctx.u32(BQ_BW));

      ctx.forRange(ctx.u32(0), numQTiles, (qt) => {
        const qStart = qt.mul(ctx.u32(BQ_BW));

        // Causal tile skip
        const skipTile = isCausal.ne(ctx.u32(0)).and(
          qStart.add(ctx.u32(BQ_BW - 1)).lt(kvBlock.mul(ctx.u32(BC_BW))),
        );

        ctx.ifThen(skipTile.not(), () => {
          // Cooperative load Q and dO tiles → shared [BQ_BW × D]
          const offsR = ctx.arange(qStart, BQ_BW);
          const offsD = ctx.arange(ctx.u32(0), D);
          const tilePtr = ctx.tilePtr(bhOff, offsR.outer(Dim), offsD.inner(ctx.u32(1)));
          const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
          const QTile = ctx.load2D("Q", tilePtr, tileMask);
          const dOTile = ctx.load2D("dO", tilePtr, tileMask);

          // Load L and D values into shared memory
          ctx.ifThen(tidx.lt(ctx.u32(BQ_BW)), () => {
            const qi = qStart.add(tidx);
            const inBounds = qi.lt(N);
            const lIdx = bhOffL.add(qi);
            lTile.write(tidx, inBounds.select(ctx.load("L_buf", lIdx), ctx.f32(0)));
            dTile.write(tidx, inBounds.select(ctx.load("D_buf", lIdx), ctx.f32(0)));
          });

          ctx.barrier();

          // Block dot: scores = K @ QTile^T → [1 × BQ_BW]  (auto-vec4)
          const scores = ctx.dot(K, QTile.T());

          // Block dot: dovs = V @ dOTile^T → [1 × BQ_BW]  (auto-vec4)
          const dovs = ctx.dot(V, dOTile.T());

          // Element-wise: compute ds*scale and p blocks for accumulation
          const dsBlk = ctx.zeros(1, BQ_BW);
          const pBlk = ctx.zeros(1, BQ_BW);
          ctx.range(0, BQ_BW, (j) => {
            const qi = qStart.add(j);
            const isActive = valid.and(qi.lt(N)).and(
              isCausal.eq(ctx.u32(0)).or(kvRow.le(qi)),
            );
            const s = scores.get(j).mul(scale);
            const p = isActive.select(s.sub(lTile.read(j)).exp(), ctx.f32(0));
            const ds = p.mul(dovs.get(j).sub(dTile.read(j)));
            dsBlk.set(j, ds.mul(scale));
            pBlk.set(j, p);
          });

          // Block dot: dkAcc += dsBlk @ QTile → [1×BQ_BW] @ [BQ_BW×D] → [1×D]  (auto-vec4 FMA)
          ctx.dotAccum(dsBlk, QTile, dkAcc);

          // Block dot: dvAcc += pBlk @ dOTile → [1×BQ_BW] @ [BQ_BW×D] → [1×D]  (auto-vec4 FMA)
          ctx.dotAccum(pBlk, dOTile, dvAcc);

          ctx.barrier();
        });
      });

      // Write dK and dV
      ctx.ifThen(valid, () => {
        ctx.tileStore("dK", dkAcc, { base: rowBase, stride: ctx.u32(1) });
        ctx.tileStore("dV", dvAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}
