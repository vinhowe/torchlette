/**
 * FlashAttention kernels expressed via tile-IR vec4 + subgroup primitives.
 *
 * Produces 4 TileKernelSpec objects that the tile compiler lowers to WGSL.
 * Uses native vec4 arrays (register + shared) and subgroup cooperative dot
 * products (4 threads per row, subgroupShuffleXor reduction) to match the
 * performance of the hand-written WGSL shaders.
 *
 * Forward:     Q,K,V → O,L  (online softmax)
 * D-precompute: dO,O → D    (per-row dot product)
 * Backward dQ: Q,K,V,L,D,dO → dQ
 * Backward dKV: Q,K,V,L,D,dO → dK,dV
 */

import type { TileKernelSpec } from "./tile-ir";
import { tiledGrid, productGrid } from "./tile-ir";
import { getSubgroupSupport } from "./matmul/types";

// Tiling parameters — match production hand-written kernels
const BR = 64;     // Q rows per workgroup
const BC = 32;     // KV rows per tile (forward, dQ)
const BQ_BW = 16;  // Q rows per tile (backward dKV)

// Subgroup cooperative parameters
const THREADS_PER_ROW = 4;

/**
 * Check if subgroup cooperative attention is usable for given headDim.
 * Requires: subgroups feature supported AND headDim/4 divisible by THREADS_PER_ROW
 * AND at least 4 vec4 elements per thread.
 */
function useSubgroupAttention(headDim: number): boolean {
  const sg = getSubgroupSupport();
  if (!sg?.supported) return false;
  const HD4 = headDim / 4;
  const D4_COUNT = HD4 / THREADS_PER_ROW;
  return HD4 % THREADS_PER_ROW === 0 && D4_COUNT >= 4;
}

// ============================================================================
// Forward Attention Kernel
// ============================================================================

export function makeForwardAttentionSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0) throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const HD4 = D / 4;
  const useSg = useSubgroupAttention(D);
  const D4_COUNT = useSg ? HD4 / THREADS_PER_ROW : HD4;
  const WG = useSg ? BR * THREADS_PER_ROW : BR; // 256 or 64

  return {
    name: `tileAttnFwd_D${D}${useSg ? "_sg" : ""}`,
    workgroupSize: WG,
    enableSubgroups: useSg,
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

      // Thread mapping
      const rowLane = useSg ? tidx.div(ctx.u32(THREADS_PER_ROW)) : tidx;
      const d4Lane = useSg ? tidx.mod(ctx.u32(THREADS_PER_ROW)) : ctx.u32(0);
      const d4Start = useSg ? d4Lane.mul(ctx.u32(D4_COUNT)) : ctx.u32(0);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = ctx.emitLet("_qRow", qBlock.mul(ctx.u32(BR)).add(rowLane));
      const valid = ctx.emitLet("_valid", qRow.lt(N));

      const bhOff = ctx.emitLet("_bhOff", bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim));
      const bhOffL = ctx.emitLet("_bhOffL", bIdx.mul(numHeads).add(hIdx).mul(N));

      // Load Q row slice into vec4 registers
      const qReg = ctx.registerVec4Array("q_reg", D4_COUNT);
      ctx.ifThen(valid, () => {
        const qBase = ctx.emitLet("_qBase", bhOff.add(qRow.mul(Dim)));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          const off = ctx.emitLet("_qOff", qBase.add(d4Start.add(d4).mul(ctx.u32(4))));
          qReg.write(d4, ctx.vec4(
            ctx.load("Q", off), ctx.load("Q", off.add(ctx.u32(1))),
            ctx.load("Q", off.add(ctx.u32(2))), ctx.load("Q", off.add(ctx.u32(3))),
          ));
        });
      });

      // Online softmax state
      const mI = ctx.emitVar("m_i", "f32", ctx.f32(-3.402823e+38));
      const lI = ctx.emitVar("l_i", "f32", ctx.f32(0));
      const oAcc = ctx.registerVec4Array("o_acc", D4_COUNT);
      // o_acc is zero-initialized by WGSL default

      // Shared K and V tiles
      const kTile = ctx.sharedVec4Array("k_tile", BC * HD4);
      const vTile = ctx.sharedVec4Array("v_tile", BC * HD4);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = ctx.emitLet("_kvStart", tile.mul(ctx.u32(BC)));

        // Cooperative load K and V tiles (strided: each thread handles WG-strided elements)
        ctx.stridedFor(tidx, ctx.u32(BC * HD4), WG, (idx) => {
          const row = idx.div(ctx.u32(HD4));
          const d4 = idx.mod(ctx.u32(HD4));
          const kvRow = kvStart.add(row);
          const inBounds = kvRow.lt(N);
          const off = ctx.emitLet("_kvOff", bhOff.add(kvRow.mul(Dim)).add(d4.mul(ctx.u32(4))));
          kTile.write(idx, inBounds.select(
            ctx.vec4(ctx.load("K", off), ctx.load("K", off.add(ctx.u32(1))),
                     ctx.load("K", off.add(ctx.u32(2))), ctx.load("K", off.add(ctx.u32(3)))),
            ctx.vec4Splat(ctx.f32(0)),
          ));
          vTile.write(idx, inBounds.select(
            ctx.vec4(ctx.load("V", off), ctx.load("V", off.add(ctx.u32(1))),
                     ctx.load("V", off.add(ctx.u32(2))), ctx.load("V", off.add(ctx.u32(3)))),
            ctx.vec4Splat(ctx.f32(0)),
          ));
        });

        ctx.barrier();

        // Score computation + online softmax
        const tileMax = ctx.emitVar("_tMax", "f32", ctx.f32(-3.402823e+38));
        const scores = ctx.emitVarArray("_scores", "f32", BC);

        ctx.forRange(ctx.u32(0), ctx.u32(BC), (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid.and(kvPos.lt(N)).and(
            isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)),
          );

          // Partial dot product
          const sPartial = ctx.emitVar("_sp", "f32", ctx.f32(0));
          ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
            sPartial.addAssign(
              qReg.read(d4).vec4Dot(kTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4))),
            );
          });

          let sVal = sPartial.get();
          if (useSg) {
            sVal = ctx.emitLet("_s1", sVal.add(sVal.subgroupShuffleXor(1)));
            sVal = ctx.emitLet("_s2", sVal.add(sVal.subgroupShuffleXor(2)));
          }
          const s = ctx.emitLet("_s", sVal.mul(scale));
          const score = isActive.select(s, ctx.f32(-3.402823e+38));
          scores.set(j, score);
          tileMax.set(isActive.select(tileMax.get().max(s), tileMax.get()));
        });

        // Online softmax update
        const mNew = ctx.emitLet("_mNew", mI.get().max(tileMax.get()));
        const correction = ctx.emitLet("_corr", mI.get().sub(mNew).exp());

        lI.set(lI.get().mul(correction));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          oAcc.write(d4, oAcc.read(d4).vec4MulScalar(correction));
        });

        ctx.forRange(ctx.u32(0), ctx.u32(BC), (j) => {
          const p = ctx.emitLet("_p", scores.get(j).sub(mNew).exp());
          lI.addAssign(p);
          ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
            oAcc.addAssign(d4,
              vTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4)).vec4MulScalar(p),
            );
          });
        });

        mI.set(mNew);
        ctx.barrier();
      });

      // Final normalization and write output
      ctx.ifThen(valid, () => {
        const invL = lI.get().gt(ctx.f32(0)).select(
          ctx.f32(1).div(lI.get()), ctx.f32(0));
        const oBase = ctx.emitLet("_oBase", bhOff.add(qRow.mul(Dim)));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          const v = oAcc.read(d4).vec4MulScalar(invL);
          const off = ctx.emitLet("_oOff", oBase.add(d4Start.add(d4).mul(ctx.u32(4))));
          ctx.emitStore("O", off, v.vec4Component(0));
          ctx.emitStore("O", off.add(ctx.u32(1)), v.vec4Component(1));
          ctx.emitStore("O", off.add(ctx.u32(2)), v.vec4Component(2));
          ctx.emitStore("O", off.add(ctx.u32(3)), v.vec4Component(3));
        });

        // Only one thread per row writes L
        if (useSg) {
          ctx.ifThen(d4Lane.eq(ctx.u32(0)), () => {
            const lse = mI.get().add(lI.get().max(ctx.f32(1e-10)).log());
            ctx.emitStore("L", bhOffL.add(qRow), lse);
          });
        } else {
          const lse = mI.get().add(lI.get().max(ctx.f32(1e-10)).log());
          ctx.emitStore("L", bhOffL.add(qRow), lse);
        }
      });
    },
  };
}

// ============================================================================
// D-Precompute Kernel
// ============================================================================

function nextPow2(n: number): number {
  let v = n - 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

export function makeDPrecomputeSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  const wgSize = nextPow2(Math.max(D, 32));

  return {
    name: `tileAttnDPrecompute_D${D}`,
    workgroupSize: wgSize,
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
      scale: "f32",
      is_causal: "u32",
    },
    grid: productGrid("batch_size", "num_heads", "seq_len"),

    kernel(ctx) {
      const tid = ctx.localIndex();
      const rowIdx = ctx.programId(0);
      const Dim = ctx.uniform("head_dim");

      const smem = ctx.sharedArray("_d_smem", wgSize, "f32");
      const base = rowIdx.mul(Dim);
      const inRange = tid.lt(Dim);

      const dOVal = ctx.load("dO", base.add(tid));
      const oVal = ctx.load("Out", base.add(tid));
      const product = dOVal.mul(oVal);

      smem.write(tid, inRange.select(product, ctx.f32(0.0)));
      ctx.barrier();

      // Tree reduction
      ctx.treeReduceSum(smem, tid, wgSize);

      ctx.guardedStore("D_val", tid.eq(ctx.u32(0)), rowIdx, smem.read(ctx.u32(0)));
    },
  };
}

// ============================================================================
// Backward dQ Kernel
// ============================================================================

export function makeBackwardDQSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0) throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const HD4 = D / 4;
  const useSg = useSubgroupAttention(D);
  const D4_COUNT = useSg ? HD4 / THREADS_PER_ROW : HD4;
  const WG = useSg ? BR * THREADS_PER_ROW : BR;

  return {
    name: `tileAttnBwdDQ_D${D}${useSg ? "_sg" : ""}`,
    workgroupSize: WG,
    enableSubgroups: useSg,
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

      const rowLane = useSg ? tidx.div(ctx.u32(THREADS_PER_ROW)) : tidx;
      const d4Lane = useSg ? tidx.mod(ctx.u32(THREADS_PER_ROW)) : ctx.u32(0);
      const d4Start = useSg ? d4Lane.mul(ctx.u32(D4_COUNT)) : ctx.u32(0);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = ctx.emitLet("_qRow", qBlock.mul(ctx.u32(BR)).add(rowLane));
      const valid = ctx.emitLet("_valid", qRow.lt(N));

      const bhOff = ctx.emitLet("_bhOff", bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim));
      const bhOffL = ctx.emitLet("_bhOffL", bIdx.mul(numHeads).add(hIdx).mul(N));

      // Load Q row slice and dO row slice into vec4 registers
      const qReg = ctx.registerVec4Array("q_reg", D4_COUNT);
      const dOReg = ctx.registerVec4Array("dO_reg", D4_COUNT);
      const dqAcc = ctx.registerVec4Array("dq_acc", D4_COUNT);
      const lVar = ctx.emitVar("_Li", "f32", ctx.f32(0));
      const dVar = ctx.emitVar("_Di", "f32", ctx.f32(0));

      ctx.ifThen(valid, () => {
        const base = ctx.emitLet("_qBase", bhOff.add(qRow.mul(Dim)));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          const off = ctx.emitLet("_ld", base.add(d4Start.add(d4).mul(ctx.u32(4))));
          qReg.write(d4, ctx.vec4(
            ctx.load("Q", off), ctx.load("Q", off.add(ctx.u32(1))),
            ctx.load("Q", off.add(ctx.u32(2))), ctx.load("Q", off.add(ctx.u32(3))),
          ));
          dOReg.write(d4, ctx.vec4(
            ctx.load("dO", off), ctx.load("dO", off.add(ctx.u32(1))),
            ctx.load("dO", off.add(ctx.u32(2))), ctx.load("dO", off.add(ctx.u32(3))),
          ));
        });
        lVar.set(ctx.load("L_buf", bhOffL.add(qRow)));
        dVar.set(ctx.load("D_buf", bhOffL.add(qRow)));
      });

      // K and V shared tiles
      const kTile = ctx.sharedVec4Array("k_tile", BC * HD4);
      const vTile = ctx.sharedVec4Array("v_tile", BC * HD4);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = ctx.emitLet("_kvStart", tile.mul(ctx.u32(BC)));

        // Cooperative load K and V tiles (strided)
        ctx.stridedFor(tidx, ctx.u32(BC * HD4), WG, (idx) => {
          const row = idx.div(ctx.u32(HD4));
          const d4 = idx.mod(ctx.u32(HD4));
          const kvRow = kvStart.add(row);
          const inBounds = kvRow.lt(N);
          const off = ctx.emitLet("_kvOff", bhOff.add(kvRow.mul(Dim)).add(d4.mul(ctx.u32(4))));
          kTile.write(idx, inBounds.select(
            ctx.vec4(ctx.load("K", off), ctx.load("K", off.add(ctx.u32(1))),
                     ctx.load("K", off.add(ctx.u32(2))), ctx.load("K", off.add(ctx.u32(3)))),
            ctx.vec4Splat(ctx.f32(0)),
          ));
          vTile.write(idx, inBounds.select(
            ctx.vec4(ctx.load("V", off), ctx.load("V", off.add(ctx.u32(1))),
                     ctx.load("V", off.add(ctx.u32(2))), ctx.load("V", off.add(ctx.u32(3)))),
            ctx.vec4Splat(ctx.f32(0)),
          ));
        });

        ctx.barrier();

        // Score + gradient computation
        ctx.forRange(ctx.u32(0), ctx.u32(BC), (j) => {
          const kvPos = kvStart.add(j);
          const isActive = valid.and(kvPos.lt(N)).and(
            isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)),
          );

          // Partial dot: Q[i] . K[j]
          const sPartial = ctx.emitVar("_sp", "f32", ctx.f32(0));
          ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
            sPartial.addAssign(
              qReg.read(d4).vec4Dot(kTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4))),
            );
          });

          let sVal = sPartial.get();
          if (useSg) {
            sVal = ctx.emitLet("_s1", sVal.add(sVal.subgroupShuffleXor(1)));
            sVal = ctx.emitLet("_s2", sVal.add(sVal.subgroupShuffleXor(2)));
          }
          const s = ctx.emitLet("_s", sVal.mul(scale));
          const p = isActive.select(s.sub(lVar.get()).exp(), ctx.f32(0));

          // Partial dot: dO[i] . V[j]
          const dovPartial = ctx.emitVar("_dp", "f32", ctx.f32(0));
          ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
            dovPartial.addAssign(
              dOReg.read(d4).vec4Dot(vTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4))),
            );
          });

          let dovVal = dovPartial.get();
          if (useSg) {
            dovVal = ctx.emitLet("_dv1", dovVal.add(dovVal.subgroupShuffleXor(1)));
            dovVal = ctx.emitLet("_dv2", dovVal.add(dovVal.subgroupShuffleXor(2)));
          }

          const ds = ctx.emitLet("_ds", p.mul(dovVal.sub(dVar.get())));

          // dQ[i] += ds * K[j]
          ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
            dqAcc.addAssign(d4,
              kTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4)).vec4MulScalar(ds),
            );
          });
        });

        ctx.barrier();
      });

      // Write dQ (each thread writes its slice, with scale applied at write)
      ctx.ifThen(valid, () => {
        const base = ctx.emitLet("_dqBase", bhOff.add(qRow.mul(Dim)));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          const v = dqAcc.read(d4).vec4MulScalar(scale);
          const off = ctx.emitLet("_dqOff", base.add(d4Start.add(d4).mul(ctx.u32(4))));
          ctx.emitStore("dQ", off, v.vec4Component(0));
          ctx.emitStore("dQ", off.add(ctx.u32(1)), v.vec4Component(1));
          ctx.emitStore("dQ", off.add(ctx.u32(2)), v.vec4Component(2));
          ctx.emitStore("dQ", off.add(ctx.u32(3)), v.vec4Component(3));
        });
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
  const HD4 = D / 4;
  const BC_BW = 64;
  const useSg = useSubgroupAttention(D);
  const D4_COUNT = useSg ? HD4 / THREADS_PER_ROW : HD4;
  const WG = useSg ? BC_BW * THREADS_PER_ROW : BC_BW; // 256 or 64

  return {
    name: `tileAttnBwdDKV_D${D}${useSg ? "_sg" : ""}`,
    workgroupSize: WG,
    enableSubgroups: useSg,
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

      const rowLane = useSg ? tidx.div(ctx.u32(THREADS_PER_ROW)) : tidx;
      const d4Lane = useSg ? tidx.mod(ctx.u32(THREADS_PER_ROW)) : ctx.u32(0);
      const d4Start = useSg ? d4Lane.mul(ctx.u32(D4_COUNT)) : ctx.u32(0);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const kvRow = ctx.emitLet("_kvRow", kvBlock.mul(ctx.u32(BC_BW)).add(rowLane));
      const valid = ctx.emitLet("_valid", kvRow.lt(N));

      const bhOff = ctx.emitLet("_bhOff", bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim));
      const bhOffL = ctx.emitLet("_bhOffL", bIdx.mul(numHeads).add(hIdx).mul(N));

      // Load K and V row slices into vec4 registers
      const kReg = ctx.registerVec4Array("k_reg", D4_COUNT);
      const vReg = ctx.registerVec4Array("v_reg", D4_COUNT);
      const dkAcc = ctx.registerVec4Array("dk_acc", D4_COUNT);
      const dvAcc = ctx.registerVec4Array("dv_acc", D4_COUNT);

      ctx.ifThen(valid, () => {
        const base = ctx.emitLet("_kvBase", bhOff.add(kvRow.mul(Dim)));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          const off = ctx.emitLet("_kvOff", base.add(d4Start.add(d4).mul(ctx.u32(4))));
          kReg.write(d4, ctx.vec4(
            ctx.load("K", off), ctx.load("K", off.add(ctx.u32(1))),
            ctx.load("K", off.add(ctx.u32(2))), ctx.load("K", off.add(ctx.u32(3))),
          ));
          vReg.write(d4, ctx.vec4(
            ctx.load("V", off), ctx.load("V", off.add(ctx.u32(1))),
            ctx.load("V", off.add(ctx.u32(2))), ctx.load("V", off.add(ctx.u32(3))),
          ));
        });
      });

      // Shared arrays for Q, dO tiles and L, D values
      const qTile = ctx.sharedVec4Array("q_tile", BQ_BW * HD4);
      const doTile = ctx.sharedVec4Array("dO_tile", BQ_BW * HD4);
      const lTile = ctx.sharedArray("L_tile", BQ_BW, "f32");
      const dTile = ctx.sharedArray("D_tile", BQ_BW, "f32");

      const numQTiles = N.add(ctx.u32(BQ_BW - 1)).div(ctx.u32(BQ_BW));

      ctx.forRange(ctx.u32(0), numQTiles, (qt) => {
        const qStart = ctx.emitLet("_qStart", qt.mul(ctx.u32(BQ_BW)));

        // Causal tile skip: guard body so we skip tiles where max qi < min kv_row
        const skipTile = isCausal.ne(ctx.u32(0)).and(
          qStart.add(ctx.u32(BQ_BW - 1)).lt(kvBlock.mul(ctx.u32(BC_BW))),
        );

        ctx.ifThen(skipTile.not(), () => {
          // Cooperative load Q and dO tiles (strided)
          ctx.stridedFor(tidx, ctx.u32(BQ_BW * HD4), WG, (idx) => {
            const row = idx.div(ctx.u32(HD4));
            const d4 = idx.mod(ctx.u32(HD4));
            const qi = qStart.add(row);
            const inBounds = qi.lt(N);
            const base = ctx.emitLet("_qOff", bhOff.add(qi.mul(Dim)).add(d4.mul(ctx.u32(4))));
            qTile.write(idx, inBounds.select(
              ctx.vec4(ctx.load("Q", base), ctx.load("Q", base.add(ctx.u32(1))),
                       ctx.load("Q", base.add(ctx.u32(2))), ctx.load("Q", base.add(ctx.u32(3)))),
              ctx.vec4Splat(ctx.f32(0)),
            ));
            doTile.write(idx, inBounds.select(
              ctx.vec4(ctx.load("dO", base), ctx.load("dO", base.add(ctx.u32(1))),
                       ctx.load("dO", base.add(ctx.u32(2))), ctx.load("dO", base.add(ctx.u32(3)))),
              ctx.vec4Splat(ctx.f32(0)),
            ));
          });

          // Load L and D values
          ctx.ifThen(tidx.lt(ctx.u32(BQ_BW)), () => {
            const qi = qStart.add(tidx);
            const inBounds = qi.lt(N);
            const lIdx = bhOffL.add(qi);
            lTile.write(tidx, inBounds.select(ctx.load("L_buf", lIdx), ctx.f32(0)));
            dTile.write(tidx, inBounds.select(ctx.load("D_buf", lIdx), ctx.f32(0)));
          });

          ctx.barrier();

          // ALL threads participate in shuffle (subgroup-uniform control flow)
          ctx.forRange(ctx.u32(0), ctx.u32(BQ_BW), (j) => {
            const qi = qStart.add(j);
            const isActive = valid.and(qi.lt(N)).and(
              isCausal.eq(ctx.u32(0)).or(kvRow.le(qi)),
            );

            // Partial dot: Q[qi] . K[kv_row]
            const sPartial = ctx.emitVar("_sp", "f32", ctx.f32(0));
            ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
              sPartial.addAssign(
                qTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4)).vec4Dot(kReg.read(d4)),
              );
            });

            let sVal = sPartial.get();
            if (useSg) {
              sVal = ctx.emitLet("_s1", sVal.add(sVal.subgroupShuffleXor(1)));
              sVal = ctx.emitLet("_s2", sVal.add(sVal.subgroupShuffleXor(2)));
            }
            const s = ctx.emitLet("_s", sVal.mul(scale));
            const p = isActive.select(s.sub(lTile.read(j)).exp(), ctx.f32(0));

            // Partial dot: dO[qi] . V[kv_row]
            const dovPartial = ctx.emitVar("_dp", "f32", ctx.f32(0));
            ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
              dovPartial.addAssign(
                doTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4)).vec4Dot(vReg.read(d4)),
              );
            });

            let dovVal = dovPartial.get();
            if (useSg) {
              dovVal = ctx.emitLet("_dv1", dovVal.add(dovVal.subgroupShuffleXor(1)));
              dovVal = ctx.emitLet("_dv2", dovVal.add(dovVal.subgroupShuffleXor(2)));
            }

            const ds = ctx.emitLet("_ds", p.mul(dovVal.sub(dTile.read(j))));
            const dsScale = ctx.emitLet("_dss", ds.mul(scale));

            // Accumulate dK and dV (each thread accumulates its vec4 slice)
            ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
              dkAcc.addAssign(d4,
                qTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4)).vec4MulScalar(dsScale),
              );
              dvAcc.addAssign(d4,
                doTile.read(j.mul(ctx.u32(HD4)).add(d4Start).add(d4)).vec4MulScalar(p),
              );
            });
          });

          ctx.barrier();
        });
      });

      // Write dK and dV
      ctx.ifThen(valid, () => {
        const base = ctx.emitLet("_dkvBase", bhOff.add(kvRow.mul(Dim)));
        ctx.forRange(ctx.u32(0), ctx.u32(D4_COUNT), (d4) => {
          const off = ctx.emitLet("_dkvOff", base.add(d4Start.add(d4).mul(ctx.u32(4))));
          const dk = dkAcc.read(d4);
          ctx.emitStore("dK", off, dk.vec4Component(0));
          ctx.emitStore("dK", off.add(ctx.u32(1)), dk.vec4Component(1));
          ctx.emitStore("dK", off.add(ctx.u32(2)), dk.vec4Component(2));
          ctx.emitStore("dK", off.add(ctx.u32(3)), dk.vec4Component(3));
          const dv = dvAcc.read(d4);
          ctx.emitStore("dV", off, dv.vec4Component(0));
          ctx.emitStore("dV", off.add(ctx.u32(1)), dv.vec4Component(1));
          ctx.emitStore("dV", off.add(ctx.u32(2)), dv.vec4Component(2));
          ctx.emitStore("dV", off.add(ctx.u32(3)), dv.vec4Component(3));
        });
      });
    },
  };
}
