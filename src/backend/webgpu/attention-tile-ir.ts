/**
 * FlashAttention kernels expressed via tile-IR Block API.
 *
 * Produces 4 TileKernelSpec objects that the tile compiler lowers to WGSL.
 * The Block API determines placement (register vs shared) from the pointer type,
 * and the compiler picks the right dot product lowering (inner vs outer product)
 * from the placement pattern.
 *
 * Forward:     Q,K,V → O,L  (online softmax)
 * D-precompute: dO,O → D    (per-row dot product)
 * Backward dQ: Q,K,V,L,D,dO → dQ
 * Backward dKV: Q,K,V,L,D,dO → dK,dV
 */

import type { TileKernelSpec, TileRangeInfo } from "./tile-ir";
import { tiledGrid, productGrid } from "./tile-ir";
import { BlockOps, type BlockCoopPtr } from "./tile-ops";

// Tiling parameters — match production kernels
const BR = 64;     // Q rows per workgroup
const BC = 32;     // KV rows per tile (forward, dQ)
const BQ_BW = 16;  // Q rows per tile (backward dKV)

// ============================================================================
// Helpers
// ============================================================================

/**
 * Build a cooperative tile pointer for loading a [rows×cols] tile from a
 * flat buffer at `baseOffset`, with `outerStride = cols` (row-major).
 */
function tilePtr(
  ctx: Parameters<TileKernelSpec["kernel"]>[0],
  baseOffset: ReturnType<typeof ctx.u32>,
  rows: number,
  cols: number,
): BlockCoopPtr {
  const outerRange: TileRangeInfo = { base: ctx.u32(0).node, size: rows };
  const innerRange: TileRangeInfo = { base: ctx.u32(0).node, size: cols };
  return {
    kind: "tile",
    baseOffset,
    outerRange,
    innerRange,
    outerStride: ctx.u32(cols),
    innerStride: ctx.u32(1),
    outerBound: ctx.u32(rows),
    innerBound: ctx.u32(cols),
  };
}

// ============================================================================
// Forward Attention Kernel
// ============================================================================

/**
 * FlashAttention forward kernel using Block API.
 *
 * Each thread handles one Q row. KV tiles loaded cooperatively into shared memory.
 * Online softmax with mi/li correction factors.
 *
 * Bindings: Q, K, V (read), O (read_write), L (read_write)
 * Uniforms: batch_size, num_heads, seq_len, head_dim, scale_u32, is_causal
 */
export function makeForwardAttentionSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  const wgSize = BR; // 64

  return {
    name: `tileAttnFwd_D${D}`,
    workgroupSize: wgSize,
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
      scale_u32: "u32",  // bitcast to f32 at use
      is_causal: "u32",
    },
    grid: tiledGrid({ x: { uniform: "seq_len", tileSize: BR }, y: "num_heads", z: "batch_size" }),

    kernel(ctx) {
      const ops = new BlockOps(ctx, { wgSize });
      const tid = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      // Convert is_causal u32 to bool for WGSL bitwise ops compatibility
      const isCausal = ctx.uniform("is_causal").ne(ctx.u32(0));

      // Scale: passed as u32, bitcast to f32
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      // Q row index
      const qRow = qBlock.mul(ctx.u32(BR)).add(tid);
      const valid = qRow.lt(N);

      // Base offset into [B, H, N, D] layout
      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);

      // Load Q row into registers [1×D]
      const q = ops.load("Q",
        { kind: "thread", base: bhOff.add(qRow.mul(Dim)), stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );

      // Online softmax state
      const mi = ops.full(1, 1, -3.402823e+38);   // max so far
      const li = ops.zeros(1, 1);                   // sum of exp so far
      const oAcc = ops.zeros(1, D);                 // output accumulator

      // Number of KV tiles
      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        // K tile [BC×D] in shared memory
        const kBase = bhOff.add(kvStart.mul(Dim));
        const k = ops.load("K", tilePtr(ctx, kBase, BC, D), { rows: BC, cols: D });

        // V tile [BC×D] in shared memory
        const vBase = bhOff.add(kvStart.mul(Dim));
        const v = ops.load("V", tilePtr(ctx, vBase, BC, D), { rows: BC, cols: D });
        ctx.barrier();

        // Scores = Q @ K^T  [1×D] × [BC×D]^T → [1×BC]
        const scores = ops.dot(q, k.T());
        scores.mul_(scale);

        // Causal masking: for each j, if kvStart + j > qRow or >= N, set to -inf
        ctx.forRange(ctx.u32(0), ctx.u32(BC), (j) => {
          const kvIdx = kvStart.add(j);
          const oob = kvIdx.ge(N);
          const causalMask = isCausal.and(kvIdx.gt(qRow));
          const shouldMask = oob.or(causalMask);
          scores.set(ctx.u32(0), j,
            shouldMask.select(ctx.f32(-3.402823e+38), scores.get(ctx.u32(0), j)));
        });

        // Online softmax update
        const tileMax = scores.max(1);            // [1×1]
        const mNew = mi.max(tileMax);             // [1×1]
        const correction = mi.sub(mNew).exp();    // [1×1]: exp(m_old - m_new)

        // Rescale running sums
        li.mul_(correction);
        oAcc.mul_(correction);

        // Softmax numerator
        scores.sub_(mNew);                        // [1×BC] -= [1×1] broadcasts
        scores.exp_();                            // [1×BC] in-place

        // Update running sum
        li.add_(scores.sum(1));

        // Output accumulation: oAcc += P @ V  [1×BC] × [BC×D] → [1×D]
        ops.dotAccum(scores, v, oAcc);

        mi.assign(mNew);
        ctx.barrier();
      });

      // Normalize: O = oAcc / li
      const result = oAcc.div(li);

      // Store output [1×D]
      ops.store("O", result,
        { base: bhOff.add(qRow.mul(Dim)), stride: Dim },
        { guard: valid },
      );

      // Store logsumexp: L[bhN + qRow] = mi + log(li)
      const lse = mi.add(li.log());
      const lIdx = bIdx.mul(numHeads).add(hIdx).mul(N).add(qRow);
      ctx.guardedStore("L", valid, lIdx, lse.get(ctx.u32(0), ctx.u32(0)));
    },
  };
}

// ============================================================================
// D-Precompute Kernel
// ============================================================================

/**
 * D[i] = sum_d(dO[i,d] * O[i,d]) — simple per-row dot product.
 *
 * One workgroup per row. wgSize = max(headDim, 32).
 * Each thread loads 1 element, then shared memory tree reduction.
 */
function nextPow2(n: number): number {
  let v = n - 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

export function makeDPrecomputeSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  // Must be power-of-2 for tree reduction correctness
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
      scale: "f32",       // unused but must match FAConfig layout
      is_causal: "u32",   // unused but must match FAConfig layout
    },
    grid: productGrid("batch_size", "num_heads", "seq_len"),

    kernel(ctx) {
      const tid = ctx.localIndex();
      const rowIdx = ctx.programId(0);
      // Use uniform so config binding stays live in WGSL
      // Note: all 6 uniform fields must be declared to match FAConfig struct layout
      // (batch_size, num_heads, seq_len used only for grid, others unused)
      const Dim = ctx.uniform("head_dim");

      // Shared memory for tree reduction
      const smem = ctx.sharedArray("_d_smem", wgSize, "f32");

      // Each thread handles element tid (wgSize >= D by construction)
      const base = rowIdx.mul(Dim);
      const inRange = tid.lt(Dim);

      // Load dO[row, tid] and O[row, tid], compute product
      const dOVal = ctx.load("dO", base.add(tid));
      const oVal = ctx.load("Out", base.add(tid));
      const product = dOVal.mul(oVal);

      // Write to shared: guarded for threads with tid >= D
      smem.write(tid, inRange.select(product, ctx.f32(0.0)));
      ctx.barrier();

      // Tree reduction in shared memory
      let stride = wgSize >> 1;
      while (stride > 0) {
        ctx.ifThen(tid.lt(ctx.u32(stride)), () => {
          const left = smem.read(tid);
          const right = smem.read(tid.add(ctx.u32(stride)));
          smem.write(tid, left.add(right));
        });
        ctx.barrier();
        stride >>= 1;
      }

      // Thread 0 writes the final result
      ctx.guardedStore("D_val", tid.eq(ctx.u32(0)), rowIdx, smem.read(ctx.u32(0)));
    },
  };
}

// ============================================================================
// Backward dQ Kernel
// ============================================================================

/**
 * Backward dQ kernel — same structure as forward.
 *
 * Each thread handles one Q row. Loops over KV tiles, computes attention scores,
 * then accumulates dQ gradients.
 *
 * dQ[i] = sum_j (dS[i,j] * K[j]) where dS = P * (dP - D[i]) * scale
 * and dP[i,j] = dO[i] @ V[j]^T
 */
export function makeBackwardDQSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  const wgSize = BR;

  return {
    name: `tileAttnBwdDQ_D${D}`,
    workgroupSize: wgSize,
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
      const ops = new BlockOps(ctx, { wgSize });
      const tid = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal").ne(ctx.u32(0));
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tid);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhN = bIdx.mul(numHeads).add(hIdx).mul(N);

      // Load Q row [1×D] and dO row [1×D] into registers
      const q = ops.load("Q",
        { kind: "thread", base: bhOff.add(qRow.mul(Dim)), stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );
      const dO_row = ops.load("dO",
        { kind: "thread", base: bhOff.add(qRow.mul(Dim)), stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );

      // Load L[i] and D[i] scalars (out-of-bounds reads return 0 via robustness)
      const lVal = ctx.emitLet("_lVal", ctx.load("L_buf", bhN.add(qRow)));
      const dVal = ctx.emitLet("_dVal", ctx.load("D_buf", bhN.add(qRow)));

      // dQ accumulator [1×D]
      const dqAcc = ops.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        // K tile [BC×D], V tile [BC×D] in shared
        const kBase = bhOff.add(kvStart.mul(Dim));
        const k = ops.load("K", tilePtr(ctx, kBase, BC, D), { rows: BC, cols: D });

        const vBase = bhOff.add(kvStart.mul(Dim));
        const v = ops.load("V", tilePtr(ctx, vBase, BC, D), { rows: BC, cols: D });
        ctx.barrier();

        // Scores = Q @ K^T  [1×BC]
        const scores = ops.dot(q, k.T());
        scores.mul_(scale);

        // Apply masking (same as forward)
        ctx.forRange(ctx.u32(0), ctx.u32(BC), (j) => {
          const kvIdx = kvStart.add(j);
          const oob = kvIdx.ge(N);
          const causalMask = isCausal.and(kvIdx.gt(qRow));
          const shouldMask = oob.or(causalMask);
          scores.set(ctx.u32(0), j,
            shouldMask.select(ctx.f32(-3.402823e+38), scores.get(ctx.u32(0), j)));
        });

        // P = exp(scores - L[i])  — use saved logsumexp for numerical stability
        scores.sub_(lVal);
        scores.exp_();

        // dP = dO @ V^T  [1×D] × [BC×D]^T → [1×BC]
        const dP = ops.dot(dO_row, v.T());

        // dS = P * (dP - D[i]) * scale
        dP.sub_(dVal);          // dP -= D[i]
        dP.mul_(scores);        // dP *= P  → now contains P * (dP - D[i])
        dP.mul_(scale);         // dP *= scale → now contains dS

        // dQ += dS @ K  [1×BC] × [BC×D] → [1×D]
        ops.dotAccum(dP, k, dqAcc);
        ctx.barrier();
      });

      // Store dQ
      ops.store("dQ", dqAcc,
        { base: bhOff.add(qRow.mul(Dim)), stride: Dim },
        { guard: valid },
      );
    },
  };
}

// ============================================================================
// Backward dKV Kernel
// ============================================================================

/**
 * Backward dK/dV kernel — reversed: KV in registers, Q/dO in shared.
 *
 * Each thread handles one KV row. Loops over Q tiles (BQ_BW rows each).
 * dK[j] = sum_i (dS[i,j] * Q[i])
 * dV[j] = sum_i (P[i,j] * dO[i])
 */
export function makeBackwardDKVSpec(headDim: number): TileKernelSpec {
  const D = headDim;
  // dKV uses 64 KV rows per workgroup
  const BC_BW = 64;
  const wg = BC_BW;

  return {
    name: `tileAttnBwdDKV_D${D}`,
    workgroupSize: wg,
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
      const ops = new BlockOps(ctx, { wgSize: wg });
      const tid = ctx.localIndex();
      const kvBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const isCausal = ctx.uniform("is_causal").ne(ctx.u32(0));
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const kvRow = kvBlock.mul(ctx.u32(BC_BW)).add(tid);
      const valid = kvRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhN = bIdx.mul(numHeads).add(hIdx).mul(N);

      // Load K row [1×D] and V row [1×D] into registers
      const k = ops.load("K",
        { kind: "thread", base: bhOff.add(kvRow.mul(Dim)), stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );
      const v = ops.load("V",
        { kind: "thread", base: bhOff.add(kvRow.mul(Dim)), stride: Dim },
        { rows: 1, cols: D, guard: valid },
      );

      // Accumulators
      const dkAcc = ops.zeros(1, D);
      const dvAcc = ops.zeros(1, D);

      // Shared arrays for L and D tiles (declared before loop, persist across iterations)
      const lSmem = ctx.sharedArray("_l_tile", BQ_BW, "f32");
      const dSmem = ctx.sharedArray("_d_tile", BQ_BW, "f32");

      const numQTiles = N.add(ctx.u32(BQ_BW - 1)).div(ctx.u32(BQ_BW));

      ctx.forRange(ctx.u32(0), numQTiles, (tile) => {
        const qStart = tile.mul(ctx.u32(BQ_BW));

        // Q tile [BQ_BW×D] in shared
        const qBase = bhOff.add(qStart.mul(Dim));
        const qTile = ops.load("Q", tilePtr(ctx, qBase, BQ_BW, D), { rows: BQ_BW, cols: D });

        // dO tile [BQ_BW×D] in shared
        const doBase = bhOff.add(qStart.mul(Dim));
        const doTile = ops.load("dO", tilePtr(ctx, doBase, BQ_BW, D), { rows: BQ_BW, cols: D });

        // Cooperative load of L[qStart..qStart+BQ_BW] and D[qStart..qStart+BQ_BW]
        // wg=64 threads, BQ_BW=16 elements: threads 0-15 each load 1 element
        ctx.ifThen(tid.lt(ctx.u32(BQ_BW)), () => {
          const qIdx = qStart.add(tid);
          const lIdx = bhN.add(qIdx);
          const inBounds = qIdx.lt(N);
          lSmem.write(tid, inBounds.select(ctx.load("L_buf", lIdx), ctx.f32(0.0)));
          dSmem.write(tid, inBounds.select(ctx.load("D_buf", lIdx), ctx.f32(0.0)));
        });

        ctx.barrier();

        // Scores = K @ Q^T  [1×D] × [BQ_BW×D]^T → [1×BQ_BW]
        const scores = ops.dot(k, qTile.T());
        scores.mul_(scale);

        // Apply masking and compute P for each j (Q row index)
        ctx.forRange(ctx.u32(0), ctx.u32(BQ_BW), (j) => {
          const qIdx = qStart.add(j);
          const oob = qIdx.ge(N);
          // Causal: Q row qIdx can only attend to KV rows <= qIdx
          // So mask if kvRow > qIdx
          const causalMask = isCausal.and(kvRow.gt(qIdx));
          const shouldMask = oob.or(causalMask);

          // Read L[qIdx] from shared
          const ljVal = lSmem.read(j);

          // P[i,j] = exp(score[j] - L[j]), masked → 0
          const rawScore = scores.get(ctx.u32(0), j);
          const maskedScore = shouldMask.select(ctx.f32(-3.402823e+38), rawScore);
          scores.set(ctx.u32(0), j, maskedScore.sub(ljVal).exp());
        });

        // dV += P @ dO  [1×BQ_BW] × [BQ_BW×D] → [1×D]
        ops.dotAccum(scores, doTile, dvAcc);

        // dP = V @ dO^T ... actually we need: for each j, dP[j] = dot(V, dO[j])
        // Compute dP via dot: V[1×D] @ dO[BQ_BW×D]^T → [1×BQ_BW]
        const dP = ops.dot(v, doTile.T());

        // Compute dS: for each j: dS[j] = P[j] * (dP[j] - D_tile[j]) * scale
        ctx.forRange(ctx.u32(0), ctx.u32(BQ_BW), (j) => {
          const pj = scores.get(ctx.u32(0), j);
          const dpj = dP.get(ctx.u32(0), j);
          const djVal = dSmem.read(j);
          const dsj = pj.mul(dpj.sub(djVal)).mul(scale);
          dP.set(ctx.u32(0), j, dsj);  // reuse dP storage for dS
        });

        // dK += dS @ Q  [1×BQ_BW] × [BQ_BW×D] → [1×D]
        ops.dotAccum(dP, qTile, dkAcc);
        ctx.barrier();
      });

      // Store dK and dV
      ops.store("dK", dkAcc,
        { base: bhOff.add(kvRow.mul(Dim)), stride: Dim },
        { guard: valid },
      );
      ops.store("dV", dvAcc,
        { base: bhOff.add(kvRow.mul(Dim)), stride: Dim },
        { guard: valid },
      );
    },
  };
}
