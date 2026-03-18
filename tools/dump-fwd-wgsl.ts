/**
 * Dump the WGSL generated for the forward attention kernel.
 */
import { initWebGPU } from "../src/backend/webgpu";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import type { TileKernelSpec } from "../src/backend/webgpu/tile-ir";
import { tiledGrid } from "../src/backend/webgpu/tile-ir";

await initWebGPU();

const BR = 64;
const BC = 32;
const F32_NEG_MAX = -3.4028234663852886e38;

function makeForwardAttentionSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnFwd_D${D}`,
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

const wgsl = compileTileKernel(makeForwardAttentionSpec(64));
const fs = await import("fs");
fs.writeFileSync("/tmp/fwd-attn.wgsl", wgsl);
console.log("WGSL written to /tmp/fwd-attn.wgsl");
console.log(`Lines: ${wgsl.split("\n").length}`);
process.exit(0);
