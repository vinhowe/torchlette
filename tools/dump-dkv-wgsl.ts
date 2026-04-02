/**
 * Dump the WGSL generated for the dKV backward attention kernel.
 */
import { initWebGPU } from "../src/backend/webgpu";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import type { TileKernelSpec } from "../src/backend/webgpu/tile-ir";
import { tiledGrid } from "../src/backend/webgpu/tile-ir";

await initWebGPU();

const BC_BW = 64;
const BQ_BW = 16;

function makeBackwardDKVSpec(headDim: number): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BC_BW;

  return {
    name: `tileAttnBwdDKV_D${D}`,
    workgroupSize: WG,
    autoBarriers: true,
    noTPR: true,
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read_write", type: "f32" },
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

const wgsl = compileTileKernel(makeBackwardDKVSpec(64));
const fs = await import("fs");
fs.writeFileSync("/tmp/dkv-fused.wgsl", wgsl);
console.log("WGSL written to /tmp/dkv-fused.wgsl");
console.log(`Lines: ${wgsl.split("\n").length}`);
process.exit(0);
