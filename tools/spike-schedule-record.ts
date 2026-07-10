/**
 * SPIKE (throwaway, timeboxed): ScheduleRecord lift + naive→flash derivation replay.
 *
 * Tests the claim: kernel *schedule* decisions (tile sizes, residency, loop
 * structure, fusion) baked into makeForwardAttentionSpec's author code can be
 * lifted into a plain-data ScheduleRecord that the tile-IR lowering READS,
 * whose DEFAULT values reproduce today's exact WGSL (byte-identical) at zero
 * perf cost — and that the naive→flash derivation can be walked as ~4 discrete
 * schedule states, each compiling to a real runnable WGSL kernel.
 *
 * PART A: the lift + null test (WGSL byte-diff, CPU-only — no GPU needed).
 * PART B: the derivation replay (GPU: differential vs S0, timing, cost model).
 */

import { F32_NEG_MAX } from "../src/backend/webgpu/shape-utils";
import { compileTileKernel } from "../src/backend/webgpu/tile-compiler";
import { tiledGrid } from "../src/backend/webgpu/tile-ir";
import type { KernelContext, TileKernelSpec } from "../src/backend/webgpu/tile-ir";

// ============================================================================
// THE LIFT: ScheduleRecord — schedule decisions as manipulable DATA
//
// Every field below was a compile-time constant or an author choice hardwired
// into makeForwardAttentionSpec's body (attention-kernel.ts). The DEFAULT
// object reproduces today's exact forward kernel.
// ============================================================================

/** Residency of the KV tile: how K/V blocks are held during the KV loop.
 *  - "shared": cooperative load into workgroup memory (today's default; ctx.load2D)
 *  - "global": no staging — each dot reads K/V straight from global (naive) */
type KVResidency = "shared" | "global";

/** How the score matrix is reduced/normalized across the KV dimension.
 *  - "online": FlashAttention running max/sum with corrections (today's default)
 *  - "materialized": full-row softmax after all scores computed (needs S in memory) */
type SoftmaxMode = "online" | "materialized";

interface ScheduleRecord {
  /** Q rows per workgroup (== workgroup size for the forward kernel). Was `BR = 64`. */
  br: number;
  /** KV rows per tile — the KV-loop blocking factor. Was `BC = 32`. */
  bc: number;
  /** Where the KV tile lives during the inner loop. Was implicitly "shared". */
  kvResidency: KVResidency;
  /** Softmax reduction strategy across KV blocks. Was implicitly "online". */
  softmax: SoftmaxMode;
  /** Whether V-accumulation is fused into the same kernel as the score/softmax
   *  (single kernel) vs split. Was implicitly true (one fused kernel). Note:
   *  a false value here is NOT expressible within one TileKernelSpec — see
   *  findings (granularity). Kept as a field to record the decision. */
  fusedPV: boolean;
}

/** DEFAULT === today's shipped forward kernel. Null-test target. */
const DEFAULT_SCHEDULE: ScheduleRecord = {
  br: 64,
  bc: 32,
  kvResidency: "shared",
  softmax: "online",
  fusedPV: true,
};

// ============================================================================
// Parameterized forward spec — the lowering now READS the ScheduleRecord.
// A byte-for-byte port of makeForwardAttentionSpec with every schedule
// constant replaced by a schedule.* read. No modifier support (spike scope:
// the null modifier / non-causal + causal-as-record cases only).
// ============================================================================

function makeForwardSpec(
  headDim: number,
  schedule: ScheduleRecord,
  causal: boolean,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const BR = schedule.br;
  const BC = schedule.bc;
  const WG = BR;

  return {
    name: `spikeAttnFwd_D${D}`,
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

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const qBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        { kind: "thread", base: qBase, stride: ctx.u32(1) },
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
        // RESIDENCY SCHEDULE DECISION: shared (staged) vs global (naive).
        // load2D stages into workgroup memory; loadGlobal2D reads straight
        // from global. Both produce a Block usable by ctx.dot.
        const K =
          schedule.kvResidency === "shared"
            ? ctx.load2D("K", tilePtr, tileMask)
            : ctx.load2D("K", tilePtr, tileMask); // see findings: no global-resident 2D block primitive

        const scores = ctx.dot(Q, K.T());

        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const active = causal
            ? valid.and(kvPos.lt(N)).and(kvPos.le(qRow))
            : valid.and(kvPos.lt(N));
          const s = scores.get(j).mul(scale);
          scores.set(j, active.select(s, ctx.f32(F32_NEG_MAX)));
        });

        if (schedule.softmax === "online") {
          const mNew = scores.max(1);
          const mMax = mNew.max(mPrev);
          const correction = mPrev.sub(mMax).exp();
          oAcc.mul_(correction);
          lPrev.mul_(correction);
          scores.sub_(mMax);
          scores.exp_();
          lPrev.add_(scores.sum(1));
          mPrev.assign(mMax);
          const V =
            schedule.kvResidency === "shared"
              ? ctx.load2D("V", tilePtr, tileMask, { reuseShared: K })
              : ctx.load2D("V", tilePtr, tileMask, { reuseShared: K });
          ctx.dotAccum(scores, V, oAcc);
        } else {
          // materialized softmax within one KV tile only == same as online for
          // a single pass; genuine materialized-across-all-KV needs S in memory
          // which one streaming kernel cannot express — see findings.
          throw new Error("materialized softmax not expressible in one kernel");
        }
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
// PART A — NULL TEST: schedule-lifted WGSL at DEFAULT vs the real kernel.
// ============================================================================

// Re-import the real forward spec factory to diff against. It's not exported,
// so we reconstruct the real kernel via the public dispatch's WGSL cache would
// require a GPU; instead we import the module and read its emitted WGSL by
// calling the private factory through a tiny re-export shim added below.
import { __spikeMakeForwardAttentionSpec } from "../src/backend/webgpu/attention-kernel";

function nullTest(): { ok: boolean; report: string } {
  const D = 64;
  const causalCases: Array<[string, boolean]> = [
    ["non-causal", false],
    ["causal", true],
  ];
  const lines: string[] = [];
  let allOk = true;
  for (const [label, causal] of causalCases) {
    const mod = causal ? { maskMods: [{ kind: "causal" as const }] } : undefined;
    const realWGSL = compileTileKernel(
      __spikeMakeForwardAttentionSpec(D, mod),
    );
    const liftedWGSL = compileTileKernel(
      makeForwardSpec(D, DEFAULT_SCHEDULE, causal),
    );
    // The spec `name` is NOT emitted into the WGSL body (it's only a JS-side
    // label / cache-key fragment), so a raw byte comparison is the honest test.
    const a = realWGSL;
    const b = liftedWGSL;
    const ok = a === b;
    allOk = allOk && ok;
    if (ok) {
      lines.push(`  [${label}] RAW BYTE-IDENTICAL (${a.length} bytes) ✓`);
    } else {
      // find first differing line
      const al = a.split("\n");
      const bl = b.split("\n");
      let diffAt = -1;
      for (let i = 0; i < Math.max(al.length, bl.length); i++) {
        if (al[i] !== bl[i]) {
          diffAt = i;
          break;
        }
      }
      lines.push(`  [${label}] DIFFERS at line ${diffAt}:`);
      lines.push(`    real:   ${JSON.stringify(al[diffAt])}`);
      lines.push(`    lifted: ${JSON.stringify(bl[diffAt])}`);
      lines.push(`    (real ${al.length} lines, lifted ${bl.length} lines)`);
    }
  }
  return { ok: allOk, report: lines.join("\n") };
}

function main() {
  console.log("=== PART A: NULL TEST (WGSL byte-diff, CPU-only) ===");
  const { ok, report } = nullTest();
  console.log(report);
  console.log(`NULL TEST: ${ok ? "PASS" : "FAIL"}`);
}

main();
