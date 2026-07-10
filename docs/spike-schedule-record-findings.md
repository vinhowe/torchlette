# Spike: ScheduleRecord lift + naive→flash derivation replay

**Branch:** `spike-schedule-record` (worktree, throwaway). **Device:** GPU 0 (A100-80GB,
clean; 3-9 were all TAINTED with 24-32GB foreign allocs — used the only free device).
**Config under test:** B=2, H=4, N=512, D=64, f32, causal. **Dawn exit-139 after full
output is benign.**

This tests the fifth and hardest lift claim: kernel *schedule* decisions (tile sizes,
residency, loop structure, fusion) baked into `makeForwardAttentionSpec`'s author code
(`src/backend/webgpu/attention-kernel.ts`) can be lifted into a plain-data `ScheduleRecord`
the lowering READS at zero perf cost, and the naive→flash derivation can be walked as ~4
discrete schedule states, each a real runnable WGSL kernel, differentially correct and
measurable.

**Verdict up front: the abstraction level PARTLY exists. The lift is real and free for
the decisions that live *inside one kernel* (tile sizes, KV-block loop factor,
online-softmax on/off, residency choice) — proven byte-identical. But the naive→flash
derivation's most important transitions (S0→S1 tiling of a GEMM, S1→S2 the
materialize→stream lemma, S2→S3 the multi-dispatch→single-kernel fusion) are NOT
`ScheduleRecord` edits of this kernel: they cross kernel and dispatch-count boundaries the
tile-IR treats as fixed structure. The editor as pitched (walk the whole derivation by
nudging one record) would hit a hard wall exactly at the interesting steps.**

---

## PART A — the lift + null test

### Lift inventory (schedule decisions found baked into `makeForwardAttentionSpec`)

| decision | where it lived | lifted into `ScheduleRecord`? |
|---|---|---|
| Q rows per workgroup `BR=64` | module `const`, == `workgroupSize` | yes — `record.br` |
| KV block factor `BC=32` | module `const`, drives `numKVTiles`, `ctx.range(0,BC)`, `arange` | yes — `record.bc` |
| KV-tile residency (workgroup memory) | implicit in `ctx.load2D` (staged) vs `ctx.load` (global) | yes — `record.kvResidency` (see granularity note) |
| softmax mode (online vs materialized) | the running-max/sum block is hardcoded | yes — `record.softmax` (only "online" is expressible; see below) |
| P·V fusion into the same kernel | the whole kernel is one dispatch | **NO** — `record.fusedPV` recorded but a `false` value is not expressible in one `TileKernelSpec` |
| workgroup shape / grid | `tiledGrid({x:{seq_len,BR}, y:heads, z:batch})` | derived from `record.br` |
| vectorization (auto-vec4 in `ctx.dot`) | inside tile-ops, below the spec | not surfaced — lives one tier down |

The lift is `tools/spike-schedule-record.ts`: `makeForwardSpec(headDim, schedule, causal)`
is a line-for-line port of the private factory with every schedule constant replaced by a
`schedule.*` read.

### NULL TEST result: **PASS — RAW BYTE-IDENTICAL**

At `DEFAULT_SCHEDULE = {br:64, bc:32, kvResidency:"shared", softmax:"online", fusedPV:true}`
the lifted kernel's emitted WGSL is **byte-for-byte identical** to the shipped kernel's:

```
[non-causal] RAW BYTE-IDENTICAL (6836 bytes) ✓
[causal]     RAW BYTE-IDENTICAL (6907 bytes) ✓
NULL TEST: PASS
```

No normalization was needed: the spec `name` is a JS-side label / cache key and is never
emitted into the WGSL body (`real.includes("tileAttn") === false`). So this is the strong
claim, not the fallback: the lift adds *zero* bytes to the generated source at default
values → identical shader module → identical pipeline → zero runtime cost. The lift is
free.

(A tiny re-export shim `__spikeMakeForwardAttentionSpec` was added to `attention-kernel.ts`
to diff against the private factory CPU-side — the WGSL compile needs no GPU.)

---

## PART B — the derivation replay

Four discrete states over `softmax(QK^T·scale, causal)·V`, each a real GPU dispatch
sequence (`tools/spike-derivation.ts`). Differential vs a CPU reference (S0 semantics),
median of 30 GPU-wall-clock iters (5 warmup), parity-sanity absolute floor asserted.

| state | ScheduleRecord? | max abs err (vs CPU) | median µs | pred MB (global R+W) |
|---|---|---|---|---|
| **S0 NAIVE** (matmul→softmax→matmul, S materialized [B,H,N,N] global, 3 dispatches) | no — below this kernel | 2.98e-8 | ~1700-1850 | 1109 |
| **S1 TILED** (S0 with shared-mem-staged Q row in the QK GEMM, S still materialized, 3 dispatches) | no — GEMM tiling is a separate kernel | 2.98e-8 | ~1740-1900 | 574 |
| **S2 STREAMED** (one fused kernel, online softmax across KV blocks — the flash lemma) | **YES — `record.bc=16`** | 5.96e-8 | ~305-333 | 4.21 |
| **S3 FUSED** (== shipped kernel) | **YES — `record` DEFAULT `bc=32`** | 5.96e-8 | ~305-330 | 4.21 |

- **All four states compile to real WGSL and run.** (828 / 1082 / 3119 / 788 / 10893 / 6927
  bytes for naiveQK / tiledQK / softmax / naivePV / fusedS2 / fusedS3.)
- **All four are differentially correct** (max abs err ≤ 6e-8 vs the CPU decomposed
  reference — f32 noise, no divergence).
- **parity-sanity holds:** ref RMS = 0.01658, and S0=S1=S2=S3 RMS = 0.01658 — every arm is
  the same *nonzero* magnitude, so no mutual-zero (dropped-submit) fake pass.
- **Rank-match (the cost-model-as-score test):** predicted-bytes ordering
  `S2 < S3 < S1 < S0`. The **coarse tier ordering that matters — fused ≪ materialized —
  matches every run, robustly and dramatically (~5.6x, ~310µs vs ~1800µs)**, exactly as the
  4 MB vs ~1 GB transfer-bytes gap predicts. The *within-tier* fine ordering is
  **noise-dominated**: across runs S0 (pred 1109 MB) and S1 (pred 574 MB) trade places
  (1848 vs 1784µs one run, 1707 vs 1743µs another) — they are within measurement spread of
  each other. **Finding: the naive transfer-bytes model predicts S1 ≪ S0 (halved bytes from
  shared-mem-staging the Q over-read), but the A100 L2 absorbs the repeated Q reads, so the
  saving does not materialize and S1≈S0 within noise. A pure global-R+W byte count is a
  reliable *tier* score but not a reliable *intra-tier* score; a real cost model needs a
  cache-reuse term. The S2/S3 tie is honestly predicted (equal bytes) and honestly measured
  (equal time) — bc only changes shared-mem staging, not global traffic.**

### S3 endpoint vs the shipped kernel

Part A already proved the lift at DEFAULT == shipped **byte-identical**. In the derivation
file, `fusedS3` (causal) emits 6927 bytes vs the shipped kernel's 6907 — a 20-byte delta —
because the derivation file inlines the causal predicate (`kvPos.le(qRow)`) directly, while
the shipped kernel routes causality through the `applySeam("attn_mask", …)` modifier layer
(#64). Functionally identical (err 6e-8, RMS match); the 20 bytes are the seam
indirection. So: **the endpoint of the derivation IS the shipped kernel — byte-exact via
the proper seam-preserving lift (Part A), functionally-exact via the inlined derivation
variant.**

---

## Where move-granularity fought the code structure (the valuable findings)

1. **The materialize→stream lemma (S1→S2) is not a record edit — it's a kernel rewrite.**
   `record.softmax` has only one expressible value: `"online"`. A genuinely
   *materialized-across-all-KV* softmax needs the full [N,N] score matrix resident, which a
   single streaming `TileKernelSpec` structurally cannot hold — the kernel is written as
   "one Q-block streams over KV blocks accumulating running max/sum." Setting
   `softmax:"materialized"` throws. **Streaming without fusing, or fusing without
   streaming, both require choosing a different kernel body, not toggling a field.** The
   record can express *degrees of tiling within the streamed kernel* (any `bc`), but not
   the streamed-vs-materialized axis itself.

2. **Fusion (P·V into the score kernel) is fixed structure, not data.** `record.fusedPV` is
   a decision the tile-IR bakes at spec-authoring time: the fused kernel is one
   `TileKernelSpec` with `ctx.dotAccum(scores, V, oAcc)` inline; an unfused variant is a
   *different number of dispatches with a materialized intermediate*. There is no
   `TileKernelSpec` knob that splits one kernel into three. The dispatch count lives above
   the tile-IR entirely (in the frontend / plan builder).

3. **`kvResidency:"global"` has no primitive.** The tile-IR's block API (`ctx.load2D` →
   staged workgroup memory, `ctx.tileLoad` → register) has no "2D block resident in global,
   read directly by `ctx.dot`" mode — every block dot goes through shared or register
   staging. So the naive/global-resident end of the residency axis (S0) is unreachable from
   *this* kernel's record; S0 had to be authored as separate hand kernels. (The residency
   axis IS reified — `Block.placement: "register"|"shared"`, per the containment analysis —
   but "global" is not one of the reified tiers for a compute block, only for storage.)

4. **The interesting transitions all cross the tile-IR's floor.** S0→S1 (tile a GEMM),
   S1→S2 (materialize→stream), S2→S3 (3-dispatch→1-kernel) each change either the kernel
   body or the dispatch count. The `ScheduleRecord` cleanly parameterizes the *last mile*
   (S2↔S3: tile-size / block-factor within the already-fused-and-streamed kernel), which is
   real and free — but that's the *easy* end of the derivation. The macro-structure moves
   are author decisions the current IR treats as fixed.

---

## What the editor would have felt like

Walking this derivation by hand, the record felt like a *trim knob, not a gearbox*. From
S3 I could slide `bc` (32→16→64) and watch a real, correct, re-timed kernel fall out
instantly — that half of the editor is genuinely there and satisfying: nudge a number,
get a new certified-correct WGSL kernel. But to go from S3 *back* to S2's ancestor S1, or
S1 to S0, I had to put the record down and open a different file: the streaming lemma and
the fusion are welded into the kernel's shape, and the dispatch count lives a tier up. The
editor as pitched — "traverse the whole naive→flash geodesic as schedule states" — would
present a smooth surface near the optimum and a cliff at every macro-structural step. The
lift proves the *decorations* (tile factors, block sizes, residency-within-a-kernel) are
already data; it also shows the *moves* that restructure the kernel (fuse/weave,
materialize↔stream) are still spelled as author code, not gestures.

## What a real ScheduleRecord campaign would need

- A **kernel-shape IR above the tile-IR** where "materialized-S 3-dispatch" and "fused
  streamed 1-dispatch" are two *states of one semantic graph*, so the fuse and the
  stream-lemma are transitions, not rewrites. (This is the "compile-from-IR / islands"
  direction in `docs/architecture-debt.md` — the derivation replay is a concrete stress
  test for it.)
- A **`global` residency primitive** (or an explicit "no-stage" block load) so the naive
  end of the residency axis is reachable from the same kernel template.
- A **cost model with a cache/reuse term**, not just global-R+W bytes — the byte model
  ranks tiers correctly but misranks within a tier (S0 vs S1), which is where an editor's
  "is this edit an improvement?" score would mislead.

## Files (all on branch `spike-schedule-record`, throwaway)

- `tools/spike-schedule-record.ts` — the lift + Part A null test (CPU-only WGSL byte-diff).
- `tools/spike-derivation.ts` — Part B: 4 real GPU kernel states, differential, timing,
  cost model. `SPIKE_COMPILE_ONLY=1` for CPU-only compile check.
- `src/backend/webgpu/attention-kernel.ts` — added `__spikeMakeForwardAttentionSpec`
  re-export shim (remove with the spike).
