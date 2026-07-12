# Schedule-State P1 — MATMUL family CUTOVER + R9 registry migration

**Branch:** `worktree-agent-a346c661dd6ff9135` (off `main` @ `9d8e0edc`). NOT pushed.
**Scope delivered:** the matmul family's LIVE lowered structure now flows through
`deriveScheduleState → applySchedule` (the schedule object is the sole WGSL writer
at the dispatch seam); the R9 `SelectionReceipt` is a first-class object every
consuming path reads; the #95 `GemvDescriptor.inputCastA/B` axis is closed.
**Scope STOPPED (net-negative deletion):** the §11 structural-generator deletions
for the tiled family did NOT go net-negative and are STOPPED — see §3, with the
architectural reason. Item 1 (route through the schedule object) and item 2 (R9)
are complete; item 3 is honestly deferred to a compile-from-IR follow-on.

---

## 1. Per-family cutover status

| Sub-family | Live WGSL writer BEFORE | Live WGSL writer AFTER | Differential guards live path |
|---|---|---|---|
| tiled (NN/NT/TN/TT, f32/f16/mixed/inputCast, big-tile) | `getOrCreatePipeline` → `generateTiledMatmulShaderTileIR(opts)` | `getOrCreatePipeline` → `compileTileKernel(realizeTiledMatmulKernel(opts))` → `deriveTiledMatmulState`+`applyTiledMatmulSchedule` | ✅ (8 cases) |
| epilogue (bias / gelu / bias+gelu+cast / residual) | same tiled path | same tiled path (epilogue rides `CodegenOptions`) | ✅ (4 cases) |
| batched | same tiled path | same tiled path | ✅ (1) |
| swapGrid | same tiled path, `swapGrid` flag | same tiled path; ProgramGridMap `swap` reified in derived state | ✅ (1) |
| split-K partials | tiled path, `kSplit=N` | same tiled path via schedule | ✅ (in tiled 8) |
| split-K reduction pass | `getOrCreateReductionPipeline` → `generateKSplitReductionShaderTileIR` | `getOrCreateReductionPipeline` → `realizeKSplitReductionWgsl` → `kSplitReductionWgsl` | ✅ (3) |
| GEMV (NT/NN + epilogue + quantB + inputCast) | `planGemvRowMatmul` / `dispatchQuantizedGemvNT` → `generateGemvShaderTileIR` | both → `realizeGemvWgsl` → `deriveGemvState`+`applyGemvSchedule` | ✅ (8) |

**The seam is one chokepoint.** Every live path — lowered `dispatchTiledMatmul`,
build-from-IR `planBareMatmul` capture, and the stage-4 stream generators
`generateBareMatmul`/`generateMatmulEpilogue` — funnels through `planTiledMatmul`,
which is the sole caller of the pipeline builders (`getOrCreatePipeline`,
`planGemvRowMatmul`, `getOrCreateReductionPipeline`). Cutting those three builders
over to `realize*` cuts over ALL live paths at once. No path calls a raw
`generate*ShaderTileIR` on the live path anymore (grep-verified: the only
remaining `generate*` calls in src are inside the skeleton's `apply*`).

**Byte-identity is structural.** `applyTiledMatmulSchedule` funnels back to the
same `createTiledMatmulKernel`; `applyGemvSchedule`/`kSplitReductionWgsl` funnel
back to `generateGemvShaderTileIR`/`generateKSplitReductionShaderTileIR`. So the
25-kernel matmul differential (`compileTileKernel(applyTiledMatmulSchedule) ==
generateTiledMatmulShaderTileIR`, etc.) proves the derived state regenerates the
old WGSL BYTE-FOR-BYTE — and now that the live path IS `apply*`, the differential
guards the LIVE path, not a side-by-side.

---

## 2. R9 migration proof table + SelectionReceipt

The matmul registry (`variants.ts`) already decomposes into R9's four things; P1
adds the `SelectionReceipt` object and routes engagement through it.

| R9 concept | Where it lives (file:symbol) | Migrated form |
|---|---|---|
| AlgorithmFamily | `MatmulVariantChoice["variant"]` = `"tiled" \| "gemv"` | unchanged; recorded as `SelectionReceipt.family` |
| ScheduleTemplate | `MatmulVariant.defaultChoice` / `.candidates` (tiled config space; gemv wg/rows space) | unchanged; the picked instance is `SelectionReceipt.choice` |
| ApplicabilityPredicate | `gemvVariant.isApplicable` / `tiledVariant.isApplicable` (geometry/dtype/epilogue/cast/pin/subgroup) | unchanged; the predicate selects `family` |
| RealizerRequirement | `realizeTiledMatmulKernel` / `realizeGemvWgsl` / `realizeKSplitReductionWgsl` | NEW — the schedule-object realizer entry points |
| SelectionReceipt | `variants.ts` `SelectionReceipt`; stamped in `planTiledMatmul` | NEW — `{family, choice, gemvEngaged, gemvEpilogueEngaged}` carried on the plan |

**Field-by-field round-trip (differential-proven, report §7 gate matrix):**
- **gemv** — mode/dtypeA/dtypeB/outputDtype/kSplit/wgSize/rowsPerWg/vec4/epilogue/
  quantB/**inputCastA/B** all round-trip through `deriveGemvState`→`applyGemvSchedule`
  (8 cases). The #95 `inputCastA/B` axis is now on `GemvDescriptor` and threaded into
  the A/B NamedValue dtype (stored-wider) + `applyGemvSchedule`'s options — closing the
  twice-flagged cast-blindness at the schedule-derivation layer.
- **tiled** — config (tileM/N/K, threadTile, vec, subgroups)/transposeMode/dtype/
  dtypeB/epilogue/batched/inputCastA/B/kSplit/swapGrid round-trip (8 cases).
- **split-K** — kSplit as the fp-reorder LemmaApplication; the reduction pass as
  family 3b (3 cases).
- **swapGrid** — reifies to `ProgramGridMap {kind:"swap", axes:[m,n]}` (R4).
- **epilogues** — bias/unary/binary/cast as no-materialization body edges (4 cases).

**Single-selection-point invariant (the #95 follow-up).** Route selection happens
ONCE — `selectMatmulChoice` at `planTiledMatmul` (single call site) — and the
receipt is stamped there. All three paths consume the same `plan.selection`; none
re-runs selection. Engagement is a receipt PROPERTY:
- lowered dispatch: `dispatchTiledMatmul` increments `gemvDispatchCount` off
  `plan.selection.gemvEngaged` (was: re-parsing the `_gemv` profiler-label string);
- generated stream: `generateBareMatmul` increments the replay-surviving
  `generatedGemvDispatchCount` off `m.selection.gemvEngaged` (was: label re-parse).

This removes the two label-string re-parses that were independent readers of the
route fact, replacing them with reads of the single receipt — the exact
distributed-decision shape the #93 int8 bypass exploited, now collapsed.

---

## 3. Deletions — STOPPED for the tiled family (net-negative NOT achieved)

**Verdict: STOP, reason recorded (per the phase rule).** The §11 tiled "dies"
facts (SSA scaffolding, raw builtin index reads, the K-loop / staging / dot
lowering) were NOT deleted, because `applyTiledMatmulSchedule` **re-calls**
`createTiledMatmulKernel` rather than **absorbing** its structure. The tiled
kernel's structural ownership therefore still lives in `matmulKernelBlockOps`
(175 lines of block-op orchestration: `configureTiles`, `forRange` K-loop,
`load2D`/`dotAccum` staging, epilogue `postAcc`, kSplit/swapGrid/batch branches).

**Why it can't go net-negative in P1 as scoped.** The elementwise skeleton's
`applySchedule` genuinely *reconstructs* its kernel from the `SemanticBodyNode`
tree (`evalBody` emits `stridedLoad`/`applyFusedOp`/`emitStore`; the loop nest is
the derived elementwise VIEW) — so its cutover CAN delete the legacy per-element
builders. The tiled matmul's `applySchedule` does NOT reconstruct from the
`SemanticSchedule` facts; it reconstructs `CodegenOptions` and re-calls the
monolithic builder. Wave 2's own readiness note (report §8) flagged exactly this:
the staging/barrier/role facts are STORED but the live tiled kernel lowers them
BELOW `dotAccum`/`load2D`, and "P1 must decide whether `applySchedule` emits them
into the block-op layer OR the block ops keep owning the barrier lowering (the
no-second-owner seam currently asserts agreement, not single-emission)." This
cutover takes the **latter, documented option**: the block-op realizer keeps
owning the barrier/staging lowering; the no-second-owner seam (`assertTiledSeam`)
asserts the schedule facts AGREE with the descriptor the realizer consumes.

Absorbing `matmulKernelBlockOps` into `applySchedule` (deriving the K-loop nest,
the cooperative-load staging edges, the dot accumulation, and the epilogue chain
from the semantic facts, byte-identically) is a **compile-from-IR effort at
P0-FULL / P2 altitude** — a large rewrite whose correctness is gated on the
byte-differential, not a P1 dispatch-routing change. Forcing it blind here would
risk the byte-differential (subtle SSA/ordering drift) for no correctness gain.
So the tiled family cutover is **routing-complete, structure-deferred**, and its
net SLOC is POSITIVE (see §5). The GEMV/reduction families are the same posture
(their `apply*` also delegates to the underlying generator).

**What this DOES buy at P1 (not net SLOC, but ownership):** the schedule object
is the SOLE writer at the dispatch seam — `createTiledMatmulKernel` /
`generateGemvShaderTileIR` / `generateKSplitReductionShaderTileIR` are now
realizer-internals of `apply*`, unreachable from live dispatch except through the
schedule object. The no-second-owner seam runs on the live path. The R9 receipt
collapses the three route readers to one. The compile-from-IR absorption that
cashes the §11 deletions is the named P1-follow-on (§6).

---

## 4. SelectionReceipt design summary

```ts
interface SelectionReceipt {
  family: "tiled" | "gemv";          // AlgorithmFamily the predicate selected
  choice: MatmulVariantChoice;        // ScheduleTemplate instance picked
  gemvEngaged: boolean;               // route engagement — a receipt PROPERTY
  gemvEpilogueEngaged: boolean;       // fused GEMV epilogue seam engaged
}
```

- **Stamped once**, at the single selection point (`planTiledMatmul` after
  `selectMatmulChoice`), carried on `MatmulStandardPlan` / `MatmulKSplitPlan`.
- **Read, never re-selected**, by all consuming paths (lowered / capture /
  generate). Engagement counters read `.gemvEngaged` — the replay-blind
  `getGemvDispatchCount` reads 0 once a decode template cuts over to the generated
  stream, so the replay-surviving signal is the generated counter, now also
  receipt-driven.
- **Interim scope.** The full three-identity Selection key (shape/device/realizer/
  version/measurement, §5/R20) lands with P3 measurement-identity. P1 needs only
  family + choice + engagement.

---

## 5. Gate matrix

| Gate | Result |
|---|---|
| `npm run build` | ✅ exit 0 |
| matmul differential (25 kernels, now guarding LIVE path) | ✅ 31 tests pass (tiled=8 epilogue=4 batched=1 swapGrid=1 splitK=3 gemv=8) |
| all schedule differentials (elementwise 21 + reduction/row-program 16 + matmul 25 + attention) | ✅ 102 tests, 4 files pass |
| `test:gates` (6/6 compiled-plan correctness) | ✅ 6/6 on device 11 |
| gemv-generated-route (route-engagement via generated counter — the Node decode smoke) | ✅ pass (GEMV baked in generated stream, matches f32 control) |
| matmul-view-input (tiled live path) | ✅ 20 pass |
| quant-gemv-parity (quantized GEMV via realizeGemvWgsl) | ✅ pass |
| parity-fullstack (compiled vs lowered, 30 steps) | ✅ max\|Δ\|=5.72e-6 (< 1e-5) |
| full CPU project | ✅ 1306 passed, 1 skipped, 0 failed (99 files) |
| full webgpu project | ✅ 948 passed, 31 skipped; the 7 non-passes ALL environmental (§7) — 0 from this change |
| profile-training distilgpt2@512 (V100 dev 11) | ✅ ~48ms/step, 5.23GB steady, LEAK OK — within noise of the V100 5.0GB/60ms baseline (byte-identical WGSL → no per-step cost) |
| weight-norm | src NET-POSITIVE (~+45 code SLOC, 0 structural deletions) — item 3 STOPPED (§3) |

---

## 6. What remains for a P1 follow-on

1. **Cash the §11 tiled "dies" facts** — absorb `matmulKernelBlockOps` into
   `applyTiledMatmulSchedule` so the K-loop nest / cooperative-load staging /
   dot / epilogue derive from the `SemanticSchedule` facts (like elementwise's
   `evalBody`), deleting `createTiledMatmulKernel`'s structural body. This is the
   net-negative deletion; it is compile-from-IR work gated on the byte-differential.
   Do GEMV/reduction the same way afterward.
2. **GEMV inputCast realizer path** — the descriptor axis is now closed at the
   schedule layer; the applicability predicate (`gemvVariant.isApplicable`) still
   bails NN casts (`hasInputCast && !transB`). When GEMV grows a load-cast path,
   the predicate becomes the single source and the f16-via-cast decode class routes.
3. **Full SelectionReceipt identity** — the shape/device/realizer/version/
   measurement keys (§5/R20) at P3.
4. **Elementwise + reduction cutover** — same routing pattern (smaller); their
   `applySchedule` already reconstructs, so those CAN go net-negative.

---

## 7. Full webgpu suite result

`948 passed, 31 skipped, 7 failed` on the first full run (device 11). **All 7
failures are ENVIRONMENTAL, none from this change** — the same two classes the
wave-2 report documented:

- **6 files** = Vulkan device-init contention (`Failed to create device chain` /
  `vkCreateDevice VK_ERROR_INITIALIZATION_FAILED`) — device 10 was at ~78-100%
  util from another tenant. `arange`, `tril-triu`, `shared-encoder-safety`,
  `webgpu.spec`, `fused-cross-entropy` ALL pass in isolation (re-run: 5 files,
  21/21). None touch matmul.
- **1 file** (`second-run-determinism` [#84], plus `distilgpt2-finetune`) =
  the gitignored `models/distilgpt2` absent from the worktree
  (`Could not find model.safetensors`). Both PASS once `models/` is symlinked
  from main (re-run: 2/2). `second-run-determinism` [#84] and
  `distilgpt2-finetune` exercise the LIVE matmul path across compounding runs and
  against the finetuning ground-truth trajectory — their pass is a strong
  cutover-correctness signal.

Net: zero code-caused failures; the suite is green modulo shared-box Vulkan
contention + the worktree's absent gitignored model weights.
