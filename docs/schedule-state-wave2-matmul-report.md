# Schedule-State P0-FULL Wave 2 — the MATMUL family

**Branch:** `schedule-state-matmul-wave2` (off `main` @ `96034b6b`).
**Scope:** `deriveScheduleState`/`applySchedule` for the matmul family, SIDE-BY-SIDE
with the live builders (no dispatch cutover — that is P1). New files:
`src/schedule/matmul-skeleton.ts`, `test/schedule/matmul-differential.spec.ts`;
printer extended for staging in `src/schedule/canonical.ts`; argmax folded into
`src/schedule/reduction-skeleton.ts`.

---

## 1. Printed tiled-matmul state (verbatim)

A 64×64×16 tiled matmul, NT, f16, with a `bias → gelu → cast_f16` epilogue —
printed through the wave-1 canonical printer (extended for the staging tier):

```
schedule-state v1
region region:matmul-doc
semantic:
  blockShapes [64,64] [16]
  ordering rowMajor[axis:m,axis:n]
  programGridMap identity
  loopNest:
    loop uid=loop:m entity=ent:loop:m axis=axis:m kind=parallel bound=ceilDiv(leaf(uniform(m)),leaf(int(64)))
      loop uid=loop:n entity=ent:loop:n axis=axis:n kind=parallel bound=ceilDiv(leaf(uniform(n)),leaf(int(64)))
        loop uid=loop:k entity=ent:loop:k axis=axis:k kind=sequential bound=ceilDiv(leaf(uniform(k)),leaf(int(16)))
  values:
    value uid=in:a entity=ent:in:a tier=global dtype=f16 alias=-
    value uid=in:b entity=ent:in:b tier=global dtype=f16 alias=-
    value uid=stage:a_tile entity=ent:stage:a_tile tier=shared dtype=f16 alias=- [staged]
    value uid=stage:b_tile entity=ent:stage:b_tile tier=shared dtype=f16 alias=- [staged]
    value uid=acc entity=ent:acc tier=register dtype=f32 alias=-
    value uid=in:epilogue_in0 entity=ent:in:epilogue_in0 tier=global dtype=f32 alias=-
    value uid=out:out entity=ent:out:out tier=global dtype=f16 alias=-
  noMaterialization:
    noMat producer=acc consumer=write across=loop:m
    noMat producer=acc consumer=write across=loop:m
  stores:
    store src=write -> tgt=out:out @loop:m
  bodies:
    body acc = dot_accum(v(stage:a_tile),v(stage:b_tile))
    body write = gelu(add(v(acc),v(in:epilogue_in0)))
  roles:
    role name=cooperative-load cond=int(1)
  sync:
    memoryEffect space=shared value=stage:a_tile interval=loop:k..loop:k
    memoryEffect space=shared value=stage:b_tile interval=loop:k..loop:k
    barrier role=cooperative-load cond=int(1) spaces=[shared] convergence=uniform
  atoms:
  lemmas:
requests:
  warpBudget -
  pipeline none
  placementPreferences:
    prefer value=acc tier=register interval=loop:k..loop:m
  cachePolicy:
receipts:
  workgroup [8,8,1]
  vecLoad value=in:a form=scalar
  vecLoad value=in:b form=scalar
```

Reading it as the tier split (§2): the **semantic** tier owns the computation-shape
— block shapes `[64,64]`/`[16]` (tileM/N, tileK), the M→N→K loop nest, the two
`[staged]` shared-memory tiles, the cooperative-load role, the per-K-tile shared
barrier, the `dot_accum` body and the `gelu(add(acc, bias))` epilogue body, and the
no-materialization edges that keep the epilogue register-resident. The **requests**
tier owns the realizer ask (acc's register-residency preference; `warpBudget -` =
realizer default; `pipeline none`). The **receipts** tier owns the physical choices
the realizer reports — the WGSL workgroup `[8,8,1]` (= tileN/threadTileN ×
tileM/threadTileM), the scalar vec-load forms. `threadTileM/N` and `vectorWidth`
appear ONLY as receipts, never in semantic identity (A-R1/A-R6). Staging is legible
via the `[staged]` tag + the two `memoryEffect` sync lines + the `barrier` line.

---

## 2. Per-sub-family kernel counts (byte-identical differential)

`compileTileKernel(applyMatmulSchedule(state, desc))` (or the direct WGSL generator
for the seams that already compile) equals the live single-source generator's WGSL
BYTE-FOR-BYTE for every case. Reported by `test/schedule/matmul-differential.spec.ts`:

```
[P0 matmul differential] byte-identical kernels covered: 25
  (tiled=8 epilogue=4 batched=1 swapGrid=1 splitK=3 gemv=8)
```

| Sub-family | Count | Cases |
|---|---|---|
| tiled | 8 | NN/NT/TN/TT f32, NN f16, NT mixed f16A/f32B, NN inputCast f32→f16, big-tile 64×64×16 t8×8 |
| epilogue | 4 | bias; gelu; bias+gelu+cast_f16; residual add (binary) |
| batched | 1 | batched NN f32 (programId(2) batch axis) |
| swapGrid | 1 | NN f32 swapGrid → ProgramGridMap `swap(axis:m,axis:n)` |
| split-K | 3 | tiled kSplit=4 partials; reduction pass count=4 f32; reduction pass count=8 f16 |
| GEMV | 8 | NT f32; NT f16 (#95 shape, bare); NT vec4; NT rowsPerWg=4; NN f32; NN f32 kSplit partials; NT bias+relu; NT quantB int8-grouped g64 |

Live single-source seams the differential calls on both sides:
`generateTiledMatmulShaderTileIR(CodegenOptions)`,
`generateKSplitReductionShaderTileIR(count, dtype)`,
`generateGemvShaderTileIR(GemvKernelOptions)`.

The wave-1 leftover derivable kernel — **argmax/argmin** — was folded into the
reduction skeleton (`deriveArgReduceState`/`applyArgReduceSchedule` →
`argReduceWGSL`), 3 new byte-identical cases (argmax/argmin over a dim); the
reduction+row-program differential now covers 16 kernels (was 13).

---

## 3. Schema deltas the shakedown forced

Wave 2 was the schema shakedown for staging edges + typed sync relations + roles +
receipts (S2 stored facts). **The wave-1 schema in `types.ts` held all of them with
no change** — the shakedown produced ZERO type deltas:

- **Staging edges** — represented as `NamedValue.allocation: "shared"` (`StagingTier
  = AddressSpace` already admits `"shared"`), no new field. The A/B tiles are
  shared-tier NamedValues; the store-into-tile-then-read is a `memoryEffect`
  `SyncRelation` (space `shared`).
- **Barriers** — the `SyncRelation` `barrier` variant (participants `ParticipantSet`,
  spaces `AddressSpace[]`, convergence `ConvergenceFact`) fit the per-K-tile
  workgroup barrier verbatim (R3 typed relation).
- **Roles** — `ParticipantSet` with a predicate-AST condition held the
  cooperative-load role (whole-workgroup ⇒ `intLit 1`).
- **Receipts** — `RealizationReceipts.workgroup` + `vecLoadForms` held the
  compilation-derived geometry and vec forms with no addition.
- **kSplit as a lemma** — `LemmaApplication` (`lemma`, `obligation`,
  `carriedStateRef`) held the fp-sum-reassociation license with no addition; the
  fp-reorder lemma is the K-split `tile`'s admitted-lemma reference.
- **ProgramGridMap `swap`** — already present from the R4 partial amendment; swapGrid
  reifies into it directly.

The ONE printer change (`canonical.ts`): a `[staged]` tag appended to shared-tier
value lines for legibility. It is purely additive text on `tier=shared` lines;
wave-0/1 states carry no shared value, so all 34 wave-0/1 kernel digests and the
golden test vector (`ddfec844386f1ac5850972312794c0b6`) are byte-unchanged (verified
green). This is not a schema delta — the schema already expressed everything; the
tag is a print-legibility affordance the task asked for.

**Conclusion:** the wave-1 `types.ts` schema is sufficient for the matmul family
including its staging/barrier/role/receipt facts. No `types.ts` field was wrong.

---

## 4. The epilogue ⊥ kSplit rule as implemented

In the live code the incompatibility is **scattered across three sites**:
`computeKSplitFactor` returns 0 when `hasEpilogue` (`dispatch.ts:759`);
`createTiledMatmulKernel` gates `postAcc` on `!kSplit` (`tile-matmul.ts:164`);
`createGemvKernel` / `planGemvRowMatmul` throw / bail when `epilogue && kSplit`
(`gemv.ts:326`, `dispatch.ts:1053`). The *reason* is structural: a K-split kernel
writes RAW f32 partials that a SEPARATE reduction pass sums — so a bias/activation
epilogue cannot be applied per-split; it must run exactly once on the summed output.

Wave 2 makes it **ONE typed legality rule read off the object**, in
`assertTiledSeam` (matmul-skeleton.ts):

```ts
const hasEpilogueChain =
  !!desc.epilogue && desc.epilogue.ops.some((o) => o.kind !== "none");
const hasKSplit = (desc.kSplit ?? 0) >= 2;
if (hasEpilogueChain && hasKSplit)
  reportNoSecondOwner(
    `legality[matmul]: epilogue ⊥ kSplit — a K-split kernel writes raw f32 partials, ` +
      `so an epilogue chain cannot be applied per-split (it must run once on the summed ` +
      `output). This state carries both an epilogue chain and the kSplit lemma.`,
  );
```

It is enforced as a property of the typed `ScheduleState` (the epilogue lives in
`bodies`/`noMaterialization`; kSplit lives in `lemmas`) rather than a codegen
conditional. The seam ALSO asserts the object is internally consistent: `kSplit`
present ⇔ the fp-reorder lemma is present, and a kSplit state carries NO
no-materialization edges. The differential's legality test
(`REFUSES a state carrying both an epilogue chain and kSplit`) drives it: deriving
such a state and applying it THROWS under strict (default). GEMV carries the same
rule in `assertGemvSeam` (epilogue ⊥ kSplit; plus quantB ⇒ NT-only, no-kSplit).

---

## 5. #95 diagnosis (report-only — feeds task #95)

**Question:** why does variant selection NOT route f16 M=1 decode projections to
GEMV NT?

**Answer: (a) an applicability-predicate gap — but the excluding predicate is
`hasInputCast`, NOT dtype.** The verbatim decision (`variants.ts:119-135`
`gemvVariant.isApplicable`):

```ts
return (
  ctx.m === 1 &&
  ctx.batchSize === 1 &&
  !ctx.hasInputCast &&
  !ctx.hasExplicitConfig &&
  ENV.TORCHLETTE_GEMV !== "0" &&
  computeGemvRoute(ctx.n, ctx.k, ctx.transB) !== null
);
```

Neither `isApplicable` nor the geometry gate `computeGemvRoute(n, k, transB)`
(`gemv.ts:217`) inspects **dtype** at all. A *bare* f16 M=1 NT projection (n>1)
routes to GEMV NT correctly — and the differential proves the f16 NT GEMV kernel
compiles byte-identically ("NT f16 (the #95 shape — bare)"). So the f16 exclusion
is NOT in the GEMV path per se.

The gap is `!ctx.hasInputCast`. In the f16 decode/AMP path the projection weights
are stored f32 and cast to f16 *during the tile load* (`inputCastA`/`inputCastB`),
which sets `hasInputCast = true` at `dispatch.ts:1152`
(`hasInputCast: !!inputCastA || !!inputCastB`). The GEMV kernels do not implement
load-casts (`variants.ts:71` — "GEMV kernels don't implement load-casts"), so
`isApplicable` returns false and the shape falls through to the tiled family. That
is the "#93 finding: the f16 path was kernel-bound" — the f16 decode projection is
kernel-bound on tiled because its input-cast presence *disqualifies* GEMV, even
though the GEMV NT kernel could run the f16 case directly if the operands were
materialized f16 (dtypeA/dtypeB f16, no cast).

**Classification:** applicability-predicate gap (`hasInputCast` excludes the whole
f16-via-cast decode class), NOT a shape-class miss (`gemv_row` is correctly
classified) and NOT a deliberate f16 fallback (no dtype check anywhere in the GEMV
route). **Fix direction for #95 (not implemented here):** either (i) give GEMV NT a
load-cast path so `hasInputCast` no longer disqualifies it, or (ii) have the decode
projection materialize f16 operands (drop the input-cast) so the bare f16 NT GEMV
kernel — already proven to compile — is selected. No routing behavior was changed
in this wave.

---

## 6. Quantized-format expressibility (#93 coordination — one paragraph)

The int8-grouped B-operand route IS expressible, and the schedule object confirms
the design intent: quant is a **SELECTION fact** (R9 registry territory), not
schedule structure. In `deriveGemvState`, the B operand is a NamedValue carrying its
LOGICAL dtype; the `QuantB` descriptor (`{scheme:"int8-grouped", groupSize}`) rides
alongside as OPERAND METADATA, never as a loop/staging/role structural change. The
differential's expressibility test proves it: the SEMANTIC tier printed for a dense
NT GEMV and a quantB NT GEMV of the same shape is **byte-identical** — same loop
nest, same block shapes, same `dot_accum` body, same store edge — because the
int8-grouped dequant (`unpackInt8Snorm · f16 scale`, `gemv.ts:377-407`) is the
realizer's operand-decode, not a second owner of any schedule fact. The seam's
ROUTING — which realizer decodes the packed `u32` weight + companion `b_scales`
binding — is out of schedule scope (a SelectionReceipt the R9 registry owns at P1).
So the schedule can EXPRESS a matmul whose B operand carries a StorageFormat as
operand metadata; whether that format is selected is a request/selection concern the
schedule declines to identify into semantic identity.

---

## 7. Gate matrix

| Gate | Result |
|---|---|
| `npm run build` | ✅ exit 0 |
| matmul differential (25 kernels) | ✅ 31 tests pass (tiled=8 epilogue=4 batched=1 swapGrid=1 splitK=3 gemv=8) |
| wave-0/1 differentials (elementwise 21 + reduction/row-program 16 incl. argmax) | ✅ 43 tests, 37 kernels (34 wave-1 + 3 argmax) |
| epilogue ⊥ kSplit typed rule + swapGrid reification + quant expressibility | ✅ pass |
| move-script: `program-map swapGrid` + `tile` on K axis → replay digest-identical | ✅ pass |
| digest stability + golden vector `ddfec844…` | ✅ unchanged |
| `npm run test:gates` (6/6) | ✅ 6/6 on device 10 |
| `npm run test` — CPU project | ✅ 1277 passed, 1 skipped, 0 failed |
| `npm run test` — webgpu project (clean run, 76 files) | ✅ 926 passed; the 4 non-passes are environmental (3 = transient GPU device-chain contention → pass 3/3 in isolation; 1 = `models/distilgpt2` absent from the worktree → passes once the gitignored `models/` is symlinked from main) — none from this wave's code |
| parity-fullstack (compiled vs lowered, 30 steps) | ✅ max\|Δ\|=8.583e-6 (< 1e-5) |
| lint (my 5 touched files) | ✅ clean (repo-wide lint pre-fails on 852 unrelated errors) |
| weight-norm | +1 src file (matmul-skeleton.ts) — the side-by-side loan |

**Contention note (operational):** the first `npm run test` run reported 232
webgpu failures — ALL `Failed to create device chain` (Vulkan device-init
contention), because an orphaned earlier full-suite run + other agents' processes
(an `agent-ac13ae20…` liveness probe, `q93-int8-decode-fix` vite server) were
competing for the single filtered device. Killed the orphan by PID; the CPU project
passed clean in the same run (1277/0), and every failing webgpu spec passes in
isolation (verified `frontend-dtype.spec.ts` 28/28). The clean webgpu re-run is
logged separately.

**Weight-norm loan:** the side-by-side skeleton adds `matmul-skeleton.ts` (+1 file,
~730 SLOC). This is the wave's declared loan, matching the elementwise/reduction
precedent — it is retired at the P1 cutover when the §11 matmul "dies" facts are
deleted (the live builders' structural ownership moves into `applySchedule`). No
existing mechanism was deleted this wave; the deletions land at P1.

---

## 8. P1-cutover readiness assessment

Wave 2 leaves the matmul family P1-ready:

- **The seam is proven byte-exact** across all sub-families on the compilation path
  (25 kernels), so `applyMatmulSchedule` can become the dispatch's kernel source at
  P1 with the differential as the regression guard.
- **The tier split is validated against real geometry** — every §11.4 fact landed in
  exactly one tier with no schema delta, so the P1 "decorations → S1 requests"
  migration has a proven target shape.
- **The R9 registry migration (P1) has its field-by-field evidence**: gemv / tiled /
  split-K / swapGrid / epilogues all round-trip; the applicability predicates
  (`isApplicable`, `computeGemvRoute`, `computeKSplitFactor`) are the
  `ApplicabilityPredicate` inputs, `classifyShape`/`DEFAULT_CONFIG`/`TUNING_SPACE`
  the `ScheduleTemplate` + candidate space.
- **Two things to settle before cutover:** (i) the staging/barrier/role facts are
  STORED in wave 2 (S2 shakedown) but the tiled kernel's live path lowers them
  BELOW `dotAccum`/`load2D` — P1 must decide whether `applySchedule` emits them into
  the block-op layer or the block ops keep owning the barrier lowering (the
  no-second-owner seam currently asserts agreement, not single-emission). (ii) The
  #95 routing gap (`hasInputCast` disqualifying f16 GEMV) is a SELECTION concern the
  P1 registry migration should fix at the applicability-predicate layer, not the
  schedule.
- **No cutover this wave** (per scope): the §11 matmul "dies" facts are deleted at
  P1, not now; the weight-norm loan is retired then.

**Commit:** `48cb9f66` on `schedule-state-matmul-wave2` (off main @ `96034b6b`; NOT
pushed). The report doc itself was committed in `48cb9f66`; this hash line is a
same-commit self-reference.
