# Schedule-State P0-FULL Wave 3 — the ATTENTION family

**Branch:** `worktree-agent-a902c5ca5f27748f6` (off `main` @ `dcabeb5c`).
**Scope:** the ATTENTION family — the last big census block and the P2
prerequisite. SIDE-BY-SIDE with the live path; NO dispatch cutover; NO behavior
change (the browser demo path is untouched — derivation READS the live kernel).
New files: `src/schedule/attention-skeleton.ts`,
`test/schedule/attention-differential.spec.ts`; printer extended for the
authored/opaque form in `src/schedule/canonical.ts`
(`printSkeleton`/`skeletonDigest`); four `make*Spec` factories + tiling constants
+ `assertBackwardSupportsModifier` exported from
`src/backend/webgpu/attention-kernel.ts` (byte-neutral — proven by the
differential; adding `export` changes no emitted WGSL).

Two shapes, one family: the fused FlashAttention kernels are **authored**
(opaque, F3 — the online-softmax composite is not yet derivable by the move
grammar); the naive decomposed attention is a **derivable composition** of the
existing matmul + row-program family skeletons. Wave 3 delivers both and the
P2-readiness assessment.

---

## 1. Printed authored-attention forward state (verbatim)

Printed through the new `printSkeleton` (the `authored` block — typed params
VISIBLE, skeleton SEALED per F3):

```
authored-skeleton v1
authored:
  kernelRef src/backend/webgpu/attention-kernel.ts::makeForwardAttentionSpec
  refusal authored — not yet re-derived. The fused online-softmax composite is reachable only via the P2 FA-derivation (rungs 0–7: merge naive three islands [S3] → tile → stream K/V → recolor accumulator → apply the online-softmax admitted lemma; F17 sequence lemma→recolor→recolor→group→stream), which the move grammar cannot yet run (S3 merge/fuse is an unbuilt engine transaction). Exit: local self-hosting (attention backward re-derived once the recomputation-identity + D-precompute lemmas are admitted; §6/§7 P4).
  skeleton sealed=opaque (F3: no loop/staging/role data)
  params:
    param bwdKvRows domain=[64] default=64
    param bwdQRows domain=[16] default=16
    param headDimension domain=[64,128,256] default=64
    param kvRows domain=[32] default=32
    param qRows domain=[64] default=64
  constraints:
    cmp(==,member(uniform(headDimension),[int(4)]),int(0))
  capability or[cmp(<,uniform(headDimension),int(256)),cmp(>=,uniform(maxComputeWorkgroupStorageSize),int(32768))]
  admittedLemmas:
    lemma uid=lemma:online-softmax-rescaling obligation=obl:online-softmax-normalizer-equals-batched-denominator carried=carried=(m:running-max,l:running-normalizer,o:partial-output);correction=exp(m_old-m_new)
```

Reading it (§6 / R10): the `authored` block owns the **DECLARED surface** — the
sealed `kernelRef`, the refusal reason (the P2-derivation narrative, so the
opaque hatch announces its own exit), the F7 skeleton-family param keys
(`qRows`/`kvRows`/`headDimension` + backward block sizes, each a domain +
default, single-sourced from the live `BR/BC/BQ_BW/BC_BW` constants), the
dependent `constraints` (the vec4 divisibility `headDimension % 4 == 0`, printed
as a predicate-AST member/cmp), the `capability` predicate (the **256-head-dim
32KB-workgroup-storage** requirement, a disjunction), and the online-softmax
`admittedLemma` (its `(m,ℓ,o)` carried state + the proof obligation it
discharges). Everything below `dotAccum`/the online recurrence — loops,
barriers, the K/V shared tiles — is **sealed** (F3 forbids an opaque skeleton
from carrying loop/staging/role data; there is no field to hold them, so there
is nothing to print). An authored kernel is legible WITHOUT masquerading as a
derived one.

---

## 2. The naive attention COMPOSITION (verbatim — the educational artifact)

Three ScheduleStates over three semantic regions, built with the EXISTING
matmul + row-program family skeletons (`deriveTiledMatmulState` ×2 +
`deriveRowProgramState`), plus the island-level structure connecting them. This
is P2's starting position. Each region's semantic tier, printed:

```
=== NAIVE ATTENTION as a derivable COMPOSITION (P2 starting position) ===

region 1 [region:attn-naive-qkT]  scores = Q @ Kᵀ
schedule-state v1
region region:attn-naive-qkT
semantic:
  blockShapes [32,32] [16]
  ordering rowMajor[axis:m,axis:n]
  programGridMap identity
  loopNest:
    loop uid=loop:m entity=ent:loop:m axis=axis:m kind=parallel bound=ceilDiv(leaf(uniform(m)),leaf(int(32)))
      loop uid=loop:n entity=ent:loop:n axis=axis:n kind=parallel bound=ceilDiv(leaf(uniform(n)),leaf(int(32)))
        loop uid=loop:k entity=ent:loop:k axis=axis:k kind=sequential bound=ceilDiv(leaf(uniform(k)),leaf(int(16)))
  values:
    value uid=in:a entity=ent:in:a tier=global dtype=f32 alias=-
    value uid=in:b entity=ent:in:b tier=global dtype=f32 alias=-
    value uid=stage:a_tile entity=ent:stage:a_tile tier=shared dtype=f32 alias=- [staged]
    value uid=stage:b_tile entity=ent:stage:b_tile tier=shared dtype=f32 alias=- [staged]
    value uid=acc entity=ent:acc tier=register dtype=f32 alias=-
    value uid=out:out entity=ent:out:out tier=global dtype=f32 alias=-
  noMaterialization:
  stores:
    store src=write -> tgt=out:out @loop:m
  bodies:
    body acc = dot_accum(v(stage:a_tile),v(stage:b_tile))
    body write = v(acc)
  roles:
    role name=cooperative-load cond=int(1)
  sync:
    memoryEffect space=shared value=stage:a_tile interval=loop:k..loop:k
    memoryEffect space=shared value=stage:b_tile interval=loop:k..loop:k
    barrier role=cooperative-load cond=int(1) spaces=[shared] convergence=uniform
  atoms:
  lemmas:

region 2 [region:attn-naive-softmax]  P = softmax(scores)
schedule-state v1
region region:attn-naive-softmax
semantic:
  blockShapes []
  ordering flat
  programGridMap identity
  loopNest:
    loop uid=loop:row entity=ent:loop:row axis=axis:row kind=parallel bound=leaf(uniform(num_rows))
      loop uid=loop:feature entity=ent:loop:feature axis=axis:feature kind=sequential bound=leaf(uniform(feature_dim))
  values:
    value uid=in:in0 entity=ent:in:in0 tier=global dtype=f32 alias=-
    value uid=out:output entity=ent:out:output tier=global dtype=f32 alias=-
    value uid=r0 entity=ent:r0 tier=register dtype=f32 alias=-
    value uid=r1 entity=ent:r1 tier=register dtype=f32 alias=-
  noMaterialization:
  stores:
    store src=write -> tgt=out:output @loop:feature
  bodies:
    body r0 = reduce_max(v(in:in0))
    body r1 = reduce_sum(exp(sub(v(in:in0),v(r0))))
    body write = div(exp(sub(v(in:in0),v(r0))),v(r1))
  roles:
  sync:
  atoms:
  lemmas:

region 3 [region:attn-naive-pv]  O = P @ V
schedule-state v1
region region:attn-naive-pv
semantic:
  blockShapes [32,32] [16]
  ordering rowMajor[axis:m,axis:n]
  programGridMap identity
  loopNest:
    loop uid=loop:m entity=ent:loop:m axis=axis:m kind=parallel bound=ceilDiv(leaf(uniform(m)),leaf(int(32)))
      loop uid=loop:n entity=ent:loop:n axis=axis:n kind=parallel bound=ceilDiv(leaf(uniform(n)),leaf(int(32)))
        loop uid=loop:k entity=ent:loop:k axis=axis:k kind=sequential bound=ceilDiv(leaf(uniform(k)),leaf(int(16)))
  values:
    value uid=in:a entity=ent:in:a tier=global dtype=f32 alias=-
    value uid=in:b entity=ent:in:b tier=global dtype=f32 alias=-
    value uid=stage:a_tile entity=ent:stage:a_tile tier=shared dtype=f32 alias=- [staged]
    value uid=stage:b_tile entity=ent:stage:b_tile tier=shared dtype=f32 alias=- [staged]
    value uid=acc entity=ent:acc tier=register dtype=f32 alias=-
    value uid=out:out entity=ent:out:out tier=global dtype=f32 alias=-
  noMaterialization:
  stores:
    store src=write -> tgt=out:out @loop:m
  bodies:
    body acc = dot_accum(v(stage:a_tile),v(stage:b_tile))
    body write = v(acc)
  roles:
    role name=cooperative-load cond=int(1)
  sync:
    memoryEffect space=shared value=stage:a_tile interval=loop:k..loop:k
    memoryEffect space=shared value=stage:b_tile interval=loop:k..loop:k
    barrier role=cooperative-load cond=int(1) spaces=[shared] convergence=uniform
  atoms:
  lemmas:

island-flow:
  region:attn-naive-qkT --scores--> region:attn-naive-softmax
  region:attn-naive-softmax --P--> region:attn-naive-pv
```

Reading it as the P2 starting position: region 1 (QK^T) and region 3 (PV) are
plain tiled matmuls (NT and NN); region 2 is the numerically-stable row softmax
(`reduce_max` → `reduce_sum(exp(x−m))` → `div(exp(x−m), l)`) — the SAME
RowProgram the graph compiler detects for a real softmax. The three carry NO
lemmas (they compute the naive result); the fused kernel's move-grammar path
adds the online-softmax lemma to `merge` these into one island. The `island-flow`
edges (`scores`, `P`) are the composite the `merge`/`fuse` S3 transaction
consumes. Each region round-trips byte-identically via the REUSED family apply
seams (§4, no duplication).

---

## 3. Per-role kernel counts (byte-identical differential)

`compileTileKernel(applyAttentionSchedule(deriveAttentionSkeleton(desc), desc))`
equals `compileTileKernel(<live make*Spec(headDim, mod)>)` BYTE-FOR-BYTE for
every case. Reported by `test/schedule/attention-differential.spec.ts`:

```
[P0 attention differential] byte-identical authored kernels covered: 14 (forward=5 dPrecompute=1 backwardDQ=4 backwardDKV=4)
```

| Role | Count | Modifier corpus |
|---|---|---|
| forward | 5 | bare, causal, sliding-window, softcap, causal+sliding-window |
| D-precompute | 1 | mod-invariant (no score/mask seam sites — one template) |
| backward-dQ | 4 | bare, causal, sliding-window, causal+sliding-window (scoreMod refused) |
| backward-dKV | 4 | bare, causal, sliding-window, causal+sliding-window (scoreMod refused) |

Live single-source seams the differential calls on both sides:
`makeForwardAttentionSpec`, `makeDPrecomputeSpec`, `makeBackwardDQSpec`,
`makeBackwardDKVSpec` (all D64, the #64 Gemma seams exercised via the modifier
corpus). Backward + scoreMod is REFUSED at the seam (the authored apply re-calls
`assertBackwardSupportsModifier` — inference-first, #64) — the same refusal the
live dispatch/plan path carries, one owner.

Total across the schedule-state census after wave 3: **76 byte-identical
kernels** (elementwise 21 + reduction/row-program 16 + matmul 25 + attention 14),
up from 62 at wave 2's end. The naive composition adds 3 region round-trips via
the reused family differentials (not counted in the 76 — they ARE the matmul /
row-program families, verified through the same seams).

---

## 4. The naive composition verified via REUSED family differentials (no duplication)

Region 1 (QK^T) and region 3 (PV): `applyTiledMatmulSchedule` vs
`generateTiledMatmulShaderTileIR` — the matmul family's byte seam. Region 2
(softmax): `applyRowProgramSchedule` vs `rowProgramToSpec` — the row-program
family's byte seam. The composition builds NO new derivation code: it calls
`deriveTiledMatmulState` ×2 and `deriveRowProgramState` ×1 (the wave-1/2 family
skeletons) and asserts each region's WGSL byte-identical through the existing
family apply. The island-flow edges connect them. This is exactly "the
composition's per-region derive/apply is byte-identical via the existing family
differentials (reuse, don't duplicate)."

---

## 5. Schema deltas the wave forced

**ZERO type deltas in `types.ts`** — matching wave 2's bar. The wave-1/2 schema
held every attention fact:

- **Authored / opaque form** — the `Skeleton` `visibility: "opaque"` variant
  (`kernelRef` + `refusalReason` + `params: TypedParamSchema`) held the fused
  kernels verbatim (F3). No new field.
- **Typed parameter schema** — `TypedParamSchema` (params with domains/defaults,
  dependent `constraints` as predicate-AST, `capabilityPredicate`) held the
  attention constraint set (vec4 + 256-head-dim workgroup storage) with no
  addition — the SAME schema the matmul family used in wave 2.
- **Online-softmax lemma** — `LemmaApplication` (`lemma` + `obligation` +
  `carriedStateRef`) held the `(m,ℓ,o)`-carried online-softmax lemma with no
  addition (§3.4 F27/F28 — the schema was authored for exactly this in wave 0).
- **The modifier seam (#64)** — the score/mask kinds ride as the cache-key
  encoder's structural fragment via the live `attnModifierKey` (the single
  source); numeric params (cap, window) are uniform DATA, not schema fields.

**Printer changes (`canonical.ts`) — additive only:** `printSkeleton` (the
`authored` block: kernelRef + refusal + sealed marker + typed params + admitted
lemmas), `printParamSchema`, `printLemma` (extracted for reuse), and
`skeletonDigest`. These are NEW functions on a NEW artifact (the authored
skeleton); they do not touch `printScheduleState`, so all wave-0/1/2 kernel
digests and the golden test vector (`ddfec844386f1ac5850972312794c0b6`) are
byte-unchanged (verified green). This is not a schema delta — the schema already
expressed the authored form; the printer gained a legible print for the
`Skeleton` union the design (§6) asked for.

**Conclusion:** the wave-1/2 `types.ts` schema is sufficient for the attention
family — including its authored/opaque form, typed param schema, capability
predicate, and online-softmax lemma. No `types.ts` field was wrong. The 256-head
capability is the design's F9 capability predicate; the vec4 constraint is the
design's dependent constraint; the online-softmax lemma is the design's §3.4
admitted-lemma application — all pre-existing schema.

---

## 6. The P2-READINESS ASSESSMENT (what the FA-derivation acceptance still needs)

Wave 3 lands the P2 STARTING POSITION (the naive three-region composition) and
the P2 TARGET (the authored fused kernel + its online-softmax lemma). The P2
acceptance narrative (§7 P2) IS the flashattention derivation, rungs 0–7:
`merge` the three naive islands → `tile` → `stream` K/V through shared →
`recolor` accumulator → apply the online-softmax lemma (F17 sequence
`lemma → recolor → recolor → group → stream`, refusal-first). Beyond this wave,
the acceptance still needs FOUR objects. Each is named buildable-now vs
needs-design:

### (a) The `merge`/`fuse` composite transaction (S3) — **NEEDS-DESIGN (engine side unbuilt)**

`fuse` is NOT a ScheduleState move (§3.5 S3): membership is owned by `Partition`
alone; the editor's "fuse" gesture is a COMPOSITE TRANSACTION at the islands
altitude (`validate interior → merge(P,a,b) → mint region' → attach → record ONE
provenance entry → roll back atomically on realization failure`). Wave 3 provides
the transaction's INPUT (the three-region composition + the island-flow edges)
and its OUTPUT identity (the authored fused schedule's digest), but the
transaction ITSELF — `fuseGesture(P, a, b, proposedInteriorSchedule)` — is
engine-side and UNBUILT. It depends on the islands `merge` legality gate
(dataflow convexity + shape/binding-count/barrier/chunking, device-keyed) which
lives in `islands-design.md §2`, not in `src/schedule/`. **Needs-design:** the
composite-transaction driver + the islands `merge` legality port into the
schedule seam. Not buildable in a P0-full wave (it is the P2 macro-move
altitude).

### (b) The `stream` move on the softmax region (streamability predicate) — **NEEDS-DESIGN (engine side); the predicate SHAPE exists**

The `stream(valueUid, loopUid)` move (§3.1) turns a materialized intermediate
into a produced/consumed-inside-a-loop value (deletes the store edge, adds a
no-materialization + carried-value edge). Its invariant: the value MUST have a
declared **head/body decomposition** over the loop's axis — streamability is
machine-checked over typed head/body terms with a recomposition law (F5), and a
value with no decomposition is REFUSED (the FA "refusal-first" boundary, F17 —
dragging streaming onto naive softmax is correctly refused because ordinary
softmax has no head/body decomposition; the online-softmax lemma is what GIVES
softmax a head/body decomposition). **Where it lives:** the NCD spike had the
streamability predicate CLIENT-SIDE; the engine-side home is the `stream` move's
executable partial function (the move algebra is P2 work — the schema's
`ScheduleMove` union has the `stream` variant, but the invariant-checking body
is unbuilt). The naive composition's softmax region (region 2) is exactly the
value the refusal fires on until the lemma is applied. **Needs-design:** the
engine-side streamability predicate (typed head/body + recomposition law) + its
refusal wiring. The predicate's typed SHAPE (predicate-AST head/body terms) is
buildable-now on the existing `PredicateAstNode`; the move-algebra body that
consults it is P2.

### (c) The online-softmax admitted lemma as an ENGINE object — **BUILDABLE-NOW (this wave instantiates the catalog entry)**

Wave 3 **DELIVERS** this. The spike had the lemma client-side; the lemma schema
(`LemmaApplication`) already existed in `types.ts`. Wave 3 instantiates the
catalog entry as an engine object: `ONLINE_SOFTMAX_LEMMA` (LemmaUid),
`ONLINE_SOFTMAX_OBLIGATION` (the proof-obligation ID —
`obl:online-softmax-normalizer-equals-batched-denominator`), and
`onlineSoftmaxLemma()` returning the `LemmaApplication` with its FIRST-CLASS
carried state `(m:running-max, l:running-normalizer, o:partial-output)` +
correction factor `exp(m_old − m_new)` (§3.4 F27 verbatim). It prints under the
authored skeleton's `admittedLemmas` block and is what distinguishes the fused
schedule from the naive composition by its lemma set (F27/F28). **Still
needs-design for full P2:** the lemma's `rewrite: BoxRewrite` (the before/after
on the affected box) and its own `differential: DifferentialRef` gate (§3.4 Lemma
schema fields) — this wave carries the APPLICATION (uid + obligation + carried
state) but not the executable box-rewrite the `merge` transaction applies. The
carried-state instance + obligation ID are buildable-now (done); the box-rewrite
+ lemma differential are P2.

### (d) The pre-registered perf-protocol baseline pin — **BUILDABLE-NOW (the pin is the authored commit; the harness is P2)**

The R24 pre-registered perf gate (§7 P2 / §8 gate 3): geometric-mean slowdown
**≤ 1.5×** vs the authored `fusedAttention` commit **at merge time**, **no single
case > 2.0×**; on the **A100 class via Dawn/Vulkan**; shapes
**{(B1, H8..12, S512, D64), (B1, H8..12, S2048, D64)}**, **f16 inputs / f32
accum**; **3 warmup + median-of-7**; a failed case REPORTED never averaged.
**The baseline PIN is buildable-now:** it is exactly this wave's authored
`fusedAttention` schedule at the merge commit — the authored skeleton's kernelRef
(`makeForwardAttentionSpec` et al.) IS the "authored `fusedAttention` commit at
merge time," and its digest (via `skeletonDigest`) is the stable identity to pin
the baseline measurement against. **The measurement harness is P2:** the derived
FA path does not exist until (a)–(c) land, so there is nothing to measure the
1.5×/2.0× ratio AGAINST yet. The pin coordinate (authored commit + shapes +
protocol) is recordable now; the `tools/`-checked derivation script that meets
the numbers is §8 gate 3 (P2).

### P2-readiness summary table

| P2 object | Status | Home |
|---|---|---|
| (a) `merge`/`fuse` composite transaction (S3) | **NEEDS-DESIGN** (engine unbuilt) | islands altitude; §3.5, `islands-design.md §2` |
| (b) `stream` on softmax region + streamability predicate | **NEEDS-DESIGN** (engine unbuilt); predicate SHAPE buildable-now | `stream` move algebra (P2); `PredicateAstNode` head/body |
| (c) online-softmax lemma as engine object | **BUILDABLE-NOW — DELIVERED this wave** (application + carried state + obligation); box-rewrite + lemma differential are P2 | `attention-skeleton.ts` `onlineSoftmaxLemma()` |
| (d) perf-protocol baseline pin | **BUILDABLE-NOW** (the pin = this authored commit's digest); harness is P2 | `skeletonDigest` of the fused skeleton; §8 gate 3 |

Net: wave 3 delivers everything a P0-full wave CAN deliver toward P2 — the
starting position (naive composition), the target (authored fused schedule), and
the online-softmax lemma engine object with its carried state + obligation. The
two remaining P2 blockers ((a) the S3 composite transaction and (b) the `stream`
move algebra) are P2 macro-move altitude, correctly OUT of a census wave's scope.

---

## 7. Gate matrix

| Gate | Result |
|---|---|
| `npm run build` | ✅ exit 0 |
| attention differential (14 authored kernels + 3 naive-composition regions) | ✅ 28 tests pass (forward=5 dPrecompute=1 backwardDQ=4 backwardDKV=4) |
| prior schedule differentials (elementwise 21 + reduction/row-program 16 + matmul 25) | ✅ all green — 76 kernels total across the census |
| all 4 schedule differential files together | ✅ 102 tests pass |
| authored-form legality (opaque F3, vec4 refusal, backward-scoreMod refusal, cache-key single source, 256-head capability) | ✅ pass |
| digest stability (authored skeleton + naive regions) + golden vector `ddfec844…` unchanged | ✅ pass |
| `npm run test:gates` (6/6) | ✅ 6/6 on device 10 |
| parity-fullstack (compiled vs lowered, 30 steps) | ✅ max\|Δ\|=7.629e-6 (< 1e-5) |
| `npm run test` — CPU project (full) | ✅ (see §8) |
| `npm run test` — webgpu project (full) | ✅ (see §8) |
| browser demo path (attention-kernel.ts behavior) | ✅ UNTOUCHED — derivation READS the kernel; the only edits are `export` keywords (byte-neutral, proven by the byte differential compiling the live factory on BOTH sides) |
| weight-norm | +2 files (`attention-skeleton.ts`, the differential spec) + printer additions — the side-by-side loan, retired at P1 cutover |

**Byte-neutrality of the attention-kernel.ts edits:** the only changes are
adding `export` to four `make*Spec` factories, the four tiling constants
(`BR/BC/BQ_BW/BC_BW`), and `assertBackwardSupportsModifier`. Adding `export`
emits no different WGSL — and the byte differential PROVES it: it compiles the
live factory (now exported, called directly) on one side and the authored
apply-seam (which re-calls the same factory) on the other, byte-for-byte equal.
No kernel body, no dispatch, no seam logic changed.

**Weight-norm loan:** the side-by-side skeleton adds `attention-skeleton.ts`
(+1 src file) + its differential spec (test, free). Printer gained `printSkeleton`
/`skeletonDigest`/`printParamSchema`/`printLemma`. This is the wave's declared
loan, matching the wave-1/2 precedent — retired at the P1 cutover when the live
`make*Spec` structural ownership moves into `applyAttentionSchedule` (and the
authored hatch shrinks per §6 rule 1 as attention is re-derived at local
self-hosting, §7 P4). No existing mechanism was deleted this wave; the deletions
land at P1/P4.

---

## 8. Commit

**Commit:** `8ba0945a` on `worktree-agent-a902c5ca5f27748f6` (off main @
`dcabeb5c`; NOT pushed). This hash line is a same-branch self-reference recorded
after the commit.
