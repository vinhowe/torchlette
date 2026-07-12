# The External Conformance Corpus — grammar-completeness as a CI-checkable claim

Track-2 of the schedule-state campaign. `docs/schedule-state-design.md` §7 P4 (R25/R4) draws a
sharp line: re-deriving the framework's *own* two hand-crafted kernels (attention backward, fused
Adam) is **local self-hosting** — it proves those two, not "the closure contains the state of the
art." **Grammar-completeness is a SEPARATE claim, gated on an EXTERNAL published-kernel conformance
corpus.** This document defines that corpus, states the claim precisely, and records what the
corpus provably does — and does not — cover.

The corpus is a **standing gate**, not a phase. It grows with reviewer-approved external kernels.
Grammar-completeness remains **FALSE** while any registered corpus entry is unrepresented (its
derivation fails or an oracle breaks).

- **Entries:** `tools/conformance/*.ts` (one runnable script per entry; each exit-0 asserts its
  derivation succeeds and its oracles hold).
- **Runner (the gate):** `tools/conformance/run-all.ts` — `npm run conformance` (or
  `TORCHLETTE_CPU_ONLY=1 npx tsx tools/conformance/run-all.ts`). Exit 0 iff every entry passes.
- **Shared spine:** `tools/conformance/harness.ts` (the `ConformanceEntry` schema + `runEntry` /
  `runModule` so an entry file is just its schema + its oracles).

---

## What the claim is, precisely

**In-closure** means: the published technique is reachable as a **finite move-script over the
EIGHT-move grammar** (`tile`, `stream`, `recolor`, `pack`, `role-partition`, `pipeline`,
`program-map`, and the S3 composite `fuse`), plus **admitted-lemma applications** from the lemma
library (`src/schedule/moves/lemma.ts` `LEMMA_LIBRARY`) whose obligations are discharged by a
numerically-gated recomposition law — starting from a **named semantic base state** and ending at a
declared outcome. The move grammar is FENCED; a technique that would need a *new move kind* is by
definition out of closure (and, per the containment analysis, is CUDA-graduation content — the
`role-partition` slot is reserved-but-empty on WGSL).

**The corpus's honesty.** The containment analysis (`kernel-editor-containment-and-ladder.md`
Part 3) prices frontier expressibility at **X ≈ 85–90%** under the *extended* vocabulary. The
corpus tests that claim against OUTSIDE kernels. Entries therefore come in two moral kinds:

1. **Reachability entries** — a published technique IS derived in-grammar (`byte-target` where an
   in-repo kernel is the exact endpoint; `numeric+cost` otherwise: numeric parity vs a reference +
   a stated cost-class effect).
2. **Boundary / negative entries** — the corpus records, honestly, a technique that is
   **out of closure** (a documented boundary, e.g. decoupled-lookback needs cross-workgroup
   forward progress WGSL cannot give), or a composition the grammar **refuses** by a typed legality
   rule (`typed-refusal` — negative knowledge; the corpus checks refusals too).

A corpus that only proved reachability would be marketing. The boundary entries are the part that
makes the completeness claim falsifiable and honest.

---

## Entry schema

Each entry is a `ConformanceModule` (`harness.ts`) with a declarative header and a runnable gate:

| Field | Meaning |
|---|---|
| `technique` + `citation` | the published kernel technique and its source |
| `baseState` | the named semantic base `ScheduleState` the derivation starts from |
| `moveScript` | the move / lemma applications performed (human-readable) |
| `outcomeKind` | `byte-target` \| `numeric+cost` \| `boundary` \| `typed-refusal` |
| `outcome` | the one-line claim the gate checks |
| `ladderRef` | which intrinsic-ladder exercise / in-closure claim it realizes |
| `run(ctx)` | the runnable gate — calls `ctx.oracle(...)` per checked claim |

The entry file ends with a self-run guard (`npx tsx tools/conformance/<id>.ts` runs it standalone)
and is registered in `run-all.ts`'s `CORPUS` array.

---

## The registered corpus (first entries)

All drawn from the intrinsic ladder's in-closure claims
(`kernel-editor-containment-and-ladder.md` §L3, the 12-exercise progression). Rung-ascending:

| # | Entry (`id`) | Published technique | Base → script | Outcome | Ladder |
|---|---|---|---|---|---|
| a | `chunked-reduction` | Two-pass chunked large reduction (tree-in-chunks + combine) | `deriveReductionState(sum)` → `tile(loop:reduce, factor)` | **numeric+cost**: tile splits the reduce axis into outer(chunks)⊃inner(chunk); the sum monoid ⇒ chunk-fold == single-pass sum; inverse un-tiles. **Boundary:** single-pass decoupled-lookback is OUT of closure. | ex 2 (rung 1) |
| b | `layernorm-welford` | One-pass LayerNorm variance via Welford (count,mean,M2) | naive `E[x²]−E[x]²` body → refuse+name `WELFORD_OBLIGATION` → lemma discharge | **numeric+cost**: F17 discharge round-trip + Welford pair-merge == single-pass variance <1e-10; two passes → one | ex 3 (rung 5) |
| c | `coalesced-transpose` | Coalesced transpose via shared-memory staging + padding | `deriveTiledMatmulState(NT)` → `recolor(in:a → shared)` | **numeric+cost**: recolor stages the strided operand onto shared; interior unchanged (numeric parity vs strided); inverse restores global; access-shape/coalescing cost class | ex 4 (rung 2) |
| d | `grouped-matmul` | Triton grouped-matmul program-id remapping (L2 super-grouping) | `deriveTiledMatmulState` → `program-map(grouped{axis:m, 8})` | **numeric+cost**: the R4 move applies (bijective), inverts to identity, is semantics-preserving (launch-order permutation) ⇒ L2-reuse cost, numeric parity vs ungrouped | R4 / ex 5-7 |
| e | `ksplit-epilogue-refusal` | kSplit ⊥ epilogue incompatibility (Stream-K / split-K epilogue rule) | tiled matmul with BOTH epilogue AND kSplit → `applyTiledMatmulSchedule` seam | **typed-refusal**: the seam THROWS the epilogue⊥kSplit legality error; kSplit-only and epilogue-only each realize cleanly (negative knowledge) | ex 7 (rung 7) |
| f | `online-softmax` | Online-softmax streaming — running (m,l,o) + exp(m_old−m_new) correction | stable softmax body → refuse+name `ONLINE_SOFTMAX_OBLIGATION` → lemma discharge | **numeric+cost**: F17 discharge round-trip + online recurrence == naive softmax·V <1e-9; O(S) materialized scores → O(1) carried state | ex 8 (rung 5) |

Current gate status: **6 entries, 37 oracles, PASS** (CPU-only; run `npm run conformance` for the
live count).

---

## What the corpus provably does NOT cover (the honest boundary)

The move grammar is fenced to WGSL's expressible space — the intrinsic ladder's **lower eight
rungs** (`kernel-editor-containment-and-ladder.md` §L1). The following are **permanent
out-of-closure boundaries**, not gaps to be closed by adding moves:

- **Single-pass decoupled-lookback** (Merrill & Garland 2016) — the single-pass scan/reduction that
  combines partials in one grid launch via an inter-workgroup lookback protocol. It requires a
  **cross-workgroup forward-progress guarantee** WebGPU does not provide. Recorded honestly in the
  `chunked-reduction` entry; the two-pass tree-in-chunks endpoint is in closure and near it.
- **Rungs 8–10** (asynchrony / heterogeneous executors / atom co-design): TMA/`cp.async`
  pipelining, warp specialization (`role-partition`), and WGMMA/TMEM atom co-design are CUDA-only.
  `role-partition` and `pipeline` are **reserved-but-empty** move slots on WGSL (a locked palette
  entry, not a stored degenerate value) — they exist in the grammar so the CUDA graduation is a
  schema *extension*, not a schema *break*, but no WGSL corpus entry uses them.
- **The sub-PTX floor** (SASS scheduling, control codes, register banking) — below any portable
  calculus; historically absorbed upward by compilers. Not representable and not a target.

Two representational discontinuities the corpus surfaces but does not paper over
(`kernel-editor-containment-and-ladder.md` §L1): the **lemma wall** (rung 5 — no schedule-feel
produces online softmax's correction factor; you cross it by *importing algebra*, which is exactly
what the `online-softmax` / `layernorm-welford` entries do) and the **latency inversion** (rung 8 —
where "cost = volume" stops being true; permanently past the WGSL boundary).

---

## How to add an entry

1. Pick a published technique with a real citation. Decide its `outcomeKind`: is it reachable
   (`byte-target` / `numeric+cost`), a documented boundary, or a typed refusal?
2. Create `tools/conformance/<id>.ts` exporting `module: ConformanceModule`. Use `harness.ts`'s
   schema and copy the shape of the nearest existing entry:
   - reachability via a **move**: build the named base (`derive*State` from `src/schedule/`), apply
     the move via `applyMove(state, move)`, assert `outcome.kind === "applied"`, check the resulting
     state, and round-trip the inverse (`applyInverse`). See `grouped-matmul` / `chunked-reduction`.
   - reachability via a **lemma**: build the naive body, assert `classifyBody(body)` REFUSES and
     `refusal.dischargedBy` names the obligation, then gate the lemma's numerical differential
     (`onlineSoftmaxDifferential` / `welfordDifferential` / …). See `online-softmax` /
     `layernorm-welford`.
   - a **typed refusal**: construct the illegal object, assert the seam throws, and assert each
     feature alone is legal (specificity). See `ksplit-epilogue-refusal`.
3. For a `byte-target` entry, prove `compileTileKernel(applyXSchedule(state, desc)) ===
   compileTileKernel(makeXSpec(...))` (byte-identity to the authored kernel), following the
   in-repo derivation scripts (`tools/fa-backward-derivation-script.ts`,
   `tools/fa-adam-derivation-script.ts`). Where a GPU numeric parity check is warranted, gate it
   behind `!ctx.cpuOnly` and tear Dawn down before exit (dynamic-import `initWebGPU`/`destroyWebGPU`,
   as the FA scripts do).
4. Register `module` in `run-all.ts`'s `CORPUS` array (ladder-order) and add a row to the table
   above with the technique, citation, and outcome.
5. `npm run conformance` must stay exit-0. The gate is the corpus.

## Gates this corpus must not disturb

The corpus is READ-ONLY against `src/`. It imports the schedule move algebra, lemma library,
skeletons, and canonical printers; it changes nothing in `src/`. The existing schedule
differentials and the FA derivation scripts (`tools/fa-*-derivation-script.ts`) stay untouched and
green; `npm run test:gates` (the GPU correctness gates) is independent of this corpus.
