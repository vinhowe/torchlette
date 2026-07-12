# Schedule-state P4 — LOCAL self-hosting report

**§7 P4 (R25):** re-derive the framework's own fastest hand-crafted kernels
in-grammar at perf parity. Exit: the authored set is **empty or atoms-only**,
where "atom" passes §3.3's MECHANICAL admissibility (single primitive effect or
hardware intrinsic — no composite loop nests, no whole authored kernels). This
report is deliverable 4: **the authored-set exit statement**.

This gates the COMPLETENESS claim, not the editor. Grammar-completeness against
an EXTERNAL published-kernel conformance corpus (R25 + R4) remains a SEPARATE
standing gate; re-deriving in-repo kernels proves only those kernels.

---

## 1. What the campaign re-derived

| Deliverable | Result | Gate |
|---|---|---|
| Three lemmas (recomputation, D-precompute, Welford) | each: BoxRewrite + carried state + numerical differential at fp-exactness | `test/schedule/moves/p4-lemmas.spec.ts` (7) |
| Attention BACKWARD (dQ/dKV/D) | derived == authored **BYTE-IDENTICAL**; naive-materialized == fused to f32-exactness; perf 0.996× geomean (≤1.5×) | `tools/fa-backward-derivation-script.ts`; `test/schedule/attention-differential.spec.ts` (+6) |
| Fused Adam (horizontal-pack) | derived == authored **BYTE-IDENTICAL** (5 variants); packed == per-param bit-exact; 1 dispatch/group | `tools/fa-adam-derivation-script.ts`; `test/schedule/adam-differential.spec.ts` (13) |

The horizontal-`pack` move is the §3 `pack` move's **real tenant** — multi-tensor
packing at PARAMETER altitude (concatenate N param/grad/m/v flats into one
`[total]` buffer, one dispatch), which the FA forward derivation never exercised.

---

## 2. THE AUTHORED-SET EXIT STATEMENT (deliverable 4)

The schedule-state census authored set (the opaque, F3 skeletons — the ONLY place
an un-re-derived kernel may hide) after this campaign:

### 2.1 The opaque skeletons that remain

| Family | Opaque? | Re-derivable NOW? | Blocker (if any) |
|---|---|---|---|
| Attention FORWARD | opaque | **derivation runs** (`tools/fa-derivation-script.ts`) | the `deriveAttentionSkeleton` skeleton stays opaque until the **S3 merge/fuse composite transaction** lands (engine-side, islands altitude — `islands-design.md §2`). The move+lemma DERIVATION is proven; the AUTOMATED cutover (skeleton flips to `derived`) waits on S3. |
| Attention BACKWARD (dQ/dKV/D) | opaque | **derivation runs** (`tools/fa-backward-derivation-script.ts`, THIS campaign) | same S3 cutover blocker as forward; the derivation reaches the authored kernels byte-identically. |
| Fused Adam | opaque | **derivation runs** (`tools/fa-adam-derivation-script.ts`, THIS campaign) | the `pack` move algebra IS built; the skeleton stays opaque until the optimizer-dispatch cutover consumes `deriveHorizontalPackedAdam` on the live path (a P1-class cutover, not a derivation gap). |

**Honest status:** the authored set is **NOT yet empty**. But it is no longer
opaque-because-underivable — every member now has a RUNNING in-grammar derivation
(moves + lemmas) that reaches the authored kernel byte-identically. What remains
is the **AUTOMATED CUTOVER** (the skeleton flipping `opaque → derived` on the live
path), which is gated by:

- **S3 merge/fuse composite transaction** (attention forward + backward): the
  naive→fused region merge is an islands-altitude engine transaction
  (`fuseGesture`), UNBUILT (wave-3 report §6a; `islands-design.md §2`). This is a
  P2-macro-move / islands-engine deliverable, correctly OUT of this campaign's
  schedule-altitude scope. The derivation script drives the merge at the
  schedule/region altitude (the off-menu machinery), proving the derivation; the
  live-path cutover needs the engine transaction.
- **Optimizer-dispatch cutover** (Adam): a P1-class live-path swap (route
  `_foreachGroupStep` through `deriveHorizontalPackedAdam`), not a derivation gap.

Per §6 rule 1, the authored hatch SHRINKS as kernels are re-derived: this campaign
moved attention-backward and Adam from **underivable** (no in-grammar derivation
existed) to **derived-modulo-cutover** (the derivation runs, byte-identical; only
the automated skeleton-flip waits on a named engine deliverable). This is the
honest form of the exit for a schedule-altitude campaign — the NAMED GRAMMAR
FAILURES (§6 rule 2) are ZERO; the remaining blockers are engine transactions with
existing design homes, not grammar gaps.

### 2.2 The atoms (atoms-by-design — §3.3 MECHANICAL admissibility)

These are admissible atoms, NOT authored kernels (single primitive effect or
hardware intrinsic — they pass §3.3 mechanically, closing R25's relabel hole):

- **`atomicAdd<f32>`** — scatter-add's CAS loop. §3.3 states verbatim: "scatter-add
  is NOT an atom — it is an elementwise schedule composed AROUND the `atomicAdd<f32>`
  atom." The atom is the single primitive effect (realization as `CASLoop` vs
  `NativeAtomic` is a RECEIPT, not identity — A-R12). The scatter-add SCHEDULE
  around it is derivable elementwise; the atom itself is the irreducible primitive.
- **the enumerated subgroup primitives** (feature-gated) — each a hardware
  intrinsic with its width contract / convergence / fallback (matmul's
  `useSubgroups` decoration; A-R13). A family name without signatures cannot be
  admitted, so they are enumerated individually.

These are the ONLY members the exit criterion PERMITS ("atoms-only"). They are not
a grammar failure — they are the base of the grammar (§3.3: some primitives are
COMPOSED AROUND, never derived).

### 2.3 Named grammar failures (§6 rule 2)

**ZERO.** Nothing in the authored set is PERMANENTLY underivable. Every opaque
skeleton has a running in-grammar derivation; the two blockers (S3 transaction,
optimizer cutover) are engine deliverables with existing design homes, not grammar
gaps. Anything permanently underivable would be a named grammar failure here — the
list is empty.

---

## 3. Schema findings (operand-residency decoration axes)

**ZERO schema findings** — matching the wave-2/3 precedent. The operand-residency
decorations (the `recolor` register/shared per-operand intent) carried the vec4 dot
structure the backward kernels need (the dQ/dKV register accumulators; §6 decoration,
not magic). The `pack` move's `concatenate` kind carried multi-tensor packing at
parameter altitude. In every case the derivation reached the authored configs
byte-identically, so no decoration axis fell short. `types.ts` gained ZERO fields
this campaign (the `LemmaApplication` / `Skeleton` / `TypedParamSchema` /
`ScheduleMove` schema held all three new lemmas + both new authored families).

---

## 4. Gate matrix

See the campaign-end commit for exit codes. In-suite gates added:
`p4-lemmas.spec.ts` (7), attention-differential backward block (+6),
`adam-differential.spec.ts` (13). Derivation scripts (reproducible):
`tools/fa-backward-derivation-script.ts`, `tools/fa-adam-derivation-script.ts`.
The FA forward derivation + all prior schedule differentials remain green
(regression — the algebra did not regress under its extension).
