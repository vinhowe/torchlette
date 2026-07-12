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
| **CUTOVER-FLIP** (2026-07-12, §2.0): attention (fwd/D/dQ/dKV) + Adam live-path body ABSORBED into the schedule | `make*Spec`/`makeAdamStepSpec` DELETED; live dispatch routes through `realizeAttentionSpec`/`realizeAdamStepSpec`; byte-identical ON the live path | 14-kernel attn + 5-variant adam differentials; GPU attn specs; optimizer trajectory gates; `test:gates` 6/6 |

The horizontal-`pack` move is the §3 `pack` move's **real tenant** — multi-tensor
packing at PARAMETER altitude (concatenate N param/grad/m/v flats into one
`[total]` buffer, one dispatch), which the FA forward derivation never exercised.

---

## 2. THE AUTHORED-SET EXIT STATEMENT (deliverable 4)

The schedule-state census authored set (the opaque, F3 skeletons — the ONLY place
an un-re-derived kernel may hide) after this campaign.

### 2.0 UPDATE (cutover-flip, 2026-07-12) — the LIVE-PATH ownership has FLIPPED

The named engine deliverables of P4's exit statement — attention (forward, D,
dQ, dKV) and fused Adam (all 5 variants) — have had their **LIVE-PATH cutover-flip
executed** (commits `4ff46439` attention, `efa87edb` Adam). The §6 rule-1 shrink
rule executed for the LIVE-PATH ownership fact: the kernel BODY structure now
**LOWERS FROM the ScheduleState** and the live dispatch routes through the schedule
chokepoint (`realizeAttentionSpec` / `realizeAdamStepSpec`, the P1/matmul/reduction
`realize*` pattern). The `make*Spec` / `makeAdamStepSpec` factories that formerly
owned the body are DELETED (relocated into `attention-skeleton.ts` /
`adam-skeleton.ts`); the schedule module is the SOLE WGSL writer at the dispatch
seam. Byte-identical on the LIVE path: the 14-kernel attention modifier differential
and the 5-variant Adam differential both green ON the live realize chokepoint; GPU
attention specs + optimizer-trajectory gates + test:gates 6/6 all pass.

**What this changes vs the pre-flip §2.1 below:** the two blockers §2.1 named — the
"AUTOMATED CUTOVER" of the LIVE path — are RESOLVED for **body ownership**. The
schedule now owns the kernel body on the live path; there is no longer a
second-owner factory. What the skeletons keep as `visibility:"opaque"` is the
INTERNAL move-derivation flip only (§2.1's S3 clause below, narrowed): the skeleton
stays opaque because its INTERNALS (the online-softmax composite; the locked
per-element Adam formula) are still authored — the AUTOMATED `opaque→derived` flip
of the internals waits on the S3 merge/fuse composite transaction (engine-side,
islands altitude, unbuilt), a named engine deliverable, not a grammar gap.

### 2.1 The opaque skeletons that remain (INTERNAL-derivation status)

| Family | Opaque INTERNALS? | Live-path body owner | Remaining blocker |
|---|---|---|---|
| Attention FORWARD | opaque (online-softmax composite) | **the schedule** (`lowerForwardAttentionBody`; live via `realizeAttentionSpec`) | the AUTOMATED internal `opaque→derived` flip waits on the **S3 merge/fuse composite transaction** (islands altitude — `islands-design.md §2`). The move+lemma DERIVATION is proven (`tools/fa-derivation-script.ts`). |
| Attention BACKWARD (dQ/dKV/D) | opaque (recompute+D-precompute composite) | **the schedule** (`lowerBackward*Body` / `lowerDPrecomputeBody`; live via `realizeAttentionSpec`) | same S3 internal-flip blocker; the derivation reaches the authored kernels byte-identically (`tools/fa-backward-derivation-script.ts`). |
| Fused Adam | opaque (locked per-element formula) | **the schedule** (`lowerAdamStepBody`; live via `realizeAdamStepSpec`) | the `pack` move algebra IS built (`deriveHorizontalPackedAdam`); the per-element update stays authored (a locked numeric formula), the internal flip is a mechanical-admissibility question, not a derivation gap. |

**Honest status:** the authored set is **NOT yet empty** — the three skeletons stay
`visibility:"opaque"` because their INTERNALS are authored. But the LIVE-PATH body
ownership has flipped: the schedule module owns every attention and Adam kernel body
on the live dispatch path, and the `make*Spec` second owners are gone. What remains
is the **INTERNAL automated flip** (`opaque → derived` of the skeleton's internals),
gated by:

- **S3 merge/fuse composite transaction** (attention forward + backward): the
  naive→fused region merge is an islands-altitude engine transaction
  (`fuseGesture`), UNBUILT (wave-3 report §6a; `islands-design.md §2`). This is a
  P2-macro-move / islands-engine deliverable, correctly OUT of this campaign's
  schedule-altitude scope. The derivation script drives the merge at the
  schedule/region altitude, proving the derivation; the internal flip needs the
  engine transaction.
- **Fused Adam** (per-element formula): the per-element Adam update is a LOCKED
  numeric formula (bias-correction via expm1, decoupled/L2 weight decay). Whether
  its internals ever go `derived` is a §3.3 mechanical-admissibility question, not a
  derivation gap — the horizontal-pack move (the DERIVABLE structure) already lands.

Per §6 rule 1, the authored hatch SHRANK: this campaign's cutover-flip moved
attention (fwd/D/dQ/dKV) and Adam from **derived-modulo-cutover** (the derivation
ran, but a second-owner factory still owned the live body) to **LIVE-PATH FLIPPED**
(the schedule owns the live body; the `make*Spec` factories are deleted). The NAMED
GRAMMAR FAILURES (§6 rule 2) remain ZERO; the remaining blocker is one engine
transaction (S3) with an existing design home, not a grammar gap.

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
skeleton has a running in-grammar derivation AND its live-path body now lowers from
the schedule (the cutover-flip, §2.0); the ONE remaining blocker (the S3 merge/fuse
transaction that flips the skeleton INTERNALS `opaque→derived`) is an engine
deliverable with an existing design home, not a grammar gap. Anything permanently
underivable would be a named grammar failure here — the list is empty.

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
