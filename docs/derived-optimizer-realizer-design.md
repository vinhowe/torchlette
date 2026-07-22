# The Derived Optimizer Realizer: the maximally platonic move

> **Status:** DESIGN (no `src/` change). P0 deliverable answering Vin's question
> verbatim — *"figure out the maximally platonic move for the derived optimizer
> program."* It is written **after** the chain-packing campaign STOPPED
> (`docs/chain-packing-design.md` P4/P5, `docs/coverage-campaign-design.md` §9
> C-EXIT): the graph-data-movement realization of the optimizer program is
> fenced by a measured co-materialization class (+99.9 % / +168.4 % peak). This
> doc rules on the realization mechanism that survives that fence.

---

## 1. The claim (one sentence)

**The optimizer program's maximally platonic realization is to DERIVE the fused
kernel's tile-IR body from the `OptimizerProgram` — a generic `OptTerm → tile-IR`
fold that is the exact sibling of the already-shipping `OptTerm → graph`
interpreter (`evalOptTerm`) — so the program becomes the single source of the
EXECUTED kernel while the fused path's DMA-packed, in-place dispatch skeleton
(and therefore its memory and speed) is kept byte-for-byte; this closes the last
`visibility:"opaque"` authored-kernel hatch (`adam-skeleton.ts`) that
schedule-state already committed to re-deriving, and it dodges the chain-packing
co-materialization STOP by construction because no graph-level `cat`/`narrow`
transient is ever emitted.**

The corollary that makes it *platonic* and not merely *another codegen path*:
the arithmetic is stated **once** as the `OptTerm` algebra (5 node kinds), and
BOTH consumers — the reference graph interpreter and the fast tile-IR fold —
are the **same structural recursion over that one algebra**, differentially
pinned. Today the arithmetic is stated **twice** (the `OptTerm` program AND the
hand-authored `emitAdamScalarBody`/`emitBiasCorrection` tile-IR), kept in
agreement by a byte differential — the exact "both sides recompute the value"
anti-pattern CLAUDE.md's first correctness principle names.

---

## 2. What is actually true today (the measured substrate — read before the options)

Four facts, verified from the live tree, that reprice the whole option space and
were NOT obvious from the chain-packing framing:

**F1 — the fused Adam body is ALREADY tile-IR, not a hand-written WGSL string.**
`generateAdamShaderTileIR = compileTileKernel(realizeAdamStepSpec(...))`
(`src/schedule/adam-skeleton.ts:214`). Schedule-state's 2026-07-12 cutover
FLIPPED the live body to lower from a `ScheduleState`
(`schedule-state-design.md:559-566`); the `makeAdamStepSpec` second-owner factory
was deleted. So "the framework EXECUTES Adam through a hand-written fused WGSL
kernel" is only half true: the *dispatch skeleton* is program-agnostic, and the
*body* already goes through the tile-IR compiler. The ONE thing still authored is
the **tile-IR the body emits** (`emitExpm1`, `emitBiasCorrection`,
`emitAdamScalarBody`, `adamUpdateBody`, ~200 lines) — hand-written per-Adam,
NOT folded from `ADAMW_PROGRAM`. The skeleton is stamped `visibility:"opaque"`
with the reason *"the per-element update is a locked numeric formula (not
derivable)"* (`assertAuthoredSeam`, `adam-skeleton.ts:233-238`). **The claim
"not derivable" is what this design falsifies.**

**F2 — "N params → one dispatch" is DMA scatter/gather, not a shader-visible
offset table.** `packed-dispatch.ts` groups items by element count
(`dispatchPackedGroup:143`), gets **one reused cached scratch buffer per role**
(`getPackedBuffers`, keyed `count:alignedBytes`, stable across steps),
`recordedCopyBufferToBuffer`-scatters each param/grad/m/v into it at byte offset
`i·elementBytes`, runs **one plain elementwise dispatch** over the concatenated
buffer, then gathers back. The kernel binds a fixed ~6-9 storage buffers (one per
role) **independent of N** — the WebGPU per-dispatch binding cap is dodged by
never binding N buffers, not by dynamic offsets. So the offset/segment concern
that O1 was feared to need in the IR **lives entirely outside the shader**; the
kernel is a contiguous-buffer elementwise map with `read_write` param/m/v
(in-place; `lowerAdamStepBody:466-482`). tile-IR already expresses exactly this —
**the shipping kernel is the existence proof.**

**F3 — the memory asymmetry is a property of the EXPRESSION of packing, measured.**
C-EXIT (A100 dw-2-1, whole-step regime, `coverage-campaign-design.md §9`):

| | fused (DMA-packed, in-place) | graph-packed (`cat`→interp→`narrow`) |
|---|---|---|
| distil@512 peak | 3736.6 MB (baseline) | +99.9 % |
| medium@512 peak | 9683.3 MB (baseline) | +168.4 %, donation-invariant |

The premium is the graph route's full-model `G`/`P`/`pNew` **graph temporaries**,
which the planner co-materializes with live activations in the merged whole-step
plan. Donation (`pNew→P`) touched ~0 of it. The fused path has no such temps: its
packed scratch is reused, its state is in-place, and it runs as a **separate
optimizer dispatch segment** after activations are freed. **Any realization that
keeps the fused dispatch structure inherits its peak by construction.**

**F4 — `evalOptTerm` already folds the program to graph ops.** `OptTerm` is 5
node kinds (`role`, `c`, `u` unary, `b` binary, `mm`; `optimizer.ts:42-56`).
`evalOptTerm` (`optimizer.ts:211-323`) is a `switch(t.k)` that recurses and emits
`rt.add/mul/div/sqrt/neg/sign/abs/exp` (scalar subtrees fold in JS; `mm` throws if
either side folds to scalar). tile-IR carries **every one** of those elementwise
ops as `BinaryOp`/`UnaryOp` node kinds (`tile-ir.ts:74-105`), and carries `matmul`
ONLY as a 2-D **block-op** (shared-memory tiled dot), never as an elementwise
expression node. So a sibling fold `OptTerm → tile-IR` is (a) structurally
identical to `evalOptTerm` and (b) **structurally incapable of lowering `mm`** —
the Muon refusal falls out of the type system, not a special case.

---

## 3. The option space, priced

Every option is measured against the house criteria: *single source at seams
(derive, don't co-recompute); no second substrate where one extends; deletions
named; typed refusals; capability-per-SLOC; STOP-class exposure.*

### O1 — `OptTerm → tile-IR` fold; keep the DMA-packed in-place skeleton

Add `lowerOptTermToTileIR(program, ctx)` — the sibling of `evalOptTerm`, mapping
each `OptTerm` node to its tile-IR expression node. The `adam-skeleton` body
STOPS being hand-authored (`emitAdamScalarBody` &co. deleted) and instead calls
the fold over `ADAMW_PROGRAM`; the skeleton (schema, bindings, `read_write`
in-place, `t`/`lr`/`bc1`/`bc2` as scalar DATA) and `packed-dispatch.ts` are
untouched. Lion/SGD route their programs through the same seam and get a fused
packed kernel they do not have today.

- **Mechanism added:** one generic fold (~120-160 SLOC), structurally a clone of
  `evalOptTerm`. **Zero new codegen substrate** — reuses tile-IR wholesale.
- **Mechanism deleted:** the Adam-specific tile-IR authoring (`emitExpm1`,
  `emitBiasCorrection`, `emitAdamScalarBody`, `adamUpdateBody`, `ADAM_PARAM_SCHEMA`
  arithmetic, ~200 SLOC) — the `opaque` internals; the skeleton flips
  `opaque→derived` (the S3 follow-on schedule-state already declared,
  `schedule-state-design.md:575`).
- **Doctrine fit:** MAXIMAL. It is the 8th application of "the latent decision
  becomes an object" and the exact exit schedule-state pre-committed for fused
  Adam (`schedule-state-design.md:554-556`). Program = single source; two folds
  of it; differential becomes *derive-and-assert* (both sides fold one source),
  not *author-two-and-assert*.
- **Memory / speed:** fused-EQUAL by construction (§2 F2/F3 — same skeleton).
- **STOP exposure:** DODGES all three fenced classes by construction (§6).
- **Migration risk:** LOW. Behind a flag; the fused body is the differential
  oracle for its own re-derivation; the skeleton/dispatch never change.

### O2 — direct `OptTerm → WGSL` string codegen

A dedicated serializer from `OptTerm` into the body slot, bypassing tile-IR.

- **Mechanism added:** a SECOND WGSL author living beside tile-IR — and it must
  re-implement tile-IR's autovectorize, f16, chunking, binding-order, and
  scalar-table machinery to reach parity.
- **Doctrine fit:** DIRECT VIOLATION. `execution-declaration-design.md §0`:
  *"there is exactly one description of how an op executes."* A bespoke WGSL
  emitter is a second description of the substrate the whole codebase spent
  seven campaigns unifying onto tile-IR.
- **Verdict:** strictly dominated by O1 (same output, second substrate). LOSES.

### O3 — keep the authored body; promote the differential to a certification

No codegen. Keep `emitAdamScalarBody` hand-authored; make the `OptTerm` program a
*specification* the kernel is *certified* against by an automated
interpret-both-over-test-vectors harness.

- **Mechanism added:** ~0 (the certification harness already exists —
  `test/schedule/adam-differential.spec.ts` + `fused-vs-elementwise.spec.ts`).
- **Doctrine fit:** WEAK — and it is **the current fenced equilibrium**. C-EXIT
  left exactly this: *"the fused `adamStep` monolith keeps its live consumer …
  asserted against the derived packed reference."* O3 is a rename of the status
  quo, not a move. It is *assert-agreement between two independently-authored
  artifacts* — the weakest form of single-source, and the precise anti-pattern of
  CLAUDE.md principle #1 (*"never let both sides independently recompute … derive
  from ONE source"*). The program never becomes the source of the EXECUTED
  kernel; it stays a co-equal oracle. A change to `ADAMW_PROGRAM` does not
  propagate — it merely fails the cert.
- **Verdict:** the honest FLOOR and the safe fallback if O1's fold ever proves
  intractable — but not platonic. LOSES to O1 on doctrine; kept as the P0 exit.

### O4 — a first-class in-place segmented graph primitive

A new graph op ("segmented elementwise map over a buffer set, in place") the
packer emits **instead of** `cat`/`narrow` — the graph route without data
movement.

- **Mechanism added:** a new multi-input/multi-output in-place graph NODE that
  the planner, liveness, donation, fusion-detect, CSE, and stream-generate must
  all learn. To be in-place it must bind N buffers per dispatch → hits the
  WebGPU binding cap → forced back onto DMA scatter/gather → **it becomes
  `packed-dispatch.ts` wearing a graph-node hat, carrying an `OptTerm` payload.**
- **Doctrine fit:** RE-ENTERS the mega-op sin. `architecture-debt.md §1` names
  `adamStep` — a single graph node with an arithmetic payload + N in-place
  bindings — as *the* op-granularity-inversion defect. O4 rebuilds it, generic
  over `OptTerm`. Every side channel (payload, in-place write-set, CSE exemption)
  it needs is a seam the ledger already bled on.
- **Verdict:** more mechanism than O1 for the same in-place result, and it
  re-enters a fenced sin class. LOSES.

---

## 4. THE RULING — O1

**O1 is the maximally platonic move.** It is the unique option that makes the
`OptimizerProgram` the single source of the *executed* kernel (not merely a
reference oracle, as O3) **without** a second codegen substrate (O2) or a
resurrected mega-op (O4), and it inherits the fused kernel's memory and speed by
KEEPING the fused kernel's dispatch skeleton — the only thing it changes is the
**provenance of the tile-IR body**, from hand-authored to program-folded.

Why the losers lose, in one line each:

- **O2 loses** because it is a second WGSL author beside tile-IR — the "no second
  substrate where one extends" rule, violated at the point of writing.
- **O3 loses** because it is the status quo renamed: two authored artifacts
  asserted-equal, which is co-recomputation, the #1 anti-pattern. It never moves
  authority to the program.
- **O4 loses** because in-place + N-buffer + arithmetic-payload IS the `adamStep`
  mega-op, generalized — it re-enters the op-granularity-inversion sin and taxes
  six subsystems O1 leaves untouched.

Why O1 is *maximally* platonic by the house criteria, positively:

1. **Single source, derived not asserted.** One algebra (`OptTerm`), two folds
   (`evalOptTerm`→graph reference, `lowerOptTermToTileIR`→fast kernel). The
   differential pins *two derivations of one source*, strictly stronger than
   today's *two authorings, one assertion* (`architecture-debt.md §1`, the
   "silently forked semantics" that L2-vs-decoupled AdamW bug was).
2. **No second substrate.** The fold targets tile-IR, the substrate seven prior
   campaigns already chose (`execution-declaration-design.md §1`). It is the
   8th "latent decision becomes an object."
3. **It is a pre-committed exit, not a new debt.** `schedule-state-design.md:554`
   lists fused Adam as an authored member slated to *re-derive*; rule 1 there
   mandates the authored set shrink monotonically. O1 executes that shrink; the
   skeleton flips `opaque→derived`.
4. **Capability-per-SLOC rises.** One ~150-line fold DELETES ~200 lines of
   Adam-specific tile-IR authoring AND gives Lion/SGD/Muon-elementwise a fused
   packed kernel they lack today (they currently run per-param graph chains). One
   seam, four clients, zero per-optimizer WGSL.
5. **Typed refusal is structural.** `mm` has no elementwise tile-IR node (§2 F4),
   so the fold *cannot* lower Muon's Newton-Schulz — the existing
   `OptimizerPackRefusal`/`assertFlattenable` (`pack-optimizer.ts:141`) is
   enforced by the type system at the fold seam, not a name-check.

**The reframe that makes O1 succeed where chain-packing STOPPED:** chain-packing
tried to make the *program* the source by expressing packing as **graph data
movement** (`cat`→`narrow`), which co-materializes `G`/`P`/`pNew` and blows peak
+168 % (§2 F3). O1 makes the *program* the source by expressing packing as a
**compiled DMA-packed kernel** — the fused path's own structure — so packing is
never graph data at all. Same platonic goal (program = single source), opposite
memory outcome, because the packing is realized below the graph, exactly where
the fused kernel already realizes it.

### The bias-correction sub-call (the one genuine design fork inside O1)

`bc1 = 1−β1^t`, `bc2 = 1−β2^t` are per-step DATA. Two ways to feed the folded
kernel; pick B:

- **A (perf-conservative):** bind `t`/`lr` as 1-element tensors (as today) and
  keep an in-kernel `expm1` bias-correction prelude. But that prelude is
  Adam-specific WGSL that the *generic* fold would not produce — it would remain
  an authored fragment. Rejected: it leaves a hatch open.
- **B (platonic — RECOMMENDED):** bind `bc1`/`bc2` as scalar-DATA roles, computed
  graph-side by `_biasCorrection` (already exists, `adam.ts:551` — the foreach
  path already does this). Then the fold is *purely* the elementwise arithmetic of
  `ADAMW_{M,V,P}_NEW` over roles; `emitExpm1`+`emitBiasCorrection` die entirely;
  the frozen-scalar class stays impossible because `bc1`/`bc2` are graph tensors
  under TAG_WRITE re-execution and the generic scalar-table. Cost: two extra
  4-byte bindings and a negligible 2-scalar graph prelude per step.

---

## 5. Phase plan (each shippable, differential-gated)

**Standing gate set (every phase, before landing):** `npm run test:gates`;
`tools/parity-fullstack-tl.ts` compiled-vs-lowered twice (`TORCHLETTE_COMPILED_PLAN=0`
vs default), ≤1e-5 / 30 steps; the 124M DiLoCo regression
`{0:9.81, 3:5.92, 6:5.15, 9:4.64}` EXACT; distil 9 / medium 18 submits EXACT
(`TORCHLETTE_PROFILE=1 … tools/profile-training.ts`, LATE steps). GPU work is
serial-exclusive (`tools/pick-gpu.sh`, HOST node toolchain).

| Phase | What lands | Deletes / cashes | Phase-specific gate |
|---|---|---|---|
| **R1** | **The fold, dark.** `lowerOptTermToTileIR(program, ctx)` in `src/schedule/` — the structural sibling of `evalOptTerm`, each `OptTerm` node → tile-IR expression node; `mm` → typed `OptimizerPackRefusal` (structural — no elementwise target). NOT wired into any live kernel. | Nets +fold, deletes nothing yet. | **New differential `tools/optterm-fold-parity.ts`:** for `ADAMW/LION/SGD_*_PROGRAM`, the fold-compiled kernel over a random flat buffer == `evalOptTerm` graph result, ≤1e-6. Muon → asserts the refusal. |
| **R2** | **Derive the Adam body from the fold, behind `TORCHLETTE_DERIVED_ADAM`.** `realizeAdamStepSpec` builds its body via `lowerOptTermToTileIR(ADAMW_PROGRAM)` instead of `emitAdamScalarBody`; `bc1`/`bc2` bound as scalar-DATA roles (fork B). Flag defaults OFF; authored body stays as the oracle. | Cashes nothing yet (flag off). | **`adam-differential.spec.ts` promoted:** derived-body == authored-body across all 5 variants, byte-identical WGSL where the algebra is identical / ≤1e-7 where reassociated. `fused-vs-elementwise.spec.ts` (12 cells) green both flag states. |
| **R3** | **THE FLIP.** `TORCHLETTE_DERIVED_ADAM` defaults ON. The authored `emitAdamScalarBody`/`emitBiasCorrection`/`emitExpm1`/`adamUpdateBody` become unreferenced; the skeleton stamps `visibility:"derived"`. Re-measure A100 peak fresh. | Cashes the authored-body default. | **THE EXIT GATE (§ below).** Plus: derived-body trajectory == authored-body over 30 steps across the compiled-plan activation threshold (Corollary 2). |
| **R4** | **THE DELETION.** Remove `emitAdamScalarBody`, `emitBiasCorrection`, `emitExpm1`, `adamUpdateBody`, the `opaque` authored-hatch machinery for Adam, and `TORCHLETTE_DERIVED_ADAM`. | Net-negative src SLOC (§6). | Full suite green; `bash tools/weight-norm.sh --log` net-negative src vector; the fold-parity differential is the surviving guard (the deleted authored body was the oracle). |
| **R5** | **Lion + SGD(+momentum) route through the fold.** They currently emit unfused per-param graph chains; give each a fused packed kernel via the same seam + `packed-dispatch.ts`. Zero per-optimizer WGSL. | Cashes the per-param chain loops in `lion.ts`/`sgd.ts`. | Lion `test/lion-distil-descent.spec.ts` (7.89→4.36) reproduced; a Lion run's submits ≤ the Adam-class budget for equal param count; `optterm-fold-parity` extended. |

Muon needs no phase: `MUON_PROGRAM` carries `mm`, the fold refuses structurally,
and Muon's internal AdamW sub-instance inherits Adam's derived routing for free
(same seam it already inherits, `chain-packing-design.md §6.1`).

### The exit gate (R3), stated as a number

The derived-body Adam kernel must match the **authored fused kernel's A100
peak to within measurement noise** — because it IS the fused dispatch structure,
this holds by construction and the gate confirms no regression:

> **distil@512 ≤ 3736.6 MB and medium@512 ≤ 9683.3 MB** (whole-step regime,
> dw-2-1 A100, C-EXIT baseline), equivalently the steady-state CLAUDE.md line
> **distil ~4.67 GB / medium ~13.47 GB, 9 / 18 submits, ~50 / ~190 ms/step.**

This is the number chain-packing could not reach (+99.9 %/+168.4 %). O1 reaches
it because it does not enter the graph-packing regime at all. The gate is
peak-parity, submit-parity, and the byte/trajectory body-differential — NOT a
memory *improvement* (there is none to win; the fused peak is already the floor).

---

## 6. Deletion ledger

Cashed at R4 (+ R5 for Lion/SGD), sized from the live files:

| Target | Location | Lines | Why it dies |
|---|---|---|---|
| `emitAdamScalarBody` | `src/schedule/adam-skeleton.ts:676-724` | ~48 | the per-element arithmetic is the folded `ADAMW_{M,V,P}_NEW` |
| `emitBiasCorrection` + `emitExpm1` | `adam-skeleton.ts:639-673` | ~35 | `bc1`/`bc2` are scalar-DATA roles from graph-side `_biasCorrection` (fork B) |
| `adamUpdateBody` (semantic op-tree) | `adam-skeleton.ts:280-353` | ~74 | it was the horizontal-pack model of the same math; `OptTerm` is that model |
| `loadAdamUniforms` bias fields, `ADAM_PARAM_SCHEMA` arithmetic | `adam-skeleton.ts` | ~30 | folded body binds roles, not an Adam-specific uniform block |
| `opaque` authored-hatch assertions for Adam | `assertAuthoredSeam` Adam branch | ~20 | skeleton is `derived`; no authored second-copy to guard |
| `TORCHLETTE_DERIVED_ADAM` flag | env accessors | — | born-with-sunset; dies at R4 |
| per-param chain loops (R5) | `lion.ts:149-206`, `sgd.ts` | ~100 | replaced by one grouped fold call |

**Net:** ~230 authored-arithmetic lines (R4) + ~100 per-param lines (R5) deleted,
against ~150 for the generic fold ⇒ **net-negative src SLOC**, with the one
net-new mechanism (the fold) warranted because it is the single seam serving
four clients with zero per-optimizer WGSL (capability-per-SLOC rises). Every
phase names its deletions in the commit (house policy).

**What this ledger does NOT claim** (honesty against the chain-packing dream):
it does **not** delete the fused dispatch skeleton, `packed-dispatch.ts`, the
`adam-batch` action, or the ~1,300-1,600 SLOC chain-packing P5 targeted. Those
retain a live consumer (the memory budget) that O1 does not remove — O1 keeps
the skeleton *on purpose*, because the skeleton is what gives the fused path its
peak. O1's deletion is smaller than chain-packing's, but it is **reachable**,
whereas chain-packing's larger deletion is STOPPED. The platonic win here is
*provenance* (program → executed body), not *skeleton removal*.

---

## 7. Risks, refusals, and STOP classes dodged BY CONSTRUCTION

### 7.1 Prior STOP classes — each dodged, structurally

- **Chain-packing P4/P5 co-materialization (`coverage §9`, the +168 % class):**
  DODGED. O1 emits no graph `cat`/`narrow`; there is no `G`/`P`/`pNew` graph
  temporary to co-materialize. The optimizer stays a DMA-packed in-place kernel
  segment — the exact structure C-EXIT measured at the baseline peak. The re-open
  condition the STOP named — *"a mechanism that attacks co-materialization
  itself, not buffer aliasing"* — is satisfied trivially: co-materialization
  never occurs because the transients never exist.
- **Donation P2 (+0 payoff):** DODGED — irrelevant. No `pNew→P` donation is
  needed; the state is `read_write` in place already.
- **Coverage C-EXIT:** DODGED — not a coverage question. The fused kernel is
  already covered/compiled; O1 changes its body's provenance, not its coverage.
- **Frozen-scalar / volatile-uniform (architecture-debt classes 1-4):** DODGED.
  Fork B makes `lr`/`bc1`/`bc2` graph-tensor DATA under the generic scalar table +
  TAG_WRITE; the folded kernel carries **no** per-step config to bake. This
  DELETES the `setAdamConfigUniforms` bias mapping rather than extending it — the
  volatile-repack special-case shrinks.
- **Everest P4b / D4 "witnessing ≠ compilation" ledger STOPs:** N/A — O1 deletes
  nothing from the recorded-build / coverage machinery; it re-sources one kernel
  body. The finiteness/coverage assumptions those STOPs protect are untouched.

### 7.2 Named risks

- **R1 — fold reassociation drift.** `evalOptTerm` folds scalar subtrees in JS
  (`1−β1` stays a constant); the tile-IR fold must fold identically or fp
  reassociation diverges from the authored body. *Mitigation:* the fold shares
  `evalOptTerm`'s constant-folding rule (same recursion); the R2 gate is
  byte-identical WGSL where the algebra matches, ≤1e-7 where the authored kernel
  deliberately reassociated (e.g. `step_size = lr·√bc2/bc1`). Name any
  reassociation as a fold lemma, do not silently match.
- **R2 — touching a live optimizer** (the framework's worst outcome). *Mitigation:*
  R1-R2 land dark behind the flag; the authored body is the differential oracle
  for its own replacement; the flip (R3) is gated on the trajectory differential
  across the compiled-plan threshold; the authored body is deleted (R4) only
  after the derived body is the proven default.
- **R3 — vec4 / f16 / unscale variants.** The authored body has a manual-vec4
  unscale path with `atomicMax` inf detection (`adam-skeleton.ts:533-609`) that is
  NOT pure elementwise arithmetic. *Mitigation:* the fold owns only the *scalar
  arithmetic body*; vectorization, f16 emission, and the atomic-inf grid-stride
  wrapper stay skeleton concerns (they are program-agnostic dispatch policy, the
  right altitude). The fold plugs the arithmetic into the existing
  `realizeAdamStepSpec` variant machinery; it does not re-derive the wrapper.
  This keeps the 5-variant `adam-differential` gate meaningful.
- **R4 — the fold becomes an Adam-shaped fold.** If `lowerOptTermToTileIR` grows
  Adam-specific branches (a bias-correction prelude, an unscale hook), it stops
  being generic and the capability-per-SLOC argument collapses. *Mitigation:*
  fork B (no in-kernel prelude); the fold's ONLY input is an `OptTerm` and a role
  binding; assert it has no optimizer-name-keyed branch (the same discipline
  `evalOptTerm` already holds).

### 7.3 Typed refusals (unchanged, now enforced at the fold seam)

- **Any `mm`/contraction node** → `OptimizerPackRefusal`, structural (no
  elementwise tile-IR target). Muon's NS core refuses; its elementwise ends and
  its AdamW sub-instance route normally.
- **Group of size 1 / membership-layout change / mixed dtype-wd-LR class** →
  per-param path, exactly as `packed-dispatch.ts`/`pack-optimizer.ts` already do.

---

## 8. Genuine taste-calls (unresolvable from code/measurement)

1. **Bias-correction fork A vs B.** Code and doctrine both point to B (fully
   scalars-as-data, closes the hatch completely). The only argument for A is if
   the two extra 4-byte bindings + 2-scalar graph prelude ever measure as a
   real per-step cost at 124M — measured elsewhere as negligible, but it is a
   perf/purity trade the author (Vin) may want to rule on. **Recommendation: B.**
2. **How far to generalize the fold in v1.** R1-R4 prove the fold on Adam alone;
   R5 cashes Lion/SGD. One could stop at R4 (Adam re-derived, hatch closed,
   net-negative SLOC already) and treat R5 as a follow-on campaign. The
   capability-per-SLOC argument is strongest *with* R5 (four clients, one seam),
   but R4 is a complete, shippable platonic result on its own. **Recommendation:
   land R1-R4 as the campaign; R5 as the immediately-following cash-in, same
   seam.**
3. **Whether O3 is "platonic enough" to be the terminal state.** This design
   rules NO — O3 is co-recomputation, the #1 anti-pattern — but O3 IS where
   C-EXIT left the tree, and if the fold's fp-reassociation parity (R1) proves
   genuinely intractable at some variant, O3 is the honest floor to fall back to.
   The ruling is O1; the fallback is a *named* O3, not a silent one.

---

## Appendix — grounding (verified this session, not re-measured)

- Fused Adam body is tile-IR: `generateAdamShaderTileIR = compileTileKernel(
  realizeAdamStepSpec(...))` (`adam-skeleton.ts:214`); live-body cutover
  2026-07-12 (`schedule-state-design.md:559`).
- Packing is DMA scatter/gather over reused per-role scratch, ~6-9 bindings
  independent of N (`packed-dispatch.ts:143-195`); kernel binds one contiguous
  buffer per role, `read_write` in-place (`adam-skeleton.ts:466-482`).
- tile-IR carries all 9 OptTerm elementwise ops as node kinds
  (`tile-ir.ts:74-105`); `matmul` only as a block-op, never elementwise — the
  Muon refusal is structural.
- `evalOptTerm` is the graph-fold sibling (`optimizer.ts:211-323`);
  `_biasCorrection` already computes `bc1`/`bc2` graph-side (`adam.ts:551`).
- Memory asymmetry: fused peak distil 3736.6 MB / medium 9683.3 MB; graph-packed
  +99.9 % / +168.4 %, donation-invariant (`coverage-campaign-design.md §9`,
  A100 dw-2-1 whole-step).
- Doctrine: fused Adam listed as an authored member to re-derive
  (`schedule-state-design.md:554-556`); authored set shrinks monotonically
  (rule 1, `:542`); `opaque` reason "locked numeric formula (not derivable)"
  (`adam-skeleton.ts:233`) — the claim this design falsifies.

**No fresh GPU probe was required:** the load-bearing feasibility fact ("tile-IR
can express an in-place multi-buffer packed elementwise kernel") is proven by the
LIVE shipping Adam kernel, which is exactly that. The memory exit number is the
existing C-EXIT A100 measurement. Measured facts, not speculation.

---

## Implementation status (R1–R4) — updated 2026-07-22

- **R1 — the fold, dark: LANDED.** `src/schedule/optterm-fold.ts`
  (`lowerOptTermToTileIR`), the structural sibling of `evalOptTerm`; `mm` refuses
  structurally via the existing `OptimizerPackRefusal`. Gates:
  `test/schedule/optterm-fold.spec.ts` (catalog, 7/7) + `tools/optterm-fold-parity.ts`
  (fold-compiled kernel == `evalOptTensor`, ≤1e-6: adamw 3.08e-7, sgd/sgd_momentum/lion
  ~1.2e-7; muon refuses). Net src +77.

- **R2 — derived Adam behind `TORCHLETTE_DERIVED_ADAM` (fork B): LANDED, flag OFF.**
  The fused body folds `ADAMW_M_NEW/V_NEW/SCALED`; `bc1`/`bc2` ride in as a `[2]`
  `bc` DATA input REPLACING `t` at input slot 4 (node arity kept at 6 → every
  executor/replay/batch seam structurally unchanged), so the in-kernel `expm1`
  prelude dies. Two NAMED reassociation lemmas (L1 `(g·g)·(1−β2)`; L2 divide-inside-
  sqrt vs the authored √bc2-factored step_size). Gates: authored byte-differential
  INTACT (`adam-differential` 13/13); `tools/derived-adam-parity.ts` derived==authored
  Δparam ≤2.98e-8, Δm=Δv bit-exact; `fused-vs-elementwise` 12/12 BOTH flag states;
  `test:gates` 5/5 BOTH (compiled==lowered for the derived path — bc-in-plan +
  stream-generate bc alias validated); 124M regression EXACT both states
  ({0:9.81,3:5.92,6:5.15,9:4.64}, memory identical). Net src +108, envFlags +1.

- **R3-FIX — candidate #1 LANDED; flip RE-CONFIRMED STOPPED (named class: reclaim-
  boundary quantization).** The R3 re-open attempt. Two things settled.

  1. **Candidate #1 landed (persistent `[2]` bc buffer, no per-step `cat`).**
     `Adam._biasCorrectionPacked` computes bias correction as ONE vectorized `expm1`
     chain over a persistent `[2]` `_lnBetas` constant (`t` broadcast over `[2]`),
     writing IN-PLACE into a persistent `[2]` `_bc` buffer via `copy_` — the exact
     `_t`/lr persistent-state pattern (buffer-stable, post-`copy_` ref so the adamStep
     nodes depend on it and it re-executes each replay). This replaces R2's two scalar
     `expm1` chains + `runtime.cat([bc1,bc2])` (fresh buffer identity every step).
     Effect: the optimizer plan's steady prelude drops from ~35 to ~18 nodes (medium
     optimizer plan **623→606**), the per-step allocation and buffer churn are gone,
     numerics UNCHANGED (derived-adam-parity Δparam 2.98e-8, Δm=Δv bit-exact —
     identical to R2; gates 5/5 both states; fullstack both arms ≤2e-6/30; packed-vs-
     unpacked 4/4; 124M EXACT with the flag ON). A100 A/B (dw-2-1, @512, steady):
     **distil CLEAN — submits 8=8 EXACT, peak 5636.6=5636.6 EXACT.** But **medium still
     submits 19→20, peak 16189→16989** — candidate #1 did NOT cross the gate.

  2. **Root cause of the medium +1 submit / +800 MB, MEASURED (not a packing miss).**
     The bc subgraph DOES pack — the adamStep batch stays intact (76→76 grouped on
     distil), the `cat` was never the issue. The real mechanism is **reclaim-boundary
     quantization**: the lowered plan inserts a reclaim (`flushSharedEncoder` +
     `flushBufferPool` = one submit) every **300 nodes** (`executor.ts:3821`, hardcoded
     when liveness-release is on — NOT the `TORCHLETTE_RECLAIM_INTERVAL` env default).
     Medium's optimizer plan is **586 nodes authored → 1 reclaim** (at node 300);
     fork-B's ~20-node graph prelude tips it to **606 → 2 reclaims** (nodes 300 AND
     600). The 2nd reclaim is the +1 submit; its warmup pool-flush placement is the
     +800 MB transient (steady *current* memory is identical, 7247≈7246). Distil's
     optimizer plan (172 nodes) never reaches 300 → 0 reclaims → clean regardless.
     Confirmed by submit-stack attribution: the extra flush is `executor.ts:3221`
     (`reclaim` action), +1/step on medium.
  - **The reclaim is LOAD-BEARING — it cannot be suppressed.** A trailing-reclaim
    guard (skip a reclaim landing within 64 nodes of the plan end — the node-600
    reclaim is only 6 nodes before the plan-final flush) was implemented, and it DID
    restore medium to **19 submits EXACT**. But it **regressed steady current memory
    +1460 MB** (8709 vs 7246, +292 storages, permanent — not a warmup transient):
    the mid-plan reclaim's `flushSharedEncoder` is what makes nodes 301–600's freed
    buffers reusable WITHIN the same plan; without it the plan allocates fresh for
    nodes 601+ and defers promotion a full step. The plan-final `advanceEpoch` flush
    does NOT recover this (it promotes `pendingRelease` but cannot enable intra-plan
    reuse after the fact). Guard REVERTED.
  - **Why the prelude cannot shed the 7 nodes needed (606→≤599).** The ~18 nodes are
    the two-branch `expm1` (`abs`/`lt`/`where` selecting a 5-term Horner series for
    small `|y|` vs `exp(y)−1` otherwise) the authored kernel emits IN-KERNEL for free.
    Both branches are live across the trajectory (bc2 with β2=0.999 needs the series;
    bc1 with β1=0.9 enters the `exp` branch by t≈3), and the series must match the
    authored 5-term form bit-exactly (parity oracle). No bit-exact reduction reaches 7
    nodes. Fork B's graph-side prelude is irreducibly ~18 nodes.
  - **Verdict:** fork B (bias correction as a graph subgraph) STRUCTURALLY cannot meet
    the "submits EXACT + peak parity" gate on gpt2-medium — its irreducible prelude
    tips the optimizer plan across a reclaim-interval boundary, and the resulting
    reclaim is load-bearing for intra-plan buffer reuse (removing it trades the submit
    for a larger steady-memory regression). This is NOT numerical (trajectory correct
    everywhere) and NOT a fusion-packing failure (the batch packs). Re-confirmed STOP.

- **R4 — the deletion: NOT REACHED** (depends on the flip). The authored body is
  retained as the differential oracle; `TORCHLETTE_DERIVED_ADAM` stays OFF by default,
  so the shipping default is unchanged. envFlags unchanged (+1 from R2).

**Re-open condition for R3 (revised — the named path is fork C, not more fork-B
scheduling):** eliminate the graph prelude's plan nodes ENTIRELY by delivering `bc`
as a host-computed `[2]` **live scalar** (the same primitive `lr` already rides:
`core/live-scalar.ts` — a persistent buffer host-written per step, graph-ordered,
replay-safe via the scalar table / TAG_WRITE). Host-side `expm1F32` from the host step
counter → zero optimizer-plan nodes → medium optimizer plan returns to 586 → submits
19 EXACT and peak 16189 by construction. The BLOCKER to validate before landing fork C
is **precision**: `bc` then compares host-`expm1F32` against the authored kernel's
GPU `exp` intrinsic (not GPU-vs-GPU as today), so the `derived==authored` Δparam parity
(currently ≤3e-8 via named lemmas L1/L2) must be re-established or a new NAMED lemma
bounding host-vs-GPU `exp` agreement stated — do not land fork C on an unnamed
reassociation. Fork B's mechanism (R1+R2, candidate #1) stays the retained dark path;
it is proven correct and is the best starting point, but it cannot be the flipped
default on large-param models.

---

## Fork C — LANDED, R3 FLIPPED, R4 DELETED (2026-07-22)

**The precision blocker CLEARED.** Measured across the trajectory (t=1..50000, both
betas) with `tools/forkc-precision-probe.ts` (V100 sivri):

- **Named lemma L3 (host-vs-GPU exp, bc-absorbed).** The Horner branch (|y|<0.25) is
  host-vs-GPU **bit-exact** (ULP=0) — the host f32 mirror (`Math.fround` discipline)
  matches the GPU tile-IR f32 ops; no divergent FMA contraction on this substrate. The
  exp branch (|y|≥0.25) diverges by the GPU `exp` intrinsic's accuracy (worst raw 32
  ULP at t=500 β1 where y=−52.68 and exp≈1.3e-23), BUT that divergence is **absorbed by
  bc=1−exp(y)**: where exp is least accurate (deep tail) bc saturates to exactly 1.0
  (error far below the f32 ULP at 1.0); where exp contributes (branch boundary) it is
  ~1 ULP accurate. **Net: bc agrees host-vs-GPU to ≤ 1 ULP (abs ≤ 2.98e-8) everywhere.**
- **Propagation.** bc enters ONLY the param update (m_new/v_new are bc-free), so Δm=Δv
  are **bit-exact (0.0)** and the ≤1-ULP bc divergence yields **Δparam ≤ 5.96e-8**
  (= 2^-24, one f32 ULP at param scale), wd-mode invariant. This RESTATES the R2 bound
  2.98e-8 → 5.96e-8 (2×, the trajectory-wide worst vs R2's single STEP=7 point); still
  ≤1e-6 gate tol and inside the design's ≤1e-7 update-magnitude claim. Lemmas L1/L2
  (kernel reassociation) are unchanged and still named; L3 is the new host-vs-GPU exp
  lemma. This is verdict (ii) — a tight named lemma, effectively as good as bit-exact
  for bc (the authored oracle itself uses the same ~ULP GPU exp intrinsic).

**Mechanism.** `bc=[bc1,bc2]` is computed on the HOST from the JS step counter
`_tHost` (`Adam._bcHost` → `expm1F32`, the SAME 5-term Horner / exp branch the authored
kernel emits) and delivered as a `[2]` **LIVE SCALAR** — the `lr` primitive
(`core/live-scalar.ts`), generalized from f32[1] to f32[n] (one primitive; the length
is the constructor's). ZERO optimizer-plan expm1 nodes; fork B's ~18-node graph prelude
(`_biasCorrectionPacked`/`_lnBetas`/`_bc`) is DELETED. The reclaim-boundary quantization
fork B tripped on is dodged by construction.

**Fork-C generalization touched four files** (the live-delivery seam, all symmetric with
`lr`): `core/scalar-slots.ts` (slot value `number | readonly number[]`),
`runtime/engine.ts` `setScalarInPlace` (vector source), `executor/step-tape-replay.ts`
(vector re-dress), `core/live-scalar.ts` (f32[n]). Adam: `_bcHost` + `_tHost` +
`_bcLive`, minus `_biasCorrectionPacked`/`_lnBetas`/`_bc`.

### R4 — the deletion (LANDED 2026-07-22)

The authored per-element arithmetic is DELETED — it was the differential oracle for its
own re-derivation, retired once fork C became the proven default. Deleted from
`src/schedule/adam-skeleton.ts`: `emitAdamScalarBody`, `emitBiasCorrection`, `emitExpm1`
(the authored per-element + expm1 bias-correction WGSL), and the dead `adamUpdateBody`
(exported, no consumer). Collapsed to derived-only: `lowerAdamStepBody` (always binds
`bc`, always `emitDerivedBiasCorrection`/`emitDerivedAdamScalarBody`), `realizeAdamStepSpec`,
`generateAdamShaderTileIR` (the `derived` param + branches gone). The
`TORCHLETTE_DERIVED_ADAM` flag is DELETED (all four reads; envFlags −1). `deriveAdamSkeleton`'s
refusalReason flips authored→derived (the body is now a theorem of `ADAMW_PROGRAM`); the
skeleton stays an opaque dispatch wrapper (F3) — `assertAuthoredSeam` is shared with
attention-skeleton and unchanged.

**Surviving guards** (the deleted authored body was the oracle): the fold-parity
differential `tools/optterm-fold-parity.ts` (fold == `evalOptTerm` program interpreter;
adamw 3.077e-7) + `test/schedule/optterm-fold.spec.ts` (catalog). Retargeted: the byte-
differential `test/schedule/adam-differential.spec.ts` now proves the two entry points
(schedule chokepoint vs direct generation) agree on the DERIVED body (single-source);
`test/schedule/derived-adam.spec.ts` asserts the derived kernel's standing structure
(binds `bc`, no in-kernel expm1). Deleted `tools/derived-adam-parity.ts` (its derived==
authored oracle is gone). `deriveHorizontalPackedAdam` (the pack derivation) survives —
separate capability, live spec consumer.

**Gates (R4, derived is the sole path):** cpu optim+schedule 207/207; `test:gates` 5/5;
`optterm-fold-parity` all programs pass; `parity-fullstack-tl` both arms agree ≤1e-6/30;
`parity-packed-vs-unpacked` adam maxDiff=0; 124M regression PASS (round 0 bit-exact
9.8089, flat memory 4906.5 MB). The opt-out `TORCHLETTE_DERIVED_ADAM=0` no longer exists.

**A100 exit gate (dw-2-1, @512, NUM_STEPS=18, authored vs fork-C-derived, same run):**

| model | submits (auth→derived) | peak MB (auth→derived) | verdict |
|---|---|---|---|
| distilgpt2 | 8 → **8 EXACT** | 5636.6 → **5636.6 EXACT** | PASS |
| gpt2-medium | 19 → **19 EXACT** | 16987.9 → **16189.0** (≤16189.0 gate, −800 MB) | PASS |

Medium's fork-B regression (submits 19→20, +800 MB) is GONE; fork C is 19 submits EXACT
and 800 MB UNDER the current authored default (the reclaim-boundary quantum flips the
other way — fork C lands on the low warmup transient). Standing gates green with the flag
ON: `test:gates` 5/5 (compiled==lowered for the fork-C bc live-scalar re-dress),
`parity-fullstack-tl` both arms ≤6e-6/30 (and fork-C==authored ≤1e-6), `derived-adam-
parity` Δparam ≤2.98e-8 / Δm=Δv bit-exact, `parity-packed-vs-unpacked` 4/4, 124M
regression PASS (round 0 bit-exact 9.8089; rounds 3/6/9 within ~3e-4 of the R2 baseline —
the L3 lemma accumulating over 200 inner steps, flat memory), cpu project clean.
