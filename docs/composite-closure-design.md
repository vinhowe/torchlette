# Composite Closure: the two folds that make the composite backward a theorem and the forward WGSL a derivation

**Status:** v1 DESIGN · 2026-07-23 · design-only, no `src/` change lands with this doc.
**Campaign:** COMPOSITE-CLOSURE — the last real derivation campaign, specified by
`docs/cleanliness-audit-2026-07.md` rung 7 ("the two missing folds") and the semantic
audit that scored the stratum 8/10 with "composite backwards + attention are the honest
hand frontier."
**Succeeds / depends on:** `docs/semantic-derivation-design.md` (the P0–P6 algebra: the
`Expr`/`CompNode` schemas, `deriv`+`normalize`, the reduction monoids, the index-map
transpose facts, `interpretComposition`); `docs/derived-optimizer-realizer-design.md`
(the O1 ruling + the `evalOptTerm` ↔ `lowerOptTermToTileIR` sibling-fold template this
campaign copies twice).
**Baseline at design time:** `srcSLOC=67317`, `envFlags=64` (`bash tools/weight-norm.sh`,
2026-07-23, post-§19 contraction closure).

---

## 0. Declaration (one sentence)

The composite backward — layernorm / rmsnorm / softmax / cross-entropy VJPs — becomes
**one structural reverse-mode pass over the same `CompNode` composition its forward is
already authored from** (fold **F1**), and the forward GPU activation bodies —
sigmoid / silu / gelu / erf / softplus — become **one structural fold of the same `Expr`
their CPU reference and gradient already derive from** (fold **F2**), so the last two
hand-written copies of one meaning stop existing and every gradient in the framework is a
theorem.

---

## 1. Lineage — why this is the *last* derivation campaign, and why it should bend the curve DOWN

The semantic-derivation era (Crystal 3, P0–P6 + §17 contraction + §19) made five op
families derive from one source each: elementwise (P0), reductions (P1), composite
*forwards* (P2), index-space (P4), optimizers (P5/§17), and — as of §19 — the contraction
adjoints. Its closing reckoning (§18.4) was honest: **+1779 srcSLOC over the campaign**,
because every prior phase built a reusable *engine* (interpret + adjoint + normalize;
the reduction monoid; the index/transpose engine; the composition interpreter; the
optimizer-program interpreter). The engines dominated the deletions.

COMPOSITE-CLOSURE is structurally different, and this is its covenant story: **it builds
no new engine.** Both folds *compose engines that already exist.*

- **F1** reuses, unchanged: `deriv` (`adjoint.ts`, the elementwise chain rule), the
  reduction `gradKind` (`reduction.ts`: sum→broadcast, mean→broadcast-scaled), and the
  index-map transpose realizers (`index-map.ts`: `broadcastOverDims`, `reduceToShape`,
  `realizeIndexAdjoint`'s `scatterZeros`). F1's *new* code is only the reverse-mode
  plumbing that threads a cotangent through the six `CompNode` kinds and accumulates it
  per input role — the composite analogue of `emit-rt.ts`'s `makeUnaryGrad`.
- **F2** is a near-exact clone of `lowerOptTermToTileIR` (`schedule/optterm-fold.ts`):
  the same recursion over a closed algebra, lowering to `BlockExpr` instead of runtime
  ops. Its inputs (sigmoid / silu / relu / gelu_tanh / gelu_erf as `Expr`) already exist
  in `catalog.ts` / `composite.ts`.

Because the engines are amortized, the deletions dominate for the first time. The
**forecast is net-negative-or-flat** (§9): F1 deletes ~165 lines of hand VJP against a
~140-line reverse-mode plumbing that reuses three engines; F2 deletes ~80 lines of
twice-written DSL against a ~75-line fold. This is the campaign that finally cashes the
"capability + legibility per SLOC" metric §18.4 named as the real scalar — not by writing
another engine, but by paying off the two seams §18.5 left explicitly chartered:

> *"§18.5 — The S3 WGSL full seam … the full `emit(Expr)→BlockExpr` rewrite for the
> elementwise/composite … WGSL is the remaining S3 work."* — that is **F2**.
> The composite *backwards*, checked "today only by oracle, not by construction"
> (audit rung 7), are **F1**.

This is the **tenth** application of the house move "the latent decision becomes an
object" — but where the ninth (semantic-derivation) made *meaning* an object, the tenth
makes the two derivations that were deferred behind the ULP seam finally *run*.

---

## 2. The measured substrate (read before the folds)

Everything below is verified against the current tree (2026-07-23), not assumed.

### 2.1 What already derives (do not re-derive)

- The composite **forwards** are `CompNode` DATA: `SOFTMAX_DEF`, `LOG_SOFTMAX_DEF`,
  `RMSNORM_DEF`, `LAYERNORM_DEF`, `CROSS_ENTROPY_DEF` (`composite.ts`), realized by
  `interpretComposition` (`emit-rt.ts`) and checked against the hand forwards by
  `test/semantic-composite.spec.ts` (the Probe-4 reference gate).
- The `CompNode` algebra: `in` (role), `kc` (const), `u` (elementwise unary),
  `b` (elementwise binary), `r` (reduce sum/max/mean along `dim`, keepdim), `gi`
  (gather-at-index). Schema-gated DATA (`assertNoCompositionBody`).
- The transpose facts F1 needs are **already built and unit-tested** (P4): reduce⇄broadcast
  (`broadcastOverDims`, `reduceToShape`), gather⇄scatter (`realizeIndexAdjoint`'s
  `scatterZeros`, reading the runtime index tensor from `IndexAdjointCtx`).
- The forward activation **constants** are single-sourced (`erf.ts`: `ERF_A`/`ERF_P`,
  `GELU_SQRT_2_OVER_PI`/`GELU_TANH_C`); both backend sites *import* them (P2). The
  triplication of *values* is dead.

### 2.2 What is still hand-written (the two folds' targets)

**The composite backwards (F1).** Grep confirms `adjoint.ts` differentiates `Expr`
only — there is **no** `CompNode` adjoint anywhere. The composite VJPs live hand-written
in two tiers:

| op | CPU / graph VJP (`decomposed-ops.ts`) | form | fused WGSL backward (`backend/webgpu/`) |
|----|----------------------------------------|------|------------------------------------------|
| softmax | inline closure, **37 SLOC** — `s·(g − Σ(s·g))` | simplified closed form | **none** (this closure is the *only* backward, CPU+GPU) |
| rmsnorm | `rmsnormBackwardImpl`, **52 SLOC** — `inv_rms·(g·w − norm·mean(g·w·norm))`, `dW=Σ(g·norm)` | simplified closed form | `rmsnorm-kernel.ts` gradX 43 / rowStats 24 / partial 40 / reduce 24 |
| layernorm | `layernormBackwardImpl` CPU branch, **76 SLOC** — naive `gradVar/gradMean` expansion | **naive chain-rule** | `layernorm-kernel.ts` gradX 50 (folded `invStd·(gn−c1−norm·c2)`) / rowStats 32 / partial 46 / reduce 30 |
| cross_entropy | dispatch stub, 10 SLOC | thin → fused | `cross-entropy-kernel.ts` ceBwd 48 — `g·(softmax − onehot)` |

Two facts drive F1's design: (a) **layernorm already computes the same VJP two ways** —
a naive expansion on CPU, a folded closed form in WGSL — so "same meaning, two algebras"
is an *existing* reconciliation, not a new risk F1 introduces; (b) the fused WGSL
backwards carry their reductions *in-kernel* (`wgReduce`/`dualWgReduce` row-programs),
which the elementwise fold cannot emit (§7).

**The forward activation bodies (F2).** The structure is written at **two** DSL sites
(plus a third re-inlined string), all over already-single-sourced constants:

| site | representation | activations, SLOC |
|------|----------------|-------------------|
| `tile-ir.ts` `BlockExpr` compound methods | fluent `BlockExpr` chain, reusable methods | `sigmoid` + `erf` only, **~33 SLOC** (clamp/fma/cdiv/floorDiv are op-lowerings, not activations — they stay) |
| `fusion-tile-ir.ts` `applyFusedOp` switch | fluent `BlockExpr` chain, op-name cases | relu, sigmoid, silu, softplus, gelu_tanh, gelu_erf, **~36 SLOC** |
| `schedule/realizers/triton-emit.ts` | Triton *string*, hard-coded `0.7978845608028654`/`0.044715` | gelu_tanh — a fourth copy that does **not** import the constants |

And a fifth: **`softplus` is hand-authored forward *and* backward in
`frontend/torchlette.ts`** (`log(1+exp(x))` fwd, `sigmoid(x)·g` bwd) and is **absent from
the semantic catalog entirely** — an oversight the campaign closes for free (§4.4).

---

## 3. F1 — the CompNode-adjoint pass (design + the simplification ruling)

### 3.1 The pass

F1 is a **reverse-mode VJP over `CompNode`**: `vjpComposition(def, dim) → { role → gradExpr }`.
It threads an upstream cotangent `ḡ` (a runtime tensor) from `def.root` back to each input
role, accumulating contributions where a role is used more than once (softmax's `x` under
both the numerator and the denominator; layernorm's `x` under `mean` and the centering).
The recursion is the reverse of `interpretComposition`, one adjoint rule per kind:

```
in  role      : accumulate ḡ into grad[role]           (leaf)
kc  const     : ⊥ (no input)
u   op,a      : push  deriv_local(op, â)·ḡ  into a     (elementwise local factor,
                                                        REUSES adjoint.ts `deriv`)
b   op,a,b    : product/quotient rule → push per-operand cotangent into a,b, each
                REDUCED back to the operand's own shape via `reduceToShape`
                (the broadcast-transpose — REUSES index-map)
r   op,a      : push the reduction transpose into a:
                  sum  → broadcastOverDims(ḡ)           (unit)          REUSES reduction gradKind
                  mean → broadcastOverDims(ḡ)·(1/N)     (broadcast-scaled)
                  max  → see §3.3 (the one lemma)
gi  a,indexRole: push scatterZeros(ḡ) into a           (gather transpose = scatter/onehot,
                                                        REUSES realizeIndexAdjoint; this is CE)
```

The pass is realized over the `RuntimeEngine` exactly as `interpretComposition` realizes
the forward — same memoized-leaf, lazy-force discipline (`emit-rt.ts`) so a cotangent that
never reaches a role never forces its saved operand. The forward intermediates the
backward needs (softmax's `s`, a norm's `inv_std`/`normalized`) are recomputed from the
saved operands, exactly as the hand closures do today.

**What is genuinely new vs reused:** new = the ~120-line reverse-mode plumbing (cotangent
accumulation + the six push-rules). Reused unchanged = `deriv` (u/b local factors),
`gradKind` (r scaling), `broadcastOverDims`/`reduceToShape`/`scatterZeros` (all three
transposes). F1 writes no transpose, no chain rule, no reduction fact — it *composes*
them. This is the covenant lever.

### 3.2 The simplification ruling — F1's core design question

The hand VJPs are algebraically *simplified* closed forms (softmax's Jacobian folded into
one reduction; rmsnorm/layernorm's `g − norm·mean(g·norm)` folding). A naive reverse-mode
pass emits the *unsimplified-but-collected* chain-rule graph. Which does F1 emit, and does
byte-identity to the hand form hold?

**Ruling: the principled middle — derived + ONE admitted simplification lemma stated as
DATA; the residual is a NAMED fp-reassociation lemma with a measured bound.** Concretely:

1. **F1 emits honest collected reverse-mode**, not a re-transcription of the folded hand
   form. Reverse-mode with per-node cotangent accumulation *already* produces the folded
   structure for softmax and rmsnorm — the "simplification" in the hand code is mostly
   (a) reusing saved `normalized`/`inv_std` and (b) collecting the two reduction-transposes
   into one `− norm·mean(g·norm)` term, both of which collected reverse-mode does by
   construction. (Worked check, softmax: reverse-mode through `div` gives
   `ḡ_u = ḡ/v`, `ḡ_v = −Σ(ḡ·u)/v²`; `ḡ_v` broadcasts back through `sum` to each `u`;
   folding through `exp` and collecting yields `s·(g − Σ(s·g))` — the hand form, up to the
   reduction's summation order.)

2. **The ONE admitted lemma, stated as data: the numerical-stability `max`-shift is a
   DETACHED node.** softmax/log_softmax subtract `max(x)` purely for range safety; its
   gradient is provably zero-sum (softmax is shift-invariant: `Σ ḡ_shifted = 0`). PyTorch
   encodes this by subtracting `max.detach()`. F1 encodes it as a `detach: true`
   annotation on the `r max` node in `SM_SHIFTED` — the composite-frame analogue of the
   eps-guard annotation (§4.5 of the semantic design). With the annotation, the reverse
   pass drops the (provably zero) argmax mask-scatter entirely, so no extra gather/scatter
   dispatch is emitted and the derived graph structurally matches the hand closed form.
   This is the *only* admitted simplification, and it is a stated fact about the math
   (shift-invariance), not a rewrite of convenience. It also aligns F1 with the P1 ruling
   that `gradKind(max) = "none"` (the value path of a max-reduce is non-differentiable; the
   arg carries) — the shift's grad genuinely *is* nothing.

3. **The residual gap is fp-reassociation, gated by a named lemma with a measured bound**
   (the L1/L2/L3 discipline). The derived reverse-mode's reduction ordering (`Σ` of a
   product) need not byte-match the hand form's ordering, and across the CPU↔GPU seam the
   `exp`/`rsqrt` transcendentals diverge (the measured `pow`/`exp` precedent, §4.5). So F1
   claims **not** byte-identity to the hand VJP. It claims: (a) **correctness** — the
   derived VJP == torch to fp32 under oracle gradcheck (this is the *construction* proof
   the audit said was missing); (b) a **stated, measured** reassociation delta vs the hand
   form, ≤ the run-to-run GPU nondeterminism floor (~1e-5/1e-6, the §19/P5 bar), named as a
   lemma — silent reorders are forbidden, the delta is written down with its magnitude.

**The cost fork (measured, not assumed).** Exactly one op could put F1's derived graph on
a training-hot path: **standalone softmax has no fused kernel**, so if C2 replaces its CPU
closure the derived graph *is* the GPU backward. (layernorm/CE/rmsnorm keep their fused
kernels — §7 — so their hot path is untouched.) Therefore C1 must **measure** the derived
softmax backward's node count + a V100 timing vs the 37-line hand closure (probe
`tools/comp-adjoint-cost.ts`). With the detach lemma the counts should be at parity; if
they are, C2 deletes the closure; if the derived graph is materially heavier, C2 keeps the
hand softmax closure and uses the derived form only as the C1 *reference*, deferring that
one deletion. This is the honest measured branch, not a blanket claim.

### 3.3 The `max`-reduce, precisely

`max` appears only as the softmax/log_softmax stability shift, always detached (§3.2 lemma).
F1 therefore never needs a differentiable max-reduce (the mask-scatter VJP). If a future
composite uses `max` as a *load-bearing* (non-detached) reduce, its adjoint is the
argmax-mask-scatter — buildable from the same `realizeIndexAdjoint` machinery — but it is
**out of scope** here (admission-pressure: no current composite needs it; do not build the
mask-scatter until a consumer exists).

---

## 4. F2 — the Expr→BlockExpr fold (design + feasibility evidence)

### 4.1 The fold

`lowerExprToTileIR(e: Expr, ctx: KernelContext, roles: {x: BlockExpr}): BlockExpr` — the
structural sibling of `lowerOptTermToTileIR`, one `Expr` kind → one `BlockExpr` method:

```
x / role  → roles.x                 c(v)   → ctx.f32(v)
neg exp log sqrt tanh sin cos abs sign floor ceil round → BlockExpr.{same}()
recip u   → ctx.f32(1).div(u)       (the one Expr node BlockExpr lacks as a primitive)
add sub mul div min max pow → BlockExpr.{same}(b)
gt ge lt le eq ne → BlockExpr.{cmp}(b)          where(c,a,b) → c.select(a, b)
erf u     → the A-S Horner over ERF_A/ERF_P (reading the ONE source), i.e. what
            `BlockExpr.erf()` / `emitErf` already build — emitted once here
```

Both backend sites route through it: `fusion-tile-ir.ts`'s `applyFusedOp` activation cases
become `lowerExprToTileIR(DEF.expr, ctx, {x: inputs[0]})`; `tile-ir.ts`'s `BlockExpr.sigmoid`
becomes a call to the fold. `isfinite` is not an activation and needs no BlockExpr node.

### 4.2 Feasibility evidence (the existence proof is already in the tree)

The fold is not speculative — its exact shape runs today for the optimizer:

- **`lowerOptTermToTileIR` (`schedule/optterm-fold.ts`, 71 SLOC)** is the identical
  recursion (role/c/u/b → `BlockExpr.{neg,sqrt,sign,abs,exp}`/`{add,sub,mul,div}`), gated
  by the R2/R5 `optterm-fold-parity` differential. F2 adds the transcendentals
  (`log,tanh,sin,cos,sqrt`), comparisons, `select`, and `erf` — all present on `BlockExpr`
  (verified: `UnaryOp`/`BinaryOp`/`CmpOp` unions + `select` + `ConstNode`). The `Expr`
  source algebra and the `BlockExpr` target surface are a near-exact match; the only
  mismatch is `Expr.recip` → `div(1,·)` (mechanical).
- The **same fold to a different codomain** already exists twice for `Expr`:
  `interpret.ts` (`Expr → number`) and `emit-rt.ts`'s `emit` (`Expr → rt.*`), the latter
  including `emitErf` realizing the A-S poly over `rt.*` — structurally what
  `lowerExprToTileIR`'s `erf` case does over `BlockExpr`.
- The activation **definitions already exist as `Expr`**: `catalog.ts` `sigmoid`
  (`recip(add(c(1),exp(neg(x))))`), `silu`, `relu` (`where(gt(x,c(0)),x,c(0))`);
  `composite.ts` `GELU_TANH_DEF`, `GELU_ERF_DEF`.

### 4.3 The parity bar (byte-where-identical, ULP-where-reassociated)

The fold changes DAG shape cosmetically (shared vs duplicated `one`; `t.mul(a5)` vs
`ctx.f32(a5).mul(t)`). Within IEEE, commuting the two operands of one `mul`/`add` is
byte-exact; only *re-nesting* an association changes bits. So the gate is the R2
adam-differential bar verbatim: **folded WGSL == hand WGSL byte-identical where the algebra
is identical, ≤1e-7 where reassociated** (`tools/expr-fold-parity.ts`), plus the standing
`fusion-tile-ir`/`tile-ir-block` GPU suites green in both flag states.

### 4.4 The two reconciliation items + the free deletion

1. **The `gelu_tanh` `clamp(inner, −10, 10)` guard** exists at `fusion-tile-ir.ts` but
   *not* in `GELU_TANH_DEF`. **Ruling: DROP it, consistent with the P2 backward ruling
   (§14) that already dropped the analytic backward's ±10 tanh clamp** as cosmetic
   (`sech²(10) ≈ 8e-9`; `tanh` is a saturating WGSL builtin, no overflow). A1 must
   *probe-verify* forward-value agreement over the training range before the drop; if a
   value diverges, the guard becomes a stated `clampInner` annotation on the def (the §4.5
   vocabulary) rather than buried in the switch — but the P2 precedent says the drop is
   safe and keeps zero guard vocabulary.
2. **`triton-emit.ts`'s re-inlined string** is a separate realizer (a Triton codegen
   surface, not `BlockExpr`); F2's tile-IR fold cannot emit into it. Minimum: import the
   `erf.ts` constants so the *values* stop being a fourth copy (the audit's confirmed
   Triton-constant divergence, cross-check row). Full Triton-from-`Expr` derivation is a
   named non-goal here (a distinct codegen target, admission-pressure).
3. **Free deletion:** add `SOFTPLUS_DEF = log(add(c(1), exp(x)))` to `catalog.ts` with
   `gradPolicy:"derive"`. Its backward auto-derives to `sigmoid(x)` (`d log(1+eˣ) =
   eˣ/(1+eˣ)`) via the *existing* elementwise adjoint — deleting **both** the hand forward
   and the hand backward in `torchlette.ts`, with no F1/F2 mechanism at all. This is the
   elementwise-skeleton's own P0 machinery finishing a job it missed.

---

## 5. The attention (+ rope) exclusion — STATED in the stratum

The semantic audit called attention "the most accidental boundary" and asked that the
exclusion be *stated*, not merely observed. It is stated here, and it is principled.

> **Attention (scaled-dot-product attention) and rope are excluded from the semantic
> derivation stratum. They are §2-category-e composites whose fused kernels compute
> different intermediates than any naive composition — attention's online-softmax running
> `(m, ℓ, o)` rescale, rope's half-split rotate — so a `CompNode`/`Expr` composition would
> be the kernel's *reference*, never its derivation (RT3). Neither has, or gets, a semantic
> composition term. They stay hand-authored, fused-kernel-first, and are fenced
> numerically (below), not by construction.**

This is not laziness: attention *could* be spelled `matmul∘softmax∘matmul`, but its kernel
is a hand-tuned online-softmax algorithm (an admitted schedule-state lemma), so claiming
the composition *is* the kernel would be the RT3 lie. The exclusion is the same one the
schedule-state stratum already draws around attention.

**Ruling on promoting the partial naive reference.** `attention-skeleton.ts` (~707–814)
holds a naive attention *backward composition* (`naiveAttentionBackwardComposition`, ~117
SLOC) — but it is a **schedule descriptor** checked by a **WGSL byte-differential**
(`attention-differential.spec.ts`, `expect(derived).toBe(live)`), not a numeric oracle. A
separate numeric fence already exists: `test/webgpu/attention-kernel.spec.ts` compares the
fused GPU kernel (fwd + dQ/dK/dV) against the CPU-decomposed SDPA. **Recommendation: do NOT
promote the schedule composition to a numeric oracle** — that would require lowering its
regions to real dispatches (a numeric harness, real cost) for no derivation payoff, since
attention is excluded by design. **The cheap, proportionate fence upgrade is a torch-oracle
SDPA op** (forward + dQ/dK/dV backward) added to `test/oracle/torch-oracle.ts`, asserting
the fused GPU kernel == torch to fp32 tolerance. That upgrades the existing
fused-vs-CPU-decomposed differential to a *torch* referee at the cost of one oracle op,
without folding attention into the algebra. State it as an **optional** hardening (one
gate row), not a campaign requirement — the exclusion stands with or without it.

`rope` is excluded on the same grounds (a standalone fused kernel, backward = the same
kernel with `sin_scale=−1`; no composition term, none owed).

---

## 6. The `pow` variable-exponent grad — confirmed principled, LEFT

Confirmed and left untouched: `catalog.ts` declares `pow` with `gradPolicy:"hand"` (a
first-class enum value), `adjoint.ts` explicitly refuses the variable-exponent case
(constant-exponent → `k·pow(a,k−1)`; variable → `c(NaN)` sentinel), and the registry loop
skips overwriting its `ttGrad`/`tsGrad`. The integer-exponent path is separately principled
(lowered to a mul chain via exponentiation-by-squaring in `torchlette.ts`, dodging the WGSL
`pow(x<0)`=NaN class; grad falls out of `mul` autograd). This is a *declared* exclusion,
not an ad-hoc hand grad — no derivation is owed. The exit census (§10) whitelists it.

---

## 7. The fused backward kernels stay the FAST path, ASSERTED against the derived reference

The task asks whether the fused hand-WGSL backward kernels (rmsnorm / layernorm / CE) fold
via F2 or stay hand. **Ruling: they STAY, asserted against F1's derived reference — the
exact fused-vs-elementwise pattern the fused adamStep used (P5) before its own body
derived.** Rationale, structural:

- The fused backward kernels carry their reductions **in-kernel** (`wgReduce`/`dualWgReduce`
  producing `mean(g·w·norm)`, `logsumexp`, row-max). F2 is an **elementwise** fold — it has
  no target for an in-kernel reduction, exactly as `lowerOptTermToTileIR` structurally
  refuses the `mm` contraction node (no elementwise tile-IR target). So the fused backward
  kernel is a **schedule-state artifact** (a row-program + the `wgReduce` admitted lemma),
  not an F2 output. Trying to fold it is the same category error as folding Muon's
  Newton–Schulz through the elementwise optterm-fold.
- Therefore the relationship is **assertion, not derivation**: a standing
  **fused-vs-derived differential** (`test/optim/fused-vs-elementwise.spec.ts`'s composite
  analogue) proves each fused backward kernel == F1's `vjpComposition` realized over `rt.*`,
  within the §3.2 measured ULP tolerance. This is the backward twin of P5's "fused adamStep
  asserted against the program" and closes RT3 for the composite *backward*: the fused
  kernel is checked against the composition, not re-owned.

**The deeper prize, named and DEFERRED (not required for closure).** The fused kernel's
per-element *epilogue* arithmetic (the `invStd·(gn − c1 − norm·c2)` combine that surrounds
the reductions) *could* one day derive via an F2-style fold, with the `wgReduce` producing
`c1`/`c2` left as the schedule-state's admitted row-program lemma — exactly how
`adam-skeleton.ts` derives the adamStep *body* via `lowerOptTermToTileIR` while
hand-authoring the DMA/grid wrapper. That is the composite analogue of the deferred
adamStep-WGSL derivation (§16.5 of the semantic design) and is chartered-but-deferred here
(admission-pressure: closure is achieved by assertion; the epilogue-fold cashes no deletion
until the schedule-state reduction-lemma seam is factored out, a separate campaign).

---

## 8. Phase plan (each independently shippable, differential-gated)

Two tracks (F1, F2) that share no code and can land in either order or in parallel. Every
phase carries the **standing gate wall** unless noted: `npm run build`; cpu suite
(semantic-derivation / -composite / -reduction / -index / -optimizer, gradcheck, the oracle
autograd/op-conformance suites via `TORCH_ORACLE_PYTHON`); `npm run test:gates`
(compiled-plan-parity ×5); `tools/parity-fullstack-tl.ts` twice (`TORCHLETTE_COMPILED_PLAN=0`
vs default, ≤1e-5/30 steps); the **124M regression** ({0:9.8089 bit-exact, 3/6/9 within
3e-4}); strict-lifetime default (throwing). GPU paths touched → the relevant fused-vs-derived
differential.

### Track F1 — the composite backward

- **C1 — the CompNode adjoint pass, dark.** Land `vjpComposition` + the `detach` annotation
  (§3.2) + the cost probe (§3.2). Not wired to any live backward. **New gate**
  `test/oracle/semantic-composite-backward.spec.ts`: `vjpComposition` realized over `rt`
  == torch gradcheck for softmax / log_softmax / rmsnorm / layernorm / CE (fwd+bwd, f32 and
  autocast-f16, 2-D and batched) — the *construction* proof the audit said was missing; AND
  == the hand VJP within the §3.2 named-reassociation bound. `tools/comp-adjoint-cost.ts`
  reports softmax derived-vs-hand node counts + V100 timing. **Nets +engine-plumbing,
  deletes nothing.**
- **C2 — route the CPU / non-fused composite backwards through the derived reference**,
  behind `TORCHLETTE_DERIVED_COMPOSITE_BWD` (born with sunset, §9). Replace the softmax
  closure, `rmsnormBackwardImpl`, and the `layernormBackwardImpl` **CPU branch** with
  `vjpComposition`. The softmax deletion is **gated on the C1 cost probe** (§3.2): delete if
  node-count parity, else keep + defer. **Gate:** CPU trajectory parity + the C1 oracle both
  flag states. **Deletes ~165 SLOC of CPU hand VJP** (softmax 37 + rmsnorm 52 + layernorm-CPU
  76), minus any deferred by the cost fork.
- **C3 — assert the fused GPU backward kernels against the derived reference** (§7). Land the
  composite fused-vs-derived differential as a **standing gate**; the fused rmsnorm/layernorm/CE
  backward kernels stay byte-untouched (124M hot path unchanged). Flip
  `TORCHLETTE_DERIVED_COMPOSITE_BWD` default ON, retire the flag. **Exit of F1:** every
  composite backward is either derived (CPU) or fused-asserted-against-derived (GPU); no hand
  composite VJP survives outside the fused kernels, which are now theorems-checked.

### Track F2 — the forward activation WGSL

- **A1 — the `Expr→BlockExpr` fold, dark.** Land `lowerExprToTileIR` + add `SOFTPLUS_DEF`
  to the catalog (§4.4). Not wired. **New gate** `tools/expr-fold-parity.ts`: folded
  activation == hand `BlockExpr`/`fusion-tile-ir` body, byte-where-identical / ≤1e-7
  where-reassociated, over {sigmoid, silu, relu, softplus, gelu_tanh, gelu_erf, erf}. The
  free softplus deletion (hand fwd+bwd in `torchlette.ts` → derived) lands here — it needs
  only the P0 elementwise machinery. Probe the gelu_tanh clamp-drop (§4.4).
- **A2 — route both backend sites through the fold**, behind `TORCHLETTE_DERIVED_ACTIVATION`
  (born with sunset). `fusion-tile-ir.ts` activation cases + `tile-ir.ts` `BlockExpr.sigmoid`/
  `.erf` call `lowerExprToTileIR`; hand bodies stay as the differential oracle (flag off).
  **Gate:** `fusion-tile-ir` (61) + `tile-ir-block` (32) GPU suites green both flag states;
  parity-fullstack + 124M exercise gelu forward end-to-end.
- **A3 — the deletion.** Remove the `fusion-tile-ir.ts` activation switch bodies + the
  `tile-ir.ts` sigmoid/erf compound methods + import the constants into `triton-emit.ts`;
  flip `TORCHLETTE_DERIVED_ACTIVATION` default ON, retire the flag. **Deletes ~80 SLOC** of
  twice/thrice-written DSL. The `expr-fold-parity` differential is the surviving guard.

---

## 9. Deletion ledger + the covenant

Sizes are code-only SLOC of the current tree; the derivable fraction is what deletes.

| track | deletes | SLOC | adds | SLOC |
|-------|---------|------|------|------|
| F1 | softmax closure + `rmsnormBackwardImpl` + `layernormBackwardImpl` CPU branch (`decomposed-ops.ts`) | **~165** | `vjpComposition` reverse-mode plumbing + `detach` annotation + oracle op (reuses `deriv`/`gradKind`/index-map — **no new engine**) | ~140 |
| F2 | `fusion-tile-ir` activation switch (~36) + `tile-ir` sigmoid/erf compounds (~33) + `torchlette.ts` hand softplus fwd+bwd (~12) | **~81** | `lowerExprToTileIR` fold + `SOFTPLUS_DEF` (clone of `optterm-fold`) | ~75 |
| | **campaign total** | **~246** | | **~215** |

**Covenant: net-negative-or-flat by construction (forecast ≈ −30 srcSLOC).** This is the
first derivation phase since P0 that should cash net-negative, and the reason is the §1
lever: it *composes existing engines* instead of building one. The net-new mechanism (two
thin folds) is warranted because each is the single seam serving what four hand sites
served — but unlike P0/P4/P5, the seam is a *reuse*, not an engine, so the deletions win.
Per house policy the campaign-end commit names deletions vs additions and re-measures
`bash tools/weight-norm.sh --log`; growth without deletion triggers re-review, not a waiver.
The fused GPU backward kernels are **not** counted as deletions — they stay, now asserted
(§7). Two flags are born with sunsets (`TORCHLETTE_DERIVED_COMPOSITE_BWD`,
`TORCHLETTE_DERIVED_ACTIVATION`); both die at C3/A3 per the env-flag-ledger convention
(soak → default → opt-out dies), mirroring `TORCHLETTE_DERIVED_ADAM`. Register both in
`docs/env-flag-ledger.md` at birth.

---

## 10. The exit gate — the greppable census

Closure is a *mechanical* claim, not a vibe: **zero hand-written arithmetic in the semantic
frame outside the STATED exclusions.** After C3 + A3, a census script (`tools/semantic-census.sh`,
a weight-norm-style hook) asserts:

- `decomposed-ops.ts` has no composite backward *arithmetic* — every composite backward is
  a call to `vjpComposition` (CPU) or a dispatch to a fused kernel that a standing
  fused-vs-derived differential pins (GPU).
- `tile-ir.ts` / `fusion-tile-ir.ts` have no activation *body* — every activation is
  `lowerExprToTileIR(DEF.expr, …)`.
- The only hand adjoints/bodies remaining in the semantic frame are the **whitelisted
  exclusions**: attention (SDPA), rope, and pow-variable-exponent — each named in this doc,
  each with its own fence (attention/rope: the schedule byte-differential + the numeric
  fused-vs-CPU-decomposed differential, optionally the torch-SDPA oracle of §5; pow: the
  declared `gradPolicy:"hand"` + the NaN-signal refusal).
- The contraction adjoints derive (§19), the elementwise/index/reduction/optimizer adjoints
  derive (P0/P1/P4/P5) — so with F1, **every gradient in the framework is a theorem or a
  named exclusion.** The census is the standing proof, run in CI alongside the weight-norm
  hook.

---

## 11. Risks (honest) and red-team

1. **The broadcast-adjoint trap, at composite altitude (§8.1 of the semantic design).**
   layernorm/rmsnorm broadcast `mean`/`inv_std` (`[.,1]`) against `x` (`[.,D]`); the binary
   adjoint must reduce each cotangent back to its operand's shape (`reduceToShape`) or emit
   wrong-shaped grads *silently*. **Mitigation:** this is exactly what P4's index algebra
   owns and F1 reuses; the C1 oracle gradcheck catches a shape-transpose error immediately
   (a wrong reduction fails torch parity, not just shape). Do not hand-roll the broadcast
   reduction in F1 — call `reduceToShape`.
2. **The softmax hot-path cost (the one training-path deletion).** §3.2. **Mitigation:** the
   C1 cost probe *gates* the C2 softmax deletion; if the derived graph is heavier, the hand
   closure stays and the derived form is reference-only. Measured, not assumed.
3. **fp-reassociation mis-bounded.** The named reassociation lemma (§3.2) could be set too
   tight (flakes) or too loose (hides a real bug). **Mitigation:** the bound is *measured*
   from the C1 derived-vs-hand + fused-vs-derived probes (the §19 precedent: state the
   number, ≤ the run-to-run GPU floor), not guessed; the oracle gradcheck is the independent
   correctness anchor so the trajectory bound only fences *reassociation*, not correctness.
4. **The gelu_tanh clamp drop.** §4.4. **Mitigation:** A1 probe-verifies forward agreement
   before dropping; the P2 backward already dropped the analytic clamp with the same
   `sech²(10)≈8e-9` reasoning, so the precedent is measured, not new.

**RT1 — "F1 is just the autograd you already have; the composite forward + existing autodiff
already gives these grads."** *No.* Today the composite backwards are *hand-written closed
forms* (`decomposed-ops.ts`) checked only by oracle — nothing derives them from the
`CompNode` forward, and the layernorm CPU/GPU pair proves two hand copies of one VJP can and
do drift in *structure*. F1 is the missing derivation: one reverse pass over the *same*
`CompNode` DATA the forward uses, so the backward *cannot* silently disagree with the
forward's meaning. The deletion (165 lines → one pass) is real.

**RT2 — "The fused kernels stay hand, so you haven't closed anything; the real arithmetic is
still in WGSL."** *The fused kernels are now theorems-checked, which is the closure that
matters.* Before C3 the fused backward is an *unchecked* second copy; after, it is asserted
== the derived composition reference (the P5 fused-adamStep pattern). Full derivation of the
fused *epilogue* is named and deferred (§7) because it cashes no deletion until the
reduction-lemma seam is factored — attempting it now is the mechanism-without-a-consumer the
admission-pressure rule forbids.

**RT3 — "Attention is arbitrary to exclude; softmax derives, so attention (which is
softmax-shaped) should too."** *Attention's kernel is an online-softmax algorithm computing
different intermediates than any composition (§5) — the composition would be its reference,
not its derivation, and claiming otherwise is the RT3 lie the semantic design already ruled
on.* The exclusion is stated, principled, and fenced numerically; the standalone `softmax`
op derives because it *is* a straight composition, the attention-internal softmax does not
because its kernel is fused into the online-softmax lemma.

---

## 12. Taste-calls (flagged for Vin — genuinely unresolvable from code/measurement)

- **T1 — the C2 softmax deletion.** Whether to delete the CPU softmax closure or keep it
  reference-only turns on the C1 cost-probe number, which does not exist until C1 lands. The
  *design* is decided (gate the deletion on measured node-count parity); the *outcome* is a
  measurement, not a taste-call — flagged only so the phase can stop cleanly either way.
- **T2 — the optional torch-SDPA oracle (§5).** Adding a torch-refereed SDPA gradcheck is a
  one-gate hardening of the attention fence, orthogonal to closure. Recommend landing it (it
  is cheap and upgrades a byte-differential to a numeric referee), but it is genuinely
  optional — Vin's call whether attention's existing fused-vs-CPU-decomposed differential is
  sufficient.

---

## 13. One-sentence test

If a composite's gradient cannot be stated as the reverse pass over the *same* `CompNode`
its forward is authored from (F1), and each forward activation body cannot be stated as the
fold of the *same* `Expr` its CPU reference derives from (F2) — such that no hand copy of the
backward or the WGSL survives outside the three named exclusions (attention, rope,
pow-variable) — then the meaning is still written twice and the campaign has not closed.

---

## 14. Post-landing verification (2026-07-23, DIVERGENCE-FIX pass)

The campaign was flagged at orchestrator verification with a reported fullstack
divergence (`tools/parity-fullstack-tl.ts` exploding 10.83 → 22176 by step 29,
"reproduced twice on V100") plus one reported CPU-project failure. A dedicated
bisect/root-cause pass **could not reproduce either** and found the campaign
numerically sound. Findings, all numbers read from the losses JSON directly:

- **Divergence NOT reproducible.** `parity-fullstack-tl.ts` was run 6+ times across
  three physical V100s, both arms (compiled default + `TORCHLETTE_COMPILED_PLAN=0`),
  on BOTH the cherry-picked target tree (legibility base + C1–A3) AND the original
  campaign branch worktree. Every run: `losses[0]=10.8282`, `losses[14]=9.3553`,
  `losses[29]≈7.9023` — healthy, and bit-identical (to fp noise) to the pre-campaign
  baseline. The reported 22176 explosion matches the documented device-taint /
  dropped-submit signature that `parity-sanity.ts`'s loss[0]-only check misses
  mid-trajectory (the "device-2 lesson"); it is not produced by any C1–A3 code path.
- **Why C2/C3 cannot move this trajectory.** C2/C3 change ONLY the *CPU-device*
  rmsnorm/layernorm backward; the WebGPU fused kernels are byte-untouched (asserted by
  `composite-fused-vs-derived.spec.ts`, 3/3). The fullstack parity trainer runs on
  WebGPU, so C2/C3 are inert to it. A2/A3 DO change the GPU forward-activation WGSL via
  the Expr fold, yet the trajectory stays bit-identical — the fold is correct for
  gelu/gelu_tanh/gelu_erf at training scale.
- **All gates green on the target tree:** cpu project 1590 passed / 0 failed
  (oracle exported — the reported CPU failure did not reproduce); `test:gates` 5/5;
  `composite-fused-vs-derived` 3/3; `semantic-census.sh` GREEN; 124M regression
  round0=9.8089 (baseline 9.81, bit-exact), 3/6/9=5.9217/5.1542/4.6406 (all within the
  V100 band), memory growth 0.0 MB; strict `[lifetime]` default, zero throws.
- **Landed vs reverted:** NOTHING reverted — F1 (C1–C3) and F2 (A1–A3) are sound and
  stay. The hand VJPs / DSL bodies remain deleted; the census whitelist is unchanged.
- **The one real gap the report correctly named — closed.** The in-suite gate #1
  (`compiled-plan-parity.spec.ts`) was a pure *differential* (compiled == lowered) and
  was blind to a fault that moves BOTH arms identically — exactly a smooth loss
  explosion or a dropped-submit collapse. Added an **absolute-sanity cell**: every step
  of BOTH arms must land in the `ln(V)`-derived band (`initialLossBand(256)` ≈
  [3.05, 7.55]); a >1e4 explosion or ~0 collapse now FAILS THE BUILD. This is the
  "differential must also assert sane absolute values" cell the report asked for,
  gating the class in-suite forever.
