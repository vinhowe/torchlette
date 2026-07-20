# Semantic Derivation: op meaning as the single source

**Status:** v1 design + feasibility probes · 2026-07-20 (Crystal Campaign 3, ratified
by Vin — "derive the reference"). **Design-only; no mechanism lands.** The probes in
§3 are self-contained (`tools/semantic-derivation-probe.ts`, host; and
`tools/semantic-derivation-gpu-probe.ts`, WebGPU) and their numbers are committed here.
**Depends on / succeeds:** docs/schedule-state-design.md (the intra-kernel semantic
stratum — the *how it runs* object), docs/execution-declaration-design.md (the
command-stream stratum + the `assertNoGeneratorLeaf` schema gate this campaign copies),
docs/architecture-debt.md (the sin taxonomy; the "single source at seams" rule).
**Sibling fold-in:** Campaign 2 (functionalization / optimizers-as-programs) enters as
Phase 5 — the composition payoff, not a separate era.

## 0. Declaration (one sentence)

An op's **semantics** — the formula it computes — is a first-class data object (an
expression term over a small primitive algebra), the single source from which its CPU
reference body, its gradient, its WGSL, and its execution declaration are all *derived*,
so the four hand-written copies of one meaning stop existing.

## 1. Lineage and why

This is the **eighth** application of the house move "the latent decision becomes an
object," and the first to reach *above* execution into meaning:

| # | campaign | the latent thing made an object |
|---|----------|--------------------------------|
| 1 | scalars-as-data | payload scalars → graph tensors |
| 2 | planner-derived buffers | buffer assignment → liveness derivation |
| 3 | compile-from-IR | compiled replay → built from IR |
| 4 | variant registry (#61) | kernel variant selection → registry data |
| 5 | islands (I0–I2) | dispatch partition → a partition object |
| 6 | ownership-derivation (#70) | liveness classification → derived-at-inquiry |
| 7 | schedule-state | the loop nest / staging → a three-tier schedule object |
| 8 | execution-declaration | the command stream → a declaration object |
| **9** | **semantic-derivation (this)** | **the op formula → a definition term** |

Schedule-state made *how a kernel runs* data; execution-declaration made *what commands
it issues* data. Both stop one altitude below meaning: they still assume a hand-written
op body somewhere upstream telling them what `sigmoid` *is*. That fact — the formula —
today lives **nowhere**. It is re-encoded, by hand, into four surfaces that must agree
and are checked only by tests that run one path at a time:

- **`erf` is written three times, character for character.** The Abramowitz–Stegun
  Horner polynomial (`0.254829592·t + −0.284496736·t² + …`) appears in
  `src/backend/cpu/numeric.ts` `erf()`, in `src/backend/webgpu/tile-ir.ts` `BlockExpr.erf()`,
  and again inside `src/frontend/custom-backward.ts` `geluErfBackward`. Three owners, one
  polynomial, no source.
- **The GELU-tanh constants** (`0.7978845608`, `0.044715`, `0.134145`) live in the CPU
  `gelu()`, the WGSL activation switch, and `geluTanhBackward` — independently.
- **`sigmoid`** is `1/(1+e⁻ˣ)` in `UNARY_OPS`, `sig·(1-sig)·g` in the registry grad, and
  `one.div(one.add(this.neg().exp()))` as a `BlockExpr` — three spellings of one function
  and its derivative.

This is precisely the disease `architecture-debt.md` names: **two sides that must agree
on a value, each recomputing it independently.** When they silently diverge you get
correct-looking-but-wrong training (the worst failure mode). The cure is the same one the
seven prior campaigns applied: derive the consequences from one source, and assert the
old owners agree at the seam until they are deleted.

**The condition that makes this better rather than worse** (the schedule-state R22
lesson): it must EARN DELETIONS. "The interpreter reproduces the CPU body byte-for-byte"
is necessary but not sufficient — an opaque re-play of the old code passes that test while
owning nothing. Acceptance is the null differential PLUS a schema gate (the definition
term is DATA — no embedded JS body, proven by an `assertNoGeneratorLeaf` analogue) PLUS the
deletion ledger (§5).

## 2. The inventory and the compression ratio

Enumerated exhaustively across the four hand-written surfaces
(`src/backend/cpu/numeric.ts`; `src/ops/registry.ts` + `src/frontend/custom-backward.ts`;
`src/backend/webgpu/ops/*` + `fusion-tile-ir.ts`; `src/executor/execution-declaration.ts`).

**82 distinct ops.** Classified by what a semantic definition needs:

| category | count | how the definition is stated | derives to |
|----------|-------|------------------------------|------------|
| (a) pure elementwise formula | 43 | an `Expr` term over scalar primitives | CPU body, grad (adjoint), WGSL, exec-decl |
| (b) reduction monoid | 6 | an associative combiner (`sum`/`max`/`min`) + optional epilogue | CPU body, grad, kernel, exec-decl |
| (c) index-space map | 14 | a stride/offset index function (views, gather, scatter, cat) | CPU body, grad (transpose of the map), kernel |
| (d) contraction | 3 | the matmul contraction (conv=im2col∘matmul, linear=matmul+bias) | CPU body, grad (matmul adjoint), kernel |
| (e) composite | 7 | a **composition of (a)–(d)** (softmax, CE, layernorm, rmsnorm, attention, rope, log_softmax) | its reference IS the composition; its *fused kernel* is schedule-derived |
| (f) stateful / refusal | 9 | a declared effect (RNG draw, fill/iota, in-place optimizer state) | **refuses** derivation; declared as a typed effect |

**The compression, two honest framings:**

- **Irreducible forward definitions: ~40** (82 → ~40, **≈2×**). Inside the elementwise
  family only ~18 scalar primitives are irreducible (`add, mul, neg, div/recip, exp, log,
  sin, cos, tanh, pow, abs, sign, floor, ceil, round, min, max, mod`); everything else
  derives — `sub = add∘neg`, `sigmoid/silu/softplus/gelu` from `exp`/`tanh`,
  `relu = max(·,0)`, `rsqrt = recip∘sqrt`, `clamp = min∘max`. Plus 3 monoids, ~3
  index/movement primitives, 1 contraction, 2 source primitives.
- **Semantic engines: ~6–8** (82 → ~7, **≈10×**): one elementwise-formula engine, one
  reduction-monoid engine, one index-remap engine, two data-movement kernels (gather,
  scatterAdd), one contraction engine, two source engines.

**The strongest compression is on the backward surface.** All 7 composites (category e)
and every elementwise activation decompose into the primitive set, adding **zero** new
forward primitives. And backward adds **essentially zero primitive semantics**: the
adjoint of every elementwise op is produced by one chain-rule pass over its definition
(§3.2); the only genuinely-hand-written backwards are the **4** contraction/composite
custom rules (`matmulBackward`, `linearBackward`, `geluTanhBackward`, `geluErfBackward`) —
and two of those four (the GELU pair) are compositional and derive too, leaving **2**
irreducible adjoint rules (the matmul VJP and its `linear` specialization). The gradient
tables are the single largest derivable surface.

## 3. The probe verdicts (numbers committed)

Two probe harnesses, both reproducible. The CPU op bodies compute in f64 and round once on
store into a `Float32Array` (`numeric.ts` discipline); both probes match that — evaluate in
f64, compare at the f32 boundary. "byte-exact" = identical f32 bit patterns; ulp/abs gaps
are reported so an algebraic-rewrite difference is not misread as a bug.

### Probe 3 — Reference: definition-interpreted CPU ref vs hand CPU body
`tools/semantic-derivation-probe.ts`. Each elementwise op is written ONCE as an `Expr`
(exactly the `UNARY_OPS`/`BINARY_OPS` formulas) and interpreted; compared to the
hand-written body over 14 unary / 35 binary sample points.

**Verdict: 19/19 ops reproduce the CPU body byte-for-byte** (maxUlp = 0 across all).
The definition IS the CPU reference — no divergence. This surface is a pure win: the
hand-written `UNARY_OPS`/`BINARY_OPS` scalar bodies are a redundant copy of the
definitions and can be *generated*.

### Probe 2 — Adjoint: derived VJP (adjoint of the definition) vs the registry grad table
Same file. Gradients are DERIVED mechanically from the definition by structural adjoint
rules (§3.2), then compared numerically to the hand-written table gradients (transcribed
from `registry.ts`).

**Verdict: 19/22 derived VJPs byte-match the table; +1 more within 4 ulp.** The **3
named divergences are the payload** — each is a real fact about the table, not a probe
artifact:

| op | gap | root cause | ruling |
|----|-----|-----------|--------|
| `sqrt` grad | 11 ulp | table is `g·0.5/(√x + 1e-8)`; adjoint is `g·0.5/√x` | the `+1e-8` is a **numerical guard, not the semantics** — it biases the gradient. Belongs in a separate numerical-guard annotation (§4.5), not the formula. |
| `log` grad | 870 ulp @ x≈1e-4 | table is `g/(x + 1e-8)`; adjoint is `g/x` | same epsilon-guard class; the large ulp gap is exactly *why* it matters — the guard visibly changes the gradient near 0. |
| `div` dB | sign-of-zero (9/105 pts) | adjoint emits unsimplified `(0·y − x·1)/y²`; table writes `−x/y²`. `0 − x` vs `−x` differ at x=0 (`−0` vs `+0`). | needs a **canonical simplification pass** on derived terms (`0·u→0`, `u·1→u`, `0−u→neg u`). Then byte-identical. |

Two of the three are the same finding: **the hand tables silently carry numerical
epsilons that are policy, not meaning.** The design must decide where those live (§4.5).
The third says derived adjoints need a small algebraic normalizer to reach byte-identity —
otherwise they agree to ~1 ulp.

### Probe 4 — Composite: RMSNorm as a composition of primitives
Same file. RMSNorm defined as `x · rsqrt(mean(x²)+ε) · w` — a composition of the mean
monoid, `mul`, `add`, `rsqrt` — evaluated and compared to an independent direct
implementation of the semantics over 200 random shapes.

**Verdict: 6300/6300 elements byte-exact** (host). The composition reproduces the
semantics exactly. The only degree of freedom is `rsqrt(mean+ε)` vs `1/sqrt(mean+ε)` — the
reciprocal-ordering choice — which is *precisely* the CPU-vs-fused-kernel divergence the
GPU leg probes.

### The GPU leg — definition interpreter vs the real WebGPU kernels
`tools/semantic-derivation-gpu-probe.ts` (WebGPU, A100). Confirms the host
definition-interpreter against what the actual kernels compute, forward and backward, plus
`api.rmsnorm` vs the composition.

**Verdict: agreement is fp32-rounding-bounded, NOT byte-identical, across the CPU↔GPU
seam.** Forward defs match the GPU kernels to ~1 ulp / maxAbs ≈ 1e-7 for most ops — but
**`exp` diverges to maxAbs = 6.1e-5** (WGSL `exp` vs `Math.exp` over |x|≤5). Backward
grads match both the adjoint and the table to fp32; `log`/`sqrt` show the epsilon-guard's
fingerprint sub-ulp (152 vs 159 byte-exact for log-vs-adjoint vs log-vs-table).
`api.rmsnorm` vs the composition: 310/512 byte-exact, maxAbs 1.19e-7.

**The load-bearing conclusion:** byte-identity is the right acceptance bar at **formula
altitude** (host: 19/19 reference, 19/22 adjoint) and for the composite reference. It is
**the wrong bar at the CPU↔GPU seam** — WGSL transcendentals (`exp` the worst-measured
instance, the direct heir of the `pow`-negative-base precedent) make it unreachable. The
GPU-derivation gate must be a **declared ULP tolerance**, not byte-equality. This is not a
weakness of the thesis; it is the honest boundary the design encodes (§4.5, §8).

## 4. The architecture

### 4.1 The definition term (the schema)

A `SemanticDefinition` is a term in a closed primitive algebra — the probe's `Expr`,
promoted to first-class:

```
Expr =
  | Input(role)                       // x | y | z (operand); g (upstream grad, backward only)
  | Const(f64)
  | Prim1(op, Expr)                    // neg exp log sqrt sin cos tanh abs sign recip …
  | Prim2(op, Expr, Expr)             // add sub mul div pow min max
  | Compare(op, Expr, Expr)           // gt ge lt le eq ne  → {0,1}
  | Select(Expr, Expr, Expr)          // where(cond, a, b)
```

A `SemanticDefinition` also carries its **kind** (the §2 category) and a small typed
frame that the higher categories need:

- **elementwise**: an `Expr` over operand roles (the above).
- **reduction**: a monoid (`sum|max|min`) + identity + optional epilogue `Expr`
  (`mean = sum ÷ count`).
- **index-space**: an index function `outIndex → inIndex` over stride/offset metadata
  (its adjoint is the *transpose* of the map — scatter is gather's adjoint and vice versa).
- **contraction**: the matmul spec (which axes contract); its adjoint is the two-matmul VJP.
- **composite**: a `Composition` — a small dataflow of the above, referenced BY the
  fused-kernel's schedule-state, never re-owned (§4.4).
- **effect** (category f): a typed refusal — see §4.6.

**The schema gate (structural, copied from execution-declaration).** A definition is DATA:
no leaf is a JS function, a WGSL string, or a buffer. An `assertNoDefinitionBody` walker
proves it, exactly as `assertNoGeneratorLeaf` does for command streams. This is what makes
"one source" structural rather than hopeful — an adapter that hides the old `sigmoid`
lambda behind an opaque leaf is **unconstructible**, so the byte differential cannot be
gamed by a wrapped generator (the R22 defense).

### 4.2 Adjoint rules as data — the derivation, not a table

Backward is not authored per-op; it is one structural pass over the forward term. The rules
(the probe's `deriv`) are the standard chain-rule cases, one per primitive:

```
d(add a b)/dv = add(d a, d b)      d(mul a b)/dv = a·db + b·da
d(exp u)/dv   = exp(u)·du          d(log u)/dv   = du/u
d(recip u)/dv = -du/u²             d(tanh u)/dv  = (1-tanh²u)·du
d(min a b)/dv = where(a≤b, da, db) d(where c a b)/dv = where(c, da, db)   …
```

The op's VJP is `g · d(def)/d(operand)`, then run through the **canonical normalizer**
(§3 Probe 2 finding: `0·u→0`, `u·1→u`, `0−u→neg u`, constant-fold) to reach the table's
byte form. Composite adjoints fall out for free — differentiating the *composition* term
gives the same graph the hand backward builds (the GELU pair confirm this: their custom
backwards are just the adjoint of the GELU composition).

**The two irreducible adjoints that do NOT derive this way:** the matmul VJP (`dA = dC·Bᵀ`,
`dB = Aᵀ·dC`) and its `linear` fusion. These are index-transpose facts about contraction,
admitted as the algebra's 2 hand-proven backward rules (the schedule-state "admitted lemma"
analogue), each with its own differential.

### 4.3 The derivation pipeline (per surface)

One definition, four derived consequences — each replacing a hand-written owner:

```
                       ┌── interpret(Expr)                → CPU reference body   (S1)
                       ├── adjoint(Expr)·g → normalize    → gradient graph       (S2)
  SemanticDefinition ──┼── emit(Expr) → BlockExpr chain   → WGSL / tile-IR       (S3)
                       └── family(kind) + arity           → ExecutionDeclaration (S4)
```

- **S1 CPU body ← `interpret`.** Probe 3: exact. The `UNARY_OPS`/`BINARY_OPS` bodies become
  a generated table (or the interpreter itself is the reference — no generation step).
- **S2 gradient ← `adjoint`.** Probe 2: exact after normalization, modulo the epsilon
  guards (§4.5). The registry `grad`/`ttGrad`/`tsGrad` fields become derived.
- **S3 WGSL ← `emit`.** The `BlockExpr` layer is *already* a compositional emitter
  (`sigmoid()`, `erf()`, `clamp()`, `fma()` are "Compound — no new IR node"); `emit` walks
  the same `Expr` into that chain. The hand `BlockExpr.sigmoid/erf` methods become derived.
  **Byte-identity holds at this altitude** (same primitives); the CPU↔GPU seam is ULP-bounded
  (§4.5).
- **S4 execution-declaration ← `family`.** Already data; the definition supplies the
  per-op classification (`ELEMENTWISE_UNARY_OPS` membership, arity) that
  `execution-declaration.ts` lists by hand.

### 4.4 Composites: reference by composition, kernel by schedule (no re-owning)

The crux question — *can softmax/layernorm/RMSNorm/CE/attention be definitions-as-
compositions whose fused kernels stay schedule-state-derived?* — answers **yes**, and the
two strata already meet cleanly:

- The composite's **semantics** is its `Composition` term (Probe 4: the RMSNorm composition
  reproduces the reference byte-exact on host). This is the single source for its CPU
  reference and its gradient (differentiate the composition).
- The composite's **fused kernel** is NOT re-derived here — schedule-state already owns it.
  Per schedule-state §6 (the 2026-07-12 cutover-flip), attention and Adam and rmsnorm kernel
  bodies **already lower from a `ScheduleState`**, marked `visibility:"opaque"` because their
  *internals* are still authored (the online-softmax lemma). Semantic-derivation supplies the
  **semantic region** those schedules reference (schedule-state's `region: SemanticRegionUid`
  foreign key) — the composition IS that region. No second owner: meaning lives here, schedule
  lives there, the `SemanticRegionUid` is the seam.

So the composite fused kernel is checked against its composition reference by the *existing*
schedule-state differential; the composition is checked against the math by Probe 4's shape.

### 4.5 Numerical guards and the CPU↔GPU seam (the epsilon finding, promoted)

Probe 2 surfaced that `log`/`sqrt` table gradients carry `+1e-8`. Ruling: **numerical
guards are annotations on a definition, never part of the formula.** A definition may carry
`guards: { denomEps?: f64, clampInner?: [lo,hi] }` (the GELU-tanh backward already clamps its
`tanh` argument to ±10); the derivation reads them where the surface wants numerical safety
and the reference/analysis path reads the pure formula. This keeps the *meaning* canonical
and byte-derivable while preserving the training-safe kernel — and makes the guard **visible
and reviewable** instead of buried in a grad lambda.

**The seam bar.** Byte-identity is the acceptance gate WITHIN an altitude (host reference,
host adjoint, WGSL-vs-WGSL). ACROSS the CPU↔GPU seam the gate is a **declared per-primitive
ULP tolerance** (measured: ~1 ulp typical, `exp` the worst at ~6e-5 abs over |x|≤5). The
definition schema therefore carries no cross-backend byte claim; it claims *one meaning*,
realized to each backend's native transcendentals within tolerance. This is the `pow`
precedent generalized, not a new risk.

### 4.6 Effects and refusal (category f)

RNG draws, fills/iota, and in-place optimizer state are **not** formulas and must refuse
formula-derivation loudly. They are declared as typed effects:

```
Effect =
  | Fill(value)                       // zeros/full/arange — pure source, index→const
  | RngDraw(distribution, streamRole) // rand/randn/bernoulli — a declared draw over an RNG stream
  | InPlace(stateRole, update: Expr)  // adamStep m/v — a state update whose Expr is derivable,
                                      //   but whose in-place effect and stream boundary are NOT
```

The RNG effect names its distribution and its stream (the §11 RNG stratum), so a draw is a
*declared* effect with reproducible identity, never a formula. This is the boundary Phase 5
(optimizers-as-programs) leans on: Adam's *arithmetic* is a derivable `Expr` (its VJP is
never needed), but its m/v in-place state and step boundary are effects — the composition
payoff is that the arithmetic derives while the effect is declared, exactly the split
`architecture-debt.md`'s foreach/islands stages want.

## 5. The deletion ledger (sized honestly, with consumer stories)

Every deletion names its consumer (house policy). Sizes are code-only SLOC of the current
files; the derivable *fraction* is what the campaign removes, not the whole file.

| surface | file | current | what deletes | consumer of the derived fact |
|---------|------|---------|--------------|------------------------------|
| S1 CPU bodies | `src/backend/cpu/numeric.ts` | 1288 | the elementwise `UNARY_OPS`/`BINARY_OPS` bodies + comparison/where bodies (the ~40 lines that are pure formula copies); reductions keep their loop bodies (the monoid is derived, the strided reduce loop is not) | `interpret(def)` is the reference; tests read it |
| S2 grad tables | `src/ops/registry.ts` | 515 | the `grad`/`ttGrad`/`tsGrad` lambdas (~120 lines) — all elementwise adjoints | `adjoint(def)·g` produces the grad graph |
| S2 custom backward | `src/frontend/custom-backward.ts` | 132 | `geluTanhBackward` + `geluErfBackward` (compositional → derived) | adjoint of the GELU composition; **`matmulBackward`/`linearBackward` STAY** (the 2 admitted contraction adjoints) |
| S3 WGSL formulas | `src/backend/webgpu/tile-ir.ts` | 2698 | the compound activation methods that are pure composition (`sigmoid`, `erf`, `clamp`, `fma`, `cdiv`) — the erf triplication collapses to one source | `emit(def)` walks the same `BlockExpr` chain |
| S4 exec-decl | `src/executor/execution-declaration.ts` | 239 | the hand-listed `ELEMENTWISE_UNARY_OPS` / `ELEMENTWISE_BINARY_WGSL` membership | `family(kind)` derives membership + arity |

**Net direction: a deletion campaign.** The removed hand-copies (~300+ SLOC of duplicated
formula/adjoint across four files, incl. the erf polynomial written 3× and the GELU
constants written 3×) exceed the added definition catalog + interpreter + adjoint pass +
normalizer (a single algebra module reused by all four surfaces). Baseline at design time:
`srcSLOC=66455`. The campaign-end commit names its deletions vs its additions per house
policy; growth without deletion triggers a re-review, not a waiver.

**The covenant lesson (R22, carried):** byte-identical regeneration does NOT prove the
definition owns anything — an opaque replay passes it. Acceptance = the null differential
PLUS the §4.1 schema gate (`assertNoDefinitionBody` — no JS body smuggled in a leaf) PLUS
this ledger actually landing. "The reference reproduces" is table stakes, not the proof.

## 6. Phases (each independently shippable)

- **P0 — the walking skeleton: the elementwise family.** Land the `Expr` schema + the
  `assertNoDefinitionBody` gate + `interpret`/`adjoint`/`normalize`/`emit`. Author the ~18
  elementwise primitives + the derived activations as definitions. **Gate:** the Probe 2/3
  differential in-suite (derived reference == CPU body byte-exact; derived adjoint == table
  after normalize, modulo declared guards) on BOTH the CPU and (ULP-toleranced) WebGPU paths.
  Delete the `UNARY_OPS`/`BINARY_OPS` bodies and the registry elementwise grad lambdas. **No
  behavior change; the guards (§4.5) become explicit annotations.**
- **P1 — reductions.** Monoid + epilogue as definition; the reduce *loop* stays a kernel,
  the monoid/mean-div derive. Reconcile with the already-data `REDUCTION_DECLARATIONS`.
- **P2 — composites.** Author softmax/log_softmax/CE/layernorm/rmsnorm/attention/rope as
  `Composition` terms; wire the composition as the schedule-state `SemanticRegionUid`'s
  region (§4.4). **Gate:** each composite's fused kernel == its composition reference via the
  existing schedule-state differential; CPU reference == composition (Probe 4 shape). Delete
  the GELU custom backwards.
- **P3 — contraction / attention adjoints.** Admit the matmul VJP + `linear` as the 2
  hand-proven backward rules with their differentials; conv=im2col∘matmul,
  linear=matmul+bias as compositions.
- **P4 — index-space.** Views/gather/scatter/cat as index-map definitions; the adjoint of an
  index map is its transpose (scatter ⇄ gather), replacing `narrowBackward` et al. with a
  derived transpose.
- **P5 — functionalization / optimizers-as-programs (Campaign 2 folds in).** Adam/SGD/scaler
  become programs over the same primitive algebra: their arithmetic is a derived `Expr`, their
  m/v state + step boundary are §4.6 effects. This is the composition payoff — the point of the
  algebra — and the exit to `architecture-debt.md`'s foreach/optimizer-island endgame.
- **P6 — the ledger.** Land the deletions; prove net-negative SLOC; retire any soak flag.

## 7. Usage boundary (what refuses)

The algebra is for *pure, functional* op meaning. It **refuses**, by typed effect, not
by silent fallthrough:

- **RNG state machines** — a draw is an `Effect(RngDraw)`, never a formula. Box-Muller's
  *arithmetic* is an `Expr`, but the draw and stream advance are effects.
- **IO / data sources** — `tensorFromArray`, checkpoint load, tokenization: not ops, out of
  scope.
- **Genuinely imperative in-place effects** — `adamStep` m/v, `unscaleGrad`'s inf flag: the
  update arithmetic derives, the effect is declared (§4.6).
- **Data-dependent control** — anything whose *structure* depends on runtime values is a
  schedule-state atom/lemma, not a definition (the online-softmax rescale is a lemma over
  there, referenced here as a composition region).

If a construct cannot be stated as a definition-term or a declared effect in one sentence,
it is held out until it can — the admission-pressure rule.

## 8. Risks (honest)

1. **The adjoint-of-broadcast trap.** The probe tests scalar elementwise; real ops broadcast,
   and the VJP of a broadcast is a *sum-reduction* back to the operand shape
   (`sumToShape` in `custom-backward.ts`). The adjoint pass must compose the scalar chain-rule
   with the broadcast-transpose (a reduction) — this is where a naive "just differentiate the
   formula" silently produces wrong-shaped grads. **Mitigation:** the index-space adjoint
   (P4, scatter⇄gather transpose) is the same machinery; broadcast is an index map and its
   adjoint is its transpose-sum. Do NOT ship P0 claiming broadcast coverage — P0 is the
   pointwise skeleton; broadcast-adjoint lands with P4's index algebra.
2. **CPU-vs-GPU numerical divergence (the `pow` precedent, measured).** §4.5. Byte-identity is
   unreachable across the seam (`exp` at 6e-5). The gate is declared ULP tolerance; a definition
   makes no cross-backend byte claim. The risk is *mis-setting the tolerance* — too tight
   flakes, too loose hides a real bug. **Mitigation:** per-primitive tolerances measured from
   the GPU probe, not guessed; the seam guard compares against tolerance, the within-altitude
   guard demands bytes.
3. **The epsilon-guard scattering.** §4.5. Guards today are buried per-lambda; promoting them to
   annotations risks either dropping one (training NaN) or over-guarding (biased grad — the
   `log +1e-8` visibly biases at x≈1e-4, 870 ulp). **Mitigation:** every guard migrated is
   diffed against its old lambda on the guard-active domain; the annotation is reviewed, not
   inherited silently.
4. **The era's length.** This is the ninth "latent decision → object" campaign; the algebra
   touches four surfaces and six phases. **Mitigation:** each phase is independently shippable
   and net-deleting; P0 alone (elementwise) is a complete win (Probe 2/3 already green) and can
   land without committing to P2+. The campaign can stop after any phase with the ledger still
   net-negative.

## 9. Red-team (3) and rulings

**RT1 — "This is just autodiff you already have; the registry `grad` fields ARE the
derivation."** *Ruling: no — they are a hand-written table, not a derivation.* The registry
stores `sigmoid` and its gradient as two independent hand-spellings; nothing checks they are
adjoints of each other. Probe 2 IS that check, and it found 3 discrepancies (2 epsilon
guards, 1 sign-of-zero) that a table cannot catch because there is no source to disagree with.
The deletion is real: 120 lines of grad lambdas → one adjoint pass + 2 admitted rules.

**RT2 — "Byte-identity is a mirage; the GPU probe shows you can't even reproduce `exp`, so the
whole 'derive the reference' claim is theater."** *Ruling: the claim is scoped, and the scope
is exactly where byte-identity holds.* The reference is CPU (host), and Probe 3 is 19/19
byte-exact there; the adjoint is a graph, and Probe 2 is 19/22 byte-exact (→ exact after
normalize, modulo declared guards). The CPU↔GPU seam was *never* claimed byte-identical —
§4.5 sets it to ULP tolerance from the start, and the `exp` outlier is the measured evidence
that this was the right call. Confusing "the reference is byte-derivable" with "every backend
is bit-identical" is the mirage; the design does not make the second claim.

**RT3 — "Composites can't really be 'just compositions' — the fused attention kernel is a
hand-tuned online-softmax algorithm, not `matmul∘softmax∘matmul`, so the composition is a
lie that will silently diverge from the kernel."** *Ruling: correct, and the design already
routes around it — that is why §4.4 does NOT re-derive the kernel.* The composition is the
*semantic reference* (what the kernel must equal); the *kernel* is schedule-state's owned
artifact, reached from naive attention by the admitted online-softmax **lemma** (schedule-state
§3.4), which computes different intermediates for the same function. The composition and the
kernel meet at the `SemanticRegionUid` seam and are reconciled by the *existing* schedule-state
per-move differential — not by pretending the kernel is a naive composition. The lie would be
claiming the composition IS the kernel; the design claims it is the kernel's *reference*.

## 10. Open questions (fork-only)

- **Q1 — guard representation.** Are numerical guards (§4.5) a fixed annotation vocabulary
  (`denomEps`, `clampInner`) or an `Expr`-level `guarded(expr, policy)` node? The former is
  simpler and covers the 3 observed cases; the latter is more general but risks re-smuggling
  imperative policy into the term. *Recommendation: fixed vocabulary until a 4th guard shape
  appears (rule of three).* Flagged for Vin only if P0 hits a guard the vocabulary can't state.

## 11. One-sentence test

If an op's meaning cannot be stated as a definition-term over the primitive algebra (or a
declared effect) in one sentence — such that its CPU reference, gradient, WGSL, and execution
declaration all *derive* from that one statement — reshape it before it lands.

## 12. Phase 0 — LANDED (2026-07-20)

The walking skeleton is in `src/ops/semantic/`. An op's meaning is a first-class
DATA term; the CPU reference body and the gradient DERIVE from it. Commits:
`the definition stratum` → `derive the CPU bodies` → `derive the gradients`.

### The definition schema, as landed
- `expr.ts` — the `Expr` term (18 unary + 8 binary + 6 compare + `where`
  primitives), the builders, and `assertNoDefinitionBody` (the
  `assertNoGeneratorLeaf` analogue: a definition is DATA — a function/string/
  buffer leaf, or a kind outside the algebra, is unconstructible; the covenant/
  R22 defense that keeps the byte differential un-gameable).
- `interpret.ts` — the f64 scalar interpreter; `compileUnary`/`compileBinary`
  ARE the CPU reference (design S1; "the interpreter itself is the reference").
- `adjoint.ts` — `deriv` (chain rule) + the CONSERVATIVE normalizer (exact IEEE
  folds only, so already-exact grads stay byte-exact) + two EARNED exact
  cancellations (`u/(u·u)→recip u`, `a·recip b→a/b`) + the `denomEps` guard.
- `catalog.ts` — the elementwise family as definitions (17 unary + 7 binary),
  guard annotations on log/sqrt, a `gradPolicy` ∈ {derive, none, hand}.
- `emit-rt.ts` — the derived VJP interpreted over the `RuntimeEngine` (LAZY,
  memoized leaves, so a grad that ignores a saved operand never forces it).

### The gate (productized Probe 2/3, standing)
`test/semantic-derivation.spec.ts` (cpu project, 42 tests), run against the
LANDED stratum: derived CPU body == hand `numeric.ts` body byte-exact (24/24);
derived VJP == hand `registry.ts` table byte-exact (22/22). **The 3 probe
divergences are RESOLVED**: `div.dB` sign-of-zero via the normalizer; `log`/
`sqrt` via the explicit guard.

### The guard ruling (the log-guard verdict — oracle-refereed)
**The table's `log`/`sqrt` `+1e-8` is a NUMERICAL GUARD (policy), not the
semantics.** Evidence from the PyTorch oracle (direct `torch.{log,sqrt}`
backward, the referee): PyTorch computes the UNGUARDED adjoint `g/x`,
`g·0.5/√x`; the table's epsilon biases the gradient near 0 —

| x | op | torch (oracle) | table (`+1e-8`) | pure adjoint |
|---|----|----------------|-----------------|--------------|
| 1e-4 | log grad  | 10000       | 9999.0001       | 10000 |
| 1e-4 | sqrt grad | 50          | 49.99995        | 50 |

So the table's guard is a *bug relative to the oracle* — it silently biases the
grad — but a defensible NaN-avoidance policy. **Ruling (design §4.5 + Q1 fixed
vocabulary):** the guard is preserved verbatim for P0 behavior-parity, promoted
to an explicit `{ denomEps: 1e-8 }` annotation on the definition — now
visible/reviewable instead of buried in a lambda. A future decision may drop it
to match the oracle; that is now a one-line annotation edit, not a code change.
The `TORCH_ORACLE_PYTHON` spec suite is unavailable in the P0 worktree (the venv
has no numpy/pip); the ruling's evidence was taken from direct torch.

### Deletions cashed, and the honest SLOC direction
Named deletions (house policy):
- `numeric.ts` `UNARY_OPS` (17) + `BINARY_OPS` (5) hand bodies → `compileUnary`/
  `compileBinary` (sub/div keep their option-carrying bodies).
- `registry.ts` elementwise grad lambdas → `makeUnaryGrad`/`makeBinaryTTGrad`:
  12 unary (relu silu sigmoid tanh neg abs exp log sqrt rsqrt sin cos) + 5 binary
  ttGrad (add mul div minimum maximum).

**P0 is NET-POSITIVE (+783 srcSLOC: 66455→67238)** — it lands the reusable
algebra ENGINE (interpret + adjoint + normalize + emit + the schema gate, ~750
code lines) against which only ~68 lines of hand copy delete. This is the
honest crossover trajectory the design's §5 ledger describes: the net-negative
is a FULL-campaign property. The engine is amortized — P1 (reduction monoids),
P2 (composites: the erf polynomial written 3×, the GELU constants written 3×),
P4 (index-space transposes), and P5 (optimizers-as-programs) all DELETE against
this same algebra with no new engine. P0 does not itself cash net-negative and
does not claim to; it builds the machine that lets the later phases cash.

### Gates (all green)
cpu: semantic-derivation (42), gradcheck (35), unary-ops. webgpu (per-device,
serial-exclusive): ops/autograd/conformance/op-registry (141), fusion suites
(92), test:gates / compiled-plan-parity (5). gate-wall `--profile training`:
build, test:gates, whole-step-diff, the tape/fullstack parity matrix
(fused×foreach × no-sched×cosine-lr, all 4), step-object-null, step-edit-null,
ring-probe, ledger-default, **124M-regression**, refusal-spec, checkpoint-seg —
all PASS. The strict-lifetime guard earned its keep: it caught the derived
`div.dA` materializing a redundant `y²` intermediate the whole-step merged plan
could not place (the silent-corruption class), fixed by the two normalizer
cancellations above.

### The WGSL seam (S3) — assessed, DEFERRED with rationale
Not landed in P0. The elementwise WGSL (`sqrt`/`exp`/… ) is already single-valued
via the registry's `wgslFnName`/`wgslInfix`/`wgslPrefix`; the remaining
triplication is the **erf polynomial and GELU constants**, which live in
`custom-backward.ts` (`geluErfBackward`) and `numeric.ts` — these are the
GELU/erf COMPOSITE, whose backward the phase table places in **P2**, not the
elementwise skeleton. Rewriting the `BlockExpr` activation emitter is load-
bearing GPU codegen at the ULP seam (the `pow`/`exp` precedent) with a marginal
P0 deletion; the design is explicit that a phase may stop with the ledger where
it is. S3 lands with P2 (composites) behind the existing schedule-state
differential + a declared per-primitive ULP tolerance at the CPU↔GPU seam.

### P1 (reductions) preconditions
- The monoid + epilogue as a definition (`sum`/`max`/`min` + `mean = sum÷count`);
  the reduce LOOP stays a kernel, only the monoid/mean-div derive.
- Reconcile with the already-data `REDUCTION_DECLARATIONS`
  (`execution-declaration.ts`) — the S4 membership derivation (`family(kind)`)
  is the natural P1 companion.
- The broadcast-adjoint trap (§8.1) stays OUT of scope until P4: P1 is the
  reduction monoid, not the broadcast-VJP (a reduction is the transpose of a
  broadcast — that machinery is P4's index algebra).

## 13. Phase 1 — LANDED (2026-07-20)

The reduction monoid stratum is in `src/ops/semantic/reduction.ts`. A reduction's
meaning is its MONOID — an identity element + an associative combiner, both
first-class `Expr` DATA — the single source from which the CPU reduce body
derives, the reduction gradient's class derives, and the `ReductionDeclaration`'s
monoid label is a projection. Commits: `the reduction monoid definitions` →
`derive the CPU reduction bodies` → `derive the reduction VJP class` → `the
declaration monoid derives` → this docs entry.

### The reduction schema, as landed
- **The monoid as data.** `ReductionDef` carries `identity` (the accumulator
  seed), `combine` (the associative fold over `x`=accumulator, `y`=element), and
  an optional `epilogue` (`x`=reduced, `y`=count) — all `Expr` terms, proven DATA
  by `assertNoReductionDefinitionBody` (the P0 schema-gate analogue, extended to
  the monoid frame). The 6 reduction ops (§2 category b): `sum`/`max`/`min`
  monoids, `mean` (the `sum` monoid + a `div(reduced, count)` epilogue — a
  COMPOSITION, not a 4th monoid), and `argmax`/`argmin` (INDEXED monoids — the
  value monoid combines, the index follows).
- **Derived facts, not stored labels.** `reduceMonoidOf(def)` PROJECTS the
  monoid label (`"sum"`/`"max"`/`"min"`) from the combine's root op — there is no
  separately-authored label to disagree. `compileArgBetter(def)` DERIVES the
  arg-reduce strictly-better predicate from the value combine (no separate
  comparison). `isStreamableMonoid` is the two-stage-form fact that REFERENCES
  `planChunkedFullReduction` + `computeChunkGeometry` as its realized image
  (composed, never re-owned — design §4.4).

### The tie-policy finding (the P1 analogue of P0's log-guard)
The `max`/`min` combiners are `where(cmp(elem, acc), elem, acc)` — **LEFT-BIASED
on ties/NaN**, exactly as the hand `if (val > best)` reduce loop was — NOT a bare
`max(x,y)` primitive, which diverges on ±0 ordering (`Math.max(-0,+0)=+0` but the
first-seen if-loop keeps `-0`) and on NaN. So the reduction family's combiner is
its OWN term; the elementwise `maximum` binary (which uses `Math.max`) is a
DIFFERENT op — the two are not forced to share a spelling. This is the byte-exact
bar that a naive "reuse the lattice op" would have silently failed.

### Deletions cashed, and the honest SLOC direction (the crossover BENDS)
Named deletions (house policy):
- `numeric.ts`: `sumAll`/`maxAll`/`minAll` (3 full-reduction helpers) + the four
  byte-identical `sum`/`max`/`min`/`mean` strided loops + the `normalizeDims`
  duplicate → ONE `reduceByMonoid` engine that reads (identity, combine,
  epilogue) from the definition (the strided LOOP stays a kernel, design §6 P1;
  only the monoid derives). `argReduce`'s `(initVal, isBetter)` params → derived.
- `torchlette.ts`: the `_dispatchReduction` op-name gradient branching
  (`opName === "max"||"min"`; `opName === "mean"`) → `def.gradKind` reads.
- `execution-declaration.ts`: the four hand-spelled monoid/meanEpilogue
  declaration literals → one `reductionDeclOf(def)` projection.

**P1 is +16 srcSLOC (66455→67238 was P0's +783; P1: 67238→67254).** This is the
crossover trajectory BENDING exactly as §5 predicted: P0 paid for the reusable
algebra ENGINE (+783); P1 adds only a thin per-op catalog (the reduction
definitions) and cashes nearly all of it back against the deduplicated reduction
loops. The engine is amortized — P1 neither re-pays it nor yet cashes
net-negative; it lands essentially FLAT (+16, ~50× smaller than P0's step), and the
net-negative remains a full-campaign property that P2 (the erf/GELU triplication),
P4 (index transposes), and P5 (optimizers) cash harder against this same algebra.

### HONEST SCOPE — the broadcast-VJP stays P4 (§8.1 honored)
The reduction GRADIENT is the broadcast-transpose (`sum`→broadcast,
`mean`→broadcast÷count; a reduction is the transpose of a broadcast). That
transpose is P4's index-algebra machinery (`_expandGrad`) and STAYS hand — P1 did
NOT ship a broadcast-VJP (the §8.1 trap). What P1 owns is the per-monoid LOCAL
factor + differentiability (`gradKind`: `sum`→unit `broadcast`, `mean`→`broadcast-scaled`
by 1/count, `max`/`min`/`arg`→`none`), single-sourced from the definition and
read by `_dispatchReduction` instead of hardcoded op-name strings. The `max`/`min`
mask-scatter VJP is likewise P4, not shipped here.

### The reduction WGSL seam (S3) — DEFERRED with rationale (as in P0)
Not landed. The GPU reduce identity/combine (`reduction-tile-ir.ts` `reduceOp`,
`makeReductionSpec`) is the realizer's, untouched. The CPU↔GPU monoid tie-break
already differs (the GPU `wgReduce` uses WGSL `max()` while the CPU loop is
left-biased) — a pre-existing ULP/tie seam, exactly the §4.5 CPU↔GPU boundary the
design encodes, not a P1 regression. Reduction WGSL derivation lands with the
composite S3 (P2) behind the existing schedule-state differential + a declared
per-primitive tolerance.

### Gates (all green)
cpu: full project **113 files, 1489 pass, 1 skip, 0 fail**, incl. gradcheck (35),
oracle op-conformance + autograd (reductions + their grads == PyTorch),
semantic-reduction (9 — schema gate, monoid projection, derived reduce ==
independent hand loop byte-exact, indexed isBetter == the arg comparison),
reduction-ops (11), execution-declaration (13). webgpu (per-device,
serial-exclusive): topk-kernel (6), compiled-plan-parity / test:gates (5),
conformance (69). gate-wall `--profile training` (build, test:gates,
whole-step-diff, fullstack/tape parity matrix, step-object/edit-null, ring-probe,
ledger-default, **124M-regression**) — the reduction paths are untouched on GPU
(only `numeric.ts` CPU bodies, the frontend grad classification, and the
declaration VALUES changed; the GPU reduce kernels + `REDUCTION_DECLARATIONS`
shape are byte-identical).

### P2 (composites) preconditions
- Author `softmax`/`log_softmax`/`cross_entropy`/`layernorm`/`rmsnorm`/`attention`/
  `rope` as `Composition` terms; wire each as the schedule-state
  `SemanticRegionUid`'s region (§4.4). **The P1 reduction monoids are the
  composite reduction sub-terms**: softmax = `exp∘(x−reduce(max))∘÷reduce(sum)`
  reuses the `max` and `sum` monoids; layernorm/rmsnorm use `mean` (the Welford
  variance pair-merge stays the realizer's admitted lemma, `reduction-skeleton.ts`
  `WELFORD_LEMMA` — referenced, not re-derived).
- **The erf/GELU triplication payoff** (the S3 deletion deferred from P0): the
  Abramowitz–Stegun `erf` polynomial written 3× and the GELU-tanh constants
  written 3× (§1) collapse to one source when the composite's WGSL emits from the
  composition; delete the GELU custom backwards (compositional → derived, §6 P2).
- **Gate:** each composite's fused kernel == its composition reference via the
  EXISTING schedule-state per-move differential (not a new mechanism); CPU
  reference == composition (Probe 4 shape); CPU↔GPU at declared ULP tolerance.

## 14. Phase 2 — LANDED (2026-07-20)

Composites derive. The GELU family is a *pure-elementwise* composition — a unary
`Expr` over the P0 algebra — so its CPU reference, gradient, and runtime
grad-graph all derive with **no new engine**; the softmax/norm family is authored
as `Composition` DATA with a Probe-4 reference gate. The named payoff — **the erf
triplication + the two GELU custom backwards — is cashed.** Commits: `the erf
realization + GELU composites` → `derive the GELU CPU bodies` → `derive the GELU
backwards (delete the custom rules)` → `the WGSL seam (single-source the erf
poly)` → `the reduction-composite reference` → this docs entry.

### The erf/GELU spine, as landed
- **`erf` admitted as an ALGEBRA PRIMITIVE** (`expr.ts`), like `exp`/`tanh`: its
  *realization* is the A-S 7.1.26 polynomial, single-sourced ONCE in
  `ops/semantic/erf.ts` (`ERF_A`/`ERF_P` + `erfApprox`); its *derivative* is the
  ANALYTIC gaussian `erf'(u)=2/√π·e^(−u²)` (`adjoint.ts`). This split (design
  §4.5) is the whole game: the forward approximates erf while the backward is the
  exact gaussian — so `gelu_erf`'s derived grad is `cdf + x·φ(x)`, reproducing the
  DELETED `geluErfBackward` (which was also analytic) to ~2e-16, NOT the derivative
  of the polynomial.
- **`GELU_TANH_DEF` / `GELU_ERF_DEF`** (`composite.ts`) — the two GELU forms as
  unary `Expr` terms; constants single-sourced from `erf.ts`. CPU body derives via
  `compileUnary`, backward via `makeUnaryGrad` (the adjoint over the runtime).
- **`emit-rt.ts` `emitErf`** realizes erf over the runtime for the backward's
  recomputed `cdf` term — reading the SAME `ERF_A`/`ERF_P`, no 2nd owner.

### THE TRIPLICATION'S DEATH (the named deletions, sized)
The A-S erf polynomial and the GELU-tanh constants had, between them, these owners
before P2; each now REFERENCES `ops/semantic/erf.ts`:

| magic-number family | owners before | after |
|---------------------|---------------|-------|
| erf A-S poly (`0.254829592…`, `p=0.3275911`) | `numeric.ts erf()`, `tile-ir.ts BlockExpr.erf`, `fusion-tile-ir.ts gelu_erf`, `custom-backward.ts geluErfBackward` — **4** | **1** (`ERF_A`/`ERF_P`) |
| GELU-tanh (`√(2/π)`, `0.044715`, `0.134145`) | `numeric.ts gelu()`, `fusion-tile-ir.ts gelu_tanh`, `custom-backward.ts geluTanhBackward` — **3** | **1** (`GELU_*`; `0.134145` derived = `3·GELU_TANH_C`) |

Named code deletions (house policy):
- `numeric.ts`: the hand `erf()` Horner helper + the tanh/erf `gelu()` loop bodies
  → `compileUnary(GELU_*_DEF.expr)` (design §4.3 S1). Byte-exact (semantic spec S1,
  0/713 mismatches over the probe grid).
- `custom-backward.ts`: `geluTanhBackward` + `geluErfBackward` (~70 code lines) →
  derived. Only the **2** admitted contraction adjoints (`matmulBackward`,
  `linearBackward`) remain — exactly the design §4.2 count.
- `tile-ir.ts` / `fusion-tile-ir.ts`: the erf poly + GELU constants → references to
  the shared source. The WGSL emit STRUCTURE is byte-identical (the fused kernels
  stay schedule-state's, §4.4) — this is the "reference shared constants" option
  the design §S3 sanctions, not an Expr-emit rewrite; it kills the triplication
  with zero GPU-codegen risk (fusion-tile-ir 61 + tile-ir-block 32 GPU tests green).

### The adjoint-vs-custom-backward verdict
| composite | derived adjoint vs the DELETED custom backward | ruling |
|-----------|-----------------------------------------------|--------|
| `gelu_erf` | maxAbs **2.2e-16** | identical — both are the analytic gaussian; the delete is free |
| `gelu_tanh` | maxAbs **1.1e-7** (only at \|x\|≳5.7) | the old backward CLAMPED its tanh arg to ±10; the derived adjoint drops the clamp. tanh has already saturated there (sech²(10)≈8e-9), so the clamp was a cosmetic guard — dropping it removes a hand-tuned magic number (no `clampInner` guard needed; the §4.5 rule-of-three is not tripped). Verified correct vs finite-difference of the derived forward to 1.3e-9. |

### The reduction composites (softmax / log_softmax / rmsnorm / layernorm)
Authored as `Composition` DATA (`composite.ts`): a small tensor dataflow over the
primitive algebra + the P1 reduction monoids, dim-agnostic, schema-gated
(`assertNoCompositionBody`). `interpretComposition` (`emit-rt.ts`) realizes a
composition over the runtime — the **Probe-4 reference**: the hand decomposed/fused
forward is proven to AGREE with the ONE composition source
(`test/semantic-composite.spec.ts`: each composite == its op forward AND == an
independent plain-JS row-wise reference, maxAbs <1e-6/1e-5 on CPU). The **fused
kernel is NOT re-derived** (§4.4): the composition is its reference, met at the
schedule-state `SemanticRegionUid` seam and checked there by the existing
per-move differential (RT3). `cross_entropy` DEFERS: its `log_softmax` core is a
composition, but the per-row gather at `target` is an INDEX-SPACE map whose
realizer is **P4's index algebra** — CE's full term (and tensor gate) lands with
P4, not by fabricating an index primitive here.

### The honest SLOC direction (the split)
`srcSLOC` 67254 (P1) → **67463 (+209)**. Two components, named per house policy:
- **The erf/GELU spine is net-NEGATIVE** (~−15 code): the erf primitive + GELU
  terms + `emitErf` (~90 code) minus the deleted custom backwards + hand erf/gelu
  bodies (~105 code). This is the design §5 ledger cashing — the triplication payoff.
- **The reduction-composite reference is net-additive** (~+220): the `Composition`
  schema + catalog + `interpretComposition` + the Probe-4 gate. This is the
  WARRANTED new mechanism — the charter's explicit ask (softmax/norm/CE as
  definitions-as-compositions), the single semantic SOURCE those ops now have, and
  the P5/schedule-state precondition (the `SemanticRegionUid` region to reference).
  It deletes nothing today because the ops were ALREADY hand-compositions; what it
  adds is the missing SOURCE + the gate proving the hand impl agrees with it.

### Gates (all green)
cpu: semantic-derivation (**46** — +4 composite rows: erf-primitive byte-exact,
GELU CPU byte-exact, GELU adjoints vs fd + analytic), semantic-composite (**5** —
schema gate + the 4 Probe-4 references), semantic-reduction (9), gradcheck (35,
incl. gelu tanh/erf backward), unary-ops (16, incl. gelu fwd/bwd). webgpu
(serial-exclusive): fusion-tile-ir (**61**), tile-ir-block (**32**) — the erf/GELU
WGSL emit. gate-wall `--profile training` (build, test:gates, whole-step-diff,
parity-fullstack, tape matrix ×4, step-object/edit-null, ring-probe, ledger,
**124M-regression**, refusal, checkpoint-seg) — PASS (`parity-fullstack` + `124M`
exercise gelu fwd + the derived backward on GPU end-to-end). The 5 torch-oracle
suites remain unavailable (no `torch` in the venv, as in P0/P1).

### P3 / P4 preconditions carried
- **P3 (contraction adjoints)**: the 2 admitted rules already isolated in
  `custom-backward.ts` (matmul + linear) — P3 admits them formally with their
  differentials; conv=im2col∘matmul, linear=matmul+bias as compositions.
- **P4 (index algebra)**: CE's target-gather, the broadcast-VJP (§8.1, still the
  reduction gradient's transpose held out of P1/P2), and scatter⇄gather transposes.
  The reduction-composite `Composition` schema is the shape P4's index maps extend.

## 15. Phase 4 — LANDED (2026-07-20)

The INDEX ALGEBRA is in `src/ops/semantic/index-map.ts`. An index-space map's
*meaning* is first-class DATA — a description of how output coordinates read input
coordinates (shapes / dims / offsets, never a closure or buffer) — and from that
ONE source every index op's gradient DERIVES by a single structural rule: **the
adjoint of an index map is its TRANSPOSE**. The nine independently hand-authored
view/gather/scatter/broadcast backward choices collapse to one `adjointIndexMap`
DATA→DATA function; the §8.1 broadcast-VJP debt P1/P2 explicitly deferred is
discharged; `cross_entropy` COMPLETES as a composition. Commits: `the index-map
schema + the transpose derivation` → `route the view/gather/scatter/reduction
backwards through the derived adjoint (delete _expandGrad)` → `cross_entropy
completes as a composition` → this docs entry.

### The index-map schema, as landed
- **The map as DATA** (`IndexMap`): `reshape`/`transpose`/`permute` (bijections),
  `narrow`/`cat` (injection / disjoint-union), `broadcast`/`reduce` (many-to-one),
  `gather`/`scatterAdd` (data-indexed) — each carrying only shapes/dims/offsets,
  proven DATA by `assertNoIndexMapBody` (the P0 schema-gate analogue extended to
  the index frame: a smuggled `narrowBackward` closure behind a leaf is
  unconstructible; the covenant/R22 defense). **The index TENSOR of a
  gather/scatter is NOT part of the map** — it is runtime data supplied to the
  realizer (as operands/eps are to `interpretComposition`), so the map stays pure
  DATA and the gate holds.
- **The transpose is the derivation** (`adjointIndexMap`, pure DATA→DATA): a
  bijection's adjoint is its inverse (permute → `invertPermutation`, transpose →
  itself, reshape → reshape-back); the DUAL PAIRS are single-sourced —
  **narrow⇄pad, cat⇄split, broadcast⇄reduce, gather⇄scatter** are each ONE
  transpose fact, not two hand copies. Unit-tested per kind + `invertPermutation`
  involution.
- **Realization composes runtime kernels, never re-owns them** (design §4.4):
  `realizeIndexAdjoint` dispatches the derived adjoint to the SAME `rt.*` movement
  the hand closures called — the transpose DECISION derives, the movement is
  referenced. `reduceToShape` (the implicit-broadcast VJP) and `broadcastOverDims`
  (the reduction VJP) live here as the broadcast⇄reduce transpose, single-sourced.

### The derived-vs-hand verdicts (byte-gated) + the named deletions
Each rewired frontend backward emits the IDENTICAL runtime movement it did by
hand; the byte gate (`test/semantic-index.spec.ts`, cpu) proves the realized
adjoint == an independent hand-written transpose f32-exact, and the PyTorch
**oracle** (now runnable — `TORCH_ORACLE_PYTHON=.venv/bin/python`; torch 2.9.1+cpu
IS present, contra the P0/P1/P2 "unavailable" note) confirms reshape / transpose /
permute / narrow / cat / gather / scatterAdd / mean forward+backward vs PyTorch.

| forward op (frontend) | hand adjoint that DELETED | derived adjoint |
|-----------------------|---------------------------|-----------------|
| `permute` | the inline `inverseDims` loop | `invertPermutation` (the map's transpose) |
| `narrow` | hardcoded `narrowBackward` choice | `narrow⇄pad` transpose |
| `cat` | the hand offset-narrow loop | `cat⇄split` transpose |
| `expand` | `_sumToShape` call | `broadcast⇄reduce` transpose |
| reductions (`sum`/`mean`) | **`_expandGrad`** (the §8.1 P1 debt) | `reduce⇄broadcast` (`broadcastOverDims`) |
| `gather` / `scatterAdd` | two hand closures | `gather⇄scatter` ONE transpose fact |
| `reshape`/`squeeze`/`unsqueeze`/`transpose` | four `rt.reshape`/`rt.transpose` closures | reshape-back / self-inverse |

Named code deletions (house policy): `torchlette.ts` `_expandGrad` (~18 code)
DELETED wholesale (routed to `broadcastOverDims`); `_sumToShape`'s 28-line body →
a 1-line delegator to the index algebra's `reduceToShape` (single-sourced); the
hand `inverseDims` loop; the seven view/gather/scatter/cat backward closures →
`backwardOfIndexMap(map)` one-liners. The broadcast⇄reduce movement now has ONE
owner (was `_sumToShape` + `_expandGrad`, two copies of the same transpose).

### cross_entropy COMPLETES (the P2 deferral, cashed)
`CROSS_ENTROPY_DEF` (`composite.ts`) is now a full `Composition`: per-sample loss =
`neg(gather(log_softmax(logits), target))`, reusing `LOG_SOFTMAX_DEF.root` and the
new **`gi` (gather-at-index) node** — the composite frame's INDEX-SPACE bridge,
whose adjoint derives through the index algebra (gather's transpose is
scatter/onehot, so CE's grad is `softmax − onehot(target)` — the exact fused-kernel
semantics, RT3's seam). The batch mean stays the caller's reduction (the fused
kernel returns per-sample loss). Gate (`test/semantic-index.spec.ts`): the
composition == the decomposed CE forward == an independent JS `−log_softmax[target]`
reference (maxAbs <1e-5, CPU). This is the single semantic SOURCE for CE that P2
deferred until the index family existed — the last composite made whole.

### The honest SLOC direction (engine-dominated, like P0)
`srcSLOC` 67463 (P2) → **67688 (+225)**. The index-algebra ENGINE (`index-map.ts`,
~226 code: the schema + the transpose derivation + the realizer + the broadcast⇄
reduce movement + the schema gate) is the warranted new mechanism — the charter's
explicit ask (the index-space category as the op algebra's index family, with
DERIVED transposes). It is net-additive exactly as P0 (+783) was: the deletions it
cashes are modest because the hand index backwards were already 1-liners
(`rt.transpose(grad, options)`) — what P4 buys is the UNIFICATION (9 hand adjoint
choices → 1 transpose fact), the four dualities single-sourced, the §8.1 P1 debt
discharged, and CE completed. The engine amortizes forward: **P5**
(optimizers-as-programs) reuses it for sharded/scattered state movement with no new
index mechanism, exactly as P1/P2 reused the P0 algebra. The full-campaign
net-negative remains a §5-ledger property.

### The WGSL seam (S3) — DEFERRED with rationale (as in P0/P1)
Not landed. The index-space movement kernels (`gather`/`scatterAdd`/`narrow`/
`transpose`/`expand` WGSL) are the realizer's, untouched — the transpose DECISION
derives here; the GPU movement is referenced (design §4.4). The CPU↔GPU tie/scatter
seam is the pre-existing §4.5 boundary, gated by the oracle + the training gates,
not byte-equality across backends.

### Gates (all green)
cpu: semantic-index (**11** — schema gate, the per-kind transpose derivation,
`invertPermutation` involution, the six realized-adjoint byte gates, CE
composition == decomposed == JS ref), semantic-composite (5), semantic-reduction
(9), semantic-derivation (46), gradcheck (35 — the view/broadcast/gather grads are
heavily exercised). **oracle** (now runnable): op-conformance + autograd +
optimizer + checkpoint parity vs PyTorch (35 — reshape/transpose/permute/narrow/
cat/gather/scatterAdd/mean fwd+bwd). Full cpu project: **1498 pass / 1 skip** (the
lone red was `websocket-relay` heads-routing — a networking flake, green on
isolated rerun; untouched by this change). gate-wall `--profile training` (build,
test:gates, whole-step-diff, parity-fullstack, tape matrix ×4, step-object/edit-
null, ring-probe, ledger-default, **124M-regression**, refusal, checkpoint-seg) —
PASS; `parity-fullstack` + `124M` exercise the derived reshape/transpose/expand/
reduction/gather backwards on GPU end-to-end (embedding = gather, CE loss), and the
**strict-lifetime** default (index-adjoint bugs are the silent-corruption class —
the derived scatter must place every element) held throughout.

### P5 (optimizers-as-programs) preconditions carried
- Adam/SGD/scaler arithmetic is a derivable `Expr` (P0 algebra); their m/v state +
  step boundary are §4.6 declared effects — the arithmetic derives, the effect is
  declared. The index algebra is the movement primitive sharded/foreach optimizer
  state needs (gather/scatter over param groups), reused not re-built.
- The 2 admitted contraction adjoints (`matmulBackward`/`linearBackward`) remain
  hand (design §4.2) — P3's formal admission is the only remaining backward that is
  not either a derived elementwise/index adjoint or a declared effect.
