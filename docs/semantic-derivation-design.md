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
