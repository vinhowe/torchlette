# Attention scoreMod / maskMod seams (FlexAttention-class) — DESIGN [PROPOSED]

Task #64. Status: **PROPOSED — design only.** No implementation lands with this
doc. Protocol: review → hardening → separate implementation go.

Driving requirement: **Gemma-2 in the browser** (→ Gemma-2-2B + Gemma Scope SAEs =
the SAE-steering-in-a-tab demo). Gemma-2 attention needs three things the current
kernel can't express as data: attention-logit **soft-capping** (`cap·tanh(x/cap)`),
**sliding-window** attention on alternating layers (local ~4096 interleaved with
global), and **GQA** (already supported via the model-side `expandKV`). The goal is
to make these **declarations** — score/mask modifiers as expressions on the one
attention skeleton — so we add **no new hand-written kernel per pattern**. This is
campaign #62's expression seams (`SeamFn`/`ctx.applySeam`, already built) exercised
on their intended use case.

---

## G0 — Decomposed audit (measure before designing)

### (a) Current attention surface inventory

There is **exactly one** attention kernel family, authored in the tile-IR DSL (not
raw WGSL), in `src/backend/webgpu/attention-kernel.ts` (**832 lines, ~709 code-only
SLOC**), plus thin buffer-plumbing wrappers in `src/backend/webgpu/ops/fused.ts`
(`fusedAttentionForward`/`fusedAttentionBackward`, L624–742, ~120 SLOC) and a
frontend router `scaledDotProductAttentionImpl` in `src/frontend/decomposed-ops.ts`
(GPU→flash, CPU→decomposed matmul+softmax+matmul; holds the CPU-path causal mask).

Four tile-IR specs:

| Spec | Lines | SLOC | Role | Special-casing |
|---|---|---|---|---|
| `makeForwardAttentionSpec` | 101–221 | 103 | forward, online-softmax flash (BR=64 Q-rows, BC=32 KV-rows); never materializes N×N | causal (inline), **no GQA, f32-only, no decode** |
| `makeDPrecomputeSpec` | 223–257 | 32 | bwd helper `D[i]=rowsum(dO·O)` | none |
| `makeBackwardDQSpec` | 259–376 | 103 | bwd dQ; **recomputes** scores per row | causal (inline) |
| `makeBackwardDKVSpec` | 378–518 | 122 | bwd dK/dV; per-KV-block, Q-tiles via shared mem | causal + **causal tile-skip** (L462–466) |

Compression baseline decomposition of the 709 SLOC:
- **Dispatch/plan/cache boilerplate ≈ 330 SLOC (47%)** — 4 near-identical `dispatch*`,
  4 `plan*`, config-buffer cache. Not math; pure repetition.
- **Softmax/matmul math skeleton ≈ 280 SLOC (40%)** — shared across all specs.
- **Per-variant special-casing ≈ 90 SLOC (13%)** — the causal predicate at **three
  sites** (fwd L182–191, dQ L357–367, dKV L495–506) + the tile-skip.

**The only axis of variation today is `isCausal` (a runtime uniform).** There is *no*
GQA branch (the model broadcasts KV with `expandKV` before the op — HF `repeat_kv`
order — so the kernel sees full head count), *no* f16 kernel, *no* decode kernel (the
decode path is hand-decomposed matmul+softmax with an additive mask in `model.ts`).
So a modifier seam does **not** face a combinatorial variant explosion to compress —
the injectable surface is small and structurally parallel: **3 score sites and 3
mask/dScore sites, six edit points.**

### (b) Seam-cost audit — where modifiers inject

**Forward, pre-softmax score site** (`attention-kernel.ts` L180–191). QK dot at L180;
scores are live and pre-softmax through L191 (softmax begins at `scores.max(1)` L193):

```ts
const scores = ctx.dot(Q, K.T());               // [1,BC] raw QK^T
ctx.range(0, BC, (j) => {
  const kvPos = kvStart.add(j);
  const isActive = valid.and(kvPos.lt(N))
    .and(isCausal.eq(ctx.u32(0)).or(kvPos.le(qRow)));   // <- MASK lives here
  scores.set(j,
    isActive.select(scores.get(j).mul(scale), ctx.f32(F32_NEG_MAX)));  // <- SCALE = trivial score mod
});
```

Two facts fall out: (1) the score modifier injects at `scores.get(j).mul(scale)` —
the existing `.mul(scale)` is *already a score modifier* (the identity family). (2)
the causal mask is **not** an additive mask tensor; it's an inline predicate that
selects `F32_NEG_MAX` for masked positions. `qRow` and `kvPos` are both in scope —
enough for positional score/mask mods. This is exactly FlexAttention's
`score_mod(score, b, h, q_idx, kv_idx)` / `mask_mod(b, h, q_idx, kv_idx)` context.

**Backward dScore** (`dS = P·(dOV − D)`), recomputed per row (flash-style, scores
are NOT saved — backward only saves `Q,K,V,L(=logsumexp),dO,O`; see `decomposed-ops.ts`
`tensorsToSave`). dQ kernel L362–366:

```ts
const s   = ctx.dotRow(Q, K, j).mul(scale);            // recomputed raw score
const p   = isActive.select(s.sub(lVar.get()).exp(), ctx.f32(0));
const ds  = p.mul(dov.sub(dVar.get()));                // dScore
ctx.accumRow(dqAcc, ds, K, j);                          // dQ += dScore·K
```

dKV kernel L500–505 is structurally identical (scale folded into `ds` at L503). **The
score modifier's derivative multiplies into `ds` right here** — and because `s` is
recomputed from Q·K, the modifier's forward value is recomputed too, so its local
derivative w.r.t. the raw score is available inline (no extra saved tensor).

**Sliding window = iteration-range restriction, not a multiply.** The forward and dQ
loops currently iterate **all** KV tiles (`numKVTiles = ceil(N/BC)`, L165/L340) — no
range restriction; causal is handled by *masking inside* the tile. Only the dKV
kernel has a loop-level skip: L462–466 skips whole Q-tiles fully above the causal
diagonal:

```ts
const skipTile = isCausal.ne(ctx.u32(0))
  .and(qStart.add(ctx.u32(BQ_BW - 1)).lt(kvBlock.mul(ctx.u32(BC_BW))));
ctx.ifThen(skipTile.not(), () => { ...tile body... });
```

**This is the structural precedent for sliding-window-as-loop-bound.** A window is a
*two-sided* affine constraint on `(q_idx, kv_idx)`; when a mask is affine, the set of
KV tiles that can contain any active position is a contiguous range → we compute a
`[loTile, hiTile)` bound and skip the rest, rather than masking every element. That is
where the perf is (a 4096-window over a 32k context touches ~1/8 the tiles).

### (c) Feasibility probe — RESULT

Ran on device 4 (V100, vk-shim): a throwaway patch injected
`cap·tanh(score·scale / cap)` at the forward pre-softmax site (using the fluent tile-IR
`.div().tanh().mul()` chain — `tanh` is a first-class `UnaryOp`, `tile-ir.ts:1142`),
diffed against a naive materialized-softmax CPU reference computing the same
(soft-cap ∘ causal) attention.

```
=== SOFTCAP PROBE (softcap=30, causal, B1 H2 N40 D64) ===
  max abs err : 1.490e-7      max rel err : 2.993e-4     mean abs err: 1.316e-8
  VERDICT: PASS (< 1e-4)
=== SOFTCAP PROBE (softcap=0 → identity/disabled branch) ===
  max abs err : 8.941e-8      (bit-parity with stock causal attention)
```

**The injection point is exactly where the audit said (L180–191), the fluent
expression compiles and runs, and the f32 numerics match the reference to ~1e-7.** The
disabled branch (softcap=0 → `select(capped, raw)` picks `raw`) reproduces stock
causal attention identically — the seam is genuinely opt-in. The probe patch is not
committed (deleted per protocol).

*Caveat surfaced:* the probe injected the modifier **by hand** into a copied spec, not
through the `applySeam` machinery. It proves the *injection point and numerics*; it
does not exercise the *seam-declaration + fingerprint* path (that's §1's design, gated
in §7).

---

## Design

### 1. The seam contract

**Reuse #62's existing infra — do not invent a new mechanism.** `tile-ir.ts` already
has (`3175–3191`, `1664`):

```ts
type SeamArgs = Record<string, BlockExpr>;
type SeamFn   = (ctx: KernelContext, value: BlockExpr, args: SeamArgs) => BlockExpr;
// TileKernelSpec.seams?: Record<string, SeamFn>
// ctx.applySeam(name, value, args) -> fn ? fn(ctx, value, args) : value   // identity if absent
```

Currently live only in GEMV's `"epilogue"` seam. Modifiers build `BlockExpr` chains
that participate in the SAME IR (CSE, constant-folding, dtype promotion) — strictly
better than WGSL-string splicing.

**Declared seam points in the four attention specs:**

- `score_mod`: wrap the pre-softmax score. Forward L189, dQ L362, dKV L500. Args
  supplied by the kernel: `{ score, qIdx, kvIdx, head, batch }` plus a bag of
  per-head params (`cap`, `window`, …) passed as uniforms→`BlockExpr`.
- `mask_mod`: replaces the `isActive` boolean. Forward L184–186, dQ L359–361, dKV
  L497–499. Returns a boolean `BlockExpr`; `false` → the existing `select(..., NEG_MAX)`
  (fwd) / `select(..., 0)` (bwd `p`). Causal becomes the *default* `mask_mod`
  (`kvIdx.le(qIdx)`), not a hardwired branch — this deletes the ~90 SLOC of
  triplicated `isCausal.eq(0).or(...)` in favor of one declared modifier.

**API surface — what model code passes.** Mirror the proven `residualHook` pattern
(`model.ts:103`, a typed callback threaded through forward options). Add to the SDPA
frontend op an optional modifier spec:

```ts
type AttnModifier = {
  key: string;                          // stable identity → fingerprint (see below)
  scoreMod?: SeamFn;                    // (ctx, score, {qIdx,kvIdx,head,...cap}) => score'
  maskMod?:  MaskSpec;                  // boolean SeamFn + optional affine descriptor
  params?: Record<string, number>;      // per-head scalars (cap, window) → uniforms
};
scaledDotProductAttention(q, k, v, scale?, isCausal?, modifier?: AttnModifier)
```

`isCausal=true` is sugar for the built-in causal `maskMod`. Model code declares
modifiers from a small library (`softCap(cap)`, `slidingWindow(w)`, `causal`) and
composes them (§4). The kernel **selection layer** sees only `modifier.key` +
`params` — it does not introspect the closures.

**Fingerprint / template identity (CRITICAL — the "single source of truth at seams"
hazard).** A different modifier splices different IR → different WGSL → it MUST get a
different cache key, or we silently reuse the wrong shader. Today the attention WGSL
cache key is **just `headDim`** (`attention-kernel.ts` `getTileIRWGSL("fwd:"+headDim,…)`,
L625) and the pipeline key is `` `${prefix}:tile:${headDim}` `` (L549) — **insufficient
the moment modifiers vary.** Design rule, mirroring GEMV's `epilogueKeyFragment`
(`gemv.ts:172`):

> The WGSL cache key AND the pipeline key MUST include `modifier.key`. `modifier.key`
> is the single source of the modifier's structural identity; params that vary at
> runtime (a per-head `cap` value) flow as **uniforms** (data), NOT into the key.
> Structural choices (soft-cap present? window present? window as loop-bound vs mask?)
> ARE in the key. Assert at the seam that a key collision implies identical emitted
> WGSL (extend the existing WGSL-cache to store+compare, like `getConfigBuffer` does
> for uniforms).

**Interaction with decode `bucketKeys` and the tape.** The decode path
(`model.ts`/`generate.ts`) is captured (`api.capture`) and keyed by `` `kv:bkt${kvBucketLen(...)}` ``
— the bucket length is the one structural discriminator the arg-tensor surface can't
express. Gemma-2's *per-layer alternating* global/local attention is a new structural
discriminator: **`modifier.key` must be folded into the capture/tape bucketKey**
exactly as steering `alpha`/`layer` are today (`taped-decode.ts` `` `steer:${layer}:a${alpha}:kv${kv}:bkt${bucketLen}` ``).
The modifier closure is closure-captured and FROZEN for one CapturedFn's lifetime
(same contract as `residualHook`): sound because one generation = one modifier set. A
window whose *value* changes per step would need the value in the key; a fixed
per-layer window (Gemma-2's case) is structurally constant → only the presence/kind is
keyed, and `params` ride as uniforms.

### 2. Backward — the paired-derivative story

**The house pattern is paired expressions (`custom-backward.ts`), and we follow it:**
a `scoreMod` is authored as a *pair* — the forward `BlockExpr` transform and its local
derivative `dScoreMod/dRawScore` as another `BlockExpr` transform. The expression
system does **not** auto-differentiate arbitrary tile-IR (there is no reverse-mode
over `BlockExpr`); requiring the author to supply `dMod` is consistent with how
matmul/linear/gelu backward are hand-paired today.

```ts
type ScoreMod = {
  fwd: SeamFn;                               // rawScore -> modScore
  dRaw: (ctx, rawScore, modScore, args) => BlockExpr;  // d(modScore)/d(rawScore)
};
```

The backward kernels multiply `dRaw` into `ds` at the existing dScore site
(dQ L365, dKV L503) **before** `accumRow`. Because scores are recomputed (flash-style),
both `rawScore` and `modScore` are available inline — no extra saved tensor.

**Worked example — soft-cap.** `f(x) = cap·tanh(x/cap)`, so
`f'(x) = 1 − tanh²(x/cap) = 1 − (f(x)/cap)²`. In tile-IR:

```ts
softCap(cap).fwd  = (ctx, s) => s.div(cap).tanh().mul(cap);
softCap(cap).dRaw = (ctx, s, m) => ctx.f32(1).sub(m.div(cap).mul(m.div(cap)));
// dRaw reuses the recomputed modScore m — cheap, no extra tanh
```

**Recompute-vs-save tradeoff under checkpointing.** The kernel already recomputes raw
scores in backward, so the modifier's forward is recomputed for free — we do **not**
save modified scores. This is the correct default (memory-flat, matches the existing
flash discipline). The only case that would motivate saving is a modifier whose
forward is far costlier than its derivative *and* whose derivative can't be expressed
from `(rawScore, modScore)` — none of Gemma-2's modifiers are like this
(soft-cap's `dRaw` is a cheap polynomial in `modScore`).

**Scope note:** backward is *designed here* but **inference-first**. See §5 — the
SAE-steering demo does not need attention backward. Backward lands only if a training
demo needs it; the paired-derivative contract above is the spec for when it does.

### 3. Sliding window as structure, not arithmetic

**Compilation rule:**

> If a `maskMod` is declared **affine** in `(qIdx, kvIdx)` — i.e. the active set is
> `{ kv : lo(q) ≤ kv ≤ hi(q) }` with `lo,hi` affine in `q` — the kernel restricts the
> KV-tile iteration range to `[floor(lo/BC), ceil(hi/BC))` (a computed loop bound),
> and applies the exact per-element predicate only in the boundary tiles. Otherwise
> (non-affine mask) fall back to multiplicative/`select` masking over the full range.

The `maskMod` therefore carries an optional **affine descriptor** `{ loOfQ, hiOfQ }`
(coefficients, as data) alongside its boolean `SeamFn`. Causal is the degenerate
affine case `lo=0, hi=q` — and its existing dKV tile-skip (L462–466) is exactly this
rule already applied by hand; we generalize it to two-sided bounds and add the
symmetric bound to the forward + dQ loops (which currently have none).

- **Sliding window `w`:** `lo(q)=q−w+1, hi(q)=q` (causal local). Affine → loop bound.
  A 4096-window over a long context skips the vast majority of tiles.
- **Composition `causal ∘ window`:** intersect the two affine constraints → the
  window's `lo` dominates; single contiguous tile range. The per-element boundary
  predicate is the `and` of both `maskMod`s.
- **Static-KV decode layout:** the decode path reads a bucketed prefix
  `kSlot.narrow(2,0,bucketLen)`. A window means: only the last `w` cache positions can
  be active for the current query. The loop-bound rule becomes a `narrow` lower bound
  (start the KV read at `max(0, len−w)`), which *shrinks the bucket* → fewer tiles AND
  less cache traffic. The affine descriptor is evaluated against `posOffset` at
  plan-build time; because the window is structurally constant per layer, it folds
  into the same bucketKey the decode path already keys on.

Where the window is applied in the *fused prefill* kernel, it's the loop-bound rule
above. In the *decode* path (currently hand-decomposed matmul+softmax with an additive
mask), the window is simplest as an additive-mask term for now (the decode path is not
the perf-critical prefill), with the `narrow` lower-bound as a follow-on optimization.

### 4. Gemma-2 worked end-to-end (on paper)

Declarations (model code, per layer):

```ts
const causal      = maskMod.causal;                       // kv <= q ; affine lo=0 hi=q
const cap         = softCap(50.0);                        // Gemma-2 attn logit cap = 50
const window      = maskMod.slidingWindow(4096);          // affine lo=q-4095 hi=q

// global layers (even): soft-cap ∘ causal
sdpa(q, expandKV(k), expandKV(v), scale, /*isCausal via*/ {
  key: "gemma2.global", scoreMod: cap, maskMod: causal });

// local layers (odd): soft-cap ∘ causal ∘ sliding-window(4096)
sdpa(q, expandKV(k), expandKV(v), scale, {
  key: "gemma2.local.w4096", scoreMod: cap,
  maskMod: maskMod.and(causal, window) });               // intersect → affine, loop-bounded
```

- **`scoreMod: cap`** injects at the three score sites; backward `dRaw = 1−(m/cap)²`
  (only if training).
- **`maskMod: causal` / `and(causal, window)`** — affine → loop-bound compilation
  (§3). Global vs local are two different `key`s → two templates → correctly distinct
  in the WGSL cache and the decode bucketKey.
- **GQA reuse:** unchanged. The model calls `expandKV` (HF `repeat_kv`) before the op,
  as Qwen3 does; the kernel sees full head count. No kernel change.
- **Final-logit soft-cap (Gemma-2 caps the LM head logits at 30.0):** this is **not an
  attention modifier** — it's a plain elementwise `cap·tanh(logits/cap)` on the
  `[B,T,vocab]` output of the LM head, expressed with the existing frontend
  `tanh`/`mul`/`div` ops (or a fused elementwise if profiled hot). It lives in the
  model's `forward` after the LM head, before the loss/argmax. Note it in the Gemma-2
  model port, not in the attention seam.

**What ELSE Gemma-2 needs beyond attention (gap audit vs existing ops):**

| Need | Status | Note |
|---|---|---|
| RMSNorm | **EXISTS** (`nn/rmsnorm.ts`, fused kernel) | reusable as-is |
| **Pre+post norm sandwich** (extra norm after attn AND after MLP) | **EXISTS** (compose two `RMSNorm` instances) | Qwen3 already has `inputNorm`+`postAttnNorm`; Gemma-2 adds post-block norms = more instances, no new op |
| RMSNorm `(1+weight)` convention | **weight-load detail** | not an op change; apply at load or bake into weight |
| GeGLU MLP | **EXISTS** (`gelu` op + gated structure) | swap Qwen3's `.silu()` → `.gelu({approximate:"tanh"})`; same gate/up/mul/down skeleton |
| QK-norm | not needed for Gemma-2 | Qwen3-only |
| RoPE | **EXISTS** (fused RoPE kernel) | Gemma-2 uses standard RoPE; reusable |
| Attn logit soft-cap | **THIS DESIGN** (scoreMod) | — |
| Final logit soft-cap | **EXISTS** (elementwise tanh) | model-level, §4 above |
| Sliding-window / global alternation | **THIS DESIGN** (maskMod + loop bound) | — |

**Net gap: essentially just the two attention modifiers in this doc.** The norm
sandwich, GeGLU, RoPE, and GQA are all reuse. This confirms the audit's finding that
Gemma-2's novel surface is concentrated in attention.

### 5. What this does NOT do (scope fence)

- **No paged attention.** Static-KV bucketed cache only (as today).
- **No block-sparse mask precomputation** beyond the affine loop-bound rule (§3). We
  do not build/precompute a block-sparsity bitmap (FlexAttention's `BlockMask`); the
  affine→loop-bound rule covers causal + sliding-window, which is all Gemma-2 needs.
  Non-affine masks fall back to full-range `select` masking (correct but not
  block-skipped) — acceptable because we have no non-affine use case.
- **No training-side flex is required for the demo.** Backward is *designed* (§2) but
  **implementable later.** **The SAE-steering-in-a-tab demo is inference-only** —
  steering perturbs the residual stream (the existing `residualHook`), reads SAE
  activations, and generates; it never calls `backward()` through attention. So the
  backward seam is not on the demo's critical path. Ship forward-only; land backward
  if/when a browser *training* demo needs soft-cap/window (e.g. fine-tuning Gemma-2).
- **No new env flag.** Modifiers are data on the op call; the null modifier is the
  default and is bit-identical to today (probe §c confirms).

### 6. Falsification duty

**Riskiest assumption:** "a user-supplied score expression injected at the pre-softmax
site produces numerically correct attention in the online-softmax kernel" — i.e. that
the injection point is right *and* the running-max/rescale machinery stays correct
when the score is nonlinearly transformed (soft-cap changes the score magnitude, which
interacts with the `mPrev`/`correction` online-softmax bookkeeping).

**Cheapest killing experiment:** the G0(c) probe — inject `cap·tanh(x/cap)` by hand,
diff vs a naive materialized-softmax reference. **Result: PASS, max abs err 1.49e-7**
(f32, causal, cap=30). The online-softmax rescale is correct under the nonlinear score
transform because the transform is applied *before* the running-max is taken (L189,
inside the per-tile score loop) — the max/rescale sees the modified scores, exactly as
a materialized softmax would. The disabled branch (cap=0) is bit-identical to stock
attention. **Assumption survives falsification.**

Residual risk not covered by the probe: (i) f16 accumulation (probe was f32 — Gemma-2
browser will likely run f16 QKV; the modifier arithmetic should stay in f32 like the
existing `scale` does, `bitcastTo("f32")`); (ii) the `applySeam`+fingerprint path (probe
injected by hand). Both are gated in §7.

### 7. Gate ladder for eventual implementation

1. **Per-modifier parity vs unfused reference composition.** For each modifier
   (soft-cap, sliding-window, causal, and compositions), the fused-kernel output must
   match a naive materialized `matmul→scoreMod→mask→softmax→matmul` reference to ~1e-4
   (f32) — the G0(c) probe, promoted to an in-suite webgpu spec and extended to each
   modifier. Include an f16-QKV variant with f32 modifier arithmetic.
2. **Qwen3 unchanged = the null-modifier gate.** Qwen3's attention is the identity
   scoreMod + causal maskMod. Its trajectories must be **bit-identical** before/after
   the seam lands (the disabled branch reproduces stock attention — probe confirms the
   principle). Run `examples/qwen3/kv-differential.ts` (cached==uncached token
   sequence) and the taped-decode gates green.
3. **Tape/decode gates.** `modifier.key` folded into the capture bucketKey;
   `taped-decode.ts` + `kv-differential.ts` green with a modifier present (Gemma-2
   local/global alternation must key distinctly and replay the right mask).
4. **Fingerprint seam assertion.** WGSL-cache stores emitted source per key; a key
   collision with differing source throws (mirror `getConfigBuffer`'s byte-compare).
5. **Backward parity (only if training lands).** Paired `dRaw` checked numerically vs
   finite-difference and vs an unfused autograd composition, per modifier.
6. **Full suite** (`npm run test`) + **gates** (`npm run test:gates`) green;
   weight-norm vector snapshot (the causal-branch deletion should hold SLOC flat or
   down — the seam replaces ~90 SLOC of triplicated `isCausal` branches).

---

## Appendix — precise anchor points (for the implementer)

- Score sites: `attention-kernel.ts` L189 (fwd), L362 (dQ), L500 (dKV).
- Mask sites: L184–186 (fwd), L359–361 (dQ), L497–499 (dKV).
- dScore sites: L365 (dQ), L503 (dKV).
- Loop-bound precedent: L462–466 (dKV causal tile-skip); fwd/dQ loops L165/L340 (no bound yet).
- Cache keys to extend: `getTileIRWGSL` key L625 + pipeline key L549 (currently `headDim` only).
- Seam infra: `tile-ir.ts` `SeamFn`/`SeamArgs` (3175–3191), `applySeam` (1664); live example `matmul/gemv.ts` epilogue (146–170, 384/449); key fragment `epilogueKeyFragment` (172–185).
- Consumer template: `model.ts` `residualHook` (103–108, 442–444, 626/638); SDPA call sites (229–235, 294); decode decomposed attn + additive mask (239–240, 301–312); `expandKV` GQA (170–180); bucketKey (`generate.ts` 294–296).
