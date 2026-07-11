# Kernel Editor: Validity & Completeness

Analysis for the browser kernel editor built on torchlette's tile-IR, with schedule edits as
napkin-calculus gestures {group-partition/tile, stream-partition, recolor/residency, fuse/weave}
(Abbott & Zardini, arXiv 2412.03317). Grounded in a full read of
`src/backend/webgpu/tile-{ir,ops,lowering,compiler,dispatch,autotune,access-analysis}.ts`,
`matmul/*`, `attention-kernel.ts`, `adam-kernel.ts`, `reduction-tile-ir.ts`,
`ops/reductions.ts`, `optim/packed-dispatch.ts`, and CLAUDE.md's failure ledger.

---

# Q1 — Is every state valid?

## 1.1 Formalization

**State** = `(G, D)` where `G` is the stratum-2 semantic graph (fixed under all stratum-3
edits) and `D` is the decoration: residency coloring `c: wires → levels`, partition labels
`(g_a, s_a)` on axes, fusion grouping, seam attachments, divisibility annotations. Moves are
partial functions on `D` only.

**Three validity tiers** (define "valid" separately at each; conflating them is where the
confusion in the question lives):

- **T0 — well-formed**: the decoration typechecks. Partition sizes divide (or pad) axis
  sizes; colorings respect level capacities `M_ℓ`; stream bodies reference declared carried
  state; fusion groups are convex in dataflow order; barriers sit in workgroup-uniform
  control flow; binding counts/sizes respect device limits. Failures here don't lower — or
  lower to garbage WGSL (the repo has seen this class escape: `let offset0 = ;` from the
  rank-0 comparisonOp bug).
- **T1 — algebraically valid relative to a lemma library L**: the decorated program denotes
  the same function as `G` **over ℝ** (exact real arithmetic), provable from the calculus's
  theorems plus the lemmas in L.
- **T2 — numerically certified**: the compiled artifact's output matches the stratum-2
  reference within declared tolerance τ on a declared input distribution, empirically
  (differential contract).

**Move classes:**

**(a) Semantics-preserving by construction.** `group-partition(axis, g)` (map over groups,
rejoin — an equality whenever the divisibility/padding precondition holds), `recolor(array,
level)` (residency never changes the function, only cost/feasibility — guarded by capacity),
`weave(axis)` (naturality of broadcasting), and `fuse(F∘G)` **when both streamability facts
are already established** (the composition theorem itself is unconditional). For these, the
applicability guard IS the soundness proof: **the move set is closed** — every T0-legal
application of a class-(a) move to a T1-valid state yields a T1-valid state. This is the
"typed rewrite" property and it is real, not aspirational: these moves are equalities in the
calculus.

**(b) Lemma-gated moves.** `stream-partition(axis, s)` requires the streamed function be
streamable along that axis — exhibit the head/body `(F, B)` decomposition with carried state.
For `sum`, `max`, `min` this is structural (associative monoid — checkable). For **softmax**
it is not: online softmax (carried state `(m, l)`, body with correction factor
`exp(m_old − m_new)`) is an algebraic *discovery* (Milakov & Gimelshein 2018), not something
a structural checker derives. Likewise: split-K's out-of-order combine (needs fp-tolerant
associativity+commutativity of the reduction), Welford variance streaming, streaming
logsumexp, and the repo's own `expm1`-Horner bias-correction rewrite in `adam-kernel.ts`
(`emitBiasCorrection`, avoiding `1−pow(β,t)` cancellation) — each is a **registered lemma**:
an entry `(op, axis-role) → (F, B, carried-state, merge)` whose truth is established outside
the calculus and admitted empirically via a differential contract. Formal statement:
**closure is relative** — soundness of the system = soundness of the kernel calculus
(proved) ∧ truth of every lemma in L (empirically certified within τ). The lemma library is
the system's trust boundary and its growth surface.

**(c) Ill-formed vs well-formed-but-wrong.** T0 failures are cheap and honest — they crash.
The dangerous class is T0-passing, T1-violating states: they run and silently produce wrong
numbers. The repo's ledger is dominated by exactly this class, and — the critical
observation — **nearly all were produced by the system's own optimizer, not by a user**:
row-program `scalarOutput` misclassification (every row got row 0's value), CSE dropping
`outputIndex` (SDPA dQ/dK collapsed onto dV), WGSL `pow(x<0)` NaN, the frozen Adam
`step_size` uniform (silently wrong LR schedule, twice misattributed to "benign fp noise").
Consequence: **T2 certification must guard the lowering/compiler, not just user edits.** A
legal move implemented by a buggy lowering produces a state the calculus calls valid and the
GPU disagrees with. The differential runs even on pure-geodesic paths.

## 1.2 Eager attention → flash: does a geodesic exist?

Yes — and the repo's `attention-kernel.ts` is literally its endpoint. The derivation, every
intermediate T1-valid:

1. **weave** over `B·H` (batch/head axes threaded over the attention block). By construction.
2. **group-partition** Q-rows with `g = BR` (=64 in the repo). Intermediate state: tiled
   attention that still materializes full score rows per Q-block. Valid, memory-heavy, slow.
3. **stream-partition** the KV axis of `softmax ∘ matmul` with `s = BC` (=32). **Requires the
   online-softmax lemma.** The lemma's carried state and body are exactly
   `attention-kernel.ts:410-468`: `mPrev/lPrev/oAcc`, `correction = exp(mPrev − mMax)`,
   `oAcc.mul_(correction)` per tile.
4. **fuse** QKᵀ → softmax → PV via the streamability-composition theorem. By construction
   given step 3.
5. **recolor**: K/V tiles → shared memory (`ctx.load2D`, with V aliasing K's slot via
   `{reuseShared: K}`), Q row and `(m, l, oAcc)` → registers.
6. Store logsumexp `L = m + log(l)` as a side output for backward recompute — note this is a
   stratum-2-visible choice (a checkpoint/recompute contract), not a schedule move; see §1.4.

Every state on this path computes attention exactly (over ℝ). Some are much slower than the
endpoint — **valid ≠ good** — but none are wrong. So the geodesic exists. The load-bearing
qualification: **it exists only because the online-softmax lemma is in L.** Delete the lemma
and there is no valid path — naive streaming of softmax (no rescale) is a well-formed-but-
wrong state, and the only route to flash passes through it. So:

> **The puzzle model and the geodesic model are the same system at different lemma-library
> completeness.** A missing lemma is a hole in the valid-state manifold; freeform mode
> tunnels through it; the endpoint differential that certifies the excursion is precisely
> the empirical evidence that mints the new lemma. Freeform is the lemma-discovery mode,
> not a concession to sloppiness.

## 1.3 (i) fp-tolerance: what "equivalent" means

Moves are **ℝ-equalities, not fp-equalities**. Every reassociation — tree vs. sequential
reduction, split-K combine order, chunked-sum chunk order — changes bits. Therefore:

- Define **per-move validity algebraically** (T1, exact over ℝ). This keeps class-(a) moves
  genuinely always-valid: no per-move numeric check needed, no tolerance bookkeeping per
  gesture.
- Define **numeric validity at the endpoint** (T2): scheduled-artifact vs. stratum-2
  reference, within a declared τ, on a declared input distribution. Do **not** assign per-move
  tolerances — they compound (triangle inequality gives Σ τᵢ over a derivation, which is
  useless), and the repo's hard-won lesson is that *component isolation cannot exonerate
  trajectories* (the clip-divergence bug was "proven benign" component-wise, twice, and was a
  real frozen-uniform bug at trajectory level). Repo precedent for the contract shape:
  `parity-fullstack-tl.ts` — per-step losses agree to ~1e-5 over 30 steps.
- Corollary: an algebraically valid move can be **numerically catastrophic** (streaming a
  cancellation-heavy sum; f16-staged tiles of large-dynamic-range data). Geodesic mode does
  not obviate T2. The green edit history proves T1; only the differential proves T2.

## 1.4 (ii) Does the stratum-2/3 split rescue always-validity?

Yes, cleanly — **if the boundary is enforced in the representation**. The rule that makes it
crisp: *a stratum-3 move may change only association and rounding; anything that changes the
ℝ-function is a stratum-2 edit.* f16 accumulation, gelu tanh-vs-erf, approximate exp, E3M0 —
all change the ℝ-semantics → they are typed-wire edits on the semantic graph, with their own
differential obligations. Under that rule, stratum-3 always-validity holds at T1 by
construction (§1.1a) and relative-to-L for lemma moves (§1.1b).

The enforcement cost is real, and the repo shows exactly where the boundary currently leaks —
places where semantics-bearing choices live in the *schedule* surface and must be hoisted:

- `smemElemType: "f16"` (`tile-ops.ts`) can stage an f32 binding as f16 in shared memory —
  precision-losing, sitting in a load option. Under the rule this is a stratum-2 wire dtype,
  not a recolor parameter.
- Matmul output-dtype promotion `max(dtypeA, dtypeB)` and the AMP cast placement — currently
  backend policy; the editor must surface them as typed wires.
- The attention backward's saved-logsumexp (recompute-from-L) is a checkpointing contract:
  which values are materialized for backward is stratum-2-visible (it changes what autograd
  sees), even though *where* they're materialized is stratum-3.

So the rescue is genuine but demands that precision and save/recompute contracts be **typed
into the semantic graph** — after which "f16 accumulate here" is a graph edit with its own
certification, and the cost model reads dtype from stratum 2 (the napkin's `q^(1+β)` knob).

## 1.5 (iii) Is geodesic mode alone sufficient? No — freeform is load-bearing

Evidence for geodesic sufficiency in the common case: the four flagship kernels are all
reachable via moves + parameters given ~3 lemmas (§Q2 audit) — day-to-day schedule work
(retile, re-stage, refuse, re-split) never needs to leave the manifold.

Evidence that freeform is structurally necessary:

1. **The lemma library is never complete.** Online softmax was a research publication; the
   repo's own expm1 bias-correction is a numerics lemma nobody's structural checker would
   emit; the next one (say, a streaming rewrite for an optimizer statistic) will arrive as an
   expert's conjecture. Geodesic-only freezes the calculus at ship-time. The design already
   concedes this — "new algebraic lemmas admissible empirically via differential contracts"
   *is* freeform mode with paperwork.
2. **The repo's own applicability checkers are admittedly incomplete shadows.** The napkin
   analysis calls `FUSIBLE_OPS` "an ad-hoc shadow of this theorem"; the fusion detector is
   consecutive-only and misses non-adjacent runs (CLAUDE.md open target #5). A user who can
   *see* a legal fusion the checker can't license needs a way to do it and certify after.
3. **Porting published kernels.** Reproducing FlashAttention-3 or a paper kernel is
   endpoint-first work: build the target state, certify, then (optionally) reconstruct the
   derivation. Forcing derivation-first inverts the expert workflow.
4. **Debugging is intentionally non-semantics-preserving** (extra debug outputs, reference-
   path swaps, bisection states). The editor needs wrong-on-purpose states.

The architectural coupling that makes dual-mode more than a compromise: **a freeform
excursion that ends T2-green should be mined** — diffed against its launch state, decomposed
into known moves where possible, and the irreducible residue offered for registration as a
named lemma/move. Otherwise expert knowledge evaporates into unexplained endpoint states and
the geodesic manifold never grows. (This also gives the honest UX framing: geodesic mode =
"can never be wrong, may be slow"; freeform mode = "certifies endpoints, mints lemmas.")

## 1.6 (iv) Dead-ends in the move graph

Distinguish two topologies:

- **With inverses: no dead-ends, in principle.** Every move is an equality, hence invertible;
  therefore every valid state is connected to the bare semantic graph by un-derivation, and
  the valid-state graph is **connected through the root**. From any valid state you can reach
  any other valid state (retreat, re-derive). The precondition is engineering, not math:
  persistent node identity across rewrites and direction-tracked, un-appliable moves (exactly
  the "load-bearing engineering" the napkin analysis flags). If undo exists, geodesic mode
  can never strand you.
- **Monotone-forward-only editing dead-ends readily.** Moves don't commute in practice:
  partition the wrong axis first and a later stream-partition's capacity precondition fails;
  greedy descent hits local optima. And the repo exhibits **implementation-induced**
  non-commutation the calculus doesn't have: epilogue fusion is disabled when kSplit is
  active (`tile-matmul.ts:164`, `computeKSplitFactor` guard) — in the calculus fuse-after-
  combine is fine, but the implementation forbids the pair, making {fuse epilogue,
  stream-partition K} mutually exclusive states rather than composable moves. The editor
  will inherit such constraint pairs and must surface them as explicit incompatibility
  edges, not silent move-greying.

So: **freeform is not forced by topology** (inverses fix reachability). It is forced by
(a) lemma incompleteness (§1.5) and (b) representation gaps — if a target kernel's structure
is not expressible as any decoration (Q2's named gaps), no move sequence reaches it,
regardless of connectivity.

## 1.7 Q1 verdict

- The move space is **closed relative to a lemma library**: class-(a) moves are always-valid
  by construction; stream-partition and fusion inherit validity from registered lemmas;
  validity itself is three-tier (well-formed / ℝ-valid-rel-L / τ-certified) and must be
  tracked as three separate lights in the UI.
- **Geodesic ruling: the geodesic exists** for eager→flash (and for all four repo flagship
  kernels) — real optimizations do NOT require wrong intermediates, *provided the needed
  lemmas are registered*. The puzzle model is what the same system looks like at a lemma
  gap. Ship both: geodesic as the default interaction (edit history = proof term), freeform
  as the lemma mine, with excursion-mining as the bridge.
- **The endpoint differential is unconditional** — it certifies the compiler and fp behavior
  even on pure-geodesic derivations. "Valid by construction" is a statement about ℝ-algebra
  and about the calculus, never about the artifact.

---

# Q2 — Is the representation complete for WebGPU kernel optimization?

Ground truth about the repo first, because it reframes the question favorably: **none of the
hand-tuned kernels are hand-written WGSL.** All of them (matmul, GEMV, attention fwd + 3-pass
bwd, reductions, Adam, LayerNorm, cross-entropy) are tile-IR programs — block-level builder
callbacks (`KernelContext`/`BlockOps`) lowered by `tile-lowering.ts` → `tile-compiler.ts` to
WGSL. Residency is already **explicit IR data** (`Block.placement: "register"|"shared"`,
`ptrKind: "thread"|"tile"`, smem materialized as `var<workgroup>`), seams are already
data-driven (`applySeam`, `EpilogueConfig` op-chains, `AttnModifierSpec`), and compile-time
schedule parameters vs. runtime uniforms are cleanly separated (structural → shader cache
key; shapes/α/scale/modifier-params → uniform buffer with the TAG_UNIFORM volatile-repack
seam). The gap is not "the schedule isn't reified" — it's that **many schedule decisions are
made algorithmically inside lowering** (residency by pointer-constructor choice, loop order
per dot skeleton, TPR auto-detected at a hardwired 4, vec4 by hardcoded divisibility
criteria, barriers by dependency analysis) rather than read from an editable record.

## 2.1 The lever inventory, classified

Legend — **MOVE**: one of the four napkin gestures. **PARAM**: a parameter of a move.
**DECORATION GAP**: expressible only if the representation grows a named decoration/move the
napkin set lacks. **OUTSIDE**: opaque-kernel escape hatch territory.

| # | Lever | Classification | Repo grounding |
|---|-------|----------------|----------------|
| 1 | Workgroup size/shape | **PARAM** of group-partition (workgroup tier) | `TileKernelSpec.workgroupSize` (`tile-ir.ts:3198`); matmul `tileM/N` via `getWorkgroupSize` (`matmul/types.ts:246`); but reductions hardwire `WG=256` |
| 2 | Shared-memory tiling | **MOVE**: group-partition + recolor→SMEM | `tileM/tileN/tileK` + cooperative `load2D`; capacity check `(tM·tK + tK·tN)·4 ≤ 16KB` (`types.ts:233`) = the napkin's `M_ℓ` constraint, literally |
| 3 | Register blocking (per-thread output tile) | **PARAM** of group-partition at a **register tier** | `threadTileM/N` (`configureTiles`), TM×TN accumulator `ctx.zeros(ttM,ttN)`. Napkin's level-graph admits a register level; the editor's palette must include it |
| 4 | Bank-conflict layout: padding | **DECORATION GAP — "layout decoration"** (intra-level) | `smemPadding` exists as an author-only knob (`tile-ops.ts:848`), default 0, unused by the flagship kernels |
| 5 | Bank-conflict layout: XOR swizzle | **DECORATION GAP — same layout decoration**, absent entirely | zero swizzle machinery in the compiler |
| 6 | Vectorization (vec4 load/compute) | **PARAM** with divisibility annotation (napkin superscripts) | `vectorize/autoVectorize` + `computeSafeVecWidth` (`tile-access-analysis.ts:708`) — an existing, real divisibility/uniformity checker; matmul `vectorWidth ∈ {1,4}`; note the **hardcoded exclusion** `!threadTileM` for smem-vec4 (the benchmarked −17..36% regression) — a policy default that must become a decoration default, not a wall |
| 7 | Loop unrolling | **DECORATION GAP (minor) — "loop-lowering attribute"** on a stream loop's projection | `unroll` flags + auto-unroll thresholds (trip≤4/≤8/≤16, `tile-compiler.ts:799,835,856`). No dataflow meaning; attaches to the pseudocode projection, not the diagram |
| 8 | Software pipelining / double-buffering | **DECORATION GAP — a genuinely new MOVE: `pipeline(loop, depth)`** on the time-projection (reorder independent transfers across iterations; semantics-preserving by construction) | **Absent** from the IR (no ping-pong smem, no prefetch); tried by hand, **−13% on V100** (CLAUDE.md). Must be expressible precisely so it can be measured and lose |
| 9 | Subgroup ops (subgroupAdd/shuffle/broadcast) | **MOVE (recolor) + palette extension: a SUBGROUP residency tier** between register and SMEM | Full IR support (`subgroupShuffleXor/Add/Max/Min/BroadcastFirst/InclusiveAdd`, `tile-ir.ts:1303-1330`); `wgReduce` auto-selects subgroup-then-smem-tree vs pure tree (`tile-ir.ts:2357-2395`); GEMV NT segmented butterfly; TPR=4 **auto-detected, hardwired** (`tile-lowering.ts:151`); subgroup size hardwired 32 (`gpu-context.ts:449`). The residency coloring needs the tier; the current auto-selection becomes the default decoration |
| 10 | Occupancy tradeoffs | **Not a lever — a COST-MODEL dimension.** Constraint side: capacity typing (T0). Objective side: see §2.4 | `validateConfig` thread≤256 + smem≤16KB are the T0 checks; nothing models resident-workgroup count |
| 11 | f16 arithmetic + packing | **Stratum-2 (typed wires)** for anything precision-losing; PARAM for rounding-preserving staging; `pack2x16` = lowering detail | `enableF16`, `inputCastA/B` (load-cast absorption), `smemElemType:"f16"` — the last is precision-losing and currently a schedule knob → must hoist (§1.4) |
| 12 | Uniformity analysis constraints | **T0 typing rule**, not a move — "uniformity typing" the checker must own | Repo patterns the rule must license: tail-thread redundant recompute (`gemv.ts:342`), `tSafe` clamp in cross-entropy, workgroup-uniform `skipTile` guard (`attention-kernel.ts:746-751`); `tile-access-analysis` already tracks thread-invariance |
| 13 | Atomics (i32/u32 only; f32 via CAS) | **OUTSIDE-ish: op-implementation strategy decoration** on scatter-class ops, not a schedule move | `atomicOp/atomicCAS/atomicAddF32` (CAS spin loop, `tile-compiler.ts:990-1045`); used by scatter_add, unscale inf-flag. The napkin has no non-injective-write concept; also the reason "fused single-pass dQ+dKV" is impossible (no f32 atomics — CLAUDE.md) |
| 14 | Indirect dispatch | **OUTSIDE** (escape hatch; becomes relevant for MoE/ragged) | Zero uses in repo; all grids host-computed `GridFn` |
| 15 | Bind-group/binding-layout choices | **OUTSIDE the diagram** (mechanical ABI), but device limits **import as T0 constraints** | Binding order, `uniformBindingIndex`, bind-group caching; `maxStorageBuffersPerShaderStage` bounds `batchedReduction` packing; `maxStorageBufferBindingSize` forces chunking (lever 18) |
| 16 | Uniform vs storage buffers; scalar volatility | **DECORATION GAP — "volatility typing"** on config values (per-step-varying ⇒ data tensor or volatile uniform) | The frozen-`step_size` disease and its fixes: TAG_UNIFORM volatile repack (`tile-dispatch.ts:230-280`), staleness guard invalidation, and Adam's endgame — t/lr as 1-element **storage tensors** + in-kernel expm1 bias correction so the config is fully static (`adam-kernel.ts:51-56, 238`). The editor must show volatility as a first-class tag or it will reintroduce the disease |
| 17 | Texture paths | **OUTSIDE / irrelevant** | Zero texture usage; storage-buffer-only backend |
| 18 | K-split / split-reduction | **MOVE**: stream-partition of the reduction axis across workgroup.z + recolor carried state to a GMEM partials buffer + derived combine pass; **lemma-gated** (fp-tolerant assoc/comm reorder) | `kSplit` two-pass: raw `[P,M,N]` partials, no atomics, separate reduction kernel (`tile-matmul.ts:261-440`); engagement policy `computeKSplitFactor` (`dispatch.ts:749`); GEMV NN reuses the same combine kernel |
| 19 | Chunking for binding-size limits | **MOVE**: group-partition at a **"bindable-window" pseudo-level** whose capacity = `maxStorageBufferBindingSize`, alignment = 256B divisibility | `planChunkedFullReduction` (`reductions.ts:494`) single-sources geometry for execute + stream-generate; sub-range bindings. Notably the per-chunk kernel is `workgroupSize:1` serial (`reduction-tile-ir.ts:489`) — a **valid, grossly suboptimal** state: the space expressing losers, in production |
| 20 | Grid traversal order (swapGrid, rasterization, L2 reuse) | **DECORATION GAP — "traversal-order decoration"** on group-partition (which group index maps to which programId, in what order) | `swapGrid` (`types.ts:39`, decided at `dispatch.ts:1068`), `splitWorkgroups2d`/`rowGrid2d` for the 65535 cap. Napkin's cost model has no reuse-by-order term |
| 21 | Epilogue fusion / seams / masks | **MOVE (fuse/weave)** — already data-driven | `EpilogueConfig` op-chain spliced post-accumulate (`tile-matmul.ts:51-101`); attention `attn_score`/`attn_mask`/`attn_dscore` seams with modifier-as-data, structure in cache keys (`attnModifierKey`); causality folded into the modifier algebra |
| 22 | Affine-mask → loop-bound (causal tile skip) | **Lemma class the napkin lacks: "predication/sparsity-to-bounds"** | dKV backward skips whole Q-tiles above the diagonal via workgroup-uniform `skipTile` (`attention-kernel.ts:746-751,818`) — a structural rewrite from mask affine-ness to iteration bounds |
| 23 | In-place / aliasing (in-place m/v, smem slot reuse) | **DECORATION GAP — "aliasing decoration"** (destination-aliasing + intra-level slot aliasing). Napkin is pure-functional | Adam updates param/m/v **in-place** (`adam-kernel.ts:296-298, 501-510`); attention V reuses K's smem slot (`{reuseShared: K}`, `:467`). Aliasing is also the repo's #1 silent-bug source (UAF/pool ledger) — it must be *typed*, not implicit |
| 24 | Horizontal packing / multi-tensor batching | **DECORATION GAP — a new MOVE: `pack(op-group)`** (horizontal fusion of independent same-shape ops with physical concat/scatter-back) | `dispatchPackedOptimizer` (`packed-dispatch.ts:206`): scatter→one Adam dispatch over `N·numElements`→gather; `batchedReduction` bounded by binding limits; also perf target #3 (bias-grad sums). Weave doesn't cover it: the batch axis doesn't exist in the data — the move materializes it |
| 25 | Kernel-family selection (GEMV vs tiled; sequential-vs-parallel reduction) | **Macro-move / derived variant.** In the calculus GEMV = tiled with M-partition collapsed to 1 + reduction recolored to subgroup lanes — reachable by moves; the repo implements it as a separate registered family (`variants.ts:117-163`) with its own tunables (`wgSize`, `rowsPerWg`) | Editor choice: either derive families by moves (honest, harder) or expose family-selection as a macro-move with the derivation attached (pragmatic) |

**Tally**: 8 pure move/param levers, 8 named decoration/move gaps (layout, traversal order,
pipelining, aliasing, packing, volatility, uniformity-typing, loop-lowering attrs — plus the
subgroup tier as a palette extension and predication as a lemma class), 4 genuinely outside
(atomics-strategy, indirect dispatch, textures, bind-group ABI).

## 2.2 Four-kernel reachability audit (the acceptance test)

**A. Tiled matmul with K-split — REACHABLE** (2 minor decorations wanting).
matmul semantic node → group-partition M,N at workgroup tier (`tileM/N`) → group-partition at
register tier (`threadTileM/N`) → recolor A,B tiles → SMEM (cooperative load) → stream-
partition K within-workgroup (`forRange` + `dotAccum`) → stream-partition K **across**
workgroups (`kSplit`, lemma: fp-tolerant reorder) with carried state recolored to GMEM
partials `[P,M,N]` → derived combine pass → fuse epilogue chain (bias/unary/binary/cast) →
vec4 as divisibility-annotated parameter. **Not expressible without gaps**: `swapGrid`
(traversal-order decoration, #20); the epilogue⊥kSplit implementation constraint (§1.6) is an
incompatibility edge, not a calculus fact. Transpose-as-stride-flip (`TransposeMode` +
`detectSimpleTranspose`) needs strides-as-layout on wires — part of the layout decoration.

**B. Flash attention skeleton (fwd + 3-pass bwd) — REACHABLE given 2 lemmas** (+1 aliasing
decoration). Forward derivation is §1.2 verbatim; block sizes BR/BC are group/stream
parameters; `headDim % 4` is a napkin divisibility superscript; the modifier seams are
fuse/weave of elementwise into the streamed body (already data-driven). Backward: D-precompute
= fuse(rowdot) + recolor D→GMEM (materialization between passes); dQ streams KV while dKV
streams Q — the same graph with the group/stream roles of the two axes **swapped**, a clean
pair of derivations from one semantic backward. **Lemmas required**: online-softmax
(registered, #b); affine-mask→loop-bound for the causal tile skip (#22 — new lemma class).
**Decoration required**: `reuseShared` smem aliasing (#23). Un-derivable residue: none.

**C. Chunked full reduction — REACHABLE, fully.** Group-partition the input axis at the
bindable-window pseudo-level (capacity = maxStorageBufferBindingSize, 256B-alignment
divisibility — `planChunkedFullReduction` IS the derived plan, single-sourced for execution
and stream-generation) → stream-partition within each chunk (here: degenerate serial loop,
`workgroupSize:1`) → combine pass over partials. Doubly useful as the editor's teaching
example: the current shipped kernel occupies a *deliberately dumb* point in the schedule
space (one thread per chunk), i.e. the calculus must and does express states that are valid
and slow — and a user could improve it by pure moves (repartition chunk-interior onto a
256-thread wgReduce) without any new vocabulary.

**D. Fused Adam — PARTIALLY REACHABLE; 2 named gaps.** Reachable part: fuse the elementwise
update chain (weave over the element axis) ✓; chunked sub-range dispatch = bindable-window
group-partition ✓ (mixed chunked/scalar binding modes, `adam-kernel.ts:472-495`); vec4 ✓;
scalars-as-data (t/lr storage tensors) + in-kernel expm1 bias correction = a registered
numerics lemma + volatility typing ✓ conceptually. **Gaps**: (1) **in-place m/v/param
updates** — destination aliasing has no napkin expression (the calculus is pure); without the
aliasing decoration the editor literally cannot draw the shipped kernel (#23). (2) **packed
multi-param dispatch** — scatter N params into one contiguous buffer, one dispatch, gather
back: horizontal `pack` move with physical re-layout (#24); weave cannot reach it because the
packed axis doesn't exist in the semantic graph. The unscale variant's atomic inf-flag is
escape-hatch atomics (#13, small and containable).

**Audit conclusion**: 3 of 4 reachable (A and B modulo minor decorations), 1 partial. No
kernel requires passing through a functionally wrong state — corroborating Q1. Every
unreachable feature is a *named decoration or move*, not an amorphous hole.

## 2.3 Named representation gaps, ranked by importance

1. **Residency palette extension: SUBGROUP and REGISTER tiers** (below SMEM), plus the
   bindable-window pseudo-level above GMEM. Needed by: matmul register blocking, GEMV NT,
   every wgReduce, TPR; chunking. Cheap (the napkin level-graph is already abstract);
   unlocks the most levers per unit of new vocabulary.
2. **Aliasing decoration** (in-place destination aliasing + intra-level slot reuse). Needed
   by: fused Adam (unreachable without it), attention smem reuse, copy_/velocity updates.
   Doubly justified: aliasing is the repo's dominant silent-corruption class — typing it in
   the editor turns the #1 bug source into a checked annotation.
3. **`pack` move (horizontal fusion with physical re-layout)**. Needed by: packed Adam,
   batchedReduction, the bias-grad-sum perf target. Without it every optimizer kernel and
   every "batch the small ops" optimization is outside the calculus.
4. **Layout decorations** (smem padding/swizzle; strides/transpose-mode on wires; grid
   traversal order incl. swapGrid). Intra-level layout is the napkin's largest blind spot;
   the repo touches all three (padding knob, TransposeMode stride flips, swapGrid).
5. **`pipeline(loop, depth)` move** (double-buffering/software pipelining as a time-
   projection rewrite). Absent from the IR today and a *known loser on V100* — which is the
   point: it must be expressible to be measured (§2.4), and it will win on other hardware.
6. **Predication / affine-mask→loop-bounds lemma class** (causal skip; future block-sparse).
7. **Volatility typing** on scalars/uniforms (static / per-dispatch / per-step-volatile) —
   the frozen-scalar disease as a type error instead of a runtime staleness guard.
8. **Uniformity typing** (barrier-uniformity legality with the licensed idioms: tail
   recompute, clamp-and-mask, workgroup-uniform guards).
9. **Loop-lowering attributes** (unroll) — minor, projection-level.
10. **Escape hatches** (kept, honestly): atomics implementation strategies, indirect
    dispatch, hand-WGSL opaque kernels — the ~20% "schedule shadow" bin, admitted with a
    differential contract and no derivation claim.

## 2.4 The losers, and what they say about the cost model

CLAUDE.md's "what didn't work" contains **valid schedules that lost**: vec4 shared-memory
K-loop (−9..36%), double-buffered K-loop (−13%, occupancy loss from 2× smem), per-shape
autotuning on distilgpt2 (no headroom). All three are transfer-neutral or transfer-favorable
— the napkin's `H = Σ(color-changing wire sizes)` and `M ≤ capacity` model **predicts none of
these losses**, because they're decided by occupancy, register pressure, barrier latency, and
instruction-selection effects one level below transfer counting. And the winners flip across
hardware (V100's 10-storage-buffer limit vs A100; subgroup size assumptions; the fact that
the repo hardwires subgroup size 32 and TPR 4 is itself a device-family bet).

Consequences for the editor:

- **The napkin cost model is the explanatory overlay, not the objective.** Keep the
  color-changing-wire readout as the *why* display (it's still the best available intuition
  pump and it correctly rank-orders the big moves — tiling, streaming, fusion). But the
  in-tab loop's third step, *profile*, is load-bearing: measured medians are the ground
  truth (`tile-autotune.ts` already implements the right protocol — warmup 3, timed 5,
  median, min-over-configs, silently skip failing configs).
- **Decorations must be device-keyed.** A schedule that wins on the desktop GPU in the tab
  loses on V100 and vice versa; the repo's caches already key by shape-class + dtype +
  epilogue and persist tuned winners per exact shape. The editor's saved schedules are
  (semantic-graph, decoration, device-class) triples, with the analytic model as the
  device-independent prior and the profile as the device-specific posterior.
- **Express-to-measure is a design requirement**: a move the cost model dislikes must still
  be applicable (greyed-out-with-prediction, not hidden), or the editor can never generate
  the evidence that retires or revives a "didn't work" entry on new hardware. The A100-era
  reversal of the V100 fusion verdict (CLAUDE.md target #5) is the in-repo proof that these
  entries expire.

## 2.5 Q2 verdict

The napkin move-set is **complete for the macro-structure** of every kernel the repo ships —
tiling, streaming, residency, fusion, splitting, chunking all land as moves/parameters, and
the four-kernel audit finds no wrong-state requirement and no amorphous residue. It is
**incomplete at two specific altitudes**: *intra-level layout* (padding/swizzle/traversal
order/strides) and *impure structure* (aliasing, in-place, horizontal packing), plus two
typing disciplines (volatility, uniformity) and one new time-axis move (pipelining). All
gaps are nameable, finite, and small relative to the calculus — the representation should be
extended, not abandoned. The cost model, by contrast, cannot be rescued analytically at the
margins that decide real wins on WebGPU-class hardware: measurement stays in the loop.

---

# What this changes about the editor design

1. **The core artifact is the derivation, not the state.** Edit history = proof term:
   sequence of (move, params, lemma-citations), invertible, with persistent node identity.
   Three validity lights per state: typechecks (T0) / ℝ-valid rel. lemma library (T1) /
   τ-certified (T2). T2 runs even on pure-geodesic derivations — it certifies the compiler.
2. **Dual mode, coupled by lemma-mining.** Geodesic mode offers only legal moves (never
   wrong, possibly slow). Freeform mode allows arbitrary decoration/term edits and certifies
   endpoints differentially. A green freeform excursion is decomposed against known moves;
   the residue is offered for registration as a named lemma (with its differential contract
   attached as the admission evidence). This is how the manifold grows — geodesic-only
   freezes the calculus, freeform-only learns nothing.
3. **Grow the decoration vocabulary before building UI**: residency palette {GMEM ·
   bindable-window · SMEM · SUBGROUP · REGISTER}; aliasing decorations; layout decorations
   (pad/swizzle/traversal-order/strides); volatility tags; uniformity typing in the T0
   checker. Add two moves: `pipeline(loop, depth)` and `pack(op-group)`. Admit atomics
   strategies, indirect dispatch, and hand-WGSL as contracted opaque shadows.
4. **Police the stratum-2/3 boundary in the type system**: wire dtypes (incl. accumulator
   and staging precision — hoist `smemElemType`), save/recompute contracts, and approximation
   choices live on the semantic graph; stratum-3 edits may change only association/rounding.
   This is what makes "every stratum-3 state valid" true rather than aspirational.
5. **Engineering path through the repo**: the tile-IR already reifies placement, seams, and
   compile-vs-runtime parameters as data. The work is to **lift the algorithmic schedule
   decisions in lowering (residency-by-pointer-kind, TPR=4, vec4 divisibility policy, loop
   order, WG=256, reduction parallel-threshold >64) into a ScheduleRecord that lowering
   reads**, with today's heuristics becoming the default decoration values. Kernels become
   (semantic spec, ScheduleRecord) pairs; the editor edits the record; `tile-autotune`'s
   factory/params machinery is already the right shape for the measurement loop.
6. **Cost display vs cost truth**: render napkin H/M (color-changing wires, capacity bars,
   the H* = Σα·M^(−β) curve) as the live explanatory overlay; treat the in-tab profiler as
   the objective; key every saved schedule by device class. Never hide a legal move because
   the model dislikes it — the "what didn't work" list is hardware-relative and the editor
   is the instrument for re-litigating it.
