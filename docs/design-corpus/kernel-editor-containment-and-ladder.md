# WGSL ⊂ CUDA? Containment, Saddle Points, and Frontier Completeness

Analysis of three questions about the geodesic kernel calculus (schedule edits as gestures from
{group-partition, stream-partition, recolor, fuse/weave} + named gaps {pack, pipeline, layout,
subgroup/register tiers}, lemma-gated, endpoint-differential-certified — see
`arxiv-2412-03317-analysis.md` and `kernel-editor-validity-completeness.md`):

1. Does the WGSL schedule move-space **embed** in the CUDA one, at the state / path / optimum level?
2. Taking the net2net/saddle-manifold analogy seriously: is a browser-found optimum a **trap or a detour**
   when the CUDA optimum needs macro-structure WGSL couldn't express a preference for?
3. Independent of porting: can CUDA-frontier performance be **completely and efficiently expressed** in a
   geodesic calculus at all — and what fraction honestly lives in moves+decorations vs opaque bindings vs
   below any portable calculus?

Grounding: repo tile-IR (`src/backend/webgpu/tile-{ir,ops,lowering,compiler,dispatch}.ts` — residency
already reified as `Block.placement: "register"|"shared"`, `ptrKind: "thread"|"tile"`; subgroup shuffle/
reduce IR nodes shipped; subgroup size hardwired 32; task #72 = adopt `subgroup_id`/`subgroup_uniformity`/
`linear_indexing` behind feature detection). Web research (mid-2026): Chrome subgroups shipped stable 134,
`subgroup_id` 144, `subgroup_uniformity` 145; `chromium-experimental-subgroup-matrix` Dawn-prototype-only;
Hopper WGMMA/TMA/cluster and Blackwell tcgen05/TMEM specifics via PTX ISA + Colfax tutorials; CuTe layout
algebra via Jay Shah's note + arXiv 2601.05972 (categorical foundations) + arXiv 2603.02298 (Cecka);
Triton limits via its own Gluon docs + arXiv 2604.23466 (cuTile eval) + arXiv 2505.23819 (Linear Layouts);
SASS floor via maxas, DeepGEMM, CuAsmRL (arXiv 2501.08071). Sources cited inline.

---

# PART 1 — The containment/embedding question

## 1.1 Formalization

A schedule state is `(G, D)`: semantic graph `G` (fixed) + decoration `D` (residency coloring over a tier
lattice, partition labels, fusion grouping, layout decorations, aliasing/volatility tags). A backend `B`
defines:

- a **tier lattice** `L_B` (memory levels with capacities),
- a **decoration vocabulary** `Dec_B`,
- a **move alphabet** `M_B` (partial functions on decorations),
- a **legality theory** `T_B` (T0 typing rules: capacities, divisibility, uniformity, forward-progress).

**Definition (embedding).** WGSL embeds in CUDA iff there is a map `ι` on states such that (i) `ι` is
induced by a monotone injection `h: L_WGSL → L_CUDA` of tier lattices and an inclusion
`Dec_WGSL ↪ Dec_CUDA`; (ii) `ι` preserves T0/T1 validity; (iii) `ι` is a **simulation** of labeled
transition systems: every WGSL move `s →_m s′` maps to a CUDA move `ι(s) →_{ι(m)} ι(s′)` of the same KIND.

Three separate questions, in increasing strength:

- **State-level**: is every valid WGSL schedule state a valid CUDA schedule state? (ι exists, preserves validity)
- **Path-level**: do geodesics extend? (ι is a simulation; browser derivations are prefixes of CUDA derivations)
- **Optimum-level**: is `opt_WGSL(G)` on the path to `opt_CUDA(G)`? (does ι map argmins near argmins)

These come apart. The punchline: **YES, YES-as-paths-not-as-geodesics, NO-in-general** — and the NO is
benign (Part 2).

## 1.2 The tier lattice, built from the evidence

WGSL residency chain (mid-2026, all confirmed):

```
global (storage buffer)
  └─ [bindable-window]      pseudo-tier: maxStorageBufferBindingSize (128MB) chunking — a WGSL-ONLY artifact
  └─ workgroup smem          16KB default limit (maxComputeWorkgroupStorageSize; adapters often 32KB+)
       └─ subgroup           SHIPPED: Chrome 134 stable; shuffle/reduce/scan/ballot; subgroup_id (144),
       │                     subgroup_uniformity (145); size VARIABLE (4–128, not compile-time constant)
       └─ thread registers   implicit — no allocation control whatsoever
            └─ [subgroup-matrix fragments]   EXPERIMENTAL (Dawn flag only, no origin trial, no ship date):
                              opaque register-resident fragments; load/store from storage/workgroup memory;
                              the ONLY layout lever is const row/col-major at the load/store boundary;
                              shapes/dtypes capability-queried, not spec-fixed. (gpuweb#4195; crbug 348702031)
```

CUDA chain (Hopper/Blackwell, from PTX ISA + Colfax):

```
global
  └─ L2 persistence window   HOST-side decoration (cudaAccessPolicyWindow on stream/graph node; hitRatio)
  └─ cluster / DSMEM         Hopper TIER: smem address space distributed over ≤8 (portable; 16 opt-in)
  │                          co-resident CTAs; mapa/cluster.sync; SM-to-SM network
  └─ smem                    up to 227KB/SM (H100), explicitly sized per kernel
       └─ TMEM               Blackwell TIER: 256KB/SM tensor memory; explicit alloc (tcgen05.alloc,
       │                     power-of-2 columns ≥32); WARP-PARTITIONED access (warp i ↔ lanes 32i..32i+31);
       │                     accumulator MUST live here for tcgen05 MMA
       └─ warp               fixed 32; shuffle/vote; mma.sync fragments (FIXED per-thread register layout)
       └─ registers          ≤255/thread; setmaxnreg = runtime re-budgeting BETWEEN warpgroups (24–256, ×8)
            └─ wgmma/tcgen05 fragments + TMA descriptors as layout-bearing objects
```

`h` maps: global→global, workgroup→smem, subgroup→warp, registers→registers. It is monotone and injective.
Three CUDA tiers have **no WGSL preimage** (L2-window, DSMEM, TMEM) — they are the "new dimensions" of
Part 2. One WGSL pseudo-tier has **no CUDA image**: the bindable-window. It maps to the trivial decoration
(CUDA has raw pointers, no 128MB binding cap), i.e. `ι` **quotients away vestigial structure**. This is
the first crack in "lossless": the port is lossless on semantics and on every load-bearing decoration, but
some WGSL decorations were workarounds for WGSL-only constraints and arrive as no-ops. (Repo example: the
whole `planChunkedFullReduction` bindable-window derivation collapses to "just index the buffer.")

## 1.3 The lever table: every CUDA-only lever classified

Classification target: new **TIER** / new **DECORATION** / new **MOVE kind** / **OUTSIDE** the kernel
calculus (host/program stratum). Cross-referenced with the WGSL gap-list already named in
`kernel-editor-validity-completeness.md` §2.3 (which named: pack, pipeline, layout, aliasing, volatility,
uniformity, subgroup/register tiers — the question is whether CUDA needs *more kinds* than those).

| CUDA lever | Classification | Detail & consequence |
|---|---|---|
| **mma.sync fragments** (Ampere) | DECORATION (fragment layout) | Fixed per-thread register distribution per atom shape (m16n8k16); `ldmatrix` exists solely to permute smem→reg into it. A *rigid* value of the layout decoration, not a new kind. |
| **wgmma.mma_async** (Hopper) | DECORATION + hard T0 constraints that back-propagate into move parameters | Pins tile-M=64 per atom; N ∈ 8·{1..32}; K·dtype-size = 32B; **B must be in smem**; A smem-or-reg; accumulator in registers; smem operands must use one of **4 swizzle atoms** (none/32/64/128B = CuTe `Swizzle<0..3,4,3>`) over 8×16B "core matrices"; **K-major mandatory for sub-16-bit dtypes**. In napkin terms: new *divisibility superscripts* + a swizzle-valued layout decoration. No new move kind — but the constraint set is narrow enough to dictate tile shapes upstream (the load-bearing fact for Part 2). |
| **tcgen05 / TMEM** (Blackwell) | new **TIER** + one new MOVE | TMEM is a genuine fifth tier (allocator, column-quantized, warp-partitioned epilogue reads; accumulator must live there; "UMMA requires no registers for data"). The CTA-pair 2-SM MMA is a new move (compute atom spanning two SMs). Epilogue structure is forced: full warpgroup to read an accumulator. |
| **cp.async / TMA + mbarrier** | the already-named **`pipeline(loop, depth)` MOVE** made real, + a DECORATION | Async global→smem engines; TMA is descriptor-driven (TensorMap: shape/strides/**swizzle**/im2col — created host-side, `__grid_constant__`). The mbarrier `expect_tx(bytes)`/`complete_tx` protocol is the *implementation* of pipeline stages; the schedule-level object is exactly the pipeline move the WGSL calculus already reserved (gap #8, tried-and-lost −13% on V100). TMA **multicast** (cluster bitmask, one L2 transaction → many CTAs' smem) is a small new move riding on the cluster tier. |
| **Warp specialization** (producer/consumer; CUTLASS cooperative vs pingpong) | genuinely **NEW MOVE KIND: `role-partition`** | Partitions the *executor* (warps), not a data axis — nothing in {group, stream, recolor, fuse, weave, pack, pipeline} expresses "these 128 threads only issue TMA; those 256 only issue MMA; they alternate roles per iteration (pingpong)." It is a scheduling of the compute resource itself. Companion move: **`setmaxnreg` register re-budgeting** between roles (a resource-transfer move on the register tier). |
| **Thread-block clusters / DSMEM** | new **TIER** (DSMEM) + moves (cluster.sync, mapa, multicast) + HOST launch decoration (cluster dims) | Cross-CTA smem addressing with hardware co-residency guarantee. WGSL has nothing (cross-workgroup communication is "the wild west" — no forward-progress guarantee; confirmed structurally absent, not just unshipped). |
| **L2 persistence** (`cudaAccessPolicyWindow`) | OUTSIDE the kernel calculus; a **recolor at a host-visible tier** | Stream/graph-node attribute, not expressible in-kernel. In the editor's terms: a residency decoration that lives at the *program* stratum (torchlette's compiled-plan level), not the kernel stratum. |
| **Persistent kernels / stream-K** | MOVE **already in the calculus** (grid re-partition + the repo's own kSplit lemma) + a **legality predicate WGSL lacks** | Stream-K = stream-partition of K across CTAs with turnstile/atomic combine — the same fp-tolerant-reorder lemma as the repo's `kSplit`. The CUDA-only part is the T0 side-condition: co-residency/forward-progress (grid ≈ #SMs, cooperative launch), which WebGPU cannot guarantee at all. So: not a new move — a new *theory axiom* in `T_CUDA` that makes an existing move legal at a scope WGSL forbids. |
| **f32/f64 atomics** | expanded op-implementation strategy (the calculus already classes atomics OUTSIDE-ish) | Directly flips repo entries: "fused single-pass dQ+dKV — impossible (no f32 atomics)" becomes possible; WGSL's CAS-loop emulation (unsound without forward progress) becomes native `atomicAdd(float*)`. |
| **f64** | stratum-2 (wire dtype), not a schedule lever | — |
| **Dynamic parallelism** | OUTSIDE (device-side host stratum) | — |
| **Streams / CUDA graphs** | OUTSIDE — program stratum | CUDA graphs ≈ torchlette's compiled-plan replay, almost exactly (record/replay a DAG, volatile params as data). The calculus already has this concept one level up. |
| **`__launch_bounds__` / maxrregcount** | occupancy DECORATION (compile-time) | Interacts (and can conflict) with setmaxnreg — an incompatibility edge of the same species as the repo's epilogue⊥kSplit. |
| **Named barriers** (16 ids/CTA) | MOVE-supporting primitive | The mechanism under role-partition; not user-facing vocabulary. |

**Tally.** Against the WGSL calculus's already-named gap list, CUDA adds: **2 new tiers** (TMEM, DSMEM —
plus L2-window at the program stratum), **1 genuinely new move kind** (`role-partition` + its
register-rebudget companion), **richer values for existing decorations** (swizzle atoms, TensorMap,
fragment layouts — all layout-decoration values; WGMMA's shape quantization = divisibility superscripts),
**1 new legality axiom** (co-residency/forward progress, enabling persistent/stream-K scope), and **wider
parameter ranges everywhere** (1024-thread CTAs vs 256; 227KB smem vs 16-32KB; native f32 atomics). Nothing
in the CUDA kernel repertoire requires a move kind outside {group, stream, recolor, fuse, weave, pack,
pipeline, layout-decorate, role-partition}. That the entire Hopper/Blackwell lever inventory folds into
the existing taxonomy plus ONE new move kind and TWO new tiers is the strongest available evidence that
the calculus's ontology is right.

## 1.4 State-level containment: YES (embedding-up-to-quotient)

For every T0/T1-valid WGSL state `s`, `ι(s)` is T0/T1-valid CUDA:

- **Capacities/ranges**: every WGSL limit is strictly inside the CUDA envelope (256 ≤ 1024 threads; 16-32KB
  ≤ 227KB smem; binding-size cap → none). T0 constraints can only *relax* under ι.
- **Uniformity**: WGSL's uniformity analysis is *stricter* than CUDA's requirements (and `subgroup_uniformity`
  only relaxes it toward CUDA's). WGSL-uniform ⇒ CUDA-legal barriers.
- **Subgroup size**: a WGSL state must be valid for all subgroup sizes the device might report (the repo
  hardwires 32 — itself a device-family bet); ι *specializes* to warp=32. Specialization of a
  universally-quantified state is lossless in this direction.
- **Semantics**: same IEEE-754 f32/f16 value semantics at T1 (ℝ-level identical by construction; fp
  differences — contraction latitude, denormals — are T2 matters, and T2 was always device-keyed:
  certification never ports, it re-runs. This was already the design's rule.)
- **The quotient**: bindable-window decorations, CAS-emulated atomics, and binding-ABI structure map to
  trivial/no-op decorations. State containment holds; *fullness* fails — ι(S_WGSL) is a measure-zero-ish
  slice of S_CUDA sitting inside the "no async, no roles, no clusters, neutral swizzle" subspace.

**Verdict: every browser schedule state is a valid CUDA schedule state.** WGSL is, in this precise sense,
a simplification of CUDA: same move kinds, restricted tiers, restricted decoration values, narrower
parameter ranges, weaker legality theory (no forward progress). The one WGSL-only structure (bindable
windows) quotients to nothing.

## 1.5 Path-level: YES as valid paths — ι is a simulation, not an isometry

Since every move kind and every in-range parameter maps, a WGSL derivation `d = m₁…m_k` maps move-by-move
to a CUDA derivation `ι(d)` through valid states: **every browser geodesic is a prefix of CUDA
derivations**, and the user can keep appending CUDA moves (recolor into TMEM, insert role-partition,
decorate swizzles, deepen pipeline) from the ported state. Formally ι is a simulation of transition
systems.

What it is NOT: an isometry of objective landscapes. "Geodesic" in the design means *path through valid
states*, and those extend perfectly. But the *optimal* path in the CUDA landscape generally does not pass
through `ι(opt_WGSL)` — because the objective changed (new tiers and atoms bend the landscape), not
because any state or move became illegal. Also, practically, continuation usually *begins with reversals*:
parameters chosen under WGSL's 256-thread/16KB constraints are immediately re-tuned (un-apply + re-apply
with new g — still inside the calculus, cost ≈ zero). Path containment is exactly as strong as it should
be: you never leave the manifold, and you never need to.

## 1.6 Optimum-level: NO in general — optima sit on different branches, with a deep shared ancestor

The mechanism, concretely, for matmul-class kernels:

- **Browser optimum** (repo's own, RTX/V100/A100-class via Dawn): 2D group-partition M,N at workgroup tier
  + register-tier threadTile 4×4 + recolor A,B→smem (scalar K-loop — vec4-smem *lost* 9-36%) + stream K
  (+ kSplit across workgroups when profitable) + fused epilogue; **no pipelining** (−13% on V100), no role
  structure, accumulate in thread registers.
- **CUDA H100 optimum** (CUTLASS/FA3-class): WGMMA atoms pin tile-M=64/warpgroup and quantize N,K; smem
  tiles in SW128 swizzle; TMA multi-stage pipeline (mbarrier expect_tx); **role-partition** into 1 producer
  + 2 consumer warpgroups (pingpong or cooperative); setmaxnreg 24/240 split; persistent grid + stream-K
  tail; epilogue restructured around warpgroup-wide accumulator reads (on Blackwell: accumulator in TMEM).

The browser optimum could not have *expressed a preference* for any of the bolded structure. Its tile
shapes were tuned against a landscape without WGMMA quantization; its loop structure has no role axis; its
smem layout has no swizzle dimension. `ι(opt_WGSL)` lands in CUDA space as a **valid, mediocre** state —
roughly "a good Ampere-era CUDA-core kernel." The CUDA optimum is on a different branch of the derivation
tree: the branches split where role-partition and the WGMMA atom substitution enter, which is *above* most
parameter decorations in the derivation.

But note where the branches share an ancestor — and how deep it is: *tiled, K-streamed, smem-staged,
epilogue-fused, split-K-capable matmul with unbound parameters*. Everything algorithmic and every lemma
(fp-tolerant K-reorder; for attention: online-softmax, the whole flash streaming skeleton) is in the
shared ancestor. Quantified evidence that the shared ancestor carries most of the value:

- Triton — which expresses ONLY the shared-ancestor layer plus scalar tuning (compiler chooses layouts;
  no explicit WS until 3.2, auto even then) — reaches **98-101% of cuBLAS on H100 GEMM** and **~98% of
  FlashAttention-2** on H100 (arXiv 2604.23466). Macro-structure + parameter search ≈ 98% on *mature*
  hardware.
- The same Triton collapses to **62% of cuBLAS on Blackwell** (ibid.), and FA3 is **1.5-2× over FA2**
  (and 1.5× over Triton's attention) on H100 — the gap being precisely {WS roles, TMA pipelines, TMEM,
  swizzle-bound atoms}: the not-yet-expressible-from-the-browser branch. New-hardware frontiers are where
  optimum-level containment fails hardest.

**Three-level verdict.** State: **contained** (ι exists, validity-preserving, vestigial WGSL structure
quotients away). Path: **contained as valid paths** (simulation; browser derivations extend without
leaving the manifold), not as optimal paths. Optimum: **not contained** — the browser optimum is generally
NOT a prefix of the CUDA optimum; they share a deep derivation ancestor (the full algorithmic layer + the
macro-schedule skeleton) and diverge at atom substitution, role-partition, and tier recoloring.

## 1.7 Does chromium-experimental-subgroup-matrix change the lattice's trajectory?

Yes, in a specific and instructive way. Status (verified): Dawn prototype behind
`chromium-experimental-subgroup-matrix`, design "done" at WGSL level (gpuweb#4195), **no origin trial, no
ship date**; opaque `subgroup_matrix_left/right/result` fragments; load/store from storage/workgroup
memory; layout lever = const row/col-major only; shapes/dtypes **capability-queried** (mirroring Metal
`simdgroup_matrix`, Vulkan `VK_KHR_cooperative_matrix`, HLSL SM6.8). Used off-browser already (Dawn-native
LLM inference, arXiv 2605.20706).

What it changes: WGSL is re-walking CUDA's own tensor-core history — it is at the **Volta-wmma stage**
(opaque fragments, no layout visibility, no async feed). The fragment DECORATION enters the WGSL
vocabulary now; its layout *rigidity* (the thing that makes CUDA optima branch-divergent) arrives with it
in embryonic form: capability-queried shape sets are exactly arch-gated atom sets. So the editor's
representation needs **capability-quantified atom families** *now* — an atom whose shape/dtype set is a
runtime query is the same schema object whether it resolves to `simdgroup_matrix` 8×8×8, mma.sync
m16n8k16, or wgmma m64nNk16. What does NOT change: the async-movement / cluster / TMEM axes have **zero**
WGSL proposals (confirmed: no cp.async analogue, no clusters, no L2 control, persistent kernels unsound —
permanent structural absences, not roadmap items). The lattice delta shrinks on the fragment axis only.
Trajectory conclusion: the browser space converges toward "Ampere-minus-async" and stops; the
Hopper/Blackwell branch (roles, async engines, new tiers) stays CUDA-only for the foreseeable horizon.

---

# PART 2 — The saddle-point analogy, taken seriously

## 2.1 The precise mapping

Net2net / network morphisms (Chen et al. 2015; Wei et al. 2016): a function-preserving map embeds a small
network's parameters into a larger architecture (new units initialized so the big net computes the SAME
function), then training continues. The embedded point sits on a high-symmetry manifold of the larger loss
landscape — typically a plateau/saddle (duplicated-unit permutation symmetry; cf. saddle-to-saddle
dynamics in deep networks), and gradient flow must break symmetry to descend further.

The dictionary, term by term:

| Saddle/morphism world | Schedule calculus world |
|---|---|
| Function computed by the net | Stratum-2 semantics `G` (preserved exactly — T1) |
| Parameter space of the small net | WGSL decoration space `D_WGSL` |
| Parameter space of the large net | CUDA decoration space `D_CUDA` |
| Function-preserving morphism (Net2WiderNet identity-init) | `ι`: new coordinates set to **neutral values** — pipeline depth 1, roles ∅, cluster 1, swizzle none, TMEM unused, L2 window unset |
| Loss | Measured runtime on the target device (device-keyed T2 landscape) |
| Gradient descent | Geodesic move sequence, profiler-guided |
| Symmetry-induced flat directions at the embedded point | Neutral decorations whose *single-move* perturbations don't improve (or regress) |
| Symmetry breaking along a coupled direction | **Bundled multi-moves** (atom-swap + swizzle + repartition + pipeline + role-partition together) |

The analogy is unusually exact because ι really is a function-preserving embedding into a
higher-dimensional space with the new coordinates at identity — that is *literally* Net2Net's
construction, transplanted from parameter space to schedule space.

## 2.2 The embedded point IS saddle-like — and the repo already measured it

The load-bearing empirical fact: at `ι(opt_WGSL)`, several single new-coordinate moves are
**non-improving or harmful in isolation but essential in bundles**:

- Pipelining alone: repo measured **−13%** (double-buffered K-loop, V100 — occupancy loss beat barrier
  savings). Pipelining inside the TMA+WS+swizzle bundle on H100: indispensable (every CUTLASS sm90
  mainloop is a multi-stage async pipeline).
- Vec4-smem alone: −9..36% (repo). Vectorized access patterns as dictated by WGMMA core matrices: mandatory.
- Warp specialization alone (no async producer worth specializing): pure overhead. With TMA: +10-15% even
  when *automatically* applied to Triton kernels (PyTorch WS blog), and structurally required for FA3-class
  overlap.

That is the discrete signature of a saddle: the objective is flat-or-adverse along each new axis
individually and descends steeply along a correlated combination. Consequence for the editor: greedy
single-move hill-climbing from a ported state **stalls at the embedding point** — the interface must offer
**macro-moves** (named bundled derivations ≈ CUTLASS kernel schedules: "TmaWarpSpecializedPingpong" is
exactly a saddle-escape direction packaged as one gesture). This is the practical content of the analogy,
not just a metaphor.

## 2.3 Trap vs detour: DETOUR — and the unwind is a *rebase*, not a retreat

Formally there are no traps: every move is invertible, the valid-state graph is connected through the root
(prior analysis §1.6), so any CUDA-optimal state is reachable from `ι(opt_WGSL)` by un-derivation +
re-derivation. The question is only the *cost* of the detour. Decompose the ported artifact into layers
and account for each:

| Layer | Content | Ports at | Evidence |
|---|---|---|---|
| **L0 — algorithmic/lemma** | which axes stream; online-softmax carried state; split-K reorder lemma; Welford; expm1 bias-correction; save/recompute contracts; fusion legality facts; the T2 differential harness itself | **100%** | The flash *lemma* is unchanged from FA1 (A100) through FA2, FA3 (H100), Triton, cuDNN attention. Lemmas are ℝ-algebra — hardware-invariant by construction. The certification harness (reference graph + tolerance + input distribution) is device-independent by design. |
| **L1 — macro-schedule skeleton** | loop-nest structure; which operand stages through which tier-KIND; fusion boundaries; K-split yes/no; grid decomposition | **~80-90%**, with named insertions | Triton expresses ~only this layer and hits 98% of cuBLAS on H100. The insertions that restructure it are enumerable: role-partition enters *above* the mainloop; TMEM changes epilogue shape; cluster/multicast changes the staging graph. |
| **L2 — parameters** | tile sizes, stage counts, vec widths, WG shape | **0% (re-tunes entirely), but re-tuning is cheap** | Un-apply+re-apply is in-calculus; the search is scalar enumeration (Triton autotuner's whole config space is BLOCK/num_warps/num_stages/num_ctas/maxnreg). Hours, automatable. |
| **L3 — CUDA-only decorations** | swizzle atoms, TensorMap, setmaxnreg split, cluster shape, L2 window | **written fresh, guided by L1** | No WGSL preimage exists; but their *slots* in the derivation are determined by the ported skeleton. |

**The unwind depth**: the expensive-looking case is role-partition, which sits near the root of the loop
nest — inserting it invalidates the derivation *suffix* below it. But because the artifact is a derivation
(proof term), the suffix is not re-discovered — it is **replayed over the new prefix with re-tuned
parameters**: a *derivation rebase*, mechanical where move preconditions still hold, prompting only at
genuine conflicts (exactly git-rebase semantics on proof terms). The prior analysis's "backing up along
valid states" is therefore too pessimistic a picture: you don't walk back; you re-root and replay.

**Net-positive accounting.** Is the browser starting point worth more than starting fresh on CUDA?

- **Yes, decisively, for the actual product use case** — novel fused/streamed kernels (attention variants,
  custom optimizers, new ops): development cost there is dominated by L0+L1 (algorithm discovery,
  correctness, streaming structure, the differential harness), all of which port at ~100%/~85%. FA3 is the
  canonical datum: the *algorithm* was given (FA2's L0), and a top team still spent the effort purely on
  L1-restructure+L3 for Hopper — i.e., that IS the detour cost, and it's incurred *from* the algorithm
  layer either way. The browser artifact delivers you to exactly where FA3's authors started, plus a
  machine-checked derivation and a portable test harness.
- **No, for commodity endpoints** — a plain dense GEMM's L0 is trivial and cuBLAS already exists; the
  correct port is a *binding*, not a derivation (Part 3's Y-bin).
- **One more real asset**: silicon correlation. The browser tunes on the user's local GPU via Dawn/Vulkan-
  or-Metal; porting to CUDA **on the same silicon** (RTX-class) keeps the T2 landscape strongly correlated
  — L2 parameter values themselves partially survive. Crossing silicon (RTX→H100) is where L2 fully
  re-tunes and L1 insertions bind — the repo's own V100→A100 verdict-flips (fusion detector, buffer-limit)
  are the in-house miniature of this.

**Ruling: DETOUR, with bounded and mostly-mechanical cost (parameter re-tune + derivation rebase around
enumerable insertions), and the browser starting point is net-positive precisely when the kernel contains
any algorithmic novelty — which is the only case the product exists for.** The one *genuine* trap-risk is
representational, not topological: if the ported state's WGSL-era decorations are treated as commitments
rather than priors, the search anchors at the saddle. Mitigation is already in the design's spirit:
decorations are (graph, decoration, **device-class**) triples — ported values must arrive marked *stale
prior*, and macro-move bundles must be offered as first-class gestures.

---

# PART 3 — Frontier completeness without the bridge

Can CUDA/cuBLAS/cuDNN/CUTLASS-frontier performance be completely and efficiently expressed in a geodesic
calculus at all?

## 3.1 CUTLASS/CuTe: the frontier already organizes itself as (decorations × moves × opaque atoms)

The claim under assessment — "CuTe's layout algebra IS essentially the layout-decoration formalism the
prior analysis named" — is **substantially correct**, with one honest caveat. Findings (Jay Shah's "A Note
on the Algebra of CuTe Layouts"; Cecka, arXiv 2603.02298; Carlisle-Shah-Stern, arXiv 2601.05972):

- A CuTe Layout is a function ℤ→ℤ represented as (Shape, Stride) nested tuples; the algebra has coalesce,
  **functional composition** (with left-distributivity over concatenation for injective right factors),
  complement, logical_divide (`A ∘ (B, complement(B))` — the tile/rest split), logical_product, and the
  zipped/tiled/raked re-associations. **Tiling operations ARE layout algebra**: `local_tile` =
  zipped_divide + slice; thread partitioning = TV-layout ∘ data-layout. The rewrite and the layout share
  one operation — decoration in the strongest sense.
- **Swizzle is IN the algebra**: `Swizzle<B,M,S>` is a bit-level bijection composed over a layout
  (`ComposedLayout`), not a bolt-on — Cecka's paper treats it as a core primitive. So WGMMA's 4 swizzle
  atoms are algebraic values of the same decoration.
- **Hardware constraints enter as algebraic objects**: an MMA/Copy Atom = PTX instruction + mandatory
  TV-layout; TiledMMA/TiledCopy tile atoms by composition. The rigid WGMMA fragment/core-matrix layouts are
  *layouts*, composable with the user's tiling choices — exactly "constraint-as-decoration."
- **The caveat: the algebra is partial.** Composition/complement/divide are defined only under divisibility
  (admissibility) conditions; CUTLASS enforces *stricter* divisibility than mathematically necessary; the
  categorical-foundations paper needs refinement (pullbacks/pushforwards) to make composition total. But
  note what the partiality is: **divisibility side-conditions — precisely the napkin's divisibility
  superscripts.** The calculus already has the right T0 mechanism for it. And Triton's 𝔽₂ **Linear
  Layouts** (arXiv 2505.23819) shows the totalization: layouts as binary matrices over 𝔽₂ on
  representation bits, uniform vocabulary subsuming shape:stride AND XOR-swizzle, generic layout-to-layout
  conversion, automatic optimal swizzling — shipped in Triton's backend, fixed real correctness bugs, avg
  1.07×/up to 1.4× across 265 kernels. **Adopt LL as the layout-decoration formalism** — it is the cleaner
  instantiation of the same idea and is production-proven.
- **The other half of the evidence**: CUTLASS 3.x deliberately keeps *schedule* OUTSIDE the layout algebra
  — dispatch policies (`MainloopSm90TmaGmmaWarpSpecialized<stages, cluster>`), kernel schedules
  (Cooperative / Pingpong), tile schedulers (persistent, Stream-K) are separate template axes ("orthogonal,
  reusable, composable" — NVIDIA's own framing). The frontier's flagship library thus factors kernel
  design as **(CuTe layout decorations) × (a small enumerable move/schedule vocabulary) × (opaque hardware
  atoms)** — which is *structurally the geodesic calculus*. Its seam is orthogonal-by-convention, not
  enforced (Pingpong×StreamK incompatibility leaks across — same species as the repo's epilogue⊥kSplit
  edge), so the calculus's explicit incompatibility edges are an *improvement* on the state of the art,
  not a compromise. CuTe DSL (CUTLASS 4.x, Python) confirms the algebra stands alone without C++ templates
  at claimed-parity performance.

## 3.2 Triton: what compiler-chosen scheduling proves about the calculus

Triton is the natural experiment for "how far does macro-structure + scalar autotuning get you when the
decorations are implicit": its autotuner searches ONLY user-enumerated scalars (BLOCK sizes, num_warps,
num_stages, num_ctas, maxnreg); it cannot express smem layout, register layout, explicit WS topology, or
cluster programming ("when [hand-tuned code wins] there is little the user can do since all the details
are hidden" — Triton's own Gluon docs). Results: **98-101% of cuBLAS (H100 GEMM), 98% of FA2 (H100
attention), 62% of cuBLAS (Blackwell GEMM), 1.5× behind FA3** (arXiv 2604.23466; FA3 paper). Two lessons:

1. **The macro-layer (the calculus's move skeleton) carries ~98% on mature hardware** — the geodesic
   vocabulary is not missing anything big for hardware one generation old.
2. **The residual on frontier hardware is expressible-structure, not magic**: everything in the
   Triton-to-FA3 gap (WS roles, TMA pipeline topology, swizzle-bound atoms, TMEM) is named CUTLASS
   vocabulary — i.e., inside an *extended* calculus, outside Triton's. The ecosystem's own trajectory
   confirms it: Triton grew **Gluon** (explicit layouts, explicit WS, warp-level programming — "lowers the
   programming model one level") and rebuilt itself on Linear Layouts. The compiler-implicit stance failed
   exactly where the calculus is explicit; the fix was to become the calculus.

## 3.3 cuBLAS/cuDNN as opaque endpoints: correct architecture, not an admission of defeat

Verified: cuBLAS dispatches via a runtime recommender; cuBLASLt exposes ~100 heuristic candidates per
shape (`cublasLtMatmulAlgoGetHeuristic`) plus Find-style autotuning; cuDNN likewise. They are autotuned
artifact libraries whose internals are partially below-PTX. In the calculus they are **named opaque
bindings with differential contracts**. Is that incompleteness? The decisive observation: **every serious
native CUDA stack makes the same architectural choice** — PyTorch/JAX bind cuBLAS/cuDNN for commodity
shapes; Triton's own matmul tutorial benchmarks *against* cuBLAS as the endpoint; even CUTLASS's pitch is
"beat cuBLAS on the shapes it wasn't tuned for" (DeepGEMM: comparable-to-better, up to 2.7× on
FP8/MoE-shaped problems cuBLAS neglected). Binding a vendor artifact where its enormous shape-specific
tuning investment wins, and deriving where you have structure it lacks (fusion, new dtypes, raggedness,
novel algorithms), is the optimal play *on native CUDA*, not a browser cope. The bin also demonstrably
shrinks (open kernels now meet/beat cuBLAS on targeted shapes) without ever emptying. The contract
mechanism (differential against the semantic graph) is exactly how such bindings should be admitted — it
is the same T2 machinery, so the architecture is uniform.

## 3.4 The sub-PTX floor, honestly

What lives below any portable calculus: SASS control codes (stall counts, barrier masks, register
reuse/yield flags) are set by **ptxas and are invisible from PTX**; register bank conflicts; instruction
scheduling. Measurements, labeled:

- **maxas/Scott Gray (Maxwell, classic)**: hand-SASS SGEMM ≈ 98% of theoretical peak, **~+5% over cuBLAS**
  via operand-collector-aware register allocation + reuse flags.
- **DeepGEMM (Hopper FP8, 2025, live example)**: hand-edited FFMA **yield/reuse bits** in compiled SASS,
  **>10%** on some shapes — and *nvcc 12.9 then absorbed the trick* (auto FFMA-interleaving), whereupon
  DeepGEMM dropped the SASS edit. The floor is real and it *evaporates upward* into compilers.
- **CuAsmRL (CGO 2025, arXiv 2501.08071 — the cleanest direct measurement)**: RL re-scheduling of SASS on
  already-optimized Triton LLM kernels: **avg +9%, max +26%, 1.09× geomean over ptxas -O3**.
- **The lore "async-tensor-core kernels are pipeline orchestration, not instruction scheduling" is
  confirmed** with a caveat: Blackwell's tcgen05 removing wgmma's 4-warp-sync cut scheduler stalls 18-23%
  in memory-bound kernels (arXiv 2512.02189) — data-availability dominates; FA3's entire design is overlap
  orchestration (H100: 989 TF matmul vs 3.9 TF special-function — a 256× ratio that only scheduling
  *structure*, not instruction order, can hide). The persistent SASS residual concentrates on
  CUDA-core-bound paths (fp32 epilogues, scaling, low-precision dequant chains).

## 3.5 Verdict: X / Y / Z

With the reasoning above (not fabricated precision — each number traces to a cited measurement):

- **X ≈ 85-90% of frontier-kernel performance is expressible as geodesic moves + decorations**, PROVIDED
  the vocabulary is the extended one: tier lattice as data {global, L2-window(host), DSMEM, smem, TMEM,
  warp, registers}; layout decorations as 𝔽₂ linear layouts (swizzle included) + TensorMap-style
  descriptors; capability-queried opaque MMA atoms with TV-layout contracts; moves + {pipeline,
  role-partition, pack, multicast, register-rebudget, persistent/stream-K under a co-residency axiom}.
  Basis: Triton's macro-only 98%-of-cuBLAS on H100; CUTLASS — which IS this vocabulary — *defining* the
  frontier (FA3 at 75-85% of hardware peak is written in it); the Blackwell-62% counterexample being
  attributable entirely to *named* missing vocabulary, all of which is in the extended list.
- **Y ≈ 5-10% as opaque-bound library calls** — measured as the performance you'd forfeit by *deriving*
  everything rather than binding cuBLAS/cuDNN on the dense standard shapes where their shape-specific
  autotuning + sub-PTX investment wins (CUTLASS gets within ~0-5% of cuBLAS generally; DeepGEMM beats it
  off-distribution). Coverage-wise this bin handles *most FLOPs* of a typical model; performance-delta-wise
  it is single-digit. Opaque bindings with differential contracts are the correct architecture even on
  native CUDA (§3.3) — this is a design confirmation, not incompleteness.
- **Z ≈ 5-10% genuinely below the calculus** (avg; **up to ~26%** on unlucky CUDA-core-bound kernels, per
  CuAsmRL) — SASS scheduling, control codes, register banking: below PTX, hence below ANY portable
  calculus, and historically absorbed upward by compilers (nvcc 12.9 FFMA-interleave; Linear Layouts
  auto-swizzle; Triton auto-WS). On tensor-core-dominated kernels this shrinks toward low single digits.

(X, Y, Z overlap rather than sum to 100: Y is an alternative route to part of X's territory, Z is a
residual multiplier on both.)

**Completeness verdict: the frontier IS expressible in a geodesic calculus to within the same residual
that separates CUTLASS itself from theoretical peak** — because CUTLASS, the artifact that defines the
open frontier, is *already organized as layout-algebra decorations × a small named schedule vocabulary ×
opaque atoms × bound endpoints*, i.e., the calculus's own architecture, independently evolved. "Efficiently"
also holds: the search spaces the frontier actually uses are small and enumerable (Triton's scalar
configs; CUTLASS's ~handful of kernel schedules × quantized tile shapes; cuBLASLt's ~100 candidates) —
the calculus structures the search rather than exploding it.

---

# CLOSING — What this means for the product

**"Optimize in browser, graduate to CUDA": real workflow, with one honest reframing.** What graduates is
the **derivation, not the tuned endpoint**. Marketing fiction version: "your browser-tuned kernel runs at
frontier speed on H100 unchanged" — false (optimum-level containment fails; ported states land as valid
Ampere-era kernels, ~60-70%-of-frontier-class on Hopper by the Triton/Blackwell evidence). Real version:
"the layers that dominate kernel-development cost — the algorithm, its lemmas, the streaming/fusion
structure, the correctness harness — are built once in the browser and port at 100%; the macro-skeleton
ports at ~85-90% with named, enumerable insertions; parameters re-tune automatically; and the artifact is
a derivation, so re-targeting is a *rebase*, not a rewrite." The browser is where kernels are *designed
and proven*; CUDA is where the last 1.5× is *harvested* — and the ported proof term delivers you to
exactly the point where FA3's authors started from FA2.

**What the representation must add NOW to keep the door open** (cheap now, expensive to retrofit):

1. **Tier lattice as data, not enum.** The napkin level-graph is already abstract; keep it that way in the
   IR so {DSMEM, TMEM, L2-window} slot in as entries with capacity/partitioning/allocation attributes
   (TMEM's column-quantized, warp-partitioned allocator is just an exotic capacity type). The repo's
   `placement: "register"|"shared"` union must become an open set.
2. **Layout decorations as 𝔽₂ linear layouts.** Subsumes strides, padding, AND XOR swizzles in one total
   algebra (vs CuTe's divisibility-partial one); production-proven in Triton; WGMMA's swizzle atoms and
   TensorMap descriptors arrive later as *values*, not new machinery. Divisibility constraints stay
   first-class T0 objects — they are literally WGMMA's shape quantization when the port happens.
3. **Capability-queried opaque atom families** — needed for WGSL subgroup-matrix *anyway* (its shapes are
   a runtime query), and the same schema object later carries mma.sync/wgmma/tcgen05 with their TV-layout
   contracts. This is the single highest-leverage addition: it is required for the browser feature that is
   already in Dawn, and it is the CUDA bridge.
4. **Reserve the `role-partition` move kind** (+ resource-rebudget companion) in the vocabulary even
   though WGSL can never use it — an empty slot costs nothing; its absence later forces a schema break.
   Same for the co-residency legality axiom (a T0 theory parameter per backend).
5. **Derivation rebase as a first-class operation** — replay a move-suffix over an edited prefix,
   prompting at broken preconditions. This *is* the graduation mechanism, and it's also just good undo/redo
   engineering that the editor needs anyway (persistent node identity across rewrites was already flagged
   as the load-bearing engineering).
6. **Ported/prior staleness marks on decorations.** Schedules are already (graph, decoration,
   device-class) triples; add provenance so ported L2/L3 values are priors for re-tuning, not commitments
   — the anti-anchoring discipline that keeps the saddle from becoming a de-facto trap.
7. **Macro-move bundles** (named composite derivations, ≈ CUTLASS kernel schedules) as first-class
   gestures — the saddle-escape directions of §2.2; also the right UX for "apply the flash recipe."
8. **Keep opaque bindings + differential contracts exactly as designed** — §3.3 shows this is the correct
   architecture on native CUDA too, and host-stratum objects (streams/graphs/L2 windows) already have a
   home one level up (torchlette's compiled-plan replay ≈ CUDA graphs).

---

## Sources (key)

- Chrome WebGPU release notes 128/129/133/134/144/145/147-148 (developer.chrome.com/blog/new-in-webgpu-*);
  gpuweb subgroups proposal; gpuweb#4195 (subgroup-matrix), crbug 348702031; gpuweb#4894 (f32 atomics,
  stalled), #5071 (64-bit atomics); WGSL F2F 2026-03-24 minutes; arXiv 2605.20706 (Llamas on the Web).
- PTX ISA (wgmma/tcgen05/mbarrier); Colfax tutorials (WGMMA, TMA, GEMM design, persistent/Stream-K,
  Blackwell TMEM & clusters); NVIDIA Hopper architecture in-depth; CUDA Programming Guide §4.9/4.11/4.13/4.18;
  CUTLASS docs (efficient GEMM, dispatch_policy.hpp, tcgen05 guide); PyTorch blogs (Hopper TMA, Ping-Pong
  GEMM, warp specialization, FA3); FlashAttention-3 paper (tridao.me/publications/flash3).
- Jay Shah, "A Note on the Algebra of CuTe Layouts" (Colfax 2024); arXiv 2601.05972 (Categorical
  Foundations for CuTe Layouts); arXiv 2603.02298 (Cecka, CuTe Layout Representation and Algebra);
  Graphene IR (ASPLOS 2023); arXiv 2505.23819 (Linear Layouts over 𝔽₂); Lei Mao's CuTe series.
- Triton docs (Config/autotune, Gluon overview, make_tensor_descriptor, persistent-matmul tutorial);
  arXiv 2604.23466 (CUDA Tile eval — Triton 62-101% of cuBLAS, H100 vs B200); maxas wiki (Scott Gray
  SGEMM); DeepGEMM (github deepseek-ai); arXiv 2501.08071 (CuAsmRL: +9% avg/+26% max over ptxas -O3);
  arXiv 2512.02189 (Blackwell microbenchmarks); cuBLASLt heuristics docs.
- Repo: `src/backend/webgpu/tile-{ir,ops,lowering,compiler,dispatch,autotune}.ts`, task #72,
  CLAUDE.md "what didn't work" ledger; prior analyses in this scratchpad.

---

# Visual & Play Design

Follow-up question: can every mechanism above be EXPRESSED AND MANIPULATED VISUALLY, holding two bars at
once — (a) **visual completeness** (fully seen, directly manipulated; no form-field-next-to-a-diagram) and
(b) **play** (the compulsion loop of a Zachtronics/Factorio-class optimization game)? Prior art below is
web-verified (sources at end of section); two findings frame everything:

- **The domain is already Zachtronics-shaped.** Open-ended optimization against antagonistic metrics, with
  no known optimum (we ship kernels we haven't optimally scheduled — Zach Barth's stated design principle:
  ship puzzles you haven't solved), visible flows, and a real machine that scores you. No re-theming needed.
- **Two of the hardest visuals have NO prior art anywhere** (verified absences): an interactive
  bank-conflict/layout manipulator, and any mainstream graft-a-derivation interaction. Those are the
  novelty/risk concentrations; everything else is assembly of proven devices.

## V1. The tier lattice + residency recoloring — strata the wires physically cross

**Visual.** The kernel view's background is a **geological cross-section**: horizontal strata, one per
tier, compute at the bottom — `GMEM / [L2-window] / smem / [DSMEM rooms] / [TMEM] / warp / registers`,
generated from the device profile (tiers-as-data, Part 1 recommendation #1). The semantic graph is drawn
ONCE (napkin rule: schedule = decoration, never a redraw); each tensor wire's **vertical position is its
residency** — a wire dips into the smem stratum where the K-tile is staged and rises back out. Every
stratum crossing renders as a **Sankey ribbon whose width ∝ bytes transferred per loop iteration**; the
napkin cost `H = Σ(color-changing wire sizes)` is therefore not a number in a panel — it is literally the
total ribbon cross-section on screen. Each stratum is a **shelf of finite width** (capacity `M_ℓ`);
resident arrays are blocks occupying shelf width; the shelf's fill bar IS the capacity check.

**Gesture.** Recolor = **grab a wire segment, drag it down a stratum**. Ghost preview shows the new
ribbons before release; a delta chip rides the cursor ("−38% GMEM traffic, +6KB smem"). Illegal drops
(shelf overflow) show the shelf flashing and the overflowing block red — the move is refused *visibly at
the resource*, not by dialog. Stream-partition interacts naturally: an over-wide block dropped on a small
shelf offers the split handle ("stream this axis at s=32 to fit"), which is exactly the napkin's
capacity-forces-streaming derivation, now a rescue gesture.

**Backend lattices, same language.** WGSL: 4 shelves + the vestigial bindable-window tick on GMEM.
Blackwell: more strata, three with special shelf physics — **DSMEM** renders the smem stratum as adjacent
rooms (one per CTA in the cluster) with doors a wire can cross into a neighbor's room (TMA multicast = one
ribbon fanning into several rooms in a single trunk); **TMEM** is a punch-card shelf beside registers
(slots quantized to power-of-2 columns ≥32 — the allocator's quantization is the shelf's visible slot
structure); **L2-window** is a highlighter band you drag across a GMEM-resident array (host-stratum
decoration, drawn at the kernel view's edge). Same grammar, more floors — the graduate-to-CUDA moment is
*the same picture growing new strata*, which is the visual proof of Part 1's state containment.

**Prior art.** Napkin wire-colors (this IS that, with position replacing color so magnitude gets the color
channel); Nsight Compute's memory workload chart (boxes-and-arrows annotated with %-of-peak — the proven
"where's the bottleneck" visual, here made editable); Factorio alt-mode (the whole-factory overlay
legibility standard); Sankey diagrams.

## V2. 𝔽₂ linear layouts / swizzles — the hardest one

Layout algebra is where formalisms go to become inscrutable; the design rule here is **never show the
algebra first — show its consequences, and let the algebra be the thing the user's drag solves for.**

**Visual (three linked lenses on one selected tile):**
1. **Tile view**: the smem tile as a cell grid, each cell tinted by its **bank** (32-column repeating
   palette) and labeled T#V# where an atom contract applies (the CuTe TV-layout rendering — currently only
   a LaTeX printer and a matplotlib gist in the wild; making it live in-browser is a verified first).
2. **Bank histogram strip** beneath: 32 slots; scrub or play one access phase and each thread drops a ball
   into the bank it hits. **Flat = fast; towers = serialization** (tower height = the conflict factor).
   The whole bank-conflict concept compresses into "don't stack the balls" — Tetris-grade readability.
   Ground truth stays honest: the strip shows *predicted* stacking; WebGPU exposes no per-counter
   equivalent of Nsight's shared-excessive metric, so measured wall-time from the in-tab profiler remains
   the arbiter, displayed beside the prediction.
3. **Bit-matrix drawer** (progressive disclosure): the 𝔽₂ layout as a small grid of bit toggles
   (input-address bits → output-address bits). Most users never open it; power users toggle bits and watch
   lenses 1-2 react instantly.

**Gestures.**
- **Swizzle dial**: cycle named atoms (none/SW32/SW64/SW128 — the WGMMA-legal set); cells animate to their
  permuted positions; the histogram visibly flattens. One knob, immediate consequence.
- **Drag-and-solve** (the novel one, enabled by 𝔽₂ linearity): grab a stack of conflicting thread-hits
  and drag it sideways onto empty banks. The system **solves the linear system over 𝔽₂** for a layout
  realizing that assignment (or proves none exists and shows the blocking constraint). Direct manipulation
  is semantically valid *because the algebra is linear* — the formalism choice (Part 3: adopt LL) is what
  makes the gesture possible. This is "drag threads onto banks" made real, and it has no prior art.
- **Divisibility superscripts as detents**: tile-size sliders click at legal multiples; hovering a detent
  names its origin ("×16: WGMMA core matrix; ×4: vec4" — constraints compose by LCM, badges stack).

**Prior art / gap.** triton-viz proves access-pattern grids teach (SIGCSE 2025; Triton Puzzles builds to
flash attention on them). Interactive bank-conflict animation: **verified absent everywhere** — static
diagrams and CLI benchmarks only (Mojo GPU Puzzles #32 is learn-by-profiling, not by watching). Risk
concentration #1, and also the largest teaching-value-per-pixel in the product.

## V3. Macro-move bundles — blueprints that jump valleys

**The problem being visualized** is Part 2's saddle: single new-axis moves regress (pipelining alone
−13%); only the bundle descends. The UI must make "this loses alone but you're mid-bundle" legible.

**Visual.** The move palette has single-move tiles and **recipe cards** (bundles ≈ CUTLASS kernel
schedules: "TMA + role-split + pipeline(3) + swizzle SW128"). A recipe card shows its mini-derivation as
3-5 glyphs plus a **predicted cost-vs-step sparkline: the valley curve** — down-up-down with the dip
shaded. Applying a bundle stamps a **Factorio-style blueprint ghost** over the kernel: every affected
wire/lane/shelf shown in ghost state, with the endpoint's cost chip. Confirm = the moves apply one-by-one
(each recorded individually — still geodesic, fully auditable/reversible), a progress rail across the top.

**Mid-bundle cost honesty.** The score panel splits into **NOW** (may be red — you're in the valley) and
**AT BUNDLE END** (predicted, green), exactly like a chess engine evaluating the position *after* the
combination rather than after the sacrifice. The "you are here" marker rides the valley sparkline. If the
user abandons mid-bundle, NOW is the truth and the remaining steps stay as a pending ghost they can resume
or revert — no silent half-states.

**Where recipes come from.** Shipped (the flash geodesic of the validity doc §1.2; the CUTLASS schedule
family), and **mined from the community**: any player's certified excursion that decomposes into a
reusable move sequence becomes a publishable recipe card (see V6) — the content treadmill is the player
base, which is Factorio's blueprint-sharing culture transplanted.

## V4. Role-partition — the lane stack (inventing the gesture for the newest move kind)

**Visual.** Below the kernel view, the **timeline**: the executor made visible as horizontal **lanes**
(one per warp/warpgroup), each carrying a glyph track of its steady-state loop body, with **one global
playhead** driving all lanes in lockstep. This is Opus Magnum's per-arm instruction track editor,
verbatim — the single most proven UI in the optimization-game canon, and it maps 1:1: arms→warpgroups,
instruction cells→schedule glyphs (LOAD/MMA/EPILOGUE/BARRIER), loop-on-wrap→steady-state mainloop, and
**leading blank cells = phase offset**, which is *literally* pingpong scheduling.

**Gesture sequence** (the role-partition move):
1. Default state: one bracketed homogeneous lane group (all warps run the same track).
2. **Drag a divider** across the lane stack → splits it into role groups (producer / consumer(s)).
3. **Deal the glyphs**: drag TMA-load glyphs into the producer track, MMA glyphs into consumer tracks.
   The mbarrier handoffs materialize automatically as **vertical baton connectors** between lanes at the
   produce/consume points (drawn, not configured — they're derived from the dataflow).
4. Pingpong vs cooperative is a visible topology choice: pingpong = two consumer lanes with offset blanks
   alternating epilogue/MMA; cooperative = two lanes bracketed onto one output tile.
5. **Register re-budget** (`setmaxnreg`): each lane group has a register **tank** at its left edge; drag
   the boundary between tanks to pour capacity from producer to consumers. T0 renders physically: the
   accumulator block must fit the consumer tank (overflows visibly if not).

**Feedback.** Press play: the playhead sweeps; you SEE producer batons landing in smem shelf slots (V1's
strata sit directly above the timeline — same screen, linked highlighting) while consumers chew the
previous stage. **Idle time renders as lane gaps (bubbles)**; the headline score is pipeline occupancy
(% of lane-time non-idle), which turns "overlap the memory and the math" into "close the gaps" — a
Factorio-idle-machine instinct. On WGSL backends the divider gesture is simply absent from the palette
(the reserved-but-empty move slot of Part 1's recommendation #4, visualized as a locked track).

**Prior art.** Opus Magnum tracks + phase-offset blanks (verified); SpaceChem's two waldos sharing one
spatial substrate with priority tiebreaks (the pattern for lanes sharing smem); RGP's
wavefront-occupancy-over-time view (the professional ancestor of "lane gaps are the enemy").

## V5. Derivation rebase — the tree you can graft

**Visual.** The edit history is a **derivation tree** docked at the side: nodes = states (each carrying
its three validity lights T0/T1/T2 and a cost chip — predicted, plus measured where profiled), edges =
moves (glyph-labeled), the current state a playhead on the tree. Branching is encouraged by making it
free: any node is one click to revisit (persistent node identity across rewrites — already flagged as the
load-bearing engineering). The tree IS the map of explored schedule space; community histogram marks (V7)
can be pinned to leaves.

**The graft gesture (rebase).** Select a move-suffix (drag along a branch), **drag it onto a different
prefix node** (or onto the same graph under a new device profile — the graduation case). The suffix
replays move-by-move: each move flashes green as its precondition re-checks, and parameters out of range
on the new target visibly **re-tune (a dial spins and settles)** rather than silently carrying.
**Partial failure is a game state, not an error**: replay halts at the first broken precondition; the
remaining moves stack up as a **pending hand of cards** hanging off the last good state, and the broken
precondition is presented as the current puzzle — with the *failing resource shown in the main view*
("stream-partition k needs 24KB shelf; you have 16 — split further, or recolor something out"), so fixing
the board lets you keep playing the hand. This is git-rebase-with-conflicts redesigned as
solitaire-with-rules — and it is the visual form of Part 2's whole argument: the detour is a rebase, its
cost is the pending hand's length.

**Prior art: none mainstream.** Git's rebase is the semantic ancestor and a famous usability disaster;
no shipped game has suffix-grafting on a derivation. Risk concentration #2 — but it is also the product's
signature interaction (it is *the graduate-to-CUDA mechanism* made tangible), so it cannot be cut, only
staged (v1: linear undo + whole-branch retarget; grafting later).

## V6. Freeform excursions & lemma mining — Zachtronics debugging, then the payoff ceremony

**Mode is diegetic, not modal-dialog.** Geodesic mode: the palette only offers applicable moves; the T1
light is green by construction; the board feels lawful. **Workbench mode** (freeform): the surface
visibly changes (blueprint-paper texture, amber T1 light "unproven"), and the tile/timeline/strata objects
become directly editable beyond the move set. Crucially the kernel **still runs while wrong** — wrongness
must be watchable, not a crash screen (Victor: show the data; eliminate hidden state).

**The red-to-green loop = the test bench.** A Zachtronics-style test panel: rows of input distributions,
each with a pass/fail lamp and an error-magnitude bar; selecting a red row renders **|Δ| as a heatmap ON
the output tensor in the main view** — wrongness is localized in space ("rows 32+ are stale" reads as
*that region glowing*), which converts debugging from scalar-staring into the SpaceChem/TIS-100 loop:
watch, hypothesize, poke, rerun. The differential harness already exists in-repo (`parity-*` tools,
autotune's warmup-3/timed-5/median protocol); the design requirement is only **sub-second re-run on edit**
for the inner loop's feel.

**The payoff ceremony (lemma minting).** When an excursion ends T2-green, the system diffs endpoint vs
launch state, **auto-decomposes against the known move set**, and presents the irreducible residue as:
**"NEW COMPONENT DISCOVERED"** — a move card with an editable name, its (F, B, carried-state, merge)
contract rendered as the card's stats, and the differential evidence stapled on as its certification seal.
The card drops into your palette; publishing puts it (attributed, usage-counted) into others'. This is the
validity doc's excursion-mining pipeline (§1.5) given its game form, and the deepest retention hook in the
design: *your* online-softmax-class discovery running inside other players' kernels. Prior art: Turing
Complete scores recursively through **user-built components** (verified) — proof that
player-made-component economies work in optimization games; Factorio blueprint culture is the
distribution model.

## V7. The objective function as game feedback

**The score pair.** Every state shows **PREDICTED** (napkin H/M — instant, device-independent, updates on
every gesture; the ghost bar) vs **MEASURED** (in-tab profiler median on the visitor's OWN GPU; the solid
bar). Their divergence is itself displayed as an anomaly badge — honest per the validity analysis (§2.4:
the analytic model cannot see occupancy/register effects; measurement is the objective, the model is the
explanation). Nothing in the Zachlike canon has this: **the score is real, on your real machine.**

**Antagonistic metrics, never collapsed.** Three separate histograms per (kernel, device-class):
**time/step · peak bytes · derivation size** (or dispatch count), each a population distribution of all
players' bests **with your mark on it** — the verified Zachtronics device, with Barth's rationale verbatim
("a global leaderboard manages to tell you is that you suck"; players see the histogram, formulate a
personal challenge, replay). The verified anti-pattern to refuse: Turing Complete collapses gates+delay
into one rank and its community documents the resulting friction. Device-keying is a *necessity* upgraded
to a feature: "fastest attention on M3 Max" is a leaderboard cuBLAS can't have.

**Shareable artifacts (both verified culture-carriers).** (a) The **schedule string** — the serialized
(graph-hash, decoration, device-class) triple, pasteable like a Factorio blueprint; importing it replays
the derivation. (b) The **looping replay GIF** — one steady-state cycle of the timeline + strata Sankey
(the kernel "breathing"), Opus Magnum's designed-to-be-GIFable loop transplanted; the recorder captures
one clean mainloop wrap.

**The FA3-on-your-laptop moment & the return loop.** A reference shelf of published derivations (the
eager→flash geodesic of the validity doc §1.2 first among them) that **replay move-by-move as watchable
tutorials**, then measure on YOUR silicon and place you on the population histogram. Return drivers:
weekly puzzle (a semantic graph + a constraint — "≤16KB smem; beat the reference on your device class" —
shipped unsolved, per the Zachtronics principle); new-device-class histograms (portability as a score);
and the lemma economy (V6) — the only loop with compounding content.

## V8. The zoom story — one camera through composition, docked lenses for projections

**Ruling: one continuous camera through the graph altitudes; deliberate modality switches for the two
non-graph projections.** Stratum-1 module tree → stratum-2 semantic graph → stratum-3 decorated kernel is
a genuine continuous zoom: nodes expand into subgraphs, decorations progressively reveal on the SAME graph
(the napkin's schedule-as-relabeling guarantees no redraw at the 2→3 seam — zoom never swaps the object,
it adds ink). This is the TensorBoard-hierarchical-clustering lineage plus the napkin's decoration rule.

**Where the camera metaphor breaks — and should**: the tile view (V2) and the timeline (V4) are not
deeper zoom levels; they are **projections along different axes** (intra-array memory space; steady-state
time). The graph camera moves through *composition* (part-of); no camera motion rotates composition into
projection, and tools that pretend otherwise ship the incoherent "semantic zoom" everyone regrets. So:
tile and timeline open as **docked lenses bound to a selection** (select a wire → its tile; select a loop
→ its lanes), with hard rules — lenses never become navigation roots, and node identity (color/ID chips)
is preserved across every view. The napkin's pseudocode diagram (columns = residency over time) is the
bridge artifact where strata and timeline meet — it's the one view where V1's shelves and V4's playhead
are the same picture. Final Victor device: the **device-class slider** (ladder of abstraction) — sweep
subgroup size / smem budget and watch which derivation-tree nodes stay green: abstraction over hardware
as a scrubbable dimension, which is precisely the containment story of Part 1 rendered as UI.

## Verdicts

**Three highest-conviction devices** (proven components, novel assembly):
1. **Strata-Sankey residency view** (V1) — the napkin cost model as a directly-manipulable picture; ribbon
   cross-section = H, shelf fill = M; recolor-by-drag. Triangulated by napkin wire-colors + Nsight's
   memory chart + Factorio alt-mode. It is simultaneously the cost model, the edit surface, and the
   containment story (CUDA = same picture, more strata).
2. **The lane-stack timeline with global playhead and phase-offset blanks** (V4) — the newest move kind
   (role-partition) lands on the single most battle-tested interaction in the genre (Opus Magnum's track
   editor); pipeline occupancy = "close the lane gaps" is instantly playable.
3. **Antagonistic histograms + schedule-string/looping-GIF sharing, device-keyed, measured on the
   visitor's own GPU** (V7) — the entire verified Zachtronics/Factorio compulsion apparatus, plus the one
   thing no Zachlike ever had: a real objective function on real personal hardware.

**Two biggest unsolved visual problems** (verified no-prior-art; novelty risk lives here):
1. **Direct manipulation of 𝔽₂ layouts/swizzles** (V2) — no interactive bank/layout manipulator exists
   anywhere; drag-and-solve-over-𝔽₂ is enabled by the formalism's linearity but completely unvalidated as
   an interaction; this is also where inscrutability pressure is highest.
2. **Derivation-tree grafting with partial-failure-as-pending-hand** (V5) — the ancestor interaction (git
   rebase) is a usability catastrophe and no game has shipped proof-suffix grafting; yet it is the
   graduation mechanism itself, so it must be staged (linear retarget first), not cut.

**Is the play-bar reachable, honestly?** Yes — for a designed subset, and that is not a concession but
the standard structure of the genre. The domain natively has every property the Zachlike compulsion loop
requires (antagonistic metrics, open-ended unsolved optimization, visible flows, shareable artifacts,
puzzles the designer hasn't solved) plus one the genre never had (real scores on the player's own
silicon). But Opus Magnum ships ~6 instructions; this calculus has ~9 move kinds × decorations × tiers ×
lemmas — nobody meets that wall as play. The resolution is vocabulary gating, and its feasibility is
already demonstrated in-domain: Triton Puzzles walks students from load/store grids to flash attention on
triton-viz visuals. Ship the **campaign** (gated move vocabulary, curated semantic graphs, histograms from
day one — the game) as the front door of the **workbench** (full calculus, freeform mode, lemma mining —
the expert tool), with one representation under both. The failure mode to refuse is shipping the workbench
with game cosmetics: the play-bar is reachable by design, not by paint.

### Section sources
- triton-viz (github.com/Deep-Learning-Profiling-Tools/triton-viz; SIGCSE 2025) · Triton Puzzles
  (github.com/gpu-mode/Triton-Puzzles) · CuTe print_layout/print_latex docs + Horace He's TV-layout gist ·
  Nsight Compute Profiling Guide (memory workload chart, shared-excessive counters, occupancy) · RGP
  wavefront-occupancy view (gpuopen.com) · Mojo GPU Puzzles #32 · compute.toys / Shadertoy (live-recompile
  loop, no in-browser profiling — our in-tab profiler is the differentiator).
- Zach Barth, SpaceChem postmortem (gamedeveloper.com — histogram/leaderboard quotes) · GDC 2019
  "Open-Ended Puzzle Design at Zachtronics" · Opus Magnum GIF-loop design + instruction-track editor
  (opus-magnum wiki; optimization blogs) · SpaceChem waldo mechanics (spacechem wiki) · Factorio
  production-statistics FFF #408 + alt-mode + blueprint strings · "The Factory Must Grow" (arXiv
  2102.04871) · Turing Complete scoring wiki + leaderboard-friction threads · HRM/7BH two-star scoring ·
  Bret Victor, "Learnable Programming" / "Up and Down the Ladder of Abstraction" (worrydream.com).

---

# Intrinsic Play & the Difficulty Ladder

Owner's correction absorbed: the question is not whether the task can be dressed as a game, but whether
the domain is play-inducing **as-is** — working directly on real kernels with the calculus, can a person
*feel their way* toward what makes increasingly complicated things faster? Three questions: is there an
intrinsic difficulty ladder; is the state of the art dynamically contained in the closure of the move-set;
and what does a real progression on real kernels look like versus the nearest existing thing.

## L1. The intrinsic ladder exists, and it is not designed — it is the machine's own cost function

There is a natural ordering, and its origin is physical, not pedagogical: **each rung is defined by which
term of the cost function becomes binding once the previous term is optimized away.** Optimize traffic and
capacity binds; optimize capacity and grid utilization binds; optimize utilization and uncovered latency
binds; and so on. Nobody chose this order — it is the machine. That is what makes the ladder *intrinsic*:
you cannot encounter rung n+1's problem in earnest until you have solved rung n's, because rung n's cost
term dominates the measurement until you remove it. The feedback signal itself sequences the curriculum.

The ladder, with each rung's load-bearing intuition and where the repo's own kernels sit:

| Rung | Binding constraint | Load-bearing intuition | Repo kernel at this rung |
|---|---|---|---|
| 0. Elementwise chains | round trips to GMEM | cost = bytes moved; arithmetic is free; fusion deletes traffic | the fusion engine; every unfused bwd op in the 06-14 profile |
| 1. Reductions | dependency depth | associativity buys parallelism (first lemma contact: fp reorder is a *license*, not a fact); thread→subgroup→workgroup→grid combining | `wgReduce`, GEMV subgroup butterfly; the deliberately-serial chunked sum (a rung-7 artifact shipped at rung-1 quality — the documented teaching example) |
| 2. Layout/coalescing | access *shape* | the same bytes cost differently by access pattern; layout is a free variable; smem as a shape-changer; first bank contact | transpose-as-stride-flip, `detectSimpleTranspose`, narrow views |
| 3. Tiled matmul | reuse ratio | arithmetic intensity; the H*(M)=α·M^(−β) curve *felt*: tiles trade capacity for traffic; register blocking as a second tier of the same idea | `tile-matmul` (tileM/N/K, threadTile) |
| 4. Structured fusion | register/capacity pressure of fusion | fusion is not free at high intensity; epilogues; where to STOP fusing | `EpilogueConfig` chains; the epilogue⊥kSplit incompatibility |
| 5. Streaming/online | memory ∝ problem size | a dependent computation can sometimes be re-associated into a running form — **the lemma wall (first discontinuity, see below)** | online softmax in `attention-kernel.ts`; cross-entropy row program |
| 6. Residency over time | liveness, not size | memory is a budget across TIME; save-vs-recompute; multi-pass structure; aliasing | attention 3-pass backward, logsumexp recompute, `reuseShared` |
| 7. Grid-level structure | machine has W workers, not ∞ | waves and tails; the grid is schedulable; split the reduction across workers and combine | `kSplit`, chunked reduction combine pass, packed Adam (`pack`) |
| 8. Asynchrony | uncovered latency | **the inversion (second discontinuity)**: add buffers/work to go FASTER; cost stops being volume and becomes overlap | — absent (pipelining not in the IR; tried by hand, −13% on V100) |
| 9. Heterogeneous executors | slowest pipeline stage | specialize workers; producer/consumer balance; phase offsets | — impossible in WGSL (role-partition reserved slot) |
| 10. Atom co-design | instruction-dictated structure | **the reversal (third discontinuity)**: the hardware atom dictates layout and quantizes tiles; you design backward from its demands | — subgroup-matrix experimental; WGMMA/TMEM CUDA-only |

Two observations the table forces:

**The repo's kernels populate rungs 0-7 completely and rungs 8-10 not at all** — and that boundary is not
an accident of effort, it is Part 1's containment boundary. The WGSL-expressible space IS the first eight
rungs; the browser can teach *everything up to the second discontinuity* on real silicon, and the rungs it
cannot host are precisely the CUDA-graduation content. The intrinsic ladder and the WGSL/CUDA seam
coincide. This is the strongest single fact for the product thesis: the browser is not a toy version of
the ladder, it is the ladder's lower eight rungs at full fidelity.

**Where feel transfers and where it provably doesn't.** Within a rung the feedback landscape is
learnable by intuition in the strict sense: responses are locally monotone (bigger tile → less traffic)
until a visible cliff (capacity, divisibility), and the napkin model correctly *rank-orders* the moves
(validity doc §2.4) — prediction and measurement agree in direction, so trial-and-feel converges. Across
rungs, the bytes-intuition of rung 0 survives as the denominator of everything above it; the ladder
compounds. But there are exactly three discontinuities where feel STOPS transferring, and they are
structural, not UI failures:

1. **The lemma wall (rung 5).** No amount of schedule-feel produces online softmax's correction factor
   `exp(m_old − m_new)` — it was a research publication (Milakov & Gimelshein 2018), not a tuning
   discovery. The manifold has a hole there; you cross it by importing algebra, not by gradient-following.
   The honest pedagogy is exactly the calculus's honest structure: geodesic feel up to the wall, then a
   *named lemma import* whose carried state you inspect, then feel resumes on the far side.
2. **The latency inversion (rung 8).** Rungs 0-7 teach "cost = volume"; rung 8's truth is "cost =
   uncovered latency," and the old intuition **actively misleads** — the repo's own −13% double-buffering
   entry is a rung-0-7 intuition ("more smem use, no traffic change, should be neutral") being punished by
   a rung-8 landscape. This is also where the analytic model officially stops explaining (transfer-neutral
   moves with large measured effects) and *measurement-over-model* becomes the teacher.
3. **The direction reversal (rung 10).** Below it you choose structure and then satisfy constraints; at it
   the atom's constraints propagate BACKWARD into every choice (WGMMA pins M=64, quantizes N/K, dictates
   swizzle). The design habit inverts from forward synthesis to backward accommodation.

A ladder with marked discontinuities is still a ladder — arguably a better one: the discontinuities are
where the domain changes what kind of thinking it rewards, which is what keeps long progressions from
being monotone grind. But intellectual honesty requires stating the converse too: **between the
discontinuities, the landscape is genuinely feel-navigable only if the measurement is stable.** Real
silicon adds noise (thermal state, tab contention, pool warmup — the repo's own profiler protocol exists
because early-step numbers lie). A learner ascribing meaning to ±8% noise learns superstitions, not
intuitions. The medians-and-warmup discipline (`tile-autotune`'s warmup-3/timed-5/median) is not a nicety;
it is the precondition for the domain being learnable by feel at all.

## L2. SOTA-in-closure: the DYNAMICS — the grammar froze a decade ago; the vocabulary never will

Question: as SOTA evolves (FA3→FA4, Blackwell kernels), does it keep landing inside the closure of the
move-set (+ growing lemma library), or does each generation add move KINDS? The history, sorted by what
each innovation actually was:

**Genuinely new move KINDS, last ~15 years:**
- **role-partition** (warp specialization): research origin **CudaDMA, SC 2011** (Bauer et al.; then the
  Singe compiler, 2014) — went *mainstream* only when Hopper's TMA made producer/consumer asymmetric
  (CUTLASS 3.x 2023, FA3 2024). One new kind — and even it is 15 years old as an idea; hardware promotion,
  not conceptual birth.
- **pack** (horizontal batching of independent same-shape work): batched GEMM (~2015), apex multi-tensor
  optimizers (~2019). Arguably a kind, arguably a degenerate weave; either way, one.
- Everything else fails the "new kind" test: persistent kernels (persistent-threads literature, ~2012) =
  grid re-partition + a legality axiom; Stream-K (PPoPP 2023) = stream-partition of K across CTAs + the
  same fp-reorder lemma as every split-K since Fermi; pipelining = software pipelining, VLIW-era, made
  cheap by cp.async (2020) and TMA (2022); megakernels (2025) = persistent + role-partition + a
  scheduler-in-software — a *composition*, plus one axiom (whole-program co-residency).

**What actually arrives every generation, in volume:**
- **Tiers**: L2 windows (Ampere 2020), DSMEM (Hopper 2022), TMEM (Blackwell 2024) — roughly one per
  hardware generation, each slotting into the tier-lattice-as-data schema with exotic shelf physics.
- **Decoration values**: swizzle atoms, TensorMap descriptors, fragment layouts, cluster shapes — every
  generation, always values of the existing layout/config decorations.
- **Lemmas**: a steady drip, order of one or two significant ones per year in the GEMM/attention world —
  online softmax (2018), flash recompute-from-LSE (2022), FA2's rebalanced work partition (2023), FP8
  two-level accumulation (DeepGEMM's CUDA-core promotion, 2025), and FA4's reported innovations
  (rescale-skipping when the running max is unchanged; polynomial exp approximation to dodge the SFU
  bottleneck — the latter actually a stratum-2 numerics contract, not even a schedule lemma). Note what
  FA4 is made of: **new lemmas + a tier retarget (TMEM) + re-tuned pipeline topology — zero new move
  kinds.**
- **Legality axioms**: rare — forward-progress/co-residency (formalized with cooperative launch, then
  clusters), async-proxy fencing (Hopper). Maybe three in the era.

**Dynamics verdict: convergent grammar, open vocabulary.** The move-KIND set has been effectively closed
since ~2014 in research terms and since Hopper in mainstream terms: across fifteen years of the most
intensely optimized software domain on earth, the count of genuinely new kinds is **~2 (role-partition,
pack)** — everything else that felt revolutionary decomposes as new tiers, new decoration values, new
lemmas, or new legality axioms over the fixed kinds. The snapshot number (~85-90% in closure) is therefore
not a decaying asset: each hardware generation grows the *vocabulary* along axes the representation
already treats as data (tier lattice, atom families, decoration values, lemma registry), while the
*grammar* — the move kinds, the expensive thing to retrofit — has a demonstrated decade-scale half-life.
Part 1's recommendation list is thus not a hedge against an unknown future; it is the observed shape of
the last decade extrapolated one step. Risk kept honest: Blackwell's CTA-pair MMA and hardware tile
scheduling (Cluster Launch Control) hint that *scheduling itself is migrating into hardware* — if that
continues, the future threat to the calculus is not a missing move kind but moves becoming
hardware-internal (unobservable, unschedulable). Oddly, that failure mode favors the calculus: what the
hardware absorbs becomes an opaque atom with a contract, which the representation already knows how to
hold.

## L3. A concrete progression on real kernels — and the honest Triton Puzzles comparison

Twelve exercises, each stated as the domain states it — *here is a semantic graph and a measured baseline
on your GPU; make it faster* — no further scaffolding. Every baseline is real (several are the repo's own
shipped or historical states). For each: the isolated intuition, the expected discovery, and where SOTA
sits relative to the closure.

1. **Bias+GELU+residual chain, unfused** (per-op dispatches — the repo's pre-fusion state). Intuition:
   traffic is the cost. Discovery: fuse; watch time scale with bytes-touched, not op count. SOTA
   (torch.compile-class full fusion): in closure, rung 0.
2. **Sum of 10⁸ elements, baseline = the SHIPPED chunked reduction** (workgroupSize:1 serial chunks —
   `reduction-tile-ir.ts:489`). Intuition: parallelism vs dependency; associativity as license. Discovery:
   tree/subgroup reduction inside chunks (pure moves, no new vocabulary — the validity doc's own teaching
   example). SOTA note, stated honestly: single-pass decoupled-lookback is OUTSIDE the WGSL closure
   (needs forward progress); the two-pass endpoint is in closure and near it.
3. **LayerNorm as three passes** (mean, var, normalize — the pre-fused state). Intuition: passes cost
   traffic; a dependent chain can sometimes stream. Discovery: two-pass → one-pass via **Welford — the
   first lemma import**, deliberately early and small, to teach what a lemma IS before rung 5 needs a
   big one. SOTA (fused row program): in closure + 1 lemma.
4. **Permute/transpose, naive strided copy.** Intuition: access shape ≠ byte count. Discovery: smem
   staging, coalesce both sides, padding kills bank conflicts (first layout decoration). SOTA: in closure.
5. **Matmul, one-thread-per-output baseline.** The big rung: smem tiles → register blocking → vec4 →
   the capacity/traffic curve traced by hand. Discovery: reuse is THE game; H*(M)=α·M^(−β) becomes
   physical. SOTA (repo tile-matmul on WGSL; CUTLASS on CUDA): in closure (browser: fully; CUDA: with
   atom vocabulary).
6. **GEMV / decode-shape matmul** (repo GEMV family). Intuition: regime identification — rung-5's reuse
   intuition is USELESS here (no reuse to be had); the roofline position, not the op name, picks the
   strategy. Discovery: subgroup butterfly, rows-per-workgroup. SOTA: in closure.
7. **Tall-skinny matmul + epilogue** (shapes that starve the grid). Intuition: the machine has W workers;
   waves and tails. Discovery: kSplit (fp-reorder lemma) + epilogue fusion + their real incompatibility
   (the repo's epilogue⊥kSplit edge — first contact with implementation-induced non-commutation). SOTA:
   in closure.
8. **Softmax, three-pass baseline.** **The lemma wall, undisguised.** Intuition isolated: streaming needs
   carried state; some carried states are inventions. Expected outcome for most people: hit the wall,
   fail honestly, import online-softmax, *inspect* its (m, l) state and correction factor until it reads
   as inevitable. SOTA: in closure + THE lemma.
9. **Attention forward, eager baseline (materialized scores), seq 2048.** The capstone composition:
   rungs 3+5+6+7 assemble into flash — the repo's `attention-kernel.ts` is the endpoint, reached by the
   validity doc §1.2 derivation. SOTA-in-browser: this. SOTA-on-earth (FA3/FA4): in the *extended*
   closure only — requires role-partition/TMA/TMEM (rungs 8-10), stated plainly as the ladder's upper
   rungs living past the browser boundary.
10. **Attention backward, autograd-composed baseline.** Intuition: memory across time; save-vs-recompute
    as a *contract*, not a trick. Discovery: recompute-from-logsumexp, the dQ/dKV axis-role swap, causal
    tile-skip (predication lemma), smem aliasing. SOTA: in closure + 2 lemmas + aliasing decoration.
11. **Optimizer step over 124M params, per-param baseline** (the repo's own historical 158-submit state).
    Intuition: horizontal structure — many small identical things ARE one big thing. Discovery: pack,
    chunked bindings, scalars-as-data (the volatility lesson, taught by the frozen-step_size story). SOTA
    (multi-tensor fused): in closure.
12. **Graduation (requires the CUDA target): the exercise-5 matmul, rebased onto H100.** Baseline: the
    ported schedule, measuring ~60-70% of cuBLAS. Intuition: the saddle — single new-axis moves regress;
    coordinated re-derivation descends. Discovery: WGMMA atom quantization reshaping "your" tiles;
    pipeline+role-partition as a bundle; the rebase as an operation. SOTA (CUTLASS-class): in extended
    closure; the remaining few percent to cuBLAS is the honest sub-PTX floor, shown, not hidden.

Note what the sequence is: **rungs 0-7 = exercises 1-11, all on torchlette's own kernels, all measurable
in the browser today; exercise 12 is the containment thesis as an exercise.** The progression wasn't
designed and then mapped to the repo — the repo already lives on the intrinsic ladder, including shipping
one deliberately-dumb state (exercise 2's baseline) and one historical disease (exercise 11's baseline)
that are better teaching baselines than anything invented could be.

**Against Triton Puzzles, honestly.** What Triton Puzzles is (verified): ~a dozen exercises teaching
*correct expression* of block programs (indexing, masking, program_id) on the Triton **interpreter** — no
GPU, no timing, no autotuning; it ends at flash-attention *correctness*. What this progression offers that
TP structurally cannot:
- **Performance as the objective, measured on the learner's own silicon.** TP never times anything; here
  the entire signal IS time, and the machine scoring you is yours. Nothing else in the education landscape
  does this in a browser tab.
- **The derivation record**: the answer to "why is it fast" is an auditable chain of moves and lemma
  citations, not a final code blob. TP's artifact is code that passes; ours is a *proof-shaped explanation*
  that measures.
- **Reaching actual SOTA endpoints** (the browser SOTA at rung 9, the honest pointer past it), with the
  gap to world-SOTA quantified rather than elided.
- **The wall, taught as a wall**: TP hands you the flash algorithm to implement; this progression lets you
  *fail to invent it*, which is the only way the lemma library's existence makes emotional sense — and
  lemma-import/minting is itself part of the loop.
- **Device plurality as content**: the same derivation measuring differently on M3 vs 4090 vs an Intel
  iGPU is signal (device-keyed decorations), not noise to be hidden.

What Triton Puzzles does better, conceded without spin:
- **Zero-setup determinism.** Colab + interpreter = identical experience for everyone, no GPU required,
  no measurement noise. Our loop *requires* WebGPU and inherits real-silicon variance; if the noise
  discipline slips, the feel-landscape corrodes (the single biggest UX risk in this whole design).
- **A tiny surface.** `tl.load/store` + block indexing is learnable in an hour; our calculus, even
  vocabulary-gated, fronts more concepts per exercise.
- **Correctness feedback is gentler**: TP's visualizer shows wrong cells immediately at interpreter speed;
  our T2 differential needs a real run, and diagnosing a red differential at rung 9 is expert work however
  well it's localized.
- **It exists, is adopted, and is institutionally embedded** (GPU-mode community, SIGCSE). We would be
  claiming a superset of a thing with real network effects.

The two are not actually competitors: TP teaches "how to say it"; this teaches "what makes it fast, on
your machine, with the reasoning preserved." TP is the missing rung −1 of this ladder — and the fact that
the world's most successful kernel-education artifact stops exactly where this progression begins is
decent evidence the thrust is real.
