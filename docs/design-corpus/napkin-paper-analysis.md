# arXiv 2412.03317 — "FlashAttention on a Napkin: A Diagrammatic Approach to Deep Learning IO-Awareness"

Vincent Abbott (UCL) & Gioele Zardini (MIT LIDS). Published TMLR 03/2025. v2 2025-01-19.
Code: https://github.com/mit-zardini-lab/Napkin (minimal — an Excel performance-model spreadsheet + README; "adding more in the future to systematize IO-Awareness").
Evaluated against our design theses: browser-editable bidirectional visual↔code representation of NN *training programs*; canonical hierarchical dataflow graph; tentative three strata (generator → semantic graph → schedule bindings); nanoGPT-speedrun completeness benchmark; kernel-level visualization included.

---

## 1. WHAT THE PAPER ACTUALLY IS

**Contribution.** A pen-and-paper *derivation calculus*, not a system. It extends Neural Circuit Diagrams (NCD, Abbott, TMLR 2024, arXiv 2402.05424) — a categorical string-diagram notation for DL architectures — with two new dimensions: (a) GPU **memory-hierarchy placement** of every array, and (b) **partitioning relabelings** that turn a semantic algorithm diagram into an IO-optimized schedule of itself. From the relabeled diagram, transfer-cost and memory-usage *performance models* are read off mechanically, and looped pseudocode is derived step-by-step. The showcase: FlashAttention (1/2/3, fp16 and fp8) falls out of the attention diagram by systematic relabeling, with an analytic performance model compared against published H100 FA3 numbers.

**Mechanism — the core devices:**
- **Arrays as wire bundles**: each tensor is a vertical column of wires, one per axis, labeled with sizes; tuples separated by dashed lines. Functions are boxes between columns; composition is horizontal placement; parallel composition (⊗) is vertical stacking.
- **Memory levels as wire COLOR**: "higher level ℓ0 arrays are colored black and lower level ℓ1 arrays are colored orange." The hierarchy itself (GMEM/L2/SMEM/registers) is an abstract level graph with "pipes" for shared memory between a level and its family of sub-cores.
- **Group partitioning (= tiling)**: an axis is relabeled with a group size g_a at the lower level; the mapped function is applied per group and rejoined. Number of groups N = a/g_a multiplies the per-group cost.
- **Stream partitioning (= recomputation/online algorithms)**: an axis relabeled with a stream size s_b as it crosses levels; valid iff the function is *streamable* — decomposable into head F (init) + body B (accumulator), "recursively decomposed until a′ ≤ s_a."
- **Fusibility theorem**: streamability is preserved under composition and weaving — this is *kernel fusion as a checkable algebraic predicate*.
- **Weaving**: mapping a function over an extra (batch) axis, drawn by threading the axis over the outputs and chosen inputs — precise broadcasting, NCD's signature device.
- **Cost readout**: transfer cost H_ℓ = "the sum of the size of arrays changing colors"; memory M_ℓ ≥ max column width at that level. Optimal costs take a canonical form H*(a⃗,M) = Σ_t α_t(a⃗)·M^(−β_t); β exposes the algorithm's memory sensitivity (matmul β=0.5, attention β=1 — hence attention's greater SMEM sensitivity). Quantization enters as q^(1+β) (fp16→fp8 ≈ ×4 for attention). Multi-level hierarchies and cross-transfer (thread-block clusters, multi-GPU) enter as bandwidth-weighted sums.
- **Pseudocode diagrams (§5)**: diagrams where columns literally are per-timestep memory residency; "the columns of pseudocode diagrams provide the size of variables required in memory and the transfers/operations we need to apply." Hardware constraints (128B coalescing, wmma/wgmma tensor-core shapes) appear as **divisibility superscripts on axis sizes**; multiple constraints impose their LCM.

**Source.** Manual specification only. No traced graph, no source AST, no framework introspection. The human writes the algebra and draws (or derives) the diagram. Correctness of derived schedules comes from the streamable-function definition, not from testing.

**Relationship to editing: none.** The diagrams in the paper are hand-drawn (the NCD lineage uses Mathcha templates). There is no tool, no renderer, no interaction, no round-trip in this paper. It is a *notation plus theorems*, whose UI is a napkin.

**What it does NOT do:**
- Does not generate kernels (no CUDA/Triton codegen; pseudocode is derived by hand following the method).
- Does not ingest existing code or models.
- Does not cover training programs at all: no optimizer, no autograd (NCD's earlier paper diagrams backprop, but Napkin is inference-kernel scoped), no data pipeline, no schedules-over-time, no phase transitions.
- Does not address graph scale or layout: every diagram is a single small kernel-sized expression.
- Validation is analytic-model-vs-published-benchmarks, not an implemented artifact.

**The follow-up that partially closes the tooling gap** — arXiv 2604.07242 (Apr 2026, same authors), "Weaves, Wires, and Morphisms: Formalizing and Implementing the Algebra of Deep Learning": formalizes broadcasting via novel *axis-stride* and *array-broadcasted* categories and ships two packages:
- **pyncd** (Python, machine-facing): algebraic construction, configuration, **compilation to PyTorch**, graph conversion.
- **tsncd** (TypeScript, human-facing): renders the same serialized terms as diagrams **in the browser**; receives terms from pyncd over JSON/WebSockets; internally layered as `Render` (framework-independent) → `Framework` → `HTMLRender`, with React support planned.
Direction of flow is strictly **algebra → {PyTorch code, diagram}**. The diagram is a projection; there is no channel from diagram back to algebra, and no parsing of PyTorch back into algebra. So even the implemented version of this research line is a *generator with a viewer*, not a round-trip editor.

---

## 2. THE LEVELS QUESTION

**Verdict: strongly supports our three-strata design at the 2↔3 seam, and supplies the formal contract we were missing there. Silent on stratum 1.**

The paper's central claim is exactly our thesis (2): **FlashAttention is not a different graph — it is the attention diagram after semantics-preserving relabelings.** Tiling = group partitioning, recomputation = stream partitioning, fusion = the streamability-composition theorem. A "schedule" is a *decoration* of the semantic diagram: (level colors on wires) × (partition sizes g, s on axes) × (divisibility superscripts). The math never changes; only labels do. This is the strongest independent endorsement available of "schedule bindings, never a different graph."

Its own level decomposition (worth contrasting, not adopting wholesale):
1. **Functional diagram** — pure math, no memory (≈ our stratum 2, semantic graph).
2. **Leveled/partitioned diagram** — same diagram + colors + g/s labels (≈ our stratum 3 *binding*, drawn in place).
3. **Pseudocode diagram** — the schedule unrolled into explicit loop/residency structure (a *projection* of stratum 3, not a fourth stratum — derived mechanically from the streamable normal form).

Two refinements this forces on our stratum-3 contract:
- **The seam contract is a theorem, not a test.** "This binding is a valid schedule of this math" can be checked structurally: every proposed streaming/fusion must exhibit the head/body (F, B) decomposition; every tiling must be a group partition of a mapped axis. That's a machine-checkable predicate at the seam — stronger than the "testable contract" we asked for.
- **Cost is derivable at the seam.** Given a binding, H = Σ(sizes of color-changing wires) and M = max column width are computable *without running anything*. Our kernel view can show live napkin math (including the q^(1+β) quantization knob and per-level bandwidth weights) as a function of the user's schedule edits.

What it does NOT support: any argument for or against our **generator stratum**. There are no config knobs, no ×N-repeat containers, no phases, no optimizers-as-graphs. The paper's world ends at a single kernel invocation. It refines the bottom seam; it neither validates nor threatens the top one. It also does not *undercut* multi-level — on the contrary, it demonstrates that levels-as-decorations-of-one-graph beats levels-as-separate-artifacts, which is an argument for keeping our strata as *projections of one canonical object* rather than three synchronized documents.

---

## 3. STEALABLE IDEAS (ranked)

1. **Wire color = memory level; cost = color-changing wires.** The single best visual device in the paper. It makes IO cost a *local, visible, attributable* property of edges instead of a global profiler number. In our kernel view: color the dataflow edges by residency (GMEM/SMEM/reg — or for torchlette: pooled buffer / arena / uniform), and render the transfer bill as literally the set of edges that change color. Why #1: it's simultaneously a rendering rule, a cost model, and (see §4) an edit affordance.

2. **Schedule-as-relabeling of the same graph.** Tiling and streaming appear as *annotations on axes* (g_a, s_b) of the unchanged semantic diagram — no second drawing. This is the visual form of thesis (2) and should be our kernel-descent interaction: zooming from semantic to schedule view never swaps the graph, it progressively reveals decorations. Why #2: it resolves the "kernel view vs math view" identity problem by construction — node identity is preserved because nothing is redrawn.

3. **Edits as named rewrites (the derivation calculus itself).** Every derivation step in the paper is one of a small set of semantics-preserving moves: group-partition(axis, g), stream-partition(axis, s), weave(axis), fuse(F,G if streamable). An editor should expose exactly these as gestures; the edit history is then a *proof* that the schedule equals the math. Why #3: it's the deepest structural steal — it turns "two-way editing at the kernel level" from a pixel-manipulation problem into a term-rewriting problem, which is what makes round-trip tractable at all.

4. **Streamability/fusibility as a checkable predicate.** The head/body (F,B) decomposition gives a green/red legality light for user-proposed fusion or streaming edits, with the failure localizable ("this op is not polymorphic along axis a"). Maps directly onto torchlette's fusion detector and tile-IR: our FUSIBLE_OPS set is an ad-hoc shadow of this theorem.

5. **Pseudocode diagrams: columns = memory residency over time.** A bridge representation between the graph view and kernel code: loop structure, per-step allocation (column width = bytes live), and transfers all in one picture, derived mechanically from the streamable normal form. This is the missing middle for our "tiling / memory movement" visualization ask — closer to what users need than either a DAG or WGSL text.

6. **Divisibility superscripts for hardware constraints.** Coalescing (128B/q) and tensor-core shape requirements annotate axis sizes as "divisible by k" superscripts; constraints compose by LCM. A tiny, dense, legible device for surfacing why a schedule knob has a step function in it (why g must be a multiple of 16). Editable form: constraint badges on axis-size fields that snap slider values.

7. **Weaving / axes-as-wires (from the NCD substrate).** Broadcasting shown precisely — which axes an op maps over vs consumes — something ordinary NN dataflow graphs (Netron, TensorBoard) simply omit. Worth adopting as the *zoomed-in lens* on an edge: expand a tensor edge into its axis wires on focus.

8. **tsncd's renderer layering** (`Render` framework-independent → `Framework` → `HTMLRender`). Sensible architecture for a browser diagram engine we'd want to hit-test and edit; also evidence that this notation renders fine in TS/HTML.

**Not available to steal:** any layout algorithm for large graphs, containment rendering, edge routing for skips, or level-of-detail machinery. NCD's columnar expression layout is math-paper-scale; skip connections are handled by wire duplication, which visibly degrades beyond a few blocks. For whole-model scale we still need the hierarchical-clustering lineage (TensorBoard's graph visualizer, Wongsuphasawat et al. 2018) — absent from this paper's world.

---

## 4. THE BROWSER-EDITABILITY GAP

General diagnosis: the entire research line assumes a *human doing math* produces the diagram; the implemented pipeline (pyncd→tsncd) hard-codes one-way flow (algebra serialized over a websocket, rendered, done). Nothing in the theory prevents bidirectionality — the theory is actually unusually *friendly* to it (finite rewrite vocabulary, structural validity predicates) — but every artifact assumes view-only-ness. Device by device:

- **Wire recoloring (residency editing).** To make color a two-way control: (a) legality checking — capacity M_ℓ ≥ max column width and streamability of anything forced to recompute; the paper supplies both checks; (b) a compilation target that accepts residency directives — for us, tile-IR schedule choices / buffer-pool vs arena placement. The paper gives the checker and cost model but no synthesis path; the synthesis is exactly what a schedule-binding stratum is for. **Gap: medium — checker exists on paper, backend hook is ours to build.**

- **Partition labels (g, s) as direct-manipulation knobs.** Trivially editable in principle (they're integers on axes); the hard part is that changing g/s changes the *derived* pseudocode diagram and performance model, so the editor must re-derive projections live. The paper's derivations are mechanical enough to automate (§5's step-by-step procedure + Appendix B configuration tables are begging to be a constraint solver). **Gap: small — this is the most immediately editable device.**

- **Rewrite-gesture editing.** Requires stable node identity *across* rewrites. On the napkin, each derivation step is a fresh figure; identity is in the reader's head. An editor must add a persistent-ID layer so that "attention's softmax box" is the same object before and after stream-partitioning — otherwise selection, undo, and provenance break. Also requires inverses: every relabeling must be un-appliable (they are, mathematically — all moves are equalities — but the artifact must track direction). **Gap: medium — pure engineering, but it's the load-bearing engineering.**

- **Pseudocode diagram ↔ loop code.** Forward direction is mechanical (paper proves it: expansion "is built from the recursively expanded definition of a streamable function"). Reverse direction (user edits the loop structure or the code, diagram updates) only works if edited code stays inside the streamable normal form — i.e., the code stratum at kernel level must be a *restricted language*, not arbitrary WGSL/CUDA. This matches our bijection-against-canonical-form stance: surface kernel code is checked metadata over the schedule term. Arbitrary hand-written kernels fall outside the bijection and must be admitted as opaque "schedule shadows" (consistent with our ~20% speedrun bin). **Gap: fundamental but honest — it draws exactly the line our design already drew.**

- **pyncd/tsncd as a base.** Would need: an inverse websocket channel; term-patch operations in pyncd; hit-testing in tsncd (the Render/Framework split localizes where); and escape from the Python-server topology entirely for our browser-native requirement — the algebra kernel must live in TS (tsncd currently only *replicates* the data structure for display; the constructors/theorems live in Python). **Verdict: steal the serialized term format and rendering rules, not the topology.**

---

## 5. VERDICT

**ADOPT (three things):**
1. **Schedule-binding = decoration, with theorem-shaped seam contract.** Formalize our stratum-3 bindings as (residency coloring × partition labels × divisibility constraints) over the *unchanged* semantic graph, validity checked by streamability/group-partition predicates, cost derived as Σ(color-changing edge sizes) with the H* = Σα·M^(−β) canonical form as the live napkin-math panel. This upgrades our seam from "testable contract" to "checkable theorem + derivable cost."
2. **Color-changing-wire cost readout** as the kernel view's primary visual device, and **pseudocode diagrams (columns = residency over time)** as the intermediate lens between semantic graph and kernel code.
3. **Edit-as-named-rewrite** for the kernel level: the editor's schedule gestures are exactly {group-partition, stream-partition, weave, fuse}, each legality-checked, each invertible, with persistent node identity across rewrites — making the edit history a correctness proof.

**REJECT / CONTRAST (two things):**
1. **Do not adopt NCD's columnar categorical syntax as the canonical surface.** It is optimized for kernel-sized expressions in papers; it has no containment/module-tree dimension, no scale story, and skips degrade it. Keep our hierarchical dataflow graph canonical; use axes-as-wires and weaving as a *zoom lens* on focused edges/ops only.
2. **Their three tiers are not our three strata.** Functional → leveled → pseudocode all live inside our strata 2–3; the paper is silent on training programs (optimizers, phases, config generators). No update to stratum 1 from this paper — our generator level remains motivated solely by the speedrun corpus evidence.

**BENCHMARK / CONTRACT ADDITIONS:**
- Sharpen the speedrun "kernel shadows" bin (~20%) with a concrete completeness criterion: *each kernel shadow should be expressible as a Napkin-style relabeling sequence of its declared math* (FlashAttention provably is; audit our others — fused LN, fused Adam, chunked reductions — and record which need moves outside the {group, stream, weave, fuse} vocabulary; any that do are genuine extensions to name).
- Add a kernel-view acceptance test: *the transfer-cost model must be derivable from the rendered schedule* (if the view can't produce the napkin math, something about the program is invisible — failing our completeness sense).
- Add to the stratum-2↔3 contract: bindings must be expressible as decorations (no graph surgery); any optimization that can't be is either a new semantic op or a bug in the contract.

**Citation neighborhood — three to know:**
- **"Weaves, Wires, and Morphisms" (arXiv 2604.07242, Abbott & Zardini 2026)** — the implemented sequel: pyncd (algebra, config, PyTorch compilation, graph conversion) + tsncd (TypeScript browser diagram renderer, JSON/websocket). The closest existing system to our thesis on earth — and still strictly one-way (algebra→diagram, algebra→code). The gap between it and us *is* our contribution; steal its serialized term format and TS rendering rules.
- **"Functor String Diagrams" (arXiv 2404.00249, Abbott et al.)** — the flexible diagram syntax underlying NCD; the formal account of the vertical-sections/columns layout if we adopt the zoom lens.
- **"Towards a Categorical Foundation of Deep Learning: A Survey" (arXiv 2410.05353)** — map of the broader formal-diagrams-for-DL space (Fong–Spivak backprop-as-functor, Cruttwell et al., string diagrams for DL); useful for knowing which formal devices exist before we invent notation. (Contrast tool the paper positions against: **Triton** — the "automated compiled methods have consistently lagged behind" foil; their answer is human-driven derivation with a calculus, ours is human-driven editing with a calculus.)

**One-line placement:** Napkin is the missing *theory* of our stratum-3 seam — a proof that "FlashAttention is a schedule of attention, never a different graph" — published with no editor, no round-trip, and no training-program scope; we should absorb its calculus as our kernel-level edit vocabulary and its cost readout as our kernel view, while keeping our own canonical form, our generator stratum, and the browser-editability requirement that this entire research line conspicuously lacks.

---

### Sources
- Paper: https://arxiv.org/abs/2412.03317 · HTML: https://arxiv.org/html/2412.03317v2 · TMLR review: https://openreview.net/pdf?id=pF2ukh7HxA
- Code: https://github.com/mit-zardini-lab/Napkin
- NCD (prior): https://arxiv.org/abs/2402.05424 · https://github.com/vtabbott/Neural-Circuit-Diagrams
- Sequel: https://arxiv.org/abs/2604.07242 · https://github.com/mit-zardini-lab/pyncd · https://github.com/mit-zardini-lab/tsncd
- Functor String Diagrams: https://arxiv.org/abs/2404.00249 · Categorical DL survey: https://arxiv.org/abs/2410.05353

*Caveat on sourcing: section details were extracted via fetched HTML summaries; the devices and quotes in §1–2 are corroborated across the abstract, two independent HTML passes, and the repos. The Weaves/Wires pipeline description is corroborated by both repo READMEs; its related-work list is lower-confidence (single lossy PDF pass).*
