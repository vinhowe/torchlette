# NCD Surface Spec — the authorable napkin notation

**Status:** DRAFT FOR REVIEW (Vin) · 2026-07-11
**Supersedes** the render layer of the ncd-spike (merged c26499dd; its term model, legality
predicates, cost derivation, projection, and findings all carry over unchanged).
**Normative references:** arXiv 2412.03317 Figures 7–12 (page renders studied 2026-07-11;
scratchpad npg-05/06.png), tsncd (mit-zardini-lab/pyncd) rendering. This spec exists
because the spike's surface — fixed grid, monochrome, no hierarchy — was "certainly
nothing like the actual NCD" (review verdict). The failure was brief-level: sequence-ui's
instrument-panel idiom was imposed on a mathematical notation, layout was not derived
from the algebra, and level-color — the notation's single most load-bearing device — was
dropped.

## 1. The visual language (normative, from the paper's figures)

1. **Arrays = columns of stacked axis labels** (math italic, overbar for weaved axes:
   q̄, x̄, d). The labels are the PRIMARY visual object. Tuple members separated by
   dashed horizontal rules. Column width/emphasis carries the "how much data" intuition.
2. **Memory level = translucent colored AREA, not stroke color alone.** A load to ℓ1
   wraps the column in a teal region (labeled "Load to ℓ1" on hover/selection); a save
   to ℓ0 wraps in magenta; sub-algorithm expansions group in soft yellow. The level
   graph itself (WGSL: global → workgroup-shared → [register]) renders as the paper's
   Figure-7 colored-node tree and is the app's level legend/config. Color assignments
   follow the paper (ℓ0 black/neutral, ℓ1 orange family) with our level graph's
   extensions; light/dark variants must preserve discriminability.
3. **Flow = smooth curves.** ℓ1-resident wires thread over/around function boxes as
   calligraphic curves (the weave). NO orthogonal grid routing. Layout is DERIVED from
   the term: horizontal adjacency = composition, vertical stacking = parallel (⊗),
   wire routing follows weave structure. Position is meaning; nothing is free-placed.
4. **Functions = clean boxes** with letters or glyph pictograms (◁ contraction; the
   softmax/online-softmax forms get their own glyphs). Authored/opaque kernels render
   as visually distinct sealed boxes (hatched border) — tunable, not enterable.
5. **The cost instrument = column-aligned table ABOVE the diagram** (the paper's
   Figure 9/11/12 form): M_ℓ and H_ℓ rows whose entries sit over the columns they
   measure, plus a Cumulative cell. Live: entries update during gestures, changed cells
   pulse. This REPLACES side gauges for the diagram view (sequence-ui panels remain for
   history, contracts, and non-diagram chrome only).
6. **Partition labels** (g_a, s_a) annotate the axis in place, small, at the level
   transition. **Divisibility superscripts** render on axis sizes.
7. **Rewrites read as diagram ≡ diagram**: applying a lemma or expanding a partition
   shows the equivalence form transiently (before ⇌ after), which is also the undo
   affordance's visual.
8. **Typography:** math italics for sizes/axes, generous whitespace, thin rules, ink-
   weight hierarchy (data flow heavy, annotation light). This surface gets ITS OWN
   design tokens — sequence-ui explicitly does NOT apply inside the diagram canvas;
   it frames it.

## 2. Interaction (the Factorio-quality layer, ON the faithful notation)

- Pannable/zoomable canvas; the diagram is the world, not a widget in a form.
- **Gestures are relabelings made tactile:** drag a g_a/s_a chip from the palette onto
  an axis (tile/stream); paint a level region across a boundary (fuse/unfuse — the
  recolor move); drop a lemma card onto a matching box (admitted rewrites). Hover
  previews the post-gesture cost row; commit animates the ≡ transition.
- **Refusals jam visibly**: an illegal drop bounces with the reason attached at the
  point of failure (streamability head/body missing → the box that lacks it is
  highlighted, not a toast).
- Streaming, when present, may animate (data batches flowing through the ℓ1 region) at
  a subtle default with a motion toggle.
- Selection opens the sequence-ui side panel (decorations detail, history, contract) —
  chrome and canvas are distinct layers with distinct design systems.

## 3. Acceptance (checkable, not vibes)

1. **Side-by-side reproduction:** the app renders the paper's Figure 8 (softmax-
   contraction with load/save regions), Figure 9 (same + column-aligned cost table),
   and Figure 11 (group-partition relabeling) from our term model such that a reader
   of the paper recognizes them immediately. Reviewed as literal side-by-side
   screenshots in the report.
2. The FA-by-gestures walkthrough (spike acceptance, carried over) replays on the new
   surface with the ≡ transitions and live column costs.
3. Layout is term-derived: no persisted x/y coordinates anywhere in the document model.
4. Vin-in-the-loop iteration: this build is NOT one-shot — checkpoint screenshots at
   (a) static Figure-8 fidelity, (b) first gesture working, (c) cost table live, each
   reviewed before proceeding.

## 4. Non-goals here

Whole-plan scale rendering (islands-lane remains the top-of-zoom view; this surface is
the intra-island/kernel magnification); the measured tier (contract unchanged from the
workbench); engine round-trip (term model unchanged; the engine channel is the sibling
work). The spike's inspector view may remain as a debug panel; it is no longer the
primary surface.

## 5. The game loop (added 2026-07-11 after play review — this section is why the surface exists)

Review verdict on the loopless surface: "not at all play inducing... feels arbitrary...
online softmax feels like random magic... notation feels dense... no idea what [derive
FA] does." Diagnosis: sandbox without the game. The game was already designed — the
intrinsic ladder (corpus: wgsl-cuda-containment.md, rung table + 12-exercise sequence).
Normative here:

1. **Every level is a want**: a baseline diagram + its cost + a TARGET (H/M budget).
   The column-aligned cost table doubles as the score; the gap to target is always
   visible. No level, no diagram — the cold start is a level select, not a canvas.
2. **Vocabulary is rung-gated** (progressive disclosure): a level exposes ONLY the
   notation devices and gestures its rung needs. Exercise 1 = wires + colors + recolor.
   Density is a failure of gating, not a property of the notation.
3. **Lemmas are EARNED at walls, never offered as buttons**: the player attempts the
   natural move, the jam explains the obstruction in-place, and only then does the
   lemma become available — with its carried state (Welford's running (μ, M2); online
   softmax's (m, ℓ) + correction factor) INSPECTABLE in the diagram after application.
   Exercise 3 (Welford) teaches lemma-as-concept small before exercise 8 (the wall).
4. **The FA capstone is reached, not clicked**: the toolbar button dies; FA is the
   final level of the slice, and by the time the player arrives every move in it is
   one they have already made once.

v1 slice = exercises 1, 3, 8, 9 (fuse-the-chain; LayerNorm→Welford; softmax lemma
wall; FA capstone). Costs are napkin-static (score = H/M vs target); the measured tier
joins when the engine bench lands. Completion state per level: target met → the level
select shows the earned cost + the move count (the climbing-trace ledger).
