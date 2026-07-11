# NCD surface review: paper Figures 7–12 versus checkpoint A

Normative page renders are retained beside the application captures so the
comparison does not depend on an external URL.

| Paper reference | Term-derived application surface |
| --- | --- |
| [Page 5: Figures 7–9](./reference-page-05.png) | [Figure 8 diagram layer](./checkpoint-a-figure-08.png) · [Figure 9 cost + diagram](./checkpoint-a-figure-09.png) |
| [Page 6: Figures 10–12](./reference-page-06.png) | [Figure 11 tiled-matmul term](./checkpoint-a-figure-11.png) |

## Honest verdict

The new surface is recognizable as the same notation family. Arrays are now
the diagram's primary marks: stacked italic axis labels and thin rules rather
than cards in a grid. Teal load regions, magenta save regions, yellow
sub-algorithm regions, dashed tuple separators, sparse function glyphs, smooth
composition curves, overbars, in-place partition labels, and the two-row cost
table reproduce the paper's visual grammar. The cost table and diagram share
the exact derived semantic-column widths. The canvas has its own paper/ink
tokens and remains discriminable in dark mode.

It is not a facsimile. Important remaining deltas are:

- The carried attention term is the full `QKᵀ → softmax → PV` path, not the
  paper's smaller `SoftMax → contraction` pedagogical term, so the application
  has more arrays and columns than Figure 8.
- The current term has no explicit weave/rearrange node. The renderer infers an
  overbar when a semantic axis occurs on multiple wires and routes smooth cubic
  curves; it cannot reproduce the paper's exact crossing/weave topology.
- Function glyphs use browser typography (`◁`, `σ`, `↻`) instead of the paper's
  TeX/TikZ glyph construction.
- The table reports evaluated element counts rather than the paper's symbolic
  expressions such as `q̄x̄ + x̄d`.
- The Figure-11 view shows in-place `gₐ` labels and partition-sized lower-level
  axes, but does not recursively expand the yellow replicated sub-algorithms.
- Tuple membership is visible as dashed separators, but the term does not
  encode concatenation/weaving operators strongly enough to derive every
  connector dot and circle used in Figures 8 and 10.

Those are term-expressivity findings, not reasons to persist layout coordinates.
All positions in this checkpoint are derived at render time from semantic
column order, wire order, box composition, tuple membership, and decorations.

## Interaction and walkthrough checkpoints

- Checkpoint B: [paint/cost preview](./checkpoint-b-mid-gesture.png),
  [streamability refusal jam](./checkpoint-b-refusal-jam.png), and
  [diagram ≡ diagram rewrite](./checkpoint-b-equivalence.png).
- Checkpoint C: FlashAttention replay
  [start](./checkpoint-c-fa-start.png) → [mid-fusion with pulsing cost cells](./checkpoint-c-fa-mid.png)
  → [fused/tiled/streamed end](./checkpoint-c-fa-end.png), plus the
  [dark-token discrimination check](./checkpoint-c-dark.png).
- Checkpoint D: [persistent gesture palette](./checkpoint-d-palette.png),
  [live valid/invalid drop targets](./checkpoint-d-valid-targets.png), and the
  [`?` keyboard reference](./checkpoint-d-help.png). See
  [CHECKPOINT-D.md](./CHECKPOINT-D.md) for controller conventions and the
  aggressive-user protocol.
- Checkpoint E: the [four-level intrinsic-ladder game loop](./CHECKPOINT-E.md),
  including goal, wall, lemma-inspection, and completion captures.
