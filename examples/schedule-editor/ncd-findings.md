# Editable NCD spike findings

These findings come from making the “FlashAttention on a Napkin” notation an
authorable, round-trippable browser term rather than a one-way figure. They are
consumer-side proposal findings, not claims about the eventual engine object.

## F1 — a wire color needs a sampled path, not one residency field

A value such as the attention score matrix is lower-level inside QK, returns to
ℓ0 for an island boundary, then enters the lower level again for softmax. One
`wire.level` cannot represent `ℓ1→ℓ0→ℓ1`. The proposal stores residency samples
at named columns. Recoloring changes one sample, which makes fusion local and
invertible. A production term should clarify whether samples, intervals, or an
explicit level-path morphism is canonical.

## F2 — NCD's H rule omits whole-dispatch multiplicity at the UI seam

The paper says H is the sum of arrays changing color. After partitioning, does
one count the lower-level tile once (the local napkin step), or multiply it by
all group/stream iterations (whole-dispatch traffic, including rereads)? The
spike uses the local-column interpretation: a color change is charged the
partitioned lower-level array size. That makes the diagram and number visibly
local, but it is not a replacement for whole-dispatch traffic. The canonical
cost object needs both `perStep` and `total(multiplicity)` views.

## F3 — M requires explicit column liveness

`M_ℓ = max column width` is only computable once the term states which arrays
are live in every column. Semantic boxes and edges alone do not select a
pseudocode expansion or lifetime overlap. The proposal makes columns and
residency samples first-class diagram data. This pressures the claim that the
pseudocode view is entirely derived: either column liveness is derivable from a
canonical normal form, or the cost cannot be read without storing projection
choices.

## F4 — global semantic-axis labels are a deliberate deviation

The notation places `g_a`/`s_a` on array axes. The same semantic axis appears on
Q, scores, probabilities, and O. Allowing independent labels produces incoherent
“half tiled” schedules and makes gesture targets ambiguous. This spike binds a
partition once to a semantic axis ID and renders it on every carrying wire. If
per-array partitioning is required later, compatibility equations must be part
of legality rather than left to convention.

## F5 — head/body strings are evidence-shaped, not machine proofs

The streamability predicate is machine-executed over declared decompositions,
but `head` and `body` are still descriptive strings. The checker proves that
every relevant box *declares* the axis, not that `F/B` recomposes to the same
function. Real theorem-shaped legality needs typed head/body terms with a
recomposition law, or proof-carrying references to an admitted catalog.

## F6 — stateless mapping and stateful streaming share one vocabulary awkwardly

QK exposes x-streamability by “append one K block of scores”; PV exposes a true
accumulator; online softmax exposes `(m,l,o)` rescaling state. All satisfy the
composition theorem, but their heads/bodies have different algebraic shapes.
The schema needs a discriminated decomposition family such as
`map/concatenate`, `monoidal-reduce`, and `state-machine`, not one free-form
record.

## F7 — the admitted lemma changes more than a decoration

Online-softmax rescaling changes the softmax box's intermediate state and body,
not only a wire label. Yet FlashAttention must remain the same semantic
attention algorithm. The spike records an explicit admitted-lemma rewrite on
the box and keeps it in proof history. The canonical layering must decide
whether this is an algorithm-term equivalence, a box implementation choice, or
a schedule decoration; calling all three “the same graph” is insufficiently
precise.

## F8 — fusion legality is local only for single producer/single consumer wires

The recolor checker finds the intermediate's producer and consumer and looks for
a common decomposed axis. Branching, multiple consumers, tuple outputs, or a
wire consumed at both ℓ0 and ℓ1 make “remove this boundary” a region-convexity
question. The maximal-sub-ℓ0 island definition needs graph connectivity and
convexity over all uses, not this spike's linear connection check.

## F9 — external transfers and island boundaries are visually the same color change

Recoloring Q's ℓ0 input to ℓ1 is a required load, not fusion. Recoloring the
middle score sample removes a kernel boundary. The gesture checker distinguishes
an interior `ℓ1→ℓ0→ℓ1` sample from an endpoint, but the notation itself does not
name those roles. Authoring needs visible `external-transfer` versus
`materialization-boundary` hit targets.

## F10 — WGSL's useful level graph is not a simple two-color chain

The spike uses ℓ0 global and one lower ℓ1 color, matching the paper's minimal
notation. Real WGSL scheduling distinguishes storage buffers, workgroup shared,
invocation-local/register-like values, uniforms, and sometimes subgroup
cooperation. Some transfers skip levels, and register allocation is not under
direct authority. A future schema needs a device/realizer level graph; blindly
adding more colors would imply authority WGSL does not possess.

## F11 — divisibility superscripts need constraint composition

An axis may simultaneously require vec4, coalescing, thread-tile, and tensor
core multiples. The renderer can show one `|k` superscript, while legality needs
the LCM and a list of contributing reasons. The proposal schema permits several
records but the spike reads the first for display. Canonical constraints should
retain causes and expose a derived LCM.

## F12 — projection loop order is not determined by decorations

Matmul has group partitions on m and n plus a stream partition on k. Either m→n
or n→m loop nesting realizes the same decorated term and changes locality. The
spike uses decoration-array order, which is deterministic but not semantic.
This is a genuine projection ambiguity: partition labels alone do not derive a
unique loop nest. A canonical ordering rule or an additional ordering
decoration is required.

## F13 — the projection cannot derive cooperative thread roles

The term derives group/stream loops, head initialization, and body calls, but
not which invocation loads which element, barriers, vector lanes, or workgroup
shape. Those facts were explicit in the prior ScheduleState skeleton. If
skeleton is only a projection, the NCD term must gain hardware mapping labels;
otherwise the projection stops above executable WGSL.

## F14 — “term → diagram → term” cannot mean parsing pixels

The round-trip test maps a canonical term to a lossless diagram model with
persistent IDs and back. The DOM is a projection of that model; it is not parsed
to recover algebra. Bidirectionality therefore means gestures emit typed term
patches from hit targets. Any renderer optimization that drops invisible fields
is a schema bug even if the picture is unchanged.

## F15 — tuple and parallel notation strains quickly

Dashed Q/K/V tuple grouping is legible at kernel scale. Repeating every wire
bundle across seven residency columns already produces a wide instrument. NCD
has no native level-of-detail or large-graph layout answer; it remains suitable
as an island zoom lens, not a replacement for the model/island hierarchy.

## F16 — bytes versus elements must be explicit

The paper's array-size algebra is naturally in elements, while hardware bills
are bytes and quantization changes both size and constraints. The instrument
reports both. H/M identity should remain element-typed plus dtype metadata;
converting to bytes is a realizer/device projection, not a silent unit choice.

## F17 — the honest FlashAttention path begins with a refusal

Dragging `s_x` onto naive attention is correctly refused: ordinary softmax has
no head/body decomposition. The admitted rescaling lemma must precede fusion
and streaming. Thus the executable sequence is `lemma → recolor → recolor →
group → stream`, not a pure three-gesture derivation. This is not a UI nuisance;
it is the exact boundary between structural relabeling and algebraic knowledge.

## F18 — pyncd/tsncd term compatibility is not yet a free import

pyncd's current terms include categorical products, compositions, rearranges,
weaves, and affine stride mappings. The spike's kernel-local schema uses flat
boxes/wires/axes plus explicit columns so H/M and editing are direct. Importing
pyncd terms would require a normalization pass that preserves weave/broadcast
semantics and stable IDs. Copying the one-way serialized format verbatim would
not by itself produce an editable cost-complete term.
