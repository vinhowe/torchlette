# Model Editor v1 Charter

**Status:** DRAFT FOR REVIEW (Vin) · 2026-07-11
**Basis:** Vin's interview answers (2026-07-10) + Q4 ruling (LIVE, 2026-07-11).
**Sibling design doc:** docs/schedule-state-design.md — shared spine, see §8.

## 0. Declaration (one sentence)

A running model is a first-class editable object: a canonical hierarchical dataflow
graph with generator/instance structure, stable node identity, and function-preserving
weight semantics — edited live in the browser, with valid PyTorch as an emit target.

## 1. Stance (the selection pressure)

This project is **open-endedness-first**, not product-backchained: negentropy is spent
like a VC with high risk tolerance; the texture is math research (build the object, follow
the feedback loops, harvest unintuitive insight), not roadmap execution. The pressure that
replaces "audience" is **representability stress**: can the object hold X without
deformation? The standing stress axes:

1. **Speedrun-completeness** — the 84-record nanogpt-speedrun catalog; v1 targets bin 1
   (architecture edits, ~47% of records).
2. **Flash-representability** — discharged by the schedule-state campaign's P2 acceptance
   (the flashattention derivation as moves+lemma).
3. **Distributed-training representability** — v1 writes the stress-test DOC only
   (can the object hold DiLoCo/FSDP-shaped structure?); no implementation.

**Object-pole primacy:** neither SEEING (visualization) nor AUTHORING (construction) leads;
both emerge from getting the object right. Decision rule for disputes: improve the object,
not either pole's UI.

## 2. The object

- **Canonical hierarchical dataflow graph.** One graph; modules are hierarchy, not a
  separate representation. The three shipped model implementations (GPT-2, Qwen3,
  Gemma-2) become **generator definitions**; checkpoints BIND to generators
  (safetensors ingest ≈ free — the staged-adapter decision). Arbitrary-source ingest is
  REFUSED in v1, in writing, here.
- **Generator vs instance with declared overrides.** Editing attention in the generator
  updates every instance; a per-index override (layer 13 differs) is a declared exception —
  the Figma component/override model. Speedrun bin 1 needs BOTH (activation swap =
  generator edit; alternating windows / layer drops / U-net pairs = per-index predicates).
- **ONE predicate language** for override selection, shared verbatim with optimizer
  param-groups and (later) structural schedules: "even layers", "2D params", "layers 3..7".
  Three consumers, one grammar — a second predicate syntax anywhere is a review-blocking
  defect.
- **Stable UIDs + provenance ledger.** Path-keys break under structural edits → silent
  wrong-weights (the worst failure class). Every node carries a UID; every edit appends a
  ledger entry. The ledger triple-pays: undo/redo, lineage ("derived from X via edits" —
  Menagerie #52's sharing substrate for free), and meaningful model-diff. Cost accepted:
  path↔uid translation at the PyTorch boundary.
- **Per-stratum bijection contract** (Q3): strata 1–2 (generator, semantic graph) —
  PyTorch has the concepts → true bijection, cheap emit. Stratum 3 (schedules) — PyTorch
  cannot represent its own schedules → canonical-form-with-emit; nothing to biject
  against. Below (islands/kernels) — opaque-paired. The contract is stated per stratum,
  never globally.

## 3. Weight semantics (function preservation at apply time)

Every structural edit changes model behavior by ZERO at the moment of application;
training then grows into the new capacity. The v1 table:

| Edit | Semantics |
|---|---|
| duplicate | SHARE (weight tying; reversible) |
| insert (block/branch) | ZERO-INIT the output projection (residual-safe; Net2Net/speedrun idiom) |
| untie | COPY at the untie moment |
| widen | **REFUSED in v1** — no canonical preserving choice; refusal with reason in the UI |

## 4. Q4 ruling: LIVE

The editor holds a **running model**: weights resident in-tab, generation streaming,
activations/SAE features paintable on the graph *because it is executing* (the #88
demo stack is the substrate — Gemma-2 + Gemma Scope shipped, steering verified).
The feedback loop IS the product thesis: touch the model → watch behavior and internals
shift → unintuitive insight.

**Edit-apply mechanism:** edits apply at generation boundaries via the tape's
invalidate→re-record path — a graph edit changes the plan fingerprint, the next
generation re-records. No execChain surgery required for v1 (the never-landed prototype
stays dead); weight carry-over across the edit follows §3's table. Structural-edit
latency budget = one re-record (~seconds), stated in the UI, not hidden.

## 5. v1 fence

**IN:** canonical graph + generator/overrides + predicate language; bin-1 architecture
edits (skips, gates, norms, activation swaps, #64 attention modifiers — softcap, sliding
window, custom score/mask mods); three models opening FROM CHECKPOINTS; live in-tab
inference with activation/SAE painting; function-preserving weight semantics; UID +
ledger; PyTorch EMIT (bijective strata only).

**OUT (each with its parking spot):** training-program layer (second substrate — future
charter); phase transitions (speedrun bin 2/3); distributed (stress DOC only, axis 3);
kernel stratum (sibling: schedule-state campaign — the shared spine keeps zoom-down
continuous); widen; arbitrary-source ingest; torch.export ingest (deferred);
lemma derivation (see sibling doc).

## 6. Three-month claim (the falsifiable bet)

1. GPT-2, Qwen3-1.7B, Gemma-2-2B open at generator level from checkpoints, in-tab.
2. **Six named speedrun techniques** applied to a live GPT-2 as edits (candidate six:
   skip-connection reroute, QK-norm, ReLU², zero-init residual projections, untied
   embeddings, attention-window predicate) — each function-preserving at apply, each a
   ledger entry, each undoable.
3. The edited model EMITS valid PyTorch that runs.
4. One **train-as-evidence** run: an edited model trains (browser or Dawn) and the edit's
   effect on the loss curve is visible vs the unedited control — the loop demonstrated
   end to end.

Miss the claim → the retrospective names which object decision was wrong, not which
deadline was tight.

## 7. Phases

- **M0 — object spec + predicate language** (design round, this charter's review):
  graph schema, generator/override representation, UID/ledger format, predicate grammar,
  PyTorch emit mapping for strata 1–2. Paper artifact: the distributed-representability
  stress DOC (axis 3).
- **M1 — read-only:** three checkpoints open; generator/instance hierarchy rendered
  (sequence-ui); no edits. Gate: parity — the opened graph re-emits the SAME forward as
  the shipped implementation (logit parity harness reused).
- **M2 — live:** in-tab inference wired (the #88 worker substrate); activation/SAE
  painting on the rendered graph. Gate: painted values match the demo's readbacks.
- **M3 — edits:** bin-1 moves + §3 weight semantics + ledger + undo. Gate: every edit
  function-preserving at apply (logit diff == 0 pre-training) + re-record applies it live.
- **M4 — emit + the speedrun six.** Gate: emitted PyTorch runs under torch; six
  techniques as ledger scripts checked into tools/.
- **M5 — train-as-evidence.** Gate: claim item 4.

## 8. Shared spine with the schedule editor

One spine, two zoom regimes: **model graph → module → island lane (sol's editor,
shipped 30a2326d) → intra-island schedule (sibling P3)**. Land once, share: UID +
provenance-ledger idiom; identity coordinates in fingerprints; sequence-ui; the
capability-profile/realizer idiom (PyTorch emit is a realizer with a capability profile —
same registry shape as WGSL/Triton in the sibling doc); legality-as-refusal-with-reason
UX. Neither editor blocks the other.

## 9. Risks

(a) **Generator inference from checkpoints** — we do NOT infer architecture from weight
names; our three implementations ARE the generators, checkpoints bind to them (that's why
arbitrary ingest is refused). (b) **Live-edit memory** — re-record under 5GB residency on
16GB Macs; the browser residency lessons (f16 tables, row-block streaming) apply; budget
measured in M2. (c) **Scope creep into the training-program layer** — fenced OUT; the
predicate language is the only shared artifact. (d) **Ledger/undo divergence from tape
state** — an edit's ledger entry and its fingerprint change must be one transaction;
tested in M3's gate. (e) **Emit drift** — M1's parity gate re-runs on every M3 edit class
(emit the edited graph, run under torch, compare against the in-tab edited model).
