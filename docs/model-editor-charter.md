# Model Editor v1 Charter

**Status:** v2 (amendment round 1) · 2026-07-11 — rewritten to the rulings in
`docs/design-amendment-round-1.md` (ratified by Vin; disposes red-team R11–R19, R26 +
findings). The v1 draft (2026-07-11) is superseded; each § carries the ruling it applies.
**Basis:** Vin's interview answers (2026-07-10) + Q4 ruling (LIVE, 2026-07-11).
**Sibling design doc:** docs/schedule-state-design.md — shared spine, see §8.
**Precedents cited:** docs/ownership-derivation-design.md (derived-fact / assert-agreement
discipline), CLAUDE.md's fence-gated destruction + `canRecycle` house law (R17).

## 0. Declaration (one sentence)

A running model is a first-class editable object: a canonical hierarchical dataflow graph
with generator/instance structure, stable node identity, a provenance spine, and
function-preserving weight semantics for the edits that HAVE a constructive identity —
edited live in the browser, with an injective emit into a defined PyTorch subset as an emit
target.

## 1. Stance (the selection pressure)

This project is **open-endedness-first**, not product-backchained: negentropy is spent like
a VC with high risk tolerance; the texture is math research (build the object, follow the
feedback loops, harvest unintuitive insight), not roadmap execution. The pressure that
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
  separate representation. The three shipped model implementations (GPT-2, Qwen3, Gemma-2)
  become **generator definitions**; checkpoints BIND to generators through the checkpoint
  manifest (§2.3, R18) — not "safetensors ingest ≈ free". Arbitrary-source ingest —
  reconstructing an UNKNOWN architecture by inference from weight names/shapes — is REFUSED
  in v1, in writing, here: that inference is heuristic and fails SILENTLY (wrong weights
  bound to wrong nodes, the worst failure class). A new model family is supported by writing
  its generator (~a day), never by inference.
- **Generator vs instance with declared overrides.** Editing attention in the generator
  updates every instance; a per-index override (layer 13 differs) is a declared exception —
  the Figma component/override model. Speedrun bin 1 needs BOTH (activation swap = generator
  edit; alternating windows / layer drops / U-net pairs = per-index predicates).

### 2.1 The typed predicate AST (R12 — a named M0 artifact)

**Ruling R12 + F5 (accepted as proposed):** override selection uses ONE typed predicate AST
with explicit leaf domains and shared combinators — NOT an untyped string grammar. It is
shared VERBATIM (the same AST type, serialization, and type-checker) with optimizer
param-groups and, through the sibling doc, structural schedule selection. Model-context leaf
domains: model instances, parameters (rank, shape, name-pattern), semantic ops, islands.
Schedule-context leaves (loops, values, roles, origin sets) are the SAME AST's other leaves
(sibling §2.5). Serialization + type errors are gated in M0.

Why this is a rule, not a preference: nobody DECIDES to diverge — it happens by drift, one
convenient shortcut at a time, until "even layers" exists in three dialects and an edit
script can't be reused as a param-group spec. Drift is stopped only by a typed AST with one
serializer that a second selector syntax would fail to type-check against (R12 — "verbatim"
was an assertion in v1; here it is a shared type + a gate).

### 2.2 The provenance spine (R11 — a named M0 artifact)

**Ruling R11 (accepted as proposed):** the v1 "shared identity spine" was three incompatible
schemes with no bridge (model nodes have persistent UIDs; live islands use final-plan integer
positions; ScheduleState had no entity UIDs). The spine is a **revisioned many-to-many map**:

```
ModelNodeUid → SemanticOpUid → (planFingerprint, nodePosition, outputIndex) → IslandUid
             → ScheduleEntityUid
```

- **Origin sets preserved through fusion and lowering** (one model node may lower to many
  LazyIR nodes; one fused node may carry many origins). Views may be outside islands; the map
  carries that.
- **Ledger operands are UIDs + a base revision**, NEVER array positions or hashes (R11 — a
  `scheduleHash` is content identity, not operand identity for edits; plan positions can change
  on re-record even when a model UID survives).
- **Invalidation and rebase are specified** (§4, §5): when any mapping is regenerated
  (re-record), only the realization arm of the map is invalidated; the semantic arm (UIDs +
  revisions) is stable.
- This is the map on which "click model node → island → schedule object", model-diff, undo,
  and provenance are ALL sound — the zoom continuum's every seam has an owner (R11).

### 2.3 Checkpoint manifest (R18 — a named M0 artifact) + stable UIDs + provenance ledger

Path-keys break under structural edits → silent wrong-weights (the worst failure class).
Every node carries a UID; every edit appends a ledger entry (§5, split into two histories).

**Ruling R18 + R26 (accepted as proposed):** a checkpoint outlives generator code; renamed/
split params, changed rotary convention, tied-vs-untied embeddings, added buffers, dtype/
quant changes can all preserve plausible names/shapes while changing semantics (the refusal
of unknown-architecture ingest does NOT protect a KNOWN family from version drift — R18). So
binding requires a **checkpoint manifest**:

```
CheckpointManifest = {
  familyId, generatorSchemaDigest,          // fail-closed on unknown versions
  params: [{ paramUid, expectedPath, shape, dtype, aliasGroup, semanticRole }],
  migrationChain                             // known-version migrations, each an auditable ledger entry
}
```

- ALL fields validated + known-logit fixtures checked BEFORE opening; **fail-closed on unknown
  versions** (R18 — silent wrong-weight binding is the charter's own worst failure class).
- Migrations produce a NEW auditable ledger entry (semantic history, §5).
- The ledger triple-pays: undo/redo, lineage ("derived from X via edits" — Menagerie #52's
  sharing substrate for free), and meaningful model-diff. Cost accepted: path↔uid translation
  at the PyTorch boundary.

### 2.4 Per-stratum contract (Q3 — with R16's injective-emit correction)

- strata 1–2 (generator, semantic graph) — PyTorch has the concepts → **injective emit into
  a defined subset** (R16 — NOT "true bijection with PyTorch"; §3.2).
- stratum 3 (schedules) — PyTorch cannot represent its own schedules → canonical-form-with-emit;
  nothing to biject against.
- below (islands/kernels) — opaque-paired.

The contract is stated per stratum, never globally.

## 3. Edits: the taxonomy split (R14 + R15) and weight semantics

**Ruling R14 + R15 (accepted as proposed):** the v1 table ("every structural edit changes
behavior by ZERO at apply") is FALSE for the candidate six — QK-norm, ReLU², an attention
window, and a skip reroute are NOT function-preserving for a general trained model, and weight
copying cannot make those operators identical. Edits split into two kinds:

### 3.1 `functionPreservingMorph` — rewrite schemas with identity equations

A morph is defined as a **REWRITE SCHEMA, not a verb** (R15): exact matched subgraph,
replacement graph, parameter/alias map, optimizer-state map, an **identity equation** the
mapping satisfies, and inverse preconditions. "duplicate" and "insert" without a named schema
are REFUSED (R15 — sharing weights says nothing about how the duplicate is CONNECTED; executing
a block twice in series, or a duplicate branch, generally changes the function).

```
FunctionPreservingMorph = {
  matchSubgraph, replacementGraph,
  paramMap, aliasMap, optimizerStateMap,
  identityEquation,        // the ℝ-valued equality the mapping satisfies (proof obligation)
  inversePreconditions     // when the morph can be undone without information loss
}
```

Only TWO of the candidate six are morphs (R14/R15 rulings):

| Morph | Schema (identity equation) |
|---|---|
| **untie-by-copy** | matched: one shared param used at N sites; replacement: N independent params, each COPYING the shared value at the untie moment; identity equation: the copies equal the original ⇒ forward unchanged. Inverse precondition: FAILS once either copy has diverged under distinct optimizer intent (R15 — "reversible" fails after divergence). |
| **zero-gated residual insertion** | matched: a residual stream at point p; replacement: p + f(x) where f's OUTPUT PROJECTION is zero-init; identity equation: zero output ⇒ residual unchanged. Precondition: the inserted branch must not affect normalization / control flow / side effects BEFORE that projection (R15 — a generic inserted block may; only a residual-safe topology preserves). |

### 3.2 `behaviorChangingEdit` — preview/commit with measured before/after

Everything else in the candidate six (QK-norm, ReLU², attention-window predicate, skip-reroute)
is a `behaviorChangingEdit` (R14): applied with an explicit **preview/commit** and a MEASURED
before/after (do NOT call activation/window/reroute edits preserving; do NOT test a degenerate
input where the changed path happens not to affect logits — R14). The three-month claim (§6) is
restated accordingly: these edits are demonstrated as behavior-CHANGING, with the change made
visible, not hidden behind a false zero-diff gate.

```
BehaviorChangingEdit = {
  editSchema,              // the graph rewrite (still a named schema, R15)
  preview, commit,        // two-phase
  measuredBeforeAfter     // logit/activation deltas on the R26 probe set, not one probe
}
```

### 3.3 The weight-semantics table (v2 — split by kind)

| Edit | Kind | Semantics |
|---|---|---|
| untie | morph | COPY at the untie moment (identity equation: copies == original); inverse fails after divergence |
| insert (zero-gated residual) | morph | ZERO-INIT the output projection (identity equation: zero output == residual unchanged); residual-safe topology REQUIRED |
| duplicate | REFUSED without a schema | "SHARE" alone does not define connection (R15); a duplicate needs a named rewrite schema (series/branch/tied-replacement each change the function differently) |
| QK-norm, ReLU², attention-window, skip-reroute | behaviorChanging | preview/commit + measured before/after (NOT function-preserving — R14) |
| widen | **REFUSED in v1** | no canonical preserving choice; refusal with reason in the UI |

## 4. Live edits as transactions (R17 — revision / CAS / epoch / quiesce)

**Ruling R17 (accepted as proposed):** "edits apply at generation boundaries" is not a
transaction or a concurrency protocol — the boundary was undefined, two browser edits can race
each other and a streaming generation, and "every cache misses naturally" is FALSE as a safety
argument (module-local pipeline, tuning, bind-group, config-buffer, and authored-kernel caches
are not all keyed solely by plan fingerprint; invalidation can occur while GPU work still owns
buffers). The editor holds a **running model** (Q4: LIVE — weights resident in-tab, generation
streaming, activations/SAE features paintable BECAUSE it is executing; the #88 demo stack is the
substrate — Gemma-2 + Gemma Scope shipped, steering verified). Live edits are TRANSACTIONS:

```
EditTransaction = { schemaVersion, baseModelRevision, opId }
```

- **commit by CAS at a defined generation epoch.** The generation epoch is the named boundary
  (not "next token / end of prompt / queue flush" ambiguity — R17): a CAS at the epoch commits
  iff `baseModelRevision` is still current.
- **an immutable model snapshot per generation** — one generation reads one frozen revision;
  concurrent edits queue or cancel (R17).
- **quiesce / fence BEFORE state demotion, defer buffer destruction** — the existing house law
  is CITED (CLAUDE.md: GPU buffer destruction is fence-gated, never immediate; route all
  destruction through `deferredDestroy`; `canRecycle` before any cache reuse). The invalidate→
  re-record path must quiesce before demotion, exactly as the training loop's markStep does
  (R17 — invalidation while GPU work owns buffers is use-after-submit).
- **publish-after-validation:** the new revision is published only after graph validation,
  weight mapping (§3), re-record, and a smoke execution ALL succeed. **Stale edits are rejected
  with the current revision** (R17), for the client to rebase.

**Edit-apply mechanism (unchanged in spirit, now transactional):** edits apply at the generation
epoch via the tape's invalidate→re-record path — a graph edit changes the plan fingerprint,
every plan-fingerprint-keyed cache misses, and the next generation records fresh. The caches NOT
keyed by plan fingerprint are handled by the quiesce+defer+CAS protocol above (R17 — "every cache
misses naturally" is not a safety argument on its own). The rejected alternative ("execChain
surgery", prototyped once, never landed): mutate the live executor's structures in place for
instant mid-generation effect — which requires hand-patching every cache coherently, i.e.
re-opening the frozen-stale-state disease surface. v1 buys ~seconds of re-record per structural
edit (stated in the UI, not hidden) and gets staleness-impossible-by-construction. Weight
carry-over across the edit follows §3.

## 5. Ledger split: semantic vs realization histories (R19)

**Ruling R19 (accepted as proposed):** the islands contract says plan-change events invalidate
the editor's undo stack, while v1 sold ONE ledger across zoom levels — contradiction. The ledger
SPLITS into two histories:

- **Semantic edit history** — IMMUTABLE; UID operands (never plan positions); inverse payloads;
  RETAINED weight/optimizer snapshots WHERE an inverse loses information (undoing an insertion
  after training discards learned weights; undoing an untie must decide which diverged copy
  survives — so the snapshot is retained, R19). This history survives re-record.
- **Realization history** — EPHEMERAL; INVALIDATED on re-record (schedule operands identified by
  plan position are stale after re-record — R19).

**Cross-stratum undo** rebases against UIDs or REFUSES with a stable conflict reason (R19): a
generator edit undone after per-index overrides were created against the newer definition must
reconcile via override precedence + three-way rebase, or refuse. Only realization commands are
invalidated on re-record; semantic undo regenerates against UIDs. Snapshot retention has a
declared budget (§9 risk b).

## 6. Three-month claim (the falsifiable bet)

1. GPT-2, Qwen3-1.7B, Gemma-2-2B open at generator level from checkpoints (via the §2.3
   manifest), in-tab.
2. **Six named speedrun techniques** applied to a live GPT-2 as edits (candidate six:
   skip-connection reroute, QK-norm, ReLU², zero-init residual projections, untied embeddings,
   attention-window predicate) — each a ledger entry, each undoable. Of these, **two are
   function-preserving morphs** (untied embeddings = untie-by-copy; zero-init residual
   projections = zero-gated residual insertion) with a zero-diff identity equation; **the other
   four are behavior-changing edits** demonstrated with measured before/after (§3.2, R14 — the
   v1 "each function-preserving at apply" claim was impossible and is retracted).
3. The edited model EMITS valid PyTorch (in the defined subset, §3.2/R16) that runs.
4. One **train-as-evidence** run: an edited model trains (browser or Dawn) and the edit's effect
   on the loss curve is **visible vs the unedited control by the M5 pre-registered criterion**
   (§7 M5, R24 — seeds, effect direction, minimum magnitude fixed before the run; "visible" is
   not noise and one run is not evidence without them).

Miss the claim → the retrospective names which object decision was wrong, not which deadline was
tight.

## 7. Phases

- **M0 — object spec + the four named artifacts** (design round, this charter's review): graph
  schema, generator/override representation, and the FOUR M0 artifacts each named and specified:
  (1) the **typed predicate AST** (§2.1, R12), (2) the **checkpoint manifest** (§2.3, R18), (3)
  the **transaction protocol** (§4, R17), (4) the **provenance spine** (§2.2, R11). Plus the
  semantic/realization ledger split (§5, R19) and the injective-emit subset definition (§3.2,
  R16). Paper artifact: the distributed-representability stress DOC (axis 3). Gate: the predicate
  AST serializes + type-errors; the manifest schema validates a real checkpoint.
- **M1 — read-only:** three checkpoints open (manifest-validated); generator/instance hierarchy
  rendered (sequence-ui); no edits. Gate: **parity with INDEPENDENT references** (R26) — see §7.1.
- **M2 — live:** in-tab inference wired (the #88 worker substrate); activation/SAE painting on the
  rendered graph. Gate: painted values match the demo's readbacks.
- **M3 — edits:** bin-1 moves + the §3 taxonomy (morphs with identity equations, behavior-changing
  edits with preview/commit) + the split ledger (§5) + undo (§5 rebase). Gate: **every claimed
  MORPH function-preserving at apply, verified by its identity equation + independent-reference
  parity, not one logit probe** (R26, §7.1); every behavior-changing edit's before/after measured;
  the transaction protocol (§4) holds under a concurrent-edit test; re-record applies the edit live.
- **M4 — emit + the speedrun six.** Gate: emitted PyTorch (defined subset, R16) runs under torch;
  `canonicalize(emit(G)) == G`, alias-group equality, state-dict round-trip, independent forward
  parity (R16); six techniques as ledger scripts checked into tools/.
- **M5 — train-as-evidence.** Gate: claim item 4, **pre-registered (R24)**: exact hardware/
  software, seeds = **3 seeded runs per arm**, pre-registered effect DIRECTION + minimum
  magnitude vs control at a fixed step count; a failed case reported, never averaged away.

### 7.1 Parity gates use INDEPENDENT references (R26)

**Ruling R26 (accepted as proposed):** if the generator, checkpoint binder, in-tab executor, and
emitter all consume the SAME mistaken UID/path map, same-path parity agrees on the WRONG model.
And a single logit probe both misses a changed branch (too weak) and fails a legitimate algebraic
identity that changes fp reduction order (too strict). So:

- parity uses **INDEPENDENT upstream references** — checkpoints/logits from upstream PyTorch, not
  our own re-emit;
- **full manifest-driven parameter value + alias-group comparison** (R18 manifest, R26) — the
  weights are verified independently, not inferred from one output;
- **multi-input probes** (multiple seeded/adversarial inputs + intermediate activation
  checkpoints), not one;
- **exact-zero ONLY for claimed morphisms**, and even then via the parameter mapping + identity
  equation ASSERTED SEPARATELY — not inferred from one logit (R26). Behavior-changing edits get
  edit-specific tolerances over a defined input domain, never exact-bit equality.

## 8. Shared spine with the schedule editor

One spine, two zoom regimes: **model graph → module → island lane (sol's editor, shipped
30a2326d) → intra-island schedule (sibling P3)**. Land once, share:

- the **provenance spine** (§2.2, R11) — the `ModelNodeUid → … → ScheduleEntityUid` map is the
  shared substrate for UID + ledger across both editors;
- the **ONE typed predicate AST** (§2.1, R12) — shared verbatim (a typed AST + one serializer +
  one type-checker), model leaves and schedule leaves in the same tree;
- identity coordinates in fingerprints (the three separated identities, sibling §5);
- sequence-ui; legality-as-refusal-with-reason UX.

**The "same registry shape" realizer claim is DELETED (R13).** PyTorch emit is NOT "a realizer
with a capability profile, same registry shape as WGSL/Triton". PyTorch emit realizes model
strata 1–2 and explicitly CANNOT represent schedules; the WGSL/Triton realizers realize a
ScheduleState into a kernel. They share no input interface, version coordinate, refusal schema,
or verification harness (R13). They are separate concrete instances; a generic `Realizer<...>`
protocol is deferred until a third genuine instance exists (rule of three, sibling §4). Reusing a
four-field record name does not make the zoom regimes compositional.

Neither editor blocks the other.

## 9. Risks

(a) **Generator inference from checkpoints** — we do NOT infer architecture from weight names; our
three implementations ARE the generators, checkpoints bind to them via the §2.3 manifest (that's
why arbitrary ingest is refused). The manifest ALSO defends a KNOWN family against version drift
(R18 — the v1 charter left that gap; the surviving red-team attack was exactly this). (b)
**Live-edit memory + snapshot budget** — re-record under 5GB residency on 16GB Macs; the browser
residency lessons (f16 tables, row-block streaming) apply; the §5 retained weight/optimizer
snapshots (undo-across-training) have a declared retention budget, measured in M2. (c) **Scope
creep into the training-program layer** — fenced OUT; the predicate AST is the only shared
artifact. (d) **Ledger/undo divergence from tape state** — an edit's SEMANTIC ledger entry and its
realization-arm invalidation are ONE transaction (§4 CAS); the realization history is separately
invalidated on re-record (§5); tested in M3's gate. (e) **Emit drift** — M1's parity gate
(independent references, R26) re-runs on every M3 edit class (emit the edited graph, run under
torch against an INDEPENDENT reference, compare parameters by manifest — not against our own
re-emit).

## 5-note. v1 fence (retained from v1)

**IN:** canonical graph + generator/overrides + the typed predicate AST; bin-1 architecture edits
(skips, gates, norms, activation swaps, #64 attention modifiers — softcap, sliding window, custom
score/mask mods) split into morphs vs behavior-changing edits (§3); three models opening FROM
CHECKPOINTS (manifest-validated); live in-tab inference with activation/SAE painting; the §3 edit
taxonomy; UID + split ledger (§5); PyTorch INJECTIVE EMIT into the defined subset (strata 1–2).

**OUT (each with its parking spot):** training-program layer (second substrate — future charter);
phase transitions (speedrun bin 2/3); distributed (stress DOC only, axis 3); kernel stratum
(sibling: schedule-state campaign — the shared spine keeps zoom-down continuous); widen;
duplicate-without-a-schema; arbitrary-source ingest; torch.export ingest (deferred); lemma
derivation (see sibling doc).
