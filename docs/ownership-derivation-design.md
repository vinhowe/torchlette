# Ownership Derivation: liveness classification as a derived fact

**Status:** DRAFT FOR REVIEW (Vin) · 2026-07-11 · Task #70 (promoted per the platonic-form review)
**Prereq:** #73 strict-lifetime default (in flight) — the tripwire net this campaign works under.

## 0. Declaration (one sentence)

A storage's liveness classification — persistent vs step-scoped vs graph-retained — is
DERIVED at the point of inquiry from single-source facts (live wrappers, their birth
generation, registered state, graph retention); it is never stored as mutable slots that
must be defended.

## 1. The indictment (why this is the correctness steepest-ascent)

Every entry in the storage-lifetime bug ledger traces to classification living as
mutable, distributed state:

| Incident | The stored-state failure |
|---|---|
| persistence-contract UAF (2026-06-10) | demotion sweep trusted stored classification over actual holders |
| pool double-release | ownership recorded twice, released twice |
| #68 Adam churn, #69 replay reclaim | replace-and-hold vs stored ownership |
| #90 harvest-view FP | per-replay handle's stored state contradicted its base's |
| #86 mech 1 | `_wrapperGen` stamped at the WRONG MOMENT (materialize, not birth) |
| #86 mech 2 / #74 mech 1 | the single WeakRef owner SLOT stolen by clones/sidecars |
| #74 `persist()` no-op | persistence recorded into a set that wasn't listening between steps |

Each fix added an invariant DEFENDING the stored state: birth-time stamping, two
steal-refusal carve-outs (`_graphRetained`, `_sidecarShare`), a second persistence
mechanism (`_durablePersistent` beside the snapshot), guard exoneration clauses
(`declaredReplay`, `viewAliasesLiveBase`). All correct; all clauses. Clauses accreting
on a classifier are the tell that classification is distributed rather than derived —
the same disease #91 cured for cache keys (key=content made drift unconstructible).
This campaign does the same for ownership: make misclassification unconstructible by
removing the stored thing that can disagree with reality.

## 2. The root flaw: ONE owner slot, MANY holders

`storageTracker` maps each storage to a SINGLE WeakRef "owner" wrapper. Reality:
multiple wrappers legitimately share one storage (views, retention clones, sidecar
pins, cross-engine handles). One slot forces a "who is THE owner" fiction; every
mechanism above is a patch on that fiction.

## 3. The derived model

**Source facts (each already exists or is a small delta):**
- **W(s)** — the set of live wrapper objects referencing storage s. Structural change:
  owner SLOT → owner SET of WeakRefs (pruned via the existing FinalizationRegistry).
- **gen(w)** — wrapper birth generation (exists post-#86, stamped in the ctor).
- **SNAP** — the step snapshot (exists).
- **REG** — registered state: modules/optimizers DECLARE their state tensors
  (nn.Module parameters/buffers already enumerate; optimizers register m/v/velocity).
  Subsumes both `persist()` calls and `_durablePersistent`.
- **G(s)** — graph retention count (exists: autograd saved-slot rc).

**Derived classification (computed at inquiry, never stored):**
```
persistent(s)  := ∃ w ∈ W(s) : w ∈ SNAP ∪ REG
graphHeld(s)   := G(s) > 0
stepScoped(s)  := ¬persistent(s) ∧ ¬graphHeld(s)     // eligible for demotion at markStep
dangling-read  := inquiry on s with W(s) = ∅ ∧ G(s) = 0 ∧ not plan-retained
```
- The steal problem dissolves: there is no slot to steal — a clone joining W(s) cannot
  UNMAKE another wrapper's snapshot membership.
- `viewAliasesLiveBase` becomes a first-class query (base-chain liveness = derived
  classification of the root), not a guard clause.
- `persist(t)` becomes sugar for REG registration (or dies — decided at review).

## 4. Deletion targets (named per policy)

The owner-slot machinery and its defenses: the two steal-refusal carve-outs, the dual
persistence sets (merged into SNAP ∪ REG), the stamp back-fill path, `releaseStepTemps`'s
owner-based demotion logic (re-expressed over derived stepScoped), and — stretch —
the guard's exoneration clauses re-derived from the model. Expected net: the
storage-tracker (currently the most clause-dense file in src/) shrinks materially;
`persist()` call sites across optimizers/modules die into registration.

## 5. Phases (assert-agreement throughout — this campaign's null differential)

- **D0 — design review** (this doc): the one open question set below.
- **D1 — owner SET + derived classification, SHADOW mode:** compute BOTH the stored
  classification and the derived one at every decision point; `assert-agreement` under
  the full suite + strict + the training tools + browser gates. Ship with zero behavior
  change. Divergences are bugs in one side or the other — each one adjudicated and
  documented (some may be latent bugs in the CURRENT model).
- **D2 — flip:** derived becomes authoritative; stored model + carve-outs DELETED.
- **D3 — registered state:** optimizer/module registration API; `persist()` sunset.
- **D4 — guard simplification:** exoneration clauses re-derived or deleted.

## 6. Gates

Strict-lifetime default (armed by #73) across everything; the D1 agreement differential
(zero unadjudicated divergences); all existing lifetime gates (#86, #90, #74 specs);
memory flatness (browser reachableStorages gate, profiler zero-leak lines, the 124M
regression); perf: wrapper-set maintenance is on the tensor-create hot path — the
browser progressive-slowdown history (V8 GC, FinalizationRegistry pressure) sets the
budget: no measurable tok/s regression in the browser LoRA baseline and no RT/step
growth regression.

## 7. Risks

(a) **Hot-path cost** of set maintenance — mitigate: per-storage arrays with lazy
pruning, measured in D1. (b) **GC-timing sensitivity** — #74 found bugs that ANY
instrumentation masks; the D1 shadow must be always-on in the differential runs, not
sampled, and browser-validated. (c) **Set semantics for cross-engine handles** — the
between-engine reset (#74) must clear W(s) contributions from a dead engine; covered by
the three #74 acceptance specs. (d) **Scope creep into scoped-memory (#66)** — the
scope-stack campaign consumes this model but is NOT this campaign; the seam is that
SNAP generalizes to scope membership later.

## 8. The open design question (for review)

Whether REG (registered state) is a set of WRAPPERS or a set of STORAGES. Wrappers
matches the snapshot's semantics and survives storage replacement (copy_-in-place keeps
the wrapper); storages would survive wrapper churn but reintroduces exactly the
replacement-and-hold ambiguity that caused #68. RECOMMENDED: wrappers, consistent with
the wrapper-level-stamps rule the implied-boundary work already established.
