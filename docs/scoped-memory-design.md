# Scoped Memory: from steps to scopes

**Status:** design, approved direction (2026-07-03). Not yet implemented.
**Companion docs:** `architecture-debt.md` (sin taxonomy), `stage4-compile-from-ir.md`
(dispatch-tape; interacts — see §8).

## 0. The decision

Torchlette moves from the step/snapshot memory model to a **scope-stack** model:
deterministic reclamation is scoped to nestable intervals, survival is either
*structural* (a tensor escapes by being returned/held by a surviving tensor) or
*manual* (`keep`/`persist`). The step becomes sugar over a scope; `markStep()`
remains as a loop idiom and eventually as compatibility API. App-facing code
should never need to know what a "step" is.

Why: JS has no refcounting, so "holding a reference" cannot be the keep signal —
GC is the only such signal and its non-determinism is precisely the leak class
we keep fixing (decode leaked ~1,600 handles/token until 4f5124c/f651365).
Every deterministic replacement makes *keeping* the explicit act (TF.js
converged on `tidy()`/`keep()`). Torchlette already has all three pieces in
embryonic form — `tidy()` (lexical scope, sync only), `beginStep`/`markStep`
(non-lexical global interval), `stepScopedCleanup` (implicit interval) — as
three surfaces over what should be ONE mechanism. "Step" is borrowed
training-loop vocabulary that has no referent in event-driven apps; the scope
(function/task) is the unit programs actually have.

## 1. What markStep actually is (the decomposition)

`markStep()` is four things fused. Only the first is user-semantic:

1. **Reclamation boundary** — `storageTracker.snapshotForStep()` /
   `releaseStepTemps()` diff: release reachable-but-not-snapshotted storages.
2. **Buffer-pool epoch** — the fence after which `pendingRelease` buffers may
   return to the pool (`endSharedEncoder()` → `flushPendingToAvailable()`,
   shared-encoder.ts:242/99). Flushing mid-step is a documented deterministic
   numerical-corruption class. All GPU destroys are fence-gated
   (`deferredDestroy`).
3. **Planner/replay scope** — the step-scoped shared `PlannerRegistry`
   (cross-plan temp packing assumes strictly-sequential plans within a step);
   wrapper-level generation stamps consumed by the demotion sweep.
4. **Quiesce point** — fence-before-demotion ordering (quiesce BEFORE the
   sweep, or the sweep's deferred destroys execute under still-pending
   submits — "used in submit while destroyed").

**Phase 0 of this campaign is re-homing (2)–(4) onto engine-observable
triggers** so the scope question becomes purely about JS-side handle lifetime:

- Pool epoch: "fence completed AND no shared encoder open" replaces
  "end of step". The invariant is about the GPU timeline, not user semantics;
  the trigger must preserve never-flush-while-encoder-open and
  fence-before-destroy exactly.
- Planner registry + generation stamps: keyed by a monotonically increasing
  **epoch id** the engine bumps at each quiesce, instead of "the step".
  Scopes and steps both map onto epochs; the strictly-sequential-plans
  assumption holds as long as scopes are single-flight (§4).
- Quiesce ordering: unchanged, just renamed — quiesce is an engine operation
  scopes can request, not a property of steps.

Phase 0 is separable, gate-able against current behavior (identical epochs in
a training loop ⇒ identical flush/sweep timing), and worth doing even if the
rest stalls.

## 2. The unification: one scope mechanism, three surfaces

A **Scope** is: a node in a stack (parent pointer), a creation list (every
RuntimeTensor/storage created while it is current), and an exit protocol.

- `tidy(fn)` — lexical, synchronous scope. Already exists as
  `TidyDispatchMode` (runtime/engine.ts:200) with `disposeNonEscaped()`.
  Becomes a thin wrapper over Scope.
- **Step** — a non-lexical scope: `markStep()` closes the current interval
  scope and opens the next. `stepScopedCleanup` (f651365) is exactly this with
  an arming protocol; it becomes the compatibility surface.
- **Async scope** — `api.scope(async () => {...})` — the new surface, and the
  one interactive apps actually need (§4).

Exit protocol (all surfaces): tensors created in the scope are released
**to the parent scope** (not destroyed outright — the parent may still be
inside its own interval), *except* escapees (§3). Release is
reachability-respecting: it feeds the existing `destroyUnreachable` machinery,
so a storage referenced by a surviving tensor (view bases via baseStorageId
chains, rcRetains) survives regardless of where it was created.

Cost note: scope tracking is O(created-in-scope) (a list, appended at dispatch
time — TidyDispatchMode already works this way). The step snapshot is
O(all-alive) (set capture + diff). The scope model is strictly cheaper and
safe to run at UI-event frequency.

## 3. Escape rules (what survives a scope)

1. **Structural**: the value returned from `tidy(fn)`/`scope(fn)` (tensors
   found in it, recursively through arrays/objects — TidyDispatchMode's
   existing escape walk) is re-parented to the enclosing scope.
2. **Reachability**: anything a surviving tensor's storage graph reaches
   (view bases, multi-output siblings still referenced) survives — enforced by
   the release machinery, not by the escape walk. This is load-bearing: a
   returned view whose base was created in-scope keeps the base.
3. **Manual**: `api.keep(t)` (alias of the existing `persist`/
   `adoptIntoSnapshot`, storage-tracker.ts:201) re-parents `t` to the ROOT
   scope (process-lifetime). For "keep for a while, not forever", re-parenting
   to a named ancestor scope is a possible extension; not in v1.
4. **Framework-internal escapes**: APIs that create deliberate crossings own
   their keeps. `collectHidden`/residual hooks persist what they hand out;
   optimizer state adopts at creation (the existing persistence-UAF contract);
   `generateChat` manages its per-token scopes internally. **Rule: the layer
   that creates a cross-scope reference is responsible for declaring it.**
   App authors should never write `keep` for framework-produced state.

## 4. Hard problem #1: async attribution (the interleaving trap)

A scope must know which creations are "its own" across `await` points. Two
concurrently-running async scopes that interleave awaits would misattribute
each other's synchronous creation bursts.

Facts that bound the problem:
- Tensor *creation* is synchronous (lazy IR nodes); awaits happen at
  force/readback/fence points. So attribution errors occur only when two async
  tasks both create tensors and interleave.
- The engine's exec lock already serializes execution ("Engine is busy") —
  true concurrency of GPU work is impossible; only creation-burst interleaving
  is at issue.
- Node has `AsyncLocalStorage`; the browser has NO standard equivalent
  (TC39 AsyncContext is a proposal). Any ambient-context design is
  Node-only or polyfill-fragile.

**Decision: single-flight ambient async scopes with a loud error.**
`api.scope(async fn)` sets the current scope; a second `scope()` entered while
one is open (from a different task) THROWS (or, explicitly opted, queues).
Rationale: silent misattribution is a silent-UAF factory; a loud error is
diagnosable and matches the engine's existing exec-lock reality. If/when
AsyncContext lands, the restriction can be relaxed without API change.
Explicit scope handles (`const s = api.openScope(); … s.close()`) are kept as
the low-level API underneath (they need no ambient magic and are what steps
compile to).

## 5. Hard problem #2: autograd is a cross-cutting lifetime

Today saved-for-backward tensors share lifecycle with their user handles
("disposing intermediates breaks autograd" — Known Semantic Limitations).
Under scopes, forward-in-scope + backward-outside-scope would release every
saved activation at scope exit → **silently wrong gradients**, the worst bug
class in the taxonomy.

**Decision: the autograd graph must independently retain its saved tensors**
(PyTorch semantics): rcRetain at save time, release at `cleanupAutogradGraph`.
This is a prerequisite, not an option — and it is independently valuable
(it also fixes the documented manual-dispose footgun). Notes:
- Retain at the STORAGE level via the existing rc machinery (the same
  machinery the compiled-plan harvest uses — see compiled-harvest-viewbase-leak
  for the leak shape to avoid: every retain needs a symmetric release path,
  including plan-invalidation/teardown paths).
- `backward()` already clears the graph (no double-backward), so release
  points are well-defined.
- Gate: forward in `tidy`, backward outside, gradients numerically identical
  to un-scoped run — on both CPU and webgpu projects; plus the strict-lifetime
  flag (`TORCHLETTE_STRICT_LIFETIME=1`) run over the training gates.

## 6. Persistent state (params, optimizer state, KV caches)

**Principle: optimizer patterns are first-class ENGINE capabilities, not
special cases.** The lifetime/update patterns optimizers need — in-place
stable-storage updates, persistent state across a step/scope boundary,
multi-output side state (m/v) — must be GENERIC primitives any op or user
optimizer can use, and correctness must be STRUCTURAL, not opt-in. Two
anti-patterns this rules out:
1. **Optimizer detection** — the engine special-casing Adam/SGD by op name or
   shape in the executor, materialization, or liveness. (Cf. "no kernel names
   above the dispatcher" — the same rule applied to lifetime.)
2. **Per-optimizer plumbing** — requiring each optimizer author to hand-wire
   `persist()`/`adopt`/`neuter` calls to avoid a UAF. A custom optimizer that
   expresses `update = f(state)` with in-place/aliased outputs should get
   correctness FOR FREE.
#68 was the forcing case, and it taught the real shape of the gap. The
NaN was NOT the hypothesized param-storage churn / in-plan liveness — it was
`m`/`v` (Adam moment state) materializing lazily MID-scope, so they were
absent from the scope-entry snapshot and `releaseStepTemps` demoted them at
scope close (the documented persistence-UAF class; `markStep` survives only
because the next `beginStep` re-snapshots them). Two outcomes:
- The genuine special-case hack — the `neuter` trick (`oldBt.destroy = noop`)
  — was DELETED, and it turned out redundant with the backend's own
  execution-time neuter (fused.ts) + `retainPlanInputRefs`, not made-redundant
  by a new primitive. So the per-optimizer buffer hack is gone: ✓ principle.
- The RESIDUAL declaration is `runtime.persist(m/v)` — the SAME escape API any
  code uses to keep a tensor past a scope, and the same contract the foreach
  path already carried. This is not engine-side optimizer detection, and it
  made the two Adam paths CONSISTENT rather than adding new special-casing.
  It is, however, still a per-optimizer call: a *new* optimizer author must
  know to persist their state.
**The remaining step toward the ideal (persist-free) is a generic "registered
persistent state" capability**: anything registered as optimizer/module state
(m/v, momentum, module buffers) is auto-root-scoped, so persistence is free at
the point of registration and no per-step / per-optimizer `persist` call is
needed. That is the true fulfillment of "structural, not opt-in" — filed as a
follow-up, larger than #68. Until then, `persist` at state creation is the
blessed, generic (non-detection) contract.

- Params: created at load, outside any scope → root-parented naturally.
- Optimizer m/v: materialize lazily INSIDE the first step's scope. The
  existing contract already handles this (`runtime.persist()` /
  `adoptIntoSnapshot` + copy_-in-place updates — the persistence-UAF fix).
  Under scopes this becomes `keep`-at-creation inside the optimizer. No model
  code changes.
- Static KV caches: allocated before the loop (root scope), updated in place
  (stable buffer identity — also the replay contract). Already correct.
- Cat-grown KV (`presentKVs` held across steps): the pattern that motivated
  default-off for stepScopedCleanup. Under scopes it requires `keep` per step
  — deliberately annoying, because the pattern is superseded by static KV.
  Kept working via the compatibility surface (default semantics unchanged
  until the major-version flip, §9).

## 7. Replay / planner / stamps

Mechanical re-keying (step id → epoch id, §1), but it touches UAF-adjacent
machinery, so each change ships with: kv-differential (both paths),
topk-equivalence, implied-step-boundary spec, `test:gates` 4/4, full suite,
and the lifetime suites. The strictly-sequential-plans assumption is preserved
by single-flight scopes (§4). Wrapper-level generation stamps stay
wrapper-level (storage-level stamps demote fused Adam m/v — documented rule).

## 8. Interaction with stage-4 dispatch-tape (#43)

The dispatch-tape endgame replays a recorded tape with near-zero per-step JS.
Scopes are what delimit a tape's transient allocations: a tape records within
one scope-epoch and its replays assume the same scope shape. Design
requirement: epoch ids (§1) are the shared vocabulary — the tape records the
epoch structure, not "steps". Neither campaign blocks the other, but both
should use epoch ids from the start so they don't need a third migration.

## 9. Migration stages (each independently gated; no big bang)

0. **Re-home the GPU epochs** (§1). Zero user-visible change. Gate: identical
   flush/sweep timing in training loop traces; full suite; test:gates.
1. **Autograd saved-tensor retention** (§5). Zero user-visible change (strictly
   fewer footguns). Gate: §5's differential + full training gates +
   leak regression (reachableStorages flat with clip+optimizer, the
   compiled-harvest gate shape).
2. **Scope stack under the hood**: TidyDispatchMode and the step snapshot
   re-implemented as Scope; `markStep`/`stepScopedCleanup`/`tidy` become
   surfaces. Behavior byte-identical (the equivalence bar: selection-
   equivalence-style before/after fingerprint of release events in a training
   loop + decode loop). This is the riskiest stage; it must be a pure refactor.
3. **`api.scope()` async surface** (§4) + `keep()` naming + framework-internal
   escapes (§3.4: collectHidden persists its outputs; generateChat drops its
   flag ceremony for a per-token scope). New gates: interleaving-throws test;
   scope-in-scope; scope+backward.
4. **Migrate in-tree loops** (gpt2-browser inference, bench-generation,
   examples) to scopes; docs flip to scope-first vocabulary; steps documented
   as the loop idiom.
5. **(major version)** Scoped semantics by default; GC-reliant mode becomes
   the opt-out flag; cat-grown-KV-style patterns require `keep` (loudly
   documented). Not before 0–4 have soaked.

## 10. Open questions (resolve during stage 2/3, not before)

- Does scope exit force pending lazy nodes it created that were never forced
  (dead lazy graph)? Proposal: no — dead nodes are dropped unforced; only
  forced storages need release bookkeeping. Verify against forceAllPending
  node.result cleanup (remote-training-plan-growth ledger).
- `keep`-to-named-ancestor (§3.3) — deferred until a real consumer exists.
- Whether `scope()` should auto-quiesce on exit (fence) or leave that to the
  next natural fence. Leaning: no implicit fence (scopes must stay cheap);
  loops that need the fence use markStep semantics, which is what the step
  surface is for.
