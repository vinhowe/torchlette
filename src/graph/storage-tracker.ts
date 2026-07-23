import {
  bumpEpoch,
  currentEpoch,
  epochTrace,
  epochTraceEnabled,
} from "../core/epoch";
import {
  findDeadTensorsAtStep,
  type TensorLifetime,
} from "./lifetime-analysis";
import { getNextStorageId } from "./node-factory";
import { rcDelete, rcGet, rcRelease } from "./refcount";
import type { StorageHandle } from "./types";

/**
 * Storage tracker — manages StorageHandle lifecycle via reference counting.
 *
 * Liveness is determined by rc (reference count):
 *   rc > 0  →  alive (some tensor or view holds a reference)
 *   rc <= 0 →  dead (eligible for destruction)
 *
 * Views keep their base alive through rcRetain on the base storage.
 * No separate "needed by views" walk is required.
 *
 * WeakRefs are kept as a safety net: when a tensor is GC'd without being
 * disposed, the WeakRef scan in destroyUnreachable() detects this and
 * releases the tensor's claim (rcRelease). This ensures storages are
 * eventually freed even without explicit dispose() or `using`.
 */
class StorageTracker {
  /** All storages created and not yet destroyed */
  private allStorages = new Map<number, StorageHandle>();

  /** Tensors alive at beginStep — these are "persistent" across the step.
   *  A tensor's storage can change within a step (via _updateLazyRef), but
   *  the tensor object is stable. Tracking the tensor (not the storage id)
   *  ensures persistence survives storage replacement (e.g., Adam m/v updates). */
  private _stepStartTensors: WeakSet<object> | null = null;

  /** REG — REGISTERED persistent state (task #70 D3; was `_durablePersistent`,
   *  task #74). Modules and optimizers DECLARE their long-lived state by
   *  registering the WRAPPER here (registerState); it is the ONE persistence
   *  source (doc §3: persistent := ∃ w ∈ W(s) : w ∈ SNAP ∪ REG), GEN-INDEPENDENT
   *  (§D1-log #3): a registered tensor is persistent whatever the boundary's
   *  closing generation, so a concurrent test perturbing the shared gen counter
   *  cannot filter it out of persistence. The step snapshot (SNAP) above is
   *  TRANSIENT — rebuilt each boundary, NULL between steps — so registration
   *  works BETWEEN steps too (the GradScaler ctor's LiveScalar, module params
   *  registered at construction). releaseStepTemps never demotes a member.
   *  WeakSet: membership dies with the wrapper (GC/dispose), so claims still
   *  release normally — copy_-in-place state updates KEEP the wrapper (only the
   *  storage changes), so an in-place-updated param/m/v stays registered with no
   *  churn (why wrappers won the §8 ruling). Cleared (rebuilt-empty) at
   *  disposeAllForNewEngine so a new engine does not inherit the dead engine's
   *  registered state (the #74 cross-engine contract). */
  private _registeredState = new WeakSet<object>();

  /** Implied-step-boundary generation stamps (see
   *  Torchlette.queueStepBoundary). Keyed by the engine EPOCH id
   *  (src/core/epoch.ts): each owning TENSOR OBJECT is stamped with the
   *  epoch current when the OBJECT is CONSTRUCTED (stampWrapperGen, from the
   *  Tensor ctor); a queued step boundary records the current epoch as its
   *  closing generation and bumps the counter.
   *  Stamps are compared only relatively (`stamp > boundaryEpoch`), so the
   *  epoch also advancing at other quiesce points is harmless — monotonic
   *  time-order is all that's used. Gen-scoped cleanup uses the stamps to
   *  separate "this step's tensors" from work the NEXT step has already
   *  lazily built by the time the boundary commits. The stamp is
   *  wrapper-level, like persistence itself: a persistent tensor's STORAGE
   *  can be replaced mid-step (fused Adam's m/v side outputs re-point the
   *  wrapper every step), so a storage-level stamp would mis-classify
   *  long-lived state as next-step work and demote it — live optimizer
   *  state corrupted, the persistence-UAF class.
   *
   *  ATTRIBUTION INVARIANT (task #86): the stamp records when the TENSOR
   *  OBJECT was born, NOT when its storage first materialized. A persistent
   *  tensor created eagerly but MATERIALIZED lazily (every `tensorFromArray`
   *  param: the object exists before the loop, but its upload node — and thus
   *  its first `trackTensor` — fires generations later, inside a step). Were
   *  the stamp set at first `trackTensor` (materialize time), such a param
   *  would carry a gen LATER than the closing boundary and be filtered OUT of
   *  the persistent snapshot in `snapshotForStep`, then reaped by
   *  `releaseStepTemps` while the live `w` still points at that storage (the
   *  minimal-loop SGD reclaimed-read false positive — and a genuine UAF the
   *  moment the pooled buffer is reused). Stamping at construction ties the
   *  generation to object identity, matching the wrapper-level persistence
   *  model. */
  private _wrapperGen = new WeakMap<object, number>();

  /** Debug counters */
  private _debugDestroyCount = 0;

  // ── Derived-ownership model: the owner SET (task #70, D2 authoritative) ────
  // The owner SET: per-storage set of live wrapper WeakRefs — W(s), doc §3. This
  // is the SINGLE SOURCE for the liveness classifier (persistent / step-scoped /
  // graph-held), replacing the former single `tensorWeakRefs` owner SLOT that
  // one holder could STEAL from another. Every trackTensor registers here
  // UNCONDITIONALLY — no steal semantics, no sharer refusal: a slot that cannot
  // be stolen cannot disagree with reality (doc §2/§3; the disease #91 cured for
  // cache keys). Pruned lazily on read (dead deref dropped) and eagerly on
  // untrack/destroy/GC-scan, so it never extends a wrapper's lifetime (WeakRef
  // only — doc §7(b)/the GC hard-boundary gate).
  private _ownerSet = new Map<number, Set<WeakRef<object>>>();

  /** Register a newly created storage. */
  register(storage: StorageHandle): void {
    this.allStorages.set(storage.id, storage);
  }

  /**
   * Register a WeakRef to the tensor that owns this storage.
   * Used by destroyUnreachable() to detect GC'd tensors.
   */
  trackTensor(storageId: number, tensorRef: object): void {
    // Register into the owner SET (W(s), doc §3) — the derived classifier's
    // single source. NO steal semantics: a sharer (retention clone / sidecar)
    // JOINS W(s) rather than fighting for a slot; the classifier reads the whole
    // set, so a graph/sidecar sharer is a plain member that _derived recognises
    // as a keep signal. This is the D2 deletion of the former owner-slot steal
    // machinery (the two sharer-refusal carve-outs and the slot back-fill stamp):
    // the slot they defended is gone, so misclassification-by-steal is
    // unconstructible.
    this._ownerSetAdd(storageId, tensorRef);
  }

  // ── owner-SET maintenance + derived classifier ───────────────────────────

  /** Add a wrapper to storage s's owner set (idempotent by object identity —
   *  a wrapper re-tracked to the same storage does not double-insert because we
   *  compare deref'd targets). Prunes dead refs opportunistically. */
  private _ownerSetAdd(storageId: number, tensorRef: object): void {
    let set = this._ownerSet.get(storageId);
    if (!set) {
      set = new Set();
      this._ownerSet.set(storageId, set);
    }
    // Dedup by identity + drop any dead refs while we are here.
    for (const ref of set) {
      const t = ref.deref();
      if (t === undefined) set.delete(ref);
      else if (t === tensorRef) return; // already a member
    }
    set.add(new WeakRef(tensorRef));
  }

  /** Live wrappers referencing storage s (W(s)), pruning dead refs. */
  private _liveOwners(storageId: number): object[] {
    const set = this._ownerSet.get(storageId);
    if (!set) return [];
    const live: object[] = [];
    for (const ref of set) {
      const t = ref.deref();
      if (t === undefined) set.delete(ref);
      else live.push(t);
    }
    if (set.size === 0) this._ownerSet.delete(storageId);
    return live;
  }

  /**
   * Does storage s still have a LIVE DURABLE owner — a wrapper that is a kept
   * holder in the derived model (REGISTERED state, step-snapshot persistent,
   * graph-retention clone, or sidecar pin)? Such a wrapper POINTING AT s makes s
   * the CURRENT storage of long-lived state, so s is live whatever its rc says.
   *
   * This is the derived owner-set model (task #70 D2 — the authoritative liveness
   * source, rc being explicitly unsound: a single wrapper holds ctor+materialize
   * rc=2) extended to the ONE site that still trusted rc alone: destruction.
   * `destroyUnreachable` reaps at rc≤0, and the fused Adam boundary transiently
   * leaves a freshly-superseded PARAM/state storage at rc≤0 while its registered
   * wrapper still points at it (the whole-step persistence-UAF: the param is read
   * by the next step's forward, its buffer already reclaimed). A wrapper that
   * MOVED off s (fused _updateLazyRef → untrackTensor) is no longer in W(s), so an
   * OLD storage after a legitimate state replacement has no durable owner and is
   * still destroyed — only a storage a live durable wrapper STILL points at is
   * protected. Mirrors `_derived`'s keptHolder minus the rc gate (the rc gate is
   * exactly what we are correcting here).
   */
  private _hasLiveDurableOwner(storageId: number): boolean {
    for (const w of this._liveOwners(storageId)) {
      if (this._registeredState.has(w)) return true;
      if (this._stepStartTensors?.has(w) === true) return true;
      if ((w as { _graphRetained?: boolean })._graphRetained === true)
        return true;
      if ((w as { _sidecarShare?: boolean })._sidecarShare === true)
        return true;
    }
    return false;
  }

  /** The set of storage ids that are the BASE of at least one live view (each
   *  such view holds an rc-retain on its base that releaseStepTemps neither sees
   *  nor releases). Computed ONCE per releaseStepTemps sweep and passed to
   *  `_derived`, so classification stays O(N) rather than O(N²). */
  private _liveViewBases(): Set<number> {
    const bases = new Set<number>();
    for (const [, s] of this.allStorages) {
      if (s.baseStorageId !== undefined && rcGet(s.id) > 0) {
        bases.add(s.baseStorageId);
      }
    }
    return bases;
  }

  /**
   * Derived classification (doc §3) — the AUTHORITATIVE liveness verdict at
   * releaseStepTemps, computed at inquiry from single-source facts (never
   * stored). `persistent` := ∃ w ∈ W(s) : w ∈ SNAP ∪ REG (REG =
   * `_durablePersistent`, gen-independent — §D1-log #3); `graphHeld` := ∃ live
   * `_graphRetained` clone (the model's G(s)>0 — the retention clone IS a
   * wrapper taking its own rc); `sidecar` := ∃ live `_sidecarShare` pin.
   * `keptHolder` := rc>0 ∧ (persistent ∨ graphHeld ∨ sidecar ∨ nonWrapperRetain)
   * — a live holder that survives the sweep. A storage is DEMOTABLE (stepScoped)
   * iff it has a live owner and NO kept holder. `maxGen` mirrors the gen-scoped
   * release for the SNAP axis (REG is exempt); a wrapper born after the boundary
   * is the next step's and excluded from the SNAP-scoped W(s).
   */
  private _derived(
    storageId: number,
    maxGen: number | undefined,
    liveViewBases: Set<number>,
  ): {
    persistent: boolean;
    graphHeld: boolean;
    sidecar: boolean;
    stepScoped: boolean;
    hasLiveOwner: boolean;
    keptHolder: boolean;
  } {
    let persistent = false;
    let graphHeld = false;
    let sidecar = false;
    let hasLiveOwner = false;
    for (const w of this._liveOwners(storageId)) {
      // REG (durable persist) is GEN-INDEPENDENT — a registered state tensor is
      // persistent whatever the boundary's closing generation. This is where the
      // derived model is STRICTLY SUPERIOR to the stored one (§D1-log #3, the
      // row-8 / #73 FP): the stored snapshot gen-filters EVERYTHING, so when a
      // concurrent test perturbs the SHARED generation counter a stepAsync param
      // (durably persist()ed by its optimizer) gets filtered OUT of the transient
      // snapshot and reaped — a false-positive UAF throw. Deriving persistence
      // from REG membership (doc §3: persistent := ∃ w ∈ W(s) : w ∈ SNAP ∪ REG,
      // no gen term on REG) makes that misclassification unconstructible. So the
      // REG check runs BEFORE the gen-filter continue.
      if (this._registeredState.has(w)) {
        persistent = true; // REG (D1) — gen-independent
        hasLiveOwner = true;
      }
      // graph-retention clones and sidecar pins are GEN-INDEPENDENT keep signals
      // (like REG): a LIVE retention clone (G(s)>0) or LiveScalar pin keeps the
      // storage alive via its OWN rc whatever the boundary's closing gen — a clone
      // minted DURING backward (stamp > maxGen, "next step" by the gen-filter's
      // reckoning) still holds the saved value THIS step's backward needs. These
      // MUST be read BEFORE the gen-filter continue, or a post-boundary clone's
      // keep-signal is dropped and its shared storage reaped under the live clone
      // (the read-a-temp-after-step() regression — implied-step-boundary.spec.ts).
      // D1 masked this: `_derived` ran in SHADOW and derivedSurvives OR'd in
      // storedSurvives, so the stored slot covered the gen-filtered clone; once
      // derived is AUTHORITATIVE (D2) the ordering bug bites.
      if ((w as { _graphRetained?: boolean })._graphRetained === true) {
        graphHeld = true; // autograd saved-slot retention clone: G(s)>0
        hasLiveOwner = true;
      }
      if ((w as { _sidecarShare?: boolean })._sidecarShare === true) {
        sidecar = true; // GradScaler LiveScalar pin: a persistence pin (task #74)
        hasLiveOwner = true;
      }
      if (maxGen !== undefined && (this._wrapperGen.get(w) ?? 0) > maxGen) {
        continue; // next-step wrapper: not part of THIS step's SNAP-scoped W(s)
      }
      hasLiveOwner = true;
      if (this._stepStartTensors?.has(w) === true) persistent = true;
    }
    // A storage is DEMOTABLE (stepScoped) at markStep only if EVERY thing that
    // keeps it alive is a step-scoped wrapper claim releaseStepTemps will drop.
    // It is KEPT if any of:
    //   • a persistent (SNAP∪REG), graph-retention, or sidecar-pin wrapper — the
    //     legitimate long-lived holders (doc §3);
    //   • a NON-wrapper retain: this storage is the BASE of a live view (the view
    //     rcRetain'd it). releaseStepTemps neither sees nor releases that retain,
    //     so reaping would UAF the view's base (the rc=1/setN=0 class from the
    //     ledger-attack probe — an old param storage still base-retained after
    //     the wrapper moved to the adamStep result). Plan-input retains are
    //     transient within a plan; releaseStepTemps runs at markStep after all
    //     plans complete, so they are not live here. rc-counting is unsound (a
    //     single wrapper holds ctor+materialize rc=2), so base retention is
    //     detected STRUCTURALLY (liveViewBases), not by rc arithmetic.
    // Ground truth: a storage already at rc≤0 is DEAD (destroyed at the next
    // destroyUnreachable) whatever the wrapper set says — a stale view whose
    // baseStorageId still names this id but which already released its base
    // retain leaves `liveViewBases` naming a dead base (the id=161 rc=0/viewBase
    // case). rc is the authority for "is anything still holding it"; the wrapper
    // classification only decides WHETHER releaseStepTemps will drop the last
    // holder. So survival = rc>0 (after the step-temp claim would be released)
    // AND a kept holder exists.
    const rc = rcGet(storageId);
    // A NON-wrapper retain keeps the storage past releaseStepTemps (which only
    // releases wrapper claims): a live view's base retain, or — when W(s)=∅ —
    // any rc at all (it cannot be a wrapper claim). Detected structurally
    // (liveViewBases, computed once) — rc-counting is unsound (one wrapper holds
    // ctor+materialize rc=2).
    const nonWrapperRetain =
      liveViewBases.has(storageId) || (!hasLiveOwner && rc > 0);
    // keptHolder := a holder that survives the sweep — a persistent (SNAP∪REG),
    // graph-retention, or sidecar-pin wrapper, or a non-wrapper retain. This is
    // the authoritative "keep it" signal: releaseStepTemps demotes a storage
    // ONLY when it has a live owner and NO kept holder (stepScoped). Reading the
    // whole owner SET (not a single steal-able slot) is what makes the former
    // reap-live class — the stale/disposed slot missing a live kept holder —
    // unconstructible.
    const keptHolder =
      rc > 0 && (persistent || graphHeld || sidecar || nonWrapperRetain);
    const stepScoped = rc > 0 && hasLiveOwner && !keptHolder;
    return {
      persistent,
      graphHeld,
      sidecar,
      stepScoped,
      hasLiveOwner,
      keptHolder,
    };
  }

  /** Test-only: owner-SET stats. `sets` = number of storages with a set entry;
   *  `liveMembers` = total LIVE (deref'd) wrappers across all sets. Used by the
   *  GC-pressure gate to prove the owner set holds only WeakRefs (after GC + a
   *  prune sweep, a dropped wrapper's membership collapses — the set never
   *  extends a wrapper's lifetime). Prunes dead refs as a side effect (same as
   *  any read of the set). */
  ownerSetStats(): { sets: number; liveMembers: number } {
    let liveMembers = 0;
    for (const [, set] of this._ownerSet) {
      for (const ref of set) {
        if (ref.deref() === undefined) set.delete(ref);
        else liveMembers++;
      }
    }
    return { sets: this._ownerSet.size, liveMembers };
  }

  /**
   * Is this storage GRAPH-HELD — saved for backward (∃ live `_graphRetained`
   * clone, G(s)>0), or the flattened base of such a graph-held view? (task #97).
   *
   * A first-class DERIVED query (the same single-source form as `viewBaseIsLive`
   * / `_derived`'s `graphHeld` axis), consumed by the stage-3 B overlay-release
   * claim seam: a producer whose value is saved for backward WILL be re-read by
   * the backward pass, so its registry entry MUST NOT be overlaid by an earlier
   * consumer's temps — no matter what the empirical last-reader observation
   * concluded (the observation is blind to the backward read, which resolves
   * lowered through getInputStorage). Gen-INDEPENDENT (a retention clone minted
   * during backward is still a live graph hold), matching `_derived`.
   */
  graphHeldAt(storageId: number): boolean {
    for (const w of this._liveOwners(storageId)) {
      if ((w as { _graphRetained?: boolean })._graphRetained === true) {
        return true;
      }
    }
    // A saved-for-backward VIEW retains its base; the base's value is read
    // through the view in backward. Flatten to the root and check it too.
    const sh = this.allStorages.get(storageId);
    let baseId = sh?.baseStorageId;
    const seen = new Set<number>();
    while (baseId !== undefined && !seen.has(baseId)) {
      seen.add(baseId);
      for (const w of this._liveOwners(baseId)) {
        if ((w as { _graphRetained?: boolean })._graphRetained === true) {
          return true;
        }
      }
      baseId = this.allStorages.get(baseId)?.baseStorageId;
    }
    return false;
  }

  /** Stamp a tensor OBJECT's generation at CONSTRUCTION (called from the
   *  Tensor ctor). Ties the persistent-vs-next-step classification to when
   *  the object was born rather than when its storage first materialized —
   *  see the _wrapperGen attribution invariant. Idempotent. */
  stampWrapperGen(tensorRef: object): void {
    if (!this._wrapperGen.has(tensorRef)) {
      this._wrapperGen.set(tensorRef, currentEpoch());
    }
  }

  /** Mark a step boundary (called when one is queued). Returns the epoch
   *  that closes — wrappers stamped <= this value belong to the step being
   *  ended — and bumps the engine epoch so everything created afterwards
   *  is stamped strictly greater. */
  bumpStepGen(): number {
    const closing = currentEpoch();
    bumpEpoch("stepBoundary");
    return closing;
  }

  /** Unregister a storage (after it's been destroyed or early-released). */
  unregister(storageId: number): void {
    this.allStorages.delete(storageId);
    // Owner set kept (see destroyStorageIds) — self-prunes via WeakRef.
  }

  /** A wrapper is leaving storage s (moved to a new storage via _updateLazyRef,
   *  or disposed). Remove it from W(s) so the classifier does not see a stale
   *  member. Mirror of _ownerSetAdd's per-wrapper removal. Public so tensor.ts
   *  can call it at the release seam (kept symmetric with the existing rcRelease
   *  there). No-op if the storage/set is gone. */
  untrackTensor(storageId: number, tensorRef: object): void {
    const set = this._ownerSet.get(storageId);
    if (!set) return;
    for (const ref of set) {
      const t = ref.deref();
      if (t === undefined || t === tensorRef) set.delete(ref);
    }
    if (set.size === 0) this._ownerSet.delete(storageId);
  }

  /**
   * Destroy all storages with rc <= 0.
   *
   * First scans WeakRefs to detect GC'd tensors and release their claims,
   * which may push more storages to rc=0. Then collects and destroys all
   * dead storages.
   */
  destroyUnreachable(): number {
    // Safety net + owner-SET prune in one sweep. For each storage's owner set,
    // drop dead WeakRefs. If a LIVE storage's set EMPTIES (every wrapper that
    // held it was GC'd without dispose — and the per-wrapper FinalizationRegistry
    // has not yet fired), release the GC'd claim so the storage can be reaped:
    // the old `tensorWeakRefs` GC-scan, now expressed over the owner SET (one
    // release per storage whose W(s) just emptied — the single-slot backstop
    // semantics; per-wrapper rc is otherwise released by the FR). The set entry
    // is then DELETED so a later sweep cannot double-release (a fresh trackTensor
    // recreates it). Sets for already-DESTROYED storages (no longer in
    // allStorages) are also dropped once empty — bounds memory (the sets are kept
    // through destruction so the classifier can observe a live holder still
    // pointing at a destroyed id, then self-prune as those wrappers die).
    for (const [id, set] of this._ownerSet) {
      for (const ref of set) {
        if (ref.deref() === undefined) set.delete(ref);
      }
      if (set.size === 0) {
        if (this.allStorages.has(id) && rcGet(id) > 0) rcRelease(id, "gc");
        this._ownerSet.delete(id);
      }
    }

    // Worklist-based cascading destruction. Each destroy can release a base
    // ref (rcRelease "view.destroyed"), potentially making the base dead too.
    // We re-collect dead storages until fixed point — single call, no caller loop.
    let totalDestroyed = 0;
    while (true) {
      // Only LIVE views protect their bases — dead views will release the
      // base retain when destroyed in this pass.
      const protectedBases = new Set<number>();
      for (const [id, storage] of this.allStorages) {
        if (storage.baseStorageId !== undefined && rcGet(id) > 0) {
          let baseId: number | undefined = storage.baseStorageId;
          while (baseId !== undefined && !protectedBases.has(baseId)) {
            protectedBases.add(baseId);
            baseId = this.allStorages.get(baseId)?.baseStorageId;
          }
        }
      }

      const toDestroy: number[] = [];
      for (const [id] of this.allStorages) {
        if (
          rcGet(id) <= 0 &&
          !protectedBases.has(id) &&
          // A storage a live durable wrapper (registered / snapshot-persistent /
          // graph-held / sidecar) still points at is the CURRENT storage of
          // long-lived state — live whatever rc says (the whole-step fused-Adam
          // persistence-UAF, where a superseded param sits at rc≤0 while its
          // registered wrapper still reads it next step). Owner-set liveness (D2)
          // over rc, at the last rc-only site.
          !this._hasLiveDurableOwner(id)
        ) {
          toDestroy.push(id);
        }
      }

      if (toDestroy.length === 0) break;
      totalDestroyed += this.destroyStorageIds(toDestroy);
    }
    if (epochTraceEnabled()) {
      epochTrace(`sweep destroyed=${totalDestroyed}`);
    }
    return totalDestroyed;
  }

  /** Destroy storages with rc <= 0, scoped to ids >= sinceId. */
  destroyUnreachableSince(sinceId: number): number {
    const toDestroy: number[] = [];
    for (const [id] of this.allStorages) {
      if (id < sinceId) continue;
      if (rcGet(id) <= 0) {
        toDestroy.push(id);
      }
    }
    if (toDestroy.length === 0) return 0;
    return this.destroyStorageIds(toDestroy);
  }

  /** Destroy a list of storage IDs. Returns count destroyed. */
  private destroyStorageIds(ids: number[]): number {
    let count = 0;
    for (const id of ids) {
      const storage = this.allStorages.get(id);
      if (!storage) continue;
      // Note: destruction here is rc-driven (rc≤0), NOT classification-driven.
      // A persistent wrapper's OLD storage is legitimately destroyed here after
      // the wrapper moves to a NEW storage (Adam m/v replacement: the wrapper
      // stays in SNAP, its old storage's rc hits 0 via _updateLazyRef). So this
      // is not a classification-consuming decision point — the shadow does NOT
      // compare here (it would false-fire on every state replacement). The
      // demotion CHOICE lives in releaseStepTemps; that is where the shadow runs.

      // Release view base ref (view is being destroyed → base loses a reference).
      // EXCEPT for compiled-plan harvested views whose base retain is owned by
      // the plan (planOwnedBaseRetain): the plan releases that retain itself
      // (at the next harvest / teardown), so releasing here too would
      // double-free the base. See StorageHandle.planOwnedBaseRetain and
      // compiled-plan.ts harvest.
      if (storage.baseStorageId !== undefined && !storage.planOwnedBaseRetain) {
        rcRelease(storage.baseStorageId, "view.destroyed");
      }
      rcDelete(id);

      const bt = storage.backendTensor;
      if (bt.ownsBuffer !== false && bt.destroy) {
        bt.destroy();
      }
      this.allStorages.delete(id);
      // Do NOT drop the owner set here. Keeping it through destruction lets the
      // classifier / [lifetime] guard observe whether a LIVE persistent/graph-
      // held wrapper still points at a destroyed storage. The set self-prunes
      // lazily as those wrappers die (WeakRef deref) and is reaped once empty in
      // destroyUnreachable; storage ids are monotonic (no id reuse to alias a
      // stale set). untrackTensor already removes a wrapper at its
      // dispose/_updateLazyRef release seam.
      count++;
      this._debugDestroyCount++;
    }
    return count;
  }

  /**
   * Snapshot tensors alive at the start of the step. These are "persistent"
   * (model params, optimizer state). Tensors created during the step are
   * step-scoped temporaries and will have their refs released at markStep.
   */
  snapshotForStep(maxGen?: number): void {
    this._stepStartTensors = new WeakSet();
    // Capture every live wrapper across the owner SETs (the classifier's single
    // source, D2). A WeakSet dedups by identity, so multiple set members of one
    // storage collapse.
    for (const [, set] of this._ownerSet) {
      for (const ref of set) {
        const target = ref.deref();
        if (!target) {
          set.delete(ref);
          continue;
        }
        // Gen-scoped snapshot (implied boundaries): tensors created AFTER the
        // boundary belong to the next step's lazily-built work — treating
        // them as persistent would exempt them from cleanup forever. REG
        // (durably-persist()ed) members are EXEMPT from the gen-filter — REG is
        // gen-independent (doc §3, §D1-log #3), so a registered state tensor is
        // persistent whatever the boundary's closing generation (unifies the
        // gen-filter asymmetry the shadow model resolved in `_derived`).
        if (
          maxGen !== undefined &&
          (this._wrapperGen.get(target) ?? 0) > maxGen &&
          !this._registeredState.has(target)
        ) {
          continue;
        }
        this._stepStartTensors.add(target);
      }
    }
  }

  /**
   * Mark a tensor created MID-STEP as persistent: add it to the active
   * step snapshot so releaseStepTemps will not reclaim its storage at
   * markStep. This is THE supported way to create long-lived state inside
   * a step (e.g. lazily-initialized optimizer state): without it, the
   * snapshot-membership rule treats every mid-step tensor as a temporary —
   * its buffer is reclaimed into the pool while the live tensor still
   * points at it, and later writes corrupt it silently (the per-param Adam
   * first-param m/v corruption). No-op when no step is active (tensors
   * created between steps are captured by the next snapshot).
   */
  adoptIntoSnapshot(tensor: object): void {
    this._stepStartTensors?.add(tensor);
  }

  /**
   * REGISTER persistent state (task #70 D3; was `persistDurable`, task #74). The
   * ONE registration primitive: modules (params/buffers) and optimizers (m/v,
   * velocity, t, lr) DECLARE their long-lived state by registering the WRAPPER
   * here. Marks the tensor persistent for the WRAPPER's lifetime (REG =
   * `_registeredState`, gen-independent), not just the active snapshot —
   * registration works BETWEEN steps (no active snapshot) and survives every
   * snapshot rebuild. Members are never demoted by releaseStepTemps. Also adopts
   * into the active step snapshot so a mid-step registration is persistent for
   * the CURRENT step immediately (before the next snapshotForStep). `persist()`
   * is a deprecated warn-once alias that delegates here.
   */
  registerState(tensor: object): void {
    this._registeredState.add(tensor);
    this._stepStartTensors?.add(tensor);
  }

  // ── Scope-surface support (api.scope / api.openScope) ─────────────────────
  // The async scope() surface (docs/scoped-memory-design.md §2-4) reuses the
  // SAME persistent-snapshot reclamation the step path uses: scope entry
  // snapshots the currently-alive tensors as this scope's "existed before"
  // baseline; scope exit adopts escapees into that baseline then
  // releaseStepTemps() releases everything created in-scope that didn't
  // escape. Nesting is LIFO save/restore of the single active snapshot: a
  // child scope steals the parent's snapshot (peekSnapshot), installs a fresh
  // one, and restores the parent's on close. This is the ONE reclamation path
  // — no second mechanism.

  /** The active persistent snapshot (opaque token; save it across a nested
   *  scope and restore it on the child's close). */
  peekSnapshot(): object | null {
    return this._stepStartTensors;
  }

  /** Install a previously-captured snapshot (restore the parent scope's on a
   *  child scope's close). */
  installSnapshot(snapshot: object | null): void {
    this._stepStartTensors = (snapshot as WeakSet<object> | null) ?? null;
  }

  /** Adopt a tensor into a SPECIFIC (possibly non-active) snapshot token —
   *  used by keep()/persist() to re-parent to an ancestor scope so a tensor
   *  survives every enclosing scope (escape-to-root). No-op for null tokens. */
  adoptIntoSnapshotToken(snapshot: object | null, tensor: object): void {
    (snapshot as WeakSet<object> | null)?.add(tensor);
  }

  /**
   * Drop the active step snapshot without releasing anything. Used when
   * step-scoped markStep cleanup (Torchlette.setStepScopedCleanup) is
   * DISABLED: its end-of-markStep snapshot must not linger, or a later bare
   * markStep — back on historical semantics — would consume it and reclaim
   * tensors created after the disable.
   */
  clearStepSnapshot(): void {
    this._stepStartTensors = null;
  }

  /**
   * Whether a storage has been DESTROYED (reclaimed). Reading a destroyed
   * storage through a stale ref is the silent-UAF class — its buffer is
   * back in the pool and may hold another op's data.
   */
  isDestroyed(storageId: number): boolean {
    return !this.allStorages.has(storageId);
  }

  /**
   * Is a reclaimed VIEW handle's base still live AND sharing its buffer? (task
   * #70 D4). A reclaimed view whose flattened base root is live and whose GPU
   * buffer IS that live base's buffer is NOT a use-after-free — the read returns
   * correct bytes (the compiled-plan HARVEST-view class: `planOwnedBaseRetain`
   * re-creates a view over a persistent base each replay, the per-replay handle
   * is reaped at markStep but its buffer aliases the live base the plan keeps
   * alive; #90 static-KV decode). This is the FIRST-CLASS form of the derived
   * model's `nonWrapperRetain` / `_liveViewBases` "a live view's base is a kept
   * holder" query (doc §3: "viewAliasesLiveBase becomes a first-class query"),
   * now OWNED by the tracker that owns `_liveViewBases` — the D4 re-derivation
   * that deletes op-dispatch's bespoke `viewAliasesLiveBase` clause.
   *
   * Single-source-of-truth seam: the base's live buffer is the ONE source; we
   * ASSERT the view's buffer equals it (viewBuf === baseBuf) before exonerating.
   * A genuinely-dangling view — base destroyed, or buffer already recycled to a
   * different buffer — fails this and the guard still fires.
   */
  viewBaseIsLive(storage: StorageHandle): boolean {
    if (storage.baseStorageId === undefined) return false;
    // Flatten the base chain to its live root (harvest chains are depth-1 but be
    // robust to nesting: a destroyed intermediate must not exonerate the read).
    let base = this.allStorages.get(storage.baseStorageId);
    while (base?.baseStorageId !== undefined) {
      const parent = this.allStorages.get(base.baseStorageId);
      if (!parent) break;
      base = parent;
    }
    if (!base) return false; // base itself reclaimed → truly dangling
    const viewBuf = (storage.backendTensor as { buffer?: unknown }).buffer;
    const baseBuf = (base.backendTensor as { buffer?: unknown }).buffer;
    return viewBuf !== undefined && viewBuf === baseBuf;
  }

  /** [step-tape 2b] The RuntimeTensor that owns a storage (if alive). A
   *  multi-plan tape skeleton uses this to re-resolve a frozen materialized
   *  ref to the owner's CURRENT storage each replay - persistent tensors
   *  updated by replace-and-hold writes (the scheduler's lr copy_) move to a
   *  new storage every step, so the recording-era ref would otherwise read
   *  the stale buffer (the frozen-LR class). */
  ownerOf(storageId: number): object | undefined {
    // Re-pointed at the owner SET (the slot is gone, D2). Prefer a persistent /
    // durably-persisted member — that is the PRINCIPAL the tape-replay rebind
    // wants (a param updated by replace-and-hold keeps one long-lived wrapper);
    // fall back to any live member.
    const owners = this._liveOwners(storageId);
    for (const w of owners) {
      if (
        this._stepStartTensors?.has(w) === true ||
        this._registeredState.has(w)
      ) {
        return w;
      }
    }
    return owners[0];
  }

  /** [P2 whole-step] Is a live owner of this storage REGISTERED STATE (REG /
   *  `registerState` — the gen-independent durable-persist signal, doc §3)?
   *  A registerState'd harvested result (the whole-step deferred loss) whose
   *  buffer is a compiled-plan result entry must survive plan invalidation for
   *  its post-boundary read; the compiled-plan teardown consults this to PARK
   *  such a buffer rather than destroy it. Merely-alive step-scoped results
   *  (the default path) are NOT registered, so parking stays whole-step-only. */
  isRegisteredStorage(storageId: number): boolean {
    for (const w of this._liveOwners(storageId)) {
      if (this._registeredState.has(w)) return true;
    }
    return false;
  }

  /** [unrolled-K P4] Live storage ids bucketed by the buffer they currently
   *  back, restricted to `candidates`. ONE pass over live storages; `bufOf`
   *  maps a storage to its current GPUBuffer (the caller supplies it to avoid a
   *  backend import here). Used by the compiled-plan teardown to decide which
   *  plan-owned buffers a live storage still aliases (park) vs. which are free
   *  to destroy — keying on the current backing buffer catches views/aliases a
   *  per-id list would miss. */
  bucketLiveStorageIdsByBuffer<B>(
    candidates: Set<B>,
    bufOf: (s: StorageHandle) => B,
  ): Map<B, number[]> {
    const m = new Map<B, number[]>();
    for (const [id, s] of this.allStorages) {
      const buf = bufOf(s);
      if (!candidates.has(buf)) continue;
      const arr = m.get(buf);
      if (arr) arr.push(id);
      else m.set(buf, [id]);
    }
    return m;
  }

  /**
   * Release tensor refs for step-scoped temporaries — driven by the DERIVED
   * classification (task #70 D2 authoritative). For each tracked storage the
   * `_derived` verdict decides survival: a storage is demoted (one wrapper claim
   * released) iff it is `stepScoped` — a live owner exists and NO kept holder
   * keeps it (persistent SNAP∪REG / graph-retention / sidecar-pin / view-base
   * retain). Persistent tensors whose storage changed within the step (Adam m/v
   * via _updateLazyRef) stay alive because a live SNAP/REG wrapper is still in
   * W(s), even if the storage id is new. Reading the whole owner SET (not a
   * single steal-able slot) is what makes the former reap-live / stale-disposed-
   * slot class (§D1-log #1) unconstructible.
   */
  releaseStepTemps(maxGen?: number): number {
    if (!this._stepStartTensors) return 0;
    let released = 0;
    // Precompute live view-base ids ONCE (keeps the sweep O(N)).
    const liveViewBases = this._liveViewBases();
    // Iterate a snapshot of the owner-set keys — the loop mutates _ownerSet via
    // _liveOwners' lazy pruning, so we must not iterate the live map.
    for (const id of [...this._ownerSet.keys()]) {
      const derived = this._derived(id, maxGen, liveViewBases);
      // Demote only a genuinely step-scoped storage: a live owner exists and no
      // kept holder keeps it. (`hasLiveOwner` false ⇒ nothing to release here —
      // the GC scan / view-base retain handles those.)
      if (!derived.stepScoped) continue;
      if (rcGet(id) > 0) {
        rcRelease(id, "stepScoped");
        released++;
      }
    }
    this._stepStartTensors = null;
    if (epochTraceEnabled()) {
      epochTrace(`releaseStepTemps released=${released}`);
    }
    return released;
  }

  getNextStorageId(): number {
    return getNextStorageId();
  }

  stats(): {
    totalStorages: number;
    reachableStorages: number;
    unreachableStorages: number;
  } {
    let reachable = 0;
    for (const [id] of this.allStorages) {
      if (rcGet(id) > 0) reachable++;
    }
    return {
      totalStorages: this.allStorages.size,
      reachableStorages: reachable,
      unreachableStorages: this.allStorages.size - reachable,
    };
  }

  /**
   * Debug: snapshot the REACHABLE (rc>0) storage set for cross-boundary
   * leak diffs (task #79). Returns per-storage id/shape/dtype/view info so a
   * caller can diff two snapshots and histogram the delta by shape. Pure read;
   * cheap enough to call between generations, not on a hot path.
   */
  debugLiveSet(): Array<{
    id: number;
    shape: readonly number[];
    dtype: string;
    view: boolean;
    base?: number;
  }> {
    const out: Array<{
      id: number;
      shape: readonly number[];
      dtype: string;
      view: boolean;
      base?: number;
    }> = [];
    for (const [id, storage] of this.allStorages) {
      if (rcGet(id) <= 0) continue;
      const bt = storage.backendTensor;
      out.push({
        id,
        shape: bt.shape ?? [],
        dtype: bt.dtype ?? "?",
        view: bt.ownsBuffer === false,
        base: storage.baseStorageId,
      });
    }
    return out;
  }

  debugCounters(): { destroyed: number } {
    const d = this._debugDestroyCount;
    this._debugDestroyCount = 0;
    return { destroyed: d };
  }

  reset(): void {
    this.allStorages.clear();
    this._ownerSet.clear();
    // Drop registered state (REG) — a fresh tracker owns no persistent state.
    this._registeredState = new WeakSet<object>();
    // Reset the step snapshot (task #70 D4).
    this._stepStartTensors = null;
  }

  /**
   * Instance-boundary teardown (a NEW engine is being constructed). The
   * storage tracker + refcount registry are MODULE-GLOBAL singletons shared
   * across every RuntimeEngine in the process; a previous engine's residual
   * storages linger in `allStorages` with their owning tensors still alive
   * (model params, optimizer m/v, lr tensors) until JavaScript GC collects
   * them — at an UNPREDICTABLE time DURING the next engine's training run. On
   * collection, the WeakRef scan in destroyUnreachable() releases the claim,
   * the storage's `bt.destroy()` runs, and its buffer is `bufferPool.release()`d
   * back into the SHARED pool MID-STEP of the live run — the exact
   * "released-to-pool mid-step" corruption class (CLAUDE.md WebGPU Buffer Pool
   * Invariants). The live run then acquires that buffer for a fresh tensor and
   * either reads stale data or has its data destroyed under it. Symptom: the
   * second-and-later training run in one process sporadically (GC-timing
   * dependent) diverges from step ~0 (task #84; the executor template cache got
   * the analogous instance-boundary reset — clearTemplateCacheForNewEngine —
   * for the SAME cross-instance-interference reason, but the storage tracker
   * was missed).
   *
   * The fix: at construction, quiescently tear down every residual storage NOW
   * (before the new engine does any work), so their buffers are released to the
   * pool at a safe boundary, never mid-step of the live run. Must be called
   * only when the backend is live (a buffer pool exists to release into) and
   * before the new engine allocates anything. Idempotent; safe with an empty
   * tracker (no residue). Asserts the tracker is empty afterward — a nonzero
   * residual would mean a storage escaped teardown and could still leak into
   * the next run (the guard that catches future regressions of this class).
   */
  disposeAllForNewEngine(): void {
    // ORPHAN the previous engine's residual storages — do NOT destroy them, and
    // do NOT touch the refcount registry. Mechanism, precisely:
    //
    // A previous engine's tensor WRAPPERS are still live JS objects (its
    // model/opt went out of scope but GC has not run) with live rc. The
    // corruption (#84) was: when GC finally collected one of those wrappers
    // DURING the next engine's step, the tracker's owner-SET scan in
    // destroyUnreachable() (or the wrapper's FinalizationRegistry) released the
    // claim, the storage's `bt.destroy()` ran, and its buffer was
    // `bufferPool.release()`d back into the SHARED pool MID-STEP of the live run
    // — the "released-to-pool mid-step" class — where the live run had already
    // acquired it. Second-and-later runs diverged, GC-timing-sporadically.
    //
    // Forgetting the storages HERE (dropping their ids from allStorages +
    // the owner sets) breaks that path at its root: destroyUnreachable() no
    // longer iterates them, so their `bt.destroy()` is never called and their
    // buffers are never returned to the pool — they are orphaned (leaked for the
    // process lifetime, but a NEW engine means the OLD one is finished; this is
    // a bounded one-time-per-construction leak, the same one-live-engine-at-a-
    // time contract clearTemplateCacheForNewEngine already assumes).
    //
    // Deliberately NOT done here: (a) destroying the buffers (they'd be re-pooled
    // now AND destroyed again by the still-live wrapper's later GC — a
    // double-release; measured to make the next run diverge deterministically);
    // (b) rcReset() (the rc registry is keyed by storage id and the previous
    // engine's ids never collide with the next engine's monotonic ids, so its
    // entries are inert; wiping it wholesale instead nulls the CURRENT engine's
    // live scalars' refcounts on later constructions — the shape-[1] `_t`/`lr`
    // reclaimed-read STRICT_LIFETIME throw). The orphaned ids' lingering rc
    // entries are harmless: the wrapper's FR does `rcRelease` on an id whose
    // storage we already forgot, which never reaches destroyStorageIds.
    //
    // Root cause of #84: this instance-boundary teardown was simply ABSENT — the
    // executor template cache got clearTemplateCacheForNewEngine for the
    // identical cross-instance-interference reason, but the storage tracker was
    // never given the analogous reset.
    this.allStorages.clear();
    this._ownerSet.clear();
    // Clear REG (registered state, task #70 D3): a new engine must not inherit
    // the dead engine's registered module/optimizer state — the same
    // cross-instance-interference class disposeAllForNewEngine exists for (#84,
    // #74). A WeakSet cannot be iterated to clear; rebuild it empty. The dead
    // engine's wrappers are going out of scope (its model/opt is finished), so
    // their membership is dropped wholesale here; a lingering-but-live dead-engine
    // wrapper cannot re-register (the engine is done). This is the doc §D3
    // "cross-engine behavior" contract.
    this._registeredState = new WeakSet<object>();
    // Clear the step snapshot (task #70 D4 — the row-8 root cause). A fresh
    // process starts with `_stepStartTensors === null`, and releaseStepTemps is a
    // NO-OP while null (it reaps nothing until the first snapshotForStep). The
    // SECOND engine in a process must start from that same null state — otherwise
    // it inherits the DEAD engine's stale snapshot WeakSet (which contains none of
    // the new engine's tensors), so the new engine's very first implied-boundary
    // releaseStepTemps runs against that stale snapshot and reaps the new engine's
    // live step-0 forward activation (owners=1, snap=false) → a [lifetime]
    // reclaimed-read throw when the optimizer force reads it.
    //
    // This is the gpt2-memorization overfit FP (indictment row 8): it is the
    // SECOND-ENGINE-IN-PROCESS class (#84), NOT a gen-perturbation. The D1 and D2
    // attributions ("concurrent test perturbs the SHARED generation counter" /
    // "runahead gen-scoping") were BOTH wrong — the reaped storage is at
    // maxGen=0/stamp=0, a clean FIRST boundary, and it reproduces DETERMINISTICALLY
    // in a single process with no parallelism: engine #0 OK, engine #1+ throw
    // (tools/t-second-engine-overfit-probe.ts). The first engine never hit it
    // because its snapshot was still null on its first boundary; the second engine
    // inherited engine #0's stale non-null snapshot. The honesty invariant of this
    // campaign: the differential surfaced that both prior attributions were wrong,
    // and D4 records the correction rather than shipping the mis-located fix (a
    // per-engine `_stepGen` counter was prototyped, measured NULL against this
    // repro, and reverted — the stale snapshot was the whole bug).
    this._stepStartTensors = null;
    if (this.allStorages.size !== 0) {
      throw new Error(
        `[storage-tracker] disposeAllForNewEngine left ${this.allStorages.size} residual storages — a storage escaped instance-boundary teardown and would leak into the next engine run (the second-in-process bimodality class, task #84).`,
      );
    }
  }

  getStorage(storageId: number): StorageHandle | undefined {
    return this.allStorages.get(storageId);
  }

  getLiveOwnedBuffers(): Set<unknown> {
    const buffers = new Set<unknown>();
    for (const [, storage] of this.allStorages) {
      const bt = storage.backendTensor;
      const buf = (bt as { buffer?: unknown }).buffer;
      if (bt.ownsBuffer !== false && buf) {
        buffers.add(buf);
      }
    }
    return buffers;
  }
}

/** Global storage tracker instance */
export const storageTracker = new StorageTracker();

// ============================================================================
// Early Release Helpers
// ============================================================================

/**
 * Check if a storage can be safely released during plan execution.
 * Uses refcount: rc <= 0 means no one references it.
 * Also checks view aliasing (another active storage uses this as base).
 */
export function canSafelyRelease(
  storage: StorageHandle,
  activeStorages: Map<number, StorageHandle>,
): boolean {
  if (rcGet(storage.id) > 0) return false;
  for (const [, active] of activeStorages) {
    if (active.baseStorageId === storage.id) return false;
  }
  return true;
}

/**
 * Release a buffer during plan execution.
 */
export function releaseBufferImmediate(storage: StorageHandle): void {
  const bt = storage.backendTensor;

  // Views: release base ref but don't destroy (don't own buffer).
  // Don't unregister — the view may still be referenced by the shared encoder.
  // Clear baseStorageId to prevent double-release when destroyStorageIds
  // later processes this view at destroyUnreachable time.
  if (bt.ownsBuffer === false) {
    if (storage.baseStorageId !== undefined) {
      rcRelease(storage.baseStorageId, "earlyRelease.viewBase");
      storage.baseStorageId = undefined;
    }
    return;
  }

  // Owning storage: release base ref if view, unregister, destroy.
  if (storage.baseStorageId !== undefined) {
    rcRelease(storage.baseStorageId, "earlyRelease.base");
    storage.baseStorageId = undefined;
  }
  rcDelete(storage.id);
  storageTracker.unregister(storage.id);
  if (bt.destroy) bt.destroy();
}

/**
 * Find and release all dead tensors at a given execution step.
 */
export function releaseDeadTensors(
  lifetimes: Map<number, TensorLifetime> | null,
  step: number,
  outputNodeIds: Set<number> | null,
  alreadyReleased: Set<number>,
  nodeToStorage: Map<number, StorageHandle>,
): void {
  if (!lifetimes || !outputNodeIds) return;
  const deadNodeIds = findDeadTensorsAtStep(
    lifetimes,
    step,
    outputNodeIds,
    alreadyReleased,
  );
  for (const deadId of deadNodeIds) {
    const storage = nodeToStorage.get(deadId);
    if (storage && canSafelyRelease(storage, nodeToStorage)) {
      releaseBufferImmediate(storage);
      nodeToStorage.delete(deadId);
      alreadyReleased.add(deadId);
    }
  }
}
