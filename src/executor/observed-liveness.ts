/**
 * Stage-1 observed cross-plan liveness (docs/stage4-compile-from-ir.md §Stage-1).
 *
 * The build-from-IR path (TORCHLETTE_BUILD_FROM_IR=1) has no after-the-fact
 * refcount signal, so it harvests the FULL action-output set of every plan —
 * every intermediate becomes an exclusive planner result entry (≈+34% memory
 * over the recorded cutover's live-result survivor set). This module recovers
 * the survivor set by OBSERVATION: from the first executed steps it learns which
 * harvested results the plan set actually consumes (the same signal the lowered
 * path prunes by), converges per recurring template, and rebuilds the template
 * with the pruned result set — collapsing the over-harvest to the default's
 * survivor memory.
 *
 * Mechanism (all cross-plan value identity flows through ONE stamp — no second
 * naming scheme; the stamp is the remat edge name stage-3 reuses):
 *  - Each harvested result is STAMPED (templateFp, nodeIndex, oi) on its storage
 *    at the replay-harvest chokepoint (compiled-plan.ts).
 *  - CONSUMED: a later plan resolving an external input whose storage carries a
 *    stamp records that pair into the producer template's needed-set (recorded
 *    eagerly at the cross-plan read seam, so it holds even if the value is
 *    reclaimed before the step ends).
 *  - SURVIVED: at the step boundary any stamped result whose storage is still
 *    alive after the step's reclamation records into its template's needed-set
 *    (params / optimizer state / user-held tensors land here).
 *  - CONVERGE + REBUILD: after K=3 fully-observed no-growth steps (no new
 *    template that step), the template is invalidated and rebuilt with
 *    harvestPairs = needed ∪ mandatory (terminal + declared outputs). Binary
 *    result/temp suffices: the needed-set converges to exactly the default
 *    cutover's live-result survivor set.
 *  - GUARD (bind-time): a plan built AFTER observation that reads a PRUNED
 *    producer output misses at the consumer's external-slot pre-population
 *    (before any side effect). Sound recovery = template invalidation + fresh
 *    lowered re-collection (recomputes the slice), GATED by a step-scoped
 *    in-place-committed counter: recompute reads CURRENT storages, so if any
 *    in-place op committed since the producer's pruned replay the value would
 *    silently differ — that miss FAILS LOUDLY instead (no version substrate
 *    exists to detect per-storage staleness). Never silent staleness.
 *  - REBUILD LIMIT: a template whose needed-set grows > 2× its converged size
 *    (guard churn) is PINNED to the conservative full harvest permanently.
 *
 * The dirty-miss counter is THE measurement deciding whether stage-3 remat ever
 * needs to serve this path; surfaced on getPayloadThrashStats().
 */

import { storageTracker } from "../graph/storage-tracker";
import type { StorageHandle } from "../graph/types";

/** Cross-plan identity of a harvested VALUE, independent of any live buffer. */
export interface ResultStamp {
  fp: number;
  ni: number;
  oi: number;
}

const K_HYSTERESIS = 3;
const REBUILD_GROWTH_LIMIT = 2;
/** Steps a template may sit idle (not executed) before its compiled plan is
 *  RETIRED — invalidated so its planner result entries are released. One-shot
 *  warmup templates (step-0/1 graph variants that never recur) otherwise pin
 *  their full conservative harvest forever: they can never converge (no
 *  re-execution to observe), and at 124M-class configs their harvested results
 *  are ~2.2 GB of dead registry buffers — THE residual over the recorded
 *  cutover, which never compiles a plan that doesn't recur. Retire is gated on
 *  every storage the plan's last replay harvested being ALREADY DESTROYED
 *  (storageTracker), so no live reader can dangle — strictly more conservative
 *  than the convergence invalidation (which rebuilds with live survivors). A
 *  retired template that re-executes simply rebuilds from IR (no lowered
 *  re-execution needed); with K_IDLE=3 an every-other-step template never
 *  thrashes. */
const K_IDLE = 3;

/** Why a pair entered the needed-set (attribution telemetry): "c" cross-plan
 *  consumed, "s" survived the step directly, "b" survived only VIA ITS VIEW
 *  BASE (the handle died but baseStorageId is alive — the deliberately-
 *  conservative view class), "g" guard-miss, "r" READ BACK (item()/cpu()/
 *  readTopK/startItemReadback — consumption OUTSIDE plan order, invisible to
 *  the plan-read seam; the loss tensor is the canonical case). First
 *  observation wins except a later non-"b" upgrades a "b" (a pair both
 *  consumed and base-alive is not a view-conservatism cost). */
type NeededSource = "c" | "s" | "b" | "g" | "r";

interface TemplateObs {
  needed: Set<string>; // union of observed needed "ni:oi"
  neededSrc: Map<string, NeededSource>; // pair → first/strongest source
  stableSteps: number; // consecutive fully-observed no-growth steps
  converged: boolean; // reached K → pruning active
  convergedSize: number; // needed.size at convergence (rebuild-limit base)
  pinned: boolean; // conservative full-harvest, permanently
  grewThisStep: boolean;
  executedThisStep: boolean;
  idleSteps: number; // consecutive boundaries without execution (retire clock)
  // ── Stage-3 remat unification (S3.0 observation; docs §Stage 3) ────────────
  /** pair → template fp of its LAST cross-plan reader within the step (the
   *  consumer whose read carried the highest step-scoped consumption order).
   *  The step-global release seam frees a consumed-only pair's registry entry
   *  INTO its last reader's build: the reader's temps overlay the dead
   *  activation after its final read (safe under strictly-sequential plans —
   *  the producer re-writes the entry before any next-step consumer reads). */
  lastReader: Map<string, number>;
  /** pair → consecutive fully-observed steps with the SAME single lastReader.
   *  Releasable only after K_HYSTERESIS stable steps (same activation
   *  condition as pruning: warmup order instability never releases). */
  lastReaderStable: Map<string, number>;
  /** pair → step-global release REVOKED permanently (a guard miss proved a
   *  consumer observation didn't see; never trust its last-reader again). */
  releaseRevoked: Set<string>;
  /** pairs that EVER survived a step boundary (directly or via view base).
   *  Distinct from neededSrc (which keeps the FIRST source — a pair consumed
   *  before it was seen surviving stays "c" there): releasability must know
   *  about ANY survival, ever, since a surviving value is user-visible and
   *  must never be overlaid. */
  everSurvived: Set<string>;
  /** pairs EVER read back (item()/cpu()/readTopK/startItemReadback) — readers
   *  outside plan order whose timing observation cannot bound. Kept in the
   *  harvest (via needed) and permanently excluded from step-global release
   *  (a future readback could land after any within-step release point). */
  everReadback: Set<string>;
  /** pair → planner registry entry backing the harvested result in the
   *  CURRENT build (bytes for telemetry; mandatory pairs are terminal/declared
   *  outputs — never releasable). `op` is the producing node's op label
   *  (debug attribution only). Re-registered wholesale at every build. */
  resultEntries: Map<
    string,
    { entryIdx: number; bytes: number; mandatory: boolean; op?: string }
  >;
  /** Registry generation resultEntries was registered against (a registry
   *  reset invalidates entry indices — identity-compared, epoch-keyed). */
  resultEntriesGen: number;
  /** entryIdx → pairs registered on it (an in-place output and its harvested
   *  view can share one entry; release requires EVERY pair on the entry to be
   *  independently releasable with the SAME last reader). */
  entryToPairs: Map<number, string[]>;
}

const key = (ni: number, oi: number): string => `${ni}:${oi}`;

// ── Module state (step-scoped structures cleared at observeStepBoundary) ──────
let enabled = false;
const templates = new Map<number, TemplateObs>();

/**
 * [task #98 phase 4 — WITNESS-TIME HARVEST] Per producer-template `fp`, the set
 * of "ni:oi" pairs the step-tape recorder physically OBSERVED read across the
 * two consecutive identical executed steps that made its tape eligible (§4.1).
 *
 * This is the #97 resolution: the observation seam (`observeConsumed`) is fed
 * ONLY by the compiled external-slot bind, so a cross-plan read that resolves
 * LOWERED through `getInputStorage` — the canonical case being the BACKWARD
 * pass re-reading a checkpoint-RECOMPUTED forward activation — is structurally
 * invisible to it, and its producer template PRUNES the activation from the
 * generated harvest (the `contiguous[512,768]` STOP). The tape witnesses at
 * END-OF-STEP time, AFTER backward + recompute ran, so every such read was
 * physically observed; the recorder records it (`stObserveWitnessRead`) and,
 * on eligibility, publishes the union here. `prunedHarvest` unions these pairs
 * into `keepAlways` so the pruned build keeps them — the build-WITH-execution
 * guarantee the recorded build had, promoted to the tape stratum.
 *
 * SINGLE SOURCE / NO SECOND OWNER (ruling 1): the ONLY producer of this map is
 * the step-tape recorder's eligibility path (`stEndStep` → `setWitnessedHarvest`).
 * observed-liveness merely CONSULTS it — it never authors it. Gen-independent
 * (keyed by template fp + ni:oi, the same coordinate the needed-set uses).
 */
const witnessedHarvest = new Map<number, Set<string>>();

/**
 * Publish (or replace) the witnessed harvest set for a producer template. Called
 * by the step-tape recorder at tape eligibility (`step-tape.ts stEndStep`). A
 * fresh eligibility for the same fp REPLACES the set (re-witnessing under a
 * changed declaration re-establishes it). Passing an empty set clears the entry
 * (a witness step that observed no cross-plan lowered read of this fp).
 */
export function setWitnessedHarvest(fp: number, pairs: Set<string>): void {
  if (pairs.size === 0) {
    witnessedHarvest.delete(fp);
    return;
  }
  witnessedHarvest.set(fp, new Set(pairs));
}

/** Drop a template's witnessed harvest (guard-4 invalidation: a re-witness must
 *  re-establish it). Called from `stInvalidateTemplate` via the recorder. */
export function clearWitnessedHarvest(fp: number): void {
  witnessedHarvest.delete(fp);
}

/** Test/telemetry: the witnessed harvest set for a template (sorted "ni:oi"). */
export function getWitnessedHarvest(fp: number): string[] {
  const s = witnessedHarvest.get(fp);
  return s ? [...s].sort() : [];
}

/** Test/telemetry: every template's witnessed harvest set (fp → sorted "ni:oi").
 *  The shadow-parity gate reads this to assert coverage of the pruned classes. */
export function getAllWitnessedHarvest(): Map<number, string[]> {
  const out = new Map<number, string[]>();
  for (const [fp, s] of witnessedHarvest) out.set(fp, [...s].sort());
  return out;
}

let stepStamped: Array<{ shId: number; checkId: number; stamp: ResultStamp }> =
  [];
let newTemplateThisStep = 0;
/** nodeId → { producer stamp, in-place-commit count at its pruned replay,
 *  released: the pair was step-globally RELEASED (stage-3 B — its buffer may
 *  be overlaid by the claimant's temps) rather than pruned-unharvested. } */
let prunedExecuted = new Map<
  number,
  { stamp: ResultStamp; mark: number; released?: boolean }
>();
let inPlaceCommits = 0;
/** [stage-3] Step-scoped consumption order: "fp|ni:oi" → the consumer fp of
 *  the HIGHEST-order read this step (last-write-wins as reads are observed in
 *  execution order — plans are strictly sequential, so observation order IS
 *  queue order). Cleared at the boundary after merging into lastReader. */
let stepLastReader = new Map<string, number>();

// ── Telemetry (folded onto getPayloadThrashStats) ────────────────────────────
let cleanMisses = 0;
let dirtyMisses = 0;
let readbackMisses = 0;
let convergenceInvalidations = 0;
let claimMisses = 0;
let pinnedTemplates = 0;
let prunedPairsRemoved = 0;
let retiredTemplates = 0;

/** Executor callback: invalidate a template's compiled plan by fingerprint
 *  (the module can't import executor.ts — circular). */
let invalidateTemplateCompiled: ((fp: number) => void) | undefined;
export function setTemplateCompiledInvalidator(fn: (fp: number) => void): void {
  invalidateTemplateCompiled = fn;
}

/** Executor callback for idle-retire: attempt to release an idle template's
 *  compiled plan. "retired" = plan destroyed now; "none" = no plan to retire;
 *  "live" = blocked — some storage its last replay harvested is still alive
 *  (retry at a later boundary). */
export type RetireResult = "retired" | "none" | "live";
let retireIdleTemplate: ((fp: number) => RetireResult) | undefined;
export function setTemplateIdleRetirer(fn: (fp: number) => RetireResult): void {
  retireIdleTemplate = fn;
}

/** [stage-3 idle-trim] Callback invoked at a STEADY step boundary (every
 *  executed template converged or pinned, no new template this step — the
 *  same activation condition as pruning). The backend registers the pool's
 *  idle-trim here; the module stays backend-agnostic. */
let steadyBoundaryTrimmer: (() => void) | undefined;
export function setSteadyBoundaryTrimmer(fn: () => void): void {
  steadyBoundaryTrimmer = fn;
}

export function setObservedLivenessEnabled(on: boolean): void {
  enabled = on;
}
export function isObservedLivenessEnabled(): boolean {
  return enabled;
}

/** [2b §5 declared-lifetime dividend] True while a MULTI-plan step TAPE replays
 *  its plan sequence. Inside a captured training step the whole step's dataflow
 *  is DECLARED, so the observation-layer's per-handle liveness verdicts do not
 *  apply to cross-plan reads: (a) the stage-3 B clear-at-release must not fire
 *  (a last-reader plan overlaying a claimed released external mid-replay would
 *  strand the next replay's read); (b) a cross-plan buffer produced by an
 *  earlier plan (or a lowered backward segment) and re-bound by the planner is
 *  reachable for the whole replay even though its RECORDING-era storage handle
 *  was demoted at the recording markStep — the destroyed-handle lifetime guard
 *  is a false positive here (correctness proven: captured trajectory tracks the
 *  uncaptured control WITHIN the cross-run fp-noise floor). Both are suppressed
 *  only for the duration of the replay; outside it the guards are unchanged. */
let stTapeReplayActive = false;
export function setStepTapeReplayActive(on: boolean): void {
  stTapeReplayActive = on;
}
export function isStepTapeReplayActive(): boolean {
  return stTapeReplayActive;
}

export function resetObservedLiveness(): void {
  templates.clear();
  stepStamped = [];
  newTemplateThisStep = 0;
  prunedExecuted = new Map();
  inPlaceCommits = 0;
  stepLastReader = new Map();
  cleanMisses = 0;
  dirtyMisses = 0;
  readbackMisses = 0;
  convergenceInvalidations = 0;
  claimMisses = 0;
  pinnedTemplates = 0;
  prunedPairsRemoved = 0;
  retiredTemplates = 0;
}

function obs(fp: number): TemplateObs {
  let t = templates.get(fp);
  if (!t) {
    t = {
      needed: new Set(),
      neededSrc: new Map(),
      stableSteps: 0,
      converged: false,
      convergedSize: 0,
      pinned: false,
      grewThisStep: false,
      executedThisStep: false,
      idleSteps: 0,
      lastReader: new Map(),
      lastReaderStable: new Map(),
      releaseRevoked: new Set(),
      everSurvived: new Set(),
      everReadback: new Set(),
      everAliased: new Set(),
      resultEntries: new Map(),
      resultEntriesGen: -1,
      entryToPairs: new Map(),
    };
    templates.set(fp, t);
  }
  return t;
}

function addNeeded(
  fp: number,
  ni: number,
  oi: number,
  src: NeededSource,
): void {
  const t = obs(fp);
  const k = key(ni, oi);
  if (!t.needed.has(k)) {
    t.needed.add(k);
    t.neededSrc.set(k, src);
    t.grewThisStep = true;
  } else if (t.neededSrc.get(k) === "b" && src !== "b") {
    // A pair kept for a stronger reason than base-survival is not a
    // view-conservatism cost — upgrade the attribution.
    t.neededSrc.set(k, src);
  }
}

/** Executor: a template cache MISS occurred this step (a new plan appeared). */
export function noteNewTemplate(): void {
  if (enabled) newTemplateThisStep++;
}

/** Executor: this template executed this step (any path). */
export function noteTemplateExecuted(fp: number): void {
  if (enabled) obs(fp).executedThisStep = true;
}

/** Step-tape replay: this template was REPLAYED via a tape skeleton. Resets
 *  ONLY the idle-retire clock — a template actively serving replays must not
 *  be idle-retired (retire destroys the compiled plan under the live skeleton;
 *  every warm capture died at ~K_IDLE steady boundaries). Deliberately does
 *  NOT set executedThisStep: replays stay invisible to the convergence/steady
 *  machinery (today's semantics) — marking them executed would let
 *  convergence invalidate + rebuild the plan mid-tape, forcing a spurious
 *  re-trace per convergence. Smallest delta that stops the retire. */
export function noteTemplateReplayed(fp: number): void {
  if (enabled) obs(fp).idleSteps = 0;
}

/** Executor: a plan that commits in-place mutations (adam / copy_ to persistent
 *  state) just executed — advances the step-scoped in-place-committed counter
 *  the guard's soundness rule reads. */
export function noteInPlaceCommit(): void {
  if (enabled) inPlaceCommits++;
}

// ── Stamp / observe ──────────────────────────────────────────────────────────

/** Stamp a harvested result and enroll it for the survival scan. checkId
 *  follows the view-base chain so an in-place param VIEW is judged by the
 *  persistent buffer owner, not the per-replay view handle (which dies at
 *  markStep even though the value persists). */
export function stampResult(
  sh: StorageHandle,
  fp: number,
  ni: number,
  oi: number,
): void {
  if (!enabled) return;
  const stamp: ResultStamp = { fp, ni, oi };
  sh.stamp = stamp;
  stepStamped.push({ shId: sh.id, checkId: sh.baseStorageId ?? sh.id, stamp });
}

/** Record cross-plan consumption of a resolved external input, if stamped.
 *  consumerFp (the READING template) feeds the stage-3 last-reader
 *  observation; undefined (lowered/legacy read sites) leaves it untracked —
 *  conservative: a pair without a stable observed last reader never releases. */
export function observeConsumed(
  storage: StorageHandle,
  consumerFp?: number,
): void {
  if (!enabled) return;
  const stamp = storage.stamp;
  if (!stamp) return;
  addNeeded(stamp.fp, stamp.ni, stamp.oi, "c");
  if (consumerFp !== undefined) {
    // Last-write-wins: observation order is queue order (strictly-sequential
    // plans), so the final write for a pair this step IS its last reader.
    stepLastReader.set(`${stamp.fp}|${key(stamp.ni, stamp.oi)}`, consumerFp);
  }
}

/**
 * [stage-3 A] Record a READBACK of a stamped result (RuntimeEngine.cpu /
 * readTopK / startItemReadback, after force). Pure observation at an
 * already-resolved storage — it must never perturb forcing or ordering (the
 * loss-readback semantics are history-sensitive; see CLAUDE.md "Moving
 * loss.item() after backward"). A read-back pair is needed (kept in the
 * harvest even when mandatory-pruning is active) and permanently excluded
 * from any future step-global release (readback timing is unordered).
 */
export function observeReadback(storage: StorageHandle): void {
  if (!enabled) return;
  const stamp = storage.stamp;
  if (!stamp) return;
  addNeeded(stamp.fp, stamp.ni, stamp.oi, "r");
  obs(stamp.fp).everReadback.add(key(stamp.ni, stamp.oi));
}

/**
 * [stage-3 A] A readback found its tensor STILL PENDING after force — the
 * pair was pruned as observed-dead and this is its first-ever reader (the
 * epistemic boundary, same as the stage-1 guard). Self-heal: grow the
 * needed-set, revoke, invalidate the producer's pruned build (next step
 * harvests the value). Returns the stamp when the miss was a pruned pair
 * (caller throws a descriptive error naming the recovery); undefined if
 * unrelated (caller falls through to its normal not-materialized error).
 */
export function readbackMiss(
  nodeId: number,
  oi: number,
): ResultStamp | undefined {
  if (!enabled) return undefined;
  const hit = prunedExecuted.get(nodeId);
  if (!hit || hit.stamp.oi !== oi) return undefined;
  const { stamp } = hit;
  readbackMisses++;
  addNeeded(stamp.fp, stamp.ni, stamp.oi, "r");
  obs(stamp.fp).everReadback.add(key(stamp.ni, stamp.oi));
  revokeRelease(stamp.fp, stamp.ni, stamp.oi);
  invalidateTemplateCompiled?.(stamp.fp);
  return stamp;
}

/** [stage-3 B] A harvested view result chained to a STAMPED base — the
 *  base pair's buffer is now readable through the view's stamp, so the base
 *  pair is permanently excluded from step-global release. Called from the
 *  replay-harvest chokepoint (every alias flows through it). */
export function noteAliasedBase(stamp: ResultStamp): void {
  if (!enabled) return;
  obs(stamp.fp).everAliased.add(key(stamp.ni, stamp.oi));
}

/** Register the pairs a pruned plan EXCLUDED from its harvest, so a later
 *  guard-miss can name the culprit and decide clean/dirty. Called at the
 *  pruned plan's replay-harvest with the current step's node ids. */
export function registerPrunedExecution(
  fp: number,
  ni: number,
  oi: number,
  nodeId: number,
  released?: boolean,
): void {
  if (!enabled) return;
  prunedExecuted.set(nodeId, {
    stamp: { fp, ni, oi },
    mark: inPlaceCommits,
    released,
  });
}

/**
 * Bind-time guard (task #98 phase 5 — DEMOTED to a should-never-fire assertion).
 * A late consumer's external input (producer nodeId, oi) is missing. If it was a
 * pruned producer, this is a pruned-then-demanded read the harvest should have
 * kept — it MUST NOT happen: both prune-soundness classes are covered upstream
 * (the overlay class by `graphHeldAt` at the claim seam, task #97 stage 2; the
 * checkpoint-recompute + shape=[] scaler-scalar classes by the recorded build's
 * harvest / the witness-time harvest, step-object §4). A zero-fire soak across
 * the full config matrix (docs/step-object-design.md §6 Phase 5) confirmed the
 * old `RecoverableGuardMiss` clean-recovery never fired on any path. So the
 * recovery net (grow-needed-set → invalidate → re-collect lowered next step) is
 * deleted and a match here throws LOUDLY with full context.
 *
 * Returns false (unrelated miss → caller rethrows the original "Input not
 * ready") ONLY when the missing input is NOT a pruned producer. A matched
 * pruned-producer miss throws — it never returns.
 */
export function guardMiss(nodeId: number, oi: number): boolean {
  if (!enabled) return false;
  const hit = prunedExecuted.get(nodeId);
  if (!hit || hit.stamp.oi !== oi) return false;
  const { stamp, mark } = hit;
  // Counters retained as the observable "never fired" signal (gated at 0 by
  // test/observed-liveness.spec.ts + witness-harvest.spec.ts). A fire here is a
  // hard bug: a covered class regressed, or a NEW pruned-read class exists that
  // the harvest coverage (graphHeldAt + recorded/witness harvest) does not net.
  if (hit.released) claimMisses++;
  const dirty = inPlaceCommits > mark;
  if (dirty) dirtyMisses++;
  else cleanMisses++;
  throw new Error(
    `[observed-liveness] guardMiss ASSERTION FIRED (should never happen, task #98 phase 5): ` +
      `a pruned producer's value was demanded at a late consumer's external slot. ` +
      `template=0x${stamp.fp.toString(16)} node=${stamp.ni} oi=${stamp.oi} nodeId=${nodeId} ` +
      `released=${hit.released ?? false} ${dirty ? `dirty(${inPlaceCommits - mark} in-place op(s) committed since replay)` : "clean"}. ` +
      `Both prune-soundness classes are supposed to be unconstructible: the overlay class via graphHeldAt ` +
      `(task #97 stage 2) and the checkpoint-recompute / shape=[] scaler-scalar classes via the recorded/witness harvest ` +
      `(step-object §4). A fire means one of those nets regressed or a new pruned-read class exists — investigate the ` +
      `producer template's harvest set, do NOT re-add lowered re-collection recovery.`,
  );
}

// ── Step boundary: survival scan + convergence ───────────────────────────────

/** Called at the step boundary AFTER the step's reclamation (destroyUnreachable
 *  + releaseStepTemps). Records survivors, advances hysteresis, converges
 *  templates, and clears step-scoped state. */
export function observeStepBoundary(): void {
  if (!enabled) return;
  // Survival: a stamped result whose storage survived reclamation is needed.
  // Attribution: "s" = the handle itself survived; "b" = only its view BASE
  // survived (the deliberately-conservative view class).
  for (const { shId, checkId, stamp } of stepStamped) {
    if (!storageTracker.isDestroyed(checkId)) {
      const direct = shId === checkId || !storageTracker.isDestroyed(shId);
      addNeeded(stamp.fp, stamp.ni, stamp.oi, direct ? "s" : "b");
      // [stage-3] Survival is recorded unconditionally (addNeeded keeps only
      // the FIRST source): a pair that ever survives is user-visible and must
      // never be step-globally released.
      obs(stamp.fp).everSurvived.add(key(stamp.ni, stamp.oi));
    }
  }
  // [stage-3] Merge this step's observed last readers into per-pair stability.
  // A pair keeps its stability streak only while the SAME consumer stays its
  // last reader across fully-observed steps; any change (including "consumed
  // in prior steps but not this one" while the producer executed) resets it.
  for (const [k, consumerFp] of stepLastReader) {
    const bar = k.indexOf("|");
    const fp = Number(k.slice(0, bar));
    const pair = k.slice(bar + 1);
    const t = obs(fp);
    if (t.lastReader.get(pair) === consumerFp) {
      t.lastReaderStable.set(pair, (t.lastReaderStable.get(pair) ?? 0) + 1);
    } else {
      t.lastReader.set(pair, consumerFp);
      t.lastReaderStable.set(pair, 1);
    }
  }
  for (const [fp, t] of templates) {
    if (!t.executedThisStep || t.lastReader.size === 0) continue;
    for (const pair of t.lastReader.keys()) {
      if (!stepLastReader.has(`${fp}|${pair}`)) t.lastReaderStable.set(pair, 0);
    }
  }
  const noNew = newTemplateThisStep === 0;
  // [stage-3 idle-trim] Steady = no new template this step AND every template
  // that executed is converged or pinned (computed before the loop clears the
  // executedThisStep flags). Idle warmup residue is only reclaimed once the
  // plan set has settled — the same activation condition as pruning.
  let steady = noNew;
  let anyExecuted = false;
  for (const t of templates.values()) {
    if (t.executedThisStep) {
      anyExecuted = true;
      if (!t.converged && !t.pinned) steady = false;
    }
  }
  for (const [fp, t] of templates) {
    if (!t.executedThisStep) {
      // Idle-retire clock: a template that stops executing (one-shot warmup
      // variants — they can never converge) releases its compiled plan once
      // idle K_IDLE boundaries AND none of its harvested storages is alive.
      if (++t.idleSteps >= K_IDLE && retireIdleTemplate) {
        const r = retireIdleTemplate(fp);
        if (r === "retired") retiredTemplates++;
        if (r !== "live") t.idleSteps = 0; // nothing left to retire — rest the clock
      }
      continue;
    }
    t.idleSteps = 0;
    if (t.pinned) {
      t.grewThisStep = false;
      t.executedThisStep = false;
      continue;
    }
    if (t.converged) {
      // Post-convergence growth is guard-driven (a late consumer). Cap churn:
      // a needed-set that grows past 2× its converged size pins conservative.
      if (
        t.grewThisStep &&
        t.needed.size > REBUILD_GROWTH_LIMIT * t.convergedSize
      ) {
        t.pinned = true;
        pinnedTemplates++;
      }
    } else if (noNew && !t.grewThisStep) {
      if (++t.stableSteps >= K_HYSTERESIS) {
        t.converged = true;
        t.convergedSize = Math.max(1, t.needed.size);
        // Invalidate the conservative compiled plan so the next execution
        // rebuilds it (build-from-IR) with the pruned needed-set — where the
        // memory win actually lands.
        convergenceInvalidations++;
        invalidateTemplateCompiled?.(fp);
      }
    } else {
      t.stableSteps = 0;
    }
    t.grewThisStep = false;
    t.executedThisStep = false;
  }
  stepStamped = [];
  prunedExecuted = new Map();
  newTemplateThisStep = 0;
  inPlaceCommits = 0;
  stepLastReader = new Map();
  if (steady && anyExecuted) steadyBoundaryTrimmer?.();
}

// ── Pruning application (build-from-IR harvest) ───────────────────────────────

/**
 * If the template has converged (and is not pinned), return the pruned harvest
 * pairs = actionPairs ∩ (needed ∪ keepAlways); the EXCLUDED pairs are returned
 * so the caller can record them for the bind-time guard. Returns undefined to
 * use the full conservative action-output set (not converged / pinned /
 * disabled). [stage-3 A] `keepAlways` is the TERMINAL node only — declared
 * outputs are pruned when observation (plan reads + survival + readbacks)
 * proves them dead; see the executor call site for the rationale.
 *
 * [task #98 phase 4] The WITNESSED harvest set (§4.1) is unioned into the keep
 * set: a pair the step-tape recorder observed read across two identical executed
 * steps is kept, EVEN THOUGH `observeConsumed` (the needed-set's only feed) never
 * saw the read. This is the #97 unblock — the checkpoint-recompute `contiguous`
 * is witnessed (backward read it lowered) and therefore never pruned. The
 * witnessed set is DERIVED data with one producer (the recorder's eligibility
 * path); consulting it here is the single-source-at-the-seam pattern, not a
 * second harvest owner.
 */
export function prunedHarvest(
  fp: number,
  actionPairs: Array<{ i: number; oi: number }>,
  keepAlways: Set<string>,
):
  | {
      kept: Array<{ i: number; oi: number }>;
      excluded: Array<{ i: number; oi: number }>;
    }
  | undefined {
  if (!enabled) return undefined;
  const t = templates.get(fp);
  if (!t || t.pinned || !t.converged) return undefined;
  const witnessed = witnessedHarvest.get(fp);
  const kept: Array<{ i: number; oi: number }> = [];
  const excluded: Array<{ i: number; oi: number }> = [];
  for (const p of actionPairs) {
    const k = key(p.i, p.oi);
    if (t.needed.has(k) || keepAlways.has(k) || witnessed?.has(k)) kept.push(p);
    else excluded.push(p);
  }
  prunedPairsRemoved += excluded.length;
  return { kept, excluded };
}

// ── Stage-3 remat unification: result-entry registration + releasability ─────

/**
 * Register the planner registry entries backing a template's harvested results
 * (called by the executor after every build-from-IR build; wholesale-replaces
 * the previous build's registrations — entries are per-build). `gen` is the
 * registry generation the indices are valid for.
 */
export function registerResultEntries(
  fp: number,
  gen: number,
  entries: Array<{
    ni: number;
    oi: number;
    entryIdx: number;
    bytes: number;
    mandatory: boolean;
    op?: string;
  }>,
): void {
  if (!enabled) return;
  const t = obs(fp);
  t.resultEntries = new Map();
  t.resultEntriesGen = gen;
  t.entryToPairs = new Map();
  for (const e of entries) {
    const k = key(e.ni, e.oi);
    t.resultEntries.set(k, {
      entryIdx: e.entryIdx,
      bytes: e.bytes,
      mandatory: e.mandatory,
      op: e.op,
    });
    let l = t.entryToPairs.get(e.entryIdx);
    if (!l) t.entryToPairs.set(e.entryIdx, (l = []));
    l.push(k);
  }
}

/**
 * [stage-3 B] Entry-level releasability: the stable last reader of ALL pairs
 * registered on the entry — undefined unless every pair is independently
 * releasable AND they agree on the reader (an entry is a single buffer; one
 * unreleasable alias poisons it).
 */
export function releasableEntryReader(
  fp: number,
  entryIdx: number,
): number | undefined {
  const t = templates.get(fp);
  const pairs = t?.entryToPairs.get(entryIdx);
  if (!pairs || pairs.length === 0) return undefined;
  let reader: number | undefined;
  for (const pk of pairs) {
    const bar = pk.indexOf(":");
    const r = releasableLastReader(
      fp,
      Number(pk.slice(0, bar)),
      Number(pk.slice(bar + 1)),
    );
    if (r === undefined) return undefined;
    if (reader === undefined) reader = r;
    else if (reader !== r) return undefined;
  }
  return reader;
}

/** S3.0 attribution: the top-N registered result entries per template by
 *  bytes, with class + op + last reader — identifies WHAT the held classes
 *  physically are. Debug-only. */
export function debugTopHeldPairs(topN = 10): Record<
  string,
  Array<{
    pair: string;
    op?: string;
    MB: number;
    cls: string;
    lastReader?: string;
  }>
> {
  const out: Record<
    string,
    Array<{
      pair: string;
      op?: string;
      MB: number;
      cls: string;
      lastReader?: string;
    }>
  > = {};
  for (const [fp, t] of templates) {
    const seen = new Set<number>();
    const rows: Array<{
      pair: string;
      op?: string;
      MB: number;
      cls: string;
      lastReader?: string;
    }> = [];
    for (const [pair, e] of t.resultEntries) {
      if (seen.has(e.entryIdx)) continue;
      seen.add(e.entryIdx);
      const bar = pair.indexOf(":");
      const ni = Number(pair.slice(0, bar));
      const oi = Number(pair.slice(bar + 1));
      let cls: string;
      if (releasableLastReader(fp, ni, oi) !== undefined) cls = "rel";
      else if (!t.converged || t.pinned) cls = "unconverged";
      else if (t.releaseRevoked.has(pair)) cls = "revoked";
      else if (t.neededSrc.get(pair) !== "c")
        cls = `src=${t.neededSrc.get(pair) ?? "none"}`;
      else if (t.everSurvived.has(pair)) cls = "survived";
      else if (e.mandatory) cls = "mandatory";
      else if (t.lastReader.get(pair) === undefined) cls = "noReader";
      else cls = "unstable";
      const lr = t.lastReader.get(pair);
      rows.push({
        pair,
        op: e.op,
        MB: +(e.bytes / 1e6).toFixed(1),
        cls,
        lastReader: lr !== undefined ? `0x${lr.toString(16)}` : undefined,
      });
    }
    rows.sort((a, b) => b.MB - a.MB);
    if (rows.length > 0) out[`0x${fp.toString(16)}`] = rows.slice(0, topN);
  }
  return out;
}

/**
 * Step-global releasability of one producer pair (the stage-3 seam): a
 * consumed-ONLY pair ("c" — never survived a step, never guard-grown, not
 * mandatory, not revoked) of a CONVERGED template, whose last cross-plan
 * reader has been the SAME template for >= K_HYSTERESIS fully-observed steps.
 * Such a pair's registry entry is dead after that reader's final read of it —
 * the reader's build may overlay its own temps onto the entry from that point
 * (the producer re-writes the entry before any next-step consumer reads).
 * Returns the stable last-reader fp, or undefined if not releasable.
 */
export function releasableLastReader(
  fp: number,
  ni: number,
  oi: number,
): number | undefined {
  if (!enabled) return undefined;
  const t = templates.get(fp);
  if (!t || !t.converged || t.pinned) return undefined;
  const k = key(ni, oi);
  if (t.releaseRevoked.has(k)) return undefined;
  if (t.neededSrc.get(k) !== "c") return undefined;
  if (t.everSurvived.has(k)) return undefined;
  if (t.everReadback.has(k)) return undefined;
  if (t.everAliased.has(k)) return undefined;
  const entry = t.resultEntries.get(k);
  // [stage-3 B] entry.mandatory is NOT an exclusion: the boundary-dead
  // mandatory-consumed class (grads read by the optimizer, forward casts
  // read by backward) IS the release target. The safety the flag used to
  // proxy is carried by observation: survival (everSurvived), readbacks
  // (everReadback — the seam that closed the epistemic hole), consumption
  // stability, and the loud clear-at-release guard.
  if (!entry) return undefined;
  if ((t.lastReaderStable.get(k) ?? 0) < K_HYSTERESIS) return undefined;
  return t.lastReader.get(k);
}

/** The registry entry registered for a pair (with the generation it is valid
 *  for) — the claim seam resolves stamps to entries through this. */
export function resultEntryFor(
  fp: number,
  ni: number,
  oi: number,
): { entryIdx: number; bytes: number; gen: number } | undefined {
  const t = templates.get(fp);
  const e = t?.resultEntries.get(key(ni, oi));
  return e
    ? { entryIdx: e.entryIdx, bytes: e.bytes, gen: t!.resultEntriesGen }
    : undefined;
}

/** Permanently revoke a pair's step-global releasability (guard miss proved an
 *  unobserved consumer). Idempotent. */
export function revokeRelease(fp: number, ni: number, oi: number): void {
  const t = templates.get(fp);
  if (t) t.releaseRevoked.add(key(ni, oi));
}

/** Drop a template's result-entry registrations (its compiled plan was
 *  invalidated or retired — the entries no longer back live results). The
 *  next build re-registers. Observation state (needed/lastReader) is kept. */
export function clearResultEntries(fp: number): void {
  const t = templates.get(fp);
  if (t) {
    t.resultEntries = new Map();
    t.resultEntriesGen = -1;
  }
}

/**
 * S3.0 instrumentation: per-producer-template summary of step-globally
 * RELEASABLE result bytes (what S3.1's overlay would free), grouped by the
 * stable last reader. Pure query — no behavior change.
 */
export function debugReleasableSummary(): Record<
  string,
  {
    releasablePairs: number;
    releasableMB: number;
    resultMB: number;
    byLastReader: Record<string, number>;
    /** Disqualifier attribution: why registered result bytes are NOT
     *  releasable — MB by first-failing check, in check order. */
    heldMB: Record<string, number>;
  }
> {
  const out: Record<
    string,
    {
      releasablePairs: number;
      releasableMB: number;
      resultMB: number;
      byLastReader: Record<string, number>;
      heldMB: Record<string, number>;
    }
  > = {};
  const mb = (b: number) => +(b / 1e6).toFixed(1);
  for (const [fp, t] of templates) {
    let resultBytes = 0;
    let relBytes = 0;
    let relPairs = 0;
    const byReader: Record<string, number> = {};
    const held: Record<string, number> = {};
    const hold = (why: string, bytes: number) => {
      held[why] = +((held[why] ?? 0) + bytes / 1e6).toFixed(1);
    };
    // Attribute each ENTRY once (multiple pairs can alias one entry — e.g. a
    // harvested view of an in-place state output; per-pair byte sums would
    // double-count). An entry is releasable only if EVERY pair on it is; its
    // held class is the strongest holder among its pairs.
    const strength = (c: string): number =>
      c === "rel"
        ? 0
        : c === "unstable"
          ? 1
          : c === "noReader"
            ? 2
            : c === "mandatory"
              ? 3
              : c === "survived"
                ? 4
                : 5; // src=*/revoked/unconverged
    const entryClass = new Map<number, { cls: string; bytes: number }>();
    for (const [pair, e] of t.resultEntries) {
      const bar = pair.indexOf(":");
      const ni = Number(pair.slice(0, bar));
      const oi = Number(pair.slice(bar + 1));
      const reader = releasableLastReader(fp, ni, oi);
      let cls: string;
      if (reader !== undefined) {
        cls = "rel";
        relPairs++;
        const rk = `0x${reader.toString(16)}`;
        byReader[rk] = +((byReader[rk] ?? 0) + e.bytes / 1e6).toFixed(1);
      } else if (!t.converged || t.pinned) cls = "unconverged";
      else if (t.releaseRevoked.has(pair)) cls = "revoked";
      else if (t.neededSrc.get(pair) !== "c")
        cls = `src=${t.neededSrc.get(pair) ?? "none"}`;
      else if (t.everSurvived.has(pair)) cls = "survived";
      else if (e.mandatory) cls = "mandatory";
      else if (t.lastReader.get(pair) === undefined) cls = "noReader";
      else cls = "unstable";
      const prev = entryClass.get(e.entryIdx);
      if (!prev || strength(cls) > strength(prev.cls)) {
        entryClass.set(e.entryIdx, { cls, bytes: e.bytes });
      }
    }
    for (const { cls, bytes } of entryClass.values()) {
      resultBytes += bytes;
      if (cls === "rel") relBytes += bytes;
      else hold(cls, bytes);
    }
    if (t.resultEntries.size > 0) {
      out[`0x${fp.toString(16)}`] = {
        releasablePairs: relPairs,
        releasableMB: mb(relBytes),
        resultMB: mb(resultBytes),
        byLastReader: byReader,
        heldMB: held,
      };
    }
  }
  return out;
}

/** Test/telemetry: per-template observed needed-set (sorted "ni:oi"). */
export function debugNeededSet(fp: number): string[] | undefined {
  const t = templates.get(fp);
  return t ? [...t.needed].sort() : undefined;
}

/** Test: every template's observed needed-set + convergence flags, keyed by
 *  fingerprint hex. The set-parity gate runs the workload under build-from-IR
 *  and the recorded cutover with observation forced on in BOTH; the observed
 *  needed-set is an intrinsic property of the workload (consumed ∪ survived,
 *  independent of the harvest mode) so it must AGREE across the two runs for
 *  every recurring template — the single-source seam-agreement assertion. */
export function debugAllNeededSets(): Record<
  string,
  {
    needed: string[];
    converged: boolean;
    pinned: boolean;
    /** Needed-pair counts by source: consumed / survived / base-only / guard. */
    srcCounts: Record<NeededSource, number>;
  }
> {
  const out: Record<
    string,
    {
      needed: string[];
      converged: boolean;
      pinned: boolean;
      srcCounts: Record<NeededSource, number>;
    }
  > = {};
  for (const [fp, t] of templates) {
    const srcCounts: Record<NeededSource, number> = {
      c: 0,
      s: 0,
      b: 0,
      g: 0,
      r: 0,
    };
    for (const src of t.neededSrc.values()) srcCounts[src]++;
    out[`0x${fp.toString(16)}`] = {
      needed: [...t.needed].sort(),
      converged: t.converged,
      pinned: t.pinned,
      srcCounts,
    };
  }
  return out;
}

export function getObservedLivenessStats(): {
  cleanMisses: number;
  dirtyMisses: number;
  pinnedTemplates: number;
  convergedTemplates: number;
  prunedPairsRemoved: number;
  retiredTemplates: number;
  /** [stage-3] Pairs currently classified step-globally releasable / their
   *  registry bytes (S3.0 instrumentation; S3.1 turns these into overlays). */
  releasablePairs: number;
  releasableMB: number;
  readbackMisses: number;
  convergenceInvalidations: number;
  claimMisses: number;
} {
  let converged = 0;
  for (const t of templates.values()) if (t.converged && !t.pinned) converged++;
  let releasablePairs = 0;
  let releasableBytes = 0;
  for (const [fp, t] of templates) {
    for (const [pair, e] of t.resultEntries) {
      const bar = pair.indexOf(":");
      if (
        releasableLastReader(
          fp,
          Number(pair.slice(0, bar)),
          Number(pair.slice(bar + 1)),
        ) !== undefined
      ) {
        releasablePairs++;
        releasableBytes += e.bytes;
      }
    }
  }
  return {
    cleanMisses,
    dirtyMisses,
    pinnedTemplates,
    convergedTemplates: converged,
    prunedPairsRemoved,
    retiredTemplates,
    releasablePairs,
    releasableMB: +(releasableBytes / 1e6).toFixed(1),
    readbackMisses,
    convergenceInvalidations,
    claimMisses,
  };
}
