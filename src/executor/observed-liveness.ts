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
import type { StorageHandle } from "../graph/types";
import { storageTracker } from "../graph/storage-tracker";

/** Cross-plan identity of a harvested VALUE, independent of any live buffer. */
export interface ResultStamp {
  fp: number;
  ni: number;
  oi: number;
}

/** Thrown by the bind-time guard when a pruned producer's value is missing but
 *  recoverable (no in-place op committed since its replay). Caught by
 *  executePlanOptimized → evict the consumer template + re-collect lowered. */
export class RecoverableGuardMiss extends Error {
  constructor(readonly stamp: ResultStamp) {
    super(
      `[observed-liveness] pruned producer template=0x${stamp.fp.toString(16)} node=${stamp.ni} oi=${stamp.oi} missing at a late consumer; recovering via lowered re-collection`,
    );
    this.name = "RecoverableGuardMiss";
  }
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
 *  conservative view class), "g" guard-miss. First observation wins except a
 *  later "c"/"s" upgrades a "b" (a pair both consumed and base-alive is not a
 *  view-conservatism cost). */
type NeededSource = "c" | "s" | "b" | "g";

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
}

const key = (ni: number, oi: number): string => `${ni}:${oi}`;

// ── Module state (step-scoped structures cleared at observeStepBoundary) ──────
let enabled = false;
const templates = new Map<number, TemplateObs>();
let stepStamped: Array<{ shId: number; checkId: number; stamp: ResultStamp }> =
  [];
let newTemplateThisStep = 0;
/** nodeId → { producer stamp, in-place-commit count at its pruned replay }. */
let prunedExecuted = new Map<number, { stamp: ResultStamp; mark: number }>();
let inPlaceCommits = 0;

// ── Telemetry (folded onto getPayloadThrashStats) ────────────────────────────
let cleanMisses = 0;
let dirtyMisses = 0;
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
export function setTemplateIdleRetirer(
  fn: (fp: number) => RetireResult,
): void {
  retireIdleTemplate = fn;
}

export function setObservedLivenessEnabled(on: boolean): void {
  enabled = on;
}
export function isObservedLivenessEnabled(): boolean {
  return enabled;
}

export function resetObservedLiveness(): void {
  templates.clear();
  stepStamped = [];
  newTemplateThisStep = 0;
  prunedExecuted = new Map();
  inPlaceCommits = 0;
  cleanMisses = 0;
  dirtyMisses = 0;
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

/** Record cross-plan consumption of a resolved external input, if stamped. */
export function observeConsumed(storage: StorageHandle): void {
  if (!enabled) return;
  const stamp = storage.stamp;
  if (stamp) addNeeded(stamp.fp, stamp.ni, stamp.oi, "c");
}

/** Register the pairs a pruned plan EXCLUDED from its harvest, so a later
 *  guard-miss can name the culprit and decide clean/dirty. Called at the
 *  pruned plan's replay-harvest with the current step's node ids. */
export function registerPrunedExecution(
  fp: number,
  ni: number,
  oi: number,
  nodeId: number,
): void {
  if (!enabled) return;
  prunedExecuted.set(nodeId, { stamp: { fp, ni, oi }, mark: inPlaceCommits });
}

/**
 * Bind-time guard: a late consumer's external input (producer nodeId, oi) is
 * missing. If it was a pruned producer, decide recovery vs loud failure.
 * Returns true if the miss was ours (handled by throwing); false if unrelated
 * (the caller rethrows the original "Input not ready").
 */
export function guardMiss(nodeId: number, oi: number): boolean {
  if (!enabled) return false;
  const hit = prunedExecuted.get(nodeId);
  if (!hit || hit.stamp.oi !== oi) return false;
  const { stamp, mark } = hit;
  // Grow the producer's needed-set and invalidate its pruned build so next
  // step re-harvests the value (conservative-then-re-pruned).
  addNeeded(stamp.fp, stamp.ni, stamp.oi, "g");
  invalidateTemplateCompiled?.(stamp.fp);
  // Soundness boundary: recompute reads CURRENT storages. If any in-place op
  // committed since the producer's replay, the recompute would silently differ.
  if (inPlaceCommits > mark) {
    dirtyMisses++;
    throw new Error(
      `[observed-liveness] UNRECOVERABLE pruned-producer miss: template=0x${stamp.fp.toString(16)} node=${stamp.ni} oi=${stamp.oi} — ${inPlaceCommits - mark} in-place op(s) committed since its pruned replay, so recompute-on-demand would read mutated storages and produce a DIFFERENT value. Failing loudly rather than silently training on stale data. The value is added to the producer's needed-set; next step will harvest it. (stage-3 rematerialization would recover this in-step.)`,
    );
  }
  cleanMisses++;
  throw new RecoverableGuardMiss(stamp);
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
    }
  }
  const noNew = newTemplateThisStep === 0;
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
      if (t.grewThisStep && t.needed.size > REBUILD_GROWTH_LIMIT * t.convergedSize) {
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
}

// ── Pruning application (build-from-IR harvest) ───────────────────────────────

/**
 * If the template has converged (and is not pinned), return the pruned harvest
 * pairs = actionPairs ∩ (needed ∪ mandatory); the EXCLUDED pairs are returned so
 * the caller can record them for the bind-time guard. Returns undefined to use
 * the full conservative action-output set (not converged / pinned / disabled).
 */
export function prunedHarvest(
  fp: number,
  actionPairs: Array<{ i: number; oi: number }>,
  mandatory: Set<string>,
): {
  kept: Array<{ i: number; oi: number }>;
  excluded: Array<{ i: number; oi: number }>;
} | undefined {
  if (!enabled) return undefined;
  const t = templates.get(fp);
  if (!t || t.pinned || !t.converged) return undefined;
  const kept: Array<{ i: number; oi: number }> = [];
  const excluded: Array<{ i: number; oi: number }> = [];
  for (const p of actionPairs) {
    const k = key(p.i, p.oi);
    if (t.needed.has(k) || mandatory.has(k)) kept.push(p);
    else excluded.push(p);
  }
  prunedPairsRemoved += excluded.length;
  return { kept, excluded };
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
    const srcCounts: Record<NeededSource, number> = { c: 0, s: 0, b: 0, g: 0 };
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
} {
  let converged = 0;
  for (const t of templates.values()) if (t.converged && !t.pinned) converged++;
  return {
    cleanMisses,
    dirtyMisses,
    pinnedTemplates,
    convergedTemplates: converged,
    prunedPairsRemoved,
    retiredTemplates,
  };
}
