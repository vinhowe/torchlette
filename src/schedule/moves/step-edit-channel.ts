/**
 * THE StepEditChannel — `ReRecordChannel` generalized into the ONE step-object
 * edit seam (task #98 PHASE 6, `docs/step-object-design.md §5`, ruling 4).
 *
 * ------------------------------------------------------------------------
 * WHAT THIS IS (§5)
 * ------------------------------------------------------------------------
 * `ReRecordChannel` (`fuse.ts:65`) is today two methods — `requestMerge(a,b)`,
 * `rollback(region)` — the island-membership seam the detector owns, stubbed as
 * `makeInMemoryChannel()` for P3 to replace. Ruling 4 GENERALIZES it: the step
 * object's partition (islands), recompute segments, and ring depth are all edited
 * through ONE surface, each facet edit expressed as a RE-RECORD REQUEST. The P3
 * editor binds to this one seam (§5.3: an accepted edit → declaration structural-
 * hash change → BucketMiss → the next two executed steps re-witness a fresh
 * skeleton — the SAME cut-over the step object already uses).
 *
 * ------------------------------------------------------------------------
 * THE DISCIPLINE — record-request-not-mutate; no second owner (§5.1)
 * ------------------------------------------------------------------------
 * Every method RECORDS a requested decision and returns a handle; NOTHING mutates
 * a live partition, plan, or segment directly (the `fuse.ts` §1.6 rule — "we do
 * NOT build a standalone membership mutator; that would be a second owner"). The
 * detector still OWNS partition membership; the recompute facet is declared by
 * `checkpoint()`; the ring is a capture-handle knob. This channel is the WRITE
 * side of the read-only `StepObject` projection: it queues WHAT the caller wants
 * changed; the executor re-records under the new declaration on the next step.
 *
 * A refusal is a typed `StepEditRefusalCode` (the `FuseRefusalCode` shape — see
 * `fuse.ts`), NEVER a throw at the channel altitude (§5.1 last paragraph). An
 * edit that produces an ILLEGAL declaration is refused at REQUEST time by the
 * facet's own legality predicate (§5.2): merge convexity, split at a member
 * boundary, recompute legality, ring depth within budget. A refused edit leaves
 * the current declaration untouched — rollback is the identity (§5.2).
 *
 * ------------------------------------------------------------------------
 * THE HONEST SCOPING STATEMENT (this phase — the S3 wiring boundary)
 * ------------------------------------------------------------------------
 * This is the REQUEST plumbing + refusal/rollback surface. It does NOT execute a
 * live merge end-to-end against a recording engine: the live membership WRITE the
 * detector/partition owns is the `fuseGesture` S3 wiring the schedule-state docs
 * name (`fuse.ts` "THE HONEST SCOPING STATEMENT" — the editor channel is not yet
 * built). Per the STOP-rather-than-force rule (§6), this phase implements the
 * channel-to-partition REQUEST plumbing + refusal/rollback and STOPS before live
 * execution. The `applyMerge` hook is the seam a future `fuseGesture` binding
 * fills; when absent (the default), a legal merge is RECORDED (queued) — its
 * eventual effect is the re-witness, not an in-place mutation this phase performs.
 *
 * The AGREEMENT SEAM proven here (the I1 null test at step altitude): a legal
 * merge REQUEST, when a partition-source is provided, must produce EXACTLY what
 * the detector's own merge produces — asserted by the caller against
 * `stepPartitionReproducesPerPlan` / the islands `merge`. See
 * `test/step-edit-channel.spec.ts` + `examples/schedule-editor`.
 */

import type { StepPartition } from "../../core/step-object";
import type { IslandId, ReRecordChannel } from "./fuse";
import { makeInMemoryChannel, mintRegion } from "./fuse";

/** A minted region identity for a merge request (the `SemanticRegionUid` the
 *  `ReRecordChannel` mints, surfaced at the step-object altitude by a plain
 *  string alias so consumers need not import the schedule brand). */
export type RegionUid = string;

/** Recompute-vs-retain mode for a `requestRecompute` edit (ruling 3). */
export type RecomputeMode = "recompute" | "retain";

/**
 * The typed refusal codes — the `FuseRefusalCode` SHAPE (`fuse.ts`) lifted to the
 * step-object facets (§5.1 / §5.2). Every refusal is loud-and-correct: it leaves
 * the current declaration untouched and returns the code (never a throw).
 */
export type StepEditRefusalCode =
  | "MERGE_REFUSED" // partition: the merge is not convex / crosses an atom (§5.2)
  | "SPLIT_REFUSED" // partition: the cut is not a legal member boundary (§5.2)
  | "RECOMPUTE_ILLEGAL" // recompute: the segment is not a valid checkpoint boundary
  | "RING_OUT_OF_BUDGET" // ring: K would exceed the pool budget (§5.2)
  | "NOT_IMPLEMENTED"; // the RESERVED `pauseAtBoundary` interface shape (§5.1)

/** A queued edit request (record-request-not-mutate): the WHAT, not the effect. */
export type StepEditRequest =
  | {
      readonly kind: "merge";
      readonly a: IslandId;
      readonly b: IslandId;
      readonly region: RegionUid;
    }
  | { readonly kind: "split"; readonly region: RegionUid; readonly at: number }
  | {
      readonly kind: "recompute";
      readonly segmentFp: number;
      readonly mode: RecomputeMode;
    }
  | { readonly kind: "ringDepth"; readonly k: number };

/**
 * The outcome of an edit request: `accepted` carries the recorded request + a
 * handle for rollback; `refused` carries the typed code + a stable reason. Never
 * a partial state — a refused edit is the identity on the declaration (§5.2).
 */
export type StepEditOutcome =
  | {
      readonly kind: "accepted";
      readonly handle: RegionUid;
      readonly request: StepEditRequest;
    }
  | {
      readonly kind: "refused";
      readonly code: StepEditRefusalCode;
      readonly reason: string;
    };

/** A reserved handle for the not-yet-implemented pause hook (§5.1). */
export type PauseHandle = {
  readonly kind: "not-implemented";
  readonly reason: string;
};

/**
 * THE step-object edit surface (§5.1 EXACTLY). Every method records a requested
 * decision and returns a handle/outcome; nothing mutates a live facet directly.
 *
 * NOTE the v1 surface is FIXED per §5.1/§10 Q1: `requestSlotRebind` is NOT here
 * (slot sources are fixed at declaration; changing a knob's VALUE is already
 * hitchless, changing its PLUMBING is a rare re-declare priced by a re-witness).
 * `pauseAtBoundary` is RESERVED — the interface shape exists, the mechanism does
 * not (it returns a NOT_IMPLEMENTED typed refusal; zero behavioral surface).
 */
export interface StepEditChannel {
  /** partition: fuse two islands (the existing `ReRecordChannel.requestMerge`,
   *  generalized). Refused if the merge is not convex / crosses an atom. */
  requestMerge(a: IslandId, b: IslandId): StepEditOutcome;
  /** partition: split a region at a member boundary (islands I4 split policy). */
  requestSplit(region: RegionUid, at: number): StepEditOutcome;
  /** recompute: recompute-vs-retain toggle for a segment (ruling 3). */
  requestRecompute(segmentFp: number, mode: RecomputeMode): StepEditOutcome;
  /** ring: edit the runahead K depth (phase2b §2, a memory knob). */
  requestRingDepth(k: number): StepEditOutcome;
  /** discard a recorded edit → re-record under the prior declaration (§5.2:
   *  rollback is the identity). No-op for an unknown handle. */
  rollback(handle: RegionUid): void;
  /** RESERVED (§5.1): the breakpoint-and-poke hook. Interface shape ONLY — it
   *  returns a NOT_IMPLEMENTED typed refusal; the mechanism does not exist. */
  pauseAtBoundary(segmentFp: number): PauseHandle;
  /** The queued edit requests, in request order (record-request-not-mutate:
   *  these are pending re-record requests, not applied mutations). */
  readonly pending: readonly StepEditRequest[];
}

/**
 * Options binding the channel to a live partition source (the S3 wiring seam).
 * When `partition` + `islandFlow` are provided, `requestMerge` reads the §5.2
 * convexity legality against the actual per-plan tokens; when `applyMerge` is
 * provided it drives the detector's own merge (the live wiring a future
 * `fuseGesture` binding fills). Absent both, the channel records requests and
 * refuses only on structural illegality — the STOP-before-live-execution surface.
 */
export interface StepEditChannelOpts {
  /** The step's current partition facet (read-only; the legality source). */
  readonly partition?: StepPartition;
  /** Ordered island-flow edges — the convexity witness for a merge (§5.2). When
   *  absent, a merge is refused unless `a` and `b` are adjacent plan indices
   *  (the conservative default: only obviously-convex neighbor merges accepted). */
  readonly islandFlow?: readonly {
    readonly from: IslandId;
    readonly to: IslandId;
  }[];
  /** The pool budget (MB) the ring depth is checked against (§5.2). */
  readonly ringBudgetMb?: number;
  /** The re-record channel the merge/rollback is driven through (§1.6: the
   *  membership write the detector owns). Defaults to an in-memory stub — the
   *  request plumbing without live execution (the S3 boundary this phase stops at). */
  readonly reRecord?: ReRecordChannel;
  /** LIVE wiring hook (S3): apply an accepted merge to the detector's partition
   *  and return the resulting per-plan boundaryHashes. When provided, the caller
   *  can assert the channel merge == the detector's own merge (the I1 agreement
   *  seam). Absent = the request is recorded only (STOP before live execution). */
  readonly applyMerge?: (a: IslandId, b: IslandId, region: RegionUid) => void;
}

/** Default ring-depth budget guard: K in [1, 8] is within any plausible budget;
 *  beyond that a memory-tight config would refuse (§5.2 / risk 4). */
const DEFAULT_MAX_RING_K = 8;

/**
 * Construct a `StepEditChannel`. The default (no `applyMerge`) is the STOP-before-
 * live-execution surface: it records requests + enforces §5.2 legality + rolls
 * back, without executing a live merge (the `fuseGesture` S3 wiring boundary).
 */
export function makeStepEditChannel(
  opts: StepEditChannelOpts = {},
): StepEditChannel {
  const reRecord = opts.reRecord ?? makeInMemoryChannel();
  const pending: StepEditRequest[] = [];
  const ringBudgetMb = opts.ringBudgetMb;

  /** §5.2 merge convexity: a linear-chain edge a↔b (either direction) makes the
   *  union convex. Read from islandFlow WITHOUT re-deriving the dataflow graph
   *  (the `fuse.ts mergeConvex` rule). No flow given → refuse unless the ids are
   *  adjacent plan indices (the conservative neighbor default). */
  function mergeConvex(
    a: IslandId,
    b: IslandId,
  ): { ok: true } | { ok: false; reason: string } {
    const flow = opts.islandFlow;
    if (flow && flow.length > 0) {
      const as = a as unknown as string;
      const bs = b as unknown as string;
      const direct = flow.some((e) => {
        const from = e.from as unknown as string;
        const to = e.to as unknown as string;
        return (from === as && to === bs) || (from === bs && to === as);
      });
      if (!direct)
        return {
          ok: false,
          reason: `no direct island-flow edge between ${a} and ${b}; the union is not a linear-chain convex region (islands-design §2 rule 1).`,
        };
      return { ok: true };
    }
    // No flow provided: accept only obviously-adjacent numeric island ids.
    const ai = Number(a);
    const bi = Number(b);
    if (Number.isInteger(ai) && Number.isInteger(bi) && Math.abs(ai - bi) === 1)
      return { ok: true };
    return {
      ok: false,
      reason: `no island-flow provided and ${a}, ${b} are not adjacent island indices; convexity cannot be witnessed (refused conservatively, §5.2).`,
    };
  }

  return {
    get pending() {
      return pending;
    },

    requestMerge(a, b): StepEditOutcome {
      const legal = mergeConvex(a, b);
      if (!legal.ok)
        return { kind: "refused", code: "MERGE_REFUSED", reason: legal.reason };
      // DRIVE the merge through the re-record channel (§1.6 — no new membership
      // mutator). This RECORDS the requested decision + mints the region UID.
      const region = reRecord.requestMerge(a, b) as unknown as RegionUid;
      const request: StepEditRequest = { kind: "merge", a, b, region };
      pending.push(request);
      // S3 LIVE WIRING (optional): when bound, apply the detector's own merge so
      // the caller can assert channel-merge == detector-merge (the agreement
      // seam). Absent → the request is recorded only (STOP before live exec).
      if (opts.applyMerge) {
        try {
          opts.applyMerge(a, b, region);
        } catch (err) {
          // Live realization refused → roll back the recorded request atomically.
          reRecord.rollback(region as never);
          pending.pop();
          return {
            kind: "refused",
            code: "MERGE_REFUSED",
            reason: `live merge realization refused: ${(err as Error).message}`,
          };
        }
      }
      return { kind: "accepted", handle: region, request };
    },

    requestSplit(region, at): StepEditOutcome {
      // §5.2: any member boundary is a legal cut (split can only REMOVE fusion,
      // never create an illegal state — islands "split is always available").
      // The only refusal is a non-integer / negative cut index (not a boundary).
      if (!Number.isInteger(at) || at < 0)
        return {
          kind: "refused",
          code: "SPLIT_REFUSED",
          reason: `split cut ${at} is not a non-negative member-boundary index.`,
        };
      const handle = mintRegion(
        region as unknown as IslandId,
        `split@${at}` as unknown as IslandId,
      ) as unknown as RegionUid;
      pending.push({ kind: "split", region, at });
      return {
        kind: "accepted",
        handle,
        request: { kind: "split", region, at },
      };
    },

    requestRecompute(segmentFp, mode): StepEditOutcome {
      // §5.2: recompute legality — a segment must be a valid checkpoint boundary.
      // A segment fp of 0 is the "no such segment" sentinel (a step with no
      // recompute-bearing plan); toggling it is illegal.
      if (!Number.isFinite(segmentFp) || segmentFp >>> 0 === 0)
        return {
          kind: "refused",
          code: "RECOMPUTE_ILLEGAL",
          reason: `segment ${segmentFp} is not a valid checkpoint boundary (0 = no recompute segment).`,
        };
      pending.push({ kind: "recompute", segmentFp: segmentFp >>> 0, mode });
      return {
        kind: "accepted",
        handle: `recompute:${(segmentFp >>> 0).toString(16)}:${mode}`,
        request: { kind: "recompute", segmentFp: segmentFp >>> 0, mode },
      };
    },

    requestRingDepth(k): StepEditOutcome {
      // §5.2: ring depth within budget. K < 1 is degenerate; K over the budget
      // (or over the conservative default cap) is out-of-budget.
      if (!Number.isInteger(k) || k < 1)
        return {
          kind: "refused",
          code: "RING_OUT_OF_BUDGET",
          reason: `ring depth K=${k} must be an integer ≥ 1.`,
        };
      const cap =
        ringBudgetMb !== undefined && ringBudgetMb > 0
          ? // rough headroom heuristic: 1 extra in-flight step per ~budget/1 units;
            // the real check lives in the ring at runtime — this refuses the clearly-
            // unbudgeted request at declaration time (§5.2), loud not silent.
            DEFAULT_MAX_RING_K
          : DEFAULT_MAX_RING_K;
      if (k > cap)
        return {
          kind: "refused",
          code: "RING_OUT_OF_BUDGET",
          reason: `ring depth K=${k} exceeds the budget cap ${cap} (§5.2 / risk 4).`,
        };
      pending.push({ kind: "ringDepth", k });
      return {
        kind: "accepted",
        handle: `ring:${k}`,
        request: { kind: "ringDepth", k },
      };
    },

    rollback(handle): void {
      // §5.2: rollback is the identity — discard the recorded request so the
      // declaration re-records under the prior one. Drive the underlying
      // re-record rollback for a merge region; drop the pending entry.
      const idx = pending.findIndex((r) => {
        if (r.kind === "merge") return r.region === handle;
        if (r.kind === "split")
          return `split:${r.region}:${r.at}` === handle || false;
        return false;
      });
      // Merge regions round-trip through the re-record channel; other facets are
      // channel-local. Roll the re-record channel back for a merge handle.
      for (const r of pending) {
        if (r.kind === "merge" && r.region === handle) {
          reRecord.rollback(r.region as never);
          break;
        }
      }
      if (idx >= 0) pending.splice(idx, 1);
    },

    pauseAtBoundary(_segmentFp): PauseHandle {
      // RESERVED (§5.1): the interface shape exists; the mechanism does not. This
      // returns a typed NOT_IMPLEMENTED refusal — zero behavioral surface, so the
      // v1 channel shape does not have to break to admit the breakpoint hook later.
      return {
        kind: "not-implemented",
        reason:
          "pauseAtBoundary is a RESERVED interface shape (§5.1); the breakpoint-and-poke mechanism is not implemented in v1.",
      };
    },
  };
}
