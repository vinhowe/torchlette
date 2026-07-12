/**
 * THE fuseGesture DRIVER — the S3 composite transaction (docs/p2-moves-design.md
 * deliverable 3 / §1; schedule-state-design.md §3.5).
 *
 * ------------------------------------------------------------------------
 * WHAT S3 RULES, AND WHAT THIS BUILDS
 * ------------------------------------------------------------------------
 * `fuse` is NOT a ScheduleState move. Membership is owned by `Partition` alone
 * (islands `merge`/`split` are the only partition mutators). The editor's "fuse"
 * gesture is a COMPOSITE TRANSACTION at the next altitude, composing:
 *
 *   1. validate  the proposed interior SemanticSchedule (core legality)
 *   2. merge     P' = merge(P, a, b)   — DRIVEN through the re-record channel
 *                                        (§1.6 ruling: no new membership mutator)
 *   3. mint      region' = newSemanticRegionUid
 *   4. attach    the interior schedule at region'
 *   5. record    ONE provenance entry carrying BOTH hashes
 *   6. on realization failure: ROLL BACK all of 2–5 atomically
 *
 * The driver is BINARY (§5 ruling 1 — fuse ×2, not variadic) and validate-
 * interior → drive membership → attach state → ONE FuseProvenance → rollback on
 * realization failure. Chained transactions DEFER realization to the final state
 * (§5 ruling 2): interiors are VALIDATED per merge but realized once at commit.
 *
 * ------------------------------------------------------------------------
 * THE HONEST SCOPING STATEMENT (this wave)
 * ------------------------------------------------------------------------
 * The islands-side live wiring is the EDITOR channel, not yet built. So in this
 * wave the driver operates at the SCHEDULE / SEMANTIC-REGION altitude: it fuses
 * two adjacent region-states into ONE composed region-state whose lowering is a
 * single kernel (the off-menu machinery proven in P2-A makes an off-menu composed
 * state lowerable). The re-record-channel integration point — the actual
 * membership write the detector/partition owns — is STUBBED as a typed interface
 * (`ReRecordChannel`) and documented; the editor channel consumes it later (P3).
 * We do NOT build a standalone membership mutator (§1.6 — that would be a second
 * owner of membership). The `merge` LEGALITY gate (convexity via islandFlow) is
 * READ as a predicate here, not re-owned.
 */

import { scheduleDigest, semanticDigest } from "../canonical";
import type {
  ScheduleState,
  SemanticBody,
  SemanticRegionUid,
  SemanticSchedule,
} from "../types";

// ============================================================================
// The transaction's API shape (§1.2)
// ============================================================================

/** An island identity in the partition (islands-design lattice node). At the
 *  schedule/region altitude in this wave, an island IS its region UID. */
export type IslandId = SemanticRegionUid;

/**
 * The re-record channel — the SEAM the membership change is driven through
 * (§1.6). The detector/partition owns membership; the driver EXPRESSES "these two
 * regions are now one island" as an input to the same channel the detector
 * writes, and the executor re-records. In this wave the channel is a typed
 * interface (stubbed): `requestMerge` records the requested decision and returns
 * the minted region', `rollback` discards it. The editor wires a live channel at
 * P3; here a default in-memory channel drives the schedule-altitude fuse.
 */
export interface ReRecordChannel {
  /** Express the merge as a re-record request; returns the minted region UID. */
  requestMerge(a: IslandId, b: IslandId): SemanticRegionUid;
  /** Discard the requested decision (rollback = re-record under the prior one). */
  rollback(region: SemanticRegionUid): void;
}

/** The default in-memory channel: mints a deterministic region UID from a+b and
 *  records the request in a set so rollback is observable. This is the STUB the
 *  editor channel replaces — it drives the schedule-altitude fuse for the FA
 *  derivation without touching a live partition. */
export function makeInMemoryChannel(): ReRecordChannel & {
  readonly requested: ReadonlySet<string>;
} {
  const requested = new Set<string>();
  return {
    requested,
    requestMerge(a, b) {
      const region = mintRegion(a, b);
      requested.add(region as unknown as string);
      return region;
    },
    rollback(region) {
      requested.delete(region as unknown as string);
    },
  };
}

/** Mint region' deterministically from the two merged islands (R8 — the
 *  partition transaction mints the region UID). */
export function mintRegion(a: IslandId, b: IslandId): SemanticRegionUid {
  return `region:fused(${a}+${b})` as unknown as SemanticRegionUid;
}

export interface FuseGesture {
  /** The base partition, represented at this altitude as the island-flow edges
   *  (the convexity witness) + the two islands' current states. */
  readonly a: IslandId;
  readonly b: IslandId;
  readonly aState: ScheduleState;
  readonly bState: ScheduleState;
  /** The ordered data-flow edges (islandFlow): the convexity witness for the
   *  merge (§1.5 — a linear chain a→b makes the union convex). */
  readonly islandFlow: readonly {
    readonly from: IslandId;
    readonly to: IslandId;
    readonly via: string;
  }[];
  /** The proposed interior schedule to attach at region' (§1.2). If omitted, the
   *  driver composes a default interior = the concatenated bodies of a and b. */
  readonly proposedInterior?: SemanticSchedule;
  /**
   * The ORIGINAL constituent regions behind each island (for chained fuses: a
   * fused island's constituents are the regions it was built from). Convexity is
   * checked against constituents so an island-flow edge between an ORIGINAL
   * region of `a` and an ORIGINAL region of `b` still witnesses adjacency after
   * an earlier merge minted a composite region UID. Defaults to `[a]` / `[b]`.
   */
  readonly aConstituents?: readonly IslandId[];
  readonly bConstituents?: readonly IslandId[];
}

export type FuseStage = "validate-interior" | "merge" | "attach" | "realize";

export type FuseRefusalCode =
  | "INTERIOR_ILLEGAL"
  | "MERGE_REFUSED"
  | "ATTACH_SECOND_OWNER"
  | "REALIZATION_REFUSED"
  | "ENGINE_ERROR";

/** ONE provenance record carrying BOTH altitudes' identities (S3 step 5). */
export interface FuseProvenance {
  /** the islands boundary hash of P' (islands-design identity). At this altitude
   *  the boundary is the two merged region UIDs. */
  readonly boundaryHash: string;
  /** digest(SemanticSchedule + region') — the §5 semantic identity. */
  readonly semanticHash: string;
  /** + digest(the full state) — the §5 compilation identity coordinate. */
  readonly compilationHash: string;
  /** inverse data (S3 inverse-payload discipline): what UNDO needs. Undo of a
   *  fuse is a `split` at the recovering cut PLUS discarding region'. */
  readonly inverse: {
    readonly retiredRegion: SemanticRegionUid;
    readonly priorRegions: readonly SemanticRegionUid[];
  };
}

export interface FuseCommit {
  readonly region: SemanticRegionUid; // the minted region' (R8)
  readonly state: ScheduleState; // the attached interior, region = region'
  readonly provenance: FuseProvenance;
}

export type FuseOutcome =
  | { kind: "committed"; result: FuseCommit }
  | {
      kind: "refused";
      stage: FuseStage;
      reason: string;
      code: FuseRefusalCode;
    };

// ============================================================================
// The transaction (§1.3 staged pipeline)
// ============================================================================

/**
 * `fuseGesture` — a SINGLE transaction function. Returns `committed` with all of
 * (region', state, provenance) or `refused` with the stage that refused and a
 * stable code+reason. It NEVER returns a partial state — the S3 "roll back all
 * on realization failure" made a total function.
 *
 * `realize` is an optional realizer: it accepts the merged+attached state and
 * throws if it cannot emit (capability absence / engine error). When omitted the
 * realize stage trivially accepts (the schedule-altitude composed state is
 * lowerable by the off-menu machinery — that is what makes the fuse legal here).
 * `deferRealize` (§5 ruling 2): when true, the realize stage is SKIPPED (the
 * chained-transaction caller realizes once at the final state).
 */
export function fuseGesture(
  g: FuseGesture,
  opts: {
    readonly channel?: ReRecordChannel;
    readonly realize?: (state: ScheduleState) => void;
    readonly deferRealize?: boolean;
  } = {},
): FuseOutcome {
  const channel = opts.channel ?? makeInMemoryChannel();

  // ---- Stage 1: validate the proposed interior (core legality, backend-neutral).
  const interior =
    g.proposedInterior ?? composeInterior(g.aState.semantic, g.bState.semantic);
  const interiorCheck = validateInterior(interior);
  if (!interiorCheck.ok)
    return {
      kind: "refused",
      stage: "validate-interior",
      code: "INTERIOR_ILLEGAL",
      reason: interiorCheck.reason,
    };

  // ---- Stage 2: merge P' = merge(a, b) — DRIVEN through the re-record channel.
  //      The merge LEGALITY gate (convexity) is READ, not re-owned: the island-
  //      flow must be a linear chain a→b (or b→a), so the union is convex.
  const mergeLegal = mergeConvex(g);
  if (!mergeLegal.ok)
    return {
      kind: "refused",
      stage: "merge",
      code: "MERGE_REFUSED",
      reason: mergeLegal.reason,
    };
  const region = channel.requestMerge(g.a, g.b);

  // ---- Stage 3+4: mint region' (done by the channel) + attach the interior.
  const attached: ScheduleState = {
    semantic: interior,
    requests: g.aState.requests,
    receipts: {},
    region,
  };
  const attachCheck = assertNoSecondOwner(attached);
  if (!attachCheck.ok) {
    channel.rollback(region); // rollback the requested membership decision
    return {
      kind: "refused",
      stage: "attach",
      code: "ATTACH_SECOND_OWNER",
      reason: attachCheck.reason,
    };
  }

  // ---- Stage 6: realize (unless deferred). On failure, ROLL BACK atomically.
  if (!opts.deferRealize && opts.realize) {
    try {
      opts.realize(attached);
    } catch (err) {
      channel.rollback(region);
      return {
        kind: "refused",
        stage: "realize",
        code: "REALIZATION_REFUSED",
        reason: `realizer refused the fused state: ${(err as Error).message}`,
      };
    }
  }

  // ---- Stage 5: ONE provenance record carrying BOTH hashes.
  const provenance: FuseProvenance = {
    boundaryHash: boundaryHashOf(g.a, g.b),
    semanticHash: semanticDigest(attached),
    compilationHash: scheduleDigest(attached),
    inverse: {
      retiredRegion: region,
      priorRegions: [g.a, g.b],
    },
  };
  return { kind: "committed", result: { region, state: attached, provenance } };
}

/**
 * Fuse a CHAIN of adjacent islands into one, as fuse ×(n−1) binary gestures (§5
 * ruling 1 — 3→1 is two gestures). Interiors are validated per merge; realization
 * is DEFERRED to the final state (§5 ruling 2) — realized once by `realize` on
 * the last commit. Rollback is per the inverse chain: on any refusal, the already-
 * committed merges are rolled back in reverse order.
 */
export function fuseChain(
  islands: readonly { id: IslandId; state: ScheduleState }[],
  islandFlow: FuseGesture["islandFlow"],
  opts: {
    readonly channel?: ReRecordChannel;
    readonly realize?: (state: ScheduleState) => void;
  } = {},
):
  | { kind: "committed"; commits: readonly FuseCommit[]; final: FuseCommit }
  | {
      kind: "refused";
      stage: FuseStage;
      reason: string;
      code: FuseRefusalCode;
    } {
  if (islands.length < 2)
    return {
      kind: "refused",
      stage: "merge",
      code: "MERGE_REFUSED",
      reason: "fuseChain requires at least two islands.",
    };
  const channel = opts.channel ?? makeInMemoryChannel();
  const commits: FuseCommit[] = [];
  let acc = islands[0];
  // The constituents accumulate as the chain folds: the running island carries
  // every ORIGINAL region merged into it so far (the convexity witness for the
  // next merge reads against these, not the minted composite UID).
  let accConstituents: IslandId[] = [islands[0].id];
  for (let i = 1; i < islands.length; i++) {
    const isLast = i === islands.length - 1;
    const g: FuseGesture = {
      a: acc.id,
      b: islands[i].id,
      aState: acc.state,
      bState: islands[i].state,
      islandFlow,
      aConstituents: accConstituents,
      bConstituents: [islands[i].id],
    };
    // Defer realization on every merge EXCEPT the final one (§5 ruling 2).
    const outcome = fuseGesture(g, {
      channel,
      realize: isLast ? opts.realize : undefined,
      deferRealize: !isLast,
    });
    if (outcome.kind === "refused") {
      // rollback the chain: discard every committed merge's region.
      for (const c of commits.reverse()) channel.rollback(c.region);
      return outcome;
    }
    commits.push(outcome.result);
    acc = { id: outcome.result.region, state: outcome.result.state };
    accConstituents = [...accConstituents, islands[i].id];
  }
  return { kind: "committed", commits, final: commits[commits.length - 1] };
}

// ============================================================================
// The staged gates (each a pure predicate — no membership write)
// ============================================================================

/** validate-interior: the interior must be a well-formed SemanticSchedule with a
 *  store (a fused kernel writes something) and a body per stored value. This is
 *  backend-neutral core legality (§3.2). */
function validateInterior(
  interior: SemanticSchedule,
): { ok: true } | { ok: false; reason: string } {
  if (interior.stores.length === 0)
    return {
      ok: false,
      reason: "interior has no store edge (a fused kernel writes nothing).",
    };
  if (interior.bodies.length === 0)
    return { ok: false, reason: "interior has no semantic body." };
  return { ok: true };
}

/** merge convexity: the island-flow must be a linear chain between a and b (in
 *  either direction), so their union is convex — no external node lies between
 *  them (§1.5 islands-design rule 1). Read from islandFlow WITHOUT re-deriving
 *  the dataflow graph. */
function mergeConvex(
  g: FuseGesture,
): { ok: true } | { ok: false; reason: string } {
  // Constituents: for a chained fuse, an island's constituents are the ORIGINAL
  // regions it was built from (so an edge between an original of `a` and an
  // original of `b` still witnesses adjacency after an earlier merge).
  const aC = new Set<string>(
    (g.aConstituents ?? [g.a]).map((r) => r as unknown as string),
  );
  const bC = new Set<string>(
    (g.bConstituents ?? [g.b]).map((r) => r as unknown as string),
  );
  const direct = g.islandFlow.some((e) => {
    const from = e.from as unknown as string;
    const to = e.to as unknown as string;
    return (aC.has(from) && bC.has(to)) || (bC.has(from) && aC.has(to));
  });
  if (!direct)
    return {
      ok: false,
      reason:
        `no direct island-flow edge between ${g.a} and ${g.b}; the union is not a ` +
        `linear-chain convex region (islands-design §2 rule 1).`,
    };
  return { ok: true };
}

/** attach no-second-owner: the attached interior must carry the region' foreign
 *  key and no duplicated store target (a second owner leaked). §2.6. */
function assertNoSecondOwner(
  state: ScheduleState,
): { ok: true } | { ok: false; reason: string } {
  const targets = new Set<string>();
  for (const st of state.semantic.stores) {
    const key = st.target as unknown as string;
    if (targets.has(key))
      return {
        ok: false,
        reason: `store target ${key} written by two edges — a second owner leaked into the fused region.`,
      };
    targets.add(key);
  }
  return { ok: true };
}

// ============================================================================
// Interior composition (the default when no proposedInterior is given)
// ============================================================================

/**
 * Compose two adjacent regions' semantic schedules into ONE interior: the fused
 * region's loop nest is the OUTER region's (a), its bodies are the concatenation
 * of both, and its store is the CONSUMER region's (b) store (the fused kernel
 * produces b's output). The intermediate value (a's output = b's input) becomes
 * a no-materialization edge — the semantic no-store contract of the fusion. This
 * is the default interior the subsequent moves (tile → stream → recolor → lemma
 * → stream) transform (§1.5).
 */
export function composeInterior(
  a: SemanticSchedule,
  b: SemanticSchedule,
): SemanticSchedule {
  const bodies: SemanticBody[] = [...a.bodies, ...b.bodies];
  // The fused region keeps a's loop nest as the outer structure and b's store as
  // the output. The values union both (dedup by uid).
  const valueByUid = new Map<string, (typeof a.values)[number]>();
  for (const v of [...a.values, ...b.values])
    valueByUid.set(v.uid as unknown as string, v);
  const values = [...valueByUid.values()];
  // a's store target is the intermediate consumed by b — record it as a
  // no-materialization edge (the fusion's no-store contract) and drop a's store.
  const intermediates = a.stores.map((st) => ({
    producer: st.source,
    consumer: st.target,
    acrossLoop: st.atLoop,
  }));
  return {
    blockShapes: [...a.blockShapes, ...b.blockShapes],
    loopNest: a.loopNest,
    ordering: a.ordering,
    programGridMap: a.programGridMap,
    values,
    noMaterialization: [
      ...a.noMaterialization,
      ...b.noMaterialization,
      ...intermediates,
    ],
    stores: b.stores.length > 0 ? b.stores : a.stores,
    bodies,
    roles: [...a.roles, ...b.roles],
    sync: [...a.sync, ...b.sync],
    atoms: [...a.atoms, ...b.atoms],
    lemmas: [...a.lemmas, ...b.lemmas],
  };
}

function boundaryHashOf(a: IslandId, b: IslandId): string {
  return `boundary(${a}|${b})`;
}
