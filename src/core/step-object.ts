/**
 * The Step Object (task #98 PHASE 1 — reify, null-clean).
 *
 * Normative source: docs/step-object-design.md §2 (the object: two phases,
 * identity/digest §2.2, guards as typed refusals §2.3, invalidation
 * subordination §2.4, the NOT-fields list §2.5) and §6 Phase 1.
 *
 * WHAT THIS IS. A whole training (or decode) step is ONE first-class object
 * that exists in TWO phases — a *declared* source form (the boundary contract +
 * dynamic-slot / recompute-segment / partition facets + ring config) and a
 * *witnessed* compiled form (the step-tape skeleton). This module REIFIES that
 * object as a DERIVED union over the existing mechanisms; it does NOT own any
 * state, exactly as islands I0 reified `Partition` (schedule-state P0 discipline:
 * derive/apply side-by-side, byte-identical, zero schema deltas).
 *
 * SINGLE SOURCE — WHY NO SECOND OWNER (ruling 1: "no second whole-step mechanism
 * may be born"). The operational mechanisms remain:
 *   - the RECORDER (`src/core/step-tape.ts`): owns the `StepTape` store, the
 *     eligibility diff, the refusal counters. It is the DECLARED-phase authority
 *     (slots, structural signature, epoch, regime) AND the digest source (its
 *     `bucketKey` already IS `hash(structKey) + ordered plan fps`, §2.2).
 *   - the REPLAY layer (`src/executor/step-tape-replay.ts`): owns the `Skeleton`
 *     store. It is the WITNESSED-phase authority (the ordered compiled plans).
 * The StepObject is a READ-ONLY VIEW that projects a facet-keyed identity over
 * those two. `deriveStepObject()` constructs one on demand from the tape state
 * and the live counters; nothing here is authored, mutated, or persisted beyond
 * the caller's local handle. This is the "loop-nest VIEW derived by a canonical
 * ordering rule" pattern (`schedule-state-design.md §0`), one stratum up.
 *
 * PHASE 1 IS NULL-CLEAN. Reifying + consulting the StepObject changes NOTHING
 * behavioral: the tape's `bucketKey`, the capture appKey, and the islands
 * boundaryHash all recompute byte-identically as projections of the one object;
 * command streams are byte-identical on decode + training tapes; digests
 * recompute byte-identical across two runs of the same config (the NULL TEST,
 * §6 Phase 1 gate). Slot NAMES are STABLE across re-witnesses (§10 ruling 1:
 * a slot survives re-witnessing as "the α slot", not "slot #7 of tape 43") —
 * the recorder already keys slots by `<op>:<fpHex>:<pos>[:inputIndex]`, a
 * declaration-stable name; the StepObject exposes it verbatim.
 *
 * SUNSET: this module rides `TORCHLETTE_STEP_TAPE` with the rest of the
 * campaign; it adds NO env flag (§6 Phase 1 gate: "no new env flags").
 */

import type { DynamicSlotSource, StepTape } from "./step-tape";

// ---------------------------------------------------------------------------
// §2.3 — Guard semantics as TYPED REFUSALS.
// ---------------------------------------------------------------------------

/**
 * The six tape guards (`step-tape.ts`, enumerated in the recorder + replay
 * headers) recast as a typed enumeration of REFUSAL REASONS. This is the
 * inspectable-data recast the design mandates (§2.3): the refusal REASONS
 * become data on the object; the underlying bookkeeping is UNCHANGED (no
 * behavior change — the recorder/replay still count and fall back exactly as
 * before). Each variant maps onto exactly one existing guard class:
 *
 *   | # | today (tape guard)                         | typed refusal        |
 *   |---|--------------------------------------------|----------------------|
 *   | 1 | structGen — op-sequence counter delta       | StructureMiss        |
 *   | 2 | bucketKey — plan fps + write/readback ids    | BucketMiss           |
 *   | 3 | scalar/payload coverage — diffImages byte    | UndeclaredVariance   |
 *   | 4 | plan validity — stInvalidateTemplate         | PlanInvalid          |
 *   | 5 | epoch/regime — engine epoch + stepScoped     | EpochMiss            |
 *   | 6 | STRICT_TAPE paranoia — any miss/verify throw | StrictNet            |
 *
 * Every refusal is LOUD-and-correct: it falls back to the normal build+execute
 * path (which re-witnesses) and, under STRICT, throws. There is NO
 * benign-divergence category (the clip-divergence lesson, CLAUDE.md).
 */
export type StepRefusalReason =
  | "StructureMiss" // guard 1: the declaration's op-structure changed
  | "BucketMiss" // guard 2: the declaration's identity bucket changed
  | "UndeclaredVariance" // guard 3: a per-step-varying byte no declared slot covers
  | "PlanInvalid" // guard 4: a referenced compiled plan invalidated
  | "EpochMiss" // guard 5: the boundary regime/epoch changed
  | "StrictNet"; // guard 6: STRICT_TAPE turned a miss/verify-diff into a throw

/** The guard number (§2.3 table) a refusal reason maps onto — the audit link
 *  from the typed reason back to the concrete tape guard it re-expresses. */
export const REFUSAL_GUARD: Readonly<Record<StepRefusalReason, number>> = {
  StructureMiss: 1,
  BucketMiss: 2,
  UndeclaredVariance: 3,
  PlanInvalid: 4,
  EpochMiss: 5,
  StrictNet: 6,
};

// ---------------------------------------------------------------------------
// §2.1 — The two-phase schema (reified, NOT authored).
// ---------------------------------------------------------------------------

/**
 * A dynamic-slot declaration WITH A STABLE NAME (§2.1 slots, §10 ruling 1). The
 * `id` is the declaration-stable name the recorder already assigns — it is a
 * pure function of (op fingerprint, node position[, input index]), so the SAME
 * declaration yields the SAME name across re-witnesses (no tape ordinal in it).
 */
export interface StepSlotDecl {
  /** Declaration-stable name: "w:<fpHex>:<pos>" (upload/TAG_WRITE),
   *  "u:<fpHex>:<pos>" (TAG_UNIFORM payload), "sc:<fpHex>:<pos>:<ii>" (scalar). */
  readonly id: string;
  /** Human-readable: op + shape (the recorder's slot.name). */
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: string;
  readonly source: DynamicSlotSource;
}

/**
 * The `partition` facet (task #98 PHASE 6, ruling 4 — `docs/step-object-design.md
 * §3.3`). `(G, P, device)` per STEP: the step's ONE partition of its whole
 * semantic graph, of which the per-plan islands boundaryHashes are a PROJECTION
 * (Objection B's answer — "the per-plan boundaryHashes are a DERIVED PROJECTION
 * of it"). SINGLE SOURCE: the detector still OWNS membership (`fusion-detect.ts`
 * `reifyPartition`); this facet is a read-only VIEW keyed to it — the same
 * "loop-nest VIEW derived by a canonical ordering rule" pattern one stratum up.
 *
 * The `plans` entries are ALIGNED with the witnessed skeleton's ordered fps (a
 * plan may repeat — its partition token repeats too, never deduped). Phase 6's
 * I1 null test (`§3.3` agreement seam) is: this projection must reproduce the
 * per-plan boundaryHashes byte-identically — asserted by
 * `stepPartitionReproducesPerPlan`. `boundaryDigest` is the step-level partition
 * identity (FNV-1a over the ordered `(fp, boundaryHash)` pairs); it is what the
 * `StepEditChannel` (`step-edit-channel.ts`) mutates a REQUEST against.
 */
export interface StepPartition {
  /** Per-plan islands partition tokens, ALIGNED with the ordered plan fps. */
  readonly plans: readonly {
    /** The plan's execution fingerprint (already carries boundaryHash, I1). */
    readonly fp: number;
    /** The plan's islands partition-identity token (`PlanPartition.boundaryHash`,
     *  I1). 0 for a plan with no reified partition (lowered/arena-free). */
    readonly boundaryHash: number;
  }[];
  /** Step-level partition identity: FNV-1a over the ordered (fp, boundaryHash)
   *  pairs. Null-stable (a static graph's partition hashes identically every
   *  step) and discriminating (any per-plan boundary change changes it). */
  readonly boundaryDigest: number;
  /** Device class the partition's legality is keyed to (islands R2 — a merge
   *  legal on A100 may be illegal on V100; §2.4 device key). Reified from the
   *  engine's backend kind; a device change is a BucketMiss. */
  readonly device: string;
}

/**
 * The DECLARED phase (§2.1). Every field is a PROJECTION of tape state — none
 * is new machinery (§2.5 NOT-fields discipline). The recompute + partition
 * facets are REFERENCES reified from what exists TODAY (checkpoint segments
 * from plan-builder barriers; partition from the per-plan boundaryHash) — NOT
 * new state (phases 3/6 build those mechanisms).
 */
export interface StepDeclaration {
  /** The boundary contract: the structural signature that hashes into STEP
   *  identity (the tape's structKey — plan fps + write positions + readback
   *  params; per-step scalar bytes are NOT in it, §2.2). Reified from the tape's
   *  bucketKey (see `deriveStepObject`). */
  readonly boundaryStructHash: string;
  /** Dynamic-slot declarations WITH STABLE NAMES (§2.1, §10 ruling 1). */
  readonly slots: readonly StepSlotDecl[];
  /** Recompute-segment facet REFERENCE (ruling 3; NOT a mechanism here). The
   *  ordered template fingerprints the step's plan sequence touches — the
   *  reference from which phase-3's `RecomputeSegment[]` will be derived
   *  (checkpoint segments are plan-builder barriers between these plans). Phase
   *  1 reifies the reference, not the recompute knob. */
  readonly recomputeRef: readonly number[];
  /** Partition facet REFERENCE (ruling 4). The per-plan boundaryHash projection —
   *  reified from the ordered plan fps (each plan's I1 boundaryHash mixes into
   *  its fp already, `executor.ts:259`). Kept as the ordered-fps reference the
   *  phase-1 null test pins; `partition` (below) is the phase-6 typed lift. */
  readonly partitionRef: readonly number[];
  /** The `partition` FACET (task #98 phase 6, ruling 4): `StepPartition` — the
   *  per-step partition of which `partitionRef` is the fp projection. A read-only
   *  view keyed to the detector's membership (single source, no second owner).
   *  The `StepEditChannel` records edit REQUESTS against its `boundaryDigest`. */
  readonly partition: StepPartition;
  /** Ring config REFERENCE (§2.1). Phase 1 has no ring knob to reify from a
   *  recorded tape (the ring lives on the capture handle, not the tape); the
   *  witnessed phase carries no K. Reified as `null` — a reference placeholder,
   *  honoring the §2.5 "no fields the doc lacks" discipline (the field exists in
   *  the §2.1 schema; its value is not derivable from the tape alone). */
  readonly ringRef: null;
}

/**
 * The WITNESSED phase (§2.1): the tape skeleton as the DERIVED compiled form.
 * Reified as the ordered plan fp sequence (the recorder's tape identity
 * coordinate). This module does NOT import the `Skeleton` type (it lives in the
 * executor layer) — the witnessed phase's IDENTITY is the ordered fps, and that
 * is what phase 1 reifies. The skeleton OBJECT stays owned by the replay layer
 * (single source).
 */
export interface StepSkeletonRef {
  /** ORDERED plan execution fp sequence (may repeat; NEVER deduped — a template
   *  can execute more than once per step). Length 1 for decode. */
  readonly orderedFps: readonly number[];
  /** DEDUPED template fps (guard-4 invalidation index). */
  readonly templateIds: readonly number[];
}

/**
 * Receipts (§2.1 / §2.5): guard-miss counters + eligibility bookkeeping that
 * hash into NEITHER identity. Reified from the recorder's live counters. NOT
 * part of the object's digest (§2.2: receipts are canonicalized OUT).
 */
export interface StepReceipts {
  readonly refusals: number;
  readonly eligiblePairs: number;
  readonly structureMisses: number;
  readonly planInvalidations: number;
  readonly boundaryResets: number;
}

/**
 * THE STEP OBJECT (§2.1). A two-phase typed datum with a canonical digest,
 * DERIVED from tape + replay state — never a second owner (see header).
 */
export interface StepObject {
  /** DECLARED phase (the source form; hashes into STEP identity via §2.2). */
  readonly declaration: StepDeclaration;
  /** WITNESSED phase (the compiled form; DERIVED, not authored). null when the
   *  step has not yet witnessed a skeleton (no eligible tape exists). */
  readonly skeleton: StepSkeletonRef | null;
  /** Engine epoch at the recording step's boundary (scoped-memory §1 vocabulary;
   *  guard 5). §2.1 names this `epoch`, NOT "step". */
  readonly epoch: number;
  /** Boundary regime (guard 5): part of identity via the structural signature,
   *  surfaced here for the EpochMiss refusal. */
  readonly regime: { readonly stepScopedCleanup: boolean };
  /** Receipts — hash into neither identity (§2.5). */
  readonly receipts: StepReceipts;
}

// ---------------------------------------------------------------------------
// §2.2 — Identity / digest (recomputable, byte-stable across identical runs).
// ---------------------------------------------------------------------------

/**
 * The step object's canonical digest (§2.2): the pair
 * `(declaration-structural-hash, ordered plan fp sequence)` — EXACTLY the tape's
 * `bucketKey = hash(structKey) + ordered plan fps`. Slot VALUES, per-step
 * scalars (batch data, bias-corrected step_size, scale), and receipts are
 * canonicalized OUT of the digest (they are DATA delivered through declared
 * slots, never identity — the frozen-scalar defense expressed as an identity
 * rule). Byte-stable across identical runs by construction: it is a pure
 * function of the structural signature + the ordered fps, both of which the
 * recorder already proves byte-stable (`phase2b.md INC-2B`: tapeCount=1 over 18
 * steps, both fused and foreach arms).
 *
 * Returns the tape's canonical bucketKey form (`b:<structHashHex>:<fp+fp+…>`),
 * so `digest === tape.bucketKey` is a pure-projection assert (see
 * `stepObjectDigestMatchesBucket`). Byte-identical to the recorder's own
 * `bucketKey` construction (`step-tape.ts stEndStep`) by construction — the
 * StepObject reads the same struct-hash and the same ordered fps.
 */
export function stepObjectDigest(obj: StepObject): string {
  const structHash = obj.declaration.boundaryStructHash;
  const fps = (obj.skeleton?.orderedFps ?? [])
    .map((fp) => (fp >>> 0).toString(16))
    .join("+");
  return `b:${structHash}:${fps}`;
}

/**
 * Pure-projection assertion (§3.1 agreement seam): the StepObject's digest MUST
 * equal the tape's own bucketKey byte-for-byte (the single-source invariant —
 * the object DERIVES its identity from the tape, it does not recompute an
 * independent one). Returns true iff they agree; a false is a reification bug
 * (the object and its source diverged), never a benign difference.
 *
 * NOTE the recorder's bucketKey uses the DEDUPED plan fps in its fp list
 * (`rec.plans.map(pl => hex(pl.fp))` — the plan sequence, which for the tape's
 * own construction is per-plan not deduped-by-template). The StepObject reifies
 * the witnessed phase's ORDERED fps (same list, same order) so the projection is
 * exact. See `deriveStepObject`.
 */
export function stepObjectDigestMatchesBucket(
  obj: StepObject,
  tape: StepTape,
): boolean {
  return stepObjectDigest(obj) === tape.bucketKey;
}

// ---------------------------------------------------------------------------
// §3.3 — The partition facet: step-level identity + the I1 agreement seam.
// ---------------------------------------------------------------------------

/**
 * FNV-1a over the ordered `(fp, boundaryHash)` pairs — the step-level partition
 * identity token (task #98 phase 6). Null-stable + partition-discriminating, the
 * same shape islands' `partitionBoundaryHash` uses one stratum down. Pure;
 * order-sensitive (a repeated plan's token repeats, never deduped).
 */
export function stepPartitionDigest(
  plans: readonly { readonly fp: number; readonly boundaryHash: number }[],
): number {
  let h = 0x811c9dc5;
  const prime = 0x01000193;
  const mix = (v: number) => {
    h ^= v & 0xff;
    h = Math.imul(h, prime);
    h ^= (v >>> 8) & 0xff;
    h = Math.imul(h, prime);
    h ^= (v >>> 16) & 0xff;
    h = Math.imul(h, prime);
    h ^= (v >>> 24) & 0xff;
    h = Math.imul(h, prime);
  };
  mix(plans.length);
  for (const p of plans) {
    mix(p.fp >>> 0);
    mix(p.boundaryHash >>> 0);
  }
  return h >>> 0;
}

/**
 * The I1 AGREEMENT SEAM at step altitude (§3.3 / Objection B). The step
 * partition is the DECLARED facet; the per-plan boundaryHashes are its DERIVED
 * projection. This assert is the phase-6 null test: the projection must
 * reproduce the per-plan boundaryHashes byte-identically against the source
 * (the executor's already-computed per-plan tokens). Returns true iff the
 * partition's per-plan `boundaryHash` list equals `perPlanBoundaryHashes`
 * position-for-position — a false is a projection bug (the step facet and the
 * detector's per-plan membership diverged), NEVER a benign difference.
 */
export function stepPartitionReproducesPerPlan(
  partition: StepPartition,
  perPlanBoundaryHashes: readonly number[],
): boolean {
  if (partition.plans.length !== perPlanBoundaryHashes.length) return false;
  for (let i = 0; i < partition.plans.length; i++) {
    if (
      partition.plans[i].boundaryHash >>> 0 !==
      perPlanBoundaryHashes[i] >>> 0
    )
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Derivation (the single-source read-over).
// ---------------------------------------------------------------------------

/**
 * DERIVE a StepObject from a witnessed `StepTape` (the recorder's stored tape)
 * plus the live receipt counters. This is the ONLY constructor — there is no
 * authored path (ruling 1: no second owner). Every field is read from the tape;
 * nothing is recomputed independently of it (the digest, when recomputed, MUST
 * equal `tape.bucketKey` — the pure-projection invariant).
 *
 * The tape's `bucketKey` is `b:<structHashHex>:<fp+fp+…>`. We split it back into
 * the (structHash, orderedFps) pair the StepObject's declaration + witnessed
 * phase carry — the exact inverse of `stEndStep`'s construction, so the
 * round-trip is byte-identical.
 */
/**
 * Project the `StepPartition` from the ordered plan fps + the per-plan islands
 * tokens the recorder captured (`tape.partitionHashes`, from the executor's
 * `PlanPartition.boundaryHash`). The two lists are aligned by plan order; if a
 * token list is missing/short (older tapes, or a plan with no reified partition),
 * the missing token is 0 — a sound null that the phase-6 null test still
 * reproduces (0 == 0). SINGLE SOURCE: no membership is computed here.
 */
function derivePartition(
  orderedFps: readonly number[],
  partitionHashes: readonly number[],
  device: string,
): StepPartition {
  const plans = orderedFps.map((fp, i) => ({
    fp: fp >>> 0,
    boundaryHash: (partitionHashes[i] ?? 0) >>> 0,
  }));
  return { plans, boundaryDigest: stepPartitionDigest(plans), device };
}

export function deriveStepObject(
  tape: StepTape,
  receipts: StepReceipts,
  /** Device class the partition's legality is keyed to (§2.4 device key;
   *  islands R2). Defaults to "webgpu" — the recorder only witnesses on the GPU
   *  path; the field exists so a device change is a BucketMiss (phase-6 §2.4). */
  device = "webgpu",
): StepObject {
  // Split the canonical bucketKey `b:<structHash>:<fps>` — the recorder's own
  // construction (`step-tape.ts stEndStep`). The struct hash is the second
  // colon-field; the fps are the remainder (may itself be empty).
  const rest = tape.bucketKey.startsWith("b:")
    ? tape.bucketKey.slice(2)
    : tape.bucketKey;
  const firstColon = rest.indexOf(":");
  const structHash = firstColon >= 0 ? rest.slice(0, firstColon) : rest;
  const fpsField = firstColon >= 0 ? rest.slice(firstColon + 1) : "";
  const orderedFps =
    fpsField.length > 0 ? fpsField.split("+").map((h) => parseInt(h, 16)) : [];

  const slots: StepSlotDecl[] = tape.slots.map((s) => ({
    id: s.id,
    name: s.name,
    shape: s.shape,
    dtype: s.dtype,
    source: s.source,
  }));

  const templateIds = [...tape.templateIds];

  const declaration: StepDeclaration = {
    boundaryStructHash: structHash,
    slots,
    // Recompute-segment fact (task #98 phase 3, ruling 3): the fps of the plans
    // that carried a checkpoint boundary this step — the DECLARED recompute
    // segments, in plan order. Empty when the step is not checkpointed. This is
    // now a REAL recompute fact derived from the observed boundaries, no longer
    // a placeholder alias of the all-fps partitionRef. The executor's arena-free
    // decision for these plans is the functional half of the same declaration
    // (driven at run time by checkpoint() via RuntimeEngine.declareRecomputeSegments).
    recomputeRef: tape.recomputeFps,
    // Partition reference: the per-plan boundaryHash projection is ALREADY mixed
    // into each plan's fp (islands I1); the ordered fps ARE that projection's
    // reference at step altitude. Phase 6 lifts it to a StepPartition (below).
    partitionRef: orderedFps,
    // The `partition` FACET (task #98 phase 6, ruling 4): a read-only view over
    // the detector's per-plan islands tokens (`tape.partitionHashes`, recorded
    // from `PlanPartition.boundaryHash`), ALIGNED with the ordered fps. Single
    // source — the detector owns membership; this projects it per step.
    partition: derivePartition(orderedFps, tape.partitionHashes ?? [], device),
    ringRef: null,
  };

  const skeleton: StepSkeletonRef = {
    orderedFps,
    templateIds,
  };

  return {
    declaration,
    skeleton,
    epoch: tape.epoch,
    regime: { stepScopedCleanup: tape.regime.stepScopedCleanup },
    receipts,
  };
}
