/**
 * The Step Object — the step-editor's PARTITION SURFACE (task #98; reduced P4b-R).
 *
 * Normative source: docs/step-object-design.md §2 (the two-phase schema §2.1, the
 * partition facet §3.3, guards as typed refusals §2.3).
 *
 * WHAT SURVIVES. The step-tape recorder + replay that this module once projected
 * a witnessed StepObject from were DELETED in P4b-R (the tape added no decode
 * value and training is served by the whole-step compiler). What remains is the
 * pole the step-editor binds to: the `StepPartition` facet + its identity digest
 * (`stepPartitionDigest` / `stepPartitionReproducesPerPlan`), consumed by
 * `src/schedule/moves/step-edit-channel.ts` (the `StepEditChannel` records edit
 * REQUESTS against a partition's `boundaryDigest`). The step-DATA type surface
 * (`StepObject` / `StepDeclaration` / `StepSlotDecl` / `StepSkeletonRef` /
 * `StepReceipts` / `stepObjectDigest` / the typed `StepRefusalReason`) is retained
 * as the declarative schema; the tape-consuming CONSTRUCTOR (`deriveStepObject`)
 * and the tape-agreement assert (`stepObjectDigestMatchesBucket`) were pruned with
 * the tape.
 *
 * SINGLE SOURCE. Membership is still owned by the detector (`fusion-detect.ts`
 * `reifyPartition`); `StepPartition` is a read-only VIEW keyed to it — the
 * "loop-nest VIEW derived by a canonical ordering rule" pattern. This module owns
 * no state and adds no env flag.
 */

/** Source class of a dynamic slot (inlined from the deleted step-tape module,
 *  P4b-R): what per-step-varying value a declared slot delivers. */
export type DynamicSlotSource = "tokenId" | "upload" | "payload" | "scalar";

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
