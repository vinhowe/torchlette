/**
 * The Step Object (task #98 phase 1) — reify, null-clean.
 *
 * Pins the properties the design's §6 Phase 1 null test demands
 * (docs/step-object-design.md §2):
 *  - the object is a PURE PROJECTION of a witnessed tape — its digest recomputes
 *    byte-identical to the tape's own `bucketKey` (§2.2 / §3.1 agreement seam;
 *    the single-source invariant, no second identity);
 *  - the digest canonicalizes receipts + per-step values OUT (§2.2 / §2.5) —
 *    two objects that differ only in receipt counters share a digest;
 *  - the digest is byte-STABLE across identical inputs (deterministic
 *    regeneration, the null-test determinism clause);
 *  - stable slot NAMES round-trip: the same declaration yields the same slot ids
 *    across re-witnesses (§10 ruling 1 — "the α slot", not "slot #7 of tape 43");
 *  - the typed-refusal enumeration maps onto the 6 tape guard classes (§2.3).
 *
 * Pure logic (no GPU) — runs in the cpu project. The full-stack null
 * DIFFERENTIAL (real decode/training tape streams byte-identical with reify on
 * vs the pre-branch baseline) is the GPU tool tools/t-step-object-null.ts.
 */
import { describe, expect, it } from "vitest";
import type { StepReceipts } from "../src/core/step-object";
import {
  deriveStepObject,
  REFUSAL_GUARD,
  type StepRefusalReason,
  stepObjectDigest,
  stepObjectDigestMatchesBucket,
  stepPartitionDigest,
  stepPartitionReproducesPerPlan,
} from "../src/core/step-object";
import type { StepTape } from "../src/core/step-tape";

// A synthetic witnessed tape shaped exactly like `stEndStep` builds one: the
// bucketKey is `b:<structHashHex>:<fp+fp+…>`, slots carry declaration-stable
// ids, templateIds is the deduped fp set.
function fakeTape(over: Partial<StepTape> = {}): StepTape {
  const fps = over.templateIds
    ? [...over.templateIds].map((f) => (f >>> 0).toString(16))
    : ["2e67bb41", "16e9a591"];
  const structHash = "deadbeef";
  return {
    bucketKey: `b:${structHash}:${fps.join("+")}`,
    entries: [],
    slots: [
      {
        id: "w:2e67bb41:21",
        name: "tensorFromArray[1,1]",
        shape: [1, 1],
        dtype: "i32",
        source: "upload",
        stable: true,
      },
      {
        id: "sc:16e9a591:959:1",
        name: "mul.scalar[1]",
        shape: [],
        dtype: "f32",
        source: "scalar",
      },
    ],
    epoch: 7,
    structGen: 3,
    regime: { stepScopedCleanup: false },
    templateIds: new Set([0x2e67bb41, 0x16e9a591]),
    recomputeFps: [],
    partitionHashes: [0x1111, 0x2222],
    recordedAtStep: 12,
    ...over,
  };
}

const NO_RECEIPTS: StepReceipts = {
  refusals: 0,
  eligiblePairs: 1,
  structureMisses: 0,
  planInvalidations: 0,
  boundaryResets: 0,
};

describe("StepObject — reify, null-clean (task #98 phase 1)", () => {
  it("digest recomputes byte-identical to the tape's own bucketKey (single source)", () => {
    const tape = fakeTape();
    const obj = deriveStepObject(tape, NO_RECEIPTS);
    // The pure-projection invariant (§3.1 agreement seam): the object DERIVES
    // its identity from the tape, never recomputing an independent one.
    expect(stepObjectDigest(obj)).toBe(tape.bucketKey);
    expect(stepObjectDigestMatchesBucket(obj, tape)).toBe(true);
  });

  it("the witnessed phase reifies the ORDERED plan fps (order + repeats preserved)", () => {
    // A repeated template (executes twice in one step): NEVER deduped in the
    // ordered fps, so the digest must carry the repeat.
    const tape = fakeTape({
      bucketKey: "b:deadbeef:abc+abc+def",
      templateIds: new Set([0xabc, 0xdef]),
    });
    const obj = deriveStepObject(tape, NO_RECEIPTS);
    expect(obj.skeleton?.orderedFps).toEqual([0xabc, 0xabc, 0xdef]);
    // Deduped template ids stay the invalidation index (guard 4).
    expect([...(obj.skeleton?.templateIds ?? [])].sort()).toEqual([
      0xabc, 0xdef,
    ]);
    expect(stepObjectDigest(obj)).toBe(tape.bucketKey);
  });

  it("digest canonicalizes RECEIPTS out (§2.2/§2.5 — receipts hash into neither)", () => {
    const tape = fakeTape();
    const a = deriveStepObject(tape, NO_RECEIPTS);
    const b = deriveStepObject(tape, {
      refusals: 99,
      eligiblePairs: 42,
      structureMisses: 7,
      planInvalidations: 5,
      boundaryResets: 3,
    });
    // Different receipts, SAME digest — receipts are not identity.
    expect(stepObjectDigest(a)).toBe(stepObjectDigest(b));
    expect(a.receipts.refusals).toBe(0);
    expect(b.receipts.refusals).toBe(99);
  });

  it("digest is byte-stable across identical inputs (determinism clause)", () => {
    // Re-derive from a fresh-but-identical tape (a fresh witness of the same
    // declaration) → byte-identical digest.
    const a = deriveStepObject(fakeTape(), NO_RECEIPTS);
    const b = deriveStepObject(fakeTape(), NO_RECEIPTS);
    expect(stepObjectDigest(a)).toBe(stepObjectDigest(b));
  });

  it("slot NAMES round-trip stable across re-witnesses (§10 ruling 1)", () => {
    // Two witnesses of the same declaration (different tape ordinals) must
    // yield the SAME slot ids — the id embeds only (op fp, pos[, ii]), no ordinal.
    const first = deriveStepObject(
      fakeTape({ recordedAtStep: 3 }),
      NO_RECEIPTS,
    );
    const second = deriveStepObject(
      fakeTape({ recordedAtStep: 99 }),
      NO_RECEIPTS,
    );
    expect(first.declaration.slots.map((s) => s.id)).toEqual(
      second.declaration.slots.map((s) => s.id),
    );
    // The α slot survives as "the α slot", not "slot #k of tape N".
    expect(first.declaration.slots.map((s) => s.id)).toEqual([
      "w:2e67bb41:21",
      "sc:16e9a591:959:1",
    ]);
  });

  it("partition reifies from the ordered fps; recompute is the DECLARED recompute segments (task #98 phase 3), ring is a placeholder (§2.5)", () => {
    // Not checkpointed: no recompute segments declared → empty recomputeRef,
    // while partition is still the full ordered-fps projection (phase 6 knob).
    const obj = deriveStepObject(fakeTape(), NO_RECEIPTS);
    expect(obj.declaration.recomputeRef).toEqual([]);
    expect(obj.declaration.partitionRef).toEqual([0x2e67bb41, 0x16e9a591]);
    // Ring config is not derivable from a tape alone → null placeholder (the
    // §2.5 discipline: no field the doc lacks, no value the source can't give).
    expect(obj.declaration.ringRef).toBeNull();
  });

  it("recomputeRef carries ONLY the recompute-bearing plan fps (a real declared fact, not the all-fps partition alias)", () => {
    // A checkpointed step: the second plan carried a checkpoint boundary. The
    // recompute facet declares exactly that plan; the partition still spans all.
    const obj = deriveStepObject(
      fakeTape({ recomputeFps: [0x16e9a591] }),
      NO_RECEIPTS,
    );
    expect(obj.declaration.recomputeRef).toEqual([0x16e9a591]);
    expect(obj.declaration.partitionRef).toEqual([0x2e67bb41, 0x16e9a591]);
    // Distinct facets — recompute is a strict subset when checkpointing is
    // selective (the two are no longer the same reference).
    expect(obj.declaration.recomputeRef).not.toEqual(
      obj.declaration.partitionRef,
    );
  });

  it("epoch is reified as the scoped-memory epoch, regime as guard-5 data", () => {
    const obj = deriveStepObject(
      fakeTape({ epoch: 11, regime: { stepScopedCleanup: true } }),
      NO_RECEIPTS,
    );
    expect(obj.epoch).toBe(11);
    expect(obj.regime.stepScopedCleanup).toBe(true);
  });

  it("the partition FACET (task #98 phase 6) projects the per-plan islands tokens, aligned + device-keyed", () => {
    const obj = deriveStepObject(fakeTape(), NO_RECEIPTS);
    // Per-plan tokens aligned with the ordered fps (fakeTape: 0x1111, 0x2222).
    expect(obj.declaration.partition.plans).toEqual([
      { fp: 0x2e67bb41, boundaryHash: 0x1111 },
      { fp: 0x16e9a591, boundaryHash: 0x2222 },
    ]);
    // Device key present (§2.4 — a device change is a BucketMiss).
    expect(obj.declaration.partition.device).toBe("webgpu");
    // The step-level partition digest is a pure function of the pairs.
    expect(obj.declaration.partition.boundaryDigest).toBe(
      stepPartitionDigest(obj.declaration.partition.plans),
    );
  });

  it("the partition facet's boundaryDigest is null-stable + partition-discriminating", () => {
    const a = deriveStepObject(fakeTape(), NO_RECEIPTS);
    const b = deriveStepObject(fakeTape(), NO_RECEIPTS);
    // Null-stable: same partition → same digest.
    expect(a.declaration.partition.boundaryDigest).toBe(
      b.declaration.partition.boundaryDigest,
    );
    // Discriminating: a different per-plan token → a different digest.
    const c = deriveStepObject(
      fakeTape({ partitionHashes: [0x1111, 0x9999] }),
      NO_RECEIPTS,
    );
    expect(c.declaration.partition.boundaryDigest).not.toBe(
      a.declaration.partition.boundaryDigest,
    );
  });

  it("the I1 agreement seam at step altitude: the projection reproduces the per-plan boundaryHashes byte-identically (§3.3)", () => {
    const obj = deriveStepObject(fakeTape(), NO_RECEIPTS);
    // The step partition reproduces the detector's per-plan tokens exactly.
    expect(
      stepPartitionReproducesPerPlan(
        obj.declaration.partition,
        [0x1111, 0x2222],
      ),
    ).toBe(true);
    // A byte-off projection is refused (a projection bug, never benign).
    expect(
      stepPartitionReproducesPerPlan(
        obj.declaration.partition,
        [0x1111, 0x2223],
      ),
    ).toBe(false);
    // A length mismatch is refused.
    expect(
      stepPartitionReproducesPerPlan(obj.declaration.partition, [0x1111]),
    ).toBe(false);
  });

  it("missing per-plan tokens project to 0 (sound null the seam still reproduces)", () => {
    // An older tape / a plan with no reified partition: partitionHashes short.
    const obj = deriveStepObject(
      fakeTape({ partitionHashes: [] }),
      NO_RECEIPTS,
    );
    expect(obj.declaration.partition.plans.map((p) => p.boundaryHash)).toEqual([
      0, 0,
    ]);
    expect(
      stepPartitionReproducesPerPlan(obj.declaration.partition, [0, 0]),
    ).toBe(true);
  });

  it("the typed-refusal enumeration maps onto the 6 tape guard classes (§2.3)", () => {
    const reasons: StepRefusalReason[] = [
      "StructureMiss",
      "BucketMiss",
      "UndeclaredVariance",
      "PlanInvalid",
      "EpochMiss",
      "StrictNet",
    ];
    // Every reason maps onto exactly one guard number 1..6, bijectively.
    const guards = reasons.map((r) => REFUSAL_GUARD[r]);
    expect(guards).toEqual([1, 2, 3, 4, 5, 6]);
    expect(new Set(guards).size).toBe(6);
  });

  it("null skeleton (no witnessed tape): digest carries only the struct hash", () => {
    // A declaration with no ordered fps (never witnessed) still has a stable
    // struct-hash digest — the empty-fps form the recorder never stores but the
    // object type admits (skeleton: null).
    const obj = deriveStepObject(fakeTape(), NO_RECEIPTS);
    const noSkeleton = { ...obj, skeleton: null };
    expect(stepObjectDigest(noSkeleton)).toBe("b:deadbeef:");
  });
});
