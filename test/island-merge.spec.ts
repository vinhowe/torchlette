/**
 * S3 FUSE LIVE WIRING — the detector's own partition merge (`mergeIslands`).
 *
 * `mergeIslands` is THE partition mutator the editor channel drives (no second
 * owner of membership — it lives in the detector's file). These pins are the
 * move-algebra laws at the reified-partition altitude (docs/islands-design.md §2,
 * docs/step-object-design.md §5.3):
 *  - I1 AGREEMENT: a channel-shaped merge (call `mergeIslands`) produces a
 *    partition byte-identical (boundaryHash) to constructing the merged islands
 *    natively and hashing them — the single-source guarantee the executor relies
 *    on when it mixes the merged token into the fingerprint.
 *  - CONVEXITY: a non-adjacent merge is refused (`MERGE_NONCONVEX`); an out-of-
 *    range/degenerate merge is refused (`MERGE_OUT_OF_RANGE`). Refusals never
 *    mutate.
 *  - DISCRIMINATION + NULL: the merged token differs from the default token and
 *    re-fingerprints the plan; the same merge twice yields the identical token.
 *  - INVERSE-BY-CONSTRUCTION: merge drops exactly one island and unions members;
 *    the default partition is the split back (rollback digest identity, executor).
 */
import { describe, expect, it } from "vitest";
import {
  computePlanFingerprint,
  type Island,
  mergeIslands,
  partitionBoundaryHash,
  reifyPartition,
} from "../src/compiler/fusion-detect";
import { createPendingRef, LazyIRNode } from "../src/graph/types";

const SEGS = [
  { kind: "sequential" as const, finalPoss: [0, 1, 2] },
  { kind: "fused" as const, finalPoss: [3, 4, 5] },
  { kind: "sequential" as const, finalPoss: [6] },
  { kind: "reduction" as const, finalPoss: [7, 8] },
];

describe("mergeIslands — the detector's own partition merge (S3)", () => {
  it("I1 agreement: channel-driven merge == native merged islands byte-identically", () => {
    const p = reifyPartition(SEGS);
    // The channel-driven path: express merge(1,2) as input to the detector fn.
    const m = mergeIslands(p, 1, 2);
    expect(m.ok).toBe(true);
    if (!m.ok) return;
    // The detector performing "the same merge natively": construct the merged
    // island set by hand (fused union of islands 1 and 2, in plan order) and hash.
    const nativeIslands: Island[] = [
      { kind: "sequential", members: [0, 1, 2] },
      { kind: "fused", members: [3, 4, 5, 6] },
      { kind: "reduction", members: [7, 8] },
    ];
    expect(m.partition.boundaryHash).toBe(partitionBoundaryHash(nativeIslands));
    // Island count dropped by exactly one; the merged island unions both members.
    expect(m.partition.islands.length).toBe(p.islands.length - 1);
    expect(m.partition.islands[1].members).toEqual([3, 4, 5, 6]);
  });

  it("order-independent: merge(a,b) == merge(b,a)", () => {
    const p = reifyPartition(SEGS);
    const ab = mergeIslands(p, 1, 2);
    const ba = mergeIslands(p, 2, 1);
    expect(ab.ok && ba.ok).toBe(true);
    if (ab.ok && ba.ok)
      expect(ab.partition.boundaryHash).toBe(ba.partition.boundaryHash);
  });

  it("null-stable: the same merge twice yields the identical token", () => {
    const p = reifyPartition(SEGS);
    const a = mergeIslands(p, 0, 1);
    const b = mergeIslands(p, 0, 1);
    expect(a.ok && b.ok).toBe(true);
    if (a.ok && b.ok)
      expect(a.partition.boundaryHash).toBe(b.partition.boundaryHash);
  });

  it("discriminates: the merged token differs from the default token", () => {
    const p = reifyPartition(SEGS);
    const m = mergeIslands(p, 1, 2);
    expect(m.ok).toBe(true);
    if (m.ok) expect(m.partition.boundaryHash).not.toBe(p.boundaryHash);
  });

  it("refuses non-adjacent (non-convex) merges without mutating", () => {
    const p = reifyPartition(SEGS);
    const before = p.islands.length;
    const m = mergeIslands(p, 0, 3);
    expect(m.ok).toBe(false);
    if (!m.ok) expect(m.code).toBe("MERGE_NONCONVEX");
    expect(p.islands.length).toBe(before); // input untouched
  });

  it("refuses degenerate / out-of-range merges", () => {
    const p = reifyPartition(SEGS);
    expect(mergeIslands(p, 1, 1)).toEqual({
      ok: false,
      code: "MERGE_OUT_OF_RANGE",
    });
    expect(mergeIslands(p, 0, 99)).toEqual({
      ok: false,
      code: "MERGE_OUT_OF_RANGE",
    });
    expect(mergeIslands(p, -1, 0)).toEqual({
      ok: false,
      code: "MERGE_OUT_OF_RANGE",
    });
  });

  it("the merged token re-fingerprints a plan (executor I1 seam)", () => {
    const x = new LazyIRNode(
      1,
      "leafInput" as never,
      [],
      [8, 4],
      "f32",
      "webgpu",
    );
    const a = new LazyIRNode(
      2,
      "relu" as never,
      [createPendingRef(x)],
      [8, 4],
      "f32",
      "webgpu",
    );
    const b = new LazyIRNode(
      3,
      "exp" as never,
      [createPendingRef(a)],
      [8, 4],
      "f32",
      "webgpu",
    );
    const nodes = [x, a, b];
    const p = reifyPartition([
      { kind: "sequential", finalPoss: [0] },
      { kind: "fused", finalPoss: [1, 2] },
    ]);
    const m = mergeIslands(p, 0, 1);
    expect(m.ok).toBe(true);
    if (!m.ok) return;
    const dflt = computePlanFingerprint(nodes);
    const edited = computePlanFingerprint(
      nodes,
      undefined,
      m.partition.boundaryHash,
    );
    // The edit re-keys the template (primary + structural), the re-witness trigger.
    expect(edited.primary).not.toBe(dflt.primary);
    expect(edited.structural).not.toBe(dflt.structural);
  });
});
