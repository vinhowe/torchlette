/**
 * Islands IR (stage I0/I1) — the dispatch-partition as first-class data.
 *
 * Pins the three properties the tape integration depends on
 * (docs/islands-design.md §1, §6):
 *  - reification is faithful (kinds + member positions survive — the
 *    unit-level nodeIndex-alignment guarantee),
 *  - boundaryHash is null-stable (same partition recomputed → same token;
 *    a static graph must never re-key its template),
 *  - boundaryHash discriminates partitions (any boundary change → a new
 *    token; two partitions of one graph must not collide).
 */
import { describe, expect, it } from "vitest";
import {
  type Island,
  partitionBoundaryHash,
  reifyPartition,
} from "../src/compiler/fusion-detect";

const SEGS = [
  { kind: "sequential" as const, finalPoss: [0, 1, 2] },
  { kind: "fused" as const, finalPoss: [3, 4, 5] },
  { kind: "sequential" as const, finalPoss: [6] },
  { kind: "reduction" as const, finalPoss: [7, 8] },
];

describe("islands partition reification (I0)", () => {
  it("preserves kinds and member positions exactly", () => {
    const p = reifyPartition(SEGS);
    expect(p.islands.map((i) => i.kind)).toEqual([
      "sequential",
      "fused",
      "sequential",
      "reduction",
    ]);
    expect(p.islands.map((i) => i.members)).toEqual([
      [0, 1, 2],
      [3, 4, 5],
      [6],
      [7, 8],
    ]);
  });

  it("copies members (mutating the source does not alias the partition)", () => {
    const segs = [{ kind: "fused" as const, finalPoss: [0, 1] }];
    const p = reifyPartition(segs);
    segs[0].finalPoss.push(99);
    expect(p.islands[0].members).toEqual([0, 1]);
  });

  it("boundaryHash is null-stable: same partition → same token", () => {
    const a = reifyPartition(SEGS);
    const b = reifyPartition(SEGS.map((s) => ({ ...s, finalPoss: [...s.finalPoss] })));
    expect(a.boundaryHash).toBe(b.boundaryHash);
    // And the standalone hash agrees with the reified one (single source).
    expect(partitionBoundaryHash(a.islands)).toBe(a.boundaryHash);
  });

  it("boundaryHash discriminates: merging two islands changes the token", () => {
    const split: Island[] = [
      { kind: "fused", members: [0] },
      { kind: "fused", members: [1, 2] },
    ];
    const merged: Island[] = [{ kind: "fused", members: [0, 1, 2] }];
    expect(partitionBoundaryHash(split)).not.toBe(partitionBoundaryHash(merged));
  });

  it("boundaryHash discriminates: island kind changes the token", () => {
    const seq: Island[] = [{ kind: "sequential", members: [0, 1] }];
    const fused: Island[] = [{ kind: "fused", members: [0, 1] }];
    expect(partitionBoundaryHash(seq)).not.toBe(partitionBoundaryHash(fused));
  });

  it("boundaryHash discriminates: moving a cut point changes the token", () => {
    const cutAt1: Island[] = [
      { kind: "fused", members: [0] },
      { kind: "fused", members: [1, 2, 3] },
    ];
    const cutAt2: Island[] = [
      { kind: "fused", members: [0, 1] },
      { kind: "fused", members: [2, 3] },
    ];
    expect(partitionBoundaryHash(cutAt1)).not.toBe(partitionBoundaryHash(cutAt2));
  });
});

