/**
 * Interval graph tests.
 */
import { describe, expect, it } from "vitest";
import {
  IntervalGraph,
  type Interval,
  intervalsOverlap,
} from "../../src/compiler/scheduler/interval-graph";

// ============================================================================
// Helpers
// ============================================================================

function iv(id: number, start: number, end: number, size = 1): Interval {
  return { id, start, end, size };
}

// ============================================================================
// intervalsOverlap
// ============================================================================

describe("intervalsOverlap", () => {
  it("non-touching ranges don't overlap", () => {
    expect(intervalsOverlap(iv(1, 0, 2), iv(2, 4, 6))).toBe(false);
    expect(intervalsOverlap(iv(1, 5, 10), iv(2, 0, 3))).toBe(false);
  });

  it("touching ranges (share endpoint) DO overlap", () => {
    expect(intervalsOverlap(iv(1, 0, 5), iv(2, 5, 10))).toBe(true);
  });

  it("nested ranges overlap", () => {
    expect(intervalsOverlap(iv(1, 0, 10), iv(2, 3, 7))).toBe(true);
  });

  it("identical ranges overlap", () => {
    expect(intervalsOverlap(iv(1, 2, 5), iv(2, 2, 5))).toBe(true);
  });

  it("order-independent", () => {
    const a = iv(1, 0, 5);
    const b = iv(2, 3, 8);
    expect(intervalsOverlap(a, b)).toBe(intervalsOverlap(b, a));
  });

  it("single-point ranges", () => {
    expect(intervalsOverlap(iv(1, 3, 3), iv(2, 3, 3))).toBe(true);
    expect(intervalsOverlap(iv(1, 3, 3), iv(2, 4, 4))).toBe(false);
  });
});

// ============================================================================
// IntervalGraph basics
// ============================================================================

describe("IntervalGraph / basics", () => {
  it("add + get", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    expect(g.get(1)).toEqual(iv(1, 0, 5));
    expect(g.get(99)).toBeUndefined();
  });

  it("all() returns all added intervals", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    g.add(iv(2, 6, 10));
    expect(g.all().length).toBe(2);
  });

  it("rejects duplicate ids", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    expect(() => g.add(iv(1, 10, 15))).toThrow(/duplicate id/i);
  });

  it("rejects inverted intervals (start > end)", () => {
    const g = new IntervalGraph();
    expect(() => g.add(iv(1, 10, 5))).toThrow(/start.*end/i);
  });
});

// ============================================================================
// overlaps
// ============================================================================

describe("IntervalGraph / overlaps", () => {
  it("returns true for overlapping intervals", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    g.add(iv(2, 3, 8));
    expect(g.overlaps(1, 2)).toBe(true);
  });

  it("returns false for disjoint intervals", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    g.add(iv(2, 10, 15));
    expect(g.overlaps(1, 2)).toBe(false);
  });

  it("returns false for unknown ids", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    expect(g.overlaps(1, 99)).toBe(false);
    expect(g.overlaps(99, 1)).toBe(false);
  });
});

// ============================================================================
// liveAt
// ============================================================================

describe("IntervalGraph / liveAt", () => {
  it("returns all intervals covering a position", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    g.add(iv(2, 3, 10));
    g.add(iv(3, 7, 12));
    expect(g.liveAt(4).map((i) => i.id).sort()).toEqual([1, 2]);
    expect(g.liveAt(8).map((i) => i.id).sort()).toEqual([2, 3]);
  });

  it("empty array outside all intervals", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 3, 5));
    expect(g.liveAt(0)).toEqual([]);
    expect(g.liveAt(10)).toEqual([]);
  });

  it("endpoints are inclusive", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 3, 5));
    expect(g.liveAt(3).map((i) => i.id)).toEqual([1]);
    expect(g.liveAt(5).map((i) => i.id)).toEqual([1]);
    expect(g.liveAt(6)).toEqual([]);
  });
});

// ============================================================================
// color (coloring)
// ============================================================================

describe("IntervalGraph / color", () => {
  it("disjoint intervals all get color 0", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 2));
    g.add(iv(2, 3, 5));
    g.add(iv(3, 6, 8));
    const c = g.color();
    expect(c.get(1)).toBe(0);
    expect(c.get(2)).toBe(0);
    expect(c.get(3)).toBe(0);
  });

  it("three all-overlapping intervals need three colors", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 10));
    g.add(iv(2, 0, 10));
    g.add(iv(3, 0, 10));
    const c = g.color();
    const used = new Set(c.values());
    expect(used.size).toBe(3);
  });

  it("chain of overlapping pairs uses 2 colors", () => {
    // a-b, b-c, c-d overlap but a-c, a-d, b-d don't: can use 2 colors.
    const g = new IntervalGraph();
    g.add(iv(1, 0, 3));
    g.add(iv(2, 2, 5));
    g.add(iv(3, 4, 7));
    g.add(iv(4, 6, 9));
    const c = g.color();
    const used = new Set(c.values());
    expect(used.size).toBe(2);
  });

  it("coloring is deterministic (same input → same output)", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    g.add(iv(2, 3, 8));
    g.add(iv(3, 6, 10));
    const c1 = g.color();
    const c2 = g.color();
    for (const id of [1, 2, 3]) {
      expect(c1.get(id)).toBe(c2.get(id));
    }
  });

  it("no two overlapping intervals share a color", () => {
    // Property test: random-ish intervals, verify invariant.
    const g = new IntervalGraph();
    const ranges: [number, number][] = [
      [0, 5],
      [2, 4],
      [3, 7],
      [6, 9],
      [8, 11],
      [10, 12],
      [1, 10],
    ];
    ranges.forEach(([s, e], i) => g.add(iv(i + 1, s, e)));
    const c = g.color();
    for (const a of g.all()) {
      for (const b of g.all()) {
        if (a.id === b.id) continue;
        if (intervalsOverlap(a, b)) {
          expect(c.get(a.id)).not.toBe(c.get(b.id));
        }
      }
    }
  });

  it("uses no more colors than max clique (optimal)", () => {
    // Two disjoint groups of 2: max clique = 2, should use 2 colors.
    const g = new IntervalGraph();
    g.add(iv(1, 0, 3));
    g.add(iv(2, 1, 4));
    // gap
    g.add(iv(3, 10, 13));
    g.add(iv(4, 11, 14));
    const c = g.color();
    expect(new Set(c.values()).size).toBe(2);
  });

  it("empty graph → empty coloring", () => {
    const g = new IntervalGraph();
    expect(g.color().size).toBe(0);
  });

  it("single interval → color 0", () => {
    const g = new IntervalGraph();
    g.add(iv(1, 0, 5));
    expect(g.color().get(1)).toBe(0);
  });
});
