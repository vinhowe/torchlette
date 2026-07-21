/**
 * Coverage-campaign C3: `splitPackItems` class-splitting partition gate (CPU —
 * pure function, no GPU). Proves the split is (a) size-capped (each sub-class's
 * Σ size ≤ maxElems where the items allow), (b) DETERMINISTIC (same params ⇒ same
 * split ⇒ stable compiled-plan templates), (c) order- and membership-preserving
 * (the flat concat is position-independent, so a sub-class is a contiguous run of
 * the original order), and (d) a no-op when the whole class already fits.
 */
import { describe, expect, it } from "vitest";
import { splitPackItems, type PackItem } from "../src/optim/pack-optimizer";

// Minimal PackItem stand-ins: only `param.shape` is read by splitPackItems.
function item(id: number, size: number): PackItem {
  return {
    id,
    param: { shape: [size] },
    grad: {},
    state: [],
  } as unknown as PackItem;
}

const sizeOf = (it: PackItem) =>
  it.param.shape.reduce((a: number, b: number) => a * b, 1);
const flatIds = (subs: PackItem[][]) => subs.flat().map((it) => it.id);

describe("splitPackItems (C3 class-splitting)", () => {
  it("returns one sub-class unchanged when the class already fits", () => {
    const items = [item(0, 100), item(1, 200), item(2, 50)]; // Σ=350
    const subs = splitPackItems(items, 1000);
    expect(subs.length).toBe(1);
    expect(subs[0].map((it) => it.id)).toEqual([0, 1, 2]);
  });

  it("Infinity threshold never splits (CPU / no-limit backend)", () => {
    const items = [item(0, 1e9), item(1, 1e9)];
    expect(splitPackItems(items, Infinity)).toEqual([items]);
  });

  it("caps each sub-class at ≤ maxElems (greedy sequential)", () => {
    const items = [item(0, 40), item(1, 40), item(2, 40), item(3, 40)];
    const subs = splitPackItems(items, 100); // 40+40=80 ok, +40=120>100 → split
    for (const s of subs) {
      const total = s.reduce((a, it) => a + sizeOf(it), 0);
      expect(total).toBeLessThanOrEqual(100);
    }
    // 40,40 | 40,40
    expect(subs.map((s) => s.map((it) => it.id))).toEqual([
      [0, 1],
      [2, 3],
    ]);
  });

  it("isolates a single item larger than maxElems into its own sub-class", () => {
    const items = [item(0, 30), item(1, 250), item(2, 30)];
    const subs = splitPackItems(items, 100);
    // 30 | 250(alone, >100 — unavoidable) | 30
    expect(subs.map((s) => s.map((it) => it.id))).toEqual([[0], [1], [2]]);
  });

  it("is deterministic and order/membership-preserving", () => {
    const items = [item(5, 60), item(2, 60), item(9, 60), item(1, 60)];
    const a = splitPackItems(items, 100);
    const b = splitPackItems(items, 100);
    expect(a.map((s) => s.map((it) => it.id))).toEqual(
      b.map((s) => s.map((it) => it.id)),
    );
    // every item appears exactly once, in the original order
    expect(flatIds(a)).toEqual([5, 2, 9, 1]);
  });
});
