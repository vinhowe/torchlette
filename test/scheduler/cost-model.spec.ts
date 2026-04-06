/**
 * Cost model tests: peak memory computation for assignments.
 */
import { describe, expect, it } from "vitest";
import {
  computePeak,
  trivialAssignment,
  totalBytes,
  type Assignment,
} from "../../src/compiler/scheduler/cost-model";
import type { TensorLiveRange } from "../../src/compiler/scheduler/live-range";

function r(
  id: number,
  start: number,
  end: number,
  size: number,
  external = false,
): TensorLiveRange {
  return { id, start, end, size, external };
}

function ranges(...entries: TensorLiveRange[]): Map<number, TensorLiveRange> {
  const m = new Map<number, TensorLiveRange>();
  for (const e of entries) m.set(e.id, e);
  return m;
}

function assign(
  slotMap: Record<number, number>,
  slotSizeMap: Record<number, number>,
): Assignment {
  return {
    tensorToSlot: new Map(Object.entries(slotMap).map(([k, v]) => [+k, v])),
    slotSizes: new Map(Object.entries(slotSizeMap).map(([k, v]) => [+k, v])),
  };
}

describe("computePeak / basics", () => {
  it("empty ranges → peak 0", () => {
    expect(computePeak(new Map(), trivialAssignment(new Map()))).toBe(0);
  });

  it("single tensor peak = its size", () => {
    const rs = ranges(r(1, 0, 5, 100));
    expect(computePeak(rs, trivialAssignment(rs))).toBe(100);
  });

  it("two disjoint tensors in own slots: peak = max(sizes)", () => {
    const rs = ranges(r(1, 0, 2, 100), r(2, 5, 10, 200));
    const a = assign({ 1: 0, 2: 1 }, { 0: 100, 1: 200 });
    expect(computePeak(rs, a)).toBe(200);
  });

  it("two disjoint tensors sharing a slot: peak = slot size", () => {
    const rs = ranges(r(1, 0, 2, 100), r(2, 5, 10, 200));
    const a = assign({ 1: 0, 2: 0 }, { 0: 200 });
    expect(computePeak(rs, a)).toBe(200);
  });

  it("two overlapping tensors in own slots: peak = sum", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 3, 8, 200));
    const a = assign({ 1: 0, 2: 1 }, { 0: 100, 1: 200 });
    expect(computePeak(rs, a)).toBe(300);
  });
});

describe("computePeak / properties", () => {
  it("peak >= max(tensor sizes)", () => {
    const rs = ranges(r(1, 0, 2, 100), r(2, 3, 5, 200), r(3, 6, 9, 50));
    const t = trivialAssignment(rs);
    const peak = computePeak(rs, t);
    expect(peak).toBeGreaterThanOrEqual(200);
  });

  it("peak <= sum of all slot sizes", () => {
    const rs = ranges(r(1, 0, 2, 100), r(2, 3, 5, 200), r(3, 6, 9, 50));
    const t = trivialAssignment(rs);
    expect(computePeak(rs, t)).toBeLessThanOrEqual(totalBytes(t));
  });
});

describe("trivialAssignment", () => {
  it("one slot per tensor", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 3, 8, 200));
    const a = trivialAssignment(rs);
    expect(a.tensorToSlot.size).toBe(2);
    expect(a.slotSizes.size).toBe(2);
  });

  it("each slot's size matches its tensor", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 3, 8, 200));
    const a = trivialAssignment(rs);
    const t1Slot = a.tensorToSlot.get(1)!;
    const t2Slot = a.tensorToSlot.get(2)!;
    expect(a.slotSizes.get(t1Slot)).toBe(100);
    expect(a.slotSizes.get(t2Slot)).toBe(200);
  });

  it("gives the NAIVE peak (sum of sizes live simultaneously)", () => {
    // Three overlapping tensors, peak should be their sum.
    const rs = ranges(r(1, 0, 10, 100), r(2, 0, 10, 200), r(3, 0, 10, 50));
    const a = trivialAssignment(rs);
    expect(computePeak(rs, a)).toBe(350);
  });
});
