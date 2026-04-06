/**
 * Scheduler algorithm tests.
 */
import { describe, expect, it } from "vitest";
import {
  computePeak,
  totalBytes,
  trivialAssignment,
} from "../../src/compiler/scheduler/cost-model";
import type { TensorLiveRange } from "../../src/compiler/scheduler/live-range";
import {
  bestFitScheduler,
  firstFitScheduler,
  validateAssignment,
} from "../../src/compiler/scheduler/schedulers";

function r(id: number, start: number, end: number, size: number): TensorLiveRange {
  return { id, start, end, size, external: false };
}

function ranges(...entries: TensorLiveRange[]): Map<number, TensorLiveRange> {
  const m = new Map<number, TensorLiveRange>();
  for (const e of entries) m.set(e.id, e);
  return m;
}

// ============================================================================
// Correctness: no two overlapping tensors share a slot
// ============================================================================

describe("firstFit / correctness", () => {
  it("disjoint intervals pack into one slot", () => {
    const rs = ranges(r(1, 0, 2, 100), r(2, 3, 5, 100), r(3, 6, 8, 100));
    const a = firstFitScheduler(rs);
    expect(a.slotSizes.size).toBe(1);
    expect(validateAssignment(rs, a).valid).toBe(true);
  });

  it("all-overlapping intervals need N slots", () => {
    const rs = ranges(r(1, 0, 10, 100), r(2, 0, 10, 200), r(3, 0, 10, 50));
    const a = firstFitScheduler(rs);
    expect(a.slotSizes.size).toBe(3);
    expect(validateAssignment(rs, a).valid).toBe(true);
  });

  it("produces valid assignment on mixed overlap", () => {
    const rs = ranges(
      r(1, 0, 5, 100),
      r(2, 3, 8, 200),
      r(3, 6, 10, 50),
      r(4, 9, 12, 75),
      r(5, 11, 15, 100),
    );
    const a = firstFitScheduler(rs);
    expect(validateAssignment(rs, a).valid).toBe(true);
  });
});

describe("bestFit / correctness", () => {
  it("produces valid assignments on random-ish inputs", () => {
    const rs = ranges(
      r(1, 0, 5, 100),
      r(2, 3, 8, 200),
      r(3, 6, 10, 50),
      r(4, 9, 12, 75),
      r(5, 2, 11, 150),
      r(6, 13, 16, 50),
    );
    const a = bestFitScheduler(rs);
    expect(validateAssignment(rs, a).valid).toBe(true);
  });
});

// ============================================================================
// Determinism
// ============================================================================

describe("schedulers / determinism", () => {
  it("first-fit is deterministic", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 3, 8, 200), r(3, 6, 10, 50));
    const a = firstFitScheduler(rs);
    const b = firstFitScheduler(rs);
    expect([...a.tensorToSlot.entries()].sort()).toEqual(
      [...b.tensorToSlot.entries()].sort(),
    );
  });

  it("best-fit is deterministic", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 3, 8, 200), r(3, 6, 10, 50));
    const a = bestFitScheduler(rs);
    const b = bestFitScheduler(rs);
    expect([...a.tensorToSlot.entries()].sort()).toEqual(
      [...b.tensorToSlot.entries()].sort(),
    );
  });
});

// ============================================================================
// Quality: peak vs trivial
// ============================================================================

describe("schedulers / quality", () => {
  it("first-fit ≤ trivial assignment peak", () => {
    // Chain-like workload where reuse is possible.
    const rs = ranges(
      r(1, 0, 2, 100),
      r(2, 1, 3, 100),
      r(3, 2, 4, 100),
      r(4, 4, 6, 100),
      r(5, 5, 7, 100),
    );
    const trivial = computePeak(rs, trivialAssignment(rs));
    const ff = computePeak(rs, firstFitScheduler(rs));
    expect(ff).toBeLessThanOrEqual(trivial);
  });

  it("best-fit ≤ trivial", () => {
    const rs = ranges(
      r(1, 0, 2, 100),
      r(2, 1, 3, 100),
      r(3, 2, 4, 100),
      r(4, 4, 6, 100),
      r(5, 5, 7, 100),
    );
    const trivial = computePeak(rs, trivialAssignment(rs));
    const bf = computePeak(rs, bestFitScheduler(rs));
    expect(bf).toBeLessThanOrEqual(trivial);
  });

  it("on chain of disjoint tensors, first-fit uses 1 slot", () => {
    const rs = ranges(r(1, 0, 2, 100), r(2, 3, 5, 100), r(3, 6, 8, 100));
    const a = firstFitScheduler(rs);
    expect(a.slotSizes.size).toBe(1);
    expect(computePeak(rs, a)).toBe(100);
  });

  it("first-fit on real-ish workload reserves less memory than trivial", () => {
    // 10 tensors, mostly disjoint pairs, with some overlaps.
    const rs = ranges(
      r(1, 0, 4, 1000),
      r(2, 1, 5, 500),
      r(3, 5, 9, 1000),
      r(4, 6, 10, 500),
      r(5, 10, 14, 1000),
      r(6, 11, 15, 500),
      r(7, 15, 19, 1000),
      r(8, 16, 20, 500),
      r(9, 20, 24, 1000),
      r(10, 21, 25, 500),
    );
    // totalBytes is what the scheduler actually optimizes — trivial reserves
    // one slot per tensor (7500), first-fit should pack disjoint chains into
    // far fewer slots.
    const trivialTotal = totalBytes(trivialAssignment(rs));
    const ffTotal = totalBytes(firstFitScheduler(rs));
    expect(ffTotal).toBeLessThan(trivialTotal);
    // Strong claim: first-fit should pack all 1000-sized and all 500-sized
    // tensors into two slots, yielding ~1500 total.
    expect(ffTotal).toBeLessThanOrEqual(2000);
    // computePeak stays near the theoretical lower bound (1500).
    const ffPeak = computePeak(rs, firstFitScheduler(rs));
    expect(ffPeak).toBeLessThanOrEqual(1500);
  });
});

// ============================================================================
// Bounds
// ============================================================================

describe("firstFit / bounds", () => {
  it("#slots <= trivial's #slots", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 6, 10, 100), r(3, 11, 15, 100));
    const ff = firstFitScheduler(rs);
    const trivial = trivialAssignment(rs);
    expect(ff.slotSizes.size).toBeLessThanOrEqual(trivial.slotSizes.size);
  });

  it("#slots >= max concurrent tensors (lower bound)", () => {
    // 3 all-overlapping tensors need at least 3 slots.
    const rs = ranges(r(1, 0, 10, 100), r(2, 0, 10, 100), r(3, 0, 10, 100));
    const ff = firstFitScheduler(rs);
    expect(ff.slotSizes.size).toBeGreaterThanOrEqual(3);
  });
});

// ============================================================================
// validateAssignment
// ============================================================================

describe("validateAssignment", () => {
  it("valid assignment returns valid=true", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 6, 10, 100));
    const a = firstFitScheduler(rs);
    expect(validateAssignment(rs, a).valid).toBe(true);
  });

  it("detects overlapping tensors in same slot", () => {
    const rs = ranges(r(1, 0, 5, 100), r(2, 3, 8, 100));
    // Force both into slot 0.
    const a = {
      tensorToSlot: new Map([
        [1, 0],
        [2, 0],
      ]),
      slotSizes: new Map([[0, 100]]),
    };
    const result = validateAssignment(rs, a);
    expect(result.valid).toBe(false);
    expect(result.conflicts.length).toBe(1);
  });
});
