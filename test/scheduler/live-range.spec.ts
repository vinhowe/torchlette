/**
 * Live range analyzer tests.
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../src/graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../../src/graph/types";
import {
  analyzeLiveRanges,
  dtypeBytes,
  liveRangeStats,
  tensorBytes,
} from "../../src/compiler/scheduler/live-range";

// ============================================================================
// Helpers
// ============================================================================

function pendingRef(node: LazyIRNode): LazyRef {
  return { kind: "pending", node };
}

function matRef(id: number): LazyRef {
  const storage = { id } as unknown as StorageHandle;
  return { kind: "materialized", storage };
}

/** Build a plan like: m1 → a → b → c, where each is a chain of add ops. */
function chainPlan(length: number, shape: number[] = [4, 4]): LazyIRNode[] {
  const plan: LazyIRNode[] = [];
  let prev: LazyRef = matRef(1000);
  for (let i = 0; i < length; i++) {
    const node = createLazyIRNode("add", [prev, matRef(1001)], shape, "f32", "cpu");
    plan.push(node);
    prev = pendingRef(node);
  }
  return plan;
}

// ============================================================================
// dtypeBytes / tensorBytes
// ============================================================================

describe("dtypeBytes", () => {
  it("covers standard dtypes", () => {
    expect(dtypeBytes("f32")).toBe(4);
    expect(dtypeBytes("f16")).toBe(2);
    expect(dtypeBytes("i32")).toBe(4);
    expect(dtypeBytes("u32")).toBe(4);
    expect(dtypeBytes("i8")).toBe(1);
    expect(dtypeBytes("u8")).toBe(1);
  });
});

describe("tensorBytes", () => {
  it("multiplies shape elements by dtype size", () => {
    expect(tensorBytes([4, 4], "f32")).toBe(16 * 4);
    expect(tensorBytes([2, 3, 4], "f16")).toBe(24 * 2);
  });
  it("scalar is one element", () => {
    expect(tensorBytes([], "f32")).toBe(4);
  });
});

// ============================================================================
// analyzeLiveRanges
// ============================================================================

describe("analyzeLiveRanges / chain", () => {
  it("producer consumed by very next node has range [i, i+1]", () => {
    const plan = chainPlan(3);
    const ranges = analyzeLiveRanges(plan);
    // plan[0] is consumed only by plan[1]
    expect(ranges.get(plan[0].id)).toMatchObject({ start: 0, end: 1 });
  });

  it("last node's output is not consumed in plan (end == start if no external)", () => {
    const plan = chainPlan(3);
    const ranges = analyzeLiveRanges(plan);
    expect(ranges.get(plan[2].id)).toMatchObject({
      start: 2,
      end: 2,
      external: false,
    });
  });

  it("every node has a range", () => {
    const plan = chainPlan(5);
    const ranges = analyzeLiveRanges(plan);
    expect(ranges.size).toBe(5);
  });

  it("start <= end for every range", () => {
    const plan = chainPlan(10);
    const ranges = analyzeLiveRanges(plan);
    for (const r of ranges.values()) {
      expect(r.start).toBeLessThanOrEqual(r.end);
    }
  });
});

describe("analyzeLiveRanges / branching", () => {
  it("producer used by multiple consumers spans all of them", () => {
    // plan: a, b=add(a, mat), c=add(a, mat), d=add(b, mat), e=add(a, mat)
    // a's consumers are at positions 1, 2, 4 → a.end=4
    const a = createLazyIRNode("add", [matRef(1), matRef(2)], [4], "f32", "cpu");
    const b = createLazyIRNode("add", [pendingRef(a), matRef(2)], [4], "f32", "cpu");
    const c = createLazyIRNode("add", [pendingRef(a), matRef(2)], [4], "f32", "cpu");
    const d = createLazyIRNode("add", [pendingRef(b), matRef(2)], [4], "f32", "cpu");
    const e = createLazyIRNode("add", [pendingRef(a), matRef(2)], [4], "f32", "cpu");
    const plan = [a, b, c, d, e];
    const ranges = analyzeLiveRanges(plan);
    expect(ranges.get(a.id)).toMatchObject({ start: 0, end: 4 });
    expect(ranges.get(b.id)).toMatchObject({ start: 1, end: 3 });
    expect(ranges.get(c.id)).toMatchObject({ start: 2, end: 2 });
  });
});

describe("analyzeLiveRanges / external nodes", () => {
  it("external nodes get end=plan.length-1", () => {
    const plan = chainPlan(5);
    const external = new Set<number>([plan[0].id]);
    const ranges = analyzeLiveRanges(plan, external);
    expect(ranges.get(plan[0].id)).toMatchObject({
      start: 0,
      end: 4,
      external: true,
    });
  });

  it("non-external nodes unaffected by external set", () => {
    const plan = chainPlan(5);
    const external = new Set<number>([plan[0].id]);
    const ranges = analyzeLiveRanges(plan, external);
    expect(ranges.get(plan[2].id)!.external).toBe(false);
  });

  it("external node with no in-plan consumers gets [start, plan.length-1]", () => {
    // plan[4] has no in-plan consumers but is external (gradient output)
    const plan = chainPlan(5);
    const external = new Set<number>([plan[4].id]);
    const ranges = analyzeLiveRanges(plan, external);
    expect(ranges.get(plan[4].id)).toMatchObject({
      start: 4,
      end: 4,
      external: true,
    });
  });

  it("external node IS extended when already extended by a consumer", () => {
    // external node at position 0, consumed at position 2 → end should be
    // max(2, plan.length-1) = plan.length-1.
    const plan = chainPlan(5);
    const external = new Set<number>([plan[0].id]);
    const ranges = analyzeLiveRanges(plan, external);
    expect(ranges.get(plan[0].id)!.end).toBe(4);
  });
});

describe("analyzeLiveRanges / size", () => {
  it("size reflects node.shape and node.dtype", () => {
    const n = createLazyIRNode("add", [matRef(1), matRef(2)], [8, 8], "f16", "cpu");
    const ranges = analyzeLiveRanges([n]);
    expect(ranges.get(n.id)!.size).toBe(64 * 2);
  });
});

// ============================================================================
// liveRangeStats
// ============================================================================

describe("liveRangeStats", () => {
  it("counts ranges and identifies external", () => {
    const plan = chainPlan(5);
    const external = new Set<number>([plan[4].id]);
    const ranges = analyzeLiveRanges(plan, external);
    const stats = liveRangeStats(ranges);
    expect(stats.count).toBe(5);
    expect(stats.external).toBe(1);
  });

  it("maxConcurrentBytes is at least maxBufferSize", () => {
    const plan = chainPlan(5, [4, 4]); // each 64 bytes
    const ranges = analyzeLiveRanges(plan);
    const stats = liveRangeStats(ranges);
    expect(stats.maxConcurrentBytes).toBeGreaterThanOrEqual(64);
  });

  it("chain plan has at most 2 concurrent tensors at any time (current + prev)", () => {
    const plan = chainPlan(10);
    const ranges = analyzeLiveRanges(plan);
    const stats = liveRangeStats(ranges);
    // In a chain, only the current node and its predecessor are live at any
    // point. Actually in our simple chain, 2 are live at consumer positions.
    expect(stats.maxConcurrentCount).toBeLessThanOrEqual(2);
  });

  it("totalBytes is sum of all range sizes", () => {
    const plan = chainPlan(3, [4, 4]); // 64 bytes each
    const ranges = analyzeLiveRanges(plan);
    const stats = liveRangeStats(ranges);
    expect(stats.totalBytes).toBe(3 * 64);
  });
});
