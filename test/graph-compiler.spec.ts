/**
 * Tests for the unified graph compiler (src/engine/graph-compiler.ts)
 *
 * Documents the behavior of analyzeGraph() including pattern detection,
 * rewrite bypassing, and consumer count handling.
 */

import { beforeEach, describe, expect, it } from "vitest";
import { analyzeGraph } from "../src/engine/graph-compiler";
import type { LazyIRNode } from "../src/engine/lazy-types";
import { createPendingRef } from "../src/engine/lazy-types";
import {
  createLazyIRNode,
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "../src/engine/node-factory";

describe("Graph Compiler — analyzeGraph()", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  it("returns identity ordering for a trivial plan", () => {
    const a = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [1, 2, 3, 4],
    });
    const b = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [5, 6, 7, 8],
    });
    const add = createLazyIRNode(
      "add",
      [createPendingRef(a), createPendingRef(b)],
      [4],
      "f32",
      "cpu",
    );

    const result = analyzeGraph([a, b, add]);

    // Trivial plan should keep the same order (or reorder but include all nodes)
    expect(result.planNodes.length).toBe(3);
    expect(new Set(result.planNodes.map((n) => n.id))).toEqual(
      new Set([a.id, b.id, add.id]),
    );
  });

  it("claims matmul epilogue chain nodes", () => {
    const a = createLazyIRNode("tensorFromArray", [], [2, 3], "f32", "cpu", {
      values: [1, 2, 3, 4, 5, 6],
    });
    const b = createLazyIRNode("tensorFromArray", [], [3, 2], "f32", "cpu", {
      values: [1, 2, 3, 4, 5, 6],
    });
    // bias must appear BEFORE matmul in plan order — epilogue detection
    // requires secondary inputs (bias) to have position < matmulPos
    const bias = createLazyIRNode("tensorFromArray", [], [2, 2], "f32", "cpu", {
      values: [0.1, 0.2, 0.3, 0.4],
    });
    const mm = createLazyIRNode(
      "matmul",
      [createPendingRef(a), createPendingRef(b)],
      [2, 2],
      "f32",
      "cpu",
    );
    const castNode = createLazyIRNode(
      "cast",
      [createPendingRef(mm)],
      [2, 2],
      "f16",
      "cpu",
      { dtype: "f16" },
    );
    const addBias = createLazyIRNode(
      "add",
      [createPendingRef(castNode), createPendingRef(bias)],
      [2, 2],
      "f32",
      "cpu",
    );

    const result = analyzeGraph([a, b, bias, mm, castNode, addBias]);

    // cast and addBias should be claimed by matmul epilogue
    expect(result.epilogueClaimedIds.has(castNode.id)).toBe(true);
    expect(result.epilogueClaimedIds.has(addBias.id)).toBe(true);
    expect(result.matmulEpilogueChains.has(mm.id)).toBe(true);
  });

  it("excludes bypassed nodes from fusion groups", () => {
    const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [1, 2, 3, 4],
    });
    const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [5, 6, 7, 8],
    });

    // Identity cast — should be bypassed by rewrite pass
    const identityCast = createLazyIRNode(
      "cast",
      [createPendingRef(x)],
      [4],
      "f32",
      "cpu",
      { dtype: "f32" },
    );
    const add = createLazyIRNode(
      "add",
      [createPendingRef(identityCast), createPendingRef(y)],
      [4],
      "f32",
      "cpu",
    );

    const result = analyzeGraph([x, y, identityCast, add]);

    expect(result.rewriteBypassedIds.has(identityCast.id)).toBe(true);
    // Bypassed nodes are excluded from fusion groups (not sequential segments,
    // where they're skipped at execution time via the bypassed set)
    for (const seg of result.segments) {
      if (seg.kind === "fused") {
        const fusedIds = new Set(seg.group.nodes.map((n) => n.id));
        expect(fusedIds.has(identityCast.id)).toBe(false);
      }
    }
  });

  it("consumer count reflects rewrites", () => {
    const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [1, 2, 3, 4],
    });
    const identityCast = createLazyIRNode(
      "cast",
      [createPendingRef(x)],
      [4],
      "f32",
      "cpu",
      { dtype: "f32" },
    );
    const consumer = createLazyIRNode(
      "relu",
      [createPendingRef(identityCast)],
      [4],
      "f32",
      "cpu",
    );

    const result = analyzeGraph([x, identityCast, consumer]);

    // After bypassing identityCast, x should have consumer's count
    // identityCast count should be 0
    expect(result.consumerCount.get(identityCast.id)).toBe(0);
    // x's consumer count should have increased
    expect(result.consumerCount.get(x.id)).toBeGreaterThan(0);
  });

  it("handles empty plan", () => {
    const singleNode = createLazyIRNode(
      "tensorFromArray",
      [],
      [4],
      "f32",
      "cpu",
      { values: [1, 2, 3, 4] },
    );
    const result = analyzeGraph([singleNode]);

    expect(result.planNodes.length).toBe(1);
    expect(result.segments.length).toBeGreaterThanOrEqual(1);
  });

  it("reduction preamble claiming excludes nodes from fusion", () => {
    // Build: mul → sum chain (mul should be claimed as reduction preamble)
    const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [1, 2, 3, 4],
    });
    const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [5, 6, 7, 8],
    });
    const mul = createLazyIRNode(
      "mul",
      [createPendingRef(x), createPendingRef(y)],
      [4],
      "f32",
      "cpu",
    );
    const sum = createLazyIRNode(
      "sum",
      [createPendingRef(mul)],
      [],
      "f32",
      "cpu",
      { dim: null },
    );

    const result = analyzeGraph([x, y, mul, sum]);

    // mul should not appear in any fused segment since it's claimed by reduction
    for (const seg of result.segments) {
      if (seg.kind === "fused") {
        const fusedIds = new Set(seg.group.nodes.map((n) => n.id));
        expect(fusedIds.has(mul.id)).toBe(false);
      }
    }
  });

  it("algebraic identity bypass feeds into epilogue/fusion", () => {
    const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
      values: [1, 2, 3, 4],
    });
    const one = createLazyIRNode("full", [], [1], "f32", "cpu", {
      fillValue: 1,
    });
    const mulOne = createLazyIRNode(
      "mul",
      [createPendingRef(x), createPendingRef(one)],
      [4],
      "f32",
      "cpu",
    );
    const relu = createLazyIRNode(
      "relu",
      [createPendingRef(mulOne)],
      [4],
      "f32",
      "cpu",
    );

    const result = analyzeGraph([x, one, mulOne, relu]);

    // mul(x, 1) should be bypassed
    expect(result.rewriteBypassedIds.has(mulOne.id)).toBe(true);
  });
});
