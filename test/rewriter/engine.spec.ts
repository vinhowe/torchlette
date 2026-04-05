/**
 * Rewrite engine tests. Verify rule application, fixed-point iteration,
 * rule priority, and correct handling of inserted/orphaned nodes.
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../src/graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../../src/graph/types";
import { applyRules, type Rule } from "../../src/compiler/rewriter/engine";
import { capture, op } from "../../src/compiler/rewriter/pattern";
import type { ConsumerMaps } from "../../src/compiler/rewriter/substitute";

// ============================================================================
// Test helpers
// ============================================================================

function pendingRef(node: LazyIRNode): LazyRef {
  return { kind: "pending", node };
}

function matRef(id: number): LazyRef {
  const storage = { id } as unknown as StorageHandle;
  return { kind: "materialized", storage };
}

function addNode(a: LazyRef, b: LazyRef): LazyIRNode {
  return createLazyIRNode("add", [a, b], [1], "f32", "cpu");
}

function mulNode(a: LazyRef, b: LazyRef): LazyIRNode {
  return createLazyIRNode("mul", [a, b], [1], "f32", "cpu");
}

function negNode(a: LazyRef): LazyIRNode {
  return createLazyIRNode("neg", [a], [1], "f32", "cpu");
}

function buildMaps(plan: LazyIRNode[]): ConsumerMaps {
  const consumers = new Map<number, LazyIRNode[]>();
  const consumerCount = new Map<number, number>();
  for (const node of plan) {
    for (const ref of node.inputs) {
      if (ref.kind === "pending") {
        const id = ref.node.id;
        const list = consumers.get(id) ?? [];
        list.push(node);
        consumers.set(id, list);
        consumerCount.set(id, (consumerCount.get(id) ?? 0) + 1);
      }
    }
  }
  return { consumers, consumerCount };
}

function assertMapsConsistent(maps: ConsumerMaps): void {
  for (const [id, list] of maps.consumers) {
    expect(maps.consumerCount.get(id) ?? 0).toBe(list.length);
  }
}

// ============================================================================
// Trivial rules
// ============================================================================

describe("engine / trivial rules", () => {
  it("no-op when no rules match", () => {
    const plan = [addNode(matRef(1), matRef(2))];
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [], maps);
    expect(stats.applied).toBe(0);
    expect(plan.length).toBe(1);
  });

  it("applies a simple neg(neg(x)) → x rule", () => {
    // Plan: final = neg(neg(m1))
    const m1 = matRef(1);
    const inner = negNode(m1);
    const outer = negNode(pendingRef(inner));
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);

    const rule: Rule = {
      name: "double-neg",
      pattern: op("neg", { inputs: [op("neg", { inputs: [capture("X")] })] }),
      rewrite: (bindings) => bindings.get("X")!,
    };

    const stats = applyRules(plan, [rule], maps);
    expect(stats.applied).toBe(1);
    expect(stats.byRule.get("double-neg")).toBe(1);
    // relu's input is now m1 directly
    expect(final.inputs[0]).toBe(m1);
    assertMapsConsistent(maps);
  });

  it("does nothing when pattern doesn't match", () => {
    const plan = [addNode(matRef(1), matRef(2))];
    const maps = buildMaps(plan);
    const rule: Rule = {
      name: "no-match",
      pattern: op("mul"),
      rewrite: () => matRef(999),
    };
    const stats = applyRules(plan, [rule], maps);
    expect(stats.applied).toBe(0);
  });
});

// ============================================================================
// Fixed-point iteration
// ============================================================================

describe("engine / fixed-point", () => {
  it("iterates when one rule enables another", () => {
    // Plan: final = neg(neg(neg(neg(m1))))
    // After 1 rewrite: final = neg(neg(m1))
    // After 2 rewrites: final = m1
    const m1 = matRef(1);
    const n1 = negNode(m1);
    const n2 = negNode(pendingRef(n1));
    const n3 = negNode(pendingRef(n2));
    const n4 = negNode(pendingRef(n3));
    const final = createLazyIRNode(
      "relu",
      [pendingRef(n4)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [n1, n2, n3, n4, final];
    const maps = buildMaps(plan);

    const rule: Rule = {
      name: "double-neg",
      pattern: op("neg", { inputs: [op("neg", { inputs: [capture("X")] })] }),
      rewrite: (bindings) => bindings.get("X")!,
    };

    const stats = applyRules(plan, [rule], maps);
    expect(stats.applied).toBe(2);
    expect(stats.passes).toBeGreaterThanOrEqual(2);
    expect(final.inputs[0]).toBe(m1);
  });

  it("stops at fixed point", () => {
    // Plan: add(m1, m2) — no rule fires, should pass once and stop.
    const plan = [addNode(matRef(1), matRef(2))];
    const maps = buildMaps(plan);
    const rule: Rule = {
      name: "no-match",
      pattern: op("mul"),
      rewrite: () => matRef(999),
    };
    const stats = applyRules(plan, [rule], maps);
    expect(stats.passes).toBe(1);
  });

  it("respects maxPasses", () => {
    // Non-terminating rule would loop forever: every rewrite creates a new
    // match. maxPasses caps it.
    // Rule: x → x (but we're not allowed to return same node, so…)
    // We use a rule that matches neg and replaces it with neg(neg(_)), which
    // the next pass could replace with _. But that's what double-neg already
    // does. Instead: rule that wraps any add in neg(neg(add)).
    const m1 = matRef(1);
    const add = addNode(m1, matRef(2));
    const final = createLazyIRNode(
      "relu",
      [pendingRef(add)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [add, final];
    const maps = buildMaps(plan);

    // Pathological rule that keeps adding wrapping. Use maxPasses to cap.
    const rule: Rule = {
      name: "wrap-neg",
      pattern: op("add", { inputs: [capture("A"), capture("B")] }),
      rewrite: (bindings, ctx) => {
        const A = bindings.get("A")!;
        const B = bindings.get("B")!;
        const newAdd = ctx.createNode("add", [A, B], [1], "f32", "cpu");
        const inner = ctx.createNode(
          "neg",
          [ctx.pendingRef(newAdd)],
          [1],
          "f32",
          "cpu",
        );
        return ctx.pendingRef(inner);
      },
    };

    const stats = applyRules(plan, [rule], maps, { maxPasses: 3 });
    expect(stats.passes).toBe(3);
    // The rule doesn't actually loop because it creates new `add` nodes
    // that DO match the pattern, so it keeps wrapping.
    expect(stats.applied).toBeGreaterThanOrEqual(3);
  });
});

// ============================================================================
// Rule priority (ordering)
// ============================================================================

describe("engine / rule priority", () => {
  it("first rule in list wins when both match", () => {
    const m1 = matRef(1);
    const n = negNode(m1);
    const final = createLazyIRNode(
      "relu",
      [pendingRef(n)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [n, final];
    const maps = buildMaps(plan);

    let firstFired = false;
    let secondFired = false;
    const rule1: Rule = {
      name: "first",
      pattern: op("neg"),
      rewrite: (bindings, ctx) => {
        firstFired = true;
        return matRef(1); // replace neg with m1
      },
    };
    const rule2: Rule = {
      name: "second",
      pattern: op("neg"),
      rewrite: () => {
        secondFired = true;
        return matRef(2);
      },
    };

    applyRules(plan, [rule1, rule2], maps);
    expect(firstFired).toBe(true);
    expect(secondFired).toBe(false);
  });
});

// ============================================================================
// Cross-capture constraints via `check`
// ============================================================================

describe("engine / check predicate", () => {
  it("rejects match when check returns false", () => {
    const plan = [addNode(matRef(1), matRef(2))];
    const maps = buildMaps(plan);

    const rule: Rule = {
      name: "conditional",
      pattern: op("add", { inputs: [capture("A"), capture("B")] }),
      rewrite: (bindings) => bindings.get("A")!,
      check: () => false, // always reject
    };

    const stats = applyRules(plan, [rule], maps);
    expect(stats.applied).toBe(0);
  });

  it("accepts match when check returns true", () => {
    const m1 = matRef(1);
    const n = addNode(m1, matRef(2));
    const final = createLazyIRNode(
      "relu",
      [pendingRef(n)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [n, final];
    const maps = buildMaps(plan);

    const rule: Rule = {
      name: "conditional",
      pattern: op("add", { inputs: [capture("A"), capture("B")] }),
      rewrite: (bindings) => bindings.get("A")!,
      check: (bindings) => bindings.get("A")!.kind === "materialized",
    };

    const stats = applyRules(plan, [rule], maps);
    expect(stats.applied).toBe(1);
    expect(final.inputs[0]).toBe(m1);
  });
});

// ============================================================================
// Infinite-loop guard
// ============================================================================

describe("engine / loop guard", () => {
  it("skips rule that returns the matched root", () => {
    const n = negNode(matRef(1));
    const final = createLazyIRNode(
      "relu",
      [pendingRef(n)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [n, final];
    const maps = buildMaps(plan);

    const rule: Rule = {
      name: "self-return",
      pattern: op("neg"),
      rewrite: (_bindings, ctx) => ctx.pendingRef(n), // same node!
    };

    const stats = applyRules(plan, [rule], maps);
    expect(stats.applied).toBe(0);
  });
});
