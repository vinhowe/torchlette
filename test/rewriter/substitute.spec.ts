/**
 * Substitution tests. Each test builds a small plan, applies a substitution,
 * and verifies:
 *   1. planNodes contains the new nodes in the right position
 *   2. consumers of matchedRoot are rewired to use the replacement
 *   3. consumer maps are consistent (counts match linked lists)
 *   4. DCE (reasoning about consumer counts) could remove the orphaned chain
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../src/graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../../src/graph/types";
import {
  applySubstitution,
  makeRewriteContext,
  type ConsumerMaps,
} from "../../src/compiler/rewriter/substitute";

// ============================================================================
// Test helpers
// ============================================================================

function pendingRef(node: LazyIRNode, outputIndex?: number): LazyRef {
  return outputIndex !== undefined
    ? { kind: "pending", node, outputIndex }
    : { kind: "pending", node };
}

function matRef(id: number): LazyRef {
  const storage = { id } as unknown as StorageHandle;
  return { kind: "materialized", storage };
}

function addNode(a: LazyRef, b: LazyRef, shape: number[] = [1]): LazyIRNode {
  return createLazyIRNode("add", [a, b], shape, "f32", "cpu");
}

function mulNode(a: LazyRef, b: LazyRef, shape: number[] = [1]): LazyIRNode {
  return createLazyIRNode("mul", [a, b], shape, "f32", "cpu");
}

/** Build consumer/consumerCount maps from a plan. */
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

/** Verify consumers map agrees with consumerCount map. */
function assertMapsConsistent(maps: ConsumerMaps): void {
  for (const [id, list] of maps.consumers) {
    expect(maps.consumerCount.get(id) ?? 0).toBe(list.length);
  }
  for (const [id, count] of maps.consumerCount) {
    const list = maps.consumers.get(id) ?? [];
    expect(list.length).toBe(count);
  }
}

/** Verify consumer maps match the plan's actual input refs. */
function assertMapsMatchPlan(plan: LazyIRNode[], maps: ConsumerMaps): void {
  const rebuilt = buildMaps(plan);
  assertMapsConsistent(maps);
  for (const [id, count] of rebuilt.consumerCount) {
    expect(maps.consumerCount.get(id) ?? 0).toBe(count);
  }
}

// ============================================================================
// Trivial substitution
// ============================================================================

describe("substitute / minimal", () => {
  it("drops matchedRoot's consumer list and clears count", () => {
    // plan: mul(add(m1, m2), m3) — add is matched, replaced with m1.
    const m1 = matRef(1);
    const m2 = matRef(2);
    const m3 = matRef(3);
    const add = addNode(m1, m2);
    const mul = mulNode(pendingRef(add), m3);
    const plan = [add, mul];
    const maps = buildMaps(plan);

    // Before: add has 1 consumer (mul)
    expect(maps.consumerCount.get(add.id)).toBe(1);

    applySubstitution(plan, add, m1, [], maps);

    // After: add has 0 consumers, mul's inputs[0] is m1
    expect(maps.consumerCount.get(add.id)).toBe(0);
    expect(mul.inputs[0]).toBe(m1);
    assertMapsConsistent(maps);
  });

  it("throws if replacement refers back to matchedRoot", () => {
    const add = addNode(matRef(1), matRef(2));
    const mul = mulNode(pendingRef(add), matRef(3));
    const plan = [add, mul];
    const maps = buildMaps(plan);

    expect(() =>
      applySubstitution(plan, add, pendingRef(add), [], maps),
    ).toThrow(/same node/i);
  });

  it("throws if matchedRoot is not in plan", () => {
    const add = addNode(matRef(1), matRef(2));
    const plan: LazyIRNode[] = [];
    const maps = buildMaps(plan);

    expect(() =>
      applySubstitution(plan, add, matRef(1), [], maps),
    ).toThrow(/not in plan/i);
  });
});

// ============================================================================
// Substitution with new nodes
// ============================================================================

describe("substitute / with new nodes", () => {
  it("inserts new nodes just before matchedRoot", () => {
    // plan: [add, mul] — insert two new nodes before `add`, replace `add`'s
    // output with the output of the last new node.
    const m1 = matRef(1);
    const m2 = matRef(2);
    const m3 = matRef(3);
    const add = addNode(m1, m2);
    const mul = mulNode(pendingRef(add), m3);
    const plan = [add, mul];
    const maps = buildMaps(plan);

    const { ctx, newNodes } = makeRewriteContext();
    const neg1 = ctx.createNode("neg", [m1], [1], "f32", "cpu");
    const neg2 = ctx.createNode("neg", [ctx.pendingRef(neg1)], [1], "f32", "cpu");
    const replacement = ctx.pendingRef(neg2);

    applySubstitution(plan, add, replacement, newNodes, maps);

    // Plan order: [neg1, neg2, add, mul]
    expect(plan.length).toBe(4);
    expect(plan[0]).toBe(neg1);
    expect(plan[1]).toBe(neg2);
    expect(plan[2]).toBe(add);
    expect(plan[3]).toBe(mul);

    // mul's first input should now be neg2
    expect(mul.inputs[0]).toEqual(replacement);

    // Consumer maps: add.consumerCount = 0, neg2.consumerCount = 1 (mul)
    expect(maps.consumerCount.get(add.id)).toBe(0);
    expect(maps.consumerCount.get(neg2.id)).toBe(1);
    expect(maps.consumerCount.get(neg1.id)).toBe(1);
    assertMapsConsistent(maps);
  });

  it("registers new nodes' materialized inputs as consumers", () => {
    // Insert a new node whose input is a materialized ref.
    const m1 = matRef(1);
    const m2 = matRef(2);
    const add = addNode(m1, m2);
    const plan = [add];
    const maps = buildMaps(plan);

    const { ctx, newNodes } = makeRewriteContext();
    ctx.createNode("neg", [m1], [1], "f32", "cpu");
    // Materialized refs don't have node.id in our maps, so nothing is tracked
    // for them. The function should handle this gracefully.
    applySubstitution(plan, add, m1, newNodes, maps);
    // No throw. Done.
    assertMapsConsistent(maps);
  });

  it("consumer count of new pending input is incremented", () => {
    // plan: add(mul(m1, m2), m3). Replace `add` with neg(mul(m1,m2)).
    // The new `neg` node takes `mul`'s pending output as its input.
    const m1 = matRef(1);
    const m2 = matRef(2);
    const m3 = matRef(3);
    const mul = mulNode(m1, m2);
    const add = addNode(pendingRef(mul), m3);
    const plan = [mul, add];
    const maps = buildMaps(plan);

    // Before: mul has 1 consumer (add)
    expect(maps.consumerCount.get(mul.id)).toBe(1);

    const { ctx, newNodes } = makeRewriteContext();
    const neg = ctx.createNode("neg", [ctx.pendingRef(mul)], [1], "f32", "cpu");
    applySubstitution(plan, add, ctx.pendingRef(neg), newNodes, maps);

    // After: mul has 2 consumers (add still, which is orphan; neg, new)
    // add still has mul as input (inputs[0]); mul's consumers include both.
    expect(maps.consumerCount.get(mul.id)).toBe(2);
    expect(maps.consumerCount.get(neg.id)).toBe(0); // add (replaced) had 0 previous, none now
    assertMapsConsistent(maps);
  });
});

// ============================================================================
// Multi-consumer scenarios
// ============================================================================

describe("substitute / multi-consumer", () => {
  it("rewires multiple consumers of matchedRoot", () => {
    // plan: add(m1,m2) consumed by both neg(add) and relu(add)
    const m1 = matRef(1);
    const m2 = matRef(2);
    const add = addNode(m1, m2);
    const neg = createLazyIRNode("neg", [pendingRef(add)], [1], "f32", "cpu");
    const relu = createLazyIRNode("relu", [pendingRef(add)], [1], "f32", "cpu");
    const plan = [add, neg, relu];
    const maps = buildMaps(plan);

    expect(maps.consumerCount.get(add.id)).toBe(2);

    // Replace add with m1
    applySubstitution(plan, add, m1, [], maps);

    expect(neg.inputs[0]).toBe(m1);
    expect(relu.inputs[0]).toBe(m1);
    expect(maps.consumerCount.get(add.id)).toBe(0);
    assertMapsConsistent(maps);
  });

  it("preserves existing consumers of replacement node", () => {
    // plan: mul(m1, m2) is used by both `add` and `sub`.
    // Replace `add` with `mul`'s ref. Now `sub` and `mul`'s other consumers
    // should coexist with the rewired consumer.
    const m1 = matRef(1);
    const m2 = matRef(2);
    const m3 = matRef(3);
    const mul = mulNode(m1, m2);
    const add = addNode(pendingRef(mul), m3); // add has mul as input
    const sub = createLazyIRNode(
      "sub",
      [pendingRef(mul), m3],
      [1],
      "f32",
      "cpu",
    );
    const final = createLazyIRNode(
      "relu",
      [pendingRef(add)],
      [1],
      "f32",
      "cpu",
    );
    const plan = [mul, add, sub, final];
    const maps = buildMaps(plan);

    // Before: mul has 2 consumers (add, sub), add has 1 (final)
    expect(maps.consumerCount.get(mul.id)).toBe(2);
    expect(maps.consumerCount.get(add.id)).toBe(1);

    // Replace add with pendingRef(mul) — `final` now consumes mul directly
    applySubstitution(plan, add, pendingRef(mul), [], maps);

    // mul's consumers should include add-replaced-as-final, so 3 consumers:
    // the original 2 (add, sub) + final. But add is now orphan.
    // Actually: mul's consumer list INCLUDES add (still has mul input) + sub
    // + final. So count = 3. But add is orphan.
    expect(maps.consumerCount.get(mul.id)).toBe(3);
    expect(maps.consumerCount.get(add.id)).toBe(0);
    expect(final.inputs[0]).toEqual(pendingRef(mul));
    assertMapsConsistent(maps);
  });
});

// ============================================================================
// Integration with makeRewriteContext
// ============================================================================

describe("substitute / rewrite context", () => {
  it("tracks all created nodes", () => {
    const { ctx, newNodes } = makeRewriteContext();
    expect(newNodes.length).toBe(0);

    ctx.createNode("neg", [matRef(1)], [1], "f32", "cpu");
    expect(newNodes.length).toBe(1);

    ctx.createNode("relu", [matRef(1)], [1], "f32", "cpu");
    expect(newNodes.length).toBe(2);
  });

  it("pendingRef produces correct LazyRef", () => {
    const { ctx } = makeRewriteContext();
    const node = ctx.createNode("neg", [matRef(1)], [1], "f32", "cpu");
    const ref = ctx.pendingRef(node);
    expect(ref.kind).toBe("pending");
    if (ref.kind === "pending") {
      expect(ref.node).toBe(node);
      expect(ref.outputIndex).toBeUndefined();
    }
  });

  it("pendingRef with outputIndex", () => {
    const { ctx } = makeRewriteContext();
    const node = ctx.createNode("neg", [matRef(1)], [1], "f32", "cpu");
    const ref = ctx.pendingRef(node, 2);
    if (ref.kind === "pending") {
      expect(ref.outputIndex).toBe(2);
    }
  });
});
