/**
 * transitive-reshape tests.
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../../src/graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../../../src/graph/types";
import { applyRules } from "../../../src/compiler/rewriter/engine";
import { transitiveReshapeRule } from "../../../src/compiler/rewriter/rules/transitive-reshape";
import type { ConsumerMaps } from "../../../src/compiler/rewriter/substitute";

function pendingRef(node: LazyIRNode): LazyRef {
  return { kind: "pending", node };
}

function matRef(id: number, shape: number[]): LazyRef {
  const storage = {
    id,
    backendTensor: { shape, dtype: "f32" },
  } as unknown as StorageHandle;
  return { kind: "materialized", storage };
}

function reshapeNode(input: LazyRef, shape: number[]): LazyIRNode {
  return createLazyIRNode("reshape", [input], shape, "f32", "cpu", {
    targetShape: shape,
  });
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

describe("transitive-reshape", () => {
  it("collapses reshape(reshape(x, s1), s2) to reshape(x, s2)", () => {
    const X = matRef(1, [4, 8]);
    const inner = reshapeNode(X, [32]);
    const outer = reshapeNode(pendingRef(inner), [2, 16]);
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [2, 16],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);

    expect(applyRules(plan, [transitiveReshapeRule], maps).applied).toBe(1);
    // outer now points directly at X; shape unchanged.
    expect(outer.inputs[0]).toBe(X);
    expect(outer.shape).toEqual([2, 16]);
    // final still uses outer.
    expect(final.inputs[0]).toEqual(pendingRef(outer));
  });

  it("preserves outer's target shape payload", () => {
    const X = matRef(1, [4, 8]);
    const inner = reshapeNode(X, [32]);
    const outer = reshapeNode(pendingRef(inner), [2, 16]);
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [2, 16],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);
    applyRules(plan, [transitiveReshapeRule], maps);
    const payload = outer.payload as { targetShape: number[] };
    expect(payload.targetShape).toEqual([2, 16]);
  });

  it("single reshape doesn't match", () => {
    const X = matRef(1, [4, 8]);
    const r = reshapeNode(X, [32]);
    const final = createLazyIRNode("relu", [pendingRef(r)], [32], "f32", "cpu");
    const plan = [r, final];
    const maps = buildMaps(plan);
    expect(applyRules(plan, [transitiveReshapeRule], maps).applied).toBe(0);
  });

  it("chains: reshape∘reshape∘reshape collapses to one", () => {
    const X = matRef(1, [4, 8]);
    const r1 = reshapeNode(X, [32]);
    const r2 = reshapeNode(pendingRef(r1), [2, 16]);
    const r3 = reshapeNode(pendingRef(r2), [16, 2]);
    const final = createLazyIRNode(
      "relu",
      [pendingRef(r3)],
      [16, 2],
      "f32",
      "cpu",
    );
    const plan = [r1, r2, r3, final];
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [transitiveReshapeRule], maps);
    expect(stats.applied).toBe(2);
    // r3 now points directly at X.
    expect(r3.inputs[0]).toBe(X);
  });

  it("if intermediate has multiple consumers, rule still fires but inner survives", () => {
    // outer and `other` both consume inner. Fusing outer→X leaves inner alive.
    const X = matRef(1, [4, 8]);
    const inner = reshapeNode(X, [32]);
    const outer = reshapeNode(pendingRef(inner), [2, 16]);
    const other = createLazyIRNode(
      "relu",
      [pendingRef(inner)],
      [32],
      "f32",
      "cpu",
    );
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [2, 16],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, other, final];
    const maps = buildMaps(plan);
    expect(applyRules(plan, [transitiveReshapeRule], maps).applied).toBe(1);
    // outer now points at X; inner still alive (other consumes it).
    expect(outer.inputs[0]).toBe(X);
    // other still points at inner.
    expect(other.inputs[0]).toEqual(pendingRef(inner));
  });
});
