/**
 * double-transpose rule tests. Verifies transpose(transpose(x, a, b), a, b)
 * collapses to x, and only when the two transposes use the SAME dim pair.
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../../src/graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../../../src/graph/types";
import { applyRules } from "../../../src/compiler/rewriter/engine";
import { doubleTransposeRule } from "../../../src/compiler/rewriter/rules/double-transpose";
import type { ConsumerMaps } from "../../../src/compiler/rewriter/substitute";

// ============================================================================
// Helpers
// ============================================================================

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

function transposeNode(
  input: LazyRef,
  inputShape: number[],
  dim0: number,
  dim1: number,
): LazyIRNode {
  const rank = inputShape.length;
  const d0 = dim0 < 0 ? rank + dim0 : dim0;
  const d1 = dim1 < 0 ? rank + dim1 : dim1;
  const outShape = [...inputShape];
  [outShape[d0], outShape[d1]] = [outShape[d1], outShape[d0]];
  return createLazyIRNode("transpose", [input], outShape, "f32", "cpu", {
    dim0,
    dim1,
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

// ============================================================================
// Matching cases
// ============================================================================

describe("double-transpose / matches", () => {
  it("collapses transpose(transpose(x, 0, 1), 0, 1)", () => {
    const X = matRef(1, [4, 8]);
    const inner = transposeNode(X, [4, 8], 0, 1); // [8, 4]
    const outer = transposeNode(pendingRef(inner), [8, 4], 0, 1); // [4, 8]
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [4, 8],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);

    const stats = applyRules(plan, [doubleTransposeRule], maps);
    expect(stats.applied).toBe(1);
    // final's input should be X directly.
    expect(final.inputs[0]).toBe(X);
  });

  it("collapses with reversed dim pair (transpose is commutative in dims)", () => {
    const X = matRef(1, [4, 8]);
    const inner = transposeNode(X, [4, 8], 1, 0); // [8, 4]
    const outer = transposeNode(pendingRef(inner), [8, 4], 0, 1); // [4, 8]
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [4, 8],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);

    expect(applyRules(plan, [doubleTransposeRule], maps).applied).toBe(1);
    expect(final.inputs[0]).toBe(X);
  });

  it("collapses with negative dims (normalizes)", () => {
    const X = matRef(1, [4, 8, 16]);
    const inner = transposeNode(X, [4, 8, 16], -2, -1); // [4, 16, 8]
    const outer = transposeNode(pendingRef(inner), [4, 16, 8], 1, 2); // [4, 8, 16]
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [4, 8, 16],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);

    expect(applyRules(plan, [doubleTransposeRule], maps).applied).toBe(1);
    expect(final.inputs[0]).toBe(X);
  });
});

// ============================================================================
// Non-matching cases
// ============================================================================

describe("double-transpose / doesn't match", () => {
  it("different dim pairs don't simplify", () => {
    // transpose(transpose(x, 0, 1), 1, 2) — not collapsible
    const X = matRef(1, [4, 8, 16]);
    const inner = transposeNode(X, [4, 8, 16], 0, 1); // [8, 4, 16]
    const outer = transposeNode(pendingRef(inner), [8, 4, 16], 1, 2); // [8, 16, 4]
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [8, 16, 4],
      "f32",
      "cpu",
    );
    const plan = [inner, outer, final];
    const maps = buildMaps(plan);

    expect(applyRules(plan, [doubleTransposeRule], maps).applied).toBe(0);
  });

  it("single transpose (no inner transpose) doesn't match", () => {
    const X = matRef(1, [4, 8]);
    const outer = transposeNode(X, [4, 8], 0, 1);
    const final = createLazyIRNode(
      "relu",
      [pendingRef(outer)],
      [8, 4],
      "f32",
      "cpu",
    );
    const plan = [outer, final];
    const maps = buildMaps(plan);

    expect(applyRules(plan, [doubleTransposeRule], maps).applied).toBe(0);
  });

  it("transpose then some other op doesn't match", () => {
    const X = matRef(1, [4, 8]);
    const inner = transposeNode(X, [4, 8], 0, 1);
    const reshape = createLazyIRNode(
      "reshape",
      [pendingRef(inner)],
      [32],
      "f32",
      "cpu",
      { targetShape: [32] },
    );
    const plan = [inner, reshape];
    const maps = buildMaps(plan);
    expect(applyRules(plan, [doubleTransposeRule], maps).applied).toBe(0);
  });
});

// ============================================================================
// Multiple applications / chaining
// ============================================================================

describe("double-transpose / chaining", () => {
  it("collapses four-deep transposes to the original", () => {
    // Apply twice: transpose(transpose(transpose(transpose(x)))) → x
    const X = matRef(1, [4, 8]);
    const t1 = transposeNode(X, [4, 8], 0, 1);
    const t2 = transposeNode(pendingRef(t1), [8, 4], 0, 1);
    const t3 = transposeNode(pendingRef(t2), [4, 8], 0, 1);
    const t4 = transposeNode(pendingRef(t3), [8, 4], 0, 1);
    const final = createLazyIRNode(
      "relu",
      [pendingRef(t4)],
      [4, 8],
      "f32",
      "cpu",
    );
    const plan = [t1, t2, t3, t4, final];
    const maps = buildMaps(plan);

    const stats = applyRules(plan, [doubleTransposeRule], maps);
    expect(stats.applied).toBe(2);
    expect(final.inputs[0]).toBe(X);
  });
});
