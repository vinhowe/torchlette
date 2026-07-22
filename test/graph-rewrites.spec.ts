/**
 * Tests for graph rewrite passes (src/engine/graph-rewrites.ts)
 *
 * Documents current behavior of simplification passes and new CSE/DCE passes.
 */

import { beforeEach, describe, expect, it } from "vitest";
import type { DType } from "../src/backend/types";
import type { RewriteContext } from "../src/compiler/graph-rewrites";
import {
  eliminateAlgebraicIdentities,
  eliminateCommonSubexpressions,
  eliminateDeadCode,
  eliminateIdentityCasts,
  eliminateRedundantContiguous,
  runPasses,
  SIMPLIFICATION_PASSES,
} from "../src/compiler/graph-rewrites";
import type {
  LazyIRNode,
  LazyRef,
  StorageHandle,
} from "../src/graph/types";
import { createPendingRef } from "../src/graph/types";
import {
  createLazyIRNode,
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "../src/graph/node-factory";

/** Build a RewriteContext from plan nodes. */
function buildContext(planNodes: LazyIRNode[]): RewriteContext {
  const consumers = new Map<number, LazyIRNode[]>();
  const consumerCount = new Map<number, number>();
  for (const node of planNodes) {
    for (const input of node.inputs) {
      if (input.kind === "pending") {
        const producerId = input.node.id;
        consumerCount.set(producerId, (consumerCount.get(producerId) ?? 0) + 1);
        if (!consumers.has(producerId)) consumers.set(producerId, []);
        consumers.get(producerId)!.push(node);
      }
    }
  }
  return { planNodes, consumers, consumerCount };
}

/** Create a materialized ref with a fake storage handle. */
function materializedRef(dtype: DType = "f32"): LazyRef {
  return {
    kind: "materialized",
    storage: {
      id: 999,
      device: "cpu",
      backendTensor: { dtype, shape: [4], buffer: null } as any,
    },
  };
}

describe("Graph Rewrites", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  // ========================================================================
  // eliminateIdentityCasts
  // ========================================================================
  describe("eliminateIdentityCasts", () => {
    it("bypasses cast(f32→f32)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
        { dtype: "f32" },
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, castNode, consumer]);
      const bypassed = new Set<number>();
      eliminateIdentityCasts(ctx, bypassed);

      expect(bypassed.has(castNode.id)).toBe(true);
      // Consumer should now reference x directly
      expect(consumer.inputs[0]).toEqual(createPendingRef(x));
    });

    it("keeps cast(f32→f16)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f16",
        "cpu",
        { dtype: "f16" },
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f16",
        "cpu",
      );

      const ctx = buildContext([x, castNode, consumer]);
      const bypassed = new Set<number>();
      eliminateIdentityCasts(ctx, bypassed);

      expect(bypassed.has(castNode.id)).toBe(false);
      expect(consumer.inputs[0]).toEqual(createPendingRef(castNode));
    });

    it("handles materialized input matching target dtype", () => {
      const matRef = materializedRef("f32");
      const castNode = createLazyIRNode("cast", [matRef], [4], "f32", "cpu", {
        dtype: "f32",
      });
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([castNode, consumer]);
      const bypassed = new Set<number>();
      eliminateIdentityCasts(ctx, bypassed);

      expect(bypassed.has(castNode.id)).toBe(true);
      expect(consumer.inputs[0]).toEqual(matRef);
    });

    it("skips cast nodes with existing result", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
        { dtype: "f32" },
      );
      castNode.result = {} as StorageHandle; // Already executed
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, castNode, consumer]);
      const bypassed = new Set<number>();
      eliminateIdentityCasts(ctx, bypassed);

      expect(bypassed.has(castNode.id)).toBe(false);
    });

    it("updates consumer counts correctly", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
        { dtype: "f32" },
      );
      const c1 = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );
      const c2 = createLazyIRNode(
        "neg",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, castNode, c1, c2]);
      const bypassed = new Set<number>();
      eliminateIdentityCasts(ctx, bypassed);

      expect(bypassed.has(castNode.id)).toBe(true);
      // x now has 2 consumers (c1 and c2), previously had 1 (castNode)
      // After redirect: old count of x was 1 (castNode), castNode had 2 consumers
      // New count: 1 + 2 - 1 = 2
      expect(ctx.consumerCount.get(x.id)).toBe(2);
      expect(ctx.consumerCount.get(castNode.id)).toBe(0);
    });
  });

  // ========================================================================
  // eliminateRedundantContiguous
  // ========================================================================
  describe("eliminateRedundantContiguous", () => {
    it("bypasses contiguous(add_result) — compute ops produce contiguous output", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const contig = createLazyIRNode(
        "contiguous",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(contig)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, contig, consumer]);
      const bypassed = new Set<number>();
      eliminateRedundantContiguous(ctx, bypassed);

      expect(bypassed.has(contig.id)).toBe(true);
      expect(consumer.inputs[0]).toEqual(createPendingRef(x));
    });

    it("keeps contiguous(transpose_result) — view ops may be non-contiguous", () => {
      const x = createLazyIRNode("transpose", [], [4], "f32", "cpu");
      const contig = createLazyIRNode(
        "contiguous",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(contig)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, contig, consumer]);
      const bypassed = new Set<number>();
      eliminateRedundantContiguous(ctx, bypassed);

      expect(bypassed.has(contig.id)).toBe(false);
    });

    it("keeps contiguous(reshape_result)", () => {
      const x = createLazyIRNode("reshape", [], [4], "f32", "cpu");
      const contig = createLazyIRNode(
        "contiguous",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, contig]);
      const bypassed = new Set<number>();
      eliminateRedundantContiguous(ctx, bypassed);

      expect(bypassed.has(contig.id)).toBe(false);
    });

    it("skips contiguous with materialized input", () => {
      const matRef = materializedRef("f32");
      const contig = createLazyIRNode(
        "contiguous",
        [matRef],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([contig]);
      const bypassed = new Set<number>();
      eliminateRedundantContiguous(ctx, bypassed);

      // Materialized inputs are not pending, so contiguous is not bypassed
      expect(bypassed.has(contig.id)).toBe(false);
    });
  });

  // ========================================================================
  // eliminateAlgebraicIdentities
  // ========================================================================
  describe("eliminateAlgebraicIdentities", () => {
    it("bypasses mul(x, 1)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const one = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 1,
      });
      const mulNode = createLazyIRNode(
        "mul",
        [createPendingRef(x), createPendingRef(one)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(mulNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, one, mulNode, consumer]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(mulNode.id)).toBe(true);
      expect(consumer.inputs[0]).toEqual(createPendingRef(x));
    });

    it("bypasses 1 * x (commutative)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const one = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 1,
      });
      const mulNode = createLazyIRNode(
        "mul",
        [createPendingRef(one), createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(mulNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, one, mulNode, consumer]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(mulNode.id)).toBe(true);
      expect(consumer.inputs[0]).toEqual(createPendingRef(x));
    });

    it("bypasses add(x, 0)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const zero = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 0,
      });
      const addNode = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(zero)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(addNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, zero, addNode, consumer]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(addNode.id)).toBe(true);
    });

    it("bypasses sub(x, 0)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const zero = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 0,
      });
      const subNode = createLazyIRNode(
        "sub",
        [createPendingRef(x), createPendingRef(zero)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(subNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, zero, subNode, consumer]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(subNode.id)).toBe(true);
    });

    it("does NOT bypass sub(0, x) — sub is non-commutative", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const zero = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 0,
      });
      const subNode = createLazyIRNode(
        "sub",
        [createPendingRef(zero), createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, zero, subNode]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(subNode.id)).toBe(false);
    });

    it("bypasses div(x, 1)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const one = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 1,
      });
      const divNode = createLazyIRNode(
        "div",
        [createPendingRef(x), createPendingRef(one)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(divNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, one, divNode, consumer]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(divNode.id)).toBe(true);
    });

    it("does NOT bypass div(1, x)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const one = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 1,
      });
      const divNode = createLazyIRNode(
        "div",
        [createPendingRef(one), createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, one, divNode]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(divNode.id)).toBe(false);
    });

    it("does not bypass mul(x, 2)", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const two = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 2,
      });
      const mulNode = createLazyIRNode(
        "mul",
        [createPendingRef(x), createPendingRef(two)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, two, mulNode]);
      const bypassed = new Set<number>();
      eliminateAlgebraicIdentities(ctx, bypassed);

      expect(bypassed.has(mulNode.id)).toBe(false);
    });
  });

  // ========================================================================
  // redirectConsumers (tested indirectly through passes)
  // ========================================================================
  describe("redirectConsumers (multi-consumer chains)", () => {
    it("redirects all consumers when a node is bypassed", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
        { dtype: "f32" },
      );
      const c1 = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );
      const c2 = createLazyIRNode(
        "neg",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );
      const c3 = createLazyIRNode(
        "abs",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, castNode, c1, c2, c3]);
      const bypassed = new Set<number>();
      eliminateIdentityCasts(ctx, bypassed);

      // All three consumers should now reference x
      for (const consumer of [c1, c2, c3]) {
        const ref = consumer.inputs[0];
        expect(ref.kind).toBe("pending");
        if (ref.kind === "pending") {
          expect(ref.node.id).toBe(x.id);
        }
      }
    });
  });

  // ========================================================================
  // CSE Pass
  // ========================================================================
  describe("eliminateCommonSubexpressions", () => {
    it("eliminates duplicate add(x, y)", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });

      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "mul",
        [createPendingRef(add1), createPendingRef(add2)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, add1, add2, consumer]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(1);
      expect(bypassed.has(add2.id)).toBe(true);
      // Consumer's second input should now reference add1
      const ref = consumer.inputs[1];
      expect(ref.kind).toBe("pending");
      if (ref.kind === "pending") {
        expect(ref.node.id).toBe(add1.id);
      }
    });

    it("does NOT eliminate add(x, y) vs add(y, x) — input order matters", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });

      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(y), createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, add1, add2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("never eliminates RNG ops", () => {
      const r1 = createLazyIRNode("rand", [], [4], "f32", "cpu");
      const r2 = createLazyIRNode("rand", [], [4], "f32", "cpu");

      const ctx = buildContext([r1, r2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("never eliminates side-output ops (optStep, fused kernels)", () => {
      const a1 = createLazyIRNode("optStep", [], [4], "f32", "cpu");
      const a2 = createLazyIRNode("optStep", [], [4], "f32", "cpu");

      const ctx = buildContext([a1, a2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("CSE after algebraic identity elimination finds more opportunities", () => {
      // mul(x, 1) → x  and  mul(y, 1) → y expose add(x, y) duplicates
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });
      const one = createLazyIRNode("full", [], [1], "f32", "cpu", {
        fillValue: 1,
      });

      // a = mul(x, 1) → x
      const mulA = createLazyIRNode(
        "mul",
        [createPendingRef(x), createPendingRef(one)],
        [4],
        "f32",
        "cpu",
      );
      // b = add(a, y) where a=x after algebraic → add(x, y)
      const addA = createLazyIRNode(
        "add",
        [createPendingRef(mulA), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      // c = add(x, y) — same as addA after mulA is bypassed
      const addB = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );

      const consumer = createLazyIRNode(
        "mul",
        [createPendingRef(addA), createPendingRef(addB)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, one, mulA, addA, addB, consumer]);
      const bypassed = new Set<number>();

      // First run algebraic identities (mulA → x)
      eliminateAlgebraicIdentities(ctx, bypassed);
      expect(bypassed.has(mulA.id)).toBe(true);

      // Now CSE should find addA and addB identical (both are add(x, y))
      const count = eliminateCommonSubexpressions(ctx, bypassed);
      expect(count).toBe(1);
      expect(bypassed.has(addB.id)).toBe(true);
    });

    it("handles nodes with matching payload (cast same dtype)", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const cast1 = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f16",
        "cpu",
        { dtype: "f16" },
      );
      const cast2 = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f16",
        "cpu",
        { dtype: "f16" },
      );

      const ctx = buildContext([x, cast1, cast2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(1);
      expect(bypassed.has(cast2.id)).toBe(true);
    });

    it("does NOT eliminate nodes with different payloads", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const cast1 = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f16",
        "cpu",
        { dtype: "f16" },
      );
      const cast2 = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "i32",
        "cpu",
        { dtype: "i32" },
      );

      const ctx = buildContext([x, cast1, cast2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("does NOT eliminate nodes with different shapes", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });

      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [2, 2],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, add1, add2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("skips already-bypassed nodes", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });

      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, add1, add2]);
      const bypassed = new Set<number>();
      bypassed.add(add1.id); // Pre-bypass add1
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      // add1 is already bypassed, so add2 can't match against it — no elimination
      expect(count).toBe(0);
    });

    it("skips nodes with existing results", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });

      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      add1.result = {} as StorageHandle;
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, add1, add2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      // add1 has result so is skipped; add2 becomes canonical — 0 eliminations
      expect(count).toBe(0);
    });

    it("handles materialized input refs in structural key", () => {
      const matRef = materializedRef("f32");
      const add1 = createLazyIRNode("relu", [matRef], [4], "f32", "cpu");
      const add2 = createLazyIRNode("relu", [matRef], [4], "f32", "cpu");

      const ctx = buildContext([add1, add2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(1);
      expect(bypassed.has(add2.id)).toBe(true);
    });

    it("handles scalar refs in structural key", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const sRef: LazyRef = { kind: "scalar", value: 42, dtype: "f32" };
      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), sRef],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), sRef],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, add1, add2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(1);
    });

    it("different scalar values produce different keys", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const s1: LazyRef = { kind: "scalar", value: 42, dtype: "f32" };
      const s2: LazyRef = { kind: "scalar", value: 99, dtype: "f32" };
      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), s1],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), s2],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, add1, add2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("does NOT merge consumers reading DIFFERENT outputs of one multi-output node", () => {
      // Regression guard: structuralKey MUST include outputIndex. A multi-output
      // node (e.g. fusedAttentionBackward → dQ/dK/dV at indices 0/1/2) feeds
      // structurally-identical consumers that differ ONLY in which output slot
      // they read. If the key drops outputIndex these keys collide, CSE merges
      // the consumers, and every consumer collapses onto a single output — a
      // silent gradient-corruption bug for any multi-output op. See
      // tools/sdpa2-diff.ts and src/compiler/graph-rewrites.ts.
      const prod = createLazyIRNode("matmul", [], [4], "f32", "cpu");
      const mk = (outputIndex: number) =>
        createLazyIRNode(
          "contiguous",
          [createPendingRef(prod, outputIndex)],
          [4],
          "f32",
          "cpu",
        );
      const c0 = mk(0);
      const c1 = mk(1);
      const c2 = mk(2);

      const ctx = buildContext([prod, c0, c1, c2]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      // Three distinct outputs → three distinct keys → nothing merged.
      expect(count).toBe(0);
      expect(bypassed.size).toBe(0);
    });

    it("DOES merge consumers reading the SAME output index (sanity)", () => {
      // The complement of the guard above: when the output slot matches, CSE
      // must still fire — the outputIndex addition didn't break ordinary CSE.
      const prod = createLazyIRNode("matmul", [], [4], "f32", "cpu");
      const mk = () =>
        createLazyIRNode(
          "contiguous",
          [createPendingRef(prod, 1)],
          [4],
          "f32",
          "cpu",
        );
      const c0 = mk();
      const c1 = mk();

      const ctx = buildContext([prod, c0, c1]);
      const bypassed = new Set<number>();
      const count = eliminateCommonSubexpressions(ctx, bypassed);

      expect(count).toBe(1);
      expect(bypassed.has(c1.id)).toBe(true);
    });
  });

  // ========================================================================
  // DCE Pass
  // ========================================================================
  describe("eliminateDeadCode", () => {
    it("eliminates nodes with zero consumers", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });
      const deadAdd = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      // liveNode is the last node (output) — should not be eliminated
      const liveNode = createLazyIRNode(
        "mul",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, deadAdd, liveNode]);
      const bypassed = new Set<number>();
      const count = eliminateDeadCode(ctx, bypassed);

      expect(count).toBe(1);
      expect(bypassed.has(deadAdd.id)).toBe(true);
      expect(bypassed.has(liveNode.id)).toBe(false);
    });

    it("cascading DCE: removing dead node makes its inputs dead", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      // dead chain: neg → abs → (no consumers)
      const neg = createLazyIRNode(
        "neg",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );
      const abs = createLazyIRNode(
        "abs",
        [createPendingRef(neg)],
        [4],
        "f32",
        "cpu",
      );
      // live output
      const live = createLazyIRNode(
        "relu",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, neg, abs, live]);
      const bypassed = new Set<number>();
      const count = eliminateDeadCode(ctx, bypassed);

      // abs is dead (no consumer), neg becomes dead after abs removed
      expect(count).toBe(2);
      expect(bypassed.has(abs.id)).toBe(true);
      expect(bypassed.has(neg.id)).toBe(true);
      expect(bypassed.has(live.id)).toBe(false);
    });

    it("does not eliminate the last node (plan output)", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const onlyNode = createLazyIRNode(
        "relu",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, onlyNode]);
      const bypassed = new Set<number>();
      const count = eliminateDeadCode(ctx, bypassed);

      expect(count).toBe(0);
      expect(bypassed.has(onlyNode.id)).toBe(false);
    });

    it("does not eliminate nodes with existing results", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const dead = createLazyIRNode(
        "relu",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );
      dead.result = {} as StorageHandle; // Already executed
      const live = createLazyIRNode(
        "neg",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, dead, live]);
      const bypassed = new Set<number>();
      const count = eliminateDeadCode(ctx, bypassed);

      expect(count).toBe(0);
    });

    it("does not eliminate already-bypassed nodes", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const dead = createLazyIRNode(
        "relu",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );
      const live = createLazyIRNode(
        "neg",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, dead, live]);
      const bypassed = new Set<number>();
      bypassed.add(dead.id);
      const count = eliminateDeadCode(ctx, bypassed);

      // Already bypassed, so doesn't count as newly eliminated
      expect(count).toBe(0);
    });

    it("CSE + DCE together: CSE creates dead nodes, DCE removes them", () => {
      const x = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [1, 2, 3, 4],
      });
      const y = createLazyIRNode("tensorFromArray", [], [4], "f32", "cpu", {
        values: [5, 6, 7, 8],
      });

      // Two identical adds — CSE will eliminate add2
      const add1 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      const add2 = createLazyIRNode(
        "add",
        [createPendingRef(x), createPendingRef(y)],
        [4],
        "f32",
        "cpu",
      );
      // A node only consumed by add2 → becomes dead after CSE
      const onlyForAdd2 = createLazyIRNode(
        "relu",
        [createPendingRef(add2)],
        [4],
        "f32",
        "cpu",
      );
      // Live output uses add1
      const output = createLazyIRNode(
        "mul",
        [createPendingRef(add1), createPendingRef(onlyForAdd2)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, y, add1, add2, onlyForAdd2, output]);
      const bypassed = new Set<number>();

      // CSE eliminates add2 → onlyForAdd2's input is now add1
      eliminateCommonSubexpressions(ctx, bypassed);
      expect(bypassed.has(add2.id)).toBe(true);

      // onlyForAdd2 still has a consumer (output), so it's live
      // But its input changed from add2 to add1, so it's not dead
      const dceCount = eliminateDeadCode(ctx, bypassed);
      // No new dead code because onlyForAdd2's consumer (output) is still there
      expect(dceCount).toBe(0);
    });
  });

  // ========================================================================
  // GraphPass interface + runPasses
  // ========================================================================
  describe("GraphPass interface", () => {
    it("SIMPLIFICATION_PASSES contains all expected passes", () => {
      const names = SIMPLIFICATION_PASSES.map((p) => p.name);
      expect(names).toContain("identity-casts");
      expect(names).toContain("redundant-contiguous");
      expect(names).toContain("algebraic-identities");
      expect(names).toContain("cse");
      expect(names).toContain("dce");
    });

    it("runPasses returns per-pass stats", () => {
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
        { dtype: "f32" },
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, castNode, consumer]);
      const bypassed = new Set<number>();
      const stats = runPasses(ctx, bypassed, SIMPLIFICATION_PASSES);

      expect(stats.get("identity-casts")).toBe(1);
      expect(stats.get("redundant-contiguous")).toBe(0);
      expect(stats.get("algebraic-identities")).toBe(0);
      expect(typeof stats.get("cse")).toBe("number");
      expect(typeof stats.get("dce")).toBe("number");
    });

    it("passes are composable — output of one feeds the next", () => {
      // identity cast → contiguous chain: cast(f32→f32) → contiguous
      const x = createLazyIRNode("add", [], [4], "f32", "cpu");
      const castNode = createLazyIRNode(
        "cast",
        [createPendingRef(x)],
        [4],
        "f32",
        "cpu",
        { dtype: "f32" },
      );
      // contiguous on cast result — after cast is bypassed, contiguous's input
      // becomes x (a compute op), so it should also be bypassed
      const contig = createLazyIRNode(
        "contiguous",
        [createPendingRef(castNode)],
        [4],
        "f32",
        "cpu",
      );
      const consumer = createLazyIRNode(
        "relu",
        [createPendingRef(contig)],
        [4],
        "f32",
        "cpu",
      );

      const ctx = buildContext([x, castNode, contig, consumer]);
      const bypassed = new Set<number>();
      const stats = runPasses(ctx, bypassed, SIMPLIFICATION_PASSES);

      expect(stats.get("identity-casts")).toBe(1);
      expect(stats.get("redundant-contiguous")).toBe(1);
      // Consumer should reference x directly
      const ref = consumer.inputs[0];
      expect(ref.kind).toBe("pending");
      if (ref.kind === "pending") {
        expect(ref.node.id).toBe(x.id);
      }
    });
  });
});
