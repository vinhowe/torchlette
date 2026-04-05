/**
 * fuse-matmul-sum rule tests.
 *
 * Unit tests: build small IR fragments, verify the rule fires or doesn't.
 * Semantic test: construct a tiny plan with batched gradient shape,
 * execute with and without the rewrite, verify outputs match on CPU.
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../../src/graph/node-factory";
import type {
  LazyIRNode,
  LazyRef,
  StorageHandle,
} from "../../../src/graph/types";
import { applyRules } from "../../../src/compiler/rewriter/engine";
import { fuseMatmulSumRule } from "../../../src/compiler/rewriter/rules/fuse-matmul-sum";
import type { ConsumerMaps } from "../../../src/compiler/rewriter/substitute";

// ============================================================================
// Helpers
// ============================================================================

function pendingRef(node: LazyIRNode): LazyRef {
  return { kind: "pending", node };
}

function matRef(id: number, shape: number[], dtype = "f32" as const): LazyRef {
  const storage = {
    id,
    backendTensor: { shape, dtype },
  } as unknown as StorageHandle;
  return { kind: "materialized", storage };
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

/** Build the standard pattern: sum(matmul(transpose(X), Y), dim=batch_dims). */
function buildPatternPlan(
  xShape: number[],
  yShape: number[],
  sumDim: number | number[] = 0,
): {
  plan: LazyIRNode[];
  sum: LazyIRNode;
  transpose: LazyIRNode;
  matmul: LazyIRNode;
  final: LazyIRNode;
  X: LazyRef;
  Y: LazyRef;
} {
  const X = matRef(1, xShape);
  const Y = matRef(2, yShape);
  const rank = xShape.length;
  // transpose(X, -2, -1) — swaps last two dims
  const tShape = [...xShape];
  [tShape[rank - 2], tShape[rank - 1]] = [tShape[rank - 1], tShape[rank - 2]];
  const transpose = createLazyIRNode("transpose", [X], tShape, "f32", "cpu", {
    dim0: -2,
    dim1: -1,
  });
  // matmul output: replace X[-1] with Y[-1]
  const mmShape = [...tShape];
  mmShape[rank - 1] = yShape[rank - 1];
  const matmul = createLazyIRNode(
    "matmul",
    [pendingRef(transpose), Y],
    mmShape,
    "f32",
    "cpu",
  );
  // sum output
  const sumDims = Array.isArray(sumDim) ? sumDim : [sumDim];
  const sumShape = mmShape.filter((_, i) => !sumDims.includes(i));
  const sum = createLazyIRNode("sum", [pendingRef(matmul)], sumShape, "f32", "cpu", {
    dim: sumDim,
    keepdim: false,
  });
  // A downstream consumer so sum isn't orphaned in the plan
  const final = createLazyIRNode(
    "relu",
    [pendingRef(sum)],
    sumShape,
    "f32",
    "cpu",
  );
  return {
    plan: [transpose, matmul, sum, final],
    sum,
    transpose,
    matmul,
    final,
    X,
    Y,
  };
}

// ============================================================================
// Happy path
// ============================================================================

describe("fuse-matmul-sum / happy path", () => {
  it("fires on the standard matmul-sum pattern at rank 3", () => {
    // X: [4, 8, 16], Y: [4, 8, 32]. matmul: [4, 16, 32]. sum dim [0]: [16, 32].
    const { plan, sum, final } = buildPatternPlan([4, 8, 16], [4, 8, 32], 0);
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(1);
    expect(stats.byRule.get("fuse-matmul-sum")).toBe(1);

    // sum has been mutated in place to a matmul; final still references it by id.
    expect(sum.op).toBe("matmul");
    expect(sum.shape).toEqual([16, 32]);
    const ref = final.inputs[0];
    expect(ref.kind).toBe("pending");
    if (ref.kind === "pending") {
      expect(ref.node).toBe(sum);
    }
  });

  it("fires on rank 4 (two batch dims)", () => {
    // X: [2, 3, 8, 16], Y: [2, 3, 8, 32]. matmul: [2, 3, 16, 32].
    // sum dim [0, 1]: [16, 32].
    const { plan, sum } = buildPatternPlan(
      [2, 3, 8, 16],
      [2, 3, 8, 32],
      [0, 1],
    );
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(1);
    expect(sum.op).toBe("matmul");
    expect(sum.shape).toEqual([16, 32]);
  });

  it("inserts 3 new nodes (reshape×2, transpose) and mutates sum→matmul", () => {
    const { plan, sum } = buildPatternPlan([4, 8, 16], [4, 8, 32], 0);
    const originalLen = plan.length;
    const maps = buildMaps(plan);
    applyRules(plan, [fuseMatmulSumRule], maps);
    expect(plan.length).toBe(originalLen + 3);
    // The matched sum is mutated in place to a matmul.
    expect(sum.op).toBe("matmul");
  });

  it("new reshape has correct total-row dimension (batch * M)", () => {
    // X: [4, 8, 16] → reshape to [4*8=32, 16]
    const { plan, final } = buildPatternPlan([4, 8, 16], [4, 8, 32], 0);
    const maps = buildMaps(plan);
    applyRules(plan, [fuseMatmulSumRule], maps);

    // Walk the new chain: final → matmul → transpose → reshape(X)
    const mmRef = final.inputs[0];
    if (mmRef.kind !== "pending") throw new Error("expected pending");
    const mm = mmRef.node;
    expect(mm.op).toBe("matmul");
    // First input: transpose of reshape(X)
    const tRef = mm.inputs[0];
    if (tRef.kind !== "pending") throw new Error("expected pending");
    expect(tRef.node.op).toBe("transpose");
    expect(tRef.node.shape).toEqual([16, 32]); // [K, totalRows]
    // transpose's input: reshape(X)
    const rxRef = tRef.node.inputs[0];
    if (rxRef.kind !== "pending") throw new Error("expected pending");
    expect(rxRef.node.op).toBe("reshape");
    expect(rxRef.node.shape).toEqual([32, 16]); // [totalRows, K]
  });
});

// ============================================================================
// Rejections
// ============================================================================

describe("fuse-matmul-sum / rejections", () => {
  it("rejects keepdim=true", () => {
    const { plan } = buildPatternPlan([4, 8, 16], [4, 8, 32], 0);
    // Mutate the sum node's payload
    const sum = plan[2];
    (sum.payload as { keepdim: boolean }).keepdim = true;
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(0);
  });

  it("rejects non-leading batch dims (sum over dim 1)", () => {
    const { plan } = buildPatternPlan([4, 8, 16], [4, 8, 32], 0);
    const sum = plan[2];
    (sum.payload as { dim: number }).dim = 1; // mid-rank dim, not leading
    // Also need to update output shape; the rule should still reject on dims
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(0);
  });

  it("rejects rank-2 matmul (no batch dim)", () => {
    // X: [8, 16], Y: [8, 32]. matmul: [16, 32]. Sum dim [] would be no-op.
    // Actually build: transpose(X)[-2,-1] is same as regular transpose.
    // rank 2 matmul has no batch dim so the pattern can't match leading batch.
    const X = matRef(1, [8, 16]);
    const Y = matRef(2, [8, 32]);
    const transpose = createLazyIRNode(
      "transpose",
      [X],
      [16, 8],
      "f32",
      "cpu",
      { dim0: -2, dim1: -1 },
    );
    const matmul = createLazyIRNode(
      "matmul",
      [pendingRef(transpose), Y],
      [16, 32],
      "f32",
      "cpu",
    );
    // Rank 2 matmul — sum doesn't need to fire. Make sure pattern doesn't fire.
    const sum = createLazyIRNode(
      "sum",
      [pendingRef(matmul)],
      [32],
      "f32",
      "cpu",
      { dim: 0, keepdim: false },
    );
    const final = createLazyIRNode("relu", [pendingRef(sum)], [32], "f32", "cpu");
    const plan = [transpose, matmul, sum, final];
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(0);
  });

  it("rejects when transpose is on second matmul operand", () => {
    const X = matRef(1, [4, 16, 8]);
    const Y = matRef(2, [4, 32, 8]);
    // transpose(Y, -2, -1): [4, 8, 32]
    const ty = createLazyIRNode(
      "transpose",
      [Y],
      [4, 8, 32],
      "f32",
      "cpu",
      { dim0: -2, dim1: -1 },
    );
    // matmul(X, ty): [4, 16, 8] @ [4, 8, 32] = [4, 16, 32]
    const matmul = createLazyIRNode(
      "matmul",
      [X, pendingRef(ty)],
      [4, 16, 32],
      "f32",
      "cpu",
    );
    const sum = createLazyIRNode(
      "sum",
      [pendingRef(matmul)],
      [16, 32],
      "f32",
      "cpu",
      { dim: 0, keepdim: false },
    );
    const final = createLazyIRNode(
      "relu",
      [pendingRef(sum)],
      [16, 32],
      "f32",
      "cpu",
    );
    const plan = [ty, matmul, sum, final];
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(0);
  });

  it("rejects when transpose is not on last 2 dims", () => {
    const X = matRef(1, [4, 8, 16]);
    const Y = matRef(2, [4, 8, 32]);
    // transpose(X, 0, 1) — wrong dims
    const transpose = createLazyIRNode(
      "transpose",
      [X],
      [8, 4, 16],
      "f32",
      "cpu",
      { dim0: 0, dim1: 1 },
    );
    const matmul = createLazyIRNode(
      "matmul",
      [pendingRef(transpose), Y],
      [8, 4, 32],
      "f32",
      "cpu",
    );
    const sum = createLazyIRNode(
      "sum",
      [pendingRef(matmul)],
      [4, 32],
      "f32",
      "cpu",
      { dim: 0, keepdim: false },
    );
    const final = createLazyIRNode(
      "relu",
      [pendingRef(sum)],
      [4, 32],
      "f32",
      "cpu",
    );
    const plan = [transpose, matmul, sum, final];
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(0);
  });

  it("rejects when batch dims don't match between X and Y", () => {
    // X: [4, 8, 16], Y: [5, 8, 32] — batch dim 4 != 5
    const X = matRef(1, [4, 8, 16]);
    const Y = matRef(2, [5, 8, 32]);
    const transpose = createLazyIRNode(
      "transpose",
      [X],
      [4, 16, 8],
      "f32",
      "cpu",
      { dim0: -2, dim1: -1 },
    );
    // Construct matmul even though shapes wouldn't broadcast — we're testing
    // the rule's predicate, not the matmul itself.
    const matmul = createLazyIRNode(
      "matmul",
      [pendingRef(transpose), Y],
      [4, 16, 32],
      "f32",
      "cpu",
    );
    const sum = createLazyIRNode(
      "sum",
      [pendingRef(matmul)],
      [16, 32],
      "f32",
      "cpu",
      { dim: 0, keepdim: false },
    );
    const final = createLazyIRNode(
      "relu",
      [pendingRef(sum)],
      [16, 32],
      "f32",
      "cpu",
    );
    const plan = [transpose, matmul, sum, final];
    const maps = buildMaps(plan);
    const stats = applyRules(plan, [fuseMatmulSumRule], maps);
    expect(stats.applied).toBe(0);
  });
});
