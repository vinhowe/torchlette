/**
 * Matcher tests. Each test builds a tiny LazyIR fragment by hand and
 * checks match/no-match + bindings.
 */
import { describe, expect, it } from "vitest";
import { createLazyIRNode } from "../../src/graph/node-factory";
import type { LazyIRNode, LazyRef, StorageHandle } from "../../src/graph/types";
import {
  any,
  capture,
  materialized,
  op,
  scalar,
} from "../../src/compiler/rewriter/pattern";
import { match, refEquals } from "../../src/compiler/rewriter/matcher";

// ============================================================================
// Test fixtures
// ============================================================================

/** Build a pending ref from a LazyIRNode. */
function pendingRef(node: LazyIRNode, outputIndex?: number): LazyRef {
  return outputIndex !== undefined
    ? { kind: "pending", node, outputIndex }
    : { kind: "pending", node };
}

/** Build a materialized ref with an arbitrary storage id. */
function matRef(id: number): LazyRef {
  const storage = { id } as unknown as StorageHandle;
  return { kind: "materialized", storage };
}

/** Build a scalar ref. */
function scalarRef(value: number): LazyRef {
  return { kind: "scalar", value, dtype: "f32" };
}

/** Test helper: construct `add(a, b)`. */
function addNode(a: LazyRef, b: LazyRef): LazyIRNode {
  return createLazyIRNode("add", [a, b], [1], "f32", "cpu");
}

/** Test helper: construct `mul(a, b)`. */
function mulNode(a: LazyRef, b: LazyRef): LazyIRNode {
  return createLazyIRNode("mul", [a, b], [1], "f32", "cpu");
}

// ============================================================================
// any
// ============================================================================

describe("matcher / any", () => {
  it("matches a pending ref", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(match(any, pendingRef(n))).not.toBeNull();
  });

  it("matches a materialized ref", () => {
    expect(match(any, matRef(1))).not.toBeNull();
  });

  it("matches a scalar ref", () => {
    expect(match(any, scalarRef(3.14))).not.toBeNull();
  });

  it("returns empty bindings", () => {
    const b = match(any, matRef(1));
    expect(b).not.toBeNull();
    expect(b!.size).toBe(0);
  });
});

// ============================================================================
// capture
// ============================================================================

describe("matcher / capture", () => {
  it("binds name to matched ref", () => {
    const r = matRef(42);
    const b = match(capture("X"), r);
    expect(b).not.toBeNull();
    expect(b!.get("X")).toBe(r);
  });

  it("binds to a pending ref", () => {
    const n = addNode(matRef(1), matRef(2));
    const r = pendingRef(n);
    const b = match(capture("X"), r);
    expect(b).not.toBeNull();
    expect(b!.get("X")).toBe(r);
  });

  it("enforces same-ref equality when name is reused", () => {
    // pattern: add(capture("X"), capture("X"))
    // input:   add(mat1, mat1) — same ref twice
    const mat1 = matRef(1);
    const n = addNode(mat1, mat1);
    const pattern = op("add", { inputs: [capture("X"), capture("X")] });
    expect(match(pattern, pendingRef(n))).not.toBeNull();
  });

  it("rejects when same capture name binds to different refs", () => {
    const n = addNode(matRef(1), matRef(2));
    const pattern = op("add", { inputs: [capture("X"), capture("X")] });
    expect(match(pattern, pendingRef(n))).toBeNull();
  });

  it("inner pattern gates the capture", () => {
    // capture("X", inner=op("add")) — capture X only if ref points to an add
    const addN = addNode(matRef(1), matRef(2));
    const mulN = mulNode(matRef(1), matRef(2));
    const pattern = capture("X", op("add"));
    expect(match(pattern, pendingRef(addN))).not.toBeNull();
    expect(match(pattern, pendingRef(mulN))).toBeNull();
  });

  it("does not leave stray bindings on inner-pattern failure", () => {
    // add(capture("X", op("mul")), any) — X requires a mul, add(add, x) should fail
    // but if we tried a deeper nested structure, a captured name must roll back.
    const inner = addNode(matRef(1), matRef(2));
    const outer = addNode(pendingRef(inner), matRef(3));
    const pattern = op("add", {
      inputs: [capture("X", op("mul")), any],
    });
    const b = match(pattern, pendingRef(outer));
    expect(b).toBeNull();
  });
});

// ============================================================================
// op
// ============================================================================

describe("matcher / op", () => {
  it("matches a pending ref pointing to the right op", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(match(op("add"), pendingRef(n))).not.toBeNull();
  });

  it("rejects on wrong op code", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(match(op("mul"), pendingRef(n))).toBeNull();
  });

  it("rejects on non-pending ref", () => {
    expect(match(op("add"), matRef(1))).toBeNull();
    expect(match(op("add"), scalarRef(1))).toBeNull();
  });

  it("matches without inputs clause (ignores inputs)", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(match(op("add"), pendingRef(n))).not.toBeNull();
  });

  it("matches with correct input arity", () => {
    const n = addNode(matRef(1), matRef(2));
    const pattern = op("add", { inputs: [any, any] });
    expect(match(pattern, pendingRef(n))).not.toBeNull();
  });

  it("rejects on wrong input arity", () => {
    const n = addNode(matRef(1), matRef(2));
    const pattern = op("add", { inputs: [any, any, any] });
    expect(match(pattern, pendingRef(n))).toBeNull();
  });

  it("recurses into inputs", () => {
    // add(mul(a, b), c)
    const inner = mulNode(matRef(1), matRef(2));
    const outer = addNode(pendingRef(inner), matRef(3));
    const pattern = op("add", {
      inputs: [op("mul", { inputs: [any, any] }), any],
    });
    expect(match(pattern, pendingRef(outer))).not.toBeNull();
  });

  it("fails recursion when nested op mismatches", () => {
    // Pattern expects add(sub(...), any), input is add(mul(...), any)
    const inner = mulNode(matRef(1), matRef(2));
    const outer = addNode(pendingRef(inner), matRef(3));
    const pattern = op("add", { inputs: [op("sub"), any] });
    expect(match(pattern, pendingRef(outer))).toBeNull();
  });

  it("applies where predicate", () => {
    const n1 = createLazyIRNode("add", [matRef(1), matRef(2)], [10], "f32", "cpu");
    const n2 = createLazyIRNode("add", [matRef(1), matRef(2)], [5], "f32", "cpu");
    const pattern = op("add", { where: (node) => node.shape[0] > 7 });
    expect(match(pattern, pendingRef(n1))).not.toBeNull();
    expect(match(pattern, pendingRef(n2))).toBeNull();
  });

  it("collects captures from nested inputs", () => {
    // add(mul(capture A, any), capture B)
    const inner = mulNode(matRef(1), matRef(2));
    const outer = addNode(pendingRef(inner), matRef(3));
    const pattern = op("add", {
      inputs: [op("mul", { inputs: [capture("A"), any] }), capture("B")],
    });
    const b = match(pattern, pendingRef(outer));
    expect(b).not.toBeNull();
    expect(b!.size).toBe(2);
    expect(b!.get("A")).toEqual(matRef(1));
    expect(b!.get("B")).toEqual(matRef(3));
  });

  it("rolls back bindings on input-mismatch failure", () => {
    // add(capture A, op("sub")) — matches capture A successfully, then sub fails.
    // A must not leak out.
    const n = addNode(matRef(1), matRef(2));
    const pattern = op("add", {
      inputs: [capture("A"), op("sub")],
    });
    expect(match(pattern, pendingRef(n))).toBeNull();
  });
});

// ============================================================================
// scalar
// ============================================================================

describe("matcher / scalar", () => {
  it("matches a scalar ref", () => {
    expect(match(scalar(), scalarRef(1))).not.toBeNull();
  });

  it("rejects non-scalar refs", () => {
    expect(match(scalar(), matRef(1))).toBeNull();
    const n = addNode(matRef(1), matRef(2));
    expect(match(scalar(), pendingRef(n))).toBeNull();
  });

  it("applies value predicate", () => {
    const p = scalar((v) => v > 10);
    expect(match(p, scalarRef(11))).not.toBeNull();
    expect(match(p, scalarRef(5))).toBeNull();
  });
});

// ============================================================================
// materialized
// ============================================================================

describe("matcher / materialized", () => {
  it("matches a materialized ref", () => {
    expect(match(materialized, matRef(1))).not.toBeNull();
  });

  it("rejects pending and scalar refs", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(match(materialized, pendingRef(n))).toBeNull();
    expect(match(materialized, scalarRef(1))).toBeNull();
  });
});

// ============================================================================
// refEquals
// ============================================================================

describe("matcher / refEquals", () => {
  it("pending refs: same node + same outputIndex → equal", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(refEquals(pendingRef(n), pendingRef(n))).toBe(true);
    expect(refEquals(pendingRef(n, 0), pendingRef(n, 0))).toBe(true);
    expect(refEquals(pendingRef(n), pendingRef(n, 0))).toBe(true); // default
  });

  it("pending refs: different outputIndex → unequal", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(refEquals(pendingRef(n, 0), pendingRef(n, 1))).toBe(false);
  });

  it("pending refs: different nodes → unequal", () => {
    const n1 = addNode(matRef(1), matRef(2));
    const n2 = addNode(matRef(1), matRef(2));
    expect(refEquals(pendingRef(n1), pendingRef(n2))).toBe(false);
  });

  it("materialized refs: same storage id → equal", () => {
    expect(refEquals(matRef(7), matRef(7))).toBe(true);
  });

  it("materialized refs: different storage id → unequal", () => {
    expect(refEquals(matRef(7), matRef(8))).toBe(false);
  });

  it("scalar refs: same value + dtype → equal", () => {
    expect(refEquals(scalarRef(3), scalarRef(3))).toBe(true);
  });

  it("scalar refs: different value → unequal", () => {
    expect(refEquals(scalarRef(3), scalarRef(4))).toBe(false);
  });

  it("cross-kind refs are never equal", () => {
    const n = addNode(matRef(1), matRef(2));
    expect(refEquals(pendingRef(n), matRef(1))).toBe(false);
    expect(refEquals(pendingRef(n), scalarRef(1))).toBe(false);
    expect(refEquals(matRef(1), scalarRef(1))).toBe(false);
  });
});

// ============================================================================
// Complex / realistic patterns
// ============================================================================

describe("matcher / realistic patterns", () => {
  it("matches the matmul-sum shape: sum(matmul(transpose(X), Y))", () => {
    // Build: sum(matmul(transpose(X), Y), dim=[0], keepdim=false)
    const xRef = matRef(100); // represents X
    const yRef = matRef(200); // represents Y
    const tx = createLazyIRNode(
      "transpose",
      [xRef],
      [4, 16, 32],
      "f32",
      "cpu",
      { dim0: -2, dim1: -1 },
    );
    const mm = createLazyIRNode(
      "matmul",
      [pendingRef(tx), yRef],
      [4, 16, 8],
      "f32",
      "cpu",
    );
    const sumNode = createLazyIRNode(
      "sum",
      [pendingRef(mm)],
      [16, 8],
      "f32",
      "cpu",
      { dim: [0], keepdim: false },
    );

    const pattern = op("sum", {
      where: (n) => {
        const p = n.payload as { keepdim?: boolean };
        return p?.keepdim === false;
      },
      inputs: [
        op("matmul", {
          inputs: [
            op("transpose", { inputs: [capture("X")] }),
            capture("Y"),
          ],
        }),
      ],
    });

    const b = match(pattern, pendingRef(sumNode));
    expect(b).not.toBeNull();
    expect(b!.get("X")).toBe(xRef);
    expect(b!.get("Y")).toBe(yRef);
  });

  it("rejects matmul-sum shape when transpose is on wrong operand", () => {
    // Build: sum(matmul(X, transpose(Y)), ...) — transpose on second, not first
    const xRef = matRef(100);
    const yRef = matRef(200);
    const ty = createLazyIRNode("transpose", [yRef], [4, 8, 16], "f32", "cpu");
    const mm = createLazyIRNode(
      "matmul",
      [xRef, pendingRef(ty)],
      [4, 16, 8],
      "f32",
      "cpu",
    );
    const sumNode = createLazyIRNode(
      "sum",
      [pendingRef(mm)],
      [16, 8],
      "f32",
      "cpu",
    );

    // Pattern expects transpose on FIRST operand.
    const pattern = op("sum", {
      inputs: [
        op("matmul", {
          inputs: [op("transpose"), capture("Y")],
        }),
      ],
    });
    expect(match(pattern, pendingRef(sumNode))).toBeNull();
  });
});
