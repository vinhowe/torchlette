/**
 * Execution-declaration schema gate (P0). CPU-only: the declaration is DATA, so
 * this exercises no GPU. Proves (a) every elementwise declaration is
 * generator-free data, (b) the declared op set agrees with the kernel-family
 * tables (single source — no op is silently dropped), (c) the schema gate
 * actually REJECTS a smuggled function/buffer leaf (the "no second owner"
 * enforcement is real, not decorative).
 */
import { describe, expect, it } from "vitest";
import {
  assertNoGeneratorLeaf,
  ELEMENTWISE_BINARY_WGSL,
  ELEMENTWISE_DECLARATIONS,
  ELEMENTWISE_UNARY_OPS,
  type ExecutionDeclaration,
  elementwiseDeclaration,
  isDeclaredElementwise,
} from "../src/executor/execution-declaration";

describe("execution-declaration schema (P0 elementwise)", () => {
  it("every declaration is generator-free data", () => {
    for (const [op, decl] of ELEMENTWISE_DECLARATIONS) {
      expect(() => assertNoGeneratorLeaf(decl, op)).not.toThrow();
    }
  });

  it("declared op set == kernel-family tables (single source)", () => {
    const expected = new Set<string>([
      ...ELEMENTWISE_BINARY_WGSL.keys(),
      ...ELEMENTWISE_UNARY_OPS,
      "gelu",
      "cast",
      "contiguous",
      "where",
    ]);
    expect(new Set(ELEMENTWISE_DECLARATIONS.keys())).toEqual(expected);
  });

  it("arity + kernel family agree with the op class", () => {
    for (const op of ELEMENTWISE_BINARY_WGSL.keys()) {
      const d = elementwiseDeclaration(op);
      expect(d?.arity).toBe(2);
      expect(d?.kernel.kernel).toBe("binaryDirect");
    }
    for (const op of ELEMENTWISE_UNARY_OPS) {
      const d = elementwiseDeclaration(op);
      expect(d?.arity).toBe(1);
      expect(d?.kernel.kernel).toBe("unaryDirect");
    }
    expect(elementwiseDeclaration("where")?.arity).toBe(3);
    expect(elementwiseDeclaration("where")?.kernel.kernel).toBe("whereDirect");
    expect(elementwiseDeclaration("cast")?.kernel.kernel).toBe("castDirect");
    expect(elementwiseDeclaration("contiguous")?.kernel.kernel).toBe(
      "contiguousDirect",
    );
    expect(elementwiseDeclaration("gelu")?.kernel.kernel).toBe("unaryDirect");
  });

  it("the family skeleton is ALLOC → UNIFORM → DISPATCH over roles", () => {
    const d = elementwiseDeclaration("add");
    expect(d).toBeDefined();
    const skel = (d as ExecutionDeclaration).skeleton;
    expect(skel.map((c) => c.op)).toEqual(["alloc", "uniform", "dispatch"]);
    const dispatch = skel[2];
    if (dispatch.op !== "dispatch") throw new Error("expected dispatch");
    expect(dispatch.bindings.map((b) => b.role)).toEqual([
      "all-inputs",
      "output",
      "params",
    ]);
  });

  it("isDeclaredElementwise is exact", () => {
    expect(isDeclaredElementwise("add")).toBe(true);
    expect(isDeclaredElementwise("where")).toBe(true);
    expect(isDeclaredElementwise("matmul")).toBe(false);
    expect(isDeclaredElementwise("sum")).toBe(false);
  });

  it("the schema gate REJECTS a smuggled function leaf", () => {
    const smuggled = {
      family: "elementwise",
      arity: 1,
      kernel: { kernel: "unaryDirect" },
      layout: "strided-ok",
      decompose: "direct",
      // A generator hidden behind an "opaque" leaf — the adapter-cheat.
      skeleton: [{ op: "dispatch", emit: () => [] }],
    } as unknown as ExecutionDeclaration;
    expect(() => assertNoGeneratorLeaf(smuggled)).toThrow(/function/);
  });

  it("the schema gate REJECTS a buffer leaf", () => {
    const withBuffer = {
      family: "elementwise",
      arity: 1,
      kernel: { kernel: "unaryDirect" },
      layout: "strided-ok",
      decompose: "direct",
      skeleton: [{ op: "alloc", bytes: new Uint32Array([1, 2, 3]) }],
    } as unknown as ExecutionDeclaration;
    expect(() => assertNoGeneratorLeaf(withBuffer)).toThrow(/buffer leaf/);
  });
});
