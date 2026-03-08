/**
 * Comprehensive tests for the unified op registry.
 * This is the single source of truth for all fusible operations.
 */
import { describe, expect, it } from "vitest";

import {
  canVectorize,
  isUnaryOp,
  OP_REGISTRY,
} from "../../src/backend/webgpu/ops/registry";

describe("Op Registry", () => {
  describe("Registry structure", () => {
    it("contains all expected activation ops", () => {
      const activations = [
        "relu",
        "gelu",
        "gelu_tanh",
        "gelu_erf",
        "silu",
        "sigmoid",
        "tanh",
        "softplus",
      ];
      for (const op of activations) {
        expect(op in OP_REGISTRY, `Missing activation op: ${op}`).toBe(true);
        expect(OP_REGISTRY[op].category).toBe("activation");
      }
    });

    it("contains all expected math ops", () => {
      const math = [
        "neg",
        "abs",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "floor",
        "ceil",
        "round",
        "sign",
      ];
      for (const op of math) {
        expect(op in OP_REGISTRY, `Missing math op: ${op}`).toBe(true);
        expect(OP_REGISTRY[op].category).toBe("math");
      }
    });

    it("contains all expected arithmetic ops", () => {
      const arithmetic = [
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "min",
        "max",
        "mod",
      ];
      for (const op of arithmetic) {
        expect(op in OP_REGISTRY, `Missing arithmetic op: ${op}`).toBe(true);
        expect(OP_REGISTRY[op].category).toBe("arithmetic");
      }
    });

    it("contains all expected comparison ops", () => {
      const comparisons = ["eq", "ne", "lt", "le", "gt", "ge"];
      for (const op of comparisons) {
        expect(op in OP_REGISTRY, `Missing comparison op: ${op}`).toBe(true);
        expect(OP_REGISTRY[op].category).toBe("comparison");
      }
    });

    it("contains ternary where op", () => {
      expect("where" in OP_REGISTRY).toBe(true);
      expect(OP_REGISTRY.where.category).toBe("ternary");
      expect(OP_REGISTRY.where.arity).toBe(3);
    });

    it("contains all expected cast ops", () => {
      const casts = ["cast_f16", "cast_f32", "cast_i32", "cast_u32"];
      for (const op of casts) {
        expect(op in OP_REGISTRY, `Missing cast op: ${op}`).toBe(true);
        expect(OP_REGISTRY[op].category).toBe("cast");
        expect(OP_REGISTRY[op].vectorizable).toBe(true);
      }
    });
  });

  describe("arity classification", () => {
    it("correctly classifies unary ops", () => {
      expect(isUnaryOp("relu")).toBe(true);
      expect(isUnaryOp("neg")).toBe(true);
      expect(isUnaryOp("sqrt")).toBe(true);
      expect(isUnaryOp("sigmoid")).toBe(true);
      expect(isUnaryOp("cast_f16")).toBe(true);
      expect(isUnaryOp("add")).toBe(false);
      expect(isUnaryOp("where")).toBe(false);
    });

    it("correctly classifies binary and ternary ops", () => {
      expect(OP_REGISTRY.add.arity).toBe(2);
      expect(OP_REGISTRY.mul.arity).toBe(2);
      expect(OP_REGISTRY.lt.arity).toBe(2);
      expect(OP_REGISTRY.where.arity).toBe(3);
    });
  });

  describe("vectorization", () => {
    it("most ops are vectorizable", () => {
      const vectorizable = [
        "relu",
        "add",
        "mul",
        "sigmoid",
        "tanh",
        "neg",
        "sqrt",
      ];
      for (const op of vectorizable) {
        expect(canVectorize(op), `Op ${op} should be vectorizable`).toBe(true);
      }
    });

    it("casts are vectorizable", () => {
      expect(canVectorize("cast_f16")).toBe(true);
      expect(canVectorize("cast_f32")).toBe(true);
    });

    it("mod is not vectorizable", () => {
      expect(canVectorize("mod")).toBe(false);
    });
  });

  describe("consistency checks", () => {
    it("all ops have valid arity, category, and fusible/vectorizable flags", () => {
      const validCategories = [
        "activation",
        "math",
        "arithmetic",
        "comparison",
        "ternary",
        "cast",
        "bitwise",
      ];
      for (const [_name, def] of Object.entries(OP_REGISTRY)) {
        expect([1, 2, 3]).toContain(def.arity);
        expect(validCategories).toContain(def.category);
        expect(typeof def.fusible).toBe("boolean");
        expect(typeof def.vectorizable).toBe("boolean");
      }
    });

    it("all non-bitwise ops are fusible", () => {
      for (const [name, def] of Object.entries(OP_REGISTRY)) {
        if (def.category === "bitwise") continue;
        expect(def.fusible, `Op ${name} should be fusible`).toBe(true);
      }
    });
  });
});
