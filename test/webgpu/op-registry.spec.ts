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
        expect(OP_REGISTRY[op].outputDtype).toBe("f32");
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

  describe("expression generation", () => {
    it("generates unary expressions", () => {
      expect(OP_REGISTRY.relu.expr("x")).toBe("select(0.0, x, x > 0.0)");
      expect(OP_REGISTRY.neg.expr("x")).toBe("(-x)");
      expect(OP_REGISTRY.sqrt.expr("x")).toBe("sqrt(x)");
      expect(OP_REGISTRY.exp.expr("x")).toBe("exp(x)");
      expect(OP_REGISTRY.log.expr("x")).toBe("log(x)");
      expect(OP_REGISTRY.tanh.expr("x")).toBe("tanh(x)");
      expect(OP_REGISTRY.rsqrt.expr("x")).toBe("inverseSqrt(x)");
      expect(OP_REGISTRY.sin.expr("x")).toBe("sin(x)");
      expect(OP_REGISTRY.cos.expr("x")).toBe("cos(x)");
      expect(OP_REGISTRY.floor.expr("x")).toBe("floor(x)");
      expect(OP_REGISTRY.ceil.expr("x")).toBe("ceil(x)");
      expect(OP_REGISTRY.round.expr("x")).toBe("round(x)");
    });

    it("generates gelu expression with tanh", () => {
      const expr = OP_REGISTRY.gelu.expr("x");
      expect(expr).toContain("tanh");
      expect(expr).toContain("0.5");
      expect(expr).toContain("clamp");
    });

    it("generates silu/sigmoid expressions", () => {
      expect(OP_REGISTRY.silu.expr("x")).toContain("exp");
      expect(OP_REGISTRY.sigmoid.expr("x")).toContain("exp");
    });

    it("generates binary arithmetic expressions", () => {
      expect(OP_REGISTRY.add.expr("a", "b")).toBe("(a + b)");
      expect(OP_REGISTRY.sub.expr("a", "b")).toBe("(a - b)");
      expect(OP_REGISTRY.mul.expr("a", "b")).toBe("(a * b)");
      expect(OP_REGISTRY.div.expr("a", "b")).toBe("(a / b)");
      expect(OP_REGISTRY.mod.expr("a", "b")).toBe("(a % b)");
      expect(OP_REGISTRY.pow.expr("a", "b")).toBe("pow(a, b)");
      expect(OP_REGISTRY.min.expr("a", "b")).toBe("min(a, b)");
      expect(OP_REGISTRY.max.expr("a", "b")).toBe("max(a, b)");
    });

    it("generates comparison expressions with select", () => {
      expect(OP_REGISTRY.eq.expr("a", "b")).toBe("select(0.0, 1.0, a == b)");
      expect(OP_REGISTRY.ne.expr("a", "b")).toBe("select(0.0, 1.0, a != b)");
      expect(OP_REGISTRY.lt.expr("a", "b")).toBe("select(0.0, 1.0, a < b)");
      expect(OP_REGISTRY.le.expr("a", "b")).toBe("select(0.0, 1.0, a <= b)");
      expect(OP_REGISTRY.gt.expr("a", "b")).toBe("select(0.0, 1.0, a > b)");
      expect(OP_REGISTRY.ge.expr("a", "b")).toBe("select(0.0, 1.0, a >= b)");
    });

    it("generates where expression", () => {
      expect(OP_REGISTRY.where.expr("cond", "a", "b")).toBe(
        "select(b, a, cond > 0.0)",
      );
    });

    it("generates cast expressions", () => {
      expect(OP_REGISTRY.cast_f16.expr("x")).toBe("f16(x)");
      expect(OP_REGISTRY.cast_f32.expr("x")).toBe("f32(x)");
      expect(OP_REGISTRY.cast_i32.expr("x")).toBe("i32(x)");
      expect(OP_REGISTRY.cast_u32.expr("x")).toBe("u32(x)");
    });

    it("uses custom zero for relu when provided", () => {
      const expr = OP_REGISTRY.relu.expr(
        "v",
        "vec4<f32>(0.0)",
        "vec4<f32>(1.0)",
      );
      expect(expr).toContain("vec4<f32>(0.0)");
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

    it("casts are vectorizable via vectorExpr", () => {
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
      ];
      for (const [_name, def] of Object.entries(OP_REGISTRY)) {
        expect([1, 2, 3]).toContain(def.arity);
        expect(validCategories).toContain(def.category);
        expect(typeof def.fusible).toBe("boolean");
        expect(typeof def.vectorizable).toBe("boolean");
      }
    });

    it("all registered ops are fusible", () => {
      for (const [name, def] of Object.entries(OP_REGISTRY)) {
        expect(def.fusible, `Op ${name} should be fusible`).toBe(true);
      }
    });
  });
});
