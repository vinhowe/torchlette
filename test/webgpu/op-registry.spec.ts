/**
 * Comprehensive tests for the unified op registry.
 * This is the single source of truth for all fusible operations.
 */
import { describe, expect, it } from "vitest";

import {
  OP_REGISTRY,
  getOpDef,
  hasOp,
  getExpr,
  isFusible,
  canVectorize,
  getArity,
  isUnaryOp,
  isBinaryOp,
  isTernaryOp,
  getOpsByCategory,
  getAllFusibleOps,
  getAllUnaryOps,
  getAllBinaryOps,
  UNARY_EXPR,
  BINARY_EXPR,
} from "../../src/backend/webgpu/ops/registry";

describe("Op Registry", () => {
  describe("Registry structure", () => {
    it("contains all expected activation ops", () => {
      const activations = ["relu", "gelu", "gelu_tanh", "gelu_erf", "silu", "sigmoid", "tanh", "softplus"];
      for (const op of activations) {
        expect(hasOp(op), `Missing activation op: ${op}`).toBe(true);
        expect(getOpDef(op)?.category).toBe("activation");
      }
    });

    it("contains all expected math ops", () => {
      const math = ["neg", "abs", "exp", "log", "sqrt", "rsqrt", "sin", "cos", "floor", "ceil", "round", "sign"];
      for (const op of math) {
        expect(hasOp(op), `Missing math op: ${op}`).toBe(true);
        expect(getOpDef(op)?.category).toBe("math");
      }
    });

    it("contains all expected arithmetic ops", () => {
      const arithmetic = ["add", "sub", "mul", "div", "pow", "min", "max", "mod"];
      for (const op of arithmetic) {
        expect(hasOp(op), `Missing arithmetic op: ${op}`).toBe(true);
        expect(getOpDef(op)?.category).toBe("arithmetic");
      }
    });

    it("contains all expected comparison ops", () => {
      const comparisons = ["eq", "ne", "lt", "le", "gt", "ge"];
      for (const op of comparisons) {
        expect(hasOp(op), `Missing comparison op: ${op}`).toBe(true);
        expect(getOpDef(op)?.category).toBe("comparison");
        expect(getOpDef(op)?.outputDtype).toBe("f32"); // Comparisons return f32
      }
    });

    it("contains all expected ternary ops", () => {
      expect(hasOp("where")).toBe(true);
      expect(getOpDef("where")?.category).toBe("ternary");
      expect(getOpDef("where")?.arity).toBe(3);
    });

    it("contains all expected cast ops", () => {
      const casts = ["cast_f16", "cast_f32", "cast_i32", "cast_u32"];
      for (const op of casts) {
        expect(hasOp(op), `Missing cast op: ${op}`).toBe(true);
        expect(getOpDef(op)?.category).toBe("cast");
        expect(getOpDef(op)?.vectorizable).toBe(true); // Casts use vectorExpr for vector mode
      }
    });
  });

  describe("getOpDef", () => {
    it("returns definition for known ops", () => {
      const def = getOpDef("relu");
      expect(def).not.toBeNull();
      expect(def?.arity).toBe(1);
      expect(def?.fusible).toBe(true);
    });

    it("returns null for unknown ops", () => {
      expect(getOpDef("unknown_op")).toBeNull();
      expect(getOpDef("matmul")).toBeNull();
      expect(getOpDef("conv2d")).toBeNull();
    });
  });

  describe("hasOp", () => {
    it("returns true for known ops", () => {
      expect(hasOp("relu")).toBe(true);
      expect(hasOp("add")).toBe(true);
      expect(hasOp("where")).toBe(true);
    });

    it("returns false for unknown ops", () => {
      expect(hasOp("unknown")).toBe(false);
      expect(hasOp("matmul")).toBe(false);
    });
  });

  describe("getExpr", () => {
    describe("unary ops", () => {
      it("generates relu expression", () => {
        expect(getExpr("relu", ["x"])).toBe("select(0.0, x, x > 0.0)");
      });

      it("generates neg expression", () => {
        expect(getExpr("neg", ["x"])).toBe("(-x)");
      });

      it("generates sqrt expression", () => {
        expect(getExpr("sqrt", ["x"])).toBe("sqrt(x)");
      });

      it("generates exp expression", () => {
        expect(getExpr("exp", ["x"])).toBe("exp(x)");
      });

      it("generates log expression", () => {
        expect(getExpr("log", ["x"])).toBe("log(x)");
      });

      it("generates gelu expression with tanh", () => {
        const expr = getExpr("gelu", ["x"]);
        expect(expr).toContain("tanh");
        expect(expr).toContain("0.5");
        expect(expr).toContain("clamp"); // Overflow protection
      });

      it("generates silu expression", () => {
        const expr = getExpr("silu", ["x"]);
        expect(expr).toContain("exp");
        expect(expr).toContain("1.0");
      });

      it("generates sigmoid expression", () => {
        const expr = getExpr("sigmoid", ["x"]);
        expect(expr).toContain("exp");
        expect(expr).toContain("1.0");
      });

      it("generates tanh expression", () => {
        expect(getExpr("tanh", ["x"])).toBe("tanh(x)");
      });

      it("generates rsqrt expression", () => {
        expect(getExpr("rsqrt", ["x"])).toBe("inverseSqrt(x)");
      });

      it("generates trig expressions", () => {
        expect(getExpr("sin", ["x"])).toBe("sin(x)");
        expect(getExpr("cos", ["x"])).toBe("cos(x)");
      });

      it("generates rounding expressions", () => {
        expect(getExpr("floor", ["x"])).toBe("floor(x)");
        expect(getExpr("ceil", ["x"])).toBe("ceil(x)");
        expect(getExpr("round", ["x"])).toBe("round(x)");
      });
    });

    describe("binary ops", () => {
      it("generates arithmetic expressions", () => {
        expect(getExpr("add", ["a", "b"])).toBe("(a + b)");
        expect(getExpr("sub", ["a", "b"])).toBe("(a - b)");
        expect(getExpr("mul", ["a", "b"])).toBe("(a * b)");
        expect(getExpr("div", ["a", "b"])).toBe("(a / b)");
        expect(getExpr("mod", ["a", "b"])).toBe("(a % b)");
      });

      it("generates pow expression", () => {
        expect(getExpr("pow", ["a", "b"])).toBe("pow(a, b)");
      });

      it("generates min/max expressions", () => {
        expect(getExpr("min", ["a", "b"])).toBe("min(a, b)");
        expect(getExpr("max", ["a", "b"])).toBe("max(a, b)");
      });

      it("generates comparison expressions with select", () => {
        expect(getExpr("eq", ["a", "b"])).toBe("select(0.0, 1.0, a == b)");
        expect(getExpr("ne", ["a", "b"])).toBe("select(0.0, 1.0, a != b)");
        expect(getExpr("lt", ["a", "b"])).toBe("select(0.0, 1.0, a < b)");
        expect(getExpr("le", ["a", "b"])).toBe("select(0.0, 1.0, a <= b)");
        expect(getExpr("gt", ["a", "b"])).toBe("select(0.0, 1.0, a > b)");
        expect(getExpr("ge", ["a", "b"])).toBe("select(0.0, 1.0, a >= b)");
      });
    });

    describe("ternary ops", () => {
      it("generates where expression", () => {
        expect(getExpr("where", ["cond", "a", "b"])).toBe("select(b, a, cond > 0.0)");
      });
    });

    describe("cast ops", () => {
      it("generates cast expressions", () => {
        expect(getExpr("cast_f16", ["x"])).toBe("f16(x)");
        expect(getExpr("cast_f32", ["x"])).toBe("f32(x)");
        expect(getExpr("cast_i32", ["x"])).toBe("i32(x)");
        expect(getExpr("cast_u32", ["x"])).toBe("u32(x)");
      });
    });

    describe("vector constants", () => {
      it("uses custom zero for relu when provided", () => {
        const expr = getExpr("relu", ["v"], { zero: "vec4<f32>(0.0)", one: "vec4<f32>(1.0)" });
        expect(expr).toContain("vec4<f32>(0.0)");
      });
    });

    describe("error handling", () => {
      it("throws for unknown ops", () => {
        expect(() => getExpr("unknown", ["x"])).toThrow("Unknown op in registry: unknown");
      });

      it("throws for insufficient inputs", () => {
        expect(() => getExpr("add", ["a"])).toThrow("requires 2 inputs");
        expect(() => getExpr("where", ["a", "b"])).toThrow("requires 3 inputs");
      });
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

    it("correctly classifies binary ops", () => {
      expect(isBinaryOp("add")).toBe(true);
      expect(isBinaryOp("mul")).toBe(true);
      expect(isBinaryOp("lt")).toBe(true);
      expect(isBinaryOp("pow")).toBe(true);
      expect(isBinaryOp("relu")).toBe(false);
      expect(isBinaryOp("where")).toBe(false);
    });

    it("correctly classifies ternary ops", () => {
      expect(isTernaryOp("where")).toBe(true);
      expect(isTernaryOp("relu")).toBe(false);
      expect(isTernaryOp("add")).toBe(false);
    });

    it("returns correct arity values", () => {
      expect(getArity("relu")).toBe(1);
      expect(getArity("add")).toBe(2);
      expect(getArity("where")).toBe(3);
      expect(getArity("unknown")).toBeNull();
    });
  });

  describe("fusibility", () => {
    it("all registered ops are fusible", () => {
      for (const op of Object.keys(OP_REGISTRY)) {
        expect(isFusible(op), `Op ${op} should be fusible`).toBe(true);
      }
    });

    it("unknown ops are not fusible", () => {
      expect(isFusible("matmul")).toBe(false);
      expect(isFusible("conv2d")).toBe(false);
      expect(isFusible("unknown")).toBe(false);
    });
  });

  describe("vectorization", () => {
    it("most ops are vectorizable", () => {
      const vectorizable = ["relu", "add", "mul", "sigmoid", "tanh", "neg", "sqrt"];
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

  describe("listing functions", () => {
    it("getOpsByCategory returns correct ops", () => {
      const activations = getOpsByCategory("activation");
      expect(activations).toContain("relu");
      expect(activations).toContain("gelu");
      expect(activations).toContain("silu");
      expect(activations).not.toContain("add");

      const arithmetic = getOpsByCategory("arithmetic");
      expect(arithmetic).toContain("add");
      expect(arithmetic).toContain("mul");
      expect(arithmetic).not.toContain("relu");
    });

    it("getAllFusibleOps returns all ops", () => {
      const fusible = getAllFusibleOps();
      expect(fusible.length).toBe(Object.keys(OP_REGISTRY).length);
    });

    it("getAllUnaryOps returns only unary ops", () => {
      const unary = getAllUnaryOps();
      for (const op of unary) {
        expect(getArity(op)).toBe(1);
      }
      expect(unary).toContain("relu");
      expect(unary).toContain("sqrt");
      expect(unary).not.toContain("add");
    });

    it("getAllBinaryOps returns only binary ops", () => {
      const binary = getAllBinaryOps();
      for (const op of binary) {
        expect(getArity(op)).toBe(2);
      }
      expect(binary).toContain("add");
      expect(binary).toContain("mul");
      expect(binary).not.toContain("relu");
    });
  });

  describe("backward compatibility exports", () => {
    it("UNARY_EXPR contains all unary ops", () => {
      expect(UNARY_EXPR.relu).toBeDefined();
      expect(UNARY_EXPR.gelu).toBeDefined();
      expect(UNARY_EXPR.silu).toBeDefined();
      expect(UNARY_EXPR.neg).toBeDefined();
      expect(UNARY_EXPR.sqrt).toBeDefined();
      expect(UNARY_EXPR.cast_f16).toBeDefined();
    });

    it("BINARY_EXPR contains all binary ops", () => {
      expect(BINARY_EXPR.add).toBeDefined();
      expect(BINARY_EXPR.sub).toBeDefined();
      expect(BINARY_EXPR.mul).toBeDefined();
      expect(BINARY_EXPR.div).toBeDefined();
      expect(BINARY_EXPR.eq).toBeDefined();
      expect(BINARY_EXPR.lt).toBeDefined();
    });

    it("UNARY_EXPR generates same output as getExpr", () => {
      expect(UNARY_EXPR.relu("x")).toBe(getExpr("relu", ["x"]));
      expect(UNARY_EXPR.neg("x")).toBe(getExpr("neg", ["x"]));
      expect(UNARY_EXPR.sqrt("x")).toBe(getExpr("sqrt", ["x"]));
    });

    it("BINARY_EXPR generates same output as getExpr", () => {
      expect(BINARY_EXPR.add("a", "b")).toBe(getExpr("add", ["a", "b"]));
      expect(BINARY_EXPR.mul("a", "b")).toBe(getExpr("mul", ["a", "b"]));
      expect(BINARY_EXPR.eq("a", "b")).toBe(getExpr("eq", ["a", "b"]));
    });
  });

  describe("consistency checks", () => {
    it("all ops have valid arity values", () => {
      for (const [name, def] of Object.entries(OP_REGISTRY)) {
        expect([1, 2, 3]).toContain(def.arity);
      }
    });

    it("all ops have valid category", () => {
      const validCategories = ["activation", "math", "arithmetic", "comparison", "ternary", "cast"];
      for (const [name, def] of Object.entries(OP_REGISTRY)) {
        expect(validCategories).toContain(def.category);
      }
    });

    it("all ops have boolean fusible and vectorizable flags", () => {
      for (const [name, def] of Object.entries(OP_REGISTRY)) {
        expect(typeof def.fusible).toBe("boolean");
        expect(typeof def.vectorizable).toBe("boolean");
      }
    });

    it("all comparison ops have outputDtype = f32", () => {
      const comparisons = getOpsByCategory("comparison");
      for (const op of comparisons) {
        expect(getOpDef(op)?.outputDtype).toBe("f32");
      }
    });
  });
});
