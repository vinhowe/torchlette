import { describe, expect, it } from "vitest";

import {
  UNARY_EXPR,
  BINARY_EXPR,
  isUnaryOp,
  isBinaryOp,
  isFusibleOp,
  getExprGenerator,
  generateFusedExpressions,
  generateFusedKernel,
  genBroadcastIndex,
  needsBroadcast,
  buildRecipeFromIR,
  generateKernelCacheKey,
  selectVectorWidth,
  canVectorize,
  getVectorType,
  genSplat,
  type FusedKernelRecipe,
  type FusedNode,
  type VectorWidth,
} from "../../src/backend/webgpu/fusion-codegen";
import type { IRNode } from "../../src/engine/ir";

describe("Fusion Codegen", () => {
  describe("Expression Templates", () => {
    describe("Unary operations", () => {
      it("generates relu expression", () => {
        // relu uses select() for vector compatibility
        expect(UNARY_EXPR.relu("x")).toBe("select(0.0, x, x > 0.0)");
      });

      it("generates gelu expression", () => {
        const expr = UNARY_EXPR.gelu("x");
        expect(expr).toContain("tanh");
        expect(expr).toContain("0.5");
      });

      it("generates silu expression", () => {
        const expr = UNARY_EXPR.silu("x");
        expect(expr).toContain("exp");
      });

      it("generates neg expression", () => {
        expect(UNARY_EXPR.neg("x")).toBe("(-x)");
      });

      it("generates sqrt expression", () => {
        expect(UNARY_EXPR.sqrt("x")).toBe("sqrt(x)");
      });

      it("generates exp expression", () => {
        expect(UNARY_EXPR.exp("x")).toBe("exp(x)");
      });

      it("generates cast expressions", () => {
        expect(UNARY_EXPR.cast_f16("x")).toBe("f16(x)");
        expect(UNARY_EXPR.cast_f32("x")).toBe("f32(x)");
        expect(UNARY_EXPR.cast_i32("x")).toBe("i32(x)");
      });
    });

    describe("Binary operations", () => {
      it("generates add expression", () => {
        expect(BINARY_EXPR.add("a", "b")).toBe("(a + b)");
      });

      it("generates sub expression", () => {
        expect(BINARY_EXPR.sub("a", "b")).toBe("(a - b)");
      });

      it("generates mul expression", () => {
        expect(BINARY_EXPR.mul("a", "b")).toBe("(a * b)");
      });

      it("generates div expression", () => {
        expect(BINARY_EXPR.div("a", "b")).toBe("(a / b)");
      });

      it("generates comparison expressions", () => {
        expect(BINARY_EXPR.eq("a", "b")).toContain("select");
        expect(BINARY_EXPR.lt("a", "b")).toContain("<");
        expect(BINARY_EXPR.gt("a", "b")).toContain(">");
      });
    });
  });

  describe("Operation classification", () => {
    it("identifies unary ops", () => {
      expect(isUnaryOp("relu")).toBe(true);
      expect(isUnaryOp("sqrt")).toBe(true);
      expect(isUnaryOp("neg")).toBe(true);
      expect(isUnaryOp("cast_f16")).toBe(true);
      expect(isUnaryOp("add")).toBe(false);
    });

    it("identifies binary ops", () => {
      expect(isBinaryOp("add")).toBe(true);
      expect(isBinaryOp("mul")).toBe(true);
      expect(isBinaryOp("sub")).toBe(true);
      expect(isBinaryOp("relu")).toBe(false);
    });

    it("identifies fusible ops", () => {
      expect(isFusibleOp("relu")).toBe(true);
      expect(isFusibleOp("add")).toBe(true);
      expect(isFusibleOp("matmul")).toBe(false);
      expect(isFusibleOp("conv2d")).toBe(false);
    });
  });

  describe("Expression generator", () => {
    it("returns generator for unary ops", () => {
      const gen = getExprGenerator("relu");
      expect(gen).not.toBeNull();
      // relu uses select() for vector compatibility
      expect(gen!(["x"])).toBe("select(0.0, x, x > 0.0)");
    });

    it("returns generator for binary ops", () => {
      const gen = getExprGenerator("add");
      expect(gen).not.toBeNull();
      expect(gen!(["a", "b"])).toBe("(a + b)");
    });

    it("returns null for non-fusible ops", () => {
      expect(getExprGenerator("matmul")).toBeNull();
      expect(getExprGenerator("unknown")).toBeNull();
    });
  });

  describe("Broadcasting", () => {
    it("detects when broadcasting is needed", () => {
      expect(needsBroadcast([10, 10], [10, 10])).toBe(false);
      expect(needsBroadcast([10, 10], [1, 10])).toBe(true);
      expect(needsBroadcast([10, 10], [10])).toBe(true);
      expect(needsBroadcast([10, 10], [1])).toBe(true);
    });

    it("generates simple index for same shapes", () => {
      const code = genBroadcastIndex([10, 10], [10, 10], "idx", "idx0");
      expect(code).toBe("let idx0 = idx;");
    });

    it("generates zero index for scalar", () => {
      const code = genBroadcastIndex([10, 10], [1], "idx", "idx0");
      expect(code).toBe("let idx0 = 0u;");
    });

    it("generates broadcast code for different shapes", () => {
      const code = genBroadcastIndex([10, 10], [1, 10], "idx", "idx0");
      expect(code).toContain("_tmp_idx0");
      expect(code).toContain("let idx0");
    });
  });

  describe("SSA Expression Generation", () => {
    it("generates expressions for simple chain", () => {
      const recipe: FusedKernelRecipe = {
        id: "test",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [10], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [10], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [10], dtype: "f32" },
          { id: 101, index: 1, shape: [10], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [10], dtype: "f32" }],
      };

      const result = generateFusedExpressions(recipe);

      expect(result.lines.length).toBeGreaterThan(0);
      expect(result.outputVar).toBeDefined();
    });

    it("inlines single-use expressions", () => {
      const recipe: FusedKernelRecipe = {
        id: "test",
        nodes: [
          { id: 1, op: "neg", inputs: [-1], shape: [10], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [10], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [10], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [10], dtype: "f32" }],
      };

      const result = generateFusedExpressions(recipe);

      // The neg should be inlined, so we should have only 1 line (for relu)
      // or the final expression might combine both
      expect(result.outputVar).toBeDefined();
    });

    it("creates variables for multi-use expressions", () => {
      const recipe: FusedKernelRecipe = {
        id: "test",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [10], dtype: "f32" },
          // Both nodes 2 and 3 use node 1
          { id: 2, op: "neg", inputs: [1], shape: [10], dtype: "f32" },
          { id: 3, op: "add", inputs: [1, 2], shape: [10], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [10], dtype: "f32" },
        ],
        outputs: [{ nodeId: 3, index: 0, shape: [10], dtype: "f32" }],
      };

      const result = generateFusedExpressions(recipe);

      // Node 1 should have a variable since it's used twice
      expect(result.varNames.has(1)).toBe(true);
      expect(result.lines.some(l => l.includes("t1"))).toBe(true);
    });
  });

  describe("Kernel Generation", () => {
    it("generates complete WGSL shader", () => {
      const recipe: FusedKernelRecipe = {
        id: "add_relu",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [100], dtype: "f32" },
          { id: 2, op: "relu", inputs: [1], shape: [100], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [100], dtype: "f32" },
          { id: 101, index: 1, shape: [100], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [100], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe);

      // Check basic structure
      expect(kernel.source).toContain("@compute");
      expect(kernel.source).toContain("@workgroup_size");
      expect(kernel.source).toContain("fn main");

      // Check bindings
      expect(kernel.source).toContain("@binding(0)");
      expect(kernel.source).toContain("@binding(1)");
      expect(kernel.source).toContain("var<storage, read> in0");
      expect(kernel.source).toContain("var<storage, read> in1");
      expect(kernel.source).toContain("var<storage, read_write> out0");

      // Check params
      expect(kernel.source).toContain("params.total_elements");

      // Check output
      expect(kernel.source).toContain("out0[idx]");

      expect(kernel.workgroupSize).toBe(256);
      expect(kernel.inputBindings).toBe(2);
    });

    it("includes operation chain in comments", () => {
      const recipe: FusedKernelRecipe = {
        id: "chain",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [10], dtype: "f32" },
          { id: 2, op: "neg", inputs: [1], shape: [10], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [10], dtype: "f32" },
        ],
        outputs: [{ nodeId: 2, index: 0, shape: [10], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe);

      expect(kernel.source).toContain("relu");
      expect(kernel.source).toContain("neg");
    });

    it("handles f16 output dtype", () => {
      const recipe: FusedKernelRecipe = {
        id: "cast_test",
        nodes: [
          { id: 1, op: "cast_f16", inputs: [-1], shape: [10], dtype: "f16", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [10], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [10], dtype: "f16" }],
      };

      const kernel = generateFusedKernel(recipe);

      expect(kernel.source).toContain("array<f16>");
    });
  });

  describe("Cache Key Generation", () => {
    it("generates unique keys for different recipes", () => {
      const recipe1: FusedKernelRecipe = {
        id: "r1",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [10], dtype: "f32" }],
        inputs: [{ id: 100, index: 0, shape: [10], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [10], dtype: "f32" }],
      };

      const recipe2: FusedKernelRecipe = {
        id: "r2",
        nodes: [{ id: 1, op: "neg", inputs: [-1], shape: [10], dtype: "f32" }],
        inputs: [{ id: 100, index: 0, shape: [10], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [10], dtype: "f32" }],
      };

      const key1 = generateKernelCacheKey(recipe1);
      const key2 = generateKernelCacheKey(recipe2);

      expect(key1).not.toBe(key2);
    });

    it("generates same key for equivalent recipes", () => {
      const recipe1: FusedKernelRecipe = {
        id: "test",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [10], dtype: "f32" }],
        inputs: [{ id: 100, index: 0, shape: [10], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [10], dtype: "f32" }],
      };

      const recipe2: FusedKernelRecipe = {
        id: "test",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [10], dtype: "f32" }],
        inputs: [{ id: 100, index: 0, shape: [10], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [10], dtype: "f32" }],
      };

      const key1 = generateKernelCacheKey(recipe1);
      const key2 = generateKernelCacheKey(recipe2);

      expect(key1).toBe(key2);
    });
  });

  describe("Recipe Building from IR", () => {
    it("builds recipe from IR nodes", () => {
      const nodeById = new Map<number, IRNode>([
        [1, { id: 1, op: "input", inputs: [], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [2, { id: 2, op: "input", inputs: [], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [3, { id: 3, op: "add", inputs: [1, 2], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [4, { id: 4, op: "relu", inputs: [3], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
      ]);

      const recipe = buildRecipeFromIR(
        [3, 4], // Nodes to fuse
        nodeById,
        [1, 2], // External inputs
      );

      expect(recipe.nodes).toHaveLength(2);
      expect(recipe.inputs).toHaveLength(2);
      expect(recipe.outputs).toHaveLength(1);
      expect(recipe.outputs[0].nodeId).toBe(4);
      expect(recipe.nodes[0].op).toBe("add");
      expect(recipe.nodes[1].op).toBe("relu");
      expect(recipe.nodes[1].isOutput).toBe(true);
    });

    it("handles nodes with mixed internal and external inputs", () => {
      const nodeById = new Map<number, IRNode>([
        [1, { id: 1, op: "input", inputs: [], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [2, { id: 2, op: "relu", inputs: [1], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [3, { id: 3, op: "input", inputs: [], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [4, { id: 4, op: "add", inputs: [2, 3], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
      ]);

      const recipe = buildRecipeFromIR(
        [2, 4], // Nodes to fuse
        nodeById,
        [1], // Only node 1 is known external
      );

      // Node 3 should be auto-discovered as external
      expect(recipe.inputs.length).toBe(2);
      expect(recipe.inputs.some(i => i.id === 1)).toBe(true);
      expect(recipe.inputs.some(i => i.id === 3)).toBe(true);
    });

    it("builds multi-output recipe from IR nodes (§15.2)", () => {
      const nodeById = new Map<number, IRNode>([
        [1, { id: 1, op: "input", inputs: [], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [2, { id: 2, op: "relu", inputs: [1], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
        [3, { id: 3, op: "neg", inputs: [2], shape: [10], dtype: "f32", epoch: 1, kind: "lazy_op" }],
      ]);

      const recipe = buildRecipeFromIR(
        [2, 3], // Nodes to fuse
        nodeById,
        [1], // External inputs
        [2, 3], // Multiple outputs
      );

      expect(recipe.outputs).toHaveLength(2);
      expect(recipe.outputs[0].nodeId).toBe(2);
      expect(recipe.outputs[1].nodeId).toBe(3);
      expect(recipe.nodes.filter(n => n.isOutput)).toHaveLength(2);
    });
  });
});

describe("Vectorization (§15.3)", () => {
  describe("selectVectorWidth", () => {
    it("selects vec4 when innermost dim divisible by 4", () => {
      expect(selectVectorWidth([100, 64], "f32")).toBe(4);
      expect(selectVectorWidth([256], "f32")).toBe(4);
      expect(selectVectorWidth([10, 20, 128], "f32")).toBe(4);
    });

    it("selects vec2 when innermost dim divisible by 2 but not 4", () => {
      expect(selectVectorWidth([100, 66], "f32")).toBe(2);
      expect(selectVectorWidth([50], "f32")).toBe(2);
      expect(selectVectorWidth([10, 20, 22], "f32")).toBe(2);
    });

    it("selects scalar for odd innermost dim", () => {
      expect(selectVectorWidth([100, 33], "f32")).toBe(1);
      expect(selectVectorWidth([17], "f32")).toBe(1);
    });

    it("selects scalar for tiny tensors", () => {
      expect(selectVectorWidth([4], "f32")).toBe(1); // Only 4 elements
      expect(selectVectorWidth([2, 2], "f32")).toBe(1); // Only 4 elements
    });

    it("selects scalar for non-float dtypes", () => {
      expect(selectVectorWidth([256], "i32")).toBe(1);
      expect(selectVectorWidth([256], "u32")).toBe(1);
      expect(selectVectorWidth([256], "bool")).toBe(1);
    });

    it("supports f16 vectorization", () => {
      expect(selectVectorWidth([256], "f16")).toBe(4);
      expect(selectVectorWidth([66], "f16")).toBe(2);
    });

    it("handles empty shape", () => {
      expect(selectVectorWidth([], "f32")).toBe(1);
    });
  });

  describe("canVectorize", () => {
    it("returns true when all inputs have matching innermost dims", () => {
      expect(canVectorize(
        [100, 64],
        [
          { shape: [100, 64], dtype: "f32" },
          { shape: [100, 64], dtype: "f32" },
        ],
        4,
      )).toBe(true);
    });

    it("returns true when input is scalar (will be splatted)", () => {
      expect(canVectorize(
        [100, 64],
        [
          { shape: [100, 64], dtype: "f32" },
          { shape: [1], dtype: "f32" }, // Scalar
        ],
        4,
      )).toBe(true);
    });

    it("returns true for broadcast with innermost dim = 1", () => {
      expect(canVectorize(
        [100, 64],
        [
          { shape: [100, 64], dtype: "f32" },
          { shape: [100, 1], dtype: "f32" }, // Broadcasts inner dim
        ],
        4,
      )).toBe(true);
    });

    it("returns false when innermost dims mismatch and neither is 1", () => {
      expect(canVectorize(
        [100, 64],
        [
          { shape: [100, 64], dtype: "f32" },
          { shape: [100, 32], dtype: "f32" }, // Different inner dim
        ],
        4,
      )).toBe(false);
    });

    it("returns false when output not divisible by vector width", () => {
      expect(canVectorize(
        [100, 33],
        [{ shape: [100, 33], dtype: "f32" }],
        4,
      )).toBe(false);
    });

    it("always returns true for scalar width", () => {
      expect(canVectorize(
        [100, 33],
        [{ shape: [100, 33], dtype: "f32" }],
        1,
      )).toBe(true);
    });
  });

  describe("getVectorType", () => {
    it("returns scalar type for width 1", () => {
      expect(getVectorType("f32", 1)).toBe("f32");
      expect(getVectorType("f16", 1)).toBe("f16");
    });

    it("returns vec2 type for width 2", () => {
      expect(getVectorType("f32", 2)).toBe("vec2<f32>");
      expect(getVectorType("f16", 2)).toBe("vec2<f16>");
    });

    it("returns vec4 type for width 4", () => {
      expect(getVectorType("f32", 4)).toBe("vec4<f32>");
      expect(getVectorType("f16", 4)).toBe("vec4<f16>");
    });
  });

  describe("genSplat", () => {
    it("returns expression unchanged for width 1", () => {
      expect(genSplat("val", "f32", 1)).toBe("val");
    });

    it("generates vec2 splat for width 2", () => {
      expect(genSplat("val", "f32", 2)).toBe("vec2<f32>(val)");
    });

    it("generates vec4 splat for width 4", () => {
      expect(genSplat("val", "f32", 4)).toBe("vec4<f32>(val)");
    });
  });

  describe("Kernel generation with vectorization", () => {
    it("generates vec4 kernel when conditions met", () => {
      const recipe: FusedKernelRecipe = {
        id: "vec4_test",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [256], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [256], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(4);
      expect(kernel.workItems).toBe(64); // 256 / 4
      expect(kernel.source).toContain("vec4");
      expect(kernel.source).toContain(".x");
      expect(kernel.source).toContain(".y");
      expect(kernel.source).toContain(".z");
      expect(kernel.source).toContain(".w");
    });

    it("generates vec2 kernel for appropriate shapes", () => {
      const recipe: FusedKernelRecipe = {
        id: "vec2_test",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [66], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [66], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [66], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(2);
      expect(kernel.workItems).toBe(33); // 66 / 2
      expect(kernel.source).toContain("vec2");
    });

    it("generates scalar kernel when vectorization disabled", () => {
      const recipe: FusedKernelRecipe = {
        id: "scalar_test",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [256], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [256], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: false });

      expect(kernel.vectorWidth).toBe(1);
      expect(kernel.workItems).toBe(256);
      expect(kernel.source).not.toContain("vec4<f32>");
    });

    it("generates scalar kernel for odd shapes", () => {
      const recipe: FusedKernelRecipe = {
        id: "odd_test",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [33], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [33], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [33], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(1);
      expect(kernel.workItems).toBe(33);
    });

    it("handles scalar input with splat in vectorized kernel", () => {
      const recipe: FusedKernelRecipe = {
        id: "splat_test",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [256], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [256], dtype: "f32" },
          { id: 101, index: 1, shape: [1], dtype: "f32" }, // Scalar
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(4);
      // Scalar should be splatted to vec4
      expect(kernel.source).toContain("vec4<f32>(in1[0])");
    });

    it("handles broadcasting with vectorization", () => {
      const recipe: FusedKernelRecipe = {
        id: "broadcast_vec_test",
        nodes: [
          { id: 1, op: "add", inputs: [-1, -2], shape: [10, 64], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [10, 64], dtype: "f32" },
          { id: 101, index: 1, shape: [1, 64], dtype: "f32" }, // Broadcast first dim
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [10, 64], dtype: "f32" }],
      };

      const kernel = generateFusedKernel(recipe, { vectorize: true });

      expect(kernel.vectorWidth).toBe(4);
      // Should have broadcast index calculation
      expect(kernel.source).toContain("bc1_0"); // Broadcast index for input 1
    });

    it("includes vector width in cache key", () => {
      const recipe: FusedKernelRecipe = {
        id: "cache_test",
        nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [256], dtype: "f32" }],
        inputs: [{ id: 100, index: 0, shape: [256], dtype: "f32" }],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      const key1 = generateKernelCacheKey(recipe, 1);
      const key4 = generateKernelCacheKey(recipe, 4);

      expect(key1).toContain("vec:1");
      expect(key4).toContain("vec:4");
      expect(key1).not.toBe(key4);
    });

    it("respects forceVectorWidth option", () => {
      const recipe: FusedKernelRecipe = {
        id: "force_test",
        nodes: [
          { id: 1, op: "relu", inputs: [-1], shape: [256], dtype: "f32", isOutput: true },
        ],
        inputs: [
          { id: 100, index: 0, shape: [256], dtype: "f32" },
        ],
        outputs: [{ nodeId: 1, index: 0, shape: [256], dtype: "f32" }],
      };

      // Force vec2 even though vec4 would be auto-selected
      const kernel = generateFusedKernel(recipe, { vectorize: true, forceVectorWidth: 2 });

      expect(kernel.vectorWidth).toBe(2);
      expect(kernel.workItems).toBe(128); // 256 / 2
    });
  });
});

describe("Fusion Dispatch", () => {
  // Note: Full dispatch tests require WebGPU, so these are unit tests
  // for the helper functions and cache

  it("cache key includes shape information", () => {
    const recipe1: FusedKernelRecipe = {
      id: "test",
      nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [10], dtype: "f32" }],
      inputs: [{ id: 100, index: 0, shape: [10], dtype: "f32" }],
      outputs: [{ nodeId: 1, index: 0, shape: [10], dtype: "f32" }],
    };

    const recipe2: FusedKernelRecipe = {
      id: "test",
      nodes: [{ id: 1, op: "relu", inputs: [-1], shape: [20], dtype: "f32" }],
      inputs: [{ id: 100, index: 0, shape: [20], dtype: "f32" }],
      outputs: [{ nodeId: 1, index: 0, shape: [20], dtype: "f32" }],
    };

    const key1 = generateKernelCacheKey(recipe1);
    const key2 = generateKernelCacheKey(recipe2);

    expect(key1).not.toBe(key2);
    expect(key1).toContain("10");
    expect(key2).toContain("20");
  });
});

describe("Multi-Output Fusion (§15.2)", () => {
  it("generates kernel with multiple output bindings", () => {
    const recipe: FusedKernelRecipe = {
      id: "multi_output",
      nodes: [
        { id: 1, op: "relu", inputs: [-1], shape: [100], dtype: "f32", isOutput: true },
        { id: 2, op: "neg", inputs: [1], shape: [100], dtype: "f32", isOutput: true },
      ],
      inputs: [
        { id: 100, index: 0, shape: [100], dtype: "f32" },
      ],
      outputs: [
        { nodeId: 1, index: 0, shape: [100], dtype: "f32" },
        { nodeId: 2, index: 1, shape: [100], dtype: "f32" },
      ],
    };

    const kernel = generateFusedKernel(recipe);

    // Check for two output bindings
    expect(kernel.source).toContain("var<storage, read_write> out0");
    expect(kernel.source).toContain("var<storage, read_write> out1");
    expect(kernel.source).toContain("out0[idx]");
    expect(kernel.source).toContain("out1[idx]");
    // Comment should show 2 outputs
    expect(kernel.source).toContain("Outputs: 2");
  });

  it("cache key differentiates multi-output from single-output", () => {
    const singleOutput: FusedKernelRecipe = {
      id: "test",
      nodes: [
        { id: 1, op: "relu", inputs: [-1], shape: [100], dtype: "f32", isOutput: true },
      ],
      inputs: [{ id: 100, index: 0, shape: [100], dtype: "f32" }],
      outputs: [{ nodeId: 1, index: 0, shape: [100], dtype: "f32" }],
    };

    const multiOutput: FusedKernelRecipe = {
      id: "test",
      nodes: [
        { id: 1, op: "relu", inputs: [-1], shape: [100], dtype: "f32", isOutput: true },
        { id: 2, op: "neg", inputs: [1], shape: [100], dtype: "f32", isOutput: true },
      ],
      inputs: [{ id: 100, index: 0, shape: [100], dtype: "f32" }],
      outputs: [
        { nodeId: 1, index: 0, shape: [100], dtype: "f32" },
        { nodeId: 2, index: 1, shape: [100], dtype: "f32" },
      ],
    };

    const key1 = generateKernelCacheKey(singleOutput);
    const key2 = generateKernelCacheKey(multiOutput);

    // Keys should be different due to different output count
    expect(key1).not.toBe(key2);
    expect(key1).toContain("out0:");
    expect(key2).toContain("out1:");
  });

  it("handles mixed output dtypes", () => {
    const recipe: FusedKernelRecipe = {
      id: "mixed_dtype",
      nodes: [
        { id: 1, op: "relu", inputs: [-1], shape: [100], dtype: "f32", isOutput: true },
        { id: 2, op: "cast_f16", inputs: [1], shape: [100], dtype: "f16", isOutput: true },
      ],
      inputs: [
        { id: 100, index: 0, shape: [100], dtype: "f32" },
      ],
      outputs: [
        { nodeId: 1, index: 0, shape: [100], dtype: "f32" },
        { nodeId: 2, index: 1, shape: [100], dtype: "f16" },
      ],
    };

    const kernel = generateFusedKernel(recipe);

    // Both output types should be present
    expect(kernel.source).toContain("out0: array<f32>");
    expect(kernel.source).toContain("out1: array<f16>");
  });
});
