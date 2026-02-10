import { beforeEach, describe, expect, it } from "vitest";

import {
  createAutocastContext,
  pushAutocast,
  type AutocastContext,
} from "../src/engine/amp";
import {
  applyAMPTransform,
  getAMPStats,
  isAMPTransformed,
} from "../src/engine/amp-ir-transform";
import type { IRGraph, IRNode } from "../src/engine/ir";

function createTestGraph(nodes: Partial<IRNode>[]): IRGraph {
  return {
    epoch: 1,
    nodes: nodes.map((n, i) => ({
      id: n.id ?? i + 1,
      op: n.op ?? "add",
      epoch: 1,
      kind: "lazy_op" as const,
      inputs: n.inputs ?? [],
      shape: n.shape ?? [2, 2],
      dtype: n.dtype ?? "f32",
    })),
    fusionGroups: [],
  };
}

describe("AMP IR Transform", () => {
  let ctx: AutocastContext;

  beforeEach(() => {
    ctx = createAutocastContext();
  });

  describe("with autocast disabled", () => {
    it("returns unchanged graph", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [] },
        { id: 2, op: "input", inputs: [] },
        { id: 3, op: "matmul", inputs: [1, 2] },
      ]);

      const result = applyAMPTransform(graph, ctx);

      expect(result.modified).toBe(false);
      expect(result.castsInserted).toBe(0);
      expect(result.graph).toBe(graph);
    });
  });

  describe("with autocast enabled", () => {
    beforeEach(() => {
      pushAutocast(ctx, { enabled: true });
    });

    it("inserts casts for matmul with f32 inputs", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f32" },
        { id: 2, op: "input", inputs: [], dtype: "f32" },
        { id: 3, op: "matmul", inputs: [1, 2], dtype: "f32" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      expect(result.modified).toBe(true);
      expect(result.castsInserted).toBe(2); // Both inputs cast to f16

      // Find cast nodes
      const castNodes = result.graph.nodes.filter((n) => n.op === "cast");
      expect(castNodes).toHaveLength(2);
      expect(castNodes[0].dtype).toBe("f16");
      expect(castNodes[1].dtype).toBe("f16");
    });

    it("does not insert casts for already f16 inputs", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f16" },
        { id: 2, op: "input", inputs: [], dtype: "f16" },
        { id: 3, op: "matmul", inputs: [1, 2], dtype: "f16" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      // No casts needed since inputs are already f16
      expect(result.castsInserted).toBe(0);
      expect(result.modified).toBe(false);
    });

    it("inserts f32 cast for sum with f16 input", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f16" },
        { id: 2, op: "sum", inputs: [1], dtype: "f16" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      expect(result.castsInserted).toBe(1);

      const castNode = result.graph.nodes.find((n) => n.op === "cast");
      expect(castNode).toBeDefined();
      expect(castNode?.dtype).toBe("f32");
    });

    it("handles mixed inputs correctly", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f32" },
        { id: 2, op: "input", inputs: [], dtype: "f16" },
        { id: 3, op: "matmul", inputs: [1, 2], dtype: "f32" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      // Only the f32 input needs a cast
      expect(result.castsInserted).toBe(1);

      const castNode = result.graph.nodes.find((n) => n.op === "cast");
      expect(castNode).toBeDefined();
      expect(castNode?.dtype).toBe("f16");
      expect(castNode?.inputs[0]).toBe(1); // Casts node 1 (f32)
    });

    it("preserves non-eligible ops", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f32" },
        { id: 2, op: "input", inputs: [], dtype: "f32" },
        { id: 3, op: "add", inputs: [1, 2], dtype: "f32" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      // add is not f16-eligible and doesn't require f32
      expect(result.castsInserted).toBe(0);
    });

    it("remaps inputs correctly through cast nodes", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f32" },
        { id: 2, op: "input", inputs: [], dtype: "f32" },
        { id: 3, op: "matmul", inputs: [1, 2], dtype: "f32" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      // Find the matmul node in the transformed graph
      const matmulNode = result.graph.nodes.find((n) => n.op === "matmul");
      expect(matmulNode).toBeDefined();

      // Its inputs should now be the cast nodes, not the original inputs
      const castNodes = result.graph.nodes.filter((n) => n.op === "cast");
      expect(matmulNode?.inputs).toContain(castNodes[0].id);
      expect(matmulNode?.inputs).toContain(castNodes[1].id);
    });

    it("handles linear op (f16-eligible)", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f32" },
        { id: 2, op: "weight", inputs: [], dtype: "f32" },
        { id: 3, op: "linear", inputs: [1, 2], dtype: "f32" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      expect(result.castsInserted).toBe(2); // Both inputs cast to f16
    });

    it("handles mean op (f32-required)", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f16" },
        { id: 2, op: "mean", inputs: [1], dtype: "f16" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      expect(result.castsInserted).toBe(1);

      const castNode = result.graph.nodes.find((n) => n.op === "cast");
      expect(castNode?.dtype).toBe("f32");
    });

    it("handles softmax op (f32-required)", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f16" },
        { id: 2, op: "softmax", inputs: [1], dtype: "f16" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      expect(result.castsInserted).toBe(1);

      const castNode = result.graph.nodes.find((n) => n.op === "cast");
      expect(castNode?.dtype).toBe("f32");
    });
  });

  describe("select-gated output dtype", () => {
    beforeEach(() => {
      pushAutocast(ctx, { enabled: true });
    });

    it("matmul outputs f16 when AMP enabled", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f32" },
        { id: 2, op: "input", inputs: [], dtype: "f32" },
        { id: 3, op: "matmul", inputs: [1, 2], dtype: "f32" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      const matmulNode = result.graph.nodes.find((n) => n.op === "matmul");
      expect(matmulNode?.dtype).toBe("f16");
    });

    it("sum outputs f32 even with f16 input", () => {
      const graph = createTestGraph([
        { id: 1, op: "input", inputs: [], dtype: "f16" },
        { id: 2, op: "sum", inputs: [1], dtype: "f16" },
      ]);

      const result = applyAMPTransform(graph, ctx);

      const sumNode = result.graph.nodes.find((n) => n.op === "sum");
      expect(sumNode?.dtype).toBe("f32");
    });
  });
});

describe("isAMPTransformed", () => {
  it("returns false for untransformed graph", () => {
    const graph = createTestGraph([
      { id: 1, op: "input", inputs: [] },
      { id: 2, op: "matmul", inputs: [1] },
    ]);

    expect(isAMPTransformed(graph)).toBe(false);
  });

  it("returns true for graph with cast nodes", () => {
    const graph = createTestGraph([
      { id: 1, op: "input", inputs: [] },
      { id: 2, op: "cast", inputs: [1] },
      { id: 3, op: "matmul", inputs: [2] },
    ]);

    expect(isAMPTransformed(graph)).toBe(true);
  });
});

describe("getAMPStats", () => {
  it("counts nodes by type", () => {
    const graph = createTestGraph([
      { id: 1, op: "input", inputs: [], dtype: "f32" },
      { id: 2, op: "cast", inputs: [1], dtype: "f16" },
      { id: 3, op: "matmul", inputs: [2], dtype: "f16" },
      { id: 4, op: "cast", inputs: [3], dtype: "f32" },
    ]);

    const stats = getAMPStats(graph);

    expect(stats.castCount).toBe(2);
    expect(stats.f16NodeCount).toBe(2);
    expect(stats.f32NodeCount).toBe(2);
  });

  it("handles empty graph", () => {
    const graph: IRGraph = { epoch: 1, nodes: [], fusionGroups: [] };

    const stats = getAMPStats(graph);

    expect(stats.castCount).toBe(0);
    expect(stats.f16NodeCount).toBe(0);
    expect(stats.f32NodeCount).toBe(0);
  });
});

describe("Complex AMP Transform Scenarios", () => {
  let ctx: AutocastContext;

  beforeEach(() => {
    ctx = createAutocastContext();
    pushAutocast(ctx, { enabled: true });
  });

  it("handles chain: input -> matmul -> sum", () => {
    const graph = createTestGraph([
      { id: 1, op: "input", inputs: [], dtype: "f32" },
      { id: 2, op: "input", inputs: [], dtype: "f32" },
      { id: 3, op: "matmul", inputs: [1, 2], dtype: "f32" },
      { id: 4, op: "sum", inputs: [3], dtype: "f32" },
    ]);

    const result = applyAMPTransform(graph, ctx);

    // 2 casts for matmul inputs
    expect(result.castsInserted).toBe(2);

    // matmul should output f16
    const matmulNode = result.graph.nodes.find((n) => n.op === "matmul");
    expect(matmulNode?.dtype).toBe("f16");

    // sum should output f32 (it's f32-required)
    const sumNode = result.graph.nodes.find((n) => n.op === "sum");
    expect(sumNode?.dtype).toBe("f32");
  });

  it("handles chain: input -> matmul -> relu -> matmul", () => {
    const graph = createTestGraph([
      { id: 1, op: "input", inputs: [], dtype: "f32" },
      { id: 2, op: "weight1", inputs: [], dtype: "f32" },
      { id: 3, op: "matmul", inputs: [1, 2], dtype: "f32" },
      { id: 4, op: "relu", inputs: [3], dtype: "f32" },
      { id: 5, op: "weight2", inputs: [], dtype: "f32" },
      { id: 6, op: "matmul", inputs: [4, 5], dtype: "f32" },
    ]);

    const result = applyAMPTransform(graph, ctx);

    // First matmul: 2 casts for inputs
    // Second matmul: input from relu already processed
    // Weight2 needs cast
    expect(result.castsInserted).toBeGreaterThanOrEqual(2);

    // Both matmuls should output f16
    const matmulNodes = result.graph.nodes.filter((n) => n.op === "matmul");
    expect(matmulNodes).toHaveLength(2);
    matmulNodes.forEach((n) => {
      expect(n.dtype).toBe("f16");
    });
  });
});
