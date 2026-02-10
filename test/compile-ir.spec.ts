import { describe, expect, it } from "vitest";
import { Engine } from "../src";

describe("compile ir scaffolding", () => {
  it("records lazy ops into a compiled graph", () => {
    const engine = new Engine();
    const compiled = engine.compile(() => {
      engine._debug_emitLazyOp("add");
      engine._debug_emitLazyOp("mul");
      engine._debug_emitLazyOp("relu");
    });

    compiled();

    const graph = engine._debug_getLastCompiledGraph();
    if (!graph) {
      throw new Error("Expected a compiled graph");
    }

    expect(graph.nodes.map((node) => node.op)).toEqual(["add", "mul", "relu"]);
    expect(graph.fusionGroups).toEqual([
      {
        id: 0,
        kind: "elementwise",
        nodeIds: graph.nodes.map((node) => node.id),
      },
    ]);
  });

  it("captures inputs and metadata for compiled nodes", () => {
    const engine = new Engine();
    const compiled = engine.compile(() => {
      const a = engine._debug_emitLazyOp("add", {
        shape: [2, 2],
        dtype: "f32",
      });
      const b = engine._debug_emitLazyOp("mul", {
        inputs: [a],
        shape: [2, 2],
        dtype: "f32",
      });
      return engine._debug_emitLazyOp("sum", {
        inputs: [b],
        shape: [],
        dtype: "f32",
      });
    });

    compiled();

    const graph = engine._debug_getLastCompiledGraph();
    if (!graph) {
      throw new Error("Expected a compiled graph");
    }

    const [addNode, mulNode, sumNode] = graph.nodes;
    expect(addNode.inputs).toEqual([]);
    expect(addNode.shape).toEqual([2, 2]);
    expect(addNode.dtype).toBe("f32");

    expect(mulNode.inputs).toEqual([addNode.id]);
    expect(mulNode.shape).toEqual([2, 2]);
    expect(mulNode.dtype).toBe("f32");

    expect(sumNode.inputs).toEqual([mulNode.id]);
    expect(sumNode.shape).toEqual([]);
    expect(sumNode.dtype).toBe("f32");
  });

  it("infers elementwise shape and dtype from inputs", () => {
    const engine = new Engine();
    const compiled = engine.compile(() => {
      const a = engine._debug_emitLazyOp("add", {
        shape: [2, 1],
        dtype: "f32",
      });
      const b = engine._debug_emitLazyOp("mul", {
        shape: [1, 3],
        dtype: "f32",
      });
      return engine._debug_emitLazyOp("add", {
        inputs: [a, b],
      });
    });

    compiled();

    const graph = engine._debug_getLastCompiledGraph();
    if (!graph) {
      throw new Error("Expected a compiled graph");
    }

    const addNode = graph.nodes[2];
    expect(addNode.shape).toEqual([2, 3]);
    expect(addNode.dtype).toBe("f32");
  });
});
