import { describe, expect, it } from "vitest";

import { Engine } from "../src";
import { buildFusionRecipes } from "../src/engine/fusion";

describe("fusion recipes", () => {
  it("captures inputs and outputs for elementwise fusion", () => {
    const engine = new Engine();
    let inputAId = 0;
    let inputBId = 0;
    let addId = 0;
    let reluId = 0;

    const compiled = engine.compile(() => {
      const inputA = engine._debug_emitLazyOp("input", {
        shape: [2, 2],
        dtype: "f32",
      });
      const inputB = engine._debug_emitLazyOp("input", {
        shape: [2, 2],
        dtype: "f32",
      });
      const add = engine._debug_emitLazyOp("add", {
        inputs: [inputA, inputB],
      });
      const relu = engine._debug_emitLazyOp("relu", {
        inputs: [add],
      });

      inputAId = inputA.id;
      inputBId = inputB.id;
      addId = add.id;
      reluId = relu.id;
      return relu;
    });

    compiled();

    const graph = engine._debug_getLastCompiledGraph();
    if (!graph) {
      throw new Error("Expected a compiled graph");
    }

    const recipes = buildFusionRecipes(graph);
    expect(recipes).toHaveLength(1);
    expect(recipes[0].inputs).toEqual([inputAId, inputBId]);
    expect(recipes[0].outputs).toEqual([reluId]);
    expect(recipes[0].nodeIds).toEqual([addId, reluId]);
    // outputDescriptors is the new multi-output structure
    expect(recipes[0].outputDescriptors).toHaveLength(1);
    expect(recipes[0].outputDescriptors[0].shape).toEqual([2, 2]);
    expect(recipes[0].outputDescriptors[0].dtype).toBe("f32");
  });
});
