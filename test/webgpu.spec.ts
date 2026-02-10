import { describe, expect, it } from "vitest";

import { Engine, Torchlette } from "../src";
import {
  getWebGPUInitError,
  initWebGPU,
  runFusedElementwise,
  webgpuBackend,
} from "../src/backend/webgpu";
import { buildFusionRecipes } from "../src/engine/fusion";
import { cpuOnly } from "./helpers/webgpu";

describe.skipIf(cpuOnly)("webgpu backend", () => {
  it("runs simple elementwise ops with cpu readback", async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }

    const api = new Torchlette("webgpu");
    const a = api.tensorFromArray([1, 2, 3, 4], [4]);
    const b = api.tensorFromArray([5, 6, 7, 8], [4]);

    const out = a.add(b).mul(b).relu();
    const values = await out.cpu();

    expect(values).toEqual([30, 48, 70, 96]);
  });

  it("supports broadcast elementwise ops", async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }

    const api = new Torchlette("webgpu");
    const a = api.tensorFromArray([1, 2], [2, 1]);
    const b = api.tensorFromArray([10, 20, 30], [1, 3]);

    const out = a.add(b);
    const values = await out.cpu();

    expect(values).toEqual([11, 21, 31, 12, 22, 32]);
  });

  it("runs fused elementwise recipes against eager outputs", async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }

    const engine = new Engine();
    let inputAId = 0;
    let inputBId = 0;
    const compiled = engine.compile(() => {
      const inputA = engine._debug_emitLazyOp("input", {
        shape: [2, 1],
        dtype: "f32",
      });
      const inputB = engine._debug_emitLazyOp("input", {
        shape: [1, 3],
        dtype: "f32",
      });
      const add = engine._debug_emitLazyOp("add", {
        inputs: [inputA, inputB],
      });
      const mul = engine._debug_emitLazyOp("mul", {
        inputs: [add, inputB],
      });
      const relu = engine._debug_emitLazyOp("relu", { inputs: [mul] });
      inputAId = inputA.id;
      inputBId = inputB.id;
      return relu;
    });

    compiled();

    const graph = engine._debug_getLastCompiledGraph();
    if (!graph) {
      throw new Error("Expected a compiled graph");
    }

    const recipes = buildFusionRecipes(graph);
    expect(recipes).toHaveLength(1);

    const inputA = webgpuBackend.ops.tensorFromArray([1, 2], [2, 1]);
    const inputB = webgpuBackend.ops.tensorFromArray([5, 6, 7], [1, 3]);

    const eager = webgpuBackend.ops.relu(
      webgpuBackend.ops.mul(webgpuBackend.ops.add(inputA, inputB), inputB),
    );

    const inputMap = new Map([
      [inputAId, inputA],
      [inputBId, inputB],
    ]);
    const orderedInputs = recipes[0].inputs.map((id) => {
      const tensor = inputMap.get(id);
      if (!tensor) {
        throw new Error(`Missing tensor for input ${id}`);
      }
      return tensor;
    });

    const fused = runFusedElementwise(graph, recipes[0], orderedInputs);
    const [eagerValues, fusedValues] = await Promise.all([
      webgpuBackend.ops.read(eager),
      webgpuBackend.ops.read(fused),
    ]);

    expect(fusedValues).toEqual(eagerValues);
  });

  it("runs matmul with cpu readback", async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }

    const api = new Torchlette("webgpu");
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = api.tensorFromArray([7, 8, 9, 10, 11, 12], [3, 2]);
    const out = a.matmul(b);
    const values = await out.cpu();

    expect(values).toEqual([58, 64, 139, 154]);
  });
});
