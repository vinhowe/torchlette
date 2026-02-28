import { describe, expect, it } from "vitest";

import { Torchlette } from "../src";
import {
  getWebGPUInitError,
  initWebGPU,
} from "../src/backend/webgpu";
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
