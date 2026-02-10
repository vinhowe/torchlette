import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  getWebGPUInitError,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("WebGPU arange", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
  });

  it("basic arange(10)", async () => {
    const t = webgpuBackend.ops.arange!(10);
    expect(t.shape).toEqual([10]);
    const data = await webgpuBackend.ops.read(t);
    expect(data).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("arange with start and step", async () => {
    const t = webgpuBackend.ops.arange!(10, 2, 3);
    expect(t.shape).toEqual([3]); // [2, 5, 8]
    const data = await webgpuBackend.ops.read(t);
    expect(data[0]).toBeCloseTo(2);
    expect(data[1]).toBeCloseTo(5);
    expect(data[2]).toBeCloseTo(8);
  });

  it("arange with float step", async () => {
    const t = webgpuBackend.ops.arange!(2, 0, 0.5);
    expect(t.shape).toEqual([4]); // [0, 0.5, 1.0, 1.5]
    const data = await webgpuBackend.ops.read(t);
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(0.5);
    expect(data[2]).toBeCloseTo(1.0);
    expect(data[3]).toBeCloseTo(1.5);
  });

  it("large arange for position indices", async () => {
    const n = 1024;
    const t = webgpuBackend.ops.arange!(n);
    expect(t.shape).toEqual([n]);
    const data = await webgpuBackend.ops.read(t);
    for (let i = 0; i < n; i++) {
      expect(data[i]).toBe(i);
    }
  });

  it("arange result can be used in binary ops", async () => {
    const a = webgpuBackend.ops.arange!(5);
    const b = webgpuBackend.ops.arange!(5);
    const sum = webgpuBackend.ops.add(a, b);
    const data = await webgpuBackend.ops.read(sum);
    expect(data).toEqual([0, 2, 4, 6, 8]);
  });
});
