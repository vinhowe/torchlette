import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  getWebGPUInitError,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("WebGPU tril/triu", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
  });

  it("tril zeros upper triangle", async () => {
    const a = webgpuBackend.ops.full!([3, 3], 1);
    const t = webgpuBackend.ops.tril!(a);
    const data = await webgpuBackend.ops.read(t);
    expect(data).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1]);
  });

  it("triu zeros lower triangle", async () => {
    const a = webgpuBackend.ops.full!([3, 3], 1);
    const t = webgpuBackend.ops.triu!(a);
    const data = await webgpuBackend.ops.read(t);
    expect(data).toEqual([1, 1, 1, 0, 1, 1, 0, 0, 1]);
  });

  it("triu with k=1 for causal mask", async () => {
    const a = webgpuBackend.ops.full!([4, 4], -1e9);
    const t = webgpuBackend.ops.triu!(a, 1);
    const data = await webgpuBackend.ops.read(t);
    // Row 0: [0, -1e9, -1e9, -1e9]
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(-1e9);
    // Row 3: [0, 0, 0, 0]
    expect(data[12]).toBeCloseTo(0);
    expect(data[15]).toBeCloseTo(0);
  });

  it("tril with k=-1", async () => {
    const a = webgpuBackend.ops.full!([3, 3], 1);
    const t = webgpuBackend.ops.tril!(a, -1);
    const data = await webgpuBackend.ops.read(t);
    expect(data).toEqual([0, 0, 0, 1, 0, 0, 1, 1, 0]);
  });

  it("triu on batched [2, 3, 3] tensor", async () => {
    const a = webgpuBackend.ops.full!([2, 3, 3], 1);
    const t = webgpuBackend.ops.triu!(a);
    const data = await webgpuBackend.ops.read(t);
    const expected = [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1];
    expect(data).toEqual(expected);
  });

  it("tril on non-square [2, 4] matrix", async () => {
    const a = webgpuBackend.ops.full!([2, 4], 1);
    const t = webgpuBackend.ops.tril!(a);
    const data = await webgpuBackend.ops.read(t);
    expect(data).toEqual([1, 0, 0, 0, 1, 1, 0, 0]);
  });

  it("causal mask pattern [1, 1, n, n]", async () => {
    const n = 4;
    const a = webgpuBackend.ops.full!([1, 1, n, n], -1e9);
    const mask = webgpuBackend.ops.triu!(a, 1);
    expect(mask.shape).toEqual([1, 1, n, n]);
    const data = await webgpuBackend.ops.read(mask);
    // Verify diagonal pattern
    expect(data[0]).toBeCloseTo(0); // [0,0] = 0 (on/below diagonal)
    expect(data[1]).toBeCloseTo(-1e9); // [0,1] = -1e9 (above diagonal)
    expect(data[5]).toBeCloseTo(0); // [1,1] = 0
    expect(data[15]).toBeCloseTo(0); // [3,3] = 0
  });
});
