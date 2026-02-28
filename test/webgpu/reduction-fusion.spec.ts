import { describe, expect, it } from "vitest";
import { Torchlette } from "../../src";
import { initWebGPU } from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

/**
 * Tests for reduction fusion correctness.
 *
 * When the engine detects elementwise ops feeding into a reduction,
 * it can fuse them into a single kernel (preamble fusion). These tests
 * verify that fused results match the decomposed CPU reference path.
 */
describe.skipIf(cpuOnly)("reduction fusion", { timeout: 30000 }, () => {
  it("relu + sum (preamble fusion) matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const cpu = new Torchlette("cpu");

    const data = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    const shape = [2, 3];

    const xGpu = gpu.tensorFromArray(data, shape);
    const gpuResult = xGpu.relu().sum({ dim: [1] });
    const gpuArr = await gpuResult.cpu();

    const xCpu = cpu.tensorFromArray(data, shape);
    const cpuResult = xCpu.relu().sum({ dim: [1] });
    const cpuArr = await cpuResult.cpu();

    expect(gpuArr.length).toBe(cpuArr.length);
    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("mul + sum (binary preamble) matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const cpu = new Torchlette("cpu");

    const aData = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const bData = [0.5, 1.5, 2.5, 0.1, 0.2, 0.3];
    const shape = [2, 3];

    const aGpu = gpu.tensorFromArray(aData, shape);
    const bGpu = gpu.tensorFromArray(bData, shape);
    const gpuResult = aGpu.mul(bGpu).sum({ dim: [1] });
    const gpuArr = await gpuResult.cpu();

    const aCpu = cpu.tensorFromArray(aData, shape);
    const bCpu = cpu.tensorFromArray(bData, shape);
    const cpuResult = aCpu.mul(bCpu).sum({ dim: [1] });
    const cpuArr = await cpuResult.cpu();

    expect(gpuArr.length).toBe(cpuArr.length);
    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("sqrt + sum (full reduction) matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const cpu = new Torchlette("cpu");

    const data = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
    const shape = [2, 3];

    const xGpu = gpu.tensorFromArray(data, shape);
    const gpuResult = xGpu.sqrt().sum();
    const gpuArr = await gpuResult.cpu();

    const xCpu = cpu.tensorFromArray(data, shape);
    const cpuResult = xCpu.sqrt().sum();
    const cpuArr = await cpuResult.cpu();

    expect(gpuArr.length).toBe(cpuArr.length);
    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("neg + sum with keepdim matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const cpu = new Torchlette("cpu");

    const data = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    const shape = [2, 3];

    const xGpu = gpu.tensorFromArray(data, shape);
    const gpuResult = xGpu.neg().sum({ dim: [1], keepdim: true });
    const gpuArr = await gpuResult.cpu();

    const xCpu = cpu.tensorFromArray(data, shape);
    const cpuResult = xCpu.neg().sum({ dim: [1], keepdim: true });
    const cpuArr = await cpuResult.cpu();

    expect(gpuArr.length).toBe(cpuArr.length);
    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("mul + add + sum (multi-op chain before reduction) matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const cpu = new Torchlette("cpu");

    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const shape = [2, 3];

    const xGpu = gpu.tensorFromArray(data, shape);
    const gpuResult = xGpu.mul(xGpu).add(xGpu).sum({ dim: [1] });
    const gpuArr = await gpuResult.cpu();

    const xCpu = cpu.tensorFromArray(data, shape);
    const cpuResult = xCpu.mul(xCpu).add(xCpu).sum({ dim: [1] });
    const cpuArr = await cpuResult.cpu();

    expect(gpuArr.length).toBe(cpuArr.length);
    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("exp + mean (reduction epilogue) matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const cpu = new Torchlette("cpu");

    const data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    const shape = [3, 3];

    const xGpu = gpu.tensorFromArray(data, shape);
    const gpuResult = xGpu.exp().mean({ dim: [1] });
    const gpuArr = await gpuResult.cpu();

    const xCpu = cpu.tensorFromArray(data, shape);
    const cpuResult = xCpu.exp().mean({ dim: [1] });
    const cpuArr = await cpuResult.cpu();

    expect(gpuArr.length).toBe(cpuArr.length);
    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("results are stable across multiple steps", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");

    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    const shape = [3, 3];

    const results: number[][] = [];
    for (let step = 0; step < 3; step++) {
      const x = gpu.tensorFromArray(data, shape);
      const result = x.exp().sum({ dim: [1] });
      results.push(await result.cpu());
      await gpu.markStep();
    }

    // All steps should produce identical results
    for (let step = 1; step < results.length; step++) {
      for (let i = 0; i < results[0].length; i++) {
        expect(results[step][i]).toBe(results[0][i]);
      }
    }
  });
});
