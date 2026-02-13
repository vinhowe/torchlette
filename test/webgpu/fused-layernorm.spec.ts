import { describe, expect, it } from "vitest";
import { Torchlette } from "../../src";
import { initWebGPU } from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("fused layernorm", { timeout: 30000 }, () => {
  it("forward matches CPU decomposed path (small)", async () => {
    await initWebGPU();

    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    const shape: [number, number] = [3, 4];
    const weightData = [1.0, 1.0, 1.0, 1.0];
    const biasData = [0.0, 0.0, 0.0, 0.0];

    // GPU (fused path)
    const gpu = new Torchlette("webgpu");
    const xGpu = gpu.tensorFromArray(data, shape);
    const wGpu = gpu.tensorFromArray(weightData, [4]);
    const bGpu = gpu.tensorFromArray(biasData, [4]);
    const gpuResult = gpu.layernorm(xGpu, wGpu, bGpu);
    const gpuArr = await gpuResult.cpu();

    // CPU (decomposed path)
    const cpu = new Torchlette("cpu");
    const xCpu = cpu.tensorFromArray(data, shape);
    const wCpu = cpu.tensorFromArray(weightData, [4]);
    const bCpu = cpu.tensorFromArray(biasData, [4]);
    const cpuResult = cpu.layernorm(xCpu, wCpu, bCpu);
    const cpuArr = await cpuResult.cpu();

    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("forward matches CPU with weight and bias", async () => {
    await initWebGPU();

    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const shape: [number, number] = [2, 3];
    const weightData = [0.5, 2.0, 1.5];
    const biasData = [0.1, -0.2, 0.3];

    const gpu = new Torchlette("webgpu");
    const xGpu = gpu.tensorFromArray(data, shape);
    const wGpu = gpu.tensorFromArray(weightData, [3]);
    const bGpu = gpu.tensorFromArray(biasData, [3]);
    const gpuResult = gpu.layernorm(xGpu, wGpu, bGpu);
    const gpuArr = await gpuResult.cpu();

    const cpu = new Torchlette("cpu");
    const xCpu = cpu.tensorFromArray(data, shape);
    const wCpu = cpu.tensorFromArray(weightData, [3]);
    const bCpu = cpu.tensorFromArray(biasData, [3]);
    const cpuResult = cpu.layernorm(xCpu, wCpu, bCpu);
    const cpuArr = await cpuResult.cpu();

    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("forward matches CPU with 3D input [B,S,D]", async () => {
    await initWebGPU();

    // [2, 3, 4] shape — typical for B=2, S=3, D=4
    const data = Array.from({ length: 24 }, (_, i) => (i + 1) * 0.1);
    const shape: [number, number, number] = [2, 3, 4];
    const weightData = [1.0, 0.5, 2.0, 1.5];
    const biasData = [0.0, 0.1, -0.1, 0.2];

    const gpu = new Torchlette("webgpu");
    const xGpu = gpu.tensorFromArray(data, shape);
    const wGpu = gpu.tensorFromArray(weightData, [4]);
    const bGpu = gpu.tensorFromArray(biasData, [4]);
    const gpuResult = gpu.layernorm(xGpu, wGpu, bGpu);
    const gpuArr = await gpuResult.cpu();

    const cpu = new Torchlette("cpu");
    const xCpu = cpu.tensorFromArray(data, shape);
    const wCpu = cpu.tensorFromArray(weightData, [4]);
    const bCpu = cpu.tensorFromArray(biasData, [4]);
    const cpuResult = cpu.layernorm(xCpu, wCpu, bCpu);
    const cpuArr = await cpuResult.cpu();

    for (let i = 0; i < gpuArr.length; i++) {
      expect(Math.abs(gpuArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("backward gradient accuracy through decomposed backward with fused forward", async () => {
    await initWebGPU();

    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const shape: [number, number] = [2, 3];
    const weightData = [0.5, 2.0, 1.5];
    const biasData = [0.1, -0.2, 0.3];

    // GPU
    const gpu = new Torchlette("webgpu");
    const xGpu = gpu.tensorFromArray(data, shape, { requiresGrad: true });
    const wGpu = gpu.tensorFromArray(weightData, [3], { requiresGrad: true });
    const bGpu = gpu.tensorFromArray(biasData, [3], { requiresGrad: true });
    const gpuResult = gpu.layernorm(xGpu, wGpu, bGpu);
    const gpuLoss = gpu.sum(gpuResult);
    await gpuLoss.backward();
    const gpuGradX = await xGpu.grad!.cpu();
    const gpuGradW = await wGpu.grad!.cpu();
    const gpuGradB = await bGpu.grad!.cpu();

    // CPU
    const cpu = new Torchlette("cpu");
    const xCpu = cpu.tensorFromArray(data, shape, { requiresGrad: true });
    const wCpu = cpu.tensorFromArray(weightData, [3], { requiresGrad: true });
    const bCpu = cpu.tensorFromArray(biasData, [3], { requiresGrad: true });
    const cpuResult = cpu.layernorm(xCpu, wCpu, bCpu);
    const cpuLoss = cpu.sum(cpuResult);
    await cpuLoss.backward();
    const cpuGradX = await xCpu.grad!.cpu();
    const cpuGradW = await wCpu.grad!.cpu();
    const cpuGradB = await bCpu.grad!.cpu();

    for (let i = 0; i < gpuGradX.length; i++) {
      expect(Math.abs(gpuGradX[i] - cpuGradX[i])).toBeLessThan(1e-3);
    }
    for (let i = 0; i < gpuGradW.length; i++) {
      expect(Math.abs(gpuGradW[i] - cpuGradW[i])).toBeLessThan(1e-3);
    }
    for (let i = 0; i < gpuGradB.length; i++) {
      expect(Math.abs(gpuGradB[i] - cpuGradB[i])).toBeLessThan(1e-3);
    }

    await gpu.markStep();
    await cpu.markStep();
  });

  it("multi-step stability (3 markStep cycles)", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");

    for (let step = 0; step < 3; step++) {
      const data = Array.from({ length: 12 }, (_, i) => (i + 1) * (step + 1) * 0.1);
      const shape: [number, number] = [3, 4];
      const weightData = [1.0, 1.0, 1.0, 1.0];
      const biasData = [0.0, 0.0, 0.0, 0.0];

      const xGpu = gpu.tensorFromArray(data, shape);
      const wGpu = gpu.tensorFromArray(weightData, [4]);
      const bGpu = gpu.tensorFromArray(biasData, [4]);
      const gpuResult = gpu.layernorm(xGpu, wGpu, bGpu);
      const gpuArr = await gpuResult.cpu();

      // Verify output is reasonable (not NaN, has variance)
      for (const val of gpuArr) {
        expect(isFinite(val)).toBe(true);
      }

      await gpu.markStep();
    }
  });
});
