import { describe, expect, it } from "vitest";
import { Torchlette } from "../../src";
import { initWebGPU } from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";
import { crossEntropy } from "../../src/nn/functional";

describe.skipIf(cpuOnly)("fused cross-entropy", { timeout: 30000 }, () => {
  it("forward matches CPU decomposed path", async () => {
    await initWebGPU();

    // GPU (fused path)
    const gpu = new Torchlette("webgpu");
    const logitsGpu = gpu.tensorFromArray(
      [1.0, 2.0, 3.0, 1.0, 0.5, 0.5],
      [3, 2],
    );
    const targetsGpu = gpu.tensorFromArray([0, 1, 0], [3]);
    const fusedLoss = gpu._crossEntropyFused(logitsGpu, targetsGpu);
    const fusedArr = await fusedLoss.cpu();

    // CPU (decomposed path)
    const cpu = new Torchlette("cpu");
    const logitsCpu = cpu.tensorFromArray(
      [1.0, 2.0, 3.0, 1.0, 0.5, 0.5],
      [3, 2],
    );
    const targetsCpu = cpu.tensorFromArray([0, 1, 0], [3]);
    const cpuLoss = crossEntropy(cpu, logitsCpu, targetsCpu, {
      reduction: "none",
    });
    const cpuArr = await cpuLoss.cpu();

    console.log("Fused:", fusedArr);
    console.log("CPU:  ", cpuArr);

    // Compare
    for (let i = 0; i < 3; i++) {
      expect(Math.abs(fusedArr[i] - cpuArr[i])).toBeLessThan(1e-4);
    }

    gpu.markStep();
    cpu.markStep();
  });

  it("mean reduction matches CPU", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const logitsGpu = gpu.tensorFromArray(
      [2.0, 1.0, 0.5, 3.0, 0.1, 0.2, 0.3, 0.4],
      [2, 4],
    );
    const targetsGpu = gpu.tensorFromArray([0, 2], [2]);
    const gpuLoss = crossEntropy(gpu, logitsGpu, targetsGpu, {
      reduction: "mean",
    });
    const gpuVal = await gpuLoss.item();

    const cpu = new Torchlette("cpu");
    const logitsCpu = cpu.tensorFromArray(
      [2.0, 1.0, 0.5, 3.0, 0.1, 0.2, 0.3, 0.4],
      [2, 4],
    );
    const targetsCpu = cpu.tensorFromArray([0, 2], [2]);
    const cpuLoss = crossEntropy(cpu, logitsCpu, targetsCpu, {
      reduction: "mean",
    });
    const cpuVal = await cpuLoss.item();

    console.log("GPU mean loss:", gpuVal);
    console.log("CPU mean loss:", cpuVal);
    expect(Math.abs(gpuVal - cpuVal)).toBeLessThan(1e-4);

    gpu.markStep();
    cpu.markStep();
  });

  it("backward matches CPU decomposed gradients", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const logitsGpu = gpu.tensorFromArray(
      [1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
      [2, 3],
      { requiresGrad: true },
    );
    const targetsGpu = gpu.tensorFromArray([2, 0], [2]);
    const gpuLoss = crossEntropy(gpu, logitsGpu, targetsGpu, {
      reduction: "mean",
    });
    await gpuLoss.backward();
    const gpuGrad = await logitsGpu.grad!.cpu();

    const cpu = new Torchlette("cpu");
    const logitsCpu = cpu.tensorFromArray(
      [1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
      [2, 3],
      { requiresGrad: true },
    );
    const targetsCpu = cpu.tensorFromArray([2, 0], [2]);
    const cpuLoss = crossEntropy(cpu, logitsCpu, targetsCpu, {
      reduction: "mean",
    });
    await cpuLoss.backward();
    const cpuGrad = await logitsCpu.grad!.cpu();

    console.log("GPU grad:", gpuGrad);
    console.log("CPU grad:", cpuGrad);

    for (let i = 0; i < gpuGrad.length; i++) {
      expect(Math.abs(gpuGrad[i] - cpuGrad[i])).toBeLessThan(1e-4);
    }

    gpu.markStep();
    cpu.markStep();
  });
});
