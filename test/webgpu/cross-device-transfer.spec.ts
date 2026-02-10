import { beforeAll, beforeEach, describe, expect, it } from "vitest";

import { registerBackend } from "../../src/backend/registry";
import { RuntimeEngine } from "../../src/runtime/engine";
import { resetBaseIdCounter } from "../../src/runtime/tensor";
import { resetNodeIdCounter } from "../../src/engine/lazy";

import { cpuOnly } from "../helpers/webgpu";
const hasWebGPU = !cpuOnly;

describe.skipIf(!hasWebGPU)("Cross-Device Transfer (CPU <-> WebGPU)", () => {
  let engine: RuntimeEngine;

  beforeAll(async () => {
    // Initialize and register WebGPU backend
    const { initWebGPU, webgpuBackend } = await import("../../src/backend/webgpu");
    await initWebGPU();
    registerBackend(webgpuBackend);
  });

  beforeEach(() => {
    engine = new RuntimeEngine("cpu");
    resetNodeIdCounter();
    resetBaseIdCounter();
  });

  it("transfers CPU tensor to WebGPU and back", async () => {
    // Create on CPU
    const cpuTensor = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    expect(cpuTensor.device).toBe("cpu");

    // Transfer to WebGPU
    const gpuTensor = engine.transfer(cpuTensor, "webgpu");
    expect(gpuTensor.device).toBe("webgpu");

    // Transfer back to CPU
    const backToCpu = engine.transfer(gpuTensor, "cpu");
    expect(backToCpu.device).toBe("cpu");

    // Force and verify data integrity
    await engine.force(backToCpu);
    const values = await engine.cpu(backToCpu);
    expect(values).toEqual([1, 2, 3, 4]);
  });

  it("computes on GPU after transfer from CPU", async () => {
    // Create on CPU
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.tensorFromArray([10, 20, 30, 40], [2, 2], "cpu");

    // Transfer to WebGPU
    const aGpu = engine.transfer(a, "webgpu");
    const bGpu = engine.transfer(b, "webgpu");

    // Compute on GPU
    const result = engine.add(aGpu, bGpu);
    expect(result.device).toBe("webgpu");

    // Read back
    await engine.force(result);
    const values = await engine.cpu(result);
    expect(values).toEqual([11, 22, 33, 44]);
  });

  it("transfers large tensor correctly", async () => {
    // Create a larger tensor
    const size = 1024;
    const data = Array.from({ length: size }, (_, i) => i);
    const cpuTensor = engine.tensorFromArray(data, [32, 32], "cpu");

    // Transfer to GPU and back
    const gpuTensor = engine.transfer(cpuTensor, "webgpu");
    const backToCpu = engine.transfer(gpuTensor, "cpu");

    await engine.force(backToCpu);
    const values = await engine.cpu(backToCpu);
    expect(values).toEqual(data);
  });

  it("matmul on GPU with CPU-created inputs", async () => {
    // Create matrices on CPU
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");
    const b = engine.tensorFromArray([5, 6, 7, 8], [2, 2], "cpu");

    // Transfer to GPU
    const aGpu = engine.transfer(a, "webgpu");
    const bGpu = engine.transfer(b, "webgpu");

    // Matmul on GPU
    const result = engine.matmul(aGpu, bGpu);
    expect(result.device).toBe("webgpu");

    // Read back and verify
    // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    await engine.force(result);
    const values = await engine.cpu(result);
    expect(values).toEqual([19, 22, 43, 50]);
  });

  it("chain of GPU operations with CPU input/output", async () => {
    // Create on CPU
    const input = engine.tensorFromArray([-1, 2, -3, 4], [2, 2], "cpu");

    // Transfer to GPU, compute, transfer back
    const gpuInput = engine.transfer(input, "webgpu");
    const gpuResult = engine.relu(gpuInput);
    const cpuResult = engine.transfer(gpuResult, "cpu");

    await engine.force(cpuResult);
    const values = await engine.cpu(cpuResult);
    expect(values).toEqual([0, 2, 0, 4]);
  });

  it("multiple transfers in computation graph", async () => {
    // Complex graph: CPU -> GPU -> compute -> GPU -> CPU -> compute -> result
    const a = engine.tensorFromArray([1, 2, 3, 4], [2, 2], "cpu");

    // First GPU computation
    const aGpu = engine.transfer(a, "webgpu");
    const bGpu = engine.relu(aGpu);

    // Transfer back to CPU for another op
    const bCpu = engine.transfer(bGpu, "cpu");

    // CPU computation
    const c = engine.tensorFromArray([10, 10, 10, 10], [2, 2], "cpu");
    const result = engine.add(bCpu, c);

    await engine.force(result);
    const values = await engine.cpu(result);
    expect(values).toEqual([11, 12, 13, 14]);
  });
});

describe.skipIf(!hasWebGPU)("Frontend Cross-Device Transfer (WebGPU)", () => {
  beforeAll(async () => {
    const { initWebGPU, webgpuBackend } = await import("../../src/backend/webgpu");
    await initWebGPU();
    registerBackend(webgpuBackend);
  });

  it("tensor.to('webgpu') transfers and computes correctly", async () => {
    const { Torchlette } = await import("../../src/frontend");
    const torch = new Torchlette("cpu");

    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    expect(a.device).toBe("cpu");

    const aGpu = a.to("webgpu");
    expect(aGpu.device).toBe("webgpu");

    const result = aGpu.relu();
    expect(result.device).toBe("webgpu");

    const values = await result.cpu();
    expect(values).toEqual([1, 2, 3, 4]);
  });

  it("toNow() forces immediate transfer", async () => {
    const { Torchlette } = await import("../../src/frontend");
    const torch = new Torchlette("cpu");

    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const aGpu = await a.toNow("webgpu");

    expect(aGpu.device).toBe("webgpu");

    // Should be able to read immediately
    const values = await aGpu.cpu();
    expect(values).toEqual([1, 2, 3, 4]);
  });

  it("round-trip transfer preserves data", async () => {
    const { Torchlette } = await import("../../src/frontend");
    const torch = new Torchlette("cpu");

    const original = [1.5, -2.5, 3.5, -4.5];
    const a = torch.tensorFromArray(original, [2, 2]);

    // CPU -> GPU -> CPU
    const roundTrip = a.to("webgpu").to("cpu");

    const values = await roundTrip.cpu();
    expect(values).toEqual(original);
  });
});
