/**
 * Tests for conv2d op — tile-IR direct convolution kernel.
 * Validates WebGPU results against CPU reference implementation.
 */
import { beforeAll, describe, expect, it } from "vitest";
import {
  type Tensor as CPUTensor,
  cpuBackend,
  tensorFromArray,
} from "../../src/backend/cpu";
import {
  getWebGPUInitError,
  initWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

// Helper: create a WebGPU tensor from data
function gpuTensor(data: number[], shape: number[]) {
  return webgpuBackend.ops.tensorFromArray(data, shape);
}

// Helper: read WebGPU tensor data
async function readGPU(tensor: unknown): Promise<number[]> {
  return webgpuBackend.ops.read(tensor);
}

// Helper: allClose comparison
function expectClose(actual: number[], expected: number[], atol = 1e-4) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(Math.abs(actual[i] - expected[i])).toBeLessThan(atol);
  }
}

describe.skipIf(cpuOnly)("Conv2d", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) throw new Error(`WebGPU init failed: ${getWebGPUInitError()}`);
  });
  it("1x1 convolution without padding", async () => {
    // input: [1, 2, 3, 3], weight: [4, 2, 1, 1] — pointwise conv
    const input = [
      // channel 0
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      // channel 1
      9, 8, 7, 6, 5, 4, 3, 2, 1,
    ];
    const weight = [
      // out_ch 0: [in_ch 0: 1, in_ch 1: 0]
      1, 0,
      // out_ch 1: [in_ch 0: 0, in_ch 1: 1]
      0, 1,
      // out_ch 2: [in_ch 0: 1, in_ch 1: 1]
      1, 1,
      // out_ch 3: [in_ch 0: 1, in_ch 1: -1]
      1, -1,
    ];

    const gpuInput = gpuTensor(input, [1, 2, 3, 3]);
    const gpuWeight = gpuTensor(weight, [4, 2, 1, 1]);

    const cpuInput = tensorFromArray(input, [1, 2, 3, 3]);
    const cpuWeight = tensorFromArray(weight, [4, 2, 1, 1]);

    const gpuOut = webgpuBackend.ops.conv2d?.(gpuInput, gpuWeight, undefined);
    const cpuOut = cpuBackend.ops.conv2d?.(
      cpuInput,
      cpuWeight,
      undefined,
    ) as CPUTensor;

    const gpuData = await readGPU(gpuOut);
    const cpuData = cpuOut.toArray();

    expect((gpuOut as { shape: number[] }).shape).toEqual([1, 4, 3, 3]);
    expectClose(gpuData, cpuData);
  });

  it("3x3 convolution with padding=1", async () => {
    // input: [1, 1, 4, 4], weight: [1, 1, 3, 3]
    const input = Array.from({ length: 16 }, (_, i) => i + 1);
    const weight = [1, 0, -1, 2, 0, -2, 1, 0, -1]; // horizontal edge detector

    const gpuInput = gpuTensor(input, [1, 1, 4, 4]);
    const gpuWeight = gpuTensor(weight, [1, 1, 3, 3]);

    const cpuInput = tensorFromArray(input, [1, 1, 4, 4]);
    const cpuWeight = tensorFromArray(weight, [1, 1, 3, 3]);

    const gpuOut = webgpuBackend.ops.conv2d?.(gpuInput, gpuWeight, undefined, {
      padding: 1,
    });
    const cpuOut = cpuBackend.ops.conv2d?.(cpuInput, cpuWeight, undefined, {
      padding: 1,
    }) as CPUTensor;

    const gpuData = await readGPU(gpuOut);
    const cpuData = cpuOut.toArray();

    expect((gpuOut as { shape: number[] }).shape).toEqual([1, 1, 4, 4]);
    expectClose(gpuData, cpuData);
  });

  it("strided convolution", async () => {
    // input: [1, 1, 6, 6], weight: [1, 1, 3, 3], stride=2
    const input = Array.from({ length: 36 }, (_, i) => i + 1);
    const weight = Array.from({ length: 9 }, () => 1); // all ones

    const gpuInput = gpuTensor(input, [1, 1, 6, 6]);
    const gpuWeight = gpuTensor(weight, [1, 1, 3, 3]);

    const cpuInput = tensorFromArray(input, [1, 1, 6, 6]);
    const cpuWeight = tensorFromArray(weight, [1, 1, 3, 3]);

    const gpuOut = webgpuBackend.ops.conv2d?.(gpuInput, gpuWeight, undefined, {
      stride: 2,
    });
    const cpuOut = cpuBackend.ops.conv2d?.(cpuInput, cpuWeight, undefined, {
      stride: 2,
    }) as CPUTensor;

    const gpuData = await readGPU(gpuOut);
    const cpuData = cpuOut.toArray();

    expect((gpuOut as { shape: number[] }).shape).toEqual([1, 1, 2, 2]);
    expectClose(gpuData, cpuData);
  });

  it("with bias", async () => {
    // input: [1, 1, 3, 3], weight: [2, 1, 1, 1], bias: [2]
    const input = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const weight = [1, -1]; // 2 output channels, 1x1 kernel
    const bias = [10, 20];

    const gpuInput = gpuTensor(input, [1, 1, 3, 3]);
    const gpuWeight = gpuTensor(weight, [2, 1, 1, 1]);
    const gpuBias = gpuTensor(bias, [2]);

    const cpuInput = tensorFromArray(input, [1, 1, 3, 3]);
    const cpuWeight = tensorFromArray(weight, [2, 1, 1, 1]);
    const cpuBias = tensorFromArray(bias, [2]);

    const gpuOut = webgpuBackend.ops.conv2d?.(gpuInput, gpuWeight, gpuBias);
    const cpuOut = cpuBackend.ops.conv2d?.(
      cpuInput,
      cpuWeight,
      cpuBias,
    ) as CPUTensor;

    const gpuData = await readGPU(gpuOut);
    const cpuData = cpuOut.toArray();

    expect((gpuOut as { shape: number[] }).shape).toEqual([1, 2, 3, 3]);
    expectClose(gpuData, cpuData);
    // Channel 0: input + 10 = [11, 12, 13, 14, 15, 16, 17, 18, 19]
    expect(gpuData.slice(0, 9)).toEqual([11, 12, 13, 14, 15, 16, 17, 18, 19]);
  });

  it("batched convolution", async () => {
    // input: [2, 1, 3, 3], weight: [1, 1, 2, 2]
    const input = [
      // batch 0
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      // batch 1
      9, 8, 7, 6, 5, 4, 3, 2, 1,
    ];
    const weight = [1, 1, 1, 1]; // 2x2 sum

    const gpuInput = gpuTensor(input, [2, 1, 3, 3]);
    const gpuWeight = gpuTensor(weight, [1, 1, 2, 2]);

    const cpuInput = tensorFromArray(input, [2, 1, 3, 3]);
    const cpuWeight = tensorFromArray(weight, [1, 1, 2, 2]);

    const gpuOut = webgpuBackend.ops.conv2d?.(gpuInput, gpuWeight, undefined);
    const cpuOut = cpuBackend.ops.conv2d?.(
      cpuInput,
      cpuWeight,
      undefined,
    ) as CPUTensor;

    const gpuData = await readGPU(gpuOut);
    const cpuData = cpuOut.toArray();

    expect((gpuOut as { shape: number[] }).shape).toEqual([2, 1, 2, 2]);
    expectClose(gpuData, cpuData);
  });

  it("multi-channel input and output", async () => {
    // input: [1, 3, 4, 4], weight: [2, 3, 3, 3], bias: [2]
    const input = Array.from({ length: 3 * 16 }, (_, i) => Math.sin(i * 0.5));
    const weight = Array.from({ length: 2 * 3 * 9 }, (_, i) =>
      Math.cos(i * 0.3),
    );
    const bias = [0.5, -0.5];

    const gpuInput = gpuTensor(input, [1, 3, 4, 4]);
    const gpuWeight = gpuTensor(weight, [2, 3, 3, 3]);
    const gpuBias = gpuTensor(bias, [2]);

    const cpuInput = tensorFromArray(input, [1, 3, 4, 4]);
    const cpuWeight = tensorFromArray(weight, [2, 3, 3, 3]);
    const cpuBias = tensorFromArray(bias, [2]);

    const gpuOut = webgpuBackend.ops.conv2d?.(gpuInput, gpuWeight, gpuBias, {
      padding: 1,
    });
    const cpuOut = cpuBackend.ops.conv2d?.(cpuInput, cpuWeight, cpuBias, {
      padding: 1,
    }) as CPUTensor;

    const gpuData = await readGPU(gpuOut);
    const cpuData = cpuOut.toArray();

    expect((gpuOut as { shape: number[] }).shape).toEqual([1, 2, 4, 4]);
    expectClose(gpuData, cpuData, 1e-3);
  });
});
