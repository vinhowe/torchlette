/**
 * Tests for matmul epilogue fusion.
 *
 * Tests that fused operations (bias, relu, gelu, etc.) produce correct results.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  dispatchMatmulWithEpilogue,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";
import type { EpilogueConfig } from "../../src/backend/webgpu/matmul";

import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

function makeValues(size: number, offset = 0): number[] {
  const values = new Array<number>(size);
  for (let i = 0; i < size; i++) {
    values[i] = ((i + offset) % 7) - 3;
  }
  return values;
}

function matmulReference(
  a: number[],
  b: number[],
  m: number,
  n: number,
  k: number,
): number[] {
  const out = new Array<number>(m * n).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let t = 0; t < k; t++) {
        sum += a[i * k + t] * b[t * n + j];
      }
      out[i * n + j] = sum;
    }
  }
  return out;
}

function biasReference(
  matmulOut: number[],
  bias: number[],
  n: number,
): number[] {
  const out = matmulOut.slice();
  for (let i = 0; i < out.length; i++) {
    out[i] += bias[i % n];
  }
  return out;
}

function reluReference(arr: number[]): number[] {
  return arr.map((x) => Math.max(0, x));
}

function geluReference(arr: number[]): number[] {
  return arr.map((x) => {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const inner = 0.7978845608 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1 + Math.tanh(inner));
  });
}

function siluReference(arr: number[]): number[] {
  return arr.map((x) => x / (1 + Math.exp(-x)));
}

function arraysClose(
  actual: number[],
  expected: number[],
  rtol = 1e-3,
  atol = 1e-3,
): boolean {
  if (actual.length !== expected.length) {
    console.log(
      `Length mismatch: actual=${actual.length}, expected=${expected.length}`,
    );
    return false;
  }
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const bound = atol + rtol * Math.abs(expected[i]);
    if (diff > bound) {
      console.log(
        `Mismatch at ${i}: actual=${actual[i]}, expected=${expected[i]}, diff=${diff}, bound=${bound}`,
      );
      return false;
    }
  }
  return true;
}

describe.runIf(isWebGPUEnabled)("matmul epilogue fusion (webgpu)", () => {
  beforeAll(async () => {
    const ok = await initWebGPU();
    if (!ok) {
      console.warn("WebGPU init failed:", getWebGPUInitError());
    }
  });

  afterAll(async () => {
    await syncWebGPU();
  });

  describe("matmul + bias", () => {
    it("fuses bias addition correctly (64x64)", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const biasVals = makeValues(n, 2);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;
      const bias = webgpuBackend.ops.tensorFromArray(biasVals, [n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, [bias]);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      // Reference: matmul then add bias
      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const expected = biasReference(matmulOut, biasVals, n);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("fuses bias addition correctly (128x64)", async () => {
      const m = 128,
        n = 64,
        k = 32;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const biasVals = makeValues(n, 2);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;
      const bias = webgpuBackend.ops.tensorFromArray(biasVals, [n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, [bias]);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const expected = biasReference(matmulOut, biasVals, n);

      expect(arraysClose(result, expected)).toBe(true);
    });
  });

  describe("matmul + activation", () => {
    it("fuses relu correctly", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "relu" }],
        additionalInputCount: 0,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, []);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const expected = reluReference(matmulOut);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("fuses gelu correctly", async () => {
      const m = 32,
        n = 32,
        k = 32;
      // Use smaller values to avoid numerical issues with gelu
      const aVals = makeValues(m * k, 0).map((x) => x * 0.1);
      const bVals = makeValues(k * n, 1).map((x) => x * 0.1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "gelu" }],
        additionalInputCount: 0,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, []);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const expected = geluReference(matmulOut);

      // GELU needs slightly higher tolerance due to approximation
      expect(arraysClose(result, expected, 1e-2, 1e-2)).toBe(true);
    });

    it("fuses silu correctly", async () => {
      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0).map((x) => x * 0.1);
      const bVals = makeValues(k * n, 1).map((x) => x * 0.1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "silu" }],
        additionalInputCount: 0,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, []);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const expected = siluReference(matmulOut);

      expect(arraysClose(result, expected, 1e-3, 1e-3)).toBe(true);
    });
  });

  describe("matmul + bias + activation (chain)", () => {
    it("fuses bias + relu correctly", async () => {
      const m = 64,
        n = 64,
        k = 64;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const biasVals = makeValues(n, 2);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;
      const bias = webgpuBackend.ops.tensorFromArray(biasVals, [n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }, { kind: "relu" }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, [bias]);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      // Reference: matmul -> bias -> relu
      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const biasOut = biasReference(matmulOut, biasVals, n);
      const expected = reluReference(biasOut);

      expect(arraysClose(result, expected)).toBe(true);
    });

    it("fuses bias + gelu correctly", async () => {
      const m = 32,
        n = 32,
        k = 32;
      const aVals = makeValues(m * k, 0).map((x) => x * 0.1);
      const bVals = makeValues(k * n, 1).map((x) => x * 0.1);
      const biasVals = makeValues(n, 2).map((x) => x * 0.1);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;
      const bias = webgpuBackend.ops.tensorFromArray(biasVals, [n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }, { kind: "gelu" }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, [bias]);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const biasOut = biasReference(matmulOut, biasVals, n);
      const expected = geluReference(biasOut);

      expect(arraysClose(result, expected, 1e-2, 1e-2)).toBe(true);
    });
  });

  describe("non-tile-aligned epilogue", () => {
    it("handles non-tile-aligned shapes with bias correctly", async () => {
      const m = 50,
        n = 70,
        k = 60;
      const aVals = makeValues(m * k, 0);
      const bVals = makeValues(k * n, 1);
      const biasVals = makeValues(n, 2);

      const a = webgpuBackend.ops.tensorFromArray(aVals, [m, k]) as any;
      const b = webgpuBackend.ops.tensorFromArray(bVals, [k, n]) as any;
      const bias = webgpuBackend.ops.tensorFromArray(biasVals, [n]) as any;

      const epilogue: EpilogueConfig = {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      };

      const c = dispatchMatmulWithEpilogue(a, b, epilogue, [bias]);

      await syncWebGPU();
      const result = await webgpuBackend.ops.read(c);

      const matmulOut = matmulReference(aVals, bVals, m, n, k);
      const expected = biasReference(matmulOut, biasVals, n);

      expect(arraysClose(result, expected)).toBe(true);
    });
  });
});
