/**
 * Tests for Adam kernel vec4 coalescing path.
 *
 * Uses tensor sizes divisible by 4 to exercise the vec4 shader variant,
 * and verifies numerical correctness against a CPU reference implementation.
 */
import { describe, expect, it, beforeAll } from "vitest";
import { Torchlette, Adam } from "../../src";
import { initWebGPU, getWebGPUInitError } from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

const ATOL = 1e-5;
const RTOL = 1e-4;

function assertClose(actual: number[], expected: number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    const tol = ATOL + RTOL * Math.abs(expected[i]);
    expect(diff).toBeLessThanOrEqual(tol);
  }
}

describe.skipIf(cpuOnly)("WebGPU Adam vec4 coalescing", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
  });

  it("vec4 path: 8-element param produces correct Adam update", { timeout: 30000 }, async () => {
    // 8 elements = divisible by 4, so vec4 path is used
    const api = new Torchlette("webgpu");
    const lr = 0.01;
    const betas: [number, number] = [0.9, 0.999];
    const eps = 1e-8;

    const wValues = [0.5, -1.0, 2.0, 1.5, -0.5, 0.25, 0.75, -0.3];
    const w = api.tensorFromArray(wValues, [4, 2], { requiresGrad: true });
    const x = api.tensorFromArray(
      [1, 2, 3, 4, 5, 6, 7, 8],
      [2, 4],
    );
    const y = api.tensorFromArray([3, 1, 2, -1], [2, 2]);

    const pred = x.matmul(w);
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.reshape([]).backward();

    if (!w.grad) throw new Error("Missing gradient");
    const gradValues = await w.grad.cpu();

    const optimizer = new Adam([w], { lr, betas, eps }, api);
    const [wUpdated] = optimizer.step();
    const gpuResult = await wUpdated.cpu();

    // CPU reference: one step of Adam
    const biasCorrection1 = 1 - betas[0];
    const biasCorrection2 = 1 - betas[1];
    const stepSize = (lr * Math.sqrt(biasCorrection2)) / biasCorrection1;

    const expected = wValues.map((p, i) => {
      const g = gradValues[i];
      const m = (1 - betas[0]) * g;
      const v = (1 - betas[1]) * g * g;
      return p - (stepSize * m) / (Math.sqrt(v) + eps);
    });

    assertClose(gpuResult, expected);
  });

  it("vec4 path: 16-element param with weight decay (AdamW)", { timeout: 30000 }, async () => {
    // 16 elements = divisible by 4
    const api = new Torchlette("webgpu");
    const lr = 0.001;
    const betas: [number, number] = [0.9, 0.999];
    const eps = 1e-8;
    const weightDecay = 0.01;

    const wValues = Array.from({ length: 16 }, (_, i) => (i - 8) * 0.1);
    const w = api.tensorFromArray(wValues, [4, 4], { requiresGrad: true });
    const x = api.tensorFromArray(
      Array.from({ length: 8 }, (_, i) => i + 1),
      [2, 4],
    );
    const y = api.tensorFromArray(
      Array.from({ length: 8 }, (_, i) => (i % 3) - 1),
      [2, 4],
    );

    const pred = x.matmul(w);
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.reshape([]).backward();

    if (!w.grad) throw new Error("Missing gradient");
    const gradValues = await w.grad.cpu();

    const optimizer = new Adam([w], { lr, betas, eps, weightDecay }, api);
    const [wUpdated] = optimizer.step();
    const gpuResult = await wUpdated.cpu();

    // CPU reference: one step of AdamW (decoupled weight decay)
    const biasCorrection1 = 1 - betas[0];
    const biasCorrection2 = 1 - betas[1];
    const stepSize = (lr * Math.sqrt(biasCorrection2)) / biasCorrection1;

    const expected = wValues.map((p, i) => {
      const g = gradValues[i];
      const m = (1 - betas[0]) * g;
      const v = (1 - betas[1]) * g * g;
      let pNew = p - (stepSize * m) / (Math.sqrt(v) + eps);
      // AdamW decoupled weight decay
      pNew = pNew - lr * weightDecay * p;
      return pNew;
    });

    assertClose(gpuResult, expected);
  });

  it("scalar fallback: 6-element param still works correctly", { timeout: 30000 }, async () => {
    // 6 elements = NOT divisible by 4, so scalar path is used
    const api = new Torchlette("webgpu");
    const lr = 0.01;
    const betas: [number, number] = [0.9, 0.999];
    const eps = 1e-8;

    const wValues = [0.5, -1.0, 2.0, 1.5, -0.5, 0.25];
    const w = api.tensorFromArray(wValues, [3, 2], { requiresGrad: true });
    const x = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const y = api.tensorFromArray([3, 1, 2, -1], [2, 2]);

    const pred = x.matmul(w);
    const diff = pred.sub(y);
    const loss = diff.mul(diff).mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.reshape([]).backward();

    if (!w.grad) throw new Error("Missing gradient");
    const gradValues = await w.grad.cpu();

    const optimizer = new Adam([w], { lr, betas, eps }, api);
    const [wUpdated] = optimizer.step();
    const gpuResult = await wUpdated.cpu();

    // CPU reference
    const biasCorrection1 = 1 - betas[0];
    const biasCorrection2 = 1 - betas[1];
    const stepSize = (lr * Math.sqrt(biasCorrection2)) / biasCorrection1;

    const expected = wValues.map((p, i) => {
      const g = gradValues[i];
      const m = (1 - betas[0]) * g;
      const v = (1 - betas[1]) * g * g;
      return p - (stepSize * m) / (Math.sqrt(v) + eps);
    });

    assertClose(gpuResult, expected);
  });
});
