import { beforeAll, describe, expect, it } from "vitest";

import { nn, Torchlette } from "../src";
import { initWebGPU } from "../src/backend/webgpu";
import { cpuOnly } from "./helpers/webgpu";

describe("Gradient Clipping", () => {
  const api = new Torchlette("cpu");

  describe("clipGradNorm_", () => {
    it("clips gradients when total norm exceeds maxNorm", async () => {
      const w1 = api.tensorFromArray([3, 4], [2], { requiresGrad: true });
      const w2 = api.tensorFromArray([0, 0], [2], { requiresGrad: true });

      // Create gradients: w1.grad = [3, 4] (norm = 5)
      const loss = w1.sum();
      await loss.backward();
      // w1.grad is all ones from sum backward, so set explicit grads
      // Actually, sum backward gives grad = [1, 1], norm = sqrt(2)

      // Let's manually create a scenario with known norms
      const a = api.tensorFromArray([3, 4], [2], { requiresGrad: true });
      const b = api.tensorFromArray([0, 0], [2], { requiresGrad: true });
      const loss2 = api.add(a.mul(api.tensorFromArray([1, 1], [2])), b).sum();
      await loss2.backward();
      // a.grad = [1, 1], b.grad = [1, 1]
      // total L2 norm = sqrt(1+1+1+1) = 2

      await nn.clipGradNorm_(api, [a, b], 1.0);

      // After clipping with maxNorm=1.0 and totalNorm=2.0,
      // grads should be scaled by 1.0 / (2.0 + 1e-6)
      const gradA = await a.grad!.cpu();
      const scale = 1.0 / (2.0 + 1e-6);
      expect(gradA[0]).toBeCloseTo(scale, 3);
      expect(gradA[1]).toBeCloseTo(scale, 3);
    });

    it("does not clip when norm is below maxNorm", async () => {
      const a = api.tensorFromArray([1, 0], [2], { requiresGrad: true });
      const loss = a.sum();
      await loss.backward();
      // a.grad = [1, 1], norm = sqrt(2) ≈ 1.414

      await nn.clipGradNorm_(api, [a], 10.0);

      // Norm = sqrt(2) < maxNorm=10, so grads should be unchanged
      const gradA = await a.grad!.cpu();
      expect(gradA[0]).toBeCloseTo(1.0, 5);
      expect(gradA[1]).toBeCloseTo(1.0, 5);
    });

    it("is a no-op when no parameters have gradients", async () => {
      const a = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      // No backward called, so no gradients — should not throw
      await nn.clipGradNorm_(api, [a], 1.0);
    });
  });

  describe("clipGradValue_", () => {
    it("clamps gradient values to [-clipValue, clipValue]", async () => {
      const a = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
      // Create large gradients
      const loss = a.mul(api.tensorFromArray([10, -20, 5], [3])).sum();
      await loss.backward();
      // a.grad = [10, -20, 5]

      nn.clipGradValue_(api, [a], 8.0);
      const grad = await a.grad!.cpu();
      expect(grad[0]).toBeCloseTo(8.0); // clamped from 10
      expect(grad[1]).toBeCloseTo(-8.0); // clamped from -20
      expect(grad[2]).toBeCloseTo(5.0); // unchanged
    });

    it("handles parameters without gradients", () => {
      const a = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      // No error should be thrown
      nn.clipGradValue_(api, [a], 1.0);
    });
  });
});

// WebGPU-specific: the L2 norm squares the gradients. WGSL's pow(x,y) =
// exp2(y*log2(x)) is NaN for x<0, so the original pow(g,2) poisoned the norm
// for any tensor with negative grad entries (i.e. essentially always) on the
// GPU — silently breaking clipping. The CPU backend uses JS ** (correct), so
// this only reproduces on WebGPU. Guards the pow(g,2)->mul(g,g) fix and the
// integer-pow lowering. See src/nn/clip-grad.ts and src/frontend/torchlette.ts.
describe.skipIf(cpuOnly)("Gradient Clipping (WebGPU, negative grads)", () => {
  const api = new Torchlette("webgpu");
  beforeAll(async () => {
    if (!(await initWebGPU())) throw new Error("WebGPU init failed");
  });

  it("clips correctly with negative gradients (no NaN from pow(g,2))", async () => {
    const a = api.tensorFromArray([1, 1], [2], { requiresGrad: true });
    // grad(a) = [-3, 4]  => total L2 norm = 5 (would be NaN if pow(-3,2)=NaN)
    const loss = api.mul(a, api.tensorFromArray([-3, 4], [2])).sum();
    await loss.backward();
    await nn.clipGradNorm_(api, [a], 1.0);
    const g = await a.grad!.cpu();
    const scale = 1.0 / (5.0 + 1e-6);
    expect(Number.isNaN(g[0]!)).toBe(false);
    expect(g[0]!).toBeCloseTo(-3 * scale, 4);
    expect(g[1]!).toBeCloseTo(4 * scale, 4);
  });

  it("pow(x, 2) and pow(x, 3) are exact for negative x", async () => {
    const x = api.tensorFromArray([-2, 3, -1], [3], { device: "webgpu" });
    const sq = await api.pow(x, 2).cpu();
    const cube = await api.pow(x, 3).cpu();
    expect(Array.from(sq)).toEqual([4, 9, 1]);
    expect(Array.from(cube)).toEqual([-8, 27, -1]);
  });
});
