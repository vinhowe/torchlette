import { describe, expect, it } from "vitest";

import { nn, Torchlette } from "../src";

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

      const totalNorm = await nn.clipGradNorm_(api, [a, b], 1.0);
      expect(totalNorm).toBeCloseTo(2.0, 1);

      // After clipping, grads should be scaled by 1.0 / (2.0 + 1e-6)
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

      const totalNorm = await nn.clipGradNorm_(api, [a], 10.0);
      expect(totalNorm).toBeCloseTo(Math.sqrt(2), 3);

      // Grads should be unchanged
      const gradA = await a.grad!.cpu();
      expect(gradA[0]).toBeCloseTo(1.0, 5);
      expect(gradA[1]).toBeCloseTo(1.0, 5);
    });

    it("returns 0 when no parameters have gradients", async () => {
      const a = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      // No backward called, so no gradients
      const totalNorm = await nn.clipGradNorm_(api, [a], 1.0);
      expect(totalNorm).toBe(0);
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
