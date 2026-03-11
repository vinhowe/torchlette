import { describe, expect, it } from "vitest";

import { Adam, SGD, Torchlette } from "../src";

describe("Parameter Groups", () => {
  const api = new Torchlette("cpu");

  describe("Adam with parameter groups", () => {
    it("applies different LR per group", async () => {
      const w1 = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      const w2 = api.tensorFromArray([3, 4], [2], { requiresGrad: true });

      // Compute gradients
      const loss = api.add(w1.sum(), w2.sum());
      await loss.backward();

      const opt = new Adam(
        [
          { params: [w1], lr: 0.1 },
          { params: [w2], lr: 0.01 },
        ],
        { lr: 0.05 }, // default, overridden by groups
        api,
      );

      expect(opt.numGroups).toBe(2);
      expect(opt.getParamGroupLRs()).toEqual([0.1, 0.01]);

      const updated = opt.step();
      expect(updated.length).toBe(2);

      // Both params should be updated (different amounts due to different LR)
      const v1 = await updated[0].cpu();
      const v2 = await updated[1].cpu();
      // With LR=0.1, w1 moves more than with LR=0.01
      expect(Math.abs(v1[0] - 1)).toBeGreaterThan(Math.abs(v2[0] - 3));
    });

    it("inherits defaults for unset group fields", async () => {
      const w1 = api.tensorFromArray([1], [1], { requiresGrad: true });
      const w2 = api.tensorFromArray([2], [1], { requiresGrad: true });

      const opt = new Adam(
        [
          { params: [w1] }, // inherits lr=0.05, weightDecay=0.1
          { params: [w2], lr: 0.01, weightDecay: 0 },
        ],
        { lr: 0.05, weightDecay: 0.1 },
        api,
      );

      expect(opt.getParamGroupLRs()).toEqual([0.05, 0.01]);
    });

    it("setLR updates all groups", () => {
      const w1 = api.tensorFromArray([1], [1], { requiresGrad: true });
      const w2 = api.tensorFromArray([2], [1], { requiresGrad: true });

      const opt = new Adam(
        [
          { params: [w1], lr: 0.1 },
          { params: [w2], lr: 0.01 },
        ],
        { lr: 0.05 },
        api,
      );

      opt.setLR(0.001);
      expect(opt.getParamGroupLRs()).toEqual([0.001, 0.001]);
    });

    it("setGroupLR updates a single group", () => {
      const w1 = api.tensorFromArray([1], [1], { requiresGrad: true });
      const w2 = api.tensorFromArray([2], [1], { requiresGrad: true });

      const opt = new Adam(
        [
          { params: [w1], lr: 0.1 },
          { params: [w2], lr: 0.01 },
        ],
        { lr: 0.05 },
        api,
      );

      opt.setGroupLR(1, 0.05);
      expect(opt.getParamGroupLRs()).toEqual([0.1, 0.05]);
    });

    it("backward compatible with flat params", async () => {
      const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      const loss = w.sum();
      await loss.backward();

      const opt = new Adam([w], { lr: 0.1 }, api);
      expect(opt.numGroups).toBe(1);
      expect(opt.getLR()).toBeCloseTo(0.1);

      const updated = opt.step();
      expect(updated.length).toBe(1);
    });
  });

  describe("SGD with parameter groups", () => {
    it("applies different LR per group", async () => {
      const w1 = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      const w2 = api.tensorFromArray([3, 4], [2], { requiresGrad: true });

      const loss = api.add(w1.sum(), w2.sum());
      await loss.backward();

      const opt = new SGD(
        [
          { params: [w1], lr: 0.5 },
          { params: [w2], lr: 0.05 },
        ],
        { lr: 0.1 },
        api,
      );

      expect(opt.numGroups).toBe(2);

      const updated = opt.step();
      const v1 = await updated[0].cpu();
      const v2 = await updated[1].cpu();

      // w1 updated with lr=0.5: w1[0] = 1 - 0.5*1 = 0.5
      expect(v1[0]).toBeCloseTo(0.5);
      // w2 updated with lr=0.05: w2[0] = 3 - 0.05*1 = 2.95
      expect(v2[0]).toBeCloseTo(2.95);
    });

    it("backward compatible with flat params", async () => {
      const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      const loss = w.sum();
      await loss.backward();

      const opt = new SGD([w], { lr: 0.1 }, api);
      expect(opt.numGroups).toBe(1);

      const updated = opt.step();
      const v = await updated[0].cpu();
      // w = [1, 2], grad = [1, 1], lr = 0.1
      // w_new = w - lr * grad = [0.9, 1.9]
      expect(v[0]).toBeCloseTo(0.9);
      expect(v[1]).toBeCloseTo(1.9);
    });
  });
});
