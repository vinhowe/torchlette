import { describe, expect, it } from "vitest";

import {
  Adam,
  CosineAnnealingLR,
  ExponentialLR,
  PolynomialLR,
  SGD,
  StepLR,
  Torchlette,
} from "../src";

describe("LR Schedulers", () => {
  function makeOptimizer() {
    const api = new Torchlette("cpu");
    const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
    return { api, opt: new Adam([w], { lr: 0.1 }, api) };
  }

  describe("StepLR", () => {
    it("decays LR by gamma every stepSize epochs", () => {
      const { opt } = makeOptimizer();
      const scheduler = new StepLR(opt, 3, 0.1);
      expect(opt.getLR()).toBeCloseTo(0.1);

      // Epochs 1,2: no decay yet (floor(1/3) = 0, floor(2/3) = 0)
      scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0.1); // 0.1 * 0.1^0
      scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0.1);

      // Epoch 3: first decay (floor(3/3) = 1)
      scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0.01); // 0.1 * 0.1^1

      // Epochs 4,5: still at first decay
      scheduler.step();
      scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0.01);

      // Epoch 6: second decay
      scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0.001); // 0.1 * 0.1^2
    });

    it("getLastLR matches optimizer LR", () => {
      const { opt } = makeOptimizer();
      const scheduler = new StepLR(opt, 2, 0.5);
      scheduler.step();
      expect(scheduler.getLastLR()).toBeCloseTo(opt.getLR());
      scheduler.step();
      expect(scheduler.getLastLR()).toBeCloseTo(opt.getLR());
    });
  });

  describe("ExponentialLR", () => {
    it("decays LR by gamma each epoch", () => {
      const { opt } = makeOptimizer();
      const scheduler = new ExponentialLR(opt, 0.9);

      scheduler.step(); // epoch 1
      expect(opt.getLR()).toBeCloseTo(0.09); // 0.1 * 0.9^1
      scheduler.step(); // epoch 2
      expect(opt.getLR()).toBeCloseTo(0.081); // 0.1 * 0.9^2
      scheduler.step(); // epoch 3
      expect(opt.getLR()).toBeCloseTo(0.0729); // 0.1 * 0.9^3
    });
  });

  describe("CosineAnnealingLR", () => {
    it("follows cosine schedule from baseLR to etaMin", () => {
      const { opt } = makeOptimizer();
      const scheduler = new CosineAnnealingLR(opt, 10, 0.001);

      // At epoch tMax/2 = 5, LR should be midpoint
      for (let i = 0; i < 5; i++) scheduler.step();
      const midLR =
        0.001 + ((0.1 - 0.001) * (1 + Math.cos((Math.PI * 5) / 10))) / 2;
      expect(opt.getLR()).toBeCloseTo(midLR);

      // At epoch tMax = 10, LR should be etaMin
      for (let i = 0; i < 5; i++) scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0.001, 5);
    });

    it("defaults etaMin to 0", () => {
      const { opt } = makeOptimizer();
      const scheduler = new CosineAnnealingLR(opt, 4);

      for (let i = 0; i < 4; i++) scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0, 5);
    });
  });

  describe("PolynomialLR", () => {
    it("decays LR polynomially to 0", () => {
      const { opt } = makeOptimizer();
      const scheduler = new PolynomialLR(opt, 10, 2.0);

      scheduler.step(); // epoch 1
      expect(opt.getLR()).toBeCloseTo(0.1 * (1 - 1 / 10) ** 2);

      scheduler.step(); // epoch 2
      expect(opt.getLR()).toBeCloseTo(0.1 * (1 - 2 / 10) ** 2);

      // At totalIters, LR should be 0
      for (let i = 0; i < 8; i++) scheduler.step();
      expect(opt.getLR()).toBeCloseTo(0, 5);
    });

    it("clamps epoch at totalIters", () => {
      const { opt } = makeOptimizer();
      const scheduler = new PolynomialLR(opt, 5);

      for (let i = 0; i < 10; i++) scheduler.step();
      // Past totalIters, LR stays at 0
      expect(opt.getLR()).toBeCloseTo(0, 5);
    });
  });

  describe("works with SGD", () => {
    it("SGD has getLR/setLR", () => {
      const api = new Torchlette("cpu");
      const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
      const opt = new SGD([w], { lr: 0.05 }, api);
      expect(opt.getLR()).toBeCloseTo(0.05);
      opt.setLR(0.01);
      expect(opt.getLR()).toBeCloseTo(0.01);
    });
  });
});
