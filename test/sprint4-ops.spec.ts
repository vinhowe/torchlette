/**
 * Tests for Sprint 4 operations:
 * - Random tensor generation (rand, randn, bernoulli)
 * - Dropout (functional and module)
 * - Cross-entropy loss
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend";
import { Dropout, dropout, crossEntropy, logSoftmax, nllLoss } from "../src/nn";

describe("Sprint 4 ops (CPU)", () => {
  const api = new Torchlette("cpu");

  describe("Random tensor generation", () => {
    it("rand creates uniform random values in [0, 1)", async () => {
      const t = api.rand([100]);
      const data = await t.cpu();
      expect(data.length).toBe(100);
      // All values should be in [0, 1)
      for (const val of data) {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }
    });

    it("rand creates correct shape", async () => {
      const t = api.rand([2, 3, 4]);
      expect(t.shape).toEqual([2, 3, 4]);
      const data = await t.cpu();
      expect(data.length).toBe(24);
    });

    it("randn creates values from standard normal", async () => {
      const t = api.randn([1000]);
      const data = await t.cpu();
      // Check mean is approximately 0 (with some tolerance)
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(mean).toBeCloseTo(0, 0); // rough check

      // Check some values are negative and some positive
      const hasNegative = data.some(v => v < 0);
      const hasPositive = data.some(v => v > 0);
      expect(hasNegative).toBe(true);
      expect(hasPositive).toBe(true);
    });

    it("bernoulli creates 0s and 1s with correct probability", async () => {
      const t = api.bernoulli([1000], 0.7);
      const data = await t.cpu();
      // All values should be 0 or 1
      for (const val of data) {
        expect(val === 0 || val === 1).toBe(true);
      }
      // Mean should be approximately 0.7
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(mean).toBeCloseTo(0.7, 1); // within 0.1
    });

    it("bernoulli with p=0 creates all zeros", async () => {
      const t = api.bernoulli([100], 0);
      const data = await t.cpu();
      expect(data.every(v => v === 0)).toBe(true);
    });

    it("bernoulli with p=1 creates all ones", async () => {
      const t = api.bernoulli([100], 1);
      const data = await t.cpu();
      expect(data.every(v => v === 1)).toBe(true);
    });

    it("zeros creates all zeros", async () => {
      const t = api.zeros([2, 3]);
      const data = await t.cpu();
      expect(data).toEqual([0, 0, 0, 0, 0, 0]);
    });

    it("ones creates all ones", async () => {
      const t = api.ones([2, 3]);
      const data = await t.cpu();
      expect(data).toEqual([1, 1, 1, 1, 1, 1]);
    });

    it("full creates tensor with specific value", async () => {
      const t = api.full([2, 3], 42);
      const data = await t.cpu();
      expect(data).toEqual([42, 42, 42, 42, 42, 42]);
    });
  });

  describe("Dropout functional", () => {
    it("drops some elements during training", async () => {
      const input = api.ones([100]);
      const output = dropout(api, input, 0.5, true);
      const data = await output.cpu();

      // Some values should be 0 (dropped)
      const numDropped = data.filter(v => v === 0).length;
      expect(numDropped).toBeGreaterThan(0);
      expect(numDropped).toBeLessThan(100);

      // Non-dropped values should be scaled by 1/(1-p) = 2
      const nonDropped = data.filter(v => v !== 0);
      for (const val of nonDropped) {
        expect(val).toBeCloseTo(2, 5);
      }
    });

    it("passes through unchanged during eval", async () => {
      const input = api.ones([100]);
      const output = dropout(api, input, 0.5, false);
      const data = await output.cpu();
      // All values should be 1 (unchanged)
      expect(data.every(v => v === 1)).toBe(true);
    });

    it("handles p=0 (no dropout)", async () => {
      const input = api.tensorFromArray([1, 2, 3, 4], [4]);
      const output = dropout(api, input, 0, true);
      const data = await output.cpu();
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("handles p=1 (all dropped)", async () => {
      const input = api.tensorFromArray([1, 2, 3, 4], [4]);
      const output = dropout(api, input, 1, true);
      const data = await output.cpu();
      expect(data).toEqual([0, 0, 0, 0]);
    });
  });

  describe("Dropout module", () => {
    it("drops elements when training", async () => {
      const drop = new Dropout(api, { p: 0.5 });
      drop.train();
      const input = api.ones([100]);
      const output = drop.forward(input);
      const data = await output.cpu();

      // Some should be dropped
      const numDropped = data.filter(v => v === 0).length;
      expect(numDropped).toBeGreaterThan(0);
    });

    it("passes through when eval", async () => {
      const drop = new Dropout(api, { p: 0.5 });
      drop.eval();
      const input = api.ones([100]);
      const output = drop.forward(input);
      const data = await output.cpu();
      expect(data.every(v => v === 1)).toBe(true);
    });

    it("toggles train/eval mode", () => {
      const drop = new Dropout(api);
      expect(drop.training).toBe(true);
      drop.eval();
      expect(drop.training).toBe(false);
      drop.train();
      expect(drop.training).toBe(true);
      drop.train(false);
      expect(drop.training).toBe(false);
    });
  });

  describe("logSoftmax", () => {
    it("computes log of softmax", async () => {
      const input = api.tensorFromArray([1, 2, 3], [3]);
      const logSm = logSoftmax(api, input, -1);
      const sm = input.softmax(-1);

      const logSmData = await logSm.cpu();
      const smData = await sm.cpu();

      // log(softmax(x)) should match
      for (let i = 0; i < 3; i++) {
        expect(logSmData[i]).toBeCloseTo(Math.log(smData[i]), 5);
      }
    });

    it("sums to log(1) = 0 when exp-ed and summed", async () => {
      const input = api.tensorFromArray([1, 2, 3], [3]);
      const logSm = logSoftmax(api, input, -1);
      const expLogSm = logSm.exp();
      const sum = expLogSm.sum();
      if (typeof sum === "number") {
        expect(sum).toBeCloseTo(1, 5);
      } else {
        const data = await sum.cpu();
        expect(data[0]).toBeCloseTo(1, 5);
      }
    });
  });

  describe("crossEntropy", () => {
    it("computes loss for simple 1D case", async () => {
      // Logits for 3 classes: [1, 2, 3]
      // Target: class 2 (index 2)
      const logits = api.tensorFromArray([1, 2, 3], [3]);
      const targets = api.tensorFromArray([2], [1]);
      const loss = crossEntropy(api, logits, targets);
      const lossValue = await loss.cpu();

      // Manual calculation:
      // softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
      // -log(0.67) ≈ 0.41
      expect(lossValue[0]).toBeCloseTo(0.41, 1);
    });

    it("computes batch loss with mean reduction", async () => {
      // Batch of 2, 3 classes
      const logits = api.tensorFromArray([
        1, 2, 3,  // sample 1: should predict class 2
        3, 2, 1,  // sample 2: should predict class 0
      ], [2, 3]);
      const targets = api.tensorFromArray([2, 0], [2]);
      const loss = crossEntropy(api, logits, targets, { reduction: "mean" });
      const lossValue = await loss.cpu();

      // Both samples have correct predictions (argmax matches target)
      // Loss should be relatively low
      expect(lossValue[0]).toBeLessThan(1);
    });

    it("computes batch loss with sum reduction", async () => {
      const logits = api.tensorFromArray([1, 2, 3, 3, 2, 1], [2, 3]);
      const targets = api.tensorFromArray([2, 0], [2]);
      const lossMean = crossEntropy(api, logits, targets, { reduction: "mean" });
      const lossSum = crossEntropy(api, logits, targets, { reduction: "sum" });

      const meanVal = await lossMean.cpu();
      const sumVal = await lossSum.cpu();

      // Sum should be 2 * mean for batch size 2
      expect(sumVal[0]).toBeCloseTo(meanVal[0] * 2, 5);
    });

    it("computes batch loss with no reduction", async () => {
      const logits = api.tensorFromArray([1, 2, 3, 3, 2, 1], [2, 3]);
      const targets = api.tensorFromArray([2, 0], [2]);
      const loss = crossEntropy(api, logits, targets, { reduction: "none" });

      expect(loss.shape).toEqual([2]);
      const lossData = await loss.cpu();
      expect(lossData.length).toBe(2);
    });

    it("is differentiable", async () => {
      const logits = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
      const targets = api.tensorFromArray([2], [1]);
      const loss = crossEntropy(api, logits, targets);
      await loss.backward();

      expect(logits.grad).not.toBeNull();
      const grad = await logits.grad?.cpu();
      expect(grad?.length).toBe(3);

      // Gradient should be: softmax(logits) - one_hot(target)
      // For target=2: grad ≈ [softmax[0], softmax[1], softmax[2] - 1]
      // Sum of gradients should be 0
      const gradSum = grad?.reduce((a, b) => a + b, 0);
      expect(gradSum).toBeCloseTo(0, 4);
    });

    it("penalizes wrong predictions more", async () => {
      // Correct prediction: high logit at target index
      const correctLogits = api.tensorFromArray([0, 0, 10], [3]);
      const targets = api.tensorFromArray([2], [1]);
      const correctLoss = crossEntropy(api, correctLogits, targets);

      // Wrong prediction: high logit at wrong index
      const wrongLogits = api.tensorFromArray([10, 0, 0], [3]);
      const wrongLoss = crossEntropy(api, wrongLogits, targets);

      const correctVal = await correctLoss.cpu();
      const wrongVal = await wrongLoss.cpu();

      expect(wrongVal[0]).toBeGreaterThan(correctVal[0]);
    });
  });

  describe("nllLoss", () => {
    it("computes NLL loss from log probabilities", async () => {
      const logProbs = api.tensorFromArray([
        Math.log(0.1), Math.log(0.2), Math.log(0.7),
      ], [3]);
      const targets = api.tensorFromArray([2], [1]);
      const loss = nllLoss(api, logProbs, targets);
      const lossValue = await loss.cpu();

      // -log(0.7) ≈ 0.357
      expect(lossValue[0]).toBeCloseTo(-Math.log(0.7), 5);
    });
  });
});
