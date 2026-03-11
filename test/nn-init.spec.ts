import { describe, expect, it } from "vitest";

import { nn, Torchlette } from "../src";

const { init } = nn;

describe("nn.init", () => {
  const api = new Torchlette("cpu");

  describe("calculateFanInAndFanOut", () => {
    it("computes fan_in and fan_out for 2D weight", () => {
      const w = api.zeros([64, 32]); // [outFeatures, inFeatures]
      const { fanIn, fanOut } = init.calculateFanInAndFanOut(w);
      expect(fanIn).toBe(32);
      expect(fanOut).toBe(64);
    });

    it("includes receptive field for 4D conv weight", () => {
      const w = api.zeros([16, 3, 5, 5]); // [outChannels, inChannels, kH, kW]
      const { fanIn, fanOut } = init.calculateFanInAndFanOut(w);
      expect(fanIn).toBe(3 * 25); // 3 * 5 * 5
      expect(fanOut).toBe(16 * 25);
    });

    it("throws for 1D tensor", () => {
      const w = api.zeros([10]);
      expect(() => init.calculateFanInAndFanOut(w)).toThrow();
    });
  });

  describe("calculateGain", () => {
    it("returns correct gains", () => {
      expect(init.calculateGain("linear")).toBe(1);
      expect(init.calculateGain("sigmoid")).toBe(1);
      expect(init.calculateGain("tanh")).toBeCloseTo(5 / 3);
      expect(init.calculateGain("relu")).toBeCloseTo(Math.sqrt(2));
      expect(init.calculateGain("leaky_relu", 0.2)).toBeCloseTo(
        Math.sqrt(2 / (1 + 0.04)),
      );
    });

    it("throws for unknown nonlinearity", () => {
      expect(() => init.calculateGain("unknown")).toThrow();
    });
  });

  describe("constant_, zeros_, ones_", () => {
    it("fills with constant", async () => {
      const t = api.zeros([3, 4]);
      init.constant_(api, t, 42);
      const data = await t.cpu();
      expect(data.every((v) => v === 42)).toBe(true);
    });

    it("fills with zeros", async () => {
      const t = api.ones([2, 3]);
      init.zeros_(api, t);
      const data = await t.cpu();
      expect(data.every((v) => v === 0)).toBe(true);
    });

    it("fills with ones", async () => {
      const t = api.zeros([2, 3]);
      init.ones_(api, t);
      const data = await t.cpu();
      expect(data.every((v) => v === 1)).toBe(true);
    });
  });

  describe("normal_", () => {
    it("fills with approximately normal distribution", async () => {
      const t = api.zeros([1000]);
      init.normal_(api, t, 0, 1);
      const data = await t.cpu();
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const variance =
        data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
      // With 1000 samples, mean and variance should be close to 0 and 1
      expect(Math.abs(mean)).toBeLessThan(0.15);
      expect(Math.abs(variance - 1)).toBeLessThan(0.3);
    });

    it("respects mean and std parameters", async () => {
      const t = api.zeros([2000]);
      init.normal_(api, t, 5, 0.1);
      const data = await t.cpu();
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean - 5)).toBeLessThan(0.05);
    });
  });

  describe("uniform_", () => {
    it("fills in range [low, high)", async () => {
      const t = api.zeros([1000]);
      init.uniform_(api, t, -2, 2);
      const data = await t.cpu();
      expect(data.every((v) => v >= -2 && v <= 2)).toBe(true);
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(Math.abs(mean)).toBeLessThan(0.3); // Should be ~0
    });
  });

  describe("kaimingNormal_", () => {
    it("fills with correct std for fan_in relu", async () => {
      const t = api.zeros([128, 64]);
      init.kaimingNormal_(api, t, {
        mode: "fan_in",
        nonlinearity: "relu",
      });
      const data = await t.cpu();
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const variance =
        data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
      // Expected std = sqrt(2) / sqrt(64) = sqrt(2/64)
      const expectedVar = 2 / 64;
      expect(Math.abs(mean)).toBeLessThan(0.05);
      expect(Math.abs(variance - expectedVar)).toBeLessThan(expectedVar * 0.3);
    });
  });

  describe("kaimingUniform_", () => {
    it("fills in correct range", async () => {
      const t = api.zeros([128, 64]);
      init.kaimingUniform_(api, t, {
        mode: "fan_in",
        nonlinearity: "relu",
      });
      const data = await t.cpu();
      const bound = Math.sqrt(2) * Math.sqrt(3 / 64);
      expect(data.every((v) => v >= -bound - 0.01 && v <= bound + 0.01)).toBe(
        true,
      );
    });
  });

  describe("xavierUniform_", () => {
    it("fills in correct range", async () => {
      const t = api.zeros([128, 64]);
      init.xavierUniform_(api, t);
      const data = await t.cpu();
      const bound = Math.sqrt(6 / (64 + 128));
      expect(data.every((v) => v >= -bound - 0.01 && v <= bound + 0.01)).toBe(
        true,
      );
    });
  });

  describe("xavierNormal_", () => {
    it("fills with correct std", async () => {
      const t = api.zeros([128, 64]);
      init.xavierNormal_(api, t);
      const data = await t.cpu();
      const mean = data.reduce((a, b) => a + b, 0) / data.length;
      const variance =
        data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length;
      const expectedVar = 2 / (64 + 128);
      expect(Math.abs(mean)).toBeLessThan(0.05);
      expect(Math.abs(variance - expectedVar)).toBeLessThan(expectedVar * 0.4);
    });
  });
});
