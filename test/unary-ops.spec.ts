/**
 * Tests for unary activation functions and math ops
 * exp, log, neg, abs, tanh, sigmoid, gelu, silu
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend";

describe("Unary ops (CPU)", () => {
  const api = new Torchlette("cpu");

  describe("exp", () => {
    it("computes element-wise exponential", async () => {
      const a = api.tensorFromArray([0, 1, 2, -1], [4]);
      const result = a.exp();
      const data = await result.cpu();
      expect(data[0]).toBeCloseTo(1.0, 5); // e^0 = 1
      expect(data[1]).toBeCloseTo(Math.E, 5); // e^1 = e
      expect(data[2]).toBeCloseTo(Math.E * Math.E, 5); // e^2
      expect(data[3]).toBeCloseTo(1 / Math.E, 5); // e^-1
    });

    it("exp backward pass", async () => {
      const a = api.tensorFromArray([0, 1, 2], [3], { requiresGrad: true });
      const result = a.exp();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx exp(x) = exp(x)
      expect(grad?.[0]).toBeCloseTo(1.0, 5); // exp(0) = 1
      expect(grad?.[1]).toBeCloseTo(Math.E, 5); // exp(1) = e
      expect(grad?.[2]).toBeCloseTo(Math.E * Math.E, 4); // exp(2)
    });
  });

  describe("log", () => {
    it("computes element-wise natural log", async () => {
      const a = api.tensorFromArray([1, Math.E, Math.E * Math.E, 0.5], [4]);
      const result = a.log();
      const data = await result.cpu();
      expect(data[0]).toBeCloseTo(0, 5); // log(1) = 0
      expect(data[1]).toBeCloseTo(1, 5); // log(e) = 1
      expect(data[2]).toBeCloseTo(2, 5); // log(e^2) = 2
      expect(data[3]).toBeCloseTo(Math.log(0.5), 5);
    });

    it("log backward pass", async () => {
      const a = api.tensorFromArray([1, 2, 4], [3], { requiresGrad: true });
      const result = a.log();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx log(x) = 1/x
      expect(grad?.[0]).toBeCloseTo(1.0, 5); // 1/1
      expect(grad?.[1]).toBeCloseTo(0.5, 5); // 1/2
      expect(grad?.[2]).toBeCloseTo(0.25, 5); // 1/4
    });
  });

  describe("neg", () => {
    it("computes element-wise negation", async () => {
      const a = api.tensorFromArray([1, -2, 3, 0], [4]);
      const result = a.neg();
      const data = await result.cpu();
      expect(data[0]).toBe(-1);
      expect(data[1]).toBe(2);
      expect(data[2]).toBe(-3);
      expect(Object.is(data[3], -0) || data[3] === 0).toBe(true); // -0 or 0
    });

    it("neg backward pass", async () => {
      const a = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
      const result = a.neg();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx (-x) = -1
      expect(grad).toEqual([-1, -1, -1]);
    });
  });

  describe("abs", () => {
    it("computes element-wise absolute value", async () => {
      const a = api.tensorFromArray([1, -2, 3, -4], [4]);
      const result = a.abs();
      const data = await result.cpu();
      expect(data).toEqual([1, 2, 3, 4]);
    });

    it("abs backward pass", async () => {
      const a = api.tensorFromArray([2, -3, 0], [3], { requiresGrad: true });
      const result = a.abs();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx |x| = sign(x)
      expect(grad?.[0]).toBe(1); // positive
      expect(grad?.[1]).toBe(-1); // negative
      expect(grad?.[2]).toBe(1); // zero (we use >= 0)
    });
  });

  describe("tanh", () => {
    it("computes element-wise tanh", async () => {
      const a = api.tensorFromArray([0, 1, -1, 2], [4]);
      const result = a.tanh();
      const data = await result.cpu();
      expect(data[0]).toBeCloseTo(0, 5); // tanh(0) = 0
      expect(data[1]).toBeCloseTo(Math.tanh(1), 5);
      expect(data[2]).toBeCloseTo(Math.tanh(-1), 5);
      expect(data[3]).toBeCloseTo(Math.tanh(2), 5);
    });

    it("tanh backward pass", async () => {
      const a = api.tensorFromArray([0, 1], [2], { requiresGrad: true });
      const result = a.tanh();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx tanh(x) = 1 - tanh(x)^2
      expect(grad?.[0]).toBeCloseTo(1.0, 5); // 1 - tanh(0)^2 = 1 - 0 = 1
      expect(grad?.[1]).toBeCloseTo(1 - Math.tanh(1) ** 2, 5);
    });
  });

  describe("sigmoid", () => {
    it("computes element-wise sigmoid", async () => {
      const a = api.tensorFromArray([0, 1, -1, 100], [4]);
      const result = a.sigmoid();
      const data = await result.cpu();
      expect(data[0]).toBeCloseTo(0.5, 5); // sigmoid(0) = 0.5
      expect(data[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 5);
      expect(data[2]).toBeCloseTo(1 / (1 + Math.exp(1)), 5);
      expect(data[3]).toBeCloseTo(1.0, 5); // sigmoid(large) ≈ 1
    });

    it("sigmoid backward pass", async () => {
      const a = api.tensorFromArray([0, 1], [2], { requiresGrad: true });
      const result = a.sigmoid();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
      const sig0 = 0.5;
      const sig1 = 1 / (1 + Math.exp(-1));
      expect(grad?.[0]).toBeCloseTo(sig0 * (1 - sig0), 5); // 0.25
      expect(grad?.[1]).toBeCloseTo(sig1 * (1 - sig1), 5);
    });
  });

  describe("gelu", () => {
    it("computes element-wise GELU", async () => {
      const a = api.tensorFromArray([0, 1, -1, 2], [4]);
      const result = a.gelu();
      const data = await result.cpu();
      // GELU(0) ≈ 0
      expect(data[0]).toBeCloseTo(0, 3);
      // GELU(x) ≈ x for large positive x
      expect(data[3]).toBeCloseTo(2 * 0.5 * (1 + Math.tanh(0.7978845608 * (2 + 0.044715 * 8))), 3);
    });

    it("gelu backward pass", async () => {
      const a = api.tensorFromArray([0, 1], [2], { requiresGrad: true });
      const result = a.gelu();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      // Just check gradients exist and are reasonable
      const grad = await a.grad?.cpu();
      expect(grad?.[0]).toBeCloseTo(0.5, 1); // GELU'(0) ≈ 0.5
    });
  });

  describe("silu", () => {
    it("computes element-wise SiLU/Swish", async () => {
      const a = api.tensorFromArray([0, 1, -1, 2], [4]);
      const result = a.silu();
      const data = await result.cpu();
      // SiLU(0) = 0 * sigmoid(0) = 0
      expect(data[0]).toBeCloseTo(0, 5);
      // SiLU(x) = x * sigmoid(x)
      expect(data[1]).toBeCloseTo(1 / (1 + Math.exp(-1)), 5);
      expect(data[2]).toBeCloseTo(-1 / (1 + Math.exp(1)), 5);
    });

    it("silu backward pass", async () => {
      const a = api.tensorFromArray([0, 1], [2], { requiresGrad: true });
      const result = a.silu();
      const loss = result.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();

      expect(a.grad).not.toBeNull();
      const grad = await a.grad?.cpu();
      // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
      const sig0 = 0.5;
      expect(grad?.[0]).toBeCloseTo(sig0, 4); // At x=0: 0.5 + 0 = 0.5
    });
  });
});
