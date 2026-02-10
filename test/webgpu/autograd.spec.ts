/**
 * WebGPU Backend Autograd Tests
 *
 * These tests verify that autograd (backward pass) works correctly
 * when using the WebGPU backend for GPU-accelerated computation.
 */

import { describe, expect, it, beforeAll } from "vitest";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu/index";
import { resetNodeIdCounter, resetStorageIdCounter } from "../../src/engine/lazy";
import { resetBaseIdCounter } from "../../src/runtime/tensor";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("WebGPU Backend Autograd", () => {
  beforeAll(async () => {
    const success = await initWebGPU();
    if (!success) {
      throw new Error("WebGPU not available");
    }
  });

  describe("Basic Operations", () => {
    it("backward through add on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      const b = api.tensorFromArray([5, 6, 7, 8], [4], { requiresGrad: true });
      const c = a.add(b);
      const loss = c.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1]);
      expect(await b.grad?.cpu()).toEqual([1, 1, 1, 1]);
    });

    it("backward through mul on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const a = api.tensorFromArray([2, 3, 4, 5], [4], { requiresGrad: true });
      const b = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      const c = a.mul(b);
      const loss = c.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      // d(a*b)/da = b, d(a*b)/db = a
      expect(await a.grad?.cpu()).toEqual([1, 2, 3, 4]); // grad_a = b
      expect(await b.grad?.cpu()).toEqual([2, 3, 4, 5]); // grad_b = a
    });

    it("backward through matmul on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const a = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
      const b = api.tensorFromArray([5, 6, 7, 8], [2, 2], { requiresGrad: true });
      const c = a.matmul(b);
      const loss = c.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(a.grad).not.toBeNull();
      expect(b.grad).not.toBeNull();
      // Gradient shapes should match input shapes
      expect(a.grad?.shape).toEqual([2, 2]);
      expect(b.grad?.shape).toEqual([2, 2]);
    });

    it("backward through relu on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const a = api.tensorFromArray([1, -1, 2, -2], [4], { requiresGrad: true });
      const b = a.relu();
      const loss = b.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(a.grad).not.toBeNull();
      // Gradient is 1 where input > 0, 0 otherwise
      expect(await a.grad?.cpu()).toEqual([1, 0, 1, 0]);
    });
  });

  describe("Complex Chains", () => {
    it("MLP-like forward backward on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      // Simulate a simple 2-layer MLP: input -> W1 -> relu -> W2 -> output
      const input = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: false });
      const W1 = api.tensorFromArray([0.5, -0.5, 0.3, 0.7], [2, 2], { requiresGrad: true });
      const W2 = api.tensorFromArray([0.1, 0.2, -0.1, 0.3], [2, 2], { requiresGrad: true });

      // Forward pass
      const h1 = input.matmul(W1);
      const h1Act = h1.relu();
      const output = h1Act.matmul(W2);
      const loss = output.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      // Backward pass
      await loss.backward();

      // Both weights should have gradients
      expect(W1.grad).not.toBeNull();
      expect(W2.grad).not.toBeNull();
      expect(W1.grad?.shape).toEqual([2, 2]);
      expect(W2.grad?.shape).toEqual([2, 2]);
    });

    it("residual connection backward on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      // ResNet-style residual: output = x + f(x)
      const x = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      const w = api.tensorFromArray([0.5, 0.5, 0.5, 0.5], [4], { requiresGrad: true });

      // f(x) = w * x
      const fx = x.mul(w);
      // residual = x + f(x)
      const residual = x.add(fx);
      const loss = residual.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // Gradient of x should be 1 + w (from both direct path and through f)
      expect(x.grad).not.toBeNull();
      const xGrad = await x.grad?.cpu();
      expect(xGrad?.[0]).toBeCloseTo(1.5, 5); // 1 + 0.5
      expect(xGrad?.[1]).toBeCloseTo(1.5, 5);
    });
  });

  describe("View Operations", () => {
    it("backward through transpose on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
      const b = a.transpose({ dim0: 0, dim1: 1 });
      const loss = b.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(a.grad).not.toBeNull();
      expect(a.grad?.shape).toEqual([2, 3]);
      expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1, 1, 1]);
    });

    it("backward through expand on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");

      const a = api.tensorFromArray([1, 2], [2, 1], { requiresGrad: true });
      const b = a.expand([2, 3]);
      const loss = b.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(a.grad).not.toBeNull();
      expect(a.grad?.shape).toEqual([2, 1]);
      // Each value was expanded 3 times, so gradient sums to 3
      expect(await a.grad?.cpu()).toEqual([3, 3]);
    });
  });

  describe("Training Loop", () => {
    it("multiple training steps on WebGPU", async () => {
      resetNodeIdCounter();
      resetStorageIdCounter();
      resetBaseIdCounter();
      const api = new Torchlette("webgpu");
      const { SGD } = await import("../../src/optim");

      const w = api.tensorFromArray([10, 10, 10, 10], [4], { requiresGrad: true });
      const optimizer = new SGD([w], { lr: 1.0 }, api);

      // Multiple training steps
      for (let i = 0; i < 3; i++) {
        optimizer.zeroGrad();

        const currentW = optimizer.getParams()[0];
        const loss = currentW.sum();
        if (typeof loss === "number") throw new Error("Expected tensor");
        await loss.backward();
        optimizer.step();
      }

      // After 3 steps with lr=1.0 and grad=1, w should be 10 - 3 = 7
      const finalW = optimizer.getParams()[0];
      const finalWeights = await finalW.cpu();
      expect(finalWeights[0]).toBeCloseTo(7, 4);
      expect(finalWeights[1]).toBeCloseTo(7, 4);
      expect(finalWeights[2]).toBeCloseTo(7, 4);
      expect(finalWeights[3]).toBeCloseTo(7, 4);
    });
  });
});
