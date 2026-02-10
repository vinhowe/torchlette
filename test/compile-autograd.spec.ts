/**
 * AOT Autograd Verification Tests (§0.1.5)
 *
 * These tests verify that autograd works correctly through:
 * - Lazy execution pipeline
 * - Fusion-enabled execution
 * - Complex operation chains
 * - Gradient accumulation
 * - Multiple backward passes
 * - Saved-for-backward correctness
 */

import { describe, expect, it, beforeEach } from "vitest";
import { Torchlette, torch, Tensor } from "../src/frontend";
import { RuntimeEngine } from "../src/runtime/engine";
import { resetNodeIdCounter, resetStorageIdCounter } from "../src/engine/lazy";
import { resetBaseIdCounter } from "../src/runtime/tensor";

describe("AOT Autograd: Basic Forward+Backward", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("backward through add", async () => {
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

  it("backward through mul", async () => {
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

  it("backward through matmul", async () => {
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

  it("backward through relu", async () => {
    const a = api.tensorFromArray([1, -1, 2, -2], [4], { requiresGrad: true });
    const b = a.relu();
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    // Gradient is 1 where input > 0, 0 otherwise
    expect(await a.grad?.cpu()).toEqual([1, 0, 1, 0]);
  });

  it("backward through sqrt", async () => {
    const a = api.tensorFromArray([4, 9, 16, 25], [4], { requiresGrad: true });
    const b = a.sqrt();
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    // d(sqrt(x))/dx = 0.5 / sqrt(x)
    const grad = await a.grad?.cpu();
    expect(grad?.[0]).toBeCloseTo(0.25, 5);  // 0.5/2
    expect(grad?.[1]).toBeCloseTo(0.5 / 3, 5);
    expect(grad?.[2]).toBeCloseTo(0.125, 5); // 0.5/4
    expect(grad?.[3]).toBeCloseTo(0.1, 5);   // 0.5/5
  });
});

describe("AOT Autograd: Complex Chains", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("backward through add -> mul -> sqrt chain", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
    const b = api.tensorFromArray([3, 2, 1, 0], [4], { requiresGrad: true });

    // c = a + b = [4, 4, 4, 4]
    // d = c * b = [12, 8, 4, 0]
    // e = sqrt(d) = [3.46, 2.83, 2, 0]
    const c = a.add(b);
    const d = c.mul(b);
    // Add small constant to avoid sqrt(0)
    const dSafe = d.add(api.tensorFromArray([0.0001, 0.0001, 0.0001, 0.0001], [4]));
    const e = dSafe.sqrt();
    const loss = e.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();
    // Gradients should flow through the entire chain
    expect(a.grad?.shape).toEqual([4]);
    expect(b.grad?.shape).toEqual([4]);
  });

  it("backward through matmul -> add -> relu -> mean", async () => {
    const x = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const w = api.tensorFromArray([0.5, -1, 2, 1.5, -0.5, 0.25], [3, 2], { requiresGrad: true });
    const bias = api.tensorFromArray([0.1, -0.2], [2], { requiresGrad: true });

    const h = x.matmul(w);        // [2, 2]
    const hBias = h.add(bias);    // [2, 2] with broadcast
    const activated = hBias.relu();
    const loss = activated.mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") throw new Error("Expected tensor");
    const scalar = loss.reshape([]);

    await scalar.backward();

    expect(x.grad).not.toBeNull();
    expect(w.grad).not.toBeNull();
    expect(bias.grad).not.toBeNull();
    expect(x.grad?.shape).toEqual([2, 3]);
    expect(w.grad?.shape).toEqual([3, 2]);
    expect(bias.grad?.shape).toEqual([2]);
  });

  it("backward with shared inputs (diamond pattern)", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

    // Diamond: a -> b, a -> c, b+c -> d
    const b = a.mul(api.tensorFromArray([2, 2, 2, 2], [4]));
    const c = a.add(api.tensorFromArray([1, 1, 1, 1], [4]));
    const d = b.add(c);
    const loss = d.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    // Gradient should accumulate from both paths
    // d = 2*a + (a + 1), so d/da = 3
    expect(await a.grad?.cpu()).toEqual([3, 3, 3, 3]);
  });
});

describe("AOT Autograd: Gradient Accumulation", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("leaf tensors accumulate gradients correctly", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
    const b = a.mul(api.tensorFromArray([2, 2, 2, 2], [4]));
    const c = b.add(api.tensorFromArray([1, 1, 1, 1], [4]));
    const loss = c.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    // Chain: c = 2*a + 1, loss = sum(c), d(loss)/da = 2
    expect(await a.grad?.cpu()).toEqual([2, 2, 2, 2]);
  });

  it("zeroGrad clears gradients", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

    // First backward
    const loss1 = a.sum();
    if (typeof loss1 === "number") throw new Error("Expected tensor");
    await loss1.backward();
    expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1]);

    // Zero grad and run again
    a.zeroGrad();
    expect(a.grad).toBeNull();
  });
});

describe("AOT Autograd: Fusion-Enabled Execution", () => {
  let engine: RuntimeEngine;
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    // Use fusion-enabled engine for these tests
    engine = new RuntimeEngine("cpu", { enableFusion: true });
    api = new Torchlette("cpu");
  });

  it("gradients correct with fusion enabled", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
    const b = api.tensorFromArray([5, 6, 7, 8], [4], { requiresGrad: true });

    // Chain of fusible ops
    const c = a.add(b);
    const d = c.mul(b);
    const loss = d.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();
    // d = (a + b) * b = a*b + b^2
    // d/da = b
    // d/db = a + 2*b
    expect(await a.grad?.cpu()).toEqual([5, 6, 7, 8]);
    const bGrad = await b.grad?.cpu();
    expect(bGrad?.[0]).toBeCloseTo(1 + 2*5, 5);  // 11
    expect(bGrad?.[1]).toBeCloseTo(2 + 2*6, 5);  // 14
    expect(bGrad?.[2]).toBeCloseTo(3 + 2*7, 5);  // 17
    expect(bGrad?.[3]).toBeCloseTo(4 + 2*8, 5);  // 20
  });

  it("gradients correct with non-fusible ops in chain", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

    // Chain: fusible -> non-fusible -> fusible
    const b = a.mul(api.tensorFromArray([2, 2, 2, 2], [4]));  // fusible
    const c = b.sum();  // non-fusible (reduction)
    if (typeof c === "number") throw new Error("Expected tensor");

    await c.backward();

    expect(a.grad).not.toBeNull();
    expect(await a.grad?.cpu()).toEqual([2, 2, 2, 2]);
  });
});

describe("AOT Autograd: Saved-for-Backward", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("saved tensors remain correct during backward", async () => {
    // Create inputs that will be saved during forward
    const a = api.tensorFromArray([4, 9, 16, 25], [4], { requiresGrad: true });

    // sqrt saves input for backward (grad = 0.5/sqrt(x))
    const b = a.sqrt();
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    // Gradients should use saved values correctly
    await loss.backward();

    expect(a.grad).not.toBeNull();
    const grad = await a.grad?.cpu();
    // d(sqrt(x))/dx = 0.5 / sqrt(x)
    expect(grad?.[0]).toBeCloseTo(0.5 / 2, 5);   // 0.25
    expect(grad?.[1]).toBeCloseTo(0.5 / 3, 5);   // ~0.167
    expect(grad?.[2]).toBeCloseTo(0.5 / 4, 5);   // 0.125
    expect(grad?.[3]).toBeCloseTo(0.5 / 5, 5);   // 0.1
  });

  it("relu saves input for backward", async () => {
    const a = api.tensorFromArray([2, -3, 4, -5], [4], { requiresGrad: true });
    const b = a.relu();
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    // relu grad is 1 where input > 0, 0 otherwise
    expect(await a.grad?.cpu()).toEqual([1, 0, 1, 0]);
  });
});

describe("AOT Autograd: Broadcasting", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("backward with broadcast in add", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b = api.tensorFromArray([10, 20], [2], { requiresGrad: true });

    const c = a.add(b);  // b broadcasts to [2, 2]
    const loss = c.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();
    expect(a.grad?.shape).toEqual([2, 2]);
    expect(b.grad?.shape).toEqual([2]);
    expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1]);
    // b gradient should sum over the broadcast dimension
    expect(await b.grad?.cpu()).toEqual([2, 2]);
  });

  it("backward with broadcast in mul", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b = api.tensorFromArray([2], [1], { requiresGrad: true });

    const c = a.mul(b);  // b broadcasts to [2, 2]
    const loss = c.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();
    // grad_a = b = [2, 2, 2, 2]
    expect(await a.grad?.cpu()).toEqual([2, 2, 2, 2]);
    // grad_b = sum(a) = 10
    expect(await b.grad?.cpu()).toEqual([10]);
  });
});

describe("AOT Autograd: View Operations", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("backward through contiguous view chain", async () => {
    // Create a chain that produces a contiguous tensor for backward
    const a = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    // mul produces contiguous result
    const b = a.mul(api.tensorFromArray([2, 2, 2, 2], [2, 2]));
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(a.grad?.shape).toEqual([2, 2]);
    expect(await a.grad?.cpu()).toEqual([2, 2, 2, 2]);
  });

  it("backward through transpose", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const b = a.transpose({ dim0: 0, dim1: 1 });
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(a.grad?.shape).toEqual([2, 3]);
    expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("backward through expand", async () => {
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

describe("AOT Autograd: Edge Cases", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("backward with 0-d tensor (scalar) output", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
    const loss = a.sum();  // Returns 0-d tensor
    if (typeof loss === "number") throw new Error("Expected tensor");

    expect(loss.shape).toEqual([]);
    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1]);
  });

  it("non-leaf tensor does not retain grad by default", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
    const b = a.mul(api.tensorFromArray([2, 2, 2, 2], [4]));
    // b.grad should be null unless retainGrad is called
    const loss = b.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    // Only leaf tensors have grad by default
    expect(a.grad).not.toBeNull();
    // Non-leaf b should not have grad unless retainGrad was called
    expect(b.grad).toBeNull();
  });

  it("tensor without requiresGrad does not get gradient", async () => {
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
    const b = api.tensorFromArray([5, 6, 7, 8], [4]); // No requiresGrad
    const c = a.add(b);
    const loss = c.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(b.grad).toBeNull();
  });
});

describe("AOT Autograd: Multiple Backward Passes", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("separate backward passes on different graphs", async () => {
    const w = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

    // First forward+backward
    const loss1 = w.sum();
    if (typeof loss1 === "number") throw new Error("Expected tensor");
    await loss1.backward();
    expect(await w.grad?.cpu()).toEqual([1, 1, 1, 1]);

    // Zero grad before second backward
    w.zeroGrad();

    // Second forward+backward with different computation
    const w2 = w.mul(api.tensorFromArray([2, 2, 2, 2], [4]));
    const loss2 = w2.sum();
    if (typeof loss2 === "number") throw new Error("Expected tensor");
    await loss2.backward();

    // Gradient should be just 2 (from d(2*w)/dw = 2)
    expect(await w.grad?.cpu()).toEqual([2, 2, 2, 2]);
  });

  it("zeroGrad resets gradients between backward passes", async () => {
    const w = api.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

    // First backward
    const loss1 = w.sum();
    if (typeof loss1 === "number") throw new Error("Expected tensor");
    await loss1.backward();
    expect(await w.grad?.cpu()).toEqual([1, 1, 1, 1]);

    // Zero grad
    w.zeroGrad();
    expect(w.grad).toBeNull();

    // Second backward with different computation
    const w2 = w.mul(api.tensorFromArray([3, 3, 3, 3], [4]));
    const loss2 = w2.sum();
    if (typeof loss2 === "number") throw new Error("Expected tensor");
    await loss2.backward();

    // Gradient should be just 3 (not accumulated)
    expect(await w.grad?.cpu()).toEqual([3, 3, 3, 3]);
  });

  it("training loop simulation: multiple iterations", async () => {
    // Simulate a simple training loop
    const weights = api.tensorFromArray([0.5, 0.5, 0.5, 0.5], [4], { requiresGrad: true });
    const target = api.tensorFromArray([1, 2, 3, 4], [4]);

    for (let i = 0; i < 3; i++) {
      // Forward: prediction = weights * input
      const input = api.tensorFromArray([1, 1, 1, 1], [4]);
      const pred = weights.mul(input);

      // Loss: MSE-like (pred - target)^2 simplified to sum of pred
      const loss = pred.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      // Backward
      weights.zeroGrad();
      await loss.backward();

      // Verify gradient exists
      expect(weights.grad).not.toBeNull();
      expect(weights.grad?.shape).toEqual([4]);
    }
  });
});

describe("AOT Autograd: Optimizer Integration", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("SGD optimizer step with autograd", async () => {
    // Import SGD
    const { SGD } = await import("../src/optim");

    const w = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const optimizer = new SGD([w], { lr: 0.1 }, api);

    // Forward
    const loss = w.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    // Backward
    await loss.backward();

    // Gradient should be 1s
    expect(await w.grad?.cpu()).toEqual([1, 1, 1, 1]);

    // Optimizer step: w = w - lr * grad = w - 0.1 * 1
    // SGD returns new tensors, doesn't modify in-place
    const [updated] = optimizer.step();

    // Updated weight should reflect the change
    const newWeights = await updated.cpu();
    expect(newWeights[0]).toBeCloseTo(0.9, 5); // 1 - 0.1
    expect(newWeights[1]).toBeCloseTo(1.9, 5); // 2 - 0.1
    expect(newWeights[2]).toBeCloseTo(2.9, 5); // 3 - 0.1
    expect(newWeights[3]).toBeCloseTo(3.9, 5); // 4 - 0.1
  });

  it("Adam optimizer step with autograd", async () => {
    const { Adam } = await import("../src/optim");

    const w = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const optimizer = new Adam([w], { lr: 0.1 }, api);

    // Forward
    const loss = w.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    // Backward
    await loss.backward();

    // Optimizer step
    const [updated] = optimizer.step();

    // Updated weight should be different from original
    const newWeights = await updated.cpu();
    // Adam with grad=1 and default params should update weights
    expect(newWeights[0]).toBeCloseTo(0.9, 5); // 1 - 0.1
    expect(newWeights[1]).toBeCloseTo(1.9, 5); // 2 - 0.1
  });

  it("multiple optimizer steps in training loop", async () => {
    const { SGD } = await import("../src/optim");

    const w = api.tensorFromArray([10, 10, 10, 10], [4], { requiresGrad: true });
    const optimizer = new SGD([w], { lr: 1.0 }, api); // Large lr for visible changes

    // Multiple training steps - use optimizer.getParams() to get current params
    for (let i = 0; i < 3; i++) {
      optimizer.zeroGrad();

      // Use current params from optimizer
      const currentW = optimizer.getParams()[0];
      const loss = currentW.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");
      await loss.backward();
      optimizer.step();
    }

    // After 3 steps with lr=1.0 and grad=1, w should be 10 - 3 = 7
    const finalW = optimizer.getParams()[0];
    const finalWeights = await finalW.cpu();
    expect(finalWeights[0]).toBeCloseTo(7, 5);
    expect(finalWeights[1]).toBeCloseTo(7, 5);
    expect(finalWeights[2]).toBeCloseTo(7, 5);
    expect(finalWeights[3]).toBeCloseTo(7, 5);
  });
});

describe("AOT Autograd: Complex Training Scenarios", () => {
  let api: Torchlette;

  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
    resetBaseIdCounter();
    api = new Torchlette("cpu");
  });

  it("MLP-like forward backward", async () => {
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

  it("shared weights (weight tying) backward", async () => {
    // Weight tying: same weights used twice
    const w = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const x = api.tensorFromArray([1, 1, 1, 1], [2, 2]);

    // Use w twice in computation
    const h1 = x.matmul(w);
    const h2 = h1.matmul(w); // Same w used again
    const loss = h2.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    // Gradient should accumulate from both uses of w
    expect(w.grad).not.toBeNull();
    expect(w.grad?.shape).toEqual([2, 2]);
  });

  it("residual connection backward", async () => {
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
