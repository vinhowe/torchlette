/**
 * Checkpoint Integration Tests
 *
 * Tests gradient correctness and memory behavior with checkpointing.
 */

import { describe, expect, it, beforeAll } from "vitest";
import { Torchlette, Tensor } from "../../src/frontend";
import { checkpoint } from "../../src/nn/checkpoint";

describe("checkpoint gradient correctness", () => {
  let api: Torchlette;

  beforeAll(() => {
    api = new Torchlette("cpu");
  });

  it("produces identical gradients for simple mul chain", async () => {
    // Without checkpoint
    const a1 = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    const b1 = api.tensorFromArray([4, 5, 6], [3], { requiresGrad: true });
    const y1 = api.mul(api.mul(a1, b1), a1);
    const loss1 = api.sum(y1, { dim: [0], keepdim: false }) as Tensor;
    await loss1.backward();
    const gradA1 = await a1.grad!.cpu();
    const gradB1 = await b1.grad!.cpu();

    // With checkpoint
    const a2 = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    const b2 = api.tensorFromArray([4, 5, 6], [3], { requiresGrad: true });
    const y2 = checkpoint(api, (x, y) => api.mul(api.mul(x, y), x), [a2, b2]);
    const loss2 = api.sum(y2, { dim: [0], keepdim: false }) as Tensor;
    await loss2.backward();
    const gradA2 = await a2.grad!.cpu();
    const gradB2 = await b2.grad!.cpu();

    // Gradients should be identical
    expect(gradA2).toEqual(gradA1);
    expect(gradB2).toEqual(gradB1);
  });

  it("produces identical gradients for matmul", async () => {
    // Without checkpoint
    const a1 = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b1 = api.tensorFromArray([5, 6, 7, 8], [2, 2], { requiresGrad: true });
    const y1 = api.matmul(a1, b1);
    const loss1 = api.sum(y1, { dim: [0, 1], keepdim: false }) as Tensor;
    await loss1.backward();
    const gradA1 = await a1.grad!.cpu();
    const gradB1 = await b1.grad!.cpu();

    // With checkpoint
    const a2 = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b2 = api.tensorFromArray([5, 6, 7, 8], [2, 2], { requiresGrad: true });
    const y2 = checkpoint(api, (x, y) => api.matmul(x, y), [a2, b2]);
    const loss2 = api.sum(y2, { dim: [0, 1], keepdim: false }) as Tensor;
    await loss2.backward();
    const gradA2 = await a2.grad!.cpu();
    const gradB2 = await b2.grad!.cpu();

    // Gradients should be identical
    expect(gradA2).toEqual(gradA1);
    expect(gradB2).toEqual(gradB1);
  });

  it("produces identical gradients for relu activation", async () => {
    // Without checkpoint
    const a1 = api.tensorFromArray([-1, 0, 1, 2], [4], { requiresGrad: true });
    const y1 = api.relu(api.mul(a1, a1));
    const loss1 = api.sum(y1, { dim: [0], keepdim: false }) as Tensor;
    await loss1.backward();
    const gradA1 = await a1.grad!.cpu();

    // With checkpoint
    const a2 = api.tensorFromArray([-1, 0, 1, 2], [4], { requiresGrad: true });
    const y2 = checkpoint(api, (x) => api.relu(api.mul(x, x)), [a2]);
    const loss2 = api.sum(y2, { dim: [0], keepdim: false }) as Tensor;
    await loss2.backward();
    const gradA2 = await a2.grad!.cpu();

    expect(gradA2).toEqual(gradA1);
  });

  it("produces identical gradients for softmax", async () => {
    // Without checkpoint
    const a1 = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const y1 = api.softmax(a1, -1);
    const loss1 = api.sum(y1, { dim: [0, 1], keepdim: false }) as Tensor;
    await loss1.backward();
    const gradA1 = await a1.grad!.cpu();

    // With checkpoint
    const a2 = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { requiresGrad: true });
    const y2 = checkpoint(api, (x) => api.softmax(x, -1), [a2]);
    const loss2 = api.sum(y2, { dim: [0, 1], keepdim: false }) as Tensor;
    await loss2.backward();
    const gradA2 = await a2.grad!.cpu();

    expect(gradA2).toEqual(gradA1);
  });

  it("handles nested operations correctly", async () => {
    // Complex chain: gelu(matmul(a, b) + c)
    const a1 = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b1 = api.tensorFromArray([5, 6, 7, 8], [2, 2], { requiresGrad: true });
    const c1 = api.tensorFromArray([0.1, 0.2, 0.3, 0.4], [2, 2], { requiresGrad: true });
    const mm1 = api.matmul(a1, b1);
    const added1 = api.add(mm1, c1);
    const y1 = api.gelu(added1);
    const loss1 = api.sum(y1, { dim: [0, 1], keepdim: false }) as Tensor;
    await loss1.backward();
    const gradA1 = await a1.grad!.cpu();
    const gradB1 = await b1.grad!.cpu();
    const gradC1 = await c1.grad!.cpu();

    // With checkpoint
    const a2 = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const b2 = api.tensorFromArray([5, 6, 7, 8], [2, 2], { requiresGrad: true });
    const c2 = api.tensorFromArray([0.1, 0.2, 0.3, 0.4], [2, 2], { requiresGrad: true });
    const y2 = checkpoint(api, (x, y, z) => {
      const mm = api.matmul(x, y);
      const added = api.add(mm, z);
      return api.gelu(added);
    }, [a2, b2, c2]);
    const loss2 = api.sum(y2, { dim: [0, 1], keepdim: false }) as Tensor;
    await loss2.backward();
    const gradA2 = await a2.grad!.cpu();
    const gradB2 = await b2.grad!.cpu();
    const gradC2 = await c2.grad!.cpu();

    // Check gradients are close (may have small floating point differences)
    for (let i = 0; i < gradA1.length; i++) {
      expect(gradA2[i]).toBeCloseTo(gradA1[i], 5);
    }
    for (let i = 0; i < gradB1.length; i++) {
      expect(gradB2[i]).toBeCloseTo(gradB1[i], 5);
    }
    for (let i = 0; i < gradC1.length; i++) {
      expect(gradC2[i]).toBeCloseTo(gradC1[i], 5);
    }
  });
});

describe("checkpoint pack/unpack hooks", () => {
  let api: Torchlette;

  beforeAll(() => {
    api = new Torchlette("cpu");
  });

  it("pack hook is called for each saved tensor", () => {
    const packCalls: number[] = [];

    const result = api.saved_tensors_hooks(
      (tensor) => {
        packCalls.push(tensor.baseId);
        return tensor; // passthrough
      },
      (packed) => packed as Tensor,
      () => {
        const a = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
        const b = api.tensorFromArray([4, 5, 6], [3], { requiresGrad: true });
        return api.mul(a, b);
      },
    );

    // mul saves both inputs
    expect(packCalls.length).toBe(2);
  });

  it("unpack hook is called during backward", async () => {
    let unpackCalls = 0;

    const a = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    const result = api.saved_tensors_hooks(
      (tensor) => tensor,
      (packed) => {
        unpackCalls++;
        return packed as Tensor;
      },
      () => api.mul(a, a),
    );

    const loss = api.sum(result, { dim: [0], keepdim: false }) as Tensor;
    await loss.backward();

    // unpack should be called for each saved tensor during backward
    expect(unpackCalls).toBeGreaterThan(0);
  });
});
