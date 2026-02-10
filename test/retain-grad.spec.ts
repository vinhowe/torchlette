/**
 * retainGrad Tests
 *
 * Tests for the retainGrad() method that allows retaining gradients
 * on non-leaf tensors during backward.
 */

import { describe, expect, it, beforeEach } from "vitest";
import { Torchlette } from "../src/frontend";

describe("retainGrad", () => {
  let torch: Torchlette;

  beforeEach(() => {
    torch = new Torchlette("cpu");
  });

  describe("Basic Functionality", () => {
    it("allows calling retainGrad on tensor with requiresGrad", () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [2, 2], {
        requiresGrad: true,
      });
      const y = x.mul(torch.tensorFromArray([2, 2, 2, 2], [2, 2]));

      // Should not throw
      y.retainGrad();

      expect(y.isRetainGrad).toBe(true);
    });

    it("throws when calling retainGrad on tensor without requiresGrad", () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [2, 2], {
        requiresGrad: false,
      });

      expect(() => x.retainGrad()).toThrow(
        "retainGrad() can only be called on tensors that require gradients",
      );
    });

    it("isRetainGrad is false by default", () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [2, 2], {
        requiresGrad: true,
      });

      expect(x.isRetainGrad).toBe(false);
    });
  });

  describe("Gradient Retention", () => {
    it("retains gradient on non-leaf tensor when retainGrad is called", async () => {
      // x is a leaf tensor
      const x = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

      // y = x * 2 is a non-leaf tensor
      const y = x.mul(torch.tensorFromArray([2, 2, 2, 2], [4]));
      y.retainGrad();

      // z = sum(y)
      const z = y.sum();
      if (typeof z === "number") throw new Error("Expected tensor");

      await z.backward();

      // Both x and y should have gradients
      expect(x.grad).not.toBeNull();
      expect(y.grad).not.toBeNull();

      // x.grad = d(sum(y))/dx = d(sum(2x))/dx = [2, 2, 2, 2]
      expect(await x.grad!.cpu()).toEqual([2, 2, 2, 2]);

      // y.grad = d(sum(y))/dy = [1, 1, 1, 1]
      expect(await y.grad!.cpu()).toEqual([1, 1, 1, 1]);
    });

    it("does not retain gradient on non-leaf without retainGrad", async () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      const y = x.mul(torch.tensorFromArray([2, 2, 2, 2], [4]));

      // Don't call retainGrad on y
      const z = y.sum();
      if (typeof z === "number") throw new Error("Expected tensor");

      await z.backward();

      // x should have gradient (leaf tensor)
      expect(x.grad).not.toBeNull();

      // y should NOT have gradient (non-leaf without retainGrad)
      expect(y.grad).toBeNull();
    });

    it("works with multiple intermediate tensors", async () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

      // Chain: x -> a -> b -> c -> sum
      const a = x.mul(torch.tensorFromArray([2, 2, 2, 2], [4]));
      const b = a.add(torch.tensorFromArray([1, 1, 1, 1], [4]));
      const c = b.mul(torch.tensorFromArray([3, 3, 3, 3], [4]));

      // Only retain grad on 'b'
      b.retainGrad();

      const loss = c.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // x (leaf) should have gradient
      expect(x.grad).not.toBeNull();

      // a (non-leaf, no retainGrad) should NOT have gradient
      expect(a.grad).toBeNull();

      // b (non-leaf, has retainGrad) SHOULD have gradient
      expect(b.grad).not.toBeNull();

      // c (non-leaf, no retainGrad) should NOT have gradient
      expect(c.grad).toBeNull();

      // b.grad = d(sum(c))/db = d(sum(3*b))/db = [3, 3, 3, 3]
      expect(await b.grad!.cpu()).toEqual([3, 3, 3, 3]);
    });
  });

  describe("Edge Cases", () => {
    it("calling retainGrad multiple times is idempotent", () => {
      const x = torch.tensorFromArray([1, 2], [2], { requiresGrad: true });
      const y = x.mul(torch.tensorFromArray([2, 2], [2]));

      y.retainGrad();
      y.retainGrad();
      y.retainGrad();

      expect(y.isRetainGrad).toBe(true);
    });

    it("retainGrad on leaf tensor still works", async () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      x.retainGrad(); // retainGrad on leaf is a no-op but should work

      const y = x.sum();
      if (typeof y === "number") throw new Error("Expected tensor");

      await y.backward();

      expect(x.grad).not.toBeNull();
      expect(await x.grad!.cpu()).toEqual([1, 1, 1, 1]);
    });

    it("works with matmul backward", async () => {
      // x: [2, 3], y: [3, 2]
      const x = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
        requiresGrad: true,
      });
      const y = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [3, 2], {
        requiresGrad: true,
      });

      const z = x.matmul(y); // [2, 2]
      z.retainGrad();

      const loss = z.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // All should have gradients
      expect(x.grad).not.toBeNull();
      expect(y.grad).not.toBeNull();
      expect(z.grad).not.toBeNull();

      // z.grad should be all ones (from sum backward)
      expect(await z.grad!.cpu()).toEqual([1, 1, 1, 1]);
    });
  });

  describe("Memory Behavior", () => {
    it("non-retained gradients are disposed (no memory leak)", async () => {
      const x = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

      // Create many intermediate tensors without retainGrad
      let current = x;
      for (let i = 0; i < 10; i++) {
        current = current.mul(torch.tensorFromArray([1.1, 1.1, 1.1, 1.1], [4]));
      }

      const loss = current.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      // This should complete without error
      await loss.backward();

      // Only x should have gradient
      expect(x.grad).not.toBeNull();
    });
  });

  describe("Integration with Operations", () => {
    it("works with relu backward", async () => {
      const x = torch.tensorFromArray([-1, 2, -3, 4], [4], {
        requiresGrad: true,
      });

      const y = x.relu();
      y.retainGrad();

      const loss = y.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      // y.grad = [1, 1, 1, 1] (from sum)
      expect(await y.grad!.cpu()).toEqual([1, 1, 1, 1]);

      // x.grad = [0, 1, 0, 1] (relu derivative: 0 where x <= 0, 1 where x > 0)
      expect(await x.grad!.cpu()).toEqual([0, 1, 0, 1]);
    });

    it("works with add backward", async () => {
      const a = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });
      const b = torch.tensorFromArray([5, 6, 7, 8], [4], { requiresGrad: true });

      const c = a.add(b);
      c.retainGrad();

      const loss = c.sum();
      if (typeof loss === "number") throw new Error("Expected tensor");

      await loss.backward();

      expect(await a.grad!.cpu()).toEqual([1, 1, 1, 1]);
      expect(await b.grad!.cpu()).toEqual([1, 1, 1, 1]);
      expect(await c.grad!.cpu()).toEqual([1, 1, 1, 1]);
    });
  });
});
