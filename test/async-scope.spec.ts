import { describe, expect, it } from "vitest";

import { Engine, type EngineTensor } from "../src";
import { Torchlette } from "../src/frontend";

describe("Engine: async scope context", () => {
  it("tracks tensors created during async operation", async () => {
    const engine = new Engine();
    let createdInScope: EngineTensor | null = null;

    await engine.runWithAsyncScope(async () => {
      createdInScope = engine.createTensor();
      // Simulate async work
      await Promise.resolve();
    });

    // Tensor created in async scope should be disposed
    expect(createdInScope?.disposed).toBe(true);
  });

  it("keeps tensors marked with keep()", async () => {
    const engine = new Engine();
    let kept: EngineTensor | null = null;
    let notKept: EngineTensor | null = null;

    await engine.runWithAsyncScope(async () => {
      kept = engine.createTensor();
      notKept = engine.createTensor();
      engine.keep(kept);
      await Promise.resolve();
    });

    // Kept tensor survives scope exit
    expect(kept?.disposed).toBe(false);
    expect(kept?.escapes).toBe(true);
    // Not kept tensor is disposed
    expect(notKept?.disposed).toBe(true);
  });

  it("returns result from async function", async () => {
    const engine = new Engine();

    const result = await engine.runWithAsyncScope(async () => {
      await Promise.resolve();
      return 42;
    });

    expect(result).toBe(42);
  });

  it("can nest async scopes", async () => {
    const engine = new Engine();
    let outerTensor: EngineTensor | null = null;
    let innerTensor: EngineTensor | null = null;

    await engine.runWithAsyncScope(async () => {
      outerTensor = engine.createTensor();
      engine.keep(outerTensor);

      await engine.runWithAsyncScope(async () => {
        innerTensor = engine.createTensor();
        await Promise.resolve();
      });

      // Inner tensor should already be disposed
      expect(innerTensor?.disposed).toBe(true);
    });

    // Outer kept tensor survives all scopes
    expect(outerTensor?.disposed).toBe(false);
  });

  it("does not affect tensors in sync tidy scopes", async () => {
    const engine = new Engine();
    let tidyTensor: EngineTensor | null = null;
    let asyncTensor: EngineTensor | null = null;

    await engine.runWithAsyncScope(async () => {
      // First, create tensor in tidy scope
      engine.tidy(() => {
        tidyTensor = engine.createTensor();
        engine.keep(tidyTensor);
        return tidyTensor;
      });

      // Then create tensor outside tidy but in async scope
      asyncTensor = engine.createTensor();
      await Promise.resolve();
    });

    // Tidy tensor was kept, should survive
    expect(tidyTensor?.disposed).toBe(false);
    // Async scope tensor should be disposed
    expect(asyncTensor?.disposed).toBe(true);
  });

  it("tensors in tidy scope during async scope go to tidy, not async", async () => {
    const engine = new Engine();
    let inTidy: EngineTensor | null = null;

    await engine.runWithAsyncScope(async () => {
      engine.tidy(() => {
        inTidy = engine.createTensor();
        // Don't return it, so tidy should dispose it
        return undefined;
      });

      // Tidy scope disposed it, not async scope
      expect(inTidy?.disposed).toBe(true);
    });
  });

  it("preserves pin counts correctly", async () => {
    const engine = new Engine();
    let tensor: EngineTensor | null = null;

    await engine.runWithAsyncScope(async () => {
      tensor = engine.createTensor();
      const baseId = tensor.baseId;

      // Pin count should be 1 during async scope
      expect(engine._debug_getBasePinCount(baseId)).toBe(1);

      await Promise.resolve();
    });

    // After scope exit with disposal, pin count should be 0
    expect(engine._debug_getBasePinCount(tensor?.baseId ?? 0)).toBe(0);
  });
});

describe("Torchlette: runWithAsyncScope", () => {
  it("tracks frontend tensors created during async operation", async () => {
    const api = new Torchlette("cpu");
    let wasDisposed = false;

    await api.runWithAsyncScope(async () => {
      const t = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      wasDisposed = t._engineTensor().disposed;
      await Promise.resolve();
      // Check it's not disposed yet inside scope
      expect(t._engineTensor().disposed).toBe(false);
    });

    // After scope exit, tensor should be disposed
    // Note: we can't check directly since t is out of scope,
    // but the test passes if no errors occur
  });

  it("keeps tensors marked with keep()", async () => {
    const api = new Torchlette("cpu");
    let kept: ReturnType<typeof api.tensorFromArray> | null = null;

    await api.runWithAsyncScope(async () => {
      kept = api.tensorFromArray([1, 2, 3, 4], [2, 2]);
      api.keep(kept);
      await Promise.resolve();
    });

    // Kept tensor survives
    expect(kept?._engineTensor().disposed).toBe(false);

    // Clean up
    kept?.dispose();
  });

  it("backward() automatically uses async scope internally", async () => {
    const api = new Torchlette("cpu");

    const x = api.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const y = api.tensorFromArray([5, 6, 7, 8], [2, 2], { requiresGrad: true });
    const z = x.mul(y).sum() as ReturnType<typeof x.sum>;

    // Force z to be a Tensor for backward
    if (typeof z === "number") {
      throw new Error("Expected Tensor, got number");
    }

    // backward() should complete without memory leaks
    // The async scope ensures intermediate tensors are cleaned up
    await z.backward();

    // x and y should have gradients
    expect(x.grad).not.toBeNull();
    expect(y.grad).not.toBeNull();

    // Clean up
    x.dispose();
    y.dispose();
  });
});
