import { describe, expect, it } from "vitest";

import { Torchlette } from "../src/frontend";
import { DEFAULT_AMP_POLICY, DISABLED_AMP_POLICY } from "../src/engine/amp";

describe("Frontend Autocast", () => {
  it("isAutocastEnabled is false by default", () => {
    const torch = new Torchlette("cpu");
    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("autocast enables within block", () => {
    const torch = new Torchlette("cpu");

    expect(torch.isAutocastEnabled).toBe(false);

    torch.autocast(() => {
      expect(torch.isAutocastEnabled).toBe(true);
    });

    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("autocast returns function result", () => {
    const torch = new Torchlette("cpu");

    const result = torch.autocast(() => {
      return 42;
    });

    expect(result).toBe(42);
  });

  it("autocast can be disabled explicitly", () => {
    const torch = new Torchlette("cpu");

    torch.autocast(() => {
      expect(torch.isAutocastEnabled).toBe(true);

      torch.autocast(
        () => {
          expect(torch.isAutocastEnabled).toBe(false);
        },
        { enabled: false },
      );

      expect(torch.isAutocastEnabled).toBe(true);
    });
  });

  it("nested autocast blocks work correctly", () => {
    const torch = new Torchlette("cpu");

    expect(torch.isAutocastEnabled).toBe(false);

    torch.autocast(() => {
      expect(torch.isAutocastEnabled).toBe(true);
      const config1 = torch.currentAutocastConfig;
      expect(config1.policy).toEqual(DEFAULT_AMP_POLICY);

      // Nested block with disabled
      torch.autocast(
        () => {
          expect(torch.isAutocastEnabled).toBe(false);
        },
        { enabled: false },
      );

      // Back to enabled
      expect(torch.isAutocastEnabled).toBe(true);
    });

    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("autocast restores state on exception", () => {
    const torch = new Torchlette("cpu");

    expect(torch.isAutocastEnabled).toBe(false);

    expect(() => {
      torch.autocast(() => {
        expect(torch.isAutocastEnabled).toBe(true);
        throw new Error("test error");
      });
    }).toThrow("test error");

    // State should be restored even after exception
    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("autocastAsync enables within async block", async () => {
    const torch = new Torchlette("cpu");

    expect(torch.isAutocastEnabled).toBe(false);

    await torch.autocastAsync(async () => {
      expect(torch.isAutocastEnabled).toBe(true);
    });

    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("autocastAsync returns async function result", async () => {
    const torch = new Torchlette("cpu");

    const result = await torch.autocastAsync(async () => {
      await Promise.resolve();
      return 42;
    });

    expect(result).toBe(42);
  });

  it("autocastAsync restores state on rejection", async () => {
    const torch = new Torchlette("cpu");

    expect(torch.isAutocastEnabled).toBe(false);

    await expect(
      torch.autocastAsync(async () => {
        expect(torch.isAutocastEnabled).toBe(true);
        throw new Error("async error");
      }),
    ).rejects.toThrow("async error");

    expect(torch.isAutocastEnabled).toBe(false);
  });

  it("currentAutocastConfig returns current config", () => {
    const torch = new Torchlette("cpu");

    const disabled = torch.currentAutocastConfig;
    expect(disabled.enabled).toBe(false);

    torch.autocast(() => {
      const enabled = torch.currentAutocastConfig;
      expect(enabled.enabled).toBe(true);
      expect(enabled.policy).toEqual(DEFAULT_AMP_POLICY);
    });
  });

  it("autocast uses custom policy", () => {
    const torch = new Torchlette("cpu");
    const customPolicy = {
      ...DEFAULT_AMP_POLICY,
      memoryDtype: "f16" as const,
    };

    torch.autocast(
      () => {
        const config = torch.currentAutocastConfig;
        expect(config.policy.memoryDtype).toBe("f16");
      },
      { policy: customPolicy },
    );
  });

  it("autocast infers deviceType from backend", () => {
    const torch = new Torchlette("cpu");

    torch.autocast(() => {
      const config = torch.currentAutocastConfig;
      expect(config.deviceType).toBe("cpu");
    });
  });

  it("autocast uses explicit deviceType", () => {
    const torch = new Torchlette("cpu");

    torch.autocast(
      () => {
        const config = torch.currentAutocastConfig;
        expect(config.deviceType).toBe("webgpu");
      },
      { deviceType: "webgpu" },
    );
  });

  it("_getAutocastContext returns context for compiled regions", () => {
    const torch = new Torchlette("cpu");

    const ctx = torch._getAutocastContext();
    expect(ctx.current.enabled).toBe(false);

    torch.autocast(() => {
      const innerCtx = torch._getAutocastContext();
      expect(innerCtx.current.enabled).toBe(true);
      // Should be the same context object
      expect(innerCtx).toBe(ctx);
    });
  });
});

describe("Frontend Autocast with Tensor Operations", () => {
  it("tensor ops work within autocast block", async () => {
    const torch = new Torchlette("cpu");

    const result = await torch.autocastAsync(async () => {
      const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = torch.tensorFromArray([5, 6, 7, 8], [2, 2]);
      const c = torch.add(a, b);
      return await c.cpu();
    });

    expect(result).toEqual([6, 8, 10, 12]);
  });

  it("matmul works within autocast block", async () => {
    const torch = new Torchlette("cpu");

    const result = await torch.autocastAsync(async () => {
      const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = torch.tensorFromArray([5, 6, 7, 8], [2, 2]);
      // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
      // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
      const c = torch.matmul(a, b);
      return await c.cpu();
    });

    expect(result).toEqual([19, 22, 43, 50]);
  });
});

describe("Frontend Autocast dtype promotion", () => {
  it("add promotes f16 matmul output + f32 tensor (residual pattern)", async () => {
    const torch = new Torchlette("cpu");

    const result = await torch.autocastAsync(async () => {
      const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = torch.tensorFromArray([1, 0, 0, 1], [2, 2]);
      // matmul under autocast → f16 output
      const matmulOut = torch.matmul(a, b);
      // f32 tensor
      const residual = torch.tensorFromArray([10, 20, 30, 40], [2, 2]);
      // add f16 + f32 should promote to f32
      const sum = torch.add(matmulOut, residual);
      return await sum.cpu();
    });

    // matmul: [1 2][1 0] = [1 2], + [10 20] = [11 22]
    //         [3 4][0 1]   [3 4]   [30 40]   [33 44]
    expect(result).toEqual([11, 22, 33, 44]);
  });

  it("mul promotes mixed dtypes during autocast", async () => {
    const torch = new Torchlette("cpu");

    const result = await torch.autocastAsync(async () => {
      const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const b = torch.tensorFromArray([1, 0, 0, 1], [2, 2]);
      const matmulOut = torch.matmul(a, b); // f16
      const scale = torch.tensorFromArray([2, 2, 2, 2], [2, 2]); // f32
      const scaled = torch.mul(matmulOut, scale);
      return await scaled.cpu();
    });

    expect(result).toEqual([2, 4, 6, 8]);
  });

  it("sub promotes mixed dtypes during autocast", async () => {
    const torch = new Torchlette("cpu");

    const result = await torch.autocastAsync(async () => {
      const a = torch.tensorFromArray([1, 0, 0, 1], [2, 2]);
      const b = torch.tensorFromArray([1, 0, 0, 1], [2, 2]);
      const matmulOut = torch.matmul(a, b); // f16
      const bias = torch.tensorFromArray([1, 1, 1, 1], [2, 2]); // f32
      const result = torch.sub(matmulOut, bias);
      return await result.cpu();
    });

    expect(result).toEqual([0, -1, -1, 0]);
  });

  it("backward works through dtype promotion", async () => {
    const torch = new Torchlette("cpu");

    const x = torch.tensorFromArray([1, 2, 3, 4], [2, 2], {
      requiresGrad: true,
    });
    const w = torch.tensorFromArray([1, 0, 0, 1], [2, 2], {
      requiresGrad: true,
    });

    const loss = torch.autocast(() => {
      const matmulOut = torch.matmul(x, w); // f16 under autocast
      const residual = torch.tensorFromArray([0, 0, 0, 0], [2, 2]); // f32
      const sum = torch.add(matmulOut, residual); // promotes f16→f32
      return sum.sum();
    });

    await (loss as any).backward();
    const xGrad = await x.grad?.cpu();
    const wGrad = await w.grad?.cpu();

    // Gradients should be finite and in f32
    expect(xGrad).toBeTruthy();
    expect(wGrad).toBeTruthy();
    for (const v of xGrad!) {
      expect(Number.isFinite(v)).toBe(true);
    }
    for (const v of wGrad!) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});
