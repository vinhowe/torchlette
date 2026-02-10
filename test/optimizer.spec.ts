import { describe, expect, it } from "vitest";

import { Adam, SGD, Torchlette } from "../src";

describe("optimizer ring-2: SGD", () => {
  it("updates parameters with a single SGD step", async () => {
    const api = new Torchlette("cpu");
    const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
    const b = api.tensorFromArray([3, 4], [2], { requiresGrad: true });

    const loss = w.mul(b).mean({ dim: 0 });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.backward();

    const opt = new SGD([w], { lr: 0.1 }, api);
    const [updated] = opt.step();

    const expected = [0.85, 1.8];
    const actual = await updated.cpu();
    expect(actual[0]).toBeCloseTo(expected[0], 6);
    expect(actual[1]).toBeCloseTo(expected[1], 6);
  });

  it("zeroGrad clears accumulated gradients", async () => {
    const api = new Torchlette("cpu");
    const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
    const b = api.tensorFromArray([3, 4], [2], { requiresGrad: true });

    const loss = w.mul(b).mean({ dim: 0 });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.backward();
    expect(w.grad).not.toBeNull();

    const opt = new SGD([w], { lr: 0.1 }, api);
    opt.zeroGrad();

    expect(w.grad).toBeNull();
  });

  it("applies weight decay to the update", async () => {
    const api = new Torchlette("cpu");
    const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });

    const loss = w.mean({ dim: 0 });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.backward();

    const opt = new SGD([w], { lr: 0.1, weightDecay: 0.1 }, api);
    const [updated] = opt.step();

    const actual = await updated.cpu();
    expect(actual[0]).toBeCloseTo(0.94, 6);
    expect(actual[1]).toBeCloseTo(1.93, 6);
  });

  it("uses momentum across steps", async () => {
    const api = new Torchlette("cpu");
    const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });
    const opt = new SGD([w], { lr: 0.1, momentum: 0.9 }, api);

    const loss1 = opt.getParams()[0].mean({ dim: 0 });
    if (typeof loss1 === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss1.backward();
    opt.step();
    opt.zeroGrad();

    const loss2 = opt.getParams()[0].mean({ dim: 0 });
    if (typeof loss2 === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss2.backward();
    const [updated] = opt.step();

    const actual = await updated.cpu();
    expect(actual[0]).toBeCloseTo(0.855, 6);
    expect(actual[1]).toBeCloseTo(1.855, 6);
  });
});

describe("optimizer ring-2: Adam", () => {
  it("updates parameters with a single Adam step", async () => {
    const api = new Torchlette("cpu");
    const w = api.tensorFromArray([1, 2], [2], { requiresGrad: true });

    const loss = w.mean({ dim: 0 });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.backward();

    const opt = new Adam([w], { lr: 0.1 }, api);
    const [updated] = opt.step();

    const actual = await updated.cpu();
    expect(actual[0]).toBeCloseTo(0.9, 6);
    expect(actual[1]).toBeCloseTo(1.9, 6);
  });
});
