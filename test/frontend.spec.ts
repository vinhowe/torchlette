import { describe, expect, it } from "vitest";

import {
  DisposedTensorError,
  FrontendTensor,
  SavedTensorModifiedError,
  Torchlette,
  torch,
} from "../src";

describe("frontend api: Tensor wrapper", () => {
  it("creates tensors and runs elementwise ops", async () => {
    const a = torch.tensorFromArray([1, 2, 3], [3]);
    const b = torch.tensorFromArray([4, 5, 6], [3]);

    const out = a.add(b);

    expect(out).toBeInstanceOf(FrontendTensor);
    expect(await out.cpu()).toEqual([5, 7, 9]);
    expect(out.baseId).not.toBe(a.baseId);
  });

  it("preserves BaseId across view/reshape", () => {
    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);

    const view = a.view([4]);
    const reshaped = a.reshape([2, 2]);

    expect(view.baseId).toBe(a.baseId);
    expect(view.shape).toEqual([4]);
    expect(reshaped.baseId).toBe(a.baseId);
  });

  it("preserves BaseId across transpose", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);

    const transposed = a.transpose({ dim0: 0, dim1: 1 });

    expect(transposed.baseId).toBe(a.baseId);
    expect(transposed.shape).toEqual([2, 2]);
    expect(await transposed.cpu()).toEqual([1, 3, 2, 4]);
  });

  it("supports sub/relu/mean helpers", async () => {
    const a = torch.tensorFromArray([1, -2], [2]);
    const b = torch.tensorFromArray([0.5, 0.5], [2]);

    const diff = a.sub(b);
    const activated = diff.relu();
    const result = activated.mean();

    expect(await diff.cpu()).toEqual([0.5, -2.5]);
    expect(await activated.cpu()).toEqual([0.5, 0]);
    // mean() now returns a Tensor, use item() to get the scalar
    if (typeof result === "number") {
      expect(result).toBe(0.25);
    } else {
      expect(await result.item()).toBe(0.25);
    }
  });

  it("guards saved tensors during backward", async () => {
    const api = new Torchlette("cpu");
    const x = api.tensorFromArray([1, -1], [2], { requiresGrad: true });
    const y = x.relu();
    api._debug_baseCommit(x.baseId, 1);

    const loss = y.mean({ dim: 0, keepdim: true });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    const scalar = loss.reshape([]);

    await expect(scalar.backward()).rejects.toThrow(SavedTensorModifiedError);
  });

  it("exposes async cpu and item helpers", async () => {
    const a = torch.tensorFromArray([1, 2, 3], [3]);
    const scalar = torch.tensorFromArray([42], []);

    await expect(a.cpu()).resolves.toEqual([1, 2, 3]);
    await expect(scalar.item()).resolves.toBe(42);
    await expect(a.item()).rejects.toThrow(
      "item() requires a single-element tensor",
    );
  });

  it("reports device and supports to() transfers", async () => {
    const api = new Torchlette("cpu");
    const a = api.tensorFromArray([1, 2], [2]);

    expect(a.device).toBe("cpu");

    const b = await a.to("cpu");
    expect(b.device).toBe("cpu");
    await expect(b.cpu()).resolves.toEqual([1, 2]);
  });

  it("blocks implicit primitive coercions", () => {
    const a = torch.tensorFromArray([1], []);

    expect(() => String(a)).toThrow(
      "Tensor cannot be implicitly converted to a primitive",
    );
    expect(() => a.valueOf()).toThrow(
      "Tensor cannot be implicitly converted to a primitive",
    );
    expect(a.toString()).toContain("Tensor(shape=[");
  });

  it("tidy disposes unkept tensors", async () => {
    let leaked: FrontendTensor | undefined;
    const kept = torch.tidy(() => {
      const a = torch.tensorFromArray([1, 2], [2]);
      const b = torch.tensorFromArray([3, 4], [2]);
      leaked = b;
      return a;
    });

    expect(await kept.cpu()).toEqual([1, 2]);
    if (!leaked) {
      throw new Error("Missing leaked tensor");
    }
    await expect(leaked.cpu()).rejects.toThrow(DisposedTensorError);
  });

  it("keep prevents tidy disposal", async () => {
    let kept: FrontendTensor | undefined;
    torch.tidy(() => {
      const a = torch.tensorFromArray([5, 6], [2]);
      torch.keep(a);
      kept = a;
    });

    if (!kept) {
      throw new Error("Missing kept tensor");
    }
    expect(await kept.cpu()).toEqual([5, 6]);
  });

  it("dispose is idempotent and blocks use", () => {
    const a = torch.tensorFromArray([7, 8], [2]);
    const b = torch.tensorFromArray([1, 1], [2]);

    a.dispose();
    expect(() => a.add(b)).toThrow(DisposedTensorError);
    expect(() => a.dispose()).not.toThrow();
  });

  it("expand returns a view that shares BaseId", async () => {
    const a = torch.tensorFromArray([1, 2, 3], [1, 3]);
    const expanded = a.expand([2, 3]);

    expect(expanded.baseId).toBe(a.baseId);
    expect(await expanded.cpu()).toEqual([1, 2, 3, 1, 2, 3]);
  });

  it("routes ops through the configured backend", async () => {
    const api = new Torchlette("mock");
    const a = api.tensorFromArray([1, 2], [2]);
    const b = api.tensorFromArray([9, 9], [2]);

    const out = a.add(b);

    expect(await out.cpu()).toEqual([1, 2]);
  });

  it("rejects tensors from different Torchlette instances", () => {
    const apiA = new Torchlette("cpu");
    const apiB = new Torchlette("cpu");
    const a = apiA.tensorFromArray([1, 2], [2]);
    const b = apiB.tensorFromArray([3, 4], [2]);

    expect(() => apiA.add(a, b)).toThrow(
      "Tensor belongs to a different Torchlette instance",
    );
  });
});

describe("frontend api: gather autograd", () => {
  it("gather backward scatters gradient to input positions", async () => {
    // input: [1, 2, 3, 4] shape [2, 2]
    // index: [0, 1] shape [2] along dim 1
    // output: [1, 4] (gathered from row 0 col 0, row 1 col 1)
    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2], { requiresGrad: true });
    const index = torch.tensorFromArray([0, 1], [2, 1]);
    const gathered = torch.gather(a, index, { dim: 1 });
    const loss = gathered.sum();

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(a.grad?.shape).toEqual([2, 2]);
    // Gradient should be 1 at gathered positions, 0 elsewhere
    // Gathered [0,0] and [1,1], so grad = [[1,0], [0,1]]
    expect(await a.grad?.cpu()).toEqual([1, 0, 0, 1]);
  });

  it("gather backward with duplicate indices accumulates", async () => {
    const a = torch.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    // Gather same index twice
    const index = torch.tensorFromArray([1, 1], [2]);
    const gathered = torch.gather(a, index, { dim: 0 });
    const loss = gathered.sum();

    await loss.backward();

    // Index 1 was gathered twice, so its gradient should be 2
    expect(await a.grad?.cpu()).toEqual([0, 2, 0]);
  });
});

describe("frontend api: scatterAdd autograd", () => {
  it("scatterAdd backward gathers gradient for src", async () => {
    // dest: [0, 0, 0] shape [3]
    // index: [1, 2] shape [2]
    // src: [10, 20] shape [2]
    // result: [0, 10, 20]
    const dest = torch.tensorFromArray([0, 0, 0], [3]);
    const index = torch.tensorFromArray([1, 2], [2]);
    const src = torch.tensorFromArray([10, 20], [2], { requiresGrad: true });
    const result = torch.scatterAdd(dest, index, src, { dim: 0 });
    const loss = result.sum();

    await loss.backward();

    expect(src.grad).not.toBeNull();
    expect(src.grad?.shape).toEqual([2]);
    // Each src element contributes to one position, so grad = [1, 1]
    expect(await src.grad?.cpu()).toEqual([1, 1]);
  });

  it("scatterAdd backward passes gradient through to dest", async () => {
    const dest = torch.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    const index = torch.tensorFromArray([0], [1]);
    const src = torch.tensorFromArray([10], [1]);
    const result = torch.scatterAdd(dest, index, src, { dim: 0 });
    const loss = result.sum();

    await loss.backward();

    expect(dest.grad).not.toBeNull();
    // All positions receive gradient 1 (sum backward is all ones)
    expect(await dest.grad?.cpu()).toEqual([1, 1, 1]);
  });
});

describe("frontend api: transpose autograd", () => {
  it("transpose backward applies inverse transpose", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
      requiresGrad: true,
    });
    const t = a.transpose({ dim0: 0, dim1: 1 }); // [3, 2]
    const loss = t.sum();

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(a.grad?.shape).toEqual([2, 3]);
    expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("transpose gradient flows through chain", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
      requiresGrad: true,
    });
    const t = a.transpose({ dim0: 0, dim1: 1 }); // [3, 2]
    const scaled = t.mul(torch.tensorFromArray([2, 2, 2, 2, 2, 2], [3, 2]));
    const loss = scaled.sum();

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(await a.grad?.cpu()).toEqual([2, 2, 2, 2, 2, 2]);
  });
});

describe("frontend api: permute", () => {
  it("permute reorders dimensions", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const p = a.permute([1, 0]);

    expect(p.shape).toEqual([3, 2]);
    expect(await p.cpu()).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("permute shares baseId (is a view)", () => {
    const a = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const p = a.permute([1, 0]);

    expect(p.baseId).toBe(a.baseId);
  });

  it("permute 3D tensor", async () => {
    // [2, 3, 4] -> [4, 2, 3]
    const data = Array.from({ length: 24 }, (_, i) => i);
    const a = torch.tensorFromArray(data, [2, 3, 4]);
    const p = a.permute([2, 0, 1]);

    expect(p.shape).toEqual([4, 2, 3]);
  });

  it("permute supports negative dims", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const p = a.permute([-1, -2]); // equivalent to [1, 0]

    expect(p.shape).toEqual([3, 2]);
    expect(await p.cpu()).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("permute backward computes inverse permutation", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
      requiresGrad: true,
    });
    const p = a.permute([1, 0]); // [3, 2]
    const loss = p.sum();

    await loss.backward();

    expect(a.grad).not.toBeNull();
    expect(a.grad?.shape).toEqual([2, 3]);
    // Gradient should be all ones
    expect(await a.grad?.cpu()).toEqual([1, 1, 1, 1, 1, 1]);
  });
});

describe("frontend api: 0-d tensors and item()", () => {
  it("sum() returns 0-d tensor", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const result = a.sum();

    expect(result.shape).toEqual([]);
    expect(await result.cpu()).toEqual([10]);
  });

  it("mean() returns 0-d tensor", async () => {
    const a = torch.tensorFromArray([2, 4, 6, 8], [2, 2]);
    const result = a.mean();

    expect(result.shape).toEqual([]);
    expect(await result.cpu()).toEqual([5]);
  });

  it("item() extracts scalar from 0-d tensor", async () => {
    const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const sumTensor = a.sum();

    const value = await sumTensor.item();

    expect(value).toBe(10);
    expect(typeof value).toBe("number");
  });

  it("item() extracts scalar from single-element tensor", async () => {
    const a = torch.tensorFromArray([42], [1]);

    const value = await a.item();

    expect(value).toBe(42);
  });

  it("item() throws for multi-element tensor", async () => {
    const a = torch.tensorFromArray([1, 2, 3], [3]);

    await expect(a.item()).rejects.toThrow("single-element");
  });

  it("0-d tensor can participate in ops via broadcast", async () => {
    const scalar = torch.tensorFromArray([5], [1]).sum(); // 0-d tensor
    const vec = torch.tensorFromArray([1, 2, 3], [3]);

    // Expand 0-d tensor for element-wise addition
    const expanded = scalar.expand([3]);
    const result = vec.add(expanded);

    expect(await result.cpu()).toEqual([6, 7, 8]);
  });
});

describe("frontend api: where", () => {
  it("where selects from x or y based on condition", async () => {
    const condition = torch.tensorFromArray([1, 0, 1, 0], [4]);
    const x = torch.tensorFromArray([10, 20, 30, 40], [4]);
    const y = torch.tensorFromArray([1, 2, 3, 4], [4]);

    const result = torch.where(condition, x, y);

    expect(result.shape).toEqual([4]);
    expect(await result.cpu()).toEqual([10, 2, 30, 4]);
  });

  it("where with 2D tensors", async () => {
    const condition = torch.tensorFromArray([1, 0, 0, 1], [2, 2]);
    const x = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const y = torch.tensorFromArray([10, 20, 30, 40], [2, 2]);

    const result = torch.where(condition, x, y);

    expect(result.shape).toEqual([2, 2]);
    expect(await result.cpu()).toEqual([1, 20, 30, 4]);
  });

  it("where with broadcasting", async () => {
    const condition = torch.tensorFromArray([1, 0], [2]);
    const x = torch.tensorFromArray([100], [1]); // broadcasts
    const y = torch.tensorFromArray([0], [1]); // broadcasts

    const result = torch.where(condition, x, y);

    expect(result.shape).toEqual([2]);
    expect(await result.cpu()).toEqual([100, 0]);
  });

  it("where treats non-zero as true", async () => {
    const condition = torch.tensorFromArray([0.5, -1, 0, 5], [4]);
    const x = torch.tensorFromArray([1, 2, 3, 4], [4]);
    const y = torch.tensorFromArray([10, 20, 30, 40], [4]);

    const result = torch.where(condition, x, y);
    expect(await result.cpu()).toEqual([1, 2, 30, 4]);
  });
});

describe("frontend api: where autograd", () => {
  it("where backward computes gradient for x", async () => {
    const condition = torch.tensorFromArray([1, 0, 1, 0], [4]);
    const x = torch.tensorFromArray([10, 20, 30, 40], [4], { requiresGrad: true });
    const y = torch.tensorFromArray([1, 2, 3, 4], [4]);

    const result = torch.where(condition, x, y);
    const loss = result.sum();
    await loss.backward();

    expect(x.grad).not.toBeNull();
    // Gradient flows only to positions where condition is true
    expect(await x.grad?.cpu()).toEqual([1, 0, 1, 0]);
  });

  it("where backward computes gradient for y", async () => {
    const condition = torch.tensorFromArray([1, 0, 1, 0], [4]);
    const x = torch.tensorFromArray([10, 20, 30, 40], [4]);
    const y = torch.tensorFromArray([1, 2, 3, 4], [4], { requiresGrad: true });

    const result = torch.where(condition, x, y);
    const loss = result.sum();
    await loss.backward();

    expect(y.grad).not.toBeNull();
    // Gradient flows only to positions where condition is false
    expect(await y.grad?.cpu()).toEqual([0, 1, 0, 1]);
  });

  it("where backward with both x and y requiring grad", async () => {
    const condition = torch.tensorFromArray([1, 0], [2]);
    const x = torch.tensorFromArray([10, 20], [2], { requiresGrad: true });
    const y = torch.tensorFromArray([1, 2], [2], { requiresGrad: true });

    const result = torch.where(condition, x, y);
    const loss = result.sum();
    await loss.backward();

    expect(await x.grad?.cpu()).toEqual([1, 0]);
    expect(await y.grad?.cpu()).toEqual([0, 1]);
  });

  it("where backward with broadcasting", async () => {
    const condition = torch.tensorFromArray([1, 0, 1], [3]);
    const x = torch.tensorFromArray([100], [1], { requiresGrad: true }); // broadcasts
    const y = torch.tensorFromArray([0, 0, 0], [3], { requiresGrad: true });

    const result = torch.where(condition, x, y);
    const loss = result.sum();
    await loss.backward();

    // x was selected at positions 0 and 2, so grad_x sums to 2
    expect(await x.grad?.cpu()).toEqual([2]);
    // y was selected at position 1
    expect(await y.grad?.cpu()).toEqual([0, 1, 0]);
  });
});
