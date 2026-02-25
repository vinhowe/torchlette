import { describe, expect, it } from "vitest";
import {
  add,
  cpuBackend,
  gather,
  getActiveBackend,
  matmul,
  mean,
  mockBackend,
  mul,
  ops,
  RuntimeEngine,
  RuntimeTensor,
  relu,
  runtimeAdd,
  runtimeCpu,
  runtimeExpand,
  runtimeMean,
  runtimeRelu,
  runtimeSub,
  runtimeTensorFromArray,
  runtimeTranspose,
  scatterAdd,
  setBackend,
  sqrt,
  sub,
  sum,
  tensorFromArray,
  transpose,
  withBackend,
} from "../src";

describe("numeric ring-2: add", () => {
  it("adds two tensors elementwise", () => {
    const a = tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = tensorFromArray([5, 6, 7, 8], [2, 2]);

    const out = add(a, b);

    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([6, 8, 10, 12]);
  });

  it("throws on size mismatch", () => {
    const a = tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = tensorFromArray([1, 2, 3], [3]);

    expect(() => add(a, b)).toThrow("broadcast");
  });

  it("broadcasts add across dimensions", () => {
    const a = tensorFromArray([1, 2, 3], [1, 3]);
    const b = tensorFromArray([10, 20, 30], [3]);

    const out = add(a, b);

    expect(out.shape).toEqual([1, 3]);
    expect(out.toArray()).toEqual([11, 22, 33]);
  });

  it("validates shape size", () => {
    expect(() => tensorFromArray([1, 2], [3])).toThrow(
      "Tensor data length does not match shape",
    );
  });

  it("uses the CPU backend by default", () => {
    setBackend("cpu");
    const active = getActiveBackend();
    expect(active.name).toBe(cpuBackend.name);
  });
});

describe("numeric ring-2: sub", () => {
  it("subtracts two tensors elementwise", () => {
    const a = tensorFromArray([5, 6, 7, 8], [2, 2]);
    const b = tensorFromArray([1, 2, 3, 4], [2, 2]);

    const out = sub(a, b);

    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([4, 4, 4, 4]);
  });

  it("supports an alpha scale for the second input", () => {
    const a = tensorFromArray([10, 10], [2]);
    const b = tensorFromArray([2, 4], [2]);

    const out = sub(a, b, { alpha: 0.5 });

    expect(out.toArray()).toEqual([9, 8]);
  });
});

describe("numeric ring-2: mul", () => {
  it("multiplies two tensors elementwise", () => {
    const a = tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = tensorFromArray([5, 6, 7, 8], [2, 2]);

    const out = mul(a, b);

    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([5, 12, 21, 32]);
  });
});

describe("numeric ring-2: matmul", () => {
  it("multiplies two 2D tensors", () => {
    const a = tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tensorFromArray([7, 8, 9, 10, 11, 12], [3, 2]);

    const out = matmul(a, b);

    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([58, 64, 139, 154]);
  });

  it("supports 1D dot products", () => {
    const a = tensorFromArray([1, 2, 3], [3]);
    const b = tensorFromArray([4, 5, 6], [3]);

    const out = matmul(a, b);

    expect(out.shape).toEqual([]);
    expect(out.toArray()).toEqual([32]);
  });

  it("supports 2D x 1D matmul", () => {
    const a = tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = tensorFromArray([7, 8, 9], [3]);

    const out = matmul(a, b);

    expect(out.shape).toEqual([2]);
    expect(out.toArray()).toEqual([50, 122]);
  });

  it("broadcasts batch dimensions", () => {
    const a = tensorFromArray([1, 0, 0, 1], [1, 2, 2]);
    const b = tensorFromArray(
      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
      [3, 2, 2],
    );

    const out = matmul(a, b);

    expect(out.shape).toEqual([3, 2, 2]);
    expect(out.toArray()).toEqual(b.toArray());
  });

  it("throws on incompatible shapes", () => {
    const a = tensorFromArray([1, 2, 3, 4], [2, 2]);
    const b = tensorFromArray([1, 2, 3, 4, 5, 6], [3, 2]);

    expect(() => matmul(a, b)).toThrow("matmul dimension mismatch");
  });
});

describe("numeric ring-2: relu", () => {
  it("clamps negative values to zero", () => {
    const a = tensorFromArray([-2, -1, 0, 3], [2, 2]);

    const out = relu(a);

    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([0, 0, 0, 3]);
  });
});

describe("numeric ring-2: transpose", () => {
  it("swaps two dimensions on a 2D tensor", () => {
    const a = tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);

    const out = transpose(a, { dim0: 0, dim1: 1 });

    expect(out.shape).toEqual([3, 2]);
    expect(out.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
  });
});

describe("numeric ring-2: sum", () => {
  it("reduces a tensor to a 0-d tensor", () => {
    const a = tensorFromArray([1, 2, 3, 4], [2, 2]);
    const result = sum(a);

    // Full reduction returns 0-d tensor (shape [])
    expect(result.shape).toEqual([]);
    expect(result.toArray()).toEqual([10]);
  });

  it("reduces over dimensions", () => {
    const a = tensorFromArray([1, 2, 3, 4], [2, 2]);

    const out = mean(a, { dim: 0 });

    if (typeof out === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    expect(out.shape).toEqual([2]);
    expect(out.toArray()).toEqual([2, 3]);
  });

  it("switches backends by name", () => {
    const backend = setBackend("cpu");
    expect(backend.name).toBe("cpu");
  });

  it("switches to the mock backend", () => {
    const backend = setBackend("mock");
    expect(backend.name).toBe(mockBackend.name);
  });

  it("exposes active backend ops helpers", () => {
    setBackend("cpu");
    const backendOps = ops();
    const a = backendOps.tensorFromArray([1, 2], [2]);
    const b = backendOps.tensorFromArray([3, 4], [2]);
    const out = backendOps.add(a, b);
    expect(out.toArray()).toEqual([4, 6]);
  });

  it("restores the previous backend with withBackend", () => {
    setBackend("cpu");
    const before = getActiveBackend().name;

    const value = withBackend("mock", (backend) => {
      const a = backend.ops.tensorFromArray([1, 2], [2]);
      const b = backend.ops.tensorFromArray([9, 9], [2]);
      return backend.ops.add(a, b).toArray();
    });

    expect(value).toEqual([1, 2]);
    expect(getActiveBackend().name).toBe(before);
  });

  it("tracks BaseId across views in the runtime tensor", async () => {
    const engine = new RuntimeEngine("cpu");
    const t = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);
    const view = engine.view(t, [4]);
    expect(view).toBeInstanceOf(RuntimeTensor);
    expect(view.baseId).toBe(t.baseId);
    expect(await engine.cpu(view)).toEqual([1, 2, 3, 4]);
  });

  it("creates new BaseIds for runtime ops", () => {
    const a = runtimeTensorFromArray([1, 2], [2]);
    const b = runtimeTensorFromArray([3, 4], [2]);
    const out = runtimeAdd(a, b);
    expect(out.baseId).not.toBe(a.baseId);
    expect(out.baseId).not.toBe(b.baseId);
  });

  it("preserves BaseId for runtime expand", async () => {
    const a = runtimeTensorFromArray([1, 2, 3], [1, 3]);
    const expanded = runtimeExpand(a, [2, 3]);

    expect(expanded.baseId).toBe(a.baseId);
    expect(await runtimeCpu(expanded)).toEqual([1, 2, 3, 1, 2, 3]);
  });

  it("exposes runtime sub/mean/relu helpers", async () => {
    const a = runtimeTensorFromArray([5, 6], [2]);
    const b = runtimeTensorFromArray([2, 4], [2]);

    const subOut = runtimeSub(a, b);
    expect(await runtimeCpu(subOut)).toEqual([3, 2]);

    const reluOut = runtimeRelu(runtimeTensorFromArray([-1, 3], [2]));
    expect(await runtimeCpu(reluOut)).toEqual([0, 3]);

    const meanOut = runtimeMean(a, { dim: 0 });
    if (typeof meanOut === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    expect(await runtimeCpu(meanOut)).toEqual([5.5]);
  });

  it("preserves BaseId for runtime transpose", async () => {
    const a = runtimeTensorFromArray([1, 2, 3, 4], [2, 2]);
    const transposed = runtimeTranspose(a, { dim0: 0, dim1: 1 });

    expect(transposed.baseId).toBe(a.baseId);
    expect(await runtimeCpu(transposed)).toEqual([1, 3, 2, 4]);
  });

  it("exposes RuntimeEngine view/reshape helpers", () => {
    const engine = new RuntimeEngine("cpu");
    const t = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);

    const view = engine.view(t, [4]);
    const reshaped = engine.reshape(t, [2, 2]);
    const transposed = engine.transpose(t, { dim0: 0, dim1: 1 });

    expect(view.baseId).toBe(t.baseId);
    expect(view.shape).toEqual([4]);
    expect(reshaped.baseId).toBe(t.baseId);
    expect(transposed.baseId).toBe(t.baseId);
    expect(transposed.shape).toEqual([2, 2]);
  });

  it("throws on invalid view/reshape sizes", async () => {
    const engine = new RuntimeEngine("cpu");
    const t = engine.tensorFromArray([1, 2, 3, 4], [2, 2]);

    // The view shape must match the tensor size, this error is thrown at force time
    const badView = engine.view(t, [3]);
    await expect(engine.cpu(badView)).rejects.toThrow(
      "View shape does not match tensor size",
    );
  });

  it("supports per-engine backend selection", async () => {
    const engine = new RuntimeEngine("mock");
    const a = engine.tensorFromArray([1, 2], [2]);
    const b = engine.tensorFromArray([9, 9], [2]);
    const out = engine.add(a, b);
    expect(await engine.cpu(out)).toEqual([1, 2]);
  });
});

describe("numeric ring-2: sqrt", () => {
  it("computes elementwise square roots", () => {
    const a = tensorFromArray([1, 4, 9, 16], [2, 2]);

    const out = sqrt(a);

    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([1, 2, 3, 4]);
  });
});

describe("numeric ring-2: gather", () => {
  it("gathers values along a dimension", () => {
    const input = tensorFromArray([10, 20, 30, 40], [4]);
    const index = tensorFromArray([3, 0, 1, 1], [4]);

    const out = gather(input, index, { dim: 0 });

    expect(out.toArray()).toEqual([40, 10, 20, 20]);
  });
});

describe("numeric ring-2: scatterAdd", () => {
  it("adds src values into target indices", () => {
    const input = tensorFromArray([1, 1, 1, 1], [4]);
    const index = tensorFromArray([0, 1, 0, 1], [4]);
    const src = tensorFromArray([1, 2, 3, 4], [4]);

    const out = scatterAdd(input, index, src, { dim: 0 });

    expect(out.toArray()).toEqual([5, 7, 1, 1]);
  });
});
