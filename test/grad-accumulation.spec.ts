import { describe, expect, it } from "vitest";

import { Torchlette } from "../src";

/**
 * `.grad` is a CROSS-CALL accumulator (PyTorch torch.Tensor.grad semantics):
 * each backward() ADDS into .grad; zeroGrad() resets it. This is what lets
 * gradient accumulation over micro-batches work without a manual accumulator
 * buffer (which the DiLoCo trainer used to hand-roll). Standard loops zero
 * each iteration, so the first backward sees a null grad and accumulate ==
 * overwrite — i.e. this change is invisible to them (verified by the rest of
 * the suite staying green).
 */
describe("gradient accumulation — .grad is a cross-call accumulator", () => {
  const api = new Torchlette("cpu");

  it("backward() accumulates into .grad across calls (no zeroGrad between)", async () => {
    const x = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    // loss1 = sum(x*x) -> d/dx = 2x = [2,4,6]
    await api.mul(x, x).sum().backward();
    expect(Array.from(await x.grad!.cpu())).toEqual([2, 4, 6]);
    // loss2 = sum(x) -> d/dx = 1 ; WITHOUT zeroGrad it adds onto the above
    await x.sum().backward();
    expect(Array.from(await x.grad!.cpu())).toEqual([3, 5, 7]); // [2,4,6] + [1,1,1]
  });

  it("zeroGrad() resets the accumulator (fresh overwrite after zero)", async () => {
    const x = api.tensorFromArray([1, 2, 3], [3], { requiresGrad: true });
    await api.mul(x, x).sum().backward(); // grad = [2,4,6]
    x.zeroGrad();
    await x.sum().backward(); // grad = [1,1,1], NOT accumulated onto [2,4,6]
    expect(Array.from(await x.grad!.cpu())).toEqual([1, 1, 1]);
  });

  it("N accumulated micro-backwards == one backward over the summed loss", async () => {
    const init = [1, 2, 3, 4];
    // Reference: a single backward over (L_a + L_b) in one graph.
    const xRef = api.tensorFromArray(init, [4], { requiresGrad: true });
    const la = api.mul(xRef, xRef).sum(); // sum(x^2) -> 2x
    const lb = xRef.sum(); //               sum(x)   -> 1
    await api.add(la, lb).backward();
    const ref = Array.from(await xRef.grad!.cpu());

    // Accumulated: two separate forward+backward passes, no zeroGrad between.
    const xAcc = api.tensorFromArray(init, [4], { requiresGrad: true });
    await api.mul(xAcc, xAcc).sum().backward();
    await xAcc.sum().backward();
    const acc = Array.from(await xAcc.grad!.cpu());

    expect(ref).toEqual([3, 5, 7, 9]); // 2x + 1
    for (let i = 0; i < ref.length; i++) expect(acc[i]).toBeCloseTo(ref[i], 6);
  });
});
