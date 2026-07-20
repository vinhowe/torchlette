import { describe, test } from "vitest";

import { Torchlette } from "../../src";
import { assertClose } from "../helpers/assertions";
import { runTorchOracleBackwardBatch } from "./torch-oracle";

describe("oracle ring-3: autograd parity (cpu)", () => {
  test("mlp mse grads match PyTorch", async () => {
    const api = new Torchlette("cpu");
    const xValues = [1, 2, 3, 4, 5, 6];
    const wValues = [0.5, -1, 2, 1.5, -0.5, 0.25];
    const bValues = [0.1, -0.2];
    const yValues = [3, 1, 2, -1];

    const x = api.tensorFromArray(xValues, [2, 3], { requiresGrad: true });
    const w = api.tensorFromArray(wValues, [3, 2], { requiresGrad: true });
    const b = api.tensorFromArray(bValues, [2], { requiresGrad: true });
    const y = api.tensorFromArray(yValues, [2, 2]);

    const pred = x.matmul(w).add(b).relu();
    const diff = pred.sub(y);
    const sq = diff.mul(diff);
    const loss = sq.mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    const scalar = loss.reshape([]);

    await scalar.backward();

    if (!x.grad || !w.grad || !b.grad) {
      throw new Error("Missing gradients after backward");
    }

    const [oracle] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.0",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wValues, shape: [3, 2] },
          { values: bValues, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const grads = [];
    for (const tensor of [x.grad, w.grad, b.grad]) {
      grads.push({
        shape: tensor.shape,
        values: await tensor.cpu(),
      });
    }

    assertClose(grads[0], oracle.grads[0]);
    assertClose(grads[1], oracle.grads[1]);
    assertClose(grads[2], oracle.grads[2]);
  });

  // The §18 guard ruling, refereed. The derived log/sqrt adjoint is UNGUARDED
  // (`g/x`, `g·0.5/√x`); torch guards neither. This asserts they agree even at
  // small x — the exact domain where the DROPPED `+1e-8` denomEps biased the
  // grad (log: ~1e-4 relative @ x=1e-4). With the old guard this test would
  // diverge near 0; unguarded, it matches torch. The standing evidence that the
  // guard was policy-bias, not semantics.
  for (const opName of ["log", "sqrt"] as const) {
    test(`${opName} backward matches PyTorch (unguarded) incl. small x`, async () => {
      const api = new Torchlette("cpu");
      const xValues = [0.0001, 0.01, 0.1, 0.5, 1.0, 2.5, 7.3];
      const shape = [xValues.length];
      const x = api.tensorFromArray(xValues, shape, { requiresGrad: true });
      const out = opName === "log" ? x.log() : x.sqrt();
      const loss = out.sum();
      const scalar =
        typeof loss === "number" ? undefined : (loss as { reshape: (s: number[]) => { backward: () => Promise<void> } }).reshape([]);
      if (!scalar) throw new Error("Expected sum to return a tensor");
      await scalar.backward();
      if (!x.grad) throw new Error("Missing gradient after backward");

      const [oracle] = await runTorchOracleBackwardBatch([
        {
          op: "backward",
          caseName: `${opName}.backward.smallx`,
          inputs: [{ values: xValues, shape }],
          options: { op: opName, requiresGrad: [true] },
        },
      ]);

      assertClose(
        { shape: x.grad.shape, values: await x.grad.cpu() },
        oracle.grads[0],
      );
    });
  }
});
