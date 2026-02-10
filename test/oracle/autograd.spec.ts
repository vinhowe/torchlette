import { describe, expect, test } from "vitest";

import { Torchlette } from "../../src";
import { runTorchOracleBackwardBatch } from "./torch-oracle";

type Payload = { shape: number[]; values: number[] };

const DEFAULT_ATOL = 1e-5;
const DEFAULT_RTOL = 1e-4;

function assertClose(actual: Payload, expected: Payload): void {
  expect(actual.shape).toEqual(expected.shape);
  expect(actual.values.length).toBe(expected.values.length);
  for (let i = 0; i < actual.values.length; i += 1) {
    const a = actual.values[i];
    const b = expected.values[i];
    const diff = Math.abs(a - b);
    const tol = DEFAULT_ATOL + DEFAULT_RTOL * Math.abs(b);
    expect(diff).toBeLessThanOrEqual(tol);
  }
}

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
});
