import { describe, expect, test } from "vitest";

import { Adam, SGD, Torchlette } from "../../src";
import { runTorchOracleBackwardBatch } from "./torch-oracle";

type Payload = { shape: number[]; values: number[] };
type AdamState = { m: number[]; v: number[] };

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

function applyAdamStep(
  params: number[],
  grads: number[],
  state: AdamState | null,
  step: number,
  options: {
    lr: number;
    beta1: number;
    beta2: number;
    eps: number;
    weightDecay: number;
  },
): { params: number[]; state: AdamState } {
  const mPrev = state?.m ?? new Array(params.length).fill(0);
  const vPrev = state?.v ?? new Array(params.length).fill(0);
  const m = new Array(params.length);
  const v = new Array(params.length);
  const updated = new Array(params.length);

  const biasCorrection1 = 1 - options.beta1 ** step;
  const biasCorrection2 = 1 - options.beta2 ** step;
  const stepSize = (options.lr * Math.sqrt(biasCorrection2)) / biasCorrection1;

  for (let i = 0; i < params.length; i += 1) {
    let grad = grads[i];
    if (options.weightDecay !== 0) {
      grad += options.weightDecay * params[i];
    }
    const mVal = options.beta1 * mPrev[i] + (1 - options.beta1) * grad;
    const vVal = options.beta2 * vPrev[i] + (1 - options.beta2) * grad * grad;
    m[i] = mVal;
    v[i] = vVal;
    updated[i] =
      params[i] - (stepSize * mVal) / (Math.sqrt(vVal) + options.eps);
  }

  return { params: updated, state: { m, v } };
}

describe("oracle ring-3: optimizer parity (cpu)", () => {
  test("one-step SGD update matches PyTorch grads", { timeout: 30000 }, async () => {
    const api = new Torchlette("cpu");
    const lr = 0.1;
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
    const loss = diff.mul(diff).mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.reshape([]).backward();

    if (!w.grad || !b.grad) {
      throw new Error("Missing gradients after backward");
    }

    const optimizer = new SGD([w, b], { lr }, api);
    const [wUpdated, bUpdated] = optimizer.step();

    const [oracle] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.sgd.0",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wValues, shape: [3, 2] },
          { values: bValues, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const expectedW = oracle.grads[1].values.map(
      (grad, index) => wValues[index] - lr * grad,
    );
    const expectedB = oracle.grads[2].values.map(
      (grad, index) => bValues[index] - lr * grad,
    );

    assertClose(
      { shape: wUpdated.shape, values: await wUpdated.cpu() },
      { shape: oracle.grads[1].shape, values: expectedW },
    );
    assertClose(
      { shape: bUpdated.shape, values: await bUpdated.cpu() },
      { shape: oracle.grads[2].shape, values: expectedB },
    );
  });

  test("weight decay update matches PyTorch grads", { timeout: 30000 }, async () => {
    const api = new Torchlette("cpu");
    const lr = 0.1;
    const weightDecay = 0.2;
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
    const loss = diff.mul(diff).mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.reshape([]).backward();

    const optimizer = new SGD([w, b], { lr, weightDecay }, api);
    const [wUpdated, bUpdated] = optimizer.step();

    const [oracle] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.weightDecay.0",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wValues, shape: [3, 2] },
          { values: bValues, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const expectedW = oracle.grads[1].values.map(
      (grad, index) =>
        wValues[index] - lr * (grad + weightDecay * wValues[index]),
    );
    const expectedB = oracle.grads[2].values.map(
      (grad, index) =>
        bValues[index] - lr * (grad + weightDecay * bValues[index]),
    );

    assertClose(
      { shape: wUpdated.shape, values: await wUpdated.cpu() },
      { shape: oracle.grads[1].shape, values: expectedW },
    );
    assertClose(
      { shape: bUpdated.shape, values: await bUpdated.cpu() },
      { shape: oracle.grads[2].shape, values: expectedB },
    );
  });

  test("momentum update matches PyTorch grads over two steps", { timeout: 30000 }, async () => {
    const api = new Torchlette("cpu");
    const lr = 0.1;
    const momentum = 0.9;
    const xValues = [1, 2, 3, 4, 5, 6];
    const wValues = [0.5, -1, 2, 1.5, -0.5, 0.25];
    const bValues = [0.1, -0.2];
    const yValues = [3, 1, 2, -1];

    const w = api.tensorFromArray(wValues, [3, 2], { requiresGrad: true });
    const b = api.tensorFromArray(bValues, [2], { requiresGrad: true });
    const x = api.tensorFromArray(xValues, [2, 3]);
    const y = api.tensorFromArray(yValues, [2, 2]);

    const optimizer = new SGD([w, b], { lr, momentum }, api);

    const pred1 = x.matmul(w).add(b).relu();
    const loss1 = pred1
      .sub(y)
      .mul(pred1.sub(y))
      .mean({ dim: [0, 1], keepdim: true });
    if (typeof loss1 === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss1.reshape([]).backward();
    optimizer.step();
    optimizer.zeroGrad();

    const [oracle1] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.momentum.0",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wValues, shape: [3, 2] },
          { values: bValues, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const vW1 = oracle1.grads[1].values.slice();
    const vB1 = oracle1.grads[2].values.slice();
    const wStep1 = wValues.map((value, index) => value - lr * vW1[index]);
    const bStep1 = bValues.map((value, index) => value - lr * vB1[index]);

    const wStep1Tensor = optimizer.getParams()[0];
    const bStep1Tensor = optimizer.getParams()[1];

    const pred2 = x.matmul(wStep1Tensor).add(bStep1Tensor).relu();
    const loss2 = pred2
      .sub(y)
      .mul(pred2.sub(y))
      .mean({ dim: [0, 1], keepdim: true });
    if (typeof loss2 === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss2.reshape([]).backward();
    const [wUpdated, bUpdated] = optimizer.step();

    const [oracle2] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.momentum.1",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wStep1, shape: [3, 2] },
          { values: bStep1, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const vW2 = oracle2.grads[1].values.map(
      (grad, index) => momentum * vW1[index] + grad,
    );
    const vB2 = oracle2.grads[2].values.map(
      (grad, index) => momentum * vB1[index] + grad,
    );
    const expectedW = wStep1.map((value, index) => value - lr * vW2[index]);
    const expectedB = bStep1.map((value, index) => value - lr * vB2[index]);

    assertClose(
      { shape: wUpdated.shape, values: await wUpdated.cpu() },
      { shape: oracle2.grads[1].shape, values: expectedW },
    );
    assertClose(
      { shape: bUpdated.shape, values: await bUpdated.cpu() },
      { shape: oracle2.grads[2].shape, values: expectedB },
    );
  });

  test("one-step Adam update matches PyTorch grads", { timeout: 30000 }, async () => {
    const api = new Torchlette("cpu");
    const lr = 0.1;
    const betas: [number, number] = [0.9, 0.999];
    const eps = 1e-8;
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
    const loss = diff.mul(diff).mean({ dim: [0, 1], keepdim: true });
    if (typeof loss === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss.reshape([]).backward();

    const optimizer = new Adam([w, b], { lr, betas, eps }, api);
    const [wUpdated, bUpdated] = optimizer.step();

    const [oracle] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.adam.0",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wValues, shape: [3, 2] },
          { values: bValues, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const options = {
      lr,
      beta1: betas[0],
      beta2: betas[1],
      eps,
      weightDecay: 0,
    };
    const expectedW = applyAdamStep(
      wValues,
      oracle.grads[1].values,
      null,
      1,
      options,
    ).params;
    const expectedB = applyAdamStep(
      bValues,
      oracle.grads[2].values,
      null,
      1,
      options,
    ).params;

    assertClose(
      { shape: wUpdated.shape, values: await wUpdated.cpu() },
      { shape: oracle.grads[1].shape, values: expectedW },
    );
    assertClose(
      { shape: bUpdated.shape, values: await bUpdated.cpu() },
      { shape: oracle.grads[2].shape, values: expectedB },
    );
  });

  test("Adam state matches PyTorch grads over two steps", { timeout: 30000 }, async () => {
    const api = new Torchlette("cpu");
    const lr = 0.1;
    const betas: [number, number] = [0.9, 0.999];
    const eps = 1e-8;
    const xValues = [1, 2, 3, 4, 5, 6];
    const wValues = [0.5, -1, 2, 1.5, -0.5, 0.25];
    const bValues = [0.1, -0.2];
    const yValues = [3, 1, 2, -1];

    const w = api.tensorFromArray(wValues, [3, 2], { requiresGrad: true });
    const b = api.tensorFromArray(bValues, [2], { requiresGrad: true });
    const x = api.tensorFromArray(xValues, [2, 3]);
    const y = api.tensorFromArray(yValues, [2, 2]);

    const optimizer = new Adam([w, b], { lr, betas, eps }, api);

    const pred1 = x.matmul(w).add(b).relu();
    const loss1 = pred1
      .sub(y)
      .mul(pred1.sub(y))
      .mean({ dim: [0, 1], keepdim: true });
    if (typeof loss1 === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss1.reshape([]).backward();
    optimizer.step();
    optimizer.zeroGrad();

    const [oracle1] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.adam.1",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wValues, shape: [3, 2] },
          { values: bValues, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const options = {
      lr,
      beta1: betas[0],
      beta2: betas[1],
      eps,
      weightDecay: 0,
    };
    const wStep1 = applyAdamStep(
      wValues,
      oracle1.grads[1].values,
      null,
      1,
      options,
    );
    const bStep1 = applyAdamStep(
      bValues,
      oracle1.grads[2].values,
      null,
      1,
      options,
    );

    const wStep1Tensor = optimizer.getParams()[0];
    const bStep1Tensor = optimizer.getParams()[1];

    const pred2 = x.matmul(wStep1Tensor).add(bStep1Tensor).relu();
    const loss2 = pred2
      .sub(y)
      .mul(pred2.sub(y))
      .mean({ dim: [0, 1], keepdim: true });
    if (typeof loss2 === "number") {
      throw new Error("Expected mean to return a tensor");
    }
    await loss2.reshape([]).backward();
    const [wUpdated, bUpdated] = optimizer.step();

    const [oracle2] = await runTorchOracleBackwardBatch([
      {
        op: "mlp_mse_backward",
        caseName: "mlp.mse.adam.2",
        inputs: [
          { values: xValues, shape: [2, 3] },
          { values: wStep1.params, shape: [3, 2] },
          { values: bStep1.params, shape: [2] },
          { values: yValues, shape: [2, 2] },
        ],
      },
    ]);

    const expectedW = applyAdamStep(
      wStep1.params,
      oracle2.grads[1].values,
      wStep1.state,
      2,
      options,
    ).params;
    const expectedB = applyAdamStep(
      bStep1.params,
      oracle2.grads[2].values,
      bStep1.state,
      2,
      options,
    ).params;

    assertClose(
      { shape: wUpdated.shape, values: await wUpdated.cpu() },
      { shape: oracle2.grads[1].shape, values: expectedW },
    );
    assertClose(
      { shape: bUpdated.shape, values: await bUpdated.cpu() },
      { shape: oracle2.grads[2].shape, values: expectedB },
    );
  });
});
