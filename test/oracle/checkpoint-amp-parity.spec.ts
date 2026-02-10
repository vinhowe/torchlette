/**
 * Checkpoint + AMP Parity Tests
 *
 * These tests verify that torchlette's checkpoint and AMP implementations
 * produce the same gradients as PyTorch's implementations.
 */

import { describe, expect, test } from "vitest";
import { Torchlette } from "../../src";
import { checkpoint } from "../../src/nn/checkpoint";
import { runTorchOracleFullBatch, type OracleCase } from "./torch-oracle";

type Payload = { shape: number[]; values: number[] };

const DEFAULT_ATOL = 1e-5;
const DEFAULT_RTOL = 1e-4;

// Looser tolerance for AMP (f16 precision)
const AMP_ATOL = 1e-2;
const AMP_RTOL = 1e-2;

function assertClose(
  actual: Payload,
  expected: Payload,
  atol = DEFAULT_ATOL,
  rtol = DEFAULT_RTOL,
): void {
  expect(actual.shape).toEqual(expected.shape);
  expect(actual.values.length).toBe(expected.values.length);
  for (let i = 0; i < actual.values.length; i += 1) {
    const a = actual.values[i];
    const b = expected.values[i];
    // Handle null values (NaN/Inf from oracle)
    if (b === null) {
      // Skip comparison for NaN/Inf
      continue;
    }
    const diff = Math.abs(a - b);
    const tol = atol + rtol * Math.abs(b);
    if (diff > tol) {
      throw new Error(
        `Mismatch at index ${i}: actual=${a}, expected=${b}, diff=${diff}, tol=${tol}`,
      );
    }
  }
}

describe("Checkpoint Parity with PyTorch", () => {
  test("simple checkpoint grads match PyTorch", { timeout: 15000 }, async () => {
    // Fixed test data
    const xValues = [0.1, 0.2, 0.3, 0.4];
    const wValues = [
      0.01, 0.02, 0.03, 0.04,
      0.05, 0.06, 0.07, 0.08,
      0.09, 0.10, 0.11, 0.12,
      0.13, 0.14, 0.15, 0.16,
    ];
    const bValues = [0.01, 0.02, 0.03, 0.04];

    // Run in torchlette with checkpoint
    const api = new Torchlette("cpu");
    const x = api.tensorFromArray(xValues, [1, 4], { requiresGrad: true });
    const w = api.tensorFromArray(wValues, [4, 4], { requiresGrad: true });
    const b = api.tensorFromArray(bValues, [4], { requiresGrad: true });

    // Simple layer function: relu(x @ w + b)
    function layerFn(input: typeof x, weight: typeof w, bias: typeof b) {
      return input.matmul(weight).add(bias).relu();
    }

    // Apply 2 layers with checkpoint
    let h = x;
    for (let i = 0; i < 2; i++) {
      h = checkpoint(api, (inp) => layerFn(inp, w, b), [h]);
    }
    const loss = h.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");

    await loss.backward();

    // Get torchlette gradients
    const torchletteGrads = {
      x: { shape: x.grad!.shape, values: await x.grad!.cpu() },
      w: { shape: w.grad!.shape, values: await w.grad!.cpu() },
      b: { shape: b.grad!.shape, values: await b.grad!.cpu() },
    };

    // Run in PyTorch via oracle
    const oracleCase: OracleCase = {
      op: "checkpoint_forward_backward",
      caseName: "checkpoint_simple",
      inputs: [
        { values: xValues, shape: [1, 4] },
        { values: wValues, shape: [4, 4] },
        { values: bValues, shape: [4] },
      ],
      options: {
        numLayers: 2,
        useCheckpoint: true,
      },
    };

    const [oracleResult] = await runTorchOracleFullBatch([oracleCase]);

    // Compare gradients
    assertClose(torchletteGrads.x, oracleResult.grads![0]!);
    assertClose(torchletteGrads.w, oracleResult.grads![1]!);
    assertClose(torchletteGrads.b, oracleResult.grads![2]!);
  });

  test("checkpoint grads equal non-checkpoint grads in torchlette", async () => {
    // This verifies checkpoint doesn't change gradient values
    const xValues = [0.1, 0.2, 0.3, 0.4];
    const wValues = [
      0.01, 0.02, 0.03, 0.04,
      0.05, 0.06, 0.07, 0.08,
      0.09, 0.10, 0.11, 0.12,
      0.13, 0.14, 0.15, 0.16,
    ];
    const bValues = [0.01, 0.02, 0.03, 0.04];

    // Model without checkpoint
    const api1 = new Torchlette("cpu");
    const x1 = api1.tensorFromArray(xValues, [1, 4], { requiresGrad: true });
    const w1 = api1.tensorFromArray(wValues, [4, 4], { requiresGrad: true });
    const b1 = api1.tensorFromArray(bValues, [4], { requiresGrad: true });

    let h1 = x1;
    for (let i = 0; i < 2; i++) {
      h1 = h1.matmul(w1).add(b1).relu();
    }
    const loss1 = h1.sum();
    if (typeof loss1 === "number") throw new Error("Expected tensor");
    await loss1.backward();

    // Model with checkpoint
    const api2 = new Torchlette("cpu");
    const x2 = api2.tensorFromArray(xValues, [1, 4], { requiresGrad: true });
    const w2 = api2.tensorFromArray(wValues, [4, 4], { requiresGrad: true });
    const b2 = api2.tensorFromArray(bValues, [4], { requiresGrad: true });

    let h2 = x2;
    for (let i = 0; i < 2; i++) {
      h2 = checkpoint(api2, (inp) => inp.matmul(w2).add(b2).relu(), [h2]);
    }
    const loss2 = h2.sum();
    if (typeof loss2 === "number") throw new Error("Expected tensor");
    await loss2.backward();

    // Compare gradients
    const grads1 = {
      x: { shape: x1.grad!.shape, values: await x1.grad!.cpu() },
      w: { shape: w1.grad!.shape, values: await w1.grad!.cpu() },
      b: { shape: b1.grad!.shape, values: await b1.grad!.cpu() },
    };
    const grads2 = {
      x: { shape: x2.grad!.shape, values: await x2.grad!.cpu() },
      w: { shape: w2.grad!.shape, values: await w2.grad!.cpu() },
      b: { shape: b2.grad!.shape, values: await b2.grad!.cpu() },
    };

    assertClose(grads2.x, grads1.x);
    assertClose(grads2.w, grads1.w);
    assertClose(grads2.b, grads1.b);
  });
});

describe("AMP Parity with PyTorch", () => {
  test("autocast forward matches PyTorch", async () => {
    const xValues = [0.1, 0.2, 0.3, 0.4];
    const wValues = [
      0.01, 0.02, 0.03, 0.04,
      0.05, 0.06, 0.07, 0.08,
      0.09, 0.10, 0.11, 0.12,
      0.13, 0.14, 0.15, 0.16,
    ];
    const bValues = [0.01, 0.02, 0.03, 0.04];

    // Run in torchlette with autocast
    const api = new Torchlette("cpu");
    const x = api.tensorFromArray(xValues, [1, 4], { requiresGrad: true });
    const w = api.tensorFromArray(wValues, [4, 4], { requiresGrad: true });
    const b = api.tensorFromArray(bValues, [4], { requiresGrad: true });

    const output = await api.autocastAsync(async () => {
      const h = x.matmul(w).add(b).relu();
      return h;
    });

    const loss = output.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.backward();

    const torchletteGrads = {
      x: { shape: x.grad!.shape, values: await x.grad!.cpu() },
      w: { shape: w.grad!.shape, values: await w.grad!.cpu() },
      b: { shape: b.grad!.shape, values: await b.grad!.cpu() },
    };

    // Run in PyTorch via oracle
    const oracleCase: OracleCase = {
      op: "amp_forward_backward",
      caseName: "amp_simple",
      inputs: [
        { values: xValues, shape: [1, 4] },
        { values: wValues, shape: [4, 4] },
        { values: bValues, shape: [4] },
      ],
      options: {
        useAmp: true,
        deviceType: "cpu",
      },
    };

    const [oracleResult] = await runTorchOracleFullBatch([oracleCase]);

    // Compare gradients (with looser tolerance for AMP)
    assertClose(torchletteGrads.x, oracleResult.grads![0]!, AMP_ATOL, AMP_RTOL);
    assertClose(torchletteGrads.w, oracleResult.grads![1]!, AMP_ATOL, AMP_RTOL);
    assertClose(torchletteGrads.b, oracleResult.grads![2]!, AMP_ATOL, AMP_RTOL);
  });
});

describe("Checkpoint + AMP Combined Parity", () => {
  test("checkpoint + autocast grads match PyTorch", async () => {
    const xValues = [0.1, 0.2, 0.3, 0.4];
    const wValues = [
      0.01, 0.02, 0.03, 0.04,
      0.05, 0.06, 0.07, 0.08,
      0.09, 0.10, 0.11, 0.12,
      0.13, 0.14, 0.15, 0.16,
    ];
    const bValues = [0.01, 0.02, 0.03, 0.04];

    // Run in torchlette with both checkpoint and autocast
    const api = new Torchlette("cpu");
    const x = api.tensorFromArray(xValues, [1, 4], { requiresGrad: true });
    const w = api.tensorFromArray(wValues, [4, 4], { requiresGrad: true });
    const b = api.tensorFromArray(bValues, [4], { requiresGrad: true });

    const output = await api.autocastAsync(async () => {
      let h = x;
      for (let i = 0; i < 2; i++) {
        h = checkpoint(api, (inp) => inp.matmul(w).add(b).relu(), [h]);
      }
      return h;
    });

    const loss = output.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.backward();

    const torchletteGrads = {
      x: { shape: x.grad!.shape, values: await x.grad!.cpu() },
      w: { shape: w.grad!.shape, values: await w.grad!.cpu() },
      b: { shape: b.grad!.shape, values: await b.grad!.cpu() },
    };

    // Run in PyTorch via oracle
    const oracleCase: OracleCase = {
      op: "checkpoint_amp_forward_backward",
      caseName: "checkpoint_amp_combined",
      inputs: [
        { values: xValues, shape: [1, 4] },
        { values: wValues, shape: [4, 4] },
        { values: bValues, shape: [4] },
      ],
      options: {
        numLayers: 2,
        useCheckpoint: true,
        useAmp: true,
        deviceType: "cpu",
      },
    };

    const [oracleResult] = await runTorchOracleFullBatch([oracleCase]);

    // Compare gradients (with looser tolerance for AMP)
    assertClose(torchletteGrads.x, oracleResult.grads![0]!, AMP_ATOL, AMP_RTOL);
    assertClose(torchletteGrads.w, oracleResult.grads![1]!, AMP_ATOL, AMP_RTOL);
    assertClose(torchletteGrads.b, oracleResult.grads![2]!, AMP_ATOL, AMP_RTOL);
  });
});

describe("Multi-layer MLP Parity", () => {
  test("3-layer MLP with checkpoint matches PyTorch", async () => {
    // More complex test with multiple layers, each with different weights
    const xValues = Array.from({ length: 8 }, (_, i) => (i + 1) * 0.1);

    // Layer 1: 8 -> 16
    const w1Values = Array.from({ length: 8 * 16 }, (_, i) => ((i % 10) - 5) * 0.01);
    const b1Values = Array.from({ length: 16 }, (_, i) => i * 0.001);

    // Layer 2: 16 -> 16
    const w2Values = Array.from({ length: 16 * 16 }, (_, i) => ((i % 7) - 3) * 0.01);
    const b2Values = Array.from({ length: 16 }, (_, i) => -i * 0.001);

    // Layer 3: 16 -> 4
    const w3Values = Array.from({ length: 16 * 4 }, (_, i) => ((i % 5) - 2) * 0.01);
    const b3Values = Array.from({ length: 4 }, (_, i) => i * 0.002);

    // Run in torchlette
    const api = new Torchlette("cpu");
    const x = api.tensorFromArray(xValues, [1, 8], { requiresGrad: true });
    const w1 = api.tensorFromArray(w1Values, [8, 16], { requiresGrad: true });
    const b1 = api.tensorFromArray(b1Values, [16], { requiresGrad: true });
    const w2 = api.tensorFromArray(w2Values, [16, 16], { requiresGrad: true });
    const b2 = api.tensorFromArray(b2Values, [16], { requiresGrad: true });
    const w3 = api.tensorFromArray(w3Values, [16, 4], { requiresGrad: true });
    const b3 = api.tensorFromArray(b3Values, [4], { requiresGrad: true });

    // Forward with checkpoint on each layer
    let h = checkpoint(api, (inp) => inp.matmul(w1).add(b1).relu(), [x]);
    h = checkpoint(api, (inp) => inp.matmul(w2).add(b2).relu(), [h]);
    h = checkpoint(api, (inp) => inp.matmul(w3).add(b3).relu(), [h]);

    const loss = h.sum();
    if (typeof loss === "number") throw new Error("Expected tensor");
    await loss.backward();

    // Run in PyTorch via oracle
    const oracleCase: OracleCase = {
      op: "checkpoint_mlp_backward",
      caseName: "checkpoint_mlp_3layer",
      inputs: [
        { values: xValues, shape: [1, 8] },
        { values: w1Values, shape: [8, 16] },
        { values: b1Values, shape: [16] },
        { values: w2Values, shape: [16, 16] },
        { values: b2Values, shape: [16] },
        { values: w3Values, shape: [16, 4] },
        { values: b3Values, shape: [4] },
      ],
      options: {
        useCheckpoint: true,
      },
    };

    const [oracleResult] = await runTorchOracleFullBatch([oracleCase]);

    // Compare x gradient
    const xGrad = { shape: x.grad!.shape, values: await x.grad!.cpu() };
    assertClose(xGrad, oracleResult.grads![0]!);

    // Compare w1, b1
    const w1Grad = { shape: w1.grad!.shape, values: await w1.grad!.cpu() };
    const b1Grad = { shape: b1.grad!.shape, values: await b1.grad!.cpu() };
    assertClose(w1Grad, oracleResult.grads![1]!);
    assertClose(b1Grad, oracleResult.grads![2]!);

    // Compare w2, b2
    const w2Grad = { shape: w2.grad!.shape, values: await w2.grad!.cpu() };
    const b2Grad = { shape: b2.grad!.shape, values: await b2.grad!.cpu() };
    assertClose(w2Grad, oracleResult.grads![3]!);
    assertClose(b2Grad, oracleResult.grads![4]!);

    // Compare w3, b3
    const w3Grad = { shape: w3.grad!.shape, values: await w3.grad!.cpu() };
    const b3Grad = { shape: b3.grad!.shape, values: await b3.grad!.cpu() };
    assertClose(w3Grad, oracleResult.grads![5]!);
    assertClose(b3Grad, oracleResult.grads![6]!);
  });
});
