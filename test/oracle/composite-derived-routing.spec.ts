/**
 * COMPOSITE-CLOSURE F1 / C3 — the landed CPU derived-backward gate.
 *
 * The rmsnorm / layernorm CPU backwards now DERIVE unconditionally through
 * `vjpComposition` (F1) — the C2 soak flag has fired and the hand closures are
 * deleted. This gate is the landed-path referee: the frontend op's CPU backward
 * (now the derived VJP) == torch autograd, with a non-degenerate upstream
 * gradient, 2-D and batched.
 *
 * softmax is absent (C1 T1: its derived backward measured heavier, so it stays
 * hand and is checked reference-only in semantic-composite-backward.spec).
 */

import { describe, test } from "vitest";

import { Torchlette } from "../../src";
import type { Tensor } from "../../src/frontend/tensor";
import { assertClose } from "../helpers/assertions";
import { runTorchOracleBackwardBatch } from "./torch-oracle";

const api = new Torchlette("cpu");

function rand(n: number, seed: number): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * 4 - 2);
  }
  return out;
}
const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

async function readGrad(p: Tensor) {
  if (!p.grad) throw new Error("missing grad");
  return { shape: p.grad.shape, values: await p.grad.cpu() };
}

const SHAPES: { batch: number[]; label: string }[] = [
  { batch: [], label: "2D" },
  { batch: [2], label: "batched" },
];
const TOL = 2e-4;

describe("C3 — rmsnorm CPU backward derives (== torch)", () => {
  for (const { batch, label } of SHAPES) {
    test(label, async () => {
      const rows = 4;
      const cols = 6;
      const shape = [...batch, rows, cols];
      const eps = 1e-6;
      const xV = rand(prod(shape), 101);
      const wV = rand(cols, 103);
      const gV = rand(prod(shape), 105);

      const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
      const w = api.tensorFromArray(wV, [cols], { requiresGrad: true });
      const out = api.rmsnorm(x, w, eps);
      await out.backward(api.tensorFromArray(gV, shape));

      const [oracle] = await runTorchOracleBackwardBatch([
        {
          op: "rmsnorm_vjp",
          caseName: `rmsnorm.${label}`,
          inputs: [
            { values: xV, shape },
            { values: wV, shape: [cols] },
            { values: gV, shape },
          ],
          options: { eps },
        },
      ]);
      assertClose(await readGrad(x), oracle.grads[0], TOL, TOL, "dX");
      assertClose(await readGrad(w), oracle.grads[1], TOL, TOL, "dW");
    });
  }
});

describe("C3 — layernorm CPU backward derives (== torch)", () => {
  for (const { batch, label } of SHAPES) {
    test(label, async () => {
      const rows = 4;
      const cols = 6;
      const shape = [...batch, rows, cols];
      const eps = 1e-5;
      const xV = rand(prod(shape), 111);
      const wV = rand(cols, 113);
      const bV = rand(cols, 115);
      const gV = rand(prod(shape), 117);

      const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
      const w = api.tensorFromArray(wV, [cols], { requiresGrad: true });
      const b = api.tensorFromArray(bV, [cols], { requiresGrad: true });
      const out = api.layernorm(x, w, b, eps);
      await out.backward(api.tensorFromArray(gV, shape));

      const [oracle] = await runTorchOracleBackwardBatch([
        {
          op: "layernorm_vjp",
          caseName: `layernorm.${label}`,
          inputs: [
            { values: xV, shape },
            { values: wV, shape: [cols] },
            { values: bV, shape: [cols] },
            { values: gV, shape },
          ],
          options: { eps },
        },
      ]);
      assertClose(await readGrad(x), oracle.grads[0], TOL, TOL, "dX");
      assertClose(await readGrad(w), oracle.grads[1], TOL, TOL, "dW");
      assertClose(await readGrad(b), oracle.grads[2], TOL, TOL, "dB");
    });
  }
});
