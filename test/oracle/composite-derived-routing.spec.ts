/**
 * COMPOSITE-CLOSURE F1 / C2 — the CPU derived-routing gate.
 *
 * C2 routes the rmsnorm / layernorm CPU backwards through the derived
 * `vjpComposition` (F1) behind `TORCHLETTE_DERIVED_COMPOSITE_BWD`. This gate
 * proves BOTH flag states are correct and mutually consistent:
 *
 *   - flag OFF (hand closure)  == torch autograd  (the incumbent oracle)
 *   - flag ON  (derived VJP)   == torch autograd  (the construction proof, landed)
 *   - derived == hand          within the named reassociation bound L-COMP
 *
 * The softmax closure is NOT routed (C1 T1: derived is heavier), so it is absent
 * here — its derived form is checked reference-only in
 * semantic-composite-backward.spec.ts.
 */

import { afterEach, describe, test } from "vitest";

import { Torchlette } from "../../src";
import { ENV } from "../../src/core/env";
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

afterEach(() => {
  ENV.TORCHLETTE_DERIVED_COMPOSITE_BWD = undefined;
});

const SHAPES: { batch: number[]; label: string }[] = [
  { batch: [], label: "2D" },
  { batch: [2], label: "batched" },
];
const BOUND = 1e-5;
const TOL = 2e-4;

describe("C2 — rmsnorm CPU backward routes derived (both flag states)", () => {
  for (const { batch, label } of SHAPES) {
    test(label, async () => {
      const rows = 4;
      const cols = 6;
      const shape = [...batch, rows, cols];
      const eps = 1e-6;
      const xV = rand(prod(shape), 101);
      const wV = rand(cols, 103);
      const gV = rand(prod(shape), 105);

      const run = async (derived: boolean) => {
        ENV.TORCHLETTE_DERIVED_COMPOSITE_BWD = derived ? "1" : undefined;
        const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
        const w = api.tensorFromArray(wV, [cols], { requiresGrad: true });
        const out = api.rmsnorm(x, w, eps);
        await out.backward(api.tensorFromArray(gV, shape));
        return { dX: await readGrad(x), dW: await readGrad(w) };
      };

      const hand = await run(false);
      const derived = await run(true);
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

      assertClose(hand.dX, oracle.grads[0], TOL, TOL, "hand dX");
      assertClose(hand.dW, oracle.grads[1], TOL, TOL, "hand dW");
      assertClose(derived.dX, oracle.grads[0], TOL, TOL, "derived dX");
      assertClose(derived.dW, oracle.grads[1], TOL, TOL, "derived dW");
      assertClose(derived.dX, hand.dX, BOUND, BOUND, "derived==hand dX");
      assertClose(derived.dW, hand.dW, BOUND, BOUND, "derived==hand dW");
    });
  }
});

describe("C2 — layernorm CPU backward routes derived (both flag states)", () => {
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

      const run = async (derived: boolean) => {
        ENV.TORCHLETTE_DERIVED_COMPOSITE_BWD = derived ? "1" : undefined;
        const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
        const w = api.tensorFromArray(wV, [cols], { requiresGrad: true });
        const b = api.tensorFromArray(bV, [cols], { requiresGrad: true });
        const out = api.layernorm(x, w, b, eps);
        await out.backward(api.tensorFromArray(gV, shape));
        return {
          dX: await readGrad(x),
          dW: await readGrad(w),
          dB: await readGrad(b),
        };
      };

      const hand = await run(false);
      const derived = await run(true);
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

      assertClose(hand.dX, oracle.grads[0], TOL, TOL, "hand dX");
      assertClose(hand.dW, oracle.grads[1], TOL, TOL, "hand dW");
      assertClose(hand.dB, oracle.grads[2], TOL, TOL, "hand dB");
      assertClose(derived.dX, oracle.grads[0], TOL, TOL, "derived dX");
      assertClose(derived.dW, oracle.grads[1], TOL, TOL, "derived dW");
      assertClose(derived.dB, oracle.grads[2], TOL, TOL, "derived dB");
      assertClose(derived.dX, hand.dX, BOUND, BOUND, "derived==hand dX");
      assertClose(derived.dW, hand.dW, BOUND, BOUND, "derived==hand dW");
      assertClose(derived.dB, hand.dB, BOUND, BOUND, "derived==hand dB");
    });
  }
});
