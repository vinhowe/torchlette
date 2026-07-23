/**
 * COMPOSITE-CLOSURE F1 / C1 — the CompNode-adjoint construction proof.
 *
 * `vjpComposition` (ops/semantic/emit-rt.ts) is ONE structural reverse-mode pass
 * over the SAME `CompNode` composition each op's forward is authored from. This
 * gate is the construction proof the audit (rung 7) said was missing: the derived
 * backward is checked
 *
 *   (1) == torch autograd (the referee) for softmax / log_softmax / rmsnorm /
 *       layernorm / cross_entropy, with a NON-degenerate upstream gradient, in
 *       2-D and batched, f32 and f16-realized; AND
 *   (2) == the HAND VJP (the frontend op's CPU backward closure) within a NAMED
 *       reassociation bound — softmax / rmsnorm / layernorm (the three with a CPU
 *       hand closure). log_softmax and cross_entropy are composition-defined ONLY
 *       (no CPU hand form exists — their fused GPU kernels are asserted in C3), so
 *       they carry the torch referee alone.
 *
 * The upstream gradient is random (not ones): softmax's g=ones case is degenerate
 * (Σ s = 1 constant → zero grad), so a ones-referee proves nothing.
 */

import { describe, test } from "vitest";

import { Torchlette } from "../../src";
import {
  CROSS_ENTROPY_DEF,
  LAYERNORM_DEF,
  LOG_SOFTMAX_DEF,
  RMSNORM_DEF,
  SOFTMAX_DEF,
  vjpComposition,
} from "../../src/ops/semantic";
import type { Tensor as RuntimeTensor } from "../../src/runtime/tensor";
import { assertClose } from "../helpers/assertions";
import { runTorchOracleBackwardBatch } from "./torch-oracle";

const api = new Torchlette("cpu");
const rt = api.runtime;

function rand(n: number, seed: number, lo = -2, hi = 2): number[] {
  const out: number[] = [];
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out.push((s / 0xffffffff) * (hi - lo) + lo);
  }
  return out;
}
const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

async function read(t: RuntimeTensor) {
  return { shape: t.shape.slice(), values: await rt.cpu(t) };
}

// ===========================================================================
// (1) THE CONSTRUCTION PROOF — vjpComposition == torch, non-degenerate g.
// ===========================================================================

const SHAPES: { batch: number[]; label: string }[] = [
  { batch: [], label: "2D" },
  { batch: [2], label: "batched" },
];

describe("vjpComposition == torch autograd (F1 construction proof)", () => {
  for (const { batch, label } of SHAPES) {
    for (const f16 of [false, true]) {
      const tag = `${label}${f16 ? " f16-realized" : ""}`;
      const tol = f16 ? 5e-2 : 2e-4;

      const cast = (t: RuntimeTensor) =>
        f16 ? api.toDtype(api._wrap(t), "f16")._unwrap() : t;

      test(`softmax ${tag}`, async () => {
        const rows = 4;
        const cols = 6;
        const shape = [...batch, rows, cols];
        const dim = shape.length - 1;
        const xV = rand(prod(shape), 11);
        const gV = rand(prod(shape), 12);
        const x = cast(api.tensorFromArray(xV, shape)._unwrap());
        const g = cast(api.tensorFromArray(gV, shape)._unwrap());
        const grads = vjpComposition(SOFTMAX_DEF, rt, dim, { x }, g);
        const [oracle] = await runTorchOracleBackwardBatch([
          {
            op: "softmax_vjp",
            caseName: `softmax.${tag}`,
            inputs: [
              { values: xV, shape },
              { values: gV, shape },
            ],
            options: { dim },
          },
        ]);
        assertClose(await read(grads.x), oracle.grads[0], tol, tol, "dX");
      });

      test(`log_softmax ${tag}`, async () => {
        const rows = 4;
        const cols = 6;
        const shape = [...batch, rows, cols];
        const dim = shape.length - 1;
        const xV = rand(prod(shape), 21);
        const gV = rand(prod(shape), 22);
        const x = cast(api.tensorFromArray(xV, shape)._unwrap());
        const g = cast(api.tensorFromArray(gV, shape)._unwrap());
        const grads = vjpComposition(LOG_SOFTMAX_DEF, rt, dim, { x }, g);
        const [oracle] = await runTorchOracleBackwardBatch([
          {
            op: "log_softmax_vjp",
            caseName: `log_softmax.${tag}`,
            inputs: [
              { values: xV, shape },
              { values: gV, shape },
            ],
            options: { dim },
          },
        ]);
        assertClose(await read(grads.x), oracle.grads[0], tol, tol, "dX");
      });

      test(`rmsnorm ${tag}`, async () => {
        const rows = 4;
        const cols = 6;
        const shape = [...batch, rows, cols];
        const dim = shape.length - 1;
        const eps = RMSNORM_DEF.eps as number;
        const xV = rand(prod(shape), 31);
        const wV = rand(cols, 33);
        const gV = rand(prod(shape), 34);
        const x = cast(api.tensorFromArray(xV, shape)._unwrap());
        const w = cast(api.tensorFromArray(wV, [cols])._unwrap());
        const g = cast(api.tensorFromArray(gV, shape)._unwrap());
        const grads = vjpComposition(RMSNORM_DEF, rt, dim, { x, w, eps }, g);
        const [oracle] = await runTorchOracleBackwardBatch([
          {
            op: "rmsnorm_vjp",
            caseName: `rmsnorm.${tag}`,
            inputs: [
              { values: xV, shape },
              { values: wV, shape: [cols] },
              { values: gV, shape },
            ],
            options: { eps },
          },
        ]);
        assertClose(await read(grads.x), oracle.grads[0], tol, tol, "dX");
        assertClose(await read(grads.w), oracle.grads[1], tol, tol, "dW");
      });

      test(`layernorm ${tag}`, async () => {
        const rows = 4;
        const cols = 6;
        const shape = [...batch, rows, cols];
        const dim = shape.length - 1;
        const eps = LAYERNORM_DEF.eps as number;
        const xV = rand(prod(shape), 41);
        const wV = rand(cols, 43);
        const bV = rand(cols, 45);
        const gV = rand(prod(shape), 46);
        const x = cast(api.tensorFromArray(xV, shape)._unwrap());
        const w = cast(api.tensorFromArray(wV, [cols])._unwrap());
        const b = cast(api.tensorFromArray(bV, [cols])._unwrap());
        const g = cast(api.tensorFromArray(gV, shape)._unwrap());
        const grads = vjpComposition(
          LAYERNORM_DEF,
          rt,
          dim,
          { x, w, b, eps },
          g,
        );
        const [oracle] = await runTorchOracleBackwardBatch([
          {
            op: "layernorm_vjp",
            caseName: `layernorm.${tag}`,
            inputs: [
              { values: xV, shape },
              { values: wV, shape: [cols] },
              { values: bV, shape: [cols] },
              { values: gV, shape },
            ],
            options: { eps },
          },
        ]);
        assertClose(await read(grads.x), oracle.grads[0], tol, tol, "dX");
        assertClose(await read(grads.w), oracle.grads[1], tol, tol, "dW");
        assertClose(await read(grads.b), oracle.grads[2], tol, tol, "dB");
      });

      test(`cross_entropy ${tag}`, async () => {
        const B = batch.length ? 2 * 4 : 4; // flatten batch into rows
        const V = 6;
        const logitsShape = [B, V];
        const dim = 1;
        const xV = rand(B * V, 51);
        const gV = rand(B, 52);
        // targets in [0, V)
        const tV = rand(B, 53, 0, V).map((v) => Math.min(V - 1, Math.floor(v)));
        const logits = cast(api.tensorFromArray(xV, logitsShape)._unwrap());
        const target = api
          .toDtype(api.tensorFromArray(tV, [B, 1]), "i32")
          ._unwrap();
        const gbar = cast(api.tensorFromArray(gV, [B, 1])._unwrap());
        const grads = vjpComposition(
          CROSS_ENTROPY_DEF,
          rt,
          dim,
          { x: logits, target },
          gbar,
        );
        const [oracle] = await runTorchOracleBackwardBatch([
          {
            op: "cross_entropy_vjp",
            caseName: `cross_entropy.${tag}`,
            inputs: [
              { values: xV, shape: logitsShape },
              { values: tV, shape: [B] },
              { values: gV, shape: [B] },
            ],
          },
        ]);
        assertClose(await read(grads.x), oracle.grads[0], tol, tol, "dLogits");
      });
    }
  }
});

// ===========================================================================
// (2) THE NAMED REASSOCIATION BOUND — vjpComposition == the hand VJP closure.
// The hand closure is reached via the frontend op's CPU backward, injected with
// the SAME random upstream gradient. Both CPU/f32: the only difference is the
// reduction/product ORDER, so the delta is a stated fp-reassociation lemma
// (L-COMP), bounded below the run-to-run GPU floor. We MEASURE it here.
// ===========================================================================

describe("vjpComposition == hand VJP (named reassociation bound L-COMP)", () => {
  const BOUND = 1e-5; // measured ceiling; see the campaign report (well under it).

  async function handGrads(
    make: () => {
      out: import("../../src/frontend/tensor").Tensor;
      params: import("../../src/frontend/tensor").Tensor[];
    },
    gV: number[],
    outShape: number[],
  ) {
    const { out, params } = make();
    const g = api.tensorFromArray(gV, outShape);
    await out.backward(g);
    const grads: { shape: number[]; values: number[] }[] = [];
    for (const p of params) {
      if (!p.grad) throw new Error("missing grad");
      grads.push({ shape: p.grad.shape, values: await p.grad.cpu() });
    }
    return grads;
  }

  for (const { batch, label } of SHAPES) {
    test(`softmax ${label}`, async () => {
      const rows = 4;
      const cols = 6;
      const shape = [...batch, rows, cols];
      const dim = shape.length - 1;
      const xV = rand(prod(shape), 61);
      const gV = rand(prod(shape), 62);
      const derived = vjpComposition(
        SOFTMAX_DEF,
        rt,
        dim,
        { x: api.tensorFromArray(xV, shape)._unwrap() },
        api.tensorFromArray(gV, shape)._unwrap(),
      );
      const dX = await read(derived.x);
      const [hand] = await handGrads(
        () => {
          const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
          return { out: api.softmax(x, dim), params: [x] };
        },
        gV,
        shape,
      );
      assertClose(dX, hand, BOUND, BOUND, "softmax dX");
    });

    test(`rmsnorm ${label}`, async () => {
      const rows = 4;
      const cols = 6;
      const shape = [...batch, rows, cols];
      const dim = shape.length - 1;
      const eps = RMSNORM_DEF.eps as number;
      const xV = rand(prod(shape), 71);
      const wV = rand(cols, 73);
      const gV = rand(prod(shape), 74);
      const derived = vjpComposition(
        RMSNORM_DEF,
        rt,
        dim,
        {
          x: api.tensorFromArray(xV, shape)._unwrap(),
          w: api.tensorFromArray(wV, [cols])._unwrap(),
          eps,
        },
        api.tensorFromArray(gV, shape)._unwrap(),
      );
      const dX = await read(derived.x);
      const dW = await read(derived.w);
      const [handX, handW] = await handGrads(
        () => {
          const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
          const w = api.tensorFromArray(wV, [cols], { requiresGrad: true });
          return { out: api.rmsnorm(x, w, eps), params: [x, w] };
        },
        gV,
        shape,
      );
      assertClose(dX, handX, BOUND, BOUND, "rmsnorm dX");
      assertClose(dW, handW, BOUND, BOUND, "rmsnorm dW");
    });

    test(`layernorm ${label}`, async () => {
      const rows = 4;
      const cols = 6;
      const shape = [...batch, rows, cols];
      const dim = shape.length - 1;
      const eps = LAYERNORM_DEF.eps as number;
      const xV = rand(prod(shape), 81);
      const wV = rand(cols, 83);
      const bV = rand(cols, 85);
      const gV = rand(prod(shape), 86);
      const derived = vjpComposition(
        LAYERNORM_DEF,
        rt,
        dim,
        {
          x: api.tensorFromArray(xV, shape)._unwrap(),
          w: api.tensorFromArray(wV, [cols])._unwrap(),
          b: api.tensorFromArray(bV, [cols])._unwrap(),
          eps,
        },
        api.tensorFromArray(gV, shape)._unwrap(),
      );
      const dX = await read(derived.x);
      const dW = await read(derived.w);
      const dB = await read(derived.b);
      const [handX, handW, handB] = await handGrads(
        () => {
          const x = api.tensorFromArray(xV, shape, { requiresGrad: true });
          const w = api.tensorFromArray(wV, [cols], { requiresGrad: true });
          const b = api.tensorFromArray(bV, [cols], { requiresGrad: true });
          return { out: api.layernorm(x, w, b, eps), params: [x, w, b] };
        },
        gV,
        shape,
      );
      assertClose(dX, handX, BOUND, BOUND, "layernorm dX");
      assertClose(dW, handW, BOUND, BOUND, "layernorm dW");
      assertClose(dB, handB, BOUND, BOUND, "layernorm dB");
    });
  }
});
