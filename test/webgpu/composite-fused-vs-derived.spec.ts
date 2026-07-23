/**
 * COMPOSITE-CLOSURE F1 / C3 — the fused-vs-derived standing differential (§7).
 *
 * The fused GPU backward kernels (rmsnorm / layernorm / cross_entropy) STAY the
 * fast path — they carry their reductions in-kernel, which the elementwise F2
 * fold has no target for (§7). They are NOT re-derived; they are ASSERTED == the
 * derived `vjpComposition` (F1) realized over the SAME webgpu runtime, within the
 * measured §3.2 ULP tolerance. This is the backward twin of P5's fused-adamStep-
 * asserted-against-the-program: the fused kernel is checked against the
 * composition, not re-owned (closes RT3 for the composite backward).
 *
 * The fused kernels are byte-untouched here — the 124M hot path is unchanged.
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../../src";
import { initWebGPU } from "../../src/backend/webgpu";
import type { Tensor } from "../../src/frontend/tensor";
import {
  CROSS_ENTROPY_DEF,
  LAYERNORM_DEF,
  RMSNORM_DEF,
  vjpComposition,
} from "../../src/ops/semantic";
import type { Tensor as RuntimeTensor } from "../../src/runtime/tensor";
import { cpuOnly } from "../helpers/webgpu";

/** Read a param's grad to CPU, asserting it exists. */
async function gradOf(t: Tensor): Promise<number[]> {
  if (!t.grad) throw new Error("missing grad");
  return t.grad.cpu();
}

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

/** max-abs and max-rel of two flat arrays. */
function delta(a: number[], b: number[]): { abs: number; rel: number } {
  let abs = 0;
  let rel = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    abs = Math.max(abs, d);
    rel = Math.max(rel, d / (Math.abs(b[i]) + 1e-6));
  }
  return { abs, rel };
}

// Measured §3.2 ULP bound: fused kernel vs derived graph, both GPU f32, differ
// only in reduction order + the fused folded epilogue. Well under the run-to-run
// GPU floor for these shapes (measured deltas printed by the test).
const ABS = 1e-5;
const REL = 1e-4;

describe.skipIf(cpuOnly)(
  "C3 — fused composite backward == derived VJP",
  { timeout: 30000 },
  () => {
    it("rmsnorm gradX/gradWeight", async () => {
      await initWebGPU();
      const gpu = new Torchlette("webgpu");
      const rt = gpu.runtime;
      const shape = [8, 16];
      const dim = 1;
      const eps = 1e-6;
      const xV = rand(prod(shape), 201);
      const wV = rand(shape[1], 203);
      const gV = rand(prod(shape), 205);

      // Fused (frontend webgpu backward).
      const x = gpu.tensorFromArray(xV, shape, { requiresGrad: true });
      const w = gpu.tensorFromArray(wV, [shape[1]], { requiresGrad: true });
      const out = gpu.rmsnorm(x, w, eps);
      await out.backward(gpu.tensorFromArray(gV, shape));
      const fX = await gradOf(x);
      const fW = await gradOf(w);

      // Derived (vjpComposition over the same rt).
      const grads = vjpComposition(
        RMSNORM_DEF,
        rt,
        dim,
        {
          x: gpu.tensorFromArray(xV, shape)._unwrap(),
          w: gpu.tensorFromArray(wV, [shape[1]])._unwrap(),
          eps,
        },
        gpu.tensorFromArray(gV, shape)._unwrap(),
      );
      const dX = await rt.cpu(grads.x);
      const dW = await rt.cpu(grads.w);

      const eX = delta(fX, dX);
      const eW = delta(fW, dW);
      console.log(
        `[C3 rmsnorm] dX abs=${eX.abs.toExponential(2)} rel=${eX.rel.toExponential(2)} | dW abs=${eW.abs.toExponential(2)} rel=${eW.rel.toExponential(2)}`,
      );
      expect(eX.abs).toBeLessThan(ABS);
      expect(eX.rel).toBeLessThan(REL);
      expect(eW.abs).toBeLessThan(ABS);
      expect(eW.rel).toBeLessThan(REL);
      await gpu.markStep();
    });

    it("layernorm gradX/gradWeight/gradBias", async () => {
      await initWebGPU();
      const gpu = new Torchlette("webgpu");
      const rt = gpu.runtime;
      const shape = [8, 16];
      const dim = 1;
      const eps = 1e-5;
      const xV = rand(prod(shape), 211);
      const wV = rand(shape[1], 213);
      const bV = rand(shape[1], 215);
      const gV = rand(prod(shape), 217);

      const x = gpu.tensorFromArray(xV, shape, { requiresGrad: true });
      const w = gpu.tensorFromArray(wV, [shape[1]], { requiresGrad: true });
      const b = gpu.tensorFromArray(bV, [shape[1]], { requiresGrad: true });
      const out = gpu.layernorm(x, w, b, eps);
      await out.backward(gpu.tensorFromArray(gV, shape));
      const fX = await gradOf(x);
      const fW = await gradOf(w);
      const fB = await gradOf(b);

      const grads = vjpComposition(
        LAYERNORM_DEF,
        rt,
        dim,
        {
          x: gpu.tensorFromArray(xV, shape)._unwrap(),
          w: gpu.tensorFromArray(wV, [shape[1]])._unwrap(),
          b: gpu.tensorFromArray(bV, [shape[1]])._unwrap(),
          eps,
        },
        gpu.tensorFromArray(gV, shape)._unwrap(),
      );
      const dX = await rt.cpu(grads.x);
      const dW = await rt.cpu(grads.w);
      const dB = await rt.cpu(grads.b);

      const eX = delta(fX, dX);
      const eW = delta(fW, dW);
      const eB = delta(fB, dB);
      console.log(
        `[C3 layernorm] dX abs=${eX.abs.toExponential(2)} | dW abs=${eW.abs.toExponential(2)} | dB abs=${eB.abs.toExponential(2)}`,
      );
      for (const e of [eX, eW, eB]) {
        expect(e.abs).toBeLessThan(ABS);
        expect(e.rel).toBeLessThan(REL);
      }
      await gpu.markStep();
    });

    it("cross_entropy gradLogits", async () => {
      await initWebGPU();
      const gpu = new Torchlette("webgpu");
      const rt = gpu.runtime;
      const B = 8;
      const V = 16;
      const dim = 1;
      const xV = rand(B * V, 221);
      const gV = rand(B, 223);
      const tV = rand(B, 225, 0, V).map((v) => Math.min(V - 1, Math.floor(v)));

      const logits = gpu.tensorFromArray(xV, [B, V], { requiresGrad: true });
      const targets = gpu.toDtype(gpu.tensorFromArray(tV, [B]), "i32");
      const loss = gpu._crossEntropyFused(logits, targets);
      await loss.backward(gpu.tensorFromArray(gV, [B]));
      const fX = await gradOf(logits);

      const grads = vjpComposition(
        CROSS_ENTROPY_DEF,
        rt,
        dim,
        {
          x: gpu.tensorFromArray(xV, [B, V])._unwrap(),
          target: gpu.toDtype(gpu.tensorFromArray(tV, [B, 1]), "i32")._unwrap(),
        },
        gpu.tensorFromArray(gV, [B, 1])._unwrap(),
      );
      const dX = await rt.cpu(grads.x as RuntimeTensor);

      const eX = delta(fX, dX);
      console.log(
        `[C3 cross_entropy] dLogits abs=${eX.abs.toExponential(2)} rel=${eX.rel.toExponential(2)}`,
      );
      expect(eX.abs).toBeLessThan(ABS);
      expect(eX.rel).toBeLessThan(REL);
      await gpu.markStep();
    });
  },
);
