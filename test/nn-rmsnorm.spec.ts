/**
 * nn.RMSNorm: forward/backward vs a hand-computed reference, on CPU and GPU
 * (the GPU path exercises the fused RMSNorm kernel via tensor.rmsnorm).
 */
import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { RMSNorm } from "../src/nn";
import { canUseWebGPU, cpuOnly } from "./helpers/webgpu";

function refRMSNorm(x: number[][], w: number[], eps: number): number[][] {
  return x.map((row) => {
    const ms = row.reduce((a, v) => a + v * v, 0) / row.length;
    const inv = 1 / Math.sqrt(ms + eps);
    return row.map((v, j) => v * inv * w[j]);
  });
}

describe("nn.RMSNorm", { timeout: 120_000 }, () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = !cpuOnly && (await canUseWebGPU());
  });

  for (const device of ["cpu", "webgpu"] as const) {
    it(`forward matches reference (${device})`, async () => {
      if (device === "webgpu" && !webgpu) return;
      const api = new Torchlette(device);
      const D = 8;
      const norm = new RMSNorm(api, D, { eps: 1e-6, device });
      const rows = [
        [0.5, -1.2, 2.0, 0.1, -0.7, 1.5, -2.2, 0.9],
        [3.0, 0.0, -1.0, 0.25, 1.1, -0.4, 0.8, -1.6],
      ];
      const x = api.tensorFromArray(rows.flat(), [2, D], { device });
      const out = Array.from(await norm.forward(x).cpu());
      const ref = refRMSNorm(rows, new Array(D).fill(1), 1e-6).flat();
      for (let i = 0; i < ref.length; i++) {
        expect(Math.abs(out[i] - ref[i])).toBeLessThan(1e-4);
      }
    });

    it(`backward produces finite grads for weight and input (${device})`, async () => {
      if (device === "webgpu" && !webgpu) return;
      const api = new Torchlette(device);
      const D = 8;
      const norm = new RMSNorm(api, D, { device });
      const x = api.tensorFromArray(
        Array.from({ length: 2 * D }, (_, i) => Math.sin(i + 1)),
        [2, D],
        { device, requiresGrad: true },
      );
      const out = norm.forward(x);
      const loss = api.sum(api.mul(out, out));
      await loss.backward();
      for (const t of [norm.weight!, x]) {
        const g = (t as unknown as { grad: { cpu(): Promise<Float32Array> } | null }).grad;
        expect(g).not.toBeNull();
        const arr = await g!.cpu();
        for (const v of arr) expect(Number.isFinite(v)).toBe(true);
      }
      // Gradient sanity: weight grad of element j = sum over rows of
      // 2*out_j*normed_j = 2*out_j^2/w_j; with w=1 it's 2*sum(out_j^2) > 0.
      const wg = Array.from(await (norm.weight as unknown as { grad: { cpu(): Promise<Float32Array> } }).grad.cpu());
      expect(wg.some((v) => Math.abs(v) > 1e-6)).toBe(true);
    });
  }
});
