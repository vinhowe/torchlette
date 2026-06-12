/**
 * Dropout semantics + RNG freshness under the full optimization stack.
 *
 * The freshness test is the load-bearing one: dropout masks must DIFFER
 * across steps even when the plan template/compiled replay is reused. A
 * frozen mask (the frozen-payload class applied to RNG seeds) would be
 * per-step-stochastic-looking and invisible to every trajectory gate —
 * training would silently use one fixed mask forever.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { Dropout } from "../src/nn";
import { canUseWebGPU, cpuOnly } from "./helpers/webgpu";

describe("nn.Dropout", { timeout: 180_000 }, () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = !cpuOnly && (await canUseWebGPU());
  });

  it("eval mode is identity; train mode zeroes ~p and scales by 1/(1-p)", async () => {
    if (!webgpu) return;
    const api = new Torchlette("webgpu", { enableFusion: true });
    const N = 4096;
    const p = 0.3;
    const drop = new Dropout(api, { p });
    const data = Array.from({ length: N }, (_, i) => 1 + (i % 7) * 0.1);
    const x = api.tensorFromArray(data, [N], { device: "webgpu" });

    drop.eval();
    const evalOut = Array.from(await drop.forward(x).cpu());
    // f32 storage: compare within float32 epsilon of the JS double inputs.
    for (let i = 0; i < N; i++)
      expect(Math.abs(evalOut[i] - data[i])).toBeLessThan(1e-6);

    drop.train();
    const out = Array.from(await drop.forward(x).cpu());
    let zeros = 0;
    const scale = 1 / (1 - p);
    for (let i = 0; i < N; i++) {
      if (out[i] === 0) {
        zeros++;
      } else {
        // Surviving elements are scaled exactly
        expect(Math.abs(out[i] - data[i] * scale)).toBeLessThan(1e-4);
      }
    }
    // Zero fraction ≈ p (4096 samples: ±5 sigma ≈ ±0.036)
    expect(zeros / N).toBeGreaterThan(p - 0.05);
    expect(zeros / N).toBeLessThan(p + 0.05);
  });

  it("masks are FRESH across steps under default optimizations (compiled replay)", async () => {
    if (!webgpu) return;
    const api = new Torchlette("webgpu", { enableFusion: true });
    const N = 2048;
    const drop = new Dropout(api, { p: 0.5 });
    drop.train();
    const x = api.tensorFromArray(new Array(N).fill(1), [N], {
      device: "webgpu",
    });
    // Identical planeach step → template + compiled replay engage from step 2+.
    const patterns: string[] = [];
    for (let step = 0; step < 6; step++) {
      await api.beginStep();
      const out = drop.forward(x);
      const arr = Array.from(await out.cpu());
      patterns.push(arr.map((v) => (v === 0 ? "0" : "1")).join(""));
      api.endStep();
      await api.markStep();
    }
    // Every step's mask must be distinct (2^-2048 collision odds).
    expect(new Set(patterns).size).toBe(patterns.length);
  });

  it("backward scales gradients by the SAME mask as forward", async () => {
    if (!webgpu) return;
    const api = new Torchlette("webgpu", { enableFusion: true });
    const N = 512;
    const p = 0.5;
    const drop = new Dropout(api, { p });
    drop.train();
    const x = api.tensorFromArray(
      Array.from({ length: N }, (_, i) => 0.5 + (i % 5) * 0.25),
      [N],
      { device: "webgpu", requiresGrad: true },
    );
    const out = drop.forward(x);
    const outVals = Array.from(await out.cpu());
    const loss = api.sum(out);
    await loss.backward();
    const g = (x as unknown as { grad: { cpu(): Promise<Float32Array> } }).grad;
    const grads = Array.from(await g.cpu());
    const scale = 1 / (1 - p);
    for (let i = 0; i < N; i++) {
      // dLoss/dx_i = scale where the mask kept the element, 0 where dropped.
      const expected = outVals[i] === 0 ? 0 : scale;
      expect(Math.abs(grads[i] - expected)).toBeLessThan(1e-4);
    }
  });
});
