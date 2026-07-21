import { describe, expect, it } from "vitest";
import { Torchlette } from "../../src";
import { initWebGPU } from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

/**
 * Tests for dispatch replay correctness.
 *
 * Dispatch replay records GPU dispatch sequences on the first execution
 * and replays them on subsequent steps to bypass JS dispatch overhead.
 * These tests verify that replayed results are bit-identical to first execution.
 *
 * Dispatch replay is always enabled (the former TORCHLETTE_DISPATCH_REPLAY
 * opt-out was removed) and is tested implicitly here by running the same
 * computation across multiple markStep() calls.
 */
describe.skipIf(cpuOnly)("dispatch replay", { timeout: 60000 }, () => {
  it("elementwise ops produce identical results across steps", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    const shape = [2, 4];

    const results: number[][] = [];
    for (let step = 0; step < 4; step++) {
      const x = gpu.tensorFromArray(data, shape);
      const y = gpu.tensorFromArray(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        shape,
      );
      const z = x.add(y).relu().mul(x);
      results.push(await z.cpu());
      await gpu.markStep();
    }

    // All steps should produce bit-identical results
    for (let step = 1; step < results.length; step++) {
      for (let i = 0; i < results[0].length; i++) {
        expect(results[step][i]).toBe(results[0][i]);
      }
    }
  });

  it("reduction ops produce identical results across steps", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const data = Array.from({ length: 24 }, (_, i) => (i + 1) * 0.1);
    const shape = [4, 6];

    const results: number[][] = [];
    for (let step = 0; step < 4; step++) {
      const x = gpu.tensorFromArray(data, shape);
      const s = x.sum({ dim: [1] });
      results.push(await s.cpu());
      await gpu.markStep();
    }

    for (let step = 1; step < results.length; step++) {
      for (let i = 0; i < results[0].length; i++) {
        expect(results[step][i]).toBe(results[0][i]);
      }
    }
  });

  it("matmul produces identical results across steps", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const M = 4,
      K = 8,
      N = 6;
    const aData = Array.from({ length: M * K }, (_, i) => (i + 1) * 0.01);
    const bData = Array.from({ length: K * N }, (_, i) => (i + 1) * 0.01);

    const results: number[][] = [];
    for (let step = 0; step < 4; step++) {
      const a = gpu.tensorFromArray(aData, [M, K]);
      const b = gpu.tensorFromArray(bData, [K, N]);
      const c = a.matmul(b);
      results.push(await c.cpu());
      await gpu.markStep();
    }

    for (let step = 1; step < results.length; step++) {
      for (let i = 0; i < results[0].length; i++) {
        expect(results[step][i]).toBe(results[0][i]);
      }
    }
  });

  it("mixed ops (matmul + elementwise + reduction) are stable", async () => {
    await initWebGPU();

    const gpu = new Torchlette("webgpu");
    const M = 4,
      K = 8,
      N = 6;
    const aData = Array.from({ length: M * K }, (_, i) => (i + 1) * 0.01);
    const bData = Array.from({ length: K * N }, (_, i) => (i + 1) * 0.01);
    const biasData = Array.from({ length: N }, (_, i) => (i + 1) * 0.1);

    const results: number[][] = [];
    for (let step = 0; step < 4; step++) {
      const a = gpu.tensorFromArray(aData, [M, K]);
      const b = gpu.tensorFromArray(bData, [K, N]);
      const bias = gpu.tensorFromArray(biasData, [N]);
      const mm = a.matmul(b);
      const biased = mm.add(bias.expand([M, N]));
      const activated = biased.relu();
      const reduced = activated.sum({ dim: [1] });
      results.push(await reduced.cpu());
      await gpu.markStep();
    }

    for (let step = 1; step < results.length; step++) {
      for (let i = 0; i < results[0].length; i++) {
        expect(results[step][i]).toBe(results[0][i]);
      }
    }
  });
});
