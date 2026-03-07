/**
 * Tests for reduction preamble/epilogue fusion.
 *
 * Verifies that elementwise → reduction patterns are detected and fused
 * into single kernels, producing correct numerical results.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend";
import { canUseWebGPU, cpuOnly } from "./helpers/webgpu";

describe.skipIf(cpuOnly)("Reduction preamble/epilogue fusion", () => {
  let hasGPU: boolean;
  let api: Torchlette;

  beforeAll(async () => {
    hasGPU = await canUseWebGPU();
    if (hasGPU) {
      api = new Torchlette("webgpu", { enableFusion: true });
    }
  });

  // Preamble: mul → sum (elementwise before reduction)
  it("fuses mul → sum (preamble)", async () => {
    if (!hasGPU) return;
    const a = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device: "webgpu" });
    const b = api.tensorFromArray([2, 3, 4, 5], [2, 2], { device: "webgpu" });
    // mul → sum should be fusible as a preamble
    const result = a.mul(b).sum();
    const val = await result.item();
    // (1*2)+(2*3)+(3*4)+(4*5) = 2+6+12+20 = 40
    expect(val).toBeCloseTo(40, 4);
    result.dispose();
    a.dispose();
    b.dispose();
    await api.markStep();
  });

  // Preamble: mul → sum along dim
  it("fuses mul → sum(dim=1) (preamble, dim reduction)", async () => {
    if (!hasGPU) return;
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
      device: "webgpu",
    });
    const b = api.tensorFromArray([2, 1, 3, 1, 2, 1], [2, 3], {
      device: "webgpu",
    });
    const result = a.mul(b).sum({ dim: 1 });
    const data = await result.cpu();
    // Row 0: 1*2 + 2*1 + 3*3 = 2+2+9 = 13
    // Row 1: 4*1 + 5*2 + 6*1 = 4+10+6 = 20
    expect(Array.from(data)).toEqual([13, 20]);
    result.dispose();
    a.dispose();
    b.dispose();
    await api.markStep();
  });

  // Preamble: mul → mean (mean = sum/count)
  it("fuses mul → mean (preamble)", async () => {
    if (!hasGPU) return;
    const a = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device: "webgpu" });
    const b = api.tensorFromArray([2, 2, 2, 2], [2, 2], { device: "webgpu" });
    const result = a.mul(b).mean();
    const val = await result.item();
    // mean of [2, 4, 6, 8] = 20/4 = 5
    expect(val).toBeCloseTo(5, 4);
    result.dispose();
    a.dispose();
    b.dispose();
    await api.markStep();
  });

  // Multi-op preamble: cast → mul → sum (chain of 2 fusible ops before reduction)
  it("fuses add → mul → sum (multi-op preamble chain)", async () => {
    if (!hasGPU) return;
    const a = api.tensorFromArray([1, 2, 3, 4], [4], { device: "webgpu" });
    const b = api.tensorFromArray([1, 1, 1, 1], [4], { device: "webgpu" });
    const c = api.tensorFromArray([2, 2, 2, 2], [4], { device: "webgpu" });
    // (a + b) * c → sum
    const result = a.add(b).mul(c).sum();
    const val = await result.item();
    // ([2,3,4,5] * [2,2,2,2]) = [4,6,8,10] → sum = 28
    expect(val).toBeCloseTo(28, 4);
    result.dispose();
    a.dispose();
    b.dispose();
    c.dispose();
    await api.markStep();
  });

  // Epilogue: sum → mul (elementwise after reduction)
  it("fuses sum → mul (epilogue)", async () => {
    if (!hasGPU) return;
    const a = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], {
      device: "webgpu",
    });
    const scale = api.tensorFromArray([2, 3], [2], { device: "webgpu" });
    // sum(dim=1) → mul(scale)
    const sumResult = a.sum({ dim: 1 }); // [6, 15]
    const result = sumResult.mul(scale); // [12, 45]
    const data = await result.cpu();
    expect(Array.from(data)).toEqual([12, 45]);
    result.dispose();
    sumResult.dispose();
    a.dispose();
    scale.dispose();
    await api.markStep();
  });

  // Correctness: compare fused vs unfused results
  it("fused result matches unfused computation", async () => {
    if (!hasGPU) return;
    const data = Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.1));
    const a = api.tensorFromArray(data, [8, 8], { device: "webgpu" });
    const b = api.tensorFromArray(
      data.map((x) => x * 2),
      [8, 8],
      { device: "webgpu" },
    );

    // Compute reference without fusion
    const unfusedApi = new Torchlette("webgpu", { enableFusion: false });
    const aRef = unfusedApi.tensorFromArray(data, [8, 8], {
      device: "webgpu",
    });
    const bRef = unfusedApi.tensorFromArray(
      data.map((x) => x * 2),
      [8, 8],
      { device: "webgpu" },
    );

    // mul → sum(dim=0)
    const fused = a.mul(b).sum({ dim: 0 });
    const unfused = aRef.mul(bRef).sum({ dim: 0 });

    const fusedData = await fused.cpu();
    const unfusedData = await unfused.cpu();

    for (let i = 0; i < fusedData.length; i++) {
      expect(fusedData[i]).toBeCloseTo(unfusedData[i], 4);
    }

    fused.dispose();
    unfused.dispose();
    a.dispose();
    b.dispose();
    aRef.dispose();
    bRef.dispose();
    await api.markStep();
    await unfusedApi.markStep();
  });
});
