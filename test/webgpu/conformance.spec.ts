/**
 * OP CONFORMANCE — WebGPU project.
 *
 * Runs the generated op matrix on the WebGPU backend, comparing every op ×
 * dtype × {contiguous, odd-size, 5-view battery, broadcast, dim sweep} against
 * the CPU backend — the full cross-implementation differential. This is the
 * structural defense against the pow(x<0) class, dtype-oblivious kernels (#59),
 * and offset-view raw-bind reads (#58).
 *
 * Auto-skips GPU-less (CI); matched by the `test/webgpu/*.spec.ts` glob in
 * vitest.config.ts's GPU_TEST_FILES.
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  getGpuUncapturedErrorCount,
  initWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { registerConformance } from "../helpers/op-catalog";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("op conformance (WebGPU, vs CPU differential)", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  registerConformance({
    device: "webgpu",
    makeApi: () => new Torchlette("webgpu"),
    makeOracle: () => new Torchlette("cpu"),
    gpuErrorCount: getGpuUncapturedErrorCount,
  });

  // -------------------------------------------------------------------------
  // #59 dtype-parametrization fixes (formerly catalogued it.fails; the fixed
  // markers flipped red → removed per the visible-debt protocol).
  // -------------------------------------------------------------------------

  // #59 FINDING #A: the fused isfinite codegen bitcast<u32> assumed f32; an f16
  // input now upcasts to f32 first (exact, preserves inf/nan), and the unary
  // out-binding is f32 (always_f32) rather than the input's f16. Guards the
  // dropped-submit hazard via the GPU error counter.
  it(
    "isfinite on f16 input is correct (no dropped submit) (#59 FINDING #A)",
    async () => {
      const gpu = new Torchlette("webgpu");
      const before = getGpuUncapturedErrorCount();
      const x = gpu
        .tensorFromArray([1, 2, -3, 100, 0.5, -0.5], [6], { device: "webgpu" })
        .toDtype("f16");
      const got = Array.from(await gpu.isfinite(x).cpu());
      // Correct is all-ones (every input finite) AND no GPU error was raised.
      expect(getGpuUncapturedErrorCount() - before).toBe(0);
      for (const v of got) expect(Math.abs(v - 1)).toBeLessThan(0.05);
    },
    30_000,
  );

  // #59 FINDING #B: max/min reductions now upcast f16→f32 before reducing (the
  // f32_required rule, exactly like sum/mean), so the kernel's f32 input binding
  // reads the right lanes. GPU must match CPU (both take the same f16 rounding).
  it(
    "max reduction on f16 input matches CPU (#59 FINDING #B)",
    async () => {
      const gpu = new Torchlette("webgpu");
      const cpu = new Torchlette("cpu");
      const data = [1.7, -0.3, 2.4, -1.9, 0.8, -2.1, 1.1, 0.05];
      // dim reduction returns a Tensor; readback via .cpu().
      const g = gpu.max(gpu.tensorFromArray(data, [2, 4]).toDtype("f16"), { dim: -1 });
      const c = cpu.max(cpu.tensorFromArray(data, [2, 4]).toDtype("f16"), { dim: -1 });
      const gArr = Array.from(await (g as { cpu(): Promise<Float32Array> }).cpu());
      const cArr = Array.from(await (c as { cpu(): Promise<Float32Array> }).cpu());
      for (let i = 0; i < cArr.length; i++)
        expect(Math.abs(gArr[i] - cArr[i])).toBeLessThan(0.05);
    },
    30_000,
  );

  // #59 PRIMARY: gather is now dtype-parametrized (f16 source → f16 output). An
  // f16 embedding table both halves residency and stops chunking sooner.
  it(
    "gather on f16 source is dtype-correct (#59)",
    async () => {
      const gpu = new Torchlette("webgpu");
      const src = gpu
        .tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { device: "webgpu" })
        .toDtype("f16");
      const idx = gpu.tensorFromArray([0, 2], [2, 1], { device: "webgpu" });
      const got = Array.from(await gpu.gather(src, idx, { dim: 1 }).cpu());
      // Correct result is [1, 6].
      expect(Math.abs(got[0] - 1)).toBeLessThan(0.05);
      expect(Math.abs(got[1] - 6)).toBeLessThan(0.05);
    },
    30_000,
  );
});
